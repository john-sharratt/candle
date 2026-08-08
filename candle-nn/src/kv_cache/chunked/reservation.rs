//! The VA reservation: address space first, physical memory second.
//!
//! Everything above this file assumes one contiguous span whose address never
//! changes. `cuMemAlloc` cannot give that — it hands back whatever the
//! allocator has, and a later growth is a different pointer. The virtual memory
//! management API separates the two halves of an allocation, which is what the
//! design needs (`docs/archived/arena_unification.md` §3.2):
//!
//! - **Reserve** a virtual address range. Costs no memory; fixes the addresses
//!   every arena, region and transient span will ever use.
//! - **Create** physical granules and **map** them into that range. This is the
//!   part that can fail, and failing is informative: the point at which the
//!   driver refuses is `C`, the capacity actually available to us.
//!
//! Measuring `C` and claiming it are therefore the same act. A separate "how
//! much is free?" query would be a lie by the time it returned — another
//! process, or our own weight loading, moves it.
//!
//! # The refusal is a failed *touch*, not a failed create
//!
//! `cuMemCreate` does not tell the truth about capacity. Probed on the target
//! machine (`vmm_overcommit_probe.py`, §9): it succeeded for every granule of a
//! 32 GiB span on a 16 GiB card. The limit appears at the first **write** to a
//! granule. So every granule this file maps is also written before it counts as
//! claimed, and a failed write is the refusal that stops the balloon — the
//! granule is unmapped and released, and the reservation ends there.
//!
//! The write is a zero-fill, which the region tier then relies on: a
//! freshly-mapped region is already zeroed, exactly as `Tensor::zeros` left the
//! slabs it replaces.
//!
//! # There is no fallback, on purpose
//!
//! A device without the VMM API cannot run this allocator at all, and
//! [`Reservation::reserve`] says so by name before it does anything else. The
//! alternative — one giant `cuMemAlloc` behind the same interface — was
//! considered and rejected: it is not a smaller version of this path but a
//! different one. Its eviction unit is the whole buffer rather than a granule,
//! so desktop pressure sheds the entire reservation at once; it cannot release
//! part of itself back for weight loading; and capacity has to be found by
//! binary search over whole allocations instead of falling out of the balloon.
//! That is a second allocation path no test on the target hardware can
//! exercise, which would be wrong by the time anything needed it.
//!
//! So the contract is: this either works or it stops, immediately, with a
//! message naming the missing capability.
//!
//! # Granularity
//!
//! The driver maps in granules, not bytes, and the granule size is a device
//! property. Every size here is rounded up to it; a reservation is always a
//! whole number of granules, which is also why region carving can assume
//! alignment without re-checking.

use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::result::device::get_attribute;
use candle::cuda_backend::cudarc::driver::result::memset_d8_sync;
use candle::cuda_backend::cudarc::driver::sys::{
    CUdevice, CUdevice_attribute_enum, CUdeviceptr, CUmemAccessDesc, CUmemAccess_flags_enum,
    CUmemAllocationGranularity_flags_enum, CUmemAllocationHandleType_enum, CUmemAllocationProp_st,
    CUmemAllocationType_enum, CUmemLocationType_enum, CUmemLocation_st, CUresult,
};
use candle::cuda_backend::cudarc::driver::{CudaContext, CudaStream};
use candle::Result;

/// A contiguous virtual address span, backed by however much physical memory
/// the driver would give us.
///
/// Dropping it unmaps and releases everything. Nothing hands out sub-ranges
/// here — carving is the region layer's job; this owns only the span and the
/// granules behind it.
pub(crate) struct Reservation {
    base: CUdeviceptr,
    reserved: usize,
    granularity: usize,
    /// One entry per granule slot of the span; `Some(handle)` when mapped.
    ///
    /// Indexed rather than appended so a caller can map a range anywhere in the
    /// span — the transient tier sits at the right end and must be backed even
    /// when the KV side to its left was cut short by a refusal.
    granules: Vec<Option<u64>>,
    mapped: usize,
    device: CUdevice,
    context: Arc<CudaContext>,
}

fn check(res: CUresult, what: &str) -> Result<()> {
    if res == CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    candle::bail!("{what} failed: {res:?}")
}

/// Stop here, by name, if this device has no virtual memory management API.
///
/// Without it there is no reservation, and without a reservation there is no
/// region tier, no fixed base pointer and no transient side — so every layer
/// above would fail anyway, one indirection at a time, with whatever error the
/// driver happened to return. This turns that into a single sentence naming the
/// capability that is missing and where to read about it.
fn require_vmm(device: CUdevice) -> Result<()> {
    // SAFETY: `device` was obtained from a live CUDA context.
    let supported = unsafe {
        get_attribute(
            device,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        )
    }
    .map_err(|e| candle::Error::Msg(format!("querying VMM support: {e}")))?;
    if supported != 0 {
        return Ok(());
    }
    candle::bail!(
        "this CUDA device reports no virtual memory management support \
         (CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 0), and the KV \
         reservation is built on it. There is deliberately no second allocation \
         path — see docs/archived/arena_unification.md §3.2. VMM is commonly absent under \
         WSL2, on vGPU/MIG-partitioned devices, and on older drivers; run on a \
         device with native VMM support, or build the single-allocation variant \
         behind `Reservation` for this target."
    )
}

/// The allocation property block describing "pinned device memory on `device`".
fn alloc_prop(device: CUdevice) -> CUmemAllocationProp_st {
    let mut prop: CUmemAllocationProp_st = unsafe { std::mem::zeroed() };
    prop.type_ = CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE;
    prop.location = CUmemLocation_st {
        type_: CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
        id: device,
    };
    prop
}

impl Reservation {
    /// Reserve `bytes` of address space on `stream`'s device, mapping nothing.
    ///
    /// Rounded up to the device's granule size, so the span is always a whole
    /// number of granules.
    ///
    /// Fails immediately, naming the capability, on a device without VMM — there
    /// is no fallback path (see the module header).
    pub(crate) fn reserve(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self> {
        let device = stream.context().cu_device();
        require_vmm(device)?;
        let prop = alloc_prop(device);
        let mut granularity: usize = 0;
        // SAFETY: `prop` is fully initialised above and `granularity` is a
        // valid out-pointer; the call only reads the former and writes the
        // latter.
        check(
            unsafe {
                candle::cuda_backend::cudarc::driver::sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "cuMemGetAllocationGranularity",
        )?;
        if granularity == 0 {
            candle::bail!("cuMemGetAllocationGranularity reported a zero granule size")
        }
        let reserved = bytes.div_ceil(granularity) * granularity;

        let mut base: CUdeviceptr = 0;
        // SAFETY: a fresh reservation — `addr` 0 lets the driver choose the
        // range, and `flags` 0 is the only defined value.
        check(
            unsafe {
                candle::cuda_backend::cudarc::driver::sys::cuMemAddressReserve(
                    &mut base, reserved, 0, 0, 0,
                )
            },
            "cuMemAddressReserve",
        )?;

        Ok(Self {
            base,
            reserved,
            granularity,
            granules: vec![None; reserved / granularity],
            mapped: 0,
            device,
            context: stream.context().clone(),
        })
    }

    /// Device address of the span's first byte.
    pub(crate) fn base(&self) -> u64 {
        self.base
    }

    /// Bytes of address space reserved (a whole number of granules).
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reserved
    }

    /// The device's granule size.
    pub(crate) fn granularity(&self) -> usize {
        self.granularity
    }

    /// Bytes of physical memory currently mapped into the span.
    pub(crate) fn mapped_bytes(&self) -> usize {
        self.mapped * self.granularity
    }

    /// Back granule `idx` with physical memory and prove it is usable.
    ///
    /// Returns `Ok(false)` when the driver refuses — either at create time or,
    /// far more often, at the zero-fill that follows (see the module header).
    /// That refusal is the measurement of `C`, not an error, so ballooning can
    /// stop cleanly; anything else *is* an error.
    pub(crate) fn map_granule(&mut self, idx: usize) -> Result<bool> {
        if idx >= self.granules.len() {
            return Ok(false);
        }
        if self.granules[idx].is_some() {
            return Ok(true);
        }
        let len = self.granularity;
        let addr = self.base + (idx * len) as u64;
        let prop = alloc_prop(self.device);
        let mut handle: u64 = 0;
        // SAFETY: `prop` describes pinned device memory on this device; the
        // handle is a valid out-pointer.
        let res = unsafe {
            candle::cuda_backend::cudarc::driver::sys::cuMemCreate(&mut handle, len, &prop, 0)
        };
        if res == CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            return Ok(false);
        }
        check(res, "cuMemCreate")?;

        // SAFETY: `addr` names granule `idx` of this reservation, which is
        // unmapped (checked above), and `handle` is a live allocation of
        // exactly `len` bytes.
        let mapped =
            unsafe { candle::cuda_backend::cudarc::driver::sys::cuMemMap(addr, len, 0, handle, 0) };
        if mapped != CUresult::CUDA_SUCCESS {
            // Release the granule rather than leak it: it is mapped nowhere.
            // SAFETY: `handle` is live and unmapped.
            unsafe { candle::cuda_backend::cudarc::driver::sys::cuMemRelease(handle) };
            if mapped == CUresult::CUDA_ERROR_OUT_OF_MEMORY {
                return Ok(false);
            }
            check(mapped, "cuMemMap")?;
        }

        // Mapping alone does not make the range usable; access has to be
        // granted explicitly, and forgetting it fails later as an illegal
        // address rather than here.
        let desc = CUmemAccessDesc {
            location: CUmemLocation_st {
                type_: CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device,
            },
            flags: CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: the range was just mapped, and `desc` names this device.
        let access = unsafe {
            candle::cuda_backend::cudarc::driver::sys::cuMemSetAccess(addr, len, &desc, 1)
        };
        if access != CUresult::CUDA_SUCCESS {
            self.discard(addr, handle);
            check(access, "cuMemSetAccess")?;
        }

        // The touch. This is the real capacity test, and it doubles as the
        // zero-fill the region tier relies on. Synchronous on purpose: it
        // returns the driver's verdict here instead of surfacing it later as a
        // sticky illegal-address on some unrelated kernel.
        self.context
            .bind_to_thread()
            .map_err(|e| candle::Error::Msg(format!("bind_to_thread: {e}")))?;
        // SAFETY: the range is mapped and read/write accessible from this
        // device, and `len` is exactly its length.
        if let Err(e) = unsafe { memset_d8_sync(addr, 0, len) } {
            self.discard(addr, handle);
            log::debug!(
                "reservation: granule {idx} refused at the touch ({e:?}) — {} B claimed",
                self.mapped_bytes()
            );
            return Ok(false);
        }

        self.granules[idx] = Some(handle);
        self.mapped += 1;
        Ok(true)
    }

    /// Unmap and release a granule that failed part-way through mapping.
    fn discard(&self, addr: CUdeviceptr, handle: u64) {
        // SAFETY: `addr` is mapped to `handle`, which is live; both are being
        // abandoned together so neither is used again.
        unsafe {
            candle::cuda_backend::cudarc::driver::sys::cuMemUnmap(addr, self.granularity);
            candle::cuda_backend::cudarc::driver::sys::cuMemRelease(handle);
        }
    }

    /// Back every granule of `[offset, offset + bytes)`, stopping at a refusal.
    ///
    /// Returns the number of bytes of the range that are backed on return —
    /// equal to the rounded-up request when the whole range was claimed, and
    /// less when the driver refused part-way. Callers size themselves from the
    /// return value rather than assuming they got what they asked for.
    pub(crate) fn map_range(&mut self, offset: usize, bytes: usize) -> Result<usize> {
        let first = offset / self.granularity;
        let last = (offset + bytes).div_ceil(self.granularity);
        let mut claimed = 0;
        for idx in first..last.min(self.granules.len()) {
            if !self.map_granule(idx)? {
                break;
            }
            claimed += self.granularity;
        }
        Ok(claimed)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        for (idx, handle) in self.granules.drain(..).enumerate() {
            let Some(handle) = handle else { continue };
            // SAFETY: each granule was mapped at `base + idx × granularity` by
            // `map_granule` and has not been unmapped since.
            unsafe {
                candle::cuda_backend::cudarc::driver::sys::cuMemUnmap(
                    self.base + (idx * self.granularity) as u64,
                    self.granularity,
                );
                candle::cuda_backend::cudarc::driver::sys::cuMemRelease(handle);
            }
        }
        // SAFETY: the span is fully unmapped above, and `reserved` is the size
        // it was reserved with.
        unsafe {
            candle::cuda_backend::cudarc::driver::sys::cuMemAddressFree(self.base, self.reserved);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::Reservation;
    use candle::{Device, Result};

    fn stream() -> Option<std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>> {
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some(d.cuda_stream()),
            _ => None,
        }
    }

    /// The allocator has no second path, so VMM support is a hard requirement
    /// of the machine rather than a preference. Asserting it here means an
    /// unsupported target says so in the test suite instead of at the first KV
    /// cache of a benchmark run.
    #[test]
    fn the_device_supports_vmm() -> Result<()> {
        let Some(s) = stream() else { return Ok(()) };
        super::require_vmm(s.context().cu_device())
    }

    /// Reserving address space costs no physical memory, so a span far larger
    /// than the card must still succeed. This is the property the whole design
    /// rests on: addresses are fixed up front, memory is claimed later.
    #[test]
    fn reserving_more_than_the_card_costs_nothing() -> Result<()> {
        let Some(s) = stream() else { return Ok(()) };
        let r = Reservation::reserve(&s, 64 * 1024 * 1024 * 1024)?;
        assert!(r.base() != 0);
        assert_eq!(r.mapped_bytes(), 0, "reserving must map nothing");
        assert!(r.reserved_bytes() >= 64 * 1024 * 1024 * 1024);
        Ok(())
    }

    /// Sizes round up to the granule, so a span is always a whole number of
    /// granules — which is what lets region carving assume alignment.
    #[test]
    fn reservations_are_whole_granules() -> Result<()> {
        let Some(s) = stream() else { return Ok(()) };
        let r = Reservation::reserve(&s, 1)?;
        let g = r.granularity();
        assert!(g > 0 && g.is_power_of_two(), "odd granule size {g}");
        assert_eq!(r.reserved_bytes(), g, "1 byte should round to one granule");
        Ok(())
    }

    /// Mapping makes memory real and readable at the reserved address, and
    /// unmapping on drop returns it.
    #[test]
    fn mapped_memory_is_usable_at_the_reserved_address() -> Result<()> {
        let Some(s) = stream() else { return Ok(()) };
        let mut r = Reservation::reserve(&s, 8 * 1024 * 1024)?;
        assert!(
            r.map_range(0, 1024 * 1024)? >= 1024 * 1024,
            "mapping 1 MiB should succeed"
        );
        assert!(r.mapped_bytes() >= 1024 * 1024);

        // Write a pattern through the reserved address and read it back: this
        // is what proves `cuMemSetAccess` was applied, since a mapped-but-
        // inaccessible range faults instead.
        let bytes = vec![0xA5u8; 4096];
        // SAFETY: the first granule is mapped and readable/writable, and 4096
        // is well inside it.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                r.base(),
                &bytes,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("htod into reservation: {e}")))?;
        let mut back = vec![0u8; 4096];
        // SAFETY: as above, reading the range just written.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                &mut back,
                r.base(),
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("dtoh from reservation: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        assert_eq!(back, bytes, "reservation did not hold what was written");
        Ok(())
    }

    /// Mapping stops at the end of the span rather than running past it.
    #[test]
    fn mapping_stops_at_the_span_end() -> Result<()> {
        let Some(s) = stream() else { return Ok(()) };
        let mut r = Reservation::reserve(&s, 1)?;
        assert!(r.map_granule(0)?, "the single granule should map");
        assert!(!r.map_granule(1)?, "a second granule must not fit");
        assert_eq!(r.mapped_bytes(), r.reserved_bytes());
        Ok(())
    }

    /// Every granule is written as it is mapped — both to prove the memory is
    /// real (the create alone does not, §3.2) and to leave it zeroed, which the
    /// region tier relies on for a freshly-claimed arena.
    #[test]
    fn mapping_leaves_the_granule_zeroed() -> Result<()> {
        let Some(s) = stream() else { return Ok(()) };
        let mut r = Reservation::reserve(&s, 1)?;
        assert!(r.map_granule(0)?);
        let mut back = vec![0xFFu8; 4096];
        // SAFETY: granule 0 is mapped read/write and 4096 is well inside it.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                &mut back,
                r.base(),
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("dtoh from reservation: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        assert!(
            back.iter().all(|&b| b == 0),
            "a mapped granule must be zero"
        );
        Ok(())
    }
}
