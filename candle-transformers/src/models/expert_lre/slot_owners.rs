//! The slot-tenancy mirror `moe_bucketize`'s owner check reads.
//!
//! One `u32` per weight-zone slot, in mapped pinned memory: the expert last
//! installed there, as `(row + 1) << 16 | expert` (0 for a slot never filled).
//! Bucketize checks every VRAM entry it snapshots for a GEMM against the tag
//! of the slot the entry points into, and traps on a mismatch — a tile about
//! to read one expert's weights under another's name, caught at the read
//! instead of as a wrong number layers later.
//!
//! **A tag changes only when a slot gains a tenant** — an install or a
//! relocation's destination — never on an eviction. An eviction retargets the
//! entry while a bucketize may already have read the old value, so clearing the
//! tag then would race the checker against its own subject; a slot that lost
//! its tenant is named by no entry once the retarget is visible and its old
//! readers are done, and the next install overwrites the stale tag.
//!
//! Built only with `tensor-assert`; the check costs one mapped load per
//! snapshotted VRAM entry.

use candle::quantized::cuda::OwnerCheck;
use candle::Result;
use cudarc::driver::sys;

pub(crate) struct SlotOwners {
    host: *mut u32,
    dev: u64,
    slots: usize,
}

// SAFETY: written only by the pipeline thread (the owner of the zone's
// tenancy) with volatile stores; read by the device.
unsafe impl Send for SlotOwners {}
unsafe impl Sync for SlotOwners {}

/// The tag of `expert` in MoE row `row`.
pub(crate) fn owner_tag(row: usize, expert: usize) -> u32 {
    ((row as u32 + 1) << 16) | expert as u32
}

impl SlotOwners {
    /// A zeroed mirror over `slots` slots — the zone's limit, so growth never
    /// outruns it.
    pub(crate) fn new(slots: usize) -> Result<Self> {
        let bytes = slots.max(1) * 4;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("slot owners: mapped allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("slot owners have no device address: {r:?}");
        }
        // SAFETY: `bytes` just allocated, not yet visible to the device.
        unsafe { std::ptr::write_bytes(raw as *mut u8, 0, bytes) };
        Ok(Self {
            host: raw as *mut u32,
            dev,
            slots,
        })
    }

    /// Slot `slot` now holds `(row, expert)`.
    pub(crate) fn set(&self, slot: usize, row: usize, expert: usize) {
        assert!(slot < self.slots, "slot owners: slot {slot} past {}", self.slots);
        // SAFETY: `slot < slots`, inside the allocation.
        unsafe { std::ptr::write_volatile(self.host.add(slot), owner_tag(row, expert)) };
    }

    /// The tag slot `slot` holds.
    pub(crate) fn get(&self, slot: usize) -> u32 {
        assert!(slot < self.slots, "slot owners: slot {slot} past {}", self.slots);
        // SAFETY: as `set`.
        unsafe { std::ptr::read_volatile(self.host.add(slot)) }
    }

    /// What bucketize is handed: these tags over a zone whose slot `s` ends
    /// `s · slot_bytes` below `zone_end`.
    pub(crate) fn check(&self, zone_end: u64, slot_bytes: usize) -> OwnerCheck {
        OwnerCheck {
            owners: self.dev,
            zone_end,
            slot_bytes: slot_bytes as u64,
            slots: self.slots as u32,
        }
    }
}

impl Drop for SlotOwners {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; the device and the
        // pipeline thread are done with it before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tag is the row plus one above the expert, so row 0 expert 0 is not
    /// the empty tag 0.
    #[test]
    fn a_tag_is_the_row_plus_one_over_the_expert() {
        assert_eq!(owner_tag(0, 0), 0x0001_0000);
        assert_eq!(owner_tag(47, 511), (48 << 16) | 511);
    }

    /// Tags land at their slot, start at 0, and the check names the device
    /// address, the zone's end and its slot size.
    #[test]
    fn tags_land_at_their_slot() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let owners = SlotOwners::new(4).unwrap();
        assert_eq!(owners.get(2), 0);
        owners.set(2, 5, 9);
        assert_eq!((owners.get(1), owners.get(2)), (0, (6 << 16) | 9));
        let c = owners.check(0x8000_0000, 0x1000);
        assert_eq!(
            (c.owners, c.zone_end, c.slot_bytes, c.slots),
            (owners.dev, 0x8000_0000, 0x1000, 4)
        );
    }
}
