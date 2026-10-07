use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use cudarc;
use cudarc::cublas::sys::cublasStatus_t;
use cudarc::cublas::CudaBlas;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, HostSlice,
    ValidAsZeroBits,
};
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use super::graph::{CaptureHub, CaptureStats, Paused, WaveCapture};
use super::info_ring::InfoRing;
use super::staged::pack_segments;
use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};
use crate::cuda_backend::{alloc_inheriting, Backing};
use crate::forbidden_alloc;
use crate::wave_provenance::LeaseOrigin;
use candle_kernels::simple::fill::{run_arange_op, FillDType};
use cudarc::driver::DevicePtr;

/// The seed every device's random generator starts from.
///
/// Fixed, and re-applied for each handle: `candle` guarantees that a freshly
/// built device draws the same numbers as the last one, which is what lets a
/// test build a device and assert against literal values (see
/// [`BackendDevice::set_seed`], which replaces the generator rather than
/// re-seeding it, for the same reason).
const DEFAULT_SEED: u64 = 299792458;

/// Largest CUDA ordinal this build holds a device for.
///
/// The same ordinals `gpu_memory::MAX_TRACKED_GPUS` indexes free-VRAM readings
/// by, and the two are meant to move together — a device this file will hand out
/// but that file cannot record an init-free reading for would silently lose the
/// KV budget gate its baseline.
const MAX_CUDA_DEVICES: usize = 16;

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<CudaStream>,
    pub(crate) blas: Arc<CudaBlas>,
    /// The fixed workspace `blas` runs in — see [`new_blas`]. Held beside the
    /// handle for as long as the handle exists.
    blas_workspace: Arc<CudaSlice<u8>>,
    curand: Arc<Mutex<CudaRng>>,
    /// Memoized device copies of small layout/info tables (dims+strides blobs the
    /// strided kernels read), carved from one ring. See [`Self::info_table`].
    info_tables: InfoTables,
    /// Memoized device copies of the token-major → group-major gather permutation.
    /// Keyed by **shape** `(rows, groups)` rather than contents, because the table is
    /// a pure function of that shape: a hit costs a two-word hash and does no host
    /// build and no upload at all. See [`Self::group_major_ids`].
    perm_tables: PermTables,
    /// The quantized-matmul path's activation staging — see
    /// [`Self::with_staging`]. Per stream, like the caches above.
    staging: Staging,
    /// Launch tables for work that synchronises its stream before it returns
    /// — see [`Self::with_synced_upload`]. Separate from `staging` so a launch
    /// that waits out a long copy never holds the forward's scratch.
    synced_staging: Staging,
    /// The wave capture shared by every clone of this handle — see
    /// [`Self::begin_wave_capture`]. Per stream, like the caches above.
    capture: Arc<CaptureHub>,
}

/// Bytes of the workspace every cuBLAS handle is given — NVIDIA's
/// recommendation for this generation of card, and ample for the GEMMs and
/// batched solves this backend issues.
const BLAS_WORKSPACE_BYTES: usize = 4 << 20;

/// A cuBLAS handle on `stream` with a workspace of its own.
///
/// Without one, cuBLAS allocates its workspace on the stream per call, which
/// a wave capture records as a graph allocation and refuses; with one, a GEMM
/// is kernels only, recorded or not.
fn new_blas(stream: &Arc<CudaStream>) -> Result<(Arc<CudaBlas>, Arc<CudaSlice<u8>>)> {
    let blas = CudaBlas::new(stream.clone()).w()?;
    // SAFETY: cuBLAS writes its workspace before reading it.
    let mut workspace = unsafe { stream.alloc::<u8>(BLAS_WORKSPACE_BYTES) }.w()?;
    let (ptr, _g) = workspace.device_ptr_mut(stream);
    // SAFETY: a live handle and a device buffer of the stated size that the
    // returned pair keeps alive beside it.
    let status = unsafe {
        cudarc::cublas::sys::cublasSetWorkspace_v2(
            *blas.handle(),
            ptr as *mut std::ffi::c_void,
            BLAS_WORKSPACE_BYTES,
        )
    };
    drop(_g);
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        crate::bail!("cublasSetWorkspace failed: {status:?}");
    }
    Ok((Arc::new(blas), Arc::new(workspace)))
}

/// One grow-only byte buffer, reused by every call that stages through it.
type Staging = Arc<Mutex<Option<CudaSlice<u8>>>>;

/// Memoized layout-table uploads, keyed by the table's contents.
type InfoTables = Arc<Mutex<InfoRing>>;

/// Memoized gather permutations, keyed by `(rows, groups)`.
type PermTables = Arc<Mutex<HashMap<(usize, usize), Arc<Uploaded<u32>>>>>;

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    /// Refuse an allocation that overlaps memory declared immutable after load.
    ///
    /// The pool is supposed to hand out disjoint blocks; a fresh allocation
    /// landing on a resident weight means it handed one out twice, and the
    /// weight is corrupted the moment anything writes through the new handle.
    /// Catching it here names the allocation site instead of leaving a wrong
    /// number to be found several layers downstream.
    ///
    /// Costs two atomic loads and two compares for an address outside every
    /// declared region, which is every allocation in a healthy run — see
    /// [`crate::readonly_regions`].
    #[cfg(feature = "tensor-assert")]
    fn guard_fresh_allocation<T>(what: &str, slice: &CudaSlice<T>, bytes: usize) {
        use cudarc::driver::DevicePtr;
        let stream = slice.stream().clone();
        let (base, _g) = slice.device_ptr(&stream);
        crate::readonly_regions::forbid_write(what, base, bytes);
    }

    /// No-op twin. Gated on the whole body rather than relying on
    /// `forbid_write`'s stub, because reading the address is not free: it
    /// clones a stream `Arc` and takes a sync guard, on a path that runs for
    /// every device allocation in the process.
    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    fn guard_fresh_allocation<T>(_what: &str, _slice: &CudaSlice<T>, _bytes: usize) {}

    /// `len` elements of device memory, **uninitialised**.
    ///
    /// # Safety
    ///
    /// The returned slice holds whatever the allocator last left there. The
    /// caller must write every element it goes on to read — a kernel that fully
    /// overwrites its output, or an explicit fill — because reading an
    /// unwritten element is undefined behaviour and, on this backend, silently
    /// returns another tenant's bytes rather than faulting.
    ///
    /// [`Self::alloc_zeros`] is the safe counterpart and is the right choice
    /// unless the buffer is provably fully written: the hot-path rule is that a
    /// buffer a kernel completely overwrites must come from here, so the zeroing
    /// memset is not paid on bytes that are about to be stamped anyway.
    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        forbidden_alloc::record("CudaDevice::alloc", len * std::mem::size_of::<T>());
        let _eager = self.pause_capture()?;
        let s = self.stream.alloc::<T>(len).w()?;
        Self::guard_fresh_allocation("CudaDevice::alloc", &s, len * std::mem::size_of::<T>());
        Ok(s)
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>> {
        forbidden_alloc::record("CudaDevice::alloc_zeros", len * std::mem::size_of::<T>());
        let _eager = self.pause_capture()?;
        let s = self.stream.alloc_zeros::<T>(len).w()?;
        Self::guard_fresh_allocation(
            "CudaDevice::alloc_zeros",
            &s,
            len * std::mem::size_of::<T>(),
        );
        Ok(s)
    }

    pub fn memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        // A host→device copy is a bulk write, and its destination is computed
        // by the caller rather than handed out by an allocator — which is why
        // the guards on the allocation paths cannot see it. An upload aimed at
        // the wrong slot writes a contiguous, slot-sized block of plausible
        // bytes over whatever was there.
        self.guard_copy_dst::<T, _>("CudaDevice::memcpy_htod", dst, src.len());
        let _eager = self.pause_capture()?;
        self.stream.memcpy_htod(src, dst).w()
    }

    /// Refuse a copy whose destination overlaps memory declared immutable.
    ///
    /// Reads the destination's address through the same accessor the copy will,
    /// so the range checked is the range written. Costs two atomic loads and two
    /// compares when nothing is declared nearby — see
    /// [`crate::readonly_regions`] — and compiles away entirely without the
    /// `tensor-assert` feature.
    #[cfg(feature = "tensor-assert")]
    fn guard_copy_dst<T, D: DevicePtrMut<T>>(&self, what: &str, dst: &mut D, elems: usize) {
        let stream = self.stream.clone();
        let (base, _g) = dst.device_ptr_mut(&stream);
        crate::readonly_regions::forbid_write(what, base, elems * std::mem::size_of::<T>());
    }

    /// No-op twin — same reasoning as [`Self::guard_fresh_allocation`], on a
    /// path that runs for every host↔device copy.
    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    fn guard_copy_dst<T, D: DevicePtrMut<T>>(&self, _what: &str, _dst: &mut D, _elems: usize) {}

    pub fn memcpy_dtov<T: DeviceRepr, Src: DevicePtr<T>>(&self, src: &Src) -> Result<Vec<T>> {
        let _eager = self.pause_capture()?;
        self.stream.memcpy_dtov(src).w()
    }

    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        // Same reasoning as `memcpy_htod`: a bulk write to a caller-computed
        // destination, invisible to every allocation-path guard.
        self.guard_copy_dst::<T, _>("CudaDevice::memcpy_dtod", dst, src.len());
        let _eager = self.pause_capture()?;
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> Result<CudaSlice<T>> {
        // Allocates as well as copying: the destination slice is fresh device
        // memory, so this is a driver allocation like the two above.
        forbidden_alloc::record("CudaDevice::memcpy_stod", std::mem::size_of_val(src));
        let _eager = self.pause_capture()?;
        let s = self.stream.memcpy_stod(src).w()?;
        // Allocates as well as copying, so it is an allocation path like the
        // two above and needs the same guard — and unlike them it writes to the
        // range immediately, so a collision here corrupts before anything else
        // gets a chance to notice.
        Self::guard_fresh_allocation("CudaDevice::memcpy_stod", &s, std::mem::size_of_val(src));
        Ok(s)
    }

    /// [`Self::memcpy_stod`] with the destination taken from `origin`'s arena.
    ///
    /// Host uploads are the one wave-path allocation provenance cannot reach on
    /// its own: a table built on the CPU has no device operand to inherit from,
    /// so there is nothing for the rule to follow. What the call sites *do* have
    /// is the arena the table is about to be read alongside — the tile and token
    /// tables are consumed by the very kernels whose operands are already on the
    /// span — so they pass that arena in explicitly.
    ///
    /// The copy itself is unchanged; only the destination moves.
    /// Host upload placed on `origin`'s span, as a plain slice plus its backing.
    ///
    /// The same work as [`Self::memcpy_stod_from`], returning the pieces
    /// `CudaStorage` is built from rather than an [`Uploaded`] guard — which is
    /// what a *tensor* constructor needs, since `CudaStorageSlice` holds the
    /// slice directly.
    ///
    /// This is the only way a host-built table can land on the reservation:
    /// there is no device operand to inherit a ticket from, so the caller names
    /// the span it belongs to. Without it every per-wave descriptor table — the
    /// DeltaNet pointer tables, the rotary layouts, the batched row maps — is a
    /// driver allocation inside the wave.
    pub fn memcpy_stod_leased<T: DeviceRepr + ValidAsZeroBits, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        origin: Backing,
    ) -> Result<(CudaSlice<T>, Backing)> {
        // An allocation that reaches the driver ends the recording segment on
        // its own; a carve from the wave's arena does not need to.
        let (mut dst, backing) = unsafe { alloc_inheriting::<T>(self, src.len(), origin)? };
        self.upload_into_slice(&mut dst, src)?;
        Ok((dst, backing))
    }

    /// Copy `src` into the front of `dst`. Inside a recording wave capture the
    /// copy is recorded (see [`Self::upload_raw`]); otherwise it is queued on
    /// the compute stream like any upload.
    pub fn upload_into_slice<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        dst: &mut CudaSlice<T>,
        src: &Src,
    ) -> Result<()> {
        let (at, _g) = dst.device_ptr_mut(&self.stream);
        // SAFETY: the slice is read on the host right here, before the guard
        // drops; a host buffer has nothing to wait for.
        let (host, _s) = unsafe { src.stream_synced_slice(&self.stream) };
        // SAFETY: `host` is `len` plain device-representable values.
        let bytes = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        self.upload_raw(at, bytes)
    }

    /// Record the copy of `src` to the device address `dst` into this thread's
    /// recording segment, staged through the wave's ring, and return `true`;
    /// return `false` and do nothing when this thread is not recording or the
    /// ring is full.
    ///
    /// For an uploader with an eager path of its own — pinned staging that
    /// outruns the pageable copy [`Self::upload_raw`] falls back to — which it
    /// then runs inside [`Self::pause_capture`], behind the launches recorded
    /// so far.
    pub fn record_upload(&self, dst: u64, src: &[u8]) -> bool {
        self.capture.record_upload(dst, src)
    }

    /// Copy `src` to the device address `dst`, which holds at least
    /// `src.len()` bytes.
    ///
    /// While this thread records a wave capture, the bytes are staged in the
    /// wave's ring and their copy is recorded into the segment, so the upload
    /// does not end it. Otherwise — or when the ring is full — the copy is
    /// queued on the compute stream behind everything issued before it; the
    /// driver has staged pageable bytes before it returns, so `src` may go.
    pub fn upload_raw(&self, dst: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() || self.record_upload(dst, src) {
            return Ok(());
        }
        let _eager = self.pause_capture()?;
        // SAFETY: the caller's `dst` holds `src.len()` bytes; queued on the
        // compute stream.
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                self.stream.cu_stream(),
            )
            .result()
            .w()
        }
    }

    pub fn memcpy_stod_from<T: DeviceRepr + ValidAsZeroBits, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        origin: Backing,
    ) -> Result<Uploaded<T>> {
        let (mut dst, backing) = unsafe { alloc_inheriting::<T>(self, src.len(), origin)? };
        self.upload_into_slice(&mut dst, src)?;
        Ok(Uploaded {
            slice: std::mem::ManuallyDrop::new(dst),
            backing,
            _anchor: None,
        })
    }

    /// Run `f` with a staging buffer of at least `bytes`, the scratch a
    /// quantized matmul quantizes its activation into before the kernel reads
    /// it.
    ///
    /// **Reused, not allocated per call.** The scratch is written and read by
    /// launches on this handle's stream, so the next call's quantize is
    /// ordered after the previous matmul's read and may take the same bytes.
    /// It grows to the widest request seen — by doubling, so a ramp of widths
    /// settles in a few steps — and is never handed back, which is what makes a
    /// steady-state wave allocation-free. The lock is held while `f` runs, so
    /// two callers on this handle cannot interleave their quantize and read.
    pub fn with_staging<R>(
        &self,
        bytes: usize,
        f: impl FnOnce(&mut CudaSlice<u8>) -> Result<R>,
    ) -> Result<R> {
        // Eager for its whole extent: the scratch is reused by the next call,
        // so a launch reading it must execute before the next upload lands.
        let _eager = self.pause_capture()?;
        let mut slot = self.staging.lock().unwrap();
        if !slot.as_ref().is_some_and(|s| s.len() >= bytes) {
            let grown = slot.as_ref().map_or(0, |s| s.len() * 2).max(bytes).max(1);
            // Dropping the old buffer frees it on this stream, after every
            // launch already queued against it.
            *slot = None;
            // SAFETY: staging is written by the quantize the caller launches
            // before its kernel reads it.
            *slot = Some(unsafe { self.alloc::<u8>(grown)? });
        }
        f(slot.as_mut().expect("sized above"))
    }

    /// Upload `data` into the staging scratch ([`Self::with_staging`]) and run
    /// `f` with its device address — for a launch's small host-built table
    /// (a job list, a carry list) that lives from the upload to the kernel's
    /// read and no further.
    ///
    /// Allocates nothing in the steady state, inside a forward or between two:
    /// the copy and the launch `f` queues are ordered on this stream, and the
    /// scratch is not handed to another caller until `f` returns.
    ///
    /// **Inside a recording wave capture it records rather than pauses**, once
    /// the scratch is already wide enough: the copy is staged through the
    /// wave's ring ([`Self::record_upload`]) and `f`'s launches are recorded
    /// behind it. A recorded chain runs in stream order, so the next call's
    /// copy into the scratch lands after this call's launches have read it —
    /// the same guarantee the eager path buys by running both now — and a
    /// pause before any eager user of the scratch submits everything recorded
    /// ahead of it. Pausing instead ended the wave's segment at every call: a
    /// QSA index append per attention layer, each a graph launch and a host
    /// round trip behind it. Growth still runs eagerly; it allocates.
    pub fn with_staged_upload<T: DeviceRepr, R>(
        &self,
        data: &[T],
        f: impl FnOnce(u64) -> Result<R>,
    ) -> Result<R> {
        let bytes = std::mem::size_of_val(data);
        // SAFETY: `data` is `bytes` of plain device-representable values; read
        // as bytes, it is exactly what the kernel reads back.
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
        if bytes > 0 {
            let slot = self.staging.lock().unwrap();
            if let Some(buf) = slot.as_ref().filter(|s| s.len() >= bytes) {
                let (ptr, _guard) = buf.device_ptr(&self.stream);
                if self.record_upload(ptr, src) {
                    // The lock is held while `f` records, as on the eager path:
                    // no other caller's copy can be recorded between this
                    // copy and the launches that read it.
                    return f(ptr);
                }
            }
        }
        self.with_staging(bytes, |buf| {
            if bytes > 0 {
                self.stream
                    .memcpy_htod(src, &mut buf.slice_mut(..bytes))
                    .w()?;
            }
            let (ptr, _guard) = buf.device_ptr(&self.stream);
            f(ptr)
        })
    }

    /// Upload `segments` as one copy into the staging scratch
    /// ([`Self::with_staged_upload`]) and run `f` with each segment's device
    /// address — for a launch fed by several host arrays (a descriptor table
    /// and the arrays it indexes), which would otherwise take one allocation
    /// apiece. Build a segment from a typed slice with [`super::staged::segment`].
    pub fn with_staged_segments<R>(
        &self,
        segments: &[&[u8]],
        f: impl FnOnce(&[u64]) -> Result<R>,
    ) -> Result<R> {
        let (bytes, offsets) = pack_segments(segments);
        self.with_staged_upload(&bytes, |base| {
            let addrs: Vec<u64> = offsets.iter().map(|&o| base + o as u64).collect();
            f(&addrs)
        })
    }

    /// Upload `segments` on `stream` into a held scratch, run `f` with each
    /// segment's device address, then synchronise `stream` — for a launch
    /// table on a stream other than this handle's (the persistence thread's
    /// copy stream), where the stream-ordered reuse
    /// [`Self::with_staged_segments`] relies on does not hold.
    ///
    /// **Reuse rests on the synchronise, not on stream order.** The scratch is
    /// locked from the upload until `stream` has drained, so the next caller —
    /// on any stream — finds every read of it retired. It grows by doubling
    /// and is never handed back, so a steady stream of launches allocates
    /// nothing. Its own lock: a caller waiting out a long copy here never
    /// holds the forward's [`Self::with_staging`].
    pub fn with_synced_upload<R>(
        &self,
        stream: &Arc<CudaStream>,
        segments: &[&[u8]],
        f: impl FnOnce(&[u64]) -> Result<R>,
    ) -> Result<R> {
        let (bytes, offsets) = pack_segments(segments);
        let _eager = self.pause_capture()?;
        let mut slot = self.synced_staging.lock().unwrap();
        if !slot.as_ref().is_some_and(|s| s.len() >= bytes.len()) {
            let grown = slot
                .as_ref()
                .map_or(0, |s| s.len() * 2)
                .max(bytes.len())
                .max(1);
            *slot = None;
            // SAFETY: every byte a launch reads is written by the upload below.
            *slot = Some(unsafe { self.alloc::<u8>(grown)? });
            // The pool hands the bytes out in this handle's stream order, and
            // `stream` may be another one: wait for the allocation to retire
            // before a copy on `stream` writes it. Growth only.
            self.stream.synchronize().w()?;
        }
        let buf = slot.as_mut().expect("sized above");
        if !bytes.is_empty() {
            stream
                .memcpy_htod(&bytes, &mut buf.slice_mut(..bytes.len()))
                .w()?;
        }
        let base = buf.device_ptr(stream).0;
        let addrs: Vec<u64> = offsets.iter().map(|&o| base + o as u64).collect();
        let out = f(&addrs);
        stream.synchronize().w()?;
        out
    }

    /// Device copy of a small layout table (the dims/strides blob a strided
    /// kernel reads), memoized by contents and carved from this handle's info
    /// ring — see [`super::info_ring`]. A steady-state wave hits the memo and
    /// uploads nothing; a new shape costs one small copy and no allocation.
    pub fn info_table(&self, info: &[usize]) -> Result<Arc<Uploaded<usize>>> {
        self.info_tables.lock().unwrap().table(self, info)
    }

    /// The token-major → group-major gather permutation, memoized per `(rows, groups)`:
    ///
    /// ```text
    /// ids[g * rows + t] = t * groups + g        for g in 0..groups, t in 0..rows
    /// ```
    ///
    /// This is the row order the grouped output projection needs. Its source `[rows, groups, w]`
    /// activation is contiguous, so group `g`'s rows are interleaved with stride `groups`; the
    /// grouped matmul wants each group's rows adjacent, and gathering with this table produces
    /// that without ever materialising the permutation in f32.
    ///
    /// **Keyed by shape, not contents.** The table is a pure function of `(rows, groups)`, so a
    /// hit is a two-word hash — no host build, no upload. Building it per call instead was the
    /// same WDDM submission storm [`Self::info_table`] exists to prevent, except worse: it also
    /// spent `O(rows·groups)` of host time per layer per wave.
    ///
    /// Entries are pool-owned (`Backing::Owned`) and outlive any single wave; kernels only read
    /// them. Cleared wholesale past a bound, like the info-table cache — in-flight users hold
    /// their own `Arc`, so clearing is safe and a rebuild is trivial.
    pub fn group_major_ids(&self, rows: usize, groups: usize) -> Result<Arc<Uploaded<u32>>> {
        let key = (rows, groups);
        let mut cache = self.perm_tables.lock().unwrap();
        if let Some(t) = cache.get(&key) {
            return Ok(t.clone());
        }
        // Clearing frees, and the build below uploads.
        let _eager = self.pause_capture()?;
        if cache.len() >= 1024 {
            cache.clear();
        }
        let mut ids: Vec<u32> = Vec::with_capacity(rows * groups);
        for g in 0..groups {
            for t in 0..rows {
                ids.push((t * groups + g) as u32);
            }
        }
        let slice = self.memcpy_stod(&ids)?;
        let t = Arc::new(Uploaded {
            slice: std::mem::ManuallyDrop::new(slice),
            backing: Backing::Owned,
            _anchor: None,
        });
        cache.insert(key, t.clone());
        Ok(t)
    }

    /// Generate an integer arange (`buf[i] = start + i*step`, exact integer arithmetic)
    /// directly on the device — no host-side build, no tiny H2D upload. `Tensor::arange`
    /// index tensors are hot-path gather indices, and per-call host uploads of them were
    /// a measured WDDM submission storm. Integer dtypes only (U8/U32/I64); float aranges
    /// keep the host build (its repeated-addition rounding is the documented semantics,
    /// which the kernel's closed form would not reproduce bit-for-bit). Start/step are
    /// passed as bits per `run_arange_op`.
    pub fn arange_int(
        &self,
        dtype: DType,
        start_bits: u64,
        step_bits: u64,
        len: usize,
    ) -> Result<CudaStorage> {
        let launch = |ptr: u64, fill_dtype: FillDType| unsafe {
            run_arange_op(
                fill_dtype as i32,
                ptr as *mut std::ffi::c_void,
                start_bits,
                step_bits,
                len,
                self.cuda_stream().cu_stream() as *mut std::ffi::c_void,
            );
        };
        let slice = match dtype {
            DType::U8 => {
                let s = unsafe { self.alloc::<u8>(len)? };
                {
                    let (ptr, _g) = s.device_ptr(&self.stream);
                    launch(ptr, FillDType::U8);
                }
                CudaStorageSlice::U8(s)
            }
            DType::U32 => {
                let s = unsafe { self.alloc::<u32>(len)? };
                {
                    let (ptr, _g) = s.device_ptr(&self.stream);
                    launch(ptr, FillDType::U32);
                }
                CudaStorageSlice::U32(s)
            }
            DType::I64 => {
                let s = unsafe { self.alloc::<i64>(len)? };
                {
                    let (ptr, _g) = s.device_ptr(&self.stream);
                    launch(ptr, FillDType::I64);
                }
                CudaStorageSlice::I64(s)
            }
            _ => crate::bail!("arange_int: integer dtypes only, got {dtype:?}"),
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }
}

/// An uploaded table, freed only if this owns it.
///
/// [`CudaDevice::memcpy_stod_from`] can return a slice over either a pool
/// allocation or a wave range, and a bare [`cudarc::driver::CudaSlice`] cannot
/// tell the difference: its `Drop` frees unconditionally, which on arena memory
/// is a `cuMemFreeAsync` against a span the pool never allocated. Carrying the
/// backing alongside the slice is what makes the right disposal the only
/// reachable one — the same job [`super::Backing`] does for [`super::CudaStorage`],
/// at the one place that hands out a raw slice.
pub struct Uploaded<T> {
    slice: std::mem::ManuallyDrop<CudaSlice<T>>,
    backing: Backing,
    /// The allocation a leased `slice` points into, when that allocation is
    /// not owned by an arena that outlives it — an info-ring entry keeps its
    /// ring buffer alive this way after the ring has moved on to a fresh one.
    /// Dropped after `slice` is disposed of, so the view never outlives it.
    _anchor: Option<Arc<CudaSlice<usize>>>,
}

impl<T> Uploaded<T> {
    /// A leased view over memory `anchor` (when given) keeps alive.
    pub(crate) fn leased(slice: CudaSlice<T>, anchor: Option<Arc<CudaSlice<usize>>>) -> Self {
        Self {
            slice: std::mem::ManuallyDrop::new(slice),
            backing: Backing::Lease(LeaseOrigin::Foreign),
            _anchor: anchor,
        }
    }
}

impl<T> std::ops::Deref for Uploaded<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &Self::Target {
        &self.slice
    }
}

impl<T> Drop for Uploaded<T> {
    fn drop(&mut self) {
        match self.backing {
            // SAFETY: `slice` is live and dropped exactly once, here.
            Backing::Owned => unsafe { std::mem::ManuallyDrop::drop(&mut self.slice) },
            // A view over a range the arena owns: the generation's reset reclaims
            // the bytes, so the memory must not be freed here. `leak` — not a bare
            // skip — is what does that correctly: it waits on the slice's
            // read/write events, destroys them, and decrements the stream's `Arc`.
            // Suppressing the drop instead would strand two `CudaEvent`s and a
            // stream refcount per upload, which the MoE path issues several times
            // per layer per token. Same obligation, and same reasoning, as
            // `CudaStorageSlice::leak_view`.
            //
            // SAFETY: `slice` is taken and consumed exactly once, here.
            Backing::Lease(_) => unsafe {
                std::mem::ManuallyDrop::take(&mut self.slice).leak();
            },
        }
    }
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> CudaFunction {
        self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    /// The stream to launch on: the capture stream while this thread records a
    /// wave ([`Self::begin_wave_capture`]), the compute stream otherwise.
    ///
    /// Launches take their stream from here. Anything that outlives the call
    /// — a slice's home stream, a fence, a readback — takes
    /// [`Self::compute_stream`] instead, because the capture stream executes
    /// nothing until its segment is launched.
    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        self.capture.launch_stream(&self.stream)
    }

    /// The device's compute stream — the legacy null stream every eager launch
    /// and every graph launch goes on.
    pub fn compute_stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    /// Record this thread's launches as a chain of graphs until the returned
    /// guard finishes.
    ///
    /// Launches are recorded, not executed, and are handed to the driver one
    /// segment at a time. A segment ends at every call that has to meet the
    /// device eagerly — an allocation, a free, an upload, a readback, a
    /// synchronise — and at every [`Self::flush_launches`], and is launched
    /// into the compute stream before that call runs, so the order every
    /// operation executes in is the order it was issued in. See
    /// [`super::graph`] for the mechanism.
    ///
    /// The wave opens **held**: nothing is recorded until the model reaches
    /// the launches it wants recorded and calls [`Self::record_launches`], so a
    /// forward's setup — its admission, tier placement and tables — runs
    /// eagerly as it always has.
    pub fn begin_wave_capture(&self) -> Result<WaveCapture> {
        self.capture.begin_wave(&self.stream)?;
        Ok(WaveCapture {
            device: self.clone(),
            finished: false,
            _thread_bound: PhantomData,
        })
    }

    /// Start recording the wave [`Self::begin_wave_capture`] opened on this
    /// thread. Does nothing on a thread with no wave open.
    pub fn record_launches(&self) -> Result<()> {
        self.capture.record()
    }

    /// Hand everything this thread has issued to the driver and nudge WDDM to
    /// submit it — the point a host protocol polling the device for a launch's
    /// output waits on. Inside a wave capture this ends the recording segment.
    pub fn flush_launches(&self) -> Result<()> {
        self.capture.flush(&self.stream)
    }

    /// Suspend this thread's wave capture, if it has one, until the guard
    /// drops: the recorded segment is launched first, and the caller's eager
    /// device calls run in order behind it.
    pub fn pause_capture(&self) -> Result<Option<Paused<'_>>> {
        self.capture.pause(&self.stream)
    }

    /// Drop `value` — something owning device memory — once every launch this
    /// thread has recorded so far has been issued; at once when it is not
    /// recording. A free is refused while a capture records, and a recorded
    /// launch may still read the memory, so it waits for its segment.
    pub fn retire<T: Send + 'static>(&self, value: T) {
        drop(self.capture.retire(Box::new(value)));
    }

    /// What this device's wave captures have done so far.
    pub fn capture_stats(&self) -> CaptureStats {
        self.capture.stats()
    }

    pub(crate) fn capture_hub(&self) -> &CaptureHub {
        &self.capture
    }

    /// The cuBLAS handle, bound to [`Self::cuda_stream`].
    ///
    /// For the BLAS calls this backend does not wrap as tensor ops. `matmul`
    /// covers GEMM; a model needing something else from the library — the
    /// batched triangular solve the DeltaNet chunked scan uses in place of an
    /// explicit matrix inverse — reaches it through here rather than opening a
    /// second handle, which would carry its own stream and lose the ordering
    /// every other op on this device relies on.
    ///
    /// Re-bound on every call, because the stream a launch belongs on depends
    /// on whether this thread is recording a wave.
    pub fn cublas(&self) -> Result<&CudaBlas> {
        let stream = self.cuda_stream();
        // SAFETY: a live handle of this device and a stream of its context.
        unsafe { cudarc::cublas::result::set_stream(*self.blas.handle(), stream.cu_stream() as _) }
            .w()?;
        // Setting the stream hands the handle back to cuBLAS's own workspace
        // pool, which allocates on the stream; give it its fixed one again.
        let (ptr, _g) = self.blas_workspace.device_ptr(&self.stream);
        // SAFETY: the handle's own workspace, alive as long as the handle.
        let status = unsafe {
            cudarc::cublas::sys::cublasSetWorkspace_v2(
                *self.blas.handle(),
                ptr as *mut std::ffi::c_void,
                BLAS_WORKSPACE_BYTES,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            crate::bail!("cublasSetWorkspace failed: {status:?}");
        }
        Ok(&self.blas)
    }

    /// Returns the underlying CUDA context.
    ///
    /// Useful for creating secondary streams ([`CudaContext::new_stream`]) or
    /// events ([`CudaContext::new_event`]) for overlapping DMA and compute.
    pub fn cuda_context(&self) -> &Arc<cudarc::driver::CudaContext> {
        &self.context
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.context.disable_event_tracking()
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        ug_cuda::code_gen::r#gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = match self.context.load_module(ptx.clone()) {
            Ok(module) => module,
            // A driver older than the installed toolkit refuses the toolkit's
            // PTX ISA at JIT time. The AOT kernels never hit this — they ship
            // SASS only — so this runtime-generated kernel is the one PTX the
            // driver is ever asked to JIT. The toolkit's own `ptxas` can still
            // assemble that PTX to native SASS for this device, and a cubin
            // loads without any PTX version check, so assemble toolkit-side
            // and load native code instead.
            Err(e) if e.0 == CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION => {
                self.load_module_via_ptxas(func_name, &ptx)?
            }
            Err(e) => return Err(e).w()?,
        };
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    /// Assemble `ptx` to this device's native SASS with the toolkit's `ptxas`
    /// and load the cubin — the fallback for a driver that refuses to JIT the
    /// toolkit's PTX ISA version (see [`Self::compile`]).
    pub(crate) fn load_module_via_ptxas(
        &self,
        func_name: &str,
        ptx: &cudarc::nvrtc::Ptx,
    ) -> Result<Arc<CudaModule>> {
        let (major, minor) = self.compute_capability()?;
        let dir = std::env::temp_dir();
        let stem = format!("candle-ug-{}-{func_name}", std::process::id());
        let ptx_path = dir.join(format!("{stem}.ptx"));
        let cubin_path = dir.join(format!("{stem}.cubin"));
        std::fs::write(&ptx_path, ptx.to_src())?;
        let module = (|| -> Result<Arc<CudaModule>> {
            let out = std::process::Command::new("ptxas")
                .arg(format!("-arch=sm_{major}{minor}"))
                .arg("-o")
                .arg(&cubin_path)
                .arg(&ptx_path)
                .output()
                .map_err(|e| {
                    crate::Error::Msg(format!("ptxas is not runnable for {func_name}: {e}"))
                })?;
            if !out.status.success() {
                crate::bail!(
                    "ptxas could not assemble the generated PTX for {func_name}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            self.context
                .load_module(cudarc::nvrtc::Ptx::from_file(&cubin_path))
                .w()
        })();
        // `cuModuleLoad` has read the file by the time it returns; the module
        // keeps its own copy, so the temp files can go whatever the outcome.
        let _ = std::fs::remove_file(&ptx_path);
        let _ = std::fs::remove_file(&cubin_path);
        module
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }
}

impl CudaDevice {
    /// Refuses a device this build carries no kernel image for.
    ///
    /// The kernels are compiled to native SASS for the architectures in
    /// `candle_kernels::BUILT_ARCHES` and **no PTX is emitted**, so a card
    /// outside that set has nothing to run and nothing to JIT from.
    ///
    /// # Why this panics
    ///
    /// Every launch on such a card fails with `cudaErrorNoKernelImageForDevice`,
    /// and the kernel launchers return `void` — so nothing observes the error.
    /// The caller reads back the `alloc_zeros` buffer it passed in and carries
    /// on, which means the entire KV data path (writes, migration,
    /// quantization, format selection, provenance) silently produces zeros.
    /// There is no numerical answer to give and no partial mode worth running:
    /// the machine cannot execute this build at all. A `Result` here would be a
    /// value some caller could log and continue past, and continuing produces
    /// confidently wrong output — so this is the one place a hard stop is
    /// correct.
    ///
    /// Adding the card is one entry in `KERNEL_ARCHES` (`candle-kernels/build_utils.rs`).
    fn validate_compute_capability(context: &Arc<cudarc::driver::CudaContext>) -> Result<()> {
        use cudarc::driver::sys::CUdevice_attribute;
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .w()? as u32;
        let minor = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .w()? as u32;
        if !candle_kernels::has_kernel_image(major, minor) {
            let built = candle_kernels::BUILT_ARCHES
                .iter()
                .map(|sm| format!("sm_{sm}"))
                .collect::<Vec<_>>()
                .join(", ");
            panic!(
                "CUDA device is SM {major}.{minor}, and this build carries kernel images only \
                 for [{built}] (native SASS, no PTX). Nothing would execute on this card: every \
                 kernel launch fails with cudaErrorNoKernelImageForDevice and returns \
                 zero-filled buffers instead of an error. Add {major}{minor} to KERNEL_ARCHES \
                 in candle-kernels/build_utils.rs and rebuild.\n\
                 (A cubin for X.y runs on X.z only when z >= y — so sm_86 does not cover an \
                 8.0 device such as the A100.)"
            );
        }
        Ok(())
    }

    /// Turn off cudarc's per-argument cross-stream event tracking, BEFORE the
    /// first allocation (only slices created after the call are affected).
    ///
    /// With tracking on, EVERY `device_ptr`/`device_ptr_mut` extraction —
    /// several per kernel launch — records a `CudaEvent` on drop and stream-
    /// waits the slice's prior read/write events once the process is in
    /// multi-stream mode. Measured over one decode-heavy gate: 1.54M
    /// `cuEventRecord` + 2.69M `cuStreamWaitEvent` + 1M `cuEventDestroy`
    /// (~3.6 events per kernel, ~4.5s of host time) for 428k launches — on
    /// WDDM, where host submission time IS the decode wall.
    ///
    /// Safety of turning it off: this engine orders every cross-stream
    /// interaction EXPLICITLY at the producer/consumer pair — the expert
    /// pipeline's `CopyBatchFence` ring and `order_copies_after_compute` /
    /// `order_compute_after_copies`, the cold-staging ring's publish events,
    /// the streamer's compute-order + plan fences, the DtoH readback events,
    /// and the `TableRing` half fences. Compute itself runs on ONE stream per
    /// device, where ordering is implicit. cudarc's per-argument events are a
    /// second, redundant safety net over those explicit fences; a
    /// cross-stream path added WITHOUT an explicit fence is a bug here by
    /// design (and what the bit-exact gates + the Fletcher-32 golden
    /// checksums exist to catch).
    fn disable_per_arg_event_tracking(context: &std::sync::Arc<cudarc::driver::CudaContext>) {
        // SAFETY: called before any allocation on this context, so no slice
        // predates the setting (the documented hazard is mixing tracked and
        // untracked slices).
        unsafe { context.disable_event_tracking() };
    }

    /// A second handle on the same device with a **stream** of its own.
    ///
    /// A stream is all it adds. This used to build the whole device again for
    /// the same ordinal, taking a fresh cuBLAS handle and curand generator with
    /// it, which is the leak [`BackendDevice::new`] documents — reached by a
    /// different door, and by a caller whose whole intent was "the same device,
    /// another stream". Everything but the stream now comes from the cached
    /// device, including the content-keyed caches below, so the two handles read
    /// each other's uploads instead of each making their own.
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let base = Self::new(ordinal)?;
        let stream = base.context.new_stream().w()?;
        let (blas, blas_workspace) = new_blas(&stream)?;
        let curand = cudarc::curand::CudaRng::new(DEFAULT_SEED, stream.clone()).w()?;
        // **Compiled modules are shared; memoised buffers are not.** A
        // `CudaModule` belongs to the context, is read-only once built, and is
        // expensive enough that rebuilding it per handle is the cost this
        // function is trying to avoid. The two table caches memoise
        // `Uploaded<_>` — device *buffers*, allocated and freed on the stream
        // that made them. Handing one to the other handle would let a kernel on
        // this stream read a buffer uploaded on the base's, with no event
        // between them and a `cuMemFreeAsync` on the far stream to race.
        Ok(Self {
            id: DeviceId::new(),
            context: base.context.clone(),
            custom_modules: base.custom_modules.clone(),
            stream,
            blas,
            blas_workspace,
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            info_tables: Arc::new(Mutex::new(InfoRing::default())),
            perm_tables: Arc::new(Mutex::new(HashMap::new())),
            staging: Arc::new(Mutex::new(None)),
            synced_staging: Arc::new(Mutex::new(None)),
            capture: Arc::new(CaptureHub::default()),
        })
    }

    /// Give this handle its own cuBLAS handle and curand generator.
    ///
    /// **Everything stateful is rebuilt; everything memoised is shared.** The
    /// caches on a `CudaDevice` are content- and shape-keyed tables of immutable
    /// bytes, so two handles reading one entry is the point of keeping them.
    /// These two are neither, and inheriting either is a silent wrong answer
    /// rather than a failure:
    ///
    /// - **cuBLAS.** A handle carries internal workspace and stream state, and
    ///   NVIDIA's contract is one handle per thread — sharing one is a data race
    ///   between concurrent GEMMs, not merely contention. It cost `candle-core`'s
    ///   `conv1d_gpu` a wrong result and two `conv2d_*_gpu` an illegal access,
    ///   intermittently, only under a full parallel suite.
    /// - **curand.** A device is built expecting to draw from [`DEFAULT_SEED`];
    ///   one advancing generator makes what a caller draws a function of who
    ///   drew before it, so a `Tensor::randn` fixture stops being a fixture. It
    ///   cost two `candle-transformers` tests, each passing alone and failing in
    ///   the suite.
    ///
    /// Both are cheap to build and neither is what leaked — 512 handles' worth of
    /// each is a passing test (`cuda_device_reuse.rs`), which is how they were
    /// ruled out as the cause of the exhaustion the cache exists to stop.
    ///
    /// Called on **both** paths out of the cache: the hit, and the loser of a
    /// first-touch race, which is also handed a clone of a shared device.
    fn give_own_stateful(&mut self) -> Result<()> {
        (self.blas, self.blas_workspace) = new_blas(&self.stream)?;
        self.curand = Arc::new(Mutex::new(CudaRng(
            cudarc::curand::CudaRng::new(DEFAULT_SEED, self.stream.clone()).w()?,
        )));
        Ok(())
    }

    /// Returns the compute capability of this device as a (major, minor) tuple.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::CUdevice_attribute;
        let major = self
            .context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .w()?;
        let minor = self
            .context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .w()?;
        Ok((major, minor))
    }

    /// Returns the L2 cache size in bytes.
    pub fn l2_cache_size(&self) -> Result<usize> {
        use cudarc::driver::sys::CUdevice_attribute;
        let size = self
            .context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .w()?;
        Ok(size as usize)
    }

    /// Streaming-multiprocessor count — the occupancy target for the int8 dense tiling heuristic.
    pub fn multiprocessor_count(&self) -> Result<usize> {
        use cudarc::driver::sys::CUdevice_attribute;
        let n = self
            .context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .w()?;
        Ok(n as usize)
    }

    /// Returns true if this device supports tensor cores (SM >= 8.0, i.e., Ampere or newer).
    pub fn supports_tensor_cores(&self) -> bool {
        self.compute_capability()
            .map(|(major, _minor)| major >= 8)
            .unwrap_or(false)
    }

    /// Returns true if this device can run the int8 `m16n8k32` tensor-core MMA used by the
    /// q8a128 × KO matmul — i.e. compute capability >= 8.0 (Ampere/Ada/Hopper). On older GPUs
    /// the int8 path has no kernel, so callers fall back to the FP16 reference path.
    pub fn supports_int8_mma(&self) -> bool {
        self.compute_capability()
            .map(|(major, _minor)| major >= 8)
            .unwrap_or(false)
    }

    /// Make this device's CUDA context current on the calling thread.
    ///
    /// A CUDA context is per-thread state. Every method here that issues driver
    /// calls binds first, so code that goes through `CudaDevice` never has to
    /// think about it — but code that takes a raw `CUstream` or device address
    /// and calls the driver itself must, and a thread that never bound gets
    /// `CUDA_ERROR_INVALID_CONTEXT` rather than anything that reads as a
    /// threading mistake.
    ///
    /// That is not hypothetical: the provenance gallery's page upload is a raw
    /// `memcpy_htod_async`, and moving the normalization warm-ups onto their own
    /// rayon pool gave them worker threads that had never bound this context.
    /// The upload failed, the scan fell back to a host walk of the whole
    /// gallery, and boot never finished.
    ///
    /// Idempotent and cheap — a `cuCtxSetCurrent` on a thread that already has
    /// it costs nothing worth measuring.
    pub fn bind_to_thread(&self) -> Result<()> {
        self.context.bind_to_thread().w()
    }

    /// Returns (free, total) GPU memory in bytes.
    ///
    /// Binds this device's CUDA context to the current thread and queries
    /// `cuMemGetInfo_v2` for the actual free and total device memory.
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        self.context.bind_to_thread().w()?;
        cudarc::driver::result::mem_get_info().w()
    }

    /// Bytes currently allocated by *our* CUDA stream-ordered memory pool on
    /// this device — our live GPU footprint (model weights + KV cache +
    /// activations), as tracked by CUDA itself via `cuMemAllocAsync`. Unlike
    /// `mem_get_info().free`, this counts only *our* allocations and excludes
    /// other processes' pageable memory entirely, so `total - pool_used` is the
    /// correct budget denominator on WDDM (where `free` is polluted by whatever
    /// desktop/IDE memory happens to be resident and the OS evicts on demand).
    ///
    /// Errors if the device doesn't use the async pool allocator (pre-Pascal /
    /// pools unsupported); callers treat that as "unknown" and fall back.
    pub fn pool_used_bytes(&self) -> Result<usize> {
        self.pool_attr(
            cudarc::driver::sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        )
    }

    /// Bytes currently *reserved from the OS* by our CUDA memory pool — the
    /// pool's high-water footprint. `reserved - used` is memory the pool holds
    /// but isn't using, available to satisfy new allocations *without* touching
    /// the driver's free VRAM. The budget gate uses this so that reusing pooled
    /// memory (e.g. after a KV seal frees float arenas) isn't mistaken for new
    /// OS pressure. See [`pool_used_bytes`].
    pub fn pool_reserved_bytes(&self) -> Result<usize> {
        self.pool_attr(
            cudarc::driver::sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        )
    }

    fn pool_attr(&self, attr: cudarc::driver::sys::CUmemPool_attribute_enum) -> Result<usize> {
        use cudarc::driver::sys;
        self.context.bind_to_thread().w()?;
        let dev = self.context.cu_device();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let mut value: u64 = 0;
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, dev)
                .result()
                .w()?;
            sys::cuMemPoolGetAttribute(pool, attr, &mut value as *mut u64 as *mut std::ffi::c_void)
                .result()
                .w()?;
        }
        Ok(value as usize)
    }

    /// Release reserved-but-free pool memory back to the OS, keeping at least
    /// `keep_bytes` reserved.
    ///
    /// One caller: the startup balloon, which allocates pool tensors to measure
    /// resident capacity `C` and must hand those bytes back before the model
    /// loads into them — the async pool would otherwise retain them and the
    /// post-balloon measurement would read them as still in use.
    ///
    /// It is not a runtime reclaim path. It was once called from the governor's
    /// relief hook and after every scheduler pressure episode, when KV lived in
    /// this pool and its freed arenas were worth returning; KV is regions now
    /// and what remains here reaches its size and stays. `cuMemPoolTrimTo`
    /// **synchronously unmaps** — not stream-ordered — so a caller must be sure
    /// no kernel holds a pointer into the freed blocks. At startup, before any
    /// kernel runs, that is trivially true. Anywhere else it is a hazard, which
    /// is why the runtime callers are gone rather than guarded.
    ///
    /// Only trims memory nothing is using — never touches live allocations.
    /// Errors if the device doesn't use the async pool allocator.
    pub fn trim_pool(&self, keep_bytes: usize) -> Result<()> {
        use cudarc::driver::sys;
        self.context.bind_to_thread().w()?;
        let dev = self.context.cu_device();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, dev)
                .result()
                .w()?;
            sys::cuMemPoolTrimTo(pool, keep_bytes).result().w()?;
        }
        Ok(())
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        // Latch how much host RAM the machine had before this process took any
        // of it. The expert cache's warm tier is sized from that reading rather
        // than from a live one, because by the time it asks, the loader has the
        // checkpoint mapped and the live figure is several GiB into a trough of
        // the engine's own digging (see `vram::launch_available_ram`). A device
        // has to exist before any weight can be loaded onto it, so this is the
        // earliest point every path that can reach a warm tier shares.
        crate::vram::snapshot_launch();

        // **One device per ordinal, for the life of the process.**
        //
        // Not for the context's sake — `CudaContext::new` retains the driver's
        // *primary* context, so every call for an ordinal already shares one. It
        // is the memoised state hanging off the handle: the compiled modules and
        // the two upload caches below, whose entries are `Uploaded`, holding a
        // `ManuallyDrop<CudaSlice>`. A device that built some and was dropped
        // does not give that memory back, so a process that asks often enough
        // runs the card down until something cannot allocate — surfacing as
        // `CUBLAS_STATUS_NOT_INITIALIZED` from whichever request is unlucky,
        // which reads as "CUDA is broken" rather than "you have made too many of
        // these". `candle-nn`'s GPU tests hit it at around 220 devices: three
        // failed only when run after the others, passed in isolation, and passed
        // under every subset. Callers are right to treat a device as a value —
        // helpers take `&Device`, tests build one per case — so the cheapness
        // has to be true rather than assumed.
        //
        // The handle's two *stateful* resources are deliberately not shared; see
        // the cache-hit path below.
        //
        // Indexed by ordinal rather than held in a map behind a lock: the cache
        // is read on any thread that touches the GPU, and there is nothing to
        // serialise — a slot is written once and read forever after. Reaching it
        // is an atomic load, not a mutex acquisition. This mirrors
        // `gpu_memory::DEVICE_INIT_FREE`, which indexes the same ordinals the
        // same way.
        //
        // `cuda_device_reuse.rs` is the regression test.
        static DEVICES: [OnceLock<CudaDevice>; MAX_CUDA_DEVICES] =
            [const { OnceLock::new() }; MAX_CUDA_DEVICES];
        let Some(slot) = DEVICES.get(ordinal) else {
            crate::bail!(
                "CUDA ordinal {ordinal} is beyond the {MAX_CUDA_DEVICES} this build indexes \
                 per-device state for. Raise `MAX_CUDA_DEVICES` (and \
                 `gpu_memory::MAX_TRACKED_GPUS`, which tracks the same ordinals) together — \
                 there is deliberately no uncached path, because an uncached device exhausts \
                 the driver's cuBLAS handles."
            )
        };
        if let Some(dev) = slot.get() {
            // **A context is current per THREAD, not per process.** Building one
            // bound it to whichever thread built it; handing that same context to
            // a second thread without binding leaves every driver call there
            // failing `CUDA_ERROR_INVALID_CONTEXT`. Constructing per call hid
            // this, because the construction did the binding.
            dev.context.bind_to_thread().w()?;
            let mut dev = dev.clone();
            // Everything memoised is shared; everything stateful is rebuilt.
            dev.give_own_stateful()?;
            return Ok(dev);
        }

        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        Self::validate_compute_capability(&context)?;
        Self::disable_per_arg_event_tracking(&context);
        let stream = context.default_stream();
        let (blas, blas_workspace) = new_blas(&stream)?;
        let curand = cudarc::curand::CudaRng::new(DEFAULT_SEED, stream.clone()).w()?;
        let dev = Self {
            id: DeviceId::new(),
            context,
            stream,
            blas,
            blas_workspace,
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            info_tables: Arc::new(Mutex::new(InfoRing::default())),
            perm_tables: Arc::new(Mutex::new(HashMap::new())),
            staging: Arc::new(Mutex::new(None)),
            synced_staging: Arc::new(Mutex::new(None)),
            capture: Arc::new(CaptureHub::default()),
        };
        // Record free VRAM now, before any model weights load, so the KV budget
        // gate can estimate our resident footprint and credit pageable memory
        // the OS evicts for us (see `gpu_memory::device_init_free`). Inside the
        // build rather than on every call: it is a first-touch reading, and a
        // cache hit happens long after weights have loaded.
        if let Ok((free, _total)) = dev.mem_get_info() {
            crate::gpu_memory::note_device_init_free(ordinal, free);
        }
        // Two threads racing the first call for an ordinal both build one; the
        // slot takes whichever arrives first and the other is dropped here,
        // which is why the winner is read back rather than `dev` returned. The
        // loser costs one device's worth of handles, once, at first touch —
        // where the leak this exists to stop is one per call, forever.
        //
        // **The loser still needs its own stateful parts.** Returning the
        // winner's clone unmodified would hand two threads one cuBLAS handle and
        // one curand generator — the exact sharing the cache-hit path above
        // rebuilds to avoid, and the exact conditions (a parallel test suite at
        // first touch) under which it was originally observed.
        let mut dev = slot.get_or_init(|| dev).clone();
        dev.give_own_stateful()?;
        Ok(dev)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
        Ok(())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 | DType::F8E4M3 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            let layout = Layout::contiguous(shape);
            // `Backing::Owned` in: the source is this function's own fresh
            // allocation, so there is no arena to inherit and the rescale's
            // output is owned like everything else here.
            super::run_affine_ffi(&slice, self, &layout, up - lo, lo, Backing::Owned)?.0
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 | DType::F8E4M3 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.memcpy_stod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.memcpy_stod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn synchronize(&self) -> Result<()> {
        // `.w()` (not `Error::wrap`) so a sticky fault or a sustained
        // out-of-memory streak on THIS call is visible to `gpu_poison` —
        // `Error::wrap`'s generic `Display` wrapping bypasses that detection
        // entirely. This call is the daemon's most frequent, unconditional
        // device round-trip (the persistence thread's hot→warm sync runs on a
        // fixed cadence regardless of load), which is exactly why a poisoned
        // context showed up here as an endless identical retry with nothing
        // ever noticing.
        let _eager = self.pause_capture()?;
        self.stream.synchronize().w()?;
        Ok(())
    }
}

impl CudaDevice {
    /// [`BackendDevice::storage_from_cpu_storage_owned`], placed on the span
    /// `origin` names instead of allocated from the driver.
    ///
    /// A host-built table has no device operand to inherit a ticket from, so the
    /// caller states which wave it belongs to. `Backing::Owned` reproduces the
    /// original behaviour exactly, which is what every load-time caller wants.
    pub fn storage_from_cpu_storage_owned_on(
        &self,
        storage: CpuStorage,
        origin: Backing,
    ) -> Result<CudaStorage> {
        macro_rules! up {
            ($s:expr, $variant:ident) => {{
                let (data, backing) = self.memcpy_stod_leased(&$s, origin)?;
                (CudaStorageSlice::$variant(data), backing)
            }};
        }
        let (slice, backing) = match storage {
            CpuStorage::U8(s) => up!(s, U8),
            CpuStorage::U32(s) => up!(s, U32),
            CpuStorage::I64(s) => up!(s, I64),
            CpuStorage::BF16(s) => up!(s, BF16),
            CpuStorage::F16(s) => up!(s, F16),
            CpuStorage::F32(s) => up!(s, F32),
            CpuStorage::F64(s) => up!(s, F64),
            CpuStorage::F8E4M3(s) => up!(s, F8E4M3),
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing,
            anchor: None,
        })
    }

    /// Uninitialised storage taken from the arena `ticket` names, or from the
    /// pool when there is none.
    ///
    /// The dtype dispatch goes through [`crate::cuda_backend::alloc_inheriting`]
    /// so the buffer and the `Backing` stamped on it are resolved together —
    /// naming them separately is what lets a wave range end up marked `Owned`.
    ///
    /// # Safety
    ///
    /// The returned storage is uninitialised; the caller must write it before
    /// anything reads it, exactly as for `BackendDevice::alloc_uninit`.
    pub(crate) unsafe fn alloc_uninit_from(
        &self,
        shape: &Shape,
        dtype: DType,
        ticket: Option<crate::wave_provenance::WaveTicket>,
    ) -> Result<CudaStorage> {
        let from = match ticket {
            Some(t) => Backing::Lease(LeaseOrigin::Wave(t)),
            None => Backing::Owned,
        };
        let elem_count = shape.elem_count();
        macro_rules! arm {
            ($ty:ty, $variant:ident) => {{
                let (data, backing) = alloc_inheriting::<$ty>(self, elem_count, from)?;
                (CudaStorageSlice::$variant(data), backing)
            }};
        }
        let (slice, backing) = match dtype {
            DType::U8 => arm!(u8, U8),
            DType::U32 => arm!(u32, U32),
            DType::I64 => arm!(i64, I64),
            DType::BF16 => arm!(bf16, BF16),
            DType::F16 => arm!(f16, F16),
            DType::F32 => arm!(f32, F32),
            DType::F64 => arm!(f64, F64),
            DType::F8E4M3 => arm!(F8E4M3, F8E4M3),
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
            backing,
            anchor: None,
        })
    }
}
