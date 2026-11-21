//! The CUDA backend's device handle and the constants that identify and bound
//! its devices.
//!
//! `CudaDevice` is the handle a CUDA tensor carries, and every handle for an
//! ordinal shares one set of memoised caches and staging buffers. The seed every
//! device's random generator starts from is fixed here, so a freshly built
//! device draws the same numbers as the last one and a test can assert against
//! literal values. The ordinal bound is kept in step with the table that records
//! free-memory readings per device, because a device handed out without a
//! reading would silently lose the budget gate its baseline.
//!
//! - `backend_device`: the handle, its memoised state and its allocations.
//! - `device_info`: properties read once and the checks on the card.
//! - `memory`: allocation, copies and staging scratch.
//! - `stream`: the device stream and the wave capture.

use super::thread_blas::{self, BlasHandle, BLAS_WORKSPACE_BYTES};
use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use cudarc;
use cudarc::cublas::sys::cublasStatus_t;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, HostSlice,
    ValidAsZeroBits,
};
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::{Arc, Mutex, OnceLock};

use super::driver_alloc::{at_site, install as install_driver_alloc_watch};
use super::graph::{enter_wave, without_recording, CaptureHub, CaptureStats, Paused, WaveCapture};
use super::held::{release_on_its_stream, Held, Release};
use super::info_ring::InfoRing;
use super::kernel_scratch;
use super::module_loading;
use super::staged::{pack_segments, SEGMENT_ALIGN};
use super::stream_owner;
use super::sync_watch;
use super::{CudaError, CudaStorage, CudaStorageSlice, StreamLease, WrapErr};
use crate::cuda_backend::{alloc_inheriting, Backing, LeaseAnchor, INHERIT_ALIGN};
use crate::forbidden_alloc;
use crate::readonly_regions::forbid_write;
use crate::wave_provenance::{wave_alloc_attributed, LeaseOrigin};
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
    /// The same scratch for every thread that is not recording a wave — see
    /// [`Self::with_staging`]. Recorded segments never touch it, so a segment
    /// launched between another thread's copy and its kernel cannot overwrite
    /// what that kernel reads.
    foreign_staging: Staging,
    /// Launch tables for work that synchronises its stream before it returns
    /// — see [`Self::with_synced_upload`]. Separate from `staging` so a launch
    /// that waits out a long copy never holds the forward's scratch.
    synced_staging: Staging,
    /// The wave capture shared by every clone of this handle — see
    /// [`Self::begin_wave_capture`]. Per stream, like the caches above.
    capture: Arc<CaptureHub>,
}

/// One grow-only byte buffer, reused by every call that stages through it —
/// held, so it stands on the reservation's held ground once there is one.
type Staging = Arc<Mutex<Option<Held<u8>>>>;

/// Memoized layout-table uploads, keyed by the table's contents.
type InfoTables = Arc<Mutex<InfoRing>>;

/// Memoized gather permutations, keyed by `(rows, groups)`.
type PermTables = Arc<Mutex<HashMap<(usize, usize), Arc<Uploaded<u32>>>>>;

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
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
    _anchor: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl<T> Uploaded<T> {
    /// A leased view over memory `anchor` (when given) keeps alive.
    pub(crate) fn leased(
        slice: CudaSlice<T>,
        anchor: Option<Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Self {
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
            // Freed on its stream through the device that owns it, behind any
            // recorded launch that may still read it (`held::release_on_its_stream`).
            //
            // SAFETY: `slice` is live and taken exactly once, here.
            Backing::Owned => {
                let slice = unsafe { std::mem::ManuallyDrop::take(&mut self.slice) };
                let stream = slice.stream().clone();
                let ptr = slice.leak();
                release_on_its_stream(Release::Pool { ptr, stream });
            }
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

/// A loaded kernel and the lease its launches go out on: take one where the
/// launch is built, and drop it once the launch is issued.
pub struct CudaFunc {
    func: CudaFunction,
    stream: StreamLease,
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

mod backend_device;
mod device_info;
mod memory;
mod stream;
