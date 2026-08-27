//! `BackendStorage` implementation for the CUDA backend (`feature = "cuda"`),
//! the production inference path for this fork.
//!
//! `CudaStorage` wraps a `CudaStorageSlice` (one `CudaSlice<T>` variant per
//! dtype) plus a `CudaDevice`; `BackendDevice for CudaDevice` lives in the
//! sibling `device.rs`. Generic tensor ops (elementwise, affine, reduce,
//! indexing, conv, matmul via `cublas`) are each a small unit struct
//! implementing `Map1`/`Map2`/`Map1Any`/`Map2InPlace`/`Map2Any`, which launch
//! the AOT-compiled PTX kernels re-exported here as `kernels`
//! (`candle_kernels::simple::*`) — PTX is embedded at compile time by
//! `candle-kernels/build.rs`, so no NVCC is needed at runtime. Quantized
//! matmul dispatches separately from `candle-core/src/quantized/cuda.rs`, and
//! the paged-decode/paged-prefill/paged-glue/provenance kernels backing the
//! three-tier KV cache are called directly from `candle-transformers` and
//! `candle-nn::kv_cache`, bypassing this generic dispatch entirely.
use crate::backend::{BackendDevice, BackendStorage};
use crate::forbidden_alloc;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
pub use crate::wave_provenance;
use crate::wave_provenance::{LeaseOrigin, WaveTicket};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
pub use candle_kernels as kernels;
pub use cudarc;

// ── Kernel breadcrumb ─────────────────────────────────────────────────────────
//
// Written immediately before every kernel FFI call (one thread-local store,
// ~1 ns) — HOST-SIDE, before the launch, so the kernels themselves are never
// touched and there is zero kernel-side perf cost.
//
// CUDA errors are ASYNCHRONOUS: a faulting kernel surfaces its error at a LATER
// synchronization point (the next launch's error check, a device sync, a memcpy)
// — often on a DIFFERENT wrapper, sometimes a different thread. A single
// "last kernel" slot therefore names the wrong kernel much of the time. We keep
// a short RING of recent launches per thread instead, so the error hook can dump
// the recent history and the true culprit is visible even when the point of
// detection drifts a few launches past the point of fault.
//
// For EXACT attribution during a debug session, run with `CUDA_LAUNCH_BLOCKING=1`
// (driver-level: every launch synchronizes, so the error surfaces AT the faulting
// launch and breadcrumb #0 is the culprit), and/or `compute-sanitizer` — the
// kernels are built with `--generate-line-info` (see `candle-kernels/build_utils`),
// so the sanitizer reports the exact `.cu` file and line of the out-of-bounds
// access at no runtime cost.
//
// Usage: call `cuda_breadcrumb!("run_foo")` at the top of each wrapper that
// dispatches to a kernel.  The macro captures `file!()` / `line!()` at the
// actual call site so the recorded location is meaningful.

/// How many recent launches to keep per thread. Small — the fault is nearly
/// always within the last handful of launches ahead of the detection point.
const KERNEL_RING_LEN: usize = 16;

thread_local! {
    static KERNEL_RING: std::cell::RefCell<[(&'static str, &'static str, u32); KERNEL_RING_LEN]> =
        const { std::cell::RefCell::new([("", "", 0); KERNEL_RING_LEN]) };
    /// Index of the NEXT slot to write (ring is `[pos-1, pos-2, …]` newest→oldest).
    static KERNEL_RING_POS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Record a breadcrumb for the current thread's latest kernel launch.
/// Called via the `cuda_breadcrumb!` macro before each kernel FFI call.
#[inline(always)]
pub fn set_kernel_breadcrumb(name: &'static str, file: &'static str, line: u32) {
    let pos = KERNEL_RING_POS.with(|p| {
        let i = p.get();
        p.set((i + 1) % KERNEL_RING_LEN);
        i
    });
    KERNEL_RING.with(|r| r.borrow_mut()[pos] = (name, file, line));
}

/// Dump the recent kernel-launch breadcrumbs on this thread, newest first.
/// Because CUDA errors are asynchronous, the faulting kernel is usually `#0`
/// but may be a few entries back — the history is what makes that visible.
/// Read from the panic hook / CUDA error sites when a `DriverError` surfaces.
pub fn last_cuda_kernel_launch() -> String {
    let pos = KERNEL_RING_POS.with(|p| p.get());
    KERNEL_RING.with(|r| {
        let ring = r.borrow();
        let mut out = Vec::new();
        for k in 0..KERNEL_RING_LEN {
            let i = (pos + KERNEL_RING_LEN - 1 - k) % KERNEL_RING_LEN;
            let (name, file, line) = ring[i];
            if name.is_empty() {
                continue;
            }
            out.push(format!("    #{k} '{name}' ({file}:{line})"));
        }
        if out.is_empty() {
            "(no kernels recorded on this thread)".to_string()
        } else {
            format!(
                "recent CUDA kernel launches on this thread, newest first — async \
                 error, fault is usually #0 but can be a few back:\n{}",
                out.join("\n")
            )
        }
    })
}

/// Set the thread-local kernel breadcrumb.  Captures `file!()` / `line!()` at
/// the macro call site so the recorded location points to the wrapper, not to
/// this utility module.
macro_rules! cuda_breadcrumb {
    ($name:expr) => {
        set_kernel_breadcrumb($name, file!(), line!())
    };
}
// ─────────────────────────────────────────────────────────────────────────────
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, PushKernelArg, ValidAsZeroBits};
use float8::F8E4M3;
use half::{bf16, f16};
use std::sync::Arc;

#[cfg(feature = "cudnn")]
pub mod cudnn;
mod device;
mod error;
mod utils;
pub use device::{CudaDevice, DeviceId, Uploaded};
pub use error::{CudaError, WrapErr};
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Out, OutS, S};

pub enum SlicePtrOrNull<T> {
    Ptr(Uploaded<T>),
    /// A cache-shared table ([`CudaDevice::info_table`]) — same launch-arg shape as
    /// `Ptr`, but the buffer is owned by the device's info-table cache and shared
    /// across launches, so there is nothing to free here.
    Shared(Arc<Uploaded<T>>),
    Null,
}

impl<T: DeviceRepr> SlicePtrOrNull<T> {
    pub fn builder_arg<'a, 'b: 'a>(&'b self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        match self {
            SlicePtrOrNull::Ptr(slice) => builder.arg(&**slice),
            SlicePtrOrNull::Shared(slice) => builder.arg(&***slice),
            SlicePtrOrNull::Null => builder.arg(&0usize),
        };
    }
}

impl crate::scalar::Scalar {
    pub fn builder_arg<'a, 'b: 'a>(&'b self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        use crate::scalar::Scalar;
        match self {
            Scalar::U8(v) => builder.arg(v),
            Scalar::U32(v) => builder.arg(v),
            Scalar::I64(v) => builder.arg(v),
            Scalar::F32(v) => builder.arg(v),
            Scalar::F64(v) => builder.arg(v),
            Scalar::F16(v) => builder.arg(v),
            Scalar::BF16(v) => builder.arg(v),
            Scalar::F8E4M3(v) => builder.arg(v),
        };
    }
}

impl SlicePtrOrNull<usize> {
    /// The dims/stride blob a strided kernel reads, served from the device's
    /// memoized info-table cache ([`CudaDevice::info_table`]): layouts repeat
    /// across launches, so the steady state is a cache hit with no upload and
    /// no allocation at all.
    pub fn params_from_layout(dev: &CudaDevice, l: &Layout) -> Result<Self> {
        let ds = if l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            SlicePtrOrNull::Shared(dev.info_table(&[l.dims(), l.stride()].concat())?)
        };
        Ok(ds)
    }
}

#[derive(Debug)]
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I64(CudaSlice<i64>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    F8E4M3(CudaSlice<F8E4M3>),
    /// Tombstone left behind when a leased storage's drop moves its slice out.
    ///
    /// `CudaSlice::leak` takes `self` by value, so it is unreachable from
    /// `Drop::drop(&mut self)` without moving the slice out first — and moving
    /// out of a type that implements `Drop` requires putting something back.
    /// This variant is that something. It is never observable: it exists only
    /// between the `mem::replace` and the end of `drop`.
    ///
    /// **Named for the hole, not for ownership.** It was `Empty`, which reads as
    /// "a slice of zero elements" and invites the guess that it means "borrowed"
    /// or "not owned" — neither is true. Non-ownership is [`Backing::Lease`]'s
    /// job and is orthogonal to this. A `Moved` slice is the absence left by the
    /// move, and the only correct thing to do with one is diverge.
    ///
    /// The tombstone survives rather than becoming `ManuallyDrop<CudaStorageSlice>`
    /// (the shape `QCudaStorage` uses) because `slice` is a **public** field with
    /// ~330 uses in this crate and ~130 more outside it; the wrapper would put a
    /// deref at every one of them to delete a variant that is unreachable by
    /// construction. `QCudaStorage`'s field is private with a handful of uses,
    /// which is why the same fix was proportionate there and is not here.
    Moved,
}

/// Who owns the device memory behind a [`CudaStorage`].
///
/// A pool allocation is freed on drop, as always. A **leased** one is an offset
/// into memory this process claimed once and never returns — letting
/// `CudaSlice::drop` reach `cuMemFreeAsync` on it would be an error, not a
/// leak. See `docs/archived/arena_unification.md` §3.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    /// Allocated from the stream-ordered pool; freed on drop.
    Owned,
    /// A view over memory owned elsewhere; drop releases the view, never the
    /// memory. The [`LeaseOrigin`] says *whose* memory, which is what lets an
    /// op allocate its output from the same arena its operand came from.
    Lease(LeaseOrigin),
}

impl Backing {
    /// The wave arena an op reading this storage should allocate its output
    /// from, or `None` to allocate from the pool.
    ///
    /// `None` for owned storage and for foreign leases alike: neither names an
    /// arena that is a scratch space to carve from.
    pub fn inherit_ticket(&self) -> Option<WaveTicket> {
        match self {
            Self::Owned => None,
            Self::Lease(origin) => origin.ticket(),
        }
    }

    /// The backing an allocation seeded by a bare ticket should inherit.
    ///
    /// For the call sites that hold a coordinate rather than an operand — a host
    /// upload, or work that reached another thread over a channel. Absence is
    /// real state: no ticket means no arena, and the allocation is an ordinary
    /// owned one.
    pub fn from_ticket(ticket: Option<WaveTicket>) -> Self {
        match ticket {
            Some(t) => Self::Lease(LeaseOrigin::Wave(t)),
            None => Self::Owned,
        }
    }
}

impl CudaStorageSlice {
    /// Diverge on the tombstone variant.
    ///
    /// [`CudaStorageSlice::Moved`] exists only between the `mem::replace` and
    /// the end of a leased [`CudaStorage`]'s `drop`, and nothing else can
    /// observe one — so every other match over the slice ends here. Returning
    /// `!` lets one arm serve matches of every result type.
    #[cold]
    #[inline(never)]
    pub fn unreachable_moved() -> ! {
        unreachable!("CudaStorageSlice::Moved escaped a leased storage's drop")
    }

    /// Size of the backing allocation in bytes.
    ///
    /// The element count times the element width, so it is comparable across
    /// dtypes — which is what makes it the right unit for allocation
    /// accounting ([`crate::forbidden_alloc`]) and for budget arithmetic.
    pub fn byte_len(&self) -> usize {
        match self {
            CudaStorageSlice::U8(s) => s.len() * std::mem::size_of::<u8>(),
            CudaStorageSlice::U32(s) => s.len() * std::mem::size_of::<u32>(),
            CudaStorageSlice::I64(s) => s.len() * std::mem::size_of::<i64>(),
            CudaStorageSlice::BF16(s) => s.len() * std::mem::size_of::<bf16>(),
            CudaStorageSlice::F16(s) => s.len() * std::mem::size_of::<f16>(),
            CudaStorageSlice::F32(s) => s.len() * std::mem::size_of::<f32>(),
            CudaStorageSlice::F64(s) => s.len() * std::mem::size_of::<f64>(),
            CudaStorageSlice::F8E4M3(s) => s.len() * std::mem::size_of::<F8E4M3>(),
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    /// Device address of the first element, whatever the element type.
    ///
    /// The read-only twin of [`Self::device_ptr_mut`], and it carries the same
    /// caveat: no guard is returned, so the caller keeps the storage alive for
    /// as long as the address is used.
    pub fn device_ptr(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> u64 {
        use cudarc::driver::DevicePtr;
        macro_rules! addr {
            ($s:expr) => {{
                let (ptr, _guard) = $s.device_ptr(stream);
                ptr
            }};
        }
        match self {
            CudaStorageSlice::U8(s) => addr!(s),
            CudaStorageSlice::U32(s) => addr!(s),
            CudaStorageSlice::I64(s) => addr!(s),
            CudaStorageSlice::BF16(s) => addr!(s),
            CudaStorageSlice::F16(s) => addr!(s),
            CudaStorageSlice::F32(s) => addr!(s),
            CudaStorageSlice::F64(s) => addr!(s),
            CudaStorageSlice::F8E4M3(s) => addr!(s),
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    /// Get a mutable device pointer for in-place operations.
    /// Returns the raw pointer that can be passed to FFI functions.
    ///
    /// NOTE: This does NOT return a guard - the caller must ensure proper synchronization.
    pub fn device_ptr_mut(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<*mut std::ffi::c_void> {
        use cudarc::driver::DevicePtrMut;
        match self {
            CudaStorageSlice::U8(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::U32(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::I64(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::BF16(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F16(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F32(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F64(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F8E4M3(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }
}

struct Clone;
impl Map1 for Clone {
    fn f<T: DeviceRepr>(
        &self,
        s: &CudaSlice<T>,
        _: &CudaDevice,
        _: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        // `CudaSlice::try_clone` allocates its destination, so it reaches the
        // driver without passing through `CudaDevice::alloc`.
        forbidden_alloc::record("CudaSlice::try_clone", s.len() * std::mem::size_of::<T>());
        // `try_clone` allocates its own destination from the pool, so the
        // copy owns its memory regardless of where the source came from.
        let _ = origin;
        Ok((s.try_clone().w()?, Backing::Owned))
    }
}

pub fn kernel_name<T: WithDType>(root: &str) -> String {
    let dtype = T::DTYPE.as_str();
    format!("{root}_{dtype}")
}

/// Convert candle DType to AffineDType for FFI dispatcher
fn dtype_to_affine_dtype(dtype: DType) -> i32 {
    use kernels::simple::affine::AffineDType;
    match dtype {
        DType::F32 => AffineDType::F32 as i32,
        DType::F64 => AffineDType::F64 as i32,
        DType::F16 => AffineDType::F16 as i32,
        DType::BF16 => AffineDType::BF16 as i32,
        DType::F8E4M3 => AffineDType::F8E4M3 as i32,
        DType::U8 => AffineDType::U8 as i32,
        DType::U32 => AffineDType::U32 as i32,
        DType::I64 => AffineDType::I64 as i32,
    }
}

/// Execute affine transformation via direct FFI call (no PTX JIT)
/// `inherit` is the operand's arena; the resolved backing comes back alongside
/// the slice so the caller can stamp it rather than claiming ownership of what
/// may be a view into a wave span.
fn run_affine_ffi(
    src: &CudaStorageSlice,
    dev: &CudaDevice,
    layout: &Layout,
    mul: f64,
    add: f64,
    inherit: Backing,
) -> Result<(CudaStorageSlice, Backing)> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el = shape.elem_count();
    let start_offset = layout.start_offset();
    let stream = dev.cuda_stream();

    // Prepare dims/strides info for non-contiguous tensors
    let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
        None
    } else {
        Some(dev.info_table(&[dims, layout.stride()].concat())?)
    };
    let info_ptr = match &info {
        Some(s) => {
            let (ptr, _guard) = s.device_ptr(&stream);
            ptr as *const usize
        }
        None => std::ptr::null(),
    };

    // Get dtype for dispatcher
    let dtype = match src {
        CudaStorageSlice::F32(_) => DType::F32,
        CudaStorageSlice::F64(_) => DType::F64,
        CudaStorageSlice::F16(_) => DType::F16,
        CudaStorageSlice::BF16(_) => DType::BF16,
        CudaStorageSlice::F8E4M3(_) => DType::F8E4M3,
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        CudaStorageSlice::U8(_) => DType::U8,
        CudaStorageSlice::U32(_) => DType::U32,
        CudaStorageSlice::I64(_) => DType::I64,
    };
    let dtype_i32 = dtype_to_affine_dtype(dtype);

    // Assigned by the macro to whatever `alloc_inheriting` resolved.
    let out_backing;

    // Execute based on dtype - allocate output and call FFI
    macro_rules! affine_impl {
        ($slice:expr, $dtype_variant:ident) => {{
            let src_slice = $slice.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let (out, resolved) = unsafe { alloc_inheriting(dev, el, inherit)? };
            out_backing = resolved;
            {
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);
                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));
                cuda_breadcrumb!("run_affine");
                unsafe {
                    kernels::simple::affine::run_affine(
                        dtype_i32,
                        el,
                        dims.len(),
                        info_ptr,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        mul,
                        add,
                    );
                }
            }
            // Guards dropped, safe to move out
            CudaStorageSlice::$dtype_variant(out)
        }};
    }

    let out = match src {
        CudaStorageSlice::F32(s) => affine_impl!(s, F32),
        CudaStorageSlice::F64(s) => affine_impl!(s, F64),
        CudaStorageSlice::F16(s) => affine_impl!(s, F16),
        CudaStorageSlice::BF16(s) => affine_impl!(s, BF16),
        CudaStorageSlice::F8E4M3(s) => affine_impl!(s, F8E4M3),
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        CudaStorageSlice::U8(s) => affine_impl!(s, U8),
        CudaStorageSlice::U32(s) => affine_impl!(s, U32),
        CudaStorageSlice::I64(s) => affine_impl!(s, I64),
    };

    Ok((out, out_backing))
}

/// Execute parametric unary operation (Elu/Powf) via direct FFI call
fn run_unary_param_ffi(
    src: &CudaStorageSlice,
    dev: &CudaDevice,
    layout: &Layout,
    op: i32,
    param: f64,
) -> Result<CudaStorageSlice> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el = shape.elem_count();
    let start_offset = layout.start_offset();
    let stream = dev.cuda_stream();

    // Prepare dims/strides info for non-contiguous tensors
    let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
        None
    } else {
        Some(dev.info_table(&[dims, layout.stride()].concat())?)
    };
    let info_ptr = match &info {
        Some(s) => {
            let (ptr, _guard) = s.device_ptr(&stream);
            ptr as *const usize
        }
        None => std::ptr::null(),
    };

    // Get dtype for dispatcher
    let dtype = match src {
        CudaStorageSlice::F32(_) => DType::F32,
        CudaStorageSlice::F64(_) => DType::F64,
        CudaStorageSlice::F16(_) => DType::F16,
        CudaStorageSlice::BF16(_) => DType::BF16,
        CudaStorageSlice::F8E4M3(_) => DType::F8E4M3,
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        CudaStorageSlice::U8(_) => DType::U8,
        CudaStorageSlice::U32(_) => DType::U32,
        CudaStorageSlice::I64(_) => DType::I64,
    };
    let dtype_i32 = dtype_to_unary_dtype(dtype);

    // Execute based on dtype - allocate output and call FFI
    macro_rules! unary_param_impl {
        ($slice:expr, $dtype_variant:ident) => {{
            let src_slice = $slice.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let out = unsafe { dev.alloc(el)? };
            {
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);
                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));
                cuda_breadcrumb!("run_unary_param_op");
                unsafe {
                    kernels::simple::unary::run_unary_param_op(
                        op,
                        dtype_i32,
                        param as f32,
                        el,
                        dims.len(),
                        info_ptr,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                    );
                }
            }
            // Guards dropped, safe to move out
            CudaStorageSlice::$dtype_variant(out)
        }};
    }

    let out = match src {
        CudaStorageSlice::F32(s) => unary_param_impl!(s, F32),
        CudaStorageSlice::F64(s) => unary_param_impl!(s, F64),
        CudaStorageSlice::F16(s) => unary_param_impl!(s, F16),
        CudaStorageSlice::BF16(s) => unary_param_impl!(s, BF16),
        CudaStorageSlice::F8E4M3(s) => unary_param_impl!(s, F8E4M3),
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        _ => crate::bail!("Parametric unary ops only support float types"),
    };

    Ok(out)
}

#[allow(unused)]
struct Im2Col1D {
    l_k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Im2Col1D {
    #[allow(unused)]
    fn l_out(&self, l: usize) -> usize {
        (l + 2 * self.padding - self.dilation * (self.l_k - 1) - 1) / self.stride + 1
    }
}

impl Map1 for Im2Col1D {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_im2col1d");
        let shape = layout.shape();
        let dims = shape.dims();
        let l_out = self.l_out(dims[2]);
        let threads = dims[0] * l_out * dims[1];
        let ds = dev.info_table(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("im2col1d not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        // SAFETY: Set later by running the kernel.
        let (dst, out_backing) = unsafe { alloc_inheriting::<T>(dev, threads * self.l_k, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_im2col1d(
                    dtype,
                    threads * self.l_k, // dst_numel
                    l_out,
                    self.l_k,
                    self.stride,
                    self.padding,
                    self.dilation,
                    ds_ptr as *const usize,
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((dst, out_backing))
    }
}

#[allow(unused)]
struct Im2Col {
    h_k: usize,
    w_k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Im2Col {
    #[allow(unused)]
    fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl Map1 for Im2Col {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_im2col");
        let shape = layout.shape();
        let dims = shape.dims();
        let (h_out, w_out) = self.hw_out(dims[2], dims[3]);
        let dst_el = dims[0] * h_out * w_out * dims[1] * self.h_k * self.w_k;
        let ds = dev.info_table(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("im2col not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        // SAFETY: Set later by running the kernel.
        let (dst, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_im2col(
                    dtype,
                    dst_el,
                    h_out,
                    w_out,
                    self.h_k,
                    self.w_k,
                    self.stride,
                    self.padding,
                    self.dilation,
                    ds_ptr as *const usize,
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((dst, out_backing))
    }
}

struct FastReduce<'a>(&'a [usize], ReduceOp);
impl Map1Any for FastReduce<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        wrap: W,
        origin: Backing,
    ) -> Result<OutS> {
        cuda_breadcrumb!("run_fast_reduce");
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;

        let (check_empty, return_index) = match self.1 {
            ReduceOp::Sum => (false, false),
            ReduceOp::Min => (true, false),
            ReduceOp::Max => (true, false),
            ReduceOp::ArgMin => (true, true),
            ReduceOp::ArgMax => (true, true),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }

        // Get dtype for FFI dispatcher
        let dtype = T::DTYPE;
        let dtype_i32 = dtype_to_fast_reduce_dtype(dtype);

        let stream = dev.cuda_stream();
        let ds = dev.info_table(&[dims.as_slice(), stride.as_slice()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        if return_index {
            use kernels::simple::reduce::FastArgReduceOp;
            let op = match self.1 {
                ReduceOp::ArgMin => FastArgReduceOp::ArgMin as i32,
                ReduceOp::ArgMax => FastArgReduceOp::ArgMax as i32,
                _ => unreachable!(),
            };
            // SAFETY: filled in by the follow up kernel.
            let (out, out_backing) = unsafe { alloc_inheriting::<u32>(dev, dst_el, origin)? };
            {
                let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
                let (src_ptr, _src_guard) = src.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                unsafe {
                    kernels::simple::reduce::run_fast_arg_reduce_op(
                        op,
                        dtype_i32,
                        src_el,
                        el_to_sum_per_block,
                        src_dims.len(),
                        ds_ptr as *const usize,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut u32,
                    );
                }
            }
            Ok((S::U32(out), out_backing))
        } else {
            use kernels::simple::reduce::FastReduceOp;
            let op = match self.1 {
                ReduceOp::Sum => FastReduceOp::Sum as i32,
                ReduceOp::Min => FastReduceOp::Min as i32,
                ReduceOp::Max => FastReduceOp::Max as i32,
                _ => unreachable!(),
            };
            // SAFETY: filled in by the follow up kernel.
            let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
            {
                let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
                let (src_ptr, _src_guard) = src.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                unsafe {
                    kernels::simple::reduce::run_fast_reduce_op(
                        op,
                        dtype_i32,
                        src_el,
                        el_to_sum_per_block,
                        src_dims.len(),
                        ds_ptr as *const usize,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                    );
                }
            }
            Ok((wrap(out), out_backing))
        }
    }
}

/// Convert candle DType to UnaryDType for FFI dispatcher
fn dtype_to_unary_dtype(dtype: DType) -> i32 {
    use kernels::simple::unary::UnaryDType;
    match dtype {
        DType::F32 => UnaryDType::F32 as i32,
        DType::F64 => UnaryDType::F64 as i32,
        DType::F16 => UnaryDType::F16 as i32,
        DType::BF16 => UnaryDType::BF16 as i32,
        DType::F8E4M3 => UnaryDType::F8E4M3 as i32,
        DType::U8 => UnaryDType::U8 as i32,
        DType::U32 => UnaryDType::U32 as i32,
        DType::I64 => UnaryDType::I64 as i32,
    }
}

/// Convert candle DType to FastReduceDType for FFI dispatcher
fn dtype_to_fast_reduce_dtype(dtype: DType) -> i32 {
    use kernels::simple::reduce::FastReduceDType;
    match dtype {
        DType::F32 => FastReduceDType::F32 as i32,
        DType::F64 => FastReduceDType::F64 as i32,
        DType::F16 => FastReduceDType::F16 as i32,
        DType::BF16 => FastReduceDType::BF16 as i32,
        DType::U32 => FastReduceDType::U32 as i32,
        DType::I64 => FastReduceDType::I64 as i32,
        DType::U8 => FastReduceDType::U8 as i32,
        DType::F8E4M3 => FastReduceDType::F8E4M3 as i32,
    }
}

/// Convert candle DType to IndexingDataDType for FFI dispatcher
fn dtype_to_indexing_data_dtype(dtype: DType) -> i32 {
    use kernels::simple::indexing::IndexingDataDType;
    match dtype {
        DType::F32 => IndexingDataDType::F32 as i32,
        DType::F64 => IndexingDataDType::F64 as i32,
        DType::U8 => IndexingDataDType::U8 as i32,
        DType::U32 => IndexingDataDType::U32 as i32,
        DType::I64 => IndexingDataDType::I64 as i32,
        DType::F16 => IndexingDataDType::F16 as i32,
        DType::BF16 => IndexingDataDType::BF16 as i32,
        DType::F8E4M3 => IndexingDataDType::F8E4M3 as i32,
    }
}

/// Convert index storage dtype to IndexDType for FFI dispatcher
fn storage_to_index_dtype(slice: &CudaStorageSlice) -> Option<i32> {
    use kernels::simple::indexing::IndexDType;
    match slice {
        CudaStorageSlice::I64(_) => Some(IndexDType::I64 as i32),
        CudaStorageSlice::U32(_) => Some(IndexDType::U32 as i32),
        CudaStorageSlice::U8(_) => Some(IndexDType::U8 as i32),
        _ => None,
    }
}

/// Convert candle DType to ScatterDType for FFI dispatcher
/// Returns None for unsupported dtypes (only f32, f64, f16, bf16 supported)
fn dtype_to_scatter_dtype(dtype: DType) -> Option<i32> {
    use kernels::simple::scatter_op::ScatterDType;
    match dtype {
        DType::F32 => Some(ScatterDType::F32 as i32),
        DType::F64 => Some(ScatterDType::F64 as i32),
        DType::F16 => Some(ScatterDType::F16 as i32),
        DType::BF16 => Some(ScatterDType::BF16 as i32),
        _ => None,
    }
}

/// Convert candle DType to RepeatPenaltyDType for FFI dispatcher
/// Returns None for unsupported dtypes (only f32, f64, f16, bf16 supported)
fn dtype_to_repeat_penalty_dtype(dtype: DType) -> Option<i32> {
    use kernels::simple::repeat_penalty::RepeatPenaltyDType;
    match dtype {
        DType::F32 => Some(RepeatPenaltyDType::F32 as i32),
        DType::F64 => Some(RepeatPenaltyDType::F64 as i32),
        DType::F16 => Some(RepeatPenaltyDType::F16 as i32),
        DType::BF16 => Some(RepeatPenaltyDType::BF16 as i32),
        _ => None,
    }
}

/// Map kernel name to UnaryOp enum value for FFI dispatcher
fn kernel_name_to_unary_op(kernel: &str) -> Option<i32> {
    use kernels::simple::unary::UnaryOp;
    match kernel {
        "ucopy" => Some(UnaryOp::Copy as i32),
        "uneg" => Some(UnaryOp::Neg as i32),
        "urecip" => Some(UnaryOp::Recip as i32),
        "uexp" => Some(UnaryOp::Exp as i32),
        "ulog" => Some(UnaryOp::Log as i32),
        "usin" => Some(UnaryOp::Sin as i32),
        "ucos" => Some(UnaryOp::Cos as i32),
        "utanh" => Some(UnaryOp::Tanh as i32),
        "uerf" => Some(UnaryOp::Erf as i32),
        "uceil" => Some(UnaryOp::Ceil as i32),
        "ufloor" => Some(UnaryOp::Floor as i32),
        "uround" => Some(UnaryOp::Round as i32),
        "unormcdf" => Some(UnaryOp::Normcdf as i32),
        "uabs" => Some(UnaryOp::Abs as i32),
        "usqr" => Some(UnaryOp::Sqr as i32),
        "usqrt" => Some(UnaryOp::Sqrt as i32),
        "ugelu" => Some(UnaryOp::Gelu as i32),
        "ugelu_erf" => Some(UnaryOp::GeluErf as i32),
        "urelu" => Some(UnaryOp::Relu as i32),
        "usilu" => Some(UnaryOp::Silu as i32),
        "usign" => Some(UnaryOp::Sign as i32),
        "usigmoid" => Some(UnaryOp::Sigmoid as i32),
        _ => None,
    }
}

impl<U: UnaryOpT> Map1 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let start_offset = layout.start_offset();
        let stream = dev.cuda_stream();
        cuda_breadcrumb!(U::KERNEL);

        // Try to use FFI dispatcher first
        if let Some(op) = kernel_name_to_unary_op(U::KERNEL) {
            let dtype = T::DTYPE;
            let dtype_i32 = dtype_to_unary_dtype(dtype);

            // Prepare dims/strides info for non-contiguous tensors
            let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
                None
            } else {
                Some(dev.info_table(&[dims, layout.stride()].concat())?)
            };

            let src_slice = &src.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, el_count, origin)? };
            {
                let info_ptr = match &info {
                    Some(s) => {
                        let (ptr, _guard) = s.device_ptr(&stream);
                        ptr as *const usize
                    }
                    None => std::ptr::null(),
                };
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

                unsafe {
                    kernels::simple::unary::run_unary_op(
                        op,
                        dtype_i32,
                        el_count,
                        dims.len(),
                        info_ptr,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                    );
                }
            }
            return Ok((out, out_backing));
        }

        // All unary operations should be handled by FFI dispatcher above
        Err(CudaError::InternalError(format!(
            "Unrecognized unary kernel '{}' - all operations should use FFI dispatcher",
            U::KERNEL
        )))
        .w()
    }
}

struct IndexSelect<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map1 for IndexSelect<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        src_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_index_select");
        let ids_l = &self.1;
        let ids = &self.0;

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "index_select ids should be u8, u32, or i64",
                expected: DType::U32,
                got: ids.dtype(),
            })
            .w()?,
        };

        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        let ds = dev.info_table(&[ids_dims, ids_l.stride()].concat())?;
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let left_size: usize = src_l.dims()[..self.2].iter().product();
        let right_size: usize = src_l.dims()[self.2 + 1..].iter().product();
        let src_dim_size = src_l.dims()[self.2];
        let ids_dim_size = ids_shape.elem_count();
        let dst_el = ids_shape.elem_count() * left_size * right_size;

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Get ids pointer based on dtype - need to keep temp slices alive
            let ids_ptr: u64 = match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_select(
                            idx_dtype,
                            data_dtype,
                            dst_el,
                            ids_dims.len(),
                            ds_ptr as *const usize,
                            ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_size,
                            src_dim_size,
                            ids_dim_size,
                            right_size,
                        );
                    }
                    ptr
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_select(
                            idx_dtype,
                            data_dtype,
                            dst_el,
                            ids_dims.len(),
                            ds_ptr as *const usize,
                            ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_size,
                            src_dim_size,
                            ids_dim_size,
                            right_size,
                        );
                    }
                    ptr
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_select(
                            idx_dtype,
                            data_dtype,
                            dst_el,
                            ids_dims.len(),
                            ds_ptr as *const usize,
                            ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_size,
                            src_dim_size,
                            ids_dim_size,
                            right_size,
                        );
                    }
                    ptr
                }
                _ => unreachable!(), // Already checked above
            };
            let _ = ids_ptr; // Suppress unused warning
        }
        Ok((out, out_backing))
    }
}

struct Gather<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map1 for Gather<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        src_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_gather");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "gather ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let el = ids_l.shape().elem_count();
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[dim];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, el, origin)? };
        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Get ids pointer - call FFI inside each match arm to keep temporaries alive
            match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_gather(
                            idx_dtype,
                            data_dtype,
                            el,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            ids_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_gather(
                            idx_dtype,
                            data_dtype,
                            el,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            ids_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_gather(
                            idx_dtype,
                            data_dtype,
                            el,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            ids_dim_sz,
                            right_sz,
                        );
                    }
                }
                _ => unreachable!(), // Already checked above
            };
        }
        Ok((out, out_backing))
    }
}

struct IndexAdd<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map2InPlace for IndexAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        cuda_breadcrumb!("run_index_add");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "index-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[0];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            ids_dim_sz,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            ids_dim_sz,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            ids_dim_sz,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                _ => unreachable!(), // Already checked above
            };
        }
        Ok(())
    }
}

struct Scatter<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map2InPlace for Scatter<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        cuda_breadcrumb!("run_scatter");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "scatter ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                _ => unreachable!(), // Already checked above
            };
        }
        Ok(())
    }
}

struct ScatterAdd<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map2InPlace for ScatterAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        cuda_breadcrumb!("run_scatter_add");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "scatter-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                        );
                    }
                }
                _ => unreachable!(), // Already checked above
            };
        }
        Ok(())
    }
}

struct Conv1D<'a>(&'a crate::conv::ParamsConv1D);
impl Map2 for Conv1D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv1d");
        // Kernel shape: (c_out, c_in_k, k_size)
        // Input shape: (b_size, c_in, l_in) or (c_in, l_in)
        let p = &self.0;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv1d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else if dims.len() == 2 {
            [&[1], dims, &[1], inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv1d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv1d(
                    dtype,
                    dst_el,
                    el,
                    l_out,
                    p.stride,
                    p.padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

struct Conv2D<'a>(&'a crate::conv::ParamsConv2D);
impl Map2 for Conv2D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv2d");
        // Kernel shape: (c_out, c_in_k, h_k, w_k)
        // Input shape: (b_size, c_in, h_in, w_in)
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv2d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv2d(
                    dtype,
                    dst_el,
                    el,
                    out_w,
                    out_h,
                    p.stride,
                    p.padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

struct Col2Im1D {
    stride: usize,
}

impl Map1 for Col2Im1D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        col: &CudaSlice<T>,
        dev: &CudaDevice,
        l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_col2im1d");
        let (b_size, l_in, c_out, k_size) = l.shape().dims4()?;
        let stride = self.stride;
        let l_out = (l_in - 1) * stride + k_size;
        let dst_el = b_size * c_out * l_out;
        let (im, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("col2im1d not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        {
            let (col_ptr, _col_guard) = col.device_ptr(&stream);
            let (im_ptr, _im_guard) = im.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_col2im1d(
                    dtype,
                    dst_el,
                    l_out,
                    l_in,
                    c_out,
                    k_size,
                    stride,
                    col_ptr as *const std::ffi::c_void,
                    im_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((im, out_backing))
    }
}

struct ConvTranspose1D<'a>(&'a crate::conv::ParamsConvTranspose1D);
impl Map2 for ConvTranspose1D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv_transpose1d");
        // Kernel shape: (c_in_k, c_out, l_k)
        // Input shape: (b_size, c_in, l_in)
        let p = &self.0;
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv_transpose1d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose1d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv_transpose1d(
                    dtype,
                    dst_el,
                    el,
                    l_out,
                    p.stride,
                    p.padding,
                    p.output_padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

struct ConvTranspose2D<'a>(&'a crate::conv::ParamsConvTranspose2D);
impl Map2 for ConvTranspose2D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv_transpose2d");
        // Kernel shape: (c_in_k, c_out, h_k, w_k)
        // Input shape: (b_size, c_in, h_in, w_in)
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv_transpose2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose2d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv_transpose2d(
                    dtype,
                    dst_el,
                    el,
                    out_w,
                    out_h,
                    p.stride,
                    p.padding,
                    p.output_padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

enum PoolOp {
    Max,
    Avg,
}

struct Pool2D {
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    op: PoolOp,
}

impl Map1 for Pool2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        dev: &CudaDevice,
        inp_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_pool2d");
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for pool {dims:?}")
        };
        let el = shape.elem_count();
        let out_w = (dims[2] - self.w_k) / self.w_stride + 1;
        let out_h = (dims[3] - self.h_k) / self.h_stride + 1;
        let dst_el = out_w * out_h * dims[0] * dims[1];

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("pool2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                match self.op {
                    PoolOp::Max => {
                        kernels::simple::conv::run_max_pool2d(
                            dtype,
                            el,
                            self.w_k,
                            self.h_k,
                            self.w_stride,
                            self.h_stride,
                            ds_ptr as *const usize,
                            inp_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                        );
                    }
                    PoolOp::Avg => {
                        kernels::simple::conv::run_avg_pool2d(
                            dtype,
                            el,
                            self.w_k,
                            self.h_k,
                            self.w_stride,
                            self.h_stride,
                            ds_ptr as *const usize,
                            inp_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
            }
        }
        Ok((out, out_backing))
    }
}

struct UpsampleNearest2D(usize, usize);
impl Map1 for UpsampleNearest2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        dev: &CudaDevice,
        inp_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_upsample_nearest2d");
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for upsample {dims:?}")
        };
        let (out_w, out_h) = (self.0, self.1);
        let dst_el = out_w * out_h * dims[0] * dims[1];

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("upsample_nearest2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = dev.info_table(&ds)?;
        let scale_w = dims[2] as f64 / out_w as f64;
        let scale_h = dims[3] as f64 / out_h as f64;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_upsample_nearest2d(
                    dtype,
                    out_w,
                    out_h,
                    scale_w,
                    scale_h,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

/// Convert condition storage dtype to WhereCondDType for FFI dispatcher
fn storage_to_where_cond_dtype(slice: &CudaStorageSlice) -> Option<i32> {
    use kernels::simple::ternary::WhereCondDType;
    match slice {
        CudaStorageSlice::I64(_) => Some(WhereCondDType::I64 as i32),
        CudaStorageSlice::U32(_) => Some(WhereCondDType::U32 as i32),
        CudaStorageSlice::U8(_) => Some(WhereCondDType::U8 as i32),
        _ => None,
    }
}

/// Convert candle DType to WhereDataDType for FFI dispatcher
fn dtype_to_where_data_dtype(dtype: DType) -> i32 {
    use kernels::simple::ternary::WhereDataDType;
    match dtype {
        DType::F32 => WhereDataDType::F32 as i32,
        DType::F64 => WhereDataDType::F64 as i32,
        DType::U8 => WhereDataDType::U8 as i32,
        DType::U32 => WhereDataDType::U32 as i32,
        DType::I64 => WhereDataDType::I64 as i32,
        DType::F16 => WhereDataDType::F16 as i32,
        DType::BF16 => WhereDataDType::BF16 as i32,
        DType::F8E4M3 => WhereDataDType::F8E4M3 as i32,
    }
}

/// Convert candle DType to ConvDType for FFI dispatcher
/// Returns None for I64 and F8E4M3 which are not supported by conv operations
fn dtype_to_conv_dtype(dtype: DType) -> Option<i32> {
    use kernels::simple::conv::ConvDType;
    match dtype {
        DType::F32 => Some(ConvDType::F32 as i32),
        DType::F64 => Some(ConvDType::F64 as i32),
        DType::F16 => Some(ConvDType::F16 as i32),
        DType::BF16 => Some(ConvDType::BF16 as i32),
        DType::U8 => Some(ConvDType::U8 as i32),
        DType::U32 => Some(ConvDType::U32 as i32),
        DType::I64 => None,    // Not supported by conv operations
        DType::F8E4M3 => None, // Not supported by conv operations
    }
}

struct WhereCond<'a>(&'a CudaStorage, &'a Layout);
impl Map2 for WhereCond<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        t: &CudaSlice<T>,
        layout_t: &Layout,
        f: &CudaSlice<T>,
        layout_f: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_where");
        let ids_l = &self.1;
        let ids = &self.0;

        // Get condition dtype for FFI
        let cond_dtype = match storage_to_where_cond_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "where conditions should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })
            .w()?,
        };

        let shape = ids_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ds =
            dev.info_table(&[dims, ids_l.stride(), layout_t.stride(), layout_f.stride()].concat())?;
        let t = &t.slice(layout_t.start_offset()..);
        let f = &f.slice(layout_f.start_offset()..);

        // Get data dtype for FFI
        let data_dtype = dtype_to_where_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, el, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (t_ptr, _t_guard) = t.device_ptr(&stream);
            let (f_ptr, _f_guard) = f.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            match &ids.slice {
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::ternary::run_where(
                            cond_dtype,
                            data_dtype,
                            el,
                            dims.len(),
                            ds_ptr as *const usize,
                            ids_ptr as *const std::ffi::c_void,
                            t_ptr as *const std::ffi::c_void,
                            f_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::ternary::run_where(
                            cond_dtype,
                            data_dtype,
                            el,
                            dims.len(),
                            ds_ptr as *const usize,
                            ids_ptr as *const std::ffi::c_void,
                            t_ptr as *const std::ffi::c_void,
                            f_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::ternary::run_where(
                            cond_dtype,
                            data_dtype,
                            el,
                            dims.len(),
                            ds_ptr as *const usize,
                            ids_ptr as *const std::ffi::c_void,
                            t_ptr as *const std::ffi::c_void,
                            f_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
                _ => unreachable!(), // Already checked above
            };
        }
        Ok((out, out_backing))
    }
}

/// Convert candle DType to BinaryDType for FFI dispatcher
fn dtype_to_binary_dtype(dtype: DType) -> i32 {
    use kernels::simple::binary::BinaryDType;
    match dtype {
        DType::F32 => BinaryDType::F32 as i32,
        DType::F64 => BinaryDType::F64 as i32,
        DType::U8 => BinaryDType::U8 as i32,
        DType::U32 => BinaryDType::U32 as i32,
        DType::I64 => BinaryDType::I64 as i32,
        DType::F16 => BinaryDType::F16 as i32,
        DType::BF16 => BinaryDType::BF16 as i32,
        DType::F8E4M3 => BinaryDType::F8E4M3 as i32,
    }
}

/// Convert BinaryInplaceOp to FFI enum value for in-place binary dispatcher
fn binary_inplace_op_to_ffi(op: crate::op::BinaryInplaceOp) -> i32 {
    use kernels::simple::binary::BinaryInplaceOp as FFIOp;
    match op {
        crate::op::BinaryInplaceOp::Add => FFIOp::Add as i32,
        crate::op::BinaryInplaceOp::Sub => FFIOp::Sub as i32,
        crate::op::BinaryInplaceOp::Mul => FFIOp::Mul as i32,
        crate::op::BinaryInplaceOp::Div => FFIOp::Div as i32,
        crate::op::BinaryInplaceOp::Min => FFIOp::Min as i32,
        crate::op::BinaryInplaceOp::Max => FFIOp::Max as i32,
    }
}

/// Map kernel name to BinaryArithOp enum value for FFI dispatcher
fn kernel_name_to_binary_arith_op(kernel: &str) -> Option<i32> {
    use kernels::simple::binary::BinaryArithOp;
    match kernel {
        "badd" => Some(BinaryArithOp::Add as i32),
        "bdiv" => Some(BinaryArithOp::Div as i32),
        "bmul" => Some(BinaryArithOp::Mul as i32),
        "bsub" => Some(BinaryArithOp::Sub as i32),
        "bminimum" => Some(BinaryArithOp::Minimum as i32),
        "bmaximum" => Some(BinaryArithOp::Maximum as i32),
        _ => None,
    }
}

impl<U: crate::op::BinaryOpT> Map2 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        lhs: &CudaSlice<T>,
        lhs_l: &Layout,
        rhs: &CudaSlice<T>,
        rhs_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!(U::KERNEL);
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let lhs_start = lhs_l.start_offset();
        let rhs_start = rhs_l.start_offset();
        let stream = dev.cuda_stream();

        // Try to use FFI dispatcher first
        if let Some(op) = kernel_name_to_binary_arith_op(U::KERNEL) {
            let dtype = T::DTYPE;
            let dtype_i32 = dtype_to_binary_dtype(dtype);

            // Prepare dims and strides info for non-contiguous tensors
            let info: Option<Arc<Uploaded<usize>>> =
                if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
                    None
                } else {
                    Some(dev.info_table(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
                };

            let lhs_slice = &lhs.slice(lhs_start..);
            let rhs_slice = &rhs.slice(rhs_start..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, elem_count, origin)? };
            {
                let info_ptr = match &info {
                    Some(s) => {
                        let (ptr, _guard) = s.device_ptr(&stream);
                        ptr as *const usize
                    }
                    None => std::ptr::null(),
                };
                let (lhs_ptr, _lhs_guard) = lhs_slice.device_ptr(&stream);
                let (rhs_ptr, _rhs_guard) = rhs_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

                unsafe {
                    kernels::simple::binary::run_binary_arith_op(
                        op,
                        dtype_i32,
                        elem_count,
                        dims.len(),
                        info_ptr,
                        lhs_ptr as *const std::ffi::c_void,
                        rhs_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                    );
                }
            }
            return Ok((out, out_backing));
        }

        // All binary operations should be handled by FFI dispatcher above
        Err(CudaError::InternalError(format!(
            "Unrecognized binary kernel '{}' - all operations should use FFI dispatcher",
            U::KERNEL
        )))
        .w()
    }
}

/// Map CmpOp to binary comparison op enum value for FFI dispatcher
fn cmp_op_to_binary_cmp(op: CmpOp) -> i32 {
    use kernels::simple::binary::BinaryCmpOp;
    match op {
        CmpOp::Eq => BinaryCmpOp::Eq as i32,
        CmpOp::Ne => BinaryCmpOp::Ne as i32,
        CmpOp::Lt => BinaryCmpOp::Lt as i32,
        CmpOp::Le => BinaryCmpOp::Le as i32,
        CmpOp::Gt => BinaryCmpOp::Gt as i32,
        CmpOp::Ge => BinaryCmpOp::Ge as i32,
    }
}

struct Cmp(CmpOp);
impl Map2Any for Cmp {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        lhs: &CudaSlice<T>,
        lhs_l: &Layout,
        rhs: &CudaSlice<T>,
        rhs_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<OutS> {
        cuda_breadcrumb!("run_binary_cmp_op");
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let lhs_start = lhs_l.start_offset();
        let rhs_start = rhs_l.start_offset();
        let stream = dev.cuda_stream();

        // Use FFI dispatcher
        let op = cmp_op_to_binary_cmp(self.0);
        let dtype = T::DTYPE;
        let dtype_i32 = dtype_to_binary_dtype(dtype);

        // Prepare dims and strides info for non-contiguous tensors
        let info: Option<Arc<Uploaded<usize>>> = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            None
        } else {
            Some(dev.info_table(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };

        let lhs_slice = &lhs.slice(lhs_start..);
        let rhs_slice = &rhs.slice(rhs_start..);
        // SAFETY: Allocated memory will be initialized by the kernel
        let (out, out_backing) = unsafe { alloc_inheriting::<u8>(dev, elem_count, origin)? };
        {
            let info_ptr = match &info {
                Some(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    ptr as *const usize
                }
                None => std::ptr::null(),
            };
            let (lhs_ptr, _lhs_guard) = lhs_slice.device_ptr(&stream);
            let (rhs_ptr, _rhs_guard) = rhs_slice.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Keep info alive for the kernel call
            let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

            unsafe {
                kernels::simple::binary::run_binary_cmp_op(
                    op,
                    dtype_i32,
                    elem_count,
                    dims.len(),
                    info_ptr,
                    lhs_ptr as *const std::ffi::c_void,
                    rhs_ptr as *const std::ffi::c_void,
                    out_ptr as *mut u8,
                );
            }
        }
        Ok((S::U8(out), out_backing))
    }
}

fn slice_src_and_dst<'a, T>(
    src: &'a CudaSlice<T>,
    src_l: &Layout,
    dst: &'a mut CudaSlice<T>,
    dst_offset: usize,
) -> (
    cudarc::driver::CudaView<'a, T>,
    cudarc::driver::CudaViewMut<'a, T>,
) {
    let src_offset = src_l.start_offset();
    let to_copy = dst
        .len()
        .saturating_sub(dst_offset)
        .min(src.len().saturating_sub(src_offset));
    let src = src.slice(src_offset..src_offset + to_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + to_copy);
    (src, dst)
}

#[derive(Debug)]
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub device: CudaDevice,
    /// Whether dropping this storage may free its device memory.
    /// [`Backing::Owned`] for everything the pool allocated — which is
    /// everything except the arena leases built by
    /// [`CudaStorage::from_leased_device_ptr`].
    pub backing: Backing,
}

impl CudaStorageSlice {
    /// Release the handle without freeing the memory behind it.
    ///
    /// Calling `leak` rather than merely suppressing the drop is load-bearing:
    /// it waits on the slice's read/write events, destroys them, and decrements
    /// the stream's `Arc`. Bare suppression (`mem::forget`) would strand two
    /// `CudaEvent`s and a stream refcount **per lease** — thousands per second
    /// on the decode path.
    ///
    /// This is the only correct way to dispose of a slice whose memory is owned
    /// elsewhere (an arena slot, a wave range). Dropping one normally calls
    /// `cuMemFreeAsync` on an interior pointer, which the driver rejects into a
    /// silently-recorded error rather than a panic — so the mistake does not
    /// announce itself.
    fn leak_view(self) {
        match self {
            Self::U8(s) => {
                s.leak();
            }
            Self::U32(s) => {
                s.leak();
            }
            Self::I64(s) => {
                s.leak();
            }
            Self::BF16(s) => {
                s.leak();
            }
            Self::F16(s) => {
                s.leak();
            }
            Self::F32(s) => {
                s.leak();
            }
            Self::F64(s) => {
                s.leak();
            }
            Self::F8E4M3(s) => {
                s.leak();
            }
            Self::Moved => Self::unreachable_moved(),
        }
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        // Exhaustive rather than `!= Lease`: a future `Backing` variant must
        // state which side it falls on here, instead of silently inheriting the
        // free path and releasing memory the storage does not own.
        match self.backing {
            Backing::Owned => return,
            Backing::Lease(_) => {}
        }
        std::mem::replace(&mut self.slice, CudaStorageSlice::Moved).leak_view();
    }
}

impl CudaStorage {
    /// Wrap `len` elements of device memory at `ptr` as a leased storage.
    ///
    /// The memory must outlive every tensor derived from this storage, and must
    /// already be valid for `T` — arena slots satisfy both: they live in a
    /// reservation held for the process lifetime, and zero-on-recycle
    /// (invariant 4) means a slot's bytes are always a legal bit pattern.
    ///
    /// # Safety
    /// `ptr` must point to at least `len` elements of `dtype`, be aligned for
    /// it, and stay live and un-aliased-for-writes for the storage's lifetime.
    pub unsafe fn from_leased_device_ptr(
        ptr: u64,
        len: usize,
        dtype: DType,
        device: &CudaDevice,
        origin: LeaseOrigin,
    ) -> Result<Self> {
        let stream = device.cuda_stream();
        let slice = match dtype {
            DType::U8 => CudaStorageSlice::U8(stream.upgrade_device_ptr::<u8>(ptr, len)),
            DType::U32 => CudaStorageSlice::U32(stream.upgrade_device_ptr::<u32>(ptr, len)),
            DType::I64 => CudaStorageSlice::I64(stream.upgrade_device_ptr::<i64>(ptr, len)),
            DType::BF16 => CudaStorageSlice::BF16(stream.upgrade_device_ptr::<bf16>(ptr, len)),
            DType::F16 => CudaStorageSlice::F16(stream.upgrade_device_ptr::<f16>(ptr, len)),
            DType::F32 => CudaStorageSlice::F32(stream.upgrade_device_ptr::<f32>(ptr, len)),
            DType::F64 => CudaStorageSlice::F64(stream.upgrade_device_ptr::<f64>(ptr, len)),
            DType::F8E4M3 => {
                CudaStorageSlice::F8E4M3(stream.upgrade_device_ptr::<F8E4M3>(ptr, len))
            }
        };
        Ok(Self {
            slice,
            device: device.clone(),
            backing: Backing::Lease(origin),
        })
    }
}

pub trait CudaDType: Sized {
    fn as_cuda_slice(s: &CudaStorage) -> Result<&CudaSlice<Self>>;
    fn as_cuda_slice_mut(s: &mut CudaStorage) -> Result<&mut CudaSlice<Self>>;
    fn wrap_cuda_slice(s: CudaSlice<Self>, dev: CudaDevice) -> CudaStorage;

    /// Wrap `len` elements at a device address the caller does not own.
    ///
    /// The typed counterpart of [`CudaStorage::from_leased_device_ptr`], for
    /// generic kernel wrappers that know their output type as a parameter
    /// rather than as a runtime [`DType`]. Same lease semantics: dropping the
    /// storage releases the handle without freeing the memory.
    ///
    /// # Safety
    /// `ptr` must point to at least `len` elements of `Self`, be aligned for
    /// it, and stay live and un-aliased-for-writes for the storage's lifetime.
    unsafe fn wrap_leased_ptr(
        ptr: u64,
        len: usize,
        dev: CudaDevice,
        origin: LeaseOrigin,
    ) -> CudaStorage;
}

macro_rules! cuda_dtype {
    ($ty:ty, $dtype:ident) => {
        impl CudaDType for $ty {
            fn as_cuda_slice(s: &CudaStorage) -> Result<&CudaSlice<Self>> {
                match &s.slice {
                    CudaStorageSlice::$dtype(data) => Ok(&data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn as_cuda_slice_mut(s: &mut CudaStorage) -> Result<&mut CudaSlice<Self>> {
                match s.slice {
                    CudaStorageSlice::$dtype(ref mut data) => Ok(data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn wrap_cuda_slice(slice: CudaSlice<Self>, device: CudaDevice) -> CudaStorage {
                let slice = CudaStorageSlice::$dtype(slice);
                CudaStorage {
                    slice,
                    device,
                    backing: Backing::Owned,
                }
            }

            unsafe fn wrap_leased_ptr(
                ptr: u64,
                len: usize,
                device: CudaDevice,
                origin: LeaseOrigin,
            ) -> CudaStorage {
                let slice = device.cuda_stream().upgrade_device_ptr::<Self>(ptr, len);
                CudaStorage {
                    slice: CudaStorageSlice::$dtype(slice),
                    device,
                    backing: Backing::Lease(origin),
                }
            }
        }
    };
}
cuda_dtype!(u8, U8);
cuda_dtype!(u32, U32);
cuda_dtype!(i64, I64);
cuda_dtype!(f16, F16);
cuda_dtype!(bf16, BF16);
cuda_dtype!(f32, F32);
cuda_dtype!(f64, F64);
cuda_dtype!(F8E4M3, F8E4M3);

/// Alignment for an inherited output. Matches what `cudaMalloc` guarantees, so a
/// wave-backed output is as aligned as the pool allocation it replaces for every
/// vectorised access a kernel makes.
pub const INHERIT_ALIGN: usize = 256;

/// Allocate an op's output from the arena its **operand** came from, falling
/// back to the pool.
///
/// This is the whole operand-provenance rule in one function. The returned
/// [`Backing`] is what the caller must stamp on the output storage: pairing the
/// slice with its backing here — rather than letting the call site name a
/// `Backing` literal next to an allocation it made separately — is what stops
/// the two drifting apart, which is the mistake that turns a wave range into a
/// double free.
///
/// # Safety
///
/// The returned slice is uninitialised. The caller must have the kernel write it
/// before anything reads it, exactly as for a bare `dev.alloc`.
pub unsafe fn alloc_inheriting<T: DeviceRepr>(
    dev: &CudaDevice,
    elem_count: usize,
    from: Backing,
) -> Result<(CudaSlice<T>, Backing)> {
    let ticket = from.inherit_ticket();
    let bytes = elem_count * std::mem::size_of::<T>();
    // Attributed, so the fall-through below is not silent: a site that lost its
    // provenance and a site whose arena overflowed both end up on `dev.alloc`
    // and are otherwise indistinguishable in any report.
    if let Some(ptr) = wave_provenance::wave_alloc_attributed(ticket, bytes, INHERIT_ALIGN) {
        // Dropping this slice bare would `cuMemFreeAsync` an address inside
        // the VMM reservation the wave arenas are carved from — memory the
        // stream-ordered pool never allocated — so the driver rejects it and
        // nothing is freed. That is what makes the window between here and
        // the caller stamping `Backing::Lease` harmless.
        let slice = dev.cuda_stream().upgrade_device_ptr::<T>(ptr, elem_count);
        let ticket = ticket.expect("a carved range implies a ticket");
        return Ok((slice, Backing::Lease(LeaseOrigin::Wave(ticket))));
    }
    Ok((dev.alloc::<T>(elem_count)?, Backing::Owned))
}

impl CudaStorage {
    pub fn wrap_cuda_slice<T: CudaDType>(slice: CudaSlice<T>, device: CudaDevice) -> CudaStorage {
        T::wrap_cuda_slice(slice, device)
    }

    /// Wrap a slice produced by [`alloc_inheriting`], carrying the backing it
    /// resolved to.
    ///
    /// [`Self::wrap_cuda_slice`] stamps `Backing::Owned`, which is a claim of
    /// ownership: dropping such a storage frees the memory. Applying it to a
    /// slice that turned out to be a view over a wave range would hand the
    /// arena's memory to the pool's free path, so an allocation site that can
    /// inherit must carry the resolved backing through to the wrap.
    pub fn wrap_cuda_slice_backed<T: CudaDType>(
        slice: CudaSlice<T>,
        device: CudaDevice,
        backing: Backing,
    ) -> CudaStorage {
        let mut storage = T::wrap_cuda_slice(slice, device);
        storage.backing = backing;
        storage
    }

    /// Wrap a borrowed device range as storage of element type `T`.
    ///
    /// Kernel wrappers use this for outputs written into a transient span:
    /// the span owns the memory and its generation frees it, so the storage
    /// must not.
    ///
    /// # Safety
    /// As [`CudaDType::wrap_leased_ptr`].
    pub unsafe fn wrap_leased_ptr<T: CudaDType>(
        ptr: u64,
        len: usize,
        device: CudaDevice,
        origin: LeaseOrigin,
    ) -> CudaStorage {
        T::wrap_leased_ptr(ptr, len, device, origin)
    }

    pub fn as_cuda_slice<T: CudaDType>(&self) -> Result<&CudaSlice<T>> {
        T::as_cuda_slice(self)
    }

    pub fn as_cuda_slice_mut<T: CudaDType>(&mut self) -> Result<&mut CudaSlice<T>> {
        T::as_cuda_slice_mut(self)
    }

    /// Copy a range of U32 data from GPU to a host buffer on a specific stream.
    ///
    /// When `dst` is backed by pinned memory (`cuMemAllocHost`), the copy is
    /// truly asynchronous — the CPU returns immediately and the DMA engine
    /// handles the transfer.  This is the async DtoH path used for routing
    /// index transfer without draining the compute pipeline.
    ///
    /// `offset` and `elem_count` identify the sub-range of the device slice
    /// to copy (from `contiguous_offsets` on the tensor layout).  `dst` must
    /// have at least `elem_count` elements.
    ///
    /// Returns `Err` if the storage is not U32.
    pub fn copy_u32_to_host_on_stream(
        &self,
        dst: &mut [u32],
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        offset: usize,
        elem_count: usize,
    ) -> Result<()> {
        match &self.slice {
            CudaStorageSlice::U32(slice) => {
                if offset + elem_count > slice.len() {
                    crate::bail!(
                        "copy_u32_to_host_on_stream: range {}..{} exceeds slice len {}",
                        offset,
                        offset + elem_count,
                        slice.len(),
                    );
                }
                if dst.len() < elem_count {
                    crate::bail!(
                        "copy_u32_to_host_on_stream: dst too small ({} < {})",
                        dst.len(),
                        elem_count,
                    );
                }
                let view = slice.slice(offset..offset + elem_count);
                stream
                    .memcpy_dtoh(&view, &mut dst[..elem_count])
                    .map_err(crate::Error::wrap)?;
                Ok(())
            }
            _ => crate::bail!(
                "copy_u32_to_host_on_stream: expected U32 storage, got {:?}",
                self.dtype(),
            ),
        }
    }

    /// In-place sparse addition - mutates the tensor directly without cloning.
    /// This is 20x+ faster than add_at_indices for large tensors with sparse updates.
    pub fn add_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get the CUDA stream for pointer access
        let stream = device.cuda_stream();

        // Mutate in-place based on dtype using direct FFI calls
        match &mut self.slice {
            CudaStorageSlice::F32(dst) => {
                use cudarc::driver::DevicePtrMut;
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_f32(
                        dst_ptr as *mut f32,
                        indices_ptr as *const u32,
                        num_indices,
                        value,
                        1usize,
                    );
                }
            }
            CudaStorageSlice::F16(dst) => {
                use cudarc::driver::DevicePtrMut;
                let value_f16 = half::f16::from_f32(value);
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_f16(
                        dst_ptr as *mut half::f16,
                        indices_ptr as *const u32,
                        num_indices,
                        value_f16,
                        1usize,
                    );
                }
            }
            CudaStorageSlice::BF16(dst) => {
                use cudarc::driver::DevicePtrMut;
                let value_bf16 = half::bf16::from_f32(value);
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_bf16(
                        dst_ptr as *mut half::bf16,
                        indices_ptr as *const u32,
                        num_indices,
                        value_bf16,
                        1usize,
                    );
                }
            }
            CudaStorageSlice::F64(dst) => {
                use cudarc::driver::DevicePtrMut;
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_f64(
                        dst_ptr as *mut f64,
                        indices_ptr as *const u32,
                        num_indices,
                        value as f64,
                        1usize,
                    );
                }
            }
            _ => crate::bail!(
                "add_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        }

        Ok(())
    }

    /// In-place sparse subtraction - mutates the tensor directly without cloning.
    /// This is 20x+ faster than sub_at_indices for large tensors with sparse updates.
    pub fn sub_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "sub_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::scatter_op::run_scatter_op_at_indices(
                kernels::simple::scatter_op::ScatterOp::Sub as i32,
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                value,
                value as f64,
                1, // stride
            );
        }

        Ok(())
    }

    /// In-place sparse subtraction with per-index values - mutates the tensor directly without cloning.
    /// Each index gets its own value: data[indices[i]] -= values[i]
    /// This is 20x+ faster than sub_at_indices for large tensors with sparse updates.
    pub fn sub_at_indices_mut_with_values(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        if indices.len() != values.len() {
            crate::bail!(
                "indices and values must have the same length, got {} and {}",
                indices.len(),
                values.len()
            );
        }

        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices and values to GPU
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "sub_at_indices_with_values is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        // For F64, we need to convert values to f64
        // Keep the allocated GPU buffers alive until after the kernel call
        let values_dev_f64: Option<CudaSlice<f64>>;
        let values_dev_f32: Option<CudaSlice<f32>>;
        let values_ptr: *const std::ffi::c_void;

        if self.dtype() == DType::F64 {
            let values_f64_vec: Vec<f64> = values.iter().map(|&v| v as f64).collect();
            values_dev_f64 = Some(device.memcpy_stod(&values_f64_vec)?);
            values_dev_f32 = None;
            let (ptr, _g) = values_dev_f64.as_ref().unwrap().device_ptr(&stream);
            values_ptr = ptr as *const std::ffi::c_void;
        } else {
            values_dev_f64 = None;
            values_dev_f32 = Some(device.memcpy_stod(values)?);
            let (ptr, _g) = values_dev_f32.as_ref().unwrap().device_ptr(&stream);
            values_ptr = ptr as *const std::ffi::c_void;
        }

        // Keep both buffers alive through the unsafe block
        let _keep_alive_f64 = &values_dev_f64;
        let _keep_alive_f32 = &values_dev_f32;

        unsafe {
            kernels::simple::scatter_op::run_sub_at_indices_with_values(
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                values_ptr,
                num_indices,
            );
        }

        Ok(())
    }

    pub fn sub_at_indices(&self, _layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        forbidden_alloc::record("CudaStorage::sub_at_indices", self.slice.byte_len());
        let device = self.device().clone();

        // Early return for empty indices
        if indices.is_empty() {
            let slice = match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            };
            return Ok(Self {
                slice,
                device,
                backing: Backing::Owned,
            });
        }

        // Clone and then mutate in-place
        let mut result = Self {
            slice: match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            },
            device,
            backing: Backing::Owned,
        };

        // Use in-place mutation method
        result.sub_at_indices_mut(_layout, indices, value)?;
        Ok(result)
    }

    /// In-place sparse division - mutates the tensor directly without cloning.
    /// This is 20x+ faster than div_at_indices for large tensors with sparse updates.
    pub fn div_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "div_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::scatter_op::run_scatter_op_at_indices(
                kernels::simple::scatter_op::ScatterOp::Div as i32,
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                value,
                value as f64,
                1, // stride
            );
        }

        Ok(())
    }

    /// In-place sparse multiplication - mutates the tensor directly without cloning.
    /// This is 20x+ faster than mul_at_indices for large tensors with sparse updates.
    pub fn mul_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "mul_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::scatter_op::run_scatter_op_at_indices(
                kernels::simple::scatter_op::ScatterOp::Mul as i32,
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                value,
                value as f64,
                1, // stride
            );
        }

        Ok(())
    }

    /// In-place repeat penalty - applies penalty based on logit sign in a single GPU kernel pass.
    /// For positive logits: divides by penalty (reduces probability)
    /// For negative/zero logits: multiplies by penalty (reduces probability)
    /// This is extremely efficient as it combines the logic from both div_at_indices and mul_at_indices.
    pub fn repeat_penalty_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        penalty: f32,
    ) -> Result<()> {
        // Early return for empty indices or penalty of 1.0 (no-op)
        if indices.is_empty() || penalty == 1.0 {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_repeat_penalty_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "repeat_penalty is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::repeat_penalty::run_repeat_penalty(
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                penalty as f64,
                std::ptr::null_mut(), // default stream
            );
        }

        Ok(())
    }

    /// In-place type conversion using CUDA cast_mut kernels.
    ///
    /// This function converts the tensor data to a new dtype in-place when possible.
    /// The buffer must be large enough to hold the destination data type.
    ///
    /// # Returns
    /// - `Ok(true)` if the conversion was performed in-place
    /// - `Ok(false)` if the buffer was too small (caller should fall back to regular to_dtype)
    ///
    /// # Safety
    /// The buffer must have at least `elem_count * max(src_size, dst_size)` bytes allocated.
    pub fn to_dtype_mut(&mut self, layout: &Layout, dtype: DType) -> Result<bool> {
        use kernels::simple::cast::CastDType;

        let src_dtype = self.dtype();

        // No-op if same dtype
        if src_dtype == dtype {
            return Ok(true);
        }

        // Contiguous tensors are required for in-place conversion.
        // The caller (Tensor::to_dtype_mut) ensures this, but we check as a safety measure.
        if !layout.is_contiguous() {
            return Ok(false);
        }

        // The in-place cast reads/writes from the buffer's base pointer (no offset) and
        // the caller resets the layout to `contiguous(offset 0)` afterwards. A contiguous
        // *view* with a non-zero start offset (e.g. the second half of a last-dim
        // `narrow`) would therefore cast the wrong elements and report the wrong region.
        // Fall back to the allocating `to_dtype`, which honours the offset.
        if layout.start_offset() != 0 {
            return Ok(false);
        }

        let elem_count = layout.shape().elem_count();
        let src_size = src_dtype.size_in_bytes();
        let dst_size = dtype.size_in_bytes();

        // Check if buffer is large enough for the destination type
        let buffer_bytes = self.buffer_byte_len();
        let required_bytes = elem_count * dst_size.max(src_size);

        if buffer_bytes < required_bytes {
            // Buffer too small, caller should use regular to_dtype
            return Ok(false);
        }

        // Convert candle DType to CastDType
        let dtype_to_cast = |dt: DType| -> i32 {
            match dt {
                DType::F32 => CastDType::F32 as i32,
                DType::F64 => CastDType::F64 as i32,
                DType::U8 => CastDType::U8 as i32,
                DType::U32 => CastDType::U32 as i32,
                DType::I64 => CastDType::I64 as i32,
                DType::F16 => CastDType::F16 as i32,
                DType::BF16 => CastDType::BF16 as i32,
                DType::F8E4M3 => CastDType::F8E4M3 as i32,
            }
        };

        let src_dtype_i32 = dtype_to_cast(src_dtype);
        let dst_dtype_i32 = dtype_to_cast(dtype);
        let stream = self.device.cuda_stream();

        // Get mutable pointer and perform in-place cast
        let buf_ptr = self.slice.device_ptr_mut(&stream)?;

        unsafe {
            kernels::simple::cast::run_cast_mut(src_dtype_i32, dst_dtype_i32, elem_count, buf_ptr);
        }

        // Now we need to "reinterpret" the slice as the new dtype.
        // This is safe because:
        // 1. We verified buffer size is sufficient
        // 2. The cast_mut kernel has converted the data in-place
        // 3. The buffer memory layout is compatible
        // `reinterpret_slice_as` allocates a *fresh buffer* and copies into it,
        // so this is a change of ownership, not a retype in place. Two things
        // follow, and both were previously missed:
        //
        //  1. The outgoing slice has to be disposed of according to the backing
        //     it had. A plain assignment would drop it — and dropping a lease
        //     calls `cuMemFreeAsync` on memory owned by an arena.
        //  2. `backing` has to follow the bytes to wherever the *new* buffer
        //     came from. The destination inherits this storage's arena, so a
        //     wave-backed cast stays on the wave; hard-coding either answer gets
        //     one of the two cases wrong (a lease marked `Owned` is a double
        //     free, a pool buffer marked `Lease` is a permanent leak).
        let (fresh, fresh_backing) = self.reinterpret_slice_as(dtype, elem_count)?;
        let previous = std::mem::replace(&mut self.slice, fresh);
        match self.backing {
            Backing::Lease(_) => previous.leak_view(),
            Backing::Owned => drop(previous),
        }
        self.backing = fresh_backing;

        Ok(true)
    }

    /// Returns the total byte length of the underlying buffer.
    fn buffer_byte_len(&self) -> usize {
        match &self.slice {
            CudaStorageSlice::U8(s) => s.len() * std::mem::size_of::<u8>(),
            CudaStorageSlice::U32(s) => s.len() * std::mem::size_of::<u32>(),
            CudaStorageSlice::I64(s) => s.len() * std::mem::size_of::<i64>(),
            CudaStorageSlice::BF16(s) => s.len() * std::mem::size_of::<bf16>(),
            CudaStorageSlice::F16(s) => s.len() * std::mem::size_of::<f16>(),
            CudaStorageSlice::F32(s) => s.len() * std::mem::size_of::<f32>(),
            CudaStorageSlice::F64(s) => s.len() * std::mem::size_of::<f64>(),
            CudaStorageSlice::F8E4M3(s) => s.len() * std::mem::size_of::<F8E4M3>(),
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    /// Reinterpret the underlying buffer as a different dtype by copying to a new typed buffer.
    ///
    /// After an in-place cast, the buffer contains valid data for the target dtype,
    /// but the CudaSlice still has the original type. This function creates a new
    /// properly-typed CudaSlice by copying the bytes.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The buffer contains valid data for the target dtype
    /// - The buffer is large enough to hold `elem_count` elements of the target dtype
    fn reinterpret_slice_as(
        &self,
        dtype: DType,
        elem_count: usize,
    ) -> Result<(CudaStorageSlice, Backing)> {
        let dev = &self.device;
        let byte_count = elem_count * dtype.size_in_bytes();
        let stream = dev.cuda_stream();

        // Get source pointer
        let src_ptr: cudarc::driver::sys::CUdeviceptr = match &self.slice {
            CudaStorageSlice::U8(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::U32(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::I64(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::BF16(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F16(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F32(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F64(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F8E4M3(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        };

        // Allocate destination buffer of the correct type and copy raw bytes.
        // The retyped buffer holds the same values this storage already holds,
        // so it belongs wherever this storage does.
        let inherit = self.backing;
        macro_rules! alloc_and_copy {
            ($ty:ty, $wrapper:path) => {{
                use cudarc::driver::DevicePtrMut;
                let (mut dst, dst_backing) =
                    unsafe { alloc_inheriting::<$ty>(dev, elem_count, inherit)? };
                let (dst_ptr, _) = dst.device_ptr_mut(&stream);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_sync(dst_ptr, src_ptr, byte_count).w()?;
                }
                Ok(($wrapper(dst), dst_backing))
            }};
        }

        match dtype {
            DType::U8 => alloc_and_copy!(u8, CudaStorageSlice::U8),
            DType::U32 => alloc_and_copy!(u32, CudaStorageSlice::U32),
            DType::I64 => alloc_and_copy!(i64, CudaStorageSlice::I64),
            DType::BF16 => alloc_and_copy!(bf16, CudaStorageSlice::BF16),
            DType::F16 => alloc_and_copy!(f16, CudaStorageSlice::F16),
            DType::F32 => alloc_and_copy!(f32, CudaStorageSlice::F32),
            DType::F64 => alloc_and_copy!(f64, CudaStorageSlice::F64),
            DType::F8E4M3 => alloc_and_copy!(F8E4M3, CudaStorageSlice::F8E4M3),
        }
    }

    pub fn div_at_indices(&self, _layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        forbidden_alloc::record("CudaStorage::div_at_indices", self.slice.byte_len());
        let device = self.device().clone();

        // Early return for empty indices
        if indices.is_empty() {
            let slice = match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            };
            return Ok(Self {
                slice,
                device,
                backing: Backing::Owned,
            });
        }

        // Clone and then mutate in-place
        let mut result = Self {
            slice: match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            },
            device,
            backing: Backing::Owned,
        };

        // Use in-place mutation method
        result.div_at_indices_mut(_layout, indices, value)?;
        Ok(result)
    }

    /// Fast scalar transfer - optimized for single-element copies.
    /// Uses slice-based transfer to copy only a single element instead of the whole tensor.
    /// This is much faster than `to_cpu_storage()` for extracting single values.
    pub fn to_cpu_scalar<T>(&self, offset: usize) -> Result<T>
    where
        T: DeviceRepr + WithDType + Copy,
    {
        use cudarc::driver::DeviceSlice;

        match &self.slice {
            CudaStorageSlice::U8(slice) if T::DTYPE == DType::U8 => {
                // Create a slice of just one element
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                // SAFETY: We've checked the dtype matches and vec has exactly 1 element
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::U32(slice) if T::DTYPE == DType::U32 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::I64(slice) if T::DTYPE == DType::I64 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::F32(slice) if T::DTYPE == DType::F32 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::F64(slice) if T::DTYPE == DType::F64 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::F16(slice) if T::DTYPE == DType::F16 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::BF16(slice) if T::DTYPE == DType::BF16 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            _ => {
                // Fallback to full transfer for unsupported types
                let cpu_storage = self.to_cpu_storage()?;
                let data = T::cpu_storage_as_slice(&cpu_storage)?;
                Ok(data[offset])
            }
        }
    }

    /// Transfer a single scalar element from GPU to CPU (type-erased version).
    /// This is used as a fast fallback when the compile-time type is unknown.
    /// Returns the value as a CpuStorage containing exactly one element.
    pub fn to_cpu_storage_scalar(&self, offset: usize) -> Result<CpuStorage> {
        use cudarc::driver::DeviceSlice;

        match &self.slice {
            CudaStorageSlice::U8(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::U8(vec))
            }
            CudaStorageSlice::U32(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::U32(vec))
            }
            CudaStorageSlice::I64(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::I64(vec))
            }
            CudaStorageSlice::BF16(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::BF16(vec))
            }
            CudaStorageSlice::F16(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F16(vec))
            }
            CudaStorageSlice::F32(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F32(vec))
            }
            CudaStorageSlice::F64(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F64(vec))
            }
            CudaStorageSlice::F8E4M3(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F8E4M3(vec))
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }
}

fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<StridedBatchedConfig<T>> {
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use cudarc::cublas::sys::cublasOperation_t;

    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // The a tensor has dims batching, k, n (rhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };
    // The b tensor has dims batching, m, k (lhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };
    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (m * n) as i64,
    })
}

impl BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let (slice, out_backing) = Clone.map(self, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn dtype(&self) -> DType {
        match self.slice {
            CudaStorageSlice::U8(_) => DType::U8,
            CudaStorageSlice::U32(_) => DType::U32,
            CudaStorageSlice::I64(_) => DType::I64,
            CudaStorageSlice::BF16(_) => DType::BF16,
            CudaStorageSlice::F16(_) => DType::F16,
            CudaStorageSlice::F32(_) => DType::F32,
            CudaStorageSlice::F64(_) => DType::F64,
            CudaStorageSlice::F8E4M3(_) => DType::F8E4M3,
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        use crate::scalar::Scalar;
        use kernels::simple::fill::FillDType;

        let dev = &self.device;
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let src_o = layout.start_offset();
        let stream = dev.cuda_stream();

        // Prepare dims/strides info for non-contiguous tensors
        let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
            None
        } else {
            Some(dev.info_table(&[dims, layout.stride()].concat())?)
        };
        let info_ptr = match &info {
            Some(s) => {
                let (ptr, _guard) = s.device_ptr(&stream);
                ptr as *const usize
            }
            None => std::ptr::null(),
        };

        // Convert scalar to bits
        let value_bits = match s {
            Scalar::U8(v) => v as u64,
            Scalar::U32(v) => v as u64,
            Scalar::I64(v) => v as u64,
            Scalar::F32(v) => v.to_bits() as u64,
            Scalar::F64(v) => v.to_bits(),
            Scalar::F16(v) => v.to_bits() as u64,
            Scalar::BF16(v) => v.to_bits() as u64,
            Scalar::F8E4M3(v) => v.to_bits() as u64,
        };

        // Get dtype and output pointer
        let (dtype_i32, out_ptr): (i32, u64) = match &mut self.slice {
            S::U8(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::U8 as i32, ptr)
            }
            S::U32(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::U32 as i32, ptr)
            }
            S::I64(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::I64 as i32, ptr)
            }
            S::BF16(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::BF16 as i32, ptr)
            }
            S::F16(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F16 as i32, ptr)
            }
            S::F32(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F32 as i32, ptr)
            }
            S::F64(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F64 as i32, ptr)
            }
            S::F8E4M3(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F8E4M3 as i32, ptr)
            }
            S::Moved => S::unreachable_moved(),
        };

        // Keep info alive for the kernel call
        let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

        unsafe {
            kernels::simple::fill::run_const_set_op(
                dtype_i32,
                el_count,
                dims.len(),
                info_ptr,
                value_bits,
                out_ptr as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        use kernels::simple::cast::CastDType;

        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dev = self.device();
        let start_o = layout.start_offset();
        let stream = dev.cuda_stream();

        // Convert candle DType to CastDType
        let dtype_to_cast = |dt: DType| -> i32 {
            match dt {
                DType::F32 => CastDType::F32 as i32,
                DType::F64 => CastDType::F64 as i32,
                DType::U8 => CastDType::U8 as i32,
                DType::U32 => CastDType::U32 as i32,
                DType::I64 => CastDType::I64 as i32,
                DType::F16 => CastDType::F16 as i32,
                DType::BF16 => CastDType::BF16 as i32,
                DType::F8E4M3 => CastDType::F8E4M3 as i32,
            }
        };

        let src_dtype = self.dtype();
        let src_dtype_i32 = dtype_to_cast(src_dtype);
        let dst_dtype_i32 = dtype_to_cast(dtype);

        // Prepare dims/strides info for non-contiguous tensors
        let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
            None
        } else {
            Some(dev.info_table(&[dims, layout.stride()].concat())?)
        };

        // Use a helper macro to reduce repetition and properly scope the guards
        // Set by `cast_impl!` to whatever `alloc_inheriting` resolved, so the
        // result storage is stamped with the backing of the buffer it actually
        // got rather than an assumed one. Left uninitialised so the compiler
        // checks every arm assigns it — an arm that allocated without recording
        // its backing is exactly the drift this pairing exists to prevent.
        let out_backing;
        macro_rules! cast_impl {
            ($inp_slice:expr, $out_ty:ty, $wrapper:path) => {{
                let inp = $inp_slice.slice(start_o..);
                // A cast reading a wave-backed operand writes into the same
                // generation: the result is consumed within the phase that
                // produced its input, which is why inheriting is right here and
                // not merely permissible.
                let (out, backing) = unsafe { alloc_inheriting::<$out_ty>(dev, el, self.backing)? };
                out_backing = backing;
                {
                    let info_ptr = match &info {
                        Some(s) => {
                            let (ptr, _guard) = s.device_ptr(&stream);
                            ptr as *const usize
                        }
                        None => std::ptr::null(),
                    };
                    let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
                    let (out_ptr, _out_guard) = out.device_ptr(&stream);
                    // Keep info alive
                    let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));
                    unsafe {
                        kernels::simple::cast::run_cast(
                            src_dtype_i32,
                            dst_dtype_i32,
                            el,
                            dims.len(),
                            info_ptr,
                            inp_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
                $wrapper(out)
            }};
        }

        let slice = match (&self.slice, dtype) {
            (CudaStorageSlice::U8(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::U8(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::U8(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::U8(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::U8(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::U8(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::U8(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::U8(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::U32(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::U32(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::U32(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::U32(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::U32(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::U32(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::U32(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::U32(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::I64(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::I64(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::I64(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::I64(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::I64(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::I64(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::I64(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::I64(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::BF16(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::BF16(inp), DType::U32) => {
                cast_impl!(inp, u32, CudaStorageSlice::U32)
            }
            (CudaStorageSlice::BF16(inp), DType::I64) => {
                cast_impl!(inp, i64, CudaStorageSlice::I64)
            }
            (CudaStorageSlice::BF16(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::BF16(inp), DType::F16) => {
                cast_impl!(inp, f16, CudaStorageSlice::F16)
            }
            (CudaStorageSlice::BF16(inp), DType::F32) => {
                cast_impl!(inp, f32, CudaStorageSlice::F32)
            }
            (CudaStorageSlice::BF16(inp), DType::F64) => {
                cast_impl!(inp, f64, CudaStorageSlice::F64)
            }
            (CudaStorageSlice::BF16(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F16(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F16(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::F16(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::F16(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F16(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::F16(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::F16(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::F16(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F32(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F32(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::F32(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::F32(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F32(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::F32(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::F32(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::F32(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F64(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F64(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::F64(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::F64(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F64(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::F64(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::F64(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::F64(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F8E4M3(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F8E4M3(inp), DType::U32) => {
                cast_impl!(inp, u32, CudaStorageSlice::U32)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::I64) => {
                cast_impl!(inp, i64, CudaStorageSlice::I64)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F16) => {
                cast_impl!(inp, f16, CudaStorageSlice::F16)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F32) => {
                cast_impl!(inp, f32, CudaStorageSlice::F32)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F64) => {
                cast_impl!(inp, f64, CudaStorageSlice::F64)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }
            (CudaStorageSlice::Moved, _) => CudaStorageSlice::unreachable_moved(),
        };
        Ok(Self {
            slice,
            device: dev.clone(),
            backing: out_backing,
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let (slice, backing) =
            run_affine_ffi(&self.slice, &device, layout, mul, add, self.backing)?;
        Ok(Self {
            slice,
            device,
            backing,
        })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        use kernels::simple::unary::UnaryParamOp;
        let device = self.device().clone();
        let slice =
            run_unary_param_ffi(&self.slice, &device, layout, UnaryParamOp::Powf as i32, e)?;
        Ok(Self {
            slice,
            device,
            backing: Backing::Owned,
        })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        use kernels::simple::unary::UnaryParamOp;
        let device = self.device().clone();
        let slice = run_unary_param_ffi(
            &self.slice,
            &device,
            layout,
            UnaryParamOp::Elu as i32,
            alpha,
        )?;
        Ok(Self {
            slice,
            device,
            backing: Backing::Owned,
        })
    }

    fn sub_at_indices(&self, layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        // Delegate to the inherent method
        self.sub_at_indices(layout, indices, value)
    }

    fn div_at_indices(&self, layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        // Delegate to the inherent method
        self.div_at_indices(layout, indices, value)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = FastReduce(sum_dims, op).map(self, &device, layout)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Cmp(op).map(self, lhs_l, rhs, rhs_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = U::V.map(self, &device, layout)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = B::V.map(self, lhs_l, rhs, rhs_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn binary_inplace_impl(
        &mut self,
        op: crate::op::BinaryInplaceOp,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;

        // lhs must be contiguous for in-place operations
        if !lhs_l.is_contiguous() {
            return Err(CudaError::InternalError(
                "in-place binary op requires contiguous lhs tensor".to_string(),
            ))
            .w();
        }

        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let lhs_start = lhs_l.start_offset();
        let rhs_start = rhs_l.start_offset();
        let device = &self.device;
        let stream = device.cuda_stream();

        let dtype = self.dtype();
        let dtype_i32 = dtype_to_binary_dtype(dtype);
        let op_i32 = binary_inplace_op_to_ffi(op);

        // Check for min/max on integer types (not supported)
        let is_integer = matches!(dtype, DType::U8 | DType::U32 | DType::I64);
        if is_integer
            && matches!(
                op,
                crate::op::BinaryInplaceOp::Min | crate::op::BinaryInplaceOp::Max
            )
        {
            return Err(CudaError::InternalError(format!(
                "min/max in-place ops not supported for integer dtype {:?}",
                dtype
            )))
            .w();
        }

        // Prepare dims and strides info for non-contiguous rhs
        // Note: lhs is guaranteed contiguous, but rhs may not be
        let info: Option<Arc<Uploaded<usize>>> = if rhs_l.is_contiguous() {
            None
        } else {
            // Only need dims and rhs_strides since lhs is contiguous
            Some(device.info_table(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };

        // Get lhs slice with offset for mutable access
        let lhs_ptr = self.slice.device_ptr_mut(&stream)?;
        let lhs_ptr_offset = match dtype {
            DType::F32 => unsafe { (lhs_ptr as *mut f32).add(lhs_start) as *mut std::ffi::c_void },
            DType::F64 => unsafe { (lhs_ptr as *mut f64).add(lhs_start) as *mut std::ffi::c_void },
            DType::U8 => unsafe { (lhs_ptr as *mut u8).add(lhs_start) as *mut std::ffi::c_void },
            DType::U32 => unsafe { (lhs_ptr as *mut u32).add(lhs_start) as *mut std::ffi::c_void },
            DType::I64 => unsafe { (lhs_ptr as *mut i64).add(lhs_start) as *mut std::ffi::c_void },
            DType::F16 => unsafe {
                (lhs_ptr as *mut half::f16).add(lhs_start) as *mut std::ffi::c_void
            },
            DType::BF16 => unsafe {
                (lhs_ptr as *mut half::bf16).add(lhs_start) as *mut std::ffi::c_void
            },
            DType::F8E4M3 => unsafe {
                (lhs_ptr as *mut float8::F8E4M3).add(lhs_start) as *mut std::ffi::c_void
            },
        };

        // Get rhs base pointer and offset it
        // We get the base pointer and manually offset it to avoid lifetime issues
        let rhs_dtype = rhs.dtype();
        let (rhs_base_ptr, _rhs_guard): (*const std::ffi::c_void, _) = match &rhs.slice {
            CudaStorageSlice::F32(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::F64(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::U8(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::U32(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::I64(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::F16(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::BF16(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::F8E4M3(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        };

        // Offset rhs pointer
        let rhs_ptr = match rhs_dtype {
            DType::F32 => unsafe {
                (rhs_base_ptr as *const f32).add(rhs_start) as *const std::ffi::c_void
            },
            DType::F64 => unsafe {
                (rhs_base_ptr as *const f64).add(rhs_start) as *const std::ffi::c_void
            },
            DType::U8 => unsafe {
                (rhs_base_ptr as *const u8).add(rhs_start) as *const std::ffi::c_void
            },
            DType::U32 => unsafe {
                (rhs_base_ptr as *const u32).add(rhs_start) as *const std::ffi::c_void
            },
            DType::I64 => unsafe {
                (rhs_base_ptr as *const i64).add(rhs_start) as *const std::ffi::c_void
            },
            DType::F16 => unsafe {
                (rhs_base_ptr as *const half::f16).add(rhs_start) as *const std::ffi::c_void
            },
            DType::BF16 => unsafe {
                (rhs_base_ptr as *const half::bf16).add(rhs_start) as *const std::ffi::c_void
            },
            DType::F8E4M3 => unsafe {
                (rhs_base_ptr as *const float8::F8E4M3).add(rhs_start) as *const std::ffi::c_void
            },
        };

        {
            let info_ptr = match &info {
                Some(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    ptr as *const usize
                }
                None => std::ptr::null(),
            };

            // Keep info guard alive for the kernel call
            let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

            // Call the FFI dispatcher - guards are still alive here
            unsafe {
                kernels::simple::binary::run_binary_inplace_op(
                    op_i32,
                    dtype_i32,
                    elem_count,
                    dims.len(),
                    info_ptr,
                    lhs_ptr_offset,
                    rhs_ptr,
                );
            }
        }

        Ok(())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            CudaStorageSlice::U8(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::U8(cpu_storage))
            }
            CudaStorageSlice::U32(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::U32(cpu_storage))
            }
            CudaStorageSlice::I64(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::I64(cpu_storage))
            }
            CudaStorageSlice::BF16(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::BF16(cpu_storage))
            }
            CudaStorageSlice::F16(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F16(cpu_storage))
            }
            CudaStorageSlice::F32(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F32(cpu_storage))
            }
            CudaStorageSlice::F64(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F64(cpu_storage))
            }
            CudaStorageSlice::F8E4M3(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F8E4M3(cpu_storage))
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = WhereCond(self, layout).map(t, t_l, f, f_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    #[cfg(not(feature = "cudnn"))]
    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        const USE_IM2COL_CONV1D: bool = true;

        let device = self.device().clone();
        if !USE_IM2COL_CONV1D {
            let (slice, out_backing) = Conv1D(params).map(self, l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
            });
        }

        let (col, col_backing) = Im2Col1D {
            l_k: params.k_size,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(self, &device, l)?;
        let col = Self {
            slice: col,
            device,
            backing: col_backing,
        };
        let l_out = params.l_out();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b * m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    #[cfg(feature = "cudnn")]
    fn conv1d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let device = self.device().clone();
        if !kernel_l.is_contiguous() {
            let (slice, out_backing) =
                Conv1D(params).map(self, inp_l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
            });
        }
        let l_out = params.l_out();
        let dst_el = params.c_out * l_out * params.b_size;
        let slice = match (&self.slice, &kernel.slice) {
            (S::U8(inp), S::U8(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<u8>(dst_el)? };
                crate::cudnn::launch_conv1d::<u8, u8>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::U8(out)
            }
            (S::BF16(inp), S::BF16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<bf16>(dst_el)? };
                // Only PSEUDO_BFLOAT16_CONFIG is supported in cudnn, there is no "true bfloat16"
                // version.
                // https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#id88
                crate::cudnn::launch_conv1d::<bf16, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::BF16(out)
            }
            (S::F16(inp), S::F16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f16>(dst_el)? };
                crate::cudnn::launch_conv1d::<f16, f16>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F16(out)
            }
            (S::F32(inp), S::F32(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f32>(dst_el)? };
                crate::cudnn::launch_conv1d::<f32, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F32(out)
            }
            (S::F64(inp), S::F64(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f64>(dst_el)? };
                crate::cudnn::launch_conv1d::<f64, f64>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F64(out)
            }
            (S::U32(_), S::U32(_)) => Err(CudaError::InternalError(
                "conv1d does not support u32".to_string(),
            ))?,
            (S::I64(_), S::I64(_)) => Err(CudaError::InternalError(
                "conv1d does not support i64".to_string(),
            ))?,
            _ => Err(CudaError::InternalError(
                "dtype mismatch in conv1d".to_string(),
            ))?,
        };
        Ok(Self {
            slice,
            device,
            backing: Backing::Owned,
        })
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        const USE_COL2IM_CONV1D_TR: bool = true;

        let device = self.device().clone();
        let can_use_col2im = kernel_l.is_contiguous()
            && params.dilation == 1
            && params.padding == 0
            && params.output_padding == 0;
        let slice = if USE_COL2IM_CONV1D_TR && can_use_col2im {
            let (b_size, c_in, l_in) = l.shape().dims3()?;
            let (c_in2, c_out, k_size) = kernel_l.shape().dims3()?;
            if !kernel_l.is_contiguous() {
                crate::bail!(
                    "convtr1d: the second argument (kernel) has to be contiguous {kernel_l:?}"
                )
            }
            if c_in != c_in2 {
                crate::bail!(
                    "convtr1d: shape mismatch on c_in {:?} {:?}",
                    l.shape(),
                    kernel_l.shape()
                )
            }
            let col = {
                // This merges the last two dimensions of the kernel together.
                let kernel_l_mm = Layout::new(
                    (b_size, c_in, k_size * c_out).into(),
                    vec![0, k_size * c_out, 1],
                    kernel_l.start_offset(),
                );
                self.matmul(
                    kernel,
                    (
                        b_size,
                        /* m */ l_in,
                        /* n */ c_out * k_size,
                        /* k */ c_in,
                    ),
                    &l.transpose(1, 2)?,
                    &kernel_l_mm,
                )?
            };
            let col_l = Layout::contiguous((b_size, l_in, c_out, k_size));
            Col2Im1D {
                stride: params.stride,
            }
            .map(&col, &device, &col_l)?
        } else {
            ConvTranspose1D(params).map(self, l, kernel, kernel_l, &device)?
        };
        Ok(Self {
            slice: slice.0,
            device,
            backing: slice.1,
        })
    }

    #[cfg(not(feature = "cudnn"))]
    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        const USE_IM2COL_CONV2D: bool = true;

        let device = self.device().clone();
        if !USE_IM2COL_CONV2D {
            let (slice, out_backing) = Conv2D(params).map(self, l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
            });
        }

        let (col, col_backing) = Im2Col {
            h_k: params.k_h,
            w_k: params.k_w,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(self, &device, l)?;
        let col = Self {
            slice: col,
            device,
            backing: col_backing,
        };
        let h_out = params.out_h();
        let w_out = params.out_w();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_h * params.k_w * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b * m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    #[cfg(feature = "cudnn")]
    fn conv2d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        if !kernel_l.is_contiguous() {
            let (slice, out_backing) =
                Conv2D(params).map(self, inp_l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
            });
        }
        let (out_w, out_h) = (params.out_w(), params.out_h());
        let dst_el = params.c_out * out_w * out_h * params.b_size;
        let slice = match (&self.slice, &kernel.slice) {
            (S::U8(inp), S::U8(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<u8>(dst_el)? };
                crate::cudnn::launch_conv2d::<u8, u8>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::U8(out)
            }
            (S::BF16(inp), S::BF16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<bf16>(dst_el)? };
                // Only PSEUDO_BFLOAT16_CONFIG is supported in cudnn, there is no "true bfloat16"
                // version.
                // https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#id88
                crate::cudnn::launch_conv2d::<bf16, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::BF16(out)
            }
            (S::F16(inp), S::F16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f16>(dst_el)? };
                crate::cudnn::launch_conv2d::<f16, f16>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F16(out)
            }
            (S::F32(inp), S::F32(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f32>(dst_el)? };
                crate::cudnn::launch_conv2d::<f32, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F32(out)
            }
            (S::F64(inp), S::F64(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f64>(dst_el)? };
                crate::cudnn::launch_conv2d::<f64, f64>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F64(out)
            }
            (S::U32(_), S::U32(_)) => Err(CudaError::InternalError(
                "conv2d does not support u32".to_string(),
            ))?,
            (S::I64(_), S::I64(_)) => Err(CudaError::InternalError(
                "conv2d does not support i64".to_string(),
            ))?,
            _ => Err(CudaError::InternalError(
                "dtype mismatch in conv2d".to_string(),
            ))?,
        };
        Ok(Self {
            slice,
            device,
            backing: Backing::Owned,
        })
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) =
            ConvTranspose2D(params).map(self, l, kernel, kernel_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Avg,
        }
        .map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Max,
        }
        .map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn upsample_nearest1d(&self, _: &Layout, _out_sz: usize) -> Result<Self> {
        crate::bail!("upsample-nearest1d is not supported on cuda")
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = UpsampleNearest2D(out_w, out_h).map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = IndexSelect(ids, ids_l, dim).map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }
    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Gather(ids, ids_l, dim).map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }
    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        Scatter(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        ScatterAdd(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let device = self.device().clone();
        let mut acc = unsafe { device.alloc_uninit(l.shape(), self.dtype())? };
        self.copy_strided_src(&mut acc, 0, l)?;
        IndexAdd(ids, ids_l, dim).map(&mut acc.slice, l, &src.slice, src_l, &device)?;
        Ok(acc)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;

        // **A degenerate GEMM never reaches cuBLAS.**
        //
        // With any of `b`, `m`, `n` zero there is no output element to compute,
        // and with `k == 0` every output is the empty sum. cuBLAS handles
        // neither: it derives its grid from the problem shape, so a zero
        // dimension launches `(0,1,1)` and the runtime rejects it with
        // `cudaErrorInvalidConfiguration`. compute-sanitizer counted **41** such
        // launches in one short run, alongside the `cudaGetLastError` that then
        // picks the sticky error up somewhere unrelated — a real error planted
        // in the runtime's state by a call that had no work to do.
        //
        // These shapes are reachable in ordinary operation, not just in tests: a
        // wave with no rows for a quantum (the deferred-glue drain runs
        // `decode_forward_cobatched(&[], &[], &[], &[])`) and a grouped GEMM
        // whose group came out empty both arrive here with a zero extent.
        //
        // The allocation still happens — `alloc_zeros` for `elem_count`, which
        // is a 0-byte request in the common case and still shows up in the
        // forbidden-allocation report as its own call site. What is avoided is
        // the cuBLAS call, and that is the whole point; a 0-byte allocation is
        // cheap and wrong to elide, because the result must be a real
        // `CudaStorage` for the caller to go on using.
        //
        // The result is pool-backed (`Backing::Owned`) rather than inheriting
        // the activation's arena the way the live path below does. That is
        // honest rather than convenient — these bytes genuinely come from the
        // pool — and it costs nothing, because a zero-extent result has no
        // elements for a later `empty_beside` to inherit a span from.
        if elem_count == 0 || k == 0 {
            // `k == 0` with real output elements is an empty SUM, which is zero
            // — the value cuBLAS itself would have written with `beta = 0`. It
            // has to be materialised rather than left uninitialised, so this is
            // one of the few places `alloc_zeros` is right (CLAUDE.md invariant
            // 6 excepts buffers read before they are written, and every element
            // here is read without any kernel writing it).
            let slice = match (&self.slice, &rhs.slice) {
                (CudaStorageSlice::BF16(_), CudaStorageSlice::BF16(_)) => {
                    CudaStorageSlice::BF16(dev.alloc_zeros::<bf16>(elem_count)?)
                }
                (CudaStorageSlice::F16(_), CudaStorageSlice::F16(_)) => {
                    CudaStorageSlice::F16(dev.alloc_zeros::<f16>(elem_count)?)
                }
                (CudaStorageSlice::F32(_), CudaStorageSlice::F32(_)) => {
                    CudaStorageSlice::F32(dev.alloc_zeros::<f32>(elem_count)?)
                }
                (CudaStorageSlice::F64(_), CudaStorageSlice::F64(_)) => {
                    CudaStorageSlice::F64(dev.alloc_zeros::<f64>(elem_count)?)
                }
                _ => {
                    return Err(
                        CudaError::InternalError("dtype mismatch in matmul op".to_string()).into(),
                    )
                }
            };
            return Ok(Self {
                slice,
                device: dev.clone(),
                backing: Backing::Owned,
            });
        }

        // The activation's arena, not the weight's: `self` is the left operand,
        // which for every `x @ W` in a forward is the value flowing through the
        // layer, while `rhs` is a model parameter that names no arena. Assigned
        // by each arm from what `alloc_inheriting` resolved, so a full span
        // falls back to the pool rather than failing.
        let inherit = self.backing;
        let out_backing;
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::BF16(lhs), CudaStorageSlice::BF16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(bf16::ONE, bf16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<bf16>(dev, elem_count, inherit)? };
                out_backing = resolved;
                // Check 16-byte alignment: bf16 is 2 bytes, so offset must be multiple of 8 elements
                // CUDA malloc guarantees 256-byte aligned base, output is fresh allocation
                let known_aligned = (lhs_l.start_offset() * 2).is_multiple_of(16)
                    && (rhs_l.start_offset() * 2).is_multiple_of(16);
                unsafe {
                    gemm_strided_batched_bf16(
                        &self.device.blas,
                        cfg,
                        rhs,
                        lhs,
                        &mut out,
                        known_aligned,
                    )
                }
                .w()?;
                CudaStorageSlice::BF16(out)
            }
            (CudaStorageSlice::F16(lhs), CudaStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(f16::ONE, f16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<f16>(dev, elem_count, inherit)? };
                out_backing = resolved;
                // Check 16-byte alignment: f16 is 2 bytes, so offset must be multiple of 8 elements
                let known_aligned = (lhs_l.start_offset() * 2).is_multiple_of(16)
                    && (rhs_l.start_offset() * 2).is_multiple_of(16);
                unsafe {
                    gemm_strided_batched_f16(
                        &self.device.blas,
                        cfg,
                        rhs,
                        lhs,
                        &mut out,
                        known_aligned,
                    )
                }
                .w()?;
                CudaStorageSlice::F16(out)
            }
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<f32>(dev, elem_count, inherit)? };
                out_backing = resolved;
                // Check 16-byte alignment: f32 is 4 bytes, so offset must be multiple of 4 elements
                let known_aligned = (lhs_l.start_offset() * 4).is_multiple_of(16)
                    && (rhs_l.start_offset() * 4).is_multiple_of(16);
                unsafe {
                    gemm_strided_batched_f32(
                        &self.device.blas,
                        cfg,
                        rhs,
                        lhs,
                        &mut out,
                        known_aligned,
                    )
                }
                .w()?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<f64>(dev, elem_count, inherit)? };
                out_backing = resolved;
                unsafe {
                    self.device
                        .blas
                        .gemm_strided_batched(cfg, rhs, lhs, &mut out)
                }
                .w()?;
                CudaStorageSlice::F64(out)
            }
            _ => {
                return Err(
                    CudaError::InternalError("dtype mismatch in matmul op".to_string()).into(),
                )
            }
        };
        let device = dev.clone();
        Ok(Self {
            slice,
            device,
            backing: out_backing,
        })
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        use kernels::simple::fill::FillDType;

        let dev = &self.device;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = dst_s as u32;
        let src_s = src_s as u32;
        let stream = dev.cuda_stream();

        match (&self.slice, &mut dst.slice) {
            (S::U8(s), S::U8(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::U8 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            (S::U32(s), S::U32(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::U32 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            (S::I64(s), S::I64(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::I64 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            (S::BF16(s), S::BF16(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::BF16 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            (S::F16(s), S::F16(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F16 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            (S::F32(s), S::F32(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F32 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            (S::F64(s), S::F64(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F64 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            (S::F8E4M3(s), S::F8E4M3(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F8E4M3 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                    );
                }
                Ok(())
            }
            _ => Err(CudaError::InternalError(
                "dtype mismatch in copy2d".to_string(),
            ))?,
        }
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        let dev = &self.device;

        // Helper macro for FFI-based strided copy
        macro_rules! copy_strided_ffi {
            ($src:expr, $dst:expr, $dtype:expr) => {{
                let (src, mut dst) = slice_src_and_dst($src, src_l, $dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    use kernels::simple::unary::UnaryOp;
                    let stream = dev.cuda_stream();

                    // Prepare dims/strides info for non-contiguous tensors
                    let info = dev.info_table(&[dims, src_l.stride()].concat())?;
                    let (info_ptr, _info_guard) = info.device_ptr(&stream);

                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

                    unsafe {
                        kernels::simple::unary::run_unary_op(
                            UnaryOp::Copy as i32,
                            $dtype,
                            el_count,
                            dims.len(),
                            info_ptr as *const usize,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
            }};
        }

        use kernels::simple::unary::UnaryDType;
        match (&self.slice, &mut dst.slice) {
            (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::BF16 as i32);
            }
            (CudaStorageSlice::F16(src), CudaStorageSlice::F16(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F16 as i32);
            }
            (CudaStorageSlice::F32(src), CudaStorageSlice::F32(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F32 as i32);
            }
            (CudaStorageSlice::F8E4M3(src), CudaStorageSlice::F8E4M3(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F8E4M3 as i32);
            }
            (CudaStorageSlice::U8(src), CudaStorageSlice::U8(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::U8 as i32);
            }
            (CudaStorageSlice::U32(src), CudaStorageSlice::U32(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::U32 as i32);
            }
            (CudaStorageSlice::I64(src), CudaStorageSlice::I64(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::I64 as i32);
            }
            (CudaStorageSlice::F64(src), CudaStorageSlice::F64(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F64 as i32);
            }
            _ => Err(CudaError::InternalError(
                "dtype mismatch in copy_strided op".to_string(),
            ))?,
        }
        Ok(())
    }
}

// Default for the reduced precision setting is false, similar to pytorch.
// https://github.com/pytorch/pytorch/issues/123157
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

unsafe fn gemm_strided_batched_f32(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<f32>,
    a: &cudarc::driver::CudaView<f32>,
    b: &cudarc::driver::CudaView<f32>,
    c: &mut CudaSlice<f32>,
    known_aligned: bool, // True when all inputs have start_offset==0 (fresh CUDA allocations are 256-byte aligned)
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let alpha = &cfg.gemm.alpha as *const f32 as *const _;
    let beta = &cfg.gemm.beta as *const f32 as *const _;

    let stream = c.stream().clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);

    // Determine alignment for tensor core eligibility.
    // Fast path: if caller knows all operands are from fresh allocations (start_offset==0),
    // CUDA guarantees 256-byte alignment, so skip the runtime pointer checks.
    // Slow path: check 16-byte alignment at runtime for sliced/narrowed tensors.
    let all_aligned = known_aligned || {
        let a_aligned = (a as usize).is_multiple_of(16);
        let b_aligned = (b as usize).is_multiple_of(16);
        let c_aligned = (c as usize).is_multiple_of(16);
        a_aligned && b_aligned && c_aligned
    };

    let (compute_type, algo) = if all_aligned && gemm_reduced_precision_f32() {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    } else {
        // Fall back to standard F32 compute without tensor cores
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        alpha,
        a as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.lda,
        cfg.stride_a,
        b as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.ldb,
        cfg.stride_b,
        beta,
        c as *mut _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
        compute_type,
        algo,
    )
}

unsafe fn gemm_strided_batched_f16(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<f16>,
    a: &cudarc::driver::CudaView<f16>,
    b: &cudarc::driver::CudaView<f16>,
    c: &mut CudaSlice<f16>,
    known_aligned: bool, // True when all inputs have start_offset==0 (fresh CUDA allocations are 256-byte aligned)
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let alpha = cfg.gemm.alpha;
    let beta = cfg.gemm.beta;
    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    let (compute_type, alpha, beta) = if gemm_reduced_precision_f16() {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            (&alpha) as *const f16 as *const _,
            (&beta) as *const f16 as *const _,
        )
    } else {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    let stream = c.stream().clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);

    // Determine alignment for tensor core eligibility.
    // Fast path: if caller knows all operands are from fresh allocations (start_offset==0),
    // CUDA guarantees 256-byte alignment, so skip the runtime pointer checks.
    // Slow path: check 16-byte alignment at runtime for sliced/narrowed tensors.
    let all_aligned = known_aligned || {
        let a_aligned = (a as usize).is_multiple_of(16);
        let b_aligned = (b as usize).is_multiple_of(16);
        let c_aligned = (c as usize).is_multiple_of(16);
        a_aligned && b_aligned && c_aligned
    };

    let algo = if all_aligned {
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP
    } else {
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT
    };

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        alpha,
        a as *const _,
        sys::cudaDataType_t::CUDA_R_16F,
        cfg.gemm.lda,
        cfg.stride_a,
        b as *const _,
        sys::cudaDataType_t::CUDA_R_16F,
        cfg.gemm.ldb,
        cfg.stride_b,
        beta,
        c as *mut _,
        sys::cudaDataType_t::CUDA_R_16F,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
        compute_type,
        algo,
    )
}

unsafe fn gemm_strided_batched_bf16(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<bf16>,
    a: &cudarc::driver::CudaView<bf16>,
    b: &cudarc::driver::CudaView<bf16>,
    c: &mut CudaSlice<bf16>,
    known_aligned: bool, // True when all inputs have start_offset==0 (fresh CUDA allocations are 256-byte aligned)
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    // The type for alpha and beta depends on the computeType.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
    let (compute_type, alpha, beta) = if gemm_reduced_precision_bf16() {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    } else {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    let stream = c.stream().clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);

    // Determine alignment for tensor core eligibility.
    // Fast path: if caller knows all operands are from fresh allocations (start_offset==0),
    // CUDA guarantees 256-byte alignment, so skip the runtime pointer checks.
    // Slow path: check 16-byte alignment at runtime for sliced/narrowed tensors.
    let all_aligned = known_aligned || {
        let a_aligned = (a as usize).is_multiple_of(16);
        let b_aligned = (b as usize).is_multiple_of(16);
        let c_aligned = (c as usize).is_multiple_of(16);
        a_aligned && b_aligned && c_aligned
    };

    let algo = if all_aligned {
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP
    } else {
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT
    };

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        alpha,
        a as *const _,
        sys::cudaDataType_t::CUDA_R_16BF,
        cfg.gemm.lda,
        cfg.stride_a,
        b as *const _,
        sys::cudaDataType_t::CUDA_R_16BF,
        cfg.gemm.ldb,
        cfg.stride_b,
        beta,
        c as *mut _,
        sys::cudaDataType_t::CUDA_R_16BF,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
        compute_type,
        algo,
    )
}
