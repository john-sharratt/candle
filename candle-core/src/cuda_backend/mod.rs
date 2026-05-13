//! Implementation of Backend traits for CUDA device
//!
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
pub use candle_kernels as kernels;
pub use cudarc;

// ── Kernel breadcrumb ─────────────────────────────────────────────────────────
//
// Written immediately before every kernel FFI call (one thread-local store,
// ~1 ns).  The panic hook reads it when a CUDA DriverError surfaces — even
// though CUDA errors are asynchronous, the last breadcrumb on the scheduler
// thread is almost always the kernel that triggered the fault.
//
// Usage: call `cuda_breadcrumb!("run_foo")` at the top of each wrapper that
// dispatches to a kernel.  The macro captures `file!()` / `line!()` at the
// actual call site so the recorded location is meaningful.

thread_local! {
    static LAST_KERNEL_LAUNCH: std::cell::RefCell<(&'static str, &'static str, u32)> =
        const { std::cell::RefCell::new(("", "", 0)) };
}

/// Set the breadcrumb for the current thread's last kernel launch.
/// Called via the `cuda_breadcrumb!` macro before each kernel FFI call.
#[inline(always)]
pub fn set_kernel_breadcrumb(name: &'static str, file: &'static str, line: u32) {
    LAST_KERNEL_LAUNCH.with(|k| *k.borrow_mut() = (name, file, line));
}

/// Return a human-readable description of the last kernel launched on this
/// thread.  Called from the panic hook installed in the binary.
pub fn last_cuda_kernel_launch() -> String {
    LAST_KERNEL_LAUNCH.with(|k| {
        let (name, file, line) = *k.borrow();
        if name.is_empty() {
            "(no kernel recorded on this thread)".to_string()
        } else {
            format!("'{name}' ({file}:{line})")
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

#[cfg(feature = "cudnn")]
pub mod cudnn;
mod device;
mod error;
mod utils;
pub use device::{CudaDevice, DeviceId};
pub use error::{CudaError, WrapErr};
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

pub enum SlicePtrOrNull<T> {
    Ptr(CudaSlice<T>),
    Null,
}

impl<T: DeviceRepr> SlicePtrOrNull<T> {
    pub fn builder_arg<'a, 'b: 'a>(&'b self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        match self {
            SlicePtrOrNull::Ptr(slice) => builder.arg(slice),
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
    pub fn params_from_layout(dev: &CudaDevice, l: &Layout) -> Result<Self> {
        let ds = if l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            SlicePtrOrNull::Ptr(dev.memcpy_stod(&[l.dims(), l.stride()].concat())?)
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
}

impl CudaStorageSlice {
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
    ) -> Result<CudaSlice<T>> {
        s.try_clone().w()
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
fn run_affine_ffi(
    src: &CudaStorageSlice,
    dev: &CudaDevice,
    layout: &Layout,
    mul: f64,
    add: f64,
) -> Result<CudaStorageSlice> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el = shape.elem_count();
    let start_offset = layout.start_offset();
    let stream = dev.cuda_stream();

    // Prepare dims/strides info for non-contiguous tensors
    let info: Option<CudaSlice<usize>> = if layout.is_contiguous() {
        None
    } else {
        Some(dev.memcpy_stod(&[dims, layout.stride()].concat())?)
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
        CudaStorageSlice::U8(_) => DType::U8,
        CudaStorageSlice::U32(_) => DType::U32,
        CudaStorageSlice::I64(_) => DType::I64,
    };
    let dtype_i32 = dtype_to_affine_dtype(dtype);

    // Execute based on dtype - allocate output and call FFI
    macro_rules! affine_impl {
        ($slice:expr, $dtype_variant:ident) => {{
            let src_slice = $slice.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let out = unsafe { dev.alloc(el)? };
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
        CudaStorageSlice::U8(s) => affine_impl!(s, U8),
        CudaStorageSlice::U32(s) => affine_impl!(s, U32),
        CudaStorageSlice::I64(s) => affine_impl!(s, I64),
    };

    Ok(out)
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
    let info: Option<CudaSlice<usize>> = if layout.is_contiguous() {
        None
    } else {
        Some(dev.memcpy_stod(&[dims, layout.stride()].concat())?)
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
    ) -> Result<CudaSlice<T>> {
        cuda_breadcrumb!("run_im2col1d");
        let shape = layout.shape();
        let dims = shape.dims();
        let l_out = self.l_out(dims[2]);
        let threads = dims[0] * l_out * dims[1];
        let ds = dev.memcpy_stod(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("im2col1d not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        // SAFETY: Set later by running the kernel.
        let dst = unsafe { dev.alloc::<T>(threads * self.l_k)? };
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
        Ok(dst)
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
    ) -> Result<CudaSlice<T>> {
        cuda_breadcrumb!("run_im2col");
        let shape = layout.shape();
        let dims = shape.dims();
        let (h_out, w_out) = self.hw_out(dims[2], dims[3]);
        let dst_el = dims[0] * h_out * w_out * dims[1] * self.h_k * self.w_k;
        let ds = dev.memcpy_stod(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("im2col not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        // SAFETY: Set later by running the kernel.
        let dst = unsafe { dev.alloc::<T>(dst_el)? };
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
        Ok(dst)
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
    ) -> Result<S> {
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
        let ds = dev.memcpy_stod(&[dims.as_slice(), stride.as_slice()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        if return_index {
            use kernels::simple::reduce::FastArgReduceOp;
            let op = match self.1 {
                ReduceOp::ArgMin => FastArgReduceOp::ArgMin as i32,
                ReduceOp::ArgMax => FastArgReduceOp::ArgMax as i32,
                _ => unreachable!(),
            };
            // SAFETY: filled in by the follow up kernel.
            let out = unsafe { dev.alloc::<u32>(dst_el)? };
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
            Ok(S::U32(out))
        } else {
            use kernels::simple::reduce::FastReduceOp;
            let op = match self.1 {
                ReduceOp::Sum => FastReduceOp::Sum as i32,
                ReduceOp::Min => FastReduceOp::Min as i32,
                ReduceOp::Max => FastReduceOp::Max as i32,
                _ => unreachable!(),
            };
            // SAFETY: filled in by the follow up kernel.
            let out = unsafe { dev.alloc::<T>(dst_el)? };
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
            Ok(wrap(out))
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
    ) -> Result<CudaSlice<T>> {
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
            let info: Option<CudaSlice<usize>> = if layout.is_contiguous() {
                None
            } else {
                Some(dev.memcpy_stod(&[dims, layout.stride()].concat())?)
            };

            let src_slice = &src.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let out = unsafe { dev.alloc::<T>(el_count)? };
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
            return Ok(out);
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
    ) -> Result<CudaSlice<T>> {
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
        let ds = dev.memcpy_stod(&[ids_dims, ids_l.stride()].concat())?;
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
        let out = unsafe { dev.alloc::<T>(dst_el)? };
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
        Ok(out)
    }
}

struct Gather<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map1 for Gather<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        src_l: &Layout,
    ) -> Result<CudaSlice<T>> {
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
        let out = unsafe { dev.alloc::<T>(el)? };
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
        Ok(out)
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
    ) -> Result<CudaSlice<T>> {
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
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else if dims.len() == 2 {
            [&[1], dims, &[1], inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv1d {dims:?}")
        };
        let ds = dev.memcpy_stod(&ds)?;

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
        Ok(out)
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
    ) -> Result<CudaSlice<T>> {
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
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv2d {dims:?}")
        };
        let ds = dev.memcpy_stod(&ds)?;

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
        Ok(out)
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
    ) -> Result<CudaSlice<T>> {
        cuda_breadcrumb!("run_col2im1d");
        let (b_size, l_in, c_out, k_size) = l.shape().dims4()?;
        let stride = self.stride;
        let l_out = (l_in - 1) * stride + k_size;
        let dst_el = b_size * c_out * l_out;
        let im = unsafe { dev.alloc::<T>(dst_el)? };

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
        Ok(im)
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
    ) -> Result<CudaSlice<T>> {
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
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose1d {dims:?}")
        };
        let ds = dev.memcpy_stod(&ds)?;

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
        Ok(out)
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
    ) -> Result<CudaSlice<T>> {
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
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose2d {dims:?}")
        };
        let ds = dev.memcpy_stod(&ds)?;

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
        Ok(out)
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
    ) -> Result<CudaSlice<T>> {
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
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = dev.memcpy_stod(&ds)?;

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
        Ok(out)
    }
}

struct UpsampleNearest2D(usize, usize);
impl Map1 for UpsampleNearest2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        dev: &CudaDevice,
        inp_l: &Layout,
    ) -> Result<CudaSlice<T>> {
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
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = dev.memcpy_stod(&ds)?;
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
        Ok(out)
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
    ) -> Result<CudaSlice<T>> {
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
        let ds = dev
            .memcpy_stod(&[dims, ids_l.stride(), layout_t.stride(), layout_f.stride()].concat())?;
        let t = &t.slice(layout_t.start_offset()..);
        let f = &f.slice(layout_f.start_offset()..);

        // Get data dtype for FFI
        let data_dtype = dtype_to_where_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
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
        Ok(out)
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
    ) -> Result<CudaSlice<T>> {
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
            let info: Option<CudaSlice<usize>> = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
                None
            } else {
                Some(dev.memcpy_stod(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
            };

            let lhs_slice = &lhs.slice(lhs_start..);
            let rhs_slice = &rhs.slice(rhs_start..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let out = unsafe { dev.alloc::<T>(elem_count)? };
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
            return Ok(out);
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
    ) -> Result<S> {
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
        let info: Option<CudaSlice<usize>> = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            None
        } else {
            Some(dev.memcpy_stod(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };

        let lhs_slice = &lhs.slice(lhs_start..);
        let rhs_slice = &rhs.slice(rhs_start..);
        // SAFETY: Allocated memory will be initialized by the kernel
        let out = unsafe { dev.alloc::<u8>(elem_count)? };
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
        Ok(S::U8(out))
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
}

pub trait CudaDType: Sized {
    fn as_cuda_slice(s: &CudaStorage) -> Result<&CudaSlice<Self>>;
    fn as_cuda_slice_mut(s: &mut CudaStorage) -> Result<&mut CudaSlice<Self>>;
    fn wrap_cuda_slice(s: CudaSlice<Self>, dev: CudaDevice) -> CudaStorage;
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
                CudaStorage { slice, device }
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

impl CudaStorage {
    pub fn wrap_cuda_slice<T: CudaDType>(slice: CudaSlice<T>, device: CudaDevice) -> CudaStorage {
        T::wrap_cuda_slice(slice, device)
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
            };
            return Ok(Self { slice, device });
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
            },
            device,
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
        self.slice = self.reinterpret_slice_as(dtype, elem_count)?;

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
    fn reinterpret_slice_as(&self, dtype: DType, elem_count: usize) -> Result<CudaStorageSlice> {
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
        };

        // Allocate destination buffer of the correct type and copy raw bytes
        macro_rules! alloc_and_copy {
            ($ty:ty, $wrapper:path) => {{
                use cudarc::driver::DevicePtrMut;
                let mut dst = unsafe { dev.alloc::<$ty>(elem_count)? };
                let (dst_ptr, _) = dst.device_ptr_mut(&stream);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_sync(dst_ptr, src_ptr, byte_count).w()?;
                }
                Ok($wrapper(dst))
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
            };
            return Ok(Self { slice, device });
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
            },
            device,
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
        let slice = Clone.map(&self.slice, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self { slice, device })
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
        let info: Option<CudaSlice<usize>> = if layout.is_contiguous() {
            None
        } else {
            Some(dev.memcpy_stod(&[dims, layout.stride()].concat())?)
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
        let info: Option<CudaSlice<usize>> = if layout.is_contiguous() {
            None
        } else {
            Some(dev.memcpy_stod(&[dims, layout.stride()].concat())?)
        };

        // Use a helper macro to reduce repetition and properly scope the guards
        macro_rules! cast_impl {
            ($inp_slice:expr, $out_ty:ty, $wrapper:path) => {{
                let inp = $inp_slice.slice(start_o..);
                let out = unsafe { dev.alloc::<$out_ty>(el)? };
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
        };
        Ok(Self {
            slice,
            device: dev.clone(),
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = run_affine_ffi(&self.slice, &device, layout, mul, add)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        use kernels::simple::unary::UnaryParamOp;
        let device = self.device().clone();
        let slice =
            run_unary_param_ffi(&self.slice, &device, layout, UnaryParamOp::Powf as i32, e)?;
        Ok(Self { slice, device })
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
        Ok(Self { slice, device })
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
        let slice = FastReduce(sum_dims, op).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = Cmp(op).map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = U::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = B::V.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
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
        let info: Option<CudaSlice<usize>> = if rhs_l.is_contiguous() {
            None
        } else {
            // Only need dims and rhs_strides since lhs is contiguous
            Some(device.memcpy_stod(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
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
        let slice = WhereCond(self, layout).map(&t.slice, t_l, &f.slice, f_l, &device)?;
        Ok(Self { slice, device })
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
            let slice = Conv1D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
            return Ok(Self { slice, device });
        }

        let col = Im2Col1D {
            l_k: params.k_size,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(&self.slice, &device, l)?;
        let col = Self { slice: col, device };
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
            let slice = Conv1D(params).map(&self.slice, inp_l, &kernel.slice, kernel_l, &device)?;
            return Ok(Self { slice, device });
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
        Ok(Self { slice, device })
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
            .map(&col.slice, &device, &col_l)?
        } else {
            ConvTranspose1D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?
        };
        Ok(Self { slice, device })
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
            let slice = Conv2D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
            return Ok(Self { slice, device });
        }

        let col = Im2Col {
            h_k: params.k_h,
            w_k: params.k_w,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(&self.slice, &device, l)?;
        let col = Self { slice: col, device };
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
            let slice = Conv2D(params).map(&self.slice, inp_l, &kernel.slice, kernel_l, &device)?;
            return Ok(Self { slice, device });
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
        Ok(Self { slice, device })
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice =
            ConvTranspose2D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
        Ok(Self { slice, device })
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Avg,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Max,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn upsample_nearest1d(&self, _: &Layout, _out_sz: usize) -> Result<Self> {
        crate::bail!("upsample-nearest1d is not supported on cuda")
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = UpsampleNearest2D(out_w, out_h).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = IndexSelect(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = Gather(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
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
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::BF16(lhs), CudaStorageSlice::BF16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(bf16::ONE, bf16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<bf16>(elem_count)? };
                // Check 16-byte alignment: bf16 is 2 bytes, so offset must be multiple of 8 elements
                // CUDA malloc guarantees 256-byte aligned base, output is fresh allocation
                let known_aligned =
                    (lhs_l.start_offset() * 2) % 16 == 0 && (rhs_l.start_offset() * 2) % 16 == 0;
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
                let mut out = unsafe { dev.alloc::<f16>(elem_count)? };
                // Check 16-byte alignment: f16 is 2 bytes, so offset must be multiple of 8 elements
                let known_aligned =
                    (lhs_l.start_offset() * 2) % 16 == 0 && (rhs_l.start_offset() * 2) % 16 == 0;
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
                let mut out = unsafe { dev.alloc::<f32>(elem_count)? };
                // Check 16-byte alignment: f32 is 4 bytes, so offset must be multiple of 4 elements
                let known_aligned =
                    (lhs_l.start_offset() * 4) % 16 == 0 && (rhs_l.start_offset() * 4) % 16 == 0;
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
                let mut out = unsafe { dev.alloc::<f64>(elem_count)? };
                unsafe {
                    self.device
                        .blas
                        .gemm_strided_batched(cfg, rhs, lhs, &mut out)
                }
                .w()?;
                CudaStorageSlice::F64(out)
            }
            _ => Err(CudaError::InternalError(
                "dtype mismatch in matmul op".to_string(),
            ))?,
        };
        let device = dev.clone();
        Ok(Self { slice, device })
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
                    let info = dev.memcpy_stod(&[dims, src_l.stride()].concat())?;
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
        let a_aligned = (a as usize) % 16 == 0;
        let b_aligned = (b as usize) % 16 == 0;
        let c_aligned = (c as usize) % 16 == 0;
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
        let a_aligned = (a as usize) % 16 == 0;
        let b_aligned = (b as usize) % 16 == 0;
        let c_aligned = (c as usize) % 16 == 0;
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
        let a_aligned = (a as usize) % 16 == 0;
        let b_aligned = (b as usize) % 16 == 0;
        let c_aligned = (c as usize) % 16 == 0;
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
