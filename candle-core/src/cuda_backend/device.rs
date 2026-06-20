use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use cudarc;
use cudarc::driver::CudaFunction;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

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
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    curand: Arc<Mutex<CudaRng>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn memcpy_dtov<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.stream.memcpy_dtov(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_stod<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.memcpy_stod(src).w()
    }
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
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
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
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
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
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
    /// Validates that the device meets the minimum compute capability (SM 8.0 / Ampere).
    /// All precompiled CUDA kernels target SM80; older GPUs will produce incorrect results.
    fn validate_compute_capability(context: &Arc<cudarc::driver::CudaContext>) -> Result<()> {
        use cudarc::driver::sys::CUdevice_attribute;
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .w()?;
        let minor = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .w()?;
        if major < 8 {
            crate::bail!(
                "CUDA device (SM {major}.{minor}) does not meet the minimum requirement of \
                 SM 8.0 (Ampere). Precompiled kernels target SM80 and will not run correctly \
                 on older hardware."
            );
        }
        Ok(())
    }

    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        Self::validate_compute_capability(&context)?;
        let stream = context.new_stream().w()?;
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let dev = Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
        };
        // Record free VRAM now, before any model weights load, so the KV budget
        // gate can estimate our resident footprint and credit pageable memory
        // the OS evicts for us (see `gpu_memory::device_init_free`).
        if let Ok((free, _total)) = dev.mem_get_info() {
            crate::gpu_memory::note_device_init_free(ordinal, free);
        }
        Ok(dev)
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
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        Self::validate_compute_capability(&context)?;
        let stream = context.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let dev = Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
        };
        // Record free VRAM now, before any model weights load, so the KV budget
        // gate can estimate our resident footprint and credit pageable memory
        // the OS evicts for us (see `gpu_memory::device_init_free`).
        if let Ok((free, _total)) = dev.mem_get_info() {
            crate::gpu_memory::note_device_init_free(ordinal, free);
        }
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
            super::run_affine_ffi(&slice, self, &layout, up - lo, lo)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
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
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}
