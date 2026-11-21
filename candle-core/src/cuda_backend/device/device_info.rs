//! Device properties read once, and the checks that refuse a card this build
//! cannot run.
//!
//! Kernels are built as native code for a fixed set of architectures with no
//! PTX to fall back on, so a card outside that set is a hard stop: launches
//! would fail without anyone observing it and the KV path would silently produce
//! zeros. Per-argument cross-stream event tracking is switched off before the
//! first allocation, because the engine orders its cross-stream work explicitly
//! at each producer and consumer, and the per-argument events cost host time on
//! every launch. Memory-pool queries report only this process's own allocations,
//! which is the budget denominator on a display driver where free memory is
//! shared with other processes.

use super::*;

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
    pub(super) fn validate_compute_capability(
        context: &Arc<cudarc::driver::CudaContext>,
    ) -> Result<()> {
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
    pub(super) fn disable_per_arg_event_tracking(
        context: &std::sync::Arc<cudarc::driver::CudaContext>,
    ) {
        // SAFETY: called before any allocation on this context, so no slice
        // predates the setting (the documented hazard is mixing tracked and
        // untracked slices).
        unsafe { context.disable_event_tracking() };
    }

    /// Give this handle its own curand generator.
    ///
    /// **Everything stateful is rebuilt; everything memoised is shared.** The
    /// caches on a `CudaDevice` are content- and shape-keyed tables of immutable
    /// bytes, so two handles reading one entry is the point of keeping them. A
    /// generator is neither: a device is built expecting to draw from
    /// [`DEFAULT_SEED`], and one advancing generator makes what a caller draws a
    /// function of who drew before it, so a `Tensor::randn` fixture stops being
    /// a fixture. It cost two `candle-transformers` tests, each passing alone and
    /// failing in the suite.
    ///
    /// The cuBLAS handle, stateful too, is not a handle's but a thread's
    /// ([`Self::cublas`], `thread_blas`).
    ///
    /// Called on **both** paths out of the cache: the hit, and the loser of a
    /// first-touch race, which is also handed a clone of a shared device.
    pub(super) fn give_own_stateful(&mut self) -> Result<()> {
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

    /// The most the pool has had in use at once since the last
    /// [`Self::reset_pool_used_peak`] — the activation peak of whatever ran in
    /// between, which `pool_used_bytes` sampled afterwards cannot see.
    pub fn pool_used_peak_bytes(&self) -> Result<usize> {
        self.pool_attr(cudarc::driver::sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_HIGH)
    }

    /// Start [`Self::pool_used_peak_bytes`] again from what is in use now.
    pub fn reset_pool_used_peak(&self) -> Result<()> {
        use cudarc::driver::sys;
        self.context.bind_to_thread().w()?;
        let dev = self.context.cu_device();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        // The driver resets a high-water mark by being handed zero.
        let mut zero: u64 = 0;
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, dev)
                .result()
                .w()?;
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                &mut zero as *mut u64 as *mut std::ffi::c_void,
            )
            .result()
            .w()?;
        }
        Ok(())
    }

    pub(super) fn pool_attr(
        &self,
        attr: cudarc::driver::sys::CUmemPool_attribute_enum,
    ) -> Result<usize> {
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
