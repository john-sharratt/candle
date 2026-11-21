//! The `BackendDevice` implementation for CUDA: one shared handle per ordinal,
//! and the allocations placed on the wave's held ground.
//!
//! A device is memoised per ordinal for the life of the process, because every
//! handle keeps its compiled modules and upload caches, and a handle built and
//! dropped per call would hold that memory until allocation fails. Only the
//! memoised parts are shared: the stateful parts, such as the random generator,
//! are rebuilt for each handle, so a draw depends on the seed rather than on who
//! drew before. An owned copy is taken from the span's held ground, so releasing
//! it is a fenced give-back rather than a driver free, and the caller names the
//! wave a host-built table belongs to.

use super::*;

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

        // Before the context: the driver reads its module-loading mode once, at
        // initialisation, and a lazy load breaks another thread's capture.
        module_loading::request_eager();
        // Before the first allocation on any device, so none goes unnamed
        // (`driver_alloc`).
        install_driver_alloc_watch();
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        module_loading::verify_eager()?;
        Self::validate_compute_capability(&context)?;
        Self::disable_per_arg_event_tracking(&context);
        // **The device's one stream**, created rather than the context's legacy
        // null stream, which cannot be captured: every launch, copy and graph
        // launch on this device goes on it, from every thread, and a wave
        // records on it directly (`stream_lease`).
        let stream = context.new_stream().w()?;
        let curand = cudarc::curand::CudaRng::new(DEFAULT_SEED, stream.clone()).w()?;
        let dev = Self {
            id: DeviceId::new(),
            context,
            stream,
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            info_tables: Arc::new(Mutex::new(InfoRing::default())),
            perm_tables: Arc::new(Mutex::new(HashMap::new())),
            staging: Arc::new(Mutex::new(None)),
            foreign_staging: Arc::new(Mutex::new(None)),
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
        // Before the handle is returned, so before anything can launch: the
        // launchers' own grow-only scratch is taken from the held allocator from
        // the first launch on, and a block they replace is retired through this
        // device's capture hub.
        kernel_scratch::install(&dev);
        stream_owner::register(&dev);
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
        // The generator fills on the device's stream: leased while it does, and
        // before its lock, which a recording thread may want while this waits.
        let _stream = self.cuda_stream();
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
            super::super::run_affine_ffi(&slice, self, &layout, up - lo, lo, Backing::Owned)?.0
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
        // The generator fills on the device's stream: leased while it does, and
        // before its lock, which a recording thread may want while this waits.
        let _stream = self.cuda_stream();
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

    #[track_caller]
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
        sync_watch::synchronize(&self.cuda_stream()).w()?;
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

    /// Uninitialised storage of `shape` that outlives every wave: an **owned
    /// copy** — values taken off a span to be kept past it (`to_owned_tensor`,
    /// `cat_owned`).
    ///
    /// Taken from the span's held ground, not from the driver: the copy into it
    /// stays one device-to-device copy, no driver call is made, and its release
    /// is a fenced give-back rather than a free — which, with no memory pools
    /// (TCC), would synchronise the stream. The storage anchors the [`Held`]
    /// buffer, so every view and clone of it keeps the range, and the last one
    /// to drop releases it behind everything queued on the stream. A ground with
    /// no room for it hands the buffer to the driver, reported by the ground.
    ///
    /// # Safety
    ///
    /// Uninitialised, as [`Self::alloc_uninit_from`].
    #[track_caller]
    pub(crate) unsafe fn alloc_owned(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let held = Held::<u8>::alloc(self, elem_count * dtype.size_in_bytes())?;
        let stream = self.cuda_stream();
        let (ptr, guard) = held.device_ptr(&stream);
        drop(guard);
        // SAFETY: `ptr` names `elem_count` elements of `dtype` (256-aligned, as
        // every held range is), kept live by the anchor for as long as any view
        // of the storage exists.
        CudaStorage::from_anchored_device_ptr(
            ptr,
            elem_count,
            dtype,
            self,
            LeaseAnchor::new(Arc::new(held)),
        )
    }

    /// Uninitialised storage for an op's output that follows its operand
    /// `from`: carved from the operand's wave arena while its phase is open,
    /// and otherwise an owned range of the held ground ([`Self::alloc_owned`]).
    /// Never the driver's.
    ///
    /// # Safety
    ///
    /// Uninitialised, as [`Self::alloc_uninit_from`].
    #[track_caller]
    pub(crate) unsafe fn alloc_following(
        &self,
        shape: &Shape,
        dtype: DType,
        from: &Backing,
    ) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let bytes = elem_count * dtype.size_in_bytes();
        if let Some((ptr, ticket)) =
            wave_alloc_attributed(from.inherit_ticket(), bytes, INHERIT_ALIGN)?
        {
            forbid_write("wave arena lease (alloc_following)", ptr, bytes);
            // SAFETY: a range of the open phase's arena, `bytes` long and
            // aligned for every dtype, released with the phase.
            return CudaStorage::from_leased_device_ptr(
                ptr,
                elem_count,
                dtype,
                self,
                LeaseOrigin::Wave(ticket),
            );
        }
        self.alloc_owned(shape, dtype)
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
