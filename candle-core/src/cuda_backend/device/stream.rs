//! The device's one stream, and the wave capture that records a thread's
//! launches as graphs.
//!
//! Every launch, copy and graph on a device goes on the one stream, so the order
//! work is issued in is the order it runs in on every thread. A thread holds a
//! lease on that stream while it issues work. A wave capture records launches
//! into a chain of graphs, and a segment ends at each call that must meet the
//! device eagerly, such as an allocation, an upload or a readback, and at each
//! flush. Memory is returned only after every recorded launch that may still
//! read it has been issued, because a free placed inside a capture would be
//! captured with it.

use super::*;

impl CudaDevice {
    /// The device's one stream, leased to this thread for as long as the lease
    /// lives (`stream_lease`).
    ///
    /// Everything on this device goes on it: launches, copies, allocations and
    /// the graphs a wave records, so issue order is execution order on every
    /// thread. While this thread records a wave its launches are what the
    /// segment captures; on any other thread the lease waits while a segment
    /// records. Take it where the work is issued and let it go with the work.
    #[track_caller]
    pub fn cuda_stream(&self) -> StreamLease {
        StreamLease::take(&self.stream)
    }

    /// The stream itself, for the wave's own machinery on the thread that owns
    /// the wave — which records on it and needs no lease to do so.
    pub(crate) fn stream_unleased(&self) -> &Arc<CudaStream> {
        &self.stream
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
    ///
    /// The caller's location names the kind of wave this is: each call site
    /// (a draft walk, a verify, a forward) keeps its own predictions of what
    /// its segments hold (`graph::ordinals`).
    #[track_caller]
    pub fn begin_wave_capture(&self) -> Result<WaveCapture> {
        self.capture.begin_wave(&self.stream, Location::caller())?;
        enter_wave(self.clone());
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
    /// output waits on. Inside a wave capture this ends the recording segment,
    /// and the caller's location is what the cut is attributed to.
    #[track_caller]
    pub fn flush_launches(&self) -> Result<()> {
        self.capture.flush(&self.stream)
    }

    /// The wave segment this thread's launches are being recorded into right
    /// now, as `(wave serial, segment ordinal within the wave)` — `None` when
    /// they reach the device as they are issued.
    pub fn recording_segment(&self) -> Option<(u64, usize)> {
        self.capture.recording_segment()
    }

    /// The serial of the wave capture open on this thread — `None` with no
    /// wave open — whatever it is doing: recording, running a segment eagerly,
    /// held or paused.
    pub fn wave_in_progress(&self) -> Option<u64> {
        self.capture.wave_here()
    }

    /// Whether this thread's wave is in a segment that runs eagerly and is not
    /// paused: an upload queued on the compute stream now lands behind the
    /// launches already issued, with no segment to end ahead of it.
    pub fn eager_segment_here(&self) -> bool {
        self.capture.eager_segment_here()
    }

    /// Hand what this thread has issued to the driver, ending no segment —
    /// for a host protocol that waits on eagerly issued launches between the
    /// points where it ends segments.
    pub fn submit_launches(&self) -> Result<()> {
        self.capture.submit(&self.stream)
    }

    /// Suspend this thread's wave capture, if it has one, until the guard
    /// drops: the recorded segment is launched first, and the caller's eager
    /// device calls run in order behind it. The caller's location is what the
    /// cut is attributed to (`graph::cuts`).
    #[track_caller]
    pub fn pause_capture(&self) -> Result<Option<Paused<'_>>> {
        self.capture.pause(&self.stream)
    }

    /// Drop `value` — something owning device memory — once every launch
    /// recorded so far has been issued; at once, under a lease, when no segment
    /// records. A free issued on the stream while it captures would be taken
    /// into the capture, and a recorded launch may still read the memory, so it
    /// waits for its segment.
    #[track_caller]
    pub fn retire<T: Send + 'static>(&self, value: T) {
        if let Some(now) = self.capture.retire(Box::new(value)) {
            let _stream = self.cuda_stream();
            drop(now);
        }
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
    /// The handle comes back holding a lease on the stream (`stream_lease`), so
    /// every call made through it is issued under that lease: drop it once the
    /// calls are made.
    ///
    /// The handle is this **thread's** ([`thread_blas`]): one per thread, made
    /// the first time the thread asks. A thread asking for its first one while
    /// it records a wave has the recording paused around the creation, which
    /// would otherwise end its own capture.
    #[track_caller]
    pub fn cublas(&self) -> Result<BlasHandle> {
        let handle = if thread_blas::has_handle(&self.stream) {
            thread_blas::handle_for(&self.stream)?
        } else {
            let _eager = self.pause_capture()?;
            thread_blas::handle_for(&self.cuda_stream())?
        };
        let stream = self.cuda_stream();
        // SAFETY: a live handle of this thread and a stream of its context.
        unsafe { cudarc::cublas::result::set_stream(*handle.handle(), stream.cu_stream() as _) }
            .w()?;
        // Setting the stream hands the handle back to cuBLAS's own workspace
        // pool, which allocates on the stream; give it its fixed one again.
        let ptr = handle.thread_blas().workspace_ptr(&self.stream);
        // SAFETY: the handle's own workspace, alive as long as the handle.
        let status = unsafe {
            cudarc::cublas::sys::cublasSetWorkspace_v2(
                *handle.handle(),
                ptr as *mut std::ffi::c_void,
                BLAS_WORKSPACE_BYTES,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            crate::bail!("cublasSetWorkspace failed: {status:?}");
        }
        Ok(handle.leased(stream))
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
        let module = self.load_ptx(func_name, &ptx)?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.cuda_stream(),
        })
    }

    /// Load run-time PTX as a module of this device's context, through the
    /// record gate: a module load can synchronise the context, which would
    /// invalidate a capture in flight on any thread.
    ///
    /// A driver older than the installed toolkit refuses the toolkit's PTX ISA
    /// at JIT time. The AOT kernels never hit this — they ship SASS only — so
    /// run-time PTX is the only kind the driver is ever asked to JIT. The
    /// toolkit's own `ptxas` can still assemble it to native SASS for this
    /// device, and a cubin loads without any PTX version check, so that refusal
    /// is answered by assembling toolkit-side and loading native code.
    pub(crate) fn load_ptx(
        &self,
        func_name: &str,
        ptx: &cudarc::nvrtc::Ptx,
    ) -> Result<Arc<CudaModule>> {
        let loaded = without_recording(|quiet| quiet.load_module(&self.context, ptx.clone()))?;
        match loaded {
            Ok(module) => Ok(module),
            Err(e) if e.0 == CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION => {
                self.load_module_via_ptxas(func_name, ptx)
            }
            Err(e) => Err(e).w(),
        }
    }

    /// Assemble `ptx` to this device's native SASS with the toolkit's `ptxas`
    /// and load the cubin — for a driver that refuses to JIT the toolkit's PTX
    /// ISA version (see [`Self::load_ptx`]).
    pub(super) fn load_module_via_ptxas(
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
            without_recording(|quiet| {
                quiet.load_module(&self.context, cudarc::nvrtc::Ptx::from_file(&cubin_path))
            })?
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
                stream: self.cuda_stream(),
            });
        }
        drop(ms);
        // Loaded with no lock held: the gate waits for every recording to end,
        // and a recording thread may be asking for a module this map holds.
        let loaded =
            without_recording(|quiet| quiet.load_module(&self.context, ptx.into()))?.w()?;
        let cuda_module = self
            .custom_modules
            .write()
            .unwrap()
            .entry(module_name.to_string())
            .or_insert(loaded)
            .clone();
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.cuda_stream(),
        })
    }
}
