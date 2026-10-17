//! The operations the driver refuses while any stream of the context captures,
//! and the one place they may be issued.
//!
//! Each is a method on [`Quiet`], which only the record gate
//! ([`super::try_without_recording`], [`super::without_recording`]) can make, so
//! none of them can run while a wave records — whichever thread asks. The raw
//! driver calls they wrap are listed in `clippy.toml`'s `disallowed-methods`,
//! and this module is the only code allowed to make them.
//!
//! Three kinds, one hazard:
//!
//! * **A context-wide synchronise** waits for every stream of the context, a
//!   capturing one included; the driver refuses it and invalidates the capture.
//! * **A module load** may synchronise the context the same way. The engine's
//!   own kernels are linked ahead of time and loaded at context creation (the
//!   backend's `module_loading`); a module loaded at run time — a kernel
//!   compiled for one shape, a custom one, a test's fixture — is loaded here.
//! * **A cuBLAS handle's creation and destruction.** The library synchronises
//!   inside `cublasCreate` where no lint can see it. Measured
//!   (`graph::tests::other_threads_calls_during_capture`): a handle made on
//!   another thread invalidated the capture, while every allocation, free,
//!   pinning, stream and module call probed beside it did not. Each
//!   `CudaDevice::new` used to make one, so any thread that built a device
//!   handle mid-wave broke the wave.

#![allow(clippy::disallowed_methods)]

use std::sync::Arc;

use cudarc::cublas::result::CublasError;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;

use super::record_gate::Quiet;

impl Quiet {
    /// Wait for every stream of `ctx` to drain — every kernel and copy queued on
    /// any of them, on any thread.
    pub fn synchronize_context(&self, ctx: &Arc<CudaContext>) -> Result<(), DriverError> {
        ctx.bind_to_thread()?;
        ctx.synchronize()
    }

    /// Load `ptx` (or a cubin it names) as a module of `ctx`.
    pub fn load_module(
        &self,
        ctx: &Arc<CudaContext>,
        ptx: Ptx,
    ) -> Result<Arc<CudaModule>, DriverError> {
        ctx.load_module(ptx)
    }

    /// A cuBLAS handle on `stream`.
    pub fn create_blas(&self, stream: &Arc<CudaStream>) -> Result<CudaBlas, CublasError> {
        CudaBlas::new(stream.clone())
    }

    /// Destroy a cuBLAS handle.
    pub fn destroy_blas(&self, blas: CudaBlas) {
        drop(blas);
    }
}
