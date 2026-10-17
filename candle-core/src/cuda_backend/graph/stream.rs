//! The streams a graph touches: the device's stream it runs on, and, for the
//! driver-level tests, a stream of their own to record single captures on.

#[cfg(test)]
use crate::cuda_backend::CudaDevice;
use crate::cuda_backend::WrapErr;
use crate::Result;
use cudarc::driver::{sys, CudaContext, CudaStream};
use std::sync::Arc;

/// The stream a replayed graph is launched into: the device's one stream, the
/// one every eager launch of the same work uses and the one a wave records on.
///
/// It has no capture method: recording on it is the capture hub's alone.
pub struct ComputeStream {
    stream: Arc<CudaStream>,
}

impl ComputeStream {
    /// `dev`'s stream.
    #[cfg(test)]
    pub fn of(dev: &CudaDevice) -> Self {
        Self {
            stream: dev.stream_unleased().clone(),
        }
    }

    pub(crate) fn from_stream(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }

    pub(crate) fn cu(&self) -> sys::CUstream {
        self.stream.cu_stream()
    }

    pub(crate) fn bind(&self) -> Result<()> {
        self.stream.context().bind_to_thread().w()
    }

    pub(crate) fn context(&self) -> Arc<CudaContext> {
        self.stream.context().clone()
    }
}

/// A created, non-blocking stream for the driver-level tests, which record
/// single captures on it to prove the facts the wave capture rests on.
///
/// Only constructible as a stream of its own, never as the null stream, and
/// reachable for launching only through the [`super::CaptureSession`] that
/// holds it by `&mut` — so while a capture is open nothing else can submit to
/// it, from any thread.
#[cfg(test)]
pub struct CaptureStream {
    pub(super) stream: Arc<CudaStream>,
}

#[cfg(test)]
impl CaptureStream {
    /// A new non-blocking stream on `dev`'s context.
    pub fn new(dev: &CudaDevice) -> Result<Self> {
        // `CudaContext::new_stream` creates with `CU_STREAM_NON_BLOCKING`.
        Ok(Self {
            stream: dev.cuda_context().new_stream().w()?,
        })
    }
}
