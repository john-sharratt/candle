//! One cuBLAS handle per thread per context, made the first time the thread
//! asks for one and kept until the thread ends.
//!
//! Two rules meet here. NVIDIA's: a handle carries workspace and stream state,
//! so one handle used by two threads at once is a data race — it cost
//! `conv1d_gpu` a wrong result and two `conv2d_*_gpu` an illegal access under a
//! parallel suite. And the driver's: creating a handle synchronises the
//! context, which a graph capture on any thread refuses and is invalidated by
//! (`graph::device_wide`). A handle per `CudaDevice::new`, the earlier rule,
//! honoured the first by breaking the second every time any thread built a
//! device handle while a wave recorded. A handle per thread honours both: it is
//! never shared, and it is made once per thread, through the record gate.
//!
//! The handle's fixed workspace is made with it: without one, cuBLAS allocates
//! its workspace on the stream per call, which a wave capture records as a
//! graph allocation and refuses; with one, a GEMM is kernels only.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaStream, DevicePtrMut};

use super::graph::without_recording;
use super::held::Held;
use super::WrapErr;
use crate::Result;

/// Bytes of the workspace every cuBLAS handle is given — NVIDIA's
/// recommendation for this generation of card, and ample for the GEMMs and
/// batched solves this backend issues.
pub(crate) const BLAS_WORKSPACE_BYTES: usize = 4 << 20;

/// A thread's handle on one context, and the workspace it runs in.
pub(crate) struct ThreadBlas {
    blas: Option<CudaBlas>,
    workspace: Held<u8>,
}

impl ThreadBlas {
    /// The workspace's device address.
    pub(crate) fn workspace_ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        use cudarc::driver::DevicePtr;
        self.workspace.device_ptr(stream).0
    }
}

impl Drop for ThreadBlas {
    fn drop(&mut self) {
        // Destroying synchronises like creating does; the thread is ending, so
        // waiting for a recording elsewhere to finish costs nothing it needs.
        if let Some(blas) = self.blas.take() {
            let _ = without_recording(|quiet| quiet.destroy_blas(blas));
        }
    }
}

/// This thread's handle, borrowed. Not `Send`: it is this thread's alone.
pub struct BlasHandle(Rc<ThreadBlas>);

impl BlasHandle {
    pub(crate) fn thread_blas(&self) -> &ThreadBlas {
        &self.0
    }
}

impl Deref for BlasHandle {
    type Target = CudaBlas;
    fn deref(&self) -> &CudaBlas {
        self.0
            .blas
            .as_ref()
            .expect("a handle is destroyed only when its thread ends")
    }
}

thread_local! {
    /// This thread's handles, by context.
    static HANDLES: RefCell<Vec<(usize, Rc<ThreadBlas>)>> = const { RefCell::new(Vec::new()) };
}

/// Whether this thread already holds a handle on `stream`'s context.
pub(crate) fn has_handle(stream: &Arc<CudaStream>) -> bool {
    let key = Arc::as_ptr(stream.context()) as usize;
    HANDLES.with(|h| h.borrow().iter().any(|(k, _)| *k == key))
}

/// This thread's handle on `stream`'s context, made on first use.
///
/// Made through the record gate, waiting for any recording to end: refused, with
/// an error, on a thread that is itself recording — `CudaDevice::cublas` pauses
/// that thread's recording around a first creation.
pub(crate) fn handle_for(stream: &Arc<CudaStream>) -> Result<BlasHandle> {
    let key = Arc::as_ptr(stream.context()) as usize;
    if let Some(found) = HANDLES.with(|h| {
        h.borrow()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, t)| t.clone())
    }) {
        return Ok(BlasHandle(found));
    }
    let blas = without_recording(|quiet| quiet.create_blas(stream))?.w()?;
    // SAFETY: cuBLAS writes its workspace before reading it.
    let mut workspace = unsafe { Held::<u8>::alloc_on(stream, BLAS_WORKSPACE_BYTES) }?;
    {
        let (ptr, _g) = workspace.device_ptr_mut(stream);
        // SAFETY: a live handle and a device buffer of the stated size, kept
        // beside it for as long as the handle exists.
        let status = unsafe {
            cudarc::cublas::sys::cublasSetWorkspace_v2(
                *blas.handle(),
                ptr as *mut std::ffi::c_void,
                BLAS_WORKSPACE_BYTES,
            )
        };
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            crate::bail!("cublasSetWorkspace failed: {status:?}");
        }
    }
    let made = Rc::new(ThreadBlas {
        blas: Some(blas),
        workspace,
    });
    HANDLES.with(|h| h.borrow_mut().push((key, made.clone())));
    Ok(BlasHandle(made))
}
