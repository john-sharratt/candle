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
//!
//! **A thread's handles are destroyed by a live thread, not by their own as it
//! ends.** A thread's locals are dropped as it exits, which on Windows runs
//! beside the CUDA driver's own teardown of that thread: a driver call made
//! there fails with `CUDA_ERROR_NOT_INITIALIZED`, and cudarc records a failure
//! in a `Drop` on the context, where the next `bind_to_thread` on any thread
//! returns it. A test whose thread had run a GEMM left exactly that behind, and
//! the next test failed to make its device handle. So an ending thread moves
//! its handles to [`ORPHANS`], and the next thread to make a handle destroys
//! them.

use std::cell::RefCell;
use std::mem::take;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaStream, DevicePtrMut};

use super::graph::without_recording;
use super::held::Held;
use super::{StreamLease, WrapErr};
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
        // Destroying synchronises like creating does, so it waits for any
        // recording to end — on a thread making its own first handle, which
        // waits for the same thing anyway.
        if let Some(blas) = self.blas.take() {
            let _ = without_recording(|quiet| quiet.destroy_blas(blas));
        }
    }
}

/// Handles whose threads have ended, for the next thread that makes a handle
/// to destroy — see the module documentation.
static ORPHANS: Mutex<Vec<ThreadBlas>> = Mutex::new(Vec::new());

/// This thread's handles, by context. Dropped as the thread ends, it hands each
/// to [`ORPHANS`] instead of destroying it.
struct Handles(Vec<(usize, Rc<ThreadBlas>)>);

impl Drop for Handles {
    fn drop(&mut self) {
        let mut orphans = ORPHANS.lock().unwrap_or_else(|p| p.into_inner());
        for (_, held) in take(&mut self.0) {
            // A `BlasHandle` is not `Send` and lives on this thread's stack, so
            // by the time its locals go, the list holds the last reference.
            if let Ok(blas) = Rc::try_unwrap(held) {
                orphans.push(blas);
            }
        }
    }
}

/// Destroy every handle an ended thread left, from this live one.
fn destroy_orphans() {
    let orphans = take(&mut *ORPHANS.lock().unwrap_or_else(|p| p.into_inner()));
    drop(orphans);
}

/// This thread's handle, borrowed. Not `Send`: it is this thread's alone.
///
/// Handed out bound to the device's stream, it holds the lease its calls go out
/// on (`stream_lease`) until it drops.
pub struct BlasHandle {
    blas: Rc<ThreadBlas>,
    _lease: Option<StreamLease>,
}

impl BlasHandle {
    fn unleased(blas: Rc<ThreadBlas>) -> Self {
        Self { blas, _lease: None }
    }

    pub(crate) fn thread_blas(&self) -> &ThreadBlas {
        &self.blas
    }

    /// This handle, its calls issued under `lease`.
    pub(crate) fn leased(self, lease: StreamLease) -> Self {
        Self {
            blas: self.blas,
            _lease: Some(lease),
        }
    }
}

impl Deref for BlasHandle {
    type Target = CudaBlas;
    fn deref(&self) -> &CudaBlas {
        self.blas
            .blas
            .as_ref()
            .expect("a handle is destroyed only when its thread ends")
    }
}

thread_local! {
    /// This thread's handles, by context.
    static HANDLES: RefCell<Handles> = const { RefCell::new(Handles(Vec::new())) };
}

/// Whether this thread already holds a handle on `stream`'s context.
pub(crate) fn has_handle(stream: &Arc<CudaStream>) -> bool {
    let key = Arc::as_ptr(stream.context()) as usize;
    HANDLES.with(|h| h.borrow().0.iter().any(|(k, _)| *k == key))
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
            .0
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, t)| t.clone())
    }) {
        return Ok(BlasHandle::unleased(found));
    }
    destroy_orphans();
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
    HANDLES.with(|h| h.borrow_mut().0.push((key, made.clone())));
    Ok(BlasHandle::unleased(made))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;
    use crate::cuda_backend::CudaDevice;
    use std::thread;

    /// **A thread that made a handle and ended leaves no error on the
    /// context**, and the next thread to make one destroys it. Destroyed by the
    /// ending thread itself, its workspace's free failed inside the driver's
    /// thread teardown and was recorded on the context, and the next
    /// `bind_to_thread` anywhere — a device handle made for the next test —
    /// failed with `CUDA_ERROR_NOT_INITIALIZED`.
    #[test]
    fn an_ended_threads_handle_leaves_the_context_clean() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (theirs, again) = (dev.clone(), dev.clone());
        thread::spawn(move || handle_for(&theirs.cuda_stream()).map(|_| ()))
            .join()
            .expect("the handle's thread panicked")?;
        dev.cuda_stream().context().check_err().w()?;
        CudaDevice::new(0)?;
        // A thread making its first handle destroys what the ended one left.
        thread::spawn(move || handle_for(&again.cuda_stream()).map(|_| ()))
            .join()
            .expect("the second thread panicked")?;
        dev.cuda_stream().context().check_err().w()?;
        Ok(())
    }
}
