//! A gate across a CUDA stream, for tests that prove a host call does not wait
//! on the stream: close it, make the call, and the call must return while the
//! gate is still shut.
#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use candle::cuda_backend::cudarc::driver::result::stream::launch_host_function;
use candle::cuda_backend::cudarc::driver::CudaStream;

/// A gate across a stream: a host function the stream runs when it reaches it,
/// which blocks until the test opens the gate — so everything enqueued after it
/// waits, deterministically and with no GPU work. It opens by itself after
/// [`StreamGate::TIMEOUT`], so a host that waits on work queued behind a closed
/// gate is held that long and then fails the test, rather than hanging it.
pub(crate) struct StreamGate {
    open: Arc<(Mutex<bool>, Condvar)>,
}

impl StreamGate {
    pub(crate) const TIMEOUT: Duration = Duration::from_secs(2);

    pub(crate) fn close_on(stream: &CudaStream) -> Self {
        let open = Arc::new((Mutex::new(false), Condvar::new()));
        let arg = Arc::into_raw(Arc::clone(&open)) as *mut c_void;
        // SAFETY: `wait_at_gate` takes back, exactly once, the reference leaked
        // here, when the stream runs it.
        unsafe { launch_host_function(stream.cu_stream(), wait_at_gate, arg) }.unwrap();
        Self { open }
    }

    pub(crate) fn open(&self) {
        let (opened, wake) = &*self.open;
        *opened.lock().unwrap() = true;
        wake.notify_all();
    }

    /// Open the gate from another thread after `delay` — for a test whose own
    /// thread is about to wait on work behind it.
    pub(crate) fn open_after(&self, delay: Duration) -> JoinHandle<()> {
        let gate = Self {
            open: Arc::clone(&self.open),
        };
        thread::spawn(move || {
            thread::sleep(delay);
            gate.open();
        })
    }
}

impl Drop for StreamGate {
    fn drop(&mut self) {
        self.open();
    }
}

/// The host function behind a [`StreamGate`]. It must not call CUDA, and does
/// not: it only waits.
unsafe extern "C" fn wait_at_gate(arg: *mut c_void) {
    // SAFETY: `arg` is the `Arc` leaked by `StreamGate::close_on`, taken back
    // once.
    let open = unsafe { Arc::from_raw(arg as *const (Mutex<bool>, Condvar)) };
    let (opened, wake) = &*open;
    let guard = opened
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _ = wake.wait_timeout_while(guard, StreamGate::TIMEOUT, |opened| !*opened);
}
