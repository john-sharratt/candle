//! Serialises the GPU tests so that concurrent kernels do not contend, and
//! aborts a test that hangs.
//!
//! A process-wide mutex lets only one heavy CUDA test run at a time, because
//! parallel kernels exhaust the device and produce false timeouts. The guard is
//! reentrant per thread, so one test can call another that takes the lock too.
//! The outermost acquisition starts a 30-second watchdog that aborts the process
//! if the guard is not released in time, and nested acquisitions reuse it. The
//! lock is only taken when the `cuda` feature compiles these tests.

use super::*;

// CUDA TEST SERIALISER
//
// CUDA kernels are GPU-bound and contend for the same device.  Running many
// of them in parallel causes resource exhaustion, false timeouts, and
// flapping results.  We serialise every significant test entry-point through
// a single global Mutex so only one heavy test runs at a time.
//
// The guard is *reentrant per thread*: a function that already holds the lock
// can call another locking function without deadlocking.  This is necessary
// because `run_all_dtype_tests` internally calls `test_repacking`, both of
// which acquire the guard.
//
// The outermost acquisition also starts a 30-second watchdog thread.  If the
// guard is dropped (test completes) before the timer fires, the watchdog exits
// quietly.  If the test hangs, it prints a message and calls process::exit.
// Nested acquisitions reuse the outer watchdog.

// Serialises the GPU tests in this directory; without `cuda` none of them are
// compiled, so nothing acquires it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) static CUDA_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
    std::sync::OnceLock::new();

std::thread_local! {
    static LOCK_DEPTH: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

/// RAII guard returned by [`acquire_cuda_test_lock`].
///
/// On drop it cancels the watchdog (if this is the outermost guard), decrements
/// the per-thread depth counter, and releases the global Mutex.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct CudaTestGuard {
    #[allow(dead_code)]
    pub(super) mutex_guard: Option<std::sync::MutexGuard<'static, ()>>,
    /// Shared flag: setting this to `true` tells the watchdog thread to exit
    /// without calling `process::exit`.  `None` for nested (reentrant) guards.
    pub(super) watchdog_done: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Drop for CudaTestGuard {
    fn drop(&mut self) {
        // Cancel the watchdog before releasing the lock so it can never fire
        // against the next test.
        if let Some(flag) = &self.watchdog_done {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        LOCK_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        // mutex_guard (if Some) is dropped here, unblocking the next waiter.
    }
}

/// Acquire the global CUDA test serialiser.
///
/// Blocks until no other test holds the lock, then returns a guard that:
/// - serialises GPU access for the duration of the test
/// - starts a 30-second watchdog that aborts the process if the guard is not
///   dropped in time (guards against silent hangs)
///
/// Nested calls on the *same* thread return immediately (reentrant); they share
/// the outer watchdog and do not reset its timer.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn acquire_cuda_test_lock() -> CudaTestGuard {
    LOCK_DEPTH.with(|d| {
        let depth = d.get();
        if depth == 0 {
            d.set(1);
            let mutex_guard = CUDA_TEST_LOCK
                .get_or_init(|| std::sync::Mutex::new(()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            // Watchdog: start only once the lock is held (timer reflects real
            // work time, not queue wait time).
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let done_clone = done.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(30));
                if !done_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("❌ TIMEOUT: CUDA test exceeded 30 s — aborting process");
                    std::process::exit(1);
                }
            });

            CudaTestGuard {
                mutex_guard: Some(mutex_guard),
                watchdog_done: Some(done),
            }
        } else {
            d.set(depth + 1);
            // Already locked on this thread; reuse outer watchdog.
            CudaTestGuard {
                mutex_guard: None,
                watchdog_done: None,
            }
        }
    })
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;

    /// Get CUDA device, returns None if not available
    pub fn get_cuda_device() -> Result<Option<Device>> {
        let device = Device::cuda_if_available(0)?;
        if device.is_cuda() {
            Ok(Some(device))
        } else {
            Ok(None)
        }
    }

    /// Skip helper - returns Ok(()) if CUDA not available
    pub fn require_cuda() -> Result<Device> {
        match get_cuda_device()? {
            Some(d) => Ok(d),
            None => {
                println!("Skipping test - CUDA not available");
                candle::bail!("CUDA not available")
            }
        }
    }
}
