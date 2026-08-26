//! One process-wide lock for this crate's GPU tests.
//!
//! The KV cache's device-side state is **process-global**: one region pool per
//! device carved from a single reservation, the transient bump domains keyed by
//! stream ordinal, and the gid pool behind them. Two tests that each build a
//! backing are therefore not independent, however separate their values look.
//!
//! Acquiring the CUDA context is shared state too, and it is the noisier of the
//! two: several tests call `Device::cuda_if_available(0)` as their first act,
//! and concurrent first-touch initialisation surfaces as
//! `CublasError(CUBLAS_STATUS_NOT_INITIALIZED)` from whichever one loses. That
//! is the failure this lock exists to remove — it is not a real defect in the
//! test's subject, which is exactly why it wasted time before being named.
//!
//! `cargo test` runs test functions on a thread pool, so anything not holding
//! this lock can interleave with anything else that is not holding it. Per-module
//! locks do not compose: this crate had two (`bump_arena`'s `SERIAL` and
//! `region_pool`'s), each serialising its own module against itself and neither
//! excluding the other or the selection tests. Shared state needs a shared lock.

/// Serialise a GPU test against every other GPU test in this crate.
///
/// Hold the returned guard for the whole test — bind it to a named local
/// (`let _gpu = gpu_serial();`), never to `_`, which drops it immediately and
/// silently reintroduces the race.
///
/// Poisoning is ignored: a panicking test leaves the *device* usable, and
/// turning one failure into a cascade of `PoisonError`s hides the original.
/// Also **attributes an async fault to the test that caused it.**
///
/// A kernel that reads out of bounds does not fail at its launch: the launch is
/// asynchronous, so the error is raised by whatever call next synchronises the
/// context — often several tests later, and after any number of CPU-only tests
/// in between. It is sticky, so every test after that one fails too. The result
/// reads as "these three GPU tests are broken" when the three are witnesses and
/// the culprit is long gone; running under `CUDA_LAUNCH_BLOCKING=1` to find it
/// serialises the launches and makes the hazard disappear.
///
/// Synchronising here, while the guard still names the test that held it, turns
/// that cascade into one failure at the source.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn gpu_serial() -> GpuGuard {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GpuGuard(Some(GPU.lock().unwrap_or_else(|e| e.into_inner())))
}

/// The `gpu_serial` guard: releases the lock, and drains the device first.
#[cfg(all(test, feature = "cuda"))]
pub(crate) struct GpuGuard(Option<std::sync::MutexGuard<'static, ()>>);

#[cfg(all(test, feature = "cuda"))]
impl Drop for GpuGuard {
    fn drop(&mut self) {
        // Only when the test is otherwise passing. A test already unwinding has
        // its own failure to report, and panicking again in a drop aborts the
        // process, which would lose it.
        if !std::thread::panicking() {
            // **Acquiring the device is part of the check, not a precondition
            // for it.** `CudaContext::bind_to_thread` opens with `check_err`,
            // which *swaps out* the recorded error state — so on a poisoned
            // context the acquisition is what fails, and it is also what
            // consumes the fault. An `if let Ok(dev)` here would therefore skip
            // silently in precisely the case this guard exists to report, and
            // destroy the error on its way past so nothing downstream could
            // report it either. Chained, so either half's failure is the
            // finding.
            //
            // A machine with no CUDA takes `cuda_if_available`'s CPU branch,
            // where `synchronize` is an infallible no-op — an `Err` here always
            // means a real device fault.
            let drained = candle::Device::cuda_if_available(0).and_then(|dev| dev.synchronize());
            if let Err(e) = drained {
                // Release before panicking so the next test can take it: a
                // guard dropped mid-panic would poison the mutex, and
                // `gpu_serial` deliberately ignores poisoning.
                self.0.take();
                panic!(
                    "GPU work from this test faulted: {e}\n\
                     The device is now poisoned, so every later GPU test in this \
                     process will fail too — this is the one that caused it."
                );
            }
        }
        self.0.take();
    }
}
