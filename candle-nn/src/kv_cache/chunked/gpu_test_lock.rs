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
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}
