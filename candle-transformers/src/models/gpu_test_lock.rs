//! One process-wide lock for GPU tests.
//!
//! The KV cache's device-side state is **process-global**: one region pool per
//! device carved out of a single reservation, the transient bump domains keyed
//! by stream ordinal, and the split-KV launcher's grow-on-demand partial pool.
//! Two tests that each build a `KvCache` are therefore not independent, however
//! separate their `KvCache` values look — they claim regions from the same free
//! list and launch onto the same streams.
//!
//! `cargo test` runs test functions on a thread pool, so anything not holding
//! this lock can interleave with anything else that is not holding it.
//!
//! # Why one lock and not one per module
//!
//! There used to be two: `decode_utils`' `GPU_LOCK` and `prefill_utils`'
//! `gpu_serial`. Each serialised its own module and neither excluded the other,
//! so a prefill test and a decode test could still run at once — and did.
//! `correctness_decode_seal_gap` failed roughly one run in four with a
//! `mae` of 0.2–0.4 against its 0.15 threshold, the signature its own comment
//! gives for "the writer's tokens were dropped". It passed every time in
//! isolation, every time within its own module, and every time under
//! `--test-threads=1`; only the full suite in parallel reproduced it.
//!
//! A lock scoped to a module is the kind of fix that looks right and leaves the
//! bug in place. The state is shared crate-wide, so the lock has to be.

#[cfg(feature = "cuda")]
use std::sync::{Mutex, MutexGuard};

/// Serialise a GPU test against every other GPU test in this crate.
///
/// Hold the returned guard for the whole test — bind it to a named local
/// (`let _gpu = gpu_serial();`), never to `_`, which drops it immediately and
/// silently reintroduces the race.
///
/// Poisoning is ignored: a panicking test leaves the *device* usable, and
/// turning one failure into a cascade of `PoisonError`s hides the original.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_serial() -> MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}
