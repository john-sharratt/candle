//! Device→host readback accounting for the DeepSeek wave path.
//!
//! The decode hot path allows exactly one readback per token — the sampler.
//! Every other D2H transfer in the wave forward is counted here so tests can
//! assert the budget, which is zero: the MoE routing reaches the streaming
//! `ExpertCache` as an async per-expert summary copy, not a blocking read.

use std::sync::atomic::{AtomicUsize, Ordering};

static WAVE_READBACKS: AtomicUsize = AtomicUsize::new(0);

/// Record one device→host readback on the wave path.
pub fn note_readback() {
    WAVE_READBACKS.fetch_add(1, Ordering::Relaxed);
}

/// Total readbacks recorded since the last reset.
pub fn readback_count() -> usize {
    WAVE_READBACKS.load(Ordering::Relaxed)
}

/// Reset the counter (test setup).
pub fn reset_readbacks() {
    WAVE_READBACKS.store(0, Ordering::Relaxed);
}
