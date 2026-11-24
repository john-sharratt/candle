//! The sweep of kept images no conversation uses, run on its own clock and
//! never on a request's path: it reads every image record
//! (`vision::tenancy::sweep_unused`), which no upload should wait on.

use std::sync::Weak;
use std::time::Duration;

use candle_conversation::vision::tenancy::sweep_unused;

use super::*;

/// How often unused images are looked for: their grace is a day, so an hour
/// late is nothing.
pub(super) const IMAGE_SWEEP_EVERY: Duration = Duration::from_secs(60 * 60);

/// Sweep unused images every [`IMAGE_SWEEP_EVERY`] while `state` lives, each
/// sweep on the blocking pool.
pub(super) fn spawn_image_sweep(state: Weak<InferenceState>) {
    tokio::spawn(async move {
        let mut every = tokio::time::interval(IMAGE_SWEEP_EVERY);
        loop {
            every.tick().await;
            let Some(live) = state.upgrade() else {
                return;
            };
            let swept = tokio::task::spawn_blocking(move || {
                sweep_unused(&live.engine.conversation().shared(), now_ms())
            })
            .await;
            match swept {
                Ok(Ok(retired)) if !retired.is_empty() => {
                    tracing::info!(retired = retired.len(), "unused kept images retired");
                }
                Ok(Ok(_)) => {}
                Ok(Err(e)) => tracing::warn!("sweeping unused images failed: {e}"),
                Err(e) => tracing::warn!("sweeping unused images panicked: {e}"),
            }
        }
    });
}
