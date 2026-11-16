//! Where the calibration phase's wall time goes.

use std::time::{Duration, Instant};

/// Wall-time attribution for the calibration phase.
///
/// The phase drives a batched wave loop from a single thread, so every bucket
/// below is time the GPU is *not* being fed: the window cannot refill and no
/// forward is in flight while the loading thread sits in one of these calls.
/// Measured against forward time the phase's prefill runs at the
/// batched-forward gate's rate, so its wall-clock rate is set entirely by how
/// much of the phase is spent here rather than by the model.
///
/// Each bucket names an engine call made once per calibration case. The
/// substrate ones (`tag`, `archive`, `distill`) each take the substrate write
/// lock and append a record under the persistence mutex, which is the same lock
/// the scheduler's wave loop needs to admit the next case.
#[derive(Default)]
pub struct Timing {
    /// `new_conversations_with_projection_batch` — one pipelined round-trip per window.
    pub create: Duration,
    /// `set_conversation_metadata` — resume tag, per case.
    pub tag: Duration,
    /// `submit_prefilled_turn` / `submit_turn_with_options`, per case.
    pub submit: Duration,
    /// `finish_turn` — the seal, per case.
    pub seal: Duration,
    /// `persist_projection_events` — the trajectory's projection sequence, per case.
    pub persist_events: Duration,
    /// `set_conversation_archived` — the atomic done-marker, per case.
    pub archive: Duration,
    /// `distill_timeline` (+ its state probe), per case.
    pub distill: Duration,
    /// `demote_timelines_hot` — incremental and boundary hot→warm sweeps.
    pub demote: Duration,
    /// The wait taken when a pass retired nothing.
    pub idle: Duration,
}

impl Timing {
    /// Run `f`, adding its wall time to the bucket `pick` selects.
    pub fn time<T>(&mut self, pick: fn(&mut Self) -> &mut Duration, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        *pick(self) += t.elapsed();
        out
    }

    pub fn log(&self, wall: Duration) {
        let ms = |d: &Duration| d.as_millis() as u64;
        let tracked = self.create
            + self.tag
            + self.submit
            + self.seal
            + self.persist_events
            + self.archive
            + self.distill
            + self.demote
            + self.idle;
        tracing::info!(
            wall_ms = ms(&wall),
            create_ms = ms(&self.create),
            tag_ms = ms(&self.tag),
            submit_ms = ms(&self.submit),
            seal_ms = ms(&self.seal),
            persist_events_ms = ms(&self.persist_events),
            archive_ms = ms(&self.archive),
            distill_ms = ms(&self.distill),
            demote_ms = ms(&self.demote),
            idle_ms = ms(&self.idle),
            tracked_ms = ms(&tracked),
            untracked_ms = ms(&wall.saturating_sub(tracked)),
            "calibration wall-time attribution"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use std::time::Duration;

    use super::Timing;

    #[test]
    fn time_is_added_to_the_bucket_named_and_no_other() {
        let mut t = Timing::default();
        let out = t.time(
            |t| &mut t.seal,
            || {
                sleep(Duration::from_millis(2));
                7
            },
        );
        assert_eq!(out, 7);
        assert!(t.seal >= Duration::from_millis(2));
        assert_eq!(t.create, Duration::ZERO);
    }
}
