//! Catching the outstanding gap up before a slot's blocks move
//! (`docs/kv_quantize_in_the_gap.md` §3.5). A gap's rows name a block by its
//! sequence and index; a view finalized onto its parent, or a tail a
//! reprojection snapshots and rebuilds, moves those blocks out from under the
//! rows, and the catch-up then discards them as stale — converted work thrown
//! away, its blocks left native for the next gap to convert again. So the
//! scheduler waits the gap out and installs it first. A gap is a few
//! milliseconds of device work, and it is waited for only when one is
//! outstanding at such a move.

use super::*;

impl Scheduler {
    /// Wait for the outstanding gap and put its converted blocks in place.
    /// Nothing when none is outstanding.
    pub(super) fn catch_up_outstanding_gap(&mut self) {
        if !self.session.gap_outstanding() {
            return;
        }
        let t = Instant::now();
        let caught = self
            .session
            .gap_wait()
            .and_then(|()| self.session.gap_catch_up());
        match caught {
            Ok(c) => self.kv_gap.caught_up(&c),
            Err(e) => tracing::warn!(
                target: "candle_conversation::scheduler::kv_gap",
                "gap catch-up before a move failed: {e}"
            ),
        }
        self.kv_gap.spent(t.elapsed());
    }
}
