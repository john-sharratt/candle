//! The gap's work on sealed residences (`docs/kv_quantize_in_the_gap.md` §5):
//! the turns the seal gap could not finish, and the idle demote. A turn's seal quantizes its last
//! blocks in gaps of its own, and what the ground could not cover is stored
//! native. Each housekeeping pass then quantizes one such residence's native
//! chunks in a gap and swaps the result in over the hot copy it read, by
//! lineage — so a compaction that relocated the copy meanwhile is installed
//! over, and a residence that moved on keeps what it has.

use super::turn_seal::{turn_seal, GapSeal};
use super::*;
use crate::substrate::IDLE_DEMOTE_GRACE_EPOCHS;

impl Scheduler {
    /// Drop the hot copy of every residence no wave has attended for
    /// [`IDLE_DEMOTE_GRACE_EPOCHS`] whose warm copy holds the same bytes, on
    /// every substrate this scheduler serves. Everything it takes is out of
    /// the working set twice over — by the epoch stamp and by its pins — and
    /// the next `elevate_to_hot` lifts it again.
    pub(super) fn demote_idle_hot(&mut self) {
        for conversation in self.distinct_substrates() {
            let shed = conversation
                .read()
                .demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS);
            if shed > 0 {
                tracing::debug!(
                    target: "candle_conversation::persistence::tier",
                    residences = shed,
                    "idle demote: dropped the hot copy of residences no wave has attended"
                );
            }
        }
    }

    /// Quantize the oldest pending-warm turn residence that still holds
    /// native chunks, across every substrate this scheduler serves. Nothing
    /// when none does.
    pub(super) fn quantize_a_native_residence(&mut self) {
        let session_policy = self.session.seal_policy();
        for conversation in self.distinct_substrates() {
            let Some((idx, hot, cc)) = conversation.read().native_pending_warm() else {
                continue;
            };
            let GapSeal::Policy(policy) = turn_seal(session_policy.as_ref(), cc) else {
                continue;
            };
            let t = Instant::now();
            let done = self.session.gap_quantize_sealed(&policy, &hot[..]);
            self.kv_gap.spent(t.elapsed());
            match done {
                Ok(Some(done)) => {
                    self.kv_gap.sealed(&done.caught);
                    self.kv_gap.residence(&done);
                    if !conversation
                        .read()
                        .install_gap_quantized(idx, &hot, done.seqs)
                    {
                        tracing::debug!(
                            target: "candle_conversation::scheduler::kv_gap",
                            residence = idx.0,
                            "gap-quantized residence moved on before its install"
                        );
                    }
                }
                Ok(None) => {}
                Err(e) => tracing::warn!(
                    target: "candle_conversation::scheduler::kv_gap",
                    residence = idx.0,
                    "gap quantize of a native residence failed: {e}"
                ),
            }
            return;
        }
    }
}
