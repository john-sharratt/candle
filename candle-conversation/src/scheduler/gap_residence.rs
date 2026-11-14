//! Sealed residences the seal gap could not finish
//! (`docs/kv_quantize_in_the_gap.md` §5). A turn's seal quantizes its last
//! blocks in gaps of its own, and what the ground could not cover is stored
//! native. Each housekeeping pass then quantizes one such residence's native
//! chunks in a gap and swaps the result in over the hot copy it read, by
//! lineage — so a compaction that relocated the copy meanwhile is installed
//! over, and a residence that moved on keeps what it has.

use super::turn_seal::{turn_seal, GapSeal};
use super::*;

impl Scheduler {
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
