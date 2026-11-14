//! The gap quantize in the batched gate, as the scheduler runs it
//! (`docs/kv_quantize_in_the_gap.md` §3, §7): a catch-up and a quantize in
//! the gap after every forward, inside the timed windows, and what they did
//! summed over a config for its line.

use std::fmt::Write as _;

use candle::{Device, Result};
use candle_nn::kv_cache::{GapCatchUp, GapKind, GapQuantize};

use crate::models::batched_inference::BatchedInferenceSession;

/// Gaps a settle runs at most before giving up on the cycle going quiet.
const SETTLE_GAPS: usize = 64;

/// What the gaps of one config did.
#[derive(Debug, Default)]
pub struct GapTally {
    gaps: usize,
    jobs: usize,
    retried: usize,
    installed: usize,
    deferred: usize,
    stale: usize,
    moved: usize,
    outstanding: usize,
    short_slots: usize,
    refused: Option<String>,
    errored: bool,
}

impl GapTally {
    /// The head of a gap: put the outstanding gap's blocks in place.
    pub fn catch_up(&mut self, session: &mut BatchedInferenceSession) -> Result<GapCatchUp> {
        let c = session.gap_catch_up()?;
        if c.caught_up {
            self.installed += c.installed;
            self.deferred += c.deferred;
            self.stale += c.stale;
            self.moved += c.moved;
            self.errored |= c.errored;
        }
        Ok(c)
    }

    /// The tail of a gap: quantize what the forwards completed.
    pub fn quantize(
        &mut self,
        session: &mut BatchedInferenceSession,
        kind: GapKind,
    ) -> Result<GapQuantize> {
        let q = session.gap_quantize(kind)?;
        if q.outstanding {
            self.outstanding += 1;
        } else if q.jobs > 0 {
            self.gaps += 1;
            self.jobs += q.jobs;
            self.retried += q.retried;
            self.short_slots += q.short.iter().sum::<usize>();
            if self.refused.is_none() {
                self.refused = q
                    .refusals
                    .first()
                    .map(|r| format!("class {} short {}: {}", r.class, r.short, r.why));
            }
        }
        Ok(q)
    }

    /// Run gaps until the cycle is quiet — none outstanding and nothing left
    /// to take — waiting for each on `device`. Outside every clock: what it
    /// leaves is what the compression figures read.
    pub fn settle(&mut self, session: &mut BatchedInferenceSession, device: &Device) -> Result<()> {
        for _ in 0..SETTLE_GAPS {
            device.synchronize()?;
            self.catch_up(session)?;
            let q = self.quantize(session, GapKind::Decode)?;
            if q.jobs == 0 && !session.gap_outstanding() {
                return Ok(());
            }
        }
        candle::bail!("the gap cycle did not go quiet in {SETTLE_GAPS} gaps")
    }

    /// The config's gap line.
    pub fn line(&self, label: &str) -> String {
        let mut s = format!(
            "[{label}] gap: {} gaps, {} jobs ({} retried) | installed {}, deferred {}, stale {}, \
             moved {} | {} skipped outstanding | lease short {} slots",
            self.gaps,
            self.jobs,
            self.retried,
            self.installed,
            self.deferred,
            self.stale,
            self.moved,
            self.outstanding,
            self.short_slots
        );
        if let Some(r) = &self.refused {
            let _ = write!(s, " (first refusal: {r})");
        }
        if self.errored {
            s.push_str(" | the claim met a code its tables do not cover");
        }
        s
    }
}
