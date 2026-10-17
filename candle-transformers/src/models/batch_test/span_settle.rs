//! The span housekeeping a scheduler runs between forwards, run by the gate
//! between decode steps.
//!
//! A prefill leaves holes behind it: the recurrent stores' prefill scratch goes
//! back on the first decode step, when the sequences stop prefilling, and the KV
//! arenas claimed while it stood sit above the ground it held. The weight side can take only what lies
//! above the highest live region, so until something packs the span the expert
//! zone stays where the prefill pushed it — measured on Flash-Next at ×8 as 69
//! free regions under a frontier at 159 with 90 regions live, and the zone ~1.1
//! GiB short through the whole decode. A daemon's scheduler packs between forwards
//! (`Scheduler::compact_kv_if_fragmented`); a gate that does not measures a decode
//! below the ceiling it reports.
//!
//! The pass is chosen the way the scheduler chooses it: by who stands at the
//! frontier. A KV arena there is the KV pass's to move; otherwise the recurrent
//! state pass packs the store arenas. Each pass is followed by the weight side's
//! reclaim, and the loop runs until a pass leaves the frontier where it found it.

use std::time::Duration;

use candle::Result;
use candle_nn::kv_cache::REGION_BYTES;

use crate::models::batched_inference::{BatchedInferenceSession, ManagedBatchedModel};

/// Wall-clock one KV pass may spend planning and claiming — the scheduler's.
const KV_BUDGET: Duration = Duration::from_millis(80);

/// Layer states one recurrent pass may move — the scheduler's.
const MAX_STATE_MOVES: usize = 256;

/// Free regions below the frontier under which a pass is not worth its copies —
/// the scheduler's `MIN_FREEABLE_ARENAS`.
const MIN_HOLES: usize = 8;

/// Passes at most, so a span that will not settle cannot hold the gate.
const MAX_PASSES: usize = 16;

/// What one settle did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SettleOutcome {
    /// Passes run — KV and recurrent together.
    pub passes: usize,
    /// The region pool's frontier before and after.
    pub frontier_before: usize,
    pub frontier_after: usize,
    /// The weight side's bytes before and after its reclaims.
    pub weight_bytes_before: usize,
    pub weight_bytes_after: usize,
}

impl SettleOutcome {
    /// The gate's one-line account of the settle.
    pub fn line(&self) -> String {
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        format!(
            "  span settle: {} pass(es), frontier {} → {}, weight zone {:.1} → {:.1} MiB \
             ({:+.0} regions)",
            self.passes,
            self.frontier_before,
            self.frontier_after,
            mib(self.weight_bytes_before),
            mib(self.weight_bytes_after),
            (self.weight_bytes_after as f64 - self.weight_bytes_before as f64)
                / REGION_BYTES as f64,
        )
    }
}

/// The settle a decode runs between its steps, with the scheduler's stall: a pass
/// that left the frontier where it found it is not run again over the same ground
/// (`compaction_stall`). Without it, eight holes under a region no pass can move
/// cost a pass inside every step of a ×1 decode for nothing.
#[derive(Debug, Default)]
pub struct SpanSettler {
    /// `(live, frontier)` a pass could not improve.
    stalled_at: Option<(usize, usize)>,
}

impl SpanSettler {
    /// [`settle_span`], unless the pool stands exactly where a pass last failed
    /// to lower it. `None` when nothing ran.
    pub fn settle<M>(
        &mut self,
        session: &BatchedInferenceSession,
        model: &M,
    ) -> Result<Option<SettleOutcome>>
    where
        M: ManagedBatchedModel + ?Sized,
    {
        let Some(stats) = session.kv_region_stats() else {
            return Ok(None);
        };
        if self.stalled_at == Some((stats.live, stats.live_watermark)) {
            return Ok(None);
        }
        let out = settle_span(session, model)?;
        if let (Some(o), Some(after)) = (out, session.kv_region_stats()) {
            self.stalled_at = (o.passes > 0 && o.frontier_after >= o.frontier_before)
                .then_some((after.live, after.live_watermark));
        }
        Ok(out.filter(|o| o.passes > 0))
    }
}

/// Pack the span and hand what packing frees to the weight side, between
/// forwards. `None` when the device keeps no region pool.
pub fn settle_span<M>(session: &BatchedInferenceSession, model: &M) -> Result<Option<SettleOutcome>>
where
    M: ManagedBatchedModel + ?Sized,
{
    let Some(start) = session.kv_region_stats() else {
        return Ok(None);
    };
    let mut out = SettleOutcome {
        passes: 0,
        frontier_before: start.live_watermark,
        frontier_after: start.live_watermark,
        weight_bytes_before: start.weight_bytes,
        weight_bytes_after: start.weight_bytes,
    };
    while out.passes < MAX_PASSES {
        let Some(stats) = session.kv_region_stats() else {
            break;
        };
        let frontier = stats.live_watermark;
        if frontier.saturating_sub(stats.live) < MIN_HOLES {
            break;
        }
        let kv_on_top = session.kv_top_rank().map(|t| t + 1) == Some(frontier);
        if kv_on_top {
            if session.compact_kv(KV_BUDGET, &mut |_| Ok(())).is_err() {
                break;
            }
        } else {
            model.compact_recurrent(MAX_STATE_MOVES)?;
        }
        model.reclaim_spare_ground();
        out.passes += 1;
        let after = session
            .kv_region_stats()
            .map_or(frontier, |s| s.live_watermark);
        if after >= frontier {
            break;
        }
    }
    if let Some(end) = session.kv_region_stats() {
        out.frontier_after = end.live_watermark;
        out.weight_bytes_after = end.weight_bytes;
    }
    Ok(Some(out))
}
