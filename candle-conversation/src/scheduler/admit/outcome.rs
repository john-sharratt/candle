//! How a fill pass ended, and the log line that says why an offer was refused.
//!
//! A refusal for the weight zone means the device is the bound and the producer
//! should back off. A refusal because the wave is already as fast as it will go
//! means the engine is working well, and the queue behind it rides the next
//! wave. Each refusal is logged at debug with the claim, the tier and both
//! floors it was read against, because a summary line cannot say which item was
//! refused or which rule refused it. A band stops at its first refusal, so
//! refusals are rare on a healthy engine.

use super::*;

/// What one fill pass took.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Filled {
    pub decodes: usize,
    pub prefills: usize,
    pub sections: usize,
    /// The pass ran the fast path: nothing had finished, so nothing was offered.
    pub skipped: bool,
    /// An admission was refused because it would have reached the weight zone —
    /// the floor, the tier's placeable width, or the decode cap. The producer's
    /// backpressure signal: the device is the bound.
    pub stopped_on_weights: bool,
    /// An admission was refused because the wave was already going as fast as
    /// it is going to go — another row would not pay for the residency it
    /// dislodges. **Not** backpressure: the engine is working well and the
    /// queue behind it simply rides the next wave.
    pub stopped_on_rate: bool,
}

/// Log one refused offer with everything that produced the answer.
///
/// **A refusal is the decision worth seeing, and it is the one the summary line
/// cannot carry.** The per-fill line says a band stopped; it cannot say which
/// item, what it would have cost, what the model projected, or which of six
/// rules said no — and those are exactly the questions asked of a run that
/// admitted nothing for an hour. One line per refusal, at debug, and refusals
/// are rare on a healthy engine by construction: the band stops at the first.
#[allow(clippy::too_many_arguments)]
pub(super) fn log_refusal(
    rate: &mut WaveRate,
    kind: Kind,
    prio: DecodePriority,
    cost: &Cost,
    before: u64,
    after: u64,
    budget: &Budget,
    refusal: rate::Refusal,
) {
    rate.note_refusal(refusal, after, budget.prefill_floor);
    tracing::debug!(
        target: "candle_conversation::scheduler::admission",
        kind = ?kind,
        prio = ?prio,
        rows = cost.rows,
        kv_mib = cost.kv >> 20,
        recurrent_mib = cost.recurrent >> 20,
        tier_mib = cost.activations >> 20,
        claimed_mib = cost.claimed_bytes() >> 20,
        // What the judgement was actually made on — the claim plus the tier.
        // Logged beside `claimed_mib` because the two differ by exactly the
        // term that used to be missing, and a run that refuses unexpectedly
        // wants to see which of them moved.
        dislodged_mib = cost.dislodged_bytes() >> 20,
        resident_before_mib = before >> 20,
        resident_after_mib = after >> 20,
        // Both floors, because a refusal is read against the one its kind stands
        // on and the gap between them is the prefill's extra room.
        prefill_floor_mib = budget.prefill_floor >> 20,
        decode_floor_mib = budget.decode_floor >> 20,
        room_mib = before.saturating_sub(budget.prefill_floor) >> 20,
        max_rows = budget.max_rows,
        max_decodes = budget.max_decodes,
        refusal = ?refusal,
        "offer refused",
    );
}

impl Filled {
    /// Offers this pass has admitted so far, of every kind.
    pub(super) fn admitted(&self) -> usize {
        self.decodes + self.prefills + self.sections
    }

    /// Record which kind of refusal ended a band.
    pub(super) fn note(&mut self, refusal: rate::Refusal) {
        use rate::Refusal;
        match refusal {
            Refusal::Worse { .. } | Refusal::Saturated { .. } => self.stopped_on_rate = true,
            Refusal::Floor { .. } | Refusal::Cap { .. } | Refusal::DecodeCap { .. } => {
                self.stopped_on_weights = true
            }
            // The wave latched full on an earlier refusal, which was itself
            // recorded when it happened. Nothing new to say.
            Refusal::Full => {}
        }
    }
}
