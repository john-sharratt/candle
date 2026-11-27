//! A decode step's knowledge-base decision (`docs/placed_section_layout.md`
//! §3.4, §4.4): the step's hits, read from the mapped summary without a wait,
//! rate each placed section; a section that has stayed under the eviction bar
//! for its steps leaves its range, and the next wave's reprojection places
//! the range without it.
//!
//! The read is the host's alone — the device published the counters after
//! the wave's last sampled layer, and the host already waits for the wave's
//! token ids, which follow — so the step costs no launch, no copy and no
//! wait (`AGENTS.md` invariants 3–4). The counts are summed into the slot's
//! unread parts, so the boundary's reprojection still folds every step's
//! hits into its momentum.

use candle_transformers::models::batched_inference::ManagedBatchedModel;

use super::projection_assembler::{fold_part, section_rates, SlotState};
use super::reprojection::ReprojectionPolicy;
use super::SequenceId;
use crate::files::scope::Bars;

/// One step of `slot`, whose projection state is `state`, under `policy`:
/// whether a section left its range, so the view must be reprojected.
pub(super) fn step_sections(
    model: &mut (dyn ManagedBatchedModel + Send),
    state: Option<&mut SlotState>,
    slot: SequenceId,
    policy: &ReprojectionPolicy,
) -> bool {
    let Some(state) = state else {
        return false;
    };
    if state.kb_rounds.is_empty() {
        return false;
    }
    let hits = match model.take_hits(slot.0) {
        Ok(Some(h)) => h,
        Ok(None) => return false,
        Err(e) => {
            tracing::debug!(slot = slot.0, error = %e, "a step's hits not read");
            return false;
        }
    };
    // A read that counted no query says nothing about any section.
    if hits.queries == 0 {
        return false;
    }
    let keys = state.hit_keys.clone();
    let rates = section_rates(&keys, &hits, &state.kb_rounds);
    fold_part(&mut state.unread_hits, &keys, hits);
    let target = policy.target.timeline;
    let leaving = {
        let read = policy.substrate.read();
        let Some(scope) = read.files_set_of(target).and_then(|s| s.scope.as_ref()) else {
            return false;
        };
        state.evict_watch.observe(
            scope,
            |k| rates.get(&k).copied().unwrap_or(0.0),
            &Bars::KNOWLEDGE_BASE,
        )
    };
    if leaving.is_empty() {
        return false;
    }
    tracing::debug!(
        target: "candle_conversation::scheduler::reproject",
        slot = slot.0,
        sections = leaving.len(),
        "sections left the knowledge base's range: their hit rate stayed under the bar",
    );
    policy.substrate.update(|s| s.files_evict(target, &leaving))
}
