//! A decode step's knowledge-base decision (`docs/placed_section_layout.md`
//! §3.4, §4.4): the step's hits, read from the mapped summary without a wait,
//! rate each placed section and each resident one out of scope. A placed
//! section that has stayed under the eviction bar for its steps leaves its
//! range; a resident one at or over the entry bar enters it in front,
//! strongest first. Either way the next wave lays the range out again
//! (`section_relayout`), and the step after it reads the new layout.
//!
//! The read is the host's alone — the device published the counters after
//! the wave's last sampled layer, and the host already waits for the wave's
//! token ids, which follow — so the step costs no launch, no copy and no
//! wait (`AGENTS.md` invariants 3–4). The counts are summed into the slot's
//! unread parts, so the boundary's reprojection still folds every step's
//! hits into its momentum.

use std::collections::HashSet;

use candle_transformers::models::batched_inference::ManagedBatchedModel;

use super::projection_assembler::{fold_part, section_rates, SlotState};
use super::reprojection::ReprojectionPolicy;
use super::SequenceId;
use crate::files::scope::{Bars, Candidate};
use crate::files::SectionKey;

/// One step of `slot`, whose projection state is `state`, under `policy`:
/// whether the range's layout changed, so the view must be laid out again.
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
    if hits.queries == 0 && hits.shadow_queries == 0 {
        return false;
    }
    let keys = state.hit_keys.clone();
    let rates = section_rates(&keys, &hits, &state.kb_rounds);
    let placed_read = hits.queries > 0;
    fold_part(&mut state.unread_hits, &keys, hits);
    let bars = Bars::KNOWLEDGE_BASE;
    let target = policy.target.timeline;
    let (leaving, candidates) = {
        let read = policy.substrate.read();
        let Some(scope) = read.files_set_of(target).and_then(|s| s.scope.as_ref()) else {
            return false;
        };
        // Only a read that counted the selection's queries says how the
        // placed sections were read.
        let leaving = match placed_read {
            true => {
                state
                    .evict_watch
                    .observe(scope, |k| rates.get(&k).copied().unwrap_or(0.0), &bars)
            }
            false => Vec::new(),
        };
        let candidates: Vec<Candidate> = state
            .kb_resident
            .iter()
            .filter(|r| !scope.holds(r.key) && rates.contains_key(&r.key))
            .map(|r| Candidate {
                key: r.key,
                tokens: r.tokens,
            })
            .collect();
        (leaving, candidates)
    };
    let read: HashSet<SectionKey> = candidates
        .iter()
        .filter(|c| rates[&c.key] > 0.0)
        .map(|c| c.key)
        .collect();
    state.kb_read.extend(read);
    let entering = candidates.iter().any(|c| rates[&c.key] >= bars.entry);
    if leaving.is_empty() && !entering {
        return false;
    }
    let rate = |k: SectionKey| rates.get(&k).copied().unwrap_or(0.0);
    let changed = policy
        .substrate
        .update(|s| s.files_step(target, &leaving, rate, &candidates, &bars));
    if changed {
        state.layout_owed = true;
        tracing::debug!(
            target: "candle_conversation::scheduler::reproject",
            slot = slot.0,
            left = leaving.len(),
            entering = candidates.iter().filter(|c| rates[&c.key] >= bars.entry).count(),
            "the knowledge base's range changed on its sections' hit rates",
        );
    }
    changed
}
