//! A decode step's knowledge-base decision (`docs/placed_section_layout.md`
//! §3.2, §3.4, §4.4): the step's hits, read from the mapped summary without a
//! wait, rate each placed section and each resident one out of scope, and
//! each rated section's momentum in the files set takes the step, after the
//! rates the question's first reading gave it (`first_reading`). A placed
//! section that has
//! stayed under the eviction bar for its steps leaves its range; a resident
//! one at or over the entry bar enters it at its rank; and the range is
//! ordered by momentum, strongest beside the question, so a section the
//! decode locks onto is drawn to the top within a few steps. Any change and
//! the next wave lays the range out again (`section_relayout`), and the step
//! after it reads the new layout.
//!
//! The read is the host's alone — the device published the counters after
//! the wave's last sampled layer, and the host already waits for the wave's
//! token ids, which follow — so the step costs no launch, no copy and no
//! wait (`AGENTS.md` invariants 3–4). The counts are summed into the slot's
//! unread parts, so the boundary's reprojection still folds every step's
//! hits into its momentum.

use std::collections::{HashMap, HashSet};

use candle_transformers::models::batched_inference::ManagedBatchedModel;

use super::projection_assembler::{fold_part, section_rates, selection_rate, SlotState};
use super::reprojection::ReprojectionPolicy;
use super::SequenceId;
use crate::files::scope::{shares, Candidate};
use crate::files::{FilesSet, SectionKey};

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
    let chance = selection_rate(&hits);
    let placed_read = hits.queries > 0;
    fold_part(&mut state.unread_hits, &keys, hits);
    let target = policy.target.timeline;
    let (leaving, candidates, bars, engaged) = {
        let read = policy.substrate.read();
        let Some(set) = read.files_set_of(target) else {
            return false;
        };
        let Some(scope) = set.scope.as_ref() else {
            return false;
        };
        let bars = set.step_bars(&rates);
        // The range is engaged while its sections are read, on the mean, at
        // least as the selection reads the whole slot: below it a share of
        // almost nothing would lock the largest of it.
        let placed: Vec<f32> = scope
            .sections()
            .filter_map(|e| rates.get(&e.key).copied())
            .collect();
        let engaged = !placed.is_empty()
            && chance.is_some_and(|c| placed.iter().sum::<f32>() / placed.len() as f32 >= c);
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
        (leaving, candidates, bars, engaged)
    };
    let read: HashSet<SectionKey> = candidates
        .iter()
        .filter(|c| rates[&c.key] > 0.0)
        .map(|c| c.key)
        .collect();
    state.kb_read.extend(read);
    // Every rated section's momentum takes this step, what orders the range.
    let changed = policy
        .substrate
        .update(|s| s.files_step(target, &rates, placed_read, engaged, &leaving, &candidates));
    state.kb_steps += 1;
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
    // The range's momenta, top first, on every change and every
    // `MOMENTA_EVERY`th step: what the lock bar is set from.
    if changed || state.kb_steps % MOMENTA_EVERY == 0 {
        let read = policy.substrate.read();
        if let Some(set) = read.files_set_of(target) {
            tracing::debug!(
                target: "candle_conversation::scheduler::reproject",
                slot = slot.0,
                step = state.kb_steps,
                changed,
                selection_rate = chance.unwrap_or(0.0),
                engaged,
                locked = set.locked.len(),
                momenta = %range_momenta(set, &rates),
                "the knowledge base's range, top first (share % of its attention/rate, * locked)",
            );
        }
    }
    changed
}

/// Steps between two momenta lines of a range that did not change.
const MOMENTA_EVERY: u64 = 32;

/// `set`'s placed sections top first, each its share of the range's attention
/// in percent and its rate this step on the band, a locked one marked `*`:
/// one token per section, rounded.
fn range_momenta(set: &FilesSet, rates: &HashMap<SectionKey, f32>) -> String {
    let Some(scope) = set.scope.as_ref() else {
        return String::new();
    };
    let share = shares(&set.placed_momenta());
    scope
        .sections()
        .map(|e| {
            let r = rates.get(&e.key).copied().unwrap_or(0.0);
            let lock = if set.locked.contains(&e.key) { "*" } else { "" };
            match share.get(&e.key) {
                Some(s) => format!("{:.1}/{r:.0}{lock}", 100.0 * s),
                None => format!("-/{r:.0}"),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
