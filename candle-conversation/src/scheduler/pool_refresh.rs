//! The pool a reprojection leaves behind a dialogue's slot
//! (`docs/hit_driven_projection.md` §7.2).
//!
//! At each reprojection the pool is chosen again ([`super::pool::choose_pool`]):
//! the conversations the slot places are its floor, and provenance's other
//! candidates follow by score, lifted by the hits their rounds ran up. Each
//! pooled conversation's turns hold their index pages at the sampled layers on
//! the model — placed once per card, shared with every slot and pool that holds
//! the same piece — and a conversation the pool drops lets go of them.

use std::collections::{HashMap, HashSet};

use crate::index_pages;
use crate::projection::{Conversation, GroupId, TimelineId, TurnIndex, TurnKey};
use crate::sequence_handle::SequenceId;

use super::pool::{choose_pool, PoolCandidate};
use super::Scheduler;

/// Conversations a reprojection admits to the pool that it was not holding,
/// at most. Admitting reads each turn's index pages, from the log for one
/// whose bytes are not in memory, so the admissions of one boundary are
/// bounded and the rest come at the next.
const POOL_ADMITS_PER_BOUNDARY: usize = 2;

impl Scheduler {
    /// Choose `slot`'s pool from `candidates` — what the belief scan scored
    /// this reprojection, per group — and hold its pieces' sampled-layer
    /// index pages on the model.
    ///
    /// The pool only advises: a page that cannot be read or held leaves its
    /// conversation out of this pool, and nothing about the slot changes.
    pub(super) fn refresh_pool(
        &mut self,
        slot: SequenceId,
        conversation: &Conversation,
        candidates: &[(GroupId, Vec<(TurnKey, f32)>)],
        lifts: &HashMap<TimelineId, f32>,
    ) {
        let placed: Vec<TimelineId> = self
            .slot_projection_state
            .get(&slot)
            .map(|s| s.placed_timelines())
            .unwrap_or_default();
        let (floors, scored, turns_of) = {
            let read = conversation.read();
            let tokens = |tl: TimelineId| -> usize {
                (0..read.turn_count(tl))
                    .map(|i| read.turn_token_count_of(tl, TurnIndex(i)))
                    .sum()
            };
            let floors: Vec<(TimelineId, usize)> =
                placed.iter().map(|&tl| (tl, tokens(tl))).collect();
            let mut best: HashMap<TimelineId, f32> = HashMap::new();
            for (_, turns) in candidates {
                for (key, score) in turns {
                    let at = best.entry(key.timeline).or_insert(*score);
                    *at = at.max(*score);
                }
            }
            let scored: Vec<PoolCandidate> = best
                .into_iter()
                .map(|(piece, score)| PoolCandidate {
                    piece,
                    tokens: tokens(piece),
                    score: score * lifts.get(&piece).copied().unwrap_or(1.0),
                })
                .collect();
            let turns_of: HashMap<TimelineId, u32> = floors
                .iter()
                .map(|(tl, _)| *tl)
                .chain(scored.iter().map(|c| c.piece))
                .map(|tl| (tl, read.turn_count(tl)))
                .collect();
            (floors, scored, turns_of)
        };
        let pool = choose_pool(&floors, &scored, self.pool_tokens);
        let held: HashSet<TimelineId> = self.pooled_pieces.get(&slot).cloned().unwrap_or_default();
        let mut keep = Vec::new();
        let mut kept: HashSet<TimelineId> = HashSet::new();
        let mut admitted = 0usize;
        for tl in pool {
            let fresh = !held.contains(&tl);
            if fresh && admitted == POOL_ADMITS_PER_BOUNDARY {
                continue;
            }
            let mut keys = Vec::new();
            let mut whole = true;
            for i in 0..turns_of.get(&tl).copied().unwrap_or(0) {
                let pages = match conversation.turn_index_page(tl, TurnIndex(i)) {
                    Ok(Some(p)) => p,
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::warn!(timeline = tl.raw(), turn = i, error = %e,
                            "pool: a turn's index page could not be read; the conversation is \
                             left out of this pool");
                        whole = false;
                        break;
                    }
                };
                let Ok(framed) = index_pages::decode(pages.bytes()) else {
                    whole = false;
                    break;
                };
                for ((_, bytes), key) in framed.iter().zip(pages.page_keys()) {
                    match self.model.pool_positional_state(slot.0, bytes, key) {
                        Ok(_) => keys.push(*key),
                        Err(e) => {
                            tracing::warn!(timeline = tl.raw(), error = %e,
                                "pool: a page could not be held; the conversation is left out");
                            whole = false;
                            break;
                        }
                    }
                }
                if !whole {
                    break;
                }
            }
            if whole {
                keep.extend(keys);
                kept.insert(tl);
                admitted += fresh as usize;
            }
        }
        if let Err(e) = self.model.retain_pooled(slot.0, &keep) {
            tracing::warn!(slot = slot.0, error = %e, "pool: dropped pieces not released");
        }
        // Each pooled conversation's K/V to the warm tier, so a round the fill
        // later chooses loads at PCIe speed rather than from the log. Turns
        // already warm or hot cost a lookup; what the warm share has no room
        // for stays cold.
        let warm: Vec<TurnKey> = kept
            .iter()
            .flat_map(|&tl| {
                (0..turns_of.get(&tl).copied().unwrap_or(0))
                    .map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect();
        self.persist_triggers.warm_turns(warm);
        self.pooled_pieces.insert(slot, kept);
    }
}
