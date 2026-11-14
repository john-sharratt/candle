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

use candle_transformers::models::piece_key::PieceKey;

use crate::index_pages;
use crate::projection::{Conversation, GroupId, TimelineId, TurnIndex, TurnKey};
use crate::sequence_handle::SequenceId;

use super::pool::{choose_pool, PoolCandidate};
use super::pooled_rounds::PoolPages;
use super::Scheduler;

/// Conversations a reprojection admits to the pool that it was not holding,
/// at most. Admitting reads each turn's index pages, from the log for one
/// whose bytes are not in memory, so the admissions of one boundary are
/// bounded and the rest come at the next.
const POOL_ADMITS_PER_BOUNDARY: usize = 2;

/// What one refresh did, for the reprojection's observe split.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct PoolRefresh {
    /// Conversations the pool holds after it.
    pub pooled: usize,
    /// Of them, kept from the last refresh without reading their held turns
    /// again.
    pub reused: usize,
    /// Turns whose index pages it read.
    pub walked: usize,
}

/// The turns of a pooled conversation a refresh reads: from `walked`, the
/// count the slot's pool already holds pages for, when every key its layout
/// names is still held; from the start otherwise. A conversation's turns are
/// append-only, so held pages past a shrunken count would name turns that are
/// gone, and that conversation is walked whole.
pub(super) fn walk_from(walked: Option<u32>, turns: u32, layout_held: bool) -> u32 {
    match walked {
        Some(w) if layout_held && w <= turns => w,
        _ => 0,
    }
}

impl Scheduler {
    /// Choose `slot`'s pool from `candidates` — what the belief scan scored
    /// this reprojection, per group — and hold its pieces' sampled-layer
    /// index pages on the model.
    ///
    /// A conversation the pool already holds keeps the pages it holds: only
    /// the turns it gained since the last refresh are read and held. Reading
    /// every turn's pages again at every reprojection — from the log, for a
    /// page not in memory — was most of a reprojection's time on a long
    /// conversation, and the decode waited on it.
    ///
    /// The pool only advises: a page that cannot be read or held leaves its
    /// conversation out of this pool, and nothing about the slot changes.
    pub(super) fn refresh_pool(
        &mut self,
        slot: SequenceId,
        conversation: &Conversation,
        candidates: &[(GroupId, Vec<(TurnKey, f32)>)],
        lifts: &HashMap<TimelineId, f32>,
    ) -> PoolRefresh {
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
        let (mut prior, mut prior_walked) = self
            .slot_projection_state
            .get_mut(&slot)
            .map(|s| {
                (
                    std::mem::take(&mut s.pool),
                    std::mem::take(&mut s.pool_walked),
                )
            })
            .unwrap_or_default();
        // What the model holds for the slot: a held conversation is reused only
        // where every page its layout names is still there.
        let model_held: HashSet<PieceKey> = self.model.pooled_keys(slot.0).into_iter().collect();
        let mut keep = Vec::new();
        let mut kept: PoolPages = HashMap::new();
        let mut kept_walked: HashMap<TimelineId, u32> = HashMap::new();
        let mut stats = PoolRefresh::default();
        let mut admitted = 0usize;
        for tl in pool {
            let fresh = !prior.contains_key(&tl);
            if fresh && admitted == POOL_ADMITS_PER_BOUNDARY {
                continue;
            }
            let turns = turns_of.get(&tl).copied().unwrap_or(0);
            let prior_layout = prior.remove(&tl).unwrap_or_default();
            let layout_held = prior_layout
                .values()
                .flatten()
                .all(|(_, key)| model_held.contains(key));
            let from = walk_from(prior_walked.remove(&tl), turns, layout_held);
            let mut keys = Vec::new();
            let mut layout: HashMap<TurnIndex, Vec<(usize, PieceKey)>> = HashMap::new();
            if from > 0 {
                keys.extend(prior_layout.values().flatten().map(|(_, key)| *key));
                layout = prior_layout;
                stats.reused += 1;
            }
            let mut whole = true;
            for i in from..turns {
                stats.walked += 1;
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
                for ((width, bytes), key) in framed.iter().zip(pages.page_keys()) {
                    match self.model.pool_positional_state(slot.0, bytes, key) {
                        Ok(_) => {
                            keys.push(*key);
                            layout.entry(TurnIndex(i)).or_default().push((*width, *key));
                        }
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
                kept.insert(tl, layout);
                kept_walked.insert(tl, turns);
                admitted += fresh as usize;
            }
        }
        stats.pooled = kept.len();
        if let Err(e) = self.model.retain_pooled(slot.0, &keep) {
            tracing::warn!(slot = slot.0, error = %e, "pool: dropped pieces not released");
        }
        // Each pooled conversation's K/V to the warm tier, so a round the fill
        // later chooses loads at PCIe speed rather than from the log. Turns
        // already warm or hot cost a lookup; what the warm share has no room
        // for stays cold.
        let warm: Vec<TurnKey> = kept
            .keys()
            .flat_map(|&tl| {
                (0..turns_of.get(&tl).copied().unwrap_or(0))
                    .map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect();
        self.persist_triggers.warm_turns(warm);
        // The pool's pages go with the slot's projection state, which the
        // assembly declares the pooled rounds from, and leave with it. A slot
        // with no state yet has placed nothing to hold a pool beside.
        if let Some(state) = self.slot_projection_state.get_mut(&slot) {
            state.pool = kept;
            state.pool_walked = kept_walked;
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::walk_from;

    /// A held conversation whose pages are all still held reads only the
    /// turns it gained; one it never walked, one whose pages the model let
    /// go, or one whose turn count shrank is read whole.
    #[test]
    fn a_held_conversation_reads_only_its_new_turns() {
        assert_eq!(walk_from(Some(5), 7, true), 5);
        assert_eq!(walk_from(Some(7), 7, true), 7);
        assert_eq!(walk_from(None, 7, true), 0);
        assert_eq!(walk_from(Some(5), 7, false), 0);
        assert_eq!(walk_from(Some(9), 7, true), 0);
    }
}
