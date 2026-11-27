//! The pool a reprojection leaves behind a dialogue's slot
//! (`docs/hit_driven_projection.md` §7.2 step 1).
//!
//! At each reprojection the pool is taken again ([`super::pool::pool_order`]):
//! the conversations the slot places, then the dialogue's working set — the
//! pieces it chose by strength within the card's size. Each pooled
//! conversation's turns hold their index pages at the sampled layers on the
//! model — placed once per card, shared with every slot and pool that holds the
//! same piece — and a conversation the pool drops lets go of them. No K/V moves
//! for a pooled piece: its rounds' K/V comes up only when the fill places one.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use candle_transformers::models::piece_key::PieceKey;

use crate::index_page_cache;
use crate::index_pages::{self, StoredPages};
use crate::projection::{Conversation, TimelineId, TurnIndex, WorkingSetShare};
use crate::sequence_handle::SequenceId;

use super::pool::pool_order;
use super::pool_fetch::fetch_pages;
use super::pooled_rounds::PoolPages;
use super::profile;
use super::Scheduler;

/// How many of each kind's strongest pieces the per-kind line names.
const NAMED_PER_KIND: usize = 8;

/// What one refresh did, for the reprojection's observe split.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct PoolRefresh {
    /// Conversations the pool holds after it.
    pub pooled: usize,
    /// Of them, kept from the last refresh without reading their held turns
    /// again.
    pub reused: usize,
    /// Conversations it admitted that the last refresh did not hold.
    pub admitted: usize,
    /// Conversations the last refresh held that it let go.
    pub dropped: usize,
    /// Turns whose index pages it read.
    pub walked: usize,
    /// Bytes of index pages it read.
    pub read_bytes: usize,
    /// Microseconds the reads took — the `pool:index_read` span.
    pub read_us: u64,
    /// Of `read_us`, fetching the turns' index pages — from memory when held,
    /// off the log when cold.
    pub fetch_us: u64,
    /// Of `read_us`, placing the sampled layers' pages on the device — every
    /// page the refresh read, in one call.
    pub place_us: u64,
    /// Of `walked`, the turns whose pages came off the log: in neither the
    /// substrate nor the index page cache.
    pub cold: usize,
    /// Of `admitted`, pieces taken up from lingering, their pages still held.
    pub revived: usize,
    /// Pieces lingering after it (`pool_linger`).
    pub lingering: usize,
}

/// One pooled conversation's walk in a refresh: the pages of the turns it
/// reads, held until the refresh has placed them all, and the layout it
/// extends.
struct PoolWalk {
    tl: TimelineId,
    turns: u32,
    /// The layout kept from the last refresh when the walk resumes there;
    /// empty when it walks the conversation whole.
    prior_layout: HashMap<TurnIndex, Vec<(usize, PieceKey)>>,
    pages: Vec<(TurnIndex, Arc<StoredPages>)>,
    /// False once a page could not be read, decoded or held: the
    /// conversation is left out of this pool.
    whole: bool,
}

impl PoolWalk {
    fn new(
        tl: TimelineId,
        turns: u32,
        from: u32,
        prior_layout: HashMap<TurnIndex, Vec<(usize, PieceKey)>>,
    ) -> Self {
        Self {
            tl,
            turns,
            prior_layout: if from > 0 {
                prior_layout
            } else {
                HashMap::new()
            },
            pages: Vec::new(),
            whole: true,
        }
    }
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

/// A kind's name in the per-kind line.
fn kind_name(share: WorkingSetShare) -> &'static str {
    match share {
        WorkingSetShare::Readings => "readings",
        WorkingSetShare::Folders => "folders",
        WorkingSetShare::Guides => "guides",
        WorkingSetShare::Journals => "journals",
    }
}

impl Scheduler {
    /// Take `slot`'s pool again — the pieces it places, then `target`'s
    /// working set — and hold its pieces' sampled-layer index pages on the
    /// model.
    ///
    /// **Admission is not throttled**: every piece the working set chose is
    /// pooled at this boundary. A conversation the pool already holds keeps
    /// the pages it holds: only the turns it gained since the last refresh
    /// are read. A new one's pages come from memory when resident and from the
    /// log when cold, in the `pool:index_read` span, whose count, bytes and
    /// time the refresh reports.
    ///
    /// The pool only advises: a page that cannot be read or held leaves its
    /// conversation out of this pool, and nothing about the slot changes.
    pub(super) fn refresh_pool(
        &mut self,
        slot: SequenceId,
        conversation: &Conversation,
        target: TimelineId,
    ) -> PoolRefresh {
        let placed: Vec<TimelineId> = self
            .slot_projection_state
            .get(&slot)
            .map(|s| s.placed_timelines())
            .unwrap_or_default();
        let (pool, turns_of) = {
            let read = conversation.read();
            let members = read
                .working_set(target)
                .map(|ws| ws.members())
                .unwrap_or_default();
            let pool = pool_order(&placed, &members);
            let turns_of: HashMap<TimelineId, u32> =
                pool.iter().map(|&tl| (tl, read.turn_count(tl))).collect();
            (pool, turns_of)
        };
        let (mut prior, mut prior_walked, mut linger) = self
            .slot_projection_state
            .get_mut(&slot)
            .map(|s| {
                (
                    std::mem::take(&mut s.pool),
                    std::mem::take(&mut s.pool_walked),
                    std::mem::take(&mut s.pool_linger),
                )
            })
            .unwrap_or_default();
        linger.age();
        let held_before: HashSet<TimelineId> = prior.keys().copied().collect();
        // What the model holds for the slot: a held conversation is reused only
        // where every page its layout names is still there.
        let model_held: HashSet<PieceKey> = self.model.pooled_keys(slot.0).into_iter().collect();
        let mut keep = Vec::new();
        let mut kept: PoolPages = HashMap::new();
        let mut kept_walked: HashMap<TimelineId, u32> = HashMap::new();
        let mut stats = PoolRefresh::default();
        let reading = profile::span("pool:index_read");
        let t_read = Instant::now();
        // Each pooled conversation's prior layout and the turn its walk starts
        // from, in pool order.
        let mut plans = Vec::with_capacity(pool.len());
        for tl in pool {
            let turns = turns_of.get(&tl).copied().unwrap_or(0);
            // Held last refresh, or lingering since it left (`pool_linger`).
            let (prior_layout, walked) = match prior.remove(&tl) {
                Some(layout) => (layout, prior_walked.remove(&tl)),
                None => match linger.take(tl) {
                    Some((layout, walked)) => {
                        stats.revived += 1;
                        (layout, Some(walked))
                    }
                    None => Default::default(),
                },
            };
            let layout_held = prior_layout
                .values()
                .flatten()
                .all(|(_, key)| model_held.contains(key));
            let from = walk_from(walked, turns, layout_held);
            plans.push((tl, turns, from, prior_layout));
        }
        // Every page the walks below read, fetched at once: a page off the log
        // waits on the disk, and one at a time they waited end to end.
        let wanted: Vec<(TimelineId, TurnIndex)> = plans
            .iter()
            .flat_map(|&(tl, turns, from, _)| (from..turns).map(move |i| (tl, TurnIndex(i))))
            .collect();
        let t_fetch = Instant::now();
        let mut pages_of = fetch_pages(conversation, &wanted);
        stats.fetch_us = t_fetch.elapsed().as_micros() as u64;
        // Walk: each conversation's new turns, their pages held until every
        // one of them is placed.
        let walks: Vec<PoolWalk> = plans
            .into_iter()
            .map(|(tl, turns, from, prior_layout)| {
                let mut walk = PoolWalk::new(tl, turns, from, prior_layout);
                if from > 0 {
                    stats.reused += 1;
                }
                for i in from..turns {
                    stats.walked += 1;
                    let (fetched, from_log) = pages_of
                        .remove(&(tl, TurnIndex(i)))
                        .expect("every walked turn was fetched");
                    let pages = match fetched {
                        Ok(Some(p)) => p,
                        Ok(None) => continue,
                        Err(e) => {
                            tracing::warn!(timeline = tl.raw(), turn = i, error = %e,
                                "pool: a turn's index page could not be read; the conversation \
                                 is left out of this pool");
                            walk.whole = false;
                            break;
                        }
                    };
                    stats.read_bytes += pages.bytes().len();
                    if from_log {
                        stats.cold += 1;
                    }
                    walk.pages.push((TurnIndex(i), pages));
                }
                walk
            })
            .collect();
        // Place: every page of every walk in one call, so the model places them
        // in one launch and waits for its copies once.
        let mut walks = walks;
        let mut entries: Vec<Vec<(TurnIndex, usize, PieceKey)>> = vec![Vec::new(); walks.len()];
        let mut items: Vec<(&[u8], PieceKey)> = Vec::new();
        let mut placing: Vec<usize> = Vec::new();
        for (w, walk) in walks.iter_mut().enumerate() {
            if !walk.whole {
                continue;
            }
            let mut walk_items = Vec::new();
            for (i, pages) in &walk.pages {
                let Ok(framed) = index_pages::decode(pages.bytes()) else {
                    walk.whole = false;
                    break;
                };
                for ((width, bytes), key) in framed.into_iter().zip(pages.page_keys()) {
                    entries[w].push((*i, width, *key));
                    walk_items.push((bytes, *key));
                }
            }
            if walk.whole && !walk_items.is_empty() {
                items.extend(walk_items);
                placing.push(w);
            }
        }
        let mut refused = false;
        if !items.is_empty() {
            let t_place = Instant::now();
            let placed = self.model.pool_positional_states(slot.0, &items);
            stats.place_us = t_place.elapsed().as_micros() as u64;
            if let Err(e) = placed {
                tracing::warn!(slot = slot.0, conversations = placing.len(), error = %e,
                    "pool: the refresh's pages could not be held; the conversations they \
                     belong to are left out of this pool");
                refused = true;
            }
        }
        drop(items);
        if refused {
            for &w in &placing {
                walks[w].whole = false;
            }
        }
        // Build: each whole walk's layout, its prior one extended by what it
        // placed.
        for (walk, entries) in walks.into_iter().zip(entries) {
            if !walk.whole {
                continue;
            }
            let mut layout = walk.prior_layout;
            keep.extend(layout.values().flatten().map(|(_, key)| *key));
            for (i, width, key) in entries {
                keep.push(key);
                layout.entry(i).or_default().push((width, key));
            }
            kept.insert(walk.tl, layout);
            kept_walked.insert(walk.tl, walk.turns);
        }
        stats.read_us = t_read.elapsed().as_micros() as u64;
        reading.end();
        stats.pooled = kept.len();
        stats.admitted = kept.keys().filter(|tl| !held_before.contains(tl)).count();
        stats.dropped = held_before
            .iter()
            .filter(|tl| !kept.contains_key(tl))
            .count();
        // What left the pool lingers with its pages (`pool_linger`); the model
        // keeps them beside the pool's own.
        for (tl, layout) in prior {
            let walked = prior_walked.get(&tl).copied().unwrap_or(0);
            linger.linger(tl, layout, walked);
        }
        stats.lingering = linger.len();
        keep.extend(linger.keys());
        // The knowledge base's resident sections pool beside the
        // conversations (`docs/placed_section_layout.md` §3.5).
        keep.extend(self.refresh_kb_residency(slot, conversation, target));
        if let Err(e) = self.model.retain_pooled(slot.0, &keep) {
            tracing::warn!(slot = slot.0, error = %e, "pool: dropped pieces not released");
        }
        self.log_pool_by_kind(conversation, target, &kept, &held_before, &placed, &stats);
        // The pool's pages go with the slot's projection state, which the
        // assembly declares the pooled rounds from, and leave with it. A slot
        // with no state yet has placed nothing to hold a pool beside.
        if let Some(state) = self.slot_projection_state.get_mut(&slot) {
            state.pool = kept;
            state.pool_walked = kept_walked;
            state.pool_linger = linger;
        }
        stats
    }

    /// One info line per kind of the pool (`docs/hit_driven_projection.md` §12
    /// item 26): its pieces and tokens, those admitted and dropped at this
    /// boundary, how many of them the indexer reads against how many the slot
    /// places, and its strongest pieces by rank, each with its page count —
    /// so a pool that favours large pieces shows it.
    fn log_pool_by_kind(
        &self,
        conversation: &Conversation,
        target: TimelineId,
        kept: &PoolPages,
        held_before: &HashSet<TimelineId>,
        placed: &[TimelineId],
        stats: &PoolRefresh,
    ) {
        let read = conversation.read();
        let Some(ws) = read.working_set(target) else {
            return;
        };
        let placed: HashSet<TimelineId> = placed.iter().copied().collect();
        let cache = index_page_cache::global()
            .map(|c| c.stats())
            .unwrap_or_default();
        tracing::info!(
            target: "candle_conversation::scheduler::pool",
            pooled = stats.pooled,
            admitted = stats.admitted,
            dropped = stats.dropped,
            index_read_turns = stats.walked,
            index_read_cold = stats.cold,
            index_read_bytes = stats.read_bytes,
            index_read_us = stats.read_us,
            fetch_us = stats.fetch_us,
            place_us = stats.place_us,
            revived = stats.revived,
            lingering = stats.lingering,
            cache_entries = cache.entries,
            cache_held_mib = cache.held_bytes >> 20,
            cache_hits = cache.hits,
            cache_misses = cache.misses,
            cache_uncached = cache.uncached,
            cache_evicted = cache.evicted,
            "pool:index_read"
        );
        for kind in [
            WorkingSetShare::Readings,
            WorkingSetShare::Folders,
            WorkingSetShare::Guides,
            WorkingSetShare::Journals,
        ] {
            let mut pieces: Vec<(TimelineId, usize, f32)> = ws
                .pooled()
                .into_iter()
                .filter(|(tl, _, share)| *share == kind && kept.contains_key(tl))
                .map(|(tl, tokens, _)| (tl, tokens, ws.score_of(tl).unwrap_or(0.0)))
                .collect();
            if pieces.is_empty() {
                continue;
            }
            pieces.sort_by(|a, b| b.2.total_cmp(&a.2));
            let tokens: usize = pieces.iter().map(|(_, t, _)| t).sum();
            let admitted = pieces
                .iter()
                .filter(|(tl, _, _)| !held_before.contains(tl))
                .count();
            let hit = pieces
                .iter()
                .filter(|(tl, _, _)| ws.hits_of(*tl) > 0.0)
                .count();
            let on_slot = pieces
                .iter()
                .filter(|(tl, _, _)| placed.contains(tl))
                .count();
            let named: Vec<String> = pieces
                .iter()
                .take(NAMED_PER_KIND)
                .enumerate()
                .map(|(rank, (tl, tokens, strength))| {
                    let name = read
                        .custom_of(*tl)
                        .and_then(|m| m.get("path").cloned())
                        .unwrap_or_else(|| tl.raw().to_string());
                    let pages: usize = kept.get(tl).map_or(0, |l| l.values().map(Vec::len).sum());
                    format!(
                        "#{} {name} pages={pages} tokens={tokens} strength={strength:.0}",
                        rank + 1
                    )
                })
                .collect();
            tracing::info!(
                target: "candle_conversation::scheduler::pool",
                kind = kind_name(kind),
                pieces = pieces.len(),
                tokens,
                admitted,
                hit,
                placed = on_slot,
                "pool by kind: {}",
                named.join(" · ")
            );
        }
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
