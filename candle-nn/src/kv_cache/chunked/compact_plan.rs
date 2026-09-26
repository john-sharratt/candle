//! Which chunk goes where in a **perfect** KV compaction — pure arithmetic over
//! an arena census, with no device in it.
//!
//! # What "perfect" means here
//!
//! One size class's arenas, laid end to end in physical address order, are one
//! flat sequence of equal-sized slots. A pass walks that sequence from the left,
//! and every time it finds a gap it fills it with the **last** live chunk on the
//! right. When the two cursors meet, the live chunks occupy a gapless prefix and
//! every slot above them is free — so the class is packed to the lowest physical
//! address it can reach, and `release_empty_arenas` can hand back every arena
//! that has fallen out of the prefix.
//!
//! That is the property the partition actually needs. `live_watermark()` — the
//! floor the wave transient tier must stand above, and so what bounds how far
//! left `weight_floor` can sit — is the *highest live region*, not the live
//! count. Packing to a gapless prefix is exactly what minimises it.
//!
//! # Why a gap-filling walk and not a sparsity heuristic
//!
//! The obvious cheaper policy is to pick the arenas that are nearly empty and
//! drain those, because an arena holding 8 live chunks returns a whole arena for
//! 8 copies where a full one costs 2,048. It is cheaper per arena and it is the
//! wrong objective: sparse arenas are as likely to be *low* in the span as high,
//! so draining by sparsity moves chunks upward as often as downward and can
//! leave the watermark exactly where it was. Occupancy is not the quantity under
//! attack; address is.
//!
//! # Minimality
//!
//! The two-cursor walk moves each relocated chunk **once**, and it relocates a
//! chunk only if that chunk sits above the final packed frontier. Both are
//! forced: any pass that reaches a gapless prefix must vacate every occupied slot
//! above the frontier, and no such chunk needs a second move because its
//! destination is free before it is written. So the move count this produces is
//! the minimum for a perfect pack — which is worth stating because the first
//! attempt at address-ordered relocation on a prior branch moved 387,759 bands
//! against a live population that needed roughly 884 moved, by ranking
//! destinations on free space instead of address.
//!
//! # Bounded, because it runs between forwards
//!
//! `max_moves` clips a pass. A clipped pass is not a broken one: everything below
//! the left cursor is already gapless and nothing has been moved upward, so the
//! pool is strictly better packed than before and the next pass resumes from a
//! shorter distance. Run every wave, the steady state is a handful of moves;
//! `max_moves` is what stops the *first* pass on a badly fragmented pool from
//! being a stall.
//!
//! # One plan per POOL, not per class
//!
//! Everything here is scoped to an [`ArenaKey`] — a `(size class, location)` pair
//! — and never to a class alone. GPU and CPU arenas share one `arena_idx`
//! namespace, so a census filtered only by stride holds warm host arenas beside
//! hot device ones and a walk over it would plan a copy between two different
//! memories. [`plan_pool`] applies the filter itself so that no caller can ask
//! for such a move.

use super::arena::ArenaKey;

/// One arena's slot occupancy.
///
/// `rank` is physical address order, not `arena_idx`: a gid's arena index is
/// whatever the pool handed out and is recycled, whereas what compaction is
/// trying to minimise is an *address*. The two are routinely different, and
/// ordering by the wrong one silently packs into the wrong end of the span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaSlots {
    /// Index this arena's gids encode — `raw = arena_idx * GID_STRIDE + chunk_idx`.
    pub arena_idx: usize,
    /// The pool this arena belongs to: its size class **and its location**.
    ///
    /// Both halves are load-bearing, and the location is the one easy to forget.
    /// GPU and CPU arenas share ONE `arena_idx` namespace — `ArenaKey` selects
    /// which pool the refcount table lives in, not which index space the arena
    /// sits in — so a census filtered by class alone contains warm CPU arenas
    /// beside hot GPU ones at the same stride. A move planned across that
    /// boundary resolves two addresses in different memories: it either faults or
    /// silently writes host bytes no attention kernel will ever read.
    /// [`plan_pool`] filters on the whole key so such a move cannot be produced.
    pub key: ArenaKey,
    /// Position in physical address order, ascending. Lower is a lower address.
    pub rank: usize,
    /// Slots this arena holds, occupied or not.
    pub capacity: usize,
    /// Occupied slot indices, ascending, each `< capacity`.
    pub occupied: Vec<u32>,
}

/// One chunk relocation: copy the slot at `from` to the slot at `to`.
///
/// Both are `(arena_idx, chunk_idx)` in the gid namespace, so a caller turns
/// either end into a device address the same way a gid does, and into a raw gid
/// by `arena_idx * GID_STRIDE + chunk_idx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkMove {
    pub from: (usize, u32),
    pub to: (usize, u32),
}

/// The moves that pack one pool, in the order they must be applied.
///
/// Deliberately not `Default`: a plan is meaningless without the pool it belongs
/// to, and a default one would name an arbitrary `(class, location)` pair that
/// some caller would eventually copy chunks through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactPlan {
    /// Moves, left-cursor order. Applying a prefix of this list is always safe:
    /// each destination is free before its move and no source is ever a
    /// destination (see [`plan_pool`]).
    pub moves: Vec<ChunkMove>,
    /// The pool this plan covers. A chunk only fits a slot of its own stride and
    /// can only be copied within one memory, so a plan spans neither.
    pub key: ArenaKey,
    /// `true` when `max_moves` stopped the walk before the cursors met — the pool
    /// is better packed than it was but not yet gapless, and another pass has
    /// work to do.
    pub clipped: bool,
}

impl CompactPlan {
    /// Nothing to do: the pool is already a gapless prefix.
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

/// Plan one pool — one `(size class, location)` pair — or `None` when it is
/// already perfectly packed.
///
/// `arenas` may hold every arena there is; this **filters on the whole key**
/// rather than trusting the caller to have done it. That is deliberate: a filter
/// by size class alone leaves warm CPU arenas in the census beside hot GPU ones
/// at the same stride, and the walk would then plan a copy between two different
/// memories. Filtering here means no caller can produce that move, and
/// `a_mixed_census_never_plans_across_pools` holds it.
///
/// `max_moves` clips the pass; zero means unbounded.
pub fn plan_pool(arenas: &[ArenaSlots], key: ArenaKey, max_moves: usize) -> Option<CompactPlan> {
    // Address order, and this pool only. `arena_idx` is not address order — see
    // `ArenaSlots::rank`.
    let mut order: Vec<&ArenaSlots> = arenas.iter().filter(|a| a.key == key).collect();
    if order.is_empty() {
        return None;
    }
    order.sort_unstable_by_key(|a| a.rank);

    // A flat occupancy view over the class's whole slot sequence. `occupied` is
    // ascending per arena, so a membership test is a binary search and the walk
    // below stays O(slots log capacity) rather than materialising a bitmap over
    // every slot of every arena — which at the 320 B class's 52,428 chunks per
    // arena would be the largest allocation in the pass.
    let occupied_at = |arena: &ArenaSlots, slot: u32| arena.occupied.binary_search(&slot).is_ok();

    // Global slot ordinals, so "left is still below right" is one comparison.
    // Prefix sums over capacity, in rank order.
    let mut base = Vec::with_capacity(order.len());
    let mut total = 0usize;
    for a in &order {
        base.push(total);
        total += a.capacity;
    }

    let mut moves: Vec<ChunkMove> = Vec::new();
    let mut clipped = false;

    // Left cursor: the lowest free slot. Right cursor: the highest occupied one.
    let mut lp = 0usize; // arena position, left
    let mut ls = 0u32; // slot within it
    let mut rp = order.len() - 1;
    // Exclusive, and pre-decremented on first use. `u32` to match the left
    // cursor and the slot indices themselves: the widest class holds 52,428
    // chunks per arena, so a slot index never needs more.
    let mut rs: u32 = order[rp].capacity as u32;

    loop {
        // Advance the left cursor to the next FREE slot.
        let left_ord = loop {
            if lp >= order.len() {
                break usize::MAX;
            }
            if ls as usize >= order[lp].capacity {
                lp += 1;
                ls = 0;
                continue;
            }
            if occupied_at(order[lp], ls) {
                ls += 1;
                continue;
            }
            break base[lp] + ls as usize;
        };
        if left_ord == usize::MAX {
            break; // No gaps at all: already packed.
        }

        // Retreat the right cursor to the next OCCUPIED slot.
        let right_ord = loop {
            if rs == 0 {
                if rp == 0 {
                    break usize::MAX;
                }
                rp -= 1;
                rs = order[rp].capacity as u32;
                continue;
            }
            rs -= 1;
            if occupied_at(order[rp], rs) {
                break base[rp] + rs as usize;
            }
        };
        if right_ord == usize::MAX {
            break; // Nothing live left to pull down.
        }

        // Cursors met or crossed: every live chunk is already below every gap.
        if right_ord <= left_ord {
            break;
        }

        if max_moves != 0 && moves.len() >= max_moves {
            clipped = true;
            break;
        }

        moves.push(ChunkMove {
            from: (order[rp].arena_idx, rs),
            to: (order[lp].arena_idx, ls),
        });
        // The destination is now occupied and the source now free; step both past
        // the slots just settled so neither cursor reconsiders them.
        ls += 1;
    }

    if moves.is_empty() {
        return None;
    }
    Some(CompactPlan {
        moves,
        key,
        clipped,
    })
}

/// The whole of what fragmentation denies the weight side, in regions.
///
/// **Two losses, and they are not the same loss.** Both push `weight_floor` right
/// and so cost expert residency, but nothing recovers them by the same means:
///
/// * **Arena fragmentation** — `watermark - live_arenas`. Free regions stranded
///   *below* the arena frontier. They are claimable, so they are not wasted in the
///   occupancy sense; what they cost is the frontier's *position*, because the wave
///   transient tier must stand above the highest live arena and the boundary is
///   measured from there. Only moving an arena down recovers this, and nothing
///   does today. Self-correcting under sustained allocation (the free list is
///   lowest-index-first, so the next claim takes the lowest hole) and persistent
///   whenever allocation pauses — which is every decode-only phase.
/// * **KV fragmentation** — `live_arenas - packed_arenas`. Arenas holding a
///   handful of live chunks. An arena keeps its whole 16 MiB region until its
///   *last* chunk goes, and nothing moves chunks between arenas, so this never
///   self-corrects at all.
///
/// `watermark - packed_arenas` is the sum, and it is the figure a perfect
/// compaction closes: pack the chunks (removing the second), which empties the high
/// arenas, which lowers the frontier (removing the first).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroundLost {
    /// One past the highest live region — the frontier the tier stands above.
    pub watermark: usize,
    /// Regions held by an arena, across every pool.
    pub live_arenas: usize,
    /// Arenas those pools' live chunks would occupy packed, across every pool.
    pub packed_arenas: usize,
}

impl GroundLost {
    /// Free regions stranded below the frontier — arena fragmentation.
    pub fn arena_holes(&self) -> usize {
        self.watermark.saturating_sub(self.live_arenas)
    }

    /// Regions a chunk pack would release — KV fragmentation.
    pub fn sparsity(&self) -> usize {
        self.live_arenas.saturating_sub(self.packed_arenas)
    }

    /// Everything a perfect compaction would hand back to the weight side.
    pub fn total(&self) -> usize {
        self.watermark.saturating_sub(self.packed_arenas)
    }

    /// **VRAM efficiency: of the ground denied to the weight side, the percentage
    /// actually holding KV.** 100 is perfectly packed at the frontier.
    ///
    /// The denominator is the **frontier**, not the live arena count, because the
    /// frontier is what the weight side actually loses — the tier stands above the
    /// highest live arena and `weight_floor` is measured from there, so a region
    /// below the frontier costs the weight side whether it is live, sparse or free.
    /// Dividing by `live_arenas` instead would score a pool with a straggler
    /// holding a high arena as perfectly efficient, which is the exact state that
    /// kills decode.
    ///
    /// 100 when nothing is live: no frontier, nothing denied.
    pub fn efficiency_pct(&self) -> usize {
        if self.watermark == 0 {
            return 100;
        }
        self.packed_arenas * 100 / self.watermark
    }
}

/// What one pool's fragmentation costs, without planning anything.
///
/// The figure a compaction is judged by, and it is **arena sparsity, not region
/// holes**. A region freed below the arena frontier is reclaimed by the next claim
/// — the region free list is lowest-index-first, so a hole is exactly what the
/// allocator consumes next, and measured over a churning pool holes peak in the
/// tens and settle to one. What does *not* self-correct is an arena holding a
/// handful of live chunks: it owns its whole 16 MiB region until its **last** chunk
/// goes, and nothing moves chunks between arenas. So the damage is the gap between
/// the arenas a pool holds and the arenas its live chunks would need if they were
/// packed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fragmentation {
    /// Arenas the pool currently holds.
    pub arenas: usize,
    /// Live chunks across them.
    pub live_chunks: usize,
    /// Arenas those chunks would occupy packed — the floor.
    pub packed_arenas: usize,
}

impl Fragmentation {
    /// Arenas a perfect pack would empty, and so regions it would hand back.
    pub fn freeable_arenas(&self) -> usize {
        self.arenas.saturating_sub(self.packed_arenas)
    }

    /// Occupancy as a percentage of the ground held, 100 meaning perfectly packed.
    pub fn occupancy_pct(&self) -> usize {
        if self.arenas == 0 {
            return 100;
        }
        self.packed_arenas * 100 / self.arenas
    }
}

/// Measure one pool's fragmentation from a census.
pub fn fragmentation(arenas: &[ArenaSlots], key: ArenaKey) -> Fragmentation {
    let mine: Vec<&ArenaSlots> = arenas.iter().filter(|a| a.key == key).collect();
    let live_chunks: usize = mine.iter().map(|a| a.occupied.len()).sum();
    // Capacity is uniform within a pool, but take the max rather than assuming:
    // an arena registered under a different geometry would otherwise divide by a
    // capacity no arena has.
    let capacity = mine.iter().map(|a| a.capacity).max().unwrap_or(0);
    let packed_arenas = if capacity == 0 {
        0
    } else {
        live_chunks.div_ceil(capacity)
    };
    Fragmentation {
        arenas: mine.len(),
        live_chunks,
        packed_arenas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::chunked::size_class::SizeClass;
    use crate::kv_cache::ArenaLocation;

    /// The pool under test — one rung of the ladder, on the device.
    fn gpu() -> ArenaKey {
        ArenaKey::new(SizeClass::at(5), ArenaLocation::Gpu)
    }

    /// The same rung in host memory. Same `arena_idx` namespace, different
    /// memory: the pair a plan must never bridge.
    fn cpu() -> ArenaKey {
        ArenaKey::new(SizeClass::at(5), ArenaLocation::Cpu)
    }

    fn arena(arena_idx: usize, rank: usize, capacity: usize, occupied: &[u32]) -> ArenaSlots {
        keyed(gpu(), arena_idx, rank, capacity, occupied)
    }

    fn keyed(
        key: ArenaKey,
        arena_idx: usize,
        rank: usize,
        capacity: usize,
        occupied: &[u32],
    ) -> ArenaSlots {
        ArenaSlots {
            arena_idx,
            key,
            rank,
            capacity,
            occupied: occupied.to_vec(),
        }
    }

    /// The defining property: after applying every move, the class's live chunks
    /// occupy a gapless prefix of the rank-ordered slot sequence.
    ///
    /// Asserted by simulation rather than by eyeballing the move list, because the
    /// property is about the end state and a plan that merely *looks* ordered can
    /// still leave a hole.
    fn assert_packed(arenas: &[ArenaSlots], plan: &CompactPlan) {
        let mut order: Vec<ArenaSlots> = arenas.to_vec();
        order.sort_unstable_by_key(|a| a.rank);
        let live_total: usize = order.iter().map(|a| a.occupied.len()).sum();

        for m in &plan.moves {
            let src = order
                .iter_mut()
                .find(|a| a.arena_idx == m.from.0)
                .expect("source arena");
            let at = src
                .occupied
                .binary_search(&m.from.1)
                .expect("source slot must be occupied when its move is applied");
            src.occupied.remove(at);

            let dst = order
                .iter_mut()
                .find(|a| a.arena_idx == m.to.0)
                .expect("destination arena");
            let ins = dst
                .occupied
                .binary_search(&m.to.1)
                .expect_err("destination slot must be FREE when its move is applied");
            dst.occupied.insert(ins, m.to.1);
        }

        assert_eq!(
            order.iter().map(|a| a.occupied.len()).sum::<usize>(),
            live_total,
            "compaction must neither create nor destroy a chunk",
        );

        // Walk the flat sequence: every occupied slot must precede every free one.
        let mut seen_gap = false;
        for a in &order {
            for s in 0..a.capacity as u32 {
                let occ = a.occupied.binary_search(&s).is_ok();
                if occ && seen_gap && !plan.clipped {
                    panic!(
                        "arena {} (rank {}) slot {s} is live above a gap — not packed",
                        a.arena_idx, a.rank,
                    );
                }
                if !occ {
                    seen_gap = true;
                }
            }
        }
    }

    /// Two half-full arenas become one full arena and one empty one.
    #[test]
    fn two_half_full_arenas_pack_into_one() {
        let arenas = vec![arena(0, 0, 4, &[0, 2]), arena(1, 1, 4, &[1, 3])];
        let plan = plan_pool(&arenas, gpu(), 0).expect("there are gaps below live chunks");
        assert_packed(&arenas, &plan);
        // Four live chunks, four slots in the low arena: the high arena empties.
        assert_eq!(plan.moves.len(), 2);
        assert!(plan.moves.iter().all(|m| m.to.0 == 0));
        assert!(plan.moves.iter().all(|m| m.from.0 == 1));
        assert!(!plan.clipped);
    }

    /// **Address order, not arena index.** The arena with the LOWER rank receives,
    /// even when its `arena_idx` is the higher of the two — the case that silently
    /// packs into the wrong end of the span if the sort key is wrong.
    #[test]
    fn the_receiving_arena_is_the_one_lowest_in_the_span() {
        let arenas = vec![
            // arena_idx 90 sits LOW (rank 0); arena_idx 3 sits HIGH (rank 1).
            arena(90, 0, 4, &[0, 1]),
            arena(3, 1, 4, &[0, 1]),
        ];
        let plan = plan_pool(&arenas, gpu(), 0).expect("gaps below live chunks");
        assert_packed(&arenas, &plan);
        assert!(
            plan.moves.iter().all(|m| m.to.0 == 90 && m.from.0 == 3),
            "chunks must move toward the low ADDRESS (rank 0 = arena_idx 90): {:?}",
            plan.moves,
        );
    }

    /// An already-packed class is refused outright — a pass that moves nothing
    /// must not be reported as work.
    #[test]
    fn a_gapless_prefix_is_already_perfect() {
        let arenas = vec![arena(0, 0, 4, &[0, 1, 2, 3]), arena(1, 1, 4, &[0, 1])];
        assert_eq!(plan_pool(&arenas, gpu(), 0), None);
    }

    /// An empty class, and a class with no live chunks at all, are both no-ops.
    #[test]
    fn nothing_live_is_nothing_to_do() {
        assert_eq!(plan_pool(&[], gpu(), 0), None);
        let arenas = vec![arena(0, 0, 4, &[]), arena(1, 1, 4, &[])];
        assert_eq!(plan_pool(&arenas, gpu(), 0), None);
    }

    /// A single arena still compacts internally: a gap at slot 0 under a live
    /// chunk at slot 3 is exactly the case that pins a watermark inside one arena.
    #[test]
    fn one_arena_still_fills_its_own_gaps() {
        let arenas = vec![arena(7, 0, 4, &[1, 3])];
        let plan = plan_pool(&arenas, gpu(), 0).expect("slot 0 is a gap below slot 3");
        assert_packed(&arenas, &plan);
        assert_eq!(
            plan.moves,
            vec![ChunkMove {
                from: (7, 3),
                to: (7, 0)
            }],
        );
    }

    /// Every relocated chunk moves exactly once, and only chunks above the final
    /// frontier move at all. This is the minimality claim in the module docs.
    #[test]
    fn each_moved_chunk_moves_once_and_only_from_above_the_frontier() {
        let arenas = vec![
            arena(0, 0, 8, &[0, 5]),
            arena(1, 1, 8, &[2, 7]),
            arena(2, 2, 8, &[1]),
        ];
        let plan = plan_pool(&arenas, gpu(), 0).expect("plenty of gaps");
        assert_packed(&arenas, &plan);

        // Five live chunks ⇒ the frontier is global ordinal 5, inside arena 0.
        // Only the chunks above it may move, and each at most once.
        let mut sources: Vec<(usize, u32)> = plan.moves.iter().map(|m| m.from).collect();
        let before = sources.len();
        sources.sort_unstable();
        sources.dedup();
        assert_eq!(before, sources.len(), "a chunk was moved twice");

        // No destination is also a source: applying a prefix of the plan must be
        // safe, which requires the two sets to be disjoint.
        let dests: Vec<(usize, u32)> = plan.moves.iter().map(|m| m.to).collect();
        for d in &dests {
            assert!(
                !sources.contains(d),
                "slot {d:?} is both a source and a destination",
            );
        }
    }

    /// A clipped pass leaves the pool strictly better packed and says so, so the
    /// caller knows to come back rather than believing the class is done.
    #[test]
    fn a_clipped_pass_is_partial_and_admits_it() {
        let arenas = vec![arena(0, 0, 8, &[7]), arena(1, 1, 8, &[0, 1, 2, 3])];
        let plan = plan_pool(&arenas, gpu(), 2).expect("gaps below live chunks");
        assert!(plan.clipped, "the cap stopped the walk");
        assert_eq!(plan.moves.len(), 2, "clipped to the cap");
        // Still sound: applying it moves chunks only downward.
        assert_packed(&arenas, &plan);
    }

    /// The cap is honoured exactly, and a cap wider than the work is not clipping.
    #[test]
    fn a_cap_wider_than_the_work_does_not_clip() {
        let arenas = vec![arena(0, 0, 4, &[3]), arena(1, 1, 4, &[0])];
        let plan = plan_pool(&arenas, gpu(), 100).expect("gaps");
        assert!(!plan.clipped);
        assert_packed(&arenas, &plan);
    }

    /// Arenas of differing capacity (the ladder's classes differ, and a partially
    /// registered arena can differ too) must still produce one flat sequence.
    #[test]
    fn arenas_of_different_capacity_form_one_sequence() {
        let arenas = vec![
            arena(0, 0, 2, &[1]),
            arena(1, 1, 5, &[0, 4]),
            arena(2, 2, 3, &[2]),
        ];
        let plan = plan_pool(&arenas, gpu(), 0).expect("gaps below live chunks");
        assert_packed(&arenas, &plan);
    }

    /// A full low arena is never disturbed: its chunks are already where they
    /// belong, and moving them would be the pure cost the sparsity policy paid.
    #[test]
    fn a_full_low_arena_is_never_a_source() {
        let arenas = vec![
            arena(0, 0, 4, &[0, 1, 2, 3]),
            arena(1, 1, 4, &[3]),
            arena(2, 2, 4, &[0]),
        ];
        let plan = plan_pool(&arenas, gpu(), 0).expect("gaps in the upper arenas");
        assert_packed(&arenas, &plan);
        assert!(
            plan.moves.iter().all(|m| m.from.0 != 0),
            "the full low arena must not be touched: {:?}",
            plan.moves,
        );
    }

    /// **A mixed census never plans across pools.**
    ///
    /// GPU and CPU arenas share one `arena_idx` namespace, so a census filtered by
    /// stride alone contains both. Here the host arenas are the ones lowest in
    /// rank — the most attractive destinations a rank-ordered walk could pick —
    /// and the plan must still not touch them: a copy from device to host memory
    /// either faults or silently writes bytes no attention kernel reads.
    #[test]
    fn a_mixed_census_never_plans_across_pools() {
        let arenas = vec![
            keyed(cpu(), 100, 0, 4, &[]),
            keyed(cpu(), 101, 1, 4, &[0]),
            keyed(gpu(), 5, 2, 4, &[3]),
            keyed(gpu(), 6, 3, 4, &[0]),
        ];
        let plan = plan_pool(&arenas, gpu(), 0).expect("the GPU pool has gaps below live chunks");
        assert_eq!(plan.key, gpu());
        for m in &plan.moves {
            assert!(
                (m.from.0 == 5 || m.from.0 == 6) && (m.to.0 == 5 || m.to.0 == 6),
                "a move left the GPU pool: {m:?}",
            );
        }
        // And the host pool plans independently, on its own ranks.
        let host = plan_pool(&arenas, cpu(), 0);
        assert!(
            host.is_some_and(|p| p.key == cpu() && p.moves.iter().all(|m| m.to.0 == 100)),
            "the host pool packs into its own lowest arena",
        );
    }

    /// **Sparsity is the damage, and this is the number.** Four arenas holding
    /// one chunk each would pack into one, so three regions are recoverable —
    /// and no amount of region-level reuse recovers them, because each arena
    /// keeps its region until its last chunk goes.
    #[test]
    fn fragmentation_counts_the_arenas_a_pack_would_empty() {
        let arenas = vec![
            arena(0, 0, 4, &[0]),
            arena(1, 1, 4, &[2]),
            arena(2, 2, 4, &[1]),
            arena(3, 3, 4, &[3]),
        ];
        let f = fragmentation(&arenas, gpu());
        assert_eq!(f.arenas, 4);
        assert_eq!(f.live_chunks, 4);
        assert_eq!(f.packed_arenas, 1, "four chunks fit one 4-slot arena");
        assert_eq!(f.freeable_arenas(), 3);
        assert_eq!(f.occupancy_pct(), 25);
    }

    /// A packed pool reports nothing recoverable — the figure must not cry wolf
    /// on a healthy pool, or the telemetry it feeds is noise.
    #[test]
    fn a_packed_pool_is_not_fragmented() {
        let arenas = vec![arena(0, 0, 4, &[0, 1, 2, 3]), arena(1, 1, 4, &[0, 1])];
        let f = fragmentation(&arenas, gpu());
        assert_eq!(f.live_chunks, 6);
        assert_eq!(f.packed_arenas, 2, "6 chunks need 2 arenas of 4");
        assert_eq!(f.freeable_arenas(), 0);
        assert_eq!(f.occupancy_pct(), 100);
    }

    /// An empty pool is not fragmented and does not divide by zero.
    #[test]
    fn an_empty_pool_reports_nothing() {
        let f = fragmentation(&[], gpu());
        assert_eq!(f.freeable_arenas(), 0);
        assert_eq!(f.occupancy_pct(), 100);
        let empties = vec![arena(0, 0, 4, &[]), arena(1, 1, 4, &[])];
        let f = fragmentation(&empties, gpu());
        assert_eq!(f.live_chunks, 0);
        assert_eq!(f.freeable_arenas(), 2, "both are reclaimable as they stand");
    }

    /// Measured per pool, so a host pool's sparsity is never reported as a
    /// device pool's.
    #[test]
    fn fragmentation_is_measured_per_pool() {
        let arenas = vec![
            arena(0, 0, 4, &[0]),
            arena(1, 1, 4, &[1]),
            keyed(cpu(), 50, 0, 4, &[0]),
            keyed(cpu(), 51, 1, 4, &[1]),
            keyed(cpu(), 52, 2, 4, &[2]),
        ];
        assert_eq!(fragmentation(&arenas, gpu()).arenas, 2);
        assert_eq!(fragmentation(&arenas, cpu()).arenas, 3);
    }

    /// **The highest live arena is the marker, and it is what kills weights.**
    ///
    /// The tier stands above the frontier and `weight_floor` is measured from
    /// there, so the weight side's ground is set by *where the topmost live arena
    /// sits* — not by how many arenas are live and not by how full they are. This
    /// pins the decomposition: 900 live arenas under a frontier of 1,000, whose
    /// chunks would pack into 300, denies 700 regions — 100 because the frontier is
    /// held high by stragglers, 600 because the arenas are sparse.
    #[test]
    fn the_frontier_is_the_marker_and_the_loss_decomposes() {
        let g = GroundLost {
            watermark: 1000,
            live_arenas: 900,
            packed_arenas: 300,
        };
        assert_eq!(g.arena_holes(), 100, "free regions below the frontier");
        assert_eq!(g.sparsity(), 600, "regions a chunk pack would release");
        assert_eq!(
            g.total(),
            700,
            "and the total is the frontier minus the floor"
        );
        assert_eq!(
            g.arena_holes() + g.sparsity(),
            g.total(),
            "the two losses must sum to the whole, or one is being double counted",
        );
    }

    /// **Efficiency is measured against the FRONTIER, not the live count.**
    ///
    /// A straggler holding one high arena over a mostly-free span is the state that
    /// kills decode, and dividing by `live_arenas` would score it as perfect. Here
    /// 300 packed arenas under a frontier of 1,000 is 30% however few arenas are
    /// live.
    #[test]
    fn efficiency_is_measured_against_the_frontier() {
        let straggler = GroundLost {
            watermark: 1000,
            live_arenas: 310,
            packed_arenas: 300,
        };
        assert_eq!(
            straggler.efficiency_pct(),
            30,
            "a high straggler over empty ground is 30% efficient, not 96%",
        );
        // The same packed floor with the frontier where it belongs is perfect.
        let packed = GroundLost {
            watermark: 300,
            live_arenas: 300,
            packed_arenas: 300,
        };
        assert_eq!(packed.efficiency_pct(), 100);
    }

    /// Nothing live is not a failure: no frontier means nothing denied.
    #[test]
    fn an_idle_pool_is_fully_efficient() {
        assert_eq!(GroundLost::default().efficiency_pct(), 100);
    }

    /// A perfectly packed pool at a frontier equal to its live count denies
    /// nothing — the healthy state, and the figure must read zero for it.
    #[test]
    fn a_packed_pool_at_its_own_frontier_denies_nothing() {
        let g = GroundLost {
            watermark: 300,
            live_arenas: 300,
            packed_arenas: 300,
        };
        assert_eq!(g.arena_holes(), 0);
        assert_eq!(g.sparsity(), 0);
        assert_eq!(g.total(), 0);
    }

    /// **Sparsity alone can be the whole loss, with no holes at all.** This is the
    /// state every measured run reached: allocation refills holes on the next
    /// claim, so `watermark == live`, while the arenas stay half empty. A figure
    /// that only counted holes would report a healthy pool here.
    #[test]
    fn a_contiguous_but_sparse_pool_still_denies_ground() {
        let g = GroundLost {
            watermark: 1915,
            live_arenas: 1915,
            packed_arenas: 1800,
        };
        assert_eq!(g.arena_holes(), 0, "no holes — the allocator refilled them");
        assert_eq!(g.sparsity(), 115);
        assert_eq!(g.total(), 115, "and the loss is real regardless");
    }

    /// A pool absent from the census is not an error and not a plan.
    #[test]
    fn a_pool_with_no_arenas_is_no_plan() {
        let arenas = vec![keyed(cpu(), 1, 0, 4, &[1])];
        assert_eq!(plan_pool(&arenas, gpu(), 0), None);
    }

    /// Exhaustive over small configurations: whatever the occupancy, the plan
    /// packs it and never moves a chunk upward.
    #[test]
    fn every_small_configuration_packs() {
        const CAP: u32 = 3;
        for mask in 0u32..(1 << (CAP * 2)) {
            let a0: Vec<u32> = (0..CAP).filter(|s| mask & (1 << s) != 0).collect();
            let a1: Vec<u32> = (0..CAP).filter(|s| mask & (1 << (s + CAP)) != 0).collect();
            let arenas = vec![
                arena(0, 0, CAP as usize, &a0),
                arena(1, 1, CAP as usize, &a1),
            ];
            let Some(plan) = plan_pool(&arenas, gpu(), 0) else {
                // Refused ⇒ must already be a gapless prefix.
                let flat: Vec<bool> = (0..CAP)
                    .map(|s| a0.contains(&s))
                    .chain((0..CAP).map(|s| a1.contains(&s)))
                    .collect();
                let live = flat.iter().filter(|b| **b).count();
                assert!(
                    flat[..live].iter().all(|b| *b),
                    "mask {mask:#b} was refused but is not packed: {flat:?}",
                );
                continue;
            };
            assert_packed(&arenas, &plan);
            for m in &plan.moves {
                let from_ord = m.from.0 * CAP as usize + m.from.1 as usize;
                let to_ord = m.to.0 * CAP as usize + m.to.1 as usize;
                assert!(
                    to_ord < from_ord,
                    "mask {mask:#b}: chunk moved upward {:?} -> {:?}",
                    m.from,
                    m.to,
                );
            }
        }
    }
}
