//! The state a slot carries from one projection to the next.
//!
//! The record of the pieces the last assembly placed, with where each one stood,
//! lets the next rebuild keep the longest prefix that still matches and re-walk
//! only from the first piece that differs. Captured glue islands are kept by their
//! content key for a fixed number of projections, not pruned to the current plan,
//! so projections that alternate between two shapes do not evict each other. Hit
//! momentum is folded in at each reprojection and lifts the next selection's
//! scores.

use super::*;

/// Per-slot state owned by the projection assembler.
///
/// `pending_user_part` holds the captured K/V **and index page** for the
/// in-flight turn's user message — populated when a `NewUserMessage` segment is
/// prefilled, cleared at seal time; survives mid-decode reprojection (which
/// truncates the slot) because it lives here on `SlotState`, not on the slot.
#[derive(Default)]
pub(in super::super) struct SlotState {
    pub(in super::super) pending_user_part: Option<CapturedSpan>,
    /// The sealed sections + turns this slot's current projection attends over.
    /// Refreshed on every `apply_projection`; consumed by the relief-eviction
    /// path as an explicit protect-list so it never drops the hot copy of a turn
    /// an in-flight prefill/decode is attending. Lives on `SlotState` so it
    /// inherits the slot's lifecycle (removal + fork re-keying) for free.
    pub(in super::super) working_set: SlotWorkingSet,
    /// Captured glue-island K/V from the previous gap-fill wave, keyed by the
    /// island's content-context hash (every preceding piece's identity + length,
    /// the island's own tokens, and its forward-bridge target — see
    /// `apply_segments_build`). An island whose key matches is Arc-injected like
    /// a sealed section instead of being recomputed by the wave: glue K is
    /// stored un-rotated and re-RoPE'd at read from its chunk `rope_base`, so a
    /// cached island stays exact under any uniform position shift, and the key
    /// guarantees its (backward-unbounded) attended context is content-identical.
    /// Retention is generation-based, not prune-to-current: belief-driven turn
    /// selection can alternate between two prefix shapes across consecutive
    /// reprojections, and pruning to the current plan's islands would make the
    /// alternating shapes evict each other forever. Entries carry the last
    /// projection generation that used them (hit or capture) and survive
    /// [`GLUE_ISLAND_RETAIN_GENERATIONS`] projections. Same lifecycle/capture
    /// pattern as `pending_user_part`.
    pub(in super::super) glue_islands: HashMap<u64, (u64, CapturedSpan)>,
    /// Monotone projection counter for the island cache's generation stamps.
    pub(in super::super) glue_generation: u64,
    /// How many of the slot's tokens its last projection placed — the sealed
    /// prefix, glue and user message, not the tail the slot forwarded itself.
    /// [`apply_segments_finish`] compares a rebuild against its slot only when
    /// the slot held more than this: a placement is chosen afresh by every
    /// projection, and only the rest exists nowhere but the slot.
    pub(in super::super) placed: usize,
    /// The pieces this slot's last complete assembly placed, in order — each
    /// one's chained identity and where the slot stood after it. A rebuild
    /// keeps the longest prefix whose identities match and re-injects only from
    /// the first piece that differs (`super::piece_identity`). Emptied by an
    /// incomplete assembly, whose slot does not hold what its pieces describe.
    pub(in super::super) placed_pieces: Vec<PlacedPiece>,
    /// The rounds the last assembly declared to the model as hit pieces, in
    /// declaration order — what each count of `take_hits` belongs to
    /// (`docs/hit_driven_projection.md` §6.4).
    pub(in super::super) hit_keys: Vec<RoundKey>,
    /// Hit momentum per placed round, folded at every reprojection — what lifts
    /// the scores the next projection selects by.
    pub(in super::super) round_hits: HitMomentum<RoundKey>,
    /// What an assembly read back before it declared new pieces, waiting for
    /// the next reprojection to fold it in with its own read — so the counts
    /// a turn's tail ran up are not lost to the next turn's opening.
    pub(in super::super) unread_hits: Vec<HitPart>,
    /// What the slot's pool holds ([`super::Scheduler::refresh_pool`]): the
    /// pages of its pooled conversations, whose unplaced rounds the
    /// declaration names for the shadow scan.
    pub(in super::super) pool: PoolPages,
    /// How many of each pooled conversation's turns [`Self::pool`] has walked.
    /// A conversation's turns are append-only, so a refresh that keeps it walks
    /// only the turns past this — the earlier ones' pages are held already.
    pub(in super::super) pool_walked: HashMap<TimelineId, u32>,
    /// Pieces that left the pool, keeping their pages for a few refreshes
    /// (`super::pool_linger`).
    pub(in super::super) pool_linger: PoolLinger,
}

/// The most read-back parts a slot keeps between reprojections; past it the
/// oldest goes.
pub(super) const UNREAD_HIT_PARTS: usize = 8;

/// Every placed round as a hit piece (`docs/hit_driven_projection.md` §6.4,
/// §8.1): its key and the positions it took, in slot order. A round is one
/// whole turn, so each placed turn is grouped into the round of the exchange it
/// belongs to. `placed[j]` records `pieces[j]`; pieces that are not turns and
/// turns with no conversation to name them by are not declared.
pub(in super::super) fn hit_rounds_of(
    pieces: &[AssembledPiece],
    placed: &[PlacedPiece],
    conv: &Substrate,
) -> (Vec<RoundKey>, Vec<Range<usize>>) {
    let mut turns_of: HashMap<TimelineId, Vec<(TurnIndex, Range<usize>)>> = HashMap::new();
    for (piece, p) in pieces.iter().zip(placed) {
        if p.tokens == 0 {
            continue;
        }
        if let AssembledPiece::Turn {
            index,
            timeline: Some(timeline),
            ..
        } = piece
        {
            let span = (p.pos_after - p.tokens) as usize..p.pos_after as usize;
            turns_of.entry(*timeline).or_default().push((*index, span));
        }
    }
    let mut declared: Vec<(RoundKey, Range<usize>)> = Vec::new();
    for (timeline, placed) in turns_of {
        for turns in conv.exchanges_of(timeline) {
            let Some(&head) = turns.first() else {
                continue;
            };
            if !placed.iter().any(|(t, _)| turns.contains(t)) {
                continue;
            }
            for (index, span) in rounds::round_spans(&turns, &placed) {
                declared.push((
                    RoundKey {
                        timeline,
                        head,
                        index: index as u32,
                    },
                    span,
                ));
            }
        }
    }
    declared.sort_by_key(|(_, span)| span.start);
    declared.into_iter().unzip()
}

impl SlotState {
    /// The conversations whose turns the last assembly placed, each once, in
    /// slot order — the pool's floor (`docs/hit_driven_projection.md` §7.2).
    pub(in super::super) fn placed_timelines(&self) -> Vec<TimelineId> {
        let mut out: Vec<TimelineId> = Vec::new();
        for p in &self.placed_pieces {
            if let Some(PieceRange::Turn(tl, _)) = p.range {
                if !out.contains(&tl) {
                    out.push(tl);
                }
            }
        }
        out
    }

    /// Every turn the last assembly placed, in slot order.
    pub(in super::super) fn placed_turns(&self) -> Vec<TurnKey> {
        self.placed_pieces
            .iter()
            .filter_map(|p| match p.range {
                Some(PieceRange::Turn(tl, idx)) => Some(TurnKey::new(tl, idx)),
                _ => None,
            })
            .collect()
    }

    /// The placed pieces by position, each named by the turn or section it is.
    pub(in super::super) fn placed_spans(&self) -> Vec<PieceSpan> {
        self.placed_pieces
            .iter()
            .map(|p| PieceSpan {
                label: match p.range {
                    Some(PieceRange::Turn(timeline, index)) => {
                        format!("turn:{}:{}", timeline.raw(), index.0)
                    }
                    Some(PieceRange::Section(id)) => format!("section:{}", id.raw()),
                    None => "glue".to_string(),
                },
                start: (p.pos_after - p.tokens) as usize,
                end: p.pos_after as usize,
            })
            .collect()
    }
}

/// One piece an assembly placed: what it is, what it occupies, and the slot as
/// it stood right after it.
#[derive(Clone)]
pub(in super::super) struct PlacedPiece {
    /// The piece's alignment identity (`piece_identity::alignment_identities`):
    /// a sealed piece's own, a forwarded piece's chained.
    pub(super) identity: u64,
    pub(super) kind: PieceKind,
    /// Blocks and positions the piece occupies.
    pub(super) blocks: usize,
    pub(super) tokens: u32,
    /// The slot's block count after the piece — where a cut after it lands.
    pub(super) blocks_after: usize,
    /// The slot's logical position after the piece.
    pub(super) pos_after: u32,
    /// The piece's contribution to the glue-island cache's running prefix hash.
    pub(super) contrib: u64,
    /// The running prefix hash after the piece, so a walk resumed here keys
    /// its islands exactly as a whole walk would.
    pub(super) prefix_h: DefaultHasher,
    /// The turn or section the piece is, for writing back where it sits when a
    /// splice moves it; `None` for glue and turn halves, which record none.
    pub(super) range: Option<PieceRange>,
    /// The index page the piece was placed with, held while the slot holds the
    /// piece: a rebuild that re-walks the turn finds the page again without
    /// reading the log (`HeldIndexPages`), and a cut or a freed slot lets it go.
    pub(super) page: Option<Arc<StoredPages>>,
}

/// Which substrate range a placed piece's position is written to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PieceRange {
    Turn(TimelineId, TurnIndex),
    Section(SectionId),
}

#[cfg(test)]
impl PlacedPiece {
    /// Where the record says the slot stood after the piece: its block and
    /// position ends, and the running prefix hash there.
    pub(in super::super) fn ends(&self) -> (usize, u32, u64) {
        (self.blocks_after, self.pos_after, self.prefix_h.finish())
    }

    /// A record of one piece the slot holds through `blocks_after` — for tests
    /// that need a kept prefix on record without walking an assembly.
    pub(in super::super) fn ending_at(blocks_after: usize) -> Self {
        Self {
            identity: 1,
            kind: PieceKind::Sealed,
            blocks: blocks_after,
            tokens: (blocks_after * 32) as u32,
            blocks_after,
            pos_after: (blocks_after * 32) as u32,
            contrib: 0,
            prefix_h: DefaultHasher::new(),
            range: None,
            page: None,
        }
    }
}

/// Lay `records` end to end from the slot's start: each one's cumulative
/// block and position ends, and the running prefix hash folded over the
/// contributions before it. A splice moves every piece after it, and this is
/// where the records learn where they now stand.
pub(super) fn restack(records: &mut [PlacedPiece]) {
    let (mut blocks, mut pos) = (0usize, 0u32);
    let mut prefix_h = DefaultHasher::new();
    for r in records {
        blocks += r.blocks;
        pos += r.tokens;
        prefix_h.write_u64(r.contrib);
        r.blocks_after = blocks;
        r.pos_after = pos;
        r.prefix_h = prefix_h.clone();
    }
}

/// Where the system prompt ends in a slot `placed` describes: the position
/// after the leading run of sections and the glue between them, `0` when the
/// projection opens on anything else.
///
/// `placed[j]` records `pieces[j]` for every piece before the deferred user
/// message, which the leading run never reaches.
pub(super) fn prompt_end_of(pieces: &[AssembledPiece], placed: &[PlacedPiece]) -> u32 {
    let lead = pieces
        .iter()
        .take_while(|p| matches!(p, AssembledPiece::Section(_) | AssembledPiece::Glue(_)))
        .count();
    match lead {
        0 => 0,
        n => placed[n - 1].pos_after,
    }
}

/// The position ranges of the pinned sections in a slot `placed` describes:
/// each section segment `segments` marks pinned, at the place its piece took.
///
/// Section pieces appear in `pieces` in the order their segments appear in
/// `segments`, so the two walk together; `placed[j]` records `pieces[j]` for
/// every piece before the deferred user message, which no section follows.
/// Adjacent ranges merge.
pub(super) fn pin_ranges_of(
    segments: &[ProjectionSegment],
    pieces: &[AssembledPiece],
    placed: &[PlacedPiece],
) -> Vec<Range<usize>> {
    let marks: Vec<Pin> = segments
        .iter()
        .filter_map(|s| match s {
            ProjectionSegment::Sealed(SealedKind::Section(r)) => Some(r.pin),
            _ => None,
        })
        .collect();
    let section_pieces = pieces
        .iter()
        .filter(|p| matches!(p, AssembledPiece::Section(_)))
        .count();
    // The walk pairs the two lists in order; were they ever to disagree, the
    // pins would land on other sections, so none is declared instead.
    if marks.len() != section_pieces {
        tracing::warn!(
            segments = marks.len(),
            pieces = section_pieces,
            "section segments and pieces disagree: no ranges pinned this projection"
        );
        return Vec::new();
    }
    let mut pins = marks.into_iter();
    let mut out: Vec<Range<usize>> = Vec::new();
    for (j, piece) in pieces.iter().enumerate() {
        if !matches!(piece, AssembledPiece::Section(_)) {
            continue;
        }
        let Some(pin) = pins.next() else { break };
        let Some(p) = placed.get(j) else { break };
        if pin == Pin::No || p.tokens == 0 {
            continue;
        }
        let range = (p.pos_after - p.tokens) as usize..p.pos_after as usize;
        match out.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ => out.push(range),
        }
    }
    out
}

impl SlotState {
    /// What a fork of this slot starts its projections from: the record of the
    /// pieces placed and the glue cache, so the fork's rebuild keeps the shared
    /// prefix. The in-flight user part belongs to the parent's own turn and stays
    /// with it.
    pub(in super::super) fn fork(&self) -> Self {
        Self {
            pending_user_part: None,
            working_set: self.working_set.clone(),
            glue_islands: self.glue_islands.clone(),
            glue_generation: self.glue_generation,
            placed: self.placed,
            placed_pieces: self.placed_pieces.clone(),
            // A fork's model sequence declares its own pieces at its first
            // projection; the parent's counts stay the parent's. What the
            // parent's indexer read of the shared history it carries on.
            hit_keys: Vec::new(),
            round_hits: self.round_hits.clone(),
            unread_hits: Vec::new(),
            // The model holds a pool's pages per sequence: the fork's holds
            // none until its own first reprojection chooses one, and walks
            // each conversation it pools whole.
            pool: PoolPages::new(),
            pool_walked: HashMap::new(),
            pool_linger: PoolLinger::default(),
        }
    }

    /// Rewrite every sealed chunk this slot's caches hold through a compaction
    /// sweep, returning the captured spans that moved.
    ///
    /// **These caches are holders, and the sweep did not know about them.** Both
    /// `pending_user_part` and each cached glue island keep an
    /// `Arc<Vec<SealedSequence>>` of K/V that a later projection Arc-injects into a
    /// slot instead of recomputing it. A compaction relocates those chunks; unless
    /// the cache is rewritten it goes on naming the slots the pass vacated, and the
    /// pool hands that ground to the next claim. The injected span then decodes
    /// against another sequence's K/V — finite, plausibly shaped and wrong, which
    /// is why it surfaces as a NaN in the first attention layer rather than as a
    /// fault, and why it took a completeness check to find rather than a crash.
    ///
    /// Shares the pass's one `Sweep`, like every other holder: these caches
    /// routinely hold the very same `Arc<Vec<ChunkGid>>` allocation a residence
    /// holds, and a second sweep would hand each its own equal-but-distinct
    /// replacement, leaving refcounts that disagree with the sharing the cache
    /// believes exists.
    pub(in super::super) fn rewrite_for_compaction(
        &mut self,
        sweep: &mut candle_nn::kv_cache::Sweep<'_>,
    ) -> candle::Result<usize> {
        let mut moved = 0usize;
        if let Some(span) = self.pending_user_part.as_mut() {
            moved += span.rewrite_for_compaction(sweep)?;
        }
        for (_, span) in self.glue_islands.values_mut() {
            moved += span.rewrite_for_compaction(sweep)?;
        }
        Ok(moved)
    }
}

impl CapturedSpan {
    /// Install the sweep's replacement for this span's K/V, if any of it moved.
    ///
    /// The index page beside it is position-free and names no chunk, so a
    /// relocation does not touch it.
    pub(super) fn rewrite_for_compaction(
        &mut self,
        sweep: &mut candle_nn::kv_cache::Sweep<'_>,
    ) -> candle::Result<usize> {
        match candle_nn::kv_cache::rewrite_sealed(&self.kv, sweep)? {
            Some(next) => {
                self.kv = Arc::new(next);
                Ok(1)
            }
            None => Ok(0),
        }
    }
}

/// How many projections an unused cached glue island survives before it is
/// dropped. Covers selection shapes that alternate every few reprojections
/// while keeping the pinned-chunk footprint bounded (~shapes x islands).
pub(super) const GLUE_ISLAND_RETAIN_GENERATIONS: u64 = 4;

impl SlotState {
    /// Drop the in-flight `NewUserMessage` capture after a successful seal.
    pub(in super::super) fn trim_post_turn(&mut self) {
        self.pending_user_part = None;
    }

    /// Advance the island-cache generation for a capture pass and prune every
    /// entry not seen (hit-refreshed or captured) within the last
    /// [`GLUE_ISLAND_RETAIN_GENERATIONS`] passes. Returns the generation the
    /// pass stamps on its inserts. Separated from the capture loop so the
    /// retention contract is unit-testable without a session.
    pub(in super::super) fn begin_island_capture(&mut self) -> u64 {
        self.glue_generation += 1;
        let horizon = self
            .glue_generation
            .saturating_sub(GLUE_ISLAND_RETAIN_GENERATIONS);
        self.glue_islands.retain(|_, (g, _)| *g >= horizon);
        self.glue_generation
    }
}

/// The sealed working set (sections + turns) a slot's projection attends over.
/// See [`SlotState::working_set`] and [`working_set_from_segments`].
#[derive(Debug, Default, Clone)]
pub(in super::super) struct SlotWorkingSet {
    pub(in super::super) sections: Vec<SectionId>,
    pub(in super::super) turns: Vec<TurnKey>,
}

/// Extract the sealed working set from a projection's segments. Only `Sealed`
/// entries reference persisted KV; `Generated`/`NewUserMessage` segments carry
/// live-prefilled tokens with no substrate residence, so they never contribute.
pub(in super::super) fn working_set_from_segments(
    segments: &[ProjectionSegment],
) -> SlotWorkingSet {
    let mut ws = SlotWorkingSet::default();
    for seg in segments {
        if let ProjectionSegment::Sealed(kind) = seg {
            match kind {
                SealedKind::Section(rs) => ws.sections.push(rs.id),
                SealedKind::Turn(rt, _) => {
                    if let Some(key) = rt.key() {
                        ws.turns.push(key);
                    }
                }
            }
        }
    }
    ws
}
