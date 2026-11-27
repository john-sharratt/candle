//! Lays a projection's pieces out in logical order, the one decision both the
//! apply walk and the debug view read.
//!
//! A sealed turn carries its own boundary markers in its grid, so it emits alone
//! and consecutive turns abut with no gap to fill. Only a turn sealed before the
//! markers moved into the grid still needs the framing the assembler emits. The
//! dialect supplies the boundary strings, and the thinking-suppression mechanism
//! is chosen once per family, so the two halves cannot drift apart.

use super::*;

/// Pre-tokenised dialect role markers the assembler wraps around
/// every `Sealed::Turn` segment to produce attention-correct turn
/// boundaries at projection time.
///
/// The two markers are the inter-turn ones — `user_start` opens a
/// turn against whatever causal prefix actually precedes it on the
/// slot, and `assistant_end` closes a turn before the next region
/// begins.  The intra-turn `user_end` and `assistant_start` markers
/// stay baked in the persisted turn bytes because their hidden
/// state is dominated by the turn's own (invariant) content.
///
/// The scheduler owns a single instance, pre-tokenised by the
/// engine from the active model's dialect at engine construction.
/// Passed to the assembler as a borrow on `ApplyContext`.
#[derive(Debug, Clone, Default)]
pub(crate) struct BoundaryMarkers {
    pub(crate) user_start: Arc<Vec<u32>>,
    pub(crate) assistant_end: Arc<Vec<u32>>,
    /// Intra-turn markers — `user_end` closes the user message and
    /// `assistant_start` opens the assistant reply. Not used by the
    /// assembler (they stay baked into persisted turn bytes), but the
    /// summary probe needs them to frame its synthetic user→assistant
    /// exchange so the model *responds* rather than continuing the prompt.
    pub(crate) user_end: Arc<Vec<u32>>,
    pub(crate) assistant_start: Arc<Vec<u32>>,
    /// The dialect's `/no_think` soft-switch (empty for non-thinking dialects).
    /// Emitted as live glue right after `user_start` on a suppressed turn so the
    /// switch sits in the user turn (where Qwen3 honours it) without being baked
    /// into any sealed turn.
    pub(crate) no_think: Arc<Vec<u32>>,
    /// The other half of [`Dialect::thinking_suppression`]: the already-closed
    /// reasoning block a family WITHOUT a soft switch (Qwen3.5 / Qwen3.8) opens
    /// its assistant turn with, exactly as its template renders
    /// `enable_thinking=false`. Empty whenever [`Self::no_think`] is non-empty —
    /// the dialect picks one mechanism, never both.
    ///
    /// Kept beside `no_think` because a caller that framed a synthetic exchange
    /// with only the soft switch silently suppressed nothing at all on that
    /// family, which is how the tree compressor came to reason unbounded inside
    /// a summary probe.
    pub(crate) no_think_block: Arc<Vec<u32>>,
    /// The role-marker strings, kept beside their tokenised forms so the seal
    /// path can locate role boundaries baked into a turn's assistant text (the
    /// code_read tool exchange `<tool_call>…<tool_response>…confirmation`) by a
    /// plain string match, and slice the sub-segment display text at them.
    pub(crate) user_start_str: String,
    pub(crate) assistant_end_str: String,
    pub(crate) user_end_str: String,
    pub(crate) assistant_start_str: String,
    /// When set, inter-turn glue attends **backward-unbounded, strictly causal**
    /// — no forward bridge into the turn it introduces (`fwd_ahead ≡ 0`, §E).
    /// The `paged-latent` glue kernel is causal-only; the forward-window path
    /// is a Qwen-only feature. Keyed on the dialect at construction.
    pub(crate) causal_only_glue: bool,
}

impl BoundaryMarkers {
    /// Pre-tokenise the dialect's role-marker strings via the
    /// caller-supplied closure.  The closure form keeps this module
    /// tokenizer-agnostic — callers (the engine) wrap their
    /// `tokenizers::Tokenizer::encode`.
    pub(crate) fn from_dialect<E, F>(
        dialect: &candle_transformers::models::dialect::Dialect,
        mut tokenize: F,
    ) -> Result<Self, E>
    where
        F: FnMut(&str) -> Result<Vec<u32>, E>,
    {
        let user_start = Arc::new(tokenize(dialect.user_start)?);
        let assistant_end = Arc::new(tokenize(dialect.assistant_end)?);
        let user_end = Arc::new(tokenize(dialect.user_end)?);
        let assistant_start = Arc::new(tokenize(dialect.assistant_start)?);
        // Both halves come from `thinking_suppression`, which owns the "exactly
        // one mechanism per family" rule; deriving them separately here is what
        // would let the two drift apart.
        let (switch, block) = dialect.thinking_suppression(true);
        let no_think = Arc::new(tokenize(switch)?);
        let no_think_block = Arc::new(tokenize(block)?);
        Ok(Self {
            user_start,
            assistant_end,
            user_end,
            assistant_start,
            no_think,
            no_think_block,
            user_start_str: dialect.user_start.to_string(),
            assistant_end_str: dialect.assistant_end.to_string(),
            user_end_str: dialect.user_end.to_string(),
            assistant_start_str: dialect.assistant_start.to_string(),
            causal_only_glue: matches!(
                dialect.dialect_type(),
                candle_transformers::models::dialect::DialectType::DeepSeek
            ),
        })
    }
}

/// One step of the assembled materialized prefix, in logical order.
///
/// This is the SINGLE source of truth for how a projection's segments + dialect
/// glue lay out. Both the apply path (which injects K/V per piece) and the
/// substrate debug view (which renders them) iterate the same
/// [`assemble_pieces`] output, so they can never silently diverge.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssembledPiece {
    /// A run of "glue" tokens — `Generated` template runs merged with the
    /// inter-turn boundary markers (`user_start` before a turn, `assistant_end`
    /// after it) — that the engine gap-fills as one island.
    Glue(Vec<u32>),
    /// A sealed system-prompt section; its K/V comes from the substrate.
    Section(SectionId),
    /// A sealed past turn; its K/V comes from the substrate.  `timeline` is the
    /// turn's conversation, stamped at projection (see [`ResolvedTurn::key`]) and
    /// carried here so the inject path never re-derives it from `group`.  `group`
    /// is retained only for diagnostics.
    Turn {
        group: GroupId,
        index: TurnIndex,
        role: crate::Role,
        timeline: Option<TimelineId>,
    },
    /// The in-flight user message, deferred to prefill after the gap-fill.
    DeferredUser(Arc<Vec<u32>>),
    /// Positions left empty on the K/V and the index alike: what follows is
    /// read this many positions further on.
    Hole(usize),
}

/// Lay out a projection's segments into the ordered [`AssembledPiece`]s the
/// engine materializes — the glue *decision*, factored out of the K/V injection
/// so it can be reused (the substrate debug view) and unit-tested in isolation.
///
/// Exactly mirrors what the apply walk produced inline: consecutive `Generated`
/// runs accumulate into one glue island; a `Sealed::Turn` is wrapped with
/// `user_start` (flushed into the glue island immediately before it) and
/// `assistant_end` (carried into the island after it, so it merges with the next
/// turn's `user_start`); `NewUserMessage` is deferred past the gap-fill.
/// [`assemble_pieces`] with the per-turn boundary question answered by the
/// caller — `bakes(timeline, index)` reports whether that turn's grid carries
/// its own `user_start` / `assistant_end`.
///
/// Turns sealed since the boundaries moved into the grid bake them; turns
/// already in a workspace's redo log do not, and their ethereal markers are
/// dropped by `TurnLayout::realize`. So the assembler emits the framing for
/// exactly the turns that lack it — which is what keeps a resumed
/// pre-existing conversation from projecting as one unbroken run with no role
/// markers anywhere.
pub(crate) fn assemble_pieces_with(
    segments: &[ProjectionSegment],
    markers: &BoundaryMarkers,
    bakes: &dyn Fn(TimelineId, TurnIndex) -> bool,
) -> Vec<AssembledPiece> {
    assemble_inner(segments, Some((markers, bakes)))
}

/// [`assemble_pieces_with`] for callers with no substrate to ask (tests, and
/// the panel's structural rendering): every turn is assumed to bake, which is
/// true of everything sealed by this build.
pub(crate) fn assemble_pieces(segments: &[ProjectionSegment]) -> Vec<AssembledPiece> {
    assemble_inner(segments, None)
}

pub(super) type BoundaryQuery<'a> = (
    &'a BoundaryMarkers,
    &'a dyn Fn(TimelineId, TurnIndex) -> bool,
);

pub(super) fn assemble_inner(
    segments: &[ProjectionSegment],
    boundaries: Option<BoundaryQuery<'_>>,
) -> Vec<AssembledPiece> {
    let mut pieces: Vec<AssembledPiece> = Vec::new();
    let mut run: Vec<u32> = Vec::new();
    fn flush(run: &mut Vec<u32>, pieces: &mut Vec<AssembledPiece>) {
        if !run.is_empty() {
            pieces.push(AssembledPiece::Glue(std::mem::take(run)));
        }
    }
    let mut i = 0;
    while i < segments.len() {
        if matches!(&segments[i], ProjectionSegment::Generated { .. }) {
            while let Some(ProjectionSegment::Generated { tokens, .. }) = segments.get(i) {
                run.extend(tokens.iter().copied());
                i += 1;
            }
            flush(&mut run, &mut pieces);
            continue;
        }
        match &segments[i] {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::Section(rs.id));
            }
            ProjectionSegment::Sealed(SealedKind::Turn(rt, role)) => {
                // **A sealed turn emits alone.**
                //
                // Its opener (`user_start`, plus the `/no_think` switch it was
                // sealed with) and its closer (`assistant_end`) are baked into
                // its own grid at submit — see `Sequence::submit_prefill_unit`
                // — so re-emitting them here would double every boundary. What
                // used to be an `assistant_end ++ user_start` island between
                // consecutive turns is now the tail of one turn's grid abutting
                // the head of the next, which is the same token stream with no
                // gap chunk to fill.
                //
                // That is the whole point: an island is a run the engine must
                // GAP-FILL, and a recurrent model cannot compute K/V for tokens
                // inserted mid-sequence. With the boundaries owned, a
                // projection of system prompt + sealed turns + a live user
                // message reserves no gap chunks at all.
                //
                // Except for a turn that does NOT own them: one sealed before
                // the boundaries moved into the grid, still sitting in a
                // workspace's redo log with ethereal markers that
                // `TurnLayout::realize` drops. For those the assembler is still
                // the only source, so it emits the framing exactly as it used
                // to — a per-turn answer, from the turn's own layout.
                let bakes = boundaries
                    .as_ref()
                    .and_then(|(_, q)| rt.timeline.map(|tl| q(tl, rt.index())))
                    .unwrap_or(true);
                if !bakes {
                    if let Some((markers, _)) = boundaries.as_ref() {
                        run.extend(markers.user_start.iter().copied());
                    }
                }
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::Turn {
                    group: rt.group(),
                    index: rt.index(),
                    role: *role,
                    timeline: rt.timeline,
                });
                if !bakes {
                    if let Some((markers, _)) = boundaries.as_ref() {
                        run.extend(markers.assistant_end.iter().copied());
                    }
                }
            }
            ProjectionSegment::NewUserMessage { tokens } => {
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::DeferredUser(tokens.clone()));
            }
            ProjectionSegment::Hole { positions } => {
                flush(&mut run, &mut pieces);
                if *positions > 0 {
                    pieces.push(AssembledPiece::Hole(*positions));
                }
            }
            ProjectionSegment::Generated { .. } => unreachable!("handled in the run loop above"),
        }
        i += 1;
    }
    flush(&mut run, &mut pieces);
    pieces
}

/// Lay out the **conversation region** of a projection into the ordered
/// [`MaterializedPiece`]s the projection panel renders — real boundary-glue
/// islands interleaved with the sealed turns, built from the SAME
/// [`assemble_pieces`] decision the engine injects from, so the panel's glue can
/// never drift from what the projection really does.
///
/// The system prompt is excluded (it is covered by
/// [`ProjectionSelection::system`]); the live, still-decoding user message is
/// excluded too (the panel renders the live turn from its own `um`/`dm`). Glue islands are decoded to text via `decode`; each
/// turn is classified (layer / group / role / token count / forest kind / why)
/// via `resolver` + `origins`.
pub(crate) fn materialize_conversation(
    segments: &[ProjectionSegment],
    origins: &HashMap<TurnKey, SelectionOrigin>,
    resolver: &dyn ContentResolver,
    schema: &Schema,
    mut decode: impl FnMut(&[u32]) -> String,
) -> Vec<MaterializedPiece> {
    // Conversation region = from the first sealed turn / deferred user onward;
    // everything before is the system prompt.
    let start = segments
        .iter()
        .position(|s| {
            matches!(
                s,
                ProjectionSegment::Sealed(SealedKind::Turn(..))
                    | ProjectionSegment::NewUserMessage { .. }
            )
        })
        .unwrap_or(segments.len());
    assemble_pieces(&segments[start..])
        .into_iter()
        .filter_map(|piece| match piece {
            AssembledPiece::Glue(tokens) => {
                let text = decode(&tokens);
                (!text.is_empty()).then_some(MaterializedPiece::Glue { text })
            }
            AssembledPiece::Turn {
                group,
                index,
                role,
                timeline,
            } => {
                let key = timeline.map(|tl| TurnKey::new(tl, index));
                Some(MaterializedPiece::Turn {
                    turn: SelectedTurn {
                        layer: layer_name_of_group(schema, group).unwrap_or("").to_string(),
                        group: group_name_of(schema, group)
                            .unwrap_or("conversation")
                            .to_string(),
                        index: index.0,
                        role: role_str(role).to_string(),
                        // A turn's identity is (timeline, index); the group alone is
                        // ambiguous once a group holds many conversations.
                        tokens: key.map_or(0, |k| resolver.turn_token_count(k)) as u32,
                        kind: key.map_or(TurnKind::Normal, |k| resolver.turn_kind(k)),
                        // A face carries its unit's reason: the unit is what was
                        // selected, as the selection record names it.
                        reason: key
                            .and_then(|k| origins.get(&resolver.unit_of(k).unwrap_or(k)))
                            .copied(),
                        timeline: timeline.map(|tl| tl.raw()),
                        // Materialized display spine — the turn is shown, and
                        // neither the belief score nor its qualification is
                        // threaded here (the carry reads
                        // `ProjectionSelection::turns`, not the materialized pieces).
                        selected: true,
                        score: 0.0,
                        qualified: false,
                    },
                })
            }
            // The live user message is a separate prefill unit (not part of
            // `project()`'s segments), so it never reaches this spine — the panel
            // renders the in-flight turn from `um`/`dm`. A stray section is not
            // part of the displayed dialogue spine, nor is a hole.
            AssembledPiece::DeferredUser(_)
            | AssembledPiece::Section(_)
            | AssembledPiece::Hole(_) => None,
        })
        .collect()
}
