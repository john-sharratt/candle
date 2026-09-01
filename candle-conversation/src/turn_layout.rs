//! Segment-vector turn layout.
//!
//! A turn (and, in a later stage, the system prompt) is described as an ordered
//! [`Vec<TurnSegment>`].  The vector is the **complete description of the turn's
//! K/V**: laying down each real segment's tokens at its offset reproduces the
//! slot's token grid exactly — there are no separate content-boundary offsets to
//! drift out of sync, because position is just the running sum of segment
//! lengths.  [`TurnLayout::validate_tiling`] enforces this, and the unit tests
//! assert it against hand-built grids.
//!
//! Each segment is **real** (occupies K/V → carries a [`KvSpan`]) or **ethereal**
//! (recorded / shown but not materialized → no span).  This one axis expresses
//! everything we used to special-case:
//!
//! - ethereal [`TurnSegment::Thinking`] — reasoning prose kept, its K/V dropped;
//! - ethereal [`TurnSegment::Glue`] — a boundary we represent but don't emit K/V
//!   for (e.g. a leading turn boundary that isn't materialized yet);
//! - real glue / real thinking / user / assistant — K/V-bearing, each a span.
//!
//! K/V materialization falls out of the segment kind: a real [`TurnSegment::Glue`]
//! is recomputed per projection from its dialect marker (re-projected), while
//! real `User` / `Thinking` / `Assistant` inject stored K/V.  The `offset` inside
//! a [`KvSpan`] is redundant with the running length sum (and asserted equal to
//! it), but kept so a *sub-range* of a turn can be projected directly — e.g.
//! inject only the answer, or window to the user half.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::normalization::Phase;

/// A segment's footprint in the turn's K/V grid: `len` token positions starting
/// at absolute `offset`.  `offset` is recoverable from the running sum of prior
/// segment lengths — [`TurnLayout::validate_tiling`] asserts the two agree — and
/// is stored so a turn sub-range can be windowed/projected without re-walking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSpan {
    pub offset: u32,
    pub len: u32,
}

impl KvSpan {
    pub fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }
    /// Exclusive end of the span (`offset + len`).
    pub fn end(&self) -> u32 {
        self.offset + self.len
    }
    /// The span as a `[start, end)` slice range, for windowing a token grid.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset as usize..self.end() as usize
    }
}

/// A dialect boundary marker.  The marker text/tokens are derived from the active
/// dialect by this kind — the layout stores only the kind (and, when real, the
/// [`KvSpan`] it occupies), never the marker bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlueKind {
    /// `<|im_start|>system\n`
    SystemStart,
    /// `<|im_start|>user\n` — opens a turn.
    UserStart,
    /// `<|im_start|>assistant\n`
    AssistantStart,
    /// `<|im_end|>\n` — closes a message / a turn.
    ImEnd,
    /// The `/no_think` soft-switch, emitted right after `user_start` on a
    /// thinking-suppressed turn.
    NoThink,
}

/// One ordered part of a turn.  Real segments carry a [`KvSpan`]; ethereal ones
/// carry none and contribute nothing to the K/V grid (their `text` / `kind` is
/// recorded for the projection record and the GUI only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnSegment {
    /// A dialect boundary marker.  `kv: Some` ⇒ REAL (recomputed per projection
    /// from the marker); `None` ⇒ ETHEREAL (recorded, not materialized).
    Glue {
        marker: GlueKind,
        #[serde(default)]
        kv: Option<KvSpan>,
    },
    /// User message body — always real.
    User { text: String, kv: KvSpan },
    /// A `<think>…</think>` reasoning block.  `kv: Some` ⇒ REAL (K/V stored);
    /// `None` ⇒ ETHEREAL (prose kept, K/V dropped).
    Thinking {
        text: String,
        #[serde(default)]
        kv: Option<KvSpan>,
    },
    /// Assistant answer body — always real; `text` optional.
    Assistant {
        #[serde(default)]
        text: Option<String>,
        kv: KvSpan,
    },
}

impl TurnSegment {
    /// The K/V span this segment occupies, or `None` when it is ethereal.
    pub fn kv(&self) -> Option<KvSpan> {
        match self {
            TurnSegment::Glue { kv, .. } | TurnSegment::Thinking { kv, .. } => *kv,
            TurnSegment::User { kv, .. } | TurnSegment::Assistant { kv, .. } => Some(*kv),
        }
    }
    /// True when the segment is materialized into the slot's K/V.
    pub fn is_real(&self) -> bool {
        self.kv().is_some()
    }
}

/// The K/V range `phase` occupies within a turn's segment list.
///
/// Taken as a slice so a caller holding persisted `TurnDecl.segments` can ask
/// without cloning them into a [`TurnLayout`] first — the offline analysis path
/// (`substrate_inspect --probe-phase` / `--gallery-phase`) reads exactly these,
/// straight off the decls it walked out of the redo log.
///
/// A phase may be split across several segments (a turn can carry more than one
/// user segment — a tool response arrives as a further user turn), so the range
/// spans from the first to the last; a phase with no real K/V (an ethereal
/// `<think>` whose K/V was dropped) yields `None`.
pub fn phase_span_of(segments: &[TurnSegment], phase: Phase) -> Option<Range<usize>> {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for seg in segments {
        let matches = matches!(
            (phase, seg),
            (Phase::User, TurnSegment::User { .. })
                | (Phase::Thinking, TurnSegment::Thinking { .. })
                | (Phase::Response, TurnSegment::Assistant { .. })
        );
        if !matches {
            continue;
        }
        if let Some(kv) = seg.kv() {
            lo = lo.min(kv.offset as usize);
            hi = hi.max(kv.end() as usize);
        }
    }
    (hi > lo && lo != usize::MAX).then_some(lo..hi)
}

/// A turn as an ordered list of segments.  Turn-level metadata (role, block span,
/// layer/group, scores, ids) lives on the owning turn entry, not here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TurnLayout {
    pub segments: Vec<TurnSegment>,
}

/// A tiling inconsistency — the segment vector does not describe a contiguous,
/// gap-free K/V grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TilingError {
    /// A real segment's `offset` did not equal the running length sum before it.
    OffsetMismatch {
        segment: usize,
        expected: u32,
        found: u32,
    },
    /// The summed K/V length did not equal the grid length the caller expected.
    LengthMismatch { expected: u32, found: u32 },
}

impl TurnLayout {
    pub fn new(segments: Vec<TurnSegment>) -> Self {
        Self { segments }
    }

    /// Total K/V positions the turn materializes — the sum of every real
    /// segment's length (ethereal segments add zero).
    pub fn kv_len(&self) -> u32 {
        self.segments
            .iter()
            .filter_map(|s| s.kv())
            .map(|s| s.len)
            .sum()
    }

    /// Assert the segments tile a contiguous K/V grid: each real segment starts
    /// exactly where the previous one ended (so `offset` is consistent with the
    /// running length), and the total matches `grid_len`.  This is THE invariant
    /// that makes the vector a faithful, complete description of the turn's K/V.
    pub fn validate_tiling(&self, grid_len: u32) -> Result<(), TilingError> {
        let mut cursor = 0u32;
        for (i, seg) in self.segments.iter().enumerate() {
            if let Some(span) = seg.kv() {
                if span.offset != cursor {
                    return Err(TilingError::OffsetMismatch {
                        segment: i,
                        expected: cursor,
                        found: span.offset,
                    });
                }
                cursor += span.len;
            }
        }
        if cursor != grid_len {
            return Err(TilingError::LengthMismatch {
                expected: grid_len,
                found: cursor,
            });
        }
        Ok(())
    }

    /// The user message body text.
    pub fn user_text(&self) -> &str {
        self.segments
            .iter()
            .find_map(|s| match s {
                TurnSegment::User { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or("")
    }

    /// The FULL assistant reply — the reasoning block (if any) immediately
    /// followed by the answer. The layout stores these split across a `Thinking`
    /// segment and an answer-only `Assistant` segment (so each renders on its own
    /// in a per-segment view); this rejoins them into the verbatim message a
    /// consumer wants whole — the chat-history view and summarization, whose
    /// `<think>…</think>` handling expects the block inline. `None` if there is no
    /// assistant segment.
    pub fn assistant_text(&self) -> Option<String> {
        let mut out = String::new();
        let mut has_assistant = false;
        for s in &self.segments {
            match s {
                TurnSegment::Thinking { text, .. } => out.push_str(text),
                TurnSegment::Assistant { text, .. } => {
                    has_assistant = true;
                    if let Some(t) = text {
                        out.push_str(t);
                    }
                }
                _ => {}
            }
        }
        has_assistant.then_some(out)
    }

    /// The `<think>…</think>` reasoning prose, if a thinking segment is present.
    pub fn thinking_text(&self) -> Option<&str> {
        self.segments.iter().find_map(|s| match s {
            TurnSegment::Thinking { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// True iff a `/no_think` glue segment is present (thinking suppressed).
    pub fn no_think(&self) -> bool {
        self.segments.iter().any(|s| {
            matches!(
                s,
                TurnSegment::Glue {
                    marker: GlueKind::NoThink,
                    ..
                }
            )
        })
    }

    /// The user message body's K/V span (`KvSpan{0,0}` if there is no user
    /// segment).
    pub fn user_span(&self) -> KvSpan {
        self.segments
            .iter()
            .find_map(|s| match s {
                TurnSegment::User { kv, .. } => Some(*kv),
                _ => None,
            })
            .unwrap_or(KvSpan { offset: 0, len: 0 })
    }

    /// The assistant answer body's K/V span (`KvSpan{0,0}` if there is no
    /// assistant segment).
    pub fn assistant_span(&self) -> KvSpan {
        self.segments
            .iter()
            .find_map(|s| match s {
                TurnSegment::Assistant { kv, .. } => Some(*kv),
                _ => None,
            })
            .unwrap_or(KvSpan { offset: 0, len: 0 })
    }

    /// Offset of the first REAL assistant-side segment — the `Thinking` span if
    /// one is present and real, else the `Assistant` span. This is the old
    /// `asst_start` content boundary.
    pub fn assistant_content_start(&self) -> u32 {
        for s in &self.segments {
            match s {
                TurnSegment::Thinking { kv: Some(span), .. } => return span.offset,
                TurnSegment::Assistant { kv, .. } => return kv.offset,
                _ => {}
            }
        }
        self.assistant_span().offset
    }

    /// Token index where the user message body begins (old `user_start`).
    pub fn user_content_start(&self) -> u32 {
        self.user_span().offset
    }

    /// Token index where the user message body ends (old `user_end`).
    pub fn user_content_end(&self) -> u32 {
        self.user_span().end()
    }

    /// Whether this turn's grid **contains its own boundary markers** — a real
    /// (`kv: Some`) leading `UserStart`.
    ///
    /// The projection assembler asks before deciding whether to emit boundary
    /// glue around this turn, because the answer is per-turn and durable: turns
    /// sealed once the boundaries moved into the grid carry them, while turns
    /// already in a workspace's redo log were sealed with ethereal ones
    /// (`kv: None`, supplied live by the assembler of the day) that
    /// [`Self::realize`] skips. Answering it globally — assuming every turn
    /// bakes — injects historical turns with no role markers at all, so an
    /// entire recovered conversation reads as one unbroken run and the model
    /// misparses who said what, silently.
    pub fn bakes_own_boundaries(&self) -> bool {
        self.segments.iter().any(|s| {
            matches!(
                s,
                TurnSegment::Glue {
                    marker: GlueKind::UserStart,
                    kv: Some(_),
                }
            )
        })
    }

    /// Slice `grid` (the turn's full token id buffer) into the token run of each
    /// real segment, in order.  `grid.len()` must equal [`Self::kv_len`].  Used
    /// by the K/V (re)build path and by tests that assert the layout reproduces
    /// the stored grid exactly.
    pub fn realize<'a>(&'a self, grid: &'a [u32]) -> Vec<(&'a TurnSegment, &'a [u32])> {
        self.segments
            .iter()
            .filter_map(|seg| seg.kv().map(|span| (seg, &grid[span.range()])))
            .collect()
    }

    /// Build the layout for a turn stored in today's flat
    /// `[user_msg][user_end][assistant_start][response]` grid, given the marker
    /// token lengths (`im_end_len` = `user_end` length, `assistant_start_len`).
    /// Bridges the current representation into segments without changing the grid
    /// — the native seal path will emit segments directly in a later stage, and
    /// can additionally split the `<think>…</think>` block out of the assistant
    /// body.  `user_body` is `[user_content_start, user_content_end)`,
    /// `assistant_start` is the first assistant-content token, `total` the grid
    /// length.  A non-zero `user_content_start` is honored as a real leading
    /// `UserStart` boundary (not used today, but representable).
    ///
    /// Equivalent to [`Self::from_flat_grid_with_tail`] with no reserved tail.
    #[allow(clippy::too_many_arguments)]
    pub fn from_flat_grid(
        user_content_start: u32,
        user_content_end: u32,
        assistant_start: u32,
        total: u32,
        im_end_len: u32,
        assistant_start_len: u32,
        user_text: String,
        assistant_text: Option<String>,
        no_think: bool,
    ) -> Self {
        Self::from_flat_grid_with_tail(
            user_content_start,
            user_content_end,
            assistant_start,
            total,
            im_end_len,
            assistant_start_len,
            0,
            user_text,
            assistant_text,
            no_think,
        )
    }

    /// [`Self::from_flat_grid`] with `trailing_marker_len` tokens of the grid
    /// reserved for the turn's own closing `<|im_end|>`.
    ///
    /// **The trailing twin of the leading boundary.** `user_content_start > 0`
    /// already means "the grid reserves room for the opener, so bake it real";
    /// this is the same statement at the other end — the assistant body becomes
    /// `[assistant_start, total − trailing_marker_len)` and the closing `ImEnd`
    /// occupies the reserved tail instead of being recorded ethereally.
    ///
    /// Both ends exist because ownership is **both ends or neither**: every
    /// inter-turn island the assembler emits is exactly `assistant_end ++
    /// user_start`, so a turn that owned only its opener would drop every
    /// closer and one that owned only its closer would drop every opener. See
    /// `docs/deltanet_state_persistence.md` §4.7a.
    ///
    /// `0` reproduces the ethereal tail exactly, which is what every caller
    /// passes today — the mechanism is representable before it is exercised,
    /// deliberately, so the caller change that reserves the room is the only
    /// thing left to get wrong.
    #[allow(clippy::too_many_arguments)]
    pub fn from_flat_grid_with_tail(
        user_content_start: u32,
        user_content_end: u32,
        assistant_start: u32,
        total: u32,
        im_end_len: u32,
        assistant_start_len: u32,
        trailing_marker_len: u32,
        user_text: String,
        assistant_text: Option<String>,
        no_think: bool,
    ) -> Self {
        let mut segments = Vec::new();
        // The turn opens with `user_start`. If the grid reserves room for it, it
        // is a baked (real) boundary; otherwise it is materialized by the
        // projection spine and recorded ETHEREALLY here — the turn still "owns"
        // the marker (the glue shift), it just isn't in this turn's own grid.
        if user_content_start > 0 {
            segments.push(TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(KvSpan::new(0, user_content_start)),
            });
        } else {
            segments.push(TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: None,
            });
        }
        // The `/no_think` soft-switch is live glue (not in this turn's grid) — so
        // it is recorded ethereally on a suppressed turn.
        if no_think {
            segments.push(TurnSegment::Glue {
                marker: GlueKind::NoThink,
                kv: None,
            });
        }
        segments.push(TurnSegment::User {
            text: user_text,
            kv: KvSpan::new(
                user_content_start,
                user_content_end.saturating_sub(user_content_start),
            ),
        });
        // The intra-turn marker region is `[user_content_end, assistant_start)`.
        // Split it into `im_end` then `assistant_start`, but derive the second
        // length from the region (not the passed marker length) so the two glue
        // spans ALWAYS tile the region exactly even if a tokenizer merged across a
        // join and the nominal marker lengths don't sum to the region width.
        let region = assistant_start.saturating_sub(user_content_end);
        let im_end = im_end_len.min(region);
        let _ = assistant_start_len; // nominal; the real split is region-driven
        segments.push(TurnSegment::Glue {
            marker: GlueKind::ImEnd,
            kv: Some(KvSpan::new(user_content_end, im_end)),
        });
        segments.push(TurnSegment::Glue {
            marker: GlueKind::AssistantStart,
            kv: Some(KvSpan::new(user_content_end + im_end, region - im_end)),
        });
        // The tail is clamped to what the grid actually has past the assistant
        // start, so an over-large reservation cannot produce a negative body.
        let tail = trailing_marker_len.min(total.saturating_sub(assistant_start));
        let body_end = total - tail;
        segments.push(TurnSegment::Assistant {
            text: assistant_text,
            kv: KvSpan::new(assistant_start, body_end.saturating_sub(assistant_start)),
        });
        // The turn closes with `<|im_end|>` (the glue shift's suffix). If the
        // grid reserves room for it, it is a baked (real) boundary the turn
        // owns; otherwise it is materialized by the projection spine and
        // recorded ETHEREALLY here — the turn still "owns" the marker, it just
        // isn't in this turn's own grid.
        segments.push(TurnSegment::Glue {
            marker: GlueKind::ImEnd,
            kv: (tail > 0).then(|| KvSpan::new(body_end, tail)),
        });
        Self { segments }
    }

    /// Replace the trailing single `Assistant` segment with a pre-tiled run of
    /// sub-segments — used when the assistant half is itself a multi-part tool
    /// exchange (`Assistant(<tool_call>) → ImEnd → UserStart → User(<tool_response>)
    /// → ImEnd → AssistantStart → Assistant(confirmation)`) rather than one body.
    /// The caller supplies segments whose spans already tile the replaced
    /// `Assistant` span exactly (offsets absolute, contiguous); `validate_tiling`
    /// still guards the result. No-op if there is no assistant segment.
    pub fn with_assistant_split(mut self, subsegments: Vec<TurnSegment>) -> Self {
        if subsegments.is_empty() {
            return self;
        }
        let Some(pos) = self
            .segments
            .iter()
            .rposition(|s| matches!(s, TurnSegment::Assistant { .. }))
        else {
            return self;
        };
        self.segments.splice(pos..=pos, subsegments);
        self
    }

    /// Split the trailing assistant segment into a leading reasoning block + the
    /// answer.  `think_len` tokens of `<think>…</think>` sit at the start of the
    /// assistant body; `thinking_text` is its prose.  `ethereal = true` drops the
    /// reasoning K/V (the answer span absorbs those positions — used when a turn's
    /// reasoning is compressed out); `false` keeps it real.  Tiling is preserved
    /// either way.  No-op if there is no assistant segment or `think_len == 0`.
    pub fn with_thinking_split(
        mut self,
        thinking_text: String,
        think_len: u32,
        ethereal: bool,
    ) -> Self {
        if think_len == 0 {
            return self;
        }
        let Some(pos) = self
            .segments
            .iter()
            .rposition(|s| matches!(s, TurnSegment::Assistant { .. }))
        else {
            return self;
        };
        let TurnSegment::Assistant { text, kv } = self.segments[pos].clone() else {
            return self;
        };
        let tlen = think_len.min(kv.len);
        let (think_kv, answer) = if ethereal {
            // Reasoning K/V dropped: the answer keeps the whole region.
            (None, kv)
        } else {
            (
                Some(KvSpan::new(kv.offset, tlen)),
                KvSpan::new(kv.offset + tlen, kv.len - tlen),
            )
        };
        // The assistant body still holds the FULL decoded text (the reasoning
        // block followed by the answer). Strip the reasoning block off the front
        // so the answer segment carries ONLY the answer — otherwise consumers
        // render `<think>…</think>` twice: once as the `Thinking` segment, then
        // again at the head of `Assistant`.
        let answer_text = text.as_ref().map(|t| match t.find(thinking_text.as_str()) {
            Some(p) => t[p + thinking_text.len()..].to_string(),
            None => t.clone(),
        });
        self.segments[pos] = TurnSegment::Assistant {
            text: answer_text,
            kv: answer,
        };
        self.segments.insert(
            pos,
            TurnSegment::Thinking {
                text: thinking_text,
                kv: think_kv,
            },
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A normal user→assistant turn's grid tiled by its segments, validated to
    /// reproduce the exact token ids.
    #[test]
    fn flat_grid_tiles_and_realizes() {
        // grid: [user(3)][im_end(2)][assistant_start(3)][response(4)] = 12 tokens
        let grid: Vec<u32> = vec![
            10, 11, 12, /*im_end*/ 90, 91, /*a_start*/ 80, 81, 82, /*resp*/ 1, 2, 3,
            4,
        ];
        let layout = TurnLayout::from_flat_grid(
            0,  // user_content_start
            3,  // user_content_end
            8,  // assistant_start (3 + 2 + 3)
            12, // total
            2,  // im_end_len
            3,  // assistant_start_len
            "hi".into(),
            Some("ok".into()),
            false,
        );
        // Tiling holds against the real grid length.
        assert_eq!(layout.validate_tiling(grid.len() as u32), Ok(()));
        assert_eq!(layout.kv_len(), 12);
        // Real segments slice the grid back exactly, in order.
        let realized = layout.realize(&grid);
        let kinds: Vec<&str> = realized
            .iter()
            .map(|(s, _)| match s {
                TurnSegment::User { .. } => "user",
                TurnSegment::Assistant { .. } => "assistant",
                TurnSegment::Glue {
                    marker: GlueKind::ImEnd,
                    ..
                } => "im_end",
                TurnSegment::Glue {
                    marker: GlueKind::AssistantStart,
                    ..
                } => "a_start",
                _ => "?",
            })
            .collect();
        assert_eq!(kinds, vec!["user", "im_end", "a_start", "assistant"]);
        assert_eq!(realized[0].1, &[10, 11, 12]); // user body
        assert_eq!(realized[1].1, &[90, 91]); // im_end marker
        assert_eq!(realized[2].1, &[80, 81, 82]); // assistant_start marker
        assert_eq!(realized[3].1, &[1, 2, 3, 4]); // response
                                                  // Concatenating the real runs reproduces the whole grid.
        let rebuilt: Vec<u32> = realized
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, grid);
    }

    /// Ethereal segments (dropped-thinking, `/no_think`) contribute no K/V and do
    /// not advance the offset cursor.
    #[test]
    fn ethereal_segments_add_no_kv() {
        let layout = TurnLayout::new(vec![
            TurnSegment::Glue {
                marker: GlueKind::NoThink,
                kv: None,
            },
            TurnSegment::User {
                text: "q".into(),
                kv: KvSpan::new(0, 5),
            },
            // reasoning recorded but its K/V was dropped — adds zero positions
            TurnSegment::Thinking {
                text: "reasoned…".into(),
                kv: None,
            },
            TurnSegment::Assistant {
                text: Some("a".into()),
                kv: KvSpan::new(5, 7),
            },
        ]);
        assert_eq!(layout.kv_len(), 12);
        assert_eq!(layout.validate_tiling(12), Ok(()));
    }

    /// **The trailing twin.** A reserved tail makes the closing `<|im_end|>` a
    /// real span the turn owns, and the assistant body stops short of it.
    ///
    /// The pair of this and `leading_boundary_offsets_user_body` is what makes
    /// both-ends ownership representable: a turn that owned only its opener
    /// would drop every closer, because the assembler's inter-turn island is
    /// exactly `assistant_end ++ user_start` and splitting it at that seam is
    /// only lossless when both halves find an owner.
    #[test]
    fn a_reserved_tail_bakes_the_closing_im_end() {
        // grid: [us(4)][user(5)][im_end(2)][a_start(3)][answer(6)][im_end(2)]
        //        0..4   4..9     9..11      11..14      14..20     20..22
        let layout = TurnLayout::from_flat_grid_with_tail(
            4,
            9,
            14,
            22,
            2,
            3,
            2, // the reserved trailing marker
            "u".into(),
            Some("a".into()),
            false,
        );
        assert_eq!(layout.validate_tiling(22), Ok(()));

        // The answer stops short of the reserved tail.
        assert_eq!(layout.assistant_span(), KvSpan::new(14, 6));

        // Both ends are real, and the turn owns them.
        match layout.segments.first() {
            Some(TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(span),
            }) => assert_eq!(*span, KvSpan::new(0, 4)),
            other => panic!("expected a real leading UserStart, got {other:?}"),
        }
        match layout.segments.last() {
            Some(TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(span),
            }) => assert_eq!(*span, KvSpan::new(20, 2)),
            other => panic!("expected a real trailing ImEnd, got {other:?}"),
        }
    }

    /// A zero reservation reproduces the ethereal tail **exactly** — which is
    /// what every caller passes today, so the mechanism can exist before it is
    /// exercised without moving a single sealed turn.
    #[test]
    fn a_zero_tail_reservation_is_byte_identical_to_the_ethereal_form() {
        let args = || {
            (
                0u32,
                3u32,
                8u32,
                12u32,
                2u32,
                3u32,
                "hi".to_string(),
                Some("ok".to_string()),
                false,
            )
        };
        let (a, b, c, d, e, f, g, h, i) = args();
        let ethereal = TurnLayout::from_flat_grid(a, b, c, d, e, f, g, h, i);
        let (a, b, c, d, e, f, g, h, i) = args();
        let explicit = TurnLayout::from_flat_grid_with_tail(a, b, c, d, e, f, 0, g, h, i);
        assert_eq!(
            ethereal, explicit,
            "a zero tail must not change the layout in any way"
        );
        assert!(matches!(
            ethereal.segments.last(),
            Some(TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: None
            })
        ));
    }

    /// An over-large reservation cannot eat the assistant body — it clamps to
    /// what the grid actually holds past the assistant start.
    #[test]
    fn an_over_large_tail_reservation_clamps() {
        let layout = TurnLayout::from_flat_grid_with_tail(
            0,
            3,
            8,
            12,
            2,
            3,
            999,
            "u".into(),
            Some("a".into()),
            false,
        );
        assert_eq!(layout.validate_tiling(12), Ok(()));
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 0));
    }

    /// A leading boundary offset (non-zero `user_content_start`) is a real
    /// `UserStart` span — the user body is offset past it.
    #[test]
    fn leading_boundary_offsets_user_body() {
        let layout =
            TurnLayout::from_flat_grid(4, 9, 14, 18, 2, 3, "u".into(), Some("a".into()), false);
        assert_eq!(layout.validate_tiling(18), Ok(()));
        match &layout.segments[0] {
            TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(span),
            } => assert_eq!(*span, KvSpan::new(0, 4)),
            other => panic!("expected leading UserStart glue, got {other:?}"),
        }
    }

    /// Splitting a real thinking block out of the assistant body keeps the grid
    /// tiled (think + answer == the old assistant span).
    #[test]
    fn thinking_split_preserves_tiling() {
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            20, // assistant region [8,20) = 12 tokens
            2,
            3,
            "u".into(),
            Some("<think>r</think>a".into()),
            false,
        )
        .with_thinking_split("<think>r</think>".into(), 5, false);
        assert_eq!(layout.validate_tiling(20), Ok(()));
        // think [8,13), answer [13,20)
        let think = layout
            .segments
            .iter()
            .find_map(|s| match s {
                TurnSegment::Thinking { kv, .. } => *kv,
                _ => None,
            })
            .unwrap();
        assert_eq!(think, KvSpan::new(8, 5));
        // The reasoning prose lives ONLY on the Thinking segment; the Assistant
        // segment's text is the answer with the `<think>…</think>` block stripped
        // (so a consumer never renders the reasoning twice).
        let (think_text, asst_text) =
            layout
                .segments
                .iter()
                .fold((None, None), |(tt, at), s| match s {
                    TurnSegment::Thinking { text, .. } => (Some(text.clone()), at),
                    TurnSegment::Assistant { text, .. } => (tt, text.clone()),
                    _ => (tt, at),
                });
        assert_eq!(think_text.as_deref(), Some("<think>r</think>"));
        assert_eq!(asst_text.as_deref(), Some("a"));
        // The Assistant SEGMENT is answer-only (per-segment rendering), but the
        // `assistant_text()` ACCESSOR rejoins reasoning + answer into the full
        // verbatim reply the chat-history view needs.
        assert_eq!(
            layout.assistant_text().as_deref(),
            Some("<think>r</think>a")
        );
    }

    /// The clean-reprefill seal contract: the sealed grid physically OMITS the
    /// `<think>…</think>` tokens (they were re-prefilled away), so `realize()`
    /// reproduces a think-free grid and the K/V carries no reasoning — while the
    /// ethereal `Thinking` segment KEEPS the reasoning text, so `assistant_text()`
    /// still yields the full verbatim reply for display/history. This is what
    /// makes a projected past turn stop attending its own reasoning.
    #[test]
    fn clean_grid_seals_thinking_text_without_its_kv() {
        // CLEAN grid — no `<think>` tokens at all:
        // [user(3)][im_end(2)][a_start(3)][answer(2)] = 10 tokens.
        let clean_grid: Vec<u32> = vec![
            10, 11, 12, /*im_end*/ 90, 91, /*a_start*/ 80, 81, 82, /*answer*/ 1, 2,
        ];
        let layout = TurnLayout::from_flat_grid(
            0,  // user_content_start
            3,  // user_content_end
            8,  // assistant_start (3 + 2 + 3)
            10, // total — the CLEAN length (reasoning tokens absent)
            2,  // im_end_len
            3,  // assistant_start_len
            "u".into(),
            // The decoded text still carries the reasoning; the split moves it
            // onto an ethereal Thinking segment.
            Some("<think>r</think>a".into()),
            false,
        )
        .with_thinking_split("<think>r</think>".into(), 5, true);

        // Tiles the CLEAN grid — the reasoning contributes ZERO K/V positions.
        assert_eq!(layout.validate_tiling(10), Ok(()));
        assert_eq!(layout.kv_len(), 10);

        // The reasoning is recorded but ethereal (no span).
        assert!(layout.segments.iter().any(
            |s| matches!(s, TurnSegment::Thinking { kv: None, text } if text == "<think>r</think>")
        ));

        // `realize()` reproduces the clean grid EXACTLY — no reasoning tokens
        // appear in any real (K/V-bearing) segment.
        let rebuilt: Vec<u32> = layout
            .realize(&clean_grid)
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, clean_grid);

        // …yet the reasoning text survives for display: `assistant_text()`
        // rejoins Thinking + Assistant into the full verbatim reply.
        assert_eq!(
            layout.assistant_text().as_deref(),
            Some("<think>r</think>a")
        );
        // The answer span itself is reasoning-free (answer-only tokens).
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 2));
    }

    /// Dropping the reasoning K/V (ethereal) leaves the answer holding the whole
    /// region — still tiled.
    #[test]
    fn ethereal_thinking_split_preserves_tiling() {
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            20,
            2,
            3,
            "u".into(),
            Some("<think>r</think>a".into()),
            false,
        )
        .with_thinking_split("<think>r</think>".into(), 5, true);
        assert_eq!(layout.validate_tiling(20), Ok(()));
        // the thinking segment is ethereal (no K/V)
        assert!(layout
            .segments
            .iter()
            .any(|s| matches!(s, TurnSegment::Thinking { kv: None, .. })));
    }

    /// **A phase span covers the phase's real tokens and nothing else.**
    ///
    /// The span indexes a signature window directly, so an off-by-one here does
    /// not fail — it silently scores the wrong tokens and shifts every phase
    /// score by a token of glue.
    #[test]
    fn phase_spans_cover_exactly_their_own_segments() {
        // [user(0..3)][im_end(3..5)][a_start(5..8)][answer(8..12)]
        let layout =
            TurnLayout::from_flat_grid(0, 3, 8, 12, 2, 3, "hi".into(), Some("ok".into()), false);
        assert_eq!(phase_span_of(&layout.segments, Phase::User), Some(0..3));
        assert_eq!(
            phase_span_of(&layout.segments, Phase::Response),
            Some(8..12)
        );
        // No think block was split out, so there is no thinking phase at all —
        // distinct from "the thinking phase scored zero".
        assert_eq!(phase_span_of(&layout.segments, Phase::Thinking), None);
    }

    /// An ETHEREAL thinking segment (text kept, K/V dropped by the clean-turn
    /// re-prefill) has no tokens in the grid, so it must report no span rather
    /// than a zero-width one at the split point.
    #[test]
    fn an_ethereal_thinking_phase_has_no_span() {
        let layout =
            TurnLayout::from_flat_grid(0, 3, 8, 20, 2, 3, "hi".into(), Some("ok".into()), false)
                .with_thinking_split("<think>r</think>".into(), 5, true);
        assert_eq!(phase_span_of(&layout.segments, Phase::Thinking), None);
        // …while the phases that DO hold K/V are unaffected by the split.
        assert_eq!(phase_span_of(&layout.segments, Phase::User), Some(0..3));
    }

    /// A tool round-trip arrives as a second user segment mid-turn (the tool
    /// response), so the user phase is discontiguous. The span covers first to
    /// last: scoring the question alone would drop the tool's own output, which
    /// is the part naming the tool.
    #[test]
    fn a_split_phase_spans_first_to_last_segment() {
        let layout = TurnLayout::new(vec![
            TurnSegment::User {
                text: "q".into(),
                kv: KvSpan::new(0, 4),
            },
            TurnSegment::Assistant {
                text: Some("call".into()),
                kv: KvSpan::new(4, 3),
            },
            TurnSegment::User {
                text: "tool result".into(),
                kv: KvSpan::new(7, 5),
            },
            TurnSegment::Assistant {
                text: Some("answer".into()),
                kv: KvSpan::new(12, 2),
            },
        ]);
        assert_eq!(phase_span_of(&layout.segments, Phase::User), Some(0..12));
        assert_eq!(
            phase_span_of(&layout.segments, Phase::Response),
            Some(4..14)
        );
    }

    /// Semantic accessors read the right segments back out of a flat-grid
    /// layout (and survive a thinking split).
    #[test]
    fn accessors_read_segments() {
        let layout =
            TurnLayout::from_flat_grid(0, 3, 8, 12, 2, 3, "hi".into(), Some("ok".into()), false);
        assert_eq!(layout.user_text(), "hi");
        assert_eq!(layout.assistant_text().as_deref(), Some("ok"));
        assert_eq!(layout.thinking_text(), None);
        assert!(!layout.no_think());
        assert_eq!(layout.user_span(), KvSpan::new(0, 3));
        assert_eq!(layout.user_content_start(), 0);
        assert_eq!(layout.user_content_end(), 3);
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 4));
        // No thinking split → assistant content starts at the assistant span.
        assert_eq!(layout.assistant_content_start(), 8);

        // /no_think turn + a real thinking split moves the assistant content
        // start back to the thinking span's offset.
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            20,
            2,
            3,
            "u".into(),
            Some("<think>r</think>a".into()),
            true,
        )
        .with_thinking_split("<think>r</think>".into(), 5, false);
        assert!(layout.no_think());
        assert_eq!(layout.thinking_text(), Some("<think>r</think>"));
        assert_eq!(layout.assistant_content_start(), 8); // thinking span offset
        assert_eq!(layout.assistant_span(), KvSpan::new(13, 7));
    }

    /// A code_read tool-exchange turn: the single trailing `Assistant` span is
    /// replaced by the real `Assistant → ImEnd → UserStart → User → ImEnd →
    /// AssistantStart → Assistant` run, and the result still tiles the grid.
    #[test]
    fn assistant_split_tiles_tool_exchange() {
        // Base: user[0,3) im_end[3,5) a_start[5,8) assistant[8,24).
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            24,
            2,
            3,
            "excerpt".into(),
            Some("call…resp…ack".into()),
            true,
        );
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 16));
        // Sub-segments tiling [8,24): tc[8,10) im_end[10,12) us[12,15) tr[15,18)
        // im_end[18,20) as[20,23) ack[23,24).
        let subs = vec![
            TurnSegment::Assistant {
                text: Some("<tool_call>".into()),
                kv: KvSpan::new(8, 2),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(10, 2)),
            },
            TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(KvSpan::new(12, 3)),
            },
            TurnSegment::User {
                text: "<tool_response>".into(),
                kv: KvSpan::new(15, 3),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(18, 2)),
            },
            TurnSegment::Glue {
                marker: GlueKind::AssistantStart,
                kv: Some(KvSpan::new(20, 3)),
            },
            TurnSegment::Assistant {
                text: Some("Read …".into()),
                kv: KvSpan::new(23, 1),
            },
        ];
        let layout = layout.with_assistant_split(subs);
        assert_eq!(layout.validate_tiling(24), Ok(()));
        // Two user segments (header + tool response) and two assistant segments
        // (tool call + confirmation).
        let users = layout
            .segments
            .iter()
            .filter(|s| matches!(s, TurnSegment::User { .. }))
            .count();
        let assts = layout
            .segments
            .iter()
            .filter(|s| matches!(s, TurnSegment::Assistant { .. }))
            .count();
        assert_eq!((users, assts), (2, 2));
        // Assistant content still begins at the first assistant body (tool call).
        assert_eq!(layout.assistant_content_start(), 8);
    }

    /// Regression: a code_read scope turn must open EXACTLY like a normal turn —
    /// the leading `user_start` is ethereal glue (`kv: None`, materialized by the
    /// projection spine), NOT a baked `UserStart` span, and there is no separate
    /// `NoThink` glue segment: the scope folds its `/no_think` into the user body
    /// and seals with `no_think() == false`, so the assembler emits exactly one
    /// `user_start` and no duplicate soft-switch. Guards against the
    /// doubled-soft-switch / mislabeled-opener bug where `user_content_start > 0` +
    /// `no_think = true` made every code_read turn reconstruct as
    /// `[user_start][/no_think][/no_think][user]…`.
    #[test]
    fn code_read_scope_opener_matches_a_normal_turn() {
        // Scope params after the fix: user_content_start = 0, no_think = false, no
        // baked soft-switch.
        let layout = TurnLayout::from_flat_grid(
            0,
            5,
            8,
            24,
            2,
            3,
            "Summarize `x` (lines 1-5) in no more than two sentences.".into(),
            Some("<tool_call>…</tool_call>…<tool_response>…".into()),
            false,
        );
        // Leading glue is an ETHEREAL user_start — the spine materializes it, it is
        // not baked from this turn's grid.
        assert!(
            matches!(
                layout.segments.first(),
                Some(TurnSegment::Glue {
                    marker: GlueKind::UserStart,
                    kv: None,
                }),
            ),
            "expected an ethereal leading UserStart, got {:?}",
            layout.segments.first(),
        );
        // No NoThink glue segment at all — the scope carries no soft-switch.
        assert!(
            !layout.segments.iter().any(|s| matches!(
                s,
                TurnSegment::Glue {
                    marker: GlueKind::NoThink,
                    ..
                }
            )),
            "a scope turn must not carry a NoThink glue segment",
        );
        // The turn's no_think flag is false, so the assembler emits no `/no_think`.
        assert!(!layout.no_think());
        assert_eq!(layout.user_content_start(), 0);
        assert_eq!(layout.assistant_content_start(), 8);
    }

    /// **T3.6 — a sealed turn's suppression flag is a property of its own bytes,
    /// and nothing later can move it.**
    ///
    /// This is the condition the bake was granted on. §10 decision 9: *"assume
    /// fixed once sealed, and add the test. If the test fails, fall back to
    /// leaving `NoThink` ethereal."* Baking the switch into the turn's grid
    /// freezes a decision that used to be re-made on every projection, so the
    /// assumption underneath it has to be checked rather than read.
    ///
    /// Three things together make it hold, and each is asserted:
    ///
    /// 1. the flag is **derived from the segments**, not stored beside them, so
    ///    there is no second copy to drift;
    /// 2. it **survives the persistence round-trip**, which is the only way a
    ///    sealed layout ever comes back; and
    /// 3. two turns sealed under different dial settings keep their **own**
    ///    answers — the leak the live-glue path had to guard against, where a
    ///    past suppressed turn put a stale switch on a later thinking turn,
    ///    cannot happen when each turn's grid holds its own.
    #[test]
    fn a_sealed_turns_no_think_flag_is_fixed_by_its_own_grid() {
        // `/no_think` is a 3-token rider after the 1-token `user_start`, so the
        // suppressed turn's body starts at 4 and the thinking one's at 1.
        let suppressed = TurnLayout::from_flat_grid_with_tail(
            4,
            9,
            11,
            16,
            1,
            2,
            1,
            "hi".into(),
            Some("there".into()),
            true,
        );
        let thinking = TurnLayout::from_flat_grid_with_tail(
            1,
            6,
            8,
            13,
            1,
            2,
            1,
            "hi".into(),
            Some("there".into()),
            false,
        );

        assert!(suppressed.no_think(), "the baked switch must be visible");
        assert!(!thinking.no_think());

        // (1) Derived, not stored: the answer is exactly "is there a NoThink
        // segment", so removing it changes the answer and nothing else holds a
        // stale copy that could disagree.
        let mut stripped = suppressed.clone();
        stripped.segments.retain(|s| {
            !matches!(
                s,
                TurnSegment::Glue {
                    marker: GlueKind::NoThink,
                    ..
                }
            )
        });
        assert!(
            !stripped.no_think(),
            "the flag survived removal of the segment it is supposed to be read \
             from — there is a second copy, and the two can drift"
        );

        // (2) Survives the round-trip the substrate actually performs.
        for layout in [&suppressed, &thinking] {
            let json = serde_json::to_string(layout).expect("serialize");
            let back: TurnLayout = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(
                back.no_think(),
                layout.no_think(),
                "a reloaded turn disagreed with the one that was sealed"
            );
            assert_eq!(back.segments, layout.segments);
        }

        // (3) Independent: neither turn's answer depends on the other's, in
        // either order.
        let both = [&suppressed, &thinking];
        assert!(both[0].no_think() && !both[1].no_think());
        let reversed = [&thinking, &suppressed];
        assert!(!reversed[0].no_think() && reversed[1].no_think());
    }

    /// **Stream equivalence — the load-bearing test for boundary ownership.**
    ///
    /// The projection spine used to emit `user_start` before every sealed turn
    /// and `assistant_end` after it, so a run of turns came out as
    ///
    /// ```text
    ///   US body₀ AE  US body₁ AE  US body₂ AE
    ///   └island┘     └─island─┘    └─island─┘
    /// ```
    ///
    /// with each `AE ++ US` pair forming one gap-filled island between turns.
    /// Now each turn's own grid carries `US … AE`, and the turns abut directly.
    ///
    /// The token stream must be **identical** — equality, not tolerance, because
    /// §4.7a proves the split is exact rather than approximate. The failure this
    /// catches is a dropped or doubled `<|im_end|>`, which reads perfectly and
    /// shifts every boundary after it.
    #[test]
    fn baked_boundaries_reproduce_the_spine_emitted_stream_exactly() {
        // Dialect markers, as token ids.
        const US: &[u32] = &[100, 101];
        const AE: &[u32] = &[200];
        const UE: &[u32] = &[150];
        const AS: &[u32] = &[160, 161];

        // Three turns' bodies.
        let bodies: [(&[u32], &[u32]); 3] = [
            (&[1, 2, 3], &[10, 11]),
            (&[4, 5], &[12, 13, 14]),
            (&[6], &[15]),
        ];

        // ── The reference: what the spine used to emit. ──────────────────────
        let mut reference: Vec<u32> = Vec::new();
        for (user, answer) in bodies {
            reference.extend_from_slice(US);
            reference.extend_from_slice(user);
            reference.extend_from_slice(UE);
            reference.extend_from_slice(AS);
            reference.extend_from_slice(answer);
            reference.extend_from_slice(AE);
        }

        // ── The baked arrangement: each turn's grid carries its own markers,
        //    and `realize()` walks only what the turn actually owns. ──────────
        let mut baked: Vec<u32> = Vec::new();
        for (user, answer) in bodies {
            let mut grid: Vec<u32> = Vec::new();
            grid.extend_from_slice(US);
            grid.extend_from_slice(user);
            grid.extend_from_slice(UE);
            grid.extend_from_slice(AS);
            grid.extend_from_slice(answer);
            grid.extend_from_slice(AE);

            let us_len = US.len() as u32;
            let user_end = us_len + user.len() as u32;
            let asst_start = user_end + UE.len() as u32 + AS.len() as u32;
            let total = grid.len() as u32;
            let layout = TurnLayout::from_flat_grid_with_tail(
                us_len,
                user_end,
                asst_start,
                total,
                UE.len() as u32,
                AS.len() as u32,
                AE.len() as u32,
                "u".into(),
                Some("a".into()),
                false,
            );
            assert_eq!(
                layout.validate_tiling(total),
                Ok(()),
                "a baked turn must tile its own grid exactly"
            );
            // Every real segment, in order — which is what the inject path walks.
            for (_, toks) in layout.realize(&grid) {
                baked.extend_from_slice(toks);
            }
        }

        assert_eq!(
            baked, reference,
            "the baked-boundary stream diverged from the spine-emitted one — a \
             dropped or doubled boundary marker reads fine and shifts every \
             token position after it"
        );
    }

    /// The same equivalence with the `/no_think` switch present: it rides the
    /// opener, inside the turn that carries it.
    ///
    /// Each turn holding its own switch is *stronger* than the spine re-deciding
    /// it per projection, not weaker — the leak the live path guarded against
    /// (a past suppressed turn putting a stale switch on a later thinking-on
    /// turn) cannot happen when the switch lives in the suppressed turn's grid.
    #[test]
    fn a_suppressed_turns_switch_is_baked_into_its_own_opener() {
        const US: &[u32] = &[100];
        const NT: &[u32] = &[42];
        const AE: &[u32] = &[200];

        // head = user_start ++ no_think, so the opener span covers both.
        let head_len = (US.len() + NT.len()) as u32;
        let mut grid: Vec<u32> = Vec::new();
        grid.extend_from_slice(US);
        grid.extend_from_slice(NT);
        grid.extend_from_slice(&[1, 2]); // user body
        grid.extend_from_slice(&[150]); // user_end
        grid.extend_from_slice(&[160]); // assistant_start
        grid.extend_from_slice(&[10]); // answer
        grid.extend_from_slice(AE);

        let layout = TurnLayout::from_flat_grid_with_tail(
            head_len,
            head_len + 2,
            head_len + 4,
            grid.len() as u32,
            1,
            1,
            AE.len() as u32,
            "u".into(),
            Some("a".into()),
            true,
        );
        assert_eq!(layout.validate_tiling(grid.len() as u32), Ok(()));

        // The opener span covers `user_start ++ no_think` together.
        match layout.segments.first() {
            Some(TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(span),
            }) => {
                assert_eq!(*span, KvSpan::new(0, head_len));
                assert_eq!(&grid[span.range()], &[100, 42]);
            }
            other => panic!("expected a real leading UserStart, got {other:?}"),
        }
        // Realising reproduces the grid, switch included.
        let rebuilt: Vec<u32> = layout
            .realize(&grid)
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, grid);
    }

    /// **A compression turn is framed exactly like a dialogue turn.**
    ///
    /// It builds its grid directly rather than going through the turn-submit
    /// funnel, and it used to carry no head at all — correct while the assembler
    /// re-emitted one around every sealed turn, and silently wrong the moment
    /// turns started owning their boundaries. A summary with no opener runs
    /// straight on from the previous turn's closer with no role marker to say
    /// whose words it is.
    ///
    /// Exactly **one** opener: not zero (the assembler stopped and the builder
    /// was not updated) and not two (both fired).
    #[test]
    fn a_compression_turn_is_framed_like_a_dialogue_turn() {
        const US: &[u32] = &[100, 101];
        const AE: &[u32] = &[200];

        // [user_start][question(3)][user_end(1)][assistant_start(1)][answer(2)][im_end]
        let mut grid: Vec<u32> = Vec::new();
        grid.extend_from_slice(US);
        let user_content_start = grid.len() as u32;
        grid.extend_from_slice(&[1, 2, 3]);
        let user_end_at = grid.len() as u32;
        grid.extend_from_slice(&[150]);
        grid.extend_from_slice(&[160]);
        let asst_start_at = grid.len() as u32;
        grid.extend_from_slice(&[10, 11]);
        grid.extend_from_slice(AE);

        let layout = TurnLayout::from_flat_grid_with_tail(
            user_content_start,
            user_end_at,
            asst_start_at,
            grid.len() as u32,
            1,
            1,
            AE.len() as u32,
            "q".into(),
            Some("a".into()),
            false,
        );
        assert_eq!(layout.validate_tiling(grid.len() as u32), Ok(()));

        let openers = layout
            .segments
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    TurnSegment::Glue {
                        marker: GlueKind::UserStart,
                        kv: Some(_)
                    }
                )
            })
            .count();
        assert_eq!(
            openers, 1,
            "a compression turn must have exactly one opener"
        );

        // Real at both ends, and it reproduces its own grid.
        assert!(matches!(
            layout.segments.last(),
            Some(TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(_)
            })
        ));
        let rebuilt: Vec<u32> = layout
            .realize(&grid)
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, grid);
    }

    /// **The `TurnHalf` window excludes the baked opener** (P5b.8).
    ///
    /// The compression pass injects a sealed turn's user half and supplies its
    /// own framing. `turn_user_sealed_half` windows on
    /// `user_content_start..user_content_end`, which now starts *past* the baked
    /// marker — so the marker and the compression pass's framing cannot both
    /// land. This pins that the window is the user body alone.
    #[test]
    fn the_user_half_window_excludes_the_baked_opener() {
        let layout = TurnLayout::from_flat_grid_with_tail(
            4, // the opener occupies [0, 4)
            9,
            14,
            22,
            2,
            3,
            2,
            "u".into(),
            Some("a".into()),
            false,
        );
        let span = layout.user_span();
        assert_eq!(
            span,
            KvSpan::new(4, 5),
            "the user half must start past the opener, or a turn-half injection \
             carries the role marker into a pass that supplies its own"
        );
        assert_eq!(layout.user_content_start(), 4);
        assert_eq!(layout.user_content_end(), 9);
    }

    /// A gap between real segments is rejected.
    #[test]
    fn tiling_rejects_gaps() {
        let layout = TurnLayout::new(vec![
            TurnSegment::User {
                text: "q".into(),
                kv: KvSpan::new(0, 3),
            },
            // starts at 5, not 3 — a 2-token hole
            TurnSegment::Assistant {
                text: None,
                kv: KvSpan::new(5, 4),
            },
        ]);
        assert_eq!(
            layout.validate_tiling(9),
            Err(TilingError::OffsetMismatch {
                segment: 1,
                expected: 3,
                found: 5,
            })
        );
    }
}
