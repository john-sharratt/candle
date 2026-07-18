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

use serde::{Deserialize, Serialize};

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
        segments.push(TurnSegment::Assistant {
            text: assistant_text,
            kv: KvSpan::new(assistant_start, total.saturating_sub(assistant_start)),
        });
        // The turn closes with `<|im_end|>` (the glue shift's suffix). Like the
        // opener, it is materialized by the projection spine, not this turn's own
        // grid, so it is recorded ethereally.
        segments.push(TurnSegment::Glue {
            marker: GlueKind::ImEnd,
            kv: None,
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
            0, 3, 8, 24, 2, 3, "excerpt".into(), Some("call…resp…ack".into()), true,
        );
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 16));
        // Sub-segments tiling [8,24): tc[8,10) im_end[10,12) us[12,15) tr[15,18)
        // im_end[18,20) as[20,23) ack[23,24).
        let subs = vec![
            TurnSegment::Assistant { text: Some("<tool_call>".into()), kv: KvSpan::new(8, 2) },
            TurnSegment::Glue { marker: GlueKind::ImEnd, kv: Some(KvSpan::new(10, 2)) },
            TurnSegment::Glue { marker: GlueKind::UserStart, kv: Some(KvSpan::new(12, 3)) },
            TurnSegment::User { text: "<tool_response>".into(), kv: KvSpan::new(15, 3) },
            TurnSegment::Glue { marker: GlueKind::ImEnd, kv: Some(KvSpan::new(18, 2)) },
            TurnSegment::Glue { marker: GlueKind::AssistantStart, kv: Some(KvSpan::new(20, 3)) },
            TurnSegment::Assistant { text: Some("Read …".into()), kv: KvSpan::new(23, 1) },
        ];
        let layout = layout.with_assistant_split(subs);
        assert_eq!(layout.validate_tiling(24), Ok(()));
        // Two user segments (header + tool response) and two assistant segments
        // (tool call + confirmation).
        let users = layout.segments.iter().filter(|s| matches!(s, TurnSegment::User { .. })).count();
        let assts = layout.segments.iter().filter(|s| matches!(s, TurnSegment::Assistant { .. })).count();
        assert_eq!((users, assts), (2, 2));
        // Assistant content still begins at the first assistant body (tool call).
        assert_eq!(layout.assistant_content_start(), 8);
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
