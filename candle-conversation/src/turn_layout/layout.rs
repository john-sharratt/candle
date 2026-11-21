//! The ordered segments of one turn, and the accessors that read its user,
//! assistant and reasoning parts back out of them.
//!
//! A turn's segments tile its K/V grid exactly: each real segment starts where the
//! previous one ended, which `validate_tiling` checks. That tiling lets the layout
//! reproduce the stored token grid, and it is what makes a faithful description of
//! the turn's K/V. Turns sealed under the older flat grid are converted into
//! segments without changing the grid, and a leading or trailing marker is owned
//! by the turn at both ends or at neither, since the boundary between two turns
//! is split exactly that way.

use super::*;

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

    /// Whether [`Self::assistant_text`] would be non-empty, without building
    /// it: an assistant segment is present and some thinking or assistant text
    /// is.
    pub fn has_assistant_text(&self) -> bool {
        let mut has_assistant = false;
        let mut any_text = false;
        for s in &self.segments {
            match s {
                TurnSegment::Thinking { text, .. } => any_text |= !text.is_empty(),
                TurnSegment::Assistant { text, .. } => {
                    has_assistant = true;
                    any_text |= text.as_deref().is_some_and(|t| !t.is_empty());
                }
                _ => {}
            }
        }
        has_assistant && any_text
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
        let (think_kv, answer) = if ethereal {
            // Reasoning K/V dropped: the answer keeps the whole region. No span
            // is recorded, so `think_len` is a property of the TEXT only and is
            // expected to exceed the clean grid's assistant region — nothing
            // downstream has to align to it.
            (None, kv)
        } else {
            // **A span that had to be clamped is worse than no span at all.**
            // The reasoning has to occupy whole index pages for a later
            // projection to window it out, and the cuts that guarantee that were
            // made against the real `think_len`. If it does not fit the
            // assistant region — the answer was empty, or `max_tokens` landed on
            // the closing marker, so `total` dropped the trailing unforwarded
            // token and reserved the closing one — then a clamped end is
            // provably NOT a page boundary, and every later projection selecting
            // this turn fails permanently. Declining to split leaves the turn
            // whole: it shows its reasoning, which is visibly imperfect, rather
            // than poisoning the conversation.
            if think_len > kv.len {
                tracing::warn!(
                    think_len,
                    assistant_kv_len = kv.len,
                    "reasoning span exceeds the assistant region — sealing the turn WITHOUT a \
                     thinking split, because a clamped span cannot fall on index page boundaries"
                );
                return self;
            }
            let tlen = think_len;
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
