//! Seals a finished turn: it builds the turn's layout from the submit-time
//! boundaries, frees the slot's chunks, and sends the caller its `Done`.
//!
//! The reasoning span is taken from the boundary the decode measured rather than
//! re-derived from the text, so it names the exact tokens that occupy the K/V.
//! Display text drops an empty think block, while a real reasoning block stays in
//! the thinking segment. A code_read tool exchange is split into its real role
//! segments before the think split runs, because its closing summary may carry a
//! think block of its own. The stored token sequence drops the one token that was
//! sampled but never forwarded, so it lines up one to one with the K/V chunk grid,
//! which is the authority on the turn's length.

use super::*;

/// Whether a seal that came back with no recurrent snapshot is missing one.
///
/// Only a snapshot that was asked for can be missing. An ephemeral timeline's
/// export is skipped — its payload is dropped on arrival, so exporting it is
/// pure cost — and a model that carries no recurrent state has none to give.
/// Anything else that returns nothing is a model declaring state it cannot
/// export, which resumes with no memory of its history and reads fluently.
pub(super) fn missing_recurrent_hook(ephemeral: bool, carries_recurrent_state: bool) -> bool {
    !ephemeral && carries_recurrent_state
}

impl Scheduler {
    /// Build a turn's [`TurnLayout`] at seal time from the submit-time content
    /// boundaries and the per-half display text. The dialect marker lengths come
    /// from the scheduler's `boundary_markers`.
    ///
    /// `think_close_at` is the turn's reasoning boundary as
    /// [`DecodeState::think_close_at`] recorded it — the index into the
    /// generated run of the `</think>` that closed the block. The reasoning
    /// span is then `[assistant_content_start, first generated + that + 1)`:
    /// the assistant body up to and including the marker.
    ///
    /// **Measured, not re-derived.** This used to locate the block by searching
    /// the decoded text for `<think>` and re-tokenising it to get a length, so
    /// the span was only approximately the tokens it described — its own comment
    /// conceded that the answer span absorbed the round-trip remainder. That was
    /// harmless while the span was dropped wholesale; it stops being harmless
    /// the moment anything windows on it. The text search survives for the
    /// segment's display prose only, which is what it was always right for.
    ///
    /// **Every turn seals its reasoning as a REAL span.** Nothing is destroyed
    /// at write time to implement a read-time rule: the sealed grid is a
    /// faithful record of what was decoded, and the projection decides what a
    /// later turn may attend (`docs/thinking_span_projection.md`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_turn_layout(
        &self,
        user_content_start: u32,
        user_content_end: u32,
        assistant_content_start: u32,
        total: u32,
        user_text: String,
        assistant_text: String,
        no_think: bool,
        think_close_at: Option<u32>,
    ) -> TurnLayout {
        // A `/no_think` decode (dialogue turns under ThinkMode::Off, and every
        // code_read scope summary) collapses its reasoning to a bare
        // `<think></think>`. The sealed K/V is already reasoning-free via the clean
        // re-prefill, but the DISPLAY text still carries that empty block. Drop
        // empty think blocks from the display text here — before it is split into
        // segments — so neither the tool-exchange confirmation nor a dialogue
        // Thinking segment surfaces an empty `<think></think>`. Real (non-empty)
        // reasoning is left intact for the dialogue Thinking segment. Segment text
        // is display-only (KV spans come from explicit offsets), so this never
        // affects tiling.
        let assistant_text = crate::think_strip::strip_empty_think_blocks(&assistant_text);
        let im_end_len = self.boundary_markers.user_end.len() as u32;
        let assistant_start_len = self.boundary_markers.assistant_start.len() as u32;
        // The turn's own closing `<|im_end|>` rides the post-decode tail, so the
        // grid ends with it and the layout bakes it as a real span the turn
        // owns. Paired with the non-zero `user_content_start` the head bake
        // produces: both ends or neither (§4.7a).
        let trailing_marker_len = self.boundary_markers.assistant_end.len() as u32;
        let layout = TurnLayout::from_flat_grid_with_tail(
            user_content_start,
            user_content_end,
            assistant_content_start,
            total,
            im_end_len,
            assistant_start_len,
            trailing_marker_len,
            user_text,
            (!assistant_text.is_empty()).then(|| assistant_text.clone()),
            no_think,
        );
        // Code_read tool-exchange turns bake their assistant→user→assistant
        // sub-structure into `assistant_text` as dialect role markers; split the
        // single Assistant span into the real sub-segments so the layout — and
        // the details view built from it — reflects the true role boundaries.
        //
        // Checked BEFORE the `<think>` split: a tool-exchange turn is identified
        // unambiguously by its baked role boundaries, and its DECODED closing
        // summary may itself contain a `<think>` block (the scope grid carries no
        // `/no_think`). Were the think split to run first, that summary-internal
        // reasoning would hijack the layout and the tool exchange would never be
        // split. A normal assistant reply never contains the role-boundary marker,
        // so this branch is a no-op for ordinary turns and they fall through to the
        // think split below.
        if let Some(subs) =
            self.tool_exchange_segments(&assistant_text, assistant_content_start, total)
        {
            return layout.with_assistant_split(subs);
        }
        // Split the `<think>…</think>` reasoning out of the assistant body.
        //
        // The span comes from the boundary decode recorded, not from the text:
        // the generated run starts at `assistant_content_start`, so its index
        // `i` is grid position `assistant_content_start + i` and a block closing
        // at `i` is `i + 1` tokens of assistant body. Any preamble the model
        // emitted before `<think>` is inside that region, which is where it
        // belongs — it is reasoning too.
        //
        // No recorded boundary means no block was decoded (a prefilled
        // assistant half, or a tokenizer with no single `</think>` token), and
        // the turn seals without a `Thinking` segment.
        let Some(close_at) = think_close_at else {
            return layout;
        };
        // `with_thinking_split` clamps to the assistant span, which is the
        // right bound: it already excludes the reserved trailing marker.
        let think_len = close_at.saturating_add(1);
        // Display prose only. An empty result is correct rather than a failure
        // — a suppressed turn's collapsed block is stripped from the display
        // text upstream (`strip_empty_think_blocks`) while its tokens still
        // occupy the K/V the span describes.
        let block = match (
            assistant_text.find("<think>"),
            assistant_text.find("</think>"),
        ) {
            (Some(o), Some(c)) if c >= o => assistant_text[o..c + "</think>".len()].to_string(),
            _ => String::new(),
        };
        layout.with_thinking_split(block, think_len, false)
    }

    /// If `asst_text` is a code_read tool exchange — `<tool_call>`, the
    /// call→response boundary (`assistant_end` + `user_start`), `<tool_response>`,
    /// the response→close boundary (`user_end` + `assistant_start`), then the
    /// confirmation — split its single `[asst_start, total)` assistant span into
    /// the real `Assistant → ImEnd → UserStart → User → ImEnd → AssistantStart →
    /// Assistant` sub-segments. Boundary token offsets come from re-tokenising the
    /// byte prefixes, clamped monotonic into the span so the pieces always tile
    /// exactly (`validate_tiling` guards the seal). Returns `None` for an ordinary
    /// single-body assistant turn (a normal reply never contains the boundary).
    pub(super) fn tool_exchange_segments(
        &self,
        asst_text: &str,
        asst_start: u32,
        total: u32,
    ) -> Option<Vec<TurnSegment>> {
        let g = &self.boundary_markers;
        let call_to_resp = format!("{}{}", g.assistant_end_str, g.user_start_str);
        let resp_to_close = format!("{}{}", g.user_end_str, g.assistant_start_str);
        if call_to_resp.is_empty() || resp_to_close.is_empty() {
            return None;
        }
        // The tool_call is the read_file JSON (no markers), so the FIRST
        // call→response boundary is the real one. The tool_response is arbitrary
        // source, which could itself contain the response→close marker string, so
        // take the LAST occurrence: the real boundary sits just before the short,
        // marker-free confirmation, so `rfind` can't be fooled by embedded markup.
        let b1 = asst_text.find(&call_to_resp)?; // tool_call ends here
        let after1 = b1 + call_to_resp.len(); // tool_response starts here
        let b2 = after1 + asst_text[after1..].rfind(&resp_to_close)?; // tool_response ends
        let after2 = b2 + resp_to_close.len(); // confirmation starts here

        let span = total.saturating_sub(asst_start);
        let tlen = |s: &str| {
            self.tokenizer
                .encode(s, false)
                .map(|t| t.get_ids().len() as u32)
                .unwrap_or(0)
        };
        // Absolute grid offset of the byte prefix `asst_text[..byte]`, clamped
        // into the span so a tokenizer merge across a join can never break tiling.
        let at = |byte: usize| asst_start + tlen(&asst_text[..byte]).min(span);
        let im_end_len = tlen(&g.user_end_str);

        let a1 = at(b1); // end of tool_call
        let a2 = at(after1).max(a1); // end of call→response boundary
        let a3 = at(b2).max(a2); // end of tool_response
        let a4 = at(after2).max(a3).min(total); // end of response→close boundary
        let im1 = im_end_len.min(a2 - a1);
        let im2 = im_end_len.min(a4 - a3);

        Some(vec![
            TurnSegment::Assistant {
                text: Some(asst_text[..b1].to_string()),
                kv: KvSpan::new(asst_start, a1 - asst_start),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(a1, im1)),
            },
            TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(KvSpan::new(a1 + im1, (a2 - a1) - im1)),
            },
            TurnSegment::User {
                text: asst_text[after1..b2].to_string(),
                kv: KvSpan::new(a2, a3 - a2),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(a3, im2)),
            },
            TurnSegment::Glue {
                marker: GlueKind::AssistantStart,
                kv: Some(KvSpan::new(a3 + im2, (a4 - a3) - im2)),
            },
            TurnSegment::Assistant {
                text: Some(asst_text[after2..].to_string()),
                kv: KvSpan::new(a4, total.saturating_sub(a4)),
            },
        ])
    }

    /// Seal one finished turn whose disposition and closing tail are done, free
    /// its slot's chunks, and send the caller its `Done`.
    pub(super) fn seal_finished_turn(&mut self, turn: FinishedTurn) {
        let FinishedTurn {
            state,
            answer,
            decode_ms,
            total_ms,
            tokens_generated,
            tokens_per_second,
            sequence_stats,
            context_tokens,
            seal_slot,
            seal_block_from,
        } = turn;
        let text = answer.text();

        // Seal-and-write step.  When `seal_action != None`, we
        // snapshot `seal_slot` and apply the appropriate substrate
        // write (turn append or section pin).  The resulting
        // `SealResult` rides along on the Done event so the
        // conversation-side post-actions (cold store) can run
        // without a second round trip.
        // To the end of the turn: the seal, then the truncate and the `Done`.
        let _seal_span = profile::span("cleanup:seal");
        let seal_result = match &state.seal_action {
            SealAction::None => None,
            action => {
                // Bundle the per-half display text and the
                // combined token sequence the seal pinned
                // into the slot.  Text and tokens carry
                // distinct shapes here: the substrate stores
                // `user_text` / `assistant_text` as the
                // human-readable strings the caller supplied,
                // while `token_ids` carries the full slot
                // token sequence (prefill + decoded body +
                // post-decode tail) so cross-process replay
                // reconstructs the exact K/V the kernel saw.
                let turn_content = if matches!(action, SealAction::Turn) {
                    // user_text comes through verbatim from
                    // submit_turn (raw user message, no role
                    // markers, no /no_think prefix).
                    // assistant_text is the model's decoded
                    // body — special tokens skipped, just the
                    // reply the user sees streamed.  The
                    // substrate stores both halves verbatim;
                    // no caller assembles a combined string.
                    // The last entry in state.generated_tokens
                    // was sampled from the most recent forward
                    // pass but never forwarded itself — the
                    // loop terminated (EOS or max_tokens) before
                    // another forward could write its K/V into
                    // the slot.  Drop it so token_ids aligns
                    // 1:1 with the K/V chunk grid.
                    //
                    // **Deliberately NOT `forwarded_generated`.** That
                    // counter answers "is anything waiting to go out",
                    // which is all the injection path needs, and it is
                    // maintained by the paths that forward — so a path
                    // that forwards without touching it reads low.
                    // Sealing from it made `token_ids` 2 tokens shorter
                    // than the chunk grid on a measured turn (287 against
                    // 289), breaking the 1:1 contract this slice exists
                    // to keep. The grid's authority is the K/V.
                    let forwarded_generated: &[u32] = state
                        .generated_tokens
                        .split_last()
                        .map(|(_, rest)| rest)
                        .unwrap_or(&[]);
                    let mut full_tokens: Vec<u32> = Vec::with_capacity(
                        state.prefill_tokens.len()
                            + forwarded_generated.len()
                            + state.post_decode_tokens.len(),
                    );
                    full_tokens.extend_from_slice(&state.prefill_tokens);
                    full_tokens.extend_from_slice(forwarded_generated);
                    full_tokens.extend_from_slice(&state.post_decode_tokens);

                    // `text` is already written-then-decoded, so it is the
                    // whole assistant half either way: a prefill turn
                    // (repo_map / code_reading) supplies it verbatim and
                    // never decodes, and a decode turn with a seeded
                    // prefix carries both parts. Choosing between them —
                    // which this did — silently dropped the decoded half
                    // of any turn that had both.
                    let assistant_text = text.clone();
                    let total = full_tokens.len() as u32;
                    let layout = self.build_turn_layout(
                        state.user_content_start,
                        state.user_content_end,
                        state.assistant_content_start,
                        total,
                        state.user_text.clone(),
                        assistant_text,
                        state.no_think,
                        state.think_close_at,
                    );
                    Some(TurnContent {
                        role: Role::Assistant,
                        tags: state.tags.clone(),
                        layout,
                        token_ids: TokenBuffer::from(full_tokens),
                    })
                } else {
                    None
                };
                self.perform_seal_and_write(
                    seal_slot,
                    seal_slot,
                    seal_block_from,
                    action,
                    turn_content,
                )
                .unwrap_or_else(|e| {
                    tracing::warn!("post-Done seal failed for slot {}: {}", seal_slot, e,);
                    None
                })
            }
        };

        // Stateless-slot housekeeping: drop the slot's chunks
        // now that `perform_seal_and_write` has captured them
        // into the substrate residence. The next turn's
        // `apply_projection` rebuilds the slot from substrate
        // anyway, so holding onto these `Arc<ChunkGid>`s
        // between turns just pins arena slots that nothing
        // reads. Truncating to 0 drops the slot's Arc refs;
        // the residence keeps the chunks alive for the next
        // projection inject.
        //
        // **Only a slot with a projection target is stateless.** A raw
        // slot has nothing to rebuild its K/V from, and its recurrent
        // state stays on it either way; emptied, its next turn would
        // read a recurrence over tokens its attention no longer holds.
        // Its K/V stays with its state until the slot is freed.
        if self.slot_targets.contains_key(&seal_slot) {
            if let Err(e) = self.session.truncate_sequence_to_blocks(seal_slot.0, 0) {
                tracing::warn!(
                    "post-seal slot truncate failed for slot {}: {}",
                    seal_slot,
                    e
                );
            }
        }
        // The slot no longer holds what its last assembly placed.
        if let Some(slot_state) = self.slot_projection_state.get_mut(&seal_slot) {
            slot_state.placed_pieces.clear();
        }
        // An unresumed timeline keeps no snapshot, so no durable
        // signal will ever park its slot — and its state is never read again.
        // It goes now (`recurrent_park`).
        self.park_never_resumed(seal_slot);

        let _ = state.event_tx.send(TurnEvent::Done(TurnResponse {
            text,
            answer,
            token_ids: state.generated_tokens,
            stats: TurnStats {
                prefill_ms: state.prefill_ms,
                decode_ms,
                total_ms,
                tokens_generated,
                tokens_per_second,
                prefill_token_count: state.prefill_token_count,
                turn_prefill_tokens: state.prefill_tokens.len(),
                context_tokens,
                finish: state.finish,
                sequence: sequence_stats,
            },
            seal: seal_result,
        }));
    }
}
