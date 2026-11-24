//! Writes content into a conversation as prefilled turns, and decodes only the
//! summary of a tool round-trip.
//!
//! Supplied content is prefilled in one forward pass and sealed as a turn, with
//! `/no_think` left to the projection as live glue rather than baked into the
//! grid. A round-trip's summary decodes under a structural `</think>` trigger, so
//! the budget goes to the summary, and a closing round bans the tool-call opener
//! so the model must answer. The repetition window covers only the summary,
//! because the dots in a directory listing would otherwise push the model off the
//! closing period.

use super::*;

/// The dialect's framing markers (`<|im_start|>system`, `<|im_end|>`, …) — the
/// glue the projection assembler wraps around the system prompt and each turn.
/// One decoded round of [`Sequence::ingest_roundtrip_chain`].
///
/// An ingest that gathers its own evidence needs all three of these: the `text`
/// to decide whether the model answered or issued a `<tool_call>`, the
/// `turn_index` to couple that call to the response the caller feeds back, and
/// the `tokens` to accumulate across the loop into the unit's ingest stat.
#[derive(Debug, Clone)]
pub struct ChainRound {
    /// The decoded assistant half — the unit's summary, or the `<tool_call>` the
    /// model issued in place of one — split as [`TurnResponse::answer`] splits
    /// it.
    pub answer: TurnText,
    /// Sealed index of the decoded turn. Deliberately NOT "the last turn": the
    /// async summariser can append a turn between the seal and the caller's
    /// [`Sequence::couple_turn`].
    pub turn_index: u32,
    /// Tokens prefilled plus decoded in this round.
    pub tokens: usize,
}

impl Sequence {
    /// Build the projection inputs the scheduler needs to run
    /// `Builder::project()` for this conversation's `target`.  The
    /// target itself is pinned on the slot (via the scheduler's
    /// `slot_targets` map at `NewSequence` time) and read from there
    /// at handler entry.
    pub(super) fn projection_inputs(&mut self) -> ProjectionInputs {
        ProjectionInputs {
            projection: Arc::clone(&self.projection),
            selection: self.selection.clone(),
            frame: {
                let window = self.frame_window.take();
                self.frame.take().map(|sections| {
                    Frame::of(
                        &self.substrate.read(),
                        sections,
                        self.target,
                        window.as_deref(),
                    )
                })
            },
        }
    }

    /// Insert a preformed turn into the conversation **without model inference**.
    ///
    /// Prefills the full user-and-assistant exchange into the KV
    /// cache in one forward pass, then runs the normal post-turn
    /// logic (summarisation tasks fire as usual with real inference).
    ///
    /// Goes through the same projection-then-prefill path as
    /// [`submit_turn`](Self::submit_turn): the parent is set to
    /// `system + projected_turns` before the exchange is prefilled,
    /// so the model's hidden states for the prefilled tokens reflect
    /// the same context the conversation would see for a normal
    /// `submit_turn`.  The only difference is `max_decode_tokens=0`
    /// — the assistant half is provided rather than generated.
    ///
    /// Useful for:
    /// - Injecting known exchanges cheaply in tests.
    /// - Seeding a conversation with prior context without re-running
    ///   the original decodes.
    ///
    /// Returns `Err(TurnInFlight)` if a turn is currently in progress.
    pub fn insert_turn(
        &mut self,
        user_message: impl Into<TurnText>,
        assistant_text: &str,
    ) -> crate::Result<()> {
        self.insert_turn_tagged(user_message, assistant_text, Vec::new())
    }

    /// [`insert_turn`](Self::insert_turn) with gather-scope tags — used to seed
    /// tagged calibration turns (e.g. `["tool"]`) that a projection policy's
    /// `tags:` filter scopes its provenance gallery to.
    pub fn insert_turn_tagged(
        &mut self,
        user_message: impl Into<TurnText>,
        assistant_text: &str,
        tags: Vec<String>,
    ) -> crate::Result<()> {
        self.insert_turn_inner(&user_message.into(), assistant_text, tags)?;
        Ok(())
    }

    /// [`insert_turn_tagged`](Self::insert_turn_tagged) + staged provenance
    /// linkage for ingest turns (repo_map folders, code_read scopes): after
    /// the turn seals, synthesizes two [`ProjectionEvent`]s — one for the
    /// user half (`start_token: 0`), one for the assistant half — whose
    /// `selection.turns` reference the turn itself and its immediate
    /// predecessor by `(timeline, index)`, and persists them keyed to the
    /// turn's stream id. Together with the turn's seal-time wide-Q signature
    /// this gives a later provenance scan the resolvable chain
    /// sig hit → event → turn.
    ///
    /// Returns the number of tokens prefilled (the full formatted grid), which
    /// the prefill-only ingest paths (repo map, code read) surface as the "tokens
    /// ingested" metric.
    pub fn insert_turn_staged(
        &mut self,
        user_message: impl Into<TurnText>,
        assistant_text: &str,
        tags: Vec<String>,
    ) -> crate::Result<usize> {
        let (assistant_content_start, turn_index, tokens) =
            self.insert_turn_inner(&user_message.into(), assistant_text, tags)?;
        let Some(idx) = turn_index else {
            // No substrate seal (no registered target) — nothing to key
            // events to; the turn itself was still prefilled.
            return Ok(tokens);
        };
        self.persist_staged_ingest_events(idx, assistant_content_start, 0.0)?;
        Ok(tokens)
    }

    /// Shared body of [`insert_turn_tagged`] / [`insert_turn_staged`]: format,
    /// tokenize, prefill through the shared projection path, drain to `Done`,
    /// finalize. Returns the assistant half's grid-token start and the sealed
    /// substrate `TurnIndex` (from the seal result — never derived by
    /// counting, which races the async summariser).
    pub(super) fn insert_turn_inner(
        &mut self,
        user_message: &TurnText,
        assistant_text: &str,
        tags: Vec<String>,
    ) -> crate::Result<(u32, Option<u32>, usize)> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }

        // Format the full exchange (user + assistant_start prefix +
        // assistant_text) and tokenize as a single prefill payload.
        //
        // Prefilled content turns (repo_map, tool ingests, …) carry SUPPLIED
        // content — they never run a decoded reasoning pass — so there is nothing
        // to suppress at ingest. The `/no_think` soft-switch is NOT baked into the
        // grid: like `submit_turn`, it is a live `Generated` glue segment the
        // projection re-emits around the sealed turn on every future projection
        // (gated on the turn's `no_think()` flag, set below). Baking it here — as
        // this path used to — sealed the switch into the turn K/V AND left the
        // assembler re-emitting a second one, so every reconstructed turn opened
        // `[user_start][/no_think][/no_think][user]…` (a doubled soft-switch, a
        // model-degrading malformed opener). The role markers follow the same
        // rule: `user_start` / `assistant_end` are stripped from the persisted
        // bytes and re-emitted as glue; only the intra-turn `user_end` /
        // `assistant_start` stay baked.
        // Format: {user}{user_end}{assistant_start}{assistant_text}. The assistant
        // role-end comes through `post_decode_tokens` — no decode here, so the EOS
        // isn't emitted by the model; the projection's live `Generated` segments
        // supply the surrounding `user_start` / `/no_think` / `assistant_end`.
        let user_text = user_message.text();
        let markers = format!(
            "{}{}",
            self.config.dialect.user_end, self.config.dialect.assistant_start,
        );
        let formatted = format!("{user_text}{markers}{assistant_text}");
        // The user half piece by piece (see `submit_turn_with_options`), then the
        // markers and the supplied reply as one string, exactly as before.
        let user_tokens = self.encode_text(user_message)?;
        let mut prefill_tokens = user_tokens.clone();
        prefill_tokens.extend_from_slice(&self.tokenize(&format!("{markers}{assistant_text}"))?);
        let post_decode_tokens = TokenBuffer::new();

        // Content boundaries inside the sealed grid. The prefill grid is
        // `[user_msg][user_end][assistant_start][assistant_text]` (no leading
        // `user_start`, no baked `/no_think` — both are live `Generated`
        // segments). The user body spans `[0, len(user_msg))`; the assistant
        // content begins after the `[user_end][assistant_start]` markers. The grid
        // is the user half's tokens then the markers', so the indices are their
        // lengths; clamp/monotonise so a tokenizer that merges across a join can
        // never invert the windows.
        let user_content_start = 0usize;
        let user_content_end = user_tokens.len();
        let assistant_content_start = user_tokens.len() + self.tokenize(&markers)?.len();
        let total = prefill_tokens.len();
        let user_content_start = (user_content_start.min(total)) as u32;
        let user_content_end =
            (user_content_end.min(total).max(user_content_start as usize)) as u32;
        let assistant_content_start = (assistant_content_start
            .min(total)
            .max(user_content_end as usize)) as u32;

        // Build the TokenizedText for both halves now — assistant
        // text is supplied directly, not decoded, so we can fill it
        // in without waiting on event_rx for token chunks.
        let user_tt = TokenizedText::new(user_text.as_str(), user_tokens);
        let asst_tokens = self.tokenize(assistant_text)?;
        let asst_tt = TokenizedText::new(assistant_text, asst_tokens);

        // Submit through the shared projection-then-prefill path with
        // `max_decode_tokens = 0` — the scheduler applies the
        // projection, prefills the full exchange, appends
        // `post_decode_tokens`, and emits Done without entering
        // decode.
        let inputs = self.projection_inputs();
        let handle = self.submit_prefill_unit(
            self.id,
            Some(inputs),
            formatted,
            prefill_tokens,
            user_text,
            user_content_start,
            user_content_end,
            assistant_content_start,
            // `no_think()` = true: the turn is a no-reasoning content turn, so the
            // projection re-emits `/no_think` as glue before it (NOT baked into the
            // grid — see above). Exactly one soft-switch on reconstruction.
            true,
            tags,
            // Structured ingest: one projection, no staged prefill segments.
            Vec::new(),
            // Prefill path: the assistant half is supplied (not decoded), so
            // hand it through to be stored verbatim as the turn's assistant_text.
            assistant_text.to_string(),
            post_decode_tokens,
            0,
            self.config.sampling.clone(),
            None,
            // A no-decode insert never samples, so no stencil can fire, none
            // can be entered, and no penalty applies to exempt from.
            Arc::new(TriggerRegistry::new()),
            None,
            false,
            None,
            Vec::new(),
            // A structured ingest carries no image.
            Vec::new(),
        )?;

        // Drain events synchronously to Done.  The handle's event_rx
        // receives Prefill (text echo) and Done; we just need to
        // observe Done so we know the parent's KV is fully populated
        // before we register the turn.
        let response = handle.wait()?;
        let turn_index = response.seal.as_ref().and_then(|s| s.turn_index);

        // Run the same post-Done finalize as a regular turn.
        self.finalize_turn_post_done(user_tt, asst_tt, response.seal.as_ref())?;
        Ok((assistant_content_start, turn_index, total))
    }

    /// Ingest one code scope as a TOOL ROUND-TRIP of two coupled turns — the
    /// correct shape for a simulated `read_file` call. Recording it as two turns
    /// (not one baked four-segment exchange) is what keeps the inter-turn role
    /// seams as REGENERATED live glue on every projection: a seam baked into K/V
    /// carries a hidden state computed against its ingest-time prefix (an empty
    /// scratch slot) and goes stale when the scope is re-injected at a different
    /// position mid-dialogue — the exact degradation provenance-selected attention
    /// forbids. Mirrors the dialogue's own tool-call recording via
    /// [`couple_turn`](Self::couple_turn):
    ///
    /// - Turn A (call): `user(request)` → `assistant(<tool_call>…)`, prefilled.
    /// - Turn B (response): `user(<tool_response>…code…)` → `assistant(summary)`,
    ///   DECODED under `/no_think` (short) — which rides as live glue, never
    ///   baked.
    ///
    /// The two turns are coupled, so provenance selecting either injects both,
    /// adjacent, with the seam regenerated between them. Serial by construction:
    /// Turn B's summary must decode with Turn A in its projected prefix. Returns
    /// tokens ingested (call prefill + decoded summary).
    pub fn ingest_scope_roundtrip(
        &mut self,
        call_user: &str,
        call_assistant: &str,
        response_user: impl Into<TurnText>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        triggers: Arc<TriggerRegistry>,
    ) -> crate::Result<usize> {
        let (call_idx, _resp_idx, tokens) = self.ingest_scope_roundtrip_indices(
            call_user,
            call_assistant,
            response_user,
            tags,
            max_summary_tokens,
            triggers,
        )?;
        // Serial path: couple the pair here (the parallel splice couples on the
        // file timeline instead, after adopting both turns — see `adopt_turn`).
        self.couple_turn(call_idx)?;
        Ok(tokens)
    }

    /// Core of [`Self::ingest_scope_roundtrip`] that returns the sealed
    /// `(call_turn_index, response_turn_index, tokens_ingested)` **without**
    /// coupling — so the parallel code_read path can run this on a per-scope FORK
    /// and then splice both turns onto the file timeline (coupling there). The
    /// two-turn shape and its live-glue seams are identical to the serial path.
    pub fn ingest_scope_roundtrip_indices(
        &mut self,
        call_user: &str,
        call_assistant: &str,
        response_user: impl Into<TurnText>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        triggers: Arc<TriggerRegistry>,
    ) -> crate::Result<(u32, u32, usize)> {
        let (idxs, tokens) = self.ingest_roundtrip_chain_indices(
            &[(TurnText::from(call_user), call_assistant.to_string())],
            response_user,
            tags,
            max_summary_tokens,
            &["file_read".to_string()],
            triggers,
        )?;
        match idxs.as_slice() {
            [call, resp] => Ok((*call, *resp, tokens)),
            _ => Err(ConversationError::Channel(
                "scope round-trip: expected exactly two turn indices".into(),
            )),
        }
    }

    /// [`Self::ingest_scope_roundtrip_indices`] awaited instead of blocked on —
    /// what the per-scope fan-out drives concurrently over its forks, so the
    /// scheduler co-batches every fork's prefill and summary decode without a
    /// thread per fork.
    pub async fn ingest_scope_roundtrip_indices_async(
        &mut self,
        call_user: &str,
        call_assistant: &str,
        response_user: &str,
        tags: Vec<String>,
        max_summary_tokens: usize,
        triggers: Arc<TriggerRegistry>,
    ) -> crate::Result<(u32, u32, usize)> {
        let (idxs, tokens) = self
            .ingest_roundtrip_chain_indices_async(
                &[(TurnText::from(call_user), call_assistant.to_string())],
                response_user,
                tags,
                max_summary_tokens,
                &["file_read".to_string()],
                triggers,
            )
            .await?;
        match idxs.as_slice() {
            [call, resp] => Ok((*call, *resp, tokens)),
            _ => Err(ConversationError::Channel(
                "scope round-trip: expected exactly two turn indices".into(),
            )),
        }
    }

    /// [`Self::ingest_roundtrip_chain_indices`] for a caller that owns the
    /// timeline itself (the serial path): couples the chain here rather than
    /// leaving it to a splice.
    ///
    /// **One round, not a whole ingest.** The decoded text comes back so the
    /// caller can see whether the model answered or issued a `<tool_call>`, run
    /// the tool, and submit the response as the next round's `decode_user` with
    /// an EMPTY `prefilled` — which is how `repo_map` drives a real agentic loop
    /// over a folder (`zend::repo_scan::converse`). The last turn is left
    /// uncoupled for exactly that reason: a round that turns out to be a call
    /// gets coupled to the response the caller then feeds back.
    ///
    /// `closing` bans the tool-call opener for this round, forcing an answer in
    /// prose on whatever the conversation has already gathered — the caller's
    /// round cap reaching its end.
    pub fn ingest_roundtrip_chain(
        &mut self,
        prefilled: &[(TurnText, String)],
        decode_user: impl Into<TurnText>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        force_tools: &[String],
        triggers: Arc<TriggerRegistry>,
        closing: bool,
    ) -> crate::Result<ChainRound> {
        let (indices, prefill_tokens, handle) = self.ingest_chain_submit(
            prefilled,
            decode_user,
            tags,
            max_summary_tokens,
            force_tools,
            triggers,
            closing,
        )?;
        let response = handle.wait_cancellable(self.substrate.cancel_token());
        let (indices, tokens, answer) =
            self.ingest_chain_settle(indices, prefill_tokens, handle, response)?;
        // Couple every turn except the last: each prefilled turn belongs with the
        // one that answers it, so the summariser sees the whole exchange rather
        // than a call with no response.
        for idx in indices.iter().take(indices.len().saturating_sub(1)) {
            self.couple_turn(*idx)?;
        }
        let turn_index = *indices.last().ok_or_else(|| {
            ConversationError::Channel("round-trip chain: no turns were sealed".into())
        })?;
        Ok(ChainRound {
            answer,
            turn_index,
            tokens,
        })
    }

    /// Ingest an N-turn tool round-trip whose LAST assistant turn is decoded.
    ///
    /// `prefilled` holds `(user, assistant)` pairs written verbatim — each is a
    /// request or a `<tool_response>` paired with the `<tool_call>` it provokes.
    /// `decode_user` is the final user turn (the last tool response); its
    /// assistant half is the model's own summary, and the only thing generated.
    ///
    /// Two turns is the `code_reading` scope shape (one call, one response);
    /// three is the `repo_map` folder shape (list, then read, then summarise).
    /// The turns are NOT coupled here, so the parallel path can run this on a
    /// fork and splice them onto the owning timeline.
    ///
    /// `force_tools` names every tool a prefilled call refers to, pinned into the
    /// catalog so each call is backed by a present definition.
    ///
    /// `triggers` steers the summary decode. It carries the `<think>` trigger
    /// bound to [`ThinkMode::Off`]'s tree — see the suppression note on the
    /// decode below for why an empty registry is not enough here.
    pub fn ingest_roundtrip_chain_indices(
        &mut self,
        prefilled: &[(TurnText, String)],
        decode_user: impl Into<TurnText>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        force_tools: &[String],
        triggers: Arc<TriggerRegistry>,
    ) -> crate::Result<(Vec<u32>, usize)> {
        let (indices, prefill_tokens, handle) = self.ingest_chain_submit(
            prefilled,
            decode_user,
            tags,
            max_summary_tokens,
            force_tools,
            triggers,
            false,
        )?;
        // Interruptible wait: a graceful shutdown mid-ingest latches the cancel
        // flag, and this returns `IngestCancelled` instead of waiting out the
        // in-flight summary decode. Dropping `handle` on that early return stops
        // the scheduler's decode at its next step. The ingest caller unwinds this
        // as "cancelled", not a decode failure.
        let response = handle.wait_cancellable(self.substrate.cancel_token());
        let (indices, tokens, _) =
            self.ingest_chain_settle(indices, prefill_tokens, handle, response)?;
        Ok((indices, tokens))
    }

    /// [`Self::ingest_roundtrip_chain_indices`] awaited instead of blocked on —
    /// the same submit, the same cancellable wait on the same latch, the same
    /// seal. This is what a concurrent scope fan-out drives per fork, so one
    /// thread can hold every fork's decode in flight at once.
    pub async fn ingest_roundtrip_chain_indices_async(
        &mut self,
        prefilled: &[(TurnText, String)],
        decode_user: impl Into<TurnText>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        force_tools: &[String],
        triggers: Arc<TriggerRegistry>,
    ) -> crate::Result<(Vec<u32>, usize)> {
        let (indices, prefill_tokens, handle) = self.ingest_chain_submit(
            prefilled,
            decode_user,
            tags,
            max_summary_tokens,
            force_tools,
            triggers,
            false,
        )?;
        // Armed across the await like every async turn wait: dropping this
        // future drops the handle (the decode winds down scheduler-side) and
        // must also clear the fork's in-flight guard, or the abandoned fork
        // rejects every later turn with `TurnInFlight`.
        let cancel = self.substrate.cancel_token().clone();
        let armed = CancelClearsInFlight::arm(self);
        let response = handle.wait_cancellable_async(&cancel).await;
        let (indices, tokens, _) =
            armed
                .disarm()
                .ingest_chain_settle(indices, prefill_tokens, handle, response)?;
        Ok((indices, tokens))
    }

    /// The tail both chain waits share: propagate the wait's verdict, read the
    /// sealed index off the response, and record the decoded turn with its
    /// staged provenance events. Returns the sealed indices, the tokens, and the
    /// decoded answer.
    pub(super) fn ingest_chain_settle(
        &mut self,
        mut indices: Vec<u32>,
        prefill_tokens: usize,
        handle: TurnHandle,
        response: crate::Result<TurnResponse>,
    ) -> crate::Result<(Vec<u32>, usize, TurnText)> {
        let response = response?;
        let resp_tokens = response.token_ids.len();
        let resp_idx = response
            .seal
            .as_ref()
            .and_then(|s| s.turn_index)
            .ok_or_else(|| {
                ConversationError::Channel(
                    "round-trip chain: decoded turn produced no index".into(),
                )
            })?;
        // Records the decoded turn + its staged provenance events.
        self.finish_turn_staged(handle, &response)?;
        indices.push(resp_idx);
        Ok((indices, prefill_tokens + resp_tokens, response.answer))
    }

    /// The head both chain waits share: selection framing, the prefilled turns,
    /// and the summary decode's submission. Returns the prefilled indices, the
    /// tokens they cost, and the in-flight summary's handle.
    pub(super) fn ingest_chain_submit(
        &mut self,
        prefilled: &[(TurnText, String)],
        decode_user: impl Into<TurnText>,
        tags: Vec<String>,
        max_summary_tokens: usize,
        force_tools: &[String],
        triggers: Arc<TriggerRegistry>,
        closing: bool,
    ) -> crate::Result<(Vec<u32>, usize, TurnHandle)> {
        // A scope round-trip frames on the dialogue prompt itself — the persona is
        // deliberately left alone. The turns it seals are later borrowed into
        // dialogue projections, and borrowed K/V carries the framing it was computed
        // under, and on the append-only ingest path that framing is NOT this
        // selection's to set: the prefix is the conversation's frame, laid on its
        // empty slot by its first turn (the schema's sections at each tree's
        // default branch), and `skip_projection` means no turn ever rebuilds it. The ingest's prompt is
        // therefore the same shared prompt KV a chat conversation uses, and the
        // schema's defaults are where its content is decided.
        //
        // The tool pin below stays because it is not ingest-only: any caller that
        // DOES re-project needs a coherent tool context, since the chain PREFILLS
        // `tool_call`s and their `tool_response`s and a model shown neither the
        // block nor those tools degrades (refusals, off-language, hallucinated tool
        // chatter). Enable the block and force-select exactly the prefilled tools
        // (see `FORCE_TOOL_SELECTOR`) — coherent, with no belief-driven catalog
        // noise.
        self.selection.set_optional(
            crate::projection::TOOLS_ENABLED_SELECTOR,
            crate::projection::OptionalState::Present,
        );
        let pinned_tools = force_tools.join(&crate::projection::FORCE_TOOL_SEPARATOR.to_string());
        self.selection
            .select(crate::projection::FORCE_TOOL_SELECTOR, pinned_tools.clone());
        // The prefilled turns — each `[user][assistant]` written verbatim, with
        // staged provenance so a later scan can resolve sig hit → event → turn.
        let mut indices: Vec<u32> = Vec::with_capacity(prefilled.len() + 1);
        let mut prefill_tokens = 0usize;
        for (user, assistant) in prefilled {
            let (acs, idx, tokens) = self.insert_turn_inner(user, assistant, tags.clone())?;
            let idx = idx.ok_or_else(|| {
                ConversationError::Channel(
                    "round-trip chain: prefilled turn produced no index".into(),
                )
            })?;
            self.persist_staged_ingest_events(idx, acs, 0.0)?;
            indices.push(idx);
            prefill_tokens += tokens;
        }
        // The final turn — submit the last tool_response and DECODE the summary,
        // `/no_think` + short + low-temp nucleus. The decode sees every prefilled
        // turn above (just recorded) in its projected prefix.
        //
        // `/no_think` alone does NOT reliably stop this hybrid 30B MoE from opening
        // a real `<think>` block: the empty block is *decoded*, not baked (see
        // `DecodeState::prefill_tokens` doc), so under sampling the model often opens
        // `<think>` and burns the whole `max_summary_tokens` budget on runaway —
        // frequently off-language (Chinese/Japanese) — reasoning, leaving a truncated
        // "thought" as the stored summary.
        //
        // **Three layers, and only the last one is structural.** `NO_THINK_SELECTOR`
        // adds the `/no_think` glue below, baked into this turn's own prefill — which
        // the model may simply ignore, because it is text. (Nothing in the prompt
        // helps: an append-only ingest is framed by the shared birth prompt at its
        // default branch, whose thinking dial is the dialogue's.)
        // `apply_think_mode` then programs the sampler's segment-close
        // budget, which for `Off` at a short summary budget collapses to a forced
        // empty block. That is enforcement, but it is *sampler* enforcement: it
        // fires only once the model has already opened the block, and it depends on
        // per-row close state holding across a wide ingest wave.
        //
        // `triggers` is the layer that cannot be ignored. It binds the `<think>`
        // trigger to `ThinkMode::Off`'s tree, whose entire body is the static run
        // `"\n\n</think>"` — it offers no free-decode span at all (asserted by
        // `stencil::think::off_has_no_free_span`), so the block closes on the token
        // after it opens and the whole budget goes to the summary. This is what
        // `compile_think_tree` means by "suppression is structural here, not
        // advisory", and leaving it out is what let a quarter of a `repo_map` pass
        // seal with a runaway thought and no summary.
        let mut summary_sampling = SamplingConfig::compression();
        // **Window the repetition penalty to the summary itself.**
        //
        // `compression()` leaves `repeat_last_n` at 0, which means "full history"
        // — and an ingest chain's history is the prefilled tool responses: a JSON
        // file listing, every entry of which carries a path like
        // `"candle-examples/examples/clip/README.md"`. At 1.3× multiplicative,
        // every `.` in every filename pushes the model off the `.` token, so the
        // summary ends on a complete clause with no full stop. Measured over one
        // pass: 31 of 61 folder summaries had no terminal period, and a summary
        // carrying its own earlier `.` was ~1.7× more likely to lack the final one
        // — the same effect, visible even against the much larger contribution
        // from the listing.
        //
        // A window keeps what the penalty is FOR and drops what it was never
        // meant to see: the degenerate loops it exists to break
        // (`"valid, valid, firm, firm, …"` — see `compression`) are local to the
        // generation, and 64 tokens is wider than any of them while being narrower
        // than the summary, so by the time the closing period is sampled the
        // window holds the model's own prose rather than a directory listing.
        summary_sampling.repeat_last_n = 64;
        // The line ends are a whole-vocabulary scan the engine's config already
        // carries; a fresh `compression()` config would repeat it per summary.
        summary_sampling.line_end_token_ids = Arc::clone(&self.config.sampling.line_end_token_ids);
        summary_sampling.apply_think_mode(ThinkMode::Off, &self.tokenizer, max_summary_tokens);
        // The CLOSING round of an agentic ingest loop: ban the tool-call opener
        // so a model still trying to gather evidence cannot open another call and
        // has to answer on what it already has. Structural, for the same reason
        // the `<think>` suppression above is: the request's prose asking for an
        // answer is text the model may ignore, and a round that seals as a
        // `<tool_call>` with nowhere to send it is the unit's whole budget spent
        // on no summary at all. Mirrors the live dialogue loop's repeat-guard
        // closing round.
        if closing {
            let open = self.config.sampling.tool_call_open_token_id;
            if open >= 0 {
                summary_sampling.banned_tokens.push(open);
            }
        }
        let mut opts = TurnOptions {
            max_tokens: Some(max_summary_tokens),
            sampling: Some(summary_sampling),
            tags,
            triggers,
            ..Default::default()
        };
        opts.selection.set_optional(
            crate::projection::NO_THINK_SELECTOR,
            crate::projection::OptionalState::Present,
        );
        // Same tools-ON pin as the conversation-level selection above: the decode's
        // own projection must also carry the coherent tool context the prefilled
        // tool_response turns refer to.
        opts.selection.set_optional(
            crate::projection::TOOLS_ENABLED_SELECTOR,
            crate::projection::OptionalState::Present,
        );
        opts.selection
            .select(crate::projection::FORCE_TOOL_SELECTOR, pinned_tools);
        let handle = self.submit_turn_with_options(decode_user, opts)?;
        Ok((indices, prefill_tokens, handle))
    }
}
