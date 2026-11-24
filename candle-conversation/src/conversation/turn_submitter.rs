//! Lays out a turn's token grid and submits it to the scheduler for prefill and
//! decode.
//!
//! The grid is the user half, its join, the assistant opener and the decoded body,
//! and the turn carries its own opener and closer so a sealed turn injects whole.
//! Thinking suppression is per turn: a `/no_think` family bakes the switch into
//! the turn that carries it, and any other prefills a closed block, so a stale
//! suppression never reaches a later turn. The window in which a dialed first turn
//! may swap the prompt checkpoint is derived from the turn count.

use super::*;

impl Sequence {
    /// Submit a user turn for processing.
    ///
    /// Returns immediately with a [`TurnHandle`] that can be waited on
    /// or streamed from. Returns `Err(TurnInFlight)` if a turn is
    /// already in progress.
    ///
    /// After consuming the handle via streaming, call [`finish_turn`](Self::finish_turn)
    /// to record the assistant response and clear the in-flight guard.
    pub fn submit_turn(&mut self, user_message: &str) -> crate::Result<TurnHandle> {
        self.submit_turn_with_options(user_message, TurnOptions::default())
    }

    /// Submit with per-turn options (sampling, max tokens). The user half's
    /// literal pieces — a tool's output, a file's text — are spelled out so no
    /// part of them becomes a control token; see [`TurnText`].
    pub fn submit_turn_with_options(
        &mut self,
        user_message: impl Into<TurnText>,
        options: TurnOptions,
    ) -> crate::Result<TurnHandle> {
        let user_message = user_message.into();
        // An image is in the turn's text as its marker, ahead of the message,
        // as its block of ids is ahead of the message's ids: the record a
        // transcript reads, and what names the image's rows to a rebuild.
        let markers: String = options
            .images
            .iter()
            .map(|i| format!("{}\n", marker(&i.id, i.width, i.height)))
            .collect();
        let user_text = format!("{markers}{}", user_message.text());
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }
        self.sync_submitted_sections();

        // Adopt this turn's section-tree selection (the composer dials).  It
        // becomes the conversation's current selection and drives every
        // projection — initial prefill and decode reprojection — until the next
        // turn changes it.
        let selection_changed = self.selection != options.selection;
        self.selection = options.selection.clone();
        self.publish_ask_state();

        // A FIRST turn whose dials select a different prompt branch: the
        // checkpoint installed at create is the default branch's, because
        // create cannot know the dials. This is the one moment it can be
        // swapped — the state is still exactly that checkpoint, and the
        // projection this submit runs will show the newly selected sections.
        // State must match what the K/V shows (§4.6); without this, every
        // conversation whose first turn carried non-default dials ran on the
        // default branch's prompt memory while projecting the dialed prompt —
        // E2 of the behavioural catalogue caught it by the checkpoint counters
        // sitting still.
        //
        // **The window is DERIVED, not tracked.** `state_is_prompt_only` says
        // this conversation was born with a checkpoint (a fork's state is real
        // and must never be swapped); the empty timeline says nothing has
        // advanced that state since. A tracked "still pristine" bool has to be
        // cleared by every path that advances the state, and the paths that
        // are not this one — `insert_turn_inner`, `submit_prefilled_turn`,
        // `catch_memory_up_to` — each seal or adopt a turn onto this timeline
        // BEFORE any later submit runs, so the turn count states the same fact
        // without anyone having to remember. Without the derived half, a
        // code_read conversation that ingested scopes and then took a dialed
        // first turn had the bare prompt checkpoint installed over the state
        // that had absorbed them: K/V holding the ingest, recurrent layers
        // having forgotten it — the very defect this block exists to remove.
        let timeline_is_empty = self.substrate.read().turn_count(self.target.timeline) == 0;
        //
        // **A changed selection is not a changed branch.** Dials that select no
        // section tree — the thinking switch, a response-length dial on a prompt
        // without that tree — leave the prompt exactly as it was born, so the
        // branch they name is the one already installed, and the swap would read
        // the ~63 MiB checkpoint back off the log (the memo holds one branch, and
        // every conversation with its own prompt is its own branch), decode it,
        // and install the state the slot already holds. Measured on eight
        // conversations submitted together: 123–300 ms per submit, the whole of
        // the batch's prefill window. Naming the branch is a walk of the sections
        // already in the substrate; only a branch that differs is rebuilt.
        if selection_changed && self.state_is_prompt_only && timeline_is_empty {
            let birth = Arc::clone(&self.birth_prefix);
            let spans = self.branch_spans.clone();
            let (selected, _) = self.prompt_branch(&birth, &spans);
            if needs_branch_swap(self.installed_branch, selected) {
                if let Some((prefix, fresh)) = self.build_branch_checkpoint(&birth, &spans)? {
                    self.restore_branch_checkpoint(prefix, fresh)?;
                    self.installed_branch = Some(prefix);
                }
            }
        }
        // Whatever branch it started from, this turn's decode advances the
        // state past the prompt — the swap window closes.
        self.state_is_prompt_only = false;

        let inputs = self.projection_inputs();
        let (handle, user_tokens) = self.submit_turn_on(
            self.id,
            inputs,
            &user_message,
            user_text.clone(),
            options,
            self.build_reprojection_policy(),
        )?;
        // Record the pending user turn (text + raw tokens for the tree).
        self.pending_user = Some(TokenizedText::new(user_text.as_str(), user_tokens));
        self.turn_in_flight = true;
        Ok(handle)
    }

    /// Record the projection and selection an [`UnsealedAsker`] reads.
    pub(in super::super) fn publish_ask_state(&self) {
        let mut state = self.ask_state.lock().unwrap();
        state.projection = Arc::clone(&self.projection);
        state.selection = self.selection.clone();
    }

    /// A handle that can put a question to this conversation without borrowing it.
    pub fn unsealed_asker(&self) -> UnsealedAsker {
        UnsealedAsker {
            submitter: self.submitter.clone(),
            id: self.id,
            state: Arc::clone(&self.ask_state),
        }
    }
}

impl TurnSubmitter {
    /// Compose a user turn's prefill and submit it for decode on `sequence_id`.
    ///
    /// The one place a turn's grid is laid out, shared by the sealing path
    /// ([`Sequence::submit_turn_with_options`], on the conversation's own slot)
    /// and the unsealed one ([`UnsealedAsker::ask`], on an ephemeral slot).
    /// Touches no conversation state — the caller owns the in-flight guard and the
    /// pending user turn — and returns the encoded user half so the sealing caller
    /// can record it. `reprojection` enables mid-decode re-projection, which only
    /// a turn that is going to be sealed has any use for.
    pub(super) fn submit_turn_on(
        &self,
        sequence_id: SequenceId,
        projection_inputs: ProjectionInputs,
        user_message: &TurnText,
        user_text: String,
        options: TurnOptions,
        reprojection: Option<ReprojectionPolicy>,
    ) -> crate::Result<(TurnHandle, TokenBuffer)> {
        // Pin the system prompt into the substrate (idempotent).
        // Subsequent SubmitTurn calls re-inject it onto the slot via
        // `apply_projection`; the upload cache amortises after the
        // first turn.

        // The persisted turn shape is
        //     user_message + user_end + assistant_start + decoded_body
        // — i.e. `user_start` is **not** in the prefill and `assistant_end`
        // is **not** appended after decode.  Both boundary markers are
        // emitted as live `Generated` segments by the projection engine
        // (the trailing `Generated(UserStart)` the scheduler appends
        // before this prefill, and the `Generated(AssistantEnd)` that
        // closes every past-turn injection on the next projection that
        // includes this turn).  Their K vectors are computed under the
        // actual runtime causal prefix on every projection, so the
        // attention-pivot K/V stays correct as projection-selected
        // context changes.  The intra-turn `user_end` and
        // `assistant_start` markers stay baked — their hidden state is
        // dominated by the turn's own (invariant) content.
        //
        // Thinking suppression is PER-TURN, driven by the composer effort dial.
        // Qwen3's `/no_think` soft-switch is only honoured from the user turn, not
        // from the system prompt — so a suppressed turn carries it in its own grid,
        // baked immediately after `user_start` by `turn_head_tokens`. Baking it
        // into the turn that carries it is what keeps the dial per-turn: a past
        // suppressed turn cannot put a stale switch on a later thinking-on turn
        // when each turn's grid holds its own.  That covers the families whose
        // switch lives in the user turn; one whose `no_think` is empty suppresses
        // in the assistant header instead, via the closed block appended below.
        // A thinking turn opens its own `<think>` either way.
        let assistant_start_marker = self.config.dialect.assistant_start;
        // Optional assistant prefill: text seeded as the start of the response so
        // the decode is forced to continue from it (e.g. `<tool_call>` commits to
        // the tool-call grammar). It is pinned into the prefill grid and sealed
        // as assistant content (see `assistant_content_start` below); the model
        // decodes the continuation. Empty when unset — the path is then identical
        // to an ordinary turn.
        let assistant_prefill = options.assistant_prefill.as_deref().unwrap_or("");
        // Whether the composer's thinking dial suppressed this turn.  Decided here
        // rather than at the seal below because it also selects the dialect's
        // suppression mechanism, and that mechanism is part of the prefill grid.
        let no_think = matches!(
            options
                .selection
                .optional(crate::projection::NO_THINK_SELECTOR),
            Some(crate::projection::OptionalState::Present)
        );
        // **How THIS dialect suppresses thinking.** `Dialect::thinking_suppression`
        // owns the split: a family with a `/no_think` soft switch carries it in the
        // user opener (`turn_head_text`), and one WITHOUT — Qwen3.5 / Qwen3.8, whose
        // `no_think` is deliberately empty — suppresses by opening the assistant turn
        // with the reasoning block already closed, exactly as its own template renders
        // `enable_thinking=false`.
        //
        // Production used to read only the user-turn half, so on this family the
        // `no_think` node emitted a zero-token segment and "thinking off" was a line
        // of prose in the system prompt rather than a structural guarantee. Measured
        // on the repo_map ingest: 22 of 22 summaries opened a block regardless, and
        // 12 of them stored the model's raw monologue as the summary.
        let closed_think = self.config.dialect.thinking_suppression(no_think).1;
        // The user half is encoded piece by piece, then the markers and lead
        // follow as one string, exactly as they always have. The half meets the
        // markers on `user_end`, a registered tag the tokenizer splits at anyway,
        // so the concatenation is the whole grid's encoding.
        // The user half opens with its images, each a block of ids — start,
        // one pad per row, end — ahead of the message; the message's spans move
        // past them.
        let (image_ids, image_spans) = self.image_blocks(&options.images)?;
        let images_len = image_ids.len();
        let mut user_tokens = TokenBuffer::from(image_ids);
        user_tokens.extend_from_slice(&self.encode_text(user_message)?);
        // The user half opens the grid, so a span's tokens in the half are its
        // tokens in the grid until the turn's head is baked on.
        let recurrent_spans = options
            .recurrent_spans
            .iter()
            .map(|s| {
                token_span(
                    &self.tokenizer,
                    &self.literal_tokenizer,
                    user_message,
                    s.bytes.clone(),
                )
                .map(|tokens| TokenSpan {
                    tokens: tokens.start + images_len..tokens.end + images_len,
                    kind: s.kind.clone(),
                })
                .map_err(|e| ConversationError::Tokenizer(e.to_string()))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let markers = format!("{}{}", self.config.dialect.user_end, assistant_start_marker);
        let assistant_head = format!("{user_text}{markers}");
        let lead = assistant_lead(closed_think, assistant_prefill);
        let formatted = format!("{assistant_head}{lead}");
        let mut prefill_tokens = user_tokens.clone();
        prefill_tokens.extend_from_slice(&self.tokenize(&format!("{markers}{lead}"))?);

        // A turn that begins inside a grammar carries that grammar's opening
        // scaffold in its prefill: everything from the tree's root up to the
        // first choice it actually leaves open. Those tokens are structural —
        // `{"name": "` and the like — so writing them costs one prefill pass
        // where decoding them costs a step each, and they cannot come out any
        // other way because they are not sampled at all.
        //
        // Appended as **token ids**, not text: they come from the compiled tree,
        // so re-tokenizing a rendering of them could merge across the join and
        // put the driver and the K/V on different nodes. The scheduler replays
        // the same walk to arrive at the same place.
        //
        // The same tokens are also recorded as the turn's *written* assistant
        // text, so the reply the caller gets back is the whole assistant half
        // rather than only the sampled tail. A caller that seeds `{"name": "` to
        // commit the shape and is then handed a fragment starting mid-JSON
        // cannot parse the shape it just enforced.
        let mut assistant_written = assistant_prefill.to_string();
        if let Some(grammar) = &options.turn_grammar {
            let (scaffold, _) = grammar.driver().opening();
            // Specials kept: `<tool_call>` and friends are part of the syntax
            // the caller is going to parse, not decoration around it.
            assistant_written
                .push_str(&self.tokenizer.decode(&scaffold, false).unwrap_or_default());
            prefill_tokens.extend_from_slice(&scaffold);
        }

        // No post-decode tail in the turn layout.  The model's
        // `<|im_end|>` EOS doesn't get forwarded (sampling stops on
        // EOS without a follow-up forward pass), and the trailing
        // `\n` that used to come from `assistant_end` is now the live
        // `Generated(AssistantEnd)` segment the projection emits
        // after the sealed turn on every future projection.
        let post_decode_tokens = TokenBuffer::new();

        let sampling = turn_bans::with_turn_bans(
            options
                .sampling
                .unwrap_or_else(|| self.config.sampling.clone()),
            &options.turn_banned_tokens,
        );
        let max_tokens = options
            .max_tokens
            .unwrap_or(self.config.max_response_tokens);

        tracing::trace!(
            max_decode_tokens = max_tokens,
            forced_eos_after = sampling.forced_eos_after,
            graceful_eos_after = sampling.graceful_eos_after,
            eos_ramp_start = sampling.eos_ramp_start,
            "turn submitted"
        );

        let free_tool_calls_from_penalties = projection_inputs
            .projection
            .schema()
            .free_tool_calls_from_penalties;
        // Content boundaries inside the sealed grid.  The prefill grid is
        // `[user_msg][user_end][assistant_start]` — the leading `user_start` (and,
        // when suppressing, `/no_think`) are live `Generated` glue segments emitted
        // by the scheduler, NOT part of the prefill — so the user body spans
        // `[0, len(user_msg))`.
        let user_content_start = 0;
        let user_content_end = user_tokens.len();
        // Assistant content begins after the assistant header AND after any
        // suppression block, but BEFORE a caller's own prefill — so that prefill's
        // K/V seals as part of the assistant turn while the block's does not.
        //
        // **The suppression block is scaffolding, not assistant content.** It is
        // prefilled, never decoded, and `strip_empty_think_blocks` removes it from
        // the stored text — so a span that began before it would describe ~4 more
        // tokens than the text it is paired with, and every consumer that maps
        // text offsets onto that span (`tool_exchange_segments` carving a code_read
        // sub-segment, the thinking split) would read shifted K/V.
        //
        // A caller's own prefill is the opposite case: a `<tool_call>` seed IS
        // content, appears in the stored text, and stays inside the span. Hence the
        // boundary sits between the two — after the block, before the seed. With
        // neither present this is `prefill_tokens.len()`, the ordinary turn's
        // "content starts where decoding starts".
        //
        // With no caller prefill this is `prefill_tokens.len()` — the block, when
        // there is one, is the whole tail of the grid, so content begins exactly
        // where decoding does. Taking that route matters: `assistant_head` embeds
        // the user message, so the `else` arm re-tokenises the entire message, and
        // routing every turn through it would pay that on each submit (a
        // `<tool_response>` carrying a file excerpt on the ingest path).
        let assistant_content_start = if assistant_prefill.is_empty() {
            prefill_tokens.len()
        } else {
            user_tokens.len() + self.tokenize(&format!("{markers}{closed_think}"))?.len()
        };
        // Clamp to the prefill length and force monotonic so a tokenizer that
        // merges across a join can never invert the windows at seal time.
        let total = prefill_tokens.len();
        let user_content_start = (user_content_start.min(total)) as u32;
        let user_content_end =
            (user_content_end.min(total).max(user_content_start as usize)) as u32;
        let assistant_content_start = (assistant_content_start
            .min(total)
            .max(user_content_end as usize)) as u32;
        let handle = self.submit_prefill_unit(
            sequence_id,
            Some(projection_inputs),
            formatted,
            prefill_tokens,
            user_text,
            user_content_start,
            user_content_end,
            assistant_content_start,
            no_think,
            options.tags.clone(),
            // Decode path: reprojection fires from the decode loop, not staged
            // prefill offsets.
            Vec::new(),
            // Decode path: whatever was *written* into the assistant half — a
            // seeded prefix, a turn grammar's opening scaffold. The decoded text
            // is appended to it, so both the reply and the seal carry the whole
            // turn. Empty for an ordinary turn, which then reads as before.
            assistant_written,
            post_decode_tokens,
            max_tokens,
            sampling,
            reprojection,
            options.triggers,
            options.turn_grammar,
            // Read off the schema, not the turn: whether tool arguments are
            // quotations or prose is a property of what this conversation *is*,
            // and a per-turn switch would be a way to get it wrong on one turn.
            free_tool_calls_from_penalties,
            options
                .recorded_turn
                .map(|g| (g, options.forced_tokens, options.forced_suffix)),
            recurrent_spans,
            image_spans,
        )?;
        Ok((handle, user_tokens))
    }

    /// `images` as the ids the user half opens with, and each image's span in
    /// them.
    fn image_blocks(&self, images: &[TurnImage]) -> crate::Result<(Vec<u32>, Vec<ImageSpan>)> {
        if images.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let tokens = VisionTokens::of(&self.tokenizer).ok_or_else(|| {
            ConversationError::Other(
                "a turn with an image, and this model's tokenizer has no vision tokens".into(),
            )
        })?;
        let mut ids = Vec::new();
        let mut spans = Vec::with_capacity(images.len());
        for image in images {
            spans.push(ImageSpan {
                start: ids.len() + 1,
                handle: image.handle.clone(),
            });
            ids.extend(tokens.block(image.handle.len()));
        }
        Ok((ids, spans))
    }
}

impl TurnSubmitter {
    /// Send a `SubmitTurn` request to the scheduler, returning the
    /// streaming handle.
    ///
    /// The shared projection-then-prefill primitive that backs every
    /// content-into-KV path — turn submission, no-decode turn
    /// insertion, and section ingestion (one section onto a fork per
    /// parallel slot).  Callers control:
    ///
    /// - `parent_id` — which sequence the scheduler materialises the
    ///   projection into (the conversation's slot for turns, a
    ///   fork's slot for a parallel section).  The scheduler resets
    ///   it to empty before applying the projection — the substrate
    ///   is the canonical parent.
    /// - `projection_inputs` — when `Some`, the scheduler runs
    ///   `Builder::project()` itself against its substrate; when
    ///   `None`, no projection is applied.
    /// - `max_decode_tokens` — `0` for no-decode prefill (insert
    ///   paths); `> 0` to follow with a decode loop.
    /// - `reprojection` — `Some` enables continuous mid-decode
    ///   re-projection; `None` runs a single-shot prefill+decode.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit_prefill_unit(
        &self,
        sequence_id: SequenceId,
        projection_inputs: Option<ProjectionInputs>,
        prefill_text: String,
        prefill_tokens: TokenBuffer,
        user_text: String,
        user_content_start: u32,
        user_content_end: u32,
        assistant_content_start: u32,
        no_think: bool,
        tags: Vec<String>,
        projection_offsets: Vec<u32>,
        prefill_assistant_text: String,
        post_decode_tokens: TokenBuffer,
        max_decode_tokens: usize,
        sampling: SamplingConfig,
        reprojection: Option<ReprojectionPolicy>,
        triggers: Arc<TriggerRegistry>,
        turn_grammar: Option<TurnGrammar>,
        free_tool_calls_from_penalties: bool,
        recorded_turn: Option<(Vec<u32>, Option<usize>, Vec<u32>)>,
        recurrent_spans: Vec<TokenSpan>,
        image_spans: Vec<ImageSpan>,
    ) -> crate::Result<TurnHandle> {
        // ── Bake the turn's own boundary markers into its grid ──────────────
        //
        // A turn owns its leading `user_start` (with its `/no_think`, when the
        // dial suppressed thinking) and its trailing `<|im_end|>`. Both used to
        // be live `Generated` glue the projection spine re-emitted around every
        // sealed turn; they are now part of the turn's own prefill, so a sealed
        // turn injects complete and the spine emits nothing between turns.
        //
        // **Both ends or neither.** Every inter-turn island the assembler used
        // to emit is exactly `assistant_end ++ user_start`, so splitting it at
        // that seam is lossless only when each turn takes both halves — leading
        // only would drop every closer, trailing only every opener. Pinned by
        // `every_inter_turn_island_is_assistant_end_then_user_start`.
        //
        // Central, so it cannot diverge between the three turn-producing paths
        // (dialogue, prefilled calibration, inserted). A path that baked its
        // opener and not its closer would read fine and shift every boundary
        // after it.
        //
        // Uniform across models, deliberately: the emitted token stream is
        // identical either way, so there is no reason for a glue-capable model
        // to take a different path. What differs is only that the marker's K is
        // now computed under the prefix present at seal rather than re-derived
        // per projection — the same approximation the substrate already makes
        // for every sealed turn's own K/V. `TurnSegment::Glue { kv: Some(..) }`
        // keeps regeneration reachable for a model that wants it.
        let head = self.turn_head_tokens(no_think)?;
        let head_len = head.len() as u32;
        let (
            prefill_text,
            prefill_tokens,
            user_content_start,
            user_content_end,
            assistant_content_start,
        ) = if head_len == 0 {
            (
                prefill_text,
                prefill_tokens,
                user_content_start,
                user_content_end,
                assistant_content_start,
            )
        } else {
            let mut baked = head;
            baked.extend_from_slice(&prefill_tokens);
            (
                format!("{}{}", self.turn_head_text(no_think), prefill_text),
                baked,
                // The user body starts past the marker; the two interior
                // boundaries shift with it. The seal's `from_flat_grid`
                // reads a non-zero `user_content_start` as "the grid
                // reserves room for the opener, so bake it real".
                user_content_start + head_len,
                user_content_end + head_len,
                assistant_content_start + head_len,
            )
        };
        // The spans move with the user body.
        let recurrent_spans: Vec<TokenSpan> = recurrent_spans
            .into_iter()
            .map(|s| TokenSpan {
                tokens: s.tokens.start + head_len as usize..s.tokens.end + head_len as usize,
                kind: s.kind,
            })
            .collect();
        let image_spans: Vec<ImageSpan> = image_spans
            .iter()
            .map(|s| s.shifted(head_len as usize))
            .collect();
        // The closing marker rides the post-decode tail, so it lands after the
        // decoded body and inside the sealed grid.
        let mut post_decode_tokens = post_decode_tokens;
        post_decode_tokens.extend_from_slice(&self.tokenize(self.config.dialect.assistant_end)?);

        // A replayed turn decodes its recording: everything the recorded grid
        // holds past this prefill, less the tail written after the decode — one
        // step per recorded id and one for the end of turn. Its triggers and
        // grammar run as they did live: what they played is in the recording,
        // and where they cut pages is part of what is being replayed.
        let (reply, max_decode_tokens) = match recorded_turn {
            Some((grid, forced_tokens, forced_suffix)) => {
                let mut ids = recorded_reply(&grid, &prefill_tokens, &post_decode_tokens)
                    .map_err(ConversationError::Other)?;
                let steps = match forced_tokens {
                    Some(forced) => {
                        ids.truncate(forced);
                        ids.extend(forced_suffix);
                        max_decode_tokens
                    }
                    None => ids.len() + 1,
                };
                let free_after = forced_tokens.is_some();
                (Some(RecordedReply { ids, free_after }), steps)
            }
            None => (None, max_decode_tokens),
        };

        let disable_reprojection = self.config.disable_reprojection;
        // Append-only ingests skip the per-turn projection rebuild and also
        // suppress continuous mid-decode reprojection.
        let reprojection = if disable_reprojection {
            None
        } else {
            reprojection
        };
        let (event_tx, event_rx) = flume::unbounded();
        self.scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                // One turn, and it is the slot's tail — the ordinary shape.
                seal_group: None,
                sequence_id,
                projection_inputs,
                prefill_tokens,
                prefill_text,
                user_text,
                user_content_start,
                user_content_end,
                assistant_content_start,
                no_think,
                tags,
                projection_offsets,
                prefill_assistant_text,
                post_decode_tokens,
                max_decode_tokens,
                sampling,
                event_tx,
                reprojection,
                disable_reprojection,
                triggers,
                turn_grammar,
                free_tool_calls_from_penalties,
                recorded_reply: reply,
                recurrent_spans,
                image_spans,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        Ok(TurnHandle::new(event_rx))
    }

    /// The text a turn's grid opens with: `user_start`, plus the `/no_think`
    /// soft-switch when this turn's dial suppressed thinking.
    ///
    /// The switch sits **inside the turn that carries it**, which is what stops
    /// a past suppressed turn from leaking a stale switch onto a later
    /// thinking-on one: each turn's grid holds its own, decided once at submit
    /// from the dial that was live then, rather than re-decided from the current
    /// dial every projection. (Qwen3 only honours the switch from the user turn,
    /// so it has to follow the opener rather than sit in the system prompt.)
    pub(super) fn turn_head_text(&self, no_think: bool) -> String {
        let switch = if no_think {
            self.config.dialect.no_think
        } else {
            ""
        };
        format!("{}{}", self.config.dialect.user_start, switch)
    }

    /// [`Self::turn_head_text`] tokenised — tokenised as ONE string, so a
    /// tokenizer that merges across the `user_start`/`no_think` join produces
    /// the same ids the model would see from the concatenated text.
    pub(super) fn turn_head_tokens(&self, no_think: bool) -> crate::Result<TokenBuffer> {
        self.tokenize(&self.turn_head_text(no_think))
    }

    /// The canonical token sequence for `text`.
    ///
    /// The format is: `[no_think] user_text user_end asst_start asst_text turn_end user_start`
    /// This matches what is prefilled during a normal `submit_turn` / `finish_turn` cycle and
    /// is used to reconstruct the context window during the pre-submit phase.
    pub(super) fn tokenize(&self, text: &str) -> crate::Result<TokenBuffer> {
        self.tokenizer
            .encode(text, false)
            .map(|enc| TokenBuffer::from(enc.get_ids()))
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))
    }

    /// Encode a user half piece by piece, its literal pieces spelled out — see
    /// [`crate::turn_text`]. The ids a submitted turn's half becomes.
    pub fn encode_text(&self, text: &TurnText) -> crate::Result<TokenBuffer> {
        encode_pieces(&self.tokenizer, &self.literal_tokenizer, text)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))
    }
}
