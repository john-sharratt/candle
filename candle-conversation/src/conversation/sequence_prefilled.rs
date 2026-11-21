//! Prefills calibration turns from a supplied trajectory, so their key-value
//! state matches a decoded turn without the decode.
//!
//! The trajectory is laid down verbatim with its projection markers stripped, and
//! each marker's token offset is recorded so the prefill projects where a real
//! decode did. Nothing is suppressed or decoded, so the grid reproduces what was
//! decoded and nothing else. A grouped submission shares one projection and one
//! selection, which is why a group holds one tool's cases; each case still seals
//! as a turn of its own.

use super::*;

impl Sequence {
    /// Submit a calibration turn whose assistant trajectory is **supplied
    /// verbatim** and prefilled in a single batched forward pass instead of
    /// decoded — the fast path that reproduces a decode-built calibration turn's
    /// KV (and thus its seal-captured wide-Q signature) without the per-token
    /// decode cost.
    ///
    /// `assistant_trajectory` is the model's real reply body (`<think>…</think>`
    /// + `<tool_call>…`) exactly as a decode produced it — NOT think-suppressed
    /// like [`insert_turn_tagged`](Self::insert_turn_tagged). The grid is
    /// `user_message + user_end + assistant_start + assistant_trajectory`, byte-
    /// identical to the decode's persisted turn, so re-tokenizing it reproduces
    /// the same tokens (modulo rare non-canonical BPE runs). `max_decode_tokens =
    /// 0`, so the scheduler prefills then seals — `perform_seal_and_write`
    /// captures the whole turn's wide-Q from KV exactly as for a decoded turn.
    ///
    /// `selection` pins the tool (via the calibration projection's tool selector) so the projection
    /// marks exactly that catalog member selected; `tags` scope the belief
    /// gallery. Returns the streaming handle — drain it to `Done` and seal it
    /// with [`finish_turn`](Self::finish_turn) exactly like a decoded turn.
    pub fn submit_prefilled_turn(
        &mut self,
        user_message: impl Into<TurnText>,
        assistant_trajectory: &str,
        projection_marker: &str,
        selection: SelectionState,
        tags: Vec<String>,
    ) -> crate::Result<TurnHandle> {
        let user_message = user_message.into();
        let user_text = user_message.text();
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }
        self.sync_submitted_sections();
        self.selection = selection;
        self.publish_ask_state();

        // **No suppression block here, unlike `submit_turn_with_options`.** That
        // path prefills the dialect's closed block because the model is about to
        // decode and must be steered; this one is handed the whole assistant
        // trajectory verbatim, so the grid must reproduce what was decoded and
        // nothing else. The trajectory carries its own `<think>` in the body when
        // it had one, and injecting scaffolding in front of it would make the
        // replayed grid differ from the decode it is calibrating against.
        let assistant_start_marker = self.config.dialect.assistant_start;
        let markers = format!("{}{}", self.config.dialect.user_end, assistant_start_marker);
        let assistant_head = format!("{user_text}{markers}");
        // The trajectory carries `projection_marker`s at the points a real decode
        // reprojected. Strip them from the prefilled text (they are not model
        // tokens), and record each one's token offset so the staged prefill wave
        // fires a projection there — reproducing the decode's per-segment
        // projection sequence. The markers are token-aligned by construction (the
        // exporter verifies the split round-trips), so tokenizing the cumulative
        // prefix up to each marker yields its exact grid offset.
        let clean_trajectory = assistant_trajectory.replace(projection_marker, "");
        let formatted = format!("{assistant_head}{clean_trajectory}");
        // The user half is encoded piece by piece, exactly as
        // `submit_turn_with_options` encodes it — a tool's output stays literal —
        // and the markers and trajectory follow as one string. The half meets
        // them on `user_end`, a registered tag the tokenizer splits at anyway, so
        // the concatenation is the whole grid's encoding.
        let user_tokens = self.encode_text(&user_message)?;
        let mut prefill_tokens = user_tokens.clone();
        prefill_tokens.extend_from_slice(&self.tokenize(&format!("{markers}{clean_trajectory}"))?);
        // No post-decode tail — `assistant_end` is a live `Generated` segment,
        // same as a decoded turn.
        let post_decode_tokens = TokenBuffer::new();

        // Record the pending user turn (text + raw tokens for the tree).
        self.pending_user = Some(TokenizedText::new(user_text.as_str(), user_tokens.clone()));

        // Content boundaries: user body `[0, len(user half))`; the supplied
        // assistant trajectory begins right after the markers. Clamp/monotonise
        // so a tokenizer that merges across a join can never invert the windows.
        let total = prefill_tokens.len();
        let user_content_start = 0u32;
        let user_content_end = user_tokens.len().min(total) as u32;
        let assistant_content_start = (user_tokens.len() + self.tokenize(&markers)?.len())
            .min(total)
            .max(user_content_end as usize) as u32;

        // Grid-token offset of each projection marker: the user half's length
        // plus the tokenized `markers + trajectory up to the marker`. `split`
        // yields one segment per marker plus a trailing remainder, so the
        // boundaries between segments are exactly the marker positions. Two
        // markers are dropped from the wave's emit list: the initial one at
        // generation start (index 0 — the handler already applied that
        // projection, and its span is carried by the next event) and the seal
        // marker flush with the trajectory end (the final projection is appended
        // at seal). The intermediate reprojections remain.
        let segments: Vec<&str> = assistant_trajectory.split(projection_marker).collect();
        let mut projection_offsets: Vec<u32> = Vec::new();
        if segments.len() > 1 {
            let end = prefill_tokens.len() as u32;
            let mut prefix = markers.clone();
            for (i, seg) in segments[..segments.len() - 1].iter().enumerate() {
                prefix.push_str(seg);
                let off = (user_tokens.len() + self.tokenize(&prefix)?.len()) as u32;
                if i > 0 && off < end {
                    projection_offsets.push(off);
                }
            }
        }

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
            // Thinking calibration turn: the trajectory carries its own `<think>`.
            false,
            tags,
            projection_offsets,
            // Prefill path: the assistant half is supplied — stored verbatim
            // (markers stripped) as the turn's assistant_text.
            clean_trajectory,
            post_decode_tokens,
            // No decode — prefill then seal.
            0,
            self.config.sampling.clone(),
            // No reprojection policy: there is no decode loop to trigger it.
            None,
            // No tool stencils on a calibration prefill, and no grammar to
            // begin inside — the trajectory is supplied, not decoded.
            Arc::new(TriggerRegistry::new()),
            None,
            // Nothing is sampled, so no penalty applies to exempt from.
            false,
            // Nothing is decoded, so there is no reply to replay.
            None,
            Vec::new(),
        )?;
        self.turn_in_flight = true;
        Ok(handle)
    }

    /// Prefill MANY turns together, each as if it were alone, and seal each as
    /// a turn of its own.
    ///
    /// The batched form of [`Self::submit_prefilled_turn`]. Each case is a
    /// prefilled turn — a user half and an assistant half, either of which may
    /// be empty. The scheduler projects the conversation once and prefills every
    /// case on its own view of that projection, all of them in the same waves
    /// ([`crate::turn_group`]): a case attends to the projection and to itself,
    /// never to a sibling, and a hybrid's recurrent state starts every case from
    /// the projection's. Each sealed turn — its K/V, its `sign(Q)` window, its
    /// recurrent snapshot — is what a lone [`Self::submit_prefilled_turn`] of
    /// that case would have sealed.
    ///
    /// **Why.** One case per forward runs the model deep in the launch-overhead
    /// regime: 353 tokens across 2.4 sequences in a 1,095 ms forward, 263 t/s,
    /// with a thousand-token backlog queued behind it. The forward costs about
    /// the same for 3,500 tokens as for 350.
    ///
    /// **A group shares one projection and one `SelectionState`, so it must be
    /// a group of ONE tool's cases.** The projection event records a single
    /// selected catalog member for the whole submission; grouping a tool's own
    /// question exemplars makes that record correct for every case in it.
    /// Mixing tools in one group would leave the event naming one tool for all
    /// of them. Retrieval would mostly survive that — `belief_gallery` resolves
    /// an exemplar's slot from its turn's tags first and only falls back to
    /// projection events when the tags name none — but the persisted event
    /// trail would be a lie, so the constraint is stated rather than leaned on.
    ///
    /// `tags` is required per case rather than defaulted for the same reason: a
    /// case with empty tags falls through to the group's selection.
    ///
    /// Each case is `(user, assistant, tags)`: the user half, the assistant half
    /// (empty for a question exemplar, the prefilled body for a dream line), and
    /// that case's tags. Returns the streaming handle plus, for each sealed
    /// turn in order, the index of the case it came from — a case with no
    /// tokens on either half is not submitted, so the two are not 1:1.
    ///
    /// The handle hears the opening projection and one `Done` for the whole
    /// group, whose seal is the last case's — or, when any case failed to seal,
    /// one `Error` naming how many did, so a partial group is never taken for a
    /// complete one.
    pub fn submit_prefilled_turn_group(
        &mut self,
        cases: &[(String, String, Vec<String>)],
        selection: SelectionState,
    ) -> crate::Result<(TurnHandle, Vec<usize>)> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }
        self.sync_submitted_sections();
        self.selection = selection;
        self.publish_ask_state();

        // Each case's grid is the whole turn a lone prefill would lay down —
        // opener, user body, the user/assistant join, and the closing marker —
        // because each case is sealed as a complete turn and must carry its own
        // brackets. `submit_prefill_unit` bakes those ends for a single turn;
        // here every case needs its own pair, so they are built directly.
        let head = self.turn_head_text(false);
        let user_end = self.config.dialect.user_end;
        let assistant_start = self.config.dialect.assistant_start;
        let assistant_end = self.config.dialect.assistant_end;

        let mut grids: Vec<CaseGrid> = Vec::with_capacity(cases.len());
        for (question, answer, _) in cases {
            let user_prefix = format!("{head}{question}");
            let assistant_head = format!("{user_prefix}{user_end}{assistant_start}");
            // The assistant body sits between its opener and closer. Empty for a
            // question exemplar (the routing is on the question); the prefilled
            // line for a dream. Its token span is measured below as the run past
            // `assistant_head`.
            let whole = format!("{assistant_head}{answer}{assistant_end}");
            // Tokenised as cumulative prefixes of ONE string, so a tokenizer
            // that merges across a join reports bounds on the same ids the model
            // will see. Measuring the pieces separately and summing would drift
            // wherever a merge crosses a boundary.
            let tokens = self.tokenize(&whole)?;
            let user_content_end = self.tokenize(&user_prefix)?.len().min(tokens.len()) as u32;
            let assistant_content_start =
                self.tokenize(&assistant_head)?.len().min(tokens.len()) as u32;
            grids.push(CaseGrid {
                tokens: tokens[..].to_vec(),
                user_content_end,
                assistant_content_start,
            });
        }

        let planned = plan_cases(&grids);
        if planned.is_empty() {
            return Err(ConversationError::Channel(
                "submit_prefilled_turn_group: no case produced any tokens".into(),
            ));
        }

        // One `TurnContent` per case: what the substrate pins on the turn it
        // becomes, and — its `token_ids` — the whole of what the case prefills.
        let im_end_len = self.tokenize(user_end)?.len() as u32;
        let assistant_start_len = self.tokenize(assistant_start)?.len() as u32;
        let trailing_len = self.tokenize(assistant_end)?.len() as u32;
        // The opener is the same text for every case, so it is measured once
        // rather than per case; `PlannedCase::layout` clamps it against each
        // case's user end. The opener is baked into every case's grid, so the
        // user body starts past it — the non-zero start is what tells the layout
        // the grid reserves room for a real opener.
        let head_len = self.tokenize(&head)?.len() as u32;
        let mut turns: Vec<TurnContent> = Vec::with_capacity(planned.len());
        let mut sources: Vec<usize> = Vec::with_capacity(planned.len());
        for (src, case) in planned {
            let (question, answer, tags) = &cases[src];
            turns.push(TurnContent {
                role: Role::User,
                tags: tags.clone(),
                layout: case.layout(
                    head_len,
                    im_end_len,
                    assistant_start_len,
                    trailing_len,
                    question.clone(),
                    answer.clone(),
                ),
                token_ids: TokenBuffer::from(std::mem::take(&mut grids[src].tokens)),
            });
            sources.push(src);
        }

        let (event_tx, event_rx) = flume::unbounded();
        let inputs = self.projection_inputs();
        self.scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                sequence_id: self.id,
                projection_inputs: Some(inputs),
                prefill_text: String::new(),
                // Each case prefills its own tokens, on its own view.
                prefill_tokens: TokenBuffer::new(),
                user_text: String::new(),
                // Every case's openers and closers are baked into its own
                // tokens, so the single-turn boundary baking must not run.
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                tags: Vec::new(),
                projection_offsets: Vec::new(),
                prefill_assistant_text: String::new(),
                // Every case closed its own bracket inside its own tokens.
                post_decode_tokens: TokenBuffer::new(),
                max_decode_tokens: 0,
                sampling: self.config.sampling.clone(),
                event_tx,
                reprojection: None,
                disable_reprojection: self.config.disable_reprojection,
                triggers: Arc::new(TriggerRegistry::new()),
                // Every case's assistant half is supplied in its tokens, so
                // there is nothing to constrain and nothing to exempt.
                turn_grammar: None,
                free_tool_calls_from_penalties: false,
                recorded_reply: None,
                recurrent_spans: Vec::new(),
                seal_group: Some(turns),
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        self.turn_in_flight = true;
        Ok((TurnHandle::new(event_rx), sources))
    }
}
