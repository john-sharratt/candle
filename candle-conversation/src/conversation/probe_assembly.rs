//! Frames a turn for the projection panel and runs the interactive projection
//! probe without holding the fork source's lock.
//!
//! The panel renders the glue between sections and turns from the dialect's own
//! functions, so what it shows is what the turn carries. The probe is a full GPU
//! round-trip, so `ProbeCtx` is a lock-free snapshot of the handles it needs. It
//! scores the whole turn's wide-Q read-only and persists nothing, because a probe
//! is a question and not evidence about a scope.

use super::*;

/// Returned by [`Sequence::glue_markers`] so the projection panel can render the
/// framing verbatim.
#[derive(Debug, Clone)]
pub struct GlueMarkers {
    pub system_start: String,
    pub system_end: String,
    pub user_start: String,
    pub user_end: String,
    pub assistant_start: String,
    pub assistant_end: String,
    /// The `/no_think` soft-switch, emitted as live glue right after `user_start`
    /// on a suppressed (effort-off) turn — see the scheduler's `no_think_current`
    /// segment. `Dialect::thinking_suppression`'s first half; empty on a family
    /// that suppresses with [`Self::no_think_block`] instead.
    pub no_think: String,
    /// The already-closed reasoning block prefilled straight after
    /// `assistant_start` on a suppressed turn — `thinking_suppression`'s second
    /// half, and the mechanism for a family with no soft switch (Qwen3.5/3.8).
    ///
    /// Carried here for the same reason [`Self::no_think`] is: the panel renders
    /// the framing verbatim, and this is framing. It is prefilled into the grid,
    /// not decoded by the model, so a panel that omitted it showed less than the
    /// turn actually contains.
    ///
    /// **Exactly one of the two is ever non-empty** — see
    /// [`GlueMarkers::from_dialect`], which is the only constructor and takes
    /// both from the one function that owns the split.
    pub no_think_block: String,
}

impl GlueMarkers {
    /// Read every marker off `dialect`, taking the two suppression halves from
    /// [`Dialect::thinking_suppression`] rather than from the fields directly.
    ///
    /// Going through that function is what keeps the panel honest: it owns the
    /// "exactly one mechanism is live" rule, so a family that suppresses with the
    /// prefilled block cannot be rendered as if it used a `/no_think` switch it
    /// does not have, and neither half can be forgotten the way the block half
    /// once was.
    pub fn from_dialect(dialect: &Dialect) -> Self {
        // `true` asks for the suppressed shape — this is a description of the
        // framing a suppressed turn *would* carry, not a claim that the current
        // turn is suppressed. The panel labels it; the dial decides it.
        let (switch, block) = dialect.thinking_suppression(true);
        Self {
            system_start: dialect.system_start.to_string(),
            system_end: dialect.system_end.to_string(),
            user_start: dialect.user_start.to_string(),
            user_end: dialect.user_end.to_string(),
            assistant_start: dialect.assistant_start.to_string(),
            assistant_end: dialect.assistant_end.to_string(),
            no_think: switch.to_string(),
            no_think_block: block.to_string(),
        }
    }
}

impl Sequence {
    /// Snapshot the handles the interactive projection probe needs into a
    /// lock-free [`ProbeCtx`]. Cheap (Arc/handle clones), so the caller drops the
    /// `base_conv` mutex immediately and runs the GPU-bound [`ProbeCtx::probe`]
    /// unlocked — the probe (a full turn round-trip) must not hold the fork source
    /// every new conversation and several status reads contend on.
    pub fn probe_ctx(&self) -> ProbeCtx {
        ProbeCtx {
            scheduler_tx: self.scheduler_tx.clone(),
            substrate: self.substrate.clone(),
            projection: Arc::clone(&self.projection),
            target: self.target,
            tokenizer: Arc::clone(&self.tokenizer),
            selection: self.selection.clone(),
            dialect: self.config.dialect.clone(),
        }
    }

    /// `(section name, authored content)` for every system-prompt section in the
    /// schema (bare sections + every collection member). Backs the projection
    /// panel's expandable section text — fetched on demand, never stored in the
    /// projection event.
    pub fn section_contents(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for item in &self.projection.schema().system_prompt.items {
            match item {
                crate::projection::SystemPromptItem::Section(s) => {
                    out.push((s.name.clone(), s.content.clone()));
                }
                crate::projection::SystemPromptItem::Collection(c) => {
                    for s in &c.sections {
                        out.push((s.name.clone(), s.content.clone()));
                    }
                }
                crate::projection::SystemPromptItem::SectionTree(t) => {
                    // One entry per (node, option): name `node` or `node:option`
                    // for selector options, so the panel can resolve each.
                    for n in &t.nodes {
                        for o in &n.options {
                            let name = if n.options.len() > 1 {
                                format!("{}:{}", n.name, o.id)
                            } else {
                                n.name.clone()
                            };
                            out.push((name, o.content.clone()));
                        }
                        // An embedded collection node lists its members.
                        if let Some(tc) = &n.collection {
                            for s in &tc.collection.sections {
                                out.push((s.name.clone(), s.content.clone()));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The conversation's default sampling config (with the model's resolved
    /// thinking / reflection-marker token IDs).  Callers clone this to derive a
    /// per-turn config — e.g. to set the dial's `segment_suppress_penalty` — and
    /// pass it back via `TurnOptions::sampling`.
    pub fn default_sampling(&self) -> SamplingConfig {
        self.config.sampling.clone()
    }

    /// The dialect's framing markers — the "glue" the assembler wraps around the
    /// system prompt and each turn (`BoundaryMarkers` tokenises exactly these).
    /// Surfaced verbatim so the projection panel can show the glue between
    /// sections/turns without re-tokenising or re-projecting.
    pub fn glue_markers(&self) -> GlueMarkers {
        GlueMarkers::from_dialect(&self.config.dialect)
    }

    /// The YAML name of this conversation's target layer (e.g. `dialogue`) — the
    /// memory tier the dialogue exchange itself lives in. The panel prefixes the
    /// conversation messages with it, derived from the schema rather than guessed.
    pub fn target_layer_name(&self) -> String {
        self.projection
            .schema()
            .layers
            .iter()
            .find(|l| l.id == self.target.layer)
            .map(|l| l.name.clone())
            .unwrap_or_default()
    }
}

/// A lock-free snapshot of the handles the interactive projection probe needs
/// ([`Sequence::probe_ctx`]), so the probe — a full GPU turn round-trip — runs
/// WITHOUT holding the `base_conv` mutex, which is the fork source every new
/// conversation and several status reads contend on.
pub struct ProbeCtx {
    pub(super) scheduler_tx: Sender<SchedulerRequest>,
    pub(super) substrate: Conversation,
    pub(super) projection: Arc<Builder>,
    pub(super) target: ProjectionTarget,
    pub(super) tokenizer: Arc<tokenizers::Tokenizer>,
    pub(super) selection: SelectionState,
    pub(super) dialect: Dialect,
}

/// What [`ProbeCtx::probe`] scored: the projection it selects, and the turn's
/// wide-Q — the prefilled turn's `question` signatures first, then one for
/// each token the model decoded after it.
pub struct Probed {
    pub event: ProjectionEvent,
    pub sigs: Vec<WideQSig>,
    /// Signatures of the prefilled turn — its opener, the question and the
    /// assistant header — ahead of the decoded ones.
    pub question: usize,
}

impl ProbeCtx {
    /// Project a typed `text` against the current substrate as if it were a new
    /// turn under this conversation's system prompt — the interactive
    /// "what would this query retrieve" probe. Warms the query under the full
    /// system prompt as a thinking turn, decodes `decode` tokens of the model's
    /// answer (at least one, so the completion path gathers the wide-Q), scores
    /// the whole turn's wide-Q read-only (`observe = false`), and returns the
    /// resulting [`ProjectionEvent`] (selected turns + sections with belief
    /// scores) plus the probe it scored, one signature a token, which a harness
    /// scores again offline under formulas of its own. Persists NOTHING to the
    /// substrate.
    pub fn probe(&self, text: &str, decode: usize) -> crate::Result<Probed> {
        let (probe, question) = self.probe_wide_q(text, decode.max(1))?;
        // The typed probe IS the question, so a separate Q-window would be
        // identical to the tail scan — `None` scans it once.
        let (scores, _) = self.substrate.score_beliefs(
            self.projection.schema(),
            self.target,
            &probe,
            None,
            // A probe writes nothing to the substrate and must teach nothing
            // either: it is a question being asked, not evidence about a scope.
            Observe::No,
            None,
        );
        let resolver = self.substrate.read_for_scored(self.target, &scores);
        let projection = self.projection.project_with_selection(
            self.target,
            &resolver,
            ProjectionMode::Prefill,
            &self.selection,
        );
        let substrate_total = resolver.total_token_count(self.target.timeline) as u32;
        let mut ev = from_projection_with_origins(
            &projection.segments,
            &projection.selection_origins,
            self.projection.schema(),
            &resolver,
            &projection.selection_scores,
            substrate_total,
            0,
            0.0,
        );
        ev.pieces = projection.fill;
        ev.unpooled = projection.unpooled;
        Ok(Probed {
            event: ev,
            sigs: probe,
            question,
        })
    }

    /// Capture the WARM folded wide-Q signature of `text` as if it were a new turn
    /// under the system prompt, WITHOUT writing anything to the substrate. An
    /// ephemeral projection slot materializes the full system prompt (warming the
    /// query's Q exactly like a real turn), prefills the turn ([`Self::turn_tokens`])
    /// and decodes `decode` tokens of its answer, so the completion path finalizes
    /// the KV and gathers the turn's wide-Q; the slot resolves to
    /// `SealAction::None`, so no turn is sealed and no belief learned. Returns the
    /// signatures and how many of them are the prefilled turn's.
    pub(super) fn probe_wide_q(
        &self,
        text: &str,
        decode: usize,
    ) -> crate::Result<(Vec<WideQSig>, usize)> {
        let query_tokens = probe_turn_tokens(&self.tokenizer, &self.dialect, text)?;
        if query_tokens.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let question = query_tokens.len();

        // Ephemeral slot bound to this conversation's target: it projects (warm)
        // but never seals.
        let slot = {
            let (tx, rx) = flume::bounded(1);
            self.scheduler_tx
                .send(SchedulerRequest::NewEphemeralSequence {
                    conversation: self.substrate.clone(),
                    target: self.target,
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            replied(&rx).map_err(|_| ConversationError::SchedulerGone)??
        };
        let free = |tx: &Sender<SchedulerRequest>| {
            let _ = tx.send(SchedulerRequest::FreeSequence { sequence_id: slot });
        };

        // Submit the query as a projected turn (full warm system prompt via
        // `apply_projection`), decoding one token so the completion path finalizes
        // the KV onto the slot and gathers the query's warm wide-Q.
        let (event_tx, event_rx) = flume::unbounded();
        if self
            .scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                // An ephemeral probe seals nothing at all — see the
                // `ephemeral_slots` branch that forces `SealAction::None`.
                seal_group: None,
                sequence_id: slot,
                // Re-projects (`disable_reprojection: false`): no frame.
                projection_inputs: Some(ProjectionInputs {
                    projection: Arc::clone(&self.projection),
                    selection: self.selection.clone(),
                    frame: None,
                }),
                prefill_tokens: TokenBuffer::from(query_tokens),
                prefill_text: String::new(),
                user_text: String::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                tags: Vec::new(),
                projection_offsets: Vec::new(),
                prefill_assistant_text: String::new(),
                post_decode_tokens: TokenBuffer::new(),
                max_decode_tokens: decode,
                sampling: SamplingConfig::argmax(),
                event_tx,
                reprojection: None,
                disable_reprojection: false,
                triggers: Arc::new(TriggerRegistry::new()),
                // A probe's answer is free thinking: nothing constrains it.
                turn_grammar: None,
                free_tool_calls_from_penalties: false,
                recorded_reply: None,
                recurrent_spans: Vec::new(),
                image_spans: Vec::new(),
            })
            .is_err()
        {
            free(&self.scheduler_tx);
            return Err(ConversationError::SchedulerGone);
        }

        // Drain to Done/Error. A channel closed before `Done` (the turn was aborted
        // without a terminal event) is a failure — surface it rather than returning
        // an empty window that reads as "query retrieved nothing".
        let mut err: Option<ConversationError> = None;
        let mut done = false;
        while let Ok(ev) = event_rx.recv() {
            match ev {
                TurnEvent::Done(_) => {
                    done = true;
                    break;
                }
                TurnEvent::Error(e) => {
                    err = Some(e);
                    break;
                }
                _ => {}
            }
        }
        if let Some(e) = err {
            free(&self.scheduler_tx);
            return Err(e);
        }
        if !done {
            free(&self.scheduler_tx);
            return Err(ConversationError::SchedulerGone);
        }

        // Drain the warm wide-Q the ephemeral turn stashed, then free.
        let (tx, rx) = flume::bounded(1);
        let sigs = if self
            .scheduler_tx
            .send(SchedulerRequest::ProbeWideSigs {
                sequence_id: slot,
                response_tx: tx,
            })
            .is_ok()
        {
            rx.recv()
                .map_err(|_| ConversationError::SchedulerGone)
                .and_then(|r| r)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        free(&self.scheduler_tx);

        // Keep the WHOLE turn window `[query .. decoded]`, not just the query — the
        // discriminating belief signal for a tool lives in the DECODE-Q (the token
        // where the model commits to the call), not the query's prefill-Q, which
        // sits near the noise floor (the call↔definition domain gap). Dropping the
        // decoded tail collapsed scores ~10× (600 → 60).
        Ok((sigs, question))
    }
}

/// The assistant half of a turn's prefill, after its role marker: the dialect's
/// closed reasoning block when the turn suppresses thinking that way, then the
/// caller's own seed (a `<tool_call>` opener, say), which continues from after
/// it.
///
/// **This is what makes thinking-off structural on Qwen3.5/3.8.** Those families
/// have no `/no_think` soft switch, so `closed_think` — the second half of
/// `Dialect::thinking_suppression` — is the whole of their suppression. Composed
/// from the seed alone, a suppressed turn prefills no block while
/// `assistant_content_start` still measures past one: thinking-off is a line of
/// prose again, and the content span claims tokens the grid does not hold.
pub(super) fn assistant_lead(closed_think: &str, assistant_prefill: &str) -> String {
    format!("{closed_think}{assistant_prefill}")
}

/// The thinking turn `text` opens, as a dialogue turn prefills it:
/// `user_start`, the question, `user_end`, `assistant_start` and the
/// dialect's thinking-on lead — what [`ProbeCtx::probe`] prefills. Empty
/// when `text` encodes to nothing.
pub(super) fn probe_turn_tokens(
    tokenizer: &tokenizers::Tokenizer,
    dialect: &Dialect,
    text: &str,
) -> crate::Result<Vec<u32>> {
    let encode = |s: &str| {
        tokenizer
            .encode(s, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))
    };
    let body = encode(text)?;
    if body.is_empty() {
        return Ok(Vec::new());
    }
    let lead = assistant_lead(dialect.thinking_suppression(false).1, "");
    let mut tokens = encode(dialect.user_start)?;
    tokens.extend(body);
    tokens.extend(encode(&format!(
        "{}{}{lead}",
        dialect.user_end, dialect.assistant_start
    ))?);
    Ok(tokens)
}
