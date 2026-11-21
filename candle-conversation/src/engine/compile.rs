//! Compiles the grammars that constrain a model's tool calls and thinking blocks.
//!
//! A tool catalog compiles once into a trigger registry bound to the
//! `<tool_call>` marker. Each turn holds that catalog to the tools it was given,
//! so a call can name only a tool the turn offered. The thinking steering tree is
//! compiled once per effort dial, and each turn binds the think trigger to its
//! dial's tree, so suppression is part of the grammar rather than of the chat
//! template. Both stay inactive when the tokenizer has no single marker token.

use super::*;

/// The compiled thinking-block steering trees, one per effort dial, built once
/// at engine init (parallel to the tool-call registry).  Each turn derives its
/// trigger registry by binding the `<think>` trigger to the dial's tree, once —
/// see [`TriggerRegistry::with_once_trigger`].
pub struct ThinkSteering {
    /// `<think>` id — the trigger the dial's tree is bound to.
    pub(super) think_open: TokenId,
    /// Empties the block the moment it opens — see [`crate::stencil::think`].
    pub(super) off: Arc<StencilTree>,
    pub(super) quick: Arc<StencilTree>,
    pub(super) balanced: Arc<StencilTree>,
    pub(super) deep: Arc<StencilTree>,
    pub(super) exhaustive: Arc<StencilTree>,
}

impl ThinkSteering {
    /// Derive a per-turn registry from `base` (e.g. the tool-call catalog) for
    /// `mode`: bind the `<think>` trigger to that dial's steering tree.
    /// The base is untouched; the result is a fresh registry.
    ///
    /// **[`ThinkMode::Off`] binds a tree too**, and this is the correction that
    /// matters. It used to *clear* the trigger on the reasoning that the
    /// `/no_think` glue would yield an empty block — true for Qwen3, false for
    /// Qwen3.5 and Qwen3.8, which have no such marker at all. On those families
    /// clearing the trigger left the model free to reason with nothing steering
    /// it: the block ran to the token ceiling and the whole decode was discarded,
    /// while the dial reported itself off.
    ///
    /// Binding `off` instead makes suppression a property of the grammar rather
    /// than of the family's chat template, so it holds for every checkpoint and
    /// a caller does not have to know which mechanism its model happens to use.
    ///
    /// **A registry has to actually be bound for any of that to hold.** A caller
    /// that passes `TurnOptions::default()` gets an empty one and no steering at
    /// all — which is how the ingest summariser came to reason unchecked despite
    /// its dial reading `Off`. See `Sequence::no_think_triggers`.
    ///
    /// **The tree fires once per turn.** It steers the block that opens the
    /// turn; a `<think>` the model writes after that is text and decodes freely
    /// (see [`TriggerRegistry::with_once_trigger`]). The base's triggers — the
    /// tool call — stay armed for every call the turn makes.
    pub fn registry_for(&self, base: &TriggerRegistry, mode: ThinkMode) -> Arc<TriggerRegistry> {
        let tree = match mode {
            ThinkMode::Off => &self.off,
            ThinkMode::Quick => &self.quick,
            ThinkMode::Balanced => &self.balanced,
            ThinkMode::Deep => &self.deep,
            ThinkMode::Exhaustive => &self.exhaustive,
        };
        Arc::new(base.with_once_trigger(self.think_open, Arc::clone(tree)))
    }
}

impl ConversationEngine {
    /// Compile a tool catalog into a [`TriggerRegistry`] for constrained
    /// tool-call decoding.  Pass the returned registry to a turn via
    /// [`TurnOptions::triggers`](crate::TurnOptions::triggers); a turn without it
    /// (or with an empty registry) free-decodes as usual.
    ///
    /// The model emits the `<tool_call>` trigger token freely; the stencil then
    /// forces the catalog's exact shape — name ∈ catalog, required params in
    /// order, enum values exact, structurally-valid JSON — for the rest of the
    /// call.  Compile once and reuse the registry across turns.
    ///
    /// If the tokenizer has no single `<tool_call>` token (the marker tokenizes
    /// to several pieces), an **empty** registry is returned — constrained
    /// decoding is simply inactive and the model free-decodes tool calls as
    /// before.  This never fails startup over a tokenizer mismatch.
    /// This engine's vocabulary for a stencil compile: its tokenizer, its end
    /// tokens and its size, reading token bytes from the index the engine
    /// builds once.
    pub(super) fn vocab(&self) -> HfVocab {
        let tok = &self.core.tokenizer;
        let bytes = self
            .core
            .token_bytes
            .get_or_init(|| Arc::new(HfVocab::token_bytes_of(tok)));
        HfVocab::new(
            (**tok).clone(),
            &self.core.eos_tokens,
            self.core.vocab_size as u64,
        )
        .with_bytes(Arc::clone(bytes))
    }

    /// Compile a stencil spec against **this engine's** vocabulary.
    ///
    /// The vocabulary is the tokenizer, the EOS id and the vocab size together,
    /// and all three belong to the loaded checkpoint — so a caller that built
    /// its own would be compiling a grammar against a model it is not running.
    /// Exposed for callers with a tree of their own: `npcd` compiles an action
    /// loop rather than the single-call assistant shape
    /// [`Self::compile_tool_stencil`] builds.
    pub fn compile_stencil(&self, spec: &crate::stencil::TreeSpec) -> crate::Result<StencilTree> {
        let vocab = self.vocab();
        compile(spec, &vocab)
            .map_err(|e| ConversationError::from(candle::Error::Msg(format!("stencil: {e}"))))
    }

    /// The `<tool_call>` trigger bound to a call loop over `tools` of at most
    /// `max_calls` calls a reply — `MAX_TOOL_CALLS_PER_TURN` for a dialogue,
    /// one for a file's reading, whose every reply is one page
    /// (`docs/hit_driven_projection.md` §8.2).
    pub fn compile_tool_stencil(
        &self,
        tools: &[ToolSpec],
        max_calls: usize,
    ) -> crate::Result<Arc<TriggerRegistry>> {
        let Some(trigger) = self.core.tokenizer.token_to_id("<tool_call>") else {
            tracing::warn!(
                "tokenizer has no single <tool_call> token — tool-call stencils are inactive \
                 (the model will free-decode tool calls)"
            );
            return Ok(Arc::new(TriggerRegistry::new()));
        };
        let mut registry = TriggerRegistry::new();
        registry.register(trigger, Arc::new(self.call_loop(tools, max_calls)?));
        Ok(Arc::new(registry))
    }

    /// The opening call of a turn that must call one of `tools` before it
    /// does anything else ([`OpeningCall`]): the marker to prefill, and the
    /// same call loop [`Self::compile_tool_stencil`] binds to the trigger,
    /// over `tools`, of at most `max_calls` calls. `None` when the tokenizer
    /// has no single `<tool_call>` token, as there is then no trigger whose
    /// loop to begin inside.
    pub fn opening_call(
        &self,
        tools: &[ToolSpec],
        max_calls: usize,
    ) -> crate::Result<Option<OpeningCall>> {
        if self.core.tokenizer.token_to_id("<tool_call>").is_none() {
            return Ok(None);
        }
        Ok(Some(OpeningCall {
            marker: ToolCallEnvelope::for_dialect(&self.core.dialect).marker,
            tree: Arc::new(self.call_loop(tools, max_calls)?),
        }))
    }

    /// The call loop over `tools` the `<tool_call>` trigger enters, ending the
    /// turn after `max_calls` calls.
    pub(super) fn call_loop(
        &self,
        tools: &[ToolSpec],
        max_calls: usize,
    ) -> crate::Result<StencilTree> {
        // The model emits the `<tool_call>` trigger itself, so the tree resumes
        // *after* that marker: its `open` is the envelope minus the marker.
        //
        // **A turn may make up to `max_calls` calls.** After each
        // one the loop offers exactly two continuations — the marker, which opens
        // another call, or the assistant-turn terminator, which ends the turn — so
        // the decoder is never free between a call and whatever follows it. That
        // is the same guarantee the single-call tree gave by baking the EOS into
        // its close (without which the model free-decodes a hallucinated answer
        // past the call), now expressed as the loop's second arm rather than as an
        // unconditional ending. The decode loop still detects the EOS in the
        // injected close run and seals the turn.
        //
        // The batching is what this buys: a model that already knows it wants
        // three files says so in one turn instead of paying a reasoning block, a
        // prefill and a belief scan for each.
        //
        // **Taken from the dialect, not written here.** The shape of a call is
        // decided by the template the weights were trained against — Qwen3.5
        // writes a nested function element, ChatML writes a JSON object — and a
        // literal in this function is a second opinion about that, free to
        // disagree with the checkpoint actually loaded.
        // The marker off the front and the turn terminator on the close, with
        // the close ending exactly ON that terminator — see
        // [`ToolCallEnvelope::for_assistant_turn`] for why the trailing newline
        // in `assistant_end` cannot be allowed to ride along.
        let envelope = ToolCallEnvelope::for_assistant_calls(&self.core.dialect);
        let close_turn = ToolCallEnvelope::turn_close(&self.core.dialect);
        let started = std::time::Instant::now();
        let spec =
            compile_tool_call_loop(tools, &envelope, max_calls, &close_turn).map_err(|e| {
                ConversationError::from(candle::Error::Msg(format!("tool stencil: {e}")))
            })?;
        let spec_ms = started.elapsed().as_millis() as u64;
        let vocab = self.vocab();
        let vocab_ms = started.elapsed().as_millis() as u64 - spec_ms;
        let tree = compile(&spec, &vocab).map_err(|e| {
            ConversationError::from(candle::Error::Msg(format!("tool stencil: {e}")))
        })?;
        tracing::info!(
            tools = tools.len(),
            spec_nodes = spec.nodes.len(),
            tree_nodes = tree.len(),
            spec_ms,
            vocab_ms,
            compile_ms = started.elapsed().as_millis() as u64 - spec_ms - vocab_ms,
            "tool stencil compiled"
        );
        Ok(tree)
    }

    /// `triggers` with every tool call held to `tools`: each name branch of
    /// the call loop narrowed to the arms those names spell
    /// ([`held_branches_first`]), so a call can write only a tool the turn was
    /// given. The compiled catalog is shared and untouched; the turn carries
    /// the held arms, a few tokens per tool per call level.
    ///
    /// Build it once for the conversation that runs under it, not per turn: it
    /// decodes the catalog's name arms once per call level. `triggers` without
    /// a tool call is returned as it is. A list naming no tool the catalog
    /// holds is an error — a turn given tools it can never call.
    ///
    /// `first_call`, when given as `(tool, min_reasoning)`, holds the turn's
    /// opening call to that one tool, the calls after it to `tools`, and the
    /// turn to at least one call after it ([`held_branches_first`]) — once the
    /// turn has reasoned for `min_reasoning` tokens. A call opened sooner is
    /// held to `tools` alone.
    pub fn holding_tools(
        &self,
        triggers: &Arc<TriggerRegistry>,
        tools: &[String],
        first_call: Option<(&str, u32)>,
    ) -> crate::Result<Arc<TriggerRegistry>> {
        let tokenizer = &self.core.tokenizer;
        let Some(trigger) = tokenizer.token_to_id("<tool_call>") else {
            return Ok(Arc::clone(triggers));
        };
        let Some(tree) = triggers.tree_for(trigger) else {
            return Ok(Arc::clone(triggers));
        };
        let decode = |t: &[TokenId]| tokenizer.decode(t, false).ok();
        let plain = held_branches(tree, tools, decode);
        if plain.is_empty() {
            return Err(ConversationError::from(candle::Error::Msg(format!(
                "tool stencil: none of {tools:?} is in the turn's catalog"
            ))));
        }
        // A first call held to one tool is held only once the turn has
        // reasoned for `min_reasoning` tokens; a call opened sooner is held
        // to the turn's tools alone.
        Ok(Arc::new(match first_call {
            Some((first, min_reasoning)) => {
                let held = held_branches_first(tree, tools, Some(first), decode);
                triggers.holding_after_reasoning(
                    trigger,
                    Arc::new(held),
                    min_reasoning,
                    Some(Arc::new(plain)),
                )
            }
            None => triggers.holding(trigger, Arc::new(plain)),
        }))
    }

    /// `triggers` with every tool call's keyed fields held to `holds`
    /// (`docs/stencil_tree.md` §9.2): a field whose values are known — a
    /// file's anchors, a part's lines — can be written only as one of them.
    /// Built for one round and dropped with it, each set's rule taken from
    /// `kept` when the round before built it ([`HeldRules`]). `triggers`
    /// without a tool call, or `holds` that build to nothing, come back as
    /// they are.
    pub fn holding_values(
        &self,
        triggers: &Arc<TriggerRegistry>,
        holds: &ValueHolds,
        kept: &mut HeldRules,
    ) -> Arc<TriggerRegistry> {
        let Some(held) = self.held_values(holds, kept) else {
            return Arc::clone(triggers);
        };
        self.triggers_holding(triggers, held)
    }

    /// `holds` built over this checkpoint's vocabulary, each set's rule taken
    /// from `kept` when the round before built it — what a turn that begins
    /// inside a grammar is held to
    /// ([`TurnGrammar::holding`](crate::stencil::TurnGrammar::holding)). `None`
    /// when they build to nothing.
    pub fn held_values(&self, holds: &ValueHolds, kept: &mut HeldRules) -> Option<Arc<HeldValues>> {
        if holds.fields.is_empty() {
            return None;
        }
        let held = HeldValues::build(holds, &self.vocab(), kept);
        (!held.is_empty()).then(|| Arc::new(held))
    }

    /// `triggers` with every tool call's keyed fields held to `held`.
    /// `triggers` without a tool call come back as they are.
    pub fn triggers_holding(
        &self,
        triggers: &Arc<TriggerRegistry>,
        held: Arc<HeldValues>,
    ) -> Arc<TriggerRegistry> {
        let Some(trigger) = self.core.tokenizer.token_to_id("<tool_call>") else {
            return Arc::clone(triggers);
        };
        Arc::new(triggers.holding_values(trigger, held))
    }

    /// `triggers` with every turn's tool calls naming each of `tools` at most
    /// once ([`NeverTwice`]): once a call names one, no later call of the turn
    /// can. Applied to a compiled catalog, every registry held from it carries
    /// it. `triggers` without a tool call, or no tools, come back as they are.
    pub fn never_twice(
        &self,
        triggers: &Arc<TriggerRegistry>,
        tools: &[String],
    ) -> Arc<TriggerRegistry> {
        let Some(trigger) = self.core.tokenizer.token_to_id("<tool_call>") else {
            return Arc::clone(triggers);
        };
        let once = NeverTwice::new(tools);
        if once.is_empty() || triggers.tree_for(trigger).is_none() {
            return Arc::clone(triggers);
        }
        Arc::new(triggers.never_twice(trigger, Arc::new(once)))
    }

    /// `triggers` with an answer that ends written on into its `Steps:`
    /// section ([`compile_steps_tail`]): one line per step of `steps`, each
    /// held to an outcome, and the turn closed by the grammar. Fewer than two
    /// steps ask for no section, and `triggers` come back as they are, as
    /// they do when the tree cannot be built (logged).
    pub fn ending_with_steps(
        &self,
        triggers: &Arc<TriggerRegistry>,
        steps: &[String],
    ) -> Arc<TriggerRegistry> {
        if steps.len() < 2 {
            return Arc::clone(triggers);
        }
        let close_turn = ToolCallEnvelope::turn_close(&self.core.dialect);
        match compile(&compile_steps_tail(steps, &close_turn), &self.vocab()) {
            Ok(tree) => Arc::new(triggers.with_tail(Arc::new(tree))),
            Err(e) => {
                tracing::warn!("an answer's steps could not be built into its grammar: {e}");
                Arc::clone(triggers)
            }
        }
    }

    /// Compile the thinking-block steering trees (one per effort dial, `Off`
    /// included) once, for reuse across turns via
    /// [`ThinkSteering::registry_for`].  Like the tool
    /// stencil, this is inactive — `Ok(None)` — when the tokenizer lacks a
    /// single `<think>`/`</think>` token, so the model free-decodes its reasoning.
    ///
    /// **The tree hands control back to the decoder at `</think>`, and cannot do
    /// otherwise.** Emitting anything after the closing tag — a tool-call marker,
    /// to put a character that has finished thinking straight into a call — reads
    /// as a natural extension and breaks the index page cut. A static run's last
    /// token is deliberately held back from the forward and rides the next decode
    /// step, which commits it through `push_committed` and arms the cut; a marker
    /// in any earlier slot goes through `push_forwarded`, whose cut flag is
    /// dropped, and `run_prefill`'s `reasoning_split` declines to split when the
    /// break token is last in the pass. The block still records its
    /// `think_close_at`, so the turn seals with a reasoning span that is not a
    /// union of whole pages and every LATER turn fails
    /// `Substrate::turn_sealed_without_thinking`. See the `Off` arm in
    /// `stencil::think` for the measured case.
    pub fn compile_think_steering(&self) -> crate::Result<Option<Arc<ThinkSteering>>> {
        let (Some(think_open), Some(think_close)) = (
            self.core.tokenizer.token_to_id("<think>"),
            self.core.tokenizer.token_to_id("</think>"),
        ) else {
            tracing::warn!(
                "tokenizer has no single <think>/</think> token — think steering is inactive \
                 (the model will free-decode its reasoning)"
            );
            return Ok(None);
        };
        let eos = self.core.eos_tokens.iter().next().copied().unwrap_or(0);
        let env = ThinkSteerEnvelope {
            think_open,
            think_close,
            eos,
            // The assistant's reasoning is followed by prose, so control returns
            // to the decoder the moment the block closes.
            after_close: "",
        };
        // Every end token, so a think span intercepts whichever one the model
        // samples — the canonical `eos` above is only what the tree writes.
        let vocab = self.vocab();
        let compile_mode = |mode: ThinkMode| -> crate::Result<Arc<StencilTree>> {
            let spec = compile_think_tree(mode, &env);
            let tree = compile(&spec, &vocab).map_err(|e| {
                ConversationError::from(candle::Error::Msg(format!("think stencil: {e}")))
            })?;
            Ok(Arc::new(tree))
        };
        Ok(Some(Arc::new(ThinkSteering {
            think_open,
            off: compile_mode(ThinkMode::Off)?,
            quick: compile_mode(ThinkMode::Quick)?,
            balanced: compile_mode(ThinkMode::Balanced)?,
            deep: compile_mode(ThinkMode::Deep)?,
            exhaustive: compile_mode(ThinkMode::Exhaustive)?,
        })))
    }
}
