//! Frames a character's turns under the mind's schema and compiles the grammar
//! each turn is held to.
//!
//! A conversation opens against the schema with its acts and identities
//! installed, and its fingerprint covers everything that frames a turn, so a
//! change to the prompt, the acts or who the character is makes an old history
//! unrejoinable. Each turn is compiled as an action loop: it begins inside a
//! reasoning block when one is wanted, then takes calls up to a bound, and ends
//! only on the turn terminator. The call's syntax comes from the checkpoint's
//! dialect, so the grammar holds a model to the shape its weights were trained
//! on. A probe runs one throwaway turn and reports what it did, including
//! whether a reasoning block opened and whether it closed, since an unclosed
//! block looks the same as a character that chose to do nothing.

use super::*;

/// The schema a character's conversation opens against, with its acts and
/// identities already installed.
pub struct Projected {
    /// The schema's own static text — identical for the whole cast.
    pub prompt: String,
    pub builder: Builder,
    pub layer: LayerId,
    pub group: GroupId,
    /// Where a character's conversations are kept: the `talk` layer's group,
    /// when the schema declares one.
    pub talk: Option<(LayerId, GroupId)>,
    /// Which members exist, so a turn pins the right ones.
    pub identities: identity::Installed,
    /// The response templates each destination's words are shaped by.
    pub templates: templates::Installed,
    /// The fingerprint of everything a conversation under this schema is
    /// framed by — the authored YAML and every member installed into it. See
    /// [`crate::engine::schema::frame`] and [`frame_fingerprint`].
    ///
    /// Not a fingerprint of [`Self::prompt`]: that is only the text before the
    /// first collection, and a change to the acts, the frame for acting or who
    /// anybody is left it untouched, so a conversation written under the old
    /// prompt went on being rejoined under the new one.
    pub frame: String,
}

/// Everything one probe turn revealed.
///
/// Reported rather than judged: a probe says what happened and the caller
/// decides whether that was right, because "should a reasoning block have
/// opened" is a property of the mission being tested and not of the machinery.
#[derive(Debug, Serialize)]
pub struct Probe {
    /// The acts, the narration around them, and every call that was refused.
    pub parsed: Parsed,
    /// The decode verbatim. The only place the truth lives when a turn produces
    /// nothing: a character that chose to do nothing and one whose reasoning
    /// block never closed are identical everywhere else.
    pub raw: String,
    /// Whether a `<think>` opened, and whether it closed. **Both**, because an
    /// unterminated block is the failure — it runs to `max_response_tokens` and
    /// the whole decode is discarded as reasoning, with nothing logged, since a
    /// decode that reasons forever is a successful decode.
    pub opened_think: bool,
    pub closed_think: bool,
    /// How big the shared frame was, so a prompt that has quietly doubled shows
    /// up as a number rather than as a slow afternoon.
    pub prompt_bytes: usize,
    /// Which assembly ran: the mind's projection, or the rendered prompt.
    pub projected: bool,
    pub ms: u64,
}

/// Compile the act catalog into an **action-loop** grammar for a whole turn.
///
/// Returned as a bare tree, not a trigger registry: the turn is entered *inside*
/// this grammar, at the node following the marker its prefill already wrote.
/// See [`Minds::grammar_for`] for why a prefilled marker and a trigger on that
/// same marker cannot both work.
///
/// The character loop is not an assistant loop. `compile_tool_stencil` builds
/// the assistant shape — one call *is* the whole reply, and its close carries
/// the turn's EOS — which for a character would silently make every turn a
/// single act, with nothing reporting the amputation. Here the close leads to a
/// choice the model makes: act again, up to [`tools::ACTS_PER_TURN`], or end the
/// turn.
///
/// `close` deliberately carries no terminator; the finishing arm does. An
/// `env.close` ending in EOS would end the turn on the first call and the loop
/// would be unreachable.
pub(super) fn compile_act_loop(
    engine: &Arc<ConversationEngine>,
    cfg: &SequenceConfig,
    thinking: identity::Deliberation,
    catalog: Catalog,
    within: &tools::Within,
) -> anyhow::Result<Arc<StencilTree>> {
    let e = engine;
    let tok = e.tokenizer();
    let (Some(_call_open), Some(think_open), Some(think_close)) = (
        tok.token_to_id("<tool_call>"),
        tok.token_to_id("<think>"),
        tok.token_to_id("</think>"),
    ) else {
        anyhow::bail!(
            "this checkpoint's tokenizer lacks a single <tool_call>/<think> token, so a turn \
             cannot be forced into shape"
        );
    };

    // **Where the turn is entered decides what the tree has to cover.**
    //
    // Not deliberating, the prefill writes the already-closed block and the
    // marker, so the tree begins at the call body and the reasoning question
    // never arises. Deliberating, the prefill stops at `<think>` and the tree
    // covers the block *and* the calls after it — one grammar, so the join at
    // `</think>` is a node edge rather than a moment the decoder is free in.
    let prelude = match thinking {
        identity::Deliberation::None => None,
        _ => {
            let steer = ThinkSteerEnvelope {
                think_open,
                think_close,
                // The turn terminator, by the name the dialect gives it — the
                // spans end on either `</think>` or EOS, and an EOS the tree
                // does not know is one it cannot end a span on.
                eos: tok
                    .token_to_id(cfg.dialect.assistant_end.trim_end())
                    .unwrap_or(0),
                // Empty because this prelude is SPLICED onto the call grammar
                // below (`compile_action_loop(…, Some(&prelude))`), so the join
                // at `</think>` is already a node edge rather than a moment the
                // decoder is free in. Injecting the marker here as well would
                // emit it twice.
                after_close: "",
            };
            Some(compile_think_tree(thinking.mode(), &steer))
        }
    };
    let specs = act_specs(catalog, within);
    compile_calls(e, cfg, &specs, catalog.calls, prelude.as_ref())
}

/// The catalog the act grammar is compiled from, and so the catalog every
/// reader of a character's decode must hold: the same list, from the same
/// `Within` under the same mode, so the grammar and the reader cannot disagree
/// about where a function block's value ends.
pub(in super::super) fn act_specs(catalog: Catalog, within: &tools::Within) -> Vec<ToolSpec> {
    tools::specs_within(catalog.mode, within)
}

/// A grammar for a turn made of calls to `specs`, at most `max_calls` of them.
///
/// The shared tail of every call-forcing grammar: the character's act loop and
/// the single forced call that answers a question. Entered after the turn's
/// prefill has written the call marker.
pub(in super::super) fn compile_calls(
    e: &ConversationEngine,
    cfg: &SequenceConfig,
    specs: &[ToolSpec],
    max_calls: usize,
    prelude: Option<&TreeSpec>,
) -> anyhow::Result<Arc<StencilTree>> {
    // **The call's shape comes from the checkpoint's own dialect.**
    //
    // It was written out here, in Qwen3's JSON form, for every model this
    // daemon might ever load — so a checkpoint whose template says otherwise
    // was constrained to a syntax it was never trained on, and nothing said so
    // because a grammar always produces *something*. Asking the dialect makes
    // the shape a property of the weights, which is the only place that can
    // know it. See `candle_transformers::models::dialect::CallStyle`.
    let base = ToolCallEnvelope::for_dialect(&cfg.dialect);
    let env = ToolCallEnvelope {
        // The tree resumes *after* the marker, which the turn's prefill has
        // already written — so the walk starts here, at the first thing that was
        // ever actually in question.
        open: base
            .open
            .strip_prefix(&base.marker)
            .unwrap_or(&base.open)
            .to_string(),
        ..base
    };
    // The terminator ends the grammar ON the end-of-turn token: ChatML's
    // `assistant_end` trails a newline, which would bury the EOS one slot from
    // the end where the decode never sees it and the model writes its next turn.
    let close_turn = ToolCallEnvelope::turn_close(&cfg.dialect);
    let spec = compile_action_loop(specs, &env, max_calls, &close_turn, prelude)?;
    Ok(Arc::new(e.compile_stencil(&spec)?))
}

/// Every deliberation level, so a turn can pick its grammar without compiling
/// one. Five trees, built once — the alternative is compiling a stencil inside
/// the tick loop, which is a per-turn cost for a thing that never changes.
pub(super) const LEVELS: [identity::Deliberation; 5] = [
    identity::Deliberation::None,
    identity::Deliberation::Quick,
    identity::Deliberation::Balanced,
    identity::Deliberation::Deep,
    identity::Deliberation::Exhaustive,
];

/// Whether a decode actually *reasoned*, as against merely carrying the closed
/// block that suppression prefills.
///
/// An unterminated `<think>` counts as reasoning: it is the runaway case, where
/// the block ran to the token ceiling and took the whole decode with it.
pub(super) fn reasoned(raw: &str) -> bool {
    let Some(open) = raw.find("<think>") else {
        return false;
    };
    let body = &raw[open + "<think>".len()..];
    match body.find("</think>") {
        Some(end) => !body[..end].trim().is_empty(),
        // Opened and never closed — the runaway.
        None => true,
    }
}

impl Minds {
    /// Hand over the schema characters think under, once it is filled.
    ///
    /// Called once at load, before the cast is woken — a character woken first
    /// would open under the rendered prompt and keep it for the day.
    /// Draw on the shared layers by bearing from here on — see
    /// [`crate::engine::bearings`].
    pub fn set_scopes(&self, scopes: Scopes) {
        self.scopes.store(Some(Arc::new(scopes)));
    }

    pub fn set_projection(&self, projected: Projected) {
        tracing::info!(
            "minds: characters think under the mind's own projection — {} bytes of shared frame, \
             acts and identity selected per turn",
            projected.prompt.len()
        );
        self.warm_dreams(&projected);
        self.close_conversations(&projected);
        self.projection.store(Some(Arc::new(projected)));
    }

    /// Close the groups the cast's conversations are written to: a conversation
    /// reads its own group as its own lineage, and nothing of anybody else's.
    ///
    /// **A group that is not a turn's own is gathered across every
    /// conversation in it**, and under a sequence rule every turn found that
    /// way is held as the reader's founding history. The mind and the talks
    /// are two groups, so a character answering somebody read every other
    /// character's working turns as its own past — Zen told a person it had
    /// "just finished reviewing" a piece a Maker in another world had been
    /// handed half an hour before — and a character at work read everybody's
    /// private conversations the same way. Scoped, with no conversation ever
    /// named in a scope, each group offers nothing to a turn that is not its
    /// own; what happened in between reaches a talk from the journal.
    pub(super) fn close_conversations(&self, projected: &Projected) {
        self.engine.mark_group_scoped(projected.group);
        if let Some((_, talk)) = projected.talk {
            self.engine.mark_group_scoped(talk);
        }
    }

    /// The installed schema, as an `Arc` a caller may hold across an await.
    pub(crate) fn projected(&self) -> Option<Arc<Projected>> {
        self.projection.load_full()
    }

    /// The system prompt this character's latest turn was made under, or `None`
    /// before it has had one.
    pub fn system_prompt(&self, npc_id: u64) -> Option<rendered::Rendered> {
        self.system_prompts.pin().get(&npc_id).cloned()
    }

    /// The schema this character's conversations open with, and the live group
    /// they are written to — so what the console reads of a layer is what the
    /// character's own projection could gather from it.
    ///
    pub(super) fn as_character(&self, npc_id: u64) -> Option<(Builder, GroupId)> {
        let p = self.projected()?;
        Some((self.scoped_for(&p.builder, npc_id), p.group))
    }

    /// `builder` opened to what this character may read of the layers that are
    /// its own: its dreams, a few lines deep.
    pub(super) fn scoped_for(&self, builder: &Builder, npc_id: u64) -> Builder {
        dreams::scoped(builder, npc_id, dreams::IN_ACTING)
    }

    /// Run one thinking step against a **throwaway** conversation, and report
    /// everything about it.
    ///
    /// The instrument for the questions that decide whether the prompt works and
    /// which nothing else can answer: does the frame produce a call at all, does
    /// provenance bring the *right* acts into focus, does a reasoning block open
    /// when it should and stay shut when it should not, and does the call come
    /// out clean. Every one of those is invisible from outside — a character
    /// that emits no act and a character that chose to do nothing look identical
    /// in the pulse, and the difference is the whole problem.
    ///
    /// **Not a character.** It mints its own conversation, keeps nothing, and
    /// retires it on the way out, so probing does not disturb a cast that is
    /// running and does not leave a timeline behind. The trade is that it pays a
    /// conversation open each time.
    ///
    /// `projected` chooses which assembly is under test: the mind's schema, or
    /// the rendered prompt. Both are real paths a character can run under and
    /// the point of the probe is to compare them.
    pub async fn probe(
        &self,
        persona: &Persona<'_>,
        mode: Mode,
        thinking: identity::Deliberation,
        perception: &str,
        projected: bool,
        within: &tools::Within,
    ) -> anyhow::Result<Probe> {
        let started = std::time::Instant::now();
        let mut cfg = self.base_config.clone();
        cfg.context_window_turns = CONTEXT_WINDOW_TURNS;

        let under = self.projected().filter(|_| projected);
        let ran_projected = under.is_some();
        let (mut sequence, prompt, selection) = match under.as_deref() {
            Some(p) => {
                let sel = p.identities.selection_for(
                    // No npc id — a probe is nobody, so it reads the generic
                    // character member. What is under test is the frame and the
                    // acts, not whose name is at the top.
                    0,
                    persona.personality,
                    persona.world_id,
                    persona.building,
                    thinking,
                );
                let seq = self.engine.new_conversation_with_projection(
                    &p.prompt,
                    p.builder.clone(),
                    p.layer,
                    p.group,
                    cfg,
                )?;
                (seq, p.prompt.clone(), sel)
            }
            None => {
                // The same envelope `compile_act_loop` compiles, so what the prompt
                // shows a character is what the grammar will hold it to.
                let system = prompt::build(
                    persona,
                    mode,
                    &for_mode(mode),
                    &ToolCallEnvelope::for_dialect(&self.base_config.dialect),
                );
                let seq = self.engine.new_conversation(&system, cfg)?;
                // The dial applies with no schema to pin members in — it is the
                // turn's own, not the projection's.
                (seq, system, identity::deliberation(thinking))
            }
        };

        // Retired rather than left standing, however this request ends — a
        // probe that accumulated a timeline per run would fill the log with
        // conversations nobody can attribute, and a console that stops
        // polling drops this future mid-decode, where only a drop guard runs.
        let _retired = Throwaway::new(Arc::clone(&self.engine), sequence.timeline_id(), "probe");

        // The acts this turn can take, and only those — the same set, under the
        // same mode, the grammar below is compiled from. See `compile_act_loop`.
        let mut selection = selection;
        tools::show_within(&mut selection, Catalog::ROOM.mode, within);
        let turn_grammar = self.grammar_for(thinking, Catalog::ROOM, within);
        let options = TurnOptions {
            assistant_prefill: self.opening(thinking, turn_grammar.is_some()),
            turn_grammar: turn_grammar.map(TurnGrammar::of),
            sampling: Some(self.sampling_for(thinking)),
            selection,
            ..Default::default()
        };
        let response = sequence
            .send_turn_with_options_async(perception, options)
            .await?;
        let raw = response.text.clone();

        Ok(Probe {
            parsed: act::parse(&raw, &act_specs(Catalog::ROOM, within)),
            // **Reasoned, not merely "contains `<think>`".**
            //
            // On a family whose suppression *is* a prefilled empty block —
            // Qwen3.5 opens the assistant turn with `<think>\n\n</think>\n\n` —
            // the literal is in every suppressed decode by construction. A probe
            // that tested for the marker therefore reported thinking on exactly
            // the turns where it had been successfully turned off, and two
            // scenarios were read as failures for a night on that basis.
            //
            // What matters is whether anything was *written between* the two
            // markers. An empty block is suppression working; a full one is the
            // model reasoning.
            opened_think: reasoned(&raw),
            closed_think: raw.contains("</think>"),
            prompt_bytes: prompt.len(),
            projected: ran_projected,
            raw,
            ms: started.elapsed().as_millis() as u64,
        })
    }

    /// What every turn's reply is forced to begin with.
    ///
    /// **The character loop is an action loop.** There is no turn on which prose
    /// is a valid output: every one is one or more acts, and doing nothing is
    /// `wait`, which is itself an act. So free-decoding the opening is not a
    /// capability — it is the two decisions the model kept getting wrong, and
    /// both are decided here instead.
    ///
    /// What every turn's reply is forced to begin with.
    ///
    /// **The character loop is an action loop.** There is no turn on which prose
    /// is a valid output: every one is one or more acts, and doing nothing is
    /// `wait`, which is itself an act. So free-decoding the opening is not a
    /// capability — it is the two decisions the model kept getting wrong, and
    /// both are decided here instead.
    ///
    /// The closed reasoning block, then the marker that commits to the act
    /// grammar. Both are *written*, not sampled, so neither can come out any
    /// other way. What continues from the marker is [`Minds::grammar_for`],
    /// handed to the turn as its opening grammar rather than left to a trigger —
    /// see there for why a prefilled marker and a trigger are incompatible.
    ///
    /// This makes the reasoning question moot rather than solved — with the
    /// block prefilled closed there is nothing to steer and no chat-template
    /// capability to depend on. A mission that genuinely wants deliberation is
    /// the exception, and it takes the ordinary path.
    /// The turn's sampling, with this turn's think mode applied.
    ///
    /// **Each think mode has its own temperature and nucleus**, and the dial
    /// picks one per turn: the reasoning row for a mission that raises it, the
    /// cast's think-off row for [`identity::Deliberation::None`], which is what
    /// almost every turn runs. The cast decodes on the checkpoint's own
    /// published rows — the daemon no longer retunes the sampling per role.
    ///
    /// The two paths that decide a turn — the probe and the live act — both
    /// come through here, which is what keeps them from drifting.
    pub(super) fn sampling_for(&self, thinking: identity::Deliberation) -> SamplingConfig {
        self.base_config
            .sampling
            .clone()
            .with_think_mode(thinking.mode(), self.base_config.max_response_tokens)
    }

    pub(super) fn opening(
        &self,
        thinking: identity::Deliberation,
        grammar_ready: bool,
    ) -> Option<String> {
        // Nothing to prefill when *this turn* compiled no grammar — it
        // free-decodes, and seeding a marker no tree is bound to would commit it
        // to a shape nothing then enforces. Worse for a reasoning block: an
        // opened `<think>` with no tree to steer it shut runs to the token
        // ceiling and the turn never ends. Gated on the turn's own grammar, not
        // a boot-time flag — a grammar can compile for an empty room and fail
        // for this one (the acts a station adds), which the flag cannot know.
        if !grammar_ready {
            return None;
        }
        Some(match thinking {
            // Block already closed, straight into the call.
            identity::Deliberation::None => {
                format!("{}<tool_call>", self.base_config.dialect.no_think_block)
            }
            // Open the block and let the tree take it from there — through the
            // reasoning, out the other side, and into the calls.
            _ => "<think>".to_string(),
        })
    }

    /// The whole-turn grammar the reply **begins inside**.
    ///
    /// # Why this is not a trigger
    ///
    /// A trigger fires on a *decoded* token: both checks that consult the
    /// registry — `scheduler::decode`'s per-token one and `scheduler::prefill`'s
    /// first-sampled-token one — run only on tokens the sampler produced. An
    /// assistant prefill is tokenized straight into the turn's K/V and passes
    /// neither.
    ///
    /// So [`Minds::opening`] and a trigger on `<tool_call>` cannot both be
    /// right, and for a long time they were both present: the marker was
    /// prefilled *and* the tree was registered against it. The marker was
    /// therefore one the model never emitted, the trigger never fired, and every
    /// character turn free-decoded while wearing the shape of a constrained one
    /// — output that opens like a call and closes itself with `</tool_call>`,
    /// enforced by nothing. It produced tool names absent from the catalog
    /// (`look`, `think`) and `say` with no `intent`, neither of which is
    /// reachable through the tree, whose name branch is masked to the catalog's
    /// trie and whose required arguments are nodes rather than suggestions.
    ///
    /// Entering the tree directly keeps the prefill — which is the cheaper and
    /// stronger way to emit a token that was never in question — and arms the
    /// walk at the node that follows it.
    /// Built for *this room*, and cached on it.
    ///
    /// A miss compiles one nine-act catalog. That is a per-turn cost the boot
    /// -time version did not pay, and it buys the two things a grammar fixed at
    /// boot could never know: which acts are reachable from where the character
    /// stands, and which names are real. Both were advisory before — advertised
    /// in the prompt, enforced by nothing — and a request the mask contradicts
    /// is a request the model is free to ignore, which it did.
    pub(in super::super) fn grammar_for(
        &self,
        thinking: identity::Deliberation,
        catalog: Catalog,
        within: &tools::Within,
    ) -> Option<Arc<StencilTree>> {
        if !self.grammar_ok {
            return None;
        }
        let key = (thinking, catalog, within.clone());
        if let Some(tree) = self.acts.pin().get(&key) {
            return Some(Arc::clone(tree));
        }
        // A compile takes a while, and a second character arriving mid-compile
        // does not wait behind it: it compiles too.
        let built =
            match compile_act_loop(&self.engine, &self.base_config, thinking, catalog, within) {
                Ok(tree) => tree,
                Err(e) => {
                    tracing::error!(
                    "the {thinking:?} grammar would not compile for this room: {e:#} — this turn \
                     free-decodes"
                );
                    return None;
                }
            };
        // Last writer wins: two characters in one room race to build the same
        // tree and either is correct, since the key determines the contents.
        self.acts.pin().insert(key, Arc::clone(&built));
        Some(built)
    }

    /// The model's own dialect and sampling, as captured at load.
    ///
    /// Handed out rather than reassembled by the caller for the reason it is
    /// held here at all: a second opinion about a checkpoint's markers or its
    /// sampling defaults is a silent way to run a model outside the settings it
    /// was tuned under.
    pub fn base_config(&self) -> SequenceConfig {
        self.base_config.clone()
    }
}
