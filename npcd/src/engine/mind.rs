//! A character's live conversation, and the decode that turns perception into
//! acts.
//!
//! # One conversation per character per day
//!
//! Each character holds a [`Sequence`] — a conversation on the substrate — for
//! the day it is living in. Its id is *derived* from `(npc_id, day)` rather than
//! allocated, and recorded against the timeline in the redo log, so the day's
//! turns are identifiable as that character's day from the log alone. A daemon
//! restarted at noon opens a fresh timeline for the rest of the day: the
//! sequence is GPU state and does not survive the process. What the derived id
//! buys is that both halves of the day carry the same name, so the morning's
//! turns are still the character's own history rather than an anonymous
//! timeline nothing can attribute.
//!
//! At the day boundary the conversation is retired and a new one opened. Retired
//! means **tombstoned**, not deleted: the turns stay in the redo log and stay
//! reachable by the gather. What changes is that they are no longer selected by
//! default. This is the mind design's soft fade made explicit at a boundary —
//! yesterday stops being transcript and becomes memory.
//!
//! # Locking
//!
//! The map of live conversations and a live conversation are two locks, taken in
//! that order and never together for long. A decode is seconds; the map is what
//! [`Minds::resident`] and every Pulse header read. Holding the map across the
//! decode — which it used to — stalled the whole daemon behind whichever
//! character happened to be thinking, on tokio worker threads.
//!
//! # Why the sequence is windowed as well
//!
//! `SequenceConfig::context_window_turns` bounds what is prefilled onto the GPU
//! per turn. That is the same discipline as [`crate::engine::window`] and for
//! the same reason, one layer down: the substrate holds everything, the gather
//! decides what is relevant, and only a bounded tail is carried verbatim. Both
//! bounds are deliberate and neither is the other's fallback.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use candle_conversation::projection::{Builder, GroupId, LayerId};
use candle_conversation::stencil::{
    compile_action_loop, compile_think_tree, StencilTree, ThinkMode, ThinkSteerEnvelope,
    ToolCallEnvelope,
};
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig, TurnOptions};
use serde::Serialize;

use crate::engine::act::{self, Parsed};
use crate::engine::event::Event;
use crate::engine::identity;
use crate::engine::prompt::{self, Persona};
use crate::engine::sleep::conversation_id;
use crate::engine::tools::{self, for_mode, Mode};
use crate::engine::window::{Speaker, Window};

/// How many completed exchanges the GPU sequence carries per turn.
///
/// Matches [`crate::engine::window::DEFAULT_TURNS`] in intent — a short verbatim
/// tail, with continuity coming from the gather — but counts exchanges rather
/// than turns, so it is half the number.
pub const CONTEXT_WINDOW_TURNS: usize = 12;

/// The GPU tail and the perception window are two bounds with one intent,
/// counted in different units — exchanges here, turns there. Held at compile
/// time rather than by a test, so a change to either constant has to reckon with
/// the other even in a build nobody runs the tests for.
const _: () = assert!(CONTEXT_WINDOW_TURNS * 2 <= crate::engine::window::DEFAULT_TURNS);

/// A character's live conversation.
struct Live {
    sequence: Sequence,
    /// The day this conversation belongs to. A mismatch against the world's day
    /// is what triggers the roll-over.
    day: u64,
    id: String,
}

/// Every character's conversation, and the engine they run on.
pub struct Minds {
    engine: Arc<Mutex<ConversationEngine>>,
    /// The model's own dialect and sampling, captured at load.
    ///
    /// From `ModelBuilder::conversation_config` rather than assembled here: the
    /// dialect's markers and the model's sampling defaults belong to the
    /// checkpoint, and a second opinion about either is a silent way to run a
    /// model outside the settings it was tuned under.
    base_config: SequenceConfig,
    /// One whole-turn grammar per deliberation level, compiled once.
    ///
    /// **This is what turns deliberation on and off**, not the dialect's
    /// `/no_think` marker: the marker exists only for Qwen3, and npcd runs
    /// Qwen3.5, where sending it is ordinary text and suppresses nothing. A
    /// grammar acts on the decoded token instead, so it holds on every family.
    ///
    /// Each level's tree covers the *whole* turn — the reasoning block, if its
    /// work calls for one, and then the acts. Empty when the checkpoint's
    /// tokenizer has no single `<tool_call>`/`<think>` token, in which case
    /// turns free-decode as they did before.
    ///
    /// Trees rather than trigger registries: a turn **begins** inside its
    /// grammar, it does not wait to enter one. See [`Minds::grammar_for`].
    ///
    /// Keyed by what the room is as well as by the level, because what a
    /// character may do and whom it may name are facts about where it stands.
    /// A cast is small and rooms are stable, so this hits almost always; the
    /// miss costs one compile of a nine-act catalog, against a decode measured
    /// in seconds.
    acts: Mutex<HashMap<(identity::Deliberation, tools::Within), Arc<StencilTree>>>,
    /// Whether the catalog compiled at boot. `false` means this checkpoint
    /// cannot be held to a shape at all, and every turn free-decodes — asked
    /// once here rather than inferred from an empty cache, which would also be
    /// true of a cache that has simply not been filled yet.
    grammar_ok: bool,
    /// The mind's own schema, once its collections are filled.
    ///
    /// Installed after this struct is published rather than handed to the
    /// constructor, because the engine is deliberately made reachable at the
    /// substrate step — before the projection is parsed — so a slow start still
    /// answers its routes. `None` for a daemon with no mind: its characters
    /// think under the rendered prompt.
    projection: RwLock<Option<Projected>>,
    /// Each conversation behind its own lock, so a decode holds only the
    /// character that is thinking — see the module's *Locking* note.
    live: Mutex<HashMap<u64, Arc<Mutex<Live>>>>,
}

/// The schema a character's conversation opens against, with its acts and
/// identities already installed.
pub struct Projected {
    /// The schema's own static text — identical for the whole cast.
    pub prompt: String,
    pub builder: Builder,
    pub layer: LayerId,
    pub group: GroupId,
    /// Which members exist, so a turn pins the right ones.
    pub identities: identity::Installed,
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
fn compile_act_loop(
    engine: &Arc<Mutex<ConversationEngine>>,
    cfg: &SequenceConfig,
    thinking: identity::Deliberation,
    within: &tools::Within,
) -> anyhow::Result<Arc<StencilTree>> {
    let e = engine.lock().unwrap();
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
    let env = ToolCallEnvelope {
        // The tree resumes *after* the marker, which the turn's prefill has
        // already written — so the walk starts here, at the first thing that was
        // ever actually in question.
        open: "\n{\"name\": \"".to_string(),
        args_open: ", \"arguments\": {".to_string(),
        close: "}}\n</tool_call>".to_string(),
        marker: "<tool_call>".to_string(),
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
                eos: tok.token_to_id(cfg.dialect.assistant_end).unwrap_or(0),
                after_close: "",
            };
            compile_think_tree(thinking.mode(), &steer)
        }
    };
    let spec = compile_action_loop(
        &tools::specs_within(Mode::Physical, within),
        &env,
        tools::ACTS_PER_TURN,
        cfg.dialect.assistant_end,
        prelude.as_ref(),
    )?;
    Ok(Arc::new(e.compile_stencil(&spec)?))
}

/// Every deliberation level, so a turn can pick its grammar without compiling
/// one. Five trees, built once — the alternative is compiling a stencil inside
/// the tick loop, which is a per-turn cost for a thing that never changes.
const LEVELS: [identity::Deliberation; 5] = [
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
fn reasoned(raw: &str) -> bool {
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

/// What one character's tick produced.
#[derive(Debug, Default)]
pub struct Thought {
    pub parsed: Parsed,
    /// Set when this tick opened a new day's conversation.
    pub rolled_over: Option<(u64, u64)>,
}

impl Minds {
    pub fn new(engine: Arc<Mutex<ConversationEngine>>, base_config: SequenceConfig) -> Self {
        let mut base_config = base_config;
        // **Program the reasoning-block close budget, or a block that opens
        // never shuts.**
        //
        // `force_segment_close_after` defaults to `0` — disabled — so nothing
        // ever rewrites the next token to `</think>`. Under the rendered prompt
        // that never mattered, because nothing invited the model to think at
        // all. Under the mind's schema it does invite one, and the block ran to
        // `max_response_tokens` and the whole decode was discarded as
        // all-reasoning: every tick produced the single word `<think>` and no
        // acts, with nothing logged as an error, because a decode that reasons
        // forever is a successful decode.
        //
        // `Off` is the budget for a character: it assumes the model self-closes
        // an empty block from the no-think glue and only caps a runaway. A
        // mission that genuinely needs thinking raises the *selector*
        // (`identity::Deliberation`); this is the backstop underneath, and a
        // backstop that is off is not a backstop.
        {
            let e = engine.lock().unwrap();
            let max = base_config.max_response_tokens;
            base_config
                .sampling
                .apply_think_mode(ThinkMode::Off, e.tokenizer(), max);
        }
        // **A cast is not an assistant, and the architecture default is tuned
        // for one.** The checkpoint's own numbers are Qwen's general-task
        // pairing — temperature 0.7 into a `top_p` 0.8 nucleus — which is the
        // right conservative choice for answering a question and the wrong one
        // for speaking as somebody. A character handed a situation much like the
        // last one lands on the same sentence out of a nucleus that narrow, and
        // did: one said the same thing, word for word, for a hundred turns.
        //
        // Qwen's wider published pairing is the one their own creative-writing
        // runs used. Taken here rather than in the architecture default, so the
        // same checkpoint serving zend keeps the narrow one — see
        // [`SamplingConfig::for_character_dialogue`].
        base_config.sampling = base_config.sampling.for_character_dialogue();
        // Compiled at boot for the empty room, which proves the catalog builds
        // at all and warms the cache for the commonest case. A checkpoint whose
        // tokenizer lacks the markers yields nothing and turns free-decode
        // exactly as before — loud, but never fatal: a daemon that refuses to
        // boot over it helps nobody.
        let acts = Mutex::new(HashMap::new());
        let mut ok = true;
        for level in LEVELS {
            match compile_act_loop(&engine, &base_config, level, &tools::Within::nowhere()) {
                Ok(tree) => {
                    acts.lock()
                        .unwrap()
                        .insert((level, tools::Within::nowhere()), tree);
                }
                Err(e) => {
                    ok = false;
                    tracing::error!(
                        "the {level:?} turn grammar would not compile: {e:#} — characters will \
                         free-decode, and a malformed act will be refused rather than prevented"
                    );
                    break;
                }
            }
        }
        match ok {
            true => tracing::info!(
                "turn grammar armed over {} acts, {} per turn — every turn is a reasoning block \
                 (only when the work calls for one) followed by calls, and nothing else is \
                 reachable. What is within reach and who may be addressed are bound per turn \
                 from the room.",
                tools::CATALOG.len(),
                tools::ACTS_PER_TURN,
            ),
            false => tracing::warn!(
                "turn grammar inactive — characters free-decode, so a required parameter is a \
                 request rather than a guarantee"
            ),
        }
        Self {
            engine,
            base_config,
            acts,
            grammar_ok: ok,
            projection: RwLock::new(None),
            live: Mutex::new(HashMap::new()),
        }
    }

    /// Hand over the schema characters think under, once it is filled.
    ///
    /// Called once at load, before the cast is woken — a character woken first
    /// would open under the rendered prompt and keep it for the day.
    pub fn set_projection(&self, projected: Projected) {
        tracing::info!(
            "minds: characters think under the mind's own projection — {} bytes of shared frame, \
             acts and identity selected per turn",
            projected.prompt.len()
        );
        *self.projection.write().unwrap() = Some(projected);
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
    pub fn probe(
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

        let guard = self.projection.read().unwrap();
        let under = guard.as_ref().filter(|_| projected);
        let ran_projected = under.is_some();
        let (mut sequence, prompt, selection) = match under {
            Some(p) => {
                let sel = p.identities.selection_for(
                    // No npc id — a probe is nobody, so it reads the generic
                    // character member. What is under test is the frame and the
                    // acts, not whose name is at the top.
                    0,
                    persona.personality,
                    persona.world_id,
                    thinking,
                );
                let seq = self
                    .engine
                    .lock()
                    .unwrap()
                    .new_conversation_with_projection(
                        &p.prompt,
                        p.builder.clone(),
                        p.layer,
                        p.group,
                        cfg,
                    )?;
                (seq, p.prompt.clone(), sel)
            }
            None => {
                let system = prompt::build(persona, mode, &for_mode(mode));
                let seq = self.engine.lock().unwrap().new_conversation(&system, cfg)?;
                // The dial applies with no schema to pin members in — it is the
                // turn's own, not the projection's.
                (seq, system, identity::deliberation(thinking))
            }
        };
        drop(guard);

        let options = TurnOptions {
            turn_grammar: self.grammar_for(thinking, within),
            selection,
            assistant_prefill: self.opening(thinking),
            ..Default::default()
        };
        let response = sequence.send_turn_with_options(perception, options)?;
        let raw = response.text.clone();
        // Retired rather than left standing: a probe that accumulated a timeline
        // per run would fill the log with conversations nobody can attribute.
        if let Ok(e) = self.engine.lock() {
            let _ = e.tombstone_timeline(sequence.timeline_id());
        }

        Ok(Probe {
            parsed: act::parse(&raw),
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
    fn opening(&self, thinking: identity::Deliberation) -> Option<String> {
        // Nothing to prefill when no grammar compiled — the turn free-decodes,
        // and seeding a marker no tree is bound to would commit it to a shape
        // nothing then enforces.
        if !self.grammar_ok {
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
    fn grammar_for(
        &self,
        thinking: identity::Deliberation,
        within: &tools::Within,
    ) -> Option<Arc<StencilTree>> {
        if !self.grammar_ok {
            return None;
        }
        let key = (thinking, within.clone());
        if let Some(tree) = self.acts.lock().unwrap().get(&key) {
            return Some(Arc::clone(tree));
        }
        // Compiled outside the cache lock: it takes the engine lock, and a
        // second character arriving mid-compile must not wait behind it.
        let built = match compile_act_loop(&self.engine, &self.base_config, thinking, within) {
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
        self.acts.lock().unwrap().insert(key, Arc::clone(&built));
        Some(built)
    }

    /// The engine, for work that is not a character thinking.
    ///
    /// [`crate::lifegen`] primes its own conversations and forks them; it is not
    /// a character taking a turn, so it does not go through [`Self::think`]. The
    /// handle is shared rather than a second engine because there is one card
    /// and one scheduler, and the generator's forks batch into the same waves as
    /// everything else.
    pub fn engine(&self) -> Arc<Mutex<ConversationEngine>> {
        Arc::clone(&self.engine)
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

    /// How many characters currently hold an open conversation.
    ///
    /// Not the same as the scheduler's population: a character wakes into the
    /// scheduler at startup and only opens a conversation on its first tick, so
    /// this trails it. The gap is exactly "how many have thought at least once",
    /// which is what the Pulse header reports.
    pub fn resident(&self) -> usize {
        self.live.lock().unwrap().len()
    }

    /// Run one thinking step: perception in, acts out.
    ///
    /// Errors are returned rather than swallowed. A decode that failed and a
    /// character that chose to do nothing produce the same empty act list, and
    /// the caller has to be able to tell them apart — Pulse renders them
    /// differently, and it should.
    // Eight, and each is a different axis of one turn: who, as what, in what
    // mode, on what day, given what arrived, against what it remembers, and
    // from what the room offers. Bundling them into a struct would name the
    // bundle rather than the axes and hide that every caller must supply all of
    // them — which is the property that matters.
    #[allow(clippy::too_many_arguments)]
    pub fn think(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        day: u64,
        events: &[Event],
        window: &Window,
        within: &tools::Within,
    ) -> anyhow::Result<Thought> {
        let mut thought = Thought::default();

        // The map is held only long enough to find or open this character's
        // conversation; the decode below runs under the conversation's own lock.
        let conversation = {
            let mut live = self.live.lock().unwrap();

            // The day boundary. Checked here rather than on a timer because a
            // timer fires on host-elapsed time and would be wrong the moment the
            // narrative clock is paused, jumped or re-paced — all of which the
            // console can do at any moment.
            if let Some(existing) = live.get(&npc_id) {
                let from = existing.lock().unwrap().day;
                if from != day {
                    // Tombstone, not delete. The turns stay in the redo log; what
                    // changes is that they stop being selected by default.
                    if let Some(l) = live.remove(&npc_id) {
                        retire(&self.engine, &l.lock().unwrap());
                    }
                    thought.rolled_over = Some((from, day));
                }
            }

            // Opened through the vacant entry rather than `contains_key` + `insert`,
            // so the map is probed once and the borrow below cannot miss.
            match live.entry(npc_id) {
                Entry::Occupied(e) => Arc::clone(e.get()),
                Entry::Vacant(slot) => {
                    let id = conversation_id(npc_id, day);
                    let mut cfg = self.base_config.clone();
                    cfg.context_window_turns = CONTEXT_WINDOW_TURNS;
                    // **The rendered prompt, until the reasoning block closes.**
                    //
                    // Everything for the projection path is built and installed
                    // — the acts, the identities, the world and the building are
                    // in the schema's collections and selected per turn, and
                    // `Projected` is handed over at boot. What is not solved is
                    // the decode: under the mind's schema the checkpoint opens
                    // `<think>` and never closes it, so the turn runs to
                    // `max_response_tokens` and is discarded whole as reasoning.
                    // Every tick then produces no acts and logs no error,
                    // because a decode that reasons forever is a successful one.
                    //
                    // Suppressing it three ways did not: the `no_think` glue
                    // present, `thinking_effort` off, and the `ThinkMode::Off`
                    // close budget programmed into the sampling config. So the
                    // remaining fault is below the prompt, and a character that
                    // cannot act is worse than one whose prompt is a rendered
                    // copy — which is what this is until that is found.
                    let system = prompt::build(persona, mode, &for_mode(mode));
                    let sequence = self.engine.lock().unwrap().new_conversation(&system, cfg)?;
                    // **The derived id, given to the substrate.** Minting it and keeping it
                    // in this struct made it a log label and nothing else: the timeline went
                    // into the redo log anonymous, so nothing downstream could attribute a
                    // day's turns to the character that lived them, and the two doc comments
                    // promising a stable per-day identity described a string that never left
                    // the process. Failure is logged rather than propagated — an unnamed
                    // timeline is worse reporting, not a character that cannot think.
                    if let Err(e) = self
                        .engine
                        .lock()
                        .unwrap()
                        .set_conversation_conv_id(sequence.timeline_id(), &id)
                    {
                        tracing::warn!(
                            "conversation {id} could not be named in the log: {e:?} — its turns \
                             will not be attributable to this character"
                        );
                    }
                    Arc::clone(slot.insert(Arc::new(Mutex::new(Live { sequence, day, id }))))
                }
            }
        };

        // Everything drained this tick, as one message. A fat batch is one
        // better-informed thinking step rather than several thrashing ones —
        // the mind design is explicit that this is what a busy character should
        // get.
        let perception = compose(events, window);
        // **The act stencil is armed here, not hoped for.** `act::parse` reads
        // the decode, and what it reads is a grammar's output rather than a
        // guess at one: the name is a real tool, the required parameters are
        // present, the JSON closes. Everything it still rejects is a genuine
        // choice the character made, which is what makes a rejection worth
        // handing back to it.
        // **Who this turn is, and how hard it thinks.**
        //
        // The identity collections are `Named`, so an unpinned one emits
        // nothing — a character with no identity at all, and silently. Every
        // collection is pinned, down to its generic member.
        //
        // The deliberation dial is set **either way**. It is a property of the
        // turn rather than of the schema — Qwen3 honours `/no_think` only from
        // the user opener, which the conversation layer bakes from this
        // selection — so a daemon running the rendered prompt needs it just as
        // much. It used to be built only on the projected branch, so the path
        // the daemon actually ran never injected the switch at all.
        let selection = self
            .projection
            .read()
            .unwrap()
            .as_ref()
            .map(|p| {
                p.identities
                    // **How hard to think comes from the work.** A mission
                    // carries its own [`identity::Deliberation`] — drafting a
                    // story into a gap in the chronicle is a thinking problem,
                    // deciding who to talk to is not. No mission exists as data
                    // yet, so every character is on the standing instruction,
                    // which is the default: act, do not deliberate.
                    .selection_for(
                        npc_id,
                        persona.personality,
                        persona.world_id,
                        identity::Deliberation::default(),
                    )
            })
            .unwrap_or_else(|| identity::deliberation(identity::Deliberation::default()));
        let options = TurnOptions {
            turn_grammar: self.grammar_for(identity::Deliberation::default(), within),
            selection,
            assistant_prefill: self.opening(identity::Deliberation::default()),
            ..Default::default()
        };
        let response = conversation
            .lock()
            .unwrap()
            .sequence
            .send_turn_with_options(&perception, options)?;
        thought.parsed = act::parse(&response.text);
        Ok(thought)
    }

    /// Retire a character's conversation — on delete, or on shutdown.
    pub fn retire_npc(&self, npc_id: u64) {
        // Taken out of the map first, then locked: a character mid-decode is
        // waited on rather than tombstoned underneath, and the map is free for
        // everyone else while we wait.
        let gone = self.live.lock().unwrap().remove(&npc_id);
        if let Some(l) = gone {
            retire(&self.engine, &l.lock().unwrap());
        }
    }
}

/// Tombstone a conversation's timeline.
///
/// Failure is logged, never propagated. A tombstone that did not land leaves
/// yesterday's turns selectable — untidy, and strictly better than refusing to
/// open today's conversation over it.
fn retire(engine: &Arc<Mutex<ConversationEngine>>, l: &Live) {
    let timeline = l.sequence.timeline_id();
    match engine.lock().unwrap().tombstone_timeline(timeline) {
        Ok(()) => tracing::info!("conversation {} retired (tombstoned)", l.id),
        Err(e) => tracing::warn!(
            "conversation {} could not be tombstoned: {e:?} — yesterday stays selectable",
            l.id
        ),
    }
}

/// What the character reads this tick.
///
/// The events, as prose, in arrival order. The window is *not* pasted in: the
/// sequence carries its own bounded tail (`context_window_turns`) and the
/// substrate carries the rest, so repeating the window here would put the same
/// turns in the context twice — once verbatim from us and once from the
/// sequence's own history — and teach the model that everything happens twice.
fn compose(events: &[Event], window: &Window) -> String {
    let _ = window;
    let mut s = String::new();
    for (i, e) in events.iter().enumerate() {
        if i > 0 {
            s.push_str("\n\n");
        }
        s.push_str(&e.prose());
    }
    s
}

/// Render a window turn for a transcript view.
pub fn render_turn(speaker: Speaker, text: &str) -> String {
    match speaker {
        Speaker::World => text.to_string(),
        Speaker::Npc => format!("→ {text}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::event::{EventKind, Salience};

    fn ev(text: &str) -> Event {
        Event::new(
            0,
            0,
            Salience::NORMAL,
            EventKind::Description { text: text.into() },
        )
    }

    /// A fat batch arrives as one message, in order — one better-informed
    /// thinking step rather than several thrashing ones.
    #[test]
    fn a_batch_composes_in_arrival_order() {
        let w = Window::with_default_cap();
        let s = compose(&[ev("the gate opens"), ev("someone shouts")], &w);
        assert_eq!(s, "the gate opens\n\nsomeone shouts");
    }

    /// **The window must not be pasted in.** The sequence carries its own
    /// bounded tail, so repeating it here would put the same turns in the
    /// context twice and teach the model that everything happens twice.
    #[test]
    fn the_window_is_not_repeated_into_the_message() {
        let mut w = Window::with_default_cap();
        w.push_world("something that already happened", 0, None);
        w.push_npc("and what I did about it", 0);
        let s = compose(&[ev("something new")], &w);
        assert_eq!(s, "something new");
        assert!(!s.contains("already happened"));
    }

    #[test]
    fn an_empty_batch_composes_to_nothing() {
        assert_eq!(compose(&[], &Window::with_default_cap()), "");
    }

    /// A restart mid-day opens a second timeline, and it must carry the same
    /// day's name so both halves stay attributable to the character.
    #[test]
    fn a_days_conversation_id_is_stable_across_a_restart() {
        assert_eq!(conversation_id(4, 9), conversation_id(4, 9));
        assert_ne!(conversation_id(4, 9), conversation_id(4, 10));
    }

    #[test]
    fn an_act_renders_distinguishably_from_perception() {
        assert_eq!(render_turn(Speaker::World, "it rains"), "it rains");
        assert_eq!(render_turn(Speaker::Npc, "waits"), "→ waits");
    }
}
