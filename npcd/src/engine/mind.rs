//! A character's live conversation, and the decode that turns perception into
//! acts.
//!
//! # One conversation per character per day
//!
//! Each character holds a [`Sequence`] — a conversation on the substrate — for
//! the day it is living in. Its id is *derived* from `(npc_id, day)` rather than
//! allocated, and recorded against the timeline in the redo log, so the day's
//! turns are identifiable as that character's day from the log alone.
//!
//! # A restart rejoins rather than starts over
//!
//! The sequence is GPU state and does not survive the process, but the
//! conversation it was writing into does. A daemon restarted at noon looks for
//! the timeline carrying this character's derived id — tagged with a fingerprint
//! of the frame it was opened under — and, finding one written under the frame
//! this process is running, opens a fresh slot onto it and carries on. The
//! turns, the recurrent state and the carried selection belief all come back.
//!
//! It used to mint a new timeline unconditionally, which made the derived id a
//! name and nothing more: a character that had lived all morning woke with no
//! memory of it beyond what the gather happened to retrieve, and the morning's
//! conversation was left live and unreadable for good. See
//! [`Minds::open_conversation`].
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
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, RwLock};

use candle_conversation::projection::{Builder, GroupId, LayerId, TimelineId};
use candle_conversation::stencil::{
    compile_action_loop, compile_think_tree, StencilTree, ThinkMode, ThinkSteerEnvelope,
    ToolCallEnvelope,
};
use candle_conversation::{
    ConversationEngine, SamplingConfig, Sequence, SequenceConfig, TurnOptions,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::engine::act::{self, Parsed};
use crate::engine::dreams;
use crate::engine::event::Event;
use crate::engine::identity;
use crate::engine::layers;
use crate::engine::prompt::{self, Persona};
use crate::engine::reflect;
use crate::engine::retention;
use crate::engine::schema::ReflectionTurns;
use crate::engine::sleep::conversation_id;
use crate::engine::tools::{self, for_mode, Mode};
use crate::engine::window::{Speaker, Window};

/// How many completed exchanges the GPU sequence carries per turn.
///
/// Matches [`crate::engine::window::DEFAULT_TURNS`] in intent — a bounded
/// verbatim tail, with continuity coming from the gather — but counts exchanges
/// rather than turns, so it is half the number.
pub const CONTEXT_WINDOW_TURNS: usize = 32;

/// The GPU tail and the perception window are two bounds with one intent,
/// counted in different units — exchanges here, turns there. Held at compile
/// time rather than by a test, so a change to either constant has to reckon with
/// the other even in a build nobody runs the tests for.
const _: () = assert!(CONTEXT_WINDOW_TURNS * 2 <= crate::engine::window::DEFAULT_TURNS);

/// The metadata key carrying a conversation's `npc-<id>-day-<n>` name.
///
/// The same string as its `conv_id`, deliberately duplicated into the free-form
/// bag: `conv_id` is a display name and answering "which timeline is this?" from
/// it means walking every conversation the workspace knows, where the metadata
/// side is an indexed lookup that already excludes tombstones. Resume runs on
/// every character's first tick, so which of those it is matters.
const META_CONVERSATION: &str = "npc_conversation";

/// The metadata key carrying the fingerprint of the frame a conversation was
/// opened under — see [`frame_fingerprint`].
const META_FRAME: &str = "frame_sha256";

/// A character's live conversation.
struct Live {
    sequence: Sequence,
    /// The day this conversation belongs to. A mismatch against the world's day
    /// is what triggers the roll-over.
    day: u64,
    id: String,
    /// The first turn retention has **not** yet retired.
    ///
    /// Zero for a conversation just minted, and [`retention::resume_watermark`]
    /// for one rejoined — the first turn that conversation has not already
    /// retired. Zero would still be *safe* there, since an already-tombstoned
    /// turn costs a lookup and no write, but a conversation that never ends gets
    /// deep enough that walking up from the bottom is the whole first minute of
    /// a restart. See [`retention::retire_expired`].
    retired_through: u32,
    /// What became of the acts this conversation's **last** decode called for,
    /// waiting to be handed back on the next turn.
    ///
    /// One entry per call the character made, in the order it made them —
    /// refusals and malformed calls included, because "that did not work" is
    /// the result a character most needs to read.
    ///
    /// # Why this is held here and not by the caller
    ///
    /// These are answers to calls made in *this conversation's* previous
    /// assistant turn, so they are only meaningful against that history. Held
    /// beside the sequence, they die with it: a day roll-over replaces the
    /// `Live` and yesterday's answers go with it rather than being delivered
    /// into a conversation that never asked the questions.
    ///
    /// See [`Minds::deliver_outcomes`] and [`compose`].
    pending: Vec<String>,
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
    /// How many turns stay verbatim in the redo log, or `None` to keep them all.
    ///
    /// **Off unless the projection asks for it**, because most conversations are
    /// finite and their transcript is the product. A character's is neither: it
    /// never ends, so nothing else would ever make one of its turns dead, and a
    /// log with no dead records is a log compaction cannot reclaim. See
    /// [`crate::engine::retention`] for what that cost in practice.
    keep_turns: Option<u64>,
    /// The two user turns a reflection sends, as the mind authors them.
    ///
    /// `None` for a mind that declares no `reflection` block, and reflection is
    /// then unavailable — there is no built-in wording to fall back to, because a
    /// second copy of a prompt is a copy that diverges from the one being edited.
    /// See [`crate::engine::schema::ReflectionTurns`].
    reflection: Option<ReflectionTurns>,
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
    /// How many turns this tick retired from the log.
    ///
    /// Reported rather than silent so an operator watching the pulse can see
    /// retention working — a log that quietly stops growing looks identical to
    /// one that quietly stopped being written. Ordinarily one; more than one
    /// means a gap was being closed.
    pub retired: u32,
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
        // for one.** A character handed a situation much like the last one lands
        // on the same act out of a narrow nucleus, and did: one said the same
        // thing, word for word, for a hundred turns, and on the card's think-off
        // row another walked between two rooms twenty times without answering
        // the question in front of it. So the think-off row is widened to
        // `1.0 / 0.95`, and the repetition load moves onto DRY and a cross-turn
        // penalty. Set on the row rather than on the temperature, so the
        // per-turn switch in [`Self::sampling_for`] still picks it — see
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
            keep_turns: None,
            reflection: None,
        }
    }

    /// Bound how much of every character's conversation stays verbatim on disk.
    ///
    /// Set from the projection's `turn_retention`, so a deployment that wants
    /// whole transcripts simply omits it. See [`crate::engine::retention`].
    pub fn keeping_turns(mut self, keep: Option<u64>) -> Self {
        match keep {
            Some(n) => tracing::info!(
                "conversations keep {n} turns verbatim; older turns are retired from the log so \
                 compaction can reclaim them"
            ),
            None => tracing::info!(
                "conversations keep every turn — the log grows without bound unless something \
                 else retires them"
            ),
        }
        self.keep_turns = keep;
        self
    }

    /// Install the reflection turns the mind authors.
    ///
    /// `None` leaves reflection unavailable and says so once at load, rather than
    /// at the first request — a deployment that has lost the block should find out
    /// from its own startup log and not from a character that will not reflect.
    pub fn asking(mut self, turns: Option<ReflectionTurns>) -> Self {
        match &turns {
            Some(_) => tracing::info!(
                "reflection armed — two authored turns into a transient conversation that is \
                 tombstoned when they are answered"
            ),
            None => tracing::warn!(
                "reflection unavailable — the mind declares no usable `reflection` block, so \
                 nothing will generate a dream brief"
            ),
        }
        self.reflection = turns;
        self
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
        self.warm_dreams(&projected);
        *self.projection.write().unwrap() = Some(projected);
    }

    /// Every layer of the mind's projection, with how much of each this
    /// character can read. `None` before the projection is loaded.
    pub fn layer_counts(&self, npc_id: u64) -> Option<Vec<layers::LayerCount>> {
        let (builder, live) = self.as_character(npc_id)?;
        Some(layers::counts(
            &self.engine.lock().unwrap(),
            &builder,
            live,
            npc_id,
        ))
    }

    /// The newest `limit` conversations of one layer this character can read.
    /// `None` for a layer the projection does not declare, or before it is
    /// loaded.
    pub fn layer_page(&self, npc_id: u64, layer: &str, limit: usize) -> Option<layers::Page> {
        let (builder, live) = self.as_character(npc_id)?;
        layers::page(
            &self.engine.lock().unwrap(),
            &builder,
            live,
            npc_id,
            layer,
            limit,
        )
    }

    /// The schema this character's conversations open with, and the live group
    /// they are written to — so what the console reads of a layer is what the
    /// character's own projection could gather from it.
    ///
    /// The projection lock is let go before anything takes the engine's.
    fn as_character(&self, npc_id: u64) -> Option<(Builder, GroupId)> {
        let projection = self.projection.read().unwrap();
        let p = projection.as_ref()?;
        Some((
            dreams::scoped(&p.builder, npc_id, dreams::IN_ACTING),
            p.group,
        ))
    }

    /// Put every dream already kept on the normalized score band.
    ///
    /// A recalled line is scored against its own **hit level** — the score it
    /// reaches when it is the answer — so the dream group's gate means what the
    /// same numbers mean on `repo_map`. The levels are learned from the traffic
    /// that reads them: every turn a character takes carries its own dreams tag,
    /// so its seal teaches its own dreams, and a line that matches every turn
    /// ends up with a high level and is discounted.
    ///
    /// **This is the cold start, not the level.** A dream is probed by its own
    /// lines, and the vote is a sum over a probe's strongest tokens — a line is
    /// one sentence where a turn probes with its whole tail, so self-match lands
    /// well below what the room reaches and live traffic lifts it within a few
    /// turns. It still beats the bare prior, which multiplies the raw vote rather
    /// than normalizing it: recalled lines scored anywhere from four thousand to
    /// a million on it.
    ///
    /// Dreams kept after this are warmed as they are written — see
    /// [`dreams::keep`].
    fn warm_dreams(&self, projected: &Projected) {
        let Some(group) = projected.builder.id_for_group(dreams::GROUP) else {
            return;
        };
        let warmed = self
            .engine
            .lock()
            .unwrap()
            .warm_group_normalization(projected.builder.schema(), group);
        tracing::info!("dreams: {warmed} dream(s) put on the normalized score band");
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
                    persona.building,
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
                // The same envelope `compile_act_loop` compiles, so what the prompt
                // shows a character is what the grammar will hold it to.
                let system = prompt::build(
                    persona,
                    mode,
                    &for_mode(mode),
                    &ToolCallEnvelope::for_dialect(&self.base_config.dialect),
                );
                let seq = self.engine.lock().unwrap().new_conversation(&system, cfg)?;
                // The dial applies with no schema to pin members in — it is the
                // turn's own, not the projection's.
                (seq, system, identity::deliberation(thinking))
            }
        };
        drop(guard);

        // The acts this turn can take, and only those — the same set, under the
        // same mode, the grammar below is compiled from. See `compile_act_loop`.
        let mut selection = selection;
        tools::show_within(&mut selection, Mode::Physical, within);
        let options = TurnOptions {
            turn_grammar: self.grammar_for(thinking, within),
            sampling: Some(self.sampling_for(thinking)),
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
    /// The turn's sampling, with this turn's think mode applied.
    ///
    /// **Each think mode has its own temperature and nucleus**, and the dial
    /// picks one per turn: the reasoning row for a mission that raises it, the
    /// cast's think-off row for [`identity::Deliberation::None`], which is what
    /// almost every turn runs. For this checkpoint the two rows differ: the
    /// preset runs both at `0.7 / 0.95`, and
    /// [`SamplingConfig::for_character_dialogue`] widens the cast's think-off
    /// row to `1.0 / 0.95` — so a character decodes hotter on an ordinary turn
    /// than when a mission has it reason.
    ///
    /// The two paths that decide a turn — the probe and the live act — both
    /// come through here, which is what keeps them from drifting.
    fn sampling_for(&self, thinking: identity::Deliberation) -> SamplingConfig {
        self.base_config
            .sampling
            .clone()
            .with_think_mode(thinking.mode(), self.base_config.max_response_tokens)
    }

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

    /// Retire every conversation this character has, so the next one it opens
    /// starts fresh instead of rejoining where it stopped.
    ///
    /// **Before the character has thought this process**, which is the only
    /// place it is called: the conversation it holds in memory is opened on its
    /// first turn, so at startup there is none to strand. The same retirement
    /// [`Self::open_conversation`] runs on everything it did not rejoin — with
    /// nothing kept — so there is one way a conversation is retired, not two.
    ///
    /// Tombstoned, not deleted: the turns become reclaimable and compaction
    /// takes them. The character's memory, beliefs and relationships are
    /// untouched; only the conversation it was carrying on is gone.
    ///
    /// Returns how many conversations it retired.
    pub fn forget_conversations(&self, npc_id: u64) -> usize {
        retire_superseded(&self.engine, npc_id, None)
    }

    /// Retire every dream this character has kept — see [`dreams::forget`].
    ///
    /// Returns how many it retired.
    pub fn forget_dreams(&self, npc_id: u64) -> usize {
        dreams::forget(&self.engine, npc_id)
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

    /// Stop a character and run one reflection — see [`crate::engine::reflect`].
    ///
    /// Goes through here rather than being built by the caller because the
    /// engine handle and the checkpoint's own [`SequenceConfig`] live here, and
    /// a reflection assembled from a second opinion about either would run the
    /// model outside the settings it was tuned under.
    ///
    /// **Touches none of this character's live state.** It does not open, read
    /// or disturb the conversation in [`Self::live`]: a reflection is its own
    /// throwaway timeline, and a character mid-decode is not waited on because
    /// there is nothing here they share.
    #[allow(clippy::too_many_arguments)]
    pub fn reflect(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        situation: &str,
        inner_thoughts: &str,
        feeling: &str,
        domain: &str,
        sampled_axes: &[String],
        on_reflection: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<reflect::Reflection> {
        // The mind's schema, so a reflection opens against the same sections a
        // live conversation does. Without it the two prompts are built by
        // different code and drift apart section by section — which is exactly
        // what had happened: the reflection's frame named none of the schema's
        // sections and carried none of its collections.
        let projected = self.projection.read().unwrap();
        let turns = self.reflection.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "this mind declares no `reflection` block, so there are no turns to send — \
                 author `reflection.question_one`, `.question_two` and `.question_two_retry` \
                 in projection.yaml"
            )
        })?;
        reflect::Reflect::new(Arc::clone(&self.engine), self.base_config.clone(), turns)
            .under(projected.as_ref())
            .run(
                npc_id,
                persona,
                mode,
                situation,
                inner_thoughts,
                feeling,
                domain,
                sampled_axes,
                on_reflection,
            )
    }

    /// Dream a brief and keep the dream — see [`crate::engine::dreams`].
    ///
    /// Two conversations, one after the other, neither of them this
    /// character's live one: the dream is decoded in a throwaway conversation
    /// and written, a line to a turn, into one of its own on the dream layer.
    /// Nothing here touches [`Self::live`], so a character mid-decode is not
    /// waited on.
    pub fn dream(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        brief: &str,
        assumption: &str,
    ) -> anyhow::Result<dreams::Kept> {
        let projected = self.projection.read().unwrap();
        let p = projected
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no schema, so no dream layer to keep a dream in"))?;
        let story = dreams::dream(&self.engine, &self.base_config, p, npc_id, persona, brief)?;
        dreams::keep(
            &self.engine,
            &self.base_config,
            p,
            npc_id,
            assumption,
            &story,
        )
    }

    /// A sample of the axes this character has dreamt along — see
    /// [`dreams::sample_axes`].
    pub fn dreamt_axes(&self, npc_id: u64, n: usize) -> Vec<String> {
        dreams::sample_axes(&self.engine, npc_id, n)
    }

    /// How many dreams this character has kept.
    pub fn dreams_kept(&self, npc_id: u64) -> usize {
        dreams::count(&self.engine, npc_id)
    }

    /// Whether this daemon can reflect at all: a mind that authors the
    /// questions, and a schema to ask them under.
    pub fn can_reflect(&self) -> bool {
        self.reflection.is_some() && self.projection.read().unwrap().is_some()
    }

    /// Rejoin this character's conversation, or open it a new one.
    ///
    /// # Rejoin before mint
    ///
    /// A character's conversation is named for `(npc_id, day)` and tagged with a
    /// fingerprint of the frame it was opened under. So the question a restart
    /// asks is answerable from the log alone: *is there a live conversation with
    /// this name, written under this frame?* If there is, the character carries
    /// on inside it — its turns, its recurrent state and its carried selection
    /// belief all come back — and if there is not, one is minted and tagged so
    /// the next restart can ask the same question.
    ///
    /// Before this, the answer was always no, because nothing was ever asked: a
    /// conversation open always minted a fresh timeline, so the one a previous
    /// process was using was abandoned the moment this one started, and a
    /// character that had lived all morning woke with no memory of it beyond
    /// what the gather happened to retrieve. Measured: 128 timelines for three
    /// characters on one day — roughly forty abandoned generations, each a full
    /// conversation's K/V left live and unreachable, which was the bulk of a
    /// 132 GB log.
    ///
    /// # Why the frame is checked and not just the name
    ///
    /// A conversation is only worth continuing under the frame that wrote it.
    /// Change the prompt, the act catalog or the call syntax and the history on
    /// disk is a history *this* character would never have produced — the model
    /// reads its own past making calls in a shape it can no longer make, which
    /// is worse than reading nothing. So a name match with a frame mismatch is
    /// superseded rather than rejoined, and says so. This is the same discipline
    /// zend's ingest cache runs on its `content_sha256` tag.
    ///
    /// # Order
    ///
    /// The rejoin is attempted **before** anything is retired, so what gets
    /// superseded is decided by the conversation we are actually holding rather
    /// than the one we hoped to hold. A resume that fails is therefore retired
    /// along with the rest, instead of being left live to fail again on every
    /// restart forever.
    fn open_conversation(
        &self,
        npc_id: u64,
        day: u64,
        persona: &Persona<'_>,
        mode: Mode,
    ) -> anyhow::Result<Live> {
        let id = conversation_id(npc_id, day);
        let mut cfg = self.base_config.clone();
        cfg.context_window_turns = CONTEXT_WINDOW_TURNS;
        // **The mind's own schema, when there is one.**
        //
        // This was a rendered copy for a long time, and the comment here said
        // why: under the schema the checkpoint opened `<think>` and never closed
        // it, so a turn ran to `max_response_tokens` and was discarded whole as
        // reasoning — a decode that reasons forever is a successful decode, so
        // every tick produced no acts and logged no error. Three suppressions
        // were tried and none held: the `no_think` glue, `thinking_effort: off`,
        // and the `ThinkMode::Off` close budget in the sampling config.
        //
        // **None of those three is the mechanism that works.** All are requests —
        // a marker this family does not honour, a selector, and a budget that
        // caps a runaway rather than preventing one. What holds is a steering
        // stencil bound to the decoded `<think>` token, which makes the block
        // unrepresentable rather than discouraged; `compile_act_loop` carries it
        // and [`Self::grammar_for`] has been putting it on every turn since.
        // The rendered prompt was working around a fault that had already been
        // fixed underneath it.
        //
        // Falling back to the rendered copy when there is no mind is not a dual
        // path kept alive for safety — a daemon with no `projection.yaml` has no
        // schema to open against, and the copy is the only prompt there is.
        let under = self.projection.read().unwrap();
        let (system, projected) = match under.as_ref() {
            Some(p) => (p.prompt.clone(), Some(p)),
            // The same envelope `compile_act_loop` compiles, so what the prompt
            // shows a character is what the grammar will hold it to.
            None => (
                prompt::build(
                    persona,
                    mode,
                    &for_mode(mode),
                    &ToolCallEnvelope::for_dialect(&self.base_config.dialect),
                ),
                None,
            ),
        };
        let frame = match projected {
            Some(p) => p.frame.clone(),
            None => frame_fingerprint(&system),
        };

        let rejoined = resumable(&self.engine, &id, &frame).and_then(|timeline| {
            let engine = self.engine.lock().ok()?;
            // Resumed against the same schema it was opened under. A conversation
            // minted on the projection and rejoined on the synthetic one would
            // carry a different prompt under the same fingerprint, which is the
            // one thing the fingerprint exists to make impossible.
            let resumed = match projected {
                Some(p) => engine.resume_conversation_with_projection(
                    timeline,
                    &system,
                    dreams::scoped(&p.builder, npc_id, dreams::IN_ACTING),
                    cfg.clone(),
                ),
                None => engine.resume_conversation(timeline, &system, cfg.clone()),
            };
            match resumed {
                Ok(sequence) => {
                    // Both read the substrate, so both are answered while the
                    // lock the resume needed is still in hand.
                    let depth = engine.timeline_turn_count(timeline);
                    let watermark = retention::resume_watermark(&engine, timeline, depth);
                    drop(engine);
                    tracing::info!(
                        depth,
                        watermark,
                        "conversation {id} rejoined — the character carries on from where it \
                         stopped rather than starting over beside its own history"
                    );
                    Some((sequence, watermark))
                }
                Err(e) => {
                    drop(engine);
                    tracing::warn!(
                        "conversation {id} could not be rejoined: {e:?} — it is superseded below \
                         and a fresh one is opened in its place"
                    );
                    None
                }
            }
        });

        // **Retire everything else this character left behind.**
        //
        // Its conversations are all named `npc-<id>-day-<n>`, so its whole
        // history in the log is exactly the timelines carrying that prefix —
        // yesterday's, and every generation a previous process abandoned. None
        // of them will be read again now that today's is in hand.
        //
        // The same move zend makes when it re-ingests a file and supersedes the
        // old conversation (`code_read`): tombstone it, and compaction drops its
        // records wholesale.
        retire_superseded(
            &self.engine,
            npc_id,
            rejoined.as_ref().map(|(s, _)| s.timeline_id()),
        );

        if let Some((sequence, watermark)) = rejoined {
            return Ok(Live {
                sequence,
                day,
                id,
                // Not zero: a conversation this deep has already retired its
                // tail, and re-walking it from the bottom would cost the first
                // few hundred turns of the restart. See
                // [`retention::resume_watermark`].
                retired_through: watermark,
                pending: Vec::new(),
            });
        }

        let sequence = {
            let engine = self.engine.lock().unwrap();
            match projected {
                // Its own dreams and nobody else's, a few lines deep — a turn
                // in the room is reminded of one when something resonates.
                Some(p) => engine.new_conversation_with_projection(
                    &system,
                    dreams::scoped(&p.builder, npc_id, dreams::IN_ACTING),
                    p.layer,
                    p.group,
                    cfg,
                )?,
                None => engine.new_conversation(&system, cfg)?,
            }
        };
        let timeline = sequence.timeline_id();
        {
            let engine = self.engine.lock().unwrap();
            // **The derived id, given to the substrate.** Minting it and keeping
            // it in this struct made it a log label and nothing else: the
            // timeline went into the redo log anonymous, so nothing downstream
            // could attribute a day's turns to the character that lived them.
            if let Err(e) = engine.set_conversation_conv_id(timeline, &id) {
                tracing::warn!(
                    "conversation {id} could not be named in the log: {e:?} — its turns will not \
                     be attributable to this character"
                );
            }
            // **What makes the next restart able to find this again.**
            //
            // The fingerprint is the shape and the name is what is searched on,
            // and they are written in that order deliberately: the name is the
            // commit marker, so a conversation `resumable` can find is one whose
            // fingerprint is already down. Written the other way, a crash
            // between the two writes would leave a conversation findable by name
            // with nothing to check it against — which is the one state that
            // would have to be guessed about rather than decided.
            //
            // A conversation missing either tag is simply never rejoined, which
            // is the behaviour there was before this. A failed tag therefore
            // costs continuity, not correctness.
            for (key, value) in [
                (META_FRAME, frame.as_str()),
                (META_CONVERSATION, id.as_str()),
            ] {
                if let Err(e) = engine.set_conversation_metadata(timeline, key, value) {
                    tracing::warn!(
                        "conversation {id} could not be tagged {key}: {e:?} — a restart will open \
                         a new conversation rather than rejoining this one"
                    );
                }
            }
        }
        Ok(Live {
            sequence,
            day,
            id,
            retired_through: 0,
            pending: Vec::new(),
        })
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
                Entry::Vacant(slot) => Arc::clone(slot.insert(Arc::new(Mutex::new(
                    self.open_conversation(npc_id, day, persona, mode)?,
                )))),
            }
        };

        // **The answers to last turn's acts, then everything that has happened
        // since — one message.**
        //
        // The results ride at the head because the protocol puts them there:
        // what a character did is answered before the world is allowed to speak
        // again. A fat batch is one better-informed thinking step rather than
        // several thrashing ones — the mind design is explicit that this is
        // what a busy character should get.
        let answers = std::mem::take(&mut conversation.lock().unwrap().pending);
        let perception = compose(&answers, events, window);
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
        let mut selection = self
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
                        persona.building,
                        identity::Deliberation::default(),
                    )
            })
            .unwrap_or_else(|| identity::deliberation(identity::Deliberation::default()));
        // **What it can do, shown beside what it is.** Every act is installed in
        // the schema's `tools` collection and a turn names the ones it can take
        // — the same set, under the same mode, `grammar_for` compiles the mask
        // from — so the list a character reads is never wider or narrower than
        // what it can decode. A character alone reads no `tell`; one standing at
        // a chronicle terminal reads the terminal's acts until it walks away.
        tools::show_within(&mut selection, Mode::Physical, within);
        let options = TurnOptions {
            turn_grammar: self.grammar_for(identity::Deliberation::default(), within),
            sampling: Some(self.sampling_for(identity::Deliberation::default())),
            selection,
            assistant_prefill: self.opening(identity::Deliberation::default()),
            // **Inside its own dreams' scope.** A turn teaches the hit levels of
            // the scopes its tags name, and the dream group is scoped to this
            // character's tag — so without it a character's own turns, the only
            // traffic its dreams are ever read against, would teach them
            // nothing. See `Minds::warm_dreams`.
            tags: vec![dreams::tag(npc_id)],
            ..Default::default()
        };
        let (response, timeline, watermark) = {
            let mut live = conversation.lock().unwrap();
            let response = live.sequence.send_turn_with_options(&perception, options)?;
            // **Whether a dream reached this turn**, said in the log whenever
            // one did. A dream is only worth having if it comes back, and it
            // comes back by winning the gather rather than by being pasted in —
            // so the only way to see it is to ask the projection what it held.
            if let Some(ev) = live.sequence.projection_event(&response.stats) {
                let recalled: Vec<(u64, f32)> = ev
                    .selection
                    .turns
                    .iter()
                    .filter(|t| t.layer == dreams::LAYER && t.selected)
                    .filter_map(|t| Some((t.timeline?, t.score)))
                    .collect();
                if !recalled.is_empty() {
                    let from: BTreeSet<u64> = recalled.iter().map(|(tl, _)| *tl).collect();
                    // The scores too: whether a line came back because it
                    // resonated or only because the group had room is the one
                    // thing a gate on this layer would be set from.
                    let scores: Vec<String> =
                        recalled.iter().map(|(_, s)| format!("{s:.0}")).collect();
                    tracing::info!(
                        "npc {npc_id}: this turn was reminded of {} line(s) from {} of its \
                         dream(s), scored {}",
                        recalled.len(),
                        from.len(),
                        scores.join("/")
                    );
                }
            }
            (response, live.sequence.timeline_id(), live.retired_through)
        };
        // **Retire whatever now sits below the horizon.**
        //
        // On the insert rather than on a timer: a turn is the only thing that
        // pushes another one out, so the moment one lands is exactly when the
        // question has a new answer. The sweep looks *back* rather than
        // assuming one turn fell out since last time, which is what makes a
        // failed write or a restarted daemon catch up instead of stranding
        // those turns below the horizon for good.
        //
        // The conversation's own lock is released first: retiring takes the
        // engine lock to write, and holding both would put every other
        // character's decode behind this one's bookkeeping.
        if let Some(keep) = self.keep_turns {
            if let Ok(engine) = self.engine.lock() {
                // **The substrate's count, not the sequence's.**
                //
                // `Sequence::turn_count` counts a user and an assistant message
                // separately and starts from zero whenever a process opens a
                // fresh sequence; turn *indices* advance once per exchange. A
                // horizon from that counter therefore runs at twice the rate of
                // the turns it indexes — measured live, `turn_count=65` gave
                // `horizon=0` and by 65 exchanges the horizon had passed the
                // last real turn, retiring the whole conversation. `keep_turns:
                // 64` came to mean keep nothing.
                let depth = engine.timeline_turn_count(timeline);
                let (through, retired) =
                    retention::retire_expired(&engine, timeline, depth, keep, watermark);
                drop(engine);
                conversation.lock().unwrap().retired_through = through;
                thought.retired = retired;
            }
        }
        thought.parsed = act::parse(&response.text);
        Ok(thought)
    }

    /// Hand back what became of the acts this character just called for.
    ///
    /// One entry per call it made, in the order it made them, whatever the
    /// verdict — an act that was refused, or a call that was malformed, needs
    /// an answer at least as much as one that worked. They ride at the head of
    /// this character's next turn as `<tool_response>` blocks; see [`compose`].
    ///
    /// Called after the acts have been enacted, because that is when their
    /// outcomes exist. A character with no live conversation is a no-op rather
    /// than an error: it has no turn for them to ride on, so there is nothing
    /// to hold them against.
    pub fn deliver_outcomes(&self, npc_id: u64, outcomes: Vec<String>) {
        if outcomes.is_empty() {
            return;
        }
        // Cloned out of the map before locking the conversation, so a character
        // mid-decode is waited on without the map held behind it.
        let live = self.live.lock().unwrap().get(&npc_id).map(Arc::clone);
        if let Some(l) = live {
            l.lock().unwrap().pending.extend(outcomes);
        }
    }

    /// Retire a character's conversation, because the character is gone.
    ///
    /// **On delete only, and deliberately not on shutdown.** A tombstone is the
    /// record that a conversation is finished, and a daemon stopping is not that
    /// — the cast is meant to still be there when it starts again. Retiring here
    /// would make [`Minds::open_conversation`]'s rejoin unreachable, since the
    /// lookup it runs on excludes tombstoned timelines by design.
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

/// A fingerprint of the frame a character thinks under.
///
/// The fingerprint of a conversation's whole frame, which is everything that
/// decides what a turn in it looks like: who the character is, which acts
/// exist, what a call to one is spelled as. Change any of them and a history
/// written before the change is a history the character can no longer produce —
/// see [`Minds::open_conversation`] for why that is a reason not to rejoin it.
///
/// `frame` is the rendered prompt for a daemon with no mind, and
/// [`crate::engine::schema::frame`] under the projection.
///
/// Content-addressed rather than versioned, so nothing has to remember to bump a
/// number when it edits a prompt. That is the property zend's ingest cache
/// relies on for the same job.
pub fn frame_fingerprint(frame: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(frame.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// The live conversation named `id` and written under `frame`, if there is one.
///
/// The lookup is indexed and already excludes tombstoned timelines, so "a
/// conversation that was retired" and "no conversation" are the same answer
/// here — which is what makes retiring one the way to refuse a resume.
///
/// A name match under a *different* frame is passed over and logged, because
/// that is the case an operator needs to see: it looks identical from outside to
/// a character that simply had no history, and the reason it had none is
/// something they just changed.
///
/// More than one candidate should not happen — the name is derived from
/// `(npc_id, day)` and every open supersedes the rest — so the deepest is taken
/// rather than the first, which makes the choice deterministic instead of
/// dependent on lookup order. The others are retired by
/// [`retire_superseded`] on the way past.
fn resumable(engine: &Arc<Mutex<ConversationEngine>>, id: &str, frame: &str) -> Option<TimelineId> {
    let engine = engine.lock().ok()?;
    let mut best: Option<(u64, TimelineId)> = None;
    for timeline in engine.find_conversations_by_metadata(META_CONVERSATION, id) {
        match engine
            .conversation_metadata(timeline)
            .and_then(|m| m.get(META_FRAME).cloned())
        {
            Some(under) if under == frame => {}
            Some(_) => {
                tracing::info!(
                    "conversation {id} was written under a different frame — superseded rather \
                     than rejoined, because a history this character could no longer produce \
                     reads worse than no history at all"
                );
                continue;
            }
            // Named but never fingerprinted, which the tag order makes
            // unreachable: the name is written second precisely so that finding
            // one means the fingerprint is already down. Reported rather than
            // folded in with a mismatch, because the two mean different things —
            // a mismatch is a prompt that changed, this is a write that tore.
            None => {
                tracing::warn!(
                    "conversation {id} carries a name but no frame fingerprint — superseded, \
                     since there is nothing to decide against"
                );
                continue;
            }
        }
        let depth = engine.timeline_turn_count(timeline);
        if best.is_none_or(|(deepest, _)| depth > deepest) {
            best = Some((depth, timeline));
        }
    }
    best.map(|(_, timeline)| timeline)
}

/// Retire every conversation this character has left behind, except `keep`.
///
/// **A character's conversations are named `npc-<id>-day-<n>`** by
/// [`conversation_id`], so its whole history in the log is exactly the
/// timelines carrying that prefix. Opening today's supersedes all of them —
/// yesterday's, and every generation a previous process abandoned.
///
/// `keep` is the one this process just rejoined, and is the only conversation
/// that survives. It is passed as the timeline rather than looked up by name
/// because the two can disagree: a resume that *failed* has a matching name and
/// must still be retired, or it stays live and fails again on every restart.
///
/// The prefix, rather than the metadata tag [`resumable`] searches on: every
/// conversation ever opened has a `conv_id`, where only those opened since the
/// tag existed carry the tag. Sweeping on the name is what lets this retire a
/// backlog it did not write.
///
/// The lookup is prefix-scoped in the substrate rather than a filter over
/// [`ConversationEngine::known_conversations`], which materialises every
/// conversation the workspace has ever held — tombstoned included, since that
/// call feeds a sidebar. Retirement bounds the live set, not the registry, so
/// that list only grows; this one is the size of one character's history.
///
/// Already-tombstoned timelines are excluded by the lookup, which is what makes
/// this affordable to run on every conversation open rather than once at boot:
/// a character woken for the first time today retires the whole backlog, and
/// every wake after that finds nothing to do.
///
/// Failure is logged, never propagated — a tombstone that did not land leaves
/// bulk on disk, and refusing to let a character think over its bookkeeping
/// would be far worse.
///
/// Returns how many it retired.
fn retire_superseded(
    engine: &Arc<Mutex<ConversationEngine>>,
    npc_id: u64,
    keep: Option<TimelineId>,
) -> usize {
    let prefix = format!("npc-{npc_id}-day-");
    let Ok(engine) = engine.lock() else {
        return 0;
    };
    let stale: Vec<(TimelineId, String)> = engine
        .conversations_with_conv_id_prefix(&prefix)
        .into_iter()
        .filter(|(tl, _)| Some(*tl) != keep)
        .collect();
    if stale.is_empty() {
        return 0;
    }
    let mut retired = 0;
    for (tl, conv) in &stale {
        match engine.tombstone_timeline(*tl) {
            Ok(()) => retired += 1,
            Err(e) => tracing::warn!(
                "superseded conversation {conv} could not be retired: {e:?} — its turns stay \
                 live in the log and nothing will ever read them"
            ),
        }
    }
    if retired > 0 {
        tracing::info!(
            retired,
            "retired conversations this character had left behind; their turns are now \
             reclaimable"
        );
    }
    retired
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

/// What the character reads this tick: the answers to what it did, then what
/// has happened since.
///
/// # The shape, and why it is this shape
///
/// A character calls acts and the world answers them. The template that answer
/// arrives in is not ours to choose — Qwen and Hermes both return a result in
/// the **user** half of the next turn, wrapped in `<tool_response>`, one block
/// per call, and the model is trained to read that wrapper as "this is what
/// came back" rather than as something a person said.
///
/// So the turn is built in two parts, in this order:
///
/// ```text
/// <tool_response>
/// You moved to the sorting room.
/// </tool_response>
///
/// The air near the door is noticeably fresher than the air by the wall.
/// ```
///
/// The results come **first** because they answer the turn before, and the
/// world's own events follow as ordinary prose. Reversing them would put the
/// world's voice between a call and its answer, which is the one arrangement
/// every tool-calling protocol forbids.
///
/// Before this, results were never returned at all: measured over 23 turns of
/// three characters, the substrate held 23 calls and zero responses. A
/// character acted and the next thing it read was the weather. Every one of
/// those 23 acts was `reflect` — with no act ever visibly causing anything,
/// there was nothing to prefer about acting over thinking.
///
/// # The window
///
/// Still *not* pasted in: the sequence carries its own bounded tail
/// (`context_window_turns`) and the substrate carries the rest, so repeating
/// the window here would put the same turns in the context twice — once
/// verbatim from us and once from the sequence's own history — and teach the
/// model that everything happens twice.
fn compose(answers: &[String], events: &[Event], window: &Window) -> String {
    let _ = window;
    let mut s = String::new();
    for a in answers {
        // The block is newline-delimited inside the tags because an outcome is
        // a sentence, not a JSON scalar — the same allowance zend's
        // `format_tool_responses` makes for an already-rendered result.
        s.push_str("<tool_response>\n");
        s.push_str(a.trim());
        s.push_str("\n</tool_response>\n");
    }
    // One blank line between the answers and the world, so the two are visibly
    // different kinds of thing rather than one run-on block.
    if !answers.is_empty() && !events.is_empty() {
        s.push('\n');
    }
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

    /// The fingerprint is what decides whether a conversation on disk is still
    /// this character's to continue, so the same frame has to produce the same
    /// answer in a process that never saw the one that wrote it. A hash of the
    /// text does; anything carrying a pointer, an ordering or a timestamp would
    /// not, and would fail by quietly never resuming.
    #[test]
    fn the_same_frame_fingerprints_the_same_in_any_process() {
        let frame = "You are Ada. You may move_to, say, ask.";
        assert_eq!(frame_fingerprint(frame), frame_fingerprint(frame));
        assert_eq!(
            frame_fingerprint(frame).len(),
            64,
            "not a sha256 hex digest"
        );
    }

    /// **Every difference has to count.** A character rejoining a conversation
    /// written under a prompt it no longer has reads its own past making calls
    /// in a shape it can no longer make. The cases below are the ones that
    /// actually change between builds: who the character is, and which acts
    /// exist.
    #[test]
    fn a_changed_frame_fingerprints_differently() {
        let base = "You are Ada. You may move_to, say, ask.";
        for changed in [
            "You are Bram. You may move_to, say, ask.",
            "You are Ada. You may move_to, say, ask, gesture.",
            "You are Ada. You may move_to, say, ask. ",
            "",
        ] {
            assert_ne!(
                frame_fingerprint(base),
                frame_fingerprint(changed),
                "{changed:?} fingerprinted the same as the frame it differs from"
            );
        }
    }

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
        let s = compose(&[], &[ev("the gate opens"), ev("someone shouts")], &w);
        assert_eq!(s, "the gate opens\n\nsomeone shouts");
    }

    /// **An act is answered, and the answer comes first.**
    ///
    /// The protocol admits no other order: a result answers the turn before it,
    /// so nothing may come between a call and its response. Measured before
    /// this existed — 23 calls, 0 responses, and all 23 acts `reflect`.
    #[test]
    fn an_answer_leads_the_turn_and_the_world_follows_it() {
        let w = Window::with_default_cap();
        let s = compose(
            &["You moved to the sorting room.".to_string()],
            &[ev("the air near the door is fresher")],
            &w,
        );
        assert_eq!(
            s,
            "<tool_response>\nYou moved to the sorting room.\n</tool_response>\n\n\
             the air near the door is fresher"
        );
    }

    /// One block per call, in the order the character called them — so a
    /// character reading them back can tell which answer belongs to which act.
    #[test]
    fn every_call_gets_its_own_block_in_call_order() {
        let w = Window::with_default_cap();
        let s = compose(
            &[
                "You moved to the sorting room.".to_string(),
                "Nobody there answered to that name.".to_string(),
            ],
            &[],
            &w,
        );
        assert_eq!(s.matches("<tool_response>").count(), 2);
        assert_eq!(s.matches("</tool_response>").count(), 2);
        assert!(
            s.find("You moved").unwrap() < s.find("Nobody there").unwrap(),
            "answers must keep the order the acts were called in: {s}"
        );
    }

    /// A refusal is an answer too — the only signal that separates "that did
    /// not work" from "nothing happened", and the two want different next acts.
    #[test]
    fn a_refusal_is_returned_like_any_other_answer() {
        let w = Window::with_default_cap();
        let s = compose(
            &["You cannot reach the vault from here.".to_string()],
            &[],
            &w,
        );
        assert!(s.starts_with("<tool_response>\n"));
        assert!(s.contains("cannot reach the vault"));
    }

    /// With nothing to report the turn is exactly what it always was — no
    /// empty wrapper for the model to read as a result that never came.
    #[test]
    fn a_turn_with_no_acts_behind_it_carries_no_wrapper() {
        let w = Window::with_default_cap();
        let s = compose(&[], &[ev("it rains")], &w);
        assert_eq!(s, "it rains");
        assert!(!s.contains("tool_response"));
    }

    /// An answer with no world event behind it still stands on its own — a
    /// character that acted during a quiet moment is told what happened.
    #[test]
    fn an_answer_alone_does_not_trail_a_blank_line() {
        let w = Window::with_default_cap();
        let s = compose(&["You put it down.".to_string()], &[], &w);
        assert_eq!(s, "<tool_response>\nYou put it down.\n</tool_response>\n");
    }

    /// **The window must not be pasted in.** The sequence carries its own
    /// bounded tail, so repeating it here would put the same turns in the
    /// context twice and teach the model that everything happens twice.
    #[test]
    fn the_window_is_not_repeated_into_the_message() {
        let mut w = Window::with_default_cap();
        w.push_world("something that already happened", 0, None);
        w.push_npc("and what I did about it", 0);
        let s = compose(&[], &[ev("something new")], &w);
        assert_eq!(s, "something new");
        assert!(!s.contains("already happened"));
    }

    #[test]
    fn an_empty_batch_composes_to_nothing() {
        assert_eq!(compose(&[], &[], &Window::with_default_cap()), "");
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
