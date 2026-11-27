//! A character's live conversation, and the decode that turns its perception
//! into acts.
//!
//! Each character holds one conversation per day, named from its id and the day
//! and recorded on the substrate. A restart rejoins that conversation when it
//! was written under the frame the process now runs, so its turns and carried
//! state come back. A day's rollover tombstones the conversation, which leaves
//! its turns reachable by the gather but no longer selected by default. Each
//! conversation sits behind its own async lock, held across a decode, so one
//! turn per character runs at a time while the rest of the cast decodes beside
//! it. The sequence is windowed to a bounded tail of turns, because the
//! substrate holds the rest and the gather decides what is relevant.
//!
//! - `conversation`: the day's conversation, its rejoin and its retirement.
//! - `journaling`: writing, folding and restoring the journal.
//! - `mind_schema`: the frame a turn is held under and its act grammar.
//! - `perception`: what a turn reads back, answers first.
//! - `reflecting`: reflections and dreams, each in its own conversation.
//! - `think`: one thinking step, perception in and acts out.
//! - `tests_identity`: the fingerprint and the day's conversation name.
//! - `tests_perception`: the order and bounds of what a turn reads.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arc_swap::ArcSwapOption;
use papaya::HashMap as LockFreeMap;
use tokio::sync::Mutex as ConversationLock;

use candle_conversation::projection::{
    Builder, Conversation, DecodePriority, GroupId, LayerId, SelectedTurn, SelectionState,
    TimelineId,
};
use candle_conversation::stencil::{
    compile_action_loop, compile_think_tree, StencilTree, ThinkMode, ThinkSteerEnvelope,
    ToolCallEnvelope, ToolSpec, TreeSpec, TurnGrammar,
};
use candle_conversation::throwaway::Throwaway;
use candle_conversation::{
    Carried, Changes, ConversationEngine, Linking, PromptDump, SamplingConfig, Sequence,
    SequenceConfig, Sharing, TurnOptions, UnsealedAsker, Want,
};
use journal_core::folding::{self, Folded};
use journal_core::journal::Holds;
use journal_core::section::{self, entry_section, JournalPrompt, Kept, META_KEPT};
use journal_core::verify::World;
use journal_core::workflow::{self, Outcome, Skip, Timing};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::engine::act::{self, Parsed};
use crate::engine::bearings::Scopes;
use crate::engine::dreams;
use crate::engine::event::Event;
use crate::engine::identity;
use crate::engine::invoke_body::Invokable;
use crate::engine::journal::desk::{EngineDesk, FOLD_TOKENS, WRITE_TOKENS};
use crate::engine::journal::section::META_OF;
use crate::engine::journal::state::JournalState;
use crate::engine::journal::style::NPC;
use crate::engine::layers;
use crate::engine::narration;
use crate::engine::narrator;
use crate::engine::prompt::{self, Persona};
use crate::engine::reflect;
use crate::engine::rendered;
use crate::engine::retention;
use crate::engine::schema::ReflectionTurns;
use crate::engine::sleep::conversation_id;
use crate::engine::talk::templates;
use crate::engine::tick::Due;
use crate::engine::tools::{self, for_mode, Catalog, Mode};
use crate::engine::window::Speaker;

mod ask;
mod talk;
mod voice;
mod words;
pub use ask::{Answer, Put};
pub use talk::{Spoken, Talk, TalkTurn};
pub use voice::Voicing;

/// A character's live conversation.
struct Live {
    sequence: Sequence,
    /// The mission section this conversation holds, if any. Reconciled against
    /// the character's mission at the head of every turn — see
    /// [`Minds::prepare`].
    mission: Carried,
    /// The journal sections this conversation holds, reconciled the same way
    /// and also whenever an entry is kept — see [`Minds::keep_journal`].
    journal: Carried,
    /// The channel sections this conversation holds, reconciled at the head
    /// of every turn against the channels the character is on.
    channels: Carried,
    /// The journal record last written to this conversation's metadata, so an
    /// unchanged journal is not written again.
    journal_record: String,
    /// The day this conversation belongs to. A mismatch against the world's day
    /// is what triggers the roll-over.
    day: u64,
    /// The chapter — missions closed — it belongs to. A mission handed in rolls
    /// it over the same way.
    chapter: u64,
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
    engine: Arc<ConversationEngine>,
    /// The substrate, taken when the minds were built. Status and the host's
    /// substrate reads use this and do not take `engine`: that mutex is held
    /// across scheduler waits, and a `/v1/status` poll blocked on it is the
    /// console sitting on "Reconnecting".
    substrate: Conversation,
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
    acts: LockFreeMap<(identity::Deliberation, Catalog, tools::Within), Arc<StencilTree>>,
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
    ///
    /// An atomic slot: a reader loads the `Arc` and holds it across any await
    /// without holding anything else.
    projection: ArcSwapOption<Projected>,
    /// What each bench draws on of the shared layers — `None` until the mind
    /// declares bearings, and then every shared layer is drawn on everywhere.
    scopes: ArcSwapOption<Scopes>,
    /// Each character's conversation, behind its own async lock so a decode
    /// holds only the character that is thinking — see the module's *Locking*
    /// note.
    ///
    /// Each value is a slot, emptied the moment its entry is removed or
    /// replaced: the map frees what it retires only in batches, and a retired
    /// entry still holding its `Resident` would keep the conversation — its
    /// scheduler slot and its resident KV — alive long after the character let
    /// it go.
    live: LockFreeMap<u64, ArcSwapOption<Resident>>,
    /// The system prompt each character's latest turn was made under, laid out
    /// from that turn's projection. Held apart from the conversation so reading
    /// it never waits on a decode.
    system_prompts: LockFreeMap<u64, rendered::Rendered>,
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
    /// Each character's opening lock, held while its conversation is opened or
    /// replaced — see [`Minds::resident_for`].
    openings: LockFreeMap<u64, Arc<ConversationLock<()>>>,
    /// The trials in flight on each character — see [`Minds::trial_own`].
    trials_in_flight: LockFreeMap<u64, Arc<AtomicUsize>>,
    /// The selection a character's trials read, with the timeline of the
    /// conversation it was prepared on.
    trial_selections: LockFreeMap<u64, (TimelineId, SelectionState)>,
}

/// A character's live conversation, and the handle that can put it a question
/// without its lock, so asking never waits for the character's turn — see
/// [`Minds::ask_under`]. One entry, so the two are replaced together.
struct Resident {
    conversation: ConversationLock<Live>,
    asker: UnsealedAsker,
}

impl Minds {
    pub fn new(engine: Arc<ConversationEngine>, base_config: SequenceConfig) -> Self {
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
        let substrate = {
            let e = &engine;
            let max = base_config.max_response_tokens;
            base_config
                .sampling
                .apply_think_mode(ThinkMode::Off, e.tokenizer(), max);
            e.conversation()
        };
        // The cast decodes on the checkpoint's own published sampling. An earlier
        // "character dialogue" tuning widened the think-off row and layered on
        // DRY, presence and a cross-turn penalty to fight repetition from the
        // sampler; it read as stilted on every prose path that borrowed it (the
        // narrator among them, whose first token it occasionally misfired) and
        // never reliably did its job. Repetition is the loop guard's to break, at
        // the grammar, where a repeated act is struck from the turn's own stencil.
        // Compiled at boot for the empty room, which proves the catalog builds
        // at all and warms the cache for the commonest case. A checkpoint whose
        // tokenizer lacks the markers yields nothing and turns free-decode
        // exactly as before — loud, but never fatal: a daemon that refuses to
        // boot over it helps nobody.
        let acts = LockFreeMap::new();
        let mut ok = true;
        for level in LEVELS {
            let nowhere = tools::Within::nowhere();
            match compile_act_loop(&engine, &base_config, level, Catalog::ROOM, &nowhere) {
                Ok(tree) => {
                    acts.pin().insert((level, Catalog::ROOM, nowhere), tree);
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
        // Also prove the grammar compiles for a character standing at a station.
        // The command table's mission acts now reach the world through the
        // effector device (`tools::routed`), so they no longer enter the mask;
        // what this catches is a *station* grammar that would not compile — an
        // `AtPart` act still in the catalogue whose branch is malformed —
        // which `nowhere()` never carries and so the empty-room check above
        // cannot see. It would otherwise surface only when a character first
        // stands there, as a `<think>` block with no tree to close it. Logged
        // loudly, not fatal: the per-turn `opening` gate makes such a turn
        // free-decode cleanly rather than run to the token ceiling.
        let mut at_table = tools::Within::nowhere();
        at_table.station = vec!["order-table".to_string()];
        at_table.invokable = vec![Invokable::new(
            "http://local/command/order-table~0/report_done",
            "report_done",
        )];
        for level in LEVELS {
            if let Err(e) = compile_act_loop(&engine, &base_config, level, Catalog::ROOM, &at_table)
            {
                tracing::error!(
                    "the {level:?} grammar at the command table would not compile: {e:#} — \
                     characters there will free-decode"
                );
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
            substrate,
            base_config,
            acts,
            grammar_ok: ok,
            projection: ArcSwapOption::empty(),
            scopes: ArcSwapOption::empty(),
            live: LockFreeMap::new(),
            system_prompts: LockFreeMap::new(),
            keep_turns: None,
            reflection: None,
            openings: LockFreeMap::new(),
            trials_in_flight: LockFreeMap::new(),
            trial_selections: LockFreeMap::new(),
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

    /// This character's live conversation, if it has one — cloned out of the
    /// map, so holding it holds nothing else.
    fn resident_of(&self, npc_id: u64) -> Option<Arc<Resident>> {
        self.live.pin().get(&npc_id)?.load_full()
    }

    /// Decode this character's next turns at `band`, or with `None` at its
    /// conversation's own. A character carrying out what a person asked of it
    /// is waited on as the phone turn that took the request was, and its turn
    /// queued with the rest of the cast left that person waiting minutes for
    /// a reading that takes seconds. Nothing to set for a character with no
    /// conversation open: its first turn opens one.
    pub async fn set_band(&self, npc_id: u64, band: Option<DecodePriority>) {
        if let Some(resident) = self.resident_of(npc_id) {
            resident.conversation.lock().await.sequence.set_band(band);
        }
    }

    /// Take this character's conversation out of the map, emptying the entry's
    /// slot so the map's retired copy holds nothing — see [`Self::live`].
    fn release(&self, npc_id: u64) -> Option<Arc<Resident>> {
        self.live.pin().remove(&npc_id)?.swap(None)
    }

    /// The tokens this character would be given if it decoded this instant —
    /// the projection recomputed now, piece by piece, with each piece's scores.
    /// Waits for a decode in progress. `None` for a character with no live
    /// conversation.
    pub async fn prompt_dump(&self, npc_id: u64) -> Option<PromptDump> {
        let resident = self.resident_of(npc_id)?;
        let held = resident.conversation.lock().await;
        held.sequence.prompt_dump()
    }

    /// The engine, for work that is not a character thinking.
    ///
    /// [`crate::lifegen`] primes its own conversations and forks them; it is not
    /// a character taking a turn, so it does not go through [`Self::think`]. The
    /// handle is shared rather than a second engine because there is one card
    /// and one scheduler, and the generator's forks batch into the same waves as
    /// everything else.
    pub fn engine(&self) -> Arc<ConversationEngine> {
        Arc::clone(&self.engine)
    }

    /// The substrate, without taking the engine mutex.
    pub fn substrate(&self) -> Conversation {
        self.substrate.clone()
    }

    /// How many characters currently hold an open conversation.
    ///
    /// Not the same as the scheduler's population: a character wakes into the
    /// scheduler at startup and only opens a conversation on its first tick, so
    /// this trails it. The gap is exactly "how many have thought at least once",
    /// which is what the Pulse header reports.
    pub fn resident(&self) -> usize {
        self.live.len()
    }

    /// Make `live` the character's conversation, with the handle it is asked
    /// through, in one entry.
    fn adopt(&self, npc_id: u64, live: Live) -> Arc<Resident> {
        let opened = Arc::new(Resident {
            asker: live.sequence.unsealed_asker(),
            conversation: ConversationLock::new(live),
        });
        let slot = ArcSwapOption::from(Some(Arc::clone(&opened)));
        if let Some(replaced) = self.live.pin().insert(npc_id, slot) {
            replaced.store(None);
        }
        opened
    }
}

mod conversation;
mod journaling;
mod mind_schema;
mod perception;
mod reflecting;
#[cfg(test)]
mod tests_identity;
#[cfg(test)]
mod tests_perception;
mod think;
mod trial;

pub use conversation::*;
use journaling::*;
pub use mind_schema::*;
pub use perception::*;
pub use think::*;
pub use trial::TrialDecode;
