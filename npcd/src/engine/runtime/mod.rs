//! The daemon's runtime: the engine's loaded state, its worlds, its characters,
//! and the thread that keeps them all thinking.
//!
//! Loading is synchronous and slow, so it runs on its own thread while the HTTP
//! server already answers. A load that fails is fatal and loud, because a daemon
//! answering requests with a broken engine is harder to diagnose than one that is
//! absent. Each hosted world has its own clock and metronome, and the `Runtime`
//! holds them over one substrate. Each character runs as a supervised task that
//! takes its turns against the world it belongs to.
//!
//! - `acting`: an act's effect on the world and the two records it leaves.
//! - `composing`: sitting a character down to write its mission's piece.
//! - `device`: effector-device calls run in-process.
//! - `hosting`: loading, placing and unhosting worlds and their characters.
//! - `ingesting`: writing the mind's documents into the substrate a window at a time.
//! - `instructions`: the standing instructions a character is given.
//! - `journaling`: drafting journal entries as tasks of their own.
//! - `lives`: running a character's episodes and the beliefs they form.
//! - `load`: the settings a load reads and the phases it runs in.
//! - `loader`: the load itself, in dependency order.
//! - `messages`: messages from outside a world to its characters and channels.
//! - `mind_hold`: holding the mind, and reacting when it arrives or changes.
//! - `missions`: missions, the command table and reports.
//! - `moments`: one world moment and what it delivers.
//! - `perceiving`: what a body can see and do from where it stands.
//! - `persona`: resolving a character's authored state into its persona.
//! - `records`: the feed line and the verdict each act leaves.
//! - `reflecting`: reflections and the dreams they may grow into.
//! - `settings`: runtime-wide settings installed at startup.
//! - `supervise`: supervising each character's task.
//! - `turn`: one character's turn, from due to answered.
//! - Tests: `tests_acts`, `tests_arrival`, `tests_device`, `tests_hosting`,
//!   `tests_mission`, `tests_standing`, `tests_support`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use futures::executor::block_on;
use futures::future::BoxFuture;
#[cfg(feature = "memory-census")]
use heap_census::{set_thread, Scope};
use tokio::runtime::{Builder as RuntimeBuilder, Handle};
use tokio::sync::oneshot;

use candle_conversation::guest::resolve_seed;
use candle_conversation::models::ModelBuilder;
use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::projection::{
    Builder, DecodePriority, GroupId, LayerId, ProjectionEvent, SelectionState, TimelineId,
};
use candle_conversation::{
    ConversationEngine, LoadPhase, Sequence, SequenceConfig, TurnEvent, TurnHandle, TurnResponse,
};
use journal_core::folding::Folded;
use journal_core::journal::Holds;
use journal_core::record::DraftRecord;
use journal_core::section::JournalPrompt;
use journal_core::workflow::Outcome as Drafted;

use axum::body::{to_bytes, Body};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::Request as HttpRequest;
use axum::http::StatusCode;
use axum::Router;
use doc_units::Doc;
use serde::{Deserialize, Serialize};
use serde_json::{from_slice, json, to_string, to_vec, Value};
use tower::ServiceExt;
use zend_tools::state::Secrets;

use crate::authority::beyond::Tools;
use crate::authority::ops::{ActSource, Voiced, WorldOp};
use crate::authority::told::Told;
use crate::authority::{Role, Standing};
use crate::bridge::is_mine as is_of_the_bridge;
use crate::effector::auth::{AskedBy, PinnedStandpoint};
use crate::effector::router::address_of;
use crate::effector::station;
use crate::effector::token::Tokens;
use crate::engine::act::Act;
use crate::engine::agenda::{self, ordered::ORDERED_ACTS};
use crate::engine::authoring;
use crate::engine::authorize::{self, critical_compass};
use crate::engine::bearings;
use crate::engine::beyond::{self, Group};
use crate::engine::body::{self, Outcome};
use crate::engine::calibrate;
use crate::engine::channels::{self, ChannelSection};
use crate::engine::compose;
use crate::engine::driver::{self, Metronome};
use crate::engine::environment::{self, Reach};
use crate::engine::event::{Event, EventKind, Salience};
use crate::engine::experience;
use crate::engine::experience::concern::{concern, Change, Hears};
use crate::engine::experience::days::Days;
use crate::engine::experience::line::first_sentence;
use crate::engine::experience::store::Journals;
use crate::engine::guardian::GuardianLog;
use crate::engine::identity;
use crate::engine::ingest;
use crate::engine::invoke_body::Invokable;
use crate::engine::journal::style::NPC;
use crate::engine::journal::world::Snapshot;
use crate::engine::life;
use crate::engine::loading::{LoadProgress, LoadStep};
use crate::engine::loopguard::{salient, Took, NUDGE, STOP};
use crate::engine::mind::{frame_fingerprint, Answer, Minds, Projected};
use crate::engine::mind_record;
use crate::engine::mission::{
    compass_line, compass_report_line, doc_compass, given_compass, is_judgement, judgement_then,
    post_work_line, progress_line, publish_compass, refused_compass, time_compass, time_step, Aim,
    DeskVerbs, DocStep, Mission, MissionPrompt, Outcome as Verdict, JUDGEMENT_LINE,
};
use crate::engine::mission_acts::{self, COMMAND_DESK};
use crate::engine::mission_gen;
use crate::engine::mission_gen::config::Config;
use crate::engine::mission_gen::gates::{life_voice, Form};
use crate::engine::mission_gen::material::strip_calls;
use crate::engine::order_table::order_takers;
use crate::engine::people;
use crate::engine::prompt::{self, Persona};
use crate::engine::prose::{decode_call, CallAsk, WORK_BAND};
use crate::engine::reflect;
use crate::engine::schema;
use crate::engine::talk::authority::Changes;
use crate::engine::talk::key::VISITOR;
use crate::engine::talk::runner::Talks;
use crate::engine::talk::templates;
use crate::engine::talk::voice::Post;
use crate::engine::terminal::acts::answers as answers_through_a_terminal;
use crate::engine::terminal::acts::reads as terminal_reads;
use crate::engine::terminal::Machine;
use crate::engine::tick::{Due, Pace, Scheduler, Shared as SharedScheduler};
use crate::engine::tools::{self, Mode, Tool};
use crate::engine::watcher::{is_ingestible, Ledger, Reconcile};
use crate::engine::witnessed;
use crate::engine::work;
use crate::mind::files::MindFiles;
use crate::mind::{Mind, MindRepo};
use crate::mind_store::{self, MindSource};
use crate::model::{self, MODEL};
use crate::npcs::{Casting, Npcs, SUSPENDED};
use crate::opstore::MindStores;
use crate::projection;
use crate::sim::orders::{Notice, EVERY_POSTER};
use crate::sim::phone::{Kind as ThreadKind, Origin, Said, Who, CHANNEL};
use crate::sim::Sim;
use crate::world::binding::Bindings;
use crate::world::{Hosted, Worlds};
use crate::world_state::place_of;
use agent::client::Agents;
use agent::fetcher::AgentFetch;
use agent::jobs::Jobs;
use agent::pusher::AgentPush;
use candle_conversation::EngineSource;
use cluster::member::{ClusterView, Link};
use host::calibration;
use host::cluster::store::ConversationStore;
use host::cluster::view::{
    admitted, catchup_detail, catchup_progress, connecting_detail, model_id,
};
use host::cluster::Join;
use host::download::ensure_model_pack;
use host::process::fatal;
use host::stopping::{Ending, Stopping};
use journals::experience::ExperienceKind;
use npc_map::world::Where;
use npc_map::{describe, perceive, route};
use substrate_git::member::{Access, Repos};
use tokio::sync::watch;

/// Resolves a character's world-clock instant, in milliseconds.
///
/// A function rather than a stored clock, because **there is no single world
/// clock.** Each world runs at its own pace and can be paused or jumped
/// independently from the console, and characters belong to worlds — so a
/// character in a world running at 60× must see its own time, not the daemon's.
/// Reading it per tick is what makes a pause take effect on the next tick
/// instead of whenever something happened to re-read a cached value.
pub type WorldClock = Arc<dyn Fn(u64) -> u64 + Send + Sync>;

/// Resolves a character's authored state at tick time.
pub type PersonaSource = Arc<dyn Fn(u64) -> Option<OwnedPersona> + Send + Sync>;

/// Records the register a character has named for itself, so it survives a
/// restart.
///
/// A sink rather than a handle to the registry, for the same reason the clock
/// and the persona are functions: the registry lives in the authored state,
/// which needs this runtime to answer its own routes. One of the two has to
/// exist first, and it is this one.
///
/// Called from the tick thread, so it must never block on anything slow. The
/// registry writes only when the register has actually changed, which makes
/// almost every call a map lookup.
pub type MoodSink = Arc<dyn Fn(u64, &str) + Send + Sync>;

/// The live engine, once loaded.
pub struct Runtime {
    // No `engine` handle here. `Minds` holds the one `Arc<Mutex<…>>` there is,
    // and a second slot pointing at the same engine would be two places to ask
    // "is it up" that could answer differently mid-startup.
    pub scheduler: SharedScheduler,
    pub progress: Arc<LoadProgress>,
    /// Where everything this daemon writes lives — the redo log, the character
    /// store, the portraits, the ingest ledger.
    ///
    /// Held rather than passed through, because the engine needs it too: the
    /// conversation layer's `workspace_path` defaults to the *process working
    /// directory*, so a daemon that does not name this ends up with its redo log
    /// wherever it happened to be launched from.
    pub data: PathBuf,
    /// The same mind, as the handle the schema is read through.
    pub mind_handle: Mind,
    /// Every hosted world's bots' experience, written to npcd's checkout of
    /// the journals repository (`docs/npc_conversations.md` §5).
    pub journals: Arc<Journals>,
    /// The conversations of the characters held here (§9).
    pub talks: Arc<Talks>,
    /// When each character's mission last changed through a conversation.
    pub talk_changes: Changes,
    /// The day each world this member writes was last seen in, so its
    /// journal hears once of a new one.
    days: Days,
    /// The mind repository, loaded and followed, with `--mind-repo`.
    mind_repos: OnceLock<Repos>,
    /// How a device-bridge job reaches an agent (`docs/zen_moon_mind.md` §15):
    /// the same agent client the mind's git jobs go through, kept so an NPC's
    /// effector verb and the operator route can submit a `BridgeCall`. Set once
    /// the member's agents are known; absent on a daemon that reaches none.
    agents: OnceLock<Agents>,
    /// The machines the house's agents run on, as last heard of
    /// (`docs/zen_everywhere.md` §5): what a mind with no body lists and
    /// connects to as terminals. Refreshed by `terminal_list` and
    /// `terminal_connect`, read by [`Self::within`].
    machines: RwLock<Vec<Machine>>,
    /// Asked for a relaunch: the mind's schema moved.
    relaunch: OnceLock<Arc<Stopping>>,
    /// What reads the authored state again when a mind pending at start
    /// arrives. Absent until installed, and then the load goes on with what
    /// it was handed.
    mind_arrived: RwLock<Option<MindArrived>>,
    /// Content hashes for every ingested layer file.
    ///
    /// Shared between the startup ingest and the watcher, deliberately: they are
    /// the same reconciliation, and two ledgers would make the first edit after a
    /// boot look like a whole-tree rewrite.
    pub ledger: Arc<Ledger>,
    /// How to ask what time it is in a character's own world.
    ///
    /// Installed after construction, like the persona source, because it reads
    /// the authored state — which needs this runtime to answer its own routes.
    /// One of the two has to exist first, and it is this one.
    clock: RwLock<Option<WorldClock>>,
    /// What the guardian has done, for the operator route to read. `None` when
    /// no guardian is configured.
    guardian_log: RwLock<Option<Arc<GuardianLog>>>,
    /// **The daemon's one handle to `data/substrate/`**, opened by the
    /// character registry and adopted here rather than opened a second time.
    ///
    /// Installed after construction for the same reason the clock is: the
    /// registry that opens it lives in the authored state, which needs this
    /// runtime to answer its routes. See [`crate::npcs::Npcs::substrate`] for
    /// what the second handle cost.
    substrate: RwLock<Option<SharedSubstrate>>,
    /// The shared cast — the very same `Arc<Npcs>` [`crate::api::Authored`]
    /// holds, adopted rather than reconstructed.
    ///
    /// Installed after construction for the same reason the substrate is: the
    /// registry that owns the cast lives in the authored state, which needs this
    /// runtime to answer its routes. It is the effector device's bridge to a
    /// character's projected state — the `/self` reads render a character's own
    /// agency/belief layers ([`crate::effector::selfsurface`]) and the
    /// `plan_*`/`orders_*` writes land in the agency plane
    /// ([`crate::effector::plan`], effector design §9.2) through
    /// [`Npcs::put_strategy_self`], which then projects on the next turn via the
    /// `agency` collection and [`crate::engine::persona`].
    ///
    /// **The same handle, never a second.** Sharing the one `Arc` preserves the
    /// one-writable-substrate-handle invariant ([`crate::npcs::Npcs::substrate`],
    /// `npcs.rs` §"One handle to the substrate"): the effector writes through the
    /// identical `Npcs`/`SharedSubstrate` the console API and the load walk use.
    npcs: OnceLock<Arc<Npcs>>,
    /// Every character's live conversation. `None` until the model is loaded —
    /// there is nothing to hold a conversation on before then.
    pub minds: RwLock<Option<Arc<Minds>>>,
    /// Who each character is, resolved from its authored layers at tick time.
    ///
    /// A function, because the layers are editable while the daemon runs: a
    /// belief changed through the authoring API has to reach the next tick, and
    /// a persona captured at startup would keep the character as it was when the
    /// process began.
    persona: RwLock<Option<PersonaSource>>,
    /// Where a named register goes to be remembered. Absent until installed,
    /// and a character that names one before then simply is not recorded.
    mood_sink: RwLock<Option<MoodSink>>,
    /// The effector device's token store — one opaque secret per character,
    /// resolving a bearer token to a body (see [`crate::effector::token`]).
    ///
    /// Installed after construction like the clock, because it reads the
    /// daemon's `tokens/` directory which is the daemon's to open once. Held as
    /// an `Arc` so the `local` router's auth layer and the in-process fast path
    /// resolve and mint through one handle rather than two.
    tokens: RwLock<Option<Arc<Tokens>>>,
    /// The `local` effector router, retained so a character's `invoke`
    /// takes the in-process fast path (effector design §8.1) — the same service
    /// the external mount answers on, driven through `oneshot` with no socket.
    ///
    /// Installed after construction because the router's own state holds a
    /// [`Weak`] handle back to this runtime, and one of the two has to exist
    /// first. Absent until then, and a call before then simply is not answered.
    effector: RwLock<Option<Router>>,
    /// The worlds this daemon is running.
    ///
    /// Distinct from `Authored::worlds`, which is the registry of world
    /// *documents* — the lore and the settings an author writes. These are the
    /// simulations: who is standing where, what is claimed, what just happened.
    /// A world can be authored without being hosted, and is, until something
    /// asks for it.
    pub hosted: Worlds,
    /// Who is present to which character, and in what mode — see
    /// [`crate::engine::interaction`]. Not behind an `RwLock` with the rest:
    /// it owns its own map and nothing else reads it.
    pub interactions: crate::engine::interaction::Interactions,
    /// Which character is which body.
    pub bodies: Bindings,
    /// One metronome per hosted world, so a world can be held still without
    /// stopping the daemon — every question an operator asks of a live world is
    /// asked of one that is moving underneath the answer.
    metronomes: Mutex<BTreeMap<String, Metronome>>,
    /// Each world's buildings, rendered once, keyed by world and then in
    /// [`npc_map::describe::places`] order. The same text for every character
    /// standing in one, and it never changes.
    places: Mutex<BTreeMap<String, Vec<(String, String)>>>,
    /// What this daemon is to each world it hosts — its authority, a replica
    /// following it, or cut off from it — and which characters it runs
    /// (`docs/replicated_substrate.md` §9). See [`crate::authority`].
    pub standing: Standing,
    /// Every store over the mind, each operation's and each author's (§10.3).
    pub ops: Arc<MindStores>,
    /// Each character's mission compass as last told to it — see
    /// [`Runtime::mission_compass`] — so it is told again only when it changes.
    compass_told: Mutex<HashMap<u64, String>>,
    /// The chapter — missions closed — each character's last turn was taken
    /// in, so the turn after a hand-in closes the chapter in its journal before
    /// its new conversation opens. See [`Runtime::turn_chapter`].
    chapters: Mutex<HashMap<u64, u64>>,
    /// The characters with a dream being written — from the moment their
    /// reflection's first question is answered to the moment the dream is kept.
    /// §7: *"At most one dream in flight per character."*
    ///
    /// **The dream, not the reflection.** A reflect while this is taken still
    /// gets its own reflection and its own answer; only the brief and the dream
    /// after it are skipped. This once guarded the whole reflection and refused
    /// every reflect behind it — a slot held for three to seven minutes against
    /// a thirty-second cooldown answered six reflects in ten with nothing.
    dreaming: Mutex<HashSet<u64>>,
    /// Which behaviour-space cell the next unattributed reflection draws from.
    ///
    /// A plain rotating counter, shared across the cast rather than kept per
    /// character. Coverage is a property of the corpus, and a counter that
    /// advanced per character would let a talkative one sit in one domain while
    /// a quiet one never left the first. See [`crate::engine::reflect::DOMAINS`].
    reflect_domain: AtomicUsize,
    /// The registers `<mind>/moods/` has a mood written for.
    ///
    /// Injected at startup rather than read here, the same as the persona
    /// source: the libraries are the daemon's to load, and the engine's job is
    /// to offer whatever it is given. Empty until then, and empty for a daemon
    /// with no mind — which drops the parameter rather than the act.
    feelings: RwLock<Vec<String>>,
    /// Each character's own idle thought, from its personality's `idle:` —
    /// what its idle wake says in place of [`NO_MISSION`] or [`YOUR_OWN_TIME`].
    /// Set as the character is embodied, on every member, since the idle wake
    /// is the running member's to deliver.
    idles: RwLock<HashMap<u64, String>>,
    /// What each character reaches beyond its world, from its personality's
    /// `reaches:` (`docs/zen_moon_mind.md` §16). Set as it is embodied and as
    /// its personality is read again, on every member: the acts are offered
    /// and performed by the member running it.
    reaches: RwLock<HashMap<u64, Vec<Group>>>,
    /// What zend's tools run with here: the network alone, the daemon's
    /// secrets, and each character's own sessions.
    pub(crate) tools: Tools,
    /// The daemon's secrets, for [`Self::tools`]: empty until the tenant reads
    /// them ([`Runtime::set_tool_secrets`]), which a search then says.
    pub(crate) tool_secrets: RwLock<Arc<Secrets>>,
    /// How soon each character may take each act again.
    ///
    /// Here rather than on a world, because the limit belongs to the person
    /// acting: a character that moved between bodies mid-fight would otherwise
    /// get a free swing.
    pub cooldowns: crate::engine::cooldown::Cooldowns,
    /// How each character is kept out of a behavioural loop, in turns rather than
    /// in real time — the counterpart to [`Self::cooldowns`], which paces the
    /// body. An act taken again meaning the same as one of its last takings is
    /// struck, for longer while the loop goes on. See [`crate::engine::loopguard`].
    pub loop_guards: crate::engine::loopguard::LoopGuards,
    /// Set on shutdown so every character task stops at its next wait rather
    /// than being killed mid-tick with a half-written turn.
    stopping: AtomicBool,
    /// The runtime character tasks, metronomes and reflections run on.
    ///
    /// Captured at construction when one is ambient — the daemon constructs
    /// its `Runtime` inside `#[tokio::main]` — and genuinely absent in the
    /// unit tests that exercise scheduling data without spawning anything.
    spawner: Option<Handle>,
    /// One running task per character, keyed by id — the supervisor's roster.
    ///
    /// A task is spawned when the character wakes into the scheduler and
    /// cancelled when it is deleted; a task that panics is restarted by its
    /// own supervisor wrapper with exponential backoff. See
    /// [`Runtime::spawn_character`].
    tasks: Mutex<BTreeMap<u64, CharacterTask>>,
    started: Instant,
}

mod acting;
mod composing;
mod device;
mod hosting;
mod ingesting;
mod instructions;
mod journaling;
mod lives;
mod load;
mod loader;
mod messages;
mod mind_hold;
mod missions;
mod moments;
#[cfg(test)]
use moments::Rest;
mod perceiving;
mod persona;
mod records;
mod reflecting;
mod settings;
mod supervise;
#[cfg(test)]
mod tests_acts;
#[cfg(test)]
mod tests_arrival;
#[cfg(test)]
mod tests_device;
#[cfg(test)]
mod tests_hosting;
#[cfg(test)]
mod tests_mission;
#[cfg(test)]
mod tests_standing;
#[cfg(test)]
mod tests_support;
mod trial;
mod turn;

pub(crate) use ingesting::*;
pub use instructions::*;
pub use lives::*;
pub use load::*;
use loader::*;
pub use persona::*;
pub use records::*;
pub use supervise::*;
use turn::*;
