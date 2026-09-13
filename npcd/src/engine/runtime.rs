//! Standing the engine up: the model, the substrate, the cast, and the thread
//! that keeps them all thinking.
//!
//! # The startup shape, and why it is an OS thread
//!
//! Loading is synchronous and slow — a GGUF onto the card, then a redo-log
//! replay, then a diff of the mind directory against what the substrate already
//! holds. None of it is async work, and all of it must happen while the HTTP
//! server is already answering, because the console's loading screen is served
//! by that server and a daemon that binds its port after loading has nothing to
//! show a person waiting.
//!
//! So: bind first, load on a plain `std::thread`, and let
//! [`crate::engine::loading::LoadProgress`] be what the two share. This is
//! zend's arrangement and it is arranged that way for the same reason.
//!
//! # Failure is fatal, and loudly
//!
//! A daemon that fails to load its model and then answers requests anyway is
//! worse than one that exits: every route degrades to a confusing error, and the
//! console shows an engine that is present and broken rather than absent. The
//! loader logs what failed and exits the process.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::projection::{Builder, ProjectionEvent, SelectionState};
use candle_conversation::{
    ConversationEngine, Sequence, SequenceConfig, TurnEvent, TurnHandle, TurnResponse,
};

/// The marker a prefilled turn's projection is recorded against.
///
/// The same string zend uses (`zend/src/tool_def.rs`), duplicated rather than
/// shared because it is a wire constant of the conversation layer that neither
/// daemon owns — and taking a dependency on the other daemon to reach it would
/// be far stranger than the fourteen bytes.
pub(crate) const PROJECTION_MARKER: &str = "<|projection|>";

/// How many of a character's existing dream axes a reflection is steered away
/// from — `docs/reflection_and_dreams.md` §5: shown every axis it had used the
/// generator recombined them; shown a random eight it found a new one.
const SAMPLED_AXES: usize = 8;

use crate::engine::act::Act;
use crate::engine::authoring;
use crate::engine::body::{self, Outcome};
use crate::engine::driver::{self, Metronome};
use crate::engine::environment;
use crate::engine::identity;
use crate::engine::ingest;
use crate::engine::life;
use crate::engine::loading::{LoadProgress, LoadStep};
use crate::engine::mind::{frame_fingerprint, Minds, Projected};
use crate::engine::prompt::{self, Persona};
use crate::engine::reflect;
use crate::engine::schema;
use crate::engine::tick::{Pace, Scheduler, Shared as SharedScheduler};
use crate::engine::tools::{self, Mode, Tool};
use crate::engine::watcher::Ledger;
use crate::mind::Mind;
use crate::model;
use crate::npcs::Casting;
use crate::world::binding::Bindings;
use crate::world::{Hosted, Worlds};
use npc_map::world::Where;
use npc_map::{describe, perceive};

/// How often the driver thread looks for characters that are due.
///
/// Not the tick rate — that is per character and salience-driven. This is only
/// how finely the scheduler's clock is quantised, so a preempt is acted on
/// within this long at worst.
const DRIVE_INTERVAL: Duration = Duration::from_millis(100);

/// Tokens one prefill forward carries when the model can take them — the
/// scheduler's per-forward *target*.
///
/// See the note at the load site. In short: the wave's fixed cost is per slab,
/// not per token, so a budget close to one document's size makes every document
/// pay a whole sweep. At the ceiling a slab carries four or five documents and
/// pays it once.
///
/// **A target, not a guarantee.** The scheduler bounds every forward by the
/// model's own width cap as well (`prefill_pass_budget`), and on a routed
/// checkpoint that cap is the narrower of the two: the expert chain carries eight
/// rows per token. Read alone, this number sized the world ingest on the routed
/// Qwen3.6-35B-A3B for a 3.3 GB transient tier a 24 GB card's partition did not
/// have, and every document in the wave failed.
const PREFILL_PASS_TOKENS: usize = 8192;

/// Resolves a character's world-clock instant, in milliseconds.
///
/// A function rather than a stored clock, because **there is no single world
/// clock.** Each world runs at its own pace and can be paused or jumped
/// independently from the console, and characters belong to worlds — so a
/// character in a world running at 60× must see its own time, not the daemon's.
/// Reading it per tick is what makes a pause take effect on the next tick
/// instead of whenever something happened to re-read a cached value.
pub type WorldClock = Arc<dyn Fn(u64) -> u64 + Send + Sync>;

/// What separates an act from what came of it in a feed line, when it landed.
///
/// A single character with spaces around it, because the console splits on it
/// to set the two halves in different faces — see `pulse.js::splitAct`. A word
/// would appear inside the prose on either side of it and the split would land
/// in the middle of a sentence.
pub const LANDED: &str = "→";

/// The same, when the world refused. Distinct from [`LANDED`] so a refusal is
/// legible as one at a glance rather than by reading the sentence.
pub const REFUSED: &str = "✗";

/// What one act left behind, for the two readers that want different things.
///
/// **A person watching and the character that acted need different sentences.**
/// The feed wants the act — `move_to — the command room` — in a scannable
/// column beside every other act. The character wants the world's verdict —
/// "You got to the command room." — because that is the only thing that says
/// whether what it tried happened.
///
/// These were one string while the feed was the only channel. Handing that
/// string to the model as a `<tool_response>` sent it its own arguments back
/// as though they were an outcome. See [`Runtime::record_act`].
#[derive(Debug, Clone)]
pub struct Recorded {
    /// The Pulse line, and the character's window.
    pub feed: String,
    /// What the character is told came of the act.
    pub answer: String,
    /// Whether the world took the act — the verdict `answer` is written from.
    /// What decides whether a `reflect` goes on to a reflection: one the world
    /// refused was not a character stopping to think.
    pub landed: bool,
}

/// What a turn with a reflection in it still owes, held until the reflection's
/// first question is answered — see [`Runtime::begin_reflection`].
#[derive(Clone, Debug)]
pub struct Owed {
    /// Every answer the turn owes, one per call, in call order. The reflect's
    /// holds the world's own line until the reflection replaces it.
    pub answers: Vec<String>,
    /// Which of `answers` is the reflect's.
    pub slot: usize,
    /// The reflect's row as recorded: the act without its result. Completed
    /// with the result once there is one — see [`Scheduler::amend_act`].
    pub row: String,
}

/// One character's dream slot, taken — see [`Runtime::claim_dream`]. Handed
/// back when dropped, so a dream that errors or panics part-way cannot leave
/// the character unable to dream again.
struct DreamSlot<'a> {
    rt: &'a Runtime,
    npc_id: u64,
}

impl Drop for DreamSlot<'_> {
    fn drop(&mut self) {
        self.rt.dreaming.lock().unwrap().remove(&self.npc_id);
    }
}

/// Everything the prompt needs about one character, owned.
///
/// Owned rather than borrowed because it crosses from the async world (where the
/// authored state lives behind an `RwLock`) into the tick thread. The alternative
/// is holding that lock across a decode, which would block every authoring write
/// for the length of a generation.
#[derive(Debug, Default, Clone)]
pub struct OwnedPersona {
    pub name: String,
    /// The personality and world this character belongs to, by slug.
    ///
    /// Never rendered — they are what a turn pins its personality's anchor, its
    /// world's setting and its building to in the projection. See
    /// [`crate::engine::identity`].
    pub personality: String,
    pub world_id: String,
    pub identity: String,
    /// The personality's anchor — the floor every character of it reads.
    ///
    /// **Carried as prose as well as pinned as a slug**, because the two
    /// conversation kinds receive it by different routes and only one of them
    /// gathers. An acting turn opens against the projection, which selects the
    /// `ANCHOR` collection member named by `personality`; a reflection builds its
    /// whole prompt from [`crate::engine::prompt::build_for`] and selects
    /// nothing, so for it the anchor has to be in the frame or it is nowhere.
    ///
    /// It was nowhere. A Maker's anchor opens *"You write a world, and you are
    /// not the only one… you have no world of your own"*, and no reflection had
    /// ever read a word of it — so a character told only `You are Tace.` and
    /// `The world you live in:` placed itself inside the story it writes, gave
    /// itself tools no Maker carries, and invented a colleague.
    pub anchor: String,
    pub manner: String,
    pub beliefs: Vec<String>,
    pub relationships: Vec<String>,
    pub intent: Option<String>,
    pub situation: String,
    pub world: String,
    /// The building this character lives in, as it remembers it. Filled in by
    /// the runtime rather than by the persona source: it comes from the map,
    /// which the authored record knows nothing about.
    pub place: String,
    /// Which part of the world `place` describes — see [`Persona::building`].
    pub building: String,
    pub mode: Mode,
}

impl OwnedPersona {
    pub fn as_persona(&self) -> Persona<'_> {
        Persona {
            name: &self.name,
            personality: &self.personality,
            world_id: &self.world_id,
            identity: &self.identity,
            anchor: &self.anchor,
            manner: &self.manner,
            beliefs: &self.beliefs,
            relationships: &self.relationships,
            intent: self.intent.as_deref(),
            situation: &self.situation,
            world: &self.world,
            place: &self.place,
            building: &self.building,
        }
    }
}

/// Resolves a character's authored state at tick time.
pub type PersonaSource = Arc<dyn Fn(u64) -> Option<OwnedPersona> + Send + Sync>;

/// Records where a character is, so a restart can put it back there.
///
/// A sink rather than a handle to the registry, for the same reason the clock
/// and the persona are functions: the registry lives in the authored state,
/// which needs this runtime to answer its own routes. One of the two has to
/// exist first, and it is this one.
///
/// Called from a world's metronome thread, so it must never block on anything
/// slow — the registry's own checkpoint gates make it a map lookup on almost
/// every call.
pub type PlaceSink = Arc<dyn Fn(u64, &str) + Send + Sync>;

/// Records the register a character has named for itself, so it survives a
/// restart the way its position does.
///
/// A sink for the same reason [`PlaceSink`] is one, and called from the tick
/// thread — so it must never block on anything slow. The registry writes only
/// when the register has actually changed, which makes almost every call a map
/// lookup.
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
    /// The mind directory, when there is one — what the file watcher watches.
    pub mind: Option<PathBuf>,
    /// The same mind, as the handle the schema is read through.
    pub mind_handle: Mind,
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
    /// **The daemon's one handle to `data/.substrate/`**, opened by the
    /// character registry and adopted here rather than opened a second time.
    ///
    /// Installed after construction for the same reason the clock is: the
    /// registry that opens it lives in the authored state, which needs this
    /// runtime to answer its routes. See [`crate::npcs::Npcs::substrate`] for
    /// what the second handle cost.
    substrate: RwLock<Option<SharedSubstrate>>,
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
    /// Where to record a body's room, so a restart reconstructs the world.
    ///
    /// Installed after construction, like the clock and the persona, and for
    /// the same reason. Absent until then, and a world that moves before it is
    /// installed simply is not checkpointed — nothing is lost that was not
    /// already going to be.
    place_sink: RwLock<Option<PlaceSink>>,
    /// Where a named register goes to be remembered. Absent until installed,
    /// and a character that names one before then simply is not recorded.
    mood_sink: RwLock<Option<MoodSink>>,
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
    /// How soon each character may take each act again.
    ///
    /// Here rather than on a world, because the limit belongs to the person
    /// acting: a character that moved between bodies mid-fight would otherwise
    /// get a free swing.
    pub cooldowns: crate::engine::cooldown::Cooldowns,
    /// Set on shutdown so the driver thread stops rather than being killed
    /// mid-tick with a half-written turn.
    stopping: AtomicBool,
    started: Instant,
}

/// Where a world's rooms live, under the authored corpus.
///
/// `map/<world_id>/`, a sibling of `worlds/<world_id>.yaml` and keyed the same
/// way. **That key is the whole reason a hosted world has an id**: every
/// character is created with a `world_id` naming the world document it belongs
/// to, so the places it can stand in have to be findable by that same name. A
/// hosted world under any other id would be a world no character could be put
/// into without a second table to reconcile the two.
///
/// A world with no map directory is authored but has nowhere to stand — which
/// is the common case, and not an error. Its characters have lore and no bodies.
pub const MAPS: &str = "map";

// A bench works on the mind itself — see `Runtime::new`, which points the
// worlds at it. **The root is wide and the reach is narrow**, rather than the
// other way round: `sim::bench::EDITABLE_AREAS` decides which parts of the mind
// answer, which keeps the dangerous set (`projection.yaml`, `mind.yaml`,
// `schema/`) named in one place beside the guard that enforces it instead of
// implied by a directory nobody would think to check.

/// What a character with nothing assigned is set on.
///
/// **Not a placeholder.** A person with no task does not stand still, and an
/// NPC that does reads as scenery — so having nothing to do is itself a
/// standing instruction, and this is it. When missions exist they replace this
/// one; they do not fill a hole it was leaving.
///
/// Worded as a disposition rather than a script. *Go to the green room* would
/// be a standing order every character in the world followed identically, which
/// is the recommendation-not-affordance failure at its worst: it fires on every
/// idle turn, for every character, and never appears in a log as anything but a
/// heartbeat.
/// **Points at the work, not at the map.** The first version sent a character
/// off to see a room it had not been in, and it did exactly that — three of
/// them toured a seventy-eight room building for hours and came back with
/// nothing to say, because a room is not a subject. What a Maker is for is in
/// front of it: the ledgers, the filed stories, the wall of dates. Naming the
/// thing within reach gives the next conversation something to be *about*,
/// which is the difference between two characters talking and two characters
/// exchanging weather.
/// **One instruction, and no verb of motion in it.**
///
/// This has now been wrong in three ways. It began as *go and see a room you
/// have not been in*, and characters toured a seventy-eight room building for
/// hours and arrived with nothing to say, because a room is not a subject. It
/// was then rewritten to point at the work — and still ended with *then go
/// where people are*, which is a movement instruction sitting in the most
/// recent position in the window, where attention weights hardest. Every act in
/// the harness became `move_to`, nineteen for nineteen.
///
/// A standing task restated every ninety seconds does not need to describe a
/// plan. It needs to name the **next** thing, once. The going will happen on
/// its own when the character has something worth carrying.
pub const NO_MISSION: &str = "Nothing has been asked of you, and you are on your own. That is \
                              not nothing to do: the work is in front of you. Take up one thing \
                              within reach — a ledger, a filed story, a date on the wall — and \
                              carry it far enough to have a view about it you could defend.";

/// How long a character must go without news before the standing task is
/// restated to it.
///
/// **A stretch of quiet, not an empty instant.** Long enough that it cannot land
/// between two turns of a conversation — an exchange runs at roughly the world's
/// moment, and a companion that is thinking, walking a stop, or simply slower
/// than the heartbeat can leave a gap of tens of seconds without the
/// conversation being over. Short enough that a character genuinely left alone
/// does not stand in a room for minutes with nothing to go on.
const IDLE_AFTER_MS: u64 = 90_000;

/// The same instruction, for a character that is not alone.
///
/// **Company changes what there is to do, so it changes the instruction.** This
/// is not advice smuggled into a percept — it is the standing task, and having
/// somebody in front of you is a different standing task from having nobody.
///
/// It exists because of what the first one did on its own. Told to go somewhere
/// new every quiet turn, two characters explored a seventy-eight room building
/// beautifully and never once held a conversation: each moved every four
/// seconds, so being in a room together lasted exactly one tick and neither had
/// a reason to stay for the second.
/// **It names them.** `{who}` is filled with who is actually standing there.
///
/// The unnamed version — "Somebody else is here" — was the thing that made a
/// room full of people unusable. `tell` and `ask` both take a name and refuse
/// one they cannot find, and the *only* place a character is told a name is the
/// situation percept, which is deliberately suppressed while nothing moves
/// (see `npc_map::delta`). So a character standing still learns who its company
/// is exactly once, on arrival, and that line then ages out of the verbatim
/// window while the person is still in front of it.
///
/// What it has after that is this instruction, restated every quiet tick in the
/// most recent position in the window — telling it to talk to somebody it can
/// no longer name. It addressed `you` and was refused; it turned to face "the
/// person standing in the anteroom"; it spoke to the room and nobody was
/// obliged to answer. Every one of those is a character reaching for an
/// addressee it has been told it has and not been told the name of.
/// **Asks for something specific, because the generic version got generic
/// answers.** "Say what you have been looking at" produced characters agreeing
/// with each other about silence and gaps for a hundred turns — three of them
/// converging on one abstraction because none had named a thing. A conversation
/// about work needs a *piece* of work in it: a page, a date, a disagreement
/// with a colleague's entry.
pub const IN_COMPANY: &str = "Nothing has been asked of you, and you are not alone. {who} \
                              here with you. `tell` or `ask` something about the work, now, and \
                              name a particular thing — a page you read, a date that will not \
                              reconcile, an entry of theirs you doubt. Ask them something they \
                              have to answer, or answer what they asked you. Address them \
                              exactly as written.";

/// What came of saying something to a character on its handset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageSent {
    /// What the character is called, by the world.
    pub with: String,
    /// How much is now waiting for it, unread.
    pub waiting_for_them: usize,
    /// Whether it carries a handset, and so can answer at all.
    pub can_reply: bool,
}

impl Runtime {
    pub fn new(mind_handle: Mind, data: &Path) -> Arc<Self> {
        // **The benches are pointed at the mind this daemon was given**, before
        // any world is hosted. Derived from the handle rather than set on a
        // hosting call: a world reached through `host` and one reached through
        // `host_authored` are the same world, and making only the second of
        // them able to edit documents was a difference nothing in the fiction
        // justifies and nothing in the signature announces.
        let hosted = Worlds::new();
        if let Some(root) = mind_handle.root() {
            hosted.set_bench_root(root);
        }
        Arc::new(Self {
            scheduler: Arc::new(Scheduler::default()),
            progress: Arc::new(LoadProgress::new()),
            data: data.to_path_buf(),
            mind: mind_handle.root().map(Path::to_path_buf),
            mind_handle,
            // Read from disk: a document already in the substrate must not be
            // prefilled again, and at four seconds each that is the difference
            // between a twelve-second boot and half an hour of one.
            ledger: Arc::new(Ledger::open(data)),
            clock: RwLock::new(None),
            substrate: RwLock::new(None),
            minds: RwLock::new(None),
            persona: RwLock::new(None),
            place_sink: RwLock::new(None),
            mood_sink: RwLock::new(None),
            hosted,
            interactions: crate::engine::interaction::Interactions::new(),
            bodies: Bindings::new(),
            metronomes: Mutex::new(BTreeMap::new()),
            places: Mutex::new(BTreeMap::new()),
            dreaming: Mutex::new(HashSet::new()),
            reflect_domain: AtomicUsize::new(0),
            feelings: RwLock::new(Vec::new()),
            cooldowns: crate::engine::cooldown::Cooldowns::new(),
            stopping: AtomicBool::new(false),
            started: Instant::now(),
        })
    }

    /// Host every authored world that has rooms to stand in.
    ///
    /// Driven by the world registry rather than by what is on disk, so a map
    /// directory nothing authored is *not* silently hosted under an id no
    /// character can name. Each world that does have one is hosted under its own
    /// id, which is what lets a character's `world_id` find the places it can be.
    ///
    /// Reports every world it tried, and what came of it. A map that does not
    /// hold together is an authoring mistake to fix, not a reason for the rest
    /// of the cast to have nowhere to stand.
    pub fn host_authored<'a>(
        self: &Arc<Self>,
        authored: &Path,
        worlds: impl IntoIterator<Item = &'a str>,
    ) -> Vec<(String, anyhow::Result<Arc<Hosted>>)> {
        let maps = authored.join(MAPS);
        worlds
            .into_iter()
            .filter(|id| maps.join(id).is_dir())
            .map(|id| (id.to_string(), self.host(id, &maps.join(id))))
            .collect()
    }

    /// Load one world and start it moving.
    ///
    /// The metronome holds only a [`Weak`] handle back here. A strong one would
    /// be a cycle — the runtime owns the metronome and the metronome would own
    /// the runtime — and the daemon would never drop, which reads as a clean
    /// shutdown that never finishes.
    pub fn host(self: &Arc<Self>, id: &str, dir: &Path) -> anyhow::Result<Arc<Hosted>> {
        let world = self.hosted.load(id, dir)?;
        let back = Arc::downgrade(self);
        let name = id.to_string();
        let moving = world.clone();
        let beat = Metronome::start(driver::EVERY, move || {
            let Some(rt) = back.upgrade() else {
                return;
            };
            if rt.stopping.load(Ordering::Relaxed) {
                return;
            }
            // Only when somebody actually covered a leg. Most moments move
            // nobody — that is what makes a large cast affordable — and a
            // checkpoint on a still world would be a world read per 500 ms per
            // world for an answer that cannot have changed.
            if rt.moment(&moving).moved > 0 {
                rt.checkpoint_places(&moving);
            }
        });
        self.metronomes.lock().unwrap().insert(name, beat);
        Ok(world)
    }

    /// Stop a world: its metronome, its minds' bodies, and the world itself.
    ///
    /// All three, in that order. Stopping the metronome first means nothing is
    /// mid-moment while the bindings go, and unbinding before releasing means
    /// no character is left pointing at a world that is no longer there.
    pub fn unhost(&self, id: &str) -> bool {
        if let Some(beat) = self.metronomes.lock().unwrap().remove(id) {
            beat.stop();
        }
        self.bodies.release_world(id);
        self.hosted.release(id)
    }

    /// Give a character a body in a hosted world, and set it to a working pace.
    ///
    /// A character with a body has somewhere to be and something in front of
    /// it, so it never goes as quiet as one that is only reacting — see
    /// [`Pace`]. It is also grounded at once rather than at the world's next
    /// moment: a character that can act before it has been told where it is
    /// would act blind.
    pub fn embody(&self, npc_id: u64, world: &str, body: &str, now_ms: u64) -> anyhow::Result<()> {
        let Some(hosted) = self.hosted.get(world) else {
            anyhow::bail!("no world `{world}` is running");
        };
        if hosted.read(|w| w.actor(body).is_none()) {
            anyhow::bail!("`{world}` has no body `{body}`");
        }
        self.bodies.bind(npc_id, world, body)?;
        self.scheduler.set_pace(npc_id, Pace::WORKING, now_ms);
        environment::push_one(&hosted, &self.scheduler, npc_id, body);
        Ok(())
    }

    /// The body id a character acts through.
    ///
    /// Derived rather than stored, so a restart reconstructs the same
    /// correspondence without a table — the same reason a timeline id is
    /// derived. A body a character is not in is a body nothing can address.
    pub fn body_id(npc_id: u64) -> String {
        format!("npc-{npc_id}")
    }

    /// Put a new character into its world, at the way in.
    ///
    /// Called when a character is created, and again on a restart for every
    /// character whose world is hosted — both are the same operation, because
    /// entering a body that is already there is not an error and binding one
    /// that is already bound changes nothing.
    ///
    /// Silent when the character's world has no map: most worlds have none, and
    /// a character with lore and no body is a character, not a failure.
    pub fn embody_in_world(
        &self,
        npc_id: u64,
        world_id: &str,
        home: Option<&str>,
        name: &str,
        remembered: Option<&str>,
        now_ms: u64,
    ) -> anyhow::Result<bool> {
        let Some(hosted) = self.hosted.get(world_id) else {
            return Ok(false);
        };
        let body = Self::body_id(npc_id);
        let placed = hosted.with(|w| {
            if w.actor(&body).is_some() {
                return Ok(());
            }
            // **Back where it was, if the world still has that room.**
            //
            // The world itself is not persisted — who is standing where lives in
            // RAM and goes with the process — so without this every character
            // re-entered at the arrival door on every restart, however far it
            // had walked. Two Makers who had spent an hour finding each other
            // were returned to the front room as strangers while their own
            // transcripts said otherwise.
            //
            // A remembered room the map no longer has is not an error: maps are
            // authored and rooms get renamed. Fall through to the door, which is
            // exactly what a character whose room was demolished should do.
            let recalled = remembered
                .and_then(Where::parse)
                .filter(|at| w.map().node_at(at).is_some());
            let at = match recalled {
                Some(at) => at,
                // Where in the world this kind of character belongs. A world is
                // bigger than any one character's part of it — a Maker starts in
                // the vault and a soldier starts in a city, and they are the same
                // world — so the door is the home area's, not the world's.
                None => match home {
                    Some(home) => w.map().arrival_in(home).ok_or_else(|| {
                        anyhow::anyhow!("`{world_id}` has no `{home}` to arrive in")
                    })?,
                    None => w.map().arrival().ok_or_else(|| {
                        anyhow::anyhow!("world `{world_id}` has nowhere to arrive")
                    })?,
                },
            };
            w.enter(&body, name, at).map_err(anyhow::Error::from)
        });
        placed?;
        // **A handset on arrival, and a line on the roster.** Two halves of one
        // thing: carrying the phone is what makes the messaging acts reachable,
        // and being on the roster is what makes this character reachable *by*
        // them. Issued here rather than in the seed because the cast is not
        // known when a world is built — a character that arrives an hour later
        // has to be callable too.
        let display = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()));
        if let Some(display) = display {
            hosted.with_sim(|s| crate::sim::seed::issue_handset(s, &body, &display));
        }
        self.embody(npc_id, world_id, &body, now_ms)?;
        Ok(true)
    }

    /// What a character is set on, as the turn that keeps it on course.
    ///
    /// One standing instruction, restated whenever nothing else is happening.
    /// A character with a mission gets that mission; a character with none gets
    /// [`NO_MISSION`] — which is not a placeholder for one. A person with
    /// nothing assigned does not stand still, and neither should a character:
    /// looking around and talking to people is what there is to do.
    pub fn nudge_for(&self, npc_id: u64) -> Option<String> {
        // Only for a character with a body. One with no world has nowhere to
        // explore and nobody to talk to, and telling it otherwise would be
        // instructing it to do something it cannot.
        let (hosted, body) = self.body_of(npc_id)?;
        // Who is here — which decides *which* standing task this is, and, when
        // there is company, is itself the most useful thing in it. Reading the
        // room already told us the names; the version of this that answered
        // only `alone: bool` threw them away and left the character to guess at
        // an addressee. See [`IN_COMPANY`].
        let company: Vec<String> = hosted.read(|w| {
            let Some(here) = w.actor(&body).map(|a| a.at.clone()) else {
                return Vec::new();
            };
            w.actors_at(&here)
                .into_iter()
                .filter(|other| other.id != body)
                .map(|other| other.name.clone())
                .collect()
        });
        Some(match company.is_empty() {
            true => NO_MISSION.to_string(),
            false => IN_COMPANY.replace(
                "{who}",
                &format!(
                    "{} {}",
                    npc_map::text::list(&company),
                    if company.len() == 1 { "is" } else { "are" },
                ),
            ),
        })
    }

    /// Take a character's body away, and let it settle back to reacting.
    pub fn disembody(&self, npc_id: u64, now_ms: u64) -> bool {
        if !self.bodies.unbind(npc_id) {
            return false;
        }
        self.scheduler.set_pace(npc_id, Pace::AMBIENT, now_ms);
        true
    }

    /// What the world offers this character *right now*, as the grammar's own
    /// vocabulary.
    ///
    /// **The same read the situation and the standing task already do**, handed
    /// to the stencil so the three cannot disagree. They did: the prompt named
    /// who was here, the standing task named them again, and the mask let the
    /// character write any string at all — so it wrote a first name, or a
    /// person in another room, and was refused. A grammar built from this is
    /// masked to exactly what the character was told.
    ///
    /// Empty for a character with no body, which is the honest answer: nobody
    /// is standing next to somebody who is nowhere.
    pub fn within(&self, npc_id: u64) -> tools::Within {
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return tools::Within::nowhere();
        };
        // Names, because that is what the grammar binds and what the character
        // was shown. It used to carry the body ids alongside, to look up whose
        // mind was behind each name and ask what it was waiting for; nothing
        // waits any more, so the second half went with the wait.
        let company: Vec<String> = hosted.read(|w| {
            let Some(mine) = w.actor(&body) else {
                return Vec::new();
            };
            w.actors_at(&mine.at.clone())
                .into_iter()
                .filter(|a| a.id != body)
                .map(|a| a.name.clone())
                .collect()
        });
        let base = tools::Within {
            company,
            // Never where it stands — see [`body::reachable`]. Walking to your
            // own room was refused, and refusal is not a lesson.
            places: body::reachable(&hosted, &body),
            // What this body has done too recently to do again. Asked here, at
            // the one place a situation is composed, so no route can build a
            // grammar that has forgotten about it.
            cooling: self.cooldowns.cooling(npc_id),
            feelings: self.feelings.read().unwrap().clone(),
            me: hosted.read(|w| w.actor(&body).map(|a| a.name.clone()).unwrap_or_default()),
            ..Default::default()
        };
        // What the world's own state adds: what this body carries, what stands
        // in the room with it, what is outside. An empty answer to any of them
        // takes the acts that need it out of the grammar, which is how a world
        // without hostiles ends up without `engage`.
        let place = hosted.place_of(&body);
        hosted.sim(|sim| base.clone().from_sim(sim, &body, &place))
    }

    /// Walk a person into the world, beside the character they came to see.
    ///
    /// **A physical interaction is physical.** Until this, "being present" meant
    /// speech delivered straight to one character's inbox while the person had
    /// no body at all: they were not in `Within::company`, so `tell`, `ask`,
    /// `give`, `touch` and `gesture` — every act whose target binds to who is
    /// standing there — were not even in the character's grammar. It heard you
    /// and could not answer you, and went on waiting for somebody to arrive
    /// while you were talking to it.
    ///
    /// So the person gets a body. It stands where the character stands, other
    /// characters in the room see it arrive, and it is taken out again when the
    /// conversation ends — see [`Runtime::leave_world`].
    ///
    /// Returns where they ended up. `None` if the character has no body to
    /// stand beside.
    pub fn enter_world_beside(
        &self,
        npc_id: u64,
        visitor: &str,
        name: &str,
    ) -> Option<npc_map::world::Where> {
        let (hosted, body) = self.body_of(npc_id)?;
        let at = hosted.read(|w| w.actor(&body).map(|a| a.at.clone()))?;
        hosted.with(|w| {
            // Already here is not an error — a console that reopens the same
            // conversation is the ordinary case, and re-entering would log a
            // second arrival for somebody who never left.
            match w.actor(visitor).is_some() {
                true => w.place(visitor, at.clone()).ok(),
                false => w.enter(visitor, name, at.clone()).ok(),
            }
        })?;
        Some(at)
    }

    /// Take a person back out of the world.
    ///
    /// `false` if they were not in it. Everything a body was holding is given
    /// up on the way — see [`npc_map::world::World::leave`].
    pub fn leave_world(&self, visitor: &str) -> bool {
        self.hosted
            .ids()
            .iter()
            .filter_map(|id| self.hosted.get(id))
            .any(|hosted| hosted.with(|w| w.leave(visitor).is_ok()))
    }

    /// Keep every visiting body with the character it came to see.
    ///
    /// **A conversation does not end because somebody walked out of the room.**
    /// A character asked to go to the chronicle goes, and a visitor left behind
    /// is talking to an empty room while the character it came for is two
    /// levels away — so the visitor goes too. Placed rather than walked: a
    /// person at a console is *with* somebody, and making them cover the
    /// distance would mean arriving after the character had moved again.
    ///
    /// Runs in the sweep, after journeys advance, so a visitor follows to where
    /// the character ended up rather than where it set out from.
    fn keep_visitors_with_their_hosts(&self, hosted: &Hosted) {
        let now = crate::api::now_ms();
        let following: Vec<(String, u64)> = self
            .interactions
            .visiting(hosted.id(), now)
            .into_iter()
            .collect();
        if following.is_empty() {
            return;
        }
        for (visitor, npc_id) in following {
            let Some((_, body)) = self.body_of(npc_id) else {
                continue;
            };
            hosted.with(|w| {
                let Some(theirs) = w.actor(&body).map(|a| a.at.clone()) else {
                    return;
                };
                let mine = w.actor(&visitor).map(|a| a.at.clone());
                if mine.as_ref() != Some(&theirs) {
                    let _ = w.place(&visitor, theirs);
                }
            });
        }
    }

    /// One moment of world time, and everything a moment entails.
    ///
    /// **Everything, in one place.** [`environment::advance`] moves journeys on
    /// and hands out what was perceived; a moment is also when a visitor
    /// catches up with whoever it is following and when an abandoned session
    /// gives its body back. Putting the last two in the metronome's closure
    /// instead left them out of every other route by which the world advances —
    /// so a world ticked any other way had visitors standing in rooms the
    /// character had left, and no test could see it because the tests advance
    /// the world directly.
    ///
    /// The order matters: journeys first, so a visitor follows to where the
    /// character ended up rather than where it set out from.
    pub fn moment(&self, hosted: &Hosted) -> environment::Moment {
        let moment = environment::advance(hosted, &self.bodies, &self.scheduler);
        self.keep_visitors_with_their_hosts(hosted);
        self.show_out_the_expired();
        moment
    }

    /// Take out the bodies of anybody whose conversation has gone quiet.
    ///
    /// **A session expiring has to actually remove somebody.** `is_live` makes
    /// an abandoned one read as gone, which is enough for a console and does
    /// nothing whatever to an actor standing in a room — so without this a
    /// visitor who closed the tab stays in the vault for ever, and the
    /// character goes on being told it has company.
    fn show_out_the_expired(&self) {
        for gone in self.interactions.take_expired(crate::api::now_ms()) {
            if gone.mode == tools::Mode::Physical {
                self.leave_world(&gone.body);
            }
        }
    }

    /// Say something to a character on its handset, as a person outside the
    /// world.
    ///
    /// **The same thread the characters use, not a side channel.** A person
    /// messaging a character is one more party on a conversation: it lands in
    /// [`crate::sim::phone`], the character is told about it by the ordinary
    /// sweep, and it answers with the ordinary `message` act. A private pipe
    /// between a console and a mind would be a different thing wearing the same
    /// word — the reply would not be the character speaking from inside the
    /// world, and nothing else in the world could ever see that it had happened.
    ///
    /// Returns what the character is called and whether it can answer. `None`
    /// when it has no body, because there is then nothing to reach it on.
    pub fn message_npc(&self, npc_id: u64, from: &str, text: &str) -> Option<MessageSent> {
        let (hosted, body) = self.body_of(npc_id)?;
        let them = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()))?;
        let sent = hosted.with_sim(|sim| {
            sim.threads.reach(from, &them);
            let _ = sim.threads.send(from, &them, text);
            // Everybody a handset can reach comes from the cast rather than the
            // room, so a person messaging in from outside has to be made known
            // to the world's phones or the character is offered nobody to
            // answer. See [`crate::sim::Sim::enrol`].
            sim.enrol(from);
            MessageSent {
                with: them.clone(),
                waiting_for_them: sim.threads.waiting_for(&them),
                // A character with no handset cannot answer at all — the
                // messaging acts are gated on carrying one — and saying so is
                // better than a console that looks like it is working.
                can_reply: sim.has_phone(&body),
            }
        });
        Some(sent)
    }

    /// The conversation between a person and a character, oldest first.
    pub fn messages_with(
        &self,
        npc_id: u64,
        from: &str,
    ) -> Option<(String, Vec<(String, String)>)> {
        let (hosted, body) = self.body_of(npc_id)?;
        let them = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()))?;
        let said = hosted.sim(|sim| {
            sim.threads
                .direct_between(from, &them)
                .map(|t| {
                    t.messages
                        .iter()
                        .map(|m| (m.from.clone(), m.intent.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        });
        Some((them, said))
    }

    /// Say something on a world's standing channel, as a person outside it.
    ///
    /// **The same channel the cast is on, not an operator broadcast.** There is
    /// already an act that reaches every character regardless of where they are
    /// standing — `pulse::broadcast` — and it is a different thing on purpose:
    /// it puts words into a mind from outside the fiction, and nothing in the
    /// world can see that it happened or answer it. This is a person speaking
    /// on a conversation, so it lands in [`crate::sim::phone`], arrives through
    /// the ordinary sweep, and can be answered with the ordinary `message` act
    /// by any character that has something to say back.
    ///
    /// Returns who is on the channel to hear it. `None` when the world is not
    /// hosted.
    pub fn say_on_channel(&self, world_id: &str, from: &str, text: &str) -> Option<Vec<String>> {
        let hosted = self.hosted.get(world_id)?;
        Some(hosted.with_sim(|sim| {
            // Enrolled first, so somebody speaking for the first time is on the
            // channel before their own words go onto it — otherwise `send`
            // refuses, having correctly found that they are on no such thread.
            sim.enrol(from);
            sim.threads
                .send(from, crate::sim::phone::CHANNEL, text)
                .unwrap_or_default()
        }))
    }

    /// What has been said on a world's standing channel, oldest first.
    ///
    /// Read as `who`, because a channel is named the same way to everybody on
    /// it but only shows what was said after each member arrived.
    pub fn channel(&self, world_id: &str, who: &str) -> Option<Vec<(String, String)>> {
        let hosted = self.hosted.get(world_id)?;
        Some(hosted.sim(|sim| {
            sim.threads
                .by_name_for(who, crate::sim::phone::CHANNEL)
                .map(|t| {
                    t.messages
                        .iter()
                        .skip(t.arrived_at(who))
                        .map(|m| (m.from.clone(), m.intent.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }))
    }

    /// Leave words on something in a world, as a person outside it.
    ///
    /// **The world's half of `post_notice`.** A character can write on a board
    /// it is standing at; this is how whoever runs the world puts something on
    /// one without a body to do it — a notice that was there before the cast
    /// arrived, an instruction from outside, a sign somebody hung years ago.
    ///
    /// It stands the surface up if the map never named one, so a world can grow
    /// a board where it needs one rather than needing its map edited and
    /// reloaded.
    ///
    /// Returns how many lines the surface now holds. `None` when the world is
    /// not hosted.
    pub fn post_in_world(
        &self,
        world_id: &str,
        at: &str,
        on: &str,
        by: &str,
        text: &str,
    ) -> Option<usize> {
        let hosted = self.hosted.get(world_id)?;
        // A place that is not in the map would be a board nobody can ever stand
        // at to read — writable, invisible, and impossible to diagnose from the
        // outside. Refused here rather than written and lost.
        let real =
            hosted.read(|w| npc_map::world::Where::parse(at).is_some_and(|p| w.node(&p).is_some()));
        if !real {
            return None;
        }
        Some(hosted.with_sim(|s| {
            s.post(at, on, by, text);
            s.postings
                .by_name_at(at, on)
                .map(|p| p.lines.len())
                .unwrap_or(0)
        }))
    }

    /// Arm the scheduling half of an act the world accepted — the **stall**.
    ///
    /// Read from [`crate::engine::cooldown::Cost::stall`] rather than named
    /// here, so an act's two costs are chosen together in one table instead of
    /// one being a rate and the other an `if` in this file. Most acts stall for
    /// nothing: only crossing a room and putting a hand on somebody take time
    /// the world can see.
    ///
    /// **The mood is recorded either way.** It rode inside the stall while
    /// `reflect` was the only act that stalled, which made a scheduling
    /// decision quietly load-bearing for a durable record: giving reflection
    /// its honest zero-length stall would have stopped every character's
    /// feeling being written down, and nothing would have said so. They are two
    /// unrelated things about the same act and are now spelled that way.
    fn arm_pause(&self, npc_id: u64, act: &Act, now_ms: u64) {
        if let Some(stall) = crate::engine::cooldown::stall_after(act.tool) {
            self.scheduler
                .pause_for(npc_id, now_ms, stall.as_millis() as u64);
        }
        self.note_feeling(npc_id, act);
    }

    /// Give a character another turn when its act came back with something.
    ///
    /// **The world answered, so the character has to get to use the answer.**
    /// What a board says exists in the act's outcome and nowhere else — nothing
    /// in the world perceives a document being read, so no sweep will ever
    /// deliver it — and a character is otherwise unscheduled the moment its
    /// queue empties. The result was a Maker that read the muster board, had the
    /// contents written into its window, went to sleep, and read the board again
    /// the next time anything woke it.
    ///
    /// Only [`body::ANSWERS`], never every act. A free follow-up after *any* act
    /// is the treadmill: a character that speaks and is immediately asked again
    /// speaks again, into the same silence, until its own window holds nothing
    /// but its own voice.
    fn arm_followup(&self, npc_id: u64, act: &Act) {
        if body::answers(act.tool) {
            self.scheduler.think_again(npc_id);
        }
    }

    /// The world and body a character acts through, if it has one.
    pub fn body_of(&self, npc_id: u64) -> Option<(Arc<Hosted>, String)> {
        let at = self.bodies.bound(npc_id)?;
        Some((self.hosted.get(&at.world)?, at.body))
    }

    /// Put one act into the world, if it is a body's act and there is a body.
    ///
    /// [`Outcome::NotOfTheBody`] when neither is true — a character with no
    /// body, or an act that happens inside a head. Those are the caller's to
    /// record as intent.
    ///
    /// The **world's** verdict comes back rather than the character's intent,
    /// and the caller must keep the two apart: an act that was refused and one
    /// that succeeded must not read the same afterwards, or a character spends
    /// the rest of the day reasoning from a move it never made.
    pub fn act_on_world(&self, npc_id: u64, act: &Act) -> Outcome {
        if !body::is_of_the_body(act.tool) {
            return Outcome::NotOfTheBody;
        }
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return Outcome::NotOfTheBody;
        };
        // Nothing is pushed here. An act changes the world; **the world's next
        // moment is when anyone perceives that**, including the character that
        // acted — one rule, one batched prefill per moment, no path where
        // perception arrives outside the sweep.
        //
        // The actor is not left in the dark meanwhile: a refusal comes back in
        // the same turn, which is the whole reason it is rendered rather than
        // logged.
        let done = body::perform(&hosted, &body, act);
        // **Only what actually happened costs a wait.** A refused act is not a
        // thing the body did, and charging for one would leave a character that
        // mis-named a room standing still for fifteen seconds over a typo.
        if done.happened() {
            // The act, not its name: `act` is two rates wearing one word, and
            // this is the only caller that knows which of them happened.
            self.cooldowns.took_act(npc_id, act);
        }
        done
    }

    /// Put one act into the world and render the single line that goes into the
    /// character's window and the Pulse feed.
    ///
    /// **The world's verdict decides which of two forms this takes.**
    ///
    /// An act that landed is recorded as the act: [`Act::summary`], the same
    /// `tool — intent` shape every act outside a world already has. The world's
    /// own reply to a successful act reads as narration — "You shout, for
    /// anyone within earshot." — and taking that instead made speech the one act that did not
    /// look like an act. That is wrong twice over: in the feed, where it broke
    /// a column of single-word tools, and in the character's own window, where
    /// the model reads its history and is being shown what an act looks like.
    ///
    /// A refusal keeps the world's line, because there the prose *is* the
    /// information — which rooms are reachable, who is actually here. So the
    /// two no longer merely differ in wording; they differ in form, which is a
    /// stronger version of the distinction this path exists to preserve.
    /// **An act that answers keeps the world's line too**, on the same
    /// reasoning as a refusal — see [`body::ANSWERS`]. Reading a document is
    /// not something the room perceives, so the contents are in the outcome and
    /// nowhere else; summarising it would record that the character read
    /// something and drop what it read.
    ///
    /// # Two readers, two lines
    ///
    /// Everything above is about the **feed** — what a person watching Pulse
    /// reads. The character is a different reader with a different need, and
    /// [`Recorded::answer`] is its line: always the world's own verdict, for
    /// every act, because that is the only thing that tells it whether what it
    /// tried actually happened.
    ///
    /// One string served both while nothing carried results back to the model.
    /// Now that `<tool_response>` does, the summary is the wrong thing to send:
    /// it hands a character *its own arguments* back as though they were an
    /// outcome. Measured live, `gesture — for Wailen to see that the room is
    /// ours for now` — which is what the character asked for, not what came of
    /// it. `body::ANSWERS` was an early, narrower version of exactly this
    /// concern, written when the feed was the only channel there was.
    pub fn record_act(&self, npc_id: u64, act: &Act) -> Recorded {
        let outcome = self.act_on_world(npc_id, act);
        let landed = outcome.happened();
        // What the character is told, whatever the verdict. `NotOfTheBody` has
        // no line of its own — nothing in a world happened — so it says so
        // rather than echoing the call back.
        let answer = match outcome.line() {
            Some(line) => line.to_string(),
            None => format!("Nothing in the world answers `{}`.", act.tool),
        };
        // **Both halves, always, in one shape.**
        //
        // The feed used to show one or the other and the choice depended on the
        // verdict: an act that landed showed the *call*, an act refused showed
        // the *world's reply*. So "Wailen Wylde is already on it." sat beside
        // "invite — Yaelis Vayne; …" with nothing saying they were the same
        // kind of event, that one had failed, or which act the sentence was
        // even about.
        //
        // Now every line reads `tool — what was asked → what came of it`, with
        // `✗` in place of the arrow when it did not land. The console splits on
        // exactly those marks to set each part in its own face
        // (`pulse.js::splitAct`), which is why they are single characters with
        // spaces around them and not words.
        let feed = match outcome {
            Outcome::Refused(why) => format!("{} {REFUSED} {why}", act.summary()),
            Outcome::Did(line) => format!("{} {LANDED} {line}", act.summary()),
            // Nothing in a world happened, so there is nothing to arrow to.
            Outcome::NotOfTheBody => act.summary(),
        };
        Recorded {
            feed,
            answer,
            landed,
        }
    }

    /// Hold a world still, or let it go again. `false` if no such world.
    pub fn hold_world(&self, id: &str, still: bool) -> bool {
        let beats = self.metronomes.lock().unwrap();
        let Some(beat) = beats.get(id) else {
            return false;
        };
        if still {
            beat.pause();
        } else {
            beat.resume();
        }
        true
    }

    /// How many moments each hosted world has taken. What a health check reads:
    /// a world whose count has stopped rising has lost its metronome.
    pub fn moments(&self) -> Vec<(String, u64, bool)> {
        self.metronomes
            .lock()
            .unwrap()
            .iter()
            .map(|(id, beat)| (id.clone(), beat.moments(), beat.is_paused()))
            .collect()
    }

    /// Supply the world-clock resolver. Called once at startup, before the
    /// driver runs.
    pub fn set_clock(&self, clock: WorldClock) {
        *self.clock.write().unwrap() = Some(clock);
    }

    /// Supply the character-state resolver. Called once at startup, before the
    /// driver runs.
    pub fn set_persona_source(&self, src: PersonaSource) {
        *self.persona.write().unwrap() = Some(src);
    }

    /// Supply the registers a character may say it is in — the ids of whatever
    /// `<mind>/moods/` holds. Called once at startup, before the driver runs.
    ///
    /// The engine offers what it is given and has no opinion about the list:
    /// which registers a world's people have is that world's content, and a
    /// second copy compiled in here would be free to disagree with the mind.
    pub fn set_feelings(&self, feelings: Vec<String>) {
        *self.feelings.write().unwrap() = feelings;
    }

    /// The registers this daemon's mind holds. What the probe builds its
    /// grammar from, so a scenario runs against the same vocabulary a live
    /// character does.
    pub fn feelings(&self) -> Vec<String> {
        self.feelings.read().unwrap().clone()
    }

    /// Supply the sink that records where bodies are. Called once at startup,
    /// before the driver runs.
    pub fn set_place_sink(&self, sink: PlaceSink) {
        *self.place_sink.write().unwrap() = Some(sink);
    }

    /// Supply the sink that records how characters feel. Called once at
    /// startup, before the driver runs.
    pub fn set_mood_sink(&self, sink: MoodSink) {
        *self.mood_sink.write().unwrap() = Some(sink);
    }

    /// Remember a register a character has just named for itself.
    ///
    /// **Only from an act the world accepted**, and only `pause` names one —
    /// so this sits beside [`Runtime::arm_pause`] rather than inside the act,
    /// for the reason the deadline does: what a room sees is the world's, and
    /// what outlives the process is the registry's.
    ///
    /// A word the mind has no mood for is still recorded. The grammar steers to
    /// the library where there is one, and where there is not the character is
    /// free-decoding — refusing what it said would mean a character that felt
    /// something the daemon had no vocabulary for felt nothing at all.
    fn note_feeling(&self, npc_id: u64, act: &Act) {
        if act.tool != "reflect" {
            return;
        }
        let Some(feeling) = act.args.get("feeling").and_then(|v| v.as_str()) else {
            return;
        };
        let feeling = feeling.trim();
        if feeling.is_empty() {
            return;
        }
        if let Some(sink) = self.mood_sink.read().unwrap().as_ref() {
            sink(npc_id, feeling);
        }
    }

    /// Checkpoint every bound body's room after a moment that moved somebody.
    ///
    /// Reads the world once, under its own lock, and hands each `(character,
    /// place)` to the sink — which decides whether that is worth a record. The
    /// decision lives there rather than here because it is about the durable
    /// record, and this is a driver thread that should not know what a record
    /// costs.
    fn checkpoint_places(&self, hosted: &Hosted) {
        let Some(sink) = self.place_sink.read().unwrap().clone() else {
            return;
        };
        let bound = self.bodies.in_world(hosted.id());
        let standing: Vec<(u64, String)> = hosted.read(|w| {
            bound
                .iter()
                .filter_map(|(npc_id, body)| Some((*npc_id, w.actor(body)?.at.to_string())))
                .collect()
        });
        for (npc_id, at) in standing {
            sink(npc_id, &at);
        }
    }

    /// Supply the already-open substrate for the engine to adopt. Called once
    /// at startup, before [`start`], with the handle
    /// [`crate::npcs::Npcs::load`] opened.
    ///
    /// **Not optional in practice.** Without it the engine opens `--data`
    /// itself, and this daemon then holds two writable handles to one log —
    /// which loses characters rather than failing. [`load`] refuses to build an
    /// engine when this is unset, so a future wiring mistake is a startup error
    /// instead of a slow leak.
    pub fn set_substrate(&self, shared: SharedSubstrate) {
        *self.substrate.write().unwrap() = Some(shared);
    }

    /// Stop one character, ask it two things, and throw the conversation away.
    ///
    /// The whole of it runs **serially**, behind the caller. That looks like a
    /// violation of the engine's standing rule that a fast clock must never wait
    /// on a slow one, and is not: reflecting is what a character does when it
    /// has already chosen to stand still, so the wait costs nothing it had not
    /// spent. The dream this produces a brief for is the part that stays
    /// asynchronous.
    ///
    /// `domain` rotates when the caller does not name one, so the behaviour
    /// space fills evenly instead of being sampled wherever the model prefers —
    /// see [`reflect::DOMAINS`]. `sampled_axes` is a **sample** of the
    /// character's existing dream corpus and must never be all of it.
    ///
    /// The dream is written after this returns, on a thread of its own — the
    /// caller gets the reflection, and the dream lands in the corpus whenever
    /// it lands. See [`Self::dream_now`].
    pub fn reflect(
        self: &Arc<Self>,
        npc_id: u64,
        situation: Option<&str>,
        inner_thoughts: &str,
        feeling: &str,
        domain: Option<&str>,
        sampled_axes: &[String],
    ) -> anyhow::Result<reflect::Reflection> {
        let r = self.reflect_with(
            npc_id,
            situation,
            inner_thoughts,
            feeling,
            domain,
            sampled_axes,
            &mut |_| true,
        )?;
        if let (Some(brief), Some(assumption)) = (r.brief.clone(), r.assumption.clone()) {
            let rt = Arc::clone(self);
            let spawned = std::thread::Builder::new()
                .name(format!("dream-{npc_id}"))
                .spawn(move || rt.dream_if_free(npc_id, &brief, &assumption));
            if let Err(e) = spawned {
                tracing::warn!("npc {npc_id}: the dream could not be started — {e}");
            }
        }
        Ok(r)
    }

    /// One reflection, handing `on_reflection` the line that crosses back the
    /// moment the first question is answered. See [`reflect::Reflect::run`].
    // Each is a separate axis of one reflection — whose, where it says it is,
    // what it is thinking and feeling, which cell, steered from what, and who
    // hears the answer first — and every caller supplies all of them.
    #[allow(clippy::too_many_arguments)]
    fn reflect_with(
        &self,
        npc_id: u64,
        situation: Option<&str>,
        inner_thoughts: &str,
        feeling: &str,
        domain: Option<&str>,
        sampled_axes: &[String],
        on_reflection: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<reflect::Reflection> {
        let who = self
            .persona_of(npc_id)
            .ok_or_else(|| anyhow::anyhow!("no such character, or it has no persona"))?;

        // Failing that, where the character is in the world's own words: the
        // percept of the room its body stands in and who is there with it. The persona's own
        // `situation` is empty for every authored character — it is filled per
        // tick from a world *delta*, and there is no delta here — so a
        // reflection taken from it opened on nothing, and a character asked to
        // dream about its day dreamed about an office.
        //
        // **The character's own account first.** The act passes it — where the
        // character says it is and what is going on — and the reflection is
        // framed on that: `docs/reflection_and_dreams.md` §3. The percept is the
        // fallback for a caller that gives none, never a replacement for one
        // that did.
        let situation = situation
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| self.situation_of(npc_id))
            .unwrap_or_else(|| who.situation.clone());

        let domain = domain.map(str::to_string).unwrap_or_else(|| {
            let n = self.reflect_domain.fetch_add(1, Ordering::Relaxed);
            reflect::DOMAINS[n % reflect::DOMAINS.len()].to_string()
        });

        // Cloned out rather than read through the guard: this runs for the
        // length of several decodes, on a thread that reads `minds` again to
        // hand the answer back.
        let minds = self
            .minds
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no engine — the daemon is still loading"))?;
        minds.reflect(
            npc_id,
            &who.as_persona(),
            who.mode,
            &situation,
            inner_thoughts,
            feeling,
            &domain,
            sampled_axes,
            on_reflection,
        )
    }

    /// Dream a brief and keep the dream. Logged, never returned: nobody is
    /// waiting on it, and a dream that did not land costs a dream.
    fn dream_now(&self, npc_id: u64, brief: &str, assumption: &str) {
        let started = Instant::now();
        let Some(minds) = self.minds.read().unwrap().clone() else {
            return;
        };
        let Some(who) = self.persona_of(npc_id) else {
            return;
        };
        match minds.dream(npc_id, &who.as_persona(), brief, assumption) {
            Ok(kept) => tracing::info!(
                "npc {npc_id}: dreamt, {} line(s) kept in {:?} — {} dream(s) now; it opens: {}",
                kept.lines.len(),
                started.elapsed(),
                minds.dreams_kept(npc_id),
                kept.lines.first().map(String::as_str).unwrap_or_default(),
            ),
            Err(e) => tracing::warn!("npc {npc_id}: the dream was not kept — {e:#}"),
        }
    }

    /// Claim this character's one reflection slot, if a reflection can run.
    ///
    /// `false` only when this daemon cannot reflect at all — no mind authoring
    /// the questions. Never for being soon after another: the character is held
    /// until its answer lands, so it cannot ask twice at once, and the work that
    /// is one at a time is the dream — see [`Self::claim_dream`].
    fn can_reflect(&self) -> bool {
        self.minds
            .read()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.can_reflect())
    }

    /// Take this character's one dream slot, if it is free. Held until the
    /// returned guard drops. See [`Self::dreaming`].
    fn claim_dream(&self, npc_id: u64) -> Option<DreamSlot<'_>> {
        // **The lock is released before any guard exists.** A guard dropped
        // while this lock is held deadlocks, because dropping one takes the
        // lock to hand the slot back — and `then_some` builds its value
        // eagerly, so on a taken slot it built a guard and dropped it on the
        // spot, inside the lock, and the thread waited on itself for ever. It
        // would also have handed back the slot it had just been refused.
        let free = self.dreaming.lock().unwrap().insert(npc_id);
        free.then(|| DreamSlot { rt: self, npc_id })
    }

    /// Dream a brief if this character is not already dreaming, and say so if
    /// it is. For the route, whose reflection has already asked for the brief.
    fn dream_if_free(&self, npc_id: u64, brief: &str, assumption: &str) {
        match self.claim_dream(npc_id) {
            Some(_slot) => self.dream_now(npc_id, brief, assumption),
            None => tracing::info!(
                "npc {npc_id}: a dream is already being written, so this brief is not dreamt"
            ),
        }
    }

    /// Answer a `reflect` with a reflection, without anybody waiting on it.
    ///
    /// **The act's answer is the reflection's first line.** `answers` is every
    /// answer the turn owes, in call order, with the world's own line for the
    /// reflect at `slot` — what the character is told if the reflection cannot
    /// give it anything better. The whole list is held back until the first
    /// question is answered, so the results still arrive one per call, in the
    /// order the calls were made, at the head of the character's next turn.
    ///
    /// **Asynchronous, and only for this character.** The reflection runs on a
    /// thread of its own. The tick driver goes straight on to everybody else;
    /// this character alone is held, so the room cannot give it a turn before
    /// its reflect has been answered. It is released the moment the answer is
    /// in — about as long as one question takes — and the rest of the
    /// reflection and the dream after it carry on behind it, on the same
    /// thread, with nobody waiting.
    pub fn begin_reflection(
        self: &Arc<Self>,
        npc_id: u64,
        situation: String,
        inner_thoughts: String,
        feeling: String,
        owed: Owed,
    ) {
        self.scheduler.hold(npc_id);
        let rt = Arc::clone(self);
        let fallback = owed.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("reflect-{npc_id}"))
            .spawn(move || {
                rt.reflect_in_background(npc_id, &situation, &inner_thoughts, &feeling, owed)
            });
        if let Err(e) = spawned {
            tracing::warn!("npc {npc_id}: the reflection could not be started — {e}");
            // No reflection, so what came back is the world's own line.
            let came_back = fallback
                .answers
                .get(fallback.slot)
                .cloned()
                .unwrap_or_default();
            self.scheduler.amend_act(
                npc_id,
                &fallback.row,
                format!("{} {LANDED} {came_back}", fallback.row),
            );
            if let Some(m) = self.minds.read().unwrap().as_ref() {
                m.deliver_outcomes(npc_id, fallback.answers);
            }
            self.scheduler.release(npc_id);
        }
    }

    /// The reflection thread's whole life: ask, answer the character, then
    /// dream. See [`Self::begin_reflection`].
    fn reflect_in_background(
        &self,
        npc_id: u64,
        situation: &str,
        inner_thoughts: &str,
        feeling: &str,
        owed: Owed,
    ) {
        /// Whatever happens below — an error, a panic — the character is let
        /// go. A character left held is one that never thinks again, with
        /// nothing anywhere saying why.
        struct Done<'a> {
            rt: &'a Runtime,
            npc_id: u64,
        }
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                // Only if it is still held. The answer normally let it go
                // minutes ago, and `release` also brings a character forward —
                // so an unconditional one here woke it for nothing every time
                // a dream finished.
                if self.rt.scheduler.is_held(self.npc_id) {
                    self.rt.scheduler.release(self.npc_id);
                }
            }
        }
        let _done = Done { rt: self, npc_id };

        let started = Instant::now();
        let Some(minds) = self.minds.read().unwrap().clone() else {
            return;
        };
        // The reflect's answer — the reflection's line, or the world's own when
        // the reflection gave none — goes in its slot, and its row is completed
        // with the same words: what came back, after the arrow.
        let deliver = |mut owed: Owed, line: Option<&str>| {
            let slot = owed.slot;
            if let (Some(line), Some(answer)) = (line, owed.answers.get_mut(slot)) {
                *answer = line.to_string();
            }
            let came_back = owed.answers.get(slot).cloned().unwrap_or_default();
            self.scheduler.amend_act(
                npc_id,
                &owed.row,
                format!("{} {LANDED} {came_back}", owed.row),
            );
            minds.deliver_outcomes(npc_id, owed.answers);
            self.scheduler.release(npc_id);
        };
        let axes = minds.dreamt_axes(npc_id, SAMPLED_AXES);
        let mut owed = Some(owed);
        let mut slot: Option<DreamSlot<'_>> = None;
        let result = self.reflect_with(
            npc_id,
            Some(situation),
            inner_thoughts,
            feeling,
            None,
            &axes,
            // Answer the character, then decide whether this reflection goes on
            // to a dream: only if the character's dream slot is free, and then
            // it is held through the dream and handed back when this returns.
            &mut |line| {
                if let Some(answers) = owed.take() {
                    deliver(answers, Some(line));
                    tracing::info!(
                        "npc {npc_id}: reflect answered in {:?}, before any dream — {line}",
                        started.elapsed()
                    );
                }
                slot = self.claim_dream(npc_id);
                slot.is_some()
            },
        );
        // A reflection that failed before it answered still owes the character
        // its answer, and the answer is that nothing came —
        // [`body::NO_REFLECTION`], the line already in its slot.
        if let Some(answers) = owed.take() {
            deliver(answers, None);
        }
        let r = match result {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("npc {npc_id}: reflection failed — {e:#}");
                return;
            }
        };
        // The turn count and each turn's tokens, because a slow reflection is
        // either too many tokens or too slow a token, and the two have different
        // fixes — the elapsed time alone cannot say which.
        tracing::info!(
            "npc {npc_id}: reflection done in {:?} — {} turn(s), {} token(s) {:?} ({} axis/axes \
             sampled){}",
            started.elapsed(),
            r.tokens.len(),
            r.tokens.iter().sum::<usize>(),
            r.tokens,
            axes.len(),
            r.fault
                .as_deref()
                .map(|f| format!(" — brief fault: {f}"))
                .unwrap_or_default()
        );
        if slot.is_none() {
            tracing::info!(
                "npc {npc_id}: a dream is already being written, so this reflection stopped at \
                 its answer"
            );
            return;
        }
        match (r.brief.as_deref(), r.assumption.as_deref()) {
            (Some(brief), Some(assumption)) => self.dream_now(npc_id, brief, assumption),
            _ => tracing::warn!("npc {npc_id}: the reflection produced no brief, so no dream"),
        }
    }

    fn persona_of(&self, npc_id: u64) -> Option<OwnedPersona> {
        let mut who = {
            let g = self.persona.read().unwrap();
            g.as_ref().and_then(|f| f(npc_id))
        }?;
        // The building it is standing in, which the authored record knows
        // nothing about — it comes from the map and the body. Only that one: a
        // world holds more than one building, and a character told about all
        // of them as "where you work" places itself in whichever it read most
        // about.
        if let Some((hosted, body)) = self.body_of(npc_id) {
            if let Some((key, text)) = self.building_of(&hosted, &body) {
                who.building = key;
                who.place = text;
            }
        }
        Some(who)
    }

    /// Every building of a world, as anyone living in it would describe it —
    /// [`describe::places`], keyed by part.
    ///
    /// Cached per world. Generating it is cheap but not free, and every
    /// character in a building would otherwise re-render the identical text on
    /// every conversation it opens.
    fn places_of(&self, hosted: &Hosted) -> Vec<(String, String)> {
        if let Some(known) = self.places.lock().unwrap().get(hosted.id()) {
            return known.clone();
        }
        let parts = hosted.read(|w| describe::places(w.map()));
        self.places
            .lock()
            .unwrap()
            .insert(hosted.id().to_string(), parts.clone());
        parts
    }

    /// The building a body is standing in: its [`identity::building_key`] and
    /// what anyone who lives there knows of it. `None` for a body that is not
    /// in the world, or standing somewhere no part of the map describes.
    fn building_of(&self, hosted: &Hosted, body: &str) -> Option<(String, String)> {
        let part = hosted.read(|w| {
            let area = w.actor(body)?.at.area.clone();
            describe::enclosing(w.map(), &area).map(str::to_string)
        })?;
        let text = self
            .places_of(hosted)
            .into_iter()
            .find(|(id, _)| *id == part)?
            .1;
        Some((identity::building_key(hosted.id(), &part), text))
    }

    /// Where a character is right now, as the world puts it: the room its body
    /// stands in and who is there. `None` for a character with no body.
    fn situation_of(&self, npc_id: u64) -> Option<String> {
        let (hosted, body) = self.body_of(npc_id)?;
        let seen = hosted.read(|w| perceive::percept(w, &body));
        (!seen.trim().is_empty()).then_some(seen)
    }

    /// What time it is in this character's world.
    ///
    /// Zero before a clock is installed — a window between construction and
    /// startup that the driver is not running in, so nothing reads it.
    pub fn world_ms(&self, npc_id: u64) -> u64 {
        let g = self.clock.read().unwrap();
        g.as_ref().map_or(0, |f| f(npc_id))
    }

    /// Whether the engine is loaded and the cast is thinking.
    pub fn is_ready(&self) -> bool {
        self.progress.is_ready()
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// Ask the driver thread to stop at its next quiet moment.
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }
}

/// What the loader needs that is not on the [`Runtime`].
pub struct LoadPlan {
    /// The characters to wake, each with the world it belongs to and the room
    /// it was last standing in. From the substrate's character store.
    ///
    /// Both travel with the id because waking a character and putting it back
    /// in its body are one step on a restart: a character woken without its
    /// world would think for a while about a place it is not standing in, and
    /// one woken without its room would do it from the front door.
    pub cast: Vec<Casting>,
    /// Every personality id the mind declares.
    ///
    /// Not the cast: this is what a *layer directory* can be named after, which
    /// is a personality (`layers/memory/zen/`), not an instantiated character.
    /// A world's biographies exist whether or not anyone has cast them.
    pub characters: Vec<String>,
    /// Who each personality, character and world *is*, for the projection's
    /// identity collections. Read in `main` where the registries live and
    /// installed on the loader thread — see [`crate::engine::identity`].
    pub authored: identity::Authored,
    /// The world clock at startup.
    pub world_ms: u64,
    /// Retire every character's conversation before the cast wakes, so each
    /// opens a fresh one — `--forget-conversations`. See
    /// [`Minds::forget_conversations`].
    pub forget_conversations: bool,
    /// Retire every dream every character has kept before the cast wakes —
    /// `--forget-dreams`. See [`Minds::forget_dreams`].
    pub forget_dreams: bool,
}

/// Begin loading, on its own thread. Returns immediately.
///
/// The HTTP server is already bound by the time this is called, so
/// `GET /v1/status` answers with a loading snapshot from the first request
/// onward rather than from whenever the model finished.
pub fn start(rt: Arc<Runtime>, plan: LoadPlan) {
    let handle = tokio::runtime::Handle::try_current().ok();
    std::thread::Builder::new()
        .name("npcd-loader".into())
        .spawn(move || {
            if let Err(e) = load(&rt, &plan, handle) {
                // Fatal, and it has to be: a daemon that answers requests with
                // a half-loaded engine shows the console something present and
                // broken, which is harder to diagnose than something absent.
                tracing::error!("engine load failed: {e:#}");
                std::process::exit(1);
            }
        })
        .expect("spawn the loader thread");
}

fn load(
    rt: &Arc<Runtime>,
    plan: &LoadPlan,
    rt_handle: Option<tokio::runtime::Handle>,
) -> anyhow::Result<()> {
    let p = &rt.progress;
    let spec = model::spec();

    // ── the model ──────────────────────────────────────────────────────────
    p.set_step(LoadStep::Model);
    p.set_detail(format!("{} {}", spec.name, spec.quant));
    tracing::info!(
        "loading {} {} ({:.1} GB) from {}",
        spec.name,
        spec.quant,
        spec.bytes as f64 / 1e9,
        spec.repo
    );

    // The download is async; the load is not. A local current-thread runtime
    // does the fetch and is dropped before the model load, exactly as zend
    // does — a nested `block_on` inside the ambient runtime would panic.
    let device = candle::Device::new_cuda(0)
        .map_err(|e| anyhow::anyhow!("no CUDA device: {e}. npcd needs a GPU to run an engine."))?;
    // A prefill forward carries 8 192 tokens here, not the 2 048 default.
    //
    // A wave's fixed cost is paid per slab regardless of width. The layer
    // ingest's documents average ~1 900 tokens, so at the default budget every
    // document was its own slab paying the whole fixed sweep — measured 1.9 s a
    // document with the GPU pinned at 100%, which is a 7× gap to the
    // batched-forward gate's rate for work that is pure prefill.
    //
    // The default is conservative because interactive serving rarely has more
    // than a turn queued and a starved forward wastes the same fixed cost on
    // fewer tokens. This daemon is the other case: the ingest queues a windowful
    // of documents at a time, and a cast of characters ticking queues many
    // small prefills at once. Both reliably keep a wide forward fed.
    // The co-resident models, if this deployment configured any. Read before the
    // engine is built and **before the long checkpoint load**, so a wrong path
    // is a refusal at second zero rather than a five-minute load followed by
    // one. A daemon with no `guests.yaml` gets an empty registry, which costs
    // one atomic load per scheduler pass and nothing else.
    let guests = crate::guests::load(&rt.data).map_err(|e| anyhow::anyhow!("guests.yaml: {e}"))?;
    if !guests.is_empty() {
        tracing::info!(
            "co-resident guests configured: {:?} — normal inference stops between waves to \
             serve them",
            guests.configured()
        );
    }

    // **The engine adopts the daemon's substrate; it does not open one.**
    //
    // The character registry opened `--data/.substrate/` at startup and goes on
    // appending character records to it. One `.substrate/` admits exactly one
    // writable handle per process — the log is opened read-write and unlocked,
    // so a second is a second append cursor and a second record index, and a
    // compaction carries forward only what its own handle walked. This daemon
    // ran that way and lost every character created while it was up.
    //
    // Naming a path here instead would also revive an older failure: unset, the
    // conversation layer falls back to the *process working directory*, so a
    // daemon launched from the repo root wrote its redo log to
    // `candle/.substrate` while the character store sat correctly under
    // `--data`. `/v1/substrate/storage` reports on `--data` and showed 65 MB
    // while the real log reached **190 GB** across a morning of re-ingests. It
    // filled the disk, and surfaced as a linker error.
    //
    // A missing handle is a refusal rather than a fallback: both failures above
    // are silent, and neither is worth risking to save a startup error.
    let shared = rt.substrate.read().unwrap().clone().ok_or_else(|| {
        anyhow::anyhow!(
            "the engine was started before the substrate was handed to it — call \
             `Runtime::set_substrate` with `Npcs::substrate()` first"
        )
    })?;
    let mut builder = model::model()
        .builder()
        .guests(guests)
        .prefill_pass_tokens(PREFILL_PASS_TOKENS)
        .substrate(shared);
    // The checkpoint's own dialect and sampling. Captured before the builder is
    // consumed, and handed to every conversation — a second opinion about either
    // would run the model outside the settings it was tuned under.
    // Sampling is the checkpoint's, taken from `SamplingConfig::for_gguf_architecture`.
    //
    // **This daemon used to state Qwen3.5's published numbers here, and that was a
    // workaround for a bug one layer down.** The auto-detect reported
    // `temp=0.7, top_k=0, top_p=0.9`, which reads like bad checkpoint metadata and is
    // not: it is `for_gguf_architecture`'s *unknown architecture* fallback, verbatim.
    // `qwen35` was missing from that table while the whole lineage loaded through it.
    //
    // Restating the four visible numbers here fixed what could be seen and hid what
    // could not. `sampling()` replaces the config wholesale, and the literal was built
    // from `SamplingConfig::default()` — so every steering field came back zeroed,
    // including `force_segment_close_after`, the hard cap that rewrites the next token
    // to `</think>`. A checkpoint that stops closing its own reasoning block then runs
    // to `max_response_tokens` and the caller discards the entire decode as
    // all-reasoning: an empty document, reported as a success. A stock checkpoint hides
    // it by always self-closing; three abliterated ones did not, and were blamed for it.
    //
    // The numbers themselves were right and now live where they belong, beside the
    // architecture that publishes them.
    let conv_config = builder.conversation_config();
    let engine = {
        // The download is async (hf-hub, reqwest); the load that follows is not.
        // A local current-thread runtime does the fetch and is dropped before
        // the load — a nested `block_on` inside the ambient runtime would panic.
        let download = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let built = download.block_on(async {
            builder.engine_with_progress(
                &device,
                // Per transformer block, which is what makes the bar move
                // during the part of startup that actually takes the time.
                Some(&|loaded, total| p.set_progress(loaded as u64, total as u64)),
            )
        });
        drop(download);
        built?
    };

    // Back into the ambient runtime so anything spawned below has a context.
    let _guard = rt_handle.as_ref().map(|h| h.enter());

    // ── the substrate ──────────────────────────────────────────────────────
    //
    // The engine's own construction replays the redo log, so by the time we
    // have an engine the substrate is already open. The step is still shown,
    // because on a large log it is where the time goes and an operator watching
    // a slow start needs to see which phase owns it.
    p.set_step(LoadStep::Substrate);
    let (done, total, _) = engine.substrate_reload_status().snapshot();
    p.set_progress(done as u64, total as u64);
    tracing::info!("substrate: {done} of {total} turns restored");

    // **The engine goes on the runtime here, not at the end of startup.**
    //
    // It used to be published only after every layer had been ingested and
    // every character woken, so on a real mind — twenty minutes of ingest —
    // every engine-backed route answered 503 for the whole of it. That is
    // exactly the window in which somebody watching a slow start wants to ask
    // the daemon something: what the guests are, how the ingest is going,
    // whether the model is the one they expected.
    //
    // Safe because nothing downstream holds the lock for longer than one call
    // (see [`SharedEngine`]) — the ingest takes it to create a window's
    // conversations and gives it straight back, so a guest submission never
    // waits behind a layer. The routes that need the *cast* still report an
    // empty one until the wake phase, which is true rather than a refusal.
    let engine: SharedEngine = Arc::new(Mutex::new(engine));
    // Retention is read from the mind's own schema, so a deployment that wants
    // whole transcripts omits the block and nothing changes for it.
    let keep_turns = crate::engine::schema::turn_retention(rt.mind.as_deref());
    // The reflection's two user turns, from the same schema. Read here rather
    // than per request: they are authored content and a reflection sends them
    // verbatim, so re-reading the file mid-run would let two reflections in the
    // same minute be asked different questions.
    let reflection = crate::engine::schema::reflection(rt.mind.as_deref());
    *rt.minds.write().unwrap() = Some(Arc::new(
        Minds::new(Arc::clone(&engine), conv_config.clone())
            .keeping_turns(keep_turns)
            .asking(reflection),
    ));

    // ── the mind's layers ──────────────────────────────────────────────────
    //
    // Every layer that declares an `ingest_unit:` and has a folder becomes turns
    // on a conversation of its own — one per layer for shared ground, and one
    // per *character* where the layer is theirs. That split is not cosmetic: a
    // life story is per-character, and three biographies in one shared
    // conversation is every character able to recall every other character's
    // childhood.
    //
    // In dependency order, lowest first, memory last — see `ingest::ORDER`.
    //
    // This phase used to walk the tree, count what it found and write nothing,
    // under a label that said "Ingesting mind layers" over a world whose
    // sixty-six documents no character could ever reach.
    // The mind's real projection, parsed once. Everything written from here on
    // is written under it — which is what gives a document something to gather
    // from and therefore a provenance signature. `None` falls back to a
    // synthetic schema that cannot gather, and says so.
    let mut projection = schema::build(rt.mind.as_deref(), "battle-cities");
    if projection.is_none() && rt.mind.is_some() {
        tracing::warn!(
            "no usable projection schema — documents will be written but will produce \
             no provenance signatures, so nothing will gather them"
        );
    }

    // ── the acts ───────────────────────────────────────────────────────────
    //
    // Before the layers, deliberately. Every layer frames on the shared system
    // prompt and the acts are part of it, so a document prefilled while they are
    // still absent captures its KV — and the wide-Q signature the gather matches
    // against — under a prompt no character will ever think under. Nothing
    // would fail; retrieval would simply be worse than it should be, for the
    // life of that substrate.
    //
    // **Every act, installed once; each turn shows the ones it can take.** The
    // world's catalog and the reflection's two answers go into one collection,
    // and an acting turn names its members from the same `specs_within` its
    // grammar is compiled from — so the list a character reads and the mask it
    // decodes under are one computation and cannot disagree.
    p.set_step(LoadStep::Tools);
    let acts: Vec<&Tool> = tools::CATALOG.iter().chain(reflect::ASKED).collect();
    let total = acts.len() as u64;
    p.set_progress(0, total);
    if let Some(built) = projection.as_mut() {
        let installed = tools::install(&mut built.builder, acts, |n, t| {
            p.set_detail(t.name);
            p.set_progress(n as u64, total);
        })?;
        tracing::info!(
            "tools: {installed} installed into the prompt, each shown on the turns that can \
             take it"
        );
    }

    // ── the prompt a character thinks under ────────────────────────────────
    //
    // **Everything goes into the schema before anything is written under it.**
    // The ordering rule `LoadStep::Tools` states applies to all of it: a layer
    // document prefilled while the system prompt is still incomplete captures
    // its signature — and the wide-Q the gather matches against — under a
    // prompt no character will ever think under.
    //
    // Who everybody is. A collection, so each member seals once and is selected
    // per turn: one copy of the vault for the world rather than one per Maker
    // standing in it.
    let identities = match projection.as_mut() {
        Some(p) => {
            // One member per building of every world, so a character is pinned
            // to the one it is standing in — see [`Runtime::building_of`].
            let places: BTreeMap<String, String> = rt
                .hosted
                .ids()
                .into_iter()
                .filter_map(|id| rt.hosted.get(&id).map(|hosted| (id, hosted)))
                .flat_map(|(id, hosted)| {
                    rt.places_of(&hosted).into_iter().map(move |(part, text)| {
                        (identity::building_key(&id, &part), prompt::building(&text))
                    })
                })
                .collect();
            identity::install(&mut p.builder, &plan.authored, &places)?
        }
        None => identity::Installed::default(),
    };

    if let (Some(p), Some(minds)) = (projection.as_ref(), rt.minds.read().unwrap().as_ref()) {
        minds.set_projection(Projected {
            prompt: p.prelude.clone(),
            builder: p.builder.clone(),
            layer: p.layer,
            group: p.group,
            identities: identities.clone(),
            // After everything is installed, so the acts and identities are in
            // it: a conversation written under a different set is superseded on
            // its next open rather than rejoined.
            frame: frame_fingerprint(&schema::frame(&p.builder)),
        });
    }

    p.set_step(LoadStep::Layers);
    match &rt.mind {
        Some(dir) => {
            let units = crate::projection::ingest_units(&rt.mind_handle);
            let sources = ingest::sources(dir, &units, &plan.characters);
            if sources.is_empty() {
                tracing::info!("mind: no layer declares content to ingest");
            }
            for source in &sources {
                let (turns, mut report) = ingest::pending(source, &rt.ledger)?;
                let total = turns.len();
                ingest::announce(p, source, 0, total);

                if total > 0 {
                    let failed = ingest_layer(
                        &engine,
                        &conv_config,
                        projection.as_ref(),
                        source,
                        &turns,
                        &rt.ledger,
                        p,
                    )?;
                    report.written -= failed;
                    report.failed += failed;
                }
                // A final flush at the layer boundary. The windows inside
                // `ingest_layer` have been flushing as they seal — see the note
                // there — so this only catches a layer that ingested nothing.
                rt.ledger.flush();
                tracing::info!(
                    "layer {}: {} written, {} unchanged, {} failed ({})",
                    source.label(),
                    report.written,
                    report.unchanged,
                    report.failed,
                    source.unit
                );
            }

            // ── the lives ──────────────────────────────────────────────────
            //
            // After the world, because a life is lived against it: an episode's
            // provenance is captured relative to what is already in the
            // substrate, so the world has to be there first for the signatures
            // to attach to anything.
            //
            // Each character's episodes run in date order, and each one's
            // `<tool_call>` blocks are executed as it lands — so a belief is
            // formed at the point in the life where it formed, by the same
            // mechanism a belief would be formed at run time.
            for (who, dir) in life::lives(dir, &plan.characters) {
                let (episodes, rejected) = life::episodes(&dir, &who);
                for r in &rejected {
                    // A life with a hole in it is still a life, but the author
                    // has to be told which document fell out.
                    tracing::warn!("life {who}: {}", r.message());
                }
                if episodes.is_empty() {
                    continue;
                }
                p.set_unit("episodes");
                p.set_detail(format!("life · {who}"));
                p.set_progress(0, episodes.len() as u64);
                match run_life(
                    &engine,
                    &conv_config,
                    projection.as_ref(),
                    &who,
                    &episodes,
                    &rt.ledger,
                    p,
                ) {
                    Ok(lived) => tracing::info!(
                        "life {who}: {} episode(s), {} belief(s), {} relationship(s), \
                         {} intention(s){}",
                        lived.episodes,
                        lived.beliefs,
                        lived.relationships,
                        lived.intents,
                        if lived.rejected == 0 {
                            String::new()
                        } else {
                            format!(" — {} call(s) refused", lived.rejected)
                        }
                    ),
                    Err(e) => tracing::warn!("life {who}: not run — {e:#}"),
                }
                rt.ledger.flush();
            }
        }
        None => tracing::info!("mind: none — nothing to ingest"),
    }

    // ── a clean slate, when asked for ──────────────────────────────────────
    //
    // Before the cast wakes, because a character opens its conversation on its
    // first thought: retired here, nothing is left for it to rejoin, and the
    // open it makes next is a fresh one.
    if plan.forget_conversations {
        if let Some(minds) = rt.minds.read().unwrap().as_ref() {
            let retired: usize = plan
                .cast
                .iter()
                .map(|c| minds.forget_conversations(c.npc_id))
                .sum();
            tracing::info!(
                "conversations: {retired} retired across {} character(s) — every character \
                 starts a fresh conversation",
                plan.cast.len()
            );
        }
    }
    if plan.forget_dreams {
        if let Some(minds) = rt.minds.read().unwrap().as_ref() {
            let retired: usize = plan
                .cast
                .iter()
                .map(|c| minds.forget_dreams(c.npc_id))
                .sum();
            tracing::info!(
                "dreams: {retired} retired across {} character(s) — every character starts \
                 with nothing dreamt",
                plan.cast.len()
            );
        }
    }

    // ── the cast ───────────────────────────────────────────────────────────
    p.set_step(LoadStep::Waking);
    p.set_progress(0, plan.cast.len() as u64);
    let now = 0;
    let mut embodied = 0;
    let mut recalled = 0;
    let mut felt = 0;
    for (i, casting) in plan.cast.iter().enumerate() {
        let id = casting.npc_id;
        let world = &casting.world_id;
        rt.scheduler.wake(id, now, plan.world_ms);
        // Back into the body it had. Entering one that is already there is not
        // an error, so this is the same call the create path makes and the two
        // do not have to agree about anything beyond the id.
        let name = rt
            .persona_of(id)
            .map(|p| p.name)
            .unwrap_or_else(|| Runtime::body_id(id));
        // No home named here: the personality knows where its characters
        // belong, and the persona source does not carry it. The room the
        // character was last in is consulted first anyway, so the door is only
        // reached for one that has none — and the create path, which does know
        // the home, named it then.
        match rt.embody_in_world(id, world, None, &name, casting.at.as_deref(), now) {
            Ok(true) => {
                embodied += 1;
                if casting.at.is_some() {
                    recalled += 1;
                }
            }
            Ok(false) => {}
            Err(e) => tracing::error!("npc {id}: no body in `{world}` — {e:#}"),
        }
        // **And back into the register it was in.** A mood is what a character
        // *is* between one thought and the next, and it lives in the window —
        // which a restart empties. Without this a character that had spent the
        // afternoon getting angrier came back with no sign of it, while its
        // conversation history said otherwise, which is the same wrong as
        // returning it to the arrival door.
        //
        // Delivered as an arrival rather than written into the prompt: the
        // prompt is rendered once when a conversation opens, so a register in
        // it would go on asserting an afternoon's mood for ever.
        if let Some(mood) = &casting.mood {
            felt += 1;
            rt.scheduler.deliver(
                id,
                plan.world_ms,
                crate::engine::event::Salience::IDLE,
                crate::engine::event::EventKind::Description {
                    text: format!(
                        "What you were feeling, when you last stopped to notice, was {mood}."
                    ),
                },
            );
        }
        p.set_progress(i as u64 + 1, plan.cast.len() as u64);
    }
    tracing::info!(
        "cast: {} character(s) awake, {embodied} standing in a world \
         ({recalled} back where they were, {felt} back in the register they were in)",
        rt.scheduler.population()
    );

    // The engine was published at the substrate step — see there. What changes
    // here is only that the load is over.
    p.mark_ready();
    tracing::info!("engine ready in {:?}", rt.uptime());

    drive(Arc::clone(rt));
    Ok(())
}

/// Drain a turn to `Done`, keeping the projection events it streams.
///
/// **The events are the provenance signature.** Each one records what a
/// projection selected out of what was already in the substrate at that moment —
/// so an episode of a life carries the world documents it happened against, and
/// a later scan over that memory has a hook rather than having to rediscover the
/// relationship from surface text.
///
/// Borrows the handle, because `finish_turn` consumes it: the stream has to be
/// fully drained before the turn can be sealed.
pub(crate) fn drain(
    handle: &TurnHandle,
    addr: &str,
) -> (Option<TurnResponse>, Vec<ProjectionEvent>) {
    let mut events = Vec::new();
    let mut response = None;
    for ev in handle.stream() {
        match ev {
            TurnEvent::Projection(e) => events.push(e),
            TurnEvent::Done(r) => {
                response = Some(r);
                break;
            }
            TurnEvent::Error(e) => {
                tracing::warn!("{addr}: prefill failed — {e}");
                break;
            }
            _ => {}
        }
    }
    (response, events)
}

/// Persist a turn's provenance signature, after the turn is sealed.
///
/// The seal-time projection is appended to whatever streamed during the turn:
/// the streamed events are the composition as it was being built, and this is
/// what it settled at. zend persists the same pair for the same reason.
///
/// Failure is logged, never propagated — a document whose signature did not
/// persist is still a document in the substrate, and refusing the whole ingest
/// over a lost hook would trade something for nothing.
pub(crate) fn persist_signatures(
    conv: &mut Sequence,
    response: &TurnResponse,
    events: &mut Vec<ProjectionEvent>,
    addr: &str,
) {
    if let Some(e) = conv.projection_event(&response.stats) {
        events.push(e);
    }
    if events.is_empty() {
        return;
    }
    if let Err(e) = conv.persist_projection_events(events) {
        tracing::warn!("{addr}: provenance signature not persisted — {e:?}");
    }
}

/// Write one layer's documents into the substrate, a windowful at a time.
///
/// Returns how many failed. The shape is zend's calibration ingest and the
/// reasoning is in [`crate::engine::ingest`]: **one conversation per document**,
/// created a windowful at a time in a single pipelined call, every turn submitted
/// before any is awaited, so the whole window co-batches into one prefill
/// forward.
///
/// The order of operations is the whole point and is easy to undo by accident:
///
/// 1. create the batch — one round trip for all of them;
/// 2. submit *every* turn, collecting handles — **nothing is awaited here**;
/// 3. only then drain each handle and seal it.
///
/// Awaiting inside step 2 — the obvious way to write it — serialises the window
/// back into one forward per document, which is exactly the 3.5-seconds-each
/// this replaced. The sequences are dropped after sealing: the turns are in the
/// substrate by then and the gather reaches them across conversations, so
/// holding the sequence would only hold its K/V.
/// **The engine, shared rather than borrowed, and locked per call.**
///
/// The ingest used to hold a `&ConversationEngine` for the whole of startup,
/// and the shared handle was only published once every layer had been ingested
/// and every character woken. So for the twenty minutes a real mind takes to
/// load, every engine-backed route answered 503 — including the guest routes,
/// which is exactly when somebody wants to look at the daemon.
///
/// Taking the lock *per engine call* rather than for the phase is the whole of
/// what makes that safe: a submission needs the lock only long enough to push a
/// job onto a queue, so it is never waiting behind a layer.
type SharedEngine = Arc<Mutex<ConversationEngine>>;

fn ingest_layer(
    engine: &SharedEngine,
    base_config: &SequenceConfig,
    proj: Option<&schema::Projection>,
    source: &ingest::LayerSource,
    turns: &[ingest::Pending],
    // Each document's hash is recorded **as its turn seals**, so a document that never
    // reached the substrate is not claimed to be in it. See `ingest::Pending`.
    ledger: &Ledger,
    p: &LoadProgress,
) -> anyhow::Result<usize> {
    let mut cfg = base_config.clone();
    // A layer document is content, not dialogue: there is no recent-turn tail to
    // carry, and the gather is the only way back to it.
    cfg.context_window_turns = 0;

    let tags = source.tags();
    // **The mind's real projection, not a synthetic one.**
    //
    // A projection runs against the conversation's own schema, and
    // `Builder::for_plain_prompt` declares one layer holding one section — so a
    // document written under it has nothing to gather from and produces no
    // signature. That is how 1,818 documents came to be written and unreachable.
    let synthetic = match proj {
        Some(_) => None,
        None => {
            let prompt = source.prompt();
            // Its frame id comes from its text, so a layer's documents are
            // written under the layer's own prompt rather than under whichever
            // plain prompt sealed a shared id first.
            let frame = engine.lock().unwrap().plain_prompt_section(&prompt)?;
            let b = Builder::for_plain_prompt(&prompt, frame);
            let l = &b.schema().layers[0];
            let (layer, group) = (l.id, l.groups[0].id);
            Some((prompt, b, layer, group))
        }
    };
    let (prompt, builder, layer_id, group_id) = match (proj, &synthetic) {
        (Some(p), _) => (p.prelude.clone(), &p.builder, p.layer, p.group),
        (None, Some((prompt, b, l, g))) => (prompt.clone(), b, *l, *g),
        (None, None) => unreachable!("synthetic is built exactly when proj is None"),
    };

    let mut failed = 0usize;
    let mut done = 0usize;
    let mut next = 0usize;
    let started = Instant::now();

    while next < turns.len() {
        let want = ingest::refill(0, turns.len() - next);
        let slice = &turns[next..next + want];
        next += want;

        // One round trip for the whole window. Creating them one at a time is
        // what starved zend's own batch to 2–4 wide before it was pipelined.
        //
        // The lock is taken for this call and released before the window is
        // submitted — see [`SharedEngine`]. Holding it across the window would
        // put every other caller, guest submissions included, behind a layer.
        let convs = engine
            .lock()
            .unwrap()
            .new_conversations_with_projection_batch(
                slice.len(),
                &prompt,
                builder,
                layer_id,
                group_id,
                &cfg,
            );

        // Submit every turn first, await none. This is the step that batches.
        let mut inflight = Vec::with_capacity(slice.len());
        for (doc, conv) in slice.iter().zip(convs) {
            let (addr, body) = (&doc.addr, &doc.body);
            let mut conv = match conv {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("{addr}: conversation not created — {e:?}");
                    failed += 1;
                    done += 1;
                    continue;
                }
            };
            // Prefilled, not decoded: both halves are written verbatim in one
            // batched forward, because the text is already on disk.
            match conv.submit_prefilled_turn(
                addr,
                body,
                PROJECTION_MARKER,
                SelectionState::new(),
                tags.clone(),
            ) {
                Ok(handle) => inflight.push((conv.timeline_id(), conv, handle, doc)),
                Err(e) => {
                    tracing::warn!("{addr}: not submitted — {e:?}");
                    failed += 1;
                    done += 1;
                }
            }
        }

        // Now drain. Each seal is what puts the turn in the substrate; the
        // sequence is dropped immediately after, freeing its K/V.
        let mut sealed: Vec<_> = Vec::with_capacity(inflight.len());
        for (timeline, mut conv, handle, doc) in inflight {
            let addr = &doc.addr;
            // Where the prefill forward is actually awaited. For a co-batched
            // window the first drain absorbs the slab and the rest return
            // quickly — measured as roughly four documents per slab, which is
            // what an 8 192-token budget over ~1 900-token documents predicts.
            //
            // **The projection events are collected here, not discarded.** They
            // are the provenance signature: what this document's projection
            // selected out of the layers already in the substrate. Dropping them
            // — which is what `_ => {}` did — leaves the document written and
            // unlinked, so a later scan has nothing to pull on.
            let (response, mut events) = drain(&handle, addr);
            match response {
                Some(r) => match conv.finish_turn(handle, &r) {
                    Ok(_) => {
                        // **The tokens this document actually cost the GPU.**
                        //
                        // Documents are the wrong unit for watching an ingest:
                        // they differ by an order of magnitude in length, so a
                        // count of them says nothing about the rate. Tokens per
                        // second is the number that compares directly against
                        // the forward-batched gate's prefill rate on the same
                        // card — which is how "this is taking a while" becomes
                        // "this is running at a tenth of what the card can do".
                        //
                        // Counted at the seal, so it only ever includes work
                        // that reached the substrate.
                        p.add_prefill_tokens(r.stats.prefill_token_count as u64);
                        persist_signatures(&mut conv, &r, &mut events, addr);
                        // **The seal is what earns the ledger entry.** Recorded here, at the
                        // one point the document is provably a turn in the substrate, so a
                        // failure on any path above leaves it un-recorded and therefore
                        // retried on the next boot.
                        ledger.reconcile(&doc.path, Some(&doc.body));
                        sealed.push(timeline);
                    }
                    Err(e) => {
                        tracing::warn!("{addr}: not sealed — {e:?}");
                        failed += 1;
                    }
                },
                None => failed += 1,
            }
            done += 1;
            ingest::announce(p, source, done, turns.len());
        }

        // **The ledger is written at the window boundary, not the layer's.**
        //
        // It used to flush once per layer, on the reasoning that a crash
        // mid-layer "costs one layer rather than the whole mind". That holds for
        // a mind of many small layers and fails completely for a real one: this
        // estate's `world/` layer is 1,267 of its 1,823 documents, so one layer
        // *is* the whole mind, and a daemon stopped anywhere inside it wrote no
        // ledger at all.
        //
        // The cost of that is not a repeated ingest — it is a **duplicated**
        // one. A document with no ledger entry reads as `Added` rather than
        // `Changed`, so nothing tombstones the turn the interrupted run already
        // sealed, and the next boot writes a second copy that the gather can
        // surface alongside the first. Two runs of 641 documents left 84 GB of
        // world history with no way to tell the copies apart.
        //
        // A window is ~4 documents, so this is a few hundred small writes across
        // a half-hour load, and the most an interruption can cost is the window
        // in flight.
        ledger.flush();

        // Give the arena back before the next window.
        //
        // Sealing writes the turn to the substrate; it does **not** reclaim the
        // hot tier, and nothing else was going to. Without this the ingest
        // degraded monotonically — a measured 385 → 602 → 760 → 891 ms per
        // document across four windows — because every document's K/V stayed
        // resident and each successive forward had less arena to work in.
        //
        // Flushing, unlike zend's incremental sweep. zend can demote its older
        // cases while newer ones are still in flight, because its window is a
        // continuous pipeline with a later boundary sweep to catch stragglers.
        // This drains completely between windows, so there is no later sweep and
        // a not-yet-warm timeline would simply not be reclaimed.
        if !sealed.is_empty() {
            if let Err(e) = engine.lock().unwrap().demote_timelines_hot(&sealed, true) {
                tracing::warn!("hot→warm demote failed: {e} — the arena will fill");
            }
        }
    }

    // One line per layer, not per window. The per-window attribution that found
    // all of this — `wait_first` against `wait_rest`, which is the only thing
    // that distinguishes a batching window from a serial one — did its job and
    // came out again; a line per window is noise once the shape is settled.
    // `docs/npc_engine_design.md` records what it measured.
    let secs = started.elapsed().as_secs_f64();
    tracing::info!(
        "layer {} ingested: {} document(s) in {:.1}s ({:.0} doc/min)",
        source.label(),
        turns.len(),
        secs,
        if secs > 0.0 {
            turns.len() as f64 * 60.0 / secs
        } else {
            0.0
        }
    );
    Ok(failed)
}

/// What running one character's life produced.
#[derive(Debug, Default)]
pub struct Lived {
    pub episodes: usize,
    pub beliefs: usize,
    pub relationships: usize,
    pub intents: usize,
    /// Calls an author wrote that could not be executed. Counted and logged,
    /// never silent — a belief that did not form leaves a character missing a
    /// conviction with nothing to say why.
    pub rejected: usize,
}

/// Run one character's life: prefill each episode in date order and execute the
/// calls it contains.
///
/// # Why the episodes are separate conversations
///
/// Same reason the layer documents are: independent sequences co-batch, an
/// appended chain cannot. But there is a second reason here that matters more —
/// each episode carries its **date and title as conversation metadata**, and a
/// conversation holds one title. An episode is a named, dated thing in the
/// character's history, and the substrate should know it as one.
///
/// # Why the calls run after the prose is sealed
///
/// The belief is a consequence of the episode, so the episode has to be in the
/// substrate before the belief that refers to it. Running the calls first would
/// write a conviction whose origin did not yet exist — which is exactly the
/// orphaned-belief shape this whole design exists to end.
#[allow(clippy::too_many_arguments)]
fn run_life(
    engine: &SharedEngine,
    base_config: &SequenceConfig,
    proj: Option<&schema::Projection>,
    who: &str,
    episodes: &[life::Episode],
    ledger: &Ledger,
    p: &LoadProgress,
) -> anyhow::Result<Lived> {
    let mut lived = Lived::default();
    let mut cfg = base_config.clone();
    cfg.context_window_turns = 0;

    for (i, ep) in episodes.iter().enumerate() {
        let Ok(raw) = std::fs::read_to_string(&ep.path) else {
            tracing::warn!("life {who}: {} unreadable", ep.path.display());
            continue;
        };
        // The ledger covers episodes too, so a restart re-lives nothing. `inspect` asks the
        // question; the answer is recorded at the seal below, so an episode that errors on the
        // way there is re-lived next boot instead of being silently lost with every belief,
        // relationship and intention it would have formed.
        if !matches!(
            ledger.inspect(&ep.path, Some(&raw)),
            crate::engine::watcher::Reconcile::Added | crate::engine::watcher::Reconcile::Changed
        ) {
            lived.episodes += 1;
            p.set_progress(i as u64 + 1, episodes.len() as u64);
            continue;
        }

        let parsed = authoring::parse(&raw);
        lived.rejected += parsed.rejected.len();
        for r in &parsed.rejected {
            tracing::warn!("life {who}: {} — {}", ep.title, r.message());
        }

        // The episode itself: prose only, with the call machinery stripped. The
        // character's history is written in its own voice, not in tool syntax.
        // Under the real projection where there is one — an episode written on a
        // synthetic schema has no world to project against and produces no
        // signature, which is the whole point of running the life against the
        // world rather than beside it.
        let mut conv = match proj {
            Some(pr) => engine.lock().unwrap().new_conversation_with_projection(
                &pr.prelude,
                pr.builder.clone(),
                pr.layer,
                pr.group,
                cfg.clone(),
            )?,
            None => engine
                .lock()
                .unwrap()
                .new_conversation(&format!("An episode from the life of {who}."), cfg.clone())?,
        };
        let address = life::address(ep);
        let handle = conv.submit_prefilled_turn(
            &address,
            &parsed.prose,
            PROJECTION_MARKER,
            SelectionState::new(),
            life::tags(ep),
        )?;
        // **This is the world projection.** The episode's projection selects out
        // of the layers already in the substrate — the world first among them,
        // which is why the world is ingested before any life runs — and
        // persisting it writes the link from this moment to the world content it
        // happened against.
        let (response, mut events) = drain(&handle, &address);
        // A silent `continue` here left the episode un-lived *and* un-retryable: the hash was
        // already banked above, so the next boot skipped it. Now the hash is written at the
        // seal, so this path simply leaves it unrecorded — but it still has to say so and
        // still has to advance the bar, which stalled for that character otherwise.
        let Some(r) = response else {
            tracing::warn!(
                "life {who}: {address} produced no response — not lived, and left unrecorded \
                 so the next boot retries it"
            );
            p.set_progress(i as u64 + 1, episodes.len() as u64);
            continue;
        };
        conv.finish_turn(handle, &r)?;
        persist_signatures(&mut conv, &r, &mut events, &address);
        // Date and title as conversation metadata, so the substrate knows when
        // this happened and what it was called — which is what a projection
        // needs to reach an episode by *when* rather than only by what it said.
        let timeline = conv.timeline_id();
        for (key, value) in [
            ("life.date", ep.date.as_str()),
            ("life.title", ep.title.as_str()),
        ] {
            if let Err(e) = engine
                .lock()
                .unwrap()
                .set_conversation_metadata(timeline, key, value)
            {
                tracing::debug!("life {who}: metadata {key} not set — {e:?}");
            }
        }
        // **The episode is in the substrate; record it.** Everything above this line can fail
        // and leave the hash unwritten, which is what makes the next boot retry rather than
        // skip. See `ingest::Pending`.
        ledger.reconcile(&ep.path, Some(&raw));
        lived.episodes += 1;

        // Now the consequences, on their own records, after the episode they
        // came from is durable.
        for call in &parsed.calls {
            match write_consequence(engine, &cfg, proj, who, ep, call) {
                Ok(()) => match call.tool {
                    "form_belief" => lived.beliefs += 1,
                    "form_relationship" | "revise_relationship" => lived.relationships += 1,
                    "leave_intent" => lived.intents += 1,
                    _ => {}
                },
                Err(e) => {
                    tracing::warn!(
                        "life {who}: {} — {} not written: {e:#}",
                        ep.title,
                        call.tool
                    );
                    lived.rejected += 1;
                }
            }
        }

        // Give the arena back — the same reason the layer ingest does.
        if let Err(e) = engine
            .lock()
            .unwrap()
            .demote_timelines_hot(&[timeline], true)
        {
            tracing::debug!("life {who}: demote failed — {e}");
        }
        p.set_progress(i as u64 + 1, episodes.len() as u64);
    }
    Ok(lived)
}

/// Write one authoring call as a tagged substrate record.
///
/// **A belief is a record in the substrate, not a file on disk.** That is the
/// design's central move: it makes a belief retrievable by the same tag-filtered
/// gather as any other layer content, editable from the console, and — because
/// it names the episode that produced it — arguable. A belief you can trace to a
/// day is one you can disagree with; a belief simply asserted is not.
fn write_consequence(
    engine: &SharedEngine,
    cfg: &SequenceConfig,
    proj: Option<&schema::Projection>,
    who: &str,
    ep: &life::Episode,
    call: &authoring::Call,
) -> anyhow::Result<()> {
    let layer = authoring::by_name(call.tool)
        .map(|t| t.writes)
        .unwrap_or("beliefs");

    // The record reads as the character's own, in the register the rest of the
    // layers use, and carries its origin. The origin is the point: it is what
    // turns a conviction into something with a history.
    let body = render_consequence(call);
    let address = format!(
        "{layer}/{who}/{} {}",
        ep.date,
        call.args
            .get("statement")
            .or_else(|| call.args.get("entity_id"))
            .or_else(|| call.args.get("intent"))
            .and_then(|v| v.as_str())
            .unwrap_or(call.tool)
    );

    let mut conv = match proj {
        Some(pr) => engine.lock().unwrap().new_conversation_with_projection(
            &pr.prelude,
            pr.builder.clone(),
            pr.layer,
            pr.group,
            cfg.clone(),
        )?,
        None => engine.lock().unwrap().new_conversation(
            &format!("What {who}'s life left them holding."),
            cfg.clone(),
        )?,
    };
    let handle = conv.submit_prefilled_turn(
        &address,
        &body,
        PROJECTION_MARKER,
        SelectionState::new(),
        vec![
            layer.to_string(),
            format!("{layer}:{who}"),
            // The episode it came from, so the origin is reachable from the
            // belief and not only the other way round.
            format!("from:{}", ep.date),
        ],
    )?;
    let (response, mut events) = drain(&handle, &address);
    let r = response.ok_or_else(|| anyhow::anyhow!("no response"))?;
    let timeline = conv.timeline_id();
    conv.finish_turn(handle, &r)?;
    // A belief carries a signature too: what the character already held when it
    // formed. That is what makes a later contradiction reachable from the belief
    // it contradicts.
    persist_signatures(&mut conv, &r, &mut events, &address);
    {
        let e = engine.lock().unwrap();
        let _ = e.set_conversation_metadata(timeline, "from.date", &ep.date);
        let _ = e.set_conversation_metadata(timeline, "from.title", &ep.title);
        let _ = e.demote_timelines_hot(&[timeline], true);
    }
    Ok(())
}

/// A call as the character would hold it.
fn render_consequence(call: &authoring::Call) -> String {
    let s = |k: &str| call.args.get(k).and_then(|v| v.as_str()).unwrap_or("");
    match call.tool {
        "form_belief" => s("statement").to_string(),
        "form_relationship" | "revise_relationship" => {
            let who = if s("display").is_empty() {
                s("entity_id")
            } else {
                s("display")
            };
            let notes = s("notes");
            if notes.is_empty() {
                who.to_string()
            } else {
                format!("{who}. {notes}")
            }
        }
        "leave_intent" => {
            let until = s("until");
            if until.is_empty() {
                s("intent").to_string()
            } else {
                format!("{}. Until: {until}", s("intent"))
            }
        }
        _ => String::new(),
    }
}

/// The driver: the thread that keeps every character thinking.
///
/// One thread for the whole cast, not one per character. A thread per character
/// would be a thousand stacks to hold a thousand idle minds, and the scheduler's
/// whole design is that idle costs nothing — a per-character thread would put
/// the cost back in a different currency.
pub fn drive(rt: Arc<Runtime>) {
    std::thread::Builder::new()
        .name("npcd-tick".into())
        .spawn(move || {
            let start = Instant::now();
            tracing::info!("tick driver running");
            while !rt.stopping() {
                let now_ms = start.elapsed().as_millis() as u64;

                for id in rt.scheduler.due_now(now_ms) {
                    // Per character, not once per pass: two characters in
                    // different worlds are at different instants, and one of
                    // those worlds may be paused.
                    let world_ms = rt.world_ms(id);
                    // The day boundary is checked before the tick, so a
                    // character that has crossed midnight wakes into its new
                    // conversation rather than acting once more in yesterday's.
                    if let Some((from, to)) = rt.scheduler.roll_day(id, world_ms) {
                        tracing::info!("npc {id}: day {from} → {to}, conversation rolled over");
                        rt.scheduler.deliver(
                            id,
                            world_ms,
                            crate::engine::event::Salience::NORMAL,
                            crate::engine::event::EventKind::Wake { day: to },
                        );
                    }
                    // What this character is set on, restated every turn.
                    // Delivered rather than synthesised inside the tick: the
                    // scheduler knows a character has an empty inbox and
                    // nothing at all about what it is for.
                    //
                    // **Only when there is nothing else to answer.** This gate
                    // has now been wrong in both directions, and the two
                    // mistakes are instructive.
                    //
                    // It was first gated on an *empty inbox*, which never
                    // happened: a character with a body is handed the situation
                    // it is standing in every moment, so the depth is never
                    // zero, so the standing task written for exactly the case of
                    // having company was the one that never arrived. Two Makers
                    // met, had nothing telling them to stay, and walked out of
                    // the room in opposite directions.
                    //
                    // Removing the gate fixed that and introduced the opposite
                    // fault. The task supersedes in its own band, so restating
                    // it never accumulates — but it does keep moving to the most
                    // recent position in the window, which is where attention
                    // weights hardest. A character mid-conversation was being
                    // told, more recently than anything its companion had
                    // actually said, that nothing had been asked of it.
                    //
                    // The question was never "is the inbox empty", it is "is
                    // anything *happening*" — and the situation is a fact about
                    // the room rather than an event. `has_news` was that
                    // question asked about this instant, which is the third way
                    // of getting it wrong: a conversation is mostly the gaps
                    // between its utterances, and in every one of those gaps
                    // there is momentarily no news queued.
                    //
                    // So the task landed *inside* conversations, and because it
                    // supersedes in its own band it sat in the most recent
                    // position in the window every time. Two characters
                    // alternated for a hundred turns — heard the other speak,
                    // were told nothing had been asked of them, heard the other
                    // speak — each being told, more recently than anything its
                    // companion had said, that nothing was going on.
                    //
                    // A stretch of quiet is what the task was always described
                    // as waiting for. See [`IDLE_AFTER_MS`].
                    // Two clocks, both `IDLE_AFTER_MS`: quiet since anything
                    // happened, and quiet since the task itself was last
                    // restated. Without the second the gate latches open — the
                    // task is not news, so nothing it does moves the first
                    // clock — and a character nobody is talking to is handed it
                    // again on every tick.
                    if rt.scheduler.nudge_due(id, world_ms, IDLE_AFTER_MS) {
                        if let Some(text) = rt.nudge_for(id) {
                            rt.scheduler.deliver(
                                id,
                                world_ms,
                                crate::engine::event::Salience::IDLE,
                                crate::engine::event::EventKind::Nudge { text },
                            );
                        }
                    }

                    let minds = rt.minds.read().unwrap().clone();
                    let persona = rt.persona_of(id);
                    let day = crate::engine::sleep::day_of(world_ms);
                    // What the grammar is built from this turn: the acts that
                    // are reachable from where this character stands, and the
                    // names it may address. Read here, once, before the decode —
                    // the world moves under a decode that takes seconds, and a
                    // grammar built halfway through it would be masked to a room
                    // that no longer matches the situation the character read.
                    let within = rt.within(id);

                    rt.scheduler.tick(id, now_ms, world_ms, |events, window| {
                        let (Some(minds), Some(p)) = (minds.as_ref(), persona.as_ref()) else {
                            // No engine yet, or a character the authored state
                            // no longer knows. Perception still lands in the
                            // window — that half needs nothing — and no acts is
                            // the honest answer rather than an invented one.
                            return Vec::new();
                        };
                        match minds.think(id, &p.as_persona(), p.mode, day, events, window, &within)
                        {
                            Ok(t) => {
                                // Reported, never swallowed: a character failing
                                // to act and one choosing not to look identical
                                // from outside and need completely different
                                // fixes.
                                //
                                // And told to the character, not only to the
                                // log. A rejection it cannot see is a character
                                // acting into silence — it has no reason to do
                                // anything differently, so it makes the same
                                // malformed call every turn for as long as it
                                // runs.
                                // Two lists, because the feed and the character
                                // read different sentences — see `Recorded`.
                                // A rejection is the one case where they agree:
                                // the world's words are all there is.
                                let mut done: Vec<String> = Vec::new();
                                let mut answers: Vec<String> = Vec::new();
                                for r in &t.parsed.rejected {
                                    tracing::warn!("npc {id}: act rejected — {r:?}");
                                    done.push(r.line());
                                    answers.push(r.line());
                                }
                                if t.parsed.is_empty() && t.parsed.rejected.is_empty() {
                                    // Chose to do nothing, and said so cleanly.
                                    // Distinct from a decode that failed, which
                                    // logged above.
                                    tracing::debug!("npc {id}: no act this tick");
                                }
                                if !t.parsed.narration.is_empty() {
                                    tracing::debug!(
                                        "npc {id}: narration (not an act) — {}",
                                        t.parsed.narration
                                    );
                                }
                                // Acts that belong to a body go to the world
                                // they stand in, and the world's verdict — not
                                // the character's intent — decides how each one
                                // is recorded. A refusal is an ordinary
                                // outcome, perceived like any other, so the
                                // character learns it went wrong rather than
                                // believing it worked. `record_act` holds the
                                // rule.
                                // **A reflect the world took is answered by a
                                // reflection** — at most one per turn. Its slot
                                // keeps `body::NO_REFLECTION` until the
                                // reflection answers.
                                let mut reflecting: Option<(Owed, [String; 3])> = None;
                                for a in &t.parsed.acts {
                                    let r = rt.record_act(id, a);
                                    done.push(r.feed);
                                    if a.tool == "reflect"
                                        && r.landed
                                        && reflecting.is_none()
                                        && rt.can_reflect()
                                    {
                                        let arg = |k: &str| {
                                            a.args
                                                .get(k)
                                                .and_then(|v| v.as_str())
                                                .unwrap_or_default()
                                                .to_string()
                                        };
                                        // Recorded as the act alone — `reflect`
                                        // and its thought — and completed with
                                        // what came back once it has.
                                        let row = a.summary();
                                        if let Some(last) = done.last_mut() {
                                            *last = row.clone();
                                        }
                                        reflecting = Some((
                                            Owed {
                                                answers: Vec::new(),
                                                slot: answers.len(),
                                                row,
                                            },
                                            [
                                                arg("situation"),
                                                arg("inner_thoughts"),
                                                arg("feeling"),
                                            ],
                                        ));
                                    }
                                    answers.push(r.answer);
                                    // A pause that landed stops the character
                                    // here. Armed after the act rather than
                                    // inside it because going quiet is
                                    // scheduling and being seen to stop is the
                                    // world's — two halves of one act, and only
                                    // this half knows the clock.
                                    rt.arm_pause(id, a, now_ms);
                                    // And an act that answered brings it
                                    // straight back, so it has a turn in which
                                    // to use what it was told.
                                    rt.arm_followup(id, a);
                                }
                                // **And the character is told what came of it.**
                                //
                                // One answer per call it made, riding at the
                                // head of its next turn as `<tool_response>` —
                                // the half of the protocol that was missing.
                                // Without it a character acts and reads the
                                // weather back, which is how every one of them
                                // came to do nothing but reflect.
                                //
                                // The world's verdict, not the feed line: what
                                // a person watching wants to read and what the
                                // character needs to know are different
                                // sentences. See `Recorded`.
                                //
                                // A turn with a reflection in it hands its answers
                                // to the reflection instead, which delivers them
                                // once its first question is answered.
                                match reflecting {
                                    Some((owed, [situation, inner, feeling])) => rt
                                        .begin_reflection(
                                            id,
                                            situation,
                                            inner,
                                            feeling,
                                            Owed { answers, ..owed },
                                        ),
                                    None => minds.deliver_outcomes(id, answers),
                                }
                                done
                            }
                            Err(e) => {
                                tracing::warn!("npc {id}: decode failed — {e:#}");
                                Vec::new()
                            }
                        }
                    });
                }
                std::thread::sleep(DRIVE_INTERVAL);
            }
            tracing::info!("tick driver stopped");
        })
        .expect("spawn the tick driver");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rt() -> Arc<Runtime> {
        // A temp path: the tests exercise scheduling, not persistence, and an
        // in-repo one would leave a ledger file behind.
        Runtime::new(Mind::new(None), &std::env::temp_dir())
    }

    /// **Everything this daemon writes goes under `--data`.**
    ///
    /// The conversation layer's `workspace_path` falls back to the *process
    /// working directory*, so a runtime that does not carry `--data` cannot pass
    /// it on and the redo log lands wherever the daemon was launched from. That
    /// happened: the character store sat correctly in `npcd/.substrate` at 65 MB
    /// while the real log grew to **190 GB** in the repo root, invisible to
    /// `/v1/substrate/storage`, until it filled the disk and surfaced as a
    /// linker error.
    #[test]
    fn the_runtime_knows_where_the_daemon_writes() {
        let data = std::env::temp_dir().join("npcd-data-probe");
        let rt = Runtime::new(Mind::new(None), &data);
        assert_eq!(
            rt.data, data,
            "the runtime cannot tell the engine where to write"
        );
        // And it is not the working directory, which is the fallback that caused
        // the whole problem.
        assert_ne!(
            rt.data,
            std::env::current_dir().unwrap_or_default(),
            "the data directory resolved to the process working directory"
        );
    }

    #[test]
    fn a_fresh_runtime_is_not_ready_and_has_no_minds() {
        let rt = rt();
        assert!(!rt.is_ready());
        assert!(rt.minds.read().unwrap().is_none());
        assert_eq!(rt.scheduler.population(), 0);
    }

    /// Stopping must be observable before the thread notices it, or a shutdown
    /// races the driver's next pass.
    #[test]
    fn stopping_is_visible_immediately() {
        let rt = rt();
        assert!(!rt.stopping());
        rt.stop();
        assert!(rt.stopping());
    }

    /// The clock is asked per character. Two characters in differently-paced
    /// worlds must not be handed the same instant — and a resolver called once
    /// per pass instead of once per character is exactly how that happens.
    #[test]
    fn the_world_clock_is_resolved_per_character() {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
        // Before a clock is installed there is no time to report, and zero is
        // the only honest answer. The driver does not run in this window.
        assert_eq!(rt.world_ms(3), 0);

        rt.set_clock(Arc::new(|npc_id| npc_id * 1_000));
        assert_eq!(rt.world_ms(3), 3_000);
        assert_eq!(rt.world_ms(7), 7_000);
    }

    /// The driver quantises the scheduler's clock; it is not the tick rate. If
    /// this ever grows past the alert heartbeat, a preempted character would
    /// wait longer for the driver than for its own metabolism.
    #[test]
    fn the_drive_interval_is_finer_than_the_fastest_heartbeat() {
        assert!(DRIVE_INTERVAL < crate::engine::tick::ALERT_HEARTBEAT);
    }

    // ── the world, hosted ───────────────────────────────────────────────────

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");

    /// The id the vault is authored under. A world is hosted under the same id
    /// its document has, because that is what a character's `world_id` names.
    const WORLD: &str = "creators-vault";

    /// A runtime hosting the vault, with its metronome held still.
    ///
    /// Paused on purpose: these assert what hosting, binding and acting *do*,
    /// and a world moving underneath them would make every one of them a race.
    /// The metronome's own behaviour is [`crate::engine::driver`]'s to prove.
    fn vaulted() -> Arc<Runtime> {
        let rt = rt();
        rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
        rt.hold_world(WORLD, true);
        rt
    }

    fn at(node: &str) -> npc_map::world::Where {
        npc_map::world::Where::new("vault-casting", node)
    }

    /// Put a body in the world and give a character to it.
    fn embody(rt: &Arc<Runtime>, npc_id: u64, body: &str, room: &str) {
        let w = rt.hosted.get(WORLD).expect("hosted");
        w.with(|w| {
            w.enter(body, format!("Maker-{npc_id:02}"), at(room))
                .unwrap()
        });
        rt.scheduler.wake(npc_id, 0, 0);
        rt.embody(npc_id, WORLD, body, 0).expect("bound");
    }

    fn window(rt: &Arc<Runtime>, npc_id: u64) -> Vec<String> {
        rt.scheduler
            .window_of(npc_id, |w| {
                w.turns().map(|t| t.text.clone()).collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Let every due character take its turn, with no decode behind it.
    fn run(rt: &Arc<Runtime>, at_ms: u64) {
        for id in rt.scheduler.due_now(at_ms) {
            rt.scheduler.tick(id, at_ms, at_ms, |_, _| Vec::new());
        }
    }

    /// **A message sent to a character has to reach its mind.**
    ///
    /// The whole path, end to end, because every piece of it was individually
    /// correct while the thing a person actually does — message a character and
    /// wait for an answer — did nothing at all. Sending writes a thread; the
    /// world's next moment is what carries it to the inbox; and until it is in
    /// the inbox the character has not been told anything.
    #[test]
    fn a_message_reaches_the_characters_inbox_on_the_next_moment() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let hosted = rt.hosted.get(WORLD).expect("hosted");

        let sent = rt
            .message_npc(1, "Johnathan Sharratt", "are you there")
            .expect("the character could be messaged");
        assert_eq!(sent.with, "Maker-01");
        assert_eq!(sent.waiting_for_them, 1, "it was not written to the thread");

        // Drain what embodying already queued, so what is asserted below can
        // only have come from the message.
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());
        assert_eq!(rt.scheduler.inbox_depth(1), Some(0), "not drained");

        let moment = environment::advance(&hosted, &rt.bodies, &rt.scheduler);
        assert_eq!(moment.messaged, 1, "the moment carried nothing to anybody");

        assert_eq!(
            rt.scheduler.inbox_depth(1),
            Some(1),
            "the message never reached the inbox"
        );
    }

    /// And it is handed over **once**. The sweep runs twice a second; a cursor
    /// that did not move would read the same message to the character for ever,
    /// which is the failure the phone's own cursor exists to prevent.
    #[test]
    fn a_message_is_handed_to_the_mind_exactly_once() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let hosted = rt.hosted.get(WORLD).expect("hosted");
        rt.message_npc(1, "Johnathan Sharratt", "are you there");
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());

        assert_eq!(
            environment::advance(&hosted, &rt.bodies, &rt.scheduler).messaged,
            1
        );
        assert_eq!(
            environment::advance(&hosted, &rt.bodies, &rt.scheduler).messaged,
            0,
            "the same message was handed over twice"
        );
    }

    #[test]
    fn a_hosted_world_is_reachable_and_moving() {
        let rt = rt();
        assert!(rt.hosted.get(WORLD).is_none(), "hosted before it was asked");

        rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
        assert!(rt.hosted.get(WORLD).is_some());
        let running = rt.moments();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].0, WORLD);
        assert!(!running[0].2, "hosted paused");
    }

    #[test]
    fn a_map_that_does_not_load_leaves_the_daemon_running() {
        // An authoring mistake must not take the console and the whole
        // authored corpus down with it.
        let rt = rt();
        assert!(rt
            .host(WORLD, Path::new(env!("CARGO_MANIFEST_DIR")))
            .is_err());
        assert!(rt.hosted.get(WORLD).is_none());
        assert!(
            rt.moments().is_empty(),
            "a metronome outlived a failed load"
        );
    }

    #[test]
    fn a_world_can_be_held_still_and_let_go() {
        let rt = vaulted();
        assert!(rt.moments()[0].2, "not paused");
        assert!(rt.hold_world(WORLD, false));
        assert!(!rt.moments()[0].2, "not resumed");
        assert!(
            !rt.hold_world("nowhere", true),
            "held a world it has not got"
        );
    }

    #[test]
    fn embodying_a_character_grounds_it_and_quickens_it() {
        // Both, at once. A character that can act before it has been told where
        // it is would act blind; one left at an ambient pace would think about
        // its work every two minutes.
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");

        assert_eq!(rt.scheduler.pace_of(1), Some(Pace::WORKING));
        run(&rt, 1);
        let read = window(&rt, 1);
        assert!(read.iter().any(|t| t.contains("band one")), "{read:?}");
    }

    #[test]
    fn what_is_within_reach_arrives_with_where_the_body_is() {
        // The two are one fact — both are functions of where it stands — so
        // they arrive together and go stale together.
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        run(&rt, 1);

        let here = window(&rt, 1)
            .into_iter()
            .find(|t| t.starts_with("You are"))
            .expect("grounded");
        assert!(here.contains("Within reach"), "{here}");
        assert!(here.contains("terminal"), "{here}");
    }

    #[test]
    fn a_corridor_says_nothing_about_what_is_within_reach() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "ring-north");
        run(&rt, 1);

        let here = window(&rt, 1)
            .into_iter()
            .find(|t| t.starts_with("You are"))
            .expect("grounded");
        assert!(!here.contains("Within reach"), "{here}");
    }

    #[test]
    fn embodying_the_same_character_twice_elsewhere_is_refused() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        rt.hosted
            .get(WORLD)
            .unwrap()
            .with(|w| w.enter("m2", "Maker-02", at("band-one")).unwrap());
        assert!(rt.embody(1, WORLD, "m2", 0).is_err());
        assert!(rt.embody(2, WORLD, "m1", 0).is_err(), "two minds, one body");
    }

    /// **A character is told about the building it is in, and a reflection
    /// opens on the room it is in.** Both failed together: every character in
    /// the world was handed the whole world as "the building you work in" —
    /// six vault levels and not one room of the Redoubt — and a reflection
    /// opened on the persona's situation, which is empty for every authored
    /// character. Asked where it was, a character in the Redoubt named a vault
    /// room; asked to reflect, it dreamed of an office.
    #[test]
    fn a_character_knows_the_building_it_stands_in_and_the_room_it_is_in() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with(|w| {
            w.enter("m2", "Maker-02", Where::new("tower-redoubt", "muster-hall"))
                .unwrap()
        });
        rt.scheduler.wake(2, 0, 0);
        rt.embody(2, WORLD, "m2", 0).expect("bound");

        let (vault_key, vault) = rt.building_of(&hosted, "m1").expect("in the vault");
        let (redoubt_key, redoubt) = rt.building_of(&hosted, "m2").expect("in the tower");
        assert_eq!(vault_key, identity::building_key(WORLD, "creators-vault"));
        assert_eq!(redoubt_key, identity::building_key(WORLD, "tower-redoubt"));
        assert!(redoubt.to_lowercase().contains("muster hall"), "{redoubt}");
        assert!(!vault.to_lowercase().contains("muster hall"), "{vault}");

        let here = rt.situation_of(2).expect("a body is somewhere");
        assert!(here.contains("muster hall"), "{here}");
        assert!(rt.situation_of(99).is_none(), "nobody is nowhere");
    }

    #[test]
    fn embodying_into_a_world_or_a_body_that_is_not_there_is_refused() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let no_world = rt.embody(1, "elsewhere", "m1", 0).unwrap_err().to_string();
        assert!(no_world.contains("elsewhere"), "{no_world}");

        let no_body = rt.embody(1, WORLD, "nobody", 0).unwrap_err().to_string();
        assert!(no_body.contains("nobody"), "{no_body}");
        assert!(!rt.bodies.is_bound(1));
    }

    #[test]
    fn disembodying_lets_a_character_settle_back_to_reacting() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        assert!(rt.disembody(1, 0));
        assert_eq!(rt.scheduler.pace_of(1), Some(Pace::AMBIENT));
        assert!(rt.body_of(1).is_none());
        assert!(!rt.disembody(1, 0), "disembodied twice");
    }

    #[test]
    fn unhosting_a_world_stops_it_and_frees_the_characters_in_it() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        embody(&rt, 2, "m2", "band-one");

        assert!(rt.unhost(WORLD));
        assert!(rt.moments().is_empty(), "the metronome kept beating");
        assert!(rt.hosted.get(WORLD).is_none());
        assert!(
            !rt.bodies.is_bound(1),
            "a character kept a body that is gone"
        );
        assert!(!rt.bodies.is_bound(2));
        assert!(!rt.unhost(WORLD), "unhosted twice");
    }

    // ── acts, landing on the world ──────────────────────────────────────────

    fn act(tool: &'static str, args: serde_json::Value) -> Act {
        Act {
            tool,
            args: args.as_object().expect("an object").clone(),
        }
    }

    #[test]
    fn an_act_that_happens_inside_a_head_never_reaches_the_world() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        assert_eq!(
            rt.act_on_world(1, &act("note_concern", json!({}))),
            Outcome::NotOfTheBody
        );
    }

    #[test]
    fn a_character_with_no_body_cannot_act_on_a_world() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        assert_eq!(
            rt.act_on_world(1, &act("speak", json!({"intent": "anything"}))),
            Outcome::NotOfTheBody
        );
    }

    #[test]
    fn speaking_reaches_the_other_character_in_the_room() {
        // The whole chain, in one daemon: an act from one mind lands in the
        // world, is perceived by the body beside it, and is read by that
        // body's mind.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let said = rt
            .act_on_world(
                1,
                &act(
                    "speak",
                    json!({"intent": "the redoubt burned twice", "to": "Maker-02"}),
                ),
            )
            .line()
            .expect("a body act")
            .to_string();
        assert!(said.contains("You tell Maker-02"), "{said}");

        // Not yet: perception happens on the world's clock, not the speaker's.
        run(&rt, 2);
        assert!(!window(&rt, 2).join("\n").contains("redoubt"));

        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler);
        run(&rt, 3);

        let heard = window(&rt, 2).join("\n");
        assert!(heard.contains("says to you"), "{heard}");
        assert!(heard.contains("redoubt"), "{heard}");
    }

    /// **Being spoken to ends a pause.**
    ///
    /// The deadlock the typed wait needed three mechanisms to prevent cannot
    /// form here: Wyneth stops, and Perrin speaking brings her straight back
    /// through the ordinary sweep. No condition to satisfy, no patience, no
    /// rule excluding Perrin from waiting back — and nothing announcing to the
    /// room that she stopped, which it no longer is.
    #[test]
    fn a_pause_is_ended_by_the_room() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let w = act("reflect", json!({}));
        assert!(
            rt.record_act(1, &w).feed.starts_with("reflect"),
            "the act stood"
        );
        rt.arm_pause(1, &w, 0);
        assert!(
            !rt.scheduler.due_now(1).contains(&1),
            "she did not actually stop"
        );

        // Perrin is told nothing about it.
        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler);
        run(&rt, 2);
        let seen = window(&rt, 2).join("\n");
        assert!(!seen.contains("lets the moment pass"), "{seen}");

        // And Perrin speaking brings her back at once, through nothing but the
        // ordinary delivery path.
        assert!(rt
            .act_on_world(2, &act("shout", json!({"intent": "that I am here"})))
            .happened());
        environment::advance(&world, &rt.bodies, &rt.scheduler);
        assert!(rt.scheduler.due_now(1).contains(&1), "she was not woken");
    }

    /// A stall nobody interrupts ends by itself.
    ///
    /// The failure this replaces: a wait nothing could answer left the
    /// character asleep forever while every view of it read the same word a
    /// merely quiet character reads, so the two were indistinguishable from
    /// outside. A stall cannot do that — its deadline is the whole of it.
    ///
    /// Driven by `move_to`, which is one of the two acts that occupy a body.
    /// It was `reflect`, from when reflection was the only thing that stalled;
    /// a thought does not take time, so it no longer does.
    #[test]
    fn a_pause_nobody_interrupts_ends_on_its_own() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        run(&rt, 1);

        rt.arm_pause(1, &act("move_to", json!({"destination": "band one"})), 0);
        assert!(
            !rt.scheduler.due_now(1_000).contains(&1),
            "it came back a second later"
        );
        // Nobody says anything, ever, and it comes back anyway.
        assert!(
            rt.scheduler.due_now(10_000_000).contains(&1),
            "it stopped for good"
        );
    }

    #[test]
    fn an_act_that_landed_is_recorded_as_an_act_whatever_the_world_said_back() {
        // Speech is the case that made this visible. The world answers a
        // successful `shout` in narration — "You shout, for anyone within
        // earshot." — and
        // recording *that* put one prose sentence in a column of single-word
        // acts, both in the feed and in the character's own window.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let said = rt.record_act(
            1,
            &act("shout", json!({"intent": "the redoubt burned twice"})),
        );
        // The feed carries both halves: what was asked, then what came of it.
        assert_eq!(
            said.feed,
            "shout — the redoubt burned twice → You shout, for anyone within earshot."
        );
        // The character, meanwhile, is told only what the world did with it —
        // it does not need to be told the name of the act it just chose.
        assert_eq!(said.answer, "You shout, for anyone within earshot.");

        // A body act and a head act now read the same way as each other, which
        // is the point — one shape, whether or not a world was involved.
        let noted = rt.record_act(1, &act("note_concern", json!({"about": "the ledger"})));
        assert!(noted.feed.starts_with("note_concern"), "{}", noted.feed);

        // And the world did take the speech: the shape of the record is not
        // the act being quietly dropped.
        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler);
        run(&rt, 10_000);
        let heard = window(&rt, 2).join("\n");
        assert!(heard.contains("redoubt"), "{heard}");
    }

    /// A refusal keeps the world's words **and** names the act it refused.
    ///
    /// The prose is what tells the character where it could actually have gone,
    /// so losing it to a uniform shape would be the reverse of the bug above.
    /// But the words alone read as a sentence dropped into a column of acts —
    /// "Wailen Wylde is already on it." beside "invite — …", with nothing
    /// saying they were the same kind of event or that one had failed.
    #[test]
    fn a_refusal_is_recorded_in_the_world_s_words_because_the_prose_is_the_point() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");

        let r = rt.record_act(
            1,
            &act("move_to", json!({"destination": "the observatory"})),
        );
        let line = r.feed;
        assert!(
            line.contains("nowhere called \"the observatory\""),
            "{line}"
        );
        assert!(line.contains("band one"), "it named nowhere real: {line}");
        // Scannable like every other line: the act's name first.
        assert!(line.starts_with("move_to"), "{line}");
        // And unmistakably not a completed one.
        assert!(
            line.contains('✗'),
            "a refusal read as a completed act: {line}"
        );

        // The character is told the world's words and nothing else — the mark
        // and the act's name are for somebody watching the feed, and a
        // character does not need to be told the name of the act it just chose.
        assert!(!r.answer.contains('✗'), "{}", r.answer);
        assert!(r.answer.contains("nowhere called"), "{}", r.answer);
    }

    /// **The character is told what happened, not what it asked for.**
    ///
    /// The feed renders an act as an act — `gesture — …` — which is right for
    /// a column somebody scans. Sent to the model as a `<tool_response>` it is
    /// the character's own arguments handed back as though they were an
    /// outcome, which is what it read live: "gesture — for Wailen to see that
    /// the room is ours for now", every turn, never once told whether anybody
    /// saw it.
    #[test]
    fn the_answer_is_the_worlds_verdict_and_the_feed_line_is_the_act() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let r = rt.record_act(1, &act("gesture", json!({"intent": "at the door"})));
        // The feed names the act, what it asked for, and what came of it.
        assert!(r.feed.starts_with("gesture — at the door → "), "{}", r.feed);
        assert!(r.feed.contains("You show"), "{}", r.feed);
        // The character gets the verdict alone — not its own arguments back.
        assert!(
            r.answer.starts_with("You show"),
            "the character was handed its own arguments: {}",
            r.answer
        );
        assert!(!r.answer.contains("gesture —"), "{}", r.answer);
    }

    /// **A reflect's row is its thought, then what came back.** Its situation
    /// and feeling are the reflection's input and are not listed; and what a
    /// pause says back is only that it happened — never the character's own
    /// words. When a reflection runs, its answer takes the arrow's side instead.
    #[test]
    fn a_reflect_is_recorded_as_its_thought_and_what_came_of_it() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let r = rt.record_act(
            1,
            &act(
                "reflect",
                json!({
                    "situation": "alone in the green room",
                    "inner_thoughts": "the box has given up a fold",
                    "feeling": "weary",
                }),
            ),
        );
        assert_eq!(
            r.feed,
            format!(
                "reflect — the box has given up a fold → {}",
                body::NO_REFLECTION
            )
        );
        assert_eq!(r.answer, body::NO_REFLECTION);
    }

    /// **One dream at a time, per character, and the slot always comes
    /// back.** What a reflect finds taken is the dream, never the reflection:
    /// it still answers, and only skips the dream.
    #[test]
    fn a_character_writes_one_dream_at_a_time() {
        let rt = rt();
        let first = rt.claim_dream(1).expect("a free slot");
        assert!(rt.claim_dream(1).is_none(), "two dreams at once");
        assert!(rt.claim_dream(2).is_some(), "one slot per character");
        drop(first);
        assert!(rt.claim_dream(1).is_some(), "the slot never came back");
    }

    #[test]
    fn an_act_the_world_refuses_reads_as_refused_rather_than_as_done() {
        // The distinction the whole path exists to preserve: a character that
        // cannot tell a refused act from a successful one spends the rest of
        // the day reasoning from a move it never made.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");

        let out = rt.act_on_world(
            1,
            &act("move_to", json!({"destination": "the observatory"})),
        );
        let Outcome::Refused(out) = out else {
            panic!("a walk to nowhere is a refusal, not {out:?}");
        };
        assert!(out.contains("nowhere called \"the observatory\""), "{out}");
        assert!(out.contains("band one"), "it named nowhere real: {out}");
        assert!(rt
            .hosted
            .get(WORLD)
            .unwrap()
            .read(|w| w.actor("m1").unwrap().walk.is_none()));
    }

    #[test]
    fn moving_is_a_journey_the_world_advances_rather_than_the_act() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let out = rt
            .act_on_world(1, &act("move_to", json!({"destination": "band one"})))
            .line()
            .expect("a body act")
            .to_string();
        assert!(out.contains("one stop"), "{out}");

        let world = rt.hosted.get(WORLD).unwrap();
        // Still where it was: setting off is not arriving.
        assert_eq!(
            world.read(|w| w.actor("m1").unwrap().at.clone()),
            at("green-room")
        );
        environment::advance(&world, &rt.bodies, &rt.scheduler);
        assert_eq!(
            world.read(|w| w.actor("m1").unwrap().at.clone()),
            at("band-one")
        );

        run(&rt, 10_000);
        let read = window(&rt, 1).join("\n");
        assert!(read.contains("You got to band one"), "{read}");
    }

    /// **Arriving on a floor tells you who else is on it, and where** — the
    /// person left behind in the room it walked out of, here.
    #[test]
    fn arriving_names_who_else_is_on_the_floor() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        let _ = rt.act_on_world(1, &act("move_to", json!({"destination": "band one"})));
        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler);

        run(&rt, 10_000);
        let read = window(&rt, 1).join("\n");
        assert!(read.contains("Elsewhere on this floor:"), "{read}");
        assert!(read.contains("in the green room"), "{read}");
    }

    #[test]
    fn what_a_character_did_comes_back_in_its_own_turn() {
        // The actor is not left waiting for the world to tell it what it just
        // did — the act goes into its own window in the same turn. What it does
        // wait for is *everyone else* perceiving it.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let spoke = act("shout", json!({"intent": "first"}));
        assert!(
            rt.act_on_world(2, &spoke).happened(),
            "the world took the speech"
        );
        // And what gets recorded for it is the act, in the shape every act has.
        assert_eq!(spoke.summary(), "shout — first");

        run(&rt, 2);
        assert!(
            !window(&rt, 1).join("\n").contains("first"),
            "it arrived outside a world moment"
        );
    }

    /// Every path into a character's window goes through a world moment. An
    /// act that pushed perception itself would be a second one, unbatched and
    /// off the sweep — and it would be invisible, because it would work.
    #[test]
    fn nothing_perceives_anything_outside_a_world_moment() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let world = rt.hosted.get(WORLD).unwrap();
        for intent in ["one", "two", "three"] {
            rt.act_on_world(1, &act("speak", json!({ "intent": intent })));
        }
        run(&rt, 2);
        let before = window(&rt, 2).len();

        environment::advance(&world, &rt.bodies, &rt.scheduler);
        // Past its heartbeat: speech to the room is worth a turn but does not
        // interrupt one, so it waits rather than preempting.
        run(&rt, 10_000);
        let after = window(&rt, 2).join("\n");
        assert!(
            window(&rt, 2).len() > before,
            "the moment delivered nothing"
        );
        for intent in ["one", "two", "three"] {
            assert!(after.contains(intent), "{intent} was lost: {after}");
        }
    }

    // ── a new character, put into its world ─────────────────────────────────

    #[test]
    fn a_new_character_arrives_at_the_way_in_and_starts_thinking() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);

        assert!(rt
            .embody_in_world(1, WORLD, None, "Maker-01", None, 0)
            .unwrap());
        let (world, body) = rt.body_of(1).expect("it has a body");
        assert_eq!(body, Runtime::body_id(1));
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            npc_map::world::Where::new("vault-command", "command-room"),
            "it did not arrive at the way in"
        );
        // And at a pace that keeps it working rather than settling.
        assert_eq!(rt.scheduler.pace_of(1), Some(Pace::WORKING));
    }

    #[test]
    fn a_character_whose_world_has_no_map_is_left_without_a_body() {
        // Most worlds have none. Lore and no body is a character, not a
        // failure, and the caller has to be able to tell that from an error.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        assert!(!rt
            .embody_in_world(1, "unmapped", None, "Maker-01", None, 0)
            .unwrap());
        assert!(rt.body_of(1).is_none());
    }

    #[test]
    fn putting_a_character_back_in_the_body_it_had_changes_nothing() {
        // The restart path and the create path are the same call, so it has to
        // be safe to make twice — once when the character was created, once
        // every boot after.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        rt.embody_in_world(1, WORLD, None, "Maker-01", None, 0)
            .unwrap();

        // It walks off the command level's arrival room — one stop, same level.
        let body = Runtime::body_id(1);
        let elsewhere = npc_map::world::Where::new("vault-command", "anteroom");
        let world = rt.hosted.get(WORLD).unwrap();
        world.with(|w| w.set_off(&body, elsewhere.clone()).unwrap());
        world.tick();
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            elsewhere
        );

        assert!(rt
            .embody_in_world(1, WORLD, None, "Maker-01", None, 0)
            .unwrap());
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            elsewhere,
            "it was sent back to the door"
        );
    }

    /// **A restart puts a body back where it was, not at the front door.**
    ///
    /// The world's own state — who is standing where — is held in RAM and goes
    /// with the process, so the character record's remembered room is the only
    /// thing that survives to rebuild it from.
    #[test]
    fn a_body_returns_to_the_room_it_was_remembered_in() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let green = "vault-casting/green-room";

        assert!(rt
            .embody_in_world(1, WORLD, None, "Maker-01", Some(green), 0)
            .unwrap());
        let (world, body) = rt.body_of(1).expect("it has a body");
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            Where::new("vault-casting", "green-room"),
            "it was sent to the arrival door instead of where it was"
        );
    }

    /// A remembered room the map no longer has is not an error — maps are
    /// authored and rooms get renamed. The character arrives at the door, which
    /// is what somebody whose room was demolished should do.
    #[test]
    fn a_body_whose_remembered_room_is_gone_arrives_at_the_door() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);

        assert!(rt
            .embody_in_world(
                1,
                WORLD,
                None,
                "Maker-01",
                Some("vault-casting/no-such-room"),
                0
            )
            .unwrap());
        let (world, body) = rt.body_of(1).expect("it has a body");
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            Where::new("vault-command", "command-room"),
            "a room the map does not have should fall through to the way in"
        );
    }

    #[test]
    fn a_body_id_is_derived_so_a_restart_finds_the_same_one() {
        // Derived rather than stored, the same reason a timeline id is: a
        // restart reconstructs the correspondence without a table that could
        // be lost or disagree.
        assert_eq!(Runtime::body_id(7), "npc-7");
        assert_ne!(Runtime::body_id(7), Runtime::body_id(8));
    }

    // ── the standing instruction ────────────────────────────────────────────

    #[test]
    fn a_character_with_nothing_asked_of_it_is_pointed_at_the_work() {
        // Having nothing to do is itself a standing instruction. An NPC that
        // stands still because nothing was assigned reads as scenery.
        //
        // **At the work, not at the map.** This used to assert "not been in" —
        // the instruction was to go and see an unvisited room, and characters
        // did precisely that: a seventy-eight room tour, hours long, that
        // produced nothing to talk about, because a room is not a subject.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        rt.embody_in_world(1, WORLD, None, "Maker-01", None, 0)
            .unwrap();

        let nudge = rt.nudge_for(1).expect("something to be getting on with");
        assert_eq!(nudge, NO_MISSION);
        assert!(nudge.contains("within reach"), "{nudge}");
        // **No verb of motion.** The standing task is the most recent thing in
        // the window, so whatever it tells a character to *do* is what the
        // character does — and every version of this that ended with somewhere
        // to go produced a cast that only ever went there.
        for motion in ["go ", "walk", "find somewhere", "where people are"] {
            assert!(
                !nudge.to_lowercase().contains(motion),
                "the standing task tells it to move (`{motion}`): {nudge}"
            );
        }
    }

    /// **Company changes the standing task.** Told to go somewhere new every
    /// quiet turn, two characters explored a seventy-eight room building and
    /// never held a conversation: each moved every four seconds, so sharing a
    /// room lasted one tick and neither had a reason to stay for the second.
    #[test]
    fn a_character_that_is_not_alone_is_told_to_stay_and_talk() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));

        // Somebody walks in, and what there is to do changes with them.
        embody(&rt, 2, "m2", "green-room");
        let together = rt.nudge_for(1).expect("bound");
        // Speaking, and about something in particular — "say what you have
        // been looking at" got three characters agreeing about silence for a
        // hundred turns, because none of them had to name a thing.
        // **Naming the act is what makes an instruction land.** "Talk to them"
        // is a wish; "`tell` or `ask`" is the branch the grammar has an arm for,
        // and the difference showed up as a cast that only ever walked.
        assert!(
            together.contains("`tell`") && together.contains("`ask`"),
            "the instruction names no act to carry it out: {together}"
        );
        assert!(
            together.contains("particular thing"),
            "it permits a vague answer: {together}"
        );
        // **And it names them.**
        //
        // This is the whole difference between an instruction a character can
        // act on and one it cannot. `tell` and `ask` take a name and refuse one
        // they cannot find, and the only other place a name appears — the
        // situation percept — is suppressed while nothing moves. Told to talk
        // to an unnamed somebody, a live character addressed `you`, was
        // refused, and did it again.
        assert!(
            together.contains("Maker-02"),
            "the standing task names no addressee: {together}"
        );
        assert!(
            !together.contains("{who}"),
            "the placeholder was never filled: {together}"
        );
        // Each is told about the other, never about itself.
        let other = rt.nudge_for(2).expect("bound");
        assert!(other.contains("Maker-01"), "{other}");
        assert!(
            !other.contains("Maker-02"),
            "it was told about itself: {other}"
        );

        // And when they part, it changes back.
        rt.hosted
            .get(WORLD)
            .unwrap()
            .with(|w| w.set_off("m2", at("band-one")).unwrap());
        rt.hosted.get(WORLD).unwrap().tick();
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));
    }

    /// **The standing task is for the quiet turns, and the situation is not
    /// news.**
    ///
    /// This gate has been wrong in both directions. Gated on inbox *depth* it
    /// never fired, because a character with a body is handed the room it is
    /// standing in every moment, so the depth is never zero — and the task
    /// written for the case of having company was the one that never arrived.
    /// Ungated it fired every turn, restating "nothing has been asked of you"
    /// more recently than anything a companion had actually said.
    ///
    /// So the question is whether anything *happened*, which is what
    /// [`crate::engine::tick::Inbox::has_news`] answers.
    #[test]
    fn the_standing_task_waits_for_a_turn_with_nothing_in_it() {
        use crate::engine::event::{EventKind, Salience};

        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");

        // Where a character is standing is a fact about the room, not something
        // that happened — and this is the case the old depth gate mistook for a
        // busy character, because a body is handed its situation whenever the
        // world moves under it.
        rt.scheduler.deliver(
            1,
            0,
            Salience::IDLE,
            EventKind::Situation {
                text: "You are in the green room.".into(),
            },
        );
        assert!(
            rt.scheduler.inbox_depth(1).unwrap() > 0,
            "the situation should be queued"
        );
        assert_eq!(
            rt.scheduler.quiet_for(1, 10_000),
            Some(10_000),
            "standing in a room is not something that happened"
        );

        // Nor is the standing task itself, or it would keep its own gate open.
        rt.scheduler.deliver(
            1,
            0,
            Salience::IDLE,
            EventKind::Nudge {
                text: NO_MISSION.into(),
            },
        );
        assert_eq!(rt.scheduler.quiet_for(1, 10_000), Some(10_000));

        // Somebody speaks, and now there is something to answer.
        rt.scheduler.deliver(
            1,
            0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Maker-02".into(),
                text: "that the redoubt burned twice".into(),
                to: crate::engine::event::Addressed::You,
            },
        );
        assert_eq!(
            rt.scheduler.quiet_for(1, 10_000),
            Some(0),
            "being spoken to is news"
        );

        // **And the quiet does not restart the moment it is read.** Draining the
        // speech leaves the character having been spoken to at t=0, not having
        // been alone forever — which is the whole point of the clock: the gap
        // between two turns of a conversation is not idleness.
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());
        assert_eq!(
            rt.scheduler.quiet_for(1, 10_000),
            Some(10_000),
            "ten seconds after the last thing said to it"
        );
        assert!(
            rt.scheduler.quiet_for(1, 10_000).unwrap() < IDLE_AFTER_MS,
            "ten seconds is a pause in a conversation, not an idle character"
        );
    }

    #[test]
    fn somebody_in_the_next_room_is_not_company() {
        // Company is who is *here*. Being able to see somebody through a
        // doorway is a reason to go to them, not a conversation.
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        embody(&rt, 2, "m2", "ring-north");
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));
    }

    #[test]
    fn a_character_with_no_body_is_not_told_to_explore_anything() {
        // It has nowhere to go and nobody to talk to; instructing it otherwise
        // is instructing it to do something it cannot.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        assert_eq!(rt.nudge_for(1), None);
    }

    /// The standing instruction is a **disposition**, not a script. A named
    /// destination would be an order every character in the world followed
    /// identically — which looks like emergence and is the opposite of it.
    #[test]
    fn the_standing_instruction_names_no_particular_place_or_person() {
        let vault = npc_map::MapSet::load_dir(ROOMS).unwrap();
        let words = NO_MISSION.to_lowercase();
        for area in vault.areas() {
            for node in &area.nodes {
                assert!(
                    !words.contains(&node.name.to_lowercase()),
                    "it names {}",
                    node.name
                );
            }
        }
        assert!(!words.contains("maker-"), "it names somebody");
    }

    #[test]
    fn a_standing_instruction_replaces_the_one_before_it() {
        use crate::engine::event::EventKind;
        let a = EventKind::Nudge { text: "one".into() };
        let b = EventKind::Nudge { text: "two".into() };
        assert_eq!(a.replaces().as_deref(), Some("nudge"));
        assert_eq!(a.replaces(), b.replaces(), "two tasks at once");
        // Its own band, beside the situation — they are different things and
        // neither may retire the other.
        assert_ne!(
            a.replaces(),
            EventKind::Situation { text: "x".into() }.replaces()
        );
    }

    #[test]
    fn a_standing_instruction_never_interrupts_and_reads_as_written() {
        use crate::engine::event::{Event, EventKind, Salience};
        let e = Event::new(
            1,
            0,
            Salience::IDLE,
            EventKind::Nudge {
                text: format!("  {NO_MISSION}\n"),
            },
        );
        assert!(!e.preempts(), "a standing task interrupted a character");
        assert_eq!(e.prose(), NO_MISSION);
    }

    // ── one daemon, several worlds ──────────────────────────────────────────

    /// **Why a hosted world has an id at all.** Every character is created with
    /// a `world_id` naming the world document it belongs to, so the places it
    /// can stand in must be findable by that same name. Anything else needs a
    /// second table to reconcile the two, and a second table is a thing that can
    /// disagree.
    #[test]
    fn a_characters_world_id_finds_the_places_it_can_stand_in() {
        let rt = vaulted();
        // What a character carries is a string from its own document.
        let world_id: String = WORLD.to_string();
        let places = rt.hosted.get(&world_id).expect("hosted under its own id");
        assert_eq!(places.id(), world_id);

        places.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.embody(1, &world_id, "m1", 0).expect("bound by world id");
        assert_eq!(
            rt.body_of(1).map(|(w, b)| (w.id().to_string(), b)),
            Some((world_id, "m1".into()))
        );
    }

    #[test]
    fn two_worlds_are_two_places_and_a_body_in_one_is_not_in_the_other() {
        let rt = rt();
        rt.host("creators-vault", Path::new(ROOMS)).unwrap();
        rt.host("second-world", Path::new(ROOMS)).unwrap();
        rt.hold_world("creators-vault", true);
        rt.hold_world("second-world", true);
        assert_eq!(rt.moments().len(), 2);

        let first = rt.hosted.get("creators-vault").unwrap();
        let second = rt.hosted.get("second-world").unwrap();
        first.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        assert!(
            second.read(|w| w.actor("m1").is_none()),
            "one world, not two"
        );

        // The same body id in two worlds is two bodies, and two characters may
        // have them.
        second.with(|w| w.enter("m1", "Someone Else", at("band-one")).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.scheduler.wake(2, 0, 0);
        rt.embody(1, "creators-vault", "m1", 0).unwrap();
        rt.embody(2, "second-world", "m1", 0)
            .expect("a different world");
    }

    #[test]
    fn unhosting_one_world_leaves_the_others_running() {
        let rt = rt();
        rt.host("creators-vault", Path::new(ROOMS)).unwrap();
        rt.host("second-world", Path::new(ROOMS)).unwrap();
        rt.hold_world("creators-vault", true);
        rt.hold_world("second-world", true);

        rt.hosted
            .get("creators-vault")
            .unwrap()
            .with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        rt.hosted
            .get("second-world")
            .unwrap()
            .with(|w| w.enter("m9", "Maker-09", at("band-one")).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.scheduler.wake(9, 0, 0);
        rt.embody(1, "creators-vault", "m1", 0).unwrap();
        rt.embody(9, "second-world", "m9", 0).unwrap();

        assert!(rt.unhost("creators-vault"));
        assert!(rt.body_of(1).is_none(), "its character kept a body");
        assert!(
            rt.body_of(9).is_some(),
            "another world's character lost one"
        );
        assert_eq!(rt.moments().len(), 1);
    }

    #[test]
    fn only_authored_worlds_with_rooms_are_hosted() {
        // Driven by the registry rather than by what is on disk: a map
        // directory nothing authored must not be hosted under an id no
        // character can name, and an authored world with no map is a world
        // whose characters have lore and no bodies — the common case.
        let dir = std::env::temp_dir().join("npcd-host-authored");
        let _ = std::fs::remove_dir_all(&dir);
        let maps = dir.join(MAPS);
        std::fs::create_dir_all(maps.join("has-rooms")).unwrap();
        std::fs::create_dir_all(maps.join("never-authored")).unwrap();
        for f in std::fs::read_dir(ROOMS).unwrap().flatten() {
            if f.path().is_file() {
                std::fs::copy(f.path(), maps.join("has-rooms").join(f.file_name())).unwrap();
            }
        }
        std::fs::create_dir_all(maps.join("has-rooms").join("parts")).unwrap();
        for f in std::fs::read_dir(Path::new(ROOMS).join("parts"))
            .unwrap()
            .flatten()
        {
            std::fs::copy(
                f.path(),
                maps.join("has-rooms").join("parts").join(f.file_name()),
            )
            .unwrap();
        }

        let rt = rt();
        // Only what the registry knows is offered. `never-authored` has a
        // directory on disk and no document, so it is not among these.
        let done = rt.host_authored(&dir, ["has-rooms", "no-rooms"]);
        // `no-rooms` is authored with nowhere to stand — lore and no bodies,
        // which is the common case and not an error.
        let ids: Vec<&str> = done.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["has-rooms"]);
        assert!(done[0].1.is_ok());
        assert!(rt.hosted.get("has-rooms").is_some());
        assert!(rt.hosted.get("never-authored").is_none());

        rt.unhost("has-rooms");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
