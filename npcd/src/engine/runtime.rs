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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

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
const PROJECTION_MARKER: &str = "<|projection|>";

use crate::engine::authoring;
use crate::engine::ingest;
use crate::engine::life;
use crate::engine::loading::{LoadProgress, LoadStep};
use crate::engine::mind::Minds;
use crate::engine::prompt::Persona;
use crate::engine::schema;
use crate::engine::tick::{Scheduler, Shared as SharedScheduler};
use crate::engine::tools::Mode;
use crate::engine::watcher::Ledger;
use crate::mind::Mind;
use crate::model;

/// How often the driver thread looks for characters that are due.
///
/// Not the tick rate — that is per character and salience-driven. This is only
/// how finely the scheduler's clock is quantised, so a preempt is acted on
/// within this long at worst.
const DRIVE_INTERVAL: Duration = Duration::from_millis(100);

/// Tokens one prefill forward carries — the model's own per-forward ceiling.
///
/// See the note at the load site. In short: the wave's fixed cost is per slab,
/// not per token, so a budget close to one document's size makes every document
/// pay a whole sweep. At the ceiling a slab carries four or five documents and
/// pays it once.
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

/// Everything the prompt needs about one character, owned.
///
/// Owned rather than borrowed because it crosses from the async world (where the
/// authored state lives behind an `RwLock`) into the tick thread. The alternative
/// is holding that lock across a decode, which would block every authoring write
/// for the length of a generation.
#[derive(Debug, Default, Clone)]
pub struct OwnedPersona {
    pub name: String,
    pub identity: String,
    pub manner: String,
    pub beliefs: Vec<String>,
    pub relationships: Vec<String>,
    pub intent: Option<String>,
    pub situation: String,
    pub world: String,
    pub mode: Mode,
}

impl OwnedPersona {
    pub fn as_persona(&self) -> Persona<'_> {
        Persona {
            name: &self.name,
            identity: &self.identity,
            manner: &self.manner,
            beliefs: &self.beliefs,
            relationships: &self.relationships,
            intent: self.intent.as_deref(),
            situation: &self.situation,
            world: &self.world,
        }
    }
}

/// Resolves a character's authored state at tick time.
pub type PersonaSource = Arc<dyn Fn(u64) -> Option<OwnedPersona> + Send + Sync>;

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
    /// Set on shutdown so the driver thread stops rather than being killed
    /// mid-tick with a half-written turn.
    stopping: AtomicBool,
    started: Instant,
}

impl Runtime {
    pub fn new(mind_handle: Mind, data: &Path) -> Arc<Self> {
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
            minds: RwLock::new(None),
            persona: RwLock::new(None),
            stopping: AtomicBool::new(false),
            started: Instant::now(),
        })
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

    fn persona_of(&self, npc_id: u64) -> Option<OwnedPersona> {
        let g = self.persona.read().unwrap();
        g.as_ref().and_then(|f| f(npc_id))
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
    /// The characters to wake, by id. From the substrate's character store.
    pub cast: Vec<u64>,
    /// Every personality id the mind declares.
    ///
    /// Not the cast: this is what a *layer directory* can be named after, which
    /// is a personality (`layers/memory/zen/`), not an instantiated character.
    /// A world's biographies exist whether or not anyone has cast them.
    pub characters: Vec<String>,
    /// The world clock at startup.
    pub world_ms: u64,
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

    let mut builder = model::model()
        .builder()
        .guests(guests)
        .prefill_pass_tokens(PREFILL_PASS_TOKENS)
        // **The engine's redo log goes under `--data`, with everything else this
        // daemon writes.**
        //
        // Unset, it falls back to the *process working directory* — so a daemon
        // launched from the repo root wrote its substrate to `candle/.substrate`
        // while the character store sat correctly in `npcd/.substrate`. Two
        // substrates, one of them somewhere nobody would look, and nothing said
        // so: `/v1/substrate/storage` reports on `--data` and showed 65 MB while
        // the real log reached **190 GB** across a morning of re-ingests. It
        // filled the disk, and the failure surfaced as a linker error.
        .workspace_path(rt.data.clone());
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
    *rt.minds.write().unwrap() = Some(Arc::new(Minds::new(
        Arc::clone(&engine),
        conv_config.clone(),
    )));

    // ── tool calibration ───────────────────────────────────────────────────
    //
    // Before the layers, deliberately. Every layer frames on the shared system
    // prompt and the tool catalog is part of it, so a document prefilled while
    // the catalog is still absent captures its KV — and the wide-Q signature the
    // gather matches against — under a prompt no character will ever think
    // under. Nothing would fail; retrieval would simply be worse than it should
    // be, for the life of that substrate.
    p.set_step(LoadStep::Calibrating);
    let tools = crate::engine::tools::CATALOG;
    p.set_progress(0, tools.len() as u64);
    for (i, t) in tools.iter().enumerate() {
        p.set_detail(t.name);
        p.set_progress(i as u64 + 1, tools.len() as u64);
    }
    tracing::info!(
        "tools: {} in the catalog, {} calibration examples",
        tools.len(),
        tools.iter().map(|t| t.examples.len()).sum::<usize>()
    );

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
    let projection = schema::build(rt.mind.as_deref(), "battle-cities");
    if projection.is_none() && rt.mind.is_some() {
        tracing::warn!(
            "no usable projection schema — documents will be written but will produce \
             no provenance signatures, so nothing will gather them"
        );
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

    // ── the cast ───────────────────────────────────────────────────────────
    p.set_step(LoadStep::Waking);
    p.set_progress(0, plan.cast.len() as u64);
    let now = 0;
    for (i, id) in plan.cast.iter().enumerate() {
        rt.scheduler.wake(*id, now, plan.world_ms);
        p.set_progress(i as u64 + 1, plan.cast.len() as u64);
    }
    tracing::info!("cast: {} character(s) awake", rt.scheduler.population());

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
fn drain(handle: &TurnHandle, addr: &str) -> (Option<TurnResponse>, Vec<ProjectionEvent>) {
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
fn persist_signatures(
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
    let synthetic = proj.is_none().then(|| {
        let prompt = source.prompt();
        let b = Builder::for_plain_prompt(&prompt);
        let l = &b.schema().layers[0];
        let (layer, group) = (l.id, l.groups[0].id);
        (prompt, b, layer, group)
    });
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
                    let minds = rt.minds.read().unwrap().clone();
                    let persona = rt.persona_of(id);
                    let day = crate::engine::sleep::day_of(world_ms);

                    rt.scheduler.tick(id, now_ms, world_ms, |events, window| {
                        let (Some(minds), Some(p)) = (minds.as_ref(), persona.as_ref()) else {
                            // No engine yet, or a character the authored state
                            // no longer knows. Perception still lands in the
                            // window — that half needs nothing — and no acts is
                            // the honest answer rather than an invented one.
                            return Vec::new();
                        };
                        match minds.think(id, &p.as_persona(), p.mode, day, events, window) {
                            Ok(t) => {
                                for r in &t.parsed.rejected {
                                    // Reported, never swallowed: a character
                                    // failing to act and one choosing not to
                                    // look identical from outside and need
                                    // completely different fixes.
                                    tracing::warn!("npc {id}: act rejected — {r:?}");
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
                                t.parsed.acts.iter().map(|a| a.summary()).collect()
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
}
