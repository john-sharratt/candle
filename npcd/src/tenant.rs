//! npcd as a tenant of the host (`docs/shared_model_host.md` §3.1).
//!
//! [`NpcdArgs`] is npcd's own settings — the `npcd` binary's flags and the
//! `npcd:` section of `motherd.yaml` — and [`NpcdArgs::resolve`] turns them into
//! an [`NpcdTenant`], refusing a mistyped `--mind` on the spot. [`NpcdTenant`]
//! stands the daemon up on the host's runtime: the authored registries, the
//! cast on its substrate, the runtime and its sinks, the hosted worlds, the
//! routes — and starts the engine load on the host's engine slot.

use std::cell::Cell;
use std::env::current_dir;
use std::net::SocketAddr;
use std::path::{absolute, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use axum::extract::{ConnectInfo, Request};
use axum::middleware::{self, Next};
use axum::Router;
use candle_conversation::engine_slot::CoreRequests;
use candle_conversation::models::DialectType;
use candle_conversation::persistence::tombstone_expiry::DEFAULT_TOMBSTONE_RETENTION;
use candle_conversation::persistence::{OpenStage, SUBSTRATE_DIR};
use candle_conversation::Capabilities;
use clap::Args;
use cluster::dispatch::Dispatch;
use futures::future::BoxFuture;
use host::guard::{Api, Guarded, Routes};
use host::{OneShot, Running, Tenant, TenantContext, TenantUi, MODEL};
use include_dir::{include_dir, Dir};
use serde::Deserialize;
use serde_json::Value;
use tokio::runtime::Handle;
use tokio::sync::RwLock;
use tokio::task::{block_in_place, spawn_blocking, JoinHandle};
use tokio::time::interval;
use web::auth::Roles;

use crate::accounts::Accounts;
use crate::api::{self, Authored};
use crate::authority;
use crate::clock;
use crate::collections::Libraries;
use crate::effector::router::{router as effector_router, Local};
use crate::effector::token::Tokens;
use crate::engine;
use crate::engine::guardian::{GuardianBuilder, GuardianConfig, RuntimeProbe};
use crate::engine::identity::Authored as Identities;
use crate::engine::runtime::{engine_asks, start as start_engine, Arrived, LoadPlan, Runtime};
use crate::identity::log_identity;
use crate::image_links::{self, ImageLinks};
use crate::images::Images;
use crate::mind::files::MindFiles;
use crate::mind::{Mind, MindRepo};
use crate::mind_store::{self, MindSource};
use crate::npcs::{Casting, Npcs};
use crate::personality_portrait;
use crate::projection;
use crate::registry::authored::{PERSONALITIES, WORLDS};
use crate::registry::Registry;
use crate::reports::NpcdReports;

/// The tenant's name: its log attribution, status key and route owner.
pub const NAME: &str = "npcd";

/// What npcd needs of a model: `<tool_call>`, `<think>` and `</think>` as one
/// token each — its act stencils and parsers key on them — and the Qwen3.5
/// dialect its prompts are written in.
pub fn requires() -> Capabilities {
    Capabilities {
        tokens: &["<tool_call>", "<think>", "</think>"],
        dialect: DialectType::Qwen35,
    }
}

/// npcd's console, `web/content/npcd`, compiled in.
pub static CONSOLE: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../web/content/npcd");

/// npcd's flags.
#[derive(Args, Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct NpcdArgs {
    /// Where the engine's own state lives — `substrate/`, `accounts/` and the
    /// mind's store, the things the daemon writes rather than a person. The
    /// `npcd` directory in the source tree when absent.
    #[arg(long)]
    pub data: Option<PathBuf>,

    /// The mind: a repository an agent holds, by its name there — the
    /// cluster's agents, or with no cluster the one in this process
    /// (`docs/substrate_git.md` §8, §9). Its packs and branches are records in
    /// the substrate and its files are read at a commit, so this machine needs
    /// neither the repository nor a way to its origin. With none, the daemon
    /// runs on the bundled schema with no authored content.
    #[arg(long, value_name = "NAME")]
    pub mind_repo: Option<String>,

    /// The branch of `--mind-repo` that is the record.
    #[arg(long, value_name = "BRANCH", default_value = DEFAULT_MIND_BRANCH)]
    pub mind_branch: String,

    /// Log the identity each request to npcd's own routes arrived with, the
    /// peer it came from, and the role it resolved to. Header **values are
    /// never logged** — each header reports only `set`, `EMPTY` or absent. Off
    /// by default and noisy when on: every such request, polls included.
    #[arg(long)]
    pub log_identity: bool,

    /// Retire every character's conversation at startup, so each one opens a
    /// fresh conversation instead of rejoining where it stopped. Each
    /// character's memory, beliefs, relationships and place in the world are
    /// untouched.
    #[arg(long)]
    #[serde(skip)]
    pub forget_conversations: bool,

    /// Retire every dream every character has kept, at startup.
    #[arg(long)]
    #[serde(skip)]
    pub forget_dreams: bool,

    /// Empty the journal every character has kept, at startup.
    #[arg(long)]
    #[serde(skip)]
    pub forget_journal: bool,

    /// Retire every conversation in the substrate at startup, and re-ingest the
    /// mind. The cast is kept: who exists, who owns them and where they stand
    /// are untouched.
    #[arg(long)]
    #[serde(skip)]
    pub wipe_conversations: bool,

    /// How many days the substrate keeps a retired conversation's tombstone
    /// before a compaction may shrink it to a 24-byte entry in the deleted set
    /// (`docs/replicated_substrate.md` §6.8). Only when the bytes are reclaimed
    /// depends on it: the conversation stays retired either way.
    #[arg(long, value_name = "DAYS", default_value_t = DEFAULT_TOMBSTONE_RETENTION_DAYS)]
    pub tombstone_retention: u64,
}

/// `--mind-branch`'s default.
const DEFAULT_MIND_BRANCH: &str = "main";

/// `--tombstone-retention`'s default, in days: the substrate's own.
const DEFAULT_TOMBSTONE_RETENTION_DAYS: u64 =
    DEFAULT_TOMBSTONE_RETENTION.as_secs() / SECONDS_PER_DAY;

const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

impl NpcdArgs {
    /// The flags that act once, at the start they are given to. A self-heal
    /// relaunch drops them (`host::OneShot`), and `motherd.yaml` cannot set
    /// them — a config file is read at every start, so one there would forget
    /// at every start. Run the `npcd` binary with the flag once instead.
    pub const ONE_SHOT: &'static [OneShot] = &[
        OneShot {
            flag: "--forget-conversations",
            takes_value: false,
        },
        OneShot {
            flag: "--forget-dreams",
            takes_value: false,
        },
        OneShot {
            flag: "--forget-journal",
            takes_value: false,
        },
        OneShot {
            flag: "--wipe-conversations",
            takes_value: false,
        },
    ];

    /// What npcd asks of the engine it shares, without loading npcd: what a
    /// host running zend alone seats in npcd's place, so zend's engine is the
    /// one built beside npcd. Its guests stay out — co-resident models only a
    /// running npcd calls.
    pub fn engine_requests(&self) -> CoreRequests {
        engine_asks(MODEL.builder()).take_core_requests()
    }
}

impl Default for NpcdArgs {
    fn default() -> Self {
        Self {
            data: None,
            mind_repo: None,
            mind_branch: DEFAULT_MIND_BRANCH.to_string(),
            log_identity: false,
            forget_conversations: false,
            forget_dreams: false,
            forget_journal: false,
            wipe_conversations: false,
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION_DAYS,
        }
    }
}

impl NpcdArgs {
    /// npcd, as a tenant: the schema resolved and the data directory chosen.
    /// Runs before logging is up.
    ///
    /// The schema first, because a mistyped `--mind` should be refused on the
    /// spot, not after the substrate has been opened and half the daemon stood
    /// up. Fatal rather than a fallback: a daemon that quietly reverted to the
    /// bundled placeholder would run with none of the named mind's content.
    pub fn resolve(self) -> anyhow::Result<NpcdTenant> {
        // Absolute from here on: the substrate's volume is measured from its
        // path, and a mirror's repository is opened by its path, so neither
        // may depend on where it is read from.
        let explicit = self
            .data
            .as_deref()
            .map(absolute)
            .transpose()
            .context("--data")?;
        // A mind repository is read from the substrate (`docs/substrate_git.md`
        // §8): its schema once the substrate is open, at start; until then the
        // bundled one stands in.
        let repo = match &self.mind_repo {
            Some(name) => {
                if explicit.is_none() {
                    anyhow::bail!("--mind-repo keeps its store under --data: name one");
                }
                let bad =
                    name.is_empty() || name.contains(['/', '\\', ':']) || name.starts_with('.');
                if bad {
                    anyhow::bail!("--mind-repo `{name}` is not a repository's name");
                }
                Some(MindSource {
                    name: name.clone(),
                    branch: self.mind_branch.clone(),
                })
            }
            None => None,
        };
        let schema = projection::Source::bundled();
        // **The compiled-in default is a path on the machine that built
        // this.** Used when it is really there; the working directory stands in
        // when it is not, and the choice is logged at start.
        let (data, fell_back) = match explicit {
            Some(dir) => (absolute(&dir).context("the data directory")?, false),
            None => {
                let built_at = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                if built_at.is_dir() {
                    (built_at, false)
                } else {
                    (current_dir()?, true)
                }
            }
        };
        Ok(NpcdTenant {
            args: self,
            schema,
            data,
            fell_back,
            repo,
        })
    }
}

/// npcd, configured and not yet started.
pub struct NpcdTenant {
    args: NpcdArgs,
    schema: projection::Source,
    data: PathBuf,
    /// The compiled-in data directory was not on this machine, and the working
    /// directory stands in.
    fell_back: bool,
    /// The mind repository the agents hold, with `--mind-repo`.
    repo: Option<MindSource>,
}

impl Tenant for NpcdTenant {
    fn name(&self) -> &'static str {
        NAME
    }

    fn requires(&self) -> Capabilities {
        requires()
    }

    fn substrate_dir(&self) -> PathBuf {
        self.data.join(SUBSTRATE_DIR)
    }

    fn ui(&self) -> TenantUi {
        TenantUi {
            hosts: vec!["bot.tokera.com".into(), "npcd.localhost".into()],
            content: &CONSOLE,
            default: false,
        }
    }

    fn start(self: Box<Self>, ctx: TenantContext) -> BoxFuture<'static, anyhow::Result<Running>> {
        Box::pin(start(*self, ctx))
    }
}

async fn start(t: NpcdTenant, ctx: TenantContext) -> anyhow::Result<Running> {
    let NpcdTenant {
        args,
        schema,
        data,
        fell_back,
        repo,
    } = t;
    // The mind is read through an agent: the cluster's, or the one in this
    // process (`docs/substrate_git.md` §9).
    if repo.is_some() && ctx.cluster.is_none() && ctx.agent.is_none() {
        anyhow::bail!(
            "--mind-repo is read through an agent, and there is none: give the process an \
             `agent:` file naming a checkout of it, or a --cluster"
        );
    }
    match &repo {
        // A repository an agent holds: read from the substrate once it opens.
        Some(r) => tracing::info!(
            "the mind is the repository `{}` on `{}`, read from the substrate",
            r.name,
            r.branch
        ),
        // A warning, not a note: a daemon with no mind starts, serves the
        // console and looks healthy, while every character in it has no world,
        // no personality and no libraries to think with.
        None => tracing::warn!(
            "no --mind-repo given: running on the {} schema's layers with no content \
             libraries, no worlds and no personalities. Pass --mind-repo <name> to run a real \
             one",
            schema.label
        ),
    }
    if fell_back {
        tracing::warn!(
            "the compiled-in data directory {} is not on this machine; falling back to the \
             working directory {} — pass --data to be sure",
            env!("CARGO_MANIFEST_DIR"),
            data.display()
        );
    }
    tracing::info!("data: {} (substrate and accounts)", data.display());

    // The cast, rebuilt from the substrate's redo log. The one read of that
    // log — every request after this is answered from memory. Shown on the
    // loading page as it runs: a store written before identities is rewritten
    // whole here, once, and on a large one that is minutes.
    let starting = Arc::clone(&ctx.starting);
    starting.set_step("opening", "Opening the substrate");
    let stage_shown: Cell<Option<OpenStage>> = Cell::new(None);
    let mut npcs = Npcs::load_reporting(&data, &|stage, done, total| {
        if stage_shown.get() != Some(stage) {
            stage_shown.set(Some(stage));
            starting.set_detail(stage.label());
        }
        starting.set_progress(done as u64, total as u64, stage.unit());
    })
    .map_err(|e| anyhow::anyhow!("opening the substrate at {}: {e:?}", data.display()))?;
    starting.set_step("starting", "Starting");
    npcs.substrate()
        .persistence
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .set_tombstone_retention(Duration::from_secs(
            args.tombstone_retention.saturating_mul(SECONDS_PER_DAY),
        ));

    // Accounts are records in the substrate, replicated but never published:
    // one carries a real email address and a provider subject id.
    let accounts = Accounts::open(npcs.substrate(), &data.join("accounts"))?;
    tracing::info!("accounts: {} known", accounts.len());

    let roles = ctx.roles.clone();
    if roles.is_empty() {
        tracing::warn!(
            "roles: nobody configured — worlds and personalities are read-only to everyone"
        );
    }

    // The mind. A repository the agents hold is read from the substrate, now
    // that it is open: the files of the commit the log names for its branch
    // (`docs/substrate_git.md` §8). One the log does not hold yet is pending:
    // the load pulls it through an agent once joined, and takes it in before
    // it builds anything from it — what is read from it here is read again
    // then (`read_again_on_arrival`). A `--mind` folder is read as it stands.
    let (schema, mind) = match &repo {
        Some(source) => {
            let (db, files) = mind_store::from_log(source, &npcs.substrate())?;
            match files {
                Some(files) => {
                    let label = mind_label(source, &files);
                    let schema = projection::Source::from_files(Arc::clone(&files), label)?;
                    (schema, Mind::of(files, MindRepo::Substrate(db)))
                }
                None => {
                    tracing::warn!(
                        "mind: the substrate holds no `{}` on `{}` yet — starting with no mind; \
                         once joined it is pulled through an agent and taken in before the \
                         engine builds anything from it",
                        source.name,
                        source.branch
                    );
                    (schema, Mind::pending(db))
                }
            }
        }
        None => {
            let mind = Mind::new(schema.dir.clone());
            (schema, mind)
        }
    };
    tracing::info!("projection schema: {}", schema.label);
    // Read before the mind moves into the API state: whether it still has to
    // arrive, for the hook that takes it in (below).
    let mind_pending = !mind.is_present();

    // Authored content — from the MIND when there is one, from the data
    // directory otherwise. Worlds and personalities are written by a person;
    // the data directory holds what the ENGINE wrote.
    let authored_dir = schema.dir.clone().unwrap_or_else(|| data.clone());
    let (worlds, personalities) = match mind.is_present() || repo.is_some() {
        true => (
            Registry::load_mind("world", &mind, WORLDS)?,
            Registry::load_mind("personality", &mind, PERSONALITIES)?,
        ),
        false => (
            Registry::load("world", authored_dir.join(WORLDS))?,
            Registry::load("personality", authored_dir.join(PERSONALITIES))?,
        ),
    };
    tracing::info!(
        "authored content: {} worlds, {} personalities",
        worlds.len(),
        personalities.len(),
    );

    let (libraries, feelings) = read_libraries(&schema);

    // The editor and the engine read the same schema through the same `Mind`.
    // Only a `--mind` folder is watched: a mind read from the substrate moves
    // when its branch does, not when a file does.
    let mind_for_engine = mind.clone();
    match mind.files() {
        Some(files) => tracing::info!("mind: {files:?}"),
        None => {
            tracing::info!("mind: none yet — the file editor will report it has nothing to edit")
        }
    }

    // Portraits, as blobs in the substrate; a character naming an image kept
    // as a file before then is pointed at the blob it became.
    let (images, renames) = Images::open(npcs.substrate(), &data.join("images"))
        .map_err(|e| anyhow::anyhow!("taking in the images at {}: {e}", data.display()))?;
    if !renames.is_empty() {
        let moved = npcs
            .repoint_portraits(&renames, api::now_ms())
            .map_err(|e| anyhow::anyhow!("repointing portraits: {e:?}"))?;
        tracing::info!("images: {moved} character portraits repointed");
    }
    let personality_portraits =
        personality_portrait::ingest(&personalities, personalities.dir(), &images);
    tracing::info!(
        "authored portraits: {} of {} personalities carry one",
        personality_portraits.len(),
        personalities.len(),
    );

    let mut authored = Authored::new(
        worlds,
        personalities,
        accounts,
        npcs,
        roles.clone(),
        libraries,
        mind,
        images,
        personality_portraits,
    );
    // A link one member signs, another member may be asked to open.
    if let Some(join) = &ctx.cluster {
        authored.image_links = ImageLinks::new(join.derive_key(image_links::PURPOSE));
    }

    // ── the engine ─────────────────────────────────────────────────────────
    //
    // The runtime is built first — it needs only the mind directory — then
    // handed to the state, then given the resolvers that read that state back.
    // That order breaks the cycle between them.
    let runtime = Runtime::new(mind_for_engine, &data)?;

    // The engine adopts the substrate the cast was just loaded from: one
    // `substrate/` takes one writable handle. See `Npcs::substrate`.
    runtime.set_substrate(authored.npcs.read().await.substrate());
    runtime.set_npcs(authored.npcs.clone());
    let authored = authored.with_runtime(runtime.clone());

    // The effector device: one durable token per character, and the `local`
    // router the engine's fast path drives in-process.
    let tokens = Arc::new(Tokens::load(data.join("tokens"))?);
    runtime.set_tokens(tokens.clone());
    let local_router = effector_router(Local::new(tokens.clone(), &runtime));
    runtime.set_effector_router(local_router.clone());

    install_sinks(&runtime, &authored, feelings);
    reload_on_move(&runtime, &authored);
    retire_drawn_pictures(&authored);
    // Only for a mind still pending: the hook holds the authored state, which
    // holds the runtime, and the load takes it back out once it has run.
    if let Some(source) = repo.as_ref().filter(|_| mind_pending) {
        read_again_on_arrival(&runtime, &authored, source.clone(), authored_dir.clone());
    }
    // A member follows each world it hosts until its lease says otherwise
    // (`docs/replicated_substrate.md` §9.2): known before the first is hosted.
    if ctx.cluster.is_some() {
        runtime.standing.set_clustered();
    }
    host_worlds(&runtime, &authored, &authored_dir).await;

    // ── the routes ─────────────────────────────────────────────────────────
    let routes = routes(
        &authored,
        &runtime,
        roles.clone(),
        local_router,
        args.log_identity,
    )?;

    // ── start the engine ───────────────────────────────────────────────────
    //
    // The loader runs on its own OS thread and returns immediately, so the
    // server binds while the model is still being fetched — the loading screen
    // has to be reachable during the load it is describing.
    let cast: Vec<Casting> = authored.npcs.read().await.cast();
    tracing::info!("engine: loading, {} character(s) to wake", cast.len());
    let characters = authored
        .personalities
        .read()
        .await
        .iter()
        .map(|r| r.id.clone())
        .collect();
    let identities = identities(&authored).await;
    start_engine(
        runtime.clone(),
        LoadPlan {
            engine: ctx.engine,
            cluster: ctx.cluster,
            forget_conversations: args.forget_conversations,
            forget_dreams: args.forget_dreams,
            forget_journal: args.forget_journal,
            wipe_conversations: args.wipe_conversations,
            cast,
            characters,
            authored: identities,
            mind_repo: repo,
            agent: ctx.agent.clone(),
            stopping: ctx.stopping.clone(),
        },
    );

    // The guardian, when `guardian.yaml` asks for one.
    let guardian = match GuardianConfig::load(&data)?.filter(|c| c.enabled) {
        Some(config) => {
            let guardian = GuardianBuilder::from_config(&config).build()?;
            runtime.set_guardian_log(guardian.log());
            tracing::info!("guardian on: {} modules", config.modules.len());
            Some(tokio::spawn(
                guardian.run(Arc::new(RuntimeProbe(runtime.clone()))),
            ))
        }
        None => None,
    };

    let stopping = runtime.clone();
    let npcs = authored.npcs.clone();
    Ok(Running {
        routes,
        reports: Arc::new(NpcdReports::new(authored, data)),
        stop: Box::new(move || Box::pin(stop(stopping, npcs, guardian))),
    })
}

/// Read the worlds and personalities again each time the mind folder moves to
/// a newer record — a publish on this member, or another member's that the
/// folder followed (`docs/replicated_substrate.md` §10.3). Holds the state
/// weakly, so the task ends with it.
fn reload_on_move(runtime: &Arc<Runtime>, authored: &Arc<Authored>) {
    let mut moves = runtime.ops.moves();
    let authored = Arc::downgrade(authored);
    tokio::spawn(async move {
        while moves.changed().await.is_ok() {
            let Some(authored) = authored.upgrade() else {
                return;
            };
            if authored.runtime.as_ref().is_some_and(|rt| rt.stopping()) {
                return;
            }
            authored.reload_authored().await;
            tracing::info!("authored: read again after the mind moved to a newer record");
        }
    });
}

/// Read again what was read from a mind pending at start, once the load has
/// taken it in (`engine::runtime::MindArrived`): the worlds and personalities
/// with their portraits, the libraries and the registers, the worlds to host
/// — and the cast, caught up from the cluster meanwhile, with who everybody
/// is, handed back to the load for the identities it installs and the
/// characters it wakes.
fn read_again_on_arrival(
    runtime: &Arc<Runtime>,
    authored: &Arc<Authored>,
    source: MindSource,
    authored_dir: PathBuf,
) {
    let state = authored.clone();
    runtime.set_mind_arrived(Arc::new(
        move || -> BoxFuture<'static, anyhow::Result<Arrived>> {
            let (state, source, authored_dir) =
                (state.clone(), source.clone(), authored_dir.clone());
            Box::pin(async move {
                state.reload_authored().await;
                if let Some(files) = state.mind.files() {
                    let label = mind_label(&source, &files);
                    // Fatal here as it is at start: a schema that cannot be read
                    // leaves characters with no registers to decode under.
                    let schema = projection::Source::from_files(files, label).map_err(|e| {
                        anyhow::anyhow!("mind: the schema that arrived could not be read: {e}")
                    })?;
                    tracing::info!("projection schema: {}", schema.label);
                    let (libraries, feelings) = read_libraries(&schema);
                    state.set_libraries(libraries);
                    if let Some(rt) = &state.runtime {
                        rt.set_feelings(feelings);
                    }
                }
                if let Some(rt) = &state.runtime {
                    host_worlds(rt, &state, &authored_dir).await;
                }
                let cast = {
                    let mut npcs = state.npcs.write().await;
                    npcs.refresh();
                    npcs.cast()
                };
                let characters: Vec<String> = state
                    .personalities
                    .read()
                    .await
                    .iter()
                    .map(|r| r.id.clone())
                    .collect();
                let worlds = state.worlds.read().await.len();
                tracing::info!(
                "authored content, read again: {worlds} worlds, {} personalities, {} character(s) \
                 to wake",
                characters.len(),
                cast.len(),
            );
                Ok(Arrived {
                    cast,
                    characters,
                    authored: identities(&state).await,
                })
            })
        },
    ));
}

/// What a mind read from the substrate is called in the log: its repository,
/// commit and branch.
fn mind_label(source: &MindSource, files: &MindFiles) -> String {
    format!(
        "{}@{} ({})",
        source.name,
        files.commit().map(|c| c.to_string()).unwrap_or_default(),
        source.branch
    )
}

/// The response and mood libraries, beside the schema — ingested untagged
/// and shared by every world — and the registers a character may say it is
/// in, the moods' ids, which a turn's decoding grammar is built over.
fn read_libraries(schema: &projection::Source) -> (Libraries, Vec<String>) {
    let libraries = Libraries::load(schema);
    tracing::info!(
        "libraries: {} responses ({} with examples), {} moods ({} with examples)",
        libraries.responses.len(),
        libraries.responses.with_examples(),
        libraries.moods.len(),
        libraries.moods.with_examples(),
    );
    let feelings = libraries
        .moods
        .sections
        .iter()
        .map(|s| s.id.clone())
        .collect();
    (libraries, feelings)
}

/// Stop npcd, in the order that loses nothing.
///
/// 1. The flag is latched and the guardian stopped, then every character task
///    is cancelled and awaited out — not cosmetic: a character's decode writes
///    turns to the substrate, and a cast killed mid-tick leaves the redo log
///    with a turn whose sealing never happened. Every dropped turn future has
///    released its scheduler slot when this returns.
/// 2. The character store's lock is taken once, so a place or mood write a
///    character left in flight lands before the substrate is closed.
/// 3. The engine shuts down: its persistence drains, its writer flushes, the
///    substrate detaches — stopping the scheduler when npcd is the engine's
///    last attachment — and the writer lock is released.
async fn stop(runtime: Arc<Runtime>, npcs: Arc<RwLock<Npcs>>, guardian: Option<JoinHandle<()>>) {
    runtime.stop();
    if let Some(g) = guardian {
        g.abort();
    }
    runtime.stop_characters().await;
    drop(npcs.write().await);
    let shut = spawn_blocking(move || runtime.shutdown_engine()).await;
    match shut {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::error!("npcd: {e:#}"),
        Err(e) => tracing::error!("npcd: engine shutdown panicked: {e}"),
    }
}

/// How often the pictures drawn for conversations are looked over for ones
/// past their day (`crate::images`): an hour, so one is gone within an hour of
/// its link expiring.
const RETIRE_DRAWN_EVERY: Duration = Duration::from_secs(60 * 60);

/// Retire, every [`RETIRE_DRAWN_EVERY`], the drawn pictures past their day
/// that no portrait uses. Every member does it; a picture another member
/// retired first has no note left to act on here. Holds the state weakly, so
/// the task ends with it.
fn retire_drawn_pictures(authored: &Arc<Authored>) {
    let authored = Arc::downgrade(authored);
    tokio::spawn(async move {
        let mut every = interval(RETIRE_DRAWN_EVERY);
        loop {
            every.tick().await;
            let Some(authored) = authored.upgrade() else {
                return;
            };
            if authored.runtime.as_ref().is_some_and(|rt| rt.stopping()) {
                return;
            }
            let mut named = authored.npcs.read().await.portrait_ids();
            named.extend(
                authored
                    .personality_portraits
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .values()
                    .cloned(),
            );
            let images = authored.images.clone();
            let now = api::now_ms();
            match spawn_blocking(move || images.retire_drawn(now, &named)).await {
                Ok(Ok(0)) => {}
                Ok(Ok(n)) => tracing::info!("images: {n} drawn pictures retired after their day"),
                Ok(Err(e)) => tracing::warn!("images: drawn pictures could not be retired: {e}"),
                Err(e) => tracing::error!("images: retiring drawn pictures panicked: {e}"),
            }
        }
    });
}

/// npcd's route tables, each over the state it reads.
struct Tables {
    /// The authored corpus and the cast.
    api: Api<Arc<Authored>>,
    /// The engine's surface.
    engine: Api<Arc<Authored>>,
    /// The effector device, under `/v1/local`.
    effector: Api<()>,
    /// The worlds' authority, which members write to
    /// (`docs/replicated_substrate.md` §9.3).
    authority: Api<Arc<Runtime>>,
}

fn tables(roles: &Roles, effector_router: Router) -> Tables {
    // The effector device authenticates by its own device token, never the
    // operator roles: a client may send both a token and `x-tokera-*`, and
    // only the token is read. Its paths name no key a gateway could route by,
    // so a cluster sends it to the best member.
    let effector = Api::<()>::new(roles.clone()).nest_self_authenticating(
        "/v1/local",
        Dispatch::Best,
        effector_router,
    );
    Tables {
        api: api::api(roles.clone()),
        engine: engine::api(roles.clone()),
        effector,
        authority: authority::routes::api(roles.clone()),
    }
}

/// The tables as one, refused when two of them overlap.
fn merged(
    api: Routes,
    engine: Routes,
    effector: Routes,
    authority: Routes,
) -> anyhow::Result<Routes> {
    api.merge("npcd api", engine, "npcd engine")?
        .merge("npcd", effector, "npcd effector")?
        .merge("npcd", authority, "npcd authority")
}

/// Every route npcd serves, and the role each needs — the table the host
/// merges with its own and the other tenants' at start, read without standing
/// npcd up.
pub fn declared(roles: &Roles) -> anyhow::Result<Vec<Guarded>> {
    let t = tables(roles, Router::new());
    let all = merged(
        Routes::table(t.api.declared()),
        Routes::table(t.engine.declared()),
        Routes::table(t.effector.declared()),
        Routes::table(t.authority.declared()),
    )?;
    Ok(all.declared)
}

/// npcd's routes, over its state.
fn routes(
    authored: &Arc<Authored>,
    runtime: &Arc<Runtime>,
    roles: Roles,
    effector_router: Router,
    log_identity_on: bool,
) -> anyhow::Result<Routes> {
    let t = tables(&roles, effector_router);
    let mut all = merged(
        t.api.into_routes(authored.clone()),
        t.engine.into_routes(authored.clone()),
        t.effector.into_routes(()),
        t.authority.into_routes(runtime.clone()),
    )?;
    if log_identity_on {
        tracing::info!(
            "--log-identity: logging the identity headers and the peer on every request to \
             npcd's own routes, as they arrive after the server has cleared them from any \
             peer it does not trust"
        );
        let log_roles = Arc::new(roles);
        all.router = all
            .router
            .layer(middleware::from_fn(move |req: Request, next: Next| {
                let roles = Arc::clone(&log_roles);
                async move {
                    let peer = req
                        .extensions()
                        .get::<ConnectInfo<SocketAddr>>()
                        .map(|ConnectInfo(a)| *a);
                    log_identity(
                        req.headers(),
                        &roles,
                        peer,
                        req.method().as_str(),
                        req.uri().path(),
                    );
                    next.run(req).await
                }
            }));
    }
    Ok(all)
}

/// The runtime's resolvers and sinks, each reading the authored state back.
fn install_sinks(runtime: &Arc<Runtime>, authored: &Arc<Authored>, feelings: Vec<String>) {
    // The clock resolver closes over the state: worlds run at their own pace,
    // and a character has to see its own world's time. `blocking_read` panics
    // on a runtime worker, so a call from inside the runtime steps off it.
    let clock_state = authored.clone();
    runtime.set_clock(Arc::new(move |npc_id: u64| {
        let read = || {
            clock::world_ms_for(
                &clock_state.npcs.blocking_read(),
                &clock_state.worlds.blocking_read(),
                npc_id,
                now_ms_i64(),
            )
        };
        match Handle::try_current() {
            Ok(_) => block_in_place(read),
            Err(_) => read(),
        }
    }));

    runtime.set_feelings(feelings);

    // Who each character is, resolved at tick time: a belief edited through
    // the authoring API has to reach the next tick.
    let persona_state = authored.clone();
    runtime.set_persona_source(Arc::new(move |npc_id: u64| {
        let npcs = persona_state.npcs.blocking_read();
        let payload = npcs.payload(npc_id)?;
        // The world's own `setting`, for the prompt's setting section.
        let world = persona_state
            .worlds
            .blocking_read()
            .get(&payload.world_id)
            .and_then(|r| {
                r.body
                    .get("setting")
                    .and_then(|d| d.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        // The personality's anchor, as prose — a reflection has no gather —
        // and where its characters belong.
        let personalities = persona_state.personalities.blocking_read();
        let personality = personalities.get(&payload.personality_id);
        let field = |k: &str| {
            personality.and_then(|r| r.body.get(k).and_then(|a| a.as_str()).map(str::to_owned))
        };
        let anchor = field("anchor").unwrap_or_default();
        let mut persona = engine::persona::of(payload, &world, &anchor);
        persona.home = field("home");
        Some(persona)
    }));

    // How it feels: a mood is what a character *is* between one thought and
    // the next.
    let mood_state = authored.clone();
    runtime.set_mood_sink(Arc::new(move |npc_id: u64, mood: &str| {
        let state = mood_state.clone();
        let mood = mood.to_string();
        run_sink(move || {
            state.npcs.blocking_write().remember_mood(npc_id, &mood);
        });
    }));
}

/// The places the cast stands in, one world at a time — `map/<world_id>/`
/// beside `worlds/<world_id>.yaml`. A map that does not hold together is an
/// authoring mistake to fix, not a reason for the rest to go unhosted.
async fn host_worlds(runtime: &Arc<Runtime>, authored: &Arc<Authored>, authored_dir: &Path) {
    let ids: Vec<String> = authored
        .worlds
        .read()
        .await
        .iter()
        .map(|r| r.id.clone())
        .collect();
    for (id, hosted) in runtime.host_authored(authored_dir, ids.iter().map(String::as_str)) {
        match hosted {
            Ok(w) => tracing::info!(
                "world `{id}` hosted — {} places",
                w.read(|w| w.map().areas().map(|a| a.nodes.len()).sum::<usize>())
            ),
            Err(e) => tracing::error!("world `{id}`: its map did not load: {e:#}"),
        }
    }
}

/// Who each personality, character and world is, for the projection's
/// identity collections — read here, where the registries live.
async fn identities(authored: &Arc<Authored>) -> Identities {
    let (npcs, personalities, worlds) = (
        authored.npcs.read().await,
        authored.personalities.read().await,
        authored.worlds.read().await,
    );
    let field = |b: &Value, k: &str| {
        b.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    Identities {
        anchors: personalities
            .iter()
            .map(|p| (p.id.clone(), field(&p.body, "anchor")))
            .collect(),
        // What a dream reads in the anchor's place.
        inner_lives: personalities
            .iter()
            .map(|p| (p.id.clone(), field(&p.body, "inner_life")))
            .collect(),
        // Rendered through the same function the fallback prompt uses, so a
        // character reads the same words either way. No anchor: it is its own
        // collection member.
        characters: npcs
            .cast()
            .iter()
            .filter_map(|c| {
                let payload = npcs.payload(c.npc_id)?;
                let owned = engine::persona::of(payload, "", "");
                Some((c.npc_id, engine::prompt::character(&owned.as_persona())))
            })
            .collect(),
        settings: worlds
            .iter()
            .map(|w| (w.id.clone(), field(&w.body, "setting")))
            .collect(),
    }
}

/// Run a fire-and-forget persistence sink write wherever it is safe to block:
/// on a runtime, deferred to the blocking pool; off one, inline. Every sink
/// write is last-writer-wins per character, so ordering does not matter.
fn run_sink(write: impl FnOnce() + Send + 'static) {
    match Handle::try_current() {
        Ok(handle) => {
            handle.spawn_blocking(write);
        }
        Err(_) => write(),
    }
}

/// Wall-clock milliseconds as the narrative clock takes them — signed, because
/// a world can be jumped backwards past its own epoch.
fn now_ms_i64() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Parser;

    /// **What npcd asks without loading is what its load asks** — its prefill
    /// pass — and no guests: a host running zend alone seats this in npcd's
    /// place, and nothing there would call a guest.
    #[test]
    fn the_engine_requests_carry_npcds_prefill_pass_and_no_guests() {
        let asks = NpcdArgs::default().engine_requests();
        assert_eq!(asks.prefill_pass_tokens, Some(8192));
        assert!(asks.guests.is_empty());
    }

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        npcd: NpcdArgs,
    }

    fn parse(args: &[&str]) -> NpcdArgs {
        let mut all = vec!["npcd"];
        all.extend_from_slice(args);
        Cli::try_parse_from(all).unwrap().npcd
    }

    /// **Off unless asked for**, each flag its own: a restart must not forget
    /// anything by accident, and a wipe is never implied by a narrower flag.
    #[test]
    fn forgetting_is_only_ever_asked_for() {
        let none = parse(&[]);
        assert!(!none.forget_conversations && !none.forget_dreams);
        assert!(!none.forget_journal && !none.wipe_conversations);
        let both = parse(&["--forget-conversations", "--forget-dreams"]);
        assert!(both.forget_conversations && both.forget_dreams && !both.wipe_conversations);
        let journal = parse(&["--forget-journal"]);
        assert!(journal.forget_journal && !journal.forget_conversations);
        assert!(parse(&["--wipe-conversations"]).wipe_conversations);
    }

    /// The flags and `motherd.yaml`'s `npcd:` section read the same settings,
    /// with the same defaults; there is no `--content` and no embedded console.
    #[test]
    fn flags_and_yaml_agree() {
        assert_eq!(parse(&[]), NpcdArgs::default());
        assert_eq!(
            serde_yaml::from_str::<NpcdArgs>("{}").unwrap(),
            NpcdArgs::default()
        );
        let flags = parse(&["--mind-repo", "battle-mind", "--log-identity"]);
        let yaml: NpcdArgs =
            serde_yaml::from_str("mind_repo: battle-mind\nlog_identity: true\n").unwrap();
        assert_eq!(flags, yaml);
        assert!(Cli::try_parse_from(["npcd", "--content", "web/content/npcd"]).is_err());
        assert!(serde_yaml::from_str::<NpcdArgs>("content: x").is_err());
    }

    /// Every one-shot flag is a real flag of the binary — a relaunch strips
    /// what the shell passed — and none can be set from `motherd.yaml`.
    #[test]
    fn the_one_shot_flags_are_flags_and_never_yaml() {
        for o in NpcdArgs::ONE_SHOT {
            assert!(Cli::try_parse_from(["npcd", o.flag]).is_ok(), "{}", o.flag);
            let key = o.flag.trim_start_matches("--").replace('-', "_");
            assert!(
                serde_yaml::from_str::<NpcdArgs>(&format!("{key}: true")).is_err(),
                "{key} must not be settable from motherd.yaml"
            );
        }
    }

    /// **A mind repository is read from the substrate**: no folder, its
    /// schema not read at resolve — the substrate is not open yet — and
    /// nothing on disk made (`docs/substrate_git.md` §8); `mind_repo:` reads
    /// the same from `motherd.yaml`.
    #[test]
    fn a_mind_repository_is_read_from_the_substrate() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().to_str().unwrap();
        let t = parse(&["--mind-repo", "mind", "--data", data])
            .resolve()
            .unwrap();
        let repo = t.repo.clone().unwrap();
        assert_eq!(repo.name.as_str(), "mind");
        assert_eq!(repo.branch.as_str(), "main");
        assert_eq!(t.schema.dir, None, "a mind in the substrate has no folder");
        assert!(t.schema.files.is_none(), "read once the substrate opens");
        assert_eq!(t.data, dir.path());

        let branch = parse(&[
            "--mind-repo",
            "mind",
            "--data",
            data,
            "--mind-branch",
            "record",
        ])
        .resolve()
        .unwrap();
        assert_eq!(branch.repo.unwrap().branch.as_str(), "record");

        assert!(
            Cli::try_parse_from(["npcd", "--mind", "../mind"]).is_err(),
            "a folder is no mind"
        );
        let no_data = parse(&["--mind-repo", "mind"]).resolve();
        assert!(no_data
            .err()
            .unwrap()
            .to_string()
            .contains("--mind-repo keeps its store under --data"));
        let yaml: NpcdArgs =
            serde_yaml::from_str(&format!("mind_repo: mind\ndata: '{data}'\n")).unwrap();
        assert_eq!(yaml, parse(&["--mind-repo", "mind", "--data", data]));
    }

    /// **A relative `--data` is made absolute at resolve**, as `motherd.yaml`'s
    /// `data: ../npcd-data` is: the substrate's volume is measured from its
    /// path. Resolving touches nothing on disk.
    #[test]
    fn a_relative_data_directory_resolves_absolute() {
        let here = current_dir().unwrap();
        let t = parse(&["--mind-repo", "mind", "--data", "unmade-npcd-data"])
            .resolve()
            .unwrap();
        assert_eq!(t.data, here.join("unmade-npcd-data"));
        let plain = parse(&["--data", "unmade-npcd-data"]).resolve().unwrap();
        assert_eq!(plain.data, here.join("unmade-npcd-data"));
        assert!(!here.join("unmade-npcd-data").exists());
    }

    /// **A mind that arrives after start is read again in place**, not
    /// restarted onto: the libraries and registers read from no mind are
    /// empty, and those read from the one that arrived take their place in
    /// the state the routes answer from.
    #[test]
    fn libraries_read_from_a_mind_that_arrived_replace_the_empty_ones() {
        let dir = tempfile::tempdir().unwrap();
        let mind = dir.path().join("mind");
        std::fs::create_dir_all(mind.join("moods")).unwrap();
        std::fs::write(mind.join(projection::SCHEMA_FILE), "layers: []\n").unwrap();
        std::fs::write(mind.join("moods").join("calm.yaml"), "template: Calm.\n").unwrap();

        let (none, felt) = read_libraries(&projection::Source::bundled());
        assert_eq!(none.moods.len(), 0);
        assert!(felt.is_empty(), "no mind, no registers");

        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let npcs = Npcs::load(&data).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &data.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &data.join("images")).unwrap();
        let authored = Authored::new(
            Registry::load("world", data.join(WORLDS)).unwrap(),
            Registry::load("personality", data.join(PERSONALITIES)).unwrap(),
            accounts,
            npcs,
            Roles::default(),
            none,
            Mind::new(None),
            images,
            Default::default(),
        );
        assert_eq!(authored.libraries().moods.len(), 0);

        let files = Arc::new(MindFiles::folder(&mind));
        let schema = projection::Source::from_files(files, "arrived".to_string()).unwrap();
        let (libraries, feelings) = read_libraries(&schema);
        assert_eq!(feelings, ["calm"]);
        authored.set_libraries(libraries);
        let now = authored.libraries();
        assert_eq!(now.moods.len(), 1);
        assert_eq!(now.moods.sections[0].id, "calm");
    }

    /// npcd needs what zend needs of the model, and the prescribed one has it.
    #[test]
    fn npcd_needs_the_three_markers_and_the_qwen35_dialect() {
        let r = requires();
        assert_eq!(r.tokens, ["<tool_call>", "<think>", "</think>"]);
        assert_eq!(r.dialect, DialectType::Qwen35);
        assert_eq!(host::MODEL.spec().chat_format, DialectType::Qwen35);
    }
}
