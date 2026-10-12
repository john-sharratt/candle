//! npcd as a tenant of the host (`docs/shared_model_host.md` §3.1).
//!
//! [`NpcdArgs`] is npcd's own settings — the `npcd` binary's flags and the
//! `npcd:` section of `tokerad.yaml` — and [`NpcdArgs::resolve`] turns them into
//! an [`NpcdTenant`], refusing a mistyped `--mind` on the spot. [`NpcdTenant`]
//! stands the daemon up on the host's runtime: the authored registries, the
//! cast on its substrate, the runtime and its sinks, the hosted worlds, the
//! routes — and starts the engine load on the host's engine slot.

use std::env::current_dir;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{ConnectInfo, Request};
use axum::middleware::{self, Next};
use axum::Router;
use candle_conversation::models::DialectType;
use candle_conversation::persistence::tombstone_expiry::DEFAULT_TOMBSTONE_RETENTION;
use candle_conversation::persistence::SUBSTRATE_DIR;
use candle_conversation::Capabilities;
use clap::Args;
use cluster::dispatch::Dispatch;
use futures::future::BoxFuture;
use host::guard::{Api, Guarded, Routes};
use host::{OneShot, Running, Tenant, TenantContext, TenantUi};
use notify::RecommendedWatcher;
use serde::Deserialize;
use serde_json::Value;
use tokio::runtime::Handle;
use tokio::sync::RwLock;
use tokio::task::{block_in_place, spawn_blocking, JoinHandle};
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
use crate::engine::runtime::{start as start_engine, LoadPlan, Runtime};
use crate::engine::watcher::{spawn as watch_mind, ReloadReport};
use crate::identity::log_identity;
use crate::images::Images;
use crate::mind::Mind;
use crate::mind_repo::MindRepo;
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

/// npcd's console: `web/content/npcd` in the source tree this binary was built
/// from.
pub const DEFAULT_UI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/content/npcd");

/// npcd's flags.
#[derive(Args, Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct NpcdArgs {
    /// Where the engine's own state lives — `substrate/` and `accounts/`, the
    /// things the daemon writes rather than a person. Also the fallback source
    /// for authored content (`worlds/`, `personalities/`) when no `--mind` is
    /// named. Defaults to the `--mind` directory, so a mind carries its own
    /// substrate beside it; with no `--mind` either, the `npcd` directory in
    /// the source tree.
    #[arg(long)]
    pub data: Option<PathBuf>,

    /// Load the projection schema and its content libraries from this
    /// directory instead of the compiled-in default. The directory must hold a
    /// `projection.yaml`; the folders beside it (`responses/`, `moods/`,
    /// `personalities/`, `worlds/`, …) are what its folder-backed collections
    /// and its authored registries read from.
    #[arg(long, value_name = "DIR")]
    pub mind: Option<PathBuf>,

    /// The mind as a repository the cluster's agents hold, by its name there —
    /// how a member with `--cluster` has a mind, in place of `--mind`
    /// (`docs/replicated_substrate.md` §10.3). It is mirrored under `--data`,
    /// at `mind/<name>`, and fetched and pushed through agents only, so this
    /// machine needs neither the repository nor a way to its origin.
    #[arg(long, value_name = "NAME")]
    pub mind_repo: Option<String>,

    /// The branch of `--mind-repo` that is the record.
    #[arg(long, value_name = "BRANCH", default_value = DEFAULT_MIND_BRANCH)]
    pub mind_branch: String,

    /// The directory the console is served from, read per request — saving a
    /// file there is the deployment. By default `web/content/npcd` in the
    /// source tree this binary was built from; a relative path given here
    /// resolves against the working directory.
    #[arg(long, default_value = DEFAULT_UI)]
    pub ui: PathBuf,

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
    /// relaunch drops them (`host::OneShot`), and `tokerad.yaml` cannot set
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
}

impl Default for NpcdArgs {
    fn default() -> Self {
        Self {
            data: None,
            mind: None,
            mind_repo: None,
            mind_branch: DEFAULT_MIND_BRANCH.to_string(),
            ui: PathBuf::from(DEFAULT_UI),
            log_identity: false,
            forget_conversations: false,
            forget_dreams: false,
            forget_journal: false,
            wipe_conversations: false,
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION_DAYS,
        }
    }
}

/// Where the substrate and accounts go: `--data` when it is given, and the
/// mind's own directory when it is not. `None` with neither, which leaves the
/// caller its compiled-in default.
///
/// **The mind, not the source tree.** A substrate is a mind's corpus ingested,
/// so run against a different mind it answers for the wrong canon.
pub fn data_dir(explicit: Option<PathBuf>, mind: Option<&Path>) -> Option<PathBuf> {
    explicit.or_else(|| mind.map(Path::to_path_buf))
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
        // A mind repository is read through its mirror under `--data`; until
        // the mirror holds the record, the daemon starts with no mind and
        // fetches it once joined (§10.3).
        let repo = match &self.mind_repo {
            Some(name) => {
                if self.mind.is_some() {
                    anyhow::bail!(
                        "--mind and --mind-repo both name the mind: a folder here, or a \
                         repository the agents hold — give one"
                    );
                }
                let data = self.data.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("--mind-repo keeps its mirror under --data: name one")
                })?;
                Some(MindRepo::new(data, name, &self.mind_branch)?)
            }
            None => None,
        };
        let mind = match &repo {
            Some(r) if r.ready() => Some(r.dir.clone()),
            Some(_) => None,
            None => self.mind.clone(),
        };
        let schema = projection::Source::resolve(mind.as_deref())?;
        // **The compiled-in default is a path on the machine that built
        // this.** Used when it is really there; the working directory stands in
        // when it is not, and the choice is logged at start.
        let (data, fell_back) = match data_dir(self.data.clone(), schema.dir.as_deref()) {
            Some(dir) => (dir, false),
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
    repo: Option<MindRepo>,
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
            content: self.args.ui.clone(),
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
    // With `--cluster` the mind is a repository the agents hold; without it
    // there are no agents to hold one (§10.3).
    match (&repo, ctx.cluster.is_some(), args.mind.is_some()) {
        (Some(_), false, _) => anyhow::bail!(
            "--mind-repo reaches the mind through the cluster's agents, and there is no \
             --cluster"
        ),
        (None, true, true) => anyhow::bail!(
            "with --cluster the mind is a repository the agents hold: name it with \
             --mind-repo, not a folder with --mind"
        ),
        _ => {}
    }
    if let Some(r) = repo.as_ref().filter(|r| !r.ready()) {
        tracing::warn!(
            "the mind repository `{}` is not mirrored at {} yet: starting with no mind; once \
             joined it is fetched through an agent and the process restarts onto it",
            r.name,
            r.dir.display()
        );
    }
    match &schema.dir {
        Some(dir) => tracing::info!(
            "projection schema: {} (collections resolve under {})",
            schema.label,
            dir.display()
        ),
        // A warning, not a note: a daemon with no mind starts, serves the
        // console and looks healthy, while every character in it has no world,
        // no personality and no libraries to think with.
        None => tracing::warn!(
            "no --mind given: running on the {} schema's layers with no content libraries, \
             no worlds and no personalities, and the substrate goes to the data directory's \
             default rather than beside a mind. Pass --mind <dir> to run a real one",
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

    // Authored content, read once — from the MIND when one is named, from the
    // data directory otherwise. Worlds and personalities are written by a
    // person; the data directory holds what the ENGINE wrote.
    let authored_dir = schema.dir.clone().unwrap_or_else(|| data.clone());
    let worlds = Registry::load("world", authored_dir.join(WORLDS))?;
    let personalities = Registry::load("personality", authored_dir.join(PERSONALITIES))?;
    tracing::info!(
        "authored content: {} worlds, {} personalities from {}",
        worlds.len(),
        personalities.len(),
        authored_dir.display()
    );

    // The cast, rebuilt from the substrate's redo log. The one read of that
    // log — every request after this is answered from memory.
    let mut npcs = Npcs::load(&data)
        .map_err(|e| anyhow::anyhow!("opening the substrate at {}: {e:?}", data.display()))?;
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

    // The response and mood libraries, beside the schema. Read once: they are
    // ingested untagged and shared by every world.
    let libraries = Libraries::load(&schema);
    tracing::info!(
        "libraries: {} responses ({} with examples), {} moods ({} with examples)",
        libraries.responses.len(),
        libraries.responses.with_examples(),
        libraries.moods.len(),
        libraries.moods.with_examples(),
    );
    // The registers a character may say it is in, captured once: the list is
    // compiled into a decoding grammar.
    let feelings: Vec<String> = libraries
        .moods
        .sections
        .iter()
        .map(|s| s.id.clone())
        .collect();

    // The editor and the engine read the same schema through the same `Mind`.
    let mind_dir = schema.dir.clone();
    let mind = Mind::new(mind_dir.clone());
    let mind_for_engine = mind.clone();
    match mind.root() {
        Some(root) => tracing::info!("mind: editable at {}", root.display()),
        None => tracing::info!("mind: none — the file editor will report it has nothing to edit"),
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

    let authored = Authored::new(
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

    // ── the engine ─────────────────────────────────────────────────────────
    //
    // The runtime is built first — it needs only the mind directory — then
    // handed to the state, then given the resolvers that read that state back.
    // That order breaks the cycle between them.
    let runtime = Runtime::new(mind_for_engine, &data);
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
            stopping: ctx.stopping.clone(),
        },
    );

    // The mind watcher, so an edited world file takes effect without a
    // restart. Held by the stop closure, for the process's lifetime — dropping
    // it stops the watch.
    let mind_watcher = match &mind_dir {
        Some(dir) => match watch_mind(
            dir,
            runtime.ledger.clone(),
            Arc::new(|report: ReloadReport| {
                tracing::info!(
                    "mind reloaded: +{} ~{} -{}",
                    report.added,
                    report.changed,
                    report.removed
                );
            }),
        ) {
            Ok(w) => Some(w),
            // Not fatal: the reload is a convenience, the engine is the
            // product.
            Err(e) => {
                tracing::warn!("mind watcher not armed: {e} — edits need a restart");
                None
            }
        },
        None => None,
    };

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
        stop: Box::new(move || Box::pin(stop(stopping, npcs, mind_watcher, guardian))),
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

/// Stop npcd, in the order that loses nothing.
///
/// 1. The mind watcher goes first, so an edit during the stop starts no
///    ingest.
/// 2. The flag is latched and the guardian stopped, then every character task
///    is cancelled and awaited out — not cosmetic: a character's decode writes
///    turns to the substrate, and a cast killed mid-tick leaves the redo log
///    with a turn whose sealing never happened. Every dropped turn future has
///    released its scheduler slot when this returns.
/// 3. The character store's lock is taken once, so a place or mood write a
///    character left in flight lands before the substrate is closed.
/// 4. The engine shuts down: its persistence drains, its writer flushes, the
///    substrate detaches — stopping the scheduler when npcd is the engine's
///    last attachment — and the writer lock is released.
async fn stop(
    runtime: Arc<Runtime>,
    npcs: Arc<RwLock<Npcs>>,
    mind_watcher: Option<RecommendedWatcher>,
    guardian: Option<JoinHandle<()>>,
) {
    drop(mind_watcher);
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

    /// **A named mind keeps its own substrate.** Without `--data`, the data
    /// directory is the mind's; `--data` still wins when it is given; and with
    /// neither there is nothing to choose, so the caller's default stands.
    #[test]
    fn the_data_directory_defaults_to_the_mind() {
        let mind = Path::new("D:/prog/mind");
        assert_eq!(data_dir(None, Some(mind)), Some(mind.to_path_buf()));
        assert_eq!(
            data_dir(Some(PathBuf::from("E:/elsewhere")), Some(mind)),
            Some(PathBuf::from("E:/elsewhere")),
            "an explicit --data wins"
        );
        assert_eq!(data_dir(None, None), None);
        assert!(parse(&["--mind", "D:/prog/mind"]).data.is_none());
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

    /// The flags and `tokerad.yaml`'s `npcd:` section read the same settings,
    /// with the same defaults; there is no `--content` and no embedded console.
    #[test]
    fn flags_and_yaml_agree() {
        assert_eq!(parse(&[]), NpcdArgs::default());
        assert_eq!(
            serde_yaml::from_str::<NpcdArgs>("{}").unwrap(),
            NpcdArgs::default()
        );
        let flags = parse(&["--mind", "../mind", "--log-identity"]);
        let yaml: NpcdArgs = serde_yaml::from_str("mind: ../mind\nlog_identity: true\n").unwrap();
        assert_eq!(flags, yaml);
        assert!(Cli::try_parse_from(["npcd", "--content", "web/content/npcd"]).is_err());
        assert!(serde_yaml::from_str::<NpcdArgs>("content: x").is_err());
    }

    /// Every one-shot flag is a real flag of the binary — a relaunch strips
    /// what the shell passed — and none can be set from `tokerad.yaml`.
    #[test]
    fn the_one_shot_flags_are_flags_and_never_yaml() {
        for o in NpcdArgs::ONE_SHOT {
            assert!(Cli::try_parse_from(["npcd", o.flag]).is_ok(), "{}", o.flag);
            let key = o.flag.trim_start_matches("--").replace('-', "_");
            assert!(
                serde_yaml::from_str::<NpcdArgs>(&format!("{key}: true")).is_err(),
                "{key} must not be settable from tokerad.yaml"
            );
        }
    }

    /// **A mind repository is mirrored under `--data`**, one or the other of
    /// a folder and a repository, never both, and a mirror not yet holding the
    /// record starts on no mind; `mind_repo:` reads the same from
    /// `tokerad.yaml`.
    #[test]
    fn a_mind_repository_is_mirrored_under_data() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().to_str().unwrap();
        let t = parse(&["--mind-repo", "mind", "--data", data])
            .resolve()
            .unwrap();
        let repo = t.repo.clone().unwrap();
        assert_eq!(repo.dir, dir.path().join("mind").join("mind"));
        assert_eq!(repo.branch.as_str(), "main");
        assert_eq!(t.schema.dir, None, "not mirrored yet: no mind");
        assert_eq!(t.data, dir.path());

        std::fs::create_dir_all(&repo.dir).unwrap();
        std::fs::write(repo.dir.join(projection::SCHEMA_FILE), projection::BUNDLED).unwrap();
        let ready = parse(&[
            "--mind-repo",
            "mind",
            "--data",
            data,
            "--mind-branch",
            "record",
        ])
        .resolve()
        .unwrap();
        assert_eq!(ready.schema.dir.as_deref(), Some(repo.dir.as_path()));
        assert_eq!(ready.repo.unwrap().branch.as_str(), "record");

        let both = parse(&["--mind-repo", "mind", "--data", data, "--mind", "../mind"]).resolve();
        assert!(both.is_err());
        let no_data = parse(&["--mind-repo", "mind"]).resolve();
        assert!(no_data
            .err()
            .unwrap()
            .to_string()
            .contains("--mind-repo keeps its mirror under --data"));
        let yaml: NpcdArgs =
            serde_yaml::from_str(&format!("mind_repo: mind\ndata: '{data}'\n")).unwrap();
        assert_eq!(yaml, parse(&["--mind-repo", "mind", "--data", data]));
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
