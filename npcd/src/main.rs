//! `npcd` binary — one executable that is the whole product.
//!
//! It serves the console and answers `/v1` from the same process. The HTTP
//! front is the [`web`] crate, given an API router as its local API for the
//! `npcd` site and the console files compiled in beside it; the DMZ deployment
//! runs the identical crate with `upstream:` URLs instead of `local`. Nothing
//! here knows which of the two it is part of.
//!
//! That router is three, merged and with no fallback under them:
//!
//! | | |
//! |---|---|
//! | [`api`] | the authored corpus, the mind, the cast, the authoring plane, accounts, portraits |
//! | [`ops`] | status, telemetry, memory, substrate storage, the log stream |
//! | [`engine`] | everything an inference engine would answer — wired, and honest that there is none |
//!
//! **Nothing is a fixture.** Every one of these routes either does its real job
//! or reports the absence: empty where empty is the measurement, `null` where
//! nothing has measured, and `503 no_engine` where the request asks for work.
//! A path none of them claims is a genuine `404`.
//!
//! It was not always. `main.rs` used to end in a `fallback_service` holding
//! `web::mock::npcd` — the console's own fixture — which answered every
//! unclaimed path with invented data, for any character id, including ones that
//! did not exist. That fixture is still built and still served by
//! `web --authoritative`, which is what it was written for.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use include_dir::{include_dir, Dir};
use tracing_subscriber::fmt::writer::MakeWriterExt;
use web::{Builder, Config, Roots};

mod accounts;
mod api;
mod clock;
mod collections;
mod console;
mod engine;
mod guard;
mod identity;
mod images;
mod lifegen;
mod logs;
mod mind;
mod model;
mod npcs;
mod ops;
mod projection;
mod registry;
mod substrate;
mod telemetry;
mod visibility;

/// The console, compiled in. Two directories, searched in order: a request for
/// `/lib/dom.js` falls through to the shared framework, `/pages/roster.js` does
/// not. Same layering as the on-disk roots the proxy uses.
static SITE: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../web/content/npcd");
static COMMON: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../web/content/common");

#[derive(Parser, Debug)]
#[command(
    name = "npcd",
    about = "NPC engine daemon — authored content, the cast and the console; no engine yet"
)]
struct Cli {
    /// Address to bind. Loopback by default — see the auth note in
    /// `docs/npc_api_gui_design.md` §8 before binding anything wider.
    #[arg(long)]
    bind: Option<SocketAddr>,

    /// Serve the console from this directory instead of the compiled-in copy,
    /// so an edit is a refresh rather than a rebuild. Point it at
    /// `web/content/npcd`; the shared root is added after it automatically.
    #[arg(long)]
    content: Option<PathBuf>,

    /// Where the engine's own state lives — `.substrate/` and `accounts/`, the
    /// things the daemon writes rather than a person. Also the fallback source
    /// for authored content (`worlds/`, `personalities/`) when no `--mind` is
    /// named. Defaults to the `npcd` directory in the source tree.
    #[arg(long)]
    data: Option<PathBuf>,

    /// Load the projection schema and its content libraries from this
    /// directory instead of the compiled-in default.
    ///
    /// The directory must hold a `projection.yaml`; the folders beside it
    /// (`responses/`, `moods/`, `personalities/`, `worlds/`, …) are what its
    /// folder-backed collections and its authored registries read from. Schema
    /// and libraries move together — see
    /// `crate::projection`. `zend` spells the same idea `--working-dir`, which
    /// for it also relocates the substrate; here `--data` already does that, so
    /// this flag moves only the mind.
    #[arg(long, value_name = "DIR")]
    mind: Option<PathBuf>,

    /// Increase log verbosity (-v debug, -vv trace).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Log the identity the gateway put on each request, and the role it
    /// resolved to.
    ///
    /// Sign-in crosses two processes on two machines, and when it fails both
    /// sides look correct in isolation: the gateway holds a session, the daemon
    /// answers `401`, and neither says what arrived on the wire between them.
    /// This prints exactly that, per request.
    ///
    /// Header **values are never logged** — the subject is a durable account
    /// identifier and the assertion is a bearer token, so both would be a
    /// credential sitting in a log file. Each header reports only `set`,
    /// `EMPTY` or absent, which is the whole of what a routing question needs.
    ///
    /// Off by default and noisy when on: every request, including the
    /// console's polls.
    #[arg(long)]
    log_identity: bool,
}

/// How many routes across both tables sit at exactly this role, for the
/// startup line.
fn count<A, B>(a: &guard::Api<A>, b: &guard::Api<B>, min: web::auth::Role) -> usize {
    a.declared()
        .iter()
        .chain(b.declared())
        .filter(|r| r.min == min)
        .count()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let level = match cli.verbose {
        0 => "npcd=info,web=info",
        1 => "npcd=debug,web=debug",
        _ => "npcd=trace,web=trace",
    };
    // Every line goes two places: the terminal, and the bus the console reads
    // from `/ws/logs`. One formatter feeds both, so what an operator sees on
    // screen and what the console shows cannot drift — a second formatter is
    // how those two end up disagreeing about what the daemon said.
    let logs = logs::LogBus::new();
    let bus = logs.clone();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into()),
        )
        // Targets on. They cost a prefix in the terminal and buy a column the
        // console can filter by — and without them a message containing a colon
        // is indistinguishable from `target: message`, so the viewer would
        // file `world: 2 loaded` under a target called `world`.
        .with_target(true)
        .with_writer(std::io::stderr.and(move || logs::BusWriter::new(bus.clone())))
        .init();

    // The projection schema, resolved before any I/O.
    //
    // First because a mistyped `--mind` should be refused on the spot, not
    // after the substrate has been opened and half the daemon stood up — the
    // error is about a command-line argument and nothing that happens in
    // between can change the answer.
    //
    // Fatal rather than a fallback: a daemon that quietly reverted to the
    // bundled placeholder would run with none of the named mind's content and
    // give no sign of it, until characters behaved as though their libraries
    // were empty. Which they would be.
    let schema = projection::Source::resolve(cli.mind.as_deref())?;
    match &schema.dir {
        Some(dir) => tracing::info!(
            "projection schema: {} (collections resolve under {})",
            schema.label,
            dir.display()
        ),
        None => tracing::info!(
            "projection schema: {} — placeholder, no layers and no content libraries",
            schema.label
        ),
    }

    let mut cfg = Config::from_yaml(
        include_str!("../npcd.web.yaml"),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
    )?;
    if let Some(bind) = cli.bind {
        cfg.server.bind = bind;
    }

    let roots = match &cli.content {
        Some(dir) => {
            let site = dir.canonicalize()?;
            // `--content web/content/npcd` implies `web/content/common`: the
            // two are one tree, and asking for both on the command line would
            // only be a chance to give a mismatched pair.
            let common = site
                .parent()
                .map(|p| p.join("common"))
                .filter(|p| p.is_dir())
                .unwrap_or_else(|| site.clone());
            Roots::disk(&[site, common])
        }
        None => Roots::embedded(&[&SITE, &COMMON]),
    };

    let data = cli
        .data
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    // Authored content, read once. Everything after this answers from memory,
    // so a URL id is a key rather than a path — see `registry`.
    //
    // It comes from the MIND when one is named, and from the data directory
    // otherwise — the same override the schema itself uses. Worlds and
    // personalities are written by a person and belong beside the corpus they
    // index: a world names which canon it admits, and a personality is what a
    // character is before it has lived anything. The data directory holds what
    // the ENGINE wrote — the substrate and accounts — and reading a world from
    // there would put the two on the wrong sides of that line.
    let authored_dir = schema.dir.clone().unwrap_or_else(|| data.clone());
    let worlds = registry::Registry::load("world", authored_dir.join("worlds"))?;
    let personalities =
        registry::Registry::load("personality", authored_dir.join("personalities"))?;
    tracing::info!(
        "authored content: {} worlds, {} personalities from {}",
        worlds.len(),
        personalities.len(),
        authored_dir.display()
    );

    // Accounts are durable but never published — `npcd/.gitignore` keeps the
    // directory out of a public repository, because a record here carries a
    // real email address and a provider subject id.
    let accounts = accounts::Accounts::load(data.join("accounts"))?;

    tracing::info!(
        "accounts: {} known, from {}",
        accounts.len(),
        data.join("accounts").display()
    );
    // Named the other way round, because the exception list is now the longer
    // one and a reader takes "MOCK for everything except …" as exhaustive. It
    // was not: it omitted the cast on the substrate, the whole mind editor,
    // `/v1/substrate/storage` and `/v1/world/:wid/collections`, all of which
    // are real.
    tracing::info!(
        "backend: real for authored content, the mind, the cast, accounts, telemetry, \
         storage and logs — FIXTURE for everything an engine would produce"
    );

    // The real routes sit *over* the mock rather than beside it: `npcd` owns
    // `/v1/world*`, `/v1/personality*`, `/v1/me*`, `/v1/telemetry` and
    // `/ws/logs`, and anything it does not answer falls through. Merging with
    // the mock would panic on the overlap; layering means each surface can
    // become real one route at a time without the console noticing.
    //
    // The two real routers *are* merged with each other — their paths are
    // disjoint and their state is unrelated, so keeping them apart is what lets
    // each hold exactly what it needs.
    // The cast, rebuilt from the substrate's redo log. The one read of that log
    // — every request after this is answered from memory.
    let npcs = npcs::Npcs::load(&data)
        .map_err(|e| anyhow::anyhow!("opening the substrate at {}: {e:?}", data.display()))?;

    // Who may change what. Configuration, decided once at startup — there is
    // deliberately no API that grants it, so it cannot drift from the file.
    let roles = cfg.roles.clone();
    if roles.is_empty() {
        // Not fatal: a read-only daemon is a legitimate thing to run. But it is
        // the difference between "the console has no Save" and "the console's
        // Save is broken", and an operator should learn it here rather than
        // from a 403 an hour later.
        tracing::warn!(
            "roles: no admins configured — worlds and personalities are read-only to everyone"
        );
    } else {
        tracing::info!(
            "roles: {} admin principal(s) configured",
            roles.admins.len()
        );
    }

    // The response and mood libraries, beside the schema. Read once: they are
    // ingested untagged and shared by every world, so there is one copy and it
    // does not change while the daemon runs.
    let libraries = collections::Libraries::load(&schema);
    tracing::info!(
        "libraries: {} responses ({} with examples), {} moods ({} with examples)",
        libraries.responses.len(),
        libraries.responses.with_examples(),
        libraries.moods.len(),
        libraries.moods.with_examples(),
    );

    // The mind directory, for the file editor. Taken from the resolved schema
    // rather than from `--mind` directly, so the editor and the collections
    // read the same root — a second answer to "where is the mind" would be
    // free to disagree the day the layout changes.
    let mind_dir = schema.dir.clone();
    let mind = mind::Mind::new(mind_dir.clone());
    // A second handle for the engine. The editor and the ingest read the same
    // schema through the same address, so they must be the same `Mind` — a
    // second one built from the path would be a second reading of the document
    // to drift from the first.
    let mind_for_engine = mind.clone();
    match mind.root() {
        Some(root) => tracing::info!("mind: editable at {}", root.display()),
        None => tracing::info!("mind: none — the file editor will report it has nothing to edit"),
    }

    let authored = api::Authored::new(
        worlds,
        personalities,
        accounts,
        npcs,
        roles.clone(),
        libraries,
        mind,
        // Portraits, beside the accounts and the substrate — things this daemon
        // writes, rather than things a person authored.
        images::Images::new(&data),
    );

    // ── the engine ─────────────────────────────────────────────────────────
    //
    // Built here and *started* after the server is bound, so the console's
    // loading screen is being served by the time the model starts loading. A
    // daemon that loads first and binds afterwards has nothing to show the
    // person waiting on it, which on a multi-gigabyte checkpoint is the whole
    // of the first minute.
    //
    // The runtime is built first — it needs only the mind directory — then
    // handed to the state, then given the two resolvers that read that state
    // back. That order is what breaks the cycle between them: `Authored` needs
    // the runtime to answer its routes, and the runtime needs `Authored` to know
    // what time it is for a character and who that character is.
    let runtime = engine::runtime::Runtime::new(mind_for_engine, &data);
    let authored = authored.with_runtime(runtime.clone());

    // The clock resolver closes over the state rather than reading a single
    // daemon-wide clock: worlds run at their own pace and can be paused
    // independently, and a character has to see its own world's time.
    // `blocking_read`, because the tick driver is a plain OS thread with no
    // async context. **This closure must never be called from inside the
    // runtime** — `blocking_read` there panics rather than waiting, and the
    // route that did it answered every request with a closed connection while
    // the loop it was showing ran perfectly underneath. The async side uses
    // `Authored::world_ms`, which reaches the same lookup by the other route.
    let clock_state = authored.clone();
    runtime.set_clock(Arc::new(move |npc_id: u64| {
        clock::world_ms_for(
            &clock_state.npcs.blocking_read(),
            &clock_state.worlds.blocking_read(),
            npc_id,
            now_ms_i64(),
        )
    }));

    // Who each character is, resolved at tick time rather than captured at
    // startup. A belief edited through the authoring API has to reach the next
    // tick — a persona snapshot taken here would keep every character as it was
    // when the process began, and the mind editor would appear to do nothing.
    let persona_state = authored.clone();
    runtime.set_persona_source(Arc::new(move |npc_id: u64| {
        let npcs = persona_state.npcs.blocking_read();
        let payload = npcs.payload(npc_id)?;
        // The world's own description, for the prompt's setting section. A
        // character with no readable world still gets a persona — an empty
        // setting is a thin prompt, a missing character is no prompt at all.
        let world = persona_state
            .worlds
            .blocking_read()
            .get(&payload.world_id)
            .and_then(|r| {
                r.body
                    .get("description")
                    .and_then(|d| d.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        Some(engine::persona::of(payload, &world))
    }));

    // The load progress is shared, not copied: `/v1/status` answers from it
    // while the loader thread is still writing to it.
    let ops_state = ops::Ops::new(logs, &data, roles.clone(), runtime.progress.clone());

    // The route table, at startup, with the role each route needs.
    //
    // Printed rather than assumed: `guard::Api` makes it impossible to register
    // a route *without* a role, and this makes the resulting table something an
    // operator can read on a line instead of inferring from source. A surprise
    // here — a write route sitting at `unauthenticated` — is the one that
    // matters, and it is the one that used to be invisible.
    let (api_routes, ops_routes) = (api::api(authored.clone()), ops::api(ops_state.clone()));
    // The surface an engine would answer — wired, and honest that there is no
    // engine behind it. Separate from `api` because the two become true at
    // different times: everything in `api` is real today.
    let engine_routes = engine::api(authored.clone());
    for r in api_routes
        .declared()
        .iter()
        .chain(ops_routes.declared())
        .chain(engine_routes.declared())
    {
        tracing::debug!("route {r}");
    }
    tracing::info!(
        "routes: {} guarded ({} open, {} user, {} admin), {} awaiting an engine",
        api_routes.declared().len() + ops_routes.declared().len() + engine_routes.declared().len(),
        count(&api_routes, &ops_routes, web::auth::Role::Unauthenticated),
        count(&api_routes, &ops_routes, web::auth::Role::User),
        count(&api_routes, &ops_routes, web::auth::Role::Admin),
        engine_routes.declared().len(),
    );

    // Shared with the logging layer below, which needs the table on every
    // request while the fallback owns it. `Arc` so that per-request sharing is
    // a refcount bump rather than a copy of the admin list, and `None` when the
    // flag is off so nothing is paid for a layer that is not installed.
    let log_roles = cli.log_identity.then(|| Arc::new(roles.clone()));

    let mut router = api_routes
        .into_router(authored.clone())
        .merge(ops_routes.into_router(ops_state))
        .merge(engine_routes.into_router(authored.clone()));

    /* **There is no fallback.**
     *
     * There was, and it was `web::mock::npcd::router()` — the console's
     * fixture, answering every path the real routes had not claimed. It
     * answered them well: correctly-shaped, plausible, and invented. A
     * character's beliefs came back for ids that did not exist; the world clock
     * reported `{"ok":true}` and moved nothing; the tool catalog listed tools
     * this daemon has never had.
     *
     * Every one of those paths is now a route of its own — real where the thing
     * is real, and `no_engine` where it is not (see [`engine`]). So an
     * unclaimed path is a genuine 404 again, which is what a 404 is for.
     *
     * The fixture still exists and is still built: `web --authoritative` serves
     * it, which is what it was written for. What it no longer does is stand
     * behind a daemon that means it. */

    // Applied to the assembled router, so it covers the guarded routes, the ops
    // routes and the fallback alike — a layer on any one of them would report
    // only what reached that one. It does **not** see static content: the
    // console's files are served by the builder and never enter this router, so
    // the log is `/v1` traffic, which is the whole of what carries identity.
    if let Some(log_roles) = log_roles {
        tracing::info!("--log-identity: logging the gateway's headers on every API request");
        router = router.layer(axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| {
                let roles = Arc::clone(&log_roles);
                async move {
                    // Borrowed straight from the request rather than copied out
                    // of it: three shared borrows that all end before `req` is
                    // moved into `next`, so the line costs no allocation.
                    identity::log_identity(
                        req.headers(),
                        &roles,
                        req.method().as_str(),
                        req.uri().path(),
                    );
                    next.run(req).await
                }
            },
        ));
    }

    // ── start the engine ───────────────────────────────────────────────────
    //
    // After the router is assembled and before `serve()` blocks. The loader
    // runs on its own OS thread and returns immediately, so the server binds
    // while the model is still being fetched — which is the point: the loading
    // screen has to be reachable during the load it is describing.
    let cast: Vec<u64> = authored
        .npcs
        .read()
        .await
        .cast()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    tracing::info!("engine: loading, {} character(s) to wake", cast.len());
    engine::runtime::start(
        runtime.clone(),
        engine::runtime::LoadPlan {
            world_ms: 0,
            cast,
            // Personalities, not the cast. A layer directory is named after a
            // personality — `layers/memory/zen/` — and a world's biographies
            // exist whether or not anybody has cast them yet.
            characters: authored
                .personalities
                .read()
                .await
                .iter()
                .map(|r| r.id.clone())
                .collect(),
        },
    );

    // The mind watcher, so an edited world file takes effect without a restart.
    // Held for the process's lifetime — dropping it stops the watch — which is
    // why it is bound rather than discarded.
    let _mind_watcher = match &mind_dir {
        Some(dir) => match engine::watcher::spawn(
            dir,
            // The runtime's ledger, filled by the startup ingest — so the first
            // edit after a boot is one changed file, not a whole-tree rewrite.
            runtime.ledger.clone(),
            Arc::new(|report: engine::watcher::ReloadReport| {
                tracing::info!(
                    "mind reloaded: +{} ~{} -{}",
                    report.added,
                    report.changed,
                    report.removed
                );
            }),
        ) {
            Ok(w) => Some(w),
            // Not fatal. A daemon that will not start because it cannot watch a
            // directory is worse than one that runs without live reload and
            // says so — the reload is a convenience, the engine is the product.
            Err(e) => {
                tracing::warn!("mind watcher not armed: {e} — edits need a restart");
                None
            }
        },
        None => None,
    };

    // Stop the tick driver on Ctrl-C, before the process goes.
    //
    // Not cosmetic. A character's decode writes turns to the substrate, and a
    // driver killed mid-tick leaves the redo log with a turn whose sealing never
    // happened. Setting the flag lets the current tick finish and the loop exit
    // at its next quiet moment — the same reason the flag exists at all.
    let stopping = runtime.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::info!("interrupt received — stopping the tick driver");
            stopping.stop();
        }
    });

    Builder::new(cfg)
        .content("npcd", roots)
        // Sign-in is the gateway's, and it names the caller on `X-Tokera-*`.
        // Believing those headers is only sound because the bind address is
        // reachable through the gateway and nothing else — see the flag's docs
        // before pointing `--bind` at a public interface.
        .behind_gateway()
        .local_api("npcd", router)
        .serve()
        .await
}

/// Wall-clock milliseconds as the narrative clock takes them.
///
/// Signed, because [`clock::Clock`] works in `i64` throughout — a world can be
/// jumped backwards past its own epoch, and an unsigned instant would wrap
/// rather than clamp.
fn now_ms_i64() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
