//! `npcd` binary — one executable that is the whole product.
//!
//! It serves the console and answers `/v1` from the same process. The HTTP
//! front is the [`web`] crate, given an API router as its local API for the
//! `npcd` site and the console files compiled in beside it; the DMZ deployment
//! runs the identical crate with `upstream:` URLs instead of `local`. Nothing
//! here knows which of the two it is part of.
//!
//! That router is currently [`web::mock::npcd`] — the console's own fixture,
//! which is the entire daemon until there is an engine to put behind it. When
//! there is, this line names `npcd::api::router()` instead and nothing else in
//! the file changes.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use include_dir::{include_dir, Dir};
use tracing_subscriber::fmt::writer::MakeWriterExt;
use web::{Builder, Config, Roots};

mod accounts;
mod api;
mod collections;
mod guard;
mod identity;
mod logs;
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
#[command(name = "npcd", about = "NPC engine daemon (mock API + console)")]
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
    tracing::info!(
        "backend: MOCK for everything except authored content, accounts, telemetry and logs"
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

    let authored = api::Authored::new(
        worlds,
        personalities,
        accounts,
        npcs,
        roles.clone(),
        libraries,
    );
    let ops_state = ops::Ops::new(logs, &data, roles.clone());

    // The route table, at startup, with the role each route needs.
    //
    // Printed rather than assumed: `guard::Api` makes it impossible to register
    // a route *without* a role, and this makes the resulting table something an
    // operator can read on a line instead of inferring from source. A surprise
    // here — a write route sitting at `unauthenticated` — is the one that
    // matters, and it is the one that used to be invisible.
    let (api_routes, ops_routes) = (api::api(authored.clone()), ops::api(ops_state.clone()));
    for r in api_routes.declared().iter().chain(ops_routes.declared()) {
        tracing::debug!("route {r}");
    }
    tracing::info!(
        "routes: {} guarded ({} open, {} user, {} admin), everything else behind `user`",
        api_routes.declared().len() + ops_routes.declared().len(),
        count(&api_routes, &ops_routes, web::auth::Role::Unauthenticated),
        count(&api_routes, &ops_routes, web::auth::Role::User),
        count(&api_routes, &ops_routes, web::auth::Role::Admin),
    );

    let router = api_routes
        .into_router(authored)
        .merge(ops_routes.into_router(ops_state))
        // The fallback is the quietest surface there is: it answers every path
        // the real routes did not claim, which today is the console's fixture
        // and tomorrow is whatever has not been migrated yet. Signed-in is the
        // floor — the console is a signed-in tool — so a route that has not
        // been written yet cannot be reached by a stranger before it is.
        .fallback_service(guard::behind(
            roles,
            web::auth::Role::User,
            web::mock::npcd::router(),
        ));

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
