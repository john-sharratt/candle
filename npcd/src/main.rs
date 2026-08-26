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
use web::{Builder, Config, Roots};

mod accounts;
mod api;
mod identity;
mod registry;

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

    /// Where authored content lives — `worlds/` and `archetypes/`, read once at
    /// start and written back when the GUI saves. Defaults to the `npcd`
    /// directory in the source tree, which is what makes a world edit a commit.
    #[arg(long)]
    data: Option<PathBuf>,

    /// The estate's shared session signing key — the same file the gateway
    /// signs with, which is what makes one sign-in carry across tokera.com,
    /// code. and bot. Defaults to `secrets/session.key` under `--data`.
    ///
    /// Without it this daemon authenticates nobody. That is deliberate: it
    /// binds a LAN address, so treating an unconfigured key as "trust whatever
    /// arrives" would turn a missing file into an open door.
    #[arg(long)]
    session_key: Option<PathBuf>,

    /// Increase log verbosity (-v debug, -vv trace).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let level = match cli.verbose {
        0 => "npcd=info,web=info",
        1 => "npcd=debug,web=debug",
        _ => "npcd=trace,web=trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into()),
        )
        .with_target(false)
        .init();

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

    // Authored content, read once. Everything after this answers from memory,
    // so a URL id is a key rather than a path — see `registry`.
    let data = cli
        .data
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let worlds = registry::Registry::load("world", data.join("worlds"))?;
    let archetypes = registry::Registry::load("archetype", data.join("archetypes"))?;
    tracing::info!(
        "authored content: {} worlds, {} archetypes from {}",
        worlds.len(),
        archetypes.len(),
        data.display()
    );

    // Accounts are durable but never published — `npcd/.gitignore` keeps the
    // directory out of a public repository, because a record here carries a
    // real email address and a provider subject id.
    let accounts = accounts::Accounts::load(data.join("accounts"))?;

    let key_path = cli
        .session_key
        .unwrap_or_else(|| data.join("secrets").join("session.key"));
    let verifier = if key_path.exists() {
        let v = identity::Verifier::from_file(&key_path)?;
        tracing::info!("sign-in: verifying assertions with {}", key_path.display());
        v
    } else {
        tracing::warn!(
            "sign-in: no key at {} — NOBODY will be authenticated. Copy the \
             gateway's session key there to enable sign-in.",
            key_path.display()
        );
        identity::Verifier::unconfigured()
    };

    tracing::info!(
        "accounts: {} known, from {}",
        accounts.len(),
        data.join("accounts").display()
    );
    tracing::info!("backend: MOCK for everything except authored content and accounts");

    // The real routes sit *over* the mock rather than beside it: `npcd` owns
    // `/v1/world*`, `/v1/archetype*` and `/v1/me*`, and anything it does not
    // answer falls through. Merging the two would panic on the overlap;
    // layering means each surface can become real one route at a time without
    // the console noticing.
    let authored = api::Authored::new(worlds, archetypes, accounts, verifier);
    let router = api::router(authored).fallback_service(web::mock::npcd::router());

    Builder::new(cfg)
        .content("npcd", roots)
        .local_api("npcd", router)
        .serve()
        .await
}
