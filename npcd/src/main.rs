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

    tracing::info!("backend: MOCK (no engine, no GPU)");
    Builder::new(cfg)
        .content("npcd", roots)
        .local_api("npcd", web::mock::npcd::router())
        .serve()
        .await
}
