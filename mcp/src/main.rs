//! `mcp` — the MCP server Claude reaches at `code.tokera.com/mcp`
//! (`docs/mcp_app.md`): an aggregator over the MCP servers zend and npcd each
//! run at its own path, reached back through the gateway as the signed-in
//! caller. Every machine that runs one joins the gateway as a member of the
//! app `mcp`, and the gateway deals `/mcp` to them in turn.
//!
//! | Module | What |
//! |---|---|
//! | [`app`] | The aggregated server: one card, every backend's tools, each call routed |
//! | [`backend`] | One backend, called through the gateway as the caller |
//! | [`sse`] | Reading a backend's event stream |
//! | [`believed`] | Whose identity headers are kept: the socket's, and a listed gateway's |
//! | [`caller`] | Who is calling, from the believed headers |
//! | [`presence`] | The cluster socket: joining as `mcp`, and serving its streams |
//! | [`guides`] | The guide library: the methods for using the tools, served as tools |
//! | [`config`] | Each machine's file, and the `served` file they share |

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use axum::middleware::from_fn_with_state;
use axum::routing::post;
use axum::Router;
use clap::Parser;
use cluster::upgrade::switch::PROOF;
use mcp_server::http;
use mcp_server::SealedStore;
use tokio::net::TcpListener;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use web::log_file::RotatingFileLog;

mod app;
mod backend;
mod believed;
mod caller;
mod config;
mod guides;
mod last_listed;
mod presence;
mod sse;

use app::App;
use backend::{Backend, Link};
use config::Config;
use last_listed::LastListed;

#[derive(Parser)]
struct Args {
    /// This machine's configuration file.
    #[arg(long, default_value = "mcp/mcp.yaml")]
    config: PathBuf,
}

/// The log's name in the state folder.
const LOG_NAME: &str = "mcp.log";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    // Stderr, which a trial's file captures, and `mcp.log` in the state
    // folder: keep starts a kept service with its output detached, and an
    // aggregator that logs only there keeps no record of a call at all. An
    // unopenable file leaves stderr alone.
    let filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let to_file = RotatingFileLog::new(&config.state, LOG_NAME).map(|w| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(w)
            .with_filter(filter())
    });
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(io::stderr)
                .with_filter(filter()),
        )
        .with(to_file)
        .init();
    let app = Arc::new(App {
        identity: config.identity(),
        store: Arc::new(SealedStore),
        backends: config
            .backends
            .iter()
            .map(|b| Backend {
                name: b.name.clone(),
                host: b.host.clone(),
                path: b.path.clone(),
            })
            .collect(),
        link: Link::new(&config.gateway)?,
        routes: RwLock::new(HashMap::new()),
        last_listed: LastListed::default(),
        library: guides::embedded()?,
    });
    tracing::info!(guides = app.library.count(), "guides loaded");
    let gateways: Arc<[IpAddr]> = config.trusted_gateways.clone().into();
    let router = Router::new()
        .route(
            presence::PATH,
            post(http::post::<App>)
                .get(http::get)
                .delete(http::delete::<App>),
        )
        .with_state(app)
        .layer(from_fn_with_state(gateways, believed::layer));
    let listener = TcpListener::bind(&config.bind).await?;
    presence::join(&config, router.clone())?;
    tracing::info!(
        bind = %config.bind,
        gateway = %config.gateway,
        backends = ?config.backends.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(),
        "serving"
    );
    // The listener is bound and the socket dialling, and this is the call
    // that accepts. The line is the whole of a stderr line: `keep` treats the
    // process as serving only once it has read it.
    eprintln!("{PROOF}");
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}
