//! `mcp` — the MCP server Claude reaches at `code.tokera.com/mcp`
//! (`docs/mcp_app.md`): an aggregator over the MCP servers zend and npcd each
//! run at its own path, reached back through the gateway as the signed-in
//! caller.
//!
//! | Module | What |
//! |---|---|
//! | [`app`] | The aggregated server: one card, every backend's tools, each call routed |
//! | [`backend`] | One backend, called through the gateway as the caller |
//! | [`sse`] | Reading a backend's event stream |
//! | [`caller`] | Who is calling, believed from the gateway |
//! | [`guides`] | The guide library: the methods for using the tools, served as tools |
//! | [`config`] | `mcp/mcp.yaml` |

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use axum::routing::post;
use axum::Router;
use clap::Parser;
use mcp_server::http;
use mcp_server::LocalStore;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

mod app;
mod backend;
mod caller;
mod config;
mod guides;
mod sse;

use app::App;
use backend::{Backend, Link};
use config::Config;

#[derive(Parser)]
struct Args {
    /// The configuration file.
    #[arg(long, default_value = "mcp/mcp.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(io::stderr)
        .init();
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let app = Arc::new(App {
        identity: config.identity(),
        gateways: config.trusted_gateways.clone(),
        store: Arc::new(LocalStore::in_dir(&config.state)?),
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
        library: guides::embedded()?,
    });
    tracing::info!(guides = app.library.count(), "guides loaded");
    let router = Router::new()
        .route(
            "/mcp",
            post(http::post::<App>)
                .get(http::get)
                .delete(http::delete::<App>),
        )
        .with_state(app);
    let listener = TcpListener::bind(&config.bind).await?;
    tracing::info!(
        bind = %config.bind,
        gateway = %config.gateway,
        backends = ?config.backends.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(),
        "serving"
    );
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}
