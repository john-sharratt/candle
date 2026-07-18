//! `zend` — Zen Code daemon.
//!
//! Run from the root of your workspace:
//!
//! ```text
//! zend                        # workspace = cwd, port 8080
//! zend /path/to/project       # explicit workspace path
//! zend --port 9090            # custom port
//! ```
//!
//! Continue configuration:
//! ```json
//! { "provider": "openai", "apiBase": "http://localhost:8080", "model": "zen-code" }
//! ```

mod api;
mod chatml;
mod code_read;
mod config;
mod conv_file_store;
mod conv_files;
mod download;
mod loading;
mod log_broadcast;
mod log_file;
mod log_line;
mod projection_event;
mod refresh_ctx;
mod repo_scan;
mod session;
mod tool_summary;
mod tools;
mod turn_sink;
mod types;
mod watcher;

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

use config::DaemonConfig;
use log_broadcast::{BusWriter, LogBus};
use session::ZendSession;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "zend",
    about = "Zen Code daemon — persistent AI coding assistant",
    long_about = None,
)]
struct Cli {
    /// Root of the workspace to analyse.  Defaults to the current directory.
    #[arg(default_value = ".")]
    workspace: PathBuf,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Increase log verbosity.  -v = DEBUG, -vv = TRACE.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Skip the startup code-reading ingest pass. Brings the daemon up fast
    /// (model + substrate + sections only) so you can test conversations
    /// without the per-file prefill sweep.
    #[arg(long)]
    skip_code_read: bool,

    /// Skip the startup repository scan. Brings the daemon up fast
    /// (model + substrate + sections only) so you can test conversations
    #[arg(long)]
    skip_repo_scan: bool,

    /// Do not run the background summariser thread (no AVL summary-forest
    /// extension, no per-conversation summarisation registration). Brings the
    /// engine up without the summariser — useful for bulk corpus prefill.
    #[arg(long)]
    disable_summariser: bool,

    /// Force a whole-store redo-log compaction once during load (after the
    /// substrate reload, before serving) instead of leaving reclaim to the
    /// incremental background maintenance pass. Physically rewrites the log,
    /// shedding tombstoned/distilled records and dead chunks, then
    /// re-reconstructs the substrate. Opt-in; the startup pays the rewrite cost.
    #[arg(long)]
    compact_substrate: bool,

    /// Address to bind the HTTP server to. Defaults to loopback only
    /// (`127.0.0.1`) — reachable from this machine alone. Pass `0.0.0.0` to
    /// listen on all IPv4 interfaces (LAN / VPN reachable). WARNING: the daemon
    /// is UNAUTHENTICATED, so only expose it on a trusted network.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Panic hook ────────────────────────────────────────────────────────────
    //
    // Force-captures a full backtrace on every panic (no RUST_BACKTRACE needed)
    // and includes the last CUDA kernel name that was launched on the crashing
    // thread.  The kernel name is written into a thread-local breadcrumb before
    // every unsafe kernel FFI call, so it is available here even though CUDA
    // errors surface asynchronously at a later DtoH / synchronize point.
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        // zend always links candle with the "cuda" feature — call unconditionally.
        let kernel = candle::last_cuda_kernel_launch();
        eprintln!("\n=== PANIC ===\n{info}\nLast CUDA kernel: {kernel}\n\n{bt}\n=============\n");
    }));

    // ── CLI (parsed first so we know verbosity before init) ──────────────────

    let cli = Cli::parse();

    let workspace = cli
        .workspace
        .canonicalize()
        .unwrap_or_else(|_| cli.workspace.clone());

    // ── Logging ───────────────────────────────────────────────────────────────
    //
    // Three fmt layers sharing the same filter:
    //  • stdout    — ANSI colours for the terminal
    //  • broadcast — plain text piped to the web log pane via WebSocket
    //  • file      — the full configured stream to <workspace>/.substrate/zend.log,
    //                fresh per run, size-capped with rotation (see `log_file`).

    let log = LogBus::new();

    let level = match cli.verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::from_default_env()
        .add_directive(format!("zend={level}").parse()?)
        .add_directive(format!("candle_conversation={level}").parse()?);

    let stdout_layer = tracing_subscriber::fmt::layer().with_filter(filter.clone());

    let ws_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(BusWriter(Arc::clone(&log)))
        .with_filter(filter.clone());

    // None (open failure) degrades to the stdout + bus sinks rather than
    // aborting boot; `Option<Layer>` is itself a `Layer` (None = no-op).
    let file_layer = log_file::RotatingFileLog::new(&workspace.join(".substrate")).map(|w| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(w)
            .with_filter(filter)
    });

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(ws_layer)
        .with(file_layer)
        .init();

    let config = DaemonConfig {
        workspace: workspace.clone(),
        port: cli.port,
        skip_code_read: cli.skip_code_read,
        skip_repo_scan: cli.skip_repo_scan,
        disable_summariser: cli.disable_summariser,
        compact_substrate: cli.compact_substrate,
    };

    if cli.skip_code_read {
        tracing::info!("--skip-code-read: startup code-reading ingest is disabled");
    }
    if cli.skip_repo_scan {
        tracing::info!("--skip-repo-scan: startup repository scan is disabled");
    }
    if cli.disable_summariser {
        tracing::info!("--disable-summariser: background summariser thread is disabled");
    }
    if cli.compact_substrate {
        tracing::info!("--compact-substrate: forcing a whole-store redo-log compaction on load");
    }

    tracing::info!(workspace = %workspace.display(), port = cli.port, "starting zend");

    // ── Session + router ──────────────────────────────────────────────────────

    let session = Arc::new(ZendSession::new(config.clone(), Arc::clone(&log)));
    session.start_loading();
    let router = api::router(Arc::clone(&session));

    // ── Bind ──────────────────────────────────────────────────────────────────

    let bind_ip: IpAddr = match cli.host.as_str() {
        "localhost" => IpAddr::from([127, 0, 0, 1]),
        h => h.parse().map_err(|_| {
            anyhow::anyhow!(
                "invalid --host {h:?}: expected an IP address, e.g. 127.0.0.1 or 0.0.0.0"
            )
        })?,
    };
    let addr = SocketAddr::new(bind_ip, config.port);
    if !bind_ip.is_loopback() {
        tracing::warn!(
            %addr,
            "binding to a non-loopback address — the UNAUTHENTICATED inference daemon \
             is now reachable from the network; only do this on a trusted network",
        );
    }
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!(
        addr = %addr,
        "ready — API: http://{addr}/v1/chat/completions \
               — web: http://{addr}/",
    );

    // ── Background: workspace scan ────────────────────────────────────────────

    tokio::spawn(async move {
        tracing::info!("scanning workspace...");
        tokio::task::spawn_blocking(move || scan_workspace(&workspace))
            .await
            .ok();
        tracing::info!("scan complete");
    });

    // ── Serve, with graceful shutdown ─────────────────────────────────────────
    //
    // On Ctrl-C / SIGTERM the server stops accepting connections and drains
    // in-flight requests; then the substrate redo log is checkpointed so the
    // last turn — including a partial in-flight tail — is durable on disk.
    let shutdown_session = Arc::clone(&session);
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("draining complete — flushing substrate…");
    shutdown_session.shutdown().await;
    tracing::info!("zend stopped");
    Ok(())
}

// ── Shutdown signal ───────────────────────────────────────────────────────────

/// Resolves when the process receives `Ctrl-C` (or `SIGTERM` on Unix),
/// triggering axum's graceful drain. A *second* `Ctrl-C` aborts immediately
/// — the escape hatch if draining wedges.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::warn!(
        "shutdown signal received — draining in-flight work; press Ctrl-C again to abort",
    );

    tokio::spawn(async {
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::error!("second Ctrl-C — aborting immediately");
            std::process::exit(130);
        }
    });
}

// ── Workspace scan ────────────────────────────────────────────────────────────

fn scan_workspace(root: &std::path::Path) {
    let file_count = std::fs::read_dir(root)
        .map(|entries| entries.flatten().filter(|e| e.path().is_file()).count())
        .unwrap_or(0);

    tracing::info!(
        root = %root.display(),
        top_level_files = file_count,
        "workspace scan placeholder",
    );
}
