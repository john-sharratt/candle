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
mod ingest;
mod loading;
mod log_broadcast;
mod log_file;
mod log_line;
mod model_choice;
mod projection_event;
mod raw_read;
mod refresh_ctx;
mod repo_scan;
mod response_section;
mod session;
mod tool_def;
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

    /// Override the daemon's working directory — where `.substrate` and an
    /// optional `projection.yaml` live. The daemon operates against this dir
    /// WITHOUT changing the terminal's cwd (paths are config-scoped, not a
    /// process `chdir`). Created if it doesn't exist. Takes precedence over the
    /// positional workspace. Use it to run a separate, uncommitted "mind" with
    /// its own substrate + tuned projection schema, e.g. `--working-dir ../mind`.
    #[arg(long)]
    working_dir: Option<PathBuf>,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Increase log verbosity.  -v = DEBUG, -vv = TRACE.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Suppress a projection sink's startup population — a turn-sink **layer**
    /// (e.g. `repo_map`, `code_reading`) or a section **collection** (e.g.
    /// `response`, `mood`), by its schema name. Repeatable (e.g. `--disable-layer
    /// repo_map --disable-layer code_reading`). The layer/collection still exists
    /// in the schema; it is just not populated at boot, and is skipped by the
    /// watcher refresh and uploads. Brings the daemon up fast without the ingest
    /// sweep.
    #[arg(long = "disable-layer", value_name = "NAME")]
    disable_layer: Vec<String>,

    /// Override the content root a derived ingest layer reads from, as
    /// `<layer>=<path>`. Repeatable (e.g. `--ingest-dir code_reading=zend/src
    /// --ingest-dir repo_map=zend`). The path is relative to the workspace, or
    /// absolute. Scopes an ingest to a subtree so a rebuilt substrate stays
    /// small instead of absorbing the whole workspace; pair with
    /// `--disable-layer` to skip a layer outright.
    #[arg(long = "ingest-dir", value_name = "LAYER=PATH")]
    ingest_dir: Vec<String>,

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

    /// DESTRUCTIVE: delete the working dir's `.substrate` directory (redo-log
    /// segments, logs — the daemon's entire persistent memory) before loading,
    /// so this run starts from a blank substrate. Exists precisely so scripts
    /// and test harnesses never have to `rm -rf` a substrate path themselves —
    /// the deletion only ever happens behind this explicit flag, scoped to the
    /// resolved working dir.
    #[arg(long)]
    wipe_substrate: bool,

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

    // `--working-dir` (if given) is the workspace; otherwise the positional path.
    // Create it first so a fresh mind directory has somewhere for `.substrate` and
    // `projection.yaml` to land, then canonicalize to an absolute path.
    let ws_arg = cli
        .working_dir
        .clone()
        .unwrap_or_else(|| cli.workspace.clone());
    if let Err(e) = std::fs::create_dir_all(&ws_arg) {
        eprintln!(
            "warning: could not create working dir {}: {e}",
            ws_arg.display()
        );
    }
    let workspace = ws_arg.canonicalize().unwrap_or(ws_arg);

    // ── `--wipe-substrate`: explicit-flag-only substrate deletion ─────────────
    //
    // Runs BEFORE logging init (the file layer writes into `.substrate/`), so
    // the wipe is complete before anything re-creates the directory. Only the
    // resolved working dir's own `.substrate` is touched.
    if cli.wipe_substrate {
        wipe_substrate(&workspace)?;
    }

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

    // ── GPU poison watchdog ──────────────────────────────────────────────────
    //
    // A sticky CUDA fault (illegal address, launch failure, device assert, ECC)
    // permanently kills the CUDA context — every later call returns the same
    // error, it can't be cleared on this context, and recreating it in-process is
    // unreliable on WDDM. The CUDA backend flags `candle::gpu_poison` on the first
    // such fault; this watchdog turns that into a clean, fast restart instead of
    // an endless cascade of identical downstream errors: log the ONE root fault
    // (with the recent-launch breadcrumb), then exit for a supervisor to relaunch.
    // The substrate redo log is crash-safe, so nothing durable is lost.
    //
    // Exit code 75 (distinct from clean-shutdown 0 and Ctrl-C 130) tells a
    // relaunch wrapper "GPU died — restart me".
    const GPU_POISON_EXIT_CODE: i32 = 75;
    std::thread::Builder::new()
        .name("gpu-poison-watchdog".into())
        .spawn(|| loop {
            // Poll tightly so the window in which other threads can pile identical
            // downstream errors onto the dead context stays small.
            std::thread::sleep(std::time::Duration::from_millis(50));
            if candle::gpu_poison::is_gpu_poisoned() {
                tracing::error!(
                    root = %candle::gpu_poison::root_fault().unwrap_or_default(),
                    "GPU context poisoned by an unrecoverable CUDA fault — exiting for \
                     a clean restart (substrate redo log is durable, nothing lost)",
                );
                // Brief pause so the root log line reaches the file/console sinks
                // before the hard exit, then go.
                std::thread::sleep(std::time::Duration::from_millis(80));
                std::process::exit(GPU_POISON_EXIT_CODE);
            }
        })
        .expect("spawn gpu-poison-watchdog");

    let disabled_layers: std::collections::HashSet<String> =
        cli.disable_layer.iter().cloned().collect();

    // `--ingest-dir <layer>=<path>` — parsed up front so a malformed pair fails
    // the launch rather than silently ingesting the whole workspace.
    let mut ingest_dirs: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for spec in &cli.ingest_dir {
        let Some((layer, path)) = spec.split_once('=') else {
            anyhow::bail!("invalid --ingest-dir {spec:?}: expected <layer>=<path>");
        };
        if layer.trim().is_empty() || path.trim().is_empty() {
            anyhow::bail!("invalid --ingest-dir {spec:?}: both <layer> and <path> are required");
        }
        let path = path.trim();
        // Fail fast on a bad path — every ingest mode reads this root, and a typo
        // would otherwise silently ingest nothing (folder/file scans) rather than
        // erroring. Relative paths resolve under the workspace; absolute replace it.
        if !workspace.join(path).is_dir() {
            anyhow::bail!(
                "invalid --ingest-dir {spec:?}: {} is not a directory",
                workspace.join(path).display(),
            );
        }
        ingest_dirs.insert(layer.trim().to_string(), path.to_string());
    }

    let config = DaemonConfig {
        workspace: workspace.clone(),
        port: cli.port,
        disabled_layers: disabled_layers.clone(),
        ingest_dirs: ingest_dirs.clone(),
        disable_summariser: cli.disable_summariser,
        compact_substrate: cli.compact_substrate,
    };

    if !disabled_layers.is_empty() {
        let mut names: Vec<&str> = disabled_layers.iter().map(String::as_str).collect();
        names.sort_unstable();
        tracing::info!(
            layers = %names.join(", "),
            "--disable-layer: startup ingest suppressed for these projection layers",
        );
    }
    if !ingest_dirs.is_empty() {
        let mut pairs: Vec<String> = ingest_dirs
            .iter()
            .map(|(layer, path)| format!("{layer}={path}"))
            .collect();
        pairs.sort();
        tracing::info!(
            overrides = %pairs.join(", "),
            "--ingest-dir: these layers ingest from an overridden content root",
        );
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
    // Force-exit rather than falling out of `main`. The substrate is already
    // flushed durably above, so the only work left is process teardown — and the
    // detached background threads (the scheduler, the GPU/CUDA worker + context,
    // the persistence pipeline) are not all cleanly joinable, so dropping the tokio
    // runtime and tearing down the CUDA context otherwise hangs the process until a
    // manual Ctrl-C. Nothing durable is lost by exiting now.
    std::process::exit(0)
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

/// Delete `workspace/.substrate` — the daemon's entire persistent memory —
/// scoped strictly to the RESOLVED working dir's own `.substrate`. The only
/// caller is the explicit `--wipe-substrate` flag; scripts and harnesses go
/// through it so no shell ever `rm -rf`s a substrate path itself. Missing
/// directory is a no-op (a fresh mind dir), failure aborts boot rather than
/// half-deleting.
fn wipe_substrate(workspace: &std::path::Path) -> anyhow::Result<()> {
    let substrate_dir = workspace.join(".substrate");
    if substrate_dir.exists() {
        eprintln!(
            "--wipe-substrate: deleting {} (persistent substrate)",
            substrate_dir.display()
        );
        if let Err(e) = std::fs::remove_dir_all(&substrate_dir) {
            anyhow::bail!(
                "--wipe-substrate: failed to delete {}: {e}",
                substrate_dir.display()
            );
        }
    } else {
        eprintln!(
            "--wipe-substrate: {} does not exist — nothing to delete",
            substrate_dir.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod wipe_tests {
    use super::wipe_substrate;

    #[test]
    fn wipes_only_the_workspaces_own_substrate() {
        let root = tempfile::tempdir().unwrap();
        let ws = root.path().join("mind");
        std::fs::create_dir_all(ws.join(".substrate")).unwrap();
        std::fs::write(ws.join(".substrate").join("seg-0.log"), b"data").unwrap();
        // A sibling workspace's substrate must be untouched.
        let other = root.path().join("other");
        std::fs::create_dir_all(other.join(".substrate")).unwrap();
        std::fs::write(other.join(".substrate").join("seg-0.log"), b"keep").unwrap();

        wipe_substrate(&ws).unwrap();
        assert!(!ws.join(".substrate").exists(), "own substrate deleted");
        assert!(
            other.join(".substrate").join("seg-0.log").exists(),
            "sibling substrate untouched"
        );
        // Idempotent: a second wipe of the now-missing dir is a no-op.
        wipe_substrate(&ws).unwrap();
    }
}
