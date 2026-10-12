//! What belongs to the process rather than to any tenant: the panic hook, the
//! one tracing subscriber, and the shutdown signal.
//!
//! Each was a copy per daemon, and each is process-wide by nature — a second
//! `tracing` subscriber panics at `init`, a second panic hook replaces the
//! first, a Ctrl-C reaches the whole process. So the host installs each once,
//! for every tenant it runs (`docs/shared_model_host.md` §1.3).

use std::backtrace::Backtrace;
#[cfg(not(unix))]
use std::future::pending;
use std::panic::set_hook;
use std::path::Path;
use std::process::exit;
use std::sync::Arc;

use candle_conversation::relief_trace;
use tokio::signal::ctrl_c;
#[cfg(unix)]
use tokio::signal::unix::{signal, SignalKind};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use crate::log_file::{is_resuming, RotatingFileLog};
use crate::logs::{BusWriter, LogBus};

/// Print a full backtrace on every panic — no `RUST_BACKTRACE` needed — with
/// the last CUDA kernel launched on the crashing thread and the relief trace.
/// The kernel name is written into a thread-local breadcrumb before every
/// kernel launch, so it is available here even though a CUDA error surfaces
/// asynchronously at a later readback or synchronize.
pub fn install_panic_hook() {
    set_hook(Box::new(|info| {
        let bt = Backtrace::force_capture();
        let kernel = candle::last_cuda_kernel_launch();
        let relief = relief_trace::dump();
        eprintln!(
            "\n=== PANIC ===\n{info}\nLast CUDA kernel: {kernel}\n{relief}\n\n{bt}\n=============\n"
        );
    }));
}

/// The filter every sink shares, at `verbose` (0 info, 1 debug, 2+ trace).
///
/// Every tenant's crate and the engine's at the chosen level. `candle_nn` sits
/// **one step below**, because its debug tier is not diagnostics: the KV
/// cache's pools narrate every arena sweep and region claim, thousands of lines
/// a second under load, which buries the events worth reading. Anything in
/// that crate a person needs to see is logged at `warn` and survives this.
pub fn filter_directives(tenants: &[&str], verbose: u8) -> Vec<String> {
    let level = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let kv_level = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    let mut d: Vec<String> = tenants.iter().map(|t| format!("{t}={level}")).collect();
    for krate in [
        "host",
        "web",
        "candle_conversation",
        "candle_transformers",
        "candle_core",
    ] {
        d.push(format!("{krate}={level}"));
    }
    d.push(format!("candle_nn={kv_level}"));
    d
}

/// Install the one subscriber: the terminal, the bus `/ws/logs` serves, and a
/// size-rotated file `<log_dir>/<log_name>`. One formatter feeds all three, so
/// what an operator sees on screen and what the console shows cannot drift.
///
/// Returns whether this start resumes after a self-heal relaunch — read before
/// the file sink consumes the marker that says so — so the caller can log the
/// boundary once logging is up.
pub fn init_logging(
    bus: &Arc<LogBus>,
    tenants: &[&str],
    verbose: u8,
    log_dir: &Path,
    log_name: &str,
) -> anyhow::Result<bool> {
    let mut filter = EnvFilter::from_default_env();
    for d in filter_directives(tenants, verbose) {
        filter = filter.add_directive(d.parse()?);
    }
    // Targets on: they cost a prefix in the terminal and buy a column the
    // console can filter by — and without them a message containing a colon is
    // indistinguishable from `target: message`.
    let stdout = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_filter(filter.clone());
    let bus_writer = Arc::clone(bus);
    let to_bus = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_target(true)
        .with_writer(move || BusWriter::new(Arc::clone(&bus_writer)))
        .with_filter(filter.clone());

    let resumed = is_resuming(log_dir);
    // An unopenable file degrades to the terminal and the bus rather than
    // failing the start; `Option<Layer>` is itself a `Layer`.
    let to_file = RotatingFileLog::new(log_dir, log_name).map(|w| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_target(true)
            .with_writer(w)
            .with_filter(filter)
    });

    tracing_subscriber::registry()
        .with(stdout)
        .with(to_bus)
        .with(to_file)
        .try_init()?;
    Ok(resumed)
}

/// A tenant cannot go on — its engine failed to load, its model is not on this
/// machine. Every tenant shares the one engine, so the process ends: say who
/// and why, and exit. A half-loaded process answering requests shows a console
/// something present and broken, which is harder to diagnose than something
/// absent.
pub fn fatal(tenant: &str, why: &str) -> ! {
    tracing::error!("{tenant}: {why} — exiting");
    eprintln!("error: {tenant}: {why}");
    exit(1)
}

/// Resolves when the process receives Ctrl-C (or `SIGTERM` on Unix). A
/// *second* Ctrl-C aborts at once — the escape hatch if the drain wedges.
pub async fn shutdown_signal() {
    let interrupted = async {
        ctrl_c().await.expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal(SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = pending::<()>();

    tokio::select! {
        _ = interrupted => {}
        _ = terminate => {}
    }
    tracing::warn!(
        "shutdown signal received — draining in-flight work; press Ctrl-C again to abort"
    );

    tokio::spawn(async {
        if ctrl_c().await.is_ok() {
            tracing::error!("second Ctrl-C — aborting immediately");
            exit(130);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tenant and the engine at the chosen level; the KV cache one step
    /// below.
    #[test]
    fn the_filter_names_every_tenant_and_holds_the_kv_cache_back() {
        assert_eq!(
            filter_directives(&["zend", "npcd"], 0),
            [
                "zend=info",
                "npcd=info",
                "host=info",
                "web=info",
                "candle_conversation=info",
                "candle_transformers=info",
                "candle_core=info",
                "candle_nn=warn",
            ]
        );
        assert_eq!(
            filter_directives(&["zend"], 1).last().unwrap(),
            "candle_nn=info"
        );
        assert_eq!(filter_directives(&["zend"], 5)[0], "zend=trace");
    }
}
