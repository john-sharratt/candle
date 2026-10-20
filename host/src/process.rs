//! What belongs to the process rather than to any tenant: the panic hook, the
//! one tracing subscriber, and the stop signal.
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
use std::thread::current;

use candle_conversation::relief_trace;
use tokio::signal::ctrl_c;
#[cfg(unix)]
use tokio::signal::unix::{signal, SignalKind};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use anyhow::anyhow;

use crate::logs::LogBus;
use web::log_file::{is_resuming, RotatingFileLog};

/// Report a full backtrace on every panic — no `RUST_BACKTRACE` needed — with
/// the last CUDA kernel launched on the crashing thread and the relief trace.
/// The kernel name is written into a thread-local breadcrumb before every
/// kernel launch, so it is available here even though a CUDA error surfaces
/// asynchronously at a later readback or synchronize.
///
/// **Logged as well as printed.** keep runs a stable daemon with stderr
/// detached, so a panic that only reached stderr went nowhere: a scheduler
/// thread that panicked left its callers hearing "scheduler has shut down"
/// and the log saying nothing about why. The log file is the record that
/// survives, and `/v1/logs` searches it.
pub fn install_panic_hook() {
    set_hook(Box::new(|info| {
        let bt = Backtrace::force_capture();
        let kernel = candle::last_cuda_kernel_launch();
        let relief = relief_trace::dump();
        let me = current();
        let name = me.name().unwrap_or("<unnamed>");
        tracing::error!(
            thread = name,
            "PANIC: {info}\nLast CUDA kernel: {kernel}\n{relief}\n\n{bt}"
        );
        eprintln!(
            "\n=== PANIC ===\n{info}\nLast CUDA kernel: {kernel}\n{relief}\n\n{bt}\n=============\n"
        );
    }));
}

/// The filter every sink shares, at `verbose` (0 info, 1 debug, 2+ trace).
///
/// Every tenant's crate, the engine's and the repositories' — the substrate's
/// git stores, the object database, the agents that fetch for them — and the
/// cluster's, at the chosen level. Without `cluster` a replication fault (a
/// catch-up gone quiet and asked again, a refused record) was never written
/// anywhere, and a member starved by its peers' catch-ups said nothing about
/// why. `candle_nn` sits
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
        "substrate_git",
        "zend_vfs",
        "agent",
        "cluster",
        "cluster_dial",
    ] {
        d.push(format!("{krate}={level}"));
    }
    d.push(format!("candle_nn={kv_level}"));
    d
}

/// Install the one subscriber: the terminal, and a size-rotated file
/// `<log_dir>/<log_name>` that feeds the bus `/ws/logs` serves and `/v1/logs`
/// searches. The file is the console's only record, so a file that cannot be
/// opened fails the start rather than leaving the console silently empty.
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

    let resumed = is_resuming(log_dir);
    let file = RotatingFileLog::new(log_dir, log_name).ok_or_else(|| {
        anyhow!(
            "the log file {} could not be opened",
            log_dir.join(log_name).display()
        )
    })?;
    bus.attach(&file);
    let to_file = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_target(true)
        .with_writer(file)
        .with_filter(filter);

    tracing_subscriber::registry()
        .with(stdout)
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

/// Resolves when the process receives Ctrl-C (or `SIGTERM` on Unix).
pub async fn stop_signal() {
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
}

/// From here on a Ctrl-C aborts at once (`exit(130)`) — the escape hatch for
/// a stop that wedges. Armed once a stop has begun, so after the Ctrl-C that
/// began it this is the second.
pub fn abort_on_ctrl_c() {
    tracing::warn!("stopping — draining in-flight work; press Ctrl-C again to abort");
    tokio::spawn(async {
        if ctrl_c().await.is_ok() {
            tracing::error!("second Ctrl-C — aborting immediately");
            exit(130);
        }
    });
}

/// Log this process's committed host memory, allocation by allocation, every
/// 30 s through the first ten minutes and every ten minutes after.
///
/// **The commit charge is what fails first, and nothing else reported it.** On
/// WhiteWorld (2026-10-20) motherd held ~130 GB committed against a 17 GB
/// working set, the machine's commit stood 0.5 GB under its limit, and the
/// process aborted three times in twenty minutes on an allocation it could not
/// commit — with no panic, since an allocation failure aborts without one, and
/// stderr detached. The pinned experts and the warm tier explained about 77 GB
/// of it; this names the rest.
pub fn spawn_ram_census() {
    let spawned = std::thread::Builder::new()
        .name("ram-census".into())
        .spawn(|| {
            use candle::vram::process_ram::{ProcessRam, RegionKind};
            let started = std::time::Instant::now();
            loop {
                if let Some(ram) = ProcessRam::capture() {
                    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                    let (p, m, i) = (
                        ram.total(RegionKind::Private),
                        ram.total(RegionKind::Mapped),
                        ram.total(RegionKind::Image),
                    );
                    let top: Vec<String> = ram
                        .largest_committed(12)
                        .iter()
                        .map(|a| {
                            format!(
                                "{:#x} {:?} {:.2}/{:.2} GiB{}",
                                a.alloc_base,
                                a.kind,
                                gib(a.committed),
                                gib(a.resident),
                                a.name.as_deref().map_or(String::new(), |n| format!(" {n}"))
                            )
                        })
                        .collect();
                    let buckets: Vec<String> = ram
                        .private_by_size()
                        .iter()
                        .filter(|b| b.committed >= 1 << 30)
                        .map(|b| {
                            format!(
                                "<={:.2}GiB x{} {:.2}GiB",
                                gib(b.upper),
                                b.count,
                                gib(b.committed)
                            )
                        })
                        .collect();
                    tracing::info!(
                        target: "host::ram",
                        private_committed_gib = format!("{:.2}", gib(p.committed)),
                        private_resident_gib = format!("{:.2}", gib(p.resident)),
                        mapped_committed_gib = format!("{:.2}", gib(m.committed)),
                        image_committed_gib = format!("{:.2}", gib(i.committed)),
                        largest = %top.join(" | "),
                        private_buckets = %buckets.join(" | "),
                        "process committed memory"
                    );
                }
                let every = if started.elapsed() < std::time::Duration::from_secs(600) {
                    30
                } else {
                    600
                };
                std::thread::sleep(std::time::Duration::from_secs(every));
            }
        });
    if let Err(e) = spawned {
        tracing::warn!("could not start the RAM census: {e}");
    }
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
                "substrate_git=info",
                "zend_vfs=info",
                "agent=info",
                "cluster=info",
                "cluster_dial=info",
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
