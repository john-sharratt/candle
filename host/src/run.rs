//! The one startup path: `host::run([tenant, …])` (`docs/shared_model_host.md`
//! §2, decision 2).
//!
//! `zend`, `npcd` and `tokerad` are the same call with different tenants. The
//! host owns, once for the process: the panic hook, the tracing subscriber, the
//! self-heal watchdog, the runtime, the telemetry sampler, the engine slot every
//! tenant takes its engine from, the HTTP server, and the exit.

use std::collections::HashMap;
use std::mem::replace;
use std::path::{Component, Path, PathBuf};
use std::process::exit;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_conversation::profile::set_gpu_span_period;
use candle_conversation::{EngineSlot, TenantRank};
use tokio::runtime::Builder as RuntimeBuilder;
use web::Roots;

use crate::access::{roles, LocalSignin};
use crate::cli::HostArgs;
use crate::ctrl_c::accept_ctrl_c;
use crate::endpoints::{routes, Shared};
use crate::guard::Routes;
use crate::logs::LogBus;
use crate::model::{ModelView, MODEL};
use crate::process::{init_logging, install_panic_hook, shutdown_signal};
use crate::self_heal::{spawn_watchdog, LaunchSpec, OneShot};
use crate::server::serve;
use crate::telemetry::Telemetry;
use crate::tenant::{Stop, Tenant, TenantContext, TenantUi};
use crate::ui::{require_dir, COMMON_DIR};

/// One forward in this many records its GPU spans in a profiling build (see
/// `set_gpu_span_period`). At a decode step of ~35 ms that is a sampled forward
/// every ~0.3 s — dozens per `/v1/profile` window — while the rest run as the
/// release build does.
const GPU_SPAN_PERIOD: u32 = 8;

/// How the process runs, whatever its tenants.
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// The binary's name — its log file is `<binary>.log`, beside the
    /// highest-ranked tenant's substrate.
    pub binary: &'static str,
    /// The binary's version, part of the UI build id.
    pub version: &'static str,
    /// Where it listens, whom it believes, how much it logs.
    pub args: HostArgs,
    /// The tenants' flags that act once and that a self-heal relaunch drops
    /// ([`LaunchSpec::capture`]).
    pub one_shot: &'static [OneShot],
}

/// Refuse two tenants naming the same substrate directory, before either
/// opens it.
pub fn check_substrates(dirs: &[(&str, PathBuf)]) -> anyhow::Result<()> {
    let mut seen: HashMap<PathBuf, &str> = HashMap::new();
    for (name, dir) in dirs {
        if let Some(other) = seen.insert(resolved(dir), name) {
            anyhow::bail!(
                "{other} and {name} both name the substrate at {} — one substrate takes one \
                 writer; give each tenant its own",
                dir.display()
            );
        }
    }
    Ok(())
}

/// `dir` as one spelling: `.` and `..` folded, then its deepest existing
/// ancestor resolved — so a directory not yet created still compares equal
/// however it is written.
fn resolved(dir: &Path) -> PathBuf {
    let mut folded = PathBuf::new();
    for part in dir.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                folded.pop();
            }
            other => folded.push(other),
        }
    }
    let mut missing = Vec::new();
    let mut at = folded.as_path();
    loop {
        if let Ok(real) = at.canonicalize() {
            return missing.iter().rev().fold(real, |p, name| p.join(name));
        }
        match (at.file_name(), at.parent()) {
            (Some(name), Some(parent)) => {
                missing.push(name.to_owned());
                at = parent;
            }
            _ => return folded,
        }
    }
}

/// Run `tenants` — ranked in the order given — until the process is told to
/// stop, then drain them in reverse and exit. Never returns: the host exits
/// once, after every tenant has stopped, or with status 1 on a failed start.
pub fn run(cfg: HostConfig, tenants: Vec<Box<dyn Tenant>>) -> ! {
    install_panic_hook();
    // As early as possible, before anything could change argv or the working
    // directory: a launch spec captured late might not be this one.
    let launch = match LaunchSpec::capture(cfg.one_shot) {
        Ok(l) => l,
        Err(e) => fail_early(&format!("could not capture the launch: {e}")),
    };
    let Some(first) = tenants.first() else {
        fail_early("no tenants to run");
    };
    // The log, the self-heal restart marker and its attempt count live beside
    // the highest-ranked tenant's substrate — the process's state, in the one
    // directory every run of this binary shares.
    let state_dir = first.substrate_dir();
    let names: Vec<&'static str> = tenants.iter().map(|t| t.name()).collect();
    let bus = LogBus::new(&names);
    let log_name = format!("{}.log", cfg.binary);
    let resumed = match init_logging(&bus, &names, cfg.args.verbose, &state_dir, &log_name) {
        Ok(r) => r,
        Err(e) => fail_early(&format!("could not start logging: {e:#}")),
    };
    if resumed {
        tracing::error!(
            "=== resumed after a self-heal restart — see the GPU-poisoned error above this \
             line in the preserved log for why =========================================="
        );
    }
    // A sticky CUDA fault or an unbroken out-of-memory streak kills the
    // context for good; nothing supervises this process, so it relaunches
    // itself (`self_heal.rs`).
    spawn_watchdog(launch, state_dir);
    // A no-op without the `profile` feature.
    set_gpu_span_period(GPU_SPAN_PERIOD);

    let runtime = match RuntimeBuilder::new_multi_thread().enable_all().build() {
        Ok(r) => r,
        Err(e) => fail(&format!("could not start the runtime: {e}")),
    };
    match runtime.block_on(serve_tenants(cfg, tenants, bus)) {
        Ok(()) => {
            tracing::info!("stopped");
            // Exit rather than fall out of `main`: every tenant drained and
            // flushed its substrate above, and the detached engine threads
            // (scheduler, GPU worker, CUDA context) are not all cleanly
            // joinable, so tearing the runtime down could otherwise hang the
            // process. Nothing durable is lost by exiting now.
            exit(0)
        }
        Err(e) => fail(&format!("{e:#}")),
    }
}

async fn serve_tenants(
    cfg: HostConfig,
    tenants: Vec<Box<dyn Tenant>>,
    bus: Arc<LogBus>,
) -> anyhow::Result<()> {
    let substrates: Vec<(&str, PathBuf)> = tenants
        .iter()
        .map(|t| (t.name(), t.substrate_dir()))
        .collect();
    check_substrates(&substrates)?;
    let listen = cfg.args.listen()?;
    let uis: Vec<(&'static str, TenantUi)> = tenants
        .iter()
        .map(|t| {
            let mut ui = t.ui();
            ui.content = require_dir(t.name(), &ui.content)?;
            Ok((t.name(), ui))
        })
        .collect::<anyhow::Result<_>>()?;
    let common = require_dir("the shared front-end", &PathBuf::from(COMMON_DIR))?;
    let local_signin = LocalSignin::new(cfg.args.local_signin.as_deref())?;
    if let Some(email) = &cfg.args.local_signin {
        tracing::warn!(
            email = %email,
            "--local-signin: a loopback caller with no forwarded identity is recognized as \
             this email — ANY other process or account on this machine, and anything that \
             relays remote traffic onto loopback (a tunnel, a proxy on this box), is \
             recognized the same way while this flag is set",
        );
    }

    let model = ModelView::of_prescribed();
    tracing::info!(
        model = %MODEL.spec().model_filename,
        tenants = %names_of(&uis),
        "starting",
    );
    let telemetry = Telemetry::new(model.clone());
    telemetry.spawn_sampler();
    let roles = roles();

    // Every tenant takes its engine from one slot: the highest-ranked builds it
    // with every tenant's requests merged, the rest attach their substrates.
    let slot = EngineSlot::new(tenants.len());
    let mut reports = Vec::new();
    let mut tenant_routes = Vec::new();
    let mut stops: Vec<(&'static str, Stop)> = Vec::new();
    for (i, tenant) in tenants.into_iter().enumerate() {
        let name = tenant.name();
        let Some(rank) = TenantRank::nth(i) else {
            stop_all(stops).await;
            anyhow::bail!(
                "{name}: at most {} tenants share one engine",
                TenantRank::MAX
            );
        };
        let ctx = TenantContext {
            engine: slot.source(rank, name, tenant.requires()),
            log: Arc::clone(&bus),
            roles: roles.clone(),
            telemetry: Arc::clone(&telemetry),
        };
        match tenant.start(ctx).await {
            Ok(started) => {
                reports.push((name, started.reports));
                tenant_routes.push((name, started.routes));
                stops.push((name, started.stop));
            }
            // The tenants already started are stopped, not left mid-load
            // when the process exits.
            Err(e) => {
                stop_all(stops).await;
                return Err(e.context(format!("{name}: start")));
            }
        }
    }

    let mut content_dirs: Vec<PathBuf> = uis.iter().map(|(_, ui)| ui.content.clone()).collect();
    content_dirs.push(common.clone());
    let shared = Arc::new(Shared {
        started_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64),
        version: cfg.version,
        content_dirs,
        model,
        tenants: reports,
        log: bus,
        telemetry,
        roles,
    });
    let all = match merge_routes(routes(shared, Roots::disk(&[common])), tenant_routes) {
        Ok(all) => all,
        Err(e) => {
            stop_all(stops).await;
            return Err(e);
        }
    };
    tracing::info!("routes: {}", all.declared.len());
    for g in &all.declared {
        tracing::debug!("route {g}");
    }

    accept_ctrl_c();
    let served = serve(&listen, &uis, all.router, local_signin, shutdown_signal()).await;
    stop_all(stops).await;
    served
}

/// Stop every started tenant in reverse rank: the lowest-ranked first, so the
/// engine's last attachment — the highest-ranked — is the one that stops the
/// scheduler.
async fn stop_all(stops: Vec<(&'static str, Stop)>) {
    for (name, stop) in stops.into_iter().rev() {
        tracing::info!("{name}: stopping");
        stop().await;
        tracing::info!("{name}: stopped");
    }
}

/// The host's shared endpoints and every tenant's own routes, as one router —
/// refused, naming the path and both owners, when two register the same path.
pub fn merge_routes(host: Routes, tenants: Vec<(&str, Routes)>) -> anyhow::Result<Routes> {
    let mut all = host;
    let mut owner = String::from("host");
    for (name, r) in tenants {
        let so_far = replace(&mut all, Routes::empty());
        all = so_far.merge(&owner, r, name)?;
        owner = format!("{owner}/{name}");
    }
    Ok(all)
}

fn names_of(uis: &[(&str, TenantUi)]) -> String {
    uis.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
}

/// A failed start, once logging is up: say why, and exit.
fn fail(why: &str) -> ! {
    tracing::error!("{why}");
    eprintln!("error: {why}");
    exit(1)
}

/// A failed start before logging exists.
fn fail_early(why: &str) -> ! {
    eprintln!("error: {why}");
    exit(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    /// Two tenants naming one substrate refuse the start, naming both; two
    /// directories are fine — however the same one is spelled.
    #[test]
    fn two_tenants_cannot_share_a_substrate() {
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("a");
        let b = tmp.path().join("b");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        assert!(check_substrates(&[("zend", a.clone()), ("npcd", b)]).is_ok());

        let spelled = tmp.path().join("b").join("..").join("a");
        let err = check_substrates(&[("zend", a), ("npcd", spelled)])
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("zend and npcd both name the substrate"),
            "{err}"
        );

        // A directory neither has created yet, spelled two ways.
        let fresh = tmp.path().join("fresh").join("substrate");
        let respelled = tmp
            .path()
            .join("nowhere")
            .join("..")
            .join("fresh")
            .join(".")
            .join("substrate");
        assert!(check_substrates(&[("zend", fresh), ("npcd", respelled)]).is_err());
    }
}
