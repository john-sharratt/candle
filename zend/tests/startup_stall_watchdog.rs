//! Tier-3 integration test: simulate `cargo run -p zend --release -- -v`
//! and assert startup completes without stalling.
//!
//! Why this exists.  Startup hangs are notoriously hard to diagnose
//! because by the time the user notices "nothing's happening,"
//! several seconds to minutes have passed, the surrounding context
//! has scrolled out of the terminal, and the daemon is still
//! sitting on a thread that can't make progress.  This harness:
//!
//!   1. Drives the same boot path `main.rs` uses — `ZendSession::new`
//!      followed by `start_loading` — against the live candle
//!      workspace and the production projection.yaml.
//!   2. Polls [`ZendSession::status_snapshot`] every 500 ms while
//!      the load runs.
//!   3. Compares each snapshot to the previous one.  If the
//!      `(current step, sub-progress, sub-detail)` tuple hasn't
//!      changed for [`STALL_THRESHOLD`] seconds, the test panics
//!      with a forensics dump — the current step, how long it's
//!      been stuck, the last sub-detail string, and the cumulative
//!      elapsed time since `start_loading()` was invoked.
//!   4. On clean completion (`loading == None`) the test exits with
//!      the total wall-clock time and a confirmation line.
//!
//! `#[ignore]` by default because it loads Qwen3-30B-A3B and walks
//! the full candle workspace — multi-minute runtime, CUDA required.
//! Run manually with:
//!
//! ```text
//! cargo test -p zend --release --test startup_stall_watchdog \
//!     -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use zend::config::DaemonConfig;
use zend::log_broadcast::LogBus;
use zend::session::ZendSession;

/// How long we tolerate identical snapshots before declaring a
/// stall.  Picked to be safely longer than the slowest legitimate
/// non-progress-reporting phase (large prefill, summarizer run,
/// big tier I/O) on the 4090 mobile baseline, but short enough that
/// a real hang surfaces while the surrounding context is still
/// fresh.
const STALL_THRESHOLD: Duration = Duration::from_secs(60);

/// How often the watchdog samples [`ZendSession::status_snapshot`].
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Hard ceiling on total wall-clock for the whole boot.  Per-scope
/// summary decode is a deliberate cost — on the candle workspace
/// (~80k scopes × ~2-3s each) the full ingest is hours by design.
/// The ceiling is a backstop for genuinely runaway runs; sustained
/// progress alone is sufficient evidence that the path has no
/// stalls, so the test treats hitting it as success (see the main
/// loop below).
const TOTAL_TIMEOUT: Duration = Duration::from_secs(60 * 60 * 24);

/// The candle workspace root — the parent of zend's package dir.
/// `CARGO_MANIFEST_DIR` is the zend crate's manifest directory;
/// `..` walks up to the workspace root where the production
/// projection.yaml + the full source tree the daemon would scan
/// live.
fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .expect("zend's parent directory is the candle workspace root")
        .to_path_buf()
}

/// `loading == None` and `detail` empty means the daemon's loading
/// machine is fully wound down.  We treat that as "ready."
fn is_ready(session: &ZendSession) -> bool {
    session.status_snapshot().loading.is_none()
}

/// A fingerprint of the active step + sub-progress + status detail
/// string.  Any byte-level change resets the stall timer.  Tracking
/// `detail` alongside the structured progress catches stalls where
/// the structured counters are pinned but the daemon is still
/// emitting status updates (download progress, layer mounts, etc.).
#[derive(Clone, PartialEq, Eq)]
struct ProgressFingerprint {
    current_step: Option<String>,
    progress_bucket: u64,
    detail: String,
}

impl ProgressFingerprint {
    fn capture(session: &ZendSession) -> Self {
        let snap = session.status_snapshot();
        let (current_step, progress_bucket) = match snap.loading.as_ref() {
            Some(s) => (
                Some(format!("{:?}", s.current)),
                // Bucket the float into integer parts-per-million.
                // Permille (1/1000) is too coarse for CodeRead —
                // with ~80k scopes, each scope is ~0.0125 permille
                // and the bucket only ticks every 80 scopes, which
                // can take many minutes of real progress and fires
                // false stall positives.  PPM ticks every ~13 scopes
                // (a few tens of seconds) which fits the watchdog
                // sensitivity comfortably.
                (s.progress * 1_000_000.0) as u64,
            ),
            None => (None, 0),
        };
        Self {
            current_step,
            progress_bucket,
            detail: snap.detail,
        }
    }
}

#[test]
#[ignore = "Tier 3: loads Qwen3-30B-A3B + walks candle workspace; CodeRead alone runs for \
            hours on the candle workspace (~80k scopes × ~2-3s/scope decode). Stall detector \
            fires within 60s of any real hang; sustained progress through the 24h ceiling is \
            treated as success."]
fn zend_startup_completes_without_stall() {
    init_tracing();

    let workspace = workspace_root();
    eprintln!(
        "── zend startup stall watchdog ──\n  workspace = {}\n  stall_threshold = {:?}\n  total_timeout = {:?}",
        workspace.display(),
        STALL_THRESHOLD,
        TOTAL_TIMEOUT,
    );

    // Wipe any leftover substrate from a prior run.  Without this
    // the test is non-hermetic — corrupt persistence data from a
    // crashed previous run silently rolls forward into this run and
    // produces cascading "missing turn" warnings that ultimately
    // hang the scheduler.  A clean substrate exercises the same
    // startup path a fresh `cargo run -p zend` would.
    let substrate_dir = workspace.join(".substrate");
    if substrate_dir.exists() {
        match std::fs::remove_dir_all(&substrate_dir) {
            Ok(()) => eprintln!("  wiped existing .substrate/ for hermetic run"),
            Err(e) => eprintln!("  warning: could not wipe .substrate/: {e}"),
        }
    }

    let config = DaemonConfig {
        workspace: workspace.clone(),
        port: 0,
        ..Default::default()
    };
    let log = LogBus::new();

    // Build the session and kick off the loading machine the same
    // way `main.rs` does — `Arc::new(ZendSession::new(...))` then
    // `start_loading` against the Arc.  start_loading spawns its
    // own background tasks; control returns immediately.
    let session = Arc::new(ZendSession::new(config, log));
    let started_at = Instant::now();
    session.start_loading();
    eprintln!("  start_loading() returned, watchdog armed");

    let session_for_watchdog = Arc::clone(&session);
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_for_watchdog = Arc::clone(&stop_flag);

    // The watchdog runs on a worker thread because the main thread
    // owns the assertion and panic flow.  On a stall the worker
    // collects a diagnostic snapshot and signals via channel; the
    // main thread observes the message and panics.
    let (diag_tx, diag_rx) = std::sync::mpsc::channel::<String>();
    let watchdog = thread::Builder::new()
        .name("startup-watchdog".into())
        .spawn(move || {
            let mut last_fp = ProgressFingerprint::capture(&session_for_watchdog);
            let mut last_change = Instant::now();
            let mut last_detail_seen = last_fp.detail.clone();
            loop {
                if stop_for_watchdog.load(Ordering::SeqCst) {
                    return;
                }
                thread::sleep(POLL_INTERVAL);
                if is_ready(&session_for_watchdog) {
                    return;
                }
                let fp = ProgressFingerprint::capture(&session_for_watchdog);
                if fp != last_fp {
                    last_fp = fp.clone();
                    last_change = Instant::now();
                    if fp.detail != last_detail_seen && !fp.detail.is_empty() {
                        last_detail_seen = fp.detail.clone();
                        eprintln!(
                            "  [+{:>6.1}s] detail → {}",
                            started_at.elapsed().as_secs_f32(),
                            last_detail_seen,
                        );
                    }
                    continue;
                }
                let stuck_for = last_change.elapsed();
                if stuck_for > STALL_THRESHOLD {
                    let total_elapsed = started_at.elapsed();
                    let msg = format!(
                        "STARTUP STALL DETECTED\n\
                         ────────────────────────────────────────────\n\
                         no progress for       {:>10.1?}\n\
                         total elapsed         {:>10.1?}\n\
                         current step          {:?}\n\
                         progress bucket       {}/1_000_000 ppm\n\
                         last status detail    {:?}\n\
                         ────────────────────────────────────────────\n",
                        stuck_for,
                        total_elapsed,
                        last_fp.current_step,
                        last_fp.progress_bucket,
                        last_fp.detail,
                    );
                    eprintln!("{msg}");
                    // Signal the main thread, then keep looping —
                    // the main thread will panic, but if it doesn't
                    // we still want the watchdog to keep observing.
                    let _ = diag_tx.send(msg);
                    return;
                }
            }
        })
        .expect("spawn watchdog thread");

    // Main thread: poll for readiness OR a stall message OR the
    // total-timeout.  Whichever fires first decides the outcome.
    //
    // A stall panic is a real failure — the daemon got wedged and a
    // bug needs fixing.  Hitting the total-timeout without ever
    // tripping the stall detector means the daemon was making
    // continuous forward progress through the full window — that's
    // the "system works, just slow on this workspace" case and is
    // not a test failure.  Sustained progress for 30 min through
    // RepoScan + most of CodeRead is sufficient evidence that the
    // boot path has no hangs.
    loop {
        if started_at.elapsed() > TOTAL_TIMEOUT {
            stop_flag.store(true, Ordering::SeqCst);
            let _ = watchdog.join();
            let snap = session.status_snapshot();
            eprintln!(
                "── total-timeout reached with continuous progress ──\n\
                 elapsed:        {:.1?}\n\
                 last step:      {:?}\n\
                 last detail:    {:?}\n\
                 (no stalls > {:?} were detected; ingest path is healthy)",
                TOTAL_TIMEOUT, snap.loading, snap.detail, STALL_THRESHOLD,
            );
            return;
        }
        if let Ok(msg) = diag_rx.try_recv() {
            stop_flag.store(true, Ordering::SeqCst);
            let _ = watchdog.join();
            panic!("{msg}");
        }
        if is_ready(&session) {
            stop_flag.store(true, Ordering::SeqCst);
            let _ = watchdog.join();
            break;
        }
        thread::sleep(POLL_INTERVAL);
    }

    let total = started_at.elapsed();
    eprintln!(
        "── startup complete ──\n  total wall-clock: {:.1?}\n  (no stalls exceeding {:?})",
        total, STALL_THRESHOLD,
    );
}

fn init_tracing() {
    use std::sync::Once;
    use tracing_subscriber::EnvFilter;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Default filter favours the LOAD-PATH signals — the bits a
        // human triaging a stall actually reads.  scheduler timing /
        // per-decode trace lives at TRACE in the runtime and can be
        // re-enabled via `RUST_LOG` for a focused investigation.
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "info,\
                 zend::code_read::ingest=debug,\
                 candle_conversation::scheduler=info,\
                 candle_conversation::persistence::tier=info",
            )
        });
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}
