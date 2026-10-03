//! Names the thread that frees a chunk slot while a compaction pass is running.
//!
//! A pass plans its moves from an occupancy census and then claims, copies and
//! sweeps. A slot freed by another thread inside that window breaks two of its
//! assumptions at once: the copy of a source nobody holds any more is named by no
//! holder (`unwitnessed`), and the freed slot goes back on the free list where the
//! pass's own claims can be handed it as a destination (`source_collisions`, a move
//! declined). Neither is a wrong read — the sweep only rewrites holders it finds — but
//! both are a pass that cannot finish packing, which is what pins the frontier.
//!
//! The pass marks its window; a last release on any other thread inside it is
//! reported with its backtrace, once per distinct site and then at powers of two.
//!
//! Behind `tensor-assert`, like the rest of the harness: without it the release path
//! carries nothing. With it, a release outside a pass pays one relaxed load.

use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread::ThreadId;

/// Whether a pass is between its census and the end of its sweep.
static IN_PASS: AtomicBool = AtomicBool::new(false);

/// The pass's own thread, whose releases are the pass's business.
static PASS_THREAD: Mutex<Option<ThreadId>> = Mutex::new(None);

/// Releases on other threads seen inside pass windows, in total.
static FOREIGN: AtomicU64 = AtomicU64::new(0);

/// Every distinct reported site, with how often it was seen.
static SITES: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

/// A pass's window, open on the thread that created it until this drops — so every
/// exit from the pass, early refusals included, closes it.
pub(crate) struct PassWindow;

impl PassWindow {
    /// Open the window on the calling thread.
    pub(crate) fn open() -> Self {
        if let Ok(mut t) = PASS_THREAD.lock() {
            *t = Some(std::thread::current().id());
        }
        FOREIGN.store(0, Ordering::Relaxed);
        IN_PASS.store(true, Ordering::Release);
        PassWindow
    }

    /// Releases on other threads inside the window so far.
    pub(crate) fn foreign(&self) -> u64 {
        FOREIGN.load(Ordering::Relaxed)
    }
}

impl Drop for PassWindow {
    fn drop(&mut self) {
        IN_PASS.store(false, Ordering::Release);
    }
}

/// Called on a slot's last release.
#[inline]
pub(crate) fn on_release(raw: i64) {
    if !IN_PASS.load(Ordering::Relaxed) {
        return;
    }
    let me = std::thread::current().id();
    if PASS_THREAD.lock().ok().and_then(|t| *t) == Some(me) {
        return;
    }
    FOREIGN.fetch_add(1, Ordering::Relaxed);
    let thread = std::thread::current()
        .name()
        .unwrap_or("<unnamed>")
        .to_string();
    let site = holder_frames(&Backtrace::force_capture().to_string());
    let Ok(mut guard) = SITES.lock() else { return };
    let sites = guard.get_or_insert_with(HashMap::new);
    let count = sites.entry(format!("{thread}\n{site}")).or_insert(0);
    *count += 1;
    if count.is_power_of_two() {
        tracing::warn!(
            target: "candle_nn::kv_cache::release_watch",
            thread,
            raw,
            count = *count,
            "a chunk slot was freed on another thread inside a compaction pass:\n{site}",
        );
    }
}

/// The frames of a backtrace in this workspace's code, each with its source line,
/// below the gid machinery and this module.
fn holder_frames(trace: &str) -> String {
    let lines: Vec<&str> = trace.lines().map(str::trim).collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if l.starts_with("at ")
            || !l.contains("candle")
            || l.contains("release_watch")
            || l.contains("gid_pool")
        {
            continue;
        }
        let at = lines
            .get(i + 1)
            .filter(|n| n.starts_with("at "))
            .map(|n| format!("  {n}"))
            .unwrap_or_default();
        out.push(format!("{l}{at}"));
        if out.len() == 12 {
            break;
        }
    }
    out.join("\n")
}
