//! The memory census, run (`docs/memory_census.md`): once a minute, a reading
//! of what this process holds — the counting allocator's figures, the heap's own
//! summary, the commit charge — kept for an hour, judged for growth, logged, and
//! served to `/v1/memory` and `/v1/telemetry`.
//!
//! | Module | What |
//! |---|---|
//! | [`untracked`] | The commit charge nothing explains |
//! | [`warnings`] | Commit against the limit, and how often a warning repeats |
//!
//! The heap's summary holds the heap's lock while it is read, so it is read
//! here, once a minute, on a blocking thread — never per request or per
//! telemetry sample. Everything polled more often reads the last minute's
//! figure, with its age.

pub mod untracked;
pub mod warnings;

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use candle_conversation::memory_report;
use candle_conversation::store_gauges::{all as store_gauges, StoreGauge};
use heap_census::history::History;
use heap_census::os::{machine_commit, MachineCommit, OsHeap};
use heap_census::reading::Reading;
use heap_census::verdict::{grown_classes, judge, Growth, Verdict};
use serde::Serialize;
use tokio::time::{interval, MissedTickBehavior};

use untracked::{untracked, Parts};
use warnings::{hours_to_limit, CommitLevel, Repeat};

/// How often a reading is taken.
pub const READ_EVERY: Duration = Duration::from_secs(60);
/// How often the census line is logged, in readings.
pub const LOG_EVERY: u32 = 5;
/// The window growth is judged over.
pub const WINDOW_MS: u64 = 30 * 60 * 1000;
/// Growth over [`WINDOW_MS`] that is reported.
pub const GROWTH_THRESHOLD: u64 = 2 << 30;

const MIB: f64 = 1024.0 * 1024.0;

/// The last reading and what it was judged.
#[derive(Debug, Clone, Serialize)]
pub struct CensusNow {
    /// When, by the wall clock, for the reader; the census's own windows are
    /// timed on a monotonic one.
    pub at_unix_ms: u64,
    /// The heap's own summary at that reading.
    #[serde(skip)]
    pub os_heap: Option<OsHeap>,
    pub untracked_bytes: Option<u64>,
    /// The machine's commit, where the platform states a limit.
    pub machine_committed_bytes: Option<u64>,
    pub machine_commit_limit_bytes: Option<u64>,
    /// Growth over the last [`WINDOW_MS`], once there is a window to judge.
    pub growth: Option<Growth>,
}

/// The cause a growth warning names, so a warning that comes to name another
/// is said at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrowthKind {
    Retention,
    Owner,
    OutsideHeap,
    NotHeld,
}

impl GrowthKind {
    fn of(v: &Verdict) -> Option<GrowthKind> {
        match v {
            Verdict::Steady => None,
            Verdict::Retention => Some(GrowthKind::Retention),
            Verdict::Owner { .. } => Some(GrowthKind::Owner),
            Verdict::OutsideHeap => Some(GrowthKind::OutsideHeap),
            Verdict::NotHeld => Some(GrowthKind::NotHeld),
        }
    }
}

struct State {
    history: History,
    now: Option<CensusNow>,
    readings: u32,
    growing: Repeat<GrowthKind>,
    commit: Repeat<CommitLevel>,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(State {
            history: History::new(),
            now: None,
            readings: 0,
            growing: Repeat::new(),
            commit: Repeat::new(),
        })
    })
}

/// The last reading, if one has been taken.
pub fn latest() -> Option<CensusNow> {
    state()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .now
        .clone()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Milliseconds on a monotonic clock, from the census's first reading: what
/// readings are stamped with, so a wall clock stepped back never reverses a
/// window.
fn mono_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// The engine's pinned host bytes and VRAM in use, from its last memory report.
fn engine_parts() -> (u64, u64) {
    memory_report::latest().map_or((0, 0), |(r, _)| {
        (
            r.experts.host_pinned_bytes,
            r.accounting.device_in_use_bytes,
        )
    })
}

/// Take one reading, judge it, and log what it calls for. Blocking: the heap
/// summary waits on the heap's lock.
pub fn read_once() {
    let at_ms = mono_ms();
    let Some(reading) = Reading::take(at_ms) else {
        return;
    };
    let commit = machine_commit();
    let (pinned, device) = engine_parts();
    let untracked = reading.heap.map(|h| {
        untracked(&Parts {
            private_bytes: reading.private_bytes,
            heap_committed_bytes: h.committed_bytes,
            pinned_bytes: pinned,
            device_in_use_bytes: device,
        })
    });
    let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
    let previous_untracked = s.now.as_ref().and_then(|n| n.untracked_bytes);
    s.history.push(reading.clone());
    let growth = s
        .history
        .window(WINDOW_MS)
        .map(|(old, new)| judge(old, new, GROWTH_THRESHOLD));
    s.readings = s.readings.wrapping_add(1);
    if s.readings % LOG_EVERY == 1 {
        log_census(&s.history, &reading, untracked, commit.as_ref());
    }
    match growth
        .as_ref()
        .and_then(|g| GrowthKind::of(&g.verdict).map(|k| (g, k)))
    {
        Some((g, kind)) => {
            if s.growing.due(kind, at_ms) {
                log_growing(g, untracked, previous_untracked);
            }
        }
        None => s.growing.clear(),
    }
    if let Some(c) = &commit {
        let level = CommitLevel::of(c);
        if level == CommitLevel::Normal {
            s.commit.clear();
        } else if s.commit.due(level, at_ms) {
            log_commit(c, level, growth.as_ref());
        }
    }
    s.now = Some(CensusNow {
        at_unix_ms: now_ms(),
        os_heap: reading.heap,
        untracked_bytes: untracked,
        machine_committed_bytes: commit.map(|c| c.committed_bytes),
        machine_commit_limit_bytes: commit.map(|c| c.limit_bytes),
        growth,
    });
}

/// Run the census for the life of the process.
pub fn spawn() {
    tokio::spawn(async move {
        let mut tick = interval(READ_EVERY);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            if tokio::task::spawn_blocking(read_once).await.is_err() {
                break;
            }
        }
    });
}

fn mib(b: u64) -> f64 {
    (b as f64 / MIB).round()
}

fn mib_i(b: i64) -> f64 {
    (b as f64 / MIB).round()
}

/// The classes that grew most over the window, as a log field.
fn grown_field(history: &History) -> String {
    let Some((old, new)) = history.window(WINDOW_MS) else {
        return String::new();
    };
    grown_classes(old, new)
        .iter()
        .map(|g| format!("≤{} B: +{} MiB", g.upto_bytes, mib_i(g.grew_bytes)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn log_census(
    history: &History,
    r: &Reading,
    untracked: Option<u64>,
    commit: Option<&MachineCommit>,
) {
    let window = history.window(WINDOW_MS);
    let grew = |f: &dyn Fn(&Reading) -> Option<u64>| {
        window.and_then(|(o, n)| Some(mib_i(f(n)? as i64 - f(o)? as i64)))
    };
    tracing::info!(
        target: "host::memory",
        private_mib = mib(r.private_bytes),
        private_grew_mib = grew(&|x| Some(x.private_bytes)),
        heap_live_mib = r.live_bytes().map(mib),
        heap_live_grew_mib = grew(&|x| x.live_bytes()),
        heap_committed_mib = r.heap_committed().map(mib),
        heap_committed_grew_mib = grew(&|x| x.heap_committed()),
        heap_summary_us = r.heap.map(|h| h.read_us),
        untracked_mib = untracked.map(mib),
        machine_commit_pct = commit.map(|c| (c.committed_bytes as f64 * 100.0 / c.limit_bytes.max(1) as f64).round()),
        grown_classes = %grown_field(history),
        "memory census"
    );
    for s in store_gauges() {
        log_store(&s);
    }
}

/// One store's structures, as its persistence pass last gauged them: what the
/// census line's untracked figure is set against when the verdict is an owner.
fn log_store(s: &StoreGauge) {
    let top_objects = s
        .substrate
        .custom_objects
        .iter()
        .take(3)
        .map(|p| {
            format!(
                "{}: {} live, {} ceased, {} MiB",
                p.prefix,
                p.live,
                p.ceased,
                mib(p.bytes)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let top_applied = s
        .persistence
        .applied
        .iter()
        .take(3)
        .map(|a| {
            format!(
                "{:?}: {} records, {} MiB",
                a.record_type,
                a.count,
                mib(a.payload_bytes)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    tracing::info!(
        target: "host::memory",
        store = %s.dir,
        gauge_us = s.took_us,
        location_index_mib = mib(s.persistence.location_index_bytes),
        location_slots = s.persistence.location_index.iter().map(|o| o.len).sum::<u64>(),
        location_held = s.persistence.location_index.iter().map(|o| o.held).sum::<u64>(),
        accounting_entries = s.persistence.accounting_entries,
        streams = s.substrate.streams,
        chunk_locations = s.substrate.chunk_locations,
        pending_records = s.substrate.pending.records,
        pending_kib = s.substrate.pending.payload_bytes / 1024,
        residence_slots = s.substrate.residences.slots,
        residence_dead = s.substrate.residences.dead,
        custom_objects = %top_objects,
        applied = %top_applied,
        "store census"
    );
}

fn log_growing(g: &Growth, untracked: Option<u64>, previous_untracked: Option<u64>) {
    let (verdict, classes) = match &g.verdict {
        Verdict::Retention => ("retention", String::new()),
        Verdict::OutsideHeap => ("outside the heap", String::new()),
        Verdict::NotHeld => ("not held by our code", String::new()),
        Verdict::Owner { classes } => (
            "owner",
            classes
                .iter()
                .map(|c| format!("≤{} B: +{} MiB", c.upto_bytes, mib_i(c.grew_bytes)))
                .collect::<Vec<_>>()
                .join(", "),
        ),
        Verdict::Steady => return,
    };
    tracing::warn!(
        target: "host::memory",
        verdict,
        over_min = g.over_ms / 60_000,
        private_grew_mib = mib_i(g.private_grew),
        heap_committed_grew_mib = g.committed_grew.map(mib_i),
        heap_live_grew_mib = g.live_grew.map(mib_i),
        untracked_mib = untracked.map(mib),
        untracked_was_mib = previous_untracked.map(mib),
        grown_classes = %classes,
        "memory growing"
    );
}

fn log_commit(c: &MachineCommit, level: CommitLevel, growth: Option<&Growth>) {
    let pct = (c.committed_bytes as f64 * 100.0 / c.limit_bytes.max(1) as f64).round();
    let hours = growth.and_then(|g| hours_to_limit(c, g.private_grew, g.over_ms));
    let hours = hours.map(|h| (h * 10.0).round() / 10.0);
    match level {
        CommitLevel::Critical => tracing::error!(
            target: "host::memory",
            committed_mib = mib(c.committed_bytes),
            limit_mib = mib(c.limit_bytes),
            pct,
            hours_to_limit = hours,
            "the machine's commit is nearly at its limit: the next allocation past it fails the process"
        ),
        _ => tracing::warn!(
            target: "host::memory",
            committed_mib = mib(c.committed_bytes),
            limit_mib = mib(c.limit_bytes),
            pct,
            hours_to_limit = hours,
            "the machine's commit is close to its limit"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A reading is taken and kept**: the commit charge, the heap's summary
    /// where the platform gives one, and the machine's commit.
    #[test]
    #[cfg(any(windows, target_os = "linux"))]
    fn a_reading_is_taken_and_kept() {
        read_once();
        let now = latest().expect("a reading was taken");
        assert!(now.at_unix_ms > 0);
        assert!(now.os_heap.is_some(), "this platform summarises its heap");
        assert!(now.untracked_bytes.is_some());
        assert!(now.machine_commit_limit_bytes.is_some_and(|l| l > 0));
        assert!(now.growth.is_none(), "one reading is no window");
    }
}
