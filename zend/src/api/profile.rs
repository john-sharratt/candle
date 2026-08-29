//! `GET /v1/profile` — the hot-path span breakdown for the window since the
//! last call.
//!
//! **Reading resets.** Two consecutive calls therefore bracket an interval
//! exactly, which is the point: an average since process start buries the thing
//! worth finding, and a rate that collapses for twenty seconds in the middle of
//! an ingest is invisible in a cumulative total. Poll once to mark the start,
//! again to close it, and the second response describes only what happened
//! between them.
//!
//! The totals come from the process-wide pipeline profiler in
//! `candle_transformers::models::profile`, which three sets of spans feed:
//! the scheduler's main-loop housekeeping (`loop:*`), the model's host-side
//! stages, and the GPU spans, whose device time is bracketed by enqueued CUDA
//! events rather than by draining the device — so profiling a wave does not
//! serialise it.
//!
//! Two caveats worth knowing before reading a table:
//!
//! * **GPU totals lag by up to one wave.** A GPU span becomes a number only when
//!   its events are harvested, and harvesting happens on the thread that
//!   recorded them (the event pool is per-thread) at the scheduler's wave
//!   boundary — not here. A span still in flight at the moment of the call is
//!   counted in the *next* window.
//! * **Host and GPU spans overlap.** A `loop:decode` that contains the kernels
//!   its forwards enqueued counts their time too, so the columns do not sum to
//!   the wall clock and are not meant to. Compare like with like: host against
//!   host to find where the loop sits, GPU against GPU to find which kernel owns
//!   the device.
//!
//! Without `--features profile` every span is compiled out; the endpoint then
//! reports `enabled: false` rather than an empty table that would read as "the
//! hot path costs nothing".

use std::sync::Mutex;
use std::time::Instant;

use axum::Json;
use candle_conversation::profile::pipeline_snapshot_and_reset;
use serde::Serialize;

/// When the window now being closed was opened. `None` until the first call,
/// whose window therefore starts at process start and is reported as such.
static WINDOW_START: Mutex<Option<Instant>> = Mutex::new(None);

#[derive(Serialize)]
pub struct Profile {
    /// `false` when the binary was built without `--features profile`, in which
    /// case `entries` is empty because nothing was measured — not because
    /// nothing took time.
    enabled: bool,
    /// Wall-clock seconds this table covers, or `null` for the first call after
    /// startup (the window opened before anyone was watching).
    window_secs: Option<f64>,
    /// Spans, largest total first.
    entries: Vec<Entry>,
}

#[derive(Serialize)]
struct Entry {
    name: String,
    total_ms: f64,
    count: u64,
    avg_ms: f64,
    /// Share of this table's total, so the dominant span is obvious without
    /// dividing by hand. Host and GPU spans overlap, so these sum past 100%.
    pct: f64,
}

pub async fn profile() -> Json<Profile> {
    let now = Instant::now();
    let window_secs = WINDOW_START
        .lock()
        .ok()
        .and_then(|mut w| w.replace(now).map(|prev| now.duration_since(prev)))
        .map(|d| d.as_secs_f64());

    let snap = pipeline_snapshot_and_reset();
    let total: f64 = snap.entries.iter().map(|(_, ms, _)| *ms).sum();
    let mut entries: Vec<Entry> = snap
        .entries
        .into_iter()
        .map(|(name, total_ms, count)| Entry {
            name,
            total_ms,
            count,
            avg_ms: total_ms / (count as f64).max(1.0),
            pct: if total > 0.0 {
                total_ms / total * 100.0
            } else {
                0.0
            },
        })
        .collect();
    entries.sort_by(|a, b| b.total_ms.total_cmp(&a.total_ms));

    Json(Profile {
        enabled: cfg!(feature = "profile"),
        window_secs,
        entries,
    })
}
