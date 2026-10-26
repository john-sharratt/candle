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
use candle_conversation::profile::{grow_tally, pipeline_snapshot_and_reset, spare_tally};
use serde::Serialize;

/// When the window now being closed was opened. `None` until the first call,
/// whose window therefore starts at process start and is reported as such.
static WINDOW_START: Mutex<Option<Instant>> = Mutex::new(None);

/// The boundary tallies as the last call read them, so a window reports what
/// moved within it — `None` until the first call, which reports since boot.
static BOUNDARY_AT: Mutex<Option<([u64; 7], [u64; 8])>> = Mutex::new(None);

/// What the elastic VRAM boundary answered over the window: the pool's answers
/// to the expert zone's growth question, and what the zone did with them.
///
/// **The spans say a decode step is expert reads; this says why the experts are
/// not resident.** A zone that never grows reads the same whether no ground was
/// free, the KV side was rising, or the ground was offered and the zone refused
/// it — and those are three different fixes. The forward gate prints these per
/// run; a daemon had no way to read them.
#[derive(Serialize)]
struct Boundary {
    /// Pool refusals: no workload measured yet.
    observing: u64,
    /// Pool refusals: KV demand rising, or the KV side bought ground since the
    /// last negotiation.
    pressure: u64,
    /// Pool refusals: the KV side holds the ground.
    occupied: u64,
    /// Pool refusals: free ground in holes below a live region.
    fragmented: u64,
    /// Regions the pool offered the weight side.
    granted_regions: u64,
    /// KV purchases from the weight side that it conceded, and that it refused.
    kv_purchases_conceded: u64,
    kv_purchases_refused: u64,
    /// The expert zone's side: growth questions asked, answered with nothing,
    /// regions offered, moves refused by the floor, asks skipped at the zone's
    /// limit, and the slots it gained.
    zone_asked: u64,
    zone_no_spare: u64,
    zone_offered_regions: u64,
    zone_target_unchanged: u64,
    zone_target_backwards: u64,
    zone_floor_refused: u64,
    zone_at_limit: u64,
    zone_gained_slots: u64,
}

fn boundary_window() -> Boundary {
    let now = (spare_tally(), grow_tally());
    let then = BOUNDARY_AT
        .lock()
        .ok()
        .and_then(|mut b| b.replace(now))
        .unwrap_or(([0; 7], [0; 8]));
    let s: [u64; 7] = std::array::from_fn(|i| now.0[i].saturating_sub(then.0[i]));
    let g: [u64; 8] = std::array::from_fn(|i| now.1[i].saturating_sub(then.1[i]));
    Boundary {
        observing: s[0],
        pressure: s[1],
        occupied: s[2],
        fragmented: s[6],
        granted_regions: s[3],
        kv_purchases_conceded: s[4],
        kv_purchases_refused: s[5],
        zone_asked: g[0],
        zone_no_spare: g[1],
        zone_offered_regions: g[2],
        zone_target_unchanged: g[3],
        zone_target_backwards: g[4],
        zone_floor_refused: g[5],
        zone_at_limit: g[6],
        zone_gained_slots: g[7],
    }
}

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
    /// What the VRAM boundary answered over the same window.
    boundary: Boundary,
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
        boundary: boundary_window(),
    })
}
