//! The engine's per-wave series — the wave, arena-format and hot→warm migration
//! rows of `GET /v1/telemetry` — and `GET /v1/phases` and `/v1/promotes`.
//!
//! All read the scheduler's **in-process** telemetry rings (see
//! `candle_conversation::phase_ring`) — the wave summary + arena split + hot→warm
//! migration timings are pushed straight from the scheduler and persistence
//! threads. One scheduler serves every tenant, so these are the engine's, not
//! any one tenant's. Nothing is parsed out of the log, so the dashboard works
//! regardless of log level and pays no file I/O. Sample times are returned as
//! elapsed seconds within the returned window (oldest = 0), so the page's "now
//! on the right" axis places the newest sample at the right edge.

use std::cmp::Reverse;

use axum::Json;
use candle_conversation::phase_ring::{
    snapshot, snapshot_migrates, snapshot_promotes, snapshot_waves,
};
use serde::Serialize;

/// The per-wave rows.
#[derive(Serialize, Default, Clone)]
pub struct WaveSeries {
    waves: Vec<Wave>,
    fmt: Vec<Fmt>,
    migrate: Vec<Migrate>,
    /// `true` once any arena / migration sample exists — the pressure panels
    /// have data.
    has_pressure: bool,
}

#[derive(Serialize, Clone)]
struct Wave {
    t: f64,
    /// Wave wall-clock and forward time, **both milliseconds** — as are every
    /// other duration on this record (`pdrain`, `delev`, `dglue`, `reproj`,
    /// `scan`, `rglue`, `layout`). The page divides `tok` by `ws` for
    /// throughput and plots the two on one axis, so a unit that disagrees with
    /// the name is a silently wrong reading rather than an error.
    ws: f64,
    fwd: f64,
    tok: f64,
    budget: f64,
    used: f64,
    backlog: f64,
    // Whole-card VRAM decomposition (MiB): KV-pool reserved, driver total, free,
    // and the KV quant/float reserved split (so the decomp chart reads from one
    // series).
    res: f64,
    tot: f64,
    free: f64,
    qkv: f64,
    fkv: f64,
    // Model weights resident in the pool (MiB), captured at startup — the
    // whole-card decomp carves these out of `slack` so the weights band is real.
    wt: f64,
    // Projection decomposition (ms): pdrain (drain minus re-attributed prefill) +
    // drain elevate/glue; reproject total + scan/glue/layout.
    pdrain: f64,
    delev: f64,
    dglue: f64,
    reproj: f64,
    scan: f64,
    rglue: f64,
    layout: f64,
    // Active-work counts: resident conversations + decode/prefill/section widths.
    slots: f64,
    dec: f64,
    pre: f64,
    sec: f64,
}
#[derive(Serialize, Clone)]
struct Fmt {
    t: f64,
    fa: f64,
    fm: f64,
    fl: f64,
    qa: f64,
    qm: f64,
    ql: f64,
}
#[derive(Serialize, Clone)]
struct Migrate {
    t: f64,
    res: f64,
    mib: f64,
    migrate: f64,
    quant: f64,
    copy: f64,
    total: f64,
}
/// One phase measurement for the GUI phase graphs: `age` seconds before now,
/// `kind` the phase label, `dur` its wall-clock (ms) this wave, `win` the wave
/// window (ms), `tok`/`by` the token / byte volume, `seq` concurrent sequences,
/// `n` the operation count (forwards / seals / evicted residences).
#[derive(Serialize, Clone)]
struct PhasePoint {
    age: f64,
    kind: &'static str,
    dur: u32,
    win: u32,
    tok: u64,
    by: u64,
    seq: u32,
    n: u32,
}

#[derive(Serialize, Default, Clone)]
pub struct Phases {
    points: Vec<PhasePoint>,
}

/// `GET /v1/phases` — the scheduler's in-process per-wave phase ring, read fresh
/// each poll (ages relative to *now*). Feeds the phase bar (2 min) and phase line
/// (60 min) graphs on `perf.html`. The ring snapshot clones a bounded but
/// non-trivial vector under a lock, so it runs on a blocking thread.
pub async fn phases() -> Json<Phases> {
    let body = tokio::task::spawn_blocking(|| {
        let points = snapshot()
            .into_iter()
            .map(|p| PhasePoint {
                age: round2(p.age_s),
                kind: p.kind,
                dur: p.dur_ms,
                win: p.window_ms,
                tok: p.tokens,
                by: p.bytes,
                seq: p.seqs,
                n: p.count,
            })
            .collect();
        Phases { points }
    })
    .await
    .unwrap_or_default();
    Json(body)
}

#[derive(Serialize, Clone)]
struct Promote {
    name: String,
    count: u64,
}
#[derive(Serialize, Default, Clone)]
pub struct Promotes {
    items: Vec<Promote>,
}

/// `GET /v1/promotes` — cumulative promote (working-set elevate) counts by
/// symbolic name, most-promoted first. The live "what's being re-elevated every
/// turn" leaderboard on `perf.html`.
pub async fn promotes() -> Json<Promotes> {
    let mut items: Vec<Promote> = snapshot_promotes()
        .into_iter()
        .map(|(name, count)| Promote { name, count })
        .collect();
    items.sort_by_key(|p| Reverse(p.count));
    Json(Promotes { items })
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Assemble the series from the rings. Ages (seconds before now) are converted to
/// an ascending elapsed `t` (oldest = 0) against a reference = the oldest sample
/// across both rings, so every series shares one time origin and the newest sample
/// lands at the right edge of the page's backwards axis. The ring clones are
/// bounded but not free, so callers run this off the async workers.
pub fn wave_series() -> WaveSeries {
    let waves = snapshot_waves();
    let migr = snapshot_migrates();
    let ref_age = waves
        .iter()
        .map(|w| w.age_s)
        .chain(migr.iter().map(|m| m.age_s))
        .fold(0.0_f64, f64::max);
    let t_of = |age: f64| round2(ref_age - age);

    let wave_series: Vec<Wave> = waves
        .iter()
        .map(|w| {
            let s = &w.sample;
            Wave {
                t: t_of(w.age_s),
                ws: s.ws_ms as f64,
                fwd: s.fwd_ms as f64,
                tok: s.tok as f64,
                budget: s.budget_mib as f64,
                used: s.used_mib as f64,
                backlog: s.backlog as f64,
                res: s.reserved_mib as f64,
                tot: s.total_mib as f64,
                free: s.free_mib as f64,
                qkv: s.quant_mib as f64,
                fkv: s.float_mib as f64,
                wt: s.weights_mib as f64,
                pdrain: s.pdrain_ms as f64,
                delev: s.drain_elevate_ms as f64,
                dglue: s.drain_glue_ms as f64,
                reproj: s.reproj_ms as f64,
                scan: s.reproj_scan_ms as f64,
                rglue: s.reproj_glue_ms as f64,
                layout: s.reproj_layout_ms as f64,
                slots: s.slots as f64,
                dec: s.decodes as f64,
                pre: s.prefills as f64,
                sec: s.sections as f64,
            }
        })
        .collect();
    let fmt_series: Vec<Fmt> = waves
        .iter()
        .filter(|w| w.sample.float_arenas > 0 || w.sample.quant_arenas > 0)
        .map(|w| {
            let s = &w.sample;
            Fmt {
                t: t_of(w.age_s),
                fa: s.float_arenas as f64,
                fm: s.float_mib as f64,
                fl: s.float_live_mib as f64,
                qa: s.quant_arenas as f64,
                qm: s.quant_mib as f64,
                ql: s.quant_live_mib as f64,
            }
        })
        .collect();
    let migrate_series: Vec<Migrate> = migr
        .iter()
        .map(|m| {
            let s = &m.sample;
            Migrate {
                t: t_of(m.age_s),
                res: s.residences as f64,
                mib: s.mib as f64,
                migrate: s.migrate_ms as f64,
                quant: s.quantize_ms as f64,
                copy: s.copy_ms as f64,
                total: s.total_ms as f64,
            }
        })
        .collect();

    let has_pressure = !fmt_series.is_empty() || !migrate_series.is_empty();
    WaveSeries {
        waves: wave_series,
        fmt: fmt_series,
        migrate: migrate_series,
        has_pressure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no engine in this process the rings are empty, and the series say
    /// so rather than inventing a row.
    #[test]
    fn no_engine_is_empty_series() {
        let s = serde_json::to_value(wave_series()).unwrap();
        assert_eq!(
            s,
            serde_json::json!({"waves": [], "fmt": [], "migrate": [], "has_pressure": false})
        );
    }
}
