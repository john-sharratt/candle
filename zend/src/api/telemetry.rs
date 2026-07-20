//! `GET /v1/telemetry` + `GET /v1/phases` — live wave / pressure / phase series
//! for the performance dashboard (`/perf.html`).
//!
//! Both endpoints read the scheduler's **in-process** telemetry rings (see
//! `candle_conversation::phase_ring`) — the wave summary + arena split + hot→warm
//! migration timings are pushed straight from the scheduler and persistence
//! threads. Nothing is parsed out of the log, so the dashboard works regardless of
//! log level and pays no file I/O. Sample times are returned as elapsed seconds
//! within the returned window (oldest = 0), so the page's "now on the right" axis
//! places the newest sample at the right edge.

use std::sync::{Arc, Mutex, OnceLock};

use axum::{extract::State, Json};
use serde::Serialize;

use crate::session::ZendSession;

#[derive(Serialize, Default, Clone)]
pub struct Telemetry {
    waves: Vec<Wave>,
    fmt: Vec<Fmt>,
    migrate: Vec<Migrate>,
    host: Host,
    /// `true` once any arena / migration sample exists — the pressure panels have
    /// data. Always reachable now (instrumented), unlike the old log-level gate.
    has_pressure: bool,
}

#[derive(Serialize, Clone)]
struct Wave {
    t: f64,
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
#[derive(Serialize, Default, Clone)]
struct Host {
    free_gib: f64,
    total_gib: f64,
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
        let points = candle_conversation::phase_ring::snapshot()
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

/// `GET /v1/telemetry` — wave / pressure / arena / migration series, all from the
/// instrumented rings. Host RAM is stamped fresh each call. The ring clones +
/// `sysinfo` refresh are bounded blocking work, kept off the async workers.
pub async fn telemetry(State(_session): State<Arc<ZendSession>>) -> Json<Telemetry> {
    let body = tokio::task::spawn_blocking(|| {
        let mut b = build_body();
        b.host = host_memory();
        b
    })
    .await
    .unwrap_or_default();
    Json(body)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Assemble the series from the rings. Ages (seconds before now) are converted to
/// an ascending elapsed `t` (oldest = 0) against a reference = the oldest sample
/// across both rings, so every series shares one time origin and the newest sample
/// lands at the right edge of the page's backwards axis.
fn build_body() -> Telemetry {
    use candle_conversation::phase_ring::{snapshot_migrates, snapshot_waves};
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
                ws: (s.ws_ms as f64) / 1000.0,
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
    Telemetry {
        waves: wave_series,
        fmt: fmt_series,
        migrate: migrate_series,
        host: Host::default(),
        has_pressure,
    }
}

fn host_memory() -> Host {
    use sysinfo::System;
    // Reuse one System across requests — refreshing memory on it is far cheaper
    // than allocating a fresh System per poll.
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();
    let mut sys = SYS
        .get_or_init(|| Mutex::new(System::new()))
        .lock()
        .unwrap();
    sys.refresh_memory();
    let gib = |b: u64| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    Host {
        free_gib: (gib(sys.available_memory()) * 10.0).round() / 10.0,
        total_gib: (gib(sys.total_memory()) * 10.0).round() / 10.0,
    }
}
