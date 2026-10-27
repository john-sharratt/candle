//! The engine's per-wave series — the wave, arena-mix and hot→warm migration
//! rows of `GET /v1/telemetry` — and `GET /v1/phases`, `/v1/projections` and
//! `/v1/layers`.
//!
//! All read the scheduler's **in-process** telemetry rings (see
//! `candle_conversation::phase_ring`) — the wave summary + arena split + hot→warm
//! migration timings are pushed straight from the scheduler and persistence
//! threads. One scheduler serves every tenant, so these are the engine's, not
//! any one tenant's. Nothing is parsed out of the log, so the dashboard works
//! regardless of log level and pays no file I/O.
//!
//! Every series is answered on the shared wall-clock grid
//! (`docs/performance_series.md`): one row per finished `slots::SLOT_MS` slot,
//! stamped `at` with the slot's start in unix milliseconds, and each answer
//! declares in `fuse` how its rows add up across a cluster — which the gateway
//! does (`cluster::dispatch::fuse`), so no console adds anything up itself.

use std::time::Instant;

use axum::Json;
use candle_conversation::layer_trace::{self, PassTrace};
use candle_conversation::phase_ring::{
    snapshot, snapshot_migrates, snapshot_mixes, snapshot_projected, snapshot_waves, MigrateSample,
    PhaseSnapshot, VramBands, WaveSample,
};
use serde::Serialize;
use serde_json::{json, Value};

use super::format_mix::{mix_series, Census, MixRow};
use super::slots::{by_slot, now_ms};

/// The engine's rows, slot by slot.
#[derive(Serialize, Default, Clone)]
pub struct WaveSeries {
    waves: Vec<Wave>,
    /// The resident KV by storage format, each row's tokens by series name.
    mix: Vec<MixRow>,
    /// The arena's series in stacking order: the top formats, then `other`.
    mix_formats: Vec<String>,
    migrate: Vec<Migrate>,
    /// `true` once any arena / migration sample exists — the pressure panels
    /// have data.
    has_pressure: bool,
    /// How the answer adds up across a cluster ([`telemetry_fuse`]).
    fuse: Value,
}

/// The formats the arena panel names; the rest are `other`.
const MIX_TOP: usize = 6;

/// How `/v1/telemetry`'s answer adds up across a cluster: each slot's rows add,
/// but a wave's and a migration pass's durations are the longest member's —
/// the cards ran side by side — so the cluster's tokens over its `ws` is the
/// cluster's rate. Host memory adds; the arena's series are every member's.
fn telemetry_fuse() -> Value {
    json!({
        "series": {
            "waves": ["ws", "fwd", "wave_ms", "quanta_ms"],
            "migrate": ["migrate", "quant", "copy", "total"],
            "mix": [],
        },
        "sum": ["host"],
        "union": ["mix_formats"],
    })
}

/// How `/v1/phases`' answer adds up: each kind's tokens, bytes, sequences,
/// operations and peak rate add; its time and the slot's window are the
/// longest member's.
fn phases_fuse() -> Value {
    json!({ "series": { "points": ["dur", "win"] } })
}

#[derive(Serialize, Clone, Debug, PartialEq)]
struct Wave {
    /// The slot's start, unix milliseconds.
    at: u64,
    /// The slot's waves' wall-clock, summed, and the longest mean forward,
    /// **both milliseconds** — as is every other duration on this record
    /// (`pdrain`, `delev`, `dglue`, `reproj`, `scan`, `rglue`, `layout`, each
    /// the slot's sum). The page divides `tok` by `ws` for throughput and
    /// plots the two on one axis, so a unit that disagrees with the name is a
    /// silently wrong reading rather than an error.
    ws: f64,
    fwd: f64,
    /// Tokens the slot's waves fed.
    tok: f64,
    // The slot's last wave's state: the KV budget and use, the backlog.
    budget: f64,
    used: f64,
    backlog: f64,
    /// The whole card at the slot's last wave, cut by what holds each byte
    /// (MiB).
    vram: Vram,
    // Projection decomposition (ms): pdrain (drain minus re-attributed prefill) +
    // drain elevate/glue; reproject total + scan/glue/layout.
    pdrain: f64,
    delev: f64,
    dglue: f64,
    reproj: f64,
    scan: f64,
    rglue: f64,
    layout: f64,
    // Active-work counts at the slot's last wave: resident conversations +
    // decode/prefill/section widths.
    slots: f64,
    dec: f64,
    pre: f64,
    sec: f64,
    /// The slot's waves — run-loop passes that ran a forward — and, per wave,
    /// its mean wall-clock and the part spent in the quanta (ms). Both `null`
    /// for a slot with no wave.
    waves: f64,
    wave_ms: Option<f64>,
    quanta_ms: Option<f64>,
}

/// One slot's wave samples, oldest first, as its row; none for no sample.
fn fold_waves(at: u64, rows: &[&WaveSample]) -> Option<Wave> {
    let last = *rows.last()?;
    let sum = |f: fn(&WaveSample) -> u64| rows.iter().map(|s| f(s)).sum::<u64>();
    let waves = sum(|s| s.waves);
    let (wave_ms, quanta_ms) =
        per_wave(waves, sum(|s| s.wave_wall_us), sum(|s| s.wave_overhead_us));
    Some(Wave {
        at,
        ws: sum(|s| s.ws_ms as u64) as f64,
        fwd: rows.iter().map(|s| s.fwd_ms).max().unwrap_or(0) as f64,
        tok: sum(|s| s.tok) as f64,
        budget: last.budget_mib as f64,
        used: last.used_mib as f64,
        backlog: last.backlog as f64,
        vram: last.vram.into(),
        pdrain: sum(|s| s.pdrain_ms) as f64,
        delev: sum(|s| s.drain_elevate_ms) as f64,
        dglue: sum(|s| s.drain_glue_ms) as f64,
        reproj: sum(|s| s.reproj_ms) as f64,
        scan: sum(|s| s.reproj_scan_ms) as f64,
        rglue: sum(|s| s.reproj_glue_ms) as f64,
        layout: sum(|s| s.reproj_layout_ms) as f64,
        slots: last.slots as f64,
        dec: last.decodes as f64,
        pre: last.prefills as f64,
        sec: last.sections as f64,
        waves: waves as f64,
        wave_ms,
        quanta_ms,
    })
}

/// `(mean wall, mean quanta)` per wave, in ms, from the window's summed
/// microseconds — `None` for a window with no wave.
fn per_wave(waves: u64, wall_us: u64, overhead_us: u64) -> (Option<f64>, Option<f64>) {
    if waves == 0 {
        return (None, None);
    }
    let ms = |us: u64| round2(us as f64 / 1000.0 / waves as f64);
    (
        Some(ms(wall_us)),
        Some(ms(wall_us.saturating_sub(overhead_us))),
    )
}
/// One wave's card, in MiB — every band but `total` adds up to `total`
/// (`candle_conversation::phase_ring::VramBands`).
#[derive(Serialize, Clone, Debug, PartialEq)]
struct Vram {
    total: f64,
    free: f64,
    context: f64,
    pool: f64,
    dense: f64,
    experts: f64,
    expert_free: f64,
    kv: f64,
    state: f64,
    transient: f64,
    reserved: f64,
    kv_free: f64,
    unaccounted: f64,
}

impl From<VramBands> for Vram {
    fn from(b: VramBands) -> Self {
        Self {
            total: b.total,
            free: b.free,
            context: b.context,
            pool: b.pool,
            dense: b.dense,
            experts: b.experts,
            expert_free: b.expert_free,
            kv: b.kv,
            state: b.state,
            transient: b.transient,
            reserved: b.reserved,
            kv_free: b.kv_free,
            unaccounted: b.unaccounted,
        }
    }
}

/// One slot's hot→warm migration passes: what they moved, and the time they
/// took (ms), summed.
#[derive(Serialize, Clone, Debug, PartialEq)]
struct Migrate {
    /// The slot's start, unix milliseconds.
    at: u64,
    res: f64,
    mib: f64,
    migrate: f64,
    quant: f64,
    copy: f64,
    total: f64,
}

/// One slot's passes, as its row.
fn fold_migrates(at: u64, rows: &[&MigrateSample]) -> Migrate {
    let sum = |f: fn(&MigrateSample) -> u64| rows.iter().map(|s| f(s)).sum::<u64>() as f64;
    Migrate {
        at,
        res: sum(|s| s.residences),
        mib: sum(|s| s.mib),
        migrate: sum(|s| s.migrate_ms),
        quant: sum(|s| s.quantize_ms),
        copy: sum(|s| s.copy_ms),
        total: sum(|s| s.total_ms),
    }
}

/// One phase over one slot, for the phase graphs: `kind` the phase label,
/// `dur` the wall-clock (ms) it held across the slot's waves, `win` those
/// waves' windows (ms) summed, `tok`/`by` the token / byte volume, `seq` the
/// most sequences one wave carried, `n` the operation count (forwards / seals
/// / evicted residences), and `peak` the best of the slot's waves' delivered
/// rate — the phase's tokens over that wave's window, tokens a second.
#[derive(Serialize, Clone, Debug, PartialEq)]
struct PhasePoint {
    /// The slot's start, unix milliseconds.
    at: u64,
    kind: &'static str,
    dur: u64,
    win: u64,
    tok: u64,
    by: u64,
    seq: u32,
    n: u64,
    peak: f64,
}

/// One slot's phase measurements, oldest first, as a row per phase in the
/// order the phases first ran. A wave's measurements share its age, which is
/// how its window is counted once.
fn fold_points(at: u64, points: &[&PhaseSnapshot]) -> Vec<PhasePoint> {
    let mut win = 0u64;
    let mut wave: Option<f64> = None;
    let mut out: Vec<PhasePoint> = Vec::new();
    for p in points {
        if wave != Some(p.age_s) {
            wave = Some(p.age_s);
            win += u64::from(p.window_ms);
        }
        let i = match out.iter().position(|q| q.kind == p.kind) {
            Some(i) => i,
            None => {
                out.push(PhasePoint {
                    at,
                    kind: p.kind,
                    dur: 0,
                    win: 0,
                    tok: 0,
                    by: 0,
                    seq: 0,
                    n: 0,
                    peak: 0.0,
                });
                out.len() - 1
            }
        };
        let q = &mut out[i];
        q.dur += u64::from(p.dur_ms);
        q.tok += p.tokens;
        q.by += p.bytes;
        q.seq = q.seq.max(p.seqs);
        q.n += u64::from(p.count);
        if p.window_ms > 0 {
            q.peak = q
                .peak
                .max(round2(p.tokens as f64 * 1000.0 / f64::from(p.window_ms)));
        }
    }
    for q in &mut out {
        q.win = win;
    }
    out
}

#[derive(Serialize, Default, Clone)]
pub struct Phases {
    points: Vec<PhasePoint>,
    /// How the answer adds up across a cluster ([`phases_fuse`]).
    fuse: Value,
}

/// `GET /v1/phases` — the scheduler's in-process per-wave phase ring, one row
/// per phase per finished slot. Feeds the phase bars (2 min) and the phase
/// throughput graph (20 min) on `perf.html`. The ring snapshot clones a
/// bounded but non-trivial vector under a lock, so it runs on a blocking
/// thread.
pub async fn phases() -> Json<Phases> {
    let body = tokio::task::spawn_blocking(|| {
        let snap = snapshot();
        let points = by_slot(&snap, now_ms(), |p| p.age_s)
            .into_iter()
            .flat_map(|(at, group)| fold_points(at, &group))
            .collect();
        Phases {
            points,
            fuse: phases_fuse(),
        }
    })
    .await
    .unwrap_or_default();
    Json(body)
}

#[derive(Serialize, Clone, Debug, PartialEq)]
struct Projected {
    /// `section` or `turns`.
    kind: &'static str,
    /// The section's name, or the conversation's sidebar label.
    name: String,
    /// Working sets that had to lift it back into VRAM.
    lifts: u64,
    /// Turns lifted across those; one per lift for a section.
    units: u64,
    /// Seconds since its last lift.
    age: f64,
}
#[derive(Serialize, Default, Clone)]
pub struct Projections {
    items: Vec<Projected>,
}

/// Most-lifted first; among equals, the most recent.
fn rank(mut items: Vec<Projected>) -> Vec<Projected> {
    items.sort_by(|a, b| b.lifts.cmp(&a.lifts).then(a.age.total_cmp(&b.age)));
    items
}

/// `GET /v1/projections` — what projection had to lift back into VRAM for a
/// working set, by name, most-lifted first: the live "what keeps coming back"
/// board on `perf.html`.
pub async fn projections() -> Json<Projections> {
    let items = snapshot_projected()
        .into_iter()
        .map(|p| Projected {
            kind: p.kind.as_str(),
            name: p.name,
            lifts: p.lifts,
            units: p.units,
            age: round2(p.age_s),
        })
        .collect();
    Json(Projections { items: rank(items) })
}

/// One routed layer of a forward, as the page draws it.
#[derive(Serialize, Clone, Debug, PartialEq)]
struct Layer {
    row: usize,
    hot: u32,
    warm: u32,
    cold: u32,
    /// Milliseconds from the previous layer's routing to this one's; `null`
    /// for a pass's first layer.
    ms: Option<f64>,
    /// The time is the pipeline thread's pace, not the device's
    /// (`LayerVisit::late`).
    late: bool,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
struct Pass {
    kind: &'static str,
    /// Seconds before the answer.
    age: f64,
    layers: Vec<Layer>,
}

#[derive(Serialize, Default, Clone)]
pub struct Layers {
    passes: Vec<Pass>,
}

fn pass_of(t: &PassTrace, now: Instant) -> Pass {
    Pass {
        kind: if t.prefill { "prefill" } else { "decode" },
        age: round2(now.saturating_duration_since(t.at).as_secs_f64()),
        layers: t
            .layers
            .iter()
            .map(|l| Layer {
                row: l.row,
                hot: l.hot,
                warm: l.warm,
                cold: l.cold,
                ms: l.secs.map(|s| (s * 1e5).round() / 100.0),
                late: l.late,
            })
            .collect(),
    }
}

/// `GET /v1/layers` — the newest decode and prefill forward, layer by layer:
/// the experts each layer routed by where they stood (VRAM, pinned host, cold
/// pack) and the device's time between consecutive layers. Empty on a model
/// with no expert cache.
pub async fn layers() -> Json<Layers> {
    let now = Instant::now();
    let passes = layer_trace::latest()
        .iter()
        .map(|t| pass_of(t, now))
        .collect();
    Json(Layers { passes })
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Assemble the series from the rings, one row per finished `slots::SLOT_MS` slot,
/// stamped with the slot's start on the wall clock — so a row lands where it
/// ran however many polls follow, and every member's rows line up. The ring
/// clones are bounded but not free, so callers run this off the async workers.
pub fn wave_series() -> WaveSeries {
    let now = now_ms();
    let waves = snapshot_waves();
    let wave_rows: Vec<Wave> = by_slot(&waves, now, |w| w.age_s)
        .into_iter()
        .filter_map(|(at, group)| {
            let samples: Vec<&WaveSample> = group.iter().map(|w| &w.sample).collect();
            fold_waves(at, &samples)
        })
        .collect();
    // A census is a state: the slot's last one stands for it.
    let mixes = snapshot_mixes();
    let censuses: Vec<Census> = by_slot(&mixes, now, |m| m.age_s)
        .into_iter()
        .filter_map(|(at, group)| {
            group.last().map(|m| Census {
                at,
                rows: m.rows.clone(),
            })
        })
        .collect();
    let mix = mix_series(&censuses, MIX_TOP);
    let migrates = snapshot_migrates();
    let migrate_rows: Vec<Migrate> = by_slot(&migrates, now, |m| m.age_s)
        .into_iter()
        .map(|(at, group)| {
            let samples: Vec<&MigrateSample> = group.iter().map(|m| &m.sample).collect();
            fold_migrates(at, &samples)
        })
        .collect();

    let has_pressure = !mix.rows.is_empty() || !migrate_rows.is_empty();
    WaveSeries {
        waves: wave_rows,
        mix: mix.rows,
        mix_formats: mix.formats,
        migrate: migrate_rows,
        has_pressure,
        fuse: telemetry_fuse(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use candle_conversation::layer_trace::LayerVisit;
    use candle_conversation::phase_ring::{migrate_sample, wave_sample};

    /// The card's bands reach the page by name, every one of them.
    #[test]
    fn the_card_reaches_the_page_band_by_band() {
        let v = serde_json::to_value(Vram::from(VramBands {
            total: 100.0,
            free: 10.0,
            context: 5.0,
            pool: 1.0,
            dense: 20.0,
            experts: 30.0,
            expert_free: 4.0,
            kv: 15.0,
            state: 3.0,
            transient: 2.0,
            reserved: 6.0,
            kv_free: 3.5,
            unaccounted: 0.5,
        }))
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "total": 100.0, "free": 10.0, "context": 5.0, "pool": 1.0,
                "dense": 20.0, "experts": 30.0, "expert_free": 4.0, "kv": 15.0,
                "state": 3.0, "transient": 2.0, "reserved": 6.0, "kv_free": 3.5,
                "unaccounted": 0.5,
            })
        );
    }

    /// A window's summed microseconds read as each wave's mean wall-clock and
    /// the part of it in the quanta; a window with no wave has neither.
    #[test]
    fn a_window_reads_as_mean_wall_and_quanta_per_wave() {
        assert_eq!(per_wave(4, 200_000, 40_000), (Some(50.0), Some(40.0)));
        assert_eq!(per_wave(0, 0, 0), (None, None));
    }

    /// Most-lifted first; a tie goes to the most recently lifted.
    #[test]
    fn projections_rank_by_lifts_then_recency() {
        let p = |name: &str, lifts: u64, age: f64| Projected {
            kind: "turns",
            name: name.into(),
            lifts,
            units: lifts,
            age,
        };
        let ranked: Vec<String> = rank(vec![p("a", 2, 9.0), p("b", 5, 30.0), p("c", 2, 1.0)])
            .into_iter()
            .map(|x| x.name)
            .collect();
        assert_eq!(ranked, vec!["b", "c", "a"]);
    }

    /// A pass reaches the page layer by layer, its gaps in milliseconds and
    /// its age from the answer.
    #[test]
    fn a_pass_reaches_the_page_layer_by_layer() {
        let at = Instant::now();
        let t = PassTrace {
            prefill: true,
            at,
            layers: vec![
                LayerVisit {
                    row: 0,
                    hot: 6,
                    warm: 1,
                    cold: 1,
                    secs: None,
                    late: false,
                },
                LayerVisit {
                    row: 1,
                    hot: 8,
                    warm: 0,
                    cold: 0,
                    secs: Some(0.0042),
                    late: true,
                },
            ],
        };
        assert_eq!(
            pass_of(&t, at + Duration::from_millis(1500)),
            Pass {
                kind: "prefill",
                age: 1.5,
                layers: vec![
                    Layer {
                        row: 0,
                        hot: 6,
                        warm: 1,
                        cold: 1,
                        ms: None,
                        late: false
                    },
                    Layer {
                        row: 1,
                        hot: 8,
                        warm: 0,
                        cold: 0,
                        ms: Some(4.2),
                        late: true
                    },
                ],
            }
        );
    }

    /// With no engine in this process the rings are empty, and the series say
    /// so rather than inventing a row — and still say how they add up.
    #[test]
    fn no_engine_is_empty_series() {
        let s = serde_json::to_value(wave_series()).unwrap();
        assert_eq!(
            s,
            serde_json::json!({
                "waves": [],
                "mix": [],
                "mix_formats": [],
                "migrate": [],
                "has_pressure": false,
                "fuse": {
                    "series": {
                        "waves": ["ws", "fwd", "wave_ms", "quanta_ms"],
                        "migrate": ["migrate", "quant", "copy", "total"],
                        "mix": [],
                    },
                    "sum": ["host"],
                    "union": ["mix_formats"],
                },
            })
        );
    }

    /// **A slot's waves are one row**: their wall-clock, tokens and projection
    /// time add; the longest mean forward stands; the state — budget, backlog,
    /// card, widths — is the last wave's; and the per-wave means come from the
    /// summed microseconds over all 22 waves.
    #[test]
    fn a_slots_waves_fold_into_one_row() {
        let first = wave_sample(
            2_000,
            30,
            500,
            100,
            50,
            10,
            VramBands::default(),
            (1, 2, 3, 4, 5, 6, 7),
            (5, 2, 1, 0),
            (10, 400_000, 100_000),
        );
        let card = VramBands {
            kv: 9.0,
            ..VramBands::default()
        };
        let second = wave_sample(
            2_500,
            45,
            700,
            110,
            60,
            20,
            card,
            (10, 20, 30, 40, 50, 60, 70),
            (6, 3, 2, 1),
            (12, 600_000, 200_000),
        );
        assert_eq!(
            fold_waves(5_000, &[&first, &second]),
            Some(Wave {
                at: 5_000,
                ws: 4_500.0,
                fwd: 45.0,
                tok: 1_200.0,
                budget: 110.0,
                used: 60.0,
                backlog: 20.0,
                vram: card.into(),
                pdrain: 11.0,
                delev: 22.0,
                dglue: 33.0,
                reproj: 44.0,
                scan: 55.0,
                rglue: 66.0,
                layout: 77.0,
                slots: 6.0,
                dec: 3.0,
                pre: 2.0,
                sec: 1.0,
                waves: 22.0,
                wave_ms: Some(45.45),
                quanta_ms: Some(31.82),
            })
        );
        assert_eq!(fold_waves(5_000, &[]), None);
    }

    /// **A slot's phases are a row each**, in the order they first ran: each
    /// wave's window counted once into the slot's (2,000 + 2,500 ms), and the
    /// decode's peak the better wave's — 500 tokens over 2 s, not 600 over
    /// 2.5 s.
    #[test]
    fn a_slots_phases_fold_into_a_row_each_with_the_best_waves_rate() {
        let point = |kind, age_s, window_ms, dur_ms, tokens, seqs, count| PhaseSnapshot {
            kind,
            age_s,
            window_ms,
            dur_ms,
            tokens,
            bytes: 0,
            seqs,
            count,
        };
        let a_decode = point("decode", 3.0, 2_000, 1_500, 500, 20, 15);
        let a_prefill = point("prefill", 3.0, 2_000, 400, 4_000, 2, 3);
        let b_decode = point("decode", 1.0, 2_500, 2_000, 600, 24, 16);
        assert_eq!(
            fold_points(10_000, &[&a_decode, &a_prefill, &b_decode]),
            vec![
                PhasePoint {
                    at: 10_000,
                    kind: "decode",
                    dur: 3_500,
                    win: 4_500,
                    tok: 1_100,
                    by: 0,
                    seq: 24,
                    n: 31,
                    peak: 250.0,
                },
                PhasePoint {
                    at: 10_000,
                    kind: "prefill",
                    dur: 400,
                    win: 4_500,
                    tok: 4_000,
                    by: 0,
                    seq: 2,
                    n: 3,
                    peak: 2_000.0,
                },
            ]
        );
    }

    /// A slot's migration passes add up, what they moved and how long they took.
    #[test]
    fn a_slots_migration_passes_add_up() {
        let a = migrate_sample(3, 10, 100, 40, 50, 120);
        let b = migrate_sample(2, 5, 200, 10, 150, 210);
        assert_eq!(
            fold_migrates(15_000, &[&a, &b]),
            Migrate {
                at: 15_000,
                res: 5.0,
                mib: 15.0,
                migrate: 300.0,
                quant: 50.0,
                copy: 200.0,
                total: 330.0,
            }
        );
    }
}
