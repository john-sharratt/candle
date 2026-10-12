//! The engine's per-wave series — the wave, arena-mix and hot→warm migration
//! rows of `GET /v1/telemetry` — and `GET /v1/phases`, `/v1/projections` and
//! `/v1/layers`.
//!
//! All read the scheduler's **in-process** telemetry rings (see
//! `candle_conversation::phase_ring`) — the wave summary + arena split + hot→warm
//! migration timings are pushed straight from the scheduler and persistence
//! threads. One scheduler serves every tenant, so these are the engine's, not
//! any one tenant's. Nothing is parsed out of the log, so the dashboard works
//! regardless of log level and pays no file I/O. Sample times are returned as
//! elapsed seconds within the returned window (oldest = 0), so the page's "now
//! on the right" axis places the newest sample at the right edge.

use std::time::Instant;

use axum::Json;
use candle_conversation::layer_trace::{self, PassTrace};
use candle_conversation::phase_ring::{
    snapshot, snapshot_migrates, snapshot_mixes, snapshot_projected, snapshot_waves, VramBands,
};
use serde::Serialize;

use super::format_mix::{mix_series, Census, MixSeries};

/// The per-wave rows.
#[derive(Serialize, Default, Clone)]
pub struct WaveSeries {
    waves: Vec<Wave>,
    /// The resident KV by storage format, over time.
    mix: MixSeries,
    migrate: Vec<Migrate>,
    /// `true` once any arena / migration sample exists — the pressure panels
    /// have data.
    has_pressure: bool,
}

/// The formats the arena panel names; the rest are `other`.
const MIX_TOP: usize = 6;

#[derive(Serialize, Clone)]
struct Wave {
    /// Seconds before the answer — a page fixes each row's clock time from
    /// it once, so a row stays where it ran however many polls follow.
    age: f64,
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
    /// The whole card at this wave, cut by what holds each byte (MiB).
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
    // Active-work counts: resident conversations + decode/prefill/section widths.
    slots: f64,
    dec: f64,
    pre: f64,
    sec: f64,
    /// The window's waves — run-loop passes that ran a forward — and, per
    /// wave, its mean wall-clock and the part spent in the quanta (ms). Both
    /// `null` for a window with no wave.
    waves: f64,
    wave_ms: Option<f64>,
    quanta_ms: Option<f64>,
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

#[derive(Serialize, Clone)]
struct Migrate {
    /// Seconds before the answer, as [`Wave::age`].
    age: f64,
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

/// Assemble the series from the rings, each row stamped with its age in seconds
/// before this answer. Not an elapsed time from the oldest sample: that origin
/// moves as the rings trim, so every row slid along the page's axis on every
/// poll whether anything ran or not. The ring clones are bounded but not free,
/// so callers run this off the async workers.
pub fn wave_series() -> WaveSeries {
    let waves = snapshot_waves();
    let migr = snapshot_migrates();

    let wave_series: Vec<Wave> = waves
        .iter()
        .map(|w| {
            let s = &w.sample;
            let (wave_ms, quanta_ms) = per_wave(s.waves, s.wave_wall_us, s.wave_overhead_us);
            Wave {
                age: round2(w.age_s),
                ws: s.ws_ms as f64,
                fwd: s.fwd_ms as f64,
                tok: s.tok as f64,
                budget: s.budget_mib as f64,
                used: s.used_mib as f64,
                backlog: s.backlog as f64,
                vram: s.vram.into(),
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
                waves: s.waves as f64,
                wave_ms,
                quanta_ms,
            }
        })
        .collect();
    let censuses: Vec<Census> = snapshot_mixes()
        .into_iter()
        .map(|m| Census {
            age: m.age_s,
            rows: m.rows,
        })
        .collect();
    let mix = mix_series(&censuses, MIX_TOP);
    let migrate_series: Vec<Migrate> = migr
        .iter()
        .map(|m| {
            let s = &m.sample;
            Migrate {
                age: round2(m.age_s),
                res: s.residences as f64,
                mib: s.mib as f64,
                migrate: s.migrate_ms as f64,
                quant: s.quantize_ms as f64,
                copy: s.copy_ms as f64,
                total: s.total_ms as f64,
            }
        })
        .collect();

    let has_pressure = !mix.rows.is_empty() || !migrate_series.is_empty();
    WaveSeries {
        waves: wave_series,
        mix,
        migrate: migrate_series,
        has_pressure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use candle_conversation::layer_trace::LayerVisit;

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
    /// so rather than inventing a row.
    #[test]
    fn no_engine_is_empty_series() {
        let s = serde_json::to_value(wave_series()).unwrap();
        assert_eq!(
            s,
            serde_json::json!({
                "waves": [],
                "mix": {"formats": [], "rows": []},
                "migrate": [],
                "has_pressure": false,
            })
        );
    }
}
