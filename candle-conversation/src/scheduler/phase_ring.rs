//! In-process ring buffer of per-wave **phase** measurements, for the live
//! performance GUI's phase graphs (`/v1/phases` → `perf.html`).
//!
//! The scheduler flips between distinct phases each wave — small-batch **decode**
//! inference, large-batch **prefill** inference (dialogue and section ingest alike), **projection**
//! (submission drain + reprojection), **sealing** (per-scope KV snapshot/sig/flush),
//! **eviction** (cold-tail / ingest demote), and **allocation** (slot promote + KV
//! pool trim). [`WaveStats`](super::WaveStats) already aggregates every one of these
//! over its 2 s window; at each flush it pushes one [`PhaseWindow`] here so the GUI
//! can render the flip as a volume-over-time timeline without a parallel metrics
//! pipeline.
//!
//! The ring is a process-global (like the `PROV_*` timing atoms next door) so the
//! `zend` telemetry endpoint can read it directly — the scheduler lives in
//! `candle-conversation`, the HTTP layer in `zend`, and both link this crate. It is
//! capped by age (60 min — the widest phase graph) and by a hard length backstop.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub use candle_nn::kv_cache::VramBands;

// ── Projection tracker ───────────────────────────────────────────────────────
// What projection lifted into VRAM for a working set, by name: each unit
// `elevate_projection_working_set` had to bring back because it was not hot.
// A unit lifted on every turn is churn — the live "what keeps coming back"
// board on `perf.html` — and an empty board is the healthy state.

/// What a lifted unit is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectedKind {
    /// A prompt section: the system prompt's parts, a tool, an ingested file.
    Section,
    /// A conversation's turns.
    Turns,
}

impl ProjectedKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProjectedKind::Section => "section",
            ProjectedKind::Turns => "turns",
        }
    }
}

/// One unit's lifts.
#[derive(Clone, Copy, Debug)]
struct Lifts {
    /// Working sets that lifted it.
    lifts: u64,
    /// Turns lifted across those — 1 per lift for a section.
    units: u64,
    last: Instant,
}

static PROJECTED: OnceLock<Mutex<HashMap<(ProjectedKind, String), Lifts>>> = OnceLock::new();

/// Record one lift of `name`, which brought back `units` turns (1 for a
/// section).
pub fn record_projected(kind: ProjectedKind, name: &str, units: u64) {
    let m = PROJECTED.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let e = g.entry((kind, name.to_string())).or_insert(Lifts {
        lifts: 0,
        units: 0,
        last: now,
    });
    e.lifts += 1;
    e.units += units;
    e.last = now;
}

/// One unit's lifts, flattened for JSON with `age_s` since its last.
pub struct ProjectedSnapshot {
    pub kind: ProjectedKind,
    pub name: String,
    pub lifts: u64,
    pub units: u64,
    pub age_s: f64,
}

/// Every unit lifted so far, unsorted.
pub fn snapshot_projected() -> Vec<ProjectedSnapshot> {
    let Some(m) = PROJECTED.get() else {
        return Vec::new();
    };
    let now = Instant::now();
    m.lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|((kind, name), l)| ProjectedSnapshot {
            kind: *kind,
            name: name.clone(),
            lifts: l.lifts,
            units: l.units,
            age_s: now.saturating_duration_since(l.last).as_secs_f64(),
        })
        .collect()
}

/// One scheduler phase within a wave. `as_str` is the stable key the GUI colors by.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PhaseKind {
    /// Small-batch decode inference (foreground token generation).
    Decode,
    /// Large-batch prefill inference: dialogue prefills and section ingest,
    /// which is prefill of a section's tokens through the same forward.
    Prefill,
    /// Submission drain + reprojection (projection wall-clock, no tokens).
    Projection,
    /// Per-scope KV seal (snapshot + provenance sig + record/flush).
    Sealing,
    /// Cold-tail / ingest-demote eviction (frees resident KV bytes).
    Eviction,
    /// Slot promote + KV-pool trim/reclaim (allocation churn).
    Allocation,
    /// Wall-clock the scheduler thread spent blocked off-thread DURING active work
    /// (persistence flush wait, device sync, lock contention, flush-block
    /// housekeeping) — the window remainder after idle is carved out.
    Blocked,
    /// Wall-clock the loop spent waiting on `rx.recv()` with no work to run — idle
    /// between requests, distinct from [`PhaseKind::Blocked`].
    Idle,
    /// Scheduler-thread work servicing engine requests: slot creation (and its
    /// snapshot / belief restore), branch-state installs, section restores, turn
    /// submissions. Real work between forwards, not a stall — it is broken out so
    /// an ingest that opens thousands of conversations reads as request handling
    /// rather than as unattributed [`PhaseKind::Blocked`] time.
    Requests,
    /// Per-wave scheduler housekeeping between the timed quanta: promoting
    /// finished prefills, regulating ingest admission, demoting cold ingest,
    /// walking the AIMD admit budget, harvesting GPU spans. Outside every
    /// quantum, so it was previously indistinguishable from
    /// [`PhaseKind::Blocked`].
    Housekeeping,
    /// Deliberate GPU/persistence wait — `device.synchronize()` draining the GPU
    /// queue + `flush_blocking` waiting on the hot→warm drain. The backpressure
    /// stall, distinct from Idle (no work) and Blocked (unattributed remainder).
    Sync,
}

impl PhaseKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PhaseKind::Decode => "decode",
            PhaseKind::Prefill => "prefill",
            PhaseKind::Projection => "projection",
            PhaseKind::Sealing => "sealing",
            PhaseKind::Eviction => "eviction",
            PhaseKind::Allocation => "allocation",
            PhaseKind::Blocked => "blocked",
            PhaseKind::Idle => "idle",
            PhaseKind::Requests => "requests",
            PhaseKind::Housekeeping => "housekeeping",
            PhaseKind::Sync => "sync",
        }
    }
}

/// One phase's contribution within a wave window. `tokens`/`seqs` are meaningful
/// for the inference phases (Decode/Prefill); `bytes` for the memory
/// phases (Eviction/Allocation); `count` is the operation count (forwards, seals,
/// evicted residences); `dur_ms` is the wall-clock the phase held this window.
#[derive(Clone, Copy)]
pub struct PhaseMeasure {
    pub kind: PhaseKind,
    pub dur_ms: u32,
    pub tokens: u64,
    pub bytes: u64,
    pub seqs: u32,
    pub count: u32,
}

/// The set of phase measurements for one wave window, tagged with the window's
/// end instant (for age computation) and its wall-clock span.
#[derive(Clone)]
pub struct PhaseWindow {
    at: Instant,
    pub window_ms: u32,
    pub phases: Vec<PhaseMeasure>,
}

/// One phase measurement flattened for JSON, with `age_s` = seconds before the
/// snapshot was taken (so the GUI can place it on a "now on the right" axis).
pub struct PhaseSnapshot {
    pub kind: &'static str,
    pub age_s: f64,
    pub window_ms: u32,
    pub dur_ms: u32,
    pub tokens: u64,
    pub bytes: u64,
    pub seqs: u32,
    pub count: u32,
}

/// Oldest window retained — the widest phase graph is 60 min.
const RING_MAX_AGE: Duration = Duration::from_secs(60 * 60);
/// Hard length backstop (~2 windows/s × 60 min ≈ 1800; leave generous slack).
const RING_MAX_LEN: usize = 8192;

static RING: OnceLock<Mutex<VecDeque<PhaseWindow>>> = OnceLock::new();

/// Record one wave's phase measurements. Called from [`WaveStats::flush`] on the
/// scheduler thread; `at` is the window-end instant. Empty windows (no phase ran)
/// are dropped so idle time doesn't dilute the ring.
pub(crate) fn push_window(window_ms: u32, phases: Vec<PhaseMeasure>) {
    if phases.is_empty() {
        return;
    }
    let at = Instant::now();
    let ring = RING.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut q = ring.lock().unwrap();
    q.push_back(PhaseWindow {
        at,
        window_ms,
        phases,
    });
    let cutoff = at.checked_sub(RING_MAX_AGE);
    while let Some(front) = q.front() {
        let too_old = cutoff.is_some_and(|c| front.at < c);
        if too_old || q.len() > RING_MAX_LEN {
            q.pop_front();
        } else {
            break;
        }
    }
}

/// Flatten the ring to per-phase snapshots, newest last, each stamped with its age
/// in seconds relative to now. Cheap clone under the lock; the JSON encode happens
/// off-lock in the caller.
pub fn snapshot() -> Vec<PhaseSnapshot> {
    let Some(ring) = RING.get() else {
        return Vec::new();
    };
    let q = ring.lock().unwrap();
    let now = Instant::now();
    let mut out = Vec::with_capacity(q.len() * 3);
    for w in q.iter() {
        let age_s = now.saturating_duration_since(w.at).as_secs_f64();
        for p in &w.phases {
            out.push(PhaseSnapshot {
                kind: p.kind.as_str(),
                age_s,
                window_ms: w.window_ms,
                dur_ms: p.dur_ms,
                tokens: p.tokens,
                bytes: p.bytes,
                seqs: p.seqs,
                count: p.count,
            });
        }
    }
    out
}

// ── Wave-summary + migration rings ───────────────────────────────────────────
// The generic wave / pressure / arena / migration panels used to be parsed out of
// the daemon log; these rings carry the same numbers straight from the scheduler
// (wave flush) and persistence thread (hot→warm pass), so the dashboard needs no
// log and no `-v`.

/// One wave's headline numbers: VRAM budget/used, prefill backlog, tokens fed,
/// mean forward time, the wall-clock window, and the whole-card VRAM
/// decomposition. All MiB.
#[derive(Clone, Copy)]
pub struct WaveSample {
    at: Instant,
    pub ws_ms: u32,
    pub fwd_ms: u32,
    pub tok: u64,
    pub budget_mib: u64,
    pub used_mib: u64,
    pub backlog: u64,
    /// The whole card at this wave, cut by what holds each byte — bands that add
    /// up to the driver's total (see [`VramBands`]).
    pub vram: VramBands,
    /// Projection decomposition (ms). `pdrain` = submission drain MINUS the prefill
    /// re-attributed to the Prefill phase, decomposed into `drain_elevate`
    /// (sealed-prefix inject / warm→hot) + `drain_glue` (submit gap-fill) + drain
    /// "other" remainder. `reproj` total decomposes into `scan` (provenance
    /// re-selection) + `reproj_glue` (gap-fill) + `layout` (view project+swap) +
    /// reproject "other".
    pub pdrain_ms: u64,
    pub drain_elevate_ms: u64,
    pub drain_glue_ms: u64,
    pub reproj_ms: u64,
    pub reproj_scan_ms: u64,
    pub reproj_glue_ms: u64,
    pub reproj_layout_ms: u64,
    /// Active-work counts at the wave: resident conversations (`slots`) and the
    /// decode / prefill / section-ingest sequence widths.
    pub slots: u32,
    pub decodes: u32,
    pub prefills: u32,
    pub sections: u32,
    /// The window's waves — run-loop passes that ran a forward — their summed
    /// wall-clock and the part of it outside the quanta, in microseconds
    /// (`scheduler::wave_timing`).
    pub waves: u64,
    pub wave_wall_us: u64,
    pub wave_overhead_us: u64,
}

/// One hot→warm migration pass's timing + volume (persistence thread).
#[derive(Clone, Copy)]
pub struct MigrateSample {
    at: Instant,
    pub residences: u64,
    pub mib: u64,
    pub migrate_ms: u64,
    pub quantize_ms: u64,
    pub copy_ms: u64,
    pub total_ms: u64,
}

/// The resident KV by storage format at one moment: `(format, tokens)`, most
/// tokens first (`candle_nn::kv_cache::FormatCensus`).
#[derive(Clone)]
pub struct MixSample {
    at: Instant,
    pub rows: Vec<(String, f64)>,
}

/// How often the persistence thread takes the census — the page's poll.
const MIX_EVERY: Duration = Duration::from_secs(4);

/// Wave/migration history horizon — matches the generic panels' 240-min cap.
const WAVE_MAX_AGE: Duration = Duration::from_secs(240 * 60);
const WAVE_MAX_LEN: usize = 16384;

static WAVES: OnceLock<Mutex<VecDeque<WaveSample>>> = OnceLock::new();
static MIGRATES: OnceLock<Mutex<VecDeque<MigrateSample>>> = OnceLock::new();
static MIXES: OnceLock<Mutex<VecDeque<MixSample>>> = OnceLock::new();

fn trim<T>(q: &mut VecDeque<T>, at: Instant, age_of: impl Fn(&T) -> Instant) {
    let cutoff = at.checked_sub(WAVE_MAX_AGE);
    while let Some(front) = q.front() {
        let too_old = cutoff.is_some_and(|c| age_of(front) < c);
        if too_old || q.len() > WAVE_MAX_LEN {
            q.pop_front();
        } else {
            break;
        }
    }
}

/// Record one wave summary (scheduler thread, at flush).
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_wave(mut s: WaveSample) {
    let at = Instant::now();
    s.at = at;
    let ring = WAVES.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut q = ring.lock().unwrap();
    q.push_back(s);
    trim(&mut q, at, |w| w.at);
}

/// Record one hot→warm migration pass (persistence thread).
pub fn push_migrate(mut s: MigrateSample) {
    let at = Instant::now();
    s.at = at;
    let ring = MIGRATES.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut q = ring.lock().unwrap();
    q.push_back(s);
    trim(&mut q, at, |m| m.at);
}

/// Record a pass that had nothing to move — at most one every [`MIX_EVERY`],
/// so the panel tells a drain that is running and idle from one that stopped
/// without a row for every empty pass.
pub fn push_idle_migrate(total_ms: u64) {
    let quiet = MIGRATES.get().is_none_or(|ring| {
        ring.lock()
            .unwrap_or_else(|e| e.into_inner())
            .back()
            .is_none_or(|m| m.at.elapsed() >= MIX_EVERY)
    });
    if quiet {
        push_migrate(migrate_sample(0, 0, 0, 0, 0, total_ms));
    }
}

/// Each substrate's latest census and when it was taken, by its tenant's rank.
/// Every tenant's persistence thread takes its own, and the card holds
/// them all: a sample is their sum, not whichever thread published last —
/// which drew the larger tenant's KV dropping to the smaller's every other
/// sample.
static MIX_PARTS: OnceLock<Mutex<HashMap<usize, (Instant, Vec<(String, f64)>)>>> = OnceLock::new();

/// Whether substrate `key` owes a census: it has taken none, or its last is
/// [`MIX_EVERY`] old. Read before taking one, so the walk runs only on the
/// passes that publish.
pub fn mix_due(key: usize) -> bool {
    let Some(parts) = MIX_PARTS.get() else {
        return true;
    };
    let parts = parts.lock().unwrap_or_else(|e| e.into_inner());
    parts
        .get(&key)
        .is_none_or(|(at, _)| at.elapsed() >= MIX_EVERY)
}

/// Record substrate `key`'s census (its persistence thread), and a sample of
/// every substrate's latest, summed.
pub fn push_mix(key: usize, rows: Vec<(String, f64)>) {
    let at = Instant::now();
    let parts = MIX_PARTS.get_or_init(|| Mutex::new(HashMap::new()));
    let summed = {
        let mut parts = parts.lock().unwrap_or_else(|e| e.into_inner());
        parts.insert(key, (at, rows));
        sum_mixes(parts.values().map(|(_, r)| r.as_slice()))
    };
    let ring = MIXES.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut q = ring.lock().unwrap_or_else(|e| e.into_inner());
    q.push_back(MixSample { at, rows: summed });
    trim(&mut q, at, |m| m.at);
}

/// Censuses added format by format, most tokens first.
fn sum_mixes<'a>(parts: impl Iterator<Item = &'a [(String, f64)]>) -> Vec<(String, f64)> {
    let mut sum: Vec<(String, f64)> = Vec::new();
    for rows in parts {
        for (name, tokens) in rows {
            match sum.iter_mut().find(|(n, _)| n == name) {
                Some((_, t)) => *t += tokens,
                None => sum.push((name.clone(), *tokens)),
            }
        }
    }
    sum.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    sum
}

/// A census flattened for JSON with `age_s` before now.
pub struct MixSnapshot {
    pub age_s: f64,
    pub rows: Vec<(String, f64)>,
}

pub fn snapshot_mixes() -> Vec<MixSnapshot> {
    let Some(ring) = MIXES.get() else {
        return Vec::new();
    };
    let q = ring.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    q.iter()
        .map(|m| MixSnapshot {
            age_s: now.saturating_duration_since(m.at).as_secs_f64(),
            rows: m.rows.clone(),
        })
        .collect()
}

/// A wave sample flattened for JSON with `age_s` before now.
pub struct WaveSnapshot {
    pub age_s: f64,
    pub sample: WaveSample,
}
/// A migration sample flattened for JSON with `age_s` before now.
pub struct MigrateSnapshot {
    pub age_s: f64,
    pub sample: MigrateSample,
}

pub fn snapshot_waves() -> Vec<WaveSnapshot> {
    let Some(ring) = WAVES.get() else {
        return Vec::new();
    };
    let q = ring.lock().unwrap();
    let now = Instant::now();
    q.iter()
        .map(|w| WaveSnapshot {
            age_s: now.saturating_duration_since(w.at).as_secs_f64(),
            sample: *w,
        })
        .collect()
}

/// The decoding turns more that [`LearnedRates::decode_step_ms_plus_wide`]
/// prices the step at: the width of a placement burst the gateway extrapolates
/// from (`docs/cooperative_admission.md` §2).
pub const PLACEMENT_WIDE: usize = 16;

/// What this card does now, as admission's rate model has learned it — what a
/// cluster member advertises for the gateway to place a turn by (the earliest
/// declared finish, `docs/cooperative_admission.md` §2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnedRates {
    /// Prefill tokens a second at the width admission is running now, at the
    /// residency the wave stands at.
    pub prefill_tok_s: f64,
    /// A decode step's time at the current decode rows, and with one row more —
    /// what one more decoding turn costs every decode on this card.
    pub decode_step_ms: f64,
    pub decode_step_ms_plus_one: f64,
    /// The step with [`PLACEMENT_WIDE`] decoding turns more — how the step grows
    /// with the turns a burst places here, which the one-more step cannot say.
    /// On a card whose every routed expert is copied the step climbs steeply
    /// with width: WhiteWorld (f93524ebc, 2026-10-20) advertised 70.5 ms for
    /// one more and stepped 451–580 ms at 23 decodes, and a burst priced at the
    /// one-more step placed all 48 of its turns there.
    pub decode_step_ms_plus_wide: f64,
    /// Prompt tokens waiting at `High` — the dialogue band a placed turn takes —
    /// that it would queue behind: admitted `High` prefills' remaining rows,
    /// queued `High` turns and `High` submissions not yet carved. Background
    /// work is not in it: a placed turn goes ahead of it.
    pub high_backlog_tokens: u64,
}

/// The newest [`LearnedRates`] the scheduler published, and when.
static LEARNED: OnceLock<Mutex<Option<(Instant, LearnedRates)>>> = OnceLock::new();

/// Publish what the rate model holds now. The scheduler calls this as it
/// runs; a reader takes the newest.
pub fn publish_learned_rates(rates: LearnedRates) {
    let cell = LEARNED.get_or_init(|| Mutex::new(None));
    if let Ok(mut c) = cell.lock() {
        *c = Some((Instant::now(), rates));
    }
    save_learned_rates(&rates);
}

/// Where the published rates are kept across restarts, and when they were last
/// written there.
static LEARNED_STORE: OnceLock<Mutex<(PathBuf, Option<Instant>)>> = OnceLock::new();

/// How often the published rates are written to their store: a figure a
/// minute old is as good a start as one a second old, and the rates publish on
/// every loop pass.
const LEARNED_SAVE_EVERY: Duration = Duration::from_secs(60);

/// **Keep the published rates at `path` across restarts, and start from them.**
///
/// A member that has just started has learned nothing, so it advertises
/// nothing, and the gateway places by power until it has — but it learns only
/// from the turns it is given. On the cluster (run 23, 2026-10-19) WhiteWorld,
/// restarted three minutes before heavy-read's burst, advertised zero for both
/// rates, every pick fell back to power for both members, and the burst was
/// placed by nothing it measured. The same card and model an hour ago is a far
/// better figure than none, and the drain and the rate model correct it within
/// a minute of the first forward.
///
/// Reads what `path` holds — rates this card published before — and, when it
/// is sound, publishes it now as the starting figure with no backlog; then
/// keeps writing what the scheduler publishes there, once a
/// [`LEARNED_SAVE_EVERY`]. Call it once, before the scheduler runs, with a
/// path that belongs to this card and model — beside the substrate the rates
/// were learned against.
pub fn keep_learned_rates_at(path: PathBuf) {
    if let Some(rates) = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| parse_learned_rates(&s))
    {
        let cell = LEARNED.get_or_init(|| Mutex::new(None));
        if let Ok(mut c) = cell.lock() {
            if c.is_none() {
                *c = Some((Instant::now(), rates));
            }
        }
    }
    let store = LEARNED_STORE.get_or_init(|| Mutex::new((path.clone(), None)));
    if let Ok(mut s) = store.lock() {
        s.0 = path;
    }
}

/// Write `rates` to the store, once a [`LEARNED_SAVE_EVERY`]; nothing without
/// one. Best effort: a store that cannot be written costs the next start its
/// head start, not this run anything.
fn save_learned_rates(rates: &LearnedRates) {
    let Some(store) = LEARNED_STORE.get() else {
        return;
    };
    let Ok(mut s) = store.lock() else {
        return;
    };
    let now = Instant::now();
    if s.1
        .is_some_and(|at| now.saturating_duration_since(at) < LEARNED_SAVE_EVERY)
    {
        return;
    }
    s.1 = Some(now);
    let text = format_learned_rates(rates);
    let tmp = s.0.with_extension("tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, &s.0);
    }
}

/// The store's form: one `key value` a line, so a figure added later reads as
/// absent from an older file rather than breaking it.
fn format_learned_rates(r: &LearnedRates) -> String {
    format!(
        "prefill_tok_s {}\ndecode_step_ms {}\ndecode_step_ms_plus_one {}\n\
         decode_step_ms_plus_wide {}\n",
        r.prefill_tok_s, r.decode_step_ms, r.decode_step_ms_plus_one, r.decode_step_ms_plus_wide
    )
}

/// The rates a store holds, with no backlog — a backlog belongs to the run
/// that had it — or `None` unless every rate is there, finite and positive.
fn parse_learned_rates(text: &str) -> Option<LearnedRates> {
    let get = |key: &str| {
        text.lines()
            .filter_map(|l| l.split_once(' '))
            .find(|(k, _)| *k == key)
            .and_then(|(_, v)| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    };
    let plus_one = get("decode_step_ms_plus_one")?;
    Some(LearnedRates {
        prefill_tok_s: get("prefill_tok_s")?,
        decode_step_ms: get("decode_step_ms")?,
        decode_step_ms_plus_one: plus_one,
        // A store an older build wrote has no wide step: the one-more step, the
        // flattest the width could be, until the first publish replaces it.
        decode_step_ms_plus_wide: get("decode_step_ms_plus_wide").unwrap_or(plus_one),
        high_backlog_tokens: 0,
    })
}

/// The rates the scheduler last published, and how old they are in seconds —
/// `None` before its rate model has learned from a forward.
pub fn learned_rates() -> Option<(LearnedRates, f64)> {
    let c = LEARNED.get()?.lock().ok()?;
    let (at, rates) = (*c)?;
    Some((
        rates,
        Instant::now().saturating_duration_since(at).as_secs_f64(),
    ))
}

/// The newest wave, without copying the ring: what a cluster member reads its
/// idle capacity from every second.
pub fn latest_wave() -> Option<WaveSnapshot> {
    let q = WAVES.get()?.lock().unwrap();
    let w = q.back()?;
    Some(WaveSnapshot {
        age_s: Instant::now().saturating_duration_since(w.at).as_secs_f64(),
        sample: *w,
    })
}

pub fn snapshot_migrates() -> Vec<MigrateSnapshot> {
    let Some(ring) = MIGRATES.get() else {
        return Vec::new();
    };
    let q = ring.lock().unwrap();
    let now = Instant::now();
    q.iter()
        .map(|m| MigrateSnapshot {
            age_s: now.saturating_duration_since(m.at).as_secs_f64(),
            sample: *m,
        })
        .collect()
}

/// Constructor used by the scheduler — `at` is stamped in [`push_wave`]. `vram`
/// is the card's bands, `proj` is `(pdrain, drain_elevate, drain_glue, reproj,
/// scan, reproj_glue, layout)` ms, `slots` is `(slots, decodes, prefills,
/// sections)`, `waves` is `(waves, wall_us, overhead_us)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wave_sample(
    ws_ms: u32,
    fwd_ms: u32,
    tok: u64,
    budget_mib: u64,
    used_mib: u64,
    backlog: u64,
    vram: VramBands,
    proj: (u64, u64, u64, u64, u64, u64, u64),
    slots: (u32, u32, u32, u32),
    waves: (u64, u64, u64),
) -> WaveSample {
    let (
        pdrain_ms,
        drain_elevate_ms,
        drain_glue_ms,
        reproj_ms,
        reproj_scan_ms,
        reproj_glue_ms,
        reproj_layout_ms,
    ) = proj;
    let (slots, decodes, prefills, sections) = slots;
    let (waves, wave_wall_us, wave_overhead_us) = waves;
    WaveSample {
        at: Instant::now(),
        ws_ms,
        fwd_ms,
        tok,
        budget_mib,
        used_mib,
        backlog,
        vram,
        pdrain_ms,
        drain_elevate_ms,
        drain_glue_ms,
        reproj_ms,
        reproj_scan_ms,
        reproj_glue_ms,
        reproj_layout_ms,
        slots,
        decodes,
        prefills,
        sections,
        waves,
        wave_wall_us,
        wave_overhead_us,
    }
}

/// Constructor used by the persistence thread — `at` is stamped in [`push_migrate`].
pub fn migrate_sample(
    residences: u64,
    mib: u64,
    migrate_ms: u64,
    quantize_ms: u64,
    copy_ms: u64,
    total_ms: u64,
) -> MigrateSample {
    MigrateSample {
        at: Instant::now(),
        residences,
        mib,
        migrate_ms,
        quantize_ms,
        copy_ms,
        total_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A store reads back as the rates it was written from, without the run's
    /// backlog; a store with any rate missing, unparsable, zero or not finite
    /// starts nothing.
    #[test]
    fn a_kept_store_reads_back_its_rates_and_no_backlog() {
        let r = LearnedRates {
            prefill_tok_s: 703.5,
            decode_step_ms: 107.6,
            decode_step_ms_plus_one: 112.25,
            decode_step_ms_plus_wide: 240.0,
            high_backlog_tokens: 3_545,
        };
        let back = parse_learned_rates(&format_learned_rates(&r)).expect("sound");
        assert_eq!(
            back,
            LearnedRates {
                high_backlog_tokens: 0,
                ..r
            }
        );
        for bad in [
            "",
            "prefill_tok_s 700\ndecode_step_ms 100\n",
            "prefill_tok_s 0\ndecode_step_ms 100\ndecode_step_ms_plus_one 100\n",
            "prefill_tok_s NaN\ndecode_step_ms 100\ndecode_step_ms_plus_one 100\n",
            "prefill_tok_s x\ndecode_step_ms 100\ndecode_step_ms_plus_one 100\n",
        ] {
            assert_eq!(parse_learned_rates(bad), None, "{bad:?}");
        }
        // A line added later is ignored by this reader, not fatal to it.
        let newer = format!("{}some_new_rate 5\n", format_learned_rates(&r));
        assert!(parse_learned_rates(&newer).is_some());
        // A store an older build wrote has no wide step: it reads as the
        // one-more step, not as a missing store.
        let older = "prefill_tok_s 700\ndecode_step_ms 100\ndecode_step_ms_plus_one 110\n";
        let back = parse_learned_rates(older).expect("an older store still seeds");
        assert_eq!(back.decode_step_ms_plus_wide, 110.0);
    }

    /// The phase / wave / migrate rings are process-global, and the harness runs
    /// tests in parallel — so a test that pushes and then reads the ring cannot
    /// run alongside another that pushes. `empty_window_is_dropped` asserts on
    /// the ring's TAIL, so a concurrent push lands there instead of its sentinel
    /// and the assertion reads someone else's entry. It only lost the race under
    /// load (a full `--workspace` run), which made a shared-state race look like
    /// flakiness. Poison-tolerant so one failure doesn't cascade.
    static RING_LOCK: Mutex<()> = Mutex::new(());

    fn ring_guard() -> std::sync::MutexGuard<'static, ()> {
        RING_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn m(kind: PhaseKind, tokens: u64) -> PhaseMeasure {
        PhaseMeasure {
            kind,
            dur_ms: 10,
            tokens,
            bytes: 0,
            seqs: 1,
            count: 1,
        }
    }

    #[test]
    fn empty_window_is_dropped() {
        let _ring = ring_guard();
        push_window(2000, vec![]);
        // Nothing pushed for an empty phase list — snapshot stays whatever the
        // prior tests left; assert only that this call added nothing by pushing a
        // sentinel after and checking it is the tail.
        push_window(2000, vec![m(PhaseKind::Decode, 42)]);
        let snap = snapshot();
        assert_eq!(snap.last().unwrap().tokens, 42);
        assert_eq!(snap.last().unwrap().kind, "decode");
    }

    #[test]
    fn snapshot_flattens_phases_with_age() {
        let _ring = ring_guard();
        push_window(
            2000,
            vec![m(PhaseKind::Prefill, 100), m(PhaseKind::Sealing, 0)],
        );
        let snap = snapshot();
        // Both phases from the window appear; ages are non-negative and finite.
        assert!(snap.iter().any(|s| s.kind == "prefill" && s.tokens == 100));
        assert!(snap.iter().any(|s| s.kind == "sealing"));
        assert!(snap.iter().all(|s| s.age_s >= 0.0 && s.age_s.is_finite()));
    }

    #[test]
    fn blocked_phase_has_stable_key() {
        assert_eq!(PhaseKind::Blocked.as_str(), "blocked");
    }

    /// The dashboard keys its colours, legend and stacking order off these
    /// strings, so a rename silently drops the band from the chart rather than
    /// failing anything. `requests` in particular exists to keep real
    /// scheduler-thread work out of the `blocked` remainder — if it stops being
    /// emitted under this key, that time goes back to reading as unattributed.
    #[test]
    fn requests_phase_has_stable_key() {
        assert_eq!(PhaseKind::Requests.as_str(), "requests");
        assert_eq!(PhaseKind::Housekeeping.as_str(), "housekeeping");
    }

    #[test]
    fn wave_and_migrate_rings_round_trip() {
        let _ring = ring_guard();
        push_wave(wave_sample(
            2000,
            74,
            96,
            14000,
            58000,
            123,
            VramBands {
                total: 65_536.0,
                free: 20_000.0,
                kv: 9_000.0,
                ..VramBands::default()
            },
            (12, 3, 4, 34, 5, 6, 7),
            (48, 8, 2, 40),
            (9, 180_000, 27_000),
        ));
        let w = snapshot_waves();
        let s = &w.last().unwrap().sample;
        assert_eq!(
            (s.waves, s.wave_wall_us, s.wave_overhead_us),
            (9, 180_000, 27_000)
        );
        assert_eq!((s.ws_ms, s.fwd_ms, s.tok), (2000, 74, 96));
        assert_eq!((s.budget_mib, s.used_mib, s.backlog), (14000, 58000, 123));
        assert_eq!(
            (s.vram.kv, s.vram.total, s.vram.free),
            (9_000.0, 65_536.0, 20_000.0)
        );
        assert_eq!(
            (s.pdrain_ms, s.drain_elevate_ms, s.drain_glue_ms),
            (12, 3, 4)
        );
        assert_eq!((s.reproj_ms, s.reproj_glue_ms), (34, 6));
        assert_eq!((s.slots, s.decodes, s.sections), (48, 8, 40));
        assert!(w.last().unwrap().age_s >= 0.0);

        push_migrate(migrate_sample(308, 2314, 77540, 72352, 5151, 77782));
        let m = snapshot_migrates();
        let ms = &m.last().unwrap().sample;
        assert_eq!((ms.residences, ms.mib, ms.total_ms), (308, 2314, 77782));
    }

    /// A census just taken is not owed again by its substrate, and a sample is
    /// every substrate's latest, summed — not the last one published.
    #[test]
    fn a_sample_sums_every_substrates_latest_census() {
        let _ring = ring_guard();
        push_mix(901, vec![("Q4_0".into(), 12.5), ("BF16".into(), 3.0)]);
        assert!(!mix_due(901));
        assert!(mix_due(902), "another substrate owes its own");
        push_mix(902, vec![("BF16".into(), 4.0)]);
        assert_eq!(
            snapshot_mixes().last().unwrap().rows,
            vec![("Q4_0".to_string(), 12.5), ("BF16".to_string(), 7.0)]
        );
    }

    /// A pass with nothing to move is recorded only once the panel has been
    /// quiet a while: straight after a recorded pass, it adds no row.
    #[test]
    fn an_empty_pass_right_after_a_recorded_one_adds_no_row() {
        let _ring = ring_guard();
        push_migrate(migrate_sample(1, 2, 3, 4, 5, 6));
        let before = snapshot_migrates().len();
        push_idle_migrate(7);
        assert_eq!(snapshot_migrates().len(), before);
    }

    /// What the scheduler publishes is what a reader takes, fresh.
    #[test]
    fn published_rates_read_back_with_their_age() {
        let rates = LearnedRates {
            prefill_tok_s: 1234.5,
            decode_step_ms: 80.0,
            decode_step_ms_plus_one: 82.5,
            decode_step_ms_plus_wide: 210.0,
            high_backlog_tokens: 4096,
        };
        publish_learned_rates(rates);
        let (read, age) = learned_rates().expect("published");
        assert_eq!(read, rates);
        assert!(age < 5.0, "just published: {age}");
    }
}
