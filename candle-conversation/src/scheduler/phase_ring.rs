//! In-process ring buffer of per-wave **phase** measurements, for the live
//! performance GUI's phase graphs (`/v1/phases` → `perf.html`).
//!
//! The scheduler flips between distinct phases each wave — small-batch **decode**
//! inference, large-batch **prefill** / **section**-ingest inference, **projection**
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
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// ── Promote tracker ──────────────────────────────────────────────────────────
// Cumulative count, keyed by symbolic name, of how many times each unit (system
// section / tool / turn bucket) was handed to `elevate_projection_working_set`.
// Fixed always-hot units (system prompt, tools) re-elevated every turn float to
// the top — the live "what's being churned" leaderboard on `perf.html`.

static PROMOTES: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();

/// Record one promote of `name` (increment its cumulative count).
pub fn record_promote(name: &str) {
    let m = PROMOTES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap();
    *g.entry(name.to_string()).or_insert(0) += 1;
}

/// Snapshot the promote counts as `(name, count)`, unsorted.
pub fn snapshot_promotes() -> Vec<(String, u64)> {
    let Some(m) = PROMOTES.get() else {
        return Vec::new();
    };
    m.lock().unwrap().iter().map(|(k, v)| (k.clone(), *v)).collect()
}

/// One scheduler phase within a wave. `as_str` is the stable key the GUI colors by.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PhaseKind {
    /// Small-batch decode inference (foreground token generation).
    Decode,
    /// Large-batch dialogue prefill inference.
    Prefill,
    /// Large-batch section-ingest prefill (code-read / repo-map).
    Section,
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
            PhaseKind::Section => "section",
            PhaseKind::Projection => "projection",
            PhaseKind::Sealing => "sealing",
            PhaseKind::Eviction => "eviction",
            PhaseKind::Allocation => "allocation",
            PhaseKind::Blocked => "blocked",
            PhaseKind::Idle => "idle",
            PhaseKind::Sync => "sync",
        }
    }
}

/// One phase's contribution within a wave window. `tokens`/`seqs` are meaningful
/// for the inference phases (Decode/Prefill/Section); `bytes` for the memory
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
/// mean forward time, the wall-clock window, the resident-arena format split, and
/// the whole-card VRAM decomposition (pool reserved + driver total/free). All MiB.
#[derive(Clone, Copy)]
pub struct WaveSample {
    at: Instant,
    pub ws_ms: u32,
    pub fwd_ms: u32,
    pub tok: u64,
    pub budget_mib: u64,
    pub used_mib: u64,
    pub backlog: u64,
    pub float_arenas: u32,
    pub float_mib: u64,
    pub float_live_mib: u64,
    pub quant_arenas: u32,
    pub quant_mib: u64,
    pub quant_live_mib: u64,
    /// KV pool reserved footprint (MiB) — quant + float + slack/fragmentation.
    pub reserved_mib: u64,
    /// Driver total / free VRAM (MiB, `cuMemGetInfo`) for the whole-card decomp.
    pub total_mib: u64,
    pub free_mib: u64,
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

/// Wave/migration history horizon — matches the generic panels' 240-min cap.
const WAVE_MAX_AGE: Duration = Duration::from_secs(240 * 60);
const WAVE_MAX_LEN: usize = 16384;

static WAVES: OnceLock<Mutex<VecDeque<WaveSample>>> = OnceLock::new();
static MIGRATES: OnceLock<Mutex<VecDeque<MigrateSample>>> = OnceLock::new();

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

/// Constructor used by the scheduler — `at` is stamped in [`push_wave`]. `vram` is
/// `(reserved, total, free)` MiB, `proj` is `(pdrain, drain_elevate, drain_glue,
/// reproj, scan, reproj_glue, layout)` ms, `slots` is `(slots, decodes, prefills,
/// sections)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wave_sample(
    ws_ms: u32,
    fwd_ms: u32,
    tok: u64,
    budget_mib: u64,
    used_mib: u64,
    backlog: u64,
    fmt: Option<(u32, u64, u64, u32, u64, u64)>,
    vram: (u64, u64, u64),
    proj: (u64, u64, u64, u64, u64, u64, u64),
    slots: (u32, u32, u32, u32),
) -> WaveSample {
    let (fa, fm, fl, qa, qm, ql) = fmt.unwrap_or((0, 0, 0, 0, 0, 0));
    let (reserved_mib, total_mib, free_mib) = vram;
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
    WaveSample {
        at: Instant::now(),
        ws_ms,
        fwd_ms,
        tok,
        budget_mib,
        used_mib,
        backlog,
        float_arenas: fa,
        float_mib: fm,
        float_live_mib: fl,
        quant_arenas: qa,
        quant_mib: qm,
        quant_live_mib: ql,
        reserved_mib,
        total_mib,
        free_mib,
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

    #[test]
    fn wave_and_migrate_rings_round_trip() {
        push_wave(wave_sample(
            2000,
            74,
            96,
            14000,
            58000,
            123,
            Some((70, 1120, 1001, 467, 7472, 3360)),
            (9000, 65536, 20000),
            (12, 3, 4, 34, 5, 6, 7),
            (48, 8, 2, 40),
        ));
        let w = snapshot_waves();
        let s = &w.last().unwrap().sample;
        assert_eq!((s.ws_ms, s.fwd_ms, s.tok), (2000, 74, 96));
        assert_eq!((s.budget_mib, s.used_mib, s.backlog), (14000, 58000, 123));
        assert_eq!((s.quant_arenas, s.quant_live_mib), (467, 3360));
        assert_eq!(
            (s.reserved_mib, s.total_mib, s.free_mib),
            (9000, 65536, 20000)
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
}
