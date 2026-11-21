//! Process-wide counters and timers for the scheduler's hot path, read by the
//! dashboard and the log rather than by the scheduler itself.
//!
//! A `PhaseTimer` emits one trace event with its elapsed time, and when its target
//! is filtered out it costs one clock read and one trace call that returns early.
//! The per-wave accumulators are swapped to zero at every flush and so describe a
//! window only. The path and branch-checkpoint counters never reset, because the
//! fallback paths they count are correct in every observable way, and only a
//! monotone count shows which path ran. The recurrent-state and wait timers
//! separate where a seal and a stall spend their time, so an unnecessary export
//! or a blocked wait appears as its own figure.

use super::*;

/// Hot-path timing instrumentation for the scheduler.
///
/// Two ergonomic shapes for the same underlying mechanism — both emit
/// one `tracing::trace!` event with elapsed-ms.  Enable observation
/// with `RUST_LOG=candle_conversation::scheduler::timing=trace`.
///
/// 1. RAII guard for "whole-scope" timing — drops on scope exit:
///    ```ignore
///    let _t = PhaseTimer::new("phase_name");
///    // ... work ...                              // event emitted here
///    ```
///
/// 2. Explicit start + finish for sub-phase timing inside one function
///    that needs intermediate values out of the timed block:
///    ```ignore
///    let t = Instant::now();
///    // ... work ...
///    record_phase(t, "phase_name");
///    ```
///
/// Cost when the target is filtered out is one `Instant::now()` and
/// one short `tracing::trace!` call (which short-circuits before
/// formatting when the subscriber rejects the event).
pub(super) struct PhaseTimer {
    pub(super) phase: &'static str,
    pub(super) start: Instant,
}

impl PhaseTimer {
    #[inline]
    pub(super) fn new(phase: &'static str) -> Self {
        Self {
            phase,
            start: Instant::now(),
        }
    }
}

impl Drop for PhaseTimer {
    #[inline]
    fn drop(&mut self) {
        record_phase(self.start, self.phase);
    }
}

/// Per-wave accumulators (µs) splitting the scope-ingest provenance sig cost into
/// its three stages, logged in the wave phase breakdown: `resolve` (host-side
/// per-layer Q-pointer resolution), `kernel` (HtoD + sign-pack launch + D2H), and
/// `assemble` (host XOR-fold of the packed bits). Reset each wave in `flush`.
pub(super) static PROV_RESOLVE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static PROV_KERNEL_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static PROV_ASSEMBLE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Provenance captures served by the GPU sign-pack path and by the CPU R16
/// fallback, since process start. **Never reset** — the timing accumulators
/// above are swapped to zero each wave, which makes them useless for answering
/// "which path ran".
///
/// The question is worth a counter because the fallback is *correct*: when the
/// GPU path is unavailable the capture still happens, still stores a signature,
/// and still scores. What changes is that a per-seal device→host copy of every
/// layer's R16 Q replaces a few KB of packed bits, and nothing about the output
/// says so. A geometry that silently demotes therefore costs throughput with no
/// symptom, which is why the assertion is on the path and not on a duration.
pub(super) static PROV_PATH_GPU: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static PROV_PATH_CPU: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `(gpu, cpu)` provenance captures since process start — see [`PROV_PATH_GPU`].
pub fn provenance_capture_path_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (PROV_PATH_GPU.load(Relaxed), PROV_PATH_CPU.load(Relaxed))
}

/// Prompt branch checkpoints computed by a prefill pass, and installed onto a
/// new conversation's slot, since process start. **Never reset.**
///
/// Counted for the same reason the provenance path is: a conversation that
/// failed to install its branch checkpoint is *correct* in every observable
/// way except the one that matters. It has the whole system prompt in K/V, it
/// answers fluently, and its recurrent layers simply do not remember the
/// prompt. Nothing in the output distinguishes that from a conversation that
/// installed one, so the only assertion that can is on the path.
pub(super) static BRANCH_CKPT_COMPUTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static BRANCH_CKPT_INSTALLED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// What a freshly created slot's recurrent state starts as.
///
/// Not a mode: a statement about what the slot IS. A conversation's slot
/// continues that conversation, so it starts from the timeline's persisted
/// state. A **scratch** slot — the summariser's compression and reproject
/// passes — merely borrows the timeline as an address so sealed-turn injection
/// can resolve it; its decode must be conditioned on the children it is
/// compressing and nothing else. Seeding it from the timeline makes every
/// summary depend on how far the conversation had got when the probe ran, and
/// costs a full ~63 MiB state install per probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StateSeed {
    /// Restore the timeline's snapshot and carried belief — a real conversation.
    FromTimeline,
    /// Start at the sequence-start state — a scratch slot bound to a timeline
    /// only for address resolution.
    Neutral,
}

/// `(computed, installed)` prompt branch checkpoints since process start.
pub fn branch_checkpoint_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        BRANCH_CKPT_COMPUTED.load(Relaxed),
        BRANCH_CKPT_INSTALLED.load(Relaxed),
    )
}

/// Record a branch checkpoint computed by the prefill pass.
pub(crate) fn note_branch_checkpoint_computed() {
    BRANCH_CKPT_COMPUTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Record a branch checkpoint installed onto a conversation's slot.
pub(crate) fn note_branch_checkpoint_installed() {
    BRANCH_CKPT_INSTALLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Cost of carrying recurrent state, accumulated since process start.
///
/// These exist to answer three questions the design left open rather than
/// closed (`docs/deltanet_state_persistence.md` P8, P10.5, P10.6), all of which
/// are of the form "is this cheap enough to leave alone?" and none of which
/// should be answered by reasoning about it:
///
/// - **P8 / T8.1** — does the turn seal's device→host state export show up in
///   seal latency? If it does, the export buffer wants pinning and the copy
///   wants overlapping; if it does not, that is async stream plumbing on the
///   seal path bought for nothing.
/// - **P10.5** — the F32 snapshot write rate, which decides whether turn
///   snapshots want bf16 storage.
/// - **P10.6** — per-turn fork traffic. The ping-pong store removed the
///   per-wave copy this used to be amortised against, and views carry state
///   now, so every dialogue turn pays one device-to-device copy of the whole
///   state.
pub(super) static SNAPSHOT_EXPORT_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static SNAPSHOT_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static SNAPSHOT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static FORK_RECURRENT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// What carrying recurrent state has cost so far.
///
/// `(seal exports, total export µs, total snapshot bytes, state forks)`. Per-fork
/// bytes are `snapshot_bytes / exports` — one state is one state, however it is
/// moved — so the fork traffic is `forks × that`.
pub fn recurrent_state_cost() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        SNAPSHOT_COUNT.load(Relaxed),
        SNAPSHOT_EXPORT_US.load(Relaxed),
        SNAPSHOT_BYTES.load(Relaxed),
        FORK_RECURRENT_COUNT.load(Relaxed),
    )
}

/// Scheduler-thread time spent blocked on a deliberate GPU/persistence **wait** —
/// `device.synchronize()` (draining the GPU queue) and `flush_blocking` (waiting
/// on the persistence thread's hot→warm drain). Accumulated across a wave and
/// swapped into the Sync phase in `flush`, so a multi-second stall shows as its own
/// band instead of hiding inside decode/prefill or the opaque "blocked" remainder.
pub(super) static WAIT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Persistence-maintenance stall (µs): a segment compaction on the persistence
/// thread holds the persistence lock while the scheduler thread's seal writes
/// block on it. Kept SEPARATE from [`WAIT_US`] because it is a pure off-thread
/// *blocked* wait: it must carve only from the `blocked` remainder, never spill
/// into decode/prefill/section the way an on-thread GPU sync legitimately can.
pub(super) static MAINT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Reprojection sub-phase timers (µs), swapped into the projection-decomposition
/// panel each wave: `scan` = probe-extract + belief scan (provenance re-selection),
/// `glue` = the shared gap-fill forward (boundary-seam tokens), `layout` = view
/// project + swap. The remainder of `reproj_ms` is "other".
pub(super) static REPROJ_SCAN_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static REPROJ_GLUE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static REPROJ_LAYOUT_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `true` while `drain_submissions` is running, so the projection-assembler
/// sub-timers (which run in BOTH the drain and reproject paths) attribute their
/// cost to the drain buckets only when it's actually a submission drain.
pub(super) static IN_DRAIN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Drain (submission-projection) sub-phase timers (µs) + prefilled token count,
/// accumulated only while [`IN_DRAIN`]: `elevate` = sealed-prefix inject / warm→hot
/// lift, `prefill` = the turn-content forward (its tokens land in `DRAIN_PREFILL_TOKENS`
/// and are re-attributed to the Prefill phase), `glue` = the submit-time gap-fill.
pub(super) static DRAIN_ELEVATE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static DRAIN_PREFILL_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static DRAIN_GLUE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) static DRAIN_PREFILL_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// MONOTONE count of prefill tokens successfully forwarded (never reset — the
/// wave-stats swap must not disturb it). The ingest admission regulator reads
/// the delta between its ticks as the "forwards are completing OOM-free at this
/// width" evidence that lets the window reopen under chronic nominal VRAM
/// pressure (see `evidence_admit_grow`).
pub(super) static PREFILL_OK_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Every token a turn-content forward ([`projection_assembler`]'s drain path)
/// has prefilled, monotone: a dialogue turn's own prompt, which drains the
/// `High` backlog outside the creep, so `publish_learned_rates` reads its delta
/// into `high_drain`.
pub(super) static TURN_CONTENT_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Add `us` to a drain sub-timer, but only during a submission drain (so the
/// shared assembler helpers don't miscount reproject-path work). Used by
/// `projection_assembler`.
pub(in super::super) fn drain_add_us(atom: &std::sync::atomic::AtomicU64, us: u64) {
    if IN_DRAIN.load(std::sync::atomic::Ordering::Relaxed) {
        atom.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
    }
}

// `timed_synchronize` stood here — `device.synchronize()` timed into `WAIT_US`, so
// a deliberate drain on the scheduler thread surfaced as the Sync phase. Its only
// caller was the heavy-backlog stall, and with that released there is no deliberate
// drain left on this thread to time. Re-add it with the next one rather than
// keeping a timer for a wait nothing performs.

/// Record a persistence-side stall into [`MAINT_US`] from another module (the
/// segment-compaction I/O in `projection::resolver` holds the persistence lock
/// while the scheduler thread's seal writes block on it — an off-thread wait the
/// scheduler can't time itself). Surfaces as the Sync phase, but carved
/// **blocked-only** in `push_phase_window`: the wall-clock added here is off-thread
/// blocked time, so if it exceeds this window's real `blocked` remainder the excess
/// is dropped rather than wrongly shrinking decode/prefill/section (which would
/// inflate their tok/s).
pub(crate) fn note_persistence_maint_us(us: u64) {
    MAINT_US.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

/// Run a blocking wait (e.g. `flush_blocking`) and add its wall-clock to
/// [`WAIT_US`], so a multi-second persistence stall surfaces as the Sync phase.
pub(super) fn timed_wait<T>(f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    WAIT_US.fetch_add(
        t.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    r
}

/// The warm lift's ring, allocating it on the first lift: pinned memory is the
/// weights', and a scheduler that never lifts — a CPU test's — takes none.
pub(super) fn warm_lift_ring(ring: &mut Option<WarmLiftRing>) -> candle::Result<&mut WarmLiftRing> {
    if ring.is_none() {
        *ring = Some(WarmLiftRing::new()?);
    }
    Ok(ring.as_mut().expect("allocated above"))
}

#[inline]
pub(super) fn record_phase(start: Instant, phase: &'static str) {
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    tracing::trace!(
        target: "candle_conversation::scheduler::timing",
        phase,
        ms,
        "phase",
    );
}
