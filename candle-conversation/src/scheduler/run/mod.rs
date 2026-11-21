//! The scheduler's wave loop: the unified step that advances decode and the
//! prefill cohort together, and the timing and limits around it.
//!
//! Each wave is one decode quantum clipped to a wall-clock slice, so the loop
//! returns often enough to admit new work, relieve KV pressure and report
//! telemetry. The creep and section chunks ride the decode sweep whenever their
//! debt allows, and finished turns wait a bounded time for their seal so that the
//! turns still decoding are not held up. Housekeeping time is decomposed into its
//! steps so a regression can be pinned to the step that caused it. When the expert
//! pipeline has exited, the process exits with a status the supervisor acts on.
//!
//! - `creep_airtime`: the creep's airtime debt and cleanup timing.
//! - `decode_loop`: the decode quantum and the work around each step.
//! - `prefill_cohort`: admission and the prefill cohort's advance.
//! - `scheduler_loop`: the main loop and the idle wait.
//! - `timing`: wall-clock attribution by phase.
//! - `tests_airtime`: tests for the creep's airtime debt.
//! - `tests_cleanup`: tests for cleanup timing.

use super::admission::{admit_quantum, budget_notches, evidence_ticks_for, ThrottleReason};
use super::creep_slice::next_layer_row_cost;
use super::prefill::{VramPhase, LOAD_TARGET};
use super::settle::{resettle_owed, SettledGround};
use super::wave_retry::{Next, WaveRetry};
use super::*;
use crate::substrate::residence::{drain_reclaimed, own_device_frees};
use candle::wave_provenance::{publish_wave_declines, DeclineSnapshot};
use candle::DeviceLocation;
use candle_nn::kv_cache::{
    is_span_full, is_tier_refusal, set_kv_free_target, set_prompts_waiting, GapKind,
};
use flume::RecvTimeoutError;
use std::process::exit;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

/// Per-sub-step housekeeping timers (microseconds), swapped out and logged at
/// each wave-window flush. The `housekeeping` phase is one band on the
/// dashboard; these decompose it so a regression can be pinned to the step that
/// caused it rather than to the aggregate.
static HOUSE_PROMOTE_US: AtomicU64 = AtomicU64::new(0);

static HOUSE_ADMIT_US: AtomicU64 = AtomicU64::new(0);

static HOUSE_DEMOTE_US: AtomicU64 = AtomicU64::new(0);

static HOUSE_GPUDRAIN_US: AtomicU64 = AtomicU64::new(0);

/// Which completion path inside `promote_finished_prefills_to_decodes` a span
/// belongs to — that step dominates housekeeping, so it is decomposed further.
pub(super) enum PromoteStep {
    Finalise,
    Compression,
}

static PROMOTE_FINALISE_US: AtomicU64 = AtomicU64::new(0);

static PROMOTE_COMPRESSION_US: AtomicU64 = AtomicU64::new(0);

/// Record one completion-path span (microseconds).
pub(super) fn note_promote_split(step: PromoteStep, us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    match step {
        PromoteStep::Finalise => PROMOTE_FINALISE_US.fetch_add(us, Relaxed),
        PromoteStep::Compression => PROMOTE_COMPRESSION_US.fetch_add(us, Relaxed),
    };
}

/// Drain the promote-path sub-timers, in ms, as `(finalise, compression)`.
pub(super) fn take_promote_split() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        PROMOTE_FINALISE_US.swap(0, Relaxed) / 1000,
        PROMOTE_COMPRESSION_US.swap(0, Relaxed) / 1000,
    )
}

/// Drain the housekeeping sub-timers, in milliseconds, as
/// `(promote, admit, demote, gpu_drain)`.
pub(super) fn take_housekeeping_split() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        HOUSE_PROMOTE_US.swap(0, Relaxed) / 1000,
        HOUSE_ADMIT_US.swap(0, Relaxed) / 1000,
        HOUSE_DEMOTE_US.swap(0, Relaxed) / 1000,
        HOUSE_GPUDRAIN_US.swap(0, Relaxed) / 1000,
    )
}

/// Wall-clock ceiling for one decode quantum ("wave"). The quantum is CLIPPED to
/// this whether or not decode has finished — unfinished sequences persist in
/// `active_decodes` and resume in the next quantum, so clipping just passes the
/// remaining work forward. Time-slicing (rather than the old fixed 32-step
/// budget) guarantees the loop returns to the top every ~2 s to re-form the creep
/// cohort with work admitted mid-wave, run admission + relief, and aggregate
/// telemetry — regardless of how fast or slow the individual steps run.
const WAVE_SLICE: Duration = Duration::from_millis(2000);

/// How long a loop that could admit nothing waits on its channel before it
/// looks again on its own. What reopens admission — a submission, the
/// persistence drain's backlog falling — arrives on the channel and ends the
/// wait at once; this bounds only a change that sends nothing.
const UNADMITTED_RECHECK: Duration = Duration::from_millis(100);

/// Hard safety cap on decode steps within one quantum, independent of the wall
/// clock. At the ~74 ms/step WDDM launch floor a 2 s slice is ~27 steps, far
/// under this; the cap only engages if decode ever runs far faster (TCC mode /
/// CUDA graphs), bounding per-quantum KV growth so the loop-top relief can't be
/// outrun before the deadline check fires.
const MAX_DECODE_STEPS: usize = 256;

/// Longest a finished turn waits for its seal while other turns are still decoding.
///
/// A finished turn's cleanup — its seal and the closing tail it forwards — runs on
/// the loop thread, so run between decode steps it holds back every turn still
/// generating. Measured on the Qwen3.6-35B-A3B with eight turns decoding together:
/// two that accepted more draft tokens finished a step early, and their ~40 ms of
/// cleanup sat in front of the other six's last step. Deferred, the early finishers
/// seal with the rest when the cohort drains — sharing its tail forward — and their
/// tokens have all streamed already, so what waits is only the `Done`. The bound is
/// what keeps a short turn from waiting out a long one's whole generation.
///
/// A turn that wrote a tool call is not deferred at all: its `Done` is what its
/// caller acts on — it runs the tool and submits the result — so waiting there
/// stalls an agent loop rather than a status line.
const CLEANUP_DEFER: Duration = Duration::from_millis(100);

/// The process exit status when the engine can serve nothing again — the expert
/// pipeline has exited. `EX_SOFTWARE`: an internal fault, not a usage error.
const ENGINE_DEAD_EXIT: i32 = 70;

mod creep_airtime;
mod decode_loop;
mod prefill_cohort;
mod scheduler_loop;
#[cfg(test)]
mod tests_airtime;
#[cfg(test)]
mod tests_cleanup;
mod timing;

use creep_airtime::*;
