//! Names the errors that mean the device has no ground right now, so callers wait
//! rather than fail.
//!
//! A full span, a standing transient tier, a full KV budget and an exhausted phase
//! arena are ground, not faults, and they clear as other work finishes. Each is a
//! marker substring with a matching recogniser, so a caller can tell a retry from
//! a failure even after the error has been wrapped on its way up.

/// Substring embedded in the error returned when a GPU KV arena cannot be
/// allocated within the VRAM budget (even after a forced compaction). The
/// scheduler matches on this via [`is_device_oom`] to drive eviction / batch
/// clipping instead of letting the driver spill KV to host memory (which on
/// WDDM silently collapses throughput rather than failing).
pub const KV_DEVICE_OOM_MARKER: &str = "kv-cache GPU VRAM budget exceeded";

/// Carried by a region claim refused because the free regions stand above a
/// placed transient tier. Not a full reservation: the ground comes back the
/// moment the scheduler's next gap between forwards takes the tier down, so a
/// caller off the scheduler thread (the persistence pass) waits for that gap
/// rather than treating it as a failure.
pub const TIER_STANDS_OVER_FREE: &str = "out of reach until the wave ends";

/// Recognize a claim refused by a standing tier — see [`TIER_STANDS_OVER_FREE`].
pub fn is_tier_standing_over_free(err: &candle::Error) -> bool {
    err.to_string().contains(TIER_STANDS_OVER_FREE)
}

/// Recognize a device-out-of-memory error — either our own budget rejection
/// ([`KV_DEVICE_OOM_MARKER`]) or a driver-reported CUDA OOM (when sysmem
/// fallback is disabled, `cuMemAlloc` returns `CUDA_ERROR_OUT_OF_MEMORY`).
pub fn is_device_oom(err: &candle::Error) -> bool {
    let s = err.to_string();
    s.contains(KV_DEVICE_OOM_MARKER)
        || s.contains("out of memory")
        || s.contains("OUT_OF_MEMORY")
        || s.contains("CUDA_ERROR_OUT_OF_MEMORY")
}

/// Substring of the error the region pool's tier placement returns when a
/// wave's transient tier cannot be placed at all: the wave is too wide for the
/// ground below the weight floor, and a retry at the same width fails the same
/// way. The scheduler matches on it ([`is_tier_refusal`]) to narrow the next
/// wave rather than repeat this one.
pub const TIER_REFUSED_MARKER: &str = "wave too wide for the partition";

/// Recognize a refused tier placement — see [`TIER_REFUSED_MARKER`].
pub fn is_tier_refusal(err: &candle::Error) -> bool {
    err.to_string().contains(TIER_REFUSED_MARKER)
}

/// Carried by a region claim refused because the span is full — no tier
/// stands over the free ground, and the weight side is at its floor
/// (`region_pool::span_region_refusal`). Like a refused tier it is ground, not
/// a fault: it comes back as turns finish and the KV side relieves, so a turn
/// that met it waits and runs again rather than failing. On `.5` (2026-10-18)
/// two deep projections' promotes filled the span beside the ingest cohorts,
/// and the recurrent-state claims of turns already in flight — a decode step,
/// a creep's first window, a reprojection — failed their turns within seconds
/// of submission.
pub const SPAN_FULL_MARKER: &str = "the span is genuinely full";

/// Carried by a wave buffer asked past what its phase arena holds
/// (`bump_arena`): the forward's span was laid out for a plan other than the
/// one it ran — a step whose placement found an earlier forward's outputs
/// still live keeps the arenas that forward sized, and a wider segment then
/// asks past them. Like a refused tier it is the engine's state, not the turn's:
/// the same step placed again once those outputs are gone is sized to its own
/// plan. Measured on BlackWorld (c673b63c9, 2026-10-19): a creep segment
/// priced at 335 MiB of forward phase carved from a 16 MiB span, and failed the
/// fourteen turns its wave carried.
pub const ARENA_EXHAUSTED_MARKER: &str = "wave arena exhausted";

/// Begins a refusal `begin_wave` gives instead of handing out spans laid out
/// for an earlier plan than the one this forward recorded, while a generation
/// holds that layout. Retryable: the generation drops and the forward placed
/// again lays out its own (`is_arena_exhausted`, which recognises both). Here
/// rather than beside `begin_wave`, so a build without CUDA still has it.
pub const SPANS_OUTGROWN_MARKER: &str = "wave spans outgrown by their plan";

/// Recognize a wave buffer asked past its arena — see [`ARENA_EXHAUSTED_MARKER`]
/// — or the refusal that now comes before it: spans a live generation holds,
/// laid out for an earlier plan than this forward's
/// ([`SPANS_OUTGROWN_MARKER`]). The same state either way,
/// and retryable either way.
pub fn is_arena_exhausted(err: &candle::Error) -> bool {
    let msg = err.to_string();
    msg.contains(ARENA_EXHAUSTED_MARKER) || msg.contains(SPANS_OUTGROWN_MARKER)
}

/// Recognize a claim refused by a full span — see [`SPAN_FULL_MARKER`].
pub fn is_span_full(err: &candle::Error) -> bool {
    err.to_string().contains(SPAN_FULL_MARKER)
}

/// Whether `msg` — an error as text, however many layers wrapped it on its way
/// up — is the device having no ground for the work right now: a refused tier
/// ([`TIER_REFUSED_MARKER`]), a full span ([`SPAN_FULL_MARKER`]) or the KV budget
/// ([`KV_DEVICE_OOM_MARKER`]). Ground that comes back as other work finishes,
/// not a fault in the work that met it.
pub fn is_ground_refusal_message(msg: &str) -> bool {
    [SPAN_FULL_MARKER, TIER_REFUSED_MARKER, KV_DEVICE_OOM_MARKER]
        .iter()
        .any(|m| msg.contains(m))
}

/// Follows [`TIER_REFUSED_MARKER`] in a refusal: the bytes the tier was short
/// of the ground it could be placed in, as `short by {bytes} B`.
pub const TIER_SHORT_BY: &str = "short by ";

/// How far short a refused tier placement fell, in bytes — `None` for any
/// error that is not one. What a caller narrowing its next wave waits to see
/// the ground grow by before it tries the refused width again.
pub fn tier_refusal_shortfall(err: &candle::Error) -> Option<usize> {
    let msg = err.to_string();
    let after_marker = &msg[msg.find(TIER_REFUSED_MARKER)? + TIER_REFUSED_MARKER.len()..];
    let digits = &after_marker[after_marker.find(TIER_SHORT_BY)? + TIER_SHORT_BY.len()..];
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    digits[..end].parse().ok()
}
