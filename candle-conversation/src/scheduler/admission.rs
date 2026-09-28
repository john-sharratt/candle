//! The VRAM-byte admission budget — the scheduler's single throttle.
//!
//! Admission is regulated in BYTES: how much VRAM the inference working set may
//! occupy. Every throttle signal moves that one setpoint, and admission is the
//! only thing that reads it.
//!
//! # Why bytes and not sequences
//!
//! A congestion window over *sequence count* cannot express the work it admits.
//! The tokens behind one queued prefill vary by two orders of magnitude — a
//! single repo-map ingest queues scopes from 48 to 5980 tokens — so a width of 9
//! is idle headroom for nine short scopes and an out-of-memory forward for nine
//! long ones. A count-based controller can therefore only ever be tuned for the
//! worst case it has recently survived: it collapses to the floor on the first
//! wide turn and re-climbs at a rate that has nothing to do with what is
//! actually queued. That is what pinned prefill to single-sequence forwards
//! while gigabytes of the card sat free.
//!
//! Charging each candidate its real KV cost makes the same budget admit nine
//! short scopes or one long one, with no tuning in between.
//!
//! # The cost model
//!
//! Two kinds of work draw on the budget, and they draw very differently:
//!
//! - **Prefill** allocates its whole KV up front, as fresh unsealed blocks in
//!   the session's configured K/V storage format. Its cost is a *stock*:
//!   [`prefill_cost_bytes`], the block-rounded token count times
//!   [`per_block_kv_bytes`].
//! - **Decode** advances one token per sequence per forward, so it allocates a
//!   new block only every [`CHUNK_SIZE`] steps — a *rate*, not a stock. It is
//!   not charged here at all: a decode is a continuation whose ground was bought
//!   when its slot was admitted, and `super::admit::fill` prices the rows it
//!   puts in the wave rather than the bytes it will eventually open.
//!
//! KV that is already resident is *not* modelled here — it is already absent
//! from the live headroom measurement. Only growth is charged.
//!
//! # What the budget is worth right now
//!
//! `Scheduler::admit_budget_ceiling` — free KV regions, less the setpoint the
//! relief pass keeps in hand. It used to live here as `available_bytes`, a live
//! device measurement plus what registered relievers claimed they could evict,
//! minus the evictable-but-pinned working set the hot->warm drain was skipping.
//! Both corrections existed because the base term described *the card*. A region
//! count describes what this process has claimed and not yet spent, and needs
//! neither.
//!
//! # What is left here, and what moved
//!
//! **The decision moved.** `super::admit::fill` is the admission path: an offer
//! joins the wave while the wave goes *faster* carrying it, judged by
//! `super::admit::rate`, and the bytes below are one term in that comparison
//! rather than the whole question. A gate that admitted by fit stopped widening
//! the moment the bytes ran out, which measured ~470 tok/s against a modelled
//! 1,210 at the same residency — a wave of 250 rows and one of 2,000 pay the
//! same expert copy, so the narrow one is not cheaper, only slower per row.
//!
//! What remains here is the arithmetic that survived it: the per-block and
//! per-prefill costing the new path still prices offers with, the AIMD setpoint
//! the *ingest* regulator moves, and the pass budget bounding one forward's
//! width.

use candle_nn::kv_cache::KvFormat;
use candle_nn::kv_cache::CHUNK_SIZE;

/// Admission-budget quantum: the byte step the setpoint grows by, and the floor
/// it can never be cut below. 256 MiB is roughly 1300 tokens of unsealed KV on a
/// 30B-class model — coarse enough that the controller is not chasing individual
/// turns, fine enough that a card has many notches between the floor and its
/// ceiling.
const ADMIT_QUANTUM_MB: u64 = 256;

/// The admission-budget quantum in bytes.
pub(super) fn admit_quantum() -> u64 {
    ADMIT_QUANTUM_MB * 1024 * 1024
}

/// Minimum wall-clock between budget cuts driven by a STANDING CONDITION —
/// [`ThrottleReason::WarmOverBudget`] — as opposed to a discrete failure. Sized to a drain pass: a cut lowers the seal rate, which
/// takes about this long to show up in the signal that caused it, so cutting
/// faster is deciding against stale evidence. See
/// `Scheduler::cut_admit_budget_leveled`.
pub(super) const LEVEL_CUT_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(5);

/// Why the admission budget moved. Carried on every throttle event so a log
/// sweep can attribute a collapsed budget to the signal that collapsed it —
/// without this the budget's trajectory is unattributable after the fact, which
/// is how a silently-climbing admission window was read as a wedged one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ThrottleReason {
    /// A forward reported device out-of-memory. The hardest evidence there is.
    DeviceOom,
    /// A relief pass ran and VRAM was still under pressure afterwards.
    ReliefSurvived,
    /// The warm KV tier outgrew its host-RAM budget plus the drain pipeline's
    /// slack — admission slows so sealing stops outrunning the warm->cold drain.
    WarmOverBudget,
    /// Forwards keep completing out-of-memory-free at the current budget.
    Throughput,
}

impl ThrottleReason {
    /// Stable lowercase tag for the `reason` log field — greppable and
    /// countable across a run.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::DeviceOom => "device_oom",
            Self::ReliefSurvived => "relief_survived",
            Self::WarmOverBudget => "warm_over_budget",
            Self::Throughput => "throughput",
        }
    }
}

/// Bytes one [`CHUNK_SIZE`]-token KV block occupies across the whole model:
/// every layer, every KV head, K and V, in the session's configured storage
/// formats.
///
/// Exact rather than per-token, because a quantized block's size is not
/// generally divisible by its element count (`Q4_0` is 18 bytes for 32
/// elements). Callers round token counts up to whole blocks, which is also what
/// the allocator does.
pub(super) fn per_block_kv_bytes(
    n_layers: usize,
    n_kv_head: usize,
    head_dim: usize,
    k: KvFormat,
    v: KvFormat,
) -> u64 {
    let slots = (n_layers as u64)
        .saturating_mul(n_kv_head as u64)
        .saturating_mul(head_dim as u64);
    let per_slot = (k.bytes_per_block() as u64).saturating_add(v.bytes_per_block() as u64);
    slots.saturating_mul(per_slot)
}

/// Bytes of **KV** a prefill of `tokens` tokens will allocate, rounded up to
/// whole blocks. Zero tokens cost nothing — an empty prefill is an error path
/// that still needs to be admitted so it can report itself.
///
/// This is only half of what admitting that prefill costs the card; see
/// [`admission_cost_bytes`].
pub(super) fn prefill_cost_bytes(tokens: usize, per_block: u64) -> u64 {
    (tokens.div_ceil(CHUNK_SIZE) as u64).saturating_mul(per_block)
}

/// Multiplicative decrease of the budget: halve, but never below `floor`.
/// Repeated application converges to `floor` and stops — the planner's
/// keep-one-in-flight rule, not the floor, is what guarantees progress.
pub(super) fn cut_budget(budget: u64, floor: u64) -> u64 {
    (budget / 2).max(floor)
}

/// Additive increase of the budget: one `quantum`, capped at `ceil`.
pub(super) fn raise_budget(budget: u64, quantum: u64, ceil: u64) -> u64 {
    budget.saturating_add(quantum).min(ceil)
}

/// How many whole quanta the budget currently holds — the byte-space analogue of
/// the old window width, and what the evidence cost scales against.
pub(super) fn budget_notches(budget: u64, quantum: u64) -> usize {
    (budget / quantum.max(1)) as usize
}

/// Evidence-based reopen under chronic nominal VRAM pressure — the escape hatch
/// from an admission wedge on a card whose steady state reads as "pressured"
/// forever (a reserved-but-unreclaimable pool gap, a tight budget band). The
/// AIMD contract says grow only when pressure clears; on such a card it never
/// does, the budget pins at the floor, and prefill runs single-sequence
/// mini-forwards at a fraction of batched throughput. The counter-evidence is
/// throughput itself: when growth is blocked ONLY by the pressure bit — the
/// budget is below its ceiling and forwards are progressing — yet they keep
/// completing out-of-memory-free tick after tick, the current budget is proven
/// sustainable. After `need`
/// consecutive such ticks, grow one quantum and re-arm. A genuinely
/// unsustainable budget surfaces as device-OOM or eviction survival, whose cut
/// resets the streak (multiplicative decrease still wins instantly).
///
/// Returns `(grow_now, new_streak)`.
pub(super) fn evidence_admit_grow(
    budget: u64,
    ceil: u64,
    progressed: bool,
    streak: usize,
    need: usize,
) -> (bool, usize) {
    if budget >= ceil || !progressed {
        return (false, 0);
    }
    let streak = streak + 1;
    if streak >= need {
        (true, 0)
    } else {
        (false, streak)
    }
}

/// Consecutive evidence ticks (one per ~2 s regulator cadence) required before
/// [`evidence_admit_grow`] reopens the budget by one quantum at the floor: ~6 s
/// of proven out-of-memory-free throughput, so a wedged budget walks back up in
/// minutes while a transient spike still cuts it instantly.
pub(super) const EVIDENCE_GROW_TICKS: usize = 3;

/// Evidence ticks required to grow a budget already holding `notches` quanta —
/// the base cost multiplied by what is already held.
///
/// A FLAT cost per quantum makes the controller charge the cliff at constant
/// speed: growing the first quantum is as cheap as the fifteenth, so under
/// chronic ingest pressure it climbs back to whatever budget last blew up, blows
/// up again, and halves to the floor — a ~60 s sawtooth that leaves prefill
/// single-sequence most of the time.
///
/// Scaling the cost by what is already held makes the approach asymptotic
/// instead: escaping the floor stays cheap, while each further quantum demands
/// proportionally more proof that it is sustainable. The budget settles just
/// under the sustainable point rather than oscillating across it, and the
/// wedge-escape property the evidence path exists for is preserved — at the
/// floor it is still the base cost.
pub(super) fn evidence_ticks_for(notches: usize) -> usize {
    EVIDENCE_GROW_TICKS.saturating_mul(notches.max(1))
}

/// How many of `lens`, taken in order, one forward carries within `budget` tokens.
///
/// In order, so nothing queued behind a long turn skips ahead of it; and at least
/// one, so a turn longer than the whole budget still runs rather than waiting for a
/// forward that will never be wide enough.
pub(super) fn admit_within(lens: impl IntoIterator<Item = usize>, budget: usize) -> usize {
    let mut used = 0usize;
    let mut admitted = 0usize;
    for len in lens {
        if admitted > 0 && used + len > budget {
            break;
        }
        used += len;
        admitted += 1;
    }
    admitted
}

#[cfg(test)]
mod tests {
    // Expected byte counts are written as the product they represent
    // (`batch * heads * tokens * …`), so a `* 1` term names a real dimension.
    #![allow(clippy::identity_op)]

    use super::*;
    use candle::DType;
    use candle_nn::kv_cache::QuantFormat;

    const MIB: u64 = 1 << 20;

    /// **In order, up to the budget, and always at least one.**
    #[test]
    fn a_forward_admits_in_order_up_to_the_budget() {
        assert_eq!(admit_within([1900, 1900, 1900], 4096), 2);
        assert_eq!(
            admit_within([1000, 1000, 1000, 1000], 4000),
            4,
            "exactly full admits every one"
        );
        assert_eq!(
            admit_within([9000, 100], 4096),
            1,
            "an over-budget first turn still runs, alone"
        );
        assert_eq!(
            admit_within([100, 9000, 100], 4096),
            1,
            "order is kept: the short turn behind a long one does not skip ahead"
        );
        assert_eq!(admit_within(std::iter::empty(), 4096), 0);
    }

    /// Raw byte assertion on the block cost: 48 layers x 4 KV heads x 128 head
    /// dim, K and V both R16 (128 bytes per 32-element block).
    #[test]
    fn per_block_bytes_are_exact() {
        let r16 = KvFormat::Quantized(QuantFormat::R16);
        let got = per_block_kv_bytes(48, 4, 128, r16, r16);
        assert_eq!(got, 48 * 4 * 128 * (128 + 128));
        assert_eq!(got, 6_291_456);

        // Float formats bill their dtype width across the whole block.
        let bf16 = KvFormat::Float(DType::BF16);
        assert_eq!(
            per_block_kv_bytes(2, 1, 8, bf16, bf16),
            2 * 1 * 8 * (2 * 32 + 2 * 32)
        );

        // Asymmetric K/V is billed asymmetrically.
        let q4 = KvFormat::Quantized(QuantFormat::Q4_0);
        let mixed = per_block_kv_bytes(1, 1, 1, r16, q4);
        assert_eq!(mixed, 128 + q4.bytes_per_block() as u64);
    }

    /// A degenerate geometry must not panic or wrap — it costs nothing.
    #[test]
    fn per_block_bytes_of_an_empty_backing_is_zero() {
        let r16 = KvFormat::Quantized(QuantFormat::R16);
        assert_eq!(per_block_kv_bytes(0, 0, 0, r16, r16), 0);
    }

    /// Admission must price a candidate in the formats a LIVE sequence occupies,
    /// not the sealed ones it settles into.
    ///
    /// On GPU a quantized-configured backing holds active K in `R16` (128 B per
    /// 32-element block — twice plain F16, it carries reserved Q-capture space)
    /// and active V in F16 (64 B). The configured pair costs far less. Pricing
    /// the sealed pair understated the working set ~3.7x, so admission cleared
    /// batches whose real KV ran to gigabytes and the allocator refused them an
    /// arena at a time.
    #[test]
    fn active_formats_price_far_above_sealed_ones() {
        use candle_nn::kv_cache::{active_kv_formats, KvFormat, QuantFormat};

        let sealed_k = KvFormat::Quantized(QuantFormat::Q4_0);
        let sealed_v = KvFormat::Quantized(QuantFormat::Q8_0);
        let (active_k, active_v) = active_kv_formats(sealed_k, true);

        assert_eq!(active_k, KvFormat::Quantized(QuantFormat::R16));
        assert_eq!(active_v, KvFormat::Float(candle::DType::F16));

        // 48 layers x 4 KV heads x 128 dims — the Qwen3-30B-A3B shape.
        let sealed = per_block_kv_bytes(48, 4, 128, sealed_k, sealed_v);
        let active = per_block_kv_bytes(48, 4, 128, active_k, active_v);
        assert!(
            active >= sealed * 3,
            "active must price at least 3x sealed (got active={active} sealed={sealed})"
        );

        // A float-configured backing never quantizes on append: active == sealed,
        // so this must not inflate anything that was already honest.
        let f = KvFormat::Float(candle::DType::BF16);
        assert_eq!(active_kv_formats(f, true), (f, f));
    }

    #[test]
    fn prefill_cost_rounds_up_to_whole_blocks() {
        let per_block = 1000;
        assert_eq!(prefill_cost_bytes(0, per_block), 0);
        // One token still allocates a whole block.
        assert_eq!(prefill_cost_bytes(1, per_block), 1000);
        assert_eq!(prefill_cost_bytes(CHUNK_SIZE, per_block), 1000);
        assert_eq!(prefill_cost_bytes(CHUNK_SIZE + 1, per_block), 2000);
        // The spread that broke count-based admission: 48 vs 5980 tokens is a
        // 125x cost difference at identical sequence count.
        let small = prefill_cost_bytes(48, per_block);
        let large = prefill_cost_bytes(5980, per_block);
        assert_eq!(small, 2000);
        assert_eq!(large, 187_000);
    }

    /// The device-unreserved clamp is what stops admission spending the pool's
    /// reuse gap — the byte range WDDM spills to host memory once the pool nears
    /// the card.
    ///
    /// Replays the measured abort: `headroom=0`, pool `reserved=15168` of a
    #[test]
    fn budget_aimd_converges_and_recovers() {
        let quantum = 256 * MIB;
        let floor = quantum;
        let ceil = 24 * quantum;

        // Multiplicative decrease halves toward the floor and stops there.
        let mut b = ceil;
        let descent: Vec<u64> = (0..6)
            .map(|_| {
                b = cut_budget(b, floor);
                b / MIB
            })
            .collect();
        assert_eq!(descent, vec![3072, 1536, 768, 384, 256, 256]);
        assert_eq!(cut_budget(floor, floor), floor);

        // Additive increase climbs one quantum and saturates at the ceiling.
        let mut b = floor;
        for _ in 0..64 {
            b = raise_budget(b, quantum, ceil);
        }
        assert_eq!(b, ceil);
        assert_eq!(raise_budget(ceil, quantum, ceil), ceil);
        assert_eq!(raise_budget(quantum, quantum, ceil), 2 * quantum);

        // Saturating add: a budget near u64::MAX cannot wrap past the ceiling.
        assert_eq!(raise_budget(u64::MAX, quantum, ceil), ceil);
    }

    #[test]
    fn notches_measure_the_budget_in_quanta() {
        let q = 256 * MIB;
        assert_eq!(budget_notches(0, q), 0);
        assert_eq!(budget_notches(q, q), 1);
        assert_eq!(budget_notches(q * 3 + 1, q), 3);
        // A zero quantum must not divide by zero.
        assert_eq!(budget_notches(q, 0), q as usize);
    }

    #[test]
    fn evidence_cost_rises_with_the_budget_already_held() {
        // Escaping the floor stays cheap — this is the wedge escape hatch.
        assert_eq!(evidence_ticks_for(1), EVIDENCE_GROW_TICKS);
        // …and every further quantum costs proportionally more proof.
        assert!(evidence_ticks_for(8) > evidence_ticks_for(4));
        assert!(evidence_ticks_for(16) > evidence_ticks_for(8));
        assert_eq!(evidence_ticks_for(15), EVIDENCE_GROW_TICKS * 15);
        // An empty budget never demands zero evidence (it would grow every tick).
        assert_eq!(evidence_ticks_for(0), EVIDENCE_GROW_TICKS);
    }

    /// Cumulative cost of climbing grows superlinearly, so the approach to a
    /// known-bad budget is asymptotic rather than a charge.
    #[test]
    fn climbing_to_a_wide_budget_costs_far_more_than_leaving_the_floor() {
        let cost = |from: usize, to: usize| -> usize { (from..to).map(evidence_ticks_for).sum() };
        let low = cost(1, 5);
        let high = cost(11, 15);
        assert!(
            high > low * 3,
            "approaching the cliff must cost far more than escaping the floor: {low} vs {high}",
        );
    }

    #[test]
    fn evidence_grow_requires_streak_and_progress() {
        let (ceil, need) = (24 * MIB, 3);
        let b = MIB;
        // Streak builds one tick at a time, grows on the third, then re-arms.
        assert_eq!(evidence_admit_grow(b, ceil, true, 0, need), (false, 1));
        assert_eq!(evidence_admit_grow(b, ceil, true, 1, need), (false, 2));
        assert_eq!(evidence_admit_grow(b, ceil, true, 2, need), (true, 0));
        // No forward progress → evidence resets (a stalled pump proves nothing).
        assert_eq!(evidence_admit_grow(b, ceil, false, 2, need), (false, 0));
        // Budget already at the ceiling → nothing to reopen.
        assert_eq!(evidence_admit_grow(ceil, ceil, true, 2, need), (false, 0));
    }

    /// **The reopen is not gated on the drain backlog any more.**
    ///
    /// It was: growth required the hot→warm backlog to sit below half its target,
    /// which made the drain's lag a veto on the engine's own width. That was the
    /// ingest throttle, and it is released — completing forwards are the only proof
    /// this controller needs, and the host-RAM guard is what still bounds the warm
    /// tier. Pinned because re-adding a backlog term here would quietly restore the
    /// throttle without touching `regulate_ingest_admission`.
    #[test]
    fn the_reopen_takes_no_backlog_argument() {
        let (ceil, need) = (24 * MIB, 3);
        // Progress alone carries the streak, whatever the drain is doing — there is
        // no longer any input through which a backlog could refuse it.
        assert_eq!(evidence_admit_grow(MIB, ceil, true, 2, need), (true, 0));
    }

    #[test]
    fn throttle_reasons_have_stable_tags() {
        for (r, tag) in [
            (ThrottleReason::DeviceOom, "device_oom"),
            (ThrottleReason::ReliefSurvived, "relief_survived"),
            (ThrottleReason::WarmOverBudget, "warm_over_budget"),
            (ThrottleReason::Throughput, "throughput"),
        ] {
            assert_eq!(r.as_str(), tag);
        }
    }
}
