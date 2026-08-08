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
//!   new block only every [`CHUNK_SIZE`] steps. Its cost is a *rate*, charged
//!   amortised — one thirty-second of a block per sequence per pass
//!   ([`decode_reserve_bytes`]) rather than a full block at the boundary. The
//!   boundary spike is real but small, and a relief pass answers it far faster
//!   than a setpoint could; charging it up front would reserve 32x what decodes
//!   hold and starve prefill outright.
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
//! # The forward reserve
//!
//! A prefill forward's transient peak — dominated by the MoE expert gather, which
//! the whole batch shares — is held back by [`reserve_for_width`]:
//! `width x per_seq`, clamped to a third of the card. This is the one place a
//! reserve is still expressed in bytes against capacity, because it is about
//! *transient activations*, not KV — the KV side's own headroom is the
//! free-region setpoint. It must be evaluated at the width admission is
//! *choosing*, not the width already in flight, which is why
//! [`plan_admission`] re-evaluates the reserve at every candidate count it
//! considers. Evaluating it once — at an in-flight width that is typically zero
//! when admission runs — reserves nothing for the batch about to be formed, and
//! is what let nine sequences through and OOMed a 16 GB card.
//!
//! # Admission order
//!
//! [`plan_admission`] admits the largest candidate that fits first, then walks
//! the whole queue in submission order fitting whatever else it can. Largest
//! first is the anti-starvation rule: a purely greedy in-order walk lets a
//! stream of small scopes keep a large one permanently un-admitted, because
//! there is never a pass where the remaining budget happens to be large enough.
//! The in-order sweep afterwards is what packs the pass full.

use candle_nn::kv_cache::KvFormat;
use candle_nn::kv_cache::CHUNK_SIZE;

/// Default admission-budget quantum: the byte step the setpoint grows by, and
/// the floor it can never be cut below. 256 MiB is roughly 1300 tokens of
/// unsealed KV on a 30B-class model — coarse enough that the controller is not
/// chasing individual turns, fine enough that a card has many notches between
/// the floor and its ceiling.
const DEFAULT_ADMIT_QUANTUM_MB: usize = 256;

/// The admission-budget quantum in bytes, overridable at process start via
/// `CANDLE_ADMIT_QUANTUM_MB` so it can be matched to a card without a rebuild.
/// Cached on first read; `0`/unparseable falls back to
/// [`DEFAULT_ADMIT_QUANTUM_MB`].
pub(super) fn admit_quantum() -> u64 {
    static Q: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *Q.get_or_init(|| {
        let mb = std::env::var("CANDLE_ADMIT_QUANTUM_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(DEFAULT_ADMIT_QUANTUM_MB);
        (mb * 1024 * 1024) as u64
    })
}

/// Minimum wall-clock between budget cuts driven by a STANDING CONDITION —
/// [`ThrottleReason::WarmOverBudget`], [`ThrottleReason::WarmBacklog`] — as opposed to
/// a discrete failure. Sized to a drain pass: a cut lowers the seal rate, which
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
    /// The hot->warm drain is falling behind the seal rate.
    WarmBacklog,
    /// The drain caught up with room to spare.
    DrainCaughtUp,
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
            Self::WarmBacklog => "warm_backlog",
            Self::DrainCaughtUp => "drain_caught_up",
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

/// The forward-transient reserve, as a function of co-batched width.
///
/// Mirrors the band the pressure and relief gates hold: `max(base, width x
/// per_seq)`, clamped to a third of the card so a width spike can never strand
/// the whole device. The terms combine with **max, not sum** — the MoE expert
/// gather dominates a prefill forward's transient peak and is shared by the
/// whole batch, so the first few sequences ride inside `base` for free and width
/// only begins to bind once `width x per_seq` overtakes it.
///
/// Admission evaluates this at the width it is CHOOSING. The pressure and relief
/// gates evaluate the same function at the width already in flight — which is
/// the right question for them and the wrong one for admission, where it is
/// typically zero at the moment of decision.
#[derive(Debug, Clone, Copy)]
pub(super) struct BandParams {
    /// Marginal transient cost attributed to each co-batched sequence.
    pub(super) per_seq: u64,
    /// Measured resident capacity, for the one-third clamp.
    pub(super) capacity: u64,
}

/// The reserve to hold back when `width` sequences are co-batched: the marginal
/// transient cost per sequence, clamped to a third of the card so a width spike
/// can never strand the whole device.
///
/// **Deliberately carries no card-fraction base.** A flat base of
/// `max(capacity/10, 2 GiB)` once sat under this, borrowed from the VRAM
/// pressure gate's band. It starved the scheduler: with ~1150 MiB available and
/// a 2048 MiB base, `available - reserve` saturated to zero at EVERY width, so
/// nothing could be admitted and only the keep-one-alive fallback ran. Decode
/// width never grew past ~3, and a MoE forward amortised over 3 sequences
/// decodes at ~3 tok/s. Forwards run fine at those widths with that much free,
/// which is the evidence the base was never a requirement — it was a policy
/// threshold about when to start shedding, borrowed as if it were a physical
/// one. The per-sequence term is the real transient estimate, and it is all
/// that remains.
pub(super) fn reserve_for_width(width: usize, p: &BandParams) -> u64 {
    (width as u64).saturating_mul(p.per_seq).min(p.capacity / 3)
}

/// Bytes the live decodes will allocate over one forward pass — the amortised
/// charge described in the module header: each of `width` sequences advances one
/// token, which is one [`CHUNK_SIZE`]th of a block.
pub(super) fn decode_reserve_bytes(width: usize, per_block: u64) -> u64 {
    (width as u64).saturating_mul(per_block) / CHUNK_SIZE as u64
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

/// Admission action chosen by the drain-backlog controller.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum BacklogAction {
    Shrink,
    Grow,
    Hold,
}

/// Pure decision core of the ingest admission regulator: from the live hot->warm
/// `backlog`, its `target`, the current `budget`/`ceil`, and whether VRAM is
/// under pressure, decide whether to narrow, widen, or hold ingest admission.
/// Hysteresis — shrink above `target`, grow only below `target / 2`, hold in the
/// deadband between — keeps the budget from flapping as the backlog jitters.
pub(super) fn backlog_admit_action(
    backlog: u64,
    target: u64,
    budget: u64,
    ceil: u64,
    vram_pressure: bool,
) -> BacklogAction {
    if backlog > target {
        BacklogAction::Shrink
    } else if backlog < target / 2 && budget < ceil && !vram_pressure {
        BacklogAction::Grow
    } else {
        BacklogAction::Hold
    }
}

/// Evidence-based reopen under chronic nominal VRAM pressure — the escape hatch
/// from an admission wedge on a card whose steady state reads as "pressured"
/// forever (a reserved-but-unreclaimable pool gap, a tight budget band). The
/// AIMD contract says grow only when pressure clears; on such a card it never
/// does, the budget pins at the floor, and prefill runs single-sequence
/// mini-forwards at a fraction of batched throughput. The counter-evidence is
/// throughput itself: when growth is blocked ONLY by the pressure bit (backlog
/// low, budget below the ceiling) yet forwards keep completing out-of-memory-free
/// tick after tick, the current budget is proven sustainable — after `need`
/// consecutive such ticks, grow one quantum and re-arm. A genuinely
/// unsustainable budget surfaces as device-OOM or eviction survival, whose cut
/// resets the streak (multiplicative decrease still wins instantly).
///
/// Returns `(grow_now, new_streak)`.
pub(super) fn evidence_admit_grow(
    backlog: u64,
    target: u64,
    budget: u64,
    ceil: u64,
    progressed: bool,
    streak: usize,
    need: usize,
) -> (bool, usize) {
    if budget >= ceil || backlog >= target / 2 || !progressed {
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

/// What one admission pass decided: which queue positions to admit, how many it
/// could not fit, and what the admitted set costs.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct AdmissionPlan {
    /// Queue positions to admit, in the order they were chosen (largest-fitting
    /// first, then submission order).
    pub(super) admitted: Vec<usize>,
    /// Candidates walked but not admitted this pass.
    pub(super) skipped: usize,
    /// Total cost of `admitted`.
    pub(super) spent: u64,
}

/// Choose what to admit from `costs` (KV bytes per queued prefill).
///
/// The two limits are **separate quantities and must stay separate**:
///
/// - `available` — what the card physically has (free + evictable − pinned).
///   The forward's [`reserve_for_width`] comes out of *this*.
/// - `budget` — the regulated setpoint, a cap on how much KV admission may add.
///
/// So the room for KV is `min(available − reserve(n), budget) − live_kv`.
/// Subtracting the reserve from the *setpoint* instead is wrong and silently
/// wedges admission: once a throttled setpoint falls below the base reserve —
/// 768 MiB against a 2048 MiB band, observed live — the difference saturates to
/// zero, nothing ever fits, and every pass falls through to the keep-one-alive
/// path admitting exactly one sequence while dozens queue.
///
/// The reserve is re-evaluated at **every candidate count considered**, not
/// once: admitting the n-th sequence has to leave `reserve_for_width(live_width
/// + n)` free, so width prices itself as the batch grows. Evaluating the band
/// once — at the width already in flight, which is usually zero when admission
/// runs — is what let nine sequences through a ceiling that had reserved nothing
/// for the forward they were about to share, and OOMed the card.
///
/// Largest-fitting first, then the whole queue in submission order — see the
/// module header for why the two passes are not the same rule applied twice.
/// The walk is over the entire queue rather than a bounded prefix: a bounded
/// prefix reintroduces exactly the head-of-line blocking the largest-first rule
/// exists to remove, one level deeper.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_admission(
    available: u64,
    budget: u64,
    live_kv: u64,
    live_width: usize,
    costs: &[u64],
    max_count: usize,
    band: &BandParams,
) -> AdmissionPlan {
    let mut admitted: Vec<usize> = Vec::new();
    let mut spent = 0u64;
    if max_count == 0 || costs.is_empty() {
        return AdmissionPlan {
            admitted,
            skipped: costs.len(),
            spent,
        };
    }

    // Bytes still free for KV if we admit `n` more sequences on top of what is
    // already in flight — the reserve grows with the width being chosen, and the
    // setpoint caps the result independently.
    let headroom_for = |n: usize| -> u64 {
        available
            .saturating_sub(reserve_for_width(live_width + n, band))
            .min(budget)
            .saturating_sub(live_kv)
    };
    let fits = |n_admitted: usize, spent: u64, cost: u64| -> bool {
        spent.saturating_add(cost) <= headroom_for(n_admitted + 1)
    };

    // Largest candidate that fits, earliest on a tie so equal-cost work still
    // drains in submission order.
    let head = costs
        .iter()
        .enumerate()
        .filter(|&(_, &c)| fits(0, 0, c))
        .max_by_key(|&(i, &c)| (c, std::cmp::Reverse(i)))
        .map(|(i, _)| i);
    if let Some(i) = head {
        spent += costs[i];
        admitted.push(i);
    }

    // Then pack the pass full in submission order.
    for (i, &c) in costs.iter().enumerate() {
        if admitted.len() >= max_count {
            break;
        }
        if Some(i) == head {
            continue;
        }
        if fits(admitted.len(), spent, c) {
            spent += c;
            admitted.push(i);
        }
    }

    AdmissionPlan {
        skipped: costs.len() - admitted.len(),
        admitted,
        spent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;
    use candle_nn::kv_cache::QuantFormat;

    const MIB: u64 = 1 << 20;

    /// A zero reserve, for the planner tests that exercise fit/order logic in
    /// isolation rather than the width-reserve interaction.
    const NO_BAND: BandParams = BandParams {
        per_seq: 0,
        capacity: 0,
    };

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

    /// The reserve must be evaluated at the width being CHOSEN — the regression
    /// this exists to prevent.
    ///
    /// Replays the measured failure. Available 2901 MiB, base reserve 2048 MiB,
    /// 384 MiB per co-batched sequence, ten cheap 200-token scopes. Evaluating
    /// the reserve once at the in-flight width (zero) leaves 853 MiB and admits
    /// nine — then the forward those nine share peaks against a reserve nobody
    /// held, and the card OOMs.
    #[test]
    fn the_reserve_is_evaluated_at_the_width_being_chosen() {
        const MIB: u64 = 1 << 20;
        let per_block = 6_291_456; // 48L x 4H x 128D, R16 K+V
        let band = BandParams {
            per_seq: 384 * MIB,
            capacity: 16375 * MIB,
        };
        let available = 2901 * MIB;
        let costs: Vec<u64> = (0..10)
            .map(|_| prefill_cost_bytes(200, per_block))
            .collect();

        // What the broken model did: subtract the reserve ONCE at width 0, then
        // fit KV against the remainder.
        let flat = available - reserve_for_width(0, &band);
        let naive = costs.iter().take_while(|&&c| c <= flat).count();
        assert!(naive >= 9, "flat reserve admits {naive}, the OOM path");

        // Width-aware: the reserve grows as the batch does, so it stops short.
        let plan = plan_admission(available, u64::MAX, 0, 0, &costs, 24, &band);
        assert_eq!(plan.admitted.len(), 6);
        // …and what it admitted genuinely fits alongside the reserve it implies.
        assert!(plan.spent + reserve_for_width(plan.admitted.len(), &band) <= available);
    }

    /// The setpoint and the availability limit are SEPARATE — the reserve comes
    /// out of availability, never out of the setpoint.
    ///
    /// This is the wedge that made every admission pass admit exactly one
    /// sequence while dozens queued: a setpoint throttled to 768 MiB, a 2048 MiB
    /// base reserve, and 3340 MiB genuinely available. Subtracting the reserve
    /// from the setpoint saturates to zero and nothing can ever fit, no matter
    /// how much room the card has.
    #[test]
    fn the_reserve_comes_out_of_availability_not_the_setpoint() {
        const MIB: u64 = 1 << 20;
        let band = BandParams {
            per_seq: 384 * MIB,
            capacity: 16375 * MIB,
        };
        let available = 3340 * MIB;
        let setpoint = 768 * MIB;
        // Costs from the observed queue: one large head plus cheap scopes.
        let mut costs = vec![564 * MIB];
        costs.extend(std::iter::repeat_n(12 * MIB, 22));

        let plan = plan_admission(available, setpoint, 0, 0, &costs, 24, &band);

        // available - base reserve = 1292 MiB, capped by the 768 MiB setpoint.
        // The 564 MiB head fits, and cheap scopes pack the rest.
        assert!(
            plan.admitted.len() > 1,
            "must admit more than the keep-one-alive fallback, got {}",
            plan.admitted.len()
        );
        assert!(plan.admitted.contains(&0), "the large head must get in");
        assert!(
            plan.spent <= setpoint,
            "spend {} exceeded setpoint {setpoint}",
            plan.spent
        );
        // Sanity: the two limits are independent — the setpoint caps spend, the
        // reserve comes out of availability. Neither is subtracted from the other.
        assert!(plan.spent <= setpoint);
        assert!(plan.spent + reserve_for_width(plan.admitted.len(), &band) <= available);
    }

    /// The setpoint really does cap spend even when the card has room to spare.
    #[test]
    fn the_setpoint_caps_spend_below_availability() {
        const MIB: u64 = 1 << 20;
        let band = BandParams {
            per_seq: 0,
            capacity: 0,
        };
        let costs = [100 * MIB; 10];
        let plan = plan_admission(u64::MAX, 250 * MIB, 0, 0, &costs, 24, &band);
        assert_eq!(plan.admitted.len(), 2);
        assert_eq!(plan.spent, 200 * MIB);
    }

    /// The admission reserve is purely width-scaled — no policy floor.
    ///
    /// This is the starvation regression. The relief band's card-fraction base
    /// (2048 MiB here) is a "start shedding below this" threshold, NOT a
    /// requirement to run a forward. Using it as an admission floor meant that
    /// with ~1150 MiB available the reserve exceeded availability at EVERY
    /// width, KV headroom saturated to zero, and only the keep-one-alive
    /// fallback ever admitted — which starved decode down to ~3 tok/s.
    #[test]
    fn the_reserve_is_width_scaled_with_no_policy_floor() {
        const MIB: u64 = 1 << 20;
        let band = BandParams {
            per_seq: 384 * MIB,
            capacity: 16375 * MIB,
        };
        assert_eq!(reserve_for_width(0, &band), 0, "no width, no reserve");
        assert_eq!(reserve_for_width(1, &band), 384 * MIB);
        assert_eq!(reserve_for_width(3, &band), 3 * 384 * MIB);
        // Still clamped to a third of the card so a spike cannot strand it.
        assert_eq!(reserve_for_width(1000, &band), 16375 * MIB / 3);

        // The measured starvation: 1151 MiB available.
        let avail = 1151 * MIB;
        assert_eq!(
            avail.saturating_sub(2048 * MIB),
            0,
            "with the old policy base, nothing could ever be admitted"
        );
        assert!(
            avail.saturating_sub(reserve_for_width(1, &band)) > 700 * MIB,
            "width-scaled reserve must leave real room for KV"
        );
    }

    /// In-flight width counts toward the reserve: a pass that already has six
    /// sequences running prices the seventh at the wider band.
    #[test]
    fn in_flight_width_raises_the_reserve_for_new_admits() {
        const MIB: u64 = 1 << 20;
        let band = BandParams {
            per_seq: 384 * MIB,
            capacity: 16375 * MIB,
        };
        let costs = [16 * MIB; 4];
        // Idle: 3000 MiB available, reserve stays at base for the first few.
        let idle = plan_admission(3000 * MIB, u64::MAX, 0, 0, &costs, 24, &band);
        assert_eq!(idle.admitted.len(), 4);
        // Six already in flight: the seventh sequence costs 7 x 384 = 2688 MiB of
        // reserve, leaving 312 MiB — still enough for the cheap KV here…
        let busy = plan_admission(3000 * MIB, u64::MAX, 0, 6, &costs, 24, &band);
        assert!(busy.admitted.len() < 4, "in-flight width must bite");
    }

    /// A decode step is charged one token, not one block — the amortisation.
    #[test]
    fn decode_reserve_is_amortised_over_a_block() {
        let per_block = 6_291_456;
        assert_eq!(decode_reserve_bytes(0, per_block), 0);
        assert_eq!(decode_reserve_bytes(1, per_block), per_block / 32);
        assert_eq!(decode_reserve_bytes(64, per_block), 64 * per_block / 32);
        // 64 concurrent decodes cost far less than one 5980-token prefill, which
        // is the whole point: decodes grow slowly, prefills land whole.
        assert!(decode_reserve_bytes(64, per_block) < prefill_cost_bytes(5980, per_block));
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
        assert_eq!(
            evidence_admit_grow(10, 100, b, ceil, true, 0, need),
            (false, 1)
        );
        assert_eq!(
            evidence_admit_grow(10, 100, b, ceil, true, 1, need),
            (false, 2)
        );
        assert_eq!(
            evidence_admit_grow(10, 100, b, ceil, true, 2, need),
            (true, 0)
        );
        // No forward progress → evidence resets (a stalled pump proves nothing).
        assert_eq!(
            evidence_admit_grow(10, 100, b, ceil, false, 2, need),
            (false, 0)
        );
        // Backlog out of the grow band → not a pressure-only block; reset.
        assert_eq!(
            evidence_admit_grow(60, 100, b, ceil, true, 2, need),
            (false, 0)
        );
        // Budget already at the ceiling → nothing to reopen.
        assert_eq!(
            evidence_admit_grow(10, 100, ceil, ceil, true, 2, need),
            (false, 0)
        );
    }

    #[test]
    fn backlog_admit_action_hysteresis() {
        use BacklogAction::{Grow, Hold, Shrink};
        let ceil = 24 * MIB;
        let target = 8000;
        let b = 4 * MIB;

        // Above target → shrink, regardless of budget position.
        assert_eq!(
            backlog_admit_action(8001, target, ceil, ceil, false),
            Shrink
        );
        assert_eq!(
            backlog_admit_action(20000, target, MIB, ceil, false),
            Shrink
        );

        // Deadband [target/2, target] → hold — no flapping as the backlog jitters.
        assert_eq!(backlog_admit_action(target, target, b, ceil, false), Hold);
        assert_eq!(
            backlog_admit_action(target / 2, target, b, ceil, false),
            Hold
        );
        assert_eq!(backlog_admit_action(5000, target, b, ceil, false), Hold);

        // Below target/2 with headroom and no VRAM pressure → grow.
        assert_eq!(backlog_admit_action(3999, target, b, ceil, false), Grow);
        assert_eq!(backlog_admit_action(0, target, MIB, ceil, false), Grow);

        // Grow is suppressed at the ceiling (nothing to reopen)…
        assert_eq!(backlog_admit_action(0, target, ceil, ceil, false), Hold);
        // …and while VRAM is under pressure (the hard floor wins over reopening).
        assert_eq!(backlog_admit_action(0, target, b, ceil, true), Hold);
        // But a high backlog still shrinks even under VRAM pressure.
        assert_eq!(backlog_admit_action(9000, target, b, ceil, true), Shrink);
    }

    #[test]
    fn plan_admits_largest_first_then_packs_in_order() {
        // Budget 100. Largest fitting is 60 (index 2), then the in-order sweep
        // takes 10 and 30, exactly exhausting the budget and leaving the 20 at
        // index 3 unaffordable — submission order, not best-fit, after the head.
        let costs = [10, 30, 60, 20];
        let plan = plan_admission(100, u64::MAX, 0, 0, &costs, 16, &NO_BAND);
        assert_eq!(plan.admitted, vec![2, 0, 1]);
        assert_eq!(plan.spent, 100);
        assert_eq!(plan.skipped, 1);
    }

    /// The starvation case the largest-first rule exists for: a long queue of
    /// cheap work must not permanently exclude one expensive item.
    #[test]
    fn a_large_candidate_is_not_starved_by_cheap_ones() {
        let mut costs = vec![50]; // the expensive one, at the back
        costs.splice(0..0, std::iter::repeat_n(10u64, 20)); // 20 cheap ones ahead
        let plan = plan_admission(100, u64::MAX, 0, 0, &costs, 16, &NO_BAND);
        assert!(
            plan.admitted.contains(&20),
            "the expensive candidate must be admitted first: {:?}",
            plan.admitted
        );
        assert_eq!(plan.admitted[0], 20);
        // …and the pass is still packed with the cheap work behind it.
        assert_eq!(plan.spent, 100);
        assert_eq!(plan.admitted.len(), 6);
    }

    #[test]
    fn plan_respects_the_count_backstop() {
        let costs = [1, 1, 1, 1, 1];
        let plan = plan_admission(u64::MAX, u64::MAX, 0, 0, &costs, 3, &NO_BAND);
        assert_eq!(plan.admitted.len(), 3);
        assert_eq!(plan.skipped, 2);
        // A zero backstop admits nothing at all.
        let none = plan_admission(u64::MAX, u64::MAX, 0, 0, &costs, 0, &NO_BAND);
        assert!(none.admitted.is_empty());
        assert_eq!(none.skipped, 5);
    }

    #[test]
    fn plan_skips_what_cannot_fit_and_keeps_walking() {
        // Only the 5s fit; the 400 is skipped, not treated as a stop signal.
        let costs = [400, 5, 400, 5];
        let plan = plan_admission(12, u64::MAX, 0, 0, &costs, 16, &NO_BAND);
        assert_eq!(plan.admitted, vec![1, 3]);
        assert_eq!(plan.spent, 10);
        assert_eq!(plan.skipped, 2);
    }

    #[test]
    fn plan_admits_nothing_when_nothing_fits() {
        let costs = [400, 500];
        let plan = plan_admission(12, u64::MAX, 0, 0, &costs, 16, &NO_BAND);
        assert!(plan.admitted.is_empty());
        assert_eq!(plan.spent, 0);
        assert_eq!(plan.skipped, 2);
        // An empty queue is not an error.
        let empty = plan_admission(12, u64::MAX, 0, 0, &[], 16, &NO_BAND);
        assert!(empty.admitted.is_empty());
        assert_eq!(empty.skipped, 0);
    }

    /// Equal-cost candidates drain in submission order — the largest-first rule
    /// must not reorder a uniform queue.
    #[test]
    fn equal_costs_keep_submission_order() {
        let costs = [7, 7, 7, 7];
        let plan = plan_admission(21, u64::MAX, 0, 0, &costs, 16, &NO_BAND);
        assert_eq!(plan.admitted, vec![0, 1, 2]);
        assert_eq!(plan.skipped, 1);
    }

    /// Zero-cost work (an empty prefill that must be admitted to report its own
    /// error) is always admissible, even at zero budget.
    #[test]
    fn zero_cost_work_is_admitted_at_zero_budget() {
        let costs = [0, 0];
        let plan = plan_admission(0, u64::MAX, 0, 0, &costs, 16, &NO_BAND);
        assert_eq!(plan.admitted.len(), 2);
        assert_eq!(plan.spent, 0);
    }

    #[test]
    fn throttle_reasons_have_stable_tags() {
        for (r, tag) in [
            (ThrottleReason::DeviceOom, "device_oom"),
            (ThrottleReason::ReliefSurvived, "relief_survived"),
            (ThrottleReason::WarmOverBudget, "warm_over_budget"),
            (ThrottleReason::WarmBacklog, "warm_backlog"),
            (ThrottleReason::DrainCaughtUp, "drain_caught_up"),
            (ThrottleReason::Throughput, "throughput"),
        ] {
            assert_eq!(r.as_str(), tag);
        }
    }
}
