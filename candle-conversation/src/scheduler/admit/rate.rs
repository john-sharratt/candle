//! The wave throughput model: how many prefill rows and how many decodes one
//! wave should carry, priced in the currency that dominates a MoE forward on a
//! streaming card — expert bytes crossing the bus — so that weight residency is
//! spent where it buys tokens and kept where it does not.
//!
//! # Two regimes, one decision
//!
//! Every offer to the wave is judged the same way: **does the wave's rate go up
//! if this joins?** The two kinds differ only in what the rate is and how the
//! copy is modelled.
//!
//! **Prefill.** A prefill token routes to 8 of 256 experts per layer, so a wave
//! of `W` tokens leaves a layer's expert untouched with probability about
//! `e^(−W/32)`; past a hundred tokens that is nothing. A prefill forward needs
//! **every** expert, and whatever is not resident is copied once per forward,
//! however many tokens ride it:
//!
//! ```text
//! T_prefill(W, R) = fixed + (E_total − R) / bw + W · c
//! ```
//!
//! with `R` the resident expert bytes, `E_total` every expert's bytes, `bw` the
//! effective copy rate and `c` the compute per token. Measured on the 4090
//! Mobile with Qwen3.6-35B-A3B (run 15): 7.5 GiB resident of 20.3 GB, a
//! 474-token forward in 1,004 ms, 0.42 ms a token — the copy is 0.8 s of it.
//! Every token added amortises that copy; every byte dislodged adds to it.
//!
//! **Decode.** A decode row touches 8 experts a layer, times `1 + draft` for a
//! speculative block. The expert cache keeps the hot set resident and the
//! Markov predictor prefetches the rest, so a routed expert is found on the
//! card at a rate that scales with how much of the model is resident:
//!
//! ```text
//! hit(R)        = min(1, hit_rate · R / E_total)   (hit_rate learned, ~1.65)
//! bytes/layer   = min(Σ 8 · (1 + draft_i), experts_per_layer) · slot · (1 − hit(R))
//! T_step(R)     = moe_layers · max(layer_secs, bytes/layer ÷ bw)
//! ```
//!
//! `hit_rate` is a **coefficient on the resident fraction, not a probability**:
//! the LRU zone holds the hot set and the predictor runs ahead of the sweep, so
//! a cache holding two fifths of the model hits far more than two fifths of the
//! time. Measured on the 35B ingest: 0.65 at 39% resident, so the coefficient is
//! ~1.65 and the hit saturates at 58% residency. It is learned from the expert
//! cache's own counters — assuming it costs a factor of two in every decode's
//! priced copy.
//!
//! The copy hides under the layer's own time, `layer_secs`, until it does not;
//! from there the bus sets the step. **A dislodge lowers `R`, which lowers the
//! hit rate, which raises the streamed bytes of every decode already in the
//! wave** — so a decode whose weights cost more step time than its own row
//! repays is refused, exactly as a prefill is.
//!
//! # What the caller supplies
//!
//! What an admission costs in residency — its wave rows' tier, its K/V, its
//! recurrent store, the regions those round to — **is the caller's to
//! compute**: it knows the allocator, this does not. So every offer carries the
//! resident weights before and after it, and the model compares the wave's rate
//! at `(now, R_before)` with the rate at `(now + this, R_after)`. Worse is
//! refused; a gain under the minimum is refused as saturated.
//!
//! # One budget, one `full`
//!
//! [`WaveRate::reset`] opens a wave; [`WaveRate::try_admit`] is offered
//! candidates in priority order. **The first is always admitted** — a wave has
//! to carry something, and the caller's head is the caller's head. After that
//! an offer is admitted while its kind's rate improves, the weights after it
//! stay above the floor and its cap holds. **The first refusal of either kind
//! latches [`WaveRate::is_full`]**, and nothing more is admitted this wave: if
//! high-priority decodes reached their limit first, the prefills behind them
//! wait, which is the point — residency is optimised for whatever the caller
//! offered first.
//!
//! # What is learned
//!
//! Three estimates, all dampened by `alpha`, all seeded so that being wrong is
//! survivable and corrected by evidence:
//!
//! * **The copy rate `bw`** — seeded at [`WaveRate::SEED_FRACTION`] of the link
//!   the device measured at load, learned from prefill forwards, whose expert
//!   set is known (all of them). A forward moving at least
//!   [`WaveRate::PROBE_BYTES`] faster than the measured link *raises the link*:
//!   the probe runs while the expert cache is still staging over the same bus
//!   and reads low.
//! * **The decode layer time** — seeded from
//!   [`DecodeModel::initial_tok_per_s`], learned from decode forwards that ran
//!   compute-bound. A bus-bound step says nothing about the layer.
//! * **The hit coefficient** — seeded at [`DecodeModel::hit_rate`], learned
//!   from the expert cache's hit and miss counters
//!   ([`WaveRate::observe_hit_rate`]).

use candle::Device;

/// The bytes a MoE model's experts occupy in total, from the figures the expert
/// cache reports at load (`moe_layers=41 experts_per_layer=256 slot_bytes=…`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertGeometry {
    pub moe_layers: usize,
    pub experts_per_layer: usize,
    /// Bytes one expert occupies in the resident zone — a slot.
    pub slot_bytes: u64,
}

impl ExpertGeometry {
    /// Qwen3.6-35B-A3B as the expert cache opened it on 2026-09-08.
    pub const QWEN36_35B_A3B: Self = Self {
        moe_layers: 41,
        experts_per_layer: 256,
        slot_bytes: 1_933_312,
    };

    /// Every expert's bytes — what a prefill forward has to have on the card,
    /// resident or copied.
    pub const fn total_bytes(&self) -> u64 {
        (self.moe_layers as u64) * (self.experts_per_layer as u64) * self.slot_bytes
    }
}

/// The measured constants of one prefill forward that are not the copy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateModel {
    /// Per-forward cost that neither width nor residency changes. Zero by
    /// default: the estimator then attributes the whole fixed cost to the copy
    /// and learns an *effective* rate, which is the number to plan with.
    pub fixed_secs: f64,
    /// Compute per token — see [`RateModel::SEED_COMPUTE_SECS_PER_TOKEN`].
    pub compute_secs_per_token: f64,
}

impl RateModel {
    /// The **seed** for compute per prefill row, before any forward has been
    /// measured: 0.55 ms.
    ///
    /// **A seed, not a constant.** [`CostFit`] learns the real value from the
    /// engine's own forwards along with the copy rate, so this is only what the
    /// planner believes for the handful of waves before the observations have
    /// enough spread to separate the two terms. It is a property of the
    /// checkpoint and the activation dtype, not of the card, and it is
    /// different on every model — which is exactly why nothing downstream may
    /// depend on this number staying what it is.
    ///
    /// Where it comes from: 263 forwards of the 35B on the 4090 Mobile,
    /// spanning 150 to 2,250 rows, fit `T(ms) = 697 + 0.553·W` with residuals
    /// of ±70 ms and no curvature. It replaced 0.42 ms, which was a two-point
    /// slope through noisy wave averages and 30% low — enough to make
    /// [`WaveRate::rate`] under-price the marginal row and over-value widening,
    /// and to bias the learned bandwidth low in proportion to the wave's width.
    ///
    /// Seeding high rather than low is the safe direction: an over-stated `c`
    /// under-values widening, which costs throughput, where an under-stated one
    /// spends residency the wave will not earn back.
    ///
    /// **Left at run 15's 0.42 ms rather than the fitted 0.553.** Both are
    /// measurements of one card and one checkpoint, and neither survives the
    /// move to another — which is the whole reason [`CostFit`] exists. The seed
    /// governs the handful of forwards before the fit is conditioned, and the
    /// fixtures throughout this module's tests are derived *through* it (`c`
    /// sets how much of run 15's forward is attributed to the copy, so
    /// `BW_RUN15` and every decode figure that reads it move with it). Changing
    /// a seed that is superseded within a dozen forwards is not worth
    /// re-deriving a test suite for.
    pub const SEED_COMPUTE_SECS_PER_TOKEN: f64 = 0.42e-3;
}

impl Default for RateModel {
    fn default() -> Self {
        Self {
            fixed_secs: 0.0,
            compute_secs_per_token: Self::SEED_COMPUTE_SECS_PER_TOKEN,
        }
    }
}

/// The decode wave's model: what one decode routes to, how much of that the
/// resident zone and its predictor catch, and how long a layer has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeModel {
    /// Experts one token routes to per layer — the top-k. 8 on the 35B.
    pub experts_per_token: usize,
    /// How well the cache converts residency into hits: a routed expert is
    /// found on the card at `hit_rate × resident_fraction`, capped at certainty.
    ///
    /// **A coefficient, not itself a probability.** The LRU zone holds the hot
    /// set and the Markov predictor prefetches ahead of the sweep, so a cache
    /// holding a third of the model hits far more than a third of the time —
    /// that is the entire point of both. Seeded at the 0.7 the predictor was
    /// measured at with everything resident, and then **learned**
    /// ([`WaveRate::observe_hit_rate`]) from the expert cache's own hit and miss
    /// counters, which is the only figure that knows what this workload's
    /// routing actually looks like. Measured on the 35B at 37% resident: a hit
    /// rate of 0.65, so the coefficient is ~1.73 — seeding it at 0.7 alone would
    /// have priced every decode's copy at two and a half times its real cost.
    ///
    /// What the shape preserves is the sensitivity the whole refinement is for:
    /// residency still multiplies, so a dislodge still raises the streamed bytes
    /// of every decode already in the wave.
    pub hit_rate: f64,
    /// The step rate the layer time is seeded from: one step at this rate,
    /// spread over the MoE layers, is the time a layer has to hide its copies
    /// behind until the waves say otherwise.
    pub initial_tok_per_s: f64,
}

impl Default for DecodeModel {
    fn default() -> Self {
        Self {
            experts_per_token: 8,
            hit_rate: 0.7,
            initial_tok_per_s: 30.0,
        }
    }
}

/// The running fit of a forward's two costs: the copy rate and the per-token
/// compute.
///
/// # Why a fit rather than two constants
///
/// A prefill forward is `T = X/bw + W·c` — non-resident bytes over the bus,
/// plus compute for the rows. Both terms are properties of *this* machine and
/// *this* checkpoint: `bw` is the card's link and how much of it the expert
/// path gets, `c` is the model's compute per row at the activation dtype. A
/// number measured on a 4090 with a 35B is wrong on a Blackwell with anything,
/// and wrong in a direction that matters — under-stating `c` makes the marginal
/// row look cheaper than it is, so the planner over-values widening and spends
/// residency for gains that are not there.
///
/// One forward is one equation in two unknowns. A *set* of forwards at
/// different widths and residencies determines both, and that is all this is:
/// the normal equations of a two-feature least squares, five running sums,
/// solved whenever the observations have enough spread to be conditioned.
///
/// # Forgetting
///
/// Each observation decays the accumulators by `1 − alpha` before adding
/// itself, so the fit tracks a machine whose behaviour moves — an expert cache
/// warming, a dtype change, thermal throttling — instead of averaging over the
/// whole run. With `alpha` at its default the fit has a memory of a few dozen
/// forwards.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CostFit {
    /// Σ X², Σ X·W, Σ W² — the Gram matrix of the two features.
    xx: f64,
    xw: f64,
    ww: f64,
    /// Σ X·T, Σ W·T — the right-hand side.
    xt: f64,
    wt: f64,
    /// Observations folded in, for the conditioning guard.
    n: u64,
}

impl CostFit {
    /// Observations required before the fit is believed at all. Two would be
    /// enough to solve the system and far too few to trust: a pair of forwards
    /// at nearly the same width gives a near-singular matrix and an answer that
    /// swings wildly on noise. The determinant guard below is the real check;
    /// this stops it being consulted while the sums are still tiny.
    const MIN_SAMPLES: u64 = 8;

    /// The fraction of `Σ X² · Σ W²` the determinant must reach for the system
    /// to be considered conditioned.
    ///
    /// The determinant is `ΣX²·ΣW² − (ΣX·W)²`, which is zero exactly when every
    /// observation has the same `X`-to-`W` ratio — forwards that all sat at one
    /// width and one residency, from which the two costs cannot be separated no
    /// matter how many there are. Requiring a real fraction of the product means
    /// the widths and residencies have genuinely varied.
    const MIN_CONDITION: f64 = 1e-3;

    /// Fold one forward in, decaying what came before.
    fn observe(&mut self, x: f64, w: f64, t: f64, alpha: f64) {
        let keep = 1.0 - alpha.clamp(0.0, 1.0);
        self.xx = self.xx * keep + x * x;
        self.xw = self.xw * keep + x * w;
        self.ww = self.ww * keep + w * w;
        self.xt = self.xt * keep + x * t;
        self.wt = self.wt * keep + w * t;
        self.n += 1;
    }

    /// Solve for `(bw, c)`, or `None` while the observations cannot separate
    /// them.
    ///
    /// Both answers are checked for physical sense before being returned: a
    /// copy rate must be positive and no faster than the link, and a per-token
    /// compute must be positive. A fit that produces either — noise, a stalled
    /// forward, a residency that moved mid-wave — is discarded rather than
    /// clamped, because a clamped nonsense answer is indistinguishable from a
    /// real one downstream.
    fn solve(&self, link_bytes_per_s: f64) -> Option<(f64, f64)> {
        if self.n < Self::MIN_SAMPLES {
            return None;
        }
        let det = self.xx * self.ww - self.xw * self.xw;
        if !(det.is_finite() && det > self.xx * self.ww * Self::MIN_CONDITION) {
            return None;
        }
        // a = 1/bw, so the copy term is a·X.
        let a = (self.ww * self.xt - self.xw * self.wt) / det;
        let c = (self.xx * self.wt - self.xw * self.xt) / det;
        if !(a.is_finite() && c.is_finite()) || a <= 0.0 || c <= 0.0 {
            return None;
        }
        let bw = 1.0 / a;
        if !bw.is_finite() || bw <= 0.0 {
            return None;
        }
        Some((bw.min(link_bytes_per_s), c))
    }
}

/// One candidate for the wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// A prefill chunk of this many rows.
    Prefill { tokens: usize },
    /// One decode sequence, riding as a block of `1 + draft` rows.
    Decode { draft: usize },
}

/// Why [`WaveRate::try_admit`] said no.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refusal {
    /// The wave reached a limit earlier this pass; nothing more is admitted
    /// until the next reset, whatever the kind.
    Full,
    /// The weights after the admission would stand under the floor. Hard: the
    /// caller's hold is the one line this never crosses.
    Floor { resident_after: u64, floor: u64 },
    /// The prefill would widen the wave past the cap the caller set at reset.
    Cap { max_tokens: usize },
    /// The wave already carries the caller's maximum number of decodes.
    DecodeCap { max_decodes: usize },
    /// The kind's rate after the admission is below the rate before it: the
    /// weights it dislodges cost more time than its rows repay. Tokens a
    /// second for a prefill, decodes a second for a decode.
    Worse { before: f64, after: f64 },
    /// The rate would improve by less than the minimum gain: the wave has
    /// saturated — for decodes, the bus sets the step and another row only
    /// stretches it — and the residency the admission would spend buys nothing
    /// worth having.
    Saturated { gain: f64 },
}

/// The answer to one admission.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Admit {
    /// Admitted. For a prefill, the wave's projected prefill rate with it, in
    /// tokens a second; for a decode, the projected decode rate, in decodes
    /// stepped a second.
    Admitted {
        projected: f64,
    },
    Refused(Refusal),
}

impl Admit {
    pub fn is_admitted(&self) -> bool {
        matches!(self, Admit::Admitted { .. })
    }
}

/// The wave throughput planner: the two models, the learned rates, and the
/// budget of the wave being composed.
#[derive(Clone, Debug)]
pub struct WaveRate {
    geometry: ExpertGeometry,
    model: RateModel,
    decode: DecodeModel,
    /// The hardware's copy rate as measured at load, bytes a second. The
    /// ceiling of what the estimator may believe.
    link_bytes_per_s: f64,
    /// The effective copy rate the forwards are actually seeing, learned.
    bw_bytes_per_s: f64,
    /// The time a MoE layer takes in a decode wave when the bus is not the
    /// limit, learned.
    layer_secs: f64,
    /// Dampening: the fraction of each observation an estimate moves by.
    alpha: f64,
    /// Relative rate gain below which an admission is refused as saturated.
    min_gain: f64,
    /// Prefill forwards folded into the copy rate so far.
    samples: u64,
    /// Decode forwards folded into the layer time so far.
    decode_samples: u64,
    /// Routing intervals folded into the hit coefficient so far.
    hit_samples: u64,
    /// The running two-cost fit — see [`CostFit`]. Learns the copy rate and the
    /// per-token compute together, so neither is a constant measured on one
    /// card with one checkpoint.
    fit: CostFit,
    // ── the wave being composed ────────────────────────────────────────────
    /// Resident weights as the wave stands: the reset figure, then each
    /// admission's `weights_after`.
    resident_now: u64,
    floor_bytes: u64,
    max_tokens: usize,
    max_decodes: usize,
    tokens: usize,
    decodes: usize,
    /// Experts the admitted decodes route to per layer, before the layer cap:
    /// `Σ experts_per_token · (1 + draft)`.
    routed_per_layer: usize,
    admitted_any: bool,
    full: bool,
}

impl WaveRate {
    /// The estimator's seed as a fraction of the measured link: conservative,
    /// so the first waves are composed for a slower bus than the card has and
    /// the model learns upward from evidence rather than down from optimism.
    pub const SEED_FRACTION: f64 = 0.8;
    /// Default dampening. This machine's throughput has a ~2× run-to-run band,
    /// so one forward moves an estimate by a sixth of the way at most.
    pub const DEFAULT_ALPHA: f64 = 0.15;
    /// Default minimum relative gain an admission must buy.
    pub const DEFAULT_MIN_GAIN: f64 = 0.01;
    /// Decode forwards the layer time must have seen before a promotion may be
    /// refused on it — see [`Self::judge_promotion`].
    ///
    /// Eight, matching [`CostFit::MIN_SAMPLES`]: enough for the dampened
    /// estimate to have travelled most of the way from its seed to the measured
    /// value (run 33 read 0.00394 at three samples against a converged 0.027),
    /// and few enough that the window costs a handful of waves rather than a
    /// phase.
    pub const MIN_DECODE_SAMPLES: u64 = 8;
    /// Bytes the load-time link probe copies. Large enough that launch latency
    /// is noise against the transfer; small enough to be a moment at load.
    pub const PROBE_BYTES: usize = 256 << 20;
    /// Timed copies the link probe takes, keeping the fastest — see
    /// [`measure_link_bytes_per_s`] for why one is not enough.
    pub const PROBE_RUNS: usize = 3;
    /// The estimator never believes a copy rate under this fraction of the
    /// link: below it the observation is measuring something other than the
    /// copy (a stall, a sync, a wave that carried a park), and it is refused.
    const MIN_LINK_FRACTION: f64 = 0.02;

    /// Build the planner against a device, measuring the host→device copy
    /// rate with one pinned transfer of [`Self::PROBE_BYTES`] and seeding the
    /// estimate at [`Self::SEED_FRACTION`] of it.
    ///
    /// The probe is the same path the expert cache streams through — pinned
    /// host memory onto the device's stream — so the figure is this machine's
    /// own, not a datasheet's: the 4090 Mobile, the 3090 box behind PCIe 3.0 and
    /// the Blackwell workstation each seed correctly without a per-machine row.
    pub fn measure(
        device: &Device,
        geometry: ExpertGeometry,
        model: RateModel,
        decode: DecodeModel,
    ) -> candle::Result<Self> {
        let link = measure_link_bytes_per_s(device)?;
        Ok(Self::with_link_rate(link, geometry, model, decode))
    }

    /// Build the planner from a known link rate, in bytes a second. The
    /// constructor every test uses, and the one a caller without a device uses.
    pub fn with_link_rate(
        link_bytes_per_s: f64,
        geometry: ExpertGeometry,
        model: RateModel,
        decode: DecodeModel,
    ) -> Self {
        assert!(
            link_bytes_per_s.is_finite() && link_bytes_per_s > 0.0,
            "a link rate must be a positive finite number of bytes a second"
        );
        assert!(
            decode.initial_tok_per_s > 0.0 && geometry.moe_layers > 0,
            "a decode step rate and a layer count are needed to seed the layer time"
        );
        assert!(
            decode.hit_rate.is_finite() && decode.hit_rate >= 0.0,
            "a hit coefficient is a non-negative finite multiplier on the resident fraction"
        );
        Self {
            geometry,
            model,
            decode,
            link_bytes_per_s,
            bw_bytes_per_s: link_bytes_per_s * Self::SEED_FRACTION,
            layer_secs: 1.0 / (decode.initial_tok_per_s * geometry.moe_layers as f64),
            alpha: Self::DEFAULT_ALPHA,
            min_gain: Self::DEFAULT_MIN_GAIN,
            samples: 0,
            decode_samples: 0,
            hit_samples: 0,
            fit: CostFit::default(),
            resident_now: 0,
            floor_bytes: 0,
            max_tokens: 0,
            max_decodes: 0,
            tokens: 0,
            decodes: 0,
            routed_per_layer: 0,
            admitted_any: false,
            full: false,
        }
    }

    /// Change the dampening. `alpha` in `(0, 1]`; 1 believes every observation
    /// outright.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        self.alpha = alpha;
        self
    }

    /// Change the minimum relative gain an admission must buy. Zero admits
    /// anything that does not make the rate worse.
    pub fn with_min_gain(mut self, min_gain: f64) -> Self {
        assert!(min_gain >= 0.0, "a minimum gain cannot be negative");
        self.min_gain = min_gain;
        self
    }

    /// The link rate measured at load, bytes a second.
    pub fn link_bytes_per_s(&self) -> f64 {
        self.link_bytes_per_s
    }

    /// The effective copy rate the model currently believes, bytes a second.
    pub fn effective_bytes_per_s(&self) -> f64 {
        self.bw_bytes_per_s
    }

    /// The effective rate as a fraction of the link — how much of the bus the
    /// expert path is actually getting.
    pub fn link_fraction(&self) -> f64 {
        self.bw_bytes_per_s / self.link_bytes_per_s
    }

    /// The time a MoE layer takes in a decode wave when the bus is not the
    /// limit.
    pub fn layer_secs(&self) -> f64 {
        self.layer_secs
    }

    /// Prefill forwards the copy rate has learned from.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Decode forwards the layer time has learned from.
    pub fn decode_samples(&self) -> u64 {
        self.decode_samples
    }

    // ── the prefill model ──────────────────────────────────────────────────

    /// Expert bytes a prefill forward copies with `resident` bytes on the card.
    pub fn non_resident_bytes(&self, resident: u64) -> u64 {
        self.geometry.total_bytes().saturating_sub(resident)
    }

    /// Projected time of a prefill forward of `tokens` rows with `resident`
    /// expert bytes on the card, in seconds.
    pub fn forward_secs(&self, tokens: usize, resident: u64) -> f64 {
        self.model.fixed_secs
            + self.non_resident_bytes(resident) as f64 / self.bw_bytes_per_s
            + tokens as f64 * self.model.compute_secs_per_token
    }

    /// Projected throughput of that forward, tokens a second. Zero for an
    /// empty wave.
    pub fn rate(&self, tokens: usize, resident: u64) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        tokens as f64 / self.forward_secs(tokens, resident)
    }

    /// The narrowest forward whose copy is worth learning from.
    ///
    /// **The whole prefill model rests on "a forward needs every expert", and
    /// that is a statement about width.** A wave of `W` rows leaves one of a
    /// layer's experts untouched with probability about
    /// `e^(−W · experts_per_token / experts_per_layer)`, so a *narrow* forward
    /// routes to a fraction of the layer and copies a fraction of the bytes —
    /// while [`Self::observe_prefill`] divides the time by *all* of them. The
    /// implied rate is then far too fast, and on this engine that is not
    /// hypothetical: a 26-row section forward read as 119 GB/s on a 25 GB/s
    /// link, and the estimate it poisoned prices every wave after it.
    ///
    /// This is the width at which the untouched share falls under 2% — about
    /// 125 rows on the 35B's 8-of-256 routing — derived from the geometry so it
    /// follows the model rather than being a constant to re-derive per card.
    pub fn min_learn_rows(&self) -> usize {
        let per = self.decode.experts_per_token.max(1) as f64;
        let ratio = self.geometry.experts_per_layer as f64 / per;
        (ratio * 50f64.ln()).ceil() as usize
    }

    /// The prefill rate ceiling as width goes to infinity with residency held:
    /// every row still costs its compute, and nothing else. What an admission
    /// dislodges is the caller's figure and lowers the real ceiling by it.
    pub fn asymptotic_rate(&self) -> f64 {
        1.0 / self.model.compute_secs_per_token
    }

    // ── the decode model ───────────────────────────────────────────────────

    /// The fraction of the model's experts that are resident, in `[0, 1]`.
    pub fn resident_fraction(&self, resident: u64) -> f64 {
        (resident as f64 / self.geometry.total_bytes() as f64).min(1.0)
    }

    /// The rate at which a routed expert is found on the card at `resident`
    /// bytes: the coefficient times the resident fraction, and never more than
    /// certainty — a learned coefficient above 1 means the cache out-performs
    /// its share, which stops being true before it reaches "always".
    pub fn hit_fraction(&self, resident: u64) -> f64 {
        (self.decode.hit_rate * self.resident_fraction(resident)).clamp(0.0, 1.0)
    }

    /// The fraction of a decode's routed experts that have to be streamed with
    /// `resident` bytes on the card: what the zone and its predictor miss.
    /// Falls as residency rises; never negative, never above 1.
    pub fn streamed_fraction(&self, resident: u64) -> f64 {
        1.0 - self.hit_fraction(resident)
    }

    /// Experts one decode of `draft` proposals routes to per layer.
    pub fn experts_per_decode(&self, draft: usize) -> usize {
        self.decode.experts_per_token.saturating_mul(1 + draft)
    }

    /// Bytes a layer streams for decodes routing to `routed` experts between
    /// them with `resident` bytes on the card: no more experts than the layer
    /// has, of which the streamed fraction crosses the bus.
    pub fn decode_layer_bytes(&self, routed: usize, resident: u64) -> f64 {
        let experts = routed.min(self.geometry.experts_per_layer) as f64;
        experts * self.geometry.slot_bytes as f64 * self.streamed_fraction(resident)
    }

    /// Seconds a layer spends copying for those decodes at that residency.
    pub fn decode_copy_secs(&self, routed: usize, resident: u64) -> f64 {
        self.decode_layer_bytes(routed, resident) / self.bw_bytes_per_s
    }

    /// Projected time of one decode step: every MoE layer takes its own time
    /// or its copy, whichever is longer.
    pub fn decode_step_secs(&self, routed: usize, resident: u64) -> f64 {
        self.geometry.moe_layers as f64
            * self.layer_secs.max(self.decode_copy_secs(routed, resident))
    }

    /// Projected decode throughput: `decodes` sequences stepped a second, for a
    /// wave whose decodes route to `routed` experts a layer at `resident`
    /// bytes. Zero for a wave with no decodes.
    pub fn decode_rate(&self, decodes: usize, routed: usize, resident: u64) -> f64 {
        if decodes == 0 {
            return 0.0;
        }
        decodes as f64 / self.decode_step_secs(routed, resident)
    }

    /// How many decodes of `draft` proposals a wave can carry at `resident`
    /// bytes before their copy outruns the layer time — the point past which
    /// the bus sets the step and another decode stops paying. At least one:
    /// the first is always admitted. When the routed set saturates the layer
    /// the copy stops growing with decodes, and if that copy fits the answer is
    /// the cap alone (`usize::MAX`).
    pub fn decode_bus_limit(&self, draft: usize, resident: u64) -> usize {
        let per = self.experts_per_decode(draft).max(1);
        let per_expert_secs = self.geometry.slot_bytes as f64 * self.streamed_fraction(resident)
            / self.bw_bytes_per_s;
        let fit = if per_expert_secs > 0.0 {
            (self.layer_secs / per_expert_secs).floor().max(0.0) as usize / per
        } else {
            usize::MAX
        };
        let saturating = self.geometry.experts_per_layer.div_ceil(per);
        if fit >= saturating {
            usize::MAX
        } else {
            fit.max(1)
        }
    }

    // ── the budget ─────────────────────────────────────────────────────────

    /// Start composing a wave: `resident_bytes` is what the weight side holds
    /// now, `floor_bytes` the residency it may not go under, `max_tokens` the
    /// widest prefill the engine will run in one forward, `max_decodes` the
    /// most decodes it will carry.
    pub fn reset(
        &mut self,
        resident_bytes: u64,
        floor_bytes: u64,
        max_tokens: usize,
        max_decodes: usize,
    ) {
        self.resident_now = resident_bytes;
        self.floor_bytes = floor_bytes;
        self.max_tokens = max_tokens;
        self.max_decodes = max_decodes;
        self.tokens = 0;
        self.decodes = 0;
        self.routed_per_layer = 0;
        self.admitted_any = false;
        self.full = false;
    }

    /// Whether the wave reached a limit this pass. Latched by the first
    /// refusal of either kind; cleared by reset.
    pub fn is_full(&self) -> bool {
        self.full
    }

    /// Prefill rows admitted into the wave since the last reset.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Decodes admitted into the wave since the last reset.
    pub fn decodes(&self) -> usize {
        self.decodes
    }

    /// Experts the admitted decodes route to per layer, before the layer cap.
    pub fn routed_per_layer(&self) -> usize {
        self.routed_per_layer
    }

    /// Resident weights as the wave stands: the reset figure, then the last
    /// admitted offer's `weights_after`.
    pub fn resident_now(&self) -> u64 {
        self.resident_now
    }

    /// Offer one candidate to the wave, in the caller's priority order, with
    /// the resident weights before it and the resident weights it would leave.
    ///
    /// The caller computes the dislodge — the tier rows, K/V and store the
    /// admission takes from the weight side, at the granularity the allocator
    /// actually claims — because only it can; this model reads the two figures
    /// and nothing else about the cost.
    ///
    /// The first candidate after a reset is admitted whatever it costs: a wave
    /// carries at least its head. After that a refusal of either kind latches
    /// [`Self::is_full`] and every later offer is refused as [`Refusal::Full`].
    pub fn try_admit(
        &mut self,
        admission: Admission,
        weights_before: u64,
        weights_after: u64,
    ) -> Admit {
        if self.full {
            return Admit::Refused(Refusal::Full);
        }
        let first = !self.admitted_any;
        let answer = match admission {
            Admission::Prefill { tokens } => {
                self.judge_prefill(tokens, weights_before, weights_after, first)
            }
            Admission::Decode { draft } => {
                self.judge_decode(draft, weights_before, weights_after, first)
            }
        };
        match answer {
            Ok(projected) => {
                self.admitted_any = true;
                self.resident_now = weights_after;
                Admit::Admitted { projected }
            }
            Err(refusal) => {
                self.full = true;
                Admit::Refused(refusal)
            }
        }
    }

    /// Judge a turn that has finished prefilling and is asking to become a
    /// decode — the one decision the decode side gets, and the only place
    /// [`Self::judge_decode`] is ever reached.
    ///
    /// **Why this is not [`Self::try_admit`].** That entry point carries two
    /// behaviours that belong to composing a wave and are wrong here. It
    /// short-circuits on the [`Self::is_full`] latch, so a boundary decision
    /// landing after some *earlier, unrelated* offer was refused would be
    /// refused for that reason rather than on this turn's merits. And it admits
    /// the first candidate after a reset unconditionally, which would hand a
    /// free pass to whichever turn happened to arrive first. A finished prefill
    /// is not competing for a place in this wave; it is asking whether the
    /// decode side can carry it at all.
    ///
    /// **The caller must hand over the counterfactual.** The slot is already
    /// resident — its K/V and store were claimed at prefill admission — so
    /// `weights_before` is residency *as if this turn were not there*
    /// (`resident_now + Cost::dislodged_bytes()`) and `weights_after` is residency
    /// as it actually stands. Offering it any other way asks the model whether
    /// to admit something already admitted, which double-counts its ground and
    /// can only answer yes.
    ///
    /// A refusal does not latch the wave: this turn is being turned away, not
    /// the ones behind it.
    pub fn judge_promotion(
        &mut self,
        draft: usize,
        weights_before: u64,
        weights_after: u64,
    ) -> Admit {
        // **Never refuse on a constant that has not been measured.** `layer_secs`
        // is seeded from the model's advertised decode rate and both measured
        // runs found that seed ~26x optimistic: 0.000813 against a converged
        // 0.018–0.028. While it is still wrong, `T_step`'s `max(layer_secs,
        // copy_secs)` is dominated by the copy term, so every additional decode
        // prices as maximally expensive and the boundary refuses turns it should
        // carry — run 32 evicted 36 that way before the estimate caught up, and
        // run 33 another 10, every one of them `Worse` at three or four decodes.
        //
        // The copy side already refuses to be trusted below `min_learn_rows`;
        // this is the same rule for the compute side. Carrying a turn that
        // should have been refused costs a wave of throughput, while refusing
        // one that should have been carried costs a whole re-prefill — so while
        // the estimate is unlearned, the cheaper error is to admit.
        if self.decode_samples < Self::MIN_DECODE_SAMPLES {
            let projected = self.decode_rate(
                self.decodes + 1,
                self.routed_per_layer
                    .saturating_add(self.experts_per_decode(draft)),
                weights_after,
            );
            self.decodes += 1;
            self.routed_per_layer = self
                .routed_per_layer
                .saturating_add(self.experts_per_decode(draft));
            self.admitted_any = true;
            self.resident_now = weights_after;
            return Admit::Admitted { projected };
        }
        match self.judge_decode(draft, weights_before, weights_after, false) {
            Ok(projected) => {
                self.admitted_any = true;
                self.resident_now = weights_after;
                Admit::Admitted { projected }
            }
            Err(refusal) => Admit::Refused(refusal),
        }
    }

    /// Fold work into the wave that the caller is **not** asking permission
    /// for, and answer with the rate the wave then projects.
    ///
    /// Two kinds of row reach a wave already paid for. A decode step is a
    /// continuation: its ground was bought when its slot was admitted and the
    /// engine owes it the forwards that finish it, so re-judging it refuses
    /// work that has already been paid for — and does it exactly when the
    /// device is tightest, which is when those turns most need to complete and
    /// hand their ground back. And a held creep group rides the next wave
    /// whole, whatever else joins it.
    ///
    /// Neither is a decision, but both are **rows**, and the offers that follow
    /// have to be judged against the wave as it will actually run. So this
    /// records them: the counts move, the residency moves, and — because the
    /// wave is no longer empty — the next offer is judged rather than taken as
    /// the head. Never refuses, and never latches [`Self::is_full`].
    pub fn charge(&mut self, admission: Admission, weights_after: u64) -> f64 {
        let projected = match admission {
            Admission::Prefill { tokens } => {
                self.tokens = self.tokens.saturating_add(tokens);
                self.rate(self.tokens, weights_after)
            }
            Admission::Decode { draft } => {
                self.decodes += 1;
                self.routed_per_layer = self
                    .routed_per_layer
                    .saturating_add(self.experts_per_decode(draft));
                self.decode_rate(self.decodes, self.routed_per_layer, weights_after)
            }
        };
        self.admitted_any = true;
        self.resident_now = weights_after;
        projected
    }

    fn judge_prefill(
        &mut self,
        tokens: usize,
        before: u64,
        after: u64,
        first: bool,
    ) -> Result<f64, Refusal> {
        let tokens_after = self.tokens.saturating_add(tokens);
        if !first {
            if tokens_after > self.max_tokens {
                return Err(Refusal::Cap {
                    max_tokens: self.max_tokens,
                });
            }
            if after < self.floor_bytes {
                return Err(Refusal::Floor {
                    resident_after: after,
                    floor: self.floor_bytes,
                });
            }
        }
        let projected = self.rate(tokens_after, after);
        if self.tokens > 0 {
            let current = self.rate(self.tokens, before);
            Self::judge_gain(current, projected, self.min_gain)?;
        }
        self.tokens = tokens_after;
        Ok(projected)
    }

    fn judge_decode(
        &mut self,
        draft: usize,
        before: u64,
        after: u64,
        first: bool,
    ) -> Result<f64, Refusal> {
        let decodes_after = self.decodes + 1;
        let routed_after = self
            .routed_per_layer
            .saturating_add(self.experts_per_decode(draft));
        if !first {
            if decodes_after > self.max_decodes {
                return Err(Refusal::DecodeCap {
                    max_decodes: self.max_decodes,
                });
            }
            if after < self.floor_bytes {
                return Err(Refusal::Floor {
                    resident_after: after,
                    floor: self.floor_bytes,
                });
            }
        }
        let projected = self.decode_rate(decodes_after, routed_after, after);
        if self.decodes > 0 {
            let current = self.decode_rate(self.decodes, self.routed_per_layer, before);
            Self::judge_gain(current, projected, self.min_gain)?;
        }
        self.decodes = decodes_after;
        self.routed_per_layer = routed_after;
        Ok(projected)
    }

    /// The one rule both kinds share: the rate must not fall, and must rise by
    /// at least `min_gain`. The tolerances keep a flat rate — an admission that
    /// dislodges nothing on a fully resident card, where every row costs exactly
    /// what it earns — from reading as worse by an ulp.
    fn judge_gain(current: f64, projected: f64, min_gain: f64) -> Result<(), Refusal> {
        if projected < current * (1.0 - 1e-9) {
            return Err(Refusal::Worse {
                before: current,
                after: projected,
            });
        }
        let gain = (projected - current) / current;
        if gain + 1e-9 < min_gain {
            return Err(Refusal::Saturated { gain });
        }
        Ok(())
    }

    // ── the feedback ───────────────────────────────────────────────────────

    /// Learn from one prefill forward: `tokens` rows ran in `forward_secs`
    /// with `resident_bytes` of experts on the card.
    ///
    /// The copy is what is left of the forward after the fixed cost and the
    /// tokens' compute, and the observed rate is the non-resident bytes over
    /// it, folded into the estimate by `alpha`. Returns the observed rate, or
    /// `None` when the forward taught nothing: no bytes to copy (a fully
    /// resident card), a forward shorter than its own compute, or a duration
    /// that is not a number. One under [`Self::MIN_LINK_FRACTION`] of the link
    /// is refused as a stall, not a rate.
    ///
    /// **A forward that beats the link raises the link, if it moved more bytes
    /// than the probe did.** The probe is one transfer taken at scheduler start,
    /// while the expert cache is still staging its resident set across the same
    /// bus, so it reads a *contended* figure and under-reports: measured at
    /// 12.03 GB/s on a PCIe 4.0 ×16 link whose forwards then sustained 15.0.
    /// Clamping to that would have pinned the estimate a quarter below the bus
    /// the engine actually has, permanently, on the strength of the worse
    /// measurement. A forward copying at least [`Self::PROBE_BYTES`] is better
    /// evidence about the link than the probe is — it is the same transfer, at
    /// size, under the real workload — so it replaces it; anything smaller is
    /// still clamped, because a rate divided out of a short copy is mostly the
    /// error in the compute model.
    pub fn observe_prefill(
        &mut self,
        tokens: usize,
        resident_bytes: u64,
        forward_secs: f64,
    ) -> Option<f64> {
        if !forward_secs.is_finite() || forward_secs <= 0.0 {
            return None;
        }
        // A forward too narrow to route across the whole layer copied a
        // fraction of the experts this would divide by — see
        // [`Self::min_learn_rows`].
        if tokens < self.min_learn_rows() {
            return None;
        }
        let bytes = self.non_resident_bytes(resident_bytes);
        if bytes == 0 {
            return None;
        }
        // **Both unknowns come out of the same forwards.** A forward is
        // `T = X/bw + W·c` with `X` the non-resident bytes, so one observation
        // is one equation in two unknowns and a *set* of them at different
        // widths and residencies determines both. Folding it in here, before
        // the copy is divided out, is what stops `c` being a constant somebody
        // measured on one card: `fit` carries the running normal equations and
        // solves them the moment the observations have enough spread to be
        // conditioned.
        self.fit
            .observe(bytes as f64, tokens as f64, forward_secs, self.alpha);
        if let Some((bw, c)) = self.fit.solve(self.link_bytes_per_s) {
            self.model.compute_secs_per_token = c;
            self.bw_bytes_per_s = bw;
            self.samples += 1;
            return Some(bw);
        }

        let copy_secs = forward_secs
            - self.model.fixed_secs
            - tokens as f64 * self.model.compute_secs_per_token;
        if copy_secs <= 0.0 {
            return None;
        }
        let raw = bytes as f64 / copy_secs;
        if raw > self.link_bytes_per_s && bytes >= Self::PROBE_BYTES as u64 {
            // **Dampened, not latched.** The link is a ceiling, so raising it on
            // the strength of one forward makes a single fast outlier the new
            // permanent truth — and it can only ever ratchet upward, since
            // nothing below the ceiling moves it back. Moving by `alpha` means
            // sustained evidence raises it and a lone outlier barely does.
            self.link_bytes_per_s += self.alpha * (raw - self.link_bytes_per_s);
        }
        let observed = raw.min(self.link_bytes_per_s);
        if observed < self.link_bytes_per_s * Self::MIN_LINK_FRACTION {
            return None;
        }
        self.bw_bytes_per_s += self.alpha * (observed - self.bw_bytes_per_s);
        self.samples += 1;
        Some(observed)
    }

    /// Learn how well the expert cache converts residency into hits, from the
    /// cache's own counters: `hit` is the fraction of routed experts it found
    /// resident over some interval, at `resident_bytes` on the card.
    ///
    /// The model prices a decode's copy at `1 − hit_rate × resident_fraction`,
    /// so what is learned is the **coefficient** — the observed hit rate over
    /// the residency that produced it. Guessing it is the one thing that cannot
    /// be done well: the LRU zone and the Markov predictor together are worth
    /// far more than the resident fraction alone, and by how much depends on the
    /// workload's routing. On the 35B ingest, 0.65 at 37% resident gives 1.73,
    /// against a seed of 0.7 — a decode's copy priced 2.5× too dear, which the
    /// wave pays for by refusing decodes that would have fitted.
    ///
    /// Returns the observed coefficient, or `None` when the interval says
    /// nothing: no routing at all, or a card holding no experts (from which no
    /// coefficient can be recovered, since every hit came from nowhere).
    /// Dampened by `alpha` like the other estimates.
    pub fn observe_hit_rate(&mut self, hit: f64, resident_bytes: u64) -> Option<f64> {
        if !hit.is_finite() || !(0.0..=1.0).contains(&hit) {
            return None;
        }
        let fraction = self.resident_fraction(resident_bytes);
        if fraction <= 0.0 {
            return None;
        }
        let observed = hit / fraction;
        self.decode.hit_rate += self.alpha * (observed - self.decode.hit_rate);
        self.hit_samples += 1;
        Some(observed)
    }

    /// Intervals the hit coefficient has learned from.
    pub fn hit_samples(&self) -> u64 {
        self.hit_samples
    }

    /// The hit coefficient the model currently believes.
    pub fn hit_rate(&self) -> f64 {
        self.decode.hit_rate
    }

    /// Every expert's bytes — the denominator of the resident fraction.
    pub fn expert_total_bytes(&self) -> u64 {
        self.geometry.total_bytes()
    }

    /// Seconds of compute the model currently believes one prefill row costs.
    ///
    /// The seed until [`CostFit`] has forwards with enough spread to separate
    /// it from the copy; measured from the engine's own forwards after that.
    pub fn compute_secs_per_token(&self) -> f64 {
        self.model.compute_secs_per_token
    }

    /// Whether the two-cost fit has enough spread to be answering, rather than
    /// the planner still running on its seeds.
    pub fn cost_fit_converged(&self) -> bool {
        self.fit.solve(self.link_bytes_per_s).is_some()
    }

    /// Learn from one decode forward: `decodes` sequences each drafting
    /// `draft` proposals stepped in `forward_secs` with `resident_bytes` of
    /// experts on the card.
    ///
    /// A step is `moe_layers` layers of the longer of the layer's own time and
    /// its copy. When the step's per-layer time exceeds the modelled copy the
    /// wave ran compute-bound and that per-layer time *is* the layer time,
    /// folded in by `alpha`. When it does not, the bus set the step and the
    /// forward says nothing about the layer: `None`, and the estimate is left
    /// alone. Also `None` for a forward with no decodes or a duration that is
    /// not a number.
    pub fn observe_decode(
        &mut self,
        decodes: usize,
        draft: usize,
        resident_bytes: u64,
        forward_secs: f64,
    ) -> Option<f64> {
        if decodes == 0 || !forward_secs.is_finite() || forward_secs <= 0.0 {
            return None;
        }
        let routed = decodes.saturating_mul(self.experts_per_decode(draft));
        let per_layer = forward_secs / self.geometry.moe_layers as f64;
        if per_layer <= self.decode_copy_secs(routed, resident_bytes) {
            return None;
        }
        self.layer_secs += self.alpha * (per_layer - self.layer_secs);
        self.decode_samples += 1;
        Some(per_layer)
    }
}

/// Pinned host→device copies of [`WaveRate::PROBE_BYTES`] on the device's
/// stream, timed to completion. Bytes a second, **the fastest** of the runs.
///
/// **The fastest, not the mean, and that is the whole reason there is more than
/// one.** The probe runs at scheduler start, which is while the expert cache is
/// still staging its resident set across the same bus — so a single reading
/// measures whatever contention it happened to land in rather than the link. On
/// the 4090 Mobile that read 2.61 GB/s against a ~25 GB/s PCIe 4.0 ×16 link, an
/// order of magnitude low, and the planner then seeded a copy rate that made a
/// prefill forward look ten seconds long. Contention can only ever make a
/// transfer slower, so the quickest of a few is the closest to the truth, and
/// the estimator learns from forwards afterwards regardless.
fn measure_link_bytes_per_s(device: &Device) -> candle::Result<f64> {
    use candle::cuda_backend::WrapErr;
    use candle::quantized::pinned_staging::PinnedBuf;
    let Device::Cuda(d) = device else {
        candle::bail!("wave rate: the link probe needs a CUDA device");
    };
    let stream = d.cuda_stream();
    let host = PinnedBuf::alloc_owned(WaveRate::PROBE_BYTES)?;
    let mut gpu = unsafe { stream.alloc::<u8>(WaveRate::PROBE_BYTES).w()? };
    // Once untimed, so the first-touch page mapping is not in the measurement.
    stream.memcpy_htod(host.as_slice(), &mut gpu).w()?;
    stream.synchronize().w()?;
    let mut best = 0.0f64;
    for _ in 0..WaveRate::PROBE_RUNS {
        let start = std::time::Instant::now();
        stream.memcpy_htod(host.as_slice(), &mut gpu).w()?;
        stream.synchronize().w()?;
        let secs = start.elapsed().as_secs_f64();
        if secs > 0.0 && !secs.is_nan() {
            best = best.max(WaveRate::PROBE_BYTES as f64 / secs);
        }
    }
    if best <= 0.0 {
        candle::bail!("wave rate: the link probe took no time");
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    /// PCIe 4.0 ×16 as the 4090 Mobile measures it.
    const LINK_4090_MOBILE: f64 = 25e9;
    /// The 3090 box: a PCIe 3.0 host at half that.
    const LINK_3090: f64 = 12e9;
    /// The Blackwell workstation on PCIe 5.0 ×16.
    const LINK_BLACKWELL: f64 = 60e9;
    /// The floor the fill defends on the 16 GB card: hold 4.7 GiB + 512 MiB.
    const FLOOR_16GB: u64 = 5_232 * MIB;
    /// Run 15's residency in calibration.
    const RESIDENT_RUN15: u64 = 7_611 * MIB;
    /// The effective copy rate run 15 implies once the tokens' compute is taken
    /// off the forward: 12.31 GB non-resident over
    /// `1,004 ms − 474 × 0.55 ms` = 743 ms.
    ///
    /// Re-derived when `c` was measured at 0.55 ms rather than 0.42 (see
    /// [`RateModel::SEED_COMPUTE_SECS_PER_TOKEN`]): a larger compute term
    /// leaves less of the forward for the copy, so the same measurement implies
    /// a *faster* bus. It also brought the two independent run-15 forwards into
    /// much closer agreement — 16.56 against 16.48 GB/s, 0.5% apart, where at
    /// 0.42 ms they were 2% apart. Two forwards of different widths agreeing
    /// more closely under the corrected slope is the strongest evidence that
    /// the slope is the thing that was wrong.
    const BW_RUN15: f64 = 15.3e9;
    /// The engine's widest prefill and widest decode set.
    const CAP: usize = 8_192;
    const DECODE_CAP: usize = 64;
    /// What the caller would report a decode admission dislodging on this
    /// model: its 160 MiB recurrent store and a few K/V regions.
    const DECODE_DISLODGE: u64 = 176 * MIB;
    const SLOT: f64 = 1_933_312.0;
    const LAYERS: f64 = 41.0;

    fn planner(link: f64) -> WaveRate {
        WaveRate::with_link_rate(
            link,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        )
    }

    /// The planner as run 15 measured it: effective rate learned, not seeded.
    ///
    /// Taught with a forward wide enough to route across the whole layer
    /// ([`WaveRate::min_learn_rows`]), because a narrower one copies a fraction
    /// of the experts the model divides by and is refused.
    fn planner_learned(link: f64, bw: f64) -> WaveRate {
        let mut p = planner(link).with_alpha(1.0);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15) as f64;
        let rows = p.min_learn_rows();
        let secs = bytes / bw + rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        p.observe_prefill(rows, RESIDENT_RUN15, secs)
            .expect("a wide enough forward");
        assert!(approx(p.effective_bytes_per_s(), bw, 1e-9));
        p.with_alpha(WaveRate::DEFAULT_ALPHA)
    }

    /// The same planner with the layer time learned to `layer_secs` exactly:
    /// one plain decode at run 15's residency, compute-bound, so the step is
    /// the layer time over every MoE layer.
    fn planner_with_layer(link: f64, bw: f64, layer_secs: f64) -> WaveRate {
        let mut p = planner_learned(link, bw).with_alpha(1.0);
        assert!(p.decode_copy_secs(8, RESIDENT_RUN15) < layer_secs);
        p.observe_decode(1, 0, RESIDENT_RUN15, layer_secs * LAYERS);
        assert!(approx(p.layer_secs(), layer_secs, 1e-9));
        p.with_alpha(WaveRate::DEFAULT_ALPHA)
    }

    fn approx(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-12)
    }

    fn prefill(tokens: usize) -> Admission {
        Admission::Prefill { tokens }
    }

    fn decode(draft: usize) -> Admission {
        Admission::Decode { draft }
    }

    /// Offer `a` as the caller would: weights before are the wave's now,
    /// weights after are those less what this admission dislodges.
    fn offer(p: &mut WaveRate, a: Admission, dislodge: u64) -> Admit {
        let before = p.resident_now();
        p.try_admit(a, before, before.saturating_sub(dislodge))
    }

    /// A planner whose layer time is both **set** and **learned**: past
    /// [`WaveRate::MIN_DECODE_SAMPLES`], so `judge_promotion` actually judges
    /// rather than taking the unlearned-estimate bypass.
    fn planner_judging(layer_secs: f64) -> WaveRate {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, layer_secs);
        while p.decode_samples() < WaveRate::MIN_DECODE_SAMPLES {
            p.observe_decode(1, 0, RESIDENT_RUN15, layer_secs * LAYERS);
        }
        assert!(
            approx(p.layer_secs(), layer_secs, 1e-9),
            "still at the target"
        );
        p
    }

    /// Offer a finished prefill for promotion the way the scheduler does: the
    /// slot is already resident, so `before` is residency with its ground added
    /// back and `after` is residency as it stands.
    fn promote(p: &mut WaveRate, draft: usize, held: u64) -> Admit {
        let after = p.resident_now();
        p.judge_promotion(draft, after.saturating_add(held), after)
    }

    /// **A latched wave must not decide a promotion.** `try_admit` refuses
    /// everything once any offer has been refused, which is right while
    /// composing a wave and wrong at the boundary: a finished prefill would be
    /// turned away because some earlier, unrelated offer did not fit, and the
    /// turn's own merits would never be read.
    #[test]
    fn a_promotion_is_judged_even_after_the_wave_latched_full() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        // The first offer after a reset takes the head bypass, so admit one to
        // clear it, then latch the wave on something that cannot fit.
        assert!(matches!(
            offer(&mut p, decode(2), DECODE_DISLODGE),
            Admit::Admitted { .. }
        ));
        let before = p.resident_now();
        let latched = p.try_admit(decode(2), before, FLOOR_16GB.saturating_sub(1));
        assert!(matches!(latched, Admit::Refused(_)), "{latched:?}");
        assert!(p.is_full(), "the wave is latched");

        // The boundary still gets a real answer.
        assert!(
            matches!(promote(&mut p, 2, DECODE_DISLODGE), Admit::Admitted { .. }),
            "a promotion is judged on its own cost, not the wave's latch"
        );
    }

    /// **A refused promotion turns away one turn, not the queue behind it.**
    /// Latching here would make the first refusal at the boundary evict every
    /// turn that finished prefilling after it, whatever the decode side could
    /// actually carry.
    #[test]
    fn a_refused_promotion_does_not_latch_the_wave() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let after = p.resident_now();
        let refused = p.judge_promotion(2, after.saturating_add(DECODE_DISLODGE), FLOOR_16GB - 1);
        assert!(
            matches!(refused, Admit::Refused(Refusal::Floor { .. })),
            "{refused:?}"
        );
        assert!(!p.is_full(), "the wave carries on for the turns behind it");
        assert!(
            matches!(promote(&mut p, 2, DECODE_DISLODGE), Admit::Admitted { .. }),
            "the next boundary decision is still made on its merits"
        );
    }

    /// **While the layer time is unlearned the boundary admits, whatever the
    /// numbers say.** The seed is ~26x optimistic, so `T_step` is dominated by
    /// the copy term and every promotion prices as ruinous — run 32 evicted 36
    /// turns that way and run 33 another 10, all `Worse` at three or four
    /// decodes, and all of them wrong once the estimate caught up.
    #[test]
    fn an_unlearned_layer_time_never_refuses_a_promotion() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(
            p.decode_samples() < WaveRate::MIN_DECODE_SAMPLES,
            "the estimate has not been learned"
        );
        // Even an offer that would breach the floor is carried while the model
        // has no measured basis to refuse on.
        let after = p.resident_now();
        assert!(matches!(
            p.judge_promotion(2, after.saturating_add(DECODE_DISLODGE), FLOOR_16GB - 1),
            Admit::Admitted { .. }
        ));
    }

    /// And once it *is* learned, the same offer is refused — so the bypass is a
    /// window, not a permanent hole.
    #[test]
    fn a_learned_layer_time_restores_the_refusal() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let after = p.resident_now();
        assert!(matches!(
            p.judge_promotion(2, after.saturating_add(DECODE_DISLODGE), FLOOR_16GB - 1),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }

    /// The promotion that would take residency under the floor is refused —
    /// the same hard line every other admission answers to.
    #[test]
    fn a_promotion_under_the_floor_is_refused() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let after = p.resident_now();
        match p.judge_promotion(2, after, FLOOR_16GB - 1) {
            Admit::Refused(Refusal::Floor { floor, .. }) => assert_eq!(floor, FLOOR_16GB),
            other => panic!("{other:?}"),
        }
    }

    /// Offer `a` with `dislodge` until refused; the count admitted and the
    /// refusal.
    fn fill_with(p: &mut WaveRate, a: Admission, dislodge: u64) -> (usize, Refusal) {
        let mut n = 0;
        loop {
            match offer(p, a, dislodge) {
                Admit::Admitted { .. } => n += 1,
                Admit::Refused(r) => return (n, r),
            }
        }
    }

    /// A 250-token exemplar's tier at about a MiB a row.
    const EXEMPLAR: usize = 250;
    const EXEMPLAR_DISLODGE: u64 = 250 * MIB;

    // ── geometry ───────────────────────────────────────────────────────────

    #[test]
    fn the_35b_carries_twenty_gigabytes_of_experts() {
        let g = ExpertGeometry::QWEN36_35B_A3B;
        assert_eq!(g.total_bytes(), 41 * 256 * 1_933_312);
        assert!(approx(g.total_bytes() as f64, 20.29e9, 0.01));
    }

    // ── the prefill model reproduces the measured run ──────────────────────

    /// Run 15: 7.5–7.7 GiB resident, 474 tokens, 1,004 ms. With the copy rate
    /// the run implied the model lands on the measured forward.
    #[test]
    fn the_model_reproduces_run_15s_forward() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let secs = p.forward_secs(474, RESIDENT_RUN15);
        assert!(
            approx(secs, 1.004, 0.02),
            "projected {secs:.3} s vs 1.004 s measured"
        );
        assert!(approx(p.rate(474, RESIDENT_RUN15), 472.0, 0.02));
    }

    /// **A second forward from the same run, independently.** The fixture's
    /// 15.3 GB/s was derived from one 474-token forward; run 15's ingest phase
    /// logged many more, and a different one — 567 rows in 1,068 ms at 7,467 MiB
    /// of effective residency — implies 15.0 GB/s through the same arithmetic.
    /// Two forwards of different widths and residencies agreeing to 2% is what
    /// makes the copy rate a property of the machine rather than a fit to one
    /// measurement, and it is the check that would fail first if the model's
    /// shape were wrong.
    #[test]
    fn a_second_measured_forward_implies_the_same_copy_rate() {
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        let resident = 7_467 * MIB;
        let observed = p
            .observe_prefill(567, resident, 1.068)
            .expect("an observation");
        assert!(approx(observed, 15.0e9, 0.02), "{observed}");
        assert!(
            approx(observed, BW_RUN15, 0.03),
            "within 3% of the fixture's rate: {observed} vs {BW_RUN15}",
        );
        // And the model then reproduces the forward it learned from.
        assert!(approx(p.forward_secs(567, resident), 1.068, 1e-9));
        assert!(approx(p.rate(567, resident), 531.0, 0.01));
    }

    /// The width dependence run 15 measured at fixed weights: 0.42 ms a token,
    /// and the copy the rest.
    #[test]
    fn the_per_token_slope_matches_the_measured_widths() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let slope =
            (p.forward_secs(1_000, RESIDENT_RUN15) - p.forward_secs(0, RESIDENT_RUN15)) / 1_000.0;
        assert!(approx(slope, RateModel::SEED_COMPUTE_SECS_PER_TOKEN, 1e-9));
        let copy = p.non_resident_bytes(RESIDENT_RUN15) as f64 / BW_RUN15;
        assert!(approx(p.forward_secs(0, RESIDENT_RUN15), copy, 1e-9));
    }

    /// Width is what buys throughput: two exemplars to eight is ~2.4× the rate
    /// for ~70% more forward time, because compute starts to count at 1,900
    /// tokens while the copy stays ~0.9 s.
    #[test]
    fn widening_the_wave_multiplies_the_rate() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let r = RESIDENT_RUN15;
        let two = p.rate(474, r);
        let eight = p.rate(1_896, r - 1_422 * MIB);
        assert!(
            eight / two > 2.2 && eight / two < 2.6,
            "ratio {}",
            eight / two
        );
        let time_ratio = p.forward_secs(1_896, r - 1_422 * MIB) / p.forward_secs(474, r);
        assert!(
            time_ratio > 1.5 && time_ratio < 1.8,
            "time ratio {time_ratio}"
        );
    }

    /// With residency held, the rate climbs toward the compute ceiling and
    /// never crosses it; a fixed per-forward cost is what it amortises.
    #[test]
    fn the_rate_saturates_at_the_compute_ceiling() {
        let model = RateModel {
            fixed_secs: 0.2,
            ..RateModel::default()
        };
        let p = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            model,
            DecodeModel::default(),
        );
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let asym = p.asymptotic_rate();
        assert!(approx(
            asym,
            1.0 / RateModel::SEED_COMPUTE_SECS_PER_TOKEN,
            1e-12
        ));
        let mut last = 0.0;
        for w in [128usize, 512, 2_048, 8_192, 16_384] {
            let rate = p.rate(w, total);
            assert!(rate > last, "monotone in width");
            assert!(rate < asym, "under the ceiling at {w}");
            last = rate;
        }
        assert!(
            approx(last, asym, 0.05),
            "16k tokens is within 5% of the ceiling"
        );
    }

    /// The model's answer for the 16 GB card as run 15 stood, with exemplars
    /// dislodging a MiB a row: 2,379 tokens a wave at ~1,210 tok/s, against
    /// 472 today.
    #[test]
    fn the_16gb_card_widens_to_the_floor_at_twelve_hundred_tokens_a_second() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let projected = p.rate(2_379, FLOOR_16GB);
        assert!(approx(projected, 1_210.0, 0.01), "{projected}");
    }

    // ── the prefill budget on the 16 GB card ───────────────────────────────

    /// Reset, then admit exemplars until the floor stops it: on run 15's
    /// residency that is nine 250-token exemplars, the refusal names the floor
    /// with the residency the tenth would have left, and the wave is full.
    #[test]
    fn prefill_admissions_stop_at_the_floor_and_say_so() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE);
        assert_eq!(
            admitted, 9,
            "2,379 MiB of room at a MiB a row is nine 250-row exemplars"
        );
        assert_eq!(p.tokens(), 2_250);
        match refusal {
            Refusal::Floor {
                resident_after,
                floor,
            } => {
                assert_eq!(floor, FLOOR_16GB);
                assert_eq!(resident_after, RESIDENT_RUN15 - 2_500 * MIB);
            }
            other => panic!("expected the floor, got {other:?}"),
        }
        assert!(p.is_full());
        assert_eq!(
            p.resident_now(),
            RESIDENT_RUN15 - 2_250 * MIB,
            "a refusal moves nothing"
        );
    }

    /// The caller's dislodge is what the decision reads: the same tokens with
    /// no dislodge at all widen to the cap, and with a heavy dislodge stop on
    /// the rate, well above the floor.
    #[test]
    fn the_callers_dislodge_drives_the_decision() {
        // Nothing dislodged: the rate only climbs, the floor is never nearer.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, 2_000, DECODE_CAP);
        let (n, refusal) = fill_with(&mut p, prefill(EXEMPLAR), 0);
        assert_eq!(n, 8);
        assert!(matches!(refusal, Refusal::Cap { max_tokens: 2_000 }));
        assert_eq!(p.resident_now(), RESIDENT_RUN15);
        // A GiB dislodged for 32 tokens: 70 ms of copy against 13 ms of
        // compute, on a wave already at 817 tok/s. Refused as worse, with two
        // GiB still above the floor.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 0).is_admitted());
        let (n, refusal) = fill_with(&mut p, prefill(32), GIB);
        assert_eq!(n, 0);
        assert!(
            matches!(refusal, Refusal::Worse { before, after } if after < before),
            "{refusal:?}"
        );
        assert_eq!(
            p.resident_now(),
            RESIDENT_RUN15,
            "stopped on the rate, not the floor"
        );
    }

    /// An admission whose dislodged weights cost more copy than its tokens
    /// repay is refused as worse, with both rates.
    #[test]
    fn an_admission_that_costs_more_copy_than_it_repays_is_refused_as_worse() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 0).is_admitted());
        let current = p.rate(1_000, RESIDENT_RUN15);
        // 64 tokens that dislodge 2 GiB: 27 ms of tokens against 140 ms of copy.
        match offer(&mut p, prefill(64), 2 * GIB) {
            Admit::Refused(Refusal::Worse { before, after }) => {
                assert!(approx(before, current, 1e-9));
                assert!(after < before);
            }
            other => panic!("expected worse, got {other:?}"),
        }
        assert!(p.is_full());
    }

    /// A dislodge in proportion to the rows — the tier at a MiB a row — never
    /// makes widening worse: `W / (F + k·W)` only climbs. The stop is then the
    /// floor or saturation.
    #[test]
    fn a_proportional_dislodge_never_makes_widening_worse() {
        let mut p = planner(2e6).with_min_gain(0.01);
        let resident = ExpertGeometry::QWEN36_35B_A3B.total_bytes() - 64 * MIB;
        p.reset(resident, 0, 1 << 20, DECODE_CAP);
        let mut last = 0.0;
        let refusal = loop {
            match offer(&mut p, prefill(64), 64 * MIB) {
                Admit::Admitted { projected } => {
                    assert!(projected > last, "monotone");
                    last = projected;
                }
                Admit::Refused(r) => break r,
            }
        };
        assert!(matches!(refusal, Refusal::Saturated { .. }), "{refusal:?}");
        assert!(
            p.tokens() >= 512 && p.tokens() <= 1_024,
            "tokens {}",
            p.tokens()
        );
    }

    /// The first offer after a reset is taken whatever it costs — under the
    /// floor, over the cap, past the bus — because a wave has to carry its
    /// head. The second is judged.
    #[test]
    fn the_first_admission_after_a_reset_is_always_taken() {
        // A prefill that would dislodge past the floor and is over the cap.
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(FLOOR_16GB + GIB, FLOOR_16GB, 100, DECODE_CAP);
        assert!(offer(&mut p, prefill(5_000), 2 * GIB).is_admitted());
        assert!(!p.is_full());
        assert!(
            p.resident_now() < FLOOR_16GB,
            "the head took the weights under the floor"
        );
        assert!(matches!(
            offer(&mut p, prefill(1), 0),
            Admit::Refused(Refusal::Cap { .. })
        ));
        // A decode whose copy alone exceeds the layer time at the seed.
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(
            p.decode_copy_secs(24, RESIDENT_RUN15) > p.layer_secs(),
            "the seed cannot hide one drafted decode"
        );
        assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        assert_eq!(p.decodes(), 1);
        // A decode into a wave whose cap is zero.
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, 0);
        assert!(offer(&mut p, decode(0), DECODE_DISLODGE).is_admitted());
        assert!(matches!(
            offer(&mut p, decode(0), DECODE_DISLODGE),
            Admit::Refused(Refusal::DecodeCap { max_decodes: 0 })
        ));
    }

    /// Barely resident: the weight side stands one exemplar above the floor.
    /// The first fits, the second would cross, and the refusal is the floor.
    #[test]
    fn a_card_at_the_floor_admits_what_fits_and_refuses_the_rest() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(FLOOR_16GB + 300 * MIB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE),
            Admit::Refused(Refusal::Floor { .. })
        ));
        assert!(p.is_full(), "the floor latched the wave full");
    }

    /// Exactly to the floor is allowed; one byte past it is not.
    #[test]
    fn the_floor_itself_is_reachable_and_not_crossable() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(FLOOR_16GB + 300 * MIB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE).is_admitted());
        assert!(
            offer(&mut p, prefill(50), 50 * MIB).is_admitted(),
            "exactly to the floor"
        );
        assert_eq!(p.resident_now(), FLOOR_16GB);
        assert!(matches!(
            offer(&mut p, prefill(1), 1),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }

    /// Under the floor already: the head is still taken (the caller's rule),
    /// and nothing else is — even an offer that dislodges nothing, because the
    /// weights after it are still under the floor.
    #[test]
    fn a_card_under_the_floor_takes_only_its_head() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(FLOOR_16GB - MIB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(200), 0).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(1), 0),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }

    /// The cap is the engine's widest forward, and it holds even with residency
    /// to spare.
    #[test]
    fn the_cap_bounds_the_wave_whatever_the_rate_says() {
        let mut p = planner(LINK_4090_MOBILE).with_min_gain(0.0);
        p.reset(9_600 * MIB, FLOOR_16GB, 1_024, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 1_000 * MIB).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(25), 25 * MIB),
            Admit::Refused(Refusal::Cap { max_tokens: 1_024 })
        ));
        assert!(p.is_full());
    }

    /// Reset forgets the wave — tokens, decodes, residency, `full` — and keeps
    /// the learning.
    #[test]
    fn reset_clears_the_budget_and_keeps_the_estimates() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        let (bw, layer) = (p.effective_bytes_per_s(), p.layer_secs());
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        assert!(offer(&mut p, prefill(500), 500 * MIB).is_admitted());
        let _ = fill_with(&mut p, prefill(500), 500 * MIB);
        assert!(p.is_full());
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert_eq!(p.tokens(), 0);
        assert_eq!(p.decodes(), 0);
        assert_eq!(p.routed_per_layer(), 0);
        assert!(!p.is_full());
        assert_eq!(p.resident_now(), RESIDENT_RUN15);
        assert_eq!(p.effective_bytes_per_s(), bw);
        assert_eq!(p.layer_secs(), layer);
    }

    /// When the copy is small (most of the experts resident) and the wave is
    /// already wide, another admission buys almost nothing, and the planner
    /// stops widening rather than spend residency for a fraction of a percent.
    #[test]
    fn a_saturated_wave_refuses_to_spend_residency_for_nothing() {
        // Blackwell: 20.3 GB of experts, 19.9 GB resident, fast bus.
        let mut p = planner(LINK_BLACKWELL).with_min_gain(0.01);
        let resident = 19_000 * MIB;
        p.reset(resident, 8 * GIB, 65_536, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, prefill(512), 512 * MIB);
        assert!(
            matches!(refusal, Refusal::Saturated { gain } if (0.0..0.01).contains(&gain)),
            "{refusal:?}"
        );
        assert!(p.resident_now() > 8 * GIB + GIB, "well short of the floor");
        assert!(p.tokens() < 65_536, "well short of the cap");
        assert!(admitted >= 2, "at least the first couple always pay");
    }

    /// With no minimum gain, only the floor or the cap ends a prefill wave
    /// whose dislodge is proportional.
    #[test]
    fn with_no_minimum_gain_only_the_floor_or_cap_ends_the_wave() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let (_, refusal) = fill_with(&mut p, prefill(128), 128 * MIB);
        assert!(matches!(refusal, Refusal::Floor { .. }), "{refusal:?}");
    }

    // ── the decode model: what residency does to the copy ──────────────────

    /// The streamed fraction is what the zone and the predictor miss: 30% with
    /// everything resident, everything with nothing resident, and in between a
    /// straight line through the resident fraction — run 15's 39% resident
    /// streams 72% of what a decode routes to.
    #[test]
    fn weight_pressure_raises_the_streamed_fraction() {
        let p = planner(LINK_4090_MOBILE);
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        assert!(approx(p.streamed_fraction(total), 0.3, 1e-12));
        assert!(approx(p.streamed_fraction(0), 1.0, 1e-12));
        assert!(approx(p.resident_fraction(RESIDENT_RUN15), 0.3933, 1e-3));
        assert!(approx(p.streamed_fraction(RESIDENT_RUN15), 0.7247, 1e-3));
        assert!(approx(p.streamed_fraction(total / 2), 0.65, 1e-12));
        assert!(
            p.streamed_fraction(RESIDENT_RUN15 - GIB) > p.streamed_fraction(RESIDENT_RUN15),
            "less resident, more streamed"
        );
        assert!(
            approx(p.resident_fraction(2 * total), 1.0, 1e-12),
            "capped at the whole model"
        );
    }

    /// The seed: one step at 30 tok/s spread over 41 MoE layers is 0.81 ms a
    /// layer. At run 15's residency a plain decode's eight experts stream 72%
    /// of 15.5 MB in 0.56 ms at the 80% seed of the 4090's link — it fits — and
    /// a drafted decode's 24 do not.
    #[test]
    fn the_layer_time_is_seeded_from_thirty_tokens_a_second() {
        let p = planner(LINK_4090_MOBILE);
        assert!(approx(p.layer_secs(), 1.0 / (30.0 * LAYERS), 1e-12));
        assert_eq!(p.experts_per_decode(0), 8);
        assert_eq!(p.experts_per_decode(2), 24);
        let streamed = p.streamed_fraction(RESIDENT_RUN15);
        assert!(approx(
            p.decode_layer_bytes(8, RESIDENT_RUN15),
            8.0 * SLOT * streamed,
            1e-12
        ));
        assert!(approx(
            p.decode_copy_secs(8, RESIDENT_RUN15),
            8.0 * SLOT * streamed / 20e9,
            1e-9
        ));
        assert!(approx(
            p.decode_copy_secs(8, RESIDENT_RUN15),
            0.5604e-3,
            1e-3
        ));
        assert!(
            p.decode_copy_secs(8, RESIDENT_RUN15) < p.layer_secs(),
            "a plain decode hides under the seed"
        );
        assert!(
            p.decode_copy_secs(24, RESIDENT_RUN15) > p.layer_secs(),
            "a drafted decode does not"
        );
    }

    /// A step is the longer of the layer and its copy over every layer, and
    /// the decode rate is the decodes over that: under the layer the rate grows
    /// with each decode, past it the bus holds the rate flat — until the
    /// decodes between them route to every expert of the layer, after which
    /// another decode adds no copy and the rate climbs again.
    #[test]
    fn the_decode_rate_grows_until_the_bus_holds_it_flat() {
        let p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        let r = RESIDENT_RUN15;
        // 2.20 ms of copy a drafted decode a layer: nine hide under 20 ms.
        assert!(approx(p.decode_copy_secs(24, r), 2.1977e-3, 1e-3));
        assert!(approx(p.decode_step_secs(24 * 9, r), LAYERS * 20e-3, 1e-12));
        assert!(approx(
            p.decode_step_secs(24 * 10, r),
            LAYERS * 10.0 * p.decode_copy_secs(24, r),
            1e-12
        ));
        assert!(p.decode_rate(9, 24 * 9, r) > p.decode_rate(8, 24 * 8, r));
        assert_eq!(p.decode_rate(0, 0, r), 0.0);
        assert_eq!(p.decode_bus_limit(2, r), 9);
        // Plain decodes: 27 hide, the 28th and 29th are both on the bus with
        // the layer unsaturated, and their rates are equal.
        assert_eq!(p.decode_bus_limit(0, r), 27);
        assert!(
            approx(
                p.decode_rate(29, 8 * 29, r),
                p.decode_rate(28, 8 * 28, r),
                1e-9
            ),
            "flat on the bus"
        );
        // Eleven drafted decodes route to 264 ≥ 256 experts: the twelfth adds
        // no copy, and the rate climbs with it.
        assert!(approx(
            p.decode_step_secs(24 * 12, r),
            p.decode_step_secs(24 * 11, r),
            1e-12
        ));
        assert!(p.decode_rate(12, 24 * 12, r) > p.decode_rate(11, 24 * 11, r));
    }

    /// Decodes are admitted while the decode rate improves. At a learned 20 ms
    /// layer and run 15's bus, with each decode's store dislodging 176 MiB:
    /// eight hide under the layer, the ninth puts the wave on the bus but still
    /// lifts the rate, and the tenth — its own copy plus the miss rate every
    /// decode's dislodge raised — lowers it, and is refused as worse.
    #[test]
    fn decodes_stop_when_another_would_lower_the_decode_rate() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, decode(2), DECODE_DISLODGE);
        assert_eq!(admitted, 9);
        assert_eq!(p.routed_per_layer(), 216);
        assert_eq!(p.resident_now(), RESIDENT_RUN15 - 9 * DECODE_DISLODGE);
        match refusal {
            Refusal::Worse { before, after } => {
                assert!(approx(
                    before,
                    p.decode_rate(9, 216, p.resident_now()),
                    1e-9
                ));
                assert!(after < before);
                assert!(approx(before, 10.285, 1e-3), "{before}");
                assert!(approx(after, 10.202, 1e-3), "{after}");
            }
            other => panic!("expected worse, got {other:?}"),
        }
        assert!(p.is_full());
        assert!(
            p.resident_now() > FLOOR_16GB,
            "stopped on the rate, not the floor"
        );
    }

    /// At fixed residency — decodes that dislodge nothing — the bus limit ends
    /// the wave as saturation: once the copy sets the step, another decode
    /// stretches it in exact proportion and the rate does not move. Plain
    /// decodes at an 18 ms layer: 24 hide, the 25th puts the wave on the bus
    /// and still lifts the rate by 2%, the 26th lifts it by nothing.
    #[test]
    fn at_fixed_residency_the_bus_limit_saturates_the_decode_rate() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 18e-3);
        assert_eq!(p.decode_bus_limit(0, RESIDENT_RUN15), 24);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, decode(0), 0);
        assert_eq!(admitted, 25);
        assert!(
            matches!(refusal, Refusal::Saturated { gain } if gain.abs() < 1e-9),
            "{refusal:?}"
        );
        assert_eq!(p.resident_now(), RESIDENT_RUN15);
    }

    /// Drafted decodes saturate the layer before the bus stops paying at this
    /// residency and layer time: eleven route to every expert, and from there
    /// each decode adds no copy, so at fixed residency only the cap ends the
    /// wave.
    #[test]
    fn drafted_decodes_that_saturate_the_layer_run_to_the_cap_at_fixed_residency() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, 40);
        let (admitted, refusal) = fill_with(&mut p, decode(2), 0);
        assert_eq!(admitted, 40);
        assert!(
            matches!(refusal, Refusal::DecodeCap { max_decodes: 40 }),
            "{refusal:?}"
        );
        assert_eq!(p.routed_per_layer(), 960);
    }

    /// The refinement itself: the same decodes on the same card, once with
    /// nothing dislodged and once with half a GiB a decode. The dislodge lowers
    /// the hit rate for every decode already in the wave, so the copy outruns
    /// the layer sooner and the rate turns down where the fixed-residency wave
    /// was still climbing — nine admitted against seven.
    #[test]
    fn dislodged_weights_refuse_decodes_a_fixed_residency_wave_would_take() {
        let resident = 14 * GIB;
        let floor = 6 * GIB;
        let fixed = {
            let mut p = planner_with_layer(LINK_3090, 9.6e9, 20e-3);
            p.reset(resident, floor, CAP, DECODE_CAP);
            let (n, refusal) = fill_with(&mut p, decode(2), 0);
            assert!(matches!(refusal, Refusal::Saturated { .. }), "{refusal:?}");
            n
        };
        let dislodging = {
            let mut p = planner_with_layer(LINK_3090, 9.6e9, 20e-3);
            p.reset(resident, floor, CAP, DECODE_CAP);
            let (n, refusal) = fill_with(&mut p, decode(2), 512 * MIB);
            assert!(
                matches!(refusal, Refusal::Worse { before, after } if after < before),
                "{refusal:?}"
            );
            assert!(
                p.resident_now() > floor + 4 * GIB,
                "the floor was nowhere near"
            );
            n
        };
        assert_eq!(fixed, 9);
        assert_eq!(dislodging, 7);
    }

    /// A decode's dislodge is judged against the floor like a prefill's: with
    /// the layer time generous, the floor is what ends a decode wave that keeps
    /// taking stores.
    #[test]
    fn a_decode_that_would_dislodge_under_the_floor_is_refused_on_the_floor() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 40e-3);
        // Room for exactly three stores above the floor.
        p.reset(
            FLOOR_16GB + 3 * DECODE_DISLODGE,
            FLOOR_16GB,
            CAP,
            DECODE_CAP,
        );
        let (admitted, refusal) = fill_with(&mut p, decode(0), DECODE_DISLODGE);
        assert_eq!(admitted, 3);
        assert!(matches!(refusal, Refusal::Floor { .. }), "{refusal:?}");
        assert_eq!(p.resident_now(), FLOOR_16GB);
    }

    /// The admitted projection is the decode rate the wave would run at,
    /// climbing with each decode under the layer.
    #[test]
    fn a_decode_admission_reports_the_projected_decode_rate() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let mut last = 0.0;
        for n in 1..=8 {
            match offer(&mut p, decode(2), DECODE_DISLODGE) {
                Admit::Admitted { projected } => {
                    assert!(projected > last, "decode {n}: {projected}");
                    assert!(approx(
                        projected,
                        p.decode_rate(n, 24 * n, p.resident_now()),
                        1e-9
                    ));
                    assert!(
                        approx(projected, n as f64 / (LAYERS * 20e-3), 1e-9),
                        "under the layer"
                    );
                    last = projected;
                }
                other => panic!("decode {n}: {other:?}"),
            }
        }
    }

    /// Speculation is what a decode pays in: at run 15's residency and a 20 ms
    /// layer, 27 plain decodes hide where 9 drafted ones or 3 wide-drafted ones
    /// do.
    #[test]
    fn plain_decodes_fit_three_times_as_many_as_drafted_ones() {
        let p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        assert_eq!(p.decode_bus_limit(0, RESIDENT_RUN15), 27);
        assert_eq!(p.decode_bus_limit(2, RESIDENT_RUN15), 9);
        assert_eq!(
            p.decode_bus_limit(7, RESIDENT_RUN15),
            3,
            "64 experts a decode"
        );
    }

    /// Once the admitted decodes route to every expert of a layer, the copy is
    /// the whole layer's streamed share whatever else joins, so the limit is
    /// that copy against the layer time — and when it fits, only the cap ends
    /// the wave.
    #[test]
    fn a_saturated_layer_costs_the_same_however_many_decodes_join() {
        // The whole layer at run 15's residency: 256 experts, 72% streamed, at
        // 15.3 GB/s is 23.4 ms; give the layer 40.
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 40e-3);
        assert!(approx(
            p.decode_copy_secs(256, RESIDENT_RUN15),
            23.44e-3,
            1e-3
        ));
        assert_eq!(
            p.decode_bus_limit(7, RESIDENT_RUN15),
            usize::MAX,
            "nothing but the cap bounds it"
        );
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, 12);
        let (admitted, refusal) = fill_with(&mut p, decode(7), 8 * MIB);
        assert_eq!(admitted, 12);
        assert!(
            matches!(refusal, Refusal::DecodeCap { max_decodes: 12 }),
            "{refusal:?}"
        );
        // 12 × 64 = 768 routed, but a layer has 256: the copy is capped there.
        assert_eq!(p.routed_per_layer(), 768);
        let r = p.resident_now();
        assert!(approx(
            p.decode_layer_bytes(768, r),
            p.decode_layer_bytes(256, r),
            1e-12
        ));
    }

    /// The slower bus hides fewer decodes for the same layer time; the faster
    /// one more. Same formula, different card.
    #[test]
    fn the_bus_sets_how_many_decodes_a_layer_can_hide() {
        // 3090 at its 9.6 GB/s seed with 12 GiB resident: 24 experts, 56%
        // streamed, 2.69 ms a decode — seven fit in 20 ms.
        let p = planner_with_layer(LINK_3090, 9.6e9, 20e-3);
        assert!(approx(p.decode_copy_secs(24, 12 * GIB), 2.685e-3, 1e-3));
        assert_eq!(p.decode_bus_limit(2, 12 * GIB), 7);
        // Blackwell at its 48 GB/s seed with 19 GB resident: 31% streamed,
        // 0.30 ms a decode; eleven drafted decodes route to every expert and
        // the whole layer's share is 3.2 ms, so the cap alone bounds the wave.
        let p = planner_with_layer(LINK_BLACKWELL, 48e9, 20e-3);
        assert!(approx(
            p.decode_copy_secs(256, 19_000 * MIB),
            3.224e-3,
            1e-3
        ));
        assert_eq!(p.decode_bus_limit(2, 19_000 * MIB), usize::MAX);
    }

    /// A fully resident card still streams what the predictor misses — 30% of
    /// every routed expert — but the whole layer's share is 9.7 ms against a
    /// 20 ms layer, so the bus never limits it, where the same layer at run
    /// 15's residency streams 72% and hides nine drafted decodes.
    #[test]
    fn a_fully_resident_card_still_pays_the_miss_rate() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        assert!(approx(p.decode_copy_secs(24, total), 0.9098e-3, 1e-3));
        assert!(approx(p.decode_copy_secs(256, total), 9.705e-3, 1e-3));
        assert_eq!(p.decode_bus_limit(2, total), usize::MAX);
        assert_eq!(p.decode_bus_limit(2, RESIDENT_RUN15), 9);
        // Plain decodes: 32 route to every expert, and at full residency 65
        // would hide, so the layer saturates first there too; at run 15's
        // residency 27 hide, short of saturation, and the bus is the limit.
        assert_eq!(p.decode_bus_limit(0, total), usize::MAX);
        assert_eq!(p.decode_bus_limit(0, RESIDENT_RUN15), 27);
    }

    // ── one budget, one `full` ─────────────────────────────────────────────

    /// Decodes first, then prefills: the decodes reach their limit, and that
    /// refusal closes the wave to the prefills behind them.
    #[test]
    fn a_decode_refusal_closes_the_wave_to_later_prefills() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let (nine, refusal) = fill_with(&mut p, decode(2), DECODE_DISLODGE);
        assert_eq!(nine, 9);
        assert!(matches!(refusal, Refusal::Worse { .. }));
        assert!(p.is_full());
        assert!(matches!(
            offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE),
            Admit::Refused(Refusal::Full)
        ));
        assert_eq!(p.tokens(), 0, "no prefill got in");
    }

    /// Decodes under their limit leave the wave open, and prefills then widen
    /// it on their own rule until the floor — from the residency the decodes'
    /// stores left, not from the reset figure.
    #[test]
    fn decodes_under_the_limit_leave_room_for_prefills() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        for _ in 0..4 {
            assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        }
        assert!(!p.is_full());
        let after_stores = RESIDENT_RUN15 - 4 * DECODE_DISLODGE;
        assert_eq!(p.resident_now(), after_stores);
        let (admitted, refusal) = fill_with(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE);
        // 2,379 − 704 = 1,675 MiB of room: six exemplars, not nine.
        assert_eq!(admitted, 6);
        assert!(matches!(refusal, Refusal::Floor { .. }));
        assert_eq!(p.decodes(), 4);
        assert_eq!(p.tokens(), 1_500);
    }

    /// Prefills first, to the floor: the floor closes the wave to the decodes
    /// behind them — the caller offered the prefills first, so the wave is
    /// theirs.
    #[test]
    fn a_prefill_floor_closes_the_wave_to_later_decodes() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let (_, refusal) = fill_with(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE);
        assert!(matches!(refusal, Refusal::Floor { .. }));
        assert!(matches!(
            offer(&mut p, decode(0), 0),
            Admit::Refused(Refusal::Full)
        ));
        assert_eq!(p.decodes(), 0);
    }

    /// Full is full for everything, including offers that dislodge nothing.
    #[test]
    fn full_refuses_every_kind_until_reset() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, 100, DECODE_CAP);
        assert!(offer(&mut p, prefill(100), 100 * MIB).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(1), 0),
            Admit::Refused(Refusal::Cap { .. })
        ));
        assert!(matches!(
            offer(&mut p, prefill(0), 0),
            Admit::Refused(Refusal::Full)
        ));
        assert!(matches!(
            offer(&mut p, decode(0), 0),
            Admit::Refused(Refusal::Full)
        ));
        p.reset(RESIDENT_RUN15, FLOOR_16GB, 100, DECODE_CAP);
        assert!(offer(&mut p, decode(0), DECODE_DISLODGE).is_admitted());
    }

    /// Interleaved offers in the caller's priority order behave as one budget:
    /// each is judged on its own model against the wave so far, and every
    /// admission's weights carry into the next — the prefills' tiers count
    /// against the decodes' floor and the decodes' stores against the
    /// prefills'.
    #[test]
    fn interleaved_offers_share_one_budget() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        for _ in 0..3 {
            assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
            assert!(offer(&mut p, prefill(300), 300 * MIB).is_admitted());
        }
        assert_eq!((p.decodes(), p.tokens()), (3, 900));
        assert_eq!(
            p.resident_now(),
            RESIDENT_RUN15 - 3 * DECODE_DISLODGE - 900 * MIB
        );
        // 951 MiB above the floor: five more stores fit, the sixth does not,
        // and the copy is still under the layer the whole way.
        for _ in 0..5 {
            assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        }
        assert!(p.decode_copy_secs(p.routed_per_layer(), p.resident_now()) < p.layer_secs());
        assert!(matches!(
            offer(&mut p, decode(2), DECODE_DISLODGE),
            Admit::Refused(Refusal::Floor { .. })
        ));
        assert!(p.is_full());
    }

    /// The caller's `weights_before` is what the current rate is read at, so a
    /// caller whose residency moved between offers (a purchase, a give-back) is
    /// judged against the truth, not against the module's last figure: the
    /// same tokens to the same `weights_after` are admitted from one baseline
    /// and refused from another.
    #[test]
    fn the_callers_weights_before_is_the_baseline() {
        let resident = RESIDENT_RUN15 - 1_000 * MIB;
        // The weight side gave a GiB back since the last offer, and this
        // 32-token admission takes it again: 70 ms of copy for 13 ms of
        // compute, worse than the wave stood.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 1_000 * MIB).is_admitted());
        assert_eq!(p.resident_now(), resident);
        match p.try_admit(prefill(32), resident + GIB, resident) {
            Admit::Refused(Refusal::Worse { before, after }) => {
                assert!(approx(before, p.rate(1_000, resident + GIB), 1e-9));
                assert!(after < before);
            }
            other => panic!("{other:?}"),
        }
        // Nothing moved since: the same 32 tokens to the same weights dislodge
        // nothing, and are admitted.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 1_000 * MIB).is_admitted());
        assert!(p.try_admit(prefill(32), resident, resident).is_admitted());
    }

    // ── other cards ────────────────────────────────────────────────────────

    /// The 3090: half the link, 12 GiB resident of 20.3 GB. The same exemplar
    /// costs longer and widening buys ~4×.
    #[test]
    fn the_3090_pays_the_slower_bus_and_gains_from_width() {
        let p = planner(LINK_3090); // seeded at 9.6 GB/s
        let r = 12 * GIB;
        let one = p.rate(250, r);
        let eight = p.rate(2_000, r - 1_750 * MIB);
        assert!(p.forward_secs(250, r) > 0.7 && p.forward_secs(250, r) < 1.1);
        assert!(
            eight / one > 3.5 && eight / one < 4.5,
            "ratio {}",
            eight / one
        );
    }

    /// Fully resident (the 72 GB Blackwell holds every expert) and nothing
    /// dislodged: there is no copy to amortise, the forward is pure compute,
    /// the rate is at the ceiling from the first row, and only the cap ends the
    /// prefill wave.
    #[test]
    fn a_fully_resident_card_is_compute_bound_and_only_the_cap_stops_it() {
        let mut p = planner(LINK_BLACKWELL).with_min_gain(0.0);
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        assert_eq!(p.non_resident_bytes(total), 0);
        assert!(approx(p.forward_secs(1_000, total), 0.42, 1e-9));
        p.reset(total, 8 * GIB, CAP, DECODE_CAP);
        let (_, refusal) = fill_with(&mut p, prefill(1_024), 0);
        assert!(
            matches!(refusal, Refusal::Cap { max_tokens: CAP }),
            "{refusal:?}"
        );
        assert_eq!(p.tokens(), CAP);
    }

    /// A small card whose whole zone is under the floor the caller asks for:
    /// the head is taken, nothing else.
    #[test]
    fn a_card_too_small_for_the_floor_takes_only_its_head() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(2 * GIB, 3 * GIB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(64), 64 * MIB).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(1), MIB),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }

    // ── learning: the copy rate ────────────────────────────────────────────

    /// From an 80% seed of a 25 GB/s link, forwards at run 15's rate bring the
    /// estimate to within 5% in a couple of dozen observations, and each step
    /// moves it by exactly `alpha` of the remaining error.
    #[test]
    fn the_copy_rate_converges_from_the_seed_to_the_observed_rate() {
        let mut p = planner(LINK_4090_MOBILE);
        assert!(
            approx(p.effective_bytes_per_s(), 20e9, 1e-9),
            "seeded at 80% of the link"
        );
        let bytes = p.non_resident_bytes(RESIDENT_RUN15) as f64;
        let secs = bytes / BW_RUN15 + 474.0 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        let before = p.effective_bytes_per_s();
        let observed = p
            .observe_prefill(474, RESIDENT_RUN15, secs)
            .expect("an observation");
        assert!(approx(observed, BW_RUN15, 1e-6));
        let expected = before + WaveRate::DEFAULT_ALPHA * (BW_RUN15 - before);
        assert!(
            approx(p.effective_bytes_per_s(), expected, 1e-9),
            "one dampened step"
        );
        for _ in 0..24 {
            p.observe_prefill(474, RESIDENT_RUN15, secs);
        }
        assert!(approx(p.effective_bytes_per_s(), BW_RUN15, 0.05));
        assert_eq!(p.samples(), 25);
    }

    /// A single outlier moves the estimate by at most `alpha` of its distance:
    /// a stalled forward cannot swing the next wave's plan.
    #[test]
    fn one_outlier_is_dampened() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15) as f64;
        let stalled = 10.0 * bytes / BW_RUN15;
        let before = p.effective_bytes_per_s();
        let observed = p
            .observe_prefill(474, RESIDENT_RUN15, stalled)
            .expect("an observation");
        assert!(observed < before / 5.0);
        let after = p.effective_bytes_per_s();
        assert!(after < before);
        assert!(
            approx(
                before - after,
                WaveRate::DEFAULT_ALPHA * (before - observed),
                1e-9
            ),
            "moved by alpha of the way, no more"
        );
    }

    /// **A big copy that beats the link corrects the link.** The probe reads a
    /// contended figure at startup; a forward that moved 12 GB at 15 GB/s is
    /// better evidence about the same bus than one 256 MiB transfer taken while
    /// the expert cache was staging over it.
    #[test]
    fn a_large_forward_faster_than_the_probe_corrects_the_probe() {
        // A probe that under-read the 4090's link by half.
        let mut p = planner(12.03e9).with_alpha(1.0);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15);
        assert!(
            bytes as usize > WaveRate::PROBE_BYTES,
            "a copy worth believing"
        );
        let rows = p.min_learn_rows();
        let compute = rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        let observed = p
            .observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 15.0e9 + compute)
            .expect("an observation");
        assert!(approx(observed, 15.0e9, 1e-9), "not clamped to 12.03");
        assert!(
            approx(p.link_bytes_per_s(), 15.0e9, 1e-9),
            "the link moved up"
        );
        assert!(approx(p.effective_bytes_per_s(), 15.0e9, 1e-9));
        // And it never moves back down on a slower forward.
        p.observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 5.0e9 + compute);
        assert!(
            approx(p.link_bytes_per_s(), 15.0e9, 1e-9),
            "the link is a high-water mark"
        );
    }

    /// **One fast forward does not become the new ceiling.** The link can only
    /// ratchet upward — nothing below it moves it back — so a lone outlier that
    /// latched would raise the ceiling permanently and let the copy rate follow
    /// it. It moves by `alpha`, so evidence has to be sustained.
    #[test]
    fn a_single_fast_outlier_barely_moves_the_link() {
        let mut p = planner(12.0e9);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15);
        let rows = p.min_learn_rows();
        let compute = rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        // A forward implying 60 GB/s — five times the measured link.
        p.observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 60.0e9 + compute);
        let after_one = p.link_bytes_per_s();
        let expected = 12.0e9 + WaveRate::DEFAULT_ALPHA * (60.0e9 - 12.0e9);
        assert!(approx(after_one, expected, 1e-9), "{after_one}");
        assert!(after_one < 20.0e9, "one outlier stays near the measurement");
        // Sustained, it does get there.
        for _ in 0..60 {
            p.observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 60.0e9 + compute);
        }
        assert!(
            approx(p.link_bytes_per_s(), 60.0e9, 0.02),
            "sustained evidence wins"
        );
    }

    /// A *small* copy that reads impossibly fast is still clamped: with few
    /// bytes to divide, the rate is mostly the error in the compute model, and
    /// that is not evidence about the bus.
    #[test]
    fn a_small_observation_faster_than_the_link_is_clamped_to_it() {
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        // 64 MiB short of full residency: a quarter of the probe's transfer.
        let nearly = total - 64 * MIB;
        let bytes = p.non_resident_bytes(nearly);
        assert!((bytes as usize) < WaveRate::PROBE_BYTES);
        // Wide enough to have routed across the layer, and long enough to have
        // paid its own compute — otherwise the copy reads as negative.
        let rows = p.min_learn_rows();
        let observed = p
            .observe_prefill(
                rows,
                nearly,
                bytes as f64 / 200e9 + rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN,
            )
            .expect("an observation");
        assert!(approx(observed, LINK_4090_MOBILE, 1e-9));
        assert!(
            approx(p.link_bytes_per_s(), LINK_4090_MOBILE, 1e-9),
            "the link held"
        );
        assert!(approx(p.effective_bytes_per_s(), LINK_4090_MOBILE, 1e-9));
    }

    /// Forwards that cannot teach the copy rate leave the estimate alone: a
    /// fully resident card, a forward shorter than its own compute, a
    /// non-finite or zero duration, and a duration so long the implied rate is
    /// a stall rather than a bus.
    #[test]
    fn prefill_forwards_that_teach_nothing_are_ignored() {
        let mut p = planner(LINK_4090_MOBILE);
        let bw = p.effective_bytes_per_s();
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        // **A forward too narrow to route across the layer.** The 26-row
        // section forwards a run opens with read as 119 GB/s on a 25 GB/s link
        // before this was guarded — the model divides by the whole expert set
        // and a narrow wave touches a fraction of it.
        assert_eq!(p.min_learn_rows(), 126, "8 of 256, to a 2% untouched share");
        assert_eq!(
            p.observe_prefill(26, RESIDENT_RUN15, 0.05),
            None,
            "too narrow to have needed every expert"
        );
        assert_eq!(
            p.observe_prefill(125, RESIDENT_RUN15, 0.9),
            None,
            "one row short"
        );
        assert_eq!(
            p.observe_prefill(474, total, 0.5),
            None,
            "fully resident: nothing copied"
        );
        assert_eq!(
            p.observe_prefill(1_000, RESIDENT_RUN15, 0.3),
            None,
            "shorter than its compute"
        );
        assert_eq!(p.observe_prefill(474, RESIDENT_RUN15, f64::NAN), None);
        assert_eq!(p.observe_prefill(474, RESIDENT_RUN15, 0.0), None);
        assert_eq!(p.observe_prefill(474, RESIDENT_RUN15, -1.0), None);
        assert_eq!(
            p.observe_prefill(474, RESIDENT_RUN15, 3_600.0),
            None,
            "a stall, not a rate"
        );
        assert_eq!(p.effective_bytes_per_s(), bw);
        assert_eq!(p.samples(), 0);
    }

    // ── learning: the layer time ───────────────────────────────────────────

    /// A compute-bound decode wave — its per-layer time above the modelled
    /// copy — teaches that per-layer time as the layer time, dampened by
    /// `alpha`, and converges on it.
    #[test]
    fn the_layer_time_is_learned_from_compute_bound_decode_forwards() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let seed = p.layer_secs();
        // Eight drafted decodes at run 15's residency: 17.6 ms of copy a layer,
        // under a true 20 ms layer.
        let true_layer = 20e-3;
        assert!(p.decode_copy_secs(192, RESIDENT_RUN15) < true_layer);
        let observed = p
            .observe_decode(8, 2, RESIDENT_RUN15, true_layer * LAYERS)
            .expect("an observation");
        assert!(approx(observed, true_layer, 1e-9));
        let expected = seed + WaveRate::DEFAULT_ALPHA * (true_layer - seed);
        assert!(approx(p.layer_secs(), expected, 1e-9));
        for _ in 0..30 {
            p.observe_decode(8, 2, RESIDENT_RUN15, true_layer * LAYERS);
        }
        assert!(approx(p.layer_secs(), true_layer, 0.01));
        assert_eq!(p.decode_samples(), 31);
    }

    /// Run 12's decode waves: seven drafted decodes stepping in ~760 ms at run
    /// 15's residency. The model puts their copy at 15.4 ms a layer against
    /// 18.5 ms a layer of step — compute-bound, so the layer time is 18.5 ms —
    /// and says eight such decodes would have hidden under it.
    #[test]
    fn the_measured_decode_waves_teach_an_eighteen_millisecond_layer() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_alpha(1.0);
        assert!(approx(
            p.decode_copy_secs(168, RESIDENT_RUN15),
            15.38e-3,
            1e-3
        ));
        let observed = p
            .observe_decode(7, 2, RESIDENT_RUN15, 0.76)
            .expect("an observation");
        assert!(approx(observed, 0.76 / LAYERS, 1e-9));
        assert!(approx(p.layer_secs(), 18.54e-3, 1e-3));
        assert_eq!(p.decode_bus_limit(2, RESIDENT_RUN15), 8);
    }

    /// A bus-bound decode wave — its per-layer time at or under the modelled
    /// copy — says nothing about the layer, and leaves the estimate alone.
    #[test]
    fn a_bus_bound_decode_forward_teaches_nothing_about_the_layer() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        // Twelve drafted decodes: 26.4 ms of copy a layer; a 400 ms step is
        // 9.8 ms a layer, under it.
        assert!(p.decode_copy_secs(288, RESIDENT_RUN15) > 0.4 / LAYERS);
        assert_eq!(p.observe_decode(12, 2, RESIDENT_RUN15, 0.4), None);
        assert!(approx(p.layer_secs(), 20e-3, 1e-9));
        // Exactly the copy is still bus-bound.
        let copy_step = p.decode_copy_secs(288, RESIDENT_RUN15) * LAYERS;
        assert_eq!(p.observe_decode(12, 2, RESIDENT_RUN15, copy_step), None);
        assert_eq!(p.decode_samples(), 1, "only the fixture's own observation");
    }

    /// Residency is part of the observation: the same seven decodes in the
    /// same 760 ms read as compute-bound at run 15's residency and as bus-bound
    /// with a third of the zone gone, where the modelled copy is longer than
    /// the step.
    #[test]
    fn the_observed_residency_decides_whether_the_forward_teaches() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_alpha(1.0);
        let starved = 2 * GIB;
        assert!(p.decode_copy_secs(168, starved) > 0.76 / LAYERS);
        assert_eq!(p.observe_decode(7, 2, starved, 0.76), None);
        assert!(p.observe_decode(7, 2, RESIDENT_RUN15, 0.76).is_some());
    }

    /// Decode forwards that cannot teach are ignored: no decodes, a non-finite
    /// or zero duration.
    #[test]
    fn decode_forwards_that_teach_nothing_are_ignored() {
        let mut p = planner(LINK_4090_MOBILE);
        let layer = p.layer_secs();
        assert_eq!(p.observe_decode(0, 2, RESIDENT_RUN15, 0.5), None);
        assert_eq!(p.observe_decode(8, 2, RESIDENT_RUN15, f64::INFINITY), None);
        assert_eq!(p.observe_decode(8, 2, RESIDENT_RUN15, 0.0), None);
        assert_eq!(p.observe_decode(8, 2, RESIDENT_RUN15, -0.5), None);
        assert_eq!(p.layer_secs(), layer);
        assert_eq!(p.decode_samples(), 0);
    }

    /// A learned bus changes the decode limit through the same copy figure the
    /// prefill side learned: the layer time is untouched, the copy is not.
    #[test]
    fn a_faster_learned_bus_hides_more_decodes_under_the_same_layer() {
        let slow = planner_with_layer(LINK_4090_MOBILE, 10e9, 16e-3);
        let fast = planner_with_layer(LINK_4090_MOBILE, 20e9, 16e-3);
        assert_eq!(slow.layer_secs(), fast.layer_secs());
        // 3.36 ms a drafted decode at 10 GB/s, 1.68 ms at 20: four hide in
        // 16 ms against nine.
        assert_eq!(slow.decode_bus_limit(2, RESIDENT_RUN15), 4);
        assert_eq!(fast.decode_bus_limit(2, RESIDENT_RUN15), 9);
    }

    // ── constructor guards and the probe ───────────────────────────────────

    #[test]
    fn the_seed_is_eighty_percent_of_the_link() {
        let p = planner(LINK_3090);
        assert_eq!(p.link_bytes_per_s(), LINK_3090);
        assert!(approx(p.link_fraction(), WaveRate::SEED_FRACTION, 1e-12));
        assert!(approx(p.effective_bytes_per_s(), 9.6e9, 1e-9));
    }

    #[test]
    #[should_panic(expected = "positive finite")]
    fn a_zero_link_rate_is_refused() {
        let _ = planner(0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be in")]
    fn an_alpha_above_one_is_refused() {
        let _ = planner(LINK_4090_MOBILE).with_alpha(1.5);
    }

    #[test]
    #[should_panic(expected = "decode step rate")]
    fn a_zero_decode_seed_rate_is_refused() {
        let _ = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                initial_tok_per_s: 0.0,
                ..DecodeModel::default()
            },
        );
    }

    /// A coefficient above 1 is ordinary — a cache holding a third of the model
    /// hits far more than a third of the time — and the hit it produces is
    /// capped at certainty rather than the coefficient being capped at 1.
    #[test]
    fn a_hit_coefficient_above_one_is_ordinary_and_the_hit_still_caps() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let p = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                hit_rate: 1.73,
                ..DecodeModel::default()
            },
        );
        // Run 15's residency: 39.3% resident × 1.73 is a 0.68 hit.
        assert!(approx(p.hit_fraction(RESIDENT_RUN15), 0.6804, 1e-3));
        assert!(approx(p.streamed_fraction(RESIDENT_RUN15), 0.3196, 1e-3));
        // Past 1/1.73 — 58% of the model — the coefficient would exceed
        // certainty. It does not; the hit stops at 1 and the copy at 0.
        assert!(approx(p.hit_fraction(total), 1.0, 1e-12));
        assert!(approx(p.streamed_fraction(total), 0.0, 1e-12));
        assert!(
            p.streamed_fraction(total / 2) > 0.0,
            "and stays a real fraction below the saturation point",
        );
    }

    #[test]
    #[should_panic(expected = "hit coefficient")]
    fn a_negative_hit_coefficient_is_refused() {
        let _ = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                hit_rate: -0.1,
                ..DecodeModel::default()
            },
        );
    }

    /// The learning width is the model's own premise, in rows: the point where
    /// a wave's routing has covered all but 2% of a layer, so that dividing the
    /// forward's copy time by *every* expert's bytes is honest.
    #[test]
    fn the_learning_width_follows_the_routing_geometry() {
        let p = planner(LINK_4090_MOBILE);
        assert_eq!(p.min_learn_rows(), 126, "8 of 256 → 32 · ln 50");
        // A wider top-k covers the layer sooner; a bigger layer takes longer.
        let wide = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                experts_per_token: 32,
                ..DecodeModel::default()
            },
        );
        assert_eq!(
            wide.min_learn_rows(),
            32,
            "four times the top-k, a quarter the rows"
        );
        // And a forward at that width does teach, where one row short does not.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_alpha(1.0);
        let rows = p.min_learn_rows();
        let secs = p.non_resident_bytes(RESIDENT_RUN15) as f64 / 15.0e9
            + rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        assert_eq!(p.observe_prefill(rows - 1, RESIDENT_RUN15, secs), None);
        assert!(p.observe_prefill(rows, RESIDENT_RUN15, secs).is_some());
    }

    /// **The coefficient is learned from the cache's own counters.** Run 15
    /// reported a 0.65 hit rate at 7,611 MiB resident of a 20.29 GB model —
    /// 39.3% — so the coefficient is 1.65, and the seed of 0.7 was pricing every
    /// decode's copy at twice its real cost.
    #[test]
    fn the_hit_coefficient_is_learned_from_the_caches_counters() {
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        assert!(
            approx(p.hit_rate(), 0.7, 1e-12),
            "seeded at the measured 0.7"
        );
        let observed = p
            .observe_hit_rate(0.65, RESIDENT_RUN15)
            .expect("a routing interval");
        assert!(approx(observed, 1.6527, 1e-3), "{observed}");
        assert!(approx(p.hit_rate(), observed, 1e-12));
        assert_eq!(p.hit_samples(), 1);
        // And the copy it prices falls by the same factor it was over by.
        let streamed = p.streamed_fraction(RESIDENT_RUN15);
        assert!(approx(streamed, 0.35, 1e-2), "{streamed}");
        let seeded = planner(LINK_4090_MOBILE).streamed_fraction(RESIDENT_RUN15);
        assert!(
            approx(seeded / streamed, 2.07, 0.02),
            "the copy was over by 2x"
        );
    }

    /// Intervals that cannot say anything about the coefficient leave it alone:
    /// a card holding no experts, and a rate that is not a fraction.
    #[test]
    fn hit_intervals_that_teach_nothing_are_ignored() {
        let mut p = planner(LINK_4090_MOBILE);
        let seed = p.hit_rate();
        assert_eq!(p.observe_hit_rate(0.5, 0), None, "nothing resident");
        assert_eq!(p.observe_hit_rate(1.5, RESIDENT_RUN15), None);
        assert_eq!(p.observe_hit_rate(-0.1, RESIDENT_RUN15), None);
        assert_eq!(p.observe_hit_rate(f64::NAN, RESIDENT_RUN15), None);
        assert_eq!(p.hit_rate(), seed);
        assert_eq!(p.hit_samples(), 0);
    }

    /// **Residency still multiplies, which is the point of the refinement.**
    /// Whatever the coefficient has learned, dislodging weights raises the
    /// streamed share of every decode already in the wave.
    #[test]
    fn a_learned_coefficient_still_answers_to_weight_pressure() {
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        p.observe_hit_rate(0.65, RESIDENT_RUN15);
        let before = p.streamed_fraction(RESIDENT_RUN15);
        let after = p.streamed_fraction(RESIDENT_RUN15 - 2 * GIB);
        assert!(
            after > before,
            "less resident, more streamed: {before} → {after}"
        );
        assert!(p.streamed_fraction(0) > p.streamed_fraction(RESIDENT_RUN15));
    }

    // ── work the caller has already paid for ───────────────────────────────

    /// A charge is not a decision: it is taken whatever it costs — past the
    /// cap, under the floor, past the bus — and it never latches the wave full.
    #[test]
    fn a_charge_is_never_refused_and_never_closes_the_wave() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(FLOOR_16GB, FLOOR_16GB, 100, 2);
        // Over the cap and at the floor.
        p.charge(prefill(5_000), FLOOR_16GB - GIB);
        // Past the decode cap, and past the bus at that residency.
        for _ in 0..8 {
            p.charge(decode(2), FLOOR_16GB - GIB);
        }
        assert_eq!(p.tokens(), 5_000);
        assert_eq!(p.decodes(), 8);
        assert_eq!(p.routed_per_layer(), 24 * 8);
        assert_eq!(p.resident_now(), FLOOR_16GB - GIB);
        assert!(!p.is_full(), "a charge is not a refusal");
    }

    /// **A charged wave is not an empty one.** The head waiver exists so a wave
    /// carries something; a wave already carrying a held creep group or a
    /// stepping decode carries something, so the next offer is judged.
    #[test]
    fn a_charged_wave_judges_its_next_offer_instead_of_taking_it_as_the_head() {
        // Uncharged: the head is taken even though it crosses the floor.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(FLOOR_16GB + 100 * MIB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(250), GIB).is_admitted());
        assert!(p.resident_now() < FLOOR_16GB);

        // Charged with a held creep first: the same offer is judged, and the
        // floor refuses it.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(FLOOR_16GB + 100 * MIB, FLOOR_16GB, CAP, DECODE_CAP);
        p.charge(prefill(300), FLOOR_16GB + 100 * MIB);
        assert!(matches!(
            offer(&mut p, prefill(250), GIB),
            Admit::Refused(Refusal::Floor { .. })
        ));
        assert_eq!(p.tokens(), 300, "the charged rows stand, the offer did not");
    }

    /// A charged decode counts against the offers that follow it: the wave's
    /// decode rate, its routed set and its residency all move, so the next
    /// decode is judged against the wave as it will actually run.
    #[test]
    fn charged_decodes_are_the_baseline_the_next_offer_is_judged_against() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let mut resident = RESIDENT_RUN15;
        for n in 1..=9 {
            resident -= DECODE_DISLODGE;
            let projected = p.charge(decode(2), resident);
            assert!(approx(projected, p.decode_rate(n, 24 * n, resident), 1e-9));
        }
        assert_eq!(p.decodes(), 9);
        // The tenth is the one `decodes_stop_when_another_would_lower_the_
        // decode_rate` refuses — and it is refused here too, because the nine
        // ahead of it were charged rather than ignored.
        assert!(matches!(
            offer(&mut p, decode(2), DECODE_DISLODGE),
            Admit::Refused(Refusal::Worse { .. })
        ));
    }

    /// A charge into an empty wave still leaves the projection readable, and a
    /// charge of nothing is a no-op on the counts.
    #[test]
    fn charging_reports_the_waves_projected_rate() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, CAP, DECODE_CAP);
        let projected = p.charge(prefill(474), RESIDENT_RUN15);
        assert!(approx(projected, 472.0, 0.02), "{projected}");
        assert_eq!(p.charge(prefill(0), RESIDENT_RUN15), projected);
        assert_eq!(p.tokens(), 474);
    }

    /// The link probe needs a device; on the CPU it says so rather than
    /// inventing a number.
    #[test]
    fn the_probe_refuses_a_cpu_device() {
        let err = WaveRate::measure(
            &Device::Cpu,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        )
        .expect_err("no link to measure on the CPU");
        assert!(err.to_string().contains("CUDA device"));
    }
}
