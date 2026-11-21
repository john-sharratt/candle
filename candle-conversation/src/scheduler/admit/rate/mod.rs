//! The wave throughput model: how many prefill rows and decodes one wave should
//! carry, priced in the expert bytes that cross the bus.
//!
//! Every offer is judged on whether the wave's rate rises if it joins. A prefill
//! forward needs every expert, so its copy is paid once however many rows ride it,
//! and the rows amortise that copy. A decode's copy is paid per layer per step and
//! grows as residency falls, so a dislodge raises the streamed bytes of every decode
//! already in the wave. The caller computes each offer's residency cost, and the
//! model learns the copy rate, the layer time and the hit exponent from the engine's
//! own forwards, each dampened so that one outlier does not steer the next wave.
//!
//! - `wave`: the budget, the first-offer rule and the latching refusals.
//! - `offer`: widening, joining and outranking judgements.
//! - `trading`: the share-weighted trade between prefill gain and decode loss.
//! - `judge`: the marginal decisions for a prefill and a promotion.
//! - `cost`: projected prefill forward time and rate.
//! - `cost_fit`: the least-squares fit of the prefill costs.
//! - `learn`: learning the copy rate, layer time and hit exponent.
//! - `decode`: decode step time, rate and bus limit.
//! - `decode_fit`: the per-layer fit of a decode step.
//! - `placement`: prices a turn placed beside running decodes.
//! - `construct`: building the planner from a measured link.
//! - `models`: expert geometry and the model's measured constants.
//! - `persist`: saving and restoring learned figures.
//! - `tests_admission`: the prefill admission rules on measured cards.
//! - `tests_decode`: the decode model against measured decode waves.
//! - `tests_fit`: the cost fits and restored models.
//! - `tests_joins`: joins, promotions and outranking offers.
//! - `tests_learning`: convergence and dampening of the learned estimates.
//! - `tests_support`: shared fixtures and constructors for the tests.
//! - `tests_trading`: the prefill and decode trade and held cohorts.

use candle::cuda_backend::link_probe;
use candle::Device;

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
    #[cfg(test)]
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
    /// A decode layer's time against the experts it fetched, learned from
    /// decode-only steps at the widths they ran ([`DecodeFit`]); what prices a
    /// step once the widths have varied, and `layer_secs` before.
    decode_fit: DecodeFit,
    /// When the decode fit last learned, for its forgetting in time.
    decode_fit_at: Option<std::time::Instant>,
    /// The positions a decode row attends, on average over the recent steps —
    /// the depth a decode the rates price is put at.
    decode_context: f64,
    /// Dampening: the fraction of each observation an estimate moves by.
    alpha: f64,
    /// Relative rate gain below which an admission is refused as saturated.
    min_gain: f64,
    /// The trades judged since the summary last read them — see
    /// [`TradeWindow`].
    trades: TradeWindow,
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
    /// The context a learned-from row attended, on average over recent forwards:
    /// what a caller that names no depth is priced at ([`Self::forward_secs`]),
    /// which is what `c` held before attention was priced apart from it.
    context_per_row: f64,
    // ── the wave being composed ────────────────────────────────────────────
    /// Resident weights as the wave stands: the reset figure, then each
    /// admission's `weights_after`.
    resident_now: u64,
    /// The residency a **prefill** admission may not take the weight side under.
    ///
    /// **Lower than [`Self::decode_floor`] on purpose, because the two kinds pay
    /// for weights on opposite schedules.** A prefill copies every expert once per
    /// forward whatever its width, so the copy amortises over rows and a byte it
    /// dislodges is paid once; the rows it buys are the whole gain. A decode's copy
    /// is paid per layer per step and scales with `1 - hit(resident)`, so residency
    /// the prefill spent is multiplied across every step of every decode that
    /// follows. One floor for both had to be set for the worse case, and that is
    /// what starved prefill width: `interleave::HOLD`'s own note records a run
    /// spending 13.5 minutes refusing prefills — 384 of them — for want of one
    /// recurrent store's ground, however wide the rows behind it.
    ///
    /// Driving residency down does not deadlock decode: the wave's head is carried
    /// whatever it costs, so a decode always runs as the head of a wave and the
    /// expert cache reloads under it.
    prefill_floor: u64,
    /// The residency a **decode** admission may not take the weight side under —
    /// the residency the engine defends so decode does not stream every layer.
    decode_floor: u64,
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

mod construct;
mod cost;
mod cost_fit;
mod decode;
mod decode_fit;
mod judge;
mod learn;
mod models;
mod offer;
mod persist;
mod placement;
#[cfg(test)]
mod tests_admission;
#[cfg(test)]
mod tests_decode;
#[cfg(test)]
mod tests_fit;
#[cfg(test)]
mod tests_joins;
#[cfg(test)]
mod tests_learning;
#[cfg(test)]
mod tests_support;
#[cfg(test)]
mod tests_trading;
mod trading;
mod wave;

use cost_fit::*;
pub use decode_fit::*;
pub use models::*;
pub use trading::*;
