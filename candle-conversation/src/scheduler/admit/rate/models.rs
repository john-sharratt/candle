//! Holds a MoE model's expert geometry and the measured constants of a prefill
//! forward that are not its copy.
//!
//! `ExpertGeometry` gives the bytes every expert occupies, which is what a prefill
//! forward must have on the card, copied or resident. `RateModel` keeps the costs
//! inside and outside a forward that neither width nor residency changes: the loop's
//! overhead per forward, which is measured from the loop's own clock because a
//! forward's duration cannot show it, and attention per attended pair, which stays
//! zero until forwards at different depths teach it. `DecodeModel` carries the
//! routing per decode and the learned hit exponent, seeded so the cache hits its
//! share of the model.

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
    ///
    /// The fixture every test in this module is derived through. Production
    /// reads the live geometry from the model rather than a recorded one — a
    /// constant here would be a second answer to a question the engine can ask.
    #[cfg(test)]
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
    /// Per-forward cost *inside* the forward that neither width nor residency changes.
    /// Zero by default and learned by [`CostFit`] from the engine's own forwards.
    pub fixed_secs: f64,
    /// Per-forward cost **outside** the forward: what the wave loop pays around it.
    ///
    /// **A forward's own duration is not what a forward costs the engine.** The loop
    /// drains submissions, promotes finished prefills, runs housekeeping and publishes
    /// telemetry between forwards, and none of it scales with rows — so it is amortised by
    /// width exactly as the expert copy is, and it is *larger* than the copy on a resident
    /// card where the copy is zero. Measured on the 30B-A3B: `loop:housekeeping` 42 ms,
    /// `drain:submit_turn` 27.6 ms, `loop:promote_finished` 12.3 ms per wave, against a
    /// decode kernel of a fraction of that.
    ///
    /// [`CostFit`] cannot learn it, because it only ever sees `forward_secs` — the
    /// interval the caller times around the forward itself. So the loop measures its own
    /// overhead and reports it through [`WaveRate::observe_overhead`]. Without this term
    /// the model prices width against the forward alone and stops widening far too early:
    /// it chose 4 rows where the engine wanted 20.
    pub overhead_secs: f64,
    /// Compute per token — see [`RateModel::SEED_COMPUTE_SECS_PER_TOKEN`].
    pub compute_secs_per_token: f64,
    /// Attention per attended pair: each row times the context it attends.
    ///
    /// **The per-token compute is flat in depth, and attention is not.** A
    /// row at 114K positions attends 114K keys where a fresh one attends a
    /// few, so pricing every row at one `c` priced a deep prompt as a shallow
    /// one: on BlackWorld (6ebf6205a, 2026-10-19) heavy-read's 8,192-row
    /// cohorts at ~114K context crept at about 100 tok/s while the model,
    /// fitted on the forwards it could learn from, advertised a cold rate many
    /// times that. Zero until [`CostFit`] sees forwards at different depths —
    /// until then the depth the fit saw is folded into `c`.
    pub attend_secs_per_pair: f64,
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
            // Zero until the loop reports its own cost. A seed would be a guess about
            // another machine's scheduler, and the loop can simply measure it.
            overhead_secs: 0.0,
            compute_secs_per_token: Self::SEED_COMPUTE_SECS_PER_TOKEN,
            attend_secs_per_pair: 0.0,
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
    /// found on the card at `resident_fraction ^ hit_exponent`.
    ///
    /// Below 1 the cache beats its share — the LRU zone holds the hot set and
    /// the Markov predictor prefetches ahead of the sweep, so a cache holding a
    /// third of the model hits far more than a third of the time; above 1
    /// residency churns faster than it is reused. **Learned**
    /// ([`WaveRate::observe_hit_rate`]) from the expert cache's own hit and
    /// miss counters, which is the only figure that knows what this workload's
    /// routing looks like; seeded at 1, the cache hitting its share.
    ///
    /// **A power, not a line through the origin.** The coefficient it replaces
    /// (`hit = c × fraction`, capped at 1) was learned at the residency the
    /// cache ran at and priced at another, and wherever the line crossed 1 it
    /// read every expert as found: on WhiteWorld (a94d106e0, 2026-10-20) a hit
    /// of 0.47 learned at its residency priced the residency a placed cohort
    /// leaves as all hits, so the step it published fell to 7.9 ms at every
    /// width against 100–500 ms stepped, and the gateway sent it 38 of 48
    /// turns; BlackWorld's read 1.000 against 82–99% measured. The power
    /// passes through the measured hit, reaches 1 only at the whole model, and
    /// still rises with residency, so a dislodge still raises the streamed
    /// bytes of every decode already in the wave.
    pub hit_exponent: f64,
    /// The step rate the layer time is seeded from: one step at this rate,
    /// spread over the MoE layers, is the time a layer has to hide its copies
    /// behind until the waves say otherwise.
    pub initial_tok_per_s: f64,
}

impl Default for DecodeModel {
    fn default() -> Self {
        Self {
            experts_per_token: 8,
            hit_exponent: 1.0,
            initial_tok_per_s: 30.0,
        }
    }
}
