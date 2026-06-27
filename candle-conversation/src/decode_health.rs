//! Decode-time health monitoring for the conversation engine.
//!
//! Detects common degradation patterns during token generation:
//!
//! | Check | Cost | Detects |
//! |---|---|---|
//! | Token repetition | CPU-only, every step | "…" collapse, single stuck token |
//! | Phrase loop | CPU-only, every step | Multi-token cyclic loops (local minima) |
//! | Logit NaN | GPU sync, per interval | NaN propagation from activation corruption |
//! | Logit Inf | GPU sync, per interval | Inf from F16/FP overflow |
//! | Logit magnitude | GPU sync, per interval | Pre-overflow saturation build-up |//! | Entropy collapse | GPU sync, per interval | Distribution collapse (\u2212\u03a3 p log p \u2193) |//!
//! All code in this module is compiled only when the `decode-health` feature is
//! enabled. The [`DecodeHealthConfig`](crate::config::DecodeHealthConfig) struct
//! is always compiled so callers can set it without feature-gating their own code.
//!
//! # Tracing target
//!
//! All warnings are emitted with:
//! ```text
//! target = "candle_conversation::decode_health"
//! ```
//! Add a tracing layer filtered to this target to capture events independently
//! from the general log stream.

use std::collections::VecDeque;

use candle::Tensor;
use candle_nn::ops::softmax;

// ── Diagnostic log sample ────────────────────────────────────────────────────

/// One sample captured at each logit check interval, stored in the full history log.
///
/// Contains every signal needed to reconstruct the health trajectory after an abort:
/// entropy and its inverse (p_max), magnitude stability (max_abs), and the decisive
/// winner margin (logit_gap).
#[derive(Debug, Clone)]
pub struct HealthSample {
    /// Decode step at which this sample was recorded.
    pub step: usize,
    /// Shannon entropy H = -sum(p * log(p)) in nats.
    pub entropy_nats: f32,
    /// Max absolute value across the pre-softmax logit vector.
    pub max_abs: f32,
    /// Peak probability mass: max(softmax(logits)).
    /// Complement of entropy — high p_max means the model is near-argmax.
    pub p_max: f32,
    /// Vocabulary index of the highest-probability token at this check step.
    pub argmax_token: u32,
    /// Pre-softmax logit gap: ln(p1) - ln(p2) = logit_1 - logit_2.
    /// The LSE normalisation cancels, so this equals the raw logit difference.
    /// Large values indicate a dominant winner with no close competitor.
    pub logit_gap: f32,
}

// ── Per-sequence state ────────────────────────────────────────────────────────

/// Per-sequence health tracking state.
///
/// Owned by `DecodeState` for the duration of one decode turn.
pub struct DecodeHealthState {
    /// Step counter — incremented once per `batch_decode_step` for this sequence.
    pub step: usize,
    /// Sliding window of recent token IDs (oldest-first), for repetition detection.
    pub recent_tokens: VecDeque<u32>,
    /// Rolling window of per-step entropy values (in nats), recorded every logit
    /// check interval. Used to detect sustained collapse before the hard floor.
    pub recent_entropies: VecDeque<f32>,
    /// Rolling window of per-step max|logit| values, recorded alongside entropies.
    ///
    /// F16 MLP overflow produces inflating max_abs approaching 65504.
    /// K/V cache corruption produces normal-scale max_abs despite entropy collapse.
    /// These two are mutually exclusive, making this the primary discriminator.
    pub recent_max_abs: VecDeque<f32>,
    /// Full diagnostic history log, appended by `check_entropy` on every call.
    ///
    /// Capped at `log_capacity`; the oldest entry is evicted when capacity is reached.
    /// Inspected by `render_health_dump` after an abort to produce the trajectory report.
    pub health_log: Vec<HealthSample>,
    /// Maximum number of entries to keep in `health_log`. `0` disables logging.
    log_capacity: usize,
    /// Dense-mode flag: set to `true` the first time entropy drops below the soft
    /// trend threshold during any interval check. Once set, every decode step
    /// triggers a full logit check so the final approach is captured at full
    /// resolution in the health log rather than just at the check interval.
    pub dense_mode: bool,
    /// Count of consecutive checks where entropy was below the hard-floor threshold
    /// **and the same argmax token was selected each time**. Reset to 0 whenever entropy
    /// recovers above the threshold or a different argmax token appears. The hard-floor
    /// abort fires only when this reaches `entropy_hard_min_consec`.
    pub hard_floor_consec: usize,
    /// The argmax token that established the current consecutive run. `None` when the
    /// counter is 0. Switching to a different low-entropy token resets the counter,
    /// preventing false positives from multi-token structural formatting sequences
    /// (e.g. `* ... *\n`) where three different tokens all happen to be low-entropy.
    pub hard_floor_consec_token: Option<u32>,
    /// Count of consecutive **interval-only** checks where entropy was below the
    /// deep interval-floor threshold, regardless of which token won. Catches multi-token
    /// cycling (e.g. `*` / `\n` / ` ` rotating) where no single token repeats but the
    /// distribution is near-deterministic on every check. Reset whenever an interval
    /// check sees entropy above the threshold. Dense-mode steps do not advance this counter.
    pub interval_floor_consec: usize,
    /// Rolling window of argmax token IDs recorded at each interval-only check.
    /// Used to detect when one structural token dominates the distribution across
    /// many consecutive interval checkpoints (content collapse without a tight
    /// consecutive repeat that the hard-floor or phrase-loop checks would catch).
    pub recent_interval_argmax: VecDeque<u32>,
    /// `true` while the model is generating inside a `<think>…</think>` block.
    ///
    /// Think-block content is intentionally near-deterministic (trained formulaic
    /// reasoning patterns), so entropy-based abort checks (`EntropyCollapse`) are
    /// suppressed in this state. All other triggers — `TokenRepetition`, `PhraseLoop`,
    /// `LogitNaN/Inf/Magnitude`, and `ArgmaxDominance` — still fire normally.
    ///
    /// Set by the decode loop when `<think>` (segment_open_token_id) is sampled;
    /// cleared when `</think>` (segment_close_token_id) is sampled.
    pub inside_think_block: bool,

    /// `true` when the sequence was created with a near-zero (≤ 0.01) sampling
    /// temperature (i.e. greedy / argmax decoding).
    ///
    /// At temperature≈0, softmax produces an exactly peaked distribution by design:
    /// one token has p≈1.0 and all others have p≈0.0, giving H≈0 nats on every
    /// single step.  The entropy-based abort checks (`EntropyCollapse`,
    /// `ArgmaxDominance`) would fire immediately and abort legitimate generation.
    ///
    /// When this flag is `true`, `check_entropy` is not called so those checks are
    /// completely skipped.  All other checks — `LogitNaN`, `LogitInf`,
    /// `LogitMagnitude`, `TokenRepetition`, `PhraseLoop` — remain active because
    /// they are not temperature-sensitive.
    ///
    /// Set once at sequence creation from `sampling_config.temperature`; never
    /// changes during the lifetime of the decode.
    pub skip_entropy_checks: bool,

    // ── Adaptive entropy baseline ────────────────────────────────────────────
    /// Entropy samples collected during the adaptive baseline warm-up window.
    /// Only populated while `entropy_baseline_mean` is `None`.
    /// Cleared after the baseline is fixed to release the allocation.
    entropy_baseline_samples: VecDeque<f32>,
    /// Mean entropy from the first `entropy_baseline_window` interval samples.
    /// `None` until the window fills.  Once set it is never updated — the
    /// session baseline is fixed at warm-up to avoid baseline drift during collapse.
    pub entropy_baseline_mean: Option<f32>,
    /// Copy of `DecodeHealthConfig::entropy_baseline_window`.
    pub entropy_baseline_window: usize,
    /// Copy of `DecodeHealthConfig::entropy_trend_relative_factor`.
    pub entropy_trend_relative_factor: f32,
    /// Copy of `DecodeHealthConfig::entropy_trend_absolute_min_nats`.
    pub entropy_trend_absolute_min_nats: f32,
    /// The most-recently computed effective trend threshold in nats.
    ///
    /// Updated by `check_entropy` every call.  `0.0` means the trend check was
    /// suppressed (warm-up period or adaptive mode disabled with threshold=0).
    /// Used by `render_health_dump` to annotate charts with the actual value that
    /// triggered (or would trigger) a sustained-collapse abort.
    pub entropy_effective_trend_threshold: f32,
    /// The most-recently computed effective interval-floor threshold in nats.
    ///
    /// Updated by `check_entropy` on every interval call.  `0.0` means the check
    /// was suppressed (warm-up) or disabled.  Works identically to
    /// `entropy_effective_trend_threshold` but for the interval-floor cycling check.
    pub entropy_effective_interval_floor_threshold: f32,
}

impl DecodeHealthState {
    /// Create a new health state.
    ///
    /// `window` — size of the repetition and trend sliding windows.
    /// `log_capacity` — max entries kept in the full diagnostic log. `0` disables.
    pub fn new(window: usize, log_capacity: usize) -> Self {
        Self {
            step: 0,
            recent_tokens: VecDeque::with_capacity(window + 1),
            recent_entropies: VecDeque::new(),
            recent_max_abs: VecDeque::new(),
            health_log: Vec::new(),
            log_capacity,
            dense_mode: false,
            hard_floor_consec: 0,
            hard_floor_consec_token: None,
            interval_floor_consec: 0,
            recent_interval_argmax: VecDeque::new(),
            inside_think_block: false,
            skip_entropy_checks: false,
            entropy_baseline_samples: VecDeque::new(),
            entropy_baseline_mean: None,
            entropy_baseline_window: 0,
            entropy_trend_relative_factor: 0.0,
            entropy_trend_absolute_min_nats: 0.04,
            entropy_effective_trend_threshold: 0.0,
            entropy_effective_interval_floor_threshold: 0.0,
        }
    }

    /// Apply adaptive-baseline config parameters from a `DecodeHealthConfig`.
    ///
    /// Call this immediately after `new()` when adaptive mode is desired.
    /// Splitting initialisation into two steps keeps `new()` lean for callers
    /// that don't use the health config (tests, benchmarks).
    pub fn apply_baseline_config(
        &mut self,
        baseline_window: usize,
        relative_factor: f32,
        absolute_min_nats: f32,
    ) {
        self.entropy_baseline_window = baseline_window;
        self.entropy_trend_relative_factor = relative_factor;
        self.entropy_trend_absolute_min_nats = absolute_min_nats;
    }

    /// Push a newly sampled token into the sliding window, capping at `window`.
    pub fn push_token(&mut self, token: u32, window: usize) {
        self.recent_tokens.push_back(token);
        while self.recent_tokens.len() > window {
            self.recent_tokens.pop_front();
        }
    }
}

// ── Events ───────────────────────────────────────────────────────────────────

/// A health degradation event detected during decode.
#[derive(Debug, Clone)]
pub enum HealthEvent {
    /// The `run_length` most-recent tokens were all identical.
    TokenRepetition {
        /// The repeated token ID.
        token: u32,
        /// Number of consecutive identical tokens that triggered the abort.
        run_length: usize,
    },
    /// A phrase of `period` tokens repeated `reps` times consecutively.
    ///
    /// This is the "local minimum" / cyclic-loop pattern: the model gets
    /// trapped cycling a multi-token phrase instead of progressing.
    PhraseLoop {
        /// Length of the repeating unit in tokens.
        period: usize,
        /// Number of consecutive full repetitions detected.
        reps: usize,
    },
    /// Logit tensor sum is NaN — indicates NaN propagation in the residual stream.
    LogitNaN { step: usize },
    /// Logit tensor sum is ±Inf — indicates Inf propagation or F16 overflow.
    ///
    /// `positive` is `true` for `+Inf`, `false` for `-Inf`.
    LogitInf { step: usize, positive: bool },
    /// Max absolute logit exceeded the configured threshold.
    LogitMagnitude {
        max_abs: f32,
        threshold: f32,
        step: usize,
    },
    /// The token distribution has collapsed: entropy H = \u2212\u03a3 p\u1d62 log p\u1d62 is critically low.
    ///
    /// A healthy distribution over a ~32k vocabulary has H \u2248 5\u201310 nats.
    /// Values below ~0.5 nats indicate the model is near-deterministically stuck.
    ///
    /// `sustained` is `true` when the full trend window was below the soft
    /// threshold; `false` when a single step hit the hard floor.
    ///
    /// `diagnosis` contains human-readable reasoning about the likely cause,
    /// derived from the entropy trend history at the time of collapse.
    EntropyCollapse {
        /// Measured entropy in nats (single-step or mean over the trend window).
        entropy_nats: f32,
        /// The threshold that was breached.
        threshold_nats: f32,
        /// True = sustained trend collapse; false = single-step hard floor.
        sustained: bool,
        step: usize,
        /// Diagnostic reasoning derived from the entropy trend at collapse time.
        diagnosis: String,
        /// Top-15 (token_id, probability, logit_rel) tuples sorted descending by
        /// probability, captured at the moment of collapse.
        /// `logit_rel` = ln(p_i) - ln(p_1): logit distance from the winner (0 for rank-1,
        /// increasingly negative for lower ranks). Allows inspecting the distribution.
        top_tokens: Vec<(u32, f32, f32)>,
    },
    /// A single argmax token dominates an excessive fraction of recent interval checks.
    ///
    /// Catches structural-token content collapse where the model generates the same
    /// style token (e.g. `…`, `*`, `—`) as the most-probable choice at most interval
    /// sample points, even if it doesn't repeat on every consecutive step.
    ArgmaxDominance {
        /// The token that dominated the recent interval checks.
        dominant_token: u32,
        /// How many of the last `window` interval checks it won.
        count: usize,
        /// The configured window size.
        window: usize,
        step: usize,
        /// Top-15 distribution at the moment of detection.
        top_tokens: Vec<(u32, f32, f32)>,
    },
}

impl std::fmt::Display for HealthEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthEvent::TokenRepetition { token, run_length } => write!(
                f,
                "token repetition: token_id={token} repeated {run_length}× consecutively"
            ),            HealthEvent::PhraseLoop { period, reps } => write!(
                f,
                "phrase loop: {period}-token phrase repeated {reps}\u{d7} consecutively (local minimum)"
            ),            HealthEvent::LogitNaN { step } => {
                write!(f, "NaN in logits at decode step {step}")
            }
            HealthEvent::LogitInf { step, positive } => {
                let sign = if *positive { "+Inf" } else { "-Inf" };
                write!(f, "{sign} in logits at decode step {step}")
            }
            HealthEvent::LogitMagnitude { max_abs, threshold, step } => write!(
                f,
                "logit magnitude {max_abs:.0} > threshold {threshold:.0} at step {step}"
            ),
            HealthEvent::EntropyCollapse { entropy_nats, threshold_nats, sustained, step, diagnosis, .. } => {
                if *sustained {
                    write!(
                        f,
                        "sustained distribution collapse: mean entropy {entropy_nats:.3} nats \
                         sustained below {threshold_nats:.3} nats (step {step}){diagnosis}"
                    )
                } else {
                    write!(
                        f,
                        "distribution collapse: entropy {entropy_nats:.3} nats \
                         < hard floor {threshold_nats:.3} nats at step {step}{diagnosis}"
                    )
                }
            }
            HealthEvent::ArgmaxDominance { dominant_token, count, window, step, .. } => write!(
                f,
                "argmax dominance: token {dominant_token} won {count}/{window} interval checks \
                 at step {step} — single structural token dominating distribution across \
                 {window} sample intervals indicates content collapse"
            ),
        }
    }
}

// ── Check functions ──────────────────────────────────────────────────────────

/// Check a logit tensor for NaN, Inf, and magnitude explosion.
///
/// Performs one or two GPU→CPU syncs (reduction kernels + transfers).
/// Call this only on the configured `logit_check_interval` to amortise cost.
///
/// Returns `Some(event)` on the first problem found, `None` if healthy.
pub fn check_logits(
    logits: &Tensor,
    magnitude_threshold: f32,
    step: usize,
) -> candle::Result<Option<HealthEvent>> {
    use candle::DType;

    // Cast once to F32, reuse for both the sum check and the magnitude check.
    // One dtype cast kernel + one reduce + one GPU→CPU transfer.
    // NaN propagates through summation; Inf is caught by is_infinite().
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let sum = logits_f32.flatten_all()?.sum_all()?.to_scalar::<f32>()?;

    if sum.is_nan() {
        return Ok(Some(HealthEvent::LogitNaN { step }));
    }
    if sum.is_infinite() {
        return Ok(Some(HealthEvent::LogitInf {
            step,
            positive: sum > 0.0,
        }));
    }

    // Only check magnitude if the sum was finite — avoids a second GPU sync
    // on already-known-bad logits.
    let max_abs = logits_f32
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    if max_abs > magnitude_threshold {
        return Ok(Some(HealthEvent::LogitMagnitude {
            max_abs,
            threshold: magnitude_threshold,
            step,
        }));
    }

    Ok(None)
}

/// Check the Shannon entropy H = \u2212\u03a3 p\u1d62 log p\u1d62 of the logit distribution.
///
/// Computes softmax(logits) then measures entropy in nats. Two fire conditions:
///
/// 1. **Hard floor** (`hard_threshold_nats > 0`): single step H < threshold. Fires
///    immediately \u2014 the model is near-argmax / maximally stuck.
///
/// 2. **Sustained trend** (`trend_window > 0`): the rolling window of the last
///    `trend_window` entropy samples are all below `trend_threshold_nats`. Fires
///    before the hard floor is reached, catching gradual collapse early.
///
/// Records the measured entropy into `state.recent_entropies` (capped at
/// `trend_window`) only when `is_interval` is `true` — i.e. on scheduled
/// interval or pre-boundary checks. Dense-mode steps set `is_interval = false`
/// so the trend window only advances at the original cadence and cannot be
/// flooded by consecutive structural tokens that happen to be low-entropy.
pub fn check_entropy(
    logits: &Tensor,
    state: &mut DecodeHealthState,
    hard_threshold_nats: f32,
    hard_min_consec: usize,
    trend_window: usize,
    trend_threshold_nats: f32,
    interval_floor_threshold_nats: f32,
    interval_floor_min_consec: usize,
    interval_argmax_dominance_window: usize,
    interval_argmax_dominance_fraction: f32,
    trend_recent_veto_window: usize,
    trend_recent_veto_factor: f32,
    structural_token_ids: &[u32],
    step: usize,
    is_interval: bool,
) -> candle::Result<Option<HealthEvent>> {
    use candle::D;

    // Compute p = softmax(logits). Cast to F32 first if needed.
    let logits_f32 = if logits.dtype() == candle::DType::F32 {
        logits.clone()
    } else {
        logits.to_dtype(candle::DType::F32)?
    };

    // Flatten to 1D: shape [vocab_size].
    let flat = logits_f32.flatten_all()?;

    // max|logit| — recorded every interval for the magnitude trend.
    // One GPU reduce + one scalar transfer; reuses the already-cast F32 tensor.
    let max_abs = flat.abs()?.max(0)?.to_scalar::<f32>()?;

    // p = softmax(logits): numerically stable LSE; finite inputs guaranteed by
    // check_logits which runs at the same interval before this function.
    let p = softmax(&flat, D::Minus1)?;

    // H = \u2212\u03a3 p\u1d62 log(p\u1d62).
    // All p\u1d62 > 0 after softmax on finite inputs, so log(p) is finite.
    // Transfer the full probability vector once — entropy, p_max, argmax, and logit_gap
    // are all computed from it in Rust, reducing GPU->CPU round-trips.
    // 32k x 4B = ~128 KB per check interval; acceptable for health diagnostics.
    let p_vec: Vec<f32> = p.to_vec1()?;

    // Single forward pass over p_vec: compute all four derived statistics.
    let mut entropy_nats = 0.0f32;
    let mut p_max = 0.0f32;
    let mut p_2nd = 0.0f32;
    let mut argmax_token = 0u32;
    for (i, &v) in p_vec.iter().enumerate() {
        if v > 0.0 {
            entropy_nats -= v * v.ln();
        }
        if v > p_max {
            p_2nd = p_max;
            p_max = v;
            argmax_token = i as u32;
        } else if v > p_2nd {
            p_2nd = v;
        }
    }
    // ln(p1) - ln(p2) = logit_1 - logit_2: the LSE normalisation cancels out.
    // Saturates at 100.0 for near-argmax distributions where p_2nd ~= 0.
    let logit_gap = if p_2nd > 1e-30 {
        p_max.ln() - p_2nd.ln()
    } else {
        100.0
    };

    // Record into trend windows and the full diagnostic log.
    // Trend windows only advance on scheduled interval/pre-boundary checks
    // (is_interval=true). Dense-mode steps are excluded so that consecutive
    // low-entropy structural tokens cannot fill the window and trigger a
    // false sustained-collapse abort.
    // Suppressed while inside a <think> block: think-block content is near-
    // deterministic by design and must not populate the entropy trend window
    // or activate dense-mode that would carry over into the response phase.
    if is_interval && trend_window > 0 && !state.inside_think_block {
        state.recent_max_abs.push_back(max_abs);
        while state.recent_max_abs.len() > trend_window {
            state.recent_max_abs.pop_front();
        }
        state.recent_entropies.push_back(entropy_nats);
        while state.recent_entropies.len() > trend_window {
            state.recent_entropies.pop_front();
        }
    }
    // ── Adaptive entropy baseline ──────────────────────────────────────────────
    // If adaptive mode is enabled (`entropy_baseline_window > 0` and
    // `entropy_trend_relative_factor > 0`), collect the first N interval samples
    // to establish the session's "healthy" entropy mean.  During this warm-up the
    // sustained-trend check is suppressed (other checks remain active).  Once the
    // baseline is fixed, the effective threshold is:
    //
    //   max(entropy_trend_absolute_min_nats, baseline_mean × relative_factor)
    //
    // capped at `trend_threshold_nats` so the adaptive value never exceeds the
    // static ceiling.  This automatically adjusts for sharp-distribution models
    // (e.g. MoE at mean ~0.3 nats → floor ≈ 0.075 nats) and soft models
    // (e.g. GPT-style at mean ~1.5 nats → floor ≈ 0.375 nats) without any
    // per-model configuration.
    let effective_trend_threshold_nats: f32 =
        if state.entropy_baseline_window > 0 && state.entropy_trend_relative_factor > 0.0 {
            // Advance baseline collection if not yet complete.
            if state.entropy_baseline_mean.is_none() && is_interval && !state.inside_think_block {
                state.entropy_baseline_samples.push_back(entropy_nats);
                if state.entropy_baseline_samples.len() >= state.entropy_baseline_window {
                    let mean = state.entropy_baseline_samples.iter().sum::<f32>()
                        / state.entropy_baseline_samples.len() as f32;
                    state.entropy_baseline_mean = Some(mean);
                    // Release the accumulator — it is no longer needed.
                    state.entropy_baseline_samples = VecDeque::new();
                    tracing::debug!(
                        target: "candle_conversation::decode_health",
                        step,
                        entropy_baseline_mean = mean,
                        "entropy baseline calibrated; adaptive trend threshold = {:.4} nats",
                        (mean * state.entropy_trend_relative_factor)
                            .max(state.entropy_trend_absolute_min_nats)
                            .min(trend_threshold_nats)
                    );
                }
            }
            match state.entropy_baseline_mean {
                Some(baseline) => {
                    if baseline < state.entropy_trend_absolute_min_nats {
                        // Model's natural entropy is at or below the absolute floor.
                        // Entropy was never high enough to collapse from — the trend
                        // check cannot distinguish normal operation from degradation.
                        // Disable it; TokenRepetition/PhraseLoop remain active.
                        0.0
                    } else {
                        // Adaptive floor: a fraction of the session's own healthy mean.
                        let adaptive = (baseline * state.entropy_trend_relative_factor)
                            .max(state.entropy_trend_absolute_min_nats);
                        // Never exceed the static ceiling configured by the operator.
                        adaptive.min(trend_threshold_nats)
                    }
                }
                None => {
                    // Still in warm-up: suppress trend check but keep other checks.
                    0.0
                }
            }
        } else {
            // Adaptive mode disabled — use the static threshold verbatim.
            trend_threshold_nats
        };
    // Persist for use by render_health_dump.
    state.entropy_effective_trend_threshold = effective_trend_threshold_nats;

    // Effective interval-floor threshold: same adaptive formula as the trend threshold
    // but applied to `interval_floor_threshold_nats`.  When adaptive mode is on and the
    // baseline is fixed, the floor becomes max(absolute_min, baseline × factor), capped
    // at the configured static ceiling.  This prevents false positives on models whose
    // healthy entropy mean sits at or below the static 0.2 nats default.
    // During warm-up (baseline not yet fixed) the interval-floor check is suppressed
    // (effective = 0.0) — the hard floor and argmax-dominance checks remain active.
    let effective_interval_floor_threshold_nats: f32 =
        if state.entropy_baseline_window > 0 && state.entropy_trend_relative_factor > 0.0 {
            match state.entropy_baseline_mean {
                Some(baseline) => {
                    if baseline < state.entropy_trend_absolute_min_nats {
                        // Same reasoning as the trend threshold above: if the model
                        // naturally operates at near-zero entropy, the interval-floor
                        // cycling check is also meaningless.
                        0.0
                    } else {
                        let adaptive = (baseline * state.entropy_trend_relative_factor)
                            .max(state.entropy_trend_absolute_min_nats);
                        adaptive.min(interval_floor_threshold_nats)
                    }
                }
                None => 0.0, // still in warm-up: suppress
            }
        } else {
            interval_floor_threshold_nats
        };
    state.entropy_effective_interval_floor_threshold = effective_interval_floor_threshold_nats;

    // Once entropy drops below the effective trend threshold on any single check,
    // switch to dense mode so every subsequent step is logged at full resolution.
    // Suppressed inside think blocks for the same reason as the trend window.
    if effective_trend_threshold_nats > 0.0
        && entropy_nats < effective_trend_threshold_nats
        && !state.inside_think_block
    {
        state.dense_mode = true;
    }
    if state.log_capacity > 0 {
        if state.health_log.len() >= state.log_capacity {
            state.health_log.remove(0);
        }
        state.health_log.push(HealthSample {
            step,
            entropy_nats,
            max_abs,
            p_max,
            argmax_token,
            logit_gap,
        });
    }

    // Hard floor: abort after `hard_min_consec` consecutive steps below threshold
    // where the *same* argmax token wins each time. Switching to a different token
    // resets the counter so that e.g. `* ... *\n` (three different structural tokens
    // each low-entropy) does not falsely trigger. Only a genuine single-token lock-on
    // (same argmax N times in a row, all below threshold) fires the abort.
    //
    // Structural tokens (newline, space, markdown punctuation) are exempt from the
    // counter entirely: a deterministic newline or asterisk is legitimate in context.
    // Repetition checks (`TokenRepetition`, `PhraseLoop`) still catch stuck structural
    // tokens if they actually repeat destructively.
    //
    // Inside a <think> block the counters are reset each call so they cannot
    // carry stale state into the response phase when the block closes.
    let argmax_is_structural = structural_token_ids.contains(&argmax_token);
    if hard_threshold_nats > 0.0 {
        if state.inside_think_block || argmax_is_structural {
            // Inside think block or structural token: reset rather than accumulate.
            state.hard_floor_consec = 0;
            state.hard_floor_consec_token = None;
        } else if entropy_nats < hard_threshold_nats {
            if state.hard_floor_consec_token == Some(argmax_token) {
                // Same token as the ongoing run — extend it.
                state.hard_floor_consec += 1;
            } else {
                // Different token: start a fresh run of length 1.
                state.hard_floor_consec = 1;
                state.hard_floor_consec_token = Some(argmax_token);
            }
        } else {
            state.hard_floor_consec = 0;
            state.hard_floor_consec_token = None;
        }
        let min_consec = hard_min_consec.max(1);
        if state.hard_floor_consec >= min_consec && !state.inside_think_block {
            let diagnosis = diagnose_collapse(
                step,
                entropy_nats,
                false,
                &state.recent_entropies,
                &state.recent_max_abs,
            );
            let top_tokens = top_k_from_pvec(&p_vec, 15);
            return Ok(Some(HealthEvent::EntropyCollapse {
                entropy_nats,
                threshold_nats: hard_threshold_nats,
                sustained: false,
                step,
                diagnosis,
                top_tokens,
            }));
        }
    }

    // Sustained trend: ALL samples in the window must be below the effective soft
    // threshold.  Requires unanimous collapse — a single high-entropy token (legit
    // roleplay content) breaks the window and prevents a false-positive abort.
    // Suppressed inside think blocks (window not populated inside think blocks anyway).
    // Uses `effective_trend_threshold_nats` which is either the static threshold or the
    // adaptive per-session floor computed above.  0.0 means suppressed (warm-up period).
    //
    // Recent-coherent veto: even when the trend window is unanimously low, check the
    // full health_log for any recent high-entropy sample.  Interval checks often land on
    // structural tokens (spaces, newlines) between semantic choices; the dense-mode steps
    // that land on actual words ARE in the log but not in `recent_entropies`.  If the log
    // shows a coherent choice (entropy > effective_threshold × veto_factor) within the
    // last `trend_recent_veto_window` entries, the trend window is a sampling artefact
    // and the abort is suppressed for this cycle.
    if !state.inside_think_block
        && trend_window > 0
        && effective_trend_threshold_nats > 0.0
        && state.recent_entropies.len() == trend_window
        && state
            .recent_entropies
            .iter()
            .all(|&h| h < effective_trend_threshold_nats)
    {
        // Apply recent-coherent veto before committing to the abort.
        let vetoed = if trend_recent_veto_window > 0 && trend_recent_veto_factor > 0.0 {
            let veto_high_water = effective_trend_threshold_nats * trend_recent_veto_factor;
            state
                .health_log
                .iter()
                .rev()
                .take(trend_recent_veto_window)
                .any(|s| s.entropy_nats >= veto_high_water)
        } else {
            false
        };
        if !vetoed {
            let mean = state.recent_entropies.iter().sum::<f32>() / trend_window as f32;
            let diagnosis = diagnose_collapse(
                step,
                mean,
                true,
                &state.recent_entropies,
                &state.recent_max_abs,
            );
            let top_tokens = top_k_from_pvec(&p_vec, 15);
            return Ok(Some(HealthEvent::EntropyCollapse {
                entropy_nats: mean,
                threshold_nats: effective_trend_threshold_nats,
                sustained: true,
                step,
                diagnosis,
                top_tokens,
            }));
        }
    }

    // Interval-floor cycling check: N consecutive interval-only checks all below
    // a deep threshold, regardless of which token wins. Catches rotating-token loops
    // (e.g. `*` / `\n` / ` `) where the hard-floor same-token requirement never fires
    // but every sampled distribution is near-deterministic for tens of steps.
    // Counter is reset while inside a think block so stale state cannot carry over.
    // Uses `effective_interval_floor_threshold_nats` (adaptive when baseline mode is on)
    // so models with naturally low entropy do not false-positive here.
    // 0.0 means suppressed (adaptive warm-up period).
    if is_interval && effective_interval_floor_threshold_nats > 0.0 && interval_floor_min_consec > 0
    {
        if state.inside_think_block || argmax_is_structural {
            // Structural tokens are legitimately deterministic; don't let them
            // accumulate the interval-floor cycling counter.
            state.interval_floor_consec = 0;
        } else if entropy_nats < effective_interval_floor_threshold_nats {
            state.interval_floor_consec += 1;
        } else {
            state.interval_floor_consec = 0;
        }
        if state.interval_floor_consec >= interval_floor_min_consec && !state.inside_think_block {
            let diagnosis = diagnose_collapse(
                step,
                entropy_nats,
                false,
                &state.recent_entropies,
                &state.recent_max_abs,
            );
            let top_tokens = top_k_from_pvec(&p_vec, 15);
            return Ok(Some(HealthEvent::EntropyCollapse {
                entropy_nats,
                threshold_nats: effective_interval_floor_threshold_nats,
                sustained: false,
                step,
                diagnosis,
                top_tokens,
            }));
        }
    }

    // Argmax dominance check: one token winning > fraction of the last
    // interval_argmax_dominance_window interval checks signals that the model is
    // stuck generating a single structural token (e.g. `…`, `*`, `—`) as the most
    // likely choice at every sample point, even without a tight consecutive repeat.
    // Only advances on interval checks; dense-mode steps are excluded to prevent
    // rapid-fire pushes that could flood the window with non-interval samples.
    if is_interval
        && interval_argmax_dominance_window > 1
        && interval_argmax_dominance_fraction > 0.0
        && !argmax_is_structural
    {
        state.recent_interval_argmax.push_back(argmax_token);
        while state.recent_interval_argmax.len() > interval_argmax_dominance_window {
            state.recent_interval_argmax.pop_front();
        }
        if state.recent_interval_argmax.len() == interval_argmax_dominance_window {
            // Count how many times the most-frequent token appears.
            let mut counts = std::collections::HashMap::<u32, usize>::new();
            for &t in &state.recent_interval_argmax {
                *counts.entry(t).or_insert(0) += 1;
            }
            if let Some((&dominant_token, &count)) = counts.iter().max_by_key(|(_, &c)| c) {
                let frac = count as f32 / interval_argmax_dominance_window as f32;
                if frac >= interval_argmax_dominance_fraction {
                    let top_tokens = top_k_from_pvec(&p_vec, 15);
                    return Ok(Some(HealthEvent::ArgmaxDominance {
                        dominant_token,
                        count,
                        window: interval_argmax_dominance_window,
                        step,
                        top_tokens,
                    }));
                }
            }
        }
    }

    Ok(None)
}

// ── Collapse diagnosis ───────────────────────────────────────────────────────

/// Return the top-k `(token_id, prob, logit_rel)` tuples from a softmax probability
/// vector, sorted descending by probability.
/// `logit_rel` = ln(p_i) - ln(p_1): logit distance from the winner (0 for rank-1).
fn top_k_from_pvec(p_vec: &[f32], k: usize) -> Vec<(u32, f32, f32)> {
    let mut pairs: Vec<(u32, f32)> = p_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    // Vocab is ~32k so a full sort is fine (~1ms). Already paid the 128KB transfer.
    pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    let ln_p1 = pairs.first().map(|(_, p)| p.ln()).unwrap_or(0.0);
    pairs
        .into_iter()
        .map(|(tok, p)| {
            let logit_rel = if p > 1e-40 { p.ln() - ln_p1 } else { -100.0 };
            (tok, p, logit_rel)
        })
        .collect()
}

/// Derive a human-readable likely-cause string from the entropy and max|logit|
/// trends at the moment of collapse.
///
/// **Primary discriminator: `max_abs` at collapse + its slope.**
///
/// These two failure modes have mutually exclusive magnitude signatures:
/// - F16 MLP overflow: max|logit| inflates toward 65504. Entropy drops *because*
///   one logit dominates an unnaturally large value. max_abs will be high and its
///   slope will be positive (activations were accumulating toward saturation).
/// - K/V cache corruption: model computes correct-scale operations on wrong
///   context. max|logit| stays in a normal range (5–50 for typical LMs).
///   Entropy collapses without any magnitude inflation.
///
/// Entropy slope and step depth are secondary signals used when magnitude
/// history is too short to be conclusive.
fn diagnose_collapse(
    step: usize,
    entropy_nats: f32,
    sustained: bool,
    entropy_history: &VecDeque<f32>,
    max_abs_history: &VecDeque<f32>,
) -> String {
    let n = entropy_history.len();

    // ── Entropy slope (secondary) ─────────────────────────────────────────────
    let prior_entropy: Vec<f32> = entropy_history
        .iter()
        .copied()
        .take(n.saturating_sub(1))
        .collect();
    let prior_entropy_max = prior_entropy.iter().copied().fold(0f32, f32::max);
    let entropy_slope = linear_slope(&prior_entropy);

    // ── Max-abs trend (primary discriminator) ─────────────────────────────────
    let max_abs_at_collapse = max_abs_history.back().copied().unwrap_or(0.0);
    let prior_max_abs: Vec<f32> = max_abs_history
        .iter()
        .copied()
        .take(max_abs_history.len().saturating_sub(1))
        .collect();
    let max_abs_slope = linear_slope(&prior_max_abs);
    // Mean of prior max_abs — "normal" logit scale for this model on this run.
    let prior_max_abs_mean = if prior_max_abs.is_empty() {
        max_abs_at_collapse
    } else {
        prior_max_abs.iter().sum::<f32>() / prior_max_abs.len() as f32
    };
    // Ratio of collapse value to prior mean — 1.0 = stable, >3 = inflating.
    let max_abs_ratio = if prior_max_abs_mean > 0.0 {
        max_abs_at_collapse / prior_max_abs_mean
    } else {
        1.0
    };

    let gradual = sustained || entropy_slope < -0.3;
    let sudden_from_healthy = prior_entropy_max > 2.0 && entropy_nats < 0.5;

    // ── Primary: magnitude-based discrimination ───────────────────────────────
    //
    // F16 saturates at 65504. Values above ~500 with a rising slope are
    // conclusive for overflow. Values below ~200 with a flat slope conclusively
    // rule it out in favour of K/V corruption. These regions don't overlap.
    let cause = if max_abs_at_collapse > 5_000.0 && max_abs_slope > 0.0 {
        // Logit scale exploded with a positive slope — activation magnitudes were
        // inflating toward F16 saturation (65504). One logit dominates the softmax,
        // collapsing the distribution. Observed: max|logit| climbing across intervals.
        format!(
            "logit magnitude explosion (high confidence): max|logit|={max_abs_at_collapse:.0} \
             rising at +{max_abs_slope:.1}/interval — \
             one or more activations are saturating, forcing the distribution to a \
             near-argmax state. The model has lost the ability to weigh alternatives."
        )
    } else if max_abs_at_collapse > 500.0 || max_abs_ratio > 3.0 {
        // Elevated magnitude without a clear slope — probable inflation in progress.
        format!(
            "logit magnitude elevation (probable): max|logit|={max_abs_at_collapse:.1} \
             ({max_abs_ratio:.1}x above session mean={prior_max_abs_mean:.1}), \
             slope={max_abs_slope:+.2}/interval — \
             logit scale is abnormally high; distribution is narrowing toward \
             a degenerate peaked state."
        )
    } else if max_abs_at_collapse < 200.0 && sudden_from_healthy && step > 100 {
        // Normal logit scale throughout, sudden entropy cliff at deep context depth.
        // The model is computing numerically reasonable values, but the distribution
        // collapsed — consistent with attention retrieving semantically wrong context.
        format!(
            "context degeneration (high confidence): max|logit|={max_abs_at_collapse:.1} \
             (session mean={prior_max_abs_mean:.1}, ratio={max_abs_ratio:.2}x) — \
             logit scale is normal but entropy collapsed suddenly at step {step}. \
             The model's attention is no longer retrieving coherent context; \
             generation has entered a degenerate attractor state."
        )
    } else if max_abs_at_collapse < 200.0 && gradual {
        // Entropy eroded steadily — precision is bleeding out of the residual stream
        // over many layers, not a sudden event.
        format!(
            "residual signal erosion: max|logit|={max_abs_at_collapse:.1} \
             (normal scale), entropy declining at {entropy_slope:+.3} nats/interval — \
             the model's internal representations are losing semantic content \
             gradually across layers, eventually unable to distinguish between tokens."
        )
    } else if max_abs_at_collapse < 200.0 && sudden_from_healthy {
        // Normal scale, sudden collapse, shallower context — less certain.
        format!(
            "context degeneration (probable): max|logit|={max_abs_at_collapse:.1} \
             (normal scale), sudden collapse from healthy distribution at step {step} \
             without magnitude inflation — attention is producing low-entropy output \
             despite numerically normal activations."
        )
    } else if n < 2 || max_abs_history.len() < 2 {
        format!(
            "insufficient measurement history (entropy samples={n}, \
             magnitude samples={}) to classify — \
             max|logit|={max_abs_at_collapse:.1} at step {step}",
            max_abs_history.len()
        )
    } else {
        format!(
            "unclassified collapse: max|logit|={max_abs_at_collapse:.1} \
             (ratio={max_abs_ratio:.1}x session mean), \
             entropy_slope={entropy_slope:+.3} nats/interval, \
             magnitude_slope={max_abs_slope:+.2}/interval, step={step}"
        )
    };

    format!(" — likely cause: {cause}")
}

/// Compute the linear regression slope over a slice of samples.
/// Returns 0.0 for fewer than 2 samples.
fn linear_slope(samples: &[f32]) -> f32 {
    let m = samples.len();
    if m < 2 {
        return 0.0;
    }
    let mf = m as f32;
    let mean_x = (mf - 1.0) / 2.0;
    let mean_y = samples.iter().sum::<f32>() / mf;
    let cov: f32 = samples
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f32 - mean_x) * (y - mean_y))
        .sum();
    let var_x: f32 = samples
        .iter()
        .enumerate()
        .map(|(i, _)| (i as f32 - mean_x).powi(2))
        .sum();
    if var_x > 0.0 {
        cov / var_x
    } else {
        0.0
    }
}

/// Check the recent-token window for a stuck-token repetition run.
///
/// CPU-only; no GPU involvement.
///
/// Returns `Some(event)` when the `threshold` most-recent tokens are all identical.
pub fn check_repetition(state: &DecodeHealthState, threshold: usize) -> Option<HealthEvent> {
    if state.recent_tokens.len() < threshold {
        return None;
    }
    // Walk backwards through the window counting the run length.
    let mut iter = state.recent_tokens.iter().rev();
    let first = *iter.next()?;
    let run_length = 1 + iter.take(threshold - 1).filter(|&&t| t == first).count();
    if run_length >= threshold {
        Some(HealthEvent::TokenRepetition {
            token: first,
            run_length,
        })
    } else {
        None
    }
}

/// Check the recent-token window for a multi-token phrase loop (local minimum).
///
/// CPU-only; no GPU involvement. O(`max_period` × `min_reps` × `max_period`) per
/// call — typically < 200 integer comparisons.
///
/// For each candidate period length `p` (1 ≤ p ≤ `max_period`), counts how many
/// times the last `p` tokens repeat immediately before themselves. Returns on the
/// first period that reaches `min_reps` complete consecutive repetitions.
///
/// Note: single-token repetition (period=1) is also caught here; prefer calling
/// [`check_repetition`] first so the more-specific variant appears in the event.
///
/// `min_total_tokens` is a lower bound on the total token span (`period × reps`)
/// that must be reached before the abort fires.  The effective minimum repetition
/// count for a given period `p` is:
///
/// ```text
/// effective_min_reps = max(min_reps, ceil(min_total_tokens / p))
/// ```
///
/// This prevents false-positives on short-period phrases that are normal in prose
/// (e.g. `lower, lower` — 2-token phrase × 2 reps = 4 tokens, below any reasonable
/// floor) while preserving sensitivity for longer phrases (a 5-token phrase × 2 reps
/// = 10 tokens is already at a floor of 10 and fires immediately).
/// Pass `0` to disable the total-token floor.
pub fn check_phrase_loop(
    state: &DecodeHealthState,
    max_period: usize,
    min_reps: usize,
    min_total_tokens: usize,
) -> Option<HealthEvent> {
    let tokens = &state.recent_tokens;
    let n = tokens.len();

    for period in 2..=max_period {
        // Effective minimum repetitions for this period, raised so that the total
        // repeated token span is at least `min_total_tokens`.
        let effective_min_reps = if min_total_tokens > 0 {
            min_reps.max(min_total_tokens.div_ceil(period))
        } else {
            min_reps
        };
        let needed = period * effective_min_reps;
        if n < needed {
            continue;
        }
        // Count consecutive copies of the phrase tokens[n-period..n]
        // by looking at the blocks immediately preceding it.
        let mut reps = 1usize;
        loop {
            let block_end = n - period * reps;
            if block_end < period {
                break;
            }
            let block_start = block_end - period;
            let phrase_start = n - period;
            let matches = (0..period).all(|i| tokens[block_start + i] == tokens[phrase_start + i]);
            if matches {
                reps += 1;
                if reps >= effective_min_reps {
                    return Some(HealthEvent::PhraseLoop { period, reps });
                }
            } else {
                break;
            }
        }
    }
    None
}

// ── Diagnostic dump ──────────────────────────────────────────────────────────

/// Render a human-readable diagnostic dump of the full decode health log.
///
/// Produces four ASCII bar charts (entropy, p_max, max|logit|, logit gap) and a
/// per-sample table covering the entire decode session up to the abort point.
/// Called after any entropy-triggered abort so the health trajectory is visible
/// in the tracing output alongside the abort message.
///
/// # Arguments
/// - `log`                  — full sample history from `DecodeHealthState::health_log`
/// - `abort_step`           — decode step at which the abort was triggered
/// - `hard_threshold_nats`  — hard-floor entropy threshold (for chart annotation)
/// - `trend_threshold_nats` — sustained-trend threshold (for chart annotation)
/// - `interval`             — steps between each check (for context)
/// - `prefill_token_count`  — prompt length in tokens (context depth at decode start)
/// - `temperature` / `top_k` / `top_p` / `rep_penalty` — sampling config snapshot
//
/// Build a filtered view of the health log for chart and table rendering.
///
/// Keeps only:
/// - Samples recorded at the normal check interval (`step % interval == 0`)
/// - Pre-boundary probe samples (`step % 32 == 31`)
/// - The last 5 samples unconditionally (dense-mode final approach)
/// - The abort sample (always the last entry)
///
/// Stats (min/max/mean/slope) are still computed from the full log.
fn filter_log_for_display(log: &[HealthSample], interval: usize) -> Vec<&HealthSample> {
    let n = log.len();
    if n == 0 {
        return vec![];
    }
    let mut keep = std::collections::BTreeSet::new();
    for (i, s) in log.iter().enumerate() {
        if interval > 0 && s.step % interval == 0 {
            keep.insert(i);
        }
        if s.step > 0 && s.step % 32 == 31 {
            keep.insert(i);
        }
    }
    // Last 5 samples (dense-mode final approach at full resolution)
    for i in n.saturating_sub(5)..n {
        keep.insert(i);
    }
    // Abort sample is always included
    keep.insert(n - 1);
    keep.into_iter().map(|i| &log[i]).collect()
}

pub fn render_health_dump(
    log: &[HealthSample],
    abort_step: usize,
    hard_threshold_nats: f32,
    trend_threshold_nats: f32,
    interval: usize,
    prefill_token_count: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    rep_penalty: f32,
    recent_tokens: &std::collections::VecDeque<u32>,
    abort_top_tokens: &[(u32, f32, f32)],
) -> String {
    use std::fmt::Write as _;
    const W: usize = 40;
    let mut out = String::with_capacity(4096);
    let n = log.len();

    let _ = writeln!(out, "\n============================== decode health diagnostic dump ==============================");
    let _ = writeln!(
        out,
        " aborted at step {abort_step}  |  prefill: {prefill_token_count} tokens  |\
         check interval: {interval} steps  |  {n} sample(s) recorded"
    );
    let _ = writeln!(
        out,
        " sampling: temp={temperature:.2}  top-k={top_k}  top-p={top_p:.2}  repeat-penalty={rep_penalty:.3}"
    );

    if log.is_empty() {
        let _ = writeln!(
            out,
            " (no samples recorded before abort -- collapsed on first check)"
        );
        let _ = writeln!(out, "=========================================================================================");
        return out;
    }

    // Build a sparse display log: interval + pre-boundary + last 5 samples.
    // Full log is still used for statistics (min/max/mean/slope).
    let display_log = filter_log_for_display(log, interval);
    let dn = display_log.len();

    // ── Shared chart helpers ──────────────────────────────────────────────────

    fn filled_bar(val: f32, lo: f32, hi: f32, width: usize) -> String {
        let range = (hi - lo).max(1e-6);
        let fill = ((val - lo) / range * width as f32)
            .round()
            .clamp(0.0, width as f32) as usize;
        format!("|{}{}|", "#".repeat(fill), " ".repeat(width - fill))
    }

    fn series_stats(vals: &[f32]) -> (f32, f32, f32) {
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        (min, max, mean)
    }

    // ── Chart 1: entropy (nats) ───────────────────────────────────────────────
    {
        let vals: Vec<f32> = log.iter().map(|s| s.entropy_nats).collect();
        let (min_v, max_v, mean_v) = series_stats(&vals);
        let slope = linear_slope(&vals);
        let hi = max_v.max(trend_threshold_nats).max(hard_threshold_nats) * 1.05;
        let _ = writeln!(
            out,
            "\n entropy (nats)  min={min_v:.3}  max={max_v:.3}  mean={mean_v:.3}  slope={slope:+.4}/interval"
        );
        let _ = writeln!(
            out,
            "   scale: 0.000 {}> {hi:.3} nats",
            "-".repeat(W.saturating_sub(4))
        );
        let mut trend_marked = false;
        let mut hard_marked = false;
        for (i, s) in display_log.iter().enumerate() {
            let abort_str = if i + 1 == dn { " [ABORT]" } else { "        " };
            let b = filled_bar(s.entropy_nats, 0.0, hi, W);
            let mut note = String::new();
            if !trend_marked && trend_threshold_nats > 0.0 && s.entropy_nats < trend_threshold_nats
            {
                let _ = write!(note, "  <- trend floor {trend_threshold_nats:.3}");
                trend_marked = true;
            }
            if !hard_marked && hard_threshold_nats > 0.0 && s.entropy_nats < hard_threshold_nats {
                let _ = write!(note, "  <- hard floor {hard_threshold_nats:.3}");
                hard_marked = true;
            }
            let _ = writeln!(
                out,
                "   step {:>5} {b} {:.3} nats{abort_str}{note}",
                s.step, s.entropy_nats
            );
        }
        let _ = writeln!(out, "              +{}+", "-".repeat(W));
    }

    // ── Chart 2: peak probability p_max ──────────────────────────────────────
    {
        let vals: Vec<f32> = log.iter().map(|s| s.p_max).collect();
        let (min_v, max_v, mean_v) = series_stats(&vals);
        let slope = linear_slope(&vals);
        let _ = writeln!(
            out,
            "\n peak prob p_max  min={min_v:.4}  max={max_v:.4}  mean={mean_v:.4}  slope={slope:+.5}/interval"
        );
        let _ = writeln!(
            out,
            "   scale: 0.000 {}> 1.000",
            "-".repeat(W.saturating_sub(4))
        );
        for (i, s) in display_log.iter().enumerate() {
            let abort_str = if i + 1 == dn { " [ABORT]" } else { "        " };
            let b = filled_bar(s.p_max, 0.0, 1.0, W);
            let _ = writeln!(out, "   step {:>5} {b} {:.4}{abort_str}", s.step, s.p_max);
        }
        let _ = writeln!(out, "              +{}+", "-".repeat(W));
    }

    // ── Chart 3: max|logit| ───────────────────────────────────────────────────
    {
        let vals: Vec<f32> = log.iter().map(|s| s.max_abs).collect();
        let (min_v, max_v, mean_v) = series_stats(&vals);
        let slope = linear_slope(&vals);
        let hi = (max_v * 1.1).max(1.0);
        let stable = if slope.abs() < 0.5 && max_v < 500.0 {
            "  STABLE"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "\n max|logit|  min={min_v:.1}  max={max_v:.1}  mean={mean_v:.1}  slope={slope:+.3}/interval{stable}"
        );
        let _ = writeln!(
            out,
            "   scale: 0 {}> {hi:.1}",
            "-".repeat(W.saturating_sub(2))
        );
        for (i, s) in display_log.iter().enumerate() {
            let abort_str = if i + 1 == dn { " [ABORT]" } else { "        " };
            let b = filled_bar(s.max_abs, 0.0, hi, W);
            let _ = writeln!(out, "   step {:>5} {b} {:.1}{abort_str}", s.step, s.max_abs);
        }
        let _ = writeln!(out, "              +{}+", "-".repeat(W));
    }

    // ── Chart 4: logit gap ────────────────────────────────────────────────────
    {
        // Cap display at 50 to keep collapsed cases readable.
        let vals: Vec<f32> = log.iter().map(|s| s.logit_gap.min(50.0)).collect();
        let (min_v, max_v, mean_v) = series_stats(&vals);
        let slope = linear_slope(&vals);
        let hi = (max_v * 1.1).max(1.0);
        let _ = writeln!(
            out,
            "\n logit gap (log_p1 - log_p2 = logit_1 - logit_2)  min={min_v:.2}  max={max_v:.2}  mean={mean_v:.2}  slope={slope:+.3}/interval"
        );
        let _ = writeln!(
            out,
            "   scale: 0 {}> {hi:.2}",
            "-".repeat(W.saturating_sub(2))
        );
        for (i, s) in display_log.iter().enumerate() {
            let abort_str = if i + 1 == dn { " [ABORT]" } else { "        " };
            let v_capped = s.logit_gap.min(50.0);
            let b = filled_bar(v_capped, 0.0, hi, W);
            let _ = writeln!(
                out,
                "   step {:>5} {b} {:.2}{abort_str}",
                s.step, s.logit_gap
            );
        }
        let _ = writeln!(out, "              +{}+", "-".repeat(W));
    }

    // ── Per-sample table ──────────────────────────────────────────────────────
    let _ = writeln!(out, "\n per-interval table:");
    let _ = writeln!(
        out,
        "   {:>3}  {:>6}  {:>8}  {:>7}  {:>7}  {:>10}  {:>10}",
        "#", "step", "entropy", "p_max", "argmax", "logit_gap", "max|logit|"
    );
    let _ = writeln!(
        out,
        "   ---  ------  --------  -------  -------  ----------  ----------"
    );
    for (i, s) in display_log.iter().enumerate() {
        let is_abort = i + 1 == dn;
        let prefix = if is_abort { " *" } else { "  " };
        let _ = writeln!(
            out,
            "{}{:>3}  {:>6}  {:>8.3}  {:>7.4}  {:>7}  {:>10.2}  {:>10.1}{}",
            prefix,
            i + 1,
            s.step,
            s.entropy_nats,
            s.p_max,
            s.argmax_token,
            s.logit_gap,
            s.max_abs,
            if is_abort { "  <- ABORT" } else { "" },
        );
    }

    // ── Final approach: last 5 individual token steps before abort ────────────
    // Shows the actual consecutive token IDs at steps abort_step-5..abort_step-1.
    // (The abort fires before this step's token is sampled, so the most-recent
    // token in recent_tokens is from abort_step-1.)
    const FINAL_N: usize = 5;
    let tok_tail: Vec<u32> = recent_tokens
        .iter()
        .rev()
        .take(FINAL_N)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if !tok_tail.is_empty() {
        let tail_len = tok_tail.len();
        let first_step = abort_step.saturating_sub(tail_len);
        let _ = writeln!(out, "\n final {tail_len} token steps before abort:");
        let _ = writeln!(out, "   {:>6}  {:>8}", "step", "token-id");
        let _ = writeln!(out, "   ------  --------");
        for (j, &tok) in tok_tail.iter().enumerate() {
            let _ = writeln!(out, "   {:>6}  {:>8}", first_step + j, tok);
        }
        let _ = writeln!(
            out,
            "   {:>6}  {:>8}  <- ABORT (not yet sampled)",
            abort_step, "?"
        );
    }

    // ── Distribution at abort step ────────────────────────────────────────────
    if !abort_top_tokens.is_empty() {
        let cum_total: f32 = abort_top_tokens.iter().map(|(_, p, _)| p).sum();
        let _ = writeln!(
            out,
            "\n distribution at abort step (top {}):",
            abort_top_tokens.len()
        );
        let _ = writeln!(
            out,
            "   {:>4}  {:>8}  {:>10}  {:>10}  {:>8}  bar",
            "rank", "token-id", "prob", "logit-rel", "cumul"
        );
        let _ = writeln!(
            out,
            "   ----  --------  ----------  ----------  --------  ------------------------------"
        );
        let max_p = abort_top_tokens.first().map(|(_, p, _)| *p).unwrap_or(1.0);
        let mut cum = 0.0f32;
        for (rank, &(tok, p, logit_rel)) in abort_top_tokens.iter().enumerate() {
            cum += p;
            let fill = ((p / max_p) * 30.0).round().clamp(0.0, 30.0) as usize;
            let bar = format!("|{}{}|", "#".repeat(fill), " ".repeat(30 - fill));
            let _ = writeln!(
                out,
                "   {:>4}  {:>8}  {:>10.6}  {:>10.3}  {:>7.4}  {bar}",
                rank + 1,
                tok,
                p,
                logit_rel,
                cum
            );
        }
        let _ = writeln!(
            out,
            "   top-{} cumulative mass: {:.6}  ({:.2}% of distribution)",
            abort_top_tokens.len(),
            cum_total,
            cum_total * 100.0
        );
    }

    let _ = writeln!(out, "\n=========================================================================================");
    out
}
