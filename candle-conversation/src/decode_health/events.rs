//! The health events a decode can raise, and what each one carries.
//!
//! Each event names one detected degradation: a repeated token, a repeated phrase,
//! NaN, infinity, an oversized logit, a collapsed distribution, or one token
//! dominating the interval checks. The collapse event carries a diagnosis derived
//! from the magnitude and entropy history, and the top of the distribution at the
//! moment it fired, so an abort explains itself when it is reported.

use super::*;

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
