//! Detects a decode whose next-token distribution has collapsed to near certainty
//! while it is stuck.
//!
//! Entropy is computed from the full probability vector and checked three ways: a
//! single step below a hard floor, a sustained low trend over a window, and a
//! cycling or dominance pattern across interval checks. The trend and floor
//! thresholds can be adaptive, fixed to the session's own healthy mean and capped
//! at the configured ceiling, so a model whose normal entropy is low is not
//! flagged. Structural tokens and output inside a reasoning block reset their
//! counters instead of accumulating, so a legitimately deterministic newline does
//! not end a decode.

use super::*;

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
pub(super) fn top_k_from_pvec(p_vec: &[f32], k: usize) -> Vec<(u32, f32, f32)> {
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
pub(super) fn diagnose_collapse(
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
pub(super) fn linear_slope(samples: &[f32]) -> f32 {
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
