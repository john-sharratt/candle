//! A readable dump of a decode's health trajectory, written after an entropy abort.
//!
//! The dump keeps the interval checks, the pre-boundary probes, the last five
//! samples and the abort sample, so the charts stay readable while the summary
//! statistics come from the full log. Four charts and a per-sample table show
//! entropy, peak probability, the largest logit magnitude and the logit gap, with
//! the thresholds marked. The last few token steps before the abort are listed, so
//! the approach to the abort can be read directly.

use super::*;

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
pub(super) fn filter_log_for_display(log: &[HealthSample], interval: usize) -> Vec<&HealthSample> {
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
