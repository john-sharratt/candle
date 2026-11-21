//! Holds the thresholds that decide when a decode has degenerated and must be
//! aborted early.
//!
//! Every check is off unless its feature is compiled in and the config is
//! enabled, so a build without the feature pays nothing. The checks catch
//! non-finite or overflowing logits, token and phrase repetition, and entropy
//! collapse, sampled on the device at a fixed interval. Each has a count of
//! consecutive samples that must agree, which keeps one structural token from
//! firing it. Structural tokens (newlines, spaces, markdown punctuation) are
//! resolved from the tokenizer and exempt from the collapse counters, while the
//! repetition checks still apply to them. `for_chat` is the preset an
//! interactive character runs under.

// Decode Health Configuration

/// Configuration for decode-time health monitoring.
///
/// Detects common degradation patterns (NaN/Inf logits, stuck-token loops)
/// that produce garbage output, and aborts the affected sequence early.
///
/// The monitoring code is compiled only with `--features decode-health`.
/// This config struct is always compiled so you can set it unconditionally.
///
/// # Zero-cost guarantee
///
/// * Feature absent: zero check code compiled.
/// * Feature present, `enabled = false`: single branch compile-time eliminated.
/// * Feature present, `enabled = true`: checks run per the intervals below.
#[derive(Debug, Clone)]
pub struct DecodeHealthConfig {
    /// Whether monitoring is active. Default: `false`.
    ///
    /// Must be `true` for any checks to run, even when the feature is enabled.
    pub enabled: bool,

    /// Size of the sliding token window for repetition detection.
    pub repetition_window: usize,

    /// Consecutive identical tokens needed to trigger an abort.
    pub repetition_threshold: usize,

    /// Maximum phrase length (in tokens) to test for cyclic loops.
    ///
    /// Checks periods 2 through `phrase_loop_max_period`. Set to `0` to disable
    /// phrase-loop detection entirely.
    pub phrase_loop_max_period: usize,

    /// Number of consecutive full repetitions of a phrase required to abort.
    pub phrase_loop_min_reps: usize,

    /// Minimum total token span (period × reps) required before a phrase-loop abort
    /// fires.  Acts as a per-period floor on `phrase_loop_min_reps`: the effective
    /// minimum repetitions for a given period `p` is
    ///
    /// ```text
    /// effective_min_reps = max(phrase_loop_min_reps,
    ///                         ceil(phrase_loop_min_total_tokens / p))
    /// ```
    ///
    /// This prevents false-positives on short-period phrases that are common in
    /// natural prose (e.g. `lower, lower` — a 2-token phrase repeated twice is only
    /// 4 tokens and is a legitimate literary device) while keeping full sensitivity
    /// for longer-period phrases (a 5-token phrase × 2 reps = 10 tokens is already
    /// at the threshold and will still fire immediately).
    ///
    /// `0` disables the total-token floor and falls back to `phrase_loop_min_reps`
    /// for all periods.
    pub phrase_loop_min_total_tokens: usize,

    /// Steps between GPU-side logit checks (NaN / Inf / magnitude).
    ///
    /// `0` disables GPU checks entirely. Default: 16 (= half the KV page size of 32).
    pub logit_check_interval: usize,

    /// Abort when max|logit| exceeds this threshold.
    /// F16 saturates at 65504; values above ~50000 indicate impending overflow.
    pub logit_magnitude_threshold: f32,

    /// Hard-floor entropy threshold in nats.
    ///
    /// Abort fires when H drops below this for `entropy_hard_min_consec` consecutive
    /// checked steps. A single dip (structural token like `*` or `\n`) is normal;
    /// requiring consecutive steps prevents false positives on punctuation bursts.
    /// `0.0` disables this check.
    pub entropy_hard_threshold_nats: f32,

    /// Number of consecutive checks that must all be below `entropy_hard_threshold_nats`
    /// before the hard-floor abort fires.
    ///
    /// `1` = original single-step behaviour (may false-positive on punctuation).
    /// `3` = recommended for chat: a structural token recovers within 1 step;
    /// a genuine attractor stays stuck for many steps.
    pub entropy_hard_min_consec: usize,

    /// Number of consecutive entropy samples (taken every `logit_check_interval`
    /// steps) that must all be below `entropy_trend_threshold_nats` to trigger
    /// a sustained-collapse abort. `0` disables trend detection.
    pub entropy_trend_window: usize,

    /// Soft entropy threshold in nats for the rolling trend window.
    ///
    /// Catches gradual collapse before the hard floor is reached.
    /// Fires only when **all** samples in the trend window are below this value.
    /// Focused roleplay and think-block generation typically oscillates 0.6–2.0 nats;
    /// set this low enough (≤1.0) that healthy oscillation doesn't trigger it.
    pub entropy_trend_threshold_nats: f32,

    /// Deep interval-floor threshold in nats.
    ///
    /// Catches multi-token cycling collapse where different tokens win each interval
    /// check but every interval distribution is near-deterministic (e.g. the model
    /// loops `*`, `\n`, `space` with p_max ≥ 0.99 for dozens of steps). Unlike the
    /// hard floor (which requires the *same* token to win consecutively), this check
    /// fires when any `entropy_interval_floor_consec` consecutive **interval** samples
    /// are all below this value — regardless of which token wins.
    /// `0.0` disables this check.
    pub entropy_interval_floor_threshold_nats: f32,

    /// Number of consecutive interval-only checks all below
    /// `entropy_interval_floor_threshold_nats` before the cycling-collapse abort fires.
    /// `0` disables. Default 1 (disabled in practice — `for_chat()` sets this to 4).
    pub entropy_interval_floor_consec: usize,

    /// Sliding window size for interval-argmax dominance detection.
    ///
    /// Tracks the argmax token from the last N interval checks. When one token
    /// wins more than `interval_argmax_dominance_fraction` of those checks, the
    /// model is treating a single structural token as its dominant choice across
    /// many steps — a reliable signal of content collapse that precedes phrase-loop.
    /// `0` or `1` disables this check.
    pub interval_argmax_dominance_window: usize,

    /// Fraction threshold for the argmax-dominance check (0.0–1.0).
    ///
    /// `0.0` disables. `for_chat()` sets `0.75` (a token must win ≥ 75% of the
    /// last `interval_argmax_dominance_window` interval checks to trigger).
    pub interval_argmax_dominance_fraction: f32,

    /// Token IDs that are considered structural formatting tokens (newline, space,
    /// asterisk, etc.) and are legitimately near-deterministic in context.
    ///
    /// When the argmax token at a low-entropy step is in this set, the step is
    /// exempt from the hard-floor and interval-floor consecutive counters, and is
    /// excluded from the argmax-dominance window.  This prevents false-positive
    /// aborts when the model is generating markdown or code that happens to align
    /// many structural tokens at check points.
    ///
    /// Repetition checks (`TokenRepetition`, `PhraseLoop`) still fire normally —
    /// a structural token that actually repeats destructively is still caught.
    ///
    /// Populated automatically by `resolve_structural_tokens`.
    pub structural_token_ids: Vec<u32>,
    /// Maximum number of `HealthSample` entries kept in the full diagnostic log.
    ///
    /// Each sample is ~40 bytes. At the default interval of 50 steps this covers
    /// `capacity * 50` decode steps before the oldest entry is evicted.
    /// `0` disables the log (render_health_dump will produce no chart data).
    /// Default: 120 (= 6 000 steps at interval 50, ~5 KB).
    pub health_log_capacity: usize,

    /// Number of initial interval samples to collect before activating the adaptive
    /// entropy trend threshold.  During this warm-up period the sustained-trend
    /// check is suppressed entirely (the hard floor, interval-floor, and argmax-
    /// dominance checks remain active at full sensitivity).
    ///
    /// Once the window fills, `entropy_baseline_mean` is fixed as the session average
    /// and the effective trend threshold becomes:
    ///   `max(entropy_trend_absolute_min_nats, mean × entropy_trend_relative_factor)`
    /// capped at `entropy_trend_threshold_nats`.
    ///
    /// `0` disables adaptive mode — `entropy_trend_threshold_nats` is used verbatim.
    pub entropy_baseline_window: usize,

    /// Factor applied to the session baseline mean to derive the adaptive trend
    /// threshold.  A value of `0.25` sets the floor at 25 % of the model's own
    /// healthy entropy average, so a sharp-distribution model (e.g. MoE with
    /// mean ~0.3 nats) gets a floor of ~0.075 nats rather than the generic 0.5.
    ///
    /// `0.0` disables adaptive mode.
    pub entropy_trend_relative_factor: f32,

    /// Absolute minimum for the adaptive trend threshold in nats.
    /// Prevents the floor from reaching zero even for maximally deterministic models.
    /// Default: 0.04 nats (≈ p_max ≥ 0.961).  Only used when adaptive mode is on.
    pub entropy_trend_absolute_min_nats: f32,

    /// Recent-coherent veto: how many trailing entries of the full diagnostic log
    /// (which includes dense-mode steps, not just interval checks) to scan before
    /// firing the sustained-trend abort.  If ANY entry in those trailing entries has
    /// entropy above `entropy_trend_threshold_nats × entropy_trend_recent_veto_factor`,
    /// the trend window is treated as a sampling artefact — the model is still
    /// generating coherent (high-entropy) tokens between interval checks — and the
    /// abort is suppressed for this check cycle.
    ///
    /// This prevents false positives caused by interval checks that happen to land
    /// on structural tokens (spaces, newlines, punctuation) while the model is
    /// actively writing coherent prose between those check points.
    /// `0` disables the veto entirely.
    pub entropy_trend_recent_veto_window: usize,

    /// Multiplier applied to `effective_trend_threshold_nats` to form the veto
    /// high-water mark.  A value of `4.0` means: if any of the last
    /// `entropy_trend_recent_veto_window` log entries exceeded 4 × the effective
    /// threshold, the model is considered alive and the trend abort is vetoed.
    /// `0.0` disables the veto.
    pub entropy_trend_recent_veto_factor: f32,
}

impl Default for DecodeHealthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // 64 tokens covers phrase_loop_max_period(8) * phrase_loop_min_reps(2) = 16
            // with room for the single-token repetition_threshold check too.
            repetition_window: 64,
            repetition_threshold: 16,
            phrase_loop_max_period: 8,
            // 2 reps: catches any phrase that repeats twice in a row regardless of period.
            // 3 was too lenient — a 5-token ellipsis block repeating twice (10 tokens of
            // garbage) would not fire. 2 reps fires on the second consecutive copy.
            phrase_loop_min_reps: 2,
            // Require at least 10 tokens of total repeated material before firing.
            // For period=2 this raises the effective min_reps to 5 (10 tokens), preventing
            // false-positives on valid literary repetition such as "lower, lower".
            // For period=5 effective_min_reps remains 2 (5×2=10 ≥ 10), so the ellipsis
            // block case that motivated min_reps=2 still fires on the second copy.
            phrase_loop_min_total_tokens: 10,
            // Steps between GPU-side logit checks. 16 = half the KV page size (32),
            // so every sample lands either at a page boundary or at the midpoint.
            // This gives clean phase coverage for diagnosing page-boundary K/V corruption
            // without phase drift (gcd(16,32)=16, unlike interval=20 where gcd=4).
            logit_check_interval: 16,
            logit_magnitude_threshold: 50_000.0,
            // Hard floor: H < 0.3 nats for `entropy_hard_min_consec` consecutive steps.
            // 0.3 nats ≈ model is 97%+ confident. Structural tokens (punctuation, asterisks,
            // newlines) dip to 0.28–0.60 for a single step then recover — requiring
            // consecutive steps prevents those from triggering.
            entropy_hard_threshold_nats: 0.3,
            // Require 1 consecutive step by default (single-step check).
            // for_chat() overrides this to 3.
            entropy_hard_min_consec: 1,
            // Trend: 5 consecutive checks (80 steps at interval=16) all below 1.0 nats.
            // 2.0 was too strict — focused roleplay / think-block generation naturally
            // oscillates between ~0.6 and ~2.0; we only want to catch sustained very-low
            // entropy where every sample in the window is below 1.0 (< ~2.7 effective
            // tokens), which indicates a real softmax attractor, not normal focused text.
            entropy_trend_window: 5,
            entropy_trend_threshold_nats: 1.0,
            // Interval floor: disabled by default (for_chat sets consec=4).
            // Threshold 0.2 nats ≈ p_max ≥ 0.819; catches multi-token cycling loops
            // where every interval check is near-deterministic but a different token wins.
            entropy_interval_floor_threshold_nats: 0.2,
            entropy_interval_floor_consec: 0,
            // Argmax dominance: disabled by default (for_chat sets fraction=0.75).
            interval_argmax_dominance_window: 8,
            interval_argmax_dominance_fraction: 0.0,
            // 120 samples * 16 steps/sample = 1 920 decode steps at ~5 KB.
            health_log_capacity: 120,
            // Adaptive baseline disabled by default — use static threshold.
            entropy_baseline_window: 0,
            entropy_trend_relative_factor: 0.0,
            entropy_trend_absolute_min_nats: 0.04,
            // Recent-coherent veto disabled by default.
            entropy_trend_recent_veto_window: 0,
            entropy_trend_recent_veto_factor: 0.0,
            // Structural token IDs: empty by default, populated by resolve_structural_tokens.
            structural_token_ids: Vec::new(),
        }
    }
}

impl DecodeHealthConfig {
    /// Preset for interactive chat: all checks enabled with sensible defaults.
    pub fn for_chat() -> Self {
        Self {
            enabled: true,
            // 8 consecutive steps below the hard floor required before abort.
            // 8 × 16 = 128 steps of the same non-structural semantic token locked
            // below 0.3 nats is a genuine single-token attractor.  Structural tokens
            // (Ġ`, Ġ-, Ċ, …) are already exempt via structural_token_ids so this
            // purely catches non-formatting token lock-on.
            entropy_hard_min_consec: 8,
            // Soft trend threshold: 0.5 nats (≈ 1.65 effective tokens).
            // Requires the model to be nearly deterministic on ALL window samples —
            // a genuine attractor, not contextually confident focused writing.
            entropy_trend_threshold_nats: 0.5,
            // 6 consecutive interval checks all below 0.2 nats: catches multi-token
            // cycling loops where no single token locks on but every interval
            // distribution is near-zero entropy. 6 × 16 steps = 96 steps of cycling.
            entropy_interval_floor_consec: 6,
            // Argmax dominance: fire when one token wins ≥ 75% of the last 8
            // interval checks (= 6/8). Non-structural tokens only.
            interval_argmax_dominance_fraction: 0.75,
            // Raise phrase-loop max period to 64 (default is 8) to catch long think-block
            // reasoning phrases that loop back on themselves (e.g. 20–40 token repeating
            // sentences).  The check is O(max_period²) per step ≈ 2048 comparisons at
            // period=64 — negligible CPU cost.
            // window must be ≥ max_period × min_reps (64×2=128); 192 gives slack.
            phrase_loop_max_period: 64,
            repetition_window: 192,
            // Adaptive baseline: collect 30 interval samples (480 steps at interval=16)
            // before activating the trend check.  Effective threshold becomes
            // max(0.04, baseline_mean × 0.25), capped at entropy_trend_threshold_nats.
            // This prevents false positives on models with naturally sharp distributions
            // (e.g. Qwen3-MoE at mean ~0.3 nats) while still catching real collapse
            // (which drives entropy far below 25 % of the session average).
            entropy_baseline_window: 30,
            entropy_trend_relative_factor: 0.25,
            entropy_trend_absolute_min_nats: 0.04,
            // Recent-coherent veto: scan the last 40 log entries (covers dense-mode
            // steps written between interval checks).  Any entry with entropy above
            // 2.5× the effective threshold means the model was coherently writing
            // and the trend window is a sampling artefact — suppress the abort.
            // At interval=16, veto_window=40 covers up to 640 steps of lookback.
            // Factor 2.5 (down from 4.0) makes the veto easier to trigger so that
            // bursts of confident structured output (code, tables) don't false-abort.
            entropy_trend_recent_veto_window: 40,
            entropy_trend_recent_veto_factor: 2.5,
            ..Self::default()
        }
    }

    /// Resolve structural formatting token IDs from the tokenizer.
    ///
    /// Structural tokens are characters that are legitimately near-deterministic
    /// in context (newline, space, tab, markdown punctuation) and should not
    /// contribute to the hard-floor or interval-floor collapse counters.
    /// Repetition checks still fire normally on these tokens.
    ///
    /// Covers both plain ASCII forms and GPT-2-style byte-level BPE encodings
    /// where whitespace and space-prefixed punctuation appear as distinct vocab
    /// entries (e.g. Ġ = U+0120 for space, Ċ = U+010A for newline).
    ///
    /// Called automatically by the engine builder after tokenizer loading.
    pub fn resolve_structural_tokens(&mut self, tokenizer: &tokenizers::Tokenizer) {
        self.structural_token_ids.clear();
        let mut add = |s: &str| {
            if let Some(id) = tokenizer.token_to_id(s) {
                if !self.structural_token_ids.contains(&id) {
                    self.structural_token_ids.push(id);
                }
            }
        };

        // Plain ASCII forms (some tokenizers use these directly).
        for s in ["\n", " ", "\t", "*", "-", "#", "`", "|", ">", "\\", "/"] {
            add(s);
        }

        // GPT-2 byte-level BPE encodes 0x0A→Ċ (U+010A), 0x09→ĉ (U+0109),
        // 0x20→Ġ (U+0120).  Qwen3 and most Llama/Mistral tokenizers use these.
        // Also include double-newline (ĊĊ) which is a common merged token.
        for s in ["\u{010A}", "\u{0109}", "\u{0120}", "\u{010A}\u{010A}"] {
            add(s);
        }

        // Space-prefixed punctuation: Ġ + char.  In GPT-2 BPE a token at the
        // start of a word carries the preceding space merged in (e.g. Ġ` = token
        // 1565 in Qwen3-8B, generated when the model opens a code fence).
        let gpt2_space = '\u{0120}';
        for c in ['*', '-', '#', '`', '|', '>', '\\', '/'] {
            let s = format!("{}{}", gpt2_space, c);
            add(&s);
        }

        tracing::trace!(
            "Resolved {} structural token IDs: {:?}",
            self.structural_token_ids.len(),
            self.structural_token_ids
        );
    }
}
