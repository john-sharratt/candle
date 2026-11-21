//! The per-decode state the health checks read and update, and the log they keep.
//!
//! A `DecodeHealthState` belongs to one decode turn and holds the sliding token
//! window, the entropy and magnitude windows, the consecutive-run counters, and a
//! capped diagnostic log. Entropy checks are skipped for greedy decoding, where a
//! near-deterministic distribution is expected, and they are suppressed inside a
//! reasoning block. The adaptive baseline is collected over the first interval
//! samples and then fixed, so it does not drift while a collapse is under way.

use super::*;

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
    pub(super) log_capacity: usize,
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
    pub(super) entropy_baseline_samples: VecDeque<f32>,
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
