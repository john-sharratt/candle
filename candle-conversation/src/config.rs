use std::collections::HashMap;

use candle_nn::kv_cache::{KvFormat, QuantFormat};
use candle_transformers::models::batched_inference::BatchedConfig;

use crate::models::Dialect;
use crate::projection::{CorruptTurnPolicy, LayerId};
use crate::token_buffer::TokenBuffer;
use crate::tree::ConversationTreeConfig;

// ────────────────────────────────────────────────────────────────────────────
// Sampling Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Comprehensive sampling configuration for token generation.
///
/// Exposes all capabilities of the batched sampling CUDA kernel:
/// - Temperature scaling
/// - Top-K (radix select, O(n) complexity)
/// - Top-P / nucleus sampling
/// - Repeat penalty (penalize recently generated tokens)
/// - Frequency penalty (penalize based on token counts)
/// - Presence penalty (penalize any repeated token equally)
/// - DRY penalty (Don't Repeat Yourself - penalize n-gram repetitions)
/// - EOS boost (encourage/discourage end-of-sequence)
/// - Banned tokens (hard constraints)
/// - Stencil constraints (constrained decoding to allowed token set)
///
/// # Example
///
/// ```
/// use candle_conversation::SamplingConfig;
///
/// // Creative writing with diverse output
/// let config = SamplingConfig::default()
///     .with_temperature(0.9)
///     .with_top_p(0.95)
///     .with_repeat_penalty(1.1);
///
/// // Precise, deterministic output
/// let config = SamplingConfig::argmax();
///
/// // Anti-repetition focused
/// let config = SamplingConfig::default()
///     .with_temperature(0.7)
///     .with_dry_penalty(0.8, 1.75, 2, 256);
/// ```
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    // ── Core Sampling ──────────────────────────────────────────────────
    /// Sampling temperature. `0.0` = argmax/greedy decoding.
    /// Higher values increase randomness. Typical range: 0.0 - 2.0.
    pub temperature: f32,

    /// Added to `temperature` for sequences currently inside a segment;
    /// `0.0` = disabled. Lets the in-segment span sample a touch hotter
    /// without affecting tokens outside the segment.
    pub segment_temp_boost: f32,

    /// Token IDs whose logits are suppressed while a sequence is inside a
    /// segment.  Shared across the batch; resolved once from the caller at
    /// engine start.  Empty = disabled.
    pub segment_suppress_tokens: Vec<i32>,

    /// Penalty subtracted from each [`Self::segment_suppress_tokens`] logit while
    /// inside a segment — the *ceiling* lever.  Set per turn by the caller: a
    /// large value HARD-suppresses the token family, a moderate value
    /// SOFT-discourages it, `0.0` = disabled.  Only applied while `in_segment`,
    /// so tokens outside the segment are untouched.
    pub segment_suppress_penalty: f32,

    /// Top-K sampling: only consider the K most likely tokens.
    /// `0` = disabled. Uses O(n) radix select algorithm.
    pub top_k: i32,

    /// Top-P / nucleus sampling: sample from smallest set of tokens
    /// whose cumulative probability exceeds P.
    /// `1.0` = disabled. Typical range: 0.9 - 0.99.
    pub top_p: f32,

    // ── Repetition Penalties ───────────────────────────────────────────
    /// Repeat penalty: multiplicative penalty for tokens in recent history.
    /// `1.0` = disabled. Typical range: 1.0 - 1.5.
    /// Applied as: `logit /= penalty` for positive logits, `logit *= penalty` for negative.
    pub repeat_penalty: f32,

    /// Frequency penalty: subtractive penalty based on token occurrence count.
    /// `0.0` = disabled. Typical range: 0.0 - 2.0.
    /// Applied as: `logit -= frequency_penalty * count`.
    pub frequency_penalty: f32,

    /// Presence penalty: subtractive penalty for any token that has appeared.
    /// `0.0` = disabled. Typical range: 0.0 - 2.0.
    /// Applied as: `logit -= presence_penalty` if count > 0.
    pub presence_penalty: f32,

    // ── DRY Penalty (Don't Repeat Yourself) ────────────────────────────
    /// DRY penalty configuration. Detects and penalizes n-gram repetitions.
    /// Set to `None` to disable.
    pub dry: Option<DryConfig>,

    // ── Repeat window ──────────────────────────────────────────────────
    /// Window size for repeat/frequency/DRY penalties.
    /// Only the most recent N tokens are considered.
    /// `0` = use full history (no windowing).
    pub repeat_last_n: i32,

    // ── Cross-turn penalty ─────────────────────────────────────────────
    /// Additive penalty subtracted from tokens that appeared in **prior turns**.
    /// Lighter than `presence_penalty` — useful for suppressing repeated greetings
    /// without harming in-turn generation quality.
    /// `0.0` = disabled.
    pub cross_turn_penalty: f32,

    // ── EOS Control ────────────────────────────────────────────────────
    /// Additive boost to the EOS token logit.
    /// Positive values encourage stopping; negative values discourage.
    /// `0.0` = disabled.
    pub eos_boost: f32,

    /// Enable dynamic EOS boost that ramps up over the response length.
    /// When `true`, the effective boost = `eos_boost * min((current_len - eos_ramp_start) / (eos_ramp_len - eos_ramp_start), 1.0) * eos_boost_max_multiplier`.
    /// Keeps the model from stopping too early while reliably stopping by `eos_ramp_len`.
    pub dynamic_eos_boost: bool,

    /// Token count where the dynamic EOS boost ramp begins.
    /// Below this length the boost is zero. Typically set to 80% of `eos_ramp_len`.
    /// `0` = ramp starts from the beginning. Ignored when `dynamic_eos_boost` is `false`.
    pub eos_ramp_start: i32,

    /// Length (in tokens) where the dynamic EOS boost reaches full strength.
    /// Ignored when `dynamic_eos_boost` is `false`. `0` disables the ramp (instant full boost).
    pub eos_ramp_len: i32,

    /// Multiplier applied to `eos_boost` at the end of the ramp.
    /// Values > 1.0 make the terminal boost stronger than the base `eos_boost`.
    /// `1.0` = no extra multiplier. Ignored when `dynamic_eos_boost` is `false`.
    pub eos_boost_max_multiplier: f32,

    // ── Segment-Close Boost ─────────────────────────────────────────────
    // Ramps up a boost on the segment-close token while a segment is open so
    // the model closes the segment within its budget.  Only active while the
    // sampler is inside a segment.
    /// Additive boost applied to the segment-close token logit.
    /// Uses the same ramp formula as EOS boost but keyed on the per-segment
    /// token count instead of total generation length.
    /// `0.0` = disabled.
    pub segment_close_boost: f32,

    /// Token ID of the segment-close token.
    /// Resolved and supplied by the caller.
    /// `-1` = disabled (no segment tracking).
    pub segment_close_token_id: i32,

    /// Token ID of the segment-open token.
    /// Resolved and supplied by the caller.
    /// `-1` = disabled.
    pub segment_open_token_id: i32,

    /// Per-segment token count where the segment-close ramp begins (e.g. 150).
    pub segment_close_ramp_start: i32,

    /// Per-segment token count where the segment-close ramp reaches full strength (e.g. 200).
    pub segment_close_ramp_len: i32,

    /// Multiplier applied to `segment_close_boost` at the end of the ramp.
    pub segment_close_max_multiplier: f32,

    // ── Generation Failsafes ────────────────────────────────────────────
    /// After this many in-segment tokens, unconditionally force the segment-close token.
    /// Hard failsafe that guarantees the segment closes even if the model would
    /// otherwise keep generating inside it indefinitely.
    ///
    /// Operates on `segment_len` (tokens since the segment opened), not total
    /// generated tokens.  `0` = disabled.  The segment-close token ID must also be
    /// resolved.  No-op when no segment is open.
    pub force_segment_close_after: i32,

    /// After this many in-segment tokens, emit the segment-close token at the next
    /// sentence boundary (`.`, `!`, `?`, or `\n`).  Softer than `force_segment_close_after`:
    /// the model finishes its current sentence before the segment closes.
    /// Operates on `segment_len`.  `0` = disabled.  Pair with `force_segment_close_after`
    /// as the hard backstop.
    pub graceful_segment_close_after: i32,

    /// Token IDs that count as sentence-end boundaries for `graceful_segment_close_after`.
    ///
    /// Resolved automatically from the tokenizer at engine startup (looks up
    /// `.`, `!`, `?`, `\n`).  If empty, `graceful_segment_close_after` is a no-op.
    pub sentence_end_token_ids: Vec<i32>,

    /// Closer phrase played when the HARD segment cap
    /// (`force_segment_close_after`) fires mid-sentence: a canned
    /// self-interruption (e.g. `" — actually, I've reasoned enough and know
    /// what to do."`), emitted one token per decode step; the sampler then
    /// emits the segment-close token itself.  Turns a mid-sentence amputation
    /// into sensible prose and primes the answer with an explicit commitment
    /// frame.  Never played at a completed sentence (graceful closes, or a
    /// hard cap that happens to land on a sentence boundary — no rescue
    /// needed) nor in a steering span that would continue reasoning after the
    /// close. Empty = the hard cap emits the bare close token.
    pub segment_close_script: Vec<u32>,

    /// After this many generated tokens, wait for the next sentence-ending
    /// token (`.`, `!`, `?`, or `\n` — resolved from the tokenizer at engine
    /// startup) and then emit EOS.  Mirrors the `graceful_segment_close_after` mechanism
    /// for closing a segment: the current sentence is allowed to complete before
    /// generation stops, preventing mid-sentence truncation.
    ///
    /// If no sentence-boundary token occurs before `forced_eos_after`, the hard
    /// backstop ensures termination regardless.  `0` = disabled.
    pub graceful_eos_after: i32,

    /// After this many generated tokens, unconditionally force EOS.
    /// Hard failsafe — guarantees termination regardless of model state.
    /// `0` = disabled. Should be > `graceful_eos_after` when both are set.
    pub forced_eos_after: i32,

    // ── Hard Constraints ───────────────────────────────────────────────
    /// Token IDs that are never allowed to be sampled.
    /// These tokens receive `-inf` logit penalty.
    pub banned_tokens: Vec<i32>,

    /// Stencil constraint: if non-empty, ONLY these tokens can be sampled.
    /// Used for constrained decoding (e.g., JSON schema, grammar).
    /// Empty = no constraint.
    pub stencil: Vec<i32>,

    // ── RNG ────────────────────────────────────────────────────────────
    /// RNG seed for reproducible sampling.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            segment_temp_boost: 0.0,
            segment_suppress_tokens: Vec::new(),
            segment_suppress_penalty: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry: None,
            repeat_last_n: 0,
            cross_turn_penalty: 0.0,
            eos_boost: 0.0,
            dynamic_eos_boost: false,
            eos_ramp_start: 0,
            eos_ramp_len: 0,
            eos_boost_max_multiplier: 1.0,
            segment_close_boost: 0.0,
            segment_close_token_id: -1,
            segment_open_token_id: -1,
            segment_close_ramp_start: 0,
            segment_close_ramp_len: 0,
            segment_close_max_multiplier: 1.0,
            force_segment_close_after: 0,
            graceful_segment_close_after: 0,
            sentence_end_token_ids: Vec::new(),
            segment_close_script: Vec::new(),
            graceful_eos_after: 0,
            forced_eos_after: 0,
            banned_tokens: Vec::new(),
            stencil: Vec::new(),
            seed: 42,
        }
    }
}

impl SamplingConfig {
    /// Greedy/argmax decoding (temperature = 0).
    pub fn argmax() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Decode config for the summary compressor. Low-temperature nucleus
    /// sampling (temperature 0.7, top-k 40, top-p 0.95) with a repetition
    /// penalty.
    ///
    /// Sampling — not pure argmax — is deliberate. Under argmax the summary
    /// locks DETERMINISTICALLY onto a single continuation, and the `repeat_penalty`
    /// then pushes that greedy pick *away* from the tokens already in context;
    /// when the attended context is even mildly off-distribution the penalised
    /// argmax lands on degenerate loops or an off-language drift (whole summaries
    /// decoding to another language), with no way to recover since the path is
    /// deterministic. Sampling from a constrained top-k/top-p nucleus keeps the
    /// decode on the plausible-English probability mass and shrugs off that mild
    /// contamination instead of committing to it.
    ///
    /// The `repeat_penalty` still breaks the occasional loop sampling can fall
    /// into (`"valid, valid, firm, firm, …"`), while leaving legitimate
    /// repetition (a repo map listing `candle-core, candle-nn, …`) intact —
    /// unlike cumulative frequency/presence penalties, which suppress such lists.
    /// Brevity comes from the per-layer `summary` prompts, not the penalty.
    /// Tuned via the `regen_summaries` example.
    pub fn compression() -> Self {
        Self::top_k(40, 0.7)
            .with_top_p(0.95)
            .with_repeat_penalty(1.3)
    }

    /// Top-K sampling with temperature.
    pub fn top_k(k: i32, temperature: f32) -> Self {
        Self {
            temperature,
            top_k: k,
            ..Default::default()
        }
    }

    /// Top-P (nucleus) sampling with temperature.
    pub fn top_p(p: f32, temperature: f32) -> Self {
        Self {
            temperature,
            top_p: p,
            ..Default::default()
        }
    }

    /// Top-K then Top-P sampling.
    pub fn top_k_top_p(k: i32, p: f32, temperature: f32) -> Self {
        Self {
            temperature,
            top_k: k,
            top_p: p,
            ..Default::default()
        }
    }

    /// Recommended sampling defaults based on a GGUF `general.architecture` string.
    ///
    /// Each model family publishes a `generation_config.json` on HuggingFace.
    /// This function returns the official defaults for known architectures,
    /// or conservative fallback defaults for unrecognised ones.
    ///
    /// Sources:
    /// - Qwen3 family: <https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/generation_config.json>
    /// - Qwen2 family: <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/raw/main/generation_config.json>
    /// - Llama/Hermes: <https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/raw/main/generation_config.json>
    pub fn for_gguf_architecture(arch: &str) -> Self {
        match arch {
            // All Qwen3 models — dense and MoE — share the same official config.
            // The MoE variant reports as "qwen2moe" in GGUF metadata.
            // EOT boost ramp params are set here; the actual <think>/<​/think>
            // token IDs are resolved automatically from the tokenizer in
            // `resolve_thinking_tokens()` during engine startup.
            "qwen3" | "qwen3moe" | "qwen2moe" => Self::top_k_top_p(40, 0.95, 0.8)
                // Matched to the LM Studio reference run: temp=0.8, top_k=40,
                // top_p=0.95, repeat_penalty=1.1.  A gentle multiplicative
                // repeat_penalty applies batch-wide.
                //
                // DRY is span-scoped: the kernel gates and windows it on
                // `dry_lens[seq]` — the current structural span (reset at
                // `<think>`/`</think>`/`<tool_call>`/`</tool_call>`, off inside
                // tool calls).  So it runs in both thinking AND the answer but
                // only ever sees the current span's own tokens.  That is what
                // makes it safe on the answer: it breaks a repeating loop without
                // penalizing verbatim reproduction of numbers, identifiers, or
                // code lifted from the prompt or an earlier span — those live
                // outside the span DRY can see.  The thinking-only temperature
                // boost lets reasoning sample a touch hotter while the answer
                // stays at the reference temp.
                .with_segment_temp_boost(0.05)
                .with_dry_penalty(0.8, 1.75, 2, 512)
                .with_repeat_penalty(1.1)
                .with_repeat_last_n(128)
                // EOT ramp: nudge </think> after 200 thinking tokens, full boost by 400
                // (segment_close_ramp_len is the ramp's absolute end, not a span).  zend overrides
                // segment_close_ramp_start/len per turn from `ThinkMode::eot_budget()`; this is the
                // fallback for non-steered callers.  These IDs are resolved from the
                // tokenizer at engine startup.
                .with_segment_close_boost(2.0, 200, 400, 5.0)
                .with_graceful_segment_close_after(220)
                .with_force_segment_close_after(300)
                // EOS limits are in total generated tokens (thinking + response).
                // Think block consumes ~200-300 tokens; leave ~500-700 for the response.
                // EOS boost ramp starts nudging at 700 total tokens, full boost by 800
                // (eos_ramp_len is the ramp's absolute end, not a span).
                // graceful_eos fires at the next sentence boundary after 800 tokens;
                // forced_eos fires unconditionally at 1000 tokens.
                // non_thinking_for_gguf_architecture overrides these back to tighter limits.
                // zend overrides all four per turn from `ThinkMode::eos_budget()`; this
                // is the fallback for non-steered callers.
                .with_dynamic_eos_boost(1.0, 700, 800, 3.0)
                .with_eos_failsafe(800, 1000),

            // Qwen2 instruct models.
            "qwen2" => Self::top_k_top_p(20, 0.8, 0.7).with_repeat_penalty(1.1),

            // Llama-family (Hermes, etc.) — top-p only, no top-k.
            "llama" => Self::top_p(0.9, 0.6),

            // Unknown architecture — conservative defaults.
            _ => Self::top_p(0.9, 0.7),
        }
    }

    /// Non-thinking sampling defaults based on a GGUF `general.architecture` string.
    ///
    /// Returns `Some` for model families that support thinking mode (Qwen3),
    /// with the official non-thinking sampling parameters.
    /// Returns `None` for models that don't support thinking.
    pub fn non_thinking_for_gguf_architecture(arch: &str) -> Option<Self> {
        match arch {
            // Qwen3 family non-thinking: matched exactly to LM Studio reference run
            // (qwen3-30b-a3b-abliterated-untied-i1 @ Q4_K_M, 2026-02-20).
            // LM Studio params: temp=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1, min_p=0.05.
            // min_p is not yet implemented in this sampler; all other params match.
            // Qwen3 non-thinking: no think block overhead, so all tokens are
            // response content.  EOS boost ramp starts at 400 tokens, full by 500.
            // graceful_eos fires at next sentence boundary after 512 tokens;
            // forced_eos fires unconditionally at 700 tokens.
            "qwen3" | "qwen3moe" | "qwen2moe" => Some(
                Self::for_gguf_architecture(arch)
                    .with_no_segment_close()
                    .with_dynamic_eos_boost(1.0, 400, 100, 3.0)
                    .with_eos_failsafe(512, 700),
            ),

            // Other model families don't have separate non-thinking params.
            _ => None,
        }
    }

    /// Named sampling presets that can be selected via `--sampler <name>`.
    ///
    /// These override the auto-detected architecture defaults. Use
    /// [`preset_names()`](Self::preset_names) to list available presets.
    ///
    /// Returns `None` if `name` is not recognized.
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            // Relaxed: high temperature, minimal filtering, light repeat penalty.
            // Good baseline for models that handle their own stopping well.
            "relaxed" => Some(Self {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.03,
                ..Default::default()
            }),

            // Creative: high diversity, light penalties to curb loops.
            "creative" => Some(
                Self::top_p(0.95, 0.9)
                    .with_repeat_penalty(1.05)
                    .with_repeat_last_n(128)
                    .with_dynamic_eos_boost(1.0, 400, 500, 3.0)
                    .with_eos_failsafe(550, 600),
            ),

            // Precise: low temperature, tight filtering, deterministic.
            "precise" => Some(
                Self::top_k_top_p(20, 0.8, 0.3)
                    .with_repeat_penalty(1.1)
                    .with_repeat_last_n(64),
            ),

            // Anti-repetition: aggressive penalties for verbose models.
            "antirep" => Some(
                Self::top_k_top_p(40, 0.95, 0.7)
                    .with_repeat_penalty(1.15)
                    .with_repeat_last_n(256)
                    .with_presence_penalty(0.6)
                    .with_dry_penalty(0.8, 1.75, 2, 512)
                    .with_dynamic_eos_boost(1.0, 300, 400, 3.0)
                    .with_eos_failsafe(450, 500),
            ),

            _ => None,
        }
    }

    /// List available preset names for `--sampler`.
    pub fn preset_names() -> &'static [&'static str] {
        &["relaxed", "creative", "precise", "antirep"]
    }

    // ── Builder Methods ────────────────────────────────────────────────

    /// Set temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set the in-segment temperature boost (added to `temperature` while a
    /// sequence is inside a segment). `0.0` disables it.
    pub fn with_segment_temp_boost(mut self, v: f32) -> Self {
        self.segment_temp_boost = v;
        self
    }

    /// Set the in-segment token suppression: the token list and the per-turn
    /// penalty (large = HARD, moderate = SOFT, `0.0` = off).
    pub fn with_segment_suppression(mut self, tokens: Vec<i32>, penalty: f32) -> Self {
        self.segment_suppress_tokens = tokens;
        self.segment_suppress_penalty = penalty;
        self
    }

    /// Set top-K.
    pub fn with_top_k(mut self, k: i32) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-P.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set repeat penalty.
    pub fn with_repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = penalty;
        self
    }

    /// Set frequency penalty.
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty.
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Enable DRY penalty.
    ///
    /// # Arguments
    /// - `multiplier`: Base penalty strength (0.0 = disabled). Typical: 0.5 - 1.0.
    /// - `base`: Exponential base for longer matches. Typical: 1.5 - 2.0.
    /// - `allowed_length`: N-grams up to this length are not penalized. Typical: 1 - 3.
    /// - `range`: How far back to look for matches (0 = full history). Typical: 256 - 512.
    pub fn with_dry_penalty(
        mut self,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        range: i32,
    ) -> Self {
        self.dry = Some(DryConfig {
            multiplier,
            base,
            allowed_length,
            range,
        });
        self
    }

    /// Set the repeat window (last N tokens considered for penalties).
    /// `0` = use full history.
    pub fn with_repeat_last_n(mut self, n: i32) -> Self {
        self.repeat_last_n = n;
        self
    }

    /// Set cross-turn penalty (penalizes tokens seen in prior turns).
    pub fn with_cross_turn_penalty(mut self, penalty: f32) -> Self {
        self.cross_turn_penalty = penalty;
        self
    }

    /// Set EOS boost.
    pub fn with_eos_boost(mut self, boost: f32) -> Self {
        self.eos_boost = boost;
        self
    }

    /// Enable dynamic EOS boost with ramp parameters.
    /// `boost` is the base boost; at `ramp_len` tokens it is multiplied by `max_multiplier`.
    /// `ramp_start` is the token count where the ramp begins — zero boost before that.
    pub fn with_dynamic_eos_boost(
        mut self,
        boost: f32,
        ramp_start: i32,
        ramp_len: i32,
        max_multiplier: f32,
    ) -> Self {
        self.eos_boost = boost;
        self.dynamic_eos_boost = true;
        self.eos_ramp_start = ramp_start;
        self.eos_ramp_len = ramp_len;
        self.eos_boost_max_multiplier = max_multiplier;
        self
    }

    /// Enable the segment-close boost.
    ///
    /// While a segment is open the sampler ramps up a boost on the
    /// segment-close token so the segment closes within its budget.
    ///
    /// The segment-open / segment-close token IDs are supplied separately by
    /// the caller — this setter only configures the ramp parameters.
    ///
    /// * `boost`          – base additive boost
    /// * `ramp_start`     – per-segment token count where ramp begins (e.g. 150)
    /// * `ramp_len`       – per-segment token count where ramp reaches full (e.g. 200)
    /// * `max_multiplier` – multiplier at full ramp
    pub fn with_segment_close_boost(
        mut self,
        boost: f32,
        ramp_start: i32,
        ramp_len: i32,
        max_multiplier: f32,
    ) -> Self {
        self.segment_close_boost = boost;
        self.segment_close_ramp_start = ramp_start;
        self.segment_close_ramp_len = ramp_len;
        self.segment_close_max_multiplier = max_multiplier;
        self
    }

    /// Disable the segment-close boost entirely.
    ///
    /// Clears all segment-close ramp parameters so the segment-close token
    /// receives no additive boost.  Use this when building configs on top of a
    /// base that already has a segment-close ramp configured (e.g. chaining on
    /// `for_gguf_architecture`).
    pub fn with_no_segment_close(mut self) -> Self {
        self.segment_close_boost = 0.0;
        self.segment_close_ramp_start = 0;
        self.segment_close_ramp_len = 0;
        self.segment_close_max_multiplier = 1.0;
        self.force_segment_close_after = 0;
        self.graceful_segment_close_after = 0;
        self
    }

    /// Resolve `<think>` / `</think>` token IDs from the tokenizer.
    ///
    /// Called automatically by the engine builder after loading the
    /// tokenizer.  If the tokens are not found, EOT boost is silently
    /// disabled (token IDs remain `-1`).
    pub fn resolve_thinking_tokens(&mut self, tokenizer: &tokenizers::Tokenizer) {
        if let Some(id) = tokenizer.token_to_id("</think>") {
            self.segment_close_token_id = id as i32;
            tracing::trace!("Resolved </think> token ID: {}", id);
        }
        if let Some(id) = tokenizer.token_to_id("<think>") {
            self.segment_open_token_id = id as i32;
            tracing::trace!("Resolved <think> token ID: {}", id);
        }
        // Resolve sentence-end token IDs for graceful_segment_close_after.
        // We probe both the bare character and common BPE compound forms.
        self.sentence_end_token_ids.clear();
        for boundary in [".", "\n", ".\n", "!\n", "?\n", "!", "?"] {
            if let Some(id) = tokenizer.token_to_id(boundary) {
                let id = id as i32;
                if !self.sentence_end_token_ids.contains(&id) {
                    self.sentence_end_token_ids.push(id);
                }
            }
        }
        tracing::trace!(
            "Resolved {} sentence-end token IDs: {:?}",
            self.sentence_end_token_ids.len(),
            self.sentence_end_token_ids
        );

        // Resolve the reflection-marker family (`Wait`/`Hmm`/`Alternatively`/
        // `Actually`) — the overthinking drivers whose logits the steering
        // suppresses while in a `<think>` block.  Each keyword is expanded to all
        // single-token variants (bare / leading-space / lower / upper) so the bias
        // can't leak through a capitalised or space-prefixed form; multi-token
        // markers (e.g. "hold on", "let me check") can't be logit-suppressed and
        // are skipped.  The per-turn penalty is set from the effort dial elsewhere.
        self.segment_suppress_tokens.clear();
        for keyword in ["Wait", "Hmm", "Alternatively", "Actually"] {
            for cand in [
                keyword.to_string(),
                format!(" {keyword}"),
                keyword.to_lowercase(),
                format!(" {}", keyword.to_lowercase()),
                keyword.to_uppercase(),
            ] {
                if let Ok(enc) = tokenizer.encode(cand.as_str(), false) {
                    if let [id] = enc.get_ids() {
                        let id = *id as i32;
                        if !self.segment_suppress_tokens.contains(&id) {
                            self.segment_suppress_tokens.push(id);
                        }
                    }
                }
            }
        }
        tracing::trace!(
            "Resolved {} reflection-marker suppress token IDs: {:?}",
            self.segment_suppress_tokens.len(),
            self.segment_suppress_tokens
        );
    }

    /// Apply the canonical `<think>`-block close steering for `mode` — the single
    /// place both the dialogue path (`zend` session) and the ingest scope-summary
    /// decode configure thinking behaviour, so the two can never drift.
    ///
    /// Resolves the `<think>`/`</think>` token IDs, sets the reflection-suppress
    /// penalty and (for `Off`) drops the thinking temp boost, then programs the
    /// per-span EOT close ramp + graceful/force cutoffs from `mode.eot_budget()`.
    ///
    /// `max_response_tokens` guards the short-output case. `Off`/`Quick`'s EOT
    /// budget (~220/300) is a *dialogue backstop* — it assumes the model self-closes
    /// an empty block from the `/no_think` glue and only caps a runaway. A short
    /// summary (`max_response_tokens` ≈ 100) can never reach a 300-token backstop,
    /// so when the block budget can't fit the response the steering collapses to a
    /// forced **empty** close — the budget goes to the answer, not runaway (often
    /// off-language) reasoning. Dialogue budgets sit well above the EOT force, so
    /// their steering is unchanged.
    ///
    /// The EOS (answer-length) budget and the mid-sentence closer script stay
    /// caller-set: they depend on the response-length dial and the dialect.
    pub fn apply_think_mode(
        &mut self,
        mode: crate::stencil::ThinkMode,
        tokenizer: &tokenizers::Tokenizer,
        max_response_tokens: usize,
    ) {
        self.resolve_thinking_tokens(tokenizer);
        self.segment_suppress_penalty = mode.suppress_penalty();
        if mode == crate::stencil::ThinkMode::Off {
            self.segment_temp_boost = 0.0;
        }
        self.set_think_close_budget(mode, max_response_tokens);
    }

    /// The tokenizer-independent half of [`Self::apply_think_mode`]: program the
    /// per-span EOT close ramp + graceful/force cutoffs from `mode.eot_budget()`,
    /// with the short-output collapse. Split out so the budget logic is unit-
    /// testable without a tokenizer.
    fn set_think_close_budget(
        &mut self,
        mode: crate::stencil::ThinkMode,
        max_response_tokens: usize,
    ) {
        let (graceful, force) = mode.eot_budget();
        // Collapse to a forced empty block ONLY for `Off` (which wants no thinking
        // at all) when its backstop can't fire within the budget. Never collapse a
        // reasoning dial: `Deep`/`Exhaustive` have large per-span think budgets that
        // are deliberately independent of the answer-length budget, and their spans
        // restart, so a small `max_response_tokens` must not empty their reasoning.
        let collapse = mode == crate::stencil::ThinkMode::Off
            && (force.max(0) as usize) >= max_response_tokens.max(1);
        if collapse {
            self.graceful_segment_close_after = 0;
            self.force_segment_close_after = 1;
            self.segment_close_ramp_start = 0;
            self.segment_close_ramp_len = 1;
        } else {
            self.graceful_segment_close_after = graceful;
            self.force_segment_close_after = force;
            self.segment_close_ramp_start = graceful;
            self.segment_close_ramp_len = force;
        }
    }

    /// Set a hard per-segment token limit after which the segment-close token is forced.
    ///
    /// When `segment_len >= n`, the next sampled token is replaced with the
    /// segment-close token regardless of the model's distribution.  This is the
    /// reliable way to guarantee the segment closes within a budget,
    /// complementing the softer `segment_close_boost` ramp.
    /// `0` = disabled.
    pub fn with_force_segment_close_after(mut self, n: i32) -> Self {
        self.force_segment_close_after = n;
        self
    }

    /// Set a soft per-segment token limit: emit the segment-close token at the next sentence boundary.
    ///
    /// Once `segment_len >= n`, the *following* step after a sentence-ending token
    /// (`.`, `!`, `?`, or `\n` — resolved from the tokenizer) forces the segment-close token.
    /// This lets the current sentence complete naturally before the segment closes.
    ///
    /// Pair with `with_force_segment_close_after` as the hard backstop (e.g. `graceful=220,
    /// force=300`) so the segment always closes within budget even if the model avoids
    /// sentence-ending tokens.  `0` = disabled.
    pub fn with_graceful_segment_close_after(mut self, n: i32) -> Self {
        self.graceful_segment_close_after = n;
        self
    }

    /// Set EOS failsafes for graceful and forced termination.
    ///
    /// After `graceful` tokens, the next `.` or `\n` triggers EOS.
    /// After `forced` tokens, EOS is emitted unconditionally.
    pub fn with_eos_failsafe(mut self, graceful: i32, forced: i32) -> Self {
        self.graceful_eos_after = graceful;
        self.forced_eos_after = forced;
        self
    }

    /// Scale all EOS-related limits proportionally to `max_response_tokens`.
    ///
    /// The existing `forced_eos_after` value is treated as the implicit default
    /// budget the config was tuned for.  All EOS fields (graceful, forced, ramp
    /// start, ramp length) are scaled by `max_response_tokens / forced_eos_after`
    /// so the shape of the curve is preserved at any budget.  Segment-close
    /// limits are absolute and are not changed.  No-ops when `forced_eos_after`
    /// is zero (EOS failsafe disabled).
    pub fn with_max_response_tokens(mut self, max_response_tokens: usize) -> Self {
        if self.forced_eos_after <= 0 {
            return self;
        }
        let scale = max_response_tokens as f32 / self.forced_eos_after as f32;
        self.forced_eos_after = max_response_tokens as i32;
        self.graceful_eos_after = ((self.graceful_eos_after as f32 * scale) as i32).max(1);
        if self.dynamic_eos_boost {
            self.eos_ramp_start = ((self.eos_ramp_start as f32 * scale) as i32).max(0);
            self.eos_ramp_len = ((self.eos_ramp_len as f32 * scale) as i32).max(1);
        }
        self
    }

    /// Add banned tokens.
    pub fn with_banned_tokens(mut self, tokens: Vec<i32>) -> Self {
        self.banned_tokens = tokens;
        self
    }

    /// Set stencil constraint (allowed tokens).
    pub fn with_stencil(mut self, tokens: Vec<i32>) -> Self {
        self.stencil = tokens;
        self
    }

    /// Set RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    // ── Query Methods ──────────────────────────────────────────────────

    /// Returns `true` if this config uses argmax (greedy) decoding.
    pub fn is_argmax(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Returns `true` if any penalty features are enabled.
    pub fn has_penalties(&self) -> bool {
        self.repeat_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
            || self.cross_turn_penalty != 0.0
    }

    /// Returns `true` if DRY penalty is enabled.
    pub fn has_dry(&self) -> bool {
        self.dry.as_ref().is_some_and(|d| d.multiplier != 0.0)
    }

    /// Returns `true` if stencil constraint is active.
    pub fn has_stencil(&self) -> bool {
        !self.stencil.is_empty()
    }
}

/// DRY (Don't Repeat Yourself) penalty configuration.
///
/// Detects when the model is about to continue a repeated n-gram pattern
/// and applies exponential penalty based on match length.
///
/// Example: If the context contains "the cat sat on the mat" and later
/// "the cat", the DRY penalty will penalize "sat" to prevent repetition.
#[derive(Debug, Clone)]
pub struct DryConfig {
    /// Base penalty multiplier. `0.0` = disabled.
    /// Applied as: `penalty = multiplier * base^(match_length - allowed_length)`
    pub multiplier: f32,

    /// Exponential base for longer matches. Typical: 1.5 - 2.0.
    /// Higher values penalize longer matches more aggressively.
    pub base: f32,

    /// N-grams up to this length are allowed without penalty.
    /// Typical: 1-3. Allows common short phrases without penalty.
    pub allowed_length: i32,

    /// How far back to search for matching patterns (in tokens).
    /// `0` = search full history. Typical: 256 - 512 tokens.
    pub range: i32,
}

impl Default for DryConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
            range: 256,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Decode Health Configuration
// ────────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────────
// Engine Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Engine-wide configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Configuration for the batched inference session (chunk size, arena size, KV format).
    pub batched_config: BatchedConfig,

    /// Maximum number of concurrent conversations.
    pub max_concurrent_conversations: usize,

    /// EOS token ID. Generation stops when this token is produced.
    pub eos_tokens: TokenBuffer,

    /// Vocabulary size for the model. Required for penalty tracking buffers.
    pub vocab_size: usize,

    /// Maximum recent token history length for repeat/DRY penalties.
    /// Larger values allow longer-range penalty detection but use more memory.
    pub max_recent_len: usize,

    /// Scheduler-specific configuration.
    pub scheduler: SchedulerConfig,

    /// When `true`, special tokens (e.g. `<think>`, `</think>`, `<|im_end|>`)
    /// are included in streamed text output instead of being stripped.
    pub show_special_tokens: bool,

    /// Optional path to write penalty state to during decoding.
    /// Useful for debugging why tokens are being penalized.
    pub penalty_log_path: Option<std::path::PathBuf>,

    /// Decode health monitoring configuration.
    ///
    /// Checks only run when built with `--features decode-health` AND
    /// `health.enabled == true`. Safe to set unconditionally.
    pub health: DecodeHealthConfig,

    /// Workspace root whose `.substrate/` directory backs the persistence
    /// redo log. When `None`, the engine opens the substrate under the
    /// process working directory (`SubstratePersistence::open`).
    pub workspace_path: Option<std::path::PathBuf>,

    /// Serialized model identity (HF repo / filename / arch / context length).
    /// Written to the substrate's `ModelSpec` record at engine startup via
    /// compare-and-insert, so the log is a self-contained, reloadable image.
    /// `None` skips the write.
    pub model_spec: Option<Vec<u8>>,

    /// The model's `tokenizer.json` bytes. Written to the substrate's
    /// `Tokenizer` record at engine startup via compare-and-insert so the log
    /// can detokenize offline without a separate tokenizer file. `None` skips
    /// the write.
    pub tokenizer: Option<Vec<u8>>,

    /// The model's chat dialect — the structural-token strings the
    /// engine needs to know about for inter-turn boundary handling.
    /// Pre-tokenised at engine construction into the assembler's
    /// `BoundaryMarkers` so every projection can wrap each
    /// `Sealed::Turn` in a live-prefilled boundary run without
    /// re-tokenising.  Defaults to ChatML (the most common dialect
    /// in the modern model lineup the engine supports); model
    /// builders override via `ret.dialect = ...`.
    pub dialect: Dialect,

    /// When `true`, the engine does not spawn the async summariser thread and
    /// new conversations are not registered for summarisation. The AVL summary
    /// forest is left un-extended (provenance scans still work on raw turns).
    /// Off by default; set via `ModelBuilder::disable_summariser`.
    pub disable_summariser: bool,

    /// Per-layer [`CorruptTurnPolicy`] (keyed by `LayerId`), applied when a turn
    /// is unrecoverable during the startup substrate reload. Set on the substrate
    /// in `ConversationEngine::new` *before* the reload thread is spawned, so the
    /// reload sees the right policy per layer (empty ⇒ every layer defaults to
    /// `DropConversation`). Populated by `ModelBuilder::corrupt_turn_policies`
    /// from the projection schema. Empty by default.
    pub layer_corrupt_turn: HashMap<LayerId, CorruptTurnPolicy>,
}

impl EngineConfig {
    pub fn new(eos_tokens: TokenBuffer) -> Self {
        let r16 = KvFormat::Quantized(QuantFormat::R16);
        Self {
            batched_config: BatchedConfig {
                k_format: r16,
                v_format: r16,
                ..BatchedConfig::default()
            },
            max_concurrent_conversations: 4,
            eos_tokens,
            vocab_size: 128_000, // Common for modern LLMs
            // 2048 tokens ≈ 10 turns of 200-token responses; large enough for DRY and
            // cross-turn penalties to cover a full roleplay conversation window without
            // letting earlier turns fall out of the penalty buffer undetected.
            max_recent_len: 2048,
            scheduler: SchedulerConfig::default(),
            show_special_tokens: false,
            penalty_log_path: None,
            health: DecodeHealthConfig::default(),
            workspace_path: None,
            model_spec: None,
            tokenizer: None,
            dialect: Dialect::chat_ml(),
            disable_summariser: false,
            layer_corrupt_turn: HashMap::new(),
        }
    }
}

/// Scheduler-specific configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Small-prefill threshold. Reserved for the unified-wave small/large batch
    /// flip (design `docs/unified_wave_inference_engine.md`); not yet read.
    pub small_prefill_threshold: usize,

    /// **Per-forward prefill token target** (`max_prefill_pass_tokens`): the
    /// tokens one prefill forward carries. The primary knob for the expert-load
    /// amortization curve (design §1.2/§6) — a larger target streams the same
    /// all-expert weight load over more tokens. Bounded above by the model-side
    /// per-forward ceiling ([`crate::models::batched_inference`]'s
    /// `MAX_PREFILL_TOKENS`); set at the point where enough parallel work is
    /// reliably queued to fill every forward, so no forward runs starved.
    pub large_prefill_max_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            small_prefill_threshold: 128,
            // 2048 tokens/forward: with the parallel scope-ingest keeping ~24
            // scopes queued at once (`CODE_READ_PARALLELISM`), every forward
            // fills to this target — saturating ~all experts by the
            // coupon-collector bound — instead of stalling on a 4096-token
            // target that the reduced-scope scope KV rarely reaches. ≤ the
            // model's `MAX_PREFILL_TOKENS` ceiling.
            large_prefill_max_tokens: 2048,
        }
    }
}

/// Pick a reasonable `max_hot_turns` ceiling from arena dimensions and
/// expected turn length.
///
/// Derives how many completed turns worth of KV data fit in the paged
/// arena before Hot → Warm eviction should fire, using the formula from
/// the sealed-sequence-tiers design:
///
/// ```text
/// hot_turns = (arena_chunks × chunk_size) / tokens_per_turn
/// ```
///
/// Model geometry (n_layers, n_kv_heads, head_dim) cancels in the ratio —
/// the limit is determined entirely by arena capacity versus turn size.
///
/// # Arguments
///
/// * `arena_chunks`      — Number of chunk slots in the KV arena
///   ([`BatchedConfig::arena_chunks`]).
/// * `chunk_size`        — Tokens per chunk ([`BatchedConfig::chunk_size`]).
/// * `tokens_per_turn`   — Expected maximum tokens per completed turn
///   (use `SequenceConfig::max_response_tokens` as a conservative upper bound).
///
/// Returns the number of hot turns, clamped to `[1, 256]`.
/// Returns `0` when `tokens_per_turn` is zero (disables eviction).
pub fn pick_max_hot_turns(arena_chunks: usize, chunk_size: usize, tokens_per_turn: usize) -> usize {
    if tokens_per_turn == 0 {
        return 0;
    }
    ((arena_chunks * chunk_size) / tokens_per_turn).clamp(1, 256)
}

// ────────────────────────────────────────────────────────────────────────────
// Sequence Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Per-conversation configuration.
#[derive(Debug, Clone)]
pub struct SequenceConfig {
    /// Dialect of the chat
    pub dialect: Dialect,

    /// Maximum tokens to generate per turn.
    pub max_response_tokens: usize,

    /// Default sampling configuration.
    pub sampling: SamplingConfig,

    /// Sequence tree policy: summarization cadence, KV tier formats,
    /// cognitive task prompts, etc.
    pub tree: ConversationTreeConfig,

    /// Number of most-recent turns to include in the context window sent to
    /// the GPU for each new turn.
    ///
    /// When non-zero the conversation rebuilds its KV sequence from scratch on
    /// every new turn by prefilling the system prompt plus the last
    /// `context_window_turns` completed exchanges (from the conversation tree)
    /// together with the new user message.  This bounds memory growth to a
    /// fixed context window regardless of conversation length.
    ///
    /// The conversation tree continues to accumulate the full history (including
    /// summarization). Only the GPU sequence is windowed.
    ///
    /// `0` = disabled — the sequence grows continuously (legacy behaviour).
    /// Default: `8`.
    pub context_window_turns: usize,

    /// Reserved for future Hot → Warm eviction policy.  Currently
    /// unused — substrate-tier eviction is not wired through to the
    /// workspace substrate yet.  Kept as a config field so callers
    /// can express their intent without churning the API.
    ///
    /// `0` = disabled (default).
    pub max_hot_turns: usize,

    /// Continuous re-projection cadence in decoded tokens.
    ///
    /// When `> 0`, the scheduler fires a provenance scan + re-projection + view
    /// swap each time the decoded-token count for an active turn becomes a
    /// multiple of this value.  The view's borrowed-block ranges are
    /// recomputed from scratch using the live Q-vector signatures of the
    /// turn-so-far as the provenance probe — letting attention "wander" through
    /// the substrate as the model's intent evolves mid-decode.
    ///
    /// Should be `>= CHUNK_SIZE` (32) so at least one freshly-sealed
    /// chunk's worth of Q-sigs is available between fires.
    ///
    /// **Default: `64`** — re-project every two sealed chunks.  Set to
    /// `0` to opt out (e.g. for benchmarking the static-projection path).
    /// Single-shot paths (RULER eval, summarisation) always disable
    /// continuous re-projection regardless of this setting.
    pub reproject_every_n_tokens: usize,

    /// Maximum tokens to include in the provenance probe at each
    /// reprojection event, looking backward from the current decode
    /// position.  Caps the "thought window" — beyond this many tokens
    /// the prior reprojection already accounted for the older intent.
    ///
    /// Should be a small multiple of `CHUNK_SIZE` (32) for tidy block
    /// arithmetic; **default: `256`** — a full tool-call turn's discriminative
    /// span (the §80.2 sweep shows the true tool is always within Top-3 at this
    /// width; a narrower 64-token tail needs a much larger budget for 100% recall).
    pub reproject_max_probe_tokens: usize,

    /// Texts that, when sampled as a single token during decode, fire a
    /// reprojection in addition to the cadence trigger — but only once
    /// **more than 16 non-trigger tokens** have been decoded since the last
    /// projection.  That content gate stops short lines and runs of trigger
    /// tokens (e.g. consecutive newlines) from each re-orienting attention.
    /// Encoded via the conversation's tokenizer at policy build time;
    /// any text that doesn't tokenize to exactly one ID is silently
    /// skipped.
    ///
    /// **Default: `["\n"]`** — re-orient attention at line breaks,
    /// which usually mark paragraph or section boundaries where the
    /// model's intent shifts.  Set to `vec![]` to disable
    /// punctuation-driven reprojection (cadence-only).
    pub reproject_trigger_texts: Vec<String>,

    /// When `true`, turns on this conversation skip the per-turn
    /// projection rebuild once the slot is seeded: the prefill appends
    /// onto the cumulative slot instead of resetting + re-projecting, and
    /// continuous mid-decode reprojection is forced off. The turn still
    /// seals into the substrate. Used by append-only utility ingests
    /// (`code_reading`, `repo_map`) where re-projecting the whole trunk
    /// every turn is unnecessary and O(n²). **Default: `false`.**
    pub disable_reprojection: bool,

    /// Per-conversation KV-compression level override for this
    /// conversation's *turns*. `None` ⇒ use the engine-wide turn policy
    /// (the level set on the model builder). `Some(level)` quantizes this
    /// conversation's sealed turns at `level` instead — used to compress
    /// cold reference layers (e.g. `code_reading` at C8) harder than live
    /// dialogue (C4). Section policies are unaffected. **Default: `None`.**
    pub kv_compression_level: Option<u8>,

    /// When `true`, this conversation's turns drop the engine-wide K-format
    /// override (Q4_KS) so K is adaptively quantized per-block like V. Only
    /// meaningful alongside `kv_compression_level`. Set for `code_reading`
    /// so its K and V are both fully quantized. **Default: `false`.**
    pub kv_disable_k_override: bool,

    /// Pin this conversation's turns to a single uniform K quant format,
    /// bypassing adaptive per-block selection *and* the engine-wide Q4_KS K
    /// override. `None` keeps the adaptive/level path. Reserved for layers that
    /// want a fixed near-lossless seal (e.g. summary turns); currently unset by
    /// all callers. **Default: `None`.**
    pub kv_force_k_format: Option<QuantFormat>,

    /// V counterpart to [`Self::kv_force_k_format`]. **Default: `None`.**
    pub kv_force_v_format: Option<QuantFormat>,

    /// When `true`, skip the hot→warm quantize pass for this conversation's
    /// turns entirely — persist K/V in the native R16/F16 form (lossless). Used
    /// to capture full-resolution tool-call exemplars for the provenance work.
    /// Overrides the level/force fields above. **Default: `false`.**
    pub kv_lossless: bool,
}

#[cfg(test)]
mod scheduler_config_tests {
    use super::SchedulerConfig;

    #[test]
    fn prefill_target_is_reliably_fillable() {
        // The scheduler per-forward target sits at the point the parallel
        // scope-ingest reliably fills every forward (amortization, design §6),
        // not chunked at 512 nor stalled waiting for a 4096-token forward the
        // reduced-scope scope KV rarely reaches. Must stay ≤ the model-side
        // `MAX_PREFILL_TOKENS` ceiling (4096).
        assert_eq!(SchedulerConfig::default().large_prefill_max_tokens, 2048);
        assert!(SchedulerConfig::default().large_prefill_max_tokens <= 4096);
    }
}

#[cfg(test)]
mod sampling_config_tests {
    use super::SamplingConfig;

    #[test]
    fn segment_temp_boost_defaults_to_zero() {
        assert_eq!(SamplingConfig::default().segment_temp_boost, 0.0);
        assert_eq!(SamplingConfig::argmax().segment_temp_boost, 0.0);
        assert_eq!(SamplingConfig::top_p(0.9, 0.7).segment_temp_boost, 0.0);
    }

    #[test]
    fn with_segment_temp_boost_sets_value() {
        let cfg = SamplingConfig::default().with_segment_temp_boost(0.05);
        assert_eq!(cfg.segment_temp_boost, 0.05);
    }

    #[test]
    fn think_close_budget_collapses_off_only_on_tight_budget() {
        use crate::stencil::ThinkMode;
        // Off with a short summary budget (100 < the 300 backstop): collapse to a
        // forced EMPTY block so the budget goes to the answer, not runaway thinking.
        let mut off_tight = SamplingConfig::compression();
        off_tight.set_think_close_budget(ThinkMode::Off, 100);
        assert_eq!(
            off_tight.force_segment_close_after, 1,
            "Off + tight → empty"
        );
        assert_eq!(off_tight.graceful_segment_close_after, 0);
        assert_eq!(off_tight.segment_close_ramp_len, 1);

        // Off with a roomy dialogue budget keeps the normal (220, 300) backstop.
        let mut off_roomy = SamplingConfig::compression();
        off_roomy.set_think_close_budget(ThinkMode::Off, 4096);
        let (g, f) = ThinkMode::Off.eot_budget();
        assert_eq!(
            off_roomy.force_segment_close_after, f,
            "Off + roomy → backstop"
        );
        assert_eq!(off_roomy.graceful_segment_close_after, g);

        // A REASONING dial is NEVER collapsed, even with a tiny answer budget — its
        // per-span think budget is independent of the answer length.
        let mut deep_tight = SamplingConfig::compression();
        deep_tight.set_think_close_budget(ThinkMode::Deep, 50);
        let (dg, df) = ThinkMode::Deep.eot_budget();
        assert_eq!(
            deep_tight.force_segment_close_after, df,
            "Deep never collapses"
        );
        assert_eq!(deep_tight.graceful_segment_close_after, dg);
        assert!(
            df > 50,
            "sanity: Deep's force budget exceeds the tiny answer budget"
        );
    }

    #[test]
    fn qwen3_preset_enables_thinking_steering() {
        for arch in ["qwen3", "qwen3moe", "qwen2moe"] {
            let cfg = SamplingConfig::for_gguf_architecture(arch);
            // Thinking-only temperature boost is enabled.
            assert_eq!(
                cfg.segment_temp_boost, 0.05,
                "{arch} should set segment_temp_boost to 0.05"
            );
            // DRY is re-enabled (kernel-gated to <think> blocks).
            let dry = cfg
                .dry
                .as_ref()
                .unwrap_or_else(|| panic!("{arch} should enable DRY"));
            assert_eq!(dry.multiplier, 0.8);
            assert_eq!(dry.base, 1.75);
            assert_eq!(dry.allowed_length, 2);
            assert_eq!(dry.range, 512);
        }
    }
}
