//! Describes how each token is drawn: the penalties, the reasoning-block
//! steering, and the sampling each think mode publishes.
//!
//! A `SamplingConfig` carries every knob the batched sampling kernel takes, from
//! temperature and nucleus through penalties, DRY and grammar stencils. The
//! per-architecture presets build it. Reasoning is steered by a close budget on
//! the `<think>` block, with a ramp that nudges the close and a hard cap that
//! forces it, and by a temperature and nucleus pair per think mode that a
//! checkpoint publishes separately. An architecture missing from the table has no
//! backstop, and it fails silently by running to the token ceiling. The
//! penalties and ramps are scoped so the grammar's own structure is not
//! penalised.

use super::*;

// Sampling Configuration

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
    ///
    /// **Scoped to a free-text value of a grammar.** It applies only while a
    /// stencil is decoding a free-text span outside the think block, and its
    /// look-back is that span's own tokens: it starts empty at every value and
    /// ends with it. Nowhere else — not in reasoning, not on the grammar's
    /// structure, not in unconstrained prose, and not in a tool call whose
    /// caller freed it from the penalties.
    pub dry: Option<DryConfig>,

    /// The repeat penalty while DRY is live, in place of
    /// [`Self::repeat_penalty`]. `None` keeps `repeat_penalty` there too.
    ///
    /// A repeat penalty tuned to stand in for DRY where DRY is off would be
    /// stacked on top of it where DRY is on; this is the value that applies
    /// there instead.
    pub dry_repeat_penalty: Option<f32>,

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

    /// How many of the sequence's most recent turns [`Self::cross_turn_penalty`]
    /// sees. `0` = every turn since the conversation opened.
    ///
    /// The penalty is flat — a token used once costs what one used a hundred
    /// times does — so without a bound it ends up on a character's whole
    /// vocabulary and stops distinguishing anything.
    pub cross_turn_window: usize,

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

    /// Token ID of the tool-call opener (`<tool_call>`), resolved beside the
    /// segment tokens. Sampled inside an open segment it is committed as the
    /// segment close instead, so a call never starts inside the reasoning
    /// block. `-1` = disabled.
    pub tool_call_open_token_id: i32,

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

    /// Token IDs that end a sentence (`.`, `!`, `?`) — the boundaries the
    /// graceful EOS waits for, together with [`Self::line_end_token_ids`].
    /// Resolved from the tokenizer at engine startup.
    pub sentence_end_token_ids: Vec<i32>,

    /// Every token ID whose text ends a line (decodes to something ending in
    /// `\n`), ascending. The boundary `graceful_segment_close_after` waits for.
    ///
    /// A thinking block is closed at the end of a line, not of a sentence: a
    /// `.` also appears inside numbers, addresses and paths, and a close fired
    /// there cut a security review mid-thought at `169.` of `169.254.169.254` —
    /// the model carried on reasoning where its answer belonged and never gave
    /// one. Reasoning breaks into lines every few dozen tokens, well inside the
    /// window between the graceful and forced closes. Resolved once, by scanning
    /// the vocabulary at engine startup; empty makes the graceful close a no-op.
    pub line_end_token_ids: Arc<[i32]>,

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

    /// Stencil constraint too wide to name: one bit per token, a clear bit
    /// forbidden. The row samples the full vocabulary under it in the wave's
    /// one dispatch, where an allow-list would carry most of the vocabulary to
    /// the device and its probabilities back.
    pub allowed_bits: Option<Arc<[u32]>>,

    /// This step samples without the repetition penalties — no presence,
    /// frequency, repeat, cross-turn or DRY history for its row. Set on one
    /// step's own copy of the config, by the grammar, where a token's having
    /// been written before says nothing about whether the model is repeating
    /// itself: see the decode loop's `step_configs`.
    pub unpenalized_step: bool,

    // ── RNG ────────────────────────────────────────────────────────────
    /// RNG seed for reproducible sampling.
    pub seed: u64,

    /// The temperature/top_p pair this checkpoint publishes for each think
    /// mode, swapped in by [`Self::set_think_mode`].
    ///
    /// `None` for a family that publishes one pairing for both modes, or whose
    /// numbers here were matched to a measured reference run rather than read
    /// off a card — then the think mode moves budgets and steering only, and
    /// whatever temperature the caller set is left alone.
    pub mode_sampling: Option<ModeSampling>,

    /// The think mode whose pair `temperature` / `top_p` currently hold, as set
    /// by [`Self::set_think_mode`]. `None` until one is applied, which means the
    /// thinking row — the arch table's own default.
    ///
    /// Recorded so that editing a row re-adopts the right one whatever order a
    /// caller applies things in, rather than leaving the old numbers standing
    /// until some later turn happens to re-apply a mode.
    pub think_mode: Option<crate::stencil::ThinkMode>,
}

/// The two temperature/top_p pairs a checkpoint publishes, one per think mode.
///
/// Qwen document these separately and they are not close: Qwen3.5 asks for
/// `1.0 / 0.95` while it is reasoning and `0.7 / 0.8` while it is not. Carrying
/// both on the config is what lets a per-turn think mode pick, rather than a
/// process committing to one of them at load and running the other mode on the
/// wrong numbers all day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeSampling {
    /// `(temperature, top_p)` for a turn that reasons in a `<think>` block.
    pub thinking: (f32, f32),
    /// `(temperature, top_p)` for a think-suppressed turn.
    pub instruct: (f32, f32),
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
            dry_repeat_penalty: None,
            repeat_last_n: 0,
            cross_turn_penalty: 0.0,
            cross_turn_window: 0,
            eos_boost: 0.0,
            dynamic_eos_boost: false,
            eos_ramp_start: 0,
            eos_ramp_len: 0,
            eos_boost_max_multiplier: 1.0,
            segment_close_boost: 0.0,
            segment_close_token_id: -1,
            segment_open_token_id: -1,
            tool_call_open_token_id: -1,
            segment_close_ramp_start: 0,
            segment_close_ramp_len: 0,
            segment_close_max_multiplier: 1.0,
            force_segment_close_after: 0,
            graceful_segment_close_after: 0,
            sentence_end_token_ids: Vec::new(),
            line_end_token_ids: Arc::from([]),
            segment_close_script: Vec::new(),
            graceful_eos_after: 0,
            forced_eos_after: 0,
            banned_tokens: Vec::new(),
            stencil: Vec::new(),
            allowed_bits: None,
            unpenalized_step: false,
            seed: 42,
            mode_sampling: None,
            think_mode: None,
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
    /// The `<think>`-block and turn-ending steering every Qwen reasoning
    /// architecture shares, applied on top of that generation's own sampling numbers.
    ///
    /// Shared rather than repeated per arm because the sampling numbers are what
    /// differ between Qwen3 and Qwen3.5+, and the steering is what does not. A second
    /// copy would let one generation keep a backstop the other quietly lost — and a
    /// missing backstop does not fail, it produces an empty answer after a full-length
    /// decode.
    pub(super) fn with_qwen_thinking_steering(self) -> Self {
        // **DRY runs only inside a free-text value of a grammar** — see
        // [`SamplingConfig::dry`]. Its window is the value being written and
        // nothing else: it starts empty at each value, so a name or a path the
        // value repeats from the prompt costs nothing, and it is off in the
        // think block and on the grammar's own structure. Unscoped, it ran over
        // the reasoning and spelled one loop a new way each time ("Keeper…
        // The.Keeper… Kepper"); without it at all, a `reflect` repeated "I am
        // standing in the catalogue." to the turn's 16k-token cap.
        //
        // **A run of four tokens repeats freely.** Counted in tokens, a name or
        // a place ("Kepler Station", "the command level") is three or four, so
        // at an allowed length of 2 a value naming somebody twice was pushed off
        // the second spelling partway through. At 4 a repeated name costs
        // nothing, and a sentence said again — "I am standing in the
        // catalogue." is seven — is priced out partway through its second time.
        //
        // The thinking-only temperature boost lets reasoning sample a touch
        // hotter while the answer stays at the reference temp.
        self.with_segment_temp_boost(0.05)
            .with_dry_penalty(0.8, 1.75, 4, 512)
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
            .with_eos_failsafe(800, 1000)
    }

    /// `(temperature, top_p)` Qwen publish for a Qwen3.5-lineage **thinking**
    /// turn, general tasks.
    ///
    /// Named rather than inlined because the two rows below differ by nothing a
    /// reader can check at a glance — a transposed digit between them is a model
    /// decoding on the other mode's card, which is silent.
    pub(super) const QWEN35_THINKING: (f32, f32) = (1.0, 0.95);

    /// `(temperature, top_p)` Qwen publish for a Qwen3.5-lineage **instruct**
    /// (think-suppressed) turn, general tasks.
    pub(super) const QWEN35_INSTRUCT: (f32, f32) = (0.7, 0.8);

    pub fn for_gguf_architecture(arch: &str) -> Self {
        match arch {
            // All Qwen3 models — dense and MoE — share the same official config.
            // The MoE variant reports as "qwen2moe" in GGUF metadata.
            // EOT boost ramp params are set here; the actual <think>/<​/think>
            // token IDs are resolved automatically from the tokenizer in
            // `resolve_thinking_tokens()` during engine startup.
            //
            // Matched to the LM Studio reference run: temp=0.8, top_k=40,
            // top_p=0.95, repeat_penalty=1.1.  A gentle multiplicative
            // repeat_penalty applies batch-wide.
            "qwen3" | "qwen3moe" | "qwen2moe" => Self::top_k_top_p(40, 0.95, 0.8)
                .with_repeat_penalty(1.1)
                .with_qwen_thinking_steering(),

            // **Qwen3.5 and later — their own published sampling, the same steering.**
            //
            // Qwen publish a *pair* of settings for this generation, one per think
            // mode, and they are not close: `1.0 / 0.95` while it reasons, `0.7 / 0.8`
            // while it does not, with `top_k=20, presence_penalty=1.5,
            // repetition_penalty=1.0` common to both. The presence penalty is what
            // their card names as reducing "endless repetitions".
            //
            // This arm returns the **thinking** row, because that is what a config
            // with no think mode applied yet should be; [`Self::set_think_mode`]
            // swaps in the other from [`ModeSampling`] the moment a turn declares
            // itself suppressed. Both rows were previously the non-thinking one —
            // `non_thinking_for_gguf_architecture` derives from this arm and
            // overrode only budgets — so the reasoning numbers existed nowhere and
            // a thinking turn decoded on the instruct card.
            //
            // The difference from the Qwen3 row above is visible in the output, not
            // just on paper: with the fallback's values the life generator produced
            // an invented world, and with these it produced the character, on canon,
            // at the right dates.
            //
            // **Being absent from this table is silent and catastrophic**, which is why
            // the list is long. The `_` arm below leaves `top_k` unset, the `</think>`
            // close ramp at `0` (disabled) and no EOS failsafe — so a checkpoint that
            // does not close its own reasoning block runs to `max_response_tokens` and
            // the caller discards the whole decode as all-reasoning. A stock checkpoint
            // hides that by always self-closing; an abliterated or merged one does not,
            // and the symptom is an empty document reported as a success. `qwen35` was
            // missing here while the entire lineage loaded through it, and three
            // separate third-party conversions were blamed for the result first.
            //
            // llama.cpp currently names this whole lineage by its first release —
            // `qwen35` / `qwen35moe` cover Qwen3.5, Qwen3.6 and Qwen3.8 alike (see
            // `candle-transformers/src/models/qwen35/mod.rs`). The `qwen36*` and
            // `qwen38*` spellings are listed against the day it stops doing that: the
            // cost of a string nobody emits is one alternation, and the cost of a
            // missing one is the paragraph above.
            //
            // **The repeat penalty is 1.05 where DRY is off, and the card's 1.0
            // where it is on.** Without DRY, the reasoning block rehearses its
            // answer until the budget closes it — "All good. Generating. [Output
            // Generation] → *Proceeds*" — and the table's reading was cut at its
            // budget in 26 of 32 replays. A gentle multiplicative penalty over the
            // recent window is what breaks that rehearsal (zend's coding row,
            // measured on "Output matches. ✅ Proceeds." 45 times over) without
            // forbidding a word that must repeat. Inside a free-text value DRY
            // carries that load, so the value samples on the card's own 1.0.
            "qwen35" | "qwen35moe" | "qwen36" | "qwen36moe" | "qwen38" | "qwen38moe" => {
                Self::top_k_top_p(20, Self::QWEN35_THINKING.1, Self::QWEN35_THINKING.0)
                    .with_presence_penalty(1.5)
                    .with_repeat_penalty(1.05)
                    .with_dry_repeat_penalty(1.0)
                    .with_mode_sampling(ModeSampling {
                        thinking: Self::QWEN35_THINKING,
                        instruct: Self::QWEN35_INSTRUCT,
                    })
                    .with_qwen_thinking_steering()
            }

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
            // (a Qwen3-30B-A3B Q4_K_M conversion, 2026-02-20).
            // LM Studio params: temp=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1, min_p=0.05.
            // min_p is not yet implemented in this sampler; all other params match.
            // Qwen3 non-thinking: no think block overhead, so all tokens are
            // response content.  EOS boost ramp starts at 400 tokens, full by 500.
            // graceful_eos fires at next sentence boundary after 512 tokens;
            // forced_eos fires unconditionally at 700 tokens.
            //
            // The 3.5-and-later strings are listed for the same reason they are in
            // [`Self::for_gguf_architecture`]: returning `None` here says "this
            // family has no thinking mode", which is false for every one of them —
            // `builder.rs` reads `<think>` straight out of their chat templates and
            // reports `thinking=true` in the same load.
            //
            // **`with_mode_sampling(Off)` is what makes this row differ in its
            // sampled numbers at all.** Everything here is derived from the thinking
            // arm, and for a long time the derivation touched only budgets — so the
            // "non-thinking parameters" this function's own doc comment promises
            // were, in temperature and top_p, the thinking ones. The families that
            // publish a pair now carry it, and the swap reads the mode off
            // [`ModeSampling`]; the ones that do not are unchanged by the call.
            "qwen3" | "qwen3moe" | "qwen2moe" | "qwen35" | "qwen35moe" | "qwen36" | "qwen36moe"
            | "qwen38" | "qwen38moe" => Some(
                Self::for_gguf_architecture(arch)
                    .with_mode_sampling_for(crate::stencil::ThinkMode::Off)
                    .with_no_segment_close()
                    // **`500` is the ramp's END, not its length.** The kernel
                    // computes `span = max(ramp_len - ramp_start, 1)`, so `100`
                    // here gave `100 - 400 = -300`, clamped to `1` — an
                    // inverted ramp that collapses to a cliff: no EOS boost at
                    // all through 400 tokens, then the full 3× multiplier from
                    // token 401 onward. Every other call site passes an end and
                    // gets the intended hundred-token span; only this one, the
                    // preset the non-thinking path actually runs under, had it
                    // backwards.
                    .with_dynamic_eos_boost(1.0, 400, 500, 3.0)
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

    /// Set the repeat penalty that applies while DRY is live
    /// ([`Self::dry_repeat_penalty`]).
    pub fn with_dry_repeat_penalty(mut self, penalty: f32) -> Self {
        self.dry_repeat_penalty = Some(penalty);
        self
    }

    /// The repeat penalty for a step, by whether DRY is live on it.
    pub fn repeat_penalty_for(&self, dry_live: bool) -> f32 {
        match (dry_live, self.dry_repeat_penalty) {
            (true, Some(p)) => p,
            _ => self.repeat_penalty,
        }
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
    /// `ramp_end` is the **absolute token count at which the boost is full**,
    /// not a length from `ramp_start`.
    ///
    /// Said here because the field it lands in is called `eos_ramp_len` and the
    /// name is a lie: the kernel computes `span = max(ramp_end - ramp_start, 1)`
    /// (`batched_sampling.cuh`), so a value *below* `ramp_start` inverts the
    /// ramp and the clamp turns it into a cliff — no boost at all until
    /// `ramp_start`, then the whole multiplier one token later. That is not a
    /// gentler ramp; it is the opposite of one, and it reads as correct at every
    /// call site.
    pub fn with_dynamic_eos_boost(
        mut self,
        boost: f32,
        ramp_start: i32,
        ramp_end: i32,
        max_multiplier: f32,
    ) -> Self {
        debug_assert!(
            ramp_end > ramp_start,
            "the EOS ramp ends at {ramp_end} and starts at {ramp_start} — an end below the start \
             collapses the ramp to a cliff at `ramp_start`, which is the opposite of what a ramp \
             is for"
        );
        self.eos_boost = boost;
        self.dynamic_eos_boost = true;
        self.eos_ramp_start = ramp_start;
        self.eos_ramp_len = ramp_end;
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

    /// Resolve `<think>` / `</think>` / `<tool_call>` token IDs from the
    /// tokenizer.
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
        if let Some(id) = tokenizer.token_to_id("<tool_call>") {
            self.tool_call_open_token_id = id as i32;
            tracing::trace!("Resolved <tool_call> token ID: {}", id);
        }
        // Resolve sentence-end token IDs for the graceful EOS.
        self.sentence_end_token_ids.clear();
        for boundary in [".", "!", "?"] {
            if let Some(id) = tokenizer.token_to_id(boundary) {
                let id = id as i32;
                if !self.sentence_end_token_ids.contains(&id) {
                    self.sentence_end_token_ids.push(id);
                }
            }
        }
        // The line ends are a vocabulary scan; a config that already carries
        // them (every per-turn clone of the engine's) keeps what it has.
        if self.line_end_token_ids.is_empty() {
            self.line_end_token_ids = line_end_token_ids(tokenizer);
        }
        tracing::debug!(
            sentence_ends = self.sentence_end_token_ids.len(),
            line_ends = self.line_end_token_ids.len(),
            "resolved sampling boundary tokens",
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
    /// budget (~220/300) is a *dialogue backstop* — it caps a runaway, and assumes
    /// something upstream has already stopped a block from opening at all
    /// (structurally, via the closed block `Dialect::thinking_suppression`
    /// prefills; the `/no_think` glue it used to lean on is empty on families
    /// that suppress that way). A short
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
        self.set_think_mode(mode, max_response_tokens);
    }

    /// [`Self::apply_think_mode`] without the tokenizer, as a builder.
    ///
    /// **This is the per-turn entry point.** The thinking token ids are resolved
    /// once at engine start and do not change, so a caller that already holds a
    /// resolved config can switch a single turn's mode without taking the engine
    /// lock for a tokenizer it does not need — which is what makes a per-turn
    /// think mode affordable at all.
    pub fn with_think_mode(
        mut self,
        mode: crate::stencil::ThinkMode,
        max_response_tokens: usize,
    ) -> Self {
        self.set_think_mode(mode, max_response_tokens);
        self
    }

    /// Everything a think mode decides except the token ids: the sampled pair,
    /// the in-segment steering, and the close budget.
    pub(super) fn set_think_mode(
        &mut self,
        mode: crate::stencil::ThinkMode,
        max_response_tokens: usize,
    ) {
        self.segment_suppress_penalty = mode.suppress_penalty();
        if mode == crate::stencil::ThinkMode::Off {
            self.segment_temp_boost = 0.0;
        }
        self.set_mode_sampling(mode);
        self.set_think_close_budget(mode, max_response_tokens);
    }

    /// Record the checkpoint's per-mode temperature/top_p pairs, and adopt the
    /// one for the current mode — the thinking row, when none has been applied.
    ///
    /// Adopting rather than only recording keeps the arch table's return value
    /// honest: what [`Self::for_gguf_architecture`] hands back is what a turn
    /// decodes on when nothing has declared a mode yet, and for these families
    /// that is the reasoning row.
    pub fn with_mode_sampling(mut self, modes: ModeSampling) -> Self {
        self.mode_sampling = Some(modes);
        self.adopt_mode_pair();
        self
    }

    /// Adopt the published temperature/top_p for one think mode, changing
    /// nothing else.
    ///
    /// A no-op on a config with no [`ModeSampling`] — a family that publishes a
    /// single pairing keeps whatever the caller set.
    pub fn with_mode_sampling_for(mut self, mode: crate::stencil::ThinkMode) -> Self {
        self.set_mode_sampling(mode);
        self
    }

    pub(super) fn set_mode_sampling(&mut self, mode: crate::stencil::ThinkMode) {
        self.think_mode = Some(mode);
        self.adopt_mode_pair();
    }

    /// Put the current mode's published pair into `temperature` / `top_p`.
    ///
    /// A no-op on a config with no [`ModeSampling`]: a family that publishes a
    /// single pairing keeps whatever the caller set.
    pub(super) fn adopt_mode_pair(&mut self) {
        let Some(modes) = self.mode_sampling else {
            return;
        };
        (self.temperature, self.top_p) = match self.think_mode {
            Some(crate::stencil::ThinkMode::Off) => modes.instruct,
            _ => modes.thinking,
        };
    }

    /// The tokenizer-independent half of [`Self::apply_think_mode`]: program the
    /// per-span EOT close ramp + graceful/force cutoffs from `mode.eot_budget()`,
    /// with the short-output collapse. Split out so the budget logic is unit-
    /// testable without a tokenizer.
    pub(super) fn set_think_close_budget(
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

    /// Hold this step to `set`: by name, or through its bitmap on a set too wide
    /// to name, with the repetition penalties lifted as a named row has them.
    pub fn constrain_to(&mut self, set: &AllowedSet) {
        match set.bits() {
            Some(bits) => {
                self.stencil.clear();
                self.allowed_bits = Some(Arc::clone(bits));
                self.unpenalized_step = true;
            }
            None => self.stencil = set.tokens().iter().map(|&t| t as i32).collect(),
        }
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
