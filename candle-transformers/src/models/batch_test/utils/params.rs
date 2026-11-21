//! Holds the parameters a gate runs its configurations with, and how each run
//! chooses its draft budget.
//!
//! A gate sets its prompts, sampling, token counts, test mode and draft budget
//! on one `TestParams` through builder methods, and the same generate path runs
//! whatever it sets. The budget is either asked of the model, which is what
//! production does, or pinned to a point on the width curve when a gate is
//! measuring the ladder itself. Speculation accepts only the model's own greedy
//! continuation, so the text, and with it the expected-output check, is the same
//! at any budget. Validation stays strict unless a majority threshold is set.
//! The story prompt is read from the shared fixtures, so a crate above measures
//! the same bytes.

use super::*;

/// How a run decides its draft budget.
///
/// Not a feature switch — both variants run the same generate path, and the
/// difference is whether the run is *using* the model's tuning or *producing*
/// it. A gate measuring the width ladder has to hold the budget still while it
/// varies the width; everything else should behave the way production does,
/// which means asking the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftBudget {
    /// Ask the model what it wants at this config's width — production
    /// behaviour, and what every gate uses unless it is measuring the ladder.
    Adaptive,
    /// Hold the budget at `k` regardless of width. `0` is a plain-decode
    /// baseline; anything else is a point on the curve being measured.
    Fixed(usize),
}

impl DraftBudget {
    /// Resolve to a token count for a wave of `width` sequences.
    pub(super) fn resolve<M: ManagedBatchedModel + ?Sized>(self, model: &M, width: usize) -> usize {
        match self {
            Self::Adaptive => model.draft_budget(width),
            Self::Fixed(k) => k,
        }
    }
}

/// The story a [`TestMode::StoryRewrite`] prompt asks the model to reproduce
/// with its character renamed.
///
/// Read from [`fixtures`], which is outside the `cuda` gate so that a harness
/// in a crate above this one measures the same bytes. Kept as a function here
/// because the gates already call it by this path, and because the alternative —
/// a second `include_str!` — is a different prompt that merely looks identical in
/// the source (see the fixtures module note).
pub fn story_prompt() -> String {
    fixtures::story_prompt()
}

impl TestParams {
    /// Create test parameters with a tokenizer from JSON string and a dialect
    pub fn new(
        generate_token_count: usize,
        tokenizer_json: &str,
        dialect: Dialect,
    ) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes()).unwrap();
        let begin_document_token = (!dialect.document_start.is_empty()).then(|| {
            tokenizer
                .token_to_id(dialect.document_start)
                .unwrap_or_default()
        });
        let token_ids = token_ids(&tokenizer);
        Ok(TestParams {
            print_outputs: false,
            skip_validation: false,
            disable_session_isolation: false,
            majority_pass_threshold: None,
            test_mode: TestMode::StoryRewrite,
            suppress_thinking: false,
            prompt_system: fixtures::system_prompt(),
            prompt_user: story_prompt(),
            per_config_prompts: Vec::new(),
            config_labels: Vec::new(),
            stop_on_eos: Vec::new(),
            names: fixtures::session_names(),
            generate_token_count,
            top_rung_token_count: None,
            tokenizer,
            token_ids,
            dialect,
            device,
            begin_document_token,
            timeout_secs: 120,
            int8mode: Int8Mode::Off,
            lora: None,
            speculative_max_draft: DraftBudget::Adaptive,
            override_k_quant: None,
            extra_rows: Vec::new(),
            routing_trace_dir: None,
        })
    }

    /// Write each config's expert routing into `dir` — see
    /// [`Self::routing_trace_dir`].
    pub fn with_routing_trace_dir(mut self, dir: PathBuf) -> Self {
        self.routing_trace_dir = Some(dir);
        self
    }

    /// Pin every sealed K block to `fmt` — see [`Self::override_k_quant`].
    /// Append rows the caller measured, below the harness's own.
    ///
    /// Installed before [`Self::validate_and_print`], which is what prints them — so a
    /// caller producing engine-driven rows measures its own after
    /// [`Self::run_loaded_collect`] and hands them over here.
    pub fn with_extra_rows(mut self, rows: Vec<ExtraRow>) -> Self {
        self.extra_rows = rows;
        self
    }

    pub fn with_override_k_quant(mut self, fmt: Option<QuantFormat>) -> Self {
        self.override_k_quant = fmt;
        self
    }

    /// Pin the draft budget instead of asking the model — `0` for a plain-decode
    /// baseline, otherwise a point on the width/budget curve being measured.
    /// See [`Self::speculative_max_draft`]; the model still clamps it.
    pub fn with_speculative(mut self, max_draft: usize) -> Self {
        self.speculative_max_draft = DraftBudget::Fixed(max_draft);
        self
    }

    /// Generate `tokens` on the C9 and C10 rows — see
    /// [`Self::top_rung_token_count`].
    pub fn with_top_rung_tokens(mut self, tokens: usize) -> Self {
        self.top_rung_token_count = Some(tokens);
        self
    }

    /// Tokens a config of `mode` generates.
    pub fn tokens_for(&self, mode: InferenceMode) -> usize {
        match (mode, self.top_rung_token_count) {
            (InferenceMode::C9 | InferenceMode::C10, Some(n)) => n,
            _ => self.generate_token_count,
        }
    }

    /// Set the inference [`Int8Mode`] shown in the comparison table's `int8` column.
    pub fn with_int8mode(mut self, mode: Int8Mode) -> Self {
        self.int8mode = mode;
        self
    }

    /// Run every sequence of every config through the named LoRA adapter.
    ///
    /// The adapter must already be loaded into the model by the `load` closure
    /// the gate hands to [`Self::run`] — this only opts the sequences in, which
    /// is the same two-step the engine uses: loaded once with the model, chosen
    /// per conversation.
    ///
    /// Every sequence, not a mixture, and deliberately so. A wave carries one
    /// adapter, so a gate that adapted half its contexts would be testing the
    /// scheduler's grouping rather than the adapter — and in this harness, which
    /// drives the session directly with no scheduler to group them, it would
    /// simply fail the forward.
    pub fn with_lora(mut self, name: impl Into<String>) -> Self {
        self.lora = Some(name.into());
        self
    }

    /// Create test parameters with default (ChatML) dialect
    pub fn new_with_defaults(generate_token_count: usize, tokenizer_json: &str) -> Result<Self> {
        Self::new(generate_token_count, tokenizer_json, Dialect::chat_ml())
    }

    /// Enable or disable thinking-mode suppression.
    /// When true, `no_think` is injected into the user content (the assistant
    /// header stays the plain `assistant_start`).
    pub fn with_suppress_thinking(mut self, suppress: bool) -> Self {
        self.suppress_thinking = suppress;
        self
    }

    /// Enable or disable printing of generated outputs
    pub fn with_print_outputs(mut self, print: bool) -> Self {
        self.print_outputs = print;
        self
    }

    /// Skip semantic output-vs-expected validation (keeps token-count validation).
    /// Useful for smaller or instruction-misaligned models where the story rewrite is not reliable.
    pub fn with_skip_validation(mut self, skip: bool) -> Self {
        self.skip_validation = skip;
        self
    }

    /// Set a custom timeout in seconds (default: 120).
    /// For larger models or slower hardware, increase this value.
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Replace the system prompt. Empty leaves the system turn with no text in
    /// it — the turn's own markers are still prefilled.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_system = prompt.into();
        self
    }

    /// Provide per-config user-prompt overrides (indexed by config position).
    /// A non-empty entry replaces `prompt_user` for that config in `run()`.
    pub fn with_per_config_prompts(mut self, prompts: Vec<String>) -> Self {
        self.per_config_prompts = prompts;
        self
    }

    /// Name each config's profile tables — see [`Self::config_labels`].
    pub fn with_config_labels(mut self, labels: Vec<String>) -> Self {
        self.config_labels = labels;
        self
    }

    /// Stop batched generation early once every session emits one of these
    /// token IDs (e.g. EOS).  Empty keeps the fixed-length benchmark behaviour.
    pub fn with_stop_on_eos(mut self, tokens: Vec<u32>) -> Self {
        self.stop_on_eos = tokens;
        self
    }

    /// Disable the cross-session distinctness heuristic while keeping the rest
    /// of the output validation enabled.
    pub fn with_disable_session_isolation(mut self, disable: bool) -> Self {
        self.disable_session_isolation = disable;
        self
    }

    /// Allow a config to pass if at least the given percentage of checks pass.
    /// When unset, validation remains strict and requires 100% success.
    pub fn with_majority_pass_threshold(mut self, pct: usize) -> Self {
        self.majority_pass_threshold = Some(pct.clamp(1, 100));
        self
    }

    /// Choose the validation strategy.
    ///
    /// * `StoryRewrite` (default) – the model must rewrite a story replacing
    ///   the main character's name.  Good for 3 B+ instruction-tuned models.
    /// * `NameGreeting` – the model just needs to mention the user's name in
    ///   its reply.  Good for small models (0.5 B+).
    pub fn with_test_mode(mut self, mode: TestMode) -> Self {
        self.test_mode = mode;
        match mode {
            TestMode::NameGreeting | TestMode::CoherenceCheck | TestMode::Skip => {
                self.prompt_system = [
                    "You are a helpful, friendly assistant.",
                    "The user's name is {INSERT_NAME}.",
                    "Always address the user by their name in every reply.",
                    "Be concise.",
                ]
                .join(" ");
                self.prompt_user = [
                    "Hello, my name is {INSERT_NAME}! I have been studying Mars and",
                    "find it fascinating. Mars is the fourth planet from the Sun, often",
                    "called the Red Planet due to iron oxide on its surface. It has a",
                    "thin atmosphere of about one percent of Earth's pressure, composed",
                    "mainly of carbon dioxide. Mars hosts Olympus Mons, the tallest",
                    "volcano in the solar system at nearly 22 kilometers high, and",
                    "Valles Marineris, a canyon stretching over 4000 kilometers. It has",
                    "two small moons, Phobos and Deimos, thought to be captured",
                    "asteroids. Surface temperatures range from minus 140 degrees at",
                    "the poles to 20 degrees at the equator. NASA's Perseverance rover",
                    "has been exploring Jezero Crater since 2021. As {INSERT_NAME},",
                    "I would love to hear your thoughts. Please greet {INSERT_NAME}",
                    "by name and share one Mars fact not mentioned above.",
                ]
                .join(" ");
            }
            TestMode::StoryRewrite => { /* already set from files */ }
        }
        self
    }

    /// This function gets the system prompt for the test based on an index, this
    /// will use the prompt_system and replace {INSERT_NAME} with the indexed name
    /// then feed it through the tokenizer to get token IDs
    ///
    /// **No system text is no system turn.** A chat template writes a system
    /// block only when there is something to put in it — Strata's server, which
    /// the Strata benchmark reproduces, renders a user-only request as the user
    /// turn alone — so an empty prompt yields the document token (where the
    /// dialect has one) and nothing else, never an empty
    /// `<|im_start|>system\n<|im_end|>\n` the request did not ask for.
    pub fn system_prompt_tokens(&self, index: usize) -> Vec<u32> {
        if self.prompt_system.is_empty() {
            return self.begin_document_token.into_iter().collect();
        }
        let name = &self.names[index % self.names.len()];
        let prompt = format!(
            "{}{}{}",
            self.dialect.system_start,
            self.prompt_system.replace("{INSERT_NAME}", name),
            self.dialect.system_end
        );
        self.begin_document_token
            .clone()
            .into_iter()
            .chain(
                self.tokenizer
                    .encode(prompt, false)
                    .unwrap()
                    .get_ids()
                    .iter()
                    .copied()
                    .filter(|id| Some(*id) != self.begin_document_token),
            )
            .collect()
    }

    /// Generate user prompt tokens for a specific session.
    ///
    /// For `StoryRewrite` mode the prompt is identical for all sessions (the
    /// template contains no `{INSERT_NAME}`).  For `NameGreeting` mode the
    /// user's name is embedded directly in the user turn so that even tiny
    /// models (which may ignore the system prompt) receive unique input
    /// tokens in the high-attention region of the context.
    /// How this dialect suppresses reasoning — see
    /// [`Dialect::thinking_suppression`], which owns the one-mechanism split.
    pub(super) fn thinking_suppression(&self) -> (&'static str, &'static str) {
        self.dialect.thinking_suppression(self.suppress_thinking)
    }

    /// The user turn's opening and `user_text`, as the prompt phase prefills
    /// them: encoded separately from [`Self::user_suffix_tokens`], so the
    /// harness can pad inside the user turn between the two.
    pub fn user_content_tokens(&self, user_text: &str) -> Vec<u32> {
        let (no_think, _) = self.thinking_suppression();
        let content = format!("{}{}{}", self.dialect.user_start, no_think, user_text);
        self.tokenizer
            .encode(content.as_str(), true)
            .unwrap()
            .get_ids()
            .iter()
            .copied()
            .filter(|id| Some(*id) != self.begin_document_token)
            .collect()
    }

    /// The user turn's close and the assistant opening — with the closed
    /// reasoning block where this dialect suppresses thinking there.
    pub fn user_suffix_tokens(&self) -> Vec<u32> {
        let (_, closed_think) = self.thinking_suppression();
        let suffix = format!(
            "{}{}{}",
            self.dialect.user_end, self.dialect.assistant_start, closed_think
        );
        self.tokenizer
            .encode(suffix.as_str(), false)
            .unwrap()
            .get_ids()
            .iter()
            .copied()
            .filter(|id| Some(*id) != self.begin_document_token)
            .collect()
    }

    /// Every token one session of a single-context config prefills for
    /// `user_text`: the system turn, then the user turn.
    pub fn prefill_token_count(&self, user_text: &str) -> usize {
        self.system_prompt_tokens(0).len()
            + self.user_content_tokens(user_text).len()
            + self.user_suffix_tokens().len()
    }

    pub fn user_prompt_tokens(&self, index: usize) -> Vec<u32> {
        let name = &self.names[index % self.names.len()];
        let user_text = self.prompt_user.replace("{INSERT_NAME}", name);
        let (no_think, closed_think) = self.thinking_suppression();
        let prompt = format!(
            "{}{}{}{}{}{}",
            self.dialect.user_start,
            no_think,
            user_text,
            self.dialect.user_end,
            self.dialect.assistant_start,
            closed_think,
        );
        self.tokenizer
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .iter()
            .copied()
            .filter(|id| Some(*id) != self.begin_document_token)
            .collect()
    }
}
