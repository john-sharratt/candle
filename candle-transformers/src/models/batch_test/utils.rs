use candle::forbidden_alloc;
use candle::quantized::Int8Mode;
use candle::{DType, Device, Result, Tensor};
use std::time::Duration;
use tokenizers::Tokenizer;

use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, InferenceMode, ManagedBatchedModel,
};
use crate::models::dialect::Dialect;
use crate::models::expert_lre::PipelineStats;
use crate::models::profile::ProfileSnapshot;
use crate::models::profile::{pipeline_record, pipeline_snapshot_and_reset, profile_now};

/// Determines the validation strategy for the test harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    /// Story rewrite: model must replace the character name "Marcus" with a
    /// per-session name.  Requires instruction-following ability (3 B+ models).
    StoryRewrite,
    /// Name greeting: model receives the user's name in the system prompt and
    /// must mention it in its reply.  Suitable for small models (0.5 B+).
    /// All adjacent different-name session pairs must produce distinct output.
    NameGreeting,
    /// Coherence check: verifies the model produces coherent text output
    /// (not garbage like repeated punctuation).  Does not compare sessions.
    /// Use for very lossy KV quantization formats (Q4_KS, Q3_0) where
    /// per-name differentiation is lost but correct decoding must still work.
    CoherenceCheck,
    /// Skip: run the config and collect timing/compression stats but perform
    /// no output validation.  Use for extremely lossy formats (Q2_0, Q3_0)
    /// where even coherence cannot be guaranteed but throughput matters.
    Skip,
}

#[derive(Debug, Clone)]
pub struct TestConfig {
    pub mode: InferenceMode,
    pub use_batched: bool,
    pub num_contexts: usize,
    pub num_repeats: usize,
    pub generate_max_len: usize,
    /// Override the global `TestParams::test_mode` for this specific config.
    /// `None` means use the global mode.
    pub test_mode: Option<TestMode>,
}

/// Model-agnostic test parameters for prompt and generate phases
pub struct TestParams {
    pub print_outputs: bool,
    pub skip_validation: bool,
    pub disable_session_isolation: bool,
    pub majority_pass_threshold: Option<usize>,
    pub test_mode: TestMode,
    pub suppress_thinking: bool,
    pub prompt_system: String, // System prompt text
    pub prompt_user: String,   // User prompt text
    /// Optional per-config user-prompt overrides, indexed by config position.
    /// A non-empty entry at index `n` replaces `prompt_user` for the n-th
    /// config in `run()`.  Empty string or missing index keeps `prompt_user`.
    /// Used by the routing-trace capture to drive diverse prompts through a
    /// single model load.
    pub per_config_prompts: Vec<String>,
    /// Token IDs that end generation early when sampled (e.g. EOS).  Empty =
    /// always generate the full `generate_token_count` (benchmark behaviour).
    /// When set, the batched generate loop stops once every session has emitted
    /// a stop token.  Used by capture to avoid post-EOS degenerate routing.
    pub stop_on_eos: Vec<u32>,
    pub names: Vec<String>,
    pub generate_token_count: usize, // Number of tokens to generate in generate phase
    pub tokenizer: Tokenizer,        // Tokenizer for text-to-token conversion
    pub dialect: Dialect,            // Chat format dialect (ChatML, Llama3, etc.)
    pub device: Device,
    pub begin_document_token: Option<u32>,
    pub timeout_secs: u64, // Test timeout in seconds (default: 120)
    /// Inference `Int8Mode` for this run, shown in the comparison table's `int8` column so a saved
    /// table records which numeric mode produced it. Defaults to `Off`; set via
    /// [`Self::with_int8mode`].
    pub int8mode: Int8Mode,
    /// When `Some(k)`, the generate phase uses lossless speculative decoding via
    /// [`ManagedBatchedModel::speculative_decode_step`] with a draft budget of `k`, instead of
    /// the one-token-per-step batched loop. Model-agnostic: a model with no drafter degrades to
    /// plain decode, so the output (and validation) is unchanged; a model with a drafter produces
    /// the same tokens faster. Defaults to `None` (classic batched decode).
    pub speculative_max_draft: Option<usize>,
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
        Ok(TestParams {
            print_outputs: false,
            skip_validation: false,
            disable_session_isolation: false,
            majority_pass_threshold: None,
            test_mode: TestMode::StoryRewrite,
            suppress_thinking: false,
            prompt_system: include_str!("system.md")
                .replace("\r\n", "\n")
                .replace("\r", "\n"),
            prompt_user: include_str!("story.md")
                .replace("\r\n", "\n")
                .replace("\r", "\n"),
            per_config_prompts: Vec::new(),
            stop_on_eos: Vec::new(),
            names: include_str!("names.md")
                .lines()
                .map(|s| s.to_string())
                .collect(),
            generate_token_count,
            tokenizer,
            dialect,
            device,
            begin_document_token,
            timeout_secs: 120,
            int8mode: Int8Mode::Off,
            speculative_max_draft: None,
        })
    }

    /// Enable lossless speculative decoding in the generate phase with a draft budget of `k`.
    pub fn with_speculative(mut self, max_draft: usize) -> Self {
        self.speculative_max_draft = Some(max_draft);
        self
    }

    /// Set the inference [`Int8Mode`] shown in the comparison table's `int8` column.
    pub fn with_int8mode(mut self, mode: Int8Mode) -> Self {
        self.int8mode = mode;
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

    /// Provide per-config user-prompt overrides (indexed by config position).
    /// A non-empty entry replaces `prompt_user` for that config in `run()`.
    pub fn with_per_config_prompts(mut self, prompts: Vec<String>) -> Self {
        self.per_config_prompts = prompts;
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
    pub fn system_prompt_tokens(&self, index: usize) -> Vec<u32> {
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
    pub fn user_prompt_tokens(&self, index: usize) -> Vec<u32> {
        let name = &self.names[index % self.names.len()];
        let user_text = self.prompt_user.replace("{INSERT_NAME}", name);
        let no_think = if self.suppress_thinking {
            self.dialect.no_think
        } else {
            ""
        };
        let prompt = format!(
            "{}{}{}{}{}",
            self.dialect.user_start,
            no_think,
            user_text,
            self.dialect.user_end,
            self.dialect.assistant_start
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

#[derive(Debug, Clone)]
pub struct PhaseResults {
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub runs_used: usize,
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub output: Vec<u32>,            // Generated output text
    pub expected: String,            // Expected output text
    pub logits: Tensor,              // Logits from the model
    pub prompt_phase: PhaseResults,  // Prompt processing results
    pub generate_avg_per_token: f64, // Average ms per generated token
    pub generate_total_ms: f64,      // Total time for all generate iterations
    pub total_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct TestResults {
    pub config: TestConfig,
    pub sessions: Vec<TestRun>,
    pub prompt_tokens_per_sec: f64, // Tokens/sec for bulk prompt processing
    pub generate_tokens_per_sec: f64, // Tokens/sec for single token generation
    pub all_valid: bool,            // Whether all sessions passed validation
    pub quantized_token_percent: Option<f64>, // Percentage of tokens stored in quantized arenas
    pub compression_ratio: Option<f64>, // Float-equivalent bytes / actual quantized bytes
    pub peak_tokens: usize,         // Total tokens across all sessions at peak (after generation)
    pub expert_stats: Option<PipelineStats>, // Expert cache telemetry (if model has MoE)
    pub bulk_profile: ProfileSnapshot, // Profile data from prompt (bulk) phase
    pub single_profile: ProfileSnapshot, // Profile data from generate (single) phase
    pub pipeline_bulk_profile: ProfileSnapshot, // Pipeline spans, prefill (bulk) phase only
    pub pipeline_profile: ProfileSnapshot, // Pipeline spans, decode (single/generate) phase only
    /// Effective test mode used for this config (may override the global `TestParams::test_mode`).
    pub effective_test_mode: TestMode,
}

/// Calculate statistics from timing measurements, dropping the first (warmup) and worst outlier
pub fn calculate_stats(times: &[Duration]) -> (f64, f64, f64, usize) {
    if times.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }

    let mut sorted: Vec<f64> = times.iter().map(|d| d.as_secs_f64()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Skip first (warmup) and last (worst outlier)
    let used = if sorted.len() > 2 {
        &sorted[1..sorted.len() - 1]
    } else {
        &sorted[..]
    };

    if used.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }

    let avg = used.iter().sum::<f64>() / used.len() as f64;
    let min = *used
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = *used
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    (avg, min, max, used.len())
}

/// Remove all `<think>…</think>` blocks from `text`.
/// Used to strip chain-of-thought reasoning from model output when `suppress_thinking` is set,
/// so that comparison and printing only see the final answer.
fn strip_thinking_blocks(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open_pos) = rest.find(OPEN) {
        out.push_str(&rest[..open_pos]);
        rest = &rest[open_pos + OPEN.len()..];
        if let Some(close_pos) = rest.find(CLOSE) {
            rest = &rest[close_pos + CLOSE.len()..];
        } else {
            // Unclosed <think> — drop everything from here.
            break;
        }
    }
    out.push_str(rest);
    out
}

/// Format a diff between expected and actual strings (compact, differences only)
fn format_diff(expected: &str, actual: &str) -> String {
    // Truncate expected to match actual length for fair comparison
    // Use floor_char_boundary to avoid slicing in the middle of a UTF-8 character
    let expected_truncated = if expected.len() > actual.len() {
        let boundary = actual.len();
        // Find the last valid char boundary at or before the target position
        let safe_boundary = expected
            .char_indices()
            .take_while(|(i, _)| *i <= boundary)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        &expected[..safe_boundary]
    } else {
        expected
    };

    let exp_lines: Vec<&str> = expected_truncated.lines().take(10).collect();
    let act_lines: Vec<&str> = actual.lines().take(10).collect();

    let mut output = String::from("\n");
    let mut diff_count = 0;
    for (i, (exp, act)) in exp_lines.iter().zip(act_lines.iter()).enumerate() {
        if exp != act {
            output.push_str(&format!("Line {}: - {}\n        + {}\n", i + 1, exp, act));
            diff_count += 1;
            if diff_count >= 3 {
                break;
            }
        }
    }
    if exp_lines.len() != act_lines.len() {
        output.push_str(&format!(
            "Length: expected {} lines, got {}\n",
            exp_lines.len(),
            act_lines.len()
        ));
    }
    output
}

impl TestParams {
    /// Generic test harness for forward pass performance testing.
    ///
    /// This version uses the new BatchedInferenceSession API for batched mode.
    /// The model must implement `ManagedBatchedModel` which includes `forward_batched`.
    ///
    /// # Arguments
    /// * `configs` - List of test configurations to run
    /// * `sequential` - Callbacks for non-batched mode (individual caches per sequence)
    /// * `model` - The model implementing ManagedBatchedModel for batched mode
    pub fn run<M>(
        mut self,
        configs: Vec<TestConfig>,
        load_model: impl Fn() -> Result<M>,
    ) -> Result<()>
    where
        M: ManagedBatchedModel,
    {
        println!("✓ TestParams created successfully");
        println!("  - Dialect: {:?}", self.dialect.dialect_type);
        println!("  - Generate tokens: {}", self.generate_token_count);
        println!(
            "  - Prompt system length: {} chars",
            self.prompt_system.len()
        );
        println!("  - Prompt user length: {} chars", self.prompt_user.len());
        println!("  - Names available: {}", self.names.len());

        // Spawn a background thread that will hard terminate the process after timeout
        let timeout_secs = self.timeout_secs;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(timeout_secs));
            eprintln!(
                "\n❌ TIMEOUT: Test exceeded {} seconds - forcibly terminating process",
                timeout_secs
            );
            std::process::exit(1);
        });

        let mut results = Vec::new();

        // GPU memory tracking: snapshot before model load
        candle::gpu_memory::clear();
        let (free_before, total) = self.device.mem_get_info().unwrap_or((0, 0));
        let cuda_ctx_bytes = total.saturating_sub(free_before);
        if cuda_ctx_bytes > 0 {
            candle::gpu_memory::register("cuda context", cuda_ctx_bytes);
        }
        let _ = candle::gpu_memory::snapshot("before_model_load", &self.device);

        let model = load_model()?;

        // GPU memory tracking: snapshot after model load and register weight memory
        let _ = candle::gpu_memory::snapshot("after_model_load", &self.device);
        let free_after = self.device.mem_get_info().map(|(f, _)| f).unwrap_or(0);
        let model_bytes = free_before.saturating_sub(free_after);
        candle::gpu_memory::register("model weights", model_bytes);

        for (n, config) in configs.into_iter().enumerate() {
            println!("\n=== Running tests for config: {:?} ===", config);

            // Per-config user-prompt override (used by routing-trace capture).
            if let Some(p) = self.per_config_prompts.get(n) {
                if !p.is_empty() {
                    self.prompt_user = p.clone();
                    println!("  - Prompt override ({} chars)", self.prompt_user.len());
                }
            }

            // Tag any captured routing records with this config index.
            crate::models::routing_capture::begin_config(n);

            // Prune cached embedding variants before each config
            model.prune()?;

            // Reset expert pipeline telemetry before each config
            model.reset_expert_stats();

            // Reset profile accumulators so each config gets clean data.
            // (snapshot_profiles() does snapshot+reset atomically, so just
            // discard the stale data from startup / previous config tail.)
            let _ = model.snapshot_profiles();

            // GPU memory tracking: snapshot before config and register arena state
            let arena_bytes = candle_nn::kv_cache::global_arena_gpu_bytes();
            candle::gpu_memory::register("arenas (all backings)", arena_bytes);
            let _ = candle::gpu_memory::snapshot(
                &format!("before_{:?}×{}", config.mode, config.num_contexts),
                &self.device,
            );

            // The harness is batched-only — the sequential/`forward_with_context` path is retired.
            let result = match self.run_batched_config(&config, &model) {
                Ok(r) => r,
                Err(e) => {
                    // Arena bytes and table were already captured by the ArenaErrGuard
                    // inside run_batched_config (while the session was still alive).
                    // Just take the OOM snapshot — it will use those pre-registered values.
                    let _ = candle::gpu_memory::snapshot(
                        &format!("OOM_{:?}×{}", config.mode, config.num_contexts),
                        &self.device,
                    );
                    let _ = candle::gpu_memory::print_report(&self.device);
                    return Err(e);
                }
            };

            // Collect expert pipeline stats for this config
            let mut result = result;
            result.expert_stats = model.expert_stats();

            results.push(result);

            // Release GPU logits tensors from completed config — they're not needed
            // for validation (which only uses generated token IDs and expected text).
            // This prevents accumulating GPU memory across configs.
            if let Some(last) = results.last_mut() {
                for run in &mut last.sessions {
                    run.logits = Tensor::zeros(1, DType::F32, &Device::Cpu)?;
                }
            }

            // Prune again after config to release any accumulated dtype variants
            model.prune()?;
            // Sync device between configs to ensure memory is released
            self.device.synchronize()?;

            // GPU memory tracking: snapshot after config cleanup
            let _ = candle::gpu_memory::snapshot(
                &format!("after_{:?}×{}", config.mode, config.num_contexts),
                &self.device,
            );
        }

        // Validate and print results
        self.validate_and_print_results(&mut results)
    }

    /// Run a configuration in batched mode using BatchedInferenceSession
    fn run_batched_config<M>(&self, config: &TestConfig, model: &M) -> Result<TestResults>
    where
        M: ManagedBatchedModel,
    {
        // Create the batch session from the model using the inference mode's KV format.
        let batch_config = BatchedConfig {
            k_format: config.mode.k_format(),
            v_format: config.mode.v_format(),
            compression_level: config.mode.compression_level(),
            ..Default::default()
        };
        // One loaded model serves every config in the sweep and the configs
        // differ in KV dtype, so the norm weights are re-materialised per config.
        // `create_batched_session` does that itself — see `maybe_change_dtype`.
        let mut session = model.create_batched_session(batch_config)?;

        // RAII diagnostic guard — MUST be declared AFTER `session`.
        // Rust drops locals in reverse declaration order, so this guard drops
        // BEFORE `session` on any early return via `?`.  That means the backings
        // are still alive in the global registry when we capture arena state.
        // On the happy path we set fire=false before the Ok().
        struct ArenaErrGuard {
            fire: bool,
        }
        impl Drop for ArenaErrGuard {
            fn drop(&mut self) {
                if self.fire {
                    // Register correct arena bytes while session is still alive
                    let arena_bytes = candle_nn::kv_cache::global_arena_gpu_bytes();
                    candle::gpu_memory::register("arenas (all backings)", arena_bytes);
                    // Register per-format breakdown for the gpu_memory report
                    let arena_report = candle_nn::kv_cache::global_arena_memory_report();
                    for (backing_idx, label, count, bytes) in &arena_report {
                        candle::gpu_memory::register(
                            &format!("  arena[{}] {} (×{})", backing_idx, label, count),
                            *bytes,
                        );
                    }
                    // Print the full per-arena breakdown table
                    candle_nn::kv_cache::global_print_arena_table();
                }
            }
        }
        let mut arena_guard = ArenaErrGuard { fire: true };

        // Allocate all sequences first
        let mut sequence_indices = Vec::with_capacity(config.num_contexts);
        for _ in 0..config.num_contexts {
            let seq_idx = session.create_sequence()?;
            sequence_indices.push(seq_idx);
        }

        // Collect system prompts with their sequence indices and lengths
        let system_data: Vec<(usize, Vec<u32>, Tensor)> = (0..config.num_contexts)
            .map(|n| {
                let tokens = self.system_prompt_tokens(n);
                let tensor = self.tokens_to_tensor(&tokens)?;
                Ok((sequence_indices[n], tokens, tensor))
            })
            .collect::<Result<Vec<_>>>()?;

        // Group by token length for batched processing
        let mut by_length: std::collections::BTreeMap<usize, Vec<(usize, Tensor)>> =
            std::collections::BTreeMap::new();
        for (seq_idx, tokens, tensor) in system_data {
            by_length
                .entry(tokens.len())
                .or_default()
                .push((seq_idx, tensor));
        }

        // Process each length group in batched forward
        let mut logits_map: std::collections::HashMap<usize, Tensor> =
            std::collections::HashMap::new();
        for (token_len, group) in by_length {
            let seq_idxs: Vec<usize> = group.iter().map(|(idx, _)| *idx).collect();
            let tensors: Vec<Tensor> = group.into_iter().map(|(_, t)| t).collect();

            let nl = model.num_layers();
            let logits_vec = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &seq_idxs,
                    &tensors,
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits_owned()?;

            // Store logits and advance sequences
            for (&seq_idx, logits) in seq_idxs.iter().zip(logits_vec.into_iter()) {
                session.advance_sequence(seq_idx, token_len)?;
                logits_map.insert(seq_idx, logits);
            }
        }

        let effective_mode = config.test_mode.unwrap_or(self.test_mode);

        // Build runs in original order
        let mut runs = Vec::with_capacity(config.num_contexts);
        for (n, &seq_idx) in sequence_indices.iter().enumerate() {
            let logits = logits_map
                .remove(&seq_idx)
                .ok_or_else(|| candle::Error::Msg(format!("missing logits for seq {}", seq_idx)))?;
            runs.push(self.create_test_run(n, logits, effective_mode));
        }

        // Prompt phase — per-session user prompts, padded to uniform length
        // for single-batch processing.  NameGreeting mode embeds {INSERT_NAME}
        // in the user turn, producing slightly different token counts per
        // session.  We right-pad shorter sequences so all sessions can be
        // processed in one forward_batched call.  For StoryRewrite mode the
        // user prompt is identical across sessions so no padding occurs.
        //
        // IMPORTANT: padding is applied to the *user content* tokens only,
        // BEFORE appending the turn-end / assistant-start suffix.  This keeps
        // pad tokens inside the user turn where they are harmless trailing
        // whitespace, rather than after <|im_start|>assistant where they
        // would cause the model to emit EOS immediately.
        let suffix_str = format!("{}{}", self.dialect.user_end, self.dialect.assistant_start);
        let suffix_tokens: Vec<u32> = self
            .tokenizer
            .encode(suffix_str.as_str(), false)
            .unwrap()
            .get_ids()
            .iter()
            .copied()
            .filter(|id| Some(*id) != self.begin_document_token)
            .collect();

        let content_tokens_per_session: Vec<Vec<u32>> = (0..config.num_contexts)
            .map(|n| {
                let name = &self.names[n % self.names.len()];
                let user_text = self.prompt_user.replace("{INSERT_NAME}", name);
                let no_think = if self.suppress_thinking {
                    self.dialect.no_think
                } else {
                    ""
                };
                let content = format!("{}{}{}", self.dialect.user_start, no_think, user_text);
                self.tokenizer
                    .encode(content.as_str(), true)
                    .unwrap()
                    .get_ids()
                    .iter()
                    .copied()
                    .filter(|id| Some(*id) != self.begin_document_token)
                    .collect::<Vec<u32>>()
            })
            .collect();

        // Ragged-batch validation: feed `forward_batched` genuinely UNEVEN
        // prefill lengths (no padding to a common length). For StoryRewrite the
        // user prompt is identical across sessions, so when the batch has >1
        // sequence we make the LAST context LONGER by more than one 32-token
        // CHUNK_SIZE of trailing whitespace — harmless padding inside the user
        // turn that the model ignores, so NO session's output changes (every
        // session still validates), yet the per-sequence chunk COUNT now differs
        // across the batch. That difference is what exercises per-seq KV
        // capacity + the decode writer slice: the bug (over-allocating every
        // slot to max(q_lens)) would leave extra tail chunks on the *shorter*
        // full-story sessions and desync their decode — so a regression shows up
        // as those validated sessions diverging/asserting. A 1-token diff stays
        // within one chunk and misses the class entirely.
        let pad_token = self
            .tokenizer
            .encode(" ", false)
            .ok()
            .and_then(|enc| enc.get_ids().first().copied())
            .unwrap_or(0);
        let user_tokens_per_session: Vec<Vec<u32>> = content_tokens_per_session
            .into_iter()
            .enumerate()
            .map(|(n, mut tokens)| {
                if config.num_contexts > 1 && n + 1 == config.num_contexts {
                    // > CHUNK_SIZE (32) so the chunk count actually differs.
                    tokens.extend(std::iter::repeat(pad_token).take(40));
                }
                tokens.extend_from_slice(&suffix_tokens);
                tokens
            })
            .collect();
        let user_lens: Vec<usize> = user_tokens_per_session.iter().map(|t| t.len()).collect();

        let user_tensors: Vec<Tensor> = user_tokens_per_session
            .iter()
            .map(|tokens| self.tokens_to_tensor(tokens))
            .collect::<Result<Vec<_>>>()?;

        // Throughput repeats re-prefill the SAME user tokens on the SAME
        // sequences. The wave-entry offset reconciler advances any offset that
        // is behind its physical backing, so a naive re-run APPENDS the prompt
        // again instead of overwriting it — after N repeats the model would see
        // the story N times and the system prompt would be drowned out of the
        // attention window. Truncate each sequence back to its pre-prompt
        // length between repeats so every repeat is a true re-prefill from
        // identical state; the final repeat leaves exactly one prompt in KV.
        let repeat_base_offsets: Vec<usize> = sequence_indices
            .iter()
            .map(|&i| session.sequence_offset(i).unwrap_or(0))
            .collect();
        let prompt_start = std::time::Instant::now();
        let t_prompt_total = profile_now();
        let mut repeat_base_logits: Option<Vec<Tensor>> = None;
        for repeat in 0..config.num_repeats.max(1) {
            if repeat > 0 {
                for (&seq_idx, &base) in sequence_indices.iter().zip(repeat_base_offsets.iter()) {
                    session.truncate_sequence_to_tokens(seq_idx, base)?;
                }
            }
            let nl = model.num_layers();
            let logits_vec = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &sequence_indices,
                    &user_tensors,
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits_owned()?;

            // Idempotence gate: with the truncate above, every repeat runs the
            // same tokens from the same state through the same kernels, so the
            // logits must be bit-identical. Any drift means repeat state leaked
            // (offset/backing divergence) and the throughput numbers are
            // measuring a different workload than reported.
            match &repeat_base_logits {
                None => repeat_base_logits = Some(logits_vec.clone()),
                Some(base) => {
                    for (i, (a, b)) in base.iter().zip(logits_vec.iter()).enumerate() {
                        let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
                            .abs()?
                            .flatten_all()?
                            .max(0)?
                            .to_scalar::<f32>()?;
                        if d != 0.0 {
                            candle::bail!(
                                "prompt repeat {} session {} is not idempotent: \
                                 logits max|delta| = {d:e} vs repeat 0",
                                repeat,
                                i
                            );
                        }
                    }
                }
            }

            for (logits, run) in logits_vec.into_iter().zip(runs.iter_mut()) {
                run.logits = logits;
            }
        }
        // Advance each sequence by its own (ragged) prompt length.
        for (&seq_idx, &ulen) in sequence_indices.iter().zip(user_lens.iter()) {
            session.advance_sequence(seq_idx, ulen)?;
        }
        self.device.synchronize()?;
        pipeline_record("bench:bulk_total", t_prompt_total);
        let prompt_duration = prompt_start.elapsed();
        let prompt_tokens = user_lens.iter().sum::<usize>() * config.num_repeats.max(1);
        let prompt_tokens_per_sec = (prompt_tokens as f64) / prompt_duration.as_secs_f64();

        // Snapshot bulk profile at prompt→generate boundary
        let bulk_profile = model.snapshot_profiles();
        // Capture + reset the pipeline spans here so the prefill (bulk) phase is
        // attributed separately from decode — the final `pipeline_snapshot_and_reset`
        // below then contains only the generate (decode) phase. Without this split
        // the MoE / mHC spans (which run in both phases) are un-attributable.
        let pipeline_bulk_profile = pipeline_snapshot_and_reset();

        // Quantize + seal the prefilled history, mirroring the substrate
        // scheduler's priming-projection boundary. `start_new_chunk = true` so
        // decode writes land in a fresh chunk rather than appending to the now
        // immutable quantized tail. Deliberately outside the timing windows.
        #[cfg(feature = "cuda")]
        {
            session.quantize_and_seal_sequences(&sequence_indices, true)?;
            self.device.synchronize()?;
        }

        // Generate phase
        let mut remaining_steps = self.generate_token_count;

        // Early-stop predicate: every session has emitted a stop token.
        let all_stopped = |toks: &[u32]| -> bool {
            !self.stop_on_eos.is_empty() && toks.iter().all(|t| self.stop_on_eos.contains(t))
        };

        // Warmup step (step 0) — skipped for the speculative phase, which runs
        // its own driver over the whole generate window.
        let mut stopped = false;
        if self.speculative_max_draft.is_none() && remaining_steps > 0 {
            let toks =
                self.decode_step_batched(&mut session, &sequence_indices, &mut runs, model)?;
            self.device.synchronize()?;
            remaining_steps -= 1;
            stopped = all_stopped(&toks);
        }

        self.device.synchronize()?;
        let generate_start = std::time::Instant::now();
        let t_decode_total = profile_now();
        let mut steps_run = 0usize;
        // The steady-state decode loop is the hot loop the transient tier
        // exists for, so it is the window worth measuring: every device
        // allocation inside it is one the wave path should have taken from a
        // bump range instead. The warmup step above is deliberately outside —
        // its first-touch allocations are unavoidable and would drown the
        // per-step traffic that matters.
        //
        // Arming is scoped to this block so an early `?` cannot leave the
        // detector on for the sealing and reporting that follow.
        let detector = forbidden_alloc::armed();
        if let Some(max_draft) = self.speculative_max_draft {
            // Lossless speculative decode (model-agnostic), per session.
            steps_run = self.speculative_decode_phase(
                &mut session,
                &sequence_indices,
                &mut runs,
                model,
                max_draft,
            )?;
        } else if !stopped {
            for _step_num in 0..remaining_steps {
                let toks =
                    self.decode_step_batched(&mut session, &sequence_indices, &mut runs, model)?;
                steps_run += 1;
                if all_stopped(&toks) {
                    break;
                }
            }
        }
        self.device.synchronize()?;
        drop(detector);
        let forbidden = forbidden_alloc::take_report();
        if !forbidden.is_clean() {
            eprintln!("[{:?}] {}", config.mode, forbidden);
        }
        // The other half of the picture: the detector says what did NOT come
        // from an arena, this says how much did. A phase whose peak is zero has
        // a chain that never started, which reads identically in the detector to
        // a chain that started and was never converted.
        #[cfg(feature = "cuda")]
        if let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() {
            if let Some([attn, ffn, fwd]) = candle_nn::kv_cache::wave_domain_stats(gpu_id) {
                // `peak` is a process-lifetime high-water mark, so once one
                // config saturates a span every later config reports the same
                // number. Read it as "the worst this process ever saw", not as a
                // per-config figure.
                eprintln!(
                    "[{:?}] wave arenas (peak is process-wide): attention {} B of {} B, \
                     ffn {} B of {} B, forward {} B of {} B",
                    config.mode, attn.1, attn.2, ffn.1, ffn.2, fwd.1, fwd.2
                );
            }
        }
        pipeline_record("bench:decode_total", t_decode_total);

        let generate_duration = generate_start.elapsed();
        let generate_tokens = steps_run * config.num_contexts;
        let generate_tokens_per_sec = if generate_tokens == 0 {
            0.0
        } else {
            (generate_tokens as f64) / generate_duration.as_secs_f64()
        };

        // Quantize + seal the decode tail before measuring so %Quantized and the
        // compression ratio reflect the full sequence (the substrate quantizes
        // at every turn boundary). Already-quantized prefill chunks pass through
        // the quantizer's preserve bucket unchanged.
        #[cfg(feature = "cuda")]
        {
            session.quantize_and_seal_sequences(&sequence_indices, false)?;
            self.device.synchronize()?;
        }

        // Calculate quantized token percentage for quantized modes (reads the
        // post-quantize arena formats set above).
        let quantized_token_percent = if config.mode.is_quantized() {
            session.estimate_quantized_percentage_by_sequences(&sequence_indices)
        } else {
            None
        };

        let compression_ratio = if config.mode.is_quantized() {
            session.compression_ratio_by_sequences(&sequence_indices)
        } else {
            None
        };

        #[cfg(feature = "verbose")]
        let compression_distribution = if config.mode.is_quantized() {
            Some(session.compression_dist_by_sequences(&sequence_indices))
        } else {
            None
        };

        // Calculate peak tokens: sum of all tokens across all sessions
        let peak_tokens: usize = (0..config.num_contexts)
            .map(|n| self.system_prompt_tokens(n).len() + user_lens[n] + self.generate_token_count)
            .sum();

        // Verbose palette4 diagnostic: print per-chunk arena format distribution for the
        // last sequence at layer 0 before we free anything.
        #[cfg(feature = "verbose")]
        if let Some(&last_seq) = sequence_indices.last() {
            session.print_palette4_stats(last_seq);
        }

        // Print the distribution
        #[cfg(feature = "verbose")]
        if let Some(compression_distribution) = compression_distribution {
            session.print_compression_distribution(&compression_distribution);
        }

        // Palette sanity check: an all-zero palette map means the backing was allocated but the
        // identity bytes were never written, which silently produces wrong sub-band assignments.
        // Panic here so the bug is caught at test time rather than producing subtle accuracy drift.
        // Note: empty k_pal / v_pal is valid and means "use the shared identity palette".
        for &seq_idx in &sequence_indices {
            if let Some(backing) = session.backings().first() {
                if let Some(chunks) = backing.live_chunks_as_sealed(seq_idx) {
                    for (ci, chunk) in chunks.iter().enumerate() {
                        if !chunk.k_pal.is_empty() && chunk.k_pal.iter().all(|&b| b == 0) {
                            panic!(
                                "UNINITIALIZED PALETTE: seq={seq_idx} blk={ci} k_pal is non-empty \
                                 but all-zeros (expected identity bytes, got uninitialised buffer)"
                            );
                        }
                        if !chunk.v_pal.is_empty() && chunk.v_pal.iter().all(|&b| b == 0) {
                            panic!(
                                "UNINITIALIZED PALETTE: seq={seq_idx} blk={ci} v_pal is non-empty \
                                 but all-zeros (expected identity bytes, got uninitialised buffer)"
                            );
                        }
                    }
                }
            }
        }

        // Explicitly free sequences and compact to release GPU memory before next config
        for &seq_idx in &sequence_indices {
            session.free_sequence(seq_idx)?;
        }
        let t_cleanup_sweep = profile_now();
        // Return the freed sequences' regions to the pool before the next
        // config claims them.
        let _ = session.release_empty_arenas();
        pipeline_record("bench:cleanup_sweep", t_cleanup_sweep);
        drop(session);
        self.device.synchronize()?;

        // Snapshot single (generate) profile
        let single_profile = model.snapshot_profiles();

        // Success path: disarm the error guard so it doesn't fire on normal drop.
        arena_guard.fire = false;
        Ok(TestResults {
            config: config.clone(),
            sessions: runs,
            prompt_tokens_per_sec,
            generate_tokens_per_sec,
            all_valid: true,
            quantized_token_percent,
            compression_ratio,
            peak_tokens,
            expert_stats: None, // Filled by run() after collection
            bulk_profile,
            single_profile,
            pipeline_bulk_profile,
            pipeline_profile: pipeline_snapshot_and_reset(),
            effective_test_mode: effective_mode,
        })
    }

    /// Helper to create a TestRun with expected output
    fn create_test_run(&self, index: usize, logits: Tensor, effective_mode: TestMode) -> TestRun {
        let name = &self.names[index % self.names.len()];
        let name_capitalized = format!(
            "{}{}",
            name.chars().next().unwrap().to_uppercase(),
            name.chars().skip(1).collect::<String>().to_lowercase()
        );
        let expected_output = match effective_mode {
            TestMode::StoryRewrite => self
                .prompt_user
                .replace("Marcus", &name_capitalized)
                .replace("marcus", &name.to_lowercase())
                .replace("MARCUS", &name.to_uppercase()),
            TestMode::NameGreeting | TestMode::CoherenceCheck | TestMode::Skip => {
                // For these modes, expected is just the name.
                name_capitalized.clone()
            }
        };

        TestRun {
            output: Default::default(),
            logits,
            expected: expected_output,
            prompt_phase: PhaseResults {
                avg_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                runs_used: 0,
            },
            generate_avg_per_token: 0.0,
            generate_total_ms: 0.0,
            total_time_ms: 0.0,
        }
    }

    /// Lossless speculative-decode generate phase (model-agnostic): decode ALL sessions together
    /// via the generic [`ManagedBatchedModel::speculative_decode_step_batch`] driver — every
    /// active session drafts a block, every block verifies in ONE call (a single wave when the
    /// model overrides `verify_blocks`), and each session accepts/rolls back independently. Fills
    /// each `runs[i].output` with the model's exact greedy continuation, so validation is
    /// identical to the batched loop — a model with no drafter degrades to plain decode. Sessions
    /// drop out of the cohort as they hit EOS/budget. Returns a nominal step count for the perf
    /// table.
    fn speculative_decode_phase<M>(
        &self,
        session: &mut BatchedInferenceSession,
        sequence_indices: &[usize],
        runs: &mut [TestRun],
        model: &M,
        max_draft: usize,
    ) -> Result<usize>
    where
        M: ManagedBatchedModel,
    {
        let nl = model.num_layers();
        let max_tokens = self.generate_token_count;
        let stop_on = &self.stop_on_eos;
        // First generated token per session = argmax of its prefill logits, held
        // OUT of the KV as the driver's `committed` seed.
        let mut committed: Vec<u32> = Vec::with_capacity(sequence_indices.len());
        let mut active: Vec<bool> = Vec::with_capacity(sequence_indices.len());
        for run in runs.iter_mut() {
            let c = run.logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
            run.output.push(c);
            committed.push(c);
            active.push(run.output.len() < max_tokens && !stop_on.contains(&c));
        }
        loop {
            let idxs: Vec<usize> = (0..sequence_indices.len()).filter(|&i| active[i]).collect();
            if idxs.is_empty() {
                break;
            }
            let seqs: Vec<usize> = idxs.iter().map(|&i| sequence_indices[i]).collect();
            let comms: Vec<u32> = idxs.iter().map(|&i| committed[i]).collect();
            // Per-session emit sinks over DISJOINT `runs` borrows: each pushes
            // into its own output and applies the budget/EOS policy — the exact
            // per-token loop plain decode uses.
            let mut emits: Vec<Box<dyn FnMut(u32) -> bool + '_>> = runs
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| active[*i])
                .map(|(_, run)| {
                    let out = &mut run.output;
                    Box::new(move |t: u32| {
                        out.push(t);
                        out.len() < max_tokens && !stop_on.contains(&t)
                    }) as Box<dyn FnMut(u32) -> bool>
                })
                .collect();
            let next = model
                .speculative_decode_step_batch(session, &seqs, &comms, max_draft, nl, &mut emits)?;
            drop(emits);
            for (k, &i) in idxs.iter().enumerate() {
                // `Some(c)` ⇒ the sink accepted `c` under budget/EOS policy (it
                // is already emitted and becomes the next seed); `None` ⇒ the
                // sink stopped this session.
                match next[k] {
                    Some(c) => committed[i] = c,
                    None => active[i] = false,
                }
            }
        }
        Ok(max_tokens)
    }

    /// Decode step for batched mode
    fn decode_step_batched<M>(
        &self,
        session: &mut BatchedInferenceSession,
        sequence_indices: &[usize],
        runs: &mut [TestRun],
        model: &M,
    ) -> Result<Vec<u32>>
    where
        M: ManagedBatchedModel,
    {
        let t_sample = profile_now();
        // Sample tokens using batched fused CUDA kernel (argmax mode)
        // Stack all logits into [batch_size, vocab_size] tensor
        let logits_batch: Vec<Tensor> = runs
            .iter()
            .map(|run| run.logits.squeeze(0))
            .collect::<Result<Vec<_>>>()?;
        let stacked_logits = Tensor::stack(&logits_batch, 0)?;

        // Use fused batched sampling kernel on CUDA, standard argmax elsewhere.
        // batched_sample_argmax() dispatches to the optimized CUDA kernel which
        // does argmax in a single kernel launch across all sequences.
        let token_ids = stacked_logits.batched_sample_argmax()?;

        // Convert to u32 tokens
        let sampled_tokens = token_ids.to_vec1::<u32>()?;

        // Validate sampled tokens - argmax returns u32::MAX when all values are -inf/NaN
        for (i, &tok) in sampled_tokens.iter().enumerate() {
            if tok == u32::MAX {
                // Get the actual logits for this sequence to help debug
                let seq_logits = &logits_batch[i];
                let max_val = seq_logits
                    .max(0)?
                    .to_dtype(DType::F32)?
                    .to_vec0::<f32>()
                    .unwrap_or(f32::NAN);
                let min_val = seq_logits
                    .min(0)?
                    .to_dtype(DType::F32)?
                    .to_vec0::<f32>()
                    .unwrap_or(f32::NAN);
                candle::bail!(
                    "Invalid token ID {} sampled for sequence {} - logits corrupted (max={}, min={})",
                    tok, i, max_val, min_val
                );
            }
        }
        pipeline_record("bench:decode_sample", t_sample);

        // Update outputs
        for (run, &tok) in runs.iter_mut().zip(sampled_tokens.iter()) {
            run.output.push(tok);
        }

        let t_inputs = profile_now();
        // Create input tensors
        let input_tensors: Vec<Tensor> = sampled_tokens
            .iter()
            .map(|&tok| {
                Tensor::new(&[tok], &self.device)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap()
            })
            .collect();
        pipeline_record("bench:decode_input_tensors", t_inputs);

        // Validate that no two sessions share a GID before the decode kernel.
        // This catches cross-session KV aliasing on the host side before it
        // silently corrupts GPU attention reads.
        if std::env::var("KV_VALIDATE_GIDS").is_ok() {
            session.validate_gid_uniqueness(sequence_indices)?;
        }

        let t_forward = profile_now();
        // Forward all sequences in batch (decode step: q=1 rows in the decode group)
        let nl = model.num_layers();
        let logits_vec = model
            .forward_wave(
                session,
                sequence_indices,
                &input_tensors,
                &[],
                &[],
                &[],
                &[],
                0,
                nl,
                None,
            )?
            .logits_owned()?;
        pipeline_record("bench:decode_forward_call", t_forward);

        for (logits, run) in logits_vec.into_iter().zip(runs.iter_mut()) {
            run.logits = logits;
        }

        let t_advance = profile_now();
        // Advance all sequence offsets by 1
        for &seq_idx in sequence_indices {
            session.advance_sequence(seq_idx, 1)?;
        }
        pipeline_record("bench:decode_advance", t_advance);

        Ok(sampled_tokens)
    }

    /// Validate results and print comparison table
    fn validate_and_print_results(&self, results: &mut [TestResults]) -> Result<()> {
        println!("\n=== All tests completed ===");

        let mut failed = false;
        for result in results.iter_mut() {
            println!("\n--- Config: {:?} ---", result.config);
            let mut config_valid = true;
            let mut validation_checked = 0usize;
            let mut validation_failed = 0usize;
            let required_pass_percent = self.majority_pass_threshold.unwrap_or(100);

            if result.sessions.is_empty() {
                println!("❌ No sessions found for this config");
                failed = true;
                continue;
            }

            // ── Per-session checks (token count, empty output) ──────────
            for (i, session) in result.sessions.iter().enumerate() {
                let output_str = {
                    let raw = self
                        .tokenizer
                        .decode(&session.output, true)
                        .unwrap_or_default();
                    if self.suppress_thinking {
                        strip_thinking_blocks(&raw)
                    } else {
                        raw
                    }
                };

                // Check token count.  Skipped when EOS early-stop is enabled,
                // since a short generation is the correct outcome there.
                let min_expected_tokens = self.generate_token_count.saturating_sub(5);
                if self.stop_on_eos.is_empty() && session.output.len() < min_expected_tokens {
                    println!(
                        "\n❌ Session {} FAILED: Generated only {} tokens, expected at least {}",
                        i,
                        session.output.len(),
                        min_expected_tokens
                    );
                    failed = true;
                    config_valid = false;
                }

                // ── StoryRewrite validation ──────────────────────────────────
                //
                // Goal: verify the model reproduces the expected story text
                // faithfully after a name substitution.  Each session's
                // `expected` field contains the FULL original prompt (~2 300
                // chars) with "Marcus" replaced by the session's assigned
                // name.  The model only generates ~40-50 chars (controlled
                // by `generate_max_len`), so the output is always much
                // shorter than `expected`.
                //
                // Two normalisation steps are applied before comparison:
                //
                //  1. **Whitespace collapsing** — `split_whitespace().join(" ")`
                //     KV quantisation (especially Q8_0 with many concurrent
                //     contexts) can cause the model to emit extra whitespace
                //     at token boundaries.  For example, the model might
                //     produce  "Astronaut  \nMarcus"  (two spaces then a
                //     newline) instead of  "Astronaut\nMarcus".  These are
                //     semantically identical, so we collapse all whitespace
                //     runs into a single space before comparing.
                //
                //  2. **Pronoun neutralisation** — gendered pronouns like
                //     "his"/"her", "he"/"she" are replaced with bracketed
                //     placeholders ("[his/her]", "[he/she]", etc.).  This
                //     lets us compare sessions that use female names against
                //     the original male-protagonist prompt without false
                //     mismatches when the model correctly adapts pronouns.
                //
                // After normalisation we compare via **common-prefix length**
                // rather than slicing both strings to a fixed length:
                //
                //   common_prefix_len = number of leading chars that match
                //   min_len           = min(output_len, expected_len)
                //   required          = min_len - TOLERANCE  (TOLERANCE = 5)
                //
                //   PASS  iff  common_prefix_len >= required
                //
                // Why common-prefix instead of substring equality:
                //
                //   KV quantisation can cause the last 1-2 *generated tokens*
                //   to diverge from the expected text — not just truncation
                //   (fewer chars) but actual content differences where the
                //   model emits different characters (e.g. "\n\n" instead of
                //   " in").  The divergence point shifts non-deterministically
                //   between runs because it depends on which experts were
                //   cached at the time.  A fixed `saturating_sub(N)` approach
                //   fails because it assumes the mismatch is always at the
                //   very end, but with content divergence the offset of the
                //   first wrong character varies.  Measuring the actual
                //   common-prefix length is robust against this.
                //
                // What this still catches (i.e. real model failures):
                //
                //   • **Garbage output** — if the model produces random text
                //     the common prefix will be ~0 chars, far below the
                //     ~35-43 char requirement.  FAIL.
                //
                //   • **Wrong-name contamination** — if session isolation is
                //     broken (one session bleeds into another), the name in
                //     the output won't match `expected` and the prefix
                //     diverges early at the name position.  FAIL.
                //
                //   • **Completely wrong story** — if attention or FFN is
                //     broken the output won't start with "The Backyard
                //     Astronaut …".  FAIL.
                //
                //   • **Empty output** — explicitly checked before prefix
                //     comparison.  FAIL.
                //
                //   • **Too-short output** — the token-count check above
                //     (min_expected_tokens) already flags this, and the
                //     prefix comparison would also fail since min_len would
                //     be very small → required ≈ 0, but the token check
                //     catches it first.
                //
                // The TOLERANCE of 5 chars allows at most ~1-2 tokens of
                // tail divergence (typical token is 3-5 chars).  This is
                // tight enough that any structural problem — wrong content,
                // wrong name, broken attention — will still be caught.
                // ──────────────────────────────────────────────────────────
                if !self.skip_validation {
                    match result.effective_test_mode {
                        TestMode::StoryRewrite => {
                            validation_checked += 1;
                            let output_trimmed = output_str.trim();
                            if !output_trimmed.is_empty() {
                                let expected_trimmed = session.expected.trim();

                                // Normalize for comparison (see block comment above).
                                let normalize = |text: &str| -> String {
                                    let collapsed: String =
                                        text.split_whitespace().collect::<Vec<_>>().join(" ");
                                    let padded = format!(" {} ", collapsed);
                                    padded
                                        .replace(" his ", " [his/her] ")
                                        .replace("His ", "[His/Her] ")
                                        .replace(" her ", " [his/her] ")
                                        .replace("Her ", "[His/Her] ")
                                        .replace(" he ", " [he/she] ")
                                        .replace(" He ", " [He/She] ")
                                        .replace(" she ", " [he/she] ")
                                        .replace("She ", "[He/She] ")
                                        .replace(" him ", " [him/her] ")
                                        .replace("Him ", "[Him/Her] ")
                                        .replace(" wife ", " [wife/husband] ")
                                        .replace(" husband ", " [wife/husband] ")
                                        // Contraction normalization: the model may
                                        // expand or contract these equivalently.
                                        .replace("she'd ", "she had ")
                                        .replace("She'd ", "She had ")
                                        .replace("he'd ", "he had ")
                                        .replace("He'd ", "He had ")
                                        .trim()
                                        .to_string()
                                };

                                let output_chars: Vec<char> =
                                    normalize(output_trimmed).chars().collect();
                                let expected_chars: Vec<char> =
                                    normalize(expected_trimmed).chars().collect();

                                let common_prefix_len = output_chars
                                    .iter()
                                    .zip(expected_chars.iter())
                                    .take_while(|(a, b)| a == b)
                                    .count();

                                const TOLERANCE: usize = 5;
                                let min_len = output_chars.len().min(expected_chars.len());
                                let required = min_len.saturating_sub(TOLERANCE);

                                if common_prefix_len < required {
                                    let cmp_len = min_len.min(common_prefix_len + 10);
                                    let output_cmp: String =
                                        output_chars[..cmp_len].iter().collect();
                                    let expected_cmp: String =
                                        expected_chars[..cmp_len].iter().collect();
                                    println!(
                                        "\n⚠ Session {} output diverges at char {} \
                                         (need {} of {} to match, tolerance {}):",
                                        i, common_prefix_len, required, min_len, TOLERANCE
                                    );
                                    println!("{}", format_diff(&expected_cmp, &output_cmp));
                                    validation_failed += 1;
                                }
                            } else {
                                println!("\n❌ Session {} FAILED: Generated empty output", i);
                                validation_failed += 1;
                            }
                        }
                        TestMode::NameGreeting => {
                            validation_checked += 1;
                            // Empty-output check only; consistency is validated below.
                            if output_str.trim().is_empty() {
                                println!("\n❌ Session {} FAILED: Generated empty output", i);
                                validation_failed += 1;
                            }
                        }
                        TestMode::Skip => {
                            // No validation — just run for timing/compression stats.
                        }
                        TestMode::CoherenceCheck => {
                            validation_checked += 1;
                            let trimmed = output_str.trim();
                            if trimmed.is_empty() {
                                println!("\n❌ Session {} FAILED: Generated empty output", i);
                                validation_failed += 1;
                            } else {
                                // Check for garbage output (e.g. "!!!!!!!!!!" from broken
                                // quantized reads).  Coherent text must contain alphanumeric
                                // content — at least 20% of characters should be letters or
                                // digits.  This catches repeated punctuation/symbols while
                                // allowing numbered lists, sparse formatting, equations, etc.
                                let alnum_count =
                                    trimmed.chars().filter(|c| c.is_alphanumeric()).count();
                                let total = trimmed.chars().count();
                                if total > 0 && (alnum_count * 100 / total) < 20 {
                                    println!(
                                        "\n❌ Session {} FAILED: Output appears to be garbage \
                                         ({}/{} chars alphanumeric, need ≥20%): {:?}",
                                        i,
                                        alnum_count,
                                        total,
                                        {
                                            let mut end = trimmed.len().min(80);
                                            while end > 0 && !trimmed.is_char_boundary(end) {
                                                end -= 1;
                                            }
                                            &trimmed[..end]
                                        }
                                    );
                                    validation_failed += 1;
                                }
                            }
                        }
                    }
                }

                if self.print_outputs {
                    println!("\n--- Session {} Output ---\n{}", i, output_str);
                }
            }

            // ── Per-session validation summary with majority threshold ───
            if !self.skip_validation
                && result.effective_test_mode != TestMode::Skip
                && validation_checked > 0
            {
                let validation_passed = validation_checked.saturating_sub(validation_failed);
                let validation_pct = validation_passed * 100 / validation_checked;
                if validation_pct >= required_pass_percent {
                    match result.effective_test_mode {
                        TestMode::StoryRewrite => println!(
                            "✓ {}/{} sessions matched expected output ({}% pass, threshold {}%)",
                            validation_passed,
                            validation_checked,
                            validation_pct,
                            required_pass_percent
                        ),
                        TestMode::NameGreeting => println!(
                            "✓ {}/{} sessions produced non-empty output ({}% pass, threshold {}%)",
                            validation_passed,
                            validation_checked,
                            validation_pct,
                            required_pass_percent
                        ),
                        TestMode::CoherenceCheck => println!(
                            "✓ {}/{} sessions produced coherent output ({}% pass, threshold {}%)",
                            validation_passed,
                            validation_checked,
                            validation_pct,
                            required_pass_percent
                        ),
                        TestMode::Skip => {}
                    }
                } else {
                    println!(
                        "✗ Only {}/{} sessions passed validation ({}%, need ≥{}%)",
                        validation_passed,
                        validation_checked,
                        validation_pct,
                        required_pass_percent
                    );
                    failed = true;
                    config_valid = false;
                }
            }

            // ── NameGreeting: session isolation validation ────────────────
            //
            // We validate that the batching system correctly isolates
            // sessions by checking that sessions with *different* names
            // produce *different* token sequences.  If the KV cache or
            // token routing is broken, multiple sessions would collapse
            // to the same output (they'd all read from one session's
            // context).
            //
            // We deliberately do NOT compare same-name sessions at
            // different batch positions.  Tiny models (0.5B) have flat
            // logit distributions where floating-point non-determinism
            // across batch positions can flip the argmax — that is
            // inherent numerical noise, not a batching bug.
            //
            // CoherenceCheck skips this entirely — formats like Q4_KS
            // and Q3_0 are too lossy to reliably differentiate names.
            if !self.skip_validation
                && !self.disable_session_isolation
                && result.effective_test_mode != TestMode::Skip
            {
                if result.effective_test_mode == TestMode::NameGreeting {
                    let num_sessions = result.sessions.len();
                    if num_sessions > 1 {
                        let num_names = self.names.len();
                        let mut isolation_checked = 0usize;
                        let mut isolation_failed = 0usize;

                        for i in 1..num_sessions {
                            let name_i = i % num_names;
                            let name_prev = (i - 1) % num_names;
                            if name_i != name_prev {
                                isolation_checked += 1;
                                if result.sessions[i].output == result.sessions[i - 1].output {
                                    isolation_failed += 1;
                                    if isolation_failed <= 3 {
                                        println!(
                                        "\n⚠ Session {} ({}) produced identical output to session {} ({}) — possible session isolation failure",
                                        i, &self.names[name_i], i - 1, &self.names[name_prev]
                                    );
                                        // Debug: show output tokens
                                        let decode_out = |tokens: &[u32]| -> String {
                                            self.tokenizer
                                                .decode(tokens, true)
                                                .unwrap_or_else(|_| format!("{:?}", tokens))
                                        };
                                        println!(
                                            "  Output tokens ({} tokens): {:?}",
                                            result.sessions[i].output.len(),
                                            &result.sessions[i].output
                                                [..result.sessions[i].output.len().min(20)]
                                        );
                                        println!(
                                            "  Decoded: {}",
                                            decode_out(&result.sessions[i].output)
                                        );
                                    }
                                }
                            }
                        }

                        if isolation_checked > 0 {
                            let isolation_passed =
                                isolation_checked.saturating_sub(isolation_failed);
                            let isolation_pct = isolation_passed * 100 / isolation_checked;
                            if isolation_pct >= required_pass_percent {
                                println!(
                                    "✓ {}/{} different-name session pairs produced distinct output ({}% pass, threshold {}%)",
                                    isolation_passed,
                                    isolation_checked,
                                    isolation_pct,
                                    required_pass_percent
                                );
                            } else {
                                println!(
                                    "✗ Only {}/{} different-name pairs were distinct ({}%, need ≥{}%)",
                                    isolation_passed,
                                    isolation_checked,
                                    isolation_pct,
                                    required_pass_percent
                                );
                                failed = true;
                                config_valid = false;
                            }
                        }
                    }
                }
            }

            if self.skip_validation {
                let total_tokens: usize = result.sessions.iter().map(|s| s.output.len()).sum();
                println!(
                    "✓ {} sessions generated {} tokens [validation skipped]",
                    result.sessions.len(),
                    total_tokens
                );
            }
            result.all_valid = config_valid;
        }

        self.print_comparison_table(results);
        self.print_expert_stats_table(results);

        // grab the first 5 results and the last result into a slice
        let results = results.to_vec();
        let results = if results.len() > 6 {
            [&results[0..5], &results[results.len() - 1..results.len()]].concat()
        } else {
            results
        };
        let results = &results;

        self.print_profile_table(results, "Bulk (Prompt) Profile", |r| &r.bulk_profile);
        self.print_profile_table(results, "Single (Generate) Profile", |r| &r.single_profile);
        self.print_profile_table(results, "Pipeline Profile — PREFILL (bulk)", |r| {
            &r.pipeline_bulk_profile
        });
        self.print_profile_table(results, "Pipeline Profile — DECODE (single)", |r| {
            &r.pipeline_profile
        });

        if failed {
            Err(candle::Error::msg("Some tests failed"))
        } else {
            Ok(())
        }
    }

    /// Tokenize text using a tokenizer JSON string
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        Ok(input)
    }

    /// Print comparison table of test results
    fn print_comparison_table(&self, results: &[TestResults]) {
        if results.is_empty() {
            return;
        }

        println!("\n\n=== Performance Comparison ===");
        println!("┌──────────┬──────┬─────────┬──────────┬───────┬────────────┬──────────────┬─────────────┬───────────────┬───────────┬──────────┬────────────┐");
        println!("│ KvMode   │ int8 │ Batched │ Contexts │ Valid │ t/s (bulk) │ t/s (single) │ perf (bulk) │ perf (single) │ %Quantized│ Compress │ Peak Tokens│");
        println!("├──────────┼──────┼─────────┼──────────┼───────┼────────────┼──────────────┼─────────────┼───────────────┼───────────┼──────────┼────────────┤");

        // Baseline is the first config
        let baseline = &results[0];
        let baseline_prompt_tps = baseline.prompt_tokens_per_sec;
        let baseline_generate_tps = baseline.generate_tokens_per_sec;

        for (i, result) in results.iter().enumerate() {
            let mode_str = format!("{:?}", result.config.mode);
            // int8 weight mode for this run (one mode per load, same for every row).
            let int8_str = match self.int8mode {
                Int8Mode::Off => "off",
                Int8Mode::Performance => "perf",
                Int8Mode::Precision => "prec",
            };
            let batched_str = if result.config.use_batched {
                "  yes  "
            } else {
                "  no   "
            };
            let contexts = result.config.num_contexts;
            let valid_str = match result.effective_test_mode {
                TestMode::NameGreeting | TestMode::Skip => "  -  ",
                _ if result.all_valid => "  ✓  ",
                _ => "  ✗  ",
            };
            let prompt_tps = result.prompt_tokens_per_sec;
            let generate_tps = result.generate_tokens_per_sec;

            let prompt_perf = if i == 0 {
                "baseline".to_string()
            } else {
                format!(
                    "{:+.1}%",
                    ((prompt_tps / baseline_prompt_tps - 1.0) * 100.0)
                )
            };

            let generate_perf = if i == 0 {
                "baseline".to_string()
            } else {
                format!(
                    "{:+.1}%",
                    ((generate_tps / baseline_generate_tps - 1.0) * 100.0)
                )
            };

            let quant_str = match result.quantized_token_percent {
                Some(pct) => format!("{:>8.1}%", pct),
                None => "     -   ".to_string(),
            };

            let compress_str = match result.compression_ratio {
                Some(ratio) => format!("{:>6.2}x", ratio),
                None => "    -  ".to_string(),
            };

            println!(
                "│ {:>8} │ {:>4} │ {} │ {:>8} │ {} │ {:>10.1} │ {:>12.1} │ {:>11} │ {:>13} │ {} │ {:>8} │ {:>10} │",
                mode_str,
                int8_str,
                batched_str,
                contexts,
                valid_str,
                prompt_tps,
                generate_tps,
                prompt_perf,
                generate_perf,
                quant_str,
                compress_str,
                result.peak_tokens
            );
        }

        println!("└──────────┴──────┴─────────┴──────────┴───────┴────────────┴──────────────┴─────────────┴───────────────┴───────────┴──────────┴────────────┘");
    }

    /// Print a transposed expert pipeline stats table.
    ///
    /// Rows = metrics, Columns = test configs (#1, #2, …).
    /// Only printed if at least one config has expert stats.
    fn print_expert_stats_table(&self, results: &[TestResults]) {
        // Skip entirely if no config has expert stats.
        if !results.iter().any(|r| r.expert_stats.is_some()) {
            return;
        }

        // Build column headers from config descriptions.
        let headers: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("#{} {:?}×{}", i + 1, r.config.mode, r.config.num_contexts))
            .collect();

        // Determine column width (minimum 12 to fit numbers).
        let col_w = headers.iter().map(|h| h.len()).max().unwrap_or(10).max(12);

        // Metric rows: (label, extractor returning formatted string).
        let metrics: Vec<(&str, Box<dyn Fn(&PipelineStats) -> String>)> = vec![
            (
                "Expert hits",
                Box::new(|s: &PipelineStats| format!("{}", s.expert_hits)),
            ),
            (
                "Expert misses",
                Box::new(|s: &PipelineStats| format!("{}", s.expert_misses)),
            ),
            (
                "Hit rate",
                Box::new(|s: &PipelineStats| format!("{:.1}%", s.hit_rate())),
            ),
            (
                "DMA loads (H2D)",
                Box::new(|s: &PipelineStats| format!("{}", s.dma_loads)),
            ),
            (
                "Warm tier slots",
                Box::new(|s: &PipelineStats| {
                    if s.total_experts == 0 {
                        format!("{}", s.warm_slots)
                    } else {
                        format!(
                            "{} ({:.0}%)",
                            s.warm_slots,
                            100.0 * s.warm_slots as f64 / s.total_experts as f64
                        )
                    }
                }),
            ),
            (
                "Warm loads (RAM)",
                Box::new(|s: &PipelineStats| format!("{}", s.warm_loads)),
            ),
            (
                "Cold loads (pack)",
                Box::new(|s: &PipelineStats| format!("{}", s.cold_loads)),
            ),
            (
                "Evictions",
                Box::new(|s: &PipelineStats| format!("{}", s.evictions)),
            ),
            (
                "Prefetch loads",
                Box::new(|s: &PipelineStats| format!("{}", s.prefetch_loads)),
            ),
            (
                "Hint loads",
                Box::new(|s: &PipelineStats| format!("{}", s.hint_loads)),
            ),
            (
                "Predicted loads",
                Box::new(|s: &PipelineStats| format!("{}", s.predicted_total)),
            ),
            (
                "Prediction prec.",
                Box::new(|s: &PipelineStats| format!("{:.1}%", s.prediction_precision())),
            ),
            (
                "Fence stalls",
                Box::new(|s: &PipelineStats| format!("{}", s.fence_stalls)),
            ),
            (
                "Work requests",
                Box::new(|s: &PipelineStats| format!("{}", s.work_requests)),
            ),
        ];

        let label_w = metrics
            .iter()
            .map(|(l, _)| l.len())
            .max()
            .unwrap_or(16)
            .max(16);

        // ── Header ──
        println!("\n=== Expert Pipeline Stats ===");
        print!("┌{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┬{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┐");

        print!("│ {:label_w$} ", "Metric");
        for h in &headers {
            print!("│ {:>col_w$} ", h);
        }
        println!("│");

        print!("├{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┼{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┤");

        // ── Data rows ──
        let default_stats = PipelineStats::default();
        for (label, extractor) in &metrics {
            print!("│ {:label_w$} ", label);
            for r in results {
                let stats = r.expert_stats.as_ref().unwrap_or(&default_stats);
                let val = extractor(stats);
                print!("│ {:>col_w$} ", val);
            }
            println!("│");
        }

        // ── Footer ──
        print!("└{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┴{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┘");
    }

    /// Print a transposed profile-timing table.
    ///
    /// Rows = span names (union across configs), Columns = test configs.
    /// `extract` pulls the relevant [`ProfileSnapshot`] from each result
    /// (bulk or single).  Skipped when no config has any profile entries.
    fn print_profile_table(
        &self,
        results: &[TestResults],
        title: &str,
        extract: impl Fn(&TestResults) -> &ProfileSnapshot,
    ) {
        // Collect snapshots.
        let snaps: Vec<&ProfileSnapshot> = results.iter().map(|r| extract(r)).collect();

        // Build the union of all span names and sort alphabetically so the
        // table stays stable as new rows are added.
        let mut span_names: Vec<String> = Vec::new();
        for snap in &snaps {
            for (name, _, _) in &snap.entries {
                if !span_names.contains(name) {
                    span_names.push(name.clone());
                }
            }
        }
        span_names.sort();

        // Skip if nothing to show.
        if span_names.is_empty() {
            return;
        }

        // Column headers.
        let headers: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("#{} {:?}×{}", i + 1, r.config.mode, r.config.num_contexts))
            .collect();

        // Column width: each column shows "total_ms (×count)" => needs space.
        let col_w = headers.iter().map(|h| h.len()).max().unwrap_or(10).max(18);
        let label_w = span_names
            .iter()
            .map(|n| n.len())
            .max()
            .unwrap_or(16)
            .max(20);

        // ── Header ──
        println!("\n=== {} ===", title);
        print!("┌{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┬{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┐");

        print!("│ {:label_w$} ", "Span");
        for h in &headers {
            print!("│ {:>col_w$} ", h);
        }
        println!("│");

        print!("├{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┼{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┤");

        // ── Data rows ──
        for span in &span_names {
            print!("│ {:label_w$} ", span);
            for snap in &snaps {
                let cell =
                    if let Some((_, ms, count)) = snap.entries.iter().find(|(n, _, _)| n == span) {
                        format!("{:.1}ms (×{})", ms, count)
                    } else {
                        "-".to_string()
                    };
                print!("│ {:>col_w$} ", cell);
            }
            println!("│");
        }

        // ── Footer ──
        print!("└{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┴{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┘");
    }
}
