//! Fluent builder for configuring and constructing a
//! [`ConversationEngine`](crate::ConversationEngine).

use super::{Model, ModelArch, ModelSpec};
use crate::config::{
    pick_max_hot_turns, DecodeHealthConfig, EngineConfig, SamplingConfig, SequenceConfig,
};
use crate::error::ConversationError;
use crate::models::DialectType;
use crate::tree::ConversationTreeConfig;
use candle::Device;
use candle_nn::kv_cache::QuantFormat;
use candle_nn::{arena_chunks_for_format, CHUNK_SIZE};
use candle_transformers::models::batched_model::BatchedInference;
use std::path::{Path, PathBuf};

// ────────────────────────────────────────────────────────────────────────────
// GGUF metadata extracted from the model file
// ────────────────────────────────────────────────────────────────────────────

/// Metadata extracted from a GGUF model file header.
///
/// Used by [`ModelBuilder::engine`] to auto-configure architecture, sampling,
/// vocab size, context length, and thinking support from the model file itself.
struct GgufInfo {
    /// `general.name` — human-readable model name.
    name: String,
    /// Detected [`ModelArch`] from `general.architecture`.
    arch: Option<ModelArch>,
    /// Recommended sampling defaults for this architecture.
    sampling: SamplingConfig,
    /// Non-thinking sampling variant (if model supports thinking).
    non_thinking: Option<SamplingConfig>,
    /// Vocab size from `output.weight` tensor shape.
    vocab_size: Option<usize>,
    /// `{arch}.context_length` — maximum sequence length.
    context_length: Option<usize>,
    /// Whether the chat template contains `<think>`.
    has_thinking: bool,
    /// Detected dialect from the chat template.
    dialect: Option<DialectType>,
}

// ────────────────────────────────────────────────────────────────────────────
// ModelBuilder
// ────────────────────────────────────────────────────────────────────────────

/// Fluent builder for configuring and constructing a
/// [`ConversationEngine`](crate::ConversationEngine).
///
/// Created via [`Model::builder`]. All setters return `self` for chaining.
///
/// ```ignore
/// let b = Model::Qwen3_8B_Q4.builder()
///     .temperature(0.8)
///     .top_p(0.95)
///     .max_response_tokens(16384)
///     .seed(123);
///
/// let engine = b.engine(&device)?;
/// let mut conv = engine.new_conversation(
///     &b.system_prompt(),
///     b.conversation_config(),
/// )?;
/// ```
#[derive(Debug, Clone)]
pub struct ModelBuilder {
    pub(super) spec: ModelSpec,
    /// Override: local GGUF model path (skips HF download).
    model_path: Option<PathBuf>,
    /// Override: system prompt text
    system_prompt: Option<String>,
    /// Override: local tokenizer.json path (skips HF download).
    tokenizer_path: Option<PathBuf>,
    /// Sampling configuration (initialised from [`ModelSpec::default_sampling`]).
    sampling: SamplingConfig,
    /// `true` if the user explicitly called any sampling setter on this builder.
    /// When `false`, [`engine()`](Self::engine) auto-detects defaults from the
    /// GGUF `general.architecture` metadata field.
    sampling_user_set: bool,
    /// Max tokens to generate per assistant turn.
    max_response_tokens: usize,
    /// KV cache sequence length.
    max_seq_len: usize,
    /// Engine-level: max concurrent conversations.
    max_concurrent: usize,
    /// Thinking mode override.
    ///
    /// - `None` (default) — no override; the model may or may not think.
    /// - `Some(true)` (`--thinking`) — explicitly enable thinking.
    /// - `Some(false)` (`--no-thinking`) — explicitly suppress thinking:
    ///   `/no_think` is prepended to the system prompt, an empty `<think></think>`
    ///   block is prefilled, and non-thinking sampling parameters are used.
    thinking: Option<bool>,
    /// KV adaptive compression level (0–9).  Defaults to C5.
    kv_compression_level: u8,
    /// Show special tokens in streamed output (debug/diagnostics).
    show_special_tokens: bool,
    /// Optional path to write penalty state to during decoding.
    penalty_log_path: Option<PathBuf>,
    /// Decode health monitoring configuration.
    health_config: DecodeHealthConfig,
    /// Maximum Hot-tier turns before triggering Hot → Warm eviction.
    /// `0` = auto-compute from arena geometry in [`engine()`](Self::engine).
    max_hot_turns: usize,
    /// Workspace root whose `.substrate/` directory backs the persistence
    /// redo log. `None` falls back to the process working directory.
    workspace_path: Option<PathBuf>,
}

impl ModelBuilder {
    /// Create a builder from a [`ModelSpec`], using its defaults.
    ///
    /// This is called automatically by [`Model::builder`] and
    /// [`Model::custom`], but can also be used directly.
    pub fn from_spec(spec: ModelSpec) -> Self {
        Self {
            sampling: spec.default_sampling.clone(),
            sampling_user_set: false,
            max_seq_len: spec.max_seq_len,
            max_response_tokens: 16384,
            max_concurrent: 4,
            system_prompt: None,
            model_path: None,
            tokenizer_path: None,
            thinking: None,
            kv_compression_level: 4,
            show_special_tokens: false,
            penalty_log_path: None,
            health_config: DecodeHealthConfig::default(),
            max_hot_turns: 0,
            workspace_path: None,
            spec,
        }
    }

    /// Set the workspace root whose `.substrate/` directory backs the
    /// persistence redo log.
    pub fn workspace_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.workspace_path = Some(path.into());
        self
    }

    // ── File overrides ─────────────────────────────────────────────────

    /// Override the GGUF model file path (skips HF download).
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Override the tokenizer.json file path (skips HF download).
    pub fn tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Set both model and tokenizer paths from a directory.
    ///
    /// Resolves to `<dir>/<model_filename>` (from the spec) and
    /// `<dir>/tokenizer.json`.
    pub fn model_dir(mut self, dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        self.model_path = Some(dir.join(&self.spec.model_filename));
        self.tokenizer_path = Some(dir.join("tokenizer.json"));
        self
    }

    /// Auto-detect model from a directory containing a `.gguf` file and
    /// `tokenizer.json`.
    ///
    /// Scans the directory for a single `.gguf` file, reads its GGUF
    /// metadata to determine architecture, dialect, sampling, context
    /// length, and thinking support.  Returns a fully configured builder
    /// that requires no `--model` preset.
    ///
    /// If the directory contains multiple `.gguf` files, the first one
    /// found (alphabetically) is used.
    ///
    /// ```ignore
    /// let builder = ModelBuilder::from_gguf_dir("/models/my-model")?;
    /// let engine = builder.engine(&device)?;
    /// ```
    pub fn from_gguf_dir(dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = dir.as_ref();

        // Find the .gguf file.
        let mut gguf_files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                ConversationError::Model(candle::Error::Msg(format!(
                    "read dir '{}': {}",
                    dir.display(),
                    e
                )))
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })
            .collect();

        gguf_files.sort_by_key(|e| e.file_name());

        let gguf_entry = gguf_files.first().ok_or_else(|| {
            ConversationError::Model(candle::Error::Msg(format!(
                "no .gguf file found in '{}'",
                dir.display()
            )))
        })?;
        let gguf_path = gguf_entry.path();
        let gguf_filename = gguf_entry.file_name().to_string_lossy().into_owned();

        tracing::info!("Auto-detected GGUF: {}", gguf_path.display());

        // Read metadata from the GGUF header.
        let info = Self::detect_sampling_from_gguf(&gguf_path)?;

        let arch = info.arch.unwrap_or(ModelArch::Llama);
        let dialect_type = info.dialect.unwrap_or(DialectType::ChatML);

        let spec = ModelSpec {
            arch,
            chat_format: dialect_type,
            dialect: dialect_type.dialect(),
            model_repo: String::new(),
            model_filename: gguf_filename,
            tokenizer_repo: String::new(),
            default_system_prompt: "You are a helpful, accurate, and concise assistant.".into(),
            max_seq_len: info.context_length.unwrap_or(8192),
            default_sampling: info.sampling.clone(),
            supports_thinking: info.has_thinking,
            inject_no_think_block: true,
            non_thinking_sampling: info.non_thinking.clone(),
        };

        let mut builder = Self::from_spec(spec);
        builder.model_path = Some(gguf_path);
        builder.tokenizer_path = Some(dir.join("tokenizer.json"));
        // The sampling came from GGUF detection, not user override.
        builder.sampling = info.sampling;
        builder.sampling_user_set = false;
        Ok(builder)
    }

    // ── Sampling ───────────────────────────────────────────────────────

    /// Set the full sampling configuration, replacing the model default.
    pub fn sampling(mut self, s: SamplingConfig) -> Self {
        self.sampling = s;
        self.sampling_user_set = true;
        self
    }

    /// Apply a named sampling preset, replacing the model default.
    ///
    /// See [`SamplingConfig::preset_names()`] for available presets.
    /// Panics if `name` is not a recognized preset.
    pub fn sampler_preset(mut self, name: &str) -> Self {
        self.sampling = SamplingConfig::preset(name)
            .unwrap_or_else(|| panic!("unknown sampler preset: '{}'", name));
        self.sampling_user_set = true;
        self
    }

    /// Adjust the temperature.
    ///
    /// Values <= 0 enable argmax (greedy) decoding.
    pub fn temperature(mut self, t: f32) -> Self {
        self.sampling.temperature = t;
        self.sampling_user_set = true;
        self
    }

    /// Set the KV adaptive compression level (0 = near-lossless, 9 = maximum).
    ///
    /// Defaults to C5 (moderate compression, minimal quality loss).
    pub fn compression_level(mut self, level: u8) -> Self {
        self.kv_compression_level = level;
        self
    }

    /// Configure decode health monitoring.
    ///
    /// Health checks are only active when the crate is built with
    /// `--features decode-health`. The config is always accepted; if the
    /// feature is absent the config is stored but has no effect.
    ///
    /// # Example
    /// ```ignore
    /// builder.health(DecodeHealthConfig::for_chat())
    /// ```
    pub fn health(mut self, config: DecodeHealthConfig) -> Self {
        self.health_config = config;
        self
    }

    /// Adjust top-p (nucleus sampling threshold).
    ///
    /// Values >= 1.0 disable top-p filtering.
    pub fn top_p(mut self, p: f32) -> Self {
        self.sampling.top_p = p;
        self.sampling_user_set = true;
        self
    }

    /// Adjust top-k filtering threshold.
    ///
    /// Values <= 0 disable top-k filtering.
    pub fn top_k(mut self, k: i32) -> Self {
        self.sampling.top_k = k;
        self.sampling_user_set = true;
        self
    }

    /// Set the repeat penalty for recently generated tokens.
    ///
    /// Values <= 1.0 disable repeat penalty.
    pub fn repeat_penalty(mut self, p: f32) -> Self {
        self.sampling.repeat_penalty = p;
        self.sampling_user_set = true;
        self
    }

    /// Set the frequency penalty.
    ///
    /// Values <= 0.0 disable frequency penalty.
    pub fn frequency_penalty(mut self, p: f32) -> Self {
        self.sampling.frequency_penalty = p;
        self.sampling_user_set = true;
        self
    }

    /// Set the presence penalty.
    ///
    /// Values <= 0.0 disable presence penalty.
    pub fn presence_penalty(mut self, p: f32) -> Self {
        self.sampling.presence_penalty = p;
        self.sampling_user_set = true;
        self
    }

    /// Set the RNG seed for sampling.
    pub fn seed(mut self, s: u64) -> Self {
        self.sampling.seed = s;
        self.sampling_user_set = true;
        self
    }

    // ── Generation limits ──────────────────────────────────────────────

    /// Maximum tokens to generate per assistant turn.
    pub fn max_response_tokens(mut self, n: usize) -> Self {
        self.max_response_tokens = n;
        self
    }

    /// Enable thinking/reasoning mode.
    ///
    /// When `true`, thinking models are allowed to produce `<think>` blocks.
    /// Has no effect on models where
    /// [`supports_thinking`](ModelSpec::supports_thinking) is `false`.
    pub fn thinking(mut self, enable: bool) -> Self {
        self.thinking = Some(enable);
        self
    }

    /// Whether thinking suppression is explicitly active.
    ///
    /// Returns `true` only when the user has explicitly passed
    /// `thinking(false)` (i.e. `--no-thinking`).  In this mode,
    /// `/no_think` is prepended to the system prompt, an empty `<think></think>`
    /// block is prefilled, and non-thinking sampling parameters are used.
    pub fn should_suppress_thinking(&self) -> bool {
        self.thinking == Some(false)
    }

    /// Enable or disable automatic boundary-injection KV prototype generation.
    ///
    /// Show special tokens in streamed output.
    ///
    /// When `true`, tokens like `<think>`, `</think>`, and `<|im_end|>`
    /// are included verbatim in the text stream instead of being stripped.
    /// Useful for debugging prompt/response formatting.
    pub fn show_special_tokens(mut self, enable: bool) -> Self {
        self.show_special_tokens = enable;
        self
    }

    /// Path to write penalty state log during decoding (for debugging).
    pub fn penalty_log(mut self, path: impl Into<PathBuf>) -> Self {
        self.penalty_log_path = Some(path.into());
        self
    }

    /// Maximum sequence length (KV cache allocation).
    pub fn max_seq_len(mut self, n: usize) -> Self {
        self.max_seq_len = n;
        self
    }

    /// Maximum concurrent conversations on the engine.
    pub fn max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = n;
        self
    }

    /// Override the auto-computed Hot → Warm eviction limit.
    ///
    /// By default, [`engine()`](Self::engine) derives `max_hot_turns` from
    /// the KV arena size and `max_response_tokens` via [`pick_max_hot_turns`].
    /// Pass a non-zero value here to fix the limit manually.
    /// `0` re-enables auto-computation.
    pub fn max_hot_turns(mut self, n: usize) -> Self {
        self.max_hot_turns = n;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Read-back & terminal methods
// ────────────────────────────────────────────────────────────────────────────

impl ModelBuilder {
    /// Access the underlying model specification.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Build a [`SequenceConfig`] reflecting all builder overrides.
    pub fn conversation_config(&self) -> SequenceConfig {
        // When thinking is suppressed and the user hasn't explicitly overridden
        // sampling, use the model's non-thinking sampling params if available.
        let sampling = if self.should_suppress_thinking() && !self.sampling_user_set {
            self.spec
                .non_thinking_sampling
                .clone()
                .unwrap_or_else(|| self.sampling.clone())
        } else {
            self.sampling.clone()
        }
        .with_max_response_tokens(self.max_response_tokens);
        SequenceConfig {
            dialect: self.spec.dialect.clone(),
            max_response_tokens: self.max_response_tokens,
            sampling,
            suppress_thinking: self.should_suppress_thinking(),
            thinking_capable: self.spec.supports_thinking,
            inject_no_think_block: self.spec.inject_no_think_block,
            tree: ConversationTreeConfig::default(),
            // 0 = incremental KV accumulation: only new tokens are prefilled each
            // turn and the KV cache is preserved across turns.  The previous value
            // of 8 triggered a full-context reset+rebuild on every turn, causing
            // large growing prefills and 2-4× excess intermediate activation memory.
            context_window_turns: 0,
            max_hot_turns: self.max_hot_turns,
            // Re-project every two sealed chunks during decode.  This is
            // the cadence the substrate-driven attention-wandering design
            // is built around — disabling it would make multi-turn
            // recall fall back to whatever ranges the scheduler's
            // initial `Builder::project()` happened to compute at
            // submit time.
            reproject_every_n_tokens: 64,
            reproject_max_probe_tokens: 64,
            // Linefeed is the most reliable paragraph/section break
            // signal across chat templates and content styles.
            reproject_trigger_texts: vec!["\n".to_string()],
        }
    }

    pub fn system_prompt<S: Into<String>>(mut self, prompt: S) -> ModelBuilder {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// The default system prompt, pre-formatted for this model's chat template.
    ///
    /// When thinking is explicitly suppressed (`--no-thinking`), `/no_think`
    /// is prepended before the system prompt text to switch the model's
    /// behavioural mode away from analytical reasoning.
    pub fn format_system_prompt(&self) -> String {
        let base = if let Some(prompt) = &self.system_prompt {
            prompt.as_str()
        } else {
            &self.spec.default_system_prompt
        };
        if self.should_suppress_thinking() {
            let with_no_think = format!("/no_think\n{base}");
            self.spec.dialect.format_system_prompt(&with_no_think)
        } else {
            self.spec.dialect.format_system_prompt(base)
        }
    }

    /// Build an [`EngineConfig`] from the builder's settings and a tokenizer.
    pub fn engine_config(&self, tokenizer: &tokenizers::Tokenizer) -> EngineConfig {
        let mut eos_tokens = Vec::new();
        for token in [
            "<|im_end|>",
            "<|endoftext|>",
            "<|end_of_text|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|reserved_special_token_31|>",
        ] {
            if let Some(id) = tokenizer.token_to_id(token) {
                tracing::trace!("✅ EOS token registered: '{}' = ID {}", token, id);
                eos_tokens.push(id);
            } else {
                tracing::debug!("⚠️  EOS token NOT FOUND in tokenizer: '{}'", token);
            }
        }
        if eos_tokens.is_empty() {
            tracing::error!("No EOS tokens found!");
        }

        // Auto-detect vocab size from tokenizer (includes special tokens).
        // This MUST match the logits dimension from the model, otherwise
        // penalty buffers will be undersized and the CUDA kernel will
        // read out-of-bounds GPU memory.
        let tok_vocab = tokenizer.get_vocab(true).len();
        // Qwen3 (and several other models) ship a tokenizer vocab smaller
        // than the model's output projection (e.g. 151_669 vs 151_936).
        // The model's logits dimension is authoritative — read it from the
        // GGUF header when we can resolve the model path; fall back to the
        // tokenizer length otherwise.
        let vocab_size = if let Ok((mp, _)) = self.resolve_paths_pub() {
            match Self::detect_sampling_from_gguf(&mp) {
                Ok(info) => info.vocab_size.unwrap_or(tok_vocab).max(tok_vocab),
                Err(_) => tok_vocab,
            }
        } else {
            tok_vocab
        };
        tracing::info!(
            "Vocab size: tokenizer={} -> using={}",
            tok_vocab,
            vocab_size
        );

        let mut ret = EngineConfig::new(eos_tokens.into());
        ret.batched_config.compression_level = Some(self.kv_compression_level);
        // Until decode's K-side honors non-identity pal_map with non-unit
        // outer scales, force K storage to uniform Q4_KS with identity
        // pal_map and unit scales for every conversation consumer. V keeps
        // full selection adaptivity. Drop the override once the decode
        // kernel's K-path is fixed.
        ret.batched_config.override_k_quant = Some(QuantFormat::Q4_KS);
        ret.batched_config.override_v_quant = None;
        ret.vocab_size = vocab_size;
        ret.max_concurrent_conversations = self.max_concurrent;
        ret.show_special_tokens = self.show_special_tokens;
        ret.penalty_log_path = self.penalty_log_path.clone();
        ret.health = self.health_config.clone();
        ret.workspace_path = self.workspace_path.clone();
        ret.model_spec = Some(self.model_spec_blob());
        // The engine uses the model's dialect to pre-tokenise the
        // inter-turn boundary markers once at scheduler construction.
        ret.dialect = self.spec.dialect.clone();
        ret
    }

    /// A stable serialized identity for the substrate's `ModelSpec` record —
    /// enough to re-load the same weights and tokenizer from HuggingFace.
    fn model_spec_blob(&self) -> Vec<u8> {
        let s = &self.spec;
        format!(
            "candle-conversation model-spec v1\n\
             arch={:?}\n\
             chat_format={:?}\n\
             model_repo={}\n\
             model_filename={}\n\
             tokenizer_repo={}\n\
             max_seq_len={}\n",
            s.arch, s.chat_format, s.model_repo, s.model_filename, s.tokenizer_repo, s.max_seq_len,
        )
        .into_bytes()
    }

    /// Load quantised model weights from a local GGUF file.
    ///
    /// Uses the builder's `max_seq_len` for KV cache sizing.
    pub fn load_model(
        &self,
        model_path: &Path,
        device: &Device,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> crate::Result<Box<dyn crate::ManagedBatchedModel + Send>> {
        let max_seq = self.max_seq_len;
        match self.spec.arch {
            ModelArch::Qwen3 => {
                use candle_transformers::models::quantized_qwen3::ModelWeights;
                // Per-layer progress not yet wired for this arch — the
                // callback simply doesn't fire. Add `progress` to the
                // arch's loader to enable it.
                let _ = progress;
                let raw = ModelWeights::from_gguf_by_path(model_path, device)?;
                let inv = raw.rope_inv_freq().ok_or_else(|| {
                    ConversationError::Model(candle::Error::Msg(
                        "model missing rope inv_freq".into(),
                    ))
                })?;
                Ok(Box::new(BatchedInference::new_with_inv_freq(
                    raw, inv, max_seq, device,
                )?))
            }
            ModelArch::Qwen3Moe => {
                use candle_transformers::models::quantized_qwen3_moe::ModelWeights;
                let raw = ModelWeights::from_gguf_by_path(model_path, device, progress)?;
                let inv = raw.rope_inv_freq().ok_or_else(|| {
                    ConversationError::Model(candle::Error::Msg(
                        "model missing rope inv_freq".into(),
                    ))
                })?;
                Ok(Box::new(BatchedInference::new_with_inv_freq(
                    raw, inv, max_seq, device,
                )?))
            }
            ModelArch::Qwen2 => {
                use candle_transformers::models::quantized_qwen2::ModelWeights;
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                let raw = ModelWeights::from_gguf_by_path(model_path, device)?;
                let inv = raw.rope_inv_freq().ok_or_else(|| {
                    ConversationError::Model(candle::Error::Msg(
                        "model missing rope inv_freq".into(),
                    ))
                })?;
                Ok(Box::new(BatchedInference::new_with_inv_freq(
                    raw, inv, max_seq, device,
                )?))
            }
            ModelArch::Llama => {
                use candle_transformers::models::quantized_llama::ModelWeights;
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                let raw = ModelWeights::from_gguf_by_path(model_path, device)?;
                let inv = raw.rope_inv_freq().ok_or_else(|| {
                    ConversationError::Model(candle::Error::Msg(
                        "model missing rope inv_freq".into(),
                    ))
                })?;
                Ok(Box::new(BatchedInference::new_with_inv_freq(
                    raw, inv, max_seq, device,
                )?))
            }
        }
    }

    /// Resolve file paths, load model and tokenizer, and build the engine.
    ///
    /// If [`model_path`](Self::model_path) / [`tokenizer_path`](Self::tokenizer_path)
    /// (or [`model_dir`](Self::model_dir)) have been set, those local files
    /// are used. Otherwise, the files are downloaded from HuggingFace
    /// (requires the `hub` crate feature).
    pub fn engine(&mut self, device: &Device) -> crate::Result<crate::ConversationEngine> {
        self.engine_with_progress(device, None)
    }

    /// Same as [`Self::engine`] but accepts an optional per-layer
    /// progress callback. The callback is invoked as
    /// `(layers_loaded, total_layers)` after each transformer block's
    /// weights are mounted — enables a UI progress bar without
    /// coupling this builder to the daemon's progress type.
    pub fn engine_with_progress(
        &mut self,
        device: &Device,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> crate::Result<crate::ConversationEngine> {
        let (model_path, tokenizer_path) = self.resolve_paths()?;

        // ── Read GGUF metadata ──────────────────────────────────────────
        // Always read the GGUF header for arch, vocab_size, context_length,
        // and thinking detection.  Only override sampling when the user
        // hasn't explicitly set it via --sampler or individual args.
        let mut gguf_vocab_size = None;
        let mut gguf_has_thinking = None;
        match Self::detect_sampling_from_gguf(&model_path) {
            Ok(info) => {
                gguf_vocab_size = info.vocab_size;
                gguf_has_thinking = Some(info.has_thinking);

                // Override model display name from GGUF `general.name`.
                self.spec.model_filename = format!("{}.gguf", info.name);

                // Override ModelArch from GGUF — ensures the correct weight
                // loader is used even when --model-dir points to a different
                // architecture than the --model preset.
                if let Some(arch) = info.arch {
                    if arch != self.spec.arch {
                        tracing::info!(
                            "Overriding ModelArch: preset={:?} → GGUF={:?}",
                            self.spec.arch,
                            arch
                        );
                        self.spec.arch = arch;
                    }
                }

                // Override max_seq_len from GGUF context_length.
                if let Some(ctx) = info.context_length {
                    if ctx != self.max_seq_len {
                        tracing::info!(
                            "Overriding max_seq_len: preset={} → GGUF={}",
                            self.max_seq_len,
                            ctx
                        );
                        self.max_seq_len = ctx;
                    }
                }

                if !self.sampling_user_set {
                    tracing::info!(
                        "Auto-detected sampling from GGUF: temp={}, top_k={}, top_p={}, repeat_penalty={}",
                        info.sampling.temperature, info.sampling.top_k, info.sampling.top_p, info.sampling.repeat_penalty
                    );
                    self.sampling = info.sampling;
                    if let Some(ref nt) = info.non_thinking {
                        tracing::info!(
                            "Auto-detected non-thinking sampling: temp={}, top_k={}, top_p={}",
                            nt.temperature,
                            nt.top_k,
                            nt.top_p
                        );
                    }
                    self.spec.non_thinking_sampling = info.non_thinking;
                }
            }
            Err(e) => {
                tracing::warn!("Could not read GGUF metadata: {e}, using preset defaults");
            }
        }

        let tokenizer = Model::load_tokenizer(&tokenizer_path)?;

        // ── Auto-detect thinking support from chat_template ───────────
        // The GGUF chat_template is the authoritative signal.  Many models
        // (e.g. non-thinking Qwen3 finetunes) have the <think> token in
        // the tokenizer but never use it — those are NOT thinking models.
        // Fall back to tokenizer check only when GGUF detection was skipped.
        let detected_thinking =
            gguf_has_thinking.unwrap_or_else(|| tokenizer.token_to_id("<think>").is_some());
        if detected_thinking != self.spec.supports_thinking {
            tracing::info!(
                "Thinking support overridden: preset={}, detected={}",
                self.spec.supports_thinking,
                detected_thinking
            );
            self.spec.supports_thinking = detected_thinking;
        }
        if self.spec.supports_thinking {
            let mode = match self.thinking {
                Some(true) => "enabled (--thinking)",
                Some(false) => "suppressed (--no-thinking)",
                None => "default (model decides)",
            };
            tracing::info!("Model supports thinking: {}", mode);
        }

        // ── Resolve <think>/<\u200b/think> token IDs from tokenizer ───────────
        // Patch the thinking token IDs into all sampling configs that have
        // EOT boost configured.  This replaces the placeholder -1 values.
        self.sampling.resolve_thinking_tokens(&tokenizer);
        if let Some(ref mut nt) = self.spec.non_thinking_sampling {
            nt.resolve_thinking_tokens(&tokenizer);
        }
        self.health_config.resolve_structural_tokens(&tokenizer);

        let mut config = self.engine_config(&tokenizer);

        // Embed the raw `tokenizer.json` so the substrate log is a
        // self-contained, offline-detokenizable image. Written once per
        // distinct model via compare-and-insert at engine startup.
        match std::fs::read(&tokenizer_path) {
            Ok(bytes) => config.tokenizer = Some(bytes),
            Err(e) => tracing::warn!(
                "could not read tokenizer.json ({}) for persistence: {e}",
                tokenizer_path.display()
            ),
        }

        // Override vocab_size with the authoritative value from GGUF metadata
        // if available.  Models often pad vocab to a power-of-2 / multiple of
        // 128 for GPU efficiency, so tokenizer.get_vocab().len() can be smaller
        // than the model's actual logits dimension.
        if let Some(model_vocab) = gguf_vocab_size {
            if model_vocab != config.vocab_size {
                tracing::info!(
                    "Overriding vocab_size: tokenizer={} → GGUF metadata={}",
                    config.vocab_size,
                    model_vocab
                );
            }
            config.vocab_size = model_vocab;
        }

        // ── Log effective configuration before loading weights ─────────
        {
            let model_name = self
                .spec
                .model_filename
                .strip_suffix(".gguf")
                .unwrap_or(&self.spec.model_filename);
            tracing::info!("Engine config · {}", model_name);
            tracing::info!(
                "  Arch: {:?}   vocab: {}   max_seq: {}",
                self.spec.arch,
                config.vocab_size,
                self.max_seq_len
            );
            tracing::info!(
                "  Sampling: temp={:.3}  top_k={}  top_p={:.3}  repeat_penalty={:.3}",
                self.sampling.temperature,
                self.sampling.top_k,
                self.sampling.top_p,
                self.sampling.repeat_penalty,
            );
            tracing::info!("  Max response tokens: {}", self.max_response_tokens);
        }

        let model = self.load_model(&model_path, device, progress)?;

        // Auto-derive max_hot_turns from arena geometry unless the caller
        // overrode it. Must happen before conversation_config() is called below.
        if self.max_hot_turns == 0 {
            let arena_chunks =
                arena_chunks_for_format(candle_nn::kv_cache::KvFormat::Float(candle::DType::F16));
            self.max_hot_turns =
                pick_max_hot_turns(arena_chunks, CHUNK_SIZE, self.max_response_tokens);
            tracing::debug!(
                "auto max_hot_turns: {} (arena {} × chunk {} / {} max_response_tokens)",
                self.max_hot_turns,
                arena_chunks,
                CHUNK_SIZE,
                self.max_response_tokens,
            );
        }

        let engine = crate::ConversationEngine::new(model, tokenizer, config)?;
        Ok(engine)
    }

    /// Read the GGUF header to extract model metadata.
    ///
    /// Returns architecture, sampling defaults, vocab_size, thinking support,
    /// model name, and context length — everything needed to configure the
    /// builder from the GGUF file itself.
    fn detect_sampling_from_gguf(model_path: &Path) -> crate::Result<GgufInfo> {
        use candle::quantized::gguf_file;
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| ConversationError::Model(candle::Error::Msg(format!("open GGUF: {e}"))))?;
        let ct = gguf_file::Content::read(&mut file).map_err(ConversationError::Model)?;
        let arch_str = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let name = ct
            .metadata
            .get("general.name")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_else(|| "<unknown>".into());
        tracing::info!("GGUF model: '{}', architecture: '{}'", name, arch_str);

        // ── Detect thinking support + dialect from chat_template ──────
        let chat_template = ct
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok())
            .cloned();
        let has_thinking = chat_template
            .as_deref()
            .map(|tmpl| tmpl.contains("<think>"))
            .unwrap_or(false);
        let dialect = chat_template.as_deref().and_then(|tmpl| {
            if tmpl.contains("<|im_start|>") {
                Some(DialectType::ChatML)
            } else if tmpl.contains("<|start_header_id|>") {
                Some(DialectType::Llama3)
            } else if tmpl.contains("[INST]") {
                Some(DialectType::Llama2)
            } else {
                None
            }
        });
        tracing::info!(
            "GGUF chat_template thinking={}, dialect={:?}",
            has_thinking,
            dialect
        );

        // Map GGUF architecture string to ModelArch for weight loading.
        let detected_arch = match arch_str.as_str() {
            "qwen3" => Some(ModelArch::Qwen3),
            "qwen3moe" | "qwen2moe" => Some(ModelArch::Qwen3Moe),
            "qwen2" => Some(ModelArch::Qwen2),
            "llama" => Some(ModelArch::Llama),
            _ => None,
        };

        // Read context_length from `{arch}.context_length` metadata.
        // Fall back to config.json `max_position_embeddings` if the GGUF key is absent.
        let context_length_key = format!("{}.context_length", arch_str);
        let gguf_context_length = ct
            .metadata
            .get(&context_length_key)
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let cfg_json_context_length: Option<usize> = model_path.parent().and_then(|d| {
            let text = std::fs::read_to_string(d.join("config.json")).ok()?;
            let v: serde_json::Value = serde_json::from_str(&text).ok()?;
            v.get("max_position_embeddings")?
                .as_u64()
                .map(|n| n as usize)
        });
        let context_length = gguf_context_length.or(cfg_json_context_length);
        if let Some(ctx) = gguf_context_length {
            tracing::info!("GGUF {}: {}", context_length_key, ctx);
        } else if let Some(ctx) = cfg_json_context_length {
            tracing::info!("config.json max_position_embeddings: {ctx}");
        }

        // Read vocab_size from the output projection tensor shape: [vocab_size, hidden_size].
        // Models with tied embeddings (e.g. Qwen2-0.5B) omit `output.weight`
        // and reuse `token_embd.weight`, which is padded to a GPU-friendly
        // multiple of 128 — so we must use the tensor shape, not the tokenizer
        // vocab count, to get the true logits dimension.
        let vocab_size = ct
            .tensor_infos
            .get("output.weight")
            .or_else(|| ct.tensor_infos.get("token_embd.weight"))
            .map(|ti| ti.shape.dims()[0]);

        let sampling = SamplingConfig::for_gguf_architecture(&arch_str);
        let non_thinking = SamplingConfig::non_thinking_for_gguf_architecture(&arch_str);
        Ok(GgufInfo {
            name,
            arch: detected_arch,
            sampling,
            non_thinking,
            vocab_size,
            context_length,
            has_thinking,
            dialect,
        })
    }

    // ── Internal ───────────────────────────────────────────────────────

    fn resolve_paths(&self) -> crate::Result<(PathBuf, PathBuf)> {
        self.resolve_paths_pub()
    }

    /// Public accessor for the same path-resolution logic as
    /// [`Self::resolve_paths`]. Used by external benchmarks (e.g. the RULER
    /// streamer) that need the resolved tokenizer path before calling
    /// [`Self::build`].
    pub fn resolve_paths_pub(&self) -> crate::Result<(PathBuf, PathBuf)> {
        match (&self.model_path, &self.tokenizer_path) {
            (Some(m), Some(t)) => Ok((m.clone(), t.clone())),
            (Some(_), None) | (None, Some(_)) => Err(ConversationError::Download(
                "set both model_path and tokenizer_path, or neither (to auto-download)".into(),
            )),
            (None, None) => self.download_or_fail(),
        }
    }

    #[cfg(feature = "hub")]
    fn download_or_fail(&self) -> crate::Result<(PathBuf, PathBuf)> {
        use hf_hub::api::sync::Api;

        let api = Api::new().map_err(|e| ConversationError::Download(e.to_string()))?;

        let model_path = api
            .model(self.spec.model_repo.clone())
            .get(&self.spec.model_filename)
            .map_err(|e| ConversationError::Download(e.to_string()))?;

        let tokenizer_path = api
            .model(self.spec.tokenizer_repo.clone())
            .get("tokenizer.json")
            .map_err(|e| ConversationError::Download(e.to_string()))?;

        Ok((model_path, tokenizer_path))
    }

    #[cfg(not(feature = "hub"))]
    fn download_or_fail(&self) -> crate::Result<(PathBuf, PathBuf)> {
        Err(ConversationError::Download(
            "local paths not set; enable the 'hub' feature to download from HuggingFace, \
             or call .model_path()/.tokenizer_path() / .model_dir()"
                .into(),
        ))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Display
// ────────────────────────────────────────────────────────────────────────────

impl std::fmt::Display for ModelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self
            .spec
            .model_filename
            .strip_suffix(".gguf")
            .unwrap_or(&self.spec.model_filename);
        write!(f, "{name}")
    }
}
