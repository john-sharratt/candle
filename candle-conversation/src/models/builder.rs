//! Fluent builder for configuring and constructing a
//! [`ConversationEngine`](crate::ConversationEngine).

use super::{Model, ModelArch, ModelSpec};
use crate::config::{
    pick_max_hot_turns, DecodeHealthConfig, EngineConfig, SamplingConfig, SequenceConfig,
};
use crate::error::ConversationError;
use crate::models::DialectType;
use crate::projection::{CorruptTurnPolicy, LayerId};
use crate::tree::ConversationTreeConfig;
use candle::{DType, Device};
use candle_nn::kv_cache::{class_for_format, elems_per_chunk, KvFormat, SizeClass, N_PALETTE};
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_model::BatchedInference;
use std::collections::HashMap;
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
    /// When `true`, the engine does not spawn the async summariser thread and
    /// new conversations are not registered for summarisation (the AVL summary
    /// forest is left un-extended). Off by default.
    disable_summariser: bool,
    /// Per-layer corrupt-turn policy (from the projection schema), forwarded to
    /// [`EngineConfig::layer_corrupt_turn`] so the startup reload drops the whole
    /// conversation (ingest layers) or just the turn (dialogue) per layer. Empty
    /// by default ⇒ every layer defaults to `DropConversation`.
    layer_corrupt_turn: HashMap<LayerId, CorruptTurnPolicy>,
    /// Directory for the persistent repacked expert pack.
    ///
    /// `None` (the default) uses a temp file, unlinked as soon as it is open, so
    /// nothing is left on disk and the repack is paid on every start.
    expert_pack_dir: Option<PathBuf>,
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
            kv_compression_level: 5,
            show_special_tokens: false,
            penalty_log_path: None,
            health_config: DecodeHealthConfig::default(),
            max_hot_turns: 0,
            workspace_path: None,
            disable_summariser: false,
            layer_corrupt_turn: HashMap::new(),
            expert_pack_dir: None,
            spec,
        }
    }

    /// Keep the repacked expert pack in `dir` instead of a temp file.
    ///
    /// The pack is the expert cache's cold tier: every expert, in the layout the
    /// kernels consume, so an eviction from VRAM can be a drop rather than a
    /// copy (`docs/expert_cache_design.md`). It is a pure function of the
    /// checkpoint, so a persistent one lets a restart skip the ~42 s repack and
    /// map straight to serving.
    ///
    /// The natural argument is the GGUF's own directory: one pack is then shared
    /// by every workspace using that checkpoint, it survives a substrate wipe,
    /// and it is deleted by the same act that deletes the model. Unset, the pack
    /// goes to a temp file that is unlinked the moment it is open — an embedder,
    /// an example or a test must never have a 16.6 GiB file appear beside its
    /// model without asking.
    ///
    /// MoE-only; the other architectures have no expert cache and ignore it.
    pub fn expert_pack_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.expert_pack_dir = Some(dir.into());
        self
    }

    /// Set the per-layer corrupt-turn policy map (from the projection schema).
    /// Forwarded to the engine so the startup reload applies the right policy per
    /// layer — `drop_conversation` for ingest layers, `drop_turn` for dialogue.
    pub fn corrupt_turn_policies(
        mut self,
        policies: HashMap<LayerId, CorruptTurnPolicy>,
    ) -> ModelBuilder {
        self.layer_corrupt_turn = policies;
        self
    }

    /// Disable the background summariser thread (and the per-conversation
    /// summarisation registration). Use to bring the engine up without the
    /// AVL summary forest running — e.g. for bulk corpus prefill.
    pub fn disable_summariser(mut self, disable: bool) -> Self {
        self.disable_summariser = disable;
        self
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
        let model_bytes = std::fs::metadata(&gguf_path)
            .map_err(|e| ConversationError::Model(candle::Error::Msg(format!("{e}"))))?
            .len();

        let arch = info.arch.unwrap_or(ModelArch::Llama);
        let dialect_type = info.dialect.unwrap_or(DialectType::ChatML);

        let spec = ModelSpec {
            arch,
            chat_format: dialect_type,
            dialect: dialect_type.dialect(),
            model_repo: String::new(),
            model_filename: gguf_filename,
            model_bytes,
            tokenizer_repo: String::new(),
            // A custom model is built from local files; there is no repo to
            // pin a revision of.
            tokenizer_rev: String::new(),
            default_system_prompt: "You are a helpful, accurate, and concise assistant.".into(),
            max_seq_len: info.context_length.unwrap_or(8192),
            default_sampling: info.sampling.clone(),
            supports_thinking: info.has_thinking,
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
            // The belief probe covers the last N decode-window tokens. 256 spans
            // a full tool-call turn's discriminative span; the §80.2 sweep shows
            // the true tool is always within Top-3 at this width (100% recall at
            // budget 3), where a 64-token tail needed budget 5. See docs §80.2.
            reproject_max_probe_tokens: 256,
            // Linefeed is the most reliable paragraph/section break
            // signal across chat templates and content styles.
            reproject_trigger_texts: vec!["\n".to_string()],
            disable_reprojection: false,
            // Default: turns use the engine-wide compression level. Utility
            // layers override these per-conversation (e.g. code_reading → C8,
            // K override dropped) via `repo_scan::utility_config` callers.
            kv_compression_level: None,
            kv_disable_k_override: false,
            kv_force_k_format: None,
            kv_force_v_format: None,
            kv_lossless: false,
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
        // The spec's dialect is the source of truth for the model family's
        // end-of-turn/document markers (e.g. DeepSeek's `<｜end▁of▁sentence｜>`);
        // the fixed list covers the families whose dialects predate it.
        let dialect_ends = [
            self.spec.dialect.document_end,
            self.spec.dialect.turn_end,
            self.spec.dialect.assistant_end,
        ];
        for token in dialect_ends
            .iter()
            .copied()
            .filter(|s| !s.is_empty())
            .chain([
                "<|im_end|>",
                "<|endoftext|>",
                "<|end_of_text|>",
                "<|eom_id|>",
                "<|eot_id|>",
                "<|reserved_special_token_31|>",
            ])
        {
            if let Some(id) = tokenizer.token_to_id(token) {
                if !eos_tokens.contains(&id) {
                    tracing::trace!("✅ EOS token registered: '{}' = ID {}", token, id);
                    eos_tokens.push(id);
                }
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
        ret.disable_summariser = self.disable_summariser;
        ret.layer_corrupt_turn = self.layer_corrupt_turn.clone();
        ret.batched_config.compression_level = Some(self.kv_compression_level);
        // Stress test: uniform-K pin REMOVED — both K and V now use fully
        // adaptive per-(head,palette) selection with non-identity pal_maps,
        // exercising the palette-straddle decode path on BOTH sides.
        // (Production pins K to a uniform Q8_KS/identity layout because the
        // provenance recall K-signature reader assumes it; adaptive K measurably
        // degrades recall. Restore `Some(QuantFormat::Q8_KS)` if recall
        // regresses — the duplication control asserts on verbatim reproduction,
        // not recall quality, so this isolates the straddle path.)
        ret.batched_config.override_k_quant = None;
        ret.batched_config.override_v_quant = None;
        // The width this architecture's own reference implementation computes
        // in, where it is known. Without it the session derives the activation
        // width from the KV storage format — which reports F16 for the
        // quantized backing this engine always uses, regardless of what the
        // model was trained and published in. See
        // `ModelArch::native_activation_dtype`.
        ret.batched_config.activation_dtype = self.spec.arch.native_activation_dtype();
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
                use candle_transformers::models::quantized_qwen3_moe::{
                    GgufLoadOptions, ModelWeights,
                };
                // The only arch with an expert cache, so the only one the pack
                // directory reaches.
                let raw = ModelWeights::from_gguf_with_options(
                    model_path,
                    device,
                    progress,
                    GgufLoadOptions {
                        int8mode: None,
                        expert_pack_dir: self.expert_pack_dir.clone(),
                    },
                )?;
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
                // The deployed Llama is the 3.x family (Llama-3.2-3B), so the
                // v3 KV-factor row applies; a Llama-2 checkpoint needs its own
                // `ModelArch` split onto `from_gguf_by_path_v2` before it can
                // load through the daemon.
                let raw = ModelWeights::from_gguf_by_path_v3(model_path, device)?;
                let inv = raw.rope_inv_freq().ok_or_else(|| {
                    ConversationError::Model(candle::Error::Msg(
                        "model missing rope inv_freq".into(),
                    ))
                })?;
                Ok(Box::new(BatchedInference::new_with_inv_freq(
                    raw, inv, max_seq, device,
                )?))
            }
            ModelArch::DeepSeekV4 => {
                use candle::quantized::Int8Mode;
                use candle_transformers::models::deepseek4::DEEPSEEK_V4;
                use candle_transformers::models::latent_moe::{BatchedEngine, Engine};
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                let _ = max_seq; // window/corpus budgets are model-derived
                let engine = Engine::load(model_path, &DEEPSEEK_V4, device, Int8Mode::Performance)
                    .map_err(ConversationError::Model)?;
                Ok(Box::new(
                    BatchedEngine::new(engine).map_err(ConversationError::Model)?,
                ))
            }
            ModelArch::Qwen35Hybrid => {
                use candle_transformers::models::quantized_qwen36_moe;
                use candle_transformers::models::qwen35::Qwen35LoadOptions;
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                // KV is allocated per ATTENTION layer, not per transformer layer,
                // and the window budget is derived from the config — see
                // `qwen35::engine::create_session`.
                let _ = max_seq;
                let model = quantized_qwen36_moe::from_gguf_path(
                    model_path,
                    device,
                    Qwen35LoadOptions {
                        // Without a directory the pack is EPHEMERAL — written to
                        // the system temp dir and unlinked as soon as it is
                        // published, so every boot repacks all 41 layers (53 s
                        // measured on the 3.6-35B) instead of reading the one
                        // beside the checkpoint.
                        expert_pack_dir: self.expert_pack_dir.clone(),
                        ..Default::default()
                    },
                )
                .map_err(ConversationError::Model)?;
                Ok(Box::new(model))
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

        // Before anything reads a token id: the checkpoint is the authority on
        // what its own ids mean, so hold the tokenizer against it here rather
        // than trusting the path it was resolved from.
        Self::verify_tokenizer_matches_checkpoint(&tokenizer, &model_path)?;

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

        // Hand the load's pool high-water back before serving starts — it is
        // several GiB held outside the KV reservation and never used again.
        // See `candle::vram::trim_pool_after_load` for why here and nowhere
        // later.
        if let Some((before, after)) = candle::vram::trim_pool_after_load(device) {
            tracing::info!(
                reclaimed_mib = before.saturating_sub(after) >> 20,
                pool_reserved_mib = after >> 20,
                "post-load: returned the load's pool high-water to the OS"
            );
        }

        // Auto-derive max_hot_turns from arena geometry unless the caller
        // overrode it. Must happen before conversation_config() is called below.
        if self.max_hot_turns == 0 {
            // A hot turn's active K/V is F16, so the F16 size class is what
            // bounds how many turns a region holds. The class is a function of
            // the *model's* palette sub-band width — `head_dim / N_PALETTE` —
            // not of `CHUNK_SIZE`; those coincide only at `head_dim == 128`.
            let head_dim = model.model_core_properties().head_dim;
            let elems = elems_per_chunk((head_dim / N_PALETTE).max(1));
            let class = class_for_format(KvFormat::Float(DType::F16), elems).ok_or_else(|| {
                ConversationError::Other(format!(
                    "no size class holds an F16 chunk of {elems} elements \
                     (head_dim {head_dim}); the ladder's top rung is \
                     {} bytes",
                    SizeClass::at(SizeClass::COUNT - 1).bytes(),
                ))
            })?;
            let arena_chunks = class.chunks_per_region();
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
    /// Check the loaded tokenizer against the checkpoint's **own** token table.
    ///
    /// `tokenizer.ggml.tokens` is the vocabulary the embedding and `lm_head`
    /// rows were built against, so it — not any repo, pin or filename — is what
    /// makes a `tokenizer.json` the right one. The GGUF ships the token list but
    /// no merges or pretokenizer, which is why a separate `tokenizer.json` is
    /// loaded at all; that copy has until now been trusted on the strength of a
    /// pinned repo constant and a comment asserting the two agree. Nothing
    /// re-established it at runtime, and zend does not pin revisions, so a
    /// wrong-but-loadable tokenizer was accepted in silence: ids stayed in
    /// range, decoded to real pieces, and the model was fed fluent-looking
    /// nonsense while every sealed turn recorded it.
    ///
    /// The comparison is against ids the model actually reserves meaning for —
    /// the whole special-token band plus a stride across the ordinary range —
    /// rather than the entire vocabulary, which would cost a full string
    /// compare of 248k entries on every boot to catch what any of these catch.
    fn verify_tokenizer_matches_checkpoint(
        tokenizer: &tokenizers::Tokenizer,
        model_path: &Path,
    ) -> crate::Result<()> {
        use candle::quantized::gguf_file;

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| ConversationError::Model(candle::Error::Msg(format!("open GGUF: {e}"))))?;
        let ct = gguf_file::Content::read(&mut file).map_err(ConversationError::Model)?;
        let Some(tokens) = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
        else {
            // Not every conversion carries the table. Say so rather than
            // implying the pair was checked.
            tracing::warn!(
                "checkpoint carries no `tokenizer.ggml.tokens`; the tokenizer cannot be \
                 verified against it and is trusted on its resolved path alone"
            );
            return Ok(());
        };

        // The GGUF's table is padded out to the output projection's width
        // (Qwen3.6-35B: 248,320 entries against the tokenizer's 248,070 real
        // tokens), so the two lengths are NOT expected to be equal — only
        // compatible. A tokenizer with MORE entries than the model has rows is
        // the unambiguous error; short of that, the agreement of the ids
        // themselves is what settles it.
        let n = tokens.len();
        let vocab = tokenizer.get_vocab(true);
        // **The id space, not the entry count.** `get_vocab().len()` counts
        // entries; added tokens sit at arbitrary high ids, so a tokenizer with
        // few entries can still address an id past the checkpoint's last row.
        // Bounding on the count would let exactly that through.
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let tok_vocab = max_id + 1;
        if tok_vocab > n {
            return Err(ConversationError::Tokenizer(format!(
                "tokenizer does not match the checkpoint: the tokenizer at {} defines \
                 {tok_vocab} tokens but the checkpoint's table has only {n} — ids past the end \
                 of the model's own vocabulary. This tokenizer was built for a different model.",
                model_path.display(),
            )));
        }

        // **The specials by id, not by position.** A mismatch on one of these
        // is the difference between ending a turn and emitting a word, and they
        // do not live at the bottom of the id space: Qwen puts `<|endoftext|>`
        // at 151643 on one model and 248044 on another, which is precisely the
        // pair this check exists to tell apart. A `0..1024` prefix — what this
        // used to probe while its comment claimed "specials first" — reaches
        // neither, and a 512-sample stride hits a given high id only by luck.
        //
        // `get_added_tokens_decoder` is the authoritative set: every special and
        // every added token, keyed by the id it actually occupies.
        let specials: Vec<usize> = tokenizer
            .get_added_tokens_decoder()
            .keys()
            .map(|&id| id as usize)
            .collect();
        let stride = (tok_vocab / 512).max(1);
        let probes = specials
            .into_iter()
            .chain(0..tok_vocab.min(1024))
            .chain((0..tok_vocab).step_by(stride))
            .filter(|&id| id < n);
        for id in probes {
            let Ok(want) = tokens[id].to_string() else {
                continue;
            };
            let got = tokenizer.id_to_token(id as u32);
            if got.as_deref() != Some(want.as_str()) {
                return Err(ConversationError::Tokenizer(format!(
                    "tokenizer does not match the checkpoint: at id {id} the GGUF's token table \
                     has {want:?} but the tokenizer at {} has {got:?}. The two disagree about \
                     what an id means, which no length check can see.",
                    model_path.display(),
                )));
            }
        }

        tracing::info!(
            vocab = n,
            "tokenizer verified against the checkpoint's own token table"
        );
        Ok(())
    }

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
            // Qwen3.6 is a point release of the Qwen3.5 architecture and ships
            // the same arch string, so both land on the hybrid loader.
            "qwen35moe" => Some(ModelArch::Qwen35Hybrid),
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

    /// Resolve a repo file, **preferring the local cache over the network**.
    ///
    /// `Api::get` consults the cache too, but only after asking the hub which
    /// revision it should be holding — so a checkpoint sitting complete on disk
    /// still cannot be opened while the hub is unreachable, and an unanswered
    /// socket stalls the load for as long as the HTTP client will wait rather
    /// than failing. These are pinned files: one filename in one repo, whose
    /// exact length the spec records. A cache hit is the answer, and asking
    /// anyway only makes startup depend on the network.
    #[cfg(feature = "hub")]
    fn resolve_repo_file(&self, repo: &str, filename: &str) -> crate::Result<PathBuf> {
        use hf_hub::api::sync::Api;
        use hf_hub::Cache;

        if let Some(hit) = cached_repo_file(&Cache::default(), repo, filename) {
            return Ok(hit);
        }
        Api::new()
            .map_err(|e| ConversationError::Download(e.to_string()))?
            .model(repo.to_string())
            .get(filename)
            .map_err(|e| ConversationError::Download(e.to_string()))
    }

    #[cfg(feature = "hub")]
    fn download_or_fail(&self) -> crate::Result<(PathBuf, PathBuf)> {
        let model_path =
            self.resolve_repo_file(&self.spec.model_repo, &self.spec.model_filename)?;
        let tokenizer_path = self.resolve_repo_file(&self.spec.tokenizer_repo, "tokenizer.json")?;
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

/// A repo file's path in `cache`, or `None` if it is not there.
///
/// Split out from [`ModelBuilder::resolve_repo_file`] so the cache-first rule
/// can be tested against a temporary cache instead of the machine's real one —
/// the rule is what keeps a daemon startable when the hub is unreachable, and
/// it is worth a test that does not depend on what happens to be downloaded.
#[cfg(feature = "hub")]
fn cached_repo_file(cache: &hf_hub::Cache, repo: &str, filename: &str) -> Option<PathBuf> {
    cache.model(repo.to_string()).get(filename)
}

#[cfg(all(test, feature = "hub"))]
mod cache_first_tests {
    use super::cached_repo_file;

    /// **A cached file resolves without the network, and a miss says so.**
    ///
    /// `Api::get` ends at the cache too, but only after asking the hub which
    /// revision it should be holding — so before this, a checkpoint sitting
    /// complete on disk could not be opened while the hub was unreachable, and
    /// an unanswered socket stalled the load for as long as the HTTP client
    /// would wait. That is not hypothetical: it cost an 18-minute hang on 20
    /// seconds of CPU, with the model never opened.
    #[test]
    fn a_cached_file_is_found_without_touching_the_network() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache = hf_hub::Cache::new(tmp.path().to_path_buf());
        let repo = "acme/widget-GGUF";

        assert!(
            cached_repo_file(&cache, repo, "widget.gguf").is_none(),
            "an empty cache must report a miss, not a phantom hit"
        );

        // Lay the file down the way hf-hub itself does — a ref pointing at a
        // commit, and the file under that commit's snapshot — so this exercises
        // the real lookup rather than a re-implementation of its path rules.
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let repo_cache = cache.model(repo.to_string());
        repo_cache.create_ref(commit).expect("create ref");
        let snapshot = tmp
            .path()
            .join(hf_hub::Repo::model(repo.to_string()).folder_name())
            .join("snapshots")
            .join(commit);
        std::fs::create_dir_all(&snapshot).expect("mkdir");
        std::fs::write(snapshot.join("widget.gguf"), b"weights").expect("write");

        assert_eq!(
            cached_repo_file(&cache, repo, "widget.gguf"),
            Some(snapshot.join("widget.gguf")),
            "a file already in the cache must resolve from it"
        );
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
