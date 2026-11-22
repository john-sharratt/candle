//! The model settings a builder reads back: its conversation configuration, its
//! system prompt, and the GGUF header it detects them from.
//!
//! The conversation configuration is assembled from the builder's overrides. When
//! thinking is suppressed and sampling was not set explicitly, it falls back to the
//! model's non-thinking sampling. The GGUF header gives the architecture, context
//! length, thinking support, and the logits width read from the output projection's
//! shape, since tied-embedding models pad their embedding rows past the tokenizer's
//! count. The header is parsed once per process and read through a cache.

use super::*;

// GGUF metadata extracted from the model file

/// Metadata extracted from a GGUF model file header.
///
/// Used by [`ModelBuilder::engine`] to auto-configure architecture, sampling,
/// vocab size, context length, and thinking support from the model file itself.
pub(super) struct GgufInfo {
    /// `general.name` — human-readable model name.
    pub(super) name: String,
    /// Detected [`ModelArch`] from `general.architecture`.
    pub(super) arch: Option<ModelArch>,
    /// Recommended sampling defaults for this architecture.
    pub(super) sampling: SamplingConfig,
    /// Non-thinking sampling variant (if model supports thinking).
    pub(super) non_thinking: Option<SamplingConfig>,
    /// Vocab size from `output.weight` tensor shape.
    pub(super) vocab_size: Option<usize>,
    /// `{arch}.context_length` — maximum sequence length.
    pub(super) context_length: Option<usize>,
    /// Whether the chat template contains `<think>`.
    pub(super) has_thinking: bool,
    /// Detected dialect from the chat template.
    pub(super) dialect: Option<DialectType>,
}

// Read-back & terminal methods

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
            reproject_every_n_tokens: REPROJECT_EVERY_N_TOKENS,
            // The belief probe covers the last N decode-window tokens. 256 spans
            // a full tool-call turn's discriminative span; the §80.2 sweep shows
            // the true tool is always within Top-3 at this width (100% recall at
            // budget 3), where a 64-token tail needed budget 5. See docs §80.2.
            reproject_max_probe_tokens: REPROJECT_MAX_PROBE_TOKENS,
            // Linefeed is the most reliable paragraph/section break
            // signal across chat templates and content styles.
            reproject_trigger_texts: REPROJECT_TRIGGER_TEXTS.map(String::from).to_vec(),
            reproject_in_thinking: true,
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

    /// Read the GGUF header to extract model metadata.
    ///
    /// Returns architecture, sampling defaults, vocab_size, thinking support,
    /// model name, and context length — everything needed to configure the
    /// builder from the GGUF file itself.
    ///
    /// Through [`gguf_header`], which parses once per process: an engine build
    /// asks for this header three times (here, the tokenizer check, and
    /// `engine_config`'s vocabulary width).
    pub(super) fn detect_sampling_from_gguf(model_path: &Path) -> crate::Result<GgufInfo> {
        let ct = gguf_header(model_path).map_err(ConversationError::Model)?;
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

        // Read context_length from `{arch}.context_length` metadata — the model
        // pack is the model's only file, so there is no `config.json` beside it
        // to fall back to.
        let context_length_key = format!("{}.context_length", arch_str);
        let context_length = ct
            .metadata
            .get(&context_length_key)
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        if let Some(ctx) = context_length {
            tracing::info!("GGUF {}: {}", context_length_key, ctx);
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
}
