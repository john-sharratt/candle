//! Loading each architecture's weights from its pack, with the RoPE schedule and
//! the selection budget it runs under.
//!
//! The builder refuses settings its architecture cannot honour before anything is
//! read, so a model is never loaded only to ignore what it was asked: a selection
//! budget for an architecture that does not select, a RoPE preset from another
//! lineage, or a tensor taken from another checkpoint on an architecture whose pack
//! build does not read one. Every architecture loads through its own loader, and
//! the loaded model reports the int8 mode it was loaded in rather than re-deriving
//! it, since free memory read after the load would differ. Adapters load after the
//! base weights, in the width the stack computes in.

use super::*;

/// A model [`ModelBuilder::load_model`] loaded, with the int8 mode it was
/// loaded in — the value a report on the model names, never re-derived.
pub struct LoadedModel {
    pub model: Box<dyn ManagedBatchedModel + Send>,
    pub int8_mode: Int8Mode,
}

/// The frequencies a GQA loader built from its file, or the refusal to go on
/// without them.
pub(super) fn stated_inv_freq(inv: Option<Vec<f32>>) -> crate::Result<Vec<f32>> {
    inv.ok_or_else(|| {
        ConversationError::Model(candle::Error::Msg("model missing rope inv_freq".into()))
    })
}

// Read-back & terminal methods

impl ModelBuilder {
    /// Wrap a GQA checkpoint in the RoPE schedule its preset names over the
    /// frequencies the file states (`docs/progressive_yarn.md` §2).
    ///
    /// θ is the file's `{arch}.rope.freq_base` and the declaration its
    /// `{arch}.rope.scaling.*`; the file-stated reach is the builder's
    /// `max_seq_len`, which is the file's `context_length` when it declares
    /// one.
    pub(super) fn wrap_gqa<M: BatchedModelCore>(
        &self,
        raw: M,
        stated_inv: Vec<f32>,
        model_path: &Path,
        device: &Device,
    ) -> crate::Result<BatchedInference<M>> {
        let ct = gguf_header(model_path).map_err(ConversationError::Model)?;
        let (theta, declared) = gguf_rope(&ct.metadata).map_err(ConversationError::Model)?;
        let schedule = self
            .spec
            .rope
            .gqa_schedule(stated_inv, theta, declared, self.max_seq_len)
            .map_err(ConversationError::Model)?;
        tracing::info!(
            "RoPE schedule: {:?}, rung ceilings {:?}",
            self.spec.rope,
            schedule.ceilings()
        );
        Ok(BatchedInference::new_with_schedule(
            raw,
            &schedule,
            self.max_seq_len,
            device,
        )?)
    }

    /// Load a model from its model pack ([`Self::resolve_model`]).
    ///
    /// Uses the builder's `max_seq_len` for KV cache sizing. The int8 mode is the one the pack was built for — chosen
    /// once, when it was resolved — and is returned with the model, never
    /// re-derived: `auto_sized` reads free VRAM, which the load itself changes.
    pub fn load_model(
        &self,
        weights: &WeightFiles,
        device: &Device,
        progress: Option<LoadProgress<'_>>,
    ) -> crate::Result<LoadedModel> {
        self.refuse_unsupported_settings()?;
        // Before anything is loaded: the link is read with the card empty and
        // the bus idle, and its landing window is the driver's, not the span's
        // — which does not exist yet. Every later ask reads the cached figure.
        link_rate(device);
        let int8_mode = ModelPack::open(&weights.pack)
            .map_err(ConversationError::Model)?
            .int8_mode;
        let mut model = self.load_arch(weights, device, int8_mode, progress)?;
        // After `load_arch` has applied the budget: a selecting model checks
        // the strata against its kernel under the budget it will run with.
        model
            .set_selection_strata(self.qsa_strata)
            .map_err(ConversationError::Model)?;
        tracing::info!(
            window = self.qsa_strata.window,
            recent = self.qsa_strata.recent,
            mode = ?self.qsa_strata.mode,
            "selection strata (a model whose attention reads every position ignores it)"
        );
        model
            .set_hit_layers(&self.hit_layers.clone().unwrap_or_default())
            .map_err(ConversationError::Model)?;
        Ok(LoadedModel { model, int8_mode })
    }

    /// Refuse a setting this spec's architecture cannot honour — before anything
    /// is read, so a model is never loaded only to ignore what it was asked.
    pub(super) fn refuse_unsupported_settings(&self) -> crate::Result<()> {
        // **Only the qwen35 pack build reads a tensor from another checkpoint.** A spec naming
        // one for any other arch would load the primary's own copy and serve a model nobody
        // asked for under this spec's name, so it is refused rather than served.
        if !self.spec.tensor_overrides.is_empty()
            && !matches!(
                self.spec.arch,
                ModelArch::Qwen35Hybrid | ModelArch::Qwen35Dense
            )
        {
            return Err(ConversationError::Other(format!(
                "{:?} cannot take tensors from another checkpoint, and this spec names {}",
                self.spec.arch,
                self.spec.tensor_overrides.len()
            )));
        }
        // **Only a selecting architecture takes a selection budget.** Any other
        // would load, ignore it, and report a dense control that was never run.
        if let Some(positions) = self.qsa_selection_budget {
            if !matches!(self.spec.arch, ModelArch::Qwen4Exp) {
                return Err(ConversationError::Other(format!(
                    "{:?} has no QSA selection, so a selection budget of {positions} \
                     position(s) would change nothing",
                    self.spec.arch
                )));
            }
        }
        // **A lineage's schedule is its loader's.** A preset naming another for
        // one would be silently ignored, so it is refused.
        if self.spec.arch.file_rope() == RopePreset::Lineage
            && self.spec.rope != RopePreset::Lineage
        {
            return Err(ConversationError::Other(format!(
                "{:?} carries its RoPE schedule in its loader, and this spec names {:?}",
                self.spec.arch, self.spec.rope
            )));
        }
        Ok(())
    }

    /// The model `spec.arch` names, loaded from its model pack and adapters —
    /// every architecture's own loader. `int8mode` is the pack's, for the
    /// loaders of a plain pack, which is an ordinary GGUF to them and repacks at
    /// load.
    pub(super) fn load_arch(
        &self,
        weights: &WeightFiles,
        device: &Device,
        int8mode: Int8Mode,
        progress: Option<LoadProgress<'_>>,
    ) -> crate::Result<Box<dyn ManagedBatchedModel + Send>> {
        let max_seq = self.max_seq_len;
        let model_path = weights.pack.as_path();
        match self.spec.arch {
            ModelArch::Qwen3 => {
                use candle_transformers::models::quantized_qwen3::ModelWeights;
                // Per-layer progress not yet wired for this arch — the
                // callback simply doesn't fire. Add `progress` to the
                // arch's loader to enable it.
                let _ = progress;
                let raw = ModelWeights::from_gguf_by_path_with_int8(model_path, device, int8mode)?;
                let inv = stated_inv_freq(raw.rope_inv_freq())?;
                Ok(Box::new(self.wrap_gqa(raw, inv, model_path, device)?))
            }
            ModelArch::Qwen3Moe => {
                use candle_transformers::models::quantized_qwen3_moe::ModelWeights;
                let raw = ModelWeights::from_pack(model_path, device, progress)?;
                let inv = stated_inv_freq(raw.rope_inv_freq())?;
                Ok(Box::new(self.wrap_gqa(raw, inv, model_path, device)?))
            }
            ModelArch::Qwen2 => {
                use candle_transformers::models::quantized_qwen2::ModelWeights;
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                let raw = ModelWeights::from_gguf_by_path_with_options(
                    model_path, device, None, int8mode,
                )?;
                let inv = stated_inv_freq(raw.rope_inv_freq())?;
                Ok(Box::new(self.wrap_gqa(raw, inv, model_path, device)?))
            }
            ModelArch::Llama => {
                use candle_transformers::models::quantized_llama::ModelWeights;
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                // The deployed Llama is the 3.x family (Llama-3.2-3B), so the
                // v3 KV-factor row applies; a Llama-2 checkpoint needs its own
                // `ModelArch` split onto `from_gguf_by_path_v2` before it can
                // load through the daemon.
                let raw =
                    ModelWeights::from_gguf_by_path_with_int8_v3(model_path, device, int8mode)?;
                let inv = stated_inv_freq(raw.rope_inv_freq())?;
                Ok(Box::new(self.wrap_gqa(raw, inv, model_path, device)?))
            }
            ModelArch::DeepSeekV4 => {
                use candle_transformers::models::deepseek4::DEEPSEEK_V4;
                use candle_transformers::models::latent_moe::{BatchedEngine, Engine};
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                let _ = max_seq; // window/corpus budgets are model-derived
                let engine = Engine::load(model_path, &DEEPSEEK_V4, device)
                    .map_err(ConversationError::Model)?;
                Ok(Box::new(
                    BatchedEngine::new(engine).map_err(ConversationError::Model)?,
                ))
            }
            ModelArch::Qwen4Exp => {
                use candle_transformers::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
                // KV is allocated per ATTENTION layer (12 of 48) and the window
                // budget is config-derived, exactly as the hybrid's is.
                let _ = max_seq;
                // The pack of the merged KO artifact, not of the vendor's
                // split; `progress` reports the pinned layers' fill, the bulk of
                // a load's wall time.
                let gpu = Qwen4ExpGpu::load_with_progress(model_path, device, progress)
                    .map_err(ConversationError::Model)?;
                let mut model = Qwen4ExpBatched::new(gpu).map_err(ConversationError::Model)?;
                if let Some(positions) = self.qsa_selection_budget {
                    model
                        .set_selection_budget(positions)
                        .map_err(ConversationError::Model)?;
                }
                Ok(Box::new(model))
            }
            ModelArch::Qwen35Hybrid => {
                use candle_transformers::models::quantized_qwen36_moe;
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                // KV is allocated per ATTENTION layer, not per transformer layer,
                // and the window budget is derived from the config — see
                // `qwen35::engine::create_session`.
                let _ = max_seq;
                // The gate donor and tensor overrides were folded in when the
                // pack was built (`pack_source::resolve_spec`).
                let model = quantized_qwen36_moe::from_pack(model_path, device)
                    .map_err(ConversationError::Model)?;
                Ok(Box::new(model))
            }
            ModelArch::Qwen35Dense => {
                use candle_transformers::models::quantized_qwen35;
                // Per-layer progress not yet wired for this arch.
                let _ = progress;
                // KV is allocated per ATTENTION layer, not per transformer
                // layer, and the window budget is derived from the config.
                let _ = max_seq;
                let mut model = quantized_qwen35::from_pack(model_path, device)
                    .map_err(ConversationError::Model)?;
                // Adapters load after the base weights and before the model is
                // handed to the scheduler, in the activation width the stack
                // computes in — PEFT stores them F32, and converting per
                // projection would be a full-tensor pass on the hot path.
                // BF16 for this lineage — the width the residual stream flows
                // in, which is what an adapter's `A` matmul consumes. Taken
                // from the arch rather than named here so the adapter cannot
                // disagree with the norms, which are materialised from the same
                // answer.
                let act = self
                    .spec
                    .arch
                    .native_activation_dtype()
                    .unwrap_or(candle::DType::F16);
                for (name, dir) in &weights.adapters {
                    model
                        .load_adapter(name, dir, act)
                        .map_err(ConversationError::Model)?;
                }
                Ok(Box::new(model))
            }
        }
    }
}
