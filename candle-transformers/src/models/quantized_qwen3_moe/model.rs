//! Defines the batch-forward configuration ladder for Qwen3-MoE, and the model's
//! reported weight footprint.
//!
//! `batched_forward_configs` is public so the engine above can run the same
//! ladder as the model's own gate, since a ceiling measured against a different
//! ladder would not be a ceiling. The rows run F16 first as the baseline, then
//! single and multi-context BF16, Q8 and Q4, with C8 between the rungs that pass
//! and the warm repeats last. The footprint the model reports is the fixed base
//! weights plus the live resident-expert footprint, which rises and falls as
//! experts move between VRAM and pinned host memory.

use super::*;

#[cfg(feature = "cuda")]
impl BatchedModelCore for ModelWeights {
    type Layer = LayerWeights;

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        for layer in &self.layers {
            layer.ln1.maybe_change_dtype(dtype)?;
            layer.ln2.maybe_change_dtype(dtype)?;
            layer.self_attn.q_norm.maybe_change_dtype(dtype)?;
            layer.self_attn.k_norm.maybe_change_dtype(dtype)?;
        }
        self.norm.maybe_change_dtype(dtype)
    }

    fn wave_shapes(&self) -> WaveShapes {
        self.wave_shapes
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_head(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.num_kv_heads)
            .unwrap_or(0)
    }

    fn head_dim(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.head_dim)
            .unwrap_or(0)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn embeddings(&self) -> Option<&Embedding> {
        self.embeddings.as_ref()
    }

    #[cfg(feature = "cuda")]
    fn host_embedding(&self) -> Option<&crate::models::host_embedding::HostEmbedding> {
        self.host_embedding.as_ref()
    }

    fn layer(&self, idx: usize) -> &Self::Layer {
        &self.layers[idx]
    }

    fn final_norm(&self) -> &RmsNorm {
        &self.norm
    }

    fn output_proj(&self) -> &QMatMul {
        &self.lm_head
    }

    fn rope_interleaved(&self) -> bool {
        false
    }

    fn prune(&self) -> Result<()> {
        if let Some(e) = &self.embeddings {
            e.compact();
        }
        if let Some(layer) = self.layers.first() {
            if let Ok(mut cis) = layer.self_attn.rotary_emb.cis.write() {
                cis.compact();
            }
        }
        Ok(())
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        self.expert_cache.as_ref().map(|cache| cache.expert_stats())
    }

    fn expert_pipeline_dead(&self) -> bool {
        self.expert_cache
            .as_ref()
            .is_some_and(|cache| cache.pipeline_dead())
    }

    fn expert_stats_shared(&self) -> Option<Arc<Mutex<PipelineStats>>> {
        self.expert_cache.as_ref().map(|cache| cache.stats_shared())
    }

    fn expert_hit_counts(&self) -> Option<(usize, usize)> {
        self.expert_cache.as_ref().map(|cache| cache.hit_counts())
    }

    fn expert_hit_references(&self) -> Option<(f64, f64)> {
        self.expert_cache
            .as_ref()
            .and_then(|cache| cache.hit_references())
    }

    fn take_device_fault(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(cache) = self.expert_cache.as_ref() {
            cache.take_segment_fault(self.device())?;
        }
        Ok(())
    }

    fn request_kv_ground(&self, regions: usize) -> u64 {
        self.expert_cache
            .as_ref()
            .map_or(0, |cache| cache.request_kv_ground(regions))
    }

    fn reclaim_spare_ground(&self) {
        if let Some(cache) = self.expert_cache.as_ref() {
            cache.reclaim_spare_ground();
        }
    }

    fn resident_weight_bytes(&self) -> Option<usize> {
        // Fixed base weights + the live resident-expert footprint (0 when no
        // global expert cache). Rises/falls as experts page VRAM↔pinned RAM.
        let experts = self
            .expert_cache
            .as_ref()
            .map_or(0, |cache| cache.resident_vram_bytes());
        Some(self.base_weight_bytes + experts)
    }

    fn reset_expert_stats(&self) {
        if let Some(cache) = &self.expert_cache {
            cache.reset_expert_stats();
        }
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        self.expert_cache
            .as_ref()
            .map_or_else(ProfileSnapshot::default, |cache| cache.snapshot_profiles())
    }

    fn k_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.k_hi
    }

    fn k_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.k_low
    }

    fn v_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.v_hi
    }

    fn v_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.v_low
    }
}

/// The batched forward gate's configuration ladder for this model.
///
/// **Public, and outside the test module, so the same ladder can be driven from above.**
/// `candle-conversation` runs these rows to establish the *ceiling* — what the forward
/// itself can do from a clean slate — and then appends rows measured through the real
/// engine, so the two sit in one table. Duplicating the ladder there instead would let
/// the two drift, and a ceiling measured against a different ladder is not a ceiling.
///
/// The rows are ordered deliberately: F16 first (it is the table's baseline), the warm
/// BF16 and Q4_0 repeats last, so a reader can see what warming was worth.
#[cfg(all(feature = "cuda", any(test, feature = "ruler-bench")))]
pub fn batched_forward_configs() -> Vec<TestConfig> {
    vec![
        // F16 single context
        TestConfig {
            mode: InferenceMode::F16,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // BF16 single context
        TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // BF16 multi-context
        TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 10,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // Q8
        TestConfig {
            mode: InferenceMode::Q8_0,
            use_batched: true,
            num_contexts: 20,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        #[cfg(feature = "huge-context")]
        TestConfig {
            mode: InferenceMode::Q8_0,
            use_batched: true,
            num_contexts: 32,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // Q4
        TestConfig {
            mode: InferenceMode::Q4_0,
            use_batched: true,
            num_contexts: 4,
            num_repeats: 1,
            test_mode: Some(TestMode::Skip),
        },
        #[cfg(feature = "huge-context")]
        TestConfig {
            mode: InferenceMode::Q4_0,
            use_batched: true,
            num_contexts: 48,
            num_repeats: 1,
            test_mode: Some(TestMode::Skip),
        },
        TestConfig {
            mode: InferenceMode::C0,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C1,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C2,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C3,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C4,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C5,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // The engine probe's clean baseline runs eight C5 turns together
        // (`candle-conversation`'s `Probe::baseline_width`), so the ceiling it is
        // compared with is this row: the same mode at the same width.
        TestConfig {
            mode: InferenceMode::C5,
            use_batched: true,
            num_contexts: 8,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C6,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C7,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 48,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // C8 sits between two rungs that pass and was simply absent from this
        // ladder — the one model in the sweep whose top rungs were never
        // exercised contiguously, which is where a selection defect hides.
        TestConfig {
            mode: InferenceMode::C8,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C9,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        TestConfig {
            mode: InferenceMode::C10,
            use_batched: true,
            #[cfg(feature = "huge-context")]
            num_contexts: 10,
            #[cfg(not(feature = "huge-context"))]
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // BF16 single context (after everything is warm)
        TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        },
        // Q4_0 wide (after everything is warm)
        TestConfig {
            mode: InferenceMode::Q4_0,
            use_batched: true,
            num_contexts: 20,
            num_repeats: 1,
            test_mode: Some(TestMode::Skip),
        },
    ]
}
