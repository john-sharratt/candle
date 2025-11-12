//! Qwen3 implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-0.6B) (architecture based on official implementations)
//!
use super::with_tracing::QMatMul;
use crate::models::causal_mask_cache::CausalMaskCache;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        let kv_cache = KvCache::new(2, 512);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
            span_attn,
        })
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        self.kv_cache.truncate(seq_len)
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }

    pub fn cache_len(&self) -> usize {
        self.kv_cache.k_cache().current_seq_len()
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Reset KV cache if we're at the first position
        if offset == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // Make tensor contiguous to avoid some strided copies
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        self.self_attn.truncate_cache(seq_len)
    }

    pub fn reset_cache(&mut self) {
        self.self_attn.reset_cache();
    }

    pub fn cache_len(&self) -> usize {
        self.self_attn.cache_len()
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    mask_cache: CausalMaskCache,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl Clone for ModelWeights {
    fn clone(&self) -> Self {
        // Clone layers with fresh KV caches
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let kv_cache_capacity = layer.self_attn.kv_cache.k_cache().max_seq_len();
                let rotary_emb = layer.self_attn.rotary_emb.clone();

                LayerWeights {
                    // These are cheap to clone (reference-counted)
                    ln1: layer.ln1.clone(),
                    ln2: layer.ln2.clone(),
                    mlp: layer.mlp.clone(),
                    self_attn: AttentionWeights {
                        q_proj: layer.self_attn.q_proj.clone(),
                        k_proj: layer.self_attn.k_proj.clone(),
                        v_proj: layer.self_attn.v_proj.clone(),
                        o_proj: layer.self_attn.o_proj.clone(),
                        q_norm: layer.self_attn.q_norm.clone(),
                        k_norm: layer.self_attn.k_norm.clone(),
                        num_heads: layer.self_attn.num_heads,
                        num_kv_heads: layer.self_attn.num_kv_heads,
                        num_kv_groups: layer.self_attn.num_kv_groups,
                        head_dim: layer.self_attn.head_dim,
                        rotary_emb,
                        // Create fresh, empty KV cache with same capacity
                        kv_cache: KvCache::new(2, kv_cache_capacity),
                        span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                    },
                }
            })
            .collect();

        Self {
            embed_tokens: self.embed_tokens.clone(),
            layers,
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            // Create fresh mask cache with same device to avoid sharing cached masks
            mask_cache: CausalMaskCache::new(self.mask_cache.device().clone()),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        }
    }
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mask_cache: CausalMaskCache::new(device.clone()),
            span,
            span_output,
        })
    }

    /// Truncate KV cache to specified sequence length
    pub fn truncate_kv_cache(&mut self, seq_len: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.truncate_cache(seq_len)?;
        }
        // Truncate the cached mask to match
        self.mask_cache.truncate(seq_len)?;
        Ok(())
    }

    /// Clear all KV caches (more efficient than forward with offset=0)
    pub fn clear_all_caches(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
        // Clear cached mask
        self.mask_cache.clear();
    }

    /// Get the current cache length (all layers should be same)
    pub fn cache_len(&self) -> usize {
        self.layers.first().map(|l| l.cache_len()).unwrap_or(0)
    }

    /// Rewind by N tokens from current position
    pub fn rewind_by_tokens(&mut self, n_tokens: usize) -> Result<()> {
        let current_len = self.cache_len();
        let target_pos = current_len.saturating_sub(n_tokens);
        self.truncate_kv_cache(target_pos)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            // Use shared mask cache with GPU-accelerated creation
            Some(self.mask_cache.get_mask(l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file;

    #[test]
    fn test_clone_with_independent_kv_cache() -> Result<()> {
        // Download a small Qwen3 model from HuggingFace
        // Using Qwen3-0.6B (smallest available for fast testing)
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        let repo = api.repo(hf_hub::Repo::with_revision(
            "unsloth/Qwen3-0.6B-GGUF".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let model_path = repo.get("Qwen3-0.6B-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("Model downloaded to: {:?}", model_path);

        let device = Device::cuda_if_available(0)?;
        println!("Using device: {:?}", device);

        // Load model
        let mut file = std::fs::File::open(&model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;

        println!("Model loaded, starting 500-token prefill...");

        // Step 1: Advance model forward by 500+ tokens to populate KV cache
        // Using token ID 1 (typically a valid token in most vocabularies)
        let prefill_tokens = 500;
        for i in 0..prefill_tokens {
            let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
            let _output = model.forward(&input, i)?;

            if (i + 1) % 100 == 0 {
                println!("  Prefill progress: {}/{}", i + 1, prefill_tokens);
            }
        }

        let original_cache_len = model.cache_len();
        assert_eq!(
            original_cache_len, prefill_tokens,
            "Original model should have {} tokens in cache",
            prefill_tokens
        );
        println!("✓ Original model cache: {} tokens", original_cache_len);

        // Step 2: Clone the model
        println!("\nCloning model...");
        let mut cloned_model = model.clone();
        let clone_initial_cache_len = cloned_model.cache_len();
        assert_eq!(
            clone_initial_cache_len, 0,
            "Cloned model should start with empty cache"
        );
        println!(
            "✓ Cloned model cache: {} tokens (empty)",
            clone_initial_cache_len
        );

        // Step 3: Advance clone forward with new prompt (different token: 2)
        println!("\nAdvancing clone with new prompt (100 tokens)...");
        let clone_tokens = 100;
        for i in 0..clone_tokens {
            let input = Tensor::new(&[2u32], &device)?.unsqueeze(0)?;
            let _output = cloned_model.forward(&input, i)?;
        }

        let clone_cache_len = cloned_model.cache_len();
        assert_eq!(
            clone_cache_len, clone_tokens,
            "Clone should have {} tokens in cache",
            clone_tokens
        );
        println!("✓ Clone cache after generation: {} tokens", clone_cache_len);

        // Step 4: Verify original model cache is still intact
        let original_cache_len_after_clone = model.cache_len();
        assert_eq!(
            original_cache_len_after_clone, prefill_tokens,
            "Original model cache should still have {} tokens (not affected by clone)",
            prefill_tokens
        );
        println!(
            "✓ Original cache after clone generation: {} tokens (unchanged)",
            original_cache_len_after_clone
        );

        // Step 5: Advance original forward with continuation (token 3)
        println!("\nAdvancing original model (50 more tokens)...");
        let original_continue_tokens = 50;
        for i in 0..original_continue_tokens {
            let input = Tensor::new(&[3u32], &device)?.unsqueeze(0)?;
            let _output = model.forward(&input, prefill_tokens + i)?;
        }

        let original_final_cache_len = model.cache_len();
        assert_eq!(
            original_final_cache_len,
            prefill_tokens + original_continue_tokens,
            "Original model should have {} tokens in cache",
            prefill_tokens + original_continue_tokens
        );
        println!(
            "✓ Original cache after continuation: {} tokens",
            original_final_cache_len
        );

        // Verify caches are completely independent
        assert_ne!(
            original_final_cache_len, clone_cache_len,
            "Original and clone should have different cache lengths"
        );

        println!("\n=== Test Summary ===");
        println!(
            "✓ Original model: {} tokens in cache",
            original_final_cache_len
        );
        println!("✓ Cloned model: {} tokens in cache", clone_cache_len);
        println!("✓ Caches are completely independent");
        println!("✓ Clone did not affect original model state");

        Ok(())
    }
}
