use crate::{
    models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm},
    utils::repeat_kv,
};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub(crate) fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3Attention {
    // projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // norms
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: KvCache,
    use_flash_attn: bool,
}

impl Qwen3Attention {
    pub(crate) fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            candle::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        let kv_cache = KvCache::new(2, 512);

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
            hidden_size,
            rotary_emb,
            kv_cache,
            use_flash_attn,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Proj
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per‑head RMSNorm
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Decide if we can use Flash Attention (only when cache is empty)
        let cache_is_empty = self.kv_cache.current_seq_len() == 0;
        let use_flash = self.use_flash_attn && cache_is_empty;

        // 6. Attention with cache population
        let ctx = if use_flash {
            // Try Flash Attention, but ALWAYS populate cache afterward
            #[cfg(feature = "flash-attn")]
            {
                match candle_flash_attn::flash_attn(&q, &k, &v, self.head_dim as f32, false) {
                    Ok(result) => {
                        // Flash succeeded! Now populate cache for next time
                        self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
                        result
                    }
                    Err(_) => {
                        // Flash failed, fall back to standard with cache
                        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
                        let k = repeat_kv(k, self.num_kv_groups)?;
                        let v = repeat_kv(v, self.num_kv_groups)?;
                        self.standard_attention(&q, &k, &v, attn_mask)?
                    }
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                // Flash not available, use standard with cache
                let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
                let k = repeat_kv(k, self.num_kv_groups)?;
                let v = repeat_kv(v, self.num_kv_groups)?;
                self.standard_attention(&q, &k, &v, attn_mask)?
            }
        } else {
            // Standard attention with KV cache (incremental generation)
            let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
            let k = repeat_kv(k, self.num_kv_groups)?;
            let v = repeat_kv(v, self.num_kv_groups)?;
            self.standard_attention(&q, &k, &v, attn_mask)?
        };

        // 7. Output proj
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn standard_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs.matmul(&v)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }

    pub fn truncate_kv_cache(&mut self, seq_len: usize) -> Result<()> {
        self.kv_cache.truncate(seq_len)
    }

    pub fn try_truncate_or_reset_kv_cache(&mut self, seq_len: usize) -> Result<bool> {
        self.kv_cache.try_truncate_or_reset(seq_len)
    }

    pub fn try_reserve_kv_cache(&mut self, additional_tokens: usize) -> bool {
        self.kv_cache.try_reserve(additional_tokens)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        rotary: Arc<Qwen3RotaryEmbedding>,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, rotary, use_flash_attn, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }

    fn truncate_kv_cache(&mut self, seq_len: usize) -> Result<()> {
        self.self_attn.truncate_kv_cache(seq_len)
    }

    fn try_truncate_or_reset_kv_cache(&mut self, seq_len: usize) -> Result<bool> {
        self.self_attn.try_truncate_or_reset_kv_cache(seq_len)
    }

    fn try_reserve_kv_cache(&mut self, additional_tokens: usize) -> bool {
        self.self_attn.try_reserve_kv_cache(additional_tokens)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                use_flash_attn,
                vb_l.pp(i),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn truncate_kv_cache(&mut self, seq_len: usize) -> Result<()> {
        for l in &mut self.layers {
            l.truncate_kv_cache(seq_len)?;
        }
        Ok(())
    }

    /// Try to truncate all layers, if any fail, reset all for consistency
    fn try_truncate_or_reset_kv_cache(&mut self, seq_len: usize) -> Result<bool> {
        let mut all_success = true;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            match layer.try_truncate_or_reset_kv_cache(seq_len) {
                Ok(success) => {
                    if !success {
                        all_success = false;
                        // If this layer failed, reset all remaining layers
                        for remaining_layer in &mut self.layers[i + 1..] {
                            remaining_layer.clear_kv_cache();
                        }
                        break;
                    }
                }
                Err(e) => {
                    // Unexpected error, reset everything
                    self.clear_kv_cache();
                    return Err(e);
                }
            }
        }

        if !all_success {
            // Some layer failed, reset all for consistency
            self.clear_kv_cache();
        }

        Ok(all_success)
    }

    fn try_reserve_kv_cache(&mut self, additional_tokens: usize) -> bool {
        for layer in &mut self.layers {
            if !layer.try_reserve_kv_cache(additional_tokens) {
                return false;
            }
        }
        true
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, use_flash_attn, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }

    pub fn truncate_kv_cache(&mut self, seq_len: usize) -> Result<()> {
        self.base.truncate_kv_cache(seq_len)
    }

    pub fn try_truncate_or_reset_kv_cache(&mut self, seq_len: usize) -> Result<bool> {
        self.base.try_truncate_or_reset_kv_cache(seq_len)
    }

    pub fn try_reserve_kv_cache(&mut self, additional_tokens: usize) -> bool {
        self.base.try_reserve_kv_cache(additional_tokens)
    }
}
