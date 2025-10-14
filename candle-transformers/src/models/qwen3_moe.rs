use crate::models::{
    qwen3::{Config as Qwen3Config, Qwen3Attention, Qwen3MLP, Qwen3RotaryEmbedding},
    with_tracing::{linear_no_bias, Linear, RmsNorm},
};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{kv_cache::KvCache, Activation, VarBuilder};
use std::sync::{Arc, Mutex};

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
    // MoE specific configuration
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    pub norm_topk_prob: bool,
}

impl From<&Config> for Qwen3Config {
    fn from(val: &Config) -> Self {
        Qwen3Config {
            vocab_size: val.vocab_size,
            hidden_size: val.hidden_size,
            intermediate_size: val.intermediate_size,
            num_hidden_layers: val.num_hidden_layers,
            num_attention_heads: val.num_attention_heads,
            head_dim: val.head_dim,
            attention_bias: val.attention_bias,
            num_key_value_heads: val.num_key_value_heads,
            max_position_embeddings: val.max_position_embeddings,
            sliding_window: val.sliding_window,
            max_window_layers: val.max_window_layers,
            tie_word_embeddings: val.tie_word_embeddings,
            rope_theta: val.rope_theta,
            rms_norm_eps: val.rms_norm_eps,
            use_sliding_window: val.use_sliding_window,
            hidden_act: val.hidden_act,
        }
    }
}

#[derive(Debug, Clone)]
struct Qwen3MLPExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLPExpert {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.moe_intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLPExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// Qwen3 Sparse MoE Block implementation
#[derive(Debug, Clone)]
struct Qwen3SparseMoeBlock {
    gate: Linear,
    experts: Vec<Qwen3MLPExpert>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl Qwen3SparseMoeBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        let vb_e = vb.pp("experts");
        for idx in 0..cfg.num_experts {
            let expert = Qwen3MLPExpert::new(cfg, vb_e.pp(idx))?;
            experts.push(expert)
        }
        Ok(Self {
            gate,
            experts,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }
}

impl Module for Qwen3SparseMoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let router_logits = xs.apply(&self.gate)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Extract topk experts per token
        let experts_per_tok = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let routing_weights = routing_weights.gather(&experts_per_tok, D::Minus1)?;

        // Extract needed data
        let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let experts_per_tok = experts_per_tok.to_vec2::<u32>()?;
        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_experts = vec![vec![]; self.experts.len()];
        for (row_idx, (rw, expert_idxs)) in routing_weights
            .iter()
            .zip(experts_per_tok.iter())
            .enumerate()
        {
            let sum_rw = rw.iter().sum::<f32>();
            for (&rw, &expert_idx) in rw.iter().zip(expert_idxs.iter()) {
                top_x[expert_idx as usize].push(row_idx as u32);
                let rw = if self.norm_topk_prob { rw / sum_rw } else { rw };
                selected_experts[expert_idx as usize].push(rw)
            }
        }

        // Process through experts
        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_experts =
                Tensor::new(selected_experts[expert_idx].as_slice(), xs.device())?
                    .reshape(((), 1))?
                    .to_dtype(xs.dtype())?;

            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            let current_hidden_states = expert_layer.forward(&current_state)?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_experts)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }

        ys.reshape((b_size, seq_len, hidden_dim))
    }
}

// MLP or MoE decision enum
#[derive(Debug, Clone)]
enum Qwen3FeedForward {
    Mlp(Qwen3MLP),
    MoE(Qwen3SparseMoeBlock),
}

impl Module for Qwen3FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::MoE(m) => m.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    feed_forward: Qwen3FeedForward,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(
        layer_idx: usize,
        cfg: &Config,
        rotary: Arc<Qwen3RotaryEmbedding>,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn =
            Qwen3Attention::new(&cfg.into(), rotary, use_flash_attn, vb.pp("self_attn"))?;

        // Decide whether to use MoE or regular MLP based on layer_idx and decoder_sparse_step
        let feed_forward =
            if cfg.num_experts > 0 && (layer_idx + 1).is_multiple_of(cfg.decoder_sparse_step) {
                Qwen3FeedForward::MoE(Qwen3SparseMoeBlock::new(cfg, vb.pp("mlp"))?)
            } else {
                Qwen3FeedForward::Mlp(Qwen3MLP::new(&cfg.into(), vb.pp("mlp"))?)
            };

        let ln1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            feed_forward,
            ln1,
            ln2,
        })
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(cache, &h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.feed_forward)?;
        x + h2
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
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(
            vb.dtype(),
            &cfg.into(),
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                i,
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

    pub fn num_layers(&self) -> usize {
        self.layers.len()
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

    pub fn forward(&self, caches: &mut [KvCache], input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(cache, &h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}

/// Inner model weights (wrapped in Arc for cheap cloning)
#[derive(Debug)]
struct ModelForCausalLMInner {
    base: Mutex<Model>,
    lm_head: Linear,
}

/// Shareable model - cheap to clone due to internal Arc
#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    inner: Arc<ModelForCausalLMInner>,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, use_flash_attn, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            inner: Arc::new(ModelForCausalLMInner {
                base: Mutex::new(base),
                lm_head,
            }),
        })
    }

    /// Create a new inference instance with its own KV cache
    pub fn new_instance(&self) -> InstanceForCausalLM {
        let num_layers = self.inner.base.lock().unwrap().num_layers();
        let caches = (0..num_layers).map(|_| KvCache::new(2, 512)).collect();
        InstanceForCausalLM {
            model: self.clone(),
            caches,
        }
    }

    fn forward_inner(
        &self,
        caches: &mut [KvCache],
        input: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let base = self.inner.base.lock().unwrap();
        base.forward(caches, input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.inner.lm_head)
    }
}

/// Instance of a model with its own KV cache state
/// This is what you use for inference - each thread gets its own instance
pub struct InstanceForCausalLM {
    model: ModelForCausalLM,
    caches: Vec<KvCache>,
}

impl InstanceForCausalLM {
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.model.forward_inner(&mut self.caches, input, offset)
    }

    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn truncate_kv_cache(&mut self, seq_len: usize) -> Result<()> {
        for cache in &mut self.caches {
            cache.truncate(seq_len)?;
        }
        Ok(())
    }

    pub fn try_truncate_or_reset_kv_cache(&mut self, seq_len: usize) -> Result<bool> {
        let mut all_success = true;

        for (i, cache) in self.caches.iter_mut().enumerate() {
            match cache.try_truncate_or_reset(seq_len) {
                Ok(success) => {
                    if !success {
                        all_success = false;
                        // If this cache failed, reset all remaining caches
                        for remaining_cache in &mut self.caches[i + 1..] {
                            remaining_cache.reset();
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
            // Some cache failed, reset all for consistency
            self.clear_kv_cache();
        }

        Ok(all_success)
    }

    pub fn try_reserve_kv_cache(&mut self, additional_tokens: usize) -> bool {
        for cache in &mut self.caches {
            if !cache.try_reserve(additional_tokens) {
                return false;
            }
        }
        true
    }
}

// Keep static helper methods on ModelForCausalLM for compatibility
impl ModelForCausalLM {
    pub fn forward(&self, caches: &mut [KvCache], input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_inner(caches, input, offset)
    }

    pub fn clear_kv_cache(caches: &mut [KvCache]) {
        for cache in caches {
            cache.reset();
        }
    }

    pub fn truncate_kv_cache(caches: &mut [KvCache], seq_len: usize) -> Result<()> {
        for cache in caches {
            cache.truncate(seq_len)?;
        }
        Ok(())
    }

    pub fn try_truncate_or_reset_kv_cache(caches: &mut [KvCache], seq_len: usize) -> Result<bool> {
        let mut all_success = true;

        for (i, cache) in caches.iter_mut().enumerate() {
            match cache.try_truncate_or_reset(seq_len) {
                Ok(success) => {
                    if !success {
                        all_success = false;
                        // If this cache failed, reset all remaining caches
                        for remaining_cache in &mut caches[i + 1..] {
                            remaining_cache.reset();
                        }
                        break;
                    }
                }
                Err(e) => {
                    // Unexpected error, reset everything
                    Self::clear_kv_cache(caches);
                    return Err(e);
                }
            }
        }

        if !all_success {
            // Some cache failed, reset all for consistency
            Self::clear_kv_cache(caches);
        }

        Ok(all_success)
    }

    pub fn try_reserve_kv_cache(caches: &mut [KvCache], additional_tokens: usize) -> bool {
        for cache in caches {
            if !cache.try_reserve(additional_tokens) {
                return false;
            }
        }
        true
    }
}
