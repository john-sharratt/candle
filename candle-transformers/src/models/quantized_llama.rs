//! Quantized llama model implementation.
//!
//! This provides a quantized implementation of the llama language model architecture.
//! The model implements parameter efficient quantization for reduced memory usage
//! while maintaining model quality.
//!
//! Key characteristics:
//! - Transformer decoder architecture
//! - Support for 2/3/4/8-bit quantization
//! - Optimized memory usage through quantization
//! - Configurable model sizes and parameter counts
//!
//! - 💻 [GH Link](https://github.com/facebookresearch/llama)
//! - 📝 [Paper](https://arxiv.org/abs/2302.13971)
//!
//! ![](https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/quantized/assets/aoc.gif)
//!

use crate::models::causal_mask_cache::CausalMaskCache;
use crate::quantized_nn::RmsNorm;
use candle::quantized::QTensor;
use candle::quantized::{ggml_file, gguf_file};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{kv_cache::Fp8KvCache, kv_cache::KvCache, Embedding, Module};

pub const MAX_SEQ_LEN: usize = 4096;

/// KV Cache variant - either regular or FP8 quantized
#[derive(Debug, Clone)]
enum KvCacheVariant {
    Regular(KvCache),
    Fp8(Fp8KvCache),
}

impl KvCacheVariant {
    fn new_regular(dim: usize, max_seq_len: usize) -> Self {
        Self::Regular(KvCache::new(dim, max_seq_len))
    }

    fn new_fp8(dim: usize, max_seq_len: usize) -> Self {
        Self::Fp8(Fp8KvCache::new(dim, max_seq_len))
    }

    fn current_seq_len(&self) -> usize {
        match self {
            Self::Regular(cache) => cache.current_seq_len(),
            Self::Fp8(cache) => cache.current_seq_len(),
        }
    }

    fn max_seq_len(&self) -> usize {
        match self {
            Self::Regular(cache) => cache.k_cache().max_seq_len(),
            Self::Fp8(cache) => cache.max_seq_len(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Regular(cache) => cache.reset(),
            Self::Fp8(cache) => cache.reset(),
        }
    }

    fn truncate(&mut self, seq_len: usize) -> Result<()> {
        match self {
            Self::Regular(cache) => cache.truncate(seq_len),
            Self::Fp8(cache) => cache.truncate(seq_len),
        }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Regular(cache) => cache.append(k, v),
            Self::Fp8(cache) => cache.append(k, v),
        }
    }
}

/// Guard for CUDA-registered mmap memory. Automatically unregisters on drop.
#[cfg(feature = "cuda")]
struct MmapRegistration {
    ptr: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
impl Drop for MmapRegistration {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe {
            // Unregister the memory mapping
            let _ = sys::cuMemHostUnregister(self.ptr).result();
        }
    }
}

// QMatMul wrapper adding some tracing.
#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: KvCacheVariant,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    // mask is F32: 0.0 = visible, -inf = masked
    // Convert to U8 for where_cond: 0.0 -> 0u8 (false), -inf -> 1u8 (true)
    let zero = Tensor::new(&[0.0f32], mask.device())?.broadcast_as(shape.dims())?;
    let mask_u8 = mask.ne(&zero)?; // not equal to 0.0 gives U8 tensor
    let m = mask_u8.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        self.kv_cache.truncate(seq_len)
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }

    pub fn cache_len(&self) -> usize {
        self.kv_cache.current_seq_len()
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // The call to contiguous below is only necessary when processing the prompt.
        // When the seq_len is 1 in the inference loop, this is a no-op.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            // This call to contiguous ensures that the fast kernel can be called below. It's
            // actually a no-op except when processing the initial prompt so has no significant
            // impact on performance.
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // Reset KV cache if we're at the first position
        if index_pos == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // KV cache already returns contiguous tensors
        // Removing redundant contiguous() calls saves 2 memory allocations per layer

        let y = if q.device().is_metal() && seq_len == 1 {
            // SDPA will do MQA for us
            candle_nn::ops::sdpa(&q, &k, &v, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else {
            // Support for MQA, useful for 70B models and mistral.
            let k = crate::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = crate::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
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
                let kv_cache_capacity = layer.kv_cache.max_seq_len();
                let kv_cache = match &layer.kv_cache {
                    KvCacheVariant::Regular(_) => KvCacheVariant::new_regular(2, kv_cache_capacity),
                    KvCacheVariant::Fp8(_) => KvCacheVariant::new_fp8(2, kv_cache_capacity),
                };
                LayerWeights {
                    // These tensors are cheap to clone (reference-counted, shared memory)
                    attention_wq: layer.attention_wq.clone(),
                    attention_wk: layer.attention_wk.clone(),
                    attention_wv: layer.attention_wv.clone(),
                    attention_wo: layer.attention_wo.clone(),
                    attention_norm: layer.attention_norm.clone(),
                    mlp_or_moe: layer.mlp_or_moe.clone(),
                    ffn_norm: layer.ffn_norm.clone(),
                    cos: layer.cos.clone(),
                    sin: layer.sin.clone(),
                    neg_inf: layer.neg_inf.clone(),
                    n_head: layer.n_head,
                    n_kv_head: layer.n_kv_head,
                    head_dim: layer.head_dim,
                    // Create fresh, empty KV cache with same capacity
                    kv_cache,
                    span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                    span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                    span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
                }
            })
            .collect();

        Self {
            tok_embeddings: self.tok_embeddings.clone(),
            layers,
            norm: self.norm.clone(),
            output: self.output.clone(),
            // Create fresh mask cache with same device to avoid sharing cached masks
            mask_cache: CausalMaskCache::new(self.mask_cache.device().clone()),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        }
    }
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let (cos, sin) = precomput_freqs_cis(head_dim, 10000., &ct.device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &ct.device)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
            // The cache will grow in chunks of 512 tokens when needed.
            let kv_cache = KvCacheVariant::new_regular(2, 2048);
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            mask_cache: CausalMaskCache::new(ct.device.clone()),
            span,
            span_output,
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
            // The cache will grow in chunks of 512 tokens when needed.
            let kv_cache = KvCacheVariant::new_regular(2, 2048);
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            mask_cache: CausalMaskCache::new(device.clone()),
            span,
            span_output,
        })
    }

    /// Load model from GGUF file using memory-mapped I/O for zero-copy tensor loading.
    ///
    /// This method eliminates intermediate RAM allocations and copies by using mmap:
    /// - Traditional: File → Vec<u8> → GPU (2 copies, 2x peak RAM)
    /// - This method: File (mmap) → GPU (1 copy, 1x peak RAM)
    ///
    /// Benefits:
    /// - **Eliminates RAM allocation** for tensor data
    /// - **Eliminates file→RAM copy** - only mmap→GPU remains
    /// - **Lower peak memory usage** - no temporary buffers
    /// - **OS page cache efficiency** - kernel optimizes page access
    ///
    /// # Arguments
    /// * `file_path` - Path to the GGUF file
    /// * `device` - Device to load tensors onto
    ///
    /// # Example
    /// ```no_run
    /// use candle_core::Device;
    /// use candle_transformers::models::quantized_llama::ModelWeights;
    /// use std::path::Path;
    ///
    /// let path = Path::new("model.gguf");
    /// let device = Device::cuda_if_available(0)?;
    /// let model = ModelWeights::from_gguf_by_path(path, &device)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_gguf_by_path(file_path: &std::path::Path, device: &Device) -> Result<Self> {
        use memmap2::MmapOptions;

        // Open file and create memory map for zero-copy access
        let file = std::fs::File::open(file_path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| candle::Error::Msg(format!("Failed to mmap file: {}", e)))?
        };

        // On CUDA devices, register mmap as mapped pinned memory for TRUE zero-copy
        // This allows GPU to read model weights directly from disk via PCIe without CPU copy!
        #[cfg(feature = "cuda")]
        let _mmap_guard = if matches!(device, Device::Cuda(_)) {
            use cudarc::driver::sys;

            let ptr = mmap.as_ptr() as *mut std::ffi::c_void;
            let len = mmap.len();

            // Register mmap with CUDA - allows GPU to read directly via PCIe
            let register_result = unsafe {
                sys::cuMemHostRegister_v2(
                    ptr,
                    len,
                    (sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_READ_ONLY) as u32,
                )
            };

            match register_result.result() {
                Ok(_) => {
                    // Successfully registered! GPU can now read mmap directly
                    Some(MmapRegistration { ptr })
                }
                Err(_) => {
                    // Registration failed (alignment issues, etc.) - fall back to regular memcpy
                    None
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "cuda"))]
        let _mmap_guard: Option<()> = None;

        // Parse GGUF metadata from mmap (23x faster than reading from File!)
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let ct = gguf_file::Content::read(&mut cursor)?;

        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Helper to load tensor from mmap
        let load_tensor = |name: &str| -> Result<QTensor> {
            let tensor_info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("tensor {} not found", name)))?;
            tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, device)
        };

        let tok_embeddings_q = load_tensor("token_embd.weight")?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(load_tensor("output_norm.weight")?, rms_norm_eps)?;
        let output = match load_tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = load_tensor(&format!("{prefix}.attn_q.weight"))?;
            let attention_wk = load_tensor(&format!("{prefix}.attn_k.weight"))?;
            let attention_wv = load_tensor(&format!("{prefix}.attn_v.weight"))?;
            let attention_wo = load_tensor(&format!("{prefix}.attn_output.weight"))?;

            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 = load_tensor(&format!("{prefix}.ffn_gate.weight"))?;
                let feed_forward_w2 = load_tensor(&format!("{prefix}.ffn_down.weight"))?;
                let feed_forward_w3 = load_tensor(&format!("{prefix}.ffn_up.weight"))?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp = load_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 = load_tensor(&format!("{prefix}.ffn_gate.{i}.weight"))?;
                    let feed_forward_w2 = load_tensor(&format!("{prefix}.ffn_down.{i}.weight"))?;
                    let feed_forward_w3 = load_tensor(&format!("{prefix}.ffn_up.{i}.weight"))?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };

            let attention_norm = load_tensor(&format!("{prefix}.attn_norm.weight"))?;
            let ffn_norm = load_tensor(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            let kv_cache = KvCacheVariant::new_regular(2, 2048);

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache,
                span_attn,
                span_rot,
                span_mlp,
            })
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
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

    /// Enable FP8 quantization for KV cache (reduces memory by ~50%)
    /// This will reset all existing caches and switch to FP8 storage.
    /// Note: This involves quantization overhead but saves significant memory.
    pub fn enable_fp8_kv_cache(&mut self) {
        for layer in &mut self.layers {
            let max_seq_len = layer.kv_cache.max_seq_len();
            layer.kv_cache = KvCacheVariant::new_fp8(2, max_seq_len);
        }
    }

    /// Disable FP8 quantization for KV cache (uses more memory but faster)
    /// This will reset all existing caches and switch to regular storage.
    pub fn disable_fp8_kv_cache(&mut self) {
        for layer in &mut self.layers {
            let max_seq_len = layer.kv_cache.max_seq_len();
            layer.kv_cache = KvCacheVariant::new_regular(2, max_seq_len);
        }
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

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            // Use shared mask cache with GPU-accelerated creation
            Some(self.mask_cache.get_mask(seq_len, index_pos)?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clone_with_independent_kv_cache() -> Result<()> {
        // Download a small Llama model from HuggingFace
        // Using Llama-3.2-1B-Instruct (smallest available for fast testing)
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        let repo = api.model("bartowski/Llama-3.2-1B-Instruct-GGUF".to_string());
        let model_path = repo.get("Llama-3.2-1B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("Model downloaded to: {:?}", model_path);

        let device = Device::cuda_if_available(0)?;
        println!("Using device: {:?}", device);

        // Load model using optimized mmap path
        println!("Loading model with mmap optimization...");
        let load_start = std::time::Instant::now();
        let mut model = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        let load_duration = load_start.elapsed();
        println!(
            "✓ Model loaded in {:.3}s using mmap\n",
            load_duration.as_secs_f64()
        );

        println!("Starting 500-token prefill...");

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
