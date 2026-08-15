//! Latent single-KV attention with sliding-window + compressed (CSA/HCA) keys.
//! Mirrors `Attention.forward` (the `start_pos == 0` prefill branch) in
//! `inference/model.py`.
//!
//! Every layer keeps a raw sliding window of the last `window_size` tokens. Compression
//! layers additionally attend to compressed KV entries — CSA layers to the top-k chosen
//! by the [`Indexer`], HCA layers to all causal entries. K and V are the *same* 512-dim
//! vector; per-head learned attention sinks add a virtual zero-value key to each softmax.
//!
//! Rather than gathering an explicit top-k index list (what the production kernel does),
//! this reference builds an additive attention mask over `[raw ‖ compressed]` keys and
//! runs dense attention — numerically identical, and far simpler to verify. The output's
//! RoPE dims are de-rotated (inverse rotation) before the grouped low-rank output
//! projection.

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::ops::softmax;

use super::compressor::Compressor;
use super::config::{Config, LayerKind};
use super::indexer::Indexer;
use super::linear::{shared_int8_pair, QLinear};
use super::rope::RotaryCache;

const NEG_BIG: f64 = -1e30;

/// One attention layer.
pub struct Attention {
    wq_a: QLinear,
    q_norm: Tensor, // [q_lora_rank]
    wq_b: QLinear,
    wkv: QLinear,
    kv_norm: Tensor, // [head_dim]
    // Per-GROUP output projection `o_g[g] @ wo_a[g]ᵀ`. The single `[ng·olr, per_group]` GGUF
    // weight is split at load into `ng` int8-KO group linears `[olr, per_group]`, so the whole
    // projection runs in int8 quant space (no per-layer F32 dequant).
    wo_a: Vec<QLinear>, // ng × [o_lora_rank, (n_heads/n_groups)*head_dim]
    wo_b: QLinear,
    attn_sink: Tensor, // [n_heads] f32
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    n_heads: usize,
    head_dim: usize,
    rope_head_dim: usize,
    n_groups: usize,
    o_lora_rank: usize,
    window_size: usize,
    ratio: usize,
    kind: LayerKind,
    eps: f64,
    softmax_scale: f64,
    // Unit weight `[head_dim]` for the per-head unweighted RMS scale, built once so
    // `rms_scale` is a single fused `rms_norm` launch instead of the eager sqr/mean/add/
    // sqrt/div chain (see `Self::rms_scale`).
    ones_hd: Tensor,
}

pub struct AttentionParams {
    pub wq_a: QLinear,
    pub q_norm: Tensor,
    pub wq_b: QLinear,
    pub wkv: QLinear,
    pub kv_norm: Tensor,
    pub wo_a: Vec<QLinear>,
    pub wo_b: QLinear,
    pub attn_sink: Tensor,
    pub compressor: Option<Compressor>,
    pub indexer: Option<Indexer>,
}

impl Attention {
    pub fn new(cfg: &Config, layer: usize, p: AttentionParams) -> Result<Self> {
        let ones_hd = Tensor::ones(cfg.head_dim, DType::F32, p.q_norm.device())?;
        Ok(Self {
            wq_a: p.wq_a,
            q_norm: p.q_norm,
            wq_b: p.wq_b,
            wkv: p.wkv,
            kv_norm: p.kv_norm,
            wo_a: p.wo_a,
            wo_b: p.wo_b,
            attn_sink: p.attn_sink,
            compressor: p.compressor,
            indexer: p.indexer,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            rope_head_dim: cfg.rope_head_dim,
            n_groups: cfg.o_groups,
            o_lora_rank: cfg.o_lora_rank,
            window_size: cfg.window_size,
            ratio: cfg.compress_ratio(layer),
            kind: cfg.layer_kind(layer),
            eps: cfg.norm_eps,
            softmax_scale: (cfg.head_dim as f64).powf(-0.5),
            ones_hd,
        })
    }

    /// Per-head unweighted RMS scale over the last dim: `x * rsqrt(mean(x²)+eps)`.
    /// A single fused `rms_norm` launch against a cached unit weight, instead of the
    /// eager sqr/mean/add/sqrt/div chain — identical math to the sibling `rms_norm`
    /// with weight ≡ 1.
    pub(crate) fn rms_scale(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.to_dtype(DType::F32)?, &self.ones_hd, self.eps as f32)
    }

    /// Prefill forward over the full prefix `x` `[b, s, dim]`. Returns `[b, s, dim]`.
    pub fn forward(&self, x: &Tensor, rope: &RotaryCache) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.n_heads, self.head_dim);
        let x = x.to_dtype(DType::F32)?;

        // wq_a and wkv both project the same `x`; quantize the activation once and share it
        // (bit-identical to two independent `forward`s — see `shared_int8_forward`).
        let (qa_raw, kv_raw) = shared_int8_pair(&x, &self.wq_a, &self.wkv)?;

        // --- query ---
        let qr = rms_norm(&qa_raw, &self.q_norm, self.eps)?; // [b,s,q_lora]
        let q = self.wq_b.forward(&qr)?.reshape((b, s, h, hd))?;
        let q = self.rms_scale(&q)?; // per-head unweighted RMS, q is [b,s,h,hd]

        // RoPE the query by TOKEN POSITION. `model.py`'s `apply_rotary_emb` indexes `freqs_cis` by
        // the sequence axis and broadcasts over heads (`view(1, seq, 1, rd)` for the 4D `q`), so
        // every head at token `t` is rotated by position `t`. `rope_last` rotates the `Minus2` axis,
        // so transpose to `[b,h,s,hd]` (seq at Minus2) before roping — NOT `[b,s,h,hd]`, where
        // Minus2 is heads (that rotated by head index — the bug confirmed against `model.py`).
        let q = q.transpose(1, 2)?.contiguous()?; // [b,h,s,hd]
        let q = self.rope_last(&q, rope, 0, false)?; // rope by token position (Minus2 = s)

        // --- raw window KV (K == V) ---
        let kv = rms_norm(&kv_raw, &self.kv_norm, self.eps)?; // [b,s,hd]
        let kv = self.rope_last(&kv, rope, 0, false)?;

        // --- compressed KV ---
        let compressed = match &self.compressor {
            Some(c) => c.forward(&x, rope)?,
            None => None,
        };
        let g = compressed.as_ref().map(|c| c.dim(1).unwrap()).unwrap_or(0);
        let kv_full = match &compressed {
            Some(c) => Tensor::cat(&[&kv, c], 1)?, // [b, s+G, hd]
            None => kv.clone(),
        };
        let k = s + g;

        // --- scores: q · kv_full ---  [b,h,s,K]  (q already [b,h,s,hd] from the RoPE transpose)
        let q_bhsd = q.contiguous()?; // [b,h,s,hd]
        let kv_t = kv_full
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, 1, hd, k))?;
        let scores = (q_bhsd.broadcast_matmul(&kv_t)? * self.softmax_scale)?; // [b,h,s,K]

        // --- attention mask over [raw ‖ compressed] ---
        let mask = self.build_mask(&x, &qr, rope, b, s, g, compressed.is_some())?; // [b,1,s,K]
        let scores = scores.broadcast_add(&mask)?;

        // --- sink softmax + value gather ---
        let o = self.sink_attend(&scores, &kv_full)?; // [b,h,s,hd]

        // --- de-rotate output RoPE dims ---
        let o = self.rope_last(&o, rope, 0, true)?; // inverse

        // --- grouped low-rank output projection ---
        // `output_proj` takes token-major [b,s,h,hd]; `o` is [b,h,s,hd] here.
        self.output_proj(&o.transpose(1, 2)?.contiguous()?, b, s)
    }

    /// DSpark `graph_dsv4` attention: **non-causal** latent attention with an **injected
    /// context**. The draft block `x` `[b, s, dim]` forms the queries; the keys/values are
    /// `[wkv(ctx) ‖ wkv(x)]` — the injected context `ctx` `[n_ctx, dim]` (the encoder's `Hctx`)
    /// projected by this layer's own `wkv`, concatenated ahead of the block's own KV. There is
    /// **no causal mask** (block-diffusion: every block position sees the whole context and every
    /// other block position). `ctx_start` / `q_start` are the absolute RoPE positions of the
    /// context and the block. SWA-only (the DSpark blocks carry no compressor). Returns
    /// `[b, s, dim]`. Mirrors the reference `forward` — same `rms_scale`/`rope_last`/`sink_attend`/
    /// `output_proj` helpers — differing only in the injected KV and the absent causal mask.
    pub fn forward_injected(
        &self,
        x: &Tensor,
        ctx: &Tensor,
        rope: &RotaryCache,
        ctx_start: usize,
        q_start: usize,
    ) -> Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        let (h, hd) = (self.n_heads, self.head_dim);
        let x = x.to_dtype(DType::F32)?;
        let n_ctx = ctx.dim(0)?;
        let ctx = ctx.to_dtype(DType::F32)?.reshape((1, n_ctx, dim))?;

        // --- queries from the draft block ---
        let qr = rms_norm(&self.wq_a.forward(&x)?, &self.q_norm, self.eps)?;
        let q = self.wq_b.forward(&qr)?.reshape((b, s, h, hd))?;
        let q = self.rms_scale(&q)?;
        let q = q.transpose(1, 2)?.contiguous()?; // [b,h,s,hd]
        let q = self.rope_last(&q, rope, q_start, false)?; // rope by block position

        // --- keys/values: injected context ‖ draft block, both via this layer's wkv ---
        let kv_b = rms_norm(&self.wkv.forward(&x)?, &self.kv_norm, self.eps)?; // [b,s,hd]
        let kv_b = self.rope_last(&kv_b, rope, q_start, false)?;
        let kv_c = rms_norm(&self.wkv.forward(&ctx)?, &self.kv_norm, self.eps)?; // [1,n_ctx,hd]
        let kv_c = self.rope_last(&kv_c, rope, ctx_start, false)?;
        let kv_c = kv_c.broadcast_as((b, n_ctx, hd))?;
        let kv_full = Tensor::cat(&[&kv_c, &kv_b], 1)?; // [b, n_ctx+s, hd]
        let k = n_ctx + s;

        // --- scores (non-causal: no additive mask) ---
        let kv_t = kv_full
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, 1, hd, k))?;
        let scores = (q.contiguous()?.broadcast_matmul(&kv_t)? * self.softmax_scale)?; // [b,h,s,k]

        // --- sink softmax + value gather, de-rotate, grouped output projection ---
        let o = self.sink_attend(&scores, &kv_full)?; // [b,h,s,hd]
        let o = self.rope_last(&o, rope, q_start, true)?; // inverse RoPE on the block positions
        self.output_proj(&o.transpose(1, 2)?.contiguous()?, b, s)
    }

    /// RoPE the trailing `rope_head_dim` dims of a `[.., seq, head_dim]` tensor (seq at
    /// `Minus2`), leaving the leading `nope` dims untouched.
    fn rope_last(
        &self,
        x: &Tensor,
        rope: &RotaryCache,
        start: usize,
        inverse: bool,
    ) -> Result<Tensor> {
        let hd = self.head_dim;
        let rd = self.rope_head_dim;
        let nope = x.narrow(D::Minus1, 0, hd - rd)?;
        let rope_part = rope.apply(&x.narrow(D::Minus1, hd - rd, rd)?, start, inverse)?;
        Tensor::cat(&[&nope, &rope_part], D::Minus1)
    }

    /// Build the additive attention mask `[b, 1, s, K]` (broadcast over heads).
    fn build_mask(
        &self,
        x: &Tensor,
        qr: &Tensor,
        rope: &RotaryCache,
        b: usize,
        s: usize,
        g: usize,
        compresses: bool,
    ) -> Result<Tensor> {
        let dev = x.device();
        // raw window: [s, s], 0 within causal window else NEG_BIG.
        let mut raw = vec![NEG_BIG as f32; s * s];
        for i in 0..s {
            let lo = (i + 1).saturating_sub(self.window_size);
            for j in lo..=i {
                raw[i * s + j] = 0.0;
            }
        }
        let raw = Tensor::from_vec(raw, (1, 1, s, s), dev)?.broadcast_as((b, 1, s, s))?;

        if !compresses || g == 0 {
            return raw.contiguous();
        }

        // compressed causal validity: [s, G], 1.0 if g < (i+1)/ratio.
        let mut causal = vec![0f32; s * g];
        for i in 0..s {
            let complete = (i + 1) / self.ratio;
            for gg in 0..complete.min(g) {
                causal[i * g + gg] = 1.0;
            }
        }
        let causal = Tensor::from_vec(causal, (1, s, g), dev)?;

        let keep = match (self.kind, &self.indexer) {
            (LayerKind::Csa, Some(indexer)) => {
                // top-k by index score among causal compressed entries.
                let score = indexer
                    .scores(x, qr, rope)?
                    .expect("indexer scores exist when g > 0"); // [b,s,G]
                let causal_add = ((&causal - 1.0)? * (-NEG_BIG))?; // 0 where causal else NEG_BIG
                let masked = score.broadcast_add(&causal_add)?; // [b,s,G]
                let top_k = indexer.top_k().min(g);
                let order = masked.arg_sort_last_dim(false)?; // desc
                let sel = order.narrow(D::Minus1, 0, top_k)?.contiguous()?; // [b,s,top_k]
                let selected = Tensor::zeros((b, s, g), DType::F32, dev)?.scatter_add(
                    &sel,
                    &Tensor::ones((b, s, top_k), DType::F32, dev)?,
                    2,
                )?;
                selected.broadcast_mul(&causal)? // [b,s,G]
            }
            _ => causal.broadcast_as((b, s, g))?.contiguous()?, // HCA: all causal
        };
        // comp_add = 0 where keep>=1 else NEG_BIG
        let comp_add = ((keep * (-NEG_BIG))? + NEG_BIG)?.reshape((b, 1, s, g))?;
        Tensor::cat(&[&raw, &comp_add], D::Minus1)
    }

    /// Softmax with a per-head learned attention sink (a virtual zero-value key), then a
    /// value gather. `scores` `[b,h,s,K]`, `values` `[b,K,hd]` → `[b,h,s,hd]`.
    fn sink_attend(&self, scores: &Tensor, values: &Tensor) -> Result<Tensor> {
        let (b, h, s, k) = scores.dims4()?;
        let hd = self.head_dim;
        // sink column: [b,h,s,1] = attn_sink[h]
        let sink = self
            .attn_sink
            .to_dtype(DType::F32)?
            .reshape((1, h, 1, 1))?
            .broadcast_as((b, h, s, 1))?;
        let aug = Tensor::cat(&[scores, &sink], D::Minus1)?; // [b,h,s,K+1]
        let probs = softmax(&aug, D::Minus1)?;
        let probs_real = probs.narrow(D::Minus1, 0, k)?.contiguous()?; // drop sink col
                                                                       // o = probs_real @ values   (broadcast values over heads)
        let v = values.reshape((b, 1, k, hd))?;
        probs_real.broadcast_matmul(&v) // [b,h,s,hd]
    }

    /// Grouped low-rank output projection: `o [b,h,s,hd]` → `[b,s,dim]`.
    pub(crate) fn output_proj(&self, o: &Tensor, b: usize, s: usize) -> Result<Tensor> {
        let (h, hd, ng, olr) = (self.n_heads, self.head_dim, self.n_groups, self.o_lora_rank);
        let per_group = (h / ng) * hd;
        // `o` is [b, s, h, hd] (TOKEN-major, exactly as the paged decode/prefill
        // kernels emit) — regroup the head axis into the `ng` o_lora groups with a
        // free reshape, NO transpose. (The paged wave used to transpose the
        // kernel's token-major output into [b,h,s,hd] just so this could transpose
        // it back — two cancelling transposes + two contiguous copies per prefill
        // seq per layer. The reference `forward`/`decode` paths, whose `sink_attend`
        // yields [b,h,s,hd], transpose into this layout before calling.)
        let o = o.reshape((b, s, ng, per_group))?;
        // Per group g: o[:,:,g,:] `[b*s, per_group]` @ wo_a[g]ᵀ -> `[b*s, olr]`, run as an int8-KO
        // linear. Concatenated over groups on the head axis → `[b, s, ng·olr]`. When every
        // `wo_a` is int8-KO the whole set runs in ONE grouped launch (see `grouped_output_proj`);
        // otherwise (Dense/Quant fixtures) the per-group loop below.
        let proj = match self.grouped_output_proj(&o, b, s)? {
            Some(p) => p,
            None => {
                let mut groups = Vec::with_capacity(ng);
                for (g, wo_a_g) in self.wo_a.iter().enumerate() {
                    let og = o
                        .narrow(2, g, 1)?
                        .contiguous()?
                        .reshape((b * s, per_group))?;
                    let proj_g = wo_a_g.forward(&og)?; // [b*s, olr]
                    groups.push(proj_g.reshape((b, s, 1, olr))?);
                }
                Tensor::cat(&groups, 2)?.reshape((b, s, ng * olr))?
            }
        };
        self.wo_b.forward(&proj)
    }

    /// One grouped int8-KO matmul over all `ng` output-projection groups instead of `ng`
    /// separate quantize+matmul launches: the groups become contiguous row-blocks of a single
    /// operand `[ng·b·s, per_group]`, each block multiplied by its own `wo_a[g]`
    /// (`grouped_qmatmul` with block offsets). Returns `None` unless every `wo_a` is int8-KO on
    /// CUDA and the shapes meet the kernel's alignment (`per_group % 128`, `o_lora_rank % 32`),
    /// so the caller falls back to the per-group loop. Bit-identical to that loop: q8a128
    /// quantizes per row (independent of grouping) and the grouped kernel equals the per-weight
    /// dense matmul on each block.
    #[cfg(feature = "cuda")]
    fn grouped_output_proj(&self, o: &Tensor, b: usize, s: usize) -> Result<Option<Tensor>> {
        use candle::quantized::cuda::{grouped_qmatmul, to_dynamic};
        use candle::quantized::Int8Mode;
        // Decode signature only (`s == 1`): each group sees few rows (`b` = the concurrent
        // decode count), so the launch-bound path wins by collapsing `ng` launches into one.
        // Prefill (`s > 1`) feeds each group many rows — those large per-group GEMMs run
        // faster as separate dense matmuls than through the MoE grouped `gemx` kernel — so it
        // keeps the per-group loop.
        if s != 1 {
            return Ok(None);
        }
        let (ng, olr) = (self.n_groups, self.o_lora_rank);
        let per_group = (self.n_heads / ng) * self.head_dim;
        if per_group % 128 != 0 || olr % 32 != 0 {
            return Ok(None);
        }
        let dev = match o.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let mut ptrs = Vec::with_capacity(ng);
        let mut wdtype = None;
        for w in &self.wo_a {
            match w.ko_weight() {
                Some((p, dt)) => {
                    ptrs.push(p);
                    wdtype = Some(dt);
                }
                None => return Ok(None),
            }
        }
        let wdtype = match wdtype {
            Some(d) => d,
            None => return Ok(None),
        };
        let bs = b * s;
        // [b,s,ng,per_group] -> group-major [ng,b,s,per_group] -> [ng*b*s, per_group]
        let o_g = o
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .reshape((ng * bs, per_group))?
            .to_dtype(DType::F32)?;
        let op = to_dynamic(&o_g, Int8Mode::Performance, &dev)?;
        let offsets: Vec<i32> = (0..=ng).map(|g| (g * bs) as i32).collect();
        let out = grouped_qmatmul(op.as_dynamic(), &ptrs, wdtype, olr, &offsets, &dev)?; // [ng*bs, olr] f32
                                                                                         // [ng*bs, olr] -> [ng, bs, olr] -> [bs, ng, olr] -> [b, s, ng*olr]
        let proj = out
            .reshape((ng, bs, olr))?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((b, s, ng * olr))?;
        Ok(Some(proj))
    }

    /// No grouped int8 path without CUDA — the caller runs the per-group loop.
    #[cfg(not(feature = "cuda"))]
    fn grouped_output_proj(&self, _o: &Tensor, _b: usize, _s: usize) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

impl Attention {
    // ── Weight/config accessors for the paged-kernel attention path (the
    // projections stay host-side; the attention math moves to the kernel). ──
    pub(crate) fn wq_a(&self) -> &QLinear {
        &self.wq_a
    }
    pub(crate) fn q_norm(&self) -> &Tensor {
        &self.q_norm
    }
    pub(crate) fn wq_b(&self) -> &QLinear {
        &self.wq_b
    }
    pub(crate) fn wkv(&self) -> &QLinear {
        &self.wkv
    }
    pub(crate) fn kv_norm(&self) -> &Tensor {
        &self.kv_norm
    }
    pub(crate) fn attn_sink(&self) -> &Tensor {
        &self.attn_sink
    }
    pub(crate) fn compressor(&self) -> Option<&Compressor> {
        self.compressor.as_ref()
    }
    pub(crate) fn indexer(&self) -> Option<&Indexer> {
        self.indexer.as_ref()
    }
    pub(crate) fn kind(&self) -> LayerKind {
        self.kind
    }
    pub(crate) fn window_size(&self) -> usize {
        self.window_size
    }
    pub(crate) fn n_heads(&self) -> usize {
        self.n_heads
    }
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub(crate) fn softmax_scale(&self) -> f64 {
        self.softmax_scale
    }
    pub(crate) fn eps(&self) -> f64 {
        self.eps
    }
}

impl Attention {
    /// Build the incremental (decode) form of this attention layer: a stateful streamer that
    /// accepts one token per `step` and returns that query's attention output `[1, 1, dim]`,
    /// bit-for-bit identical to the corresponding row of `forward` over the full prefix.
    ///
    /// Handles all three layer kinds: SWA (window only), HCA (window ‖ all-causal-compressed),
    /// and CSA (window ‖ indexer top-k compressed).
    pub fn decoder(&self) -> Result<IncrementalAttention<'_>> {
        Ok(IncrementalAttention {
            a: self,
            window: Vec::with_capacity(self.window_size),
            comp: self.compressor.as_ref().map(|c| c.incremental()),
            comp_entries: Vec::new(),
            idx: self.indexer.as_ref().map(|ix| ix.incremental()),
            pos: 0,
            capture_indexer_space: false,
            captured_space: None,
        })
    }
}

/// Streaming (decode-time) counterpart to [`Attention`]. Maintains the raw sliding-window ring,
/// the incremental compressed cache, and (CSA) the incremental indexer, and for each new token
/// attends the single query over `window ‖ selected-compressed` keys with the same per-head sink
/// softmax and output de-rotation as prefill. Because prefill masks excluded keys with
/// `exp(-1e30) == 0` (an exact f32 underflow) and gathers values in increasing position order,
/// summing over only the in-window / selected keys — in that same order — reproduces the prefill
/// row bit-for-bit (proven by `incremental_attention_matches_prefill_*`).
pub struct IncrementalAttention<'a> {
    a: &'a Attention,
    /// Raw window KV rows, each `[head_dim]` and already RoPE'd at its own position; holds at
    /// most `window_size` rows (oldest evicted), oldest→newest.
    window: Vec<Tensor>,
    /// Incremental attention-side compressor (`None` for SWA), driving `comp_entries`.
    comp: Option<super::compressor::IncrementalCompressor>,
    /// Compressed KV entries emitted so far, each `[head_dim]`, RoPE'd at group-start positions.
    comp_entries: Vec<Tensor>,
    /// Incremental indexer (CSA layers only) selecting which compressed entries the query attends.
    idx: Option<super::indexer::IncrementalIndexer<'a>>,
    /// Next query position (0-based).
    pos: usize,
    /// When set, `step` stores the Indexer's (query, weights, entry keys)
    /// spaces for its query — the per-layer recall-validation capture.
    capture_indexer_space: bool,
    /// The captured spaces from the most recent `step` (see
    /// [`IncrementalIndexer::capture_space`]).
    pub captured_space: Option<(Tensor, Tensor, Tensor)>,
}

impl IncrementalAttention<'_> {
    /// Arm/disarm the per-step Indexer space capture (recall validation).
    pub fn set_capture_indexer_space(&mut self, on: bool) {
        self.capture_indexer_space = on;
        if !on {
            self.captured_space = None;
        }
    }

    /// Feed the next token's hidden state `x` (`[dim]` / `[1, dim]` / `[1, 1, dim]`) and return
    /// its attention output `[1, 1, dim]` — equal to row `pos` of the prefill `forward`.
    pub fn step(&mut self, x: &Tensor, rope: &RotaryCache) -> Result<Tensor> {
        let a = self.a;
        let (h, hd) = (a.n_heads, a.head_dim);
        let pos = self.pos;
        // A single token: its element count is the model `dim` regardless of `[dim]`/`[1,dim]`
        // /`[1,1,dim]` shape.
        let din = x.elem_count();
        let x = x.reshape((1, 1, din))?.to_dtype(DType::F32)?;

        // wq_a and wkv both project the same token `x`; quantize the activation once and share it.
        let (qa_raw, kv_raw) = shared_int8_pair(&x, &a.wq_a, &a.wkv)?;

        // --- query: shared low-rank qr → wq_b → per-head RMS scale → RoPE at `pos` ---
        let qr = rms_norm(&qa_raw, &a.q_norm, a.eps)?; // [1,1,q_lora]
        let q = a.wq_b.forward(&qr)?.reshape((1, 1, h, hd))?;
        let q = a.rms_scale(&q)?;
        // RoPE the query by TOKEN POSITION `pos` (matches the fixed prefill / `model.py`: q roped
        // by seq position, broadcast over heads). Transpose to `[1,h,1,hd]` so seq is at Minus2.
        let q = q.transpose(1, 2)?.contiguous()?; // [1,h,1,hd]
        let q = a.rope_last(&q, rope, pos, false)?; // rope by token position `pos`

        // --- raw window KV (K == V): RoPE at `pos`, append to the ring ---
        let kv = rms_norm(&kv_raw, &a.kv_norm, a.eps)?; // [1,1,hd]
        let kv = a.rope_last(&kv, rope, pos, false)?;
        self.window.push(kv.reshape((hd,))?);
        if self.window.len() > a.window_size {
            self.window.remove(0);
        }

        // --- compressed KV: emit a new attention entry when this token completes a group ---
        if let Some(comp) = self.comp.as_mut() {
            if let Some(entry) = comp.push(&x, rope)? {
                self.comp_entries.push(entry.reshape((hd,))?);
            }
        }
        // --- indexer cache (CSA): accumulate one indexer entry per completed group ---
        if let Some(idx) = self.idx.as_mut() {
            idx.push(&x, rope)?;
        }

        // --- which compressed entries this query attends (sorted ascending group order) ---
        //   SWA: none; HCA: all causal; CSA: indexer top-k.
        if self.capture_indexer_space {
            self.captured_space = match self.idx.as_ref() {
                Some(idx) => idx.capture_space(&x, &qr, rope, pos)?,
                None => None,
            };
        }
        let selected: Vec<usize> = match (a.kind, self.idx.as_ref()) {
            (LayerKind::Csa, Some(idx)) => idx.select(&x, &qr, rope, pos)?,
            (LayerKind::Hca, _) => (0..self.comp_entries.len()).collect(),
            _ => Vec::new(),
        };

        // --- gather the attended keys: window ‖ selected compressed entries ---
        let mut keys: Vec<&Tensor> = self.window.iter().collect();
        for &gg in &selected {
            keys.push(&self.comp_entries[gg]);
        }
        let kcount = keys.len();
        let kv_full = Tensor::stack(&keys, 0)?; // [K, hd]

        // --- scores: q · kvᵀ · scale  → [1,h,1,K]  (q already [1,h,1,hd] from the RoPE transpose)
        let q_bhsd = q.contiguous()?; // [1,h,1,hd]
        let kv_t = kv_full.t()?.contiguous()?.reshape((1, 1, hd, kcount))?;
        let scores = (q_bhsd.broadcast_matmul(&kv_t)? * a.softmax_scale)?; // [1,h,1,K]

        // --- sink softmax + value gather → de-rotate at `pos` → grouped output projection ---
        let o = a.sink_attend(&scores, &kv_full.reshape((1, kcount, hd))?)?; // [1,h,1,hd]
        let o = a.rope_last(&o, rope, pos, true)?;
        // `output_proj` takes token-major [b,s,h,hd]; `o` is [1,h,1,hd] here.
        let out = a.output_proj(&o.transpose(1, 2)?.contiguous()?, 1, 1)?; // [1,1,dim]

        self.pos += 1;
        Ok(out)
    }
}

/// RMSNorm with a learned weight: `x * rsqrt(mean(x²)+eps) * w`.
pub(crate) fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    // Fused rmsnorm (one launch) instead of the ~6 eager ops (sqr/mean/add/sqrt/
    // div/mul). `candle_nn::ops::rms_norm` is `x·rsqrt(mean(x²)+eps)·w`, exactly
    // this function's math (`mean(x²) = Σx²/hidden`), computed in f32 for f32
    // input — bit-equivalent within reduction-order tolerance. Called 3-4× per
    // layer, so this removes hundreds of tiny launches per token.
    let x = x.to_dtype(DType::F32)?;
    let w = w.to_dtype(DType::F32)?;
    candle_nn::ops::rms_norm(&x, &w, eps as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, IndexOp};

    /// The fused unit-weight `rms_norm` used by `Attention::rms_scale` reproduces the eager
    /// `x·rsqrt(mean(x²)+eps)` it replaces (within f32 reduction-order tolerance).
    #[test]
    fn rms_scale_fused_matches_eager() -> Result<()> {
        let dev = Device::Cpu;
        let hd = 512usize;
        let eps = 1e-6f64;
        let x = Tensor::randn(0f32, 2.0, (2, 3, 8, hd), &dev)?;
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let eager = x.broadcast_div(&(ms + eps)?.sqrt()?)?;
        let ones = Tensor::ones(hd, DType::F32, &dev)?;
        let fused = candle_nn::ops::rms_norm(&x, &ones, eps as f32)?;
        let diff = (eager - fused)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "rms_scale fused vs eager max abs diff {diff}");
        Ok(())
    }

    /// The fused weighted `rms_norm` used by the compressor's `rms_norm`/`rms_norm_entry`
    /// reproduces the eager `x·rsqrt(mean(x²)+eps)·w` it replaces.
    #[test]
    fn rms_norm_weighted_fused_matches_eager() -> Result<()> {
        let dev = Device::Cpu;
        let hd = 512usize;
        let eps = 1e-6f64;
        let x = Tensor::randn(0f32, 2.0, (5, hd), &dev)?;
        let w = Tensor::randn(1f32, 0.3, hd, &dev)?;
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let eager = x.broadcast_div(&(ms + eps)?.sqrt()?)?.broadcast_mul(&w)?;
        let fused = candle_nn::ops::rms_norm(&x, &w, eps as f32)?;
        let diff = (eager - fused)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "weighted rms_norm fused vs eager max abs diff {diff}"
        );
        Ok(())
    }

    /// The grouped int8-KO output projection (`grouped_output_proj`, decode `s == 1`)
    /// reproduces the per-group loop it replaces: same permute/reshape/offset layout, and
    /// `grouped_qmatmul` equals the per-weight dense matmul on each block. Guards the
    /// reshape/offset logic that the full-model gate would otherwise be the only cover for.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn grouped_output_proj_matches_per_group() -> Result<()> {
        use candle::quantized::{GgmlDType, Int8Mode, QMatMul, QTensor};
        let dev = Device::new_cuda(0)?;
        let (h, hd, ng, olr) = (8usize, 128usize, 2usize, 64usize);
        let per_group = (h / ng) * hd; // 512 (% 128 == 0)
        let dim = 256usize; // width of the dummy non-wo_a projections; unused by output_proj

        let mut cfg = Config::tiny();
        cfg.n_heads = h;
        cfg.head_dim = hd;
        cfg.o_groups = ng;
        cfg.o_lora_rank = olr;
        cfg.dim = dim;

        // int8-KO `wo_a[g]` = `[olr, per_group]` (the twin the engine path loads).
        let mk_ko = || -> Result<QLinear> {
            let w = Tensor::randn(0f32, 0.3, (olr, per_group), &dev)?;
            let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
            let ko = QMatMul::from_qtensor(qt)?.repack_for_optimization(Int8Mode::Performance)?;
            Ok(QLinear::from_int8(ko))
        };
        let wo_a: Vec<QLinear> = (0..ng).map(|_| mk_ko()).collect::<Result<_>>()?;

        let p = AttentionParams {
            wq_a: dense(cfg.q_lora_rank, dim, &dev)?,
            q_norm: Tensor::ones(cfg.q_lora_rank, DType::F32, &dev)?,
            wq_b: dense(h * hd, cfg.q_lora_rank, &dev)?,
            wkv: dense(hd, dim, &dev)?,
            kv_norm: Tensor::ones(hd, DType::F32, &dev)?,
            wo_a,
            wo_b: dense(dim, ng * olr, &dev)?,
            attn_sink: Tensor::randn(0f32, 1.0, h, &dev)?,
            compressor: None,
            indexer: None,
        };
        let att = Attention::new(&cfg, 0, p)?;

        // `grouped_output_proj` takes the reshaped `o` `[b, s, ng, per_group]` (s == 1 decode).
        let b = 3usize;
        let o = Tensor::randn(0f32, 1.0, (b, 1, ng, per_group), &dev)?;
        let grouped = att
            .grouped_output_proj(&o, b, 1)?
            .expect("int8-KO wo_a + aligned shapes → grouped path");

        // Per-group reference (the loop the grouped path replaces).
        let mut groups = Vec::with_capacity(ng);
        for (g, wo_a_g) in att.wo_a.iter().enumerate() {
            let og = o.narrow(2, g, 1)?.contiguous()?.reshape((b, per_group))?;
            groups.push(wo_a_g.forward(&og)?.reshape((b, 1, olr))?);
        }
        let reference = Tensor::cat(&groups, 2)?.reshape((b, 1, ng * olr))?;

        let diff = (grouped - reference)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-3, "grouped vs per-group max abs diff {diff}");
        Ok(())
    }

    /// Sink softmax: with a very negative sink the result approaches plain softmax; with a
    /// very positive sink the output magnitude shrinks (mass leaks to the zero-value key).
    #[test]
    fn sink_softmax_scalar() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let hd = cfg.head_dim;
        // Minimal attention just to reach sink_attend; only n_heads/head_dim used there.
        let att = mk_dummy(&cfg, &dev)?;
        let (b, h, s, k) = (1usize, cfg.n_heads, 2usize, 3usize);
        let scores = Tensor::randn(0f32, 1.0, (b, h, s, k), &dev)?;
        let values = Tensor::randn(0f32, 1.0, (b, k, hd), &dev)?;

        // Reference with the module's own sink values.
        let sinkv = att.attn_sink.to_vec1::<f32>()?;
        let sc = scores.i(0)?.to_vec3::<f32>()?; // [h,s,k]
        let vv = values.i(0)?.to_vec2::<f32>()?; // [k,hd]
        let got = att.sink_attend(&scores, &values)?.i(0)?.to_vec3::<f32>()?; // [h,s,hd]
        for head in 0..h {
            for i in 0..s {
                let mut logits: Vec<f32> = (0..k).map(|j| sc[head][i][j]).collect();
                logits.push(sinkv[head]); // sink col
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
                let z: f32 = exps.iter().sum();
                for c in 0..hd {
                    let mut acc = 0f32;
                    for j in 0..k {
                        acc += exps[j] / z * vv[j][c];
                    }
                    assert!(
                        (got[head][i][c] - acc).abs() < 1e-4,
                        "h{head}i{i}c{c}: {} vs {acc}",
                        got[head][i][c]
                    );
                }
            }
        }
        Ok(())
    }

    /// Full forward runs and stays finite for each layer kind (SWA, CSA, HCA).
    #[test]
    fn forward_all_layer_kinds() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        for layer in [0usize, 2, 3] {
            let kind = cfg.layer_kind(layer);
            let rope = layer_rope(&cfg, layer, &dev)?;
            let att = mk_attention(&cfg, layer, &dev)?;
            let x = Tensor::randn(0f32, 1.0, (2, 20, cfg.dim), &dev)?;
            let out = att.forward(&x, &rope)?;
            assert_eq!(out.dims(), &[2, 20, cfg.dim], "kind {kind:?}");
            assert!(
                out.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| v.is_finite()),
                "non-finite output for {kind:?}"
            );
        }
        Ok(())
    }

    /// The streaming (decode) attention emits, token-by-token, outputs numerically equal to the
    /// prefill `forward` over the full prefix — the SWA/HCA decode-equivalence foundation.
    /// Prefill masks excluded keys to `exp(-1e30) == 0` and gathers values in position order, so
    /// summing only the in-window / causal keys reproduces each row.
    fn incremental_attention_case(cfg: &Config, layer: usize, s: usize) -> Result<()> {
        let dev = Device::Cpu;
        let rope = layer_rope(cfg, layer, &dev)?;
        let att = mk_attention(cfg, layer, &dev)?;
        let x = Tensor::randn(0f32, 1.0, (1, s, cfg.dim), &dev)?;

        let prefill = att.forward(&x, &rope)?; // [1, s, dim]

        let mut dec = att.decoder()?;
        let mut outs: Vec<Tensor> = Vec::with_capacity(s);
        for t in 0..s {
            outs.push(dec.step(&x.i((0, t))?, &rope)?); // [1,1,dim]
        }
        let streamed = Tensor::cat(&outs, 1)?; // [1, s, dim]

        let a = prefill.flatten_all()?.to_vec1::<f32>()?;
        let b = streamed.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 2e-5,
            "layer {layer} ({:?}): prefill vs streamed max|Δ| = {max_abs}",
            cfg.layer_kind(layer)
        );
        Ok(())
    }

    #[test]
    fn incremental_attention_matches_prefill_swa() -> Result<()> {
        // SWA (layer 0): s > window_size (8) exercises the ring eviction.
        incremental_attention_case(&Config::tiny(), 0, 12)
    }

    #[test]
    fn incremental_attention_matches_prefill_hca() -> Result<()> {
        // HCA layer with a small ratio so compressed groups form at tiny `s`: attends
        // window ‖ all-causal-compressed. compress_ratios[3] = 2 → Hca (from_ratio: else-arm).
        let mut cfg = Config::tiny();
        cfg.compress_ratios = vec![0, 0, 4, 2, 4, 128];
        assert_eq!(cfg.layer_kind(3), LayerKind::Hca);
        incremental_attention_case(&cfg, 3, 11)
    }

    #[test]
    fn incremental_attention_matches_prefill_csa() -> Result<()> {
        // CSA layer (ratio 4, overlapping compressor + indexer). `index_topk = 2` is smaller
        // than the 4 groups formed at s=18, so the indexer top-k genuinely selects a subset —
        // exercising the incremental indexer scoring + selection against the prefill path.
        let mut cfg = Config::tiny();
        cfg.index_topk = 2;
        assert_eq!(cfg.layer_kind(2), LayerKind::Csa);
        incremental_attention_case(&cfg, 2, 18)
    }

    fn layer_rope(cfg: &Config, layer: usize, dev: &Device) -> Result<RotaryCache> {
        let (theta, orig) = cfg.rope_params(layer);
        RotaryCache::new(
            cfg.rope_head_dim,
            theta,
            orig,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            dev,
        )
    }

    fn mk_dummy(cfg: &Config, dev: &Device) -> Result<Attention> {
        mk_attention(cfg, 0, dev)
    }

    fn dense(rows: usize, cols: usize, dev: &Device) -> Result<QLinear> {
        Ok(QLinear::from_weight(Tensor::randn(
            0f32,
            0.5,
            (rows, cols),
            dev,
        )?))
    }

    fn mk_attention(cfg: &Config, layer: usize, dev: &Device) -> Result<Attention> {
        let (h, hd, _rd, ng, olr) = (
            cfg.n_heads,
            cfg.head_dim,
            cfg.rope_head_dim,
            cfg.o_groups,
            cfg.o_lora_rank,
        );
        let (compressor, indexer) = build_compress(cfg, layer, dev)?;
        let p = AttentionParams {
            wq_a: dense(cfg.q_lora_rank, cfg.dim, dev)?,
            q_norm: Tensor::ones(cfg.q_lora_rank, DType::F32, dev)?,
            wq_b: dense(h * hd, cfg.q_lora_rank, dev)?,
            wkv: dense(hd, cfg.dim, dev)?,
            kv_norm: Tensor::ones(hd, DType::F32, dev)?,
            wo_a: (0..ng)
                .map(|_| dense(olr, (h / ng) * hd, dev))
                .collect::<Result<Vec<_>>>()?,
            wo_b: dense(cfg.dim, ng * olr, dev)?,
            attn_sink: Tensor::randn(0f32, 1.0, h, dev)?,
            compressor,
            indexer,
        };
        Attention::new(cfg, layer, p)
    }

    fn build_compress(
        cfg: &Config,
        layer: usize,
        dev: &Device,
    ) -> Result<(Option<Compressor>, Option<Indexer>)> {
        if !cfg.layer_kind(layer).compresses() {
            return Ok((None, None));
        }
        let ratio = cfg.compress_ratio(layer);
        let hd = cfg.head_dim;
        let coff = if ratio == 4 { 2 } else { 1 };
        let compressor = Compressor::new(
            Tensor::randn(0f32, 0.5, (coff * hd, cfg.dim), dev)?,
            Tensor::randn(0f32, 0.5, (coff * hd, cfg.dim), dev)?,
            Tensor::randn(0f32, 0.5, (ratio, coff * hd), dev)?,
            Tensor::ones(hd, DType::F32, dev)?,
            ratio,
            hd,
            cfg.rope_head_dim,
            cfg.norm_eps,
        );
        let indexer = if cfg.layer_kind(layer).has_indexer() {
            let ihd = cfg.index_head_dim;
            let icoff = if ratio == 4 { 2 } else { 1 };
            let icomp = Compressor::new(
                Tensor::randn(0f32, 0.5, (icoff * ihd, cfg.dim), dev)?,
                Tensor::randn(0f32, 0.5, (icoff * ihd, cfg.dim), dev)?,
                Tensor::randn(0f32, 0.5, (ratio, icoff * ihd), dev)?,
                Tensor::ones(ihd, DType::F32, dev)?,
                ratio,
                ihd,
                cfg.rope_head_dim,
                cfg.norm_eps,
            );
            Some(Indexer::new(
                QLinear::from_weight(Tensor::randn(
                    0f32,
                    0.5,
                    (cfg.index_n_heads * ihd, cfg.q_lora_rank),
                    dev,
                )?),
                Tensor::randn(0f32, 0.5, (cfg.index_n_heads, cfg.dim), dev)?,
                icomp,
                cfg.index_n_heads,
                ihd,
                cfg.rope_head_dim,
                cfg.index_topk,
            ))
        } else {
            None
        };
        Ok((Some(compressor), indexer))
    }
}
