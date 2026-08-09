//! CSA lightning `Indexer`: scores every compressed KV entry against the query and
//! selects the top-k to attend to. Mirrors `Indexer.forward` in `inference/model.py`.
//!
//! Two simplifications vs the reference, both exact or within QAT tolerance:
//! * The randomized Hadamard rotation applied to both `q` and the indexer cache cancels
//!   in the score dot product (`Hq · Hk = q · k` for orthogonal `H`), so it is omitted.
//! * FP4 fake-quantization of `q`/cache is the P7 layer; omitting it keeps the scores at
//!   full precision (the model was FP4-QAT'd to tolerate that quant).
//!
//! The module returns the raw per-entry scores `[b, s, G]`; causal masking and the
//! top-k cut are applied by the attention module, which owns the query positions.

use candle::{DType, Result, Tensor, D};

use super::compressor::Compressor;
use super::linear::QLinear;
use super::rope::RotaryCache;

/// A CSA indexer for one layer.
#[derive(Debug, Clone)]
pub struct Indexer {
    wq_b: QLinear,         // [n_heads*head_dim, q_lora_rank] — int8-KO on the engine path
    weights_proj: QLinear, // [n_heads, dim] — int8-KO on the engine path
    compressor: Compressor,
    n_heads: usize,
    head_dim: usize,
    rope_head_dim: usize,
    top_k: usize,
    softmax_scale: f64,
}

impl Indexer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq_b: impl Into<QLinear>,
        weights_proj: impl Into<QLinear>,
        compressor: Compressor,
        n_heads: usize,
        head_dim: usize,
        rope_head_dim: usize,
        top_k: usize,
    ) -> Self {
        Self {
            wq_b: wq_b.into(),
            weights_proj: weights_proj.into(),
            compressor,
            n_heads,
            head_dim,
            rope_head_dim,
            top_k,
            softmax_scale: (head_dim as f64).powf(-0.5),
        }
    }

    /// The Indexer's query-side spaces for the token at `pos`: the roped
    /// per-head query `[n_heads, head_dim]` and the per-head gate weights
    /// `[n_heads]` (the `hd^-0.5 · h^-0.5` scale folded in) — the score is
    /// `Σ_h relu(q_h·k)·w_h` against any entry-key set (Vec entries on the
    /// reference path, the `FloatGallery` on the kernel path).
    pub fn query_space(
        &self,
        x: &Tensor,
        qr: &Tensor,
        rope: &RotaryCache,
        pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd, rd) = (self.n_heads, self.head_dim, self.rope_head_dim);
        // q = wq_b(qr) → per-head → RoPE the trailing rd dims by TOKEN POSITION
        // `pos`, broadcast over heads (matches `Indexer::scores` / `model.py`).
        // Transpose to [1,h,1,hd] so `rope.apply`'s Minus2 axis is seq (len 1).
        let qr = qr.reshape((1, 1, ()))?.to_dtype(DType::F32)?;
        let q = self
            .wq_b
            .forward(&qr)?
            .reshape((1, 1, h, hd))?
            .transpose(1, 2)?
            .contiguous()?; // [1,h,1,hd]
        let q_nope = q.narrow(D::Minus1, 0, hd - rd)?;
        let q_rope = rope.apply(&q.narrow(D::Minus1, hd - rd, rd)?, pos, false)?;
        let q = Tensor::cat(&[&q_nope, &q_rope], D::Minus1)?.reshape((h, hd))?;

        let scale = self.softmax_scale * (h as f64).powf(-0.5);
        let weights = (self
            .weights_proj
            .forward(&x.reshape((1, ()))?.to_dtype(DType::F32)?)?
            * scale)?
            .reshape(h)?; // [h]
        Ok((q, weights))
    }

    /// Batched form of [`Self::query_space`]'s two GEMMs across `n` concurrent
    /// decode sessions: `xs` `[n, dim]` (raw hiddens) and `qr_all` `[n,
    /// q_lora_rank]` (the shared q-normed low-rank) → the PRE-RoPE per-head query
    /// `q_raw` `[n, n_heads, head_dim]` and the gate weights `[n, n_heads]`. Both
    /// projections are row-independent, so this is bit-identical per row to
    /// `query_space`; the position-dependent RoPE stays per session in
    /// [`Self::rope_query`] (cheap elementwise, applied at each session's own
    /// decode position).
    pub fn query_gemm_batched(&self, xs: &Tensor, qr_all: &Tensor) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.n_heads, self.head_dim);
        let n = qr_all.dim(0)?;
        let q_raw = self
            .wq_b
            .forward(&qr_all.to_dtype(DType::F32)?)?
            .reshape((n, h, hd))?;
        let scale = self.softmax_scale * (h as f64).powf(-0.5);
        let weights =
            (self.weights_proj.forward(&xs.to_dtype(DType::F32)?)? * scale)?.reshape((n, h))?;
        Ok((q_raw, weights))
    }

    /// Per-session RoPE for one row of [`Self::query_gemm_batched`]'s `q_raw`
    /// (`q_raw_row` `[n_heads, head_dim]`) at token position `pos` — the exact
    /// rotation `query_space` applies (transpose to `[1,h,1,hd]` so RoPE's seq
    /// axis is length 1, rotate the trailing `rope_head_dim`), returning the
    /// roped per-head query `[n_heads, head_dim]`.
    pub fn rope_query(&self, q_raw_row: &Tensor, rope: &RotaryCache, pos: usize) -> Result<Tensor> {
        let (h, hd, rd) = (self.n_heads, self.head_dim, self.rope_head_dim);
        let q = q_raw_row
            .reshape((1, 1, h, hd))?
            .transpose(1, 2)?
            .contiguous()?; // [1,h,1,hd]
        let q_nope = q.narrow(D::Minus1, 0, hd - rd)?;
        let q_rope = rope.apply(&q.narrow(D::Minus1, hd - rd, rd)?, pos, false)?;
        Tensor::cat(&[&q_nope, &q_rope], D::Minus1)?.reshape((h, hd))
    }

    /// A streaming compressor over the Indexer's own key space — the kernel
    /// path drives this directly (`push_raw`) to feed the `FloatGallery`.
    pub fn incremental_compressor(&self) -> super::compressor::IncrementalCompressor {
        self.compressor.incremental()
    }

    /// The Indexer's key-space compressor (shared layer weights) — used to batch
    /// its projection across concurrent decode sessions before the per-session
    /// stateful push.
    pub fn compressor(&self) -> &Compressor {
        &self.compressor
    }

    /// The Indexer's roped per-head query for the token at `pos`, flattened to a
    /// single band `[n_heads * head_dim]` in `(head, dim)` order — the exact
    /// input to `WideQSig::from_band` (Artifact D of
    /// docs/deepseek_turn_seal_persistence.md). The provenance wide-Q for a
    /// DeepSeek turn is `sign` of this band per Indexer head: the model's
    /// LEARNED significance space, read straight from the Indexer, rather than
    /// the R16 cross-layer sign-fold every other model uses. The head ordering
    /// matches `from_band`'s `(head × head_dim + dim)` bit layout, so the packed
    /// signature plugs into the unchanged selection scan.
    pub fn query_band(
        &self,
        x: &Tensor,
        qr: &Tensor,
        rope: &RotaryCache,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let (q, _w) = self.query_space(x, qr, rope, pos)?;
        q.flatten_all()?.to_vec1::<f32>()
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Per-compressed-entry relevance scores `[b, s, G]` for the query represented by the
    /// shared low-rank `qr` `[b, s, q_lora_rank]` and the raw hidden `x` `[b, s, dim]`.
    /// Returns `None` when the prefix is too short to form any compressed entry.
    pub fn scores(&self, x: &Tensor, qr: &Tensor, rope: &RotaryCache) -> Result<Option<Tensor>> {
        let (b, s, _) = x.dims3()?;
        let h = self.n_heads;
        let hd = self.head_dim;
        let rd = self.rope_head_dim;

        // Indexer compressed cache: [b, G, hd] (already RMSNorm'd + RoPE'd by the compressor).
        let kv = match self.compressor.forward(x, rope)? {
            None => return Ok(None),
            Some(kv) => kv,
        };
        let g = kv.dim(1)?;

        // q = wq_b(qr) -> per-head, RoPE the trailing rd dims by TOKEN POSITION (positions 0..s),
        // broadcast over heads — matching `model.py` (`freqs_cis` indexed by seq, `view(1,seq,1,rd)`).
        // Transpose to [b,h,s,hd] so `rope.apply`'s Minus2 axis is seq, not heads.
        let qr = qr.to_dtype(DType::F32)?;
        let q = self
            .wq_b
            .forward(&qr)?
            .reshape((b, s, h, hd))?
            .transpose(1, 2)?
            .contiguous()?; // [b,h,s,hd]
        let q_nope = q.narrow(D::Minus1, 0, hd - rd)?;
        let q_rope = rope.apply(&q.narrow(D::Minus1, hd - rd, rd)?, 0, false)?; // Minus2 = s
        let q = Tensor::cat(&[&q_nope, &q_rope], D::Minus1)?
            .transpose(1, 2)?
            .contiguous()?; // [b,s,h,hd]

        // score[b,s,h,g] = q · kv  (K = the compressed entries)
        let q3 = q.reshape((b, s * h, hd))?;
        let kv_t = kv.transpose(1, 2)?.contiguous()?; // [b, hd, g]
        let score = q3.matmul(&kv_t)?.reshape((b, s, h, g))?;

        // per-head gate weights, scaled by softmax_scale * n_heads^-0.5
        let scale = self.softmax_scale * (h as f64).powf(-0.5);
        let weights = (self.weights_proj.forward(&x.to_dtype(DType::F32)?)? * scale)?; // [b,s,h]

        // index_score = sum_h relu(score) * weights
        let out = score
            .relu()?
            .broadcast_mul(&weights.unsqueeze(D::Minus1)?)?
            .sum(2)?; // [b,s,g]
        Ok(Some(out))
    }

    /// Build the incremental (decode) form of this indexer: maintains the indexer-side
    /// compressed cache one token at a time and, per query, selects the top-k causal entries.
    pub fn incremental(&self) -> IncrementalIndexer<'_> {
        IncrementalIndexer {
            idx: self,
            comp: self.compressor.incremental(),
            entries: Vec::new(),
        }
    }
}

/// Streaming (decode-time) counterpart to [`Indexer`]. Accumulates the indexer's compressed
/// cache token-by-token (via an [`IncrementalCompressor`], entries bit-identical to prefill) and
/// per query selects the top-k entries by index score. At decode step `t` every accumulated
/// entry is causal, so selecting the `min(top_k, n)` highest-scored entries reproduces prefill's
/// `top_k`-over-`G`-then-`×causal` masking exactly. Selected group indices are returned **sorted
/// ascending** so the caller gathers compressed keys in the same column order prefill uses.
pub struct IncrementalIndexer<'a> {
    idx: &'a Indexer,
    comp: super::compressor::IncrementalCompressor,
    /// Indexer compressed entries emitted so far, each `[index_head_dim]`.
    entries: Vec<Tensor>,
}

impl IncrementalIndexer<'_> {
    /// Feed the next token; when it completes a group, append the emitted indexer entry.
    pub fn push(&mut self, x: &Tensor, rope: &RotaryCache) -> Result<()> {
        if let Some(entry) = self.comp.push(x, rope)? {
            self.entries.push(entry.reshape((self.idx.head_dim,))?);
        }
        Ok(())
    }

    /// Select the group indices (sorted ascending) the query attends to: the top-k accumulated
    /// entries by index score for the query `(x, qr)` at token position `pos`. Empty when no
    /// entry yet.
    /// The Indexer's learned spaces for the query at `pos`: the roped per-head
    /// query `[n_heads, head_dim]`, the per-head gate weights `[n_heads]`
    /// (scale folded in), and the accumulated entry keys `[n, head_dim]` —
    /// exactly what the two-stage BDP-recall→precision selection consumes and
    /// what the per-layer recall validation sweeps. `None` until an entry
    /// exists.
    pub fn capture_space(
        &self,
        x: &Tensor,
        qr: &Tensor,
        rope: &RotaryCache,
        pos: usize,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        let n = self.entries.len();
        if n == 0 {
            return Ok(None);
        }
        let (q, weights) = self.idx.query_space(x, qr, rope, pos)?;
        let kv = Tensor::stack(&self.entries.iter().collect::<Vec<_>>(), 0)?; // [n, hd]
        Ok(Some((q, weights, kv)))
    }

    pub fn select(
        &self,
        x: &Tensor,
        qr: &Tensor,
        rope: &RotaryCache,
        pos: usize,
    ) -> Result<Vec<usize>> {
        let Some((q, weights, kv)) = self.capture_space(x, qr, rope, pos)? else {
            return Ok(Vec::new());
        };
        let n = self.entries.len();
        let score = q.matmul(&kv.t()?.contiguous()?)?; // [h, n]
        let index_score = score
            .relu()?
            .broadcast_mul(&weights.reshape(((), 1))?)?
            .sum(0)?; // [n]
        let scores = index_score.to_vec1::<f32>()?;

        // Top-k by score, then sorted ascending (prefill gathers compressed keys in group order).
        let top_k = self.idx.top_k.min(n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let mut selected: Vec<usize> = order.into_iter().take(top_k).collect();
        selected.sort_unstable();
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, IndexOp};

    /// `query_band` (Artifact D wide-Q source) returns the roped per-head
    /// Indexer query flattened in `(head, dim)` order — exactly the band
    /// `WideQSig::from_band` signs and packs. Locks length + ordering against
    /// `query_space`'s q, and demonstrates the `from_band` sign-pack rule over
    /// that ordering.
    #[test]
    fn query_band_matches_query_space_pack_order() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, qlr, h, hd, rd, ratio) = (8usize, 6usize, 3usize, 8usize, 4usize, 4usize);
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let comp = Compressor::new(
            Tensor::randn(0f32, 1.0, (2 * hd, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (2 * hd, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, 2 * hd), &dev)?,
            Tensor::ones(hd, DType::F32, &dev)?,
            ratio,
            hd,
            rd,
            1e-6,
        );
        let wq_b = Tensor::randn(0f32, 1.0, (h * hd, qlr), &dev)?;
        let wproj = Tensor::randn(0f32, 1.0, (h, dim), &dev)?;
        let idx = Indexer::new(QLinear::from_weight(wq_b), wproj, comp, h, hd, rd, 4);

        let pos = 5usize;
        let x = Tensor::randn(0f32, 1.0, (dim,), &dev)?;
        let qr = Tensor::randn(0f32, 1.0, (qlr,), &dev)?;
        let band = idx.query_band(&x, &qr, &rope, pos)?;
        assert_eq!(band.len(), h * hd, "band is n_heads × head_dim");

        // The band IS query_space's q flattened in (head, dim) row order.
        let (q_space, _w) = idx.query_space(&x, &qr, &rope, pos)?;
        let q_flat = q_space.flatten_all()?.to_vec1::<f32>()?;
        for (i, (&a, &b)) in band.iter().zip(&q_flat).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "band[{i}] != query_space q[{i}]");
        }

        // `from_band`'s pack rule (bit i of head hh set iff band[hh*hd+i] >= 0)
        // over that ordering is well-defined and deterministic.
        let wph = hd.div_ceil(64);
        let mut words = vec![0u64; h * wph];
        for hh in 0..h {
            for i in 0..hd {
                if band[hh * hd + i] >= 0.0 {
                    words[hh * wph + i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        // Head 0's low word reflects head 0's dims only (ordering isolation).
        let expect_h0: u64 = (0..hd)
            .filter(|&i| band[i] >= 0.0)
            .fold(0u64, |w, i| w | (1u64 << (i % 64)));
        assert_eq!(words[0], expect_h0);
        Ok(())
    }

    /// Index scores equal a scalar transcription `Σ_h relu(q_h·k_g)·w_h`, with the gate
    /// `w = (x·Wᵀ)·(hd^-0.5·h^-0.5)`. The reference rebuilds `q` and the compressed cache
    /// exactly as the module does (same RoPE), then dots them by hand.
    #[test]
    fn scores_match_scalar_reference() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, qlr, h, hd, rd, ratio) = (8usize, 6usize, 3usize, 8usize, 4usize, 4usize);
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let s = 12;
        let x = Tensor::randn(0f32, 1.0, (1, s, dim), &dev)?;
        let qr = Tensor::randn(0f32, 1.0, (1, s, qlr), &dev)?;
        let comp = Compressor::new(
            Tensor::randn(0f32, 1.0, (2 * hd, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (2 * hd, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, 2 * hd), &dev)?,
            Tensor::ones(hd, DType::F32, &dev)?,
            ratio,
            hd,
            rd,
            1e-6,
        );
        let wq_b = Tensor::randn(0f32, 1.0, (h * hd, qlr), &dev)?;
        let wproj = Tensor::randn(0f32, 1.0, (h, dim), &dev)?;
        let idx = Indexer::new(
            QLinear::from_weight(wq_b.clone()),
            wproj.clone(),
            comp.clone(),
            h,
            hd,
            rd,
            4,
        );
        let got = idx.scores(&x, &qr, &rope)?.unwrap(); // [1,s,G]

        // Rebuild q exactly as the module does (RoPE by token position, broadcast over heads via
        // the [b,h,s,hd] transpose), and kv from the compressor, then dot.
        let kv = comp.forward(&x, &rope)?.unwrap(); // [1,G,hd]
        let g = kv.dim(1)?;
        let q = qr
            .broadcast_matmul(&wq_b.t()?)?
            .reshape((1, s, h, hd))?
            .transpose(1, 2)?
            .contiguous()?; // [1,h,s,hd]
        let q_nope = q.narrow(D::Minus1, 0, hd - rd)?;
        let q_rope = rope.apply(&q.narrow(D::Minus1, hd - rd, rd)?, 0, false)?;
        let q = Tensor::cat(&[&q_nope, &q_rope], D::Minus1)?
            .transpose(1, 2)?
            .contiguous()?; // [1,s,h,hd]
        let qv = q.i(0)?.to_vec3::<f32>()?; // [s,h,hd]
        let kvv = kv.i(0)?.to_vec2::<f32>()?; // [G,hd]
        let scale = (hd as f64).powf(-0.5) * (h as f64).powf(-0.5);
        let wv = (x.i(0)?.matmul(&wproj.t()?)? * scale)?.to_vec2::<f32>()?; // [s,h]
        let gotv = got.i(0)?.to_vec2::<f32>()?;
        for t in 0..s {
            for gg in 0..g {
                let mut acc = 0f32;
                for head in 0..h {
                    let dot: f32 = (0..hd).map(|c| qv[t][head][c] * kvv[gg][c]).sum();
                    acc += dot.max(0.0) * wv[t][head];
                }
                assert!(
                    (gotv[t][gg] - acc).abs() < 1e-3,
                    "t{t} g{gg}: {} vs {acc}",
                    gotv[t][gg]
                );
            }
        }
        Ok(())
    }
}
