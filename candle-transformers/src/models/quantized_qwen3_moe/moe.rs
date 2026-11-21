//! The routed half of a sparse MoE layer: the router, the top-k route and the
//! device-side expert forward.
//!
//! `SparseMoeBlock` runs the router, the device's route and the expert cache's
//! forward, and nothing on that path reads the routing back to the host. Qwen3.5
//! routes through the same block and adds a shared expert beside it. The experts
//! run only as int8 grouped GEMMs, so a float activation is quantised per row and
//! per 128-wide tile first. Router logits are checked before the route clamps
//! them, so a NaN is still visible when it is found. `capture_routing` is the
//! test-only observer that does read the route back.

use super::*;

// Sparse MoE Block

/// The routed half of a sparse MoE layer: router → top-k → expert cache.
///
/// `pub(crate)` because Qwen3.5 routes over the same machinery — same
/// `ExpertCache`, same device-side expert forward
/// ([`ExpertCache::forward_routed`]), same bucketize — and differs only by
/// adding an always-active shared expert alongside it (`qwen35::quantized_moe`). Exposing the existing block is
/// deliberately all that is done for that: no fields move, no trait is
/// reshaped, and this model is otherwise untouched.
pub(crate) struct SparseMoeBlock {
    pub(crate) gate: QMatMul,
    pub(crate) cache: Arc<ExpertCache>,
    pub(crate) moe_layer_idx: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) norm_topk_prob: bool,
}

impl SparseMoeBlock {
    /// The routed half of the layer, on the device: router → `moe_route` →
    /// the expert cache's device forward ([`ExpertCache::forward_routed`]).
    ///
    /// `acts` is the ln2 output as a `DynamicActs`. The experts run only as the
    /// int8 q8a128 × KO grouped GEMM, so a `Float` activation — what a DeltaNet
    /// layer carrying a LoRA hands over — is quantized here, per row and per
    /// 128-tile, exactly as it would be if the gather came first. Nothing on
    /// this path reads routing back to the host.
    #[cfg(feature = "cuda")]
    pub(crate) fn forward_dynamic<'w>(
        &self,
        acts: DynamicActs<'w>,
        out_dtype: DType,
        decode: &DecodeRows,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        let num_tokens = match &acts {
            DynamicActs::Float(t) => {
                let (b, s, _) = t.dims3()?;
                b * s
            }
            DynamicActs::Int8(op) => op.rows,
        };
        let router_logits = self
            .gate
            .forward_dynamic(acts.as_dynamic(), out_dtype)?
            .reshape((num_tokens, ()))?;
        self.forward_with_logits(router_logits, acts, out_dtype, decode, wave, &[])
    }

    /// [`Self::forward_dynamic`] from router logits the caller already projected —
    /// `[num_tokens, n_experts]`, rows at any pitch with unit-stride columns, so a router
    /// that shared its launch with other projections of the same activation hands over
    /// its columns of the wider row in place. `look_ahead`, when given, is the next
    /// rows' routers from the same launch ([`ExpertCache::forward_routed`]).
    #[cfg(feature = "cuda")]
    pub(crate) fn forward_with_logits<'w>(
        &self,
        router_logits: LiveTensor<'w>,
        acts: DynamicActs<'w>,
        out_dtype: DType,
        decode: &DecodeRows,
        wave: Option<&'w WaveGeneration>,
        look_ahead: &[LookAhead<'_>],
    ) -> Result<LiveTensor<'w>> {
        let (b_size, seq_len, hidden_dim) = match &acts {
            DynamicActs::Float(t) => t.dims3()?,
            DynamicActs::Int8(op) => match op.lead.as_slice() {
                &[b, s] => (b, s, op.cols),
                other => candle::bail!(
                    "SparseMoeBlock::forward_with_logits: expected [b, seq] lead, got {other:?}"
                ),
            },
        };
        let k = self.num_experts_per_tok;
        let t = profile_now();
        // The router's own output is the first thing that can be bad here, and
        // `moe_route` clamps a sentinel out of `-inf`/NaN logits — so a NaN in
        // the logits is fixed up into a plausible route rather than propagating
        // visibly. Assert BEFORE the clamp, or the evidence is gone.
        #[cfg(feature = "tensor-assert")]
        router_logits.assert(candle::tensor_assert::site(
            "moe.router_logits.L",
            self.moe_layer_idx,
        ));
        let (top_k_weights, top_k_indices) = moe_route(&router_logits, k, self.norm_topk_prob)?;
        #[cfg(feature = "tensor-assert")]
        top_k_weights.assert(candle::tensor_assert::site(
            "moe.route_weights.L",
            self.moe_layer_idx,
        ));
        if routing_capture::is_enabled() {
            self.capture_routing(&top_k_weights, &top_k_indices)?;
        }
        pipeline_record("moe:route", t);

        let op = match acts {
            DynamicActs::Int8(op) => op,
            DynamicActs::Float(xs) => {
                let Device::Cuda(cuda_dev) = xs.device().clone() else {
                    candle::bail!("SparseMoeBlock::forward_dynamic: expected a CUDA device")
                };
                match to_dynamic(&xs, Int8Mode::Precision, &cuda_dev, SumScale::Raw)? {
                    DynamicActs::Int8(op) => op,
                    DynamicActs::Float(_) => candle::bail!(
                        "SparseMoeBlock::forward_dynamic: an int8 quantize returned float"
                    ),
                }
            }
        };
        let ys = self.cache.forward_routed(
            op,
            &top_k_weights,
            &top_k_indices,
            self.moe_layer_idx,
            decode,
            out_dtype,
            wave,
            look_ahead,
        )?;
        debug_assert_eq!(ys.dim(1)?, hidden_dim);
        ys.reshape((b_size, seq_len, hidden_dim))
    }

    /// Record this layer's routing for the offline predictor evaluation — a
    /// test-only observer. It reads the route back to the host, which the
    /// expert forward itself never does; the forward is identical with it on
    /// or off.
    #[cfg(feature = "cuda")]
    pub(super) fn capture_routing(
        &self,
        weights: &LiveTensor<'_>,
        indices: &LiveTensor<'_>,
    ) -> Result<()> {
        let idx: Vec<Vec<u32>> = indices.to_vec2()?;
        let w: Vec<Vec<f32>> = weights.to_vec2()?;
        let mut mass: BTreeMap<usize, f32> = BTreeMap::new();
        for (row_i, row_w) in idx.iter().zip(w.iter()) {
            for (&e, &wv) in row_i.iter().zip(row_w.iter()) {
                *mass.entry(e as usize).or_insert(0.0) += wv;
            }
        }
        let expert_ids: Vec<usize> = mass.keys().copied().collect();
        let mass: Vec<f32> = mass.values().copied().collect();
        routing_capture::record(self.moe_layer_idx, &expert_ids, &mass);
        Ok(())
    }
}

// FeedForward enum (dense MLP or MoE)

// One per layer for the model's lifetime, so the variant spread costs a few
// hundred bytes per layer at most — and this is matched on in the forward path,
// where a `Box` would add a pointer chase to every FFN.
#[allow(clippy::large_enum_variant)]
pub(super) enum FeedForward {
    Mlp(QuantizedMlp),
    MoE(SparseMoeBlock),
}

/// A layer's FFN before the expert cache exists.
///
/// The load order is dense-weights-then-span-then-experts (see
/// `docs/archived/elastic_vram_partition.md` §4): the reservation is sized from a live
/// measurement taken once every dense tensor is resident, and the expert cache
/// is filled into the span that measurement produced. A MoE layer's
/// `SparseMoeBlock` holds an `Arc<ExpertCache>`, so it cannot be built during
/// the dense loop — this is what the loop produces instead, and the cache is
/// grafted on afterwards.
///
/// Only the expert half is deferred. The router gate is a dense tensor and is
/// loaded in the loop with the rest of them.
// Load-time only, one per layer, consumed when the expert cache is grafted on.
#[allow(clippy::large_enum_variant)]
pub(super) enum PendingFfn {
    Mlp(QuantizedMlp),
    MoE { gate: QMatMul, moe_layer_idx: usize },
}
