//! The production MoE layer: the existing routed block, plus the shared
//! expert this family adds.
//!
//! Qwen3.5's MoE is Qwen3-MoE's with one addition, so this reuses rather than
//! restates. The routed half is [`SparseMoeBlock`] verbatim — same
//! [`ExpertCache`], same GPU-native/host dispatch fork, same counting-sort —
//! and needs nothing from this family: 256 experts is inside
//! `moe_bucketize`'s `MAX_EXPERTS`, and the `> 128` check in `gpu_dispatch`
//! degrades to the host path rather than failing, which is the documented
//! behaviour for an oversized id space.
//!
//! What is new is the **shared expert**: an ordinary SwiGLU that every token
//! goes through, scaled by a per-token scalar `sigmoid(w_gate · x)`, summed
//! with the routed output. Qwen3-MoE has no equivalent — there is no `shexp`
//! tensor anywhere in that model — so it lives here.
//!
//! Ordering note: the routed block *consumes* its activation (the gather is
//! the activation's last reader, so it is moved, not borrowed). The shared
//! expert and its gate therefore run **first**, off a borrow, and the routed
//! call takes ownership last.

use candle::quantized::cuda::DynamicActs;
use candle::{DType, LiveTensor, Result};
use candle_nn::kv_cache::WaveGeneration;

use crate::models::quantized_matmul::QMatMul;
use crate::models::quantized_mlp::QuantizedMlp;
use crate::models::quantized_qwen3_moe::SparseMoeBlock;

/// One Qwen3.5 MoE layer.
///
/// `pub(crate)` because it holds a `pub(crate)` [`SparseMoeBlock`]: the
/// routed half is the engine's, not this model's, and is not part of any
/// public surface.
pub(crate) struct Qwen35MoeBlock {
    /// Router + top-k + expert cache — the shared implementation.
    pub routed: SparseMoeBlock,
    /// The always-active shared expert.
    pub shared: QuantizedMlp,
    /// `[1, hidden]` — projects each token to the shared expert's scalar
    /// gate, pre-sigmoid.
    pub shared_gate: QMatMul,
}

/// The shared expert's contribution: `sigmoid(w_gate · x) · shared(x)`.
///
/// A free function so it can be exercised against the F32 reference without
/// standing up an [`ExpertCache`] — the routed half is already covered by
/// Qwen3-MoE's own gates, and this is the part that is new.
pub fn shared_expert_contribution<'w>(
    shared: &QuantizedMlp,
    shared_gate: &QMatMul,
    acts: &DynamicActs<'w>,
    out_dtype: DType,
) -> Result<LiveTensor<'w>> {
    // One width for both: the shared expert's result is summed into the MoE
    // combine, which runs at the experts' working dtype, so there is no
    // narrower store to ask for here.
    let y = shared.forward_dynamic(acts, out_dtype, out_dtype)?;
    // The gate weight is padded to a full KO tile (see `SHARED_GATE_TILE`), so
    // the projection yields a tile's worth of outputs and only the first is the
    // gate — the rest are the zero rows. Narrowing unconditionally is also
    // correct for an unpadded weight, which keeps this free of any dependence
    // on the numeric path the weights were built for.
    let gate = shared_gate.forward_dynamic(acts.as_dynamic(), out_dtype)?;
    let last = gate.rank() - 1;
    let gate = gate.narrow(last, 0, 1)?;
    let gate = candle_nn::ops::sigmoid(&gate)?;
    // The gate is one scalar per token and `y` is `[.., hidden]`; both carry
    // the same leading dims, so the broadcast is over the last one.
    y.broadcast_mul(&gate)
}

impl Qwen35MoeBlock {
    /// `routed(x) + sigmoid(w_gate · x) · shared(x)`.
    ///
    /// Matches `qwen35moe.cpp`'s combine and the F32 reference in
    /// [`super::moe`], which is validated against llama.cpp.
    pub fn forward_dynamic<'w>(
        &self,
        acts: DynamicActs<'w>,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        // Shared expert first — see the module note on ownership.
        let gated = shared_expert_contribution(&self.shared, &self.shared_gate, &acts, out_dtype)?;
        let routed = self.routed.forward_dynamic(acts, out_dtype, wave)?;
        &routed + &gated
    }
}

// The 35B-pinned parity test for the shared expert lives with the model it
// pins: `models/quantized_qwen35_moe.rs`.
