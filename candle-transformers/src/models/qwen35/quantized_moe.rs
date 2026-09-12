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
) -> Result<SharedExpert<'w>> {
    // One width for both: the shared expert's result is summed into the MoE
    // combine, which runs at the experts' working dtype, so there is no
    // narrower store to ask for here.
    // **Every step of the shared half, probed asynchronously.**
    //
    // `moe.shared_gated` checks the product `y * sigmoid(gate)`, which cannot
    // say which operand went bad — and `sigmoid` never manufactures a NaN from
    // a finite input, so the answer is one of the two. These are `assert`, not
    // `checkpoint`: one kernel each, no fence, no readback, so the drain can
    // rank them by when they actually went bad without draining the pipeline
    // the fault needs to reproduce.
    let y = shared.forward_dynamic(acts, out_dtype, out_dtype)?;
    y.assert("moe.shared.mlp_out");
    // The gate weight is padded to a full KO tile (see `SHARED_GATE_TILE`), so
    // the projection yields a tile's worth of outputs and only the first is the
    // gate — the rest are the zero rows. Narrowing unconditionally is also
    // correct for an unpadded weight, which keeps this free of any dependence
    // on the numeric path the weights were built for.
    let gate = shared_gate.forward_dynamic(acts.as_dynamic(), out_dtype)?;
    // Before AND after the narrow: if the raw tile is clean and row 0 is not,
    // the pad is contaminating the row the gate is read from — which is the one
    // way this differs from an ordinary projection.
    gate.assert("moe.shared.gate_tile");
    let last = gate.rank() - 1;
    let gate = gate.narrow(last, 0, 1)?;
    gate.assert("moe.shared.gate_row0");
    let gate = candle_nn::ops::sigmoid(&gate)?;
    gate.assert("moe.shared.gate_sigmoid");
    // The gate is one scalar per token and `y` is `[.., hidden]`; both carry
    // the same leading dims, so the broadcast is over the last one.
    //
    // **The product's operands travel with it.** `moe.shared_gated` is the site
    // an armed capture keeps landing on, and it checks `y * sigmoid(gate)` —
    // which, as the note above says, cannot say which operand went bad. The
    // capture then reported `Context dumped: []` and named the product. The two
    // asserts above *are* the operands, but they are asynchronous, so the
    // panic's "every checkpoint upstream passed" does not cover them and the
    // question stayed open across three runs.
    //
    // Returned rather than checkpointed here because the site name carries the
    // layer index, which only the caller knows.
    let gated = y.broadcast_mul(&gate)?;
    Ok(SharedExpert { gated, y, gate })
}

/// The shared expert's product and the two operands it was made from.
///
/// `sigmoid` never manufactures a NaN from a finite input, so when the product
/// is non-finite the answer is one of these two — and an armed capture that
/// carries them says which, in the run that produced it, rather than in the run
/// after next.
pub struct SharedExpert<'w> {
    pub gated: LiveTensor<'w>,
    /// `shared(x)` — the expert's own output, before gating.
    pub y: LiveTensor<'w>,
    /// `sigmoid(w_gate · x)`, one scalar per token.
    pub gate: LiveTensor<'w>,
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
        let shared = shared_expert_contribution(&self.shared, &self.shared_gate, &acts, out_dtype)?;
        let gated = shared.gated;
        let routed = self.routed.forward_dynamic(acts, out_dtype, wave)?;
        // The three values the layer's output is made of, checked where they
        // are still separable.
        //
        // The routed half is instrumented all the way down; the SHARED half was
        // not instrumented at all, and it is the other half of the sum. Its
        // `sigmoid` and its per-token broadcast are the only broadcast ops in
        // the FFN — and a broadcast add is what the fault's kernel breadcrumb
        // named. Checking `gated` and `routed` apart, then their sum, is what
        // separates "one of the addends was already bad" from "the combine
        // produced it".
        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::checkpoint;
            use candle::tensor_assert::site;
            if let candle::Device::Cuda(d) = routed.device() {
                let li = self.routed.moe_layer_idx;
                // Its two operands, so the capture says WHICH went bad rather
                // than only that the product did — see `SharedExpert`.
                checkpoint(
                    site("moe.shared_gated.L", li),
                    &gated,
                    &[("shared_y", &shared.y), ("gate_sigmoid", &shared.gate)],
                    d,
                )?;
                checkpoint(site("moe.routed_sum_in.L", li), &routed, &[], d)?;
                let sum = (&routed + &gated)?;
                checkpoint(
                    site("moe.combined.L", li),
                    &sum,
                    &[("routed", &routed), ("gated", &gated)],
                    d,
                )?;
                return Ok(sum);
            }
        }
        &routed + &gated
    }
}

// The 35B-pinned parity test for the shared expert lives with the model it
// pins: `models/quantized_qwen35_moe.rs`.
