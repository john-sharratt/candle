//! The production MoE layer: the existing routed block, plus the shared
//! expert this family adds.
//!
//! Qwen3.5's MoE is Qwen3-MoE's with one addition, so this reuses rather than
//! restates. The routed half is [`SparseMoeBlock`] verbatim — same
//! [`ExpertCache`], same device-side expert forward, same bucketize — and
//! needs nothing from this family: 256 experts is inside `moe_bucketize`'s
//! `MAX_EXPERTS` (512), which the live table checks at load.
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
use candle::quantized::decode_rows::DecodeRows;
use candle::{DType, LiveTensor, Result, Tensor};
use candle_nn::kv_cache::WaveGeneration;
use candle_nn::ops::sigmoid;

use crate::models::quantized_matmul::QMatMul;
use crate::models::quantized_mlp::QuantizedMlp;
use crate::models::quantized_qwen3_moe::SparseMoeBlock;

use super::shared_residual::add_moe_residual;

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
    shared_expert_parts(shared, shared_gate, acts, out_dtype)?.gated()
}

/// The shared expert's two halves before they are combined: its output and its
/// raw per-row gate.
pub struct SharedExpertParts<'w> {
    /// `shared(x)`, at the activation's leading dims.
    pub y: LiveTensor<'w>,
    /// `w_gate · x`, pre-sigmoid, `[rows, 1]` — the first column of the padded
    /// gate projection, a strided view that is never compacted.
    pub gate: LiveTensor<'w>,
}

impl<'w> SharedExpertParts<'w> {
    /// `sigmoid(gate) · y`, at `y`'s shape. The gate is one scalar per row,
    /// `[rows, 1]`; `y` is viewed as `[rows, hidden]` for the broadcast — a free
    /// reshape of the contiguous result, where reshaping the strided gate column
    /// would copy it — so every leading shape of `y` lines up row for row.
    pub fn gated(&self) -> Result<LiveTensor<'w>> {
        let dims = self.y.dims().to_vec();
        let hidden = dims[dims.len() - 1];
        let rows = self.y.elem_count() / hidden;
        self.y
            .reshape((rows, hidden))?
            .broadcast_mul(&sigmoid(&self.gate)?)?
            .reshape(dims)
    }
}

/// [`shared_expert_contribution`] stopped short of the gate's sigmoid and the
/// multiply, for a consumer that applies them where it already reads the result
/// (Qwen3.8-Flash-Next's hyper-connection combine).
pub fn shared_expert_parts<'w>(
    shared: &QuantizedMlp,
    shared_gate: &QMatMul,
    acts: &DynamicActs<'w>,
    out_dtype: DType,
) -> Result<SharedExpertParts<'w>> {
    // One width for both: the shared expert's result is summed into the MoE
    // combine, which runs at the experts' working dtype, so there is no
    // narrower store to ask for here.
    let y = shared.forward_dynamic(acts, out_dtype, out_dtype)?;
    // The gate weight is padded to a full KO tile (see `SHARED_GATE_TILE`), so
    // the projection yields a tile's worth of outputs and only the first is the
    // gate — the rest are the zero rows. Narrowing unconditionally is also
    // correct for an unpadded weight, which keeps this free of any dependence
    // on the numeric path the weights were built for. Viewed as `[rows, width]`
    // first — a free reshape of the contiguous projection — so the column is a
    // 2-D strided view a kernel can read through its row stride.
    let gate = shared_gate.forward_dynamic(acts.as_dynamic(), out_dtype)?;
    let width = gate.dim(gate.rank() - 1)?;
    let gate = gate
        .reshape((gate.elem_count() / width, width))?
        .narrow(1, 0, 1)?;
    Ok(SharedExpertParts { y, gate })
}

/// One MoE layer's output in the three parts it is summed from:
/// `routed + shared · sigmoid(gate)`.
pub struct MoeParts<'w> {
    /// The routed experts' weighted sum.
    pub routed: LiveTensor<'w>,
    /// The shared expert, ungated.
    pub shared: SharedExpertParts<'w>,
}

impl Qwen35MoeBlock {
    /// The layer's three parts, uncombined — see [`MoeParts`]. For a consumer
    /// that folds the combine into a pass it already makes.
    pub fn forward_parts<'w>(
        &self,
        acts: DynamicActs<'w>,
        out_dtype: DType,
        decode: &DecodeRows,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<MoeParts<'w>> {
        // **The three projections of the layer input as one launch.** The router, the
        // shared expert's fused gate_up and its gate all read the same q8a128 operand, and
        // at decode width each was a narrow split-K launch of its own, latency-bound at a
        // few microseconds apiece. Stacked, they are one launch writing
        // `[router | gate_up | gate]` side by side, and every column is the one its own
        // launch computed, bit for bit (`QMatMul::forward_stacked`). The parts below read
        // their columns of that row in place.
        if let (DynamicActs::Int8(op), Some(gate_up)) = (&acts, self.shared.fused_gate_up()) {
            let n_experts = self.routed.gate.weight_dims()[0];
            let n_gate_up = gate_up.weight_dims()[0];
            let half = n_gate_up / 2;
            let stacked = QMatMul::forward_stacked(
                op,
                &[&self.routed.gate, gate_up, &self.shared_gate],
                out_dtype,
            )?;
            let width = stacked.dim(stacked.rank() - 1)?;
            let rows = stacked.elem_count() / width;
            let flat = stacked.reshape((rows, width))?;
            let lead: Vec<usize> = op.lead.clone();
            let gate_half = flat.narrow(1, n_experts, half)?;
            let up_half = flat.narrow(1, n_experts + half, half)?;
            let y = self
                .shared
                .forward_from_gate_up(&gate_half, &up_half, out_dtype)?;
            let mut y_dims = lead;
            y_dims.push(y.dim(y.rank() - 1)?);
            let shared = SharedExpertParts {
                y: y.reshape(y_dims)?,
                // The gate weight is padded to a full KO tile; its first column is the gate.
                gate: flat.narrow(1, n_experts + n_gate_up, 1)?,
            };
            let logits = flat.narrow(1, 0, n_experts)?;
            let routed = self
                .routed
                .forward_with_logits(logits, acts, out_dtype, decode, wave)?;
            return Ok(MoeParts { routed, shared });
        }
        // Shared expert first — see the module note on ownership.
        let shared = shared_expert_parts(&self.shared, &self.shared_gate, &acts, out_dtype)?;
        let routed = self.routed.forward_dynamic(acts, out_dtype, decode, wave)?;
        Ok(MoeParts { routed, shared })
    }

    /// `x += routed(a) + sigmoid(w_gate · a) · shared(a)`, where `a` is the
    /// layer's normed activation and `x` the residual stream it came from.
    ///
    /// Matches `qwen35moe.cpp`'s combine and the F32 reference in
    /// [`super::moe`], which is validated against llama.cpp. The combine and the
    /// residual add are one launch ([`add_moe_residual`]): the parts are at
    /// `work_dtype` and narrow to the stream's type after their sum, exactly
    /// where the separate launches did.
    pub fn forward_residual<'w>(
        &self,
        x: &mut Tensor,
        acts: DynamicActs<'w>,
        work_dtype: DType,
        decode: &DecodeRows,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<()> {
        let MoeParts { routed, shared } = self.forward_parts(acts, work_dtype, decode, wave)?;
        // The values the layer's output is made of, checked where they are
        // still separable — the routed sum, the shared expert and its gate apart,
        // then the residual they land in. Bad on an input names the addend; bad
        // only on the residual names the combine.
        #[cfg(feature = "tensor-assert")]
        let cuda = {
            use crate::models::nan_capture::checkpoint;
            use candle::tensor_assert::site;
            use candle::Device;
            match routed.device() {
                Device::Cuda(d) => {
                    let li = self.routed.moe_layer_idx;
                    checkpoint(site("moe.shared_out.L", li), &shared.y, &[], d)?;
                    checkpoint(site("moe.shared_gate.L", li), &shared.gate, &[], d)?;
                    checkpoint(site("moe.routed_sum_in.L", li), &routed, &[], d)?;
                    checkpoint(site("moe.residual_in.L", li), x, &[], d)?;
                    Some(d.clone())
                }
                _ => None,
            }
        };
        add_moe_residual(x, &routed, &shared)?;
        #[cfg(feature = "tensor-assert")]
        if let Some(d) = cuda {
            use crate::models::nan_capture::checkpoint;
            use candle::tensor_assert::site;
            checkpoint(
                site("moe.combined.L", self.routed.moe_layer_idx),
                x,
                &[
                    ("routed", &routed),
                    ("shared", &shared.y),
                    ("gate", &shared.gate),
                ],
                &d,
            )?;
        }
        Ok(())
    }
}

// The 35B-pinned parity test for the shared expert lives with the model it
// pins: `models/quantized_qwen35_moe.rs`.
