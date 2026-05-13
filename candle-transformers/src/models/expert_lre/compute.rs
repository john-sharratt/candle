//! SwiGLU expert compute — grouped and per-expert paths.
//!
//! On CUDA, [`compute_experts_grouped`] processes all routed experts in
//! 3 grouped `QMatMul` kernel launches (gate, up, down) with fused
//! gather/scatter, replacing ~60 individual launches per layer.
//!
//! On non-CUDA targets, [`compute_expert_contribution_gpu_weights`]
//! falls back to per-expert sequential computation.

// Re-export QMatMul so other submodules can reference it through `compute::QMatMul`.
pub(crate) use crate::models::quantized_matmul::QMatMul;

use super::types::ExpertSlot;
#[cfg(feature = "cuda")]
use crate::models::profile::{profile_now, ProfileAccumulator};
use candle::{Result, Tensor};
#[cfg(not(feature = "cuda"))]
use candle_nn::Module;

/// Compute one expert's SwiGLU contribution and accumulate into `ys`.
///
/// Uses GPU-resident routing weights — gathers them directly from a flat
/// GPU tensor, avoiding a device→host→device round-trip.
///
/// * `weights_flat` — flattened `[num_tokens * k]` routing weights (GPU, F32)
/// * `flat_w_indices` — CPU `&[u32]` indices into `weights_flat` for this
///   expert's tokens (computed as `token_idx * k + slot_within_k`)
#[cfg(not(feature = "cuda"))]
pub fn compute_expert_contribution_gpu_weights(
    xs: &Tensor,
    ys: &mut Tensor,
    slot: &ExpertSlot,
    top_x_slice: &[u32],
    weights_flat: &Tensor,
    flat_w_indices: &[u32],
) -> Result<()> {
    if top_x_slice.is_empty() {
        return Ok(());
    }
    let dev = xs.device();
    let top_x_t = Tensor::new(top_x_slice, dev)?;
    // Gather weights from GPU — no host→device weight upload
    let w_idx_t = Tensor::new(flat_w_indices, dev)?;
    let sel_w = weights_flat.index_select(&w_idx_t, 0)?.reshape(((), 1))?;

    let cur = xs.index_select(&top_x_t, 0)?;

    // SwiGLU: silu(gate @ cur) * (up @ cur) → down  [fused kernel]
    let gate_out = slot.gate_proj.forward(&cur)?;
    let up_out = slot.up_proj.forward(&cur)?;
    let intermediate = candle_nn::ops::silu_mul(&gate_out, &up_out)?;
    let out = slot.down_proj.forward(&intermediate)?;

    let sel_w = sel_w.to_dtype(out.dtype())?;
    let weighted = out.broadcast_mul(&sel_w)?;
    *ys = ys.index_add(&top_x_t, &weighted, 0)?;
    Ok(())
}

// =============================================================================
// GROUPED EXPERT COMPUTE (single kernel launch for all experts)
// =============================================================================

/// Extract the CUDA data pointer from a QMatMul wrapper.
///
/// Returns `(device_ptr, shape, ggml_dtype)` for the underlying quantized tensor.
/// Returns None if the tensor is not a CUDA QTensor.
#[cfg(feature = "cuda")]
fn extract_weight_info(
    qmm: &QMatMul,
) -> Result<(u64, candle::Shape, candle::quantized::GgmlDType)> {
    match qmm.inner() {
        candle::quantized::QMatMul::QTensor(qt) => {
            let ptr = qt
                .cuda_data_ptr()
                .ok_or_else(|| candle::Error::Msg("expected CUDA QTensor".into()))?;
            Ok((ptr, qt.shape().clone(), qt.dtype()))
        }
        _ => Err(candle::Error::Msg(
            "grouped expert compute requires QTensor variant".into(),
        )),
    }
}

/// Grouped expert SwiGLU: processes all experts in 3 grouped matmul launches.
///
/// Instead of looping over experts and launching separate kernels for each,
/// this function:
/// 1. Gathers activations for all experts into a stacked tensor
/// 2. Launches 3 grouped matmuls (gate, up, down) — each is a single kernel
/// 3. Applies fused SiLU-Mul between gate and up outputs
/// 4. Scatters weighted results back to the output tensor
///
/// This reduces ~20 experts × 3 matmuls = ~60 kernel launches down to 3.
///
/// # Arguments
/// * `xs` — input activations `[num_tokens, hidden_dim]`
/// * `ys` — output tensor to accumulate into (modified in-place)
/// * `experts` — slice of `(slot, token_ids, weight_ids)` for each expert
/// * `weights_flat` — GPU-resident routing weights `[num_tokens * k]`
///
/// # Expert layout
/// Each expert has 1–N tokens from potentially different sessions.
/// The `token_ids` index into `xs` and `ys`; `weight_ids` index into `weights_flat`.
#[cfg(feature = "cuda")]
pub fn compute_experts_grouped(
    xs: &Tensor,
    ys: &mut Tensor,
    experts: &[(&ExpertSlot, &[u32], &[u32])],
    weights_flat: &Tensor,
    profile: &mut ProfileAccumulator,
) -> Result<()> {
    if experts.is_empty() {
        return Ok(());
    }

    let dev = xs.device();
    let num_experts = experts.len();

    // ── Get CUDA device (needed for fused kernels) ──
    let cuda_dev = match dev {
        candle::Device::Cuda(d) => d,
        _ => {
            return Err(candle::Error::Msg(
                "grouped expert compute requires CUDA device".into(),
            ))
        }
    };

    // ── Build per-expert metadata ──
    // Collect all token_ids and weight_ids, build expert_offsets prefix sum
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut all_weight_ids: Vec<u32> = Vec::new();
    let mut expert_offsets: Vec<i32> = Vec::with_capacity(num_experts + 1);
    expert_offsets.push(0);

    for &(_slot, toks, wids) in experts {
        all_token_ids.extend_from_slice(toks);
        all_weight_ids.extend_from_slice(wids);
        expert_offsets.push(all_token_ids.len() as i32);
    }

    let total_batch = all_token_ids.len();
    if total_batch == 0 {
        return Ok(());
    }

    // ── Upload token_ids once (shared by gather + scatter) ──
    let tok_ids_dev = cuda_dev.memcpy_stod(&all_token_ids)?;

    // ── Gather stacked activations (fused kernel) ──
    let t = profile_now();
    let stacked_xs =
        candle::quantized::cuda::fused_moe_gather(xs, &tok_ids_dev, total_batch, cuda_dev)?; // [total_batch, K]
    profile.record("gemm_gather", t);

    // ── Extract weight pointers for each projection ──
    let mut gate_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut up_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut down_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut gate_shape = None;
    let mut down_shape = None;
    let mut gate_dtype = None;
    let mut down_dtype = None;

    for &(slot, _, _) in experts {
        let (gp, gs, gd) = extract_weight_info(&slot.gate_proj)?;
        let (up, _, _) = extract_weight_info(&slot.up_proj)?;
        let (dp, ds, dd) = extract_weight_info(&slot.down_proj)?;
        gate_ptrs.push(gp);
        up_ptrs.push(up);
        down_ptrs.push(dp);
        if gate_shape.is_none() {
            gate_shape = Some(gs);
            down_shape = Some(ds);
            gate_dtype = Some(gd);
            down_dtype = Some(dd);
        }
    }

    let gate_shape = gate_shape.unwrap();
    let down_shape = down_shape.unwrap();
    let gate_dtype = gate_dtype.unwrap();
    let down_dtype = down_dtype.unwrap();

    // Gate/Up shape: [intermediate_dim, hidden_dim] → nrows=intermediate_dim, ncols=hidden_dim
    let (gate_nrows, gate_ncols) = gate_shape.dims2()?;
    // Down shape: [hidden_dim, intermediate_dim] → nrows=hidden_dim, ncols=intermediate_dim
    let (down_nrows, down_ncols) = down_shape.dims2()?;

    // ── Grouped gate matmul ──
    // stacked_xs [total_batch, K] × gate_proj [intermediate_dim, K]^T → [total_batch, intermediate_dim]
    let act_storage = stacked_xs.storage_and_layout().0;
    let act_cuda = match &*act_storage {
        candle::Storage::Cuda(s) => s,
        _ => {
            return Err(candle::Error::Msg(
                "expected CUDA storage for activations".into(),
            ))
        }
    };
    let act_layout = stacked_xs.layout().clone();

    let t = profile_now();
    let gate_out = candle::quantized::cuda::grouped_matmul_gemx(
        &gate_ptrs,
        gate_dtype,
        gate_nrows,
        gate_ncols,
        act_cuda,
        &act_layout,
        &expert_offsets,
        cuda_dev,
    )?;
    profile.record("gemm_gate", t);

    // ── Grouped up matmul ──
    let t = profile_now();
    let up_out = candle::quantized::cuda::grouped_matmul_gemx(
        &up_ptrs,
        gate_dtype, // up_proj has same dtype as gate_proj
        gate_nrows, // up has same shape as gate
        gate_ncols,
        act_cuda,
        &act_layout,
        &expert_offsets,
        cuda_dev,
    )?;
    profile.record("gemm_up", t);

    // ── Fused SiLU-Mul ──
    let t = profile_now();
    let intermediate = candle_nn::ops::silu_mul(&gate_out, &up_out)?;
    profile.record("gemm_silu_mul", t);

    // ── Grouped down matmul ──
    // intermediate [total_batch, intermediate_dim] × down_proj [hidden_dim, intermediate_dim]^T → [total_batch, hidden_dim]
    let inter_storage = intermediate.storage_and_layout().0;
    let inter_cuda = match &*inter_storage {
        candle::Storage::Cuda(s) => s,
        _ => unreachable!("intermediate should be CUDA"),
    };
    let inter_layout = intermediate.layout().clone();
    let t = profile_now();
    let down_out = candle::quantized::cuda::grouped_matmul_gemx(
        &down_ptrs,
        down_dtype,
        down_nrows,
        down_ncols,
        inter_cuda,
        &inter_layout,
        &expert_offsets,
        cuda_dev,
    )?;
    profile.record("gemm_down", t);

    // ── Deterministic scatter: token-major reorder + sequential k-expert reduce ──
    //
    // `down_out` is in expert-major order. Using atomicAdd is non-deterministic for
    // BF16 on SM < 900. Fix: sort assignments by (token_id, expert_pos) to group each
    // token's contributions consecutively, then use one block per output token with
    // sequential F32 accumulation — no atomicAdd.
    //
    // Variable-k: when called from the two-call pipeline (hits + misses), different
    // tokens may have different numbers of expert contributions per call. We use
    // per-token prefix-sum offsets instead of assuming uniform k.
    let num_tokens = xs.dim(0)?; // = b_size * seq_len (or b_size for decode)

    // Build permutation: sort expert-major indices by (token_id, original_pos).
    let mut perm: Vec<u32> = (0..total_batch as u32).collect();
    perm.sort_by_key(|&i| (all_token_ids[i as usize], i));

    // Build per-token prefix-sum offsets from the permutation.
    // token_starts[t] = start index in reordered array for token t.
    // token_starts[t+1] - token_starts[t] = number of experts for token t (variable k).
    let mut token_starts: Vec<i32> = vec![0i32; num_tokens + 1];
    for &orig_idx in &perm {
        let token_id = all_token_ids[orig_idx as usize] as usize;
        token_starts[token_id + 1] += 1;
    }
    for i in 1..=num_tokens {
        token_starts[i] += token_starts[i - 1];
    }
    let token_starts_dev = cuda_dev.memcpy_stod(&token_starts)?;

    // Reorder weight_ids to token-major using the same permutation.
    let reordered_weight_ids: Vec<u32> = perm
        .iter()
        .map(|&i| all_weight_ids[i as usize])
        .collect();
    let reordered_wt_ids_dev = cuda_dev.memcpy_stod(&reordered_weight_ids)?;

    // Upload perm so the kernel can gather from down_out directly — no index_select needed.
    let perm_dev = cuda_dev.memcpy_stod(&perm)?;

    let t = profile_now();
    candle::quantized::cuda::fused_deterministic_scatter(
        ys,
        &down_out,
        &perm_dev,
        weights_flat,
        &reordered_wt_ids_dev,
        &token_starts_dev,
        num_tokens,
        cuda_dev,
    )?;
    profile.record("gemm_scatter", t);

    Ok(())
}
