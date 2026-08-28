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
use super::types::MoeInput;
#[cfg(feature = "cuda")]
use crate::models::profile::{profile_now, ProfileAccumulator};
use candle::cuda_backend::wave_provenance::{LeaseOrigin, WaveTicket};
use candle::cuda_backend::Backing;
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
pub(crate) fn extract_weight_info(
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
/// `wave` seeds this phase's inheritance chain. The gather is the first
/// allocation made inside the FFN generation, so it takes the ticket directly;
/// the gate/up/SwiGLU/down GEMMs after it inherit from their operands and need
/// no further mention of the wave.
pub fn compute_experts_grouped(
    input: &MoeInput,
    ys: &mut Tensor,
    experts: &[(&ExpertSlot, &[u32], &[u32])],
    weights_flat: &Tensor,
    profile: &mut ProfileAccumulator,
    wave: Option<WaveTicket>,
) -> Result<()> {
    if experts.is_empty() {
        return Ok(());
    }

    // Device from the output tensor — a q8a128 `input` carries no device.
    let dev = ys.device();
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

    // **Both id tables are device indices, so they are checked here — before the
    // gather — rather than where they happen to be used on the host.**
    //
    // `token_ids` index `xs`/`ys` and `weight_ids` index `weights_flat`, and
    // both are handed to kernels that dereference them with no bound of their
    // own. The one host-side index that would notice a bad `token_id` is the
    // `token_starts[token_id + 1]` accumulation in the scatter setup below, and
    // that runs *after* the gather and all three grouped GEMMs — so an
    // out-of-range id reaches `fused_moe_gather_q8a128` as a device offset
    // first and reads off the end of the activation arena. The Rust bounds
    // check never gets the chance, and the result is a
    // CUDA_ERROR_ILLEGAL_ADDRESS attributed to whichever thread next
    // synchronises.
    //
    // Refusing here costs one pass over a few thousand `u32`s against a kernel
    // launch that cannot be undone (`docs/elastic_vram_partition.md`
    // principle 7). The ids come from routing, and this engine has already had
    // one degenerate-routing fault — `moe_route` leaking a `bi = n_experts`
    // sentinel on `-inf`/NaN logits — whose clamp fixed the symptom while the
    // NaN-logit source stayed open. This is where that would surface next.
    let n_tokens_in = ys.dim(0)?;
    let n_weights = weights_flat.elem_count();
    for (e, &(_slot, toks, wids)) in experts.iter().enumerate() {
        if let Some(&bad) = toks.iter().find(|&&t| t as usize >= n_tokens_in) {
            candle::bail!(
                "grouped expert compute: expert {e} routes token id {bad}, but this \
                 forward has only {n_tokens_in} tokens. The gather would read past \
                 the activation arena and fault inside the kernel."
            );
        }
        if let Some(&bad) = wids.iter().find(|&&w| w as usize >= n_weights) {
            candle::bail!(
                "grouped expert compute: expert {e} names routing-weight id {bad}, \
                 but `weights_flat` holds {n_weights}. The scatter would read past \
                 it and fault inside the kernel."
            );
        }
    }

    // ── Upload token_ids once (shared by gather + scatter) ──
    // Host-built tables have no device operand to inherit from, so they name
    // the submitting layer's span directly — the same span the gather and
    // scatter operands they index into were carved from.
    let upload_root = Backing::from_ticket(wave);
    let tok_ids_dev = cuda_dev.memcpy_stod_from(&all_token_ids, upload_root)?;

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
        match (&gate_shape, &down_shape) {
            (None, None) => {
                gate_shape = Some(gs);
                down_shape = Some(ds);
                gate_dtype = Some(gd);
                down_dtype = Some(dd);
            }
            // **Every expert must match the first, because only the first is
            // read.** `grouped_qmatmul` takes ONE `nrows` and ONE dtype and
            // applies them to the whole pointer array, so an expert whose
            // weights disagree is walked at another expert's stride. Reading
            // long runs off the end of that slot, and a slot near the top of
            // the weight zone is adjacent to `span_end` — past which the
            // reservation is reserved but never mapped, so the over-read is a
            // genuine CUDA_ERROR_ILLEGAL_ADDRESS rather than merely wrong data.
            // That is the fault this drained to `grouped_qmatmul(gate)`.
            //
            // The GPU-native path already refuses a mixed set for exactly this
            // reason (`GpuDispatchTables::build` returns `None` on a dims
            // mismatch and keeps the host path). The host path had no such
            // check, so the mismatch it was falling back to was unguarded.
            (Some(g0), Some(d0)) => {
                if gs.dims() != g0.dims()
                    || ds.dims() != d0.dims()
                    || Some(gd) != gate_dtype
                    || Some(dd) != down_dtype
                {
                    candle::bail!(
                        "grouped expert compute: expert {} has gate {:?}/{:?} down {:?}/{:?}, \
                         but the batch was sized from the first expert's gate {:?}/{:?} down \
                         {:?}/{:?}. The grouped GEMM applies one stride to every weight \
                         pointer, so this expert would be read at the wrong length and walk \
                         off its slot.",
                        gate_ptrs.len() - 1,
                        gs.dims(),
                        gd,
                        ds.dims(),
                        dd,
                        g0.dims(),
                        gate_dtype,
                        d0.dims(),
                        down_dtype,
                    );
                }
            }
            _ => unreachable!("gate_shape and down_shape are set together"),
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

    // **Does every weight pointer still name an expert slot?**
    //
    // The uniformity check above catches a *mismatched* expert. This catches the
    // consequence directly, and so covers any other way a pointer goes wrong —
    // a slot stale from relocation, an eviction that reused the address, an
    // arithmetic slip in `slot_base`. It is the check that caught the residency
    // table `truncate_tables` could not reach: two experts resolving to one
    // slot, and slots an exact multiple of `slot_bytes` below the floor.
    //
    // The pointer is checked, not its extent. A slot's byte length here would
    // have to be derived a second time — `slot_offsets` sizes one from
    // `geom.*_repacked_size`, the GEMX/KO packing the kernel walks, which is
    // not the GGML `elems/block × type_size` this file can compute — and a
    // guard that invents a size reports its own arithmetic as an overrun.
    if let Some(layout) = candle_nn::kv_cache::span_layout(0) {
        let check = |ptrs: &[u64], side: &str| -> Result<()> {
            for (e, &p) in ptrs.iter().enumerate() {
                if p == 0 {
                    candle::bail!("grouped expert compute: {side} pointer for expert {e} is null");
                }
                // Expert slots live at or above `weight_floor`, KV below it. A
                // weight pointer that has fallen below the floor is naming
                // ground the KV side now owns — which is what a slot relocation
                // or eviction the tables did not follow looks like from here.
                //
                // `weight_floor` is read from the live layout, not assumed; it
                // moves whenever the weight side concedes ground.
                if p >= layout.span_base && p < layout.weight_floor {
                    candle::bail!(
                        "grouped expert compute: {side} weights for expert {e} are at {p:#x}, \
                         below the weight floor {:#x} — that is KV ground, not an expert \
                         slot. The pointer outlived the slot it named (relocation or \
                         eviction), and the GEMM would read whatever the KV side put there.",
                        layout.weight_floor,
                    );
                }
                // **Outside the reservation is the case that actually faults.**
                //
                // Everything *inside* the span is mapped end to end, so a wrong
                // pointer there reads garbage and raises nothing. The addresses
                // that genuinely fault are below `span_base` and at or past
                // `span_end`, where the reservation is reserved but never
                // backed. An expert weight pointer should be neither: the whole
                // weight zone lives in `[weight_floor, span_end)`.
                //
                // `slot_base(i) = span_end − (i+1)·slot_bytes` walks *down* as
                // the index rises, so a slot index past what the span can hold
                // produces an address below `span_base` — outside, unmapped,
                // and fatal on first touch.
                if p < layout.span_base || p >= layout.span_end {
                    candle::bail!(
                        "grouped expert compute: {side} weights for expert {e} are at {p:#x}, \
                         outside the reservation [{:#x},{:#x}). That address is not backed by \
                         any mapping, so the GEMM faults on first touch — this is the \
                         CUDA_ERROR_ILLEGAL_ADDRESS, caught at the descriptor.",
                        layout.span_base,
                        layout.span_end,
                    );
                }
            }
            Ok(())
        };
        check(&gate_ptrs, "gate")?;
        check(&up_ptrs, "up")?;
        check(&down_ptrs, "down")?;
    }

    // **The GEMM's other operand: the row ranges it hands each expert.**
    //
    // `grouped_qmatmul` gives expert `e` the rows
    // `[expert_offsets[e], expert_offsets[e+1])` of the gathered activations, so
    // these offsets are the kernel's only bound on a buffer holding exactly
    // `total_batch` rows. An offset past the end walks off the gather output,
    // and the kernel has no way to notice.
    //
    // Every quantity compared here is one the surrounding code already computed
    // — `expert_offsets` is the prefix sum built above, `total_batch` is
    // `all_token_ids.len()`. A guard that derives a bound independently is
    // testing its own arithmetic, not the code.
    if expert_offsets.len() != num_experts + 1 {
        candle::bail!(
            "grouped expert compute: {} expert offsets for {num_experts} experts (want {})",
            expert_offsets.len(),
            num_experts + 1
        );
    }
    if expert_offsets[0] != 0 {
        candle::bail!(
            "grouped expert compute: expert offsets start at {}, not 0",
            expert_offsets[0]
        );
    }
    for w in expert_offsets.windows(2) {
        if w[1] < w[0] {
            candle::bail!(
                "grouped expert compute: expert offsets go backwards ({} then {}), so one \
                 expert would be given a negative row count",
                w[0],
                w[1]
            );
        }
    }
    let last = *expert_offsets.last().expect("checked non-empty above");
    if last as usize != total_batch {
        candle::bail!(
            "grouped expert compute: expert offsets end at {last} but the gather produced \
             {total_batch} rows. The GEMM would read rows past the end of the gathered \
             activations."
        );
    }

    // ── Gather + grouped gate/up/down → down_out. B3 int8: byte-gather the already-quantized
    // q8a1024 router input (no gather-then-quantize); Off: float-gather then FP gemx. ──
    let down_out = match input {
        MoeInput::Q8(op) => {
            use candle::quantized::cuda::{
                fused_moe_gather_q8a128, grouped_qmatmul, silu_mul_q8a128, DynamicTensor,
            };
            let t = profile_now();
            let stacked_q8 = fused_moe_gather_q8a128(
                op,
                &tok_ids_dev,
                total_batch,
                cuda_dev,
                wave.map_or(Backing::Owned, |t| Backing::Lease(LeaseOrigin::Wave(t))),
            )?;
            profile.record("gemm_gather", t);
            let t = profile_now();
            let gate_out = grouped_qmatmul(
                DynamicTensor::Int8(&stacked_q8),
                &gate_ptrs,
                gate_dtype, // KO twin
                gate_nrows,
                &expert_offsets,
                cuda_dev,
                stacked_q8.backing(),
            )?;
            profile.record("gemm_gate", t);
            let t = profile_now();
            let up_out = grouped_qmatmul(
                DynamicTensor::Int8(&stacked_q8),
                &up_ptrs,
                gate_dtype, // up shares gate's KO dtype
                gate_nrows,
                &expert_offsets,
                cuda_dev,
                stacked_q8.backing(),
            )?;
            profile.record("gemm_up", t);
            // B4: fused SwiGLU → q8a128 (silu(gate)·up quantized in one kernel), feeds the down GEMM.
            let t = profile_now();
            let inter_acts =
                silu_mul_q8a128(&gate_out, &up_out, cuda_dev, gate_out.cuda_backing())?;
            profile.record("gemm_silu_mul", t);
            let t = profile_now();
            let down_out = grouped_qmatmul(
                DynamicTensor::Int8(&inter_acts),
                &down_ptrs,
                down_dtype, // KO twin
                down_nrows,
                &expert_offsets,
                cuda_dev,
                inter_acts.backing(),
            )?;
            profile.record("gemm_down", t);
            // Handed on as F32, the type the int8 matmul emits. The fused scatter
            // is selected by where it *writes* — `ys` — and reads this operand as
            // the GEMM's native F32, so converting here would be a full-tensor
            // pass per expert group per layer to hand the kernel a type it does
            // not want. `fused_deterministic_scatter` validates rather than
            // converts, and says so by name.
            down_out
        }
        MoeInput::Float(xs) => {
            let t = profile_now();
            let stacked_xs =
                candle::quantized::cuda::fused_moe_gather(xs, &tok_ids_dev, total_batch, cuda_dev)?;
            profile.record("gemm_gather", t);
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

            let t = profile_now();
            let intermediate = candle_nn::ops::silu_mul(&gate_out, &up_out)?;
            profile.record("gemm_silu_mul", t);

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
            down_out
        }
    };

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
    let num_tokens = input.shape()?.0; // = b_size * seq_len (or b_size for decode)

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
    let token_starts_dev = cuda_dev.memcpy_stod_from(&token_starts, upload_root)?;

    // Reorder weight_ids to token-major using the same permutation.
    let reordered_weight_ids: Vec<u32> = perm.iter().map(|&i| all_weight_ids[i as usize]).collect();
    let reordered_wt_ids_dev = cuda_dev.memcpy_stod_from(&reordered_weight_ids, upload_root)?;

    // Upload perm so the kernel can gather from down_out directly — no index_select needed.
    let perm_dev = cuda_dev.memcpy_stod_from(&perm, upload_root)?;

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
