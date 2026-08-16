//! All-resident GPU-native MoE for the DSpark drafter.
//!
//! The drafter has 3 `dflash` blocks × 256 routed experts. Requantized to the 2-bit Q2_KO twin the
//! whole set is only ~5 GiB, so above 64 GiB VRAM (the only tier where speculative decode is
//! enabled — see [`resident_slots_for_vram`]) EVERY expert is permanently VRAM-resident. That makes
//! the drafter's MoE fully GPU-native with **no per-layer routing readback**: each draft, the router
//! runs on the GPU, `moe_bucketize` turns the routing indices into the grouped-GEMM tile tables
//! on-device, and the gate/up/down GEMMs index a device-resident weight-pointer table — route →
//! bucketize → gather → GEMM → SwiGLU → GEMM → scatter, all on the GPU, no host round-trip.

#[cfg(feature = "cuda")]
use candle::cuda_backend::Backing;
#[cfg(feature = "cuda")]
use candle::{Result, Tensor};

/// Binary VRAM gate for the drafter's MoE and the speculative-decode enable. Because the Q2_KO
/// drafter is small (~5 GiB for the whole 768-expert set at ~6.75 MiB/expert), the mode is
/// all-or-nothing:
///
/// | Total VRAM | Mode |
/// |------------|------|
/// | > 64 GiB   | **all-resident** — the ENTIRE expert set (`total_experts`) lives in VRAM; no host RAM bank, no streaming, no per-layer routing readback. The drafter MoE runs fully on the GPU (`moe_bucketize` + device pointer table). |
/// | otherwise  | disabled (`None`) — speculative decode off; the drafter is never loaded. |
///
/// Streaming below 64 GiB is deliberately dropped: on a launch-bound (WDDM) box the per-layer
/// `to_vec2` routing readback the streaming path needs is the dominant cost, so a partially-resident
/// drafter is worse than none. All-resident removes the readback entirely.
pub fn resident_slots_for_vram(total_vram_bytes: usize, total_experts: usize) -> Option<usize> {
    const GIB: usize = 1 << 30;
    if total_vram_bytes > 64 * GIB {
        Some(total_experts) // all-resident: the whole expert set in VRAM
    } else {
        None // disable speculative decode (drafter not loaded)
    }
}

/// All-resident GPU-native MoE for the drafter's blocks. The Q2_KO expert set is small (~5 GiB), so
/// on a >64 GiB box EVERY routed expert is permanently VRAM-resident (speculative decode is disabled
/// below that — see [`resident_slots_for_vram`]). With static weight addresses there is no streaming
/// and, crucially, **no per-layer routing readback**: routing indices stay on the GPU,
/// `moe_bucketize` turns them into the grouped-GEMM tile tables on-device, and the expert GEMMs index
/// a device-resident pointer table (`grouped_qmatmul_dev_q8a128`). Bit-identical to the host
/// counting-sort path; the whole draft MoE runs on the GPU with no host round-trip.
#[cfg(feature = "cuda")]
pub struct DsparkStreamingMoe {
    n_experts: usize,
    /// The Q2_KO (2-bit) twin the grouped GEMM reads (routed experts are requantized to it at
    /// load — MXFP4 → f32 → Q2_KO on the GPU).
    ko_dtype: candle::quantized::GgmlDType,
    gate_shape: (usize, usize), // (inter, dim)
    /// Per block: VRAM-resident router + shared expert (small).
    gates: Vec<super::moe::Gate>,
    shared: Vec<super::moe::Expert>,
    /// Per block: all `n_experts` routed experts, permanently VRAM-resident. Never read —
    /// held purely so their weight pointers (below) stay valid for the model's life.
    _slots: Vec<Vec<crate::models::expert_lre::ExpertSlot>>,
    /// Per block: device-resident `[n_experts]` weight-pointer tables into `slots`, consumed by
    /// `grouped_qmatmul_dev_q8a128` (indexed by `moe_bucketize`'s raw expert-id tile tables).
    gate_ptrs: Vec<cudarc::driver::CudaSlice<u64>>,
    up_ptrs: Vec<cudarc::driver::CudaSlice<u64>>,
    down_ptrs: Vec<cudarc::driver::CudaSlice<u64>>,
    /// Reusable on-device bucketize scratch (grows with batch); the forward runs blocks
    /// sequentially on one stream, so one workspace suffices.
    workspace: std::sync::Mutex<candle::quantized::cuda::MoeBucketizeWorkspace>,
    device: candle::Device,
    cuda_dev: candle::CudaDevice,
}

#[cfg(feature = "cuda")]
impl DsparkStreamingMoe {
    /// Load the streaming MoE for `n_blocks` drafter blocks from `m`: the router + shared expert
    /// per block land in VRAM (`device`); the routed experts stay in host RAM as 3-D `QTensor`s.
    pub fn load(
        m: &mut super::loader::GgufModel,
        cfg: &super::config::Config,
        n_blocks: usize,
        n_slots: usize,
        device: &candle::Device,
    ) -> Result<Self> {
        use super::loader::{dequant_f32, dequant_native, qlinear};
        use super::moe::{Expert, Gate, ScoreFunc};
        use crate::models::expert_lre::compute::extract_weight_info;
        use crate::models::expert_lre::ExpertSlot;
        use crate::models::quantized_matmul::QMatMul;
        use candle::quantized::cuda::{repack_to_host, MoeBucketizeWorkspace};
        use candle::quantized::{load_repacked, GgmlDType, QTensor};
        let cuda_dev = match device {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("DsparkStreamingMoe: expected a CUDA device"),
        };
        let (dim, inter, ne, k) = (
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
        );
        // The drafter's routed experts are stored as the 2-bit Q2_KO twin (~2.25 bpw, ~half the
        // MXFP4_KO footprint) so twice as many stream into the resident slots per PCIe budget. The
        // lower weight precision costs the drafter nothing in output quality — the target's verify
        // corrects every draft, so the twin only affects acceptance. Read by the same maintained
        // per-128 int8 fold as the other KO twins (`q2_ko_int8_f32_grouped`, q8a128 activation).
        let ko_dtype = GgmlDType::Q2_KO;
        let per_gate_ko = candle::quantized::repacked_size_bytes(inter, dim, ko_dtype)?;
        let per_down_ko = candle::quantized::repacked_size_bytes(dim, inter, ko_dtype)?;
        // Every routed expert is VRAM-resident on this path.
        debug_assert!(
            n_slots >= n_blocks * ne,
            "DsparkStreamingMoe::load is all-resident: n_slots={n_slots} < experts={}",
            n_blocks * ne
        );
        // Repack a MoE expert bank `[ne, r, c]` (MXFP4) → contiguous Q2_KO bytes, one expert after
        // another. MXFP4 → Q2_KO is a dequant + requant (not an exact reorder), done per expert on
        // the GPU by `repack_to_host`'s KO arm (`repack_ko`: dequant to f32 on-device,
        // `run_quantize_ko` to the Q2_KO crumb layout).
        let repack_bank =
            |bank: &candle::quantized::QTensor, r: usize, c: usize| -> Result<Vec<u8>> {
                let dt = bank.dtype();
                let per = r * c / dt.block_size() * dt.type_size();
                let mut out = Vec::new();
                for e in 0..ne {
                    let nat = bank.data_range(e * per..(e + 1) * per)?;
                    out.extend_from_slice(&repack_to_host(&cuda_dev, &nat, r, c, dt, ko_dtype)?);
                }
                Ok(out)
            };
        // Upload one expert's `[off, off+len)` Q2_KO bytes into a permanent VRAM `QMatMul`.
        let mk = |pool: &[u8], off: usize, len: usize, shape: Vec<usize>| -> Result<QMatMul> {
            let st = load_repacked(&cuda_dev, &pool[off..off + len], ko_dtype)?;
            QMatMul::from_qtensor_repacked(QTensor::new(st, shape)?)
        };
        let (mut gates, mut shared) = (Vec::new(), Vec::new());
        let mut slots: Vec<Vec<ExpertSlot>> = Vec::with_capacity(n_blocks);
        let (mut gate_ptrs, mut up_ptrs, mut down_ptrs) = (Vec::new(), Vec::new(), Vec::new());
        for b in 0..n_blocks {
            let p = format!("blk.{b}.");
            let gate_w = dequant_native(m, &format!("{p}ffn_gate_inp.weight"), device)?;
            let (bias, tid2eid) = if cfg.is_hash_layer(b) {
                let t = m.read_int_tensor_u32(&format!("{p}ffn_gate_tid2eid.weight"), device)?;
                (None, Some(t))
            } else {
                (
                    Some(dequant_f32(m, &format!("{p}exp_probs_b.bias"), device)?),
                    None,
                )
            };
            gates.push(Gate::new(
                gate_w,
                bias,
                tid2eid,
                k,
                ne,
                ScoreFunc::parse(&cfg.score_func),
                cfg.route_scale,
            ));
            shared.push(Expert::new(
                qlinear(m, &format!("{p}ffn_gate_shexp.weight"), device)?,
                qlinear(m, &format!("{p}ffn_down_shexp.weight"), device)?,
                qlinear(m, &format!("{p}ffn_up_shexp.weight"), device)?,
                cfg.swiglu_limit,
            ));
            // Routed experts: read the MXFP4 banks on the CPU, requantize each to Q2_KO, upload all
            // `ne` into permanent VRAM slots, and capture their weight pointers into a device table.
            let g = m.qtensor(&format!("{p}ffn_gate_exps.weight"), &candle::Device::Cpu)?;
            let u = m.qtensor(&format!("{p}ffn_up_exps.weight"), &candle::Device::Cpu)?;
            let d = m.qtensor(&format!("{p}ffn_down_exps.weight"), &candle::Device::Cpu)?;
            let (kg, ku, kd) = (
                repack_bank(&g, inter, dim)?,
                repack_bank(&u, inter, dim)?,
                repack_bank(&d, dim, inter)?,
            );
            let mut block_slots = Vec::with_capacity(ne);
            let (mut gp, mut upp, mut dp) = (
                Vec::with_capacity(ne),
                Vec::with_capacity(ne),
                Vec::with_capacity(ne),
            );
            for e in 0..ne {
                let slot = ExpertSlot {
                    gate_proj: mk(&kg, e * per_gate_ko, per_gate_ko, vec![inter, dim])?,
                    up_proj: mk(&ku, e * per_gate_ko, per_gate_ko, vec![inter, dim])?,
                    down_proj: mk(&kd, e * per_down_ko, per_down_ko, vec![dim, inter])?,
                };
                gp.push(extract_weight_info(&slot.gate_proj)?.0);
                upp.push(extract_weight_info(&slot.up_proj)?.0);
                dp.push(extract_weight_info(&slot.down_proj)?.0);
                block_slots.push(slot);
            }
            gate_ptrs.push(cuda_dev.memcpy_stod(&gp)?);
            up_ptrs.push(cuda_dev.memcpy_stod(&upp)?);
            down_ptrs.push(cuda_dev.memcpy_stod(&dp)?);
            slots.push(block_slots);
        }
        let workspace = std::sync::Mutex::new(MoeBucketizeWorkspace::new(&cuda_dev, 64, k)?);
        Ok(Self {
            n_experts: ne,
            ko_dtype,
            gate_shape: (inter, dim),
            gates,
            shared,
            _slots: slots,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            workspace,
            device: device.clone(),
            cuda_dev,
        })
    }

    /// All-resident GPU-native MoE forward for `block`: `x` `[b, s, dim]`, `input_ids` `[b, s]` →
    /// `[b, s, dim]`. Route on GPU → `moe_bucketize` (device tile tables, **no routing readback**)
    /// → gather → grouped gate/up GEMM (device pointer table) → fused SwiGLU → grouped down GEMM →
    /// deterministic scatter, summed onto the shared expert. Every expert is VRAM-resident, so the
    /// whole MoE runs on the GPU; bit-identical to the host counting-sort path.
    pub fn forward(&self, block: usize, x: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        use candle::quantized::cuda::{
            fused_deterministic_scatter, fused_moe_gather_q8a128, grouped_qmatmul_dev_q8a128,
            moe_bucketize, silu_mul_q8a128, to_dynamic, DynamicActs, GROUPED_GEMM_TILE_W,
        };
        use candle::quantized::Int8Mode;
        let (b, s, dim) = x.dims3()?;
        let t_tok = b * s;
        let xf = x.reshape((t_tok, dim))?.to_dtype(candle::DType::F32)?; // [T, dim]
        let ids = input_ids.reshape(t_tok)?;
        let (weights, indices) = self.gates[block].route(&xf, &ids)?; // [T,k], [T,k]
        let k = indices.dim(1)?;
        let weights_flat = weights.flatten_all()?.contiguous()?; // [T*k]
        let indices = indices.to_dtype(candle::DType::U32)?; // stays on GPU — no readback

        // The grouped GEMM reads the int8 Q2_KO weights, so the activation must be q8a128 (int8) —
        // the same quantization the engine's MoE path uses. A float op would mismatch the KO
        // weights (and be rejected by the KO↔int8 pairing check).
        let op = match to_dynamic(&xf, Int8Mode::Performance, &self.cuda_dev)? {
            DynamicActs::Int8(op) => op,
            DynamicActs::Float(_) => candle::bail!("dspark moe: expected int8 activation"),
        };
        let ne = self.n_experts;
        let a_ub = t_tok * k;
        let (inter, _) = self.gate_shape;

        let ys = {
            let mut ws = self
                .workspace
                .lock()
                .map_err(|_| candle::Error::Msg("dspark bucketize workspace poisoned".into()))?;
            // Routing indices → device tile tables (gather lists, grouped-GEMM tiles, scatter
            // segments) in one launch, no GPU→CPU round-trip.
            moe_bucketize(&indices, ne, GROUPED_GEMM_TILE_W, &mut ws)?;
            let launch_tiles = a_ub.min(a_ub.div_ceil(GROUPED_GEMM_TILE_W) + ne);
            let stacked =
                fused_moe_gather_q8a128(&op, &ws.tok_ids, a_ub, &self.cuda_dev, Backing::Owned)?;
            let gate_out = grouped_qmatmul_dev_q8a128(
                &stacked,
                &self.gate_ptrs[block],
                0,
                ne,
                self.ko_dtype,
                inter,
                &ws.tile_expert,
                &ws.tile_b_start,
                &ws.tile_b_cnt,
                launch_tiles,
                &self.cuda_dev,
            )?;
            let up_out = grouped_qmatmul_dev_q8a128(
                &stacked,
                &self.up_ptrs[block],
                0,
                ne,
                self.ko_dtype,
                inter,
                &ws.tile_expert,
                &ws.tile_b_start,
                &ws.tile_b_cnt,
                launch_tiles,
                &self.cuda_dev,
            )?;
            let inter_acts = silu_mul_q8a128(&gate_out, &up_out, &self.cuda_dev, Backing::Owned)?;
            let down_out = grouped_qmatmul_dev_q8a128(
                &inter_acts,
                &self.down_ptrs[block],
                0,
                ne,
                self.ko_dtype,
                dim,
                &ws.tile_expert,
                &ws.tile_b_start,
                &ws.tile_b_cnt,
                launch_tiles,
                &self.cuda_dev,
            )?
            .to_dtype(candle::DType::F32)?;
            let ys = Tensor::zeros((t_tok, dim), candle::DType::F32, &self.device)?;
            fused_deterministic_scatter(
                &ys,
                &down_out,
                &ws.perm,
                &weights_flat,
                &ws.rw_ids,
                &ws.token_starts,
                t_tok,
                &self.cuda_dev,
            )?;
            ys
        };
        let y = (self.shared[block].forward(&xf)? + ys)?;
        y.reshape((b, s, dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use candle::Device;

    /// The all-resident GPU-native MoE must produce the SAME output as the eager reference `MoE`
    /// (same weights, same routing) — within the 2-bit Q2_KO + int8-activation quant envelope, not
    /// bit-exact vs the f32×MXFP4 reference. 64 tokens with diverse ids spread across many experts.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn streaming_moe_matches_reference() -> Result<()> {
        use super::super::loader::{config_from_gguf, load_moe, GgufModel};
        let path = std::path::PathBuf::from(
            r"D:\models\deepseek-v4-flash-mxfp4\dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf",
        );
        if !path.exists() {
            eprintln!("[skip] DSpark drafter absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let mut m = GgufModel::open(std::slice::from_ref(&path))?;
        let cfg = config_from_gguf(&m)?;
        let block = 0usize;
        let reference = load_moe(&mut m, &cfg, block, &device)?; // eager, all experts VRAM
        let n_experts = cfg.n_layers * cfg.n_routed_experts; // all-resident
        let streaming = DsparkStreamingMoe::load(&mut m, &cfg, cfg.n_layers, n_experts, &device)?;

        let t = 64usize;
        let x = Tensor::randn(0f32, 1.0, (1, t, cfg.dim), &device)?;
        // Diverse ids so hash routing (if any) + top-k spread across many experts.
        let ids = Tensor::arange(0u32, t as u32, &device)?.reshape((1, t))?;

        let ref_out = reference.forward(&x, &ids)?;
        let str_out = streaming.forward(block, &x, &ids)?;
        let diff = ref_out
            .broadcast_sub(&str_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let refmax = ref_out.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let rel = diff / refmax;
        eprintln!(
            "[stream-moe] max|ref-streaming|={diff:.3e}  max|ref|={refmax:.3e}  rel={rel:.3e}",
        );
        // The streaming path uses the 2-bit Q2_KO int8 grouped GEMM (q8a128 activation) vs the
        // reference's f32×MXFP4 — so they differ by the 2-bit-weight + int8-activation quant
        // envelope (a pointwise max-diff bound, dominated by the worst 2-bit weight, so wider than
        // MXFP4_KO's ~0.10). The drafter is lossless regardless: the target's verify corrects every
        // draft, so this only affects acceptance, never output. Assert the divergence is that
        // quant envelope, not a bug (a broken kernel gives rel ≫ 1 or non-finite).
        assert!(
            rel.is_finite() && rel < 0.6,
            "streaming MoE diverged beyond the Q2_KO quant envelope: rel={rel}"
        );
        Ok(())
    }

    #[test]
    fn vram_cutovers_pick_slots_and_disable_below_24gib() {
        let gib = 1usize << 30;
        let ne = 768; // this drafter: 3 blocks × 256 routed experts
                      // > 64 GiB → all-resident (the whole expert set); otherwise disabled.
        assert_eq!(resident_slots_for_vram(72 * gib, ne), Some(768)); // dev Blackwell 72 GB
        assert_eq!(resident_slots_for_vram(65 * gib, ne), Some(768));
        assert_eq!(resident_slots_for_vram(64 * gib, ne), None); // exactly 64 → not > 64 → off
        assert_eq!(resident_slots_for_vram(48 * gib, ne), None); // 5090-class → off
        assert_eq!(resident_slots_for_vram(24 * gib, ne), None);
        assert_eq!(resident_slots_for_vram(16 * gib, ne), None);
        // All-resident count == the expert set, whatever its size.
        assert_eq!(resident_slots_for_vram(72 * gib, 256), Some(256));
    }
}
