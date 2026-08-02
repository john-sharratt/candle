//! GPU-native MoE dispatch tables for the all-resident expert cache.
//!
//! With every expert permanently VRAM-resident, each expert's weight addresses
//! are static for the life of the model. This module captures them ONCE into
//! device-resident pointer tables — flat `[n_layers × n_experts]` u64 per
//! projection — indexed directly inside the grouped GEMM by `moe_bucketize`'s
//! RAW-expert-id tile tables plus a per-layer base offset. That closes the last
//! host dependency in the expert path: routing indices no longer round-trip
//! GPU→CPU→GPU per layer, which is the dominant decode stall on WDDM.
//!
//! Built at cache construction, after the (synchronous) prewarm has staged
//! every expert into its final VRAM slot: for the threaded/mmap cache that is
//! `ExpertCache::new` after `startup_two_tier`, for the reader path
//! `new_prepopulated`. Both hold the slot map by value at that point, so no
//! cross-thread choreography is needed. Construction is best-effort: any
//! anomaly (a missing expert, a non-uniform shape or dtype, a non-CUDA slot)
//! yields `None` and the caller stays on the host-orchestrated path.

use super::cache::ExpertCacheInner;
use super::compute::extract_weight_info;
use candle::cuda_backend::CudaDevice;
use candle::quantized::cuda::MoeBucketizeWorkspace;
use candle::quantized::GgmlDType;
use cudarc::driver::CudaSlice;
use std::sync::Mutex;

/// Static device-resident expert dispatch tables covering every MoE layer the
/// cache serves.
pub struct GpuDispatchTables {
    /// Per-expert gate-projection weight pointers, `[n_layers × n_experts]`.
    pub gate_ptrs: CudaSlice<u64>,
    /// Per-expert up-projection weight pointers, `[n_layers × n_experts]`.
    pub up_ptrs: CudaSlice<u64>,
    /// Per-expert down-projection weight pointers, `[n_layers × n_experts]`.
    pub down_ptrs: CudaSlice<u64>,
    /// Experts per layer (contiguous ids `0..n_experts`).
    pub n_experts: usize,
    /// Lowest MoE layer index the tables cover (layer rows are contiguous
    /// from here; a single-layer cache stores its one layer at row 0).
    min_layer: usize,
    /// Number of contiguous layer rows.
    n_layers: usize,
    /// Gate/up output rows (intermediate dim) — uniform across experts.
    pub gate_nrows: usize,
    /// Down output rows (hidden dim) — uniform across experts.
    pub down_nrows: usize,
    /// Gate/up weight dtype (KO twin on the int8 path) — uniform.
    pub gate_dtype: GgmlDType,
    /// Down weight dtype — uniform.
    pub down_dtype: GgmlDType,
    /// Reusable bucketize workspace shared by every layer's forwards. The
    /// forward path runs layers sequentially on one thread and every kernel
    /// rides the single compute stream, so a lock per layer plus stream
    /// ordering is the whole synchronisation story; the workspace grows on
    /// demand for larger batches (the replaced buffers free stream-ordered).
    pub workspace: Mutex<MoeBucketizeWorkspace>,
}

impl GpuDispatchTables {
    /// Element offset of `moe_layer_idx`'s expert row block inside the flat
    /// tables, or `None` if the layer is outside the covered range.
    pub fn expert_base(&self, moe_layer_idx: usize) -> Option<usize> {
        let row = moe_layer_idx.checked_sub(self.min_layer)?;
        if row >= self.n_layers {
            return None;
        }
        Some(row * self.n_experts)
    }

    /// Build the tables from a fully-staged slot set. `inner.key_to_slot` maps
    /// `(moe_layer_idx, expert_idx) → slot`; the covered layer range is
    /// `[min_layer, max_layer]` and must form a COMPLETE grid with uniform
    /// shapes/dtypes, else `None` (host path).
    pub fn build(inner: &ExpertCacheInner, device: &CudaDevice) -> Option<Self> {
        let keys = &inner.key_to_slot;
        if keys.is_empty() {
            return None;
        }
        let min_layer = keys.keys().map(|&(l, _)| l).min()?;
        let max_layer = keys.keys().map(|&(l, _)| l).max()?;
        let n_layers = max_layer - min_layer + 1;
        let n_experts = keys.keys().map(|&(_, e)| e + 1).max()?;
        if n_experts > 128 || keys.len() != n_layers * n_experts {
            return None; // sparse grid or oversized id space — host path
        }

        let total = n_layers * n_experts;
        let mut gate: Vec<u64> = Vec::with_capacity(total);
        let mut up: Vec<u64> = Vec::with_capacity(total);
        let mut down: Vec<u64> = Vec::with_capacity(total);
        let mut dims: Option<(usize, usize, GgmlDType, GgmlDType)> = None;
        for l in min_layer..=max_layer {
            for e in 0..n_experts {
                let slot_idx = *keys.get(&(l, e))?;
                let slot = inner.slots.get(slot_idx)?.as_ref()?;
                let (gp, gs, gd) = extract_weight_info(&slot.gate_proj).ok()?;
                let (upp, us, ud) = extract_weight_info(&slot.up_proj).ok()?;
                let (dp, ds, dd) = extract_weight_info(&slot.down_proj).ok()?;
                let (g_rows, _) = gs.dims2().ok()?;
                let (u_rows, _) = us.dims2().ok()?;
                let (d_rows, _) = ds.dims2().ok()?;
                if u_rows != g_rows || ud != gd {
                    return None;
                }
                match dims {
                    None => dims = Some((g_rows, d_rows, gd, dd)),
                    Some(prev) if prev != (g_rows, d_rows, gd, dd) => return None,
                    Some(_) => {}
                }
                gate.push(gp);
                up.push(upp);
                down.push(dp);
            }
        }
        let (gate_nrows, down_nrows, gate_dtype, down_dtype) = dims?;

        // The device-table GEMM is the int8 (q8a128 × KO) grouped kernel; a
        // slot set staged with non-KO weights has no GPU-native twin, so it
        // keeps the host path rather than erroring on every forward.
        if !gate_dtype.is_ko() || !down_dtype.is_ko() {
            tracing::warn!(
                ?gate_dtype,
                ?down_dtype,
                "expert cache: non-KO expert weights — GPU-native dispatch disabled (host path)"
            );
            return None;
        }
        // The bucketize kernel launches on the device's compute-stream handle
        // while its consumers (gather / grouped GEMM / scatter) launch on the
        // legacy null stream; they are ordered only because candle's compute
        // stream IS the null stream. If that ever changes, fall back to the
        // host path instead of silently racing the tables.
        if !device.cuda_stream().cu_stream().is_null() {
            tracing::warn!(
                "expert cache: compute stream is not the legacy null stream — GPU-native \
                 dispatch disabled (host path)"
            );
            return None;
        }

        let gate_ptrs = device.memcpy_stod(&gate).ok()?;
        let up_ptrs = device.memcpy_stod(&up).ok()?;
        let down_ptrs = device.memcpy_stod(&down).ok()?;
        let workspace = MoeBucketizeWorkspace::new(device, 64, 8).ok()?;
        tracing::info!(
            n_layers,
            n_experts,
            gate_nrows,
            down_nrows,
            "expert cache: GPU-native dispatch tables built (routing readback eliminated)"
        );
        Some(Self {
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            n_experts,
            min_layer,
            n_layers,
            gate_nrows,
            down_nrows,
            gate_dtype,
            down_dtype,
            workspace: Mutex::new(workspace),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::ExpertSlot;
    use super::*;
    use crate::models::quantized_matmul::QMatMul;
    use candle::quantized::cuda::QCudaStorage;
    use candle::quantized::{GgmlDType, QStorage, QTensor};
    use candle::{Device, Shape, Tensor};

    const NROWS: usize = 64;
    const NCOLS: usize = 1024;

    /// One expert's gate/up/down as KO-repacked CUDA QMatMuls (the production
    /// weight shape `extract_weight_info` reads).
    fn ko_expert(
        device: &Device,
        cuda: &candle::cuda_backend::CudaDevice,
        seed: u32,
    ) -> ExpertSlot {
        let proj = |salt: u32| -> QMatMul {
            let w: Vec<f32> = (0..NROWS * NCOLS)
                .map(|i| {
                    (((i as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add(seed * 31 + salt)
                        % 1000) as f32
                        / 500.0)
                        - 1.0
                })
                .collect();
            let t = Tensor::from_vec(w, (NROWS, NCOLS), device).unwrap();
            let shape = Shape::from((NROWS, NCOLS));
            let mut q = QCudaStorage::zeros(cuda, NROWS * NCOLS, GgmlDType::Q6_K).unwrap();
            let (storage, _l) = t.storage_and_layout();
            let cs = match &*storage {
                candle::Storage::Cuda(c) => c,
                _ => unreachable!(),
            };
            q.quantize(cs).unwrap();
            let ko = q.repack_ko(&shape, GgmlDType::Q6_KO).unwrap();
            let qt = QTensor::new(QStorage::Cuda(ko), (NROWS, NCOLS)).unwrap();
            QMatMul::from_qtensor_repacked(qt).unwrap()
        };
        ExpertSlot {
            gate_proj: proj(1),
            up_proj: proj(2),
            down_proj: proj(3),
        }
    }

    /// Assemble an `ExpertCacheInner` with a `layers × experts` grid starting
    /// at `min_layer`; `skip` omits one `(layer, expert)` to test sparseness.
    fn make_inner(
        device: &Device,
        cuda: &candle::cuda_backend::CudaDevice,
        min_layer: usize,
        n_layers: usize,
        n_experts: usize,
        skip: Option<(usize, usize)>,
    ) -> ExpertCacheInner {
        let mut slots = Vec::new();
        let mut key_to_slot = std::collections::HashMap::new();
        for l in min_layer..min_layer + n_layers {
            for e in 0..n_experts {
                if skip == Some((l, e)) {
                    continue;
                }
                key_to_slot.insert((l, e), slots.len());
                slots.push(Some(ko_expert(device, cuda, (l * 131 + e) as u32)));
            }
        }
        let n = slots.len();
        ExpertCacheInner {
            slots,
            free_slots: vec![],
            key_to_slot,
            last_used: vec![0; n],
            generation: 0,
            slot_to_key: vec![None; n],
            expert_scores: vec![],
            num_moe_layers: 0,
            experts_per_layer: 0,
        }
    }

    #[test]
    fn build_complete_grid_yields_tables_with_correct_bases() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        // Layers 5..7, 3 experts each — exercises the min_layer offset.
        let inner = make_inner(&device, &cuda, 5, 2, 3, None);
        let gd = GpuDispatchTables::build(&inner, &cuda).expect("complete KO grid must build");
        assert_eq!(gd.n_experts, 3);
        assert_eq!(gd.gate_nrows, NROWS);
        assert_eq!(gd.down_nrows, NROWS);
        assert!(gd.gate_dtype.is_ko() && gd.down_dtype.is_ko());
        assert_eq!(gd.gate_ptrs.len(), 6);
        assert_eq!(gd.expert_base(5), Some(0));
        assert_eq!(gd.expert_base(6), Some(3));
        assert_eq!(gd.expert_base(4), None, "below the covered range");
        assert_eq!(gd.expert_base(7), None, "above the covered range");
    }

    #[test]
    fn build_rejects_sparse_grid() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let inner = make_inner(&device, &cuda, 0, 2, 3, Some((1, 1)));
        assert!(
            GpuDispatchTables::build(&inner, &cuda).is_none(),
            "a missing (layer, expert) must fall back to the host path"
        );
    }

    #[test]
    fn build_rejects_non_ko_weights() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        // Plain Q6_K (no KO repack): the int8 device-table GEMM has no kernel
        // for it, so the tables must not build.
        let w: Vec<f32> = (0..NROWS * NCOLS).map(|i| (i % 7) as f32 - 3.0).collect();
        let t = Tensor::from_vec(w, (NROWS, NCOLS), &device).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q6_K).unwrap();
        let proj =
            || QMatMul::from_qtensor(QTensor::quantize(&t, GgmlDType::Q6_K).unwrap()).unwrap();
        let _ = qt;
        let slot = ExpertSlot {
            gate_proj: proj(),
            up_proj: proj(),
            down_proj: proj(),
        };
        let mut key_to_slot = std::collections::HashMap::new();
        key_to_slot.insert((0usize, 0usize), 0usize);
        let inner = ExpertCacheInner {
            slots: vec![Some(slot)],
            free_slots: vec![],
            key_to_slot,
            last_used: vec![0],
            generation: 0,
            slot_to_key: vec![None],
            expert_scores: vec![],
            num_moe_layers: 0,
            experts_per_layer: 0,
        };
        assert!(
            GpuDispatchTables::build(&inner, &cuda).is_none(),
            "non-KO weights must fall back to the host path"
        );
    }
}
