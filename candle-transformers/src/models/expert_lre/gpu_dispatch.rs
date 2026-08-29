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
//! `ExpertCache::new` after the startup fill, for the reader path
//! `new_prepopulated`. Both hold the slot map by value at that point, so no
//! cross-thread choreography is needed. Construction is best-effort: any
//! anomaly (a missing expert, a non-uniform shape or dtype, a non-CUDA slot)
//! yields `None` and the caller stays on the host-orchestrated path.

use super::cache::ExpertCacheInner;
use super::compute::extract_weight_info;
use candle::cuda_backend::CudaDevice;
use candle::quantized::cuda::MoeBucketizeWorkspace;
use candle::quantized::GgmlDType;
use candle_kernels::simple::moe_bucketize::MAX_EXPERTS;
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
    /// Gate/up output rows (intermediate dim) — uniform across the whole grid.
    pub gate_nrows: usize,
    /// Down output rows (hidden dim) — uniform across the whole grid.
    pub down_nrows: usize,
    /// Gate/up weight dtype, **per covered layer row**.
    gate_dtype: Vec<GgmlDType>,
    /// Down weight dtype, **per covered layer row**.
    ///
    /// Per layer rather than one value for the grid, because a dynamically
    /// quantized checkpoint assigns bit-widths by measured layer sensitivity —
    /// Unsloth's `UD-Q4_K_M` of the 3.6-35B carries `Q5_KO` down-projections up
    /// to layer 33 and `Q6_KO` from 34. That is the checkpoint being correct,
    /// not inconsistent, and it is not a conversion to paper over: the grouped
    /// GEMM already takes `weight_dtype` as a per-CALL argument and is called
    /// once per layer, so the dtype only ever had to reach it. Collapsing it to
    /// one value here is what refused those checkpoints the device path
    /// entirely — silently, since the host path answers identically.
    ///
    /// Uniformity is still required WITHIN a layer: one layer is one grouped
    /// GEMM call over all its experts, and that call takes one dtype.
    down_dtype: Vec<GgmlDType>,
    /// Reusable bucketize workspace shared by every layer's forwards. The
    /// forward path runs layers sequentially on one thread and every kernel
    /// rides the single compute stream, so a lock per layer plus stream
    /// ordering is the whole synchronisation story; the workspace grows on
    /// demand for larger batches (the replaced buffers free stream-ordered).
    pub workspace: Mutex<MoeBucketizeWorkspace>,
}

/// Report that the device path was lost.
///
/// At WARN, and deliberately not at DEBUG: an all-resident cache that cannot
/// build these tables runs every MoE layer through a blocking per-layer routing
/// readback instead, which is a multiple on decode latency. The host path is
/// correct — that is exactly what makes this worth shouting about, because
/// nothing else will. Measured on the 3.6-35B: 11.3 s of a 45 s window went to
/// those syncs, and no line anywhere said why.
fn decline(reason: &str) {
    tracing::warn!(
        reason,
        "expert cache: GPU-native MoE dispatch unavailable — every layer will \
         take the host path's blocking routing readback"
    );
}

impl GpuDispatchTables {
    /// Element offset of `moe_layer_idx`'s expert row block inside the flat
    /// tables, or `None` if the layer is outside the covered range.
    pub fn expert_base(&self, moe_layer_idx: usize) -> Option<usize> {
        Some(self.layer_row(moe_layer_idx)? * self.n_experts)
    }

    /// Row index of `moe_layer_idx` within the covered range.
    fn layer_row(&self, moe_layer_idx: usize) -> Option<usize> {
        let row = moe_layer_idx.checked_sub(self.min_layer)?;
        (row < self.n_layers).then_some(row)
    }

    /// This layer's gate/up weight dtype, or `None` if it is outside the range.
    pub fn gate_dtype(&self, moe_layer_idx: usize) -> Option<GgmlDType> {
        self.gate_dtype.get(self.layer_row(moe_layer_idx)?).copied()
    }

    /// This layer's down weight dtype, or `None` if it is outside the range.
    pub fn down_dtype(&self, moe_layer_idx: usize) -> Option<GgmlDType> {
        self.down_dtype.get(self.layer_row(moe_layer_idx)?).copied()
    }

    /// Build the tables from a fully-staged slot set. `inner.key_to_slot` maps
    /// `(moe_layer_idx, expert_idx) → slot`; the covered layer range is
    /// `[min_layer, max_layer]` and must form a COMPLETE grid with uniform
    /// shapes/dtypes, else `None` (host path).
    ///
    /// **Every `None` here is logged with its reason.** The host path is
    /// lossless, so a decline costs no correctness and shows up only as decode
    /// running several times slower with the cause three crates from the
    /// symptom. Measured on the 3.6-35B, whose 256 experts tripped the id-space
    /// bound below: 11.3 s of a 45 s window went to per-layer routing readback
    /// syncs, and nothing said why.
    pub fn build(inner: &ExpertCacheInner, device: &CudaDevice) -> Option<Self> {
        let keys = &inner.key_to_slot;
        if keys.is_empty() {
            decline("no staged experts");
            return None;
        }
        let min_layer = keys.keys().map(|&(l, _)| l).min()?;
        let max_layer = keys.keys().map(|&(l, _)| l).max()?;
        let n_layers = max_layer - min_layer + 1;
        let n_experts = keys.keys().map(|&(_, e)| e + 1).max()?;
        // The id space the *kernel* supports, named from the kernel rather than
        // repeated here — this bound previously read `128`, which was the routed
        // width of the model in front of it at the time and not a limit of
        // anything, so the next lineage to double its expert count silently lost
        // the whole device path.
        if n_experts > MAX_EXPERTS {
            decline(&format!(
                "{n_experts} experts per layer exceeds the bucketize kernel's {MAX_EXPERTS}"
            ));
            return None;
        }
        if keys.len() != n_layers * n_experts {
            decline(&format!(
                "sparse expert grid: {} staged for {n_layers} layers × {n_experts} experts",
                keys.len()
            ));
            return None;
        }

        let total = n_layers * n_experts;
        let mut gate: Vec<u64> = Vec::with_capacity(total);
        let mut up: Vec<u64> = Vec::with_capacity(total);
        let mut down: Vec<u64> = Vec::with_capacity(total);
        // Shapes must be uniform across the whole grid — every layer's output
        // feeds the same downstream tensors. Dtypes need only be uniform within
        // a layer, which is the granularity the GEMM is called at.
        let mut shape: Option<(usize, usize)> = None;
        let mut gate_dtype: Vec<GgmlDType> = Vec::with_capacity(n_layers);
        let mut down_dtype: Vec<GgmlDType> = Vec::with_capacity(n_layers);
        for l in min_layer..=max_layer {
            let mut layer_dtype: Option<(GgmlDType, GgmlDType)> = None;
            for e in 0..n_experts {
                let slot_idx = *keys.get(&(l, e))?;
                let slot = inner.slots.get(slot_idx)?.as_ref()?;
                let (gp, gs, gd) = extract_weight_info(&slot.gate_proj).ok()?;
                let (upp, us, ud) = extract_weight_info(&slot.up_proj).ok()?;
                let (dp, ds, dd) = extract_weight_info(&slot.down_proj).ok()?;
                let (g_rows, g_cols) = gs.dims2().ok()?;
                let (u_rows, _) = us.dims2().ok()?;
                let (d_rows, d_cols) = ds.dims2().ok()?;
                if u_rows != g_rows || ud != gd {
                    decline("gate and up projections disagree on shape or dtype");
                    return None;
                }
                // Dimensional contract of the device-table pipeline, enforced
                // HERE so exotic model dims keep the host path instead of
                // erroring on every forward: the q8a1024 byte-row gather needs
                // hidden (gate K) % 1024, the grouped GEMM needs N % 32 and
                // K % 128 for both projections.
                if g_cols % 1024 != 0 || g_rows % 32 != 0 || d_rows % 32 != 0 || d_cols % 128 != 0 {
                    decline(&format!(
                        "dims outside the device-table GEMM contract \
                         (hidden {g_cols} %1024, inter {g_rows} %32, \
                         down {d_rows} %32 / {d_cols} %128)"
                    ));
                    return None;
                }
                match shape {
                    None => shape = Some((g_rows, d_rows)),
                    Some(prev) if prev != (g_rows, d_rows) => {
                        decline(&format!(
                            "layer {l} expert {e} is (gate_rows {g_rows}, down_rows {d_rows}) \
                             but the grid so far is (gate_rows {}, down_rows {}) — every \
                             covered layer must share one shape",
                            prev.0, prev.1,
                        ));
                        return None;
                    }
                    Some(_) => {}
                }
                // One layer is one grouped GEMM call per projection, and that
                // call takes one weight dtype — so a layer whose experts
                // disagree cannot be dispatched, while layers that disagree
                // with EACH OTHER are simply different calls.
                match layer_dtype {
                    None => layer_dtype = Some((gd, dd)),
                    Some(prev) if prev != (gd, dd) => {
                        decline(&format!(
                            "layer {l} expert {e} is {gd:?}/{dd:?} but its layer is \
                             {:?}/{:?} — one layer is one GEMM call, so its experts \
                             must share a dtype",
                            prev.0, prev.1,
                        ));
                        return None;
                    }
                    Some(_) => {}
                }
                gate.push(gp);
                up.push(upp);
                down.push(dp);
            }
            let (g, d) = layer_dtype?;
            gate_dtype.push(g);
            down_dtype.push(d);
        }
        let (gate_nrows, down_nrows) = shape?;

        // The device-table GEMM is the int8 (q8a128 × KO) grouped kernel; a
        // slot set staged with non-KO weights has no GPU-native twin, so it
        // keeps the host path rather than erroring on every forward.
        if let Some(row) = gate_dtype
            .iter()
            .zip(&down_dtype)
            .position(|(g, d)| !g.is_ko() || !d.is_ko())
        {
            decline(&format!(
                "layer {} has non-KO expert weights ({:?}/{:?}) — the device GEMM is \
                 the int8 q8a128 × KO kernel",
                min_layer + row,
                gate_dtype[row],
                down_dtype[row],
            ));
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
        // Name the dtype span, not just the count: on a dynamically quantized
        // checkpoint these differ per layer, and "built" alone would leave the
        // reader unable to tell a uniform grid from a mixed one.
        let mut distinct: Vec<(GgmlDType, GgmlDType)> = gate_dtype
            .iter()
            .zip(&down_dtype)
            .map(|(g, d)| (*g, *d))
            .collect();
        distinct.dedup();
        tracing::info!(
            n_layers,
            n_experts,
            gate_nrows,
            down_nrows,
            dtype_runs = distinct.len(),
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
        ko_expert_dtype(device, cuda, seed, GgmlDType::Q6_KO)
    }

    /// The same, at a chosen KO width — for the mixed-precision fixtures, where
    /// the point is that two layers legitimately differ.
    fn ko_expert_dtype(
        device: &Device,
        cuda: &candle::cuda_backend::CudaDevice,
        seed: u32,
        ko: GgmlDType,
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
            let repacked = q.repack_ko(&shape, ko).unwrap();
            let qt = QTensor::new(QStorage::Cuda(repacked), (NROWS, NCOLS)).unwrap();
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
            zone: full_zone(n),
            key_to_slot,
            last_used: vec![0; n],
            generation: 0,
            slot_to_key: vec![None; n],
            expert_scores: vec![],
            num_moe_layers: 0,
            experts_per_layer: 0,
            pinned_layers: 0,
            warm_backed: vec![],
        }
    }

    /// A zone of `n` slots with every one occupied — what these fixtures mean by
    /// "the cache is full". Drained through the real API rather than
    /// constructed full, so the fixture cannot disagree with the allocator about
    /// what occupancy looks like.
    fn full_zone(n: usize) -> candle_nn::kv_cache::WeightZone {
        let mut zone = candle_nn::kv_cache::WeightZone::new(1 << 30, 4096, n, n, 0);
        for _ in 0..n {
            zone.alloc().expect("a fresh zone has every slot free");
        }
        zone
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
        for l in [5, 6] {
            assert!(gd.gate_dtype(l).expect("covered").is_ko());
            assert!(gd.down_dtype(l).expect("covered").is_ko());
        }
        assert_eq!(gd.gate_ptrs.len(), 6);
        assert_eq!(gd.expert_base(5), Some(0));
        assert_eq!(gd.expert_base(6), Some(3));
        assert_eq!(gd.expert_base(4), None, "below the covered range");
        assert_eq!(gd.expert_base(7), None, "above the covered range");
        assert_eq!(gd.gate_dtype(4), None, "dtype follows the covered range");
        assert_eq!(gd.gate_dtype(7), None, "dtype follows the covered range");
    }

    /// The id-space bound must be the KERNEL's, not the routed width of
    /// whichever model was in front of it when the line was written.
    ///
    /// This is the regression that cost a lineage its entire device path: the
    /// bound read `128` because Qwen3-MoE has 128 experts, and the 3.6-35B's 256
    /// then fell to the host path — losslessly, so nothing failed and decode
    /// simply ran several times slower with 11.3 s of every 45 going to
    /// per-layer routing readbacks. A grid AT the kernel's maximum must build.
    #[test]
    fn build_accepts_the_kernels_full_expert_id_space() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        // Compile-time, because it is a fact about a constant: the bucketize
        // kernel is documented for 256 routed experts, and that bound is what
        // the table builder trusts.
        const _: () = assert!(MAX_EXPERTS >= 256);
        let inner = make_inner(&device, &cuda, 0, 1, MAX_EXPERTS, None);
        let gd = GpuDispatchTables::build(&inner, &cuda)
            .expect("a complete grid at the kernel's own maximum must build");
        assert_eq!(gd.n_experts, MAX_EXPERTS);
        assert_eq!(gd.expert_base(0), Some(0));
    }

    /// A dynamically quantized checkpoint varies expert bit-width per LAYER,
    /// and that must build — the grouped GEMM takes `weight_dtype` per call and
    /// is called per layer, so the only real requirement is uniformity within a
    /// layer.
    ///
    /// This is the regression this test exists for: the 3.6-35B's `UD-Q4_K_M`
    /// carries `Q5_KO` down-projections through layer 33 and `Q6_KO` from 34,
    /// the builder collapsed dtype to one grid-wide value, and the whole device
    /// path was refused. Losslessly — the host path answers identically — so
    /// the only symptom was every MoE layer paying a blocking routing readback.
    #[test]
    fn build_accepts_per_layer_dtypes() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let mut inner = make_inner(&device, &cuda, 0, 2, 2, None);
        // Re-quantize layer 1's experts to a different KO width, exactly the
        // shape a sensitivity-driven quantizer produces.
        for e in 0..2 {
            let slot_idx = inner.key_to_slot[&(1, e)];
            inner.slots[slot_idx] = Some(ko_expert_dtype(
                &device,
                &cuda,
                (100 + e) as u32,
                GgmlDType::Q5_KO,
            ));
        }
        let gd = GpuDispatchTables::build(&inner, &cuda)
            .expect("per-layer dtypes must build — one GEMM call per layer");
        assert_eq!(gd.down_dtype(0), Some(GgmlDType::Q6_KO));
        assert_eq!(
            gd.down_dtype(1),
            Some(GgmlDType::Q5_KO),
            "each layer must keep its OWN dtype, not the grid's first",
        );
    }

    /// …but experts WITHIN one layer still cannot disagree: that layer is a
    /// single grouped GEMM call, which takes a single dtype.
    #[test]
    fn build_rejects_mixed_dtypes_inside_one_layer() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let mut inner = make_inner(&device, &cuda, 0, 1, 2, None);
        let slot_idx = inner.key_to_slot[&(0, 1)];
        inner.slots[slot_idx] = Some(ko_expert_dtype(&device, &cuda, 7, GgmlDType::Q5_KO));
        assert!(
            GpuDispatchTables::build(&inner, &cuda).is_none(),
            "one layer is one GEMM call; its experts must share a dtype",
        );
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
            zone: full_zone(1),
            key_to_slot,
            last_used: vec![0],
            generation: 0,
            slot_to_key: vec![None],
            expert_scores: vec![],
            num_moe_layers: 0,
            experts_per_layer: 0,
            pinned_layers: 0,
            warm_backed: vec![],
        };
        assert!(
            GpuDispatchTables::build(&inner, &cuda).is_none(),
            "non-KO weights must fall back to the host path"
        );
    }
}
