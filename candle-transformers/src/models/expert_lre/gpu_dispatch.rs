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
#[cfg(feature = "tensor-assert")]
use super::slot_integrity::{arm_watch, describe};
use super::slot_integrity::{Drifted, SlotIntegrity};
use super::zone_geometry::ZoneGeometry;
use candle::cuda_backend::CudaDevice;
#[cfg(feature = "tensor-assert")]
use candle::quantized::cuda::ko_repacked_bytes;
use candle::quantized::cuda::MoeBucketizeWorkspace;
use candle::quantized::GgmlDType;
#[cfg(feature = "tensor-assert")]
use candle::Shape;
use candle_kernels::simple::moe_bucketize::MAX_EXPERTS;
use cudarc::driver::CudaSlice;
#[cfg(feature = "tensor-assert")]
use std::sync::Arc;
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
    /// The weight zone's capacity when these tables were captured.
    ///
    /// The tables hold raw slot addresses, taken once on the reasoning that an
    /// all-resident cache's weight addresses are static. That reasoning has a
    /// hole: **all-resident at startup is not resident for ever.** The zone
    /// concedes ground to the KV side under memory pressure
    /// (`WeightZone::retract_to`), and every slot at or past the new frontier
    /// stops being an expert slot — the KV arena allocates that same ground and
    /// writes to it, while these tables still point there.
    ///
    /// The host path already refuses a weight pointer that has fallen below the
    /// live `weight_floor` ("that is KV ground, not an expert slot"). The
    /// GPU-native path had no equivalent, because it never re-reads the layout.
    /// This is what lets it notice. See [`Self::zone_moved`].
    built_capacity: usize,
    /// The lowest slot address the tables reference, i.e. `slot_base(capacity-1)`
    /// at capture time. Recorded rather than recomputed so the comparison
    /// survives a change in how slot addresses are derived.
    built_floor: u64,
    /// The cache's concession count when these tables were captured.
    ///
    /// The load-bearing one. Capacity and floor both return to their original
    /// values when the zone grows back after conceding, so a comparison of
    /// either passes while the conceded slots hold something else entirely.
    /// This does not come back.
    built_concede_epoch: u64,
    /// Fletcher-32 of every weight in the grid, taken once after the fill.
    ///
    /// The grid is fully resident and nothing evicts, so these are the whole
    /// truth about what should be at those addresses for the life of the
    /// process. Re-checked only when something has already gone wrong — see
    /// [`Self::verify_integrity`].
    integrity: Option<SlotIntegrity>,
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
        // Real allocated bytes per weight, in the same order as the pointers.
        let mut gate_bytes: Vec<i64> = Vec::with_capacity(total);
        let mut up_bytes: Vec<i64> = Vec::with_capacity(total);
        let mut down_bytes: Vec<i64> = Vec::with_capacity(total);
        // Shapes must be uniform across the whole grid — every layer's output
        // feeds the same downstream tensors. Dtypes need only be uniform within
        // a layer, which is the granularity the GEMM is called at.
        let mut shape: Option<(usize, usize)> = None;
        // The two projections' K extents, for the weight fingerprints — which
        // need a byte length, and that is a function of the full shape.
        //
        // They are NOT the same number: gate/up map hidden → intermediate, so
        // their K is the hidden width; down maps intermediate → hidden, so its
        // K is the intermediate width. Using one for the other reports a
        // mismatch of exactly the hidden:intermediate ratio on every down weight
        // in the grid — an arithmetic error wearing the costume of a discovery.
        //
        // Only the fingerprint reads them, so they exist only where it does.
        #[cfg(feature = "tensor-assert")]
        let (mut cols, mut down_cols): (Option<usize>, Option<usize>) = (None, None);
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
                    None => {
                        shape = Some((g_rows, d_rows));
                        #[cfg(feature = "tensor-assert")]
                        {
                            cols = Some(g_cols);
                            down_cols = Some(d_cols);
                        }
                    }
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
                // Every expert weight in the grid, examined once as the tables
                // are built — one slot per (layer, projection), accumulating
                // all `n_experts` of them, so a bad weight names its layer and
                // which of the three projections holds it.
                //
                // Table build runs once at load, outside any wave, which is the
                // only place a quantized assert belongs: it dequantizes through
                // the KO kernels into the staging scratch. Weights do not change
                // between forwards, so checking them here and never again is
                // both sufficient and the only affordable option.
                #[cfg(feature = "tensor-assert")]
                {
                    use candle::tensor_assert::site;
                    if let candle::quantized::QMatMul::QTensor(qt) = slot.gate_proj.inner() {
                        qt.assert(site("expert.gate_proj.L", l));
                    }
                    if let candle::quantized::QMatMul::QTensor(qt) = slot.up_proj.inner() {
                        qt.assert(site("expert.up_proj.L", l));
                    }
                    if let candle::quantized::QMatMul::QTensor(qt) = slot.down_proj.inner() {
                        qt.assert(site("expert.down_proj.L", l));
                    }
                }
                // The REAL allocated extent of each weight, asked of the tensor
                // rather than recomputed from its shape and dtype.
                //
                // Recomputing is what faulted the driver: `ko_repacked_bytes`
                // over a single assumed dtype is wrong the moment a checkpoint
                // varies dtype per layer, which this one does (Q5_KO through
                // layer 33, Q6_KO from 34). More to the point, the recomputed
                // figure is what the GEMM's TILING implies it will read, and the
                // allocation is what actually exists — so keeping both and
                // comparing them is the measurement, not an inconvenience.
                for (v, qmm) in [
                    (&mut gate_bytes, &slot.gate_proj),
                    (&mut up_bytes, &slot.up_proj),
                    (&mut down_bytes, &slot.down_proj),
                ] {
                    match qmm.inner() {
                        candle::quantized::QMatMul::QTensor(qt) => {
                            v.push(qt.storage_size_in_bytes() as i64)
                        }
                        _ => return None,
                    }
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
        // Read the weight asserts HERE, while they are still the only thing in
        // the slot table. The first wave starts a fresh epoch and clears them,
        // so a report deferred past this point would always read clean.
        #[cfg(feature = "tensor-assert")]
        {
            let d = candle::Device::Cuda(device.clone());
            match candle::tensor_assert::drain(&d) {
                Ok(all) => {
                    // Report the element count, not just the verdict. "No bad
                    // sites" reads identically whether every weight was
                    // examined and passed or nothing was examined at all, and
                    // those are opposite facts — a dequant that declined (an
                    // unsupported format, a shape the KO kernel refuses) leaves
                    // a registered slot holding zero elements and would
                    // otherwise print as a clean bill of health.
                    let w: Vec<_> = all
                        .iter()
                        .filter(|f| f.name.starts_with("expert."))
                        .collect();
                    let elems: u64 = w.iter().map(|f| f.elems as u64).sum();
                    let bad = w.iter().filter(|f| f.is_bad()).count();
                    let lo = w.iter().filter_map(|f| f.min).fold(f32::INFINITY, f32::min);
                    let hi = w
                        .iter()
                        .filter_map(|f| f.max)
                        .fold(f32::NEG_INFINITY, f32::max);
                    if bad > 0 {
                        tracing::error!(
                            sites = bad,
                            examined = elems,
                            "expert cache: NON-FINITE EXPERT WEIGHTS — listed above"
                        );
                    } else if elems == 0 {
                        tracing::error!(
                            slots = w.len(),
                            "expert cache: weight asserts examined ZERO elements — the dequant \
                             declined every weight, so this is NOT a clean result"
                        );
                    } else {
                        tracing::info!(
                            slots = w.len(),
                            examined = elems,
                            min = lo,
                            max = hi,
                            "expert cache: every expert weight dequantizes finite"
                        );
                    }
                }
                Err(e) => tracing::warn!(error = %e, "expert cache: weight assert drain failed"),
            }
        }
        // Fingerprint the grid, once, now that it is filled and the tables
        // address it. This is the reference every later suspicion is measured
        // against: the weights are static and fully resident, so any slot whose
        // fingerprint moves afterwards was written to by something that had no
        // business writing there.
        // Diagnostic only. It reads every resident weight through the same
        // pointers the GEMM uses, and a length that disagrees with what was
        // actually allocated faults the driver and takes the process with it —
        // which is not a risk a production forward should carry for a
        // measurement it never reads.
        // Does what the GEMM's tiling implies it will read agree with what was
        // actually allocated?
        //
        // The kernel indexes a weight by chunk arithmetic over
        // `(nrows, ncols, dtype)`; the loader allocated whatever it allocated.
        // Nothing checks that those agree, and if the implied extent is the
        // larger one the GEMM reads past the end of an expert into whichever
        // allocation follows it — which would present exactly as one corrupt
        // slot among clean neighbours, with contents that look like other data.
        //
        // Compared here, once, where both figures exist.
        #[cfg(feature = "tensor-assert")]
        if let (Some(k), Some(dk)) = (cols, down_cols) {
            let mut mismatched = 0usize;
            let mut first: Option<(usize, usize, &'static str, usize, i64)> = None;
            for (name, rows_of, k, dtypes, real) in [
                ("gate", gate_nrows, k, &gate_dtype, &gate_bytes),
                ("up", gate_nrows, k, &gate_dtype, &up_bytes),
                ("down", down_nrows, dk, &down_dtype, &down_bytes),
            ] {
                for (i, &have) in real.iter().enumerate() {
                    let row = i / n_experts;
                    let Some(&dt) = dtypes.get(row) else { continue };
                    let Ok(want) = ko_repacked_bytes(&Shape::from((rows_of, k)), dt) else {
                        continue;
                    };
                    if want as i64 != have {
                        mismatched += 1;
                        first.get_or_insert((row, i % n_experts, name, want, have));
                    }
                }
            }
            if let Some((row, e, name, want, have)) = first {
                tracing::error!(
                    mismatched,
                    total_weights = 3 * total,
                    layer_row = row,
                    expert = e,
                    proj = name,
                    tiling_implies = want,
                    actually_allocated = have,
                    "expert cache: WEIGHT EXTENT MISMATCH — the grouped GEMM's tiling reads a \
                     different number of bytes than the loader allocated. If it reads MORE, it \
                     reads past the end of an expert into whatever follows."
                );
            } else {
                tracing::info!(
                    checked = 3 * total,
                    "expert cache: every weight's allocation matches what the GEMM's tiling reads"
                );
            }
        }
        #[cfg(feature = "tensor-assert")]
        let integrity = Self::fingerprint_grid(
            device,
            &gate_ptrs,
            &up_ptrs,
            &down_ptrs,
            &gate_bytes,
            &up_bytes,
            &down_bytes,
            n_experts,
        );
        #[cfg(not(feature = "tensor-assert"))]
        let integrity: Option<SlotIntegrity> = None;
        let built_capacity = inner.zone.capacity();
        let built_floor = inner.zone.slot_base(built_capacity.saturating_sub(1));
        let built_concede_epoch = inner.geometry.concede_epoch();
        tracing::info!(
            built_capacity,
            built_floor = format!("{built_floor:#x}"),
            built_concede_epoch,
            "expert cache: dispatch tables pinned to this weight-zone geometry — any concession \
             of ground invalidates them"
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
            built_capacity,
            built_floor,
            built_concede_epoch,
            integrity,
            workspace: Mutex::new(workspace),
        })
    }

    /// The zone capacity these tables were captured against.
    pub fn built_capacity(&self) -> usize {
        self.built_capacity
    }

    /// The lowest slot address these tables reference.
    pub fn built_floor(&self) -> u64 {
        self.built_floor
    }

    /// Whether the weight zone has moved out from under these tables.
    ///
    /// The tables hold raw slot addresses captured once. `WeightZone` is
    /// elastic: `retract_to` concedes slots to the KV side under pressure, and
    /// the KV arena then allocates and writes that same ground. A table entry
    /// pointing into it is no longer an expert weight — it is whatever KV put
    /// there — and the GEMM reads it as one.
    ///
    /// Returns the current geometry when it disagrees with what was captured,
    /// so the caller can say what changed rather than only that something did.
    pub fn zone_moved(&self, geometry: &ZoneGeometry) -> Option<(usize, u64)> {
        let now_capacity = geometry.capacity();
        let now_floor = geometry.frontier();
        // **Any concession, ever** — not merely a zone that is smaller *now*.
        //
        // The wave transient tier buys ground when it does not fit
        // (`region_pool::buy_ground`), the weight side concedes slots at the
        // frontier, the tier stands on them and writes activations there, and
        // the zone then grows back. Capacity and floor read exactly as they did
        // at load, while the conceded slots hold tier leftovers and the tables
        // still name them as experts. That is the whole corruption, and it is
        // invisible to the two comparisons below — which are kept because they
        // also catch a zone that is currently smaller, a state the epoch alone
        // would not distinguish from a completed concede-and-regrow.
        if geometry.concede_epoch() != self.built_concede_epoch {
            return Some((now_capacity, now_floor));
        }
        (now_capacity < self.built_capacity || now_floor > self.built_floor)
            .then_some((now_capacity, now_floor))
    }

    /// Fingerprint every resident weight, or `None` if the grid cannot be
    /// described well enough to read safely.
    #[cfg(feature = "tensor-assert")]
    #[allow(clippy::too_many_arguments)]
    fn fingerprint_grid(
        device: &CudaDevice,
        gate_ptrs: &CudaSlice<u64>,
        up_ptrs: &CudaSlice<u64>,
        down_ptrs: &CudaSlice<u64>,
        gate_bytes: &[i64],
        up_bytes: &[i64],
        down_bytes: &[i64],
        n_experts: usize,
    ) -> Option<SlotIntegrity> {
        match SlotIntegrity::capture(
            device, gate_ptrs, up_ptrs, down_ptrs, gate_bytes, up_bytes, down_bytes, n_experts,
        ) {
            Ok(si) => {
                tracing::info!(
                    weights = si.covered(),
                    bytes = si.bytes_covered(),
                    "expert cache: resident weight fingerprints taken — a slot that drifts from \
                     these was written after load"
                );
                // Make the check reachable from every capture site in the
                // forward, not just the ones that happen to hold a handle to
                // the expert cache. A second copy is fingerprinted for the
                // probe so it owns everything it reads.
                match SlotIntegrity::capture(
                    device, gate_ptrs, up_ptrs, down_ptrs, gate_bytes, up_bytes, down_bytes,
                    n_experts,
                ) {
                    Ok(probe) => {
                        // Shared by both registrations: the whole-grid check the
                        // dump runs, and the rotating shard scan every layer
                        // boundary runs. One baseline, so a slot the scan has
                        // already reported and re-baselined does not reappear in
                        // the dump as though it were fresh.
                        let probe = Arc::new(probe);
                        let (grid, shard) = (probe.clone(), probe);
                        let (d1, d2) = (device.clone(), device.clone());
                        crate::models::nan_capture::set_integrity_probe(move || {
                            match grid.verify(&d1) {
                                Ok(v) => v.iter().map(|d| describe(&d1, d)).collect(),
                                Err(e) => vec![format!("integrity re-check failed: {e}")],
                            }
                        });
                        crate::models::nan_capture::set_shard_scan(move |i| {
                            match shard.scan_shard(&d2, i) {
                                // Hand the first drifted slot to the watch:
                                // the scan bounds the write to its sweep
                                // period, the watch bounds it to a layer.
                                Ok(v) => v
                                    .iter()
                                    .map(|d| {
                                        arm_watch(&d2, d, n_experts);
                                        describe(&d2, d)
                                    })
                                    .collect(),
                                Err(e) => vec![format!("shard scan failed: {e}")],
                            }
                        });
                    }
                    Err(e) => tracing::warn!(error = %e, "expert cache: no integrity probe"),
                }
                Some(si)
            }
            Err(e) => {
                tracing::warn!(error = %e, "expert cache: could not fingerprint the grid");
                None
            }
        }
    }

    /// Re-fingerprint the grid and report every slot that has changed since the
    /// fill.
    ///
    /// Deliberately NOT run per wave: the weights are static, so the reference
    /// taken at load is permanently valid and re-deriving it would be bandwidth
    /// spent learning the same number. Call it when something has already gone
    /// wrong — an empty result clears the resident weights and moves the search
    /// on, and a non-empty one names the slot and how it differs.
    #[allow(dead_code)]
    pub fn verify_integrity(&self, device: &CudaDevice) -> Vec<Drifted> {
        let Some(si) = &self.integrity else {
            return Vec::new();
        };
        match si.verify(device) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(error = %e, "expert cache: integrity re-check failed");
                Vec::new()
            }
        }
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
    use std::sync::Arc;

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
            geometry: Arc::new(ZoneGeometry::new(
                n,
                full_zone(n).slot_base(n.saturating_sub(1)),
            )),
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

    #[test]
    fn a_retracted_zone_invalidates_the_tables_but_a_grown_one_does_not() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let mut inner = make_inner(&device, &cuda, 0, 2, 3, None);
        let gd = GpuDispatchTables::build(&inner, &cuda).expect("complete KO grid must build");

        assert_eq!(
            gd.zone_moved(&inner.geometry),
            None,
            "an untouched zone must not invalidate its own tables"
        );

        // Through `grow_zone`/`retract_zone`, not `inner.zone` directly: the
        // check reads the PUBLISHED geometry, so a test that moved the zone
        // behind the publication's back would pass while the production path
        // stayed blind — which is the exact shape of the bug this guards.
        //
        // Growing hands the weight side MORE ground. Every slot the tables
        // reference is still an expert slot, so the addresses stay valid.
        let before = inner.zone.capacity();
        inner.grow_zone(before + 4);
        assert_eq!(
            gd.zone_moved(&inner.geometry),
            None,
            "growth cannot invalidate: the tables' slots are all still theirs"
        );

        // Retracting concedes the slots at the frontier to the KV side. The
        // tables still point at them, and the KV arena will write there — which
        // is the corruption this guard exists to refuse.
        inner.retract_zone(1);
        let moved = gd
            .zone_moved(&inner.geometry)
            .expect("a retraction must invalidate the captured addresses");
        assert!(
            moved.0 < gd.built_capacity(),
            "capacity shrank: {} < {}",
            moved.0,
            gd.built_capacity()
        );
        assert!(
            moved.1 > gd.built_floor(),
            "slots are carved downward, so fewer slots means a HIGHER floor: \
             {:#x} > {:#x}",
            moved.1,
            gd.built_floor()
        );
    }

    #[test]
    fn a_concede_then_regrow_still_invalidates_the_tables() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let mut inner = make_inner(&device, &cuda, 0, 2, 3, None);
        let gd = GpuDispatchTables::build(&inner, &cuda).expect("complete KO grid must build");
        let (cap0, floor0) = (
            inner.zone.capacity(),
            inner.zone.slot_base(inner.zone.capacity() - 1),
        );

        // Concede ground to the KV side, then take it all back. This is what
        // the wave transient tier does when it does not fit: it buys ground,
        // stands on the conceded slots, writes activations over them, and the
        // zone grows back afterwards.
        inner.retract_zone(1);
        inner.grow_zone(cap0);

        // The geometry is restored EXACTLY — which is why comparing it is not
        // enough, and why this test exists.
        assert_eq!(inner.zone.capacity(), cap0, "capacity came back");
        assert_eq!(
            inner.zone.slot_base(inner.zone.capacity() - 1),
            floor0,
            "floor came back"
        );

        assert!(
            gd.zone_moved(&inner.geometry).is_some(),
            "the tables must still be refused: the conceded slots held tier \
             leftovers, not the experts these addresses name"
        );
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
            geometry: Arc::new(ZoneGeometry::new(1, full_zone(1).slot_base(0))),
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
