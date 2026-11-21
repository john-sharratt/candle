//! Runs the matrix product over several experts in one launch.
//!
//! Each expert's contiguous range of token rows is split into tiles and every
//! expert runs in the same kernel, so a projection costs one launch regardless of
//! how many experts it touches. The host-built path packs the weight pointers and
//! tile tables into one blob that the kernel reads in place from a mapped pinned
//! ring, with no per-launch copy. The device-table path reads the tables bucketize
//! wrote, so routing never returns to the host; with a live expert table, worker
//! blocks copy remote experts into scratch and compute them. The int8 arm takes
//! only KO weights with q8a128 activations, and a mismatched pairing is rejected
//! rather than computed.

use super::*;

// GROUPED EXPERT MATMUL (segmented dispatch)

/// Grouped expert matmul: processes multiple experts via segmented dispatch.
///
/// Builds a `VxSegment` array (one per expert) and makes a single call to
/// `run_quantized_matmul`. The C dispatcher loops over segments internally,
/// giving each expert full greedy batch decomposition (TC, iter, bulk, remainder).
/// No device table allocation or memcpy — segment descriptors are host-side.
///
/// # Arguments
/// * `weight_ptrs` - GPU device pointers to each expert's weight data (K/128 format)
/// * `weight_dtype` - Quantization type (same for all experts in this call)
/// * `nrows` - N (output features per expert, same for all)
/// * `ncols` - K (input features per expert, same for all)
/// * `activations` - Stacked activations `[total_batch, K]` on GPU
/// * `act_layout` - Layout of activations tensor
/// * `expert_offsets` - `[num_experts + 1]` prefix sum of per-expert batch counts
///   (CPU slice: `[0, n0, n0+n1, ..., total_batch]`)
/// * `device` - CUDA device
///
/// # Returns
/// A `Tensor` with shape `[total_batch, N]` on the same CUDA device.
pub(super) fn grouped_matmul_gemx_impl<'w>(
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    activations: &CudaStorage,
    act_layout: &crate::Layout,
    expert_offsets: &[i32],
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    use crate::cuda_backend::CudaStorageSlice;

    let num_experts = weight_ptrs.len();
    if num_experts == 0 {
        crate::bail!("grouped_matmul_gemx: no experts provided");
    }
    // The grouped kernel writes full 32-row (N_TILE) blocks with no partial-row
    // guard, so nrows must be a multiple of 32 (all MoE expert dims are).
    if !nrows.is_multiple_of(32) {
        crate::bail!("grouped_matmul_gemx: nrows={nrows} must be a multiple of 32");
    }
    if expert_offsets.len() != num_experts + 1 {
        crate::bail!(
            "grouped_matmul_gemx: expert_offsets.len() = {} but expected {} (num_experts + 1)",
            expert_offsets.len(),
            num_experts + 1
        );
    }
    let total_batch = *expert_offsets.last().unwrap() as usize;
    if total_batch == 0 {
        // No tokens to process — return zero output
        let out_shape: Shape = vec![0, nrows].into();
        let out_slice = unsafe { device.alloc::<f16>(0)? };
        let out_storage = CudaStorage::wrap_cuda_slice(out_slice, device.clone());
        return Ok(crate::tensor::from_storage(
            crate::Storage::Cuda(out_storage),
            out_shape,
            crate::op::BackpropOp::none(),
            false,
        ));
    }

    let k = ncols;
    let (batch_size_from_act, k_act) = match act_layout.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => crate::bail!(
            "grouped_matmul_gemx: unexpected activation shape {:?}",
            act_layout.shape()
        ),
    };
    if k_act != k {
        crate::bail!("grouped_matmul_gemx: K mismatch: weight ncols={k}, activation K={k_act}");
    }
    if batch_size_from_act != total_batch {
        crate::bail!(
            "grouped_matmul_gemx: batch mismatch: activation batch={batch_size_from_act}, offsets total={total_batch}"
        );
    }

    // Convert dtype
    let qtype = dtype_to_qtype(weight_dtype)? as i32;

    // Determine Y type and element size
    let ytype = match &activations.slice {
        CudaStorageSlice::F16(_) => YType::F16,
        CudaStorageSlice::BF16(_) => YType::BF16,
        CudaStorageSlice::F32(_) => YType::F32,
        _ => crate::bail!(
            "grouped_matmul_gemx: unsupported activation dtype {:?}",
            activations.dtype()
        ),
    };

    // The output inherits the activations' arena and matches their dtype (the FP
    // grouped kernels store at the activation width). Only the pool arm owns a
    // slice; the wave arm stays a bare pointer until the wrap, so a `?` between
    // here and there cannot drop a `CudaSlice` over arena memory.
    let out_elems = nrows * total_batch;
    let (dst_ptr, owned_out, out_backing) =
        resolve_out(activations.dtype(), activations.backing, device, out_elems)?;

    // Grouped dispatch: split each expert's [offset..offset+cnt) batch range into
    // <=16-token MMA tiles, then run ALL experts in ONE kernel launch. The kernel
    // grid (num_tiles x row_tiles) spans every expert at once, so both the launch
    // count (N -> 1 per projection) and per-expert occupancy are fixed together.
    let mut tile_expert: Vec<i32> = Vec::new();
    let mut tile_b_start: Vec<i32> = Vec::new();
    let mut tile_b_cnt: Vec<i32> = Vec::new();
    for e in 0..num_experts {
        let mut s = expert_offsets[e];
        let end = expert_offsets[e + 1];
        while s < end {
            let cnt = (end - s).min(16);
            tile_expert.push(e as i32);
            tile_b_start.push(s);
            tile_b_cnt.push(cnt);
            s += cnt;
        }
    }
    let num_tiles = tile_expert.len();
    // Token tiles ride grid.y (row tiles are the fast axis for L2 reuse).
    if num_tiles > 65535 {
        crate::bail!("grouped_matmul_gemx: num_tiles={num_tiles} exceeds grid.y max 65535");
    }

    {
        let stream = device.cuda_stream();

        // Pack the weight-pointer array + all 3 i32 tile tables into ONE blob.
        // weight_ptrs (u64) go first (8-aligned at the 16-aligned ring slot);
        // the three i32 tables follow at 4-aligned offsets. The kernel reads
        // each table via base + offset, IN PLACE from the device-mapped pinned
        // table ring — no per-launch H2D copy, no device allocation.
        let off_te = num_experts * 8;
        let off_tbs = off_te + num_tiles * 4;
        let off_tbc = off_tbs + num_tiles * 4;
        let total_bytes = off_tbc + num_tiles * 4;
        let mut packed: Vec<u8> = Vec::with_capacity(total_bytes);
        for &w in weight_ptrs {
            packed.extend_from_slice(&w.to_le_bytes());
        }
        for &x in &tile_expert {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_start {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_cnt {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        let ring = table_ring(device)?;
        let y_elem = match ytype {
            YType::F32 => 4usize,
            _ => 2, // F16 / BF16
        };
        // The FP grouped kernels store at the activation's width.
        let row_fast = grouped_grid_row_fast(
            total_batch * k * y_elem,
            total_batch * GROUPED_FP_ROW_TILE * y_elem,
            device,
        ) as i32;

        macro_rules! dispatch_grouped {
            ($y_data:expr) => {{
                let y_view = match act_layout.contiguous_offsets() {
                    Some((o1, o2)) => $y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "grouped_matmul_gemx",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                let status = ring.with_table(&packed, |base| {
                    let wptr_ptr = base;
                    let te_ptr = base + off_te as u64;
                    let tbs_ptr = base + off_tbs as u64;
                    let tbc_ptr = base + off_tbc as u64;
                    unsafe {
                        run_grouped_quantized_matmul(
                            wptr_ptr as *const std::ffi::c_void,
                            te_ptr as *const std::ffi::c_void,
                            tbs_ptr as *const std::ffi::c_void,
                            tbc_ptr as *const std::ffi::c_void,
                            y_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            k as i32,     // ncols_x = K
                            nrows as i32, // nrows_x = N
                            k as i32,     // y_stride = K (stacked activations)
                            nrows as i32, // dst_stride = N (stacked output)
                            num_tiles as i32,
                            qtype,
                            ytype as i32,
                            2, // FP grouped kernels ignore the int8 tile mode
                            row_fast,
                            // sum_norm: the FP grouped kernels do not take it —
                            // this is the float activation path, which has no
                            // q8a128 header. The launcher drops it for ytype != 3.
                            SumScale::Raw.as_code(),
                            // Host-built tables over resident weights: no wait.
                            std::ptr::null(),
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                })?;
                check_matmul_status(status, "grouped_matmul_gemx")?;
            }};
        }

        match &activations.slice {
            CudaStorageSlice::F16(y_data) => {
                dispatch_grouped!(y_data);
            }
            CudaStorageSlice::BF16(y_data) => {
                dispatch_grouped!(y_data);
            }
            CudaStorageSlice::F32(y_data) => {
                dispatch_grouped!(y_data);
            }
            _ => unreachable!("ytype and activation slice should match"),
        }
    }

    let out_shape: Shape = vec![total_batch, nrows].into();
    tensor_from_owned_out(owned_out, dst_ptr, out_backing, out_shape, device)
}

/// Single-launch grouped MoE expert matmul over FP activations (F16/BF16/F32):
/// FP16 tensor-core MMA with Q4_K weights dequantized on the fly. The INT8-MMA path
/// is `grouped_matmul_gemx_q8a128` (q8a128 activations). See docs/archived/q8_matmul_pipeline.md.
pub fn grouped_matmul_gemx(
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    activations: &CudaStorage,
    act_layout: &crate::Layout,
    expert_offsets: &[i32],
    device: &CudaDevice,
) -> Result<crate::Tensor> {
    grouped_matmul_gemx_impl(
        weight_ptrs,
        weight_dtype,
        nrows,
        ncols,
        activations,
        act_layout,
        expert_offsets,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_matmul_gemx_q8a128<'w>(
    act_ptr: u64,
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    total_batch: usize,
    expert_offsets: &[i32],
    device: &CudaDevice,
    origin: Backing,
    sum_scale: SumScale,
) -> Result<crate::LiveTensor<'w>> {
    let num_experts = weight_ptrs.len();
    if num_experts == 0 {
        crate::bail!("grouped_matmul_gemx_q8a128: no experts provided");
    }
    let active: usize = (0..num_experts)
        .filter(|&e| expert_offsets[e + 1] > expert_offsets[e])
        .count();
    let avg_rows = total_batch.checked_div(active).unwrap_or(0);
    let n_sub = grouped_int8_n_sub(avg_rows, &[weight_dtype]);
    let row_fast = grouped_int8_row_fast(total_batch, ncols, nrows, n_sub, device);
    grouped_matmul_gemx_q8a128_with_mode(
        act_ptr,
        weight_ptrs,
        weight_dtype,
        nrows,
        ncols,
        total_batch,
        expert_offsets,
        device,
        origin,
        n_sub,
        row_fast,
        sum_scale,
    )
}

/// [`grouped_matmul_gemx_q8a128`] with the token-tile mode AND grid axis order
/// chosen by the caller — the test seam that proves the wide modes and both
/// grid orders bit-equal mode-2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped_matmul_gemx_q8a128_with_mode<'w>(
    act_ptr: u64,
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    total_batch: usize,
    expert_offsets: &[i32],
    device: &CudaDevice,
    origin: Backing,
    n_sub: usize,
    row_fast: bool,
    // The activation operand's Σx convention. Threaded rather than read off an
    // operand because this takes a raw `act_ptr` — the caller holding the
    // `Q8a128Operand` is the only place that knows.
    sum_scale: SumScale,
) -> Result<crate::LiveTensor<'w>> {
    let num_experts = weight_ptrs.len();
    if num_experts == 0 {
        crate::bail!("grouped_matmul_gemx_q8a128: no experts provided");
    }
    if !nrows.is_multiple_of(32) {
        crate::bail!("grouped_matmul_gemx_q8a128: nrows={nrows} must be a multiple of 32");
    }
    if !ncols.is_multiple_of(128) {
        crate::bail!("grouped_matmul_gemx_q8a128: ncols={ncols} must be a multiple of 128");
    }
    if !matches!(n_sub, 2 | 4 | 8) || (n_sub > 2 && !weight_dtype.is_ko()) {
        crate::bail!(
            "grouped_matmul_gemx_q8a128: tile mode n_sub={n_sub} unsupported for {weight_dtype:?}"
        );
    }
    let qtype = dtype_to_qtype(weight_dtype)? as i32;

    let tile_w = 16 * n_sub;
    let mut tile_expert: Vec<i32> = Vec::new();
    let mut tile_b_start: Vec<i32> = Vec::new();
    let mut tile_b_cnt: Vec<i32> = Vec::new();
    for e in 0..num_experts {
        let mut s = expert_offsets[e];
        let end = expert_offsets[e + 1];
        while s < end {
            let cnt = (end - s).min(tile_w as i32);
            tile_expert.push(e as i32);
            tile_b_start.push(s);
            tile_b_cnt.push(cnt);
            s += cnt;
        }
    }
    let num_tiles = tile_expert.len();
    // Token tiles ride grid.y (row tiles are the fast axis for L2 reuse).
    if num_tiles > 65535 {
        crate::bail!("grouped_matmul_gemx_q8a128: num_tiles={num_tiles} exceeds grid.y max 65535");
    }

    let (dst_ptr, owned_dst, out_backing) =
        resolve_typed_out::<f32>(origin, device, nrows * total_batch)?;

    {
        let off_te = num_experts * 8;
        let off_tbs = off_te + num_tiles * 4;
        let off_tbc = off_tbs + num_tiles * 4;
        let total_bytes = off_tbc + num_tiles * 4;
        let mut packed: Vec<u8> = Vec::with_capacity(total_bytes);
        for &w in weight_ptrs {
            packed.extend_from_slice(&w.to_le_bytes());
        }
        for &x in &tile_expert {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_start {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_cnt {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        // The kernel reads the descriptor blob IN PLACE from the device-mapped
        // pinned table ring — no per-launch H2D copy, no device allocation.
        let status = table_ring(device)?.with_table(&packed, |base| {
            let wptr_ptr = base;
            let te_ptr = base + off_te as u64;
            let tbs_ptr = base + off_tbs as u64;
            let tbc_ptr = base + off_tbc as u64;
            unsafe {
                crate::set_kernel_breadcrumb("run_grouped_quantized_matmul", file!(), line!());
                run_grouped_quantized_matmul(
                    wptr_ptr as *const std::ffi::c_void,
                    te_ptr as *const std::ffi::c_void,
                    tbs_ptr as *const std::ffi::c_void,
                    tbc_ptr as *const std::ffi::c_void,
                    act_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32, // ncols_x = K
                    nrows as i32, // nrows_x = N
                    ncols as i32, // y_stride (unused by int8 kernel; ABI)
                    nrows as i32, // dst_stride = N
                    num_tiles as i32,
                    qtype,
                    YType::Q8A128 as i32,
                    n_sub as i32,
                    row_fast as i32,
                    sum_scale.as_code(),
                    // Host-built tables over resident weights: no wait.
                    std::ptr::null(),
                    device.cuda_stream().cu_stream() as *mut std::ffi::c_void,
                )
            }
        })?;
        check_matmul_status(status, "grouped_matmul_gemx_q8a128")?;
    }

    let out_shape: Shape = vec![total_batch, nrows].into();
    tensor_from_out::<f32>(owned_dst, dst_ptr, out_backing, out_shape, device)
}

/// Grouped (MoE) quantized matmul over a [`DynamicTensor`] activation, dispatched by
/// numeric mode: `Int8` runs the q8a128 tensor-core grouped path (mode-1 only — the grouped
/// kernel has no mode-2, so an operand carrying `Q8A128X` from M≥64 still runs mode-1; the
/// operand's `ytype` is ignored here), `Float` runs the dequant-weight float grouped path.
/// M and K come from the activation (operand shape / tensor shape); `nrows` is N per
/// expert, `expert_offsets` partitions the M token rows across `weight_ptrs`.
/// KO weights and int8 (q8a128) activations are EXCLUSIVELY paired: the int8 tensor-core
/// kernels' per-128 deferred-scale fold reads only KO weights (the per-128 grid), and KO weights
/// are only consumed through the int8 path. Plain k-quants (Q2_K/Q4_K/…) have finer per-sub
/// scales the per-128 int8 fold does not apply — they must be re-quantized to a KO twin (`to_ko`)
/// to run int8; through the FP grouped path they stay non-KO. So `DynamicTensor::Int8` must pair
/// with a KO weight and `DynamicTensor::Float` with a non-KO weight — any cross combination has
/// no kernel and is rejected here rather than silently producing garbage.
pub(super) fn ensure_qmatmul_pairing(input: &DynamicTensor, weight_dtype: GgmlDType) -> Result<()> {
    let is_int8 = matches!(input, DynamicTensor::Int8(_));
    if weight_dtype.is_ko() != is_int8 {
        crate::bail!(
            "qmatmul: KO weights and int8 q8a128 activations are exclusively paired — got \
             weight {weight_dtype:?} (KO={}) with {} activations. Pair Int8 activations with \
             KO weights, and Float activations with non-KO weights.",
            weight_dtype.is_ko(),
            if is_int8 { "Int8" } else { "Float" },
        );
    }
    Ok(())
}

pub fn grouped_qmatmul<'w>(
    input: DynamicTensor,
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    expert_offsets: &[i32],
    device: &CudaDevice,
    origin: Backing,
) -> Result<crate::LiveTensor<'w>> {
    ensure_qmatmul_pairing(&input, weight_dtype)?;
    match input {
        DynamicTensor::Int8(op) => op.with_device_ptr(device, |act_ptr| {
            grouped_matmul_gemx_q8a128(
                act_ptr,
                weight_ptrs,
                weight_dtype,
                nrows,
                op.cols,
                op.rows,
                expert_offsets,
                device,
                origin,
                op.sum_scale,
            )
        }),
        DynamicTensor::Float(t) => {
            let (storage, layout) = t.storage_and_layout();
            let cuda = match &*storage {
                crate::Storage::Cuda(c) => c,
                _ => crate::bail!("grouped_qmatmul: Float input must be a CUDA tensor"),
            };
            let ncols = *layout.shape().dims().last().unwrap();
            // The FP grouped path writes through `&mut CudaSlice`, so it owns
            // its output rather than taking a planned slot. An owned tensor
            // coerces into any `'w` — it outlives every wave — so the caller
            // sees the same type either way. This is the non-int8 path: the
            // expert GEMMs of an F16/BF16 config still allocate, and the
            // detector reports them as such.
            let _ = origin;
            grouped_matmul_gemx_impl(
                weight_ptrs,
                weight_dtype,
                nrows,
                ncols,
                cuda,
                layout,
                expert_offsets,
                device,
            )
        }
    }
}

/// Grouped (MoE) q8a128 quantized matmul with **device-resident dispatch tables**:
/// the GPU-native twin of [`grouped_qmatmul`]'s int8 arm. Where the host path
/// packs per-active-expert weight pointers + tile tables on the CPU (requiring
/// the routing indices to round-trip GPU→CPU first), this variant reads
/// everything from device memory:
///  * `weight_ptrs_dev` — the expert pointer table (u64 rows; without `live`
///    every entry is a VRAM address, with it the table is bucketize's
///    per-invocation snapshot, below); `expert_base` is the element
///    offset of THIS layer's `[n_experts]` row block inside it, so
///    `tile_expert` carries RAW expert ids straight from `moe_bucketize`;
///  * `tile_expert` / `tile_b_start` / `tile_b_cnt` — `moe_bucketize`'s device
///    tile tables, launched at the `launch_tiles` upper bound; padding tiles
///    (`b_cnt == 0`) exit in the kernel without touching the pointer table.
///
/// Output is `[total_batch, nrows]` f32 with padding rows UNWRITTEN — the
/// deterministic scatter's segment tables never reference them. Every output
/// row has exactly one writing block and a fixed K order, so the result does
/// not depend on the order of the tiles in the tables.
///
/// **A live table.** With `live` set, the table is `moe_bucketize`'s snapshot of
/// the layer's live-table entries, and bucketize has put the remote (non-VRAM)
/// experts' tiles first and listed them: the launch gains a row of
/// `live.workers` worker blocks that copy each remote expert's row tiles into
/// VRAM scratch and compute them — a cold one once `live.live_row` names it —
/// while the other tiles run as from any resident table (see "A live expert
/// table" in `kernel.cuh`).
#[allow(clippy::too_many_arguments)]
pub fn grouped_qmatmul_dev_q8a128<'w>(
    op: &Q8a128Operand<'w>,
    weight_ptrs_dev: &CudaSlice<u64>,
    expert_base: usize,
    n_experts: usize,
    weight_dtype: GgmlDType,
    nrows: usize,
    tile_expert: &CudaSlice<i32>,
    tile_b_start: &CudaSlice<i32>,
    tile_b_cnt: &CudaSlice<i32>,
    launch_tiles: usize,
    n_sub: usize,
    live: Option<&MoeLive>,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    ensure_qmatmul_pairing(&DynamicTensor::Int8(op), weight_dtype)?;
    // The tile tables were built `16·n_sub` wide; a launch at another width
    // mis-strides every batch slice. Wide modes exist for the KO rows only.
    if !matches!(n_sub, 2 | 4 | 8) || (n_sub > 2 && !weight_dtype.is_ko()) {
        crate::bail!(
            "grouped_qmatmul_dev_q8a128: tile mode n_sub={n_sub} unsupported for {weight_dtype:?}"
        );
    }
    if !nrows.is_multiple_of(32) {
        crate::bail!("grouped_qmatmul_dev_q8a128: nrows={nrows} must be a multiple of 32");
    }
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "grouped_qmatmul_dev_q8a128: ncols={} must be a multiple of 128",
            op.cols
        );
    }
    if launch_tiles == 0 {
        crate::bail!("grouped_qmatmul_dev_q8a128: launch_tiles must be > 0");
    }
    // Token tiles ride grid.y (row tiles are the fast axis for L2 reuse).
    if launch_tiles > 65535 {
        crate::bail!(
            "grouped_qmatmul_dev_q8a128: launch_tiles={launch_tiles} exceeds grid.y max 65535"
        );
    }
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    let total_batch = op.rows;
    let ncols = op.cols;

    // `tile_expert` values are bounded by the ROUTER's expert count (the
    // bucketize input), so the whole `[n_experts]` row this layer indexes must
    // lie inside the table — checking only the base would let a router/table
    // width mismatch dereference past the end (or into the next layer's row).
    if n_experts == 0
        || expert_base
            .checked_add(n_experts)
            .is_none_or(|end| end > weight_ptrs_dev.len())
    {
        crate::bail!(
            "grouped_qmatmul_dev_q8a128: expert row [{expert_base}, {expert_base}+{n_experts}) \
             exceeds table len {}",
            weight_ptrs_dev.len()
        );
    }
    // Inherits the gathered activation's arena, so the whole expert chain —
    // gate, up, SwiGLU, down — stays in the FFN generation the gather seeded.
    let (dst_ptr, owned_dst, out_backing) =
        resolve_typed_out::<f32>(op.backing(), device, nrows * total_batch)?;
    {
        let stream = device.cuda_stream();
        let (wp_base, _g0) = weight_ptrs_dev.device_ptr(&stream);
        // Offset to this layer's `[n_experts]` row inside the flat table.
        let wp = wp_base + (expert_base * std::mem::size_of::<u64>()) as u64;
        let (te, _g1) = tile_expert.device_ptr(&stream);
        let (tbs, _g2) = tile_b_start.device_ptr(&stream);
        let (tbc, _g3) = tile_b_cnt.device_ptr(&stream);
        // A live launch has its grid fixed by the launcher (row tiles on x, the
        // workers' rows ahead of the tiles); any other launch picks its axis order
        // for L2.
        let row_fast = grouped_int8_row_fast(total_batch, ncols, nrows, n_sub, device) as i32;
        let live_ptr = live.map_or(std::ptr::null(), |l| l as *const MoeLive);
        op.with_device_ptr(device, |act_ptr| {
            let status = unsafe {
                run_grouped_quantized_matmul(
                    wp as *const std::ffi::c_void,
                    te as *const std::ffi::c_void,
                    tbs as *const std::ffi::c_void,
                    tbc as *const std::ffi::c_void,
                    act_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32, // ncols_x = K
                    nrows as i32, // nrows_x = N
                    ncols as i32, // y_stride (unused by int8 kernel; ABI)
                    nrows as i32, // dst_stride = N
                    launch_tiles as i32,
                    qtype,
                    YType::Q8A128 as i32,
                    n_sub as i32,
                    row_fast,
                    // Read off the operand, so the matmul interprets the header
                    // the way whoever quantized these blocks wrote it.
                    op.sum_scale.as_code(),
                    live_ptr,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
            check_matmul_status(status, "grouped_matmul_gemx_dynamic")
        })?;
    }

    let out_shape: Shape = vec![total_batch, nrows].into();
    tensor_from_out::<f32>(owned_dst, dst_ptr, out_backing, out_shape, device)
}
