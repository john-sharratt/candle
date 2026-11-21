//! Calls the GPU kernels that choose each KV block's storage format, and the
//! helpers that keep their numbering in step with the CUDA side.
//!
//! For each chunk and head, the selection kernel reads the candidate error
//! surfaces and returns the format tag each block takes. The wrappers stage the
//! head-GID table in the layout the kernel reads and keep only the outputs a
//! caller uses, so a test-oriented call discards the palette diagnostics. Format
//! codes are shared with the CUDA headers, and bits per element are derived from
//! the Rust block structs and checked against the kernel's own table, so a block
//! size change fails a test on both sides. Candidates are sorted by that cost
//! before they are uploaded.

use super::*;

/// Thin unsafe wrapper around `run_select_kv_format_palette4_paged`.
///
/// # Safety
/// All device pointers must be valid. `head_gids_ptr` must have
/// `total_heads * 8` i64 entries (K/V GIDs for each of the 4 palette bands,
/// in HeadGids order: `head * 8 + palette * 2 + is_v`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn select_kv_format_palette4_paged(
    per_head_table_ptr: u64,
    head_gids_ptr: u64,
    q_relevance_median_ptr: u64,
    q_relevance_spread_ptr: u64,
    k_head_amax_ptr: u64,
    v_head_amax_ptr: u64,
    k_head_p95_ptr: u64,
    v_head_p95_ptr: u64,
    k_cand_ptr: u64,
    v_cand_ptr: u64,
    n_k_cand: i32,
    n_v_cand: i32,
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
    total_heads: i32,
    blocks_per_head: i32,
    n_kv_head: i32,
    arena_chunks: i32,
    valid_ranges_ptr: u64,
    k_palette_tags_ptr: u64,
    v_palette_tags_ptr: u64,
    k_palette_scale_ptr: u64,
    v_palette_scale_ptr: u64,
    k_palette_map_ptr: u64,
    v_palette_map_ptr: u64,
    k_effective_block_tags_ptr: u64,
    v_effective_block_tags_ptr: u64,
    k_head_tags_ptr: u64,
    v_head_tags_ptr: u64,
    q_relevance_out_ptr: u64,
    stream: *mut std::ffi::c_void,
) {
    crate::set_kernel_breadcrumb("run_select_kv_format_palette4_paged", file!(), line!());
    run_select_kv_format_palette4_paged(
        per_head_table_ptr as *const i64,
        head_gids_ptr as *const i64,
        q_relevance_median_ptr as *mut f32,
        q_relevance_spread_ptr as *mut f32,
        k_head_amax_ptr as *mut f32,
        v_head_amax_ptr as *mut f32,
        k_head_p95_ptr as *mut f32,
        v_head_p95_ptr as *mut f32,
        k_cand_ptr as *const i32,
        v_cand_ptr as *const i32,
        n_k_cand,
        n_v_cand,
        k_threshold_hi,
        k_threshold_lo,
        v_threshold_hi,
        v_threshold_lo,
        total_heads,
        blocks_per_head,
        n_kv_head,
        arena_chunks,
        valid_ranges_ptr as *const i32,
        k_palette_tags_ptr as *mut i32,
        v_palette_tags_ptr as *mut i32,
        k_palette_scale_ptr as *mut f32,
        v_palette_scale_ptr as *mut f32,
        k_palette_map_ptr as *mut i32,
        v_palette_map_ptr as *mut i32,
        k_effective_block_tags_ptr as *mut i32,
        v_effective_block_tags_ptr as *mut i32,
        k_head_tags_ptr as *mut i32,
        v_head_tags_ptr as *mut i32,
        q_relevance_out_ptr as *mut f32,
        stream,
    );
}

/// Gids staged per (chunk, head) for the selection kernels: K and V for each
/// of the 4 palette bands, in HeadGids order (`head * 8 + palette * 2 + is_v`).
/// Must match `GIDS_PER_HEAD` in candle-nn's `head_gids.rs` and `arena_table.cuh`.
pub(super) const SELECT_GIDS_PER_HEAD: usize = 8;

/// Paged format selection — runs the fused per-head palette-4 selection kernel
/// and returns the per-block effective format tags. Used by tests.
///
/// Wraps `select_kv_format_palette4_paged_batched_raw_from_device_ptrs` and keeps
/// only the `k_eff_block_tags` / `v_eff_block_tags` outputs (per-block tags after
/// palette slot expansion); the per-slot palette tags, scales, pal_maps, and
/// other diagnostic outputs are discarded.
pub fn select_kv_format_paged_batched_raw(
    per_head_table_gpu: &CudaSlice<i64>,
    head_gids: &[i64],
    k_candidates: &[GgmlDType],
    v_candidates: &[GgmlDType],
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
    blocks_per_head: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<i32>, CudaSlice<i32>)> {
    let n_chunks = head_gids.len() / (n_kv_head * SELECT_GIDS_PER_HEAD);
    let total_heads = n_chunks * n_kv_head;
    let total_blocks = total_heads * blocks_per_head;

    if total_blocks == 0 {
        let empty = dev.alloc_zeros::<i32>(0)?;
        return Ok((empty, dev.alloc_zeros::<i32>(0)?));
    }

    let gids_gpu = dev.memcpy_stod(head_gids)?;
    // Test-oriented wrapper: full chunks only (offset 0, all 32 tokens).
    let full_ranges = vec![32i32; n_chunks];

    let selected = unsafe {
        let stream = dev.cuda_stream();
        let (pht_ptr, _pht_guard) = per_head_table_gpu.device_ptr(&stream);
        let (gids_ptr, _gids_guard) = gids_gpu.device_ptr(&stream);
        select_palette4_owned(
            pht_ptr,
            gids_ptr,
            n_chunks,
            k_candidates,
            v_candidates,
            [
                k_threshold_hi,
                k_threshold_lo,
                v_threshold_hi,
                v_threshold_lo,
            ],
            blocks_per_head,
            n_kv_head,
            arena_chunks,
            &full_ranges,
            dev,
        )?
    };

    Ok((
        dev.memcpy_stod(&selected.k_eff_block_tags)?,
        dev.memcpy_stod(&selected.v_eff_block_tags)?,
    ))
}

#[allow(clippy::type_complexity)]
pub fn reduce_head_format_stats(
    k_dev_ptr: u64,
    v_dev_ptr: u64,
    blocks_per_head: usize,
    n_kv_head: usize,
    num_chunks: usize,
    dev: &CudaDevice,
) -> Result<(
    CudaSlice<i32>,
    CudaSlice<i32>,
    CudaSlice<i32>,
    CudaSlice<i32>,
)> {
    let total_heads = num_chunks * n_kv_head;
    let total_blocks = total_heads * blocks_per_head;
    let mut k_head_out = dev.alloc_zeros::<i32>(total_heads)?;
    let mut v_head_out = dev.alloc_zeros::<i32>(total_heads)?;
    let mut k_eff_out = dev.alloc_zeros::<i32>(total_blocks)?;
    let mut v_eff_out = dev.alloc_zeros::<i32>(total_blocks)?;

    {
        let stream = dev.cuda_stream();
        let k_blk_ptr = k_dev_ptr;
        let v_blk_ptr = v_dev_ptr;
        let (k_head_ptr, _kh_guard) = k_head_out.device_ptr_mut(&stream);
        let (v_head_ptr, _vh_guard) = v_head_out.device_ptr_mut(&stream);
        let (k_eff_ptr, _ke_guard) = k_eff_out.device_ptr_mut(&stream);
        let (v_eff_ptr, _ve_guard) = v_eff_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb("run_reduce_head_stats_format", file!(), line!());
            run_reduce_head_stats_format(
                k_blk_ptr as *const i32,
                v_blk_ptr as *const i32,
                k_head_ptr as *mut i32,
                v_head_ptr as *mut i32,
                k_eff_ptr as *mut i32,
                v_eff_ptr as *mut i32,
                blocks_per_head as i32,
                n_kv_head as i32,
                num_chunks as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    Ok((k_head_out, v_head_out, k_eff_out, v_eff_out))
}

/// Per-head format selection: returns most conservative per-(chunk, head) format.
/// Uses the fused palette4 kernel; slot 0 = highest-amax blocks = most conservative format.
#[allow(clippy::too_many_arguments)]
pub fn select_kv_format_paged_per_head(
    per_head_table_gpu: &CudaSlice<i64>,
    head_gids: &[i64],
    k_candidates: &[GgmlDType],
    v_candidates: &[GgmlDType],
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
    blocks_per_head: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<i32>, CudaSlice<i32>)> {
    let n_chunks = head_gids.len() / (n_kv_head * SELECT_GIDS_PER_HEAD);
    let total_heads = n_chunks * n_kv_head;

    if total_heads == 0 || n_kv_head == 0 {
        let empty = dev.alloc_zeros::<i32>(0)?;
        return Ok((empty, dev.alloc_zeros::<i32>(0)?));
    }

    let gids_gpu = dev.memcpy_stod(head_gids)?;
    // Test-oriented wrapper: full chunks only (offset 0, all 32 tokens).
    let full_ranges = vec![32i32; n_chunks];

    let selected = unsafe {
        let stream = dev.cuda_stream();
        let (pht_ptr, _pht_guard) = per_head_table_gpu.device_ptr(&stream);
        let (gids_ptr, _gids_guard) = gids_gpu.device_ptr(&stream);
        select_palette4_owned(
            pht_ptr,
            gids_ptr,
            n_chunks,
            k_candidates,
            v_candidates,
            [
                k_threshold_hi,
                k_threshold_lo,
                v_threshold_hi,
                v_threshold_lo,
            ],
            blocks_per_head,
            n_kv_head,
            arena_chunks,
            &full_ranges,
            dev,
        )?
    };

    Ok((
        dev.memcpy_stod(&selected.k_head_tags)?,
        dev.memcpy_stod(&selected.v_head_tags)?,
    ))
}

/// F16 code from CUDA `select_kv_format.cuh` (`#define SELECT_FMT_F16 1`).
pub const SELECT_FMT_F16: i32 = 1;

/// BF16 code from CUDA `select_kv_format.cuh` (`#define SELECT_FMT_BF16 2`).
pub const SELECT_FMT_BF16: i32 = 2;

pub fn ggml_to_select_qtype(dtype: GgmlDType) -> Result<i32> {
    match dtype {
        GgmlDType::Q4_0 => Ok(QType::Q4_0 as i32),
        GgmlDType::Q4_1 => Ok(QType::Q4_1 as i32),
        GgmlDType::Q5_0 => Ok(QType::Q5_0 as i32),
        GgmlDType::Q5_1 => Ok(QType::Q5_1 as i32),
        GgmlDType::Q8_0 => Ok(QType::Q8_0 as i32),
        GgmlDType::Q8_1 => Ok(QType::Q8_1 as i32),
        GgmlDType::Q4_KS => Ok(QType::Q4_KS as i32),
        GgmlDType::Q8_KS => Ok(QType::Q8_KS as i32),
        GgmlDType::Q2_0 => Ok(QType::Q2_0 as i32),
        GgmlDType::Q3_0 => Ok(QType::Q3_0 as i32),
        GgmlDType::R16 => Ok(QType::R16 as i32),
        GgmlDType::Q0 => Ok(QType::Q0 as i32),
        GgmlDType::Q1_S => Ok(QType::Q1_S as i32),
        GgmlDType::Q2_S => Ok(QType::Q2_S as i32),
        GgmlDType::Q2_A => Ok(QType::Q2_A as i32),
        GgmlDType::Q2_1 => Ok(QType::Q2_1 as i32),
        GgmlDType::Q3_1 => Ok(QType::Q3_1 as i32),
        GgmlDType::Q0_V => Ok(QType::Q0_V as i32),
        GgmlDType::Q1_A => Ok(QType::Q1_A as i32),
        GgmlDType::Q0_X => Ok(QType::Q0_X as i32),
        GgmlDType::Q0_M2 => Ok(QType::Q0_M2 as i32),
        GgmlDType::Q0_M4 => Ok(QType::Q0_M4 as i32),
        GgmlDType::F16 => Ok(SELECT_FMT_F16),
        GgmlDType::BF16 => Ok(SELECT_FMT_BF16),
        _ => crate::bail!("select_kv_format: unsupported format {:?}", dtype),
    }
}

/// Returns 4× the bits-per-element for a QType code (matches CUDA `format_bpe_x4`).
/// Used to sort candidates ascending by BPE before uploading to the GPU kernel.
///
/// Derives from `GgmlDType::type_size() * 32 / GgmlDType::block_size()` which is
/// `bits_per_elem × 4` (×4 gives exact integers for fractional-bpe formats).
/// This chains through `size_of::<BlockX>()`, which is itself locked to the CUDA
/// `block_x` struct via the `static_assert(sizeof(block_x) == N)` in
/// `candle-kernels/src/blocks.cuh`. So Rust block struct → CUDA struct →
/// kernel's `format_bpe_x4` all stay in sync automatically.
///
/// Unknown codes (or formats without a block struct that's a kernel candidate)
/// return 256 — the worst-case sentinel matching CUDA's `format_bpe_x4` default.
pub fn select_qtype_bpe_x4(code: i32) -> i32 {
    match select_qtype_to_ggml(code) {
        Ok(d) => {
            let ts = d.type_size() as i32;
            let bs = d.block_size() as i32;
            // bpe_x4 = (bytes × 8 / block_size) × 4 = bytes × 32 / block_size.
            // For 32-element blocks this collapses to `bytes`, which matches
            // the kernel's "bpe_x4 ≡ block_bytes for 32-element blocks" identity.
            ts * 32 / bs
        }
        Err(_) => 256, // unknown → worst, matches CUDA default
    }
}

/// Convert a QType integer code (from GPU output) back to GgmlDType.
pub fn select_qtype_to_ggml(code: i32) -> Result<GgmlDType> {
    match code {
        c if c == QType::Q4_0 as i32 => Ok(GgmlDType::Q4_0),
        c if c == QType::Q4_1 as i32 => Ok(GgmlDType::Q4_1),
        c if c == QType::Q5_0 as i32 => Ok(GgmlDType::Q5_0),
        c if c == QType::Q5_1 as i32 => Ok(GgmlDType::Q5_1),
        c if c == QType::Q8_0 as i32 => Ok(GgmlDType::Q8_0),
        c if c == QType::Q8_1 as i32 => Ok(GgmlDType::Q8_1),
        c if c == QType::Q4_KS as i32 => Ok(GgmlDType::Q4_KS),
        c if c == QType::Q8_KS as i32 => Ok(GgmlDType::Q8_KS),
        c if c == QType::Q2_0 as i32 => Ok(GgmlDType::Q2_0),
        c if c == QType::Q3_0 as i32 => Ok(GgmlDType::Q3_0),
        c if c == QType::R16 as i32 => Ok(GgmlDType::R16),
        c if c == QType::Q0 as i32 => Ok(GgmlDType::Q0),
        c if c == QType::Q1_S as i32 => Ok(GgmlDType::Q1_S),
        c if c == QType::Q2_S as i32 => Ok(GgmlDType::Q2_S),
        c if c == QType::Q2_A as i32 => Ok(GgmlDType::Q2_A),
        c if c == QType::Q2_1 as i32 => Ok(GgmlDType::Q2_1),
        c if c == QType::Q3_1 as i32 => Ok(GgmlDType::Q3_1),
        c if c == QType::Q0_V as i32 => Ok(GgmlDType::Q0_V),
        c if c == QType::Q1_A as i32 => Ok(GgmlDType::Q1_A),
        c if c == QType::Q0_X as i32 => Ok(GgmlDType::Q0_X),
        c if c == QType::Q0_M2 as i32 => Ok(GgmlDType::Q0_M2),
        c if c == QType::Q0_M4 as i32 => Ok(GgmlDType::Q0_M4),
        SELECT_FMT_F16 => Ok(GgmlDType::F16),
        SELECT_FMT_BF16 => Ok(GgmlDType::BF16),
        _ => crate::bail!("select_qtype_to_ggml: unknown QType code {}", code),
    }
}

#[cfg(test)]
pub(super) mod select_qtype_bpe_x4_tests {
    use super::*;

    /// Cross-check against the CUDA `format_bpe_x4` table in
    /// `candle-kernels/src/quantize/select_kv_format.cuh:346-373`. If a block
    /// struct changes size, both this test and the kernel `static_assert`
    /// would fire — so a single failure here is the canary.
    #[test]
    fn matches_cuda_format_bpe_x4() {
        let cases: &[(i32, i32)] = &[
            (SELECT_FMT_F16, 64),
            (SELECT_FMT_BF16, 64),
            (QType::Q8_KS as i32, 36),
            (QType::Q8_1 as i32, 36),
            (QType::Q8_0 as i32, 34),
            (QType::Q5_1 as i32, 24),
            (QType::Q5_0 as i32, 22),
            (QType::Q4_KS as i32, 20),
            (QType::Q4_1 as i32, 20),
            (QType::Q4_0 as i32, 18),
            (QType::Q3_1 as i32, 16),
            (QType::Q3_0 as i32, 14),
            (QType::Q2_1 as i32, 12),
            (QType::Q2_A as i32, 10),
            (QType::Q2_0 as i32, 10),
            (QType::Q2_S as i32, 9),
            (QType::Q0_M4 as i32, 8),
            (QType::Q1_A as i32, 6),
            (QType::Q1_S as i32, 5),
            (QType::Q0_M2 as i32, 3),
            (QType::Q0_V as i32, 2),
            (QType::Q0_X as i32, 2),
            (QType::Q0 as i32, 1),
        ];
        for &(code, expected) in cases {
            let got = select_qtype_bpe_x4(code);
            assert_eq!(
                got, expected,
                "select_qtype_bpe_x4({code}) = {got}, expected {expected} (kernel format_bpe_x4)"
            );
        }
    }
}
