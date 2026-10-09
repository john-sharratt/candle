//! FFI bindings for quantized batched matmul kernels
//!
//! Dispatcher for quantized matrix-vector multiplication kernels.
//! Selects the appropriate kernel based on quantization type, Y type, and tensor core usage.

use core::ffi::c_void;

/// The most weight segments one split-K dense launch takes ([`run_dense_int8_splitk`]),
/// mirroring `SPLITK_MAX_SEGS` in `quantized/splitk_segs.cuh`.
pub const SPLITK_MAX_SEGS: usize = 4;

/// Segment descriptor for segmented dispatch.
/// One per expert (MoE) or one total (non-MoE).
/// Matches C-side `vx_segment_t` in dispatcher.cu.
#[repr(C)]
pub struct VxSegment {
    /// Device pointer to quantized weight data
    pub weights: *const c_void,
    /// Number of batches in this segment (greedy decomposition boundary)
    pub batch_count: i32,
}

// Safety: VxSegment contains a raw pointer that is only dereferenced on the GPU side.
// It is constructed on the host and passed to CUDA kernels via FFI.
unsafe impl Send for VxSegment {}
unsafe impl Sync for VxSegment {}

/// A grouped GEMM launched over a **live** expert table — field-for-field the C
/// `MoeLive` in `quantized/moe_live.cuh`, passed by value to the kernel. The
/// kernel entry (`kernel.cuh`, "A live expert table") documents the worker
/// blocks and the counter layout.
///
/// Every field is a device address or a plain number:
/// - `abort` — a mapped host `u32` the host sets non-zero to end every wait;
/// - `fault` — a mapped host `u64` the first worker whose wait ends without its
///   expert (aborted, or past `spin_limit_ns`) claims with what it waited on;
///   any other cold wait that sees it set gives its expert up at once, and the
///   host fails the forward (`MOE_FAULT_*` below for the bits);
/// - `live_row` — this projection's row of the live table (mapped host
///   memory), where a worker waits for a cold expert; every other expert's
///   address comes from the launch's weight table, `moe_bucketize`'s snapshot;
/// - `remote`, `header` — `moe_bucketize`'s remote-expert list and header;
/// - `remote_dst`, `dst_offset` — each remote expert's promotion slot image (or
///   0), and where this projection sits in one: the workers store every slice
///   they copy there too;
/// - `counter` — this launch's work counter (zeroed by `moe_bucketize`);
/// - `scratch`, `slot_bytes` — the workers' VRAM slots;
/// - `stall` — the profile build's `u64[STALL_WORDS]` per-row counters (0
///   otherwise);
/// - `ahead`, `ahead_done` — the gate launch's read-ahead items and their piece
///   counters (`moe_read_ahead.cuh`), 0 on the up and down launches;
/// - `spin_limit_ns` — the backstop on any single wait;
/// - `workers` — the worker-block count;
/// - `row` — the launch's MoE row, which a fault names.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeLive {
    pub abort: u64,
    pub fault: u64,
    pub live_row: u64,
    pub remote: u64,
    pub remote_dst: u64,
    pub header: u64,
    pub counter: u64,
    pub scratch: u64,
    pub slot_bytes: u64,
    pub dst_offset: u64,
    pub stall: u64,
    pub ahead: u64,
    pub ahead_done: u64,
    pub spin_limit_ns: u64,
    pub workers: i32,
    pub row: i32,
}

/// The fault word's fields (`MoeLive::fault`), mirrored from `kernel.cuh`:
/// bit 63 set when the wait ended on the abort word rather than the spin limit,
/// the row in bits 48–62, the expert in bits 32–47, the microseconds waited in
/// bits 0–31 (saturating).
pub const MOE_FAULT_ABORTED: u64 = 1 << 63;
pub const MOE_FAULT_ROW_SHIFT: u32 = 48;
pub const MOE_FAULT_EXPERT_SHIFT: u32 = 32;

/// `u64` profile counters per row in `MoeLive::stall`, mirrored from
/// `MOE_LIVE_STALL_WORDS` in `kernel.cuh`.
pub const STALL_WORDS: usize = 7;

/// Quantization type enum for the matmul dispatcher (`run_quantized_matmul`).
///
/// Integer values MUST match `GgmlDType` (candle-core) — `GgmlDType` is the
/// single source of truth for quant-format numbering across this whole
/// workspace. The CUDA `dispatcher.cu` uses its own 14-entry kernel-lookup
/// table internally, accessed via a `qtype_to_matmul_kernel_index` helper
/// (see `block_compact.cuh`) — the enum numbering no longer has to be
/// contiguous 0..13 just to index it.
///
/// Only the formats the matmul dispatcher actually has kernels for are
/// listed here; the KV-quant-only formats (Q4_KS/Q8_KS/Q0 family/etc.) are
/// intentionally absent because `run_quantized_matmul` would reject them
/// anyway.
#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QType {
    // Values are `GgmlDType as u32` discriminants.
    Q_AWQ = 5,
    Q_AWQ_G64 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q8_K = 9,
    Q6_K = 11,
    Q5_0 = 12,
    Q5_1 = 13,
    Q5_K = 14,
    Q4_0 = 15,
    Q4_1 = 16,
    Q4_K = 17,
    Q3_K = 21,
    Q2_K = 24,
    // Byte-permuted ("ordered") twins of the K-quant compact blocks for the q8a128
    // int8 path: qs made contiguous, per-sub scales grouped at the tail. GPU-only
    // weight layouts (produced by an on-GPU permutation of the K block) — not
    // GgmlDTypes, so they have no on-disk / CPU form. Values mirror QTYPE_* in
    // block_compact.cuh (45-48, the first slots free past GgmlDType's storage dtypes).
    Q4_KO = 45,
    Q5_KO = 46,
    Q6_KO = 47,
    Q8_KO = 48,
    // Lane-major per-sub MXFP4 for the q8a128 int8 path: the routed MXFP4 experts,
    // byte-permuted into the KO lane layout; the kernel runs one int32 MMA per 32-K sub and
    // folds each with its own E8M0 scale in FP — exact. Stays 4-bit. Value 50 mirrors
    // QTYPE_MXFP4_KO. Native MXFP4 (49) has no matmul kernel, so it is intentionally absent
    // from this kernel-only enum.
    MXFP4_KO = 50,
    /// Lane-major per-128 affine KO twin at 2-bit — the smallest KO weight (value 0..3, stored
    /// as the 2-bit crumb region Q6_KO carries, used as the whole value). Value 51 mirrors
    /// QTYPE_Q2_KO. Read by the maintained per-128 int8 fold; GPU-only.
    Q2_KO = 51,
    /// Lane-major per-128 affine KO twin at 3-bit (value 0..7) — `Q3_K`'s same-width twin: a
    /// 2-bit crumb plane (Q2_KO's region) at bits 0-1 plus a 1-bit hi plane (Q5_KO's region)
    /// at bit 2, and no `ql`. Value 52 mirrors QTYPE_Q3_KO. GPU-only.
    Q3_KO = 52,
}

/// Y vector type enum (matches dispatcher ytype parameter).
/// MUST match C++ dispatcher ordering: 0=F16, 1=BF16, 2=F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum YType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    /// q8a128 activations → INT8-MMA path (tensor-core only, F32 output). `vy` is a
    /// `block_q8a128` buffer rather than an FP tensor. No FP fallback — callers only select this
    /// when tensor cores are available. There is ONE activation type: the q8a1024 byte layout is
    /// position-independent and identical for both matmul modes. The dispatcher picks mode-1
    /// (Bm=16) vs mode-2 (Bm=32 weight-reuse) from the token count at a threshold — the mode is a
    /// kernel/tiling property, not an attribute of the activation.
    Q8A128 = 3,
}

/// Store width for the int8 dense matmul's output.
///
/// The MMA accumulator is F32 in registers whichever variant runs; this only picks
/// the width of the final store, so the narrow variants are bit-identical to the
/// F32 one followed by a cast — with the cast's launch and its second buffer gone.
/// MUST match the dispatcher's table ordering: 0=F16, 1=BF16, 2=F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OutDType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
}

/// Status returned by the quantized-matmul launchers.
///
/// Mirrors `QMM_*` in `candle-kernels/src/quantized/matmul_status.cuh`. A kernel
/// table miss returns [`MatmulStatus::NoKernel`] rather than leaving the output
/// buffer untouched, which is the difference between an error and silently wrong
/// numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulStatus {
    Ok,
    BadQType,
    NoSegments,
    BadYType,
    NoKernel,
    BadOutDType,
    BadSplit,
    BadTileMode,
    LaunchFailed,
    Unknown(i32),
}

impl MatmulStatus {
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            1 => Self::BadQType,
            2 => Self::NoSegments,
            3 => Self::BadYType,
            4 => Self::NoKernel,
            5 => Self::BadOutDType,
            6 => Self::BadSplit,
            7 => Self::BadTileMode,
            8 => Self::LaunchFailed,
            other => Self::Unknown(other),
        }
    }

    /// Human-readable reason a launch did not happen, or `None` when it did.
    pub fn failure(self) -> Option<&'static str> {
        match self {
            Self::Ok => None,
            Self::BadQType => Some("quantization format has no matmul kernel"),
            Self::NoSegments => Some("no weight segments"),
            Self::BadYType => Some("unsupported activation type"),
            Self::NoKernel => Some("no kernel for this (format, output dtype) pair"),
            Self::BadOutDType => Some("unsupported output dtype"),
            Self::BadSplit => {
                Some("split-K depth or narrow geometry out of range, or a format that never splits")
            }
            Self::BadTileMode => Some("int8 token-tile width with no kernel"),
            Self::LaunchFailed => Some("the kernel launch returned an error"),
            Self::Unknown(_) => Some("unrecognised launcher status"),
        }
    }
}

extern "C" {
    /// Dispatches to the appropriate quantized matmul kernel.
    ///
    /// # Parameters
    /// - `segments`: Host array of segment descriptors (one per expert or single for non-MoE)
    /// - `num_segments`: Length of segments array
    /// - `vy`: Y vector (activations), type determined by `ytype`
    /// - `dst`: Output buffer — same type as Y on the FP path, `out_dtype` on the int8 path
    /// - `ncols_x`: Number of columns in X (input features)
    /// - `nrows_x`: Number of rows in X (output features)
    /// - `nrows_y`: Number of rows in Y
    /// - `nrows_dst`: Number of rows in output
    /// - `qtype`: Quantization type (0-9, see QType enum)
    /// - `ytype`: Y vector type (0-2, see YType enum). Note: F32 (2) only for Q4_K.
    /// - `weight_bytes`: Weight tensor size in bytes (for L2 cache dispatch decision, FP path)
    /// - `tile_mode`: int8 dense tiling select — 0 = mode-1 (Bm=16), 1 = mode-2 (Bm=32
    ///   weight-reuse), 2 = mode-4 (Bm=64 × 128-row prefill tile, KO formats). Decided in Rust
    ///   by `q8a128_dense_tile` (candle-core); ignored by the FP path.
    /// - `out_dtype`: int8 dense store width (see [`OutDType`]); ignored by the FP path, where
    ///   the output dtype is the activation dtype.
    ///
    /// Returns a [`MatmulStatus`] code — non-zero means no kernel ran and `dst` is untouched.
    pub fn run_quantized_matmul(
        segments: *const VxSegment,
        num_segments: i32,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        qtype: i32,
        ytype: i32,
        weight_bytes: usize,
        tile_mode: i32,
        out_dtype: i32,
        // The activation operand's `SumScale::as_code()` — 0 raw Σx, 1 Σx/amax.
        // Read by the int8 dense path only; the FP kernels carry no q8a128 header.
        sum_norm: i32,
        // The stream every launch is issued on: the device's compute stream, or a
        // capture stream when the launches are being recorded into a graph.
        stream: *mut c_void,
    ) -> i32;

    /// Split-K int8 dense matmul: q8a128 activations `[M, K]` × `num_segs` KO weights of one
    /// format, `weights[s]` of `nrows[s]` rows (each a multiple of 32, at most
    /// [`SPLITK_MAX_SEGS`]) → `dst [M, ΣN]` at `out_dtype`, segment `s` in the columns after
    /// the segments before it. K is cut into `splits` slices so a decode-width projection
    /// with a narrow N fills the card; segments that read one operand share the launch.
    /// - `weights`, `nrows`: HOST arrays of `num_segs` entries.
    /// - `ws`: `K/128 × M × ΣN` F32 partials — one per K tile — device memory.
    /// - `counters`: one u32 per (16-token tile, 32-row tile), all ZERO on entry; the kernel
    ///   leaves them zero, so one buffer serves every launch ordered on the same stream.
    ///
    /// The sum over K tiles runs in tile order in whichever block finishes last — the chain the
    /// unsplit kernel folds in for the affine KO formats, so every output column is the unsplit
    /// kernel's bit for bit, whatever the segments beside it. MXFP4 is not split (its per-sub
    /// fold has no per-tile partial). Returns a [`MatmulStatus`] code.
    pub fn run_dense_int8_splitk(
        weights: *const *const c_void,
        nrows: *const i32,
        num_segs: i32,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        total_batch: i32,
        qtype: i32,
        out_dtype: i32,
        sum_norm: i32,
        splits: i32,
        ws: *mut f32,
        counters: *mut u32,
        stream: *mut c_void,
    ) -> i32;

    /// Narrow int8 dense matmul for decode width: q8a128 activations `[M ≤ 8, K]` ×
    /// `num_segs` KO weights of one format, `weights[s]` of `nrows[s]` rows (each a multiple of
    /// 32, at most [`SPLITK_MAX_SEGS`]) → `dst [M, ΣN]` at `out_dtype`, segment `s` in the
    /// columns after the segments before it. One block per 8-row output tile, its `warps`
    /// warps (1..=16) walking contiguous ranges of K and summing their per-tile folds in shared
    /// memory, in tile order — the unsplit kernel's chain, so every output is the unsplit
    /// kernel's bit for bit. No scratch. MXFP4 does not run narrow.
    /// - `weights`, `nrows`: HOST arrays of `num_segs` entries.
    ///
    /// Returns a [`MatmulStatus`] code: `BadSplit` for M outside 1..=8, a warp count outside
    /// 1..=16, a K whose shared memory exceeds the narrow cap, or a format with no narrow entry.
    pub fn run_dense_int8_narrow(
        weights: *const *const c_void,
        nrows: *const i32,
        num_segs: i32,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        total_batch: i32,
        qtype: i32,
        out_dtype: i32,
        sum_norm: i32,
        warps: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Fused-activation int8 dense matmul for a Q8_KO weight `[N, K]`:
    /// `dst [M, N] (F32) = silu(proj[:, 0..K]) · Wᵀ`, the activation quantized to q8a128 by
    /// the kernel's own tile loader — the bytes `gr_silu_q8` would have written, without its
    /// launch. `proj` is `[M, proj_stride]` F32 (`proj_stride ≥ K`, rows 16-byte aligned);
    /// `mode2` picks the Bm=32 tile as the dense launch does. Returns a [`MatmulStatus`] code:
    /// `NoKernel` for a K that is not whole 128-tiles inside a row, or an N not whole 32-rows.
    pub fn run_dense_int8_silu_q8ko_f32(
        weights: *const c_void,
        proj: *const f32,
        proj_stride: i32,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        total_batch: i32,
        sum_norm: i32,
        mode2: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Segmented qkv int8 dense matmul: one launch over a shared q8a128 activation × up to 3 KO
    /// weights of possibly-different formats, writing the concatenated `[M, N_total]` output.
    /// - `h_segs`: HOST pointer to a `num_segs`-long (≤3) `qkv_seg_t` array (24 bytes each); the
    ///   launcher copies it into by-value kernel params, so there is no per-call device upload.
    /// - `act`: device pointer to the shared q8a128 activation.
    /// - `total_n_tiles`: Σ ceil(seg_n/32) (= grid.y); `dst_stride` = N_total.
    /// - `mode2`: 0 = mode-1 (Bm=16), 1 = mode-2 (Bm=32).
    /// - `out_dtype`: store width for `dst` (see [`OutDType`]).
    ///
    /// Returns a [`MatmulStatus`] code — non-zero means no kernel ran and `dst` is untouched.
    pub fn run_qkv_segmented_matmul(
        h_segs: *const c_void,
        num_segs: i32,
        act: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        total_n_tiles: i32,
        total_batch: i32,
        dst_stride: i32,
        mode2: i32,
        out_dtype: i32,
        // The activation operand's `SumScale::as_code()` — 0 raw Σx, 1 Σx/amax.
        sum_norm: i32,
        // The stream the launch is issued on.
        stream: *mut c_void,
    ) -> i32;

    /// Single-launch grouped matmul over all MoE expert tiles.
    ///
    /// All pointer arguments are DEVICE pointers prepared by the caller:
    /// - `weight_ptrs`: `u64[num_experts]` — each expert's K/128 weight pointer
    /// - `tile_expert`: `i32[num_tiles]` — owning expert id per tile
    /// - `tile_b_start`: `i32[num_tiles]` — stacked-batch start row per tile
    /// - `tile_b_cnt`: `i32[num_tiles]` — tokens in the tile (1..16)
    /// - `vy`: stacked activations `[total_batch, K]`
    /// - `dst`: stacked output `[total_batch, N]`
    ///
    /// `ncols_x = K`, `nrows_x = N`, `y_stride = K`, `dst_stride = N`. One block per
    /// (expert-tile, row-tile), one launch; `row_fast` picks the grid axis order
    /// (1 = row tiles on x, the L2-friendly order when the stacked activation
    /// exceeds L2 — both orders are bit-identical, schedule only).
    pub fn run_grouped_quantized_matmul(
        weight_ptrs: *const c_void,
        tile_expert: *const c_void,
        tile_b_start: *const c_void,
        tile_b_cnt: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        y_stride: i32,
        dst_stride: i32,
        num_tiles: i32,
        qtype: i32,
        ytype: i32,
        // int8 token-tile width / 16 — 2 (Bm 32), 4 (Bm 64) or 8 (Bm 128).
        // The tile tables must be built at 16·n_sub; wide modes exist for the
        // KO rows only (the caller gates on `is_ko`). FP paths ignore it.
        n_sub: i32,
        row_fast: i32,
        // The activation operand's `SumScale::as_code()` — 0 raw Σx, 1 Σx/amax.
        // Read on `ytype == 3` (q8a128) only; the FP grouped kernels do not take
        // it, and the launcher builds a shorter argument list for them.
        sum_norm: i32,
        // A launch over a live expert table ([`MoeLive`]) — the grid gains its
        // worker row; null for every other launch. Read on `ytype == 3` only.
        live: *const MoeLive,
        // The stream to launch on — the device handle's own.
        stream: *mut c_void,
    ) -> i32;

    /// Repack quantized weights to GEMX format (K/128 with embedded scales).
    ///
    /// This removes scale data from the weights (scales should be extracted
    /// separately via extract_scales before calling this) and reorders the
    /// quant bytes for optimal tensor core access patterns.
    ///
    /// # Parameters
    /// - `data`: Weight tensor data (device pointer, modified in-place)
    /// - `nrows`: Number of rows in tensor
    /// - `ncols`: Number of columns in tensor
    /// - `qtype`: Quantization type (0-9, see QType enum)
    ///
    /// # Returns
    /// New size in bytes of the repacked data, or -1 on error
    ///
    /// # Safety
    /// - src_data must be a valid device pointer to the source quantized data
    /// - dst_data must be a valid device pointer with at least get_repacked_size_bytes() bytes
    /// - Returns 0 on success, -1 on error
    #[link_name = "run_repack_gemx"]
    pub fn run_repack_gemx(
        src_data: *const core::ffi::c_void,
        dst_data: *mut core::ffi::c_void,
        nrows: i32,
        ncols: i32,
        qtype: i32,
    ) -> i32;

    /// Get the size of repacked weights without actually repacking.
    ///
    /// # Parameters
    /// - `nrows`: Number of rows in tensor
    /// - `ncols`: Number of columns in tensor
    /// - `qtype`: Quantization type (0-9)
    ///
    /// # Returns
    /// Size in bytes of repacked data, or -1 if format not supported
    pub fn get_repacked_size_bytes(nrows: i32, ncols: i32, qtype: i32) -> i64;

    /// Check if a quantization type supports GEMX repacking.
    ///
    /// # Parameters
    /// - `qtype`: Quantization type (0-9)
    ///
    /// # Returns
    /// 1 if supported, 0 if not
    #[link_name = "is_gemx_supported"]
    pub fn is_gemx_supported(qtype: i32) -> i32;

    /// Dequantize repacked quantized tensor to float32.
    ///
    /// Uses the same element mapping as the matmul loader, allowing direct
    /// debugging of the loader's element indexing without Y multiplication.
    ///
    /// # Parameters
    /// - `x`: Repacked quantized blocks (device pointer)
    /// - `scales`: External scales (device pointer, format depends on qtype)
    /// - `out`: Output float32 buffer (device pointer, nrows × ncols)
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns (must be multiple of block size)
    /// - `qtype`: Quantization type (0-9, see QType enum)
    /// - `stream`: the stream the launch is issued on
    ///
    /// Note: K/128 blocks have embedded scales - no external scales parameter.
    ///
    /// # Returns
    /// 0 on success, -1 on error
    pub fn run_dequantize(
        x: *const c_void,
        out: *mut c_void,
        nrows: i32,
        ncols: i32,
        qtype: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Get the output size (in floats) for dequantizing a tensor.
    ///
    /// # Parameters
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns
    ///
    /// # Returns
    /// Number of float32 output elements (nrows × ncols)
    pub fn get_dequantize_output_size(nrows: i32, ncols: i32) -> i64;

    /// Get dispatch info string describing which kernels will be used.
    ///
    /// Returns a string describing the kernel dispatch plan for a given
    /// batch size and weight tensor size. Useful for benchmarking and debugging.
    ///
    /// # Parameters
    /// - `batch_size`: Number of vectors to process
    /// - `weight_bytes`: Weight tensor size in bytes (determines L2 vs DRAM path)
    /// - `buffer`: Output buffer to write the kernel description (C string)
    /// - `buffer_len`: Size of output buffer (recommend 64+ bytes)
    ///
    /// # Returns
    /// Number of characters written (excluding null terminator), or -1 on error
    ///
    /// # Examples
    /// - "s2i8(16)" - single s2_iter8 kernel for batch 16
    /// - "s2i4(8)+s3(3)" - s2_iter4 for 8 batches + s3 for 3 batches
    /// - "tc32(32)+s8(8)" - tensor core for 32 + s8 for remainder
    pub fn get_dispatch_info(
        batch_size: i32,
        weight_bytes: usize,
        buffer: *mut i8,
        buffer_len: i32,
    ) -> i32;

    /// Flush L2 cache by reading through a buffer larger than L2.
    ///
    /// This is useful for benchmarking to simulate realistic cache conditions
    /// where different matrices alternate and cannot all fit in L2 cache.
    ///
    /// # Parameters
    /// - `buffer`: Pre-allocated device buffer (should be >= 2x L2 cache size)
    /// - `size`: Size of buffer in bytes
    ///
    /// # Safety
    /// - buffer must be a valid device pointer
    /// - Synchronizes the device before returning
    pub fn flush_l2_cache(buffer: *const c_void, size: usize);
}

// =============================================================================
// GEMX TENSOR CORE SUPPORT
// =============================================================================
// GEMX tensor core kernels are dispatched through run_quantized_matmul
// when USE_TC=true in the kernel instantiation. The dispatcher automatically
// uses tensor cores for batch >= 16 on SM80+ when F16 activations are used.
//
// The following utility functions remain for workspace allocation checks.
// =============================================================================

// NOTE: GEMX kernel is integrated into the standard quantized_matmul dispatch path.
// Use run_quantized_matmul with GEMX-repacked weights (K/128 with embedded scales).

// =============================================================================
// SAFE RUST WRAPPERS
// =============================================================================

/// Get a human-readable description of the kernel dispatch plan.
///
/// Returns a string like "s2i8(16)" or "tc32(32)+s8(8)" describing which
/// kernels will be used for the given batch size and weight tensor size.
///
/// # Arguments
/// * `batch_size` - Number of vectors to process
/// * `weight_bytes` - Weight tensor size in bytes (determines L2 vs DRAM path)
///
/// # Returns
/// A String describing the dispatch plan, or an error message on failure.
pub fn dispatch_info(batch_size: i32, weight_bytes: usize) -> String {
    let mut buffer = [0i8; 128];
    let result = unsafe {
        get_dispatch_info(
            batch_size,
            weight_bytes,
            buffer.as_mut_ptr(),
            buffer.len() as i32,
        )
    };
    if result < 0 {
        return "error".to_string();
    }
    // Convert C string to Rust String
    let len = result as usize;
    let bytes: Vec<u8> = buffer[..len].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).to_string()
}

#[cfg(test)]
mod matmul_status_tests {
    //! Pin every status code to its variant. The values are the `QMM_*` defines in
    //! `matmul_status.cuh`; a drift on either side turns a refused launch into a
    //! different refusal — or into `Ok`.
    use super::MatmulStatus;

    #[test]
    fn every_launcher_status_maps_to_its_variant() {
        let expected = [
            (0, MatmulStatus::Ok),
            (1, MatmulStatus::BadQType),
            (2, MatmulStatus::NoSegments),
            (3, MatmulStatus::BadYType),
            (4, MatmulStatus::NoKernel),
            (5, MatmulStatus::BadOutDType),
            (6, MatmulStatus::BadSplit),
            (7, MatmulStatus::BadTileMode),
            (8, MatmulStatus::LaunchFailed),
            (9, MatmulStatus::Unknown(9)),
        ];
        for (code, status) in expected {
            assert_eq!(MatmulStatus::from_code(code), status, "code {code}");
        }
    }

    /// Only `Ok` means a kernel ran; every other code is a reason.
    #[test]
    fn only_ok_is_not_a_failure() {
        for code in 0..=9 {
            let failed = MatmulStatus::from_code(code).failure().is_some();
            assert_eq!(failed, code != 0, "code {code}");
        }
    }
}

#[cfg(test)]
mod matmul_qtype_lock_tests {
    //! Pin the exact integer value for every `QType` variant in this file.
    //! Values must match `GgmlDType` in candle-core. Any drift will also
    //! break the C++ `QTYPE_*` lock in `block_compact.cuh` (which uses the
    //! same values) and the `qtype_to_matmul_kernel_index` mapping in
    //! `block_compact.cuh`.
    use super::QType;

    #[test]
    fn matmul_qtype_values_are_stable() {
        assert_eq!(QType::Q_AWQ as i32, 5);
        assert_eq!(QType::Q_AWQ_G64 as i32, 6);
        assert_eq!(QType::Q8_0 as i32, 7);
        assert_eq!(QType::Q8_1 as i32, 8);
        assert_eq!(QType::Q8_K as i32, 9);
        assert_eq!(QType::Q6_K as i32, 11);
        assert_eq!(QType::Q5_0 as i32, 12);
        assert_eq!(QType::Q5_1 as i32, 13);
        assert_eq!(QType::Q5_K as i32, 14);
        assert_eq!(QType::Q4_0 as i32, 15);
        assert_eq!(QType::Q4_1 as i32, 16);
        assert_eq!(QType::Q4_K as i32, 17);
        assert_eq!(QType::Q3_K as i32, 21);
        assert_eq!(QType::Q2_K as i32, 24);
        // KO byte-permuted twins — mirror QTYPE_Q*_KO in block_compact.cuh.
        assert_eq!(QType::Q4_KO as i32, 45);
        assert_eq!(QType::Q5_KO as i32, 46);
        assert_eq!(QType::Q6_KO as i32, 47);
        assert_eq!(QType::Q8_KO as i32, 48);
        assert_eq!(QType::MXFP4_KO as i32, 50);
    }
}
