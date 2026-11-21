//! Describes the palette-4 KV head record and converts KV blocks between arena
//! formats.
//!
//! A palette-4 head stores its K and V data in four palettes, each in its own
//! format, with a 2-bit map that assigns every dimension to one palette. The
//! record that lists those pointers, formats and scales has a fixed layout tied
//! to the head width, and the functions here write it byte for byte. Conversion
//! takes source and destination maps that may differ, runs one kernel for K and
//! one for V, and reads each chunk's live token window so that dead slots count as
//! zero. Without that, a stale value in a dead slot would be folded into a block
//! scale and decode the live tokens as NaN.

use super::*;

// Palette4 buffered conversion API

/// KvHead byte layout — parameterized by head_dim (128 or 256), mirroring
/// `kv_head_byte_size<HD>()` and the accessors in slot_types.cuh. The record
/// is `[k_pal hd/4][v_pal hd/4][k_ptr 4×u64][v_ptr 4×u64][k_fmt 4][v_fmt 4]
/// [k_scale 4×f32][v_scale 4×f32]` = hd/2 + 104 bytes.
pub const KVHEAD_N_PAL: usize = 4;

/// The widest head the palette records support; pal-map buffers are sized for
/// it and only the first `head_dim / 4` bytes are live.
pub const KVHEAD_MAX_HD: usize = 256;

/// A 2-bit-packed palette assignment map, sized for the widest head.
pub type PalMapBytes = [u8; KVHEAD_MAX_HD / 4];

pub(super) const fn kvhead_size(hd: usize) -> usize {
    hd / 2 + 104
}

pub(super) const fn kvhead_v_pal_off(hd: usize) -> usize {
    hd / 4
}

pub(super) const fn kvhead_k_ptr_off(hd: usize) -> usize {
    hd / 2
}

pub(super) const fn kvhead_v_ptr_off(hd: usize) -> usize {
    hd / 2 + 32
}

pub(super) const fn kvhead_k_fmt_off(hd: usize) -> usize {
    hd / 2 + 64
}

pub(super) const fn kvhead_v_fmt_off(hd: usize) -> usize {
    hd / 2 + 68
}

pub(super) const fn kvhead_k_scale_off(hd: usize) -> usize {
    hd / 2 + 72
}

pub(super) const fn kvhead_v_scale_off(hd: usize) -> usize {
    hd / 2 + 88
}

/// The head widths the palette4 convert kernel is instantiated for.
pub fn kvhead_supported_head_dim(hd: usize) -> bool {
    matches!(hd, 128 | 256)
}

/// Maps a `GgmlDType` to the 1-byte ArenaFormat code expected by the palette4
/// convert kernel.  Returns an error for types that the kernel does not
/// support.
pub fn ggml_dtype_to_arena_fmt_code(dtype: GgmlDType) -> Result<u8> {
    let code: u8 = match dtype {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::BF16 => 2,
        GgmlDType::R16 => 3,
        GgmlDType::P2 => 4,
        GgmlDType::QAWQ => 5,
        GgmlDType::QAWQ_G64 => 6,
        GgmlDType::Q8_0 => 7,
        GgmlDType::Q8_1 => 8,
        GgmlDType::Q8_K => 9,
        GgmlDType::Q8_KS => 10,
        GgmlDType::Q6_K => 11,
        GgmlDType::Q5_0 => 12,
        GgmlDType::Q5_1 => 13,
        GgmlDType::Q5_K => 14,

        GgmlDType::Q4_0 => 15,
        GgmlDType::Q4_1 => 16,
        GgmlDType::Q4_K => 17,
        GgmlDType::Q4_KS => 18,

        GgmlDType::Q3_0 => 19,
        GgmlDType::Q3_1 => 20,
        GgmlDType::Q3_K => 21,

        GgmlDType::Q2_0 => 22,
        GgmlDType::Q2_1 => 23,
        GgmlDType::Q2_K => 24,
        GgmlDType::Q2_S => 25,
        GgmlDType::Q2_A => 26,

        GgmlDType::Q1_S => 27,

        GgmlDType::Q0_V => 28,
        GgmlDType::Q1_A => 29,
        GgmlDType::Q0_X => 30,
        GgmlDType::Q0_M2 => 31,
        GgmlDType::Q0_M4 => 32,
        GgmlDType::Q0 => 33,

        _ => crate::bail!(
            "ggml_dtype_to_arena_fmt_code: unsupported dtype {:?}",
            dtype
        ),
    };
    Ok(code)
}

/// Per-head descriptor for `quantize_palette4_convert_buffered`.
///
/// Describes one `(layer, kv_head)` conversion job.  Separate src and dst
/// arena pointers and format codes are required because the kernel reads from
/// `src_kvhead_ptrs` arena pointers and writes to `dst_kvhead_ptrs` arena
/// pointers (different GPU memory regions, potentially different formats).
///
/// `k_pal_map` / `v_pal_map` are 2-bit-packed arrays covering `head_dim`
/// dimensions (the first `head_dim / 4` bytes are live). Each 2-bit field
/// selects which palette (0-3) owns that dimension. Use
/// `identity_pal_map(head_dim)` to get the standard identity map
/// (dims 0..head_dim/4 → pal 0, …). Src and dst may have independent pal_maps.
pub struct PalHeadDesc {
    /// Raw device pointers to source K arena data for palettes 0-3.
    pub k_src_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// Raw device pointers to source V arena data for palettes 0-3.
    pub v_src_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// GgmlDType format of each source K palette arena.
    pub k_src_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// GgmlDType format of each source V palette arena.
    pub v_src_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// 2-bit-packed source K palette assignment map.
    pub k_src_pal_map: PalMapBytes,
    /// 2-bit-packed source V palette assignment map.
    pub v_src_pal_map: PalMapBytes,
    /// Outer scale baked into the source K arena per palette (f32, 1.0 for
    /// float-typed sources). The encoder kernel divides dequantized source
    /// values by this so a re-compression chain (quant → quant) round-trips
    /// correctly. Pass 1.0 for any float source (F16, F32, BF16, R16) — those
    /// have no outer-scale concept.
    pub k_src_scales: [f32; KVHEAD_N_PAL],
    /// Outer scale baked into the source V arena per palette (f32, 1.0 for
    /// float-typed sources). Same convention as k_src_scales.
    pub v_src_scales: [f32; KVHEAD_N_PAL],
    /// Raw device pointers to destination K arena data for palettes 0-3.
    pub k_dst_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// Raw device pointers to destination V arena data for palettes 0-3.
    pub v_dst_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// GgmlDType format of each destination K palette arena.
    pub k_dst_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// GgmlDType format of each destination V palette arena.
    pub v_dst_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// 2-bit-packed destination K palette assignment map.
    pub k_dst_pal_map: PalMapBytes,
    /// 2-bit-packed destination V palette assignment map.
    pub v_dst_pal_map: PalMapBytes,
    /// Post-dequant scale written into the dst KvHead for K (f32, default 1.0).
    /// The decode kernel multiplies dequantized K values by this scale per palette.
    pub k_dst_scales: [f32; KVHEAD_N_PAL],
    /// Post-dequant scale written into the dst KvHead for V (f32, default 1.0).
    pub v_dst_scales: [f32; KVHEAD_N_PAL],
    /// Live token window of the chunk this descriptor converts, packed
    /// `(offset << 8) | len` with `len >= 1` and `offset + len <= 32`
    /// ([`FULL_CHUNK_WINDOW`] for a full chunk; it applies to every chunk of
    /// a multi-chunk job). The kernel reads every token outside it as zero:
    /// a partial chunk's dead slots hold whatever the recycled ground held,
    /// and each block's scale is taken over all 32 tokens, so a stale inf
    /// there would decode the live tokens as NaN.
    pub valid_range: i32,
}

/// [`PalHeadDesc::valid_range`] of a full 32-token chunk: offset 0, length 32.
pub const FULL_CHUNK_WINDOW: i32 = 32;

/// Refuse a packed `(offset << 8) | len` window that is not a non-empty run
/// of tokens inside one 32-token chunk — what the selection and the convert
/// read a chunk's live tokens by.
pub fn check_token_window(range: i32) -> Result<()> {
    let (lo, len) = (range >> 8, range & 0xff);
    if !(0..32).contains(&lo) || len == 0 || lo + len > 32 {
        crate::bail!("valid_range {range:#x} is not a token window inside one 32-token chunk");
    }
    Ok(())
}

/// Build the identity 2-bit-packed palette map for `head_dim` (only the first
/// `head_dim / 4` bytes are live; the rest stay zero).
pub fn identity_pal_map(head_dim: usize) -> PalMapBytes {
    debug_assert!(kvhead_supported_head_dim(head_dim));
    let pal_dim = head_dim / KVHEAD_N_PAL;
    let mut out = [0u8; KVHEAD_MAX_HD / 4];
    for d in 0..head_dim {
        let p = (d / pal_dim) as u8;
        out[d / 4] |= (p & 0x3) << (2 * (d % 4));
    }
    out
}

/// Build a balanced pseudo-random 2-bit-packed palette map for `head_dim`.
///
/// Assigns each dim to one of 4 palettes using a Fisher-Yates shuffle. Every
/// palette is assigned exactly `head_dim / 4` dims but the assignment is
/// non-contiguous and pseudo-random. The caller supplies a seed/IV so different
/// randomization events can generate different maps while remaining reproducible.
pub fn shuffled_pal_map(head_dim: usize, seed: u64) -> PalMapBytes {
    debug_assert!(kvhead_supported_head_dim(head_dim));
    let pal_dim = head_dim / KVHEAD_N_PAL;
    // Start with exactly pal_dim dims per palette (sequential assignment), then
    // shuffle with a caller-provided seed so the palette sizes stay balanced
    // while the dim routing varies between randomization events.
    let mut assign = [0u8; KVHEAD_MAX_HD];
    for (d, a) in assign.iter_mut().enumerate().take(head_dim) {
        *a = (d / pal_dim) as u8;
    }

    // Fisher-Yates with Knuth multiplicative LCG.
    let mut rng: u64 = seed ^ 0x9e3779b97f4a7c15u64;
    if rng == 0 {
        rng = 0x9e3779b97f4a7c15u64;
    }
    for i in (1..head_dim).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        assign.swap(i, j);
    }

    // Pack: 4 dims per byte, 2 bits per dim in little-endian order.
    let mut out = [0u8; KVHEAD_MAX_HD / 4];
    for d in 0..head_dim {
        out[d / 4] |= (assign[d] & 0x3) << (2 * (d % 4));
    }
    out
}

/// Serialize arena pointers, formats, pal_map, and scales into a KvHead byte
/// record for `head_dim` (`head_dim / 2 + 104` bytes).
#[allow(clippy::too_many_arguments)]
pub fn build_kvhead_bytes_raw(
    head_dim: usize,
    k_arena_ptrs: &[u64; KVHEAD_N_PAL],
    v_arena_ptrs: &[u64; KVHEAD_N_PAL],
    k_fmts: &[GgmlDType; KVHEAD_N_PAL],
    v_fmts: &[GgmlDType; KVHEAD_N_PAL],
    k_pal_map: &PalMapBytes,
    v_pal_map: &PalMapBytes,
    k_scales: &[f32; KVHEAD_N_PAL],
    v_scales: &[f32; KVHEAD_N_PAL],
) -> Result<Vec<u8>> {
    if !kvhead_supported_head_dim(head_dim) {
        crate::bail!("build_kvhead_bytes_raw: unsupported head_dim {head_dim}");
    }
    let mut head = vec![0u8; kvhead_size(head_dim)];
    write_kvhead_bytes(
        head_dim,
        k_arena_ptrs,
        v_arena_ptrs,
        k_fmts,
        v_fmts,
        k_pal_map,
        v_pal_map,
        k_scales,
        v_scales,
        &mut head,
    )?;
    Ok(head)
}

/// [`build_kvhead_bytes_raw`] into `head`, the caller's slice of exactly one
/// record (`head_dim / 2 + 104` bytes) — so a table of records is written
/// into one buffer with no allocation per record. Every byte of `head` is
/// written.
#[allow(clippy::too_many_arguments)]
pub fn write_kvhead_bytes(
    head_dim: usize,
    k_arena_ptrs: &[u64; KVHEAD_N_PAL],
    v_arena_ptrs: &[u64; KVHEAD_N_PAL],
    k_fmts: &[GgmlDType; KVHEAD_N_PAL],
    v_fmts: &[GgmlDType; KVHEAD_N_PAL],
    k_pal_map: &PalMapBytes,
    v_pal_map: &PalMapBytes,
    k_scales: &[f32; KVHEAD_N_PAL],
    v_scales: &[f32; KVHEAD_N_PAL],
    head: &mut [u8],
) -> Result<()> {
    if !kvhead_supported_head_dim(head_dim) {
        crate::bail!("write_kvhead_bytes: unsupported head_dim {head_dim}");
    }
    if head.len() != kvhead_size(head_dim) {
        crate::bail!(
            "write_kvhead_bytes: a {} byte slice for a {} byte record",
            head.len(),
            kvhead_size(head_dim)
        );
    }
    head.fill(0);
    let pal_bytes = head_dim / 4;
    head[..pal_bytes].copy_from_slice(&k_pal_map[..pal_bytes]);
    head[kvhead_v_pal_off(head_dim)..kvhead_v_pal_off(head_dim) + pal_bytes]
        .copy_from_slice(&v_pal_map[..pal_bytes]);
    for p in 0..KVHEAD_N_PAL {
        let k_ptr_off = kvhead_k_ptr_off(head_dim) + p * 8;
        head[k_ptr_off..k_ptr_off + 8].copy_from_slice(&k_arena_ptrs[p].to_le_bytes());
        let v_ptr_off = kvhead_v_ptr_off(head_dim) + p * 8;
        head[v_ptr_off..v_ptr_off + 8].copy_from_slice(&v_arena_ptrs[p].to_le_bytes());
        head[kvhead_k_fmt_off(head_dim) + p] = ggml_dtype_to_arena_fmt_code(k_fmts[p])?;
        head[kvhead_v_fmt_off(head_dim) + p] = ggml_dtype_to_arena_fmt_code(v_fmts[p])?;
        let k_f32 = k_scales[p].to_le_bytes();
        let v_f32 = v_scales[p].to_le_bytes();
        let ks = kvhead_k_scale_off(head_dim) + p * 4;
        let vs = kvhead_v_scale_off(head_dim) + p * 4;
        head[ks..ks + 4].copy_from_slice(&k_f32);
        head[vs..vs + 4].copy_from_slice(&v_f32);
    }
    Ok(())
}

/// Palette4 KV-cache format conversion.
///
/// Converts K and V between arbitrary arena formats using 4-palette metadata:
/// serialises the `PalHeadDesc` descriptors into KvHead GPU structs plus one
/// token window per descriptor, uploads them in one staged buffer, and runs
/// the kernel for K and then V. Source and destination may have independent
/// palette maps; routing is a per-dim xlat table the kernel builds on-GPU.
///
/// # Arguments
/// * `descs` - Row-major `[num_layers][num_kv_heads]` slice of head descriptors.
/// * `num_kv_heads` - KV heads per layer.
/// * `num_layers` - Number of transformer layers.
/// * `num_chunks` - 32-token chunks per head to convert.
/// * `stager` - Optional pinned staging allocator for fully async upload.
/// * `dev` - CUDA device.
///
/// `descs.len()` must equal `num_layers * num_kv_heads`.
pub fn quantize_palette4_convert_buffered(
    head_dim: usize,
    descs: &[PalHeadDesc],
    num_kv_heads: usize,
    num_layers: usize,
    num_chunks: usize,
    generation: &super::super::pinned_staging::Generation,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    if num_kv_heads == 0 || num_layers == 0 || num_chunks == 0 {
        return Ok(());
    }
    if !kvhead_supported_head_dim(head_dim) {
        crate::bail!("quantize_palette4_convert_buffered: unsupported head_dim {head_dim}");
    }
    let kvhead_bytes = kvhead_size(head_dim);

    // Guard against i32 overflow in the kernel's grid/count parameters.
    for (name, val) in [
        ("num_kv_heads", num_kv_heads),
        ("num_layers", num_layers),
        ("num_chunks", num_chunks),
    ] {
        if val > i32::MAX as usize {
            crate::bail!("quantize_palette4_convert_buffered: {name}={val} exceeds i32::MAX");
        }
    }

    let expected = num_layers * num_kv_heads;
    if descs.len() != expected {
        crate::bail!(
            "quantize_palette4_convert_buffered: descs.len()={} != num_layers*num_kv_heads={}",
            descs.len(),
            expected
        );
    }

    // Validate that every arena pointer referenced by the pal_map is non-null.
    // A zero pointer would cause CUDA_ERROR_ILLEGAL_ADDRESS inside the kernel,
    // which is much harder to diagnose than a Rust-level error here.
    for (i, desc) in descs.iter().enumerate() {
        // Determine which palettes are actually referenced by each pal_map.
        let check_ptrs = |ptrs: &[u64; KVHEAD_N_PAL],
                          pal_map: &PalMapBytes,
                          side: &str,
                          kv: &str|
         -> Result<()> {
            let mut used = [false; KVHEAD_N_PAL];
            for d in 0..head_dim {
                let p = ((pal_map[d / 4] >> (2 * (d % 4))) & 0x3) as usize;
                used[p] = true;
            }
            for p in 0..KVHEAD_N_PAL {
                if used[p] && ptrs[p] == 0 {
                    crate::bail!(
                        "quantize_palette4_convert_buffered: desc[{i}] {kv} {side} \
                         arena pointer for palette {p} is null (0) but pal_map references it"
                    );
                }
            }
            Ok(())
        };
        check_ptrs(&desc.k_src_arena_ptrs, &desc.k_src_pal_map, "src", "K")?;
        check_ptrs(&desc.k_dst_arena_ptrs, &desc.k_dst_pal_map, "dst", "K")?;
        check_ptrs(&desc.v_src_arena_ptrs, &desc.v_src_pal_map, "src", "V")?;
        check_ptrs(&desc.v_dst_arena_ptrs, &desc.v_dst_pal_map, "dst", "V")?;
        check_token_window(desc.valid_range)
            .map_err(|e| e.context(format!("quantize_palette4_convert_buffered: desc[{i}]")))?;
    }

    // Pack everything into one GPU allocation:
    //
    //   [ src KvHead[0..N]    ]   offset 0
    //   [ dst KvHead[0..N]    ]   offset N × KVHEAD_SIZE
    //   [ valid_range[0..N]   ]   offset 2N × KVHEAD_SIZE (i32 each)
    //
    // Total = N × (2 × KVHEAD_SIZE + 4) bytes.  The kernel computes head
    // pointers as base + job * KVHEAD_SIZE (src) and base + (N + job) *
    // KVHEAD_SIZE (dst), so no pointer arrays are needed. KVHEAD_SIZE is a
    // multiple of 8 at every supported head_dim, so the window array is
    // 4-byte aligned.
    let layout = Palette4ConvertLayout::new(head_dim, expected);
    // Per-palette outer scales live inside each dst KvHead struct (f32 at
    // HD/2+72 / HD/2+88), so the encoder (multiply by outer) and decoder
    // (divide by outer) share a single source of truth.
    let total_bytes = layout.bytes();

    // Build the CPU image directly in pinned memory (no intermediate Vec).
    let mut buf = generation.alloc(total_bytes)?;

    for (i, desc) in descs.iter().enumerate() {
        let src_bytes = build_kvhead_bytes_raw(
            head_dim,
            &desc.k_src_arena_ptrs,
            &desc.v_src_arena_ptrs,
            &desc.k_src_fmts,
            &desc.v_src_fmts,
            &desc.k_src_pal_map,
            &desc.v_src_pal_map,
            &desc.k_src_scales,
            &desc.v_src_scales,
        )?;
        let dst_bytes = build_kvhead_bytes_raw(
            head_dim,
            &desc.k_dst_arena_ptrs,
            &desc.v_dst_arena_ptrs,
            &desc.k_dst_fmts,
            &desc.v_dst_fmts,
            &desc.k_dst_pal_map,
            &desc.v_dst_pal_map,
            &desc.k_dst_scales,
            &desc.v_dst_scales,
        )?;

        let src_off = layout.src_off(i);
        let dst_off = layout.dst_off(i);
        buf[src_off..src_off + kvhead_bytes].copy_from_slice(&src_bytes);
        buf[dst_off..dst_off + kvhead_bytes].copy_from_slice(&dst_bytes);
        let range_off = layout.ranges_off() + i * 4;
        buf[range_off..range_off + 4].copy_from_slice(&desc.valid_range.to_le_bytes());
    }

    // Async H2D upload via stager (deferred cleanup).
    let gpu_buf = generation.submit(buf)?;

    // SAFETY: the buffer was just laid out for `n` descriptors and uploaded on
    // `stream`; it drops stream-ordered after the launches.
    unsafe {
        palette4_convert_launch(
            gpu_buf.dev_ptr(),
            head_dim,
            num_kv_heads,
            num_layers,
            num_chunks,
            stream,
        )?;
    }
    // gpu_buf drops here — stream-ordered after the kernel launches.
    Ok(())
}

/// Where each part of a palette-4 convert buffer of `n` descriptors sits:
/// `n` source KvHead records, then `n` destination records, then `n` token
/// windows (`i32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette4ConvertLayout {
    pub n: usize,
    pub record_bytes: usize,
}

impl Palette4ConvertLayout {
    pub fn new(head_dim: usize, n: usize) -> Self {
        Self {
            n,
            record_bytes: kvhead_size(head_dim),
        }
    }

    pub fn src_off(&self, i: usize) -> usize {
        i * self.record_bytes
    }

    pub fn dst_off(&self, i: usize) -> usize {
        (self.n + i) * self.record_bytes
    }

    pub fn ranges_off(&self) -> usize {
        2 * self.n * self.record_bytes
    }

    pub fn bytes(&self) -> usize {
        self.ranges_off() + self.n * 4
    }
}

/// Launch the palette-4 convert, K then V, over the buffer at `base_ptr`,
/// laid out by [`Palette4ConvertLayout`] for `num_layers × num_kv_heads`
/// descriptors. A launch the driver refuses is an error, so a caller never
/// treats a convert that did not run as written.
///
/// # Safety
///
/// `base_ptr` is a device buffer of `Palette4ConvertLayout::bytes()` whose
/// records and windows are written, ahead of this launch on `stream`, and
/// which outlives it. Every pointer a record names is a writable (destination)
/// or readable (source) band of the record's format.
pub unsafe fn palette4_convert_launch(
    base_ptr: u64,
    head_dim: usize,
    num_kv_heads: usize,
    num_layers: usize,
    num_chunks: usize,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let layout = Palette4ConvertLayout::new(head_dim, num_layers * num_kv_heads);
    // Grid: dim3(num_kv_heads, num_layers) → exactly n blocks per launch.
    let ranges_ptr = (base_ptr + layout.ranges_off() as u64) as *const i32;
    let raw_stream = stream.cu_stream() as *mut _;
    for (is_k, label) in [
        (1, "run_quantize_palette4_convert (K)"),
        (0, "run_quantize_palette4_convert (V)"),
    ] {
        crate::set_kernel_breadcrumb(label, file!(), line!());
        let rc = run_quantize_palette4_convert(
            base_ptr as *const u8,
            ranges_ptr,
            layout.n as i32,
            num_kv_heads as i32,
            num_layers as i32,
            num_chunks as i32,
            is_k,
            head_dim as i32,
            raw_stream,
        );
        if rc != 0 {
            crate::bail!("{label}: the launch was refused ({rc})");
        }
    }
    Ok(())
}
