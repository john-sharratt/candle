//! Rust wrapper + rung-2 standalone harness for the `paged-deepseek` decode
//! kernel (single-latent K≡V hybrid attention, HEAD_DIM=512).
//!
//! The wrapper drives the kernel over candle CUDA tensors. The test harness
//! builds a **synthetic** window slot (SlotHeader / TokenSlice / KvHead raw
//! bytes + FP8 band arenas, no substrate, no model) plus a synthetic compressed
//! gallery + selection, and validates against two oracles:
//!
//! * **(a) arithmetic mirror** — a CPU replica of the kernel's exact numerics
//!   (FP8 round-trip, per-band int8 quantization, integer QK dots, the
//!   `fast_exp` PTX cubic e^x, fmaf accumulation order, split-KV combine +
//!   sink fold, and the f64-reduced polynomial RoPE trig), gated **bit-exact**
//!   — with RoPE live as well as zeroed. Every kernel operation is either an
//!   explicit `_rn` intrinsic or plain exact-rounded IEEE arithmetic, so the
//!   mirror reproduces it to the bf16 output bit.
//! * **(b) model-quality reference** — a plain float sink-softmax attention
//!   (the `sink_attend` semantics) over the same inputs, gated at int8-scale
//!   tolerance.

#![allow(clippy::too_many_arguments)]

use candle::{DType, Result, Tensor};

pub const HEAD_DIM: usize = 512;
pub const ROPE_DIM: usize = 64;
pub const NOPE_DIM: usize = HEAD_DIM - ROPE_DIM;
pub const N_BANDS: usize = 4;
pub const SUB_DIM: usize = HEAD_DIM / N_BANDS;
pub const CHUNK: usize = 32;
/// KvHead record byte size at HEAD_DIM=512 (slot_types.cuh layout).
pub const KVHEAD_BYTES: usize = HEAD_DIM / 2 + 104;

/// Launch the DeepSeek hybrid decode over one wave of slots. All tensors live
/// on the same CUDA device; `q`/`kv_new` are BF16 **pre-RoPE**, the compressed
/// gallery is F32 pre-RoPE. Returns the de-rotated attention output
/// `[slots, n_q_head, 512]` BF16.
#[cfg(feature = "cuda")]
pub fn paged_deepseek_decode(
    q: &Tensor,          // [slots, H, 512] bf16
    headers: &Tensor,    // [slots*24] u8 (SlotHeader array)
    kv_new: &Tensor,     // [slots, 512] bf16
    comp: &Tensor,       // [g_total, 512] f32
    comp_pos: &Tensor,   // [g_total] u32
    comp_idx: &Tensor,   // [slots, max_sel] u32
    comp_cnt: &Tensor,   // [slots] u32
    sinks: &Tensor,      // [H] f32
    rope_freqs: &Tensor, // [32] f32
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
    dbg: Option<&Tensor>, // [16608] f32 stage-dump (mirror diagnostics)
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_deepseek_decode requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();
    let hdr_ptr = {
        let (storage, layout) = headers.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => {
                let slice = c.as_cuda_slice::<u8>()?;
                let (p, _guard) = slice.device_ptr(&stream);
                p + layout.start_offset() as u64
            }
            _ => candle::bail!("expected CUDA storage for headers"),
        }
    };
    paged_deepseek_decode_raw(
        q,
        hdr_ptr,
        kv_new,
        comp,
        comp_pos,
        comp_idx,
        comp_cnt,
        sinks,
        rope_freqs,
        softmax_scale,
        window_size,
        num_splits_override,
        true, // tensor-headers path is the live buffer: advance the write-len
        dbg,
    )
}

/// As [`paged_deepseek_decode`] but with the `SlotHeader` array given as a raw
/// device address — the form the production cache's `build_decode_metadata`
/// hands out (a pinned-stager `GpuBuf`, not a tensor). The caller keeps the
/// buffer alive across the call.
#[cfg(feature = "cuda")]
pub fn paged_deepseek_decode_raw(
    q: &Tensor,
    headers_ptr: u64,
    kv_new: &Tensor,
    comp: &Tensor,
    comp_pos: &Tensor,
    comp_idx: &Tensor,
    comp_cnt: &Tensor,
    sinks: &Tensor,
    rope_freqs: &Tensor,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
    commit_write_len: bool,
    dbg: Option<&Tensor>,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;

    let (num_slots, n_q_head, hd) = q.dims3()?;
    if hd != HEAD_DIM {
        candle::bail!("paged_deepseek_decode: head_dim {hd} != {HEAD_DIM}");
    }
    let max_sel = comp_idx.dim(1)?;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_deepseek_decode requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();

    let out = Tensor::zeros((num_slots, n_q_head, HEAD_DIM), DType::BF16, q.device())?;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let ptr = match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<$ty>()?;
                    let (p, _guard) = slice.device_ptr(&stream);
                    p + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                }
                _ => candle::bail!("expected CUDA storage"),
            };
            ptr
        }};
    }

    let q_ptr = cuda_ptr!(q, half::bf16);
    let hdr_ptr = headers_ptr;
    let out_ptr = cuda_ptr!(&out, half::bf16);
    let kv_ptr = cuda_ptr!(kv_new, half::bf16);
    let comp_p = cuda_ptr!(comp, f32);
    let cpos_p = cuda_ptr!(comp_pos, u32);
    let cidx_p = cuda_ptr!(comp_idx, u32);
    let ccnt_p = cuda_ptr!(comp_cnt, u32);
    let sink_p = cuda_ptr!(sinks, f32);
    let freq_p = cuda_ptr!(rope_freqs, f32);
    let dbg_p = match dbg {
        Some(t) => cuda_ptr!(t, f32),
        None => 0u64,
    };

    unsafe {
        candle_kernels::paged_deepseek::run_paged_deepseek_decode_bf16(
            q_ptr as *const core::ffi::c_void,
            hdr_ptr as *const u8,
            out_ptr as *mut core::ffi::c_void,
            kv_ptr as *const core::ffi::c_void,
            comp_p as *const f32,
            cpos_p as *const u32,
            cidx_p as *const u32,
            ccnt_p as *const u32,
            sink_p as *const f32,
            freq_p as *const f32,
            num_slots as i32,
            n_q_head as i32,
            softmax_scale,
            window_size as i32,
            max_sel as i32,
            num_splits_override as i32,
            commit_write_len as i32,
            dbg_p as *mut f32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    q.device().synchronize()?;
    Ok(out)
}

/// The prefill entry: many queries over a SETTLED slot (all latents written +
/// committed before the call — no fused scatter). `q` `[total_q, H, 512]`
/// bf16 pre-RoPE, `q_pos` `[total_q]` u32, per-query selections. Numerics are
/// identical to running the decode entry once per token.
#[cfg(feature = "cuda")]
pub fn paged_deepseek_prefill(
    q: &Tensor,
    headers: &Tensor,
    q_pos: &Tensor,
    kv_fresh: Option<(&Tensor, usize)>,
    comp: &Tensor,
    comp_pos: &Tensor,
    comp_idx: &Tensor,
    comp_cnt: &Tensor,
    sinks: &Tensor,
    rope_freqs: &Tensor,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_deepseek_prefill requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();
    let hdr_ptr = {
        let (storage, layout) = headers.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => {
                let slice = c.as_cuda_slice::<u8>()?;
                let (p, _guard) = slice.device_ptr(&stream);
                p + layout.start_offset() as u64
            }
            _ => candle::bail!("expected CUDA storage for headers"),
        }
    };
    paged_deepseek_prefill_raw(
        q,
        hdr_ptr,
        q_pos,
        kv_fresh,
        comp,
        comp_pos,
        comp_idx,
        comp_cnt,
        sinks,
        rope_freqs,
        softmax_scale,
        window_size,
        num_splits_override,
    )
}

/// As [`paged_deepseek_prefill`] but with the slot header as a raw device
/// address (the wave path's `build_decode_metadata` form).
#[cfg(feature = "cuda")]
pub fn paged_deepseek_prefill_raw(
    q: &Tensor,
    headers_ptr: u64,
    q_pos: &Tensor,
    // This layer's just-computed latents `[fresh_rows, 512]` bf16 keyed at
    // `fresh_base + j` — the batched-wave source for tokens not yet written to
    // the arena. `None` on the settled-slot path.
    kv_fresh: Option<(&Tensor, usize)>,
    comp: &Tensor,
    comp_pos: &Tensor,
    comp_idx: &Tensor,
    comp_cnt: &Tensor,
    sinks: &Tensor,
    rope_freqs: &Tensor,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;

    let (total_q, n_q_head, hd) = q.dims3()?;
    if hd != HEAD_DIM {
        candle::bail!("paged_deepseek_prefill: head_dim {hd} != {HEAD_DIM}");
    }
    let max_sel = comp_idx.dim(1)?;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_deepseek_prefill requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();
    let out = Tensor::zeros((total_q, n_q_head, HEAD_DIM), DType::BF16, q.device())?;

    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let ptr = match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<$ty>()?;
                    let (p, _guard) = slice.device_ptr(&stream);
                    p + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                }
                _ => candle::bail!("expected CUDA storage"),
            };
            ptr
        }};
    }

    let q_ptr = cuda_ptr!(q, half::bf16);
    let hdr_ptr = headers_ptr;
    let out_ptr = cuda_ptr!(&out, half::bf16);
    let pos_ptr = cuda_ptr!(q_pos, u32);
    let (fresh_ptr, fresh_rows, fresh_base) = match kv_fresh {
        Some((t, base)) => (cuda_ptr!(t, half::bf16), t.dim(0)?, base),
        None => (0u64, 0usize, 0usize),
    };
    let comp_p = cuda_ptr!(comp, f32);
    let cpos_p = cuda_ptr!(comp_pos, u32);
    let cidx_p = cuda_ptr!(comp_idx, u32);
    let ccnt_p = cuda_ptr!(comp_cnt, u32);
    let sink_p = cuda_ptr!(sinks, f32);
    let freq_p = cuda_ptr!(rope_freqs, f32);

    unsafe {
        candle_kernels::paged_deepseek::run_paged_deepseek_prefill_bf16(
            q_ptr as *const core::ffi::c_void,
            hdr_ptr as *const u8,
            out_ptr as *mut core::ffi::c_void,
            pos_ptr as *const u32,
            fresh_ptr as *const core::ffi::c_void,
            comp_p as *const f32,
            cpos_p as *const u32,
            cidx_p as *const u32,
            ccnt_p as *const u32,
            sink_p as *const f32,
            freq_p as *const f32,
            total_q as i32,
            n_q_head as i32,
            softmax_scale,
            window_size as i32,
            max_sel as i32,
            fresh_rows as i32,
            fresh_base as i32,
            num_splits_override as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    q.device().synchronize()?;
    Ok(out)
}

/// Scatter glue latents into their RESERVED gap chunks (per-row block index +
/// in-block offset from the reprojection's descriptors). Launch BEFORE any
/// attention pass of the same layer on the same stream.
#[cfg(feature = "cuda")]
pub fn paged_deepseek_glue_scatter(
    kv: &Tensor,      // [rows, 512] bf16 pre-RoPE latents
    headers_ptr: u64, // this layer's SlotHeader (single slot)
    slices: &Tensor,  // [rows] u32 gap block index
    in_blk: &Tensor,  // [rows] u32 in-block offset
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let rows = kv.dim(0)?;
    let dev = match kv.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("glue scatter requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();
    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let ptr = match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<$ty>()?;
                    let (p, _guard) = slice.device_ptr(&stream);
                    p + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                }
                _ => candle::bail!("expected CUDA storage"),
            };
            ptr
        }};
    }
    let kv_p = cuda_ptr!(kv, half::bf16);
    let sl_p = cuda_ptr!(slices, u32);
    let ib_p = cuda_ptr!(in_blk, u32);
    unsafe {
        candle_kernels::paged_deepseek::run_paged_deepseek_glue_scatter_bf16(
            kv_p as *const core::ffi::c_void,
            headers_ptr as *const u8,
            sl_p as *const u32,
            ib_p as *const u32,
            rows as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(())
}

// ─── FP8 E4M3 (e4m3fn: no inf, max ±448) host codec ─────────────────────────
// Used to author the synthetic band arenas and by the mirror. The harness only
// feeds exactly-representable values through it, so any correct rounding
// implementation agrees byte-for-byte with the device converter.

pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f32;
    if e == 0x0F && (b & 0x07) == 0x07 {
        return f32::NAN; // e4m3fn: only S.1111.111 is NaN
    }
    if e == 0 {
        sign * (m / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return sign;
    }
    if a >= 448.0 {
        return sign | 0x7E; // saturate to ±448 (e4m3fn has no inf)
    }
    // Round to the nearest representable magnitude, ties to even mantissa.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for cand in 0u8..0x7F {
        let cv = e4m3_to_f32(cand);
        if cv.is_nan() {
            continue;
        }
        let err = (cv - a).abs();
        if err < best_err || (err == best_err && cand & 1 == 0) {
            best_err = err;
            best = cand;
        }
    }
    sign | best
}

// ─── Synthetic slot construction (rung-2 harness) ────────────────────────────

/// A hand-built single-conversation window slot: the raw `SlotHeader` /
/// `TokenSlice` / `KvHead` bytes plus the FP8 band arenas, exactly as the
/// production cache lays them out — but authored from a plain
/// `[n_tokens, 512]` f32 window with no substrate involved.
#[cfg(feature = "cuda")]
pub struct SyntheticSlots {
    /// `[n_chunks_total, N_BANDS, CHUNK*SUB_DIM]` u8 — FP8 band arenas.
    pub bands: Tensor,
    /// `[n_chunks_total * KVHEAD_BYTES]` u8.
    pub kvheads: Tensor,
    /// `[n_chunks_total * 16]` u8 — TokenSlice array (all slots concatenated).
    pub slices: Tensor,
    /// `[num_slots * 24]` u8 — SlotHeader array.
    pub headers: Tensor,
    /// Per slot: (first chunk index, n_chunks, n_tokens).
    pub slot_meta: Vec<(usize, usize, usize)>,
}

#[cfg(feature = "cuda")]
impl SyntheticSlots {
    /// Build slots from per-slot pre-RoPE window latents (each `[n_tokens][512]`
    /// f32, values FP8-representable). Chunk `c` of a slot holds tokens
    /// `[32c, 32c+32)` at rope positions starting `32c`; the last chunk is the
    /// writer (its `len` excludes the incoming token).
    pub fn build(dev: &candle::Device, windows: &[Vec<[f32; HEAD_DIM]>]) -> Result<Self> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;

        let cuda = match dev {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("SyntheticSlots requires a CUDA device"),
        };
        let stream = cuda.cuda_stream();

        let mut slot_meta = Vec::new();
        let mut chunk_base = 0usize;
        for w in windows {
            // The writer chunk must have room for the incoming token.
            let n_chunks = w.len() / CHUNK + 1;
            slot_meta.push((chunk_base, n_chunks, w.len()));
            chunk_base += n_chunks;
        }
        let n_chunks_total = chunk_base;

        // Band arenas: FP8-encode each token's 512 dims into 4 bands.
        let mut band_bytes = vec![0u8; n_chunks_total * N_BANDS * CHUNK * SUB_DIM];
        for (slot, w) in windows.iter().enumerate() {
            let (first_chunk, _, _) = slot_meta[slot];
            for (t, latent) in w.iter().enumerate() {
                let chunk = first_chunk + t / CHUNK;
                let within = t % CHUNK;
                for (d, &v) in latent.iter().enumerate() {
                    let band = d / SUB_DIM;
                    let in_band = d % SUB_DIM;
                    band_bytes
                        [(chunk * N_BANDS + band) * CHUNK * SUB_DIM + within * SUB_DIM + in_band] =
                        f32_to_e4m3(v);
                }
            }
        }
        let bands = Tensor::from_vec(band_bytes, (n_chunks_total, N_BANDS, CHUNK * SUB_DIM), dev)?;
        let bands_addr = {
            let (storage, _) = bands.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<u8>()?;
                    let (p, _g) = slice.device_ptr(&stream);
                    p
                }
                _ => unreachable!(),
            }
        };

        // KvHead records: one per chunk (n_kv_head = 1). Identity band map,
        // FP8 format tag (34), outer scales 1.0, v_* mirrors k_* (K≡V).
        let mut kvhead_bytes = vec![0u8; n_chunks_total * KVHEAD_BYTES];
        for chunk in 0..n_chunks_total {
            let rec = &mut kvhead_bytes[chunk * KVHEAD_BYTES..(chunk + 1) * KVHEAD_BYTES];
            // k_pal / v_pal: 2-bit band ids packed 4/byte, ascending dim.
            for d in 0..HEAD_DIM {
                let band = (d / SUB_DIM) as u8;
                rec[d >> 2] |= band << ((d & 3) * 2);
                rec[HEAD_DIM / 4 + (d >> 2)] |= band << ((d & 3) * 2);
            }
            for band in 0..N_BANDS {
                let addr = bands_addr + ((chunk * N_BANDS + band) * CHUNK * SUB_DIM) as u64;
                let kp = HEAD_DIM / 2 + band * 8;
                rec[kp..kp + 8].copy_from_slice(&addr.to_le_bytes());
                let vp = HEAD_DIM / 2 + 32 + band * 8;
                rec[vp..vp + 8].copy_from_slice(&addr.to_le_bytes());
                rec[HEAD_DIM / 2 + 64 + band] = 34; // ArenaFormat::F8E4M3
                rec[HEAD_DIM / 2 + 68 + band] = 34;
                let ks = HEAD_DIM / 2 + 72 + band * 4;
                rec[ks..ks + 4].copy_from_slice(&1.0f32.to_le_bytes());
                let vs = HEAD_DIM / 2 + 88 + band * 4;
                rec[vs..vs + 4].copy_from_slice(&1.0f32.to_le_bytes());
            }
        }
        let kvheads = Tensor::from_vec(kvhead_bytes, n_chunks_total * KVHEAD_BYTES, dev)?;
        let kvheads_addr = {
            let (storage, _) = kvheads.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<u8>()?;
                    let (p, _g) = slice.device_ptr(&stream);
                    p
                }
                _ => unreachable!(),
            }
        };

        // TokenSlice array: offset 0, len = tokens in chunk (writer excludes
        // the incoming token), rope = 32*chunk_in_slot.
        let mut slice_bytes = vec![0u8; n_chunks_total * 16];
        for (slot, w) in windows.iter().enumerate() {
            let (first_chunk, n_chunks, n_tokens) = slot_meta[slot];
            for c in 0..n_chunks {
                let g = first_chunk + c;
                let rec = &mut slice_bytes[g * 16..g * 16 + 16];
                let start = c * CHUNK;
                let len = n_tokens.saturating_sub(start).min(CHUNK);
                rec[0..2].copy_from_slice(&0u16.to_le_bytes()); // offset
                rec[2..4].copy_from_slice(&(len as u16).to_le_bytes());
                rec[4..8].copy_from_slice(&(start as u32).to_le_bytes()); // rope
                let ka = kvheads_addr + (g * KVHEAD_BYTES) as u64;
                rec[8..16].copy_from_slice(&ka.to_le_bytes());
            }
            let _ = w;
        }
        let slices = Tensor::from_vec(slice_bytes, n_chunks_total * 16, dev)?;
        let slices_addr = {
            let (storage, _) = slices.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<u8>()?;
                    let (p, _g) = slice.device_ptr(&stream);
                    p
                }
                _ => unreachable!(),
            }
        };

        // SlotHeaders.
        let mut header_bytes = vec![0u8; windows.len() * 24];
        for (slot, _) in windows.iter().enumerate() {
            let (first_chunk, n_chunks, _) = slot_meta[slot];
            let rec = &mut header_bytes[slot * 24..slot * 24 + 24];
            rec[0..4].copy_from_slice(&(n_chunks as u32).to_le_bytes());
            rec[4..8].copy_from_slice(&((n_chunks - 1) as u32).to_le_bytes()); // writer = last
            let sa = slices_addr + (first_chunk * 16) as u64;
            rec[8..16].copy_from_slice(&sa.to_le_bytes());
            // position_map_ptr unused by the DeepSeek kernel.
            rec[16..24].copy_from_slice(&0u64.to_le_bytes());
        }
        let headers = Tensor::from_vec(header_bytes, windows.len() * 24, dev)?;

        Ok(Self {
            bands,
            kvheads,
            slices,
            headers,
            slot_meta,
        })
    }
}

// ─── CPU mirror oracle (a): the kernel's exact arithmetic on the host ────────

/// The kernel-side `fast_exp` e^x, mirrored **instruction-for-instruction**
/// from the PTX variant that actually runs on SM80+
/// (`exp_f32_high_softmax_ptx`, fast_exp.cuh — `FAST_EXP_USE_PTX=1`): its
/// FMA-tuned cubic coefficients differ from the C-source constants, its Horner
/// steps are explicit `fma.rn`, and `cvt.rmi` is floor. Every op here maps 1:1
/// to that asm, so the replica is bit-exact. (The archive also compiles with
/// `-fmad=false`, so *implicit* contraction never appears anywhere else in the
/// kernel — plain mul/add mirrors plain mul/add.)
pub fn ds_exp_mirror(x: f32) -> f32 {
    let log2_e = f32::from_bits(0x3FB8_AA3B);
    let c3 = f32::from_bits(0x3D9D_9653);
    let c2 = f32::from_bits(0x3E69_1E05);
    let c1 = f32::from_bits(0x3F31_F70E);
    let xc = x.max(-88.0);
    let x2 = xc * log2_e;
    let xi = x2.floor(); // cvt.rmi
    let xf = x2 - xi;
    let mut poly = xf.mul_add(c3, c2); // fma.rn xf·C3 + C2
    poly = poly.mul_add(xf, c1); //       poly·xf + C1
    poly = poly.mul_add(xf, 1.0); //      poly·xf + 1
    let scale = f32::from_bits((((xi as i32) + 127) << 23) as u32);
    poly * scale
}

/// Per-band int8 quantization: returns (int8 values, band scales) exactly as
/// the kernel computes them (max-abs per 128-dim band, `mx/127`, zero→1,
/// round-nearest-even, clamp ±127).
pub fn quant_bands(v: &[f32; HEAD_DIM]) -> ([i8; HEAD_DIM], [f32; N_BANDS]) {
    let mut q = [0i8; HEAD_DIM];
    let mut scales = [1.0f32; N_BANDS];
    for band in 0..N_BANDS {
        let lo = band * SUB_DIM;
        let mut mx = 0.0f32;
        for d in lo..lo + SUB_DIM {
            mx = mx.max(v[d].abs());
        }
        let mut s = mx / 127.0;
        if s == 0.0 {
            s = 1.0;
        }
        scales[band] = s;
        let inv = 1.0 / s;
        for d in lo..lo + SUB_DIM {
            let x = (v[d] * inv).clamp(-127.0, 127.0);
            q[d] = x.round_ties_even() as i32 as i8;
        }
    }
    (q, scales)
}

/// One key's contribution to the logit: Σ_band (int32 dot) · sQ · sK, summed
/// in band order — the kernel's `((p0+p1)+p2)+p3` float chain.
pub fn mirror_logit(
    q_i8: &[i8; HEAD_DIM],
    q_s: &[f32; N_BANDS],
    k_i8: &[i8; HEAD_DIM],
    k_s: &[f32; N_BANDS],
) -> f32 {
    let mut total = 0.0f32;
    for band in 0..N_BANDS {
        let lo = band * SUB_DIM;
        let mut acc: i32 = 0;
        for d in lo..lo + SUB_DIM {
            acc += (q_i8[d] as i32) * (k_i8[d] as i32);
        }
        total += (acc as f32) * q_s[band] * k_s[band];
    }
    total
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::Device;

    const H: usize = 64;

    struct Case {
        n_win: usize,
        window_size: usize,
        g_total: usize,
        ratio: usize,
        selected: Vec<usize>,
        num_splits: usize,
        zero_rope: bool,
    }

    struct Inputs {
        window: Vec<[f32; HEAD_DIM]>, // FP8-exact stored latents
        kv_new: [f32; HEAD_DIM],      // FP8-exact incoming latent
        q: Vec<[f32; HEAD_DIM]>,      // per head, bf16-exact
        comp: Vec<[f32; HEAD_DIM]>,   // bf16-exact compressed entries
        comp_pos: Vec<u32>,
        sinks: Vec<f32>,
        freqs: Vec<f32>,
    }

    /// Deterministic pseudo-random values: FP8-exact grid k*0.25, |k| ≤ 15.
    fn fp8_exact(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let k = ((*seed >> 33) % 31) as i64 - 15;
        k as f32 * 0.25
    }

    /// bf16-exact values (round-trip through half::bf16).
    fn bf16_exact(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (((*seed >> 33) % 2001) as f32 - 1000.0) / 500.0;
        half::bf16::from_f32(v).to_f32()
    }

    fn gen_inputs(case: &Case, seed: u64) -> Inputs {
        let mut s = seed;
        let window: Vec<[f32; HEAD_DIM]> = (0..case.n_win)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let kv_new: [f32; HEAD_DIM] = std::array::from_fn(|_| fp8_exact(&mut s));
        let q: Vec<[f32; HEAD_DIM]> = (0..H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let comp: Vec<[f32; HEAD_DIM]> = (0..case.g_total)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let comp_pos: Vec<u32> = (0..case.g_total).map(|g| (g * case.ratio) as u32).collect();
        let sinks: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs: Vec<f32> = if case.zero_rope {
            vec![0.0; ROPE_DIM / 2]
        } else {
            super::super::rope::yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
                .into_iter()
                .map(|f| f as f32)
                .collect()
        };
        Inputs {
            window,
            kv_new,
            q,
            comp,
            comp_pos,
            sinks,
            freqs,
        }
    }

    /// The kernel's `rope_angle` (f64 reduction → quadrant + residual),
    /// replicated exactly — exact-rounded f64 ops are bit-identical across
    /// CPU and GPU.
    fn rope_angle(pos: u32, freq: f32) -> (f32, i32) {
        let mut a = pos as f64 * freq as f64;
        let t = (a * 0.159_154_943_091_895_35).floor();
        a -= t * 6.283_185_307_179_586;
        let q = (a * 0.636_619_772_367_581_4 + 0.5).floor();
        let r = (a - q * std::f64::consts::FRAC_PI_2) as f32;
        (r, (q as i32) & 3)
    }

    /// The kernel's `ds_sincos` polynomial, op-for-op (plain mul/add — the
    /// archive compiles `-fmad=false`).
    fn ds_sincos(r: f32, k: i32) -> (f32, f32) {
        let x2 = r * r;
        let mut sp_in = -1.951_529_589_1e-4_f32;
        sp_in = sp_in * x2 + 8.332_160_873_6e-3;
        sp_in = sp_in * x2 + -1.666_665_461_1e-1;
        let rt = r * x2;
        let sp = r + rt * sp_in;
        let mut cp_in = 2.443_315_711_809_948e-5_f32;
        cp_in = cp_in * x2 + -1.388_731_625_493_765e-3;
        cp_in = cp_in * x2 + 4.166_664_568_298_827e-2;
        let x4 = x2 * x2;
        let mut cp = 1.0 - 0.5 * x2;
        cp = cp + x4 * cp_in;
        match k {
            0 => (sp, cp),
            1 => (cp, -sp),
            2 => (-sp, -cp),
            _ => (-cp, sp),
        }
    }

    fn mirror_sincos(pos: u32, freq: f32) -> (f32, f32) {
        let (r, k) = rope_angle(pos, freq);
        ds_sincos(r, k)
    }

    fn rope_vec(v: &mut [f32; HEAD_DIM], pos: u32, freqs: &[f32]) {
        for k in 0..ROPE_DIM / 2 {
            let (s, c) = mirror_sincos(pos, freqs[k]);
            let d = NOPE_DIM + 2 * k;
            let (x0, x1) = (v[d], v[d + 1]);
            v[d] = x0 * c - x1 * s;
            v[d + 1] = x0 * s + x1 * c;
        }
    }

    /// The full CPU mirror: replicates tiling, quantization, softmax phases,
    /// split ranges, combine + sink fold + de-rotation. Returns `[H][512]` f32
    /// (pre-BF16-store).
    fn mirror(case: &Case, inp: &Inputs, softmax_scale: f32) -> Vec<[f32; HEAD_DIM]> {
        let n_win = case.n_win;
        let q_pos = n_win as u32;

        // Keys, in kernel order. Window token t lives at chunk t/32; the
        // writer's incoming token is the FP8 round-trip of kv_new.
        let n_chunks = n_win / CHUNK + 1;
        let mut chunk_lens: Vec<usize> = (0..n_chunks)
            .map(|c| n_win.saturating_sub(c * CHUNK).min(CHUNK))
            .collect();
        // slice_eff_len: writer (+1 for the scattered incoming token).
        *chunk_lens.last_mut().unwrap() += 1;

        let stored: Vec<[f32; HEAD_DIM]> = inp
            .window
            .iter()
            .map(|w| std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(w[d]))))
            .chain(std::iter::once(std::array::from_fn(|d| {
                e4m3_to_f32(f32_to_e4m3(inp.kv_new[d]))
            })))
            .collect();

        // Tile list: per chunk, tiles of 8; then compressed tiles of 8.
        //
        // The kernel quantizes int8 from the roped f32 REGISTERS (full
        // precision), while the PV reads the bf16-STAGED latent — the mirror
        // must keep both, or zero-rope tests pass (fp8 ⊂ bf16, staging is
        // identity) while live-rope diverges.
        struct Key {
            pv: [f32; HEAD_DIM],  // post-RoPE, post-bf16-staging (PV read)
            k_i8: [i8; HEAD_DIM], // int8 from the roped f32 (QK read)
            k_s: [f32; N_BANDS],
            valid: bool,
        }
        let mut tiles: Vec<Vec<Key>> = Vec::new();
        for (c, &len) in chunk_lens.iter().enumerate() {
            let n_tiles = len.div_ceil(8);
            for t in 0..n_tiles {
                let mut keys = Vec::with_capacity(8);
                for w in 0..8 {
                    let within = t * 8 + w;
                    let mut key = Key {
                        pv: [0.0; HEAD_DIM],
                        k_i8: [0; HEAD_DIM],
                        k_s: [1.0; N_BANDS],
                        valid: false,
                    };
                    if within < len {
                        let pos = (c * CHUNK + within) as u32;
                        // Sliding-window + causal bound.
                        if pos <= q_pos && pos as i64 > q_pos as i64 - case.window_size as i64 {
                            let mut l = stored[c * CHUNK + within];
                            rope_vec(&mut l, pos, &inp.freqs);
                            let (ki, ks) = quant_bands(&l); // from roped f32
                            key.k_i8 = ki;
                            key.k_s = ks;
                            key.pv = std::array::from_fn(|d| half::bf16::from_f32(l[d]).to_f32());
                            key.valid = true;
                        }
                    }
                    keys.push(key);
                }
                tiles.push(keys);
            }
        }
        let n_comp_tiles = case.selected.len().div_ceil(8);
        for t in 0..n_comp_tiles {
            let mut keys = Vec::with_capacity(8);
            for w in 0..8 {
                let e = t * 8 + w;
                let mut key = Key {
                    pv: [0.0; HEAD_DIM],
                    k_i8: [0; HEAD_DIM],
                    k_s: [1.0; N_BANDS],
                    valid: false,
                };
                if e < case.selected.len() {
                    let gid = case.selected[e];
                    let mut l = inp.comp[gid];
                    rope_vec(&mut l, inp.comp_pos[gid], &inp.freqs);
                    let (ki, ks) = quant_bands(&l);
                    key.k_i8 = ki;
                    key.k_s = ks;
                    key.pv = std::array::from_fn(|d| half::bf16::from_f32(l[d]).to_f32());
                    key.valid = true;
                }
                keys.push(key);
            }
            tiles.push(keys);
        }

        // Q: rope at q_pos, band quant.
        let q_prep: Vec<([i8; HEAD_DIM], [f32; N_BANDS])> = inp
            .q
            .iter()
            .map(|qh| {
                let mut v = *qh;
                rope_vec(&mut v, q_pos, &inp.freqs);
                quant_bands(&v)
            })
            .collect();

        // Per-split flash accumulation over the split's tile range.
        let n_tiles = tiles.len();
        let num_splits = case.num_splits;
        let tiles_per_split = n_tiles.div_ceil(num_splits);
        let mut out = vec![[0.0f32; HEAD_DIM]; H];

        for (h, out_h) in out.iter_mut().enumerate() {
            let mut partials: Vec<(f32, f32, [f32; HEAD_DIM])> = Vec::new();
            for split in 0..num_splits {
                let lo = (split * tiles_per_split).min(n_tiles);
                let hi = (lo + tiles_per_split).min(n_tiles);
                let mut m = -1e38f32;
                let mut l = 0.0f32;
                let mut acc = [0.0f32; HEAD_DIM];
                for tile in &tiles[lo..hi] {
                    // Key quant + logits.
                    let mut sc = [0.0f32; 8];
                    let mut tile_max = -1e38f32;
                    for (t, key) in tile.iter().enumerate() {
                        let s = if key.valid {
                            mirror_logit(&q_prep[h].0, &q_prep[h].1, &key.k_i8, &key.k_s)
                                * softmax_scale
                        } else {
                            -1e38
                        };
                        sc[t] = s;
                        tile_max = tile_max.max(s);
                    }
                    let new_m = m.max(tile_max);
                    let alpha = ds_exp_mirror(m - new_m);
                    l *= alpha;
                    for a in acc.iter_mut() {
                        *a *= alpha;
                    }
                    for (t, key) in tile.iter().enumerate() {
                        let beta = if sc[t] > -1e37 {
                            ds_exp_mirror(sc[t] - new_m)
                        } else {
                            0.0
                        };
                        l += beta;
                        for d in 0..HEAD_DIM {
                            acc[d] = beta.mul_add(key.pv[d], acc[d]);
                        }
                    }
                    m = new_m;
                }
                partials.push((m, l, acc));
            }

            // Combine + sink fold + de-rotation (per dim).
            let mut gm = -1e38f32;
            for &(m, _, _) in &partials {
                gm = gm.max(m);
            }
            let sink = inp.sinks[h];
            let m_fin = gm.max(sink);
            let mut val = [0.0f32; HEAD_DIM];
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                let mut ll = 0.0f32;
                for &(m, l, ref a) in &partials {
                    if !(m > -1e37) {
                        continue;
                    }
                    let w = ds_exp_mirror(m - m_fin);
                    acc = a[d].mul_add(w, acc);
                    ll = l.mul_add(w, ll);
                }
                ll += ds_exp_mirror(sink - m_fin);
                val[d] = acc / ll.max(1e-10);
            }
            // Inverse rotation at q_pos.
            for k in 0..ROPE_DIM / 2 {
                let (s, c) = mirror_sincos(q_pos, inp.freqs[k]);
                let d = NOPE_DIM + 2 * k;
                let (x0, x1) = (val[d], val[d + 1]);
                val[d] = x0 * c + x1 * s;
                val[d + 1] = x1 * c - x0 * s;
            }
            *out_h = val;
        }
        out
    }

    /// Oracle (b): plain float sink-softmax attention over the same inputs.
    fn float_reference(case: &Case, inp: &Inputs, softmax_scale: f32) -> Vec<[f32; HEAD_DIM]> {
        let q_pos = case.n_win as u32;
        let mut keys: Vec<[f32; HEAD_DIM]> = Vec::new();
        for (t, w) in inp
            .window
            .iter()
            .chain(std::iter::once(&inp.kv_new))
            .enumerate()
        {
            let pos = t as u32;
            if pos <= q_pos && pos as i64 > q_pos as i64 - case.window_size as i64 {
                let mut l: [f32; HEAD_DIM] =
                    std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(w[d])));
                rope_vec(&mut l, pos, &inp.freqs);
                keys.push(l);
            }
        }
        for &gid in &case.selected {
            let mut l = inp.comp[gid];
            rope_vec(&mut l, inp.comp_pos[gid], &inp.freqs);
            keys.push(l);
        }

        let mut out = vec![[0.0f32; HEAD_DIM]; H];
        for h in 0..H {
            let mut qv = inp.q[h];
            rope_vec(&mut qv, q_pos, &inp.freqs);
            let mut logits: Vec<f32> = keys
                .iter()
                .map(|k| {
                    let mut acc = 0.0f64;
                    for d in 0..HEAD_DIM {
                        acc += qv[d] as f64 * k[d] as f64;
                    }
                    (acc as f32) * softmax_scale
                })
                .collect();
            logits.push(inp.sinks[h]); // sink column
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f64> = logits.iter().map(|&v| ((v - m) as f64).exp()).collect();
            let z: f64 = exps.iter().sum();
            let mut val = [0.0f32; HEAD_DIM];
            for (k, key) in keys.iter().enumerate() {
                let p = (exps[k] / z) as f32;
                for d in 0..HEAD_DIM {
                    val[d] += p * key[d];
                }
            }
            for k in 0..ROPE_DIM / 2 {
                let (s, c) = mirror_sincos(q_pos, inp.freqs[k]);
                let d = NOPE_DIM + 2 * k;
                let (x0, x1) = (val[d], val[d + 1]);
                val[d] = x0 * c + x1 * s;
                val[d + 1] = x1 * c - x0 * s;
            }
            out[h] = val;
        }
        out
    }

    fn run_kernel(case: &Case, inp: &Inputs, softmax_scale: f32) -> Result<Vec<f32>> {
        let dev = Device::new_cuda(0)?;
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&inp.window))?;

        let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
        let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let kvf: Vec<f32> = inp.kv_new.to_vec();
        let kv_new = Tensor::from_vec(kvf, (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
        let comp = Tensor::from_vec(compf, (case.g_total.max(1), HEAD_DIM), &dev)?;
        let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total.max(1), &dev)?;
        let max_sel = case.selected.len().max(1);
        let mut idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
        idx.resize(max_sel, u32::MAX);
        let comp_idx = Tensor::from_vec(idx, (1, max_sel), &dev)?;
        let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
        let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;

        let out = paged_deepseek_decode(
            &q,
            &slots.headers,
            &kv_new,
            &comp,
            &comp_pos,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &freqs,
            softmax_scale,
            case.window_size,
            case.num_splits,
            None,
        )?;
        out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }

    fn bf16_round_host(vals: &[f32]) -> Vec<f32> {
        vals.iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect()
    }

    /// Rung-2 prefill gate: row `i` of the prefill kernel over a settled slot
    /// holding tokens `[0..n)` must be BIT-IDENTICAL to the decode kernel run
    /// with window `[0..i)` + incoming token `i` — the prefill inherits the
    /// decode's whole proven oracle chain (mirror, float reference, arena
    /// equivalence) row by row.
    #[test]
    #[ignore]
    fn prefill_rows_equal_decode_steps() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 50usize;
        let window_size = 128usize;
        let mut s = 42u64;
        let dev = Device::new_cuda(0)?;

        let tokens: Vec<[f32; HEAD_DIM]> = (0..n)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let qs: Vec<[f32; HEAD_DIM]> = (0..n * H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let sinks_v: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs_v: Vec<f32> =
            super::super::rope::yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
                .into_iter()
                .map(|f| f as f32)
                .collect();
        let sinks = Tensor::from_vec(sinks_v.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;

        // Prefill: one settled slot holding all n tokens.
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&tokens.to_vec()))?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;
        let prefill = paged_deepseek_prefill(
            &q_all,
            &slots.headers,
            &q_pos,
            None,
            &comp,
            &comp_pos,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &freqs,
            softmax_scale,
            window_size,
            1,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // Fresh-source equivalence: arena holds only [0..k); rows [k..n) come
        // in as this-layer fresh latents. Queries [k..n) must be BIT-IDENTICAL
        // to the settled-slot run (the kernel FP8-round-trips fresh keys so the
        // bits match what the arena would return).
        let split_at = 20usize;
        let fslots =
            SyntheticSlots::build(&dev, std::slice::from_ref(&tokens[..split_at].to_vec()))?;
        let fresh_vals: Vec<f32> = tokens[split_at..]
            .iter()
            .flat_map(|t| t.iter().copied())
            .collect();
        let kv_fresh =
            Tensor::from_vec(fresh_vals, (n - split_at, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let qf_tail: Vec<f32> = qs[split_at * H..]
            .iter()
            .flat_map(|h| h.iter().copied())
            .collect();
        let q_tail =
            Tensor::from_vec(qf_tail, (n - split_at, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let pos_tail = Tensor::from_vec(
            (split_at as u32..n as u32).collect::<Vec<_>>(),
            n - split_at,
            &dev,
        )?;
        let idx_tail = Tensor::full(u32::MAX, (n - split_at, 1), &dev)?;
        let cnt_tail = Tensor::zeros(n - split_at, DType::U32, &dev)?;
        let fresh_out = paged_deepseek_prefill(
            &q_tail,
            &fslots.headers,
            &pos_tail,
            Some((&kv_fresh, split_at)),
            &comp,
            &comp_pos,
            &idx_tail,
            &cnt_tail,
            &sinks,
            &freqs,
            softmax_scale,
            window_size,
            1,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        // Key VALUES are bit-identical (the kernel FP8-round-trips fresh keys),
        // but the 8-key TILE PARTITION shifts (the arena's 20-token chunk pads
        // its last tile; fresh rows tile separately), so the tile-batched
        // softmax accumulates in a different association order — the benign
        // few-ulp variance any split-KV repartition has. Gate: anchor BOTH
        // runs against an f64 sink-softmax reference over the same
        // FP8-round-tripped keys — the fresh path must sit no farther from
        // truth than the settled path does.
        let settled_tail = &prefill[split_at * H * HEAD_DIM..];
        let stored: Vec<[f32; HEAD_DIM]> = tokens
            .iter()
            .map(|t| std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(t[d]))))
            .collect();
        let mut ref_tail: Vec<f32> = Vec::with_capacity((n - split_at) * H * HEAD_DIM);
        for i in split_at..n {
            let mut keys: Vec<[f32; HEAD_DIM]> = Vec::with_capacity(i + 1);
            for (t, tok) in stored.iter().enumerate().take(i + 1) {
                let mut l = *tok;
                rope_vec(&mut l, t as u32, &freqs_v);
                keys.push(l);
            }
            for h in 0..H {
                let mut qv = qs[i * H + h];
                rope_vec(&mut qv, i as u32, &freqs_v);
                let mut logits: Vec<f64> = keys
                    .iter()
                    .map(|k| {
                        let mut acc = 0.0f64;
                        for d in 0..HEAD_DIM {
                            acc += qv[d] as f64 * k[d] as f64;
                        }
                        acc * softmax_scale as f64
                    })
                    .collect();
                logits.push(sinks_v[h] as f64);
                let m = logits.iter().cloned().fold(f64::MIN, f64::max);
                let exps: Vec<f64> = logits.iter().map(|&v| (v - m).exp()).collect();
                let z: f64 = exps.iter().sum();
                let mut val = [0.0f64; HEAD_DIM];
                for (t, key) in keys.iter().enumerate() {
                    let p = exps[t] / z;
                    for d in 0..HEAD_DIM {
                        val[d] += p * key[d] as f64;
                    }
                }
                for k in 0..ROPE_DIM / 2 {
                    let (s, c) = mirror_sincos(i as u32, freqs_v[k]);
                    let d = NOPE_DIM + 2 * k;
                    let (x0, x1) = (val[d], val[d + 1]);
                    val[d] = x0 * c as f64 + x1 * s as f64;
                    val[d + 1] = x1 * c as f64 - x0 * s as f64;
                }
                ref_tail.extend(val.iter().map(|&v| v as f32));
            }
        }
        let d_settled = max_abs_diff(settled_tail, &ref_tail);
        let d_fresh = max_abs_diff(&fresh_out, &ref_tail);
        eprintln!(
            "[fresh] settled-vs-ref {d_settled:.6}, fresh-vs-ref {d_fresh:.6}, \
             settled-vs-fresh {:.6}",
            max_abs_diff(settled_tail, &fresh_out)
        );
        assert!(
            d_fresh <= d_settled * 1.5 + 1e-3,
            "fresh path drifts beyond the settled path's own truth distance: \
             fresh {d_fresh} vs settled {d_settled}"
        );

        // Decode oracle per row (spot rows — each is a full kernel launch).
        for &i in &[0usize, 1, 17, 31, 32, 33, 49] {
            let window: Vec<[f32; HEAD_DIM]> = tokens[..i].to_vec();
            let dslots = SyntheticSlots::build(&dev, std::slice::from_ref(&window))?;
            let qi: Vec<f32> = qs[i * H..(i + 1) * H]
                .iter()
                .flat_map(|h| h.iter().copied())
                .collect();
            let q_one = Tensor::from_vec(qi, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kv_new =
                Tensor::from_vec(tokens[i].to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let d_idx = Tensor::full(u32::MAX, (1, 1), &dev)?;
            let d_cnt = Tensor::zeros(1, DType::U32, &dev)?;
            let dec = paged_deepseek_decode(
                &q_one,
                &dslots.headers,
                &kv_new,
                &comp,
                &comp_pos,
                &d_idx,
                &d_cnt,
                &sinks,
                &freqs,
                softmax_scale,
                window_size,
                1,
                None,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            let mismatches = row
                .iter()
                .zip(&dec)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "row {i}: {mismatches}/{} bits differ (max |Δ| = {})",
                dec.len(),
                max_abs_diff(row, &dec)
            );
        }
        Ok(())
    }

    fn run_case(case: &Case, seed: u64) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(case, seed);
        let kernel = run_kernel(case, &inp, softmax_scale)?;
        let mirror_out: Vec<f32> = mirror(case, &inp, softmax_scale)
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        let reference: Vec<f32> = float_reference(case, &inp, softmax_scale)
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        Ok((kernel, bf16_round_host(&mirror_out), reference))
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Oracle (a1): zero-rope, splits=1 — every arithmetic path except trig,
    /// gated BIT-EXACT (the mirror reproduces the kernel to the bf16 bit).
    #[test]
    #[ignore]
    fn mirror_bit_exact_zero_rope() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: true,
        };
        let (kernel, mirror_v, _) = run_case(&case, 42)?;
        let mismatches: Vec<(usize, usize, f32, f32)> = kernel
            .iter()
            .zip(&mirror_v)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i / HEAD_DIM, i % HEAD_DIM, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "bit mismatches: {}/{} (max |Δ| = {}); first 8 (head, dim, kernel, mirror): {:?}",
            mismatches.len(),
            kernel.len(),
            max_abs_diff(&kernel, &mirror_v),
            &mismatches[..mismatches.len().min(8)]
        );
        Ok(())
    }

    /// The device `ds_exp` and the CPU mirror replica must agree bit-for-bit
    /// across the softmax input range — the foundation every bit-exact gate
    /// stands on.
    #[test]
    #[ignore]
    fn ds_exp_device_matches_mirror() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let dev = Device::new_cuda(0)?;
        let cuda = match &dev {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = cuda.cuda_stream();
        // Dense sweep of the softmax-relevant range plus edge values.
        let mut xs: Vec<f32> = (0..200_000).map(|i| -50.0 + i as f32 * 0.00025).collect();
        xs.extend_from_slice(&[0.0, -0.0, -1e38, -88.0, -87.999, f32::MIN_POSITIVE]);
        let n = xs.len();
        let input = Tensor::from_vec(xs.clone(), n, &dev)?;
        let out = Tensor::zeros(n, DType::F32, &dev)?;
        {
            let (si, _) = input.storage_and_layout();
            let (so, _) = out.storage_and_layout();
            let (ip, _g1) = match &*si {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            let (op, _g2) = match &*so {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            unsafe {
                candle_kernels::paged_deepseek::run_deepseek_exp_probe(
                    ip as *const f32,
                    op as *mut f32,
                    n as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
            dev.synchronize()?;
        }
        let got = out.to_vec1::<f32>()?;
        let mismatches: Vec<(f32, f32, f32)> = xs
            .iter()
            .zip(&got)
            .filter(|(&x, &g)| g.to_bits() != ds_exp_mirror(x).to_bits())
            .map(|(&x, &g)| (x, g, ds_exp_mirror(x)))
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "ds_exp device≠mirror; first (x, device, mirror): {mismatches:?}"
        );
        Ok(())
    }

    /// Step-2 gate (a): FP8 window latents written through the PRODUCTION
    /// chunked backing (single-latent mode) read back exactly as the host FP8
    /// round-trip — the raw-byte write→read contract.
    #[test]
    #[ignore]
    fn arena_write_read_round_trip() -> Result<()> {
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};
        let dev = Device::new_cuda(0)?;
        let backing = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            HEAD_DIM,
            KvFormat::Float(DType::F8E4M3),
            KvFormat::Float(DType::F8E4M3),
            &dev,
            256,
            None,
        )?;
        backing.set_single_latent(true);

        let mut s = 7u64;
        let n = 50usize;
        let vals: Vec<f32> = (0..n * HEAD_DIM).map(|_| fp8_exact(&mut s)).collect();
        let latent = Tensor::from_vec(vals.clone(), (1, 1, n, HEAD_DIM), &dev)?;
        // The production calling convention: KvCache over the chunked backing,
        // write then commit the length.
        let mut cache = candle_nn::kv_cache::KvCache::new(2, 256);
        cache.set_chunked_backing(&backing, 0, None)?;
        cache.chunked_write_kv(0, &latent, &latent)?;
        cache.set_current_seq_len(n)?;

        let (rk, rv) = cache.chunked_read_kv(0, n)?;
        let got_k = rk.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let got_v = rv.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for (i, &v) in vals.iter().enumerate() {
            let expect = e4m3_to_f32(f32_to_e4m3(v));
            assert_eq!(
                got_k[i].to_bits(),
                expect.to_bits(),
                "K[{i}]: {} vs {expect}",
                got_k[i]
            );
            // K≡V: the V read aliases the K bytes.
            assert_eq!(got_v[i].to_bits(), expect.to_bits(), "V[{i}] alias");
        }
        Ok(())
    }

    /// Step-2 gate (b): the decode kernel over PRODUCTION-built slot tables
    /// (backing + `build_decode_metadata`) produces bit-identical output to
    /// the same case over the hand-built `SyntheticSlots`.
    #[test]
    #[ignore]
    fn arena_backed_matches_synthetic() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, BatchedInferenceSession};
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};

        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(&case, 42);

        // Synthetic-slot run (the rung-2 baseline).
        let synth = run_kernel(&case, &inp, softmax_scale)?;

        // Production-backing run over the same inputs.
        let dev = Device::new_cuda(0)?;
        let backing = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            HEAD_DIM,
            KvFormat::Float(DType::F8E4M3),
            KvFormat::Float(DType::F8E4M3),
            &dev,
            256,
            None,
        )?;
        backing.set_single_latent(true);
        let cfg = BatchedConfig {
            k_format: KvFormat::Float(DType::F8E4M3),
            v_format: KvFormat::Float(DType::F8E4M3),
            initial_seq_len: 256,
            ..Default::default()
        };
        let mut session =
            BatchedInferenceSession::new_with_backings(vec![backing.clone()], cfg, &dev);
        let seq = session.create_sequence()?;

        let flat: Vec<f32> = inp.window.iter().flat_map(|w| w.iter().copied()).collect();
        let latent = Tensor::from_vec(flat, (1, 1, case.n_win, HEAD_DIM), &dev)?;
        let mut cache = candle_nn::kv_cache::KvCache::new(2, 256);
        cache.set_chunked_backing(&backing, seq, None)?;
        cache.chunked_write_kv(0, &latent, &latent)?;
        cache.set_current_seq_len(case.n_win)?;
        session.set_sequence_offset(seq, case.n_win)?;

        let generation = session.begin_stager_generation();
        let (_pm, headers, _stride) = session.build_decode_metadata(&[seq], &generation)?;
        let headers = headers.expect("decode metadata headers");

        let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
        let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let kv_new =
            Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
        let comp = Tensor::from_vec(compf, (case.g_total, HEAD_DIM), &dev)?;
        let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total, &dev)?;
        let max_sel = case.selected.len();
        let idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
        let comp_idx = Tensor::from_vec(idx, (1, max_sel), &dev)?;
        let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
        let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;

        let out = paged_deepseek_decode_raw(
            &q,
            headers.dev_ptr(),
            &kv_new,
            &comp,
            &comp_pos,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &freqs,
            softmax_scale,
            case.window_size,
            case.num_splits,
            true, // single-step audit against the live buffer
            None,
        )?;
        drop(generation);
        let arena_out = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let mismatches = synth
            .iter()
            .zip(&arena_out)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "arena-backed vs synthetic: {mismatches}/{} bit mismatches (max |Δ| = {})",
            synth.len(),
            max_abs_diff(&synth, &arena_out)
        );
        Ok(())
    }

    /// The kernel's RoPE trig and the CPU mirror must agree bit-for-bit over
    /// the (position, frequency) grid the tests exercise.
    #[test]
    #[ignore]
    fn sincos_device_matches_mirror() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let dev = Device::new_cuda(0)?;
        let cuda = match &dev {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = cuda.cuda_stream();
        let freqs: Vec<f32> = super::super::rope::yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let mut pos_v: Vec<i32> = Vec::new();
        let mut freq_v: Vec<f32> = Vec::new();
        for pos in (0..200).chain([1_000, 100_000, 1_000_000]) {
            for &f in &freqs {
                pos_v.push(pos);
                freq_v.push(f);
            }
        }
        let n = pos_v.len();
        let pos_t = Tensor::from_vec(pos_v.iter().map(|&p| p as u32).collect::<Vec<_>>(), n, &dev)?;
        let freq_t = Tensor::from_vec(freq_v.clone(), n, &dev)?;
        let out = Tensor::zeros(2 * n, DType::F32, &dev)?;
        {
            let (sp, _) = pos_t.storage_and_layout();
            let (sf, _) = freq_t.storage_and_layout();
            let (so, _) = out.storage_and_layout();
            let (pp, _g1) = match &*sp {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            let (fp, _g2) = match &*sf {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            let (op, _g3) = match &*so {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            unsafe {
                candle_kernels::paged_deepseek::run_deepseek_sincos_probe(
                    pp as *const i32,
                    fp as *const f32,
                    op as *mut f32,
                    n as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
            dev.synchronize()?;
        }
        let got = out.to_vec1::<f32>()?;
        let mismatches: Vec<(i32, f32, (f32, f32), (f32, f32))> = (0..n)
            .filter_map(|i| {
                let (ms, mc) = mirror_sincos(pos_v[i] as u32, freq_v[i]);
                let (gs, gc) = (got[2 * i], got[2 * i + 1]);
                if gs.to_bits() != ms.to_bits() || gc.to_bits() != mc.to_bits() {
                    Some((pos_v[i], freq_v[i], (gs, gc), (ms, mc)))
                } else {
                    None
                }
            })
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "sincos device≠mirror; first (pos, freq, device(s,c), mirror(s,c)): {mismatches:?}"
        );
        Ok(())
    }

    /// Stage-by-stage divergence localizer: runs the tiny live-rope case with
    /// the kernel's stage dump and compares each staged quantity against the
    /// mirror — the first diverging stage names the bug.
    #[test]
    #[ignore]
    fn mirror_stage_dump_rope() -> Result<()> {
        let case = Case {
            n_win: 1,
            window_size: 128,
            g_total: 0,
            ratio: 4,
            selected: vec![],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(&case, 42);
        let dev = Device::new_cuda(0)?;
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&inp.window))?;
        let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
        let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let kv_new =
            Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (1, 1), &dev)?;
        let comp_cnt = Tensor::zeros(1, DType::U32, &dev)?;
        let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
        let dbg = Tensor::zeros(16608, DType::F32, &dev)?;
        let _ = paged_deepseek_decode(
            &q,
            &slots.headers,
            &kv_new,
            &comp,
            &comp_pos,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &freqs,
            softmax_scale,
            case.window_size,
            1,
            Some(&dbg),
        )?;
        let d = dbg.to_vec1::<f32>()?;

        // Mirror stage quantities for tile 0 (keys: window token pos 0 + writer
        // pos 1) and heads 0..16.
        let stored: Vec<[f32; HEAD_DIM]> = inp
            .window
            .iter()
            .map(|w| std::array::from_fn(|dd| e4m3_to_f32(f32_to_e4m3(w[dd]))))
            .chain(std::iter::once(std::array::from_fn(|dd| {
                e4m3_to_f32(f32_to_e4m3(inp.kv_new[dd]))
            })))
            .collect();
        let q_pos = case.n_win as u32;

        // Q stage.
        for h in 0..16 {
            let mut v = inp.q[h];
            rope_vec(&mut v, q_pos, &inp.freqs);
            let (qi, qs) = quant_bands(&v);
            for p in 0..N_BANDS {
                let got = d[h * N_BANDS + p];
                assert_eq!(
                    got.to_bits(),
                    qs[p].to_bits(),
                    "scaleQ[{h}][{p}]: device {got} vs mirror {}",
                    qs[p]
                );
            }
            for dd in 0..HEAD_DIM {
                let got = d[64 + h * HEAD_DIM + dd] as i32 as i8;
                assert_eq!(
                    got, qi[dd],
                    "sQ[{h}][{dd}]: device {got} vs mirror {}",
                    qi[dd]
                );
            }
        }

        // K stage (tile 0: both keys). int8 quant from the roped f32; the
        // staged bf16 is the PV read.
        for t in 0..2usize {
            let mut l = stored[t];
            rope_vec(&mut l, t as u32, &inp.freqs);
            let mut staged = l;
            for x in staged.iter_mut() {
                *x = half::bf16::from_f32(*x).to_f32();
            }
            let (ki, ks) = quant_bands(&l);
            for dd in 0..HEAD_DIM {
                let got = d[12384 + t * HEAD_DIM + dd];
                assert_eq!(
                    got.to_bits(),
                    staged[dd].to_bits(),
                    "kv_f[{t}][{dd}]: device {got} vs mirror {}",
                    staged[dd]
                );
            }
            for p in 0..N_BANDS {
                let got = d[8256 + t * N_BANDS + p];
                assert_eq!(
                    got.to_bits(),
                    ks[p].to_bits(),
                    "scaleK[{t}][{p}]: device {got} vs mirror {}",
                    ks[p]
                );
            }
            for dd in 0..HEAD_DIM {
                let got = d[8288 + t * HEAD_DIM + dd] as i32 as i8;
                assert_eq!(
                    got, ki[dd],
                    "sK[{t}][{dd}]: device {got} vs mirror {}",
                    ki[dd]
                );
            }
        }
        Ok(())
    }

    /// Bisection probe: grows the key count from 1 (writer only) upward to
    /// localize any mirror divergence to the smallest failing configuration.
    #[test]
    #[ignore]
    fn mirror_probe_minimal() -> Result<()> {
        for (n_win, zero_rope) in [
            (0usize, true),
            (1, true),
            (3, true),
            (8, true),
            (9, true),
            (1, false),
            (3, false),
            (9, false),
        ] {
            let case = Case {
                n_win,
                window_size: 128,
                g_total: 0,
                ratio: 4,
                selected: vec![],
                num_splits: 1,
                zero_rope,
            };
            let (kernel, mirror_v, _) = run_case(&case, 42)?;
            let mismatches: Vec<(usize, usize, f32, f32)> = kernel
                .iter()
                .zip(&mirror_v)
                .enumerate()
                .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                .map(|(i, (a, b))| (i / HEAD_DIM, i % HEAD_DIM, *a, *b))
                .collect();
            if !mismatches.is_empty() {
                let (kernel, mirror_v, reference) = run_case(&case, 42)?;
                eprintln!(
                    "[probe] n_win={n_win} zero_rope={zero_rope}: kernel-vs-ref {:.6}, mirror-vs-ref {:.6}, kernel-vs-mirror {:.6}",
                    max_abs_diff(&kernel, &reference),
                    max_abs_diff(&mirror_v, &reference),
                    max_abs_diff(&kernel, &mirror_v),
                );
            }
            assert!(
                mismatches.is_empty(),
                "n_win={n_win} zero_rope={zero_rope}: {} mismatches; first 4: {:?}",
                mismatches.len(),
                &mismatches[..mismatches.len().min(4)]
            );
        }
        Ok(())
    }

    /// Oracle (a1) under split-KV (3 splits + one empty-split config): the
    /// combine's split merge must stay bit-exact.
    #[test]
    #[ignore]
    fn mirror_bit_exact_zero_rope_splits() -> Result<()> {
        for num_splits in [3usize, 5] {
            let case = Case {
                n_win: 50,
                window_size: 128,
                g_total: 10,
                ratio: 4,
                selected: vec![0, 2, 5, 8],
                num_splits,
                zero_rope: true,
            };
            let (kernel, mirror_v, _) = run_case(&case, 7)?;
            let n_mismatch = kernel
                .iter()
                .zip(&mirror_v)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                n_mismatch,
                0,
                "splits={num_splits}: bit mismatches {n_mismatch} (max |Δ| = {})",
                max_abs_diff(&kernel, &mirror_v)
            );
        }
        Ok(())
    }

    /// Oracle (a1) with a tight sliding window (the position clamp) and no
    /// compressed entries (pure-SWA layer shape).
    #[test]
    #[ignore]
    fn mirror_bit_exact_sliding_window_swa() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 16,
            g_total: 0,
            ratio: 4,
            selected: vec![],
            num_splits: 1,
            zero_rope: true,
        };
        let (kernel, mirror_v, _) = run_case(&case, 3)?;
        let n_mismatch = kernel
            .iter()
            .zip(&mirror_v)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            n_mismatch,
            0,
            "bit mismatches {n_mismatch} (max |Δ| = {})",
            max_abs_diff(&kernel, &mirror_v)
        );
        Ok(())
    }

    /// Oracle (a2): live RoPE, gated **bit-exact** — the kernel's trig is the
    /// mirrorable `rope_angle`/`ds_sincos` pair (f64 reduction + plain-f32
    /// minimax polynomials), so RoPE introduces no unmirrorable operation.
    #[test]
    #[ignore]
    fn mirror_bit_exact_with_rope() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 2,
            zero_rope: false,
        };
        let (kernel, mirror_v, _) = run_case(&case, 42)?;
        let mismatches: Vec<(usize, usize, f32, f32)> = kernel
            .iter()
            .zip(&mirror_v)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i / HEAD_DIM, i % HEAD_DIM, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "rope bit mismatches: {}/{}; first 8: {:?}",
            mismatches.len(),
            kernel.len(),
            &mismatches[..mismatches.len().min(8)]
        );
        Ok(())
    }

    /// Oracle (b): the float sink-softmax reference at int8-scale tolerance —
    /// the model-quality gate.
    #[test]
    #[ignore]
    fn float_reference_tolerance() -> Result<()> {
        for (seed, zero_rope) in [(42u64, false), (11, true)] {
            let case = Case {
                n_win: 50,
                window_size: 128,
                g_total: 10,
                ratio: 4,
                selected: vec![1, 3, 7],
                num_splits: 1,
                zero_rope,
            };
            let (kernel, _, reference) = run_case(&case, seed)?;
            let scale = reference.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = max_abs_diff(&kernel, &reference);
            assert!(
                d < 0.03 * scale.max(1.0),
                "zero_rope={zero_rope}: |Δ| = {d} vs scale {scale}"
            );
        }
        Ok(())
    }

    /// Glue-scatter readback gate: a latent written into its reserved gap slot
    /// by `paged_deepseek_glue_scatter` must be BIT-IDENTICAL, through the
    /// decode kernel, to the same latent authored host-side into the arena.
    /// This is the read-after-scatter-through-attention proof the wave path's
    /// phase-A glue relies on: scatter targets, band pointer math, FP8
    /// conversion, and window visibility of the scattered token.
    #[test]
    #[ignore]
    fn glue_scatter_equals_authored_window() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;

        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 40usize; // 2 chunks: [0..32) settled + [32..40) writer
        let glue_slots = [3usize, 35]; // one per chunk, mid-chunk targets
        let mut s = 7u64;
        let dev = Device::new_cuda(0)?;

        let tokens: Vec<[f32; HEAD_DIM]> = (0..n)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let q: Vec<[f32; HEAD_DIM]> = (0..H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let kv_new: [f32; HEAD_DIM] = std::array::from_fn(|_| fp8_exact(&mut s));
        let sinks_v: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs_v: Vec<f32> =
            super::super::rope::yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
                .into_iter()
                .map(|f| f as f32)
                .collect();

        let run = |window: &Vec<[f32; HEAD_DIM]>, scatter: bool| -> Result<Vec<f32>> {
            let slots = SyntheticSlots::build(&dev, std::slice::from_ref(window))?;
            if scatter {
                let cuda = match &dev {
                    Device::Cuda(c) => c.clone(),
                    _ => unreachable!(),
                };
                let stream = cuda.cuda_stream();
                let headers_addr = {
                    let (storage, _) = slots.headers.storage_and_layout();
                    match &*storage {
                        Storage::Cuda(c) => {
                            let slice = c.as_cuda_slice::<u8>()?;
                            let (p, _g) = slice.device_ptr(&stream);
                            p
                        }
                        _ => unreachable!(),
                    }
                };
                let mut lat = Vec::with_capacity(glue_slots.len() * HEAD_DIM);
                for &g in &glue_slots {
                    lat.extend_from_slice(&tokens[g]);
                }
                let kv = Tensor::from_vec(lat, (glue_slots.len(), HEAD_DIM), &dev)?
                    .to_dtype(DType::BF16)?;
                let slices: Vec<u32> = glue_slots.iter().map(|&g| (g / CHUNK) as u32).collect();
                let in_blk: Vec<u32> = glue_slots.iter().map(|&g| (g % CHUNK) as u32).collect();
                let slices = Tensor::from_vec(slices, glue_slots.len(), &dev)?;
                let in_blk = Tensor::from_vec(in_blk, glue_slots.len(), &dev)?;
                paged_deepseek_glue_scatter(&kv, headers_addr, &slices, &in_blk)?;
            }
            let qf: Vec<f32> = q.iter().flat_map(|h| h.iter().copied()).collect();
            let qt = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kvt =
                Tensor::from_vec(kv_new.to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
            let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
            let comp_idx = Tensor::from_vec(vec![u32::MAX], (1, 1), &dev)?;
            let comp_cnt = Tensor::from_vec(vec![0u32], 1, &dev)?;
            let sinks = Tensor::from_vec(sinks_v.clone(), H, &dev)?;
            let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
            let out = paged_deepseek_decode(
                &qt,
                &slots.headers,
                &kvt,
                &comp,
                &comp_pos,
                &comp_idx,
                &comp_cnt,
                &sinks,
                &freqs,
                softmax_scale,
                128,
                2,
                None,
            )?;
            out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
        };

        // Path A: every token authored host-side.
        let authored = run(&tokens, false)?;
        // Path B: the glue slots authored as ZERO, then filled by the scatter
        // kernel from the true latents.
        let mut gapped = tokens.clone();
        for &g in &glue_slots {
            gapped[g] = [0.0; HEAD_DIM];
        }
        let scattered = run(&gapped, true)?;

        let mismatches = authored
            .iter()
            .zip(&scattered)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "scattered glue diverges from authored window: {mismatches}/{} values",
            authored.len()
        );
        // Sanity: the gap actually mattered (the zeroed run differs), so the
        // scatter path — not luck — produced the agreement.
        let unfilled = run(&gapped, false)?;
        assert_ne!(
            authored, unfilled,
            "test degenerate: zeroed glue slots did not change the output"
        );
        Ok(())
    }
}
