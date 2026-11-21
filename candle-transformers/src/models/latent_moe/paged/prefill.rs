//! Launches the paged latent prefill kernel, which attends many queries over
//! one settled slot in a single pass.
//!
//! The prefill's output matches running the decode kernel once per token. The
//! queries are mutually independent, so a large launch is chunked to the fixed
//! workspace without changing a bit. The corpus pre-pass ropes and quantises
//! the cache on the first chunk into scratch that the attention then reads. The
//! kernel writes every element of its output, so that output is allocated
//! uninitialised.

use super::*;

/// The prefill entry: many queries over a SETTLED slot (all latents written +
/// committed before the call — no fused scatter). `q` `[total_q, H, 512]`
/// bf16 pre-RoPE, `q_pos` `[total_q]` u32, per-query selections. Numerics are
/// identical to running the decode entry once per token.
#[cfg(feature = "cuda")]
pub fn paged_latent_prefill(
    q: &Tensor,
    headers: &Tensor,
    q_pos: &Tensor,
    kv_new: Option<(&Tensor, usize)>,
    cache: &CorpusCache,
    comp_idx: &Tensor,
    comp_cnt: &Tensor,
    sinks: &Tensor,
    rope_tab: &Tensor,
    ws: &LatentWorkspace,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
    // Writer-chunk float format tag: the new-token diagonal fake-quants to it.
    store_fmt: u8,
    // Staging scope for the 1-seq `seq_of`/`new_meta` tables this builds. A
    // caller outside a wave opens its own with `desc::scope` — the arena is
    // refcounted, so a scope taken inside one nests for free.
    generation: &Generation,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_latent_prefill requires CUDA tensors"),
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
    // Single-seq convenience: build the 1-seq batched tables (all queries → seq 0)
    // and dispatch to the batched `_raw`. Both are staged in the arena rather
    // than uploaded — `seq_of` is `total_q` zeros, which as a tensor would be an
    // allocation plus a memset for a table the kernel only reads.
    let total_q = q.dim(0)?;
    let seq_of = desc::stage_slice(&vec![0u32; total_q], generation)?;
    let (kv_t, rows, base) = match kv_new {
        Some((t, base)) => (t.clone(), t.dim(0)? as u32, base as u32),
        None => (
            Tensor::zeros((1, HEAD_DIM), DType::BF16, q.device())?,
            0u32,
            0u32,
        ),
    };
    // 1-seq new-token metadata: {rows, base, start=0, -}.
    let new_meta = desc::stage_slice(&[rows, base, 0u32, 0u32], generation)?;
    paged_latent_prefill_raw(
        q,
        hdr_ptr,
        q_pos,
        seq_of.ptr(),
        &kv_t,
        new_meta.ptr(),
        cache,
        comp_idx,
        comp_cnt,
        sinks,
        rope_tab,
        softmax_scale,
        window_size,
        num_splits_override,
        store_fmt,
        ws,
    )
}

/// As [`paged_latent_prefill`] but with the slot header as a raw device
/// address (the wave path's `build_decode_metadata` form).
#[cfg(feature = "cuda")]
pub fn paged_latent_prefill_raw(
    q: &Tensor,
    // Base of the prefill seqs' contiguous SlotHeader array (one slot per seq);
    // `seq_of[qi]` indexes into it.
    headers_ptr: u64,
    q_pos: &Tensor,
    // `[total_q]` u32 — which prefill seq each query belongs to (selects its
    // arena slot + new-token diagonal slice; the whole prefill fleet in one
    // launch). A raw device address: the wave builds this host-side and stages
    // it in its pinned arena, so it never round-trips through a device
    // allocation and an upload (see `desc.rs`).
    seq_of_ptr: u64,
    // All prefill seqs' just-computed latents packed `[total_new, 512]` bf16;
    // seq `s`'s rows start at `new_meta[s].start`, keyed at `new_meta[s].base + j`.
    kv_new: &Tensor,
    // `[n_seq, 4]` u32 per-seq new-token diagonal metadata: {rows, base, start, -},
    // staged like `seq_of_ptr`.
    new_meta_ptr: u64,
    cache: &CorpusCache,
    comp_idx: &Tensor,
    comp_cnt: &Tensor,
    sinks: &Tensor,
    rope_tab: &Tensor,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
    // Writer-chunk float format tag (`ArenaFormatTag::as_u8`): the fresh
    // diagonal fake-quants to it (FP8 rounds; BF16/F16/F32 read direct).
    store_fmt: u8,
    ws: &LatentWorkspace,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;

    let (total_q, n_q_head, hd) = q.dims3()?;
    if hd != HEAD_DIM {
        candle::bail!("paged_latent_prefill: head_dim {hd} != {HEAD_DIM}");
    }
    let max_sel = comp_idx.dim(1)?;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_latent_prefill requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();
    // Pure kernel output (final store): allocate uninit, the kernel writes all.
    // F32 — full-precision output straight into the int8 out-proj (no widening).
    let out = Tensor::empty((total_q, n_q_head, HEAD_DIM), DType::F32, q.device())?;
    // The prefill's split factor is 1 unless pinned by the override.
    let num_splits = num_splits_override.clamp(1, 32);
    // Queries are mutually independent (per-query position, selection, and
    // combine row), so the launch is CHUNKED to the fixed workspace: each
    // chunk's rows·splits fits WORKSPACE_CAP, and chunked launches are
    // bit-identical per row to one big launch.
    let q_chunk = WORKSPACE_CAP / (n_q_head * num_splits);
    if q_chunk == 0 {
        candle::bail!(
            "paged_latent_prefill: {n_q_head} heads × {num_splits} splits exceeds \
             WORKSPACE_CAP {WORKSPACE_CAP}"
        );
    }
    let (partial_acc, partial_ml) = (&ws.acc, &ws.ml);

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
    let out_ptr = cuda_ptr!(&out, f32); // kernel emits F32
    let pos_ptr = cuda_ptr!(q_pos, u32);
    let seq_ptr = seq_of_ptr;
    let new_ptr = cuda_ptr!(kv_new, half::bf16);
    let meta_ptr = new_meta_ptr;
    // Two-region corpus cache (the same the decode reads) — no per-prefill
    // rebuild from f32; the pre-pass dequantizes, ropes, then bakes into the int8
    // QK/PV scratch below.
    let ni8_p = cuda_ptr!(&cache.nope_i8, u8);
    let nsc_p = cuda_ptr!(&cache.nope_scale, f32);
    let rbf_p = cuda_ptr!(&cache.rope_bf, half::bf16);
    let cpos_p = cuda_ptr!(&cache.comp_pos, u32);
    let cidx_p = cuda_ptr!(comp_idx, u32);
    let ccnt_p = cuda_ptr!(comp_cnt, u32);
    let sink_p = cuda_ptr!(sinks, f32);
    let tab_p = cuda_ptr!(rope_tab, f32);
    let pacc_p = cuda_ptr!(partial_acc, f32);
    let pml_p = cuda_ptr!(partial_ml, f32);
    // Per-prefill BAKED scratch: the pre-pass ropes the two-region cache at each
    // entry's position and bakes into these once (g_total on the first chunk);
    // the attention kernel reads the baked int8 (QK) + gathers the pre-quantized
    // V bytes (PV), skipping the per-query RoPE.
    let g_total = cache.nope_i8.dim(0)?;
    // Pure pre-pass outputs (every row written when g_total>0, empty-shaped when 0).
    let comp_i8 = Tensor::empty((g_total, HEAD_DIM), DType::U8, q.device())?;
    let comp_scale = Tensor::empty((g_total, N_BANDS), DType::F32, q.device())?;
    // Pre-quantized PV operand: comp_v8 holds the corpus V bytes at the
    // per-dim-global scale comp_vmax.
    let comp_v8 = Tensor::empty((g_total, HEAD_DIM), DType::U8, q.device())?;
    // comp_vmax is the pre-pass's atomicMax accumulator — MUST start zeroed.
    let comp_vmax = Tensor::zeros(HEAD_DIM, DType::F32, q.device())?;
    let ci8_p = cuda_ptr!(&comp_i8, u8);
    let cscl_p = cuda_ptr!(&comp_scale, f32);
    let cv8_p = cuda_ptr!(&comp_v8, u8);
    let cvm_p = cuda_ptr!(&comp_vmax, f32);

    let mut base = 0usize;
    while base < total_q {
        let len = (total_q - base).min(q_chunk);
        // Rope+quant the corpus once, on the first chunk (all chunks share the
        // scratch on the ordered stream); later chunks pass g_total=0.
        let g_pass = if base == 0 { g_total as i32 } else { 0 };
        // Query-indexed inputs/outputs advance by `base`; the key sources
        // (arena, kv_new, comp) are launch-invariant.
        unsafe {
            candle_kernels::paged_latent::run_paged_latent_prefill_bf16(
                (q_ptr + (base * n_q_head * HEAD_DIM * 2) as u64) as *const core::ffi::c_void,
                hdr_ptr as *const u8,
                (out_ptr + (base * n_q_head * HEAD_DIM * 4) as u64) as *mut core::ffi::c_void, // F32 out: 4 B/elem
                (pos_ptr + (base * 4) as u64) as *const u32,
                (seq_ptr + (base * 4) as u64) as *const u32, // per-query seq, advanced with the chunk
                new_ptr as *const core::ffi::c_void,
                ni8_p as *const u8,
                nsc_p as *const f32,
                rbf_p as *const core::ffi::c_void,
                cpos_p as *const u32,
                (cidx_p + (base * max_sel * 4) as u64) as *const u32,
                (ccnt_p + (base * 4) as u64) as *const u32,
                sink_p as *const f32,
                tab_p as *const f32,
                pacc_p as *mut f32,
                pml_p as *mut f32,
                ci8_p as *mut u8,
                cscl_p as *mut f32,
                cv8_p as *mut u8,
                cvm_p as *mut f32,
                g_pass,
                len as i32,
                n_q_head as i32,
                softmax_scale,
                window_size as i32,
                max_sel as i32,
                meta_ptr as *const u32,
                num_splits as i32,
                store_fmt as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            );
        }
        base += len;
    }
    // No device sync (see `paged_latent_decode_raw`): stream ordering makes `out`
    // valid for the same-stream consumers without draining the GPU per call.
    Ok(out)
}
