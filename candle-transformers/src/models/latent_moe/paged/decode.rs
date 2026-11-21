//! Launches the paged latent decode kernel over a wave of slots and returns
//! its attention output.
//!
//! The kernel writes one final value per element, so the output is allocated
//! uninitialised and the call does not synchronise: stream order makes it
//! valid for every consumer on the same stream. Leading slots advance their
//! write length on the device and scatter their own token, while slots past
//! the bound were prepared by the caller. The raw-address form takes the slot
//! headers from a buffer the caller keeps alive across the launch.

use super::*;

/// gallery is F32 pre-RoPE. Returns the de-rotated attention output
/// `[slots, n_q_head, 512]` BF16.
#[cfg(feature = "cuda")]
pub fn paged_latent_decode(
    q: &Tensor,          // [slots, H, 512] bf16
    headers: &Tensor,    // [slots * SLOT_HEADER_BYTES] u8 (SlotHeader array)
    kv_new: &Tensor,     // [slots, 512] bf16
    cache: &CorpusCache, // persistent position-free int8 corpus
    comp_idx: &Tensor,   // [slots, max_sel] u32
    comp_cnt: &Tensor,   // [slots] u32
    q_pos: &Tensor,      // [slots] u32 query position (explicit)
    sinks: &Tensor,      // [H] f32
    rope_tab: &Tensor,   // [ROPE_TAB_LEN] f32 (build_rope_table)
    ws: &LatentWorkspace,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
    dbg: Option<&Tensor>, // f32 stage-dump, DBG_LEN long (mirror diagnostics)
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_latent_decode requires CUDA tensors"),
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
    paged_latent_decode_raw(
        q,
        hdr_ptr,
        kv_new,
        cache,
        comp_idx,
        comp_cnt,
        q_pos,
        sinks,
        rope_tab,
        softmax_scale,
        window_size,
        num_splits_override,
        // Tensor-headers path is the live buffer: every slot advances the
        // write-len on-device and fused-scatters its own token.
        q.dim(0)?,
        q.dim(0)?,
        ws,
        dbg,
    )
}

/// As [`paged_latent_decode`] but with the `SlotHeader` array given as a raw
/// device address — the form the production cache's `build_decode_metadata`
/// hands out (a pinned-stager `GpuBuf`, not a tensor). The caller keeps the
/// buffer alive across the call.
#[cfg(feature = "cuda")]
pub fn paged_latent_decode_raw(
    q: &Tensor,
    headers_ptr: u64,
    kv_new: &Tensor,
    cache: &CorpusCache,
    comp_idx: &Tensor,
    comp_cnt: &Tensor,
    q_pos: &Tensor,
    sinks: &Tensor,
    rope_tab: &Tensor,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
    // Leading slots that advance the write-len on-device (live buffers) —
    // slots past the bound hold throwaway host-patched snapshots. Plain rows
    // lead the wave, so a prefix count expresses every wave shape: a plain
    // decode wave passes the full slot count, a pure verify wave 0, a mixed
    // wave its plain-row count.
    commit_rows: usize,
    // Leading slots whose token the kernel fused-scatters; slots past the
    // bound were pre-scattered by the caller (host writeback, write-len
    // committed over them). The speculative-verify rows run as virtual slots
    // over ONE shared writer slice, where per-slot scatters would clobber.
    scatter_rows: usize,
    ws: &LatentWorkspace,
    dbg: Option<&Tensor>,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;

    let (num_slots, n_q_head, hd) = q.dims3()?;
    if hd != HEAD_DIM {
        candle::bail!("paged_latent_decode: head_dim {hd} != {HEAD_DIM}");
    }
    let max_sel = comp_idx.dim(1)?;
    let dev = match q.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("paged_latent_decode requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();

    // Pure kernel output (final store, not accumulation — the acc/ml workspace
    // holds the running softmax): allocate uninit, the kernel writes every
    // element. F32 — the kernel emits full-precision output (the int8 out-proj
    // consumes F32, so no bf16→F32 widening cast downstream).
    let out = Tensor::empty((num_slots, n_q_head, HEAD_DIM), DType::F32, q.device())?;
    let num_splits = decode_num_splits(num_slots, n_q_head, num_splits_override);
    if num_slots * n_q_head * num_splits > WORKSPACE_CAP {
        candle::bail!(
            "paged_latent_decode: {num_slots} slots × {n_q_head} heads × {num_splits} splits \
             exceeds WORKSPACE_CAP {WORKSPACE_CAP}"
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
    let kv_ptr = cuda_ptr!(kv_new, half::bf16);
    let ni8_p = cuda_ptr!(&cache.nope_i8, u8);
    let nsc_p = cuda_ptr!(&cache.nope_scale, f32);
    let rbf_p = cuda_ptr!(&cache.rope_bf, half::bf16);
    let cidx_p = cuda_ptr!(comp_idx, u32);
    let ccnt_p = cuda_ptr!(comp_cnt, u32);
    let cpos_p = cuda_ptr!(&cache.comp_pos, u32);
    let qpos_p = cuda_ptr!(q_pos, u32);
    let sink_p = cuda_ptr!(sinks, f32);
    let tab_p = cuda_ptr!(rope_tab, f32);
    let pacc_p = cuda_ptr!(partial_acc, f32);
    let pml_p = cuda_ptr!(partial_ml, f32);
    let dbg_p = match dbg {
        Some(t) => cuda_ptr!(t, f32),
        None => 0u64,
    };

    unsafe {
        candle_kernels::paged_latent::run_paged_latent_decode_bf16(
            q_ptr as *const core::ffi::c_void,
            hdr_ptr as *const u8,
            out_ptr as *mut core::ffi::c_void,
            kv_ptr as *const core::ffi::c_void,
            ni8_p as *const u8,
            nsc_p as *const f32,
            rbf_p as *const core::ffi::c_void,
            cidx_p as *const u32,
            ccnt_p as *const u32,
            cpos_p as *const u32,
            qpos_p as *const u32,
            sink_p as *const f32,
            tab_p as *const f32,
            pacc_p as *mut f32,
            pml_p as *mut f32,
            num_slots as i32,
            n_q_head as i32,
            softmax_scale,
            window_size as i32,
            max_sel as i32,
            num_splits as i32,
            commit_rows as i32,
            scatter_rows as i32,
            dbg_p as *mut f32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    // No device sync here: the kernel and every downstream consumer of `out`
    // (output projection, mHC, the eventual logits readback) run on the same
    // stream, so stream ordering already guarantees `out` is complete before it
    // is read. A per-call `synchronize()` drained the GPU and blocked the host
    // 43× per decode token (once per layer) — the dominant decode cost — for no
    // correctness benefit. The workspace (`ws.acc`/`ws.ml`) is likewise reused
    // stream-ordered across layers, so it needs no barrier either.
    Ok(out)
}

/// FP8-E4M3 writer-format tag (`KvFormat::to_tag`) — the single-latent window's
/// default storage. The synthetic-slot test/bench callers author FP8 slots, so
/// their new-token diagonal fake-quants to this.
#[cfg(feature = "cuda")]
pub(crate) fn fp8_store_tag() -> u8 {
    use candle_nn::kv_cache::KvFormat;
    KvFormat::Float(DType::F8E4M3).to_tag()
}
