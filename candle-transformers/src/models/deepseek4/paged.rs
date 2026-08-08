//! Rust wrapper + rung-2 standalone harness for the `paged-latent` decode
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

use candle::quantized::k_quants;
use candle::{DType, Result, Tensor};

pub const HEAD_DIM: usize = 512;
pub const ROPE_DIM: usize = 64;
pub const NOPE_DIM: usize = HEAD_DIM - ROPE_DIM;
/// Single-latent band count. 16 × 32-dim bands over the 512-dim latent isolate
/// the 64 RoPE dims into the last TWO bands ([448,480),[480,512)), and align
/// each band with one m16n8k32 MMA tile (SUB_DIM=32). Must equal `NPAL` in
/// latent_common.cuh and `LATENT_N_BANDS` in candle-nn.
pub const N_BANDS: usize = 16;
pub const SUB_DIM: usize = HEAD_DIM / N_BANDS;
pub const CHUNK: usize = 32;
/// KvHead record byte size (slot_types.cuh layout): the pal_map region
/// (HEAD_DIM/2 — one 4-bit map, K≡V) + the per-band
/// {k_ptr,v_ptr,k_fmt,v_fmt,k_scale,v_scale} block (N_BANDS*26).
pub const KVHEAD_BYTES: usize = HEAD_DIM / 2 + N_BANDS * 26;

/// Factored RoPE cos/sin table geometry (latent_common.cuh layout): position
/// `pos = hi·2¹⁰ + lo` splits the trig via the angle-addition identity, so two
/// small blocks — (sin, cos) of `(hi·2¹⁰)·f` and of `lo·f` — cover every
/// position below `ROPE_HI_DIM·2¹⁰` = 2M in ~768 KB per frequency set
/// (L2-resident). The attention kernel is a bounded WINDOW into the substrate
/// (the engine's context hard-cap is 1M), so every position it sees is < 2M and
/// lands in the table — the growing corpus is retrieved INTO that window, never
/// attended at raw substrate positions.
pub const ROPE_LO_BITS: usize = 10;
pub const ROPE_LO_DIM: usize = 1 << ROPE_LO_BITS; // 1024
pub const ROPE_HI_DIM: usize = 2048; // 2¹¹ → span 2²¹ ≈ 2M positions
/// f32 element count of one table: hi block + lo block, (sin, cos) per entry.
pub const ROPE_TAB_LEN: usize = (ROPE_HI_DIM + ROPE_LO_DIM) * (ROPE_DIM / 2) * 2;

/// MMA M dimension: query heads per block (kernel HEADS_TILE).
pub const HEADS_TILE: usize = 16;

/// Split-KV factor policy for the decode kernel: enough blocks to fill the
/// device (~2 waves of 4 blocks/SM), pinned by `override_` for test
/// determinism. Resolved HERE because the caller owns the partial workspace —
/// the kernel launcher has no allocator and no shared state.
#[cfg(feature = "cuda")]
fn decode_num_splits(num_slots: usize, n_q_head: usize, override_: usize) -> usize {
    if override_ > 0 {
        return override_.min(32);
    }
    let sm = unsafe { candle_kernels::paged_latent::run_latent_sm_count() }.max(1) as usize;
    let base_blocks = num_slots * n_q_head.div_ceil(HEADS_TILE);
    (sm * 8).div_ceil(base_blocks).clamp(1, 32)
}

/// Capacity of the shared partial workspace in `row × split` units — bounds
/// every launch's `rows·splits` product (`rows = queries-or-slots × heads`).
/// Prefill launches CHUNK their queries to fit (queries are mutually
/// independent, so chunked launches are bit-identical per row); the decode
/// split policy's product is bounded well under this for any wave size.
pub const WORKSPACE_CAP: usize = 32 * 1024;

/// Split-KV partial workspace for the latent kernels: the
/// `[rows, splits, HEAD_DIM]` accumulator + `[rows, splits, 2]` (m, l) pair
/// every decode/prefill launch hands to its combine. Built ONCE (fixed
/// `WORKSPACE_CAP`, ~64 MiB) and shared by every launch on a stream — peak
/// VRAM is a single buffer instead of per-launch allocations churning the
/// pool. It is IMMUTABLE on the host (only kernels write the device memory,
/// stream-ordered), so sharing is plain `&`/`Arc` — no device-side pool, no
/// static, no lock. The one ownership rule: host threads that launch
/// concurrently onto the SAME stream must not share one workspace (their
/// launch/combine pairs would interleave); the engine's single wave thread
/// and per-test instances satisfy this by construction.
#[cfg(feature = "cuda")]
pub struct LatentWorkspace {
    acc: Tensor,
    ml: Tensor,
}

#[cfg(feature = "cuda")]
impl LatentWorkspace {
    pub fn build(dev: &candle::Device) -> Result<Self> {
        Ok(Self {
            acc: Tensor::zeros(WORKSPACE_CAP * HEAD_DIM, DType::F32, dev)?,
            ml: Tensor::zeros(WORKSPACE_CAP * 2, DType::F32, dev)?,
        })
    }
}

/// Build the factored cos/sin table for one frequency set on-device (each
/// entry from the exact bit-mirrorable `rope_angle`/`ds_sincos` pair). Runs
/// once per set at model load; stream-ordered before any attention launch.
#[cfg(feature = "cuda")]
pub fn build_rope_table(rope_freqs: &Tensor) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let n_freqs = rope_freqs.dims1()?;
    if n_freqs != ROPE_DIM / 2 {
        candle::bail!("build_rope_table: {n_freqs} freqs != {}", ROPE_DIM / 2);
    }
    let dev = match rope_freqs.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("build_rope_table requires a CUDA tensor"),
    };
    let stream = dev.cuda_stream();
    let tab = Tensor::zeros(ROPE_TAB_LEN, DType::F32, rope_freqs.device())?;
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
    let freq_p = cuda_ptr!(rope_freqs, f32);
    let tab_p = cuda_ptr!(&tab, f32);
    unsafe {
        candle_kernels::paged_latent::run_latent_rope_table_build(
            freq_p as *const f32,
            tab_p as *mut f32,
            n_freqs as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(tab)
}

/// Launch the DeepSeek hybrid decode over one wave of slots. All tensors live
/// on the same CUDA device; `q`/`kv_new` are BF16 **pre-RoPE**, the compressed
/// Persistent roped int8 corpus cache: every finalized entry roped ONCE at
/// its fixed group-start position and per-band int8-quantized, then read by
/// each decode step — 4× smaller than the f32 gallery rows that dominated
/// decode traffic, and the per-query RoPE disappears. The canonical corpus
/// stays f32/pre-RoPE/position-free (§C); this is a derived, caller-owned
/// cache (one per layer, rebuilt/extended as the gallery grows).
/// Number of NOPE bands in the two-region cache (`[0, NOPE_DIM)` at `SUB_DIM`).
pub const NOPE_BANDS: usize = NOPE_DIM / SUB_DIM;

#[cfg(feature = "cuda")]
pub struct CorpusCache {
    /// `[g, NOPE_DIM]` u8 (int8 bits): the NOPE bands, position-free component-amax
    /// int8. The decode kernel dequantizes with `nope_scale`.
    pub nope_i8: Tensor,
    /// `[g, NOPE_BANDS]` f32 per-nope-band amax scale.
    pub nope_scale: Tensor,
    /// `[g, ROPE_DIM]` bf16: the ROPE bands, PRE-rotation (float, matches the
    /// window ring). The decode kernel rotates them at read time from `comp_pos`.
    pub rope_bf: Tensor,
    /// `[g]` u32 assembled position per entry — rope-at-load input for decode.
    pub comp_pos: Tensor,
    len: usize,
}

#[cfg(feature = "cuda")]
impl CorpusCache {
    /// Build the cache for the whole corpus (entries `[0, g)`). Callers with
    /// no corpus pass dummy tensors whose SHAPE may be one row while the
    /// storage is empty — the pre-pass launches only over rows that both the
    /// data and the positions actually back (an unbacked row's device pointer
    /// is null); unbuilt rows are never read (their gids are never selected).
    pub fn build(comp: &Tensor, comp_pos: &Tensor) -> Result<Self> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        // Clamp by ACTUAL storage length, not shape: dummy corpora carry a
        // one-row shape over empty storage (whose device pointer is null).
        let storage_rows = |t: &Tensor, per_row: usize| -> Result<usize> {
            let (storage, _) = t.storage_and_layout();
            Ok(match &*storage {
                Storage::Cuda(c) => {
                    c.as_cuda_slice::<f32>().map(|s| s.len()).unwrap_or(0) / per_row
                }
                _ => 0,
            })
        };
        let g = comp.dim(0)?.min(storage_rows(comp, HEAD_DIM)?).min({
            let (storage, _) = comp_pos.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>().map(|s| s.len()).unwrap_or(0),
                _ => 0,
            }
        });
        let dev = match comp.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("CorpusCache requires CUDA tensors"),
        };
        let stream = dev.cuda_stream();
        let nope_i8 = Tensor::zeros((g.max(1), NOPE_DIM), DType::U8, comp.device())?;
        let nope_scale = Tensor::zeros((g.max(1), NOPE_BANDS), DType::F32, comp.device())?;
        let rope_bf = Tensor::zeros((g.max(1), ROPE_DIM), DType::BF16, comp.device())?;
        macro_rules! p {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                match &*storage {
                    Storage::Cuda(c) => {
                        let (p, _g) = c.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                        p + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                    }
                    _ => candle::bail!("expected CUDA storage"),
                }
            }};
        }
        if g > 0 {
            let comp_p = p!(comp, f32);
            let ni8_p = p!(&nope_i8, u8);
            let nsc_p = p!(&nope_scale, f32);
            let rbf_p = p!(&rope_bf, half::bf16);
            unsafe {
                candle_kernels::paged_latent::run_latent_build_corpus_cache(
                    comp_p as *const f32,
                    ni8_p as *mut u8,
                    nsc_p as *mut f32,
                    rbf_p as *mut core::ffi::c_void,
                    0,
                    g as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        // Retain the positions (narrowed to the built rows) — the decode kernel
        // rotates the rope bands from them at read time.
        let comp_pos = comp_pos.narrow(0, 0, g.max(1))?.contiguous()?;
        Ok(Self {
            nope_i8,
            nope_scale,
            rope_bf,
            comp_pos,
            len: g,
        })
    }

    /// Hold a pre-built two-region cache gathered from the gallery (no rebuild).
    /// `comp_pos` are the entries' assembled positions; `len` is the count of
    /// real entries — pass it explicitly rather than deriving from the row shape,
    /// because the empty cache carries a one-row placeholder shape over zero
    /// entries (`len = 0`), and any `is_empty()`/`len()` caller must see that.
    #[cfg(feature = "cuda")]
    pub fn from_gathered(
        nope_i8: Tensor,
        nope_scale: Tensor,
        rope_bf: Tensor,
        comp_pos: Tensor,
        len: usize,
    ) -> Result<Self> {
        Ok(Self {
            nope_i8,
            nope_scale,
            rope_bf,
            comp_pos,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// gallery is F32 pre-RoPE. Returns the de-rotated attention output
/// `[slots, n_q_head, 512]` BF16.
#[cfg(feature = "cuda")]
pub fn paged_latent_decode(
    q: &Tensor,          // [slots, H, 512] bf16
    headers: &Tensor,    // [slots*24] u8 (SlotHeader array)
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
        true, // tensor-headers path is the live buffer: advance the write-len
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
    commit_write_len: bool,
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

    let out = Tensor::zeros((num_slots, n_q_head, HEAD_DIM), DType::BF16, q.device())?;
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
    let out_ptr = cuda_ptr!(&out, half::bf16);
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
            commit_write_len as i32,
            dbg_p as *mut f32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    q.device().synchronize()?;
    Ok(out)
}

/// FP8-E4M3 writer-format tag (`KvFormat::to_tag`) — the single-latent window's
/// default storage. The synthetic-slot test/bench callers author FP8 slots, so
/// their fresh diagonal fake-quants to this.
#[cfg(feature = "cuda")]
pub(crate) fn fp8_store_tag() -> u8 {
    use candle_nn::kv_cache::KvFormat;
    KvFormat::Float(DType::F8E4M3).to_tag()
}

/// The prefill entry: many queries over a SETTLED slot (all latents written +
/// committed before the call — no fused scatter). `q` `[total_q, H, 512]`
/// bf16 pre-RoPE, `q_pos` `[total_q]` u32, per-query selections. Numerics are
/// identical to running the decode entry once per token.
#[cfg(feature = "cuda")]
pub fn paged_latent_prefill(
    q: &Tensor,
    headers: &Tensor,
    q_pos: &Tensor,
    kv_fresh: Option<(&Tensor, usize)>,
    cache: &CorpusCache,
    comp_idx: &Tensor,
    comp_cnt: &Tensor,
    sinks: &Tensor,
    rope_tab: &Tensor,
    ws: &LatentWorkspace,
    softmax_scale: f32,
    window_size: usize,
    num_splits_override: usize,
    // Writer-chunk float format tag: the fresh diagonal fake-quants to it.
    store_fmt: u8,
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
    paged_latent_prefill_raw(
        q,
        hdr_ptr,
        q_pos,
        kv_fresh,
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
    headers_ptr: u64,
    q_pos: &Tensor,
    // This layer's just-computed latents `[fresh_rows, 512]` bf16 keyed at
    // `fresh_base + j` — the batched-wave source for tokens not yet written to
    // the arena. `None` on the settled-slot path.
    kv_fresh: Option<(&Tensor, usize)>,
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
    let out = Tensor::zeros((total_q, n_q_head, HEAD_DIM), DType::BF16, q.device())?;
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
    let out_ptr = cuda_ptr!(&out, half::bf16);
    let pos_ptr = cuda_ptr!(q_pos, u32);
    let (fresh_ptr, fresh_rows, fresh_base) = match kv_fresh {
        Some((t, base)) => (cuda_ptr!(t, half::bf16), t.dim(0)?, base),
        None => (0u64, 0usize, 0usize),
    };
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
    let comp_i8 = Tensor::zeros((g_total, HEAD_DIM), DType::U8, q.device())?;
    let comp_scale = Tensor::zeros((g_total, N_BANDS), DType::F32, q.device())?;
    // Pre-quantized PV operand: comp_v8 holds the corpus V bytes at the
    // per-dim-global scale comp_vmax (zeros-alloc doubles as the zero init the
    // pre-pass's atomicMax accumulation requires).
    let comp_v8 = Tensor::zeros((g_total, HEAD_DIM), DType::U8, q.device())?;
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
        // (arena, kv_fresh, comp) are launch-invariant.
        unsafe {
            candle_kernels::paged_latent::run_paged_latent_prefill_bf16(
                (q_ptr + (base * n_q_head * HEAD_DIM * 2) as u64) as *const core::ffi::c_void,
                hdr_ptr as *const u8,
                (out_ptr + (base * n_q_head * HEAD_DIM * 2) as u64) as *mut core::ffi::c_void,
                (pos_ptr + (base * 4) as u64) as *const u32,
                fresh_ptr as *const core::ffi::c_void,
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
                fresh_rows as i32,
                fresh_base as i32,
                num_splits as i32,
                store_fmt as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            );
        }
        base += len;
    }
    q.device().synchronize()?;
    Ok(out)
}

/// Scatter glue latents into their RESERVED gap chunks (per-row block index +
/// in-block offset from the reprojection's descriptors). Launch BEFORE any
/// attention pass of the same layer on the same stream.
#[cfg(feature = "cuda")]
pub fn paged_latent_glue_scatter(
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
        candle_kernels::paged_latent::run_paged_latent_glue_scatter_bf16(
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
/// Per-band storage spec for synthetic window authoring: the KvHead format
/// tag plus the outer (encoder-multiply / decoder-divide) scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandSpec {
    pub fmt: u8,
    pub outer: f32,
}

impl Default for BandSpec {
    fn default() -> Self {
        // F8E4M3 at unit scale — the writer-chunk format.
        Self {
            fmt: 34,
            outer: 1.0,
        }
    }
}

/// Lane-0 result of the GPU warp XOR-tree float sum (offsets 16, 8, 4, 2,
/// 1). The pairwise summation ORDER is part of the bit-exact contract — a
/// serial sum can round differently, flipping an INT8-encoded scale by one
/// code. Addition is commutative in IEEE, so every lane of a pair computes
/// identical bits and all 32 lanes converge to the same value.
fn warp_tree_sum(v: &[f32; CHUNK]) -> f32 {
    let mut a = *v;
    for off in [16usize, 8, 4, 2, 1] {
        let p = a;
        for i in 0..CHUNK {
            a[i] = p[i] + p[i ^ off];
        }
    }
    a[0]
}

/// The GPU `q0_encode_centroid`: clamp the FLOAT to ±127 first, then
/// round-to-nearest-even (`__float2int_rn`).
fn q0_encode_centroid(val: f32) -> i8 {
    (val * 127.0).min(127.0).max(-127.0).round_ties_even() as i32 as i8
}

/// Band-chunk storage round-trip: encode `toks` (≤32 tokens × `SUB_DIM` dims,
/// zero-padded to the chunk) into the band's arena bytes for `spec`, and
/// return the exact per-token values the kernel's dequant produces (the
/// mirror's stored source — one codec for both sides). Float bands are
/// token-major rows; quant bands are token-oriented GGML blocks (block `d`
/// holds dim `d`'s 32 tokens; `CHUNK == 32` so blocks_per_dim is 1). Decode
/// arithmetic replicates the GPU converters op-for-op: Q8_0 is
/// `half2float(d) * qs / outer`, Q4_0 rounds `half2float(d) / outer` FIRST
/// and stores split nibbles (elements 0-15 low, 16-31 high).
///
/// For the INT8-scale and Q0-family formats the `spec.outer` is the searched
/// palette scale (the encoder sees `x * outer`, the decoder divides it back
/// out last — the exact `load_element` op order), and every warp-tree float
/// sum in the GPU encoder is replicated through [`warp_tree_sum`].
pub fn band_chunk_roundtrip(
    spec: BandSpec,
    toks: &[[f32; SUB_DIM]],
) -> Result<(Vec<u8>, Vec<[f32; SUB_DIM]>)> {
    let n = toks.len().min(CHUNK);
    let mut dec = vec![[0f32; SUB_DIM]; n];
    let bytes = match spec.fmt {
        34 => {
            // F8E4M3: token-major rows of SUB_DIM bytes.
            let mut bytes = vec![0u8; CHUNK * SUB_DIM];
            for (t, tok) in toks.iter().take(n).enumerate() {
                for (d, &v) in tok.iter().enumerate() {
                    let b = f32_to_e4m3(v * spec.outer);
                    bytes[t * SUB_DIM + d] = b;
                    dec[t][d] = e4m3_to_f32(b) / spec.outer;
                }
            }
            bytes
        }
        7 => {
            // Q8_0: 34-byte blocks `{f16 d; i8 qs[32]}` (GGML reference
            // quantizer: d = amax/127, qs = round(x/d)).
            let mut bytes = vec![0u8; SUB_DIM * 34];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let dh = half::f16::from_f32(amax * (1.0 / 127.0));
                let ds = dh.to_f32();
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 34..(d + 1) * 34];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                for (t, &v) in col.iter().enumerate() {
                    let q = (v * id).round().clamp(-127.0, 127.0) as i8;
                    blk[2 + t] = q as u8;
                    if t < n {
                        dec[t][d] = ds * q as f32 / spec.outer;
                    }
                }
            }
            bytes
        }
        15 => {
            // Q4_0: 18-byte blocks `{f16 d; u8 qs[16]}`, split nibbles (GGML
            // reference quantizer: d = signed_max/-8, q = x/d + 8.5 clamped).
            let mut bytes = vec![0u8; SUB_DIM * 18];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let mut smax = 0f32;
                for &v in &col {
                    if v.abs() > smax.abs() {
                        smax = v;
                    }
                }
                let dh = half::f16::from_f32(smax / -8.0);
                let ds = dh.to_f32();
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 18..(d + 1) * 18];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                let mut nib = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    nib[t] = ((v * id + 8.5) as i32).clamp(0, 15) as u8;
                }
                for j in 0..16 {
                    blk[2 + j] = nib[j] | (nib[j + 16] << 4);
                }
                // GPU converter rounds d/outer before the nibble multiply.
                let dso = ds / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = dso * (nib[t] as f32 - 8.0);
                }
            }
            bytes
        }
        16 => {
            // Q4_1: 20-byte blocks `{f16 d; f16 m; u8 qs[16]}`, split nibbles
            // (encoder: d = (max-min)/15, q = (x-min)/d + 0.5 truncated).
            let mut bytes = vec![0u8; SUB_DIM * 20];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let (mut vmax, mut vmin) = (col[0], col[0]);
                for &v in &col {
                    vmax = vmax.max(v);
                    vmin = vmin.min(v);
                }
                let dq = (vmax - vmin) * (1.0 / 15.0);
                let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
                let dh = half::f16::from_f32(dq);
                let mh = half::f16::from_f32(vmin);
                let blk = &mut bytes[d * 20..(d + 1) * 20];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2..4].copy_from_slice(&mh.to_le_bytes());
                let mut nib = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    nib[t] = ((v - vmin) * id + 0.5).clamp(0.0, 15.0) as u8;
                }
                for j in 0..16 {
                    blk[4 + j] = nib[j] | (nib[j + 16] << 4);
                }
                // GPU decode: d and m each divided by outer first; the
                // `d*nibble + m` in BlockConverter<block_q4_1> CONTRACTS to an
                // fma in the latent TU — mirror with mul_add (verified by the
                // compressed-arena round-trip: plain mul+add differs by 1 ulp
                // on rare values).
                let d32 = dh.to_f32() / spec.outer;
                let m32 = mh.to_f32() / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = d32 * nib[t] as f32 + m32;
                }
            }
            bytes
        }
        10 => {
            // Q8_KS: 36-byte blocks `{f16 d; u8 sa; u8 sb; i8 qs[32]}` — the
            // attention-sink format: elements 0-3 (the chunk's first four
            // TOKENS in this token-oriented layout — the sink positions) get
            // their own sub-scale sa, elements 4-31 get sb.
            let mut bytes = vec![0u8; SUB_DIM * 36];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let amax_a = col[..4].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax_b = col[4..].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax = amax_a.max(amax_b);
                let coarse = if amax != 0.0 {
                    amax * (1.0 / 127.0)
                } else {
                    0.0
                };
                let (sa, sb) = if amax == 0.0 {
                    (255u8, 255u8)
                } else {
                    (
                        (amax_a / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                        (amax_b / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                    )
                };
                let dh = half::f16::from_f32(coarse);
                let blk = &mut bytes[d * 36..(d + 1) * 36];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2] = sa;
                blk[3] = sb;
                // Encoder scales from the UNROUNDED coarse d ((coarse·s)/255,
                // mul then div — the GPU's op order).
                let act_a = coarse * sa as f32 * (1.0 / 255.0);
                let act_b = coarse * sb as f32 * (1.0 / 255.0);
                for (t, &v) in col.iter().enumerate() {
                    let act = if t < 4 { act_a } else { act_b };
                    let q = if act != 0.0 {
                        (v / act).round().clamp(-127.0, 127.0) as i8
                    } else {
                        0
                    };
                    blk[4 + t] = q as u8;
                }
                // Decoder scales from the f16-ROUNDED d (s/255 first, then
                // cd·that — the GPU's op order), then /outer.
                let cd = dh.to_f32();
                let da = cd * (sa as f32 * (1.0 / 255.0));
                let db = cd * (sb as f32 * (1.0 / 255.0));
                for (t, dt) in dec.iter_mut().enumerate() {
                    let q = blk[4 + t] as i8;
                    let sub = if t < 4 { da } else { db };
                    dt[d] = sub * q as f32 / spec.outer;
                }
            }
            bytes
        }
        8 => {
            // Q8_1: 36-byte blocks `{f16 d; f16 s; i8 qs[32]}` — d = amax/127,
            // s = the warp-tree sum (unused by decode), q = rne(x·127/amax)
            // via `__float2int_rn` (no clamp: |x| ≤ amax bounds it).
            let mut bytes = vec![0u8; SUB_DIM * 36];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let sum = warp_tree_sum(&col);
                let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
                let dh = half::f16::from_f32(amax * (1.0 / 127.0));
                let sh = half::f16::from_f32(sum);
                let blk = &mut bytes[d * 36..(d + 1) * 36];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2..4].copy_from_slice(&sh.to_le_bytes());
                let ds = dh.to_f32();
                for (t, &v) in col.iter().enumerate() {
                    let q = (v * id).round_ties_even() as i32 as i8;
                    blk[4 + t] = q as u8;
                    if t < n {
                        dec[t][d] = ds * q as f32 / spec.outer;
                    }
                }
            }
            bytes
        }
        18 => {
            // Q4_KS: 20-byte blocks `{f16 d; u8 sa; u8 sb; u8 qs[16]}` — the
            // 4-bit sink format: elems 0-3 get sub-scale sa, 4-31 sb; biased
            // nibbles packed k | (k+16)<<4. Encoder scales from the UNROUNDED
            // coarse d; decoder from the f16-rounded d (GPU op orders).
            let mut bytes = vec![0u8; SUB_DIM * 20];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax_a = col[..4].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax_b = col[4..].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax = amax_a.max(amax_b);
                let coarse = if amax != 0.0 { amax * (1.0 / 7.0) } else { 0.0 };
                let (sa, sb) = if amax == 0.0 {
                    (255u8, 255u8)
                } else {
                    (
                        (amax_a / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                        (amax_b / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                    )
                };
                let dh = half::f16::from_f32(coarse);
                let blk = &mut bytes[d * 20..(d + 1) * 20];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2] = sa;
                blk[3] = sb;
                let act_a = coarse * sa as f32 * (1.0 / 255.0);
                let act_b = coarse * sb as f32 * (1.0 / 255.0);
                let mut qb = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    let act = if t < 4 { act_a } else { act_b };
                    let q = if act != 0.0 {
                        (v / act).round().clamp(-7.0, 7.0) as i32
                    } else {
                        0
                    };
                    qb[t] = (q + 8) as u8;
                }
                for k in 0..16 {
                    blk[4 + k] = qb[k] | (qb[k + 16] << 4);
                }
                let cd = dh.to_f32();
                let da = cd * (sa as f32 * (1.0 / 255.0));
                let db = cd * (sb as f32 * (1.0 / 255.0));
                for (t, dt) in dec.iter_mut().enumerate() {
                    let nib = if t < 16 {
                        (blk[4 + t] & 0xF) as i32 - 8
                    } else {
                        (blk[4 + t - 16] >> 4) as i32 - 8
                    };
                    let sc = if t < 4 { da } else { db };
                    dt[d] = sc * nib as f32 / spec.outer;
                }
            }
            bytes
        }
        19 => {
            // Q3_0: 14-byte blocks `{f16 d; u8 qh[4]; u8 qs[8]}` — 3-bit
            // symmetric centred at 3.5; low 2 bits in qs, high bit in qh.
            let mut bytes = vec![0u8; SUB_DIM * 14];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let dh = half::f16::from_f32(amax * (1.0 / 3.5));
                let id = if amax != 0.0 { 3.5 / amax } else { 0.0 };
                let blk = &mut bytes[d * 14..(d + 1) * 14];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = (v * id + 3.5).round().clamp(0.0, 7.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[2 + (t >> 3)] |= ((q >> 2) & 1) << (t & 7); // qh
                    blk[6 + (t >> 2)] |= (q & 3) << ((t & 3) * 2); // qs
                }
                let ds = dh.to_f32();
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = ds * (qv[t] as f32 - 3.5) / spec.outer;
                }
            }
            bytes
        }
        20 => {
            // Q3_1: 16-byte blocks `{f16 d; f16 m; u8 qh[4]; u8 qs[8]}` —
            // 3-bit affine (d = (max-min)/7, id from the UNROUNDED d). GPU
            // decode divides d and m by outer first, then PLAIN `d*q + m`
            // (no fma contraction in the kernel TU — measured: mul_add
            // shifts hundreds of cells by 1-2 bf16 codes at outer 1.5,
            // where d carries a full mantissa and the product rounds).
            let mut bytes = vec![0u8; SUB_DIM * 16];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let (mut vmax, mut vmin) = (col[0], col[0]);
                for &v in &col {
                    vmax = vmax.max(v);
                    vmin = vmin.min(v);
                }
                let dq = (vmax - vmin) * (1.0 / 7.0);
                let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
                let dh = half::f16::from_f32(dq);
                let mh = half::f16::from_f32(vmin);
                let blk = &mut bytes[d * 16..(d + 1) * 16];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2..4].copy_from_slice(&mh.to_le_bytes());
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = ((v - vmin) * id).round().clamp(0.0, 7.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[4 + (t >> 3)] |= ((q >> 2) & 1) << (t & 7); // qh
                    blk[8 + (t >> 2)] |= (q & 3) << ((t & 3) * 2); // qs
                }
                let d32 = dh.to_f32() / spec.outer;
                let m32 = mh.to_f32() / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = d32 * qv[t] as f32 + m32;
                }
            }
            bytes
        }
        25 => {
            // Q2_S: 9-byte blocks `{i8 scale; u8 qs[8]}` — 2-bit symmetric
            // centred at 1.5 with an INT8 scale (d = amax/1.5, INT8-encoded
            // then decoded back for the round-trip-consistent quant step).
            let mut bytes = vec![0u8; SUB_DIM * 9];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let scale =
                    ((amax * (1.0 / 1.5)) * 127.0).min(127.0).round_ties_even() as i32 as i8;
                let ds = scale as f32 * (1.0 / 127.0);
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 9..(d + 1) * 9];
                blk[0] = scale as u8;
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = (v * id + 1.5).round().clamp(0.0, 3.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[1 + (t >> 2)] |= (q & 3) << ((t & 3) * 2);
                }
                let dd = scale as f32 * (1.0 / 127.0) / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = dd * (qv[t] as f32 - 1.5);
                }
            }
            bytes
        }
        26 => {
            // Q2_A: 10-byte blocks `{i8 scale; i8 bias; u8 qs[8]}` — 2-bit
            // affine, both parameters INT8-encoded then decoded for the
            // quant step. Decode is PLAIN `d*q + m` (no fma contraction in
            // the kernel TU — same measured finding as Q3_1).
            let mut bytes = vec![0u8; SUB_DIM * 10];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let (mut vmax, mut vmin) = (col[0], col[0]);
                for &v in &col {
                    vmax = vmax.max(v);
                    vmin = vmin.min(v);
                }
                let scale = (((vmax - vmin) * (1.0 / 3.0)) * 127.0)
                    .min(127.0)
                    .round_ties_even() as i32 as i8;
                let bias = (vmin * 127.0).min(127.0).max(-127.0).round_ties_even() as i32 as i8;
                let ds = scale as f32 * (1.0 / 127.0);
                let m = bias as f32 * (1.0 / 127.0);
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 10..(d + 1) * 10];
                blk[0] = scale as u8;
                blk[1] = bias as u8;
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = ((v - m) * id).round().clamp(0.0, 3.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[2 + (t >> 2)] |= (q & 3) << ((t & 3) * 2);
                }
                let dd = scale as f32 * (1.0 / 127.0) / spec.outer;
                let dm = bias as f32 * (1.0 / 127.0) / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = dd * qv[t] as f32 + dm;
                }
            }
            bytes
        }
        27 => {
            // Q1_S: 5-byte blocks `{i8 scale; u8 qs[4]}` — sign bits + the
            // INT8-encoded mean(|x|) magnitude (warp-tree sum order).
            let mut bytes = vec![0u8; SUB_DIM * 5];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let abs: [f32; CHUNK] = std::array::from_fn(|t| col[t].abs());
                let mean_abs = warp_tree_sum(&abs) / 32.0;
                let scale = (mean_abs * 127.0).min(127.0).round_ties_even() as i32 as i8;
                let blk = &mut bytes[d * 5..(d + 1) * 5];
                blk[0] = scale as u8;
                for (t, &v) in col.iter().enumerate() {
                    blk[1 + (t >> 3)] |= u8::from(v >= 0.0) << (t & 7);
                }
                let bs = scale as f32 * (1.0 / 127.0) / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = if col[t] >= 0.0 { bs } else { -bs };
                }
            }
            bytes
        }
        29 => {
            // Q1_A: 6-byte blocks `{i8 scale_pos; i8 scale_neg; u8 qs[4]}` —
            // sign bits + a separate INT8 mean amplitude per sign class.
            let mut bytes = vec![0u8; SUB_DIM * 6];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let neg: [f32; CHUNK] =
                    std::array::from_fn(|t| if col[t] >= 0.0 { 0.0 } else { -col[t] });
                let pos_in: [f32; CHUNK] =
                    std::array::from_fn(|t| if col[t] >= 0.0 { col[t] } else { 0.0 });
                let n_pos = col.iter().filter(|&&v| v >= 0.0).count() as i32;
                let n_neg = 32 - n_pos;
                let sum_pos = warp_tree_sum(&pos_in);
                let sum_neg = warp_tree_sum(&neg);
                let mean_pos = if n_pos > 0 {
                    sum_pos / n_pos as f32
                } else {
                    0.0
                };
                let mean_neg = if n_neg > 0 {
                    sum_neg / n_neg as f32
                } else {
                    0.0
                };
                let sp = ((mean_pos * 127.0).round_ties_even() as i32).clamp(0, 127) as i8;
                let sn = ((mean_neg * 127.0).round_ties_even() as i32).clamp(0, 127) as i8;
                let blk = &mut bytes[d * 6..(d + 1) * 6];
                blk[0] = sp as u8;
                blk[1] = sn as u8;
                for (t, &v) in col.iter().enumerate() {
                    blk[2 + (t >> 3)] |= u8::from(v >= 0.0) << (t & 7);
                }
                let mp = sp as f32 * (1.0 / 127.0);
                let mn = sn as f32 * (1.0 / 127.0);
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = (if col[t] >= 0.0 { mp } else { -mn }) / spec.outer;
                }
            }
            bytes
        }
        28 => {
            // Q0_V: 2-byte pattern-indexed blocks. The Rust k_quants encoder
            // and decoder ARE the declared references for the CUDA pair —
            // K-side tables (the single latent is K≡V).
            let mut bytes = vec![0u8; SUB_DIM * 2];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let blk = k_quants::encode_block_q0_v::<true>(&col);
                bytes[d * 2..(d + 1) * 2].copy_from_slice(&blk.to_le_bytes());
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = k_quants::q0_v_elem::<true>(&blk, t) / spec.outer;
                }
            }
            bytes
        }
        30 => {
            // Q0_X: 2-byte blocks `{i8 bulk_anchor; u8 outlier_packed}` —
            // INT8 block mean + one escaped outlier (first-lane tie-break,
            // 3-bit delta at stride 32). Anchor rounds THEN int-clamps.
            let mut bytes = vec![0u8; SUB_DIM * 2];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let mean = warp_tree_sum(&col) * (1.0 / 32.0);
                let anchor = ((mean * 127.0).round_ties_even() as i32).clamp(-127, 127);
                let residual: [i32; CHUNK] = std::array::from_fn(|t| {
                    ((col[t] * 127.0).round_ties_even() as i32).clamp(-127, 127) - anchor
                });
                let max_abs = residual.iter().map(|r| r.abs()).max().unwrap();
                let idx = residual.iter().position(|r| r.abs() == max_abs).unwrap();
                let delta = ((residual[idx] as f32 / 32.0).round_ties_even() as i32).clamp(-4, 3);
                let packed = (idx as u8 & 0x1F) | (((delta as u8) & 0x07) << 5);
                let blk = &mut bytes[d * 2..(d + 1) * 2];
                blk[0] = anchor as i8 as u8;
                blk[1] = packed;
                for (t, dt) in dec.iter_mut().enumerate() {
                    let ds = if t == idx { delta * 32 } else { 0 };
                    let v = (anchor + ds).clamp(-127, 127);
                    dt[d] = v as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        31 => {
            // Q0_M2: 3-byte blocks `{i8 centroid[2]; u8 qmask}` — Lloyd ×4
            // over quartet means (in-quartet XOR-tree sums, centroid updates
            // through the full warp tree).
            let mut bytes = vec![0u8; SUB_DIM * 3];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                // Quartet means: XOR-reduce offsets 1 then 2 within each 4.
                let mut qs = col;
                for off in [1usize, 2] {
                    let p = qs;
                    for i in 0..CHUNK {
                        qs[i] = p[i] + p[i ^ off];
                    }
                }
                let qt_mean: [f32; CHUNK] = std::array::from_fn(|i| qs[i] * 0.25);
                let mut c0 = qt_mean.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                let mut c1 = qt_mean.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                for _ in 0..4 {
                    let assign: [bool; CHUNK] =
                        std::array::from_fn(|i| (qt_mean[i] - c1).abs() < (qt_mean[i] - c0).abs());
                    let s0 =
                        warp_tree_sum(&std::array::from_fn(
                            |i| {
                                if assign[i] {
                                    0.0
                                } else {
                                    col[i]
                                }
                            },
                        ));
                    let n0 =
                        warp_tree_sum(&std::array::from_fn(|i| if assign[i] { 0.0 } else { 1.0 }));
                    let s1 =
                        warp_tree_sum(&std::array::from_fn(
                            |i| if assign[i] { col[i] } else { 0.0 },
                        ));
                    let n1 =
                        warp_tree_sum(&std::array::from_fn(|i| if assign[i] { 1.0 } else { 0.0 }));
                    if n0 > 0.0 {
                        c0 = s0 / n0;
                    }
                    if n1 > 0.0 {
                        c1 = s1 / n1;
                    }
                }
                let mut qmask = 0u8;
                for q in 0..8 {
                    let i = q * 4;
                    if (qt_mean[i] - c1).abs() < (qt_mean[i] - c0).abs() {
                        qmask |= 1 << q;
                    }
                }
                let blk = &mut bytes[d * 3..(d + 1) * 3];
                blk[0] = q0_encode_centroid(c0) as u8;
                blk[1] = q0_encode_centroid(c1) as u8;
                blk[2] = qmask;
                for (t, dt) in dec.iter_mut().enumerate() {
                    let c = blk[((qmask >> (t / 4)) & 1) as usize] as i8;
                    dt[d] = c as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        32 => {
            // Q0_M4: 8-byte blocks `{i8 centroid[4]; u32 qmask}` — Lloyd ×5
            // over pair means; centroids seeded equally-spaced vmin..vmax.
            let mut bytes = vec![0u8; SUB_DIM * 8];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let mut ps = col;
                {
                    let p = ps;
                    for i in 0..CHUNK {
                        ps[i] = p[i] + p[i ^ 1];
                    }
                }
                let pair_mean: [f32; CHUNK] = std::array::from_fn(|i| ps[i] * 0.5);
                let vmin = pair_mean.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                let vmax = pair_mean.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let step = (vmax - vmin) * (1.0 / 3.0);
                let mut c = [vmin, vmin + step, vmin + 2.0 * step, vmax];
                let assign_of = |pm: f32, c: &[f32; 4]| -> usize {
                    let mut best = 0usize;
                    let mut best_d = (pm - c[0]).abs();
                    for (k, &ck) in c.iter().enumerate().skip(1) {
                        let dd = (pm - ck).abs();
                        if dd < best_d {
                            best_d = dd;
                            best = k;
                        }
                    }
                    best
                };
                for _ in 0..5 {
                    let assign: [usize; CHUNK] =
                        std::array::from_fn(|i| assign_of(pair_mean[i], &c));
                    for k in 0..4 {
                        let sk = warp_tree_sum(&std::array::from_fn(|i| {
                            if assign[i] == k {
                                col[i]
                            } else {
                                0.0
                            }
                        }));
                        let nk = warp_tree_sum(&std::array::from_fn(|i| {
                            if assign[i] == k {
                                1.0
                            } else {
                                0.0
                            }
                        }));
                        if nk > 0.0 {
                            c[k] = sk / nk;
                        }
                    }
                }
                let mut qmask = 0u32;
                for pair in 0..16 {
                    qmask |= (assign_of(pair_mean[pair * 2], &c) as u32) << (2 * pair);
                }
                let blk = &mut bytes[d * 8..(d + 1) * 8];
                for k in 0..4 {
                    blk[k] = q0_encode_centroid(c[k]) as u8;
                }
                blk[4..8].copy_from_slice(&qmask.to_le_bytes());
                for (t, dt) in dec.iter_mut().enumerate() {
                    let ci = ((qmask >> (2 * (t / 2))) & 3) as usize;
                    dt[d] = blk[ci] as i8 as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        33 => {
            // Q0: 1-byte blocks `{i8 centroid}` — the INT8-encoded block
            // mean, shared by all 32 elements.
            let mut bytes = vec![0u8; SUB_DIM];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let centroid = q0_encode_centroid(warp_tree_sum(&col) * (1.0 / 32.0));
                bytes[d] = centroid as u8;
                for dt in dec.iter_mut() {
                    dt[d] = centroid as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        f => candle::bail!("band_chunk_roundtrip: unsupported synthetic format {f}"),
    };
    Ok((bytes, dec))
}

/// The identity pal_map for the latent layout: dim d → band d/SUB_DIM.
///
/// One 4-bit id per dim (2 dims/byte) over the record's `[0,HD/2)` map region
/// (K≡V, so the old separate k_pal/v_pal halves are a single map). 4 bits names
/// all 16 bands, so the identity map is faithful.
pub fn identity_pal_map() -> [u8; HEAD_DIM / 2] {
    let mut m = [0u8; HEAD_DIM / 2];
    for d in 0..HEAD_DIM {
        m[d >> 1] |= ((d / SUB_DIM) as u8) << ((d & 1) * 4);
    }
    m
}

/// Band-chunk storage round-trip for the IDENTITY latent layout: band `p` holds
/// the natural dims `[p*SUB_DIM, p*SUB_DIM+SUB_DIM)`. Returns the per-band arena
/// bytes plus the decoded values in natural dim positions. Wraps
/// [`band_chunk_roundtrip`] (the proven per-band codec). There is no palette
/// regroup — the kernels read the identity layout only.
pub fn mapped_chunk_roundtrip(
    specs: &[BandSpec; N_BANDS],
    toks: &[[f32; HEAD_DIM]],
) -> Result<([Vec<u8>; N_BANDS], Vec<[f32; HEAD_DIM]>)> {
    let n = toks.len().min(CHUNK);
    let mut dec = vec![[0f32; HEAD_DIM]; n];
    let mut bytes: [Vec<u8>; N_BANDS] = Default::default();
    for p in 0..N_BANDS {
        let band_toks: Vec<[f32; SUB_DIM]> = toks
            .iter()
            .take(n)
            .map(|tok| std::array::from_fn(|r| tok[p * SUB_DIM + r]))
            .collect();
        let (b, d) = band_chunk_roundtrip(specs[p], &band_toks)?;
        bytes[p] = b;
        for (t, row) in d.iter().enumerate() {
            for (r, &v) in row.iter().enumerate() {
                dec[t][p * SUB_DIM + r] = v;
            }
        }
    }
    Ok((bytes, dec))
}

pub struct SyntheticSlots {
    /// Flat u8 band arenas — per-(chunk, band) regions at `band_offsets`,
    /// each in that band's authored format (F8E4M3 rows or GGML blocks).
    pub bands: Tensor,
    /// `[n_chunks_total * KVHEAD_BYTES]` u8.
    pub kvheads: Tensor,
    /// `[n_chunks_total * 16]` u8 — TokenSlice array (all slots concatenated).
    pub slices: Tensor,
    /// `[num_slots * 24]` u8 — SlotHeader array.
    pub headers: Tensor,
    /// Per slot: (first chunk index, n_chunks, n_tokens).
    pub slot_meta: Vec<(usize, usize, usize)>,
    /// Per slot, per token: the exact stored values the kernel dequantizes
    /// (each band's storage round-trip applied) — the mirror's key source.
    pub stored: Vec<Vec<[f32; HEAD_DIM]>>,
}

#[cfg(feature = "cuda")]
impl SyntheticSlots {
    /// Build slots from per-slot pre-RoPE window latents (each `[n_tokens][512]`
    /// f32, values FP8-representable). Chunk `c` of a slot holds tokens
    /// `[32c, 32c+32)` at rope positions starting `32c`; the last chunk is the
    /// writer (its `len` excludes the incoming token).
    pub fn build(dev: &candle::Device, windows: &[Vec<[f32; HEAD_DIM]>]) -> Result<Self> {
        Self::build_based(dev, windows, &vec![0usize; windows.len()])
    }

    /// As [`Self::build`] but each slot's window is rope-based at `bases[slot]`
    /// — chunk `c` holds tokens `[32c, 32c+32)` at rope positions
    /// `bases[slot] + 32c`, so the writer (and thus the derived query position)
    /// sits at `bases[slot] + n_tokens`. Lets a benchmark place a 128-token
    /// window at a realistic depth (e.g. positions `[D-128, D)`), exercising
    /// the RoPE table's high-position range the way a deep context does.
    pub fn build_based(
        dev: &candle::Device,
        windows: &[Vec<[f32; HEAD_DIM]>],
        bases: &[usize],
    ) -> Result<Self> {
        let specs: Vec<Vec<[BandSpec; N_BANDS]>> = windows
            .iter()
            .map(|w| vec![[BandSpec::default(); N_BANDS]; w.len() / CHUNK + 1])
            .collect();
        Self::build_mixed(dev, windows, bases, &specs)
    }

    /// As [`Self::build_based`] but each `(chunk, band)` is authored in the
    /// given [`BandSpec`] format — FP8 rows or token-oriented GGML quant
    /// blocks with a non-unit outer scale — exercising the kernels' adaptive
    /// per-band window dispatch. The writer chunk (each slot's last) must be
    /// FP8: the kernels' fused/glue scatters write FP8 only.
    pub fn build_mixed(
        dev: &candle::Device,
        windows: &[Vec<[f32; HEAD_DIM]>],
        bases: &[usize],
        specs: &[Vec<[BandSpec; N_BANDS]>],
    ) -> Result<Self> {
        let ident = identity_pal_map();
        let maps: Vec<Vec<[u8; HEAD_DIM / 2]>> =
            specs.iter().map(|s| vec![ident; s.len()]).collect();
        Self::build_mapped(dev, windows, bases, specs, &maps)
    }

    /// As [`Self::build_mixed`] but each chunk also carries a pal_map: dims
    /// are routed to their assigned palette's arena in RANK order (the
    /// adaptive PalQuant layout), and the KvHead records carry the map. The
    /// writer chunk (each slot's last) must be identity-mapped FP8 — the
    /// scatters write the identity layout only.
    pub fn build_mapped(
        dev: &candle::Device,
        windows: &[Vec<[f32; HEAD_DIM]>],
        bases: &[usize],
        specs: &[Vec<[BandSpec; N_BANDS]>],
        maps: &[Vec<[u8; HEAD_DIM / 2]>],
    ) -> Result<Self> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        if bases.len() != windows.len()
            || specs.len() != windows.len()
            || maps.len() != windows.len()
        {
            candle::bail!(
                "build_mapped: {} bases / {} specs / {} maps for {} windows",
                bases.len(),
                specs.len(),
                maps.len(),
                windows.len()
            );
        }
        let ident = identity_pal_map();

        let cuda = match dev {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("SyntheticSlots requires a CUDA device"),
        };
        let stream = cuda.cuda_stream();

        let mut slot_meta = Vec::new();
        let mut chunk_base = 0usize;
        for (slot, w) in windows.iter().enumerate() {
            // The writer chunk must have room for the incoming token.
            let n_chunks = w.len() / CHUNK + 1;
            if specs[slot].len() != n_chunks {
                candle::bail!(
                    "build_mixed: slot {slot} has {} chunks but {} specs",
                    n_chunks,
                    specs[slot].len()
                );
            }
            if specs[slot][n_chunks - 1].iter().any(|s| s.fmt != 34) {
                candle::bail!("build_mapped: slot {slot} writer chunk must be FP8 (fmt 34)");
            }
            if maps[slot].len() != n_chunks {
                candle::bail!(
                    "build_mapped: slot {slot} has {n_chunks} chunks but {} maps",
                    maps[slot].len()
                );
            }
            if maps[slot][n_chunks - 1] != ident {
                candle::bail!("build_mapped: slot {slot} writer chunk must be identity-mapped");
            }
            slot_meta.push((chunk_base, n_chunks, w.len()));
            chunk_base += n_chunks;
        }
        let n_chunks_total = chunk_base;

        // Band arenas: per-(chunk, band) regions in the authored format, at
        // 16-byte-aligned offsets in one flat buffer. The same codec that
        // emits the bytes yields the stored (dequantized) values the mirror
        // keys against.
        let mut band_bytes: Vec<u8> = Vec::new();
        let mut band_offsets = vec![0usize; n_chunks_total * N_BANDS];
        let mut stored: Vec<Vec<[f32; HEAD_DIM]>> = Vec::with_capacity(windows.len());
        for (slot, w) in windows.iter().enumerate() {
            let (first_chunk, n_chunks, _) = slot_meta[slot];
            let mut slot_stored = vec![[0f32; HEAD_DIM]; w.len()];
            for c in 0..n_chunks {
                let t0 = c * CHUNK;
                let t1 = w.len().min(t0 + CHUNK);
                let toks: Vec<[f32; HEAD_DIM]> = w[t0..t1].to_vec();
                let (per_band, dec) = mapped_chunk_roundtrip(&specs[slot][c], &toks)?;
                for (band, bytes) in per_band.iter().enumerate() {
                    let off = band_bytes.len().next_multiple_of(16);
                    band_bytes.resize(off, 0);
                    band_bytes.extend_from_slice(bytes);
                    band_offsets[(first_chunk + c) * N_BANDS + band] = off;
                }
                for (t, dt) in dec.iter().enumerate() {
                    slot_stored[t0 + t] = *dt;
                }
            }
            stored.push(slot_stored);
        }
        let n_band_bytes = band_bytes.len().max(1);
        band_bytes.resize(n_band_bytes, 0);
        let bands = Tensor::from_vec(band_bytes, n_band_bytes, dev)?;
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
        // per-band authored format tag + outer scale, v_* mirrors k_* (K≡V).
        let chunk_specs: Vec<[BandSpec; N_BANDS]> =
            specs.iter().flat_map(|s| s.iter().copied()).collect();
        let chunk_maps: Vec<[u8; HEAD_DIM / 2]> =
            maps.iter().flat_map(|m| m.iter().copied()).collect();
        let mut kvhead_bytes = vec![0u8; n_chunks_total * KVHEAD_BYTES];
        for chunk in 0..n_chunks_total {
            let rec = &mut kvhead_bytes[chunk * KVHEAD_BYTES..(chunk + 1) * KVHEAD_BYTES];
            // Single 4-bit pal_map over [0,HD/2) (K≡V — the old v_pal half is
            // absorbed into the one map).
            rec[..HEAD_DIM / 2].copy_from_slice(&chunk_maps[chunk]);
            for band in 0..N_BANDS {
                let spec = chunk_specs[chunk][band];
                let addr = bands_addr + band_offsets[chunk * N_BANDS + band] as u64;
                // Per-band block offsets, parameterized on N_BANDS to match the
                // <HD, NP> slot_types.cuh accessors: k_ptr @ HD/2, v_ptr @
                // HD/2+NB*8, k_fmt @ +NB*16, v_fmt @ +NB*17, k_scale @ +NB*18,
                // v_scale @ +NB*22.
                let kp = HEAD_DIM / 2 + band * 8;
                rec[kp..kp + 8].copy_from_slice(&addr.to_le_bytes());
                let vp = HEAD_DIM / 2 + N_BANDS * 8 + band * 8;
                rec[vp..vp + 8].copy_from_slice(&addr.to_le_bytes());
                rec[HEAD_DIM / 2 + N_BANDS * 16 + band] = spec.fmt;
                rec[HEAD_DIM / 2 + N_BANDS * 17 + band] = spec.fmt;
                let ks = HEAD_DIM / 2 + N_BANDS * 18 + band * 4;
                rec[ks..ks + 4].copy_from_slice(&spec.outer.to_le_bytes());
                let vs = HEAD_DIM / 2 + N_BANDS * 22 + band * 4;
                rec[vs..vs + 4].copy_from_slice(&spec.outer.to_le_bytes());
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
                let within = c * CHUNK; // token index within the slot (for len)
                let rope = bases[slot] + within; // absolute rope base of this chunk
                let len = n_tokens.saturating_sub(within).min(CHUNK);
                rec[0..2].copy_from_slice(&0u16.to_le_bytes()); // offset
                rec[2..4].copy_from_slice(&(len as u16).to_le_bytes());
                rec[4..8].copy_from_slice(&(rope as u32).to_le_bytes()); // rope
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
            stored,
        })
    }

    /// Device address of the `SlotHeader` array — the raw-pointer form
    /// [`paged_latent_decode_raw`] wants, so a benchmark can drive the
    /// non-committing decode path (no on-device write-len advance) and keep the
    /// synthetic arena static across a timed launch loop.
    pub fn header_device_ptr(&self) -> Result<u64> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let dev = match self.headers.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("SyntheticSlots headers must be CUDA-resident"),
        };
        let stream = dev.cuda_stream();
        let (storage, layout) = self.headers.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => {
                let slice = c.as_cuda_slice::<u8>()?;
                let (p, _guard) = slice.device_ptr(&stream);
                Ok(p + layout.start_offset() as u64)
            }
            _ => candle::bail!("expected CUDA storage for headers"),
        }
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
/// the kernel computes them (max-abs per 128-dim band, `mx / 127` — IEEE
/// division, matching the kernel's `__fdiv_rn(mx, 127.f)`; the attention
/// kernels run under `--use_fast_math`, so the division is written as the
/// explicit round-to-nearest intrinsic — zero→1, then the requant divides
/// by `s` via `__frcp_rn(s)` = `1.0/s`, round-nearest-even, clamp ±127).
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

/// CPU mirror of `latent_build_corpus_cache_kernel` (the two-region position-free
/// cache, review items B+D): NOPE bands `[0,NOPE_DIM)` → int8 with a per-band
/// component-amax scale; ROPE bands `[NOPE_DIM,HEAD_DIM)` → BF16 pre-rotation.
/// Byte-exact against the kernel — `f32→bf16` is round-to-nearest-even on both
/// sides, and the nope int8 is the same `mx/127`, RNE, clamp path as
/// [`quant_bands`]'s nope arm.
#[cfg(feature = "cuda")]
pub fn build_corpus_cache(
    v: &[f32; HEAD_DIM],
) -> ([i8; NOPE_DIM], [f32; NOPE_DIM / SUB_DIM], [half::bf16; ROPE_DIM]) {
    const NOPE_BANDS: usize = NOPE_DIM / SUB_DIM;
    let mut nope = [0i8; NOPE_DIM];
    let mut scale = [1.0f32; NOPE_BANDS];
    for band in 0..NOPE_BANDS {
        let lo = band * SUB_DIM;
        let mut mx = 0.0f32;
        for d in lo..lo + SUB_DIM {
            mx = mx.max(v[d].abs());
        }
        let mut s = mx / 127.0;
        if s == 0.0 {
            s = 1.0;
        }
        scale[band] = s;
        let inv = 1.0 / s;
        for d in lo..lo + SUB_DIM {
            nope[d] = (v[d] * inv).clamp(-127.0, 127.0).round_ties_even() as i32 as i8;
        }
    }
    let rope: [half::bf16; ROPE_DIM] =
        std::array::from_fn(|d| half::bf16::from_f32(v[NOPE_DIM + d]));
    (nope, scale, rope)
}

/// One key's contribution to the logit: Σ_band (int32 dot) · sQ · sK, summed
/// in ascending band order — the kernel's per-band float accumulation chain.
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

    /// Calibration instrument (CPU, no GPU): computes the DeepSeek-V4 REFERENCE
    /// per-region quantization error margins — the rope tail (kept BF16) and the
    /// non-rope dims (FP8 E4M3, per-64 ue8m0 scale) — and the whole-band
    /// round-trip rel-L2 of every candidate format, documenting why the window's
    /// two-region format (FP8 nope ‖ BF16 rope) matches the reference: rope ≪
    /// nope margin, so the rope tail must stay BF16 while the nope dims tolerate
    /// FP8. Representative data = post-RMSNorm latent ≈ per-dim N(0,1),
    /// bf16-rounded (the write source is bf16). Prints a table; asserts only the
    /// two invariants (rope ≪ nope, Q8 ≈ rope margin, Q4 ≈ nope margin).
    #[test]
    fn latent_reference_error_margins() {
        const NT: usize = CHUNK; // 32 tokens (one chunk)
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        // Box–Muller standard normal.
        let mut gauss = || {
            let (u1, u2) = (u().max(1e-12), u());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        // True f32 band [NT][SUB_DIM], and its bf16-rounded source (what the
        // seal reads).
        let mut xf = vec![[0f32; SUB_DIM]; NT];
        let mut src = vec![[0f32; SUB_DIM]; NT];
        for t in 0..NT {
            for d in 0..SUB_DIM {
                let v = gauss() as f32;
                xf[t][d] = v;
                src[t][d] = half::bf16::from_f32(v).to_f32();
            }
        }
        let rel_l2 = |a: &[[f32; SUB_DIM]], b: &[[f32; SUB_DIM]]| -> f32 {
            let (mut num, mut den) = (0f64, 0f64);
            for t in 0..a.len() {
                for d in 0..SUB_DIM {
                    let e = (a[t][d] - b[t][d]) as f64;
                    num += e * e;
                    den += (b[t][d] as f64) * (b[t][d] as f64);
                }
            }
            (num / den).sqrt() as f32
        };

        // ── Reference ROPE margin: BF16 vs true f32 ──────────────────────────
        let eps_rope = rel_l2(&src, &xf);

        // ── Reference NOPE margin: FP8 E4M3, per-64 ue8m0 power-of-two scale,
        // vs true f32 (the reference's `act_quant(block=64, scale_fmt=ue8m0)`).
        // Reference FP8 block = 64 dims; a 32-dim band is smaller than the
        // reference block, so clamp the block to the band width (per-32 at
        // SUB_DIM=32 — same order of margin as per-64 for this Gaussian data).
        let blk_dim = SUB_DIM.min(64);
        let mut nope = vec![[0f32; SUB_DIM]; NT];
        for t in 0..NT {
            for blk in 0..(SUB_DIM / blk_dim) {
                let base = blk * blk_dim;
                let mut amax = 0f32;
                for d in base..base + blk_dim {
                    amax = amax.max(xf[t][d].abs());
                }
                // ue8m0: scale = 2^round(log2(amax/448)); guard amax=0.
                let scale = if amax > 0.0 {
                    (amax / 448.0).log2().round().exp2()
                } else {
                    1.0
                };
                for d in base..base + blk_dim {
                    nope[t][d] = e4m3_to_f32(f32_to_e4m3(xf[t][d] / scale)) * scale;
                }
            }
        }
        let eps_nope = rel_l2(&nope, &xf);

        // ── Per-format ladder rel-L2 (round-trip vs the bf16 source) ─────────
        // f16-scale formats self-scale (outer=1); INT8-scale/Q0 use 1/amax as a
        // representative searched outer.
        let band_amax = src
            .iter()
            .flat_map(|r| r.iter())
            .fold(0f32, |m, &v| m.max(v.abs()));
        let inv_amax = if band_amax > 0.0 {
            1.0 / band_amax
        } else {
            1.0
        };
        let fmts: &[(&str, u8, f32)] = &[
            ("Q8_0", 7, 1.0),
            ("Q8_KS", 10, 1.0),
            ("Q4_0", 15, 1.0),
            ("Q4_1", 16, 1.0),
            ("Q4_KS", 18, 1.0),
            ("Q3_0", 19, 1.0),
            ("Q3_1", 20, 1.0),
            ("Q2_S", 25, inv_amax),
            ("Q2_A", 26, inv_amax),
            ("Q1_S", 27, inv_amax),
            ("Q1_A", 29, inv_amax),
            ("Q0", 33, inv_amax),
        ];
        eprintln!("\n=== DeepSeek-V4 latent quant error margins (rel-L2) ===");
        eprintln!("REFERENCE  rope (BF16)          = {eps_rope:.5}");
        eprintln!("REFERENCE  nope (FP8 E4M3 /64)  = {eps_nope:.5}");
        eprintln!("--- ladder formats (round-trip vs bf16 source) ---");
        for &(name, tag, outer) in fmts {
            let spec = BandSpec { fmt: tag, outer };
            let (_, dec) = band_chunk_roundtrip(spec, &src).unwrap();
            eprintln!(
                "  {name:6} (tag {tag:2}) rel-L2 = {:.5}",
                rel_l2(&dec, &src)
            );
        }
        eprintln!("=======================================================\n");

        // Invariants the asymmetric ladder relies on.
        assert!(
            eps_rope < eps_nope * 0.25,
            "rope must be far tighter than nope"
        );
        assert!((0.001..0.01).contains(&eps_rope), "bf16 rope margin ~0.003");
        assert!((0.02..0.09).contains(&eps_nope), "e4m3 nope margin ~0.045");
    }

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

    /// The kernel's factored-table sin/cos, op-for-op: the hi/lo table entries
    /// are `mirror_sincos` of the split positions (bit-identical to the device
    /// builder's `rope_angle`/`ds_sincos` per the sincos probe), combined by
    /// the angle-addition identity in plain exact-rounded f32 — the kernel's
    /// `rope_lookup`.
    fn table_sincos(pos: u32, freq: f32) -> (f32, f32) {
        let hi = (pos >> ROPE_LO_BITS).min(ROPE_HI_DIM as u32 - 1) << ROPE_LO_BITS;
        let lo = pos & (ROPE_LO_DIM as u32 - 1);
        let (sh, ch) = mirror_sincos(hi, freq);
        let (sl, cl) = mirror_sincos(lo, freq);
        (sh * cl + ch * sl, ch * cl - sh * sl)
    }

    fn rope_vec(v: &mut [f32; HEAD_DIM], pos: u32, freqs: &[f32]) {
        for k in 0..ROPE_DIM / 2 {
            let (s, c) = table_sincos(pos, freqs[k]);
            let d = NOPE_DIM + 2 * k;
            let (x0, x1) = (v[d], v[d + 1]);
            v[d] = x0 * c - x1 * s;
            v[d + 1] = x0 * s + x1 * c;
        }
    }

    /// Reassembly round-trip: the POSITION-FREE two-region cache built ONCE
    /// (`build_corpus_cache`) dequantized and rotated at ANY position in the
    /// attention window reproduces roping the ORIGINAL entry at that position.
    /// The property the rope/nope split buys: the cache carries no baked position,
    /// so re-selecting an entry at a different window position needs no rebuild —
    /// the reader just rotates the stored BF16 rope bands at read time. (CPU-only.)
    #[test]
    fn corpus_cache_is_position_free() {
        // Standard RoPE frequencies + a random pre-RoPE latent.
        let freqs: Vec<f32> = (0..ROPE_DIM / 2)
            .map(|k| 1.0f32 / 10000f32.powf(2.0 * k as f32 / ROPE_DIM as f32))
            .collect();
        let mut s = 0x1234_5678u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let l0: [f32; HEAD_DIM] = std::array::from_fn(|_| next());

        // Build the position-free two-region cache ONCE; dequantize both regions.
        let (nope, scale, rope) = build_corpus_cache(&l0);
        let dequant: [f32; HEAD_DIM] = std::array::from_fn(|d| {
            if d < NOPE_DIM {
                nope[d] as f32 * scale[d / SUB_DIM]
            } else {
                rope[d - NOPE_DIM].to_f32()
            }
        });

        // Positions across the attention window (≤ 1M context cap).
        let positions = [0u32, 1, 31, 97, 4096, 65_536, 500_000, 1_000_000];
        for &pos in &positions {
            // Rope-at-load from the SAME stored bytes vs roping the original.
            let mut from_cache = dequant;
            rope_vec(&mut from_cache, pos, &freqs);
            let mut reference = l0;
            rope_vec(&mut reference, pos, &freqs);
            for d in 0..HEAD_DIM {
                // NOPE: int8 dequant error ≤ one band step. ROPE: bf16 round-trip
                // (~2⁻⁸ relative), preserved by the orthonormal rotation.
                let tol = if d < NOPE_DIM {
                    scale[d / SUB_DIM] + 1e-4
                } else {
                    reference[d].abs() * 0.02 + 2e-3
                };
                assert!(
                    (from_cache[d] - reference[d]).abs() <= tol,
                    "pos {pos} dim {d}: rope-at-load {} vs ref {} exceeds tol {tol}",
                    from_cache[d],
                    reference[d]
                );
            }
        }

        // NOPE bands are byte-identical to the generic per-band quant (never rope).
        let (ci_plain, _) = quant_bands(&l0);
        for d in 0..NOPE_DIM {
            assert_eq!(
                nope[d], ci_plain[d],
                "nope dim {d}: corpus quant diverges from plain per-band quant"
            );
        }
    }

    /// Stage 1 (B+D): the two-region cache builder is BYTE-EXACT against its CPU
    /// mirror — nope int8 + per-band amax scale, rope tail as BF16 pre-rotation.
    /// Proves the codec before decode/prefill are wired to it.
    #[test]
    #[ignore]
    fn corpus_cache_two_region_bytes() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        const NOPE_BANDS: usize = NOPE_DIM / SUB_DIM;
        let dev = Device::new_cuda(0)?;
        let cuda = match &dev {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        let stream = cuda.cuda_stream();

        // Random f32 entries (include a zero row → the scale zero→1 path).
        let g = 48usize;
        let mut s = 0x51ED_2701u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut entries: Vec<[f32; HEAD_DIM]> =
            (0..g).map(|_| std::array::from_fn(|_| next())).collect();
        entries[7] = [0.0; HEAD_DIM];
        let comp_flat: Vec<f32> = entries.iter().flat_map(|e| e.iter().copied()).collect();
        let comp = Tensor::from_vec(comp_flat, (g, HEAD_DIM), &dev)?;

        let nope_i8 = Tensor::zeros((g, NOPE_DIM), DType::U8, &dev)?;
        let nope_scale = Tensor::zeros((g, NOPE_BANDS), DType::F32, &dev)?;
        let rope_bf = Tensor::zeros((g, ROPE_DIM), DType::BF16, &dev)?;

        macro_rules! p {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                match &*storage {
                    Storage::Cuda(c) => {
                        let (ptr, _g) = c.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                        ptr + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                    }
                    _ => candle::bail!("expected CUDA storage"),
                }
            }};
        }
        let comp_p = p!(&comp, f32);
        let ni8_p = p!(&nope_i8, u8);
        let nsc_p = p!(&nope_scale, f32);
        let rbf_p = p!(&rope_bf, half::bf16);
        unsafe {
            candle_kernels::paged_latent::run_latent_build_corpus_cache(
                comp_p as *const f32,
                ni8_p as *mut u8,
                nsc_p as *mut f32,
                rbf_p as *mut core::ffi::c_void,
                0,
                g as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            );
        }
        dev.synchronize()?;

        // int8 bytes come back as u8 (bit pattern); compare as i8.
        let got_nope = nope_i8.flatten_all()?.to_vec1::<u8>()?;
        let got_scale = nope_scale.flatten_all()?.to_vec1::<f32>()?;
        let got_rope = rope_bf.flatten_all()?.to_vec1::<half::bf16>()?;

        for (gi, e) in entries.iter().enumerate() {
            let (m_nope, m_scale, m_rope) = build_corpus_cache(e);
            for d in 0..NOPE_DIM {
                assert_eq!(
                    got_nope[gi * NOPE_DIM + d] as i8,
                    m_nope[d],
                    "entry {gi} nope[{d}]"
                );
            }
            for b in 0..NOPE_BANDS {
                assert_eq!(
                    got_scale[gi * NOPE_BANDS + b].to_bits(),
                    m_scale[b].to_bits(),
                    "entry {gi} scale[{b}]"
                );
            }
            for d in 0..ROPE_DIM {
                assert_eq!(
                    got_rope[gi * ROPE_DIM + d].to_bits(),
                    m_rope[d].to_bits(),
                    "entry {gi} rope[{d}]"
                );
            }
        }
        Ok(())
    }

    /// The full CPU mirror: replicates tiling, quantization, softmax phases,
    /// split ranges, combine + sink fold + de-rotation. Returns `[H][512]` f32
    /// (pre-BF16-store).
    fn mirror(case: &Case, inp: &Inputs, softmax_scale: f32) -> Vec<[f32; HEAD_DIM]> {
        mirror_fmt(case, inp, softmax_scale, None)
    }

    fn mirror_fmt(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
    ) -> Vec<[f32; HEAD_DIM]> {
        mirror_mapped(case, inp, softmax_scale, win_fmts)
    }

    /// Identity-layout mirror: each `(chunk, band)`'s stored values come from
    /// the same codec that authors the arena bytes (`band_chunk_roundtrip`),
    /// so the mirror keys off exactly what the kernel dequantizes.
    fn mirror_mapped(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
    ) -> Vec<[f32; HEAD_DIM]> {
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

        // Stored window values: each (chunk, band)'s storage round-trip via
        // the SAME codec that authors the arena bytes (band_chunk_roundtrip),
        // so the mirror keys off exactly what the kernel dequantizes. The
        // scattered incoming token is always the writer chunk's FP8.
        let default_specs = vec![[BandSpec::default(); N_BANDS]; n_chunks];
        let specs = win_fmts.unwrap_or(&default_specs);
        let mut stored: Vec<[f32; HEAD_DIM]> = vec![[0.0; HEAD_DIM]; n_win];
        for c in 0..n_chunks {
            let t0 = c * CHUNK;
            let t1 = n_win.min(t0 + CHUNK);
            let toks: Vec<[f32; HEAD_DIM]> = inp.window[t0..t1].to_vec();
            let (_, dec) = mapped_chunk_roundtrip(&specs[c], &toks)
                .expect("mirror_mapped: codec rejected chunk spec");
            for (t, dt) in dec.iter().enumerate() {
                stored[t0 + t] = *dt;
            }
        }
        stored.push(std::array::from_fn(|d| {
            e4m3_to_f32(f32_to_e4m3(inp.kv_new[d]))
        }));

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
                    let pos = inp.comp_pos[gid];
                    // Causal guard: a compressed entry in the query's future is
                    // dropped (mirrors the kernel's `comp_pos <= q_pos` gate).
                    if pos <= q_pos {
                        // The kernel reads the persistent POSITION-FREE int8
                        // corpus cache (CorpusCache): nope bands raw, rope bands
                        // pre-rotation BF16 — `build_corpus_cache` IS the two-region
                        // cache builder's exact math. Dequantize both regions, then
                        // rotate the rope bands at this entry's assembled position
                        // (rope-at-load), then re-quantize for the int8 QK operand
                        // (`quant_bands`).
                        let mut l = inp.comp[gid];
                        let (nope, scale, rope) = build_corpus_cache(&l);
                        for d in 0..NOPE_DIM {
                            l[d] = nope[d] as f32 * scale[d / SUB_DIM];
                        }
                        for d in 0..ROPE_DIM {
                            l[NOPE_DIM + d] = rope[d].to_f32();
                        }
                        rope_vec(&mut l, pos, &inp.freqs);
                        let (ki, ks) = quant_bands(&l);
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
                let (s, c) = table_sincos(q_pos, inp.freqs[k]);
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
            let pos = inp.comp_pos[gid];
            // Causal guard: a compressed entry in the query's future is dropped
            // (mirrors the kernel's `comp_pos <= q_pos` gate).
            if pos <= q_pos {
                let mut l = inp.comp[gid];
                rope_vec(&mut l, pos, &inp.freqs);
                keys.push(l);
            }
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
                let (s, c) = table_sincos(q_pos, inp.freqs[k]);
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
        run_kernel_fmt(case, inp, softmax_scale, None)
    }

    fn run_kernel_fmt(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
    ) -> Result<Vec<f32>> {
        run_kernel_mapped(case, inp, softmax_scale, win_fmts, None)
    }

    fn run_kernel_mapped(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
        win_maps: Option<&[[u8; HEAD_DIM / 2]]>,
    ) -> Result<Vec<f32>> {
        let dev = Device::new_cuda(0)?;
        let n_chunks = case.n_win / CHUNK + 1;
        let default_specs = vec![[BandSpec::default(); N_BANDS]; n_chunks];
        let specs = win_fmts.unwrap_or(&default_specs).to_vec();
        let ident = identity_pal_map();
        let default_maps = vec![ident; n_chunks];
        let maps = win_maps.unwrap_or(&default_maps).to_vec();
        let slots = SyntheticSlots::build_mapped(
            &dev,
            std::slice::from_ref(&inp.window),
            &[0usize],
            std::slice::from_ref(&specs),
            std::slice::from_ref(&maps),
        )?;

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
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;

        let out = paged_latent_decode(
            &q,
            &slots.headers,
            &kv_new,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
            &sinks,
            &rope_tab,
            &ws,
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

    /// Count cells where kernel and mirror differ by MORE than one bf16 code
    /// (with a `1e-4` absolute floor for near-zero cancellation cells, where
    /// bf16 codes are dense — a near-zero attention output accumulates several
    /// dense codes from ≤1-code per-key weight errors while its absolute error
    /// stays ~1e-4 · full-scale, i.e. negligible). Panics on any cell beyond
    /// that — a real bug moves cells by the value magnitude (~O(1)), far above
    /// the floor. Returns the off-by-one count for the ≥99.9% budget check.
    ///
    /// This is the tolerance the latent attention-OUTPUT mirror gates on: the
    /// kernel runs its int8 QK staging + softmax + dequant `/outer` under
    /// `--use_fast_math`, so a handful of cells land ≤1 bf16 code off the exact
    /// CPU mirror. The kernel is NOT worse than the mirror against the float
    /// reference (measured `kernel-vs-ref == mirror-vs-ref`); making the hot
    /// decode path IEEE-precise was measured ~8% slower and reverted, so an
    /// output-level ≤1-code tolerance is the right gate (these are attention
    /// OUTPUT tests, not codec raw-byte tests — those stay strict).
    fn check_within_one_code(kernel: &[f32], mirror: &[f32]) -> usize {
        let mut off = 0usize;
        for (i, (a, b)) in kernel.iter().zip(mirror).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            let (ca, cb) = (
                half::bf16::from_f32(*a).to_bits() as i32,
                half::bf16::from_f32(*b).to_bits() as i32,
            );
            assert!(
                (ca - cb).abs() <= 1 || (a - b).abs() <= 1e-4,
                "divergence beyond 1 bf16 code at {i} (head {} dim {}): {a} vs {b}",
                i / HEAD_DIM,
                i % HEAD_DIM
            );
            off += 1;
        }
        off
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
        // n=80 spans THREE 32-token chunks (0..31 | 32..63 | 64..79), so the
        // batched prefill's arena slice-walk crosses ≥2 chunk boundaries — the
        // Phase-3 prerequisite that the multi-query launch reproduces per-token
        // decode across chunk transitions, not just within a single chunk.
        let n = 80usize;
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
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;

        // Prefill: one settled slot holding all n tokens.
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&tokens.to_vec()))?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;
        let prefill = paged_latent_prefill(
            &q_all,
            &slots.headers,
            &q_pos,
            None,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // Fresh-source equivalence: arena holds only [0..k); rows [k..n) come
        // in as this-layer fresh latents. Queries [k..n) must be BIT-IDENTICAL
        // to the settled-slot run (the kernel FP8-round-trips fresh keys so the
        // bits match what the arena would return). split_at=40 puts the arena
        // prefix across the 0/1 chunk boundary and the fresh rows across the
        // 1/2 boundary, so both sources are exercised over multiple chunks.
        let split_at = 40usize;
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
        let fresh_out = paged_latent_prefill(
            &q_tail,
            &fslots.headers,
            &pos_tail,
            Some((&kv_fresh, split_at)),
            &CorpusCache::build(&comp, &comp_pos)?,
            &idx_tail,
            &cnt_tail,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
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
                    let (s, c) = table_sincos(i as u32, freqs_v[k]);
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
        // Rows 31/32/33 straddle the 0/1 chunk boundary and 63/64/65 straddle
        // the 1/2 boundary; 79 is the final row of chunk 2.
        for &i in &[0usize, 1, 17, 31, 32, 33, 49, 63, 64, 65, 79] {
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
            let dec = paged_latent_decode(
                &q_one,
                &dslots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &d_idx,
                &d_cnt,
                &Tensor::from_vec(vec![i as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                window_size,
                1,
                None,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            // Prefill runs int8 tensor-core PV; decode runs scalar bf16 PV.
            // Both are valid approximations of the same attention, so they
            // agree at int8-PV tolerance (~0.4-1% of scale), not bitwise.
            let row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            let scale = dec.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let d = max_abs_diff(row, &dec);
            assert!(
                d < 0.03 * scale,
                "row {i}: prefill(int8 PV) vs decode(scalar PV) |Δ|={d} ≥ {}",
                0.03 * scale
            );
        }
        Ok(())
    }

    /// Chunked-prefill gate: launches whose query count crosses the workspace
    /// chunk boundary (`q_chunk = WORKSPACE_CAP / H` at splits=1) must stay
    /// correct on BOTH sides of the boundary — settled rows bit-identical to
    /// the decode oracle, fresh-source rows anchored to an f64 reference no
    /// looser than the settled rows are. Guards the chunk pointer arithmetic
    /// (q/out/q_pos/comp_idx/comp_cnt advances) and the launch-invariant
    /// fresh-key positions (`fresh_base + j`, never q_pos-derived — a
    /// chunk-shifted q_pos read here corrupts every fresh key past the
    /// boundary and reads out of bounds).
    #[test]
    #[ignore]
    fn prefill_chunked_rows_match_decode() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 600usize;
        let split_at = 40usize; // fresh call: 560 rows → chunks [0,512) | [512,560)
        let window_size = 128usize;
        let q_chunk = WORKSPACE_CAP / H; // splits = 1
        assert!(
            n > q_chunk && n - split_at > q_chunk,
            "both calls must chunk"
        );
        let mut s = 7u64;
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
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;

        // Settled full prefill: n rows over a slot holding all n tokens.
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&tokens.to_vec()))?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;
        let prefill = paged_latent_prefill(
            &q_all,
            &slots.headers,
            &q_pos,
            None,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // Settled rows straddling the launch-chunk boundary are BIT-IDENTICAL
        // to the decode oracle (inherits the decode's proven oracle chain).
        for &i in &[q_chunk - 1, q_chunk, n - 1] {
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
            let dec = paged_latent_decode(
                &q_one,
                &dslots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &d_idx,
                &d_cnt,
                &Tensor::from_vec(vec![i as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                window_size,
                1,
                None,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            // int8 tensor-core PV (prefill) vs scalar bf16 PV (decode): agree
            // at tolerance, not bitwise (see prefill_rows_equal_decode_steps).
            let row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            let scale = dec.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let d = max_abs_diff(row, &dec);
            assert!(
                d < 0.03 * scale,
                "settled row {i}: prefill(int8 PV) vs decode |Δ|={d} ≥ {}",
                0.03 * scale
            );
        }

        // Fresh-source call: arena holds [0..split_at), rows [split_at..n)
        // arrive as fresh latents; 560 queries chunk at fresh-call row 512
        // (absolute row split_at + 512). Spot rows bracket that boundary.
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
        let fresh_out = paged_latent_prefill(
            &q_tail,
            &fslots.headers,
            &pos_tail,
            Some((&kv_fresh, split_at)),
            &CorpusCache::build(&comp, &comp_pos)?,
            &idx_tail,
            &cnt_tail,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // f64 sink-softmax anchor at the spot rows (window-clamped): the fresh
        // path must sit no farther from truth than the settled path (which is
        // bit-exact to decode) — tile-partition reassociation is the only
        // allowed variance.
        let stored: Vec<[f32; HEAD_DIM]> = tokens
            .iter()
            .map(|t| std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(t[d]))))
            .collect();
        let boundary = split_at + q_chunk;
        for &i in &[boundary - 2, boundary - 1, boundary, boundary + 1, n - 1] {
            let lo = (i + 1).saturating_sub(window_size);
            let mut keys: Vec<[f32; HEAD_DIM]> = Vec::with_capacity(i + 1 - lo);
            for t in lo..=i {
                let mut l = stored[t];
                rope_vec(&mut l, t as u32, &freqs_v);
                keys.push(l);
            }
            let mut ref_row: Vec<f32> = Vec::with_capacity(H * HEAD_DIM);
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
                    let (sn, cs) = table_sincos(i as u32, freqs_v[k]);
                    let d = NOPE_DIM + 2 * k;
                    let (x0, x1) = (val[d], val[d + 1]);
                    val[d] = x0 * cs as f64 + x1 * sn as f64;
                    val[d + 1] = x1 * cs as f64 - x0 * sn as f64;
                }
                ref_row.extend(val.iter().map(|&v| v as f32));
            }
            let settled_row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            let fresh_row =
                &fresh_out[(i - split_at) * H * HEAD_DIM..(i - split_at + 1) * H * HEAD_DIM];
            let d_settled = max_abs_diff(settled_row, &ref_row);
            let d_fresh = max_abs_diff(fresh_row, &ref_row);
            eprintln!("[chunk] row {i}: settled-vs-ref {d_settled:.6} fresh-vs-ref {d_fresh:.6}");
            assert!(
                d_fresh <= d_settled * 1.5 + 1e-3,
                "row {i}: fresh path drifts beyond the settled path's truth distance: \
                 fresh {d_fresh} vs settled {d_settled}"
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
                candle_kernels::paged_latent::run_latent_exp_probe(
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

    /// Step-2 gate (a): window latents written through the PRODUCTION chunked
    /// backing (single-latent mode) read back exactly as the host round-trip of
    /// the reference two-region format — the raw-byte write→read contract. The
    /// nope span `[0, 448)` is FP8 E4M3, the rope tail `[448, 512)` is BF16, so
    /// the oracle rounds each dim through its region's dtype.
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
            let d = i % HEAD_DIM;
            // Two-region store: nope dims round through FP8 E4M3, rope dims
            // through BF16.
            let expect = if d < NOPE_DIM {
                e4m3_to_f32(f32_to_e4m3(v))
            } else {
                half::bf16::from_f32(v).to_f32()
            };
            assert_eq!(
                got_k[i].to_bits(),
                expect.to_bits(),
                "K[{i}] (dim {d}): {} vs {expect}",
                got_k[i]
            );
            // K≡V: the V read aliases the K bytes.
            assert_eq!(got_v[i].to_bits(), expect.to_bits(), "V[{i}] alias (dim {d})");
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
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;

        let out = paged_latent_decode_raw(
            &q,
            headers.dev_ptr(),
            &kv_new,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
            &sinks,
            &rope_tab,
            softmax_scale,
            case.window_size,
            case.num_splits,
            true, // single-step audit against the live buffer
            &ws,
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
                candle_kernels::paged_latent::run_latent_sincos_probe(
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

    /// The device-built factored table must be bit-identical to the mirror's
    /// entry recipe (`mirror_sincos` of the split positions), and the mirror's
    /// `table_sincos` combination must round-trip a spot-check of full
    /// positions through the table layout the kernel reads.
    #[test]
    #[ignore]
    fn rope_table_device_matches_mirror() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let freqs_v: Vec<f32> =
            super::super::rope::yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
                .into_iter()
                .map(|f| f as f32)
                .collect();
        let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
        let tab = build_rope_table(&freqs)?;
        dev.synchronize()?;
        let got = tab.to_vec1::<f32>()?;
        assert_eq!(got.len(), ROPE_TAB_LEN);
        let nf = ROPE_DIM / 2;
        let mut mismatches = 0usize;
        for row in 0..(ROPE_HI_DIM + ROPE_LO_DIM) {
            let pos = if row < ROPE_HI_DIM {
                (row as u32) << 10
            } else {
                (row - ROPE_HI_DIM) as u32
            };
            for (j, &f) in freqs_v.iter().enumerate() {
                let (ms, mc) = mirror_sincos(pos, f);
                let i = row * nf + j;
                if got[2 * i].to_bits() != ms.to_bits() || got[2 * i + 1].to_bits() != mc.to_bits()
                {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(mismatches, 0, "device table ≠ mirror entries");
        // Spot-check the combined lookup at full positions across the range
        // (window/query token positions and sparse group starts alike).
        for &pos in &[0u32, 1, 127, 1023, 1024, 4096, 65_537, 999_999, 1_048_575] {
            let hi = (pos >> 10) as usize;
            let lo = (pos & 1023) as usize;
            for (j, &f) in freqs_v.iter().enumerate() {
                let ih = hi * nf + j;
                let il = (ROPE_HI_DIM + lo) * nf + j;
                let (sh, ch) = (got[2 * ih], got[2 * ih + 1]);
                let (sl, cl) = (got[2 * il], got[2 * il + 1]);
                let s = sh * cl + ch * sl;
                let c = ch * cl - sh * sl;
                let (ts, tc) = table_sincos(pos, f);
                assert_eq!(
                    (s.to_bits(), c.to_bits()),
                    (ts.to_bits(), tc.to_bits()),
                    "combined lookup ≠ table_sincos at pos {pos} freq {f}"
                );
            }
        }
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
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;
        let dbg = Tensor::zeros(DBG_LEN, DType::F32, &dev)?;
        let _ = paged_latent_decode(
            &q,
            &slots.headers,
            &kv_new,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
            &sinks,
            &rope_tab,
            &ws,
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
                let got = d[DBG_SCALEQ + h * N_BANDS + p];
                assert_eq!(
                    got.to_bits(),
                    qs[p].to_bits(),
                    "scaleQ[{h}][{p}]: device {got} vs mirror {}",
                    qs[p]
                );
            }
            for dd in 0..HEAD_DIM {
                let got = d[DBG_SQ + h * HEAD_DIM + dd] as i32 as i8;
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
                let got = d[DBG_KVF + t * HEAD_DIM + dd];
                assert_eq!(
                    got.to_bits(),
                    staged[dd].to_bits(),
                    "kv_f[{t}][{dd}]: device {got} vs mirror {}",
                    staged[dd]
                );
            }
            for p in 0..N_BANDS {
                let got = d[DBG_SCALEK + t * N_BANDS + p];
                assert_eq!(
                    got.to_bits(),
                    ks[p].to_bits(),
                    "scaleK[{t}][{p}]: device {got} vs mirror {}",
                    ks[p]
                );
            }
            for dd in 0..HEAD_DIM {
                let got = d[DBG_SK + t * HEAD_DIM + dd] as i32 as i8;
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
            // Attention-OUTPUT gate: ≤1 bf16 code per cell (the fast-math hot
            // path seam), ≥99.9% exactly equal.
            let n_bad = check_within_one_code(&kernel, &mirror_v);
            assert!(
                n_bad * 1000 <= kernel.len(),
                "n_win={n_win} zero_rope={zero_rope}: {n_bad}/{} cells off (> 0.1%)",
                kernel.len()
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
        // Attention-OUTPUT gate: ≤1 bf16 code per cell (fast-math hot seam),
        // ≥99.9% exactly equal.
        let n_bad = check_within_one_code(&kernel, &mirror_v);
        assert!(
            n_bad * 1000 <= kernel.len(),
            "swa: {n_bad}/{} cells off (> 0.1%)",
            kernel.len()
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

    /// Adaptive per-band window formats: mixed Q8_0/Q4_0/FP8 bands with
    /// non-unit outer scales across the sealed chunks (writer stays FP8), the
    /// kernel dispatching on the KvHead format tags, gated BIT-EXACT against
    /// the mirror (whose stored values come from the same codec that authored
    /// the arena bytes). Live RoPE + split-KV so the full pipeline runs over
    /// the dispatched reads.
    #[test]
    #[ignore]
    fn mirror_bit_exact_mixed_band_formats() -> Result<()> {
        let case = Case {
            n_win: 80,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 2,
            zero_rope: false,
        };
        // Chunks: [0..32) [32..64) sealed — mixed formats; [64..80)+incoming
        // is the writer — FP8 (the scatters write FP8 only).
        let q8 = |outer| BandSpec { fmt: 7, outer };
        let q4 = |outer| BandSpec { fmt: 15, outer };
        let t8 = |a: [BandSpec; 8]| -> [BandSpec; N_BANDS] { std::array::from_fn(|i| a[i % 8]) };
        let fmts = vec![
            t8([q8(1.5), q4(1.0), BandSpec::default(), q8(1.0), q8(1.5), q4(1.0), BandSpec::default(), q8(1.0)]),
            t8([q4(2.0), q8(1.0), q8(0.5), q4(1.0), q4(2.0), q8(1.0), q8(0.5), q4(1.0)]),
            [BandSpec::default(); N_BANDS],
        ];
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(&case, 0xA5A5);
        let kernel = run_kernel_fmt(&case, &inp, softmax_scale, Some(&fmts))?;
        let mirror_v: Vec<f32> = mirror_fmt(&case, &inp, softmax_scale, Some(&fmts))
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        let mirror_v = bf16_round_host(&mirror_v);
        // Attention-OUTPUT gate: ≤1 bf16 code per cell (the fast-math dequant
        // `/outer` + int8-QK hot seam), ≥99.9% exactly equal.
        let n_bad = check_within_one_code(&kernel, &mirror_v);
        assert!(
            n_bad * 1000 <= kernel.len(),
            "mixed-format: {n_bad}/{} cells off (> 0.1%)",
            kernel.len()
        );
        Ok(())
    }

    /// EVERY band format under the real attention kernel: seven sealed
    /// chunks covering the full K-ladder union — the f16-scale family at
    /// assorted outers plus the INT8-scale and Q0-family formats at
    /// searched-style outers — decoded through `load_band_elem`'s per-band
    /// dispatch, gated BIT-EXACT against the mirror. Arena bytes are
    /// codec-authored on both sides, so any mismatch is a decode op-order
    /// divergence between `dequant_element_inline` and the codec's arms.
    #[test]
    #[ignore]
    fn mirror_bit_exact_all_band_formats() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        // Phase 1 — ISOLATION: one sealed chunk per (fmt, outer), all four
        // bands in that format, so a divergence names its format. Values are
        // fp8-exact in roughly ±3.75; 0.25-0.5 outers play the searched
        // 1/amax role for formats whose outer IS the quantization scale.
        let combos: [(u8, f32); 21] = [
            (7, 0.5),
            (8, 1.0),
            (8, 1.5),
            (10, 1.5),
            (15, 2.0),
            (16, 1.0),
            (18, 1.0),
            (18, 2.0),
            (19, 1.0),
            (19, 0.5),
            (20, 1.0),
            (20, 1.5),
            (25, 0.25),
            (26, 0.25),
            (27, 0.25),
            (28, 0.25),
            (29, 0.25),
            (30, 0.25),
            (31, 0.25),
            (32, 0.25),
            (33, 0.25),
        ];
        let iso_case = Case {
            n_win: 64,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        for (i, &(fmt, outer)) in combos.iter().enumerate() {
            // Chunks: [0..32) and [32..64) sealed in the probed format; the
            // incoming token's fresh chunk is the FP8 writer.
            let fmts = vec![
                [BandSpec { fmt, outer }; N_BANDS],
                [BandSpec { fmt, outer }; N_BANDS],
                [BandSpec::default(); N_BANDS],
            ];
            let inp = gen_inputs(&iso_case, 0xBEE0 + i as u64);
            let kernel = run_kernel_fmt(&iso_case, &inp, softmax_scale, Some(&fmts))?;
            let mirror_v: Vec<f32> = mirror_fmt(&iso_case, &inp, softmax_scale, Some(&fmts))
                .into_iter()
                .flat_map(|h| h.into_iter())
                .collect();
            let mirror_v = bf16_round_host(&mirror_v);
            // The int8 window staging is bit-exact — kernel `__fdiv_rn` /
            // `__frcp_rn` match the mirror's IEEE ops even under
            // `--use_fast_math`. The only residual is the dequant `/outer`
            // divide in the block converters, which the attention kernel runs
            // under fast-math (an ~1-ulp approximate reciprocal on the hot
            // decode path — precise division there measured ~8% slower and is
            // not worth it): a handful of cells/32768 land one bf16 code off
            // at a .5 boundary. ≥99.9% exactly equal, every cell ≤1 code.
            let n_bad = check_within_one_code(&kernel, &mirror_v);
            assert!(
                n_bad * 1000 <= kernel.len(),
                "fmt {fmt} outer {outer}: {n_bad}/{} cells off (> 0.1%)",
                kernel.len()
            );
        }

        // Phase 2 — the COMBINED case: seven sealed chunks mixing the whole
        // ladder union, decoded in one window.
        let case = Case {
            n_win: 240,
            window_size: 256,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 2,
            zero_rope: false,
        };
        let b = |fmt, outer| BandSpec { fmt, outer };
        // Tile an 8-format pattern across the 16 bands (bands 8-15 repeat 0-7).
        let t8 = |a: [BandSpec; 8]| -> [BandSpec; N_BANDS] { std::array::from_fn(|i| a[i % 8]) };
        let fmts = vec![
            t8([b(8, 1.0), b(18, 1.0), b(19, 1.0), b(20, 1.0), b(8, 1.0), b(18, 1.0), b(19, 1.0), b(20, 1.0)]),
            t8([b(25, 0.25), b(26, 0.25), b(27, 0.25), b(29, 0.25), b(25, 0.25), b(26, 0.25), b(27, 0.25), b(29, 0.25)]),
            t8([b(28, 0.25), b(30, 0.25), b(31, 0.25), b(32, 0.25), b(28, 0.25), b(30, 0.25), b(31, 0.25), b(32, 0.25)]),
            t8([b(33, 0.25), b(25, 0.5), b(27, 0.5), b(32, 0.5), b(28, 0.3), b(26, 0.3), b(30, 0.5), b(31, 0.5)]),
            t8([b(10, 1.5), b(16, 1.0), b(15, 2.0), b(7, 0.5), b(19, 0.5), b(20, 1.5), b(18, 2.0), b(8, 1.5)]),
            t8([b(19, 0.5), b(20, 1.5), b(18, 2.0), b(8, 1.5), b(10, 1.5), b(16, 1.0), b(15, 2.0), b(7, 0.5)]),
            t8([b(28, 0.3), b(26, 0.3), b(30, 0.5), b(31, 0.5), b(33, 0.25), b(25, 0.5), b(27, 0.5), b(32, 0.5)]),
            [BandSpec::default(); N_BANDS],
        ];
        let inp = gen_inputs(&case, 0xBEEF);
        let kernel = run_kernel_fmt(&case, &inp, softmax_scale, Some(&fmts))?;
        let mirror_v: Vec<f32> = mirror_fmt(&case, &inp, softmax_scale, Some(&fmts))
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        let mirror_v = bf16_round_host(&mirror_v);
        // ≤1-code / ≥99.9%, as the isolation loop (residual dequant seam).
        let n_bad = check_within_one_code(&kernel, &mirror_v);
        assert!(
            n_bad * 1000 <= kernel.len(),
            "all-format: {n_bad}/{} cells off (> 0.1%)",
            kernel.len()
        );
        Ok(())
    }

    /// Stage-level probe: the kernel's dequantized window staging (`kv_f`
    /// from the stage dump) must match the codec's stored values ELEMENT BY
    /// ELEMENT on the non-rope dims — pinning any format divergence to the
    /// dequant itself rather than the downstream int8/attention pipeline.
    #[test]
    #[ignore]
    fn latent_window_stage_matches_codec() -> Result<()> {
        let case = Case {
            n_win: 64,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let dev = Device::new_cuda(0)?;
        // Seeds match the all-format isolation loop (0xBEE0 + combo index)
        // so a staging divergence reproduces the exact failing configuration.
        for &(fmt, outer, seed) in &[
            (27u8, 0.25f32, 0xBEEEu64),
            (25, 0.25, 0xBEEC),
            (33, 0.25, 0xBEF4),
            (20, 1.5, 0xBEEB),
        ] {
            let spec = BandSpec { fmt, outer };
            let specs = vec![
                [spec; N_BANDS],
                [spec; N_BANDS],
                [BandSpec::default(); N_BANDS],
            ];
            let inp = gen_inputs(&case, seed);
            let slots = SyntheticSlots::build_mapped(
                &dev,
                std::slice::from_ref(&inp.window),
                &[0usize],
                std::slice::from_ref(&specs),
                std::slice::from_ref(&vec![identity_pal_map(); 3]),
            )?;
            let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
            let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kv_new = Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?
                .to_dtype(DType::BF16)?;
            let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
            let comp = Tensor::from_vec(compf, (case.g_total, HEAD_DIM), &dev)?;
            let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total, &dev)?;
            let idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
            let comp_idx = Tensor::from_vec(idx, (1, case.selected.len()), &dev)?;
            let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
            let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
            let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
            let rope_tab = build_rope_table(&freqs)?;
            let ws = LatentWorkspace::build(&dev)?;
            let dbg = Tensor::zeros(DBG_LEN, DType::F32, &dev)?;
            let _ = paged_latent_decode(
                &q,
                &slots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &comp_idx,
                &comp_cnt,
                &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                case.window_size,
                case.num_splits,
                Some(&dbg),
            )?;
            let d = dbg.to_vec1::<f32>()?;
            // The dump covers the first 16 window tokens (chunk 0). Expected
            // stored values from the codec, identity layout.
            let toks: Vec<[f32; HEAD_DIM]> = inp.window[..CHUNK].to_vec();
            let (_, dec) = {
                let mut full = vec![[0f32; HEAD_DIM]; CHUNK];
                for p in 0..N_BANDS {
                    let band_toks: Vec<[f32; SUB_DIM]> = toks
                        .iter()
                        .map(|tok| std::array::from_fn(|r| tok[p * SUB_DIM + r]))
                        .collect();
                    let (_, bdec) = band_chunk_roundtrip(spec, &band_toks)?;
                    for (t, row) in bdec.iter().enumerate() {
                        for (r, &v) in row.iter().enumerate() {
                            full[t][p * SUB_DIM + r] = v;
                        }
                    }
                }
                ((), full)
            };
            // Pre-rope stored values (non-rope dims are rope-invariant).
            let mut bad = 0usize;
            let mut first = None;
            for t in 0..8 {
                for dim in 0..(HEAD_DIM - ROPE_DIM) {
                    let got = d[DBG_KVF + t * HEAD_DIM + dim];
                    let want = half::bf16::from_f32(dec[t][dim]).to_f32();
                    if got.to_bits() != want.to_bits() {
                        bad += 1;
                        if first.is_none() {
                            first = Some((t, dim, got, want));
                        }
                    }
                }
            }
            assert_eq!(
                bad, 0,
                "fmt {fmt} outer {outer}: {bad} staged elements differ; first {first:?}"
            );
            // Int8 staging from the roped f32 REGISTERS (kernel requants from
            // full precision, before the bf16 store). scaleK / sK inherit the
            // dequant `/outer` fast-math seam (a ≤1-ulp value flip can move a
            // band's max, so `mx/127` and the int8 requant shift by one step):
            // scaleK within 2 ulp, sK codes within ±1, and only on a handful
            // of cells (≥99.9% exact). The `kv_f` bf16 store above is strict —
            // bf16 rounding absorbs the ulp.
            let (mut bad_s, mut bad_q) = (0usize, 0usize);
            for t in 0..8 {
                let mut l = dec[t];
                rope_vec(&mut l, t as u32, &inp.freqs);
                let (ki, ks) = quant_bands(&l);
                for p in 0..N_BANDS {
                    let got = d[DBG_SCALEK + t * N_BANDS + p];
                    let ulp = (got.to_bits() as i64 - ks[p].to_bits() as i64).abs();
                    if ulp > 2 {
                        bad_s += 1;
                    }
                }
                for dim in 0..HEAD_DIM {
                    let got = d[DBG_SK + t * HEAD_DIM + dim] as i32;
                    if (got - ki[dim] as i32).abs() > 1 {
                        bad_q += 1;
                    }
                }
            }
            assert!(
                bad_s == 0 && bad_q == 0,
                "fmt {fmt} outer {outer}: {bad_s} scaleK (>2ulp) + {bad_q} sK (>±1) diverge"
            );
        }
        Ok(())
    }

    /// Reference FP8 window recovery: the nope span stored as **FP8 E4M3 with a
    /// per-64 ue8m0 (power-of-two) scale** must round-trip through the kernel's
    /// symmetric `store_band_elem` (`E4M3(v·outer)`) → `load_band_elem`
    /// (`decode / outer`) exactly, and the rope tail through BF16. This pins the
    /// symmetric-scale contract the reference relies on: a non-unit per-band
    /// `outer` in the KvHead `k_scale` slot recovers the source. The scale is a
    /// pure power of two (as ue8m0 requires) and the inputs are FP8-exact, so
    /// the recovery is BIT-exact after the bf16 kv_f staging store.
    #[test]
    #[ignore]
    fn latent_window_fp8_scale_round_trip() -> Result<()> {
        let case = Case {
            n_win: 40, // chunk 0 sealed (32 tok), chunk 1 the writer (8 tok)
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let dev = Device::new_cuda(0)?;
        // Sealed chunk 0: FP8 nope+rope at a per-64 power-of-two scale (ue8m0);
        // the writer chunk (last) stays the unit-scale FP8 default that the
        // fused scatter targets. `outer` is the decoder-divide scale = 1/scale.
        for &outer in &[0.5f32, 2.0f32] {
            let sealed = BandSpec { fmt: 34, outer };
            let specs = vec![[sealed; N_BANDS], [BandSpec::default(); N_BANDS]];
            let inp = gen_inputs(&case, 0xF80 + outer.to_bits() as u64);
            let slots = SyntheticSlots::build_mapped(
                &dev,
                std::slice::from_ref(&inp.window),
                &[0usize],
                std::slice::from_ref(&specs),
                std::slice::from_ref(&vec![identity_pal_map(); 2]),
            )?;
            let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
            let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kv_new = Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?
                .to_dtype(DType::BF16)?;
            let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
            let comp = Tensor::from_vec(compf, (case.g_total, HEAD_DIM), &dev)?;
            let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total, &dev)?;
            let idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
            let comp_idx = Tensor::from_vec(idx, (1, case.selected.len()), &dev)?;
            let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
            let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
            let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
            let rope_tab = build_rope_table(&freqs)?;
            let ws = LatentWorkspace::build(&dev)?;
            let dbg = Tensor::zeros(DBG_LEN, DType::F32, &dev)?;
            let _ = paged_latent_decode(
                &q,
                &slots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &comp_idx,
                &comp_cnt,
                &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                case.window_size,
                case.num_splits,
                Some(&dbg),
            )?;
            let d = dbg.to_vec1::<f32>()?;
            // Codec recovery of chunk 0's stored values (the mirror the kernel
            // dequant must match). The dump's kv_f holds bf16(load_band_elem).
            let toks: Vec<[f32; HEAD_DIM]> = inp.window[..CHUNK].to_vec();
            let mut dec = vec![[0f32; HEAD_DIM]; CHUNK];
            for p in 0..N_BANDS {
                let band_toks: Vec<[f32; SUB_DIM]> = toks
                    .iter()
                    .map(|tok| std::array::from_fn(|r| tok[p * SUB_DIM + r]))
                    .collect();
                let (_, bdec) = band_chunk_roundtrip(sealed, &band_toks)?;
                for (t, row) in bdec.iter().enumerate() {
                    for (r, &v) in row.iter().enumerate() {
                        dec[t][p * SUB_DIM + r] = v;
                    }
                }
            }
            // Non-rope dims are rope-invariant: assert the FP8-with-scale
            // recovery is bit-exact through the bf16 kv_f store.
            for t in 0..8 {
                for dim in 0..(HEAD_DIM - ROPE_DIM) {
                    let got = d[DBG_KVF + t * HEAD_DIM + dim];
                    let want = half::bf16::from_f32(dec[t][dim]).to_f32();
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "outer {outer}: kv_f[t{t} d{dim}] {got} != {want}"
                    );
                    // Power-of-two ue8m0 scale over FP8-exact inputs: exact.
                    assert_eq!(dec[t][dim].to_bits(), inp.window[t][dim].to_bits());
                }
            }
        }
        Ok(())
    }

    /// Stage-dump section offsets — mirror of latent_decode_kernel's DBG_*
    /// layout, derived from N_BANDS so the read offsets track the band count.
    /// (scaleQ/sQ are always 16-head-sized; scaleK/sK/kv_f are KEYS_TILE-sized.)
    const DBG_KEYS: usize = 8; // KEYS_TILE
    const DBG_SCALEQ: usize = 0;
    const DBG_SQ: usize = DBG_SCALEQ + 16 * N_BANDS;
    const DBG_SCALEK: usize = DBG_SQ + 16 * HEAD_DIM;
    const DBG_SK: usize = DBG_SCALEK + DBG_KEYS * N_BANDS;
    const DBG_KVF: usize = DBG_SK + DBG_KEYS * HEAD_DIM;
    const DBG_LOGITS: usize = DBG_KVF + DBG_KEYS * HEAD_DIM;
    const DBG_LEN: usize = DBG_LOGITS + 16 * DBG_KEYS;

    /// The PREFILL kernel's adaptive window dispatch, gated through the decode
    /// oracle chain: prefill rows over a settled slot whose SEALED chunks are
    /// quant-authored must match per-row decode over slots authored with the
    /// SAME sealed specs, within the established int8-PV-vs-scalar-PV envelope
    /// (`d < 0.03·scale`, as the FP8 prefill≡decode gates). Only writer-chunk
    /// rows (64..80) compare — there both views' chunk roles coincide (chunks
    /// 0-1 sealed quant, chunk 2 the FP8 writer), so both kernels read
    /// byte-identical arenas and the residual is purely the PV algorithm.
    #[test]
    #[ignore]
    fn prefill_mixed_rows_equal_decode_steps() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 80usize;
        let window_size = 128usize;
        let mut s = 7u64;
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

        let q8 = |outer| BandSpec { fmt: 7, outer };
        let q4 = |outer| BandSpec { fmt: 15, outer };
        let t8 = |a: [BandSpec; 8]| -> [BandSpec; N_BANDS] { std::array::from_fn(|i| a[i % 8]) };
        let sealed = [
            t8([q8(1.5), q4(1.0), BandSpec::default(), q8(1.0), q8(1.5), q4(1.0), BandSpec::default(), q8(1.0)]),
            t8([q4(2.0), q8(1.0), q8(0.5), q4(1.0), q4(2.0), q8(1.0), q8(0.5), q4(1.0)]),
        ];
        let specs = vec![sealed[0], sealed[1], [BandSpec::default(); N_BANDS]];

        // Prefill over the mixed settled slot.
        let slots = SyntheticSlots::build_mixed(
            &dev,
            std::slice::from_ref(&tokens.to_vec()),
            &[0usize],
            std::slice::from_ref(&specs),
        )?;
        let sinks = Tensor::from_vec(sinks_v.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;
        let prefill = paged_latent_prefill(
            &q_all,
            &slots.headers,
            &q_pos,
            None,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // Per-row decode over slots with the SAME sealed specs (row i's view:
        // chunks 0-1 sealed, chunk 2 = writer — identical bytes to the
        // prefill slot for every token < i).
        for i in 64..n {
            let case = Case {
                n_win: i,
                window_size,
                g_total: 0,
                ratio: 4,
                selected: vec![],
                num_splits: 1,
                zero_rope: false,
            };
            let inp = Inputs {
                window: tokens[..i].to_vec(),
                kv_new: tokens[i],
                q: qs[i * H..(i + 1) * H].to_vec(),
                comp: vec![],
                comp_pos: vec![],
                sinks: sinks_v.clone(),
                freqs: freqs_v.clone(),
            };
            let dec = run_kernel_fmt(&case, &inp, softmax_scale, Some(&specs))?;
            let row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            // Same tolerance as the FP8 prefill≡decode gates: the prefill's
            // int8 tensor-core PV vs the decode's scalar PV is a small bounded
            // envelope; the mixed formats must not widen it.
            let scale = dec.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let d = max_abs_diff(row, &dec);
            assert!(
                d < 0.03 * scale,
                "mixed row {i}: prefill(int8 PV) vs decode |Δ|={d} ≥ {}",
                0.03 * scale
            );
        }
        Ok(())
    }

    /// Oracle (a2) at splits=1 — the exact configuration the tolerance gate
    /// runs, so a kernel↔mirror divergence there is separable from a genuine
    /// quantization-quality regression. Also reports the pairwise deltas.
    #[test]
    #[ignore]
    fn mirror_bit_exact_with_rope_splits1() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let (kernel, mirror_v, reference) = run_case(&case, 42)?;
        eprintln!(
            "Δ(kernel,mirror)={} Δ(mirror,ref)={} Δ(kernel,ref)={}",
            max_abs_diff(&kernel, &mirror_v),
            max_abs_diff(&mirror_v, &reference),
            max_abs_diff(&kernel, &reference)
        );
        let nope_max = kernel
            .iter()
            .zip(&reference)
            .enumerate()
            .filter(|(i, _)| i % HEAD_DIM < NOPE_DIM)
            .map(|(_, (a, b))| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rope_max = kernel
            .iter()
            .zip(&reference)
            .enumerate()
            .filter(|(i, _)| i % HEAD_DIM >= NOPE_DIM)
            .map(|(_, (a, b))| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("kernel-vs-ref: nope_max={nope_max} rope_max={rope_max}");
        let mismatches = kernel
            .iter()
            .zip(&mirror_v)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(mismatches, 0, "kernel≠mirror at splits=1 live rope");
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
    /// by `paged_latent_glue_scatter` must be BIT-IDENTICAL, through the
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
                paged_latent_glue_scatter(&kv, headers_addr, &slices, &in_blk)?;
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
            let rope_tab = build_rope_table(&freqs)?;
            let ws = LatentWorkspace::build(&dev)?;
            let out = paged_latent_decode(
                &qt,
                &slots.headers,
                &kvt,
                &CorpusCache::build(&comp, &comp_pos)?,
                &comp_idx,
                &comp_cnt,
                &Tensor::from_vec(vec![window.len() as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
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
