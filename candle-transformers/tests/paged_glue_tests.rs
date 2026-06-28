//! Correctness gate for the paged-glue attention kernel (decode-derivative).
//!
//! The glue kernel re-prefills the reprojection "glue" tokens: G query tokens
//! attend a sealed prefix (read from the paged KV arena) + earlier glue, and
//! write their own K/V. For the contiguous-tail glue the scheduler emits, that
//! is exactly a normal incremental prefill — so the gate is an **A/B against the
//! proven `paged_prefill_batched` kernel**: build two slots with the SAME sealed
//! prefix, append the SAME glue tokens — once via normal prefill (reference),
//! once via the glue kernel — and require the attention outputs to match.
//!
//! HD128 (SUB_HEAD_DIM=32, the kernel's m16n8k32 palette width), GQA. F16 arena
//! first (identity dequant — isolates the streaming / flash-state / writeback /
//! causal structure); the quantized-arena variant layers on after this passes.

#![cfg(feature = "cuda")]

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::quantized::pinned_staging::{PinnedBuf, PinnedStager};
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    quantize_sealed_in_place, ChunkedKvBacking, CompressionPolicy, KvCache, KvFormat, CHUNK_SIZE,
};
use candle_transformers::models::prefill_utils::compute_rope_cs;
use candle_transformers::models::slot_state::{
    pack_position_entry, tensor_u8_device_ptr, SlotStateHost, TokenSliceHost,
};

const N_KV_HEAD: usize = 2;
const N_HEAD: usize = 8; // hpg = 4
const HEAD_DIM: usize = 128;
const MAX_BLOCKS: usize = 160; // up to ~5120-token prefixes for the depth sweep
const DIFF_TOLERANCE: f32 = 1e-2;

/// Deterministic synthetic Q/K/V, FLAT-packed `[n_tokens, n_*head, head_dim]`.
fn make_qkv(n_tokens: usize, seed: u64, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
    fn pseudo(i: usize, j: usize, k: usize, seed: u64) -> f32 {
        let mut x = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add((k as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
            .wrapping_add(seed);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    let mut q = Vec::with_capacity(n_tokens * N_HEAD * HEAD_DIM);
    let mut k = Vec::with_capacity(n_tokens * N_KV_HEAD * HEAD_DIM);
    let mut v = Vec::with_capacity(n_tokens * N_KV_HEAD * HEAD_DIM);
    for t in 0..n_tokens {
        for h in 0..N_HEAD {
            for d in 0..HEAD_DIM {
                q.push(pseudo(t, h, d, seed ^ 0x11));
            }
        }
        for h in 0..N_KV_HEAD {
            for d in 0..HEAD_DIM {
                k.push(pseudo(t, h, d, seed ^ 0x22));
                v.push(pseudo(t, h, d, seed ^ 0x33));
            }
        }
    }
    let q = Tensor::from_vec(q, (n_tokens, N_HEAD, HEAD_DIM), device)?;
    let k = Tensor::from_vec(k, (n_tokens, N_KV_HEAD, HEAD_DIM), device)?;
    let v = Tensor::from_vec(v, (n_tokens, N_KV_HEAD, HEAD_DIM), device)?;
    Ok((q, k, v))
}

/// Pure-tensor f32 reference: plain causal GQA attention of the glue queries
/// over `[sealed | glue]` K/V, identity RoPE (inv_freq=0). Glue token `t` (at
/// absolute position `sealed+t`) attends columns `0..=sealed+t`. This is the
/// ground truth the glue kernel must reproduce — independent of any kernel.
fn reference_attn(
    q_glue: &Tensor,
    k_seal: &Tensor,
    v_seal: &Tensor,
    k_glue: &Tensor,
    v_glue: &Tensor,
    sealed: usize,
    glue: usize,
    device: &Device,
) -> Result<Tensor> {
    let hpg = N_HEAD / N_KV_HEAD;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let qd = q_glue
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?; // [glue,N_HEAD,HD]
    let ks = k_seal
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?; // [sealed,N_KV,HD]
    let vs = v_seal
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let kg = k_glue
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?; // [glue,N_KV,HD]
    let vg = v_glue
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let kv_at = |col: usize, kvh: usize, d: usize, src: &[f32], srcg: &[f32]| -> f32 {
        if col < sealed {
            src[(col * N_KV_HEAD + kvh) * HEAD_DIM + d]
        } else {
            srcg[((col - sealed) * N_KV_HEAD + kvh) * HEAD_DIM + d]
        }
    };
    let mut out = vec![0f32; glue * N_HEAD * HEAD_DIM];
    for t in 0..glue {
        let pos = sealed + t;
        for h in 0..N_HEAD {
            let kvh = h / hpg;
            let mut scores = vec![0f32; pos + 1];
            let mut m = f32::NEG_INFINITY;
            for (col, sc) in scores.iter_mut().enumerate() {
                let mut dot = 0f32;
                for d in 0..HEAD_DIM {
                    dot += qd[(t * N_HEAD + h) * HEAD_DIM + d] * kv_at(col, kvh, d, &ks, &kg);
                }
                *sc = dot * scale;
                m = m.max(*sc);
            }
            let mut denom = 0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0f32;
                for (col, &w) in scores.iter().enumerate() {
                    acc += w * kv_at(col, kvh, d, &vs, &vg);
                }
                out[(t * N_HEAD + h) * HEAD_DIM + d] = acc / denom;
            }
        }
    }
    Tensor::from_vec(out, (glue, N_HEAD, HEAD_DIM), device)
}

/// Forward-window golden: like [`reference_attn`] but each glue row ADDITIONALLY
/// attends the first `fwd_b` columns of section B unconditionally (the forward
/// look), on top of its causal `[A | glue]` columns. Identity RoPE. This is the
/// ground truth the kernel must reproduce when B is resident in the slot's
/// position_map at `[kv_len, kv_len+fwd_b)`.
#[allow(clippy::too_many_arguments)]
fn reference_attn_fwd(
    q_glue: &Tensor,
    k_seal: &Tensor,
    v_seal: &Tensor,
    k_glue: &Tensor,
    v_glue: &Tensor,
    k_b: &Tensor,
    v_b: &Tensor,
    sealed: usize,
    glue: usize,
    fwd_b: usize,
    device: &Device,
) -> Result<Tensor> {
    let hpg = N_HEAD / N_KV_HEAD;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let flat = |t: &Tensor| -> Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
    let qd = flat(q_glue)?; // [glue,N_HEAD,HD]
    let ks = flat(k_seal)?; // [sealed,N_KV,HD]
    let vs = flat(v_seal)?;
    let kg = flat(k_glue)?; // [glue,N_KV,HD]
    let vg = flat(v_glue)?;
    let kb = flat(k_b)?; // [b_len,N_KV,HD]
    let vb = flat(v_b)?;
    // Causal column `col` of [A | glue]: A for col<sealed else glue.
    let cg = |col: usize, kvh: usize, d: usize, src: &[f32], srcg: &[f32]| -> f32 {
        if col < sealed {
            src[(col * N_KV_HEAD + kvh) * HEAD_DIM + d]
        } else {
            srcg[((col - sealed) * N_KV_HEAD + kvh) * HEAD_DIM + d]
        }
    };
    let bcol = |col: usize, kvh: usize, d: usize, src: &[f32]| -> f32 {
        src[(col * N_KV_HEAD + kvh) * HEAD_DIM + d]
    };
    let mut out = vec![0f32; glue * N_HEAD * HEAD_DIM];
    for t in 0..glue {
        let pos = sealed + t; // causal columns are 0..=pos
        for h in 0..N_HEAD {
            let kvh = h / hpg;
            // Scores: pos+1 causal columns, then fwd_b forward B columns.
            let mut scores = vec![0f32; pos + 1 + fwd_b];
            let mut m = f32::NEG_INFINITY;
            for (col, sc) in scores.iter_mut().enumerate().take(pos + 1) {
                let mut dot = 0f32;
                for d in 0..HEAD_DIM {
                    dot += qd[(t * N_HEAD + h) * HEAD_DIM + d] * cg(col, kvh, d, &ks, &kg);
                }
                *sc = dot * scale;
                m = m.max(*sc);
            }
            for bc in 0..fwd_b {
                let mut dot = 0f32;
                for d in 0..HEAD_DIM {
                    dot += qd[(t * N_HEAD + h) * HEAD_DIM + d] * bcol(bc, kvh, d, &kb);
                }
                let s = dot * scale;
                scores[pos + 1 + bc] = s;
                m = m.max(s);
            }
            let mut denom = 0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0f32;
                for (col, &w) in scores.iter().enumerate().take(pos + 1) {
                    acc += w * cg(col, kvh, d, &vs, &vg);
                }
                for bc in 0..fwd_b {
                    acc += scores[pos + 1 + bc] * bcol(bc, kvh, d, &vb);
                }
                out[(t * N_HEAD + h) * HEAD_DIM + d] = acc / denom;
            }
        }
    }
    Tensor::from_vec(out, (glue, N_HEAD, HEAD_DIM), device)
}

fn fresh_backing(device: &Device) -> Result<ChunkedKvBacking> {
    ChunkedKvBacking::new(
        1,
        N_KV_HEAD,
        HEAD_DIM,
        DType::F16,
        device,
        MAX_BLOCKS * CHUNK_SIZE,
    )
}

fn bind_cache(backing: &ChunkedKvBacking, slot: usize) -> Result<KvCache> {
    let mut cache = KvCache::new(2, MAX_BLOCKS * CHUNK_SIZE);
    cache.force_dtype(DType::F16);
    cache.set_chunked_backing(backing, slot, None)?;
    Ok(cache)
}

/// Build the sealed prefix directly in the F16 arena (no prefill kernel): the
/// arena stores raw un-rotated K/V (the kernel re-RoPEs at read; identity RoPE
/// here), exactly what `reference_attn` reads.
fn build_sealed_arena(
    backing: &ChunkedKvBacking,
    cache: &mut KvCache,
    k_seal: &Tensor,
    v_seal: &Tensor,
    sealed: usize,
) -> Result<()> {
    // make_qkv yields [sealed, n_kv_head, hd]; write_contiguous wants
    // [1, n_kv_head, sealed, hd].
    let kr = k_seal
        .to_dtype(DType::F16)?
        .unsqueeze(0)?
        .transpose(1, 2)?
        .contiguous()?;
    let vr = v_seal
        .to_dtype(DType::F16)?
        .unsqueeze(0)?
        .transpose(1, 2)?
        .contiguous()?;
    backing.ensure_for_offset(0, 0, sealed)?;
    backing.write_contiguous(0, 0, &kr, &vr)?;
    backing.set_len(0, sealed);
    cache.set_current_seq_len(sealed)?;
    Ok(())
}

fn fresh_quant_backing(device: &Device, policy: &CompressionPolicy) -> Result<ChunkedKvBacking> {
    ChunkedKvBacking::new_with_format_adaptive(
        1,
        N_KV_HEAD,
        HEAD_DIM,
        KvFormat::Float(DType::F16),
        KvFormat::Float(DType::F16),
        device,
        MAX_BLOCKS * CHUNK_SIZE,
        Some(policy.clone()),
    )
}

/// Build the sealed prefix as genuine palette-quantized chunks (the reproject's
/// real storage): write F16, seal the turn, quantize-in-place per the policy,
/// re-inject. The glue kernel then reads the prefix through the C0-C9 palette
/// dequant path rather than the F16 identity path.
fn build_sealed_arena_quant(
    backing: &ChunkedKvBacking,
    cache: &mut KvCache,
    k_seal: &Tensor,
    v_seal: &Tensor,
    sealed: usize,
    policy: &CompressionPolicy,
    device: &Device,
) -> Result<()> {
    let kr = k_seal
        .to_dtype(DType::F16)?
        .unsqueeze(0)?
        .transpose(1, 2)?
        .contiguous()?;
    let vr = v_seal
        .to_dtype(DType::F16)?
        .unsqueeze(0)?
        .transpose(1, 2)?
        .contiguous()?;
    backing.ensure_for_offset(0, 0, sealed)?;
    backing.write_contiguous(0, 0, &kr, &vr)?;
    backing.set_len(0, sealed);
    let real_chunks = sealed.div_ceil(CHUNK_SIZE).max(1);
    backing.truncate_sequence_to_blocks(0, real_chunks)?;
    let r16 = backing.record_turn(0)?;
    let copy_stream = match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => candle::bail!("cuda only"),
    };
    let mut scratch: Option<PinnedBuf> = None;
    let warm =
        quantize_sealed_in_place(backing, &[&r16], policy, device, &copy_stream, &mut scratch)?;
    backing.truncate_sequence_to_blocks(0, 0)?;
    backing.inject_sealed_at_tail(0, &warm[0])?;
    cache.set_current_seq_len(sealed)?;
    Ok(())
}

/// Device pointer of a CUDA tensor's storage (kept alive via the returned guard).
fn dev_ptr_u32(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let cs = match &*storage {
        candle::Storage::Cuda(c) => c,
        _ => candle::bail!("expected cuda tensor"),
    };
    let stream = cs.device().cuda_stream();
    let s = cs.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);
    let (p, _g) = s.device_ptr(&stream);
    Ok(p)
}

fn dev_ptr_f16(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let cs = match &*storage {
        candle::Storage::Cuda(c) => c,
        _ => candle::bail!("expected cuda tensor"),
    };
    let stream = cs.device().cuda_stream();
    let s = cs
        .as_cuda_slice::<half::f16>()?
        .slice(layout.start_offset()..);
    let (p, _g) = s.device_ptr(&stream);
    Ok(p)
}

/// A resident "section B" (the forward window): B's K/V sealed in its own F16
/// backing, with `TokenSliceHost` slices resolved against that backing. The
/// backing/cache are held alive by the struct so the device arena pointers baked
/// into the slices stay valid through the kernel launch.
struct BSection {
    _backing: ChunkedKvBacking,
    _cache: KvCache,
    /// B's sealed-chunk slices (arena pointers resolved against `_backing`).
    slices: Vec<TokenSliceHost>,
    /// Assembled (downstream-of-glue) sequence position of each B column
    /// (`len == b_len`). Identity RoPE in the tests makes the value numerically
    /// inert, but it mirrors the production `col_actual_pos` layout.
    positions: Vec<u32>,
    b_len: usize,
}

/// Build section B as sealed F16 chunks in a fresh backing — the same way the
/// prefix is built — and resolve its slices, ready to be appended into another
/// slot's `position_map` at `[kv_len, kv_len+b_len)`.
fn build_b_section(
    k_b: &Tensor,
    v_b: &Tensor,
    b_len: usize,
    assembled_base: usize,
    device: &Device,
) -> Result<BSection> {
    let backing = fresh_backing(device)?;
    let mut cache = bind_cache(&backing, 0)?;
    build_sealed_arena(&backing, &mut cache, k_b, v_b, b_len)?;
    let arena_info = backing.resolve_arena_info()?;
    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let mut cum: u32 = 0;
    let slices: Vec<TokenSliceHost> = chunks
        .iter()
        .map(|c| {
            let rope_base = cum;
            cum += c.token_count as u32;
            TokenSliceHost::from_sealed_chunk(c, rope_base, N_KV_HEAD, HEAD_DIM, &arena_info)
        })
        .collect();
    let positions = (assembled_base as u32..(assembled_base + b_len) as u32).collect();
    Ok(BSection {
        _backing: backing,
        _cache: cache,
        slices,
        positions,
        b_len,
    })
}

/// Append `glue` tokens over the cache's prefix via the glue kernel.
///
/// `fwd_window` is the forward B-head window cap; `b_section`, when present, is
/// staged into this slot's `position_map` at `[kv_len, kv_len+b_len)` so the glue
/// rows attend `min(fwd_window, b_len)` B columns. `(0, None)` is the
/// backward-only path (bit-identical to the pre-window kernel).
#[allow(clippy::too_many_arguments)]
fn run_glue(
    backing: &ChunkedKvBacking,
    cache: &mut KvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    glue: usize,
    rope_cs: &Tensor,
    stager: &PinnedStager,
    device: &Device,
    fwd_window: usize,
    b_section: Option<&BSection>,
) -> Result<(Tensor, f64)> {
    let prefix_len = cache.current_seq_len();
    let kv_len = prefix_len + glue;
    let arena_info = backing.resolve_arena_info()?;
    backing.ensure_for_batch_entries(&[(0, prefix_len)], glue)?;

    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);
    let mut slot = SlotStateHost::from_sealed_chunks(
        &chunks,
        N_KV_HEAD,
        HEAD_DIM,
        &arena_info,
        writer_start,
        true,
    );
    slot.extend_for_write_region(glue, CHUNK_SIZE);

    // Stage section B (the forward window) into THIS slot's position_map at packed
    // indices [kv_len, kv_len+b_len): append B's slices and map B's columns onto
    // them. B is read-only — the glue scatter only writes `t < glue` — exactly the
    // layout the kernel's forward window assumes.
    let mut b_positions: Vec<u32> = Vec::new();
    if let Some(bs) = b_section {
        let b_slice_base = slot.slices.len();
        for s in &bs.slices {
            slot.slices.push(s.clone());
        }
        for (li, s) in bs.slices.iter().enumerate() {
            let sidx = (b_slice_base + li) as u32;
            let off = s.offset as u32;
            for i in 0..s.len as u32 {
                slot.position_map.push(pack_position_entry(sidx, off + i));
            }
        }
        b_positions = bs.positions.clone();
    }

    // Two-section layout, mirroring build_slot_headers: resident slices
    // (meta=Some, quantized prefix) point kvheads_ptr at their device meta-pool
    // record; scratch slices (meta=None) serialize into a records buffer. This
    // exercises the resident-record device_addr path on GPU.
    enum KvSrc {
        Resident(u64),
        Scratch(usize),
    }
    let mut records_buf: Vec<u8> = Vec::new();
    let mut srcs: Vec<KvSrc> = Vec::with_capacity(slot.slices.len());
    for s in &slot.slices {
        match &s.meta {
            Some(meta) => {
                let addr = cache.k_cache().chunked_meta_device_addr(meta);
                assert_ne!(addr, 0, "resident slice has no device record");
                srcs.push(KvSrc::Resident(addr));
            }
            None => {
                let off = records_buf.len();
                s.serialize_record(&mut records_buf);
                srcs.push(KvSrc::Scratch(off));
            }
        }
    }
    if records_buf.is_empty() {
        records_buf.push(0u8);
    }
    let records_tensor = Tensor::from_slice(&records_buf, records_buf.len(), device)?;
    let records_base = tensor_u8_device_ptr(&records_tensor)?;

    let mut slice_buf = Vec::with_capacity(slot.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
    for (i, s) in slot.slices.iter().enumerate() {
        let kvheads_ptr = match srcs[i] {
            KvSrc::Resident(addr) => addr,
            KvSrc::Scratch(off) => records_base + off as u64,
        };
        s.serialize_slice_header(&mut slice_buf, kvheads_ptr);
    }
    let slices_tensor = Tensor::from_slice(&slice_buf, slice_buf.len(), device)?;
    let slices_ptr = tensor_u8_device_ptr(&slices_tensor)?;

    // position_map → device (sealed-only reads + write region). entry = (slice<<16)|in_blk.
    let pm = slot.position_map.clone();
    let pm_tensor = Tensor::from_slice(&pm, pm.len().max(1), device)?;
    let pm_ptr = dev_ptr_u32(&pm_tensor)?;

    // Glue write targets: derive from position_map at the write positions.
    let mut wslice = Vec::with_capacity(glue);
    let mut winblk = Vec::with_capacity(glue);
    for t in 0..glue {
        let e = pm[prefix_len + t];
        wslice.push(e >> 16);
        winblk.push(e & 0xffff);
    }
    let glue_write_slice = Tensor::from_vec(wslice, glue, device)?;
    let glue_write_in_blk = Tensor::from_vec(winblk, glue, device)?;

    // Varlen meta (single slot) + identity col_actual_pos for [0, kv_len), then
    // B's assembled positions for the forward window at [kv_len, kv_len+b_len).
    let cu_seqlens_q = Tensor::from_vec(vec![0u32, glue as u32], 2, device)?;
    let q_lens = Tensor::from_vec(vec![glue as u32], 1, device)?;
    let kv_lens = Tensor::from_vec(vec![kv_len as u32], 1, device)?;
    let cu_kvlens = Tensor::from_vec(vec![0u32, kv_len as u32], 2, device)?;
    let mut cap: Vec<u32> = (0..kv_len as u32).collect();
    cap.extend_from_slice(&b_positions);
    let col_actual_pos = Tensor::from_vec(cap.clone(), cap.len().max(1), device)?;
    let b_avail_n = b_section.map(|bs| bs.b_len).unwrap_or(0);
    let b_avail = Tensor::from_vec(vec![b_avail_n as u32], 1, device)?;

    // 24-byte SlotHeader.
    let mut hdr = Vec::with_capacity(24);
    hdr.extend_from_slice(&(slot.slices.len() as u32).to_le_bytes());
    hdr.extend_from_slice(&slot.write_slice.to_le_bytes());
    hdr.extend_from_slice(&slices_ptr.to_le_bytes());
    hdr.extend_from_slice(&pm_ptr.to_le_bytes());
    let generation = stager.begin_generation();
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers_gpu = generation.submit(pinned)?;
    let headers_ptr = headers_gpu.dev_ptr();

    // F16 q/k/v + output.
    let qf = q.to_dtype(DType::F16)?.contiguous()?;
    let kf = k.to_dtype(DType::F16)?.contiguous()?;
    let vf = v.to_dtype(DType::F16)?.contiguous()?;
    let out = Tensor::zeros((glue, N_HEAD, HEAD_DIM), DType::F16, device)?;

    let softmax_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let stream = match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => candle::bail!("cuda only"),
    };
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;

    device.synchronize()?;
    let t_kernel = std::time::Instant::now();
    unsafe {
        candle_kernels::paged_glue::run_paged_glue_fp16(
            dev_ptr_f16(&qf)? as *const std::ffi::c_void,
            headers_ptr as *const u8,
            dev_ptr_f16(&out)? as *mut std::ffi::c_void,
            1,
            glue as i32,
            N_HEAD as i32,
            N_KV_HEAD as i32,
            HEAD_DIM as i32,
            softmax_scale,
            dev_ptr_f16(&kf)? as *const std::ffi::c_void,
            dev_ptr_f16(&vf)? as *const std::ffi::c_void,
            {
                let (s, l) = rope_cs.storage_and_layout();
                let c = match &*s {
                    candle::Storage::Cuda(c) => c,
                    _ => candle::bail!("cuda"),
                };
                let sl = c.as_cuda_slice::<f32>()?.slice(l.start_offset()..);
                let (p, _g) = sl.device_ptr(&stream);
                p as *const f32
            },
            0,
            dev_ptr_u32(&cu_seqlens_q)? as *const u32,
            dev_ptr_u32(&q_lens)? as *const u32,
            dev_ptr_u32(&kv_lens)? as *const u32,
            dev_ptr_u32(&col_actual_pos)? as *const u32,
            dev_ptr_u32(&cu_kvlens)? as *const u32,
            dev_ptr_u32(&glue_write_slice)? as *const u32,
            dev_ptr_u32(&glue_write_in_blk)? as *const u32,
            fwd_window as u32,
            dev_ptr_u32(&b_avail)? as *const u32,
            stream_ptr,
        );
    }
    device.synchronize()?;
    let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1e3;

    cache.set_current_seq_len(kv_len)?;
    drop(headers_gpu);
    drop(slices_tensor);
    drop(pm_tensor);
    Ok((out, kernel_ms))
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?;
    d.to_scalar::<f32>()
}

/// A/B gate: glue kernel == normal prefill over the same sealed prefix.
#[test]
fn paged_glue_matches_normal_prefill_f16() -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, &device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, &device)?;

    let cases: &[(usize, usize)] = &[(4, 1), (20, 8), (32, 5), (7, 12), (64, 16), (100, 20)];
    let mut failures = Vec::new();

    for &(sealed, glue) in cases {
        let (q_seal, k_seal, v_seal) = make_qkv(sealed, 0x5EA1 ^ sealed as u64, &device)?;
        let (q_glue, k_glue, v_glue) = make_qkv(glue, 0x61E ^ glue as u64, &device)?;

        let _ = &q_seal; // sealed Q is irrelevant; only its K/V populate the arena

        // Reference: pure-tensor f32 golden (independent of any kernel).
        let out_a = reference_attn(
            &q_glue, &k_seal, &v_seal, &k_glue, &v_glue, sealed, glue, &device,
        )?;

        // Test slot: sealed prefix in the F16 arena (direct injection), then the
        // glue kernel appends the glue tokens.
        let backing_b = fresh_backing(&device)?;
        let mut cache_b = bind_cache(&backing_b, 0)?;
        build_sealed_arena(&backing_b, &mut cache_b, &k_seal, &v_seal, sealed)?;
        let (out_b, _ms) = run_glue(
            &backing_b,
            &mut cache_b,
            &q_glue,
            &k_glue,
            &v_glue,
            glue,
            &rope_cs,
            &stager,
            &device,
            0,
            None,
        )?;

        let diff = max_abs_diff(&out_a, &out_b)?;
        eprintln!("paged-glue (sealed={sealed}, glue={glue}): max abs diff = {diff:.6e}");
        if !(diff < DIFF_TOLERANCE) {
            failures.push(format!("(sealed={sealed}, glue={glue}): diff={diff:.6e}"));
        }
    }

    if !failures.is_empty() {
        candle::bail!(
            "paged-glue diverged from normal prefill in {} case(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

/// Forward-window gate: with section B resident in the slot's position_map at
/// `[kv_len, kv_len+b_len)`, glue rows must attend the first `min(fwd_window,
/// b_len)` columns of B. Two properties are checked per case:
///   1. the forward output matches the forward golden (glue→B cross-attention),
///   2. it DIFFERS from the backward-only (`fwd_window==0`) output — proving the
///      window actually opened B rather than being a silent no-op.
/// `fwd_window` is also driven above and below `b_len` to exercise the per-slot
/// `min(fwd_window, b_avail)` clamp.
#[test]
fn paged_glue_forward_window_f16() -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, &device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, &device)?;

    // (sealed, glue, b_len, fwd_window): the last two cases drive fwd_window below
    // and above b_len to exercise the clamp (effective fwd_b = min(fwd_window, b)).
    let cases: &[(usize, usize, usize, usize)] =
        &[(32, 8, 16, 16), (64, 12, 24, 8), (100, 20, 16, 32)];
    let mut failures = Vec::new();

    for &(sealed, glue, b_len, fwd_window) in cases {
        let fwd_b = fwd_window.min(b_len);
        let (_q_seal, k_seal, v_seal) = make_qkv(sealed, 0x5EA1 ^ sealed as u64, &device)?;
        let (q_glue, k_glue, v_glue) = make_qkv(glue, 0x61E ^ glue as u64, &device)?;
        let (_q_b, k_b, v_b) = make_qkv(b_len, 0xB0B ^ b_len as u64, &device)?;

        // Backward-only kernel (fwd_window == 0): no B in the stream.
        let backing_bwd = fresh_backing(&device)?;
        let mut cache_bwd = bind_cache(&backing_bwd, 0)?;
        build_sealed_arena(&backing_bwd, &mut cache_bwd, &k_seal, &v_seal, sealed)?;
        let (out_bwd, _) = run_glue(
            &backing_bwd, &mut cache_bwd, &q_glue, &k_glue, &v_glue, glue, &rope_cs, &stager,
            &device, 0, None,
        )?;

        // Forward kernel: B resident, attend min(fwd_window, b_len) of it.
        let bsec = build_b_section(&k_b, &v_b, b_len, sealed + glue, &device)?;
        let backing_fwd = fresh_backing(&device)?;
        let mut cache_fwd = bind_cache(&backing_fwd, 0)?;
        build_sealed_arena(&backing_fwd, &mut cache_fwd, &k_seal, &v_seal, sealed)?;
        let (out_fwd, _) = run_glue(
            &backing_fwd, &mut cache_fwd, &q_glue, &k_glue, &v_glue, glue, &rope_cs, &stager,
            &device, fwd_window, Some(&bsec),
        )?;

        // Forward golden (glue→B open) — the kernel must reproduce it.
        let out_ref = reference_attn_fwd(
            &q_glue, &k_seal, &v_seal, &k_glue, &v_glue, &k_b, &v_b, sealed, glue, fwd_b, &device,
        )?;

        let diff_ref = max_abs_diff(&out_ref, &out_fwd)?;
        let diff_open = max_abs_diff(&out_bwd, &out_fwd)?;
        eprintln!(
            "paged-glue fwd (sealed={sealed}, glue={glue}, b_len={b_len}, fwd_window={fwd_window} \
             -> fwd_b={fwd_b}): diff_vs_ref={diff_ref:.6e}, diff_vs_backward={diff_open:.6e}"
        );
        if !(diff_ref < DIFF_TOLERANCE) {
            failures.push(format!(
                "(sealed={sealed}, glue={glue}, fwd_b={fwd_b}): forward output != golden, diff={diff_ref:.6e}"
            ));
        }
        // With fwd_b > 0 and B resident, opening the window MUST change the output;
        // a ~0 delta means B was not actually in the stream (the no-op case).
        if !(diff_open > 1e-3) {
            failures.push(format!(
                "(sealed={sealed}, glue={glue}, fwd_b={fwd_b}): forward window did not change output (diff_vs_backward={diff_open:.6e}) — B not in the stream"
            ));
        }
    }

    if !failures.is_empty() {
        candle::bail!(
            "paged-glue forward window failed in {} case(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

/// Isolated kernel-time sweep (not a correctness gate). Confirms WHERE the glue
/// kernel spends time: depth sweep (fixed glue) exposes per-column cost — a
/// sync-bound kernel scales ~linearly in kv_len with a large ms/col constant;
/// glue sweep (fixed depth) exposes the cross-tile dequant redundancy (cost
/// scaling with ceil(G/GLUE_G_TILE) rather than G). Run with:
///   cargo test -p candle-transformers --test paged_glue_tests --release \
///     --features cuda paged_glue_kernel_timing -- --ignored --nocapture
#[test]
#[ignore]
fn paged_glue_kernel_timing() -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, &device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, &device)?;
    let policy = CompressionPolicy::new(5); // mid compression (realistic reproject)
    let hpg = N_HEAD / N_KV_HEAD;

    let time_one = |sealed: usize, glue: usize| -> Result<f64> {
        let (_q, k_seal, v_seal) = make_qkv(sealed, 0xD0 ^ sealed as u64, &device)?;
        let (q_glue, k_glue, v_glue) = make_qkv(glue, 0x61E ^ glue as u64, &device)?;
        let backing = fresh_quant_backing(&device, &policy)?;
        let mut cache = bind_cache(&backing, 0)?;
        build_sealed_arena_quant(
            &backing, &mut cache, &k_seal, &v_seal, sealed, &policy, &device,
        )?;
        let (_out, ms) = run_glue(
            &backing, &mut cache, &q_glue, &k_glue, &v_glue, glue, &rope_cs, &stager, &device, 0,
            None,
        )?;
        Ok(ms)
    };

    // Warmup (JIT / first-launch overhead off the measured path).
    let _ = time_one(256, 8)?;

    eprintln!("--- glue kernel: depth sweep (glue=64, hpg={hpg}, n_kv_head={N_KV_HEAD}) ---");
    for &sealed in &[512usize, 1024, 2048, 4096] {
        let glue = 64;
        let ms = time_one(sealed, glue)?;
        let kv = sealed + glue;
        eprintln!(
            "  kv_len={kv:5} glue={glue:3}  ->  {ms:8.3} ms   ({:.5} ms/col)",
            ms / kv as f64
        );
    }
    eprintln!("--- glue kernel: glue sweep (depth=2048) ---");
    for &glue in &[8usize, 32, 64, 128] {
        let sealed = 2048;
        let ms = time_one(sealed, glue)?;
        let tiles = glue.div_ceil(8);
        eprintln!(
            "  glue={glue:4} kv_len={}  tiles={tiles:2}  ->  {ms:8.3} ms",
            sealed + glue
        );
    }
    Ok(())
}

/// Quant-arena gate: the glue kernel over a genuine palette-quantized prefix
/// (the reproject's real storage). Reference is the F32 golden over the ORIGINAL
/// K/V, so the diff carries the quantization error; a near-lossless policy keeps
/// it well under a loose structural gate (a real read bug craters to ~0.3, as
/// the smem-launch failure showed at 0.36).
#[test]
fn paged_glue_matches_golden_quant() -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, &device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, &device)?;
    let policy = CompressionPolicy::new(0); // near-lossless (C0)
    const QUANT_TOL: f32 = 6e-2;

    let cases: &[(usize, usize)] = &[(32, 5), (64, 16), (100, 20)];
    let mut failures = Vec::new();
    for &(sealed, glue) in cases {
        let (_q_seal, k_seal, v_seal) = make_qkv(sealed, 0x5EA1 ^ sealed as u64, &device)?;
        let (q_glue, k_glue, v_glue) = make_qkv(glue, 0x61E ^ glue as u64, &device)?;
        let out_ref = reference_attn(
            &q_glue, &k_seal, &v_seal, &k_glue, &v_glue, sealed, glue, &device,
        )?;

        let backing = fresh_quant_backing(&device, &policy)?;
        let mut cache = bind_cache(&backing, 0)?;
        build_sealed_arena_quant(
            &backing, &mut cache, &k_seal, &v_seal, sealed, &policy, &device,
        )?;
        let (out, _ms) = run_glue(
            &backing, &mut cache, &q_glue, &k_glue, &v_glue, glue, &rope_cs, &stager, &device, 0,
            None,
        )?;

        let diff = max_abs_diff(&out_ref, &out)?;
        eprintln!("paged-glue quant (sealed={sealed}, glue={glue}): max abs diff = {diff:.6e}");
        if !(diff < QUANT_TOL) {
            failures.push(format!("(sealed={sealed}, glue={glue}): diff={diff:.6e}"));
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "paged-glue quant diverged from golden in {} case(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}
