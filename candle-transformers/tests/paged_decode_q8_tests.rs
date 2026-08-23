//! Byte-parity tests for the paged-decode combine kernel's fused q8a1024 emit
//! (B2): the attention context leaves the combine kernel already quantized, so
//! `o_proj` runs int8 with no FP context store and no standalone quantize.
//!
//! What is pinned, as raw expected bytes:
//!
//! * The per-128-tile amax/Σx reduction — a 32-lane warp butterfly followed by
//!   a pairwise combine of the tile's four warp results — at head_dim 128 (one
//!   tile per query head, the historical path) and head_dim 256 (two tiles per
//!   head, the gated-lineage width).
//! * The q8a1024 flat-grouped placement: `flat_tile = row·(head_dim/128) + t`,
//!   qs bytes and the `{scale, Σx}` half2 at their block offsets.
//! * The round-through-O chain: the quantized value is the O-rounded context —
//!   exactly the FP kernel's output — so the expected bytes are derived from a
//!   plain FP decode of the same slot with no tolerance anywhere.
//! * The fused output gate: `sigmoid(g)` computed in f64, narrowed to F32,
//!   rounded through O, multiplied in O precision — the exact arithmetic of
//!   the unfused `sigmoid(gate) ⊙ ctx` chain it replaces.
//!
//! The CPU reference reproduces the kernel's arithmetic operation for
//! operation: IEEE f32 adds in identical order are bit-identical across
//! CPU/GPU, and the kernel writes its scale divisions as explicit `__fdiv_rn`
//! (the latent kernels' mirror-op rule) so they stay IEEE under the archive's
//! `--use_fast_math`. The gate's sigmoid runs in double on both sides
//! (fast-math rewrites float `expf` to the approximate intrinsic, but leaves
//! double `exp` alone); CUDA's double `exp` is ≤1 ulp vs libm's, and a 1-ulp
//! f64 difference survives the narrow to F32 only within ~2⁻²⁹ of an F32
//! boundary — gate values whose sigmoid sits on a bf16 rounding boundary are
//! additionally nudged off it at generation, so the bytes are deterministic
//! by construction, no tolerance anywhere.

#![cfg(feature = "cuda")]

use candle::quantized::pinned_staging::PinnedStager;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{ChunkedKvBacking, KvCache, CHUNK_SIZE};
use candle_transformers::models::prefill_utils::{
    compute_rope_cs, paged_decode_attn, paged_decode_attn_q8, paged_prefill_batched,
};
use half::{bf16, f16};
use std::sync::{Mutex, MutexGuard};

// One GPU, one process-global quantized arena table — same serialization rule
// as the other kernel test binaries.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn gpu_serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const MAX_BLOCKS: usize = 256;
const HISTORY_TOKENS: usize = 100;

#[derive(Clone, Copy)]
struct Geom {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

// ──────────────────────────────────────────────────────────────────────
// Slot construction: one prefilled history the decode step attends over
// ──────────────────────────────────────────────────────────────────────

fn pseudo(i: usize, j: usize, k: usize, seed: u64) -> f32 {
    let mut x = (i as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((j as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add((k as u64).wrapping_mul(0x165667B19E3779F9))
        .wrapping_add(seed.wrapping_mul(0x94D049BB133111EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    (x as i64 as f32) / (i64::MAX as f32) * 0.5
}

/// Flat `[n_tokens, n_head, head_dim]` Q and `[n_tokens, n_kv_head, head_dim]`
/// K/V — the ragged prefill's layout, BF16.
fn make_qkv(
    g: Geom,
    n_tokens: usize,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut q = Vec::with_capacity(n_tokens * g.n_head * g.head_dim);
    let mut k = Vec::with_capacity(n_tokens * g.n_kv_head * g.head_dim);
    let mut v = Vec::with_capacity(n_tokens * g.n_kv_head * g.head_dim);
    for t in 0..n_tokens {
        for h in 0..g.n_head {
            for d in 0..g.head_dim {
                q.push(pseudo(t, h, d, seed ^ 0x111));
            }
        }
        for h in 0..g.n_kv_head {
            for d in 0..g.head_dim {
                k.push(pseudo(t, h, d, seed ^ 0x222));
                v.push(pseudo(t, h, d, seed ^ 0x333));
            }
        }
    }
    Ok((
        Tensor::from_vec(q, (n_tokens, g.n_head, g.head_dim), device)?.to_dtype(DType::BF16)?,
        Tensor::from_vec(k, (n_tokens, g.n_kv_head, g.head_dim), device)?.to_dtype(DType::BF16)?,
        Tensor::from_vec(v, (n_tokens, g.n_kv_head, g.head_dim), device)?.to_dtype(DType::BF16)?,
    ))
}

fn build_history_slot(
    g: Geom,
    seed: u64,
    rope_cs: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<(ChunkedKvBacking, KvCache)> {
    let backing =
        ChunkedKvBacking::new(4, g.n_kv_head, g.head_dim, DType::BF16, device, MAX_BLOCKS)?;
    let mut cache = KvCache::new(2, 64);
    cache.force_dtype(DType::BF16);
    cache.set_chunked_backing(&backing, 0, None)?;

    let (q, k, v) = make_qkv(g, HISTORY_TOKENS, seed, device)?;
    // Chunk allocation is the caller's job (production runs it in wave
    // admission before the forward; the prefill entry neither resets nor
    // allocates).
    backing.ensure_for_batch_entries(&[(0, 0)], HISTORY_TOKENS)?;
    let rope_offsets = Tensor::zeros(1, DType::U32, device)?;
    let generation = stager.begin_generation();
    {
        let mut caches_arr: [&mut KvCache; 1] = [&mut cache];
        let _ = paged_prefill_batched(
            None,
            &mut caches_arr[..],
            &[0],
            &q,
            &k,
            &v,
            1,
            &[HISTORY_TOKENS],
            g.n_head,
            g.n_kv_head,
            g.head_dim,
            None,
            &rope_offsets,
            rope_cs,
            false,
            &generation,
            &std::cell::RefCell::new(None),
        )?;
    }
    cache.set_current_seq_len(HISTORY_TOKENS)?;
    Ok((backing, cache))
}

// ──────────────────────────────────────────────────────────────────────
// One decode step through the production wrappers, FP or q8
// ──────────────────────────────────────────────────────────────────────

enum DecodeEmit<'a> {
    Float,
    Q8 { gate: Option<&'a Tensor> },
}

/// Builds the slot header (slices + position map) exactly as the production
/// `build_decode_metadata` does and runs one decode step. Header state is
/// rebuilt per call, so FP and q8 runs over the same slot see identical
/// metadata and identical split partials.
// Mirrors the production decode launch's own argument list.
#[allow(clippy::too_many_arguments)]
fn decode_one_slot(
    g: Geom,
    backing: &ChunkedKvBacking,
    cache: &KvCache,
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope_cs: &Tensor,
    emit: DecodeEmit<'_>,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_transformers::models::slot_state::{
        tensor_u8_device_ptr, SlotStateHost, TokenSliceHost,
    };

    let seq_offset = cache.current_seq_len();
    let arena_info = backing.resolve_arena_info()?;
    backing.ensure_for_batch_entries(&[(0, seq_offset)], 1)?;

    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);

    let mut slot = SlotStateHost::from_sealed_chunks(
        &chunks,
        g.n_kv_head,
        g.head_dim,
        &arena_info,
        writer_start,
        true,
    );
    slot.extend_for_write_region(1, CHUNK_SIZE);

    let mut records_buf: Vec<u8> = Vec::new();
    let mut rec_offset: Vec<Option<usize>> = Vec::with_capacity(slot.slices.len());
    for s in &slot.slices {
        if s.meta.is_some() {
            rec_offset.push(None);
        } else {
            rec_offset.push(Some(records_buf.len()));
            s.serialize_record(&mut records_buf);
        }
    }
    let records_tensor = if records_buf.is_empty() {
        Tensor::zeros(1, DType::U8, device)?
    } else {
        Tensor::from_slice(&records_buf, records_buf.len(), device)?
    };
    let records_base = tensor_u8_device_ptr(&records_tensor)?;

    let mut slice_buf = Vec::with_capacity(slot.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
    for (s, off) in slot.slices.iter().zip(&rec_offset) {
        let kvheads_ptr = match s.meta.as_ref() {
            Some(m) => m.device_addr(),
            None => records_base + off.expect("non-meta slice has a record offset") as u64,
        };
        s.serialize_slice_header(&mut slice_buf, kvheads_ptr);
    }
    let slices_tensor = Tensor::from_slice(&slice_buf, slice_buf.len(), device)?;
    let slices_base_ptr = tensor_u8_device_ptr(&slices_tensor)?;

    let mut pm = slot.position_map.clone();
    if pm.is_empty() {
        pm.push(0);
    }
    let pm_tensor = Tensor::from_slice(&pm, pm.len(), device)?;
    let pm_base_ptr = {
        let (storage, layout) = pm_tensor.storage_and_layout();
        let cs = match &*storage {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("expected CUDA storage"),
        };
        let stream = cs.device().cuda_stream();
        let s = cs.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);
        let (p, _g) = s.device_ptr(&stream);
        p
    };

    let mut hdr = Vec::with_capacity(24);
    let n_slices = slot.slices.len() as u32;
    let write_slice = slot.write_slice;
    hdr.extend_from_slice(&n_slices.to_le_bytes());
    hdr.extend_from_slice(&write_slice.to_le_bytes());
    hdr.extend_from_slice(&slices_base_ptr.to_le_bytes());
    hdr.extend_from_slice(&pm_base_ptr.to_le_bytes());

    let generation = stager.begin_generation();
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers_gpu = generation.submit(pinned)?;
    let headers_ptr = headers_gpu.dev_ptr();

    let softmax_scale = 1.0f32 / (g.head_dim as f32).sqrt();
    let out = match emit {
        DecodeEmit::Float => paged_decode_attn(
            None,
            q,
            headers_ptr,
            DType::BF16,
            g.n_head,
            g.n_kv_head,
            g.head_dim,
            softmax_scale,
            k_new,
            v_new,
            rope_cs,
            false,
        )?,
        DecodeEmit::Q8 { gate } => paged_decode_attn_q8(
            None,
            q,
            headers_ptr,
            DType::BF16,
            g.n_head,
            g.n_kv_head,
            g.head_dim,
            softmax_scale,
            k_new,
            v_new,
            rope_cs,
            false,
            gate,
        )?,
    };
    drop(headers_gpu);
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────
// CPU reference: the combine kernel's emit arithmetic, operation for
// operation
// ──────────────────────────────────────────────────────────────────────

/// The kernel's full-warp `__shfl_xor_sync` butterfly over 32 lanes: every
/// step folds `lane ^ off` in, offsets 16,8,4,2,1. All lanes converge to the
/// same value; lane 0's is returned. IEEE f32 adds in this exact order are
/// bit-identical to the GPU's.
fn warp_butterfly(vals: &[f32]) -> (f32, f32) {
    assert_eq!(vals.len(), 32);
    let mut amax: Vec<f32> = vals.iter().map(|v| v.abs()).collect();
    let mut sum: Vec<f32> = vals.to_vec();
    for off in [16usize, 8, 4, 2, 1] {
        let pa = amax.clone();
        let ps = sum.clone();
        for lane in 0..32 {
            amax[lane] = pa[lane].max(pa[lane ^ off]);
            sum[lane] = ps[lane] + ps[lane ^ off];
        }
    }
    (amax[0], sum[0])
}

/// Expected q8a1024 bytes for one 128-value tile of O-rounded context values,
/// mirroring the combine kernel: warp butterflies, the pairwise
/// `(w0,w2)+(w1,w3)` combine, `id = 127/amax`, round-to-nearest-even quant,
/// `{amax/127, Σx}` as f16.
fn expected_tile_bytes(vr: &[f32]) -> ([u8; 128], [u8; 4]) {
    assert_eq!(vr.len(), 128);
    let w: Vec<(f32, f32)> = (0..4)
        .map(|i| warp_butterfly(&vr[i * 32..(i + 1) * 32]))
        .collect();
    let tile_amax = (w[0].0.max(w[2].0)).max(w[1].0.max(w[3].0));
    let tile_sum = (w[0].1 + w[2].1) + (w[1].1 + w[3].1);
    let id = if tile_amax != 0.0 {
        127.0f32 / tile_amax
    } else {
        0.0
    };
    let mut qs = [0u8; 128];
    for (i, &x) in vr.iter().enumerate() {
        qs[i] = ((x * id).round_ties_even() as i32 as i8) as u8;
    }
    let scale = f16::from_f32(tile_amax / 127.0);
    let sum_h = f16::from_f32(tile_sum);
    let mut ds = [0u8; 4];
    ds[..2].copy_from_slice(&scale.to_le_bytes());
    ds[2..].copy_from_slice(&sum_h.to_le_bytes());
    (qs, ds)
}

fn q8a1024_qs_off(flat_tile: usize) -> usize {
    (flat_tile >> 3) * 1152 + (flat_tile & 7) * 128
}

fn q8a1024_ds_off(flat_tile: usize) -> usize {
    (flat_tile >> 3) * 1152 + 1024 + (flat_tile & 7) * 16
}

/// `sigmoid(g)` as the kernel computes it: f64 math narrowed to F32, rounded
/// to bf16.
fn sigmoid_bf16(g: f32) -> f32 {
    bf16::from_f32((1.0f64 / (1.0f64 + (-(g as f64)).exp())) as f32).to_f32()
}

/// True when `g`'s sigmoid sits robustly inside its bf16 rounding interval:
/// the kernel's double `exp` is ≤1 ulp vs libm's, and only a sigmoid within
/// that (tiny) wobble of a bf16 boundary could round differently on the two
/// sides. The slack is kept far wider than the actual f64 wobble so a value
/// that passes is robust with margin to spare.
fn sigmoid_is_bf16_robust(g: f32) -> bool {
    let sig = (1.0f64 / (1.0f64 + (-(g as f64)).exp())) as f32;
    let slack = 16.0 * f32::EPSILON;
    let r = bf16::from_f32(sig);
    bf16::from_f32(sig * (1.0 - slack)) == r && bf16::from_f32(sig * (1.0 + slack)) == r
}

// ──────────────────────────────────────────────────────────────────────
// The parity check
// ──────────────────────────────────────────────────────────────────────

fn check_q8_parity(g: Geom, seed: u64) -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);

    // Identity RoPE (zero frequencies) — rotation is out of scope here.
    let inv_freq = Tensor::zeros(g.head_dim / 2, DType::F32, &device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, g.head_dim, &device)?;

    let (backing, cache) = build_history_slot(g, seed, &rope_cs, &stager, &device)?;

    let (q_dec, k_new, v_new) = make_qkv(g, 1, seed ^ 0xD3C0DE, &device)?;

    // Gate values in a moderate range so sigmoid stays informative. Any value
    // whose sigmoid sits on a bf16 rounding boundary (where the kernel's
    // fast-math `expf` could round differently from libm) is nudged off it —
    // deterministic by construction, no reseeding.
    let cols = g.n_head * g.head_dim;
    let gate_vals: Vec<f32> = (0..cols)
        .map(|i| {
            let mut gv = pseudo(0, 0, i, seed ^ 0x9A7E) * 8.0;
            // bf16 has 256 codes per binade; a step of 1/32 moves sigmoid far
            // past any boundary within a few tries. The check runs on the
            // bf16-rounded value — that is what the kernel reads.
            for _ in 0..64 {
                if sigmoid_is_bf16_robust(bf16::from_f32(gv).to_f32()) {
                    return gv;
                }
                gv += 0.03125;
            }
            panic!("could not nudge gate value {gv} off a bf16 sigmoid boundary");
        })
        .collect();
    let gate = Tensor::from_vec(gate_vals, (1, cols), &device)?.to_dtype(DType::BF16)?;

    // FP context — the source of the expected bytes (already O-rounded).
    let fp = decode_one_slot(
        g,
        &backing,
        &cache,
        &q_dec,
        &k_new,
        &v_new,
        &rope_cs,
        DecodeEmit::Float,
        &stager,
        &device,
    )?;
    let ctx: Vec<f32> = fp.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let gate_f32: Vec<f32> = gate.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    let tiles_per_row = g.head_dim / 128;
    let total_tiles = g.n_head * tiles_per_row; // one slot
    for (label, gate_arg) in [("ungated", None), ("gated", Some(&gate))] {
        let q8 = decode_one_slot(
            g,
            &backing,
            &cache,
            &q_dec,
            &k_new,
            &v_new,
            &rope_cs,
            DecodeEmit::Q8 { gate: gate_arg },
            &stager,
            &device,
        )?;
        let bytes = q8.flatten_all()?.to_vec1::<u8>()?;
        assert_eq!(
            bytes.len(),
            total_tiles.div_ceil(8) * 1152,
            "{label}: q8 byte size"
        );

        for row in 0..g.n_head {
            for t in 0..tiles_per_row {
                let mut vr = [0f32; 128];
                for (i, slot) in vr.iter_mut().enumerate() {
                    let mut x = ctx[row * g.head_dim + t * 128 + i];
                    if gate_arg.is_some() {
                        let sig = sigmoid_bf16(gate_f32[row * g.head_dim + t * 128 + i]);
                        x = bf16::from_f32(x * sig).to_f32();
                    }
                    *slot = x;
                }
                let (want_qs, want_ds) = expected_tile_bytes(&vr);
                let flat = row * tiles_per_row + t;
                let qs_off = q8a1024_qs_off(flat);
                let ds_off = q8a1024_ds_off(flat);
                assert_eq!(
                    &bytes[qs_off..qs_off + 128],
                    &want_qs[..],
                    "{label}: qs bytes diverge at head_dim {} row {row} tile {t}",
                    g.head_dim,
                );
                assert_eq!(
                    &bytes[ds_off..ds_off + 4],
                    &want_ds[..],
                    "{label}: ds bytes diverge at head_dim {} row {row} tile {t}",
                    g.head_dim,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn q8_emit_matches_the_fp_context_quantize_at_head_dim_128() -> Result<()> {
    let _serial = gpu_serial();
    check_q8_parity(
        Geom {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 128,
        },
        0xB2_128,
    )
}

#[test]
fn q8_emit_matches_the_fp_context_quantize_at_head_dim_256() -> Result<()> {
    let _serial = gpu_serial();
    check_q8_parity(
        Geom {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 256,
        },
        0xB2_256,
    )
}
