//! **Is the paged decode attention a function of the tokens alone, or of how
//! they are laid out in chunks?**
//!
//! The design treats K/V as freely recombinable 32-token chunks: a projected
//! turn is assembled from borrowed sections with holes between them, and the
//! attention over that patchwork is supposed to equal the attention over the
//! same tokens packed in place. `kernel_layout_tests.rs` checks that under a
//! `1e-2` tolerance at head dim 128 with no GQA — and passes while every partial
//! layout differs by `1e-5`–`1e-4`. That tolerance is exactly the size of the
//! defect this file pins.
//!
//! Two things are different here:
//!
//! - **The model's own geometry.** 24 query heads over 2 KV heads at head dim
//!   256 is `heads_per_group = 12`, which routes to `int8_decode_tile_kernel`
//!   (the split/combine path) rather than the warp kernel the other harness
//!   exercises. It is the kernel the daemon runs.
//! - **Exactness, not tolerance.** Two slots holding bit-identical K/V at
//!   identical positions must produce bit-identical output, because the trunk
//!   amplifies a one-ULP difference at the first attention layer into a changed
//!   token: measured end to end, a borrowed prefix that ends mid-chunk called a
//!   tool 3 times in 10 where the same tokens packed in place called it 6 in
//!   10. So the assertion is `diff == 0`, and a failure names the layout.
//!
//! The mechanism under test, from reading the kernel: each tile's softmax
//! quantizes `p = exp2(s − m_new)` to int8 against the split's RUNNING max, so a
//! token's code depends on which tokens share its tile and on what the running
//! max was when that tile arrived — both properties of the physical layout. The
//! combine hides it while every differing tile's max sits ≥ 2^24 below the
//! global max; the recent tokens and the fresh one are exactly the ones that
//! do not, and a fresh writer chunk is exactly what regroups them.
//!
//! Prefill is already exact under every one of these layouts (the other
//! harness reports `0.0` for all of them), so a failure here is the decode
//! kernel's alone.
//!
//! The same question from the other side closes the file: a decode retried
//! after a failed forward, which had left the device's writer length a token
//! ahead of the host, must attend its own tokens once each and write nowhere
//! but its own chunk — mid-chunk and at a chunk's last slot.
#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use candle::quantized::pinned_staging::PinnedStager;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{ChunkedKvBacking, KvCache, KvFormat, QuantFormat, CHUNK_SIZE};
use candle_transformers::models::prefill_utils::{paged_decode_attn, paged_prefill_batched};
use candle_transformers::models::rope_schedule::{RopeRungs, RopeSchedule};
use candle_transformers::models::slot_header::{SlotHeaderHost, SLOT_HEADER_BYTES};
use std::sync::{Mutex, MutexGuard};

static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn gpu_serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Qwen3.8-Flash-Next's attention geometry — the point of this file.
const N_HEAD: usize = 24;
const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 256;
const MAX_BLOCKS: usize = 256;

/// Segment plans. Slot A prefills the total in one go; slot B is built by
/// injecting each segment sealed, so its chunk usages are exactly the plan and
/// every segment after the first starts in a fresh chunk. The decode then
/// writes ONE more token into each.
///
/// The first group isolates the crossing: the same packing in both slots, the
/// only difference being that slot B's writer boundary sits past its partial
/// tail, so the decode's token lands in a fresh chunk of its own rather than in
/// the partial one. That is what a projected turn does at every section
/// boundary that does not fall on a multiple of 32.
const CASES: &[(&str, &[usize])] = &[
    // ── the crossing, isolated ─────────────────────────────────────────
    ("crossing_9", &[9]),
    ("crossing_32_9", &[32, 9]),
    ("crossing_3x32_25", &[32, 32, 32, 25]),
    (
        "crossing_deep_1225",
        &[
            32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 9,
        ],
    ),
    // ── holes in the middle ────────────────────────────────────────────
    ("hole_7_then_full", &[7, 32, 32, 32]),
    ("hole_mid", &[32, 7, 32, 32, 25]),
    ("holes_mixed", &[5, 17, 32, 11, 32, 25, 6]),
    ("holes_many_small", &[3, 3, 3, 3, 3, 3, 3, 3, 4]),
    // ── the aligned control: must be exact even today ──────────────────
    ("aligned_2x32", &[32, 32]),
    // ── one window, at and around its 32-column edge ───────────────────
    // A 31-token prompt's first decode fills the window exactly (kv_len
    // 32); the next step opens a second window with one token in it.
    ("one_window_30", &[30]),
    ("one_window_31", &[31]),
    ("one_window_32", &[32]),
];

/// The decode step's fresh query and K/V. `peaked` makes the new token's key
/// point along its own query, so its score is the row max by a margin — the
/// self-attention shape a real decode has, and the case where the token's
/// int8 code is most sensitive to which tile it shares.
#[derive(Clone, Copy)]
enum Step {
    Random,
    Peaked,
}

/// The arena the K/V sits in, which decides the kernel path that reads it.
///
/// - `F16`: a float-configured backing. K and V are plain F16 and every quad
///   is vector-readable — the kernel's vector path.
/// - `R16`: a quantised-configured backing, as the model's own is. Live K
///   accumulates in R16 (raw F16 with the Q-capture space) until a seal
///   quantises it, which the kernel reads through its BLOCK path — decode by
///   (palette, rank) with a staging round trip — while V stays F16 on the
///   vector path. The two paths must agree with each other and with the
///   reference; a defect in the block path's staging is invisible to the F16
///   arm and was, the first time.
#[derive(Clone, Copy)]
enum Arena {
    F16,
    R16,
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Arena::F16 => "f16",
            Arena::R16 => "r16",
        })
    }
}

#[test]
fn decode_output_is_independent_of_chunk_layout() -> Result<()> {
    let _serial = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures = Vec::new();
    for arena in [Arena::F16, Arena::R16] {
        for step in [Step::Random, Step::Peaked] {
            for &(name, segments) in CASES {
                let r = run_case(name, segments, arena, step, &device, &stager)?;
                let label = format!("{name:<22} {arena:?} {step:?}");
                match r {
                    Cmp { max_abs: 0.0, .. } => eprintln!("decode {label}  exact"),
                    c => {
                        eprintln!(
                            "decode {label}  DIFFERS: max|d|={:.3e} in {} of {} elements",
                            c.max_abs, c.n_diff, c.n
                        );
                        failures.push(label);
                    }
                }
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "the decode kernel's output depends on the chunk layout in {} case(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

impl std::fmt::Debug for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Step::Random => "random",
            Step::Peaked => "peaked",
        })
    }
}

struct Cmp {
    max_abs: f32,
    n_diff: usize,
    n: usize,
}

fn run_case(
    name: &str,
    segments: &[usize],
    arena: Arena,
    step: Step,
    device: &Device,
    stager: &PinnedStager,
) -> Result<Cmp> {
    let total: usize = segments.iter().sum();
    let seed = hash_str(name);
    let (q_all, k_all, v_all) = make_qkv(total, device, seed)?;
    let rope = RopeRungs::new(
        &RopeSchedule::stated(vec![0f32; HEAD_DIM / 2], usize::MAX)?,
        device,
    )?;

    let (backing_a, mut cache_a) =
        build_control_slot(arena, total, &q_all, &k_all, &v_all, &rope, stager, device)?;
    let (backing_b, mut cache_b) = build_segmented_slot(
        arena, segments, &q_all, &k_all, &v_all, &rope, stager, device,
    )?;
    assert_layout(&cache_a, &cache_b, total, segments, name);

    // **Several decode steps, in lockstep.** One step exercises the read of a
    // layout; the next exercises the read of what the previous step WROTE —
    // the token the kernel scattered into the writer chunk, and the host's
    // bookkeeping of that chunk's length — which a single step never sees.
    // Both slots take the same new token every step, so they can part only
    // through the kernel. The history the reference attends grows with them.
    let mut k_hist = k_all.clone();
    let mut v_hist = v_all.clone();
    let mut worst = Cmp {
        max_abs: 0.0,
        n_diff: 0,
        n: 0,
    };
    for s in 0..STEPS {
        let (q_dec, k_new, v_new) = make_qkv(1, device, seed ^ (0xD3C0DE + s as u64))?;
        let q_dec = q_dec.squeeze(2)?.contiguous()?;
        let v_new = v_new.squeeze(2)?.contiguous()?;
        let k_new = match step {
            Step::Random => k_new.squeeze(2)?.contiguous()?,
            Step::Peaked => {
                // One key per KV head, along the group's first query, scaled
                // so q·k clears every random key's score.
                let mut rows = Vec::with_capacity(N_KV_HEAD);
                for kv in 0..N_KV_HEAD {
                    let qh = kv * (N_HEAD / N_KV_HEAD);
                    rows.push(q_dec.narrow(1, qh, 1)?.affine(0.25, 0.0)?);
                }
                Tensor::cat(&rows, 1)?.contiguous()?
            }
        };

        let out_a = decode_one(
            &backing_a, &cache_a, &q_dec, &k_new, &v_new, &rope, stager, device,
        )?;
        let out_b = decode_one(
            &backing_b, &cache_b, &q_dec, &k_new, &v_new, &rope, stager, device,
        )?;
        // The same two slots through the PRODUCTION slot header — the
        // serializer the engine's decode and `decode_ab` build from — which
        // must describe the layout exactly as the test-side header does.
        let out_ap = decode_one_production(
            &backing_a, &cache_a, &q_dec, &k_new, &v_new, &rope, stager, device,
        )?;
        let out_bp = decode_one_production(
            &backing_b, &cache_b, &q_dec, &k_new, &v_new, &rope, stager, device,
        )?;

        // **Against a reference, not only against each other.** Two layouts
        // that agree to the bit can both be wrong — a defect in how the kernel
        // reads a partial writer chunk is the same defect in both slots. The
        // reference is the attention in F32 over the same F16-rounded inputs;
        // the kernel's int8 Q/K/V/P quantisation puts it a few 1e-3 away,
        // never further. Any layout whose output falls outside that band is
        // wrong, exact or not.
        let reference = reference_attention(&q_dec, &k_hist, &v_hist, &k_new, &v_new)?;
        for (label, out) in [
            ("control", &out_a),
            ("segmented", &out_b),
            ("control/production-header", &out_ap),
            ("segmented/production-header", &out_bp),
        ] {
            let (cos, mae) = against(out, &reference)?;
            if cos < 0.995 || mae > 5e-3 {
                candle::bail!(
                    "{name} step {s}: the {label} slot's decode is WRONG against the F32 \
                     reference: cos={cos:.5} mae={mae:.3e}"
                );
            }
        }
        for (label, test_hdr, prod_hdr) in
            [("control", &out_a, &out_ap), ("segmented", &out_b, &out_bp)]
        {
            let c = compare(test_hdr, prod_hdr)?;
            if c.max_abs > 0.0 {
                candle::bail!(
                    "{name} step {s}: the {label} slot decodes DIFFERENTLY through the \
                     production slot header than through the test header: \
                     max_abs={:.3e} n_diff={}/{}",
                    c.max_abs,
                    c.n_diff,
                    c.n
                );
            }
        }
        let c = compare(&out_a, &out_b)?;
        if c.max_abs > 0.0 && worst.max_abs == 0.0 {
            eprintln!("    ({name} first differs at decode step {s})");
        }
        worst = Cmp {
            max_abs: c.max_abs.max(worst.max_abs),
            n_diff: c.n_diff.max(worst.n_diff),
            n: c.n,
        };

        // The step's token is now in both slots' writer chunks: the host
        // bookkeeping the scheduler does per token, and the reference's
        // history.
        let n = total + s + 1;
        for (backing, cache) in [(&backing_a, &mut cache_a), (&backing_b, &mut cache_b)] {
            backing.set_len(0, n);
            cache.set_current_seq_len(n)?;
        }
        // The history is F16 like the step's tensors: the arena width the
        // kernel was handed, which the reference computes from.
        k_hist = Tensor::cat(&[&k_hist, &k_new.unsqueeze(2)?], 2)?;
        v_hist = Tensor::cat(&[&v_hist, &v_new.unsqueeze(2)?], 2)?;
    }
    Ok(worst)
}

/// Decode steps per case. The packed control's writer chunk holds
/// `total % 32` tokens after its prefill and the segmented slot's a fresh
/// one, so within a few steps one of them crosses a chunk boundary — the
/// event that used to be where two layouts parted.
const STEPS: usize = 4;

/// The decode attention in F32 over F16-rounded inputs, the way decode_ab's
/// golden computes it: one query row per head over `ctx` prefill tokens plus
/// the new one, GQA group `h / (N_HEAD / N_KV_HEAD)`. No RoPE — the harness's
/// rope rungs are the identity (zero frequencies), so positions do not rotate.
fn reference_attention(
    q_dec: &Tensor,
    k_all: &Tensor,
    v_all: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
) -> Result<Vec<f32>> {
    let f16 = |t: &Tensor| -> Result<Vec<f32>> {
        t.to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    // q_dec: [1, N_HEAD, HD]; k_all/v_all: [1, N_KV_HEAD, ctx, HD]; k_new/v_new: [1, N_KV_HEAD, HD]
    let ctx = k_all.dim(2)?;
    let (q, k, v, kn, vn) = (
        f16(q_dec)?,
        f16(k_all)?,
        f16(v_all)?,
        f16(k_new)?,
        f16(v_new)?,
    );
    let group = N_HEAD / N_KV_HEAD;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0f32; N_HEAD * HEAD_DIM];
    for h in 0..N_HEAD {
        let g = h / group;
        let qb = h * HEAD_DIM;
        let mut logits = vec![0f32; ctx + 1];
        for (t, lg) in logits.iter_mut().enumerate().take(ctx) {
            let kb = (g * ctx + t) * HEAD_DIM;
            *lg = (0..HEAD_DIM).map(|d| q[qb + d] * k[kb + d]).sum::<f32>() * scale;
        }
        let knb = g * HEAD_DIM;
        logits[ctx] = (0..HEAD_DIM).map(|d| q[qb + d] * kn[knb + d]).sum::<f32>() * scale;
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for lg in logits.iter_mut() {
            *lg = (*lg - m).exp();
            sum += *lg;
        }
        for d in 0..HEAD_DIM {
            let mut acc = 0f32;
            for (t, &w) in logits.iter().enumerate().take(ctx) {
                acc += w * v[(g * ctx + t) * HEAD_DIM + d];
            }
            acc += logits[ctx] * vn[knb + d];
            out[qb + d] = acc / sum;
        }
    }
    Ok(out)
}

/// Cosine similarity and mean absolute error of a kernel output against the
/// reference.
fn against(out: &Tensor, reference: &[f32]) -> Result<(f32, f32)> {
    let o = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(o.len(), reference.len(), "output and reference sizes");
    let (mut dot, mut na, mut nb, mut mae) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in o.iter().zip(reference.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
        mae += (*a - *b).abs() as f64;
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    Ok((cos as f32, (mae / o.len() as f64) as f32))
}

fn compare(a: &Tensor, b: &Tensor) -> Result<Cmp> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0f32;
    let mut n_diff = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d != 0.0 {
            n_diff += 1;
        }
        max_abs = max_abs.max(d);
    }
    Ok(Cmp {
        max_abs,
        n_diff,
        n: a.len(),
    })
}

// ── slot construction ────────────────────────────────────────────────────

fn fresh_backing(arena: Arena, device: &Device) -> Result<ChunkedKvBacking> {
    match arena {
        Arena::F16 => ChunkedKvBacking::new(4, N_KV_HEAD, HEAD_DIM, DType::F16, device, MAX_BLOCKS),
        // A quantised-configured backing: the live writer format is what
        // `active_kv_formats` gives a quantised K on the GPU — R16 — and the
        // sealed format never comes into it, since nothing here seals.
        Arena::R16 => ChunkedKvBacking::new_with_format(
            4,
            N_KV_HEAD,
            HEAD_DIM,
            KvFormat::Quantized(QuantFormat::Q8_0),
            KvFormat::Quantized(QuantFormat::Q8_0),
            device,
            MAX_BLOCKS,
        ),
    }
}

fn bind(backing: &ChunkedKvBacking, batch_idx: usize) -> Result<KvCache> {
    let mut cache = KvCache::new(2, 64);
    cache.force_dtype(DType::F16);
    cache.set_chunked_backing(backing, batch_idx, None)?;
    Ok(cache)
}

fn build_control_slot(
    arena: Arena,
    total: usize,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rope: &RopeRungs,
    stager: &PinnedStager,
    device: &Device,
) -> Result<(ChunkedKvBacking, KvCache)> {
    let backing = fresh_backing(arena, device)?;
    let mut cache = bind(&backing, 0)?;
    run_prefill(&mut cache, q, k, v, total, rope, stager)?;
    Ok((backing, cache))
}

/// Each segment is prefilled fresh into a scratch slot, sealed, and injected
/// onto slot 0 — the projection's mechanism, so slot 0's writer boundary ends
/// up past the last segment's partial tail exactly as a projected slot's does.
fn build_segmented_slot(
    arena: Arena,
    segments: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rope: &RopeRungs,
    stager: &PinnedStager,
    device: &Device,
) -> Result<(ChunkedKvBacking, KvCache)> {
    let backing = fresh_backing(arena, device)?;
    let mut cache = bind(&backing, 0)?;
    let mut scratch = bind(&backing, 1)?;
    let mut start = 0usize;
    for &len in segments {
        let qs = q.narrow(2, start, len)?.contiguous()?;
        let ks = k.narrow(2, start, len)?.contiguous()?;
        let vs = v.narrow(2, start, len)?.contiguous()?;
        backing.truncate_sequence_to_blocks(1, 0)?;
        scratch.set_current_seq_len(0)?;
        run_prefill(&mut scratch, &qs, &ks, &vs, len, rope, stager)?;
        // Drop the trailing empty chunk the prefill's decode-priming may have
        // appended, so it does not become a phantom slice in slot 0.
        backing.truncate_sequence_to_blocks(1, len.div_ceil(CHUNK_SIZE))?;
        let sealed = backing.record_turn(1)?;
        backing.inject_sealed_at_tail(0, &sealed)?;
        start += len;
        cache.set_current_seq_len(start)?;
    }
    Ok((backing, cache))
}

fn assert_layout(a: &KvCache, b: &KvCache, total: usize, segments: &[usize], name: &str) {
    let ua: Vec<usize> = a
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap()
        .iter()
        .map(|c| c.token_count as usize)
        .collect();
    let ub: Vec<usize> = b
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap()
        .iter()
        .map(|c| c.token_count as usize)
        .collect();
    assert_eq!(
        ua.iter().sum::<usize>(),
        total,
        "[{name}] slot A tokens {ua:?}"
    );
    assert_eq!(ub, segments, "[{name}] slot B layout");
}

/// Flat-packed `[total, n_head, head_dim]` at the cache's F16: the prefill kernel
/// validates its operand widths rather than converting them (hot-path invariant
/// 1b), so the harness hands it what a projection would.
fn flatten(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let flat = |t: &Tensor| -> Result<Tensor> {
        t.transpose(1, 2)?
            .squeeze(0)?
            .to_dtype(DType::F16)?
            .contiguous()
    };
    Ok((flat(q)?, flat(k)?, flat(v)?))
}

fn run_prefill(
    cache: &mut KvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    rope: &RopeRungs,
    stager: &PinnedStager,
) -> Result<()> {
    let offset = cache.current_seq_len();
    let kc = cache.k_cache();
    if let (Some(backing), Some(slot)) = (kc.chunked_backing(), kc.chunked_slot()) {
        backing.ensure_for_batch_entries(&[(slot, offset)], seq_len)?;
    }
    let (qf, kf, vf) = flatten(q, k, v)?;
    let generation = stager.begin_generation();
    let mut caches: [&mut KvCache; 1] = [cache];
    paged_prefill_batched(
        None,
        &mut caches[..],
        &[offset],
        &qf,
        &kf,
        &vf,
        1,
        &[seq_len],
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        None,
        rope,
        false,
        &generation,
        &std::cell::RefCell::new(None),
        None,
    )?;
    caches[0].set_current_seq_len(offset + seq_len)?;
    Ok(())
}

/// One decode step on a slot, with the slot state serialized the way
/// `sync_decode_gpu_chunks` serializes it for the kernel.
fn decode_one(
    backing: &ChunkedKvBacking,
    cache: &KvCache,
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope: &RopeRungs,
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
        N_KV_HEAD,
        HEAD_DIM,
        &arena_info,
        writer_start,
        true,
    );
    slot.extend_for_write_region(1, CHUNK_SIZE);

    let mut records: Vec<u8> = Vec::new();
    let mut rec_off: Vec<Option<usize>> = Vec::with_capacity(slot.slices.len());
    for s in &slot.slices {
        if s.meta.is_some() {
            rec_off.push(None);
        } else {
            rec_off.push(Some(records.len()));
            s.serialize_record(&mut records, None);
        }
    }
    let records_t = if records.is_empty() {
        Tensor::zeros(1, DType::U8, device)?
    } else {
        Tensor::from_slice(&records, records.len(), device)?
    };
    let records_base = tensor_u8_device_ptr(&records_t)?;

    let mut slices: Vec<u8> =
        Vec::with_capacity(slot.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
    for (s, off) in slot.slices.iter().zip(&rec_off) {
        let kvheads_ptr = match s.meta.as_ref() {
            Some(m) => m.device_addr(),
            None => records_base + off.expect("transient slice has a record") as u64,
        };
        s.serialize_slice_header(&mut slices, kvheads_ptr);
    }
    let slices_t = Tensor::from_slice(&slices, slices.len(), device)?;
    let slices_ptr = tensor_u8_device_ptr(&slices_t)?;

    let mut pm = slot.position_map.clone();
    if pm.is_empty() {
        pm.push(0);
    }
    let pm_t = Tensor::from_slice(&pm, pm.len(), device)?;
    let pm_ptr = {
        let (storage, layout) = pm_t.storage_and_layout();
        let cs = match &*storage {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("expected CUDA storage"),
        };
        let stream = cs.device().cuda_stream();
        let s = cs.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);
        let (p, _g) = s.device_ptr(&stream);
        p
    };

    let mut hdr = Vec::with_capacity(SLOT_HEADER_BYTES);
    SlotHeaderHost {
        n_slices: slot.slices.len() as u32,
        write_slice: slot.write_slice,
        slices_ptr,
        position_map_ptr: pm_ptr,
        rope_rung: 0,
    }
    .write(&mut hdr);
    let generation = stager.begin_generation();
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers = generation.submit(pinned)?;

    let out = paged_decode_attn(
        None,
        q,
        headers.dev_ptr(),
        DType::F16,
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        k_new,
        v_new,
        rope,
        false,
        None,
    )?;
    let owned = out.to_owned_tensor()?;
    drop(headers);
    drop(slices_t);
    drop(pm_t);
    Ok(owned)
}

// ── a rolled-back decode, mid-chunk and at a chunk's last slot ───────────
//
// The decode kernel commits its token's length on the device and the host
// counts the token only once the step has run, so a forward that fails after
// its attention ran leaves the device a token ahead of a host that never
// moved. These two cases run one such failure and the same step again, taken
// back by `rollback_to_offset` (the fix) and by a plain `truncate_to_offset`
// (the rollback before it), with a sequence claimed beside slot 0 in the same
// arena.

/// **Mid-chunk, a plain truncate leaves the retry attending its token twice and
/// the device a token ahead after it; the rollback leaves neither.** With 8
/// tokens held, the failed attempt writes slot 8 and commits 9 on the device.
/// After a plain truncate the retry writes its token again at slot 9, attends
/// both copies and commits 10 — one past the 9 the host then counts, which every
/// later step inherits until something re-serialises the buffer. After the
/// rollback the retry reads 8, writes slot 8, attends the host's tokens and its
/// own once, and commits 9.
#[test]
fn a_rolled_back_decode_mid_chunk_attends_its_token_once() -> Result<()> {
    const HELD: usize = 8;
    let Some((device, stager, rope)) = edge_fixture()? else {
        return Ok(());
    };
    for arena in [Arena::F16, Arena::R16] {
        let fixed = retry_after_failure(arena, HELD, Rollback::Fixed, &rope, &stager, &device)?;
        let plain = retry_after_failure(arena, HELD, Rollback::Truncate, &rope, &stager, &device)?;
        for (label, r) in [("rollback", &fixed), ("truncate", &plain)] {
            r.check_neighbour(arena, label)?;
            r.check_failed_commit(arena, label, HELD + 1)?;
        }
        check_against_reference(arena, HELD, &fixed.retry, &device)?;
        fixed.check_retry_commit(arena, "rollback", HELD + 1)?;
        plain.check_retry_commit(arena, "truncate", HELD + 2)?;
        if compare(&fixed.retry, &plain.retry)?.max_abs == 0.0 {
            candle::bail!(
                "{arena:?}: the truncate's retry, attending its token twice, decoded exactly as \
                 the rollback's"
            );
        }
    }
    Ok(())
}

/// **At a chunk's last free slot nothing is written past the chunk, and either
/// take-back's retry attends its own tokens.** With 31 tokens held, the failed
/// attempt writes slot 31 and commits the chunk full. A plain truncate leaves the
/// device reading it full, so the kernel's `within < CHUNK_SIZE` bound skips the
/// retry's scatter and its commit and it attends the 32 slots — the 32nd the
/// failed attempt's copy of the same token at the same position. The edge heals
/// itself there; what it may never do is reach the sequence beside it.
#[test]
fn a_rolled_back_decode_at_a_chunk_edge_writes_only_its_chunk() -> Result<()> {
    const HELD: usize = CHUNK_SIZE - 1;
    let Some((device, stager, rope)) = edge_fixture()? else {
        return Ok(());
    };
    for arena in [Arena::F16, Arena::R16] {
        for (label, rollback) in [
            ("rollback", Rollback::Fixed),
            ("truncate", Rollback::Truncate),
        ] {
            let r = retry_after_failure(arena, HELD, rollback, &rope, &stager, &device)?;
            r.check_neighbour(arena, label)?;
            r.check_failed_commit(arena, label, CHUNK_SIZE)?;
            r.check_retry_commit(arena, label, CHUNK_SIZE)?;
            check_against_reference(arena, HELD, &r.retry, &device)?;
        }
    }
    Ok(())
}

/// The device, the stager and identity rope rungs, or `None` without CUDA.
fn edge_fixture() -> Result<Option<(Device, PinnedStager, RopeRungs)>> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(None);
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let rope = RopeRungs::new(
        &RopeSchedule::stated(vec![0f32; HEAD_DIM / 2], usize::MAX)?,
        &device,
    )?;
    Ok(Some((device, stager, rope)))
}

/// The retry's output against the F32 attention over slot 0's `held` tokens and
/// the step's own, once.
fn check_against_reference(arena: Arena, held: usize, out: &Tensor, device: &Device) -> Result<()> {
    let (_, k, v) = make_qkv(held, device, EDGE_SEED)?;
    let (q_dec, k_new, v_new) = step_qkv(device)?;
    let reference = reference_attention(&q_dec, &k, &v, &k_new, &v_new)?;
    let (cos, mae) = against(out, &reference)?;
    if cos < 0.995 || mae > 5e-3 {
        candle::bail!(
            "{arena:?} held {held}: the retry does not attend its own tokens once each: \
             cos={cos:.5} mae={mae:.3e}"
        );
    }
    Ok(())
}

/// Seed of slot 0's prefill tokens.
const EDGE_SEED: u64 = 0xED6E;

/// How a failed decode step is taken back.
#[derive(Clone, Copy)]
enum Rollback {
    /// `KvCache::rollback_to_offset`: the cut, and the device brought back.
    Fixed,
    /// `KvCache::truncate_to_offset` alone, as the wave rollback did before.
    Truncate,
}

/// What one failed-then-retried decode step did.
struct EdgeRetry {
    /// The retry's attention output.
    retry: Tensor,
    /// Every slice's `len` on the device after the failed attempt, and which
    /// slice is the writer.
    failed_lens: Vec<u16>,
    /// The same as the retry's sync handed them to its kernel, and after it.
    retry_in_lens: Vec<u16>,
    retry_lens: Vec<u16>,
    write_slice: usize,
    /// Whether the neighbouring sequence's K/V read back bit for bit.
    neighbour_intact: bool,
}

impl EdgeRetry {
    fn check_neighbour(&self, arena: Arena, label: &str) -> Result<()> {
        if !self.neighbour_intact {
            candle::bail!("{arena:?} {label}: the decode wrote into the sequence beside it");
        }
        Ok(())
    }

    /// The failed attempt's commit reached the device: its writer reads `want`.
    fn check_failed_commit(&self, arena: Arena, label: &str, want: usize) -> Result<()> {
        self.check_writer(arena, label, "the failed attempt", &self.failed_lens, want)
    }

    /// The retry left its writer reading `want` on the device.
    fn check_retry_commit(&self, arena: Arena, label: &str, want: usize) -> Result<()> {
        self.check_writer(arena, label, "the retry", &self.retry_lens, want)
    }

    fn check_writer(
        &self,
        arena: Arena,
        label: &str,
        after: &str,
        lens: &[u16],
        want: usize,
    ) -> Result<()> {
        if lens.get(self.write_slice) != Some(&(want as u16)) {
            candle::bail!(
                "{arena:?} {label}: after {after} the writer slice {} reads {:?} on the device, \
                 not {want} (handed to the retry as {:?})",
                self.write_slice,
                lens,
                self.retry_in_lens
            );
        }
        Ok(())
    }
}

/// The decode step's query and new K/V, the same in every run.
fn step_qkv(device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
    let (q, k, v) = make_qkv(1, device, 0x57E9)?;
    Ok((
        q.squeeze(2)?.contiguous()?,
        k.squeeze(2)?.contiguous()?,
        v.squeeze(2)?.contiguous()?,
    ))
}

/// Slot 0 prefilled with `held` tokens, slot 1 with a full chunk beside it; one
/// decode of slot 0 that fails after its kernel ran, taken back by `rollback`,
/// and the same step again.
fn retry_after_failure(
    arena: Arena,
    held: usize,
    rollback: Rollback,
    rope: &RopeRungs,
    stager: &PinnedStager,
    device: &Device,
) -> Result<EdgeRetry> {
    let backing = fresh_backing(arena, device)?;
    let mut cache = bind(&backing, 0)?;
    let mut neighbour = bind(&backing, 1)?;
    let (q, k, v) = make_qkv(held, device, EDGE_SEED)?;
    run_prefill(&mut cache, &q, &k, &v, held, rope, stager)?;
    let (qn, kn, vn) = make_qkv(CHUNK_SIZE, device, 0xB1D)?;
    run_prefill(&mut neighbour, &qn, &kn, &vn, CHUNK_SIZE, rope, stager)?;
    let bits = |c: &KvCache| -> Result<Vec<u32>> {
        let (k, v) = c.chunked_read_kv(0, CHUNK_SIZE)?;
        let mut out = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        out.extend(v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
        Ok(out.iter().map(|x| x.to_bits()).collect())
    };
    let neighbour_before = bits(&neighbour)?;

    // The prefill committed on the host; the engine marks the buffer for the
    // next sync, which builds it at the host's length.
    backing.mark_decode_writer_stale(&[(0, held)])?;
    let (q_dec, k_new, v_new) = step_qkv(device)?;
    // The failed attempt: its kernel scatters the token and commits the
    // length on the device, and the host never advances.
    let (_, failed_lens, write_slice) =
        decode_live(&backing, held, &q_dec, &k_new, &v_new, rope, stager, device)?;
    match rollback {
        Rollback::Fixed => cache.rollback_to_offset(held)?,
        Rollback::Truncate => cache.truncate_to_offset(held)?,
    }
    let retry_in_lens = live_lens(&backing, held, stager, device)?;
    let (retry, retry_lens, _) =
        decode_live(&backing, held, &q_dec, &k_new, &v_new, rope, stager, device)?;
    Ok(EdgeRetry {
        retry,
        failed_lens,
        retry_in_lens,
        retry_lens,
        write_slice,
        neighbour_intact: bits(&neighbour)? == neighbour_before,
    })
}

/// Every slice's `len` of slot 0's live buffer as a sync at `offset` hands it
/// to a decode — the sync the next [`decode_live`] repeats.
fn live_lens(
    backing: &ChunkedKvBacking,
    offset: usize,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Vec<u16>> {
    use candle_transformers::models::slot_state::TokenSliceHost;

    let entries = [(0usize, offset)];
    backing.ensure_for_batch_entries(&entries, 1)?;
    let arena_info = backing.resolve_arena_info()?;
    let generation = stager.begin_generation();
    let (seq_ptrs, _) =
        backing.sync_decode_gpu_chunks_snapshot(&entries, &arena_info, &generation, &[false])?;
    let (ptr, n_slices, _) = seq_ptrs[0];
    device.synchronize()?;
    let mut raw = vec![0u8; n_slices as usize * TokenSliceHost::SLICE_HEADER_SIZE];
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("the slice table lives on a CUDA device");
    };
    // SAFETY: `ptr` is the slot's live slice table of `n_slices` headers, and
    // the sync's uploads have landed (synchronised above).
    unsafe { cuda.read_raw(ptr, &mut raw) }?;
    Ok(raw
        .as_chunks::<{ TokenSliceHost::SLICE_HEADER_SIZE }>()
        .0
        .iter()
        .map(|h| u16::from_le_bytes([h[2], h[3]]))
        .collect())
}

/// One decode step of slot 0 at `offset` on its LIVE slot buffer — the
/// zero-copy pointer a plain decode row is handed, whose writer length the
/// kernel commits on the device — with every slice's `len` read back after it.
fn decode_live(
    backing: &ChunkedKvBacking,
    offset: usize,
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope: &RopeRungs,
    stager: &PinnedStager,
    device: &Device,
) -> Result<(Tensor, Vec<u16>, usize)> {
    use candle_transformers::models::slot_state::TokenSliceHost;

    let entries = [(0usize, offset)];
    backing.ensure_for_batch_entries(&entries, 1)?;
    let arena_info = backing.resolve_arena_info()?;
    let generation = stager.begin_generation();
    let (seq_ptrs, _) =
        backing.sync_decode_gpu_chunks_snapshot(&entries, &arena_info, &generation, &[false])?;
    let (ptr, n_slices, write_slice) = seq_ptrs[0];
    let mut hdr = Vec::with_capacity(SLOT_HEADER_BYTES);
    SlotHeaderHost {
        n_slices,
        write_slice,
        slices_ptr: ptr,
        position_map_ptr: 0,
        rope_rung: 0,
    }
    .write(&mut hdr);
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers = generation.submit_resident(pinned)?;
    let out = paged_decode_attn(
        None,
        q,
        headers.dev_ptr(),
        DType::F16,
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        k_new,
        v_new,
        rope,
        false,
        None,
    )?;
    let owned = out.to_owned_tensor()?;
    device.synchronize()?;
    let mut raw = vec![0u8; n_slices as usize * TokenSliceHost::SLICE_HEADER_SIZE];
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("the slice table lives on a CUDA device");
    };
    // SAFETY: `ptr` is the slot's live slice table of `n_slices` headers, and
    // the decode and its commit have finished (synchronised above).
    unsafe { cuda.read_raw(ptr, &mut raw) }?;
    let lens = raw
        .as_chunks::<{ TokenSliceHost::SLICE_HEADER_SIZE }>()
        .0
        .iter()
        .map(|h| u16::from_le_bytes([h[2], h[3]]))
        .collect();
    drop(headers);
    Ok((owned, lens, write_slice as usize))
}

// ── deterministic inputs ─────────────────────────────────────────────────

/// One decode step of slot 0 through the PRODUCTION slot header: the
/// backing's own GPU slice-table snapshot (`sync_decode_gpu_chunks_snapshot`,
/// the path the engine's decode and `decode_ab` build from), launched exactly
/// as [`decode_one`] launches the test-side header.
fn decode_one_production(
    backing: &ChunkedKvBacking,
    cache: &KvCache,
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope: &RopeRungs,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    use candle_transformers::models::slot_state::{SlotStateHost, TokenSliceHost};

    let seq_offset = cache.current_seq_len();
    let entries = [(0usize, seq_offset)];
    backing.ensure_for_batch_entries(&entries, 1)?;
    // The persistent decode slot buffer's writer length self-increments only
    // on decode steps; tokens a prefill or an injection wrote sit past its
    // stale tail until the writer region is re-serialised — the engine marks
    // it after every such write (`mark_decode_slot_states`) and the sync below
    // re-serialises it. Mirror it.
    backing.mark_decode_writer_stale(&entries)?;
    let arena_info = backing.resolve_arena_info()?;
    let generation = stager.begin_generation();
    let (seq_ptrs, _stats) =
        backing.sync_decode_gpu_chunks_snapshot(&entries, &arena_info, &generation, &[false])?;
    let (ptr, n_slices, write_slice) = seq_ptrs[0];

    // The production slice table must say what the test-side one says:
    // slice for slice, (offset, len, rope) and which slice is the writer.
    // The two are built from the same chunk state by different code; the
    // kernel reads only this, so a disagreement here is the whole story of
    // any output difference between the two headers.
    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);
    let mut expect = SlotStateHost::from_sealed_chunks(
        &chunks,
        N_KV_HEAD,
        HEAD_DIM,
        &arena_info,
        writer_start,
        true,
    );
    expect.extend_for_write_region(1, CHUNK_SIZE);
    let mut hdr = Vec::with_capacity(SLOT_HEADER_BYTES);
    SlotHeaderHost {
        n_slices,
        write_slice,
        slices_ptr: ptr,
        position_map_ptr: 0,
        rope_rung: 0,
    }
    .write(&mut hdr);
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers = generation.submit_resident(pinned)?;
    // Read the slice table back only once the generation's uploads are on
    // the device — after the header submit, behind a full device sync.
    device.synchronize()?;
    let mut raw = vec![0u8; n_slices as usize * TokenSliceHost::SLICE_HEADER_SIZE];
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("the slice table lives on a CUDA device");
    };
    // SAFETY: `ptr` is the slot's live slice table of `n_slices` headers.
    unsafe { cuda.read_raw(ptr, &mut raw) }?;
    let prod: Vec<(u16, u16, u32)> = raw
        .as_chunks::<{ TokenSliceHost::SLICE_HEADER_SIZE }>()
        .0
        .iter()
        .map(|h| {
            (
                u16::from_le_bytes([h[0], h[1]]),
                u16::from_le_bytes([h[2], h[3]]),
                u32::from_le_bytes([h[4], h[5], h[6], h[7]]),
            )
        })
        .collect();
    let test: Vec<(u16, u16, u32)> = expect
        .slices
        .iter()
        .map(|s| (s.offset, s.len, s.rope))
        .collect();
    if prod != test || write_slice != expect.write_slice {
        candle::bail!(
            "production slot header disagrees with the test-side header at seq_offset \
             {seq_offset}:\n  production: write_slice={write_slice} slices(offset,len,rope)={prod:?}\n  \
             test-side:  write_slice={} slices(offset,len,rope)={test:?}",
            expect.write_slice
        );
    }

    let out = paged_decode_attn(
        None,
        q,
        headers.dev_ptr(),
        DType::F16,
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        k_new,
        v_new,
        rope,
        false,
        None,
    )?;
    let owned = out.to_owned_tensor()?;
    drop(headers);
    Ok(owned)
}

fn make_qkv(n_tokens: usize, device: &Device, seed: u64) -> Result<(Tensor, Tensor, Tensor)> {
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
    let mut q = Vec::with_capacity(n_tokens * N_HEAD * HEAD_DIM);
    let mut k = Vec::with_capacity(n_tokens * N_KV_HEAD * HEAD_DIM);
    let mut v = Vec::with_capacity(n_tokens * N_KV_HEAD * HEAD_DIM);
    for t in 0..n_tokens {
        for h in 0..N_HEAD {
            for d in 0..HEAD_DIM {
                q.push(pseudo(t, h, d, seed ^ 0x111));
            }
        }
        for h in 0..N_KV_HEAD {
            for d in 0..HEAD_DIM {
                k.push(pseudo(t, h, d, seed ^ 0x222));
                v.push(pseudo(t, h, d, seed ^ 0x333));
            }
        }
    }
    // The fixture arenas are F16, so the operands are emitted at that width once, here —
    // the attention wrappers validate the dtype and never convert it.
    let q = Tensor::from_vec(q, (1, n_tokens, N_HEAD, HEAD_DIM), device)?
        .to_dtype(DType::F16)?
        .transpose(1, 2)?
        .contiguous()?;
    let k = Tensor::from_vec(k, (1, n_tokens, N_KV_HEAD, HEAD_DIM), device)?
        .to_dtype(DType::F16)?
        .transpose(1, 2)?
        .contiguous()?;
    let v = Tensor::from_vec(v, (1, n_tokens, N_KV_HEAD, HEAD_DIM), device)?
        .to_dtype(DType::F16)?
        .transpose(1, 2)?
        .contiguous()?;
    Ok((q, k, v))
}

fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}
