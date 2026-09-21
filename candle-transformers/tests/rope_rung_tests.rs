//! Every paged attention kernel rotates each sequence by the RoPE rung its own
//! `SlotHeader` names, and only Q's rotary pairs take that rung's YaRN `m²`
//! (`docs/progressive_yarn.md` §4.2, §6, §10).
//!
//! **No bleed.** A launch serves sequences on different rungs, and nothing
//! rung-dependent is a launch parameter: each CTA maps to one sequence and
//! reads its table base and `m²` from that sequence's header. So a sequence's
//! output in a shared launch must equal, bit for bit, its output in a launch of
//! its own at the same rung. Each kernel is checked with the rungs interleaved
//! in launch order — slot 0 (where a launch-wide rung would be read from) on the
//! top rung, and no slot at the index of its own rung — so a kernel that took
//! any one sequence's rung for the whole launch, or indexed the rung by slot,
//! produces a different number somewhere. The same checks assert that the rungs'
//! own outputs differ, without which the equality would prove nothing.
//!
//! **One `m²`, on Q's rotary pairs.** A decode against a host reference that
//! rotates K at scale 1 and Q with `m²` on its rotary pairs only, with inputs
//! built so the kernel's int8 quantization of Q and K is exact; the reference's
//! error band is derived from the remaining arithmetic and is narrow enough to
//! reject `m⁰`, `m`, `m⁴` and `m²`-on-every-pair.
//!
//! Kernels reached (the decode route is chosen in `launch_int8_decode_attn` by
//! head dim and heads per group):
//!
//! | Test | Kernel |
//! |---|---|
//! | `decode_sequences_on_different_rungs_share_a_launch_without_bleed` | `int8_decode_bmma_kernel` (HD 128, hpg 2) + combine |
//! | `stripe_decode_…` | `int8_decode_stripe_kernel` (HD 64, hpg 2) + combine |
//! | `warp_per_head_decode_…` | `int8_decode_kernel`, 16 warps (HD 128, hpg 12) + combine |
//! | `tile_decode_…` | `int8_decode_tile_kernel` (HD 256, hpg 2) + combine |
//! | `prefill_…` | `paged_prefill_int8` through `paged_prefill_batched`, headers from the production `build_slot_headers` |
//! | `glue_…` | `run_paged_glue_fp16` |
//! | `q_scale_applies_to_q_rotary_pairs_only` | `int8_decode_bmma_kernel` (HD 128, hpg 2) + combine |

#![cfg(feature = "cuda")]
// Kernel-gate harness: the launch wrappers take the kernel's flat argument list.
#![allow(clippy::too_many_arguments)]

use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard};

use candle::backend::BackendStorage;
use candle::bail;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::quantized::pinned_staging::PinnedStager;
use candle::{DType, Device, Result, Storage, Tensor};
use candle_kernels::paged_glue::run_paged_glue_fp16;
use candle_nn::kv_cache::{ChunkedKvBacking, KvCache, CHUNK_SIZE};
use candle_transformers::models::prefill_utils::{paged_decode_attn, paged_prefill_batched};
use candle_transformers::models::rope_schedule::{mscale, RopeRungs, RopeSchedule, Rung};
use candle_transformers::models::slot_header::{SlotHeaderHost, SLOT_HEADER_BYTES};
use candle_transformers::models::slot_state::{
    tensor_u8_device_ptr, SlotStateHost, TokenSliceHost,
};
use half::f16;

// These tests share one GPU and the process-global quantized arena table and
// decode partial pool, so they must not run concurrently. Poison from a
// panicking sibling is recovered: every test builds its own state.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn gpu_serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn cuda_device() -> Option<Device> {
    match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => Some(d),
        _ => None,
    }
}

// ──────────────────────────────────────────────────────────────────────
// Geometry and schedules
// ──────────────────────────────────────────────────────────────────────

/// One attention geometry. `n_head / n_kv_head` and `head_dim` together pick
/// the decode kernel.
#[derive(Clone, Copy)]
struct Geometry {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl Geometry {
    fn softmax_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// HD 128, two query heads per group: the batched-M MMA decode kernel.
const HD128: Geometry = Geometry {
    n_head: 4,
    n_kv_head: 2,
    head_dim: 128,
};

/// HD 64, two query heads per group: the CUDA-core warp-stripe decode kernel
/// (no 32-wide palette per MMA k-step at this width).
const HD64: Geometry = Geometry {
    n_head: 4,
    n_kv_head: 2,
    head_dim: 64,
};

/// HD 128, twelve query heads per group: past the stripe family's 8, so the
/// 16-warp warp-per-head decode kernel.
const HD128_WIDE: Geometry = Geometry {
    n_head: 12,
    n_kv_head: 1,
    head_dim: 128,
};

/// HD 256, two query heads per group: the INT8 tile decode kernel.
const HD256: Geometry = Geometry {
    n_head: 2,
    n_kv_head: 1,
    head_dim: 256,
};

/// The paged-glue tests' geometry: HD 128, four query heads per group.
const GLUE: Geometry = Geometry {
    n_head: 8,
    n_kv_head: 2,
    head_dim: 128,
};

const MAX_BLOCKS: usize = 256;

/// A progressive YaRN ladder over trained window `l0`: factor 1 to `l0`, 2 to
/// `2·l0`, 4 to `4·l0`, with the temperature. Three rungs with three different
/// frequency sets and three different `m²`.
fn three_rung_schedule(rope_dim: usize, theta: f32, l0: usize) -> Result<RopeSchedule> {
    RopeSchedule::yarn(
        rope_dim,
        theta,
        l0,
        vec![
            Rung {
                ceiling: l0,
                factor: 1.0,
            },
            Rung {
                ceiling: 2 * l0,
                factor: 2.0,
            },
            Rung {
                ceiling: 4 * l0,
                factor: 4.0,
            },
        ],
        true,
    )
}

/// The rung each slot of a shared launch names, in launch order. Slot 0 — the
/// slot a launch-wide rung would be read from — is on the top rung, and no slot
/// sits at the index of its own rung.
const LAUNCH_RUNGS: [u32; 3] = [2, 0, 1];

// ──────────────────────────────────────────────────────────────────────
// Inputs
// ──────────────────────────────────────────────────────────────────────

/// Deterministic value in `[-0.5, 0.5)`.
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

/// Q/K/V for `n_tokens` positions, FLAT-packed `[n_tokens, n_*head, head_dim]`
/// in F16 — the arena's type, so the kernels take them as handed.
fn make_qkv(
    g: Geometry,
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
                q.push(pseudo(t, h, d, seed ^ 0x11));
            }
        }
        for h in 0..g.n_kv_head {
            for d in 0..g.head_dim {
                k.push(pseudo(t, h, d, seed ^ 0x22));
                v.push(pseudo(t, h, d, seed ^ 0x33));
            }
        }
    }
    let q = Tensor::from_vec(q, (n_tokens, g.n_head, g.head_dim), device)?.to_dtype(DType::F16)?;
    let k =
        Tensor::from_vec(k, (n_tokens, g.n_kv_head, g.head_dim), device)?.to_dtype(DType::F16)?;
    let v =
        Tensor::from_vec(v, (n_tokens, g.n_kv_head, g.head_dim), device)?.to_dtype(DType::F16)?;
    Ok((q, k, v))
}

/// `t` stacked `n` times along dim 0: the same row handed to every slot.
fn repeat_rows(t: &Tensor, n: usize) -> Result<Tensor> {
    Tensor::cat(&vec![t.clone(); n], 0)?.contiguous()
}

// ──────────────────────────────────────────────────────────────────────
// Backing, cache, prefill
// ──────────────────────────────────────────────────────────────────────

fn fresh_backing(g: Geometry, device: &Device) -> Result<ChunkedKvBacking> {
    ChunkedKvBacking::new(4, g.n_kv_head, g.head_dim, DType::F16, device, MAX_BLOCKS)
}

fn bind(backing: &ChunkedKvBacking, slot: usize) -> Result<KvCache> {
    let mut cache = KvCache::new(2, 64);
    cache.force_dtype(DType::F16);
    cache.set_chunked_backing(backing, slot, None)?;
    Ok(cache)
}

/// One `paged_prefill_batched` launch over `caches`, each appending its
/// `q_lens[i]` tokens after its current length. `q`/`k`/`v` are FLAT-packed in
/// cache order. The headers — each sequence's rung included — come from the
/// production `build_slot_headers`. Returns the flat attention output.
fn prefill(
    g: Geometry,
    caches: &mut [&mut KvCache],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_lens: &[usize],
    rope: &RopeRungs,
    stager: &PinnedStager,
) -> Result<Tensor> {
    let offsets: Vec<usize> = caches.iter().map(|c| c.current_seq_len()).collect();
    // The writer chunks must exist before the kernel runs: it writes into them
    // and allocates nothing.
    for (cache, (&offset, &len)) in caches.iter().zip(offsets.iter().zip(q_lens)) {
        let kc = cache.k_cache();
        let (Some(backing), Some(slot)) = (kc.chunked_backing(), kc.chunked_slot()) else {
            bail!("prefill: the cache is not bound to a chunked backing");
        };
        backing.ensure_for_batch_entries(&[(slot, offset)], len)?;
    }
    let b_sz = caches.len();
    let generation = stager.begin_generation();
    let out = paged_prefill_batched(
        None,
        caches,
        &offsets,
        q,
        k,
        v,
        b_sz,
        q_lens,
        g.n_head,
        g.n_kv_head,
        g.head_dim,
        None,
        rope,
        false,
        &generation,
        &RefCell::new(None),
        None,
    )?;
    for (cache, (&offset, &len)) in caches.iter_mut().zip(offsets.iter().zip(q_lens)) {
        cache.set_current_seq_len(offset + len)?;
    }
    out.to_owned_tensor()
}

// ──────────────────────────────────────────────────────────────────────
// Device pointers
// ──────────────────────────────────────────────────────────────────────

fn dev_ptr_u32(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cs) = &*storage else {
        bail!("expected a CUDA tensor");
    };
    let stream = cs.device().cuda_stream();
    let s = cs.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);
    let (p, _g) = s.device_ptr(&stream);
    Ok(p)
}

fn dev_ptr_f16(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cs) = &*storage else {
        bail!("expected a CUDA tensor");
    };
    let stream = cs.device().cuda_stream();
    let s = cs.as_cuda_slice::<f16>()?.slice(layout.start_offset()..);
    let (p, _g) = s.device_ptr(&stream);
    Ok(p)
}

// ──────────────────────────────────────────────────────────────────────
// Decode: per-slot payload, one launch over any set of slots
// ──────────────────────────────────────────────────────────────────────

/// One slot's decode payload — its slice table, records and position map on
/// the device — as `sync_decode_gpu_chunks` serialises it. Everything but the
/// rung; the header is written at launch, so the same payload can be launched
/// at any rung, alone or beside other slots.
struct DecodeSlot {
    n_slices: u32,
    write_slice: u32,
    slices_ptr: u64,
    pm_ptr: u64,
    _records: Tensor,
    _slices: Tensor,
    _pm: Tensor,
}

/// Build `slot`'s payload for one decode step. The kernel's write-length commit
/// lands in this payload's own slice table, and the host length is not
/// advanced, so decoding the same slot again — alone or in a shared launch —
/// scatters the same token to the same place and reads the same history.
fn decode_slot(
    g: Geometry,
    backing: &ChunkedKvBacking,
    cache: &KvCache,
    slot: usize,
    device: &Device,
) -> Result<DecodeSlot> {
    let seq_offset = cache.current_seq_len();
    backing.ensure_for_batch_entries(&[(slot, seq_offset)], 1)?;
    let arena_info = backing.resolve_arena_info()?;
    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);
    let mut state = SlotStateHost::from_sealed_chunks(
        &chunks,
        g.n_kv_head,
        g.head_dim,
        &arena_info,
        writer_start,
        true,
    );
    state.extend_for_write_region(1, CHUNK_SIZE);

    // Two sections: out-of-line KvHead records for float slices, then the
    // 16-byte slice headers pointing at them (or at a quantized slice's
    // device-resident meta record).
    let mut records: Vec<u8> = Vec::new();
    let mut rec_off: Vec<Option<usize>> = Vec::with_capacity(state.slices.len());
    for s in &state.slices {
        if s.meta.is_some() {
            rec_off.push(None);
        } else {
            rec_off.push(Some(records.len()));
            // No span layout in a fixture: these are the test's own arenas.
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
        Vec::with_capacity(state.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
    for (s, off) in state.slices.iter().zip(&rec_off) {
        let kvheads_ptr = match (s.meta.as_ref(), off) {
            (Some(m), _) => m.device_addr(),
            (None, Some(off)) => records_base + *off as u64,
            (None, None) => bail!("decode slot: a float slice has no record"),
        };
        s.serialize_slice_header(&mut slices, kvheads_ptr);
    }
    let slices_t = Tensor::from_slice(&slices, slices.len(), device)?;
    let slices_ptr = tensor_u8_device_ptr(&slices_t)?;

    let mut pm = state.position_map.clone();
    if pm.is_empty() {
        pm.push(0);
    }
    let pm_t = Tensor::from_slice(&pm, pm.len(), device)?;
    let pm_ptr = dev_ptr_u32(&pm_t)?;

    Ok(DecodeSlot {
        n_slices: state.slices.len() as u32,
        write_slice: state.write_slice,
        slices_ptr,
        pm_ptr,
        _records: records_t,
        _slices: slices_t,
        _pm: pm_t,
    })
}

/// One `paged_decode_attn` launch over `slots`, each slot's header naming its
/// own rung. `q` is `[slots, n_head, head_dim]`, `k_new`/`v_new`
/// `[slots, n_kv_head, head_dim]`, all F16. Returns `[slots, n_head, head_dim]`.
fn decode_launch(
    g: Geometry,
    slots: &[(&DecodeSlot, u32)],
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope: &RopeRungs,
    stager: &PinnedStager,
) -> Result<Tensor> {
    let mut hdr = Vec::with_capacity(slots.len() * SLOT_HEADER_BYTES);
    for (s, rung) in slots {
        SlotHeaderHost {
            n_slices: s.n_slices,
            write_slice: s.write_slice,
            slices_ptr: s.slices_ptr,
            position_map_ptr: s.pm_ptr,
            rope_rung: *rung,
        }
        .write(&mut hdr);
    }
    let generation = stager.begin_generation();
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers = generation.submit(pinned)?;
    let out = paged_decode_attn(
        None,
        q,
        headers.dev_ptr(),
        DType::F16,
        g.n_head,
        g.n_kv_head,
        g.head_dim,
        g.softmax_scale(),
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

// ──────────────────────────────────────────────────────────────────────
// Bit comparisons
// ──────────────────────────────────────────────────────────────────────

/// The tensor's values as F32 bit patterns. F16 → F32 widening is exact, so
/// two F16 tensors have equal bits here exactly when their own bits are equal.
fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn assert_same_bits(label: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let (a, b) = (bits(got)?, bits(want)?);
    if a.len() != b.len() {
        bail!("{label}: {} elements against {}", a.len(), b.len());
    }
    let mut n_diff = 0usize;
    let mut max_abs = 0f32;
    let mut first = usize::MAX;
    let mut last = 0usize;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if x != y {
            n_diff += 1;
            first = first.min(i);
            last = i;
            max_abs = max_abs.max((f32::from_bits(*x) - f32::from_bits(*y)).abs());
        }
    }
    if n_diff != 0 {
        bail!(
            "{label}: {n_diff} of {} elements differ, element {first} to {last} \
             (max |d| = {max_abs:.3e})",
            a.len()
        );
    }
    Ok(())
}

/// Every pair of `outs` differs somewhere. Without it an equality between a
/// shared launch and solo launches could hold because the rung changes nothing.
fn assert_all_differ(label: &str, outs: &[(u32, Tensor)]) -> Result<()> {
    let b: Vec<Vec<u32>> = outs.iter().map(|(_, t)| bits(t)).collect::<Result<_>>()?;
    for i in 0..outs.len() {
        for j in i + 1..outs.len() {
            if b[i] == b[j] {
                bail!(
                    "{label}: rung {} and rung {} produce identical outputs — the rungs do not \
                     change this input, so the no-bleed check above proves nothing",
                    outs[i].0,
                    outs[j].0
                );
            }
        }
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// 1. Decode: sequences on different rungs share a launch without bleed
// ──────────────────────────────────────────────────────────────────────

/// History every decode slot holds before its decode step.
///
/// Short on purpose. The split-KV factor is sized from the launch's slot count,
/// so a solo launch and a three-slot launch split each slot's KV differently
/// unless the work runs out first; with 41 tokens (≤ 7 tiles of any decode
/// kernel's tiling) both launches have more splits than tiles on any card with
/// 21 SMs or more, so every split holds the same single tile in both and the
/// combine merges the same partials in the same order. Equality is then a
/// property of the rung's source alone.
const DECODE_HISTORY: usize = 40;

/// A decode route under test: the geometry that selects the kernel and the
/// schedule's rotary width.
struct DecodePath {
    name: &'static str,
    geometry: Geometry,
    rope_dim: usize,
    theta: f32,
}

/// Three slots holding identical history, decoding the identical token, their
/// headers naming rungs [`LAUNCH_RUNGS`] in ONE launch: each slot's output
/// equals, bit for bit, the same slot decoded alone at its rung, and the three
/// rungs' outputs differ from each other.
fn decode_no_bleed(path: &DecodePath) -> Result<()> {
    let _serial = gpu_serial();
    let Some(device) = cuda_device() else {
        eprintln!("skipping: CUDA device required");
        return Ok(());
    };
    let stager = PinnedStager::new_from_device(&device);
    let g = path.geometry;
    let rope = RopeRungs::new(
        &three_rung_schedule(path.rope_dim, path.theta, 64)?,
        &device,
    )?;
    assert_eq!(rope.n_rungs(), LAUNCH_RUNGS.len());

    let backing = fresh_backing(g, &device)?;
    let (q_hist, k_hist, v_hist) = make_qkv(g, DECODE_HISTORY, 0x4157_0001, &device)?;
    let mut caches = Vec::with_capacity(LAUNCH_RUNGS.len());
    for slot in 0..LAUNCH_RUNGS.len() {
        let mut cache = bind(&backing, slot)?;
        prefill(
            g,
            &mut [&mut cache],
            &q_hist,
            &k_hist,
            &v_hist,
            &[DECODE_HISTORY],
            &rope,
            &stager,
        )?;
        caches.push(cache);
    }
    let (q, k_new, v_new) = make_qkv(g, 1, 0x4157_0002, &device)?;

    // Each slot alone, at the rung the shared launch will give it.
    let mut solo: Vec<(u32, Tensor)> = Vec::with_capacity(LAUNCH_RUNGS.len());
    for (slot, &rung) in LAUNCH_RUNGS.iter().enumerate() {
        let s = decode_slot(g, &backing, &caches[slot], slot, &device)?;
        let out = decode_launch(g, &[(&s, rung)], &q, &k_new, &v_new, &rope, &stager)?;
        solo.push((rung, out));
    }
    assert_all_differ(path.name, &solo)?;

    // All three in one launch.
    let slots: Vec<DecodeSlot> = (0..LAUNCH_RUNGS.len())
        .map(|slot| decode_slot(g, &backing, &caches[slot], slot, &device))
        .collect::<Result<_>>()?;
    let launch: Vec<(&DecodeSlot, u32)> = slots.iter().zip(LAUNCH_RUNGS).collect();
    let n = LAUNCH_RUNGS.len();
    let shared = decode_launch(
        g,
        &launch,
        &repeat_rows(&q, n)?,
        &repeat_rows(&k_new, n)?,
        &repeat_rows(&v_new, n)?,
        &rope,
        &stager,
    )?;
    for (slot, (rung, alone)) in solo.iter().enumerate() {
        assert_same_bits(
            &format!(
                "{}: slot {slot} on rung {rung}, shared launch vs alone",
                path.name
            ),
            &shared.narrow(0, slot, 1)?,
            alone,
        )?;
    }

    // **The slot's rung table is its only source.** Solo against shared cannot
    // see a kernel that reads something from rung 0 — the tile kernel's unit
    // step, say — because the solo run reads it too. So: rung 1 of a two-rung
    // set without temperature must equal, bit for bit, rung 0 of a set holding
    // only rung 1's frequencies, where there is nothing else to read.
    let two = RopeRungs::new(
        &RopeSchedule::yarn(
            path.rope_dim,
            path.theta,
            64,
            vec![
                Rung {
                    ceiling: 64,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 128,
                    factor: 4.0,
                },
            ],
            false,
        )?,
        &device,
    )?;
    let only = RopeRungs::new(
        &RopeSchedule::stated(two.inv_freq(1).to_vec(), usize::MAX)?,
        &device,
    )?;
    let upper = {
        let s = decode_slot(g, &backing, &caches[0], 0, &device)?;
        decode_launch(g, &[(&s, 1)], &q, &k_new, &v_new, &two, &stager)?
    };
    let lower = {
        let s = decode_slot(g, &backing, &caches[0], 0, &device)?;
        decode_launch(g, &[(&s, 0)], &q, &k_new, &v_new, &two, &stager)?
    };
    let sole = {
        let s = decode_slot(g, &backing, &caches[0], 0, &device)?;
        decode_launch(g, &[(&s, 0)], &q, &k_new, &v_new, &only, &stager)?
    };
    assert_all_differ(path.name, &[(0, lower), (1, upper.clone())])?;
    assert_same_bits(
        &format!(
            "{}: rung 1 of two against the same frequencies alone",
            path.name
        ),
        &upper,
        &sole,
    )?;
    Ok(())
}

/// The 16-warp decode kernel scores every token of its 16-token tile. Its
/// int8 tensor-core QK^T computes eight tokens per instruction, so a tile is two
/// instructions' worth; scoring only the first eight left the other eight
/// logits as whatever the block's shared memory last held, and the output
/// changed with the launch that ran before it. Twelve query heads over one KV
/// head (the geometry that selects this kernel), decoded at rung 2 against the
/// f64 reference of `q_scale_applies_to_q_rotary_pairs_only`, whose band is
/// narrow enough that one wrong logit in a 49-token history falls outside it.
#[test]
fn warp_per_head_decode_matches_the_host_reference() -> Result<()> {
    let _serial = gpu_serial();
    let Some(device) = cuda_device() else {
        eprintln!("skipping: CUDA device required");
        return Ok(());
    };
    let stager = PinnedStager::new_from_device(&device);
    let g = HD128_WIDE;
    let rope = RopeRungs::new(
        &RopeSchedule::yarn(
            Q4_ROPE_DIM,
            Q4_THETA,
            Q4_L0,
            vec![
                Rung {
                    ceiling: Q4_L0,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 2 * Q4_L0,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 8 * Q4_L0,
                    factor: Q4_FACTOR as f32,
                },
            ],
            true,
        )?,
        &device,
    )?;
    let inp = q4_inputs(g, &device)?;
    let backing = fresh_backing(g, &device)?;
    let mut cache = bind(&backing, 0)?;
    prefill(
        g,
        &mut [&mut cache],
        &inp.q_history,
        &inp.k_history,
        &inp.v_history,
        &[Q4_HISTORY],
        &rope,
        &stager,
    )?;
    let slot = decode_slot(g, &backing, &cache, 0, &device)?;
    let out = decode_launch(
        g,
        &[(&slot, Q4_RUNG)],
        &inp.q,
        &inp.k_new,
        &inp.v_new,
        &rope,
        &stager,
    )?;
    let m2 = rope.q_scale(Q4_RUNG) as f64;
    let want = reference_decode(
        g,
        &rope,
        Q4_RUNG,
        QScale {
            rotary: m2,
            pass: 1.0,
        },
        true,
        &inp.q_host,
        &inp.keys_host,
        &inp.values_host,
    )?;
    assert_within("warp-per-head decode, rung 2", &out, &want)
}

/// HD 128 at two heads per group: `int8_decode_bmma_kernel`.
#[test]
fn decode_sequences_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
    decode_no_bleed(&DecodePath {
        name: "bmma decode",
        geometry: HD128,
        rope_dim: 128,
        theta: 1e6,
    })
}

/// HD 64 at two heads per group: `int8_decode_stripe_kernel`.
#[test]
fn stripe_decode_sequences_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
    decode_no_bleed(&DecodePath {
        name: "stripe decode",
        geometry: HD64,
        rope_dim: 64,
        theta: 1e4,
    })
}

/// HD 128 at twelve heads per group: the 16-warp `int8_decode_kernel`.
#[test]
fn warp_per_head_decode_sequences_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
    decode_no_bleed(&DecodePath {
        name: "warp-per-head decode",
        geometry: HD128_WIDE,
        rope_dim: 128,
        theta: 1e6,
    })
}

/// HD 256 at two heads per group: `int8_decode_tile_kernel`, whose unit step
/// between consecutive tokens is `LO[1]` of the slot's own rung table. Partial
/// rotary (64 of 256 dims), as the hybrid lineage runs.
#[test]
fn tile_decode_sequences_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
    decode_no_bleed(&DecodePath {
        name: "tile decode",
        geometry: HD256,
        rope_dim: 64,
        theta: 1e7,
    })
}

// ──────────────────────────────────────────────────────────────────────
// 2. Prefill: the same, through the production header builder
// ──────────────────────────────────────────────────────────────────────

/// Trained window of the prefill schedule: ceilings 32, 64, 128.
const PREFILL_L0: usize = 32;

/// `(history, q_len, seed)` per sequence, in launch order. The reach
/// `history + q_len` is 100, 28 and 52 — rungs 2, 0 and 1 under
/// `three_rung_schedule(.., PREFILL_L0)` — with unequal `q_len`.
const PREFILL_SEQS: [(usize, usize, u64); 3] = [(90, 10, 0xA1), (20, 8, 0xB2), (40, 12, 0xC3)];

type Qkv = (Tensor, Tensor, Tensor);

/// A fresh backing holding only this sequence's history in slot 0, then its
/// new tokens prefilled alone. Returns the new tokens' attention output.
fn prefill_alone(
    g: Geometry,
    history: &Qkv,
    new: &Qkv,
    rope: &RopeRungs,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    let backing = fresh_backing(g, device)?;
    let mut cache = bind(&backing, 0)?;
    let hist_len = history.0.dim(0)?;
    prefill(
        g,
        &mut [&mut cache],
        &history.0,
        &history.1,
        &history.2,
        &[hist_len],
        rope,
        stager,
    )?;
    let new_len = new.0.dim(0)?;
    prefill(
        g,
        &mut [&mut cache],
        &new.0,
        &new.1,
        &new.2,
        &[new_len],
        rope,
        stager,
    )
}

/// Three sequences whose production headers (`build_slot_headers`, rung from
/// `rope.rung_for(offset + q_len)`) land on rungs 2, 0 and 1, prefilled in ONE
/// `paged_prefill_batched` launch over a sealed history each: every sequence's
/// output equals, bit for bit, its prefill alone. The sequence on rung 0 also
/// equals its prefill under plain RoPE — rung 0 is the trained RoPE exactly —
/// and the other two differ from theirs, so the rungs are what the equality
/// is about.
#[test]
fn prefill_sequences_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
    let _serial = gpu_serial();
    let Some(device) = cuda_device() else {
        eprintln!("skipping: CUDA device required");
        return Ok(());
    };
    let stager = PinnedStager::new_from_device(&device);
    let g = HD128;
    let rope = RopeRungs::new(&three_rung_schedule(128, 1e6, PREFILL_L0)?, &device)?;
    let plain = RopeRungs::new(&RopeSchedule::plain(128, 1e6, 4 * PREFILL_L0), &device)?;

    // The schedule puts the sequences where the test says it does.
    let rungs: Vec<u32> = PREFILL_SEQS
        .iter()
        .map(|&(h, n, _)| rope.rung_for(h + n))
        .collect::<Result<_>>()?;
    assert_eq!(rungs, LAUNCH_RUNGS, "the prefill reaches' rungs");

    let histories: Vec<Qkv> = PREFILL_SEQS
        .iter()
        .map(|&(h, _, seed)| make_qkv(g, h, seed ^ 0x4157, &device))
        .collect::<Result<_>>()?;
    let news: Vec<Qkv> = PREFILL_SEQS
        .iter()
        .map(|&(_, n, seed)| make_qkv(g, n, seed ^ 0x9E3, &device))
        .collect::<Result<_>>()?;

    let mut alone = Vec::with_capacity(PREFILL_SEQS.len());
    for (i, &rung) in rungs.iter().enumerate() {
        let out = prefill_alone(g, &histories[i], &news[i], &rope, &stager, &device)?;
        let under_plain = prefill_alone(g, &histories[i], &news[i], &plain, &stager, &device)?;
        let label = format!("prefill: sequence {i} (rung {rung}) against plain RoPE");
        if rung == 0 {
            assert_same_bits(&label, &out, &under_plain)?;
        } else if bits(&out)? == bits(&under_plain)? {
            bail!("{label}: identical — rung {rung} changes nothing for this input");
        }
        alone.push(out);
    }

    // All three in one launch, histories in slots 0..3 of one backing.
    let backing = fresh_backing(g, &device)?;
    let mut caches: Vec<KvCache> = Vec::with_capacity(PREFILL_SEQS.len());
    for (slot, history) in histories.iter().enumerate() {
        let mut cache = bind(&backing, slot)?;
        let hist_len = history.0.dim(0)?;
        prefill(
            g,
            &mut [&mut cache],
            &history.0,
            &history.1,
            &history.2,
            &[hist_len],
            &rope,
            &stager,
        )?;
        caches.push(cache);
    }
    let q_lens: Vec<usize> = PREFILL_SEQS.iter().map(|&(_, n, _)| n).collect();
    let cat = |pick: fn(&Qkv) -> &Tensor| -> Result<Tensor> {
        Tensor::cat(&news.iter().map(pick).collect::<Vec<_>>(), 0)?.contiguous()
    };
    let (q, k, v) = (cat(|t| &t.0)?, cat(|t| &t.1)?, cat(|t| &t.2)?);
    let mut refs: Vec<&mut KvCache> = caches.iter_mut().collect();
    let shared = prefill(g, &mut refs, &q, &k, &v, &q_lens, &rope, &stager)?;

    let mut start = 0usize;
    for (i, (&n, &rung)) in q_lens.iter().zip(&rungs).enumerate() {
        assert_same_bits(
            &format!("prefill: sequence {i} on rung {rung}, shared launch vs alone"),
            &shared.narrow(0, start, n)?,
            &alone[i],
        )?;
        start += n;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// 3. Glue: the same, through the glue kernel
// ──────────────────────────────────────────────────────────────────────

const GLUE_SEALED: usize = 40;
const GLUE_TOKENS: usize = 6;

/// A conversation in its own backing: `GLUE_SEALED` tokens written straight to
/// the F16 arena in slot 0 (stored un-rotated; the kernel rotates at read).
fn glue_conversation(g: Geometry, device: &Device) -> Result<(ChunkedKvBacking, KvCache)> {
    let backing = fresh_backing(g, device)?;
    let mut cache = bind(&backing, 0)?;
    let (_, k, v) = make_qkv(g, GLUE_SEALED, 0x61E0_0001, device)?;
    // `[sealed, n_kv, hd]` → `[1, n_kv, sealed, hd]`, the layout `write_contiguous` takes.
    let kr = k.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
    let vr = v.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
    backing.ensure_for_offset(0, 0, GLUE_SEALED)?;
    backing.write_contiguous(0, 0, &kr, &vr)?;
    backing.set_len(0, GLUE_SEALED);
    cache.set_current_seq_len(GLUE_SEALED)?;
    Ok((backing, cache))
}

/// One conversation's glue payload: slice table, records, position map, and
/// each glue token's write target in the writer region.
struct GlueSlot {
    n_slices: u32,
    write_slice: u32,
    slices_ptr: u64,
    pm_ptr: u64,
    kv_len: usize,
    wslice: Vec<u32>,
    winblk: Vec<u32>,
    _slices: Tensor,
    _pm: Tensor,
    _records: Tensor,
}

/// Where a slice's KvHead record lives: a quantized slice's device-resident
/// meta record, or an offset into the payload's own records buffer.
enum KvHeads {
    Resident(u64),
    Scratch(usize),
}

fn glue_slot(
    g: Geometry,
    backing: &ChunkedKvBacking,
    cache: &mut KvCache,
    device: &Device,
) -> Result<GlueSlot> {
    let prefix_len = cache.current_seq_len();
    let kv_len = prefix_len + GLUE_TOKENS;
    backing.ensure_for_batch_entries(&[(0, prefix_len)], GLUE_TOKENS)?;
    let arena_info = backing.resolve_arena_info()?;
    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);
    let mut state = SlotStateHost::from_sealed_chunks(
        &chunks,
        g.n_kv_head,
        g.head_dim,
        &arena_info,
        writer_start,
        true,
    );
    state.extend_for_write_region(GLUE_TOKENS, CHUNK_SIZE);

    // Resident (quantized) slices point at their device meta record; float
    // slices serialise a record into this payload.
    let mut records: Vec<u8> = Vec::new();
    let mut kvheads: Vec<KvHeads> = Vec::with_capacity(state.slices.len());
    for s in &state.slices {
        match &s.meta {
            Some(meta) => kvheads.push(KvHeads::Resident(
                cache.k_cache().chunked_meta_device_addr(meta),
            )),
            None => {
                kvheads.push(KvHeads::Scratch(records.len()));
                // No span layout in a fixture: these are the test's own arenas.
                s.serialize_record(&mut records, None);
            }
        }
    }
    if records.is_empty() {
        records.push(0u8);
    }
    let records_t = Tensor::from_slice(&records, records.len(), device)?;
    let records_base = tensor_u8_device_ptr(&records_t)?;

    let mut slices = Vec::with_capacity(state.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
    for (s, kv) in state.slices.iter().zip(&kvheads) {
        let kvheads_ptr = match *kv {
            KvHeads::Resident(addr) => addr,
            KvHeads::Scratch(off) => records_base + off as u64,
        };
        s.serialize_slice_header(&mut slices, kvheads_ptr);
    }
    let slices_t = Tensor::from_slice(&slices, slices.len(), device)?;
    let slices_ptr = tensor_u8_device_ptr(&slices_t)?;

    let pm = state.position_map.clone();
    let pm_t = Tensor::from_slice(&pm, pm.len().max(1), device)?;
    let pm_ptr = dev_ptr_u32(&pm_t)?;

    // Each glue token's write target: `(slice << 16) | in_blk` at its position.
    let mut wslice = Vec::with_capacity(GLUE_TOKENS);
    let mut winblk = Vec::with_capacity(GLUE_TOKENS);
    for t in 0..GLUE_TOKENS {
        let e = pm[prefix_len + t];
        wslice.push(e >> 16);
        winblk.push(e & 0xffff);
    }
    cache.set_current_seq_len(kv_len)?;
    Ok(GlueSlot {
        n_slices: state.slices.len() as u32,
        write_slice: state.write_slice,
        slices_ptr,
        pm_ptr,
        kv_len,
        wslice,
        winblk,
        _slices: slices_t,
        _pm: pm_t,
        _records: records_t,
    })
}

/// One glue launch over `slots`, every slot taking the same glue tokens
/// (`[GLUE_TOKENS, n_*head, head_dim]`, F16), causal. Returns the flat output
/// `[slots · GLUE_TOKENS, n_head, head_dim]`.
fn glue_launch(
    g: Geometry,
    slots: &[(&GlueSlot, u32)],
    glue: &Qkv,
    rope: &RopeRungs,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    let b = slots.len();
    let total_q = b * GLUE_TOKENS;
    let mut hdr = Vec::with_capacity(b * SLOT_HEADER_BYTES);
    let mut cu = vec![0u32];
    let mut wslice: Vec<u32> = Vec::with_capacity(total_q);
    let mut winblk: Vec<u32> = Vec::with_capacity(total_q);
    for (i, (s, rung)) in slots.iter().enumerate() {
        SlotHeaderHost {
            n_slices: s.n_slices,
            write_slice: s.write_slice,
            slices_ptr: s.slices_ptr,
            position_map_ptr: s.pm_ptr,
            rope_rung: *rung,
        }
        .write(&mut hdr);
        cu.push(((i + 1) * GLUE_TOKENS) as u32);
        wslice.extend_from_slice(&s.wslice);
        winblk.extend_from_slice(&s.winblk);
    }
    let max_kv = slots.iter().map(|(s, _)| s.kv_len).max().unwrap_or(0);
    let kv_lens: Vec<u32> = slots.iter().map(|(s, _)| s.kv_len as u32).collect();

    let qf = repeat_rows(&glue.0, b)?;
    let kf = repeat_rows(&glue.1, b)?;
    let vf = repeat_rows(&glue.2, b)?;
    let out = Tensor::zeros((total_q, g.n_head, g.head_dim), DType::F16, device)?;
    let cu_seqlens_q = Tensor::from_vec(cu, b + 1, device)?;
    let q_lens = Tensor::from_vec(vec![GLUE_TOKENS as u32; b], b, device)?;
    let kv_lens = Tensor::from_vec(kv_lens, b, device)?;
    let glue_write_slice = Tensor::from_vec(wslice, total_q, device)?;
    let glue_write_in_blk = Tensor::from_vec(winblk, total_q, device)?;
    let fwd_ahead = Tensor::from_vec(vec![0u32; total_q], total_q, device)?;

    let generation = stager.begin_generation();
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers = generation.submit(pinned)?;

    let Device::Cuda(dev) = device else {
        bail!("glue launch: CUDA device required");
    };
    let stream_ptr = dev.cuda_stream().cu_stream() as *mut c_void;
    let rungs = rope.ffi()?;
    device.synchronize()?;
    // SAFETY: every pointer is a live device allocation held until the
    // synchronize below; lengths match the kernel's documented shapes.
    unsafe {
        run_paged_glue_fp16(
            dev_ptr_f16(&qf)? as *const c_void,
            headers.dev_ptr() as *const u8,
            dev_ptr_f16(&out)? as *mut c_void,
            b as i32,
            GLUE_TOKENS as i32,
            total_q as i32,
            max_kv as i32,
            g.n_head as i32,
            g.n_kv_head as i32,
            g.head_dim as i32,
            g.softmax_scale(),
            dev_ptr_f16(&kf)? as *const c_void,
            dev_ptr_f16(&vf)? as *const c_void,
            rungs,
            0,
            dev_ptr_u32(&cu_seqlens_q)? as *const u32,
            dev_ptr_u32(&q_lens)? as *const u32,
            dev_ptr_u32(&kv_lens)? as *const u32,
            dev_ptr_u32(&glue_write_slice)? as *const u32,
            dev_ptr_u32(&glue_write_in_blk)? as *const u32,
            dev_ptr_u32(&fwd_ahead)? as *const u32,
            stream_ptr,
        );
    }
    device.synchronize()?;
    drop(headers);
    Ok(out)
}

/// Three conversations with the identical sealed prefix and glue tokens, each
/// in its own backing, their headers naming rungs [`LAUNCH_RUNGS`] in ONE glue
/// launch: each equals, bit for bit, the same conversation glued alone at its
/// rung, and the three rungs' outputs differ.
#[test]
fn glue_sequences_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
    let _serial = gpu_serial();
    let Some(device) = cuda_device() else {
        eprintln!("skipping: CUDA device required");
        return Ok(());
    };
    let stager = PinnedStager::new_from_device(&device);
    let g = GLUE;
    let rope = RopeRungs::new(&three_rung_schedule(128, 1e6, 64)?, &device)?;
    let glue = make_qkv(g, GLUE_TOKENS, 0x61E0_0002, &device)?;

    // A glue launch writes its tokens' K/V, so every run gets fresh conversations.
    let mut solo: Vec<(u32, Tensor)> = Vec::with_capacity(LAUNCH_RUNGS.len());
    for &rung in &LAUNCH_RUNGS {
        let (backing, mut cache) = glue_conversation(g, &device)?;
        let s = glue_slot(g, &backing, &mut cache, &device)?;
        solo.push((
            rung,
            glue_launch(g, &[(&s, rung)], &glue, &rope, &stager, &device)?,
        ));
    }
    assert_all_differ("glue", &solo)?;

    let mut convs = Vec::with_capacity(LAUNCH_RUNGS.len());
    for _ in &LAUNCH_RUNGS {
        convs.push(glue_conversation(g, &device)?);
    }
    let slots: Vec<GlueSlot> = convs
        .iter_mut()
        .map(|(backing, cache)| glue_slot(g, backing, cache, &device))
        .collect::<Result<_>>()?;
    let launch: Vec<(&GlueSlot, u32)> = slots.iter().zip(LAUNCH_RUNGS).collect();
    let shared = glue_launch(g, &launch, &glue, &rope, &stager, &device)?;
    for (slot, (rung, alone)) in solo.iter().enumerate() {
        assert_same_bits(
            &format!("glue: conversation {slot} on rung {rung}, shared launch vs alone"),
            &shared.narrow(0, slot * GLUE_TOKENS, GLUE_TOKENS)?,
            alone,
        )?;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// 4. m² on Q's rotary pairs only, against a host reference
// ──────────────────────────────────────────────────────────────────────

/// Partial rotary: 32 rotary pairs of the 64 half-split pairs at HD 128, so
/// pairs 32..64 pass through.
const Q4_ROPE_DIM: usize = 64;
const Q4_THETA: f32 = 1e4;
const Q4_L0: usize = 64;
/// The rung under test: factor 4, `m² = (0.1·ln 4 + 1)²`.
const Q4_RUNG: u32 = 2;
const Q4_FACTOR: f64 = 4.0;
/// History length; the decoded token is position `Q4_HISTORY`.
const Q4_HISTORY: usize = 48;
/// The rotary pair Q and K carry: dims 5 and 69, which lie in quantization
/// windows 0 and 2.
const Q4_PAIR: usize = 5;
/// The pass-through dim Q and K carry: dim 40, window 1, pair 40 ≥ 32.
const Q4_PASS: usize = 40;
/// Width of the decode kernel's int8 quantization windows (one palette).
const QUANT_WINDOW: usize = 32;

/// Per query head of [`HD128`]: `(dim 5, dim 69, dim 40)`. F16-exact.
const Q4_Q: [(f32, f32, f32); 4] = [
    (6.0, -4.0, 3.0),
    (-5.0, 5.5, -2.5),
    (4.5, 6.0, 2.0),
    (-6.5, 3.5, -3.5),
];

// ── The reference's error band ──────────────────────────────────────────
//
// Every Q and K vector the test builds has at most one nonzero element in each
// 32-dim quantization window, before and after rotation (a half-split pair
// `(f, f + 64)` spans windows `w` and `w + 2`). The kernel quantizes a window
// to `round(x · 127 / amax)`, which for a lone element is exactly ±127, and
// dequantizes it by `amax / 127`: the int8 path returns each element to within
// a few f32 roundings. What is left, per step:
//
// 1. **Logits.** Each product in `q'·k'` passes through the Q scale multiply,
//    the rotation (two products and an add per component), the quantization
//    scale, the dequantization, the accumulation and `softmax_scale`: at most a
//    dozen roundings of relative size `u = 2⁻²⁴`, each relative to the pair's
//    L1 norm rather than the rotated component (a rotation can cancel).
//    `δ_j ≤ LOGIT_ROUNDINGS · u · softmax_scale · Σ_d |q|₁,d · |k_j|₁,d`, with
//    `LOGIT_ROUNDINGS = 64` covering the dozen more than five times over.
// 2. **Softmax.** The kernel's `exp` is the cubic fast exp, documented at
//    0.009 % relative error; `EXP_REL_ERR = 1e-4` bounds it. A logit error of
//    `δ` and an exp error of `ε` move each weight's ratio `p'_j / p_j` within
//    `e^{±2(δ + atanh ε)}`, so `Σ_j |p'_j − p_j| ≤ e^{2Δ} − 1` with
//    `Δ = max_j δ_j + 1.01 ε`, and since `Σ p' = Σ p` the output moves by at
//    most `(e^{2Δ} − 1) · max_j |v_jd − o_d|`.
// 3. **Accumulation.** `n` fused multiply-adds into the output and the row
//    sum, the running-max rescales, the split merge and the final divide:
//    bounded by `8 (n + 2) u · max_j |v_jd|`.
// 4. **Emit.** The F16 store: half an ulp, `2⁻¹¹` relative, plus `2⁻²⁵`
//    absolute below the normal range.
//
// With this test's magnitudes the band is ~1e-3; `m` against `m²` moves the
// rotary logits by 14 %, which moves the output by well over ten times that.
// The test asserts the separation rather than assuming it.

const LOGIT_ROUNDINGS: f64 = 64.0;
const EXP_REL_ERR: f64 = 1e-4;
const F32_UNIT_ROUNDOFF: f64 = 1.0 / (1u64 << 24) as f64;
const F16_HALF_ULP_REL: f64 = 1.0 / (1u64 << 11) as f64;
const F16_SUBNORMAL_HALF_ULP: f64 = 1.0 / (1u64 << 25) as f64;

/// The scales the host reference applies to Q's rotary and pass-through pairs.
#[derive(Clone, Copy)]
struct QScale {
    rotary: f64,
    pass: f64,
}

const UNSCALED: QScale = QScale {
    rotary: 1.0,
    pass: 1.0,
};

/// `x` rotated to `pos` on `rung` with the half-split pairing `(f, f + hd/2)`,
/// through `rope.cos_sin` — the kernel's own table values — then scaled per
/// pair; and each element's rounding magnitude, its pair's L1 norm times the
/// pair's scale.
fn rotate(
    rope: &RopeRungs,
    rung: u32,
    pos: usize,
    x: &[f32],
    scale: QScale,
) -> (Vec<f64>, Vec<f64>) {
    let half = x.len() / 2;
    let mut rotated = vec![0f64; x.len()];
    let mut magnitude = vec![0f64; x.len()];
    for f in 0..half {
        let (c, s) = rope.cos_sin(rung, pos, f);
        let (c, s) = (c as f64, s as f64);
        let sc = if f < rope.pairs() {
            scale.rotary
        } else {
            scale.pass
        };
        let (lo, hi) = (x[f] as f64, x[f + half] as f64);
        rotated[f] = (lo * c - hi * s) * sc;
        rotated[f + half] = (hi * c + lo * s) * sc;
        let m = (lo.abs() + hi.abs()) * sc.abs();
        magnitude[f] = m;
        magnitude[f + half] = m;
    }
    (rotated, magnitude)
}

/// Whether every quantization window of `x` holds at most one nonzero element.
fn one_value_per_window(x: &[f64]) -> bool {
    x.chunks(QUANT_WINDOW)
        .all(|w| w.iter().filter(|v| **v != 0.0).count() <= 1)
}

/// The reference decode output and its error band, per element.
struct Reference {
    out: Vec<f64>,
    band: Vec<f64>,
}

/// The decode attention in f64: one query per head at the last position over
/// every key position `0..n`, K rotated at scale 1 and Q with `qs`, both on
/// `rung`. `q` is `[n_head, hd]`; `keys`/`values` are `[n, n_kv_head, hd]`.
fn reference_decode(
    g: Geometry,
    rope: &RopeRungs,
    rung: u32,
    qs: QScale,
    v_int8: bool,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
) -> Result<Reference> {
    let hd = g.head_dim;
    let n = keys.len() / (g.n_kv_head * hd);
    let q_pos = n - 1;
    let hpg = g.n_head / g.n_kv_head;
    let scale = g.softmax_scale() as f64;
    let mut out = vec![0f64; g.n_head * hd];
    let mut band = vec![0f64; g.n_head * hd];
    for h in 0..g.n_head {
        let kv = h / hpg;
        let (qr, qa) = rotate(rope, rung, q_pos, &q[h * hd..(h + 1) * hd], qs);
        if !one_value_per_window(&qr) {
            bail!("reference: query head {h} has two values in one quantization window");
        }
        let mut logits = Vec::with_capacity(n);
        let mut delta = 0f64;
        for j in 0..n {
            let base = (j * g.n_kv_head + kv) * hd;
            let (kr, ka) = rotate(rope, rung, j, &keys[base..base + hd], UNSCALED);
            if !one_value_per_window(&kr) {
                bail!("reference: key {j} head {kv} has two values in one quantization window");
            }
            logits.push(scale * qr.iter().zip(&kr).map(|(a, b)| a * b).sum::<f64>());
            let mag: f64 = qa.iter().zip(&ka).map(|(a, b)| a * b).sum();
            delta = delta.max(LOGIT_ROUNDINGS * F32_UNIT_ROUNDOFF * scale * mag);
        }
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = logits.iter().map(|s| (s - m).exp()).collect();
        let sum: f64 = w.iter().sum();
        let spread = (2.0 * (delta + 1.01 * EXP_REL_ERR)).exp_m1();
        for d in 0..hd {
            let v = |j: usize| values[(j * g.n_kv_head + kv) * hd + d] as f64;
            let o: f64 = (0..n).map(|j| w[j] * v(j)).sum::<f64>() / sum;
            let dev = (0..n).map(|j| (v(j) - o).abs()).fold(0f64, f64::max);
            let vmax = (0..n).map(|j| v(j).abs()).fold(0f64, f64::max);
            let mut b = spread * dev + 8.0 * (n as f64 + 2.0) * F32_UNIT_ROUNDOFF * vmax;
            if v_int8 {
                // The kernel quantises each V token to int8 under ONE scale,
                // `max_d |v| / 127`, so each element moves by at most half a step
                // and the output, a convex combination, by at most the largest
                // half step of any token.
                b += (0..n)
                    .map(|j| {
                        (0..hd)
                            .map(|e| values[(j * g.n_kv_head + kv) * hd + e].abs() as f64)
                            .fold(0f64, f64::max)
                    })
                    .fold(0f64, f64::max)
                    / 254.0;
            }
            out[h * hd + d] = o;
            band[h * hd + d] = b + F16_HALF_ULP_REL * (o.abs() + b) + F16_SUBNORMAL_HALF_ULP;
        }
    }
    Ok(Reference { out, band })
}

/// Every element of `got` lies within the reference's band.
fn assert_within(label: &str, got: &Tensor, reference: &Reference) -> Result<()> {
    let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if got.len() != reference.out.len() {
        bail!(
            "{label}: {} elements against {}",
            got.len(),
            reference.out.len()
        );
    }
    let mut worst = 0f64;
    let mut outside = 0usize;
    let mut first = usize::MAX;
    let mut last = 0usize;
    for (i, ((g, o), b)) in got.iter().zip(&reference.out).zip(&reference.band).enumerate() {
        let e = (*g as f64 - o).abs();
        // A NaN output is outside every band.
        if e.is_nan() || e > *b {
            outside += 1;
            first = first.min(i);
            last = i;
        }
        worst = worst.max(e / b);
    }
    if outside != 0 {
        bail!(
            "{label}: {outside} of {} elements outside the reference band, element {first} to \
             {last} (worst at {worst:.2}× the band)",
            got.len()
        );
    }
    eprintln!("{label}: within the reference band, worst element at {worst:.3}× the band");
    Ok(())
}

/// Some element of `alt` lies more than twice the band from the reference, so
/// no output inside the reference's band is inside the same band around `alt`.
fn assert_separated(label: &str, reference: &Reference, alt: &Reference) -> Result<()> {
    let sep = reference
        .out
        .iter()
        .zip(&alt.out)
        .zip(&reference.band)
        .map(|((r, a), b)| (r - a).abs() / (2.0 * b))
        .fold(0f64, f64::max);
    if sep.is_nan() || sep <= 1.0 {
        bail!(
            "{label} lies within twice the band of the reference (separation {sep:.3}): the \
             test cannot tell the two apart"
        );
    }
    eprintln!("{label}: separated from the reference by {sep:.1}× twice the band");
    Ok(())
}

/// The inputs of the `m²` test, F16-rounded, with their host copies.
struct Q4Inputs {
    /// `[1, n_head, hd]`: the rotary pair and the pass-through dim.
    q: Tensor,
    /// `[1, n_head, hd]`: the pass-through dim alone.
    q_pass: Tensor,
    /// `[Q4_HISTORY, n_head, hd]`: zeros — the history prefill's queries,
    /// whose output is not read.
    q_history: Tensor,
    k_history: Tensor,
    v_history: Tensor,
    /// `[1, n_kv_head, hd]`: the decoded token's K and V.
    k_new: Tensor,
    v_new: Tensor,
    q_host: Vec<f32>,
    q_pass_host: Vec<f32>,
    /// `[Q4_HISTORY + 1, n_kv_head, hd]`: history then the decoded token.
    keys_host: Vec<f32>,
    values_host: Vec<f32>,
}

fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// Q carries [`Q4_Q`]; each key carries a rotary pair in `[-4, 4)²` and a
/// pass-through value in `[-3, 3)`, nothing else; values are dense in
/// `[-1, 1)`.
fn q4_inputs(g: Geometry, device: &Device) -> Result<Q4Inputs> {
    let hd = g.head_dim;
    let half = hd / 2;
    let n = Q4_HISTORY + 1;
    let mut q = vec![0f32; g.n_head * hd];
    let mut q_pass = vec![0f32; g.n_head * hd];
    for (h, &(lo, hi, pass)) in Q4_Q.iter().enumerate().take(g.n_head) {
        q[h * hd + Q4_PAIR] = lo;
        q[h * hd + Q4_PAIR + half] = hi;
        q[h * hd + Q4_PASS] = pass;
        q_pass[h * hd + Q4_PASS] = pass;
    }
    let mut keys = vec![0f32; n * g.n_kv_head * hd];
    let mut values = vec![0f32; n * g.n_kv_head * hd];
    for j in 0..n {
        for kv in 0..g.n_kv_head {
            let base = (j * g.n_kv_head + kv) * hd;
            keys[base + Q4_PAIR] = 8.0 * pseudo(j, kv, 0, 0x0400_0001);
            keys[base + Q4_PAIR + half] = 8.0 * pseudo(j, kv, 1, 0x0400_0001);
            keys[base + Q4_PASS] = 6.0 * pseudo(j, kv, 2, 0x0400_0001);
            for d in 0..hd {
                values[base + d] = 2.0 * pseudo(j, kv, d, 0x0400_0002);
            }
        }
    }
    let q = Tensor::from_vec(q, (1, g.n_head, hd), device)?.to_dtype(DType::F16)?;
    let q_pass = Tensor::from_vec(q_pass, (1, g.n_head, hd), device)?.to_dtype(DType::F16)?;
    let keys = Tensor::from_vec(keys, (n, g.n_kv_head, hd), device)?.to_dtype(DType::F16)?;
    let values = Tensor::from_vec(values, (n, g.n_kv_head, hd), device)?.to_dtype(DType::F16)?;
    Ok(Q4Inputs {
        q_host: host_f32(&q)?,
        q_pass_host: host_f32(&q_pass)?,
        keys_host: host_f32(&keys)?,
        values_host: host_f32(&values)?,
        q_history: Tensor::zeros((Q4_HISTORY, g.n_head, hd), DType::F16, device)?,
        k_history: keys.narrow(0, 0, Q4_HISTORY)?.contiguous()?,
        v_history: values.narrow(0, 0, Q4_HISTORY)?.contiguous()?,
        k_new: keys.narrow(0, Q4_HISTORY, 1)?.contiguous()?,
        v_new: values.narrow(0, Q4_HISTORY, 1)?.contiguous()?,
        q,
        q_pass,
    })
}

/// One slot, one decoded query, on rung 2 of two schedules that differ only in
/// the temperature. With it, the kernel's output matches a host reference that
/// scales Q's rotary pairs by `m²` and nothing else, and is separated from
/// every other placement of the scale (`1`, `m`, `m⁴`, `m²` on every pair);
/// without it, the output matches the unscaled reference. With Q on its
/// pass-through dim alone, the two schedules agree bit for bit — the pass-through
/// pairs take no scale — and on rung 0, where `m² = 1`, so does the full query.
#[test]
fn q_scale_applies_to_q_rotary_pairs_only() -> Result<()> {
    let _serial = gpu_serial();
    let Some(device) = cuda_device() else {
        eprintln!("skipping: CUDA device required");
        return Ok(());
    };
    let stager = PinnedStager::new_from_device(&device);
    let g = HD128;
    let schedule = |temperature: bool| {
        RopeSchedule::yarn(
            Q4_ROPE_DIM,
            Q4_THETA,
            Q4_L0,
            vec![
                Rung {
                    ceiling: Q4_L0,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 2 * Q4_L0,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 8 * Q4_L0,
                    factor: Q4_FACTOR as f32,
                },
            ],
            temperature,
        )
    };
    let warm = RopeRungs::new(&schedule(true)?, &device)?;
    let cold = RopeRungs::new(&schedule(false)?, &device)?;

    // The two schedules differ in the temperature and nothing else.
    let m2 = warm.q_scale(Q4_RUNG);
    assert_eq!(m2, (mscale(Q4_FACTOR) * mscale(Q4_FACTOR)) as f32);
    assert!(m2 > 1.0);
    assert_eq!(warm.q_scale(0), 1.0);
    assert_eq!(cold.q_scale(Q4_RUNG), 1.0);
    assert_eq!(warm.pairs(), Q4_ROPE_DIM / 2);
    for pos in [0, 1, 17, Q4_HISTORY] {
        for f in 0..g.head_dim / 2 {
            assert_eq!(
                warm.cos_sin(Q4_RUNG, pos, f),
                cold.cos_sin(Q4_RUNG, pos, f),
                "rung {Q4_RUNG} table at position {pos}, pair {f}"
            );
        }
    }

    let inp = q4_inputs(g, &device)?;
    let backing = fresh_backing(g, &device)?;
    let mut cache = bind(&backing, 0)?;
    prefill(
        g,
        &mut [&mut cache],
        &inp.q_history,
        &inp.k_history,
        &inp.v_history,
        &[Q4_HISTORY],
        &warm,
        &stager,
    )?;
    // A fresh payload per decode: the kernel commits its write length into the
    // payload's slice table, so a reused one would read the previous decode's
    // token as history.
    let decode = |rope: &RopeRungs, rung: u32, q: &Tensor| {
        let slot = decode_slot(g, &backing, &cache, 0, &device)?;
        decode_launch(
            g,
            &[(&slot, rung)],
            q,
            &inp.k_new,
            &inp.v_new,
            rope,
            &stager,
        )
    };
    let reference = |rope: &RopeRungs, qs: QScale, q: &[f32]| {
        reference_decode(g, rope, Q4_RUNG, qs, false, q, &inp.keys_host, &inp.values_host)
    };

    // With the temperature: m² on the rotary pairs, nothing on the pass-through.
    let m2 = m2 as f64;
    let hot = decode(&warm, Q4_RUNG, &inp.q)?;
    let want = reference(
        &warm,
        QScale {
            rotary: m2,
            pass: 1.0,
        },
        &inp.q_host,
    )?;
    assert_within("temperature on, rung 2", &hot, &want)?;
    let m = m2.sqrt();
    for (label, qs) in [
        ("no scale", UNSCALED),
        (
            "m on the rotary pairs",
            QScale {
                rotary: m,
                pass: 1.0,
            },
        ),
        (
            "m⁴ on the rotary pairs",
            QScale {
                rotary: m2 * m2,
                pass: 1.0,
            },
        ),
        (
            "m² on every pair",
            QScale {
                rotary: m2,
                pass: m2,
            },
        ),
    ] {
        assert_separated(label, &want, &reference(&warm, qs, &inp.q_host)?)?;
    }

    // Without it: the unscaled reference, and a different output.
    let flat = decode(&cold, Q4_RUNG, &inp.q)?;
    assert_within(
        "temperature off, rung 2",
        &flat,
        &reference(&cold, UNSCALED, &inp.q_host)?,
    )?;
    if bits(&hot)? == bits(&flat)? {
        bail!("rung 2 decodes identically with and without the temperature");
    }

    // Q on its pass-through dim alone: the scale has nothing to act on, so the
    // two schedules agree to the bit — and both match the reference.
    let hot_pass = decode(&warm, Q4_RUNG, &inp.q_pass)?;
    let flat_pass = decode(&cold, Q4_RUNG, &inp.q_pass)?;
    assert_same_bits(
        "pass-through-only query, temperature on vs off",
        &hot_pass,
        &flat_pass,
    )?;
    assert_within(
        "pass-through-only query, temperature on",
        &hot_pass,
        &reference(
            &warm,
            QScale {
                rotary: m2,
                pass: 1.0,
            },
            &inp.q_pass_host,
        )?,
    )?;

    // Rung 0 carries m² = 1: the two schedules agree to the bit.
    assert_same_bits(
        "rung 0, temperature on vs off",
        &decode(&warm, 0, &inp.q)?,
        &decode(&cold, 0, &inp.q)?,
    )?;
    Ok(())
}
