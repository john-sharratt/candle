//! Synthetic decode-fixture construction.
//!
//! A [`Fixture`] is a fully-built paged-decode problem: a KV arena populated
//! by prefilling deterministic synthetic tokens at a chosen storage format,
//! plus the freshly-projected decode-step Q/K/V. [`Fixture::decode`] builds
//! the per-slot `SlotHeader` array, launches one decode step, and advances
//! every slot past the token it scattered.
//!
//! The slot-state the kernel walks is built by the production decode metadata
//! path (`ChunkedKvBacking::sync_decode_gpu_chunks_snapshot` on the live
//! device-resident slot, the same call `build_decode_metadata_at` makes), so
//! any A/B divergence points at the kernel, not the harness — and the timings
//! see the memory placement production sees.

use std::time::Duration;

use candle::quantized::pinned_staging::{PinnedBuf, PinnedStager};
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    quantize_sealed_in_place, ChunkedKvBacking, CompressionPolicy, KvCache, KvFormat, CHUNK_SIZE,
};
use candle_transformers::models::prefill_utils::{
    compute_rope_cs, paged_decode_attn, paged_prefill_batched,
};

use crate::formats::ArenaFmt;
use crate::scenarios::Scenario;

/// Resolve the (qkv, arena_dtype, force_dtype) triple for a (scenario, format).
///
/// `qkv` is the dtype of the decode-step Q/k_new/v_new and the kernel output.
/// `arena_dtype` is what `paged_decode_attn` is told the arena is (it selects
/// the typed kernel path). `force` is the cache's active-chunk write dtype.
fn resolve_dtypes(sc: &Scenario, fmt: ArenaFmt) -> (DType, DType, DType) {
    match fmt {
        ArenaFmt::Float(DType::F16) => (DType::F16, DType::F16, DType::F16),
        ArenaFmt::Float(DType::BF16) => (DType::BF16, DType::BF16, DType::BF16),
        // FP8 float arenas dispatch through the bf16 typed path in v2.
        ArenaFmt::Float(DType::F8E4M3) => (DType::BF16, DType::F8E4M3, DType::F8E4M3),
        ArenaFmt::Float(other) => (sc.compute, other, other),
        ArenaFmt::Quant(_) => (sc.compute, sc.compute, sc.compute),
        // RealQuant: source/active arena is F16, decode computes in F16.
        ArenaFmt::RealQuant { .. } => (DType::F16, DType::F16, DType::F16),
    }
}

/// Which rotary table the fixture is built with.
///
/// The golden gate compares the kernel against a plain FP32 attention that
/// does not replicate RoPE, so it builds with `Identity` (all-zero inverse
/// frequencies: cos = 1, sin = 0 — the kernel still runs its rotary path,
/// the rotation is just the identity). The bench builds with `Real` so the
/// rotary arithmetic is timed as production runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rope {
    Identity,
    Real,
}

/// A built, ready-to-run synthetic decode problem.
pub struct Fixture {
    pub scenario: Scenario,
    backing: ChunkedKvBacking,
    caches: Vec<KvCache>,
    q_dec: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    rope_cs: Tensor,
    arena_dtype: DType,
    softmax_scale: f32,
}

impl Fixture {
    /// Build the arena (prefill `ctx_len` synthetic tokens per slot) and the
    /// decode-step inputs. Returns an `Err` if the (scenario, format) pairing
    /// is unsupported (e.g. a format that cannot be a uniform sealing format);
    /// the caller treats that as a skip, not a fatal error.
    pub fn build(
        sc: &Scenario,
        fmt: ArenaFmt,
        rope: Rope,
        device: &Device,
        stager: &PinnedStager,
    ) -> Result<Fixture> {
        let (qkv_dt, arena_dtype, force_dt) = resolve_dtypes(sc, fmt);
        let max_blocks = sc.kv_len().div_ceil(CHUNK_SIZE) + 2;
        let max_seq = max_blocks * CHUNK_SIZE;

        // RealQuant builds an adaptive F16-source backing whose compression
        // candidate arenas are pre-warmed, then quantizes the sealed sequence
        // after prefill (see below).
        let policy: Option<CompressionPolicy> = match fmt {
            ArenaFmt::RealQuant {
                level,
                override_fmt,
            } => {
                let mut p = CompressionPolicy::new(level);
                if let Some(qf) = override_fmt {
                    p = p
                        .with_override_k_quant(Some(qf))
                        .with_override_v_quant(Some(qf));
                }
                Some(p)
            }
            _ => None,
        };

        // A segmented layout is built through one scratch slot past the
        // active ones: each segment is prefilled there, sealed, and injected
        // onto its slot.
        let n_backing_slots = sc.num_slots + usize::from(!sc.segments.is_empty());
        let backing = if let Some(pol) = &policy {
            ChunkedKvBacking::new_with_format_adaptive(
                n_backing_slots,
                sc.n_kv_head,
                sc.head_dim,
                KvFormat::Float(DType::F16),
                KvFormat::Float(DType::F16),
                device,
                max_seq,
                Some(*pol),
            )?
        } else {
            match fmt.kv_format() {
                KvFormat::Float(dt) => ChunkedKvBacking::new(
                    n_backing_slots,
                    sc.n_kv_head,
                    sc.head_dim,
                    dt,
                    device,
                    max_seq,
                )?,
                kf @ KvFormat::Quantized(_) => ChunkedKvBacking::new_with_format(
                    n_backing_slots,
                    sc.n_kv_head,
                    sc.head_dim,
                    kf,
                    kf,
                    device,
                    max_seq,
                )?,
            }
        };

        let inv_freq = make_inv_freq(sc.head_dim, rope, device)?;
        let rope_cs = compute_rope_cs(&inv_freq, max_blocks, sc.head_dim, device)?;
        let rope_offsets = Tensor::zeros(1, DType::U32, device)?;

        let mut caches = Vec::with_capacity(sc.num_slots);
        for slot in 0..sc.num_slots {
            let mut cache = KvCache::new(2, max_seq);
            cache.force_dtype(force_dt);
            cache.set_chunked_backing(&backing, slot, None)?;
            let seed = 0x51A7_0000u64 ^ (slot as u64).wrapping_mul(0x9E37_79B9);
            let (q, k, v) = make_prefill_qkv(sc, sc.ctx_len, seed, device)?;
            if sc.segments.is_empty() {
                // The prefill writes into the slot's write region, which the
                // scheduler allocates before the pass; the harness does the same.
                backing.ensure_for_batch_entries(&[(slot, 0)], sc.ctx_len)?;
                run_prefill(
                    &mut cache,
                    &q,
                    &k,
                    &v,
                    sc.ctx_len,
                    sc,
                    &rope_cs,
                    &rope_offsets,
                    stager,
                )?;
            } else {
                // Segment by segment through the scratch slot: prefill it
                // fresh, drop the empty writer chunk the prefill's decode
                // priming may have appended, seal, inject onto this slot. The
                // slot's chunk usages come out exactly as `segments` says, and
                // every segment after the first starts in a fresh chunk — the
                // projection's own mechanism. Stored K is unrotated, so the
                // scratch prefill's positions do not matter; the decode ropes
                // by the injected position.
                let scratch = sc.num_slots;
                let mut scratch_cache = KvCache::new(2, max_seq);
                scratch_cache.force_dtype(force_dt);
                scratch_cache.set_chunked_backing(&backing, scratch, None)?;
                let mut start = 0usize;
                for &len in sc.segments {
                    let qs = q.narrow(0, start, len)?.contiguous()?;
                    let ks = k.narrow(0, start, len)?.contiguous()?;
                    let vs = v.narrow(0, start, len)?.contiguous()?;
                    backing.truncate_sequence_to_blocks(scratch, 0)?;
                    scratch_cache.set_current_seq_len(0)?;
                    backing.ensure_for_batch_entries(&[(scratch, 0)], len)?;
                    run_prefill(
                        &mut scratch_cache,
                        &qs,
                        &ks,
                        &vs,
                        len,
                        sc,
                        &rope_cs,
                        &rope_offsets,
                        stager,
                    )?;
                    backing.truncate_sequence_to_blocks(scratch, len.div_ceil(CHUNK_SIZE))?;
                    let sealed = backing.record_turn(scratch)?;
                    backing.inject_sealed_at_tail(slot, &sealed)?;
                    start += len;
                    cache.set_current_seq_len(start)?;
                }
                assert_eq!(start, sc.ctx_len, "segments sum to ctx_len");
            }

            // RealQuant: convert the freshly-sealed R16 sequence into genuine
            // palette4-quantized chunks via the production quantize-on-evict
            // path, then re-inject so the decode reads quantized K/V. The
            // adaptive policy yields non-unity palette maps; an override policy
            // yields a uniform format with a unity palette.
            if let Some(pol) = &policy {
                // The chunks actually holding tokens — a holed layout has more
                // of them than `ctx_len / 32` — less the empty writer chunk the
                // prefill's decode priming may have appended past them.
                let real_chunks = if sc.segments.is_empty() {
                    sc.ctx_len.div_ceil(CHUNK_SIZE).max(1)
                } else {
                    backing.sequence_block_count(slot).unwrap_or(0).max(1)
                };
                backing.truncate_sequence_to_blocks(slot, real_chunks)?;
                let r16 = backing.record_turn(slot)?;
                let copy_stream = match device {
                    Device::Cuda(d) => d.cuda_stream(),
                    _ => candle::bail!("real-quant requires a CUDA device"),
                };
                let mut scratch: Option<PinnedBuf> = None;
                let warm = quantize_sealed_in_place(
                    &backing,
                    &[&r16],
                    pol,
                    device,
                    &copy_stream,
                    &mut scratch,
                )?;
                backing.truncate_sequence_to_blocks(slot, 0)?;
                backing.inject_sealed_at_tail(slot, &warm[0])?;
                cache.set_current_seq_len(sc.ctx_len)?;
            }

            // The persistent decode slot buffer's lengths self-increment only
            // on decode steps; the prefill's tokens sit past its stale tail
            // until the writer region is re-serialised — once, after the
            // prefill, as the engine's `refresh_decode_slot_state` does.
            backing.refresh_decode_writer_slice(&[(slot, cache.current_seq_len())])?;
            caches.push(cache);
        }

        let (q_dec, k_new, v_new) = make_decode_qkv(sc, qkv_dt, device)?;

        Ok(Fixture {
            scenario: sc.clone(),
            backing,
            caches,
            q_dec,
            k_new,
            v_new,
            rope_cs,
            arena_dtype,
            softmax_scale: 1.0f32 / (sc.head_dim as f32).sqrt(),
        })
    }

    /// Run one INT8 decode step at every slot's current position, then advance
    /// each slot by the token the kernel just scattered — the same two moves
    /// the production scheduler makes per token, so repeated calls walk the
    /// context forward one token at a time (through write-chunk boundaries
    /// included) instead of replaying a frozen state. Returns the output and
    /// the device-synchronized kernel wall time.
    pub fn decode(&mut self, device: &Device, stager: &PinnedStager) -> Result<(Tensor, Duration)> {
        let sc = &self.scenario;

        // Ensure every slot's writer chunk exists BEFORE resolving arena
        // pointers — matching the production scheduler order. Resolving first
        // and then growing a (quantized) arena per-slot inside the loop would
        // leave already-built headers pointing at stale, reallocated base
        // addresses → CUDA illegal access. This is why batched + quant decode
        // failed while float / single-slot did not.
        let entries: Vec<(usize, usize)> = self
            .caches
            .iter()
            .enumerate()
            .map(|(slot, c)| (slot, c.current_seq_len()))
            .collect();
        self.backing.ensure_for_batch_entries(&entries, 1)?;
        let arena_info = self.backing.resolve_arena_info()?;

        // The slot-state (slice headers + KvHead records) comes from the same
        // path production decode uses: `sync_decode_gpu_chunks_snapshot` with
        // an all-false snapshot mask is the LIVE path — each sequence's
        // serialised chunk state lives in its device-resident slot-state slot,
        // and the kernel's own `commit_decode_write_len_kernel` advances the
        // write chunk's length on device after every token. Memory placement is
        // part of what is being measured: the kernel opens on a dependent chain
        // of descriptor loads, so a zero-copy host mapping here would add a
        // PCIe round trip per link that production never pays.
        let generation = stager.begin_generation();
        let snapshot_mask = vec![false; entries.len()];
        let (seq_ptrs, _stats) = self.backing.sync_decode_gpu_chunks_snapshot(
            &entries,
            &arena_info,
            &generation,
            &snapshot_mask,
        )?;

        // 16-byte SlotHeader: n_slices, write_slice, slices_ptr — the kernel
        // derives every position from the slice walk.
        let mut hdr_all: Vec<u8> = Vec::with_capacity(16 * sc.num_slots);
        for &(ptr, n_slices, write_slice) in &seq_ptrs {
            hdr_all.extend_from_slice(&n_slices.to_le_bytes());
            hdr_all.extend_from_slice(&write_slice.to_le_bytes());
            hdr_all.extend_from_slice(&ptr.to_le_bytes());
        }

        // Device-resident, as `build_decode_metadata_at` submits them: every
        // block opens on its slot header.
        let mut pinned = generation.alloc(hdr_all.len())?;
        pinned.copy_from_slice(&hdr_all);
        let headers_gpu = generation.submit_resident(pinned)?;
        let headers_ptr = headers_gpu.dev_ptr();

        // Time the kernel with CUDA events on the device's (persistent) stream
        // — pure GPU kernel time, excluding the host-side launch/sync overhead
        // that `Instant` + `synchronize` would add equally to both backends and
        // thereby compress the speedup ratio. The kernel launches on this same
        // stream (candle's `cuda_stream()` returns the stored stream).
        use candle::cuda_backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let cstream = match device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => candle::bail!("decode timing requires a CUDA device"),
        };
        let ev_err = |e| candle::Error::Msg(format!("cuda event: {e:?}"));
        device.synchronize()?;
        let start = cstream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(ev_err)?;
        let out = paged_decode_attn(
            // This is a kernel A/B harness, not the wave path: it times one
            // decode in isolation, so there is no generation to carve from and
            // the output is an ordinary pool allocation.
            None,
            &self.q_dec,
            headers_ptr,
            self.arena_dtype,
            sc.n_q_head,
            sc.n_kv_head,
            sc.head_dim,
            self.softmax_scale,
            &self.k_new,
            &self.v_new,
            &self.rope_cs,
            sc.rope_interleaved,
            None,
        )?;
        let stop = cstream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(ev_err)?;
        let ms = start.elapsed_ms(&stop).map_err(ev_err)?;
        let elapsed = Duration::from_secs_f64(ms as f64 / 1000.0);
        drop(headers_gpu);

        // The kernel scattered this token's K/V into each slot's write chunk
        // and committed the new write length on device; the host-side offset
        // follows, so the next call decodes the token after it.
        for cache in &mut self.caches {
            let len = cache.current_seq_len();
            cache.set_current_seq_len(len + 1)?;
        }
        Ok((out, elapsed))
    }
}

/// RoPE inverse-frequency table for `head_dim` (theta = 10000), F32, shape
/// `(head_dim/2,)`. [`Rope::Identity`] is all-zeros, so the rotation is the
/// identity and the FP32 golden can be plain attention with no RoPE to
/// replicate; the kernels still run their rotary path (cos=1, sin=0).
fn make_inv_freq(head_dim: usize, rope: Rope, device: &Device) -> Result<Tensor> {
    let half = head_dim / 2;
    if rope == Rope::Identity {
        return Tensor::zeros(half, DType::F32, device);
    }
    let mut v = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2 * i) as f32 / head_dim as f32;
        v.push(1.0f32 / 10000f32.powf(exp));
    }
    Tensor::from_vec(v, half, device)
}

/// FP32 ground-truth decode attention over the *same* synthetic K/V the fixture
/// prefilled, assuming identity RoPE (so only valid against a [`Rope::Identity`] fixture).
/// K/V are F16-rounded to match the arena storage precision; for real-quant
/// arenas a kernel that reads K correctly lands within quant precision of this,
/// while a structural (e.g. palette) K-read bug diverges far more.
pub fn golden_decode(sc: &Scenario, device: &Device) -> Result<Tensor> {
    let (nq, nkv, hd) = (sc.n_q_head, sc.n_kv_head, sc.head_dim);
    let group = nq / nkv;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let to_f16_vec = |t: &Tensor| -> Result<Vec<f32>> {
        t.to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    // Decode-step Q/k_new/v_new (num_slots, n_head, hd), F16-rounded.
    let (q_d, k_d, v_d) = make_decode_qkv(sc, DType::F16, device)?;
    let qv = to_f16_vec(&q_d)?;
    let knv = to_f16_vec(&k_d)?;
    let vnv = to_f16_vec(&v_d)?;

    let mut out = vec![0f32; sc.num_slots * nq * hd];
    for slot in 0..sc.num_slots {
        let seed = 0x51A7_0000u64 ^ (slot as u64).wrapping_mul(0x9E37_79B9);
        let (_qp, kp, vp) = make_prefill_qkv(sc, sc.ctx_len, seed, device)?; // (ctx, nkv, hd)
        let kpv = to_f16_vec(&kp)?;
        let vpv = to_f16_vec(&vp)?;
        let ctx = sc.ctx_len;
        for h in 0..nq {
            let g = h / group;
            let qbase = slot * nq * hd + h * hd;
            // logits over ctx prefill tokens + 1 decode token
            let mut logits = vec![0f32; ctx + 1];
            for (t, lg) in logits.iter_mut().enumerate().take(ctx) {
                // token-major (ctx, nkv, hd): token t, kv-group g.
                let kbase = (t * nkv + g) * hd;
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += qv[qbase + d] * kpv[kbase + d];
                }
                *lg = dot * scale;
            }
            let knbase = (slot * nkv + g) * hd;
            let mut dot = 0f32;
            for d in 0..hd {
                dot += qv[qbase + d] * knv[knbase + d];
            }
            logits[ctx] = dot * scale;
            // softmax
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for lg in logits.iter_mut() {
                *lg = (*lg - m).exp();
                sum += *lg;
            }
            let inv = 1.0f32 / sum;
            // weighted sum of V
            let obase = slot * nq * hd + h * hd;
            for d in 0..hd {
                let mut acc = 0f32;
                for (t, &w) in logits.iter().enumerate().take(ctx) {
                    acc += w * vpv[(t * nkv + g) * hd + d];
                }
                acc += logits[ctx] * vnv[knbase + d];
                out[obase + d] = acc * inv;
            }
        }
    }
    Tensor::from_vec(out, (sc.num_slots, nq, hd), device)
}

/// Deterministic pseudo-random value in roughly [-0.5, 0.5].
fn pseudo(i: usize, j: usize, k: usize, seed: u64) -> f32 {
    let mut x = (i as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((k as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
        .wrapping_add(seed.wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let v = (x as i64 as f32) / (i64::MAX as f32);
    v * 0.5
}

/// Synthetic prefill Q/K/V, shape `(1, n_head, n_tokens, head_dim)` in F32 (the
/// cache converts to its `force_dtype` on write). Q uses `n_q_head`, K/V use
/// `n_kv_head`.
fn make_prefill_qkv(
    sc: &Scenario,
    n_tokens: usize,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut q = Vec::with_capacity(n_tokens * sc.n_q_head * sc.head_dim);
    let mut k = Vec::with_capacity(n_tokens * sc.n_kv_head * sc.head_dim);
    let mut v = Vec::with_capacity(n_tokens * sc.n_kv_head * sc.head_dim);
    for t in 0..n_tokens {
        for h in 0..sc.n_q_head {
            for d in 0..sc.head_dim {
                q.push(pseudo(t, h, d, seed ^ 0x111));
            }
        }
        for h in 0..sc.n_kv_head {
            for d in 0..sc.head_dim {
                k.push(pseudo(t, h, d, seed ^ 0x222));
                v.push(pseudo(t, h, d, seed ^ 0x333));
            }
        }
    }
    // Flat / ragged prefill layout: token-major (total_q, n_head, head_dim).
    // The vec was filled token-major (t, h, d), so it maps directly with no
    // transpose. `paged_prefill_batched` takes total_q = sum(q_lens) rows.
    let q = Tensor::from_vec(q, (n_tokens, sc.n_q_head, sc.head_dim), device)?;
    let k = Tensor::from_vec(k, (n_tokens, sc.n_kv_head, sc.head_dim), device)?;
    let v = Tensor::from_vec(v, (n_tokens, sc.n_kv_head, sc.head_dim), device)?;
    Ok((q, k, v))
}

/// Synthetic decode-step Q/k_new/v_new, shape `(num_slots, n_head, head_dim)`
/// in the compute dtype. Per-slot seeds keep the batch heterogeneous.
fn make_decode_qkv(
    sc: &Scenario,
    qkv_dt: DType,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut q = Vec::with_capacity(sc.num_slots * sc.n_q_head * sc.head_dim);
    let mut k = Vec::with_capacity(sc.num_slots * sc.n_kv_head * sc.head_dim);
    let mut v = Vec::with_capacity(sc.num_slots * sc.n_kv_head * sc.head_dim);
    for s in 0..sc.num_slots {
        let seed = 0xDEC0_0000u64 ^ (s as u64).wrapping_mul(0x1000_0001B3);
        for h in 0..sc.n_q_head {
            for d in 0..sc.head_dim {
                q.push(pseudo(s, h, d, seed ^ 0xA11));
            }
        }
        for h in 0..sc.n_kv_head {
            for d in 0..sc.head_dim {
                k.push(pseudo(s, h, d, seed ^ 0xB22));
                v.push(pseudo(s, h, d, seed ^ 0xC33));
            }
        }
    }
    let q = Tensor::from_vec(q, (sc.num_slots, sc.n_q_head, sc.head_dim), device)?
        .to_dtype(qkv_dt)?
        .contiguous()?;
    let k = Tensor::from_vec(k, (sc.num_slots, sc.n_kv_head, sc.head_dim), device)?
        .to_dtype(qkv_dt)?
        .contiguous()?;
    let v = Tensor::from_vec(v, (sc.num_slots, sc.n_kv_head, sc.head_dim), device)?
        .to_dtype(qkv_dt)?
        .contiguous()?;
    Ok((q, k, v))
}

/// Prefill `seq_len` tokens into a single slot's cache.
// Mirrors the kernel launch's own argument list.
#[allow(clippy::too_many_arguments)]
/// Prefill `n_tokens` rows of `q`/`k`/`v` onto `cache` at its current offset.
fn run_prefill(
    cache: &mut KvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_tokens: usize,
    sc: &Scenario,
    rope_cs: &Tensor,
    rope_offsets: &Tensor,
    stager: &PinnedStager,
) -> Result<()> {
    let offset = cache.current_seq_len();
    let generation = stager.begin_generation();
    let mut caches_arr: [&mut KvCache; 1] = [cache];
    let _ = paged_prefill_batched(
        None,
        &mut caches_arr[..],
        &[offset],
        q,
        k,
        v,
        1,
        &[n_tokens],
        sc.n_q_head,
        sc.n_kv_head,
        sc.head_dim,
        None,
        rope_offsets,
        rope_cs,
        sc.rope_interleaved,
        &generation,
        None,
    )?;
    caches_arr[0].set_current_seq_len(offset + n_tokens)?;
    Ok(())
}
