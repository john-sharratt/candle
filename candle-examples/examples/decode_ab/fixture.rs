//! Synthetic decode-fixture construction.
//!
//! A [`Fixture`] is a fully-built paged-decode problem: a KV arena populated
//! by prefilling deterministic synthetic tokens at a chosen storage format,
//! plus the freshly-projected decode-step Q/K/V. [`Fixture::decode`] rebuilds
//! the per-slot `SlotHeader` array and launches one decode kernel backend.
//!
//! The construction mirrors `decode_one_slot` in
//! `candle-transformers/tests/kernel_layout_tests.rs` — the same host-side
//! metadata code the production scheduler uses — so any A/B divergence points
//! at the kernel, not the harness.

use std::time::Duration;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    quantize_sealed_in_place, ChunkedKvBacking, CompressionPolicy, KvCache, KvFormat, CHUNK_SIZE,
};
use candle::quantized::pinned_staging::PinnedBuf;
use candle_transformers::models::prefill_utils::{
    compute_rope_cs, paged_decode_attn_with_backend, paged_prefill_batched, DecodeBackend,
};
use candle_transformers::models::slot_state::{
    tensor_u8_device_ptr, SlotStateHost, TokenSliceHost,
};

use candle::quantized::pinned_staging::PinnedStager;

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
        device: &Device,
        stager: &PinnedStager,
    ) -> Result<Fixture> {
        let (qkv_dt, arena_dtype, force_dt) = resolve_dtypes(sc, fmt);
        let max_blocks = sc.kv_len().div_ceil(CHUNK_SIZE) + 2;
        let max_seq = max_blocks * CHUNK_SIZE;

        // RealQuant builds an adaptive F16-source backing whose compression
        // candidate arenas are pre-warmed, then quantizes the sealed sequence
        // after prefill (see below). Palette4 quant requires head_dim==128.
        let policy: Option<CompressionPolicy> = match fmt {
            ArenaFmt::RealQuant { level, override_fmt } => {
                if sc.head_dim != 128 {
                    candle::bail!("real-quant requires head_dim=128, got {}", sc.head_dim);
                }
                let mut p = CompressionPolicy::new(level);
                if let Some(qf) = override_fmt {
                    p = p.with_override_k_quant(Some(qf)).with_override_v_quant(Some(qf));
                }
                Some(p)
            }
            _ => None,
        };

        let backing = if let Some(pol) = &policy {
            ChunkedKvBacking::new_with_format_adaptive(
                sc.num_slots,
                sc.n_kv_head,
                sc.head_dim,
                KvFormat::Float(DType::F16),
                KvFormat::Float(DType::F16),
                device,
                max_seq,
                Some(pol.clone()),
            )?
        } else {
            match fmt.kv_format() {
                KvFormat::Float(dt) => ChunkedKvBacking::new(
                    sc.num_slots,
                    sc.n_kv_head,
                    sc.head_dim,
                    dt,
                    device,
                    max_seq,
                )?,
                kf @ KvFormat::Quantized(_) => ChunkedKvBacking::new_with_format(
                    sc.num_slots,
                    sc.n_kv_head,
                    sc.head_dim,
                    kf,
                    kf,
                    device,
                    max_seq,
                )?,
            }
        };

        // Non-zero RoPE so the kernel's rotary path is actually exercised.
        let inv_freq = make_inv_freq(sc.head_dim, device)?;
        let rope_cs = compute_rope_cs(&inv_freq, max_blocks, sc.head_dim, device)?;
        let rope_offsets = Tensor::zeros(1, DType::U32, device)?;

        let mut caches = Vec::with_capacity(sc.num_slots);
        for slot in 0..sc.num_slots {
            let mut cache = KvCache::new(2, max_seq);
            cache.force_dtype(force_dt);
            cache.set_chunked_backing(&backing, slot, None)?;
            let seed = 0x51A7_0000u64 ^ (slot as u64).wrapping_mul(0x9E37_79B9);
            let (q, k, v) = make_prefill_qkv(sc, sc.ctx_len, seed, device)?;
            run_prefill(
                &mut cache,
                &q,
                &k,
                &v,
                sc,
                &rope_cs,
                &rope_offsets,
                stager,
            )?;

            // RealQuant: convert the freshly-sealed R16 sequence into genuine
            // palette4-quantized chunks via the production quantize-on-evict
            // path, then re-inject so the decode reads quantized K/V. The
            // adaptive policy yields non-unity palette maps; an override policy
            // yields a uniform format with a unity palette.
            if let Some(pol) = &policy {
                let real_chunks = sc.ctx_len.div_ceil(CHUNK_SIZE).max(1);
                backing.truncate_sequence_to_blocks(slot, real_chunks)?;
                let r16 = backing.record_turn(slot, sc.ctx_len)?;
                let copy_stream = match device {
                    Device::Cuda(d) => d.cuda_stream(),
                    _ => candle::bail!("real-quant requires a CUDA device"),
                };
                let mut scratch: Option<PinnedBuf> = None;
                let warm =
                    quantize_sealed_in_place(&backing, &[&r16], pol, device, &copy_stream, &mut scratch)?;
                backing.truncate_sequence_to_blocks(slot, 0)?;
                backing.inject_sealed_at_tail(slot, &warm[0])?;
                cache.set_current_seq_len(sc.ctx_len)?;
            }

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

    /// Run a single decode step on the given backend. Rebuilds the per-slot
    /// `SlotHeader` array fresh each call, so the call is idempotent and both
    /// backends see identical pristine arena state. Returns the output and the
    /// device-synchronized kernel wall time.
    pub fn decode(
        &self,
        backend: DecodeBackend,
        device: &Device,
        stager: &PinnedStager,
    ) -> Result<(Tensor, Duration)> {
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

        let mut keepalive: Vec<Tensor> = Vec::new();
        let mut hdr_all: Vec<u8> = Vec::with_capacity(24 * sc.num_slots);

        for cache in self.caches.iter() {
            let chunks = cache
                .k_cache()
                .chunked_live_chunks_as_sealed()
                .unwrap_or_default();
            let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);

            let mut st = SlotStateHost::from_sealed_chunks(
                &chunks,
                sc.n_kv_head,
                sc.head_dim,
                &arena_info,
                writer_start,
            );
            st.extend_for_write_region(1, CHUNK_SIZE);

            let slice_size = TokenSliceHost::serialized_size(sc.n_kv_head, sc.head_dim);
            let mut sbuf = Vec::with_capacity(st.slices.len() * slice_size);
            for s in &st.slices {
                s.serialize_into(&mut sbuf);
            }
            let stensor = if sbuf.is_empty() {
                Tensor::zeros(1, DType::U8, device)?
            } else {
                Tensor::from_slice(&sbuf, sbuf.len(), device)?
            };
            let slices_ptr = tensor_u8_device_ptr(&stensor)?;

            let mut pm = st.position_map.clone();
            if pm.is_empty() {
                pm.push(0);
            }
            let pm_tensor = Tensor::from_slice(&pm, pm.len(), device)?;
            let pm_ptr = u32_tensor_device_ptr(&pm_tensor)?;

            hdr_all.extend_from_slice(&(st.slices.len() as u32).to_le_bytes());
            hdr_all.extend_from_slice(&st.write_slice.to_le_bytes());
            hdr_all.extend_from_slice(&slices_ptr.to_le_bytes());
            hdr_all.extend_from_slice(&pm_ptr.to_le_bytes());

            keepalive.push(stensor);
            keepalive.push(pm_tensor);
        }

        let generation = stager.begin_generation();
        let mut pinned = generation.alloc(hdr_all.len())?;
        pinned.copy_from_slice(&hdr_all);
        let headers_gpu = generation.submit(pinned)?;
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
        let start = cstream.record_event(Some(CU_EVENT_DEFAULT)).map_err(ev_err)?;
        let out = paged_decode_attn_with_backend(
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
            backend,
        )?;
        let stop = cstream.record_event(Some(CU_EVENT_DEFAULT)).map_err(ev_err)?;
        let ms = start.elapsed_ms(&stop).map_err(ev_err)?;
        let elapsed = Duration::from_secs_f64(ms as f64 / 1000.0);

        drop(headers_gpu);
        drop(keepalive);
        Ok((out, elapsed))
    }
}

/// RoPE inverse-frequency table for `head_dim` (theta = 10000), F32, shape
/// `(head_dim/2,)`. With `DECODE_AB_IDENTITY_ROPE` set, returns all-zeros so
/// RoPE is the identity — this lets the FP32 golden be plain attention (no RoPE
/// to replicate). The kernels still run their rotary path (cos=1, sin=0).
fn make_inv_freq(head_dim: usize, device: &Device) -> Result<Tensor> {
    let half = head_dim / 2;
    if std::env::var("DECODE_AB_IDENTITY_ROPE").is_ok() {
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
/// prefilled, assuming identity RoPE (so only valid under `DECODE_AB_IDENTITY_ROPE`).
/// K/V are F16-rounded to match the arena storage precision; for real-quant
/// arenas a kernel that reads K correctly lands within quant precision of this,
/// while a structural (e.g. palette) K-read bug diverges far more.
pub fn golden_decode(
    sc: &Scenario,
    device: &Device,
) -> Result<Tensor> {
    let (nq, nkv, hd) = (sc.n_q_head, sc.n_kv_head, sc.head_dim);
    let group = nq / nkv;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let to_f16_vec = |t: &Tensor| -> Result<Vec<f32>> {
        Ok(t.to_dtype(DType::F16)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    };
    // Decode-step Q/k_new/v_new (num_slots, n_head, hd), F16-rounded.
    let (q_d, k_d, v_d) = make_decode_qkv(sc, DType::F16, device)?;
    let qv = to_f16_vec(&q_d)?;
    let knv = to_f16_vec(&k_d)?;
    let vnv = to_f16_vec(&v_d)?;

    let mut out = vec![0f32; sc.num_slots * nq * hd];
    for slot in 0..sc.num_slots {
        let seed = 0x51A7_0000u64 ^ (slot as u64).wrapping_mul(0x9E37_79B9);
        let (_qp, kp, vp) = make_prefill_qkv(sc, sc.ctx_len, seed, device)?; // (1,nkv,ctx,hd)
        let kpv = to_f16_vec(&kp)?;
        let vpv = to_f16_vec(&vp)?;
        let ctx = sc.ctx_len;
        for h in 0..nq {
            let g = h / group;
            let qbase = slot * nq * hd + h * hd;
            // logits over ctx prefill tokens + 1 decode token
            let mut logits = vec![0f32; ctx + 1];
            for (t, lg) in logits.iter_mut().enumerate().take(ctx) {
                let kbase = (g * ctx + t) * hd;
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
                    acc += w * vpv[(g * ctx + t) * hd + d];
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
    let q = Tensor::from_vec(q, (1, n_tokens, sc.n_q_head, sc.head_dim), device)?
        .transpose(1, 2)?
        .contiguous()?;
    let k = Tensor::from_vec(k, (1, n_tokens, sc.n_kv_head, sc.head_dim), device)?
        .transpose(1, 2)?
        .contiguous()?;
    let v = Tensor::from_vec(v, (1, n_tokens, sc.n_kv_head, sc.head_dim), device)?
        .transpose(1, 2)?
        .contiguous()?;
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
fn run_prefill(
    cache: &mut KvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sc: &Scenario,
    rope_cs: &Tensor,
    rope_offsets: &Tensor,
    stager: &PinnedStager,
) -> Result<()> {
    let offset = cache.current_seq_len();
    let generation = stager.begin_generation();
    let mut caches_arr: [&mut KvCache; 1] = [cache];
    let _ = paged_prefill_batched(
        &mut caches_arr[..],
        &[offset],
        q,
        k,
        v,
        1,
        sc.ctx_len,
        sc.n_q_head,
        sc.n_kv_head,
        sc.head_dim,
        None,
        rope_offsets,
        rope_cs,
        sc.rope_interleaved,
        0,
        &generation,
    )?;
    caches_arr[0].set_current_seq_len(offset + sc.ctx_len)?;
    Ok(())
}

/// Extract the raw device pointer of a U32 tensor's storage.
fn u32_tensor_device_ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let cs = match &*storage {
        candle::Storage::Cuda(c) => c,
        _ => candle::bail!("expected CUDA storage for position_map"),
    };
    let stream = cs.device().cuda_stream();
    let s = cs.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);
    let (p, _g) = s.device_ptr(&stream);
    Ok(p)
}
