//! Replay + benchmark of a captured paged-prefill kernel call.
//!
//! Loads a bincode `PrefillCapture` fixture (produced by `ZEND_PREFILL_CAPTURE`
//! and trimmed to its largest slot via the `trim_prefill_fixture` example),
//! rebuilds the cached KV prefix into a fresh `ChunkedKvBacking`, and replays
//! the exact `paged_prefill_batched` call — then NaN/inf-checks the output and
//! benchmarks the kernel under a bounded wall-time budget (multiple runs to
//! average out thermal noise; total test < 2 s).
//!
//! CUDA-only; skips gracefully when no CUDA device or fixture is present.

#![cfg(feature = "cuda")]

use std::cell::RefCell;
use std::time::{Duration, Instant};

use candle::quantized::pinned_staging::PinnedStager;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{ChunkedKvBacking, HostSealedChunk, KvCache, KvFormat};
use candle_transformers::models::prefill_capture::PrefillCapture;
use candle_transformers::models::prefill_utils::paged_prefill_batched;

fn qkv_dtype(tag: u8) -> DType {
    match tag {
        1 => DType::F16,
        2 => DType::BF16,
        _ => DType::F32,
    }
}

/// The cached prefix: chunks that actually hold sealed tokens. The capture also
/// records the (empty) writer chunks `ensure_for_offsets` pre-allocated for the
/// new tokens; we drop those so `paged_prefill_batched` allocates + writes its
/// own writer region (matching the real forward), rather than re-injecting
/// empty chunks as sealed.
fn prefix_host_chunks(
    slot: &candle_transformers::models::prefill_capture::SlotCapture,
) -> Vec<HostSealedChunk> {
    slot.chunks
        .iter()
        .filter(|c| c.token_count > 0)
        .map(|c| HostSealedChunk {
            offset: c.offset,
            token_count: c.token_count,
            k_formats: c.k_formats.clone(),
            v_formats: c.v_formats.clone(),
            k_pal: c.k_pal.clone(),
            v_pal: c.v_pal.clone(),
            k_scale: c.k_scale.clone(),
            v_scale: c.v_scale.clone(),
            kv_bytes: c.kv_bytes.clone(),
        })
        .collect()
}

/// Rebuild every slot's cached prefix and return a `KvCache` per slot, each with
/// its logical length set to its prefix `offset` (the new q_len tokens are
/// written by the prefill itself). Each cache binds to a freshly
/// `alloc_sequence`'d slot; a distinct scratch slot is used to rebuild the
/// prefix (via `load_sealed_from_host` + `record_turn`, which is read-only) and
/// then `inject_sealed_at_tail` Arc-clones it into the cache slot with the
/// writer boundary set after it, so the prefill appends. The scratch is freed
/// (the prefix chunks survive via the Arc into the cache slot). Use
/// [`free_caches`] to release the cache slots afterward.
fn build_caches(
    backing: &ChunkedKvBacking,
    cap: &PrefillCapture,
    dtype: DType,
    device: &Device,
) -> Result<Vec<KvCache>> {
    let mut caches = Vec::with_capacity(cap.slots.len());
    for slot in &cap.slots {
        let cache_slot = backing.alloc_sequence()?;
        let scratch = backing.alloc_sequence()?;
        debug_assert_ne!(cache_slot, scratch);

        let host = prefix_host_chunks(slot);
        let sealed = backing.load_sealed_from_host(scratch, &host, device)?;

        let mut cache = KvCache::new(2, 64);
        cache.force_dtype(dtype);
        cache.set_chunked_backing(backing, cache_slot, None)?;
        backing.inject_sealed_at_tail(cache_slot, &sealed)?;
        cache.set_current_seq_len(slot.offset)?;

        backing.free_sequence(scratch)?;
        caches.push(cache);
    }
    Ok(caches)
}

/// Release the backing slots the caches were bound to (so the next rebuild
/// reuses them rather than growing the backing).
fn free_caches(backing: &ChunkedKvBacking, caches: &[KvCache]) -> Result<()> {
    for c in caches {
        if let Some(idx) = c.k_cache().chunked_batch_idx() {
            backing.free_sequence(idx)?;
        }
    }
    Ok(())
}

#[test]
fn prefill_replay_runs_and_benchmarks() -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping prefill_replay: CUDA device required");
            return Ok(());
        }
    };

    // ── Load the bincode PrefillCapture fixture ────────────────────────────
    let path = std::env::var("ZEND_PREFILL_REPLAY").unwrap_or_else(|_| {
        format!(
            "{}/tests/fixtures/prefill_fixture.bin",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping prefill_replay: no fixture at {path}: {e}");
            return Ok(());
        }
    };
    let cap: PrefillCapture =
        bincode::deserialize(&bytes).map_err(|e| candle::Error::Msg(format!("bincode: {e}")))?;

    let n_head = cap.n_head;
    let n_kv_head = cap.n_kv_head;
    let head_dim = cap.head_dim;
    let b_sz = cap.slots.len();
    assert!(b_sz > 0, "capture has no slots");
    let qkv_dt = qkv_dtype(cap.qkv_dtype_tag);

    // ── Build a ChunkedKvBacking matching the captured arena formats ───────
    // Inspect the captured per-sub-band format tags: float-only → a plain float
    // backing at that dtype; any quantized tag → adaptive backing with an F16
    // ceiling (load_sealed_from_host allocates each block per its own decoded
    // formats regardless, so the ceiling only warms the arenas).
    let mut any_quant = false;
    let mut float_ceiling = DType::F16;
    for slot in &cap.slots {
        for ch in &slot.chunks {
            for &t in ch.k_formats.iter().chain(ch.v_formats.iter()) {
                match KvFormat::from_tag(t) {
                    Some(KvFormat::Float(dt)) => {
                        if dt == DType::F32 {
                            float_ceiling = DType::F32;
                        } else if dt == DType::BF16 && float_ceiling == DType::F16 {
                            float_ceiling = DType::BF16;
                        }
                    }
                    Some(KvFormat::Quantized(_)) => any_quant = true,
                    None => candle::bail!("capture has unknown KV format tag {t}"),
                }
            }
        }
    }

    // Cover the largest prefix + new tokens, with headroom for the writer region
    // the prefill allocates. +1 batch slot for the scratch rebuild sequence.
    let max_seq: usize = cap
        .slots
        .iter()
        .map(|s| s.offset + s.q_len)
        .max()
        .unwrap_or(0)
        + 4096;
    let batch_cap = b_sz + 1;

    let backing = if any_quant {
        ChunkedKvBacking::new_with_format_adaptive(
            batch_cap,
            n_kv_head,
            head_dim,
            KvFormat::Float(DType::F16),
            KvFormat::Float(DType::F16),
            &device,
            max_seq,
            None,
        )?
    } else {
        ChunkedKvBacking::new(
            batch_cap,
            n_kv_head,
            head_dim,
            float_ceiling,
            &device,
            max_seq,
        )?
    };

    // ── Rebuild q/k/v + RoPE tensors ───────────────────────────────────────
    let total_q: usize = cap.slots.iter().map(|s| s.q_len).sum();
    let q = Tensor::from_vec(cap.q.clone(), (total_q, n_head, head_dim), &device)?
        .to_dtype(qkv_dt)?
        .contiguous()?;
    let k = Tensor::from_vec(cap.k.clone(), (total_q, n_kv_head, head_dim), &device)?
        .to_dtype(qkv_dt)?
        .contiguous()?;
    let v = Tensor::from_vec(cap.v.clone(), (total_q, n_kv_head, head_dim), &device)?
        .to_dtype(qkv_dt)?
        .contiguous()?;
    let rope_offsets = Tensor::from_vec(cap.rope_offsets.clone(), b_sz, &device)?;
    let rope_cs = Tensor::from_vec(cap.rope_cs.clone(), (cap.rope_cs_rows, head_dim), &device)?;

    let offsets: Vec<usize> = cap.slots.iter().map(|s| s.offset).collect();
    let q_lens: Vec<usize> = cap.slots.iter().map(|s| s.q_len).collect();

    let stager = PinnedStager::new_from_device(&device);
    let run = |caches: &mut [KvCache]| -> Result<Tensor> {
        let mut refs: Vec<&mut KvCache> = caches.iter_mut().collect();
        let generation = stager.begin_generation();
        paged_prefill_batched(
            None,
            &mut refs[..],
            &offsets,
            &q,
            &k,
            &v,
            b_sz,
            &q_lens,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_offsets,
            &rope_cs,
            cap.rope_interleaved,
            &generation,
            &RefCell::new(None),
        )
    };

    // ── Correctness: run once, assert all-finite ───────────────────────────
    let mut caches = build_caches(&backing, &cap, qkv_dt, &device)?;
    let out = run(&mut caches)?;
    let host = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert!(!host.is_empty(), "prefill produced an empty output");
    let bad = host.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        bad,
        0,
        "prefill output has {bad}/{} non-finite (NaN/inf) values",
        host.len()
    );
    free_caches(&backing, &caches)?;
    drop(caches);

    // ── Benchmark: warmups + timed runs, bounded to keep total < 2s ─────────
    // Each timed run replays a freshly-rebuilt prefill (the kernel writes new
    // KV, so the slot must start at its prefix offset every run); only the
    // kernel call is timed, with a stream sync on each side.
    let mut samples: Vec<f64> = Vec::new();
    let budget = Duration::from_millis(800);
    let start = Instant::now();
    let mut warmups = 2usize;
    while (samples.len() < 50 && start.elapsed() < budget) || warmups > 0 {
        let mut bench = build_caches(&backing, &cap, qkv_dt, &device)?;
        device.synchronize()?;
        let t0 = Instant::now();
        let _ = run(&mut bench)?;
        device.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if warmups > 0 {
            warmups -= 1;
        } else {
            samples.push(ms);
        }
        free_caches(&backing, &bench)?;
        drop(bench);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let min = samples.first().copied().unwrap_or(0.0);
    let median = samples.get(n / 2).copied().unwrap_or(0.0);
    let max = samples.last().copied().unwrap_or(0.0);
    eprintln!(
        "prefill_replay: b_sz={b_sz} total_q={total_q} max_prefix={} runs={n} \
         min={min:.3}ms median={median:.3}ms max={max:.3}ms",
        offsets.iter().max().copied().unwrap_or(0),
    );
    assert!(n > 0, "no benchmark samples collected");

    Ok(())
}
