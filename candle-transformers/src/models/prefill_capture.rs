//! Capture a single paged-prefill kernel call — its packed Q/K/V inputs, the
//! cached KV chunks it attends, and the geometry/RoPE params — to a binary
//! fixture, so the attention kernel can be replayed in isolation in a unit test
//! (kernel-optimization work + a perf regression guard).
//!
//! Entirely gated behind `ZEND_PREFILL_CAPTURE=<path>`: a no-op unless that env
//! var is set, and then it dumps exactly ONE call (the first whose summed
//! `kv_len` exceeds `ZEND_PREFILL_CAPTURE_MIN_KV`, default 20000) per process.
//! CUDA-only — the kernel and the KV gather are CUDA paths.
//!
//! What is NOT captured (regenerated on replay, never round-tripped): GPU
//! pointers, slot headers, slices, the position_map, resident `meta` records.
//! `build_slot_headers` rebuilds those from the chunk state every call.

#[cfg(feature = "cuda")]
use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::KvCache;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};

/// One sealed chunk's portable host data (mirror of `candle_nn`'s
/// `HostSealedChunk`, with serde). `kv_bytes` is the raw (possibly quantized)
/// arena data, un-rotated; `k_formats`/`v_formats` are `KvFormat::to_tag()`.
#[derive(Clone, Serialize, Deserialize)]
pub struct ChunkCapture {
    pub offset: u16,
    pub token_count: u16,
    pub k_formats: Vec<u8>,
    pub v_formats: Vec<u8>,
    pub k_pal: Vec<u8>,
    pub v_pal: Vec<u8>,
    pub k_scale: Vec<f32>,
    pub v_scale: Vec<f32>,
    pub kv_bytes: Vec<u8>,
}

/// One sequence/slot in the batch: its cached prefix (sealed chunks) plus the
/// host-side geometry needed to rebuild it and drive the kernel.
#[derive(Clone, Serialize, Deserialize)]
pub struct SlotCapture {
    /// Cached prefix length (tokens already in the slot before this prefill).
    pub offset: usize,
    /// New (query) tokens this prefill writes for the slot.
    pub q_len: usize,
    /// Sealed prefix chunks (the cached KV the new tokens attend).
    pub chunks: Vec<ChunkCapture>,
}

/// A full single-layer paged-prefill call, replayable into `paged_prefill_batched`.
#[derive(Clone, Serialize, Deserialize)]
pub struct PrefillCapture {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rope_interleaved: bool,
    /// dtype of the packed Q/K/V: 1 = F16, 2 = BF16, 3 = F32.
    pub qkv_dtype_tag: u8,
    /// Packed `[total_q, n_head, head_dim]`, flattened to f32 (re-cast on load).
    pub q: Vec<f32>,
    /// Packed `[total_q, n_kv_head, head_dim]`, flattened to f32.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Per-sequence RoPE base position, `[b_sz]`.
    pub rope_offsets: Vec<u32>,
    /// RoPE cos/sin table `[rope_cs_rows, head_dim]`, flattened f32.
    pub rope_cs: Vec<f32>,
    pub rope_cs_rows: usize,
    /// One entry per sequence/slot, in batch order.
    pub slots: Vec<SlotCapture>,
}

impl PrefillCapture {
    /// Reduce the capture to the single slot with the largest cached prefix,
    /// slicing the packed Q/K/V down to that slot's token range. Produces a
    /// small, committable fixture that still exercises the kernel's dominant
    /// cost (per-token attention over a long prefix), at the cost of the
    /// batch-width dimension. Pure host data manipulation — no device needed.
    pub fn keep_largest_slot(&self) -> PrefillCapture {
        let idx = self
            .slots
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.offset)
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Exclusive prefix-sum of q_len gives this slot's row range in the
        // flat-packed Q/K/V (cu_seqlens order).
        let start: usize = self.slots[..idx].iter().map(|s| s.q_len).sum();
        let q_len = self.slots[idx].q_len;
        let q_row = self.n_head * self.head_dim;
        let kv_row = self.n_kv_head * self.head_dim;
        PrefillCapture {
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
            rope_interleaved: self.rope_interleaved,
            qkv_dtype_tag: self.qkv_dtype_tag,
            q: self.q[start * q_row..(start + q_len) * q_row].to_vec(),
            k: self.k[start * kv_row..(start + q_len) * kv_row].to_vec(),
            v: self.v[start * kv_row..(start + q_len) * kv_row].to_vec(),
            rope_offsets: vec![self.rope_offsets[idx]],
            rope_cs: self.rope_cs.clone(),
            rope_cs_rows: self.rope_cs_rows,
            slots: vec![self.slots[idx].clone()],
        }
    }
}

#[cfg(feature = "cuda")]
static CAPTURED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "cuda")]
fn dtype_tag(dt: DType) -> u8 {
    match dt {
        DType::F16 => 1,
        DType::BF16 => 2,
        _ => 3,
    }
}

#[cfg(feature = "cuda")]
fn tensor_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// If `ZEND_PREFILL_CAPTURE=<path>` is set and this call's summed `kv_len`
/// exceeds the threshold, serialize it to `<path>` (once per process) via
/// bincode. No-op otherwise — cheap env check on the hot path when disabled.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn maybe_capture(
    caches: &[&mut KvCache],
    offsets: &[usize],
    q_packed: &Tensor,
    k_packed: &Tensor,
    v_packed: &Tensor,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
) {
    let path = match std::env::var("ZEND_PREFILL_CAPTURE") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    if CAPTURED.load(Ordering::Relaxed) {
        return;
    }
    let min_kv: usize = std::env::var("ZEND_PREFILL_CAPTURE_MIN_KV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let sum_kv: usize = offsets
        .iter()
        .zip(q_lens.iter())
        .map(|(&o, &l)| o + l)
        .sum();
    if sum_kv < min_kv {
        return;
    }
    // Claim the single capture slot; only the first winner past the threshold
    // proceeds (the per-layer hook fires many times — we want one layer).
    if CAPTURED.swap(true, Ordering::SeqCst) {
        return;
    }

    match build_and_write(
        &path,
        caches,
        offsets,
        q_packed,
        k_packed,
        v_packed,
        q_lens,
        n_head,
        n_kv_head,
        head_dim,
        rope_offsets,
        rope_cs,
        rope_interleaved,
    ) {
        Ok(bytes) => tracing::info!(
            target: "candle_transformers::prefill_capture",
            path = %path,
            sum_kv,
            seqs = caches.len(),
            bytes,
            "captured prefill kernel fixture"
        ),
        Err(e) => tracing::warn!(
            target: "candle_transformers::prefill_capture",
            "prefill capture failed: {e}"
        ),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn build_and_write(
    path: &str,
    caches: &[&mut KvCache],
    offsets: &[usize],
    q_packed: &Tensor,
    k_packed: &Tensor,
    v_packed: &Tensor,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
) -> Result<usize> {
    let device: &Device = q_packed.device();
    let rope_cs_rows = rope_cs.dim(0)?;

    let mut slots = Vec::with_capacity(caches.len());
    for (i, cache) in caches.iter().enumerate() {
        let host = match cache.k_cache().chunked_dump_sealed_to_host(device) {
            Some(r) => r?,
            None => Vec::new(),
        };
        let chunks = host
            .into_iter()
            .map(|h| ChunkCapture {
                offset: h.offset,
                token_count: h.token_count,
                k_formats: h.k_formats,
                v_formats: h.v_formats,
                k_pal: h.k_pal,
                v_pal: h.v_pal,
                k_scale: h.k_scale,
                v_scale: h.v_scale,
                kv_bytes: h.kv_bytes,
            })
            .collect();
        slots.push(SlotCapture {
            offset: offsets[i],
            q_len: q_lens[i],
            chunks,
        });
    }

    let cap = PrefillCapture {
        n_head,
        n_kv_head,
        head_dim,
        rope_interleaved,
        qkv_dtype_tag: dtype_tag(q_packed.dtype()),
        q: tensor_f32(q_packed)?,
        k: tensor_f32(k_packed)?,
        v: tensor_f32(v_packed)?,
        rope_offsets: rope_offsets
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?,
        rope_cs: tensor_f32(rope_cs)?,
        rope_cs_rows,
        slots,
    };

    let bytes =
        bincode::serialize(&cap).map_err(|e| candle::Error::Msg(format!("bincode: {e}")))?;
    std::fs::write(path, &bytes).map_err(|e| candle::Error::Msg(format!("write {path}: {e}")))?;
    Ok(bytes.len())
}
