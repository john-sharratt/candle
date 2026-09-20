//! The fixture format of one captured paged-prefill kernel call — its packed
//! Q/K/V inputs, the cached KV chunks it attends, and the geometry/RoPE params —
//! which `tests/prefill_replay.rs` replays in isolation (kernel-optimization
//! work + a perf regression guard), and `examples/trim_prefill_fixture.rs`
//! trims to its largest slot.
//!
//! What is NOT captured (regenerated on replay, never round-tripped): GPU
//! pointers, slot headers, slices, the position_map, resident `meta` records.
//! `build_slot_headers` rebuilds those from the chunk state every call.

use serde::{Deserialize, Serialize};

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
    /// The RoPE `(cos, sin)` table the call rotated by, `[rope_cs_rows,
    /// head_dim]` flattened, frequency `i` at `2i`. Its row 1 is the rotation
    /// by one position, which is how a replay recovers the frequencies
    /// ([`Self::inv_freq`]).
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

    /// The RoPE frequencies the captured call rotated by, one per
    /// `head_dim / 2` pairs: the angle of row 1, the rotation by one position.
    /// A pass-through pair's `(1, 0)` recovers as frequency 0 — the identity at
    /// every position, as it was.
    pub fn inv_freq(&self) -> Vec<f32> {
        let row1 = &self.rope_cs[self.head_dim..2 * self.head_dim];
        (0..self.head_dim / 2)
            .map(|i| row1[2 * i + 1].atan2(row1[2 * i]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row 1's angle is each pair's frequency, and a pass-through pair's is 0.
    #[test]
    fn frequencies_come_back_from_row_one() {
        let head_dim = 4;
        let w = [0.5f32, 0.0];
        let mut rope_cs = vec![0f32; 3 * head_dim];
        for pos in 0..3 {
            for (i, &f) in w.iter().enumerate() {
                let a = pos as f32 * f;
                rope_cs[pos * head_dim + 2 * i] = a.cos();
                rope_cs[pos * head_dim + 2 * i + 1] = a.sin();
            }
        }
        let cap = PrefillCapture {
            n_head: 1,
            n_kv_head: 1,
            head_dim,
            rope_interleaved: false,
            qkv_dtype_tag: 3,
            q: vec![],
            k: vec![],
            v: vec![],
            rope_offsets: vec![],
            rope_cs,
            rope_cs_rows: 3,
            slots: vec![],
        };
        let got = cap.inv_freq();
        assert!((got[0] - 0.5).abs() < 1e-6, "{got:?}");
        assert_eq!(got[1], 0.0);
    }
}
