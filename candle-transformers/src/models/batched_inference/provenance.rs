//! Gathers and packs the per-token provenance data that retrieval scores the
//! paged cache by: raw R16 chunk contents and the sign bits of their queries.
//!
//! Sign bits are packed on the device, so only the packed words come back to the
//! host. The band count is a property of how the arena stores each head, and a
//! derived count would only disagree with the storage. A band wider than the
//! packing word declines rather than truncating. Every provenance-bearing layer
//! must present the same block set, and a speculative head's layer is left out
//! because its history is its own. Including it would make the gather fail on
//! every turn, and retrieval would fall back to recency without saying so.

use super::*;

/// One R16 chunk's unpacked contents: `(block_idx, k_flat, v_flat, q_flat)`.
///
/// Each flat vec holds `n_kv_head × N_PALETTE × CHUNK_SIZE × sub_head_dim`
/// values in `[head][palette][token][sub_dim]` order. `block_idx` is absolute,
/// so a caller can place the chunk even though non-R16 chunks leave gaps.
pub type R16ChunkDump = (usize, Vec<f32>, Vec<f32>, Vec<f32>);

/// GPU-packed wide-Q provenance sign bits for a scope, produced by
/// [`BatchedInferenceSession::gather_provenance_sign_packed`] and consumed by the
/// scheduler's `gather_wide_sigs` (assemble raw `WideQSig` → fold).
pub struct ProvSignPacked {
    /// Warp-major packed sign bits — `packed[warp * CHUNK_SIZE + token]`, bit `d`
    /// set iff Q dim `d` of that sub-band is `>= 0` (`d` in `0..sub_head_dim`).
    /// Warp index = `((layer*n_blocks + block_pos)*n_kv_head + head)*n_palette + palette`,
    /// where `block_pos` indexes [`Self::block_indices`].
    pub packed: Vec<u64>,
    /// Absolute chunk indices captured (identical across all layers).
    pub block_indices: Vec<usize>,
    pub n_layers: usize,
    pub n_kv_head: usize,
    pub n_palette: usize,
    /// `head_dim / n_palette` — bits packed per palette sub-band (`<= 64`, the
    /// width of a physical R16 band at `head_dim` 256).
    pub sub_head_dim: usize,
}

/// A scope's provenance sources, resolved by
/// [`BatchedInferenceSession::resolve_provenance_sources`] and packed by
/// [`BatchedInferenceSession::pack_provenance_signs`].
pub struct ProvSources {
    /// Per provenance-bearing layer, each block's source, in block order.
    pub layers: Vec<Vec<ProvSource>>,
    /// Absolute chunk indices resolved (identical across all layers).
    pub blocks: Vec<usize>,
}

/// A scope's packed signs in [`ProvSignPacked`] order — layer-major, then
/// block, each block `block_words` words — taking a [`ProvSource::Q`] block's
/// warps from `read` (the sign-pack's output over every `Q` block's pointers,
/// in the same order) and a [`ProvSource::Signed`] block's from its captured
/// bits. `None` when `read` is not exactly the `Q` blocks' warps or a block
/// is not `block_words` long.
pub fn lay_out_prov_signs(
    layers: &[Vec<ProvSource>],
    read: &[u64],
    block_words: usize,
) -> Option<Vec<u64>> {
    if block_words == 0 {
        return None;
    }
    let blocks = layers.first().map_or(0, Vec::len);
    let mut packed = Vec::with_capacity(layers.len() * blocks * block_words);
    let mut from_kernel = read.chunks_exact(block_words);
    if !from_kernel.remainder().is_empty() {
        return None;
    }
    for source in layers.iter().flatten() {
        let words: &[u64] = match source {
            ProvSource::Q(ptrs) if ptrs.len() * CHUNK_SIZE == block_words => from_kernel.next()?,
            ProvSource::Signed(signs) if signs.len() == block_words => signs,
            _ => return None,
        };
        packed.extend_from_slice(words);
    }
    from_kernel.next().is_none().then_some(packed)
}

impl BatchedInferenceSession {
    /// Dump raw K, V, and Q float data for R16 chunks of a sequence.
    ///
    /// Returns `(block_idx, k_flat_f32, v_flat_f32, q_flat_f32)` for every
    /// sealed KV block in `seq_idx` whose K arena is R16.  Blocks stored in
    /// other formats (float, quantized non-R16) are silently skipped.
    ///
    /// Layout of each flat vec: `[head][palette][token][sub_dim]` —
    /// `n_kv_head × N_PALETTE × CHUNK_SIZE × sub_head_dim` values.
    ///
    /// Returns an empty vec when R16 is not in use or the sequence has no
    /// sealed blocks.  `layer_idx` selects which layer's backing to read;
    /// layer 0 is typical for provenance indexing.
    /// Dump R16 KV chunk data for the requested block range.
    ///
    /// Returns one `(block_idx, k_flat, v_flat, q_flat)` per R16 chunk
    /// whose K side resides in an R16 quantized arena.  Non-R16 chunks
    /// are skipped (creating gaps); each tuple carries the absolute
    /// block index so callers can reconstruct the position.
    ///
    /// `block_range = None` walks every chunk in the sequence (legacy
    /// full-sequence semantics — used by turn-end signature extraction
    /// and KV-data debug dumps); `Some((lo, hi))` walks only chunks
    /// in `[lo, hi)`, which is what the scheduler's mid-decode
    /// re-projection wants (probe window is at most a handful of
    /// chunks regardless of context depth).
    pub fn dump_r16_kv_for_provenance(
        &self,
        seq_idx: usize,
        layer_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Vec<R16ChunkDump>> {
        match self.backings.get(layer_idx) {
            Some(backing) => backing.dump_sequence_r16_kv_chunks(seq_idx, block_range),
            None => Ok(vec![]),
        }
    }

    /// Fast CUDA gather for provenance: single kernel + one DtoH copy.
    ///
    /// Replaces `dump_r16_kv_for_provenance` in the mid-decode reproject path,
    /// where N synchronous `memcpy_dtov` calls per `\n` token caused ~2 t/s decode.
    /// Falls back to the slow path on CPU or if V is not float-F16.
    #[cfg(feature = "cuda")]
    pub fn gather_r16_kv_for_provenance(
        &self,
        seq_idx: usize,
        layer_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Vec<R16ChunkDump>> {
        match self.backings.get(layer_idx) {
            Some(backing) => backing.gather_r16_kv_probe(seq_idx, block_range),
            None => Ok(vec![]),
        }
    }

    /// Gather R16 KV data for multiple provenance layers in one call.
    ///
    /// Calls `gather_r16_kv_probe` for each entry in `layer_indices` and
    /// returns results in the same order.  Exists so callers can retrieve
    /// all three provenance layers (syn, sem, prag) through a single API
    /// boundary rather than three separate calls.
    ///
    /// Each layer's probe is sequential (HtoD → kernel → DtoH per layer).
    /// Falls back to the slow path on CPU.  Returns one result vec per entry
    /// in `layer_indices`.
    #[cfg(feature = "cuda")]
    pub fn gather_r16_kv_provenance_layers(
        &self,
        seq_idx: usize,
        layer_indices: &[usize],
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Vec<Vec<R16ChunkDump>>> {
        layer_indices
            .iter()
            .map(|&layer_idx| match self.backings.get(layer_idx) {
                Some(backing) => backing.gather_r16_kv_probe(seq_idx, block_range),
                None => Ok(vec![]),
            })
            .collect()
    }

    /// GPU-side wide-Q provenance capture: one kernel launch across ALL layers
    /// reads the R16 Q, signs it, and bit-packs it — only the packed bits come
    /// back to the host. Replaces the per-layer f16 K/Q/V D2H + CPU sign pass
    /// (~48 blocking round-trips per scope) with one HtoD (pointers), one launch
    /// and one DtoH (a few KB). The caller (`gather_wide_sigs`) assembles the raw
    /// per-token `WideQSig` from `packed` and folds it — bit-identical to the CPU
    /// path. Returns `None` (caller falls back to the CPU gather) when not CUDA,
    /// when any layer has no R16 blocks, when the layers' block sets disagree, or
    /// when `sub_head_dim > 64` (can't pack into a u64).
    pub fn gather_provenance_sign_packed(
        &self,
        seq_idx: usize,
        block_range: Option<(usize, usize)>,
        n_layers: usize,
    ) -> candle::Result<Option<ProvSignPacked>> {
        let Some(sources) = self.resolve_provenance_sources(seq_idx, block_range, n_layers)? else {
            return Ok(None);
        };
        self.pack_provenance_signs(&sources)
    }

    /// Sign-pack a resolved scope: one kernel launch over every R16 block's Q
    /// across all its layers, then each block's warps laid into the
    /// warp-major [`ProvSignPacked`] order — the kernel's for a block read from
    /// Q, the gap's captured bits for a block it converted. `None` when the
    /// kernel declines (non-CUDA, unsupported width) or a captured block's
    /// bits are not this geometry's size.
    pub fn pack_provenance_signs(
        &self,
        sources: &ProvSources,
    ) -> candle::Result<Option<ProvSignPacked>> {
        let sub_head_dim = self.prov_sub_head_dim();
        let n_kv_head = self.n_kv_head();
        let n_palette = self.prov_n_palette();
        let block_words = n_kv_head * n_palette * CHUNK_SIZE;
        if sub_head_dim == 0 || block_words == 0 {
            return Ok(None);
        }
        let all_ptrs: Vec<i64> = sources
            .layers
            .iter()
            .flatten()
            .filter_map(|s| match s {
                ProvSource::Q(ptrs) => Some(ptrs.as_slice()),
                ProvSource::Signed(_) => None,
            })
            .flatten()
            .copied()
            .collect();
        let read = if all_ptrs.is_empty() {
            Vec::new()
        } else {
            self.run_prov_sign_pack(&all_ptrs, sub_head_dim)?
        };
        let Some(packed) = lay_out_prov_signs(&sources.layers, &read, block_words) else {
            return Ok(None);
        };
        Ok(Some(ProvSignPacked {
            packed,
            block_indices: sources.blocks.clone(),
            n_layers: sources.layers.len(),
            n_kv_head,
            // Derived from the same rule as `sub_head_dim` above, and it has to
            // be: the two travel together in this struct, and a consumer that
            // reconstructs `head_dim = n_palette * sub_head_dim` from a derived
            // width beside a constant band count reads 128 for a 256-wide head.
            n_palette,
            sub_head_dim,
        }))
    }

    /// How many palette bands a head is **physically stored in** — the number
    /// of Q pointers a [`ProvSource::Q`] carries per head.
    ///
    /// **This is a property of the arena, not a free choice.** The R16 chunk
    /// lays each head out in `N_PALETTE` bands of `head_dim / N_PALETTE` dims, the
    /// pointer resolution walks exactly that many bands, and the sign-pack
    /// kernel's contract is `sub_head_dim = head_dim / N_PALETTE`. Deriving a
    /// different band count here does not re-band the storage — it only makes
    /// this side disagree with it.
    ///
    /// It was briefly derived as "the smallest power of two with
    /// `head_dim / p <= 32`" to stop the fast path declining at `head_dim`
    /// 256. That reasoning was right about the symptom and wrong about the
    /// fix: at 256 it yields 8 bands of 32, so the kernel reads only the low
    /// 32 dims of each physical 64-dim band and the host lays band `p` at
    /// `p * 32` instead of `p * 64` — half of every signature dropped, the
    /// rest dim-permuted, and no longer bit-identical to the CPU fold it is
    /// documented to match.
    ///
    /// The band count was never the thing to change. What declined at 256 was
    /// the kernel's **word width**: it packed a band into a `u32`, and the
    /// physical band there is 64 dims. It packs a `u64` now, so a 256-wide head
    /// takes the fast path with its real banding intact.
    pub fn prov_n_palette(&self) -> usize {
        candle_nn::kv_cache::N_PALETTE
    }

    /// The provenance sign-pack sub-band width (`head_dim / n_palette`), or 0
    /// when the geometry cannot be packed into one word — callers treat 0 as
    /// "use the CPU path".
    ///
    /// The bound is 64, the kernel's word width and the width of a physical R16
    /// band at `head_dim` 256 — so both production geometries (128 → 32-dim
    /// bands, 256 → 64-dim bands) take the GPU path. It was 32, which declined
    /// at 256 and sent every seal of the hybrid through a full R16 device→host
    /// copy; the signatures were correct, so nothing but a path assertion could
    /// see it (`hybrid_capture_takes_the_gpu_sign_pack_path`).
    ///
    /// A wider band than 64 still declines rather than silently truncating: the
    /// band is the arena's, not this function's, and narrowing it here would
    /// drop dims (see [`Self::prov_n_palette`]).
    pub fn prov_sub_head_dim(&self) -> usize {
        let head_dim = self.head_dim();
        if head_dim == 0 {
            return 0;
        }
        let sub = head_dim / self.prov_n_palette();
        if sub == 0 || sub > candle_kernels::simple::prov_sign_pack::MAX_SUB_HEAD_DIM {
            0
        } else {
            sub
        }
    }

    /// Resolve where every provenance-bearing layer's blocks take their
    /// `sign(Q)` from — R16 Q addresses or the gap's captured bits — plus the
    /// block index set, identical across layers. Split from the launch so the
    /// caller times the two apart. `None` (→ CPU path) when not CUDA, when any
    /// layer has no block with Q or signs, when the layers' block sets
    /// disagree, or when the geometry cannot be packed into one word.
    pub fn resolve_provenance_sources(
        &self,
        seq_idx: usize,
        block_range: Option<(usize, usize)>,
        n_layers: usize,
    ) -> candle::Result<Option<ProvSources>> {
        if self.backings.is_empty() || self.n_kv_head() == 0 || self.prov_sub_head_dim() == 0 {
            return Ok(None);
        }
        // **The provenance-bearing layers, which are not always all of them.**
        //
        // `n_layers` comes from the model's `provenance_capture_layers`, and it
        // has to be the same number the fold was derived from: the fold's
        // geometry is `layers × kv_heads`, so a gather over one more layer than
        // the fold expects does not produce a slower answer, it produces a
        // different one.
        //
        // It also has to exclude a speculative head's layer. The gather is
        // all-or-nothing — every layer must agree on the same block set, and
        // any disagreement returns `None` — and a draft head's K/V is its own
        // history, at its own seal level, so its blocks never match the trunk's.
        // Including it returned `None` on every turn: the turn sealed, the K/V
        // was written, and only the signature was silently absent, leaving
        // retrieval on recency with nothing saying so.
        let mut layers: Vec<Vec<ProvSource>> = Vec::new();
        let mut block_indices: Option<Vec<usize>> = None;
        let take = n_layers.min(self.backings.len());
        for backing in &self.backings[..take] {
            let (blocks, sources): (Vec<usize>, Vec<ProvSource>) = backing
                .provenance_sources(seq_idx, block_range)?
                .into_iter()
                .unzip();
            if blocks.is_empty() {
                return Ok(None);
            }
            match &block_indices {
                None => block_indices = Some(blocks),
                Some(prev) if *prev != blocks => return Ok(None),
                _ => {}
            }
            layers.push(sources);
        }
        match block_indices {
            Some(blocks) if !blocks.is_empty() => Ok(Some(ProvSources { layers, blocks })),
            _ => Ok(None),
        }
    }

    /// Launch the sign-pack kernel over already-resolved (concatenated) Q-chunk
    /// pointers — one kernel + one D2H for the whole batch. Empty on non-CUDA.
    pub fn run_prov_sign_pack(
        &self,
        all_ptrs: &[i64],
        sub_head_dim: usize,
    ) -> candle::Result<Vec<u64>> {
        match self.backings.first() {
            Some(b) => b.run_prov_sign_pack(all_ptrs, sub_head_dim),
            None => Ok(Vec::new()),
        }
    }

    /// Per-chunk `(offset, len, cum_before)` real-token window for a sequence — the
    /// exact layout attention reads, so a provenance / diagnostic gather can check
    /// only real slots and skip partial-chunk padding. Chunk structure is identical
    /// across layer backings, so layer 0 is authoritative.
    pub fn provenance_chunk_layout(
        &self,
        seq_idx: usize,
        seq_offset: usize,
    ) -> Vec<(u16, u16, usize)> {
        self.backings
            .first()
            .map(|b| b.provenance_chunk_layout(seq_idx, seq_offset))
            .unwrap_or_default()
    }
}
