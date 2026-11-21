//! Reports how a backing's K/V bands are stored, for diagnostics and policy.
//!
//! Occupancy is split between float and quantized storage, and class histograms
//! count live slots by size class. Bits per element are counted over compressed
//! bands only, so the ratio reflects compression rather than raw storage. A
//! block counts as quantized only when a band is in a non-R16 format, because R16
//! is raw capture and not compression.

use super::*;

impl ChunkedKvBacking {
    /// GPU arena occupancy split float vs quant across the pool this backing
    /// shares (arenas are pooled globally across layers with the same head
    /// config, so one backing's view is the whole model's). Feeds the per-wave
    /// `kv-pool` diagnostic that validates the compress-to-free rung shrinks the
    /// float side. See [`GpuArenaClassStats`].
    pub fn gpu_arena_class_stats(&self) -> GpuArenaClassStats {
        self.inner.pool.gpu_class_stats()
    }

    /// Calculate the percentage of a sequence's tokens that are stored in quantized arenas.
    ///
    /// Returns (quantized_tokens, total_tokens) based on which chunks are in quantized arenas.
    /// Each chunk holds `chunk_size` tokens.
    ///
    /// This validates that the actual ChunkRefs for a sequence point to quantized storage,
    /// not just that quantized arenas exist.
    pub fn quantized_token_stats(&self, batch_idx: usize) -> Result<(usize, usize)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return Ok((0, 0)), // No sequence at this slot
        };

        // A block counts as "quantized" if ANY of its bands is recorded in a
        // non-R16 quantized format. R16 is a raw storage format (K+Q capture),
        // not compression. The question is asked of the chunk's own tags — a
        // size-class arena has no format to report.
        let mut quantized_tokens = 0usize;
        let total_tokens = slot.block_count() * CHUNK_SIZE;
        for cw in slot.chunks_slice() {
            let any_quant = cw
                .bands()
                .any(|(_, tag)| tag.is_quantized() && tag != ArenaFormatTag::R16);
            if any_quant {
                quantized_tokens += CHUNK_SIZE;
            }
        }
        Ok((quantized_tokens, total_tokens))
    }

    /// Compute byte-level compression statistics for a sequence.
    /// Live band slots per size class for one sequence, plus how many of them
    /// are in formats **narrower than the smallest class**.
    ///
    /// This is audit A12's instrumentation. Splitting the 320 B rung into
    /// {64, 160, 320} would save 256 B/slot on `Q0`/`Q0_V`/`Q0_X` and
    /// 160 B/slot on `Q0_M2`/`Q1_S`, at the cost of two more classes whose
    /// steady-state partial tails run about half a region each — ≈16 MiB.
    /// Break-even is therefore ≈16 MiB ÷ ~200 B ≈ 65–84 K live slots in
    /// sub-320 formats, roughly **2 % of a ~4.8 M-slot pool** on this card.
    ///
    /// **Decision rule: split the low end only if sub-320 formats exceed ~2 %
    /// of live slots.** That fails at the C4/C5 production default, which never
    /// selects them, and is worth re-checking at C9/C10. This is what makes the
    /// rule answerable rather than a guess.
    ///
    /// Returns `(per-class live slots, sub-320 live slots)`.
    pub fn class_histogram(&self, batch_idx: usize) -> ([usize; SizeClass::COUNT], usize) {
        let mut per_class = [0usize; SizeClass::COUNT];
        let mut narrow = 0usize;
        let elems = self.inner.elems_per_chunk();
        let smallest = SizeClass::at(0).bytes();

        let Ok(state) = self.state.read() else {
            return (per_class, narrow);
        };
        let Some(slot) = state.sequences.get(batch_idx).and_then(|s| s.as_ref()) else {
            return (per_class, narrow);
        };
        for cw in slot.chunks_slice() {
            for (_, tag) in cw.bands() {
                let Some(payload) = payload_bytes_for_tag(tag, elems) else {
                    continue;
                };
                if let Some(class) = class_for_payload(payload) {
                    per_class[class.index()] += 1;
                }
                if payload < smallest {
                    narrow += 1;
                }
            }
        }
        (per_class, narrow)
    }

    pub fn compression_bpe(&self, batch_idx: usize) -> candle::Result<(f64, usize)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return Ok((0.0f64, 0)),
        };

        if slot.is_empty() {
            return Ok((0.0f64, 0));
        }

        // Walk every (head, palette, K/V) band individually. Each covers
        // CHUNK_SIZE × (head_dim / N_PALETTE) elements. Float and R16 bands are
        // excluded from both numerator and denominator so the ratio is
        // CR = 16 / effective_bpe over compressed bands only.
        let mut actual = 0f64;
        let mut n_quant = 0usize;
        for cw in slot.chunks_slice() {
            for (_, tag) in cw.bands() {
                let Some(KvFormat::Quantized(qf)) = tag.to_kv_format() else {
                    continue;
                };
                if qf == QuantFormat::R16 {
                    continue;
                }
                actual += qf.bits_per_elem() as f64 * cw.usage as f64;
                n_quant += cw.usage as usize;
            }
        }
        Ok((actual, n_quant))
    }

    /// [`Self::compression_bpe`] for one side: K bands when `is_value` is
    /// false, V bands when true. Bands interleave `[K, V, K, V, …]` per
    /// `(head, palette)` — the stride [`Self::compression_dist`] reads — so the
    /// ladder can show how a rung splits its bits between the two.
    pub fn compression_bpe_side(
        &self,
        batch_idx: usize,
        is_value: bool,
    ) -> candle::Result<(f64, usize)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let Some(slot) = state.sequences.get(batch_idx).and_then(|s| s.as_ref()) else {
            return Ok((0.0f64, 0));
        };
        let start = usize::from(is_value);
        let mut actual = 0f64;
        let mut n_quant = 0usize;
        for cw in slot.chunks_slice() {
            for (_, tag) in cw.bands().skip(start).step_by(2) {
                let Some(KvFormat::Quantized(qf)) = tag.to_kv_format() else {
                    continue;
                };
                if qf == QuantFormat::R16 {
                    continue;
                }
                actual += qf.bits_per_elem() as f64 * cw.usage as f64;
                n_quant += cw.usage as usize;
            }
        }
        Ok((actual, n_quant))
    }

    pub fn compression_dist(
        &self,
        batch_idx: usize,
        is_value: bool,
        ret: &mut HashMap<GgmlDType, usize>,
    ) {
        let state = self.state.read().unwrap();
        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return,
        };

        // Bands are interleaved [K_head0_pal0, V_head0_pal0, ...]; stride by 2
        // from 0 (K) or 1 (V) so only one side is counted. Each band reports the
        // format its own tag records.
        let start = usize::from(is_value);
        for cw in slot.chunks_slice() {
            for (_, tag) in cw.bands().skip(start).step_by(2) {
                let Some(fmt) = tag.to_kv_format() else {
                    continue;
                };
                let dtype = match fmt {
                    KvFormat::Float(dt) => dt.to_ggml_dtype(),
                    KvFormat::Quantized(qf) => qf.to_ggml_dtype(),
                };
                // One count per band slot, matching the per-block diagnostics
                // that show N_PALETTE × n_kv_head slots per format.
                *(ret.entry(dtype).or_default()) += 1;
            }
        }
    }

    /// Build the combined chunk_meta row for one batch slot.
    ///
    /// Returns one [`ChunkMeta`] entry per logical block.  Pass the result to
    /// [`ChunkMeta::into_u32s`] to flatten it for [`candle::Tensor::from_vec`].
    ///
    /// `seq_len` is the current total token count for the sequence; it is used
    /// to compute the active tail usage dynamically.
    pub fn chunk_meta_row(&self, batch_idx: usize, seq_len: usize) -> Vec<ChunkMeta> {
        let state = self.state.read().expect("chunked state lock poisoned");
        let max_blocks = state.max_blocks;
        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return vec![ChunkMeta::default(); max_blocks],
        };

        // Build block entries from chunks.
        // For the last block, compute usage dynamically from seq_len so that the
        // chunk_meta correctly reflects tokens about to be written (decode) or
        // being written in the current prefill batch.  This mirrors the old
        // active-block dynamic-usage path in HEAD.
        let n = slot.block_count();
        let mut committed: usize = 0;
        let mut row: Vec<ChunkMeta> = slot
            .chunks_slice()
            .iter()
            .enumerate()
            .map(|(i, cw)| {
                let rope = committed as i32;
                let usage = if i + 1 == n && seq_len > 0 {
                    // Last (write-target) block: derive from seq_len so that tokens
                    // not yet committed via set_current_seq_len are still visible.
                    (seq_len.saturating_sub(committed)).min(CHUNK_SIZE) as u32
                } else {
                    cw.usage
                };
                committed += usage as usize;
                ChunkMeta::new(usage, rope, cw.offset)
            })
            .collect();

        // Pad with default (unmapped) entries to reach max_blocks.
        row.resize(max_blocks, ChunkMeta::default());
        row
    }

    /// Report GPU memory usage broken down by arena format.
    ///
    /// Returns a Vec of `(format_label, arena_count, total_bytes)` tuples,
    /// one entry per unique format (e.g. "BF16", "F16", "Q8_0").
    pub fn memory_report(&self) -> Result<Vec<(String, usize, usize)>> {
        self.inner.storage.read(|s| {
            let mut by_format: std::collections::BTreeMap<String, (usize, usize)> =
                std::collections::BTreeMap::new();
            for arena in s.arenas().values() {
                let label = arena.format_label();
                let bytes = arena.gpu_memory_bytes();
                let entry = by_format.entry(label).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += bytes;
            }
            by_format
                .into_iter()
                .map(|(label, (count, bytes))| (label, count, bytes))
                .collect()
        })
    }

    /// Total GPU memory used by all arenas in this backing (bytes).
    pub fn total_arena_gpu_bytes(&self) -> Result<usize> {
        self.inner
            .storage
            .read(|s| s.arenas().values().map(|a| a.gpu_memory_bytes()).sum())
    }
}
