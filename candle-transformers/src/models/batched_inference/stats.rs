//! Reports how much KV a sequence holds and in which formats, and reads the
//! sequence's length as one all of its layers can serve.
//!
//! The token count is the minimum across the stream's layers, not layer 0 alone.
//! A windowed prefill advances layer 0 first, so reading layer 0 would move the
//! offset past layers still pending resume. Block counts are read the same way.
//! Compression figures are taken per side, because a rung can trade bits between
//! K and V so that the combined ratio stands still while the sides move in
//! opposite directions. Setting an offset outright clamps a slot that ran ahead
//! of its backing, since the backing length is the truth at a wave boundary.

use super::*;

impl BatchedInferenceSession {
    /// Token count actually covered by a sequence's live block table — the sum
    /// of live-chunk token counts in the layer-0 backing, i.e. exactly the
    /// `cum` the slot-header build walks (all layers share chunk boundaries).
    /// This is the physical ground truth the wave-boundary offset reconciler
    /// compares `session.offset` against; `current_seq_len` is the write
    /// cursor and reads 0 for freshly injected slots whose tables already
    /// hold sealed tokens, so it cannot serve as the backing length.
    pub fn sequence_backing_tokens(&self, idx: usize) -> Option<usize> {
        let caches = self.sequence_caches(idx)?;
        if caches.caches.is_empty() {
            return None;
        }
        // The token count the live block table covers, taken as the MINIMUM
        // across layers — not layer 0 alone. During a windowed wave the creep
        // prefill advances layers incrementally (layer 0 first), so layer 0 runs
        // AHEAD of the layers still pending resume. Reading only layer 0 would
        // let the forward-entry reconciler advance the slot's offset past those
        // lagging layers; a decode that then reads a lagging layer trips the
        // slot-header count invariant ("block table lost chunks"). The minimum is
        // the token prefix EVERY layer has materialised, so an offset clamped to
        // it is valid on every layer, and as the wave completes and the layers
        // converge the minimum rises to the true length. For a uniform (settled)
        // slot every layer is equal, so this is identical to reading layer 0.
        // The STREAM's layers only. A draft head's layer is stepped by the
        // head's own pass, so it holds nothing at all until the head runs and
        // trails by a token mid-wave once it does — either way its prefix is not
        // evidence about the stream's, and a minimum that included it would
        // report 0 and drive the wave-entry reconciler to clamp the sequence
        // back to the start of its history. See [`KvLayers`].
        let _g = span("session:backing_tokens");
        let mut min_cum: Option<usize> = None;
        for cache in caches.caches.iter().take(self.stream_layers) {
            let mut cum: usize = 0;
            cache.k_cache().chunked_visit_live_chunks(|it| {
                for c in it {
                    cum += c.token_count as usize;
                }
            });
            min_cum = Some(min_cum.map_or(cum, |m: usize| m.min(cum)));
        }
        min_cum
    }

    /// Set a sequence's logical offset outright. Used by the wave-boundary
    /// offset reconciler to clamp a slot whose offset ran AHEAD of its physical
    /// backing (a projection that dropped un-liftable sections injects fewer
    /// tokens than the planner counted): every wave's kv metadata is derived
    /// from this offset, and the attention kernels resolve each position
    /// through the slot's physical block table — an offset past the backing
    /// sends them past the end of the slot's staged state into neighboring
    /// uploads. The backing length is the single source of truth at a wave
    /// boundary; positions are slot-relative (slice ropes), so the clamped
    /// value is also the correct RoPE base for new tokens.
    pub fn set_sequence_offset(&mut self, idx: usize, offset: usize) -> Result<()> {
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }
        if let Some(ref mut state) = self.sequences[idx] {
            state.offset = offset;
            Ok(())
        } else {
            candle::bail!("sequence {} not allocated", idx);
        }
    }

    /// Validate that no two sessions share a raw GID across all layers.
    ///
    /// Returns `Ok(())` if clean, or an error listing every aliased GID with
    /// the GPU pointer and both owner locations.  Call before the decode kernel
    /// to confirm or rule out cross-session KV contamination.
    pub fn validate_gid_uniqueness(&self, seq_indices: &[usize]) -> Result<()> {
        let entries: Vec<(usize, usize)> = seq_indices
            .iter()
            .map(|&i| (i, self.sequence_offset(i).unwrap_or(0)))
            .collect();

        let mut all_violations: Vec<String> = Vec::new();
        for (layer_idx, backing) in self.backings.iter().enumerate() {
            let v = backing.validate_gid_uniqueness(&entries)?;
            for s in v {
                all_violations.push(format!("[layer {layer_idx}] {s}"));
            }
        }

        if all_violations.is_empty() {
            Ok(())
        } else {
            candle::bail!(
                "GID aliasing detected across {} sessions ({} violations):\n{}",
                seq_indices.len(),
                all_violations.len(),
                all_violations.join("\n")
            )
        }
    }

    /// Ensure capacity for writing tokens at the given offsets.
    pub fn ensure_capacity(&self, sequence_indices: &[usize], add: usize) -> Result<()> {
        for backing in &self.backings {
            for &idx in sequence_indices.iter() {
                let offset = self.sequence_offset(idx).unwrap_or(0);
                backing.ensure_for_offset(idx, offset, add)?;
            }
        }
        Ok(())
    }

    /// **Deprecated**: `no_quantize` is no longer used in the flat-chunks model.
    /// This method is a no-op kept for API compatibility.
    #[allow(unused_variables)]
    pub fn set_no_quantize(&self, seq_idx: usize, no_quantize: bool) -> Result<()> {
        Ok(())
    }

    pub fn estimate_quantized_percentage(&self) -> Option<f64> {
        let sequences = self
            .sequences
            .iter()
            .enumerate()
            .filter_map(|(n, v)| v.as_ref().map(|_| n))
            .collect::<Vec<_>>();
        self.estimate_quantized_percentage_by_sequences(&sequences)
    }

    pub fn estimate_quantized_percentage_by_sequences(
        &self,
        sequence_indices: &[usize],
    ) -> Option<f64> {
        let caches_and_offsets = self.caches_for_sequences(sequence_indices);
        let mut total_quantized = 0usize;
        let mut total_tokens = 0usize;

        for (seq_idx, _, caches) in caches_and_offsets {
            // Sample the first layer only — all layers share one arena pool so layer 0 is
            // representative for an estimate.  Use seq_idx as batch_idx; that is the slot
            // index assigned when the sequence was registered in the backing.
            if let Some(kv_cache) = caches.caches.first() {
                if let Some(result) = kv_cache.quantized_token_stats(seq_idx) {
                    let Ok((quant, total)) = result else {
                        continue;
                    };
                    total_quantized += quant;
                    total_tokens += total;
                }
            }
        }

        if total_tokens > 0 {
            let percent = (total_quantized as f64 / total_tokens as f64) * 100.0;
            Some(percent)
        } else {
            None
        }
    }

    /// Compression ratio across sequences, measured over all layers.
    ///
    /// Returns the ratio of F16-equivalent bytes to actual bytes. A ratio > 1.0
    /// means compression is happening. For uniform modes this is deterministic;
    /// for adaptive modes it reflects the actual per-chunk format distribution.
    pub fn compression_ratio_by_sequences(&self, sequence_indices: &[usize]) -> Option<f64> {
        let mut total_actual = 0.0f64;
        let mut total_elements = 0usize;

        let caches_and_offsets = self.caches_for_sequences(sequence_indices);
        for (seq_idx, _, caches) in &caches_and_offsets {
            for kv_cache in &caches.caches {
                if let Some(result) = kv_cache.compression_bpe(*seq_idx) {
                    total_actual += result.0;
                    total_elements += result.1;
                }
            }
        }

        if total_elements > 1 {
            let bpe = total_actual / total_elements as f64;
            Some(16.0f64 / bpe)
        } else {
            None
        }
    }

    /// [`Self::compression_ratio_by_sequences`] for one side — K when
    /// `is_value` is false. A rung that trades bits between K and V shows it
    /// here and nowhere else: the combined ratio can stand still while the two
    /// sides move in opposite directions.
    pub fn compression_ratio_by_side(
        &self,
        sequence_indices: &[usize],
        is_value: bool,
    ) -> Option<f64> {
        let mut total_actual = 0.0f64;
        let mut total_elements = 0usize;
        for (seq_idx, _, caches) in &self.caches_for_sequences(sequence_indices) {
            for kv_cache in &caches.caches {
                if let Some((actual, n)) = kv_cache.compression_bpe_side(*seq_idx, is_value) {
                    total_actual += actual;
                    total_elements += n;
                }
            }
        }
        (total_elements > 1).then(|| 16.0f64 / (total_actual / total_elements as f64))
    }

    /// One side's format histogram across sequences — K when `is_value` is
    /// false — as `(format, bands)`.
    pub fn compression_dist_by_side(
        &self,
        sequence_indices: &[usize],
        is_value: bool,
    ) -> ahash::HashMap<GgmlDType, usize> {
        let mut ret = ahash::HashMap::default();
        for (seq_idx, _, caches) in &self.caches_for_sequences(sequence_indices) {
            for kv_cache in &caches.caches {
                kv_cache.compression_dist_side(*seq_idx, is_value, &mut ret);
            }
        }
        ret
    }

    /// Compression distribution across sequences, measured over all layers.
    pub fn compression_dist_by_sequences(
        &self,
        sequence_indices: &[usize],
    ) -> ahash::HashMap<GgmlDType, usize> {
        let mut ret = ahash::HashMap::default();

        let caches_and_offsets = self.caches_for_sequences(sequence_indices);
        for (seq_idx, _, caches) in &caches_and_offsets {
            for kv_cache in &caches.caches {
                kv_cache.compression_dist(*seq_idx, &mut ret);
            }
        }

        ret
    }

    pub fn get_sequence_stats(&self, seq_idx: usize) -> SequenceStats {
        let mut ret = SequenceStats::default();
        for (actual_seq_idx, _, caches) in self.caches_for_sequences(&[seq_idx]) {
            for kv_cache in &caches.caches {
                if let Some(result) = kv_cache.quantized_token_stats(actual_seq_idx) {
                    let Ok((quant, total)) = result else {
                        continue;
                    };
                    ret.quantized_tokens += quant as u64;
                    ret.total_tokens += total as u64;
                    ret.active_tokens += total as u64;
                }
            }
        }
        ret
    }
}

#[derive(Debug, Default, Clone)]
pub struct SequenceStats {
    /// How many tokens are current active in the conversation window
    pub active_tokens: u64,

    /// How many tokens are there in total that make up the conversation history
    pub total_tokens: u64,

    /// How many of the tokens have been quantized to save space
    pub quantized_tokens: u64,
}
