//! Captures each turn's per-token provenance signature from the Q values on the
//! device and folds it into the form that is stored.
//!
//! The GPU path packs the sign bits on the device and XOR-folds them on the host,
//! and it must agree bit for bit with the CPU path, because both produce the
//! stored signatures. A capture gathers only the stream layers, since a draft
//! head's K/V describes a speculative continuation and not the conversation's
//! history. Each token's signature is taken over its real window in each chunk,
//! so partial-chunk padding never contributes. A geometry the fold cannot express
//! is reported once per process rather than once per token. Removing a reasoning
//! span from a signature run shifts the indices after it, which is safe because
//! the only positional reader reads the user span that lies before it.

use super::*;

/// Assemble per-token raw wide `sign(Q)` from the GPU-packed sub-band bits
/// ([`ProvSignPacked`]) and fold each to the compact provenance signature. This
/// is the on-CPU tail of the GPU fast path in [`Scheduler::gather_wide_sigs`] and
/// is **bit-identical** to `fold_provenance_checked(WideQSig::from_band(..), fold)`:
/// a head's `head_dim` sign bits are its `n_palette` sub-band u64s laid down at
/// global dims `[p*sub_head_dim, (p+1)*sub_head_dim)` — exactly how `from_band`
/// packs them (bit `i` → word `i/64`, bit `i%64`). Only real tokens (per the
/// chunk `layout`) are emitted, skipping partial-chunk padding.
///
/// `fold` is the model's own derived fold, not the locked constants: it is what
/// the record will be stamped with, so folding under anything else stores bits
/// that disagree with their own description.
pub(super) fn assemble_folded_prov_sigs(
    packed: &ProvSignPacked,
    layout: &[(u16, u16, usize)],
    head_dim: usize,
    fold: FoldParams,
) -> Vec<WideQSig> {
    let chunk = candle_nn::CHUNK_SIZE;
    let wph = head_dim.div_ceil(64);
    let n_layers = packed.n_layers;
    let n_kv_head = packed.n_kv_head;
    let n_palette = packed.n_palette;
    let sub = packed.sub_head_dim;
    let n_blocks = packed.block_indices.len();
    let n_raw_heads = n_layers * n_kv_head;
    if wph == 0 || n_raw_heads == 0 || sub == 0 {
        return Vec::new();
    }
    // Checked once, not per token: every token of this capture has the same
    // `n_raw_heads` and `wph`, so the verdict cannot differ between them. A fold
    // that cannot fill all three groups would leave two thirds of every
    // signature zero, and an all-zero group agrees with everything it is scored
    // against — store nothing rather than that.
    if !fold_fits(n_raw_heads, wph, fold) {
        warn_unfoldable_capture(n_raw_heads, fold);
        return Vec::new();
    }

    let mut out = Vec::new();
    for (block_pos, &block_idx) in packed.block_indices.iter().enumerate() {
        let Some(&(offset, len, _cum)) = layout.get(block_idx) else {
            continue; // no metadata — can't place this block's tokens
        };
        let offset = offset as usize;
        for j in 0..len as usize {
            let t = offset + j; // physical token slot within the chunk
            if t >= chunk {
                break;
            }
            let mut words = vec![0u64; n_raw_heads * wph];
            for layer in 0..n_layers {
                for head in 0..n_kv_head {
                    let hidx = layer * n_kv_head + head;
                    for p in 0..n_palette {
                        let warp =
                            ((layer * n_blocks + block_pos) * n_kv_head + head) * n_palette + p;
                        let Some(&bits) = packed.packed.get(warp * chunk + t) else {
                            continue;
                        };
                        // Palette p → global head-dims [p*sub, (p+1)*sub); place its
                        // `sub` bits at that offset within the head's `wph` words,
                        // splitting across the word boundary if it straddles one.
                        let bit_off = p * sub;
                        let w = bit_off / 64;
                        let sh = bit_off % 64;
                        if let Some(word) = words.get_mut(hidx * wph + w) {
                            *word |= bits << sh;
                        }
                        if sh + sub > 64 {
                            if let Some(word) = words.get_mut(hidx * wph + w + 1) {
                                *word |= bits >> (64 - sh);
                            }
                        }
                    }
                }
            }
            out.push(fold_provenance_fitted(
                &WideQSig {
                    n_heads: n_raw_heads as u16,
                    words,
                },
                fold,
            ));
        }
    }
    out
}

/// One warning per process for a capture geometry the fold cannot express.
///
/// Per-token would be one line per token of every turn; per-turn would still be
/// a flood. The condition is a property of the model's geometry, so it is the
/// same answer every time and saying it once is saying it.
pub(super) fn warn_unfoldable_capture(n_raw_heads: usize, fold: FoldParams) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            n_raw_heads,
            heads_per_layer = fold.heads_per_layer,
            group_sizes = ?fold.group_sizes,
            "provenance capture refused: the model's fold cannot fill all three \
             layer-groups from this stack, so no signature is stored. Retrieval \
             falls back to recency for these turns."
        );
    });
}

/// A turn's per-token signature run with `span` removed.
///
/// The run is 1:1 with the turn's real-token grid, so `span` is a grid range.
/// Removing it leaves the run 1:1 with the **windowed** turn — which is what a
/// projection of that turn actually injects — and shifts every index after the
/// hole.
///
/// **That shift is safe here and would not be everywhere**, so it is worth
/// saying why. Exactly one consumer indexes this run positionally:
/// `Substrate::user_sig_span`, which reads the layout's `User` segments. On a
/// dialogue turn the user body lies wholly before the assistant body, hence
/// wholly before the reasoning, so the hole is past everything it addresses. The
/// one shape with a `User` segment *after* the assistant body is a code_read
/// tool exchange, and that returns from `build_turn_layout` before the thinking
/// split — it has no `Thinking` segment, so `span` is `None` and this is never
/// called for it. The other consumer, the belief gallery, scores each signature
/// independently and reads no index at all.
///
/// A free function so the range arithmetic is testable without a scheduler.
pub(super) fn sigs_without_span(
    sigs: Vec<WideQSig>,
    span: std::ops::Range<usize>,
) -> Vec<WideQSig> {
    let end = span.end.min(sigs.len());
    if span.start >= end {
        return sigs;
    }
    sigs.into_iter()
        .enumerate()
        .filter_map(|(i, s)| (!(span.start..end).contains(&i)).then_some(s))
        .collect()
}

impl Scheduler {
    /// Digest a sealed turn's hot K/V content — see
    /// [`SchedulerRequest::ReadTurnKvDigest`].
    ///
    /// Gathers with the SAME machinery the cold write uses
    /// ([`crate::persistence::thread::build_grid`] → per-chunk
    /// [`crate::persistence::record::ChunkPayload`]), so what is hashed is
    /// exactly what a tier migration would persist: offsets, band formats,
    /// palette maps, outer scales, and the raw arena bytes.
    #[cfg(any(test, feature = "test-helpers"))]
    pub(super) fn handle_read_turn_kv_digest(
        &mut self,
        sequence_id: SequenceId,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Result<Option<ContentHash>, ConversationError> {
        let Some(conversation) = self.slot_conversations.get(&sequence_id) else {
            return Err(ConversationError::Channel(format!(
                "ReadTurnKvDigest: unknown sequence {sequence_id}"
            )));
        };
        let Some(sealed) = conversation.read().turn_sealed_of(timeline, index) else {
            return Ok(None);
        };
        let backings = self.session.backings();
        let grid = crate::persistence::thread::build_grid(&sealed, backings, self.model.device())
            .map_err(ConversationError::Model)?;
        let mut h = ContentHasher::new();
        for layer in 0..grid.n_layers() {
            h.update(&(layer as u32).to_le_bytes());
            for chunk in grid.layer(layer) {
                h.update(&chunk.token_count.to_le_bytes());
                h.update(&chunk.payload.offset.to_le_bytes());
                h.update(&chunk.payload.k_formats);
                h.update(&chunk.payload.v_formats);
                h.update(&chunk.payload.k_pal);
                h.update(&chunk.payload.v_pal);
                for s in &chunk.payload.k_scale {
                    h.update(&s.to_le_bytes());
                }
                for s in &chunk.payload.v_scale {
                    h.update(&s.to_le_bytes());
                }
                h.update(&chunk.payload.kv_bytes);
            }
        }
        Ok(Some(h.finish()))
    }

    /// Calls the R16 dump path for each layer and returns results in the
    /// same order as `layer_indices`.  Each element is
    /// `(layer_idx, Vec<(block_idx, k_flat, v_flat, q_flat)>)`.
    pub(super) fn handle_extract_raw_kvq(
        &self,
        seq_idx: usize,
        layer_indices: &[usize],
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<(usize, Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>)>, ConversationError> {
        layer_indices
            .iter()
            .map(|&layer_idx| {
                let blocks = self
                    .session
                    .gather_r16_kv_for_provenance(seq_idx, layer_idx, block_range)
                    .map_err(ConversationError::Model)?;
                Ok((layer_idx, blocks))
            })
            .collect()
    }

    /// Capture the turn's per-token provenance signature from R16, one [`WideQSig`] per
    /// REAL token in token order. It walks each chunk's real-token window
    /// `[offset, offset+len)` from [`provenance_chunk_layout`] — the exact slots attention
    /// reads — so interior partial chunks (section / glue / reproject boundaries)
    /// contribute only their real tokens, never their zero padding; the result is 1:1
    /// aligned with the turn's `Tokens` record. Every token of each block is captured (no
    /// structural filter; filtering is a query-time concern). Each token's raw all-heads /
    /// all-layers `sign(Q)` is folded to the compact signature under this model's own
    /// derived fold (three layer-groups, heads separate) before storing. A block the gap
    /// quantized contributes the `sign(Q)` bits the gap captured before its convert; a
    /// block with neither R16 Q nor captured bits contributes nothing.
    pub(super) fn gather_wide_sigs(
        &self,
        seq_id: SequenceId,
        range: (usize, usize),
    ) -> Vec<WideQSig> {
        // **The STREAM layers, not every KV layer.** A signature is a statement
        // about the conversation's own history, and a draft head's K/V is not
        // that — it is the speculative continuation the head proposed. Folding
        // it in would describe two histories as one, and on the GPU path it did
        // worse than that: the capture is all-or-nothing across layers, so the
        // head's disagreeing block set made every turn's signature come back
        // empty.
        //
        // Both paths take the same bound, because they must produce the same
        // bits — the fold's geometry is `layers × kv_heads`, so a CPU fallback
        // counting one more layer than the kernel would not be a slower answer
        // but a different one. And both take the number the FOLD was derived
        // from, which is the only way that guarantee holds.
        let n_layers = self.prov_capture_layers.min(self.session.num_layers());
        let n_kv_head = self.session.n_kv_head();
        let head_dim = self.session.head_dim();
        if n_layers == 0 || n_kv_head == 0 || head_dim == 0 || range.1 <= range.0 {
            return Vec::new();
        }
        let seq_off = self.session.sequence_offset(seq_id.0).unwrap_or(0);
        let layout = self.session.provenance_chunk_layout(seq_id.0, seq_off);

        // GPU fast path: read R16 Q + sign + bit-pack on device in ONE launch
        // across all layers, then D2H only the packed bits (a few KB) and XOR-fold
        // them here — bit-identical to the CPU path below. Falls through to the CPU
        // gather on non-CUDA / unsupported geometry / no R16 blocks. Split into
        // resolve / kernel / assemble so the wave breakdown shows where the
        // per-scope cost lives (see the `PROV_*_US` accumulators).
        use std::sync::atomic::Ordering::Relaxed;
        let sub = self.session.prov_sub_head_dim();
        if sub > 0 {
            let t_res = Instant::now();
            let resolved = self
                .session
                .resolve_provenance_sources(seq_id.0, Some(range), n_layers)
                .ok()
                .flatten();
            PROV_RESOLVE_US.fetch_add(t_res.elapsed().as_micros() as u64, Relaxed);
            if let Some(sources) = resolved {
                let t_run = Instant::now();
                let packed = self.session.pack_provenance_signs(&sources).ok().flatten();
                PROV_KERNEL_US.fetch_add(t_run.elapsed().as_micros() as u64, Relaxed);
                if let Some(ps) = packed {
                    let t_asm = Instant::now();
                    let sigs = assemble_folded_prov_sigs(&ps, &layout, head_dim, self.prov_fold);
                    PROV_ASSEMBLE_US.fetch_add(t_asm.elapsed().as_micros() as u64, Relaxed);
                    PROV_PATH_GPU.fetch_add(1, Relaxed);
                    return sigs;
                }
            }
        }

        let layers: Vec<usize> = (0..n_layers).collect();
        let dump =
            match self
                .session
                .gather_r16_kv_provenance_layers(seq_id.0, &layers, Some(range))
            {
                Ok(d) if !d.is_empty() && !d[0].is_empty() => d,
                _ => return Vec::new(),
            };
        PROV_PATH_CPU.fetch_add(1, Relaxed);
        // Once per capture, not per token — same shape for every token here.
        if !fold_fits(n_layers * n_kv_head, head_dim.div_ceil(64), self.prov_fold) {
            warn_unfoldable_capture(n_layers * n_kv_head, self.prov_fold);
            return Vec::new();
        }
        let chunk = candle_nn::CHUNK_SIZE;
        let band_len = n_layers * n_kv_head * head_dim;
        let n_blocks = dump[0].len();
        let mut out = Vec::with_capacity(n_blocks * chunk);
        for bi in 0..n_blocks {
            // Absolute chunk index of this gathered block → its real-token window.
            let block_idx = dump[0][bi].0;
            let Some(&(offset, len, _cum)) = layout.get(block_idx) else {
                continue; // no metadata for this chunk — skip (cannot place its tokens)
            };
            let offset = offset as usize;
            for j in 0..len as usize {
                let t = offset + j; // physical slot of real token j
                if t >= chunk {
                    break;
                }
                let mut band = Vec::with_capacity(band_len);
                let mut ok = true;
                for layer in &dump {
                    if bi >= layer.len() {
                        ok = false;
                        break;
                    }
                    let q_flat = &layer[bi].3;
                    for h in 0..n_kv_head {
                        band.extend_from_slice(&extract_q_vector_r16(
                            q_flat, t, h, n_kv_head, head_dim, chunk,
                        ));
                    }
                }
                if ok && band.len() == band_len {
                    // Fold the raw all-heads/all-layers sign(Q) into the compact
                    // provenance signature (three layer-groups, per-layer stagger,
                    // heads separate) before storing — 16× smaller than the full
                    // wide-Q, and what the z-score late-fusion retrieval scores.
                    // See docs §23. Same fold as the GPU path above, and the same
                    // fold the record is stamped with.
                    out.push(fold_provenance_fitted(
                        &WideQSig::from_band(&band, head_dim),
                        self.prov_fold,
                    ));
                }
            }
        }
        out
    }
}
