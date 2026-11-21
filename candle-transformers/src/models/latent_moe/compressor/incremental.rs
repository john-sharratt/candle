//! The streaming compressor used during decode.
//!
//! A decode step absorbs one token and emits an entry only when its group of
//! `ratio` tokens completes. The overlapping compressor keeps the last completed
//! group's rows as the next group's previous half, so the streamed entries match
//! the prefill form over the same prefix. Completed groups go to the pool as their
//! rows, with no copy. Snapshot and restore let speculative decoding roll the
//! state back to an accepted prefix, and `close` finalises a partial group at a
//! turn seal.

use super::*;

/// Streaming (decode-time) counterpart to [`Compressor`]. The prefill `Compressor::forward`
/// recomputes every compressed entry from the full prefix on each step; during incremental
/// decode we instead accumulate the current group's `ratio` token projections and emit one
/// entry the moment the group completes. For the overlapping (`ratio == 4`) compressor the
/// entry also pools the *previous* group's first-half projection rows, so the streamer
/// retains the last completed group's `(kv, score+ape)` to serve as the next group's "prev"
/// half — exactly the `overlap_transform` shift `prev[g] = prev_src[g-1]` done batch-wise in
/// prefill. The emitted entry is `pool → RMSNorm → RoPE(at group-start position g·ratio)`,
/// numerically equal to the prefill entry (proven by `incremental_matches_prefill`).
pub struct IncrementalCompressor {
    pub(super) c: Compressor,
    /// Current (incomplete) group's per-token projections, each `[1, cd]` F32.
    pub(super) kv_rows: Vec<Tensor>,
    pub(super) score_rows: Vec<Tensor>,
    /// Last completed group's RAW `kv` and `score` rows (`ratio` × `[1, cd]`),
    /// retained as the overlapping compressor's "prev" half. Rows, not a
    /// `[ratio, cd]` block, because rows are what the pool descriptor consumes —
    /// a completed group is retained by moving the `Vec`, with no copy at all.
    /// `None` before the first group completes.
    pub(super) prev_kv_group: Option<Vec<Tensor>>,
    pub(super) prev_score_group: Option<Vec<Tensor>>,
    /// Index of the next group to emit (its RoPE position is `group_idx · ratio`).
    pub(super) group_idx: usize,
}

/// A point-in-time copy of an [`IncrementalCompressor`]'s streaming state — the
/// partial-group buffers, the overlap prev-group halves, and the group counter.
/// Tensors are immutable in candle, so the clones are `Arc` bumps: taking a
/// snapshot is O(buffered rows) pointer copies, no data movement. Used by the
/// speculative-decode verify path to roll the compressor back to the accepted
/// prefix after a partial accept (rejected draft tokens must not stay absorbed).
#[derive(Clone)]
pub struct CompressorState {
    pub(super) kv_rows: Vec<Tensor>,
    pub(super) score_rows: Vec<Tensor>,
    pub(super) prev_kv_group: Option<Vec<Tensor>>,
    pub(super) prev_score_group: Option<Vec<Tensor>>,
    pub(super) group_idx: usize,
}

impl IncrementalCompressor {
    /// Snapshot the streaming state (see [`CompressorState`]). Cheap: `Arc`
    /// clones of the buffered rows + the counter.
    pub fn state_snapshot(&self) -> CompressorState {
        CompressorState {
            kv_rows: self.kv_rows.clone(),
            score_rows: self.score_rows.clone(),
            prev_kv_group: self.prev_kv_group.clone(),
            prev_score_group: self.prev_score_group.clone(),
            group_idx: self.group_idx,
        }
    }

    /// Restore a [`Self::state_snapshot`] — the compressor behaves exactly as it
    /// did at snapshot time (bit-identical emissions for identical subsequent
    /// rows; the snapshotted tensors are immutable).
    pub fn state_restore(&mut self, s: CompressorState) {
        self.kv_rows = s.kv_rows;
        self.score_rows = s.score_rows;
        self.prev_kv_group = s.prev_kv_group;
        self.prev_score_group = s.prev_score_group;
        self.group_idx = s.group_idx;
    }

    /// Feed one token's hidden state `x` (`[dim]` / `[1, dim]` / `[1, 1, dim]`) at the next
    /// sequence position and, when it completes a group of `ratio` tokens, return that group's
    /// compressed entry `[1, 1, d]` (post RMSNorm + RoPE). Returns `None` mid-group.
    ///
    /// `rope` must be the same `RotaryCache` the prefill path uses for this compressor.
    pub fn push(&mut self, x: &Tensor, rope: &RotaryCache) -> Result<Option<Tensor>> {
        let (kv, score) = self.c.project_row(x)?;
        self.push_projected_roped(&kv, &score, rope)
    }

    /// As [`Self::push`] but the caller supplies the already-projected `kv`/`score`
    /// rows (each `[1, coff·d]`, from [`Compressor::project_rows`]) instead of the
    /// raw hidden `x` — buffering + roped emit only, no per-row projection GEMV.
    /// Bit-identical to `push` on the same token (the projection is `push`'s only
    /// difference). The prefill fast path batches the projection once and streams
    /// the rows through here.
    pub fn push_projected_roped(
        &mut self,
        kv: &Tensor,
        score: &Tensor,
        rope: &RotaryCache,
    ) -> Result<Option<Tensor>> {
        self.kv_rows.push(kv.clone());
        self.score_rows.push(score.clone());
        if self.kv_rows.len() < self.c.ratio {
            return Ok(None);
        }
        Some(self.emit_group(rope)).transpose()
    }

    /// As [`Self::push`] but emitting the **pre-RoPE** entry plus its
    /// group-start position — the paged-kernel path's form (the kernel applies
    /// RoPE at read time from the stored position; storage stays
    /// position-free).
    pub fn push_raw(&mut self, x: &Tensor) -> Result<Option<(Tensor, u32)>> {
        let (kv, score) = self.c.project_row(x)?;
        self.push_projected(&kv, &score)
    }

    /// As [`Self::push_raw`] but the caller supplies the already-projected
    /// `kv`/`score` rows (each `[1, coff·d]`, from [`Compressor::project_rows`])
    /// instead of the raw hidden `x` — buffering + pre-RoPE emit only, no per-row
    /// projection GEMV. Bit-identical to `push_raw` on the same token. Feeds the
    /// batched-projection prefill fast path.
    pub fn push_projected(&mut self, kv: &Tensor, score: &Tensor) -> Result<Option<(Tensor, u32)>> {
        self.kv_rows.push(kv.clone());
        self.score_rows.push(score.clone());
        if self.kv_rows.len() < self.c.ratio {
            return Ok(None);
        }
        Some(self.emit_group_raw()).transpose()
    }

    /// Batched projection over a whole prefix `xs` (`[n, dim]` / `[1, n, dim]`)
    /// for this compressor's own weights — the source rows for
    /// [`Self::push_projected`] / [`Self::push_projected_roped`]. One GEMM pair
    /// instead of `n` per-row `project_row` GEMVs.
    pub fn project_rows(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        self.c.project_rows(xs)
    }

    /// Number of rows currently buffered in the incomplete group (`0..ratio`).
    /// The causal-visibility bound for a batched prefill: after consuming `t+1`
    /// rows this call, `(buffered_len() + t + 1) / ratio` groups are complete.
    pub fn buffered_len(&self) -> usize {
        self.kv_rows.len()
    }

    /// This compressor's group size (`ratio`).
    pub fn ratio(&self) -> usize {
        self.c.ratio
    }

    /// Snapshot this sequence's streaming state + its new rows as one
    /// [`SeqAssemble`] for [`assemble_groups_batched`]. Pure `Arc` clones —
    /// no device work. `kv`/`score` are this sequence's `[n, cd]` slices of
    /// the wave-batched projection.
    pub fn assemble_input(&self, kv: &Tensor, score: &Tensor) -> Result<SeqAssemble> {
        let n = kv.dim(0)?;
        let mut kv_pieces: Vec<Tensor> = Vec::with_capacity(self.kv_rows.len() + 1);
        let mut sc_pieces: Vec<Tensor> = Vec::with_capacity(self.score_rows.len() + 1);
        kv_pieces.extend(self.kv_rows.iter().cloned());
        sc_pieces.extend(self.score_rows.iter().cloned());
        kv_pieces.push(kv.clone());
        sc_pieces.push(score.clone());
        Ok(SeqAssemble {
            kv_pieces,
            sc_pieces,
            l0: self.kv_rows.len(),
            n,
            prev_kv: self.prev_kv_group.clone(),
            prev_sc: self.prev_score_group.clone(),
            group_idx: self.group_idx,
        })
    }

    /// Install the state advance [`assemble_groups_batched`] computed for this
    /// sequence — the batched twin of the state writes at the end of
    /// [`Self::assemble_groups`].
    pub fn assemble_apply(&mut self, upd: AssembleUpdate) {
        self.kv_rows = upd.kv_rows;
        self.score_rows = upd.score_rows;
        if let (Some(pk), Some(ps)) = (upd.prev_kv, upd.prev_sc) {
            self.prev_kv_group = Some(pk);
            self.prev_score_group = Some(ps);
        }
        self.group_idx += upd.group_add;
    }

    /// Batched, carried-state-aware group emission: consume `n` pre-projected
    /// rows (`kv`/`score` each `[n, cd]`, from [`Compressor::project_rows`]) and
    /// emit EVERY complete group they form in one batched pool — bit-identical to
    /// streaming the same rows one at a time through [`Self::push_projected`]
    /// (`rope = None`, pre-RoPE attn entries; matches `emit_group_raw`) or
    /// [`Self::push_projected_roped`] (`rope = Some`, roped indexer keys; matches
    /// `emit_group`). The carried partial-group buffer and the overlap prev-group
    /// seed the first emitted group, so a prefill that resumes mid-group
    /// (`base > 0`, non-empty buffer) is handled exactly. Returns
    /// `(entries [G, head_dim], group-start positions [G])`, or `None` when the
    /// rows complete no group (all buffered for the next call).
    pub fn emit_groups_projected(
        &mut self,
        kv: &Tensor,
        score: &Tensor,
        rope: Option<&RotaryCache>,
    ) -> Result<Option<(Tensor, Vec<u32>)>> {
        // Composition of the two halves below — behavior identical to the former
        // monolithic form (still gated by `emit_groups_batched_matches_streamed`).
        match self.assemble_groups(kv, score)? {
            None => Ok(None),
            Some(gp) => {
                let scope = desc::scope(self.c.device())?;
                let entry = self.c.pool_and_norm(&[&gp], rope, &scope)?;
                Ok(Some((entry, gp.positions)))
            }
        }
    }

    /// The batchable first half of [`Self::emit_groups_projected`]: combine the
    /// carried partial group with the new rows, and for every group that now
    /// completes produce its **pool inputs** `[groups, P, d]` (`P = 2r` overlap /
    /// `r` non-overlap) + group-start positions, WITHOUT pooling. Advances the
    /// per-seq streaming state (`prev_*_group`, partial buffer, `group_idx`).
    ///
    /// The state advance reads only the RAW group rows (`kv_g`/`score_g`/
    /// `kv_comb`), never the pooled/normed/roped entry, so several sequences'
    /// `GroupPool`s can be concatenated and pooled in ONE launch
    /// ([`Compressor::pool_and_norm`]) — the prefill wave batches the pool across
    /// all prompt sequences this way. `None` when no group completes (all buffered).
    pub fn assemble_groups(&mut self, kv: &Tensor, score: &Tensor) -> Result<Option<GroupPool>> {
        let r = self.c.ratio;
        let cd = self.c.cd();
        let n = kv.dim(0)?;

        // Combined stream = carried partial-group rows ++ the new rows, so a
        // resume mid-group starts exactly where the per-token buffer left off.
        let l0 = self.kv_rows.len();
        let (kv_comb, score_comb) = if l0 == 0 {
            (kv.clone(), score.clone())
        } else {
            let ck = Tensor::cat(&self.kv_rows.iter().collect::<Vec<_>>(), 0)?;
            let cs = Tensor::cat(&self.score_rows.iter().collect::<Vec<_>>(), 0)?;
            (Tensor::cat(&[&ck, kv], 0)?, Tensor::cat(&[&cs, score], 0)?)
        };
        let n_tot = l0 + n;
        let groups = n_tot / r;
        if groups == 0 {
            // No group completes: buffer every combined row for the next call.
            // ONE forced copy of the whole (tiny, < ratio rows) combined buffer
            // breaks the pin on the batched-projection input `kv` (a dim-0 narrow,
            // which plain `contiguous` would NOT copy); the per-row entries are
            // then free views into that small owned buffer. Replaces the former
            // per-row `force_contiguous` storm (2·n_tot copies → 2).
            self.kv_rows = retain_rows(&kv_comb)?;
            self.score_rows = retain_rows(&score_comb)?;
            return Ok(None);
        }
        let cutoff = groups * r;

        // [groups, r, cd] RAW group rows — `ape` rides in the pool descriptor,
        // so no biased copy is made here or retained.
        let kv_g = kv_comb.narrow(0, 0, cutoff)?.reshape((groups, r, cd))?;
        let score_g = score_comb.narrow(0, 0, cutoff)?.reshape((groups, r, cd))?;

        // prev[0] = the carried previous group's rows (absent for a true group
        // 0), prev[j>0] = group j−1's first half — which is this same block read
        // one group earlier, so the shift `overlap_transform` does with a gather
        // is a narrow offset here.
        let prev = match (&self.prev_kv_group, &self.prev_score_group) {
            (Some(pk), Some(ps)) => Some((pk.as_slice(), ps.as_slice())),
            _ => None,
        };
        let spans = self.c.group_spans(&kv_g, &score_g, 0, groups, prev)?;

        let positions: Vec<u32> = (0..groups)
            .map(|j| ((self.group_idx + j) * r) as u32)
            .collect();

        // Retain state: the last complete group as the next overlap prev, the
        // trailing rows as the next partial buffer, and advance the group index.
        // Each retained run costs a SINGLE forced copy out of the combined-stream
        // buffer (a dim-0 narrow that plain `contiguous` would not copy — leaving
        // the big batched-projection buffer pinned), so the stream buffer frees
        // while the retained state stays bounded to ≤2·r rows. Reads only the RAW
        // group rows — pooling-independent, so the pool can be deferred and
        // batched across sequences.
        let rem = n_tot - cutoff;
        self.prev_kv_group = Some(retain_rows(
            &kv_g.narrow(0, groups - 1, 1)?.reshape((r, cd))?,
        )?);
        self.prev_score_group = Some(retain_rows(
            &score_g.narrow(0, groups - 1, 1)?.reshape((r, cd))?,
        )?);
        self.kv_rows = retain_rows(&kv_comb.narrow(0, cutoff, rem)?)?;
        self.score_rows = retain_rows(&score_comb.narrow(0, cutoff, rem)?)?;
        self.group_idx += groups;

        Ok(Some(GroupPool { spans, positions }))
    }

    /// Absorb ONE pre-projected row (`[1, cd]`) and, when it completes a group,
    /// return that group's pool INPUTS without pooling — the deferred-pool twin
    /// of [`Self::push_projected`], and what the decode wave absorbs each step.
    /// The completed groups from every decode session then pool together in a
    /// single [`Compressor::pool_and_norm`].
    ///
    /// Distinct from [`Self::assemble_groups`], which combines a whole RUN of
    /// new rows with the carried buffer and is the prefill form: that rebuild
    /// costs a `cat` plus two `force_contiguous` copies EVERY call, which is the
    /// right trade when it amortises over a prompt but not when it runs per
    /// token. Here a step that completes no group costs nothing on the device —
    /// an `Arc` bump onto the row buffer — and at `ratio` 4 that is three decode
    /// steps in four.
    pub fn assemble_row(&mut self, kv: &Tensor, score: &Tensor) -> Result<Option<GroupPool>> {
        self.kv_rows.push(kv.clone());
        self.score_rows.push(score.clone());
        if self.kv_rows.len() < self.c.ratio {
            return Ok(None);
        }
        Some(self.assemble_group_raw()).transpose()
    }

    /// The buffered complete group's pool inputs + its start position, advancing
    /// the streaming state (`prev_*_group`, buffer, `group_idx`). The caller must
    /// have checked the group is full.
    ///
    /// Costs NOTHING on the device. The group's rows are handed to the pool as
    /// one segment each, read where they lie, and retained as the next group's
    /// previous half by moving the row `Vec` — no concatenation into a `[r, cd]`
    /// block, no channel-window copy for the overlap split, no `ape` add. Reads
    /// only the RAW rows, never a pooled result, which is what lets the pool
    /// defer and batch across sessions.
    pub(super) fn assemble_group_raw(&mut self) -> Result<GroupPool> {
        let d = self.c.head_dim;
        let r = self.c.ratio;

        let mut segs: Vec<PoolSegs> = Vec::with_capacity(2 * r);
        if self.c.overlap {
            // Group 0 has no previous group; its `−∞` pad pools to nothing, so
            // the segment is simply absent.
            if let (Some(pk), Some(ps)) = (&self.prev_kv_group, &self.prev_score_group) {
                segs.extend(self.c.row_segs(pk, ps, 0)?);
            }
            segs.extend(self.c.row_segs(&self.kv_rows, &self.score_rows, d)?);
        } else {
            segs.extend(self.c.row_segs(&self.kv_rows, &self.score_rows, 0)?);
        }

        // Retain this group as the next group's "prev" half, then reset the
        // current buffer — a `Vec` move, not a copy.
        self.prev_kv_group = Some(std::mem::take(&mut self.kv_rows));
        self.prev_score_group = Some(std::mem::take(&mut self.score_rows));

        let g = self.group_idx;
        self.group_idx += 1;
        Ok(GroupPool {
            spans: vec![PoolSpan { segs }],
            positions: vec![(g * r) as u32],
        })
    }

    /// Pool → RMSNorm (NO RoPE): the position-free entry `[1, 1, d]` and its
    /// group-start position. Assemble + pool, so the streamed reference and the
    /// deferred-pool decode path share ONE pool implementation
    /// ([`Compressor::pool_and_norm`]) rather than two that must be kept in
    /// step. RoPE is the caller's concern: `emit_group` applies it, while the
    /// kernel path stores pre-RoPE and rotates at read.
    pub(super) fn emit_group_raw(&mut self) -> Result<(Tensor, u32)> {
        let d = self.c.head_dim;
        let gp = self.assemble_group_raw()?;
        let scope = desc::scope(self.c.device())?;
        let entry = self.c.pool_and_norm(&[&gp], None, &scope)?;
        Ok((entry.reshape((1, 1, d))?, gp.positions[0]))
    }

    /// Finalize the trailing partial group at a **turn seal**: pool the
    /// `< ratio` currently-buffered rows into a corpus entry and return it
    /// pre-RoPE (`[1, 1, d]`) alongside its group-start position, mirroring
    /// [`Self::emit_group_raw`] but over fewer than `ratio` rows. Returns `None`
    /// when nothing is buffered (the group boundary fell exactly on the seal).
    ///
    /// For the overlapping (`ratio == 4`) compressor the pool still includes the
    /// **previous complete group's first-half rows** (the retained `prev_*`
    /// halves) — the closed partial is a softmax-weighted latent exactly like a
    /// full group, so the attention kernel merges it with no special case
    /// (docs/deepseek/deepseek_turn_seal_persistence.md Artifact B). `close` is terminal:
    /// it clears the buffer and advances `group_idx`; the compressor must not be
    /// pushed again after it.
    pub fn close(&mut self) -> Result<Option<(Tensor, u32)>> {
        let n = self.kv_rows.len();
        if n == 0 {
            return Ok(None);
        }
        let d = self.c.head_dim;
        let r = self.c.ratio;

        // Ragged by construction — `r` previous rows and only `n` current ones —
        // which the segment table carries with no special case at all.
        let mut segs: Vec<PoolSegs> = Vec::with_capacity(r + n);
        if self.c.overlap {
            // The partial group may itself be group 0: no previous group, so
            // there is no previous segment (the `−∞` pad pools to nothing).
            if let (Some(pk), Some(ps)) = (&self.prev_kv_group, &self.prev_score_group) {
                segs.extend(self.c.row_segs(pk, ps, 0)?);
            }
            segs.extend(self.c.row_segs(&self.kv_rows, &self.score_rows, d)?);
        } else {
            segs.extend(self.c.row_segs(&self.kv_rows, &self.score_rows, 0)?);
        }

        // Terminal: consume the buffer, but do NOT retain a new `prev_*` (nothing
        // follows a seal-close in this compressor's lifetime).
        self.kv_rows.clear();
        self.score_rows.clear();

        let g = self.group_idx;
        self.group_idx += 1;
        let gp = GroupPool {
            spans: vec![PoolSpan { segs }],
            positions: vec![(g * r) as u32],
        };
        // The same pool every other emit path runs — a closed partial is a
        // softmax-weighted latent exactly like a full group, just over fewer rows.
        let scope = desc::scope(self.c.device())?;
        let entry = self.c.pool_and_norm(&[&gp], None, &scope)?;
        Ok(Some((entry.reshape((1, 1, d))?, gp.positions[0])))
    }

    pub(super) fn emit_group(&mut self, rope: &RotaryCache) -> Result<Tensor> {
        let d = self.c.head_dim;
        let gp = self.assemble_group_raw()?;
        let scope = desc::scope(self.c.device())?;
        self.c
            .pool_and_norm(&[&gp], Some(rope), &scope)?
            .reshape((1, 1, d))
    }
}
