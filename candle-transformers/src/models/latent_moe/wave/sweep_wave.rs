//! One forward of the batched engine over a wave of decode, prefill and glue rows.
//!
//! The sweep splits rows into decode, prefill and glue groups, slides each
//! writer's sliding window before the arena is sized, and builds one metadata set
//! for the whole wave. Decode attends in one paged launch, and the projections,
//! compressor pools and gallery appends are each issued once per layer over every
//! row. Prefill absorbs each prompt in one launch per layer. The head scores each
//! decode row and each prefill sequence's last row only. The bracket around the
//! sweep restores the captured corpus state when it fails, so a failed wave leaves
//! no trace in the corpus.

use super::*;

impl BatchedEngine {
    /// The sweep body proper — see [`WaveSweep::sweep`] above for the
    /// failure-atomicity bracket around it.
    pub(super) fn sweep_wave_inner(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        let WaveGroups {
            n_decode,
            n_prefill,
            seq_ids,
            inputs,
            pending_glue,
            generation: wave_generation,
            layer_start,
            layer_end,
            x_in,
            mode,
            act_dtype: _,
            adapter,
        } = wave;
        // Refused rather than ignored. Dropping the name here would serve the
        // BASE model under an adapter's name, which reads as a bad fine-tune
        // rather than as an unsupported architecture.
        if let Some(name) = adapter {
            candle::bail!(
                "latent_moe wave: this architecture has no LoRA support, so adapter \
                 `{name}` cannot be applied"
            );
        }
        // The KV↔expert boundary's GROWING direction, in the one gap it is
        // legal in: between forwards, before this wave opens any state. Spare
        // KV regions above the KV side's recent high-water go back to the
        // weight zone as resident-expert slots — without this the boundary
        // only ever moves toward KV (`request_kv_ground` buys on the spot) and
        // expert residency ratchets down across a long run. The shrink
        // direction needs no call here: a KV claim that runs out buys its
        // ground itself. Mirrors the blanket `BatchedModel` wave's phase 0. A
        // segment of a step had it when the step prepared it.
        if !matches!(mode, SweepMode::InStep(_)) {
            self.engine.experts().reclaim_spare_ground();
        }
        let e = &self.engine;
        let cfg = e.cfg();
        let n_layers = self.engine.layer_count();
        let hc = e.hc();

        // Group split, in the driver's internal `[decode | prefill | glue]`
        // order (1-token prefills already folded into the decode group).
        let n_glue_members = seq_ids
            .len()
            .checked_sub(n_decode + n_prefill)
            .ok_or_else(|| candle::Error::Msg("deepseek wave: group bounds exceed batch".into()))?;
        let (dec_member_seqs, rest) = seq_ids.split_at(n_decode);
        let (pre_member_seqs, glue_member_seqs) = rest.split_at(n_prefill);
        let (dec_member_inputs, rest_in) = inputs.split_at(n_decode);
        let (pre_member_inputs, glue_member_inputs) = rest_in.split_at(n_prefill);
        debug_assert_eq!(glue_member_seqs.len(), n_glue_members);

        // The verifying sequences, read once for the whole wave. Their blocks
        // arrive as multi-token PREFILL members (one context per sequence — the
        // wave driver's contract), and are expanded here to one decode-class
        // row per block position: the same seq on consecutive rows, exactly the
        // per-token decode stream, batched. Verify members must LEAD the
        // prefill group so the decode-class rows `[plain | verify]` stay one
        // contiguous slice of the packed buffer — the batched decode front-end
        // (projection, assemble, pool, select, gather) runs on a single narrow.
        let verify_seqs: Vec<usize> = self
            .verify_all_rows
            .read()
            .map_err(|_| candle::Error::Msg("verify_all_rows lock poisoned".into()))?
            .clone();
        let mut n_verify_members = 0usize;
        for (i, s) in pre_member_seqs.iter().enumerate() {
            if verify_seqs.contains(s) {
                if i != n_verify_members {
                    candle::bail!(
                        "verify wave: verify members must lead the prefill group \
                         (prefill {pre_member_seqs:?}, verify set {verify_seqs:?})"
                    );
                }
                n_verify_members += 1;
            }
        }

        // Decode-class rows: the plain decode members, then each verify block
        // position as one row of its sequence. From here down the sweep's row
        // model is unchanged from the private-loop era — `decode_seqs` may name
        // the same sequence on consecutive rows (a verify block), and every
        // consumer below already handles that.
        // Every member's ids in one readback, then split by group.
        if dec_member_inputs.iter().any(|t| t.device().is_cuda())
            || pre_member_inputs.iter().any(|t| t.device().is_cuda())
            || glue_member_inputs.iter().any(|t| t.device().is_cuda())
        {
            super::super::readback::note_readback();
        }
        let mut ids = host_token_ids(
            dec_member_inputs
                .iter()
                .chain(pre_member_inputs.iter())
                .chain(glue_member_inputs.iter()),
            None,
        )?
        .per_input
        .into_iter();
        let dec_member_ids: Vec<Vec<u32>> = ids.by_ref().take(dec_member_inputs.len()).collect();
        let pre_member_ids: Vec<Vec<u32>> = ids.by_ref().take(pre_member_inputs.len()).collect();
        let glue_ids: Vec<Vec<u32>> = ids.collect();

        let mut dec_class_seqs: Vec<usize> = dec_member_seqs.to_vec();
        let mut decode_ids: Vec<u32> = Vec::with_capacity(dec_member_ids.len());
        for (seq, toks) in dec_member_seqs.iter().zip(&dec_member_ids) {
            let [tok] = toks.as_slice() else {
                candle::bail!(
                    "decode member {seq} carries {} tokens; a decode row is one",
                    toks.len()
                );
            };
            decode_ids.push(*tok);
        }
        for m in 0..n_verify_members {
            for &tok in &pre_member_ids[m] {
                dec_class_seqs.push(pre_member_seqs[m]);
                decode_ids.push(tok);
            }
        }
        let decode_seqs: &[usize] = &dec_class_seqs;

        // The remaining prefill members are ordinary (creep) prompts.
        let prefill_seqs: &[usize] = &pre_member_seqs[n_verify_members..];
        let prefill_ids: Vec<Vec<u32>> = pre_member_ids[n_verify_members..].to_vec();
        let prefill_lens: Vec<usize> = prefill_ids.iter().map(|v| v.len()).collect();
        let glue_seqs: &[usize] = glue_member_seqs;
        let glue_lens: Vec<usize> = glue_ids.iter().map(|v| v.len()).collect();

        // Glue descriptors: taken off the session by the driver, one per glue
        // sequence in order. Positions derive from the reserved gap chunks:
        // block logical start + (in-block offset − chunk offset). DeepSeek glue
        // is causal-only (locked §E) — fwd_ahead must be 0.
        let glue_desc: Vec<PendingGlue> = if glue_seqs.is_empty() {
            Vec::new()
        } else {
            let desc = pending_glue.ok_or_else(|| {
                candle::Error::Msg("glue rows without staged PendingGlue descriptors".into())
            })?;
            if desc.len() != glue_seqs.len() {
                candle::bail!(
                    "PendingGlue count {} != glue seq count {}",
                    desc.len(),
                    glue_seqs.len()
                );
            }
            desc
        };
        let mut glue_pos: Vec<Vec<u32>> = Vec::with_capacity(glue_seqs.len());
        for (gi, &seq) in glue_seqs.iter().enumerate() {
            let d = &glue_desc[gi];
            if d.fwd_ahead.iter().any(|&f| f != 0) {
                candle::bail!("deepseek glue is causal-only (fwd_ahead must be 0)");
            }
            let chunks = session.backings()[0]
                .live_chunks_as_sealed(seq)
                .unwrap_or_default();
            let mut block_start = Vec::with_capacity(chunks.len());
            let mut block_off = Vec::with_capacity(chunks.len());
            let mut cum = 0usize;
            for c in &chunks {
                block_start.push(cum);
                block_off.push(c.offset as usize);
                cum += c.token_count as usize;
            }
            let mut pos = Vec::with_capacity(d.write_slice.len());
            for (s_idx, &blk) in d.write_slice.iter().enumerate() {
                let b = blk as usize;
                if b >= block_start.len() {
                    candle::bail!("glue descriptor block {b} out of table range");
                }
                let within = d.write_in_blk[s_idx] as usize;
                pos.push((block_start[b] + within.saturating_sub(block_off[b])) as u32);
            }
            if pos.len() != glue_lens[gi] {
                candle::bail!(
                    "glue descriptor tokens {} != glue input tokens {}",
                    pos.len(),
                    glue_lens[gi]
                );
            }
            glue_pos.push(pos);
        }

        // Sequence offsets (position of the first new token per sequence).
        // A sequence may occupy SEVERAL decode rows (a speculative verify
        // block: one row per block position); the k-th occurrence sits at
        // `offset + k`, so consecutive rows of one seq are consecutive
        // positions — exactly the per-token decode stream, batched.
        let decode_pos: Vec<usize> = {
            let mut seen: HashMap<usize, usize> = HashMap::new();
            decode_seqs
                .iter()
                .map(|&s| {
                    let k = seen.entry(s).or_insert(0);
                    let p = session.sequence_offset(s).unwrap_or(0) + *k;
                    *k += 1;
                    p
                })
                .collect()
        };
        let prefill_base: Vec<usize> = prefill_seqs
            .iter()
            .map(|&s| session.sequence_offset(s).unwrap_or(0))
            .collect();

        // Slide each writer sequence's sliding-window RING before sizing the
        // arena for this wave: free the front chunks that have fully exited the
        // `window_size` window ending at the sequence's EARLIEST query position
        // in this wave. Decode has one query (`decode_pos`); a batched prefill
        // spans `[base, base+len)`, whose earliest row (`base`) has the
        // furthest-back window — evicting on `base` (the MIN, never the latest)
        // is what keeps a prefill row from losing a window key it still needs.
        // Glue rows scatter into reserved gap chunks and must not touch their
        // block tables, so they never evict. Positions stay ABSOLUTE across the
        // slide (the freed count folds into each backing's `base_pos`, seeding
        // the serialised rope), so attention is unchanged and stays consistent
        // with the absolute-positioned compressed corpus. All layers evict
        // identically (lockstep chunk layout), so the evicted count is uniform;
        // `resident = absolute − base_pos` is what the ARENA (set_len /
        // write_contiguous / decode metadata) is addressed by, while q_pos, the
        // fresh diagonal base, and the corpus stay absolute. `base_pos` is 0
        // until a sequence first exceeds `window_size`, so short sequences are
        // byte-identical to the never-evicted path.
        let window = cfg.window_size;
        let mut decode_base = vec![0u32; decode_seqs.len()];
        {
            // Evict once per SEQUENCE at its earliest row (the first
            // occurrence — occurrences are consecutive positions), exactly the
            // MIN-position rule the prefill span uses; later rows of the same
            // seq reuse the slid base.
            let mut base_of: HashMap<usize, u32> = HashMap::new();
            for (i, (&s, &pos)) in decode_seqs.iter().zip(&decode_pos).enumerate() {
                let bp = match base_of.get(&s) {
                    Some(&bp) => bp,
                    None => {
                        let mut bp = 0u32;
                        for backing in session.backings() {
                            bp = backing.evict_window_front(s, window, pos)?;
                        }
                        base_of.insert(s, bp);
                        bp
                    }
                };
                decode_base[i] = bp;
            }
        }
        let mut prefill_base_ev = vec![0u32; prefill_seqs.len()];
        for (pi, (&s, &base)) in prefill_seqs.iter().zip(&prefill_base).enumerate() {
            let mut bp = 0u32;
            for backing in session.backings() {
                bp = backing.evict_window_front(s, window, base)?;
            }
            prefill_base_ev[pi] = bp;
        }
        // Resident (arena) offsets = absolute − evicted-front. Equal to the
        // absolute values until the ring first slides.
        let decode_resident: Vec<usize> = decode_pos
            .iter()
            .zip(&decode_base)
            .map(|(&p, &b)| p - b as usize)
            .collect();
        let prefill_resident: Vec<usize> = prefill_base
            .iter()
            .zip(&prefill_base_ev)
            .map(|(&b, &e)| b - e as usize)
            .collect();

        // Per-seq corpus state (reset on slot reuse).
        for (&s, &base) in prefill_seqs.iter().zip(&prefill_base) {
            self.ensure_seq_state(s, base)?;
        }
        for &s in decode_seqs.iter().chain(glue_seqs) {
            // A decode row is normally mid-sequence (never reset) — but the
            // driver folds 1-token prefills into the decode group, and a FRESH
            // sequence rerouted that way must still reset its corpus state,
            // exactly as the prefill path it left would have. The session
            // offset tells the two apart.
            let start = session.sequence_offset(s).unwrap_or(0);
            self.ensure_seq_state(s, if start == 0 { 0 } else { 1 })?;
        }

        // Residual: fresh waves embed all rows [decode… | prefill… | glue…]
        // flat and expand into the mHC stream; resumed waves reuse it.
        let total_rows: usize = decode_seqs.len()
            + prefill_lens.iter().sum::<usize>()
            + glue_lens.iter().sum::<usize>();
        let mut h = match x_in {
            Some(resume) => {
                // Gathered into a stream of this wave's own, in its row order,
                // leaving what it was gathered from as it was; then the mHC axes.
                let buf =
                    Tensor::empty((total_rows, resume.width()), DType::F32, e.engine_device())?;
                resume.gather_into(&buf)?;
                buf.reshape((1, total_rows, cfg.hc_mult, cfg.dim))?
            }
            None => {
                let mut flat_ids: Vec<u32> = decode_ids.clone();
                for ids in prefill_ids.iter().chain(&glue_ids) {
                    flat_ids.extend_from_slice(ids);
                }
                let rows = self.embed_rows(&flat_ids)?;
                hc.expand(&rows)?
            }
        };

        // `verify_seqs` was read in the prologue (it also shaped the row
        // expansion). Verifying rows' attention runs the DECODE kernel over
        // per-position virtual slots with the rows pre-scattered, the per-row
        // capture stashes their projected compressor rows (rollback replay
        // source), and the head scores every decode-class row as it always
        // does.
        // A MIXED wave leads with plain decode rows (live slots, on-device
        // write-len commit, fused scatter) and ends with the verify blocks'
        // virtual rows (throwaway snapshot headers, pre-scattered) —
        // `verify_blocks` builds the rows in exactly that order, so a single
        // prefix bound (`n_plain_rows`) routes the kernel per row.
        let is_verify_wave = !verify_seqs.is_empty();
        let n_plain_rows = decode_seqs
            .iter()
            .position(|s| verify_seqs.contains(s))
            .unwrap_or(decode_seqs.len());
        if is_verify_wave
            && (!prefill_seqs.is_empty()
                || decode_seqs[n_plain_rows..]
                    .iter()
                    .any(|s| !verify_seqs.contains(s)))
        {
            candle::bail!(
                "verify wave: decode rows {:?} must be a plain prefix then a verify tail \
                 (verify set {:?})",
                decode_seqs,
                verify_seqs
            );
        }

        // Metadata: commit each layer's CPU chunk usage to the committed
        // prefix, then serialize slot headers for decode ++ prefill ++ glue
        // sequences (glue tables are already the scheduler-reserved state —
        // no usage commit). Prefill sequences get ONE HEADER SNAPSHOT PER
        // TOKEN: prompt token `t` absorbs through the SAME per-token decode
        // step the decode rows use, so its launch needs headers serialized at
        // offset `base + t`. This is the ONLY batched absorption that
        // bit-matches per-token decode: the decode kernel bf16-STAGES the
        // current token's PV (diagonal), while a settled-slot prefill would
        // read the diagonal as FP8 from the arena — a lossy difference that
        // heavy early-layer self-attention amplifies into CSA selection flips
        // (measured: garbage output). A single-launch batched prefill that
        // reproduces the decode diagonal needs a kernel-level bf16 diagonal
        // source (§L optimization note); until then, per-token is correct.
        // Snapshot `t` covers every layer; non-prefill slots serialize
        // identically in each snapshot, so snapshot 0 serves decode + glue.
        let all_seqs: Vec<usize> = decode_seqs
            .iter()
            .chain(prefill_seqs)
            .chain(glue_seqs)
            .copied()
            .collect();
        // Verify groups: (seq, resident-at-first-row, block_len), one per
        // sequence, from its CONSECUTIVE decode rows in the verify tail.
        // Empty on plain waves.
        let verify_groups: Vec<(usize, usize, usize)> = if is_verify_wave {
            let mut gs: Vec<(usize, usize, usize)> = Vec::new();
            for (i, &s) in decode_seqs.iter().enumerate().skip(n_plain_rows) {
                match gs.last_mut() {
                    Some(g) if g.0 == s => g.2 += 1,
                    _ => gs.push((s, decode_resident[i], 1)),
                }
            }
            gs
        } else {
            Vec::new()
        };
        // Per-group first-row offset into the wave's decode rows (GLOBAL row
        // index — the plain prefix precedes the verify tail).
        let verify_row_start: Vec<usize> = {
            let mut starts = Vec::with_capacity(verify_groups.len());
            let mut off = n_plain_rows;
            for &(_, _, s_len) in &verify_groups {
                starts.push(off);
                off += s_len;
            }
            starts
        };
        for backing in session.backings() {
            // Commit each sequence's prefix at its FIRST row's resident offset
            // — a verify block's later rows are the same seq at +k, and
            // committing those would advance the prefix over uncommitted
            // draft positions.
            let mut committed: HashSet<usize> = HashSet::new();
            for (&s, &resident) in decode_seqs.iter().zip(&decode_resident) {
                if committed.insert(s) {
                    backing.set_len(s, resident);
                }
            }
        }
        let s_meta = span("deepseek:wave_metadata");
        // The wave's pinned-arena guard. Everything staged from it stays valid
        // until this drops (line ~2426), and the arena resets then — freeing is
        // by lifetime, and nothing here syncs.
        //
        // ARENA BUDGET. Two draws, both bump-allocated:
        //   * the paged metadata below (slice + header snapshots), the original
        //     consumer;
        //   * the descriptor tables every batched kernel of a compression layer
        //     reads — the two compressor pools, the gallery-append scatter, the
        //     corpus gather, the BDP recall, the selector, the compressed-index
        //     expansion, the arena writeback and the prefill slot metadata —
        //     sized by `desc::wave_desc_bytes` for THIS wave's width.
        // The second is the addition this campaign makes, so it is computed and
        // checked against the arena rather than assumed: at 64 slots × 43 layers
        // it is a few hundred kilobytes against a 128 MB arena, but a future
        // width, ratio or prompt length should trip the assert rather than
        // silently start allocating overflow arenas mid-wave.
        let desc_bytes = desc::wave_desc_bytes(
            decode_seqs.len() + prefill_seqs.len(),
            self.engine.layer_count(),
            self.engine
                .cfg()
                .compress_ratios
                .iter()
                .copied()
                .max()
                .unwrap_or(1),
            // The per-token draws are sized by the LONGEST slot in the wave —
            // a decode slot contributes one token, a prompt its whole length.
            prefill_lens
                .iter()
                .chain(glue_lens.iter())
                .copied()
                .max()
                .unwrap_or(1)
                .max(1),
        );
        debug_assert!(
            desc_bytes < desc::STAGER_ARENA_BYTES / 2,
            "wave descriptor tables draw {desc_bytes} B of the {} B stager arena — the paged \
             metadata shares it, so re-check the arena size",
            desc::STAGER_ARENA_BYTES
        );
        // The wave driver's stager generation — one arena guard for the whole
        // wave (it opened before this sweep and drops after the logits are
        // copied off, so everything staged below outlives every kernel that
        // reads it).
        let generation = wave_generation;
        // Glue-only sequences never decode-write (their latents scatter into
        // reserved gap chunks) — excluding them from the metadata build's
        // write-chunk ensure keeps their block tables untouched, so the
        // scheduler's turn-start block accounting stays exact.
        let non_writer: Vec<usize> = glue_seqs
            .iter()
            .copied()
            .filter(|s| !decode_seqs.contains(s) && !prefill_seqs.contains(s))
            .collect();
        // Prefill absorbs each prompt in ONE `paged_latent_prefill` launch per
        // layer (the per-token decode-step loop it replaces was the dominant
        // wall-clock cost — see the wave profile). So each prefill seq needs a
        // SINGLE header snapshot at its committed-prefix length `base` (the arena
        // walk covers `[0,base)`; the fresh bf16 latents cover `[base,base+s)`),
        // with its whole write-back range ensured up-front. Decode/glue seqs
        // serialize their live offsets in the same snapshot.
        // Metadata offsets are RESIDENT (arena-relative). Decode seqs must be
        // listed explicitly — `build_decode_metadata_at` otherwise falls back to
        // the session's ABSOLUTE offset, which overshoots the arena once the ring
        // has slid. Equal to absolute until the first eviction.
        let mut overrides: Vec<(usize, usize)> =
            Vec::with_capacity(prefill_seqs.len() + decode_seqs.len());
        // Verify sequences are skipped here: their override is the BLOCK
        // length (`resident + s_len`, pushed with the group commit below), and
        // `build_decode_metadata_at` takes the FIRST match per seq.
        for (&s, &resident) in decode_seqs.iter().zip(&decode_resident) {
            if !verify_seqs.contains(&s) {
                overrides.push((s, resident));
            }
        }
        for (pi, &s) in prefill_seqs.iter().enumerate() {
            let resident = prefill_resident[pi];
            // Allocate the WHOLE prompt's write chunks up front — this extends the
            // block table over `[resident, resident+s_len)` WITHOUT moving the
            // writer/committed prefix (`set_len` stays `resident`), so the header
            // snapshot built below covers every write chunk. The post-launch arena
            // writeback (`paged_latent_glue_scatter`) addresses those chunks
            // through that immutable snapshot, so they MUST be present before it is
            // taken (a chunk allocated after the snapshot is invisible to the
            // scatter → OOB). The header the attention reads is unchanged: extra
            // trailing chunks past `resident` are never read (window/pos-bounded).
            for backing in session.backings() {
                backing.ensure_for_offset(s, resident, prefill_lens[pi])?;
                backing.set_len(s, resident);
            }
            overrides.push((s, resident));
        }
        // DECODE rows always serialize their slot state through the LIVE
        // persistent gpu_chunks buffer (the cheap path Qwen/Llama use): a decode
        // row's write chunk is pre-ensured, so it never reallocs during the layer
        // loop, and the decode kernel commits its write-len on-device
        // (`commit_write_len=true`) so the buffer advances for the next step. This
        // skips the per-layer host snapshot COPY that dominated `wave_metadata`,
        // for EVERY wave — pure-decode AND mixed. Only rows that mutate the arena
        // mid-forward — PREFILL (absorbs across chunk boundaries in one launch)
        // and GLUE (gap-chunk scatter) — still snapshot, so their `slices_ptr`
        // survives the reallocation. Bit-exactness of the live vs snapshot decode
        // path across chunk boundaries is gated by
        // `decode_live_buffer_matches_snapshot_multistep`.
        // Speculative-verify: capacity-ensure every block's whole write range
        // BEFORE the standard header build — `ensure_for_offset` above uses
        // positional block math, which under-allocates on a slid ring (chunk
        // count exceeds positional blocks while the writer tail lacks free
        // slots), and a chunk allocated after the snapshot is invisible to the
        // writeback scatter (→ OOB).
        if is_verify_wave {
            for &(vseq, resident, s_len) in &verify_groups {
                for backing in session.backings() {
                    backing.ensure_for_batch_entries(&[(vseq, resident)], s_len)?;
                }
            }
        }
        // Speculative-verify rows run the DECODE kernel over one virtual slot
        // per block position — decode numerics by construction (fp PV,
        // read-time rope, decode softmax structure), vs the prefill kernel's
        // int8-PV envelope that flips narrow-margin argmaxes. All of a block's
        // virtual slots share IDENTICAL headers serialized with the write
        // length committed over the whole block (the rows are host-written
        // before each layer's launch); per-slot causal visibility comes from
        // the kernel's `key_pos <= q_pos` bound alone.
        //
        // Commit every block's write length BEFORE the (single) header build:
        // the virtual-slot headers must expose the block rows physically, and
        // each position map must cover [0, resident+s_len). `set_len` never
        // touches the serialized slot buffer, so `mark_decode_writer_stale`
        // marks it and the header build's sync re-serialises the whole writer
        // region — every chunk from the writer boundary to the writer, not the
        // writer alone. A block that spanned into a fresh chunk needs that: an
        // EARLIER wave's metadata build validated the buffer at pre-`set_len`
        // lengths, so a writer-only patch would leave the spanned block's
        // earlier rows reading short through the stale predecessor slice (an
        // acceptance collapse to 1.4 tok/step with a lossless-assert kill).
        //
        // (Each block's write range was capacity-ensured ABOVE, so the
        // writeback snapshot covers every chunk `set_len` fills here.)
        //
        // Every block's length is set on a backing before that backing is
        // marked once for all of them: one state lock per backing, not one per
        // verify group per backing.
        if is_verify_wave {
            let entries: Vec<(usize, usize)> = verify_groups
                .iter()
                .map(|&(vseq, _, _)| (vseq, 0))
                .collect();
            for backing in session.backings() {
                for &(vseq, resident, s_len) in &verify_groups {
                    backing.set_len(vseq, resident + s_len);
                }
                backing.mark_decode_writer_stale(&entries)?;
            }
            for &(vseq, resident, s_len) in &verify_groups {
                overrides.push((vseq, resident + s_len));
            }
        }
        // ONE metadata build for the whole wave — plain decode rows serialize
        // through the LIVE slot buffer (cheap reuse pointer), verify rows get
        // immutable per-row snapshots of the freshly committed block state
        // (duplicate (seq, offset) rows share one snapshot copy), prefill/glue
        // as always. Header slot i pairs with decode row i in the single
        // launch below; prefill/glue headers follow the decode rows.
        let snapshot_seqs: Vec<usize> = prefill_seqs
            .iter()
            .chain(glue_seqs)
            .chain(verify_seqs.iter())
            .copied()
            .collect();
        let std_meta = {
            let (headers, stride) = session.build_decode_metadata_at(
                0..session.num_layers(),
                &all_seqs,
                generation,
                &overrides,
                &non_writer,
                &snapshot_seqs,
            )?;
            let headers = headers.ok_or_else(|| candle::Error::Msg("no decode metadata".into()))?;
            (headers, stride)
        };
        let hdr_of = |layer: usize, seq_slot: usize| -> u64 {
            let (headers, stride) = &std_meta;
            headers.dev_ptr() + (layer as u64) * stride + (seq_slot * SLOT_HEADER_BYTES) as u64
        };
        // Per-group scatter header: a group's positions share IDENTICAL
        // headers (committed block write length), so its FIRST row's header
        // carries the chunk records the row scatter addresses through.
        let verify_scatter_hdr =
            |layer: usize, gi: usize| -> u64 { hdr_of(layer, verify_row_start[gi]) };
        s_meta.end();

        let mut state = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;

        // Wave-invariant decode query positions: built ONCE here instead of rebuilding
        // the `[n_dec]` u32 upload inside every layer (the positions are fixed for the
        // whole wave). `None` when the wave carries no decode rows.
        let q_pos_dec_t = if decode_seqs.is_empty() {
            None
        } else {
            Some(Tensor::from_vec(
                decode_pos.iter().map(|&p| p as u32).collect::<Vec<u32>>(),
                decode_seqs.len(),
                h.device(),
            )?)
        };
        // Wave-invariant prefill query positions (`base..base+s_len` per prefill
        // seq, packed in prefill order): hoisted for the same reason, and built
        // as ONE upload — every layer's prefill launch reads this same buffer.
        let prefill_q_pos: Option<Tensor> = if prefill_seqs.is_empty() {
            None
        } else {
            let pos: Vec<u32> = prefill_base
                .iter()
                .zip(&prefill_lens)
                .flat_map(|(&base, &s_len)| base as u32..(base + s_len) as u32)
                .collect();
            let n = pos.len();
            Some(Tensor::from_vec(pos, n, h.device())?)
        };
        // The MoE token-id list is identical across layers; assemble and upload
        // it once, so every layer's router reads the same device buffer.
        let flat_ids: Vec<u32> = decode_ids
            .iter()
            .copied()
            .chain(prefill_ids.iter().chain(&glue_ids).flatten().copied())
            .collect();
        let n_ids = flat_ids.len();
        let flat_ids = Tensor::from_vec(flat_ids, n_ids, h.device())?;

        // DSpark target-feature capture (paper Eq. 2): when a drafter is attached and this wave runs
        // through the head (full stack), stash `head_reduce(h)` after each real target layer — the
        // per-layer hidden `fc`/`Wc` consumes. Keyed by layer so the head can order them by
        // `target_layers` and substitute the post-output-norm hidden for the `n_layers` sentinel.
        let capture_targets = layer_end == n_layers && !self.target_layers.is_empty();
        let mut target_reduced: Vec<(usize, Tensor)> = Vec::with_capacity(self.target_layers.len());

        for l in layer_start..layer_end {
            let layer = e.engine_layer(l);
            let a = &layer.attn;
            let st = &self.layer_static[l];
            let rope = e.rope_for(l);
            // Device handle for the sync-bracketed fine-span attribution below
            // (cheap Arc clone; only the sync arms compile in under `profile`).
            let dev = h.device().clone();

            // Attention sub-block.
            let s_attn_pre = span("deepseek:hc_pre_norm");
            let s_hcpre = span("attn:hc_pre");
            let (x, post, comb) = hc.pre(&h, &layer.hc_attn)?;
            s_hcpre.end();
            let s_anorm = span("attn:norm");
            let x = rms_norm(&x, &layer.attn_norm, cfg.norm_eps)?;
            s_anorm.end();
            s_attn_pre.end();

            // Phase A — glue scatter FIRST: every glue row's latent lands in
            // its reserved gap chunk before ANY attention pass of this layer
            // reads the arena (stream-ordered; glue keys then read like any
            // window key — no double-source, no garbage gaps).
            let glue_row_base = decode_seqs.len() + prefill_lens.iter().sum::<usize>();
            let mut glue_proj: Vec<(Tensor, Tensor)> = Vec::with_capacity(glue_seqs.len());
            let s_glue_scatter = span_if(!glue_seqs.is_empty(), "deepseek:glue_scatter");
            {
                let mut cursor = glue_row_base;
                // Every island's scatter collected first, then ONE launch: each
                // run names its own slot, so there is nothing to serialise.
                let mut runs: Vec<GlueRun> = Vec::with_capacity(glue_seqs.len());
                for (gi, _seq) in glue_seqs.iter().enumerate() {
                    let g_len = glue_lens[gi];
                    let xs = x.narrow(1, cursor, g_len)?.to_dtype(DType::F32)?;
                    let qr = rms_norm(&a.wq_a().forward(&xs)?, a.q_norm(), a.eps())?;
                    let kv = rms_norm(&a.wkv().forward(&xs)?, a.kv_norm(), a.eps())?;
                    let kv_bf = kv.reshape((g_len, a.head_dim()))?.to_dtype(DType::BF16)?;
                    let d = &glue_desc[gi];
                    runs.push(GlueRun {
                        kv: kv_bf,
                        headers_ptr: hdr_of(l, decode_seqs.len() + prefill_seqs.len() + gi),
                        slices: d.write_slice.clone(),
                        in_blk: d.write_in_blk.clone(),
                    });
                    glue_proj.push((xs, qr));
                    cursor += g_len;
                }
                super::super::paged::paged_latent_glue_scatter(&runs, generation)?;
            }
            drop(s_glue_scatter);

            let mut attn_rows: Vec<Tensor> = Vec::with_capacity(total_rows);
            // Decode rows: one kernel step per sequence.
            // Decode rows: all sessions attend in ONE `paged_latent_decode`
            // launch over every decode slot (grid.x = n_decode), instead of a
            // per-session launch loop. Per session we run the same host
            // projections + on-device corpus select/gather, then concatenate the
            // gathered compressed blocks — each slot's selection is the dense
            // range `[offset, offset+k)` into the concat, so per-session galleries
            // stay isolated with no cross-bleed and no GID readback.
            if !decode_seqs.is_empty() {
                let s_decode = span("deepseek:decode_attn");
                let s_dprep = span("decode:prep");
                let n_dec = decode_seqs.len();
                let (h, hd) = (a.n_heads(), a.head_dim());

                // Batched attention projections: ONE GEMM each over ALL decode
                // rows (the decode rows are the first `n_dec` of `x`), replacing
                // the per-session projection GEMVs. Bit-identical per row (matmul
                // + last-dim norms are row-independent).
                let s_proj = span("dprep:proj");
                let xs_dec = x.narrow(1, 0, n_dec)?.to_dtype(DType::F32)?; // [1,n_dec,dim]
                                                                           // wq_a and wkv share `xs_dec`; quantize the activation once for both.
                let (qa_raw, kv_raw) = shared_int8_pair(&xs_dec, a.wq_a(), a.wkv())?;
                let qr_all = rms_norm(&qa_raw, a.q_norm(), a.eps())?; // [1,n_dec,qa]
                let q_all = a.wq_b().forward(&qr_all)?.reshape((1, n_dec, h, hd))?;
                let q_all = a.rms_scale(&q_all)?;
                let q_bf_all = q_all.reshape((n_dec, h, hd))?.to_dtype(DType::BF16)?; // [n_dec,h,hd]
                let kv_all = rms_norm(&kv_raw, a.kv_norm(), a.eps())?;
                let kv_bf_all = kv_all.reshape((n_dec, hd))?.to_dtype(DType::BF16)?; // [n_dec,hd]
                                                                                     // Batched compressor projections (shared layer weights) over all
                                                                                     // decode rows — the stateless part of the corpus push; each
                                                                                     // session's stateful pooling/emit then streams its pre-projected
                                                                                     // row (bit-identical to the per-session `push_raw`/`push`).
                let comp_proj = match a.compressor() {
                    Some(c) => Some(c.project_rows(&xs_dec)?), // (kv[n_dec,cd], score[n_dec,cd])
                    None => None,
                };
                let icomp_proj = match a.indexer() {
                    Some(ix) => Some(ix.compressor().project_rows(&xs_dec)?),
                    None => None,
                };
                // Batched indexer query projection AND rope (CSA layers have an
                // indexer): one GEMM over all decode rows for `wq_b` +
                // `weights_proj`, then ONE rope chain over all rows at their
                // per-row slot positions (`rope_query_at`) — the per-row
                // narrow/reshape/rope chain was ~5 launches × rows × CSA
                // layers of pure WDDM submission tax. Bit-identical per row to
                // `query_space`.
                let idx_query = match a.indexer() {
                    Some(ix) => {
                        let qr_2d = qr_all.reshape((n_dec, ()))?;
                        let xs_2d = xs_dec.reshape((n_dec, ()))?;
                        let (q_raw, weights) = ix.query_gemm_batched(&xs_2d, &qr_2d)?;
                        let pos32: Vec<u32> = decode_pos.iter().map(|&p| p as u32).collect();
                        let q_roped = ix.rope_query_at(&q_raw, rope, &pos32)?; // [n_dec,h,hd]
                        Some((ix, q_roped, weights))
                    }
                    None => None,
                };
                s_proj.end();

                // Pass 1a: per-session corpus ASSEMBLE — advance each session's
                // streaming compressor state with its pre-projected row and keep
                // the completed group's pool inputs, WITHOUT pooling. The pool
                // then runs once across the whole decode set in pass 1b.
                let mut asms: Vec<DecodeAssemble> = Vec::with_capacity(n_dec);
                // Verify waves: this layer's streaming state at every ROW
                // boundary (`Arc`-clone cheap), captured as the loop advances —
                // the rollback installs one of these directly on a partial
                // accept instead of replaying the accepted rows.
                let mut row_snaps: Vec<(
                    Option<super::super::compressor::CompressorState>,
                    Option<super::super::compressor::CompressorState>,
                    usize,
                )> = Vec::new();
                // Each row's corpus size AS OF ITSELF: this session's gallery
                // length entering the wave plus the groups its rows completed
                // THROUGH this one. It has to be counted here rather than read
                // back from the gallery, because pass 1b appends every row's
                // groups at once — so a `gallery.len()` taken afterwards would
                // hand an earlier row of a verify block the entries of a LATER
                // draft token. This keeps each row's view causal, exactly as the
                // per-row append it replaces did.
                let mut appended: HashMap<usize, usize> = HashMap::new();
                let mut row_entries: Vec<usize> = Vec::with_capacity(n_dec);
                let s_dasm = span("dprep:assemble_all");
                for (i, &seq) in decode_seqs.iter().enumerate() {
                    let comp_row = match &comp_proj {
                        Some((k, s)) => Some((k.narrow(0, i, 1)?, s.narrow(0, i, 1)?)),
                        None => None,
                    };
                    let icomp_row = match &icomp_proj {
                        Some((k, s)) => Some((k.narrow(0, i, 1)?, s.narrow(0, i, 1)?)),
                        None => None,
                    };
                    let entry = state.get_mut(&seq).expect("ensured above");
                    let asm = kernel_attn_decode_assemble(
                        &mut entry.layers[l],
                        comp_row.as_ref().map(|(k, s)| (k, s)),
                        icomp_row.as_ref().map(|(k, s)| (k, s)),
                    )?;
                    let n_new = asm.comp_gp.as_ref().map_or(0, |g| g.positions.len());
                    let done = appended.entry(seq).or_insert(0);
                    *done += n_new;
                    let ls = &entry.layers[l];
                    let n_entries = ls.gallery.as_ref().map_or(0, |g| g.len()) + *done;
                    row_entries.push(n_entries);
                    if is_verify_wave {
                        row_snaps.push((
                            ls.comp.as_ref().map(|c| c.state_snapshot()),
                            ls.icomp.as_ref().map(|c| c.state_snapshot()),
                            n_entries,
                        ));
                    }
                    asms.push(asm);
                    if l + 1 == n_layers {
                        entry.absorbed = decode_pos[i] + 1;
                    }
                }
                s_dasm.end();

                // Pass 1b: pool EVERY decode row's completed groups in ONE launch
                // per compressor family, then append. `comp` = attention entries
                // (pre-RoPE), `icomp` = indexer keys (roped); both share group
                // boundaries. The decode twin of the prefill pool batch, and
                // bit-identical per group to the per-row pool it replaces
                // (`pool_batched_across_seqs_matches_per_seq`).
                let s_dpool = span("dprep:pool");
                let comp_refs: Vec<Option<&GroupPool>> =
                    asms.iter().map(|m| m.comp_gp.as_ref()).collect();
                let icomp_refs: Vec<Option<&GroupPool>> =
                    asms.iter().map(|m| m.icomp_gp.as_ref()).collect();
                let comp_entries = match a.compressor() {
                    Some(c) => pool_across_seqs(c, &comp_refs, None, generation)?,
                    None => None,
                };
                let icomp_keys = match a.indexer() {
                    Some(ix) => {
                        pool_across_seqs(ix.compressor(), &icomp_refs, Some(rope), generation)?
                    }
                    None => None,
                };
                // Append every session's groups in ONE pass. The galleries are
                // pulled out of the state map together (they are distinct
                // sessions, so the borrows are disjoint) and handed over in slot
                // order.
                if let Some(entries) = comp_entries.as_ref() {
                    let mut by_seq: HashMap<usize, &mut FloatGallery> = state
                        .iter_mut()
                        .filter_map(|(s, e)| e.layers[l].gallery.as_mut().map(|g| (*s, g)))
                        .collect();
                    append_pooled(
                        entries,
                        icomp_keys.as_ref(),
                        &comp_refs,
                        decode_seqs,
                        &mut by_seq,
                        generation,
                    )?;
                }
                s_dpool.end();

                // Pass 1c: capture each row's selection intent against its OWN
                // corpus size (the selection itself batches over all sessions
                // below).
                let mut sels: Vec<DecodeSel> = Vec::with_capacity(n_dec);
                for (i, &entries) in row_entries.iter().enumerate().take(n_dec) {
                    // This slot's row of the batched, already-roped indexer
                    // query — bare views into the batched tensors.
                    let (q_idx_i, w_i) = match &idx_query {
                        Some((ix, q_roped, weights)) => {
                            let qi = q_roped
                                .narrow(0, i, 1)?
                                .reshape((ix.n_heads(), ix.head_dim()))?;
                            let wi = weights.narrow(0, i, 1)?.reshape(ix.n_heads())?;
                            (Some(qi), Some(wi))
                        }
                        None => (None, None),
                    };
                    sels.push(kernel_attn_decode_capture(a, entries, q_idx_i, w_i)?);
                }
                // Distribute this layer's row-boundary states into each block's
                // pre-verify snapshot (rows are grouped consecutively, in the
                // same order as `verify_groups`).
                if is_verify_wave {
                    let mut vs = self
                        .verify_snap
                        .write()
                        .map_err(|_| candle::Error::Msg("verify_snap lock poisoned".into()))?;
                    for (gi, &(vseq, _, s_len)) in verify_groups.iter().enumerate() {
                        let row0 = verify_row_start[gi];
                        if let Some(snap) = vs.get_mut(&vseq) {
                            if let Some(lsnap) = snap.layers.get_mut(l) {
                                lsnap.row_states = row_snaps[row0..row0 + s_len].to_vec();
                            }
                        }
                    }
                }
                s_dprep.end();

                // Batched selection: ONE launch per Stage-1 kernel over EVERY CSA
                // decode session's gallery, replacing the per-session two-stage
                // selection loop (whose `topm_select` — a single-warp serial bin
                // scan × sessions — dominated the decode selection cost). HCA
                // sessions attend all causal entries; empty/SWA select nothing.
                let s_dsel = span("decode:select");
                let mut csa_idx: Vec<usize> = Vec::new();
                let mut csa_gals: Vec<&FloatGallery> = Vec::new();
                let mut csa_q: Vec<Tensor> = Vec::new();
                let mut csa_w: Vec<Tensor> = Vec::new();
                for (i, &seq) in decode_seqs.iter().enumerate() {
                    if let DecodeSel::TwoStage { q_idx, weights } = &sels[i] {
                        let g = state.get(&seq).expect("ensured").layers[l]
                            .gallery
                            .as_ref()
                            .expect("CSA session has a gallery");
                        csa_idx.push(i);
                        csa_gals.push(g);
                        csa_q.push(q_idx.clone());
                        csa_w.push(weights.clone());
                    }
                }
                let mut sel_gids: Vec<Option<Tensor>> = (0..n_dec).map(|_| None).collect();
                let mut cnts: Vec<u32> = vec![0u32; n_dec];
                if !csa_gals.is_empty() {
                    let ix = a.indexer().expect("CSA layer has an indexer");
                    let batched = two_stage_select_batched(
                        &csa_gals,
                        &csa_q,
                        &csa_w,
                        shortlist_m(ix.top_k()),
                        ix.top_k(),
                        generation,
                    )?;
                    for (j, (gids, k)) in batched.into_iter().enumerate() {
                        let i = csa_idx[j];
                        cnts[i] = k as u32;
                        if k > 0 {
                            sel_gids[i] = Some(gids);
                        }
                    }
                }
                // ONE arange per layer, shared by every all-entries session as a
                // narrowed VIEW. Building it per session is an allocation plus a
                // launch each for what is the same ascending run every time —
                // with 16 sessions × 43 layers that is tens of thousands of
                // launches per run to materialise `0..n` repeatedly. The values
                // are identical, so the gather is bit-for-bit unchanged.
                let all_n: Vec<usize> = sels
                    .iter()
                    .filter_map(|s| match s {
                        DecodeSel::AllEntries(n) => Some(*n),
                        _ => None,
                    })
                    .collect();
                let all_ids = match all_n.iter().max() {
                    // `max(1)` keeps the allocation legal when every all-entries
                    // session is empty; the narrow below still yields 0 rows.
                    Some(&m) => Some(Tensor::arange(0u32, m.max(1) as u32, &dev)?),
                    None => None,
                };
                for (i, sel) in sels.iter().enumerate() {
                    if let DecodeSel::AllEntries(n) = *sel {
                        cnts[i] = n as u32;
                        sel_gids[i] = Some(
                            all_ids
                                .as_ref()
                                .expect("an all-entries session implies the shared arange")
                                .narrow(0, 0, n)?,
                        );
                    }
                }
                s_dsel.end();

                // Pass 2: gather EVERY session's selected HOT two-region rows into
                // one pre-allocated block in a SINGLE batched launch (each slot's
                // rows at its dense range `[offset, offset+k)`) — no per-region
                // `index_select`, no cross-session `cat`, no per-session launch.
                let s_dgather = span("decode:gather");
                let mut offsets: Vec<u32> = Vec::with_capacity(n_dec);
                let mut off = 0u32;
                for &c in cnts.iter().take(n_dec) {
                    offsets.push(off);
                    off += c;
                }
                let total_k = off as usize;
                let cache = if total_k == 0 {
                    st.empty_corpus_cache()?
                } else {
                    // Gather-kernel output block (total_k>0 here — the ==0 case
                    // took empty_corpus_cache above): every row written by the
                    // gather, so allocate uninit and skip the memset.
                    let out_nope = Tensor::empty((total_k, NOPE_DIM), DType::U8, &dev)?;
                    let out_scale = Tensor::empty((total_k, NOPE_BANDS), DType::F32, &dev)?;
                    let out_rope = Tensor::empty((total_k, ROPE_DIM), DType::BF16, &dev)?;
                    let out_pos = Tensor::empty(total_k, DType::U32, &dev)?;
                    let mut gg: Vec<&FloatGallery> = Vec::with_capacity(n_dec);
                    let mut ggids: Vec<Tensor> = Vec::with_capacity(n_dec);
                    let mut goff: Vec<u32> = Vec::with_capacity(n_dec);
                    for (i, &seq) in decode_seqs.iter().enumerate() {
                        if let Some(gids) = &sel_gids[i] {
                            let g = state.get(&seq).expect("ensured").layers[l]
                                .gallery
                                .as_ref()
                                .expect("selection implies a gallery");
                            gg.push(g);
                            ggids.push(gids.clone());
                            goff.push(offsets[i]);
                        }
                    }
                    gather_corpus_batched(
                        &gg, &ggids, &goff, &out_nope, &out_scale, &out_rope, &out_pos, generation,
                    )?;
                    super::super::paged::CorpusCache::from_gathered(
                        out_nope, out_scale, out_rope, out_pos, total_k,
                    )?
                };
                s_dgather.end();
                let s_dcache = span("decode:cache");
                // The batched projections already produced these contiguous
                // `[n_dec, …]` tensors; use them directly (the per-session
                // narrow-then-cat that rebuilt them was a redundant device copy).
                let q_all = q_bf_all; // [n_dec, H, hd]
                let kv_all = kv_bf_all; // [n_dec, hd]
                let max_sel = cnts.iter().map(|&c| c as usize).max().unwrap_or(0).max(1);
                // Each slot's selection is the dense range [offset, offset+k) —
                // strictly ascending, the compressed-index contract the kernel
                // documents — so the whole [n_dec, max_sel] matrix follows from
                // two numbers per slot. ONE launch expands them, reading the
                // table straight out of the wave's pinned arena; `comp_cnt`
                // falls out of the same pass because the decode kernel wants the
                // counts as a device array too.
                let (comp_idx, comp_cnt) =
                    super::super::comp_idx::build(&offsets, &cnts, max_sel, &dev, generation)?;
                // Explicit per-slot query position (the decode kernel no longer
                // derives it from the writer slice, so the windowless slot works
                // and the compressed causal guard has a reference). Hoisted
                // above the layer loop — wave-fixed.
                let q_pos_dec = q_pos_dec_t
                    .as_ref()
                    .expect("decode branch runs only when the wave has decode rows");
                s_dcache.end();
                // `cache` is the gathered two-region hot cache (built above from
                // the gallery's pre-built int8 — no per-wave rebuild).
                let s_dkern = span("decode:kernel");
                // Speculative-verify tail rows are VIRTUAL decode slots: their
                // rows are written to the arena FIRST (same `store_band_elem`
                // encode the fused scatter uses → byte-identical read-back) so
                // later positions read earlier ones; causal visibility is the
                // kernel's `key_pos <= q_pos` bound; the shared per-group
                // headers carry the pre-committed block write length. Each
                // row's RESIDENT offset maps to its (slice, in_blk) by WALKING
                // the chunk table — positional `off/32` math shears once the
                // sliding ring has slid.
                if is_verify_wave {
                    let mut runs: Vec<GlueRun> = Vec::with_capacity(verify_groups.len());
                    for (gi, &(vseq, base_resident, s_len)) in verify_groups.iter().enumerate() {
                        let row0 = verify_row_start[gi];
                        let (wslice, wblk): (Vec<u32>, Vec<u32>) = {
                            let chunks = session.backings()[l]
                                .live_chunks_as_sealed(vseq)
                                .unwrap_or_default();
                            let mut map: Vec<(u32, u32)> = Vec::with_capacity(s_len);
                            let mut cum = 0usize;
                            for (si, c) in chunks.iter().enumerate() {
                                let cnt = c.token_count as usize;
                                for w in 0..cnt {
                                    let r = cum + w;
                                    if r >= base_resident && r < base_resident + s_len {
                                        map.push((si as u32, c.offset as u32 + w as u32));
                                    }
                                }
                                cum += cnt;
                            }
                            if map.len() != s_len {
                                candle::bail!(
                                    "verify writeback: chunk walk covered {} of {} block rows \
                                     (seq {vseq}, base_resident {base_resident}, {} chunks)",
                                    map.len(),
                                    s_len,
                                    chunks.len()
                                );
                            }
                            map.into_iter().unzip()
                        };
                        runs.push(GlueRun {
                            kv: kv_all.narrow(0, row0, s_len)?,
                            headers_ptr: verify_scatter_hdr(l, gi),
                            slices: wslice,
                            in_blk: wblk,
                        });
                    }
                    super::super::paged::paged_latent_glue_scatter(&runs, generation)?;
                }
                // ONE launch over plain AND verify rows. The plain prefix
                // (`n_plain_rows`) uses live persistent slot buffers — the
                // kernel fused-scatters each of its tokens and commits its
                // write-len on-device for the next step; the verify tail's
                // throwaway snapshot headers do neither (rows pre-written
                // above, lengths patched host-side).
                let out = super::super::paged::paged_latent_decode_raw(
                    &q_all,
                    hdr_of(l, 0),
                    &kv_all,
                    &cache,
                    &comp_idx,
                    &comp_cnt,
                    q_pos_dec,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    0,
                    n_plain_rows,
                    n_plain_rows,
                    st.ws(),
                    None,
                )?;
                s_dkern.end();
                // Batched output projection: ONE `output_proj` over all decode
                // rows (`b = n_dec`) instead of a per-session call. `output_proj`
                // is already batch-parametrized — its inner loop is over the 8
                // o_lora groups, not sessions — so this is bit-identical per row
                // (the group GEMMs are row-independent) and collapses `8·n_dec`
                // group-GEMM launches to 8.
                let s_doutp = span("decode:outproj");
                // Kernel output is token-major [n_dec, h, hd]; `output_proj` takes
                // [b, s, h, hd] (here b=n_dec, s=1).
                let o = out.reshape((n_dec, 1, h, hd))?; // kernel emits F32
                let proj = a.output_proj(&o, n_dec, 1)?; // [n_dec, 1, dim]
                                                         // One reshaped view instead of `n_dec` narrow slices — the rows are
                                                         // concatenated along axis 1 below exactly as the prefill/glue rows are,
                                                         // so `[1, n_dec, dim]` is bit-identical and skips the narrow round-trip.
                attn_rows.push(proj.reshape((1, n_dec, ()))?);
                s_doutp.end();
                s_decode.end();
            }
            // Prefill rows: each prompt is absorbed in ONE batched
            // `paged_latent_prefill` launch per layer (argmax-equal to per-token
            // decode absorption, validated by `wave_prefill_state_matches_decode_steps`),
            // which replaced the per-token launch loop that dominated the profile.
            let s_prefill = span_if(!prefill_seqs.is_empty(), "deepseek:prefill_attn");
            // ── Prefill projections: ONE batched projection over the WHOLE prompt
            // span (all sequences' rows), exactly as decode's `dprep` batches over
            // all decode rows — one `shared_int8_pair`/`wq_b`/`project_rows` instead
            // of one set per sequence. Row-independent ⇒ bit-identical.
            let s_pprep = span("prefill:prep");
            let prefill_total: usize = prefill_lens.iter().sum();
            let proj = if prefill_total == 0 {
                None
            } else {
                let xs_all = x.narrow(1, decode_seqs.len(), prefill_total)?;
                Some(
                    super::super::kernel_attention::kernel_attn_prefill_project_batched(
                        a, &xs_all,
                    )?,
                )
            };
            // ── Prefill pass 1: the compressor ASSEMBLE (state advance +
            // deferred pool inputs), FLEET-BATCHED. Three phases: (1) per-seq
            // state snapshots (Arc clones, no device work); (2) ONE batched
            // assemble per compressor family — ~a dozen device ops for the
            // whole fleet where the per-seq loop issued ~15 PER SEQUENCE
            // (measured at width 20: 15.7s of a 29.8s prefill wall, every
            // enqueue stalling under WDDM submission pressure); (3) per-seq
            // state install + PrefillPrep assembly (host math + free views).
            let mut preps: Vec<PrefillPrep> = Vec::with_capacity(prefill_seqs.len());
            if !prefill_seqs.is_empty() {
                let p = proj.as_ref().expect("prefill rows imply a projection");
                let s_asm = span("pprep:assemble");
                // Phase 1 — collect. Compressor presence is LAYER-uniform
                // (every seq's layer state is built from the same layer kind),
                // so the fleet either all contribute or none do.
                let mut comp_ins: Vec<SeqAssemble> = Vec::with_capacity(prefill_seqs.len());
                let mut icomp_ins: Vec<SeqAssemble> = Vec::with_capacity(prefill_seqs.len());
                let mut l0s: Vec<usize> = Vec::with_capacity(prefill_seqs.len());
                let mut bases: Vec<usize> = Vec::with_capacity(prefill_seqs.len());
                let mut ratio_comp = 1usize;
                let mut off = 0usize;
                for (pi, &seq) in prefill_seqs.iter().enumerate() {
                    let s_len = prefill_lens[pi];
                    let ls = &state.get(&seq).expect("ensured above").layers[l];
                    if let (Some(c), Some((k, sc))) = (&ls.comp, &p.comp_proj) {
                        comp_ins.push(c.assemble_input(
                            &k.narrow(0, off, s_len)?,
                            &sc.narrow(0, off, s_len)?,
                        )?);
                        ratio_comp = c.ratio();
                    }
                    if let (Some(ic), Some((k, sc))) = (&ls.icomp, &p.icomp_proj) {
                        icomp_ins.push(ic.assemble_input(
                            &k.narrow(0, off, s_len)?,
                            &sc.narrow(0, off, s_len)?,
                        )?);
                    }
                    l0s.push(ls.comp.as_ref().map_or(0, |c| c.buffered_len()));
                    bases.push(ls.gallery.as_ref().map_or(0, |g| g.len()));
                    off += s_len;
                }
                // Phase 2 — fleet-wide assemble per compressor family.
                let comp_outs = if comp_ins.len() == prefill_seqs.len() {
                    let template = state.get(&prefill_seqs[0]).expect("ensured above").layers[l]
                        .comp
                        .as_ref()
                        .expect("phase 1 collected a comp input for every seq");
                    Some(assemble_groups_batched(template, &comp_ins)?)
                } else {
                    None
                };
                let icomp_outs = if icomp_ins.len() == prefill_seqs.len() {
                    let template = state.get(&prefill_seqs[0]).expect("ensured above").layers[l]
                        .icomp
                        .as_ref()
                        .expect("phase 1 collected an icomp input for every seq");
                    Some(assemble_groups_batched(template, &icomp_ins)?)
                } else {
                    None
                };
                // Phase 3 — install state + build each seq's PrefillPrep.
                let mut comp_outs = comp_outs.map(|v| v.into_iter());
                let mut icomp_outs = icomp_outs.map(|v| v.into_iter());
                let mut off = 0usize;
                for (pi, &seq) in prefill_seqs.iter().enumerate() {
                    let s_len = prefill_lens[pi];
                    let e = state.get_mut(&seq).expect("ensured above");
                    let ls = &mut e.layers[l];
                    let comp_gp = match comp_outs.as_mut().map(|it| it.next()) {
                        Some(Some((gp, upd))) => {
                            ls.comp
                                .as_mut()
                                .expect("comp outputs imply a compressor")
                                .assemble_apply(upd);
                            gp
                        }
                        _ => None,
                    };
                    let icomp_gp = match icomp_outs.as_mut().map(|it| it.next()) {
                        Some(Some((gp, upd))) => {
                            ls.icomp
                                .as_mut()
                                .expect("icomp outputs imply an indexer compressor")
                                .assemble_apply(upd);
                            gp
                        }
                        _ => None,
                    };
                    // The shared-boundary contract binds comp and icomp ONLY
                    // where both exist (CSA layers): an HCA layer has a
                    // compressor but no indexer, so its icomp side is always
                    // absent and only the comp pool exists.
                    debug_assert!(
                        ls.icomp.is_none()
                            || comp_gp.as_ref().map(|g| &g.positions)
                                == icomp_gp.as_ref().map(|g| &g.positions),
                        "comp/icomp group boundaries diverged (seq {seq}): {:?} vs {:?}",
                        comp_gp.as_ref().map(|g| &g.positions),
                        icomp_gp.as_ref().map(|g| &g.positions),
                    );
                    let g_total = comp_gp.as_ref().map_or(0, |g| g.positions.len());
                    // Each token sees the entries present before this prefill
                    // plus the groups completed through it; bounding the select
                    // to that prefix reproduces the per-token incremental
                    // gallery exactly.
                    let n_visible: Vec<usize> = (0..s_len)
                        .map(|t| bases[pi] + ((l0s[pi] + t + 1) / ratio_comp).min(g_total))
                        .collect();
                    preps.push(PrefillPrep {
                        kv_bf: p.kv_bf.narrow(0, off, s_len)?,
                        qr_all: p.qr_all.narrow(1, off, s_len)?,
                        xs: p.xs.narrow(1, off, s_len)?,
                        comp_gp,
                        icomp_gp,
                        n_visible,
                        base_entries: bases[pi],
                        g_total,
                    });
                    off += s_len;
                }
                s_asm.end();
            }
            s_pprep.end();

            // ── Prefill pool: pool every sequence's completed compressor groups
            // in ONE launch across the whole prompt fleet (`Compressor::pool_and_norm`)
            // instead of a per-seq pool — the prefill hot spot. `comp` = attention
            // entries (pre-RoPE), `icomp` = indexer keys (roped); both share group
            // boundaries. Bit-identical per group to the per-seq pool
            // (`pool_batched_across_seqs_matches_per_seq`).
            let s_ppool = span("prefill:pool");
            let comp_refs: Vec<Option<&super::super::compressor::GroupPool>> =
                preps.iter().map(|p| p.comp_gp.as_ref()).collect();
            let icomp_refs: Vec<Option<&super::super::compressor::GroupPool>> =
                preps.iter().map(|p| p.icomp_gp.as_ref()).collect();
            let comp_entries = match a.compressor() {
                Some(c) => pool_across_seqs(c, &comp_refs, None, generation)?,
                None => None,
            };
            let icomp_keys = match a.indexer() {
                Some(ix) => pool_across_seqs(ix.compressor(), &icomp_refs, Some(rope), generation)?,
                None => None,
            };
            s_ppool.end();

            // Append the whole fleet's groups BEFORE part A, in one pass. The
            // appends are independent across sequences and part A's selection
            // reads each gallery only after its own append, so hoisting them out
            // of the loop is the same order of effects with one launch set
            // instead of one per sequence.
            let s_pappend = span("ppush:append");
            if let Some(entries) = comp_entries.as_ref() {
                let mut by_seq: HashMap<usize, &mut FloatGallery> = state
                    .iter_mut()
                    .filter_map(|(s, e)| e.layers[l].gallery.as_mut().map(|g| (*s, g)))
                    .collect();
                append_pooled(
                    entries,
                    icomp_keys.as_ref(),
                    &comp_refs,
                    prefill_seqs,
                    &mut by_seq,
                    generation,
                )?;
            }
            s_pappend.end();

            // ── Prefill pass 2 — FULLY BATCHED across sequences (invariant 5):
            // part A collects each seq's corpus gids + per-token (comp_idx,
            // comp_cnt); part B gathers the packed corpus, runs ONE multi-slot
            // attention kernel over the whole prompt fleet (each query indexes its
            // seq's arena slot + new-token diagonal slice via seq_of), and out-projects
            // in one GEMM; part C writes each seq's prompt latents back to its arena
            // (deferred — its set_len must not extend the committed prefix before
            // the batched kernel reads it). ──
            struct PfSeq {
                gids: Option<Tensor>, // packed-corpus entries to gather (None ⇒ empty)
                g: usize,             // this seq's corpus size
                comp_idx: Tensor,     // [s_len, max_sel_i] ids into the PACKED corpus
                comp_cnt: Tensor,     // [s_len]
            }
            let dev = x.device().clone();
            // Batched indexer query GEMM over the WHOLE prompt span — two
            // GEMMs per CSA layer-wave instead of two per SEQUENCE (the
            // per-seq form was 16 launches at width 8, each one a stall
            // point under submission pressure). Rows are seq-independent ⇒
            // bit-identical; each seq's slice ropes at its own base below.
            let idx_q_all = match (a.indexer(), proj.as_ref()) {
                (Some(ix), Some(p)) => Some(ix.query_gemm_batched(
                    &p.xs.reshape((prefill_total, ()))?,
                    &p.qr_all.reshape((prefill_total, ()))?,
                )?),
                _ => None,
            };
            let mut row_off = 0usize;
            let mut pf: Vec<PfSeq> = Vec::with_capacity(prefill_seqs.len());
            let mut seq_of_host: Vec<u32> = Vec::with_capacity(prefill_total);
            // Per-seq new-token diagonal metadata, flat {rows, base, start, -} × n_seq.
            let mut new_meta_host: Vec<u32> = Vec::with_capacity(prefill_seqs.len() * 4);
            let mut g_off = 0u32; // running packed-corpus row offset
            let mut new_off = 0u32; // running packed kv_new row offset

            // Part A: select + build (per seq); the appends ran above, and NO
            // gather/kernel yet.
            for (pi, &seq) in prefill_seqs.iter().enumerate() {
                let s_len = prefill_lens[pi];
                let base = prefill_base[pi];
                let prep = &preps[pi];
                let entry = state.get_mut(&seq).expect("ensured above");
                let q_rows = match &idx_q_all {
                    Some((q_raw, weights)) => Some((
                        q_raw.narrow(0, row_off, s_len)?,
                        weights.narrow(0, row_off, s_len)?,
                    )),
                    None => None,
                };
                row_off += s_len;
                let sel = super::super::kernel_attention::kernel_attn_prefill_select(
                    a,
                    entry.layers[l].gallery.as_ref(),
                    prep,
                    rope,
                    base,
                    q_rows,
                )?;
                let s_pgather = span("prefill:gather");
                // Build (gids, LOCAL comp_idx, comp_cnt, g) — same Device/Host
                // arms as before, but WITHOUT gathering (the gather is batched in
                // part B). Device: gather 0..n_corpus (absolute ids); Host: union
                // + remap. Both then get offset into the packed corpus below.
                let (gids, mut comp_idx, comp_cnt, g) = match sel {
                    PrefillSel::Device {
                        comp_idx,
                        comp_cnt,
                        n_corpus,
                    } => {
                        let gids = if n_corpus > 0 {
                            Some(Tensor::arange(0u32, n_corpus as u32, &dev)?)
                        } else {
                            None
                        };
                        (gids, comp_idx, comp_cnt, n_corpus)
                    }
                    PrefillSel::Host(idx_rows) => {
                        let mut union: Vec<u32> = idx_rows.iter().flatten().copied().collect();
                        union.sort_unstable();
                        union.dedup();
                        let remap: HashMap<u32, u32> = union
                            .iter()
                            .enumerate()
                            .map(|(i, &g)| (g, i as u32))
                            .collect();
                        let max_sel = idx_rows.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
                        let mut idx_flat = vec![u32::MAX; s_len * max_sel];
                        let mut cnt_v = vec![0u32; s_len];
                        for (t, gids) in idx_rows.iter().enumerate() {
                            for (j, &g) in gids.iter().enumerate() {
                                idx_flat[t * max_sel + j] = remap[&g];
                            }
                            cnt_v[t] = gids.len() as u32;
                        }
                        let gids_t = if union.is_empty() {
                            None
                        } else {
                            Some(Tensor::from_vec(union.clone(), union.len(), &dev)?)
                        };
                        (
                            gids_t,
                            Tensor::from_vec(idx_flat, (s_len, max_sel), &dev)?,
                            Tensor::from_vec(cnt_v, s_len, &dev)?,
                            union.len(),
                        )
                    }
                };
                // Shift this seq's LOCAL ids into the PACKED corpus by g_off
                // (u32::MAX pads stay MAX — the kernel bounds by comp_cnt).
                if g_off > 0 && g > 0 {
                    let (r, c) = comp_idx.dims2()?;
                    let sentinel = Tensor::full(u32::MAX, (r, c), &dev)?;
                    let full_off = Tensor::full(g_off, (r, c), &dev)?;
                    let shifted = comp_idx.broadcast_add(&full_off)?;
                    comp_idx = comp_idx.lt(&sentinel)?.where_cond(&shifted, &comp_idx)?;
                }
                s_pgather.end();
                for _ in 0..s_len {
                    seq_of_host.push(pi as u32);
                }
                // {rows, base, start, -} for this seq.
                new_meta_host.extend_from_slice(&[s_len as u32, base as u32, new_off, 0]);
                new_off += s_len as u32;
                g_off += g as u32;
                pf.push(PfSeq {
                    gids,
                    g,
                    comp_idx,
                    comp_cnt,
                });
            }

            // Part B (only when the wave has prefill rows — decode-only waves skip
            // it): batched gather → packed corpus, ONE multi-slot kernel, one
            // out-proj; then part C writeback.
            if !prefill_seqs.is_empty() {
                let projref = proj.as_ref().expect("prefill rows imply a projection");
                let total_g = g_off as usize;
                let s_pkern = span("prefill:kernel");
                let cache = if total_g == 0 {
                    st.empty_corpus_cache()?
                } else {
                    let out_nope = Tensor::empty((total_g, NOPE_DIM), DType::U8, &dev)?;
                    let out_scale = Tensor::empty((total_g, NOPE_BANDS), DType::F32, &dev)?;
                    let out_rope = Tensor::empty((total_g, ROPE_DIM), DType::BF16, &dev)?;
                    let out_pos = Tensor::empty(total_g, DType::U32, &dev)?;
                    let mut gg: Vec<&FloatGallery> = Vec::new();
                    let mut ggids: Vec<Tensor> = Vec::new();
                    let mut goff: Vec<u32> = Vec::new();
                    let mut off = 0u32;
                    for (pi, p) in pf.iter().enumerate() {
                        if let Some(gids) = &p.gids {
                            gg.push(
                                state.get(&prefill_seqs[pi]).expect("ensured").layers[l]
                                    .gallery
                                    .as_ref()
                                    .expect("gids imply a gallery"),
                            );
                            ggids.push(gids.clone());
                            goff.push(off);
                            off += p.g as u32;
                        }
                    }
                    gather_corpus_batched(
                        &gg, &ggids, &goff, &out_nope, &out_scale, &out_rope, &out_pos, generation,
                    )?;
                    super::super::paged::CorpusCache::from_gathered(
                        out_nope, out_scale, out_rope, out_pos, total_g,
                    )?
                };
                // Pad each seq's comp_idx to the global max_sel and cat over tokens.
                let max_sel = pf
                    .iter()
                    .map(|p| p.comp_idx.dim(1))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .max()
                    .unwrap_or(1);
                let mut idx_parts: Vec<Tensor> = Vec::with_capacity(pf.len());
                for p in &pf {
                    let (r, cur) = p.comp_idx.dims2()?;
                    if cur < max_sel {
                        let pad = Tensor::full(u32::MAX, (r, max_sel - cur), &dev)?;
                        idx_parts.push(Tensor::cat(&[&p.comp_idx, &pad], 1)?);
                    } else {
                        idx_parts.push(p.comp_idx.clone());
                    }
                }
                let comp_idx = Tensor::cat(&idx_parts.iter().collect::<Vec<_>>(), 0)?;
                let comp_cnt = Tensor::cat(&pf.iter().map(|p| &p.comp_cnt).collect::<Vec<_>>(), 0)?;
                // Per-query slot selector and per-seq diagonal metadata: read
                // once by the kernel and never by a tensor op, so they go
                // straight into the wave's pinned arena instead of costing a
                // device allocation plus a pageable upload each.
                let seq_of = desc::stage_slice(&seq_of_host, generation)?;
                let new_meta = desc::stage_slice(&new_meta_host, generation)?;
                let q_pos_all = prefill_q_pos
                    .as_ref()
                    .expect("built whenever the wave carries a prefill sequence");
                let out = super::super::paged::paged_latent_prefill_raw(
                    &projref.q_bf,
                    hdr_of(l, decode_seqs.len()),
                    q_pos_all,
                    seq_of.ptr(),
                    &projref.kv_bf,
                    new_meta.ptr(),
                    &cache,
                    &comp_idx,
                    &comp_cnt,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    0,
                    session.backings()[l].k_format().to_tag(),
                    st.ws(),
                )?;
                s_pkern.end();
                let s_poutp = span("prefill:outproj");
                let o_all = out.reshape((1, prefill_total, a.n_heads(), a.head_dim()))?;
                attn_rows.push(a.output_proj(&o_all, 1, prefill_total)?);
                s_poutp.end();

                // Part C: writeback each seq's prompt latents to its arena (deferred —
                // set_len must run AFTER the batched kernel read the committed prefix).
                let s_pwb = span("prefill:writeback");
                let mut runs: Vec<GlueRun> = Vec::with_capacity(prefill_seqs.len());
                for (pi, &seq) in prefill_seqs.iter().enumerate() {
                    let s_len = prefill_lens[pi];
                    let base = prefill_base[pi];
                    let base_resident = base - prefill_base_ev[pi] as usize;
                    let (wslice, wblk): (Vec<u32>, Vec<u32>) = (0..s_len)
                        .map(|t| {
                            let off = base_resident + t;
                            ((off / CHUNK_SIZE) as u32, (off % CHUNK_SIZE) as u32)
                        })
                        .unzip();
                    runs.push(GlueRun {
                        kv: preps[pi].kv_bf.clone(),
                        headers_ptr: hdr_of(l, decode_seqs.len() + pi),
                        slices: wslice,
                        in_blk: wblk,
                    });
                    session.backings()[l].set_len(seq, base_resident + s_len);
                    if l + 1 == n_layers {
                        state.get_mut(&seq).expect("ensured above").absorbed = base + s_len;
                    }
                }
                super::super::paged::paged_latent_glue_scatter(&runs, generation)?;
                s_pwb.end();
            }
            drop(s_prefill);
            // Phase D — glue attention: each glue row attends its causal
            // window at its TRUE position, keys read from the arena (its own
            // island included — scattered in phase A). Compressed selection
            // and the compression-seam fold are step-7 scope; the corpus
            // state does not absorb glue tokens.
            let s_glue_attn = span_if(!glue_seqs.is_empty(), "deepseek:glue_attn");
            for (gi, _seq) in glue_seqs.iter().enumerate() {
                let g_len = glue_lens[gi];
                let (xs, qr) = &glue_proj[gi];
                let q = a
                    .wq_b()
                    .forward(qr)?
                    .reshape((1, g_len, a.n_heads(), a.head_dim()))?;
                let q = a.rms_scale(&q)?;
                let q_bf = q
                    .reshape((g_len, a.n_heads(), a.head_dim()))?
                    .to_dtype(DType::BF16)?;
                let dev = xs.device();
                let q_pos_t = Tensor::from_vec(glue_pos[gi].clone(), g_len, dev)?;
                let empty_idx = Tensor::full(u32::MAX, (g_len, 1), dev)?;
                let empty_cnt = Tensor::zeros(g_len, DType::U32, dev)?;
                // Glue rows read only their arena window — one slot, no new-token
                // diagonal (rows=0), no corpus.
                let seq_of = desc::stage_slice(&vec![0u32; g_len], generation)?;
                let kv_dummy = Tensor::zeros((1, HEAD_DIM), DType::BF16, dev)?;
                let new_meta = desc::stage_slice(&[0u32, 0, 0, 0], generation)?;
                let out = super::super::paged::paged_latent_prefill_raw(
                    &q_bf,
                    hdr_of(l, decode_seqs.len() + prefill_seqs.len() + gi),
                    &q_pos_t,
                    seq_of.ptr(),
                    &kv_dummy,
                    new_meta.ptr(),
                    &st.empty_corpus_cache()?,
                    &empty_idx,
                    &empty_cnt,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    1,
                    session.backings()[l].k_format().to_tag(),
                    st.ws(),
                )?;
                // F32 token-major [g_len, h, hd] → [1, g_len, h, hd] = [b, s, h, hd].
                let o = out.reshape((1, g_len, a.n_heads(), a.head_dim()))?;
                attn_rows.push(a.output_proj(&o, 1, g_len)?);
            }
            drop(s_glue_attn);
            let s_hc_post = span("deepseek:hc_post_attn");
            let x = Tensor::cat(&attn_rows, 1)?; // [1, rows, dim]
            let h1 = hc.post(&x, &h, &post, &comb)?;
            s_hc_post.end();

            // MoE sub-block: one batched call per layer per wave over every row.
            // The routing stays on the device; the expert cache's pipeline
            // thread reads the layer's routing summary off the critical path.
            let s_moe = span("deepseek:moe");
            let s_moe_hcpre = span("moe:hc_pre");
            let (x, post, comb) = hc.pre(&h1, &layer.hc_ffn)?;
            s_moe_hcpre.end();
            let moe = e.moe_forward_batch(layer, &x, &flat_ids)?;
            let s_moe_hcpost = span("moe:hc_post");
            h = hc.post(&moe, &h1, &post, &comb)?;
            s_moe_hcpost.end();
            s_moe.end();

            // Capture this layer's hidden for the drafter if it is a target layer. Per DeepSeek's
            // `inference/model.py` (`main_hiddens.append(h.mean(dim=2))`), the target feature is the
            // MEAN over the mHC copies of the RAW post-layer residual — not `head_reduce` (the final
            // output collapse) and not normed (the drafter's own `main_norm` does the only norm).
            if capture_targets && self.target_layers.contains(&l) {
                target_reduced.push((l, h.mean(2)?)); // [1, rows, hc, dim] → [1, rows, dim]
            }
        }
        drop(state);

        if layer_end < n_layers {
            // Pause: persist the mHC stream flattened to a plain 3-D hidden
            // shape (opaque to the scheduler; the driver permutes and holds it).
            let flat = h.reshape((1, total_rows, cfg.hc_mult * cfg.dim))?;
            return Ok((
                WavePhase::Residual(TensorCat::from_cat_tensor(flat, 0)?),
                None,
            ));
        }

        // A batched prefill wrote its tokens outside the decode kernel (the
        // writeback scatter, committed by `set_len`), so each prefill seq's
        // cached decode slot buffer — built at the pre-prefill base offset
        // during this wave's metadata build — is behind the host. The
        // live-buffer decode path reuses that buffer, so mark it: the next
        // metadata build's sync re-serialises its writer region, or rebuilds it
        // if the prefill crossed into a chunk it was not built for (all layers
        // are absorbed in this final segment; paid once after each prefill —
        // steady-state decode never pays it). One mark per backing covers every
        // prefill sequence.
        session.mark_decode_slot_states(prefill_seqs)?;

        // Head: decode rows + each prefill sequence's LAST row.
        let s_head = span("deepseek:head_lm");
        // Rows to score: every decode row (the [0,n_dec) prefix) + each prefill
        // sequence's LAST row, selected from the residual stream BEFORE the
        // head's reduce and norm — both are row-wise, so reducing only the
        // scored rows is the same per-row arithmetic, and a prefill wave scores
        // a handful of its rows. The selection is one [1, R, …] block, so
        // lm_head runs as a SINGLE batched GEMM.
        //
        // Every decode row gets a logits row — on a verify wave that is every
        // block position (the per-position next-token prediction the
        // speculative driver verifies), one row per position of each block.
        let mut sel_rows: Vec<u32> = (0..decode_seqs.len() as u32).collect();
        let mut scored_seqs: Vec<usize> = decode_seqs.to_vec();
        // Absolute token position of each scored row (for the position-keyed
        // target-feature stash). `decode_pos` is OCCURRENCE-indexed: a verify
        // block's k-th row sits at `offset + k`, so each block row's feature is
        // stashed under its own position — the accepted positions' features are
        // then present for the NEXT draft's window (a per-row `sequence_offset`
        // here collapsed every block row onto one key, starving the drafter
        // right after each multi-token accept).
        let mut scored_pos: Vec<usize> = decode_pos.clone();
        let mut cursor = decode_seqs.len();
        for (pi, &s_len) in prefill_lens.iter().enumerate() {
            let base = prefill_base[pi];
            if verify_seqs.contains(&prefill_seqs[pi]) {
                for r in 0..s_len {
                    sel_rows.push((cursor + r) as u32);
                    scored_seqs.push(prefill_seqs[pi]);
                    scored_pos.push(base + r);
                }
            } else {
                sel_rows.push((cursor + s_len - 1) as u32);
                scored_seqs.push(prefill_seqs[pi]);
                scored_pos.push(base + s_len - 1);
            }
            cursor += s_len;
        }
        let r_total = sel_rows.len();
        let scored_h = select_head_rows(&h, sel_rows.clone(), 1, None)?; // [1, R, hc, dim]
        let reduced = hc.head_reduce(&scored_h, e.hc_head())?; // [1, R, dim]
        let scored_hidden = rms_norm(&reduced, e.output_norm(), cfg.norm_eps)?;
        let logits_all = e.lm_head().forward(&scored_hidden)?; // [1,R,vocab]

        // Stash each scored row's target-layer feature — the drafter's
        // conditioning source (`fc` input, paper `Hctx = RMSNorm(Wc·[H^{l₁};…;H^{lₘ}])`),
        // read by `speculative_draft` next step. The per-layer captures (ordered
        // by `target_layers`) are scored-row-selected and concatenated along the
        // feature axis → one `[m·dim]` vector per row, keyed by its absolute
        // position so the next draft picks the feature at `q_start-1`. Drafter only.
        if capture_targets {
            let per_layer: Vec<Tensor> = self
                .target_layers
                .iter()
                .map(|&tl| -> Result<Tensor> {
                    let captured =
                        target_reduced
                            .iter()
                            .find(|(l, _)| *l == tl)
                            .ok_or_else(|| {
                                candle::Error::msg(format!("target layer {tl} not captured"))
                            })?;
                    select_head_rows(&captured.1, sel_rows.clone(), 1, None) // [1, R, dim]
                })
                .collect::<Result<_>>()?;
            let feats = Tensor::cat(&per_layer, 2)?; // [1, R, m·dim]
            let mdim = per_layer.len() * cfg.dim;
            let mut tf = self
                .target_feat
                .write()
                .map_err(|_| candle::Error::Msg("target_feat lock poisoned".into()))?;
            // ACCUMULATE a sliding window of the recent positions' features — the drafter attends to
            // the last `window_size` target hiddens (DSparkAttention's main_kv ring), not just the
            // current one. Rejected-draft positions are overwritten when re-decoded next step; we
            // keep `window + block_size` positions per seq to bound growth.
            let window = cfg.window_size;
            for (r, (&sq, &pos)) in scored_seqs.iter().zip(scored_pos.iter()).enumerate() {
                let feat = feats.narrow(1, r, 1)?.reshape((mdim,))?;
                tf.entry(sq).or_default().insert(pos, feat);
            }
            for &sq in scored_seqs.iter() {
                if let Some(m) = tf.get_mut(&sq) {
                    if let Some(&newest) = m.keys().max() {
                        let floor = newest.saturating_sub(window + 8);
                        m.retain(|&p, _| p >= floor);
                    }
                }
            }
        }
        s_head.end();
        // One `[R, vocab]` block; the driver splits it into per-row `[1, vocab]`
        // logits (unit segments along dim 0) — row order `[decode-class | creep
        // last rows]`, which is the `[decode | prefill]` order the caller reads.
        let logits_flat = logits_all.reshape((r_total, cfg.vocab_size))?;
        Ok((
            WavePhase::Logits(TensorCat::from_cat_tensor(logits_flat, 0)?),
            None,
        ))
    }
}
