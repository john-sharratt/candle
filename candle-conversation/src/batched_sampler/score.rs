//! Scoring a dispatch, where `score_rows` runs one kernel dispatch over a block of rows and places the bitmaps those rows read.
//!
//! `score_rows` scores the rows against the logits block in one dispatch.
//! `place_bitmaps` pairs each masked row's bitmap address with its dials, and `bitmaps_beside` checks that a row of logits can read a bitmap.

use super::*;

impl BatchedSampler {
    /// Score `rows` against the logits block `logits2d` in one dispatch, and
    /// leave every state exactly as it was.
    ///
    /// **A row is priced against the history it would be reached with.** A
    /// speculative step's row `p` is reached only once the sequence has
    /// committed its first `p` drafts, so its penalties, RNG offset and
    /// segment edges are those of the state the commits of `row.ahead` leave.
    /// Each sequence is advanced through its rows' `ahead` tokens here exactly
    /// as [`Self::commit_row`] would advance it, each row's inputs are read off
    /// the state as it then stands, and the sequence is rewound once its rows
    /// are read. Every row then goes to the kernel together — one launch and
    /// one readback for the step, where scoring each position after the last
    /// was committed costs both per position, serialised.
    ///
    /// A row the `ahead` tokens cannot reach — the resolution of an earlier
    /// one would commit something other than its draft — scores as `None`.
    ///
    /// Forced rows need no logits; an allow-list row's device work is enqueued
    /// beside the kernel's and read back after it, so every row of the
    /// dispatch shares one drain.
    pub fn score_rows(
        &self,
        logits2d: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
        rows: &[RowSpec<'_>],
        typical: Option<TypicalAcceptance>,
    ) -> candle::Result<Vec<Option<Scored>>> {
        if configs.len() != states.len() {
            candle::bail!(
                "score_rows: {} configs for {} sequences",
                configs.len(),
                states.len()
            );
        }
        for (r, row) in rows.iter().enumerate() {
            if row.seq >= states.len() {
                candle::bail!(
                    "score_rows: row {r} names sequence {} of {}",
                    row.seq,
                    states.len()
                );
            }
            if row.draft.is_some() && typical.is_none() {
                candle::bail!("score_rows: row {r} tests a draft with no acceptance rule");
            }
        }
        if let Some(path) = &self.penalty_log_path {
            let presence = configs.first().map_or(0.0, |c| c.presence_penalty);
            if let Err(e) = self.write_penalty_log(path, states, presence) {
                tracing::warn!("Failed to write penalty log: {}", e);
            }
        }
        let on_cuda = matches!(self.device, Device::Cuda(_));
        // The cross-turn table is read only when a full-vocabulary row carries
        // the penalty, so it is only gathered then.
        let cross_turn = rows.iter().any(|r| {
            let c = configs[r.seq];
            c.stencil.is_empty() && c.cross_turn_penalty != 0.0
        });

        let n_seq = states.len();
        let mut marks: Vec<Option<SamplingMark>> = (0..n_seq).map(|_| None).collect();
        let mut recorded: Vec<Vec<u32>> = vec![Vec::new(); n_seq];
        let mut ahead_of: Vec<&[u32]> = vec![&[][..]; n_seq];
        let mut reachable = vec![true; n_seq];
        let mut out: Vec<Option<Scored>> = vec![None; rows.len()];
        let mut kernel = KernelRows::new(self.vocab_size, self.max_recent_len, cross_turn);
        let mut allow: Vec<(usize, PendingAllowList, RowKey)> = Vec::new();
        let mut allow_rows: Vec<(usize, AllowRow, RowKey)> = Vec::new();

        let gathered = (|| -> candle::Result<()> {
            for (r, row) in rows.iter().enumerate() {
                let s = row.seq;
                let config = configs[s];
                let state = &mut *states[s];
                let done = recorded[s].len();
                // Marked only when a row is about to advance it: a plain decode
                // step's rows have nothing ahead, and copying each sequence's
                // window to restore it unchanged would be the whole cost.
                if marks[s].is_none() && row.ahead.len() > done {
                    marks[s] = Some(state.mark());
                }
                if row.ahead.len() < done || row.ahead[..done] != ahead_of[s][..done] {
                    candle::bail!(
                        "score_rows: row {r}'s committed prefix does not extend the one \
                         sequence {s}'s previous row was scored after"
                    );
                }
                for &t in &row.ahead[done..] {
                    let (recorded_token, reaches) = self.advance(state, config, t);
                    recorded[s].push(recorded_token);
                    reachable[s] &= reaches;
                }
                ahead_of[s] = row.ahead;
                if !reachable[s] {
                    continue;
                }
                let key = state.row_key();
                match config.stencil.as_slice() {
                    // Forced: the single allowed token, decided without logits.
                    [forced] => {
                        out[r] = Some(Scored {
                            raw: *forced as u32,
                            kind: RowKind::Constrained,
                            rng_after: state.rng_offset.wrapping_add(1),
                            key,
                        });
                    }
                    // Small allow-list: a tiny gather + sample over just the
                    // allowed logits — on CUDA every such row of the step in
                    // one dispatch after the kernel's.
                    [_, _, ..] if on_cuda => {
                        allow_rows.push((r, self.allow_row(row.logits_row, config, state), key));
                    }
                    [_, _, ..] => {
                        let pending =
                            self.issue_allow_list(logits2d, row.logits_row, config, state)?;
                        allow.push((r, pending, key));
                    }
                    [] if on_cuda
                        && config.allowed_bits.is_some()
                        && !Self::bitmaps_beside(logits2d) =>
                    {
                        tracing::debug!(
                            target: "candle_conversation::stencil",
                            seq = row.seq,
                            "a wide grammar mask sampled by name: its logits are off the span"
                        );
                        allow_rows.push((r, self.allow_row(row.logits_row, config, state), key));
                    }
                    [] if on_cuda => kernel.push(r, state, config, row, typical, key),
                    [] => {
                        let raw = self.sample_row_cpu(logits2d, row, state, config, typical)?;
                        out[r] = Some(Scored {
                            raw,
                            kind: match config.allowed_bits {
                                Some(_) => RowKind::Constrained,
                                None => RowKind::Free,
                            },
                            rng_after: state.rng_offset.wrapping_add(1),
                            key,
                        });
                    }
                }
            }
            Ok(())
        })();
        for (s, mark) in marks.into_iter().enumerate() {
            if let Some(mark) = mark {
                states[s].rewind(mark, &recorded[s]);
            }
        }
        gathered?;

        if !kernel.is_empty() {
            let (tokens, rngs) = self.launch_rows(logits2d, &kernel)?;
            for (k, &(r, key)) in kernel.scored.iter().enumerate() {
                out[r] = Some(Scored {
                    raw: tokens[k],
                    // A row under a bitmap is the grammar's, as an allow-list
                    // row is: what it drew stands.
                    kind: match kernel.configs[k].allowed_bits {
                        Some(_) => RowKind::Constrained,
                        None => RowKind::Free,
                    },
                    rng_after: rngs[k],
                    key,
                });
            }
        }
        for (r, pending, key) in allow {
            out[r] = Some(Scored {
                raw: self.finish_allow_list(pending)?,
                kind: RowKind::Constrained,
                rng_after: key.rng_offset.wrapping_add(1),
                key,
            });
        }
        if !allow_rows.is_empty() {
            let specs: Vec<&AllowRow> = allow_rows.iter().map(|(_, a, _)| a).collect();
            let results = self.run_allow_rows(logits2d, &specs)?;
            for ((r, row, key), result) in allow_rows.into_iter().zip(results) {
                out[r] = Some(Scored {
                    raw: finish_allow_row(row, result)?,
                    kind: RowKind::Constrained,
                    rng_after: key.rng_offset.wrapping_add(1),
                    key,
                });
            }
        }
        Ok(out)
    }

    /// Whether a row of `logits` can read a bitmap: the forward priced the
    /// bitmaps on the span the logits stand on. Rows held off the span have
    /// none, and sample by name.
    pub(super) fn bitmaps_beside(logits: &Tensor) -> bool {
        // The unit tests' logits stand on no wave, and they test the kernel.
        cfg!(test) || logits.wave_ticket().is_some()
    }

    /// `rows`' dials with each masked row's bitmap address, and the buffer those
    /// addresses name, carved beside `logits` and written from the host.
    pub(super) fn place_bitmaps(
        &self,
        logits: &Tensor,
        rows: &KernelRows<'_>,
    ) -> candle::Result<(Vec<SeqDials>, Option<Tensor>)> {
        let mut dials = rows.seq_dials.clone();
        if rows.masked.is_empty() {
            return Ok((dials, None));
        }
        let Device::Cuda(dev) = &self.device else {
            candle::bail!("place_bitmaps: the kernel rows run on CUDA");
        };
        let width = rows.vocab.div_ceil(32);
        let bitmaps = logits.empty_beside((rows.masked.len(), width), DType::U32)?;
        let stream = dev.cuda_stream();
        let base = {
            let (storage, layout) = bitmaps.storage_and_layout();
            let candle::Storage::Cuda(c) = &*storage else {
                candle::bail!("place_bitmaps: bitmaps must be on CUDA");
            };
            c.slice.device_ptr(&stream) + (layout.start_offset() * 4) as u64
        };
        // SAFETY: `u32`s read as their bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                rows.allowed_bits.as_ptr() as *const u8,
                rows.allowed_bits.len() * 4,
            )
        };
        dev.upload_raw(base, bytes)?;
        tracing::debug!(
            target: "candle_conversation::stencil",
            rows = rows.masked.len(),
            bytes = bytes.len(),
            on_wave = logits.wave_ticket().is_some(),
            "grammar bitmaps carved beside the logits"
        );
        for (i, &row) in rows.masked.iter().enumerate() {
            let at = base + (i * width * 4) as u64;
            dials[row].allowed_bits_lo = at as u32;
            dials[row].allowed_bits_hi = (at >> 32) as u32;
        }
        Ok((dials, Some(bitmaps)))
    }
}
