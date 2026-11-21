//! The per-row types a batch is built from (keys, marks, specs, kinds, scores) and the scoring pass that fills them, plus the logit description used in logs.

use super::*;

/// What [`SequenceSamplingState::mark`] saves: every field an advance changes
/// except the dense counts, which [`SequenceSamplingState::rewind`] walks back
/// from the tokens recorded instead of copying a vocabulary.
pub(super) struct SamplingMark {
    pub(super) counted_len: usize,
    pub(super) recent_tokens: Vec<i32>,
    pub(super) current_len: i32,
    pub(super) in_segment: bool,
    pub(super) segment_len: i32,
    pub(super) dry_span_len: i32,
    pub(super) close_script_pos: Option<usize>,
    pub(super) degenerate_run: u32,
    pub(super) degenerate: bool,
    pub(super) rng_offset: u64,
}

/// The state a row was scored against, as [`SequenceSamplingState::row_key`]
/// reads it. A row commits only onto a state with the same key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowKey {
    pub(super) counted: usize,
    pub(super) current_len: i32,
    pub(super) in_segment: bool,
    pub(super) segment_len: i32,
    pub(super) dry_span_len: i32,
    pub(super) in_free_text: bool,
    pub(super) in_tool_call: bool,
    pub(super) writing_call: bool,
    pub(super) close_script_pos: Option<usize>,
    pub(super) degenerate_run: u32,
    pub(super) rng_offset: u64,
    pub(super) last: Option<i32>,
}

/// One row of a dispatch ([`BatchedSampler::score_rows`]).
#[derive(Debug, Clone, Copy)]
pub struct RowSpec<'a> {
    /// The sequence it samples: an index into the dispatch's states.
    pub seq: usize,
    /// The row of the logits block it reads.
    pub logits_row: usize,
    /// The proposal a speculative verify row tests.
    pub draft: Option<u32>,
    /// The tokens the sequence commits before this row within the step —
    /// empty for its first row. A sequence's rows each extend the previous
    /// one's by a token.
    pub ahead: &'a [u32],
}

/// How a row's token is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RowKind {
    /// Sampled over the vocabulary, then resolved — segment close, the EOS
    /// failsafes — and tracked for segment edges.
    Free,
    /// Forced or drawn from a grammar's allow-list: recorded as it stands.
    Constrained,
}

/// A row's pick, scored and not yet committed
/// ([`BatchedSampler::commit_row`]).
#[derive(Debug, Clone, Copy)]
pub struct Scored {
    pub(super) raw: u32,
    pub(super) kind: RowKind,
    pub(super) rng_after: u64,
    pub(super) key: RowKey,
}

/// Segment-close override for one sampled token, applied after sampling on
/// both the CPU and GPU paths.
///
/// Three tiers:
/// - a closer script in flight overrides everything: it plays the configured
///   phrase to its end, then emits the segment-close token itself (the close
///   is appended by this function, not stored in the script, so a played
///   script can never fail to close the segment);
/// - the GRACEFUL cap closes with the bare token at the end of a line — no
///   rescue needed;
/// - the HARD cap starts the configured closer script (a canned
///   self-interruption that turns the mid-sentence amputation into sensible
///   prose and primes the answer with an explicit commitment). It falls back
///   to the bare close token when no script is configured, when the sentence
///   happens to already be complete, or when the steering span would drop the
///   close and carry on decoding — a terminal closing statement does not belong
///   in the middle of a span that continues.
///
/// When this returns `Some`, the token is authoritative for the step: the EOS
/// failsafes must not replace it (they fire on a later step, once the segment
/// is closed and `in_segment` is false).
/// Summarise one `[batch, vocab]` logits row for the degenerate-decode report.
///
/// Names what the row actually is — how many entries are non-finite, its
/// min/max, and whether every entry is identical — so the fault is diagnosable
/// from the log alone. An all-equal row means the forward wrote nothing
/// (argmax lands on index 0, which is `!` in the Qwen vocabularies); a
/// non-finite row means the arithmetic blew up. Those have different causes and
/// the guard cannot distinguish them without looking.
pub(super) fn describe_logit_row(logits2d: &Tensor, row: usize) -> candle::Result<String> {
    let values = logits2d.i(row)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = values.len();
    let non_finite = values.iter().filter(|v| !v.is_finite()).count();
    let finite_min = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let finite_max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let first = values.first().copied().unwrap_or(0.0);
    let all_equal = values.iter().all(|v| *v == first);
    // The argmax names what the row actually wanted. A sampled token that is
    // not this one, at a logit far below it, is the sampler disagreeing with
    // the distribution rather than the forward producing a bad row — and those
    // two faults are indistinguishable from the emitted text alone.
    let argmax = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(format!(
        "logits row: vocab={n} non_finite={non_finite} finite_min={finite_min:.6} \
         finite_max={finite_max:.6} first={first:.6} argmax={argmax} all_equal={all_equal}"
    ))
}

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
}
