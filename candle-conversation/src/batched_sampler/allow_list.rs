//! The allow-list path: candidate-set rows the kernel constrains, their bitmaps, the deferred read-back, and the CPU finish of a sampled allow row.

use super::*;

/// A stencil-constrained row's sample with its device work enqueued
/// ([`BatchedSampler::issue_allow_list`]) and not yet read back: the allowed
/// tokens its result indexes, and the row's processor, which holds the RNG the
/// host half draws from.
pub(super) struct PendingAllowList {
    pub(super) allow: Vec<u32>,
    /// Allowed tokens the row had banned, counted by the masking that already
    /// visits them — for a failure's report.
    pub(super) banned_n: usize,
    pub(super) processor: LogitsProcessor,
    pub(super) pending: PendingSample,
}

/// A stencil-constrained row waiting for the dispatch that runs every such row
/// of a step in one launch ([`BatchedSampler::run_allow_rows`]).
pub(super) struct AllowRow {
    /// Row of the logits block it is drawn from.
    pub(super) logits_row: usize,
    /// The allowed tokens, in the order the result indexes them.
    pub(super) allow: Vec<u32>,
    /// Allowed positions the row may not produce.
    pub(super) banned: Vec<bool>,
    /// ≤ 0 asks for the argmax.
    pub(super) temperature: f32,
    /// Holds the RNG the host half draws from.
    pub(super) processor: LogitsProcessor,
}

/// What the device half of an allow-list row read back.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum AllowResult {
    /// The local index of the first maximum (temperature ≤ 0).
    Index(usize),
    /// The softmax at the row's temperature, one per allowed token.
    Probs(Vec<f32>),
}

impl AllowResult {
    /// Whether the row's allowed logits were all numbers: no
    /// [`ALLOW_NOT_FINITE`] in place of an index, no probability a NaN or an
    /// infinity left non-finite.
    pub(super) fn finite(&self) -> bool {
        match self {
            AllowResult::Index(i) => *i != ALLOW_NOT_FINITE as usize,
            AllowResult::Probs(p) => p.iter().all(|x| x.is_finite()),
        }
    }
}

/// The tokens a bitmap allows, in order.
pub(super) fn tokens_of(bits: &[u32]) -> Vec<u32> {
    let mut tokens = Vec::new();
    for (w, word) in bits.iter().enumerate() {
        let mut rest = *word;
        while rest != 0 {
            tokens.push(w as u32 * 32 + rest.trailing_zeros());
            rest &= rest - 1;
        }
    }
    tokens
}

/// Result words the dispatch writes for `row`.
pub(super) fn allow_result_words(row: &AllowRow) -> usize {
    if row.temperature <= 0.0 {
        1
    } else {
        row.allow.len()
    }
}

/// The host half: the allowed token the row's result picks — the argmax's, or
/// the one the row's processor draws from the probabilities, exactly as it
/// draws from any probability vector.
pub(super) fn finish_allow_row(row: AllowRow, result: AllowResult) -> candle::Result<u32> {
    let AllowRow {
        allow,
        banned,
        mut processor,
        ..
    } = row;
    let local = match result {
        AllowResult::Index(i) => i,
        AllowResult::Probs(p) => {
            let n = p.len();
            let probs = Tensor::from_vec(p, n, &Device::Cpu)?;
            // A handle, not a copy: read only if the sample fails.
            let kept = probs.clone();
            processor
                .sample_finish(PendingSample::Probs(probs))
                .map_err(|e| {
                    let banned_n = banned.iter().filter(|&&b| b).count();
                    allow_sample_failure(e, &allow, banned_n, non_finite_count(&kept))
                })? as usize
        }
    };
    allow.get(local).copied().ok_or_else(|| {
        candle::Error::Msg(format!(
            "allow-list row: picked position {local} of a {}-token list",
            allow.len()
        ))
    })
}

/// A stencil-constrained row that could not be sampled, with what decides why:
/// how many tokens the grammar allowed at the step, how many of those the row
/// had banned, and how many of the probabilities were not finite. A softmax
/// is NaN two ways — the model's logits went non-finite, or every allowed
/// token was banned to −inf — and the sampler's own error ("a weight is
/// negative, too large or not a valid number") reads the same for both.
///
/// Built only on the failure path: a row that samples pays for none of it.
pub(super) fn allow_sample_failure(
    e: candle::Error,
    allow: &[u32],
    banned_n: usize,
    non_finite: usize,
) -> candle::Error {
    let shown: Vec<u32> = allow.iter().take(8).copied().collect();
    candle::Error::Msg(format!(
        "{e} — allow-list row: {} allowed, {banned_n} of them banned, {non_finite} \
         probabilities not finite; first allowed {shown:?}",
        allow.len()
    ))
}

/// Entries of a probability vector that are not finite; `0` if it cannot be
/// read. For [`allow_sample_failure`] only.
pub(super) fn non_finite_count(probs: &Tensor) -> usize {
    probs
        .to_vec1::<f32>()
        .map_or(0, |v| v.iter().filter(|x| !x.is_finite()).count())
}

impl BatchedSampler {
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

    /// An allow-list row for [`Self::run_allow_rows`], with the processor its
    /// host half draws from — seeded from `state` as it stands now, as
    /// [`Self::issue_allow_list`] seeds one.
    pub(super) fn allow_row(
        &self,
        logits_row: usize,
        config: &SamplingConfig,
        state: &SequenceSamplingState,
    ) -> AllowRow {
        let allow: Vec<u32> = match &config.allowed_bits {
            Some(bits) => tokens_of(bits),
            None => config.stencil.iter().map(|&t| t as u32).collect(),
        };
        let banned_set: std::collections::HashSet<u32> =
            config.banned_tokens.iter().map(|&b| b as u32).collect();
        let banned = allow.iter().map(|t| banned_set.contains(t)).collect();
        let seed = config.seed.wrapping_add(state.rng_offset);
        AllowRow {
            logits_row,
            allow,
            banned,
            temperature: config.temperature,
            processor: LogitsProcessor::from_sampling(seed, config_to_sampling(config)),
        }
    }

    /// The device half of every allow-list row of a step, in one launch: each
    /// row's allowed logits gathered, and their argmax or their softmax at the
    /// row's temperature read back — the same device work
    /// [`Self::issue_allow_list`] does per row with tensor ops, without a
    /// buffer of its own. The device's held launch-table scratch carries the
    /// inputs and the results, and is read back in one copy.
    pub(super) fn run_allow_rows(
        &self,
        logits2d: &Tensor,
        rows: &[&AllowRow],
    ) -> candle::Result<Vec<AllowResult>> {
        use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async;
        use candle::cuda_backend::staged::segment;
        use candle::cuda_backend::WrapErr;
        use candle_kernels::sampling::run_allow_list_rows;

        let Device::Cuda(dev) = &self.device else {
            candle::bail!("allow-list rows: the batched dispatch runs on CUDA");
        };
        let (dtype_code, elem) = match logits2d.dtype() {
            DType::F32 => (0i32, 4usize),
            DType::F16 => (1, 2),
            DType::BF16 => (2, 2),
            other => candle::bail!("allow-list rows: logits of {other:?} are not gathered"),
        };
        let layout = logits2d.layout();
        let stride = layout.stride();
        if stride.len() != 2 || stride[1] != 1 {
            candle::bail!(
                "allow-list rows: a logits block of stride {stride:?} — each row must be dense"
            );
        }
        let stream = dev.cuda_stream();
        let base = {
            let (storage, _) = logits2d.storage_and_layout();
            let candle::Storage::Cuda(c) = &*storage else {
                candle::bail!("allow-list rows: logits must be on CUDA");
            };
            c.slice.device_ptr(&stream)
        };
        let row_addr = |r: usize| base + ((layout.start_offset() + r * stride[0]) * elem) as u64;

        let mut logits = Vec::with_capacity(rows.len());
        let mut allow = Vec::new();
        let mut allow_off = Vec::with_capacity(rows.len());
        let mut allow_len = Vec::with_capacity(rows.len());
        let mut temp = Vec::with_capacity(rows.len());
        let mut out_off = Vec::with_capacity(rows.len());
        let mut out_words = 0usize;
        for row in rows {
            logits.push(row_addr(row.logits_row));
            allow_off.push(allow.len() as u32);
            allow_len.push(row.allow.len() as u32);
            allow.extend(
                row.allow
                    .iter()
                    .zip(&row.banned)
                    .map(|(&t, &banned)| if banned { u32::MAX } else { t }),
            );
            temp.push(row.temperature);
            out_off.push(out_words as u32);
            out_words += allow_result_words(row);
        }
        let mut words = vec![0u32; out_words];
        dev.with_synced_upload(
            &stream,
            &[
                segment(&logits),
                segment(&allow),
                segment(&allow_off),
                segment(&allow_len),
                segment(&temp),
                segment(&out_off),
            ],
            out_words * std::mem::size_of::<u32>(),
            |at, room| {
                // SAFETY: every address names an uploaded table, and `room` the
                // results' space, held until this closure returns; each logits
                // address is a dense row of the live block; the kernel writes
                // every result word, and the readback is ordered behind the
                // launch and waited for before the scratch is let go.
                unsafe {
                    run_allow_list_rows(
                        at[0] as *const u64,
                        at[1] as *const u32,
                        at[2] as *const u32,
                        at[3] as *const u32,
                        at[4] as *const f32,
                        at[5] as *const u32,
                        rows.len() as u32,
                        dtype_code,
                        room as *mut u32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                    memcpy_dtoh_async(&mut words, room, stream.cu_stream()).w()?;
                }
                stream.synchronize().w()
            },
        )?;
        let results: Vec<AllowResult> = rows
            .iter()
            .zip(&out_off)
            .map(|(row, &at)| {
                let at = at as usize;
                if row.temperature <= 0.0 {
                    AllowResult::Index(words[at] as usize)
                } else {
                    AllowResult::Probs(
                        words[at..at + row.allow.len()]
                            .iter()
                            .map(|&w| f32::from_bits(w))
                            .collect(),
                    )
                }
            })
            .collect();
        // A row whose allowed logits held a NaN or a +inf: the kernel marks an
        // argmax so, and leaves a softmax non-finite. Refused as the
        // full-vocabulary rows are, before anything is drawn from it.
        let verdicts: Vec<u32> = results.iter().map(|r| u32::from(!r.finite())).collect();
        refuse_nonfinite(&verdicts)?;
        Ok(results)
    }

    /// The device half of a row constrained to its stencil allow-list: gather
    /// just the allowed logits (a handful) and enqueue the row's strategy over
    /// them. `O(allow-list)`, never the full vocab. Nothing is read back here —
    /// [`Self::finish_allow_list`] does that.
    pub(super) fn issue_allow_list(
        &self,
        logits2d: &Tensor,
        row: usize,
        config: &SamplingConfig,
        state: &SequenceSamplingState,
    ) -> candle::Result<PendingAllowList> {
        let allow: Vec<u32> = config.stencil.iter().map(|&t| t as u32).collect();
        let idx = Tensor::from_vec(allow.clone(), allow.len(), logits2d.device())?;
        let gathered = logits2d.i(row)?.index_select(&idx, 0)?;
        let (gathered, banned_n) = apply_banned_local(&gathered, &allow, config)?;
        let seed = config.seed.wrapping_add(state.rng_offset);
        let processor = LogitsProcessor::from_sampling(seed, config_to_sampling(config));
        let pending = processor.sample_issue(&gathered)?;
        Ok(PendingAllowList {
            allow,
            banned_n,
            processor,
            pending,
        })
    }

    /// The host half: read the row's result back and map it to its allowed
    /// token. Committing it is [`Self::commit_row`]'s.
    pub(super) fn finish_allow_list(&self, row: PendingAllowList) -> candle::Result<u32> {
        let PendingAllowList {
            allow,
            banned_n,
            mut processor,
            pending,
        } = row;
        // A handle, read only on a failure: the sample's own readback is the
        // only one a row that succeeds pays.
        let probs = match &pending {
            PendingSample::Probs(p) => Some(p.clone()),
            PendingSample::Token(_) => None,
        };
        let local = processor.sample_finish(pending).map_err(|e| {
            let non_finite = probs.as_ref().map_or(0, non_finite_count);
            allow_sample_failure(e, &allow, banned_n, non_finite)
        })? as usize;
        Ok(allow[local])
    }
}
