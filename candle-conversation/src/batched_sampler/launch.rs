//! The two sampling paths for a wave: the CPU row sampler, and the batched CUDA kernel launch with its penalty log.

use super::*;

impl BatchedSampler {
    /// The host sampler's pick for one unconstrained (full-vocab) row, against
    /// `state` as it stands — candle's built-in sampling, for a sampler that
    /// is not on a CUDA device. Committing it is [`Self::commit_row`]'s.
    pub(super) fn sample_row_cpu(
        &self,
        logits: &Tensor,
        row: &RowSpec<'_>,
        state: &SequenceSamplingState,
        config: &SamplingConfig,
        typical: Option<TypicalAcceptance>,
    ) -> candle::Result<u32> {
        // This sequence's row of the block.
        let seq_logits = logits.i(row.logits_row)?;

        // Apply this row's banned tokens (a small deny-list, e.g. a few EOS
        // ids) by setting just those values to `-inf`.  Cheap on CPU — only the
        // banned values change (apply_banned copies the row to host F32 to do
        // it); a no-op when the list is empty.
        let seq_logits = apply_banned(&seq_logits, config)?;

        // Structural: ban the think-close token while outside a think block —
        // there is nothing for it to close there, and a stray one derails the
        // turn (see `think_close_ban_active`).
        let seq_logits = if think_close_ban_active(config, state) {
            let dtype = seq_logits.dtype();
            let dims = seq_logits.dims().to_vec();
            let mut v: Vec<f32> = seq_logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            ban(&mut v, config.segment_close_token_id as u32);
            Tensor::from_vec(v, dims, seq_logits.device())?.to_dtype(dtype)?
        } else {
            seq_logits
        };

        // Token suppression: while inside a segment, subtract the per-turn
        // penalty from each suppress-token logit. Mirrors the kernel's
        // in-segment gate, so tokens outside the segment are never touched.
        let seq_logits = if state.in_segment
            && config.segment_suppress_penalty != 0.0
            && !config.segment_suppress_tokens.is_empty()
        {
            apply_suppression(&seq_logits, config)?
        } else {
            seq_logits
        };
        // The checkpoint's padded tail is not a token.
        let seq_logits = seq_logits.narrow(0, 0, self.live_vocab)?;

        // In-segment steering: while this sequence is inside a segment,
        // sample a touch hotter (temperature + segment_temp_boost).
        // Mirrors the kernel's per-seq gate so tokens outside the segment
        // stay at the base temperature.  DRY is GPU-only — the CPU
        // LogitsProcessor has no DRY path, so there is nothing to gate here for it.
        let temperature = if state.in_segment {
            config.temperature + config.segment_temp_boost
        } else {
            config.temperature
        };
        let sampling = if state.in_segment && config.segment_temp_boost != 0.0 {
            let mut boosted = config.clone();
            boosted.temperature += config.segment_temp_boost;
            config_to_sampling(&boosted)
        } else {
            config_to_sampling(config)
        };
        let seed = config.seed.wrapping_add(state.rng_offset);
        let mut processor = LogitsProcessor::from_sampling(seed, sampling);
        let mut sampled = processor.sample(&seq_logits).map_err(|e| {
            // Read only on a failure: which of the two ways a softmax goes
            // NaN this row took — non-finite logits, or everything at −inf.
            let (nan, inf, neg_inf) = seq_logits
                .to_dtype(DType::F32)
                .and_then(|l| l.to_vec1::<f32>())
                .map(|v| {
                    v.iter().fold((0, 0, 0), |(n, p, m), x| {
                        (
                            n + usize::from(x.is_nan()),
                            p + usize::from(*x == f32::INFINITY),
                            m + usize::from(*x == f32::NEG_INFINITY),
                        )
                    })
                })
                .unwrap_or((0, 0, 0));
            candle::Error::Msg(format!(
                "{e} — full row of {}: {nan} NaN, {inf} +inf, {neg_inf} −inf logits; \
                 {} banned",
                self.live_vocab,
                config.banned_tokens.len()
            ))
        })?;
        // A verify row's draft, accepted on the typical-acceptance rule over
        // the same top-k/nucleus distribution the kernel measures it on.
        if let (Some(draft), Some(typical)) = (row.draft, typical) {
            if temperature > 0.0 {
                let values: Vec<f32> = seq_logits.to_dtype(DType::F32)?.to_vec1()?;
                if typical_accepts_row(
                    &values,
                    temperature,
                    config.top_k,
                    config.top_p,
                    draft,
                    typical,
                ) {
                    sampled = draft;
                }
            }
        }
        Ok(sampled)
    }

    /// Run the kernel over every row `rows` gathered — one launch and one
    /// readback for the dispatch — answering with each row's pick and the RNG
    /// offset it leaves. `logits` is the `[rows, vocab]` block the rows name.
    pub(super) fn launch_rows(
        &self,
        logits: &Tensor,
        rows: &KernelRows<'_>,
    ) -> candle::Result<(Vec<u32>, Vec<u64>)> {
        let batch_size = rows.len();

        // Determine dtype from logits
        let dtype = logits.dtype();
        let dtype_enum = match dtype {
            DType::F32 => KernelDType::F32 as i32,
            DType::F16 => KernelDType::F16 as i32,
            DType::BF16 => KernelDType::BF16 as i32,
            _ => {
                return Err(candle::Error::Msg(format!(
                    "unsupported dtype for sampling: {:?}",
                    dtype
                )))
            }
        };

        let logits_vocab_size = logits.dim(1)? as i32;

        // Validate that our penalty buffer vocab_size matches the logits.
        // A mismatch means token_counts is undersized and the kernel would
        // read out-of-bounds GPU memory for high token IDs (e.g. EOS tokens).
        if (self.vocab_size as i32) != logits_vocab_size {
            return Err(candle::Error::Msg(format!(
                "vocab_size mismatch: sampler has {} but logits have {} — \
                 set EngineConfig::vocab_size to match the model/tokenizer",
                self.vocab_size, logits_vocab_size
            )));
        }
        let vocab_size = logits_vocab_size;

        // This path only ever receives unconstrained (full-vocab) rows —
        // stencil-constrained rows are resolved beside it and never reach
        // here. `config` (the first row's) supplies the scalar FFI arguments
        // below, but those are only the null-fallback defaults: the kernel reads
        // its real per-row dials from the `seq_dials` array, so no row inherits
        // row 0's dials.
        let config = rows.configs[0];

        // Get DRY params
        let (dry_multiplier, dry_base, dry_allowed_length, dry_range) =
            if let Some(ref dry) = config.dry {
                (dry.multiplier, dry.base, dry.allowed_length, dry.range)
            } else {
                (0.0, 1.75, 2, 0)
            };

        // Get EOS token
        let eos_token_id = self.eos_tokens.iter().copied().next().unwrap_or(0);

        // Each row's OWN deny-list, plus the structural think-close ban for rows
        // outside a block (see `KernelRows::push`).
        let (banned_tokens, num_banned, banned_per_seq) = banned_buffer(&rows.banned);
        let banned_tokens = &banned_tokens;

        // Each masked row's bitmap, on the span beside the logits it masks. The
        // tensor stands until the launch is queued; the span's reset reclaims it.
        let (seq_dials, _bitmaps) = self.place_bitmaps(logits, rows)?;

        // No stencil here — constrained rows were resolved before the kernel.
        let stencil: &[i32] = &[];
        let stencil_size = 0i32;

        // Allocate output buffer
        let mut output_tokens = vec![0u32; batch_size];
        let mut rng_offsets = rows.rng_offsets.clone();

        // Compute EOS ramp params
        let (eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier) = if config.dynamic_eos_boost {
            (
                config.eos_ramp_start,
                config.eos_ramp_len,
                config.eos_boost_max_multiplier,
            )
        } else {
            (0, 0, 0.0)
        };

        // Token suppression (the in-segment ceiling lever).
        // The token list is shared across the batch (config[0]); the penalty is
        // per-sequence (large = HARD ban, moderate = SOFT, 0.0 = off). Activate
        // only when the list is non-empty AND at least one sequence has a nonzero
        // penalty — otherwise pass null/0 so the kernel skips it entirely.
        let suppress_tokens: Vec<i32> = config.segment_suppress_tokens.clone();
        let suppress_active =
            !suppress_tokens.is_empty() && rows.suppress_penalties.iter().any(|&p| p != 0.0);

        let segment_close_active =
            config.segment_close_boost != 0.0 && config.segment_close_token_id >= 0;
        let (
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
        ) = if segment_close_active {
            (
                config.segment_close_boost,
                config.segment_close_token_id,
                config.segment_close_ramp_start,
                config.segment_close_ramp_len,
                config.segment_close_max_multiplier,
            )
        } else {
            (0.0, -1, 0, 0, 0.0)
        };

        let stamp_span = profile::span("sample:stamp");
        let mut tables = self
            .penalty_tables
            .lock()
            .map_err(|_| candle::Error::Msg("sampler count tables poisoned".into()))?;
        let token_table = tables
            .tokens
            .stamp(&self.device, batch_size, &rows.counts)?;
        let cross_table = if rows.cross_turn {
            match tables.cross.stamp(&self.device, batch_size, &rows.cross) {
                Ok(stamped) => Some(stamped),
                Err(e) => {
                    // The token table is already stamped; leave it zero.
                    tables.tokens.clear(token_table)?;
                    return Err(e);
                }
            }
        } else {
            None
        };
        stamp_span.end();

        // Invoke the CUDA kernel
        let launch_span = profile::span("sample:launch_readback");
        let launched = self.invoke_cuda_kernel(
            logits,
            batch_size as i32,
            vocab_size,
            dtype_enum,
            config.temperature,
            config.top_k,
            config.top_p,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_range,
            config.eos_boost,
            eos_token_id as i32,
            eos_ramp_start,
            eos_ramp_len,
            eos_boost_max_multiplier,
            config.cross_turn_penalty,
            cross_table.as_ref().map(|t| t.table()),
            &rows.current_lens,
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
            &rows.segment_lens,
            &rows.dry_lens,
            config.segment_temp_boost,
            &suppress_tokens,
            &rows.suppress_penalties,
            suppress_active,
            token_table.table(),
            banned_tokens,
            num_banned,
            banned_per_seq,
            &rows.recent_tokens,
            &rows.recent_lens,
            stencil,
            stencil_size,
            &mut output_tokens,
            config.seed,
            &mut rng_offsets,
            &seq_dials,
        );
        launch_span.end();
        // Both cleared whether or not the launch succeeded, and the second
        // whether or not the first did: a table left stamped would price the
        // next dispatch's rows with this one's counts.
        let clear_span = profile::span("sample:clear");
        let tokens_cleared = tables.tokens.clear(token_table);
        let cross_cleared = match cross_table {
            Some(cross) => tables.cross.clear(cross),
            None => Ok(()),
        };
        clear_span.end();
        drop(tables);
        launched?;
        tokens_cleared?;
        cross_cleared?;
        Ok((output_tokens, rng_offsets))
    }

    pub(super) fn write_penalty_log(
        &self,
        log_path: &std::path::Path,
        states: &[&mut SequenceSamplingState],
        presence_penalty: f32,
    ) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(log_path)?;

        // Write the presence penalty value at the top
        writeln!(file, "=== PRESENCE PENALTY: {} ===", presence_penalty)?;
        writeln!(file)?;

        for (batch_idx, state) in states.iter().enumerate() {
            writeln!(file, "=== Batch {} ===", batch_idx)?;
            writeln!(file, "Vocab size: {}", state.token_counts.len())?;
            writeln!(file)?;

            // Write nonzero token counts
            let nonzero_tokens: Vec<_> = state
                .token_counts
                .iter()
                .enumerate()
                .filter(|(_, &count)| count > 0)
                .collect();

            if nonzero_tokens.is_empty() {
                writeln!(file, "No penalties applied yet")?;
            } else {
                writeln!(file, "Tokens with nonzero counts (will be penalized):")?;
                for (token_id, &count) in nonzero_tokens {
                    writeln!(file, "  Token {}: count={}", token_id, count)?;
                }
            }

            writeln!(file)?;
            writeln!(
                file,
                "Recent token history ({} tokens):",
                state.recent_tokens.len()
            )?;
            for (i, &token_id) in state.recent_tokens.iter().enumerate() {
                write!(file, "  {} ", token_id)?;
                if (i + 1) % 20 == 0 {
                    writeln!(file)?;
                }
            }
            if !state.recent_tokens.is_empty() {
                writeln!(file)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }

    /// Invoke the CUDA kernel with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn invoke_cuda_kernel(
        &self,
        logits: &Tensor,
        batch_size: i32,
        vocab_size: i32,
        dtype: i32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: i32,
        dry_range: i32,
        eos_boost: f32,
        eos_token_id: i32,
        eos_ramp_start: i32,
        eos_ramp_len: i32,
        eos_boost_max_multiplier: f32,
        cross_turn_penalty: f32,
        cross_turn_counts: Option<&Tensor>,
        current_lens: &[i32],
        segment_close_boost: f32,
        segment_close_token_id: i32,
        segment_close_ramp_start: i32,
        segment_close_ramp_len: i32,
        segment_close_max_multiplier: f32,
        segment_lens: &[i32],
        dry_lens: &[i32],
        segment_temp_boost: f32,
        suppress_tokens: &[i32],
        suppress_penalties: &[f32],
        suppress_active: bool,
        token_counts: &Tensor,
        banned_tokens: &[i32],
        num_banned: i32,
        banned_per_seq: i32,
        recent_tokens: &[i32],
        recent_lens: &[i32],
        stencil: &[i32],
        stencil_size: i32,
        output_tokens: &mut [u32],
        seed: u64,
        rng_offsets: &mut [u64],
        seq_dials: &[SeqDials],
    ) -> candle::Result<()> {
        // Get the CUDA device and stream
        let cuda_device = match &self.device {
            Device::Cuda(dev) => dev,
            _ => return Err(candle::Error::Msg("expected CUDA device".into())),
        };

        let stream = cuda_device.cuda_stream();

        // Get logits storage and layout
        let (logits_storage, logits_layout) = logits.storage_and_layout();
        let cuda_storage = match &*logits_storage {
            candle::Storage::Cuda(cs) => cs,
            _ => return Err(candle::Error::Msg("logits must be on CUDA".into())),
        };

        // The count tables are already on the device; hold their storage for
        // the launch.
        let (tc_storage, tc_layout) = token_counts.storage_and_layout();
        let cross_storage = cross_turn_counts.map(Tensor::storage_and_layout);

        // Every small per-dispatch array in one upload. Per-sequence dials go
        // in as raw 4-byte words — `SeqDials` is packed `f32`/`i32` fields (its
        // C twin `batched_sampling::SeqDials` has the identical layout), so the
        // byte image is what the kernel reads back. The arrays the kernel
        // writes — RNG offsets, outputs, then each row's non-finite verdict —
        // go last so one copy reads them all.
        const SEQ_DIALS_WORDS: usize = std::mem::size_of::<SeqDials>() / 4;
        // SAFETY: `SeqDials` is `#[repr(C)]` with only 4-byte `f32`/`i32`
        // fields and no padding, so a contiguous slice of them is a valid
        // `[i32]` of `len * SEQ_DIALS_WORDS` words.
        let seq_dials_words: &[i32] = unsafe {
            std::slice::from_raw_parts(
                seq_dials.as_ptr() as *const i32,
                seq_dials.len() * SEQ_DIALS_WORDS,
            )
        };
        let rows = output_tokens.len();
        let mut pack = ArgPack::new();
        let cur_lens_at = pack.push_i32(current_lens);
        let segment_lens_at = pack.push_i32(segment_lens);
        let dry_lens_at = pack.push_i32(dry_lens);
        let banned_at = pack.push_i32(banned_tokens);
        let suppress_tok_at = pack.push_i32(suppress_tokens);
        let suppress_pen_at = pack.push_f32(suppress_penalties);
        let recent_at = pack.push_i32(recent_tokens);
        let recent_lens_at = pack.push_i32(recent_lens);
        let stencil_at = pack.push_i32(stencil);
        let seq_dials_at = pack.push_i32(seq_dials_words);
        let rng_at = pack.push_u64(rng_offsets);
        let output_at = pack.reserve(rows);
        let nonfinite_at = pack.reserve(rows);
        let words = pack.words();
        let mut args = self
            .args
            .lock()
            .map_err(|_| candle::Error::Msg("sampler argument buffer poisoned".into()))?;
        if !args.as_ref().is_some_and(|a| a.len() >= words.len()) {
            let grown = args
                .as_ref()
                .map_or(0, |a| a.len() * 2)
                .max(words.len())
                .max(1);
            // The old buffer is freed on this stream, after the readback that
            // ended the dispatch it served.
            *args = None;
            // SAFETY: every word a dispatch reads is uploaded below before the
            // kernel is queued, and the tail it reads back is written by it.
            *args = Some(unsafe { Held::<u32>::alloc(cuda_device, grown) }?);
        }
        let packed = args.as_mut().expect("sized above");
        stream
            .memcpy_htod(words, &mut packed.slice_mut(..words.len()))
            .map_err(|e| candle::Error::Msg(format!("failed to upload sampling args: {e}")))?;
        let mut reference = self
            .reference_flags_dev
            .lock()
            .map_err(|_| candle::Error::Msg("sampler reference flags poisoned".into()))?;
        if reference.is_none() && !self.reference_flags.is_empty() {
            // SAFETY: every byte is written by the upload just below.
            let mut flags = unsafe { Held::<u8>::alloc(cuda_device, self.reference_flags.len()) }?;
            stream
                .memcpy_htod(&self.reference_flags, &mut *flags)
                .map_err(|e| candle::Error::Msg(format!("DRY reference flags: {e}")))?;
            *reference = Some(flags);
        }

        // Get device pointers and call kernel in a scoped block
        // so guards are dropped before download
        {
            let (tc_ptr, _g1) = count_table_ptr(&tc_storage, tc_layout, &stream)?;
            let cross = match &cross_storage {
                Some((storage, layout)) => Some(count_table_ptr(storage, layout, &stream)?),
                None => None,
            };
            let (base, _g2) = packed.device_ptr_mut(&stream);
            let at = |word: usize| base + (word * 4) as u64;
            let cur_lens_ptr = at(cur_lens_at);
            let segment_lens_ptr = at(segment_lens_at);
            let dry_lens_ptr = at(dry_lens_at);
            let suppress_tok_ptr = at(suppress_tok_at);
            let suppress_pen_ptr = at(suppress_pen_at);
            let ban_ptr = at(banned_at);
            let recent_ptr = at(recent_at);
            let recent_lens_ptr = at(recent_lens_at);
            let stencil_ptr = at(stencil_at);
            let output_ptr = at(output_at);
            let nonfinite_ptr = at(nonfinite_at);
            let rng_ptr = at(rng_at);
            let seq_dials_ptr = at(seq_dials_at);
            let reference_view = reference.as_ref().map(|flags| flags.device_ptr(&stream));
            let reference_ptr = reference_view
                .as_ref()
                .map_or(std::ptr::null(), |(ptr, _)| *ptr as *const u8);

            // Helper closure to call kernel with logits pointer
            let call_kernel = |logits_ptr: *const std::ffi::c_void| unsafe {
                run_batched_sampling(
                    logits_ptr,
                    batch_size,
                    vocab_size,
                    self.live_vocab as i32,
                    dtype,
                    temperature,
                    top_k,
                    top_p,
                    repeat_penalty,
                    frequency_penalty,
                    presence_penalty,
                    dry_multiplier,
                    dry_base,
                    dry_allowed_length,
                    dry_range,
                    eos_boost,
                    eos_token_id,
                    eos_ramp_start,
                    eos_ramp_len,
                    eos_boost_max_multiplier,
                    cross_turn_penalty,
                    match &cross {
                        Some((cross_ptr, _)) => *cross_ptr as *const i32,
                        None => std::ptr::null(),
                    },
                    cur_lens_ptr as *const i32,
                    segment_close_boost,
                    segment_close_token_id,
                    segment_close_ramp_start,
                    segment_close_ramp_len,
                    segment_close_max_multiplier,
                    segment_lens_ptr as *const i32,
                    dry_lens_ptr as *const i32,
                    reference_ptr,
                    segment_temp_boost,
                    if suppress_active {
                        suppress_tok_ptr as *const i32
                    } else {
                        std::ptr::null()
                    },
                    if suppress_active {
                        suppress_tokens.len() as i32
                    } else {
                        0
                    },
                    if suppress_active {
                        suppress_pen_ptr as *const f32
                    } else {
                        std::ptr::null()
                    },
                    tc_ptr as *const i32,
                    ban_ptr as *const i32,
                    num_banned,
                    banned_per_seq,
                    recent_ptr as *const i32,
                    recent_lens_ptr as *const i32,
                    self.max_recent_len as i32,
                    if stencil_size > 0 {
                        stencil_ptr as *const i32
                    } else {
                        std::ptr::null()
                    },
                    stencil_size,
                    output_ptr as *mut u32,
                    nonfinite_ptr as *mut u32,
                    seed,
                    rng_ptr as *mut u64,
                    // Null on an empty (guard) upload, else the per-row dials.
                    if seq_dials.is_empty() {
                        std::ptr::null()
                    } else {
                        seq_dials_ptr as *const std::ffi::c_void
                    },
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            };

            // Match on dtype to get the properly-typed slice and its pointer
            let start_offset = logits_layout.start_offset();
            match &cuda_storage.slice {
                CudaStorageSlice::F32(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = (ptr + (start_offset as u64 * 4)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::F16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = (ptr + (start_offset as u64 * 2)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::BF16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = (ptr + (start_offset as u64 * 2)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                _ => {
                    return Err(candle::Error::Msg(format!(
                        "unsupported dtype for sampling: {:?}",
                        logits.dtype()
                    )));
                }
            }
        } // Guards dropped here

        // One stream-ordered copy reads the RNG offsets, the outputs and the
        // verdicts back. It is queued behind the kernel, and a device-to-host
        // copy into pageable memory returns only once it has completed, so no
        // separate synchronise is needed.
        let tail = stream
            .memcpy_dtov(&packed.slice(rng_at..nonfinite_at + rows))
            .map_err(|e| candle::Error::Msg(format!("failed to download sampling results: {e}")))?;
        let readback = split_readback(&tail, rows);
        // A row the kernel judged non-finite fails the dispatch before
        // anything is handed back, so no state moves for the token it drew.
        refuse_nonfinite(&readback.nonfinite)?;
        output_tokens.copy_from_slice(&readback.outputs);
        rng_offsets.copy_from_slice(&readback.rng);

        Ok(())
    }
}
