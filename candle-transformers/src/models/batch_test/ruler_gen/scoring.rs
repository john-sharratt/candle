//! Scores a prediction against the expected answers, and runs the batched prefill
//! and greedy decode that produces the predictions.
//!
//! Needle and variable-tracing samples pass only when every expected string appears
//! in the prediction, ignoring case, so a partial answer fails. The common-word task
//! passes when at least half the expected words appear, since its answer is a set.
//! The runner batches every sample into one session, so the decode kernel works
//! across the whole batch. Prompts are prefilled in chunks that match the cache's
//! chunk size, and sequences are released as they finish. Generation ends at an
//! end-of-sequence token, and the predictions come back in the order the samples
//! were given.

use super::*;

/// Score a single prediction against the expected outputs.
///
/// - NIAH / VT: all expected strings must appear verbatim in `pred` (case-insensitive).
/// - CWE: at least half the expected words must appear in `pred`.
pub fn score_ruler_sample(task: RulerTask, pred: &str, outputs: &[String]) -> bool {
    let pred_lower = pred.to_lowercase();
    match task {
        RulerTask::NiahSingle1 | RulerTask::NiahMultiKey2 | RulerTask::Vt => {
            outputs.iter().all(|o| pred_lower.contains(o.as_str()))
        }
        RulerTask::Cwe => {
            let hits = outputs
                .iter()
                .filter(|o| pred_lower.contains(o.as_str()))
                .count();
            hits * 2 >= outputs.len()
        }
    }
}

// ── Batched inference runner ──────────────────────────────────────────────────

/// Run batched prefill + greedy decode for a slice of `RulerSample`s.
///
/// Uses a single `BatchedInferenceSession` for all samples in the slice,
/// exploiting the paged-decode kernel's batch parallelism.
///
/// # Arguments
/// * `model` — any model implementing `ManagedBatchedModel`
/// * `tokenizer` — Qwen3 tokenizer
/// * `samples` — the samples to evaluate
/// * `mode` — `None` for F16 (no compression), `Some(m)` for quantized KV
/// * `max_gen_tokens` — maximum decode steps per sequence
/// * `eos_ids` — token IDs that terminate generation
/// * `device` — GPU device
///
/// # Returns
/// One prediction string per sample (in the same order).
pub fn run_ruler_eval<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    samples: &[RulerSample],
    mode: Option<InferenceMode>,
    max_gen_tokens: usize,
    eos_ids: &[u32],
    device: &Device,
    timings_out: Option<&mut (f64, f64)>,
) -> Result<Vec<String>> {
    let n = samples.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Tokenize all prompts.
    let prompt_tokens: Vec<Vec<u32>> = samples
        .iter()
        .map(|s| {
            tokenizer
                .encode(s.input.as_str(), false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| candle::Error::Msg(e.to_string()))
        })
        .collect::<Result<_>>()?;

    let max_prompt_len = prompt_tokens.iter().map(|t| t.len()).max().unwrap_or(0);

    // Build session.
    let session_config = match mode {
        Some(m) => BatchedConfig {
            k_format: m.k_format(),
            v_format: m.v_format(),
            compression_level: m.compression_level(),
            ..BatchedConfig::default()
        },
        None => BatchedConfig::default(),
    };
    let mut session = model.create_batched_session(session_config)?;

    let seq_indices: Vec<usize> = (0..n)
        .map(|_| session.create_sequence())
        .collect::<Result<_>>()?;

    // ── Batched prefill ───────────────────────────────────────────────────────
    let t_prefill_start = std::time::Instant::now();
    let mut last_logits: Vec<Option<Tensor>> = vec![None; n];
    let mut offset = 0usize;

    while offset < max_prompt_len {
        let chunk_end_max = (offset + PREFILL_STEP).min(max_prompt_len);

        // Group sequences that still have real tokens at this offset, keyed by chunk length.
        let mut by_chunk_len: std::collections::BTreeMap<usize, (Vec<usize>, Vec<Tensor>)> =
            std::collections::BTreeMap::new();
        for (i, tokens) in prompt_tokens.iter().enumerate() {
            if offset >= tokens.len() {
                continue; // fully prefilled
            }
            let end = chunk_end_max.min(tokens.len());
            let chunk_len = end - offset;
            // forward_batched reads input_len from dims()[1], so inputs must be 2D [1, seq_len].
            let tensor = Tensor::new(&tokens[offset..end], device)?.unsqueeze(0)?;
            let e = by_chunk_len
                .entry(chunk_len)
                .or_insert_with(|| (vec![], vec![]));
            e.0.push(seq_indices[i]);
            e.1.push(tensor);
        }

        for (chunk_len, (group_seqs, group_inputs)) in by_chunk_len {
            let nl = model.num_layers();
            let logits_vec = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &group_seqs,
                    &group_inputs,
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits_owned()?;
            for (&seq_idx, logits) in group_seqs.iter().zip(logits_vec.into_iter()) {
                let orig_i = seq_indices
                    .iter()
                    .position(|&s| s == seq_idx)
                    .expect("seq_idx not found");
                session.advance_sequence(seq_idx, chunk_len)?;
                last_logits[orig_i] = Some(logits);
            }
        }

        offset = chunk_end_max;
    }

    // ── Batched decode ────────────────────────────────────────────────────────
    let prefill_elapsed = t_prefill_start.elapsed().as_secs_f64();
    let t_decode_start = std::time::Instant::now();
    let mut generated: Vec<Vec<u32>> = vec![Vec::with_capacity(max_gen_tokens); n];
    let mut done: Vec<bool> = vec![false; n];

    let seed_rows: Vec<&Tensor> = last_logits
        .iter()
        .map(|lo| {
            lo.as_ref()
                .ok_or_else(|| candle::Error::Msg("no prefill logits".into()))
        })
        .collect::<Result<_>>()?;
    let mut current_tokens: Vec<u32> = greedy_rows(&seed_rows)?;

    for _step in 0..max_gen_tokens {
        // Mark EOS.
        for i in 0..n {
            if !done[i] && eos_ids.contains(&current_tokens[i]) {
                done[i] = true;
            }
        }
        // Push non-EOS tokens.
        for i in 0..n {
            if !done[i] {
                generated[i].push(current_tokens[i]);
            }
        }
        if done.iter().all(|&d| d) {
            break;
        }

        let active: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
        let active_seqs: Vec<usize> = active.iter().map(|&i| seq_indices[i]).collect();
        let active_inputs: Vec<Tensor> = active
            .iter()
            .map(|&i| Tensor::new(&[current_tokens[i]], device)?.unsqueeze(0))
            .collect::<Result<_>>()?;

        let nl = model.num_layers();
        let logits_vec = model
            .forward_wave(
                &mut session,
                &active_seqs,
                &active_inputs,
                &[],
                &[],
                &[],
                &[],
                0,
                nl,
                None,
            )?
            .logits_owned()?;
        for &seq_idx in &active_seqs {
            session.advance_sequence(seq_idx, 1)?;
        }
        let picked = greedy_rows(&logits_vec.iter().collect::<Vec<_>>())?;
        for (orig_i, token) in active.iter().zip(picked) {
            current_tokens[*orig_i] = token;
        }
    }

    let decode_elapsed = t_decode_start.elapsed().as_secs_f64();
    if let Some(out) = timings_out {
        *out = (prefill_elapsed, decode_elapsed);
    }

    // Release sequences.
    for &idx in &seq_indices {
        session.free_sequence(idx)?;
    }
    session.release_empty_arenas()?;

    // Decode token IDs to strings.
    prompt_tokens
        .iter()
        .enumerate()
        .map(|(i, _)| {
            tokenizer
                .decode(&generated[i], true)
                .map(|s| s.trim().to_string())
                .map_err(|e| candle::Error::Msg(e.to_string()))
        })
        .collect()
}
