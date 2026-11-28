//! Runs the generate phase through the model's speculative driver, with the
//! output plain greedy decode would produce.
//!
//! Each session drafts the depth its own acceptance makes cheapest per token,
//! and the acceptance is reported rather than asserted, since it is a property
//! of the text. Each session's first token is the argmax of its prefill logits,
//! checked for NaN or infinity before it seeds the drafter, so a corrupted
//! forward fails there instead of as unreadable output several steps later.
//! Every token passes the same budget and end-of-sequence policy as plain
//! decode, so the output stays comparable with each gate's expected text. The
//! per-token counts come from what was emitted.

use super::*;

impl TestParams {
    /// Lossless speculative-decode generate phase (model-agnostic): decode ALL sessions together
    /// via the generic [`ManagedBatchedModel::speculative_decode_step_batch`] driver — every
    /// active session drafts a block, every block verifies in ONE call (a single wave when the
    /// model overrides `verify_blocks`), and each session accepts/rolls back independently. Fills
    /// each `runs[i].output` with the model's exact greedy continuation, so validation is
    /// identical to the batched loop — a model with no drafter degrades to plain decode. Sessions
    /// drop out of the cohort as they hit EOS/budget. Returns a nominal step count for the perf
    /// table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn speculative_decode_phase<M>(
        &self,
        session: &mut BatchedInferenceSession,
        sequence_indices: &[usize],
        runs: &mut [TestRun],
        model: &M,
        max_draft: usize,
        max_tokens: usize,
        #[cfg(feature = "cuda")] gaps: &mut GapTally,
    ) -> Result<usize>
    where
        M: ManagedBatchedModel + ?Sized,
    {
        let nl = model.num_layers();
        let stop_on = &self.stop_on_eos;
        // First generated token per session = argmax of its prefill logits, held
        // OUT of the KV as the driver's `committed` seed. One fused greedy launch
        // over every session's row and one read-back — the sampler the decode
        // steps use — rather than an argmax and a synchronising read per session.
        let mut committed: Vec<u32> = Vec::with_capacity(sequence_indices.len());
        let mut active: Vec<bool> = Vec::with_capacity(sequence_indices.len());
        // The prefill's logits are owned copies — the forward that made them has
        // closed — so the seed work runs in an aside: the stack, the picks and
        // their logits are bumps of one held range, given back as one.
        let logits_bytes: usize = runs
            .iter()
            .map(|r| r.logits.elem_count() * r.logits.dtype().size_in_bytes())
            .sum();
        let device = runs
            .first()
            .map(|r| r.logits.device().clone())
            .unwrap_or(Device::Cpu);
        let (picked, seeds, live) = aside(&device, aside_bytes(logits_bytes), || {
            let stacked = Tensor::cat(&runs.iter().map(|r| &r.logits).collect::<Vec<_>>(), 0)?;
            // Every pick of this decode — the seed here and each accept-walk
            // row — stops at the tokenizer's last id; the padded tail is not a
            // token.
            let live = live_vocab(self.token_ids, stacked.dim(1)?);
            let seeds = stacked.batched_sample_argmax(live)?;
            // The logit each seed was chosen at. Every value of a row being
            // -inf or NaN is what a corrupted forward looks like from here, and
            // the greedy pick of such a row is a token the model never scored.
            // Caught at the seed because a bad seed feeds the drafter and the
            // whole block, so the failure would otherwise surface as unreadable
            // output several steps later.
            let picked = stacked
                .gather(&seeds.unsqueeze(1)?, 1)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            Ok((picked, seeds.to_vec1::<u32>()?, live))
        })?;
        for (i, run) in runs.iter_mut().enumerate() {
            let c = seeds[i];
            if picked[i].is_nan() || picked[i] == f32::NEG_INFINITY {
                let row = run.logits.squeeze(0)?;
                let f = |t: Result<Tensor>| -> f32 {
                    t.and_then(|x| x.to_dtype(DType::F32)?.to_vec0::<f32>())
                        .unwrap_or(f32::NAN)
                };
                candle::bail!(
                    "invalid token id sampled for sequence {i} — prefill logits corrupted \
                     (max={}, min={})",
                    f(row.max(0)),
                    f(row.min(0))
                );
            }
            run.output.push(c);
            committed.push(c);
            active.push(run.output.len() < max_tokens && !stop_on.contains(&c));
        }
        // Acceptance is the whole economics of speculation: a step costs about
        // one plain wave and yields however many tokens the verify confirmed,
        // so mean accepted/step IS the speedup ceiling. Reported rather than
        // asserted — it is a property of the text, not of the code.
        let (mut steps, mut emitted, mut drafted) = (0usize, 0usize, 0usize);
        // Each session drafts the depth its own acceptance makes cheapest per
        // token at the model's drafted-token cost, clipped at the model's
        // ladder — the same rule the scheduler applies.
        let mut depths = vec![DraftDepth::default(); sequence_indices.len()];
        let token_cost = model.draft_token_cost();
        // Greedy: this harness seeds from the prefill row's argmax and
        // validates every run against a fixed expected string, so the
        // speculative path must reproduce plain greedy decode token for token.
        // One chooser for the run, so the buffer its picks land in is held
        // across steps rather than made per step.
        let mut chooser = GreedyChooser::new(live);
        #[cfg(feature = "cuda")]
        let mut settler = SpanSettler::default();
        let t_spec = std::time::Instant::now();
        // Wall time of each step. A step returns its emitted tokens to the host,
        // so it is complete when the call returns, and the first one is the
        // prefill → decode pivot: whatever the switch costs (the weight zone
        // growing back, the decode working set re-promoted) lands there.
        let mut step_ms: Vec<f64> = Vec::new();
        loop {
            let idxs: Vec<usize> = (0..sequence_indices.len()).filter(|&i| active[i]).collect();
            if idxs.is_empty() {
                break;
            }
            steps += 1;
            let t_step = std::time::Instant::now();
            let lens: Vec<usize> = idxs.iter().map(|&i| runs[i].output.len()).collect();
            let before: usize = lens.iter().sum();
            let seqs: Vec<usize> = idxs.iter().map(|&i| sequence_indices[i]).collect();
            let comms: Vec<u32> = idxs.iter().map(|&i| committed[i]).collect();
            let budgets: Vec<usize> = idxs
                .iter()
                .map(|&i| depths[i].budget(max_draft, token_cost))
                .collect();
            // Per-session emit sinks over DISJOINT `runs` borrows: each pushes
            // into its own output and applies the budget/EOS policy — the exact
            // per-token loop plain decode uses.
            let mut emits: Vec<Box<dyn FnMut(u32) -> bool + '_>> = runs
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| active[*i])
                .map(|(_, run)| {
                    let out = &mut run.output;
                    Box::new(move |t: u32| {
                        out.push(t);
                        out.len() < max_tokens && !stop_on.contains(&t)
                    }) as Box<dyn FnMut(u32) -> bool>
                })
                .collect();
            let step = model.speculative_decode_step_batch(
                session,
                &seqs,
                &comms,
                &budgets,
                nl,
                &mut chooser,
                &mut emits,
            )?;
            drop(emits);
            // What each session actually drafted, not what it asked for: a drafter
            // that proposes less (or nothing) must not read as rejected drafts.
            drafted += step.drafted.iter().sum::<usize>();
            for (k, &i) in idxs.iter().enumerate() {
                depths[i].record(step.drafted[k], runs[i].output.len() - lens[k]);
            }
            for (k, &i) in idxs.iter().enumerate() {
                // `Some(c)` ⇒ the sink accepted `c` under budget/EOS policy (it
                // is already emitted and becomes the next seed); `None` ⇒ the
                // sink stopped this session.
                match step.next[k] {
                    Some(c) => committed[i] = c,
                    None => active[i] = false,
                }
            }
            emitted += idxs.iter().map(|&i| runs[i].output.len()).sum::<usize>() - before;
            // Between forwards, as a scheduler does on every wave: the gap's
            // catch-up, then pack what the last forward left and let the weight
            // side take it, then the gap's quantize over what it completed —
            // the scheduler's order. The first decode step is the one that hands
            // the prefill scratch back, so its holes appear here. Inside the
            // step's clock, which is where a daemon pays.
            #[cfg(feature = "cuda")]
            {
                gaps.catch_up(session)?;
                if let Some(settled) = settler.settle(session, model, gaps)? {
                    if settled.frontier_after < settled.frontier_before
                        || settled.weight_bytes_after > settled.weight_bytes_before
                    {
                        println!("{}", settled.line());
                    }
                }
                gaps.quantize(session, GapKind::Decode)?;
            }
            step_ms.push(t_step.elapsed().as_secs_f64() * 1e3);
        }
        let secs = t_spec.elapsed().as_secs_f64();
        if let Some((&first, rest)) = step_ms.split_first() {
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for &ms in rest {
                lo = lo.min(ms);
                hi = hi.max(ms);
            }
            let mean = if rest.is_empty() {
                0.0
            } else {
                rest.iter().sum::<f64>() / rest.len() as f64
            };
            println!(
                "  decode steps: first {first:.1} ms | rest {} × mean {mean:.1} ms \
                 (min {:.1}, max {hi:.1})",
                rest.len(),
                if rest.is_empty() { 0.0 } else { lo },
            );
        }
        println!(
            "  speculative: {steps} steps, {emitted} tokens, {:.2} accepted/step \
             ({:.1} tok/s over the cohort, {drafted} drafted at a ceiling of {max_draft})",
            if steps > 0 {
                emitted as f64 / steps as f64
            } else {
                0.0
            },
            if secs > 0.0 {
                emitted as f64 / secs
            } else {
                0.0
            },
        );
        // Tokens per session, not steps: the perf table multiplies this by the
        // context count to get the generated total, and a speculative step emits
        // more than one. Derived from what was actually emitted rather than from
        // the budget, so a cohort that stopped early on EOS is not credited with
        // tokens it never produced.
        Ok(emitted / sequence_indices.len().max(1))
    }
}
