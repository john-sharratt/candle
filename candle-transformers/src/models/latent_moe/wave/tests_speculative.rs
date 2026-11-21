//! Tests that speculative decoding produces the same tokens as plain greedy decoding.
//!
//! Speculative decoding is lossless by construction, so its token stream must
//! match plain greedy decoding exactly, with or without a drafter attached. The
//! speedup test runs one prompt both ways and reports mean committed tokens per
//! step, the acceptance histogram and the phase timings. The real-text test
//! reports acceptance without asserting identity, because the batched verify
//! prefill matches the decode path only approximately on some tokens.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    // These are end-to-end gates against the real DeepSeek-V4-Flash checkpoint, so
    // they name the concrete architecture. The engine's model-independent tests
    // run against `arch::test_arch` instead.
    use crate::models::deepseek4::{gate_packs, DEEPSEEK_V4};

    use crate::models::gpu_test_lock::gpu_serial;

    // Lossless gates: they assert speculation reproduces plain GREEDY decode
    // token for token, so they choose greedily.
    use crate::models::speculative_choice::GreedyChooser;

    use candle::IndexOp;

    /// Speculative decode is **lossless**: the same prompt driven through
    /// `ManagedBatchedModel::speculative_decode_step` (draft → verify → accept
    /// the longest matching prefix → roll back the rest) must yield a token
    /// stream **bit-identical** to plain greedy
    /// [`wave_prefill_then_decode_end_to_end`]. That is true by construction —
    /// accepted tokens are the target's own argmaxes — so any difference is a
    /// bug in the driver, `verify_block`, or the `truncate_sequence_to_tokens`
    /// rollback, which is exactly what this pins down.
    ///
    /// Runs with whatever drafter the model supplies: none ⇒ the driver
    /// degenerates to plain decode (still must match); a real one ⇒ same
    /// stream, fewer waves. `--ignored` (needs a checkpoint).
    #[test]
    #[ignore]
    fn wave_speculative_step_is_lossless() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let Some(pack) = gate_packs::target(&device)? else {
            eprintln!("[skip] merged file absent");
            return Ok(());
        };
        let engine = Engine::load(&pack, &DEEPSEEK_V4, &device)?;
        let model = BatchedEngine::new(engine)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = ManagedBatchedModel::num_layers(&model);

        // Prefill the prompt; the first generated token is the argmax of the last
        // prefill row — held OUT of the KV as the driver's `committed` seed.
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        if logits.is_empty() {
            return Err(candle::Error::msg("prefill produced no logits"));
        }
        let mut committed = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        // Speculative decode loop: each step commits ≥1 token (the model's exact
        // greedy continuation), draft length up to 4.
        let mut gen = vec![committed];
        let mut steps = 0usize;
        let t0 = std::time::Instant::now();
        while gen.len() < 12 && committed != eos {
            steps += 1;
            let next = model.speculative_decode_step(
                &mut session,
                seq,
                committed,
                4,
                n_layers,
                &mut GreedyChooser::whole_row(),
                &mut |t| {
                    gen.push(t);
                    gen.len() < 12 && t != eos
                },
            )?;
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[spec] ids={gen:?}");
        eprintln!("[spec] continuation={text:?}");
        eprintln!(
            "[spec] {steps} spec steps, {} tokens in {dt:.1}s = {:.2} tok/s",
            gen.len(),
            gen.len() as f32 / dt
        );
        assert_eq!(
            *gen.last().unwrap(),
            eos,
            "speculative path did not stop on EOS within 12 tokens: {text:?}"
        );
        let answer = tokenizer
            .decode(&gen[..gen.len() - 1], false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        assert_eq!(
            answer.trim(),
            "Paris",
            "speculative path must answer exactly \"Paris\" (lossless vs greedy): {text:?}"
        );
        Ok(())
    }

    /// [`wave_speculative_step_is_lossless`], now with a REAL drafter attached
    /// rather than the degenerate no-drafter path — so `speculative_draft`
    /// proposes an actual block each step and the whole integration is live:
    /// the drafter's streaming MoE, its injected-context backbone, the Markov
    /// sampler, and accept/rollback. The stream must still be bit-identical to
    /// plain greedy decode; a drafter can only change how many waves it takes to
    /// get there, never what comes out.
    ///
    /// Also reports mean committed tokens/step. With the sequential
    /// `verify_block` that is not yet a wall-clock win (verify runs one wave per
    /// block token), but tokens/step > 1 is the ACCEPTANCE signal a batched
    /// verify converts into speedup on this launch-bound decode — measured by
    /// [`wave_speculative_speedup_with_drafter`]. Skips when the target or the
    /// drafter checkpoint is absent from the folder below. `--ignored`.
    #[test]
    #[ignore]
    fn wave_speculative_step_with_drafter_is_lossless() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let (Some(pack), Some(dspark)) =
            (gate_packs::target(&device)?, gate_packs::drafter(&device)?)
        else {
            eprintln!("[skip] target or DSpark drafter absent");
            return Ok(());
        };
        // Combined load: the drafter lands in the DENSE tier (before the span
        // reservation), so the elastic boundary balances target-experts vs KV
        // around it — see `Engine::load_with_drafter`.
        let (engine, drafter) =
            Engine::load_with_drafter(&pack, Some(&dspark), &DEEPSEEK_V4, &device)?;
        let model =
            BatchedEngine::new(engine)?.with_drafter(drafter.expect("dspark path given"))?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = ManagedBatchedModel::num_layers(&model);

        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        if logits.is_empty() {
            return Err(candle::Error::msg("prefill produced no logits"));
        }
        let mut committed = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        let max_draft = 4usize;
        let mut gen = vec![committed];
        let mut steps = 0usize;
        let t0 = std::time::Instant::now();
        while gen.len() < 12 && committed != eos {
            steps += 1;
            // The driver emits each accepted token; we keep it and stop at the first EOS — the exact
            // per-token loop plain decode uses (the drafter drafts a whole block that can run PAST
            // EOS; those post-EOS argmaxes are simply never generated). No special block handling.
            let next = model.speculative_decode_step(
                &mut session,
                seq,
                committed,
                max_draft,
                n_layers,
                &mut GreedyChooser::whole_row(),
                &mut |t| {
                    gen.push(t);
                    gen.len() < 12 && t != eos
                },
            )?;
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        let per_step = gen.len() as f32 / steps.max(1) as f32;
        eprintln!("[spec-dspark] ids={gen:?}");
        eprintln!("[spec-dspark] continuation={text:?}");
        eprintln!(
            "[spec-dspark] {steps} spec steps for {} tokens ⇒ {per_step:.2} tokens/step \
             (max_draft={max_draft}); {dt:.1}s",
            gen.len(),
        );
        assert_eq!(
            *gen.last().unwrap(),
            eos,
            "speculative+DSpark did not stop on EOS within 12 tokens: {text:?}"
        );
        let answer = tokenizer
            .decode(&gen[..gen.len() - 1], false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        assert_eq!(
            answer.trim(),
            "Paris",
            "speculative+DSpark must answer exactly \"Paris\" (lossless vs greedy): {text:?}"
        );
        Ok(())
    }

    /// The SPEEDUP measurement: a longer, drafter-friendly generation decoded two ways from the same
    /// prefill — (1) plain greedy (one wave per token), (2) speculative with the DSpark drafter +
    /// **batched** `verify_block` (draft a block, verify it in ONE wave, accept the matching prefix).
    /// Asserts the two token streams are **identical** (speculative is lossless) and reports the
    /// wall-clock speedup + mean accepted tokens/step. Because DeepSeek decode is launch-bound, a
    /// batched K-token verify costs ~one decode, so accepted drafts translate almost directly into
    /// throughput. `--ignored` (needs the 156 GB target + drafter; the KV VRAM reserve is
    /// installed from the engine's measured post-resident budget, so no tuning is needed).
    #[test]
    #[ignore]
    fn wave_speculative_speedup_with_drafter() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let (Some(pack), Some(dspark)) =
            (gate_packs::target(&device)?, gate_packs::drafter(&device)?)
        else {
            eprintln!("[skip] target or DSpark drafter absent");
            return Ok(());
        };
        let (engine, drafter) =
            Engine::load_with_drafter(&pack, Some(&dspark), &DEEPSEEK_V4, &device)?;
        let model =
            BatchedEngine::new(engine)?.with_drafter(drafter.expect("dspark path given"))?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        // A deterministic, structured continuation the drafter predicts well (high acceptance).
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Count from 1 to 30, separated by \
             commas.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");
        let n_layers = ManagedBatchedModel::num_layers(&model);
        const MAX_NEW: usize = 64;

        // Prefill helper → returns (session, first committed token).
        let prefill = |model: &BatchedEngine| -> Result<(BatchedInferenceSession, usize, u32)> {
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let seq = session.create_sequence()?;
            let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt_t),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, ids.len())?;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("no prefill logits"));
            }
            let first = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            Ok((session, seq, first))
        };

        // ── Greedy baseline: one wave per token. ──
        let (mut gsession, gseq, gfirst) = prefill(&model)?;
        // Drop the prefill's spans so the snapshot below is pure plain-decode.
        #[cfg(feature = "profile")]
        let _ = crate::models::profile::pipeline_snapshot_and_reset();
        let mut greedy = vec![gfirst];
        let mut committed = gfirst;
        let t0 = std::time::Instant::now();
        while greedy.len() < MAX_NEW && committed != eos {
            let t = Tensor::from_vec(vec![committed], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut gsession,
                &[gseq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            gsession.advance_sequence(gseq, 1)?;
            committed = step
                .logits
                .as_ref()
                .ok_or_else(|| candle::Error::msg("no decode logits"))?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            greedy.push(committed);
        }
        let greedy_dt = t0.elapsed().as_secs_f32();
        // Per-phase anatomy at IDENTICAL corpus: the greedy snapshot is the
        // plain decode wave's per-layer cost, the spec snapshot (below) the
        // verify wave's — their diff is exactly what the decode-shaped-verify
        // work must collapse.
        #[cfg(feature = "profile")]
        {
            let snap = crate::models::profile::pipeline_snapshot_and_reset();
            let mut es = snap.entries.clone();
            es.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[greedy-prof] plain decode waves (total ms, count):");
            for (name, ms, count) in es.iter().take(28) {
                eprintln!("  {name:34} {ms:9.1}ms  x{count}");
            }
        }

        // ── Speculative: drafter + batched verify_block, up to 4 drafts/step. ──
        let (mut ssession, sseq, sfirst) = prefill(&model)?;
        #[cfg(feature = "profile")]
        let _ = crate::models::profile::pipeline_snapshot_and_reset();
        let mut spec = vec![sfirst];
        let mut committed = sfirst;
        let mut steps = 0usize;
        let mut accepts: Vec<usize> = Vec::new(); // committed tokens per step (1 = 0 drafts accepted)
        let t0 = std::time::Instant::now();
        while spec.len() < MAX_NEW && committed != eos {
            steps += 1;
            let before = spec.len();
            let next = model.speculative_decode_step(
                &mut ssession,
                sseq,
                committed,
                4,
                n_layers,
                &mut GreedyChooser::whole_row(),
                &mut |t| {
                    spec.push(t);
                    spec.len() < MAX_NEW && t != eos
                },
            )?;
            accepts.push(spec.len() - before); // committed tokens this step
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let spec_dt = t0.elapsed().as_secs_f32();
        #[cfg(feature = "profile")]
        {
            let snap = crate::models::profile::pipeline_snapshot_and_reset();
            let mut es = snap.entries.clone();
            es.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[spec-prof] verify waves + drafts (total ms, count):");
            for (name, ms, count) in es.iter().take(28) {
                eprintln!("  {name:34} {ms:9.1}ms  x{count}");
            }
        }
        eprintln!("[spec-speedup] per-step committed (1=no draft accepted): {accepts:?}");

        let greedy_tps = greedy.len() as f32 / greedy_dt.max(1e-6);
        let spec_tps = spec.len() as f32 / spec_dt.max(1e-6);
        let text = tokenizer.decode(&spec, false).unwrap_or_default();
        eprintln!("[spec-speedup] continuation={text:?}");
        eprintln!(
            "[spec-speedup] greedy {greedy_tps:.2} tok/s ({} tok, {greedy_dt:.2}s) | speculative \
             {spec_tps:.2} tok/s ({} tok, {steps} steps ⇒ {:.2} tok/step, {spec_dt:.2}s) | \
             SPEEDUP {:.2}×",
            greedy.len(),
            spec.len(),
            spec.len() as f32 / steps.max(1) as f32,
            spec_tps / greedy_tps,
        );
        // Speculative commits whole blocks, so it may overshoot MAX_NEW by up to block_size-1; the
        // tokens must be identical on the common prefix (lossless — accepted tokens are the target's
        // own argmaxes).
        let n = spec.len().min(greedy.len());
        assert_eq!(
            &spec[..n],
            &greedy[..n],
            "speculative + batched verify must be BIT-IDENTICAL to greedy decode"
        );
        Ok(())
    }

    /// DSpark acceptance on REAL (non-counting) text — the honest metric. Counting is trivially
    /// predictable; this drafts an open-ended coding answer and reports mean accepted length
    /// (tokens/step) + the per-position histogram, the number the research target (~5+/6) is about.
    /// It does NOT assert bit-identical to greedy decode (the batched-verify prefill path is only
    /// tolerance-equal to the decode path on subtle tokens — a separate correctness item); it
    /// measures how well the drafter conditions on real context via the sliding target-hidden window.
    #[test]
    #[ignore]
    fn wave_speculative_realtext_acceptance() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let (Some(pack), Some(dspark)) =
            (gate_packs::target(&device)?, gate_packs::drafter(&device)?)
        else {
            eprintln!("[skip] target or DSpark drafter absent");
            return Ok(());
        };
        let (engine, drafter) =
            Engine::load_with_drafter(&pack, Some(&dspark), &DEEPSEEK_V4, &device)?;
        let model =
            BatchedEngine::new(engine)?.with_drafter(drafter.expect("dspark path given"))?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let eos = tokenizer.token_to_id("<｜end▁of▁sentence｜>").expect("eos");
        let n_layers = ManagedBatchedModel::num_layers(&model);
        let max_draft = 5usize;
        const MAX_NEW: usize = 128;

        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Write a Python function that returns the \
             nth Fibonacci number, with a short docstring.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let mut committed = step
            .logits
            .as_ref()
            .ok_or_else(|| candle::Error::msg("no prefill logits"))?[0]
            .i(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        let mut gen = vec![committed];
        let mut steps = 0usize;
        let mut accepts: Vec<usize> = Vec::new();
        let t0 = std::time::Instant::now();
        while gen.len() < MAX_NEW && committed != eos {
            steps += 1;
            let before = gen.len();
            let next = model.speculative_decode_step(
                &mut session,
                seq,
                committed,
                max_draft,
                n_layers,
                &mut GreedyChooser::whole_row(),
                &mut |t| {
                    gen.push(t);
                    gen.len() < MAX_NEW && t != eos
                },
            )?;
            accepts.push(gen.len() - before);
            match next {
                Some(c) => committed = c,
                None => break,
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let text = tokenizer.decode(&gen, false).unwrap_or_default();
        let mean = gen.len() as f32 / steps.max(1) as f32;
        // Histogram of committed-per-step (1 = no draft accepted … max_draft+1 = full block + bonus).
        let mut hist = vec![0usize; max_draft + 2];
        for &a in &accepts {
            hist[a.min(max_draft + 1)] += 1;
        }
        eprintln!("[spec-real] continuation={text:?}");
        eprintln!("[spec-real] per-step accepted: {accepts:?}");
        eprintln!(
            "[spec-real] {} tokens, {steps} steps ⇒ {mean:.2} tokens/step (max_draft={max_draft}); \
             {:.2} tok/s; committed-per-step histogram (idx=tokens): {hist:?}",
            gen.len(),
            gen.len() as f32 / dt.max(1e-6),
        );
        #[cfg(feature = "profile")]
        {
            // Aggregate forward_wave phase timings — dominated by the verify_block prefills, so this
            // shows WHERE the ~370 ms/step verify goes (attention vs MoE vs compressor/gallery/seal).
            let snap = crate::models::profile::pipeline_snapshot_and_reset();
            let mut es = snap.entries.clone();
            es.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[verify-prof] forward_wave phases (total ms across run, count):");
            for (name, ms, count) in es.iter().take(24) {
                eprintln!("  {name:34} {ms:9.1}ms  x{count}");
            }
        }
        Ok(())
    }
}
