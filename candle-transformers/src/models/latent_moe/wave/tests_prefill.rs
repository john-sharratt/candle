//! End-to-end tests of the batched prefill path against the real checkpoint.
//!
//! Each test is ignored by default because it needs a checkpoint and CUDA, and
//! each takes the crate-wide GPU lock so two model loads never overlap.
//! `wave_prefill_then_decode_end_to_end` is the gate on the wave's trait surface:
//! a prefill followed by greedy decode must reproduce the expected answer. The
//! probes that follow are discriminators run when that gate fails. They name the
//! layer or the state at which batched prefill and per-token decode first differ.

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

    use candle::IndexOp;

    // Full-model integration tests serialize on the CRATE-WIDE
    // [`gpu_test_lock::gpu_serial`], not a lock local to this module. Each loads
    // a multi-hundred-GB checkpoint with its multi-GB PINNED expert pool, so two
    // running at once exhaust host page-locked memory (`cuMemAllocHost` OOM) —
    // and the tests that must not overlap no longer all live in this file (the
    // batched-forwarding sweep is in `models::deepseek4`). A module-scoped lock
    // would serialize this file against itself and nothing else, which is the
    // failure mode `gpu_test_lock`'s own docs were written about.

    /// The wave path end to end: one batched prefill followed by decode waves
    /// through `ManagedBatchedModel::forward_wave`, greedy-decoded to EOS,
    /// must reproduce the continuation the per-token kernel engine produces for
    /// the same prompt. This is the semantic gate on the scheduler-facing trait
    /// surface — that prefill, the arena, selection, and the decode loop compose
    /// into a coherent generation rather than each merely passing in isolation.
    ///
    /// The prompt/expected pair is a fixture (a capital-city question), not the
    /// subject: what is under test is the wave loop. Model-quality gates live in
    /// `models::deepseek4`. Ignored (needs a checkpoint + CUDA).
    #[test]
    #[ignore]
    fn wave_prefill_then_decode_end_to_end() -> Result<()> {
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

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = ManagedBatchedModel::num_layers(&model);
        super::super::super::readback::reset_readbacks();

        // Prefill the whole prompt in one wave.
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let t0 = std::time::Instant::now();
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
        eprintln!(
            "[wave] prefill {} tokens in {:.1}s",
            ids.len(),
            t0.elapsed().as_secs_f32()
        );
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        if logits.is_empty() {
            return Err(candle::Error::msg("prefill wave produced no logits"));
        }
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        // Greedy decode until EOS (bounded) through decode waves. STRICT gate:
        // the answer must be exactly "Paris" followed by EOS — the same stream
        // the per-token kernel engine produces; anything longer is wave-path
        // degradation, not a style choice.
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");
        let mut gen = vec![next];
        let mut decode_waves = 0usize;
        let t1 = std::time::Instant::now();
        while gen.len() < 12 && next != eos {
            let tok = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&tok),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            decode_waves += 1;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("decode wave produced no logits"));
            }
            next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            gen.push(next);
        }
        let dt = t1.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave] generated ids={gen:?}");
        eprintln!("[wave] continuation={text:?}");
        eprintln!(
            "[wave] {decode_waves} decode waves in {dt:.1}s = {:.2} tok/s",
            decode_waves as f32 / dt
        );
        assert_eq!(
            *gen.last().unwrap(),
            eos,
            "wave path did not stop on EOS within 12 tokens: {text:?}"
        );
        let answer = tokenizer
            .decode(&gen[..gen.len() - 1], false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        assert_eq!(
            answer.trim(),
            "Paris",
            "wave path must answer exactly \"Paris\": {text:?}"
        );

        // Readback budget: the wave forward makes no device→host transfer at
        // all — the MoE routing reaches the expert pipeline as a summary
        // bucketize writes into mapped pinned memory, never a blocking read. Sampling (`to_scalar` above) is the
        // one-per-token the budget allows and belongs to the caller.
        let got = super::super::super::readback::readback_count();
        assert_eq!(got, 0, "wave-path readbacks: {got}");
        Ok(())
    }

    /// **Failed-wave corpus atomicity.** A wave that dies mid-sweep must leave
    /// the streaming corpus (galleries + compressor states + `absorbed`)
    /// exactly as it entered — the retry otherwise absorbs the same positions
    /// twice, and the duplicated, shifted groups corrupt every later
    /// selection (the repeat-itself failure [`VerifySnapshot`] documents).
    ///
    /// Simulated rather than fault-injected, because a failed wave IS
    /// precisely "the forward ran, nothing was committed": run a decode wave,
    /// discard its result without advancing the session, restore the capture,
    /// and require the continuation to match a reference sequence that never
    /// took the detour. The capture/restore pair under test is exactly what
    /// `WaveSweep::sweep`'s failure bracket runs.
    #[test]
    #[ignore]
    fn wave_failed_sweep_leaves_no_corpus_trace() -> Result<()> {
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
        // A prompt whose greedy continuation runs long enough to decode
        // through several compressor groups on both sides of the detour.
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Tell me a short story about a \
             lighthouse keeper.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let n_layers = ManagedBatchedModel::num_layers(&model);

        // Prefill + commit; returns the greedy next token.
        let prefill = |model: &BatchedEngine,
                       session: &mut BatchedInferenceSession,
                       seq: usize|
         -> Result<u32> {
            let t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let step = model.forward_wave(
                session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, ids.len())?;
            step.logits_owned()?[0].i(0)?.argmax(0)?.to_scalar::<u32>()
        };
        // One COMMITTED greedy decode step; returns the next token.
        let decode_step = |model: &BatchedEngine,
                           session: &mut BatchedInferenceSession,
                           seq: usize,
                           tok: u32|
         -> Result<u32> {
            let t = Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            step.logits_owned()?[0].i(0)?.argmax(0)?.to_scalar::<u32>()
        };

        // Reference: prefill + 9 committed tokens, no detour.
        let seq_a = session.create_sequence()?;
        let mut tok = prefill(&model, &mut session, seq_a)?;
        let mut reference = vec![tok];
        for _ in 0..9 {
            tok = decode_step(&model, &mut session, seq_a, tok)?;
            reference.push(tok);
        }

        // Perturbed: same prompt, 4 committed tokens, then a DISCARDED wave
        // (the forward runs, the session is not advanced — a failed wave's
        // exact residue), restore, and 5 more committed tokens.
        let seq_b = session.create_sequence()?;
        let mut tok = prefill(&model, &mut session, seq_b)?;
        let mut stream = vec![tok];
        for _ in 0..4 {
            tok = decode_step(&model, &mut session, seq_b, tok)?;
            stream.push(tok);
        }
        assert_eq!(
            stream[..5],
            reference[..5],
            "streams diverged before the detour — the corpus lifecycle is not \
             what this failure is about"
        );

        let snaps = model.capture_corpus_state(&[seq_b])?;
        let absorbed_before = snaps[0].state.as_ref().map(|(a, _)| *a).unwrap();
        let glens_before: Vec<usize> = snaps[0]
            .state
            .as_ref()
            .map(|(_, ls)| ls.iter().map(|(_, _, g)| *g).collect())
            .unwrap();

        // The doomed wave: full forward, result discarded, nothing committed.
        {
            let t = Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu)?;
            let _ = model.forward_wave(
                &mut session,
                &[seq_b],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
        }
        // Self-check: the doomed wave really advanced the corpus — otherwise
        // this test exercises nothing.
        let mid = model.capture_corpus_state(&[seq_b])?;
        let absorbed_mid = mid[0].state.as_ref().map(|(a, _)| *a).unwrap();
        assert_eq!(
            absorbed_mid,
            absorbed_before + 1,
            "the discarded wave absorbed nothing — the detour is not reaching \
             the corpus state"
        );

        model.restore_corpus_state(snaps)?;
        let after = model.capture_corpus_state(&[seq_b])?;
        let absorbed_after = after[0].state.as_ref().map(|(a, _)| *a).unwrap();
        let glens_after: Vec<usize> = after[0]
            .state
            .as_ref()
            .map(|(_, ls)| ls.iter().map(|(_, _, g)| *g).collect())
            .unwrap();
        assert_eq!(absorbed_after, absorbed_before, "absorbed not restored");
        assert_eq!(glens_after, glens_before, "gallery lengths not restored");

        for _ in 0..5 {
            tok = decode_step(&model, &mut session, seq_b, tok)?;
            stream.push(tok);
        }
        assert_eq!(
            stream, reference,
            "continuation diverged after a discarded wave + restore — the \
             corpus state was not returned to its entry value"
        );
        Ok(())
    }

    /// Bisect between the wave's two absorption paths. Same prompt as
    /// [`wave_chat_template_oracle`], but absorbed one token at a time through
    /// the DECODE path with no batched prefill — the per-token regime the
    /// single-session kernel engine already proves correct on the same weights.
    ///
    /// It is the discriminator, not a gate in its own right: a coherent
    /// generation here means batched prefill
    /// (`prefill_layer_rows`/`write_contiguous`/`set_len`) corrupts state, while
    /// an incoherent one means the wave decode state containers themselves
    /// diverge. Run it when the end-to-end gate fails.
    #[test]
    #[ignore]
    fn wave_decode_only_absorption_bisect() -> Result<()> {
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
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
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

        let mut step_row = |tok: u32| -> Result<u32> {
            let t = Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("decode wave produced no logits"));
            }
            logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()
        };

        // Absorb the prompt through decode steps (per-token regime).
        let mut next = 0u32;
        for &t in &ids {
            next = step_row(t)?;
        }
        let mut gen = vec![next];
        while gen.len() < 16 && next != eos {
            next = step_row(next)?;
            gen.push(next);
        }
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave-bisect] generated ids={gen:?}");
        eprintln!("[wave-bisect] continuation={text:?}");
        eprintln!("[wave-bisect] crisp={}", gen.len() >= 2 && gen[1] == eos);
        Ok(())
    }

    /// Layer-resolved divergence probe: run BOTH absorption paths segmented
    /// one layer at a time (residual pause/resume), capturing the residual at
    /// every layer boundary. The first layer whose residual diverges between
    /// batched prefill and per-token decode names the defective computation.
    /// Also prints each layer's kind + gallery size for the interpretation.
    #[test]
    #[ignore]
    fn wave_prefill_residual_divergence() -> Result<()> {
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
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let n = ids.len();
        let n_layers = ManagedBatchedModel::num_layers(&model);

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq_a = session.create_sequence()?; // per-token decode absorb
        let seq_b = session.create_sequence()?; // batched prefill

        // A: per-token, layer-segmented; keep residual after each layer for
        // every token: res_a[l] = [n] rows.
        let mut res_a: Vec<Vec<Tensor>> = vec![Vec::with_capacity(n); n_layers];
        for &t in &ids {
            let tt = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
            let mut resid = model.claim_residual(&session, 1)?;
            for l in 0..n_layers {
                model.forward_wave(
                    &mut session,
                    &[seq_a],
                    std::slice::from_ref(&tt),
                    &[],
                    &[],
                    &[],
                    &[],
                    l,
                    l + 1,
                    Some(&mut resid),
                )?;
                // The final layer runs the head; every other hands its residual
                // on through the arena.
                if l + 1 < n_layers {
                    res_a[l].push(resid.committed().to_tensor()?.unsqueeze(0)?);
                }
            }
            session.advance_sequence(seq_a, 1)?;
        }

        // B: batched prefill, layer-segmented; res_b[l] = [1, n, hc*dim].
        let pt = Tensor::from_vec(ids.clone(), (1, n), &Device::Cpu)?;
        let mut res_b: Vec<Tensor> = Vec::with_capacity(n_layers);
        let mut resid = model.claim_residual(&session, n)?;
        for l in 0..n_layers {
            model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq_b],
                std::slice::from_ref(&pt),
                &[],
                &[],
                l,
                l + 1,
                Some(&mut resid),
            )?;
            if l + 1 < n_layers {
                res_b.push(resid.committed().to_tensor()?.unsqueeze(0)?);
            }
        }
        session.advance_sequence(seq_b, n)?;

        // Compare per layer: prefill row t vs decode step t residual.
        let state = model.seq_state.read().unwrap();
        let ea = state.get(&seq_a).expect("seq_a state");
        for l in 0..res_b.len() {
            let rb = &res_b[l]; // [1, n, hcdim]
            let hcdim = rb.dim(2)?;
            let mut worst = 0f32;
            let mut worst_t = 0usize;
            for t in 0..n {
                let a_row = res_a[l][t].reshape(hcdim)?.to_dtype(DType::F32)?;
                let b_row = rb.i((0, t))?.reshape(hcdim)?.to_dtype(DType::F32)?;
                let d = (a_row - b_row)?.abs()?.max_all()?.to_scalar::<f32>()?;
                if d > worst {
                    worst = d;
                    worst_t = t;
                }
            }
            let glen = ea.layers[l].gallery.as_ref().map_or(0, |g| g.len());
            let kind = {
                let e = model.engine.engine_layer(l);
                format!("{:?}", e.attn.kind())
            };
            eprintln!(
                "[resid-div] layer {l:2} ({kind:>13}, gallery {glen:2}): max|Δresidual| = {worst:.6} (worst token {worst_t})"
            );
        }
        Ok(())
    }

    /// Prefill-vs-decode state audit: absorb the SAME prompt into two
    /// sequences — one through per-token decode waves (proven crisp), one
    /// through the batched prefill — then diff every layer's arena window
    /// bytes, corpus state, and finally the next decode step's logits. The
    /// first divergent artifact is the batched-prefill bug.
    #[test]
    #[ignore]
    fn wave_prefill_state_matches_decode_steps() -> Result<()> {
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
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let n = ids.len();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq_a = session.create_sequence()?; // decode-step absorb (reference)
        let seq_b = session.create_sequence()?; // batched prefill (suspect)
        let n_layers = ManagedBatchedModel::num_layers(&model);

        // A: per-token decode waves.
        for &t in &ids {
            let tt = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
            model.forward_wave(
                &mut session,
                &[seq_a],
                std::slice::from_ref(&tt),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq_a, 1)?;
        }
        // B: one batched prefill wave.
        let pt = Tensor::from_vec(ids.clone(), (1, n), &Device::Cpu)?;
        let step_b = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq_b],
            std::slice::from_ref(&pt),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq_b, n)?;
        let logits_b = step_b.logits_owned()?.swap_remove(0);
        let next_b = logits_b.i(0)?.argmax(0)?.to_scalar::<u32>()?;
        eprintln!("[state-diff] prefill argmax token={next_b}");

        // Window bytes per layer. The decode-absorbed slot's HOST population
        // count lags one token (the final token was written in-kernel with no
        // host set_len after it), so compare the fully-covered [0, n-1).
        let cmp_len = n - 1;
        for l in 0..n_layers {
            let b = &session.backings()[l];
            let (ka, _) = b.read_contiguous(seq_a, 0, cmp_len)?;
            let (kb, _) = b.read_contiguous(seq_b, 0, cmp_len)?;
            let d = (ka.to_dtype(DType::F32)? - kb.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            if d != 0.0 {
                eprintln!("[state-diff] layer {l}: window max|Δ| = {d}");
            }
        }
        // Corpus state per layer.
        {
            let state = model.seq_state.read().unwrap();
            let ea = state.get(&seq_a).expect("seq_a state");
            let eb = state.get(&seq_b).expect("seq_b state");
            for l in 0..n_layers {
                let ga = ea.layers[l].gallery.as_ref();
                let gb = eb.layers[l].gallery.as_ref();
                let (la, lb) = (ga.map_or(0, |g| g.len()), gb.map_or(0, |g| g.len()));
                if la != lb {
                    eprintln!("[state-diff] layer {l}: gallery len {la} vs {lb}");
                    continue;
                }
                if let (Some(ga), Some(gb)) = (ga, gb) {
                    if la > 0 {
                        let ea_t = ga.attn_entries()?.to_dtype(DType::F32)?;
                        let eb_t = gb.attn_entries()?.to_dtype(DType::F32)?;
                        let d = (&ea_t - &eb_t)?.abs()?.max_all()?.to_scalar::<f32>()?;
                        if d != 0.0 {
                            let na = ea_t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            let nb = eb_t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            let va: Vec<f32> = ea_t.i((0, 0..4))?.to_vec1::<f32>()?;
                            let vb: Vec<f32> = eb_t.i((0, 0..4))?.to_vec1::<f32>()?;
                            eprintln!(
                                "[state-diff] layer {l}: gallery entries max|Δ| = {d} \
                                 |A|={na:.4} |B|={nb:.4} A[0][..4]={va:?} B[0][..4]={vb:?}"
                            );
                        }
                        let pa = ga.positions()?.to_vec1::<u32>()?;
                        let pb = gb.positions()?.to_vec1::<u32>()?;
                        if pa != pb {
                            eprintln!("[state-diff] layer {l}: gallery positions {pa:?} vs {pb:?}");
                        }
                    }
                }
            }
        }

        // Next decode step on both with the same token.
        let tok = Tensor::from_vec(vec![next_b], (1, 1), &Device::Cpu)?;
        let toks = [tok.clone(), tok];
        let step = model.forward_wave(
            &mut session,
            &[seq_a, seq_b],
            &toks,
            &[],
            &[],
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        let logits = step.logits_owned()?;
        let arg_a = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let arg_b = logits[1].i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let dl = (logits[0].to_dtype(DType::F32)? - logits[1].to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        eprintln!("[state-diff] next-step argmax: decode-absorbed={arg_a} prefill-absorbed={arg_b} max|Δlogits|={dl}");
        assert_eq!(
            arg_a, arg_b,
            "decode step after prefill diverges from decode step after per-token absorb"
        );
        Ok(())
    }

    /// Oracle for the conversation gate: the EXACT chat-templated token stream
    /// the conversation engine assembles (BOS + system text + the user/assistant
    /// role markers around the question), fed as ONE plain contiguous prefill
    /// through the reference wave path and greedy-decoded to EOS.
    ///
    /// What it prints is what the conversation engine MUST reproduce, which is
    /// what makes it a bisect: a coherent generation here with an incoherent one
    /// from the conversation engine puts the fault in that engine's glue and
    /// position wiring, not in the prompt or the wave path. The role markers are
    /// the model's, taken from its `Dialect`; the wave path never sees them as
    /// anything but token ids.
    #[test]
    #[ignore]
    fn wave_chat_template_oracle() -> Result<()> {
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
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        eprintln!("[wave-conv] prompt ids={ids:?}");
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
            return Err(candle::Error::msg("prefill wave produced no logits"));
        }
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        let mut gen = vec![next];
        while gen.len() < 24 && next != eos {
            let tok = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&tok),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            let logits = step.logits_owned()?;
            if logits.is_empty() {
                return Err(candle::Error::msg("decode wave produced no logits"));
            }
            next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            gen.push(next);
        }
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave-conv] generated ids={gen:?}");
        eprintln!("[wave-conv] continuation={text:?}");
        eprintln!(
            "[wave-conv] stopped_on_eos={} after {} tokens",
            next == eos,
            gen.len()
        );
        Ok(())
    }
}
