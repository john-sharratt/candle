//! Oracle tests that the hybrid wave matches an F32 reference, and that a forward
//! split in two matches the same forward run whole.
//!
//! The reference is an F32 implementation on the CPU with no wave, paged KV,
//! quantized projections or recurrent bookkeeping, so agreement with it shows
//! that the engine's arithmetic is right. Where they disagree, a sweep over prefix
//! lengths locates the fault: a smooth decay is accumulated precision, and a step
//! at a chunk boundary is a carry that went wrong. A forward split at a point
//! must land on the logits of the whole forward, which checks that the recurrent
//! state and the paged KV compose across the split. Decode is checked on its own,
//! since it runs a different kernel.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::batch_test::test_helpers::{gate_pack, hf_get};

    use crate::models::batch_test::utils::TestParams;

    use crate::models::batched_inference::BatchedConfig;

    use crate::models::dialect::Dialect;

    use crate::models::model_pack::{PackFamily, PackRequest};

    use crate::models::quantized_qwen35::{from_pack, TOKENIZER_REPO, TOKENIZER_REV};

    use crate::models::qwen35::loader::load_reference_model;

    use candle::quantized::gguf_file::Content;

    use candle::quantized::Int8Mode;

    use hf_hub::RepoType;

    use std::io::{BufReader, Seek, SeekFrom};

    /// The BF16 0.8B these oracles compare against: the checkpoint the F32
    /// reference reads, packed for the wave at `Int8Mode::Off` so both sides do
    /// the same arithmetic on the same weights.
    pub(in super::super) const BF16_0_8B: (&str, &str, &str) = (
        "unsloth/Qwen3.5-0.8B-GGUF",
        "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
        "Qwen3.5-0.8B-BF16.gguf",
    );

    /// [`BF16_0_8B`] loaded through its pack.
    pub(in super::super) fn bf16_wave_model(device: &Device) -> Result<HybridBatched> {
        let request = PackRequest::of(
            PackFamily::Qwen35,
            BF16_0_8B,
            (TOKENIZER_REPO, TOKENIZER_REV),
            Some(Int8Mode::Off),
        );
        from_pack(&gate_pack(&request, device)?, device)
    }

    /// **The wave's oracle.** Prefill the same tokens through the F32
    /// reference — which is token-identical to llama.cpp — and through
    /// `forward_wave`, and require the same next token.
    ///
    /// This is the check that separates "the engine is wrong" from "the model
    /// answered differently than the fixture expects". A gate that generates
    /// fluent but off-task text tells you nothing about which of the two you
    /// are looking at; this does, because the reference has no wave, no paged
    /// KV, no quantized projections and no recurrent bookkeeping — it is the
    /// arithmetic alone.
    ///
    /// The 0.8B on purpose: it is the largest of the family that dequantizes
    /// to F32 in host RAM (the 9B would need ~36 GB), and a wave bug is
    /// geometry, not scale — the same 3:1 schedule, the same partial rotary,
    /// the same head_dim 256.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-0.8B GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                qwen35::forward::tests::wave_matches_the_reference -- --ignored --nocapture"]
    fn wave_matches_the_reference() -> Result<()> {
        let (repo, rev, file) = BF16_0_8B;
        let path = hf_get(repo, RepoType::Model, rev, file)?;
        let tok_path = hf_get(
            TOKENIZER_REPO,
            RepoType::Model,
            TOKENIZER_REV,
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::Msg(format!("load tokenizer: {e}")))?;

        // A prompt with the family's real turn structure, so a mis-encoded
        // special token would show up here rather than only in the gate.
        let prompt_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
                           <|im_start|>user\nThe capital of France is<|im_end|>\n\
                           <|im_start|>assistant\n<think>\n\n</think>\n\n";
        let tokens: Vec<u32> = tokenizer
            .encode(prompt_text, false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", tokens.len());
        // Every marker must be one id, not spelled out — the gate's symptom
        // (the model emitting "user") is what a split marker looks like.
        let decoded = tokenizer
            .decode(&tokens, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        assert_eq!(decoded, prompt_text, "prompt did not round-trip");

        // ── The reference, on CPU, in F32 ──
        let file = std::fs::File::open(&path)?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader)?;
        reader.seek(SeekFrom::Start(0))?;
        let reference = load_reference_model(&content, &mut reader, &Device::Cpu)?;

        // Prompt length is the first thing to rule out when a row goes
        // non-finite: the chunked scan carries state across 64-token chunks,
        // and a short probe would never reach the second one.
        let finite = |t: &Tensor| -> Result<bool> {
            let m = t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            Ok(m.is_finite())
        };
        let plain: Vec<u32> = tokenizer
            .encode(
                "The quick brown fox jumps over the lazy dog. Pack my box with five \
                 dozen liquor jugs. How vexingly quick daft zebras jump. The five \
                 boxing wizards jump quickly at dawn every single morning.",
                false,
            )
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        for (label, ids) in [("plain", &plain), ("chat", &tokens)] {
            // One full-length forward answers the question in the passing
            // case; the per-prefix sweep — a session and a forward per length,
            // O(n²) token-work — runs only on failure, where locating the
            // first bad length is exactly what it is for. Sweeping
            // unconditionally made this the slowest part of the suite, for a
            // diagnostic no passing run reads.
            let mut st = reference.new_session()?;
            let full = reference.forward(ids, &mut st)?;
            if finite(&full)? {
                println!(
                    "  reference [{label}]: finite over all {} tokens",
                    ids.len()
                );
                continue;
            }
            let mut first_bad = None;
            for n in 1..=ids.len() {
                let mut st = reference.new_session()?;
                let l = reference.forward(&ids[..n], &mut st)?;
                if !finite(&l)? {
                    first_bad = Some(n);
                    break;
                }
            }
            let n = first_bad.expect("the full forward was non-finite, so some prefix is");
            println!(
                "  reference [{label}]: finite up to {} tokens, non-finite at {n} \
                 (token id {})",
                n - 1,
                ids[n - 1]
            );
        }

        let mut ref_state = reference.new_session()?;
        let ref_logits = reference.forward(&tokens, &mut ref_state)?;
        let ref_last = ref_logits.narrow(0, tokens.len() - 1, 1)?.squeeze(0)?;
        assert!(
            finite(&ref_last)?,
            "the reference itself produced a non-finite logit row — fix the \
             oracle before comparing anything to it"
        );
        let ref_argmax = ref_last.argmax(0)?.to_scalar::<u32>()?;

        // ── The production wave, on the GPU ──
        let device = Device::new_cuda(0)?;
        let model = bf16_wave_model(&device)?;
        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let ids = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &device)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            &[ids],
            &[],
            &[],
            0,
            model.num_layers(),
            None,
        )?;
        let logits = step
            .logits
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("wave returned no logits".into()))?;
        let wave_last = logits[0].flatten_all()?.to_device(&Device::Cpu)?;
        let wave_argmax = wave_last.argmax(0)?.to_scalar::<u32>()?;
        // The logits sit on the forward span and `WaveResult` holds its guard,
        // which is what stops the span being reclaimed underneath them. That
        // guard also *is* the live wave as far as the partition is concerned,
        // so anything that would create an arena — the second session below —
        // has to wait for it. `wave_last` is already a host copy.
        drop(step);
        drop(session);

        let name = |id: u32| {
            tokenizer
                .decode(&[id], false)
                .unwrap_or_else(|_| format!("<{id}>"))
        };
        println!(
            "reference → {ref_argmax} {:?}   wave → {wave_argmax} {:?}",
            name(ref_argmax),
            name(wave_argmax)
        );

        // Cosine over the whole row, reported before the assert so a near-miss
        // (quantization) reads differently from a structural break.
        let a = ref_last.to_dtype(DType::F32)?;
        let b = wave_last.to_dtype(DType::F32)?;
        let dot = a.mul(&b)?.sum_all()?.to_scalar::<f32>()?;
        let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let cos = dot / (na * nb).max(1e-6);
        println!("logit cosine {cos:.6}");

        assert_eq!(
            ref_argmax,
            wave_argmax,
            "the wave and the reference disagree on the next token \
             ({ref_argmax} {:?} vs {wave_argmax} {:?}, cosine {cos:.4})",
            name(ref_argmax),
            name(wave_argmax)
        );
        assert!(cos > 0.99, "logit rows diverged: cosine {cos}");

        // ── The same comparison on the gate's own prompt ──
        //
        // ~800 tokens of system instruction plus the story, which is the shape
        // the story-rewrite gate actually runs. A short prompt exercises one
        // chunk of the recurrent scan and one page of KV; this one crosses
        // several of both, so it is where a carry between chunks or a stale
        // page shows up. Agreement here says the gate's answer — whatever it
        // is — is the model's, not the engine's.
        let params = TestParams::new(4, &std::fs::read_to_string(&tok_path)?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true);
        let mut gate_tokens = params.system_prompt_tokens(0);
        gate_tokens.extend(params.user_prompt_tokens(0));
        println!("gate prompt: {} tokens", gate_tokens.len());

        // Swept by prefix length rather than measured once at the end. A single
        // number cannot tell precision loss from a structural fault; the shape
        // of the curve can. Smooth decay is accumulation (the wave computes in
        // the session's activation dtype, the reference in F32); a step at a
        // boundary is a carry between recurrent chunks or a KV page.
        // Two KV dtypes, because they answer different questions. The
        // reference stores nothing — it keeps every layer's whole history in
        // F32 — so an F32 cache is the closest the engine gets to it and any
        // gap there is the engine's. BF16 is what a session actually runs, and
        // the difference between the two curves is the price of the cache.
        let mut worst = 1.0f32;
        for kv_dtype in [DType::F32, DType::BF16] {
            println!("  ── KV {kv_dtype:?} ──");
            worst = worst.min(sweep(
                &reference,
                &model,
                &device,
                &gate_tokens,
                kv_dtype,
                &name,
            )?);
        }
        assert!(
            worst > 0.99,
            "the wave drifts from the reference as context grows (worst cosine \
             {worst:.4}) — accumulation would be gradual and small; this is not"
        );

        // ── Decode ──
        //
        // Prefill and decode share almost nothing: different attention kernel,
        // and on the recurrent side a fused one-token step that updates the
        // state in place rather than the chunked scan. A prefill that matches
        // the reference exactly says nothing about either. Greedy on both
        // sides, so a divergence shows up as a different token and not as a
        // sampling difference.
        let mut ref_state = reference.new_session()?;
        let ref_logits = reference.forward(&gate_tokens, &mut ref_state)?;
        let mut ref_tok = ref_logits
            .narrow(0, gate_tokens.len() - 1, 1)?
            .squeeze(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        let mut session =
            model.create_batched_session(BatchedConfig::default().with_dtype(DType::BF16))?;
        let seq = session.create_sequence()?;
        let ids = Tensor::from_vec(gate_tokens.clone(), (1, gate_tokens.len()), &device)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            &[ids],
            &[],
            &[],
            0,
            model.num_layers(),
            None,
        )?;
        let mut wave_tok = step.logits.as_ref().expect("logits")[0]
            .flatten_all()?
            .argmax(0)?
            .to_scalar::<u32>()?;
        drop(step);
        assert_eq!(
            ref_tok,
            wave_tok,
            "first sampled token already differs ({:?} vs {:?})",
            name(ref_tok),
            name(wave_tok)
        );

        for i in 0..8 {
            let rl = reference.forward(&[ref_tok], &mut ref_state)?;
            ref_tok = rl.get(0)?.argmax(0)?.to_scalar::<u32>()?;

            let ids = Tensor::from_vec(vec![wave_tok], (1, 1), &device)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                &[ids],
                &[],
                &[],
                &[],
                &[],
                0,
                model.num_layers(),
                None,
            )?;
            wave_tok = step.logits.as_ref().expect("logits")[0]
                .flatten_all()?
                .argmax(0)?
                .to_scalar::<u32>()?;
            drop(step);

            println!(
                "  decode {i}: reference {:?}   wave {:?}{}",
                name(ref_tok),
                name(wave_tok),
                if ref_tok == wave_tok {
                    ""
                } else {
                    "   ← DIVERGED"
                }
            );
            assert_eq!(
                ref_tok,
                wave_tok,
                "decode step {i} diverged: reference {:?}, wave {:?}",
                name(ref_tok),
                name(wave_tok)
            );
        }
        Ok(())
    }

    /// **The wave against itself.** Prefilling `[0, n)` in one forward and in
    /// two must land on the same logits.
    ///
    /// This needs no oracle, which is what makes it worth having: the
    /// reference is a second implementation and can be wrong (it was — a
    /// chunked-scan overflow put NaN in its logits at 23 tokens), so a
    /// disagreement between the two says only that they differ. Splitting the
    /// *same* implementation says something sharper. Everything that carries
    /// across the split — the recurrent `S` and conv tail, the paged K/V, the
    /// offsets — has to compose, and a carry that is dropped, double-applied
    /// or written at the wrong position shows up here and nowhere in a
    /// single-shot run.
    ///
    /// Splits are chosen against `CHUNK_SIZE` (32): inside the first chunk, on
    /// its boundary, and past it.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-0.8B GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                qwen35::forward::tests::wave_one_shot_equals_segmented -- --ignored --nocapture"]
    fn wave_one_shot_equals_segmented() -> Result<()> {
        let tok_path = hf_get(
            TOKENIZER_REPO,
            RepoType::Model,
            TOKENIZER_REV,
            "tokenizer.json",
        )?;
        let params = TestParams::new(4, &std::fs::read_to_string(&tok_path)?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true);
        let mut tokens = params.system_prompt_tokens(0);
        tokens.extend(params.user_prompt_tokens(0));
        tokens.truncate(200);

        let device = Device::new_cuda(0)?;
        let model = bf16_wave_model(&device)?;

        // One forward over `[0, n)`, optionally split at `cut`.
        let run = |cut: Option<usize>| -> Result<Tensor> {
            let mut session =
                model.create_batched_session(BatchedConfig::default().with_dtype(DType::F32))?;
            let seq = session.create_sequence()?;
            let bounds: Vec<(usize, usize)> = match cut {
                None => vec![(0, tokens.len())],
                Some(c) => vec![(0, c), (c, tokens.len())],
            };
            let mut last = None;
            for (a, b) in bounds {
                let ids = Tensor::from_vec(tokens[a..b].to_vec(), (1, b - a), &device)?;
                let step = model.forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &[seq],
                    &[ids],
                    &[],
                    &[],
                    0,
                    model.num_layers(),
                    None,
                )?;
                let row = step
                    .logits
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("no logits".into()))?[0]
                    .flatten_all()?
                    .to_device(&Device::Cpu)?;
                drop(step);
                // What the next segment will build its RoPE positions, causal
                // mask and admission truncation from. All three read the same
                // number, so if it is wrong the segment is not merely
                // imprecise — it attends the wrong span at the wrong offsets.
                // The session's own offset is advanced lazily — the next
                // forward's entry reconciliation adopts the backing — so the
                // backing is the number that has to be right here.
                let backing = session.sequence_backing_tokens(seq).unwrap_or(usize::MAX);
                assert_eq!(
                    backing, b,
                    "after prefilling [{a}, {b}) the KV should hold {b} tokens"
                );
                last = Some(row);
            }
            last.ok_or_else(|| candle::Error::Msg("no segments".into()))
        };

        let whole = run(None)?;
        let cos = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let a = a.to_dtype(DType::F32)?;
            let b = b.to_dtype(DType::F32)?;
            let dot = a.mul(&b)?.sum_all()?.to_scalar::<f32>()?;
            let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            Ok(dot / (na * nb).max(1e-6))
        };

        // The same split through the F32 reference. It shares the mixer core
        // (`delta_net_mix`) with the production path but none of the paged KV,
        // so a split it survives and the wave does not is attention-side, and
        // one they both fail is the recurrence composing wrongly. It reads the
        // checkpoint itself: the pack holds the streamed projections repacked.
        let (repo, rev, file) = BF16_0_8B;
        let path = hf_get(repo, RepoType::Model, rev, file)?;
        let file = std::fs::File::open(&path)?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader)?;
        reader.seek(SeekFrom::Start(0))?;
        let reference = load_reference_model(&content, &mut reader, &Device::Cpu)?;
        let ref_run = |cut: Option<usize>| -> Result<Tensor> {
            let mut st = reference.new_session()?;
            let bounds: Vec<(usize, usize)> = match cut {
                None => vec![(0, tokens.len())],
                Some(c) => vec![(0, c), (c, tokens.len())],
            };
            let mut last = None;
            for (a, b) in bounds {
                let l = reference.forward(&tokens[a..b], &mut st)?;
                last = Some(l.narrow(0, b - a - 1, 1)?.squeeze(0)?);
            }
            last.ok_or_else(|| candle::Error::Msg("no segments".into()))
        };
        let ref_whole = ref_run(None)?;

        let mut worst = 1.0f32;
        for cut in [16usize, 32, 33, 64, 128] {
            let split = run(Some(cut))?;
            let c = cos(&whole, &split)?;
            let rc = cos(&ref_whole, &ref_run(Some(cut))?)?;
            worst = worst.min(c);
            println!(
                "  split at {cut:>3}: wave {c:.6} (argmax {} vs {})   reference {rc:.6}",
                whole.argmax(0)?.to_scalar::<u32>()?,
                split.argmax(0)?.to_scalar::<u32>()?
            );
        }
        assert!(
            worst > 0.999,
            "prefilling in two forwards does not equal prefilling in one \
             (worst cosine {worst:.5}) — something that must carry across the \
             split is not carrying"
        );
        Ok(())
    }

    /// Cosine between the reference's and the wave's final logit row at a
    /// range of prefix lengths. Returns the worst.
    pub(in super::super) fn sweep(
        reference: &crate::models::qwen35::model::Qwen35Model,
        model: &HybridBatched,
        device: &Device,
        gate_tokens: &[u32],
        kv_dtype: DType,
        name: &dyn Fn(u32) -> String,
    ) -> Result<f32> {
        let finite = |t: &Tensor| -> Result<bool> {
            let m = t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            Ok(m.is_finite())
        };
        let mut worst = 1.0f32;
        for n in [32usize, 64, 65, 128, 256, 512, gate_tokens.len()] {
            let n = n.min(gate_tokens.len());
            let ids_host = &gate_tokens[..n];

            let mut ref_state = reference.new_session()?;
            let ref_logits = reference.forward(ids_host, &mut ref_state)?;
            let ref_last = ref_logits.narrow(0, n - 1, 1)?.squeeze(0)?;
            assert!(finite(&ref_last)?, "reference non-finite at {n} tokens");
            let ref_argmax = ref_last.argmax(0)?.to_scalar::<u32>()?;

            let mut session =
                model.create_batched_session(BatchedConfig::default().with_dtype(kv_dtype))?;
            let seq = session.create_sequence()?;
            let ids = Tensor::from_vec(ids_host.to_vec(), (1, n), device)?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                &[ids],
                &[],
                &[],
                0,
                model.num_layers(),
                None,
            )?;
            let wave_last = step
                .logits
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("wave returned no logits".into()))?[0]
                .flatten_all()?
                .to_device(&Device::Cpu)?;
            let wave_argmax = wave_last.argmax(0)?.to_scalar::<u32>()?;
            drop(step);
            drop(session);

            let a = ref_last.to_dtype(DType::F32)?;
            let b = wave_last.to_dtype(DType::F32)?;
            let dot = a.mul(&b)?.sum_all()?.to_scalar::<f32>()?;
            let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let c = dot / (na * nb).max(1e-6);
            worst = worst.min(c);
            println!(
                "    {n:>4} tokens: cosine {c:.6}   ref {:?} / wave {:?}{}",
                name(ref_argmax),
                name(wave_argmax),
                if ref_argmax == wave_argmax {
                    ""
                } else {
                    "   ← ARGMAX DIVERGED"
                }
            );
        }
        Ok(worst)
    }
}
