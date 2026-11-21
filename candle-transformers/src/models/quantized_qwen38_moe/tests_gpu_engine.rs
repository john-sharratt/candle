//! GPU engine tests: the wave's greedy continuation, the draft head's proposals,
//! end-of-turn, QSA at depth, and a NaN bisection.
//!
//! The smoke test requires the oracle's exact greedy continuation rather than a
//! substring, since a first token survives most downstream corruption. The
//! draft-head test checks the first proposal against the trunk's known next tokens,
//! because speculation is lossless and a miswired head would otherwise show only as
//! a missing speedup. The QSA-at-depth test compares a selected read with a dense
//! control on the same prompt, and requires the same answer tokens with the stop
//! tokens honoured. The end-of-turn test needs thinking closed and both end tokens
//! recognised, since the config's `eos_token_id` alone never ends a chat turn.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::batch_test::utils::{TestConfig, TestMode};

    use candle::Tensor;

    use super::super::tests_support::tests::{engine, tokenizer, tokenizer_json};

    /// GPU engine smoke: load this card's engine pack onto the card, run
    /// the probe prompt through `forward_wave` (prefill + greedy decode), and
    /// require the oracle's own continuation. The first run of the wave path
    /// end to end: embed → GR → GDN spans → paged attention → 512-expert MoE
    /// → PLE → head, all through `drive_wave`.
    #[test]
    #[ignore = "loads this card's engine artifact (builds the expert pack on first \
                run) and needs a GPU"]
    fn test_engine_wave_paris_smoke() -> Result<()> {
        use crate::models::batched_inference::BatchedConfig;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::IndexOp;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let t0 = std::time::Instant::now();
        let gpu = Qwen4ExpGpu::load(&pack, &device)?;
        println!("✓ engine loaded in {:.0}s", t0.elapsed().as_secs_f32());
        let model = Qwen4ExpBatched::new(gpu)?;

        let tok = tokenizer()?;
        let prompt = "The capital of France is";
        let ids: Vec<u32> = tok
            .encode(prompt, true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", ids.len());

        use crate::models::batched_inference::ManagedBatchedModel;
        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = ManagedBatchedModel::num_layers(&model);

        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let t1 = std::time::Instant::now();
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
        println!(
            "✓ prefill {} tokens in {:.1}s",
            ids.len(),
            t1.elapsed().as_secs_f32()
        );
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        let mut gen = vec![next];
        let t2 = std::time::Instant::now();
        for _ in 0..11 {
            let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
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
            next = step.logits_owned()?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            gen.push(next);
        }
        let dt = t2.elapsed().as_secs_f32();
        let text = tok
            .decode(&gen, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        println!("generated ids: {gen:?}");
        println!("continuation: {text:?}");
        println!("decode: {:.2} tok/s", 11.0 / dt);
        // The exact greedy continuation, not a substring probe. Decode here is
        // deterministic — same checkpoint, same prompt, argmax at every step —
        // so the token ids are a fixed expected value and anything else is a
        // regression worth failing on.
        //
        // A `contains("Paris")` check passed while this model was answering
        // `" Paris*\",\n    \"What is the capital of France?\","`: the first
        // token survives almost any corruption downstream of it, because it is
        // produced by prefill before the damage lands. Giving the draft head a
        // KV layer put an unwritten layer into the trunk's stream aggregates,
        // which clamped the sequence offset back to 0 on every wave entry
        // (`KvLayers`), and the substring check reported ok throughout.
        const EXPECTED: [u32; 12] = [
            11751, 13, 561, 6511, 314, 9564, 369, 19241, 13, 561, 6511, 314,
        ];
        assert_eq!(
            gen.as_slice(),
            EXPECTED.as_slice(),
            "greedy continuation drifted: {text:?}"
        );
        Ok(())
    }

    /// **The draft head proposes the tokens the trunk actually produces.**
    ///
    /// The bring-up gate for the head, and deliberately upstream of the whole
    /// verify path: speculation is lossless, so a head wired subtly wrong costs
    /// no correctness anywhere — it just proposes tokens nothing accepts, and
    /// the only symptom is a speedup that never arrives. That is invisible in
    /// an end-to-end measurement, which reports "speculation does not pay here"
    /// in exactly the same shape whether the ladder is too deep or the head is
    /// reading its input through the wrong norm.
    ///
    /// So this asks the narrow question directly. After the same prefill
    /// [`test_engine_wave_paris_smoke`] uses, the trunk's continuation is known
    /// exactly — `[13, 561, 6511, 314]` after the committed `11751` — so the
    /// head's proposals can be compared against ground truth rather than
    /// against a threshold.
    ///
    /// The floor is the FIRST proposal, not a count. Proposal `j > 0` is
    /// conditioned on the head's own earlier guesses, so a head that is right
    /// once and then drifts is still working; a head whose very first token is
    /// wrong is reading a different model than the trunk.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_draft_head_proposes_the_trunks_tokens \
                -- --ignored --nocapture --test-threads=1"]
    fn test_draft_head_proposes_the_trunks_tokens() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::IndexOp;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&pack, &device)?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;
        let ids: Vec<u32> = tok
            .encode("The capital of France is", true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

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
        let committed = step.logits_owned()?[0]
            .i(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        // The head must be loaded at all: a budget of 0 here would make every
        // assertion below vacuous.
        assert!(
            model.draft_budget(1) > 0,
            "this artifact reports a zero draft budget at width 1 — it carries no NextN head"
        );
        let proposals = model.speculative_draft(&mut session, &[seq], &[committed], 4)?;
        let got = &proposals[0];
        // What the trunk really says next, from the smoke's pinned continuation.
        const TRUTH: [u32; 4] = [13, 561, 6511, 314];
        let matched = got
            .iter()
            .zip(TRUTH.iter())
            .take_while(|(a, b)| a == b)
            .count();
        println!("committed: {committed}");
        println!("drafted:   {got:?}");
        println!("truth:     {TRUTH:?}");
        println!("accepted prefix: {matched} of {}", got.len());

        assert!(
            !got.is_empty(),
            "the head proposed nothing — it had no seed, which after a prefill means \
             `head_wave_pass` never ran"
        );
        assert_eq!(
            got[0], TRUTH[0],
            "the head's FIRST proposal disagrees with the trunk: drafted {got:?} against \
             {TRUTH:?}. Every later position is conditioned on the head's own guesses, so \
             this one is the wiring check — a mismatch here means the head's input \
             assembly or output collapse is reading the model differently than the trunk."
        );

        // The head's KV must be exactly where it was: a walk leaves no trace,
        // and a head layer left longer than the trunk's is a skew the next wave
        // would "heal" by truncating a token the caller was already given.
        let lens: Vec<usize> = session
            .sequence_caches(seq)
            .expect("live slot")
            .caches
            .iter()
            .map(|c| {
                let mut n = 0usize;
                c.k_cache().chunked_visit_live_chunks(|it| {
                    for ch in it {
                        n += ch.token_count as usize;
                    }
                });
                n
            })
            .collect();
        println!("layer token counts after drafting: {lens:?}");
        assert!(
            lens.windows(2).all(|w| w[0] == w[1]),
            "the draft walk left the layers uneven: {lens:?}"
        );
        Ok(())
    }

    /// **The model stops on its own.** A chat turn that asks for a one-word
    /// answer, decoded until the model emits an end token or a cap is reached.
    ///
    /// [`test_engine_wave_paris_smoke`] cannot answer this and never could: it
    /// prompts with the bare completion prefix `"The capital of France is"` and
    /// runs a fixed 11 iterations. Nothing there asks the model to finish, so
    /// continuing into "The capital of Germany is Berlin" is the *correct*
    /// response to that input — the loop simply stops counting. Termination is
    /// a different property and needs a prompt that has an end.
    ///
    /// Three things have to line up for this to pass, and each fails
    /// differently:
    ///
    /// 1. **The chat template is right.** Wrong turn markers and the model
    ///    continues the transcript — writing the user's next turn instead of
    ///    ending its own.
    /// 2. **Thinking is closed.** This checkpoint carries `<think>` / `</think>`
    ///    (248068/248069). Qwen3's non-thinking form pre-fills the assistant
    ///    turn with an already-closed block; without it the model reasons for
    ///    hundreds of tokens and a cap this size looks exactly like a failure
    ///    to stop.
    /// 3. **The end token is reachable at all.** `<|im_end|>` is 248046 —
    ///    *not* the 248044 the config names as `eos_token_id`, which is
    ///    `<|endoftext|>` and ends a document rather than a turn. A stop list
    ///    built from `eos_token_id` alone never fires on a chat turn, which is
    ///    why the conversation layer resolves both by name.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_engine_stops_on_end_of_turn \
                -- --ignored --nocapture --test-threads=1"]
    fn test_engine_stops_on_end_of_turn() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::IndexOp;

        /// `<|im_end|>` — the end of an assistant TURN.
        const IM_END: u32 = 248_046;
        /// `<|endoftext|>` — the end of a DOCUMENT, and what the config calls
        /// `eos_token_id`. Accepted as a stop here because either one means the
        /// model chose to finish.
        const ENDOFTEXT: u32 = 248_044;
        /// Generous enough that a model which genuinely wants to keep talking
        /// is not mistaken for one that stopped.
        const MAX_NEW: usize = 48;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&pack, &device)?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;

        // Qwen3's chat form with thinking disabled: the assistant turn opens
        // with an already-closed think block, so the model answers directly.
        let prompt = "<|im_start|>user\nWhat is the capital of France? \
                      Answer with a single word.<|im_end|>\n\
                      <|im_start|>assistant\n<think>\n\n</think>\n\n";
        // `false`: the template above supplies its own special tokens, so the
        // post-processor must not add another set around them.
        let ids: Vec<u32> = tok
            .encode(prompt, false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", ids.len());

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
        let mut next = step.logits_owned()?[0]
            .i(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        let mut gen: Vec<u32> = Vec::new();
        let mut stopped_at: Option<u32> = None;
        for _ in 0..MAX_NEW {
            if next == IM_END || next == ENDOFTEXT {
                stopped_at = Some(next);
                break;
            }
            gen.push(next);
            let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
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
            next = step.logits_owned()?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
        }
        let answer = tok
            .decode(&gen, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        println!("answer ids: {gen:?}");
        println!("answer: {answer:?}");
        println!(
            "stopped on: {}",
            match stopped_at {
                Some(IM_END) => "<|im_end|>".to_string(),
                Some(ENDOFTEXT) => "<|endoftext|>".to_string(),
                Some(other) => format!("token {other}"),
                None => format!("NOTHING — still generating after {MAX_NEW} tokens"),
            }
        );

        let stop = stopped_at.ok_or_else(|| {
            candle::Error::Msg(format!(
                "the model never ended its turn: {MAX_NEW} tokens generated with no \
                 <|im_end|> ({IM_END}) or <|endoftext|> ({ENDOFTEXT}) — got {answer:?}"
            ))
        })?;
        assert!(
            stop == IM_END || stop == ENDOFTEXT,
            "stopped on an unexpected token {stop}"
        );
        assert!(
            answer.contains("Paris"),
            "the model ended its turn but answered {answer:?} rather than Paris — \
             termination is right, the answer is not"
        );
        // A one-word instruction, honoured. Loose enough not to police
        // punctuation or a leading space, tight enough that a model which
        // ended its turn only after a paragraph still fails.
        assert!(
            gen.len() <= 4,
            "asked for a single word and got {} tokens: {answer:?}",
            gen.len()
        );
        Ok(())
    }

    /// **QSA at depth on the engine.** A prompt past the 2051-position
    /// selection width, so the 12 full-attention layers genuinely select
    /// instead of degenerating to dense — the regime the engine could not
    /// reach before the indexer was built.
    ///
    /// What this asserts, and why each part is here:
    ///
    /// 1. **The indexer engaged.** `qsa_rows_selected` counts rows a
    ///    selection was built for; zero would mean the whole run took the
    ///    identity path and the rest of the test proved nothing.
    /// 2. **Narrowing is free.** The same engine runs the same prompt with the
    ///    budget widened past the depth, which makes `selection_engages` false
    ///    and the read dense. The selected run must produce the **same tokens**
    ///    — not merely an answer that still mentions the needle. This is the
    ///    claim QSA makes, and it is the one a subtly wrong selection (an
    ///    off-by-one in the block arithmetic, a mis-indexed row) fails.
    /// 3. **The read reached the needle at all.** It sits in the FIRST
    ///    sentence, thousands of tokens behind the query, and its distinctive
    ///    prefix appears in both answers. Without this, two identically broken
    ///    reads would agree and pass (2).
    /// 4. **It is deterministic**, run to run, which a race in the streaming
    ///    top-k or in the per-sequence cache would break.
    /// 5. **Decode carries it.** The continuation is generated one token at a
    ///    time, so the decode path's selection (one row per slot, built
    ///    against a cache the prefill filled) is exercised too.
    ///
    /// **The dense control is load-bearing, and its absence made this test
    /// wrong.** Asserting the needle alone conflates two failures: a selection
    /// that dropped the block, and a prompt the checkpoint does not answer. It
    /// was the second — the prompt was a bare continuation ("Answer: the label
    /// was"), which an instruct model answers by switching into assistant mode
    /// rather than finishing the sentence, and the dense read failed it exactly
    /// as the selected one did.
    ///
    /// The selection ITSELF is pinned against the CPU oracle's
    /// `qsa_selection_mask` in `qwen4exp::indexer` and the attention's
    /// honouring of it in `tests/qsa_kernel_tests.rs`, both without a model in
    /// the way — this is the end-to-end statement those two make possible.
    #[test]
    #[ignore = "loads this card's engine artifact and prefills >2100 tokens. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_engine_qsa_at_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn test_engine_qsa_at_depth() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::IndexOp;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let t0 = std::time::Instant::now();
        let mut gpu = Qwen4ExpGpu::load(&pack, &device)?;
        let released_top_k = gpu.cfg.indexer.top_k;
        let width = released_top_k + 4 - 1;
        println!(
            "✓ engine loaded in {:.0}s (selection width {width})",
            t0.elapsed().as_secs_f32()
        );
        // **The DENSE control, on the same model and the same prompt.** A needle
        // that the selected read loses says nothing on its own: the continuation
        // could be lost to the selection, or the checkpoint could simply not
        // answer this prompt. Widening the budget past the prompt makes
        // `selection_engages` false — the same code path, reading every cell —
        // so the two runs differ in exactly one thing.
        let dense_top_k = 1 << 20;
        gpu.cfg.indexer.top_k = dense_top_k;
        let mut model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;

        // The needle first, then filler until the query is far past the
        // budget: everything between is uniform, so a dense read and a
        // selected read see very different things.
        let mut text = String::from(
            "The expedition's logbook opens with a single line: the sample recovered \
             from the northern ridge was labelled AMBERGRIS-7.\n\n",
        );
        let filler =
            "The team recorded the weather each morning and stowed the equipment each night. ";
        while tok
            .encode(text.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .len()
            < 2400
        {
            text.push_str(filler);
        }
        // **In the shape the checkpoint was trained on**, with thinking
        // suppressed — the same framing `test_engine_stops_on_end_of_turn`
        // uses. Asked as a bare continuation ("Answer: the label was"), this
        // is an instruct model handed something that is not a turn: it
        // recovers the first token of the needle and then switches into
        // assistant mode, emitting `\n\n<think>` instead of finishing the
        // word. Measured — the DENSE read did it too, so the shape was
        // failing the test, not the selection. The empty `<think></think>`
        // keeps the answer inside the seven decode steps this measures.
        let text = format!(
            "<|im_start|>user\n{text}\n\nWhat label was written on the sample from the \
             northern ridge? Answer with the label alone.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let ids: Vec<u32> = tok
            .encode(text.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        assert!(
            ids.len() > width,
            "prompt of {} tokens is inside the {width}-cell budget — selection would be \
             the identity and this test would assert nothing",
            ids.len()
        );
        println!("prompt: {} tokens", ids.len());

        let n_layers = ManagedBatchedModel::num_layers(&model);
        let run = |model: &Qwen4ExpBatched| -> Result<(String, u64)> {
            let before = model.qsa_rows_selected();
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let seq = session.create_sequence()?;
            let prompt = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let t = std::time::Instant::now();
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, ids.len())?;
            println!("  prefill {:.1}s", t.elapsed().as_secs_f32());
            let logits = step.logits_owned()?;
            let m = logits[0].abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(m.is_finite(), "non-finite logits under QSA selection");
            // **The turn's stop tokens, and the decode honours them.**
            // `<|im_end|>` ends an assistant turn and `<|endoftext|>` ends a
            // document; past either, the model is continuing something it
            // already finished and what it emits is not a claim about
            // anything. Decoding through it made this test compare that
            // garbage: two runs answered `AMBER` identically and then filled
            // the remaining steps differently — one with `<|endoftext|>` then
            // `<|im_start|>` repeats, the other with `<|im_end|>` then an
            // invented user turn — and the determinism assertion read that as
            // a race in the selection.
            const IM_END: u32 = 248_046;
            const ENDOFTEXT: u32 = 248_044;
            let stop = |t: u32| t == IM_END || t == ENDOFTEXT;

            let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            let mut gen = vec![next];
            // Decode is timed on its own: with the read capped at 2051 cells
            // this rate is the STEADY STATE for any depth, which is what makes
            // it the denominator §6.2's kernel question needs.
            let t_dec = std::time::Instant::now();
            const DECODE_STEPS: usize = 7;
            for _ in 0..DECODE_STEPS {
                if stop(next) {
                    break;
                }
                let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
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
                next = step.logits_owned()?[0]
                    .i(0)?
                    .argmax(0)?
                    .to_scalar::<u32>()?;
                gen.push(next);
            }
            let dec = t_dec.elapsed().as_secs_f64();
            println!(
                "  decode {DECODE_STEPS} steps in {:.2}s = {:.1} tok/s ({:.1} ms/step)",
                dec,
                DECODE_STEPS as f64 / dec,
                1000.0 * dec / DECODE_STEPS as f64
            );
            model.release_sequence(seq)?;
            // **The answer, without the marker that ends it.** `<|im_end|>` and
            // `<|endoftext|>` both mean the model chose to finish, and which of
            // the two wins is a coin-flip between near-tied logits that a
            // selection reading 2,051 of 2,444 cells is entitled to move.
            // Comparing it would be asserting bit-identical logits, which
            // narrowing does not promise and does not need to: what has to
            // survive the narrowing is the answer.
            let body: Vec<u32> = gen.iter().copied().take_while(|&t| !stop(t)).collect();
            let out = tok
                .decode(&body, false)
                .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
            Ok((out, model.qsa_rows_selected() - before))
        };

        // ── The dense control first ──────────────────────────────────────────
        // Same engine, same prompt, budget above the depth so the selection is
        // the identity. What this run recovers is the ceiling; what the selected
        // run recovers is measured against it, and only the DIFFERENCE is QSA's.
        let (dense, dense_rows) = run(&model)?;
        println!("dense control: {dense:?}  (QSA rows selected: {dense_rows})");
        assert_eq!(
            dense_rows, 0,
            "the dense control narrowed {dense_rows} row(s) — a budget of {dense_top_k} was \
             supposed to put every cell inside it, so this is not a control at all"
        );

        model.set_selection_budget(released_top_k)?;
        let (first, rows) = run(&model)?;
        println!("continuation: {first:?}  (QSA rows selected: {rows})");
        // §6.2: what the decode kernel's share actually is at a QSA-capped
        // read, on the geometry the stale kernel comment calls exotic (hpg 12
        // at head_dim 256, which runs single-stage). Measure before touching
        // the kernel — with `--features profile`, otherwise this is empty.
        {
            use crate::models::profile::{gpu_drain_blocking, pipeline_snapshot_and_reset};
            gpu_drain_blocking();
            let mut snap = pipeline_snapshot_and_reset();
            if !snap.entries.is_empty() {
                snap.entries
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let total: f64 = snap.entries.iter().map(|e| e.1).sum();
                println!("\n=== QSA-depth profile (total {total:.1} ms) ===");
                for (name, ms, calls) in snap.entries.iter().take(18) {
                    println!(
                        "  {name:<28} {ms:>9.1} ms  {:>5.1}%  ×{calls}",
                        100.0 * ms / total
                    );
                }
            }
        }
        assert!(
            rows > 0,
            "no row was ever narrowed — the indexer never engaged, so nothing here \
             tested the selection"
        );
        // **The needle's distinctive prefix, not the whole label.** Greedy
        // decode answers this prompt with the single token `AMBER` and then
        // ends the turn — the DENSE read does exactly the same, so the rest of
        // `AMBERGRIS-7` is this checkpoint's terseness and not something a read
        // can recover. `AMBER` still only comes from the needle: nothing in the
        // filler resembles it, so this remains a retrieval claim rather than a
        // formatting one, which is what the "plausible alternative" the doc
        // above worries about would fail.
        const NEEDLE: &str = "AMBER";
        assert!(
            dense.to_uppercase().contains(NEEDLE),
            "the DENSE read lost the needle: {dense:?} — nothing below is about the \
             selection, because reading every cell did not answer this prompt"
        );
        assert!(
            first.to_uppercase().contains(NEEDLE),
            "the needle did not survive the SELECTED read: {first:?} — the dense control \
             recovered it ({dense:?}), so the selection dropped the block that held it"
        );
        // **The claim QSA actually makes, and the strongest one available
        // here.** Reading 2,051 of 2,444 cells produced the same answer as
        // reading all of them — not merely an answer that still mentions the
        // needle, but the same tokens. A selection that dropped the needle's
        // block, or kept it and reordered the rest, fails here while a
        // `contains` check would pass.
        assert_eq!(
            first, dense,
            "the selected read diverged from the dense one: {first:?} against {dense:?} — \
             {rows} row(s) were narrowed, and narrowing is supposed to leave the answer alone"
        );
        let (second, _) = run(&model)?;
        assert_eq!(first, second, "selection at depth is nondeterministic");
        println!("✓ QSA engaged, dense-identical, needle recovered, deterministic");
        Ok(())
    }

    /// Layer-range NaN bisect over the GPU wave (§7.3 method): run the prompt
    /// through `[0, k)` for growing `k` and report the residual's max-abs —
    /// the first non-finite `k` names the layer, and the schedule names the
    /// subsystem (3:1 GDN/attention, PLE at layer 1, MoE everywhere).
    #[test]
    #[ignore = "loads this card's engine artifact; diagnostic"]
    fn test_engine_wave_nan_bisect() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&pack, &device)?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;
        let ids: Vec<u32> = tok
            .encode("The capital of France is", true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let n_layers = ManagedBatchedModel::num_layers(&model);
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;

        for k in [0usize, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 40, 47, 48] {
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let seq = session.create_sequence()?;
            let mut residual = model.claim_residual(&session, ids.len())?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt_t),
                &[],
                &[],
                0,
                k.min(n_layers),
                Some(&mut residual),
            )?;
            if k < n_layers {
                let r = residual.committed().to_tensor()?;
                let m = r
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_scalar::<f32>()?;
                println!("layers [0,{k:>2}): residual max|x| = {m:e}");
            } else {
                let lg = step.logits_owned()?;
                let m = lg[0]
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_scalar::<f32>()?;
                println!("layers [0,{k:>2}) + head: logits max|x| = {m:e}");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_parallel_batched_forwarding \
                -- --ignored --nocapture --test-threads=1"]
    fn test_parallel_batched_forwarding() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        println!("\n=== Qwen3.8-Flash-Next (qwen4exp) batched forwarding ===\n");
        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);

        // 256 tokens, not 64: a wide prefill concedes the expert zone (to 11 GB
        // at ×8 and 5 GB at ×16 on a 24 GB card), and the decode behind it
        // rebuilds its working set over its first steps. At 64 tokens — 13
        // speculative steps — that recovery was a third of the decode, so the
        // rate measured the pivot rather than the decode. C10, the calibration
        // probe, stays at the 64 its threshold row was derived against.
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_top_rung_tokens(64)
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600)
            // Each config's routing, for replaying residency and prediction
            // policies offline (`expert_lre::replay`).
            .with_routing_trace_dir(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../target/routing_traces/qwen38_flash_next"),
            );

        let configs = batched_forward_configs(&device);

        let load = || {
            let gpu = Qwen4ExpGpu::load(&pack, &device)?;
            // The checkpoint must still be the geometry this engine was
            // validated for — a silent architecture change fails here, not in
            // a kernel.
            assert_eq!(gpu.cfg.num_layers, 48);
            assert_eq!(gpu.cfg.attn_head_dim, 256);
            assert_eq!(gpu.cfg.moe.n_experts, 512);
            assert_eq!(gpu.cfg.moe.n_experts_used, 10);
            let m = Qwen4ExpBatched::new(gpu)?;
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

    /// **Eight sequences whose prefill does not fit one wave.** The gate's ×8
    /// row alone: on a 16 GiB card its 4,160 prompt rows are sliced into
    /// several passes, so every sequence's prefill is a chain of long spans at
    /// rising offsets — each reading the state the last one installed through
    /// the prefill scratch's exchange — and decode starts from the end of that
    /// chain. On a wider card the row runs whole and checks the same output.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_eight_wide_sliced_prefill \
                -- --ignored --nocapture --test-threads=1"]
    fn test_eight_wide_sliced_prefill() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600);
        let configs = vec![TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 8,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        }];
        params.run(configs, || {
            let gpu = Qwen4ExpGpu::load(&pack, &device)?;
            Qwen4ExpBatched::new(gpu)
        })
    }
}
