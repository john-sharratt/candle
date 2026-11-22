//! Tests that a greedy decode of one prompt is reproducible, and localises any
//! difference when it is not.
//!
//! `wave_decode_is_reproducible` requires three independent greedy decodes to give
//! identical token streams. `wave_decode_step_bitwise_probe` replays one decode
//! step from bit-identical KV state and compares logits bitwise, then compares
//! residuals by depth, so the first differing depth names the layer that
//! introduces the fault. Prefill and decode are gated separately, so a fix to one
//! is not judged by a test the other fails.

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

    /// **Reproducibility gate.** Three independent greedy decodes of one prompt
    /// must produce identical token streams — greedy argmax from a fresh session
    /// is a pure function of the weights. DeepSeek-V4-Flash currently FAILS this
    /// (see `docs/archived/deepseek/deepseek_decode_reproducibility.md`); the assertion is here so
    /// the day it passes is recorded rather than guessed at.
    ///
    /// Uses the shared `decode_reproducibility` helper, the same one the Qwen3-MoE
    /// and Llama gates call, so the three models' verdicts are comparable: same
    /// harness, same pass count, same machine. That comparison is what localises
    /// the fault — Qwen3-MoE shares this model's `expert_lre` cache and Llama has
    /// no expert cache at all, and both reproduce.
    ///
    /// Token ids go in on the **host**: DeepSeek's embedding lookup is a CPU
    /// `index_select` (invariant 3's sanctioned transfer), unlike the other two
    /// models, which read ids on the device.
    #[test]
    #[ignore]
    fn wave_decode_is_reproducible() -> Result<()> {
        use crate::models::batch_test::utils::decode_reproducibility;

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
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Write one sentence about the \
             sea.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let (agree, streams) =
            decode_reproducibility(&model, &Device::Cpu, &ids, 32, 3, "deepseek")?;
        for (i, s) in streams.iter().enumerate() {
            let text = tokenizer
                .decode(s, false)
                .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
            eprintln!("[repro:deepseek] pass{} = {text:?}", i + 1);
        }
        assert!(
            agree,
            "decode is not reproducible run-to-run; see docs/archived/deepseek/deepseek_decode_reproducibility.md"
        );
        Ok(())
    }

    /// **The localising probe for the reproducibility defect.**
    ///
    /// Comparing token streams (`wave_decode_is_reproducible`) proves the bug
    /// exists but says nothing about where it lives: argmax hides sub-ULP
    /// differences until one crosses a routing boundary, and each verdict costs
    /// a 50 s three-pass run. This replays **one decode step from bit-identical
    /// KV state** instead — prefill once, then repeatedly {decode one token,
    /// compare, roll the KV back} — so every repeat sees exactly the same
    /// inputs, the model is loaded once, and each sample costs one forward.
    ///
    /// Two axes:
    ///  * **repeat** — full-depth logits over `REPEATS` replays. Bitwise, so a
    ///    single differing ULP is caught where argmax would swallow it.
    ///  * **depth** — the same replay truncated to `layer_end = d` (which
    ///    returns the mHC residual rather than logits), swept over `d`. The
    ///    first `d` that differs names the layer that introduces the fault.
    ///
    /// Reports the number of differing elements and max|Δ| at each depth, which
    /// separates the two failure shapes: a *reassociated float sum* differs in
    /// the last few ULPs, whereas a *stale/racing read* differs grossly.
    #[test]
    #[ignore]
    fn wave_decode_step_bitwise_probe() -> Result<()> {
        const REPEATS: usize = 24;

        use crate::models::batch_test::utils::{decode_replay_probe, prefill_replay_probe};

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
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Write one sentence about the \
             sea.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        // Token ids go in on the HOST: DeepSeek's embedding lookup is a CPU
        // `index_select`, unlike the other models, which read ids on the device.
        // Prefill and decode are gated SEPARATELY. End-to-end token streams
        // cannot attribute a divergence to one or the other, so a fix to either
        // gets judged by a test the other also fails — which is exactly how the
        // decode fix came to look ineffective.
        let dirty_prefill = prefill_replay_probe(&model, &Device::Cpu, &ids, 6, "deepseek")?;
        let dirty_repeats = decode_replay_probe(&model, &Device::Cpu, &ids, REPEATS, "deepseek")?;
        assert_eq!(
            dirty_prefill, 0,
            "prefill is not bitwise repeatable: {dirty_prefill}/5 re-prefills of the same \
             prompt diverged; see docs/archived/deepseek/deepseek_decode_reproducibility.md"
        );

        assert_eq!(
            dirty_repeats,
            0,
            "a replayed decode step from identical KV state is not bitwise repeatable: \
             {dirty_repeats}/{} replays diverged; \
             see docs/archived/deepseek/deepseek_decode_reproducibility.md",
            REPEATS - 1
        );
        Ok(())
    }
}
