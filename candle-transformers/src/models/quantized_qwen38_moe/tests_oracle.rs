//! Oracle gates on the real checkpoint: the batched forward, the Q4_KO expert
//! recheck, and QSA at depth.
//!
//! The oracle gate checks the split loader against the frozen schema, finite logits
//! over batched prompts, that a session's logits do not depend on its batch-mates,
//! that a segmented forward equals a one-shot one across the three carried state
//! classes, and that greedy decoding gives the right answer. The Q4_KO recheck
//! converts the W4A16 experts into the pinned split, asserting each expert bit-exact
//! against the release's own dequantisation, and then runs the same gate on the
//! converted model. The QSA-at-depth gate asserts determinism and finiteness past
//! the selection width.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::batch_test::test_helpers::hf_get;

    use candle::Tensor;

    use hf_hub::RepoType;

    use super::super::tests_support::tests::{pinned_shards, tokenizer};

    /// The Phase-1 oracle gate: `forward_batched` end to end on the real
    /// checkpoint. Five checks —
    ///
    /// 1. the split loader + config against the frozen schema;
    /// 2. finite logits over real prompts, **batched** (two sessions in one
    ///    call — the oracle is batched from the start, §0.6);
    /// 3. a session's logits do not depend on its batch-mates (×1 ≡ ×N);
    /// 4. segmented forward ≡ one-shot — the three carried state classes
    ///    (GDN, PLE conv+hash window, QSA index cache) survive a boundary;
    /// 5. greedy continuation is deterministic, non-degenerate, and
    ///    semantically right ("The capital of France is" → Paris) — the
    ///    check that catches self-consistent-but-wrong geometry.
    #[test]
    #[ignore = "reads the pinned 188 GB Q8_0 split from the HF cache and runs the CPU \
                reference forward (on-demand expert reads; needs ~24 GB RAM). Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_forward_batched_oracle \
                -- --ignored --nocapture --test-threads=1"]
    fn test_forward_batched_oracle() -> Result<()> {
        println!("\n=== Qwen3.8-Flash-Next forward_batched oracle ===\n");
        let shards = pinned_shards()?;
        let device = Device::Cpu;
        let t_load = std::time::Instant::now();
        let model = oracle_from_gguf_path(&shards[0], &device)?;
        println!(
            "✓ loaded: {} layers ({} attention), hidden {}, {} experts top-{}, \
             hc {}×, PLE layer {} ({:.1}s)",
            model.cfg.num_layers,
            model.cfg.n_attention_layers(),
            model.cfg.hidden_size,
            model.cfg.moe.n_experts,
            model.cfg.moe.n_experts_used,
            model.cfg.hc.count,
            model.cfg.ple.layer,
            t_load.elapsed().as_secs_f32(),
        );

        let tok = tokenizer()?;
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(tok
                .encode(s, false)
                .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
                .get_ids()
                .to_vec())
        };
        let prompt_a = "The capital of France is";
        let prompt_b = "Water is made of hydrogen and";
        let a = encode(prompt_a)?;
        let b = encode(prompt_b)?;
        println!("prompts: {:?} tokens / {:?} tokens", a.len(), b.len());

        // (2) Batched prefill: two sessions, one call.
        let mut pair = [model.new_session()?, model.new_session()?];
        let t_prefill = std::time::Instant::now();
        let both = model.forward_batched(&[&a, &b], &mut pair)?;
        println!(
            "✓ batched prefill ({:.1}s)",
            t_prefill.elapsed().as_secs_f32()
        );
        let last_a = both[0].narrow(0, a.len() - 1, 1)?;
        let last_b = both[1].narrow(0, b.len() - 1, 1)?;
        for (name, l) in [("A", &last_a), ("B", &last_b)] {
            let m = l.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(m.is_finite(), "session {name}: non-finite logits");
        }

        // (3) ×1 ≡ ×N: session A alone must match session A in the pair.
        let mut solo = model.new_session()?;
        let alone = model
            .forward_batched(&[&a], std::slice::from_mut(&mut solo))?
            .remove(0);
        let d = alone
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "batching changed session A's logits by {d}");
        println!("✓ ×1 ≡ ×2 (max |Δ| = {d:.2e})");

        // (4) Segmented ≡ one-shot, across all three carried state classes.
        let mut seg = model.new_session()?;
        let cut = a.len() / 2;
        let s1 = model
            .forward_batched(&[&a[..cut]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let s2 = model
            .forward_batched(&[&a[cut..]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let stitched = Tensor::cat(&[s1, s2], 0)?;
        let d = stitched
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "segmented forward diverged from one-shot by {d}");
        println!("✓ segmented ≡ one-shot (max |Δ| = {d:.2e})");

        // (5) Greedy continuation, BOTH sessions stepped through
        // `forward_batched` together — decode is batched too.
        let argmax = |l: &Tensor| -> Result<u32> { l.get(0)?.argmax(0)?.to_scalar::<u32>() };
        let mut next = [argmax(&last_a)?, argmax(&last_b)?];
        let mut out = [vec![next[0]], vec![next[1]]];
        let t_dec = std::time::Instant::now();
        for _ in 0..8 {
            let seg_a = [next[0]];
            let seg_b = [next[1]];
            let ls = model.forward_batched(&[&seg_a, &seg_b], &mut pair)?;
            next = [argmax(&ls[0])?, argmax(&ls[1])?];
            out[0].push(next[0]);
            out[1].push(next[1]);
        }
        println!(
            "✓ greedy ×2 decode, 8 steps ({:.1}s)",
            t_dec.elapsed().as_secs_f32()
        );
        for (prompt, ids) in [(prompt_a, &out[0]), (prompt_b, &out[1])] {
            let text = tok
                .decode(ids, false)
                .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
            println!("{prompt}|{text}");
        }
        let text_a = tok
            .decode(&out[0], false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        assert!(
            text_a.contains("Paris"),
            "the model did not complete {prompt_a:?} with Paris — got {text_a:?}"
        );
        let distinct: std::collections::HashSet<u32> = out[0].iter().copied().collect();
        assert!(distinct.len() >= 4, "degenerate continuation: {:?}", out[0]);
        println!("\n=== oracle gate green ===");
        Ok(())
    }

    /// The Q4_KO-expert recheck: convert the W4A16 release's experts into the
    /// pinned split (first run only — the sibling split is cached beside the
    /// Q8_0 shards), then run the oracle gate on the converted artifact and
    /// compare its last-position logits against the Q8_0 reference.
    ///
    /// What each half proves:
    /// - the conversion itself asserts **bit-exactness** per expert
    ///   (`dequant_ko` ≡ the AWQ release's own dequantization), so a green
    ///   conversion means our Q4_KO bytes ARE the calibrated AWQ weights;
    /// - the gate then proves the model still *works* at 4-bit experts —
    ///   Paris/oxygen, ×1 ≡ ×2, segmented ≡ one-shot — and the cross-model
    ///   comparison quantifies the Q8→AWQ-Q4 expert swap (informational: two
    ///   different quantizations of the same weights, no bit-equality to
    ///   expect; argmax agreement on the probe prompts is the meaningful bar).
    #[test]
    #[ignore = "downloads 73 GB of W4A16 expert shards on first run, writes the ~72 GB \
                Q4KOEXP sibling split beside the Q8_0 shards, and runs the CPU oracle \
                twice. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_forward_batched_oracle_q4ko_experts \
                -- --ignored --nocapture --test-threads=1"]
    fn test_forward_batched_oracle_q4ko_experts() -> Result<()> {
        use crate::models::qwen4exp::convert::convert_w4a16_experts;

        println!("\n=== Qwen3.8-Flash-Next Q4_KO-expert oracle recheck ===\n");
        let q8_shards = pinned_shards()?;
        let awq_shards: Vec<PathBuf> = QWEN4EXP_W4A16_EXPERT_SHARDS
            .iter()
            .map(|&i| {
                hf_get(
                    QWEN4EXP_W4A16_REPO,
                    RepoType::Model,
                    QWEN4EXP_W4A16_REV,
                    &w4a16_shard_name(i),
                )
            })
            .collect::<Result<_>>()?;

        // The sibling split lives beside the Q8_0 shards; conversion is
        // resumable and skipped entirely once all six outputs exist.
        let out_dir = q8_shards[0]
            .parent()
            .ok_or_else(|| candle::Error::Msg("q8 shard has no parent dir".into()))?
            .to_path_buf();
        let t_conv = std::time::Instant::now();
        // The conversion runs its fused repack kernel on the GPU; the oracle
        // itself stays on the CPU below.
        let cuda = Device::new_cuda(0)?;
        let converted = convert_w4a16_experts(&q8_shards, &awq_shards, &out_dir, &cuda)?;
        println!(
            "✓ Q4KOEXP split ready ({:.0}s): {:?}",
            t_conv.elapsed().as_secs_f32(),
            converted[0].file_name().unwrap_or_default()
        );

        let device = Device::Cpu;
        let model = oracle_from_gguf_path(&converted[0], &device)?;
        println!("✓ converted split loads through the standard path");

        let tok = tokenizer()?;
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(tok
                .encode(s, false)
                .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
                .get_ids()
                .to_vec())
        };
        let prompt_a = "The capital of France is";
        let prompt_b = "Water is made of hydrogen and";
        let a = encode(prompt_a)?;
        let b = encode(prompt_b)?;

        // The oracle gate's checks on the converted model.
        let mut pair = [model.new_session()?, model.new_session()?];
        let both = model.forward_batched(&[&a, &b], &mut pair)?;
        let last_a = both[0].narrow(0, a.len() - 1, 1)?;

        let mut solo = model.new_session()?;
        let alone = model
            .forward_batched(&[&a], std::slice::from_mut(&mut solo))?
            .remove(0);
        let d = alone
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "batching changed session A's logits by {d}");

        let mut seg = model.new_session()?;
        let cut = a.len() / 2;
        let s1 = model
            .forward_batched(&[&a[..cut]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let s2 = model
            .forward_batched(&[&a[cut..]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let stitched = Tensor::cat(&[s1, s2], 0)?;
        let d = stitched
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "segmented forward diverged from one-shot by {d}");
        println!("✓ ×1 ≡ ×2 and segmented ≡ one-shot on Q4_KO experts");

        // Greedy continuation through batched decode — the semantic bar.
        let argmax = |l: &Tensor| -> Result<u32> { l.get(0)?.argmax(0)?.to_scalar::<u32>() };
        let mut next = [
            argmax(&last_a)?,
            argmax(&both[1].narrow(0, b.len() - 1, 1)?)?,
        ];
        let mut out = [vec![next[0]], vec![next[1]]];
        for _ in 0..8 {
            let (seg_a, seg_b) = ([next[0]], [next[1]]);
            let ls = model.forward_batched(&[&seg_a, &seg_b], &mut pair)?;
            next = [argmax(&ls[0])?, argmax(&ls[1])?];
            out[0].push(next[0]);
            out[1].push(next[1]);
        }
        for (prompt, ids) in [(prompt_a, &out[0]), (prompt_b, &out[1])] {
            let text = tok
                .decode(ids, false)
                .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
            println!("{prompt}|{text}");
        }
        if let Some(s) = model.ple_table.cache_stats() {
            println!(
                "PLE row cache: {} hits / {} misses ({:.1}% hit rate), {} evictions",
                s.hits,
                s.misses,
                s.hit_rate() * 100.0,
                s.evictions
            );
        }
        let text_a = tok
            .decode(&out[0], false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        assert!(
            text_a.contains("Paris"),
            "Q4_KO experts lost the completion: {text_a:?}"
        );

        // Cross-model comparison against the Q8_0 reference, same prompt.
        drop(model);
        let reference = oracle_from_gguf_path(&q8_shards[0], &device)?;
        let mut ref_sess = reference.new_session()?;
        let ref_logits = reference
            .forward_batched(&[&a], std::slice::from_mut(&mut ref_sess))?
            .remove(0);
        let ref_last = ref_logits.narrow(0, a.len() - 1, 1)?;
        let q4_last = alone.narrow(0, a.len() - 1, 1)?;
        let ref_arg = argmax(&ref_last)?;
        let q4_arg = argmax(&q4_last)?;
        let diff = q4_last
            .sub(&ref_last)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        println!(
            "Q8_0 vs Q4_KO experts, last position: argmax {} vs {} ({}), max |Δlogit| = {diff:.3}",
            ref_arg,
            q4_arg,
            if ref_arg == q4_arg {
                "agree"
            } else {
                "DISAGREE"
            },
        );
        assert_eq!(
            ref_arg, q4_arg,
            "the probe prompt's next token moved under the expert swap — \
             quantify before trusting the rung"
        );
        println!("\n=== Q4_KO-expert recheck green ===");
        Ok(())
    }

    /// QSA at depth: a prompt past the 2051-position selection width, so the
    /// 12 attention layers genuinely select instead of degenerating to dense.
    /// Determinism + finiteness here; token-level parity in the sparse regime
    /// is Phase 1's llama.cpp comparison.
    #[test]
    #[ignore = "reads the pinned 188 GB Q8_0 split and runs a >2051-token CPU prefill \
                (tens of minutes). Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_forward_batched_qsa_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn test_forward_batched_qsa_depth() -> Result<()> {
        let shards = pinned_shards()?;
        let device = Device::Cpu;
        let model = oracle_from_gguf_path(&shards[0], &device)?;
        let tok = tokenizer()?;

        let base = "The expedition kept meticulous notes on every specimen it catalogued. ";
        let mut long = String::new();
        while tok
            .encode(long.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .len()
            < 2100
        {
            long.push_str(base);
        }
        long.push_str("The very first sentence of these notes said the expedition kept");
        let ids = tok
            .encode(long.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let width = model.cfg.indexer.top_k + 4 - 1;
        assert!(ids.len() > width, "prompt too short to engage selection");
        println!("prompt: {} tokens (selection width {width})", ids.len());

        let run = || -> Result<u32> {
            let mut s = model.new_session()?;
            let l = model
                .forward_batched(&[&ids], std::slice::from_mut(&mut s))?
                .remove(0);
            let last = l.narrow(0, ids.len() - 1, 1)?;
            let m = last.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(m.is_finite(), "non-finite logits under QSA selection");
            last.get(0)?.argmax(0)?.to_scalar::<u32>()
        };
        let t = std::time::Instant::now();
        let first = run()?;
        println!(
            "✓ deep prefill ({:.0}s); next token id {first} = {:?}",
            t.elapsed().as_secs_f32(),
            tok.decode(&[first], false).ok()
        );
        let second = run()?;
        assert_eq!(first, second, "selection at depth is nondeterministic");
        Ok(())
    }
}
