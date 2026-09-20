//! The progressive-YaRN gates, run on the models whose rungs they back
//! (`docs/progressive_yarn.md` §10, gates 3–5).
//!
//! One `#[ignore]`d test per model, each loading its checkpoint once and
//! swapping the RoPE schedule underneath it for the control — nothing stored
//! depends on a rung (I1), so the same weights answer under both schedules and
//! the two reads differ in the schedule alone.
//!
//! The gate: past the trained window, the progressive schedule recovers the
//! needles and gives them a lower NLL than rung-1 extrapolation, by more than
//! the model's own run-to-run spread; inside it, the two agree within that
//! spread.
//!
//! **The spread is measured, not assumed.** Each test repeats one progressive
//! read and takes the difference between the two as its noise floor. On a
//! deterministic model it is zero and the comparisons are exact; Flash-Next is
//! not run-to-run deterministic (two reads of one prompt under one schedule
//! differed by 8e-5 nats at 128K), and a difference below that is not one the
//! schedule made.
//!
//! ```text
//! cargo test --release --features cuda -p candle-transformers --lib \
//!     batch_test::yarn_gate_models::<test> -- --ignored --nocapture --test-threads=1
//! ```

use candle::quantized::Int8Mode;
use candle::{Device, Result};
use hf_hub::RepoType;
use tokenizers::Tokenizer;

use super::test_helpers::hf_get;
use super::yarn_gates::{
    answer_of, crossing_read, greedy_together, needle_prompt, needle_read, report, Read, NEEDLE_KEY,
};
use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
use crate::models::batched_model::{ensure_vram_governor, BatchedInference};
use crate::models::rope_schedule::{DeclaredScaling, RopePreset, RopeSchedule};

fn tokenizer(repo: &str, rev: &str) -> Result<Tokenizer> {
    let p = hf_get(repo, RepoType::Model, rev, "tokenizer.json")?;
    Tokenizer::from_file(&p).map_err(|e| candle::Error::Msg(format!("tokenizer: {e}")))
}

/// Gate 3 over `depths` for one loaded model: every depth's read, in order.
fn reads<M: ManagedBatchedModel>(
    model: &M,
    tok: &Tokenizer,
    label: &str,
    depths: &[usize],
) -> Result<Vec<(Read, u32)>> {
    depths
        .iter()
        .map(|&d| {
            let (read, rung, n) = needle_read(model, BatchedConfig::default(), tok, d)?;
            report(label, d, n, rung, &read, tok);
            Ok((read, rung))
        })
        .collect()
}

/// The model's run-to-run spread: the same progressive read at `depth` again,
/// against `first`. Zero on a deterministic model.
fn noise_floor<M: ManagedBatchedModel>(
    model: &M,
    tok: &Tokenizer,
    depth: usize,
    first: &Read,
) -> Result<f64> {
    let (again, rung, n) = needle_read(model, BatchedConfig::default(), tok, depth)?;
    report("progressive, again", depth, n, rung, &again, tok);
    let noise = (again.nll - first.nll).abs();
    println!("  run-to-run spread at depth {depth}: {noise:.4e} nats");
    Ok(noise)
}

/// Judge gate 3: inside the window (rung 0 under both) the reads agree within
/// `noise`; past it progressive recovers the needles and beats the control's
/// NLL by more than `noise`.
fn judge(depths: &[usize], progressive: &[(Read, u32)], control: &[(Read, u32)], noise: f64) {
    for ((d, (p, rung)), (c, _)) in depths.iter().zip(progressive).zip(control) {
        if *rung == 0 {
            assert!(
                (p.nll - c.nll).abs() <= noise,
                "depth {d}: inside the trained window both schedules are rung 1, and they \
                 disagree by more than the run-to-run spread {noise:e} — progressive {p:?} \
                 against control {c:?}"
            );
        } else {
            assert!(
                p.recovered,
                "depth {d} (rung {rung}): progressive YaRN lost the needles — {p:?}"
            );
            assert!(
                p.nll + noise < c.nll,
                "depth {d} (rung {rung}): progressive NLL {} is not below rung-1 \
                 extrapolation's {} by more than the run-to-run spread {noise:e}",
                p.nll,
                c.nll
            );
        }
    }
}

/// **Qwen3-30B-A3B, the original 32K release**: gates 3, 4 and 5.
///
/// Gate 3 at 16K (inside the window), 48K and 63K (the ×2 rung) and 112K
/// (the ×4 rung). Gate 4 decodes an 8K conversation on the trained rung and a
/// 63K one on the ×2 rung in the same waves and requires each to say what it
/// says alone. Gate 5 grows a 30K conversation past 32,768 with a 10K tool
/// result, so the sequence changes rung between two turns, and reads back
/// needles placed before the change.
#[test]
#[ignore = "loads Qwen3-30B-A3B Q4_K_M (~18 GB) and prefills to 112K; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn yarn_gates_qwen3_30b_a3b() -> Result<()> {
    use crate::models::quantized_qwen3_moe::{GgufLoadOptions, ModelWeights};

    let device = Device::new_cuda(0)?;
    let tok = tokenizer("Qwen/Qwen3-30B-A3B", "main")?;
    let path = hf_get(
        "unsloth/Qwen3-30B-A3B-GGUF",
        RepoType::Model,
        "main",
        "Qwen3-30B-A3B-Q4_K_M.gguf",
    )?;
    ensure_vram_governor(&device);
    let weights = ModelWeights::from_gguf_with_options(
        &path,
        &device,
        None,
        GgufLoadOptions {
            int8mode: Some(Int8Mode::auto(&device)),
            expert_pack_dir: path.parent().map(|p| p.to_path_buf()),
        },
    )?;
    let inv = weights
        .rope_inv_freq()
        .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
    let progressive =
        RopePreset::qwen3().gqa_schedule(inv.clone(), Some(1e6), DeclaredScaling::None, 40_960)?;
    let control = RopeSchedule::stated(inv, usize::MAX)?;
    let depths = [16_384, 49_152, 64_512, 114_688];

    println!("\n=== Qwen3-30B-A3B: progressive YaRN gates ===\n");
    let model = BatchedInference::new_with_schedule(weights, &progressive, 4096, &device)?;
    let prog = reads(&model, &tok, "progressive", &depths)?;
    let noise = noise_floor(&model, &tok, depths[1], &prog[1].0)?;

    // Gate 4: one conversation per rung, decoding in the same waves — long
    // enough for the four-line answer and its end of turn.
    const ANSWER_TOKENS: usize = 48;
    let a = needle_prompt(&tok, 8_192)?;
    let b = needle_prompt(&tok, 64_512)?;
    let cfg = BatchedConfig::default;
    let (alone_a, _) = greedy_together(&model, cfg(), std::slice::from_ref(&a), ANSWER_TOKENS)?;
    let (alone_b, _) = greedy_together(&model, cfg(), std::slice::from_ref(&b), ANSWER_TOKENS)?;
    let (mixed, rungs) = greedy_together(&model, cfg(), &[a, b], ANSWER_TOKENS)?;
    println!("  gate 4: rungs {rungs:?}, alone {alone_a:?} / {alone_b:?}, mixed {mixed:?}");
    assert_eq!(
        rungs,
        vec![0, 1],
        "the two conversations must sit on different rungs"
    );
    // The answers, to the turn's end: what each conversation SAYS.
    let answer_a = answer_of(&tok, &alone_a[0])?;
    let answer_b = answer_of(&tok, &alone_b[0])?;
    for (i, (want, what)) in [(&answer_a, "rung-0"), (&answer_b, "rung-1")]
        .iter()
        .enumerate()
    {
        let said = tok.decode(want, false).unwrap_or_default();
        assert!(
            said.contains(NEEDLE_KEY),
            "the {what} conversation alone did not answer with the needle: {said:?}"
        );
        assert_eq!(
            &answer_of(&tok, &mixed[i])?,
            *want,
            "the {what} conversation's answer changed in a mixed wave"
        );
    }

    // Gate 5: a ceiling crossed between two turns.
    let (before, after, crossed) =
        crossing_read(&model, BatchedConfig::default(), &tok, 30_720, 10_240)?;
    report("crossing 30K → 40K", 40_960, 0, after, &crossed, &tok);
    assert_eq!(
        (before, after),
        (0, 1),
        "the conversation must cross one ceiling"
    );
    assert!(
        crossed.recovered,
        "the needles placed before the rung change were lost after it — {crossed:?}"
    );

    let model = BatchedInference::new_with_schedule(model.into_inner(), &control, 4096, &device)?;
    let ctrl = reads(&model, &tok, "rung-1 extrapolation", &depths)?;
    judge(&depths, &prog, &ctrl, noise);
    println!("\n✓ Qwen3-30B-A3B progressive YaRN gates hold");
    Ok(())
}

/// **Qwen3.5-0.8B**: gate 3 at 384K and 512K — the gate that backs its rungs,
/// which the 0.8B's own card does not publish (§2) — with 128K inside the
/// window as the agreement check.
#[test]
#[ignore = "loads Qwen3.5-0.8B and prefills to 512K; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn yarn_gates_qwen35_0_8b() -> Result<()> {
    use crate::models::quantized_qwen35::{
        from_gguf_path, QWEN35_0_8B, TOKENIZER_REPO, TOKENIZER_REV,
    };
    use crate::models::qwen35::rope::LINEAGE_L0;
    use crate::models::qwen35::Qwen35LoadOptions;

    let device = Device::new_cuda(0)?;
    let tok = tokenizer(TOKENIZER_REPO, TOKENIZER_REV)?;
    let (repo, rev, file) = QWEN35_0_8B;
    let path = hf_get(repo, RepoType::Model, rev, file)?;
    ensure_vram_governor(&device);
    let mut model = from_gguf_path(&path, &device, Qwen35LoadOptions::default())?;
    let depths = [131_072, 393_216, 524_288 - 8_192];
    assert!(depths[1] > LINEAGE_L0);

    println!("\n=== Qwen3.5-0.8B: progressive YaRN gates ===\n");
    let prog = reads(&model, &tok, "progressive", &depths)?;
    let noise = noise_floor(&model, &tok, depths[1], &prog[1].0)?;
    let control = RopeSchedule::stated(model.rope().inv_freq(0).to_vec(), usize::MAX)?
        .with_lo_angle(model.rope().lo_angle());
    model.set_rope_schedule(&control)?;
    let ctrl = reads(&model, &tok, "rung-1 extrapolation", &depths)?;
    judge(&depths, &prog, &ctrl, noise);
    println!("\n✓ Qwen3.5-0.8B progressive YaRN gates hold");
    Ok(())
}

/// **Qwen3.8-Flash-Next**: gate 3 at 384K and 512K, with QSA selecting at its
/// released budget under both schedules, and 128K inside the window.
#[test]
#[ignore = "loads the merged Flash-Next engine GGUF and prefills to 512K; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn yarn_gates_qwen38_flash_next() -> Result<()> {
    use crate::models::quantized_qwen38_moe::{TOKENIZER_REPO, TOKENIZER_REV};
    use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
    use std::path::PathBuf;

    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let tok = tokenizer(TOKENIZER_REPO, TOKENIZER_REV)?;
    let gpu = Qwen4ExpGpu::load(&merged, &device, Int8Mode::auto(&device))?;
    let mut model = Qwen4ExpBatched::new(gpu)?;
    let depths = [131_072, 393_216, 524_288 - 8_192];

    println!("\n=== Qwen3.8-Flash-Next: progressive YaRN gates ===\n");
    let prog = reads(&model, &tok, "progressive", &depths)?;
    let noise = noise_floor(&model, &tok, depths[1], &prog[1].0)?;
    let control = RopeSchedule::stated(model.rope().inv_freq(0).to_vec(), usize::MAX)?
        .with_lo_angle(model.rope().lo_angle());
    model.set_rope_schedule(&control)?;
    let ctrl = reads(&model, &tok, "rung-1 extrapolation", &depths)?;
    judge(&depths, &prog, &ctrl, noise);
    println!("\n✓ Qwen3.8-Flash-Next progressive YaRN gates hold");
    Ok(())
}
