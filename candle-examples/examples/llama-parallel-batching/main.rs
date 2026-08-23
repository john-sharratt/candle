//! Continuous-batching demo for quantized Llama-3.2-1B via `BatchedInference`.
//!
//! Two sequences with independent KV state are prefilled to **deliberately
//! different lengths**, then decoded together — one `forward_wave` per step
//! covering both — and the same work is re-timed one sequence at a time. The
//! ratio is what batching buys on this model: the weights are read once per
//! wave instead of once per sequence, so a second sequence costs far less than
//! a second pass.
//!
//! Misaligned prefill lengths are the point, not an accident: they put the two
//! sequences at different positions in their KV arenas, which is the case a
//! batched decode has to get right and a uniform-length demo never exercises.
//!
//! Not a CLI — a fixed harness. Downloads `bartowski/Llama-3.2-1B-Instruct-GGUF`
//! and runs. For throughput across many sessions with output-validity checking,
//! see `quantized_llama`'s `test_parallel_batched_forwarding` instead; this is a
//! usage example, not a benchmark gate.

use candle::{Device, Result, Tensor};
use candle_transformers::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
};
use candle_transformers::models::batched_model::BatchedInference;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::{api::sync::Api, Repo, RepoType};

/// Generation steps timed in each of the two decode regimes.
const DECODE_STEPS: usize = 32;

fn main() -> Result<()> {
    println!("\n=== Llama Parallel Continuous Batching ===\n");

    let api = Api::new().map_err(|e| candle::Error::Msg(format!("HF API error: {e}")))?;
    let repo = api.repo(Repo::with_revision(
        "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    println!("Downloading Llama-3.2-1B-Instruct...");
    let model_path = repo
        .get("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        .map_err(|e| candle::Error::Msg(format!("Download failed: {e}")))?;

    let device = Device::cuda_if_available(0)?;
    println!("Device: {device:?}\n");

    println!("Loading model...");
    let t0 = std::time::Instant::now();
    let inner = ModelWeights::from_gguf_by_path_v3(&model_path, &device)?;
    let inv_freq = inner
        .rope_inv_freq()
        .ok_or_else(|| candle::Error::Msg("model has no RoPE inv_freq".into()))?;
    let model = BatchedInference::new_with_inv_freq(inner, inv_freq, 4096, &device)?;
    println!("Loaded in {:.2}s\n", t0.elapsed().as_secs_f64());

    let tok_path = api
        .model("NousResearch/Hermes-3-Llama-3.1-8B".to_string())
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(format!("tokenizer download failed: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_file(tok_path)
        .map_err(|e| candle::Error::Msg(format!("tokenizer load failed: {e}")))?;

    let prompt = |body: &str| {
        format!("<|start_header_id|>user<|end_header_id|>\n\n{body}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    };
    // Different lengths on purpose — see the module docs.
    let prompts = [
        prompt("Explain how a neural network learns, covering weights, biases, activation functions, backpropagation, and gradient descent in depth."),
        prompt("Describe the water cycle."),
    ];

    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, true)
            .map_err(|e| candle::Error::Msg(format!("tokenization failed: {e}")))?
            .get_ids()
            .to_vec())
    };
    let toks: Vec<Vec<u32>> = prompts.iter().map(|p| encode(p)).collect::<Result<_>>()?;
    println!(
        "Prompt lengths: seq0 = {}, seq1 = {} (misaligned by {})\n",
        toks[0].len(),
        toks[1].len(),
        toks[0].len().abs_diff(toks[1].len())
    );

    // ── Batched: one wave per step over both sequences ──────────────────
    let (batched, batched_time) = run(&model, &device, &toks, true)?;
    // ── Sequential: one wave per sequence per step ──────────────────────
    let (sequential, sequential_time) = run(&model, &device, &toks, false)?;

    for (i, ids) in batched.iter().enumerate() {
        let text = tokenizer
            .decode(ids, false)
            .map_err(|e| candle::Error::Msg(format!("decode failed: {e}")))?;
        println!("Sequence {i} ({} tokens): {}\n", ids.len(), text.trim());
    }

    // The two regimes run the same model on the same inputs, so they must agree
    // token for token. A divergence means batching changed the arithmetic —
    // exactly the cross-sequence bleed this shape is meant to expose.
    if batched == sequential {
        println!("Batched and sequential streams are identical ✓");
    } else {
        println!("WARNING: batched and sequential streams DIVERGED");
    }

    println!("\n=== Summary ===");
    println!("{DECODE_STEPS} decode steps × {} sequences", toks.len());
    println!("  batched:    {:.3}s", batched_time.as_secs_f64());
    println!("  sequential: {:.3}s", sequential_time.as_secs_f64());
    println!(
        "  speedup:    {:.2}x",
        sequential_time.as_secs_f64() / batched_time.as_secs_f64()
    );
    Ok(())
}

/// Prefill both prompts, then decode `DECODE_STEPS` tokens for each.
///
/// With `batched`, every step is ONE `forward_wave` naming both sequences; with
/// `batched = false`, each step issues one wave per sequence. Same session
/// shape, same inputs — only the grouping differs, which is what makes the two
/// wall-clocks comparable.
fn run(
    model: &BatchedInference<ModelWeights>,
    device: &Device,
    prompts: &[Vec<u32>],
    batched: bool,
) -> Result<(Vec<Vec<u32>>, std::time::Duration)> {
    let mut session = model.create_batched_session(BatchedConfig::default())?;

    let mut seqs = Vec::with_capacity(prompts.len());
    for _ in prompts {
        seqs.push(session.create_sequence()?);
    }

    // Prefill: one wave carrying every prompt as a PREFILL row.
    let inputs: Vec<Tensor> = prompts
        .iter()
        .map(|p| Tensor::new(&p[..], device))
        .collect::<Result<_>>()?;
    let logits = prefill_wave(model, &mut session, &seqs, &inputs)?;
    for (i, &s) in seqs.iter().enumerate() {
        session.advance_sequence(s, prompts[i].len())?;
    }

    let mut next: Vec<u32> = logits.iter().map(argmax).collect::<Result<_>>()?;
    let mut out: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();

    // Decode, timed.
    let t0 = std::time::Instant::now();
    for _ in 1..DECODE_STEPS {
        let step: Vec<Tensor> = next
            .iter()
            .map(|&t| Tensor::new(&[t], device))
            .collect::<Result<_>>()?;
        let logits = if batched {
            decode_wave(model, &mut session, &seqs, &step)?
        } else {
            let mut per_seq = Vec::with_capacity(seqs.len());
            for (i, &s) in seqs.iter().enumerate() {
                let one = std::slice::from_ref(&step[i]);
                let mut l = decode_wave(model, &mut session, &[s], one)?;
                per_seq.push(l.pop().ok_or_else(no_logits)?);
            }
            per_seq
        };
        for (i, &s) in seqs.iter().enumerate() {
            session.advance_sequence(s, 1)?;
            next[i] = argmax(&logits[i])?;
            out[i].push(next[i]);
        }
    }
    let elapsed = t0.elapsed();

    for &s in &seqs {
        session.free_sequence(s)?;
    }
    Ok((out, elapsed))
}

/// One wave whose rows are PREFILL rows (multi-token prompts).
fn prefill_wave(
    model: &BatchedInference<ModelWeights>,
    session: &mut BatchedInferenceSession,
    seqs: &[usize],
    inputs: &[Tensor],
) -> Result<Vec<Tensor>> {
    let nl = model.num_layers();
    model
        .forward_wave(session, &[], &[], seqs, inputs, &[], &[], 0, nl, None)?
        .logits_owned()
}

/// One wave whose rows are DECODE rows (a single token each).
fn decode_wave(
    model: &BatchedInference<ModelWeights>,
    session: &mut BatchedInferenceSession,
    seqs: &[usize],
    inputs: &[Tensor],
) -> Result<Vec<Tensor>> {
    let nl = model.num_layers();
    model
        .forward_wave(session, seqs, inputs, &[], &[], &[], &[], 0, nl, None)?
        .logits_owned()
}

fn no_logits() -> candle::Error {
    candle::Error::Msg("forward_wave returned no logits".into())
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let flat = logits.flatten_all()?;
    flat.argmax(flat.rank() - 1)?.to_scalar::<u32>()
}
