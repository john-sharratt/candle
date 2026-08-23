//! RULER long-context evaluation for quantized GGUF models.
//!
//! Reads a JSONL file of RULER tasks (each line: `{"index": N, "input": "...", "outputs": [...]}`),
//! processes them in batches of --batch-size sequences through the same
//! BatchedInferenceSession (parallel prefill + parallel decode), and writes a
//! predictions JSONL (`{"index": N, "pred": "..."}`) for scoring.
//!
//! Usage:
//!   cargo run --release --features cuda --example ruler-eval -- \
//!     --model qwen3-30b-a3b --compression c5 \
//!     --input-jsonl ruler_tasks_32k.jsonl \
//!     --output-jsonl ruler_preds_c5_32k.jsonl \
//!     --batch-size 8 --max-gen-tokens 50

// Batch-evaluation harness; the runner takes the batched-session argument list.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_question_mark,
    clippy::redundant_closure,
    clippy::useless_conversion
)]

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle::quantized::gguf_file;
use candle::{Device, Result, Tensor};
use candle_transformers::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, InferenceMode, ManagedBatchedModel,
};
use candle_transformers::models::batched_model::BatchedInference;
use candle_transformers::models::{
    quantized_llama, quantized_qwen2, quantized_qwen3, quantized_qwen3_moe,
};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use tokenizers::Tokenizer;

/// Step size matching KV cache CHUNK_SIZE so sealed chunks get compressed between steps.
const PREFILL_STEP: usize = 32;

/// Token IDs that signal end-of-generation for each model family.
const QWEN_EOS_IDS: &[u32] = &[151645, 151643]; // <|im_end|>, <|endoftext|>
const LLAMA_EOS_IDS: &[u32] = &[128009, 128001, 2]; // <|eot_id|>, <|end_of_text|>, </s>

// ── Model registry (shared with perplexity-eval) ──────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum ModelFamily {
    Qwen3_30bA3b,
    Qwen3_14b,
    Qwen3_8b,
    Qwen3_8bQ8,
    Qwen2_7b,
    Llama3_2_3b,
}

impl ModelFamily {
    fn gguf_repo(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
            Self::Qwen3_14b => "unsloth/Qwen3-14B-GGUF",
            Self::Qwen3_8b | Self::Qwen3_8bQ8 => "unsloth/Qwen3-8B-GGUF",
            Self::Qwen2_7b => "Qwen/Qwen2-7B-Instruct-GGUF",
            Self::Llama3_2_3b => "VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF",
        }
    }

    fn gguf_filename(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
            Self::Qwen3_14b => "Qwen3-14B-Q4_K_M.gguf",
            Self::Qwen3_8b => "Qwen3-8B-Q4_K_M.gguf",
            Self::Qwen3_8bQ8 => "Qwen3-8B-Q8_0.gguf",
            Self::Qwen2_7b => "qwen2-7b-instruct-q4_0.gguf",
            Self::Llama3_2_3b => "model-Q4_K_M.gguf",
        }
    }

    fn tokenizer_repo(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "Qwen/Qwen3-30B-A3B",
            Self::Qwen3_14b => "Qwen/Qwen3-14B",
            Self::Qwen3_8b | Self::Qwen3_8bQ8 => "Qwen/Qwen3-8B",
            Self::Qwen2_7b => "Qwen/Qwen2-7B-Instruct",
            Self::Llama3_2_3b => "NousResearch/Hermes-3-Llama-3.2-3B",
        }
    }

    fn arch(&self) -> ModelArch {
        match self {
            Self::Qwen3_30bA3b => ModelArch::Qwen3Moe,
            Self::Qwen3_14b | Self::Qwen3_8b | Self::Qwen3_8bQ8 => ModelArch::Qwen3,
            Self::Qwen2_7b => ModelArch::Qwen2,
            Self::Llama3_2_3b => ModelArch::Llama,
        }
    }

    fn eos_ids(&self) -> &'static [u32] {
        match self {
            Self::Llama3_2_3b => LLAMA_EOS_IDS,
            _ => QWEN_EOS_IDS,
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "Qwen3-30B-A3B",
            Self::Qwen3_14b => "Qwen3-14B",
            Self::Qwen3_8b => "Qwen3-8B-Q4",
            Self::Qwen3_8bQ8 => "Qwen3-8B-Q8",
            Self::Qwen2_7b => "Qwen2-7B",
            Self::Llama3_2_3b => "Llama-3.2-3B",
        }
    }
}

#[derive(Debug, Clone)]
enum ModelArch {
    Qwen3,
    Qwen3Moe,
    Qwen2,
    Llama,
}

// ── Compression arg (same as perplexity-eval) ─────────────────────────────────

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, ValueEnum)]
enum CompressionArg {
    None,
    F16,
    Bf16,
    Q8_0,
    Q8_1,
    Q4_0,
    Q3_0,
    Q2_0,
    C0,
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
    C9,
    C10,
}

impl CompressionArg {
    fn to_inference_mode(&self) -> Option<InferenceMode> {
        match self {
            Self::None => Option::None,
            Self::F16 => Some(InferenceMode::F16),
            Self::Bf16 => Some(InferenceMode::BF16),
            Self::Q8_0 => Some(InferenceMode::Q8_0),
            Self::Q8_1 => Some(InferenceMode::Q8_1),
            Self::Q4_0 => Some(InferenceMode::Q4_0),
            Self::Q3_0 => Some(InferenceMode::Q3_0),
            Self::Q2_0 => Some(InferenceMode::Q2_0),
            Self::C0 => Some(InferenceMode::C0),
            Self::C1 => Some(InferenceMode::C1),
            Self::C2 => Some(InferenceMode::C2),
            Self::C3 => Some(InferenceMode::C3),
            Self::C4 => Some(InferenceMode::C4),
            Self::C5 => Some(InferenceMode::C5),
            Self::C6 => Some(InferenceMode::C6),
            Self::C7 => Some(InferenceMode::C7),
            Self::C8 => Some(InferenceMode::C8),
            Self::C9 => Some(InferenceMode::C9),
            Self::C10 => Some(InferenceMode::C10),
        }
    }

    fn label(&self) -> &str {
        match self {
            Self::None => "none",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q4_0 => "Q4_0",
            Self::Q3_0 => "Q3_0",
            Self::Q2_0 => "Q2_0",
            Self::C0 => "C0",
            Self::C1 => "C1",
            Self::C2 => "C2",
            Self::C3 => "C3",
            Self::C4 => "C4",
            Self::C5 => "C5",
            Self::C6 => "C6",
            Self::C7 => "C7",
            Self::C8 => "C8",
            Self::C9 => "C9",
            Self::C10 => "C10",
        }
    }
}

// ── Model enum ────────────────────────────────────────────────────────────────

enum Model {
    Qwen3(quantized_qwen3::ModelWeights),
    Qwen3Moe(quantized_qwen3_moe::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Llama(quantized_llama::ModelWeights),
}

impl Model {
    fn load(arch: &ModelArch, model_path: &std::path::Path, device: &Device) -> Result<Self> {
        match arch {
            ModelArch::Qwen3Moe => {
                let m =
                    quantized_qwen3_moe::ModelWeights::from_gguf_by_path(model_path, device, None)?;
                Ok(Model::Qwen3Moe(m))
            }
            _ => {
                let mut file = std::fs::File::open(model_path)?;
                let content = gguf_file::Content::read(&mut file)?;
                match arch {
                    ModelArch::Qwen3 => Ok(Model::Qwen3(quantized_qwen3::ModelWeights::from_gguf(
                        content, &mut file, device,
                    )?)),
                    ModelArch::Qwen2 => Ok(Model::Qwen2(quantized_qwen2::ModelWeights::from_gguf(
                        content, &mut file, device,
                    )?)),
                    ModelArch::Llama => Ok(Model::Llama(quantized_llama::ModelWeights::from_gguf(
                        content, &mut file, device,
                    )?)),
                    ModelArch::Qwen3Moe => unreachable!(),
                }
            }
        }
    }
}

// ── BatchedModel wrapper ──────────────────────────────────────────────────────

enum BatchedModel {
    Qwen3(BatchedInference<quantized_qwen3::ModelWeights>),
    Qwen3Moe(BatchedInference<quantized_qwen3_moe::ModelWeights>),
    Qwen2(BatchedInference<quantized_qwen2::ModelWeights>),
    Llama(BatchedInference<quantized_llama::ModelWeights>),
}

impl BatchedModel {
    fn from_model(model: Model, device: &Device) -> Result<Self> {
        match model {
            Model::Qwen3(m) => {
                let inv_freq = m
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
                Ok(Self::Qwen3(BatchedInference::new_with_inv_freq(
                    m, inv_freq, 4096, device,
                )?))
            }
            Model::Qwen3Moe(m) => {
                let inv_freq = m
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
                Ok(Self::Qwen3Moe(BatchedInference::new_with_inv_freq(
                    m, inv_freq, 4096, device,
                )?))
            }
            Model::Qwen2(m) => {
                let inv_freq = m
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
                Ok(Self::Qwen2(BatchedInference::new_with_inv_freq(
                    m, inv_freq, 4096, device,
                )?))
            }
            Model::Llama(m) => {
                let inv_freq = m
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
                Ok(Self::Llama(BatchedInference::new_with_inv_freq(
                    m, inv_freq, 4096, device,
                )?))
            }
        }
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        match self {
            Self::Qwen3(bi) => bi.create_batched_session(config),
            Self::Qwen3Moe(bi) => bi.create_batched_session(config),
            Self::Qwen2(bi) => bi.create_batched_session(config),
            Self::Llama(bi) => bi.create_batched_session(config),
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            Self::Qwen3(bi) => bi.num_layers(),
            Self::Qwen3Moe(bi) => bi.num_layers(),
            Self::Qwen2(bi) => bi.num_layers(),
            Self::Llama(bi) => bi.num_layers(),
        }
    }

    /// One wave over `seqs`, returning a logit tensor per sequence.
    ///
    /// Prefill and decode rows are separate arguments to `forward_wave` — the
    /// engine batches multi-token prompt rows and single-token decode rows in
    /// the same launch — so the caller says which it has rather than the model
    /// inferring it from input length.
    fn wave(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        inputs: &[Tensor],
        prefill: bool,
    ) -> Result<Vec<Tensor>> {
        let nl = self.num_layers();
        let (dec_s, dec_i, pre_s, pre_i): (&[usize], &[Tensor], &[usize], &[Tensor]) = if prefill {
            (&[], &[], seqs, inputs)
        } else {
            (seqs, inputs, &[], &[])
        };
        let step = match self {
            Self::Qwen3(bi) => {
                bi.forward_wave(session, dec_s, dec_i, pre_s, pre_i, &[], &[], 0, nl, None)
            }
            Self::Qwen3Moe(bi) => {
                bi.forward_wave(session, dec_s, dec_i, pre_s, pre_i, &[], &[], 0, nl, None)
            }
            Self::Qwen2(bi) => {
                bi.forward_wave(session, dec_s, dec_i, pre_s, pre_i, &[], &[], 0, nl, None)
            }
            Self::Llama(bi) => {
                bi.forward_wave(session, dec_s, dec_i, pre_s, pre_i, &[], &[], 0, nl, None)
            }
        }?;
        step.logits_owned()
    }
}

// ── JSONL task / prediction structs ──────────────────────────────────────────

#[derive(Deserialize)]
struct RulerTask {
    /// Unique index within the JSONL file (used to correlate predictions).
    index: usize,
    /// The full prompt text (may be many thousands of tokens long).
    input: String,
    /// Expected answers (for reference; scoring is done by Python harness).
    #[allow(dead_code)]
    outputs: Vec<String>,
}

#[derive(Serialize)]
struct RulerPred {
    index: usize,
    pred: String,
}

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "RULER long-context evaluation for quantized GGUF models")]
struct Args {
    /// Model to evaluate.
    #[arg(long)]
    model: ModelFamily,

    /// Path to GGUF file (downloads from HF if omitted).
    #[arg(long)]
    model_file: Option<String>,

    /// Path to tokenizer.json (downloads from HF if omitted).
    #[arg(long)]
    tokenizer: Option<String>,

    /// Input JSONL file (RULER task format).
    #[arg(long)]
    input_jsonl: String,

    /// Output JSONL file for predictions.
    #[arg(long)]
    output_jsonl: String,

    /// KV cache compression mode.
    #[arg(long, default_value = "none")]
    compression: CompressionArg,

    /// Maximum tokens to generate per task.
    #[arg(long, default_value_t = 50)]
    max_gen_tokens: usize,

    /// Total KV token budget across all sequences in one batch.
    /// Batch size is derived as floor(token_budget / max_prompt_len), clamped to [1, tasks].
    /// Default 65536 fits Qwen3-30B-A3B on 16 GB VRAM at any supported context length.
    #[arg(long, default_value_t = 65536)]
    token_budget: usize,

    /// Override the derived batch size explicitly (ignores --token-budget).
    #[arg(long)]
    batch_size: Option<usize>,

    /// Use CPU (default: CUDA).
    #[arg(long)]
    cpu: bool,

    /// Only process the first N tasks (for quick smoke tests).
    #[arg(long)]
    limit: Option<usize>,
}

// ── Core generate logic ───────────────────────────────────────────────────────

/// Tokenise a text string using the given tokenizer.
fn tokenize(tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<u32>> {
    let enc = tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
    Ok(enc.get_ids().to_vec())
}

/// Greedy argmax over a logit tensor of shape [vocab] or [1, vocab].
fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()
}

/// Process a mini-batch of tasks in parallel through one BatchedInferenceSession.
///
/// # Prefill
/// Each task may have a slightly different prompt length.  We iterate through
/// chunk positions [0, PREFILL_STEP, 2*PREFILL_STEP, ...] up to the longest
/// prompt in the batch.  At each position we collect all sequences that still
/// have real tokens there, group them by chunk length (usually just one group
/// since RULER tasks at a fixed context size are nearly equal length), and
/// call `forward_batched` for each group.  This avoids padding pollution and
/// correctly captures the last-real-token logits per sequence.
///
/// # Decode
/// After prefill we run one `forward_batched` call per decode step across all
/// still-active sequences (those that haven't yet emitted EOS or reached
/// `max_gen_tokens`).  This is where the throughput win comes from: instead of
/// one sequence at a time we exploit the full batch parallelism of the paged
/// decode kernel.
fn run_batch(
    model: &BatchedModel,
    tokenizer: &Tokenizer,
    tasks: &[&RulerTask],
    prompt_tokens: &[&Vec<u32>],
    compression: &CompressionArg,
    max_gen_tokens: usize,
    eos_ids: &[u32],
    device: &Device,
) -> anyhow::Result<Vec<String>> {
    let n = tasks.len();

    // Use the pre-tokenized prompts passed in.
    let max_prompt_len = prompt_tokens.iter().map(|t| t.len()).max().unwrap_or(0);

    // Build the session.
    let session_config = match compression.to_inference_mode() {
        Some(mode) => BatchedConfig {
            k_format: mode.k_format(),
            v_format: mode.v_format(),
            compression_level: mode.compression_level(),
            ..BatchedConfig::default()
        },
        None => BatchedConfig::default(),
    };
    let mut session = model.create_batched_session(session_config)?;

    // Create one sequence per task and remember its index.
    let seq_indices: Vec<usize> = (0..n)
        .map(|_| session.create_sequence())
        .collect::<Result<_>>()?;

    // ── Batched Prefill ───────────────────────────────────────────────────────
    // last_logits[i] will hold the logits from the last real-token chunk for seq i.
    let mut last_logits: Vec<Option<Tensor>> = vec![None; n];

    let mut offset = 0usize;
    while offset < max_prompt_len {
        let chunk_end_max = (offset + PREFILL_STEP).min(max_prompt_len);

        // Collect sequences that still have real tokens at this offset.
        // Group by actual chunk length so sequences with different remaining
        // lengths (only differs in the very last chunk) are forwarded correctly.
        let mut by_chunk_len: std::collections::BTreeMap<usize, (Vec<usize>, Vec<Tensor>)> =
            std::collections::BTreeMap::new();

        for (i, tokens) in prompt_tokens.iter().enumerate() {
            if offset >= tokens.len() {
                // This sequence is fully prefilled.
                continue;
            }
            let end = chunk_end_max.min(tokens.len());
            let chunk_len = end - offset;
            let chunk = &tokens[offset..end];
            let tensor = Tensor::new(chunk, device)?;
            let entry = by_chunk_len
                .entry(chunk_len)
                .or_insert_with(|| (vec![], vec![]));
            entry.0.push(seq_indices[i]);
            entry.1.push(tensor);
        }

        for (chunk_len, (group_seq_idxs, group_inputs)) in by_chunk_len {
            let logits_vec = model.wave(&mut session, &group_seq_idxs, &group_inputs, true)?;
            // Advance each sequence and save its logits.
            for ((&seq_idx, logits), orig_i) in group_seq_idxs
                .iter()
                .zip(logits_vec.into_iter())
                .zip(by_chunk_positions_to_orig(&seq_indices, &group_seq_idxs))
            {
                session.advance_sequence(seq_idx, chunk_len)?;
                last_logits[orig_i] = Some(logits);
            }
        }

        offset = chunk_end_max;
    }

    // ── Batched Decode ────────────────────────────────────────────────────────
    // Per-sequence state.
    let mut generated: Vec<Vec<u32>> = vec![Vec::with_capacity(max_gen_tokens); n];
    let mut done: Vec<bool> = vec![false; n];

    // Derive first token for each sequence from last prefill logits.
    let mut current_tokens: Vec<u32> = last_logits
        .iter()
        .map(|lo| {
            lo.as_ref()
                .map(|l| argmax(l))
                .unwrap_or(Err(candle::Error::Msg("no prefill logits".into())))
        })
        .collect::<Result<_>>()?;

    for _step in 0..max_gen_tokens {
        // Mark any sequences that just emitted EOS.
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

        // Build inputs only for still-active sequences.
        let active_indices: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
        let active_seq_idxs: Vec<usize> = active_indices.iter().map(|&i| seq_indices[i]).collect();
        let active_inputs: Vec<Tensor> = active_indices
            .iter()
            .map(|&i| Ok(Tensor::new(&[current_tokens[i]], device)?))
            .collect::<Result<_>>()?;

        let logits_vec = model.wave(&mut session, &active_seq_idxs, &active_inputs, false)?;

        // Advance all active sequences.
        for &seq_idx in &active_seq_idxs {
            session.advance_sequence(seq_idx, 1)?;
        }

        // Sample next tokens for active sequences; leave done sequences unchanged.
        for (orig_i, logits) in active_indices.iter().zip(logits_vec.iter()) {
            current_tokens[*orig_i] = argmax(logits)?;
        }
    }

    // Free sequences.
    for &seq_idx in &seq_indices {
        session.free_sequence(seq_idx)?;
    }
    session.release_empty_arenas()?;

    // Decode token IDs to strings.
    let predictions = generated
        .iter()
        .map(|toks| {
            tokenizer
                .decode(toks, true)
                .map(|s| s.trim().to_string())
                .map_err(anyhow::Error::msg)
        })
        .collect::<anyhow::Result<_>>()?;

    Ok(predictions)
}

/// Given the full `seq_indices` vec and a subset `group_seq_idxs`,
/// return the original 0-based positions of each group member.
fn by_chunk_positions_to_orig(seq_indices: &[usize], group_seq_idxs: &[usize]) -> Vec<usize> {
    group_seq_idxs
        .iter()
        .map(|&gsidx| {
            seq_indices
                .iter()
                .position(|&s| s == gsidx)
                .expect("seq_idx not found in seq_indices")
        })
        .collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    // ── Load tokenizer ────────────────────────────────────────────────────────
    let tokenizer = match &args.tokenizer {
        Some(path) => Tokenizer::from_file(path).map_err(anyhow::Error::msg)?,
        None => {
            let repo = args.model.tokenizer_repo();
            println!("Downloading tokenizer from {repo}...");
            let api = hf_hub::api::sync::Api::new()?;
            let path = api.model(repo.to_string()).get("tokenizer.json")?;
            Tokenizer::from_file(path).map_err(anyhow::Error::msg)?
        }
    };

    // ── Load model ────────────────────────────────────────────────────────────
    let model_path: std::path::PathBuf = match &args.model_file {
        Some(p) => p.into(),
        None => {
            let repo = args.model.gguf_repo();
            let filename = args.model.gguf_filename();
            println!("Downloading {filename} from {repo}...");
            let api = hf_hub::api::sync::Api::new()?;
            api.model(repo.to_string()).get(filename)?
        }
    };

    println!(
        "Loading {} ({})...",
        args.model.name(),
        args.compression.label()
    );
    let model = Model::load(&args.model.arch(), &model_path, &device)?;
    let batched = BatchedModel::from_model(model, &device)?;
    println!("Model loaded.");

    // ── Read tasks ────────────────────────────────────────────────────────────
    let input_file = std::fs::File::open(&args.input_jsonl)?;
    let reader = std::io::BufReader::new(input_file);
    let mut tasks: Vec<RulerTask> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let task: RulerTask = serde_json::from_str(trimmed)?;
        tasks.push(task);
    }
    if let Some(limit) = args.limit {
        tasks.truncate(limit);
    }
    println!("Loaded {} tasks from {}", tasks.len(), args.input_jsonl);

    // ── Pre-tokenize to derive batch size ─────────────────────────────────────
    // We need the max prompt length to budget VRAM correctly.
    print!("Tokenizing prompts... ");
    std::io::stdout().flush()?;
    let all_prompt_tokens: Vec<Vec<u32>> = tasks
        .iter()
        .map(|t| tokenize(&tokenizer, &t.input))
        .collect::<anyhow::Result<_>>()?;
    let max_prompt_len = all_prompt_tokens.iter().map(|t| t.len()).max().unwrap_or(1);
    println!("max prompt = {} tokens", max_prompt_len);

    let batch_size = match args.batch_size {
        Some(b) => b.max(1),
        None => (args.token_budget / max_prompt_len).max(1).min(tasks.len()),
    };
    println!(
        "Batch size: {} (token_budget={}, max_prompt_len={})",
        batch_size, args.token_budget, max_prompt_len
    );

    // ── Run tasks in batches ──────────────────────────────────────────────────
    let output_file = std::fs::File::create(&args.output_jsonl)?;
    let mut writer = std::io::BufWriter::new(output_file);
    let eos_ids = args.model.eos_ids();
    let n_tasks = tasks.len();

    let mut completed = 0usize;
    for (chunk_tasks, chunk_tokens) in tasks
        .chunks(batch_size)
        .zip(all_prompt_tokens.chunks(batch_size))
    {
        let task_refs: Vec<&RulerTask> = chunk_tasks.iter().collect();
        let token_refs: Vec<&Vec<u32>> = chunk_tokens.iter().collect();
        let chunk_max = chunk_tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        println!(
            "[{}/{}] batch of {} tasks, max_prompt={} tokens (indices {:?})...",
            completed + 1,
            n_tasks,
            chunk_tasks.len(),
            chunk_max,
            chunk_tasks.iter().map(|t| t.index).collect::<Vec<_>>()
        );

        let preds = run_batch(
            &batched,
            &tokenizer,
            &task_refs,
            &token_refs,
            &args.compression,
            args.max_gen_tokens,
            eos_ids,
            &device,
        )?;

        for (task, pred) in chunk_tasks.iter().zip(preds.iter()) {
            let preview = if pred.len() > 60 { &pred[..60] } else { pred };
            println!("  index={} => {:?}", task.index, preview);
            let record = RulerPred {
                index: task.index,
                pred: pred.clone(),
            };
            writeln!(writer, "{}", serde_json::to_string(&record)?)?;
        }
        completed += chunk_tasks.len();
    }
    writer.flush()?;

    println!(
        "\nDone. {} predictions written to {}",
        n_tasks, args.output_jsonl
    );
    Ok(())
}
