//! WikiText-2 perplexity evaluation for quantized GGUF models.
//!
//! Evaluates language model quality by computing perplexity on the WikiText-2
//! test set. Supports all model families from §9.9 of the unbounded context paper.
//!
//! Supports KV cache compression modes C0-C9 for measuring quality impact.
//!
//! Usage:
//!   # Baseline (no compression):
//!   cargo run --release --features cuda --example perplexity-eval -- \
//!     --model qwen3-8b --text-file wiki.test.raw
//!
//!   # With compression:
//!   cargo run --release --features cuda --example perplexity-eval -- \
//!     --model qwen3-8b --text-file wiki.test.raw --compression c5

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

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
use tokenizers::Tokenizer;

const DEFAULT_CONTEXT_SIZE: usize = 2048;

/// Step size for incremental processing with compression.
/// Matches the KV cache CHUNK_SIZE so sealed chunks get compressed between steps.
const COMPRESSION_STEP_SIZE: usize = 32;

#[derive(Debug, Clone, ValueEnum)]
enum ModelFamily {
    /// Qwen3-30B-A3B (MoE architecture)
    Qwen3_30bA3b,
    /// Qwen3-14B dense
    Qwen3_14b,
    /// Qwen3-8B dense Q4_K_M
    Qwen3_8b,
    /// Qwen3-8B dense Q8_0
    Qwen3_8bQ8,
    /// Qwen2-7B dense Q4_0
    Qwen2_7b,
    /// Qwen2-7B dense Q8_0
    Qwen2_7bQ8,
    /// Qwen2-0.5B dense Q4_0
    Qwen2_0_5b,
    /// Qwen2-0.5B native (F16 weights)
    Qwen2_0_5bF16,
    /// Llama-3.1-8B
    Llama3_1_8b,
    /// Llama-3.2-3B Q4_K_M
    Llama3_2_3b,
    /// Llama-3.2-3B native (F16 weights)
    Llama3_2_3bF16,
}

impl ModelFamily {
    fn gguf_repo(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
            Self::Qwen3_14b => "unsloth/Qwen3-14B-GGUF",
            Self::Qwen3_8b | Self::Qwen3_8bQ8 => "unsloth/Qwen3-8B-GGUF",
            Self::Qwen2_7b | Self::Qwen2_7bQ8 => "Qwen/Qwen2-7B-Instruct-GGUF",
            Self::Qwen2_0_5b | Self::Qwen2_0_5bF16 => "Qwen/Qwen2-0.5B-Instruct-GGUF",
            Self::Llama3_1_8b => "bartowski/Meta-Llama-3.1-8B-GGUF",
            Self::Llama3_2_3b => "VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF",
            Self::Llama3_2_3bF16 => "bartowski/Llama-3.2-3B-Instruct-GGUF",
        }
    }

    fn gguf_filename(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
            Self::Qwen3_14b => "Qwen3-14B-Q4_K_M.gguf",
            Self::Qwen3_8b => "Qwen3-8B-Q4_K_M.gguf",
            Self::Qwen3_8bQ8 => "Qwen3-8B-Q8_0.gguf",
            Self::Qwen2_7b => "qwen2-7b-instruct-q4_0.gguf",
            Self::Qwen2_7bQ8 => "qwen2-7b-instruct-q8_0.gguf",
            Self::Qwen2_0_5b => "qwen2-0_5b-instruct-q4_0.gguf",
            Self::Qwen2_0_5bF16 => "qwen2-0_5b-instruct-fp16.gguf",
            Self::Llama3_1_8b => "Meta-Llama-3.1-8B-Q4_K_M.gguf",
            Self::Llama3_2_3b => "model-Q4_K_M.gguf",
            Self::Llama3_2_3bF16 => "Llama-3.2-3B-Instruct-f16.gguf",
        }
    }

    fn tokenizer_repo(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "Qwen/Qwen3-30B-A3B",
            Self::Qwen3_14b => "Qwen/Qwen3-14B",
            Self::Qwen3_8b | Self::Qwen3_8bQ8 => "Qwen/Qwen3-8B",
            Self::Qwen2_7b | Self::Qwen2_7bQ8 => "Qwen/Qwen2-7B-Instruct",
            Self::Qwen2_0_5b | Self::Qwen2_0_5bF16 => "Qwen/Qwen2-0.5B-Instruct",
            Self::Llama3_1_8b => "NousResearch/Hermes-3-Llama-3.1-8B",
            Self::Llama3_2_3b | Self::Llama3_2_3bF16 => "NousResearch/Hermes-3-Llama-3.2-3B",
        }
    }

    fn arch(&self) -> ModelArch {
        match self {
            Self::Qwen3_30bA3b => ModelArch::Qwen3Moe,
            Self::Qwen3_14b | Self::Qwen3_8b | Self::Qwen3_8bQ8 => ModelArch::Qwen3,
            Self::Qwen2_7b | Self::Qwen2_7bQ8 | Self::Qwen2_0_5b | Self::Qwen2_0_5bF16 => {
                ModelArch::Qwen2
            }
            Self::Llama3_1_8b | Self::Llama3_2_3b | Self::Llama3_2_3bF16 => ModelArch::Llama,
        }
    }

    /// Whether this model supports KV cache compression (requires head_dim=128).
    /// Qwen2-0.5B variants have head_dim=64 and cannot use the palette4 kernel.
    fn supports_kv_compression(&self) -> bool {
        !matches!(self, Self::Qwen2_0_5b | Self::Qwen2_0_5bF16)
    }

    fn name(&self) -> &str {
        match self {
            Self::Qwen3_30bA3b => "Qwen3-30B-A3B",
            Self::Qwen3_14b => "Qwen3-14B",
            Self::Qwen3_8b => "Qwen3-8B-Q4",
            Self::Qwen3_8bQ8 => "Qwen3-8B-Q8",
            Self::Qwen2_7b => "Qwen2-7B-Q4",
            Self::Qwen2_7bQ8 => "Qwen2-7B-Q8",
            Self::Qwen2_0_5b => "Qwen2-0.5B-Q4",
            Self::Qwen2_0_5bF16 => "Qwen2-0.5B-F16",
            Self::Llama3_1_8b => "Llama-3.1-8B",
            Self::Llama3_2_3b => "Llama-3.2-3B-Q4",
            Self::Llama3_2_3bF16 => "Llama-3.2-3B-F16",
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum CompressionArg {
    None,
    F16,
    Bf16,
    Q8_0,
    Q8_1,
    Q8Ks,
    Q8Q4,
    Q4_0,
    Q4_1,
    Q4Ks,
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
            Self::Q8Ks => Some(InferenceMode::Q8_KS),
            Self::Q8Q4 => Some(InferenceMode::Q8_Q4),
            Self::Q4_0 => Some(InferenceMode::Q4_0),
            Self::Q4_1 => Some(InferenceMode::Q4_1),
            Self::Q4Ks => Some(InferenceMode::Q4_KS),
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
            Self::Q8Ks => "Q8_KS",
            Self::Q8Q4 => "Q8_Q4",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q4Ks => "Q4_KS",
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

#[derive(Debug, Clone)]
enum ModelArch {
    Qwen3,
    Qwen3Moe,
    Qwen2,
    Llama,
}

/// Wraps all supported model types for uniform perplexity evaluation.
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
                let m = quantized_qwen3_moe::ModelWeights::from_gguf_by_path(model_path, device)?;
                Ok(Model::Qwen3Moe(m))
            }
            _ => {
                let mut file = std::fs::File::open(model_path)?;
                let content = gguf_file::Content::read(&mut file)?;
                match arch {
                    ModelArch::Qwen3 => {
                        let m =
                            quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)?;
                        Ok(Model::Qwen3(m))
                    }
                    ModelArch::Qwen2 => {
                        let m =
                            quantized_qwen2::ModelWeights::from_gguf(content, &mut file, device)?;
                        Ok(Model::Qwen2(m))
                    }
                    ModelArch::Llama => {
                        let m =
                            quantized_llama::ModelWeights::from_gguf(content, &mut file, device)?;
                        Ok(Model::Llama(m))
                    }
                    ModelArch::Qwen3Moe => unreachable!(),
                }
            }
        }
    }

    fn create_kv_caches(
        &self,
        capacity: usize,
    ) -> candle_transformers::models::kv_cache_utils::KvCaches {
        match self {
            Model::Qwen3(m) => m.create_kv_caches(capacity),
            Model::Qwen3Moe(m) => m.create_kv_caches(capacity),
            Model::Qwen2(m) => m.create_kv_caches(capacity),
            Model::Llama(m) => m.create_kv_caches(capacity),
        }
    }

    fn forward_all_logits(
        &self,
        caches: &mut candle_transformers::models::kv_cache_utils::KvCaches,
        input: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        match self {
            Model::Qwen3(m) => m.forward_all_logits(caches, input, offset),
            Model::Qwen3Moe(m) => m.forward_all_logits(caches, input, offset),
            Model::Qwen2(m) => m.forward_all_logits(caches, input, offset),
            Model::Llama(m) => m.forward_all_logits(caches, input, offset),
        }
    }
}

/// Wraps all model types as BatchedInference for paged attention with compression.
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
                let mut bi = BatchedInference::new_with_inv_freq(m, inv_freq, 4096, device)?;
                bi.set_all_logits(true);
                Ok(Self::Qwen3(bi))
            }
            Model::Qwen3Moe(m) => {
                let inv_freq = m
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
                let mut bi = BatchedInference::new_with_inv_freq(m, inv_freq, 4096, device)?;
                bi.set_all_logits(true);
                Ok(Self::Qwen3Moe(bi))
            }
            Model::Qwen2(m) => {
                let inv_freq = m
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
                let mut bi = BatchedInference::new_with_inv_freq(m, inv_freq, 4096, device)?;
                bi.set_all_logits(true);
                Ok(Self::Qwen2(bi))
            }
            Model::Llama(m) => {
                let inv_freq = m
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
                let mut bi = BatchedInference::new_with_inv_freq(m, inv_freq, 4096, device)?;
                bi.set_all_logits(true);
                Ok(Self::Llama(bi))
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

    /// Forward pass returning logits for ALL positions via paged attention.
    fn forward_all_logits_batched(
        &self,
        session: &mut BatchedInferenceSession,
        seq_idx: usize,
        input: &Tensor,
    ) -> Result<Tensor> {
        let mut logits = match self {
            Self::Qwen3(bi) => bi.forward_batched(session, &[seq_idx], &[input.clone()])?,
            Self::Qwen3Moe(bi) => bi.forward_batched(session, &[seq_idx], &[input.clone()])?,
            Self::Qwen2(bi) => bi.forward_batched(session, &[seq_idx], &[input.clone()])?,
            Self::Llama(bi) => bi.forward_batched(session, &[seq_idx], &[input.clone()])?,
        };
        logits
            .pop()
            .ok_or_else(|| candle::Error::Msg("empty batched logits result".into()))
    }
}

const ALL_SWEEP_MODES: &[CompressionArg] = &[
    CompressionArg::None,
    CompressionArg::F16,
    CompressionArg::Bf16,
    CompressionArg::Q8_0,
    CompressionArg::Q8_1,
    CompressionArg::Q8Ks,
    CompressionArg::Q4_0,
    CompressionArg::Q4_1,
    CompressionArg::Q4Ks,
    CompressionArg::Q3_0,
    CompressionArg::Q2_0,
    CompressionArg::C0,
    CompressionArg::C1,
    CompressionArg::C2,
    CompressionArg::C3,
    CompressionArg::C4,
    CompressionArg::C5,
    CompressionArg::C6,
    CompressionArg::C7,
    CompressionArg::C8,
    CompressionArg::C9,
    CompressionArg::C10,
];

#[derive(Parser, Debug)]
#[command(about = "WikiText-2 perplexity evaluation for quantized GGUF models")]
struct Args {
    /// Model to evaluate (ignored when --sweep is used).
    #[arg(long, required_unless_present = "sweep")]
    model: Option<ModelFamily>,

    /// Path to GGUF model file. If omitted, downloads from HuggingFace Hub.
    #[arg(long)]
    model_file: Option<String>,

    /// Path to tokenizer.json. If omitted, downloads from HuggingFace Hub.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Path to WikiText-2 test set (raw text file).
    #[arg(
        long,
        default_value = "candle-examples/examples/perplexity-eval/wiki.test.raw"
    )]
    text_file: String,

    /// Context window size for evaluation chunks.
    #[arg(long, default_value_t = DEFAULT_CONTEXT_SIZE)]
    context_size: usize,

    /// Stride for sliding window. Defaults to context_size (no overlap).
    #[arg(long)]
    stride: Option<usize>,

    /// Maximum number of tokens to evaluate (0 = all).
    #[arg(long, default_value_t = 0)]
    max_tokens: usize,

    /// Use CPU instead of CUDA.
    #[arg(long)]
    cpu: bool,

    /// KV cache compression mode (C0-C9). Omit or use 'none' for baseline.
    #[arg(long, default_value = "none")]
    compression: CompressionArg,

    /// Run full sweep: all models × all compression modes. Prints a matrix.
    #[arg(long)]
    sweep: bool,

    /// Models to include in sweep (default: all). Ignored outside of --sweep.
    #[arg(long, value_delimiter = ',')]
    sweep_models: Vec<ModelFamily>,

    /// Compression modes to include in sweep (default: all). Ignored outside of --sweep.
    #[arg(long, value_delimiter = ',')]
    sweep_modes: Vec<CompressionArg>,

    /// Context sizes to sweep over (e.g. 128,256,512,1024,2048,4096).
    /// When set alongside --sweep, produces a context×mode PPL table for a single model.
    /// Requires exactly one model in --sweep-models.
    #[arg(long, value_delimiter = ',')]
    sweep_contexts: Vec<usize>,
}

/// Evaluate perplexity without compression (original path, full-chunk forward).
fn eval_baseline(
    model: &Model,
    tokens: &[u32],
    context_size: usize,
    stride: usize,
    device: &Device,
) -> anyhow::Result<(f64, u64)> {
    let n_tokens = tokens.len();
    let mut total_nll = 0.0f64;
    let mut total_count = 0u64;
    let mut chunk_idx = 0u32;

    let start_positions: Vec<usize> = (0..n_tokens.saturating_sub(1)).step_by(stride).collect();
    let n_chunks = start_positions.len();

    println!(
        "Evaluating {} chunks (context={}, stride={})...\n",
        n_chunks, context_size, stride
    );

    for &begin in &start_positions {
        let end = (begin + context_size).min(n_tokens);
        if end - begin < 2 {
            break;
        }

        let input_tokens = &tokens[begin..end - 1];
        let target_tokens = &tokens[begin + 1..end];

        let loss_start = if begin == 0 || stride >= context_size {
            0
        } else {
            context_size.saturating_sub(stride).saturating_sub(1)
        };
        let loss_start = loss_start.min(input_tokens.len().saturating_sub(1));

        let mut caches = model.create_kv_caches(context_size);

        let input = Tensor::new(input_tokens, device)?.unsqueeze(0)?;
        let logits = model.forward_all_logits(&mut caches, &input, 0)?;
        let logits = logits.squeeze(0)?;

        let scoring_logits = if loss_start > 0 {
            logits.narrow(0, loss_start, logits.dim(0)? - loss_start)?
        } else {
            logits
        };
        let scoring_targets = &target_tokens[loss_start..];
        let targets_tensor = Tensor::new(scoring_targets, device)?;

        let loss = candle_nn::loss::cross_entropy(&scoring_logits, &targets_tensor)?;
        let loss_val = loss.to_vec0::<f32>()? as f64;
        let n_scored = scoring_targets.len() as u64;

        total_nll += loss_val * n_scored as f64;
        total_count += n_scored;

        chunk_idx += 1;
        if chunk_idx % 10 == 0 || chunk_idx == n_chunks as u32 {
            let running_ppl = (total_nll / total_count as f64).exp();
            println!(
                "  Chunk {}/{}: loss={:.4}, running PPL={:.2} ({} tokens scored)",
                chunk_idx, n_chunks, loss_val, running_ppl, total_count
            );
        }
    }

    Ok((total_nll, total_count))
}

/// Context window size for compressed evaluation, sized to fit RTX 4090 Mobile 16 GB VRAM
/// with large models (14B, ~8.5 GB weights → ~7.5 GB headroom for KV + activations).
/// Float KV arena + quant KV arena + 48-layer forward-pass activations exhausted 16 GB
/// at 8 K–16 K context on 14B models; 4 K is the safe ceiling.
/// C0–C7: 4096, C8–C10: 4096, uniform quant modes: 4096.
fn compressed_context_size(_mode: InferenceMode) -> usize {
    4096
}

/// Evaluate perplexity with KV cache compression via paged attention.
///
/// Processes tokens in CHUNK_SIZE steps so compression reconciliation
/// affects subsequent attention computations.
fn eval_compressed(
    batched_model: &BatchedModel,
    mode: InferenceMode,
    tokens: &[u32],
    context_size_override: Option<usize>,
    _stride: usize,
    device: &Device,
) -> anyhow::Result<(f64, u64)> {
    // Use explicit override when provided (e.g. context sweep); otherwise use the
    // mode-specific safe ceiling for the RTX 4090 Mobile 16 GB VRAM.
    let context_size = context_size_override.unwrap_or_else(|| compressed_context_size(mode));
    let stride = context_size; // non-overlapping: each chunk is independent

    let n_tokens = tokens.len();
    let mut total_nll = 0.0f64;
    let mut total_count = 0u64;
    let mut chunk_idx = 0u32;

    let start_positions: Vec<usize> = (0..n_tokens.saturating_sub(1)).step_by(stride).collect();
    let n_chunks = start_positions.len();
    let step_size = COMPRESSION_STEP_SIZE;

    println!(
        "Evaluating {} chunks (context={}, stride={}, step={}, mode={:?})...\n",
        n_chunks, context_size, stride, step_size, mode
    );

    // Build config for this compression mode.
    // initial_seq_len uses the default (2048) so the arena starts small and
    // grows dynamically; pre-allocating the full context_size upfront would OOM
    // on large models. This matches the reference wiring in batch_test/utils.rs.
    let config = BatchedConfig {
        k_format: mode.k_format(),
        v_format: mode.v_format(),
        compression_level: mode.compression_level(),
        ..BatchedConfig::default()
    };

    for &begin in &start_positions {
        let end = (begin + context_size).min(n_tokens);
        if end - begin < 2 {
            break;
        }

        let input_tokens = &tokens[begin..end - 1];
        let target_tokens = &tokens[begin + 1..end];

        let loss_start = if begin == 0 || stride >= context_size {
            0
        } else {
            context_size.saturating_sub(stride).saturating_sub(1)
        };
        let loss_start = loss_start.min(input_tokens.len().saturating_sub(1));

        // Create a fresh session for each chunk
        let mut session = batched_model.create_batched_session(config.clone())?;
        let seq_idx = session.create_sequence()?;

        // Process in incremental steps, scoring each step immediately to avoid
        // accumulating O(context_size × vocab) GPU tensors (which would OOM at 4K context).
        let mut chunk_nll = 0.0f64;
        let mut chunk_count = 0u64;
        let mut offset = 0usize;

        while offset < input_tokens.len() {
            let step_end = (offset + step_size).min(input_tokens.len());
            let step_tokens = &input_tokens[offset..step_end];
            let step_len = step_tokens.len();

            let input = Tensor::new(step_tokens, device)?.unsqueeze(0)?;

            let logits = batched_model.forward_all_logits_batched(&mut session, seq_idx, &input)?;
            let logits = logits.squeeze(0)?; // [step_len, vocab]

            // Advance KV cache offset (reconciliation happens inside forward_layer_batched)
            session.advance_sequence(seq_idx, step_len)?;

            // Score this step immediately — only tokens at/after loss_start in the chunk.
            // With non-overlapping stride, loss_start == 0 always, so this is always all tokens.
            let within_step_skip = loss_start.saturating_sub(offset);
            if within_step_skip < step_len {
                let scoring_len = step_len - within_step_skip;
                let scoring_logits = if within_step_skip > 0 {
                    logits.narrow(0, within_step_skip, scoring_len)?
                } else {
                    logits
                };
                let scoring_logits = scoring_logits.to_dtype(candle::DType::F32)?;
                let step_targets = &target_tokens[offset + within_step_skip..offset + step_len];
                let targets_tensor = Tensor::new(step_targets, device)?;
                let loss = candle_nn::loss::cross_entropy(&scoring_logits, &targets_tensor)?;
                let loss_val = loss.to_vec0::<f32>()? as f64;
                chunk_nll += loss_val * scoring_len as f64;
                chunk_count += scoring_len as u64;
            }

            offset += step_len;
        }

        let chunk_loss = if chunk_count > 0 {
            chunk_nll / chunk_count as f64
        } else {
            0.0
        };
        total_nll += chunk_nll;
        total_count += chunk_count;

        // Free the sequence to release KV memory
        session.free_sequence(seq_idx)?;
        session.compact()?;

        chunk_idx += 1;
        if chunk_idx % 10 == 0 || chunk_idx == n_chunks as u32 {
            let running_ppl = (total_nll / total_count as f64).exp();
            println!(
                "  Chunk {}/{}: loss={:.4}, running PPL={:.2} ({} tokens scored)",
                chunk_idx, n_chunks, chunk_loss, running_ppl, total_count
            );
        }
    }

    Ok((total_nll, total_count))
}

/// Load token ids from a text file using the given tokenizer repo (downloads if needed).
fn load_tokens(
    text_file: &str,
    tokenizer_path: Option<&str>,
    tokenizer_repo: &str,
    max_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let tokenizer = match tokenizer_path {
        Some(path) => Tokenizer::from_file(path).map_err(anyhow::Error::msg)?,
        None => {
            println!("Downloading tokenizer from {}...", tokenizer_repo);
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(tokenizer_repo.to_string());
            let path = repo.get("tokenizer.json")?;
            Tokenizer::from_file(path).map_err(anyhow::Error::msg)?
        }
    };

    let text = std::fs::read_to_string(text_file)?;
    if text.is_empty() {
        anyhow::bail!("Text file is empty");
    }

    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(anyhow::Error::msg)?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    if max_tokens > 0 && tokens.len() > max_tokens {
        tokens.truncate(max_tokens);
    }

    if tokens.len() < 2 {
        anyhow::bail!("Need at least 2 tokens for perplexity evaluation");
    }

    Ok(tokens)
}

/// Resolve model path: use provided path or download from HF hub.
fn resolve_model_path(
    model_file: Option<&str>,
    gguf_repo: &str,
    gguf_filename: &str,
) -> anyhow::Result<std::path::PathBuf> {
    match model_file {
        Some(path) => Ok(std::path::PathBuf::from(path)),
        None => {
            println!("Downloading model from {}/{}...", gguf_repo, gguf_filename);
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(gguf_repo.to_string());
            Ok(repo.get(gguf_filename)?)
        }
    }
}

/// Run all requested compression modes for a single model.
/// Returns a Vec of (mode_label, ppl) results.
/// Loads the model weights twice: once for baseline (none), once for all compressed modes.
fn run_model_all_modes(
    family: &ModelFamily,
    model_file: Option<&str>,
    tokens: &[u32],
    modes: &[CompressionArg],
    context_size: usize,
    stride: usize,
    device: &Device,
) -> anyhow::Result<Vec<(String, String)>> {
    let model_path = resolve_model_path(model_file, family.gguf_repo(), family.gguf_filename())?;

    let has_baseline = modes.iter().any(|m| matches!(m, CompressionArg::None));
    let compressed_modes: Vec<&CompressionArg> = modes
        .iter()
        .filter(|m| !matches!(m, CompressionArg::None))
        .collect();

    let mut results: Vec<(String, String)> = Vec::new();

    // Baseline pass (none mode)
    if has_baseline {
        println!("\n  [{}] mode=none", family.name());
        let model = Model::load(&family.arch(), &model_path, device)?;
        match eval_baseline(&model, tokens, context_size, stride, device) {
            Ok((nll, count)) => {
                let ppl = (nll / count as f64).exp();
                results.push(("none".to_string(), format!("{:.2}", ppl)));
            }
            Err(e) => {
                results.push(("none".to_string(), format!("ERR:{}", e)));
            }
        }
    }

    // Compressed pass (all non-none modes in one BatchedModel load)
    if !compressed_modes.is_empty() {
        if !family.supports_kv_compression() {
            println!(
                "\n  [{}] skipping compressed modes (head_dim != 128, KV compression not supported)",
                family.name()
            );
            for mode_arg in &compressed_modes {
                results.push((mode_arg.label().to_string(), "N/A".to_string()));
            }
            return Ok(results);
        }
        println!("\n  [{}] loading for compressed modes...", family.name());
        let model = Model::load(&family.arch(), &model_path, device)?;
        let batched = BatchedModel::from_model(model, device)?;

        for mode_arg in &compressed_modes {
            let mode = mode_arg.to_inference_mode().unwrap();
            let label = mode_arg.label();
            println!("\n  [{}] mode={}", family.name(), label);
            match eval_compressed(&batched, mode, tokens, None, stride, device) {
                Ok((nll, count)) => {
                    let ppl = (nll / count as f64).exp();
                    results.push((label.to_string(), format!("{:.2}", ppl)));
                }
                Err(e) => {
                    results.push((label.to_string(), format!("ERR:{}", e)));
                }
            }
        }
    }

    Ok(results)
}

fn print_matrix(
    models: &[ModelFamily],
    modes: &[CompressionArg],
    matrix: &std::collections::HashMap<(usize, usize), String>,
) {
    let mode_labels: Vec<&str> = modes.iter().map(|m| m.label()).collect();
    let model_names: Vec<&str> = models.iter().map(|m| m.name()).collect();

    // Column widths
    let model_col_w = model_names
        .iter()
        .map(|n| n.len())
        .max()
        .unwrap_or(10)
        .max(10);
    let mode_col_w = mode_labels
        .iter()
        .map(|l| l.len())
        .max()
        .unwrap_or(6)
        .max(7);

    // Header row
    let header: String = std::iter::once(format!("{:<width$}", "Model", width = model_col_w))
        .chain(
            mode_labels
                .iter()
                .map(|l| format!("{:>width$}", l, width = mode_col_w)),
        )
        .collect::<Vec<_>>()
        .join(" | ");
    println!("\n{}", header);
    println!("{}", "-".repeat(header.len()));

    for (mi, model) in models.iter().enumerate() {
        let row: String = std::iter::once(format!("{:<width$}", model.name(), width = model_col_w))
            .chain(modes.iter().enumerate().map(|(ci, _)| {
                let val = matrix
                    .get(&(mi, ci))
                    .cloned()
                    .unwrap_or_else(|| "-".to_string());
                format!("{:>width$}", val, width = mode_col_w)
            }))
            .collect::<Vec<_>>()
            .join(" | ");
        println!("{}", row);
    }
    println!();
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    if args.sweep {
        // Sweep mode: iterate models × modes, print matrix
        let all_model_variants: Vec<ModelFamily> = vec![
            ModelFamily::Qwen3_30bA3b,
            ModelFamily::Qwen3_14b,
            ModelFamily::Qwen3_8b,
            ModelFamily::Qwen3_8bQ8,
            ModelFamily::Qwen2_7b,
            ModelFamily::Qwen2_7bQ8,
            ModelFamily::Qwen2_0_5b,
            ModelFamily::Qwen2_0_5bF16,
            ModelFamily::Llama3_2_3b,
            ModelFamily::Llama3_2_3bF16,
        ];

        let models: Vec<ModelFamily> = if args.sweep_models.is_empty() {
            all_model_variants
        } else {
            args.sweep_models.clone()
        };

        let modes: Vec<CompressionArg> = if args.sweep_modes.is_empty() {
            ALL_SWEEP_MODES.to_vec()
        } else {
            args.sweep_modes.clone()
        };

        let stride = args.stride.unwrap_or(args.context_size);

        println!(
            "=== PPL Sweep: {} models × {} modes ===",
            models.len(),
            modes.len()
        );
        println!("Context={}, Stride={}", args.context_size, stride);

        // Collect results into a matrix indexed by (model_idx, mode_idx)
        let mut matrix: std::collections::HashMap<(usize, usize), String> =
            std::collections::HashMap::new();
        let mode_labels: Vec<&str> = modes.iter().map(|m| m.label()).collect();

        if args.sweep_contexts.is_empty() {
            for (mi, family) in models.iter().enumerate() {
                println!("\n===== {} =====", family.name());

                // Load tokens for this model's tokenizer (tokenizer may differ per model family)
                let tokens = match load_tokens(
                    &args.text_file,
                    args.tokenizer.as_deref(),
                    family.tokenizer_repo(),
                    args.max_tokens,
                ) {
                    Ok(t) => {
                        println!("Tokens: {}", t.len());
                        t
                    }
                    Err(e) => {
                        println!("  FAILED to load tokens: {}", e);
                        for (ci, label) in mode_labels.iter().enumerate() {
                            matrix.insert((mi, ci), format!("ERR:{}", label));
                        }
                        continue;
                    }
                };

                let results = match run_model_all_modes(
                    family,
                    None,
                    &tokens,
                    &modes,
                    args.context_size,
                    stride,
                    &device,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        println!("  FAILED: {}", e);
                        for (ci, _) in mode_labels.iter().enumerate() {
                            matrix.insert((mi, ci), "FAIL".to_string());
                        }
                        continue;
                    }
                };

                // Map results back to mode indices
                for (ci, mode_arg) in modes.iter().enumerate() {
                    let label = mode_arg.label();
                    if let Some((_, ppl)) = results.iter().find(|(l, _)| l == label) {
                        matrix.insert((mi, ci), ppl.clone());
                    }
                }
            }
        } // end if sweep_contexts.is_empty() / for models

        // ── Context sweep (context×mode table for a single model) ──────────────────
        if !args.sweep_contexts.is_empty() {
            if models.len() != 1 {
                anyhow::bail!("--sweep-contexts requires exactly one model in --sweep-models");
            }
            let family = &models[0];
            println!(
                "\n=== Context Sweep: {} contexts × {} modes for {} ===",
                args.sweep_contexts.len(),
                modes.len(),
                family.name()
            );

            let tokens = load_tokens(
                &args.text_file,
                args.tokenizer.as_deref(),
                family.tokenizer_repo(),
                args.max_tokens,
            )?;
            println!("Tokens: {}", tokens.len());

            let model_path = resolve_model_path(None, family.gguf_repo(), family.gguf_filename())?;

            let mode_labels: Vec<&str> = modes.iter().map(|m| m.label()).collect();

            // ctx_matrix[(ctx_idx, mode_idx)] = ppl string
            let mut ctx_matrix: std::collections::HashMap<(usize, usize), String> =
                std::collections::HashMap::new();

            // Iterate modes-outer so each mode gets a fresh BatchedModel (avoids
            // pinned-arena accumulation / OOM when running many small-context chunks).
            for (ci, mode_arg) in modes.iter().enumerate() {
                println!("\n  === mode={} ===", mode_arg.label());
                if matches!(mode_arg, CompressionArg::None) {
                    for (ri, &ctx) in args.sweep_contexts.iter().enumerate() {
                        let baseline_model = Model::load(&family.arch(), &model_path, &device)?;
                        match eval_baseline(&baseline_model, &tokens, ctx, ctx, &device) {
                            Ok((nll, count)) => {
                                let ppl = (nll / count as f64).exp();
                                ctx_matrix.insert((ri, ci), format!("{:.2}", ppl));
                            }
                            Err(e) => {
                                ctx_matrix.insert((ri, ci), format!("ERR:{}", e));
                            }
                        }
                    }
                    continue;
                }
                // Load a fresh BatchedModel for each (mode, context) pair to avoid
                // pinned-arena accumulation across different context sizes.
                let mode = mode_arg.to_inference_mode().unwrap();
                for (ri, &ctx) in args.sweep_contexts.iter().enumerate() {
                    println!("  ctx={} mode={}", ctx, mode_arg.label());
                    let model = Model::load(&family.arch(), &model_path, &device)?;
                    let batched = BatchedModel::from_model(model, &device)?;
                    match eval_compressed(&batched, mode, &tokens, Some(ctx), ctx, &device) {
                        Ok((nll, count)) => {
                            let ppl = (nll / count as f64).exp();
                            ctx_matrix.insert((ri, ci), format!("{:.2}", ppl));
                        }
                        Err(e) => {
                            ctx_matrix.insert((ri, ci), format!("ERR:{}", e));
                        }
                    }
                }
            }

            // Print context×mode table
            let ctx_col_w = 8usize;
            let mode_col_w = mode_labels
                .iter()
                .map(|l| l.len())
                .max()
                .unwrap_or(6)
                .max(7);
            let header: String =
                std::iter::once(format!("{:<width$}", "Context", width = ctx_col_w))
                    .chain(
                        mode_labels
                            .iter()
                            .map(|l| format!("{:>width$}", l, width = mode_col_w)),
                    )
                    .collect::<Vec<_>>()
                    .join(" | ");
            println!("\n{}", header);
            println!("{}", "-".repeat(header.len()));
            for (ri, &ctx) in args.sweep_contexts.iter().enumerate() {
                let row: String = std::iter::once(format!("{:<width$}", ctx, width = ctx_col_w))
                    .chain(modes.iter().enumerate().map(|(ci, _)| {
                        let val = ctx_matrix
                            .get(&(ri, ci))
                            .cloned()
                            .unwrap_or_else(|| "-".to_string());
                        format!("{:>width$}", val, width = mode_col_w)
                    }))
                    .collect::<Vec<_>>()
                    .join(" | ");
                println!("{}", row);
            }
            println!();

            // Save CSV
            let csv_file = format!("ppl_ctx_sweep_{}.csv", family.name().replace(' ', "_"));
            let mut csv = format!("Context,{}\n", mode_labels.join(","));
            for (ri, &ctx) in args.sweep_contexts.iter().enumerate() {
                let row_vals: Vec<String> = (0..modes.len())
                    .map(|ci| {
                        ctx_matrix
                            .get(&(ri, ci))
                            .cloned()
                            .unwrap_or_else(|| "-".to_string())
                    })
                    .collect();
                csv.push_str(&format!("{},{}\n", ctx, row_vals.join(",")));
            }
            std::fs::write(&csv_file, &csv)?;
            println!("Saved context sweep to {}", csv_file);

            return Ok(());
        }

        print_matrix(&models, &modes, &matrix);

        // Also save to CSV
        let csv_file = format!("ppl_sweep_ctx{}.csv", args.context_size);
        let mut csv = format!("Model,{}\n", mode_labels.join(","));
        for (mi, family) in models.iter().enumerate() {
            let row_vals: Vec<String> = (0..modes.len())
                .map(|ci| {
                    matrix
                        .get(&(mi, ci))
                        .cloned()
                        .unwrap_or_else(|| "-".to_string())
                })
                .collect();
            csv.push_str(&format!("{},{}\n", family.name(), row_vals.join(",")));
        }
        std::fs::write(&csv_file, csv)?;
        println!("Saved matrix to {}", csv_file);

        return Ok(());
    }

    // Single-model mode
    let model_family = args
        .model
        .as_ref()
        .expect("--model required when not using --sweep");
    let compression_mode = args.compression.to_inference_mode();
    let compression_label = args.compression.label();

    println!("=== WikiText-2 Perplexity Evaluation ===");
    println!("Model:        {}", model_family.name());
    println!("Device:       {:?}", device);
    println!("Context size: {}", args.context_size);
    println!("Compression:  {}", compression_label);

    let tokens = load_tokens(
        &args.text_file,
        args.tokenizer.as_deref(),
        model_family.tokenizer_repo(),
        args.max_tokens,
    )?;
    println!("Total tokens: {}", tokens.len());

    let model_path = resolve_model_path(
        args.model_file.as_deref(),
        model_family.gguf_repo(),
        model_family.gguf_filename(),
    )?;
    println!("Loading model from {:?}...", model_path);
    let model = Model::load(&model_family.arch(), &model_path, &device)?;
    println!("Model loaded.");

    let stride = args.stride.unwrap_or(args.context_size);

    let (total_nll, total_count) = if let Some(mode) = compression_mode {
        if !model_family.supports_kv_compression() {
            anyhow::bail!(
                "Model {} has head_dim=64 and does not support KV cache compression. \
                 Use --compression none.",
                model_family.name()
            );
        }
        println!(
            "Setting up paged attention for compression mode {:?}...",
            mode
        );
        let batched = BatchedModel::from_model(model, &device)?;
        eval_compressed(
            &batched,
            mode,
            &tokens,
            Some(args.context_size),
            stride,
            &device,
        )?
    } else {
        eval_baseline(&model, &tokens, args.context_size, stride, &device)?
    };

    let final_ppl = (total_nll / total_count as f64).exp();
    let avg_nll = total_nll / total_count as f64;

    println!("\n=== Results ===");
    println!("Model:          {}", model_family.name());
    println!("Compression:    {}", compression_label);
    println!("Tokens scored:  {}", total_count);
    println!("Avg NLL:        {:.4}", avg_nll);
    println!("Perplexity:     {:.2}", final_ppl);
    println!();

    Ok(())
}
