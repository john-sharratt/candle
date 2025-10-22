use candle_core::{DType, Tensor};
use std::time::{Duration, Instant};

use crate::{generation::GenerationFactory, model::ModelError};

/// Result of token sampling with detailed timing breakdown
pub struct SampleTokenResult {
    pub next_token: u32,
    pub timings: SampleTokenTimings,
}

/// Detailed timing breakdown for generating new tokens
#[derive(Debug, Clone)]
pub struct SampleTokenTimings {
    pub input: Duration,
    pub forward: Duration,
    pub next_token: NextTokenTimings,
    pub total: Duration,
}

/// Sample a single token with detailed timing breakdown - OPTIMIZED VERSION
///
/// Key optimizations:
/// 1. Only ONE synchronization at the very end (not 5!)
/// 2. Removed unnecessary dtype conversion (Q5 outputs F32 already)
/// 3. Only pass the LAST token for continuation (not all tokens)
pub(super) fn forward_and_sample_token(
    tokens: &[u32],
    factory: &mut GenerationFactory,
    sync: bool,
) -> Result<SampleTokenResult, ModelError> {
    let generate_start = Instant::now();

    // Input tensor creation - OPTIMIZATION: Only use last token if this is continuation
    let input_start = Instant::now();
    let input_tokens = if factory.current_tokens.is_empty() {
        // First token: use all tokens (prompt)
        tokens
    } else {
        // Continuation: only use the NEW token (last one)
        &tokens[tokens.len() - 1..]
    };
    
    let input = Tensor::new(input_tokens, &factory.device)?.unsqueeze(0)?;
    let input_time = input_start.elapsed();

    // Forward pass - NO SYNC (let GPU work asynchronously)
    let forward_start = Instant::now();
    let logits = factory.forward(&input, factory.current_tokens.len())?;
    let forward_time = forward_start.elapsed();

    // Grab the next token and timings
    let next_token = sample_token(logits, factory, sync)?;

    let total_time = generate_start.elapsed();
    Ok(SampleTokenResult {
        next_token: next_token.next_token,
        timings: SampleTokenTimings {
            forward: forward_time,
            input: input_time,
            next_token: next_token.timings,
            total: total_time,
        },
    })
}

/// Result of token sampling with detailed timing breakdown
pub struct NextTokenResult {
    pub next_token: u32,
    pub timings: NextTokenTimings,
}

/// Detailed timing breakdown for token sampling
#[derive(Debug, Clone)]
pub struct NextTokenTimings {
    pub postprocess: Duration,
    pub penalty: Duration,
    pub sample: Duration,
    pub transfer: Duration,
}

/// Sample the next token from logits - OPTIMIZED VERSION
///
/// Key optimizations:
/// 1. Removed unnecessary to_dtype(F32) conversion (Q5 already outputs F32)
/// 2. Only ONE synchronization at the end before transfer
/// 3. Penalties and sampling execute in pipeline without stalls
pub fn sample_token(
    logits: Tensor,
    factory: &mut GenerationFactory,
    sync: bool,
) -> Result<NextTokenResult, ModelError> {
    // Logits post-processing - OPTIMIZATION: Removed to_dtype(F32), Q5 already outputs F32
    let postprocess_start = Instant::now();
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let postprocess_time = postprocess_start.elapsed();

    // Apply penalties - NO SYNC (let GPU work asynchronously)
    let penalty_start = Instant::now();
    let penalized_logits = factory.apply_penalties(&logits)?;
    let penalty_time = penalty_start.elapsed();

    // Sampling - NO SYNC (kernel launches asynchronously)
    let sample_start = Instant::now();
    let next_token_tensor = penalized_logits
        .sample_multinomial(
            factory.config.temperature as f32,
            factory.config.top_k,
            factory.config.top_p,
            factory.config.seed,
        )
        .map_err(|e| ModelError::SamplingError(e.to_string()))?;
    let sample_time = sample_start.elapsed();

    // ONLY sync ONCE before CPU transfer (this is required)
    let transfer_start = Instant::now();
    if sync {
        factory.device.synchronize()?;
    }
    let next_token = next_token_tensor.to_scalar::<u32>()?;
    let transfer_time = transfer_start.elapsed();

    Ok(NextTokenResult {
        next_token,
        timings: NextTokenTimings {
            postprocess: postprocess_time,
            penalty: penalty_time,
            sample: sample_time,
            transfer: transfer_time,
        },
    })
}
