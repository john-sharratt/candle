/// Example: Using sub_at_indices_mut() for LLM repetition penalty
///
/// This demonstrates the performance-optimized pattern for applying
/// repetition penalties during token sampling.
use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    // Simulate LLM logits (batch_size=4, vocab_size=151936)
    let mut logits = Tensor::randn(0f32, 1f32, (4, 151936), &device)?;

    // Tokens to penalize (e.g., from generation history)
    let penalty_tokens = vec![42, 100, 256, 512, 1024, 2048];
    let penalty_value = 5.0;

    println!(
        "Applying repetition penalty to {} tokens...",
        penalty_tokens.len()
    );

    // ✅ FAST: In-place mutation (no tensor clone)
    // ~22µs on CUDA for 150K vocab
    logits.sub_at_indices_mut(&penalty_tokens, penalty_value)?;

    println!("Penalty applied successfully!");

    // For comparison, the immutable version would be:
    // let penalized_logits = logits.sub_at_indices(&penalty_tokens, penalty_value)?;
    // This would take ~55µs due to tensor cloning

    Ok(())
}

// Pattern for Battle Cities sampler:
fn apply_repetition_penalty_fast(
    mut logits: Tensor,
    penalty_tokens: &[u32],
    penalty: f32,
) -> Result<Tensor> {
    // Use mutable API for performance
    logits.sub_at_indices_mut(penalty_tokens, penalty)?;
    Ok(logits)
}

// If you need to preserve the original logits:
fn apply_repetition_penalty_preserve_original(
    logits: &Tensor,
    penalty_tokens: &[u32],
    penalty: f32,
) -> Result<Tensor> {
    // Clone explicitly, then use mutable API
    let mut penalized = logits.clone();
    penalized.sub_at_indices_mut(penalty_tokens, penalty)?;
    Ok(penalized)
}
