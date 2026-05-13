// Test the batched argmax CUDA kernel from candle-flash-attn
// This verifies that the custom CUDA kernel produces correct results

use candle::{Device, Result, Tensor};
use std::time::Instant;

const BATCH_SIZE: usize = 1000;
const VOCAB_SIZE: usize = 128000; // Typical LLM vocab size

fn main() -> Result<()> {
    println!("🧪 Testing Batched Argmax CUDA Kernel");
    println!("   Batch Size: {}", BATCH_SIZE);
    println!("   Vocab Size: {}", VOCAB_SIZE);
    println!();

    let device = Device::new_cuda(0)?;

    // Create test logits with known maximum positions
    // For each row, set a specific position to have the highest value
    let mut logits_data = vec![0.0f32; BATCH_SIZE * VOCAB_SIZE];
    let expected_indices: Vec<u32> = (0..BATCH_SIZE)
        .map(|i| ((i * 12345 + 7) % VOCAB_SIZE) as u32) // Deterministic "random" positions
        .collect();

    for (row, &max_idx) in expected_indices.iter().enumerate() {
        let row_start = row * VOCAB_SIZE;
        // Fill with small values
        for j in 0..VOCAB_SIZE {
            logits_data[row_start + j] = (j as f32) * 0.001 - 50.0; // Range -50 to ~78
        }
        // Set the expected max position to a high value
        logits_data[row_start + max_idx as usize] = 100.0;
    }

    let logits = Tensor::from_vec(logits_data, (BATCH_SIZE, VOCAB_SIZE), &device)?;
    println!("✅ Created test logits on GPU: {:?}", logits.shape());

    // Test standard argmax
    println!("\n🔥 Testing standard argmax...");
    let start = Instant::now();
    let standard_result = logits.argmax(1)?;
    let standard_time = start.elapsed();

    let standard_indices = standard_result.to_vec1::<u32>()?;
    println!("   Time: {:?}", standard_time);
    println!("   Output dtype: {:?}", standard_result.dtype());
    println!(
        "   First 10 results: {:?}",
        &standard_indices[..10.min(standard_indices.len())]
    );

    // Verify standard argmax results
    let mut standard_correct = 0;
    for (i, (&result, &expected)) in standard_indices
        .iter()
        .zip(expected_indices.iter())
        .enumerate()
    {
        if result == expected {
            standard_correct += 1;
        } else if i < 5 {
            println!(
                "   Mismatch at {}: got {}, expected {}",
                i, result, expected
            );
        }
    }
    println!(
        "   Accuracy: {}/{} ({:.1}%)",
        standard_correct,
        BATCH_SIZE,
        100.0 * standard_correct as f64 / BATCH_SIZE as f64
    );

    // Test batched argmax from flash-attn
    println!("\n🔥 Testing batched CUDA kernel argmax...");
    let start = Instant::now();
    let batched_result = candle_flash_attn::batched_argmax(&logits)?;
    let batched_time = start.elapsed();

    let batched_indices = batched_result.to_vec1::<u32>()?;
    println!("   Time: {:?}", batched_time);
    println!("   Output dtype: {:?}", batched_result.dtype());
    println!(
        "   First 10 results: {:?}",
        &batched_indices[..10.min(batched_indices.len())]
    );

    // Verify batched argmax results
    let mut batched_correct = 0;
    for (i, (&result, &expected)) in batched_indices
        .iter()
        .zip(expected_indices.iter())
        .enumerate()
    {
        if result == expected {
            batched_correct += 1;
        } else if i < 5 {
            println!(
                "   Mismatch at {}: got {}, expected {}",
                i, result, expected
            );
        }
    }
    println!(
        "   Accuracy: {}/{} ({:.1}%)",
        batched_correct,
        BATCH_SIZE,
        100.0 * batched_correct as f64 / BATCH_SIZE as f64
    );

    // Compare timing
    println!("\n📊 Performance Comparison:");
    println!("   Standard argmax: {:?}", standard_time);
    println!("   Batched kernel:  {:?}", batched_time);
    let speedup = standard_time.as_secs_f64() / batched_time.as_secs_f64();
    if speedup > 1.0 {
        println!("   Speedup: {:.2}x", speedup);
    } else {
        println!("   Slowdown: {:.2}x", 1.0 / speedup);
    }

    // Verify both produce same results
    println!("\n🔍 Comparing results...");
    let matches = standard_indices
        .iter()
        .zip(batched_indices.iter())
        .filter(|(a, b)| a == b)
        .count();
    if matches == BATCH_SIZE {
        println!("   ✅ All {} results match!", BATCH_SIZE);
    } else {
        println!("   ⚠️  Only {}/{} results match!", matches, BATCH_SIZE);
    }

    println!("\n✅ Test completed!");
    Ok(())
}
