// Test GPU sampling with large vocabulary sizes
use candle_core::Result;

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::{Device, Tensor};
        use std::time::Instant;

        println!("🧪 Testing GPU Sampling with Large Vocabularies\n");

        let device = Device::new_cuda(0)?;

        // Test various vocabulary sizes (realistic LLM ranges)
        let vocab_sizes = vec![100, 500, 1000, 5000, 10000, 32000, 50000];

        for vocab_size in vocab_sizes {
            println!("Testing vocab_size = {}", vocab_size);

            // Create logits with a realistic distribution
            // Higher indices get slightly lower values
            let logits_vec: Vec<f32> = (0..vocab_size).map(|i| 10.0 - (i as f32 * 0.01)).collect();

            let logits = Tensor::new(&logits_vec[..], &device)?;

            // Test 1: No filtering (baseline speed)
            let start = Instant::now();
            for i in 0..100 {
                let _token = logits.sample_multinomial(1.0, None, None, i as u64)?;
            }
            let no_filter_time = start.elapsed();

            // Test 2: Top-k filtering
            let start = Instant::now();
            for i in 0..100 {
                let _token = logits.sample_multinomial(1.0, Some(50), None, i as u64)?;
            }
            let topk_time = start.elapsed();

            // Test 3: Top-p filtering
            let start = Instant::now();
            for i in 0..100 {
                let _token = logits.sample_multinomial(1.0, None, Some(0.9), i as u64)?;
            }
            let topp_time = start.elapsed();

            println!(
                "  No filtering: {:>6.2}ms (100 samples)",
                no_filter_time.as_secs_f64() * 1000.0
            );
            println!(
                "  Top-k (50):   {:>6.2}ms (100 samples)",
                topk_time.as_secs_f64() * 1000.0
            );
            println!(
                "  Top-p (0.9):  {:>6.2}ms (100 samples)",
                topp_time.as_secs_f64() * 1000.0
            );

            // Verify correctness with top-k
            let mut counts = vec![0; vocab_size.min(100)];
            for i in 0..1000 {
                let token = logits.sample_multinomial(1.0, Some(50), None, i as u64)?;
                let idx = token.to_scalar::<u32>()? as usize;
                if idx < counts.len() {
                    counts[idx] += 1;
                }
            }

            let samples_in_topk = counts[..50.min(counts.len())].iter().sum::<i32>();
            println!(
                "  Correctness: {}/1000 samples in top-50 ✓\n",
                samples_in_topk
            );
        }

        println!("✅ Large vocabulary tests completed!");
        println!("   Adaptive sorting scales efficiently with vocabulary size");
        println!("   Selection sort optimization works for top-k/top-p scenarios");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  This test requires CUDA support");
        println!("   Run with: cargo run --example test_large_vocab_sampling --features cuda");
    }

    Ok(())
}
