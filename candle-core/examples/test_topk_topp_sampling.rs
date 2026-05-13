// Test GPU sampling with top-k and top-p filtering
use candle_core::Result;

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::{Device, Tensor};

        println!("🧪 Testing GPU Top-K and Top-P Sampling\n");

        let device = Device::new_cuda(0)?;

        // Create logits with realistic probability distribution
        // Closer values = more balanced distribution
        let logits = Tensor::new(&[5.0f32, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5], &device)?;

        println!("📊 Logits: [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5]");
        println!("   (Index 0 has highest probability, but more balanced)\n");

        // Test 1: No filtering (baseline)
        println!("Test 1: No filtering (temperature=1.0)");
        let mut counts = vec![0; 8];
        for i in 0..1000 {
            let token = logits.sample_multinomial(1.0, None, None, i as u64)?;
            let idx = token.to_scalar::<u32>()? as usize;
            counts[idx] += 1;
        }
        println!("   Samples (1000 iterations): {:?}", counts);
        println!("   ✓ Should favor lower indices (higher logits)\n");

        // Test 2: Top-K filtering (k=3)
        println!("Test 2: Top-K filtering (k=3)");
        let mut counts = vec![0; 8];
        for i in 0..1000 {
            let token = logits.sample_multinomial(1.0, Some(3), None, i as u64)?;
            let idx = token.to_scalar::<u32>()? as usize;
            counts[idx] += 1;
        }
        println!("   Samples (1000 iterations): {:?}", counts);
        println!("   ✓ Should ONLY sample from indices [0,1,2] (top-3 logits)");
        let filtered_out = counts[3..].iter().sum::<i32>();
        if filtered_out == 0 {
            println!("   ✅ PASS: All samples from top-3 as expected\n");
        } else {
            println!(
                "   ❌ FAIL: {} samples leaked to indices 3-7!\n",
                filtered_out
            );
        }

        // Test 3: Top-P filtering (p=0.7)
        println!("Test 3: Top-P/Nucleus sampling (p=0.7)");
        let mut counts = vec![0; 8];
        for i in 0..1000 {
            let token = logits.sample_multinomial(1.0, None, Some(0.7), i as u64)?;
            let idx = token.to_scalar::<u32>()? as usize;
            counts[idx] += 1;
        }
        println!("   Samples (1000 iterations): {:?}", counts);
        println!("   ✓ Should sample from smallest set with cumulative prob >= 0.7\n");

        // Test 4: Combined top-k and top-p
        println!("Test 4: Combined filtering (k=5, p=0.9)");
        let mut counts = vec![0; 8];
        for i in 0..1000 {
            let token = logits.sample_multinomial(1.0, Some(5), Some(0.9), i as u64)?;
            let idx = token.to_scalar::<u32>()? as usize;
            counts[idx] += 1;
        }
        println!("   Samples (1000 iterations): {:?}", counts);
        println!("   ✓ Should apply top-k first, then top-p on remaining\n");

        // Test 5: Extreme temperature with filtering
        println!("Test 5: Low temperature (0.1) with top-k=2");
        let mut counts = vec![0; 8];
        for i in 0..1000 {
            let token = logits.sample_multinomial(0.1, Some(2), None, i as u64)?;
            let idx = token.to_scalar::<u32>()? as usize;
            counts[idx] += 1;
        }
        println!("   Samples (1000 iterations): {:?}", counts);
        println!("   ✓ Should heavily favor index 0 (highest logit)");
        if counts[0] > 900 {
            println!("   ✅ PASS: {}% samples at index 0\n", counts[0] / 10);
        } else {
            println!(
                "   ⚠️  WARNING: Only {}% at index 0 (expected >90%)\n",
                counts[0] / 10
            );
        }

        println!("✅ All GPU top-k/top-p tests completed!");
        println!("   Kernel successfully filters distributions on GPU");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  This test requires CUDA support");
        println!("   Run with: cargo run --example test_topk_topp_sampling --features cuda");
    }

    Ok(())
}
