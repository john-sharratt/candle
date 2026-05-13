/// Example demonstrating the performance difference between implicit and explicit GPU→CPU transfers
/// in sampling operations.
#[cfg(test)]
mod performance_demo {
    use crate::{Device, Result, Tensor};

    #[test]
    fn demonstrate_sampling_performance() -> Result<()> {
        println!("\n🚀 GPU-Native Sampling Performance Demonstration\n");

        // Create large logits tensor for performance testing
        let logits_data: Vec<f32> = (0..10000).map(|i| (i as f32) * 0.001).collect();

        #[cfg(feature = "cuda")]
        {
            if crate::utils::cuda_is_available() {
                let cuda_device = Device::new_cuda(0)?;
                let logits_gpu =
                    Tensor::from_vec(logits_data.clone(), logits_data.len(), &cuda_device)?;

                println!("🔥 CUDA Device Performance Test:");
                println!("   Logits tensor shape: {:?}", logits_gpu.shape());
                println!("   Logits tensor device: {:?}", logits_gpu.device());

                // Test GPU-native sampling
                let token = logits_gpu.sample_multinomial(0.8, Some(50), Some(0.9), 42)?;
                println!(
                    "   ✅ GPU-native sampling result: device={:?}, token={:?}",
                    token.device(),
                    token.to_scalar::<u32>()?
                );

                return Ok(());
            }
        }

        #[cfg(feature = "metal")]
        {
            if crate::utils::metal_is_available() {
                let metal_device = Device::new_metal(0)?;
                let logits_gpu =
                    Tensor::from_vec(logits_data.clone(), logits_data.len(), &metal_device)?;

                println!("🍎 Metal Device Performance Test:");
                println!("   Logits tensor shape: {:?}", logits_gpu.shape());
                println!("   Logits tensor device: {:?}", logits_gpu.device());

                // Test GPU-native sampling
                let token = logits_gpu.sample_multinomial(0.8, Some(50), Some(0.9), 42)?;
                println!(
                    "   ✅ GPU-native sampling result: device={:?}, token={:?}",
                    token.device(),
                    token.to_scalar::<u32>()?
                );

                return Ok(());
            }
        }

        // Fallback to CPU demonstration
        let cpu_device = Device::Cpu;
        let logits_cpu = Tensor::from_vec(logits_data, 10000, &cpu_device)?;

        println!("💻 CPU Device Performance Test:");
        println!("   Logits tensor shape: {:?}", logits_cpu.shape());
        println!("   Logits tensor device: {:?}", logits_cpu.device());

        // Test CPU sampling
        let token = logits_cpu.sample_multinomial(0.8, Some(50), Some(0.9), 42)?;
        println!("   ✅ CPU sampling result: token={}", token);

        println!("\n📊 Performance Notes:");
        println!("   • GPU-native sampling with parallel reduction (256 threads)");
        println!("   • Temperature scaling fully implemented on GPU");
        println!("   • Top-k/top-p parameters accepted but not yet implemented in kernel");
        println!("   • Final sampling is serial (only thread 0) - could be optimized");
        println!("   • This method replaces inefficient LogitsProcessor.sample()");

        Ok(())
    }

    #[test]
    fn compare_with_explicit_transfer() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;

        println!("\n🔄 Transfer Pattern Comparison\n");

        // Simulate current LogitsProcessor approach (implicit transfer)
        println!("❌ Old approach (LogitsProcessor with implicit transfer):");
        println!("   1. logits.to_device(&Device::Cpu)? // Hidden inside sample()");
        println!("   2. CPU-based sampling");
        println!("   3. Return result");

        // Our new approach (explicit control)
        println!("\n✅ New approach (explicit device management):");
        println!("   1. Keep logits on GPU as long as possible");
        println!("   2. logits.sample_multinomial() // GPU→CPU only when needed");
        println!("   3. Return result on same device as input");

        let token = logits.sample_multinomial(1.0, None, None, 42)?;
        println!("   Sample result: {}", token);

        println!("\n🎯 Key Benefits:");
        println!("   • Explicit device control");
        println!("   • Consistent device placement");
        println!("   • Future GPU kernel compatibility");
        println!("   • No hidden performance costs");

        Ok(())
    }
}

// Note: The actual GPU-native sampling implementation is in candle-core/src/sampling.rs
// and candle-kernels/src/multinomial.cu. This file only contains demonstration/test code.
