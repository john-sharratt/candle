/// Simple GPU vs CPU sampling benchmark
/// Measures actual sampling performance without loop overhead
use candle_core::{Device, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧪 GPU vs CPU Sampling Benchmark\n");

    const VOCAB_SIZE: usize = 4000;
    const ITERATIONS: usize = 1000;

    println!("📊 Configuration:");
    println!("   Vocabulary Size: {}", VOCAB_SIZE);
    println!("   Iterations: {}", ITERATIONS);
    println!("   Shared Memory: {}KB\n", (VOCAB_SIZE * 12) / 1024);

    // GPU Test
    let device = Device::new_cuda(0)?;
    let logits_gpu = Tensor::randn(0.0, 1.0, VOCAB_SIZE, &device)?;

    println!("🔥 GPU Sampling...");
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _sample = logits_gpu.sample_multinomial(0.8, None, None, 42)?;
    }
    let gpu_time = start.elapsed();
    println!("   Time: {:?}", gpu_time);
    println!(
        "   Per sample: {:.1}μs",
        gpu_time.as_micros() as f64 / ITERATIONS as f64
    );

    // CPU Test
    let logits_cpu = logits_gpu.to_device(&Device::Cpu)?;

    println!("\n💻 CPU Sampling...");
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _sample = logits_cpu.sample_multinomial(0.8, None, None, 42)?;
    }
    let cpu_time = start.elapsed();
    println!("   Time: {:?}", cpu_time);
    println!(
        "   Per sample: {:.1}μs",
        cpu_time.as_micros() as f64 / ITERATIONS as f64
    );

    // Results
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("\n🚀 Results:");
    println!("   GPU: {:?} total", gpu_time);
    println!("   CPU: {:?} total", cpu_time);
    println!("   Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("\n✅ GPU is FASTER! 🎉");
    } else {
        println!(
            "\n⚠️  GPU is slower (kernel launch overhead dominates for {} iterations)",
            ITERATIONS
        );
        println!("   Note: GPU excels at batched processing, not individual launches");
    }

    Ok(())
}
