use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🎯 Real-World Inference: GPU Transfer Overhead Analysis\n");
    println!("📊 Scenario: Token-by-token generation (typical LLM inference)");
    println!("   - Each step: forward pass → logits on GPU → sample 1 token → continue");
    println!("   - QWEN3: 32K vocabulary");
    println!("   - Measuring: Time per token generation step\n");

    let vocab_size = 32000;
    let iterations = 1000; // 1000 inference steps
    let temperature = 0.8;
    let top_k = Some(50);
    let top_p = Some(0.9);

    // CPU-based inference (logits stay on CPU)
    println!("💻 CPU-based Inference:");
    let cpu_device = Device::Cpu;
    let cpu_logits = Tensor::randn(0f32, 1.0, (vocab_size,), &cpu_device)?;

    let start = Instant::now();
    for i in 0..iterations {
        // Logits already on CPU, no transfer needed
        let _token = cpu_logits.sample_multinomial(temperature, top_k, top_p, i as u64)?;
    }
    let cpu_time = start.elapsed();

    println!(
        "   Total time: {}ms for {} steps",
        cpu_time.as_millis(),
        iterations
    );
    println!(
        "   Per-step:   {:.1}μs/token",
        cpu_time.as_micros() as f64 / iterations as f64
    );

    // GPU-based inference with CPU sampling (traditional approach - requires transfer)
    println!("\n🔄 GPU-based Inference with CPU Sampling (Traditional):");
    let cuda_device = Device::new_cuda(0)?;
    let gpu_logits = Tensor::randn(0f32, 1.0, (vocab_size,), &cuda_device)?;

    let start = Instant::now();
    for i in 0..iterations {
        // GPU→CPU transfer
        let cpu_logits = gpu_logits.to_device(&cpu_device)?;

        // CPU sampling
        let token = cpu_logits.sample_multinomial(temperature, top_k, top_p, i as u64)?;

        // CPU→GPU transfer (if we need token on GPU for next step)
        let _gpu_token = token.to_device(&cuda_device)?;
    }
    let gpu_cpu_time = start.elapsed();

    println!(
        "   Total time: {}ms for {} steps",
        gpu_cpu_time.as_millis(),
        iterations
    );
    println!(
        "   Per-step:   {:.1}μs/token",
        gpu_cpu_time.as_micros() as f64 / iterations as f64
    );

    // GPU-based inference with GPU sampling (GPU-native - no transfer!)
    println!("\n🚀 GPU-based Inference with GPU Sampling (GPU-Native):");

    let start = Instant::now();
    for i in 0..iterations {
        // NO transfers! Sample directly on GPU
        let _token = gpu_logits.sample_multinomial(temperature, top_k, top_p, i as u64)?;
    }
    let gpu_native_time = start.elapsed();

    println!(
        "   Total time: {}ms for {} steps",
        gpu_native_time.as_millis(),
        iterations
    );
    println!(
        "   Per-step:   {:.1}μs/token",
        gpu_native_time.as_micros() as f64 / iterations as f64
    );

    // Analysis
    println!("\n📈 Performance Analysis:");
    println!(
        "   CPU baseline:             {:.1}μs/token",
        cpu_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   GPU with transfers:       {:.1}μs/token",
        gpu_cpu_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "   GPU-native (no transfers): {:.1}μs/token",
        gpu_native_time.as_micros() as f64 / iterations as f64
    );

    let transfer_overhead = gpu_cpu_time.as_micros() as f64 / iterations as f64
        - cpu_time.as_micros() as f64 / iterations as f64;
    println!(
        "\n   Transfer overhead per token: {:.1}μs",
        transfer_overhead
    );

    let speedup_vs_traditional = gpu_cpu_time.as_nanos() as f64 / gpu_native_time.as_nanos() as f64;
    println!(
        "   GPU-native vs GPU+transfers: {:.2}x speedup ✅",
        speedup_vs_traditional
    );

    let ms_saved = gpu_cpu_time.as_millis() as i64 - gpu_native_time.as_millis() as i64;
    println!(
        "   Time saved for {} tokens: {}ms ({:.2}s)",
        iterations,
        ms_saved,
        ms_saved as f64 / 1000.0
    );

    // Typical generation scenario
    println!("\n💡 Real-World Impact:");
    let tokens_per_response = 500; // Typical response length
    let response_time_traditional =
        (gpu_cpu_time.as_micros() as f64 / iterations as f64) * tokens_per_response as f64 / 1000.0;
    let response_time_gpu_native = (gpu_native_time.as_micros() as f64 / iterations as f64)
        * tokens_per_response as f64
        / 1000.0;

    println!(
        "   Generating {} tokens with GPU model:",
        tokens_per_response
    );
    println!(
        "     Traditional (with transfers): {:.1}ms",
        response_time_traditional
    );
    println!(
        "     GPU-native (no transfers):    {:.1}ms",
        response_time_gpu_native
    );
    println!(
        "     Savings per response:         {:.1}ms",
        response_time_traditional - response_time_gpu_native
    );

    Ok(())
}
