// Run with: cargo run --release --features cuda --example test_softmax --package candle-nn
use candle::{DType, Device, IndexOp, Tensor};

fn main() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;

    println!("Testing softmax on CUDA...");

    // Create a simple tensor
    let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device)?;
    println!("Input tensor:\n{}", x);

    // Apply softmax using candle_nn
    let result = candle_nn::ops::softmax(&x, 1)?;
    println!("Softmax result:\n{}", result);

    // Compute expected result on CPU for verification
    let x_cpu = Tensor::new(
        &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        &Device::Cpu,
    )?;
    let expected = candle_nn::ops::softmax(&x_cpu, 1)?;
    println!("Expected (CPU):\n{}", expected);

    // Compare
    let diff = (result.to_device(&Device::Cpu)? - &expected)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    println!("Total absolute difference: {}", diff);

    if diff < 1e-5 {
        println!("✓ Softmax test PASSED");
    } else {
        println!("✗ Softmax test FAILED");
    }

    println!("\nTesting RmsNorm on CUDA...");

    // Test rmsnorm
    let x = Tensor::rand(-1.0f32, 1.0, (2, 4), &device)?;
    let alpha = Tensor::ones((4,), DType::F32, &device)?;
    let eps = 1e-5f32;

    println!("Input tensor:\n{}", x);
    let result = candle_nn::ops::rms_norm(&x, &alpha, eps)?;
    println!("RmsNorm result:\n{}", result);

    // Compute expected on CPU
    let x_cpu = x.to_device(&Device::Cpu)?;
    let alpha_cpu = Tensor::ones((4,), DType::F32, &Device::Cpu)?;
    let expected = candle_nn::ops::rms_norm(&x_cpu, &alpha_cpu, eps)?;
    println!("Expected (CPU):\n{}", expected);

    let diff = (result.to_device(&Device::Cpu)? - &expected)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    println!("Total absolute difference: {}", diff);

    if diff < 1e-4 {
        println!("✓ RmsNorm test PASSED");
    } else {
        println!("✗ RmsNorm test FAILED");
    }

    // Test RoPE (non-interleaved) embedding
    println!("\nTesting RoPE (non-interleaved) on CUDA...");

    let seq_len = 4;
    let head_dim = 8;
    let b_sz = 2;
    let n_head = 3;

    // Create input tensor [batch, seq_len, n_heads, head_dim]
    let src = Tensor::rand(-1.0f32, 1.0, (b_sz, seq_len, n_head, head_dim), &device)?;

    // Create cos/sin tables
    let cos = Tensor::rand(-1.0f32, 1.0, (seq_len, head_dim / 2), &device)?;
    let sin = Tensor::rand(-1.0f32, 1.0, (seq_len, head_dim / 2), &device)?;

    println!("RoPE input shape: {:?}", src.shape());

    // Apply rope using candle_nn
    let result = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    println!("RoPE result shape: {:?}", result.shape());

    // Compute on CPU
    let src_cpu = src.to_device(&Device::Cpu)?;
    let cos_cpu = cos.to_device(&Device::Cpu)?;
    let sin_cpu = sin.to_device(&Device::Cpu)?;
    let expected = candle_nn::rotary_emb::rope(&src_cpu, &cos_cpu, &sin_cpu)?;

    let result_cpu = result.to_device(&Device::Cpu)?;
    let diff = (&result_cpu - &expected)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    println!("Total absolute difference: {}", diff);

    if diff < 1e-3 {
        println!("✓ RoPE (non-interleaved) test PASSED");
    } else {
        println!("✗ RoPE (non-interleaved) test FAILED (diff = {})", diff);
        // Print some values for debugging
        println!("GPU result sample:\n{}", result_cpu.i((0, 0, 0, ..))?);
        println!("CPU result sample:\n{}", expected.i((0, 0, 0, ..))?);
    }

    // Test RoPE_I (interleaved) embedding - this is what quantized llama uses
    println!("\nTesting RoPE_I (interleaved) on CUDA...");

    let seq_len = 4;
    let head_dim = 8;
    let b_sz = 2;
    let n_head = 3;

    // Create input tensor [batch, n_heads, seq_len, head_dim]
    let src = Tensor::rand(-1.0f32, 1.0, (b_sz, n_head, seq_len, head_dim), &device)?;

    // Create cos/sin tables [seq_len, head_dim / 2]
    let cos = Tensor::rand(-1.0f32, 1.0, (seq_len, head_dim / 2), &device)?;
    let sin = Tensor::rand(-1.0f32, 1.0, (seq_len, head_dim / 2), &device)?;

    println!("RoPE_I input shape: {:?}", src.shape());

    // Apply rope_i using candle_nn
    let result = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    println!("RoPE_I result shape: {:?}", result.shape());

    // Compute on CPU
    let src_cpu = src.to_device(&Device::Cpu)?;
    let cos_cpu = cos.to_device(&Device::Cpu)?;
    let sin_cpu = sin.to_device(&Device::Cpu)?;
    let expected = candle_nn::rotary_emb::rope_i(&src_cpu, &cos_cpu, &sin_cpu)?;

    let result_cpu = result.to_device(&Device::Cpu)?;
    let diff = (&result_cpu - &expected)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    println!("Total absolute difference: {}", diff);

    if diff < 1e-3 {
        println!("✓ RoPE_I (interleaved) test PASSED");
    } else {
        println!("✗ RoPE_I (interleaved) test FAILED (diff = {})", diff);
        // Print some values for debugging
        println!("GPU result sample:\n{}", result_cpu.i((0, 0, 0, ..))?);
        println!("CPU result sample:\n{}", expected.i((0, 0, 0, ..))?);
    }

    // Test argmax on GPU
    println!("\nTesting argmax on CUDA...");

    // Create test logits [batch=2, vocab=10]
    let logits = Tensor::new(
        &[
            [0.1f32, 0.2, 0.9, 0.1, 0.0, 0.3, 0.1, 0.4, 0.2, 0.1], // max at index 2
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95, 0.1],   // max at index 8
        ],
        &device,
    )?;

    let argmax_result = logits.argmax(1)?;
    let argmax_cpu = argmax_result.to_vec1::<u32>()?;
    println!("Argmax result: {:?}", argmax_cpu);

    if argmax_cpu == vec![2, 8] {
        println!("✓ Argmax test PASSED");
    } else {
        println!("✗ Argmax test FAILED (expected [2, 8])");
    }

    Ok(())
}
