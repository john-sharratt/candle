# GPU Multinomial Sampling Implementation Status

## Problem Identified ✅

The current implementation is **15x SLOWER** than CPU because:

1. **Kernel Launch Overhead**: Launching CUDA kernel 10,000 times
   - Each launch: ~10-20μs overhead
   - Total: 10,000 × 20μs = 200ms overhead
   
2. **No Parallelism**: Single-threaded kernel
   - Only using 1 of thousands of GPU cores
   - Not leveraging GPU's strength

3. **Shared Memory Limitation**: 48KB limit
   - QWEN3 (32K vocab) needs 384KB
   - Current kernel fails for large vocabularies

## Solution Architecture ✅

Based on PyTorch's MultinomialKernel.cu:

### 1. Batched Processing
```rust
// OLD (wrong): Call kernel N times
for i in 0..N {
    kernel_launch(logits[i]) // 20μs overhead each
}

// NEW (correct): Call kernel ONCE
kernel_launch_batch(logits, N) // 20μs overhead total
// grid=(N, 1, 1) - N blocks process N samples in parallel
```

### 2. Pre-generate Random Numbers on GPU
```rust
// Generate all random samples at once on GPU
let random_samples = Tensor::rand(0.0, 1.0, (batch_size,), device)?;

// Pass to kernel - no CPU involvement!
multinomial_batch_kernel(logits, random_samples, output);
```

### 3. Parallel Execution within Block
- Each block: 256 threads
- Parallel softmax computation (reduction)
- Parallel prefix sum (scan)
- Binary search for sampling

## Implementation Status

### ✅ Completed
1. Created `multinomial_batch.cu` kernel
   - Parallel softmax with block reduction
   - Cumulative distribution (prefix sum)
   - Binary search for sampling
   - Follows PyTorch's architecture

2. Identified root causes:
   - Kernel launch overhead (per-sample calls)
   - Shared memory limitation
   - Single-threaded execution

### ⏳ TODO
1. **Rust API Changes**
   ```rust
   // Need batched API
   fn sample_multinomial_batch(
       logits: &Tensor,  // [batch_size, vocab_size]
       temperature: f32,
       top_k: Option<usize>,
       top_p: Option<f64>,
       seed: u64
   ) -> Result<Tensor>  // [batch_size] of u32
   ```

2. **Random Number Generation**
   - Pre-generate random samples on GPU
   - Use cuRAND or Tensor::rand()
   - No CPU transfers!

3. **Large Vocabulary Support**
   - For vocab > 4K: use global memory
   - Query device capabilities: `cudaDeviceProp.sharedMemPerBlock`
   - Conditional path selection

4. **Top-K/Top-P Filtering**
   - Current batched kernel has simplified filtering
   - Need proper parallel sorting/filtering
   - Can use GPU sorting primitives

## Performance Expectations

Based on PyTorch benchmarks:

| Approach | Time (10K samples) | Speedup |
|----------|-------------------|---------|
| Current (per-sample kernel) | ~18 seconds | 0.07x (15x slower!) |
| CPU baseline | ~1.2 seconds | 1.0x |
| **Target (batched GPU)** | **~20-50ms** | **24-60x faster** |

### Breakdown:
- Random generation: ~5ms (10,000 floats on GPU)
- Kernel execution: ~15-40ms (depends on vocab size)
- Memory transfers: 0ms (all on GPU!)

## References

- PyTorch Implementation: `aten/src/ATen/native/cuda/MultinomialKernel.cu`
- Key functions:
  - `sampleMultinomialOnce`: Optimized for single sample
  - `sampleMultinomialWithReplacement`: Batched sampling
  - Uses `curand_uniform4` for random generation
  - Binary search for efficient sampling

## Next Steps

Priority 1 (Critical):
1. Implement batched Rust API
2. Integrate random number generation
3. Test with 10K batch

Priority 2 (Important):
1. Add global memory fallback for large vocabs
2. Implement proper top-k/top-p filtering
3. Performance benchmarking

Priority 3 (Nice to have):
1. Support for different sampling strategies
2. Multi-GPU support
3. Kernel fusion optimizations
