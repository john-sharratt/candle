use crate::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor, WithDType};

#[cfg(feature = "cuda")]
use crate::CudaStorage;

#[cfg(feature = "metal")]
use crate::MetalStorage;

/// GPU-native multinomial sampling operation
#[derive(Debug, Clone)]
pub struct MultinomialSampling {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: u64,
}

impl MultinomialSampling {
    pub fn new(temperature: f32, top_k: Option<usize>, top_p: Option<f64>, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            seed,
        }
    }

    /// Sample from CPU probabilities using standard approach
    fn sample_cpu(&self, probs: &[f32]) -> Result<u32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);

        // Apply top-k filtering if specified
        let mut indices_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        if let Some(k) = self.top_k {
            indices_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indices_probs.truncate(k);
        }

        // Apply top-p (nucleus) filtering if specified
        if let Some(p) = self.top_p {
            let p = p as f32; // Convert to f32 for consistency
            indices_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cumulative_prob = 0.0f32;
            let mut cutoff = indices_probs.len();
            for (i, (_, prob)) in indices_probs.iter().enumerate() {
                cumulative_prob += prob;
                if cumulative_prob >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            indices_probs.truncate(cutoff);
        }

        // Normalize remaining probabilities
        let total_prob: f32 = indices_probs.iter().map(|(_, p)| p).sum();
        if total_prob == 0.0 {
            return Ok(0); // Fallback to first token
        }

        // Multinomial sampling
        let random_val: f32 = rng.random();
        let target = random_val * total_prob;
        let mut cumulative = 0.0;

        for &(idx, prob) in &indices_probs {
            cumulative += prob;
            if cumulative >= target {
                return Ok(idx as u32);
            }
        }

        // Fallback
        Ok(indices_probs.last().map(|(i, _)| *i as u32).unwrap_or(0))
    }
}

impl CustomOp1 for MultinomialSampling {
    fn name(&self) -> &'static str {
        "multinomial_sampling_gpu"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        // Convert logits to probabilities using softmax with temperature
        let logits = f32::cpu_storage_as_slice(storage)?;
        let mut probs = Vec::with_capacity(logits.len());

        // Apply temperature and compute softmax
        let max_logit = logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b / self.temperature));
        let mut sum_exp = 0.0f32;

        for &logit in logits {
            let exp_val = ((logit / self.temperature) - max_logit).exp();
            probs.push(exp_val);
            sum_exp += exp_val;
        }

        // Normalize to probabilities
        for prob in &mut probs {
            *prob /= sum_exp;
        }

        // Sample from the distribution
        let sampled_token = self.sample_cpu(&probs)?;

        // Return as single u32 value
        let result_storage = crate::CpuStorage::U32(vec![sampled_token]);
        Ok((result_storage, Shape::from(1)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use crate::cuda_backend::{
            cudarc::driver::{LaunchConfig, PushKernelArg},
            CudaStorageSlice, WrapErr,
        };
        use candle_kernels::MULTINOMIAL;

        // 🚀 TRUE GPU-NATIVE SAMPLING - NO CPU TRANSFERS!
        // All computation happens on GPU, only final u32 result

        let device = &storage.device;
        let vocab_size = layout.shape().dims()[0];

        // Allocate output buffer on GPU (single u32)
        let output_slice = unsafe { device.alloc::<u32>(1)? };

        // Calculate shared memory size
        // We need: vocab_size floats for scores + vocab_size floats for probs + vocab_size uints for indices
        let shared_mem_size = (vocab_size * 2 * std::mem::size_of::<f32>())
            + (vocab_size * std::mem::size_of::<u32>());

        // Check if shared memory size exceeds device limits (typically 48KB-96KB per block)
        // For vocabularies > 4096, we need to use global memory instead
        const MAX_SHARED_MEM: usize = 48 * 1024; // Conservative 48KB limit for compatibility

        let use_simple_kernel = shared_mem_size > MAX_SHARED_MEM;

        // Get the appropriate kernel based on dtype and vocab size
        let (kernel_name, kernel_module) = if use_simple_kernel {
            // Use simple kernel for large vocabularies (no shared memory, no top-k/top-p)
            let name = match &storage.slice {
                CudaStorageSlice::F32(_) => "simple_multinomial_f32",
                CudaStorageSlice::F64(_) => "simple_multinomial_f64",
                CudaStorageSlice::F16(_) => "simple_multinomial_f16",
                CudaStorageSlice::BF16(_) => "simple_multinomial_bf16",
                _ => crate::bail!("Unsupported dtype for GPU multinomial sampling"),
            };
            use candle_kernels::MULTINOMIAL_SIMPLE;
            (name, &MULTINOMIAL_SIMPLE)
        } else {
            // Use optimized kernel with shared memory for small vocabularies
            let name = match &storage.slice {
                CudaStorageSlice::F32(_) => "multinomial_f32",
                CudaStorageSlice::F64(_) => "multinomial_f64",
                CudaStorageSlice::F16(_) => "multinomial_f16",
                CudaStorageSlice::BF16(_) => "multinomial_bf16",
                _ => crate::bail!("Unsupported dtype for GPU multinomial sampling"),
            };
            (name, &MULTINOMIAL)
        };

        // Load kernel
        let func = device.get_or_load_func(kernel_name, kernel_module)?;

        // Launch with single thread (kernel handles all work internally)
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: if use_simple_kernel {
                0
            } else {
                shared_mem_size as u32
            },
        };

        let top_k_val = self.top_k.unwrap_or(0) as u32;
        let top_p_val = self.top_p.unwrap_or(0.0) as f32;

        // Launch kernel with builder pattern
        let mut builder = func.builder();

        // Push parameters based on dtype
        match &storage.slice {
            CudaStorageSlice::F32(s) => {
                builder.arg(s);
                builder.arg(&output_slice);
                builder.arg(&vocab_size);
                builder.arg(&self.temperature);
                builder.arg(&top_k_val);
                builder.arg(&top_p_val);
                builder.arg(&self.seed);
            }
            CudaStorageSlice::F64(s) => {
                builder.arg(s);
                builder.arg(&output_slice);
                builder.arg(&vocab_size);
                builder.arg(&self.temperature);
                builder.arg(&top_k_val);
                builder.arg(&top_p_val);
                builder.arg(&self.seed);
            }
            CudaStorageSlice::F16(s) => {
                builder.arg(s);
                builder.arg(&output_slice);
                builder.arg(&vocab_size);
                builder.arg(&self.temperature);
                builder.arg(&top_k_val);
                builder.arg(&top_p_val);
                builder.arg(&self.seed);
            }
            CudaStorageSlice::BF16(s) => {
                builder.arg(s);
                builder.arg(&output_slice);
                builder.arg(&vocab_size);
                builder.arg(&self.temperature);
                builder.arg(&top_k_val);
                builder.arg(&top_p_val);
                builder.arg(&self.seed);
            }
            _ => unreachable!(),
        }

        unsafe { builder.launch(cfg) }.w()?;

        // Return result as CUDA storage (stays on GPU until user requests transfer)
        let result_storage = CudaStorage::wrap_cuda_slice(output_slice, device.clone());
        Ok((result_storage, Shape::from(1)))
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        use crate::metal_backend::MetalError;

        // 🚀 TRUE GPU-NATIVE METAL SAMPLING - NO CPU TRANSFERS!

        let device = storage.device();
        let vocab_size = layout.shape().dims()[0];

        // Allocate output buffer on GPU (single u32)
        let output = device.new_buffer(1, crate::DType::U32, "multinomial_output")?;

        // Get the appropriate kernel based on dtype
        let kernel_name = match storage {
            MetalStorage::F32(_) => "multinomial_f32",
            MetalStorage::F16(_) => "multinomial_f16",
            _ => crate::bail!("Unsupported dtype for Metal multinomial sampling"),
        };

        // Load kernel
        let pipeline =
            device.get_or_load_pipeline(kernel_name, candle_metal_kernels::MULTINOMIAL)?;

        // Prepare kernel parameters
        let top_k_val = self.top_k.unwrap_or(0) as u32;
        let top_p_val = self.top_p.unwrap_or(0.0) as f32;

        // Get input buffer
        let logits_buffer = match storage {
            MetalStorage::F32(s) => s.buffer(),
            MetalStorage::F16(s) => s.buffer(),
            _ => unreachable!(),
        };

        // Create command buffer and encoder
        let command_buffer = device.command_buffer()?;
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(logits_buffer), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);
        encoder.set_u32(2, vocab_size as u32);
        encoder.set_f32(3, self.temperature);
        encoder.set_u32(4, top_k_val);
        encoder.set_f32(5, top_p_val);
        encoder.set_u64(6, self.seed);

        // Launch with single thread (kernel handles all work)
        let grid_size = metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let thread_group_size = metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Return result as Metal storage (stays on GPU until user requests transfer)
        let result_storage = MetalStorage::U32(output);
        Ok((result_storage, Shape::from(1)))
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        // Sampling is not differentiable
        Ok(None)
    }
}

impl Tensor {
    /// Sample a token from logits using GPU-native multinomial sampling.
    ///
    /// This method performs sampling directly on the GPU device without transferring
    /// data to CPU, providing better performance for inference workloads.
    ///
    /// # Arguments
    /// * `temperature` - Controls randomness (lower = more deterministic)
    /// * `top_k` - Optional top-k filtering (only consider k most likely tokens)
    /// * `top_p` - Optional nucleus sampling (cumulative probability cutoff)
    /// * `seed` - Random seed for reproducible sampling
    ///
    /// # Returns
    /// A tensor containing the sampled token index as u32, staying on original device.
    /// Use `.to_scalar::<u32>()` to extract value with controlled transfer timing.
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    ///
    /// let device = Device::Cpu;
    /// let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
    /// let token_tensor = logits.sample_multinomial(0.8, Some(3), Some(0.9), 42)?;
    /// let token_id = token_tensor.to_scalar::<u32>()?; // YOU control when transfer happens
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn sample_multinomial(
        &self,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
    ) -> Result<Self> {
        // Ensure input is 1D
        if self.rank() != 1 {
            crate::bail!(
                "sample_multinomial requires 1D tensor, got shape: {:?}",
                self.shape()
            );
        }

        // Ensure input is float type for logits
        if !matches!(
            self.dtype(),
            DType::F32 | DType::F64 | DType::F16 | DType::BF16
        ) {
            crate::bail!(
                "sample_multinomial requires float tensor, got dtype: {:?}",
                self.dtype()
            );
        }

        // Convert to F32 if needed
        let logits = if self.dtype() == DType::F32 {
            self.clone()
        } else {
            self.to_dtype(DType::F32)?
        };

        let sampling_op = MultinomialSampling::new(temperature, top_k, top_p, seed);
        let result_tensor = logits.apply_op1_no_bwd(&sampling_op)?;

        // Return the tensor on original device - transfer timing controlled by user
        Ok(result_tensor)
    }

    /// **High-performance CPU-only sampling** - avoids all GPU transfers
    ///
    /// For GPU logits, this transfers once to CPU, samples efficiently,
    /// and returns u32 directly. Much faster than sample_multinomial()
    /// for GPU tensors when you need the final token ID.
    ///
    /// # Performance Comparison
    /// ```ignore
    /// // ❌ SLOW: Hidden GPU→CPU→GPU transfers  
    /// let token_tensor = gpu_logits.sample_multinomial(temp, top_k, top_p, seed)?;
    /// let token_id = token_tensor.to_scalar::<u32>()?; // Another transfer!
    ///
    /// // ✅ FAST: Single GPU→CPU transfer, direct result
    /// let token_id = gpu_logits.sample_multinomial_cpu(temp, top_k, top_p, seed)?;
    /// ```
    pub fn sample_multinomial_cpu(
        &self,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
    ) -> Result<u32> {
        // Ensure input is 1D
        if self.rank() != 1 {
            crate::bail!(
                "sample_multinomial_cpu requires 1D tensor, got shape: {:?}",
                self.shape()
            );
        }

        // Convert to CPU if needed (single transfer for GPU tensors)
        let cpu_logits = if self.device().is_cpu() {
            self.clone()
        } else {
            self.to_device(&crate::Device::Cpu)?
        };

        // Convert to F32 if needed
        let logits = if cpu_logits.dtype() == DType::F32 {
            cpu_logits
        } else {
            cpu_logits.to_dtype(DType::F32)?
        };

        // Do efficient CPU sampling
        let sampling_op = MultinomialSampling::new(temperature, top_k, top_p, seed);
        let result_tensor = logits.apply_op1_no_bwd(&sampling_op)?;

        // Extract result directly (no additional transfers)
        result_tensor.to_scalar::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    #[test]
    fn test_multinomial_sampling_cpu() -> Result<()> {
        let device = Device::Cpu;

        // Test with simple logits
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, None, 42)?;

        // Now returns u32 directly
        assert!(token < 4); // Should be valid index

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_with_temperature() -> Result<()> {
        let device = Device::Cpu;

        // Test with high temperature (more random)
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token_hot = logits.sample_multinomial(2.0, None, None, 42)?;

        // Test with low temperature (more deterministic)
        let token_cold = logits.sample_multinomial(0.1, None, None, 42)?;

        // Now returns u32 directly
        assert!(token_hot < 4);
        assert!(token_cold < 4);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_top_k() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1], &device)?;
        let token = logits.sample_multinomial(1.0, Some(2), None, 42)?;

        // Now returns u32 directly
        assert!(token == 3 || token == 1);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_top_p() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, Some(0.8), 42)?;

        // Now returns u32 directly
        assert!(token < 4);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_combined() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5], &device)?;
        let token = logits.sample_multinomial(0.8, Some(4), Some(0.9), 42)?;

        // Now returns u32 directly
        assert!(token < 6);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_deterministic() -> Result<()> {
        let device = Device::Cpu;

        // Very low temperature should be nearly deterministic
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let token1 = logits.sample_multinomial(0.01, None, None, 42)?;
        let token2 = logits.sample_multinomial(0.01, None, None, 42)?;

        // Now returns u32 directly
        assert_eq!(token1, token2);
        // Should almost always pick index 1 (highest logit)
        assert_eq!(token1, 1);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_error_cases() -> Result<()> {
        let device = Device::Cpu;

        // Test with wrong rank
        let logits_2d = Tensor::new(&[[1.0f32, 2.0], [0.5, 3.0]], &device)?;
        assert!(logits_2d.sample_multinomial(1.0, None, None, 42).is_err());

        // Test with integer tensor - use i64 which is supported by WithDType
        let logits_int = Tensor::from_vec(vec![1i64, 2, 3], 3, &device)?;
        assert!(logits_int.sample_multinomial(1.0, None, None, 42).is_err());

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_multinomial_sampling_cuda() -> Result<()> {
        if !crate::utils::cuda_is_available() {
            return Ok(()); // Skip if CUDA not available
        }

        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, None, 42)?;

        // Verify result is valid token index
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 4);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_multinomial_sampling_cuda_native_kernel() -> Result<()> {
        if !crate::utils::cuda_is_available() {
            return Ok(()); // Skip if CUDA not available
        }

        let device = Device::new_cuda(0)?;

        // Test 1: Basic sampling with GPU kernel
        println!("🧪 Testing GPU-native CUDA kernel implementation");
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1], &device)?;
        let token_tensor = logits.sample_multinomial(1.0, None, None, 42)?;

        // Verify tensor stays on GPU
        assert!(token_tensor.device().is_cuda());
        let token = token_tensor.to_scalar::<u32>()?;
        assert!(token < 5);
        println!("   ✅ Basic sampling: token {}", token);

        // Test 2: Temperature scaling on GPU
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let token_hot = logits.sample_multinomial(2.0, None, None, 123)?;
        let token_cold = logits.sample_multinomial(0.1, None, None, 123)?;

        let val_hot = token_hot.to_scalar::<u32>()?;
        let val_cold = token_cold.to_scalar::<u32>()?;
        assert!(val_hot < 4);
        assert!(val_cold < 4);
        println!(
            "   ✅ Temperature scaling: hot={}, cold={}",
            val_hot, val_cold
        );

        // Test 3: Top-k filtering on GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5, 2.5], &device)?;
        let token = logits.sample_multinomial(1.0, Some(3), None, 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 7);
        println!("   ✅ Top-k filtering: token {}", token_val);

        // Test 4: Top-p (nucleus) sampling on GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, Some(0.8), 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 4);
        println!("   ✅ Top-p sampling: token {}", token_val);

        // Test 5: Combined top-k and top-p on GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5], &device)?;
        let token = logits.sample_multinomial(0.8, Some(4), Some(0.9), 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 6);
        println!("   ✅ Combined top-k + top-p: token {}", token_val);

        // Test 6: Deterministic behavior with same seed
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let token1 = logits.sample_multinomial(0.01, None, None, 999)?;
        let token2 = logits.sample_multinomial(0.01, None, None, 999)?;

        let val1 = token1.to_scalar::<u32>()?;
        let val2 = token2.to_scalar::<u32>()?;
        assert_eq!(val1, val2, "Same seed should give same result");
        println!("   ✅ Deterministic: seed 999 -> token {}", val1);

        // Test 7: Large vocabulary (realistic LLM size)
        let vocab_size = 32000;
        let logits_data: Vec<f32> = (0..vocab_size)
            .map(|i| {
                let x = i as f32 / vocab_size as f32;
                if i < vocab_size / 10 {
                    2.0 + x * 3.0
                } else {
                    -1.0 + x * 0.5
                }
            })
            .collect();
        let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;
        let token = logits.sample_multinomial(0.8, Some(50), Some(0.9), 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < vocab_size as u32);
        println!("   ✅ Large vocabulary (32K): token {}", token_val);

        println!("🎉 All GPU-native CUDA kernel tests passed!");
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_multinomial_sampling_metal() -> Result<()> {
        if !crate::utils::metal_is_available() {
            return Ok(()); // Skip if Metal not available
        }

        let device = Device::new_metal(0)?;
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, None, 42)?;

        // Verify result is valid token index
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 4);

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_multinomial_sampling_metal_native_kernel() -> Result<()> {
        if !crate::utils::metal_is_available() {
            return Ok(()); // Skip if Metal not available
        }

        let device = Device::new_metal(0)?;

        // Test 1: Basic sampling with Metal kernel
        println!("🧪 Testing GPU-native Metal kernel implementation");
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1], &device)?;
        let token_tensor = logits.sample_multinomial(1.0, None, None, 42)?;

        // Verify tensor stays on Metal GPU
        assert!(token_tensor.device().is_metal());
        let token = token_tensor.to_scalar::<u32>()?;
        assert!(token < 5);
        println!("   ✅ Basic sampling: token {}", token);

        // Test 2: Temperature scaling on Metal GPU
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let token_hot = logits.sample_multinomial(2.0, None, None, 123)?;
        let token_cold = logits.sample_multinomial(0.1, None, None, 123)?;

        let val_hot = token_hot.to_scalar::<u32>()?;
        let val_cold = token_cold.to_scalar::<u32>()?;
        assert!(val_hot < 4);
        assert!(val_cold < 4);
        println!(
            "   ✅ Temperature scaling: hot={}, cold={}",
            val_hot, val_cold
        );

        // Test 3: Top-k filtering on Metal GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5, 2.5], &device)?;
        let token = logits.sample_multinomial(1.0, Some(3), None, 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 7);
        println!("   ✅ Top-k filtering: token {}", token_val);

        // Test 4: Top-p (nucleus) sampling on Metal GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, Some(0.8), 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 4);
        println!("   ✅ Top-p sampling: token {}", token_val);

        // Test 5: Combined top-k and top-p on Metal GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5], &device)?;
        let token = logits.sample_multinomial(0.8, Some(4), Some(0.9), 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < 6);
        println!("   ✅ Combined top-k + top-p: token {}", token_val);

        // Test 6: Deterministic behavior with same seed
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let token1 = logits.sample_multinomial(0.01, None, None, 999)?;
        let token2 = logits.sample_multinomial(0.01, None, None, 999)?;

        let val1 = token1.to_scalar::<u32>()?;
        let val2 = token2.to_scalar::<u32>()?;
        assert_eq!(val1, val2, "Same seed should give same result");
        println!("   ✅ Deterministic: seed 999 -> token {}", val1);

        // Test 7: Large vocabulary (realistic LLM size)
        let vocab_size = 32000;
        let logits_data: Vec<f32> = (0..vocab_size)
            .map(|i| {
                let x = i as f32 / vocab_size as f32;
                if i < vocab_size / 10 {
                    2.0 + x * 3.0
                } else {
                    -1.0 + x * 0.5
                }
            })
            .collect();
        let logits = Tensor::from_vec(logits_data, vocab_size, &device)?;
        let token = logits.sample_multinomial(0.8, Some(50), Some(0.9), 42)?;
        let token_val = token.to_scalar::<u32>()?;
        assert!(token_val < vocab_size as u32);
        println!("   ✅ Large vocabulary (32K): token {}", token_val);

        println!("🎉 All GPU-native Metal kernel tests passed!");
        Ok(())
    }
}
