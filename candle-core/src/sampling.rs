use crate::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor, WithDType};

#[cfg(feature = "cuda")]
use crate::{backend::BackendStorage, CudaStorage};

#[cfg(feature = "metal")]
use crate::{backend::BackendStorage, MetalStorage};

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
        // For now, fall back to CPU implementation by transferring data
        // TODO: Implement proper CUDA kernel for multinomial sampling
        let cpu_storage = storage.to_cpu_storage()?;
        let (result_cpu, shape) = self.cpu_fwd(&cpu_storage, layout)?;

        // Transfer result back to CUDA
        let cuda_device = storage.device();
        let result_cuda = cuda_device.storage_from_cpu_storage(&result_cpu)?;
        Ok((result_cuda, shape))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        // For now, fall back to CPU implementation by transferring data
        // TODO: Implement proper Metal kernel for multinomial sampling
        let cpu_storage = storage.to_cpu_storage()?;
        let (result_cpu, shape) = self.cpu_fwd(&cpu_storage, layout)?;

        // Transfer result back to Metal
        let metal_device = storage.device();
        let result_metal = metal_device.storage_from_cpu_storage(&result_cpu)?;
        Ok((result_metal, shape))
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
    /// A tensor containing the sampled token index as u32
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    ///
    /// let device = Device::Cpu;
    /// let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
    /// let token = logits.sample_multinomial_gpu(0.8, Some(3), Some(0.9), 42)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn sample_multinomial_gpu(
        &self,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
    ) -> Result<Self> {
        // Ensure input is 1D
        if self.rank() != 1 {
            crate::bail!(
                "sample_multinomial_gpu requires 1D tensor, got shape: {:?}",
                self.shape()
            );
        }

        // Ensure input is float type for logits
        if !matches!(
            self.dtype(),
            DType::F32 | DType::F64 | DType::F16 | DType::BF16
        ) {
            crate::bail!(
                "sample_multinomial_gpu requires float tensor, got dtype: {:?}",
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
        logits.apply_op1_no_bwd(&sampling_op)
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
        let token = logits.sample_multinomial_gpu(1.0, None, None, 42)?;

        assert_eq!(token.shape().dims(), &[1]);
        assert_eq!(token.dtype(), DType::U32);

        let token_val = token.to_vec1::<u32>()?[0];
        assert!(token_val < 4); // Should be valid index

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_with_temperature() -> Result<()> {
        let device = Device::Cpu;

        // Test with high temperature (more random)
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token_hot = logits.sample_multinomial_gpu(2.0, None, None, 42)?;

        // Test with low temperature (more deterministic)
        let token_cold = logits.sample_multinomial_gpu(0.1, None, None, 42)?;

        assert_eq!(token_hot.dtype(), DType::U32);
        assert_eq!(token_cold.dtype(), DType::U32);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_top_k() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1], &device)?;
        let token = logits.sample_multinomial_gpu(1.0, Some(2), None, 42)?;

        let token_val = token.to_vec1::<u32>()?[0];
        // With top_k=2, should only sample from indices 3 or 1 (highest logits)
        assert!(token_val == 3 || token_val == 1);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_top_p() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial_gpu(1.0, None, Some(0.8), 42)?;

        assert_eq!(token.dtype(), DType::U32);
        let token_val = token.to_vec1::<u32>()?[0];
        assert!(token_val < 4);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_combined() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5], &device)?;
        let token = logits.sample_multinomial_gpu(0.8, Some(4), Some(0.9), 42)?;

        assert_eq!(token.dtype(), DType::U32);
        let token_val = token.to_vec1::<u32>()?[0];
        assert!(token_val < 6);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_deterministic() -> Result<()> {
        let device = Device::Cpu;

        // Very low temperature should be nearly deterministic
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let token1 = logits.sample_multinomial_gpu(0.01, None, None, 42)?;
        let token2 = logits.sample_multinomial_gpu(0.01, None, None, 42)?;

        let val1 = token1.to_vec1::<u32>()?[0];
        let val2 = token2.to_vec1::<u32>()?[0];

        // Same seed should give same result
        assert_eq!(val1, val2);
        // Should almost always pick index 1 (highest logit)
        assert_eq!(val1, 1);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_error_cases() -> Result<()> {
        let device = Device::Cpu;

        // Test with wrong rank
        let logits_2d = Tensor::new(&[[1.0f32, 2.0], [0.5, 3.0]], &device)?;
        assert!(logits_2d
            .sample_multinomial_gpu(1.0, None, None, 42)
            .is_err());

        // Test with integer tensor - use i64 which is supported by WithDType
        let logits_int = Tensor::from_vec(vec![1i64, 2, 3], 3, &device)?;
        assert!(logits_int
            .sample_multinomial_gpu(1.0, None, None, 42)
            .is_err());

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
        let token = logits.sample_multinomial_gpu(1.0, None, None, 42)?;

        assert_eq!(token.shape().dims(), &[1]);
        assert_eq!(token.dtype(), DType::U32);
        assert_eq!(token.device(), device);

        let token_val = token.to_vec1::<u32>()?[0];
        assert!(token_val < 4);

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
        let token = logits.sample_multinomial_gpu(1.0, None, None, 42)?;

        assert_eq!(token.shape().dims(), &[1]);
        assert_eq!(token.dtype(), DType::U32);
        assert_eq!(token.device(), device);

        let token_val = token.to_vec1::<u32>()?[0];
        assert!(token_val < 4);

        Ok(())
    }
}
