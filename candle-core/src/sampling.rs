use crate::{
    CpuStorage, CustomOp1, DType, InplaceOp2, Layout, LiveTensor, Result, Shape, Tensor, WithDType,
};

#[cfg(feature = "cuda")]
use crate::cuda_backend::argmax_rows::argmax_rows_f32;
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
        use crate::cuda_backend::cudarc::driver::DevicePtr;
        use crate::cuda_backend::CudaStorageSlice;
        use candle_kernels::simple::multinomial::{
            calculate_shared_mem_size, calculate_workspace_size, MultinomialDType,
            DEFAULT_NUM_THREADS,
        };

        // 🚀 TRUE GPU-NATIVE SAMPLING - NO CPU TRANSFERS!
        // All computation happens on GPU via FFI

        let device = &storage.device;
        let vocab_size = layout.shape().dims()[0];
        let stream = device.cuda_stream();

        // Allocate output buffer on GPU (single u32)
        let output_slice = unsafe { device.alloc::<u32>(1)? };

        // Allocate workspace for sorting (needed for top-k/top-p)
        let workspace_size = calculate_workspace_size(vocab_size);
        let workspace_slice = unsafe { device.alloc::<f32>(workspace_size / 4)? };

        // Get dtype for dispatcher
        let dtype = match &storage.slice {
            CudaStorageSlice::F32(_) => MultinomialDType::F32 as i32,
            CudaStorageSlice::F64(_) => MultinomialDType::F64 as i32,
            CudaStorageSlice::F16(_) => MultinomialDType::F16 as i32,
            CudaStorageSlice::BF16(_) => MultinomialDType::BF16 as i32,
            _ => crate::bail!("Unsupported dtype for GPU multinomial sampling"),
        };

        let top_k_val = self.top_k.unwrap_or(0) as u32;
        let top_p_val = self.top_p.unwrap_or(1.0) as f32;
        let num_threads = DEFAULT_NUM_THREADS;
        let shared_mem_size = calculate_shared_mem_size(num_threads);

        // All FFI calls wrapped in a block so guards drop before we move output_slice
        {
            // Get pointers for FFI
            let (output_ptr, _out_guard) = output_slice.device_ptr(&stream);
            let (workspace_ptr, _ws_guard) = workspace_slice.device_ptr(&stream);

            // Call FFI based on dtype
            match &storage.slice {
                CudaStorageSlice::F32(s) => {
                    let (src_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        candle_kernels::simple::multinomial::run_multinomial(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            output_ptr as *mut u32,
                            workspace_ptr as *mut f32,
                            vocab_size,
                            self.temperature,
                            top_k_val,
                            top_p_val,
                            self.seed,
                            num_threads,
                            shared_mem_size,
                            std::ptr::null_mut(),
                        );
                    }
                }
                CudaStorageSlice::F64(s) => {
                    let (src_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        candle_kernels::simple::multinomial::run_multinomial(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            output_ptr as *mut u32,
                            workspace_ptr as *mut f32,
                            vocab_size,
                            self.temperature,
                            top_k_val,
                            top_p_val,
                            self.seed,
                            num_threads,
                            shared_mem_size,
                            std::ptr::null_mut(),
                        );
                    }
                }
                CudaStorageSlice::F16(s) => {
                    let (src_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        candle_kernels::simple::multinomial::run_multinomial(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            output_ptr as *mut u32,
                            workspace_ptr as *mut f32,
                            vocab_size,
                            self.temperature,
                            top_k_val,
                            top_p_val,
                            self.seed,
                            num_threads,
                            shared_mem_size,
                            std::ptr::null_mut(),
                        );
                    }
                }
                CudaStorageSlice::BF16(s) => {
                    let (src_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        candle_kernels::simple::multinomial::run_multinomial(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            output_ptr as *mut u32,
                            workspace_ptr as *mut f32,
                            vocab_size,
                            self.temperature,
                            top_k_val,
                            top_p_val,
                            self.seed,
                            num_threads,
                            shared_mem_size,
                            std::ptr::null_mut(),
                        );
                    }
                }
                _ => unreachable!(),
            }
        }

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

// ============================================================================
// Batched Sampling — fused GPU kernel for sampling from [batch, vocab] logits
// ============================================================================

/// GPU-fused batched sampling operation.
///
/// Takes a 2D [batch_size, vocab_size] logits tensor and returns a 1D [batch_size]
/// tensor of sampled token IDs (u32). All sampling (argmax, top-k, top-p,
/// temperature scaling) happens in a single CUDA kernel launch.
///
/// This is dramatically faster than per-sequence sampling because:
/// - Single kernel launch processes all sequences in parallel
/// - No host↔device round-trips between sequences
/// - Fused top-k radix select + softmax + sampling in one pass
#[derive(Debug, Clone)]
pub struct BatchedSampling {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u64,
    /// Per-sequence RNG offsets (must be batch_size elements).
    /// Each call increments the offset for the sequence it sampled.
    /// If empty, offsets start at 0 for all sequences.
    pub rng_offsets: Vec<u64>,
    /// Tokens a row may produce: the tokenizer's last id + 1. A checkpoint
    /// pads its output projection past it, and the padded tail of each row
    /// carries no probability. Clamped to the row width, so the row width
    /// itself means "no padded tail".
    pub live_vocab: usize,
}

impl BatchedSampling {
    /// Create a new batched sampling op with the given parameters.
    ///
    /// - `temperature`: 0.0 = argmax/greedy, >0 = stochastic
    /// - `top_k`: 0 = disabled, >0 = only consider top-k tokens
    /// - `top_p`: 1.0 = disabled, <1.0 = nucleus sampling
    /// - `seed`: RNG seed for reproducible sampling
    /// - `live_vocab`: tokens a row may produce (see the field)
    pub fn new(temperature: f32, top_k: i32, top_p: f32, seed: u64, live_vocab: usize) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            seed,
            rng_offsets: Vec::new(),
            live_vocab,
        }
    }

    /// Greedy/argmax sampling (temperature=0, no top-k/top-p) over the first
    /// `live_vocab` tokens of each row.
    pub fn argmax(live_vocab: usize) -> Self {
        Self::new(0.0, 0, 1.0, 0, live_vocab)
    }
}

impl CustomOp1 for BatchedSampling {
    fn name(&self) -> &'static str {
        "batched_sampling"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        // CPU fallback: per-sequence argmax (or softmax+sample for temp>0)
        let dims = layout.shape().dims();
        if dims.len() != 2 {
            crate::bail!(
                "BatchedSampling requires 2D [batch, vocab] tensor, got {:?}",
                layout.shape()
            );
        }
        let batch_size = dims[0];
        let vocab_size = dims[1];
        // The dtypes the CUDA kernel takes; half precision widens to f32
        // exactly, so every comparison below is the one the narrow value makes.
        let widened: Vec<f32>;
        let logits: &[f32] = match storage {
            CpuStorage::F32(v) => v,
            CpuStorage::F16(v) => {
                widened = v.iter().map(|x| x.to_f32()).collect();
                &widened
            }
            CpuStorage::BF16(v) => {
                widened = v.iter().map(|x| x.to_f32()).collect();
                &widened
            }
            _ => crate::bail!("BatchedSampling: unsupported dtype, expected F32/F16/BF16"),
        };

        // The padded tail of a row is not a token.
        let live = self.live_vocab.min(vocab_size);
        let mut tokens = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let offset = layout.start_offset() + b * vocab_size;
            let seq_logits = &logits[offset..offset + live];

            if self.temperature <= 0.0 {
                // Argmax
                let (best_idx, _) = seq_logits.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    },
                );
                tokens.push(best_idx as u32);
            } else {
                // Simple temperature-scaled sampling
                use rand::{Rng, SeedableRng};
                let rng_offset = if b < self.rng_offsets.len() {
                    self.rng_offsets[b]
                } else {
                    0
                };
                let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed.wrapping_add(rng_offset));
                let max_logit = seq_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = seq_logits
                    .iter()
                    .map(|&l| ((l - max_logit) / self.temperature).exp())
                    .collect();
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }
                let u: f32 = rng.random();
                let mut cumsum = 0.0f32;
                let mut chosen = live - 1;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if u < cumsum {
                        chosen = i;
                        break;
                    }
                }
                tokens.push(chosen as u32);
            }
        }

        Ok((CpuStorage::U32(tokens), Shape::from(batch_size)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use crate::cuda_backend::cudarc::driver::DevicePtr;
        use crate::cuda_backend::CudaStorageSlice;

        let dims = layout.shape().dims();
        if dims.len() != 2 {
            crate::bail!(
                "BatchedSampling requires 2D [batch, vocab] tensor, got {:?}",
                layout.shape()
            );
        }
        let batch_size = dims[0];
        let device = &storage.device;
        let stream = device.cuda_stream();

        // Map candle dtype to kernel dtype enum
        let dtype_code = match &storage.slice {
            CudaStorageSlice::F32(_) => 0i32,  // F32
            CudaStorageSlice::F16(_) => 1i32,  // F16
            CudaStorageSlice::BF16(_) => 2i32, // BF16
            _ => crate::bail!("BatchedSampling: unsupported dtype, expected F32/F16/BF16"),
        };

        // Allocate output buffer [batch_size] u32
        let output_slice = unsafe { device.alloc::<u32>(batch_size)? };
        {
            let (output_ptr, _out_guard) = output_slice.device_ptr(&stream);
            self.launch_cuda(storage, layout, output_ptr, dtype_code)?;
        }
        let result_storage = CudaStorage::wrap_cuda_slice(output_slice, device.clone());
        Ok((result_storage, Shape::from(batch_size)))
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

impl BatchedSampling {
    /// The fused sampler over `storage`'s `[batch, vocab]` rows, writing one
    /// `u32` token per row at `output_ptr` — device memory the caller owns.
    #[cfg(feature = "cuda")]
    fn launch_cuda(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
        output_ptr: u64,
        dtype_code: i32,
    ) -> Result<()> {
        use crate::cuda_backend::cudarc::driver::DevicePtr;
        use crate::cuda_backend::CudaStorageSlice;

        let dims = layout.shape().dims();
        let batch_size = dims[0];
        let vocab_size = dims[1];
        let device = &storage.device;
        let stream = device.cuda_stream();

        // A greedy pick over F32 rows spreads each row across many blocks
        // (`argmax_rows`) — the same pick, ties included, as the fused
        // sampler's one-block-a-row argmax, which streams a whole vocabulary row
        // through one SM. Every other dtype, and every draw, takes the fused
        // sampler.
        if self.temperature <= 0.0 {
            if let CudaStorageSlice::F32(s) = &storage.slice {
                let (ptr, _guard) = s.device_ptr(&stream);
                return argmax_rows_f32(
                    device,
                    ptr + layout.start_offset() as u64 * 4,
                    batch_size,
                    vocab_size,
                    self.live_vocab.min(vocab_size),
                    output_ptr,
                );
            }
        }

        // RNG offsets [batch_size] u64 — for a draw only. An argmax
        // (`temperature <= 0`) reads no RNG and the kernel takes a null pointer
        // there, so a greedy pick uploads nothing: it runs once per accept-walk
        // position and once per drafted token.
        let rng_slice = if self.temperature <= 0.0 && self.rng_offsets.is_empty() {
            None
        } else {
            let rng_data: Vec<u64> = if self.rng_offsets.len() == batch_size {
                self.rng_offsets.clone()
            } else {
                vec![0u64; batch_size]
            };
            Some(device.memcpy_stod(&rng_data)?)
        };

        // All guards must live until after the FFI call
        {
            let rng = rng_slice.as_ref().map(|s| s.device_ptr(&stream));
            let rng_ptr = rng.as_ref().map_or(0, |(p, _)| *p);

            // Helper closure to invoke the kernel with a logits pointer
            let call_kernel = |logits_ptr: u64| unsafe {
                candle_kernels::sampling::run_batched_sampling(
                    logits_ptr as *const std::ffi::c_void,
                    batch_size as i32,
                    vocab_size as i32,
                    self.live_vocab.min(vocab_size) as i32,
                    dtype_code,
                    self.temperature,
                    self.top_k,
                    self.top_p,
                    1.0,              // repeat_penalty (disabled)
                    0.0,              // frequency_penalty (disabled)
                    0.0,              // presence_penalty (disabled)
                    0.0,              // dry_multiplier (disabled)
                    1.75,             // dry_base (default)
                    2,                // dry_allowed_length (default)
                    0,                // dry_range (disabled)
                    0.0,              // eos_boost (disabled)
                    -1,               // eos_token_id (disabled)
                    0,                // eos_ramp_start (disabled)
                    0,                // eos_ramp_len (disabled)
                    0.0,              // eos_boost_max_multiplier (disabled)
                    0.0,              // cross_turn_penalty (disabled)
                    std::ptr::null(), // cross_turn_counts
                    std::ptr::null(), // current_lens
                    0.0,              // segment_close_boost (disabled)
                    -1,               // segment_close_token_id (disabled)
                    0,                // segment_close_ramp_start (disabled)
                    0,                // segment_close_ramp_len (disabled)
                    0.0,              // segment_close_max_multiplier (disabled)
                    std::ptr::null(), // segment_lens
                    std::ptr::null(), // dry_lens
                    0.0,              // segment_temp_boost (disabled)
                    std::ptr::null(), // suppress_tokens (disabled)
                    0,                // suppress_count (disabled)
                    std::ptr::null(), // suppress_penalties (disabled)
                    std::ptr::null(), // token_counts
                    std::ptr::null(), // banned_tokens
                    0,                // num_banned_tokens
                    0,                // banned_tokens_per_seq
                    std::ptr::null(), // recent_tokens
                    std::ptr::null(), // recent_lens
                    0,                // max_recent_len
                    std::ptr::null(), // stencil
                    0,                // stencil_size
                    output_ptr as *mut u32,
                    self.seed,
                    rng_ptr as *mut u64,
                    std::ptr::null(), // seq_dials — this path is uniform-config
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            };

            match &storage.slice {
                CudaStorageSlice::F32(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = ptr + (layout.start_offset() as u64 * 4);
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::F16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = ptr + (layout.start_offset() as u64 * 2);
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::BF16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = ptr + (layout.start_offset() as u64 * 2);
                    call_kernel(logits_ptr);
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}

/// [`BatchedSampling`] writing its tokens into a `u32` buffer the caller holds
/// rather than a fresh allocation — see [`LiveTensor::batched_sample_argmax_into`].
struct SampleInto(BatchedSampling);

impl InplaceOp2 for SampleInto {
    fn name(&self) -> &'static str {
        "batched_sampling_into"
    }

    fn cpu_fwd(
        &self,
        out: &mut CpuStorage,
        out_l: &Layout,
        s: &CpuStorage,
        l: &Layout,
    ) -> Result<()> {
        let (tokens, _) = self.0.cpu_fwd(s, l)?;
        let (CpuStorage::U32(tokens), CpuStorage::U32(dst)) = (tokens, out) else {
            crate::bail!("batched_sample_argmax_into: the output must be U32");
        };
        let at = out_l.start_offset();
        dst[at..at + tokens.len()].copy_from_slice(&tokens);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        out: &mut CudaStorage,
        out_l: &Layout,
        s: &CudaStorage,
        l: &Layout,
    ) -> Result<()> {
        use crate::cuda_backend::cudarc::driver::DevicePtr;
        use crate::cuda_backend::CudaStorageSlice;

        let dtype_code = match &s.slice {
            CudaStorageSlice::F32(_) => 0i32,
            CudaStorageSlice::F16(_) => 1i32,
            CudaStorageSlice::BF16(_) => 2i32,
            _ => crate::bail!("BatchedSampling: unsupported dtype, expected F32/F16/BF16"),
        };
        let CudaStorageSlice::U32(dst) = &out.slice else {
            crate::bail!("batched_sample_argmax_into: the output must be U32");
        };
        let stream = out.device.cuda_stream();
        let (base, _guard) = dst.device_ptr(&stream);
        let output_ptr = base + (out_l.start_offset() * std::mem::size_of::<u32>()) as u64;
        self.0.launch_cuda(s, l, output_ptr, dtype_code)
    }
}

impl<'w> LiveTensor<'w> {
    /// Sample a token from logits using GPU-native multinomial sampling.
    ///
    /// This method performs sampling directly on the GPU device without transferring
    /// data to CPU, providing better performance for inference workloads.
    ///
    /// # Arguments
    /// * `temperature` - Controls randomness (lower = more deterministic)
    /// * `top_k` - Optional top-k filtering (limits to k most likely tokens)
    /// * `top_p` - Optional nucleus sampling (limits to tokens with cumulative probability >= p)
    /// * `seed` - Random seed for reproducible sampling
    ///
    /// # GPU Implementation Status
    /// - ✅ Parallel softmax with temperature (256 threads)
    /// - ✅ Fast random sampling with curand
    /// - ✅ top_k filtering (in-kernel sorting)
    /// - ✅ top_p/nucleus sampling (in-kernel prefix sum)
    ///
    /// All operations execute on GPU with minimal overhead. For large vocabularies,
    /// top-k/top-p filtering uses workspace memory allocated on-demand.
    ///
    /// # Returns
    /// A rank-0 tensor containing the sampled token index. Call `.to_scalar()` to extract.
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Tensor, Device, DType};
    ///
    /// let device = Device::Cpu;
    /// let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
    /// let token = logits.sample_multinomial(0.8, None, None, 42)?; // GPU-native
    /// let token_id = token.to_scalar::<u32>()?; // Extract when needed
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

        // The kernel returns a [1] tensor, squeeze it to scalar
        result_tensor.squeeze(0)
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

    /// Batched greedy/argmax sampling via the fused GPU kernel.
    ///
    /// Takes a 2D [batch_size, vocab_size] logits tensor and returns a 1D [batch_size]
    /// tensor of u32 token IDs. On CUDA, this dispatches to the fused batched sampling
    /// kernel; on CPU, it falls back to per-sequence argmax.
    ///
    /// This is `self.argmax(1)` over the first `live_vocab` columns — the
    /// tokens the tokenizer can name — using the fused kernel on GPU. A
    /// checkpoint pads its logits row past them; pass the row width for one
    /// that does not.
    pub fn batched_sample_argmax(&self, live_vocab: usize) -> Result<Self> {
        if self.rank() != 2 {
            crate::bail!(
                "batched_sample_argmax requires 2D [batch, vocab] tensor, got {:?}",
                self.shape()
            );
        }
        // Ensure contiguous layout for the fused kernel
        let logits = self.contiguous()?;
        let op = BatchedSampling::argmax(live_vocab);
        logits.apply_op1_no_bwd(&op)
    }

    /// [`Self::batched_sample_argmax`] writing its `[batch]` tokens into `out`,
    /// a contiguous `u32` buffer of exactly `batch` elements the caller holds,
    /// rather than into a fresh allocation.
    ///
    /// For a loop that picks a token per step and keeps every pick: the steps
    /// write rows of one buffer the loop sized once, so a step allocates
    /// nothing. The logits must already be contiguous — they are read in place,
    /// and a copy to make them so would be an allocation of its own.
    pub fn batched_sample_argmax_into(&self, live_vocab: usize, out: &Self) -> Result<()> {
        if self.rank() != 2 {
            crate::bail!(
                "batched_sample_argmax_into requires 2D [batch, vocab] logits, got {:?}",
                self.shape()
            );
        }
        if !self.is_contiguous() {
            crate::bail!("batched_sample_argmax_into: the logits must be contiguous");
        }
        let batch = self.dim(0)?;
        if out.dtype() != DType::U32 || !out.is_contiguous() || out.elem_count() != batch {
            crate::bail!(
                "batched_sample_argmax_into: the output must be a contiguous U32 buffer of \
                 {batch} elements, got {:?} {:?}",
                out.dtype(),
                out.shape()
            );
        }
        out.inplace_op2(self, &SampleInto(BatchedSampling::argmax(live_vocab)))
    }

    /// Batched sampling with full control over temperature, top-k, top-p and seed.
    ///
    /// Takes a 2D [batch_size, vocab_size] logits tensor and returns a 1D [batch_size]
    /// tensor of u32 token IDs. On CUDA, all sampling logic runs in a single fused
    /// kernel launch across all sequences.
    ///
    /// # Arguments
    /// * `temperature` - 0.0 for argmax/greedy, >0 for stochastic sampling
    /// * `top_k` - 0 to disable, >0 to keep only the top-k most probable tokens
    /// * `top_p` - 1.0 to disable, <1.0 for nucleus sampling
    /// * `seed` - RNG seed for reproducible sampling
    /// * `live_vocab` - tokens a row may produce; the row width for no padded tail
    pub fn batched_sample(
        &self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        seed: u64,
        live_vocab: usize,
    ) -> Result<Self> {
        if self.rank() != 2 {
            crate::bail!(
                "batched_sample requires 2D [batch, vocab] tensor, got {:?}",
                self.shape()
            );
        }
        // Ensure contiguous layout for the fused kernel
        let logits = self.contiguous()?;
        let op = BatchedSampling::new(temperature, top_k, top_p, seed, live_vocab);
        logits.apply_op1_no_bwd(&op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    /// The in-place argmax writes each row's pick into the row of the buffer it
    /// is handed and nowhere else, on the host and on the device alike.
    #[test]
    fn argmax_into_writes_the_row_it_is_given() -> Result<()> {
        let mut devices = vec![Device::Cpu];
        if let Ok(cuda) = Device::new_cuda(0) {
            devices.push(cuda);
        }
        for dev in devices {
            let logits = Tensor::new(&[[0.1f32, 3.0, 0.2], [5.0, 1.0, 0.0]], &dev)?;
            let buf = Tensor::new(&[[7u32, 7], [7, 7], [7, 7]], &dev)?;
            logits.batched_sample_argmax_into(3, &buf.get(1)?)?;
            assert_eq!(
                buf.to_vec2::<u32>()?,
                vec![vec![7, 7], vec![1, 0], vec![7, 7]]
            );
            // The padded tail carries no probability.
            logits.batched_sample_argmax_into(1, &buf.get(2)?)?;
            assert_eq!(buf.get(2)?.to_vec1::<u32>()?, vec![0, 0]);
            // A buffer of the wrong width is refused, not written past.
            assert!(logits
                .batched_sample_argmax_into(3, &buf.flatten_all()?)
                .is_err());
        }
        Ok(())
    }

    /// The fused sampler's greedy pick, as its kernel computes it: thread
    /// `i mod 1024` keeps the first index of its stride class that is strictly
    /// greater than its running maximum (which `fmaxf` updates, so NaN never
    /// registers), then a tree keeps the lower thread on ties.
    fn sampler_greedy(row: &[f32], live: usize) -> u32 {
        const THREADS: usize = 1024;
        let mut vals = [f32::NEG_INFINITY; THREADS];
        let mut idxs = [0u32; THREADS];
        for t in 0..THREADS {
            for i in (t..live).step_by(THREADS) {
                let v = row[i];
                if v > vals[t] {
                    idxs[t] = i as u32;
                }
                vals[t] = vals[t].max(v);
            }
        }
        let (mut best_v, mut best_i) = (vals[0], idxs[0]);
        for t in 1..THREADS {
            if vals[t] > best_v {
                best_v = vals[t];
                best_i = idxs[t];
            }
        }
        best_i
    }

    /// The many-blocks greedy pick is the fused sampler's pick on every row:
    /// ties resolved by the lower `i mod 1024` and then the lower index, +0
    /// tying −0, NaN never winning, a −∞ row answering 0, a padded tail
    /// ignored — and on pseudo-random vocabulary-width rows.
    #[test]
    fn the_greedy_pick_is_the_fused_samplers_pick() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else {
            return Ok(());
        };
        let width = 248_320usize;
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let at = |pairs: &[(usize, f32)]| {
            let mut r = vec![0f32; width];
            for &(i, v) in pairs {
                r[i] = v;
            }
            r
        };
        rows.push(at(&[(5, 1.0), (1027, 1.0)])); // 1027: class 3 beats class 5
        rows.push(at(&[(3, 1.0), (1027, 1.0)])); // same class: the lower index
        rows.push(at(&[(2051, 1.0), (1027, 1.0)]));
        rows.push(vec![f32::NEG_INFINITY; width]);
        let mut nan_row = vec![f32::NEG_INFINITY; width];
        nan_row[0] = f32::NAN;
        nan_row[9] = 2.0;
        rows.push(nan_row);
        rows.push(vec![f32::NAN; width]);
        let mut zeros = vec![-1f32; width];
        zeros[7] = -0.0;
        zeros[2000] = 0.0;
        rows.push(zeros);
        let mut tail = vec![0f32; width];
        tail[width - 1] = 9.0; // past the live vocabulary below
        tail[width - 3] = 4.0; // in the slice's last partial quad
        rows.push(tail);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..5 {
            rows.push(
                (0..width)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        // A coarse grid, so a row holds repeated maxima.
                        ((s >> 40) % 4096) as f32 / 64.0
                    })
                    .collect(),
            );
        }
        let live = width - 2;
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let logits = Tensor::from_vec(flat, (rows.len(), width), &dev)?;
        let got = logits.batched_sample_argmax(live)?.to_vec1::<u32>()?;
        let want: Vec<u32> = rows.iter().map(|r| sampler_greedy(r, live)).collect();
        assert_eq!(got, want);
        assert_eq!(&want[..4], &[1027, 3, 1027, 0]);
        assert_eq!(want[4], 9);
        assert_eq!(want[5], 0);
        assert_eq!(want[6], 7);
        assert_eq!(want[7], (width - 3) as u32);
        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_cpu() -> Result<()> {
        let device = Device::Cpu;

        // Test with simple logits
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, None, 42)?;
        let token_id = token.to_scalar::<u32>()?;

        assert!(token_id < 4); // Should be valid index

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_with_temperature() -> Result<()> {
        let device = Device::Cpu;

        // Test with high temperature (more random)
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token_hot = logits
            .sample_multinomial(2.0, None, None, 42)?
            .to_scalar::<u32>()?;

        // Test with low temperature (more deterministic)
        let token_cold = logits
            .sample_multinomial(0.1, None, None, 42)?
            .to_scalar::<u32>()?;

        assert!(token_hot < 4);
        assert!(token_cold < 4);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_top_k() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1], &device)?;
        let token = logits
            .sample_multinomial(1.0, Some(2), None, 42)?
            .to_scalar::<u32>()?;

        assert!(token == 3 || token == 1);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_top_p() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits
            .sample_multinomial(1.0, None, Some(0.8), 42)?
            .to_scalar::<u32>()?;

        assert!(token < 4);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_combined() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5], &device)?;
        let token = logits
            .sample_multinomial(0.8, Some(4), Some(0.9), 42)?
            .to_scalar::<u32>()?;

        assert!(token < 6);

        Ok(())
    }

    #[test]
    fn test_multinomial_sampling_deterministic() -> Result<()> {
        let device = Device::Cpu;

        // Very low temperature should be nearly deterministic
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let token1 = logits
            .sample_multinomial(0.01, None, None, 42)?
            .to_scalar::<u32>()?;
        let token2 = logits
            .sample_multinomial(0.01, None, None, 42)?
            .to_scalar::<u32>()?;

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
        let token = logits
            .sample_multinomial(1.0, None, None, 42)?
            .to_scalar::<u32>()?;
        assert!(token < 5);
        println!("   ✅ Basic sampling: token {}", token);

        // Test 2: Temperature scaling on GPU
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let val_hot = logits
            .sample_multinomial(2.0, None, None, 123)?
            .to_scalar::<u32>()?;
        let val_cold = logits
            .sample_multinomial(0.1, None, None, 123)?
            .to_scalar::<u32>()?;
        assert!(val_hot < 4);
        assert!(val_cold < 4);
        println!(
            "   ✅ Temperature scaling: hot={}, cold={}",
            val_hot, val_cold
        );

        // Test 3: Top-k filtering on GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5, 2.5], &device)?;
        let token = logits
            .sample_multinomial(1.0, Some(3), None, 42)?
            .to_scalar::<u32>()?;
        assert!(token < 7);
        println!("   ✅ Top-k filtering: token {}", token);

        // Test 4: Top-p (nucleus) sampling on GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits
            .sample_multinomial(1.0, None, Some(0.8), 42)?
            .to_scalar::<u32>()?;
        assert!(token < 4);
        println!("   ✅ Top-p sampling: token {}", token);

        // Test 5: Combined top-k and top-p on GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5], &device)?;
        let token = logits
            .sample_multinomial(0.8, Some(4), Some(0.9), 42)?
            .to_scalar::<u32>()?;
        assert!(token < 6);
        println!("   ✅ Combined top-k + top-p: token {}", token);

        // Test 6: Deterministic behavior with same seed
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let val1 = logits
            .sample_multinomial(0.01, None, None, 999)?
            .to_scalar::<u32>()?;
        let val2 = logits
            .sample_multinomial(0.01, None, None, 999)?
            .to_scalar::<u32>()?;
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
        let token_scalar = token.to_vec0::<u32>()?;
        assert!(token_scalar < vocab_size as u32);
        println!("   ✅ Large vocabulary (32K): token {}", token_scalar);

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
        let token = logits
            .sample_multinomial(1.0, None, None, 42)?
            .to_scalar::<u32>()?;
        assert!(token < 4);

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
        let token = logits.sample_multinomial(1.0, None, None, 42)?;
        assert!(token < 5);
        println!("   ✅ Basic sampling: token {}", token);

        // Test 2: Temperature scaling on Metal GPU
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let val_hot = logits.sample_multinomial(2.0, None, None, 123)?;
        let val_cold = logits.sample_multinomial(0.1, None, None, 123)?;
        assert!(val_hot < 4);
        assert!(val_cold < 4);
        println!(
            "   ✅ Temperature scaling: hot={}, cold={}",
            val_hot, val_cold
        );

        // Test 3: Top-k filtering on Metal GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5, 2.5], &device)?;
        let token = logits.sample_multinomial(1.0, Some(3), None, 42)?;
        assert!(token < 7);
        println!("   ✅ Top-k filtering: token {}", token);

        // Test 4: Top-p (nucleus) sampling on Metal GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?;
        let token = logits.sample_multinomial(1.0, None, Some(0.8), 42)?;
        assert!(token < 4);
        println!("   ✅ Top-p sampling: token {}", token);

        // Test 5: Combined top-k and top-p on Metal GPU
        let logits = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 0.1, 1.5], &device)?;
        let token = logits.sample_multinomial(0.8, Some(4), Some(0.9), 42)?;
        assert!(token < 6);
        println!("   ✅ Combined top-k + top-p: token {}", token);

        // Test 6: Deterministic behavior with same seed
        let logits = Tensor::new(&[1.0f32, 5.0, 0.5, 2.0], &device)?;
        let val1 = logits.sample_multinomial(0.01, None, None, 999)?;
        let val2 = logits.sample_multinomial(0.01, None, None, 999)?;
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
        assert!(token < vocab_size as u32);
        println!("   ✅ Large vocabulary (32K): token {}", token);

        println!("🎉 All GPU-native Metal kernel tests passed!");
        Ok(())
    }
}
