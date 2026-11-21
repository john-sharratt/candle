//! Updates a few entries of a tensor's last dimension, for penalties and biases
//! on logits.
//!
//! The `*_at_indices` methods subtract, divide, add or multiply a value at the
//! listed indices, and the `_mut` forms do it in place. The functional forms
//! clone the whole tensor first, which the comments measure as the bulk of their
//! cost on a 150K vocabulary. `repeat_penalty_mut` applies one penalty to a set
//! of token indices in a single pass, dividing positive logits and multiplying
//! the others.

use super::*;

/// The tensor API, generic over how long the storage is valid.
///
/// Unlike [`crate::quantized::LiveQTensor`], this block is *not* split into
/// `'static` constructors and `'w` accessors, because here the `-> Result<Self>`
/// methods are overwhelmingly **views** — `reshape`, `narrow`, `t`,
/// `broadcast_as` — and a view of a lease has to stay a lease. Returning `Self`
/// is what makes that true by construction. The handful of methods that
/// allocate leave `'w` free, which is sound because owned memory outlives every
/// choice of `'w`; the discipline that matters is applied at the *lease*
/// constructors instead, where `'w` is tied to a borrow of whatever owns the
/// bytes.
impl<'w> LiveTensor<'w> {
    /// Subtract a scalar value from elements at specific indices.
    ///
    /// This is optimized for sparse updates (e.g., repetition penalties in text generation).
    /// Much faster than broadcast operations when updating a small subset of indices.
    /// Subtracts a scalar value from specific indices in the tensor's last dimension.
    ///
    /// This is optimized for sparse updates (e.g., repetition penalty in LLM sampling)
    /// where only a small fraction of vocabulary logits need adjustment.
    ///
    /// # Performance Note (CUDA)
    /// Due to Candle's immutable API design, this operation must clone the entire tensor
    /// even for sparse updates. For a 150K vocab tensor:
    /// - Clone: ~20ms (bottleneck)
    /// - Kernel: <1ms for 50 indices
    ///
    /// **For 20x+ speedup, use `sub_at_indices_mut()` instead (requires mutable tensor)**
    ///
    /// **Workarounds for performance-critical code:**
    /// 1. Batch multiple sparse updates into a single call
    /// 2. Minimize consecutive sub_at_indices operations
    /// 3. Use sub_at_indices_mut() if you have a mutable tensor
    ///
    /// # Arguments
    /// * `indices` - Indices in the last dimension to update (can contain duplicates)
    /// * `value` - Scalar value to subtract from each indexed element
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let penalty_tokens = vec![10u32, 25u32, 10u32]; // Token 10 penalized twice
    /// let result = logits.sub_at_indices(&penalty_tokens, 5.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn sub_at_indices(&self, indices: &[u32], value: f32) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "sub_at_indices",
                });
            }
        }

        // Dispatch to backend-specific implementation
        let storage = self
            .storage()
            .sub_at_indices(self.layout(), indices, value)?;

        let op = BackpropOp::none();
        Ok(from_storage(storage, self.shape().clone(), op, false))
    }

    /// In-place sparse subtraction - mutates the tensor directly without cloning.
    ///
    /// **This is 20x+ faster than `sub_at_indices()` for large tensors!**
    ///
    /// # Performance (150K vocab, 50 indices):
    /// - `sub_at_indices()`: ~21ms (20ms clone + 1ms kernel)
    /// - `sub_at_indices_mut()`: ~1ms (no clone, just kernel)
    ///
    /// # Important Notes
    /// - Requires mutable tensor
    /// - Cannot be used with tensors that have backprop tracking
    /// - Modifies tensor in-place (no new tensor returned)
    /// - **CUDA/CPU only** - Metal not yet supported
    ///
    /// # Arguments
    /// * `indices` - Indices in the last dimension to update (can contain duplicates)
    /// * `value` - Scalar value to subtract from each indexed element
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let mut logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let penalty_tokens = vec![10u32, 25u32, 10u32];
    /// logits.sub_at_indices_mut(&penalty_tokens, 5.0)?;  // 20x faster!
    /// # Ok(())
    /// # }
    /// ```
    pub fn sub_at_indices_mut(&mut self, indices: &[u32], value: f32) -> Result<()> {
        if self.elem_count() == 0 {
            return Ok(());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "sub_at_indices_mut",
                });
            }
        }

        // Get mutable access to storage
        let layout = self.layout().clone();
        let mut storage = self.storage_mut();

        // Dispatch to backend-specific in-place implementation
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(cuda_storage) = &mut *storage {
            return cuda_storage.sub_at_indices_mut(&layout, indices, value);
        }

        if let Storage::Cpu(cpu_storage) = &mut *storage {
            return cpu_storage.sub_at_indices_mut(&layout, indices, value);
        }

        // Metal not yet supported for in-place mutation
        Err(Error::Msg(
            "sub_at_indices_mut is only supported on CPU and CUDA backends".to_string(),
        ))
    }

    /// In-place sparse subtraction with per-index values - mutates the tensor directly without cloning.
    ///
    /// **Each index gets its own value to subtract: `tensor[indices[i]] -= values[i]`**
    ///
    /// This is useful for applying different penalties to different tokens in language models,
    /// such as frequency-based penalties where each token has a different penalty value.
    ///
    /// **This is 20x+ faster than cloning + sub_at_indices() for large tensors!**
    ///
    /// # Performance (150K vocab, 50 indices):
    /// - Clone + loop: ~21ms (20ms clone + 1ms updates)
    /// - `sub_at_indices_mut_with_values()`: ~1ms (no clone, just kernel)
    ///
    /// # Important Notes
    /// - Requires mutable tensor
    /// - Cannot be used with tensors that have backprop tracking
    /// - Modifies tensor in-place (no new tensor returned)
    /// - **CUDA/CPU only** - Metal not yet supported
    /// - `indices.len()` must equal `values.len()`
    ///
    /// # Arguments
    /// * `indices` - Indices in the last dimension to update (can contain duplicates)
    /// * `values` - Values to subtract from each corresponding index
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let mut logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let penalty_tokens = vec![10u32, 25u32, 42u32];
    /// let penalty_values = vec![5.0f32, 3.0, 7.0]; // Different penalty for each token
    /// logits.sub_at_indices_mut_with_values(&penalty_tokens, &penalty_values)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn sub_at_indices_mut_with_values(
        &mut self,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        if indices.len() != values.len() {
            return Err(Error::Msg(format!(
                "indices and values must have the same length, got {} and {}",
                indices.len(),
                values.len()
            )));
        }

        if self.elem_count() == 0 {
            return Ok(());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "sub_at_indices_mut_with_values",
                });
            }
        }

        // Get mutable access to storage
        let layout = self.layout().clone();
        let mut storage = self.storage_mut();

        // Dispatch to backend-specific in-place implementation
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(cuda_storage) = &mut *storage {
            return cuda_storage.sub_at_indices_mut_with_values(&layout, indices, values);
        }

        if let Storage::Cpu(cpu_storage) = &mut *storage {
            return cpu_storage.sub_at_indices_mut_with_values(&layout, indices, values);
        }

        // Metal not yet supported for in-place mutation
        Err(Error::Msg(
            "sub_at_indices_mut_with_values is only supported on CPU and CUDA backends".to_string(),
        ))
    }

    /// Divide tensor values at specific indices by a scalar.
    ///
    /// This operation is useful for applying repeat penalties in language models:
    /// `logits[token_ids] /= repeat_penalty` for penalizing repeated tokens.
    ///
    /// # Performance
    /// - Uses sparse GPU operations (only updates specified indices)
    /// - ~40-50x faster than full tensor iteration for sparse updates
    /// - For better performance on large tensors, use `div_at_indices_mut()`
    ///
    /// # Arguments
    /// * `indices` - Token IDs in the last dimension to update (duplicates allowed)
    /// * `value` - Divisor value (typically 1.0-1.5 for repeat penalty)
    ///
    /// # Returns
    /// New tensor with values at indices divided by the value.
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let repeated_tokens = vec![10u32, 25u32, 10u32]; // Token 10 penalized twice
    /// let result = logits.div_at_indices(&repeated_tokens, 1.1)?; // Reduce probability by 10%
    /// # Ok(())
    /// # }
    /// ```
    pub fn div_at_indices(&self, indices: &[u32], value: f32) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "div_at_indices",
                });
            }
        }

        // Dispatch to backend-specific implementation
        let storage = self
            .storage()
            .div_at_indices(self.layout(), indices, value)?;

        let op = BackpropOp::none();
        Ok(from_storage(storage, self.shape().clone(), op, false))
    }

    /// In-place sparse division - mutates the tensor directly without cloning.
    ///
    /// **This is 20x+ faster than `div_at_indices()` for large tensors!**
    ///
    /// # Performance (150K vocab, 50 indices):
    /// - `div_at_indices()`: ~21ms (20ms clone + 1ms kernel)
    /// - `div_at_indices_mut()`: ~1ms (no clone, just kernel)
    ///
    /// # Important Notes
    /// - Requires mutable tensor
    /// - Cannot be used with tensors that have backprop tracking
    /// - Modifies tensor in-place (no new tensor returned)
    /// - **CUDA/CPU only** - Metal not yet supported
    ///
    /// # Arguments
    /// * `indices` - Indices in the last dimension to update (can contain duplicates)
    /// * `value` - Divisor value (typically 1.0-1.5 for repeat penalty)
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let mut logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let repeated_tokens = vec![10u32, 25u32, 10u32];
    /// logits.div_at_indices_mut(&repeated_tokens, 1.1)?;  // 20x faster!
    /// # Ok(())
    /// # }
    /// ```
    pub fn div_at_indices_mut(&mut self, indices: &[u32], value: f32) -> Result<()> {
        if self.elem_count() == 0 {
            return Ok(());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "div_at_indices_mut",
                });
            }
        }

        // Get mutable access to storage
        let layout = self.layout().clone();
        let mut storage = self.storage_mut();

        // Dispatch to backend-specific in-place implementation
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(cuda_storage) = &mut *storage {
            return cuda_storage.div_at_indices_mut(&layout, indices, value);
        }

        if let Storage::Cpu(cpu_storage) = &mut *storage {
            return cpu_storage.div_at_indices_mut(&layout, indices, value);
        }

        // Metal not yet supported for in-place mutation
        Err(Error::Msg(
            "div_at_indices_mut is only supported on CPU and CUDA backends".to_string(),
        ))
    }

    /// Add a scalar value to tensor values at specific indices.
    ///
    /// This operation is useful for applying bias operations in language models:
    /// `logits[token_ids] += bias` for boosting specific token probabilities.
    ///
    /// # Performance
    /// - Uses sparse GPU operations (only updates specified indices)
    /// - For better performance on large tensors, use `add_at_indices_mut()`
    ///
    /// # Arguments
    /// * `indices` - Token IDs in the last dimension to update (duplicates allowed)
    /// * `value` - Value to add
    ///
    /// # Returns
    /// New tensor with values at indices increased by the value.
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let boost_tokens = vec![10u32, 25u32];
    /// let result = logits.add_at_indices(&boost_tokens, 5.0)?; // Add +5 bias
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_at_indices(&self, indices: &[u32], value: f32) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "add_at_indices",
                });
            }
        }

        // For non-mutable version, clone and mutate
        let mut result = self.clone();
        result.add_at_indices_mut(indices, value)?;
        Ok(result)
    }

    /// In-place sparse addition - mutates the tensor directly without cloning.
    ///
    /// **This is 20x+ faster than `add_at_indices()` for large tensors!**
    ///
    /// # Performance (150K vocab, 50 indices):
    /// - `add_at_indices()`: ~21ms (20ms clone + 1ms kernel)
    /// - `add_at_indices_mut()`: ~1ms (no clone, just kernel)
    ///
    /// # Important Notes
    /// - Requires mutable tensor
    /// - Cannot be used with tensors that have backprop tracking
    /// - Modifies tensor in-place (no new tensor returned)
    /// - **CUDA/CPU only** - Metal not yet supported
    ///
    /// # Arguments
    /// * `indices` - Indices in the last dimension to update (can contain duplicates)
    /// * `value` - Value to add
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let mut logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let boost_tokens = vec![10u32, 25u32];
    /// logits.add_at_indices_mut(&boost_tokens, 5.0)?;  // 20x faster!
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_at_indices_mut(&mut self, indices: &[u32], value: f32) -> Result<()> {
        if self.elem_count() == 0 {
            return Ok(());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "add_at_indices_mut",
                });
            }
        }

        // Get mutable access to storage
        let layout = self.layout().clone();
        let mut storage = self.storage_mut();

        // Dispatch to backend-specific in-place implementation
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(cuda_storage) = &mut *storage {
            return cuda_storage.add_at_indices_mut(&layout, indices, value);
        }

        if let Storage::Cpu(cpu_storage) = &mut *storage {
            return cpu_storage.add_at_indices_mut(&layout, indices, value);
        }

        // Metal not yet supported for in-place mutation
        Err(Error::Msg(
            "add_at_indices_mut is only supported on CPU and CUDA backends".to_string(),
        ))
    }

    /// Multiply tensor values at specific indices by a scalar.
    ///
    /// This operation is useful for applying scaling operations in language models:
    /// `logits[token_ids] *= scale_factor` for boosting or reducing specific token probabilities.
    ///
    /// # Performance
    /// - Uses sparse GPU operations (only updates specified indices)
    /// - For better performance on large tensors, use `mul_at_indices_mut()`
    ///
    /// # Arguments
    /// * `indices` - Token IDs in the last dimension to update (duplicates allowed)
    /// * `value` - Multiplier value
    ///
    /// # Returns
    /// New tensor with values at indices multiplied by the value.
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let boost_tokens = vec![10u32, 25u32];
    /// let result = logits.mul_at_indices(&boost_tokens, 2.0)?; // Double probability
    /// # Ok(())
    /// # }
    /// ```
    pub fn mul_at_indices(&self, indices: &[u32], value: f32) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "mul_at_indices",
                });
            }
        }

        // For non-mutable version, clone and mutate
        let mut result = self.clone();
        result.mul_at_indices_mut(indices, value)?;
        Ok(result)
    }

    /// In-place sparse multiplication - mutates the tensor directly without cloning.
    ///
    /// **This is 20x+ faster than `mul_at_indices()` for large tensors!**
    ///
    /// # Performance (150K vocab, 50 indices):
    /// - `mul_at_indices()`: ~21ms (20ms clone + 1ms kernel)
    /// - `mul_at_indices_mut()`: ~1ms (no clone, just kernel)
    ///
    /// # Important Notes
    /// - Requires mutable tensor
    /// - Cannot be used with tensors that have backprop tracking
    /// - Modifies tensor in-place (no new tensor returned)
    /// - **CUDA/CPU only** - Metal not yet supported
    ///
    /// # Arguments
    /// * `indices` - Indices in the last dimension to update (can contain duplicates)
    /// * `value` - Multiplier value
    ///
    /// # Example
    /// ```
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let mut logits = Tensor::zeros((2, 1000), candle_core::DType::F32, &Device::Cpu)?;
    /// let boost_tokens = vec![10u32, 25u32];
    /// logits.mul_at_indices_mut(&boost_tokens, 2.0)?;  // 20x faster!
    /// # Ok(())
    /// # }
    /// ```
    pub fn mul_at_indices_mut(&mut self, indices: &[u32], value: f32) -> Result<()> {
        if self.elem_count() == 0 {
            return Ok(());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "mul_at_indices_mut",
                });
            }
        }

        // Get mutable access to storage
        let layout = self.layout().clone();
        let mut storage = self.storage_mut();

        // Dispatch to backend-specific in-place implementation
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(cuda_storage) = &mut *storage {
            return cuda_storage.mul_at_indices_mut(&layout, indices, value);
        }

        if let Storage::Cpu(cpu_storage) = &mut *storage {
            return cpu_storage.mul_at_indices_mut(&layout, indices, value);
        }

        // Metal not yet supported for in-place mutation
        Err(Error::Msg(
            "mul_at_indices_mut is only supported on CPU and CUDA backends".to_string(),
        ))
    }

    /// Apply repeat penalty to logits at specific token indices in-place.
    ///
    /// This is a highly optimized GPU kernel that applies different operations based on logit sign:
    /// - **Positive logits**: Divided by penalty (reduces probability)
    /// - **Negative/zero logits**: Multiplied by penalty (reduces probability)
    ///
    /// This combines the logic of `div_at_indices_mut` and `mul_at_indices_mut` into a single
    /// GPU kernel pass, avoiding a CPU round-trip to read logit values and two separate kernel launches.
    ///
    /// # Performance
    /// Compared to the CPU-based approach (read logits, separate positive/negative, two kernel calls):
    /// - **3x+ faster** for typical repeat penalty scenarios (50-200 tokens)
    /// - Single kernel launch vs. CPU→GPU→CPU→GPU roundtrip
    /// - No memory allocation for positive/negative token lists
    ///
    /// # Arguments
    /// * `indices` - Token IDs to penalize (typically recent assistant tokens)
    /// * `penalty` - Penalty value (e.g., 1.1 for 10% reduction). Use 1.0 for no-op.
    ///
    /// # Example
    /// ```ignore
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let mut logits = Tensor::randn(0f32, 1f32, (2, 50000), &Device::cuda_if_available(0)?)?;
    /// let repeated_tokens = vec![42u32, 123u32, 456u32];
    /// logits.repeat_penalty_mut(&repeated_tokens, 1.15)?;  // 15% penalty
    /// # Ok(())
    /// # }
    /// ```
    pub fn repeat_penalty_mut(&mut self, indices: &[u32], penalty: f32) -> Result<()> {
        if self.elem_count() == 0 || penalty == 1.0 {
            return Ok(());
        }

        // Validate indices
        let dims = self.dims();
        let vocab_size = dims[dims.len() - 1];

        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(Error::DimOutOfRange {
                    shape: self.shape().clone(),
                    dim: (dims.len() - 1) as i32,
                    op: "repeat_penalty_mut",
                });
            }
        }

        // Get mutable access to storage
        #[cfg(feature = "cuda")]
        let layout = self.layout().clone();
        let mut storage = self.storage_mut();

        // Dispatch to backend-specific in-place implementation
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(cuda_storage) = &mut *storage {
            return cuda_storage.repeat_penalty_mut(&layout, indices, penalty);
        }

        if let Storage::Cpu(cpu_storage) = &mut *storage {
            // CPU fallback: read values and apply penalty based on sign
            match cpu_storage {
                crate::CpuStorage::F32(data) => {
                    for &idx in indices {
                        let idx = idx as usize;
                        let current = data[idx];
                        data[idx] = if current > 0.0 {
                            current / penalty
                        } else {
                            current * penalty
                        };
                    }
                    return Ok(());
                }
                crate::CpuStorage::F64(data) => {
                    let penalty_f64 = penalty as f64;
                    for &idx in indices {
                        let idx = idx as usize;
                        let current = data[idx];
                        data[idx] = if current > 0.0 {
                            current / penalty_f64
                        } else {
                            current * penalty_f64
                        };
                    }
                    return Ok(());
                }
                _ => {
                    return Err(Error::UnsupportedDTypeForOp(
                        cpu_storage.dtype(),
                        "repeat_penalty_mut",
                    )
                    .bt());
                }
            }
        } // Metal not yet supported for in-place mutation
        Err(Error::Msg(
            "repeat_penalty_mut is only supported on CPU and CUDA backends".to_string(),
        ))
    }
}
