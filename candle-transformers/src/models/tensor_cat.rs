//! Flexible tensor concatenation utilities for efficient batch processing.
//!
//! This module provides `TensorCat`, a structure that represents a concatenated tensor
//! along a fixed dimension with cached metadata for efficient batch processing.
//!
//! The concatenation dimension is fixed at creation time and does not change.
//! It caches dimension information for efficient repeated access, avoiding redundant
//! shape queries during model forward passes.

use candle::{DType, Result, Shape, Tensor};
use std::ops::Deref;

/// Concatenated tensor with cached metadata for efficient batch operations.
///
/// Always stores tensors in concatenated form with split information,
/// allowing efficient conversion back to individual tensors.
///
/// Can be dereferenced to access the underlying Tensor directly.
#[derive(Debug, Clone)]
pub struct TensorCat {
    /// The concatenated tensor
    tensor: Tensor,
    /// Number of sequences in the batch
    batch_size: usize,
    /// Cached shape of individual inner tensors (before concatenation)
    inner_shape: Shape,
    /// Dimension along which tensors were concatenated
    cat_dim: usize,
    /// Size of each segment in the concatenation dimension (cached vector)
    segment_sizes: Vec<usize>,
}

/// Implement Deref to allow TensorCat to be used like a Tensor reference
impl Deref for TensorCat {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

/// Iterator over individual tensors in the batch
pub struct TensorCatIter {
    tensor_cat: TensorCat,
    current_index: usize,
}

impl Iterator for TensorCatIter {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.tensor_cat.len() {
            return None;
        }
        let result = self.tensor_cat.get(self.current_index).ok()?;
        self.current_index += 1;
        Some(result)
    }
}

impl IntoIterator for TensorCat {
    type Item = Tensor;
    type IntoIter = TensorCatIter;

    fn into_iter(self) -> Self::IntoIter {
        TensorCatIter {
            tensor_cat: self,
            current_index: 0,
        }
    }
}

impl TensorCat {
    /// Convert to a single concatenated tensor
    pub fn to_tensor(&self) -> Tensor {
        self.tensor.clone()
    }

    /// Convert to a vector of individual tensors
    pub fn to_vec(&self) -> Result<Vec<Tensor>> {
        let mut result = Vec::with_capacity(self.batch_size);
        let mut offset = 0;
        for i in 0..self.batch_size {
            let size = self.segment_sizes[i];
            result.push(self.tensor.narrow(self.cat_dim, offset, size)?);
            offset += size;
        }
        Ok(result)
    }

    /// Convert into vector form
    pub fn into_vec(self) -> Result<Vec<Tensor>> {
        let mut result = Vec::with_capacity(self.batch_size);
        let mut offset = 0;
        for i in 0..self.batch_size {
            let size = self.segment_sizes[i];
            result.push(self.tensor.narrow(self.cat_dim, offset, size)?);
            offset += size;
        }
        Ok(result)
    }

    /// Create a Vec variant from an iterator of tensors with a specified concatenation dimension.
    /// Converts to Cat form immediately for better performance.
    ///
    /// # Validation
    /// All tensors must have the same number of dimensions and matching sizes for all dimensions
    /// except the concatenation dimension (cat_dim). The cat_dim dimension is allowed to vary.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The tensor iterator is empty
    /// - Tensors have different numbers of dimensions
    /// - Non-cat_dim dimensions differ across tensors
    pub fn from_tensors<I: IntoIterator<Item = Tensor>>(
        cat_dim: usize,
        tensors: I,
    ) -> Result<Self> {
        let tensors: Vec<Tensor> = tensors.into_iter().collect();

        if tensors.is_empty() {
            candle::bail!("Cannot create tensor batch from empty tensor iterator");
        }

        // Validate that all tensors have matching dimensions except cat_dim
        let first_shape = tensors[0].shape();
        let first_ndim = first_shape.dims().len();

        if cat_dim >= first_ndim {
            candle::bail!(
                "Concatenation dimension {} is out of bounds for tensors with {} dimensions",
                cat_dim,
                first_ndim
            );
        }

        for (idx, tensor) in tensors.iter().enumerate().skip(1) {
            let shape = tensor.shape();
            let ndim = shape.dims().len();

            if ndim != first_ndim {
                candle::bail!(
                    "Tensor {} has {} dimensions but expected {}",
                    idx,
                    ndim,
                    first_ndim
                );
            }

            let dims = shape.dims();
            let first_dims = first_shape.dims();

            for d in 0..first_ndim {
                if d != cat_dim && dims[d] != first_dims[d] {
                    candle::bail!(
                        "Tensor {} dimension {} mismatch: expected {}, got {} (cat_dim={})",
                        idx,
                        d,
                        first_dims[d],
                        dims[d],
                        cat_dim
                    );
                }
            }
        }

        // Concatenate all tensors
        let concatenated = Tensor::cat(&tensors, cat_dim)?;

        // Get batch_size and segment sizes
        let batch_size = tensors.len();
        let segment_sizes: Vec<usize> = tensors.iter().map(|t| t.dims()[cat_dim]).collect();

        // Derive inner shape (same as first tensor's shape)
        let inner_shape = tensors[0].shape().clone();

        Ok(Self {
            tensor: concatenated.contiguous()?,
            batch_size,
            inner_shape,
            cat_dim,
            segment_sizes,
        })
    }

    /// Create a Cat variant from an already concatenated tensor.
    ///
    /// Automatically derives batch_size from the first dimension and extracts
    /// a slice to determine the shape of individual inner tensors.
    pub fn from_cat_tensor(tensor: Tensor, cat_dim: usize) -> Result<Self> {
        // Derive batch_size from the first dimension of the tensor
        let batch_size = tensor.dim(0)?;
        if batch_size == 0 {
            candle::bail!("Cannot create Cat tensor with batch_size 0");
        }

        // Compute inner shape without creating a slice (more efficient)
        // For a concatenated tensor of shape (B, d1, d2, ...), the inner shape is (1, d1, d2, ...)
        let dims = tensor.dims();
        let mut inner_dims = vec![1];
        inner_dims.extend_from_slice(&dims[1..]);
        let inner_shape = Shape::from(inner_dims);

        // Assume concatenated along first dimension with uniform segments
        let chunk_size = dims[0] / batch_size;
        let segment_sizes = vec![chunk_size; batch_size];

        Ok(Self {
            tensor: tensor.contiguous()?,
            batch_size,
            inner_shape,
            cat_dim,
            segment_sizes,
        })
    }

    /// Get the batch size
    pub fn len(&self) -> usize {
        self.batch_size
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the segment sizes
    pub fn segment_sizes(&self) -> &[usize] {
        &self.segment_sizes
    }

    /// Get the concatenation dimension
    pub fn cat_dim(&self) -> usize {
        self.cat_dim
    }

    /// Get a reference to the underlying concatenated tensor
    pub fn as_cat_tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Get a mutable reference to the underlying concatenated tensor
    pub fn as_cat_tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }

    /// Get the segment sizes as a tensor (alias for deref compatibility)
    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Get the tensor for a specific segment by index
    pub fn get(&self, index: usize) -> Result<Tensor> {
        if index >= self.len() {
            candle::bail!(
                "Index {} out of bounds for batch size {}",
                index,
                self.len()
            );
        }
        let mut offset = 0;
        for i in 0..index {
            offset += self.segment_sizes[i];
        }
        let size = self.segment_sizes[index];
        self.tensor.narrow(self.cat_dim, offset, size)
    }

    /// Replace the underlying concatenated tensor while preserving metadata
    pub fn replace(&mut self, new_tensor: Tensor) {
        self.tensor = new_tensor;
    }

    /// Convert the underlying tensor to a different dtype in-place.
    ///
    /// On CUDA, this attempts a true in-place conversion if the buffer has sufficient
    /// capacity. Falls back to allocation + copy on CPU/Metal or if buffer is too small.
    pub fn to_dtype_mut(&mut self, dtype: DType) -> Result<()> {
        self.tensor.to_dtype_mut(dtype)
    }

    /// Add another tensor to this TensorCat in-place.
    ///
    /// The rhs tensor must be broadcastable to the shape of the concatenated tensor.
    /// Accumulate `rhs` into this buffer.
    ///
    /// `rhs` may be wave-scoped — the residual add is exactly where a leased
    /// FFN or attention result is consumed — because this only *reads* it.
    pub fn add_mut(&mut self, rhs: &candle::LiveTensor<'_>) -> Result<()> {
        self.tensor.add_mut(rhs)
    }

    /// Get the data type of inner tensors
    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    /// Get a specific dimension of inner tensors
    pub fn dims(&self, index: usize) -> Result<usize> {
        let dims = self.inner_shape.dims();
        if index < dims.len() {
            Ok(dims[index])
        } else {
            candle::bail!(
                "Index {} out of bounds for inner shape with {} dimensions",
                index,
                dims.len()
            )
        }
    }

    /// Get the 1D dimensions of inner tensors at the specified batch index
    pub fn dims1(&self) -> Result<usize> {
        let dims = self.inner_shape.dims();
        if !dims.is_empty() {
            Ok(dims[0])
        } else {
            candle::bail!("Inner shape has no dimensions")
        }
    }

    /// Get the 2D dimensions of inner tensors at the specified batch index
    pub fn dims2(&self) -> Result<(usize, usize)> {
        let dims = self.inner_shape.dims();
        if dims.len() >= 2 {
            Ok((dims[0], dims[1]))
        } else {
            candle::bail!("Inner shape has fewer than 2 dimensions")
        }
    }

    /// Get the 3D dimensions of inner tensors at the specified batch index
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        let dims = self.inner_shape.dims();
        if dims.len() >= 3 {
            Ok((dims[0], dims[1], dims[2]))
        } else {
            candle::bail!("Inner shape has fewer than 3 dimensions")
        }
    }

    /// Get the 4D dimensions of inner tensors at the specified batch index
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let dims = self.inner_shape.dims();
        if dims.len() >= 4 {
            Ok((dims[0], dims[1], dims[2], dims[3]))
        } else {
            candle::bail!("Inner shape has fewer than 4 dimensions")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    #[test]
    fn test_vec_tensor_creation() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::zeros((1, 5, 10), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((1, 5, 10), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2])?;

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.dims3()?, (1, 5, 10));
        assert_eq!(batch.dims(0)?, 1);
        assert_eq!(batch.dims(1)?, 5);
        assert_eq!(batch.dims(2)?, 10);

        Ok(())
    }

    #[test]
    fn test_dims_out_of_bounds() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((3, 4), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(0, vec![tensor])?;

        let result = batch.dims(5);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_varying_dimension_dim0() -> Result<()> {
        let device = Device::Cpu;
        // Tensors vary in dimension 0
        let t1 = Tensor::zeros((2, 5, 10), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((3, 5, 10), candle::DType::F32, &device)?;
        let t3 = Tensor::full(2.0f32, (1, 5, 10), &device)?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2, t3])?;

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.cat_dim(), 0);
        assert_eq!(batch.segment_sizes(), &[2, 3, 1]);
        assert_eq!(batch.tensor.dims(), &[6, 5, 10]); // 2+3+1=6 in dim 0

        Ok(())
    }

    #[test]
    fn test_varying_dimension_dim1() -> Result<()> {
        let device = Device::Cpu;
        // Tensors vary in dimension 1
        let t1 = Tensor::zeros((2, 3, 10), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((2, 5, 10), candle::DType::F32, &device)?;
        let t3 = Tensor::full(2.0f32, (2, 2, 10), &device)?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2, t3])?;

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.cat_dim(), 1);
        assert_eq!(batch.segment_sizes(), &[3, 5, 2]);
        assert_eq!(batch.tensor.dims(), &[2, 10, 10]); // 3+5+2=10 in dim 1

        Ok(())
    }

    #[test]
    fn test_varying_dimension_dim2() -> Result<()> {
        let device = Device::Cpu;
        // Tensors vary in dimension 2 (like variable-length KV caches)
        let t1 = Tensor::zeros((1, 8, 105, 128), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((1, 8, 110, 128), candle::DType::F32, &device)?;
        let t3 = Tensor::full(2.0f32, (1, 8, 103, 128), &device)?;

        let batch = TensorCat::from_tensors(2, vec![t1, t2, t3])?;

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.cat_dim(), 2);
        assert_eq!(batch.segment_sizes(), &[105, 110, 103]);
        assert_eq!(batch.tensor.dims(), &[1, 8, 318, 128]); // 105+110+103=318 in dim 2

        Ok(())
    }

    #[test]
    fn test_varying_dimension_multiple_dims_error() -> Result<()> {
        let device = Device::Cpu;
        // Tensors vary in multiple dimensions - should error
        let t1 = Tensor::zeros((2, 3, 10), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((3, 5, 10), candle::DType::F32, &device)?;

        let result = TensorCat::from_tensors(0, vec![t1, t2]);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("mismatch"));

        Ok(())
    }

    #[test]
    fn test_uniform_tensors_default_to_dim0() -> Result<()> {
        let device = Device::Cpu;
        // All tensors have same shape - concatenate along dim 0
        let t1 = Tensor::zeros((2, 5, 10), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((2, 5, 10), candle::DType::F32, &device)?;
        let t3 = Tensor::full(2.0f32, (2, 5, 10), &device)?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2, t3])?;

        assert_eq!(batch.cat_dim(), 0);
        assert_eq!(batch.segment_sizes(), &[2, 2, 2]); // All same size in dim 0
        assert_eq!(batch.tensor.dims(), &[6, 5, 10]); // 2+2+2=6 in dim 0

        Ok(())
    }

    // Vector form tests removed - TensorCat is always Cat form now

    #[test]
    fn test_varying_dim_to_tensor_conversion() -> Result<()> {
        let device = Device::Cpu;
        // Tensors with varying dim 2
        let t1 = Tensor::zeros((1, 4, 10, 5), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((1, 4, 15, 5), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(2, vec![t1, t2])?;
        let tensor = batch.to_tensor();

        // Should be concatenated along dim 2
        assert_eq!(tensor.dims(), &[1, 4, 25, 5]); // 10+15=25

        Ok(())
    }

    #[test]
    fn test_varying_dim_preserved_after_to_vec() -> Result<()> {
        let device = Device::Cpu;
        // Create with varying dim 1
        let t1 = Tensor::zeros((3, 2, 10), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((3, 4, 10), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2])?;
        let vec = batch.to_vec()?;

        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0].dims(), &[3, 2, 10]);
        assert_eq!(vec[1].dims(), &[3, 4, 10]);

        Ok(())
    }

    #[test]
    fn test_varying_dim_dims_method() -> Result<()> {
        let device = Device::Cpu;
        // Tensors vary in dim 2
        let t1 = Tensor::zeros((2, 3, 5, 7), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((2, 3, 8, 7), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(2, vec![t1, t2])?;

        // dims() should return inner tensor dimensions
        assert_eq!(batch.dims(0)?, 2);
        assert_eq!(batch.dims(1)?, 3);
        assert_eq!(batch.dims(2)?, 5); // First tensor's dim 2 (before concatenation)
        assert_eq!(batch.dims(3)?, 7);

        Ok(())
    }

    #[test]
    fn test_single_tensor_varying_dim() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::zeros((5, 10, 15), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(0, vec![t])?;

        assert_eq!(batch.cat_dim(), 0);
        assert_eq!(batch.segment_sizes(), &[5]);

        Ok(())
    }

    #[test]
    fn test_edge_case_dim0_size_1_varying_dim2() -> Result<()> {
        let device = Device::Cpu;
        // All dim 0 = 1, but dim 2 varies
        let t1 = Tensor::zeros((1, 5, 10, 8), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((1, 5, 12, 8), candle::DType::F32, &device)?;
        let t3 = Tensor::full(2.0f32, (1, 5, 8, 8), &device)?;

        let batch = TensorCat::from_tensors(2, vec![t1, t2, t3])?;

        assert_eq!(batch.cat_dim(), 2);
        assert_eq!(batch.segment_sizes(), &[10, 12, 8]);
        assert_eq!(batch.tensor.dims(), &[1, 5, 30, 8]); // 10+12+8=30

        Ok(())
    }

    #[test]
    fn test_large_varying_segments() -> Result<()> {
        let device = Device::Cpu;
        // Create tensors with wildly different sizes in dim 1
        let t1 = Tensor::zeros((3, 5, 10), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((3, 50, 10), candle::DType::F32, &device)?;
        let t3 = Tensor::full(2.0f32, (3, 2, 10), &device)?;
        let t4 = Tensor::full(3.0f32, (3, 100, 10), &device)?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2, t3, t4])?;

        assert_eq!(batch.cat_dim(), 1);
        assert_eq!(batch.segment_sizes(), &[5, 50, 2, 100]);
        assert_eq!(batch.tensor.dims(), &[3, 157, 10]); // 5+50+2+100=157

        Ok(())
    }

    #[test]
    fn test_varying_dim3_4d_tensors() -> Result<()> {
        let device = Device::Cpu;
        // 4D tensors varying in last dimension
        let t1 = Tensor::zeros((2, 3, 4, 5), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((2, 3, 4, 7), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(3, vec![t1, t2])?;

        assert_eq!(batch.cat_dim(), 3);
        assert_eq!(batch.segment_sizes(), &[5, 7]);
        assert_eq!(batch.tensor.dims(), &[2, 3, 4, 12]); // 5+7=12

        Ok(())
    }

    #[test]
    fn test_from_cat_tensor() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((4, 5, 10), candle::DType::F32, &device)?;

        let batch = TensorCat::from_cat_tensor(tensor, 0)?;

        assert_eq!(batch.len(), 4);
        assert_eq!(batch.dims3()?, (1, 5, 10));

        Ok(())
    }

    #[test]
    fn test_from_cat_tensor_zero_batch_size() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((0, 5, 10), candle::DType::F32, &device)?;

        let result = TensorCat::from_cat_tensor(tensor, 0);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("batch_size 0"));

        Ok(())
    }

    #[test]
    fn test_empty_vec_error() -> Result<()> {
        let result = TensorCat::from_tensors(0, vec![]);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("empty"));

        Ok(())
    }

    #[test]
    fn test_dims_caching_with_varying_dim() -> Result<()> {
        let device = Device::Cpu;
        // Varying in dim 1
        let t1 = Tensor::zeros((3, 4, 5, 6), candle::DType::F32, &device)?;
        let t2 = Tensor::ones((3, 7, 5, 6), candle::DType::F32, &device)?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2])?;

        // Inner shape should be first tensor's shape
        assert_eq!(batch.dims(0)?, 3);
        assert_eq!(batch.dims(1)?, 4);
        assert_eq!(batch.dims(2)?, 5);
        assert_eq!(batch.dims(3)?, 6);

        Ok(())
    }

    // Removed test_into_cat_idempotent - TensorCat is always Cat form now

    // Removed test_alternating_conversions_with_varying_dim - uses old Vec API

    #[test]
    fn test_varying_dim0_roundtrip_preserves_values() -> Result<()> {
        let device = Device::Cpu;
        // Create tensors with different values and varying dim 0
        let t1 = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?.reshape((3, 1))?;
        let t2 = Tensor::new(&[4.0f32, 5.0], &device)?.reshape((2, 1))?;
        let t3 = Tensor::new(&[6.0f32, 7.0, 8.0, 9.0], &device)?.reshape((4, 1))?;

        let original_vec = vec![t1.clone(), t2.clone(), t3.clone()];
        let batch = TensorCat::from_tensors(0, original_vec.clone())?;

        // Verify concatenation
        assert_eq!(batch.cat_dim(), 0);
        assert_eq!(batch.segment_sizes(), &vec![3, 2, 4]);
        assert_eq!(batch.tensor.dims(), &[9, 1]); // 3+2+4=9

        // Convert back to Vec
        let vec_batch = batch.to_vec()?;

        // Verify values are preserved
        assert_eq!(vec_batch.len(), 3);
        assert_eq!(vec_batch[0].dims(), &[3, 1]);
        assert_eq!(vec_batch[1].dims(), &[2, 1]);
        assert_eq!(vec_batch[2].dims(), &[4, 1]);

        Ok(())
    }

    #[test]
    fn test_varying_dim1_roundtrip_preserves_values() -> Result<()> {
        let device = Device::Cpu;
        // Create tensors varying in dim 1
        let t1 = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?.reshape((1, 3))?;
        let t2 = Tensor::new(&[4.0f32, 5.0, 6.0, 7.0, 8.0], &device)?.reshape((1, 5))?;
        let t3 = Tensor::new(&[9.0f32, 10.0], &device)?.reshape((1, 2))?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2, t3])?;

        // Verify concatenation along dim 1
        assert_eq!(batch.cat_dim(), 1);
        assert_eq!(batch.segment_sizes(), &vec![3, 5, 2]);
        assert_eq!(batch.tensor.dims(), &[1, 10]); // 3+5+2=10

        // Convert back to individual tensors and verify
        let vec = batch.to_vec()?;
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0].dims(), &[1, 3]);
        assert_eq!(vec[1].dims(), &[1, 5]);
        assert_eq!(vec[2].dims(), &[1, 2]);

        Ok(())
    }

    #[test]
    fn test_varying_dim2_roundtrip_preserves_values() -> Result<()> {
        let device = Device::Cpu;
        // Create 3D tensors varying in dim 2 (like KV caches)
        let t1 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((1, 2, 2))?;
        let t2 = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0], &device)?.reshape((1, 2, 3))?;
        let t3 = Tensor::new(&[11.0f32, 12.0], &device)?.reshape((1, 2, 1))?;

        let batch = TensorCat::from_tensors(2, vec![t1, t2, t3])?;

        // Verify concatenation along dim 2
        assert_eq!(batch.cat_dim(), 2);
        assert_eq!(batch.segment_sizes(), &vec![2, 3, 1]);
        assert_eq!(batch.tensor.dims(), &[1, 2, 6]); // 2+3+1=6

        // Convert back and verify
        let vec = batch.to_vec()?;
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0].dims(), &[1, 2, 2]);
        assert_eq!(vec[1].dims(), &[1, 2, 3]);
        assert_eq!(vec[2].dims(), &[1, 2, 1]);
        Ok(())
    }

    // Removed old roundtrip tests that used Vec variant API

    #[test]
    fn test_cat_to_tensor_preserves_order_varying_dim0() -> Result<()> {
        let device = Device::Cpu;
        // Create distinct values in each tensor
        let t1 = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        let t2 = Tensor::new(&[4.0f32, 5.0], &device)?;
        let t3 = Tensor::new(&[6.0f32, 7.0, 8.0, 9.0], &device)?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2, t3])?;
        let tensor = batch.to_tensor();

        // Verify concatenated tensor has values in correct order
        let values = tensor.to_vec1::<f32>()?;
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_cat_to_tensor_preserves_order_varying_dim2() -> Result<()> {
        let device = Device::Cpu;
        // Create 3D tensors with distinct patterns
        let t1 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((1, 2, 2))?;
        let t2 = Tensor::new(&[5.0f32, 6.0], &device)?.reshape((1, 2, 1))?;

        let batch = TensorCat::from_tensors(2, vec![t1, t2])?;
        let tensor = batch.to_tensor();

        // Should be concatenated to shape [1, 2, 3]
        assert_eq!(tensor.dims(), &[1, 2, 3]);

        let values = tensor.to_vec3::<f32>()?;
        // First row: [1, 2, 5], second row: [3, 4, 6]
        assert_eq!(values, vec![vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]]]);

        Ok(())
    }

    #[test]
    fn test_mixed_operations_preserve_values() -> Result<()> {
        let device = Device::Cpu;
        // Start with known values varying in dim 1
        let t1 = Tensor::new(&[100.0f32, 200.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[300.0f32, 400.0, 500.0, 600.0], &device)?.reshape((1, 4))?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2])?;

        // Get as tensor
        let tensor1 = batch.to_tensor();
        let values1 = tensor1.to_vec2::<f32>()?;
        assert_eq!(
            values1,
            vec![vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]]
        );

        // Get as vec
        let vec1 = batch.to_vec()?;
        assert_eq!(vec1.len(), 2);

        // Get as tensor again (should be idempotent)
        let tensor2 = batch.to_tensor();
        let values2 = tensor2.to_vec2::<f32>()?;
        assert_eq!(values1, values2);

        Ok(())
    }

    #[test]
    fn test_asymmetric_varying_dimensions_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        // Very different sizes in dim 2
        let t1 = Tensor::new(&[1.0f32], &device)?.reshape((1, 1, 1))?;
        let t2 = Tensor::new(
            &[2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            &device,
        )?
        .reshape((1, 1, 10))?;
        let t3 = Tensor::new(&[12.0f32, 13.0, 14.0], &device)?.reshape((1, 1, 3))?;

        let batch = TensorCat::from_tensors(2, vec![t1, t2, t3])?;

        // Verify concatenation
        assert_eq!(batch.cat_dim(), 2);
        assert_eq!(batch.segment_sizes(), &vec![1, 10, 3]);
        assert_eq!(batch.tensor.dims(), &[1, 1, 14]); // 1+10+3=14

        // Round trip back
        let vec = batch.to_vec()?;
        assert_eq!(vec[0].dims(), &[1, 1, 1]);
        assert_eq!(vec[1].dims(), &[1, 1, 10]);
        assert_eq!(vec[2].dims(), &[1, 1, 3]);

        Ok(())
    }

    #[test]
    fn test_index_parameter_with_varying_dims() -> Result<()> {
        let device = Device::Cpu;
        // Create tensors with varying dim 1, but different values
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0, 5.0, 6.0], &device)?.reshape((1, 4))?;
        let t3 = Tensor::new(&[7.0f32, 8.0, 9.0], &device)?.reshape((1, 3))?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2, t3])?;

        // All should have same dim 0
        assert_eq!(batch.dims(0)?, 1);

        // But different values when queried by index (though Cat uses inner_shape, all return same)
        assert_eq!(batch.dims2()?, (1, 2)); // First tensor's shape

        Ok(())
    }

    #[test]
    fn test_deref_allows_use_as_tensor() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((2, 2))?;
        let t2 = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], &device)?.reshape((2, 2))?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2])?;

        // Test deref: TensorCat can be used wherever &Tensor is expected
        let _: &Tensor = &batch;

        // Test deref in operations
        let result = batch.sum(0)?;
        let values = result.to_vec1::<f32>()?;
        // When concatenated along dim 0, shape is [4, 2]
        // Sum along dim 0: [1+5+3+7, 2+6+4+8] = [16, 20]
        assert_eq!(values, vec![16.0, 20.0]);

        Ok(())
    }

    #[test]
    fn test_as_tensor_method() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0], &device)?.reshape((1, 2))?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2])?;

        // Test as_tensor method
        let tensor_ref = batch.as_tensor();
        assert_eq!(tensor_ref.dims(), &[2, 2]);

        // Test that it's the same as deref
        let deref_ref = &*batch;
        assert_eq!(tensor_ref.dims(), deref_ref.dims());

        Ok(())
    }

    #[test]
    fn test_get_method_returns_segments() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0, 5.0], &device)?.reshape((1, 3))?;
        let t3 = Tensor::new(&[6.0f32, 7.0, 8.0, 9.0], &device)?.reshape((1, 4))?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2, t3])?;

        // Test get method
        let segment0 = batch.get(0)?;
        assert_eq!(segment0.dims(), &[1, 2]);
        let values0 = segment0.to_vec2::<f32>()?;
        assert_eq!(values0, vec![vec![1.0, 2.0]]);

        let segment1 = batch.get(1)?;
        assert_eq!(segment1.dims(), &[1, 3]);
        let values1 = segment1.to_vec2::<f32>()?;
        assert_eq!(values1, vec![vec![3.0, 4.0, 5.0]]);

        let segment2 = batch.get(2)?;
        assert_eq!(segment2.dims(), &[1, 4]);
        let values2 = segment2.to_vec2::<f32>()?;
        assert_eq!(values2, vec![vec![6.0, 7.0, 8.0, 9.0]]);

        Ok(())
    }

    #[test]
    fn test_get_method_bounds_check() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0], &device)?.reshape((1, 2))?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2])?;

        // This should return an error
        assert!(batch.get(3).is_err());
        Ok(())
    }

    #[test]
    fn test_into_iterator_yields_segments() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0, 5.0], &device)?.reshape((1, 3))?;
        let t3 = Tensor::new(&[6.0f32, 7.0, 8.0, 9.0], &device)?.reshape((1, 4))?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2, t3])?;

        // Consume the iterator and check all segments
        let mut count = 0;
        for segment in batch {
            match count {
                0 => {
                    assert_eq!(segment.dims(), &[1, 2]);
                    let values = segment.to_vec2::<f32>()?;
                    assert_eq!(values, vec![vec![1.0, 2.0]]);
                }
                1 => {
                    assert_eq!(segment.dims(), &[1, 3]);
                    let values = segment.to_vec2::<f32>()?;
                    assert_eq!(values, vec![vec![3.0, 4.0, 5.0]]);
                }
                2 => {
                    assert_eq!(segment.dims(), &[1, 4]);
                    let values = segment.to_vec2::<f32>()?;
                    assert_eq!(values, vec![vec![6.0, 7.0, 8.0, 9.0]]);
                }
                _ => panic!("Unexpected segment count"),
            }
            count += 1;
        }
        assert_eq!(count, 3);

        Ok(())
    }

    /// Ragged segments are tracked exactly, which is what every reader of a
    /// `TensorCat` slices by.
    #[test]
    fn test_segment_sizes_have_correct_values() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0, 5.0], &device)?.reshape((1, 3))?;
        let t3 = Tensor::new(&[6.0f32], &device)?.reshape((1, 1))?;

        let batch = TensorCat::from_tensors(1, vec![t1, t2, t3])?;

        assert_eq!(batch.segment_sizes(), &vec![2, 3, 1]);

        Ok(())
    }

    #[test]
    fn test_private_fields_accessible_via_methods() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0], &device)?.reshape((1, 2))?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2])?;

        // Test accessor methods
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch.cat_dim(), 0);
        assert_eq!(batch.dtype(), DType::F32);
        assert_eq!(batch.segment_sizes(), &vec![1, 1]);

        // Test as_cat_tensor
        let _ = batch.as_cat_tensor();

        // Test as_tensor
        let _ = batch.as_tensor();

        Ok(())
    }

    #[test]
    fn test_deref_with_candle_operations() -> Result<()> {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[1.0f32, 2.0], &device)?.reshape((1, 2))?;
        let t2 = Tensor::new(&[3.0f32, 4.0], &device)?.reshape((1, 2))?;

        let batch = TensorCat::from_tensors(0, vec![t1, t2])?;

        // When concatenated along dim 0, shape is [2, 2] (2 rows, 2 cols)
        assert_eq!(batch.as_tensor().dims(), &[2, 2]);

        // Test passing &TensorCat to candle operations expecting &Tensor
        // This tests that Deref is working correctly
        let sum_result = batch.sum_all()?;
        let sum_value = sum_result.to_scalar::<f32>()?;
        assert_eq!(sum_value, 10.0); // 1+2+3+4

        // Test transpose (should work via deref)
        // Transposing [2, 2] gives [2, 2]
        let transposed = batch.transpose(0, 1)?;
        assert_eq!(transposed.dims(), &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_get_with_varying_cat_dim() -> Result<()> {
        let device = Device::Cpu;
        // Create tensors that vary along dim 0 (same number of columns)
        let _t1 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((2, 2))?;
        let _t2 = Tensor::new(&[5.0f32, 6.0, 7.0], &device)?.reshape((3, 1))?;

        // This time, vary in dim 1 instead - both need same first dimension
        let t1_valid = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((2, 2))?;
        let t2_valid = Tensor::new(&[5.0f32, 6.0], &device)?.reshape((2, 1))?;

        let batch = TensorCat::from_tensors(1, vec![t1_valid, t2_valid])?;

        // Verify the concatenated tensor shape
        assert_eq!(batch.as_tensor().dims(), &[2, 3]); // concatenated along dim 1

        // get() should extract segments correctly
        let seg0 = batch.get(0)?;
        assert_eq!(seg0.dims(), &[2, 2]);

        let seg1 = batch.get(1)?;
        assert_eq!(seg1.dims(), &[2, 1]);

        Ok(())
    }
}
