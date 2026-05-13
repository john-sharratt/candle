//! Efficient GPU-accelerated causal mask caching for transformer models.
//!
//! This module provides optimized causal mask creation and caching with:
//! - GPU-native tensor operations (no CPU loops)
//! - Mask expansion for batched processing
//! - Mask truncation for cache management
//! - Single mask reuse across all layers

use candle::{DType, Device, Result, Tensor};

/// Cache for causal attention masks with expansion and truncation support
#[derive(Debug, Clone)]
pub struct CausalMaskCache {
    /// Cached mask: ((seq_len, offset), mask_tensor)
    last_mask: Option<((usize, usize), Tensor)>,
    device: Device,
}

impl CausalMaskCache {
    /// Create a new mask cache for the given device
    pub fn new(device: Device) -> Self {
        Self {
            last_mask: None,
            device,
        }
    }

    /// Get or create a causal mask for the given dimensions.
    ///
    /// # Arguments
    /// * `seq_len` - Number of query positions (batch size)
    /// * `offset` - Number of cached positions (KV cache length)
    ///
    /// # Returns
    /// Mask tensor of shape (1, 1, seq_len, seq_len + offset) where:
    /// - 0.0 = position is visible (can attend)
    /// - -inf = position is masked (cannot attend)
    pub fn get_mask(&mut self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let cache_key = (seq_len, offset);

        // Check for exact cache hit
        if let Some((last_key, ref mask)) = self.last_mask {
            if last_key == cache_key {
                return Ok(mask.clone());
            }

            // Try mask expansion for batched processing
            let (last_seq_len, last_offset) = last_key;
            if last_seq_len == seq_len && last_offset < offset {
                if let Some(expanded) = self.try_expand_mask(mask, seq_len, last_offset, offset)? {
                    self.last_mask = Some((cache_key, expanded.clone()));
                    return Ok(expanded);
                }
            }
        }

        // Create new mask from scratch
        let mask = self.create_mask_gpu(seq_len, offset)?;
        self.last_mask = Some((cache_key, mask.clone()));
        Ok(mask)
    }

    /// Create a causal mask using GPU tensor operations
    /// Returns F32 mask where 0.0 = visible, -inf = masked
    fn create_mask_gpu(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let total_len = seq_len + offset;

        // Create range tensors on GPU as U32 for comparison
        let row_indices = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .unsqueeze(1)? // (seq_len, 1)
            .broadcast_as((seq_len, total_len))?;

        let col_indices = Tensor::arange(0u32, total_len as u32, &self.device)?
            .unsqueeze(0)? // (1, total_len)
            .broadcast_as((seq_len, total_len))?;

        // mask[i,j] = (j > i + offset) ? -inf : 0.0
        // Calculate threshold as U32: row_indices + offset
        let offset_tensor =
            Tensor::new(&[offset as u32], &self.device)?.broadcast_as((seq_len, total_len))?;
        let threshold = (row_indices + offset_tensor)?;

        // Compare as U32: col_indices > threshold returns U8 (0 or 1)
        let mask_u8 = col_indices.gt(&threshold)?;

        // Convert U8 to F32 for mask values: 0u8 -> 0.0f32, 1u8 -> -inf
        // Use where_cond with U8 condition
        let neg_inf =
            Tensor::new(&[f32::NEG_INFINITY], &self.device)?.broadcast_as((seq_len, total_len))?;
        let zero = Tensor::new(&[0.0f32], &self.device)?.broadcast_as((seq_len, total_len))?;
        let mask = mask_u8.where_cond(&neg_inf, &zero)?;

        // Add batch and head dimensions: (seq_len, total_len) -> (1, 1, seq_len, total_len)
        mask.unsqueeze(0)?.unsqueeze(0)
    }

    /// Try to expand an existing mask by concatenating new columns
    fn try_expand_mask(
        &self,
        mask: &Tensor,
        seq_len: usize,
        old_offset: usize,
        new_offset: usize,
    ) -> Result<Option<Tensor>> {
        let old_total_len = seq_len + old_offset;
        let new_total_len = seq_len + new_offset;
        let cols_to_add = new_total_len - old_total_len;

        if cols_to_add == 0 {
            return Ok(None);
        }

        // Remove batch/head dimensions: (1, 1, seq_len, old_total) -> (seq_len, old_total)
        let mask_core = mask.squeeze(0)?.squeeze(0)?;

        // Create new columns of zeros (all cached positions are visible)
        let new_cols = Tensor::zeros((seq_len, cols_to_add), DType::F32, &self.device)?;

        // Concatenate along column dimension
        let expanded_mask = Tensor::cat(&[&mask_core, &new_cols], 1)?;

        // Re-add batch and head dimensions
        let expanded_mask = expanded_mask.unsqueeze(0)?.unsqueeze(0)?;

        Ok(Some(expanded_mask))
    }

    /// Truncate the cached mask to match a smaller cache length
    pub fn truncate(&mut self, new_cache_len: usize) -> Result<()> {
        if let Some(((mask_seq_len, mask_offset), ref mask)) = self.last_mask {
            let mask_total_len = mask_seq_len + mask_offset;

            // Try to slice the mask if dimensions are compatible
            if mask_seq_len == mask_seq_len
                && new_cache_len < mask_total_len
                && new_cache_len >= mask_seq_len
            {
                // Slice the mask: keep only first new_cache_len columns
                // Mask shape: (1, 1, mask_seq_len, mask_total_len)
                if let Ok(truncated_mask) = mask.narrow(3, 0, new_cache_len) {
                    let new_offset = new_cache_len - mask_seq_len;
                    self.last_mask = Some(((mask_seq_len, new_offset), truncated_mask));
                    return Ok(());
                }
            }
        }

        // Clear cache if truncation not possible
        self.clear();
        Ok(())
    }

    /// Clear the cached mask
    pub fn clear(&mut self) {
        self.last_mask = None;
    }

    /// Get the device this cache is associated with
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = CausalMaskCache::new(device.clone());

        // First call creates mask
        let mask1 = cache.get_mask(10, 0)?;
        assert_eq!(mask1.dims(), &[1, 1, 10, 10]);

        // Second call reuses cached mask
        let mask2 = cache.get_mask(10, 0)?;
        assert_eq!(mask2.dims(), &[1, 1, 10, 10]);

        Ok(())
    }

    #[test]
    fn test_mask_expansion() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = CausalMaskCache::new(device.clone());

        // Create initial mask
        let mask1 = cache.get_mask(5, 0)?;
        assert_eq!(mask1.dims(), &[1, 1, 5, 5]);

        // Expand with offset (simulating batched processing)
        let mask2 = cache.get_mask(5, 5)?;
        assert_eq!(mask2.dims(), &[1, 1, 5, 10]);

        // Further expansion
        let mask3 = cache.get_mask(5, 10)?;
        assert_eq!(mask3.dims(), &[1, 1, 5, 15]);

        Ok(())
    }

    #[test]
    fn test_mask_truncation() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = CausalMaskCache::new(device.clone());

        // Build up cache
        cache.get_mask(5, 0)?;
        cache.get_mask(5, 5)?;
        cache.get_mask(5, 10)?;

        // Truncate
        cache.truncate(10)?;

        // Next call should use truncated mask
        let mask = cache.get_mask(5, 5)?;
        assert_eq!(mask.dims(), &[1, 1, 5, 10]);

        Ok(())
    }
}
