//! Multi-layer KV cache container with shared mask cache.
//!
//! This module provides `KvCaches` for managing Key-Value caches across
//! multiple transformer layers with a shared causal attention mask cache.

use super::kv_cache::KvCache;
use candle::{Device, Result, Tensor};

/// Causal mask cache for efficient mask reuse across forward passes.
///
/// This should be replaced with an import from candle-transformers when used there,
/// but is provided here as a minimal trait bound for standalone usage.
pub trait CausalMaskProvider {
    fn get_mask(&mut self, seq_len: usize, offset: usize) -> Result<Tensor>;
    fn clear(&mut self);
    fn truncate(&mut self, seq_len: usize) -> Result<()>;
}

/// Container for KV caches with shared mask cache.
///
/// Manages a vector of KV caches (one per transformer layer) along with
/// a shared causal attention mask cache for efficient mask reuse.
#[derive(Debug, Clone)]
pub struct KvCaches<M> {
    pub caches: Vec<KvCache>,
    pub(crate) mask_cache: M,
}

impl<M: CausalMaskProvider> KvCaches<M> {
    /// Create a new KvCaches instance.
    ///
    /// # Arguments
    /// * `caches` - Vector of KV caches, one per transformer layer
    /// * `mask_cache` - Shared causal mask cache implementation
    pub fn new(caches: Vec<KvCache>, mask_cache: M) -> Self {
        Self { caches, mask_cache }
    }

    /// Get the current sequence length from the first cache.
    ///
    /// All caches should maintain the same sequence length in synchronized usage.
    pub fn current_seq_len(&self) -> usize {
        self.caches
            .first()
            .map(|c| c.current_seq_len())
            .unwrap_or(0)
    }

    /// Get the maximum sequence length from the first cache.
    pub fn max_seq_len(&self) -> usize {
        self.caches.first().map(|c| c.max_seq_len()).unwrap_or(0)
    }

    /// Get the dtype from the first cache.
    ///
    /// Returns the dtype of cached tensors, or BF16 as default if no caches exist.
    /// All caches should maintain the same dtype in synchronized usage.
    pub fn dtype(&self) -> candle::DType {
        self.caches
            .first()
            .map(|c| c.dtype())
            .unwrap_or(candle::DType::F32)
    }

    pub fn force_dtype(&mut self, dtype: candle::DType) {
        for cache in &mut self.caches {
            cache.force_dtype(dtype);
        }
    }

    /// Reset all caches to empty state.
    ///
    /// This clears both the KV caches and the mask cache.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
        self.mask_cache.clear();
    }

    /// Truncate all caches to the specified sequence length.
    ///
    /// # Arguments
    /// * `seq_len` - Target sequence length to truncate to
    pub fn truncate(&mut self, seq_len: usize) -> Result<()> {
        for cache in &mut self.caches {
            cache.truncate(seq_len)?;
        }
        self.mask_cache.truncate(seq_len)?;
        Ok(())
    }

    /// Get a causal attention mask for the given sequence length and offset.
    ///
    /// # Arguments
    /// * `seq_len` - Length of the current sequence
    /// * `offset` - Position offset (number of previously cached tokens)
    pub fn get_mask(&mut self, seq_len: usize, offset: usize) -> Result<Tensor> {
        self.mask_cache.get_mask(seq_len, offset)
    }

    /// Get the number of layers (caches) in this container.
    pub fn layer_count(&self) -> usize {
        self.caches.len()
    }

    /// Check integrity of all KV caches and return aggregate result.
    ///
    /// Returns a single `CacheIntegrityResult` that summarizes the state of all layers:
    /// - `Empty` if any cache is empty
    /// - `Invalid` with total NaN count across all layers if any NaNs are found
    /// - `Valid` if all caches are valid and non-empty
    pub fn check_integrity(&self) -> Result<super::kv_cache::CacheIntegrityResult> {
        use super::kv_cache::CacheIntegrityResult;

        let mut total_nans = 0usize;
        let mut total_elements = 0usize;
        let mut any_empty = false;

        for cache in &self.caches {
            match cache.check_integrity()? {
                CacheIntegrityResult::Empty => {
                    any_empty = true;
                }
                CacheIntegrityResult::Valid => {
                    // Continue checking other caches
                }
                CacheIntegrityResult::Invalid {
                    nan_count,
                    total_elements: elements,
                    percentage: _,
                } => {
                    total_nans += nan_count;
                    total_elements += elements;
                }
            }
        }

        if any_empty {
            Ok(CacheIntegrityResult::Empty)
        } else if total_nans > 0 {
            let percentage = (total_nans as f64 / total_elements as f64) * 100.0;
            Ok(CacheIntegrityResult::Invalid {
                nan_count: total_nans,
                total_elements,
                percentage,
            })
        } else {
            Ok(CacheIntegrityResult::Valid)
        }
    }

    /// Ensure all KV caches are in one of the specified dtypes.
    /// If already in one of the specified dtypes, returns without conversion.
    /// Otherwise, converts to the first dtype in the list.
    pub fn ensure_dtype(&mut self, dtypes: &[candle::DType]) -> Result<()> {
        for cache in &mut self.caches {
            cache.ensure_dtype(dtypes)?;
        }
        Ok(())
    }
}

// For simple usage without a mask cache, provide a no-op implementation
impl KvCaches<NoMaskCache> {
    /// Create KvCaches without mask caching support.
    pub fn new_without_masks(caches: Vec<KvCache>, device: Device) -> Self {
        Self {
            caches,
            mask_cache: NoMaskCache { _device: device },
        }
    }
}

/// No-op mask cache for cases where mask caching is not needed.
#[derive(Debug, Clone)]
pub struct NoMaskCache {
    _device: Device,
}

impl CausalMaskProvider for NoMaskCache {
    fn get_mask(&mut self, _seq_len: usize, _offset: usize) -> Result<Tensor> {
        candle::bail!("Mask cache not available in this configuration")
    }

    fn clear(&mut self) {
        // No-op
    }

    fn truncate(&mut self, _seq_len: usize) -> Result<()> {
        Ok(())
    }
}
