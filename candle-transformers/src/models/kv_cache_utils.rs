//! Shared utilities for KV cache management in quantized models.
//!
//! This module provides concrete implementations of KvCaches and SequenceContext
//! for transformer models using CausalMaskCache.

use crate::models::causal_mask_cache::CausalMaskCache;
use candle::{Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;

// Re-export the core types from candle-nn
pub use candle_nn::kv_caches::CausalMaskProvider;

// Implement CausalMaskProvider for CausalMaskCache
impl CausalMaskProvider for CausalMaskCache {
    fn get_mask(&mut self, seq_len: usize, offset: usize) -> Result<Tensor> {
        self.get_mask(seq_len, offset)
    }

    fn clear(&mut self) {
        self.clear();
    }

    fn truncate(&mut self, seq_len: usize) -> Result<()> {
        self.truncate(seq_len)
    }
}

/// Container for KV caches with CausalMaskCache.
///
/// This is a concrete instantiation of `candle_nn::kv_caches::KvCaches<M>`
/// with `CausalMaskCache` as the mask provider.
pub type KvCaches = candle_nn::kv_caches::KvCaches<CausalMaskCache>;

/// Helper to create KvCaches with CausalMaskCache.
pub fn new_kv_caches(caches: Vec<KvCache>, device: Device) -> KvCaches {
    candle_nn::kv_caches::KvCaches::new(caches, CausalMaskCache::new(device))
}

/// Context for a single sequence in continuous batching.
///
/// This is a concrete instantiation of `candle_nn::sequence_context::SequenceContext<C>`
/// with `KvCaches` (which uses `CausalMaskCache`).
pub type SequenceContext<'a> = candle_nn::sequence_context::SequenceContext<'a, KvCaches>;
