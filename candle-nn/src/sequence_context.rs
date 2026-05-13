//! Sequence context for continuous batching scenarios.
//!
//! This module provides `SequenceContext` for bundling the state needed
//! to process a single sequence in a continuous batching inference system.

use candle::Tensor;

use crate::{kv_caches::CausalMaskProvider, KvCaches};

/// Context for a single sequence in continuous batching.
///
/// Bundles KV cache reference, position offset, and input tokens into a
/// single type-safe structure for forwarding through transformer layers.
///
/// # Type Parameters
/// * `'a` - Lifetime of the borrowed cache and tensor references
/// * `C` - Type of the KV cache container (e.g., `KvCaches`)
///
/// # Fields
/// * `kv_caches` - Mutable reference to this sequence's KV cache
/// * `offset` - Position offset (number of tokens already cached)
/// * `input_ids` - Input token IDs for this forward pass
/// * `input_len` - Length of the input sequence
/// * `write_offset_shift` - Physical KV write offset shift (0 = no shift). Used for right-packed
///   partial blocks: shift = `chunk_size - token_count`, so tokens are stored at physical positions
///   `shift..shift+token_count` rather than `0..token_count`.
///
/// # Example
/// ```ignore
/// let ctx = SequenceContext {
///     kv_caches: &mut my_caches,
///     offset: 10,  // Already processed 10 tokens
///     input_ids: &tokens,
///     input_len: tokens.dims2()?.1,
///     write_offset_shift: 0,
/// };
/// let output = model.forward_with_context(ctx)?;
/// ```
pub struct SequenceContext<'a, C> {
    /// Mutable reference to this sequence's KV cache
    pub kv_caches: &'a mut C,
    /// Position offset in the sequence (number of cached tokens)
    pub offset: usize,
    /// Input token IDs for this forward pass
    pub input_ids: &'a Tensor,
    /// Length of the input sequence (number of tokens in input_ids)
    pub input_len: usize,
    /// Physical KV write offset shift for right-packed partial blocks (0 = no shift).
    /// When non-zero, the prefill kernel writes KV at physical positions
    /// `write_offset_shift..write_offset_shift + seq_len` rather than `0..seq_len`.
    /// Must be 0 for normal (non-static-chunk) prefills.
    pub write_offset_shift: usize,
}

impl<'a, M> SequenceContext<'a, KvCaches<M>>
where
    M: CausalMaskProvider,
{
    pub fn dtype(&self) -> candle::DType {
        self.kv_caches.dtype()
    }
}
