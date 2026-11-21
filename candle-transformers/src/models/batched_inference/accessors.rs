//! Read-only views of a session's KV layout, dtype and layer count, for models
//! and the scheduler to query without touching the cache.
//!
//! The dtype answers differ on purpose. `dtype` describes sealed storage, while
//! `activation_dtype` and `kv_live_dtype` describe the width the residual stream
//! and live K/V bytes are held in. F16 stops at 65504 where BF16 keeps F32's
//! exponent, so conflating them lets large outputs become `inf` at the
//! narrowing. `stream_layers` bounds every aggregate taken across layers, which
//! keeps a draft head's own history out of it.

use super::*;

impl BatchedInferenceSession {
    /// Get the KV cache K format.
    pub fn k_format(&self) -> KvFormat {
        self.config.k_format
    }

    /// Get the KV cache V format.
    pub fn v_format(&self) -> KvFormat {
        self.config.v_format
    }

    /// Get the dtype used for KV cache storage (for float formats only).
    ///
    /// Returns BF16 for quantized formats as a default compute dtype.
    pub fn dtype(&self) -> DType {
        self.config.k_format.dtype().unwrap_or(DType::BF16)
    }

    /// The dtype activations arrive in for this session — what a model's norm
    /// weights must be materialised in before it runs a forward.
    ///
    /// Deliberately **not** [`Self::dtype`], and the difference is not cosmetic.
    /// `dtype` describes *sealed storage* and falls back to BF16 for a quantized
    /// format, which is right for the accounting it feeds.
    ///
    /// When [`BatchedConfig::activation_dtype`] is set, that is the answer: a
    /// model that knows the width its own reference implementation computes in
    /// states it, and nothing about KV storage overrides it.
    ///
    /// Otherwise the width is derived from the KV format, which is right only
    /// while K is stored float. For a quantized backing the live arena really
    /// is F16 (K in `R16`, V in plain F16), so the derivation yields F16 — and
    /// that is a statement about the *arena*, not about the residual stream.
    /// The two are unrelated, and conflating them costs range: F16 and BF16 are
    /// both 16 bits, but F16 stops at 65504 where BF16 carries F32's exponent.
    /// A model whose blocks legitimately produce 1e5–1e6 outputs has them
    /// silently become `inf` at the narrowing, and the next layer turns the
    /// `inf` into a NaN.
    pub fn activation_dtype(&self) -> DType {
        crate::models::batched_model::activation_dtype(
            self.config
                .activation_dtype
                .or_else(|| self.config.k_format.dtype())
                .unwrap_or(DType::F16),
        )
    }

    /// The width a live sequence's K/V bytes are actually held in.
    ///
    /// Not the sealed format and not the activation width: while a sequence is
    /// live its K sits in `R16` — raw F16 with Q-capture space — and its V in
    /// plain F16, whatever the sealed storage will later be. So a quantized
    /// format answers F16 here, and a float format answers itself.
    ///
    /// This is what Q/K/V are projected in, because K and V *become* these
    /// bytes. It equals [`Self::activation_dtype`] for every model whose
    /// activations and KV agree, and differs for one that computes wider than
    /// it stores.
    pub fn kv_live_dtype(&self) -> DType {
        self.config.k_format.dtype().unwrap_or(DType::F16)
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Leading KV layers a wave steps together — every layer except a draft
    /// head's.
    ///
    /// The bound for any aggregate taken *across* layers. A speculative head's
    /// K/V is its own history rather than the conversation's, so including it
    /// makes such an aggregate describe two different things at once — see the
    /// `stream_layers` field.
    pub fn stream_layers(&self) -> usize {
        self.stream_layers
    }

    /// Get the KV backing for a specific layer.
    pub fn backing(&self, layer: usize) -> Option<&ChunkedKvBacking> {
        self.backings.get(layer)
    }

    /// Get all backings (one per layer).
    pub fn backings(&self) -> &[ChunkedKvBacking] {
        &self.backings
    }

    /// Number of KV heads in this session's backing.
    pub fn n_kv_head(&self) -> usize {
        self.backings.first().map(|b| b.n_kv_head()).unwrap_or(0)
    }

    /// Head dimension in this session's backing.
    pub fn head_dim(&self) -> usize {
        self.backings.first().map(|b| b.head_dim()).unwrap_or(0)
    }
}
