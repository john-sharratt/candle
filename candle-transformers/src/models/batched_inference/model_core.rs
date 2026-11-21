//! Carries the static properties of a model that a planner needs before any
//! session runs: layer counts, head geometry, provenance layers and thresholds.
//!
//! Provenance layers are the two per signature band, laid out in the order
//! `[syn_l0, syn_l4, sem_l0, sem_l4, prag_l0, prag_l4]`. On a hybrid stack the
//! count of layers that carry a Q can be below the transformer depth, and the
//! provenance fold is sized from that count. Gap-fill support and recurrent-state
//! carrying are model properties, read before planning, so the planner does not
//! find out from a refusal inside a forward. The properties are captured once,
//! before the model moves to the scheduler thread.

// ManagedBatchedModel Trait and Blanket Implementation

/// Trait for models that support session-managed batched inference.
///
/// This trait provides a high-level API for batched inference using
/// [`BatchedInferenceSession`] to manage KV cache state.
///
/// Layer index pairs for the three provenance signature depths.
///
/// Layout: `[syn_l0, syn_l4, sem_l0, sem_l4, prag_l0, prag_l4]`.
///
/// Each band uses two layers — band-start (`l0 = centre − 4`, clamped) and
/// band-centre (`l4`) — whose multi-head Q sign bits are XOR-folded into a
/// single 128-bit signature per token (`MH_XOR_QQ_l0xl4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvenanceLayerIndices {
    pub syn_l0: usize,
    pub syn_l4: usize,
    pub sem_l0: usize,
    pub sem_l4: usize,
    pub prag_l0: usize,
    pub prag_l4: usize,
}

impl ProvenanceLayerIndices {
    /// Flat array form `[syn_l0, syn_l4, sem_l0, sem_l4, prag_l0, prag_l4]`.
    pub fn as_array(self) -> [usize; 6] {
        [
            self.syn_l0,
            self.syn_l4,
            self.sem_l0,
            self.sem_l4,
            self.prag_l0,
            self.prag_l4,
        ]
    }
}

/// Static model properties that are cheap to copy and independent of any inference session.
///
/// Captured once (before the model moves to the scheduler thread) via
/// [`ManagedBatchedModel::model_core_properties`] and stored wherever these values are needed.
#[derive(Clone, Copy, Debug)]
pub struct ModelCoreProperties {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub n_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Layer-index pairs for the three provenance signature depth bands.
    pub provenance_layer_indices: ProvenanceLayerIndices,
    /// Per-model multiplier for the K high adaptive threshold.
    pub k_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the K low adaptive threshold.
    pub k_low_error_threshold_factor: f32,
    /// Per-model multiplier for the V high (strict) adaptive threshold.
    pub v_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the V low (lenient) adaptive threshold.
    pub v_low_error_threshold_factor: f32,
    /// How many layers actually have a Q to capture — the fold's layer count.
    ///
    /// Equal to `num_layers` on a uniform transformer and to the **attention**
    /// layer count on a hybrid, where three quarters of the stack is recurrent
    /// and has no Q in a KV cache to read. The provenance fold groups
    /// `[n − 2, 1, 1]` over this, so passing transformer depth on a hybrid
    /// would size the lower group for layers that contribute nothing.
    pub provenance_capture_layers: usize,
    /// Whether this model can compute K/V for tokens inserted **mid-sequence**
    /// — a glue island the projection reserves a gap chunk for.
    ///
    /// **Gap-fill is not the same as prefill.** Appending at the writer tail is
    /// ordinary prefill and every model does it; filling a hole in the middle of
    /// a sequence requires each inserted token's output to depend only on what
    /// precedes it *positionally*, which attention gives for free and a
    /// recurrence cannot give at all — token *t*'s output depends on the
    /// accumulated state over everything before it, in order, and that state has
    /// already moved past the hole.
    ///
    /// A real model property, like `head_dim`. The planner reads it **before**
    /// planning rather than discovering it from a bail inside `forward_wave`,
    /// because by then the projection has already been built around islands the
    /// model cannot fill.
    pub can_gap_fill: bool,
    /// Whether this model keeps per-sequence state outside the KV cache — the
    /// recurrent matrices and conv tails of a hybrid stack.
    ///
    /// The mirror of [`Self::can_gap_fill`], and needed at the same altitude:
    /// the prompt builder has to know, before it starts, whether a branch
    /// checkpoint is something it must compute. A conversation on a model that
    /// carries no such state needs none, and the pass must not run — 96
    /// full-prompt prefills for a quantity nothing will read.
    ///
    /// [`ManagedBatchedModel::carries_recurrent_state`] answers the same
    /// question where a `&dyn` model is in hand; this carries it to code that
    /// only has the copied properties.
    pub carries_recurrent_state: bool,
}
