//! The complete recurrent state of one sequence, as the model hands it over.
//!
//! A model's recurrence is not one thing. The delta-rule layers are one class
//! and have a shared shape ([`ExportedLayerState`]); everything else a model
//! carries forward — Flash-Next's PLE cache and QSA index, for instance — is
//! shaped like nothing else and travels as bytes only that model reads.
//!
//! This type is the two of them together, and it exists so they cannot come
//! apart. Both are captured at one instant, from one sequence, and describe the
//! same token count; a path that passed the layers alone and picked the blob up
//! separately could pair a state from one seal with a blob from another and
//! produce a sequence whose two halves of memory disagree about how much
//! history they have seen.

use std::sync::Arc;

use candle_transformers::models::delta_net::ExportedLayerState;

/// One sequence's recurrent state, both classes, from a single export.
#[derive(Clone, Debug, Default)]
pub struct ExportedState {
    /// Fingerprint of the layer schedule + DeltaNet dims these rows were
    /// produced under. Restore refuses a mismatch rather than scattering a
    /// foreign layout into the state arena.
    pub schedule_hash: u64,
    /// The delta-rule layers, in trunk order.
    pub layers: Vec<ExportedLayerState>,
    /// The model's own encoding of whatever else it carries. Empty for a model
    /// with no such state; opaque to everything between here and that model.
    pub aux: Vec<u8>,
}

impl ExportedState {
    /// True when there is nothing to install — neither class of state present.
    ///
    /// A model can legitimately produce one without the other: a pure-attention
    /// trunk with a PLE cache has an aux blob and no layers, and the delta-rule
    /// stacks have layers and no blob. Only both being empty means "no state".
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty() && self.aux.is_empty()
    }
}

/// The same state prepared for a fan-out install, where every slot in a group
/// receives byte-identical bytes and the payload is shared rather than cloned.
#[derive(Clone, Debug)]
pub struct SharedState {
    pub schedule_hash: u64,
    pub layers: Arc<[ExportedLayerState]>,
    pub aux: Arc<[u8]>,
}

impl From<&ExportedState> for SharedState {
    fn from(s: &ExportedState) -> Self {
        Self {
            schedule_hash: s.schedule_hash,
            layers: s.layers.clone().into(),
            aux: s.aux.clone().into(),
        }
    }
}
