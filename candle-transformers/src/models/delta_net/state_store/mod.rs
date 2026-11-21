//! One sequence's DeltaNet memory across every recurrent layer, advanced by waves
//! that either land whole or leave no trace.
//!
//! Each layer has a state, a timeline of rows not yet applied, and a scratch held
//! only while the sequence prefills. A short span writes its timeline and a long
//! span writes the scratch, so a failed wave is undone by dropping its bookkeeping.
//! Export and import carry the state as little-endian floats for the turn-seal
//! snapshot.
//!
//! - `layers`: each layer's slot, timeline and scratch.
//! - `store_access`: creating the store, reset and access to the layers.
//! - `store_waves`: opening, advancing, committing and rolling back a wave.
//! - `folding`: folding pending rows into the state.
//! - `store_steps`: steps, undo, and forks.
//! - `store_io`: export, import and the decode addresses.
//! - `snapshots`: checked snapshots and the layout hash.
//! - `compaction`: moving halves into lower slots.
//! - `host_buffers`: the pinned readback and upload buffers.
//! - Tests: `tests_layers`, `tests_memory`, `tests_snapshots`, `tests_steps`,
//!   `tests_cuda`.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::{
    cuda_backend::cudarc::driver::result::{memcpy_dtod_async, memcpy_htod_async},
    cuda_backend::cudarc::driver::CudaEvent,
    quantized::pinned_staging::PinnedBuf,
    CudaDevice, DType, LeaseAnchor, Storage,
};
#[cfg(feature = "cuda")]
use std::sync::Mutex;

use candle::{Device, Error, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    arena_regions, claim_arena_slots, plan_slot_moves, slot_share_bytes, ArenaSlot, SlotTenant,
    SpanRegion, SLOT_ALIGN,
};

#[cfg(feature = "cuda")]
use super::cuda::{delta_net_fold, fold_runs_fused, FoldEntry};
use super::host_rows;
use super::mix::{DeltaNetState, SpanWrite};
use super::timeline::{fold_by_ops, TimelineLayout, TIMELINE_ROWS};
use super::types::{DeltaNetDims, LayerKind};

/// One recurrent layer's state, exported as LE F32 bytes. Field-for-field the
/// persistence `SnapshotLayer` payload row.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportedLayerState {
    pub layer_index: u32,
    pub n_v_heads: u32,
    pub d_v: u32,
    pub d_k: u32,
    pub state: Vec<u8>,
    pub conv_channels: u32,
    pub conv_tail_cols: u32,
    pub conv_tail: Vec<u8>,
}

/// One sequence's recurrent memory across every DeltaNet layer.
pub struct RecurrentStateStore {
    dims: DeltaNetDims,
    hash: u64,
    slots: Vec<LayerSlot>,
    /// Whether a wave is open.
    ///
    /// One flag for the store rather than one per slot: the wave operations act
    /// on every slot together, so a per-slot answer could only ever disagree
    /// with its neighbours by being wrong.
    open: bool,
    /// Whether the wave open — or, after its commit, the last one — handed any
    /// layer to a span, and whether any of them was long. A commit that ran
    /// layers and no long span releases the scratch: the sequence is no longer
    /// prefilling. One that ran no layer at all — a segment whose window holds
    /// only attention layers — says nothing either way and keeps it.
    ran_this_wave: bool,
    long_this_wave: bool,
    /// Whether this store's state was put here deliberately — by a fork or a
    /// restore — and must therefore survive one `offset == 0` reset.
    ///
    /// Store-level, unlike `advanced`, because seeding is a property of where
    /// the whole state came from rather than of which layers a sweep reached.
    seeded: bool,
    /// The step open on this store, if any — see [`Self::begin_step`].
    step: Option<StepRecord>,
    device: Device,
}

mod compaction;
mod folding;
mod host_buffers;
mod layers;
mod snapshots;
mod store_access;
mod store_io;
mod store_steps;
mod store_waves;
#[cfg(test)]
mod tests_cuda;
#[cfg(test)]
mod tests_layers;
#[cfg(test)]
mod tests_memory;
#[cfg(test)]
mod tests_snapshots;
#[cfg(test)]
mod tests_steps;

pub use compaction::*;
pub use folding::*;
pub use host_buffers::*;
use layers::*;
pub use snapshots::*;
