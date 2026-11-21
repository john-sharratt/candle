//! Keeps one sequence's serialised slot state in the layout the attention kernel
//! reads, and uploads only the parts that changed.
//!
//! The slot holds 16-byte slice headers in one array, followed by the
//! out-of-line `KvHead` records. The records begin past room for the layout's
//! full capacity rather than past the live entries, so a chunk appended within
//! that capacity moves no record and the entries before it stay as they are. No
//! copy ever reads the host copy: uploads read a staging buffer, so the host
//! copy can be rewritten when host state changes and the host never waits for
//! the GPU to reach an upload.
//!
//! - `chunk_pin`: the handles a header depends on, held for as long as a launch reads them.
//! - `encoding`: the byte layout of headers, records and the palette map.
//! - `guard`: mutable access that marks dirty chunks and uploads them when it drops.
//! - `slot`: the per-sequence host copy and device slot.
//! - `snapshot`: stable per-generation copies for deferred launches.
//! - `staging`: the pinned buffers uploads go through and how one is chosen.
//! - `tests_pin`: tests for when a pin goes stale.
//! - `tests_snapshot`: tests for the header rebase and the records checksum.
//! - `tests_upload`: tests for upload ranges and staging choice.

use super::head_gids::HeadGids;
use super::meta_pool::MetaGid;
use super::slot_state_arena::{self, SlotStateSlot};
use super::slot_upload_batch;
use super::types::ChunkWindow;
use crate::kv_cache::arena_table::ResolvedArenaInfo;
#[cfg(test)]
use crate::kv_cache::arena_table::N_PALETTE;
use candle::cuda_backend::cudarc::driver::result::memcpy_htod_async;
use candle::cuda_backend::cudarc::driver::CudaEvent;
use candle::cuda_backend::{vacate_epoch, WrapErr};
use candle::quantized::pinned_staging::{give_recycled_wc, take_recycled_wc, PinnedBuf};
use candle::CudaDevice;
use std::sync::Arc;

mod chunk_pin;
mod encoding;
mod guard;
mod slot;
mod snapshot;
mod staging;
#[cfg(test)]
mod tests_pin;
#[cfg(test)]
mod tests_snapshot;
#[cfg(test)]
mod tests_upload;

pub use chunk_pin::*;
pub(crate) use encoding::*;
pub(crate) use guard::*;
pub(crate) use slot::*;
use snapshot::*;
use staging::*;
