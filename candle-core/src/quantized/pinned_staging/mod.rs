//! Pinned host memory that the GPU reads over PCIe, staged in generations so the
//! arena is never reset under a kernel still reading it.
//!
//! The device maps a bump arena of pinned memory, so kernels read staging data
//! directly, with no copy and no driver call on the submit path. A generation
//! guard keeps the arena alive while such reads may happen. An overflow slab
//! takes over when the arena fills during a generation, rather than falling back
//! to slower copies. Dropping a generation never blocks: the reset is deferred to
//! the next generation, behind an event.
//!
//! - `arena`: the mapped pinned slab and its bump offset.
//! - `buffers`: the staging buffer kinds and the recycler.
//! - `generation`: the lifetime guard and the deferred reset.
//! - `stager`: allocation and submission to the GPU.
//! - Tests: `tests_pinned`.

use super::resident_ring::{Carve, RingPlan};
use crate::cuda_backend::graph::outside_capture;
use crate::cuda_backend::{stream_alloc, Held, WrapErr};
use crate::{CudaDevice, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

/// Default arena slab size (128 MB).
const DEFAULT_ARENA_SIZE: usize = 128 * 1024 * 1024;

/// Allocations at or below this threshold use the bump arena.
const BUMP_THRESHOLD: usize = 16 * 1024 * 1024;

/// Alignment for bump allocations (16 bytes — sufficient for uint4 / SSE).
const BUMP_ALIGN: usize = 16;

mod arena;
mod buffers;
mod generation;
mod stager;
#[cfg(test)]
mod tests_pinned;

use arena::*;
pub use buffers::*;
pub use generation::*;
pub use stager::*;
