//! The paged belief scan: it builds a small per-scan index over the resident
//! gallery arena and launches the paged BDP kernel.
//!
//! The gallery records stay resident in the arena. A scan uploads only the
//! addressing: the page addresses for each group, one entry per page and segment,
//! the segment prefixes and the probe. The kernel runs the same code as the
//! contiguous path, in its paged mode, and the host tally is shared with that path.
//!
//! - `build_index`: building the per-scan index.
//! - `index_cache`: reusing and leasing cached indices.
//! - `index_types`: the scan's types and device arrays.
//! - `launch`: launching the kernel and reading back its output.
//! - `scan_api`: the public scan entry points.
//! - `tests_index_lifecycle`: index cache lifecycle tests.
//! - `tests_scan_parity`: paged and contiguous parity tests.
//! - `tests_support`: the host statistics reference.
//! - `tests_votes`: vote, dedup and statistics tests.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async;
use candle::cuda_backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::staged::segment;
use candle::cuda_backend::Held;
use candle::{CudaDevice, Device, Result};
use candle_kernels::provenance::{
    bdp_bmma_supported, bdp_imma_supported, bdp_take_pending_error, run_batched_bdp_scan,
    run_bmma_bdp_scan, run_imma_bdp_scan, BdpStats,
};
use core::ffi::c_void;

use crate::persistence::streams::StreamId;
use crate::scheduler::profile;

use super::super::gpu::{needle_tally_segments, BDP_TQ};
use super::super::vote_law::VoteLaw;
use super::super::WideQSig;
use super::batch_ensure::Wanted;
use super::index_entries::IndexEntries;
use super::lease::ScanHold;
use super::pages::PAGE_TOKENS;
use super::probe_rows::ProbeRows;
use super::scan_plan::ScanPlan;
use super::scan_split::{self, LaunchTiming};
use super::token_votes::widen_rows;
use super::{
    Ensured, GalleryArena, GroupSet, Holder, IndexMiss, Keep, KeepMask, TokenStats, TokenVotes,
    TurnSigs,
};

mod build_index;
mod index_cache;
mod index_types;
mod launch;
mod scan_api;
#[cfg(test)]
use launch::*;
#[cfg(test)]
mod tests_index_lifecycle;
#[cfg(test)]
mod tests_scan_parity;
#[cfg(test)]
mod tests_support;
#[cfg(test)]
mod tests_votes;

pub use index_types::*;
