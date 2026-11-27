//! Sequence allocation and management for the chunked KV backing.
//!
//! The backing maps each batch index to a slot that holds a block table. These
//! methods allocate, free, fork, share, view, seal and truncate those tables,
//! together with the block-level copy-on-write and the metadata reads that go
//! with them.
//!
//! - `alloc`: allocating and freeing slots.
//! - `fork`: forking a sequence into another slot.
//! - `position_gap`: holes in a slot's position space.
//! - `queries`: per-block metadata reads.
//! - `sealed`: injecting and splicing sealed chunks.
//! - `sharing`: prefix sharing and borrowed chunks.
//! - `tail`: recording turns and truncating slots.
//! - `view`: view slots over a parent.
//! - `window`: the sliding-window ring.

use std::cmp;
use std::ops::Range;
use std::sync::Arc;

use candle::{DType, Result, Tensor};

use super::head_gids::GIDS_PER_HEAD;
use crate::CHUNK_SIZE;

use super::band_codec::encode_band;
use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
use super::io::read_band_chunk;
use super::types::{ChunkWindow, DetachedChunks, QSign, SealedChunk, SealedSequence};
use super::{ChunkedKvBacking, SequenceState};
use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};
use crate::kv_cache::{active_kv_formats, KvFormat};

mod alloc;
mod fork;
mod position_gap;
mod queries;
mod sealed;
mod sharing;
mod tail;
mod view;
mod window;
