//! The QSA indexer on the device: each sequence's compressed keys, and the
//! selection that a wave's rows are scored by.
//!
//! A completed block is pooled and normed once and stored un-rotated. The scorer
//! rotates each key as it reads it, so a page can be placed at a new position
//! without any of its bytes changing. Only the rows of a block still filling are
//! held raw. Wave atomicity uses snapshot and restore, since an append only
//! advances a count. Capacity is claimed at admission and never inside a layer,
//! because a forward holds the arena window shut while it runs.
//!
//! - `append`: appends a wave's index keys in one pair of launches.
//! - `bound`: bounds the scratch a scoring pass may carve.
//! - `cache`: one sequence's index cache on one layer.
//! - `device`: device addresses of tensors and descriptor tables.
//! - `pages`: injected pages, and the state a rewind or compaction needs.
//! - `project`: projects a wave's keys and queries in one GEMM each.
//! - `rewind`: undoes a failed wave, and cuts a cache back.
//! - `rotate`: rotates rows at their own positions and rungs.
//! - `score`: scores a segment's queries into the wave's score buffer.
//! - `select`: builds the wave's selection table.
//! - Tests: `tests_device`, `tests_kernel`, `tests_pages`, `tests_placement`, `tests_pricing`, `tests_record`, `tests_rewind`, `tests_support`.

use candle::engine_events::note;
use candle::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::append_grids::AppendGrids;
use super::capture_rows::CaptureRows;
use super::config::IndexerConfig;
use super::hits::block_range;
use super::index_keys::scale_plane_offset;
/// Block keys per live-tail key page — see `index_keys`.
pub use super::index_keys::PAGE_BLOCKS;
use super::index_keys::{
    alloc_buffers, key_addr, key_page_bytes, write_host, KeyPages, SnapshotBuffer, SnapshotBuffers,
    SNAPSHOT_BUFFERS,
};
use super::key_q8::Q8Keys;
use super::qsa::IndexerWeights;
use super::qsa_select::{
    entry_block, entry_cells, max_entries_for, max_gathered_for, max_keep, merge_ranges,
    selected_width, PinRow, Strata, MAX_PINS, MAX_RATIO,
};
use super::resident_page::{ResidentPage, CODE_GROUP};
use super::rows_matmul::rows_matmul_t;
use super::spec::SpecCapture;
use crate::models::decode_diag::{self, RawSelection};
use crate::models::delta_net::mix::SeqSpan;
use crate::models::delta_net::RecurrentCompaction;
use crate::models::operand_guard::{expect_dense, expect_dense_dtype};
use crate::models::qsa_selection::QsaSelection;
use crate::models::selection_strata::Recent;
use crate::models::wave_buffers::{wave_empty_ticketed, wave_from_vec_ticketed};
use candle::cuda_backend::matmul_f32_into;
use candle::wave_provenance::WaveTicket;
use candle_kernels::simple::f32_rows_matmul::F32_ROWS_MATMUL_MAX_ROWS;
#[cfg(feature = "cuda")]
use candle_kernels::simple::qsa_topk::{qsa_topk_split_parts, SPLIT_KEYS};
use candle_kernels::simple::qsa_topk::{CUT_WORDS, MAX_ENTRIES, MAX_KEEP, NO_CUT};
use candle_nn::kv_cache::{
    arena_regions, plan_slot_moves, relocate_tensor, ArenaSlot, SlotTenant,
    PREFILL_KV_STAGE_MIN_Q_LEN,
};
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::ptr::null_mut;

use crate::models::rope_schedule::FactoredRope;

/// One sequence's index cache for one full-attention layer.
#[derive(Debug)]
pub struct IndexCache {
    /// Prepared block keys (pooled, normed, un-rotated), in pages of the span's
    /// QSA-index arenas. Only `[0, n_blocks)` is live.
    keys: KeyPages,
    n_blocks: usize,
    /// `[MAX_RATIO, head_dim]` F32 — the raw projected keys of the block still
    /// filling, contiguous in `[0, n_open)`. Fewer than `ratio` rows; empty
    /// when the cache ends on a block boundary.
    ///
    /// One buffer rather than the list of arrival pieces it used to be. The
    /// list existed to avoid a `cat` per step, but the batched append reads
    /// these rows from a KERNEL, which needs one base pointer and a row stride
    /// — and the carry that fills it is itself batched across the wave, so the
    /// per-step copy the list was avoiding does not come back.
    raw: Tensor,
    /// Rows of [`Self::raw`] that are live.
    n_open: usize,
    /// Rewind copies of [`Self::raw`], claimed with it — see [`Self::snapshot`].
    snaps: SnapshotBuffers,
    /// Index rows for positions this sequence holds but never forwarded —
    /// prefixes whose K/V arrived by injection.
    ///
    /// **The case that makes this necessary is section ingest.** A section is
    /// prefilled against an Arc-injected prefix: the prefix's K/V already
    /// exists, so copying it is free and the forward attends to real preceding
    /// context. The index cannot be copied that way — its keys come from hidden
    /// states, which injection never computes — so without these pages a query
    /// at position 22,020 asks for 5,505 blocks against a cache holding one.
    ///
    /// **They are pages and not more rows of `keys` because they are ragged.**
    /// A section advances the sequence by its real token count, not by a whole
    /// number of blocks, so the next section's pooling starts mid-block and its
    /// rows do not line up with `(pos + 1) / ratio`. Concatenating them into the
    /// uniform buffer would silently address the wrong block for every position
    /// after the first boundary. Each page therefore keeps its own width and
    /// [`Self::candidates_at`] walks them, exactly as
    /// [`PagedIndex`](super::paged_index::PagedIndex) does for a window
    /// reconstructed from turn records.
    ///
    /// Empty for a sequence that forwarded everything it holds, which is every
    /// sequence the gates run — there the walk reduces to the uniform formula
    /// and the scorer sees a single page.
    ///
    /// **Shared, not owned.** Each page is resident on the span once and held by
    /// `Arc`: a fork, a view carve and every other slot that injected the same
    /// piece read the same rows, exactly as they borrow the same K/V chunks.
    pages: Vec<PlacedPage>,
    /// Exclusive prefix sum of [`Self::pages`] row counts.
    ///
    /// Rows stay ordinal, and only rows: the scorer's candidate axis is a dense
    /// concatenation of every page's rows, so row `k` of page `p` is candidate
    /// `page_rows[p] + k` however the pages are positioned. Positions are the
    /// other thing entirely — see [`PlacedPage::base`].
    page_rows: Vec<usize>,
    /// Absolute position the live tail opens at.
    ///
    /// **Recorded, not accumulated.** It used to be the running sum of the page
    /// widths, which made every position downstream of an unindexed span wrong
    /// by that span's width; it is now whatever the last placement said, so a
    /// span nothing indexed is a hole and nothing else.
    tail_base: usize,
    /// Position the conversation's system prompt ends at — `0` when the slot
    /// holds none.
    ///
    /// Set by the projection that placed the prompt, and the reason it lives
    /// here rather than beside the sequence: a stratified selection ranks the
    /// prompt's blocks in every window, so it has to describe the same layout
    /// the pages do, and travel wherever they travel — a view carve, a move, a
    /// rebuild that keeps the prefix.
    prompt_end: usize,
    /// The position ranges of the system prompt every query attends whole —
    /// the pinned items (`docs/hit_driven_projection.md` §5). Sorted,
    /// disjoint; empty when nothing is pinned. Like [`Self::prompt_end`] it
    /// describes the page layout, so it moves with every splice and cut.
    pins: Vec<Range<usize>>,
    /// Counts every change to the pins or the page layout — what a cached
    /// [`PinRow`] is valid for.
    layout_gen: u64,
    /// The last [`Self::pin_row`] built: `(layout_gen, ratio, max_blocks,
    /// row)`. A row changes only when the pins or the pages do, and it is
    /// asked for at every selecting layer of every wave.
    pin_cache: Mutex<Option<(u64, usize, usize, PinRow)>>,
}

mod append;
mod bound;
mod cache;
mod device;
mod pages;
mod project;
mod rewind;
mod rotate;
mod score;
mod select;
#[cfg(test)]
mod tests_device;
#[cfg(test)]
mod tests_kernel;
#[cfg(test)]
mod tests_pages;
#[cfg(test)]
mod tests_placement;
#[cfg(test)]
mod tests_pricing;
#[cfg(test)]
mod tests_record;
#[cfg(test)]
mod tests_rewind;
#[cfg(test)]
mod tests_support;

pub use append::*;
pub use bound::*;
pub(super) use device::*;
pub use pages::*;
pub use project::*;
pub use rotate::*;
pub use select::*;
