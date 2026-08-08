//! Test modules for chunked KV cache components.
//!
//! This module organizes comprehensive tests for all major files:
//!
//! - `types_tests` - Tests for ChunkHandle, ChunkRef, SlotState, ChunkedState
//! - `arena_tests` - Tests for Arena, ArenaKey, ArenaStorage, StoragePolicy, ChunkStatus
//! - `backing_tests` - Tests for ChunkedKvBacking constructors and accessors
//! - `alloc_tests` - Tests for allocation methods
//! - `io_tests` - Tests for read_contiguous and write_contiguous
//! - `sequence_ops_tests` - Tests for alloc/free/share/fork operations
//! - `chunk_ops_tests` - Tests for migrate, copy, convert, prepare, reconcile
//! - `selection_table_tests` - Tests for the selection table's (chunk, head) rows

mod alloc_tests;
mod arena_tests;
mod backing_tests;
mod chunk_ops_tests;
mod compress_tests;
pub mod dump_reader;
mod gather_r16_tests;
mod gpu_chunks_tests;
mod io_tests;
mod kv_stats_tests;
mod selection_table_tests;
mod sequence_ops_tests;
mod types_tests;
