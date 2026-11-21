//! `repo_map` layer ingestion: one conversation per directory, each summarized
//! and tagged with the content key that retrieval and the pass both use.
//!
//! Each folder a branch lists is explored with one list round-trip, after which
//! its conversation decodes a short summary of what the folder is for. That
//! summary is the layer's retrieval surface, so a question about a subsystem
//! matches the prose describing its folder rather than only a filename the asker
//! already knew. The structure mirrors `code_read`: a bounded pool runs the
//! per-unit conversations, each is freed once its turns seal, and each unit's
//! tools read the commit it was found on, so its turns show what its key names.
//!
//! - `committed`: the committed-folder record and the boot sweep.
//! - `ingest_plan`: the per-pass steering and tags.
//! - `kv_estimate`: the KV cost of one more open conversation.
//! - `pool`: the bounded worker pool over a pass's folders.
//! - `process_dir`: the ingest of one folder.
//! - `prompt`: the shared prompt text the tree formats turns with.
//! - `scan_slot`: the gate on how many conversations are open.
//! - `width`: the pool width and its bounds.
//! - `tests_unit`: tests for naming, completion, tags and steering.
//! - `tests_width`: tests for the width arithmetic.

pub mod binary_sniff;
pub mod converse;
pub mod dir_unit;
mod link;
pub mod render;
pub mod types;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use candle_conversation::memory_report::MemoryReport;
use candle_conversation::projection::{self, TimelineId};
use candle_conversation::stencil::{
    HeldRules, ThinkMode, ToolCallEnvelope, TriggerRegistry, ValueHolds,
};
use candle_conversation::{is_stall_message, ConversationEngine, Sequence, SequenceConfig};
use zend_tools::ToolContext;
use zend_vfs::{Oid, RepoFiles};

use crate::branch_ingest::keys::{branches_value, BRANCHES_KEY, CONTENT_KEY};
use crate::branch_ingest::plan::Committed;
use crate::branch_ingest::reader::settles_here;
use crate::branch_ingest::units::FolderUnit;
use crate::held_paths;
use crate::ingest_backlog::{self, Work};
use crate::ingest_holdoff;
use crate::ingest_lease::{self, Share};
use crate::ingest_report::{Failures, IngestReport};
use crate::ingest_snapshots::{finished, Unit};
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::types::ToolMode;

pub use binary_sniff::is_binary_sample;
pub use dir_unit::DirUnit;
pub(crate) use link::ingest_link;
pub use types::Language;

/// Hard `max_tokens` on a folder summary decode. The request asks for two
/// sentences; this bounds the runaway case without clipping a summary that
/// enumerates a few of the folder's parts. Matches the `code_reading` scope
/// budget, which produces summaries of the same shape and length.
const FOLDER_SUMMARY_MAX_TOKENS: usize = 200;

/// Tolerated per-directory ingest failures before the whole pass aborts. A
/// handful of unreadable directories shouldn't sink a workspace scan; a flood
/// means something systemic (out of KV VRAM) and continuing just burns GPU.
const MAX_DECODE_FAILURES: usize = 24;

mod committed;
mod ingest_plan;
mod kv_estimate;
mod pool;
mod process_dir;
mod prompt;
mod scan_slot;
#[cfg(test)]
mod tests_unit;
#[cfg(test)]
mod tests_width;
mod width;

pub use committed::*;
pub use ingest_plan::*;
use kv_estimate::*;
use pool::*;
use process_dir::*;
pub(crate) use prompt::*;
use scan_slot::*;
pub use width::*;

// ── Tests ────────────────────────────────────────────────────────────────────
