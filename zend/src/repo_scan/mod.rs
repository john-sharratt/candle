//! `repo_map` layer ingestion: each folder a branch lists is sealed into the
//! knowledge base as its listing (`docs/provenance_system_collection.md`
//! §12 D14), and recorded as a file's text is.
//!
//! - `binary_sniff`: whether a file's bytes are text.
//! - `committed`: the folder records the pass plans against.
//! - `ingest_plan`: the pass's entry and the ingest base's config.
//! - `pool`: the bounded worker pool over a pass's folders.
//! - `process_dir`: the seal of one folder.
//! - `prompt`: the shared prompt text the tree formats turns with.
//! - `types`: the languages a file is read as.

pub mod binary_sniff;
pub mod types;

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::files::{FileRecord, FileUnit};
use candle_conversation::{is_stall_message, ConversationEngine, Sequence, SequenceConfig};
use zend_tools::ToolContext;
use zend_vfs::Oid;

use crate::branch_ingest::keys::unversioned;
use crate::branch_ingest::plan::Committed;
use crate::branch_ingest::units::FolderUnit;
use crate::code_read::committed_of;
use crate::files::ingest::{ingest_text, FileSource};
use crate::files::listing::listing_text;
use crate::files::records;
use crate::ingest_backlog::{self, Work};
use crate::ingest_holdoff;
use crate::ingest_lease::{self, Share};
use crate::ingest_report::{Failures, IngestReport};
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;

pub use binary_sniff::is_binary_sample;
pub use types::Language;

/// Tolerated per-folder seal failures before the whole pass aborts. A
/// handful of unreadable folders shouldn't sink a workspace scan; a flood
/// means something systemic.
const MAX_DECODE_FAILURES: usize = 24;

/// Workers sealing folders at once: a listing is a few hundred tokens of
/// prefill, so the width is the files pool's.
const FOLDER_PARALLELISM: usize = crate::code_read::CODE_READ_PARALLELISM;

mod committed;
mod ingest_plan;
mod pool;
mod process_dir;
mod prompt;

pub use committed::*;
pub use ingest_plan::*;
use pool::*;
use process_dir::*;
pub(crate) use prompt::*;
