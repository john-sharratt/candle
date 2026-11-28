//! Ingestion of the `code_reading` layer: each file is cut into chunks of
//! whole lines and sealed into the knowledge base as members, then recorded
//! (`docs/provenance_system_collection.md` §7). Nothing is decoded.
//!
//! A file is keyed by its path and blob id, and sealed only when no record
//! carries that key, or the one that does names members that do not restore
//! from the record alone. Its bytes are read at the commit it was found on.
//!
//! - `binary_mark`: the record of keys sniffed as binary.
//! - `boot_restore`: the recorded files restored from their records before
//!   the pass.
//! - `chains`: the records the pass plans against.
//! - `config`: the pool width and the limits.
//! - `ingest`: the worker pool and the upload path.
//! - `ingest_rate`: the pace of the ingest.
//! - `process`: sealing one file and writing its record.
//! - `reading`: the prompt, tools and grammar a dialogue's file reads run on.
//! - `tests_code_read`: tests of the reading rules.

pub mod binary_mark;
mod ingest_rate;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::files::{FileRecord, FileUnit};
use candle_conversation::projection::TimelineId;
use candle_conversation::substrate::Substrate;
use candle_conversation::{
    is_ground_refusal_message, is_stall_message, ConversationEngine, Sequence,
};
use zend_tools::ToolContext;
use zend_vfs::{ObjectFormat, Oid};

use crate::branch_ingest::filter::{language_of, MAX_FILE_BYTES};
use crate::branch_ingest::keys::file_key;
use crate::branch_ingest::plan::Committed;
use crate::files::ingest::{ingest_text, FileSource};
use crate::files::records;
use crate::ingest_backlog::{self, Work};
use crate::ingest_holdoff;
use crate::ingest_lease::{self, Share};
use crate::ingest_report::Failures;
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_path::split;
use crate::repo_scan::{is_binary_sample, Language};
use crate::workspace::UPLOADS_REPO;

mod boot_restore;
mod chains;
mod config;
mod ingest;
mod process;

pub use chains::*;
pub use config::*;
pub use ingest::*;
use process::*;
