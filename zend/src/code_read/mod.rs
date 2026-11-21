//! Ingestion of the `code_reading` layer: each file becomes one hidden
//! conversation that reads it and answers.
//!
//! A file's conversation runs the way a live dialogue turn runs: forked from the
//! same prelude, tools on and limited to `file_read`, and thinking off. Whatever
//! it says once it stops calling tools is the file's summary, so no separate
//! summary step runs. A file is keyed by its path and blob id, and it is read
//! only when no conversation carries that key; its reads name the commit the
//! file was found on. Each worker owns one file's conversation from start to
//! finish, so the pool is a bounded set of real concurrent conversations.
//!
//! - `binary_mark`: the record of keys sniffed as binary.
//! - `chains`: whether a file's reading is finished.
//! - `config`: the pool width and the limits.
//! - `ingest`: the worker pool and the upload path.
//! - `ingest_rate`: the pace of the ingest.
//! - `lines`: the line counts a reading reports.
//! - `link`: the priming-chain link a file's conversation continues.
//! - `process`: reading one file and committing its key.
//! - `reading`: the prompt, tools and grammar of a reading.
//! - `reading_break`: where a reading's chain is found broken.
//! - `reading_holds`: the holds placed on a reading's call.
//! - `tests_code_read`: tests of the reading rules.

pub mod binary_mark;
mod ingest_rate;
mod lines;
mod link;
mod reading_break;
mod reading_holds;

pub(crate) use link::ingest_link;
use reading_holds::reading_holds;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::chain_health::produced_no_answer;
use candle_conversation::persistence::identity::now_ms;
use candle_conversation::projection::{
    OptionalState, SelectionState, TimelineId, FORCE_TOOL_SELECTOR, NO_THINK_SELECTOR,
    TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::{
    HeldRules, HeldValues, OpeningCall, ToolSpec, TriggerRegistry, TurnGrammar,
};
use candle_conversation::substrate::Substrate;
use candle_conversation::{
    is_ground_refusal_message, is_stall_message, ConversationEngine, Sequence, TurnOptions,
    TurnText,
};
use serde_json::Value;
use zend_tools::tools::file::render::{range_value, windows, PAGE_LINES};
use zend_tools::ToolContext;
use zend_vfs::{ObjectFormat, Oid};

use self::lines::line_count;
use self::reading_break::{reading_break, Broken};
use crate::branch_ingest::filter::{language_of, MAX_FILE_BYTES};
use crate::branch_ingest::keys::{
    branches_value, file_key, quant_tags, BLOB_KEY, BRANCHES_KEY, COMMIT_KEY, CONTENT_KEY,
    LINES_KEY,
};
use crate::branch_ingest::plan::Committed;
use crate::branch_ingest::reader::{settles_here, whose, Whose};
use crate::calibrate::{calibrate_committed, CalibrateSpec};
use crate::ingest_backlog::{self, Work};
use crate::ingest_holdoff;
use crate::ingest_lease::{self, Share};
use crate::ingest_report::Failures;
use crate::ingest_snapshots::{finished, Unit};
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_path::split;
use crate::repo_scan::{is_binary_sample, Language};
use crate::tool_round;
use crate::tools::{catalog_for, format_tool_responses, ToolResult};
use crate::types::ToolMode;
use crate::workspace::UPLOADS_REPO;

mod chains;
mod config;
mod ingest;
mod process;
mod reading;
#[cfg(test)]
mod tests_code_read;

pub use chains::*;
pub use config::*;
pub use ingest::*;
use process::*;
pub use reading::*;
