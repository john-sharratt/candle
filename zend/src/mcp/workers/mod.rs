//! Runs a task on a worker, answers a worker that asked, and waits on workers.
//!
//! A task's turn is an ordinary dialogue turn on the worker's own conversation, so
//! it is projected with provenance and journaled like any dialogue. It runs in a
//! task of its own, so the call that started it answers within its wait. When a
//! turn ends, the task drafts its journal at once and settles the worker as asking,
//! stuck, done or failed. A done answer is audited for its citations and kept whole,
//! to be continued with `repo_task_reply`. A failed task is archived.
//!
//! - `answer`: answers a worker that asked, is stuck, or was kept.
//! - `constants`: prompt wording, deadlines and retry intervals.
//! - `handoff`: moves a running worker between members at a round boundary.
//! - `journal_draft`: drafts a task's journal and explains an empty one.
//! - `journal_view`: reads journal entries and a turn's reply and question.
//! - `outcome`: decides how a turn ended.
//! - `record`: the beat and the lines written while a turn runs.
//! - `resume`: restarts a worker that a restart cut off.
//! - `run`: the turn's life, its checks and its settling.
//! - `settle`: settles a worker and audits its answer.
//! - `start`: starts a new worker under its thread.
//! - `state`: the lock on checking and beginning a turn.
//! - `turn_start`: begins a worker's turn under its lease.
//! - `wait`: waits on workers and reports their progress.
//! - `tests_workers`: tests of the task rules.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::http::{HeaderName, HeaderValue, Method};
use candle_conversation::persistence::SharedSubstrate;
use futures::StreamExt;
use host::cluster::up::send_up;
use host::substrate_views::TimelineDetail;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Notify;
use web::auth::Role as AccessRole;

use super::activity::{Activity, Doing};
use super::archiving;
use super::audit::{Audit, Turn};
use super::caller::{Caller, Person};
use super::chats::naming::is_chat_thread;
use super::chats::talk;
use super::driven::{Driven, DRIVEN_SELECTOR};
use super::names;
use super::nudges::{self, Sender};
use super::ops::NODE_HEADER;
use super::outcomes::{self, Outcome};
use super::records::Records;
use super::resuming;
use super::roles::{WorkerKind, ROLE_SELECTOR};
use super::steps;
use super::stops;
use super::thread_export;
use super::threads::{self, Effort, Entry, Note, Priority, State, Worker, BEAT_EVERY_MS};
use crate::access;
use crate::api::chat::{apply_tools_dial, dial_selection};
use crate::hand_records::{self, Carrying};
use crate::ingest_lease::{self, Guard, Share};
use crate::session::{NoDraft, Resume, StreamItem, ZendSession, CLOSED_BY_CLAUDE};
use crate::tools::Dispatch;
use crate::worker_hop;

mod answer;
mod constants;
mod handoff;
mod journal_draft;
mod journal_view;
mod outcome;
mod record;
mod resume;
mod run;
mod settle;
mod start;
mod state;
#[cfg(test)]
mod tests_workers;
mod turn_start;
mod wait;

pub use answer::*;
pub use constants::*;
pub use handoff::*;
pub use journal_draft::*;
pub use journal_view::*;
pub use outcome::*;
use record::*;
pub use resume::*;
use run::*;
use settle::*;
pub use start::*;
pub use state::*;
pub(super) use turn_start::*;
pub use wait::*;
