//! The working set — what a dialogue already knows about the code
//! (`docs/zend_working_set.md`).
//!
//! Every dialogue carries a token budget of already-ingested `code_reading`
//! files and `repo_map` folders, projected ahead of its own turns in the order
//! they entered: **locks** the fast path served, and **provenance** — the files
//! the conversation keeps attending to, by momentum, starting with the
//! **seeds** the schema names. The unit is always a whole ingest conversation.

mod config;
pub mod marks;
mod state;

pub use config::WorkingSetConfig;
pub use state::{Candidate, Limits, Refusal, WorkingSet};
