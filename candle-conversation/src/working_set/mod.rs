//! The working set — what a dialogue already knows about the code
//! (`docs/zend_working_set.md`).
//!
//! Every dialogue carries already-ingested `code_reading` files, `repo_map`
//! folders and guidance, projected ahead of its own turns in the order they
//! entered: the files the conversation keeps attending to, by momentum,
//! starting with the **seeds** the schema names. The unit is always a whole
//! ingest conversation.

mod config;
pub mod marks;
mod state;
mod type_shares;

pub use config::WorkingSetConfig;
pub use state::{Candidate, Limits, WorkingSet};
pub use type_shares::{TypeCap, TypeShareSpec, TypeShares};
