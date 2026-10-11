//! Records another member wrote, applied here
//! (`docs/replicated_substrate.md` §6.5).
//!
//! - [`Incoming`] parses one record as it arrives and checks its payload.
//! - [`SubstratePersistence::apply_remote`](super::SubstratePersistence::apply_remote)
//!   decides each record of a batch — deferred, duplicate, dominated,
//!   superseded or written — and adopts the turns the batch completed.
//! - [`turn_is_complete`] says when a turn's records are all here.

mod adopt;
mod apply;
#[cfg(test)]
mod apply_tests;
mod completeness;
mod incoming;
mod outcome;

pub use completeness::turn_is_complete;
pub use incoming::Incoming;
pub use outcome::{Refusal, RemoteBatch, RemoteOutcome};
