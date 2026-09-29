//! Drive the real engine into KV fragmentation, then measure what it costs.
//!
//! The probe is the specification for the KV compaction pass and the gate that keeps it
//! honest. It stands up a real [`ConversationEngine`](crate::ConversationEngine) on a
//! scratch substrate, runs overlapping conversations until the arena pool is fragmented,
//! then asks three questions and fails on any of them:
//!
//! 1. **Are the answers still right?** A compaction relocates a chunk, and every holder
//!    of that chunk's identity has to be rewritten in the same window. Miss one and
//!    nothing faults — every address in the reservation is mapped — so a sequence reads
//!    whatever now occupies the vacated slot and answers from another sequence's KV.
//!    Only a content check sees that.
//! 2. **Is the ground below the arena frontier actually holding KV?** The frontier is
//!    what the weight side loses, so this is the number that sets expert residency and
//!    therefore decode.
//! 3. **Did the weight side take what the KV side released?** Lowering the frontier only
//!    makes it *possible* for `weight_floor` to move left; something has to move it.
//!
//! # Adding a model
//!
//! Add a [`ModelProfile`] row to [`profile::profiles`] and a test case naming it.
//! Everything else here is model-agnostic. The row's thresholds are **measured, not
//! copied** — see that module for why a threshold is a property of the model's KV
//! geometry rather than a constant.
//!
//! # Where it lives, and why it is not a test file
//!
//! In `src/` rather than `tests/` for the same reason
//! `candle_transformers::models::batch_test` is: both the integration tests and the
//! `kv_fragmentation` example driver call it, and a harness reachable from only one of
//! those ends up duplicated. The example exists because the interesting runs are long
//! and want flags; the tests exist so a model is one line to cover.

mod probe;
mod profile;
mod run;

pub use probe::{Probe, ProbeOutcome};
pub use profile::{names, profile, profiles, ModelProfile, StoryGate};
pub use run::{run, run_on_model};
