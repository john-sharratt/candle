//! The cluster simulator (`docs/cooperative_admission.md` §3): members whose
//! forwards cost what their wave logs measured, one link between them, and a
//! placement policy deciding where each turn runs.
//!
//! | Module | |
//! |---|---|
//! | [`model`] | a member's step costs and how residency loss slows them |
//! | [`link`] | the path between members, FIFO |
//! | [`workload`] | conversations, turns and their prefill/decode segments |
//! | [`policy`] | the placements compared |
//! | [`engine`] | the event loop |
//! | [`report`] | what a run comes to |

pub mod engine;
pub mod link;
pub mod model;
pub mod policy;
pub mod report;
pub mod workload;
