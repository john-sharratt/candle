//! The elastic warm tier (`docs/elastic_warm_tier.md`): the pinned part of the
//! expert warm tier, grown while this process is quiet and shrunk when it
//! pages or the machine's free RAM falls fast.
//!
//! | Module | Concern |
//! |---|---|
//! | [`policy`] | The law: readings in, grow / shrink / hold out |
//! | [`controller`] | The thread that reads, asks the policy and carries it out |
//! | [`region`] | The pinned segments the pad's resident slots index into |

pub(crate) mod controller;
pub(crate) mod policy;
pub(crate) mod region;
