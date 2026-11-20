//! The elastic warm tier (`docs/elastic_warm_tier.md`): the pageable part of
//! the expert warm tier, grown while this process is quiet and shrunk when it
//! pages or the machine's free RAM falls fast. The pinned part is sized once
//! at startup and never touched.
//!
//! | Module | Concern |
//! |---|---|
//! | [`policy`] | The law: readings in, grow / shrink / hold out |
//! | [`controller`] | The thread that reads, asks the policy and carries it out |
//! | [`paged`] | The pageable chunks |

pub(crate) mod controller;
pub(crate) mod paged;
pub(crate) mod policy;
