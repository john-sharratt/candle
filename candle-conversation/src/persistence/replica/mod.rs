//! What a member has applied, what it deferred, and where every record it
//! holds is (`docs/replicated_substrate.md` §6.2, §6.5, §7.2).
//!
//! - The **applied set** ([`AppliedSet`]) is where a catch-up is up to: per
//!   origin, everything through a number and the ranges held above it.
//! - The **deferred set** ([`DeferredRecord`]) holds the records not written
//!   because this build does not decode their type.
//! - Both persist in a `Replica` record ([`ReplicaPayload`]).
//! - The **location index** ([`LocationIndex`]) finds a record by its
//!   identity, wherever maintenance has moved it.

mod applied_set;
mod location_index;
mod payload;

pub use applied_set::{AppliedSet, OriginApplied};
pub use location_index::{Located, LocationIndex};
pub use payload::{DeferredRecord, ReplicaPayload};
