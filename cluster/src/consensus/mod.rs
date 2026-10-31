//! Consensus among an app's members (§5.9): Paxos with durable acceptors, over
//! whatever carries the frames.
//!
//! - [`Paxos`]: a member's engine — its acceptor, its rounds, its leases.
//!   Two primitives: a **lease** `(key, ttl)` and a one-time **decision**
//!   `(key, value)`.
//! - [`votes`]: the acceptor's durable state, the preallocated vote journal.
//! - [`MaxMembers`]: the quorum's base, raised and set in any order.
//! - [`rendezvous`]: who proposes for a lease, and where a character is lent.
//!
//! The gateways carry the vote. The one with the lowest node id is also the
//! tie-break acceptor for a group of two (§5.9): its promise completes that
//! quorum when one member is down, and it counts in no larger group.

mod acceptor;
mod armed;
mod clock;
mod dedup;
mod maps;
mod paxos;
mod quorum;
mod rearm;
pub mod rendezvous;
mod value;
pub mod votes;
mod wall;

pub use acceptor::DECIDED;
pub use armed::{remaining_s, ArmedLease, LeaseRow};
pub use clock::Millis;
pub use paxos::{seats, Config, KeyStatus, Out, Paxos, HOLDER_MARGIN};
pub use quorum::{Change, MaxMembers};
pub use value::{Value, MAX_DECISION};
pub use wall::WallMillis;
