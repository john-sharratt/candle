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
//! The gateways only carry the vote: they hold no Paxos state and are never
//! asked to vote.

mod acceptor;
mod clock;
mod dedup;
mod maps;
mod paxos;
mod quorum;
pub mod rendezvous;
mod value;
pub mod votes;

pub use acceptor::DECIDED;
pub use clock::Millis;
pub use paxos::{Config, KeyStatus, Out, Paxos, HOLDER_MARGIN};
pub use quorum::{Change, MaxMembers};
pub use value::Value;
