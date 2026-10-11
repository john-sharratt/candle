//! A member's replication of its substrate (§6, §7): its own events out to the
//! group, the group's events in, catch-ups after an absence and after a
//! restart, and the rejoin of a member that wrote while cut off.
//!
//! The substrate is a [`Store`]; [`MemoryStore`] is one in memory, which the
//! tests and the burst harnesses replicate between. The [`Replicator`] runs
//! one app's replication on a member and is driven by its sockets.

mod answer;
mod catchup;
mod consensus;
mod memory;
mod receive;
mod replicator;
mod send;
mod serve;
mod store;
mod tick;
mod watch;

pub use answer::{answer_ranges, complete_ranges, HEADERS_PER_FRAME};
pub use catchup::{Progress, Purpose, Requester, FETCH_TRIES, FETCH_WINDOW};
pub use consensus::{REJOIN_RETRY, REJOIN_WAIT};
pub use memory::{digest, MemoryRecord, MemoryStore, Stamp};
pub use replicator::{CatchupView, Phase, Replication, ReplicationView, Replicator, Stats};
pub use send::{FLUSH_WAIT, GRACE};
pub use store::{Applied, Listing, Store, Tap, GEN_DETACHED};
pub use tick::QUIET;
pub use watch::{periodic_due, GapWatch, GAP_WAIT, PERIODIC};
