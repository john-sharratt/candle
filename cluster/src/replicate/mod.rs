//! A member's replication of its substrate (§6, §7): its own events out to the
//! group, the group's events in, catch-ups after an absence and after a
//! restart, and the rejoin of a member that wrote while cut off.
//!
//! The substrate is a [`Store`]; [`MemoryStore`] is one in memory, which the
//! tests and the burst harnesses replicate between. The [`Replicator`] runs
//! one app's replication on a member and is driven by its sockets.

mod answer;
mod answering;
mod busy_warnings;
mod catchup;
mod consensus;
mod coord;
mod drain;
mod fault;
mod lease;
mod memory;
mod pulse;
mod quorum_governor;
mod receive;
mod replicator;
mod send;
mod serve;
mod store;
mod tick;
mod upgrade;
#[cfg(test)]
mod view_tests;
mod watch;

pub use answer::{answer_ranges, complete_ranges, HEADERS_PER_FRAME};
pub use catchup::{Progress, Purpose, Requester, FETCH_TRIES, FETCH_WINDOW};
pub use consensus::{REJOIN_RETRY, REJOIN_WAIT};
pub use coord::{COMMANDS_KEPT, COORD_EVERY};
pub use drain::HandedBack;
pub use fault::{floor, judge, Cause, Fault, Verdict, FLOOR_MIN};
pub use lease::{LeaseEvent, LEASE_TTL, RENEW_EVERY};
pub use memory::{digest, MemoryRecord, MemoryStore, Stamp};
pub use quorum_governor::{Downgrade, EXECUTE_AFTER, FLOOR as QUORUM_FLOOR, PROPOSE_AFTER};
pub use replicator::{
    AppCoordFn, Ask, CatchupView, Phase, QuorumView, Replication, ReplicationView, Replicator,
    Stats, Waiting,
};
pub use send::{FLUSH_WAIT, GRACE};
pub use store::{Applied, Disk, Listing, Store, Tap, Tombstones, GEN_DETACHED};
pub use tick::QUIET;
pub use upgrade::UPGRADE_PREFIX;
pub use watch::{periodic_due, GapWatch, GAP_WAIT, PERIODIC};
