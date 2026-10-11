//! A world's state as records (`docs/replicated_substrate.md` §9.1).
//!
//! What a world *is* — where each body stands, the lift, and everything its
//! [`crate::sim::Sim`] holds: the command table, each mission and operation,
//! each body's working set and pack, each device, thread and posting — is one
//! keyed record per entity, `world/<wid>/<entity>`. Only the world's authority
//! writes them, each beat writing only the entities that changed, stamped with
//! its lease's epoch; an entity that ceases gets a tombstone on its key. Every
//! other member holds the same records, replicated like any other, and a world
//! is hosted, followed and reloaded from them.
//!
//! | Module | |
//! |---|---|
//! | [`keys`] | where the records live |
//! | [`split`] | a facet taken apart into one record per entity, and back |
//! | [`snapshot`] | a hosted world as records |
//! | [`apply`] | records put into a hosted world |
//! | [`diff`] | what a beat changed |
//! | [`store`] | the records in the substrate |
//! | [`legacy`] | the one-time conversion of a world kept before records |

pub mod apply;
pub mod diff;
pub mod keys;
pub mod legacy;
pub mod snapshot;
pub mod split;
pub mod store;

pub use apply::{apply, place_of, sim_of};
pub use diff::{diff, Changes};
pub use snapshot::{sim_records, world_records, Records};

use crate::world::Hosted;

/// A hosted world's records as they stand now.
pub fn records_of(hosted: &Hosted) -> Records {
    let mut records = hosted.read(world_records);
    records.extend(hosted.sim(sim_records));
    records
}
