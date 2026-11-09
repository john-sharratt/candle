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
//! | [`snapshot`] | a hosted world's map and sim as records |
//! | [`apply`] | records put into a hosted world |
//! | [`diff`] | what a beat changed |
//! | [`store`] | the records in the substrate |
//! | [`legacy`] | the one-time conversion of a world kept before records |

pub mod apply;
pub mod diff;
pub mod keys;
pub mod legacy;
pub mod renamed;
pub mod snapshot;
pub mod split;
pub mod store;

pub use apply::{apply, place_of, sim_of};
pub use diff::{diff, Changes};
pub use snapshot::{sim_records, sim_records_of, sim_value, world_records, Records};

use npc_map::world::World;

use crate::discord::dm::record::Dms;
use crate::discord::server::record::Servers;
use crate::discord::world::record::Mirror;
use crate::world::Hosted;

/// A hosted world's records as they stand now: its map's, its [`Sim`]'s, its
/// bots' DMs, servers and channels, and its own Discord server and what is
/// mirrored there (`docs/npcd_discord.md` §4.2).
///
/// Under the world lock only what needs it is done — the bodies' state copied
/// out, the sim serialised — and the records are built and split after.
///
/// [`Sim`]: crate::sim::Sim
pub fn records_of(hosted: &Hosted) -> Records {
    let state = hosted.read(World::state);
    let sim = hosted.sim(sim_value);
    let mut records = world_records(state);
    records.extend(sim.map(sim_records_of).unwrap_or_default());
    records.extend(hosted.dms(Dms::records));
    records.extend(hosted.servers(Servers::records));
    records.extend(hosted.mirror(Mirror::records));
    records
}
