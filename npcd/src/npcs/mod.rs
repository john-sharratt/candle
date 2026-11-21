//! The character registry: every character this daemon knows, held in memory
//! and backed by the substrate's redo log.
//!
//! Every read is a lookup in the published cast, which takes no lock and stays
//! whole for as long as a reader holds it. The log is read once, at load, and
//! the substrate handle opened there is the one the engine adopts, because a
//! second writable handle would build an index that misses characters created
//! later and drop them at the next compaction. Each write appends one record per
//! character and supersedes the last. Deleting writes a tombstone, which keeps
//! the id taken. Writes run on the persistence owner's thread, and the cast is
//! published only after its records are durable.
//!
//! - `methods`: the reads and writes a character's record takes.
//! - `validation`: the bounds and state rules a record must meet.
//! - `wire`: the JSON and base-36 forms characters take on the wire.
//! - Tests: `tests_concurrency`, `tests_fields`, `tests_lifecycle`.

mod cast;
mod suspend;

use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;
use std::mem;
use std::path::Path;
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use candle_conversation::models::identity::LegacySpec;
use candle_conversation::persistence::identity::RecordIdentity;
use candle_conversation::persistence::record::{
    AuthoredBelief, AuthoredRelationship, AuthoredStrategy, Modulation, NpcPayload, RecordType,
};
use candle_conversation::persistence::{
    Admit, OpenProgress, SharedSubstrate, SubstratePersistence, SUBSTRATE_DIR,
};
use candle_conversation::substrate::Substrate;
use host::MODEL;
use serde_json::{json, Value};
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::task::block_in_place;

use web::auth::session::Identity;

use crate::registry::id;

pub use cast::Cast;
pub use suspend::Manager;

#[derive(Debug)]
pub enum NpcError {
    /// The caller is not this character's owner. Deliberately indistinguishable
    /// from "no such character" at the API boundary — see [`Npcs::visible_to`].
    NotFound,
    Invalid(&'static str),
    Persist(String),
}

/// Which strategy an agency write lands on.
#[derive(Clone, Copy, Debug)]
pub enum StrategyTarget<'a> {
    /// The strategy with this id — created, or replaced.
    Id(&'a str),
    /// A new child of this strategy, its id minted from the record the write
    /// supersedes (`docs/npcd_cast_lock_free.md` §2.1).
    ChildOf(&'a str),
}

/// The cast, and the one handle to the substrate it is written through.
pub struct Npcs {
    /// The published cast. Replaced only on the persistence owner's thread,
    /// after the records that make it different are durable.
    current: ArcSwap<Cast>,
    /// **The daemon's only handle to the substrate**, shared with the engine
    /// rather than opened twice. See [`Npcs::substrate`].
    shared: SharedSubstrate,
    /// Characters a write took in from records replicated since the last
    /// refresh, for the next [`Npcs::refresh`] to report as changed: the write
    /// published them, so the refresh no longer finds them moved.
    taken_in: Mutex<Vec<u64>>,
}

mod methods;
#[cfg(test)]
mod tests_concurrency;
#[cfg(test)]
mod tests_fields;
#[cfg(test)]
mod tests_lifecycle;
mod validation;
mod wire;

pub use methods::*;
pub use validation::*;
pub use wire::*;
