//! The tick scheduler: one loop per character, each waiting on its own inbox.
//!
//! Every character is an async task that sleeps on its own inbox, so an idle
//! character costs a parked future and no decode. An arrival that asks for a
//! response wakes the character it names at once, and quieter information waits
//! for that character's next wake. A pace sets how alert a character reads,
//! while a suspension, a rest or a pause takes a character off the schedule
//! until something reaches it.
//!
//! - `inbox`: a character's loop state, its inbox and its standing task.
//! - `pace`: how alert a character may become, and the unscheduled deadline.
//! - `records`: the Pulse rows a tick leaves, with ids as strings.
//! - `scheduler_events`: delivering events and handing back an unfinished turn.
//! - `scheduler_journal`: reading and drafting a character's journal.
//! - `scheduler_membership`: suspension, rest, pauses and the population.
//! - `scheduler_ticks`: the wait and the three phases of a tick.
//! - `tests_events`: news, wakes and the turn an act's answer is owed.
//! - `tests_journal`: journal cadence, day roll-over and the recent feed.
//! - `tests_scheduling`: due-ness, preemption, batching and ages.
//! - `tests_suspension`: suspension, rest and pauses.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use journal_core::journal::Holds;
use journal_core::section::JournalPrompt;
use journal_core::span::Span;
use serde::Serialize;
use tokio::sync::Notify;

use crate::engine::event::{Event, EventKind, Salience};
use crate::engine::held;
use crate::engine::journal::closing;
use crate::engine::journal::state::{JournalState, Waiting};
use crate::engine::mission::Mission;
use crate::engine::sleep::{DayAction, DayTracker};
use crate::engine::window::Window;
use crate::npcs;

/// The scheduler: every character's loop state, and the wakes that run them.
pub struct Scheduler {
    inboxes: Mutex<HashMap<u64, Inbox>>,
    /// Monotonic event sequence, shared by every character so the Pulse view can
    /// order arrivals across the whole cast.
    seq: AtomicU64,
    /// Monotonic tick counter, same reason.
    ticks: AtomicU64,
    /// Recent ticks, newest last. Bounded — this is an instrument, not a log.
    recent: Mutex<VecDeque<TickRecord>>,
    recent_cap: usize,
    /// The scheduler's own clock, for answering *how long ago*.
    ///
    /// Its own rather than the driver's: `drive` starts an `Instant` when the
    /// loop begins, minutes after `Runtime::new`, so a duration measured against
    /// one and read against the other is out by however long loading took. And
    /// its own rather than the wall clock, because the answer is handed to a
    /// browser whose clock is not this machine's — an age computed here is
    /// right whatever the reader's clock says, where a timestamp is not.
    born: Instant,
    /// The characters that are suspended: never due, whatever their inbox
    /// holds, until they are resumed. See [`Self::follow_suspended`].
    ///
    /// Kept beside the inboxes rather than in them, because whether a character
    /// is suspended is a fact about its record, not about this member's running
    /// of it: it holds for a character this member is not lent, and it has to be
    /// standing already on the moment a lease brings one here and wakes it.
    suspended: Mutex<HashSet<u64>>,
    /// The characters at rest — held as a suspended one is, by the engine
    /// rather than by their owner — each with the scheduler's clock when its
    /// rest began. See [`Self::rest`].
    resting: Mutex<HashMap<u64, u64>>,
    /// A digest of the whole-world view last carried to each mind that is its
    /// world. See [`Self::carry_whole`].
    whole: Mutex<HashMap<u64, u64>>,
    /// Where a character's conversation lines and purposes go instead of its
    /// inbox: its conversations answer them, never its mind
    /// (`docs/npc_conversations.md` §9). Set once by the runtime.
    talk: OnceLock<Talk>,
}

/// What takes a conversation's events off the scheduler.
pub type Talk = Box<dyn Fn(u64, EventKind) + Send + Sync>;

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(512)
    }
}

/// Shared handle.
pub type Shared = Arc<Scheduler>;

mod inbox;
mod pace;
mod records;
mod scheduler_events;
mod scheduler_journal;
mod scheduler_membership;
mod scheduler_ticks;
#[cfg(test)]
mod tests_events;
#[cfg(test)]
mod tests_journal;
#[cfg(test)]
mod tests_scheduling;
#[cfg(test)]
mod tests_suspension;

pub use inbox::*;
pub use pace::*;
pub use records::*;
