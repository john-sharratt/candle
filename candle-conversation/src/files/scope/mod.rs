//! A reserved range's scope (`docs/placed_section_layout.md` §3.2, §3.4): the
//! sections placed in it, strongest at the top beside the question, and the
//! holes the ones that left behind them.
//!
//! - [`place`] — one place in the range: a section, or a hole;
//! - [`range_scope`] — the places, top first, and the moves that change them:
//!   entering in front, leaving a hole, making room from the front of a full
//!   range, reordering by momentum, compacting;
//! - [`momentum`] — each section's momentum over the decode, the order;
//! - [`bars`] — the entry and eviction bars a step decides against;
//! - [`step`] — the opening, and the entries a step's hit rates admit;
//! - [`watch`] — the placed sections a step's hit rates evict;
//! - [`resident`] — the unplaced sections kept hot beside the range.

pub mod bars;
pub mod momentum;
pub mod nucleus;
pub mod place;
pub mod range_scope;
pub mod resident;
pub mod step;
pub mod watch;

pub use bars::{band, Bars};
pub use momentum::{Momentum, MOMENTUM_KEEP};
pub use nucleus::{
    nucleus, relock, shares, EVICT_OF_EVEN, KEEP_OF_LOCKED, P_LOCK, P_UNLOCK, REORDER_SHARE,
};
pub use place::{Entry, Place};
pub use range_scope::RangeScope;
pub use resident::{resident, RESIDENT_RANGES};
pub use step::{Candidate, StepChange};
pub use watch::EvictWatch;
