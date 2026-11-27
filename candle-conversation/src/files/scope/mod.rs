//! A reserved range's scope (`docs/placed_section_layout.md` §3.2, §3.4): the
//! sections placed in it, in the order they entered, and the holes the ones
//! that left behind them.
//!
//! - [`place`] — one place in the range: a section, or a hole;
//! - [`range_scope`] — the places, top first, and the moves that change them:
//!   entering in front, leaving a hole, sliding a full range, compacting;
//! - [`bars`] — the entry and eviction bars a step decides against;
//! - [`step`] — one step's decision from the sections' hit rates.

pub mod bars;
pub mod place;
pub mod range_scope;
pub mod step;

pub use bars::Bars;
pub use place::{Entry, Place};
pub use range_scope::RangeScope;
pub use step::{Candidate, StepChange};
