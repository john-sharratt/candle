//! The segment-vector description of a turn's K/V, and what each segment occupies.
//!
//! A turn is an ordered list of segments, and their running lengths place every
//! real segment, so the vector is the complete description of the turn's K/V.
//! There are no separate boundary offsets to fall out of step with it. Real
//! segments occupy K/V and inject stored K/V, while ethereal ones are recorded but
//! not materialised. `validate_tiling` enforces the invariant, and the tests check
//! it against hand-built grids.
//!
//! - `kv_span`: a segment's start offset and length in the grid.
//! - `layout`: the turn's segments and the accessors over them.
//! - `segment`: the segment kinds and their real or ethereal state.
//! - `thinking_length`: a turn's reasoning length, exact or estimated.
//!
//! - `tests_grid`: tiling and realisation of the grid.
//! - `tests_sealed`: sealed turns keep their own boundaries and flags.
//! - `tests_thinking`: the reasoning split and phase spans.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::normalization::Phase;

mod kv_span;
mod layout;
mod segment;
#[cfg(test)]
mod tests_grid;
#[cfg(test)]
mod tests_sealed;
#[cfg(test)]
mod tests_thinking;
mod thinking_length;

pub use kv_span::*;
pub use layout::*;
pub use segment::*;
pub use thinking_length::*;
