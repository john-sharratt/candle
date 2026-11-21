//! Handles a `SubmitTurn` request by projecting the turn, lifting its working set,
//! carving its view and gap-filling its prefix.
//!
//! A submit over a parent that still has a turn registered first winds that turn
//! down, since its caller dropped the handle, and is refused when that earlier turn
//! is live or still prefilling, because racing a second decode onto the slot would
//! interleave their K/V. The projection's sealed history is priced as an admission
//! before it is lifted, so a burst of reads cannot take the expert zone's ground
//! unpriced. A submission still waiting on the slot is answered rather than left
//! waiting, because it can no longer be projected onto it. A resumed parent's
//! recurrent state is staged rather than restored, so a conversation holds no
//! device ground until one of its turns needs it.
//!
//! - `submit_turn`: the `SubmitTurn` handler and its wind-down, admission and projection steps.

use super::*;

mod submit_turn;
