//! Runs one chat turn as an async task, from the decode through to the reply the
//! client sees.
//!
//! Every wait in the turn is an await, so no runtime thread is pinned while the
//! decode runs. The per-conversation guard held across the decode is what keeps a
//! second submit for the same conversation from running beside it. The turn is
//! kept whole even when the client leaves, so the substrate never holds half a
//! turn decided by a socket. Each round seals before its tools run, and the tool
//! loop is bounded by a cap.
//!
//! - `stream_turn`: the turn task itself.

use super::*;

mod stream_turn;

pub(super) use stream_turn::*;
