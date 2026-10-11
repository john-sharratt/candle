//! Delivery bookkeeping (§4.3, §6.3): the sender's outbox of its own events,
//! and the frame counting `RECEIPT`s refer to.

mod outbox;
mod receipts;

pub use outbox::{EventState, Outbox, SocketId};
pub use receipts::{FrameCounter, ReceiptPacer};
