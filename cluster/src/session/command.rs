//! What the session's writer is asked to send.

use super::lanes::PendingEvent;
use crate::close::Close;
use crate::frame::Frame;

pub enum Command {
    Control(Frame),
    /// Stream frames and `PULSE`, in the order produced.
    Ordered(Frame),
    CatchUp(Frame),
    Event(PendingEvent),
    /// Send this close at once and stop.
    Close(Close),
    /// The socket is gone: stop without sending.
    Stop,
}
