//! A message pipe: the WebSocket under a session, as two halves. A binary
//! message is one frame; a close carries its code and reason (§3.4). The
//! sender also reports how many bytes it has handed the transport that the
//! peer has not yet taken, which gates the catch-up lane (§4.2).
//!
//! [`memory::pair`] is the in-memory pipe the tests and harnesses run on;
//! [`socket::split`] puts the same two traits over a WebSocket, whose library's
//! messages a [`socket::Codec`] maps — [`tungstenite::Tungstenite`] for a
//! member's dialled socket.

pub mod drain;
pub mod heard;
pub mod memory;
pub mod socket;
pub mod tungstenite;

use std::future::Future;

use tokio::time::Instant;

pub use drain::Drain;
pub use heard::Heard;

use crate::close::Close;

/// One message on the pipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    Frame(Vec<u8>),
    Close(Close),
}

impl Message {
    /// The bytes it occupies in flight.
    pub fn len(&self) -> usize {
        match self {
            Message::Frame(b) => b.len(),
            Message::Close(c) => 2 + c.reason.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The pipe is gone: the peer closed it or dropped its half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipeClosed;

pub trait PipeTx: Send + 'static {
    /// Hand one message to the transport, waiting while it is full.
    fn send(&mut self, message: Message) -> impl Future<Output = Result<(), PipeClosed>> + Send;

    /// Bytes handed to the transport that the peer has not taken yet, and a
    /// wake when that falls.
    fn drain(&self) -> &Drain;

    fn unsent_bytes(&self) -> usize {
        self.drain().unsent()
    }
}

pub trait PipeRx: Send + 'static {
    /// The next message, or `None` once the pipe is gone.
    fn recv(&mut self) -> impl Future<Output = Option<Message>> + Send;

    /// When bytes last arrived from the peer, a message still arriving
    /// included; `None` when the transport cannot tell, and then only a whole
    /// message is heard.
    fn heard(&self) -> Option<Instant> {
        None
    }
}
