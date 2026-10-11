//! A session: one admitted socket, multiplexing virtual connections, substrate
//! events and control frames (§4, §5.1).
//!
//! [`start`] splits the socket into a writer task — the three lanes of §4.2 —
//! and a reader task, and returns a [`SessionHandle`] to send with, a receiver
//! of what arrives ([`Incoming`]), and how the session ended.
//!
//! Stream ids are chosen by the side that opens: odd for the gateway, even for
//! a member, so the two never pick the same one.

mod command;
mod lanes;
mod reader;
mod stream;
mod writer;

pub use lanes::{PendingEvent, CATCHUP_LOW_WATER};
pub use reader::DEAD_AFTER;
pub use stream::{VirtualConn, INITIAL_CREDIT, WINDOW_STEP};
pub use writer::PING_EVERY;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::oneshot;

use command::Command;
use stream::{Registry, Shared};

use crate::close::Close;
use crate::fragment::{split, Whole};
use crate::frame::{Frame, Open};
use crate::pipe::{PipeRx, PipeTx};

/// How many arrivals wait for the owner before the reader stops reading.
const INCOMING_DEPTH: usize = 1024;

/// Which end of the socket this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Gateway,
    Member,
}

impl Role {
    /// Whether this end picks `stream` when it opens one.
    fn opens(self, stream: u32) -> bool {
        match self {
            Role::Gateway => !stream.is_multiple_of(2),
            Role::Member => stream.is_multiple_of(2),
        }
    }

    fn first_stream(self) -> u32 {
        match self {
            Role::Gateway => 1,
            Role::Member => 2,
        }
    }
}

/// What arrives on a session.
pub enum Incoming {
    /// A control or catch-up frame for the owner.
    Frame(Frame),
    /// A record, reassembled from its fragments.
    Event { catchup: bool, whole: Whole },
    /// The peer opened a virtual connection.
    Stream(VirtualConn),
}

/// How a session ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ended {
    /// The peer closed it.
    Closed(Close),
    /// This end closed it for a protocol violation.
    Refused(Close),
    /// The transport dropped.
    Dropped,
    /// No frame for 45 s.
    Dead,
}

/// The session ended before the request could be queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionGone;

/// Sends on a session. Cheap to clone.
#[derive(Clone)]
pub struct SessionHandle {
    role: Role,
    commands: UnboundedSender<Command>,
    registry: Registry,
    next_stream: Arc<AtomicU32>,
}

impl SessionHandle {
    /// Open a virtual connection towards `target`.
    pub fn open(&self, target: &str) -> Result<VirtualConn, SessionGone> {
        let id = self.next_stream.fetch_add(2, Ordering::Relaxed);
        debug_assert!(self.role.opens(id));
        let shared = Shared::new();
        self.registry.lock().unwrap().insert(id, shared.clone());
        self.commands
            .send(Command::Ordered(Frame::Open(Open {
                stream: id,
                target: target.to_string(),
            })))
            .map_err(|_| SessionGone)?;
        Ok(VirtualConn::new(
            id,
            target.to_string(),
            shared,
            self.commands.clone(),
            self.registry.clone(),
        ))
    }

    /// Queue a control frame: it goes before anything else.
    pub fn control(&self, frame: Frame) -> Result<(), SessionGone> {
        self.commands
            .send(Command::Control(frame))
            .map_err(|_| SessionGone)
    }

    /// Queue a frame on the ordered lane (`PULSE`).
    pub fn ordered(&self, frame: Frame) -> Result<(), SessionGone> {
        self.commands
            .send(Command::Ordered(frame))
            .map_err(|_| SessionGone)
    }

    /// Queue a catch-up frame (`HEADERS`, `DONE`, `HAVE`, `BUILD_CHUNK`).
    pub fn catchup(&self, frame: Frame) -> Result<(), SessionGone> {
        self.commands
            .send(Command::CatchUp(frame))
            .map_err(|_| SessionGone)
    }

    /// Queue one record as `EVENT` fragments: live on the ordered lane, or on
    /// the catch-up lane when `catchup` or addressed to `reply_to`. Resolves
    /// to the session frame number its last fragment went out as; dropped
    /// unsent if the session ends first.
    pub fn event(
        &self,
        origin: u64,
        oseq: u64,
        record: &[u8],
        reply_to: Option<u64>,
        catchup: bool,
    ) -> Result<oneshot::Receiver<u64>, SessionGone> {
        let (done, sent) = oneshot::channel();
        self.commands
            .send(Command::Event(PendingEvent {
                fragments: split(origin, oseq, record, reply_to).into(),
                catchup,
                done: Some(done),
            }))
            .map_err(|_| SessionGone)?;
        Ok(sent)
    }

    /// Close the session with `close`, sent ahead of anything queued.
    pub fn close(&self, close: Close) {
        let _ = self.commands.send(Command::Close(close));
    }
}

/// A started session.
pub struct Session {
    pub handle: SessionHandle,
    pub incoming: mpsc::Receiver<Incoming>,
    pub ended: oneshot::Receiver<Ended>,
}

/// Run a session over an admitted socket.
pub fn start<T: PipeTx, R: PipeRx>(role: Role, tx: T, rx: R) -> Session {
    let (commands, command_rx) = mpsc::unbounded_channel();
    let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
    let (incoming_tx, incoming) = mpsc::channel(INCOMING_DEPTH);
    let (ended_tx, ended) = oneshot::channel();
    tokio::spawn(writer::run(tx, command_rx));
    tokio::spawn({
        let commands = commands.clone();
        let registry = registry.clone();
        async move {
            let how = reader::run(rx, role, commands, registry, incoming_tx).await;
            let _ = ended_tx.send(how);
        }
    });
    Session {
        handle: SessionHandle {
            role,
            commands,
            registry,
            next_stream: Arc::new(AtomicU32::new(role.first_stream())),
        },
        incoming,
        ended,
    }
}

#[cfg(test)]
mod tests;
