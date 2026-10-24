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
mod gauge;
mod lanes;
mod reader;
mod stream;
mod writer;

pub use gauge::{Gauges, Lane};
pub use lanes::{PendingEvent, CATCHUP_LOW_WATER};
pub use reader::DEAD_AFTER;
pub use stream::{VirtualConn, INITIAL_CREDIT, WINDOW_STEP};
pub use writer::PING_EVERY;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::oneshot;
use tokio::time::Instant;

use command::Command;
use stream::{Registry, Shared};

use crate::close::Close;
use crate::fragment::{split, Whole};
use crate::frame::{Frame, Open};
use crate::pipe::{Heard, PipeRx, PipeTx};

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
    /// A record, reassembled from its fragments. `frame` is the session frame
    /// number of its last fragment: once it is relayed, every frame through
    /// that number has been taken in (§4.3).
    Event {
        catchup: bool,
        whole: Whole,
        frame: u64,
    },
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
    gauges: Arc<Gauges>,
    heard: Heard,
}

impl SessionHandle {
    /// When the peer was last heard: a whole frame, or the bytes of one still
    /// arriving when the transport can tell.
    pub fn heard(&self) -> Option<Instant> {
        self.heard.at()
    }

    /// Whether the session has ended: nothing sent on it is delivered, and
    /// [`Self::open`] fails.
    pub fn is_gone(&self) -> bool {
        self.commands.is_closed()
    }

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
        // Gauged before it is queued, so the writer never takes off what was
        // not yet put on.
        let bytes = frame.encoded_len();
        self.gauges.queued(Lane::CatchUp, bytes);
        self.commands.send(Command::CatchUp(frame)).map_err(|_| {
            self.gauges.written(Lane::CatchUp, bytes);
            SessionGone
        })
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
        let pending = PendingEvent {
            fragments: split(origin, oseq, record, reply_to).into(),
            catchup,
            done: Some(done),
        };
        let lane = pending.lane();
        self.gauges.queued(lane, record.len());
        self.commands.send(Command::Event(pending)).map_err(|_| {
            self.gauges.written(lane, record.len());
            SessionGone
        })?;
        Ok(sent)
    }

    /// What this session holds queued and unwritten, by lane (§4.4).
    pub fn gauges(&self) -> &Gauges {
        &self.gauges
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
    let gauges = Arc::new(Gauges::default());
    let heard = rx.heard().unwrap_or_default();
    tokio::spawn(writer::run(tx, command_rx, gauges.clone()));
    tokio::spawn({
        let commands = commands.clone();
        let registry = registry.clone();
        let heard = heard.clone();
        async move {
            let how = reader::run(rx, heard, role, commands, registry, incoming_tx).await;
            let _ = ended_tx.send(how);
        }
    });
    Session {
        handle: SessionHandle {
            role,
            commands,
            registry,
            next_stream: Arc::new(AtomicU32::new(role.first_stream())),
            gauges,
            heard,
        },
        incoming,
        ended,
    }
}

#[cfg(test)]
mod tests;
