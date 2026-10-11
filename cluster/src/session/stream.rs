//! A virtual connection (§5.1): one stream's bytes as `AsyncRead +
//! AsyncWrite`, which both ends hand to the HTTP machinery they already run.
//!
//! Flow control (§4.2): each direction starts with 1 MiB of credit. The writer
//! spends credit as it sends `DATA`; the reader returns it in `WINDOW` frames
//! as it consumes, so a slow reader stalls only its own stream.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc::UnboundedSender;

use super::command::Command;
use crate::frame::{Data, Frame, Window, MAX_DATA};

/// The credit a stream starts with, each way.
pub const INITIAL_CREDIT: usize = 1 << 20;
/// Consumed bytes are returned in a `WINDOW` once this many have built up.
pub const WINDOW_STEP: usize = 64 << 10;

#[derive(Default)]
struct State {
    recv: VecDeque<Vec<u8>>,
    /// How much of `recv`'s front chunk has been read.
    recv_off: usize,
    eof: bool,
    reset: bool,
    read_waker: Option<Waker>,
    send_credit: usize,
    write_waker: Option<Waker>,
    /// Consumed and not yet returned as credit.
    unreturned: usize,
}

/// A stream's state, shared by its connection and the session's reader.
pub struct Shared {
    state: Mutex<State>,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                send_credit: INITIAL_CREDIT,
                ..State::default()
            }),
        })
    }

    /// Bytes arrived for the reader.
    pub fn deliver(&self, bytes: Vec<u8>) {
        let mut s = self.state.lock().unwrap();
        if s.reset || s.eof {
            return;
        }
        s.recv.push_back(bytes);
        if let Some(w) = s.read_waker.take() {
            w.wake();
        }
    }

    /// The peer returned `credit`.
    pub fn credit(&self, credit: usize) {
        let mut s = self.state.lock().unwrap();
        s.send_credit += credit;
        if let Some(w) = s.write_waker.take() {
            w.wake();
        }
    }

    /// The peer finished sending.
    pub fn finish(&self) {
        let mut s = self.state.lock().unwrap();
        s.eof = true;
        if let Some(w) = s.read_waker.take() {
            w.wake();
        }
    }

    /// The stream was aborted, or its session ended.
    pub fn reset(&self) {
        let mut s = self.state.lock().unwrap();
        s.reset = true;
        if let Some(w) = s.read_waker.take() {
            w.wake();
        }
        if let Some(w) = s.write_waker.take() {
            w.wake();
        }
    }
}

/// Every open stream of a session, by id.
pub type Registry = Arc<Mutex<HashMap<u32, Arc<Shared>>>>;

/// One virtual connection.
pub struct VirtualConn {
    id: u32,
    target: String,
    shared: Arc<Shared>,
    commands: UnboundedSender<Command>,
    registry: Registry,
    write_closed: bool,
}

impl VirtualConn {
    pub(super) fn new(
        id: u32,
        target: String,
        shared: Arc<Shared>,
        commands: UnboundedSender<Command>,
        registry: Registry,
    ) -> Self {
        Self {
            id,
            target,
            shared,
            commands,
            registry,
            write_closed: false,
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    /// What the opener asked for (§5.1).
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Abort the stream: the peer's reads and writes fail.
    pub fn reset(mut self) {
        self.abort();
    }

    fn abort(&mut self) {
        if !self.write_closed {
            self.write_closed = true;
            let _ = self
                .commands
                .send(Command::Ordered(Frame::Reset { stream: self.id }));
        }
        self.shared.reset();
    }
}

fn reset_error() -> io::Error {
    io::Error::new(io::ErrorKind::ConnectionReset, "the stream was reset")
}

impl AsyncRead for VirtualConn {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let mut s = this.shared.state.lock().unwrap();
        if s.recv.is_empty() {
            if s.reset {
                return Poll::Ready(Err(reset_error()));
            }
            if s.eof {
                return Poll::Ready(Ok(()));
            }
            s.read_waker = Some(cx.waker().clone());
            return Poll::Pending;
        }
        let mut read = 0;
        while buf.remaining() > 0 {
            let off = s.recv_off;
            let Some(front) = s.recv.front() else { break };
            let n = (front.len() - off).min(buf.remaining());
            buf.put_slice(&front[off..off + n]);
            read += n;
            if off + n == front.len() {
                s.recv.pop_front();
                s.recv_off = 0;
            } else {
                s.recv_off = off + n;
            }
        }
        s.unreturned += read;
        if s.unreturned >= WINDOW_STEP {
            let credit = s.unreturned;
            s.unreturned = 0;
            let _ = this.commands.send(Command::Control(Frame::Window(Window {
                stream: this.id,
                credit: credit as u32,
            })));
        }
        Poll::Ready(Ok(()))
    }
}

impl AsyncWrite for VirtualConn {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        let mut s = this.shared.state.lock().unwrap();
        if s.reset {
            return Poll::Ready(Err(reset_error()));
        }
        if this.write_closed {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "the stream's write side is finished",
            )));
        }
        if s.send_credit == 0 {
            s.write_waker = Some(cx.waker().clone());
            return Poll::Pending;
        }
        let n = buf.len().min(MAX_DATA).min(s.send_credit);
        s.send_credit -= n;
        drop(s);
        let sent = this.commands.send(Command::Ordered(Frame::Data(Data {
            stream: this.id,
            bytes: buf[..n].to_vec(),
        })));
        match sent {
            Ok(()) => Poll::Ready(Ok(n)),
            Err(_) => Poll::Ready(Err(reset_error())),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if !this.write_closed {
            this.write_closed = true;
            let _ = this
                .commands
                .send(Command::Ordered(Frame::Fin { stream: this.id }));
        }
        Poll::Ready(Ok(()))
    }
}

impl Drop for VirtualConn {
    fn drop(&mut self) {
        // A connection dropped without finishing its write side was abandoned:
        // the peer is told, rather than left waiting for bytes that never come.
        if !self.write_closed {
            self.abort();
        }
        self.registry.lock().unwrap().remove(&self.id);
    }
}
