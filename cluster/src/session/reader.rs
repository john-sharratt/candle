//! The session's reader: decodes each frame and hands it on — stream frames to
//! their connection, `PING` to a `PONG`, `EVENT` fragments to reassembly, and
//! everything else to the session's owner. A socket with nothing from the peer
//! for 45 s — no frame, and no bytes of one still arriving — is dead (§3.2); a
//! frame that fails to decode, a handshake frame after admission, or a stream
//! opened out of turn is a protocol violation, closed 4400.

use std::time::Duration;

use tokio::sync::mpsc::{Sender, UnboundedSender};
use tokio::time::error::Elapsed;
use tokio::time::timeout;

use super::command::Command;
use super::stream::{Registry, Shared, VirtualConn};
use super::{Ended, Incoming, Role};
use crate::close::{Close, CloseCode};
use crate::fragment::Reassembly;
use crate::frame::{name, Frame};
use crate::pipe::{Heard, Message, PipeRx};

pub const DEAD_AFTER: Duration = Duration::from_secs(45);

pub async fn run<R: PipeRx>(
    mut rx: R,
    heard: Heard,
    role: Role,
    commands: UnboundedSender<Command>,
    registry: Registry,
    incoming: Sender<Incoming>,
) -> Ended {
    let mut reassembly = Reassembly::new();
    // The peer's frames counted from one, as its writer counts them (§4.3).
    let mut number = 0u64;
    let violation = |reason: String| {
        let close = Close::new(CloseCode::Malformed, reason);
        let _ = commands.send(Command::Close(close.clone()));
        Ended::Refused(close)
    };
    let ended = loop {
        let message = match heard_within(&mut rx, &heard).await {
            Err(_) => {
                // How many frames came before the silence tells a peer that
                // never spoke from one that stopped.
                tracing::warn!(frames = number, "no frame from the peer for 45 s: closing");
                let close = Close::new(CloseCode::GoingAway, "no frame from the peer for 45 s");
                let _ = commands.send(Command::Close(close));
                break Ended::Dead;
            }
            Ok(None) => {
                let _ = commands.send(Command::Stop);
                break Ended::Dropped;
            }
            Ok(Some(Message::Close(close))) => {
                // Answer the close, as a WebSocket does, and stop writing.
                let _ = commands.send(Command::Close(close.clone()));
                break Ended::Closed(close);
            }
            Ok(Some(Message::Frame(bytes))) => bytes,
        };
        number += 1;
        let frame = match Frame::decode(&message) {
            Ok(f) => f,
            Err(e) => break violation(e.to_string()),
        };
        if frame.kind().is_handshake() {
            break violation(format!("{} after admission", name(frame.kind())));
        }
        match frame {
            Frame::Ping(nonce) => {
                let _ = commands.send(Command::Control(Frame::Pong(nonce)));
            }
            Frame::Pong(_) => {}
            Frame::Open(open) => {
                let taken = {
                    let mut streams = registry.lock().unwrap();
                    if role.opens(open.stream) || streams.contains_key(&open.stream) {
                        None
                    } else {
                        let shared = Shared::new();
                        streams.insert(open.stream, shared.clone());
                        Some(shared)
                    }
                };
                let Some(shared) = taken else {
                    break violation(format!("stream {} opened out of turn", open.stream));
                };
                let conn = VirtualConn::new(
                    open.stream,
                    open.target,
                    shared,
                    commands.clone(),
                    registry.clone(),
                );
                let _ = incoming.send(Incoming::Stream(conn)).await;
            }
            Frame::Data(data) => {
                let stream = registry.lock().unwrap().get(&data.stream).cloned();
                if let Some(stream) = stream {
                    stream.deliver(data.bytes);
                }
            }
            Frame::Window(window) => {
                let stream = registry.lock().unwrap().get(&window.stream).cloned();
                if let Some(stream) = stream {
                    stream.credit(window.credit as usize);
                }
            }
            Frame::Fin { stream } => {
                let stream = registry.lock().unwrap().get(&stream).cloned();
                if let Some(stream) = stream {
                    stream.finish();
                }
            }
            Frame::Reset { stream } => {
                let stream = registry.lock().unwrap().remove(&stream);
                if let Some(stream) = stream {
                    stream.reset();
                }
            }
            Frame::Event { catchup, event } => match reassembly.accept(event) {
                Ok(Some(whole)) => {
                    let _ = incoming
                        .send(Incoming::Event {
                            catchup,
                            whole,
                            frame: number,
                        })
                        .await;
                }
                Ok(None) => {}
                Err(e) => break violation(e.to_string()),
            },
            other => {
                let _ = incoming.send(Incoming::Frame(other)).await;
            }
        }
    };
    for (_, stream) in registry.lock().unwrap().drain() {
        stream.reset();
    }
    ended
}

/// The next message, unless the peer has gone [`DEAD_AFTER`] without being
/// heard. A message whose bytes are still arriving keeps the wait going: a
/// mebibyte fragment over a slow link takes minutes to arrive whole, from a
/// peer that is alive throughout. A whole message stamps `heard` too, so a
/// transport that cannot stamp its reads is heard by its messages.
async fn heard_within<R: PipeRx>(rx: &mut R, heard: &Heard) -> Result<Option<Message>, Elapsed> {
    let mut wait = DEAD_AFTER;
    loop {
        match timeout(wait, rx.recv()).await {
            Ok(message) => {
                if message.is_some() {
                    heard.stamp();
                }
                return Ok(message);
            }
            Err(elapsed) => match heard.at().map(|at| at.elapsed()) {
                Some(since) if since < DEAD_AFTER => wait = DEAD_AFTER - since,
                _ => return Err(elapsed),
            },
        }
    }
}
