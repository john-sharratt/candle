//! The session's reader: decodes each frame and hands it on — stream frames to
//! their connection, `PING` to a `PONG`, `EVENT` fragments to reassembly, and
//! everything else to the session's owner. A socket with nothing from the peer
//! for 45 s — no frame, and no bytes of one still arriving — is dead (§3.2); a
//! frame that fails to decode, a handshake frame after admission, or a stream
//! opened out of turn is a protocol violation, closed 4400. On a session with a
//! member on an older protocol, only the frames that cross protocols are read
//! at all.

use std::time::Duration;

use tokio::sync::mpsc::{Sender, UnboundedSender};
use tokio::time::error::Elapsed;
use tokio::time::timeout;

use super::command::Command;
use super::hearing::Hearing;
use super::stream::{Registry, Shared, VirtualConn};
use super::{Ended, Incoming, Role};
use crate::close::{Close, CloseCode};
use crate::fragment::Reassembly;
use crate::frame::{name, Frame, Kind};
use crate::pipe::{Message, PipeRx};

pub const DEAD_AFTER: Duration = Duration::from_secs(45);

/// Whether the frame in `message` crosses protocols, by its tag.
fn crosses(message: &[u8]) -> bool {
    message
        .first()
        .and_then(|tag| Kind::from_tag(*tag).ok())
        .is_some_and(Kind::crosses_protocols)
}

pub async fn run<R: PipeRx>(
    mut rx: R,
    hearing: Hearing,
    role: Role,
    behind: bool,
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
        let message = match heard_within(&mut rx, &hearing).await {
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
        // Read by its tag alone: its body may be a layout this build cannot
        // decode, which is the reason the session is on this footing.
        if behind && !crosses(&message) {
            continue;
        }
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
                if hand_on(&incoming, Incoming::Stream(conn)).await.is_err() {
                    break stalled(&commands);
                }
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
            Frame::Event { catchup, event } => {
                // A fetched record's fragment is a catch-up's answer arriving,
                // though the record is whole only at its last.
                if catchup || event.reply_to.is_some() {
                    hearing.catchup();
                }
                match reassembly.accept(event) {
                    Ok(Some(whole)) => {
                        let event = Incoming::Event {
                            catchup,
                            whole,
                            frame: number,
                        };
                        if hand_on(&incoming, event).await.is_err() {
                            break stalled(&commands);
                        }
                    }
                    Ok(None) => {}
                    Err(e) => break violation(e.to_string()),
                }
            }
            other => {
                if matches!(other, Frame::Headers(_) | Frame::Done(_)) {
                    hearing.catchup();
                }
                if hand_on(&incoming, Incoming::Frame(other)).await.is_err() {
                    break stalled(&commands);
                }
            }
        }
    };
    for (_, stream) in registry.lock().unwrap().drain() {
        stream.reset();
    }
    ended
}

/// The owner took nothing within [`DEAD_AFTER`].
struct OwnerStalled;

/// Hand `item` to the session's owner, unless it takes nothing for
/// [`DEAD_AFTER`].
///
/// **The hand-off is bounded like the read.** While the reader waits here it
/// reads nothing: no ping is answered, no close is seen, and the silence check
/// around the read never runs. Unbounded, an owner that stopped taking frames
/// held its sessions open forever — on WhiteWorld (2026-11-20) npcd's and
/// zend's both reported `admitted` with no close for an hour, the gateway long
/// gone, while 27,229 events queued unsent. A dropped receiver — the owner
/// gone — is no stall: the item has nowhere to go, and the session ends as
/// its owner does.
async fn hand_on(incoming: &Sender<Incoming>, item: Incoming) -> Result<(), OwnerStalled> {
    match timeout(DEAD_AFTER, incoming.send(item)).await {
        Ok(_) => Ok(()),
        Err(_) => Err(OwnerStalled),
    }
}

/// Close a session whose owner stalled, saying so.
fn stalled(commands: &UnboundedSender<Command>) -> Ended {
    tracing::warn!(
        "this member took no frame from its session for 45 s: closing it, so it is redialled \
         and the stall is seen"
    );
    let close = Close::new(
        CloseCode::GoingAway,
        "this member took no frame from its session for 45 s",
    );
    let _ = commands.send(Command::Close(close));
    Ended::Stalled
}

/// The next message, unless the peer has gone [`DEAD_AFTER`] without being
/// heard. A message whose bytes are still arriving keeps the wait going: a
/// mebibyte fragment over a slow link takes minutes to arrive whole, from a
/// peer that is alive throughout. Each whole message is noted in `hearing`.
async fn heard_within<R: PipeRx>(
    rx: &mut R,
    hearing: &Hearing,
) -> Result<Option<Message>, Elapsed> {
    let mut wait = DEAD_AFTER;
    loop {
        match timeout(wait, rx.recv()).await {
            Ok(message) => {
                if message.is_some() {
                    hearing.message();
                }
                return Ok(message);
            }
            Err(elapsed) => match hearing.last().map(|at| at.elapsed()) {
                Some(since) if since < DEAD_AFTER => wait = DEAD_AFTER - since,
                _ => return Err(elapsed),
            },
        }
    }
}
