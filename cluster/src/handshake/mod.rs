//! The handshake (§3.3): `HELLO`, `CHALLENGE`, `PROVE`, then `ADMIT` or a
//! close. Nothing but these frames crosses the socket before `ADMIT`; a peer
//! that sends anything else is closed 4400.

mod gateway;
mod key;
mod limiter;
mod member;
mod proof;

pub use gateway::admit;
pub use key::{GroupKey, KeyError};
pub use limiter::ProofLimiter;
pub use member::join;
pub use proof::{gateway_proof, member_proof, nonce, verify_gateway, verify_member};

use std::time::Duration;

use thiserror::Error;
use tokio::time::timeout;

use crate::close::{Close, CloseCode};
use crate::frame::{name, Frame};
use crate::pipe::{Message, PipeClosed, PipeRx, PipeTx};

/// The protocol version this build speaks.
pub const PROTO: u16 = 1;

/// A handshake with no frame from the peer for this long is dead, as a session
/// with no frame for this long is (§3.2).
pub const HANDSHAKE_IDLE: Duration = Duration::from_secs(45);

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HandshakeError {
    /// The peer closed the socket.
    #[error("closed by the peer — {0}")]
    Closed(Close),
    /// This end closed it, with the code it sent.
    #[error("refused — {0}")]
    Refused(Close),
    /// The transport went away mid-handshake.
    #[error("the socket dropped during the handshake")]
    Dropped,
}

impl From<PipeClosed> for HandshakeError {
    fn from(_: PipeClosed) -> Self {
        HandshakeError::Dropped
    }
}

async fn send<T: PipeTx>(tx: &mut T, frame: &Frame) -> Result<(), HandshakeError> {
    tx.send(Message::Frame(frame.encode())).await?;
    Ok(())
}

/// Close with `close`, and report it as this end's refusal.
async fn refuse<T: PipeTx>(tx: &mut T, close: Close) -> HandshakeError {
    // The peer may already be gone; the refusal stands either way.
    let _ = tx.send(Message::Close(close.clone())).await;
    HandshakeError::Refused(close)
}

/// The peer's next frame, or why there is none: a close, a dropped socket, an
/// undecodable frame (closed 4400) or 45 s of silence (closed 4400).
async fn next<T: PipeTx, R: PipeRx>(tx: &mut T, rx: &mut R) -> Result<Frame, HandshakeError> {
    let message = match timeout(HANDSHAKE_IDLE, rx.recv()).await {
        Ok(Some(m)) => m,
        Ok(None) => return Err(HandshakeError::Dropped),
        Err(_) => {
            return Err(refuse(
                tx,
                Close::new(CloseCode::Malformed, "no handshake frame for 45 s"),
            )
            .await)
        }
    };
    match message {
        Message::Close(close) => Err(HandshakeError::Closed(close)),
        Message::Frame(bytes) => match Frame::decode(&bytes) {
            Ok(frame) => Ok(frame),
            Err(e) => Err(refuse(tx, Close::new(CloseCode::Malformed, e.to_string())).await),
        },
    }
}

/// A frame the handshake does not expect here: closed 4400.
async fn unexpected<T: PipeTx>(tx: &mut T, got: &Frame, wanted: &str) -> HandshakeError {
    refuse(
        tx,
        Close::new(
            CloseCode::Malformed,
            format!(
                "expected {wanted}, got {} during the handshake",
                name(got.kind())
            ),
        ),
    )
    .await
}

#[cfg(test)]
mod tests;
