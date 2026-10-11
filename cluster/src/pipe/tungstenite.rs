//! `tokio-tungstenite`'s messages as a pipe's: what a member's dialled socket
//! speaks.

use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;

use super::socket::{Classified, Codec};
use crate::close::Close;

/// The code a close with no frame reads as (RFC 6455 §7.1.5).
const NO_STATUS: u16 = 1005;

pub struct Tungstenite;

impl Codec for Tungstenite {
    type Message = Message;

    fn binary(bytes: Vec<u8>) -> Message {
        Message::Binary(bytes)
    }

    fn close(close: &Close) -> Message {
        Message::Close(Some(CloseFrame {
            code: CloseCode::from(close.code),
            reason: close.reason.clone().into(),
        }))
    }

    fn classify(message: Message) -> Classified {
        match message {
            Message::Binary(bytes) => Classified::Frame(bytes),
            // Never a frame: handed on as one so the decoder refuses it.
            Message::Text(text) => Classified::Frame(text.into_bytes()),
            Message::Close(Some(frame)) => Classified::Close(Close {
                code: u16::from(frame.code),
                reason: frame.reason.into_owned(),
            }),
            Message::Close(None) => Classified::Close(Close {
                code: NO_STATUS,
                reason: String::new(),
            }),
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => Classified::Skip,
        }
    }
}
