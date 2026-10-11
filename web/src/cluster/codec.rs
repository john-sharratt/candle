//! axum's WebSocket messages as a cluster pipe's: what a member's socket
//! speaks at the gateway.

use axum::extract::ws::{CloseFrame, Message};
use cluster::close::Close;
use cluster::pipe::socket::{Classified, Codec};

/// The code a close with no frame reads as (RFC 6455 §7.1.5).
const NO_STATUS: u16 = 1005;

pub struct AxumWs;

impl Codec for AxumWs {
    type Message = Message;

    fn binary(bytes: Vec<u8>) -> Message {
        Message::Binary(bytes)
    }

    fn close(close: &Close) -> Message {
        Message::Close(Some(CloseFrame {
            code: close.code,
            reason: close.reason.clone().into(),
        }))
    }

    fn classify(message: Message) -> Classified {
        match message {
            Message::Binary(bytes) => Classified::Frame(bytes),
            // Never a frame: handed on as one so the decoder refuses it.
            Message::Text(text) => Classified::Frame(text.into_bytes()),
            Message::Close(Some(frame)) => Classified::Close(Close {
                code: frame.code,
                reason: frame.reason.into_owned(),
            }),
            Message::Close(None) => Classified::Close(Close {
                code: NO_STATUS,
                reason: String::new(),
            }),
            Message::Ping(_) | Message::Pong(_) => Classified::Skip,
        }
    }
}
