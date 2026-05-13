//! UDP session tools: `udp_session_{open,send,recv,list,close}`.
//!
//! Bound UDP sockets for datagram-based protocols (DNS, SNMP, DHCP, RADIUS,
//! QUIC, custom UDP protocols, etc.).  Like TCP sessions, UDP uses hex for
//! non-text payloads — see the wire format notes in the `tcp_session` module.
//!
//! # Send / recv
//!
//! - **Send**: pass `data` (text) or `data_hex` (hex bytes).  A `peer` address
//!   can override the session's default peer per-send (useful for protocols that
//!   reply from a different port than they received on).
//! - **Recv**: `recv_wait_sec` controls how long to wait for a datagram.  Returns
//!   the data and the source address of the sender.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `bind_failed` | Could not bind a local socket |
//! | `session_not_found` | Session ID not in registry |
//! | `send_failed` | Sendto error |
//! | `recv_failed` | Recvfrom error or timeout |
//! | `invalid_params` | Malformed peer address |
//!
//! # Confirmation policy
//!
//! `udp_session_send` confirms every call.  Open, recv, list, and close do not.

use thiserror::Error;
use crate::ToolError;

pub mod open;
pub mod send;
pub mod recv;
pub mod list;
pub mod close;

pub use open::UDP_SESSION_OPEN;
pub use send::UDP_SESSION_SEND;
pub use recv::UDP_SESSION_RECV;
pub use list::UDP_SESSION_LIST;
pub use close::UDP_SESSION_CLOSE;

#[derive(Debug, Error)]
pub enum UdpError {
    #[error("bind failed: {0}")]
    BindFailed(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("send failed: {0}")]
    SendFailed(String),
    #[error("recv failed: {0}")]
    RecvFailed(String),
    #[error("invalid params: {0}")]
    InvalidParams(String),
}

impl ToolError for UdpError {
    fn code(&self) -> &'static str {
        match self {
            UdpError::BindFailed(_) => "bind_failed",
            UdpError::SessionNotFound(_) => "session_not_found",
            UdpError::SendFailed(_) => "send_failed",
            UdpError::RecvFailed(_) => "recv_failed",
            UdpError::InvalidParams(_) => "invalid_params",
        }
    }
}
