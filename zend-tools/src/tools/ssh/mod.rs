//! SSH session tools: `ssh_session_{open,exec,exec_async,poll,list,close}`.
//!
//! Each session holds an open SSH connection (via `ssh2`) with a persistent shell
//! channel.  Commands are dispatched using a sentinel-and-nonce protocol that
//! captures stdout, stderr, exit code, and post-command cwd in one round trip
//! without a PTY.
//!
//! # Two execution modes
//!
//! - **Synchronous** (`ssh_session_exec`): sends the command, waits for the
//!   sentinel, returns the full output.  Blocks the session for the duration.
//! - **Asynchronous** (`ssh_session_exec_async` + `ssh_session_poll`): returns
//!   immediately with a `process_id`; output accumulates in per-process ring
//!   buffers (1 MiB each).  Up to 4 concurrent async commands per session.
//!
//! # Host key verification
//!
//! Trust-on-first-use (TOFU): the first successful connection records the host
//! fingerprint; subsequent connections to the same host:port must match.  A
//! mismatch returns `host_key_mismatch` rather than silently connecting.
//!
//! # Output limits
//!
//! Stdout and stderr are each capped at [`MAX_OUTPUT`] (32 KiB).  Output beyond
//! the cap is dropped and the `stdout_truncated` / `stderr_truncated` flags are
//! set in the response.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `connection_failed` | TCP connect or SSH handshake error |
//! | `auth_failed` | Wrong key, wrong password, or no auth methods succeeded |
//! | `credential_not_found` | Named credential not in store |
//! | `session_not_found` | Session ID not in registry |
//! | `session_dead` | Connection was lost or timed out |
//! | `session_busy` | Concurrent synchronous exec attempted |
//! | `process_not_found` | `process_id` not in registry (poll) |
//! | `timeout` | Command exceeded `timeout_sec` |
//! | `session_limit_exceeded` | 5-session-per-user cap reached |
//! | `denied_by_user` | Confirmation prompt was rejected |
//! | `concurrency_cap_exceeded` | 4 async commands already running in this session |
//!
//! # Confirmation policy
//!
//! `ssh_session_open` confirms once (shows host + credential name).
//! `ssh_session_exec` and `ssh_session_exec_async` confirm every call (shows exact command).
//! List, poll (read-only), and close do not confirm.

use std::io::Read;

use thiserror::Error;

use crate::ToolError;

pub mod open;
pub mod exec;
pub mod exec_async;
pub mod poll;
pub mod list;
pub mod close;

pub use open::SSH_SESSION_OPEN;
pub use exec::SSH_SESSION_EXEC;
pub use exec_async::SSH_SESSION_EXEC_ASYNC;
pub use poll::SSH_SESSION_POLL;
pub use list::SSH_SESSION_LIST;
pub use close::SSH_SESSION_CLOSE;

pub const MAX_OUTPUT: usize = 32 * 1024; // 32 KiB

#[derive(Debug, Error)]
pub enum SshError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("authentication failed: {0}")]
    AuthFailed(String),
    #[error("credential not found: {0}")]
    CredentialNotFound(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("session is no longer alive")]
    SessionDead,
    #[error("session is busy")]
    SessionBusy,
    #[error("process not found: {0}")]
    ProcessNotFound(String),
    #[error("command timed out")]
    Timeout,
    #[error("session limit exceeded")]
    SessionLimitExceeded,
    #[error("denied by user")]
    DeniedByUser,
    #[error("concurrency cap exceeded")]
    ConcurrencyCapExceeded,
}

impl ToolError for SshError {
    fn code(&self) -> &'static str {
        match self {
            SshError::ConnectionFailed(_) => "connection_failed",
            SshError::AuthFailed(_) => "auth_failed",
            SshError::CredentialNotFound(_) => "credential_not_found",
            SshError::SessionNotFound(_) => "session_not_found",
            SshError::SessionDead => "session_dead",
            SshError::SessionBusy => "session_busy",
            SshError::ProcessNotFound(_) => "process_not_found",
            SshError::Timeout => "timeout",
            SshError::SessionLimitExceeded => "session_limit_exceeded",
            SshError::DeniedByUser => "denied_by_user",
            SshError::ConcurrencyCapExceeded => "concurrency_cap_exceeded",
        }
    }
}

pub fn exec_simple(session: &ssh2::Session, cmd: &str) -> Result<String, ssh2::Error> {
    let mut channel = session.channel_session()?;
    channel.exec(cmd)?;
    let mut out = String::new();
    channel.read_to_string(&mut out).ok();
    channel.wait_close().ok();
    Ok(out)
}
