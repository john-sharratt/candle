//! How a bot's gateway connection stands, as its shard keeps it up to date
//! for the trace (`docs/npcd_discord.md` §5.5).

use std::sync::{Arc, Mutex};

use serde::Serialize;

/// A shard's connection, now.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Link {
    /// Whether the connection has reached `READY` or `RESUMED` and is open.
    pub connected: bool,
    /// The session Discord opened, while there is one to resume.
    pub session: Option<String>,
    /// When the connection now open reached `READY` or `RESUMED`.
    pub since_ms: Option<u64>,
    /// When the gateway last dispatched anything.
    pub last_event_ms: Option<u64>,
    /// When the gateway last acknowledged a heartbeat.
    pub last_heartbeat_ack_ms: Option<u64>,
}

/// A link a shard writes and the trace reads.
pub type SharedLink = Arc<Mutex<Link>>;

/// Change `link` with `f`.
pub fn update(link: &SharedLink, f: impl FnOnce(&mut Link)) {
    f(&mut link.lock().expect("link lock"));
}

/// What `link` says now.
pub fn read(link: &SharedLink) -> Link {
    link.lock().expect("link lock").clone()
}
