//! What zend tells the gateway's bus about its pushes
//! (`docs/mcp_file_tools.md` §5.3, §5.4).
//!
//! | Module | |
//! |---|---|
//! | [`member_bus`] | the bus, reached over this member's sockets to its gateways |
//! | [`offline`] | the bus of a daemon in no cluster: nothing heard, no gate asked |
//! | [`announced`] | the [`Push`](zend_vfs::Push) every repository's pushes go through: gated, then announced |
//!
//! Every push Zen makes to a workspace repository — a worker's `git_push`,
//! `file_edit_and_commit`, an agent job's — goes through a repository's
//! pusher ([`zend_tools::ToolContext::pusher`]), so this is the one place
//! each is seen. A repository the manifest names a `gate:` for is asked on
//! `gate/<repo>` before its branch moves; every branch a push moves is
//! announced on `repo/<repo>` once it has.

pub mod announced;
pub mod member_bus;
pub mod offline;

use std::time::Duration;

pub use host::cluster::bus::Unasked;
use serde_json::Value;

/// The bus, as zend uses it. Called from a tool's own thread, which may block.
pub trait Bus: Send + Sync {
    /// Send `payload` on `topic` to every gateway's bus.
    fn publish(&self, topic: &str, payload: Value);

    /// Publish `payload` on `topic` and wait at most `wait` for the first
    /// payload on `reply`, asking each gateway until one has a subscriber.
    fn ask(
        &self,
        topic: &str,
        payload: Value,
        reply: &str,
        wait: Duration,
    ) -> Result<Value, Unasked>;
}
