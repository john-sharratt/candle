//! The timings, caps and app names the hub is configured with.
//!
//! Each app a member may join is named here. The services keep runs on each
//! machine are listed apart from the replication groups: their members are
//! admitted on whatever build they run and are never certified or rolled out.
//! The caps bound a member's queues. Live events unsent past `LAG_CAP` make it
//! lagging, catch-up bytes past `CATCHUP_CAP` are dropped, and relayed frames
//! past `PENDING_CAP` fail its admission. A member that sends no `CAPACITY` for
//! `CAPACITY_SILENCE` is silent, since a deadlocked process keeps its socket
//! open while sending nothing.

use super::*;

/// How often every member is sent `MEMBERS` whether or not anything changed.
pub const MEMBERS_EVERY: Duration = Duration::from_secs(5);

/// How many refused handshakes the status page lists.
pub const REFUSALS_KEPT: usize = 100;

/// The apps a member may join (§3.1). `keep`, `mcp`, `zenj` and
/// `battlecities` are [`SERVICES`]: listed on the status page, and not
/// replication groups.
pub const APPS: [&str; 7] = ["zend", "npcd", AGENT_APP, KEEP, MCP, ZENJ, BATTLECITIES];

/// The agents' app (§11).
pub(super) const AGENT_APP: &str = "agent";

/// The failsafe's app. It joins so the status page can list it, and so an
/// Admin's operations tools reach it.
pub const KEEP: &str = "keep";

/// The MCP aggregator's app (`docs/mcp_app.md`): the `/mcp` route's members.
pub const MCP: &str = "mcp";

/// The DJ's app (`docs/zenj.md`): the `/v1/zenj` routes' member, on the
/// machine the decks are cabled to.
pub const ZENJ: &str = "zenj";

/// The Battle Cities game server's app (`docs/battlecities_app.md`): the
/// members the `battlecities` site's requests and the game's datagrams go
/// to. It is built from its own repository, so its build count is that
/// repository's and is compared with nothing here.
pub const BATTLECITIES: &str = "battlecities";

/// The apps keep runs on each machine and releases there itself: each member
/// is admitted on whatever build it runs, holds no substrate and no model,
/// and is never certified or rolled out. Their members serve as they join.
pub const SERVICES: [&str; 4] = [KEEP, MCP, ZENJ, BATTLECITIES];

/// Whether `app` is one of the [`SERVICES`].
pub fn is_service(app: &str) -> bool {
    SERVICES.contains(&app)
}

/// Whether `app` takes a candidate, a canary share, or a rollout. `agent`
/// follows the daemons' floor; the [`SERVICES`] do none of it.
pub fn rolls_out(app: &str) -> bool {
    app != AGENT_APP && !is_service(app)
}

/// Live events unsent on a member's ordered lane past which it is lagging.
pub const LAG_CAP: usize = 256 << 20;

/// Bytes on a member's catch-up lane past which a catch-up frame is dropped —
/// twice the requester's fetch window ([`FETCH_WINDOW`]), so a full window and
/// the headers beside it fit.
pub const CATCHUP_CAP: usize = 2 * FETCH_WINDOW as usize;

/// How often each socket's `RECEIPT` timer and lagging state are looked at.
pub(super) const RECEIPT_TICK: Duration = Duration::from_millis(100);

/// How long a member may send no `CAPACITY` before it is silent (§4.4): it
/// sends one every [`crate::member::CAPACITY_EVERY`], and pauses only while
/// the records it is applying fill their budget. A deadlocked process keeps
/// its socket open — the transport still answers — while sending none.
pub const CAPACITY_SILENCE: Duration = Duration::from_secs(15);

/// The most relayed frames a [`Pending`] keeps for its session. A handshake
/// during which more pile up fails its admission, and the member dials again
/// into a quieter moment. The state reports held back are not counted: they
/// are kept to what each kind last said ([`Held`]).
pub(in super::super) const PENDING_CAP: usize = 4096;

/// Past this, a [`Pending`] has outlived any handshake: `ADMIT_SEND` bounds
/// its send, and its [`PendingGuard`] undoes it however its task ends, so only
/// a task never polled again leaves one standing. That breaks an invariant,
/// and the gateway says so loudly at each `HELLO` it refuses for it.
pub(in super::super) const PENDING_OVERDUE: Duration =
    Duration::from_secs(ADMIT_SEND.as_secs() + 5);

pub(in super::super) fn hex(node: u64) -> String {
    format!("{node:016x}")
}
