//! The gateway status page's data (§12.2): the Apps, Faults and Members
//! blocks, Claims, Refusals, Traffic and Failures, and the coordination log
//! (§12.3), as `GET /gateway/status.json` serves them. Each member row carries
//! the member's last `COORD` (§12.1), from which the page draws the
//! Generations, Consensus, Replication, Recovery and Tombstones blocks.

use serde::{Serialize, Serializer};

use super::log::Entry;
use super::registry::MemberView;
use super::traffic::TrafficView;
use super::upgrades::held::HeldBuild;
use super::upgrades::UpgradesView;
use super::worlds_view::{DisconnectedRow, WorldView};
use crate::coord::{CoordState, FaultView};
use crate::frame::{BuildId, MemberState};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HubStatus {
    /// The build this gateway runs: the one thing a script asking "is the
    /// gateway on the new web?" otherwise could not read anywhere.
    pub build: BuildId,
    pub gateway: String,
    pub now_ms: u64,
    /// Admitting no sockets, its members sent to their other clusters
    /// (§12.4 `drain-gateway`).
    pub draining: bool,
    pub apps: Vec<AppView>,
    pub faults: Vec<FaultRow>,
    pub members: Vec<MemberRow>,
    pub claims: Vec<ClaimView>,
    /// The World leases and NPC leases blocks (§9.2, §9.3).
    pub worlds: Vec<WorldView>,
    /// The Disconnected worlds block (§9.4).
    pub disconnected: Vec<DisconnectedRow>,
    pub refusals: Vec<Refusal>,
    pub traffic: TrafficView,
    /// The Upgrades block: per app, its candidate and certification, and its
    /// rollout (§3.8–§3.11).
    pub upgrades: Vec<UpgradesView>,
    /// Every build any member holds, newest first.
    pub builds: Vec<HeldBuild>,
    pub log: Vec<Entry>,
}

/// A member in fault mode, or within twice the floor of it (§7.7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FaultRow {
    pub app: String,
    pub node: String,
    pub host: String,
    /// `None` for a member still out of fault mode, near the floor.
    pub fault: Option<FaultView>,
    pub free: u64,
    pub floor: u64,
    /// Its own events not yet sent, or not yet receipted.
    pub unsent: usize,
}

impl FaultRow {
    /// The row for `row`, when it belongs in the block.
    pub fn of(row: &MemberRow) -> Option<FaultRow> {
        let c = row.coord.as_ref()?;
        let near = c.disk.total > 0 && c.disk.free < 2 * c.disk.floor;
        if c.fault.is_none() && !near {
            return None;
        }
        Some(FaultRow {
            app: row.app.clone(),
            node: row.node.clone(),
            host: row.host.clone(),
            fault: c.fault.clone(),
            free: c.disk.free,
            floor: c.disk.floor,
            unsent: c.outbox.unsent + c.outbox.in_flight,
        })
    }
}

/// What an app's mode reads as on the page (§5.7). The routing knows two —
/// clustered once a member is ready, dumb before — but "dumb" said the app had
/// no member, when one joined and still loading serves its loading page in place
/// of a down upstream. That is `loading`.
pub fn mode_name(clustered: bool, standing_by: bool) -> &'static str {
    match (clustered, standing_by) {
        (true, _) => "clustered",
        (false, true) => "loading",
        (false, false) => "dumb",
    }
}

/// One app on this gateway.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppView {
    pub app: String,
    /// `clustered`, `loading` or `dumb` ([`mode_name`]), and since when the
    /// routing's mode last changed.
    pub mode: &'static str,
    pub since_ms: u64,
    #[serde(serialize_with = "build_opt")]
    pub floor: Option<BuildId>,
    pub floor_set_by: Option<String>,
    /// A newer build's pending floor, and when its grace period ends — `None`
    /// while its candidate loads (§3.5).
    #[serde(serialize_with = "build_opt")]
    pub pending_floor: Option<BuildId>,
    pub grace_ends_ms: Option<u64>,
    pub model_lock: Option<String>,
    pub members: usize,
    pub ready: usize,
    /// `max_members` and its quorum, as the app's members report them
    /// (§5.9); the greatest any reports, while they converge.
    pub max_members: Option<u32>,
    pub quorum: Option<usize>,
}

/// One admitted member.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemberRow {
    pub app: String,
    pub node: String,
    pub host: String,
    #[serde(serialize_with = "build")]
    pub build: BuildId,
    pub local_build: bool,
    pub weight: u32,
    pub power: Option<u32>,
    pub idle: Option<bool>,
    pub slots_free: Option<u32>,
    /// Requests this gateway has handed it and it is still answering (§5.3).
    pub in_flight: u32,
    pub state: &'static str,
    pub session: u64,
    pub connected_ms: u64,
    /// Its clock against this gateway's, measured at admission from its
    /// `HELLO`: positive when it runs ahead.
    pub clock_offset_ms: i64,
    /// Its last `COORD` (§12.1).
    pub coord: Option<CoordState>,
    pub last_close: Option<String>,
    /// Its own highest `oseq`, from its `ACK`s.
    pub high_water: u64,
    pub lagging: bool,
    /// Admitted beside a serving member and not yet in sync with it: routed
    /// no request.
    pub syncing: bool,
    #[serde(flatten)]
    pub queues: Queues,
}

/// A member's queues on this gateway (§4.4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Queues {
    /// Live events queued and not yet written to it, in bytes.
    pub live_bytes: usize,
    /// Its catch-up lane, in bytes.
    pub catchup_bytes: usize,
    /// Live events not queued for it while it lagged.
    pub skipped: u64,
    /// Catch-up frames dropped at its cap.
    pub dropped: u64,
    /// `PULSE`s dropped under pressure (§9.1).
    pub pulses_dropped: u64,
}

/// What the gateway knows of a member's socket beyond the registry.
#[derive(Debug, Clone, Default)]
pub struct SocketReport {
    pub connected_ms: u64,
    pub clock_offset_ms: i64,
    pub coord: Option<CoordState>,
    pub queues: Queues,
}

impl MemberRow {
    pub fn of(m: &MemberView, socket: SocketReport, last_close: Option<String>) -> Self {
        let c = m.capacity.as_ref();
        Self {
            app: m.app.clone(),
            node: format!("{:016x}", m.node),
            host: m.host.clone(),
            build: m.build.clone(),
            local_build: m.build.is_local(),
            weight: m.weight,
            power: c.map(|c| c.power),
            idle: c.map(|c| c.idle),
            slots_free: c.map(|c| c.slots_free),
            in_flight: m.in_flight,
            state: state_name(m.state()),
            session: m.session,
            connected_ms: socket.connected_ms,
            clock_offset_ms: socket.clock_offset_ms,
            coord: socket.coord,
            last_close,
            high_water: m.high_water,
            lagging: m.lagging,
            syncing: m.syncing,
            queues: socket.queues,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClaimView {
    pub app: String,
    pub key: String,
    pub node: String,
    pub epoch_ms: u64,
}

/// A refused handshake (§12.2 Refusals).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Refusal {
    pub at_ms: u64,
    pub app: String,
    pub ip: String,
    /// Empty when the socket sent no `HELLO`.
    pub node: String,
    pub host: String,
    pub code: u16,
    pub reason: String,
}

pub fn state_name(s: MemberState) -> &'static str {
    match s {
        MemberState::Loading => "loading",
        MemberState::Ready => "ready",
        MemberState::Upgrading => "upgrading",
        MemberState::Fault => "fault",
        MemberState::Draining => "draining",
        MemberState::Disabled => "disabled",
    }
}

#[derive(Serialize)]
struct BuildJson<'a> {
    count: u32,
    hash: &'a str,
    local: &'a str,
}

fn build<S: Serializer>(b: &BuildId, s: S) -> Result<S::Ok, S::Error> {
    BuildJson {
        count: b.count,
        hash: &b.hash,
        local: &b.local,
    }
    .serialize(s)
}

fn build_opt<S: Serializer>(b: &Option<BuildId>, s: S) -> Result<S::Ok, S::Error> {
    b.as_ref()
        .map(|b| BuildJson {
            count: b.count,
            hash: &b.hash,
            local: &b.local,
        })
        .serialize(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::{DiskView, OutboxView};

    const GIB: u64 = 1 << 30;

    /// A ready member makes the app clustered whatever else is joined; a
    /// joined member still loading reads as `loading`, not as an app with no
    /// member; only an app with nobody joined is `dumb`.
    #[test]
    fn an_app_with_a_loading_member_reads_as_loading() {
        assert_eq!(mode_name(true, false), "clustered");
        assert_eq!(mode_name(true, true), "clustered");
        assert_eq!(mode_name(false, true), "loading");
        assert_eq!(mode_name(false, false), "dumb");
    }

    fn row(coord: Option<CoordState>) -> MemberRow {
        MemberRow {
            app: "zend".into(),
            node: "0000000000000002".into(),
            host: "m2".into(),
            build: BuildId {
                count: 1,
                hash: "h1".into(),
                local: String::new(),
            },
            local_build: false,
            weight: 1,
            power: None,
            idle: None,
            slots_free: None,
            in_flight: 0,
            state: "ready",
            session: 1,
            connected_ms: 0,
            clock_offset_ms: 0,
            coord,
            last_close: None,
            high_water: 0,
            lagging: false,
            syncing: false,
            queues: Queues::default(),
        }
    }

    fn disk(free: u64) -> CoordState {
        CoordState {
            disk: DiskView {
                free,
                total: 100 * GIB,
                floor: 2 * GIB,
            },
            outbox: OutboxView {
                unsent: 3,
                in_flight: 4,
            },
            ..CoordState::default()
        }
    }

    /// A member in fault mode is listed, and so is one within twice the floor
    /// of it, as a warning; one with room, or that has not reported, is not.
    #[test]
    fn the_faults_block_lists_faults_and_members_near_the_floor() {
        assert_eq!(FaultRow::of(&row(None)), None);
        assert_eq!(FaultRow::of(&row(Some(disk(10 * GIB)))), None);
        let near = FaultRow::of(&row(Some(disk(3 * GIB)))).unwrap();
        assert_eq!(
            (near.fault, near.free, near.floor, near.unsent),
            (None, 3 * GIB, 2 * GIB, 7)
        );
        let mut faulted = disk(10 * GIB);
        faulted.fault = Some(FaultView {
            reason: "a write failed for lack of space".into(),
            ..FaultView::default()
        });
        assert!(FaultRow::of(&row(Some(faulted))).unwrap().fault.is_some());
    }
}
