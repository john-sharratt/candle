//! The gateway status page's data (§12.2): the Apps and Members blocks,
//! Claims, Refusals, Traffic and Failures, and the coordination log (§12.3),
//! as `GET /gateway/status.json` serves them.

use serde::{Serialize, Serializer};

use super::log::Entry;
use super::registry::MemberView;
use super::traffic::TrafficView;
use crate::frame::{BuildId, MemberState};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HubStatus {
    pub gateway: String,
    pub now_ms: u64,
    pub apps: Vec<AppView>,
    pub members: Vec<MemberRow>,
    pub claims: Vec<ClaimView>,
    pub refusals: Vec<Refusal>,
    pub traffic: TrafficView,
    pub log: Vec<Entry>,
}

/// One app on this gateway.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppView {
    pub app: String,
    /// `dumb` or `clustered` (§5.7), and since when.
    pub mode: &'static str,
    pub since_ms: u64,
    #[serde(serialize_with = "build_opt")]
    pub floor: Option<BuildId>,
    pub floor_set_by: Option<String>,
    pub model_lock: Option<String>,
    pub members: usize,
    pub ready: usize,
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
    pub state: &'static str,
    pub session: u64,
    pub connected_ms: u64,
    /// The size of its last `COORD`, which this build does not decode.
    pub coord_bytes: usize,
    pub last_close: Option<String>,
}

impl MemberRow {
    pub fn of(
        m: &MemberView,
        connected_ms: u64,
        coord_bytes: usize,
        last_close: Option<String>,
    ) -> Self {
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
            state: state_name(m.state()),
            session: m.session,
            connected_ms,
            coord_bytes,
            last_close,
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
