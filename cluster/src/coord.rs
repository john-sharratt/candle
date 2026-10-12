//! What a member reports of its coordination state (§12.1): the `COORD` frame's
//! body, sent every 2 s and within a second of any change, on every socket.
//! A gateway sees its own queues and the frames it carries; this is what only
//! the member can see — its mode, generation, acceptor state, catch-ups, disk.
//!
//! The body is this structure's JSON, fields in declaration order, so one state
//! has one encoding.

use serde::{Deserialize, Serialize};

use crate::replicate::{CatchupView, Phase, Stats};
use crate::upgrade::decisions::{Rollout, Settled, Verdict};
use crate::upgrade::watchdog::Failure;
use crate::version::info::Opens;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordState {
    /// `joined`, `cut_off`, `rejoining`, `reloading`, `draining` or `fault`.
    pub mode: String,
    /// Its builds, as upgrades see them (§3.8–§3.11).
    pub upgrade: Option<UpgradeView>,
    /// It holds records it wrote cut off, not yet placed by a generation.
    pub detached: bool,
    pub fault: Option<FaultView>,
    pub gen: GenView,
    /// Records held back for a type this build does not decode (§6.5).
    pub deferred: Vec<DeferredView>,
    /// Every `--cluster` it was given, and its socket there.
    pub clusters: Vec<ClusterLine>,
    pub paxos: PaxosView,
    /// Each world of the app, as this member stands to it (§9.2–§9.4).
    pub worlds: Vec<WorldLine>,
    /// The NPC leases it holds (§9.3).
    pub npc_leases: Vec<NpcLeaseLine>,
    pub claims: Vec<ClaimLine>,
    /// Per origin: how far it has applied, its gaps, and the origin's
    /// announced `high_water` (§7.1).
    pub applied: Vec<AppliedLine>,
    pub outbox: OutboxView,
    pub catchups: Vec<CatchupView>,
    pub tombstones: TombstoneView,
    pub disk: DiskView,
    pub stats: Stats,
    /// The maintenance commands run on it lately, newest last (§12.4).
    pub commands: Vec<CommandLine>,
}

/// What a member reports of upgrades (§3.9, §12.1): what a build must open to
/// replace its own, what it holds and fetches, and where it stands in a
/// rollout.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeView {
    /// Its binary: `zend`, `npcd`, `tokerad`, `tokera-agent`.
    pub kind: String,
    pub platform: String,
    /// What a build must open to replace this one: the log format, every
    /// record type its substrate holds, the model it runs.
    pub log_format: Option<u32>,
    pub census: Vec<u8>,
    pub model: Option<String>,
    /// Every artifact it can serve.
    pub holds: Vec<HoldLine>,
    /// The builds it keeps in `builds/`.
    pub kept: Vec<KeptLine>,
    pub fetching: Option<FetchLine>,
    /// As a candidate: each cluster's verdict.
    pub candidate: Option<CandidateLine>,
    /// Its place in a rollout: empty when idle, `waiting` for its lease,
    /// `installing` once it holds it, `upgrading` when admitted below the
    /// floor, `stranded` or `failed`.
    pub step: String,
    /// Why no build can be installed here.
    pub stranded: Option<String>,
    /// The last install that failed and was restored.
    pub failure: Option<Failure>,
    /// The builds it never tries again: `count-hash`.
    pub refused: Vec<String>,
    /// The latest verdict on each build the app has settled.
    pub verdicts: Vec<VerdictLine>,
    /// The app's latest forced rollout.
    pub rollout: Option<RolloutLine>,
    /// Whether the app's automatic rollout is paused.
    pub paused: bool,
}

/// One artifact a member can serve.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HoldLine {
    /// `count-hash`.
    pub build: String,
    pub kind: String,
    pub platform: String,
    pub size: u64,
    pub sha256: String,
    /// `running`, `kept`, `verified` or `offered`.
    pub source: String,
    /// What the build opens, for a binary whose answer is known.
    pub opens: Option<Opens>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeptLine {
    pub build: String,
    pub bytes: u64,
    pub retired_ms: u64,
}

/// An artifact being fetched.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchLine {
    pub build: String,
    pub kind: String,
    pub bytes: u64,
    pub size: u64,
    /// How often it picked up again from its part file.
    pub resumes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateLine {
    pub build: String,
    /// Per cluster: `pass`, `fail`, `no_opinion`, or empty until it sends one.
    pub verdicts: Vec<String>,
    /// The verdict it proposed, once it has.
    pub proposed: Option<Settled>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerdictLine {
    pub build: String,
    /// The verdict's number under `certify/<app>/<build>/`.
    pub n: u64,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutLine {
    pub n: u64,
    pub decision: Rollout,
}

/// One maintenance command run on the member: who ran it, with what, and how
/// it ended.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandLine {
    /// Its job on the member.
    pub job: u64,
    pub command: String,
    /// Its arguments, as JSON.
    pub args: String,
    /// Who ran it: the caller's email, or `local` on the member's own console.
    pub by: String,
    pub started_ms: u64,
    /// `ok`, or why it failed; `None` while it runs.
    pub outcome: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultView {
    /// Why: the disk is full, or a write failed for lack of space.
    pub reason: String,
    pub since_ms: u64,
    /// The claims it handed back.
    pub claims_released: usize,
    /// It still votes, through its vote journal (§5.9).
    pub voting: bool,
    /// Bytes freed since it entered.
    pub freed: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenView {
    /// The generation it writes under.
    pub current: u64,
    /// The generation its last rejoin took, if it has rejoined.
    pub last_rejoin: Option<u64>,
    /// The greatest `gen/<k>` decision it knows.
    pub decided: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferredView {
    pub record_type: String,
    pub origin: String,
    pub count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterLine {
    pub url: String,
    pub state: String,
    pub session: Option<u64>,
    pub last_close: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaxosView {
    pub max_members: u32,
    pub quorum: usize,
    pub keys: Vec<KeyLine>,
}

/// One Paxos key as this member knows it, ballots as `round.node`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyLine {
    pub key: String,
    /// A round in progress, `ballot phase`.
    pub round: Option<String>,
    /// The members that promised or accepted it, against the quorum.
    pub voters: Vec<String>,
    /// The higher ballot a `NACK` beat it with.
    pub beaten_by: Option<String>,
    /// A lease it holds: milliseconds left on its own clock.
    pub held_ms: Option<u64>,
    /// Its acceptor's promised and accepted ballots.
    pub promised: Option<String>,
    pub accepted: Option<String>,
}

/// One world, as a member stands to it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldLine {
    pub world: String,
    /// `authority`, `replica` or `disconnected`.
    pub role: String,
    /// The lease epoch it writes the world's state under, as its authority.
    pub epoch: Option<u64>,
    /// When it took this role, in wall-clock milliseconds.
    pub since_ms: u64,
    /// A handover it waits on before it runs the world: the records of the
    /// member that handed it back, `<origin> through <oseq>` (§9.6).
    pub waiting: Option<String>,
    /// As the world's authority: every character's lease it lends.
    pub lent: Vec<NpcLeaseLine>,
}

/// One character's NPC lease.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpcLeaseLine {
    pub world: String,
    pub npc: u64,
    /// The member that runs it; empty when an admin freed it.
    pub holder: String,
    pub expires_ms: u64,
    /// The member rendezvous names now, waiting for the holder to hand it
    /// back.
    pub next: Option<String>,
}

/// What the app adds to its member's report: its worlds and the NPC leases it
/// holds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppCoord {
    pub worlds: Vec<WorldLine>,
    pub npc_leases: Vec<NpcLeaseLine>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimLine {
    pub key: String,
    pub epoch_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedLine {
    pub origin: String,
    pub through: u64,
    /// `oseq`s missing below the highest it holds.
    pub missing: u64,
    pub high_water: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutboxView {
    pub unsent: usize,
    pub in_flight: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TombstoneView {
    /// Timeline tombstones held as records.
    pub held: usize,
    /// Of those, the ones waiting only for their age.
    pub waiting_age: usize,
    /// Expired tombstones kept as deleted-set entries.
    pub deleted: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskView {
    pub free: u64,
    pub total: u64,
    /// The fault floor: 2 GiB, or 1% of the volume if that is larger.
    pub floor: u64,
}

impl CoordState {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a coordination state always serializes")
    }

    pub fn decode(bytes: &[u8]) -> Option<CoordState> {
        serde_json::from_slice(bytes).ok()
    }

    /// The mode a member's phase and conditions add up to.
    pub fn mode_of(phase: Phase, drained: bool, fault: bool) -> &'static str {
        if fault {
            return "fault";
        }
        if drained {
            return "draining";
        }
        match phase {
            Phase::Joined => "joined",
            Phase::CutOff => "cut_off",
            Phase::Rejoining => "rejoining",
            Phase::Reloading => "reloading",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One state has one encoding: the JSON of the fields in declaration
    /// order, and it decodes back.
    #[test]
    fn a_state_encodes_to_exact_bytes() {
        let state = CoordState {
            mode: "joined".into(),
            gen: GenView {
                current: 2,
                last_rejoin: None,
                decided: 2,
            },
            paxos: PaxosView {
                max_members: 3,
                quorum: 2,
                keys: vec![],
            },
            outbox: OutboxView {
                unsent: 1,
                in_flight: 4,
            },
            ..CoordState::default()
        };
        let want = concat!(
            r#"{"mode":"joined","upgrade":null,"detached":false,"fault":null,"#,
            r#""gen":{"current":2,"last_rejoin":null,"decided":2},"deferred":[],"clusters":[],"#,
            r#""paxos":{"max_members":3,"quorum":2,"keys":[]},"worlds":[],"npc_leases":[],"#,
            r#""claims":[],"applied":[],"#,
            r#""outbox":{"unsent":1,"in_flight":4},"catchups":[],"#,
            r#""tombstones":{"held":0,"waiting_age":0,"deleted":0},"#,
            r#""disk":{"free":0,"total":0,"floor":0},"#,
            r#""stats":{"applied":0,"refused":0,"reloads":0,"catchups":0,"answered":0,"#,
            r#""fetched_for_others":0,"resent":0},"commands":[]}"#
        );
        assert_eq!(String::from_utf8(state.encode()).unwrap(), want);
        assert_eq!(CoordState::decode(want.as_bytes()), Some(state));
        assert_eq!(CoordState::decode(b"{"), None);
    }

    #[test]
    fn a_fault_outranks_a_drain_and_a_drain_the_phase() {
        assert_eq!(CoordState::mode_of(Phase::Joined, true, true), "fault");
        assert_eq!(CoordState::mode_of(Phase::Joined, true, false), "draining");
        assert_eq!(CoordState::mode_of(Phase::CutOff, false, false), "cut_off");
    }
}
