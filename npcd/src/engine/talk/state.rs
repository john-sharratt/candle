//! What a conversation keeps between turns (`docs/npc_conversations.md`
//! §4.2), as its timeline's metadata: replicated with the substrate, so any
//! member holding the character's lease resumes it where it stopped.
//!
//! Every key is `talk.*`, and every value a plain string, so a record read on
//! another member, or after a restart, is the record written.

use std::collections::{BTreeMap, BTreeSet};

use journals::line::LineId;
use journals::time::Stamp;

use crate::engine::talk::key::Party;

pub const PARTY: &str = "talk.party";
pub const THREAD: &str = "talk.thread";
pub const LAST_TURN: &str = "talk.last_turn";
pub const SEEN: &str = "talk.seen";
pub const ANSWERED_SEQ: &str = "talk.answered_seq";
pub const ACTIVATION: &str = "talk.activation";
pub const PENDING_CRITICAL: &str = "talk.pending_critical";
pub const COMPASS: &str = "talk.compass";
pub const OPEN: &str = "talk.open";
pub const GENERATION: &str = "talk.generation";

/// One conversation's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TalkState {
    pub party: Party,
    /// The pair's stable thread id (§11.1).
    pub thread: String,
    /// When the last turn finished, in real time: the catch-up's bookmark.
    pub last_turn: Stamp,
    /// The journal lines read within the overlap window before `last_turn`
    /// (§6.1), so a late arrival is read once and only once.
    pub seen: BTreeSet<LineId>,
    /// The highest thread seq taken as input.
    pub answered_seq: u64,
    /// Turns in the current activation (§8).
    pub activation: u32,
    /// The critical step the NPC's last line asked an authorizer for (§7.2).
    pub pending_critical: Option<String>,
    /// The fingerprint of the mission compass last shown.
    pub compass: Option<u64>,
    /// Open, or ended by a goodbye.
    pub open: bool,
    pub generation: u32,
}

impl TalkState {
    /// A conversation not yet begun on `thread`, answering only what arrives
    /// after `answered_seq` — the mind's delivered cursor, so it does not
    /// answer the whole history (§9).
    pub fn begin(party: Party, thread: String, answered_seq: u64, now: Stamp) -> TalkState {
        TalkState {
            party,
            thread,
            last_turn: now,
            seen: BTreeSet::new(),
            answered_seq,
            activation: 0,
            pending_critical: None,
            compass: None,
            open: true,
            generation: 0,
        }
    }

    /// The state as its timeline's metadata.
    pub fn to_meta(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        let mut put = |k: &str, v: String| {
            m.insert(k.to_string(), v);
        };
        put(PARTY, party_text(&self.party));
        put(THREAD, self.thread.clone());
        put(LAST_TURN, self.last_turn.to_string());
        put(
            SEEN,
            self.seen
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        put(ANSWERED_SEQ, self.answered_seq.to_string());
        put(ACTIVATION, self.activation.to_string());
        put(PENDING_CRITICAL, self.pending_critical.clone().unwrap_or_default());
        put(
            COMPASS,
            self.compass.map(|c| format!("{c:016x}")).unwrap_or_default(),
        );
        put(OPEN, self.open.to_string());
        put(GENERATION, self.generation.to_string());
        m
    }

    /// The state read back from its timeline's metadata; `None` for a
    /// timeline that is not a conversation's, or whose record is damaged.
    pub fn from_meta(m: &BTreeMap<String, String>) -> Option<TalkState> {
        let get = |k: &str| m.get(k).map(String::as_str);
        let optional = |k: &str| get(k).filter(|v| !v.is_empty()).map(str::to_string);
        Some(TalkState {
            party: party_of(get(PARTY)?)?,
            thread: get(THREAD)?.to_string(),
            last_turn: Stamp::parse(get(LAST_TURN)?).ok()?,
            seen: match get(SEEN)? {
                "" => BTreeSet::new(),
                ids => ids
                    .split(',')
                    .map(LineId::parse)
                    .collect::<Result<_, _>>()
                    .ok()?,
            },
            answered_seq: get(ANSWERED_SEQ)?.parse().ok()?,
            activation: get(ACTIVATION)?.parse().ok()?,
            pending_critical: optional(PENDING_CRITICAL),
            compass: match optional(COMPASS) {
                Some(c) => Some(u64::from_str_radix(&c, 16).ok()?),
                None => None,
            },
            open: get(OPEN)?.parse().ok()?,
            generation: get(GENERATION)?.parse().ok()?,
        })
    }
}

/// A party as its record writes it: `person:{handle}` or `npc:{id}`.
fn party_text(p: &Party) -> String {
    match p {
        Party::Person(handle) => format!("person:{handle}"),
        Party::Npc(npc) => format!("npc:{npc}"),
    }
}

fn party_of(s: &str) -> Option<Party> {
    match s.split_once(':')? {
        ("person", handle) if !handle.is_empty() => Some(Party::Person(handle.to_string())),
        ("npc", id) => id.parse().ok().map(Party::Npc),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> Stamp {
        Stamp::parse("2026-11-10T09:15:00Z").unwrap()
    }

    #[test]
    fn a_new_conversation_s_record_is_exactly_these_bytes() {
        let s = TalkState::begin(
            Party::Person("u_8812abcd".into()),
            "dm:npc-7:u_8812abcd".into(),
            4,
            at(),
        );
        let m = s.to_meta();
        let pairs: Vec<(&str, &str)> = m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(
            pairs,
            [
                ("talk.activation", "0"),
                ("talk.answered_seq", "4"),
                ("talk.compass", ""),
                ("talk.generation", "0"),
                ("talk.last_turn", "2026-11-10T09:15:00Z"),
                ("talk.open", "true"),
                ("talk.party", "person:u_8812abcd"),
                ("talk.pending_critical", ""),
                ("talk.seen", ""),
                ("talk.thread", "dm:npc-7:u_8812abcd"),
            ]
        );
        assert_eq!(TalkState::from_meta(&m), Some(s));
    }

    #[test]
    fn a_conversation_in_progress_round_trips() {
        let id = |s: &str| LineId::parse(s).unwrap();
        let s = TalkState {
            party: Party::Npc(12),
            thread: "dm:npc-12:npc-7".into(),
            last_turn: at(),
            seen: [id("2026-11-10-0914-0a1b"), id("2026-11-10-0914-ff00")].into(),
            answered_seq: 31,
            activation: 3,
            pending_critical: Some("Sign the lease".into()),
            compass: Some(0x00ab_cdef_0123_4567),
            open: false,
            generation: 2,
        };
        let m = s.to_meta();
        assert_eq!(m[SEEN], "2026-11-10-0914-0a1b,2026-11-10-0914-ff00");
        assert_eq!(m[COMPASS], "00abcdef01234567");
        assert_eq!(m[PARTY], "npc:12");
        assert_eq!(TalkState::from_meta(&m), Some(s));
    }

    #[test]
    fn a_timeline_that_is_no_conversation_has_no_state() {
        assert_eq!(TalkState::from_meta(&BTreeMap::new()), None);
        let mut m = TalkState::begin(Party::Npc(1), "t".into(), 0, at()).to_meta();
        m.insert(PARTY.into(), "robot:1".into());
        assert_eq!(TalkState::from_meta(&m), None);
    }
}
