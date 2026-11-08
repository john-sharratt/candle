//! Each conversation's own journal (§2.4, §6.6): named for the minute the
//! conversation began and the channel it began on, under its group's
//! `conversations`, and opened at the conversation's first write. It holds
//! the conversation's focus, its running tasks and the episodes of what it
//! did. A conversation that only reads writes nothing: its frame names the
//! journal it would open, so `here` means the same before and after.
//!
//! A conversation is a session's calls, for as long as the session lasts —
//! the aggregator's own boundary (`docs/mcp_user_context.md` §2.5). Every
//! memory call of a session moves its clock on. The journal a session opened
//! is kept in a replicated record, `mcp/conversation/<group>/<session digest>`
//! (`docs/zen_memory_and_guides.md` Phase 2), so a call carried to another
//! member, or here after a restart, goes on writing it; a session no record
//! names begins a conversation of its own. This process keeps what it has
//! seen in memory too, so a call it has seen reads no record.
//!
//! Grok opens a session for every call it makes, so its calls are followed
//! by the person and group instead ([`followed_by`]): a drive is one
//! conversation until it goes quiet.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use journals::actor::Actor;
use journals::ops::wire::path_from_wire;
use journals::ops::{OnTaken, Op};
use journals::path::JournalPath;
use journals::scope::{conversation_name, Channel, Section};
use journals::slug::Slug;
use journals::time::Stamp;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{held, Memory};
use crate::mcp::records::Records;

/// What every session's record of its journal is kept under.
const RECORD_PREFIX: &str = "mcp/conversation/";

/// The record naming the journal `session` opened in `group`: the session
/// kept by a digest, so no session id is written into a key.
fn record_key(group: &Slug, session: &str) -> String {
    let digest = Sha256::digest(session.as_bytes());
    let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
    format!("{RECORD_PREFIX}{group}/{hex}")
}

/// The journal `records` say `session` opened in `group`, when they say one.
fn recorded(records: &dyn Records, group: &Slug, session: &str) -> Option<JournalPath> {
    let folder = records.get(&record_key(group, session))?;
    path_from_wire(folder.as_str()?).ok()
}

/// The sessions remembered; past this, the idlest are let go.
const KEPT: usize = 4_096;

/// A session's conversation, as this process knows it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kept {
    /// Begun at this moment and written to by nothing yet, under the name
    /// reserved for its journal: one no journal and no other begun
    /// conversation holds.
    Begun(Stamp, Slug),
    /// Its journal, opened by its first write.
    Open(JournalPath),
}

/// Each session's conversation, and when it last had a call.
#[derive(Default)]
struct Sessions(HashMap<String, (Kept, Stamp)>);

impl Sessions {
    fn get(&self, session: &str) -> Option<&Kept> {
        self.0.get(session).map(|(k, _)| k)
    }

    fn keep(&mut self, session: &str, kept: Kept, at: Stamp) {
        if self.0.len() >= KEPT && !self.0.contains_key(session) {
            if let Some(idlest) = self
                .0
                .iter()
                .min_by_key(|(_, (_, at))| *at)
                .map(|(s, _)| s.clone())
            {
                self.0.remove(&idlest);
            }
        }
        self.0.insert(session.to_string(), (kept, at));
    }

    /// The names begun conversations hold for the journals they will open.
    fn reserved(&self) -> impl Iterator<Item = &Slug> {
        self.0.values().filter_map(|(k, _)| match k {
            Kept::Begun(_, name) => Some(name),
            Kept::Open(_) => None,
        })
    }
}

/// The conversations this process has seen.
#[derive(Default)]
pub struct Conversations {
    /// Held only for a look or a keep: a read never waits on a write.
    sessions: Mutex<Sessions>,
    /// Held from the look to the open, so two calls of one session arriving
    /// together open one journal between them.
    opening: Mutex<()>,
}

/// `group`'s `conversations` section.
pub fn section(group: &Slug) -> JournalPath {
    JournalPath::of_section(group, Section::Conversations)
}

/// The name a conversation begun at `at` on `channel` takes:
/// `2026-10-31-0712-grok`.
pub fn name_at(at: Stamp, channel: Channel) -> Slug {
    Slug::parse(&conversation_name(&at.dated_minute(), channel)).expect("a dated name is a name")
}

/// What a conversation of `person`'s in `group` through `channel` is
/// followed by: its MCP `session` — or, for Grok, which opens a session for
/// every call (sixteen conversations in six minutes, 2026-10-31), the person
/// and group, so its calls run on as one conversation.
pub fn followed_by(channel: Channel, person: &Slug, group: &Slug, session: &str) -> String {
    match channel {
        Channel::Grok => format!("{person}@{group}:{}", channel.name()),
        Channel::ClaudeCode | Channel::ClaudeAi | Channel::Zen | Channel::Other => {
            session.to_string()
        }
    }
}

/// The name a conversation begun at `at` on `channel` reserves in a section
/// whose journals and begun conversations hold `taken`: its minute's name,
/// numbered past any already held — as its journal's create would number it.
pub fn reserve(at: Stamp, channel: Channel, taken: &BTreeSet<Slug>) -> Slug {
    name_at(at, channel).numbered(taken)
}

/// Where `session`'s conversation in `group` stands: its journal — as this
/// process kept it, or as `records` name it — or, when nothing has written to
/// it, the moment it began and the name it reserved, begun now when this is
/// its first call. Writes nothing; the session's clock moves on.
fn kept(
    memory: &Memory,
    records: Option<&dyn Records>,
    group: &Slug,
    channel: Channel,
    session: &str,
    now: Stamp,
) -> Result<Kept, String> {
    let tree = memory.tree()?;
    let mut sessions = held(&memory.conversations.sessions);
    let found = |p: &JournalPath| p.group() == group && tree.get(p).is_some();
    let kept = match sessions.get(session) {
        Some(Kept::Open(p)) if found(p) => Kept::Open(p.clone()),
        Some(Kept::Begun(at, name)) => Kept::Begun(*at, name.clone()),
        _ => match records
            .and_then(|r| recorded(r, group, session))
            .filter(found)
        {
            Some(p) => Kept::Open(p),
            None => {
                let mut taken: BTreeSet<Slug> = tree
                    .get(&section(group))
                    .map(|j| j.children.keys().cloned().collect())
                    .unwrap_or_default();
                taken.extend(sessions.reserved().cloned());
                Kept::Begun(now, reserve(now, channel, &taken))
            }
        },
    };
    sessions.keep(session, kept.clone(), now);
    Ok(kept)
}

/// The journal `session`'s conversation in `group` is written to, as a read
/// sees it: opened, or the one its first write opens. Writes nothing.
pub fn here(
    memory: &Memory,
    records: Option<&dyn Records>,
    group: &Slug,
    channel: Channel,
    session: &str,
) -> Result<JournalPath, String> {
    match kept(memory, records, group, channel, session, Stamp::now())? {
        Kept::Open(p) => Ok(p),
        Kept::Begun(_, name) => section(group).child(name).map_err(|e| e.to_string()),
    }
}

/// The journal of `session`'s conversation in `group`, opened now when this
/// is the conversation's first write, under the name its reads were given,
/// and named in `records` for every member.
pub fn open(
    memory: &Memory,
    records: Option<&dyn Records>,
    person: &Slug,
    group: &Slug,
    channel: Channel,
    session: &str,
) -> Result<JournalPath, String> {
    let _opening = held(&memory.conversations.opening);
    let now = Stamp::now();
    let (began, name) = match kept(memory, records, group, channel, session, now)? {
        Kept::Open(p) => return Ok(p),
        Kept::Begun(at, name) => (at, name),
    };
    let server = Actor::Server {
        person: person.clone(),
        channel,
    };
    let when = began.to_string()[..16].replace('T', " ");
    // The reserved name, numbered only if another member opened it first.
    let op = Op::Create {
        under: section(group),
        name,
        cue: format!(
            "open for the conversation through {} begun {when}",
            channel.spoken()
        ),
        on_taken: OnTaken::Number,
    };
    let path = memory
        .write_at(&server, &op, began)?
        .created
        .expect("create makes a journal");
    held(&memory.conversations.sessions).keep(session, Kept::Open(path.clone()), now);
    if let Some(r) = records {
        if let Err(e) = r.put(&record_key(group, session), Value::String(path.folder())) {
            tracing::warn!(%path, "journals: this conversation's journal was not recorded for the other members: {e}");
        }
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    fn zen() -> Slug {
        Slug::parse("zen").unwrap()
    }

    #[test]
    fn a_conversation_is_named_for_its_minute_and_channel_under_its_group() {
        let at = Stamp::parse("2026-10-31T07:12:42Z").unwrap();
        assert_eq!(name_at(at, Channel::Grok).as_str(), "2026-10-31-0712-grok");
        assert_eq!(
            name_at(at, Channel::ClaudeCode).as_str(),
            "2026-10-31-0712-claude-code"
        );
        assert_eq!(section(&zen()).folder(), "groups/zen/conversations");
    }

    /// **Grok's calls are one conversation whatever their sessions**; every
    /// other channel's conversation is its session.
    #[test]
    fn a_grok_conversation_is_followed_by_its_person_not_its_session() {
        let john = Slug::parse("john").unwrap();
        let grok = |s| followed_by(Channel::Grok, &john, &zen(), s);
        assert_eq!(grok("s1"), "john@zen:grok");
        assert_eq!(grok("s1"), grok("s2"));
        let other = Slug::parse("battle-cities").unwrap();
        assert_ne!(followed_by(Channel::Grok, &john, &other, "s1"), grok("s1"));
        for c in [Channel::ClaudeCode, Channel::ClaudeAi, Channel::Other] {
            assert_eq!(followed_by(c, &john, &zen(), "s1"), "s1");
            assert_ne!(
                followed_by(c, &john, &zen(), "s1"),
                followed_by(c, &john, &zen(), "s2")
            );
        }
    }

    /// **A session is one conversation however long it pauses**, and another
    /// session is another.
    #[test]
    fn a_session_is_one_conversation_however_long_it_pauses() {
        let mut s = Sessions::default();
        let p = Kept::Open(section(&zen()));
        let t = Stamp::parse("2026-10-30T12:00:00Z").unwrap();
        s.keep("s", p.clone(), t);
        assert_eq!(s.get("s"), Some(&p));
        assert_eq!(s.get("other"), None);
    }

    /// **A conversation that has only read reserves the name its journal
    /// opens under**: its first call's minute, numbered past a journal or
    /// another begun conversation holding it, so no read is told a name that
    /// is another's.
    #[test]
    fn a_begun_conversation_reserves_a_name_no_one_holds() {
        let began = Stamp::parse("2026-10-31T07:12:42Z").unwrap();
        let free = BTreeSet::new();
        assert_eq!(
            reserve(began, Channel::Grok, &free).as_str(),
            "2026-10-31-0712-grok"
        );
        let held = BTreeSet::from([Slug::parse("2026-10-31-0712-grok").unwrap()]);
        assert_eq!(
            reserve(began, Channel::Grok, &held).as_str(),
            "2026-10-31-0712-grok.2"
        );
        let mut s = Sessions::default();
        s.keep(
            "a",
            Kept::Begun(began, Slug::parse("2026-10-31-0712-grok").unwrap()),
            began,
        );
        s.keep("b", Kept::Open(section(&zen())), began);
        let reserved: Vec<&str> = s.reserved().map(Slug::as_str).collect();
        assert_eq!(reserved, ["2026-10-31-0712-grok"]);
    }

    /// **A session's journal is found by its record where this process never
    /// saw it**, keyed by a digest that writes no session id; a record naming
    /// another group's journal, or none, is no find.
    #[test]
    fn a_sessions_journal_is_found_by_its_record() {
        let records = MemRecords::default();
        let key = record_key(&zen(), "session-1");
        assert!(key.starts_with("mcp/conversation/zen/") && key.len() == 21 + 16);
        assert!(!key.contains("session-1"));
        assert_eq!(recorded(&records, &zen(), "session-1"), None);
        records
            .put(
                &key,
                Value::String("groups/zen/conversations/2026-11-08-0308-claude-code".into()),
            )
            .unwrap();
        assert_eq!(
            recorded(&records, &zen(), "session-1").map(|p| p.show()),
            Some("@zen/conversations/2026-11-08-0308-claude-code".to_string())
        );
        assert_eq!(recorded(&records, &zen(), "session-2"), None);
    }

    #[test]
    fn the_idlest_sessions_are_let_go_past_the_cap() {
        let mut s = Sessions::default();
        let p = Kept::Open(section(&zen()));
        for i in 0..KEPT + 3 {
            s.keep(&format!("s{i}"), p.clone(), Stamp::from_secs(i as i64));
        }
        assert_eq!(s.0.len(), KEPT);
        assert!(s.get("s0").is_none() && s.get("s2").is_none());
        assert!(s.get(&format!("s{}", KEPT + 2)).is_some());
    }
}
