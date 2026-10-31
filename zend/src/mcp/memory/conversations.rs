//! Each conversation's own journal (§2.4, §6.6): named for the minute the
//! conversation began and the channel it began on, under its group's
//! `conversations`, and opened at the conversation's first write. It holds
//! the conversation's focus, its running tasks and the episodes of what it
//! did. A conversation that only reads writes nothing: its frame names the
//! journal it would open, so `here` means the same before and after.
//!
//! A conversation is a session's run of calls with no gap of [`GAP_SECS`] —
//! the aggregator's own boundary (`docs/mcp_user_context.md` §2.5). Every
//! memory call of a session moves its clock on. A session this process has
//! not seen — a new conversation, or one carried here by a restart or from
//! another member — begins a conversation of its own: two conversations are
//! never written onto one.
//!
//! Grok opens a session for every call it makes, so its calls are followed
//! by the person and group instead ([`followed_by`]): a drive is one
//! conversation until it goes quiet.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use journals::actor::Actor;
use journals::ops::{OnTaken, Op};
use journals::path::JournalPath;
use journals::scope::{conversation_name, Channel, Section};
use journals::slug::Slug;
use journals::time::Stamp;

use super::{held, Memory};

/// A conversation ends after this long without a call: the aggregator's
/// `CONVERSATION_GAP_MS`.
pub const GAP_SECS: i64 = 30 * 60;

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
    fn get(&self, session: &str, now: Stamp) -> Option<&Kept> {
        self.0
            .get(session)
            .filter(|(_, last)| now.secs() - last.secs() < GAP_SECS)
            .map(|(k, _)| k)
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
        Channel::ClaudeCode | Channel::ClaudeAi | Channel::Other => session.to_string(),
    }
}

/// The name a conversation begun at `at` on `channel` reserves in a section
/// whose journals and begun conversations hold `taken`: its minute's name,
/// numbered past any already held — as its journal's create would number it.
pub fn reserve(at: Stamp, channel: Channel, taken: &BTreeSet<Slug>) -> Slug {
    name_at(at, channel).numbered(taken)
}

/// Where `session`'s conversation in `group` stands: its journal, or — when
/// nothing has written to it — the moment it began and the name it reserved,
/// begun now when this is its first call. Writes nothing; the session's
/// clock moves on.
fn kept(
    memory: &Memory,
    group: &Slug,
    channel: Channel,
    session: &str,
    now: Stamp,
) -> Result<Kept, String> {
    let tree = memory.tree()?;
    let mut sessions = held(&memory.conversations.sessions);
    let kept = match sessions.get(session, now) {
        Some(Kept::Open(p)) if p.group() == group && tree.get(p).is_some() => Kept::Open(p.clone()),
        Some(Kept::Begun(at, name)) => Kept::Begun(*at, name.clone()),
        _ => {
            let mut taken: BTreeSet<Slug> = tree
                .get(&section(group))
                .map(|j| j.children.keys().cloned().collect())
                .unwrap_or_default();
            taken.extend(sessions.reserved().cloned());
            Kept::Begun(now, reserve(now, channel, &taken))
        }
    };
    sessions.keep(session, kept.clone(), now);
    Ok(kept)
}

/// The journal `session`'s conversation in `group` is written to, as a read
/// sees it: opened, or the one its first write opens. Writes nothing.
pub fn here(
    memory: &Memory,
    group: &Slug,
    channel: Channel,
    session: &str,
) -> Result<JournalPath, String> {
    match kept(memory, group, channel, session, Stamp::now())? {
        Kept::Open(p) => Ok(p),
        Kept::Begun(_, name) => section(group).child(name).map_err(|e| e.to_string()),
    }
}

/// The journal of `session`'s conversation in `group`, opened now when this
/// is the conversation's first write, under the name its reads were given.
pub fn open(
    memory: &Memory,
    person: &Slug,
    group: &Slug,
    channel: Channel,
    session: &str,
) -> Result<JournalPath, String> {
    let _opening = held(&memory.conversations.opening);
    let now = Stamp::now();
    let (began, name) = match kept(memory, group, channel, session, now)? {
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
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn a_session_silent_past_the_gap_is_a_new_conversation() {
        let mut s = Sessions::default();
        let p = Kept::Open(section(&zen()));
        let t = Stamp::parse("2026-10-30T12:00:00Z").unwrap();
        s.keep("s", p.clone(), t);
        assert_eq!(s.get("s", t.plus(GAP_SECS - 1)), Some(&p));
        assert_eq!(s.get("s", t.plus(GAP_SECS)), None);
        assert_eq!(s.get("other", t), None);
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

    #[test]
    fn the_idlest_sessions_are_let_go_past_the_cap() {
        let mut s = Sessions::default();
        let p = Kept::Open(section(&zen()));
        for i in 0..KEPT + 3 {
            s.keep(&format!("s{i}"), p.clone(), Stamp::from_secs(i as i64));
        }
        assert_eq!(s.0.len(), KEPT);
        let late = Stamp::from_secs(KEPT as i64);
        assert!(s.get("s0", late).is_none() && s.get("s2", late).is_none());
        assert!(s.get(&format!("s{}", KEPT + 2), late).is_some());
    }
}
