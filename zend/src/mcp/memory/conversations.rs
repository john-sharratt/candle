//! Each conversation's own journal (§2.4, §6.6): opened under its group's
//! `conversations` at the conversation's first call, named for the minute it
//! began and the channel it began on. It holds the conversation's focus, its
//! running tasks and the episodes of what it did.
//!
//! A conversation is a session's run of calls with no gap of [`GAP_SECS`] —
//! the aggregator's own boundary (`docs/mcp_user_context.md` §2.5). Every
//! memory call of a session moves its clock on. A session this process has
//! not seen — a new conversation, or one carried here by a restart or from
//! another member — opens a journal of its own: two conversations are never
//! written onto one.

use std::collections::HashMap;
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

/// Each session's journal, and when it last had a call.
#[derive(Default)]
struct Sessions(HashMap<String, (JournalPath, Stamp)>);

impl Sessions {
    fn get(&self, session: &str, now: Stamp) -> Option<&JournalPath> {
        self.0
            .get(session)
            .filter(|(_, last)| now.secs() - last.secs() < GAP_SECS)
            .map(|(p, _)| p)
    }

    fn keep(&mut self, session: &str, path: JournalPath, at: Stamp) {
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
        self.0.insert(session.to_string(), (path, at));
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

/// The journal of the conversation `session` is in, in `group`, opened now
/// when this is its first call: the session's clock moved on either way.
pub fn of(
    memory: &Memory,
    person: &Slug,
    group: &Slug,
    channel: Channel,
    session: &str,
) -> Result<JournalPath, String> {
    let _opening = held(&memory.conversations.opening);
    let now = Stamp::now();
    let tree = memory.tree()?;
    let known = held(&memory.conversations.sessions)
        .get(session, now)
        .filter(|p| p.group() == group && tree.get(p).is_some())
        .cloned();
    if let Some(p) = known {
        held(&memory.conversations.sessions).keep(session, p.clone(), now);
        return Ok(p);
    }
    let server = Actor::Server {
        person: person.clone(),
        channel,
    };
    let began = now.to_string()[..16].replace('T', " ");
    let op = Op::Create {
        under: section(group),
        name: name_at(now, channel),
        cue: format!(
            "open for the conversation through {} begun {began}",
            channel.spoken()
        ),
        on_taken: OnTaken::Number,
    };
    let path = memory
        .write_at(&server, &op, now)?
        .created
        .expect("create makes a journal");
    held(&memory.conversations.sessions).keep(session, path.clone(), now);
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

    #[test]
    fn a_session_silent_past_the_gap_is_a_new_conversation() {
        let mut s = Sessions::default();
        let p = section(&zen());
        let t = Stamp::parse("2026-10-30T12:00:00Z").unwrap();
        s.keep("s", p.clone(), t);
        assert_eq!(s.get("s", t.plus(GAP_SECS - 1)), Some(&p));
        assert_eq!(s.get("s", t.plus(GAP_SECS)), None);
        assert_eq!(s.get("other", t), None);
    }

    #[test]
    fn the_idlest_sessions_are_let_go_past_the_cap() {
        let mut s = Sessions::default();
        let p = section(&zen());
        for i in 0..KEPT + 3 {
            s.keep(&format!("s{i}"), p.clone(), Stamp::from_secs(i as i64));
        }
        assert_eq!(s.0.len(), KEPT);
        let late = Stamp::from_secs(KEPT as i64);
        assert!(s.get("s0", late).is_none() && s.get("s2", late).is_none());
        assert!(s.get(&format!("s{}", KEPT + 2), late).is_some());
    }
}
