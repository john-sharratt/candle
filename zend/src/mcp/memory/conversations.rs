//! Each conversation's own journal (§2): opened under its channel's at the
//! conversation's first episode, named for the minute it began, and the one
//! its episodes go on.
//!
//! A conversation is a session's run of calls with no gap of [`GAP_SECS`] —
//! the aggregator's own boundary (`docs/mcp_user_context.md` §2.5). Every
//! memory call of a session moves its clock on ([`touch`]), not only an
//! episode. A session this process has not seen — a new conversation, or one
//! carried here by a restart or from another member — opens a journal of its
//! own: two conversations are never written onto one.

use std::collections::HashMap;
use std::sync::Mutex;

use journals::actor::Actor;
use journals::ops::{OnTaken, Op};
use journals::path::{JournalPath, Owner};
use journals::scope::Channel;
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

    fn touch(&mut self, session: &str, now: Stamp) {
        if let Some((_, last)) = self.0.get_mut(session) {
            if now.secs() - last.secs() < GAP_SECS {
                *last = now;
            }
        }
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

/// The channel's journal a person's conversations go under.
pub fn channel_journal(person: &Slug, channel: Channel) -> JournalPath {
    let names = ["channels", channel.name()]
        .iter()
        .map(|n| Slug::parse(n).expect("a section's and a channel's names are names"))
        .collect();
    JournalPath::new(Owner::Person(person.clone()), names).expect("channels/<channel> is a place")
}

/// The name a conversation begun at `at` takes: `2026-10-30-2105`.
pub fn name_at(at: Stamp) -> Slug {
    Slug::parse(&at.dated_minute()).expect("a minute is a name")
}

/// A memory call of `session` now: its conversation goes on.
pub fn touch(memory: &Memory, session: &str) {
    held(&memory.conversations.sessions).touch(session, Stamp::now());
}

/// The journal of the conversation `session` is in, when it has one yet.
pub fn existing(memory: &Memory, session: &str) -> Option<JournalPath> {
    let tree = memory.tree().ok()?;
    let known = held(&memory.conversations.sessions)
        .get(session, Stamp::now())
        .cloned();
    known.filter(|p| tree.get(p).is_some())
}

/// The journal of the conversation `session` is in, opened now when this is
/// its first episode.
pub fn of(
    memory: &Memory,
    person: &Slug,
    channel: Channel,
    session: &str,
) -> Result<JournalPath, String> {
    let _opening = held(&memory.conversations.opening);
    if let Some(p) = existing(memory, session) {
        held(&memory.conversations.sessions).keep(session, p.clone(), Stamp::now());
        return Ok(p);
    }
    let now = Stamp::now();
    let server = Actor::Server {
        person: person.clone(),
        channel,
    };
    let began = now.to_string()[..16].replace('T', " ");
    let op = Op::Create {
        under: channel_journal(person, channel),
        name: name_at(now),
        cue: format!("open for the conversation through {channel} begun {began}"),
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

    fn john() -> Slug {
        Slug::parse("john").unwrap()
    }

    #[test]
    fn a_conversation_is_named_for_its_minute_under_its_channel() {
        let at = Stamp::parse("2026-10-30T21:05:42Z").unwrap();
        assert_eq!(name_at(at).as_str(), "2026-10-30-2105");
        assert_eq!(
            channel_journal(&john(), Channel::Grok).folder(),
            "persons/john/channels/grok"
        );
    }

    #[test]
    fn a_session_silent_past_the_gap_is_a_new_conversation() {
        let mut s = Sessions::default();
        let p = channel_journal(&john(), Channel::Grok);
        let t = Stamp::parse("2026-10-30T12:00:00Z").unwrap();
        s.keep("s", p.clone(), t);
        assert_eq!(s.get("s", t.plus(GAP_SECS - 1)), Some(&p));
        assert_eq!(s.get("s", t.plus(GAP_SECS)), None);
        assert_eq!(s.get("other", t), None);
    }

    /// **Any memory call keeps a conversation going**, not only an episode;
    /// a session already past its gap is not brought back by one.
    #[test]
    fn a_call_moves_the_clock_on_within_the_gap() {
        let mut s = Sessions::default();
        let p = channel_journal(&john(), Channel::Grok);
        let t = Stamp::parse("2026-10-30T12:00:00Z").unwrap();
        s.keep("s", p.clone(), t);
        s.touch("s", t.plus(GAP_SECS - 60));
        assert_eq!(s.get("s", t.plus(GAP_SECS + 60)), Some(&p));
        s.touch("s", t.plus(3 * GAP_SECS));
        assert_eq!(s.get("s", t.plus(3 * GAP_SECS)), None);
        s.touch("unseen", t);
        assert_eq!(s.get("unseen", t), None);
    }

    #[test]
    fn the_idlest_sessions_are_let_go_past_the_cap() {
        let mut s = Sessions::default();
        let p = channel_journal(&john(), Channel::Grok);
        for i in 0..KEPT + 3 {
            s.keep(&format!("s{i}"), p.clone(), Stamp::from_secs(i as i64));
        }
        assert_eq!(s.0.len(), KEPT);
        let late = Stamp::from_secs(KEPT as i64);
        assert!(s.get("s0", late).is_none() && s.get("s2", late).is_none());
        assert!(s.get(&format!("s{}", KEPT + 2), late).is_some());
    }
}
