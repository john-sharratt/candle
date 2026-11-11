//! One conversation on a phone, who is on it, and how far each of them has
//! read and been told.
//!
//! **Members are ids** — a body id for a character, an account handle for a
//! person — never the names they go by: a name changes and collides, and a
//! thread that keyed on one lost its party the day it did. Each member keeps
//! the name it is shown by, refreshed whenever the world says it again
//! ([`super::Threads::enrol`]).
//!
//! **Every cursor is a seq**, never an index: how far a member has read, how
//! far its mind has been told, where it came in. A seq names one line for
//! ever, so trimming the front of a thread moves no cursor and no cursor can
//! come to point at the wrong line.

use serde::{Deserialize, Serialize};

use super::line::{now_ms, Message, Said};

/// Most lines a direct thread keeps. What was said before is in each side's
/// journal (`docs/npc_conversations.md` §5), which is where a conversation
/// recalls it from.
pub const KEEP_DIRECT: usize = 64;

/// Most lines a group keeps: six voices in one record need more room for the
/// same number of turns each.
pub const KEEP_GROUP: usize = 128;

/// What sort of thread it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Two members, for ever.
    Direct,
    /// A named group, or a channel.
    Group,
}

/// Somebody on a thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    /// A body id or an account handle.
    pub id: String,
    /// What they are called where the thread is shown.
    pub name: String,
    /// The seq said last before they came in: they are shown what is said
    /// after it, never the backlog.
    pub joined: u64,
    /// The highest seq they have read.
    pub read: u64,
    /// The highest seq their mind has been told of.
    pub delivered: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thread {
    pub id: String,
    /// What a group is called. A direct thread is called by its other
    /// member's name, to each side.
    pub name: String,
    pub kind: Kind,
    pub members: Vec<Member>,
    /// What has been said, oldest first, trimmed to what its kind keeps.
    pub messages: Vec<Message>,
    /// Whether it is offered. A direct thread is never closed.
    pub open: bool,
    /// Part of the world rather than something somebody started: a world's
    /// channel, or a Discord channel a bot is in. Never evicted.
    pub permanent: bool,
    /// What it is for, in its own words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    /// When anything was last said on it, on [`super::Threads`]'s counter.
    pub touched: u64,
    /// The seq the next line takes.
    pub next_seq: u64,
}

impl Thread {
    pub(super) fn new(id: String, name: &str, kind: Kind, members: Vec<Member>) -> Thread {
        Thread {
            id,
            name: name.to_string(),
            kind,
            members,
            messages: Vec::new(),
            open: true,
            permanent: false,
            topic: None,
            touched: 0,
            next_seq: 1,
        }
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn has(&self, id: &str) -> bool {
        self.members.iter().any(|m| m.id == id)
    }

    pub fn member(&self, id: &str) -> Option<&Member> {
        self.members.iter().find(|m| m.id == id)
    }

    pub(super) fn member_mut(&mut self, id: &str) -> Option<&mut Member> {
        self.members.iter_mut().find(|m| m.id == id)
    }

    /// Everybody but `than`.
    pub fn others(&self, than: &str) -> Vec<&Member> {
        self.members.iter().filter(|m| m.id != than).collect()
    }

    /// The ids of everybody but `than`.
    pub fn other_ids(&self, than: &str) -> Vec<String> {
        self.others(than)
            .into_iter()
            .map(|m| m.id.clone())
            .collect()
    }

    /// What this thread is called to member `who`: a group by its name, a
    /// direct thread by the other member's.
    pub fn as_named_to(&self, who: &str) -> String {
        match self.kind {
            Kind::Group => self.name.clone(),
            Kind::Direct => self
                .others(who)
                .first()
                .map_or_else(|| self.name.clone(), |m| m.name.clone()),
        }
    }

    /// The seq of the newest line, or `0` before anything was said.
    pub fn last_seq(&self) -> u64 {
        self.next_seq - 1
    }

    /// Add `who`, at the end of what has been said. `false` when they are
    /// on it already.
    pub(super) fn add(&mut self, who: &str, name: &str) -> bool {
        if self.has(who) {
            return false;
        }
        let at = self.last_seq();
        self.members.push(Member {
            id: who.to_string(),
            name: name.to_string(),
            joined: at,
            read: at,
            delivered: at,
        });
        true
    }

    /// Say `said` from `from`, naming it with the thread's next seq and
    /// dropping the oldest once the thread holds more than its kind keeps.
    /// The sender has read what they said.
    pub(super) fn say(&mut self, from: &str, said: Said) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.messages.push(Message {
            from: from.to_string(),
            text: said.text,
            intent: said.intent,
            seq,
            at_ms: now_ms(),
            origin: said.origin,
            to: said.to,
            requests: said.requests,
            account: said.account,
        });
        if let Some(m) = self.member_mut(from) {
            m.read = seq;
            m.delivered = seq;
        }
        let keep = match self.kind {
            Kind::Direct => KEEP_DIRECT,
            Kind::Group => KEEP_GROUP,
        };
        if let Some(over) = self.messages.len().checked_sub(keep).filter(|o| *o > 0) {
            self.messages.drain(..over);
        }
        seq
    }

    /// What was said since `who` came in, oldest first.
    pub fn since_joined(&self, who: &str) -> &[Message] {
        let joined = self.member(who).map_or(u64::MAX, |m| m.joined);
        let from = self.messages.partition_point(|m| m.seq <= joined);
        &self.messages[from..]
    }

    /// Lines from others after `who`'s `cursor`, oldest first.
    fn after(&self, who: &str, cursor: u64) -> Vec<&Message> {
        self.since_joined(who)
            .iter()
            .filter(|m| m.seq > cursor && m.from != who)
            .collect()
    }

    /// What `who` has not read.
    pub fn unread(&self, who: &str) -> Vec<&Message> {
        let read = self.member(who).map_or(0, |m| m.read);
        self.after(who, read)
    }

    pub fn unread_for(&self, who: &str) -> usize {
        self.unread(who).len()
    }

    /// What `who`'s mind has not been told of.
    pub fn undelivered(&self, who: &str) -> Vec<&Message> {
        let told = self.member(who).map_or(0, |m| m.delivered);
        self.after(who, told)
    }

    /// Mark `who`'s reading up to `seq` — never past what has been said, and
    /// never back. A line that arrived after `seq` stays unread, which a
    /// read of "everything" would have hidden.
    pub(super) fn read_through(&mut self, who: &str, seq: u64) {
        let last = self.last_seq();
        if let Some(m) = self.member_mut(who) {
            m.read = m.read.max(seq.min(last));
        }
    }

    /// Mark everything said so far as told to `who`'s mind.
    pub(super) fn mark_delivered(&mut self, who: &str) {
        let last = self.last_seq();
        if let Some(m) = self.member_mut(who) {
            m.delivered = last;
        }
    }

    /// Whether the newest line since `who` came in is somebody else's: a
    /// thread waiting on `who`'s answer, read or not.
    pub fn awaits(&self, who: &str) -> bool {
        self.since_joined(who).last().is_some_and(|m| m.from != who)
    }

    /// `who`'s own newest line, when they have said one since coming in.
    pub fn last_said_by(&self, who: &str) -> Option<&Message> {
        self.since_joined(who).iter().rev().find(|m| m.from == who)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> Thread {
        let mut t = Thread::new("dm:npc-1:u_8".into(), "", Kind::Direct, Vec::new());
        t.add("npc-1", "Zen");
        t.add("u_8", "Mira");
        t
    }

    /// **Trimming moves no cursor**: every cursor is a seq, so a member who
    /// read through seq 40 has read through the same line after the front of
    /// the thread is dropped.
    #[test]
    fn trimming_moves_no_cursor() {
        let mut t = pair();
        for i in 0..(KEEP_DIRECT + 20) {
            t.say("u_8", Said::words(&format!("line {i}")));
        }
        t.read_through("npc-1", 40);
        assert_eq!(t.messages.len(), KEEP_DIRECT);
        assert_eq!(t.messages[0].seq, 21);
        assert_eq!(t.member("npc-1").unwrap().read, 40);
        let unread: Vec<u64> = t.unread("npc-1").iter().map(|m| m.seq).collect();
        assert_eq!(unread, (41..=84).collect::<Vec<_>>());
        assert_eq!(t.last_seq(), 84);
    }

    /// **Reading through a seq leaves what came after it unread**, and never
    /// runs past what has been said or back below what was read.
    #[test]
    fn reading_through_a_seq_is_bounded() {
        let mut t = pair();
        for text in ["one", "two", "three"] {
            t.say("u_8", Said::words(text));
        }
        t.read_through("npc-1", 2);
        assert_eq!(t.unread("npc-1").len(), 1);
        assert_eq!(t.unread("npc-1")[0].text, "three");
        t.read_through("npc-1", 1);
        assert_eq!(t.member("npc-1").unwrap().read, 2, "never back");
        t.read_through("npc-1", 99);
        assert_eq!(t.member("npc-1").unwrap().read, 3, "never past the end");
    }

    /// **A newcomer is shown what is said after they came in**, and the
    /// sender has read and been told of their own line.
    #[test]
    fn a_newcomer_sees_only_what_comes_after_and_the_sender_has_read_their_own() {
        let mut t = Thread::new("g1".into(), "the sweep", Kind::Group, Vec::new());
        t.add("npc-1", "Wren");
        t.say("npc-1", Said::words("before"));
        t.add("npc-2", "Soren");
        assert!(t.unread("npc-2").is_empty());
        assert!(t.undelivered("npc-2").is_empty());
        t.say("npc-1", Said::words("after"));
        assert_eq!(t.unread("npc-2")[0].text, "after");
        assert!(t.unread("npc-1").is_empty());
        assert!(t.undelivered("npc-1").is_empty());
        assert!(!t.add("npc-2", "Soren"), "added twice");
    }

    /// **A direct thread is called by the other member's name, to each
    /// side**, and what waits on a member is a thread whose newest line is
    /// another's.
    #[test]
    fn a_direct_thread_is_named_for_the_other_side_and_awaits_whoever_spoke_last_to() {
        let mut t = pair();
        assert_eq!(t.as_named_to("npc-1"), "Mira");
        assert_eq!(t.as_named_to("u_8"), "Zen");
        t.say("u_8", Said::words("is it in?"));
        assert!(t.awaits("npc-1") && !t.awaits("u_8"));
        t.say("npc-1", Said::words("It's at the door."));
        assert!(t.awaits("u_8") && !t.awaits("npc-1"));
        assert_eq!(t.last_said_by("npc-1").unwrap().text, "It's at the door.");
        assert_eq!(t.other_ids("npc-1"), ["u_8"]);
    }
}
