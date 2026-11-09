//! What waits on a member, and for whom it is read.
//!
//! The main loop reads, is reminded of and answers its **groups and
//! channels**. A direct thread is a conversation's (`engine::talk`): its
//! lines are handed to the conversation as they land, and the conversation
//! reads through the line it answered ([`super::Threads::read_through`]), so
//! the main loop is never told of them nor offered them.

use std::cmp::Reverse;

use super::thread::{Kind, Thread};
use super::{Message, Threads};

impl Threads {
    /// Read everything waiting for `who` on one of its groups, named the way
    /// `who` names it: what it had not read, marked read through the newest
    /// line *as it stood* — a line said after is not marked.
    pub fn read(&mut self, who: &str, named: &str) -> Result<Vec<Message>, String> {
        let Some(t) = self.by_kind_for(who, Kind::Group, named) else {
            return Err(format!("You have no group or channel called {named}."));
        };
        let (id, last) = (t.id.clone(), t.last_seq());
        let fresh: Vec<Message> = t.unread(who).into_iter().cloned().collect();
        self.read_through(who, &id, last);
        Ok(fresh)
    }

    /// Everything on `who`'s groups its mind has not been told of, thread by
    /// thread, under the name `who` calls each.
    pub fn undelivered_for(&self, who: &str) -> Vec<(String, Kind, Vec<Message>)> {
        self.groups_of(who)
            .into_iter()
            .filter_map(|t| {
                let waiting: Vec<Message> = t.undelivered(who).into_iter().cloned().collect();
                (!waiting.is_empty()).then(|| (t.as_named_to(who), t.kind(), waiting))
            })
            .collect()
    }

    /// Mark every group as told, for `who`.
    pub fn mark_delivered_for(&mut self, who: &str) {
        for t in self.threads.values_mut() {
            if t.kind() == Kind::Group && t.has(who) {
                t.mark_delivered(who);
            }
        }
    }

    /// Lines on direct threads that landed for `who` since its mind was
    /// told, thread by thread: what a conversation is handed. Marks them told.
    pub fn take_direct_for(&mut self, who: &str) -> Vec<(String, Vec<Message>)> {
        let mut out = Vec::new();
        for t in self.threads.values_mut() {
            if t.kind() != Kind::Direct || !t.has(who) {
                continue;
            }
            let lines: Vec<Message> = t.undelivered(who).into_iter().cloned().collect();
            if !lines.is_empty() {
                t.mark_delivered(who);
                out.push((t.id.clone(), lines));
            }
        }
        out
    }

    /// How many lines wait unread for `who` across its groups.
    pub fn waiting_for(&self, who: &str) -> usize {
        self.groups_of(who)
            .into_iter()
            .map(|t| t.unread_for(who))
            .sum()
    }

    /// The groups where somebody else spoke last since `who` came in, most
    /// recently spoken first: what the main loop's `reply` may answer on.
    pub fn awaiting_names_for(&self, who: &str) -> Vec<String> {
        let mut waiting: Vec<&Thread> = self
            .groups_of(who)
            .into_iter()
            .filter(|t| t.awaits(who))
            .collect();
        waiting.sort_by_key(|t| Reverse(t.touched));
        waiting.into_iter().map(|t| t.as_named_to(who)).collect()
    }

    /// Which groups have something unread, by the name `who` uses.
    pub fn waiting_names_for(&self, who: &str) -> Vec<String> {
        self.groups_of(who)
            .into_iter()
            .filter(|t| t.unread_for(who) > 0)
            .map(|t| t.as_named_to(who))
            .collect()
    }

    /// Every direct thread of `npc`'s where the other side said something
    /// above `npc`'s read cursor: the thread, the other member, and those
    /// lines, oldest first. What a conversation's rescan answers when the
    /// character starts or wakes on a member (`docs/npc_conversations.md` §9).
    pub fn unanswered_for(&self, npc: &str) -> Vec<(String, String, Vec<Message>)> {
        self.direct_of(npc)
            .into_iter()
            .filter_map(|t| {
                let read = t.member(npc).map_or(0, |m| m.read);
                let lines: Vec<Message> = t
                    .since_joined(npc)
                    .iter()
                    .filter(|m| m.from != npc && m.seq > read)
                    .cloned()
                    .collect();
                let other = t.other_ids(npc).into_iter().next()?;
                (!lines.is_empty()).then(|| (t.id.clone(), other, lines))
            })
            .collect()
    }

    /// Whether a direct thread with `who` waits on `npc`'s answer: the other
    /// side's newest line is above `npc`'s read cursor. What Discord's typing
    /// indicator reads.
    pub fn direct_waits_on(&self, npc: &str, who: &str) -> bool {
        self.direct_between(npc, who).is_some_and(|t| {
            let read = t.member(npc).map_or(0, |m| m.read);
            t.since_joined(npc).iter().any(|m| m.from != npc && m.seq > read)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::phone::{Said, Who, CHANNEL};

    fn who(id: &str) -> Who {
        Who::new(id, &id.to_uppercase())
    }

    /// **The main loop is told of its groups and never of a direct thread**;
    /// a conversation takes the direct thread's lines, once.
    #[test]
    fn groups_are_the_main_loops_and_direct_lines_are_taken_once() {
        let mut p = Threads::new();
        let pair = p.reach(&who("npc-1"), &who("u_8"));
        let ch = p.join_channel(CHANNEL, &who("npc-1"));
        p.join(&ch, &who("npc-2"));
        p.send_on(&pair, "u_8", Said::words("is it in?")).unwrap();
        p.send_on(&ch, "npc-2", Said::words("lift is stuck")).unwrap();
        let told = p.undelivered_for("npc-1");
        assert_eq!(told.len(), 1);
        assert_eq!(told[0].0, CHANNEL);
        assert_eq!(p.waiting_for("npc-1"), 1, "the direct line is not the main loop's");
        let taken = p.take_direct_for("npc-1");
        assert_eq!(taken, vec![(pair.clone(), vec![p.get(&pair).unwrap().messages[0].clone()])]);
        assert!(p.take_direct_for("npc-1").is_empty(), "taken twice");
    }

    /// **Reading a group marks it read as it stood**, and what waits for an
    /// answer stays waiting until it is answered.
    #[test]
    fn reading_a_group_marks_only_what_was_there_and_awaiting_needs_an_answer() {
        let mut p = Threads::new();
        let g = p.open_standing("zenworld #general", &[who("npc-1"), who("u_8")]);
        p.send_on(&g, "u_8", Said::words("zen?")).unwrap();
        assert_eq!(p.awaiting_names_for("npc-1"), ["zenworld #general"]);
        assert_eq!(p.read("npc-1", "zenworld #general").unwrap().len(), 1);
        assert_eq!(p.waiting_for("npc-1"), 0);
        assert_eq!(p.awaiting_names_for("npc-1"), ["zenworld #general"], "read is not answered");
        p.send_on(&g, "npc-1", Said::words("here")).unwrap();
        assert!(p.awaiting_names_for("npc-1").is_empty());
        assert!(p.read("npc-1", "U_8").unwrap_err().contains("no group"));
    }

    /// **A direct thread waits on the character while the other side's newest
    /// line is above its read cursor**, and stops once it reads through it —
    /// answered or written down.
    #[test]
    fn a_direct_thread_waits_until_read_through() {
        let mut p = Threads::new();
        let pair = p.reach(&who("npc-1"), &who("u_8"));
        assert!(!p.direct_waits_on("npc-1", "u_8"));
        let (seq, _) = p.send_on(&pair, "u_8", Said::words("is it in?")).unwrap();
        assert!(p.direct_waits_on("npc-1", "u_8"));
        p.read_through("npc-1", &pair, seq);
        assert!(!p.direct_waits_on("npc-1", "u_8"));
    }
}
