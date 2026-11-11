//! Channels and groups (`docs/npc_conversations.md` §11.2): the main loop's
//! fire-and-forget threads. A world's standing [`CHANNEL`], the Discord
//! channels a bot stands in, and the groups characters open.
//!
//! Members are ids, and a newcomer comes in at the end of what has been
//! said. Each thread is bounded by what it keeps ([`super::thread::KEEP_GROUP`]);
//! the groups characters open, which a character may open on any turn, are
//! bounded in number too — the quietest goes past [`KEEP_GROUPS`]. Nothing
//! permanent, and no direct thread, is ever a candidate.

use std::slice;

use super::thread::{Kind, Thread};
use super::{Threads, Who};

/// What every world's standing channel is called: the plainest name for the
/// one thread every character with a handset is on from arrival.
pub const CHANNEL: &str = "the channel";

/// How many groups characters have opened a world keeps; past it, the one
/// that went quiet longest ago goes.
pub const KEEP_GROUPS: usize = 64;

impl Threads {
    /// A new group thread named `name`, with `members`.
    fn new_group(&mut self, name: &str, members: &[Who], permanent: bool) -> String {
        self.next += 1;
        let id = format!("th{}", self.next);
        let mut t = Thread::new(id.clone(), name, Kind::Group, Vec::new());
        for m in members {
            t.add(&m.id, &m.name);
        }
        t.permanent = permanent;
        t.touched = self.tick();
        self.threads.insert(id.clone(), t);
        id
    }

    /// The world's standing channel, with `who` on it — join-or-create, and
    /// idempotent: called on every arrival.
    pub fn join_channel(&mut self, name: &str, who: &Who) -> String {
        let found = self
            .threads
            .values()
            .find(|t| t.permanent && t.kind() == Kind::Group && t.name == name)
            .map(|t| t.id.clone());
        match found {
            Some(id) => {
                self.join(&id, who);
                id
            }
            None => self.new_group(name, slice::from_ref(who), true),
        }
    }

    /// Open a named group outright.
    pub fn open_group(&mut self, name: &str, members: &[Who]) -> String {
        let id = self.new_group(name, members, false);
        self.forget_quiet_groups();
        id
    }

    /// A group that stands as part of the world for as long as what it
    /// mirrors does — a Discord channel a character's bot is in.
    pub fn open_standing(&mut self, name: &str, members: &[Who]) -> String {
        self.new_group(name, members, true)
    }

    /// Drop the groups characters opened that went quiet longest ago, down to
    /// [`KEEP_GROUPS`]. The one just opened is the most recent, so it is never
    /// the one dropped.
    fn forget_quiet_groups(&mut self) {
        let mut age: Vec<(u64, String)> = self
            .threads
            .values()
            .filter(|t| t.kind() == Kind::Group && !t.permanent)
            .map(|t| (t.touched, t.id.clone()))
            .collect();
        let Some(over) = age.len().checked_sub(KEEP_GROUPS).filter(|o| *o > 0) else {
            return;
        };
        age.sort();
        for (_, id) in age.into_iter().take(over) {
            self.threads.remove(&id);
        }
    }

    /// Bring `who` onto group `id` at its end. `false` when they are on it,
    /// there is no such thread, or it is a direct thread — which never gains
    /// a third member.
    pub fn join(&mut self, id: &str, who: &Who) -> bool {
        match self.threads.get_mut(id) {
            Some(t) if t.kind() == Kind::Group => t.add(&who.id, &who.name),
            _ => false,
        }
    }

    /// Call group `id` `name` from now on.
    pub fn rename(&mut self, id: &str, name: &str) -> bool {
        match self.threads.get_mut(id) {
            Some(t) if t.kind() == Kind::Group => {
                t.name = name.to_string();
                true
            }
            _ => false,
        }
    }

    /// Say what group `id` is for; a blank topic is none.
    pub fn describe(&mut self, id: &str, topic: Option<&str>) -> bool {
        let Some(t) = self.threads.get_mut(id) else {
            return false;
        };
        t.topic = topic
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        true
    }

    /// Close group `id`: nobody is offered it, what was said is kept, and it
    /// stops being permanent. A direct thread is never closed.
    pub fn close(&mut self, id: &str) -> bool {
        match self.threads.get_mut(id) {
            Some(t) if t.kind() == Kind::Group => {
                t.open = false;
                t.permanent = false;
                true
            }
            _ => false,
        }
    }

    /// Open group `id` again, standing.
    pub fn reopen(&mut self, id: &str) -> bool {
        match self.threads.get_mut(id) {
            Some(t) if t.kind() == Kind::Group => {
                t.open = true;
                t.permanent = true;
                true
            }
            _ => false,
        }
    }

    /// Bring `who` onto one of `by`'s groups, named the way `by` names it.
    /// Refused on a direct thread: a group starts with `open_group`.
    pub fn invite(&mut self, by: &str, named: &str, who: &Who) -> Result<Kind, String> {
        let Some(id) = self.by_name_for(by, named).map(|t| t.id.clone()) else {
            return Err(format!("You have no conversation with {named}."));
        };
        let t = self.threads.get_mut(&id).expect("just found");
        if t.kind() == Kind::Direct {
            return Err(format!(
                "Your conversation with {named} is the two of you. To talk with more people \
                 at once, open a group."
            ));
        }
        if !t.add(&who.id, &who.name) {
            return Err(format!("{} is already on it.", who.name));
        }
        Ok(Kind::Group)
    }

    /// Leave one of `who`'s groups. A direct thread is never left: `sign_off`
    /// ends the conversation, and the thread stays the pair's.
    pub fn leave(&mut self, who: &str, named: &str) -> Result<(), String> {
        let Some(id) = self.by_name_for(who, named).map(|t| t.id.clone()) else {
            return Err(format!("You have no conversation with {named}."));
        };
        let t = self.threads.get_mut(&id).expect("just found");
        if t.kind() == Kind::Direct {
            return Err(format!(
                "Your conversation with {named} stays yours: sign off to end it."
            ));
        }
        t.members.retain(|m| m.id != who);
        if t.members.is_empty() {
            t.open = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::phone::Said;

    fn who(id: &str) -> Who {
        Who::new(id, &id.to_uppercase())
    }

    fn crew() -> Vec<Who> {
        ["npc-1", "npc-2", "npc-3"].map(who).to_vec()
    }

    #[test]
    fn joining_the_channel_twice_neither_duplicates_it_nor_the_member() {
        let mut p = Threads::new();
        let a = p.join_channel(CHANNEL, &who("npc-1"));
        let b = p.join_channel(CHANNEL, &who("npc-1"));
        assert_eq!(a, b);
        assert_eq!(p.get(&a).unwrap().members.len(), 1);
        p.join_channel(CHANNEL, &who("npc-2"));
        assert_eq!(p.get(&a).unwrap().members.len(), 2);
        assert_eq!(p.get(&a).unwrap().kind(), Kind::Group);
    }

    /// **The groups characters open are bounded; nothing permanent and no
    /// direct thread is ever dropped for them.**
    #[test]
    fn only_opened_groups_are_dropped_and_the_quietest_first() {
        let mut p = Threads::new();
        let channel = p.join_channel(CHANNEL, &who("npc-1"));
        let pair = p.reach(&who("npc-1"), &who("u_8"));
        let dead = p.open_group("dead air", &crew());
        let live = p.open_group("live", &crew());
        p.send_on(&live, "npc-1", Said::words("still going"))
            .unwrap();
        for i in 0..(KEEP_GROUPS * 2) {
            p.open_group(&format!("crew {i}"), &crew());
        }
        assert!(p.get(&channel).is_some() && p.get(&pair).is_some());
        assert!(p.get(&dead).is_none(), "the quiet group survived");
        let groups = p
            .iter()
            .filter(|t| t.kind() == Kind::Group && !t.permanent)
            .count();
        assert_eq!(groups, KEEP_GROUPS);
    }

    /// **A direct thread is never invited into, left or closed.**
    #[test]
    fn a_direct_thread_takes_no_third_member_and_is_never_left() {
        let mut p = Threads::new();
        let id = p.reach(&who("npc-1"), &who("u_8"));
        assert!(p
            .invite("npc-1", "U_8", &who("npc-2"))
            .unwrap_err()
            .contains("open a group"));
        assert!(p.leave("npc-1", "U_8").unwrap_err().contains("sign off"));
        assert!(!p.close(&id) && !p.join(&id, &who("npc-2")));
        assert_eq!(p.get(&id).unwrap().members.len(), 2);
    }

    #[test]
    fn a_standing_group_is_renamed_closed_and_reopened() {
        let mut p = Threads::new();
        let id = p.open_standing("zenworld #general", &[who("npc-1")]);
        assert!(p.rename(&id, "zenworld #lobby"));
        assert_eq!(p.names_for("npc-1"), ["zenworld #lobby"]);
        assert!(p.close(&id));
        assert!(p.names_for("npc-1").is_empty());
        assert!(p.reopen(&id));
        assert!(p.describe(&id, Some(" findings ")));
        assert_eq!(p.get(&id).unwrap().topic.as_deref(), Some("findings"));
    }

    #[test]
    fn a_group_member_leaves_and_an_invitee_comes_in_at_its_end() {
        let mut p = Threads::new();
        let id = p.open_group("the sweep", &crew());
        p.send_on(&id, "npc-1", Said::words("before")).unwrap();
        assert_eq!(
            p.invite("npc-1", "the sweep", &who("npc-4")),
            Ok(Kind::Group)
        );
        assert!(p.get(&id).unwrap().unread("npc-4").is_empty());
        p.leave("npc-1", "the sweep").unwrap();
        assert_eq!(p.names_for("npc-2"), ["the sweep"]);
        assert!(p.get(&id).unwrap().open);
    }
}
