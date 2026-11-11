//! The thing a character carries that lets it reach somebody who is not here
//! (`docs/npc_conversations.md` §11).
//!
//! | A room | A phone |
//! |---|---|
//! | one at a time | as many threads as you have |
//! | whoever is standing there | whoever is on the thread |
//! | now | whenever, and it waits |
//!
//! It is carried — [`PHONE`] is an item in the pack — so it can be absent, and
//! a character without one has no threads and no phone acts.
//!
//! | Module | |
//! |---|---|
//! | [`line`] | one thing said: its words, who said it, what it asked |
//! | [`thread`] | one thread: members by id, every cursor a seq |
//! | [`direct`] | one thread per pair, for ever |
//! | [`channel`] | the world's channel, Discord channels, groups |
//! | [`read`] | what waits, unread and untold, for whom |
//!
//! **Direct threads are conversations' and groups are the main loop's.** A
//! direct thread is the one a character and its party hold a conversation on
//! (`engine::talk`); the main loop reads, answers and leaves only groups and
//! channels, so its acts never touch a thread a conversation is answering.

pub mod channel;
pub mod direct;
pub mod line;
pub mod read;
pub mod thread;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::unread::Reminded;

pub use channel::{CHANNEL, KEEP_GROUPS};
pub use direct::{direct_id, other_of};
pub use line::{Message, Origin, Said};
pub use thread::{Kind, Member, Thread, KEEP_DIRECT, KEEP_GROUP};

pub(crate) use line::now_ms;

/// The item that makes any of this reachable. Carried like anything else.
pub const PHONE: &str = "handset";

/// What `read` takes for every group with something unread at once.
pub const ALL_MESSAGES: &str = "messages";

/// Somebody a thread can hold: their id, and the name they are shown by.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Who {
    pub id: String,
    pub name: String,
}

impl Who {
    pub fn new(id: &str, name: &str) -> Who {
        Who {
            id: id.to_string(),
            name: name.to_string(),
        }
    }
}

/// Every thread in a world.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Threads {
    threads: BTreeMap<String, Thread>,
    next: u64,
    /// Ticks once per thing said, so groups can be ordered by how recently
    /// anybody spoke on them.
    clock: u64,
    /// What each member was last reminded waits unread, and when — see
    /// [`crate::sim::unread`].
    #[serde(default)]
    pub(crate) reminded: BTreeMap<String, Reminded>,
}

impl Threads {
    pub fn new() -> Threads {
        Threads::default()
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    pub fn len(&self) -> usize {
        self.threads.len()
    }

    pub fn get(&self, id: &str) -> Option<&Thread> {
        self.threads.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Thread> {
        self.threads.values()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Call `id` `name` wherever it is a member: the world said it again.
    pub fn enrol(&mut self, id: &str, name: &str) {
        for t in self.threads.values_mut() {
            if let Some(m) = t.member_mut(id) {
                if m.name != name {
                    m.name = name.to_string();
                }
            }
        }
    }

    /// The name `id` is shown by on any thread it is on.
    pub fn name_of(&self, id: &str) -> Option<&str> {
        self.threads
            .values()
            .find_map(|t| t.member(id))
            .map(|m| m.name.as_str())
    }

    /// The open threads `who` is on.
    pub fn of(&self, who: &str) -> Vec<&Thread> {
        self.threads
            .values()
            .filter(|t| t.open && t.has(who))
            .collect()
    }

    /// The open groups and channels `who` is on: the main loop's threads.
    pub fn groups_of(&self, who: &str) -> Vec<&Thread> {
        self.of(who)
            .into_iter()
            .filter(|t| t.kind() == Kind::Group)
            .collect()
    }

    /// What `who` calls each of its threads.
    pub fn names_for(&self, who: &str) -> Vec<String> {
        self.of(who)
            .into_iter()
            .map(|t| t.as_named_to(who))
            .collect()
    }

    /// What `who` calls each of its threads of `kind`.
    pub fn names_of_kind_for(&self, who: &str, kind: Kind) -> Vec<String> {
        self.of(who)
            .into_iter()
            .filter(|t| t.kind() == kind)
            .map(|t| t.as_named_to(who))
            .collect()
    }

    /// The channels `who` is on: the world's own, and each Discord channel
    /// that stands as one.
    pub fn channels_for(&self, who: &str) -> Vec<&Thread> {
        self.groups_of(who)
            .into_iter()
            .filter(|t| t.permanent)
            .collect()
    }

    /// One of `who`'s threads of `kind`, by the name `who` calls it or its id.
    pub fn by_kind_for(&self, who: &str, kind: Kind, name: &str) -> Option<&Thread> {
        self.by_name_among(who, name, |t| t.kind() == kind)
    }

    /// One of `who`'s threads, by the name `who` calls it or its id.
    pub fn by_name_for(&self, who: &str, name: &str) -> Option<&Thread> {
        self.by_name_among(who, name, |_| true)
    }

    fn by_name_among(
        &self,
        who: &str,
        name: &str,
        keep: impl Fn(&Thread) -> bool,
    ) -> Option<&Thread> {
        let want = name.trim().to_lowercase();
        self.of(who)
            .into_iter()
            .filter(|t| keep(t))
            .find(|t| t.as_named_to(who).to_lowercase() == want || t.id == want)
    }

    /// Say `said` on thread `id` from `from`, who is on it, and it is open:
    /// its seq, and the other members' ids.
    pub fn send_on(
        &mut self,
        id: &str,
        from: &str,
        said: Said,
    ) -> Result<(u64, Vec<String>), String> {
        if !self.threads.get(id).is_some_and(|t| t.open && t.has(from)) {
            return Err(format!("{from} is on no open conversation {id}."));
        }
        let touched = self.tick();
        let t = self.threads.get_mut(id).expect("just found");
        t.touched = touched;
        let seq = t.say(from, said);
        Ok((seq, t.other_ids(from)))
    }

    /// Say `said` on one of `from`'s threads, named the way `from` names it.
    pub fn send(
        &mut self,
        from: &str,
        named: &str,
        said: Said,
    ) -> Result<(u64, Vec<String>), String> {
        let Some(id) = self.by_name_for(from, named).map(|t| t.id.clone()) else {
            return Err(format!("You have no conversation with {named}."));
        };
        self.send_on(&id, from, said)
    }

    /// Mark `who`'s reading of thread `id` up to `seq`.
    pub fn read_through(&mut self, who: &str, id: &str, seq: u64) -> bool {
        match self.threads.get_mut(id) {
            Some(t) if t.has(who) => {
                t.read_through(who, seq);
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn who(id: &str, name: &str) -> Who {
        Who::new(id, name)
    }

    /// **A person and a group of the same name are two threads**, found by
    /// kind and by id.
    #[test]
    fn threads_are_found_by_kind_name_and_id() {
        let mut p = Threads::new();
        let direct = p.reach(&who("npc-1", "Zen"), &who("u_8", "Wren"));
        let group = p.open_group(
            "Wren",
            &[
                who("npc-1", "Zen"),
                who("u_8", "Wren"),
                who("npc-2", "Wick"),
            ],
        );
        assert_eq!(
            p.by_kind_for("npc-1", Kind::Direct, "wren")
                .map(|t| t.id.clone()),
            Some(direct.clone())
        );
        assert_eq!(
            p.by_kind_for("npc-1", Kind::Group, "Wren")
                .map(|t| t.id.clone()),
            Some(group.clone())
        );
        assert_eq!(
            p.by_name_for("npc-1", &direct).map(|t| t.id.clone()),
            Some(direct)
        );
        assert_eq!(p.groups_of("npc-1").len(), 1);
    }

    /// **What is sent reaches everybody else on the thread**, by id, and a
    /// thread its sender is not on refuses it.
    #[test]
    fn what_is_sent_reaches_the_others_and_a_stranger_is_refused() {
        let mut p = Threads::new();
        let g = p.open_group(
            "the sweep",
            &[
                who("npc-1", "Wren"),
                who("npc-2", "Soren"),
                who("npc-3", "Orion"),
            ],
        );
        let (seq, reached) = p.send("npc-1", "the sweep", Said::words("moving")).unwrap();
        assert_eq!(seq, 1);
        assert_eq!(reached, ["npc-2", "npc-3"]);
        assert!(p
            .send_on(&g, "npc-9", Said::words("x"))
            .unwrap_err()
            .contains("no open conversation"));
        assert!(p
            .send("npc-1", "nobody", Said::words("x"))
            .unwrap_err()
            .contains("no conversation"));
    }

    /// **A name the world says again is the name shown everywhere**; the id
    /// does not change.
    #[test]
    fn enrolling_refreshes_a_members_name_on_every_thread() {
        let mut p = Threads::new();
        p.reach(&who("npc-1", "Zen"), &who("u_8", "Mira"));
        p.join_channel(CHANNEL, &who("u_8", "Mira"));
        p.enrol("u_8", "Mira K.");
        assert_eq!(p.names_of_kind_for("npc-1", Kind::Direct), ["Mira K."]);
        assert!(p
            .iter()
            .all(|t| t.member("u_8").is_none_or(|m| m.name == "Mira K.")));
        assert_eq!(p.name_of("u_8"), Some("Mira K."));
    }

    #[test]
    fn what_a_world_holds_is_the_same_across_two_builds() {
        let build = || {
            let mut p = Threads::new();
            p.reach(&who("npc-1", "Wren"), &who("npc-2", "Soren"));
            p.open_group("the sweep", &[who("npc-1", "Wren"), who("npc-3", "Orion")]);
            p
        };
        assert_eq!(build(), build());
    }
}
