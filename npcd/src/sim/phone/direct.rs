//! Direct threads (`docs/npc_conversations.md` §11.1): one per pair, for
//! ever.
//!
//! A direct thread's id is derived from its two members' ids, sorted, so
//! reaching somebody again — from either side, after any restart, on any
//! member of the cluster — finds the same thread: its seq never restarts, it
//! is never evicted, and it is never closed by its members. `sign_off` ends a
//! *conversation*; the thread it was held on stays.

use super::thread::{Kind, Thread};
use super::{Threads, Who};

/// The id of the direct thread between `a` and `b`, whichever asks.
pub fn direct_id(a: &str, b: &str) -> String {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    format!("dm:{lo}:{hi}")
}

/// The other member of direct thread `id`, to `me`: read off the id itself,
/// so a member whose copy of the world has not caught up with the thread
/// still knows who it is with (`docs/npc_conversations.md` §9). A member id
/// may hold a colon (`discord:{id}`), so the known side is matched whole.
pub fn other_of(id: &str, me: &str) -> Option<String> {
    let pair = id.strip_prefix("dm:")?;
    let other = match (pair.strip_prefix(&format!("{me}:")), pair.strip_suffix(&format!(":{me}"))) {
        (Some(other), _) | (None, Some(other)) => other,
        (None, None) => return None,
    };
    (direct_id(me, other) == id).then(|| other.to_string())
}

impl Threads {
    /// The direct thread between `a` and `b`, if there is one.
    pub fn direct_between(&self, a: &str, b: &str) -> Option<&Thread> {
        self.get(&direct_id(a, b))
    }

    /// The direct thread between `from` and `to`: the one that stands, or a
    /// new one. Their names are refreshed either way.
    pub fn reach(&mut self, from: &Who, to: &Who) -> String {
        let id = direct_id(&from.id, &to.id);
        if !self.threads.contains_key(&id) {
            let touched = self.tick();
            let mut t = Thread::new(id.clone(), "", Kind::Direct, Vec::new());
            t.add(&from.id, &from.name);
            t.add(&to.id, &to.name);
            t.touched = touched;
            self.threads.insert(id.clone(), t);
        }
        for who in [from, to] {
            if let Some(m) = self.threads.get_mut(&id).and_then(|t| t.member_mut(&who.id)) {
                m.name = who.name.clone();
            }
        }
        id
    }

    /// The direct threads `who` is on.
    pub fn direct_of(&self, who: &str) -> Vec<&Thread> {
        self.threads
            .values()
            .filter(|t| t.kind() == Kind::Direct && t.has(who))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::phone::Said;

    fn who(id: &str, name: &str) -> Who {
        Who::new(id, name)
    }

    /// **One thread per pair, whichever side reaches and however often**,
    /// named by sorting the pair.
    #[test]
    fn a_pair_has_one_thread_found_from_either_side() {
        let mut p = Threads::new();
        let zen = who("npc-1", "Zen");
        let mira = who("u_8", "Mira");
        let a = p.reach(&zen, &mira);
        assert_eq!(a, "dm:npc-1:u_8");
        assert_eq!(p.reach(&mira, &zen), a);
        assert_eq!(p.len(), 1);
        assert_eq!(direct_id("u_8", "npc-1"), a);
    }

    /// **Its seq never restarts**: reached again after a hundred lines, it is
    /// the same thread with the same count.
    #[test]
    fn reaching_again_continues_the_seq() {
        let mut p = Threads::new();
        let zen = who("npc-1", "Zen");
        let mira = who("u_8", "Mira");
        let id = p.reach(&zen, &mira);
        for i in 0..100 {
            p.send_on(&id, "u_8", Said::words(&format!("{i}"))).unwrap();
        }
        assert_eq!(p.reach(&zen, &mira), id);
        assert_eq!(p.send_on(&id, "npc-1", Said::words("hi")).unwrap().0, 101);
    }

    /// **The other member is read off the id**, a colon in either id included.
    #[test]
    fn the_other_member_is_read_off_the_thread_id() {
        assert_eq!(other_of("dm:npc-1:u_8", "npc-1").as_deref(), Some("u_8"));
        assert_eq!(other_of("dm:npc-1:u_8", "u_8").as_deref(), Some("npc-1"));
        let id = direct_id("discord:42", "npc-5");
        assert_eq!(id, "dm:discord:42:npc-5");
        assert_eq!(other_of(&id, "npc-5").as_deref(), Some("discord:42"));
        assert_eq!(other_of(&id, "discord:42").as_deref(), Some("npc-5"));
        assert_eq!(other_of("dm:npc-1:u_8", "npc-2"), None);
        assert_eq!(other_of("group:7", "npc-1"), None);
    }

    /// **A name is refreshed when the pair is reached again** — the id is who
    /// they are, the name only what they are shown as.
    #[test]
    fn reaching_again_refreshes_the_names() {
        let mut p = Threads::new();
        let id = p.reach(&who("npc-1", "Zen"), &who("u_8", "Mira"));
        p.reach(&who("npc-1", "Zen"), &who("u_8", "Mira K."));
        assert_eq!(p.get(&id).unwrap().as_named_to("npc-1"), "Mira K.");
    }
}
