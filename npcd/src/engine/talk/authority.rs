//! Who may change a character's mission through a conversation
//! (`docs/npc_conversations.md` §7.2): its owner, and its world's authorizers,
//! known by account handle — never by a name, which an identity provider can
//! change and two people can share.
//!
//! An NPC party is never an authority over another character's mission.
//! Everyone else may get a promise, and the character's main loop decides.
//!
//! **The thrash guard.** A mission changes through a conversation at most
//! once per [`CHANGE_COOLDOWN_MS`] of world time, except by its owner.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::engine::authorize::is_authorizer;
use crate::engine::talk::key::Party;

/// When each character's mission last changed through a conversation, in
/// world time: what the thrash guard reads.
#[derive(Default)]
pub struct Changes {
    at: Mutex<HashMap<u64, u64>>,
}

impl Changes {
    pub fn new() -> Changes {
        Changes::default()
    }

    pub fn last(&self, npc: u64) -> Option<u64> {
        self.at.lock().unwrap_or_else(|e| e.into_inner()).get(&npc).copied()
    }

    pub fn mark(&self, npc: u64, world_ms: u64) {
        self.at
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(npc, world_ms);
    }
}

/// How long, in world time, a mission changed through a conversation stays as
/// it was before another conversation may change it.
pub const CHANGE_COOLDOWN_MS: u64 = 600_000;

/// Who the party is to the character's mission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Standing {
    Owner,
    Authorizer,
    Nobody,
}

/// `party`'s standing over the character owned by `owner`, in a world whose
/// authorizers are `authorizers` (account handles).
pub fn standing(party: &Party, owner: Option<&str>, authorizers: &[String]) -> Standing {
    let Party::Person(handle) = party else {
        return Standing::Nobody;
    };
    if owner == Some(handle.as_str()) {
        return Standing::Owner;
    }
    match is_authorizer(handle, authorizers) {
        true => Standing::Authorizer,
        false => Standing::Nobody,
    }
}

/// Whether `party` may change the mission now, the last change through a
/// conversation having landed at `last_changed_ms` of world time.
pub fn may_change(standing: Standing, last_changed_ms: Option<u64>, now_ms: u64) -> bool {
    match standing {
        Standing::Owner => true,
        Standing::Authorizer => {
            last_changed_ms.is_none_or(|at| now_ms >= at + CHANGE_COOLDOWN_MS)
        }
        Standing::Nobody => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn john() -> Party {
        Party::Person("u_john0001".into())
    }

    #[test]
    fn the_owner_and_an_authorizer_have_standing_and_nobody_else_does() {
        let authorizers = ["u_mira0002".to_string()];
        assert_eq!(standing(&john(), Some("u_john0001"), &authorizers), Standing::Owner);
        assert_eq!(
            standing(&Party::Person("u_mira0002".into()), Some("u_john0001"), &authorizers),
            Standing::Authorizer
        );
        assert_eq!(
            standing(&Party::Person("u_stranger".into()), Some("u_john0001"), &authorizers),
            Standing::Nobody
        );
        assert_eq!(standing(&Party::Npc(7), None, &authorizers), Standing::Nobody);
    }

    #[test]
    fn an_authorizer_waits_out_the_cooldown_and_the_owner_never_does() {
        assert!(may_change(Standing::Authorizer, None, 0));
        assert!(!may_change(Standing::Authorizer, Some(1_000), 1_000 + CHANGE_COOLDOWN_MS - 1));
        assert!(may_change(Standing::Authorizer, Some(1_000), 1_000 + CHANGE_COOLDOWN_MS));
        assert!(may_change(Standing::Owner, Some(1_000), 1_001));
        assert!(!may_change(Standing::Nobody, None, 0));
    }
}
