//! Who a conversation is with, and the timeline it is kept on
//! (`docs/npc_conversations.md` §4.1).
//!
//! A conversation is keyed by `(npc, party)` and lives on the timeline
//! `npc-{npc}-talk-{party}-g{generation}`. The generation moves on only when
//! the frame it was decoded under changes; the older generations are retired
//! by prefix, the way the mind's day timelines are.

use serde::{Deserialize, Serialize};

/// The prefix a person's body carries in every world: `visitor:{handle}`.
pub const VISITOR: &str = "visitor:";

/// Who a character's conversation is with.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum Party {
    /// A person, by the public handle of their account (`u_…`).
    Person(String),
    /// Another character, by its npc id.
    Npc(u64),
}

impl Party {
    /// The party a thread member id names: a body a mind is bound to is that
    /// character, and anything else on a direct thread is a person's handle.
    /// `mind_of` is the world's bindings (`Bindings::mind_of`).
    pub fn of_member(member: &str, mind_of: impl Fn(&str) -> Option<u64>) -> Party {
        match mind_of(member) {
            Some(npc) => Party::Npc(npc),
            None => Party::Person(member.to_string()),
        }
    }

    /// The party a body standing in a room is, when it is a person's: a
    /// `visitor:{handle}` body is the person with that account (§12).
    pub fn of_visitor(body: &str) -> Option<Party> {
        body.strip_prefix(VISITOR)
            .filter(|h| !h.is_empty())
            .map(|h| Party::Person(h.to_string()))
    }

    /// Whether the party is a person.
    pub fn is_person(&self) -> bool {
        matches!(self, Party::Person(_))
    }

    /// The party as a timeline id names it: `{handle}` or `npc{id}`. Never
    /// contains `-g`, so a generation suffix cannot be mistaken for it.
    fn slug(&self) -> String {
        match self {
            Party::Person(handle) => handle.replace('-', "_"),
            Party::Npc(npc) => format!("npc{npc}"),
        }
    }
}

/// Every conversation timeline of `npc`, whoever it is with:
/// `npc-{npc}-talk-`. What `retire_npc` tombstones.
pub fn all_of(npc: u64) -> String {
    format!("npc-{npc}-talk-")
}

/// Every generation of the conversation between `npc` and `party`:
/// `npc-{npc}-talk-{party}-g`.
pub fn generations_of(npc: u64, party: &Party) -> String {
    format!("{}{}-g", all_of(npc), party.slug())
}

/// The timeline of one generation of the conversation.
pub fn timeline(npc: u64, party: &Party, generation: u32) -> String {
    format!("{}{generation}", generations_of(npc, party))
}

/// The generation a timeline id names, when it is one of this conversation's.
pub fn generation_of(npc: u64, party: &Party, timeline: &str) -> Option<u32> {
    timeline
        .strip_prefix(&generations_of(npc, party))?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_conversation_names_its_timeline_by_party_and_generation() {
        let mira = Party::Person("u_8812".into());
        assert_eq!(timeline(7, &mira, 0), "npc-7-talk-u_8812-g0");
        assert_eq!(timeline(7, &mira, 3), "npc-7-talk-u_8812-g3");
        assert_eq!(timeline(7, &Party::Npc(12), 1), "npc-7-talk-npc12-g1");
        assert_eq!(generations_of(7, &mira), "npc-7-talk-u_8812-g");
        assert_eq!(all_of(7), "npc-7-talk-");
    }

    #[test]
    fn a_handle_with_a_dash_cannot_forge_a_generation() {
        let odd = Party::Person("u-g9".into());
        assert_eq!(timeline(7, &odd, 2), "npc-7-talk-u_g9-g2");
        assert_eq!(generation_of(7, &odd, "npc-7-talk-u_g9-g2"), Some(2));
    }

    #[test]
    fn a_generation_is_read_back_only_from_its_own_conversation() {
        let mira = Party::Person("u_8812".into());
        assert_eq!(generation_of(7, &mira, "npc-7-talk-u_8812-g4"), Some(4));
        assert_eq!(generation_of(7, &mira, "npc-8-talk-u_8812-g4"), None);
        assert_eq!(generation_of(7, &mira, "npc-7-talk-npc12-g4"), None);
        assert_eq!(generation_of(7, &mira, "npc-7-day-3"), None);
    }

    #[test]
    fn a_member_id_names_its_party() {
        let minds = |body: &str| (body == "zen").then_some(12);
        assert_eq!(Party::of_member("zen", minds), Party::Npc(12));
        assert_eq!(
            Party::of_member("u_8812", minds),
            Party::Person("u_8812".into())
        );
    }

    #[test]
    fn a_visitor_body_is_the_person_of_its_handle() {
        assert_eq!(
            Party::of_visitor("visitor:u_8812"),
            Some(Party::Person("u_8812".into()))
        );
        assert_eq!(Party::of_visitor("visitor:"), None);
        assert_eq!(Party::of_visitor("npc-12"), None);
    }

    #[test]
    fn a_party_round_trips_through_its_record() {
        let wire = serde_json::to_string(&Party::Person("u_8812".into())).unwrap();
        assert_eq!(wire, "{\"kind\":\"person\",\"id\":\"u_8812\"}");
        let wire = serde_json::to_string(&Party::Npc(12)).unwrap();
        assert_eq!(wire, "{\"kind\":\"npc\",\"id\":12}");
        assert_eq!(serde_json::from_str::<Party>(&wire).unwrap(), Party::Npc(12));
    }
}
