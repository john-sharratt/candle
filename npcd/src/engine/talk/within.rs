//! What a conversation's turn can do (`docs/npc_conversations.md` §6.2): the
//! thread's own catalog, compiled from the character's `Within` with every
//! world set emptied and an explicit allow-list over what is left.
//!
//! The party is the conversation's whole company: `reply`, `sign_off`,
//! `promise` and `remind` bind them and nobody else, and a hand-off through
//! `message` or `reach_out` goes to anybody on the phone **but** them. Its
//! cooling is its own — a person is never made to wait on the character's
//! cooldowns, and the character's are never stamped by a conversation (§8).

use crate::engine::talk::key::Party;
use crate::engine::tools::Within;

/// Every act a conversation may take, in the order the decision reads them.
pub const ACTS: &[&str] = &[
    "reply",
    "sign_off",
    "reflect",
    "promise",
    "remind",
    "message",
    "reach_out",
    "mission_add",
    "mission_edit",
    "mission_cancel",
    "mission_done",
];

/// The acts that change the character's mission: offered only to a party with
/// authority over it, outside the thrash cooldown (§7.2).
pub const MISSION_ACTS: &[&str] = &[
    "mission_add",
    "mission_edit",
    "mission_cancel",
    "mission_done",
];

/// The acts that keep a conversation talking, struck once an NPC↔NPC
/// activation has run its turns (§8): what is left is the journal or a
/// goodbye.
pub const TALKING_ACTS: &[&str] = &["reply", "message"];

/// The conversation a turn belongs to, as the turn's catalog needs it.
#[derive(Clone, Debug)]
pub struct OnThread<'a> {
    pub party: &'a Party,
    /// The party as the character names them on the phone — the direct
    /// thread's name, and what every party-bound argument takes.
    pub name: &'a str,
    /// What the party owes the character, by what was promised.
    pub owed: Vec<String>,
    /// Whether the party may change the character's mission: its owner or an
    /// authorizer, outside the thrash cooldown (§7.2).
    pub may_change_mission: bool,
    /// The brief of the mission a character that takes orders carries — what
    /// its ordered mission acts name (`engine::agenda::ordered`). `None` for
    /// one carrying nothing, or keeping an agenda.
    pub carried: Option<String>,
    /// Whether this NPC↔NPC activation has run its turns (§8).
    pub capped: bool,
    /// The conversation's own struck acts: the NPC↔NPC reply cooldown.
    pub cooling: Vec<String>,
}

impl Within {
    /// This character's catalog for a turn of its conversation with
    /// `thread.party`, built from where it stands.
    pub fn on_thread(&self, thread: &OnThread) -> Within {
        let party = thread.name.to_string();
        let others = |names: &[String]| -> Vec<String> {
            names.iter().filter(|n| **n != party).cloned().collect()
        };
        let allow: Vec<String> = ACTS
            .iter()
            .filter(|act| **act != "remind" || !thread.party.is_person())
            .filter(|act| thread.may_change_mission || !MISSION_ACTS.contains(act))
            .filter(|act| !thread.capped || !TALKING_ACTS.contains(act))
            .map(|act| act.to_string())
            .collect();
        // A character keeping an agenda is offered priorities; one without
        // them takes orders, and its mission acts are the ordered ones, over
        // the one mission it carries.
        let orders = thread.may_change_mission && self.priorities.is_empty();
        let (agenda, priorities) = match (thread.may_change_mission, orders) {
            (false, _) => (Vec::new(), Vec::new()),
            (true, false) => (self.agenda.clone(), self.priorities.clone()),
            (true, true) => (thread.carried.iter().cloned().collect(), Vec::new()),
        };
        Within {
            me: self.me.clone(),
            feelings: self.feelings.clone(),
            company: vec![party.clone()],
            // On a thread the party is who hears what is said, so they stand
            // where the room's people would for the speech gate: `promise` and
            // `remind` bind them.
            people: vec![party.clone()],
            awaiting: vec![party.clone()],
            leavable: vec![party.clone()],
            owed: thread.owed.clone(),
            private: others(&self.private),
            channels: self.channels.clone(),
            threads: others(&self.threads),
            contacts: others(&self.contacts),
            agenda,
            priorities,
            orders,
            cooling: thread.cooling.clone(),
            allow: Some(allow),
            ..Within::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::tools::{specs_within, Catalog};
    use crate::sim::agenda::Priority;

    /// Zen, standing in a room with somebody in it, on the phone to Mira and
    /// Wick, keeping an agenda.
    fn zen() -> Within {
        let mut within = Within::among(&["Orion Vance"]);
        within.me = "Zen".into();
        within.feelings = vec!["calm".into(), "wry".into()];
        within.threads = vec!["Mira".into(), "Wick".into(), "the channel".into()];
        within.private = vec!["Mira".into(), "Wick".into()];
        within.channels = vec!["the channel".into()];
        within.contacts = vec!["Soren".into()];
        within.carried = vec!["a torch".into()];
        within.readable = vec!["the muster board".into()];
        within.agenda = vec!["Tidy the lounge".into()];
        within.priorities = Priority::ALL
            .iter()
            .map(|p| p.as_str().to_string())
            .collect();
        within
    }

    /// The acts offered, in [`ACTS`]'s order: which acts is the question,
    /// not the catalog's order.
    fn offered(within: &Within) -> Vec<String> {
        let specs = specs_within(Catalog::THREAD.mode, within);
        ACTS.iter()
            .filter(|a| specs.iter().any(|s| s.name == **a))
            .map(|a| a.to_string())
            .collect()
    }

    fn mira() -> Party {
        Party::Person("u_mira".into())
    }

    fn thread<'a>(party: &'a Party, name: &'a str) -> OnThread<'a> {
        OnThread {
            party,
            name,
            owed: Vec::new(),
            may_change_mission: false,
            carried: None,
            capped: false,
            cooling: Vec::new(),
        }
    }

    #[test]
    fn a_stranger_is_offered_talk_and_hand_offs_and_nothing_of_the_room() {
        let party = mira();
        let within = zen().on_thread(&thread(&party, "Mira"));
        assert_eq!(
            offered(&within),
            [
                "reply",
                "sign_off",
                "reflect",
                "promise",
                "message",
                "reach_out"
            ]
            .map(String::from)
        );
    }

    #[test]
    fn the_party_is_the_whole_company_and_never_a_hand_off() {
        let party = mira();
        let within = zen().on_thread(&thread(&party, "Mira"));
        assert_eq!(within.company, ["Mira"]);
        assert_eq!(within.awaiting, ["Mira"]);
        assert_eq!(within.leavable, ["Mira"]);
        assert_eq!(within.private, ["Wick"]);
        assert_eq!(within.threads, ["Wick", "the channel"]);
        assert_eq!(within.channels, ["the channel"]);
        assert_eq!(within.contacts, ["Soren"]);
        assert!(within.carried.is_empty() && within.readable.is_empty());
    }

    #[test]
    fn an_authority_over_the_mission_is_offered_the_mission_acts() {
        let party = mira();
        let within = zen().on_thread(&OnThread {
            may_change_mission: true,
            ..thread(&party, "Mira")
        });
        assert_eq!(
            offered(&within),
            [
                "reply",
                "sign_off",
                "reflect",
                "promise",
                "message",
                "reach_out",
                "mission_add",
                "mission_edit",
                "mission_cancel",
                "mission_done",
            ]
            .map(String::from)
        );
    }

    /// **An ordered body is offered the ordered acts**: the agenda's names
    /// over the one mission it carries, with no title and no priority.
    #[test]
    fn an_ordered_body_is_offered_its_own_mission_acts_over_what_it_carries() {
        let party = mira();
        let mut wick = zen();
        wick.agenda.clear();
        wick.priorities.clear();
        let within = wick.on_thread(&OnThread {
            may_change_mission: true,
            carried: Some("Keep the lounge tidy.".into()),
            ..thread(&party, "Mira")
        });
        assert!(within.orders);
        assert_eq!(within.agenda, ["Keep the lounge tidy."]);
        assert_eq!(
            offered(&within),
            [
                "reply",
                "sign_off",
                "reflect",
                "promise",
                "message",
                "reach_out",
                "mission_add",
                "mission_edit",
                "mission_cancel",
                "mission_done",
            ]
            .map(String::from)
        );
        let specs = specs_within(Catalog::THREAD.mode, &within);
        let params = |name: &str| -> Vec<String> {
            let spec = specs.iter().find(|s| s.name == name).unwrap();
            spec.params.iter().map(|p| p.name.clone()).collect()
        };
        assert_eq!(params("mission_add"), ["brief", "steps"]);
        assert_eq!(params("mission_edit"), ["mission", "brief", "steps"]);
        let edit = specs.iter().find(|s| s.name == "mission_edit").unwrap();
        assert_eq!(
            edit.params[0].enum_values,
            Some(vec!["Keep the lounge tidy.".to_string()])
        );
    }

    /// Carrying nothing, it can only be given a mission.
    #[test]
    fn an_ordered_body_carrying_nothing_can_only_be_given_one() {
        let party = mira();
        let mut wick = zen();
        wick.agenda.clear();
        wick.priorities.clear();
        let within = wick.on_thread(&OnThread {
            may_change_mission: true,
            ..thread(&party, "Mira")
        });
        let missions: Vec<String> = offered(&within)
            .into_iter()
            .filter(|a| a.starts_with("mission_"))
            .collect();
        assert_eq!(missions, ["mission_add"]);
    }

    /// Without standing, an ordered body is offered no mission act at all.
    #[test]
    fn an_ordered_body_offers_a_stranger_no_mission_act() {
        let party = mira();
        let mut wick = zen();
        wick.agenda.clear();
        wick.priorities.clear();
        let within = wick.on_thread(&OnThread {
            carried: Some("Keep the lounge tidy.".into()),
            ..thread(&party, "Mira")
        });
        assert!(!within.orders);
        assert!(!offered(&within).iter().any(|a| a.starts_with("mission_")));
    }

    #[test]
    fn another_character_can_be_reminded_of_what_it_owes() {
        let party = Party::Npc(12);
        let within = zen().on_thread(&OnThread {
            owed: vec!["the garden report".into()],
            ..thread(&party, "Wick")
        });
        assert_eq!(
            offered(&within),
            [
                "reply",
                "sign_off",
                "reflect",
                "promise",
                "remind",
                "message",
                "reach_out"
            ]
            .map(String::from)
        );
        assert_eq!(within.private, ["Mira"]);
    }

    #[test]
    fn a_capped_activation_can_only_write_it_down_or_say_goodbye() {
        let party = Party::Npc(12);
        let within = zen().on_thread(&OnThread {
            capped: true,
            ..thread(&party, "Wick")
        });
        assert_eq!(
            offered(&within),
            ["sign_off", "reflect", "promise", "reach_out"].map(String::from)
        );
    }

    #[test]
    fn a_conversation_cools_only_on_its_own_strikes() {
        let party = Party::Npc(12);
        let mut busy = zen();
        busy.cooling = vec!["promise".into()];
        let within = busy.on_thread(&OnThread {
            cooling: vec!["reply".into()],
            ..thread(&party, "Wick")
        });
        let acts = offered(&within);
        assert!(!acts.contains(&"reply".to_string()), "{acts:?}");
        assert!(acts.contains(&"promise".to_string()), "{acts:?}");
    }
}
