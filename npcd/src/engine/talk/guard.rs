//! A conversation's own guards (`docs/npc_conversations.md` §8): they are
//! per conversation and never touch the character's.
//!
//! | Guard | Rule |
//! |---|---|
//! | Repeats | words that say again what this thread's last four lines of the character's said go to the journal instead: nothing is struck |
//! | Reply cooldown | NPC↔NPC only, six seconds per party; people are never made to wait |
//! | Activation cap | NPC↔NPC only, [`TALK_TURNS`] turns per side, then the journal or a goodbye |
//! | Unprompted contact | a purpose for a party who has not answered the character's last line waits until they do, or an hour passes |

use crate::engine::loopguard::would_repeat;
use crate::engine::talk::key::Party;

/// Turns per side in one NPC↔NPC activation.
pub const TALK_TURNS: u32 = 8;

/// How long an NPC↔NPC reply cools, per party.
pub const REPLY_COOLS_MS: u64 = 6_000;

/// How long a purpose waits for a party who has not answered.
pub const UNANSWERED_HOLD_MS: u64 = 3_600_000;

/// Whether this activation has run its turns.
pub fn capped(party: &Party, activation: u32) -> bool {
    !party.is_person() && activation > TALK_TURNS
}

/// The acts this conversation has struck: `reply`, for an NPC party, inside
/// six seconds of the character's last line to them.
pub fn cooling(party: &Party, last_said_ms: Option<u64>, now_ms: u64) -> Vec<String> {
    match (party.is_person(), last_said_ms) {
        (false, Some(said)) if now_ms < said + REPLY_COOLS_MS => vec!["reply".to_string()],
        _ => Vec::new(),
    }
}

/// Whether a purpose for the party waits: the thread's last line is the
/// character's own, still unanswered, said less than an hour ago.
pub fn holds_purpose(last_was_mine: bool, last_at_ms: u64, now_ms: u64) -> bool {
    last_was_mine && now_ms < last_at_ms + UNANSWERED_HOLD_MS
}

/// What a conversation does with words it is about to send to `party`, given
/// its own last lines on their thread: send them, or — when they say again
/// what was said — write the attempt down instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Words {
    Send(String),
    Journal(String),
}

pub fn words_for(party_name: &str, own_last: &[&str], text: &str) -> Words {
    match would_repeat(own_last, text) {
        false => Words::Send(text.to_string()),
        true => Words::Journal(format!(
            "Started to tell {party_name} again that {}",
            text.trim()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mira() -> Party {
        Party::Person("u_mira".into())
    }

    #[test]
    fn only_an_npc_exchange_is_capped_and_only_past_its_turns() {
        assert!(!capped(&Party::Npc(2), TALK_TURNS));
        assert!(capped(&Party::Npc(2), TALK_TURNS + 1));
        assert!(!capped(&mira(), 100));
    }

    #[test]
    fn a_person_is_never_made_to_wait_for_a_reply() {
        assert!(cooling(&mira(), Some(1_000), 1_001).is_empty());
        assert_eq!(cooling(&Party::Npc(2), Some(1_000), 6_999), ["reply"]);
        assert!(cooling(&Party::Npc(2), Some(1_000), 7_000).is_empty());
        assert!(cooling(&Party::Npc(2), None, 0).is_empty());
    }

    #[test]
    fn a_purpose_waits_an_hour_for_a_party_who_has_not_answered() {
        assert!(holds_purpose(true, 0, UNANSWERED_HOLD_MS - 1));
        assert!(!holds_purpose(true, 0, UNANSWERED_HOLD_MS));
        assert!(!holds_purpose(false, 0, 1));
    }

    #[test]
    fn telling_the_party_the_same_again_is_written_down_instead() {
        let parcel = "The parcel came this morning and is waiting at the front door beside the \
                      green bins.";
        assert_eq!(
            words_for("Mira", &[parcel], parcel),
            Words::Journal(format!("Started to tell Mira again that {parcel}"))
        );
        assert_eq!(
            words_for("Mira", &[parcel], "The garden report is done."),
            Words::Send("The garden report is done.".into())
        );
    }
}
