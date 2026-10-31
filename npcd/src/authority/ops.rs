//! The writes a member makes to a world it does not run, sent to the world's
//! authority (`docs/replicated_substrate.md` §9.3): each one the authority makes
//! in its own world and answers, as `POST /v1/world/:wid/op`. A member that runs
//! the world — its authority, or one disconnected from it — makes the same
//! write in place, so both run one path.
//!
//! **What a character's turn writes** is two of these: [`WorldOp::Begin`], what
//! its arrival met and what the table hands it, and an [`WorldOp::Act`] for each
//! act that changes the world. What changes only the character's own mind stays
//! where the character is. **What a person or an operator writes** about one
//! character — a message, standing beside it, a mission — is one each.

use serde::{Deserialize, Serialize};

use crate::engine::mission::{Mission, StepOutcome};
use crate::engine::runtime::MessageSent;
use crate::sim::phone::Origin;

/// The decision a character's act was made in: where it stood and who was with
/// it when its turn began, so the authority answers the act from the company
/// it was offered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decided {
    pub at: String,
    pub company: Vec<String>,
}

/// One write to a world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorldOp {
    /// A character's turn begins.
    Begin { npc: u64 },
    /// One act of a character's turn, as the call line it made.
    Act {
        npc: u64,
        call: String,
        decided: Option<Decided>,
        /// Where its body stood when the turn's grammar was built, `area/node`.
        pinned_at: Option<String>,
    },
    /// A person's message to a character, on its handset — from the console,
    /// with no `origin`, or from where `origin` says.
    Message {
        npc: u64,
        from: String,
        text: String,
        #[serde(default)]
        origin: Option<Origin>,
    },
    /// A person comes to stand beside a character.
    EnterBeside {
        npc: u64,
        visitor: String,
        name: String,
    },
    /// A person leaves the world.
    Leave { visitor: String },
    /// A visitor is put beside the character it follows, at `at`.
    Place { visitor: String, at: String },
    /// A mission lodged for a character, or carried at once when `start`.
    Lodge {
        npc: u64,
        mission: Box<Mission>,
        start: bool,
    },
    /// A step of a character's mission ticked off, or one added.
    Step {
        npc: u64,
        done: Option<String>,
        outcome: Option<StepOutcome>,
        add: Option<String>,
    },
    /// A character's mission called off.
    Cancel { npc: u64 },
    /// A person speaks on the world's open channel.
    Channel { from: String, text: String },
    /// A person leaves words on something in the world.
    Posting {
        at: String,
        on: String,
        by: String,
        text: String,
    },
    /// A character's holder hands it back at a turn boundary, its own writes
    /// reaching `through`, for its next holder to wait for.
    HandBack { npc: u64, through: u64 },
    /// An admin frees a character's lease, for a holder that will not return.
    Release { npc: u64 },
}

/// What the world answered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum OpReply {
    /// A turn's opening: what its arrival signed off, the mission compass when
    /// it has news, and what the table at which it stands hands it.
    Began {
        arrival: Option<String>,
        compass: Option<String>,
        summons: Option<String>,
    },
    /// An act's outcome, for the feed and for the character, and whether it
    /// moved the character's mission.
    Performed {
        feed: String,
        answer: String,
        landed: bool,
        departing: bool,
        moved: bool,
    },
    /// A message sent; `None` when the character has no body to reach.
    Sent { sent: Option<MessageSent> },
    /// Where a person now stands; `None` when there was nowhere.
    Placed { at: Option<String> },
    /// A mission lodged: the standing task to rouse the character with, when
    /// it was carried at once.
    Lodged { now_text: Option<String> },
    /// A step ticked off and one added, as asked.
    Stepped {
        ticked: Option<bool>,
        added: Option<bool>,
    },
    /// Who heard what was said on the channel; `None` with no such world.
    Heard { by: Option<Vec<String>> },
    /// How many lines the surface holds now; `None` with no such place.
    Posted { lines: Option<usize> },
    /// Whether the write took.
    Done { ok: bool },
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// An op travels as its name and its fields; a reply as its kind.
    #[test]
    fn an_op_and_its_reply_travel_as_tagged_json() {
        let op = WorldOp::Act {
            npc: 7,
            call: r#"{"tool":"speak","intent":"hello"}"#.into(),
            decided: Some(Decided {
                at: "vault-casting/green-room".into(),
                company: vec!["m2".into()],
            }),
            pinned_at: None,
        };
        let wire = serde_json::to_value(&op).unwrap();
        assert_eq!(
            wire,
            json!({
                "op": "act",
                "npc": 7,
                "call": "{\"tool\":\"speak\",\"intent\":\"hello\"}",
                "decided": {"at": "vault-casting/green-room", "company": ["m2"]},
                "pinned_at": null,
            })
        );
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), op);
        assert_eq!(
            serde_json::to_value(OpReply::Done { ok: true }).unwrap(),
            json!({"reply": "done", "ok": true})
        );
        let message = WorldOp::Message {
            npc: 9,
            from: "Wren S".into(),
            text: "hello".into(),
            origin: Some(Origin::Discord {
                message: 31,
                channel: 32,
            }),
        };
        let wire = serde_json::to_value(&message).unwrap();
        assert_eq!(
            wire,
            json!({
                "op": "message",
                "npc": 9,
                "from": "Wren S",
                "text": "hello",
                "origin": {"kind": "discord", "message": 31, "channel": 32},
            })
        );
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), message);
        assert_eq!(
            serde_json::from_value::<WorldOp>(
                json!({"op": "message", "npc": 9, "from": "Wren S", "text": "hello"})
            )
            .unwrap(),
            WorldOp::Message {
                npc: 9,
                from: "Wren S".into(),
                text: "hello".into(),
                origin: None,
            },
            "a member that sends no origin sends one from the console"
        );
        assert_eq!(
            serde_json::to_value(WorldOp::HandBack {
                npc: 3,
                through: 40
            })
            .unwrap(),
            json!({"op": "hand_back", "npc": 3, "through": 40})
        );
    }
}
