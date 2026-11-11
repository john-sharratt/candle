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

use std::collections::BTreeMap;

use npc_map::MapEdit;
use serde::{Deserialize, Serialize};

use crate::discord::world::record::Guild;
use crate::engine::mission::{Mission, StepOutcome};
use crate::engine::runtime::MessageSent;
use crate::sim::phone::{Origin, Who};

/// Where an act came from (`docs/npc_conversations.md` §7.1): the
/// character's mind loop, or one of its conversations, with the party.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum ActSource {
    #[default]
    Mind,
    /// A conversation with `party`: a body id or an account handle.
    Talk { party: String },
}

/// The line the lease holder voiced for a phone act
/// (`docs/npc_conversations.md` §7.1), and the critical step it asks an
/// authorizer to approve, when it asks for one (§7.2): what lands on the
/// thread in place of the act's intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Voiced {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests: Option<String>,
}

impl Voiced {
    /// Words that ask for nothing.
    pub fn words(text: &str) -> Voiced {
        Voiced {
            text: text.to_string(),
            requests: None,
        }
    }
}

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
        /// The line the lease holder voiced for a phone act: what lands on
        /// the thread in place of the act's intent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        said: Option<Voiced>,
        #[serde(default)]
        source: ActSource,
    },
    /// A person's message to a character, on its handset — from the console,
    /// with no `origin`, or from where `origin` says. `from` is the person
    /// by their account.
    Message {
        npc: u64,
        from: Who,
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
    /// An authorizer said yes to the critical step a conversation asked them
    /// to sign (`docs/npc_conversations.md` §7.2): signed when `step` is
    /// still the mission's next and `by` — an account handle — is one of the
    /// world's authorizers, both checked here, on the authority.
    Authorize {
        npc: u64,
        step: String,
        by: String,
        /// Who said yes, by name, for what the character is told.
        name: String,
    },
    /// `npc` read `thread` through `seq`: a conversation that answered what
    /// it took in with a journal entry rather than a line
    /// (`docs/npc_conversations.md` §8), so the party's typing indicator —
    /// the authority's — stops.
    ReadThrough { npc: u64, thread: String, seq: u64 },
    /// An operator reshapes the world's walkable map, through the token of
    /// character `npc`, whose owner keeps the change
    /// (`docs/npc_conversations.md` §5.1.1): applied by the authority, so it
    /// is journaled once, wherever `/reshape` was asked.
    Reshape { npc: u64, edit: MapEdit },
    /// A person speaks on the world's open channel — from the console, with
    /// no `origin`, or from the world's Discord server's `#the-channel`.
    Channel {
        from: Who,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<Origin>,
    },
    /// The world's Discord server set, or taken away (`docs/npcd_discord.md`
    /// §4.4): its `discord/guild` record.
    Guild { guild: Option<Guild> },
    /// Somebody wrote in the channel of the world's server that mirrors
    /// thread `thread` — a group they are on, or a channel a person made:
    /// said on the thread, from `origin`, its author brought onto the thread
    /// at its end on their first message there. `to` is the character it was
    /// said to, when it mentions, answers or names one.
    Mirrored {
        thread: String,
        from: Who,
        text: String,
        origin: Origin,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        to: Option<String>,
    },
    /// Somebody wrote in Discord channel `channel` of a server character
    /// `npc`'s bot is in: said on the thread that mirrors it, its author
    /// brought onto the thread on their first message
    /// (`docs/npcd_discord.md` §3.4). `to` is the character, when the
    /// message was said to it — it mentions or answers the bot, or names it.
    ChannelMessage {
        npc: u64,
        channel: u64,
        from: Who,
        text: String,
        origin: Option<Origin>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        to: Option<String>,
    },
    /// Something the character is told happened to it — its bot added to a
    /// Discord server, or removed from one — delivered as a description that
    /// wakes it.
    Notice { npc: u64, text: String },
    /// A person leaves words on something in the world.
    Posting {
        at: String,
        on: String,
        by: String,
        text: String,
    },
    /// A character's holder hands it back at a turn boundary, its own writes
    /// reaching `through`, for its next holder to wait for along with every
    /// wait the lease already names.
    HandBack { npc: u64, through: u64 },
    /// An admin frees a character's lease, for a holder that will not return.
    Release { npc: u64 },
    /// Notes or credentials set on a device part, named by its network address
    /// (`docs/zen_moon_mind.md` §15): what the world keeps to describe and work
    /// it. `notes` replaces when given; `creds` are merged in.
    AmendDevice {
        address: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        notes: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        creds: BTreeMap<String, String>,
    },
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
    /// it was carried at once — or `refused`, for a character that takes no
    /// orders, with nothing lodged. Absent, `refused` is a lodge taken.
    Lodged {
        now_text: Option<String>,
        #[serde(default)]
        refused: bool,
    },
    /// A step ticked off and one added, as asked.
    Stepped {
        ticked: Option<bool>,
        added: Option<bool>,
    },
    /// Who heard what was said on the channel; `None` with no such world.
    Heard { by: Option<Vec<String>> },
    /// How many lines the surface holds now; `None` with no such place.
    Posted { lines: Option<usize> },
    /// A world's map reshaped: who it relocated, and whether the change was
    /// kept — `written`, `drafted`, `ephemeral` or `failed`, with `detail`
    /// saying more — or `refused`, the world's own words on why it would not
    /// take the edit, with nothing changed.
    Reshaped {
        #[serde(default)]
        relocated: Vec<String>,
        #[serde(default)]
        durable: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refused: Option<String>,
    },
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
            said: None,
            source: ActSource::Mind,
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
                "source": {"from": "mind"},
            })
        );
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), op);
        let voiced = WorldOp::Act {
            npc: 7,
            call: r#"{"tool":"reply","to":"Mira","intent":"that it is in"}"#.into(),
            decided: None,
            pinned_at: None,
            said: Some(Voiced {
                text: "It's at the door.".into(),
                requests: Some("Sign the lease".into()),
            }),
            source: ActSource::Talk {
                party: "u_8812".into(),
            },
        };
        let wire = serde_json::to_value(&voiced).unwrap();
        assert_eq!(
            wire["said"],
            json!({"text": "It's at the door.", "requests": "Sign the lease"})
        );
        assert_eq!(
            serde_json::to_value(Voiced::words("Hi.")).unwrap(),
            json!({"text": "Hi."})
        );
        assert_eq!(wire["source"], json!({"from": "talk", "party": "u_8812"}));
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), voiced);
        assert_eq!(
            serde_json::to_value(OpReply::Done { ok: true }).unwrap(),
            json!({"reply": "done", "ok": true})
        );
        let message = WorldOp::Message {
            npc: 9,
            from: Who::new("u_8812", "Wren S"),
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
                "from": {"id": "u_8812", "name": "Wren S"},
                "text": "hello",
                "origin": {"kind": "discord", "message": 31, "channel": 32},
            })
        );
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), message);
        let back = WorldOp::HandBack {
            npc: 3,
            through: 40,
        };
        let wire = json!({"op": "hand_back", "npc": 3, "through": 40});
        assert_eq!(serde_json::to_value(&back).unwrap(), wire);
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), back);
    }

    /// **A server channel's message and a notice travel as these**, exactly.
    #[test]
    fn a_channel_message_and_a_notice_travel_as_tagged_json() {
        let wren = Who::new("discord:31", "Wren (@wren)");
        let said = WorldOp::ChannelMessage {
            npc: 9,
            channel: 610,
            from: wren.clone(),
            text: "hi all".into(),
            origin: Some(Origin::Discord {
                message: 901,
                channel: 610,
            }),
            to: None,
        };
        let wire = serde_json::to_value(&said).unwrap();
        assert_eq!(
            wire,
            json!({
                "op": "channel_message",
                "npc": 9,
                "channel": 610,
                "from": {"id": "discord:31", "name": "Wren (@wren)"},
                "text": "hi all",
                "origin": {"kind": "discord", "message": 901, "channel": 610},
            })
        );
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), said);
        let said_to = WorldOp::ChannelMessage {
            npc: 9,
            channel: 610,
            from: wren,
            text: "Zen, you there?".into(),
            origin: None,
            to: Some("npc-9".into()),
        };
        let wire = serde_json::to_value(&said_to).unwrap();
        assert_eq!(
            wire,
            json!({
                "op": "channel_message",
                "npc": 9,
                "channel": 610,
                "from": {"id": "discord:31", "name": "Wren (@wren)"},
                "text": "Zen, you there?",
                "origin": null,
                "to": "npc-9",
            })
        );
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), said_to);
        let told = WorldOp::Notice {
            npc: 9,
            text: "You were removed from the Discord server zenworld.".into(),
        };
        let wire = serde_json::to_value(&told).unwrap();
        assert_eq!(
            wire,
            json!({
                "op": "notice",
                "npc": 9,
                "text": "You were removed from the Discord server zenworld.",
            })
        );
        assert_eq!(serde_json::from_value::<WorldOp>(wire).unwrap(), told);
    }
}
