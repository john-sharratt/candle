//! The bodies of the character and channel routes: what a mission, a message
//! and a channel post send and answer. The handlers deserialize into these and
//! build their answers from them, and the MCP tools ([`crate::mcp`]) read
//! those answers back as the same types — one definition for both.
//!
//! A character is named by its wire id, `npc_id` — a base-36 string.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::mission::StepOutcome;

/// What a route answers a refused request with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiError {
    pub error: String,
    #[serde(default)]
    pub detail: String,
}

/// One character, as the roster lists it — the fields a caller names it by and
/// reads its standing from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Npc {
    pub npc_id: String,
    pub name: String,
    #[serde(default)]
    pub world_id: Option<String>,
    pub state: String,
    /// The register it last said it was in; none for one never asked.
    #[serde(default)]
    pub mood: Option<String>,
    #[serde(default)]
    pub personality_name: Option<String>,
}

/// `GET /v1/npc`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpcList {
    pub items: Vec<Npc>,
}

/// `POST /v1/npc/:nid/mission`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct LodgeRequest {
    /// The brief — what is being asked of the character.
    pub prompt: String,
    /// The steps, in order. Empty, steps are drawn from the rooms, machines
    /// and people the brief names.
    #[serde(default)]
    pub todo: Vec<String>,
    /// Carry it at once, rousing the character to read it now, rather than
    /// queue it for the character to collect at the command table.
    #[serde(default)]
    pub start: bool,
}

/// What lodging a mission did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodgeReply {
    /// Queued for collection.
    pub lodged: bool,
    /// Carried at once.
    pub carrying: bool,
    pub prompt: String,
    /// How many steps it has.
    pub steps: usize,
}

/// One step of a mission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepView {
    pub text: String,
    pub done: bool,
    #[serde(default)]
    pub outcome: Option<StepOutcome>,
    /// Whether it is the report itself, which only filing the report closes.
    #[serde(default)]
    pub reports: bool,
}

/// A filed report: the character's verdict and account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportView {
    /// `pass` or `fail`.
    pub outcome: String,
    pub notes: String,
}

/// One mission, as an operator reads it back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionView {
    pub prompt: String,
    pub open: bool,
    /// Where it came from: `{kind: "lodged", by}`, `{kind: "random", routine}`
    /// or `{kind: "generated", …}`.
    pub origin: Value,
    /// The documents it is about, for a generated one.
    #[serde(default)]
    pub work: Value,
    pub todo: Vec<StepView>,
    #[serde(default)]
    pub answer: Option<String>,
    #[serde(default)]
    pub observed: Value,
    #[serde(default)]
    pub report: Option<ReportView>,
}

/// `GET /v1/npc/:nid/mission`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionStatus {
    pub on_mission: bool,
    /// The open one, or else the last finished.
    pub mission: Option<MissionView>,
    /// The last finished.
    pub finished: Option<MissionView>,
    /// How many more are queued for it.
    pub lodged: usize,
}

/// `POST /v1/npc/:nid/mission/step`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StepRequest {
    /// Sign off the first open step with this text.
    #[serde(default)]
    pub done: Option<String>,
    /// How it turned out: achieved when absent.
    #[serde(default)]
    pub outcome: Option<StepOutcome>,
    /// Add a step with this text.
    #[serde(default)]
    pub add: Option<String>,
}

/// What a step request did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepReply {
    pub ticked: Option<bool>,
    pub added: Option<bool>,
}

/// `POST /v1/npc/:nid/mission/cancel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelReply {
    pub cancelled: bool,
}

/// A message, a channel post: something said.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Say {
    pub text: String,
}

/// What saying something to a character on its handset did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageSent {
    pub sent: String,
    pub from: String,
    pub to: String,
    /// How much is now waiting for it, unread.
    pub waiting_for_them: usize,
    /// Whether it carries a handset, and so can answer at all.
    pub can_reply: bool,
}

/// One line said, and by whom.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Said {
    pub from: String,
    pub text: String,
}

/// `GET /v1/npc/:nid/message`: the caller's thread with a character.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Messages {
    pub messages: Vec<Said>,
    pub in_a_world: bool,
    /// The character, by the name the world gives it.
    #[serde(default)]
    pub with: Option<String>,
    /// The caller, as the world names them.
    #[serde(default, rename = "as")]
    pub as_name: Option<String>,
}

/// `GET /v1/world/:wid/channel`: what the channel carried since the caller
/// joined it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Channel {
    pub channel: String,
    pub world: String,
    #[serde(rename = "as")]
    pub as_name: String,
    pub messages: Vec<Said>,
}

/// What speaking on a world's channel did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelSent {
    pub sent: String,
    pub channel: String,
    pub world: String,
    pub from: String,
    /// Who was there to hear it.
    pub heard_by: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_lodge_request_defaults_to_queued_with_steps_drawn_from_the_brief() {
        let r: LodgeRequest =
            serde_json::from_value(json!({"prompt": "Check the valve."})).unwrap();
        assert_eq!(
            r,
            LodgeRequest {
                prompt: "Check the valve.".into(),
                todo: vec![],
                start: false
            }
        );
    }

    #[test]
    fn a_thread_and_a_channel_read_their_as() {
        let m: Messages = serde_json::from_value(json!({
            "messages": [{"from": "Pax", "text": "hi"}],
            "in_a_world": true, "with": "Pax", "as": "John"
        }))
        .unwrap();
        assert_eq!(m.as_name.as_deref(), Some("John"));
        let empty: Messages =
            serde_json::from_value(json!({"messages": [], "in_a_world": false})).unwrap();
        assert_eq!((empty.with, empty.as_name), (None, None));
        let c = Channel {
            channel: "the channel".into(),
            world: "w1".into(),
            as_name: "John".into(),
            messages: vec![],
        };
        assert_eq!(
            serde_json::to_string(&c).unwrap(),
            r#"{"channel":"the channel","world":"w1","as":"John","messages":[]}"#
        );
    }

    #[test]
    fn a_step_outcome_is_its_word() {
        assert_eq!(
            serde_json::to_string(&StepRequest {
                done: Some("scan it".into()),
                outcome: Some(StepOutcome::Thwarted),
                add: None
            })
            .unwrap(),
            r#"{"done":"scan it","outcome":"thwarted","add":null}"#
        );
    }
}
