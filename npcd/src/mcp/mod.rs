//! Battle Cities' MCP server, at `/v1/mcp` (`docs/mcp_app.md` §4): the tools
//! that operate the game's characters — list them, give and manage their
//! missions, message them, read and speak on their world's channel.
//!
//! **Each tool is its route, called in-process.** A tool calls the handler
//! that answers the same act over HTTP, with the caller's own headers, and
//! reads its answer back as the route's body type ([`crate::engine::bodies`]).
//! So what the caller may do, and how a refusal is worded, is the route's: a
//! character they do not own is one npcd does not show them. The one check a
//! route makes outside its handler is its role, and the one tool whose route
//! asks more than `User` — speaking on a channel, a Creator's — makes it
//! itself.
//!
//! A call reaches the member holding the character or world it names
//! (`Derived::McpSubject`), where that route's handler would run.
//!
//! | Module | Tool |
//! |---|---|
//! | [`list`] | `npc_list` — the caller's characters |
//! | [`mission`] | `npc_mission` — what a character carries, and how its last went |
//! | [`give`] | `npc_mission_give` — give one a mission |
//! | [`step`] | `npc_mission_step` — sign off or add a step |
//! | [`cancel`] | `npc_mission_cancel` — call off its mission |
//! | [`message`] | `npc_message` — message it on its handset |
//! | [`messages`] | `npc_messages` — the caller's thread with it |
//! | [`channel`] | `npc_channel` — a world's channel |
//! | [`channel_post`] | `npc_channel_post` — speak on a world's channel |
//!
//! [`args`] reads a tool's arguments, [`route`] reads a handler's answer, and
//! [`view`] words a mission.

use std::future::Future;
use std::net::IpAddr;
use std::sync::{Arc, LazyLock};

use axum::http::HeaderMap;
use mcp_server::{protocol, Call, Identity, Server, Store};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use web::auth::session::Identity as Person;
use web::auth::Role;

use crate::api::Authored;
use crate::engine::bodies::Said;
use crate::identity::require;

pub mod args;
pub mod cancel;
pub mod channel;
pub mod channel_post;
pub mod give;
pub mod list;
pub mod message;
pub mod messages;
pub mod mission;
pub mod route;
pub mod step;
pub mod view;

/// What the server is, plainly: it operates a game's characters.
pub const DESCRIPTION: &str = "Battle Cities: operate the game's characters — list them, give and \
manage their missions, message them, and read and speak on their world's channel.";

/// How the tools go together.
pub const INSTRUCTIONS: &str = "The npc_* tools operate Battle Cities, Tokera's game. npc_list \
names your characters; the others take a character's id from it. Give a mission with \
npc_mission_give and follow it with npc_mission; talk to a character with npc_message and read its \
answers with npc_messages.";

static IDENTITY: LazyLock<Identity> = LazyLock::new(|| Identity {
    name: "battle-cities".into(),
    title: "Battle Cities".into(),
    description: DESCRIPTION.into(),
    instructions: INSTRUCTIONS.into(),
    icon: None,
    version: env!("CARGO_PKG_VERSION").into(),
});

/// A tool's answer: its text, and whether it is an error the model reads as
/// one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub text: String,
    pub is_error: bool,
}

impl Answer {
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
        }
    }

    /// The text a tool made, or what the route refused, as an error.
    pub fn of(result: Result<String, String>) -> Self {
        match result {
            Ok(text) => Self::ok(text),
            Err(refused) => Self::error(refused),
        }
    }
}

/// One Battle Cities tool.
pub trait Tool {
    const NAME: &'static str;
    fn definition() -> Value;
    fn call(
        s: &Arc<Authored>,
        headers: &HeaderMap,
        args: &Value,
    ) -> impl Future<Output = Answer> + Send;
}

/// Every tool, in the order `tools/list` shows them.
pub fn definitions() -> Value {
    json!([
        list::List::definition(),
        mission::Mission::definition(),
        give::Give::definition(),
        step::Step::definition(),
        cancel::Cancel::definition(),
        message::Message::definition(),
        messages::Thread::definition(),
        channel::Read::definition(),
        channel_post::Post::definition(),
    ])
}

/// Run tool `name` with `args`, as the caller `headers` name.
pub async fn call(s: &Arc<Authored>, headers: &HeaderMap, name: &str, args: &Value) -> Answer {
    match name {
        list::List::NAME => list::List::call(s, headers, args).await,
        mission::Mission::NAME => mission::Mission::call(s, headers, args).await,
        give::Give::NAME => give::Give::call(s, headers, args).await,
        step::Step::NAME => step::Step::call(s, headers, args).await,
        cancel::Cancel::NAME => cancel::Cancel::call(s, headers, args).await,
        message::Message::NAME => message::Message::call(s, headers, args).await,
        messages::Thread::NAME => messages::Thread::call(s, headers, args).await,
        channel::Read::NAME => channel::Read::call(s, headers, args).await,
        channel_post::Post::NAME => channel_post::Post::call(s, headers, args).await,
        other => Answer::error(format!("there is no tool `{other}`")),
    }
}

/// Lines said, one per line: `from: text`.
pub fn said(lines: &[Said]) -> String {
    lines
        .iter()
        .map(|s| format!("{}: {}", s.from, s.text.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

impl Server for Authored {
    type Caller = Person;
    type Answer = Answer;

    fn identity(&self) -> &Identity {
        &IDENTITY
    }

    /// The signed-in person the gateway named; the host has already cleared
    /// the identity headers from every peer that is not a trusted gateway.
    fn caller(&self, headers: &HeaderMap, _peer: Option<IpAddr>) -> Option<Person> {
        require(headers, &self.roles, Role::User)
            .ok()?
            .into_identity()
    }

    /// The first 16 hex digits of `sha256(provider \0 sub)`, as zend keys a
    /// person: a session id carries its owner, and a subject id is never put
    /// on the wire.
    fn owner(&self, caller: &Person) -> String {
        let digest = Sha256::digest(format!("{}\0{}", caller.provider, caller.sub).as_bytes());
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }

    fn store(&self) -> Result<Arc<dyn Store>, String> {
        Ok(self.mcp_sessions.clone())
    }

    async fn tools(_this: &Arc<Self>, _call: &Call<Self>) -> Result<Value, String> {
        Ok(definitions())
    }

    async fn call(this: &Arc<Self>, call: &Call<Self>, name: &str, args: &Value) -> Answer {
        self::call(this, &call.headers, name, args).await
    }

    fn result(&self, answer: &Answer) -> Value {
        protocol::text_result(&answer.text, answer.is_error)
    }

    async fn delivered(_this: &Arc<Self>, _call: &Call<Self>, _answer: &Answer) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_an_npc_name_a_description_and_an_object_schema() {
        let defs = definitions();
        let names: Vec<&str> = defs
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                assert!(d["description"].as_str().unwrap().len() > 40, "{d}");
                assert_eq!(d["inputSchema"]["type"], "object", "{d}");
                d["name"].as_str().unwrap()
            })
            .collect();
        assert_eq!(
            names,
            [
                "npc_list",
                "npc_mission",
                "npc_mission_give",
                "npc_mission_step",
                "npc_mission_cancel",
                "npc_message",
                "npc_messages",
                "npc_channel",
                "npc_channel_post"
            ]
        );
    }

    /// **Every tool acting on a character or a world names it as the cluster
    /// routes by** — `npc` or `world` — so the call reaches the member holding
    /// it (`cluster::dispatch::MCP_SUBJECTS`).
    #[test]
    fn every_tool_names_its_subject_as_the_cluster_routes_by() {
        for d in definitions().as_array().unwrap() {
            let props = d["inputSchema"]["properties"].as_object().unwrap();
            let required: Vec<&str> = d["inputSchema"]["required"]
                .as_array()
                .map(|r| r.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            if d["name"] == "npc_list" {
                assert!(required.is_empty());
                continue;
            }
            let subject = ["npc", "world"]
                .into_iter()
                .find(|k| props.contains_key(*k))
                .unwrap_or_else(|| panic!("{} names no subject", d["name"]));
            assert!(required.contains(&subject), "{}", d["name"]);
        }
    }

    #[test]
    fn a_refusal_is_an_error_answer_in_the_routes_words() {
        let a = Answer::of(Err("npc_not_found: no such character (404)".into()));
        assert!(a.is_error);
        assert_eq!(a.text, "npc_not_found: no such character (404)");
    }
}
