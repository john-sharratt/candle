//! npcd's MCP server, at `/v1/npc/mcp` (`docs/mcp_app.md` §4): the tools that
//! manage the caller's characters as **bots** — list them, give and manage
//! their missions, message them, read and speak on their world's channel. The
//! model meets them as bots to be managed; a bot is an npcd character, named by
//! its `npc_id`.
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
//! A call reaches the member holding the bot or world it names
//! (`Derived::McpSubject`), where that route's handler would run.
//!
//! | Module | Tool |
//! |---|---|
//! | [`list`] | `bot_list` — the caller's bots |
//! | [`mission_read`] | `bot_mission_read` — what a bot carries, and how its last went |
//! | [`give`] | `bot_mission_give` — give one a mission |
//! | [`step`] | `bot_mission_step` — sign off or add a step |
//! | [`cancel`] | `bot_mission_cancel` — call off its mission |
//! | [`message_send`] | `bot_message_send` — message it on its handset |
//! | [`message_read`] | `bot_message_read` — the caller's thread with it |
//! | [`channel_read`] | `bot_channel_read` — a world's channel |
//! | [`channel_post`] | `bot_channel_post` — speak on a world's channel |
//! | [`image`] | `image_generate` — draw a picture, answered as an image and a link to the file |
//! | [`image_edit`] | `image_edit` — change a picture by a prompt, answered the same way |
//! | [`image_upload`] | `image_upload` — a link the user's own picture is sent to, for `image_edit` |
//!
//! [`args`] reads a tool's arguments, [`route`] reads a handler's answer,
//! [`view`] words a mission, and [`edit_mode`] is `image_edit`'s modes.
//!
//! `image_generate` and `image_upload` act on nothing stored, so like
//! `bot_list` they name no subject and go to the best member. `image_edit`
//! names its picture as `image`, which the gateway places among the members
//! that hold it (`Authored::hand_over_image`).

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
pub mod channel_post;
pub mod channel_read;
pub mod edit_mode;
pub mod give;
pub mod image;
pub mod image_edit;
pub mod image_upload;
pub mod list;
pub mod message_read;
pub mod message_send;
pub mod mission_read;
pub mod route;
pub mod step;
pub mod view;

/// What the server is, plainly: it manages bots.
pub const DESCRIPTION: &str =
    "Bots: manage your bots — list them, give and manage their missions, \
message them, and read and speak on their world's channel — and generate images.";

/// How the tools go together.
pub const INSTRUCTIONS: &str =
    "The bot_* tools manage your bots: autonomous agents, each living in \
a world and acting on its own. bot_list names them; the others take a bot's id from it. Give a \
mission with bot_mission_give and follow it with bot_mission_read; talk to a bot with \
bot_message_send and read its replies with bot_message_read. Any request to draw or create a \
picture, image or photo goes to image_generate — the user's own image model, rather than an SVG or \
widget you draw yourself; read the `image` guide first. It returns the picture for you to look \
at, with a link to the file — the user may not see the image, so send them the file from the link. \
image_edit changes a picture by a prompt — one image_generate drew, by its link, or the user's own \
sent with image_upload — to add or remove a detail or refine it.";

static IDENTITY: LazyLock<Identity> = LazyLock::new(|| Identity {
    name: "bots".into(),
    title: "Bots".into(),
    description: DESCRIPTION.into(),
    instructions: INSTRUCTIONS.into(),
    icon: None,
    version: env!("CARGO_PKG_VERSION").into(),
});

/// A tool's answer: its text, whether it is an error the model reads as one,
/// and a picture when the tool drew one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub text: String,
    pub is_error: bool,
    pub picture: Option<Picture>,
}

/// A picture an answer carries, base64, as an MCP image block takes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub data: String,
    pub mime_type: String,
}

impl Answer {
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
            picture: None,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
            picture: None,
        }
    }

    /// `text` with the picture it describes.
    pub fn picture(text: impl Into<String>, picture: Picture) -> Self {
        Self {
            picture: Some(picture),
            ..Self::ok(text)
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

/// One bot tool.
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
        mission_read::Mission::definition(),
        give::Give::definition(),
        step::Step::definition(),
        cancel::Cancel::definition(),
        message_send::Message::definition(),
        message_read::Thread::definition(),
        channel_read::Read::definition(),
        channel_post::Post::definition(),
        image::Generate::definition(),
        image_edit::Edit::definition(),
        image_upload::Upload::definition(),
    ])
}

/// Run tool `name` with `args`, as the caller `headers` name.
pub async fn call(s: &Arc<Authored>, headers: &HeaderMap, name: &str, args: &Value) -> Answer {
    match name {
        list::List::NAME => list::List::call(s, headers, args).await,
        mission_read::Mission::NAME => mission_read::Mission::call(s, headers, args).await,
        give::Give::NAME => give::Give::call(s, headers, args).await,
        step::Step::NAME => step::Step::call(s, headers, args).await,
        cancel::Cancel::NAME => cancel::Cancel::call(s, headers, args).await,
        message_send::Message::NAME => message_send::Message::call(s, headers, args).await,
        message_read::Thread::NAME => message_read::Thread::call(s, headers, args).await,
        channel_read::Read::NAME => channel_read::Read::call(s, headers, args).await,
        channel_post::Post::NAME => channel_post::Post::call(s, headers, args).await,
        image::Generate::NAME => image::Generate::call(s, headers, args).await,
        image_edit::Edit::NAME => image_edit::Edit::call(s, headers, args).await,
        image_upload::Upload::NAME => image_upload::Upload::call(s, headers, args).await,
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

/// The `tools/call` result `answer` is sent as: a text block, and after it the
/// picture as an image block — the text is what every client reads, the image
/// what one that shows pictures does.
pub fn result_of(answer: &Answer) -> Value {
    let mut result = protocol::text_result(&answer.text, answer.is_error);
    if let (Some(p), Some(content)) = (&answer.picture, result["content"].as_array_mut()) {
        content.push(json!({"type": "image", "data": p.data, "mimeType": p.mime_type}));
    }
    result
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
        result_of(answer)
    }

    async fn delivered(_this: &Arc<Self>, _call: &Call<Self>, _answer: &Answer) {}
}

#[cfg(test)]
mod tests {
    use cluster::dispatch::MCP_SUBJECTS;

    use super::*;

    #[test]
    fn every_tool_has_its_name_a_description_and_an_object_schema() {
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
                "bot_list",
                "bot_mission_read",
                "bot_mission_give",
                "bot_mission_step",
                "bot_mission_cancel",
                "bot_message_send",
                "bot_message_read",
                "bot_channel_read",
                "bot_channel_post",
                "image_generate",
                "image_edit",
                "image_upload"
            ]
        );
    }

    /// **Every tool acting on a bot, a world or a stored picture names it as
    /// the cluster routes by** ([`MCP_SUBJECTS`]), so the call reaches a
    /// member holding it; a draw and an upload name nothing and go anywhere.
    #[test]
    fn every_tool_names_its_subject_as_the_cluster_routes_by() {
        for d in definitions().as_array().unwrap() {
            let props = d["inputSchema"]["properties"].as_object().unwrap();
            let required: Vec<&str> = d["inputSchema"]["required"]
                .as_array()
                .map(|r| r.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            if d["name"] == "bot_list" {
                assert!(required.is_empty());
                continue;
            }
            // Acts on nothing stored: any member draws it, or takes the upload.
            // An edit names its picture (`image`), and is placed among the
            // members that hold it.
            if d["name"] == "image_generate" || d["name"] == "image_upload" {
                assert!(MCP_SUBJECTS.into_iter().all(|k| !props.contains_key(k)));
                continue;
            }
            let subject = MCP_SUBJECTS
                .into_iter()
                .find(|k| props.contains_key(*k))
                .unwrap_or_else(|| panic!("{} names no subject", d["name"]));
            assert!(required.contains(&subject), "{}", d["name"]);
        }
    }

    #[test]
    fn a_picture_follows_its_text_as_an_image_block() {
        let plain = result_of(&Answer::ok("done"));
        assert_eq!(plain["content"].as_array().unwrap().len(), 1);
        let drawn = result_of(&Answer::picture(
            "Drew it.",
            Picture {
                data: "iVBO".into(),
                mime_type: "image/png".into(),
            },
        ));
        assert_eq!(
            drawn["content"],
            json!([
                {"type": "text", "text": "Drew it."},
                {"type": "image", "data": "iVBO", "mimeType": "image/png"}
            ])
        );
        assert_eq!(drawn["isError"], false);
    }

    #[test]
    fn a_refusal_is_an_error_answer_in_the_routes_words() {
        let a = Answer::of(Err("npc_not_found: no such character (404)".into()));
        assert!(a.is_error);
        assert_eq!(a.text, "npc_not_found: no such character (404)");
    }
}
