//! Talking on a conversation of the web UI's through the chat tools
//! (`docs/zen_chat.md` §6): `zen_chat_send` and `zen_chat_read` given a
//! conversation's id rather than a chat's.
//!
//! **A send is the web UI's own request, sent for the caller.** It goes up
//! this member's socket to a gateway as `POST /v1/chat/completions`, stating
//! the caller's identity (`ops::up`), so everything a person's browser meets
//! it meets too: the gateway places it on the member holding the conversation's
//! claim — or, the claim let go, on one holding the conversation complete — the
//! route checks the caller's role and that they may open the conversation, and
//! the turn's tools read the cluster as the caller. Nothing here decides where
//! a turn runs or who may run it; the route's rules are the only ones.
//!
//! Who may name a conversation is the web routes' rule (`ownership`): a
//! conversation the web UI made names no owner and opens for an Admin alone.
//! One the caller may not open, or that is not there, is answered alike — as
//! not there — so a refusal never says one exists.
//!
//! | Module | What |
//! |---|---|
//! | [`target`] | What a tool's `chat` names: a chat's id, or a conversation's |
//! | [`reply`] | The turn's stream as the route answered it |
//! | [`prose`] | A turn's text as a reader of it wants it |

mod prose;
mod reply;
mod target;

pub use target::{target, Target};

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::Method;
use candle_conversation::Role as TurnRole;
use serde_json::{json, Value};

use self::reply::heard;
use crate::chatml::split_turn;
use crate::mcp::caller::Caller;
use crate::mcp::ops;
use crate::mcp::ownership::Viewer;
use crate::mcp::threads::Effort;
use crate::mcp::workers;
use crate::session::{ConvEntry, ConversationDials, ZendSession};
use crate::turn_claims;
use crate::types::{Role, ToolMode};

/// The route a send goes up to: the web UI's own.
const ROUTE: &str = "/v1/chat/completions";

/// The least a send waits for the route to answer at all: long enough for the
/// gateway to place the turn, so an answer that never came means the route was
/// not reached rather than that the reply is slow.
const ANSWERED_WITHIN: Duration = Duration::from_secs(10);

/// How often a waiting read looks again: the turn runs on whichever member
/// holds the conversation, and its rounds reach this one replicated.
const LOOK_EVERY: Duration = Duration::from_secs(2);

/// What a conversation the caller may not open, or that is not there, is
/// answered — the same either way.
pub fn not_there(id: &str) -> String {
    format!("No conversation `{id}` you can open: it is not there, or it is not yours to open.")
}

/// What a send is refused with while the conversation is answering.
fn still_answering(id: &str) -> String {
    format!(
        "Zen is still answering on conversation `{id}`: read its answer with `zen_chat_read \
         {{chat: \"{id}\"}}`, then send this"
    )
}

/// The conversation `id`, when `caller` may open it and it is there — read on
/// the blocking pool, since the listing reads the substrate.
async fn opened(
    session: &Arc<ZendSession>,
    caller: &Caller,
    id: &str,
) -> Result<ConvEntry, String> {
    session.shared_substrate().ok_or_else(workers::loading)?;
    if !Viewer::of_caller(caller).may_open(id) {
        return Err(not_there(id));
    }
    let (session, wanted) = (Arc::clone(session), id.to_string());
    tokio::task::spawn_blocking(move || {
        session
            .list_conversations(true)
            .into_iter()
            .find(|c| c.id == wanted)
    })
    .await
    .map_err(|e| format!("the conversation could not be read: {e}"))?
    .ok_or_else(|| not_there(id))
}

/// Whether a turn is running on `id`: here, or on the member holding it.
fn answering(session: &ZendSession, id: &str) -> bool {
    session.live_turn(id).is_some() || turn_claims::held_elsewhere(id)
}

/// The request a send makes: `text` as the next user message, at the dials
/// the conversation last ran at — the web UI sends its composer's on every
/// turn, and a turn sent without them would reset the conversation's — with
/// the thinking effort `effort` when the caller names one.
pub fn request(
    id: &str,
    text: &str,
    dials: Option<ConversationDials>,
    effort: Option<Effort>,
) -> Value {
    let mut body = json!({
        "model": "zen-code",
        "stream": true,
        "conv_id": id,
        "messages": [{"role": "user", "content": text}],
    });
    if let Some(d) = dials {
        body["effort"] = json!(d.effort);
        body["verbosity"] = json!(d.verbosity);
        body["think"] = json!(d.think);
        body["tools"] = json!(ToolMode::from_level(d.tools).id());
    }
    if let Some(e) = effort {
        body["effort"] = json!(e.level());
        body["think"] = json!(true);
    }
    body
}

/// Send `text` on the conversation `id` for `caller`, waiting up to `wait` for
/// the reply; Zen thinks at `effort` when one is named, else at the
/// conversation's own dial.
pub async fn send(
    session: &Arc<ZendSession>,
    caller: &Caller,
    id: &str,
    text: &str,
    effort: Option<Effort>,
    wait: Duration,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("the `text` is empty: say what to tell Zen".into());
    }
    let conv = opened(session, caller, id).await?;
    if conv.archived {
        return Err(format!(
            "Conversation `{id}` is archived: it takes no more turns"
        ));
    }
    if answering(session, id) {
        return Err(still_answering(id));
    }
    let body = request(id, text, session.conversation_dials(id), effort);
    let reply = ops::up(
        session,
        caller,
        Method::POST,
        ROUTE,
        None,
        Some(&body),
        wait.max(ANSWERED_WITHIN),
    )
    .await
    .map_err(|e| {
        format!(
            "{e}. The turn may have started: read the conversation with `zen_chat_read {{chat: \
             \"{id}\"}}` before sending again"
        )
    })?;
    match reply.status {
        200 => {}
        404 => return Err(not_there(id)),
        503 => {
            return Err(format!(
                "No member can take conversation `{id}`'s turn now — the one holding it is \
                 restarting or out of reach. Nothing was sent; send it again shortly"
            ))
        }
        status => {
            let said: String = String::from_utf8_lossy(&reply.body)
                .chars()
                .take(300)
                .collect();
            return Err(format!("The turn was refused ({status}): {said}"));
        }
    }
    let got = heard(&reply.body);
    let mut out = format!("Sent on conversation `{id}` — {}.", conv.label);
    match got.error {
        Some(error) => {
            out.push_str(&format!(" Its turn failed: {error}"));
            let wrote = prose::assistant(&got.text);
            if !wrote.is_empty() {
                out.push_str(&format!("\n\nWhat it wrote first:\n{wrote}"));
            }
        }
        None if got.done => {
            out.push_str(" Zen answered:\n\n");
            out.push_str(&prose::assistant(&got.text));
        }
        None => out.push_str(&format!(
            " Zen is still answering: read its answer with `zen_chat_read {{chat: \"{id}\"}}`."
        )),
    }
    Ok(out)
}

/// The conversation `id`'s messages, each a role and its text as a reader
/// wants it, in order — its tool rounds' results counted, system text left
/// out.
fn messages(session: &ZendSession, id: &str) -> Vec<(Role, String)> {
    let history = session.conversation_history(id).unwrap_or_default();
    history
        .iter()
        .flat_map(|m| {
            let role = match m.role {
                TurnRole::User => Role::User,
                TurnRole::Assistant => Role::Assistant,
                TurnRole::System => Role::System,
            };
            split_turn(role, &m.text)
        })
        .filter_map(|(role, text)| match role {
            Role::User => Some((role, prose::user(&text))),
            Role::Assistant => Some((role, prose::assistant(&text))),
            Role::System | Role::Tool => None,
        })
        .collect()
}

/// Where a read with no `after` starts: after the last message a person — or
/// Claude, through a send — wrote, so it gives Zen's reply to it.
pub fn start_after(said: &[(Role, String)]) -> usize {
    said.iter()
        .rposition(|(role, text)| *role == Role::User && !text.starts_with('['))
        .map_or(0, |i| i + 1)
}

/// The conversation's messages after `from`, each under its number, and
/// whether Zen is still answering.
pub fn render(
    id: &str,
    label: &str,
    said: &[(Role, String)],
    from: usize,
    answering: bool,
) -> String {
    let mut out = format!(
        "Conversation `{id}` — {label}: {} message{}.",
        said.len(),
        if said.len() == 1 { "" } else { "s" }
    );
    if from >= said.len() {
        out.push_str(&format!(" Nothing after #{from}."));
    }
    for (n, (role, text)) in said.iter().enumerate().skip(from) {
        let who = match role {
            Role::Assistant => "Zen",
            _ if text.starts_with('[') => "Tools",
            _ => "User",
        };
        out.push_str(&format!("\n\n#{} {who}:\n{text}", n + 1));
    }
    if answering {
        out.push_str(&format!(
            "\n\nZen is still answering. Read on with `zen_chat_read {{chat: \"{id}\", after: {}}}`.",
            said.len()
        ));
    }
    out
}

/// Read the conversation `id` for `caller` after its message `after` (by
/// default after the last message a person wrote), waiting up to `wait` while
/// Zen answers and has said nothing new.
pub async fn read(
    session: &Arc<ZendSession>,
    caller: &Caller,
    id: &str,
    after: Option<usize>,
    wait: Duration,
) -> Result<String, String> {
    let conv = opened(session, caller, id).await?;
    let deadline = Instant::now() + wait;
    loop {
        let (said, running) = {
            let session = Arc::clone(session);
            let id = id.to_string();
            tokio::task::spawn_blocking(move || (messages(&session, &id), answering(&session, &id)))
                .await
                .map_err(|e| format!("the conversation could not be read: {e}"))?
        };
        let from = after.unwrap_or_else(|| start_after(&said));
        let replied = said
            .iter()
            .skip(from)
            .any(|(role, _)| *role == Role::Assistant);
        let left = deadline.saturating_duration_since(Instant::now());
        if !running || replied || left.is_zero() {
            return Ok(render(id, &conv.label, &said, from, running));
        }
        tokio::time::sleep(left.min(LOOK_EVERY)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn said(pairs: &[(Role, &str)]) -> Vec<(Role, String)> {
        pairs
            .iter()
            .map(|(r, t)| (r.clone(), t.to_string()))
            .collect()
    }

    /// **A send carries the conversation's own dials**, so a turn sent through
    /// a chat tool runs as the web UI's next turn would; an effort named
    /// overrides the conversation's.
    #[test]
    fn a_send_carries_the_conversations_dials() {
        let dials = ConversationDials {
            effort: 2,
            verbosity: 3,
            think: true,
            tools: 2,
        };
        assert_eq!(
            request("c-1", "fix it", Some(dials), None),
            json!({
                "model": "zen-code", "stream": true, "conv_id": "c-1",
                "messages": [{"role": "user", "content": "fix it"}],
                "effort": 2, "verbosity": 3, "think": true, "tools": "comprehensive",
            })
        );
        let deep = request("c-1", "fix it", Some(dials), Some(Effort::Deep));
        assert_eq!(
            (deep["effort"].clone(), deep["think"].clone()),
            (json!(3), json!(true))
        );
        let bare = request("c-1", "fix it", None, None);
        assert!(bare.get("effort").is_none() && bare.get("tools").is_none());
    }

    /// **A read with no `after` starts after the last message a person
    /// wrote**: tool results are not one.
    #[test]
    fn a_read_starts_after_the_last_persons_message() {
        let s = said(&[
            (Role::User, "Fix the paper."),
            (Role::Assistant, "[called file_edit]"),
            (Role::User, "[1 tool result]"),
            (Role::Assistant, "Done."),
        ]);
        assert_eq!(start_after(&s), 1);
        assert_eq!(start_after(&s[1..]), 0);
    }

    #[test]
    fn a_read_renders_each_message_under_its_number() {
        let s = said(&[
            (Role::User, "Fix the paper."),
            (Role::Assistant, "[called file_edit]"),
            (Role::User, "[1 tool result]"),
        ]);
        assert_eq!(
            render("c-1", "Paper", &s, 1, true),
            "Conversation `c-1` — Paper: 3 messages.\n\n#2 Zen:\n[called file_edit]\n\n#3 \
             Tools:\n[1 tool result]\n\nZen is still answering. Read on with `zen_chat_read \
             {chat: \"c-1\", after: 3}`."
        );
        assert_eq!(
            render("c-1", "Paper", &s, 3, false),
            "Conversation `c-1` — Paper: 3 messages. Nothing after #3."
        );
    }
}
