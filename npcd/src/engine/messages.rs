//! Carries a person's messages to a character and the character's replies, on the
//! same handset and thread that the characters use among themselves.
//!
//! A message goes onto the thread the characters use, so the ordinary sweep tells
//! the character about it and the character answers with its ordinary message act.
//! Reading the conversation returns an empty list for a character with no body,
//! which is an honest answer. The world's channel and posting routes are
//! creator-only, because they reach every character in a world rather than one
//! character that someone owns, and ownership cannot express that.

use super::*;

/// `POST /v1/npc/:nid/message` — say something to a character on its handset.
///
/// Goes onto the same thread the characters use, so the character is told about
/// it by the ordinary sweep and answers with the ordinary `message` act. A
/// person is a party to the conversation rather than an operator poking at one.
pub(crate) async fn post_message(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    body: Result<Json<Say>, JsonRejection>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let said = said(body);
    let text = said.trim();
    if text.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_message",
            "a message needs something in it",
        );
    }
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("messaging a character");
    };
    let not_in_a_world = || {
        err(
            StatusCode::CONFLICT,
            "not_in_a_world",
            "that character has no body in a world, so there is nothing to reach it on",
        )
    };
    let Some((hosted, _)) = rt.body_of(npc_id) else {
        return not_in_a_world();
    };
    let op = WorldOp::Message {
        npc: npc_id,
        from: Who::new(&handle, &me),
        text: text.to_string(),
        origin: None,
    };
    let sent = match rt.world_write(hosted.id(), op).await {
        Ok(OpReply::Sent { sent: Some(sent) }) => sent,
        Ok(_) => return not_in_a_world(),
        Err(e) => return world_unreached(&e),
    };
    Json(MessageSent {
        sent: text.to_string(),
        from: me,
        to: sent.with,
        waiting_for_them: sent.waiting_for_them,
        can_reply: sent.can_reply,
    })
    .into_response()
}

/// `GET /v1/npc/:nid/message` — the conversation so far, oldest first.
pub(crate) async fn get_messages(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let me = speaking_as(&id, &handle, &s.roles);
    let said = s
        .runtime
        .as_ref()
        .and_then(|rt| rt.messages_with(npc_id, &handle));
    let Some((them, messages)) = said else {
        // Not an error: a character with no body has nothing to say on a
        // handset yet, and the console renders an empty conversation.
        return Json(Messages {
            messages: Vec::new(),
            in_a_world: false,
            with: None,
            as_name: None,
        })
        .into_response();
    };
    Json(Messages {
        messages: messages
            .into_iter()
            .map(|(from, text)| Said { from, text })
            .collect(),
        in_a_world: true,
        with: Some(them),
        as_name: Some(me),
    })
    .into_response()
}

/// `POST /v1/world/:wid/posting` — leave words on something in a world.
///
/// **Creator only**, for the reason the channel's write side is: it writes into
/// a world rather than into a character somebody owns, and ownership has
/// nothing to say about that.
///
/// The body names `at` (the room, as `area/node`), `on` (the surface), and
/// `text`. A surface the map never named is stood up on the spot, so a world can
/// grow a board without its map being edited and reloaded — but the *room* must
/// exist, or the board would be writable, unreachable, and invisible.
pub(super) async fn post_posting(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let field = |k: &str| {
        body.get(k)
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string()
    };
    let (at, on, text) = (field("at"), field("on"), field("text"));
    if at.is_empty() || on.is_empty() || text.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "incomplete",
            "a posting needs `at` (area/node), `on` (what to write on) and `text`",
        );
    }
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("posting into a world");
    };
    let op = WorldOp::Posting {
        at: at.clone(),
        on: on.clone(),
        by: me.clone(),
        text: text.clone(),
    };
    let lines = match rt.world_write(&wid, op).await {
        Ok(OpReply::Posted { lines }) => lines,
        Ok(_) => None,
        Err(e) => return world_unreached(&e),
    };
    let Some(lines) = lines else {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_place",
            "that world is not hosted, or it has no such room — a board in a room that does not \
             exist could never be read",
        );
    };
    Json(json!({
        "posted": text,
        "world": wid,
        "at": at,
        "on": on,
        "by": me,
        "lines": lines,
    }))
    .into_response()
}

/// `GET /v1/world/:wid/channel` — what has been said on a world's open channel.
///
/// Read as the caller, so it shows what the channel has carried since they
/// joined it and not the hours before — the same rule every other member reads
/// it under.
pub(crate) async fn get_channel(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading a world's channel");
    };
    let Some(said) = rt.channel(&wid, &handle) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_world",
            "that world is not hosted, so it has no channel",
        );
    };
    Json(Channel {
        channel: CHANNEL.to_string(),
        world: wid,
        as_name: me,
        messages: said
            .into_iter()
            .map(|(from, text)| Said { from, text })
            .collect(),
    })
    .into_response()
}

/// `POST /v1/world/:wid/channel` — say something on a world's open channel.
///
/// **Creator only.** Everything else on this daemon is scoped by ownership: a
/// route reaches the characters the caller owns and no others. This one reaches
/// every character in a world at once, which is not a thing ownership can
/// express — so it is bound to the one role that is *about* standing outside
/// the whole world rather than owning part of it.
pub(crate) async fn post_channel(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
    body: Result<Json<Say>, JsonRejection>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let said = said(body);
    let text = said.trim();
    if text.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_message",
            "a message needs something in it",
        );
    }
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("speaking on a world's channel");
    };
    let op = WorldOp::Channel {
        from: Who::new(&handle, &me),
        text: text.to_string(),
        origin: None,
    };
    let heard = match rt.world_write(&wid, op).await {
        Ok(OpReply::Heard { by }) => by,
        Ok(_) => None,
        Err(e) => return world_unreached(&e),
    };
    let Some(heard) = heard else {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_world",
            "that world is not hosted, so it has no channel",
        );
    };
    Json(ChannelSent {
        sent: text.to_string(),
        channel: CHANNEL.to_string(),
        world: wid,
        from: me,
        heard_by: heard,
    })
    .into_response()
}
