//! Opens, reads, speaks into and ends the sessions in which a person is with a
//! character, either physically in its room or by message.
//!
//! A physical session places the person in the character's room, which is what
//! turns the in-world verbs back on, since a character with no body cannot be
//! stood beside. A session that has gone quiet reads as gone rather than ended.
//! Ending one takes the person back out of the room, so the character is not told
//! that company is present when it is not. In a message session a line is a
//! message on the person's thread, answered by their conversation. A slash command
//! in either kind of session is the operator reaching into the character.

use super::*;

/// Who this caller is, to a character.
pub(super) fn as_interlocutor(
    id: &web::auth::session::Identity,
    handle: &str,
    roles: &web::auth::Roles,
) -> Interlocutor {
    Interlocutor {
        kind: "operator".into(),
        id: handle.to_string(),
        display: speaking_as(id, handle, roles),
    }
}

/// Interactions this character is in.
pub(super) async fn list_interactions(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return Json(json!({ "interactions": [], "engine_connected": false })).into_response();
    };
    // The wall clock: whether a session has gone quiet is a fact about a person
    // at a console, not about the world's own time. See `Interaction::last_ms`.
    let now = crate::api::now_ms();
    let live: Vec<Value> = rt
        .interactions
        .for_npc(npc_id, now)
        .iter()
        .map(|ix| ix.wire(now))
        .collect();
    Json(json!({ "interactions": live, "engine_connected": true })).into_response()
}

/// Open one, or continue the one already open with this person in this mode.
pub(super) async fn open_interaction(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("opening an interaction");
    };
    // Physical by default: standing in the room together is the ordinary way to
    // be present to a character, and the one an operator who did not say means.
    let wanted = body
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("physical");
    let Some(mode) = crate::engine::tools::Mode::parse(wanted) else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_mode",
            "a mode is physical or instant_message",
        );
    };
    // A character with no body cannot be stood next to. The messaging modes
    // reach somebody who is nowhere near, which is the whole point of them, so
    // only the physical one needs a place.
    if mode == crate::engine::tools::Mode::Physical && rt.body_of(npc_id).is_none() {
        return err(
            StatusCode::CONFLICT,
            "not_in_a_world",
            "that character has no body to stand beside — open a messaging mode instead",
        );
    }
    // Two clocks, and they are not interchangeable: going quiet is a fact about
    // a person at a console, and the world's own instant is only the label the
    // session carries. See `Interaction::last_ms`.
    let world_ms = s.world_ms(npc_id);
    let now = crate::api::now_ms();
    let who = as_interlocutor(&id, &handle, &s.roles);
    let world = rt
        .body_of(npc_id)
        .map(|(hosted, _)| hosted.id().to_string());
    let ix = rt
        .interactions
        .open(npc_id, mode, who.clone(), world.clone(), now, world_ms);

    // **Physical means physical.** The person walks into the room the character
    // is standing in, so it appears in `Within::company` — which is what turns
    // `tell`, `ask`, `give`, `touch` and `gesture` back on. Without a body it
    // heard you and could not answer you, and went on waiting for somebody to
    // arrive while you were talking to it.
    let standing = match (mode == crate::engine::tools::Mode::Physical, &world) {
        (true, Some(world)) => {
            let op = WorldOp::EnterBeside {
                npc: npc_id,
                visitor: ix.body.clone(),
                name: who.display.clone(),
            };
            match rt.world_write(world, op).await {
                Ok(OpReply::Placed { at }) => at,
                Ok(_) => None,
                Err(e) => return world_unreached(&e),
            }
        }
        _ => None,
    };
    let mut wire = ix.wire(now);
    if let Some(at) = standing {
        wire["you_are_at"] = json!(at);
    }
    Json(wire).into_response()
}

pub(super) async fn interaction(
    State(s): State<Arc<Authored>>,
    Path(ix): Path<String>,
) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    let now = crate::api::now_ms();
    match rt.interactions.get(&ix, now) {
        Some(found) => Json(found.wire(now)).into_response(),
        // A session that has gone quiet reads as gone rather than as ended:
        // there is nothing left to look at either way, and the console's own
        // "no such interaction" is the honest thing to show.
        None => err(StatusCode::NOT_FOUND, "interaction_not_found", &ix),
    }
}

/// End one, and walk the person back out of the world.
pub(super) async fn end_interaction(
    State(s): State<Arc<Authored>>,
    Path(ix): Path<String>,
) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    let session = rt.interactions.get(&ix, crate::api::now_ms());
    if !rt.interactions.end(&ix) {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    }
    // The body goes with the session. A person who closed the conversation has
    // gone, and a room that still lists them is a room where a character is
    // told it has company that is not there.
    let left = match session {
        Some(was) if was.mode == crate::engine::tools::Mode::Physical => match &was.world {
            Some(world) => {
                let op = WorldOp::Leave {
                    visitor: was.body.clone(),
                };
                matches!(
                    rt.world_write(world, op).await,
                    Ok(OpReply::Done { ok: true })
                )
            }
            None => rt.leave_world(&was.body),
        },
        _ => false,
    };
    Json(json!({ "interaction_id": ix, "state": "ended", "left_the_world": left })).into_response()
}

/// Say something to the character, inside a session.
///
/// In an instant-message session a line is a message on the person's thread
/// with the character, answered by their conversation; everything else below
/// is a physical session's, and a slash command's in either.
///
/// **The same door everything else goes through.** The line is parsed by
/// [`crate::engine::slash`] and delivered by the scheduler, exactly as
/// `/v1/npc/:nid/pulse` does — so what is said here is a thing that happened to
/// the character rather than a private aside, and the rest of the world sees it
/// the way it sees anything else.
pub(super) async fn inject(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(ix): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("speaking to a character");
    };
    let now = crate::api::now_ms();
    let Some(session) = rt.interactions.get(&ix, now) else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    // The session says which character; ownership is still checked, because an
    // interaction id is not a capability.
    if let Err(r) = owned(&s, &headers, &npcs::npc_id_wire(session.npc_id)).await {
        return *r;
    }
    let line = body
        .get("line")
        .or_else(|| body.get("text"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if line.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_line",
            "there is nothing in that to say",
        );
    }
    // **A chat is a conversation** (`docs/npc_conversations.md` §4.1): what is
    // typed in an instant-message session goes on the person's direct thread
    // with the character, by their account, and the character's conversation
    // with them answers it — the one path a Discord DM takes too. A slash
    // command is still the operator reaching into the character.
    if session.mode == crate::engine::tools::Mode::InstantMessage && !line.starts_with('/') {
        let Some((hosted, _)) = rt.body_of(session.npc_id) else {
            return err(
                StatusCode::CONFLICT,
                "not_in_a_world",
                "that character has no body in a world, so there is nothing to reach it on",
            );
        };
        let who = &session.interlocutor;
        let op = WorldOp::Message {
            npc: session.npc_id,
            from: Who::new(&who.id, &who.display),
            text: line.to_string(),
            origin: None,
        };
        return match rt.world_write(hosted.id(), op).await {
            Ok(OpReply::Sent { sent: Some(_) }) => {
                rt.interactions.touched(&ix, now);
                Json(json!({ "delivered": true, "on_the_phone": true })).into_response()
            }
            Ok(_) => err(
                StatusCode::CONFLICT,
                "not_in_a_world",
                "that character has no handset to reach it on",
            ),
            Err(e) => world_unreached(&e),
        };
    }
    let parsed = match crate::engine::slash::parse(line) {
        // Named, or the character is told "you says to you" — see
        // [`crate::engine::slash::Parsed::attributed_to`].
        Ok(p) => p.attributed_to(&session.interlocutor.display),
        // A typo is a 400 naming the near miss, never speech — sending `/hrut`
        // to a character as dialogue is the one outcome that looks like it
        // worked.
        Err(e) => return err(StatusCode::BAD_REQUEST, "bad_command", &e.message()),
    };
    let world_ms = s.world_ms(session.npc_id);
    let kind = match parsed.kind {
        crate::engine::event::EventKind::Sleep { .. } => crate::engine::event::EventKind::Sleep {
            day: crate::clock::day_of(world_ms),
        },
        other => other,
    };
    let prose =
        crate::engine::event::Event::new(0, world_ms, parsed.salience, kind.clone()).prose();
    if !rt
        .scheduler
        .deliver(session.npc_id, world_ms, parsed.salience, kind)
    {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "that character is not awake, so nothing can reach it",
        );
    }
    rt.interactions.touched(&ix, now);
    Json(json!({ "delivered": true, "prose": prose })).into_response()
}
