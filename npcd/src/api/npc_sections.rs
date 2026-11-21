//! The routes for a character's beliefs, relationships, agency and modulation.
//!
//! A read of a character is rendered by whichever view asked for it. Every
//! authoring write goes through one shared closure over the cast, which
//! resolves the caller and supersedes the character's record in the same way
//! for each of these routes.

use super::*;

/* ── the authoring plane (§16) ───────────────────────────────────────────────
 *
 * What an operator says a character believes, who they know, what they are
 * trying to do, and where their affect sits. Every write here is an authoring
 * act and comes back marked `origin: "authored"`, so it stays distinguishable
 * from whatever the evidence process later earns.
 *
 * All of it is stored on the character's own record and supersedes with it —
 * see [`crate::npcs`]. `User` is the bar and *ownership* is the rest of the
 * answer, checked per record in `Npcs`, because a role cannot express "yours".
 *
 * The reads report the engine's measurements as **absent**: a belief has no
 * disconfirmation until something weighed evidence against it. These used to
 * fall through to the fixture, which answered with three invented beliefs for
 * every character — including ones that do not exist. */

/// Every belief an operator has stated.
pub(super) async fn get_beliefs(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::beliefs_wire).await
}

pub(super) async fn put_belief(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, bid)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Response {
    write_npc(s, headers, nid, move |n, id, owner, now| {
        n.put_belief(id, owner, &bid, &body, now)
    })
    .await
}

pub(super) async fn delete_belief(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, bid)): Path<(String, String)>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.delete_belief(npc_id, &owner, &bid, now_ms()) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => err(StatusCode::NOT_FOUND, "belief_not_found", &bid),
        Err(e) => npc_err(e),
    }
}

pub(super) async fn get_relationships(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::relationships_wire).await
}

pub(super) async fn put_relationship(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, eid)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Response {
    write_npc(s, headers, nid, move |n, id, owner, now| {
        n.put_relationship(id, owner, &eid, &body, now)
    })
    .await
}

pub(super) async fn get_agency(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::agency_wire).await
}

pub(super) async fn put_strategy(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, sid)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Response {
    write_npc(s, headers, nid, move |n, id, owner, now| {
        n.put_strategy(id, owner, &sid, &body, now)
    })
    .await
}

pub(super) async fn get_modulation(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::modulation_wire).await
}

pub(super) async fn put_modulation(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    write_npc(s, headers, nid, move |n, id, owner, now| {
        n.put_modulation(id, owner, &body, now)
    })
    .await
}

/// One read of a character, rendered by whichever view asked.
pub(super) async fn read_npc(
    s: Arc<Authored>,
    headers: HeaderMap,
    nid: String,
    view: fn(&candle_conversation::persistence::record::NpcPayload) -> Value,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.view().visible_to(npc_id, &owner) {
        Some(n) => Json(view(n)).into_response(),
        None => err(StatusCode::NOT_FOUND, "npc_not_found", "no such character"),
    }
}

/// One authoring write, whichever it is.
///
/// The write itself is a closure over `Npcs` so every one of these routes
/// resolves its caller and supersedes the record the same way.
pub(super) async fn write_npc<F>(
    s: Arc<Authored>,
    headers: HeaderMap,
    nid: String,
    write: F,
) -> Response
where
    F: FnOnce(&Npcs, u64, &str, u64) -> Result<Value, NpcError>,
{
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match write(&s.npcs, npc_id, &owner, now_ms()) {
        Ok(mut v) => {
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}
