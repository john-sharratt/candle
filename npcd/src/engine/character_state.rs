//! Serves what a character is doing and holds (its layers, memory, projections
//! and monitor), read from the engine without inventing values for what has not
//! run.
//!
//! Each route answers 404 for a character that does not exist, since a fixture
//! that answered for any id hid that. Counts a character has not run are real
//! zeros, because a character that has never run holds no turns. A paging
//! measurement or a monitor band that nothing has taken is absent, because the
//! console renders null as not measured. The live state socket stays open and
//! quiet when nothing emits, so the console does not read a silent daemon as a
//! fault. It closes when the client goes away.

use super::*;

/// Live character state — ticks, monitor bands, inbox depth.
///
/// A real socket that connects, holds, and says nothing, because nothing is
/// emitting. The console's roster subscribes to this to light its state dots,
/// and a socket that accepts and stays quiet is exactly right: the dots stay as
/// the listing drew them.
///
/// Not a 503, unlike the routes above. A refusal here would put the console
/// into its reconnect loop — backing off, retrying, reporting a fault — over a
/// daemon that is working perfectly and simply has nothing to say yet.
pub(super) async fn events(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(|mut socket| async move {
        // Held open until the client goes away. Reading is what notices that:
        // a browser closing a tab sends a close frame, and without this the
        // task would linger until the process ended.
        while let Some(Ok(msg)) = socket.recv().await {
            if matches!(msg, Message::Close(_)) {
                return;
            }
        }
    })
}

/// Confirm the caller owns this character, and give back its id.
///
/// Every route here is about one character, so every one of them 404s for a
/// character that does not exist — which is the difference the fixture could
/// not make, since it answered for any id at all.
///
/// `Box`ed on the error side, matching [`owner_of`]. An axum `Response` is a
/// large value and a `Result` is as big as its widest arm, so an unboxed one
/// makes every success on this path carry the refusal's footprint.
pub(super) async fn owned(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    nid: &str,
) -> Result<u64, Box<Response>> {
    Ok(owned_by(s, headers, nid).await?.2)
}

/// [`owned`], with the caller it resolved along the way.
///
/// The routes that write to a character — a line, a command, a mission — need to
/// say *who* wrote it, and that is the identity and account handle this has
/// already read. `nid` is the wire form, base-36: a route that parsed the path
/// segment as a decimal number answered 400 for every real id.
pub(crate) async fn owned_by(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    nid: &str,
) -> Result<(Identity, String, u64), Box<Response>> {
    let (id, owner) = owner_of(s, headers).await?;
    let not_found = || {
        Box::new(err(
            StatusCode::NOT_FOUND,
            "npc_not_found",
            "no such character",
        ))
    };
    let Some(npc_id) = npcs::npc_id_of_wire(nid) else {
        return Err(not_found());
    };
    if s.npcs.view().visible_to(npc_id, &owner).is_none() {
        return Err(not_found());
    }
    Ok((id, owner, npc_id))
}

/// The layer occupancy: every layer the schema declares, and what is in it.
///
/// Nothing is, yet. The turn and token counts are real zeros — a character that
/// has never run genuinely holds no turns — while `resident` is a measurement
/// of paging that nothing has taken, so it is absent.
pub(super) async fn substrate(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    // The layers are the mind's: every one `projection.yaml` declares, in its
    // order, whatever the engine is doing.
    let declared = projection::layers(&s.mind).unwrap_or_default();
    // A brief engine-lock read (substrate counts), taken inline: it holds no
    // lock across an await and costs the handler what any host-side read does.
    let live = minds_of(&s).and_then(|minds| minds.layer_counts(npc_id));
    Json(json!({
        "layers": declared.iter().map(|l| {
            let name = l.get("name").and_then(Value::as_str).unwrap_or_default();
            let held = live.as_ref().and_then(|c| c.iter().find(|c| c.layer == name));
            json!({
                "layer": name,
                "window": l.get("window").cloned().unwrap_or(Value::Null),
                // Real zeros without an engine: a character that has never
                // run holds nothing in any layer.
                "conversation_count": held.map_or(0, |c| c.conversation_count),
                "turn_count": held.map_or(0, |c| c.turn_count),
            })
        }).collect::<Vec<_>>(),
        "engine_connected": live.is_some(),
    }))
    .into_response()
}

/// How many conversations a layer view carries: the newest, and a flag that
/// there are older ones.
pub(super) const LAYER_PAGE: usize = 50;

/// One layer's conversations, as this character can read them — see
/// [`layers::readable`]. The layer has to be one the mind declares.
pub(super) async fn layer(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, name)): Path<(String, String)>,
) -> Response {
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let declared = projection::layers(&s.mind).unwrap_or_default();
    let known = declared
        .iter()
        .any(|l| l.get("name").and_then(Value::as_str) == Some(name.as_str()));
    if !known {
        // Checked, because it can be: a typo in a layer name should be a 404
        // rather than an empty list that looks like an empty layer.
        return err(
            StatusCode::NOT_FOUND,
            "no_such_layer",
            &format!("`{name}` is not a layer this mind declares"),
        );
    }
    let Some(minds) = minds_of(&s) else {
        return Json(json!({
            "layer": name,
            "conversations": [],
            "more": false,
            "engine_connected": false,
        }))
        .into_response();
    };
    // Inline for the same reason as the counts above: a brief engine-lock
    // read, no await under it.
    let page = minds.layer_page(npc_id, &name, LAYER_PAGE);
    Json(json!({
        "layer": name,
        "conversations": page.as_ref().map(|p| json!(p.conversations)).unwrap_or(json!([])),
        "more": page.as_ref().is_some_and(|p| p.more),
        "engine_connected": page.is_some(),
    }))
    .into_response()
}

/// The live minds, when the engine is up — how a route reads what a
/// character's conversations actually hold rather than what the mind declares.
pub(crate) fn minds_of(s: &Authored) -> Option<Arc<Minds>> {
    s.runtime.as_ref()?.minds.read().unwrap().clone()
}

/// One turn's stored form. There are no turns, so there is no turn.
pub(super) async fn turn(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, _layer, _turn)): Path<(String, String, String)>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return *r;
    }
    err(
        StatusCode::NOT_FOUND,
        "turn_not_found",
        "this character has no turns — nothing has run",
    )
}

/// What the character remembers having lived. Nothing has happened to it.
pub(super) async fn memory(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return *r;
    }
    Json(json!({ "items": [], "next_cursor": Value::Null, "engine_connected": false }))
        .into_response()
}

pub(super) async fn projection_now(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    projection_absent(&s, &headers, &nid).await
}

pub(super) async fn projection_at(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, _tick)): Path<(String, String)>,
) -> Response {
    projection_absent(&s, &headers, &nid).await
}

/// A projection is a composition made for one tick. None has been made.
///
/// `404` rather than an empty budget: an empty projection would say the gather
/// ran and found nothing, which is a different and much more alarming claim
/// than "nothing has run".
pub(super) async fn projection_absent(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    nid: &str,
) -> Response {
    if let Err(r) = owned(s, headers, nid).await {
        return *r;
    }
    err(
        StatusCode::NOT_FOUND,
        "no_projection",
        "no projection has been composed for this character — nothing has run",
    )
}

/// The metacognition monitor's band and overlap trace.
pub(super) async fn monitor(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return *r;
    }
    // Absent, not `healthy`. A band is a verdict on a character's attention,
    // and this one has had none — the console renders `null` as "not measured".
    Json(json!({
        "band": Value::Null,
        "overlap": Value::Null,
        "engine_connected": false,
    }))
    .into_response()
}

pub(super) async fn probe(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return *r;
    }
    no_engine("probing retrieval")
}

pub(super) async fn perceive(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return *r;
    }
    no_engine("delivering an event to a character")
}
