//! The routes for worlds and personalities: their listings, clocks, collections
//! and documents.
//!
//! Hidden worlds and personalities are left out of a listing until a whole word
//! of the filter names them, so typing letters cannot discover them. The
//! narrative clock is computed from a stored anchor. Only an admin can change it,
//! and a request touches only the fields it names, so pausing does not reset
//! the pace.

use super::*;

/// Every world, minus the hidden ones the filter has not named.
///
/// `q` is the console's filter box. A world with `hidden: true` is left out
/// until a **whole word** of `q` names it — see [`crate::visibility`]. Filtering
/// happens here rather than in the browser because a list the client narrows is
/// a list the client was first sent in full.
pub(super) async fn list_worlds(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let query = q.get("q").map(String::as_str).unwrap_or_default();
    let reveal = revealing(&s, &headers, &q);
    let npcs = s.npcs.view();
    let counts = npcs.counts_by(|n| n.world_id.as_str());
    let worlds: Vec<Value> = seen_all(&s, &headers, &s.worlds, WORLDS)
        .await
        .iter()
        .filter(|r| reveal || visibility::listable(&r.id, &r.body, query))
        .map(|r| with_count(with_id("world_id", &r.id, &r.body), &counts, &r.id))
        .collect();
    Json(json!({ "worlds": worlds })).into_response()
}

pub(super) async fn get_world(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
) -> Response {
    let record = seen(&s, &headers, &s.worlds, WORLDS, &wid).await;
    let npcs = s.npcs.view();
    let counts = npcs.counts_by(|n| n.world_id.as_str());
    match record {
        Some(r) => {
            let mut body = with_count(with_id("world_id", &r.id, &r.body), &counts, &r.id);
            // The narrative clock, computed rather than stored: what is on disk
            // is an anchor, and the time now is a function of how long ago it
            // was taken. See [`crate::clock`].
            if let Some(map) = body.as_object_mut() {
                let now = now_ms() as i64;
                map.insert("time".into(), Clock::of_world(&r.body, now).wire(now));
            }
            Json(body).into_response()
        }
        None => err(StatusCode::NOT_FOUND, "world_not_found", &wid),
    }
}

/// Move a world's clock, or change how fast it runs.
///
/// Every character in the world dates what they remember by this, so it is an
/// admin's — the same bar as editing the world document, which is where it is
/// written.
///
/// Both operations in one route because the console offers them together and
/// they are the same write: the body may carry `world_ms` to jump to, `scale`
/// to change the pace, `paused`, `year_offset` to set the calendar that many
/// years ahead of the anchor, or any combination. What is absent is left
/// alone rather than defaulted, so a request that only pauses does not silently
/// reset the speed.
pub(super) async fn put_world_time(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let author = match caller::author(&s, &headers) {
        Ok(a) => a,
        Err(r) => return *r,
    };
    let current = {
        let reg = s.worlds.read().await;
        records::record(&author, &reg, WORLDS, &wid).map(|r| r.body)
    };
    let Some(current) = current else {
        return err(StatusCode::NOT_FOUND, "world_not_found", &wid);
    };

    let now = now_ms() as i64;
    let mut clock = Clock::of_world(&current, now);

    // The pace first, so a request that changes both banks the elapsed run at
    // the old speed before the jump replaces the anchor.
    let scale = body.get("scale").and_then(Value::as_f64);
    let paused = body.get("paused").and_then(Value::as_bool);
    if scale.is_some() || paused.is_some() {
        let scale = scale.unwrap_or(clock.scale);
        if !scale.is_finite() || scale < 0.0 {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_field",
                "`scale` is world time per real time, and cannot be negative",
            );
        }
        clock = clock.set_pace(scale, paused.unwrap_or(clock.paused), now);
    }
    // The offset before the jump, so a request that sets both names the time
    // as the world will read it under the new offset.
    if let Some(years) = body.get("year_offset") {
        let Some(years) = years.as_i64().and_then(|y| i32::try_from(y).ok()) else {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_field",
                "`year_offset` is a whole number of years, positive for the future",
            );
        };
        clock = clock.with_year_offset(years, now);
    }
    if let Some(to) = body.get("world_ms").and_then(Value::as_i64) {
        clock = clock.jump_to(to, now);
    }

    let next = Value::Object(clock::with_clock(&current, clock));
    match records::save(&author, "world", WORLDS, &wid, &next) {
        Ok(()) => {
            s.wrote(&author).await;
            Json(clock.wire(now)).into_response()
        }
        Err(e) => registry_err(e),
    }
}

/// The section libraries a world's lens is assembled from.
///
/// The same answer for every world, deliberately: responses and moods are
/// ingested untagged, so every world shares both the text and its KV. The route
/// is per-world because the console asks it per-world and because a world will
/// one day be able to override a collection — when it can, this is where that
/// happens.
pub(super) async fn world_collections(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
) -> Response {
    let Some(excludes) = seen(&s, &headers, &s.worlds, WORLDS, &wid)
        .await
        .map(|r| collections::excludes_of(&r.body))
    else {
        return err(StatusCode::NOT_FOUND, "world_not_found", &wid);
    };
    Json(s.libraries().world_wire(&excludes)).into_response()
}

/// What a personality contributes to a projection, from its own document.
///
/// The counterpart of [`world_collections`], and the last of the two
/// collection views to stop being invented — see
/// [`collections::personality_wire`].
pub(super) async fn personality_collections(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(aid): Path<String>,
) -> Response {
    let Some(body) = seen(&s, &headers, &s.personalities, PERSONALITIES, &aid)
        .await
        .map(|r| r.body)
    else {
        return err(StatusCode::NOT_FOUND, "personality_not_found", &aid);
    };
    Json(collections::personality_wire(&body)).into_response()
}

pub(super) async fn put_world(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    save(&s, &headers, "world", WORLDS, "world_id", &wid, body).await
}

/// Delete a world's document. The world running from it stops — and its
/// state records are tombstoned (`docs/replicated_substrate.md` §9.1) — once
/// the document is gone from the folder: at once over a mind that keeps no
/// drafts, when the deletion is published over one that does.
pub(super) async fn delete_world(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
) -> Response {
    remove(&s, &headers, WORLDS, "world_not_found", &wid).await
}

/// Every personality, minus the hidden ones the filter has not named.
///
/// Same rule as worlds — `hidden: true` in the document, revealed by a whole
/// word of `q`. Nothing in the mind uses it yet; it is here because the two
/// listings should not answer the same question two different ways, and because
/// the day a character is meant to be discreet the flag is already the answer.
pub(super) async fn list_personalities(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let query = q.get("q").map(String::as_str).unwrap_or_default();
    let reveal = revealing(&s, &headers, &q);
    let npcs = s.npcs.view();
    let counts = npcs.counts_by(|n| n.personality_id.as_str());
    let personalities: Vec<Value> = seen_all(&s, &headers, &s.personalities, PERSONALITIES)
        .await
        .iter()
        .filter(|r| reveal || visibility::listable(&r.id, &r.body, query))
        .map(|r| {
            with_portrait_image(
                with_count(with_id("personality_id", &r.id, &r.body), &counts, &r.id),
                &s,
                &r.id,
            )
        })
        .collect();
    Json(json!({ "personalities": personalities })).into_response()
}

pub(super) async fn get_personality(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(aid): Path<String>,
) -> Response {
    let record = seen(&s, &headers, &s.personalities, PERSONALITIES, &aid).await;
    let npcs = s.npcs.view();
    let counts = npcs.counts_by(|n| n.personality_id.as_str());
    match record {
        Some(r) => Json(with_portrait_image(
            with_count(with_id("personality_id", &r.id, &r.body), &counts, &r.id),
            &s,
            &r.id,
        ))
        .into_response(),
        None => err(StatusCode::NOT_FOUND, "personality_not_found", &aid),
    }
}

/// Add the servable id for the portrait this personality was authored with.
///
/// Written into the `portrait` block beside the author's own `image:` and
/// `prompt:` rather than at the top level, so the console reads one object for
/// everything about the picture. The author's path stays exactly as written —
/// this only adds the id the image route answers to, which is a fact about this
/// daemon's store and not about the document.
pub(super) fn with_portrait_image(mut v: Value, s: &Authored, id: &str) -> Value {
    let Some(image_id) = s
        .personality_portraits
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(id)
        .cloned()
    else {
        return v;
    };
    let Some(map) = v.as_object_mut() else {
        return v;
    };
    map.entry(personality_portrait::FIELD)
        .or_insert_with(|| json!({}));
    if let Some(p) = map
        .get_mut(personality_portrait::FIELD)
        .and_then(Value::as_object_mut)
    {
        p.insert("image_id".to_string(), json!(image_id));
    }
    v
}

pub(super) async fn put_personality(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(aid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    save(
        &s,
        &headers,
        "personality",
        PERSONALITIES,
        "personality_id",
        &aid,
        body,
    )
    .await
}

pub(super) async fn delete_personality(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(aid): Path<String>,
) -> Response {
    remove(&s, &headers, PERSONALITIES, "personality_not_found", &aid).await
}
