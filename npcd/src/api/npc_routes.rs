//! The routes for a character: the cast's listing, creation, edits, ownership,
//! backup, and the reflection and dream routes.
//!
//! Ownership is checked per record inside each handler, and a character that is
//! missing or belongs to someone else answers as not found. Character ids cross
//! the wire as base-36 strings, because a u64 past 2^53 does not survive a
//! JavaScript client exactly. A create checks that its world and personality
//! exist and belong together, then wakes the character so it starts thinking
//! without a restart. Reflection and dreaming hold their request while they
//! run, and they stop the rest of the cast while they do.

use super::*;

/// The caller's characters. No total is returned — see §8.3.
/// `GET /v1/npcs/export` — the whole cast as records, for a backup.
///
/// Every character, live and tombstoned, so a restore into a fresh substrate
/// brings the cast back with its ids intact. Admin-only and owner-blind: this
/// is an operator backup of the entire deployment, not a per-account listing.
pub(super) async fn export_npcs(State(s): State<Arc<Authored>>) -> Response {
    let npcs = s.npcs.view().export_all();
    Json(json!({ "count": npcs.len(), "npcs": npcs })).into_response()
}

/// `POST /v1/npcs/import` — restore backed-up character records.
///
/// Body is `{"npcs": [<record>, …]}` from [`export_npcs`]. Each record is
/// written verbatim (id, owner, revision, last place preserved). The characters
/// come back on the next daemon start, which loads them from the log and spawns
/// their loops — the ordinary boot path, so nothing here needs to embody them.
pub(super) async fn import_npcs(
    State(s): State<Arc<Authored>>,
    Json(body): Json<Value>,
) -> Response {
    let payloads: Vec<NpcPayload> =
        match serde_json::from_value(body.get("npcs").cloned().unwrap_or(Value::Null)) {
            Ok(v) => v,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("bad npcs payload: {e}")).into_response()
            }
        };
    match s.npcs.import(payloads) {
        Ok(n) => {
            // A restored record may be suspended.
            suspend::follow(&s);
            Json(json!({ "imported": n })).into_response()
        }
        Err(e) => {
            tracing::error!(error = ?e, "npc import failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "import failed").into_response()
        }
    }
}

/// `GET /v1/substrate/npcd/objects/export` — every kept custom object, its tombstones
/// and the blobs it names, as JSON Lines (`kept_objects`). Read off the async
/// runtime: it reads every blob a kept object names.
pub(super) async fn export_objects(State(s): State<Arc<Authored>>) -> Response {
    let shared = s.npcs.substrate();
    match spawn_blocking(move || shared.export_kept_objects("npcd")).await {
        Ok(Ok(text)) => ([(header::CONTENT_TYPE, "application/x-ndjson")], text).into_response(),
        Ok(Err(e)) => {
            tracing::error!(error = ?e, "object export failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("export failed: {e}"),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("export failed: {e}"),
        )
            .into_response(),
    }
}

/// `POST /v1/substrate/npcd/objects/import` — write the union of the exports the body
/// holds, one after another — one from each member — leaving every key
/// already live as it is, and answer what was written.
pub(super) async fn import_objects(State(s): State<Arc<Authored>>, body: String) -> Response {
    let shared = s.npcs.substrate();
    match spawn_blocking(move || shared.import_kept_objects(&body, "npcd")).await {
        Ok(Ok(report)) => Json(report).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, format!("import refused: {e}")).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("import failed: {e}"),
        )
            .into_response(),
    }
}

/// `GET /v1/npcs/owners` — every living character with its owner, the accounts
/// a character can be handed to, and the caller's own id among them.
pub(super) async fn get_owners(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let (_, me) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let npcs = s.npcs.view().owners();
    let accounts = s.accounts.read().await.known();
    Json(json!({ "me": me, "accounts": accounts, "npcs": npcs })).into_response()
}

/// `PUT /v1/npc/:nid/owner` — hand a character to another account.
///
/// Body `{"owner_id": "u_…"}`. Only an account that exists may be named: a
/// character handed to an id nobody signs in as is one nobody can see again.
pub(super) async fn put_owner(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    let Some(to) = body.get("owner_id").and_then(Value::as_str) else {
        return err(
            StatusCode::BAD_REQUEST,
            "owner_missing",
            "say whose it is to be: {\"owner_id\": \"u_…\"}",
        );
    };
    let known = s
        .accounts
        .read()
        .await
        .known()
        .iter()
        .any(|a| a["user_id"] == to);
    if !known {
        return err(
            StatusCode::BAD_REQUEST,
            "owner_unknown",
            "no account has that id — an owner is somebody who has signed in",
        );
    }
    match s.npcs.reassign(npc_id, to, now_ms()) {
        Ok(mut v) => {
            tracing::info!(npc = %nid, owner = to, "character reassigned");
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

pub(crate) async fn list_npcs(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let filter = Filter {
        tag: q.get("tag").map(String::as_str),
        state: q.get("state").map(String::as_str),
        // A world id that names nothing filters to nothing rather than being
        // ignored: silently widening a filter shows characters the caller asked
        // not to see, which is the wrong way to fail.
        world_id: q.get("world_id").map(String::as_str),
        q: q.get("q").map(String::as_str),
        // Hidden characters are opt-in per request, never the default: the
        // listing somebody screen-shares should not contain them.
        include_hidden: matches!(q.get("hidden").map(String::as_str), Some("1" | "true")),
    };
    let mut items = s.npcs.view().list(&owner, &filter);
    let reg = s.personalities.read().await;
    for it in &mut items {
        name_personality(it, &reg);
    }
    Json(json!({ "items": items })).into_response()
}

/// Fill in `personality_name` from the authored document.
///
/// A character record stores the slug and nothing else — a copy of the name
/// would be a second place for it to live and go stale the moment the file is
/// retitled. The roster wants a name, so the join happens here, once per
/// listing, against a map that is already in memory. A slug whose file is gone
/// keeps no name and the page falls back to showing the slug, which is the
/// honest thing to show for a reference that no longer resolves.
pub(crate) fn name_personality(npc: &mut Value, reg: &Registry) {
    let Some(id) = npc.get("personality_id").and_then(Value::as_str) else {
        return;
    };
    let Some(name) = reg
        .get(id)
        .and_then(|r| r.body.get("name").and_then(Value::as_str))
        .map(str::to_owned)
    else {
        return;
    };
    if let Some(map) = npc.as_object_mut() {
        map.insert("personality_name".to_string(), json!(name));
    }
}

pub(super) async fn get_npc(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        // An unparseable id is simply not a character anybody has.
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.view().get(npc_id, &owner) {
        Ok(mut v) => {
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// `POST /v1/npc/:nid/reflect` — stop a character and have it think once.
///
/// Runs two turns in a conversation that is **never persisted and tombstoned on
/// the way out**: the first asks what comes to the character where it stands,
/// the second asks it to describe a dream it has not had. The first answer is
/// what a live tick would hand back to the character; the second is a brief for
/// a dream that a separate conversation would decode.
///
/// **Serial.** It holds the request for every decode the reflection runs — the
/// two questions, the loosed turn and the repair passes — and it stops every
/// other character thinking while it runs, the same way the portrait route does.
/// Reflecting is what a character does when it has already chosen to stand
/// still, so the wait is not the fast clock blocking on a slow one.
///
/// The response is split into parts rather than returned as one blob because
/// they have genuinely different audiences — see
/// [`crate::engine::reflect::Reflection`].
pub(super) async fn post_reflect(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    // Ownership before anything expensive: a reflection costs several decodes
    // and stops the cast, and neither should be spendable on somebody else's
    // character.
    if let Err(e) = s.npcs.view().get(npc_id, &owner) {
        return npc_err(e);
    }
    if let Some(r) = suspend::refused(&s, npc_id) {
        return r;
    }

    // What is going through its head. Required, because it is the half of a
    // reflection the character supplies and there is no sensible default for
    // it — a reflection with nothing in it reflects on nothing.
    let inner = body
        .get("inner_thoughts")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if inner.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "inner_thoughts_required",
            "say what is going through its head — a reflection needs something to be about",
        );
    }
    let feeling = body
        .get("feeling")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    // Where the character says it is — what the act passes. Optional here: a
    // caller that names none gets the world's own percept of the room.
    let situation = body
        .get("situation")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    // Absent means rotate. Naming one is for a caller walking the space
    // deliberately — a test, or a fill of a cell the corpus is thin in.
    let domain = body
        .get("domain")
        .and_then(Value::as_str)
        .map(str::to_string);
    // A SAMPLE of the character's existing axes, never all of them. Shown
    // everything it has dreamt, a generator stops inventing and recombines.
    let axes: Vec<String> = body
        .get("sampled_axes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let Some(rt) = s.runtime.clone() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_engine",
            "this daemon has no engine",
        );
    };
    // Awaited directly: the decodes yield on their turn channels, so the
    // route parks a future rather than a worker — and a client that
    // disconnects drops it, which stops the decode it was waiting on.
    let out = rt
        .reflect(
            npc_id,
            situation.as_deref(),
            &inner,
            &feeling,
            domain.as_deref(),
            &axes,
        )
        .await;

    match out {
        Ok(r) => Json(r).into_response(),
        Err(e) => err(
            StatusCode::SERVICE_UNAVAILABLE,
            "reflect_failed",
            &format!("{e}"),
        ),
    }
}

/// `POST /v1/npc/:nid/dream` — decode a single dream on demand.
///
/// A dream ordinarily falls out of a reflection on the character's own clock,
/// sporadically. This decodes one now from a supplied `brief` — the
/// present-tense premise the dreamer lives through — so a caller can see what
/// the dreaming produces without waiting for the tick that would have asked for
/// it. It is kept like any dream, so it also lands on the character's dream
/// layer.
///
/// **Serial**, like the reflection route: it holds the request for the decode
/// and stops the cast while it runs, which is what a dream costs.
pub(super) async fn post_dream(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (id, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    // Ownership — unless the caller may see the whole cast (admin or creator).
    // An on-demand dream is an inspection tool: an operator watching the mind
    // should be able to poke any character in it, the same way the pulse already
    // lets them read one.
    if !s.roles.of(Some(&id)).at_least(Role::Admin) {
        if let Err(e) = s.npcs.view().get(npc_id, &owner) {
            return npc_err(e);
        }
    }
    if let Some(r) = suspend::refused(&s, npc_id) {
        return r;
    }

    // The premise the dream is decoded from — present tense, the dreamer living
    // it. Required: a dream with no premise dreams nothing.
    let brief = body
        .get("brief")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if brief.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "brief_required",
            "a dream needs a `brief` — the present-tense premise it dreams from",
        );
    }
    // The axis the dream suspends, recorded on the kept dream. Optional: a
    // one-off dream asked for by hand need not name one.
    let assumption = body
        .get("assumption")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .unwrap_or("a dream asked for by hand")
        .to_string();

    let Some(rt) = s.runtime.clone() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_engine",
            "this daemon has no engine",
        );
    };
    match rt.dream_once(npc_id, &brief, &assumption).await {
        Ok(kept) => Json(json!({
            // Stringified for the same reason every id on the wire is: a u64
            // past 2^53 does not survive a JavaScript client exact.
            "timeline": kept.timeline.to_string(),
            "passages": kept.passages,
            "text": kept.passages.join("\n\n"),
        }))
        .into_response(),
        Err(e) => err(
            StatusCode::SERVICE_UNAVAILABLE,
            "dream_failed",
            &format!("{e}"),
        ),
    }
}

pub(super) async fn create_npc(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let (id, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    // The two references have to name documents that exist. `Npcs::create`
    // checks their *shape* — it holds no registry and cannot do more — so a
    // character pointed at a deleted personality would be written happily and
    // fail at spawn, long after the page that made it was closed.
    if let Some(r) = missing_ref(&s, &body).await {
        return r;
    }
    let created = s.npcs.create(&id, &owner, &body, now_ms());
    match created {
        Ok(mut v) => {
            // Put the new character into the loop straight away.
            //
            // The cast is woken at startup, so without this a character created
            // while the daemon runs never ticks — it exists, it is listed, and
            // it is not alive. The failure reads as the engine ignoring it, and
            // the only fix a person would find is a restart. `wake` is
            // idempotent, so doing it here costs nothing on the startup path.
            if let Some(rt) = s.runtime.as_ref() {
                if let Some(id) = v.get("npc_id").and_then(npc_id_of) {
                    // **Woken here when it runs here.** Under a cluster a
                    // character runs on the member its world's authority lends
                    // it to (`docs/replicated_substrate.md` §9.3), which wakes
                    // it there; a daemon with no cluster runs them all.
                    if rt.standing.runs(id) {
                        let world_ms = s.world_ms(id);
                        rt.scheduler.wake(id, 0, world_ms);
                        // And its task with it — waking puts the inbox in the
                        // scheduler; the task is what waits on it and thinks.
                        rt.spawn_character(id);
                        tracing::info!("npc {id}: created and woken");
                    }

                    // And given a body, if its world has anywhere to stand. A
                    // character whose world is authored but unmapped is a
                    // character with lore and no body — which is most of them,
                    // and not a failure.
                    let name = v.get("name").and_then(Value::as_str).unwrap_or_default();
                    let world = v
                        .get("world_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    // Where this kind of character belongs, from its
                    // personality — a Maker starts in the vault, a soldier in a
                    // city, and they are the same world — and whether the world
                    // may order it at all. Read from its persona, as every
                    // other path that embodies a character reads them.
                    let persona = rt.persona_of_async(id).await;
                    let kind = persona.as_ref().map(OwnedPersona::kind).unwrap_or_default();
                    // No remembered room: this character has just been created
                    // and has never stood anywhere. It arrives at the door.
                    match rt.embody_in_world(id, world, kind, name, None, 0) {
                        Ok(true) => tracing::info!("npc {id}: standing in `{world}`"),
                        Ok(false) => {}
                        // Not fatal to the create: the character exists and is
                        // thinking. Loud, because a character that should have
                        // a body and has not is invisible otherwise — it looks
                        // exactly like one whose world was never mapped.
                        Err(e) => tracing::error!("npc {id}: no body in `{world}` — {e:#}"),
                    }
                }
            }
            name_personality(&mut v, &*s.personalities.read().await);
            (StatusCode::CREATED, Json(v)).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// A character id off the wire.
///
/// The id is serialised as a base-36 **string** ([`npcs::npc_id_wire`]),
/// because a `u64` past 2^53 does not survive a JavaScript client — so
/// reading it back out of a response body is a base-36 parse, not `as_u64`,
/// and using the latter here silently found nothing.
pub(super) fn npc_id_of(v: &Value) -> Option<u64> {
    v.as_str()
        .and_then(npcs::npc_id_of_wire)
        .or_else(|| v.as_u64())
}

/// The 400 for a create that names a world or personality this daemon does not
/// have, or `None` when both resolve.
///
/// A malformed reference answers with its shape, not with `unknown_world`.
/// Both are refusals and the lookup itself is safe — an id is a `BTreeMap` key
/// here, never a path — but "no world `../../etc`" reads as an invitation to go
/// and create one, where `id::check`'s message says what is actually wrong.
pub(super) async fn missing_ref(s: &Arc<Authored>, body: &Value) -> Option<Response> {
    let named = |key: &str| {
        body.get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };

    for (key, code) in [
        ("world_id", "unknown_world"),
        ("personality_id", "unknown_personality"),
    ] {
        let name = named(key);
        // An absent or non-string reference is `Npcs::create`'s to report, in
        // the `invalid_arguments` shape the console already handles.
        if name.is_empty() {
            continue;
        }
        if let Err(e) = registry::id::check(&name) {
            return Some(err(
                StatusCode::BAD_REQUEST,
                "invalid_arguments",
                &format!("{key}: {e}"),
            ));
        }
        let known = if key == "world_id" {
            s.worlds.read().await.get(&name).is_some()
        } else {
            s.personalities.read().await.get(&name).is_some()
        };
        if !known {
            return Some(err(StatusCode::BAD_REQUEST, code, &name));
        }
    }

    // Both documents exist. The character also has to belong to the world it is
    // being created in: a personality is written for one world and names it in
    // `world:`, so any other pairing is a character in a setting it has no
    // canon for.
    //
    // Checked here rather than left to the console's filtering, because the
    // console is presentation: a create posted by curl, by a script, or by a
    // page held open while a personality was re-homed would otherwise write a
    // character that fails at spawn, long after the page that made it closed —
    // exactly the failure the reference check above exists to prevent.
    let (world, personality) = (named("world_id"), named("personality_id"));
    if !world.is_empty() && !personality.is_empty() {
        // The world's cast, if it names one. A world that names none admits
        // everyone — the standing default, and what stops the first world to
        // declare a cast from emptying every other.
        let cast: Option<Vec<String>> = s.worlds.read().await.get(&world).and_then(|r| {
            r.body
                .get("personalities")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
        });
        if let Some(cast) = cast {
            if !cast.iter().any(|c| c == &personality) {
                return Some(err(
                    StatusCode::BAD_REQUEST,
                    "personality_not_of_world",
                    &format!("`{world}` does not cast `{personality}`"),
                ));
            }
        }
    }
    None
}

pub(super) async fn patch_npc(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.patch(npc_id, &owner, &body, now_ms()) {
        Ok(mut v) => {
            // A state edit may suspend or resume it.
            suspend::follow(&s);
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// Replace a character's tags.
///
/// A `PUT` because it is the whole set, not an addition — the console holds the
/// chips and sends what is left after one is removed, which is a replacement
/// however it was reached.
pub(super) async fn put_npc_tags(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let tags = body.get("tags").cloned().unwrap_or(Value::Null);
    patch_one(s, headers, nid, json!({ "tags": tags })).await
}

/// Hide or unhide a character.
///
/// Discretion, not encryption: a hidden character is left out of the default
/// listing and found again by filtering for one of its tags (§8.3). The console
/// says so beside the checkbox, and warns when the last tag is removed from a
/// hidden character, because that is the combination that makes one unreachable.
pub(super) async fn put_npc_hidden(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let hidden = body.get("hidden").cloned().unwrap_or(Value::Null);
    patch_one(s, headers, nid, json!({ "hidden": hidden })).await
}

/// The shared body of the single-field routes above.
///
/// They go through `Npcs::patch` rather than writing the record themselves, so
/// there is one validator, one supersession and one place a character's fields
/// are checked. A route that wrote its own record would be a second answer to
/// "what may a tag be".
pub(super) async fn patch_one(
    s: Arc<Authored>,
    headers: HeaderMap,
    nid: String,
    patch: Value,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.patch(npc_id, &owner, &patch, now_ms()) {
        Ok(mut v) => {
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// Delete: one superseding record with `state: "tombstoned"`. The id stays
/// taken, because the acts the character already committed still name it.
pub(super) async fn delete_npc(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npcs::npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.delete(npc_id, &owner, now_ms()) {
        Ok(()) => {
            // Take the character out of the engine too. Without this its loop
            // keeps ticking against an authored record that no longer resolves —
            // the persona source returns `None`, so it thinks about nothing for
            // ever, and its conversation is never retired.
            if let Some(rt) = s.runtime.as_ref() {
                // Its mind, its body and everything waiting on either — the
                // same step every other member takes as the deletion reaches
                // it (`Runtime::follow_cast`).
                rt.forget_character(npc_id);
                let minds = rt.minds.read().unwrap().clone();
                if let Some(minds) = minds {
                    minds.retire_npc(npc_id).await;
                }
            }
            StatusCode::NO_CONTENT.into_response()
        }
        Err(e) => npc_err(e),
    }
}
