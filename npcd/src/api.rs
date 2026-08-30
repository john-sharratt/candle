//! The first routes `npcd` answers itself rather than handing to the mock.
//!
//! Worlds and personalities are authored files (see [`crate::registry`]), so they
//! are real even though there is no engine yet — a save here writes YAML into
//! the repository and shows up in `git status`. Everything else still falls
//! through to `web::mock::npcd`, which is why this is a thin router laid *over*
//! that one rather than a fork of it: as each surface becomes real it moves
//! here, one route at a time, and the console never notices.
//!
//! The ids in these paths never become paths. `GET /v1/world/{wid}` is a lookup
//! in a `BTreeMap` that was filled at boot, and an id that is not a key is a
//! 404 — there is no filesystem call on the read path to traverse. The one
//! place an id becomes a file name is a save, and that goes through
//! `registry::id::check` first.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json,
};
// Only the test-only `router` builds one directly; `main` goes through `api`.
#[cfg(test)]
use axum::Router;
use serde_json::{json, Value};
use tokio::sync::RwLock;
use web::auth::session::Identity;

use web::auth::{Role, Roles};

use crate::accounts::{self, Accounts, NameError, PatchError};
use crate::clock::{self, Clock};
use crate::collections::{self, Libraries};
use crate::guard::Api;
use crate::identity::require;
use crate::images::{ImageError, Images};
use crate::mind::catalog::CatalogError;
use crate::mind::doc::{DocError, Wrote};
use crate::mind::{
    catalog as mind_catalog, parts as mind_parts, section as mind_section, Address, Mind, MindPath,
    Scope,
};
use crate::npcs::{self, Filter, NpcError, Npcs};
use crate::registry::{self, PutError, Registry};
use crate::visibility;

/// Both authored collections, shared with the HTTP layer.
///
/// A write lock is held only for the duration of a save. Reads are the common
/// case by orders of magnitude and take the read lock, so a GUI listing worlds
/// never waits on another GUI editing one.
pub struct Authored {
    pub worlds: RwLock<Registry>,
    pub personalities: RwLock<Registry>,
    pub accounts: RwLock<Accounts>,
    /// The cast, in memory, backed by the substrate — see [`crate::npcs`].
    /// A write lock only for a create/edit/delete; listing takes the read lock.
    pub npcs: RwLock<Npcs>,
    /// Who is an admin, from the config. Not behind a lock: it is decided at
    /// startup and there is deliberately no way to change it while running —
    /// see [`web::auth::role`].
    pub roles: Roles,
    /// The response and mood libraries, read once from the mind. No lock: they
    /// are authored in files and nothing here writes them.
    pub libraries: Libraries,
    /// Uploaded portraits, on disk. No lock: it holds a path, and the writes
    /// are content-addressed and atomic — two uploads of the same image are the
    /// same file, and of different ones are different names.
    pub images: Images,
    /// The mind directory, for the file editor. No lock: it holds a path, and
    /// the filesystem is the thing being shared — two saves to one document
    /// race in the OS whatever this does, and each is atomic (see
    /// [`crate::mind::doc`]).
    pub mind: Mind,
}

impl Authored {
    pub fn new(
        worlds: Registry,
        personalities: Registry,
        accounts: Accounts,
        npcs: Npcs,
        roles: Roles,
        libraries: Libraries,
        mind: Mind,
        images: Images,
    ) -> Arc<Self> {
        Arc::new(Self {
            worlds: RwLock::new(worlds),
            personalities: RwLock::new(personalities),
            accounts: RwLock::new(accounts),
            npcs: RwLock::new(npcs),
            roles,
            libraries,
            images,
            mind,
        })
    }
}

/// The routes this daemon owns, to be layered over the mock.
///
/// Every line names the role it needs, because [`Api`] has no method that adds
/// a route without one — see [`crate::guard`]. The check runs there, not in the
/// handlers, so there is exactly one place per route where the answer lives.
///
/// Paths whose methods differ in role are registered twice, which axum merges:
/// reading a world is open and writing one is an admin's, and one line each is
/// what makes that visible.
pub fn api(state: Arc<Authored>) -> Api<Arc<Authored>> {
    Api::new(state.roles.clone())
        // Authored content. Anyone may read it; only an admin may change a file
        // on disk. The mind is not under version control, so a bad write is not
        // a row to restore — it is prose somebody wrote, gone.
        .route("/v1/world", Role::Unauthenticated, get(list_worlds))
        .route("/v1/world/:wid", Role::Unauthenticated, get(get_world))
        // The response and mood libraries, read from the mind. This used to
        // fall through to the fixture, which answered with six invented
        // templates where the mind holds 596 — see [`crate::collections`].
        .route(
            "/v1/world/:wid/collections",
            Role::Unauthenticated,
            get(world_collections),
        )
        .route(
            "/v1/world/:wid",
            Role::Admin,
            put(put_world).delete(delete_world),
        )
        // The narrative clock. Admin for the same reason the document is: every
        // character in the world dates what they remember by it.
        .route("/v1/world/:wid/time", Role::Admin, put(put_world_time))
        .route(
            "/v1/personality",
            Role::Unauthenticated,
            get(list_personalities),
        )
        .route(
            "/v1/personality/:aid",
            Role::Unauthenticated,
            get(get_personality),
        )
        .route(
            "/v1/personality/:aid",
            Role::Admin,
            put(put_personality).delete(delete_personality),
        )
        // The anchor, the facets and the doctrine, from the document itself.
        .route(
            "/v1/personality/:aid/collections",
            Role::Unauthenticated,
            get(personality_collections),
        )
        // The authored corpus — canon, craft, characters, settings. Addressed
        // by what things ARE (`canon/ammo/bolt`), never by where they are
        // stored; see [`crate::mind::address`].
        //
        // Reading is `User` rather than open, unlike the documents above. The
        // difference is enumeration: `/v1/personality/cindy-tan` answers
        // somebody who already knows the id, while listing hands out the whole
        // corpus a level at a time, which is exactly the browsing the `hidden`
        // flag exists to prevent. The console is a signed-in tool, so this
        // costs nothing that was available anyway.
        .route("/v1/mind/list", Role::User, get(mind_list))
        .route("/v1/mind/entry", Role::User, get(mind_entry))
        .route(
            "/v1/mind/entry",
            Role::Admin,
            put(put_mind_entry).delete(delete_mind_entry),
        )
        // The same entry as fields rather than as text, so it can be edited by
        // somebody who does not know YAML. A save patches the values into the
        // document that is there, keeping the authoring comments above every
        // key — see [`crate::mind::section`].
        .route("/v1/mind/fields", Role::User, get(mind_fields))
        .route("/v1/mind/fields", Role::Admin, put(put_mind_fields))
        // The nine layers, from the mind's own schema rather than from a second
        // copy of them — see [`schema_layers`].
        .route("/v1/schema/layers", Role::User, get(schema_layers))
        // The cast. These replace the console's fixture: a character here is a
        // record in the substrate owned by the signed-in account. `User` is the
        // bar; *ownership* is the rest of the answer and is checked per-record
        // in the handler, because a role cannot express "yours" (§8.2).
        .route("/v1/npc", Role::User, get(list_npcs).post(create_npc))
        .route(
            "/v1/npc/:nid",
            Role::User,
            get(get_npc).patch(patch_npc).delete(delete_npc),
        )
        // Two fields with routes of their own, because the console edits them
        // from controls that are nowhere near the rest of the form: a tag chip
        // and a checkbox, each saving on the spot. They are `PATCH` underneath
        // — same validation, same record, same supersession — and exist so the
        // console does not have to send a whole character to add one tag.
        .route("/v1/npc/:nid/tags", Role::User, put(put_npc_tags))
        .route("/v1/npc/:nid/hidden", Role::User, put(put_npc_hidden))
        // The authoring plane (§16). `User` is the bar; ownership is the rest
        // of the answer and is checked per record.
        .route("/v1/npc/:nid/beliefs", Role::User, get(get_beliefs))
        .route(
            "/v1/npc/:nid/beliefs/:bid",
            Role::User,
            put(put_belief).delete(delete_belief),
        )
        .route(
            "/v1/npc/:nid/relationships",
            Role::User,
            get(get_relationships),
        )
        .route(
            "/v1/npc/:nid/relationships/:eid",
            Role::User,
            put(put_relationship),
        )
        .route("/v1/npc/:nid/agency", Role::User, get(get_agency))
        .route("/v1/npc/:nid/agency/:sid", Role::User, put(put_strategy))
        .route(
            "/v1/npc/:nid/modulation",
            Role::User,
            get(get_modulation).put(put_modulation),
        )
        .route(
            "/v1/npc/:nid/environment",
            Role::User,
            get(get_environment).put(put_environment),
        )
        // A portrait is a file, and needs no engine — see [`crate::images`].
        .route("/v1/npc/:nid/portrait", Role::User, put(put_portrait))
        .route("/v1/image/:iid", Role::User, get(get_image))
        // An account and its profile are the caller's own, so `User` and then
        // the record is keyed by their subject — there is no id in these paths
        // to belong to somebody else.
        .route("/v1/me", Role::User, get(me))
        .route(
            "/v1/me/profile",
            Role::User,
            get(get_profile).put(put_profile),
        )
        .route(
            "/v1/me/profile/history",
            Role::User,
            get(get_profile_history),
        )
        .route(
            "/v1/me/profile/history/:rev",
            Role::User,
            get(get_profile_revision),
        )
        .route(
            "/v1/me/profile/restore/:rev",
            Role::User,
            post(restore_profile),
        )
        .route("/v1/me/unique-name", Role::User, put(put_unique_name))
}

/// The finished router, for tests. `main` builds it from [`api`] so it can read
/// the route table on the way past.
#[cfg(test)]
pub fn router(state: Arc<Authored>) -> Router {
    api(state.clone()).into_router(state)
}

/// The caller's identity, on a route the registration already put behind
/// [`Role::User`].
///
/// The role bar is not re-checked here — [`crate::guard`] enforced it before
/// the handler ran. What this does is *retrieve* the identity, which handlers
/// need for the part a role cannot express: which records are yours. The
/// `require` call is how it gets one safely, and it doubles as a backstop if a
/// route is ever registered at a lower bar than its handler assumes.
fn caller(s: &Arc<Authored>, headers: &HeaderMap) -> Result<Identity, Box<Response>> {
    Ok(require(headers, &s.roles, Role::User)?
        .into_identity()
        .expect("`Role::User` is only reachable with an identity"))
}

// ── the cast ─────────────────────────────────────────────────────────────────

/// The caller's account id, creating the local record on first sight.
///
/// Every NPC route needs it: ownership is authorization (§8.2), so there is no
/// such thing as an anonymous read of a character.
pub async fn owner_of(
    s: &Arc<Authored>,
    headers: &HeaderMap,
) -> Result<(Identity, String), Box<Response>> {
    let id = caller(s, headers)?;
    let me = s
        .accounts
        .write()
        .await
        .upsert(&id, now_ms())
        .map_err(|e| {
            tracing::error!(error = %e, "account upsert failed");
            Box::new(err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            ))
        })?;
    // The account's public handle (`u_1a2b3c4d`), not the provider's subject.
    // The subject never leaves `accounts`, and a character record is a durable
    // artifact in a store that outlives this daemon — the last place to write
    // an identity provider's id.
    let owner = me["user_id"].as_str().map(str::to_owned).ok_or_else(|| {
        Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "account_malformed",
            "account has no user_id",
        ))
    })?;
    Ok((id, owner))
}

/// Map a registry error to a response.
///
/// `NotFound` covers both "no such id" and "not yours" on purpose: a 403 would
/// confirm that an id exists, which is enough to enumerate somebody else's cast
/// one guess at a time — the §8.3 leak by another route.
fn npc_err(e: NpcError) -> Response {
    match e {
        NpcError::NotFound => err(StatusCode::NOT_FOUND, "npc_not_found", "no such character"),
        NpcError::Invalid(field) => err(
            StatusCode::BAD_REQUEST,
            "invalid_field",
            &format!("`{field}` is missing or out of range"),
        ),
        NpcError::Persist(detail) => {
            tracing::error!(error = %detail, "npc write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "npc_write_failed",
                &detail,
            )
        }
    }
}

/// The caller's characters. No total is returned — see §8.3.
async fn list_npcs(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
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
    let mut items = s.npcs.read().await.list(&owner, &filter);
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
fn name_personality(npc: &mut Value, reg: &Registry) {
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

async fn get_npc(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        // An unparseable id is simply not a character anybody has.
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.read().await.get(npc_id, &owner) {
        Ok(mut v) => {
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

async fn create_npc(
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
    match s.npcs.write().await.create(&id, &owner, &body, now_ms()) {
        Ok(mut v) => {
            name_personality(&mut v, &*s.personalities.read().await);
            (StatusCode::CREATED, Json(v)).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// The 400 for a create that names a world or personality this daemon does not
/// have, or `None` when both resolve.
///
/// A malformed reference answers with its shape, not with `unknown_world`.
/// Both are refusals and the lookup itself is safe — an id is a `BTreeMap` key
/// here, never a path — but "no world `../../etc`" reads as an invitation to go
/// and create one, where `id::check`'s message says what is actually wrong.
async fn missing_ref(s: &Arc<Authored>, body: &Value) -> Option<Response> {
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

async fn patch_npc(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.write().await.patch(npc_id, &owner, &body, now_ms()) {
        Ok(mut v) => {
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
async fn put_npc_tags(
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
async fn put_npc_hidden(
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
async fn patch_one(s: Arc<Authored>, headers: HeaderMap, nid: String, patch: Value) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.write().await.patch(npc_id, &owner, &patch, now_ms()) {
        Ok(mut v) => {
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

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
async fn get_beliefs(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::beliefs_wire).await
}

async fn put_belief(
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

async fn delete_belief(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, bid)): Path<(String, String)>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s
        .npcs
        .write()
        .await
        .delete_belief(npc_id, &owner, &bid, now_ms())
    {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => err(StatusCode::NOT_FOUND, "belief_not_found", &bid),
        Err(e) => npc_err(e),
    }
}

async fn get_relationships(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::relationships_wire).await
}

async fn put_relationship(
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

async fn get_agency(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::agency_wire).await
}

async fn put_strategy(
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

async fn get_modulation(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::modulation_wire).await
}

async fn put_modulation(
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

async fn get_environment(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    read_npc(s, headers, nid, npcs::environment_wire).await
}

async fn put_environment(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    write_npc(s, headers, nid, move |n, id, owner, now| {
        n.put_environment(id, owner, &body, now)
    })
    .await
}

/// One read of a character, rendered by whichever view asked.
async fn read_npc(
    s: Arc<Authored>,
    headers: HeaderMap,
    nid: String,
    view: fn(&candle_conversation::persistence::record::NpcPayload) -> Value,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.read().await.visible_to(npc_id, &owner) {
        Some(n) => Json(view(n)).into_response(),
        None => err(StatusCode::NOT_FOUND, "npc_not_found", "no such character"),
    }
}

/// One authoring write, whichever it is.
///
/// The write itself is a closure over `Npcs` so the lock is taken once, here,
/// and every one of these routes supersedes the record the same way.
async fn write_npc<F>(s: Arc<Authored>, headers: HeaderMap, nid: String, write: F) -> Response
where
    F: FnOnce(&mut Npcs, u64, &str, u64) -> Result<Value, NpcError>,
{
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    let mut npcs = s.npcs.write().await;
    match write(&mut npcs, npc_id, &owner, now_ms()) {
        Ok(mut v) => {
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// Upload a portrait for a character.
///
/// The raw image as the body, not a multipart form: there is one file and no
/// other fields, so a boundary-encoded envelope would be ceremony around a byte
/// string. The format is decided from the bytes — see [`crate::images`].
///
/// The console called this "uploaded" and then dropped the file: `create()`
/// posted a name, a world, a personality and a description, and the image
/// existed only as an object URL that went away with the tab. The record has
/// carried `portrait_image_id` the whole time with nothing to put in it.
async fn put_portrait(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    body: axum::body::Bytes,
) -> Response {
    let id = match s.images.put(&body) {
        Ok(id) => id,
        Err(e) => return image_err(e),
    };
    // Recorded on the character, so the portrait survives a restart and the
    // console can find it again from the listing.
    //
    // Through `set_portrait`, not `patch`: an image id is minted here from the
    // bytes just stored, and `patch` takes what a person types. A caller must
    // not be able to name one — every id in the store is valid, so there would
    // be nothing to reject.
    write_npc(s, headers, nid, move |n, npc_id, owner, now| {
        n.set_portrait(npc_id, owner, id, "uploaded", now)
    })
    .await
}

/// Serve an uploaded image.
///
/// `User`, like everything else about a character. Not `unauthenticated`: an id
/// is a content hash and unguessable, but "unguessable" is not a permission,
/// and these are pictures of somebody's characters.
async fn get_image(State(s): State<Arc<Authored>>, Path(id): Path<String>) -> Response {
    match s.images.get(&id) {
        Ok((bytes, mime)) => {
            let mut res = bytes.into_response();
            let h = res.headers_mut();
            if let Ok(v) = HeaderValue::from_str(mime) {
                h.insert(header::CONTENT_TYPE, v);
            }
            // Content-addressed, so the bytes at an id never change and this can
            // be cached hard. The one place in this daemon where that is true.
            h.insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("public, max-age=31536000, immutable"),
            );
            res
        }
        Err(e) => image_err(e),
    }
}

fn image_err(e: ImageError) -> Response {
    match e {
        ImageError::NotAnImage => err(StatusCode::BAD_REQUEST, "not_an_image", &e.to_string()),
        ImageError::TooLarge(_) => err(StatusCode::PAYLOAD_TOO_LARGE, "too_large", &e.to_string()),
        ImageError::NotFound => err(StatusCode::NOT_FOUND, "image_not_found", &e.to_string()),
        ImageError::Io(io) => {
            // The path is not in the reply: it is the estate's own shape, and a
            // stranger who can provoke a failed write should not get a map of
            // the disk out of it.
            tracing::error!(error = %io, "image store failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "the image could not be stored",
            )
        }
    }
}

/// Delete: one superseding record with `state: "tombstoned"`. The id stays
/// taken, because the acts the character already committed still name it.
async fn delete_npc(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.write().await.delete(npc_id, &owner, now_ms()) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => npc_err(e),
    }
}

/// Who the caller is here, creating the local account on first sight.
async fn me(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match caller(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.write().await.upsert(&id, now_ms()) {
        // The role travels with the account, because the console has to know
        // it: a page that offers a Save the server will refuse is a page that
        // teaches its user the product is broken. It is reported, never
        // accepted — this is the same value the server already decided with,
        // not a claim the client gets to make.
        Ok(mut me) => {
            if let Some(map) = me.as_object_mut() {
                map.insert("role".to_string(), json!(s.roles.of(Some(&id))));
            }
            Json(me).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "account upsert failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                "could not write the account record",
            )
        }
    }
}

async fn get_profile(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match caller(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.get(&id) {
        Some(me) => Json(me["profile"].clone()).into_response(),
        // Signed in but never seen: `/v1/me` is what creates the record.
        None => err(StatusCode::NOT_FOUND, "account_not_found", "no account yet"),
    }
}

async fn put_profile(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(patch): Json<Value>,
) -> Response {
    let id = match caller(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    // Everything the caller sent is checked before any of it is merged. The
    // store repairs a wrong-typed field by blanking it — which is right for a
    // record read off disk and wrong for a live request, where it would answer
    // 200 while destroying what the author had written.
    let Some(patch) = patch.as_object() else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "expected a JSON object",
        );
    };
    if let Err(e) = accounts::check_patch(patch) {
        return match e {
            PatchError::NotText(field) => err(
                StatusCode::BAD_REQUEST,
                "bad_request",
                &format!("`{field}` must be a string"),
            ),
            PatchError::BadGender => err(
                StatusCode::BAD_REQUEST,
                "bad_gender",
                &format!("expected one of {}", accounts::GENDERS.join(", ")),
            ),
        };
    }
    match s.accounts.write().await.put_profile(&id, patch, now_ms()) {
        Ok(Some(me)) => Json(me["profile"].clone()).into_response(),
        Ok(None) => err(StatusCode::NOT_FOUND, "account_not_found", "no account yet"),
        Err(e) => {
            tracing::error!(error = %e, "profile write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            )
        }
    }
}

/// Every revision the author has had, live one first.
///
/// Worth its own route rather than a field on `/v1/me`: a profile is revised
/// rarely and read on every page load, so the history is the larger half of the
/// record and almost never the part being asked for.
async fn get_profile_history(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match caller(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.profile_history(&id) {
        Some(revisions) => Json(json!({ "revisions": revisions })).into_response(),
        None => err(StatusCode::NOT_FOUND, "account_not_found", "no account yet"),
    }
}

/// One revision in full.
///
/// Separate from the index because the index is a chooser and this is the
/// reading: with hundreds of revisions, sending every paragraph in order to
/// render a list of dates is the wrong trade by two orders of magnitude.
async fn get_profile_revision(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(rev): Path<u64>,
) -> Response {
    let id = match caller(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.profile_revision(&id, rev) {
        Some(v) => Json(v).into_response(),
        None => err(
            StatusCode::NOT_FOUND,
            "revision_not_found",
            "no such revision",
        ),
    }
}

/// Bring a superseded revision back as the live one.
///
/// `POST`, not `PUT`: it is not idempotent. Restoring the same revision twice
/// produces two new revisions, because each is an edit in its own right — the
/// second says "still this" a minute after the first, and both are true.
async fn restore_profile(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(rev): Path<u64>,
) -> Response {
    let id = match caller(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.write().await.restore_profile(&id, rev, now_ms()) {
        Ok(Some(me)) => Json(me["profile"].clone()).into_response(),
        Ok(None) => err(
            StatusCode::NOT_FOUND,
            "revision_not_found",
            "no such revision",
        ),
        Err(e) => {
            tracing::error!(error = %e, "profile restore failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            )
        }
    }
}

/// Set the name characters address this author by.
///
/// Separate from the profile because it fails differently: a profile edit is
/// prose and always succeeds, while a name can be the wrong shape or already
/// somebody else's. 409 is the interesting one — it is the only place in the
/// account surface where one author's data can block another's write, and the
/// console has to be able to say so rather than report a generic failure.
async fn put_unique_name(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let id = match caller(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(name) = body.get("unique_name").and_then(|v| v.as_str()) else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "expected a `unique_name` string",
        );
    };
    match s.accounts.write().await.put_unique_name(&id, name) {
        Ok(me) => Json(me).into_response(),
        Err(NameError::Shape(why)) => err(StatusCode::BAD_REQUEST, "bad_unique_name", why),
        Err(NameError::Taken) => err(
            StatusCode::CONFLICT,
            "name_taken",
            "that name is already in use",
        ),
        Err(NameError::NoAccount) => {
            err(StatusCode::NOT_FOUND, "account_not_found", "no account yet")
        }
        Err(NameError::Io(e)) => {
            tracing::error!(error = %e, "unique name write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            )
        }
    }
}

/// The console reads `world_id` and `personality_id`; the file knows only its own
/// name. Joining the two here keeps the id out of the authored document, where
/// it would be a second place for the same fact to live and disagree.
fn with_id(key: &str, id: &str, body: &Value) -> Value {
    let mut out = body.clone();
    if let Some(map) = out.as_object_mut() {
        map.insert(key.to_string(), json!(id));
    }
    out
}

pub fn err(status: StatusCode, code: &str, detail: &str) -> Response {
    (status, Json(json!({ "error": code, "detail": detail }))).into_response()
}

// ── the mind's files ────────────────────────────────────────────────────────

/// Everything a mind request needs, or the refusal that says why not.
///
/// The three handlers below all begin the same way — is there a mind, does the
/// path parse, which world is this scoped to — so it happens once. The scope is
/// resolved here too, which is what stops a handler forgetting to apply it.
async fn mind_request(
    s: &Arc<Authored>,
    q: &std::collections::HashMap<String, String>,
) -> Result<(std::path::PathBuf, Option<Address>, Scope), Response> {
    let Some(root) = s.mind.root() else {
        return Err(err(
            StatusCode::NOT_FOUND,
            "no_mind",
            "this daemon was started without --mind, so it has no authored content to edit",
        ));
    };
    // `?id=` is an address in the corpus — `canon/ammo/bolt` — not a path. The
    // absent case is the corpus itself.
    let addr = Address::parse(q.get("id").map(String::as_str).unwrap_or_default())
        .map_err(|e| err(StatusCode::BAD_REQUEST, "unknown_address", &e.to_string()))?;

    // `?world=` is optional. Absent means the whole mind, which is the right
    // default for an editor: a world is a lens on one corpus, and somebody
    // editing the corpus should not have to choose a lens first.
    let scope = match q.get("world").map(String::as_str).filter(|w| !w.is_empty()) {
        None => Scope::unscoped(),
        Some(wid) => match s.worlds.read().await.get(wid) {
            Some(r) => Scope::of_world(&r.body),
            None => {
                return Err(err(StatusCode::BAD_REQUEST, "unknown_world", wid));
            }
        },
    };
    Ok((root.to_path_buf(), addr, scope))
}

/// The section category of a `responses/` or `moods/` file, from the libraries
/// already in memory.
///
/// The scope needs it to apply a world's `excludes`, and reading it from the
/// loaded library rather than from disk keeps a directory listing a directory
/// listing — otherwise browsing `responses/` would open 596 files to decide
/// what to show.
fn category_lookup(s: &Arc<Authored>) -> impl Fn(&MindPath) -> Option<String> + '_ {
    move |path: &MindPath| {
        let area = path.area()?;
        let name = path.name();
        // Only a file directly inside the folder is a section.
        if path.segments().len() != 2 {
            return None;
        }
        let id = name.strip_suffix(".yaml")?;
        let library = match area {
            "responses" => &s.libraries.responses,
            "moods" => &s.libraries.moods,
            _ => return None,
        };
        library
            .sections
            .iter()
            .find(|sec| sec.id == id)
            .map(|sec| sec.category.clone())
    }
}

/// What is inside a place in the corpus.
///
/// With no `?id=`, the corpus itself — the nine sections. A caller never names
/// a directory, so there is no directory to be refused: a folder that is not
/// part of the corpus has no address at all.
async fn mind_list(
    State(s): State<Arc<Authored>>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let (root, addr, scope) = match mind_request(&s, &q).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let cat = category_lookup(&s);
    let (id, title, parent, has_text, children) = match &addr {
        None => (
            String::new(),
            "The mind".to_owned(),
            None,
            false,
            Ok(mind_catalog::sections(&root, &scope, &cat)),
        ),
        Some(a) => (
            a.as_str(),
            a.title(),
            a.parent().map(|p| p.as_str()),
            // Whether *this* has text of its own, which is what lets a topic be
            // opened as well as opened into.
            a.entry_path()
                .and_then(|p| p.resolve(&root).ok())
                .map(|f| f.is_file())
                .unwrap_or(false),
            mind_catalog::children(&root, a, &scope, &cat),
        ),
    };
    match children {
        Ok(nodes) => Json(json!({
            "id": id,
            "title": title,
            "parent": parent,
            "has_text": has_text,
            "scoped": !scope.is_unscoped(),
            "children": nodes.iter().map(mind_catalog::Node::wire).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => catalog_err(e),
    }
}

/// The addressed thing, and everything needed to act on it.
///
/// Every write handler begins the same way and the scope check is the part that
/// must not be forgotten, so it happens here once. `None` — the corpus itself —
/// is refused: there is no text at the root to read, write or remove.
async fn mind_entry_of(
    s: &Arc<Authored>,
    q: &std::collections::HashMap<String, String>,
) -> Result<(std::path::PathBuf, Address), Response> {
    let (root, addr, scope) = mind_request(s, q).await?;
    let Some(addr) = addr else {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "unknown_address",
            "name something in the mind",
        ));
    };
    let cat = category_lookup(s);
    // Asked of whichever path the address has, so a topic is checked by its
    // tag and an entry by its own file.
    let path = addr.collection_path().or_else(|| addr.entry_path());
    if let Some(p) = path {
        if !scope.admits(&p, &cat) {
            return Err(err(
                StatusCode::FORBIDDEN,
                "out_of_scope",
                "this world does not include that",
            ));
        }
    }
    Ok((root, addr))
}

/// Read the text of an entry, or a topic's overview.
async fn mind_entry(
    State(s): State<Arc<Authored>>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let (root, addr) = match mind_entry_of(&s, &q).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    match mind_catalog::read(&root, &addr) {
        Ok(d) => Json(json!({
            "id": addr.as_str(),
            "title": addr.title(),
            "text": d.text,
            "chars": d.text.chars().count(),
        }))
        .into_response(),
        Err(e) => doc_err(e),
    }
}

/// Write an entry — creating it, or replacing what is there.
///
/// `?new=1` refuses to land on something that exists, which is what "add"
/// needs: a create that overwrote would take somebody's work with no error.
async fn put_mind_entry(
    State(s): State<Arc<Authored>>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    let (root, addr) = match mind_entry_of(&s, &q).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let Some(text) = body.get("text").and_then(Value::as_str) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_field",
            "`text` is missing or is not a string",
        );
    };
    let must_be_new = matches!(q.get("new").map(String::as_str), Some("1" | "true"));
    match mind_catalog::write(&root, &addr, text, must_be_new) {
        Ok(Wrote::Created) => (
            StatusCode::CREATED,
            Json(json!({ "id": addr.as_str(), "title": addr.title(), "created": true })),
        )
            .into_response(),
        Ok(Wrote::Updated) => {
            Json(json!({ "id": addr.as_str(), "title": addr.title(), "created": false }))
                .into_response()
        }
        Err(e) => doc_err(e),
    }
}

/// The projection layers, as the schema declares them.
///
/// Read from the mind's own `projection.yaml` — the same document
/// `settings/projection` edits, through the same reader — so there is no second
/// copy of the nine layers to drift from the first. There was: the console's
/// fixture answered this route with hand-written specs, and by the time anybody
/// looked it had `action` at budget priority 95 where the schema said 100.
/// Nothing could have noticed, because the two were never compared.
///
/// The layers go out in the schema's **own vocabulary** — `name`,
/// `gather_scope`, `budget`, `groups` — rather than translated into a shape of
/// this route's own. A translation is one more place for the two to disagree,
/// and the author's words are the ones the editor shows.
async fn schema_layers(State(s): State<Arc<Authored>>) -> Response {
    let Some(root) = s.mind.root() else {
        return err(
            StatusCode::NOT_FOUND,
            "no_mind",
            "this daemon has no mind directory, so it declares no layers",
        );
    };
    // The address, and the keys that say where its parts live. Asking the
    // address rather than writing `layers` here keeps this route honest if the
    // schema is ever held under a different key.
    let addr = match Address::parse("settings/projection") {
        Ok(Some(a)) => a,
        _ => {
            tracing::error!("the projection schema has no address");
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "could not address the projection schema",
            );
        }
    };
    let Some((list_key, id_key)) = addr.parts() else {
        tracing::error!("the projection schema declares no parts");
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "io_error",
            "the projection schema declares no layers",
        );
    };
    let doc = match mind_catalog::read(root, &addr) {
        Ok(d) => d,
        Err(e) => return doc_err(e),
    };
    match mind_parts::list(&doc.text, list_key, id_key) {
        Ok(items) => Json(json!({
            "layers": items.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => err(
            StatusCode::UNPROCESSABLE_ENTITY,
            "malformed_schema",
            &e.to_string(),
        ),
    }
}

/// An entry as editable fields, rather than as its text.
///
/// A document that is not a mapping — a list, a bare scalar — has no fields, and
/// says so with `not_fields` rather than an error the console would show as a
/// failure. It is a fact about the document, and the console offers the text
/// editor instead.
async fn mind_fields(
    State(s): State<Arc<Authored>>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let (root, addr) = match mind_entry_of(&s, &q).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let doc = match mind_catalog::read(&root, &addr) {
        Ok(d) => d,
        Err(e) => return doc_err(e),
    };
    // Whichever key is the address is the one that cannot be edited: `id` for a
    // section, `name` for a projection layer.
    let id_key = addr.part().map(|p| p.id_key).unwrap_or("id");
    match mind_section::parse(&doc.text, id_key) {
        Ok(fields) => Json(json!({
            "id": addr.as_str(),
            "title": addr.title(),
            "fields": fields.iter().map(mind_section::Field::wire).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => err(
            StatusCode::UNPROCESSABLE_ENTITY,
            "not_fields",
            &e.to_string(),
        ),
    }
}

/// Save an entry from its fields, keeping every comment in the file.
///
/// The values are patched into the document that is already there rather than
/// serialised over it — see [`crate::mind::section`]. Nothing about the file
/// changes except the values that changed, so the authoring notes above each
/// key survive an edit made by somebody who never saw them.
async fn put_mind_fields(
    State(s): State<Arc<Authored>>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    let (root, addr) = match mind_entry_of(&s, &q).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let Some(values) = body.get("values").and_then(Value::as_object) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_field",
            "`values` is missing or is not an object",
        );
    };
    let wanted = match mind_section::to_document(values) {
        Ok(m) => m,
        Err(why) => return err(StatusCode::BAD_REQUEST, "invalid_field", &why),
    };

    // A *part* — one layer of the projection schema — is written straight
    // through. What it reads as is a rendering of that layer alone, with no
    // comments of its own to protect; the splice that keeps the file's comments
    // happens where they are, when the part goes back into the document that
    // holds it. See [`crate::mind::parts`].
    if addr.part().is_some() {
        let text = match serde_yaml::to_string(&Value::Object(wanted)) {
            Ok(t) => t,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_field", &e.to_string()),
        };
        return match mind_catalog::write(&root, &addr, &text, false) {
            Ok(_) => Json(json!({ "id": addr.as_str(), "title": addr.title() })).into_response(),
            Err(e) => doc_err(e),
        };
    }

    let current = match mind_catalog::read(&root, &addr) {
        Ok(d) => d,
        Err(e) => return doc_err(e),
    };
    // `splice` returns `None` when it cannot patch — the document does not
    // parse, or its own read-back check found the result did not say what was
    // asked. Refused rather than falling back to a whole rewrite: the fallback
    // would silently cost the file its comments, which is the one thing this
    // path exists to protect.
    let Some(next) = registry::yaml_edit::splice(&current.text, &wanted) else {
        return err(
            StatusCode::CONFLICT,
            "cannot_patch",
            "this document could not be edited field by field without rewriting it, \
             which would lose its comments — edit it as text instead",
        );
    };
    match mind_catalog::write(&root, &addr, &next, false) {
        Ok(_) => Json(json!({ "id": addr.as_str(), "title": addr.title() })).into_response(),
        Err(e) => doc_err(e),
    }
}

/// Remove an entry's text.
///
/// A topic keeps whatever is inside it — those have addresses of their own, and
/// one button must not become a recursive delete.
async fn delete_mind_entry(
    State(s): State<Arc<Authored>>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let (root, addr) = match mind_entry_of(&s, &q).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    match mind_catalog::remove(&root, &addr) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => doc_err(e),
    }
}

/// A listing failure, as a status and a code the console can branch on.
fn catalog_err(e: CatalogError) -> Response {
    match e {
        CatalogError::NotFound => err(StatusCode::NOT_FOUND, "not_found", &e.to_string()),
        CatalogError::OutOfScope => err(StatusCode::FORBIDDEN, "out_of_scope", &e.to_string()),
        CatalogError::Io(io) => {
            tracing::error!(error = %io, "mind listing failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "could not read that part of the mind",
            )
        }
    }
}

/// A document failure, likewise.
fn doc_err(e: DocError) -> Response {
    match e {
        // A path error cannot come from an address that parsed — the names were
        // checked when it did — so this is a bug rather than a bad request, and
        // it is reported as one rather than blamed on the caller.
        DocError::Path(p) => {
            tracing::error!(error = %p, "an address resolved to a path the rules refuse");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "could not resolve that",
            )
        }
        DocError::NotFound => err(StatusCode::NOT_FOUND, "not_found", &e.to_string()),
        DocError::IsADirectory => err(StatusCode::BAD_REQUEST, "not_a_document", &e.to_string()),
        DocError::NotText => err(StatusCode::BAD_REQUEST, "not_text", &e.to_string()),
        DocError::TooLarge(_) => err(StatusCode::PAYLOAD_TOO_LARGE, "too_large", &e.to_string()),
        DocError::Exists => err(StatusCode::CONFLICT, "already_exists", &e.to_string()),
        DocError::CannotPatch => err(StatusCode::CONFLICT, "cannot_patch", &e.to_string()),
        DocError::Io(io) => {
            // The path is deliberately not in the reply: an I/O message can
            // carry the absolute location of the mind on disk, which is the
            // estate's internal shape and not the caller's business.
            tracing::error!(error = %io, "mind write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "could not write that",
            )
        }
    }
}

/// Wall-clock milliseconds, for stamping a record. A clock before the epoch is
/// not a reason to refuse a write, so it reads as zero.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Attach the count of living characters that name this document.
///
/// A listing figure, not an accounting one: hidden characters are excluded (see
/// [`Npcs::counts_by`]), so it is a floor rather than a total.
fn with_count(mut v: Value, counts: &BTreeMap<&str, usize>, id: &str) -> Value {
    if let Some(map) = v.as_object_mut() {
        map.insert(
            "npc_count".to_string(),
            json!(counts.get(id).copied().unwrap_or(0)),
        );
    }
    v
}

/// Whether this request may see hidden documents.
///
/// `?reveal=1` asks; **being an admin is what answers**. The parameter is a
/// request, never a grant — an unauthenticated caller appending it to the URL
/// gets the same listing as one who did not, because the role is read from the
/// gateway's headers on this request and not from anything the client says.
///
/// The console sends it while an admin holds RIGHT ALT, which is the whole of
/// the gesture: the key is a convenience for someone who already has the role,
/// so that a listing being screen-shared does not contain what a screenshot
/// should not. It is discretion, not access control — `GET /v1/world/earth`
/// still answers anyone who knows the id, exactly as it did before.
fn revealing(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    q: &std::collections::HashMap<String, String>,
) -> bool {
    if !matches!(q.get("reveal").map(String::as_str), Some("1" | "true")) {
        return false;
    }
    let id = crate::identity::identify(headers).ok();
    s.roles.of(id.as_ref()).at_least(Role::Admin)
}

/// Every world, minus the hidden ones the filter has not named.
///
/// `q` is the console's filter box. A world with `hidden: true` is left out
/// until a **whole word** of `q` names it — see [`crate::visibility`]. Filtering
/// happens here rather than in the browser because a list the client narrows is
/// a list the client was first sent in full.
async fn list_worlds(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let query = q.get("q").map(String::as_str).unwrap_or_default();
    let reveal = revealing(&s, &headers, &q);
    let npcs = s.npcs.read().await;
    let counts = npcs.counts_by(|n| n.world_id.as_str());
    let reg = s.worlds.read().await;
    let worlds: Vec<Value> = reg
        .iter()
        .filter(|r| reveal || visibility::listable(&r.id, &r.body, query))
        .map(|r| with_count(with_id("world_id", &r.id, &r.body), &counts, &r.id))
        .collect();
    Json(json!({ "worlds": worlds })).into_response()
}

async fn get_world(State(s): State<Arc<Authored>>, Path(wid): Path<String>) -> Response {
    let npcs = s.npcs.read().await;
    let counts = npcs.counts_by(|n| n.world_id.as_str());
    match s.worlds.read().await.get(&wid) {
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
/// to change the pace, `paused`, or any combination. What is absent is left
/// alone rather than defaulted, so a request that only pauses does not silently
/// reset the speed.
async fn put_world_time(
    State(s): State<Arc<Authored>>,
    Path(wid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let Some(current) = s.worlds.read().await.get(&wid).map(|r| r.body.clone()) else {
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
    if let Some(to) = body.get("world_ms").and_then(Value::as_i64) {
        clock = clock.jump_to(to, now);
    }

    let next = Value::Object(clock::with_clock(&current, clock));
    match s.worlds.write().await.put(&wid, next) {
        Ok(()) => Json(clock.wire(now)).into_response(),
        Err(e) => registry_err(e),
    }
}

// No identity parameter on the four write handlers: the registration put them
// behind `Role::Admin` and `guard` refused anything below it before this ran.
// Taking the headers here only to re-derive an answer already given is how two
// places end up disagreeing about one rule.
/// The section libraries a world's lens is assembled from.
///
/// The same answer for every world, deliberately: responses and moods are
/// ingested untagged, so every world shares both the text and its KV. The route
/// is per-world because the console asks it per-world and because a world will
/// one day be able to override a collection — when it can, this is where that
/// happens.
async fn world_collections(State(s): State<Arc<Authored>>, Path(wid): Path<String>) -> Response {
    let Some(excludes) = s
        .worlds
        .read()
        .await
        .get(&wid)
        .map(|r| collections::excludes_of(&r.body))
    else {
        return err(StatusCode::NOT_FOUND, "world_not_found", &wid);
    };
    Json(s.libraries.world_wire(&excludes)).into_response()
}

/// What a personality contributes to a projection, from its own document.
///
/// The counterpart of [`world_collections`], and the last of the two
/// collection views to stop being invented — see
/// [`collections::personality_wire`].
async fn personality_collections(
    State(s): State<Arc<Authored>>,
    Path(aid): Path<String>,
) -> Response {
    let Some(body) = s
        .personalities
        .read()
        .await
        .get(&aid)
        .map(|r| r.body.clone())
    else {
        return err(StatusCode::NOT_FOUND, "personality_not_found", &aid);
    };
    Json(collections::personality_wire(&body)).into_response()
}

async fn put_world(
    State(s): State<Arc<Authored>>,
    Path(wid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    save(&s.worlds, "world_id", &wid, body).await
}

async fn delete_world(State(s): State<Arc<Authored>>, Path(wid): Path<String>) -> Response {
    remove(&s.worlds, "world_not_found", &wid).await
}

/// Every personality, minus the hidden ones the filter has not named.
///
/// Same rule as worlds — `hidden: true` in the document, revealed by a whole
/// word of `q`. Nothing in the mind uses it yet; it is here because the two
/// listings should not answer the same question two different ways, and because
/// the day a character is meant to be discreet the flag is already the answer.
async fn list_personalities(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let query = q.get("q").map(String::as_str).unwrap_or_default();
    let reveal = revealing(&s, &headers, &q);
    let npcs = s.npcs.read().await;
    let counts = npcs.counts_by(|n| n.personality_id.as_str());
    let reg = s.personalities.read().await;
    let personalities: Vec<Value> = reg
        .iter()
        .filter(|r| reveal || visibility::listable(&r.id, &r.body, query))
        .map(|r| with_count(with_id("personality_id", &r.id, &r.body), &counts, &r.id))
        .collect();
    Json(json!({ "personalities": personalities })).into_response()
}

async fn get_personality(State(s): State<Arc<Authored>>, Path(aid): Path<String>) -> Response {
    let npcs = s.npcs.read().await;
    let counts = npcs.counts_by(|n| n.personality_id.as_str());
    match s.personalities.read().await.get(&aid) {
        Some(r) => Json(with_count(
            with_id("personality_id", &r.id, &r.body),
            &counts,
            &r.id,
        ))
        .into_response(),
        None => err(StatusCode::NOT_FOUND, "personality_not_found", &aid),
    }
}

async fn put_personality(
    State(s): State<Arc<Authored>>,
    Path(aid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    save(&s.personalities, "personality_id", &aid, body).await
}

async fn delete_personality(State(s): State<Arc<Authored>>, Path(aid): Path<String>) -> Response {
    remove(&s.personalities, "personality_not_found", &aid).await
}

/// Shared save path.
///
/// The id is taken from the URL and the body's own id field is discarded rather
/// than trusted: a document that could name its own file is a document that
/// could name somebody else's.
async fn save(reg: &RwLock<Registry>, key: &str, id: &str, mut body: Value) -> Response {
    if let Some(map) = body.as_object_mut() {
        map.remove(key);
        // `npc_count` is computed at read time, exactly like the id, and for
        // the same reason must be stripped here rather than by the client: an
        // ordinary read-edit-write cycle sends back everything it was given,
        // so a client that forgets writes a stale count into the author's file
        // and the file grows a field that disagrees with the cast forever
        // after. A caller cannot be the thing that keeps a derived value out of
        // a document.
        map.remove("npc_count");
    } else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_arguments",
            "the body must be a JSON object",
        );
    }
    match reg.write().await.put(id, body.clone()) {
        Ok(()) => Json(with_id(key, id, &body)).into_response(),
        Err(e) => registry_err(e),
    }
}

async fn remove(reg: &RwLock<Registry>, missing: &str, id: &str) -> Response {
    match reg.write().await.delete(id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => err(StatusCode::NOT_FOUND, missing, id),
        Err(e) => registry_err(e),
    }
}

/// Turn a registry failure into a response, deciding what may leave the machine.
///
/// The author's mistakes are returned verbatim, because the message is the
/// useful part — `id::check` names what is wrong with an id better than any
/// paraphrase. A server-side failure is **logged and generalised**: its detail
/// is the absolute path of a file on this host, and a stranger who can provoke
/// a failed write should not get a map of the filesystem out of it.
fn registry_err(e: PutError) -> Response {
    if e.is_callers_fault() {
        return err(StatusCode::BAD_REQUEST, "invalid_arguments", &e.to_string());
    }
    tracing::error!(error = %e, "registry write failed");
    err(
        StatusCode::INTERNAL_SERVER_ERROR,
        "write_failed",
        "the document could not be written; see the daemon log",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// A directory of this test's own.
    ///
    /// The tag names it and a counter keeps it unique. It used to be the tag
    /// plus `SystemTime::now()`, which is only as fine as the platform clock —
    /// around 15ms on Windows — so two tests sharing a tag, or starting in the
    /// same tick, were handed the same directory and wrote over each other.
    fn tmp(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-api-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // A previous run of the same PID could have left one behind.
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn app() -> Router {
        router(state(tmp("state")))
    }

    /// The subject the tests treat as an admin. Every other subject is an
    /// ordinary user, which is what makes the two easy to tell apart in a
    /// request.
    const ADMIN: &str = "google-admin";

    /// Shared state, for a test that makes several requests against one store.
    /// `router` consumes its state, so the `Arc` is what gets reused.
    fn state(base: std::path::PathBuf) -> Arc<Authored> {
        Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            Accounts::load(base.join("accounts")).unwrap(),
            // A real substrate, in the test's own directory — the registry has
            // no in-memory mode, and one that only existed for tests would be a
            // second code path exercised by nothing else.
            Npcs::load(&base).unwrap(),
            // Parsed from YAML rather than built by hand, so these tests fail
            // if the config shape the deployment writes stops matching the type
            // the daemon reads.
            serde_yaml::from_str(&format!("admins:\n  - sub: {ADMIN}\n")).unwrap(),
            // No mind, so no libraries — the collections route answers with
            // two empty collections, which is the truth for a test directory.
            crate::collections::Libraries::load(&crate::projection::Source::resolve(None).unwrap()),
            // The mind editor points at the test's own directory, so a test
            // that writes a document writes it here and nowhere near a real
            // one. `mind_state` below is the same thing with content seeded.
            Mind::new(Some(base.clone())),
            Images::new(&base),
        )
    }

    /// State whose mind directory holds a small tree to browse and edit.
    ///
    /// Seeded to the shape of the real mind — a world layer with tags, the two
    /// section folders, the two document folders — so a test asserting the
    /// world filter is asserting against the layout the filter was written for.
    fn mind_state(base: std::path::PathBuf) -> Arc<Authored> {
        for dir in [
            "layers/world/ammo",
            "layers/world/armor",
            "layers/memory/cindy-tan",
            "layers/memory/commander",
            "responses",
            "moods",
        ] {
            std::fs::create_dir_all(base.join(dir)).unwrap();
        }
        std::fs::write(base.join("layers/world/ammo/bolt.md"), "a bolt").unwrap();
        std::fs::write(base.join("layers/world/armor/plate.md"), "a plate").unwrap();
        std::fs::write(base.join("layers/memory/cindy-tan/first.md"), "hers").unwrap();
        std::fs::write(base.join("layers/memory/commander/first.md"), "his").unwrap();
        state(base)
    }

    /// The headers the gateway sets for a signed-in caller.
    ///
    /// This is the whole authentication story on this side: the gateway
    /// resolved a session cookie and named the caller. `npcd` believes it
    /// because `web` clears these on ingress unless the server declared
    /// `behind_gateway`, which is asserted in `web/tests/roles.rs` rather than
    /// here — an in-process router cannot observe its own ingress.
    fn signed_in(sub: &str) -> Vec<(&'static str, String)> {
        vec![
            ("x-tokera-user", sub.to_owned()),
            // The issuer half of the account key. A subject without one is
            // refused — see `identity::identify` — so a caller that omits this
            // is not signed in, which is what the daemon should think of a
            // gateway older than the key it is being asked for.
            ("x-tokera-provider", "google".into()),
            ("x-tokera-email", "wren@example.com".into()),
            ("x-tokera-name", "Wren S".into()),
        ]
    }

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, body)
    }

    fn get(path: &str, sub: Option<&str>) -> Request<Body> {
        let mut b = Request::builder().uri(path);
        for (k, v) in sub.map(signed_in).unwrap_or_default() {
            b = b.header(k, v);
        }
        b.body(Body::empty()).unwrap()
    }

    fn send(path: &str, method: &str, sub: &str, body: Value) -> Request<Body> {
        let mut b = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json");
        for (k, v) in signed_in(sub) {
            b = b.header(k, v);
        }
        b.body(Body::from(body.to_string())).unwrap()
    }

    /// A request whose body is bytes rather than JSON — a portrait upload.
    fn bytes(path: &str, method: &str, sub: &str, body: Vec<u8>) -> Request<Body> {
        let mut b = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/octet-stream");
        for (k, v) in signed_in(sub) {
            b = b.header(k, v);
        }
        b.body(Body::from(body)).unwrap()
    }

    /// The response body as bytes, for the routes that do not answer JSON.
    async fn call_raw(app: Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, bytes.to_vec())
    }

    #[tokio::test]
    async fn a_signed_caller_gets_an_account_created_on_first_sight() {
        let (status, me) = call(app(), get("/v1/me", Some("google-1"))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(me["email"], "wren@example.com");
        assert_eq!(me["display"], "Wren S");
        assert!(me["user_id"].as_str().unwrap().starts_with("u_"));
    }

    /// A caller the gateway did not name is anonymous, and stays that way.
    #[tokio::test]
    async fn without_the_gateways_headers_nobody_is_signed_in() {
        let (status, body) = call(app(), get("/v1/me", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"], "unauthorized");
    }

    /// The subject is the account key, so a caller who sends everything *but*
    /// a subject must not become an account — least of all the one their email
    /// happens to name.
    #[tokio::test]
    async fn a_caller_with_no_subject_cannot_borrow_an_email_to_become_someone() {
        let req = Request::builder()
            .uri("/v1/me")
            .header("x-tokera-email", "admin@tokera.com")
            .header("x-tokera-name", "Admin")
            .body(Body::empty())
            .unwrap();
        let (status, _) = call(app(), req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_profile_edit_round_trips_for_its_owner() {
        let state = state(tmp("profile"));
        let a = "google-1";

        // `/v1/me` is what creates the record.
        let (s, _) = call(router(state.clone()), get("/v1/me", Some(a))).await;
        assert_eq!(s, StatusCode::OK);

        let req = send(
            "/v1/me/profile",
            "PUT",
            a,
            json!({"description": "Ex-surveyor."}),
        );
        let (s, p) = call(router(state.clone()), req).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(p["description"], "Ex-surveyor.");
        assert_eq!(p["revision"], 1);

        let (s, p) = call(router(state.clone()), get("/v1/me/profile", Some(a))).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(p["description"], "Ex-surveyor.");

        // The superseded revision is still readable — the Save button promises
        // exactly this, and an NPC citing the old text depends on it.
        let (s, h) = call(
            router(state.clone()),
            get("/v1/me/profile/history", Some(a)),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let revisions = h["revisions"].as_array().unwrap();
        assert_eq!(revisions.len(), 2);
        assert_eq!(revisions[0]["preview"], "Ex-surveyor.");
        assert_eq!(revisions[0]["live"], true);
        assert_eq!(revisions[1]["live"], false);

        // And the superseded one reads in full from its own address.
        let (s, old) = call(router(state), get("/v1/me/profile/history/0", Some(a))).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(old["revision"], 0);
        assert_eq!(old["description"], "");
    }

    /// Restoring is an edit, not a rewind.
    ///
    /// The text comes back as a *new* revision and the one it replaced is
    /// tombstoned like any other save. Moving the counter backwards would leave
    /// two different profiles claiming one revision number, and an NPC citing
    /// the earlier would be pointing at text it never read.
    #[tokio::test]
    async fn a_restored_revision_comes_back_as_a_new_one() {
        let state = state(tmp("restore"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;

        for text in ["Surveyor.", "Ex-surveyor.", "Something else."] {
            let req = send("/v1/me/profile", "PUT", a, json!({ "description": text }));
            assert_eq!(call(router(state.clone()), req).await.0, StatusCode::OK);
        }

        // Revision 1 said "Surveyor." — bring it back.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/me/profile/restore/1")
            .header("x-tokera-user", a)
            .header("x-tokera-provider", "google")
            .body(Body::empty())
            .unwrap();
        let (s, p) = call(router(state.clone()), req).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(p["description"], "Surveyor.");
        assert_eq!(p["revision"], 4, "a restore did not advance the counter");

        // Nothing was lost: the index now holds five, all distinct revisions.
        let (_, h) = call(
            router(state.clone()),
            get("/v1/me/profile/history", Some(a)),
        )
        .await;
        let revs = h["revisions"].as_array().unwrap();
        assert_eq!(revs.len(), 5);
        let numbers: Vec<u64> = revs
            .iter()
            .map(|r| r["revision"].as_u64().unwrap())
            .collect();
        assert_eq!(numbers, vec![4, 3, 2, 1, 0], "revision numbers collided");

        // An unknown revision is a 404, not a silent no-op.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/me/profile/restore/99")
            .header("x-tokera-user", a)
            .header("x-tokera-provider", "google")
            .body(Body::empty())
            .unwrap();
        let (s, e) = call(router(state), req).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(e["error"], "revision_not_found");
    }

    /// One author cannot read or restore another's revisions — the URL names a
    /// revision, never a person.
    #[tokio::test]
    async fn revisions_belong_to_the_caller_and_nobody_else() {
        let state = state(tmp("revowner"));
        let (a1, a2) = ("google-1", "google-2");
        call(router(state.clone()), get("/v1/me", Some(a1))).await;
        call(router(state.clone()), get("/v1/me", Some(a2))).await;

        let req = send("/v1/me/profile", "PUT", a1, json!({"description": "mine"}));
        call(router(state.clone()), req).await;

        // The second author's revision 1 does not exist, however the first's does.
        let (s, _) = call(
            router(state.clone()),
            get("/v1/me/profile/history/1", Some(a2)),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/me/profile/restore/1")
            .header("x-tokera-user", a2)
            .header("x-tokera-provider", "google")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            call(router(state.clone()), req).await.0,
            StatusCode::NOT_FOUND
        );

        // And the first author's text is untouched by any of it.
        let (_, p) = call(router(state), get("/v1/me/profile", Some(a1))).await;
        assert_eq!(p["description"], "mine");
    }

    #[tokio::test]
    async fn the_history_is_no_more_readable_than_the_profile_it_belongs_to() {
        let (status, _) = call(app(), get("/v1/me/profile/history", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    /// Two accounts must not see each other's profile.
    #[tokio::test]
    async fn one_signed_in_user_cannot_read_or_write_another() {
        let state = state(tmp("two"));
        let (a1, a2) = ("google-1", "google-2");

        call(router(state.clone()), get("/v1/me", Some(a1))).await;
        call(router(state.clone()), get("/v1/me", Some(a2))).await;

        let req = send("/v1/me/profile", "PUT", a1, json!({"description": "mine"}));
        call(router(state.clone()), req).await;

        // The second user's profile is untouched — there is no id in the URL to
        // point at somebody else in the first place, which is the design.
        let (_, p2) = call(router(state), get("/v1/me/profile", Some(a2))).await;
        assert_eq!(p2["description"], "");
    }

    fn set_name(a: &str, name: &str) -> Request<Body> {
        send(
            "/v1/me/unique-name",
            "PUT",
            a,
            json!({ "unique_name": name }),
        )
    }

    #[tokio::test]
    async fn a_unique_name_is_the_authors_to_set_and_survives_the_next_sign_in() {
        let state = state(tmp("uname"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;

        let (s, me) = call(router(state.clone()), set_name(a, "ridge-walker")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(me["unique_name"], "ridge-walker");

        // The provider re-asserts `Wren S` on every sign-in; the chosen name
        // must not be reverted by it.
        let (_, me) = call(router(state), get("/v1/me", Some(a))).await;
        assert_eq!(me["unique_name"], "ridge-walker");
        assert_eq!(me["display"], "Wren S");
    }

    /// The one place one author's data can refuse another's write. A character
    /// addresses a person by this name, so two of them is an ambiguous target.
    #[tokio::test]
    async fn two_authors_cannot_share_one_name() {
        let state = state(tmp("clash"));
        let (a1, a2) = ("google-1", "google-2");
        call(router(state.clone()), get("/v1/me", Some(a1))).await;
        call(router(state.clone()), get("/v1/me", Some(a2))).await;

        let (s, _) = call(router(state.clone()), set_name(a1, "ridge-walker")).await;
        assert_eq!(s, StatusCode::OK);

        // Different case, same address.
        let (s, body) = call(router(state.clone()), set_name(a2, "Ridge-Walker")).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_eq!(body["error"], "name_taken");

        // Setting your own name to what it already is is not a clash with
        // yourself.
        let (s, _) = call(router(state), set_name(a1, "ridge-walker")).await;
        assert_eq!(s, StatusCode::OK);
    }

    #[tokio::test]
    async fn a_malformed_name_is_refused_with_a_reason_worth_showing() {
        let state = state(tmp("shape"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;

        let too_long = "w".repeat(25);
        for bad in ["x", "-wren", "wren-", "wren s", "wrén", too_long.as_str()] {
            let (s, body) = call(router(state.clone()), set_name(a, bad)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad:?} should be refused");
            assert_eq!(body["error"], "bad_unique_name");
            assert!(
                body["detail"].as_str().is_some_and(|d| !d.is_empty()),
                "{bad:?} refused without saying why"
            );
        }
    }

    /// `gender` is a choice, so the API enforces the same two options the
    /// console offers — a value a character cannot read must not be storable
    /// just because the caller skipped the GUI.
    #[tokio::test]
    async fn gender_takes_only_the_values_a_character_can_read() {
        let state = state(tmp("gender"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;

        for good in ["Male", "Female", ""] {
            let req = send("/v1/me/profile", "PUT", a, json!({ "gender": good }));
            let (s, p) = call(router(state.clone()), req).await;
            assert_eq!(s, StatusCode::OK, "{good:?} should be accepted");
            assert_eq!(p["gender"], good);
        }

        for bad in ["male", "M", "Other", "they/them", "—"] {
            let req = send("/v1/me/profile", "PUT", a, json!({ "gender": bad }));
            let (s, body) = call(router(state.clone()), req).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad:?} should be refused");
            assert_eq!(body["error"], "bad_gender");
        }

        // And a refusal must not have written a revision on its way out.
        let (_, p) = call(router(state), get("/v1/me/profile", Some(a))).await;
        assert_eq!(p["revision"], 3, "a rejected write bumped the revision");
    }

    /// The wrong *kind* of value is refused too, and does not quietly erase
    /// what was there.
    ///
    /// Only strings used to be checked, so `{"gender": null}` and
    /// `{"gender": 3}` walked past the guard, merged, and were blanked by the
    /// store's repair pass — 200 OK, choice destroyed. The same held for the
    /// prose fields: `{"description": 42}` wiped a paragraph and reported
    /// success.
    #[tokio::test]
    async fn a_wrong_typed_field_is_refused_and_changes_nothing() {
        let state = state(tmp("types"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;

        let seed = send(
            "/v1/me/profile",
            "PUT",
            a,
            json!({"gender": "Male", "description": "Ex-surveyor."}),
        );
        assert_eq!(call(router(state.clone()), seed).await.0, StatusCode::OK);

        for bad in [
            json!({"gender": null}),
            json!({"gender": 3}),
            json!({"description": 42}),
            json!({"history": ["a"]}),
        ] {
            let req = send("/v1/me/profile", "PUT", a, bad.clone());
            let (s, body) = call(router(state.clone()), req).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad} was accepted");
            assert_eq!(body["error"], "bad_request", "{bad}");
        }

        // Nothing was merged, nothing was blanked, no revision was spent.
        let (_, p) = call(router(state), get("/v1/me/profile", Some(a))).await;
        assert_eq!(p["gender"], "Male");
        assert_eq!(p["description"], "Ex-surveyor.");
        assert_eq!(p["revision"], 1);
    }

    /// A body that is not an object is refused before it can retire a revision.
    #[tokio::test]
    async fn a_body_that_is_not_an_object_is_refused() {
        let state = state(tmp("shape2"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;

        for body in [json!([]), json!("x"), json!(3), json!(null)] {
            let req = send("/v1/me/profile", "PUT", a, body.clone());
            let (s, e) = call(router(state.clone()), req).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body} was accepted");
            assert_eq!(e["error"], "bad_request");
        }

        // The live revision is untouched and history has not grown.
        let (_, p) = call(router(state.clone()), get("/v1/me/profile", Some(a))).await;
        assert_eq!(p["revision"], 0);
        let (_, h) = call(router(state), get("/v1/me/profile/history", Some(a))).await;
        assert_eq!(h["revisions"].as_array().unwrap().len(), 1);
    }

    /// Every write is behind the same gate, not just the reads.
    #[tokio::test]
    async fn setting_a_name_needs_the_gateway_to_have_named_you() {
        let req = Request::builder()
            .method("PUT")
            .uri("/v1/me/unique-name")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"unique_name":"impostor"}"#))
            .unwrap();
        let (status, _) = call(app(), req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    /// Author a world and a personality so a character has something to name.
    /// A registry write is a real YAML file, which is the point: these are the
    /// same documents the mind ships, reached through the same route the
    /// console uses.
    ///
    /// Always as [`ADMIN`], because authoring is an admin's job now — a test
    /// that set this up as an ordinary user would be testing a path the
    /// deployment does not have.
    async fn author(state: &Arc<Authored>) {
        let w = send(
            "/v1/world/battle-cities",
            "PUT",
            ADMIN,
            json!({ "name": "Battle Cities", "public": true }),
        );
        assert_eq!(call(router(state.clone()), w).await.0, StatusCode::OK);
        // `world` is not optional: a character is written for one world and
        // every create is checked against it, so a fixture without one is a
        // personality no world can host — see
        // `a_character_can_only_be_created_in_the_world_it_belongs_to`.
        let p = send(
            "/v1/personality/commander",
            "PUT",
            ADMIN,
            json!({
                "name": "Commander",
                "world": "battle-cities",
                "anchor": "Position is read before people are.",
            }),
        );
        assert_eq!(call(router(state.clone()), p).await.0, StatusCode::OK);
    }

    /// A character names its world and personality by the slug that IS their
    /// file name. Naming one that does not exist is the author's mistake to see
    /// now, not the engine's to hit at spawn.
    #[tokio::test]
    async fn a_character_may_only_name_documents_that_exist() {
        let state = state(tmp("refs"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;
        author(&state).await;

        let ok = send(
            "/v1/npc",
            "POST",
            a,
            json!({ "name": "Varek", "world_id": "battle-cities", "personality_id": "commander" }),
        );
        let (s, npc) = call(router(state.clone()), ok).await;
        assert_eq!(s, StatusCode::CREATED, "{npc}");
        assert_eq!(npc["world_id"], "battle-cities");
        assert_eq!(npc["personality_id"], "commander");

        for (body, code) in [
            (
                json!({ "name": "A", "world_id": "atlantis", "personality_id": "commander" }),
                "unknown_world",
            ),
            (
                json!({ "name": "A", "world_id": "battle-cities", "personality_id": "wizard" }),
                "unknown_personality",
            ),
        ] {
            let (s, e) = call(router(state.clone()), send("/v1/npc", "POST", a, body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST);
            assert_eq!(e["error"], code);
        }

        // A traversal never reaches the registry lookup — it is refused for its
        // shape, by the same gate that decides what may become a file name.
        let bad = json!({ "name": "A", "world_id": "../../etc", "personality_id": "commander" });
        let (s, e) = call(router(state.clone()), send("/v1/npc", "POST", a, bad)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(e["error"], "invalid_arguments");

        // Only the one character survived all of that.
        let (_, list) = call(router(state), get("/v1/npc", Some(a))).await;
        assert_eq!(list["items"].as_array().unwrap().len(), 1);
    }

    /// **The whole route table, written down.**
    ///
    /// `guard::Api` makes it impossible to *forget* a role — the compiler
    /// requires one. This is the other half: it makes it impossible to *add*
    /// one nobody looked at. A new route fails this test until somebody writes
    /// its line here, which is a moment where the role gets chosen deliberately
    /// rather than copied from the route above it.
    ///
    /// Read the `unauthenticated` rows first. Those are the ones where being
    /// wrong is expensive, and there should never be a `put`, `post`, `patch`
    /// or `delete` among them.
    #[test]
    fn the_route_table_is_what_we_think_it_is() {
        let api = api(state(tmp("table")));
        let got: Vec<(&str, &str)> = api
            .declared()
            .iter()
            .map(|r| (r.path, r.min.as_str()))
            .collect();

        assert_eq!(
            got,
            [
                // Authored content: readable by anyone, writable by admins.
                ("/v1/world", "unauthenticated"),
                ("/v1/world/:wid", "unauthenticated"),
                ("/v1/world/:wid/collections", "unauthenticated"),
                ("/v1/world/:wid", "admin"),
                ("/v1/world/:wid/time", "admin"),
                ("/v1/personality", "unauthenticated"),
                ("/v1/personality/:aid", "unauthenticated"),
                ("/v1/personality/:aid", "admin"),
                // Read-only, from the personality's own document — open for the
                // same reason reading the document is.
                ("/v1/personality/:aid/collections", "unauthenticated"),
                // The authored corpus. `user` to read rather than open, because
                // this one *enumerates* — see the route's own comment.
                ("/v1/mind/list", "user"),
                ("/v1/mind/entry", "user"),
                ("/v1/mind/entry", "admin"),
                ("/v1/mind/fields", "user"),
                ("/v1/mind/fields", "admin"),
                // The schema's own layers, read from the mind. `user` for the
                // same reason the three rows above are: it is a reading of the
                // corpus, and reading the corpus is a signed-in act.
                ("/v1/schema/layers", "user"),
                // The cast: signed in, then ownership per record.
                ("/v1/npc", "user"),
                ("/v1/npc/:nid", "user"),
                ("/v1/npc/:nid/tags", "user"),
                ("/v1/npc/:nid/hidden", "user"),
                // The authoring plane. Every one of these is a write to the
                // caller's own character, so `user` plus the ownership check
                // inside — never `admin`, which would mean an operator could
                // not author their own cast.
                ("/v1/npc/:nid/beliefs", "user"),
                ("/v1/npc/:nid/beliefs/:bid", "user"),
                ("/v1/npc/:nid/relationships", "user"),
                ("/v1/npc/:nid/relationships/:eid", "user"),
                ("/v1/npc/:nid/agency", "user"),
                ("/v1/npc/:nid/agency/:sid", "user"),
                ("/v1/npc/:nid/modulation", "user"),
                ("/v1/npc/:nid/environment", "user"),
                // A portrait, and the bytes back. `user` rather than open: an
                // id is a content hash and unguessable, and unguessable is not
                // a permission.
                ("/v1/npc/:nid/portrait", "user"),
                ("/v1/image/:iid", "user"),
                // The caller's own account.
                ("/v1/me", "user"),
                ("/v1/me/profile", "user"),
                ("/v1/me/profile/history", "user"),
                ("/v1/me/profile/history/:rev", "user"),
                ("/v1/me/profile/restore/:rev", "user"),
                ("/v1/me/unique-name", "user"),
            ]
        );
    }

    /// **The finding this closes.** Every route that changes a file on disk was
    /// reachable with no headers at all: `PUT /v1/world/x` returned 200 and
    /// wrote into the mind, `DELETE` returned 204 and removed it. The mind is
    /// not under version control, so that was unrecoverable prose one request
    /// away from anybody who could reach the port.
    #[tokio::test]
    async fn only_an_admin_may_change_authored_content() {
        let state = state(tmp("roles"));
        author(&state).await;

        let doc = json!({ "name": "Renamed" });
        for (path, kind) in [
            ("/v1/world/battle-cities", "world"),
            ("/v1/personality/commander", "personality"),
        ] {
            // Anonymous: 401, because signing in is the thing that would help.
            let anon = Request::builder()
                .method("PUT")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(doc.to_string()))
                .unwrap();
            let (s, e) = call(router(state.clone()), anon).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "{kind} PUT was open: {e}");
            assert_eq!(e["error"], "unauthorized");

            // Signed in and not an admin: 403, because signing in again will
            // not help and telling them to try wastes their afternoon.
            let (s, e) = call(
                router(state.clone()),
                send(path, "PUT", "google-1", doc.clone()),
            )
            .await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{kind} PUT allowed a user: {e}");
            assert_eq!(e["error"], "forbidden");
            assert_eq!(e["required_role"], "admin");
            assert_eq!(e["role"], "user");

            // Delete is the same gate. It is the more destructive of the two.
            let (s, _) = call(
                router(state.clone()),
                send(path, "DELETE", "google-1", json!({})),
            )
            .await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{kind} DELETE allowed a user");

            // The admin may.
            let (s, _) = call(router(state.clone()), send(path, "PUT", ADMIN, doc.clone())).await;
            assert_eq!(s, StatusCode::OK, "{kind} PUT refused the admin");
        }

        // And nothing above changed what anybody can read.
        let (s, _) = call(router(state.clone()), get("/v1/world", None)).await;
        assert_eq!(s, StatusCode::OK, "reading became privileged");
        let (s, _) = call(router(state), get("/v1/personality/commander", None)).await;
        assert_eq!(s, StatusCode::OK);
    }

    /// **Creation and editing, end to end, through the substrate.**
    ///
    /// Every write appends one record keyed by `npc_id` and the newest wins on
    /// replay, so this asserts the *reload* rather than the response: a
    /// character edited and then read back from a freshly opened store is the
    /// only thing that proves the edit was durable rather than in memory.
    #[tokio::test]
    async fn a_character_is_created_and_edited_in_the_substrate() {
        let dir = tmp("cast");
        let st = state(dir.clone());
        let a = "google-1";
        call(router(st.clone()), get("/v1/me", Some(a))).await;
        author(&st).await;

        // Create.
        let (s, npc) = call(
            router(st.clone()),
            send(
                "/v1/npc",
                "POST",
                a,
                json!({
                    "name": "Varek", "world_id": "battle-cities",
                    "personality_id": "commander",
                    "persona_description": "Fifty-three, a former staff sergeant."
                }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{npc}");
        let id = npc["npc_id"].as_str().unwrap().to_string();
        assert_eq!(npc["revision"], 1);
        assert_eq!(
            npc["state"], "idle",
            "a character that has never ticked is idle"
        );

        // Edit the authored fields — name, persona, metabolism, state.
        let (s, npc) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}"),
                "PATCH",
                a,
                json!({
                    "name": "Varek the Elder",
                    "persona_description": "Sixty-one now, and slower.",
                    "heartbeat_ms": 300_000,
                    "salience_gate": 0.7,
                    "state": "suspended",
                    "environment_enabled": false
                }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{npc}");
        assert_eq!(npc["revision"], 2, "an edit supersedes rather than mutates");

        // The two single-field routes the console edits from a chip and a
        // checkbox. They are PATCH underneath, so they advance the revision too.
        let (s, npc) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/tags"),
                "PUT",
                a,
                json!({ "tags": ["north", "command"] }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{npc}");
        assert_eq!(npc["tags"], json!(["north", "command"]));

        let (s, npc) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/hidden"),
                "PUT",
                a,
                json!({ "hidden": true }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{npc}");
        assert_eq!(npc["hidden"], true);

        // **Reopen the store.** Everything above must come back off the log.
        let again = state(dir);
        let (s, back) = call(
            router(again.clone()),
            get(&format!("/v1/npc/{id}"), Some(a)),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{back}");
        assert_eq!(back["name"], "Varek the Elder");
        assert_eq!(back["persona"]["description"], "Sixty-one now, and slower.");
        assert_eq!(back["tick"]["heartbeat_ms"], 300_000);
        assert_eq!(back["tick"]["salience_gate"], 0.7);
        assert_eq!(back["state"], "suspended");
        assert_eq!(back["environment_enabled"], false);
        assert_eq!(back["tags"], json!(["north", "command"]));
        assert_eq!(back["hidden"], true);
        assert_eq!(
            back["personality_name"], "Commander",
            "joined from the document"
        );

        // Hidden keeps it out of the default listing and a tag brings it back —
        // discretion, not encryption (§8.3).
        let (_, list) = call(router(again.clone()), get("/v1/npc", Some(a))).await;
        assert!(list["items"].as_array().unwrap().is_empty());
        let (_, list) = call(
            router(again.clone()),
            get("/v1/npc?tag=north&hidden=1", Some(a)),
        )
        .await;
        assert_eq!(list["items"].as_array().unwrap().len(), 1);

        // Delete tombstones it, and the tombstone survives a reload too.
        let (s, _) = call(
            router(again.clone()),
            send(&format!("/v1/npc/{id}"), "DELETE", a, json!({})),
        )
        .await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        let (s, _) = call(router(again), get(&format!("/v1/npc/{id}"), Some(a))).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    /// The single-field routes are the caller's own character only, like every
    /// other write on a record.
    #[tokio::test]
    async fn tags_and_hidden_belong_to_the_owner() {
        let st = state(tmp("own-edit"));
        let (a, b) = ("google-1", "google-2");
        call(router(st.clone()), get("/v1/me", Some(a))).await;
        call(router(st.clone()), get("/v1/me", Some(b))).await;
        author(&st).await;

        let (_, npc) = call(
            router(st.clone()),
            send(
                "/v1/npc",
                "POST",
                a,
                json!({
                    "name": "Varek", "world_id": "battle-cities", "personality_id": "commander"
                }),
            ),
        )
        .await;
        let id = npc["npc_id"].as_str().unwrap().to_string();

        for (path, body) in [
            (format!("/v1/npc/{id}/tags"), json!({ "tags": ["stolen"] })),
            (format!("/v1/npc/{id}/hidden"), json!({ "hidden": true })),
        ] {
            let (s, _) = call(router(st.clone()), send(&path, "PUT", b, body)).await;
            // 404, never 403: a 403 confirms the id exists, which is enough to
            // enumerate somebody's cast one guess at a time.
            assert_eq!(
                s,
                StatusCode::NOT_FOUND,
                "{path} was writable by a stranger"
            );
        }

        // And a malformed value is refused rather than written.
        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/hidden"),
                "PUT",
                a,
                json!({ "hidden": "yes" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }

    /// **Hidden worlds, and the whole word that reveals them.**
    ///
    /// A prefix must not work. Incremental discovery — type a letter, see what
    /// appears — is exactly the browsing the flag exists to prevent, so this
    /// walks the prefixes of the word and asserts each one finds nothing.
    #[tokio::test]
    async fn a_hidden_world_is_listed_only_when_a_whole_word_names_it() {
        let st = state(tmp("hidden"));
        for (id, body) in [
            ("battle-cities", json!({ "name": "Battle Cities" })),
            ("earth", json!({ "name": "Earth", "hidden": true })),
        ] {
            let r = send(&format!("/v1/world/{id}"), "PUT", ADMIN, body);
            assert_eq!(call(router(st.clone()), r).await.0, StatusCode::OK);
        }

        let ids = |v: &Value| -> Vec<String> {
            v["worlds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| w["world_id"].as_str().unwrap().to_string())
                .collect()
        };

        // Unfiltered: the visible one, and only that.
        for q in ["", "?q="] {
            let (_, v) = call(router(st.clone()), get(&format!("/v1/world{q}"), None)).await;
            assert_eq!(ids(&v), ["battle-cities"], "`{q}` revealed a hidden world");
        }

        // **The property that matters.** No prefix of the hidden world's name
        // ever produces it, so typing letters and watching the list cannot
        // discover it — which is what makes filtering-as-you-type safe here.
        //
        // The visible world may well appear against some of these: `e` is a
        // substring of `battle-cities`. That is the ordinary filter doing its
        // job, and it is the *hidden* one this is about.
        for q in ["?q=e", "?q=ea", "?q=ear", "?q=eart", "?q=arth", "?q=rth"] {
            let (_, v) = call(router(st.clone()), get(&format!("/v1/world{q}"), None)).await;
            assert!(
                !ids(&v).contains(&"earth".to_string()),
                "`{q}` revealed a hidden world"
            );
        }

        // The whole word, in any casing, and alongside another term.
        for q in ["?q=earth", "?q=Earth", "?q=sydney%20earth"] {
            let (_, v) = call(router(st.clone()), get(&format!("/v1/world{q}"), None)).await;
            let got = ids(&v);
            assert!(
                got.contains(&"earth".to_string()),
                "`{q}` did not reveal it"
            );
        }

        // And a visible world still narrows on an ordinary substring, which is
        // what the filter box does the rest of the time.
        let (_, v) = call(router(st.clone()), get("/v1/world?q=batt", None)).await;
        assert_eq!(ids(&v), ["battle-cities"]);

        // Knowing the id is the same as knowing the word: a direct fetch works,
        // which is the documented shape of this — discretion, not access
        // control.
        let (s, _) = call(router(st), get("/v1/world/earth", None)).await;
        assert_eq!(s, StatusCode::OK);
    }

    /// A character can only be created in the world it belongs to.
    ///
    /// The console filters the pairing out of the create form, but the console
    /// is presentation. This is the daemon refusing it — a create posted by
    /// curl, or by a page held open while a personality was re-homed, must not
    /// write a character into a setting it has no canon for.
    #[tokio::test]
    async fn a_character_can_only_be_created_in_the_world_it_belongs_to() {
        let st = state(tmp("hostable"));
        for (path, body) in [
            (
                "/v1/world/earth",
                json!({ "name": "Earth", "personalities": ["cindy-tan"] }),
            ),
            (
                "/v1/world/battle-cities",
                json!({ "name": "Battle Cities", "personalities": ["commander"] }),
            ),
            // Casts nobody, so it admits everybody — the standing default, and
            // what keeps the first world to declare a cast from emptying the
            // rest.
            ("/v1/world/sandbox", json!({ "name": "Sandbox" })),
            ("/v1/personality/cindy-tan", json!({ "name": "Cindy Tan" })),
            ("/v1/personality/commander", json!({ "name": "Commander" })),
        ] {
            let r = send(path, "PUT", ADMIN, body);
            assert_eq!(
                call(router(st.clone()), r).await.0,
                StatusCode::OK,
                "{path}"
            );
        }

        let create = |world: &str, personality: &str| {
            send(
                "/v1/npc",
                "POST",
                ADMIN,
                json!({ "name": "X", "world_id": world, "personality_id": personality }),
            )
        };

        // Each in its own world.
        for (world, who) in [("earth", "cindy-tan"), ("battle-cities", "commander")] {
            let (s, _) = call(router(st.clone()), create(world, who)).await;
            assert_eq!(s, StatusCode::CREATED, "{who} refused in {world}");
        }

        // And in the other's, refused both ways round — naming both sides, so
        // the reader is not left guessing which of the two they got wrong.
        for (world, who) in [("battle-cities", "cindy-tan"), ("earth", "commander")] {
            let (s, v) = call(router(st.clone()), create(world, who)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{world} cast {who}");
            assert_eq!(v["error"], "personality_not_of_world");
            let detail = v["detail"].as_str().unwrap_or_default();
            for expected in [world, who] {
                assert!(
                    detail.contains(expected),
                    "detail omitted {expected}: {detail}"
                );
            }
        }

        // A world that casts nobody admits everybody. This is the default the
        // cast worlds are an exception to, and it is what makes adding the key
        // to one world safe: it must not turn a full list into an empty one
        // everywhere it has not been written yet.
        for who in ["cindy-tan", "commander"] {
            let (s, _) = call(router(st.clone()), create("sandbox", who)).await;
            assert_eq!(s, StatusCode::CREATED, "sandbox refused {who}");
        }
    }

    // ── the corpus ──────────────────────────────────────────────────────────

    /// With no address, the corpus is its nine sections — not a folder listing.
    #[tokio::test]
    async fn the_corpus_opens_on_its_sections() {
        let base = tmp("mind-sections");
        let st = mind_state(base);
        let (s, v) = call(router(st), get("/v1/mind/list", Some("google-1"))).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["title"], "The mind");
        assert_eq!(v["parent"], Value::Null);
        assert_eq!(v["scoped"], false, "no ?world= means the whole mind");

        let ids: Vec<&str> = v["children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            [
                "canon",
                "agency",
                "beliefs",
                "memory",
                "responses",
                "moods",
                "characters",
                "worlds",
                "settings"
            ]
        );
        assert_eq!(v["children"][0]["title"], "World knowledge");
        assert!(
            v["children"][0]["blurb"].is_string(),
            "a section says what it is"
        );
    }

    /// **The abstraction, asserted.** Nothing on the wire is a file: no
    /// extension, no directory, no `layers/`. A reader sees topics and entries.
    #[tokio::test]
    async fn nothing_on_the_wire_is_a_file() {
        let base = tmp("mind-nofiles");
        let st = mind_state(base);

        for place in ["", "canon", "canon/ammo", "characters", "settings"] {
            let (s, v) = call(
                router(st.clone()),
                get(&format!("/v1/mind/list?id={place}"), Some("google-1")),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "listing {place}");
            let body = v.to_string();
            for leak in [
                ".md",
                ".yaml",
                "layers/",
                "personalities/",
                "is_dir",
                "\"ext\"",
            ] {
                assert!(
                    !body.contains(leak),
                    "`{leak}` leaked while listing {place}"
                );
            }
        }

        // And the same of an entry.
        let (_, v) = call(
            router(st),
            get("/v1/mind/entry?id=canon/ammo/bolt", Some("google-1")),
        )
        .await;
        assert_eq!(v["id"], "canon/ammo/bolt");
        assert_eq!(v["title"], "Bolt");
        assert!(
            v["chars"].is_number(),
            "a length, not a byte count of a file"
        );
        let body = v.to_string();
        for leak in [".md", "layers/", "bytes"] {
            assert!(!body.contains(leak), "`{leak}` leaked from an entry");
        }
    }

    /// A topic stored as a page beside a folder is one thing with text of its
    /// own, and the console never learns there were two files.
    #[tokio::test]
    async fn a_topic_is_one_thing_that_both_holds_and_says() {
        let base = tmp("mind-topic");
        let st = mind_state(base.clone());
        std::fs::write(base.join("layers/world/ammo.md"), "all about ammo").unwrap();

        let (_, v) = call(
            router(st.clone()),
            get("/v1/mind/list?id=canon", Some("google-1")),
        )
        .await;
        let ammo = v["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == "canon/ammo")
            .expect("listed");
        assert_eq!(ammo["kind"], "collection");
        assert_eq!(ammo["count"], 1, "one entry inside");
        assert_eq!(ammo["has_text"], true, "and an overview of its own");
        assert_eq!(
            v["children"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["title"] == "Ammo")
                .count(),
            1,
            "one row, not a folder and a file"
        );

        // Opening it gives the overview; listing it gives the entries.
        let (_, v) = call(
            router(st.clone()),
            get("/v1/mind/entry?id=canon/ammo", Some("google-1")),
        )
        .await;
        assert_eq!(v["text"], "all about ammo");
        let (_, v) = call(
            router(st),
            get("/v1/mind/list?id=canon/ammo", Some("google-1")),
        )
        .await;
        assert_eq!(v["has_text"], true);
        assert_eq!(v["children"][0]["id"], "canon/ammo/bolt");
    }

    /// The whole round trip, through the API, ending on disk — which is the
    /// requirement: a save that only updated a listing would be a save that did
    /// not happen.
    #[tokio::test]
    async fn an_entry_is_read_written_and_removed_through_the_api() {
        let base = tmp("mind-doc");
        let st = mind_state(base.clone());
        let entry = "/v1/mind/entry?id=canon/ammo/bolt";

        let (s, v) = call(router(st.clone()), get(entry, Some("google-1"))).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["text"], "a bolt");
        assert_eq!(v["title"], "Bolt");

        let (s, v) = call(
            router(st.clone()),
            send(entry, "PUT", ADMIN, json!({ "text": "a longer bolt" })),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["created"], false);
        // The address became a file, and the bytes reached the disk — the whole
        // requirement, asserted where the abstraction meets the filesystem.
        assert_eq!(
            std::fs::read_to_string(base.join("layers/world/ammo/bolt.md")).unwrap(),
            "a longer bolt"
        );

        let (s, _) = call(router(st.clone()), send(entry, "DELETE", ADMIN, json!({}))).await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        assert!(!base.join("layers/world/ammo/bolt.md").exists());
        let (s, _) = call(router(st), get(entry, Some("google-1"))).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    /// Adding an item creates the file and any folder above it, and `?new=1`
    /// refuses to land on something that is already there.
    #[tokio::test]
    async fn adding_an_item_creates_it_and_will_not_overwrite() {
        let base = tmp("mind-add");
        let st = mind_state(base.clone());
        // No extension anywhere: the caller names the thing, and the section
        // decides how it is stored.
        let entry = "/v1/mind/entry?id=canon/ammo/shell&new=1";

        let (s, v) = call(
            router(st.clone()),
            send(entry, "PUT", ADMIN, json!({ "text": "a shell" })),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        assert_eq!(v["created"], true);
        assert_eq!(v["id"], "canon/ammo/shell");
        assert_eq!(
            std::fs::read_to_string(base.join("layers/world/ammo/shell.md")).unwrap(),
            "a shell"
        );

        // Again, and it is refused rather than taking the first one's place.
        let (s, v) = call(
            router(st.clone()),
            send(entry, "PUT", ADMIN, json!({ "text": "different" })),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_eq!(v["error"], "already_exists");
        assert_eq!(
            std::fs::read_to_string(base.join("layers/world/ammo/shell.md")).unwrap(),
            "a shell",
            "the first write survived"
        );

        // A whole new topic, in one call.
        let (s, _) = call(
            router(st.clone()),
            send(
                "/v1/mind/entry?id=canon/brand/new&new=1",
                "PUT",
                ADMIN,
                json!({ "text": "x" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        assert!(base.join("layers/world/brand/new.md").is_file());

        // The section decides the format, so the same shape of address lands as
        // YAML where the section is structured.
        let (s, _) = call(
            router(st),
            send(
                "/v1/mind/entry?id=characters/new-hire&new=1",
                "PUT",
                ADMIN,
                json!({ "text": "id: new-hire" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        assert!(base.join("personalities/new-hire.yaml").is_file());
    }

    /// **The security property.** Every spelling of "leave the mind" is refused
    /// at the API, and the file outside it is never touched.
    #[tokio::test]
    async fn the_mind_editor_cannot_reach_outside_the_mind() {
        let base = tmp("mind-escape");
        let st = mind_state(base.clone());
        let outside = base.parent().unwrap().join("npcd-api-escape-witness.md");
        let _ = std::fs::remove_file(&outside);
        std::fs::write(&outside, "untouched").unwrap();

        // Two families, and the address makes them different failures.
        //
        // The first four are not addresses at all — `layers` and `..` are not
        // sections, so there is nothing to resolve. The rest parse as a section
        // and a name, and are stopped by the name rules that outlive the
        // abstraction: an address is a nicer spelling of a path, never a way
        // around one.
        for id in [
            "..",
            "layers/world/ammo/bolt",
            "../npcd-api-escape-witness",
            "/etc/passwd",
            "c:/windows/system32/drivers/etc/hosts",
            "canon/../../npcd-api-escape-witness",
            "canon/..%2F..%2Fnpcd-api-escape-witness",
            "canon/..\\..\\npcd-api-escape-witness",
            "canon/nul",
            "settings/package",
        ] {
            let uri = format!("/v1/mind/entry?id={id}");
            let (s, _) = call(router(st.clone()), get(&uri, Some("google-1"))).await;
            assert!(
                s == StatusCode::BAD_REQUEST || s == StatusCode::NOT_FOUND,
                "GET {id} answered {s}"
            );
            let (s, _) = call(
                router(st.clone()),
                send(&uri, "PUT", ADMIN, json!({ "text": "owned" })),
            )
            .await;
            assert!(
                s == StatusCode::BAD_REQUEST || s == StatusCode::NOT_FOUND,
                "PUT {id} answered {s}"
            );
        }

        assert_eq!(
            std::fs::read_to_string(&outside).unwrap(),
            "untouched",
            "a file outside the mind was modified"
        );
        let _ = std::fs::remove_file(&outside);
    }

    /// Only `.md` and `.yaml`. A PUT that could name any extension would let
    /// the editor drop a file the daemon later reads as something else.
    #[tokio::test]
    async fn an_address_cannot_choose_how_it_is_stored() {
        let base = tmp("mind-ext");
        let st = mind_state(base.clone());

        // A caller cannot ask for an extension, because an address has no place
        // to put one — `canon/x.exe` is a *name*, so it becomes `x.exe.md` and
        // is still markdown in the canon folder. Nothing executable can be
        // written anywhere, by anyone, by any spelling.
        let (s, _) = call(
            router(st.clone()),
            send(
                "/v1/mind/entry?id=canon/x.exe&new=1",
                "PUT",
                ADMIN,
                json!({ "text": "x" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        assert!(base.join("layers/world/x.exe.md").is_file());
        assert!(!base.join("layers/world/x.exe").exists(), "an executable");

        // And a section that stores YAML stores YAML, whatever the name says.
        let (s, _) = call(
            router(st),
            send(
                "/v1/mind/entry?id=characters/y.md&new=1",
                "PUT",
                ADMIN,
                json!({ "text": "id: y" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        assert!(base.join("personalities/y.md.yaml").is_file());
    }

    /// Reading needs a session; writing needs an admin. The mind is not under
    /// version control, so a bad write is prose somebody wrote, gone.
    #[tokio::test]
    async fn reading_needs_a_session_and_writing_needs_an_admin() {
        let base = tmp("mind-roles");
        let st = mind_state(base);
        let entry = "/v1/mind/entry?id=canon/ammo/bolt";

        // Anonymous: refused even to read, because listing enumerates.
        let (s, _) = call(router(st.clone()), get(entry, None)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        let (s, _) = call(router(st.clone()), get("/v1/mind/list", None)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);

        // A signed-in user reads.
        let (s, _) = call(router(st.clone()), get(entry, Some("google-1"))).await;
        assert_eq!(s, StatusCode::OK);

        // And does not write.
        for (method, body) in [("PUT", json!({ "text": "no" })), ("DELETE", json!({}))] {
            let (s, _) = call(
                router(st.clone()),
                send(entry, method, "google-not-an-admin", body),
            )
            .await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{method} by a plain user");
        }
    }

    /// The world's own filters apply to the file tree, so browsing inside a
    /// world shows that world's corpus and not the whole mind.
    #[tokio::test]
    async fn a_world_scopes_what_the_corpus_shows() {
        let base = tmp("mind-scope");
        let st = mind_state(base);
        let w = send(
            "/v1/world/battle-cities",
            "PUT",
            ADMIN,
            json!({ "name": "Battle Cities", "selects": ["ammo"], "personalities": ["commander"] }),
        );
        assert_eq!(call(router(st.clone()), w).await.0, StatusCode::OK);

        let ids = |v: &Value| -> Vec<String> {
            v["children"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["id"].as_str().unwrap().to_string())
                .collect()
        };

        // `selects` gates the canon topics.
        let (s, v) = call(
            router(st.clone()),
            get(
                "/v1/mind/list?world=battle-cities&id=canon",
                Some("google-1"),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["scoped"], true);
        assert_eq!(ids(&v), ["canon/ammo"], "armor is not selected");

        // The cast gates the per-character memory.
        let (_, v) = call(
            router(st.clone()),
            get(
                "/v1/mind/list?world=battle-cities&id=memory",
                Some("google-1"),
            ),
        )
        .await;
        assert_eq!(ids(&v), ["memory/commander"]);

        // Naming an excluded topic directly is not a way past it.
        let (s, v) = call(
            router(st.clone()),
            get(
                "/v1/mind/list?world=battle-cities&id=canon/armor",
                Some("google-1"),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_eq!(v["error"], "out_of_scope");

        // Nor is opening an entry inside it, or writing one.
        let entry = "/v1/mind/entry?world=battle-cities&id=canon/armor/plate";
        let (s, _) = call(router(st.clone()), get(entry, Some("google-1"))).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, _) = call(
            router(st.clone()),
            send(entry, "PUT", ADMIN, json!({ "text": "owned" })),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN);

        // Unscoped, both are there — the filter is the world's, not a property
        // of the thing.
        let (_, v) = call(router(st), get("/v1/mind/list?id=canon", Some("google-1"))).await;
        assert_eq!(ids(&v), ["canon/ammo", "canon/armor"]);
    }

    #[tokio::test]
    async fn an_unknown_world_is_refused_rather_than_ignored() {
        let base = tmp("mind-badworld");
        let st = mind_state(base);
        let (s, v) = call(
            router(st),
            get("/v1/mind/list?world=atlantis", Some("google-1")),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"], "unknown_world");
    }

    /// A daemon with no `--mind` says so once, rather than failing five ways.
    #[tokio::test]
    async fn without_a_mind_the_editor_says_it_has_nothing_to_edit() {
        let base = tmp("mind-none");
        let st = Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            Accounts::load(base.join("accounts")).unwrap(),
            Npcs::load(&base).unwrap(),
            serde_yaml::from_str(&format!("admins:\n  - sub: {ADMIN}\n")).unwrap(),
            crate::collections::Libraries::load(&crate::projection::Source::resolve(None).unwrap()),
            Mind::new(None),
            Images::new(&base),
        );
        let (s, v) = call(router(st), get("/v1/mind/list", Some("google-1"))).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(v["error"], "no_mind");
    }

    /// A folder is never removed through this API: one click must not become a
    /// recursive delete.
    #[tokio::test]
    async fn deleting_a_topic_removes_its_text_and_keeps_what_is_inside() {
        let base = tmp("mind-rmdir");
        let st = mind_state(base.clone());
        std::fs::write(base.join("layers/world/ammo.md"), "overview").unwrap();

        let (s, _) = call(
            router(st.clone()),
            send("/v1/mind/entry?id=canon/ammo", "DELETE", ADMIN, json!({})),
        )
        .await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        // The overview is gone; the entries under it are not. One button must
        // never become a recursive delete.
        assert!(!base.join("layers/world/ammo.md").exists());
        assert!(base.join("layers/world/ammo/bolt.md").is_file());

        // It is still a topic, now without text of its own.
        let (_, v) = call(router(st), get("/v1/mind/list?id=canon", Some("google-1"))).await;
        let ammo = v["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == "canon/ammo")
            .expect("still listed");
        assert_eq!(ammo["has_text"], false);
        assert_eq!(ammo["count"], 1);
    }

    /// A section file, with the shape and the comments a real one has.
    fn seed_section(base: &std::path::Path) {
        std::fs::create_dir_all(base.join("responses")).unwrap();
        std::fs::write(
            base.join("responses/accept.yaml"),
            r#"id: accept
category: accept
description: Accepting what was offered.

# The frozen structural mode — its KV is loaded once the section is selected.
template: |
  The tactical self is gone.

# Provenance lead-ins. FIXED SHAPE: 4 turns. Target: 16.
examples:
  - note: Late apology.
    turns:
      - role: user
        content: |
          "I'm late."
      - role: assistant
        thinking: |
          They take it lightly.
"#,
        )
        .unwrap();
    }

    /// A document arrives as fields, in file order, each with the control it
    /// wants — and the author's comment attached to the field it describes.
    #[tokio::test]
    async fn an_entry_can_be_read_as_fields() {
        let base = tmp("mind-fields");
        seed_section(&base);
        let st = mind_state(base);

        let (s, v) = call(
            router(st),
            get("/v1/mind/fields?id=responses/accept", Some("google-1")),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let fields = v["fields"].as_array().unwrap();
        let keys: Vec<&str> = fields.iter().map(|f| f["key"].as_str().unwrap()).collect();
        assert_eq!(
            keys,
            ["id", "category", "description", "template", "examples"]
        );

        let by = |k: &str| fields.iter().find(|f| f["key"] == k).unwrap().clone();
        assert_eq!(by("id")["readonly"], true, "the id is the address");
        assert_eq!(by("category")["kind"], "line");
        assert_eq!(by("template")["kind"], "text");
        assert_eq!(by("examples")["kind"], "conversations");

        // The author's own note, on the field it is about.
        assert!(by("template")["note"]
            .as_str()
            .unwrap()
            .contains("frozen structural mode"));
        assert!(by("examples")["note"]
            .as_str()
            .unwrap()
            .contains("FIXED SHAPE"));

        // The conversation, as turns.
        let turns = by("examples")["value"][0]["turns"].clone();
        assert_eq!(turns[0]["role"], "user");
        assert!(turns[0]["content"].as_str().unwrap().contains("I'm late"));
        assert_eq!(turns[1]["role"], "assistant");
        assert!(turns[1]["thinking"].is_string());
    }

    /// **The property the whole field editor rests on.** Saving from a form
    /// must not cost the file its comments — 701 of 712 real section files have
    /// them, and they are the corpus's best documentation.
    #[tokio::test]
    async fn saving_fields_keeps_every_comment_in_the_file() {
        let base = tmp("mind-fields-save");
        seed_section(&base);
        let st = mind_state(base.clone());
        let before = std::fs::read_to_string(base.join("responses/accept.yaml")).unwrap();

        // Edit a scalar and add a turn to the conversation — the two things the
        // form does — and send back every field, as the console would.
        let (s, _) = call(
            router(st.clone()),
            send(
                "/v1/mind/fields?id=responses/accept",
                "PUT",
                ADMIN,
                json!({ "values": {
                    "id": "accept",
                    "category": "acceptance",
                    "description": "Accepting what was offered.",
                    "template": "The tactical self is gone.\n",
                    "examples": [{
                        "note": "Late apology.",
                        "turns": [
                            { "role": "user", "content": "\"I'm late.\"\n" },
                            { "role": "assistant", "content": "A tip of the head.\n" },
                            { "role": "assistant", "thinking": "They take it lightly.\n" },
                        ],
                    }],
                }}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);

        let after = std::fs::read_to_string(base.join("responses/accept.yaml")).unwrap();
        for comment in [
            "# The frozen structural mode",
            "# Provenance lead-ins. FIXED SHAPE: 4 turns. Target: 16.",
        ] {
            assert!(after.contains(comment), "lost `{comment}`\n---\n{after}");
        }
        assert_ne!(after, before, "nothing was written");

        // And the edits are really there, read back through the same door.
        let (_, v) = call(
            router(st),
            get("/v1/mind/fields?id=responses/accept", Some("google-1")),
        )
        .await;
        let fields = v["fields"].as_array().unwrap();
        let by = |k: &str| fields.iter().find(|f| f["key"] == k).unwrap().clone();
        assert_eq!(by("category")["value"], "acceptance");
        assert_eq!(
            by("examples")["value"][0]["turns"]
                .as_array()
                .unwrap()
                .len(),
            3,
            "the added turn survived"
        );
    }

    /// Not every document is a set of fields, and that is a fact rather than a
    /// failure — the console offers the text editor for those.
    #[tokio::test]
    async fn a_document_that_is_not_fields_says_so() {
        let base = tmp("mind-notfields");
        let st = mind_state(base.clone());
        std::fs::write(base.join("layers/world/ammo/bolt.md"), "# just prose\n").unwrap();
        let (s, v) = call(
            router(st),
            get("/v1/mind/fields?id=canon/ammo/bolt", Some("google-1")),
        )
        .await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(v["error"], "not_fields");
    }

    /// The projection schema, as it really is: banner comments between the
    /// layers, a nested budget, and a list of selection groups.
    fn seed_projection(base: &std::path::Path) {
        std::fs::write(
            base.join("projection.yaml"),
            r#"# The projection schema.
layers:
  # ── World ──────────────────────────────────────────────────────────────────
  - name: world
    description: |
      Shared knowledge about the setting.
    window: 8000
    score_threshold: 0.30
    gather_scope: shared
    decode_priority: low
    budget:
      priority: 70
      adaptive:
        gain: 2.0
    groups:
      - id: canon
        selection: { kind: top_k, k: 6 }

  # ── Beliefs ────────────────────────────────────────────────────────────────
  - name: beliefs
    description: |
      What the character holds to be true.
    window: 4000
    gather_scope: conversation
    budget:
      priority: 90
"#,
        )
        .unwrap();
    }

    /// **A projection layer is an entry like any other.**
    ///
    /// It is not a file — the nine of them live in one seven-hundred-line
    /// document — so this is the whole of what makes them editable: the schema
    /// lists its layers, each one opens on its own, and it arrives as controls
    /// rather than as YAML. Nothing about the route is special; the address
    /// does the work.
    #[tokio::test]
    async fn a_projection_layer_lists_and_opens_as_fields() {
        let base = tmp("mind-layers");
        seed_projection(&base);
        let st = mind_state(base);

        // The schema holds its layers, in the order the document writes them.
        let (s, v) = call(
            router(st.clone()),
            get("/v1/mind/list?id=settings/projection", Some("google-1")),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let ids: Vec<&str> = v["children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            ["settings/projection/world", "settings/projection/beliefs"]
        );

        let (s, v) = call(
            router(st),
            get(
                "/v1/mind/fields?id=settings/projection/world",
                Some("google-1"),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let fields = v["fields"].as_array().unwrap();
        let by = |k: &str| fields.iter().find(|f| f["key"] == k).unwrap().clone();

        // The name is the address, so it is shown and not editable.
        assert_eq!(by("name")["readonly"], true);
        assert_eq!(by("window")["kind"], "number");
        assert_eq!(by("window")["value"], 8000);
        assert_eq!(by("description")["kind"], "text");
        // A vocabulary the engine fixes is a select, not a place to make a typo.
        assert_eq!(by("gather_scope")["kind"], "choice");
        assert_eq!(
            by("gather_scope")["choices"],
            json!(["conversation", "shared"])
        );
        assert_eq!(
            by("decode_priority")["choices"],
            json!(["low", "normal", "high"])
        );

        // A mapping is its own fields, all the way down.
        assert_eq!(by("budget")["kind"], "group");
        let budget = by("budget")["fields"].clone();
        let bf = |k: &str| {
            budget
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["key"] == k)
                .unwrap()
                .clone()
        };
        assert_eq!(bf("priority")["kind"], "number");
        assert_eq!(bf("adaptive")["kind"], "group");
        assert_eq!(bf("adaptive")["fields"][0]["key"], "gain");

        // And a list of mappings is rows of fields.
        assert_eq!(by("groups")["kind"], "rows");
        assert_eq!(by("groups")["rows"][0][0]["value"], "canon");
    }

    /// **The authoring plane persists, and reports the engine's part absent.**
    ///
    /// These eight routes used to fall through to the console's fixture, which
    /// answered with three invented beliefs, three relationships and three
    /// strategies — the same nine for every character, including ones that did
    /// not exist. The console's `+ Author` button was a toast saying an engine
    /// was required, for a write that needs no engine at all: §16 calls this
    /// the authoring plane precisely because it is what a person types.
    #[tokio::test]
    async fn the_authoring_plane_is_written_and_kept() {
        let st = state(tmp("authoring"));
        let a = "google-1";
        call(router(st.clone()), get("/v1/me", Some(a))).await;
        author(&st).await;

        let (s, npc) = call(
            router(st.clone()),
            send(
                "/v1/npc",
                "POST",
                a,
                json!({ "name": "Varek", "world_id": "battle-cities", "personality_id": "commander" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
        let id = npc["npc_id"].as_str().unwrap().to_string();

        // A character nobody has authored holds nothing — not three fixtures.
        let (s, v) = call(
            router(st.clone()),
            get(&format!("/v1/npc/{id}/beliefs"), Some(a)),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["beliefs"].as_array().unwrap().len(), 0);

        // State one.
        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/beliefs/hess_word"),
                "PUT",
                a,
                json!({ "statement": "Hess keeps his word.", "confidence": 0.72 }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);

        let (_, v) = call(
            router(st.clone()),
            get(&format!("/v1/npc/{id}/beliefs"), Some(a)),
        )
        .await;
        let b = &v["beliefs"][0];
        assert_eq!(b["statement"], "Hess keeps his word.");
        assert_eq!(b["confidence"], 0.72);
        assert_eq!(b["origin"], "authored");
        // The evidence process has not run, so its measurements are absent
        // rather than zero — a belief with `disconfirmation: 0` reads as
        // weighed and unshaken, which is a claim nothing here can make.
        assert!(b["disconfirmation"].is_null());
        assert!(b["under_pressure"].is_null());
        assert!(b["history"].is_null());

        // A dial set is a dial kept, and the other two are not reset by it.
        let (_, v) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/modulation"),
                "PUT",
                a,
                json!({ "threat": 0.66 }),
            ),
        )
        .await;
        assert_eq!(v["modulation"]["threat"], 0.66);
        assert_eq!(v["modulation"]["curiosity"], 0.5, "an untouched dial moved");

        // The environment is config, and it saves.
        let (_, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/environment"),
                "PUT",
                a,
                json!({ "enabled": false, "system_prompt": "A ridge at dusk." }),
            ),
        )
        .await;
        let (_, v) = call(
            router(st.clone()),
            get(&format!("/v1/npc/{id}/environment"), Some(a)),
        )
        .await;
        assert_eq!(v["enabled"], false);
        assert_eq!(v["system_prompt"], "A ridge at dusk.");

        // A strategy under a parent that does not exist is refused, rather than
        // silently becoming a root and losing the nesting.
        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/agency/flank"),
                "PUT",
                a,
                json!({ "statement": "Flank east.", "parent_id": "nope" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // Out-of-range dials are refused, not clamped.
        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/modulation"),
                "PUT",
                a,
                json!({ "threat": 4.0 }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // And none of it belongs to anybody else.
        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/beliefs/hess_word"),
                "PUT",
                "google-2",
                json!({ "statement": "Not yours." }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // **A portrait is stored and attached.** The upload used to be dropped
        // by the console; then, briefly, it was dropped by the daemon —
        // `patch` has no `portrait_image_id` field, so a write through it was
        // a silent no-op that answered 200.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[7u8; 32]);
        let (s, v) = call(
            router(st.clone()),
            bytes(&format!("/v1/npc/{id}/portrait"), "PUT", a, png.clone()),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let image_id = v["portrait"]["image_id"]
            .as_str()
            .unwrap_or_else(|| panic!("no portrait on the record: {v}"))
            .to_string();
        assert_eq!(v["portrait"]["origin"], "uploaded");

        // And it comes back, byte for byte.
        let (s, got) = call_raw(
            router(st.clone()),
            get(&format!("/v1/image/{image_id}"), Some(a)),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(got, png, "the bytes served are not the bytes uploaded");

        // A client cannot name an image id itself — that would let one point at
        // an upload it does not own, and every id in the store is valid.
        let (_, v) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}"),
                "PATCH",
                a,
                json!({ "portrait_image_id": "img_0000000000000000.png" }),
            ),
        )
        .await;
        assert_eq!(
            v["portrait"]["image_id"], image_id,
            "PATCH accepted a portrait id from the caller"
        );

        // Deleting one leaves the character.
        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/beliefs/hess_word"),
                "DELETE",
                a,
                json!({}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        let (_, v) = call(router(st), get(&format!("/v1/npc/{id}/beliefs"), Some(a))).await;
        assert_eq!(v["beliefs"].as_array().unwrap().len(), 0);
    }

    /// **The narrative clock writes, and the world remembers it.**
    ///
    /// This route used to be the console's fixture answering `{"ok":true}` to
    /// everything, under a console dialog that said it affected every character
    /// in the world. It moved nothing.
    #[tokio::test]
    async fn the_world_clock_is_set_and_kept() {
        let base = tmp("world-clock");
        std::fs::create_dir_all(base.join("worlds")).unwrap();
        std::fs::write(
            base.join("worlds/ardh.yaml"),
            "# The world's own header, which a save must not eat.\nid: ardh\nname: Ardh\n",
        )
        .unwrap();
        let st = mind_state(base.clone());

        // A world that has never been set still reports a clock: started now,
        // at real time, which is what an author who has not thought about it
        // means.
        let (s, v) = call(router(st.clone()), get("/v1/world/ardh", None)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["time"]["scale"], 1.0);
        assert_eq!(v["time"]["paused"], false);
        assert!(v["time"]["world_ms"].as_i64().unwrap() > 0);

        // Jump it somewhere and speed it up.
        let (s, v) = call(
            router(st.clone()),
            send(
                "/v1/world/ardh/time",
                "PUT",
                ADMIN,
                json!({ "world_ms": 5_000_000, "scale": 60 }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["scale"], 60.0);
        assert!(v["world_ms"].as_i64().unwrap() >= 5_000_000);

        // It is on disk as an anchor, and the header survived the write.
        let text = std::fs::read_to_string(base.join("worlds/ardh.yaml")).unwrap();
        assert!(text.contains("# The world's own header"), "{text}");
        assert!(text.contains("at_ms"), "no anchor written:\n{text}");

        // And it is still there on the next read, having advanced rather than
        // reset.
        let (_, v) = call(router(st.clone()), get("/v1/world/ardh", None)).await;
        assert_eq!(v["time"]["scale"], 60.0);
        assert!(v["time"]["world_ms"].as_i64().unwrap() >= 5_000_000);

        // Pausing keeps the pace it was running at, so resuming does not land
        // on 1×.
        let (_, v) = call(
            router(st.clone()),
            send(
                "/v1/world/ardh/time",
                "PUT",
                ADMIN,
                json!({ "paused": true }),
            ),
        )
        .await;
        assert_eq!(v["paused"], true);
        assert_eq!(v["scale"], 60.0, "pausing forgot the pace");

        // Setting the clock is an admin's.
        let (s, _) = call(
            router(st),
            send(
                "/v1/world/ardh/time",
                "PUT",
                "google-plain-user",
                json!({ "scale": 1 }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN);
    }

    /// **A personality's collections come from its own document.**
    ///
    /// The fixture this replaces invented an anchor, four identity facets and a
    /// doctrine, and served the same five for every character — on the page an
    /// author opens to check what they wrote.
    #[tokio::test]
    async fn a_personalitys_collections_are_read_from_its_document() {
        let base = tmp("persona-collections");
        std::fs::create_dir_all(base.join("personalities")).unwrap();
        std::fs::write(
            base.join("personalities/keeper.yaml"),
            "id: keeper\ncategory: identity\nanchor: |\n  You keep the tower.\n\
             personality:\n  voice: |\n    Short sentences.\n  processing: |\n    Weigh what you saw.\n",
        )
        .unwrap();
        let st = mind_state(base);

        let (s, v) = call(router(st), get("/v1/personality/keeper/collections", None)).await;
        assert_eq!(s, StatusCode::OK);
        let cols = v["collections"].as_array().unwrap();
        let by = |n: &str| cols.iter().find(|c| c["name"] == n).unwrap().clone();

        // The anchor, as itself.
        let anchor = by("identity_anchor");
        assert_eq!(anchor["sections"][0]["id"], "anchor");
        assert!(anchor["sections"][0]["template"]
            .as_str()
            .unwrap()
            .contains("You keep the tower"));

        // One section per facet the document declares — not a fixed four.
        let identity = by("identity");
        let ids: Vec<&str> = identity["sections"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["processing", "voice"]);
        assert!(identity["sections"][0]["chars"].as_u64().unwrap() > 0);

        // No doctrine in this document, so no doctrine collection — 69 of the
        // mind's 74 personalities are in that position, and five empty rows on
        // each of them is not a reading of anything.
        assert!(cols.iter().all(|c| c["name"] != "doctrine"));
    }

    /// **`/v1/schema/layers` is the schema, not a copy of it.**
    ///
    /// It used to be answered by the console's fixture, with the nine layers
    /// written out a second time — and they drifted: the fixture had `action`
    /// at budget priority 95 where the schema said 100. Nothing compared them,
    /// so nothing could report it. Reading the same document the editor writes
    /// is what makes that impossible rather than merely unlikely.
    #[tokio::test]
    async fn the_layer_schema_is_read_from_the_mind_itself() {
        let base = tmp("schema-layers");
        seed_projection(&base);
        let st = mind_state(base.clone());

        let (s, v) = call(
            router(st.clone()),
            get("/v1/schema/layers", Some("google-1")),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let layers = v["layers"].as_array().expect("layers");
        assert_eq!(layers.len(), 2);
        // The schema's own vocabulary, verbatim — not translated into a shape
        // this route invented, which would be the second copy all over again.
        assert_eq!(layers[0]["name"], "world");
        assert_eq!(layers[0]["window"], 8000);
        assert_eq!(layers[0]["gather_scope"], "shared");
        assert_eq!(layers[0]["budget"]["priority"], 70);
        assert_eq!(layers[0]["groups"][0]["id"], "canon");

        // And it follows the file. An edit through the layer editor is visible
        // here on the next read, because there is only the one document.
        let (s, _) = call(
            router(st.clone()),
            send(
                "/v1/mind/fields?id=settings/projection/world",
                "PUT",
                ADMIN,
                // Every field, as the console sends them — the form holds the
                // whole layer and puts it all back.
                json!({ "values": {
                    "name": "world",
                    "description": "Shared knowledge about the setting.\n",
                    "window": 12345,
                    "score_threshold": 0.30,
                    "gather_scope": "shared",
                    "decode_priority": "low",
                    "budget": { "priority": 70, "adaptive": { "gain": 2.0 } },
                    "groups": [{ "id": "canon", "selection": { "kind": "top_k", "k": 6 } }],
                } }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let (_, v) = call(router(st), get("/v1/schema/layers", Some("google-1"))).await;
        assert_eq!(
            v["layers"][0]["window"], 12345,
            "the route kept its own copy"
        );
    }

    /// Saving a layer changes that layer, and the document it lives in keeps
    /// every banner comment between the others.
    #[tokio::test]
    async fn saving_a_layer_leaves_the_rest_of_the_schema_alone() {
        let base = tmp("mind-layer-save");
        seed_projection(&base);
        let st = mind_state(base.clone());
        let before = std::fs::read_to_string(base.join("projection.yaml")).unwrap();

        let (s, _) = call(
            router(st.clone()),
            send(
                "/v1/mind/fields?id=settings/projection/world",
                "PUT",
                ADMIN,
                json!({ "values": {
                    "name": "world",
                    "description": "Shared knowledge about the setting.\n",
                    "window": 9000,
                    "score_threshold": 0.30,
                    "gather_scope": "shared",
                    "decode_priority": "low",
                    "budget": { "priority": 70, "adaptive": { "gain": 2.0 } },
                    "groups": [{ "id": "canon", "selection": { "kind": "top_k", "k": 6 } }],
                } }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);

        let after = std::fs::read_to_string(base.join("projection.yaml")).unwrap();
        assert!(after.contains("window: 9000"), "{after}");
        assert!(after.contains("# ── World ─"), "{after}");
        assert!(after.contains("# ── Beliefs ─"), "lost a comment:\n{after}");
        assert!(after.contains("# The projection schema."), "{after}");
        // One line, and only one.
        let b: Vec<&str> = before.lines().collect();
        let a: Vec<&str> = after.lines().collect();
        assert_eq!(a.len(), b.len(), "\n{after}");
        let moved: Vec<usize> = (0..b.len()).filter(|&i| b[i] != a[i]).collect();
        assert_eq!(moved.len(), 1, "changed lines {moved:?}:\n{after}");

        // Delete on a layer must never reach the schema it is part of.
        let (s, _) = call(
            router(st),
            send(
                "/v1/mind/entry?id=settings/projection/world",
                "DELETE",
                ADMIN,
                json!({}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert!(base.join("projection.yaml").exists(), "the schema went");
    }

    /// Reading fields is a signed-in read; saving them is an admin's write —
    /// the same rule as the text they are a view of.
    #[tokio::test]
    async fn the_field_editor_follows_the_same_roles() {
        let base = tmp("mind-fields-roles");
        seed_section(&base);
        let st = mind_state(base);
        let uri = "/v1/mind/fields?id=responses/accept";

        let (s, _) = call(router(st.clone()), get(uri, None)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        let (s, _) = call(router(st.clone()), get(uri, Some("google-1"))).await;
        assert_eq!(s, StatusCode::OK);
        let (s, _) = call(
            router(st),
            send(uri, "PUT", "google-plain-user", json!({ "values": {} })),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN);
    }

    /// `?reveal=1` is a request, and the role is what answers it.
    ///
    /// The console sends it while an admin holds RIGHT ALT. The parameter
    /// itself grants nothing: an anonymous caller or an ordinary user can
    /// append it to the URL and gets exactly the listing they would have got
    /// without it, because the role is read from the gateway's headers on this
    /// request rather than from anything the client says.
    #[tokio::test]
    async fn reveal_shows_hidden_documents_to_an_admin_and_to_nobody_else() {
        let st = state(tmp("reveal"));
        for (id, body) in [
            ("battle-cities", json!({ "name": "Battle Cities" })),
            ("earth", json!({ "name": "Earth", "hidden": true })),
        ] {
            let r = send(&format!("/v1/world/{id}"), "PUT", ADMIN, body);
            assert_eq!(call(router(st.clone()), r).await.0, StatusCode::OK);
        }
        let ids = |v: &Value| -> Vec<String> {
            v["worlds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| w["world_id"].as_str().unwrap().to_string())
                .collect()
        };

        // The admin, holding the key.
        for q in ["?reveal=1", "?reveal=true"] {
            let (_, v) = call(
                router(st.clone()),
                get(&format!("/v1/world{q}"), Some(ADMIN)),
            )
            .await;
            assert!(
                ids(&v).contains(&"earth".to_string()),
                "`{q}` hid it from an admin"
            );
        }

        // Everybody else, asking for exactly the same thing.
        for who in [None, Some("google-someone-else")] {
            let (_, v) = call(router(st.clone()), get("/v1/world?reveal=1", who)).await;
            assert_eq!(ids(&v), ["battle-cities"], "reveal granted to {who:?}");
        }

        // And an admin who is *not* asking still gets the discreet listing —
        // the point is a screen share, and being an admin is the normal state
        // for whoever is sharing it.
        let (_, v) = call(router(st.clone()), get("/v1/world", Some(ADMIN))).await;
        assert_eq!(ids(&v), ["battle-cities"]);

        // A value that is not the opt-in is not the opt-in.
        for q in ["?reveal=0", "?reveal=", "?reveal=yes"] {
            let (_, v) = call(
                router(st.clone()),
                get(&format!("/v1/world{q}"), Some(ADMIN)),
            )
            .await;
            assert_eq!(ids(&v), ["battle-cities"], "`{q}` revealed a hidden world");
        }
    }

    /// A world admits a subset of the shared craft libraries. Same files, two
    /// worlds, different answers.
    #[tokio::test]
    async fn a_world_gets_only_the_section_categories_it_admits() {
        let st = state(tmp("excludes"));
        for (id, body) in [
            (
                "battle-cities",
                json!({ "name": "Battle Cities", "excludes": ["sexual", "intimate"] }),
            ),
            ("earth", json!({ "name": "Earth" })),
        ] {
            let r = send(&format!("/v1/world/{id}"), "PUT", ADMIN, body);
            assert_eq!(call(router(st.clone()), r).await.0, StatusCode::OK);
        }

        // This test's daemon has no mind, so both libraries are empty and the
        // interesting assertion is that the *declaration* reaches the wire —
        // the filtering itself is covered in `collections`.
        let (_, v) = call(
            router(st.clone()),
            get("/v1/world/battle-cities/collections", None),
        )
        .await;
        assert_eq!(
            v["collections"][0]["excludes"],
            json!(["sexual", "intimate"])
        );

        let (_, v) = call(router(st.clone()), get("/v1/world/earth/collections", None)).await;
        assert_eq!(
            v["collections"][0]["excludes"],
            json!([]),
            "a world that names nothing admits everything"
        );

        // And an unknown world is still a 404 rather than an empty library.
        let (s, _) = call(router(st), get("/v1/world/atlantis/collections", None)).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    /// The role is reported on the account so the console can hide a control
    /// the server would refuse — and it is the server's answer, never a claim
    /// the client makes.
    #[tokio::test]
    async fn the_account_carries_the_role_the_server_decided() {
        let state = state(tmp("role-me"));
        let (_, me) = call(router(state.clone()), get("/v1/me", Some(ADMIN))).await;
        assert_eq!(me["role"], "admin");
        let (_, me) = call(router(state), get("/v1/me", Some("google-1"))).await;
        assert_eq!(me["role"], "user");
    }

    /// A server-side write failure must not hand back the absolute path of a
    /// file on this machine. The author's own mistakes still come back in full,
    /// because the message is the useful part.
    #[tokio::test]
    async fn a_refused_write_says_what_is_wrong_without_mapping_the_disk() {
        let st = state(tmp("errs"));

        // The author's mistake: an id that could not be a file name.
        let (s, e) = call(
            router(st.clone()),
            send("/v1/world/Not-Valid", "PUT", ADMIN, json!({ "name": "x" })),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(e["error"], "invalid_arguments");
        assert!(
            e["detail"].as_str().unwrap().contains("lowercase"),
            "the reason was not reported: {e}"
        );

        // Too large: also the caller's, and the limit is part of the API.
        let (s, e) = call(
            router(st.clone()),
            send(
                "/v1/world/big",
                "PUT",
                ADMIN,
                json!({ "name": "x", "setting": "a".repeat(300 * 1024) }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(
            e["detail"].as_str().unwrap().contains("limit"),
            "the limit was not named: {e}"
        );

        // Something that is not a plain file where the document should go — a
        // directory here, a symlink in the case the guard exists for. Refused
        // rather than followed, and the reason says what is wrong without
        // saying where.
        let dir = tmp("errs-clash");
        std::fs::create_dir_all(dir.join("worlds").join("taken.yaml")).unwrap();
        let clash = state(dir.clone());
        let (s, e) = call(
            router(clash),
            send("/v1/world/taken", "PUT", ADMIN, json!({ "name": "x" })),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{e}");
        assert!(e["detail"].as_str().unwrap().contains("not a file"), "{e}");

        // Nothing above put a path from this machine on the wire.
        let text = e.to_string();
        assert!(!text.contains(":\\"), "a Windows path leaked: {text}");
        assert!(!text.contains("/Users/"), "a path leaked: {text}");
    }

    /// The listings carry the number of living characters that name each
    /// document, which is what makes "reaches all N characters" a fact rather
    /// than copy.
    #[tokio::test]
    async fn the_listings_count_the_characters_that_name_them() {
        let state = state(tmp("counts"));
        let a = "google-1";
        call(router(state.clone()), get("/v1/me", Some(a))).await;
        author(&state).await;

        let (_, w) = call(router(state.clone()), get("/v1/world", Some(a))).await;
        assert_eq!(w["worlds"][0]["npc_count"], 0, "nothing has been made yet");

        for name in ["Varek", "Ilse"] {
            let b =
                json!({ "name": name, "world_id": "battle-cities", "personality_id": "commander" });
            let (s, _) = call(router(state.clone()), send("/v1/npc", "POST", a, b)).await;
            assert_eq!(s, StatusCode::CREATED);
        }

        let (_, w) = call(router(state.clone()), get("/v1/world", Some(a))).await;
        assert_eq!(w["worlds"][0]["npc_count"], 2);

        // A read-edit-write cycle must not write the count back into the file.
        // The obvious client sends back what it was given; if the server did
        // not strip a derived field, the document would grow a count that
        // disagrees with the cast from the next character onward.
        let doc = w["worlds"][0].clone();
        let (s, _) = call(
            router(state.clone()),
            send("/v1/world/battle-cities", "PUT", ADMIN, doc),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let stored = state
            .worlds
            .read()
            .await
            .get("battle-cities")
            .unwrap()
            .body
            .clone();
        assert!(
            stored.get("npc_count").is_none(),
            "count leaked into the document: {stored}"
        );
        assert!(
            stored.get("world_id").is_none(),
            "id leaked into the document: {stored}"
        );
        assert_eq!(
            stored["name"], "Battle Cities",
            "the rest of the document survived"
        );
        let (_, p) = call(
            router(state.clone()),
            get("/v1/personality/commander", Some(a)),
        )
        .await;
        assert_eq!(p["npc_count"], 2);
        assert_eq!(p["personality_id"], "commander");
    }
}
