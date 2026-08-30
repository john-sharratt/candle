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
    http::{HeaderMap, StatusCode},
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
use crate::collections::{self, Libraries};
use crate::guard::Api;
use crate::identity::require;
use crate::npcs::{Filter, NpcError, Npcs};
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
}

impl Authored {
    pub fn new(
        worlds: Registry,
        personalities: Registry,
        accounts: Accounts,
        npcs: Npcs,
        roles: Roles,
        libraries: Libraries,
    ) -> Arc<Self> {
        Arc::new(Self {
            worlds: RwLock::new(worlds),
            personalities: RwLock::new(personalities),
            accounts: RwLock::new(accounts),
            npcs: RwLock::new(npcs),
            roles,
            libraries,
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
async fn owner_of(
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

fn err(status: StatusCode, code: &str, detail: &str) -> Response {
    (status, Json(json!({ "error": code, "detail": detail }))).into_response()
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

/// Every world, minus the hidden ones the filter has not named.
///
/// `q` is the console's filter box. A world with `hidden: true` is left out
/// until a **whole word** of `q` names it — see [`crate::visibility`]. Filtering
/// happens here rather than in the browser because a list the client narrows is
/// a list the client was first sent in full.
async fn list_worlds(
    State(s): State<Arc<Authored>>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let query = q.get("q").map(String::as_str).unwrap_or_default();
    let npcs = s.npcs.read().await;
    let counts = npcs.counts_by(|n| n.world_id.as_str());
    let reg = s.worlds.read().await;
    let worlds: Vec<Value> = reg
        .iter()
        .filter(|r| visibility::listable(&r.id, &r.body, query))
        .map(|r| with_count(with_id("world_id", &r.id, &r.body), &counts, &r.id))
        .collect();
    Json(json!({ "worlds": worlds })).into_response()
}

async fn get_world(State(s): State<Arc<Authored>>, Path(wid): Path<String>) -> Response {
    let npcs = s.npcs.read().await;
    let counts = npcs.counts_by(|n| n.world_id.as_str());
    match s.worlds.read().await.get(&wid) {
        Some(r) => Json(with_count(
            with_id("world_id", &r.id, &r.body),
            &counts,
            &r.id,
        ))
        .into_response(),
        None => err(StatusCode::NOT_FOUND, "world_not_found", &wid),
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
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let query = q.get("q").map(String::as_str).unwrap_or_default();
    let npcs = s.npcs.read().await;
    let counts = npcs.counts_by(|n| n.personality_id.as_str());
    let reg = s.personalities.read().await;
    let personalities: Vec<Value> = reg
        .iter()
        .filter(|r| visibility::listable(&r.id, &r.body, query))
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
        )
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
        let p = send(
            "/v1/personality/commander",
            "PUT",
            ADMIN,
            json!({ "name": "Commander", "anchor": "Position is read before people are." }),
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
                ("/v1/personality", "unauthenticated"),
                ("/v1/personality/:aid", "unauthenticated"),
                ("/v1/personality/:aid", "admin"),
                // The cast: signed in, then ownership per record.
                ("/v1/npc", "user"),
                ("/v1/npc/:nid", "user"),
                ("/v1/npc/:nid/tags", "user"),
                ("/v1/npc/:nid/hidden", "user"),
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
