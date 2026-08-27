//! The first routes `npcd` answers itself rather than handing to the mock.
//!
//! Worlds and archetypes are authored files (see [`crate::registry`]), so they
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

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, put},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::sync::RwLock;
use web::auth::session::Identity;

use crate::accounts::{self, Accounts, NameError, PatchError};
use crate::identity::{self, NotSignedIn};
use crate::registry::Registry;

/// Both authored collections, shared with the HTTP layer.
///
/// A write lock is held only for the duration of a save. Reads are the common
/// case by orders of magnitude and take the read lock, so a GUI listing worlds
/// never waits on another GUI editing one.
pub struct Authored {
    pub worlds: RwLock<Registry>,
    pub archetypes: RwLock<Registry>,
    pub accounts: RwLock<Accounts>,
}

impl Authored {
    pub fn new(worlds: Registry, archetypes: Registry, accounts: Accounts) -> Arc<Self> {
        Arc::new(Self {
            worlds: RwLock::new(worlds),
            archetypes: RwLock::new(archetypes),
            accounts: RwLock::new(accounts),
        })
    }
}

/// The routes this daemon owns, to be layered over the mock.
pub fn router(state: Arc<Authored>) -> Router {
    Router::new()
        .route("/v1/world", get(list_worlds))
        .route(
            "/v1/world/:wid",
            get(get_world).put(put_world).delete(delete_world),
        )
        .route("/v1/archetype", get(list_archetypes))
        .route(
            "/v1/archetype/:aid",
            get(get_archetype)
                .put(put_archetype)
                .delete(delete_archetype),
        )
        .route("/v1/me", get(me))
        .route("/v1/me/profile", get(get_profile).put(put_profile))
        .route("/v1/me/profile/history", get(get_profile_history))
        .route("/v1/me/unique-name", put(put_unique_name))
        .with_state(state)
}

/// Establish the caller, or produce the 401 that says why not.
///
/// Boxed because a `Response` is large and this is the cold path — clippy is
/// right that returning one by value in a `Result` is a big move for the common
/// case, which is success.
fn caller(headers: &HeaderMap) -> Result<Identity, Box<Response>> {
    identity::identify(headers).map_err(|NotSignedIn| {
        Box::new(err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "not signed in",
        ))
    })
}

/// Who the caller is here, creating the local account on first sight.
async fn me(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match caller(&headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.write().await.upsert(&id, now_ms()) {
        Ok(me) => Json(me).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "account upsert failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "account_write_failed",
                &e.to_string(),
            )
        }
    }
}

async fn get_profile(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match caller(&headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.get(&id.sub) {
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
    let id = match caller(&headers) {
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
    match s
        .accounts
        .write()
        .await
        .put_profile(&id.sub, patch, now_ms())
    {
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
    let id = match caller(&headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match s.accounts.read().await.profile_history(&id.sub) {
        Some(revisions) => Json(json!({ "revisions": revisions })).into_response(),
        None => err(StatusCode::NOT_FOUND, "account_not_found", "no account yet"),
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
    let id = match caller(&headers) {
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
    match s.accounts.write().await.put_unique_name(&id.sub, name) {
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

/// The console reads `world_id` and `archetype_id`; the file knows only its own
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

async fn list_worlds(State(s): State<Arc<Authored>>) -> Response {
    let reg = s.worlds.read().await;
    let worlds: Vec<Value> = reg
        .iter()
        .map(|r| with_id("world_id", &r.id, &r.body))
        .collect();
    Json(json!({ "worlds": worlds })).into_response()
}

async fn get_world(State(s): State<Arc<Authored>>, Path(wid): Path<String>) -> Response {
    match s.worlds.read().await.get(&wid) {
        Some(r) => Json(with_id("world_id", &r.id, &r.body)).into_response(),
        None => err(StatusCode::NOT_FOUND, "world_not_found", &wid),
    }
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

async fn list_archetypes(State(s): State<Arc<Authored>>) -> Response {
    let reg = s.archetypes.read().await;
    let archetypes: Vec<Value> = reg
        .iter()
        .map(|r| with_id("archetype_id", &r.id, &r.body))
        .collect();
    Json(json!({ "archetypes": archetypes })).into_response()
}

async fn get_archetype(State(s): State<Arc<Authored>>, Path(aid): Path<String>) -> Response {
    match s.archetypes.read().await.get(&aid) {
        Some(r) => Json(with_id("archetype_id", &r.id, &r.body)).into_response(),
        None => err(StatusCode::NOT_FOUND, "archetype_not_found", &aid),
    }
}

async fn put_archetype(
    State(s): State<Arc<Authored>>,
    Path(aid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    save(&s.archetypes, "archetype_id", &aid, body).await
}

async fn delete_archetype(State(s): State<Arc<Authored>>, Path(aid): Path<String>) -> Response {
    remove(&s.archetypes, "archetype_not_found", &aid).await
}

/// Shared save path.
///
/// The id is taken from the URL and the body's own id field is discarded rather
/// than trusted: a document that could name its own file is a document that
/// could name somebody else's.
async fn save(reg: &RwLock<Registry>, key: &str, id: &str, mut body: Value) -> Response {
    if let Some(map) = body.as_object_mut() {
        map.remove(key);
    } else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_arguments",
            "the body must be a JSON object",
        );
    }
    match reg.write().await.put(id, body.clone()) {
        Ok(()) => Json(with_id(key, id, &body)).into_response(),
        // A rejected id is the author's mistake to see, not a 500 — the message
        // from `id::check` names what is wrong with it.
        Err(e) => err(StatusCode::BAD_REQUEST, "invalid_arguments", &e.to_string()),
    }
}

async fn remove(reg: &RwLock<Registry>, missing: &str, id: &str) -> Response {
    match reg.write().await.delete(id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => err(StatusCode::NOT_FOUND, missing, id),
        Err(e) => err(StatusCode::BAD_REQUEST, "invalid_arguments", &e.to_string()),
    }
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

    /// Shared state, for a test that makes several requests against one store.
    /// `router` consumes its state, so the `Arc` is what gets reused.
    fn state(base: std::path::PathBuf) -> Arc<Authored> {
        Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("archetype", base.join("archetypes")).unwrap(),
            Accounts::load(base.join("accounts")).unwrap(),
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
        let (s, h) = call(router(state), get("/v1/me/profile/history", Some(a))).await;
        assert_eq!(s, StatusCode::OK);
        let revisions = h["revisions"].as_array().unwrap();
        assert_eq!(revisions.len(), 2);
        assert_eq!(revisions[0]["description"], "Ex-surveyor.");
        assert_eq!(revisions[0]["live"], true);
        assert_eq!(revisions[1]["live"], false);
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
}
