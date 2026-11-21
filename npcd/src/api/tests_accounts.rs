//! Tests for the account, profile and name routes and their role gates.
//!
//! Each test runs in a directory of its own and signs in through the gateway's
//! headers as an admin, a user or a guest. The tests check that a profile
//! revision stays readable after it is superseded, that a restore produces a
//! new revision, and that one author cannot read or write another's profile.
//! They also check that a draw needs the admin role where the route requires it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use axum::body::Body;

    use axum::http::Request;

    use cluster::member::FROM_MEMBER;

    use tower::ServiceExt;

    use web::cluster::ClusterPeer;

    use crate::authorize::Authorization;

    /// A directory of this test's own.
    ///
    /// The tag names it and a counter keeps it unique. It used to be the tag
    /// plus `SystemTime::now()`, which is only as fine as the platform clock —
    /// around 15ms on Windows — so two tests sharing a tag, or starting in the
    /// same tick, were handed the same directory and wrote over each other.
    pub(in super::super) fn tmp(tag: &str) -> std::path::PathBuf {
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

    pub(in super::super) fn app() -> Router {
        router(state(tmp("state")))
    }

    /// The subject the tests treat as an admin. The tests' other subjects are
    /// listed as users, which is what makes the two easy to tell apart in a
    /// request; any subject the table does not name is a guest.
    pub(in super::super) const ADMIN: &str = "google-admin";

    /// The subjects the tests sign in as who are on the role table as users.
    pub(in super::super) const USERS: [&str; 6] = [
        "google-1",
        "google-2",
        "google-user",
        "google-someone-else",
        "google-not-an-admin",
        "google-plain-user",
    ];

    /// The role table these tests run under: [`ADMIN`] and [`USERS`].
    pub(in super::super) fn people() -> String {
        let mut yaml = format!("people:\n  - sub: {ADMIN}\n    role: admin\n");
        for u in USERS {
            yaml.push_str(&format!("  - sub: {u}\n    role: user\n"));
        }
        yaml
    }

    /// Shared state, for a test that makes several requests against one store.
    /// `router` consumes its state, so the `Arc` is what gets reused.
    pub(in super::super) fn state(base: std::path::PathBuf) -> Arc<Authored> {
        // The real ingest, against the test's own personalities directory, so a
        // test that seeds a personality with a portrait exercises the path the
        // daemon takes rather than a hand-built map that cannot disagree with it.
        let personalities = Registry::load("personality", base.join("personalities")).unwrap();
        // A real substrate, in the test's own directory — the registry has no
        // in-memory mode, and one that only existed for tests would be a
        // second code path exercised by nothing else.
        let npcs = Npcs::load(&base).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &base.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &base.join("images")).unwrap();
        let personality_portraits = crate::personality_portrait::ingest(
            &personalities,
            &base.join("personalities"),
            &images,
        );
        Arc::new(Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            personalities,
            accounts,
            npcs,
            // Parsed from YAML rather than built by hand, so these tests fail
            // if the config shape the deployment writes stops matching the type
            // the daemon reads.
            serde_yaml::from_str(&people()).unwrap(),
            // No mind, so no libraries — the collections route answers with
            // two empty collections, which is the truth for a test directory.
            crate::collections::Libraries::load(&crate::projection::Source::resolve(None).unwrap()),
            // The mind editor points at the test's own directory, so a test
            // that writes a document writes it here and nowhere near a real
            // one. `mind_state` below is the same thing with content seeded.
            Mind::new(Some(base.clone())),
            images,
            personality_portraits,
        ))
    }

    /// The headers the gateway sets for a signed-in caller.
    ///
    /// This is the whole authentication story on this side: the gateway
    /// resolved a session cookie and named the caller. `npcd` believes it
    /// because the server clears these on ingress from every peer it does not
    /// trust (`web::Builder::trust`), which is asserted in `web/tests/roles.rs`
    /// and `host/src/server.rs` rather than here — an in-process router cannot
    /// observe its own ingress.
    pub(in super::super) fn signed_in(sub: &str) -> Vec<(&'static str, String)> {
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

    pub(in super::super) async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, body)
    }

    pub(in super::super) fn get(path: &str, sub: Option<&str>) -> Request<Body> {
        let mut b = Request::builder().uri(path);
        for (k, v) in sub.map(signed_in).unwrap_or_default() {
            b = b.header(k, v);
        }
        b.body(Body::empty()).unwrap()
    }

    pub(in super::super) fn send(
        path: &str,
        method: &str,
        sub: &str,
        body: Value,
    ) -> Request<Body> {
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
    pub(in super::super) fn bytes(
        path: &str,
        method: &str,
        sub: &str,
        body: Vec<u8>,
    ) -> Request<Body> {
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
    pub(in super::super) async fn call_raw(
        app: Router,
        req: Request<Body>,
    ) -> (StatusCode, Vec<u8>) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, bytes.to_vec())
    }

    /// Sign `sub` in as the host's `/v1/me` does for npcd: its account,
    /// created on first sight.
    pub(in super::super) async fn sign_in(state: &Arc<Authored>, sub: &str) -> Value {
        let mut headers = HeaderMap::new();
        for (k, v) in signed_in(sub) {
            headers.insert(k, v.parse().unwrap());
        }
        let id = crate::identity::identify(&headers).expect("signed in");
        account(state, &id).await.expect("the account is written")
    }

    #[tokio::test]
    async fn a_signed_caller_gets_an_account_created_on_first_sight() {
        let me = sign_in(&state(tmp("first-sight")), "google-1").await;
        assert_eq!(me["email"], "wren@example.com");
        assert_eq!(me["display"], "Wren S");
        assert!(me["user_id"].as_str().unwrap().starts_with("u_"));
    }

    /// **A prompt is authorized for a member or an admin, never a user**, and
    /// only an admin mints one without the check. With no engine the check
    /// cannot run, which a member and an admin hear as `check_unavailable`.
    #[tokio::test]
    async fn prompt_authorization_is_for_members_and_admins() {
        let st = state(tmp("authorize"));
        let ask = |sub: &str, mint: bool| {
            send(
                "/v1/prompt/authorize",
                "POST",
                sub,
                json!({ "purpose": "image", "prompt": "a lighthouse", "mint": mint }),
            )
        };
        let (status, body) = call(router(Arc::clone(&st)), ask("google-1", false)).await;
        assert_eq!(
            (status, body["error"].as_str()),
            (StatusCode::FORBIDDEN, Some("members_only"))
        );

        let (status, body) = call(router(Arc::clone(&st)), ask(ADMIN, true)).await;
        assert_eq!(status, StatusCode::OK);
        let a: Authorization = serde_json::from_value(body["authorization"].clone()).unwrap();
        assert_eq!(
            st.authorizations
                .verify(&a, "image", "a lighthouse", now_ms()),
            Ok(()),
            "an admin's mint is this member's signature over the prompt"
        );

        let (status, body) = call(router(Arc::clone(&st)), ask(ADMIN, false)).await;
        assert_eq!(
            (status, body["error"].as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, Some("check_unavailable"))
        );

        let from_member = |mint: bool| {
            let mut req = Request::builder()
                .method("POST")
                .uri("/v1/prompt/authorize")
                .header("content-type", "application/json")
                .header(FROM_MEMBER, "00000000000000ab")
                .body(Body::from(
                    json!({ "purpose": "image", "prompt": "a lighthouse", "mint": mint })
                        .to_string(),
                ))
                .unwrap();
            req.extensions_mut().insert(ClusterPeer { cluster: 0 });
            req
        };
        let (status, body) = call(router(Arc::clone(&st)), from_member(false)).await;
        assert_eq!(
            (status, body["error"].as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, Some("check_unavailable")),
            "a member's ask runs the check"
        );
        let (status, body) = call(router(st), from_member(true)).await;
        assert_eq!(
            (status, body["error"].as_str()),
            (StatusCode::FORBIDDEN, Some("admins_only")),
            "only an admin mints"
        );
    }

    /// The account is the caller's own: profile routes need a signed-in caller.
    #[tokio::test]
    async fn without_the_gateways_headers_nobody_is_signed_in() {
        let (status, body) = call(app(), get("/v1/me/profile", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"], "unauthorized");
    }

    /// The subject is the account key, so a caller who sends everything *but*
    /// a subject must not reach an account — least of all the one their email
    /// happens to name.
    #[tokio::test]
    async fn a_caller_with_no_subject_cannot_borrow_an_email_to_become_someone() {
        let req = Request::builder()
            .uri("/v1/me/profile")
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

        // Signing in is what creates the record.
        sign_in(&state, a).await;

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
        sign_in(&state, a).await;

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
        sign_in(&state, a1).await;
        sign_in(&state, a2).await;

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

        sign_in(&state, a1).await;
        sign_in(&state, a2).await;

        let req = send("/v1/me/profile", "PUT", a1, json!({"description": "mine"}));
        call(router(state.clone()), req).await;

        // The second user's profile is untouched — there is no id in the URL to
        // point at somebody else in the first place, which is the design.
        let (_, p2) = call(router(state), get("/v1/me/profile", Some(a2))).await;
        assert_eq!(p2["description"], "");
    }

    pub(in super::super) fn set_name(a: &str, name: &str) -> Request<Body> {
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
        sign_in(&state, a).await;

        let (s, me) = call(router(state.clone()), set_name(a, "ridge-walker")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(me["unique_name"], "ridge-walker");

        // The provider re-asserts `Wren S` on every sign-in; the chosen name
        // must not be reverted by it.
        let me = sign_in(&state, a).await;
        assert_eq!(me["unique_name"], "ridge-walker");
        assert_eq!(me["display"], "Wren S");
    }

    /// The one place one author's data can refuse another's write. A character
    /// addresses a person by this name, so two of them is an ambiguous target.
    #[tokio::test]
    async fn two_authors_cannot_share_one_name() {
        let state = state(tmp("clash"));
        let (a1, a2) = ("google-1", "google-2");
        sign_in(&state, a1).await;
        sign_in(&state, a2).await;

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
        sign_in(&state, a).await;

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

    /// **A restricted draw is an admin's ask, twice over.** The route table
    /// puts `/v1/guest/image` at `admin`; the handler *also* requires `admin`
    /// for `lora: restricted` specifically, because that flag waives the
    /// prompt's compliance gate and must not ride along if the route's level
    /// is ever relaxed. This pins the contract from the outside: a caller
    /// below admin never draws restricted, and an admin's ask gets past
    /// authorisation — to the engine check, since a test daemon has none.
    #[tokio::test]
    async fn a_restricted_draw_needs_the_admin_role() {
        let st = state(tmp("restricted-draw"));
        let body = json!({ "prompt": "a lantern", "lora": "restricted" });
        let (s, _) = call(
            router(st.clone()),
            send("/v1/guest/image", "POST", "g1", body.clone()),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, b) = call(
            router(st.clone()),
            send("/v1/guest/image", "POST", ADMIN, body),
        )
        .await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{b}");
    }

    /// **A guest — signed in, not on the role table — owns nothing and runs
    /// nothing**: creating or listing characters and the model routes are the
    /// table's people's, refused before the handler runs.
    #[tokio::test]
    async fn a_guest_owns_nothing_and_runs_nothing() {
        let st = state(tmp("guest"));
        for (path, method) in [("/v1/npc", "POST"), ("/v1/image/cutout", "POST")] {
            let (s, out) = call(
                router(st.clone()),
                send(path, method, "google-stranger", json!({})),
            )
            .await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{method} {path}: {out}");
            assert_eq!(out["role"], "guest", "{out}");
        }
        let (s, _) = call(router(st.clone()), get("/v1/npc", Some("google-stranger"))).await;
        assert_eq!(s, StatusCode::FORBIDDEN);

        // Their own account is the one thing a guest owns.
        sign_in(&st, "google-stranger").await;
        let req = send(
            "/v1/me/profile",
            "PUT",
            "google-stranger",
            json!({"description": "Just looking."}),
        );
        let (s, p) = call(router(st.clone()), req).await;
        assert_eq!(s, StatusCode::OK, "{p}");
        let (s, p) = call(router(st), get("/v1/me/profile", Some("google-stranger"))).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(p["description"], "Just looking.");
    }

    /// The console's own draw route (`/v1/image/generate`) sits at `user`, so
    /// the handler's re-check is the ONLY thing between an ordinary user and
    /// the restricted checkpoint with its waived prompt gate. Called directly
    /// here, exactly as that route calls it.
    #[tokio::test]
    async fn a_users_restricted_ask_is_refused_by_the_handler_itself() {
        let st = state(tmp("restricted-handler"));
        let mut headers = axum::http::HeaderMap::new();
        for (k, v) in signed_in("g1") {
            headers.insert(k, v.parse().unwrap());
        }
        let body: crate::guest_routes::ImageBody =
            serde_json::from_value(json!({ "prompt": "a lantern", "lora": "restricted" })).unwrap();
        let res = crate::guest_routes::post_image(State(st), headers, Json(body)).await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
    }
}
