//! Tests for character creation, editing, suspension and the authoring routes.
//!
//! The store is reopened after each write, so an edit is proven durable and not
//! just held in memory. A wrong-typed field is refused and spends no revision.
//! Hidden worlds are tested so that no prefix of their name reveals them, and
//! a character can only be created in the world it belongs to.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_accounts::tests::{
        bytes, call, call_raw, get, send, sign_in, state, tmp, ADMIN,
    };

    use super::super::tests_mind::tests::mind_state;
    use crate::engine::api as engine_api;

    /// `gender` is a choice, so the API enforces the same two options the
    /// console offers — a value a character cannot read must not be storable
    /// just because the caller skipped the GUI.
    #[tokio::test]
    async fn gender_takes_only_the_values_a_character_can_read() {
        let state = state(tmp("gender"));
        let a = "google-1";
        sign_in(&state, a).await;

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
        sign_in(&state, a).await;

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
        sign_in(&state, a).await;

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

    /// Author a world and a personality so a character has something to name.
    /// A registry write is a real YAML file, which is the point: these are the
    /// same documents the mind ships, reached through the same route the
    /// console uses.
    ///
    /// Always as [`ADMIN`], because authoring is an admin's job now — a test
    /// that set this up as an ordinary user would be testing a path the
    /// deployment does not have.
    pub(in super::super) async fn author(state: &Arc<Authored>) {
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
        sign_in(&state, a).await;
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
        sign_in(&st, a).await;
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
                    "state": "suspended"
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
        // The first handle closes first, as a restarting daemon's does: a
        // substrate takes one writer, and a second open beside a live one is
        // refused.
        drop(st);
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
        sign_in(&st, a).await;
        sign_in(&st, b).await;
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

    /// **Suspending, through the routes.** An owner suspends their own
    /// character and a stranger cannot; the listing says which are suspended;
    /// a suspended character's model is not run by the routes that decode for
    /// it; the whole-cast route counts what it changed and reaches everybody's
    /// for an admin; the state comes back off the log after a restart; and
    /// resuming the whole cast resumes only what suspending it suspended.
    #[tokio::test]
    async fn characters_are_suspended_and_resumed_through_the_routes() {
        let dir = tmp("suspend");
        let st = state(dir.clone());
        let (a, b) = ("google-1", "google-2");
        sign_in(&st, a).await;
        sign_in(&st, b).await;
        sign_in(&st, ADMIN).await;
        author(&st).await;

        let mut ids = Vec::new();
        for (who, name) in [(a, "Varek"), (a, "Ilse"), (b, "Wren")] {
            let (s, npc) = call(
                router(st.clone()),
                send(
                    "/v1/npc",
                    "POST",
                    who,
                    json!({ "name": name, "world_id": "battle-cities", "personality_id": "commander" }),
                ),
            )
            .await;
            assert_eq!(s, StatusCode::CREATED, "{npc}");
            assert_eq!(npc["state"], "idle", "a new character runs");
            ids.push(npc["npc_id"].as_str().unwrap().to_string());
        }
        let (varek, ilse, wren) = (&ids[0], &ids[1], &ids[2]);

        // A body without the flag says what it wants.
        let (s, e) = call(
            router(st.clone()),
            send(&format!("/v1/npc/{varek}/suspend"), "POST", a, json!({})),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(e["error"], "suspended_missing");

        // A stranger's character is absent, never forbidden.
        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{wren}/suspend"),
                "POST",
                a,
                json!({ "suspended": true }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        let (s, npc) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{varek}/suspend"),
                "POST",
                a,
                json!({ "suspended": true }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{npc}");
        assert_eq!(npc["state"], "suspended");
        assert_eq!(npc["npc_id"], json!(varek));

        let (_, list) = call(router(st.clone()), get("/v1/npc", Some(a))).await;
        let mut states: Vec<(String, String)> = list["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                (
                    n["name"].as_str().unwrap().to_string(),
                    n["state"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        states.sort();
        assert_eq!(
            states,
            vec![
                ("Ilse".to_string(), "idle".to_string()),
                ("Varek".to_string(), "suspended".to_string())
            ]
        );

        // The routes that run the model refuse a suspended character outright.
        let (s, e) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{varek}/reflect"),
                "POST",
                a,
                json!({ "inner_thoughts": "the gate" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT, "{e}");
        assert_eq!(e["error"], "npc_suspended");
        let (s, e) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{varek}/dream"),
                "POST",
                a,
                json!({ "brief": "the gate is open" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT, "{e}");
        assert_eq!(e["error"], "npc_suspended");

        // The caller's whole cast: one more changed, and the stranger's runs on.
        let (s, v) = call(
            router(st.clone()),
            send("/v1/npcs/suspend", "POST", a, json!({ "suspended": true })),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v, json!({ "suspended": true, "changed": 1 }));
        let (_, w) = call(router(st.clone()), get(&format!("/v1/npc/{wren}"), Some(b))).await;
        assert_eq!(w["state"], "idle");

        // An admin's reaches everybody's.
        let (s, v) = call(
            router(st.clone()),
            send(
                "/v1/npcs/suspend",
                "POST",
                ADMIN,
                json!({ "suspended": true }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v, json!({ "suspended": true, "changed": 1 }));
        let (s, npc) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{ilse}/suspend"),
                "POST",
                ADMIN,
                json!({ "suspended": false }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{npc}");
        assert_eq!(npc["state"], "idle");

        // A restart keeps every one of them as it was left.
        drop(st);
        let again = state(dir);
        for (id, who, want) in [
            (varek, a, "suspended"),
            (ilse, a, "idle"),
            (wren, b, "suspended"),
        ] {
            let (s, npc) = call(
                router(again.clone()),
                get(&format!("/v1/npc/{id}"), Some(who)),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "{npc}");
            assert_eq!(npc["state"], want, "{}", npc["name"]);
        }
        // Resuming the whole cast puts back what suspending it suspended, and
        // leaves the one its owner suspended on its own as it is.
        let (s, v) = call(
            router(again.clone()),
            send(
                "/v1/npcs/suspend",
                "POST",
                ADMIN,
                json!({ "suspended": false }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v, json!({ "suspended": false, "changed": 1 }));
        for (id, who, want) in [(varek, a, "suspended"), (wren, b, "idle")] {
            let (_, npc) = call(
                router(again.clone()),
                get(&format!("/v1/npc/{id}"), Some(who)),
            )
            .await;
            assert_eq!(npc["state"], want, "{}", npc["name"]);
        }
    }

    /// **A trial is refused for the first thing standing in its way**: a
    /// stranger's character is absent — though an admin reaches past whose
    /// it is — a malformed body says what is wrong,
    /// a running character is refused because its context would move under
    /// the trial, and a suspended one reaches the engine — which this daemon
    /// is not running.
    #[tokio::test]
    async fn a_trial_is_refused_until_the_character_is_suspended() {
        let st = state(tmp("trial"));
        let (a, b) = ("google-1", "google-2");
        sign_in(&st, a).await;
        sign_in(&st, b).await;
        sign_in(&st, ADMIN).await;
        author(&st).await;
        let engine = || engine_api(st.roles.clone()).into_routes(st.clone()).router;
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
        assert_eq!(s, StatusCode::CREATED, "{npc}");
        let varek = npc["npc_id"].as_str().unwrap().to_string();
        let path = format!("/v1/npc/{varek}/trial");
        let trial = json!({
            "turn": "Which way is the gate?",
            "tools": [{ "name": "answer", "params": [
                { "name": "value", "type": "string", "required": true, "enum": ["north", "south"] }
            ]}]
        });

        let (s, _) = call(engine(), send(&path, "POST", b, trial.clone())).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "a stranger's character is absent");
        let (s, e) = call(engine(), send(&path, "POST", ADMIN, trial.clone())).await;
        assert_eq!(
            (s, e["error"].as_str()),
            (StatusCode::CONFLICT, Some("npc_not_suspended")),
            "an admin reaches past ownership: {e}"
        );

        let (s, e) = call(
            engine(),
            send(&path, "POST", a, json!({ "turn": "x", "tools": [] })),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{e}");
        assert_eq!(e["error"], "bad_trial");
        assert_eq!(
            e["detail"],
            "a trial needs at least one tool in `tools` to answer through"
        );

        let (s, e) = call(engine(), send(&path, "POST", a, trial.clone())).await;
        assert_eq!(s, StatusCode::CONFLICT, "{e}");
        assert_eq!(e["error"], "npc_not_suspended");

        let (s, _) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{varek}/suspend"),
                "POST",
                a,
                json!({ "suspended": true }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let (s, e) = call(engine(), send(&path, "POST", a, trial)).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{e}");
        assert_eq!(e["error"], "no_engine");
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
        sign_in(&st, a).await;
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

        // Not to a stranger by its id alone — unguessable is not a permission —
        // but to anyone holding a link signed for it, until it expires.
        let (s, _) = call_raw(
            router(st.clone()),
            get(&format!("/v1/image/{image_id}"), None),
        )
        .await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        let link = st.image_links.path(&image_id, now_ms() / 1000, Kept::Drawn);
        let (s, got) = call_raw(router(st.clone()), get(&link, None)).await;
        assert_eq!(s, StatusCode::OK, "{link}");
        assert_eq!(got, png, "the link served other bytes");
        let last = if link.ends_with('0') { '1' } else { '0' };
        let forged = format!("{}{last}", &link[..link.len() - 1]);
        let (s, v) = call(router(st.clone()), get(&forged, None)).await;
        assert_eq!(
            (s, v["error"].as_str()),
            (StatusCode::FORBIDDEN, Some("link_invalid"))
        );
        let (trimmed, _) = link.split_once('?').unwrap();
        let (s, v) = call(router(st.clone()), get(trimmed, None)).await;
        assert_eq!(
            (s, v["error"].as_str()),
            (StatusCode::FORBIDDEN, Some("link_invalid"))
        );
        let expired = st.image_links.path(&image_id, 0, Kept::Drawn);
        let (s, v) = call(router(st.clone()), get(&expired, None)).await;
        assert_eq!(
            (s, v["error"].as_str()),
            (StatusCode::FORBIDDEN, Some("link_expired"))
        );

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

    /// **The narrate route is reachable and refuses legibly.**
    ///
    /// Its companion to the portrait test above, and the same shape: this state
    /// has no engine, so what is pinned is that the route exists, that it is
    /// admin-gated with the rest of `/v1/life`, that a bad node key is refused
    /// before anything else happens, and that a missing prose guest comes back
    /// as a code a console can branch on rather than as a 404 it has to guess
    /// from.
    #[tokio::test]
    async fn a_life_stratum_can_be_narrated_and_the_refusal_is_legible() {
        let base = tmp("narrate");
        let st = mind_state(base);
        // Admin, with the rest of `/v1/life`: this route writes prose into the
        // substrate that a character will believe it remembers.
        let a = ADMIN;
        sign_in(&st, a).await;
        author(&st).await;

        // An ordinary user cannot reach it at all.
        let (s, _) = call(
            router(st.clone()),
            send(
                "/v1/life/commander/node/story/narrate",
                "POST",
                "google-1",
                json!({}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN, "narrate is not admin-gated");

        // A character the registry does not know is refused as *that*, rather
        // than as a life with no plan — the two have different fixes.
        let (_, v) = call(
            router(st.clone()),
            send("/v1/life/nobody/node/story/narrate", "POST", a, json!({})),
        )
        .await;
        assert_eq!(v["error"], "personality_not_found", "{v}");

        // A real character with no plan yet says so, and says it before
        // reaching the guest: there is nothing to narrate, so nothing should be
        // evicted to find that out.
        let (_, v) = call(
            router(st.clone()),
            send(
                "/v1/life/commander/node/story/narrate",
                "POST",
                a,
                json!({}),
            ),
        )
        .await;
        assert_eq!(
            v["error"], "no_life_plan",
            "a character with no plan was reported as something else: {v}"
        );

        // An unknown field is refused rather than ignored — a caller sending a
        // `prompt` is asking for something this route deliberately does not
        // offer, and narrating from the plan instead would look like it was
        // honoured. This is what `Option<Json<_>>` silently defeated: axum turns
        // a parse failure into `None`, so the defaults answered 200.
        let (s, _) = call(
            router(st),
            send(
                "/v1/life/commander/node/story/narrate",
                "POST",
                a,
                json!({ "prompt": "make it up" }),
            ),
        )
        .await;
        assert_eq!(
            s,
            StatusCode::UNPROCESSABLE_ENTITY,
            "an unknown field was accepted"
        );
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

        // A world can be set years ahead. Paused at 5 000 000 ms, 400 years on
        // is exactly 146 097 days later.
        let (s, v) = call(
            router(st.clone()),
            send(
                "/v1/world/ardh/time",
                "PUT",
                ADMIN,
                json!({ "year_offset": 400 }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["year_offset"], 400);
        assert!(v["world_ms"].as_i64().unwrap() >= 5_000_000 + 146_097 * 86_400_000);
        let (_, v) = call(router(st.clone()), get("/v1/world/ardh", None)).await;
        assert_eq!(v["time"]["year_offset"], 400);
        let (s, _) = call(
            router(st.clone()),
            send(
                "/v1/world/ardh/time",
                "PUT",
                ADMIN,
                json!({ "year_offset": "later" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

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

        // One section per facet the document declares, in the order it declares
        // them — not a fixed four.
        let identity = by("identity");
        let ids: Vec<&str> = identity["sections"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["voice", "processing"]);
        assert!(identity["sections"][0]["chars"].as_u64().unwrap() > 0);

        // No doctrine in this document, so no doctrine collection — 69 of the
        // mind's 74 personalities are in that position, and five empty rows on
        // each of them is not a reading of anything.
        assert!(cols.iter().all(|c| c["name"] != "doctrine"));
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
}
