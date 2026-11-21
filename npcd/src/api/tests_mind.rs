//! Tests for the mind editor's addresses, scope and storage rules.
//!
//! The corpus opens on its sections, and nothing on the wire is a file. No
//! spelling of an address leaves the mind, and an address cannot choose the
//! extension it is stored under. A world's filters apply to browsing, and a
//! field save keeps every comment in the file. Reading needs a session and
//! writing needs an admin.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::Source;

    use super::super::tests_accounts::tests::{call, get, people, send, state, tmp, ADMIN};

    /// State whose mind directory holds a small tree to browse and edit.
    ///
    /// Seeded to the shape of the real mind — a world layer with tags, the two
    /// section folders, the two document folders — so a test asserting the
    /// world filter is asserting against the layout the filter was written for.
    pub(in super::super) fn mind_state(base: std::path::PathBuf) -> Arc<Authored> {
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
                "eras",
                "stories",
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
        let npcs = Npcs::load(&base).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &base.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &base.join("images")).unwrap();
        let st = Arc::new(Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            accounts,
            npcs,
            serde_yaml::from_str(&people()).unwrap(),
            crate::collections::Libraries::load(&crate::projection::Source::resolve(None).unwrap()),
            Mind::new(None),
            images,
            BTreeMap::new(),
        ));
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
    pub(in super::super) fn seed_section(base: &std::path::Path) {
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
    pub(in super::super) fn seed_projection(base: &std::path::Path) {
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

    /// **npcd's layers in `/v1/schema/layers` are the schema, not a copy of
    /// it.**
    ///
    /// They used to be answered by the console's fixture, with the nine layers
    /// written out a second time — and they drifted: the fixture had `action`
    /// at budget priority 95 where the schema said 100. Nothing compared them,
    /// so nothing could report it. Reading the same document the editor writes
    /// is what makes that impossible rather than merely unlikely.
    #[tokio::test]
    async fn the_layer_schema_is_read_from_the_mind_itself() {
        let base = tmp("schema-layers");
        seed_projection(&base);
        let st = mind_state(base.clone());

        let layers = schema_layers(&st).expect("the schema reads");
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
        assert_eq!(
            schema_layers(&st).expect("the schema reads")[0]["window"],
            12345,
            "the reader kept its own copy"
        );
    }

    /// **With no mind, the layers listed are the bundled schema's** — the ones
    /// the engine runs — not an empty list beside nine running layers.
    #[test]
    fn a_daemon_with_no_mind_lists_the_bundled_layers() {
        let base = tmp("schema-layers-bundled");
        let npcs = Npcs::load(&base).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &base.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &base.join("images")).unwrap();
        let st = Arc::new(Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            accounts,
            npcs,
            serde_yaml::from_str(&people()).unwrap(),
            Libraries::load(&Source::resolve(None).unwrap()),
            Mind::new(None),
            images,
            BTreeMap::new(),
        ));
        let names: Vec<String> = schema_layers(&st)
            .expect("the bundled schema reads")
            .iter()
            .map(|l| l["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            [
                "perception",
                "action",
                "interaction",
                "talk",
                "memory",
                "world",
                "environment",
                "agency",
                "relationships",
                "beliefs",
            ]
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
}
