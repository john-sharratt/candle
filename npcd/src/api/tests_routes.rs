//! Tests that pin the whole route table and the role each route needs.
//!
//! A new route fails the table test until its line is written, so its role is
//! chosen on purpose. The tests check that anonymous writes are refused, that
//! changing authored content is admin-only while reading it is not, and that
//! a refused write does not expose a path on this machine. The listings are
//! checked to count the characters that name each document.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use axum::body::Body;

    use axum::http::Request;

    use crate::engine::api as engine_api;

    use super::super::tests_accounts::tests::{call, get, send, sign_in, state, tmp, ADMIN};

    use super::super::tests_characters::tests::author;

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
        let api = api(Roles::default());
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
                // A flood fill on the host — no model, no drain, no engine — so
                // it sits in this table rather than the engine's and answers on
                // a daemon that has none.
                ("/v1/image/cutout", "user"),
                // Members and admins, gated in the handler on the member stamp.
                ("/v1/prompt/authorize", "unauthenticated"),
                // The co-resident models. `admin` including the read, because a
                // guest job **stops the world**: the scheduler evicts the
                // engine's KV working set to make room, and every character in
                // every world stops thinking until it finishes. A request that
                // costs one line to make and a minute of the whole estate to
                // serve wants a hand on it rather than a rate limit.
                ("/v1/guest", "admin"),
                ("/v1/guest/image", "admin"),
                // Free prose on the resident model. `admin`: a free-text
                // generator open to every account is a generator for anything.
                ("/v1/generate/prose", "admin"),
                // One stratum of a life, in the narrator's voice. `admin` with
                // the rest of `/v1/life` — it writes prose into the substrate
                // that a character will believe it remembers.
                ("/v1/life/:who/node/:key/narrate", "admin"),
                // A character's authored life. **`admin` for the reads too**,
                // which is the one place this table departs from the pattern
                // above it, and deliberately: the plan carries the SEED — the
                // arc, the shape, the world events a life hangs on — which is
                // the authoring intent behind a character rather than anything
                // the character is. Every write puts prose into the substrate
                // that a character will believe it remembers, which is a larger
                // act than editing a page of canon.
                ("/v1/life/catalog", "admin"),
                ("/v1/life/:who", "admin"),
                ("/v1/life/:who/seed", "admin"),
                ("/v1/life/:who/node/:key", "admin"),
                ("/v1/life/:who/day", "admin"),
                ("/v1/life/:who/day/:date", "admin"),
                ("/v1/life/:who/day/:date/consequences", "admin"),
                ("/v1/life/:who/generate", "admin"),
                ("/v1/life/:who/job", "admin"),
                ("/v1/life/:who/cancel", "admin"),
                // The authored corpus. `user` to read rather than open, because
                // this one *enumerates* — see the route's own comment.
                ("/v1/mind/list", "user"),
                ("/v1/mind/entry", "user"),
                ("/v1/mind/entry", "admin"),
                ("/v1/mind/fields", "user"),
                ("/v1/mind/fields", "admin"),
                // An admin's drafts: read anywhere, discarded and published on
                // the best member.
                ("/v1/mind/drafts", "admin"),
                ("/v1/mind/drafts", "admin"),
                ("/v1/mind/publish", "admin"),
                ("/v1/mind/sync", "admin"),
                // The cast: signed in, then ownership per record. Two
                // registrations because the listing reads anywhere and the
                // create goes to the best member.
                ("/v1/npc", "user"),
                ("/v1/npc", "user"),
                // The MCP server: open, naming its own caller and sending
                // nobody to sign in with the challenge.
                ("/v1/npc/mcp", "unauthenticated"),
                // The whole cast in and out as records, for backup/restore —
                // `admin`, a full-estate read and a bulk write.
                ("/v1/npcs/export", "admin"),
                ("/v1/substrate/npcd/objects/export", "admin"),
                ("/v1/substrate/npcd/objects/import", "admin"),
                ("/v1/npcs/import", "admin"),
                // Who owns what, and handing a character over: `admin`, the one
                // write that crosses ownership.
                ("/v1/npcs/owners", "admin"),
                ("/v1/npc/:nid/owner", "admin"),
                ("/v1/npc/:nid", "user"),
                ("/v1/npc/:nid/tags", "user"),
                ("/v1/npc/:nid/hidden", "user"),
                // Suspending one character, and every character the caller
                // may manage: `user` plus the ownership check inside, which an
                // admin passes for any character.
                ("/v1/npc/:nid/suspend", "user"),
                ("/v1/npcs/suspend", "user"),
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
                ("/v1/npc/:nid/reflect", "user"),
                ("/v1/npc/:nid/dream", "user"),
                ("/v1/npc/:nid/modulation", "user"),
                // A portrait, and the bytes back. `user` rather than open: an
                // id is a content hash and unguessable, and unguessable is not
                // a permission.
                ("/v1/npc/:nid/portrait", "user"),
                // Drawing one, and what it can be drawn with. `user` like the
                // upload: it is a portrait of the caller's own character, and
                // admin-only would mean nobody could use the button on the
                // character they just made.
                ("/v1/npc/:nid/portrait/generate", "user"),
                // The character's own Discord bot: the owner's, like the
                // portrait, with the 404 of a missing character for anyone
                // else.
                ("/v1/npc/:nid/discord", "user"),
                // What any character's bot has done on Discord: `admin`, a
                // read across ownership.
                ("/v1/npc/:nid/discord/trace", "admin"),
                ("/v1/image/:iid", "user"),
                // A signed link to the same bytes: open, because the signature
                // over the id and an expiry is the permission, and the handler
                // refuses anything else.
                ("/v1/image/link/:iid", "unauthenticated"),
                // A picture handed in through an upload link: the signature is
                // the permission, checked before the body is read.
                ("/v1/image/upload", "unauthenticated"),
                // The caller's own account. `/v1/me` itself is the host's. The
                // read and the write are registered apart for their dispatch.
                ("/v1/me/profile", "guest"),
                ("/v1/me/profile", "guest"),
                ("/v1/me/profile/history", "guest"),
                ("/v1/me/profile/history/:rev", "guest"),
                ("/v1/me/profile/restore/:rev", "guest"),
                ("/v1/me/unique-name", "guest"),
                // The caller's own Discord link: the read, the unlink, and a
                // code to link with.
                ("/v1/me/discord", "guest"),
                ("/v1/me/discord", "guest"),
                ("/v1/me/discord/link", "guest"),
            ]
        );
    }

    /// The same whole-table assertion, for the engine's routes.
    ///
    /// **These are merged into the router separately** (the tenant builds
    /// `engine::api(..)` and merges it), so the table above — which is built
    /// from this crate's `api(..)` alone — never saw them, and the routes that
    /// reach a character somebody else owns (`/simulate`, `/pulse/broadcast`,
    /// both `admin`) had no guard test at all. This closes that: read the
    /// `user` rows against their ownership checks, and the two
    /// `creator` rows (the world channel's write side and a world posting)
    /// against the fact that they reach a whole world at once, which ownership
    /// cannot express.
    #[test]
    fn the_engine_route_table_is_what_we_think_it_is() {
        let api = engine_api(Roles::default());
        let got: Vec<(&str, &str)> = api
            .declared()
            .iter()
            .map(|r| (r.path, r.min.as_str()))
            .collect();

        assert_eq!(
            got,
            [
                // The substrate as it actually is, and the character's memory —
                // reads of one owned character.
                ("/v1/npc/:nid/substrate", "user"),
                ("/v1/npc/:nid/substrate/layer/:layer", "user"),
                ("/v1/npc/:nid/substrate/turn/:layer/:turn", "user"),
                ("/v1/npc/:nid/memory", "user"),
                // The substrate inspector: admin, like `/v1/substrate` — it reads
                // every character's conversations, whoever owns them.
                ("/v1/substrate/npcd/system-prompt", "admin"),
                ("/v1/substrate/npcd/layer/:name", "admin"),
                ("/v1/substrate/npcd/layer/:name/toggle", "admin"),
                ("/v1/substrate/npcd/timeline/:tl", "admin"),
                // The scenario harness: admin, because it spends the card's time
                // and reports the daemon's own prompt.
                ("/v1/simulate", "admin"),
                // Instruments over one owned character.
                ("/v1/npc/:nid/projection", "user"),
                ("/v1/npc/:nid/projection/:tick", "user"),
                ("/v1/npc/:nid/system-prompt", "user"),
                ("/v1/npc/:nid/prompt-tokens", "user"),
                ("/v1/npc/:nid/monitor", "user"),
                ("/v1/npc/:nid/project", "user"),
                ("/v1/npc/:nid/perceive", "user"),
                // Interactions with one owned character.
                ("/v1/npc/:nid/interaction", "user"),
                ("/v1/interaction/:ix", "user"),
                ("/v1/interaction/:ix/inject", "user"),
                ("/v1/interaction/:ix/stream", "user"),
                // Messaging a character on its handset.
                ("/v1/npc/:nid/message", "user"),
                // Lodging a mission for a character and reading how it went, and
                // calling one off.
                ("/v1/npc/:nid/mission", "user"),
                ("/v1/npc/:nid/mission/cancel", "user"),
                ("/v1/npc/:nid/mission/step", "user"),
                ("/v1/npc/:nid/ask", "user"),
                ("/v1/npc/:nid/mind_control", "user"),
                // The world's open channel: reading is a user's, speaking on it
                // is a creator's, because it reaches every character in a world.
                ("/v1/world/:wid/channel", "user"),
                ("/v1/world/:wid/channel", "creator"),
                ("/v1/world/:wid/posting", "creator"),
                // The world's Discord server: a creator's, as the channel's
                // write side is.
                ("/v1/world/:wid/discord", "creator"),
                // The command list is a user's. The catalog and its
                // calibration are the host's `/v1/tools`.
                ("/v1/commands", "user"),
                // Pulse: the cast's loop as an instrument, and reaching into it.
                ("/v1/pulse", "user"),
                ("/v1/pulse/census", "user"),
                ("/v1/pulse/world", "user"),
                ("/v1/npc/:nid/pulse", "user"),
                // The influence primitive: a line into one owned character's
                // world. `user` plus the ownership check, exactly like `pulse`.
                ("/v1/npc/:nid/direct", "user"),
                ("/v1/npc/:nid/window", "user"),
                ("/v1/npc/:nid/scan", "user"),
                ("/v1/npc/:nid/journal", "user"),
                ("/v1/npc/:nid/journal/draft", "user"),
                ("/v1/npc/:nid/journal/:eid", "user"),
                // Broadcast and announce reach characters the caller does not
                // own — the two pulse routes that are admin for exactly that
                // reason.
                ("/v1/pulse/broadcast", "admin"),
                ("/v1/pulse/announce", "admin"),
                // The command table and clearing the board — admin, they drive
                // the whole cast.
                ("/v1/pulse/command-table", "admin"),
                // The operator's house bridge — work a device through an
                // in-network agent, admin like the table.
                ("/v1/pulse/bridge", "admin"),
                // Set a device's notes and credentials on its world part, admin.
                ("/v1/pulse/device", "admin"),
                // Pair with a device and keep its credentials, admin.
                ("/v1/pulse/pair", "admin"),
                // What the guardian has found and done — names every
                // character, so admin.
                ("/v1/pulse/guardian", "admin"),
                ("/v1/pulse/missions/cancel", "admin"),
                // Starting named characters over: admin, and never undone.
                ("/v1/pulse/start-over", "admin"),
                ("/v1/pulse/missions/generate", "admin"),
                // The pool's read and its clearing are registered apart for
                // their dispatch: any member reads it, every member clears the
                // worlds it runs.
                ("/v1/pulse/missions/pool", "admin"),
                ("/v1/pulse/missions/pool", "admin"),
                // The table's operations — admin, like the table itself. The
                // list and a review are registered apart, as the pool is.
                ("/v1/pulse/operations", "admin"),
                ("/v1/pulse/operations", "admin"),
                ("/v1/pulse/operations/:wid/:oid", "admin"),
                ("/v1/pulse/operations/:wid/:oid/cancel", "admin"),
                ("/v1/pulse/operations/:wid/:oid/step", "admin"),
                ("/v1/pulse/operations/:wid/:oid/retry", "admin"),
                ("/v1/pulse/operations/:wid/:oid/document", "admin"),
                // Generation on the resident model, over the caller's own cast.
                ("/v1/generate/description", "user"),
                ("/v1/generate/description/stream", "user"),
                ("/v1/generate/name", "user"),
                ("/v1/generate/attributes", "user"),
                ("/v1/image/generate", "user"),
                ("/v1/image/models", "user"),
                ("/v1/image/queue", "user"),
                // The push stream.
                ("/ws/events", "user"),
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
        sign_in(&state, a).await;
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
