//! Tests for the effector device's path and for composing a piece through it.
//!
//! The device tests drive a real invoke through the in-process router and check
//! that the character reads the world's own line rather than the HTTP envelope, and
//! that a malformed body is answered rather than panicking. The compose tests check
//! that a refusal happens before any engine time is spent. A url is reduced to a
//! router path under the same rules however the model wrote it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use serde_json::json;

    use super::super::tests_support::tests::{device_act, device_daemon, embody, vaulted, WORLD};

    // ── the effector device, enacted ────────────────────────────────────────

    /// A url is stripped to a router path however the model spells it — the
    /// in-fiction host removed, a missing leading slash added, and the bare host
    /// mapped to the near-you index (effector design §7).
    #[test]
    fn a_url_becomes_a_router_path() {
        assert_eq!(device_path("http://local/"), "/");
        assert_eq!(device_path("http://local"), "/");
        assert_eq!(
            device_path("http://local/lift/command-shaft/call"),
            "/lift/command-shaft/call"
        );
        // The bare path, already stripped, is left as it is.
        assert_eq!(device_path("/lift/command-shaft"), "/lift/command-shaft");
        // A path with no leading slash is given one.
        assert_eq!(device_path("lift/command-shaft"), "/lift/command-shaft");
        // Surrounding whitespace does not change the route.
        assert_eq!(device_path("  http://local/history "), "/history");
    }

    /// **`enact_device` really reaches the effector.** Standing on the shaft's top
    /// landing, an `invoke` of the lift's `call` runs the world's own `lift_call`
    /// and reads back the world's own line (the success `detail`, not the HTTP
    /// envelope); a malformed `body` is answered prescriptively rather than
    /// panicking (effector design §5, §6, §12).
    #[tokio::test]
    async fn enact_device_reaches_the_effector_and_acts() {
        const WORLD: &str = "creators-vault";
        let rt = device_daemon();

        // A body on the top landing, with the car away, so the call sets it going.
        let npc_id = 1;
        let body = Runtime::body_id(npc_id);
        let world = rt.hosted.get(WORLD).expect("hosted");
        let landings: Vec<Where> = world.read(|w| w.shaft().to_vec());
        let top = landings.len() - 1;
        world.with(|w| {
            w.enter(&body, "Maker-01".to_string(), landings[top].clone())
                .expect("a real landing")
        });
        rt.bodies.bind(npc_id, WORLD, &body).expect("bound");

        // invoke — reaches `effector_invoke`, the world takes it, and the
        // character reads the world's own line back (the success `detail` from
        // `enact_response`, not the HTTP envelope): a landed lift call comes back
        // as the sentence a body acting any other way would have read.
        let inv = rt
            .enact_device(
                npc_id,
                &device_act(
                    "invoke",
                    json!({ "url": "http://local/lift/command-shaft/call" }),
                ),
                None,
            )
            .await;
        assert!(inv.landed, "the invoke did not land: {inv:?}");
        assert!(
            !inv.answer.trim().is_empty(),
            "a landed invoke read back an empty line: {inv:?}"
        );
        // The world's own words, not a JSON envelope — the same prose a character
        // reads in its `<tool_response>`.
        assert!(
            serde_json::from_str::<Value>(&inv.answer).is_err(),
            "the invoke answer should be the world's line, not a JSON envelope: {}",
            inv.answer
        );
        // The feed line carries the status for the operator watching the pulse.
        assert!(
            inv.feed.contains("200"),
            "the feed lost the status: {}",
            inv.feed
        );

        // A malformed body — a prescriptive `bad_json` answer, and no panic.
        let bad = rt
            .enact_device(
                npc_id,
                &device_act(
                    "invoke",
                    json!({ "url": "http://local/lift/command-shaft/call", "body": "{not json" }),
                ),
                None,
            )
            .await;
        assert!(
            !bad.landed,
            "a malformed body must not read as landed: {bad:?}"
        );
        let err: Value = serde_json::from_str(&bad.answer).expect("the bad-body answer is JSON");
        assert_eq!(err["error"], json!("bad_json"), "{err}");
        assert_eq!(err["field"], json!("body"), "{err}");

        // A refusal surfaces its reason on the feed, not a bare status — the same
        // world line the character reads. Riding the lift with no floor named is
        // refused, and the feed carries that line so the pulse shows *why*.
        let refused = rt
            .enact_device(
                npc_id,
                &device_act(
                    "invoke",
                    json!({ "url": "http://local/lift/command-shaft/use" }),
                ),
                None,
            )
            .await;
        assert!(
            !refused.landed,
            "the floorless ride should be refused: {refused:?}"
        );
        assert!(
            !refused.answer.trim().is_empty(),
            "a refusal read back an empty reason: {refused:?}"
        );
        assert!(
            refused.feed.contains(refused.answer.trim()),
            "the feed dropped the refusal reason (a bare status): feed={:?} answer={:?}",
            refused.feed,
            refused.answer
        );
    }

    /// **Composing is refused honestly, never faked.** With no piece of work to
    /// write there is nothing to sit down to; nothing is written into the working
    /// set, and the refusal says why.
    #[tokio::test]
    async fn compose_is_refused_without_a_piece_to_write() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let compose = Act {
            tool: "compose",
            args: serde_json::Map::new(),
        };
        let r = rt.compose(1, &compose).await;
        assert!(!r.landed);
        assert_eq!(r.answer, "You carry no piece of work to write.");
        assert!(r.feed.starts_with("compose ✗"), "{}", r.feed);
    }

    /// **A piece the mission has committed is not composed again**: the
    /// refusal points at the report, before any engine time is spent.
    #[tokio::test]
    async fn compose_is_refused_once_the_piece_is_committed() {
        use crate::engine::mission::{Mission, Origin, Todo, Work};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let doc = "layers/life/quill/3085 The Seal.md";
        let mut m = Mission::new(
            "write it",
            vec![
                Todo::new(format!("write {doc} and commit it")),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Random {
                routine: "r".into(),
            },
        )
        .with_work(Work {
            writes: doc.into(),
            reads: vec![],
            min_words: 250,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools: Vec::new(),
        });
        m.committed(&[doc.to_string()]);
        let hosted = rt.hosted.get(WORLD).expect("hosted");
        hosted.with_sim(|s| s.missions.assign("m1", m));
        let compose = Act {
            tool: "compose",
            args: serde_json::Map::new(),
        };
        let r = rt.compose(1, &compose).await;
        assert!(!r.landed);
        assert_eq!(
            r.answer,
            "layers/life/quill/3085 The Seal.md is written and committed — that step of your \
             mission is done. What is left is your report at the table."
        );
    }

    /// **A piece is written at a desk that composes.** At a chronicle terminal,
    /// which reads documents but does not compose them, a Maker was told "at a
    /// desk, `scan` shows its address" and scanned the terminal it stood at for
    /// twelve hours; the compass gives the way to a writing desk instead.
    #[tokio::test]
    async fn the_compass_to_write_points_past_a_terminal_that_only_reads() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let rt = vaulted();
        let terminal = npc_map::world::Where::new("vault-chronicle", "early-range");
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with(|w| w.enter("m1", "Maker-01".to_string(), terminal).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.embody(1, WORLD, "m1", 0).expect("bound");
        let mission = Mission::new(
            "Write the story.",
            vec![Todo::new("write layers/stories/y.md")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        );
        hosted.with_sim(|s| s.missions.assign("m1", mission));
        let told = rt.mission_compass(1).expect("a compass");
        assert!(
            told.starts_with(
                "Your mission, next: write layers/stories/y.md. A piece is written at a writing \
                 desk: the nearest is in "
            ),
            "{told}"
        );
    }
}
