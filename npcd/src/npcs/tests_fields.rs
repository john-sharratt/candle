//! Checks the field rules: the bound on each authoring layer, id shape, refusal
//! of bad input, and the wire shape.
//!
//! A record is rewritten whole on every edit, so every layer must stop growing
//! at its cap while staying editable. Invalid fields are refused rather than
//! defaulted, and a refused write leaves the revision alone. A hidden character
//! moves no count that a poller could read, which is why the listings can be
//! decorated with counts without leaking it.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use crate::engine::bodies::NpcList;

    use super::*;

    use super::super::tests_lifecycle::tests::{body, ident, nid, tmp, ME};

    /// **Each authoring layer is bounded, and reaching the bound still leaves
    /// the character editable.**
    ///
    /// These are the only collections on the record that grow, and the record is
    /// rewritten whole on every write, carried forward by every compaction and
    /// relocated by every maintenance pass. An uncapped layer is a character
    /// that gets more expensive to keep every time somebody saves.
    ///
    /// The cap is on appending. Revising or deleting an existing entry works at
    /// the limit — a full character that could not be edited down would be a
    /// worse trap than the growth.
    #[test]
    fn an_authoring_layer_stops_growing_but_stays_editable() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());

        for i in 0..MAX_BELIEFS {
            let b = json!({ "statement": format!("belief {i}") });
            n.put_belief(id, ME, &format!("b{i}"), &b, 2_000)
                .unwrap_or_else(|e| panic!("belief {i} refused: {e:?}"));
        }
        assert!(
            matches!(
                n.put_belief(id, ME, "one-too-many", &json!({ "statement": "no" }), 3_000),
                Err(NpcError::Invalid("beliefs"))
            ),
            "the layer grew past its cap"
        );

        // At the cap, an existing belief still revises...
        n.put_belief(id, ME, "b0", &json!({ "statement": "revised" }), 4_000)
            .expect("an existing entry must stay editable at the cap");
        let view = n.view();
        let held = view.payload(id).expect("still there");
        assert_eq!(
            held.beliefs.len(),
            MAX_BELIEFS,
            "revising must not have appended"
        );
        assert_eq!(
            held.beliefs
                .iter()
                .find(|b| b.belief_id == "b0")
                .map(|b| b.statement.as_str()),
            Some("revised")
        );
        // ...and deleting one makes room again.
        assert!(n.delete_belief(id, ME, "b0", 5_000).unwrap());
        n.put_belief(id, ME, "fresh", &json!({ "statement": "yes" }), 6_000)
            .expect("a deletion frees a slot");
    }

    /// **`put_strategy_self` writes the character's own agency owner-blind.**
    ///
    /// The effector device is the character's own body and carries no operator
    /// owner, so this resolves the owner from the record and lands the write in
    /// the same `agency` layer the owner-scoped `put_strategy` writes — the
    /// bridge the projected store depends on (effector design §9.2).
    #[test]
    fn put_strategy_self_resolves_the_owner_and_writes_the_agency() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());

        // No owner passed — the caller does not have one.
        let (_, landed) = n
            .put_strategy_self(
                id,
                StrategyTarget::Id("hold-the-gate"),
                &json!({ "statement": "Hold the toll gate.", "state": "active" }),
                2_000,
            )
            .expect("the owner-blind write lands");
        assert_eq!(landed, "hold-the-gate");

        let view = n.view();
        let held = view.payload(id).expect("still there");
        let strat = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "hold-the-gate")
            .expect("the strategy is in the agency layer");
        assert_eq!(strat.statement, "Hold the toll gate.");
        assert_eq!(strat.state, "active");

        // A character nobody has is absent, the same as every other read path —
        // never a panic, never a write under a mislaid owner.
        assert!(matches!(
            n.put_strategy_self(
                id + 1,
                StrategyTarget::Id("x"),
                &json!({ "statement": "no" }),
                3_000
            ),
            Err(NpcError::NotFound)
        ));
    }

    #[test]
    fn filters_narrow_and_do_not_silently_pass_everything() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let mut a = body("Varek");
        a["tags"] = json!(["north", "campaign-2"]);
        n.create(&ident("u1"), ME, &a, 1_000).unwrap();
        let mut b = json!({ "name": "Ilse", "world_id": "earth", "personality_id": "commander" });
        b["tags"] = json!(["market"]);
        n.create(&ident("u1"), ME, &b, 1_001).unwrap();
        let n = n.view();

        let by_tag = |t| {
            n.list(
                ME,
                &Filter {
                    tag: Some(t),
                    ..Default::default()
                },
            )
        };
        assert_eq!(by_tag("north").len(), 1);
        assert_eq!(by_tag("market").len(), 1);
        assert_eq!(by_tag("nowhere").len(), 0);

        // Free text reaches the name and the tags, case-insensitively.
        let q = |s| {
            n.list(
                ME,
                &Filter {
                    q: Some(s),
                    ..Default::default()
                },
            )
        };
        assert_eq!(q("VAREK").len(), 1);
        assert_eq!(q("mar").len(), 1);
        assert_eq!(q("zzz").len(), 0);

        let w = |id| {
            n.list(
                ME,
                &Filter {
                    world_id: Some(id),
                    ..Default::default()
                },
            )
        };
        assert_eq!(w("battle-cities").len(), 1);
        assert_eq!(w("earth").len(), 1);
        assert_eq!(w("sandbox").len(), 0);
    }

    /// Ids must not collide, and must not be sequential — a sequential id says
    /// how many characters exist and in what order, which is the enumeration
    /// leak by another route.
    #[test]
    fn minted_ids_are_unique_and_not_sequential() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let mut ids = Vec::new();
        for i in 0..24 {
            let c = n
                .create(&ident("u1"), ME, &body(&format!("N{i}")), 1_000 + i)
                .unwrap();
            ids.push(nid(&c));
        }
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "ids collided");
        assert!(ids.iter().all(|&v| v != 0));
        // Not a counter: consecutive ids differ by more than 1 essentially
        // always. One accidental neighbour would be astronomically unlikely.
        assert!(
            ids.windows(2).filter(|w| w[1].abs_diff(w[0]) == 1).count() == 0,
            "ids look sequential"
        );
    }

    /// A durable record written from a typo is worse than a refused write.
    #[test]
    fn invalid_fields_are_refused_rather_than_defaulted() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        assert!(matches!(
            n.create(
                &ident("u1"),
                ME,
                &json!({ "world_id": "1", "personality_id": "1" }),
                1
            ),
            Err(NpcError::Invalid("name"))
        ));
        assert!(matches!(
            n.create(
                &ident("u1"),
                ME,
                &json!({ "name": "  ", "world_id": "1", "personality_id": "1" }),
                1
            ),
            Err(NpcError::Invalid("name"))
        ));
        assert!(matches!(
            n.create(
                &ident("u1"),
                ME,
                &json!({ "name": "A", "personality_id": "1" }),
                1
            ),
            Err(NpcError::Invalid("world_id"))
        ));

        let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = nid(&c);
        for bad in [
            json!({ "state": "melting" }),
            json!({ "state": "tombstoned" }),
            json!({ "heartbeat_ms": 10 }),
            json!({ "salience_gate": 1.5 }),
            json!({ "hidden": "yes" }),
        ] {
            assert!(
                matches!(n.patch(id, ME, &bad, 2_000), Err(NpcError::Invalid(_))),
                "accepted {bad}"
            );
        }
        // And none of the refusals bumped the revision.
        assert_eq!(n.view().get(id, ME).unwrap()["revision"], 1);
    }

    /// A reference is a file name. Anything that could not become one is
    /// refused here rather than written and discovered at spawn, and the rule
    /// is the registry's own — `world_id: "../etc"` names nothing, but a record
    /// carrying it is a durable reference to a path.
    #[test]
    fn a_reference_that_could_not_be_a_file_name_is_refused() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        for bad in [
            "../etc/passwd",
            "Battle-Cities", // uppercase: the registry is lowercase-only
            "battle cities",
            "-leading",
            "con",
            "",
        ] {
            let b = json!({ "name": "A", "world_id": bad, "personality_id": "commander" });
            assert!(
                matches!(
                    n.create(&ident("u1"), ME, &b, 1),
                    Err(NpcError::Invalid("world_id"))
                ),
                "accepted world_id `{bad}`"
            );
        }
        // A number is no longer a reference: these are slugs, and `4` was the
        // shape the previous `u64` ids took on the wire.
        let numeric = json!({ "name": "A", "world_id": 4, "personality_id": "commander" });
        assert!(matches!(
            n.create(&ident("u1"), ME, &numeric, 1),
            Err(NpcError::Invalid("world_id"))
        ));
    }

    /// The figure the personalities and worlds listings decorate themselves
    /// with. Hidden characters are **included** on purpose: a count that moved
    /// when one was hidden would say so to anybody polling it, which is the
    /// §8.3 leak with a smaller denominator than the roster's.
    #[test]
    fn hiding_a_character_does_not_move_a_count() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let mut b = json!({ "name": "Ilse", "world_id": "earth", "personality_id": "drifter" });
        b["hidden"] = json!(true);
        n.create(&ident("u1"), ME, &b, 1_001).unwrap();
        // Another owner's character still counts: doctrine reaches every
        // character of a personality, not only the publisher's.
        let mut c =
            json!({ "name": "Toll-keeper", "world_id": "earth", "personality_id": "commander" });
        c["tags"] = json!([]);
        n.create(&ident("u2"), "u_other", &c, 1_002).unwrap();

        let view = n.view();
        let worlds = view.counts_by(|x| x.world_id.as_str());
        assert_eq!(worlds.get("battle-cities"), Some(&1));
        assert_eq!(worlds.get("earth"), Some(&2), "the hidden one still counts");

        let people = view.counts_by(|x| x.personality_id.as_str());
        assert_eq!(people.get("commander"), Some(&2), "across owners");
        assert_eq!(people.get("drifter"), Some(&1), "hidden, and still counted");

        // The property that matters: hiding a character moves nothing. An
        // observer polling this cannot tell that anything happened.
        let owned = |view: &Cast| -> BTreeMap<String, usize> {
            view.counts_by(|x| x.personality_id.as_str())
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        };
        let before = owned(&view);
        let ilse: u64 = view
            .list(
                ME,
                &Filter {
                    include_hidden: true,
                    ..Default::default()
                },
            )
            .iter()
            .find(|v| v["name"] == "Ilse")
            .map(nid)
            .unwrap();
        n.patch(ilse, ME, &json!({ "hidden": false }), 1_500)
            .unwrap();
        assert_eq!(
            owned(&n.view()),
            before,
            "un-hiding moved the count, so hiding would too"
        );

        // A deleted character stops counting.
        let id: u64 = n
            .view()
            .list(ME, &Filter::default())
            .iter()
            .find(|v| v["name"] == "Varek")
            .map(nid)
            .unwrap();
        n.delete(id, ME, 2_000).unwrap();
        assert_eq!(
            n.view()
                .counts_by(|x| x.world_id.as_str())
                .get("battle-cities"),
            None
        );
    }

    /// Engine-derived values are absent, not zero. A character nothing has run
    /// has no pending count and no monitor band, and saying `0`/`healthy` would
    /// be a measurement nobody took.
    #[test]
    fn the_wire_shape_reports_engine_values_as_absent() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let v = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();

        assert_eq!(v["monitor"], Value::Null);
        assert_eq!(v["tick"]["pending_events"], Value::Null);
        assert_eq!(v["tick"]["last_tick_ms"], Value::Null);
        // Authored configuration IS present.
        assert_eq!(v["tick"]["heartbeat_ms"], DEFAULT_HEARTBEAT_MS);
        // Ids cross as strings — a u64 above 2^53 would be rounded as a number.
        assert!(v["npc_id"].is_string());
        assert!(v["world_id"].is_string());
        assert!(v["owner_id"].is_string());
    }

    /// **The roster reads as the MCP tools read it** (`engine::bodies::NpcList`):
    /// what `GET /v1/npc` lists is the shape `npc_list` names a character by.
    #[test]
    fn the_roster_reads_as_the_mcp_tools_npc_list() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let v = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let listed: NpcList =
            serde_json::from_value(json!({ "items": n.view().list(ME, &Filter::default()) }))
                .unwrap();
        assert_eq!(listed.items.len(), 1);
        let npc = &listed.items[0];
        assert_eq!(npc.npc_id, v["npc_id"].as_str().unwrap());
        assert_eq!(npc.name, "Varek");
        assert_eq!(npc.state, v["state"].as_str().unwrap());
        assert_eq!(npc.mood, None, "never asked is absent");
    }

    /// **A refused write publishes nothing and appends nothing**
    /// (`docs/npcd_cast_lock_free.md` §2): the cast a reader loads afterwards is
    /// the very one it loaded before, and a restart reads the same.
    #[test]
    fn a_refused_write_publishes_nothing() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());
        let before = n.view();

        assert!(matches!(
            n.put_belief(id, ME, "b", &json!({ "statement": "  " }), 2_000),
            Err(NpcError::Invalid("statement"))
        ));
        assert!(
            Arc::ptr_eq(&before, &n.view()),
            "a refused write published a cast"
        );

        // The one writer lets go before the restart opens the log again.
        drop(n);
        let reopened = Npcs::load(&dir).unwrap().view();
        assert_eq!(reopened.get(id, ME).unwrap()["revision"], 1);
    }
}
