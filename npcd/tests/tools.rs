//! Every act in the catalog, against a world that is actually standing.
//!
//! The specification for what a tool *does*, written as the thing it is: bring a
//! world up deterministically, put a body in it, invoke one act, and assert what
//! changed. One test per property rather than one per tool, because a tool with
//! one test has a demo and a tool with five has a contract.
//!
//! # Why an integration test rather than unit tests beside the code
//!
//! The unit tests in `sim` prove each store keeps its own invariants, and the
//! ones in `engine::enact` prove each act reaches its store. Neither proves the
//! thing that actually has to be true: **that the act a character is offered, in
//! the room it is standing in, does what the catalog says it does.** That joins
//! the map, the world state, the availability rules, the argument bindings and
//! the dispatch — five things that are individually correct and can still
//! disagree.
//!
//! It is also where the two directions get checked against each other. A tool
//! being *offered* and a tool *working* are separate mechanisms: the first is
//! `specs_within` reading live sets, the second is `body::perform` mutating
//! them. A tool can be offered and do nothing, or work and never be offered, and
//! both failures are invisible from inside either half.
//!
//! # Determinism
//!
//! Every world here is built from the shipped maps and the shipped seeds, with
//! no clock and no randomness, so an assertion about *exactly which* names an
//! argument admits is a real assertion rather than a hopeful one.

use npc_map::world::Where;
use npcd::engine::act::Act;
use npcd::engine::body::{perform, Outcome};
use npcd::engine::tools::{self, specs_within, Mode, Within};
use npcd::sim::field::Resource;
use npcd::sim::seed;
use npcd::world::Hosted;
use serde_json::{json, Value};

// ── scaffolding ─────────────────────────────────────────────────────────────

const MAPS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");

fn act(tool: &'static str, args: Value) -> Act {
    Act {
        tool,
        args: args.as_object().expect("an object").clone(),
    }
}

/// The vault, with one Maker in the room orders are taken in.
fn vault() -> Hosted {
    let h = Hosted::load("creators-vault", MAPS).expect("the shipped vault must load");
    h.with(|w| {
        w.enter(
            "m1",
            "Perrin Vastwood",
            Where::new("vault-command", "command-room"),
        )
        .unwrap()
    });
    h
}

/// The tower and the waste, with two Companions in the ruins, kitted out.
fn waste() -> Hosted {
    let h = Hosted::load("battle-cities", MAPS).expect("the shipped world must load");
    let at = Where::new("the-waste", "ruins");
    h.with(|w| {
        w.enter("c1", "Wren", at.clone()).unwrap();
        w.enter("c2", "Soren", at).unwrap();
    });
    h.with_sim(|s| {
        seed::outfit(s, "c1");
        seed::outfit(s, "c2");
    });
    h
}

/// Move a body, so a test can stand somewhere the act it is testing works.
fn stand(h: &Hosted, body: &str, area: &str, node: &str) {
    h.with(|w| w.place(body, Where::new(area, node)).unwrap());
}

/// What the grammar would offer this body, standing where it is.
fn offered(h: &Hosted, body: &str) -> Vec<String> {
    let place = h.place_of(body);
    let company: Vec<String> = h.read(|w| {
        let me = w.actor(body).expect("a body");
        w.actors()
            .filter(|a| a.id != body && a.at == me.at)
            .map(|a| a.name.clone())
            .collect()
    });
    let within = Within {
        company,
        places: npcd::engine::body::reachable(h, body),
        me: h.read(|w| w.actor(body).map(|a| a.name.clone()).unwrap_or_default()),
        ..Default::default()
    };
    let within = h.sim(|s| within.from_sim(s, body, &place));
    specs_within(Mode::Physical, &within)
        .into_iter()
        .map(|t| t.name)
        .collect()
}

/// The values an argument admits, here. `None` when the act is not offered at
/// all, `Some(vec![])` when it is offered with the argument free.
fn admits(h: &Hosted, body: &str, tool: &str, param: &str) -> Option<Vec<String>> {
    let place = h.place_of(body);
    let company: Vec<String> = h.read(|w| {
        let me = w.actor(body).expect("a body");
        w.actors()
            .filter(|a| a.id != body && a.at == me.at)
            .map(|a| a.name.clone())
            .collect()
    });
    let within = Within {
        company,
        places: npcd::engine::body::reachable(h, body),
        me: h.read(|w| w.actor(body).map(|a| a.name.clone()).unwrap_or_default()),
        ..Default::default()
    };
    let within = h.sim(|s| within.from_sim(s, body, &place));
    let specs = specs_within(Mode::Physical, &within);
    let t = specs.iter().find(|t| t.name == tool)?;
    let p = t.params.iter().find(|p| p.name == param)?;
    Some(p.enum_values.clone().unwrap_or_default())
}

fn did(o: &Outcome) -> &str {
    match o {
        Outcome::Did(s) => s,
        other => panic!("expected the act to happen, got {other:?}"),
    }
}

fn refused(o: &Outcome) -> &str {
    match o {
        Outcome::Refused(s) => s,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ── the two directions agree ────────────────────────────────────────────────

/// **Everything offered can be performed, and everything performed is offered.**
///
/// The join that neither half can check alone. `specs_within` decides what a
/// character may call; `body::perform` decides what happens when it does. A tool
/// present in one and absent from the other is a character either spending turns
/// on nothing or being refused for using its own vocabulary — and both look fine
/// from inside the half that is right.
#[test]
fn every_act_a_room_offers_is_an_act_that_is_performed() {
    for h in [vault(), waste()] {
        let body = if h.id() == "creators-vault" {
            "m1"
        } else {
            "c1"
        };
        for name in offered(&h, body) {
            assert!(
                npcd::engine::body::is_of_the_body(&name),
                "`{name}` is offered in {} and nothing performs it",
                h.id()
            );
        }
    }
}

/// **Every act in the catalogue reaches an implementation.**
///
/// [`npcd::engine::body::is_of_the_body`] says a name is *claimed* by a
/// dispatcher; it does not say the dispatcher does anything with it. An act
/// that falls through to the generic "nothing here knows how to …" is declared,
/// described, grammar-bound, offered in a room and chosen by a character — and
/// then does nothing at all. That is the shape of a stub, and it is invisible
/// from either half on its own: the catalogue looks complete and the dispatch
/// compiles.
///
/// Driven by each act's own calibration example, which is valid JSON for that
/// act by construction — `every_tool_carries_calibration_examples` already
/// asserts it parses and names only declared parameters.
///
/// **A refusal is a pass.** A world with no tower refuses `produce`, a
/// character holding nothing refuses `give`, and both are the world working. It
/// is the *fallthrough* that fails: the two strings below are what a dispatcher
/// says when nothing matched the name.
#[test]
fn every_act_in_the_catalog_reaches_an_implementation() {
    let worlds = [(vault(), "m1"), (waste(), "c1")];
    let mut unreached: Vec<&str> = Vec::new();

    for tool in tools::CATALOG.iter() {
        let example = tool.examples.first().expect("every act carries one");
        let args: Value = serde_json::from_str(example.call)
            .unwrap_or_else(|e| panic!("`{}` example is not JSON: {e}", tool.name));
        let a = Act {
            tool: tool.name,
            args: args.as_object().expect("an object").clone(),
        };
        // Reached in *either* world: an act belongs to the vault or to the
        // field, and being refused by the one it does not belong to is the
        // empty-set rule doing its job rather than a missing implementation.
        let reached = worlds.iter().any(|(h, body)| {
            let out = perform(h, body, &a);
            let line = out.line().unwrap_or_default();
            !line.starts_with("Nothing here knows how to")
                && !line.contains("needs to know what it is about")
        });
        if !reached {
            unreached.push(tool.name);
        }
    }

    assert!(
        unreached.is_empty(),
        "{} act(s) are in the catalogue and nothing performs them: {unreached:?}",
        unreached.len()
    );
}

/// Acts whose whole product is what they say, so a world unchanged afterwards
/// is correct rather than suspicious.
///
/// Two groups, and the reason differs:
///
/// * **Readings** — `file_read`, `bench_status` and the rest of
///   [`npcd::engine::body::ANSWERS`]. Answering a question is the work.
/// * **Acts that land somewhere other than [`npcd::sim`]** — speech and
///   attention go into the map, which is where a room's events live, and
///   `wait_for` is armed by the scheduler outside the world entirely.
const CHANGES_NOTHING_IN_THE_SIM: &[&str] = &[
    // Readings.
    "file_read",
    "file_list",
    "library_read",
    "portrait_prompt_read",
    "bench_diff",
    "bench_status",
    "bench_log",
    "bench_blame",
    "read",
    "scan",
    // Land in the map, not the sim.
    "say",
    "tell",
    "ask",
    "gesture",
    "send_image",
    "move_to",
    "follow",
    "observe",
    // Ports a body home, which is a position and so the map's business.
    "recall",
    // Lands in the map like the rest of that group, and its other half — the
    // deadline — is the scheduler's, outside the world entirely.
    "reflect",
    // Reads the record and reports; the piece it presents is unchanged by
    // being presented.
    "creator_present",
];

/// **An act that reports success changed something.**
///
/// The failure this is for does not look like a failure from anywhere else: the
/// act is in the catalogue, the grammar admits it, the dispatcher matches it,
/// it returns `Did`, the character reads a sentence saying it worked — and the
/// world is byte-identical afterwards. Four acts were in exactly that state
/// (`structure_lay_out_scenes`, `structure_test_the_want`,
/// `structure_find_the_slack`, `record_tidy_index`): a character did the
/// reading, said so, and the next one to pick the piece up found no sign of it.
///
/// Compared on the serialised sim, because that is the whole of what a world
/// holds that is not its map — so this cannot be satisfied by changing a field
/// nothing reads.
#[test]
fn every_act_that_reports_success_leaves_the_world_changed() {
    let mut inert: Vec<String> = Vec::new();

    for tool in tools::CATALOG.iter() {
        if CHANGES_NOTHING_IN_THE_SIM.contains(&tool.name) {
            continue;
        }
        let example = tool.examples.first().expect("every act carries one");
        let args: Value = serde_json::from_str(example.call).expect("valid by construction");
        let a = Act {
            tool: tool.name,
            args: args.as_object().expect("an object").clone(),
        };
        for (h, body) in [(vault(), "m1"), (waste(), "c1")] {
            let before = h.sim(|s| serde_json::to_string(s).expect("a sim serialises"));
            let out = perform(&h, body, &a);
            if !out.happened() {
                continue;
            }
            let after = h.sim(|s| serde_json::to_string(s).expect("a sim serialises"));
            if before == after {
                inert.push(format!("{} (in {})", tool.name, h.id()));
            }
        }
    }

    assert!(
        inert.is_empty(),
        "{} act(s) reported success and changed nothing: {inert:#?}",
        inert.len()
    );
}

/// A world instantiates only what it is, and the mechanism is an empty set
/// rather than a flag.
#[test]
fn the_vault_offers_no_act_that_belongs_to_the_battlefield() {
    let h = vault();
    let acts = offered(&h, "m1");
    // `recall` is *not* in this list: the vault has a muster point of its own
    // (`teleport_to: vault-command/command-room`), so porting back to it is a
    // journey a Maker really can make. It is the one act here that looks like
    // Battle Cities' and is not.
    for absent in ["engage", "gather", "produce", "command_tower"] {
        assert!(
            !acts.contains(&absent.to_string()),
            "the vault offered `{absent}`, which nothing there can answer"
        );
    }
    // And the things it *does* have are there. `say` is not among them and is
    // not missing: this Maker is standing alone, and speech needs somebody to
    // hear it — see `tools::SAY`.
    for present in ["move_to", "claim", "read", "reflect"] {
        assert!(
            acts.contains(&present.to_string()),
            "the vault lost `{present}`"
        );
    }
}

#[test]
fn the_waste_offers_what_is_actually_out_there() {
    let h = waste();
    let acts = offered(&h, "c1");
    for present in ["engage", "gather", "give", "equip", "use", "act"] {
        assert!(
            acts.contains(&present.to_string()),
            "the ruins lost `{present}`"
        );
    }
    // No tower on the ground, so nothing that speaks to one.
    assert!(!acts.contains(&"produce".to_string()));
}

// ── arguments bind to what is actually there ────────────────────────────────

#[test]
fn what_may_be_gathered_is_what_is_in_this_ground_and_nothing_else() {
    let h = waste();
    assert_eq!(
        admits(&h, "c1", "gather", "what"),
        Some(vec!["the burnt-out carrier".to_string()])
    );

    stand(&h, "c1", "the-waste", "east-ridge");
    assert_eq!(
        admits(&h, "c1", "gather", "what"),
        Some(vec!["the ore seam".to_string()]),
        "walking to another place did not change what is in the ground"
    );

    stand(&h, "c1", "tower-redoubt", "bridge");
    assert!(
        admits(&h, "c1", "gather", "what").is_none(),
        "`gather` was offered indoors, where there is nothing to work"
    );
}

#[test]
fn a_worked_out_seam_stops_being_offered_rather_than_refusing() {
    let h = waste();
    // Thirty in the carrier, ten an act.
    for _ in 0..3 {
        assert!(perform(
            &h,
            "c1",
            &act("gather", json!({"what":"the burnt-out carrier"}))
        )
        .happened());
    }
    assert!(
        admits(&h, "c1", "gather", "what").is_none(),
        "an act that can take nothing was still offered"
    );
}

/// A machine offers its own states and not another machine's — **less the one
/// it is already in.**
///
/// `Device::set` accepts the current mode, so setting a thing to where it
/// already stands succeeds and changes nothing. Measured live, a character set
/// the accession desk to `reading` three times running while it was already
/// `reading`, told "You set the accession desk to reading" each time. It leaves
/// the branch the same way every other impossible act does — see
/// `Sim::modes_here`.
#[test]
fn a_machine_offers_its_own_states_less_the_one_it_is_in() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "rampart");
    let turret = admits(&h, "c1", "operate", "mode").expect("a turret stands on the rampart");
    assert!(turret.contains(&"air only".to_string()), "{turret:?}");
    assert!(
        !turret.contains(&"hold fire".to_string()),
        "the turret was offered the policy it is already on: {turret:?}"
    );

    stand(&h, "c1", "tower-redoubt", "gatehouse");
    let door = admits(&h, "c1", "operate", "mode").expect("a door stands in the gatehouse");
    // Open is where it stands, so what is left is what it could be moved to.
    assert_eq!(door, vec!["closed", "locked"]);
    assert!(
        !door.contains(&"air only".to_string()),
        "a door was offered a turret's firing policy"
    );
}

#[test]
fn only_what_is_carried_may_be_handed_over_readied_or_used() {
    let h = waste();
    let carried = admits(&h, "c1", "give", "what").expect("company and a pack");
    assert!(carried.contains(&"bolt rounds".to_string()));
    assert!(carried.contains(&"stimpak".to_string()));

    // Readying and using split by what the thing is, not by what is carried.
    let equippable = admits(&h, "c1", "equip", "what").expect("kit");
    assert!(equippable.contains(&"combat armour".to_string()));
    assert!(
        !equippable.contains(&"stimpak".to_string()),
        "a stimpak was offered for readying"
    );
    let usable = admits(&h, "c1", "use", "what").expect("kit");
    assert!(usable.contains(&"stimpak".to_string()));
    assert!(
        !usable.contains(&"bolt rounds".to_string()),
        "ammunition is spent by the simulator, not used by hand"
    );
}

#[test]
fn a_body_carrying_nothing_is_offered_none_of_the_acts_that_need_a_pack() {
    let h = waste();
    h.with_sim(|s| {
        for id in [
            "mono_sword",
            "plasma_rifle",
            "combat_armour",
            "bolt",
            "stimpak",
            "scanner",
        ] {
            let n = s.pack("c1").get(id).map(|i| i.count).unwrap_or(0);
            if n > 0 {
                s.pack_mut("c1").take(id, n);
            }
        }
    });
    let acts = offered(&h, "c1");
    for gone in ["give", "equip", "use"] {
        assert!(
            !acts.contains(&gone.to_string()),
            "`{gone}` was offered to a character carrying nothing"
        );
    }
}

#[test]
fn the_tower_offers_only_what_it_can_pay_for() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "bridge");
    let rich = admits(&h, "c1", "command_tower", "action").expect("a tower");
    assert!(rich.contains(&"relocate".to_string()));

    h.with_sim(|s| {
        let t = s.tower.as_mut().unwrap();
        let all = t.stock_of(Resource::Energy);
        t.draw(Resource::Energy, all);
    });
    let poor = admits(&h, "c1", "command_tower", "action").expect("still a tower");
    assert!(
        !poor.contains(&"relocate".to_string()),
        "a tower with no energy was invited to fold"
    );
    assert!(
        poor.contains(&"drill down".to_string()),
        "the acts it can still afford went with the one it cannot"
    );
}

#[test]
fn only_what_the_stockpile_covers_may_be_made() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "foundry");
    let makeable = admits(&h, "c1", "produce", "what").expect("a foundry");
    assert!(makeable.contains(&"bolt rounds".to_string()));
    assert!(
        !makeable.contains(&"a companion".to_string()),
        "twelve nanobots were offered a fifty-nanobot companion"
    );
}

#[test]
fn a_promise_that_was_never_made_cannot_be_recalled() {
    let h = waste();
    // Nothing owed yet, so `remind` has nothing to name and is not offered.
    assert!(
        admits(&h, "c1", "remind", "which").is_none(),
        "`remind` was offered with nothing to remind anybody of"
    );

    assert!(perform(
        &h,
        "c2",
        &act(
            "promise",
            json!({"to":"Wren","what":"the eastern sweep","by":"dusk"})
        )
    )
    .happened());
    assert_eq!(
        admits(&h, "c1", "remind", "which"),
        Some(vec!["the eastern sweep".to_string()]),
        "the promise just made is what may be recalled"
    );
}

// ── acts change the world, or say why not ───────────────────────────────────

#[test]
fn gathering_moves_the_stuff_from_the_ground_into_the_pack() {
    let h = waste();
    let before = h.sim(|s| s.field.deposit("burnt_carrier").unwrap().remaining);
    let out = perform(
        &h,
        "c1",
        &act("gather", json!({"what":"the burnt-out carrier"})),
    );
    assert!(did(&out).contains("metal"), "{out:?}");
    assert_eq!(
        h.sim(|s| s.field.deposit("burnt_carrier").unwrap().remaining),
        before - 10
    );
    assert_eq!(
        h.sim(|s| s.pack("c1").get("metal").map(|i| i.count)),
        Some(10)
    );
}

#[test]
fn giving_is_all_or_nothing() {
    let h = waste();
    let out = perform(
        &h,
        "c1",
        &act(
            "give",
            json!({"what":"bolt rounds","to":"Soren","count":"25"}),
        ),
    );
    assert!(out.happened(), "{out:?}");
    assert_eq!(h.sim(|s| s.pack("c1").get("bolt").unwrap().count), 35);
    assert_eq!(h.sim(|s| s.pack("c2").get("bolt").unwrap().count), 85);

    let over = perform(
        &h,
        "c1",
        &act(
            "give",
            json!({"what":"bolt rounds","to":"Soren","count":"900"}),
        ),
    );
    assert!(
        refused(&over).contains("35"),
        "the real count should be named: {over:?}"
    );
    assert_eq!(
        h.sim(|s| s.pack("c1").get("bolt").unwrap().count),
        35,
        "a refused hand-over still moved something"
    );
}

#[test]
fn a_consumable_is_spent_and_a_piece_of_gear_is_not() {
    let h = waste();
    perform(
        &h,
        "c1",
        &act("use", json!({"what":"stimpak","on":"Soren"})),
    );
    assert_eq!(h.sim(|s| s.pack("c1").get("stimpak").unwrap().count), 2);
    perform(&h, "c1", &act("use", json!({"what":"advanced scanner"})));
    assert_eq!(
        h.sim(|s| s.pack("c1").get("scanner").unwrap().count),
        1,
        "gear was consumed by being used"
    );
}

#[test]
fn readying_a_thing_takes_it_out_of_what_may_be_readied_and_leaves_it_carried() {
    let h = waste();
    perform(&h, "c1", &act("equip", json!({"what":"combat armour"})));
    assert!(h.sim(|s| s.pack("c1").get("combat_armour").unwrap().equipped));
    let again = admits(&h, "c1", "equip", "what").expect("more kit");
    assert!(
        !again.contains(&"combat armour".to_string()),
        "offered to ready it twice"
    );
    assert!(
        h.sim(|s| s.carried("c1"))
            .contains(&"combat armour".to_string()),
        "readying it lost it"
    );
}

#[test]
fn a_stance_stands_until_it_is_replaced_or_broken_off() {
    let h = waste();
    perform(
        &h,
        "c1",
        &act("engage", json!({"posture":"press","target":"a mech"})),
    );
    assert_eq!(
        h.sim(|s| s.field.stance("c1").unwrap().posture.clone()),
        "press"
    );
    assert_eq!(
        h.sim(|s| s.field.stance("c1").unwrap().target.clone()),
        Some("a mech".into())
    );

    perform(
        &h,
        "c1",
        &act(
            "engage",
            json!({"posture":"fall back","priority":"whatever is firing on us"}),
        ),
    );
    let s = h.sim(|s| s.field.stance("c1").cloned()).expect("a stance");
    assert_eq!(s.posture, "fall back");
    assert_eq!(s.target, None, "the old target survived a new stance");

    perform(&h, "c1", &act("engage", json!({"posture":"break off"})));
    assert!(h.sim(|s| s.field.stance("c1").is_none()));
}

#[test]
fn operating_a_thing_puts_it_into_that_state_for_everybody() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "gatehouse");
    stand(&h, "c2", "tower-redoubt", "gatehouse");
    perform(
        &h,
        "c1",
        &act("operate", json!({"what":"the blast door","mode":"locked"})),
    );

    // **The other body sees the same door in the same state — one world, not
    // two.**
    //
    // Asserted through the *situation*, which is where the state of a machine
    // now reaches a character. This used to read the door and check the answer,
    // which was the only way a body could learn a mode — and it was also the
    // act that returned nothing else, reported success, and looped: `read` is
    // in `body::ANSWERS`, so a character was brought straight back to use what
    // it had learnt and read the same door again. Perception here is pushed the
    // moment anything changes; a body standing in front of a door does not
    // spend a turn finding out that it is shut.
    let seen = h
        .with_both(|w, s| npcd::engine::reach::line(w, s, "c2"))
        .expect("the gatehouse holds something");
    assert!(
        seen.contains("blast door (locked)"),
        "the second body was not told the state the first one set: {seen}"
    );
}

#[test]
fn a_refusal_names_what_the_thing_would_have_taken() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "gatehouse");
    let out = perform(
        &h,
        "c1",
        &act(
            "operate",
            json!({"what":"the blast door","mode":"free fire"}),
        ),
    );
    let why = refused(&out);
    assert!(why.contains("open") && why.contains("locked"), "{why}");
}

#[test]
fn claiming_a_station_makes_it_unclaimable_by_anybody_else() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "foundry");
    stand(&h, "c2", "tower-redoubt", "foundry");
    assert!(perform(&h, "c1", &act("claim", json!({"what":"fabricator 1"}))).happened());

    let taken = perform(&h, "c2", &act("claim", json!({"what":"fabricator 1"})));
    assert!(!taken.happened(), "two bodies took one bay");
}

#[test]
fn an_order_taken_leaves_the_board_and_comes_back_when_it_is_given_up() {
    let h = vault();
    let before = h.sim(|s| s.unheld_orders().len());
    assert_eq!(before, 2);

    let what = "close the longest silence in the record";
    assert!(perform(&h, "m1", &act("claim", json!({"what": what}))).happened());
    assert_eq!(h.sim(|s| s.unheld_orders().len()), 1);

    assert!(perform(&h, "m1", &act("release", json!({}))).happened());
    assert_eq!(
        h.sim(|s| s.unheld_orders().len()),
        2,
        "giving it back did not put it where the next taker looks"
    );
}

#[test]
fn a_verdict_stands_against_the_thing_rather_than_evaporating() {
    let h = vault();
    perform(
        &h,
        "m1",
        &act(
            "record_verdict",
            json!({
                "on": "the third era",
                "judgement": "it cannot be filed as it stands",
                "what_would_change_it": "the dates reconciled with the spans either side"
            }),
        ),
    );
    let v = h.sim(|s| s.ledger.verdicts_on("the third era").len());
    assert_eq!(v, 1);
    assert!(h.sim(|s| s.ledger.verdicts_on("the third era")[0]
        .what_would_change_it
        .is_some()));
}

#[test]
fn folding_the_tower_moves_it_and_spends_what_it_costs() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "bridge");
    let before = h.sim(|s| s.tower.as_ref().unwrap().stock_of(Resource::Energy));
    let out = perform(
        &h,
        "c1",
        &act(
            "command_tower",
            json!({"action":"relocate","x":"-300","y":"180"}),
        ),
    );
    assert!(out.happened(), "{out:?}");
    h.sim(|s| {
        let t = s.tower.as_ref().unwrap();
        assert_eq!((t.at.x, t.at.y), (-300, 180));
        assert!(t.stock_of(Resource::Energy) < before, "a fold cost nothing");
    });
}

#[test]
fn drilling_in_takes_the_fold_away_until_the_tower_surfaces() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "bridge");
    perform(
        &h,
        "c1",
        &act("command_tower", json!({"action":"drill down","depth":"60"})),
    );
    assert_eq!(h.sim(|s| s.tower.as_ref().unwrap().depth), 60);

    let acts = admits(&h, "c1", "command_tower", "action").expect("a tower");
    assert!(
        !acts.contains(&"relocate".to_string()),
        "a buried tower was offered a fold"
    );
    assert!(acts.contains(&"surface".to_string()));

    perform(&h, "c1", &act("command_tower", json!({"action":"surface"})));
    assert_eq!(h.sim(|s| s.tower.as_ref().unwrap().depth), 0);
    assert!(admits(&h, "c1", "command_tower", "action")
        .unwrap()
        .contains(&"relocate".to_string()));
}

#[test]
fn a_batch_spends_the_stock_and_takes_a_queue() {
    let h = waste();
    stand(&h, "c1", "tower-redoubt", "foundry");
    let before = h.sim(|s| s.tower.as_ref().unwrap().stock_of(Resource::Metal));
    let out = perform(
        &h,
        "c1",
        &act(
            "produce",
            json!({"what":"bolt rounds","count":"3","queue":"2"}),
        ),
    );
    assert!(out.happened(), "{out:?}");
    h.sim(|s| {
        let t = s.tower.as_ref().unwrap();
        assert_eq!(t.stock_of(Resource::Metal), before - 60);
        assert_eq!(t.queued().len(), 1);
    });
    assert!(
        !h.sim(|s| s.free_queues()).contains(&"2".to_string()),
        "a queue in use was still offered as free"
    );
}

#[test]
fn recall_brings_a_body_home_from_the_open_ground() {
    let h = waste();
    assert_eq!(h.place_of("c1"), "the-waste/ruins");
    let out = perform(&h, "c1", &act("recall", json!({})));
    assert!(out.happened(), "{out:?}");
    assert_eq!(h.place_of("c1"), "tower-redoubt/muster-hall");
}

#[test]
fn a_coordinate_off_the_map_is_refused_and_a_place_on_it_is_read() {
    let h = waste();
    let off = perform(&h, "c1", &act("scan", json!({"x":"999999","y":"0"})));
    assert!(refused(&off).contains("off the map"), "{off:?}");

    let on = perform(&h, "c1", &act("scan", json!({"at":"the-waste/east-ridge"})));
    assert!(
        did(&on).contains("drone"),
        "the scan missed what is standing there: {on:?}"
    );
}

/// **A scan of a place named the way the grammar names it finds what is there.**
///
/// The test above passes a raw `area/node` key, and passed throughout the whole
/// time this act was broken: `Sim::hostiles` matches on keys, so a key worked
/// and the *name* the grammar actually offers never did. Live, every scan of
/// everywhere came back "Nothing moving" — not an empty room, a failed lookup
/// wearing the same words.
///
/// So this drives the path a character drives: a name out of the same list the
/// `at` arm is built from.
#[test]
fn a_scan_by_the_name_the_grammar_offers_finds_what_is_there() {
    use npcd::engine::body::destinations;
    let h = waste();
    let (name, _) = destinations(&h, "c1")
        .into_iter()
        .find(|(_, w)| w.node == "east-ridge")
        .expect("the east ridge is somewhere c1 can see");

    let out = perform(&h, "c1", &act("scan", json!({ "at": name })));
    assert!(
        did(&out).contains("drone"),
        "a scan by name missed what a scan by key finds: {out:?}"
    );
}

/// And a scan reports **people**, which is the answer that matters in a
/// building. Hostiles and deposits are the waste's vocabulary; a vault has
/// neither, so before this a scan there could only ever say nothing.
#[test]
fn a_scan_says_who_is_standing_there() {
    let h = vault();
    // Whatever the fixture's first destination is — the point is that somebody
    // standing in a room this character can see gets reported, not which room.
    let (name, place) = npcd::engine::body::destinations(&h, "m1")
        .into_iter()
        .next()
        .expect("a vault character can see somewhere");
    h.with(|w| w.enter("m9", "Maker-09", place).unwrap());

    let out = perform(&h, "m1", &act("scan", json!({ "at": name })));
    assert!(
        did(&out).contains("Maker-09"),
        "a scan of an occupied room did not name who was in it: {out:?}"
    );
}

#[test]
fn sleeping_is_recorded_so_the_world_can_wake_you() {
    let h = vault();
    assert!(perform(&h, "m1", &act("sleep", json!({"until":"dawn"}))).happened());
    assert_eq!(
        h.sim(|s| s.ledger.asleep("m1").map(str::to_string)),
        Some("dawn".into())
    );
    assert!(h.with_sim(|s| s.ledger.wake("m1")));
    assert!(h.sim(|s| s.ledger.asleep("m1").is_none()));
}

#[test]
fn acting_on_somebody_who_is_not_here_is_refused() {
    let h = waste();
    let nobody = act("act", json!({"on":"Nobody","intent":"steady them"}));
    let out = perform(&h, "c1", &nobody);
    assert!(refused(&out).contains("not here"), "{out:?}");

    // And somebody who is here is not.
    let steady = act("act", json!({"on":"Soren","intent":"steady him"}));
    assert!(perform(&h, "c1", &steady).happened());

    // The same act covers what it is aimed at doing — there is no separate tool
    // for a blow, and a world that refused one would be refusing the war.
    let down = act(
        "act",
        json!({"on":"Soren","intent":"put him down before he reaches the ridge"}),
    );
    assert!(perform(&h, "c1", &down).happened());
}

/// **Your own body is a thing you can act on.** Aimed at nobody rather than at
/// yourself: the world refuses a body that addresses itself, and it is right to
/// — closing your own wound is seen by everybody standing there and felt by
/// nobody else, which is what an unaimed showing is.
#[test]
fn a_character_can_act_on_itself_even_with_nobody_to_see_it() {
    let h = waste();
    let mend = act(
        "act",
        json!({"on":"yourself","intent":"get the wound closed before it costs me the arm"}),
    );
    assert!(perform(&h, "c1", &mend).happened());

    // The grammar offers `yourself` in lower case; a model that echoes the
    // capitalisation of a sentence must not be refused for it.
    let shouted = act(
        "act",
        json!({"on":"Yourself","intent":"get my weapon clear"}),
    );
    assert!(perform(&h, "c1", &shouted).happened());
}

// ── the phone ───────────────────────────────────────────────────────────────

/// Put two bodies on the roster with handsets, standing well apart.
fn phones() -> Hosted {
    let h = waste();
    h.with(|w| {
        w.place("c2", Where::new("tower-redoubt", "bridge"))
            .unwrap()
    });
    h.with_sim(|s| {
        seed::issue_handset(s, "c1", "Wren");
        seed::issue_handset(s, "c2", "Soren");
    });
    h
}

/// **A phone is carried, so it can be absent — and then nothing is offered.**
///
/// The gate on the whole surface, and it is the ordinary empty-set rule rather
/// than a mode or a flag: no handset, no threads, no contacts, no acts.
#[test]
fn a_character_with_no_handset_is_offered_no_way_to_message_anybody() {
    let h = waste();
    let acts = offered(&h, "c1");
    for gone in ["message", "reach_out", "invite", "open_group", "sign_off"] {
        assert!(
            !acts.contains(&gone.to_string()),
            "`{gone}` without a handset"
        );
    }

    let h = phones();
    let acts = offered(&h, "c1");
    assert!(acts.contains(&"reach_out".to_string()), "{acts:?}");
    // **And with a handset there is always somewhere to send.** This asserted
    // the opposite — no conversation, no `message` — which was true while the
    // only threads were ones somebody had started. A handset now arrives with
    // the world's open channel already on it (`sim::phone::CHANNEL`), because a
    // character that has to open a conversation before it can reach anybody is
    // the isolated character the channel exists to stop being.
    assert!(
        acts.contains(&"message".to_string()),
        "a handset was issued and there was nowhere to send: {acts:?}"
    );
    let threads = admits(&h, "c1", "message", "to").expect("the channel is a thread");
    assert_eq!(threads, vec![npcd::sim::phone::CHANNEL], "{threads:?}");
}

/// Taking the handset off somebody takes the acts with it.
#[test]
fn losing_the_handset_is_something_the_world_can_do_to_you() {
    let h = phones();
    perform(
        &h,
        "c1",
        &act("reach_out", json!({"to":"Soren","intent":"where are you"})),
    );
    assert!(offered(&h, "c1").contains(&"message".to_string()));

    h.with_sim(|s| {
        s.pack_mut("c1").take(npcd::sim::phone::PHONE, 1);
    });
    let acts = offered(&h, "c1");
    assert!(
        !acts.contains(&"message".to_string()) && !acts.contains(&"reach_out".to_string()),
        "the acts survived the handset: {acts:?}"
    );
}

/// **A message reaches somebody who is not here** — which is the entire point,
/// and the thing `say` cannot do.
#[test]
fn a_message_reaches_somebody_a_building_away() {
    let h = phones();
    assert_ne!(h.place_of("c1"), h.place_of("c2"), "they must be apart");

    let out = perform(
        &h,
        "c1",
        &act(
            "reach_out",
            json!({"to":"Soren","intent":"the ridge is clear"}),
        ),
    );
    assert!(out.happened(), "{out:?}");
    assert_eq!(h.sim(|s| s.messages_waiting("Soren")), 1);
}

/// It waits. A room reaches you because you are standing in it; a phone reaches
/// you because something arrived while you were doing something else.
#[test]
fn a_message_waits_until_it_is_looked_at() {
    let h = phones();
    perform(
        &h,
        "c1",
        &act("reach_out", json!({"to":"Soren","intent":"answer me"})),
    );
    assert_eq!(h.sim(|s| s.messages_waiting("Soren")), 1);
    assert_eq!(
        h.sim(|s| s.messages_waiting("Wren")),
        0,
        "your own message waited for you"
    );

    // Doing something else does not consume it.
    perform(&h, "c2", &act("observe", json!({"target":"the room"})));
    assert_eq!(
        h.sim(|s| s.messages_waiting("Soren")),
        1,
        "a message was lost"
    );

    h.with_sim(|s| {
        s.threads.read("Soren", "Wren").unwrap();
    });
    assert_eq!(h.sim(|s| s.messages_waiting("Soren")), 0);
}

/// Several at once, which a room cannot do.
#[test]
fn a_character_holds_more_than_one_conversation_at_a_time() {
    let h = phones();
    h.with_sim(|s| seed::issue_handset(s, "c3", "Orion Vance"));
    perform(
        &h,
        "c1",
        &act("reach_out", json!({"to":"Soren","intent":"one"})),
    );
    perform(
        &h,
        "c1",
        &act("reach_out", json!({"to":"Orion Vance","intent":"two"})),
    );

    // Two of its own, beside the channel every handset carries.
    let threads = admits(&h, "c1", "message", "to").expect("two conversations");
    assert_eq!(threads.len(), 3, "{threads:?}");
    assert!(threads.contains(&"Soren".to_string()));
    assert!(threads.contains(&"Orion Vance".to_string()));
    assert!(threads.contains(&npcd::sim::phone::CHANNEL.to_string()));
}

/// A direct thread becomes a group by gaining somebody, and one send then
/// reaches all of them.
#[test]
fn a_conversation_becomes_a_group_and_one_message_reaches_everybody_on_it() {
    let h = phones();
    h.with_sim(|s| seed::issue_handset(s, "c3", "Orion Vance"));
    perform(
        &h,
        "c1",
        &act("reach_out", json!({"to":"Soren","intent":"start"})),
    );

    let out = perform(
        &h,
        "c1",
        &act("invite", json!({"to":"Soren","who":"Orion Vance"})),
    );
    assert!(out.happened(), "{out:?}");

    perform(
        &h,
        "c1",
        &act("message", json!({"to":"Soren","intent":"both of you"})),
    );
    assert_eq!(h.sim(|s| s.messages_waiting("Soren")), 2);
    assert_eq!(
        h.sim(|s| s.messages_waiting("Orion Vance")),
        1,
        "the new member should see what was sent after they joined, and not before"
    );
}

/// A named group opened outright, without growing one from a pair.
#[test]
fn a_group_can_be_opened_with_several_people_at_once() {
    let h = phones();
    h.with_sim(|s| seed::issue_handset(s, "c3", "Orion Vance"));
    let out = perform(
        &h,
        "c1",
        &act(
            "open_group",
            json!({
                "called":"the boundary",
                "with":"Soren, Orion Vance",
                "intent":"nobody date anything yet"
            }),
        ),
    );
    assert!(out.happened(), "{out:?}");
    assert_eq!(h.sim(|s| s.messages_waiting("Soren")), 1);
    assert_eq!(h.sim(|s| s.messages_waiting("Orion Vance")), 1);
    assert!(admits(&h, "c1", "message", "to")
        .unwrap()
        .contains(&"the boundary".to_string()));
}

/// Reaching out is for somebody you have no thread with; messaging is for
/// somebody you do. Offering both for one person would be a choice with a
/// wrong answer.
#[test]
fn somebody_you_are_already_talking_to_is_not_offered_for_reaching_out() {
    let h = phones();
    assert!(admits(&h, "c1", "reach_out", "to")
        .unwrap()
        .contains(&"Soren".to_string()));

    perform(
        &h,
        "c1",
        &act("reach_out", json!({"to":"Soren","intent":"hello"})),
    );
    let contacts = admits(&h, "c1", "reach_out", "to").unwrap_or_default();
    assert!(
        !contacts.contains(&"Soren".to_string()),
        "already talking to them and still offered a fresh start: {contacts:?}"
    );
}

/// **A second group of the same name is refused, and the reason is not
/// tidiness.**
///
/// Two threads a character calls the same thing are ambiguous to `message`,
/// `invite` and `sign_off`, which all resolve by that name. And the pair became
/// two identical arms in the turn's grammar, which the stencil refuses outright
/// — so the whole tree failed to compile, the turn free-decoded, and the
/// character was told its output was not a call, permanently, because nothing
/// it could do afterwards removed the duplicate. Observed live at four-second
/// intervals against a group somebody had called "none" twice.
#[test]
fn a_second_group_of_the_same_name_is_refused() {
    let h = phones();
    h.with_sim(|s| seed::issue_handset(s, "c3", "Orion Vance"));
    assert!(perform(
        &h,
        "c1",
        &act(
            "open_group",
            json!({"called":"none","with":"Soren, Orion Vance"})
        ),
    )
    .happened());

    let again = perform(
        &h,
        "c1",
        &act(
            "open_group",
            json!({"called":"none","with":"Soren, Orion Vance"}),
        ),
    );
    assert!(!again.happened(), "a duplicate name was allowed: {again:?}");
    assert!(again.line().unwrap().contains("none"), "{again:?}");

    // One group of its own, beside the standing channel — so two arms, not the
    // three a duplicate would have made.
    h.sim(|s| {
        let mut names = s.threads.names_for("Wren");
        names.sort();
        assert_eq!(names, vec!["none", npcd::sim::phone::CHANNEL], "{names:?}");
    });
}

/// Leaving a group leaves it standing for the others.
#[test]
fn leaving_a_group_does_not_end_it_for_everybody_else() {
    let h = phones();
    h.with_sim(|s| seed::issue_handset(s, "c3", "Orion Vance"));
    perform(
        &h,
        "c1",
        &act(
            "open_group",
            json!({"called":"the boundary","with":"Soren, Orion Vance"}),
        ),
    );
    let out = perform(
        &h,
        "c1",
        &act(
            "sign_off",
            json!({"to":"the boundary","intent":"that I am out of it"}),
        ),
    );
    assert!(out.happened(), "{out:?}");

    // Its own conversations are gone; the standing channel is not something it
    // was able to sign off from in the first place — see `Choices::Leavable`.
    assert_eq!(
        admits(&h, "c1", "message", "to").unwrap_or_default(),
        vec![npcd::sim::phone::CHANNEL]
    );
    assert!(
        admits(&h, "c1", "sign_off", "to")
            .unwrap_or_default()
            .is_empty(),
        "the open channel was offered as something to leave"
    );
    h.sim(|s| {
        let mut names = s.threads.names_for("Soren");
        names.sort();
        assert_eq!(names, vec!["the boundary", npcd::sim::phone::CHANNEL]);
    });
}

/// The room hears nothing of it. That is the difference the world represents.
#[test]
fn what_is_said_on_a_thread_does_not_reach_the_room() {
    let h = phones();
    h.with(|w| w.place("c2", Where::new("the-waste", "ruins")).unwrap());
    // Standing together now, and still messaging rather than speaking.
    perform(
        &h,
        "c1",
        &act("reach_out", json!({"to":"Soren","intent":"quietly"})),
    );
    // It went to the thread, not to the room: it is waiting to be read rather
    // than having been heard.
    assert_eq!(h.sim(|s| s.messages_waiting("Soren")), 1);
}

// ── the whole catalog, held to its own rules ────────────────────────────────

/// A tool that is never offered anywhere in either shipped world is either dead
/// or unreachable, and both are worth failing over.
#[test]
fn every_act_in_the_catalog_is_reachable_somewhere_in_a_shipped_world() {
    let vault = vault();
    let waste = waste();
    let mut seen: Vec<String> = Vec::new();

    // **Every room in both worlds, rather than a list somebody maintains.** A
    // hand-written list of interesting rooms is a list that goes stale the
    // moment a station moves, and the failure it produces — "this act is
    // unreachable" — reads as a catalog bug rather than a test that stopped
    // looking in the right place.
    for (h, body) in [(&vault, "m1"), (&waste, "c1")] {
        let rooms: Vec<(String, String)> = h.read(|w| {
            w.map()
                .areas()
                .flat_map(|a| {
                    a.nodes
                        .iter()
                        .map(|n| (a.id.clone(), n.id.clone()))
                        .collect::<Vec<_>>()
                })
                .collect()
        });
        for (area, node) in rooms {
            // A body can only be put somewhere it can be — a world file that
            // names a room in another building is not this world's to stand in.
            if h.with(|w| w.place(body, Where::new(&area, &node))).is_err() {
                continue;
            }
            for name in offered(h, body) {
                if !seen.contains(&name) {
                    seen.push(name);
                }
            }
        }
    }

    // A promise has to exist before it can be recalled, so `remind` only becomes
    // reachable once one stands. Made here rather than excluded, because the
    // point of the test is that everything is reachable *somehow*.
    stand(&waste, "c2", "the-waste", "ruins");
    stand(&waste, "c1", "the-waste", "ruins");
    perform(
        &waste,
        "c2",
        &act(
            "promise",
            json!({"to":"Wren","what":"the sweep","by":"dusk"}),
        ),
    );
    for name in offered(&waste, "c1") {
        if !seen.contains(&name) {
            seen.push(name);
        }
    }

    // Messaging-only acts are reachable by mode rather than by place, so they
    // are checked against the mode that offers them.
    let messaging: Vec<&str> = tools::for_mode(Mode::InstantMessage)
        .iter()
        .map(|t| t.name)
        .collect();

    for t in tools::CATALOG.iter() {
        let reachable = seen.iter().any(|s| s == t.name) || messaging.contains(&t.name);
        assert!(
            reachable,
            "`{}` is in the catalog and no room in either shipped world offers it",
            t.name
        );
    }
}

/// Standing somewhere with nothing in it must still leave a character able to
/// act. A room that offers nothing is a character that can only wait to be
/// spoken to.
///
/// **Speech is not one of them, and that is the point of the test.** An empty
/// corridor has nobody to speak to, and offering speech there is what produced
/// a cast standing alone narrating the scenery at itself. What is left has to
/// be enough to spend a turn on without it.
#[test]
fn there_is_always_something_to_do_even_in_an_empty_corridor() {
    let h = vault();
    stand(&h, "m1", "vault-command", "ring-north");
    let acts = offered(&h, "m1");
    assert!(!acts.is_empty(), "a corridor offered nothing at all");
    for present in ["move_to", "reflect"] {
        assert!(
            acts.contains(&present.to_string()),
            "a corridor lost `{present}`"
        );
    }
    assert!(
        !acts.contains(&"say".to_string()),
        "an empty corridor offered speech, which reaches nobody"
    );
    assert!(
        !acts.contains(&"observe".to_string()),
        "`observe` is back — looking returned what the percept had already \
         handed over, which is a turn spent to learn nothing"
    );
}

/// **Every station affords something, and every attachment names a real part.**
///
/// The invariant used to live in `npc-map`, which could check it only because
/// it carried this engine's vocabulary. Now that the act names the station, it
/// belongs here — and it checks the direction that can actually go wrong: a
/// `Tool::at` naming a part id nothing in the world places is an act that is
/// declared, documented, dispatched, and forever unreachable.
#[test]
fn every_attachment_names_a_part_the_world_actually_has() {
    let set = npc_map::MapSet::load_dir(MAPS).expect("the shipped maps");
    for t in tools::CATALOG.iter() {
        for part in t.at {
            assert!(
                set.part(part).is_some(),
                "`{}` attaches to `{part}`, which no world places",
                t.name
            );
        }
    }
}

/// A station a body can work at must afford at least one act, or it is a seat
/// with a desk's description.
#[test]
fn every_station_in_the_vault_affords_an_act() {
    let set = npc_map::MapSet::load_dir(MAPS).expect("the shipped maps");
    for level in set.children("creators-vault") {
        for node in &level.nodes {
            for (part, _) in set.parts_of(node, npc_map::part::PartKind::Station) {
                let affords = tools::CATALOG
                    .iter()
                    .any(|t| t.at.contains(&part.id.as_str()));
                assert!(
                    affords,
                    "{}/{} places `{}`, and no act attaches to it",
                    level.id, node.id, part.id
                );
            }
        }
    }
}

/// An act that names a station must say it is `AtPart`, and one that names none
/// must not. The two halves are a single fact about where an act can be done,
/// and holding them apart is how one of them goes stale.
#[test]
fn a_station_act_names_where_it_is() {
    use npcd::engine::tools::Availability;
    for t in tools::CATALOG.iter() {
        match t.availability {
            Availability::AtPart => assert!(
                !t.at.is_empty(),
                "`{}` is reachable only at a station and names none",
                t.name
            ),
            _ => assert!(
                t.at.is_empty(),
                "`{}` names a station but is not gated on standing at one",
                t.name
            ),
        }
    }
}

/// Two worlds, built twice, are the same world — which is what makes every
/// assertion above about *exactly* which values an argument admits meaningful.
#[test]
fn a_world_built_twice_is_the_same_world() {
    let a = seed::for_world("battle-cities", None);
    let b = seed::for_world("battle-cities", None);
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
}
