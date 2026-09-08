//! A daemon with a vault in it, driven through what a consumer can reach.
//!
//! [`makers`](makers.rs) proves the perception path with the pieces wired by
//! hand. This proves the **assembly**: a `Runtime` that hosts a world, gives
//! characters bodies, moves the world on its own, and puts acts into it — all
//! through the library's public surface, with no test-only seam.
//!
//! The one thing standing in for something real is the decode. `Runtime` has no
//! model in these tests, so acts are supplied rather than generated — which is
//! the right seam, because what is under test is everything either side of the
//! model and none of it is the model.
//!
//! The world is held still throughout. These assert what hosting, binding and
//! acting *do*; a metronome beating underneath would make each of them a race,
//! and the metronome has its own tests.

use std::path::Path;
use std::sync::Arc;

use serde_json::json;

use npc_map::world::Where;

use npcd::engine::act::Act;
use npcd::engine::body::Outcome;
use npcd::engine::environment;
use npcd::engine::runtime::Runtime;
use npcd::engine::tick::Pace;
use npcd::mind::Mind;

const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");

/// The id the vault is authored under.
///
/// A world is hosted under the id its document has, because that is what every
/// character's `world_id` names — the same key the corpus is already partitioned
/// by (`worlds/<id>.yaml`, `layers/world/<id>/`, and now `map/<id>/`).
const WORLD: &str = "creators-vault";

/// A daemon hosting the vault, held still.
fn daemon() -> Arc<Runtime> {
    let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
    rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
    rt.hold_world(WORLD, true);
    rt
}

fn at(node: &str) -> Where {
    Where::new("vault-casting", node)
}

/// A Maker: a body in the vault and a character in the scheduler.
fn maker(rt: &Arc<Runtime>, npc_id: u64, body: &str, room: &str) {
    let world = rt.hosted.get(WORLD).expect("hosted");
    world.with(|w| {
        w.enter(body, format!("Maker-{npc_id:02}"), at(room))
            .expect("a real room")
    });
    rt.scheduler.wake(npc_id, 0, 0);
    rt.embody(npc_id, WORLD, body, 0).expect("bound");
}

/// One moment of world time, the way the metronome would.
fn moment(rt: &Arc<Runtime>) {
    let world = rt.hosted.get(WORLD).expect("hosted");
    environment::advance(&world, &rt.bodies, &rt.scheduler);
}

/// Let every character that is due take its turn.
fn think(rt: &Arc<Runtime>, at_ms: u64) {
    for id in rt.scheduler.due_now(at_ms) {
        rt.scheduler.tick(id, at_ms, at_ms, |_, _| Vec::new());
    }
}

/// Everything one character is holding, as one string.
fn reads(rt: &Arc<Runtime>, npc_id: u64) -> String {
    rt.scheduler
        .window_of(npc_id, |w| {
            w.turns().map(|t| t.text.clone()).collect::<Vec<_>>()
        })
        .unwrap_or_default()
        .join("\n")
}

fn act(tool: &'static str, args: serde_json::Value) -> Act {
    Act {
        tool,
        args: args.as_object().expect("an object").clone(),
    }
}

fn standing(rt: &Arc<Runtime>, body: &str) -> Where {
    rt.hosted
        .get(WORLD)
        .expect("hosted")
        .read(|w| w.actor(body).expect("in the world").at.clone())
}

// =========================================================================
// The slice: one speaks, the other walks over
// =========================================================================

#[test]
fn one_maker_calls_another_over_and_the_other_comes() {
    // The whole assembly in one scene, and every step of it crosses a seam
    // that did not exist before: an act reaching a world, a world scoping who
    // perceives it, a salience deciding whose turn comes next, and a journey
    // the world advances rather than the act.
    let rt = daemon();
    maker(&rt, 1, "m1", "green-room");
    maker(&rt, 2, "m2", "band-one");
    moment(&rt);
    think(&rt, 1);

    // Out of earshot: the green room and band one are different rooms.
    rt.act_on_world(1, &act("speak", json!({"intent": "come and look at this"})));
    moment(&rt);
    think(&rt, 10_000);
    assert!(
        !reads(&rt, 2).contains("come and look"),
        "it carried through a wall"
    );

    // So Maker-01 goes to find them.
    let set_off = rt
        .act_on_world(1, &act("move_to", json!({"destination": "band one"})))
        .line()
        .expect("a body act")
        .to_string();
    assert!(set_off.contains("one stop"), "{set_off}");

    moment(&rt);
    assert_eq!(standing(&rt, "m1"), at("band-one"));
    think(&rt, 20_000);
    assert!(
        reads(&rt, 1).contains("You got to band one"),
        "{}",
        reads(&rt, 1)
    );
    assert!(
        reads(&rt, 2).contains("Maker-01"),
        "nobody noticed it arrive"
    );

    // Now it can be heard, and being addressed brings the other's turn forward.
    rt.act_on_world(
        1,
        &act(
            "speak",
            json!({"intent": "the redoubt burned twice", "to": "Maker-02"}),
        ),
    );
    moment(&rt);
    think(&rt, 20_001);
    let heard = reads(&rt, 2);
    assert!(heard.contains("says to you"), "{heard}");
    assert!(heard.contains("redoubt"), "{heard}");
}

#[test]
fn a_maker_reads_where_it_is_and_what_it_can_do_there() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    think(&rt, 1);

    let read = reads(&rt, 1);
    assert!(read.contains("You are working in band one") || read.contains("You are in band one"));
    assert!(read.contains("Within reach"), "{read}");
    assert!(read.contains("terminal"), "{read}");
}

#[test]
fn what_a_maker_can_do_changes_as_it_walks() {
    // The payoff of hanging tools on the room rather than on the character:
    // walking out takes them, and the change arrives in the same block as the
    // change of place, because they are one fact.
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    think(&rt, 1);
    assert!(reads(&rt, 1).contains("Within reach"));

    rt.act_on_world(1, &act("move_to", json!({"destination": "the north run"})));
    moment(&rt);
    think(&rt, 10_000);

    let situations: Vec<String> = rt
        .scheduler
        .window_of(1, |w| {
            w.turns()
                .map(|t| t.text.clone())
                .filter(|t| t.starts_with("You are"))
                .collect::<Vec<_>>()
        })
        .unwrap();
    // One situation, and it is the corridor's — which offers nothing.
    assert_eq!(situations.len(), 1, "{situations:?}");
    assert!(situations[0].contains("north run"), "{situations:?}");
    assert!(!situations[0].contains("Within reach"), "{situations:?}");
}

#[test]
fn a_journey_across_the_building_is_three_moments_and_one_decision() {
    // The character names a destination once. Getting there is the world's
    // business, and it is told when it arrives — not before.
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    think(&rt, 1);

    let out = rt
        .act_on_world(
            1,
            &act("move_to", json!({"destination": "the command room"})),
        )
        .line()
        .expect("a body act")
        .to_string();
    assert!(out.contains("3 stops"), "{out}");

    for step in 1..=2 {
        moment(&rt);
        think(&rt, step * 10_000);
        assert!(
            !reads(&rt, 1).contains("You got to"),
            "told it arrived after {step} of 3"
        );
    }
    moment(&rt);
    think(&rt, 30_000);
    assert_eq!(
        standing(&rt, "m1"),
        Where::new("vault-command", "command-room")
    );
    assert!(reads(&rt, 1).contains("You got to the command room"));
}

// =========================================================================
// A character created into a world, and the loop it runs
// =========================================================================

/// Everything one character has been handed, as the model would read it.
///
/// The whole window rather than the last turn: a character reasons over what
/// it is holding, and a test that read only the newest line would pass for an
/// engine that dropped everything before it.
fn context(rt: &Arc<Runtime>, npc_id: u64) -> String {
    reads(rt, npc_id)
}

#[test]
fn a_character_created_into_a_world_starts_standing_somewhere_real() {
    let rt = daemon();
    rt.scheduler.wake(1, 0, 0);
    assert!(rt
        .embody_in_world(1, WORLD, None, "Maker-01", None, 0)
        .unwrap());

    think(&rt, 1);
    let read = context(&rt, 1);
    assert!(read.starts_with("You are"), "{read}");
    assert!(read.contains("command room"), "{read}");
}

#[test]
fn a_character_with_nothing_asked_of_it_is_pointed_at_the_work() {
    // The idle turn is not an empty one. This is what keeps a long run moving
    // when no mission has been given, and it is what a Maker does all day.
    //
    // And it points at the *work*: an instruction to go and see an unvisited
    // room is one characters followed exactly, touring the building for hours
    // and arriving with nothing to say, because a room is not a subject.
    let rt = daemon();
    rt.scheduler.wake(1, 0, 0);
    rt.embody_in_world(1, WORLD, None, "Maker-01", None, 0)
        .unwrap();

    let nudge = rt.nudge_for(1).expect("something to be getting on with");
    rt.scheduler.deliver(
        1,
        0,
        npcd::engine::event::Salience::IDLE,
        npcd::engine::event::EventKind::Nudge { text: nudge },
    );
    think(&rt, 1);

    let read = context(&rt, 1);
    // Alone, so the task is to take up something in front of it and form a
    // view worth putting to somebody.
    assert!(read.contains("within reach"), "{read}");
    assert!(
        read.contains("You are"),
        "it was told what to do and not where it is"
    );
}

#[test]
fn the_standing_instruction_is_restated_and_never_accumulates() {
    // Restated because a long run loses its grip on the objective before it
    // loses anything else; superseding because two of them is a character
    // working to a task it has been taken off.
    let rt = daemon();
    rt.scheduler.wake(1, 0, 0);
    rt.embody_in_world(1, WORLD, None, "Maker-01", None, 0)
        .unwrap();

    for round in 0..5 {
        let nudge = rt.nudge_for(1).unwrap();
        rt.scheduler.deliver(
            1,
            0,
            npcd::engine::event::Salience::IDLE,
            npcd::engine::event::EventKind::Nudge { text: nudge },
        );
        think(&rt, round * 10_000 + 1);
    }

    let held = rt
        .scheduler
        .window_of(1, |w| {
            w.turns()
                .filter(|t| t.replaces.as_deref() == Some("nudge"))
                .count()
        })
        .unwrap();
    assert_eq!(held, 1, "the character is holding {held} tasks");
}

#[test]
fn a_maker_told_to_explore_can_reach_everywhere_it_might_go() {
    // The instruction says to go somewhere it has not been. Every room in the
    // building has to actually be namable and reachable from where a character
    // starts, or the instruction is one it cannot follow.
    let rt = daemon();
    rt.scheduler.wake(1, 0, 0);
    rt.embody_in_world(1, WORLD, None, "Maker-01", None, 0)
        .unwrap();

    let world = rt.hosted.get(WORLD).unwrap();
    let rooms: Vec<(String, String)> = world.read(|w| {
        w.map()
            .children("creators-vault")
            .iter()
            .flat_map(|a| {
                a.nodes
                    .iter()
                    .map(|n| (a.id.clone(), n.name.clone()))
                    .collect::<Vec<_>>()
            })
            .collect()
    });
    assert!(rooms.len() > 30, "only {} rooms", rooms.len());

    let mut reached = 0;
    for (area, name) in &rooms {
        // Named the way the memory names it, from wherever the body now is.
        let out = rt.act_on_world(1, &act("move_to", json!({ "destination": name })));
        let Some(line) = out.line() else {
            panic!("move_to was not a body act");
        };
        assert!(
            !line.contains("no "),
            "`{name}` on {area} could not be named: {line}"
        );
        // Walk it, so the next one is named from somewhere else.
        for _ in 0..4 {
            moment(&rt);
        }
        reached += 1;
    }
    assert_eq!(reached, rooms.len());
}

#[test]
fn two_makers_exploring_the_same_building_find_each_other() {
    // What the instruction is for. Both are told to look around and talk to
    // people; the building is what makes that possible, and the test is that
    // one of them ends up somewhere it can see or hear the other.
    let rt = daemon();
    for id in [1u64, 2] {
        rt.scheduler.wake(id, 0, 0);
        rt.embody_in_world(id, WORLD, None, &format!("Maker-{id:02}"), None, 0)
            .unwrap();
    }
    moment(&rt);
    think(&rt, 1);

    // They arrived at the same door, so each already knows the other is there.
    assert!(context(&rt, 1).contains("Maker-02"), "{}", context(&rt, 1));

    // One wanders off; the other is told it left.
    rt.act_on_world(1, &act("move_to", json!({"destination": "the anteroom"})));
    moment(&rt);
    think(&rt, 10_000);
    assert!(context(&rt, 2).contains("Maker-01"), "{}", context(&rt, 2));
}

// =========================================================================
// What the assembly refuses
// =========================================================================

#[test]
fn a_maker_cannot_act_on_a_world_it_has_no_body_in() {
    let rt = daemon();
    rt.scheduler.wake(9, 0, 0);
    assert_eq!(
        rt.act_on_world(9, &act("speak", json!({"intent": "anything"}))),
        Outcome::NotOfTheBody
    );
}

#[test]
fn an_act_the_world_refuses_is_read_as_refused() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    let out = rt.act_on_world(
        1,
        &act("move_to", json!({"destination": "the observatory"})),
    );
    let Outcome::Refused(out) = out else {
        panic!("a walk to nowhere is a refusal, not {out:?}");
    };
    assert!(out.contains("nowhere called \"the observatory\""), "{out}");
    assert_eq!(standing(&rt, "m1"), at("band-one"), "it went anyway");
}

#[test]
fn a_character_that_gets_a_body_never_goes_as_quiet_as_one_without() {
    let rt = daemon();
    rt.scheduler.wake(9, 0, 0);
    maker(&rt, 1, "m1", "band-one");

    assert_eq!(rt.scheduler.pace_of(1), Some(Pace::WORKING));
    assert_eq!(rt.scheduler.pace_of(9), Some(Pace::AMBIENT));

    // And giving the body back lets it settle again.
    assert!(rt.disembody(1, 0));
    assert_eq!(rt.scheduler.pace_of(1), Some(Pace::AMBIENT));
    assert_eq!(
        rt.act_on_world(1, &act("speak", json!({"intent": "x"}))),
        Outcome::NotOfTheBody
    );
}

#[test]
fn taking_the_world_away_takes_every_body_with_it() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    maker(&rt, 2, "m2", "green-room");

    assert!(rt.unhost(WORLD));
    assert!(rt.body_of(1).is_none());
    assert!(rt.body_of(2).is_none());
    assert!(rt.moments().is_empty(), "the metronome kept beating");
    assert_eq!(
        rt.act_on_world(1, &act("speak", json!({"intent": "into the void"}))),
        Outcome::NotOfTheBody
    );
}

// =========================================================================
// A crowd
// =========================================================================

#[test]
fn sixteen_makers_share_one_building_and_only_the_room_hears() {
    let rt = daemon();
    let rooms = ["band-one", "band-two", "green-room", "watch"];
    for i in 1..=16u64 {
        maker(&rt, i, &format!("m{i:02}"), rooms[i as usize % 4]);
    }
    moment(&rt);
    think(&rt, 1);

    rt.act_on_world(
        1,
        &act(
            "speak",
            json!({"intent": "somebody should look at the redoubt"}),
        ),
    );
    moment(&rt);
    think(&rt, 100_000);

    let heard: Vec<u64> = (1..=16)
        .filter(|i| reads(&rt, *i).contains("redoubt"))
        .collect();
    assert!(!heard.is_empty(), "nobody heard it");
    assert!(
        heard.len() < 8,
        "one utterance reached {} of 16: {heard:?}",
        heard.len()
    );
    assert!(!heard.contains(&1), "it heard itself");
    // Everyone who heard it was in the room with the speaker.
    let room = standing(&rt, "m01");
    for i in &heard {
        assert_eq!(
            standing(&rt, &format!("m{i:02}")),
            room,
            "m{i:02} was elsewhere"
        );
    }
}

#[test]
fn a_quiet_building_costs_nothing_to_run() {
    // Most moments are quiet, and a quiet moment must deliver nothing at all —
    // this is what makes a cast affordable, and it is invisible when it breaks
    // because everything still works, only more expensively every tick.
    let rt = daemon();
    for i in 1..=8u64 {
        maker(&rt, i, &format!("m{i}"), "green-room");
    }
    moment(&rt);
    think(&rt, 1);

    for _ in 0..20 {
        let world = rt.hosted.get(WORLD).unwrap();
        let m = environment::advance(&world, &rt.bodies, &rt.scheduler);
        assert!(m.is_quiet(), "an idle building cost something: {m:?}");
    }
}

#[test]
fn nothing_a_maker_reads_carries_the_shape_of_the_machinery() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    maker(&rt, 2, "m2", "band-one");
    moment(&rt);

    rt.act_on_world(
        1,
        &act(
            "speak",
            json!({"intent": "you have the redoubt", "to": "Maker-02"}),
        ),
    );
    rt.act_on_world(2, &act("move_to", json!({"destination": "the green room"})));
    moment(&rt);
    think(&rt, 10_000);

    for npc_id in [1, 2] {
        let read = reads(&rt, npc_id);
        assert!(!read.is_empty(), "{npc_id} read nothing at all");
        for leak in [
            "vault-casting",
            "band-one",
            "green-room",
            "Happening",
            "Witnessed",
            "EventKind",
            "Some(",
            "npc_id",
            "{",
            "}",
        ] {
            assert!(!read.contains(leak), "{npc_id} leaked {leak:?}: {read}");
        }
    }
}
