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
// The phone: a message that actually arrives
// =========================================================================

/// Give a body a handset, so the messaging acts are reachable at all.
fn hand_out_phones(rt: &Arc<Runtime>, who: &[(&str, &str)]) {
    let world = rt.hosted.get(WORLD).expect("hosted");
    world.with_sim(|s| {
        let roster: Vec<String> = who.iter().map(|(_, name)| (*name).to_string()).collect();
        s.set_roster(roster);
        for (body, name) in who {
            npcd::sim::seed::issue_handset(s, body, name);
        }
    });
}

/// **A message reaches the other character's mind.**
///
/// The one test that says the messaging system is real. A message is written
/// into a thread, and nothing in the map carries it — the recipient may be on
/// the other side of the world — so it arrives only if the sweep delivers it.
/// Before that pass existed the sender was told "they will see it when they
/// next look" and the recipient was never told there was anything to look at.
#[test]
fn a_message_reaches_the_other_characters_mind() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    // Deliberately in a *different room*: a phone is not a room, and this must
    // not be arriving because they can see each other.
    maker_at(&rt, 2, "m2", "vault-chronicle", "early-range");
    hand_out_phones(&rt, &[("m1", "Maker-01"), ("m2", "Maker-02")]);

    let opened = rt.record_act(
        1,
        &act("reach_out", json!({"to": "Maker-02", "intent": "that I need a word"})),
    );
    assert!(!opened.contains("meant to reach"), "{opened}");
    let sent = rt.record_act(
        1,
        &act("message", json!({"to": "Maker-02", "intent": "that the redoubt burned twice"})),
    );
    assert!(!sent.contains("no conversation"), "{sent}");

    // Nothing has reached the other mind until the world moves.
    moment(&rt);
    think(&rt, 1);

    let read = reads(&rt, 2);
    assert!(
        read.contains("that the redoubt burned twice"),
        "the message never reached the other character: {read}"
    );
    assert!(read.contains("Maker-01"), "the sender was not named: {read}");
}

/// Handed over once, not on every moment afterwards.
#[test]
fn a_message_is_delivered_once_and_not_again_every_moment() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    maker_at(&rt, 2, "m2", "vault-chronicle", "early-range");
    hand_out_phones(&rt, &[("m1", "Maker-01"), ("m2", "Maker-02")]);

    rt.record_act(
        1,
        &act("reach_out", json!({"to": "Maker-02", "intent": "that I need a word"})),
    );
    rt.record_act(1, &act("message", json!({"to": "Maker-02", "intent": "once only"})));
    for _ in 0..4 {
        moment(&rt);
        think(&rt, 1);
    }
    let read = reads(&rt, 2);
    assert_eq!(
        read.matches("once only").count(),
        1,
        "the message was delivered more than once: {read}"
    );
}

/// A character is not handed back what it said itself.
#[test]
fn your_own_message_does_not_come_back_to_you() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    maker_at(&rt, 2, "m2", "vault-chronicle", "early-range");
    hand_out_phones(&rt, &[("m1", "Maker-01"), ("m2", "Maker-02")]);

    rt.record_act(
        1,
        &act("reach_out", json!({"to": "Maker-02", "intent": "that I need a word"})),
    );
    rt.record_act(1, &act("message", json!({"to": "Maker-02", "intent": "mine alone"})));
    moment(&rt);
    think(&rt, 1);

    let mine = reads(&rt, 1);
    assert!(
        !mine.contains("messages you"),
        "a character was told about its own message: {mine}"
    );
}

// =========================================================================
// A person talking to a character
// =========================================================================

/// **A person messaging a character is a party on a thread, not a side
/// channel.** What the console posts lands in the same `sim::phone` the
/// characters use, the character is told by the ordinary sweep, and it answers
/// with the ordinary `message` act — so the reply is the character speaking
/// from inside the world rather than a chat window bolted to the side of it.
#[test]
fn a_person_can_message_a_character_and_it_is_told() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    hand_out_phones(&rt, &[("m1", "Maker-01")]);

    let sent = rt
        .message_npc(1, "Wren S", "that the third era will not reconcile")
        .expect("the character has a body");
    assert_eq!(sent.with, "Maker-01");
    assert!(sent.can_reply, "a character with a handset could not answer");
    assert_eq!(sent.waiting_for_them, 1);

    moment(&rt);
    think(&rt, 1);
    let read = reads(&rt, 1);
    assert!(
        read.contains("that the third era will not reconcile"),
        "the character was never told: {read}"
    );
    assert!(read.contains("Wren S"), "the person was not named: {read}");
}

/// And the character can answer, on the same thread, which the person then
/// reads back.
#[test]
fn a_character_answers_a_person_on_the_same_thread() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    hand_out_phones(&rt, &[("m1", "Maker-01")]);
    rt.message_npc(1, "Wren S", "that the third era will not reconcile");
    moment(&rt);

    // The character answers with the ordinary act, naming the thread the way it
    // was told it — which is the person's own name.
    let replied = rt.record_act(
        1,
        &act("message", json!({"to": "Wren S", "intent": "that I dated it against both neighbours"})),
    );
    assert!(!replied.contains("no conversation"), "{replied}");

    let (with, said) = rt.messages_with(1, "Wren S").expect("a body");
    assert_eq!(with, "Maker-01");
    assert_eq!(said.len(), 2, "{said:?}");
    assert_eq!(said[0].0, "Wren S");
    assert_eq!(said[1].0, "Maker-01");
    assert!(said[1].1.contains("dated it against both neighbours"));
}

/// A person is written onto the roster, or the character is offered nobody to
/// answer — the messaging arguments are bound to who a handset can reach.
#[test]
fn messaging_a_character_puts_the_person_within_its_reach() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    hand_out_phones(&rt, &[("m1", "Maker-01")]);
    rt.message_npc(1, "Wren S", "anything");

    let within = rt.within(1);
    assert!(
        within.threads.iter().any(|t| t == "Wren S"),
        "the person is not a conversation the character can answer: {:?}",
        within.threads
    );
}

/// A character with no handset cannot answer, and the caller is told so rather
/// than left waiting for a reply that cannot come.
#[test]
fn a_character_with_no_handset_says_it_cannot_answer() {
    let rt = daemon();
    maker(&rt, 1, "m1", "band-one");
    let sent = rt.message_npc(1, "Wren S", "anything").expect("has a body");
    assert!(!sent.can_reply);
}

/// A character with no body has nothing to be reached on.
#[test]
fn messaging_a_character_with_no_body_reaches_nothing() {
    let rt = daemon();
    assert!(rt.message_npc(404, "Wren S", "anything").is_none());
    assert!(rt.messages_with(404, "Wren S").is_none());
}

// =========================================================================
// The bench, reached the way a character reaches it
// =========================================================================
//
// Everything else that exercises the editing acts calls `work::perform`
// directly. These do not: they go in at `Runtime::record_act`, which is the
// function the decode loop calls with whatever the model emitted, and they
// assert against **bytes on disk**. What is under test is the wiring — that an
// act named by a model reaches the code that edits documents, and that nothing
// in between is standing in for it.

/// A mind with documents in it, and a daemon pointed at that mind.
///
/// `Mind::new(Some(dir))` is the whole of the wiring: a runtime given a mind
/// points its worlds at it, so a world hosted by any route can edit documents.
fn daemon_with_a_mind(name: &str) -> (Arc<Runtime>, std::path::PathBuf) {
    let mind = std::env::temp_dir().join(format!("npcd-vault-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&mind);
    std::fs::create_dir_all(mind.join("layers/eras")).unwrap();
    std::fs::create_dir_all(mind.join("moods")).unwrap();
    std::fs::write(
        mind.join("layers/eras/third.md"),
        "# the third era\n\nIt burned in the spring.\n",
    )
    .unwrap();
    std::fs::write(
        mind.join("moods/undone.yaml"),
        "id: undone\ncategory: mood\n# why it reads this way\ndescription: As it stands.\n",
    )
    .unwrap();

    let rt = Runtime::new(Mind::new(Some(mind.clone())), &std::env::temp_dir());
    rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
    rt.hold_world(WORLD, true);
    (rt, mind)
}

/// Put a body at a named station rather than in the casting hall.
fn maker_at(rt: &Arc<Runtime>, npc_id: u64, body: &str, area: &str, node: &str) {
    let world = rt.hosted.get(WORLD).expect("hosted");
    world.with(|w| {
        w.enter(body, format!("Maker-{npc_id:02}"), Where::new(area, node))
            .expect("a real room")
    });
    rt.scheduler.wake(npc_id, 0, 0);
    rt.embody(npc_id, WORLD, body, 0).expect("bound");
}

/// **The whole chain, from the function the decode loop calls to the disk.**
///
/// If any part of this were a stub the assertion at the end would still pass
/// against an in-memory fixture — so the assertion is deliberately the file,
/// read back with `std::fs`.
#[test]
fn an_act_from_the_decode_loop_reaches_a_document_on_disk() {
    let (rt, mind) = daemon_with_a_mind("chronicle");
    maker_at(&rt, 1, "m1", "vault-chronicle", "early-range");

    // The era the mind holds is a thing the world knows about, because hosting
    // indexed it.
    let opened = rt.record_act(1, &act("bench_branch", json!({"what": "the third era"})));
    assert!(!opened.contains("nothing called"), "the era was not indexed: {opened}");

    rt.record_act(
        1,
        &act(
            "chronicle_add_entry",
            json!({"to": "the third era", "what": "The redoubt fell before the thaw."}),
        ),
    );
    assert_eq!(
        std::fs::read_to_string(mind.join("layers/eras/third.md")).unwrap(),
        "# the third era\n\nIt burned in the spring.\n",
        "the disk moved before the commit"
    );

    rt.record_act(1, &act("bench_commit", json!({"why": "dated the fall"})));
    let after = std::fs::read_to_string(mind.join("layers/eras/third.md")).unwrap();
    assert!(after.contains("The redoubt fell before the thaw."), "{after}");
    assert!(after.contains("It burned in the spring."), "the entry replaced the document");
}

/// The file acts, through the same door, including the read that a model would
/// actually see.
#[test]
fn the_file_acts_reach_the_mind_through_the_runtime() {
    let (rt, mind) = daemon_with_a_mind("files");
    maker_at(&rt, 1, "m1", "vault-chronicle", "early-range");

    let read = rt.record_act(1, &act("file_read", json!({"path": "layers/eras/third.md"})));
    assert!(read.contains("1  # the third era"), "not the real numbered read: {read}");

    rt.record_act(
        1,
        &act("file_edit", json!({
            "path": "layers/eras/third.md",
            "old_str": "in the spring",
            "new_str": "in the autumn"
        })),
    );
    rt.record_act(1, &act("bench_commit", json!({"why": "against its neighbours"})));
    assert!(std::fs::read_to_string(mind.join("layers/eras/third.md"))
        .unwrap()
        .contains("in the autumn"));
}

/// The craft library, edited at the station that carries it, with the comments
/// still there afterwards — the splice is in the live path, not just the unit
/// test.
#[test]
fn a_mood_edited_at_a_story_desk_keeps_its_comments() {
    let (rt, mind) = daemon_with_a_mind("mood");
    maker_at(&rt, 1, "m1", "vault-story", "first-room");

    rt.record_act(
        1,
        &act("library_write", json!({
            "kind": "mood", "id": "undone",
            "field": "description", "text": "The whole interior rearranged."
        })),
    );
    rt.record_act(1, &act("bench_commit", json!({"why": "it read as two things"})));

    let on_disk = std::fs::read_to_string(mind.join("moods/undone.yaml")).unwrap();
    assert!(on_disk.contains("The whole interior rearranged."), "{on_disk}");
    assert!(on_disk.contains("# why it reads this way"), "the splice was bypassed: {on_disk}");
}

/// **The guard is live too.** A path out of the mind is refused by the act, not
/// by something a test set up.
#[test]
fn a_path_out_of_the_mind_is_refused_through_the_runtime() {
    let (rt, mind) = daemon_with_a_mind("escape");
    maker_at(&rt, 1, "m1", "vault-chronicle", "early-range");

    for bad in ["../stolen.md", "projection.yaml", "layers/eras/../../stolen.md"] {
        rt.record_act(1, &act("file_write", json!({"path": bad, "content": "owned"})));
    }
    rt.record_act(1, &act("bench_commit", json!({"why": "…"})));
    assert!(!mind.parent().unwrap().join("stolen.md").exists());
    assert!(!mind.join("projection.yaml").exists(), "the daemon's own schema was written");
}

/// A world hosted by a daemon with no mind has nothing to edit, and says so
/// rather than inventing somewhere.
#[test]
fn a_daemon_with_no_mind_refuses_the_editing_acts() {
    let rt = daemon();
    maker_at(&rt, 1, "m1", "vault-chronicle", "early-range");
    let out = rt.record_act(1, &act("file_read", json!({"path": "layers/eras/third.md"})));
    assert!(out.contains("no documents"), "{out}");
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
