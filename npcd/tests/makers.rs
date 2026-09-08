//! Makers in the vault, from the world to what the model would read.
//!
//! Every piece of this path is unit-tested where it lives. **Nothing tests them
//! joined**, because the module that joins them in production does not exist
//! yet — so this hand-wires the chain and asserts the thing that actually
//! matters: what one Maker ends up reading, and when it is woken to read it.
//!
//! ```text
//!   World  ──▶ Attention ──▶ Delta ──▶ digest ──▶ Scheduler ──▶ Window
//!   (npc-map: who is where, what just happened)   (npcd: whose turn, what it reads)
//! ```
//!
//! The one thing standing in for something real is the decode: `Scheduler::tick`
//! takes the acts as a closure, so a test drives the whole loop without a GPU.
//! Everything else — the map, the scope rules, the salience ladder, the
//! supersession band, the heartbeat — is the production article.
//!
//! This doubles as the specification for [`npcd::engine::environment`]. Whatever
//! that module ends up looking like, these are the assertions it has to keep
//! true.
//!
//! It reaches the engine the way any other consumer would — through the library,
//! with no test-only seam — which is the rule the harness design sets: needing a
//! back door is evidence of a missing capability, not a reason to add one.

use npc_map::delta::{Attention, Delta};
use npc_map::world::{Where, World};
use npc_map::MapSet;

use npcd::engine::environment::carried;
use npcd::engine::event::{Addressed, EventKind, Salience};
use npcd::engine::perceived::{situation, Perceived};
use npcd::engine::tick::Scheduler;

/// The shipped vault, which is the Makers' whole world.
fn vault() -> World {
    World::new(
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped vault must load"),
    )
}

fn casting(node: &str) -> Where {
    Where::new("vault-casting", node)
}

/// A Maker: a body in the world and a mind in the scheduler, under one id.
struct Maker {
    id: &'static str,
    npc_id: u64,
}

fn crew(world: &mut World, sched: &Scheduler, who: &[(&'static str, &str, &str)]) -> Vec<Maker> {
    who.iter()
        .enumerate()
        .map(|(i, (id, name, room))| {
            let npc_id = 1000 + i as u64;
            world.enter(*id, *name, casting(room)).expect("a real room");
            sched.wake(npc_id, 0, 0);
            Maker { id, npc_id }
        })
        .collect()
}

/// **The wiring under test.** One body's delta, delivered to its mind.
///
/// The composition is [`npcd::engine::environment::carried`] — the production
/// article, not a copy of it — driven here against a bare `World` rather than a
/// hosted one, so these assertions hold for a caller that owns its own world
/// and not only for the one shape the daemon happens to host.
fn deliver(world: &mut World, attention: &mut Attention, sched: &Scheduler, m: &Maker) -> Delta {
    let delta = attention.take(world, m.id);
    for Perceived { kind, salience } in carried(world, &delta) {
        sched.deliver(m.npc_id, world.now(), salience, kind);
    }
    delta
}

/// Run every mind that is due, recording what each one read.
fn run_due(sched: &Scheduler, at_ms: u64) -> Vec<(u64, Vec<String>)> {
    sched
        .due_now(at_ms)
        .into_iter()
        .filter_map(|npc_id| {
            sched
                .tick(npc_id, at_ms, at_ms, |_events, _window| Vec::new())
                .map(|r| (npc_id, r.perceived))
        })
        .collect()
}

/// Everything one mind read across a run, flattened.
fn read(runs: &[(u64, Vec<String>)], npc_id: u64) -> Vec<String> {
    runs.iter()
        .filter(|(id, _)| *id == npc_id)
        .flat_map(|(_, lines)| lines.clone())
        .collect()
}

// =========================================================================
// The slice: one speaks, the other is woken by it
// =========================================================================

#[test]
fn a_maker_spoken_to_is_woken_and_reads_that_it_was_spoken_to() {
    // The whole point of the chain, end to end. Being addressed crosses the
    // preempt threshold in `npc-map`'s ladder, converts to a salience above
    // `PREEMPT_AT`, and the scheduler brings the tick forward — three separate
    // decisions in three modules that have to agree.
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(
        &mut w,
        &s,
        &[
            ("m1", "Maker-01", "green-room"),
            ("m2", "Maker-02", "green-room"),
        ],
    );
    let (one, two) = (&crew[0], &crew[1]);

    // Ground both, and let the arrival traffic settle.
    for m in [one, two] {
        deliver(&mut w, &mut a, &s, m);
    }
    run_due(&s, 0);

    w.tell("m1", "m2", "the redoubt burned twice").unwrap();
    let delta = deliver(&mut w, &mut a, &s, two);
    assert!(delta.preempts(), "being addressed did not preempt");

    // Woken at once rather than at its own heartbeat, which is minutes away.
    // Asked once: `due_now` pops the queue, so checking it and then running it
    // would consume the wake before the tick could use it.
    let runs = run_due(&s, 1);
    assert!(
        runs.iter().any(|(id, _)| *id == two.npc_id),
        "the addressed Maker was not woken"
    );
    let heard = read(&runs, two.npc_id).join("\n");
    assert!(heard.contains("Maker-01 says to you"), "{heard}");
    assert!(heard.contains("the redoubt burned twice"), "{heard}");
}

#[test]
fn a_maker_overhearing_the_same_words_is_not_woken_by_them() {
    // The other half, and the one that would be invisible if it broke: a
    // third Maker in the room hears it, reads it as somebody else's business,
    // and is not interrupted.
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(
        &mut w,
        &s,
        &[
            ("m1", "Maker-01", "green-room"),
            ("m2", "Maker-02", "green-room"),
            ("m3", "Maker-03", "green-room"),
        ],
    );
    for m in &crew {
        deliver(&mut w, &mut a, &s, m);
    }
    run_due(&s, 0);

    w.tell("m1", "m2", "the redoubt burned twice").unwrap();
    let bystander = deliver(&mut w, &mut a, &s, &crew[2]);

    assert!(!bystander.preempts(), "overhearing interrupted a Maker");
    // One call, because it pops: what is due at this instant, and the bystander
    // is not among it.
    assert!(
        !s.due_now(1).contains(&crew[2].npc_id),
        "the bystander was woken"
    );

    // It still *heard* it — nothing was dropped, only left to wait.
    let runs = run_due(&s, 200_000); // past the idle heartbeat
    let heard = read(&runs, crew[2].npc_id).join("\n");
    assert!(heard.contains("Maker-01 says to Maker-02"), "{heard}");
}

#[test]
fn nothing_said_in_a_room_reaches_a_maker_outside_it() {
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(
        &mut w,
        &s,
        &[
            ("m1", "Maker-01", "green-room"),
            ("out", "Maker-09", "ring-south"),
        ],
    );
    for m in &crew {
        deliver(&mut w, &mut a, &s, m);
    }
    run_due(&s, 0);

    w.say("m1", "the redoubt burned twice").unwrap();
    deliver(&mut w, &mut a, &s, &crew[1]);

    let runs = run_due(&s, 200_000);
    let heard = read(&runs, crew[1].npc_id).join("\n");
    assert!(
        !heard.contains("redoubt"),
        "it carried through a wall: {heard}"
    );
}

// =========================================================================
// The situation, in the window a mind actually reads from
// =========================================================================

#[test]
fn a_maker_is_only_ever_standing_in_one_place() {
    // The join that fails silently: two situations in the window and the
    // character carries a room it has left, with the older one reading as
    // current once the newer falls past the cap.
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(&mut w, &s, &[("m1", "Maker-01", "band-one")]);
    let m = &crew[0];

    for (round, room) in ["green-room", "relations", "watch", "band-two"]
        .iter()
        .enumerate()
    {
        w.set_off("m1", casting(room)).unwrap();
        w.tick();
        deliver(&mut w, &mut a, &s, m);
        run_due(&s, (round as u64 + 1) * 300_000);
    }

    let standing: Vec<String> = s
        .window_of(m.npc_id, |win| {
            win.turns()
                .map(|t| t.text.clone())
                .filter(|t| t.starts_with("You are"))
                .collect::<Vec<_>>()
        })
        .expect("the Maker is in the scheduler");
    assert_eq!(
        standing.len(),
        1,
        "standing in {} places: {standing:?}",
        standing.len()
    );
    assert!(standing[0].contains("band two"), "{}", standing[0]);
}

#[test]
fn a_maker_that_has_not_moved_is_not_told_where_it_is_again() {
    // What pays for a mind that never blocks: an unchanged situation renders
    // byte-identically, so there is nothing to deliver at all.
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(&mut w, &s, &[("m1", "Maker-01", "band-one")]);
    let m = &crew[0];

    assert!(deliver(&mut w, &mut a, &s, m).percept.is_some());
    run_due(&s, 0);
    for _ in 0..20 {
        assert!(
            deliver(&mut w, &mut a, &s, m).is_empty(),
            "an idle Maker cost something"
        );
    }
}

// =========================================================================
// A crowd, which is what the vault actually is
// =========================================================================

#[test]
fn sixteen_makers_only_wake_the_ones_something_happened_to() {
    let mut w = vault();
    let s = Scheduler::new(256);
    let mut a = Attention::new();

    let rooms = ["band-one", "band-two", "green-room", "watch"];
    let names: Vec<(String, String, &str)> = (1..=16)
        .map(|i| {
            (
                format!("m{i:02}"),
                format!("Maker-{i:02}"),
                rooms[i % rooms.len()],
            )
        })
        .collect();
    let crew: Vec<Maker> = names
        .iter()
        .enumerate()
        .map(|(i, (id, name, room))| {
            let npc_id = 2000 + i as u64;
            w.enter(id.clone(), name.clone(), casting(room)).unwrap();
            s.wake(npc_id, 0, 0);
            Maker {
                id: Box::leak(id.clone().into_boxed_str()),
                npc_id,
            }
        })
        .collect();

    for m in &crew {
        deliver(&mut w, &mut a, &s, m);
    }
    run_due(&s, 0);

    // One Maker speaks, in one room, to nobody in particular.
    let speaker = &crew[0];
    w.say(speaker.id, "somebody should look at the redoubt")
        .unwrap();
    let woken: Vec<u64> = crew
        .iter()
        .filter(|m| !deliver(&mut w, &mut a, &s, m).is_empty())
        .map(|m| m.npc_id)
        .collect();

    // Only the room hears it: the four in band one, less the speaker itself.
    assert!(
        woken.len() < crew.len() / 2,
        "one utterance reached {} of {} Makers",
        woken.len(),
        crew.len()
    );
    assert!(!woken.contains(&speaker.npc_id), "it heard itself");
}

#[test]
fn a_maker_crossing_the_building_reads_one_line_per_stop() {
    // Movement is condensed, so a journey is not narrated doorway by doorway —
    // and the three stops are the three a person would name.
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(&mut w, &s, &[("m1", "Maker-01", "band-one")]);
    let m = &crew[0];
    deliver(&mut w, &mut a, &s, m);
    run_due(&s, 0);

    let stops = w
        .set_off("m1", Where::new("vault-command", "command-room"))
        .unwrap();
    assert_eq!(stops, 3);

    let mut situations = 0;
    for stop in 0..stops {
        w.tick();
        let d = deliver(&mut w, &mut a, &s, m);
        run_due(&s, (stop as u64 + 1) * 300_000);
        if d.percept.is_some() {
            situations += 1;
        }
    }
    assert_eq!(situations, stops, "a stop passed without re-grounding");

    // It was told it arrived, once, and told it in its own voice — the outcome
    // of something it set in motion three stops ago, which is the one thing a
    // body is told about its own doings.
    let told = s
        .window_of(m.npc_id, |win| {
            win.turns()
                .map(|t| t.text.clone())
                .filter(|t| t.contains("got to"))
                .collect::<Vec<_>>()
        })
        .expect("in the scheduler");
    assert_eq!(told.len(), 1, "{told:?}");
    assert!(
        told[0].starts_with("You got to the command room"),
        "{told:?}"
    );
}

// =========================================================================
// What a mind reads is prose, all the way down
// =========================================================================

#[test]
fn nothing_a_maker_reads_carries_the_shape_of_the_machinery() {
    // The path crosses two crates, three renderers and a serialisation
    // boundary. Any one of them leaking an id, a field name or a debug format
    // would put it in front of the model.
    let mut w = vault();
    let s = Scheduler::new(256);
    let mut a = Attention::new();
    let crew = crew(
        &mut w,
        &s,
        &[
            ("m1", "Maker-01", "band-one"),
            ("m2", "Maker-02", "band-one"),
            ("m3", "Maker-03", "green-room"),
        ],
    );

    for m in &crew {
        deliver(&mut w, &mut a, &s, m);
    }
    w.take("m1", Some("r-okonkwo")).unwrap();
    w.tell("m1", "m2", "you have the redoubt").unwrap();
    w.say("m2", "since yesterday").unwrap();
    w.set_off("m2", casting("green-room")).unwrap();
    w.tick();
    for m in &crew {
        deliver(&mut w, &mut a, &s, m);
    }

    let runs = run_due(&s, 400_000);
    let everything: Vec<String> = runs.iter().flat_map(|(_, l)| l.clone()).collect();
    assert!(!everything.is_empty(), "nobody read anything");

    for line in &everything {
        for leak in [
            "vault-casting",
            "band-one",
            "Happening",
            "Witnessed",
            "EventKind",
            "Some(",
            "None",
            "npc_id",
            "salience",
            "{",
            "}",
        ] {
            assert!(!line.contains(leak), "leaked {leak:?}: {line}");
        }
    }

    // Ids never reach the model; names always do.
    let joined = everything.join("\n");
    assert!(
        joined.contains("Maker-01") || joined.contains("Maker-02"),
        "{joined}"
    );
}

#[test]
fn a_situation_and_the_news_arrive_as_separate_things() {
    // They are different kinds of thing — a point in time and a change over
    // time — and only one of them supersedes. Folded into one event, the news
    // would be retired along with the situation it arrived beside.
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(
        &mut w,
        &s,
        &[
            ("m1", "Maker-01", "band-one"),
            ("m2", "Maker-02", "band-one"),
        ],
    );
    for m in &crew {
        deliver(&mut w, &mut a, &s, m);
    }
    run_due(&s, 0);

    // Something happens *and* the room changes, in one moment.
    w.say("m2", "the redoubt burned twice").unwrap();
    w.set_off("m2", casting("green-room")).unwrap();
    w.tick();

    let d = deliver(&mut w, &mut a, &s, &crew[0]);
    let carried = carried(&w, &d);
    let situations = carried
        .iter()
        .filter(|p| matches!(p.kind, EventKind::Situation { .. }))
        .count();
    let news = carried.len() - situations;
    assert_eq!(situations, 1, "{carried:?} situations");
    assert!(news >= 1, "the utterance was folded into the situation");

    // And only the situation claims a band.
    for p in &carried {
        match p.kind {
            EventKind::Situation { .. } => assert!(p.kind.replaces().is_some()),
            _ => assert!(p.kind.replaces().is_none(), "{:?} superseded", p.kind),
        }
    }
}

#[test]
fn the_addressee_survives_every_hop_between_the_world_and_the_model() {
    // Four representations of one fact — a hold on the world's side, an
    // `Addressed` on the engine's, a line of prose, and a window turn. It is
    // the sort of thing that survives three hops and is lost on the fourth.
    let mut w = vault();
    let s = Scheduler::new(64);
    let mut a = Attention::new();
    let crew = crew(
        &mut w,
        &s,
        &[
            ("m1", "Maker-01", "green-room"),
            ("m2", "Maker-02", "green-room"),
            ("m3", "Maker-03", "green-room"),
        ],
    );
    for m in &crew {
        deliver(&mut w, &mut a, &s, m);
    }
    run_due(&s, 0);

    w.tell("m1", "m2", "get out of here").unwrap();

    let told = carried(&w, &a.peek(&w, "m2"));
    let heard = carried(&w, &a.peek(&w, "m3"));
    assert!(told.iter().any(|p| matches!(
        &p.kind,
        EventKind::Speech {
            to: Addressed::You,
            ..
        }
    )));
    assert!(heard.iter().any(|p| matches!(
        &p.kind,
        EventKind::Speech { to: Addressed::Other { who }, .. } if who == "Maker-02"
    )));

    for m in &crew[1..] {
        deliver(&mut w, &mut a, &s, m);
    }
    let runs = run_due(&s, 400_000);
    assert!(read(&runs, crew[1].npc_id)
        .join("\n")
        .contains("says to you"));
    assert!(read(&runs, crew[2].npc_id)
        .join("\n")
        .contains("says to Maker-02"));
}

#[test]
fn a_situation_is_delivered_quietly_and_never_interrupts() {
    // A changed situation says the world moved, not that anything wants
    // answering. Weighing it to preempt would make every passing body an
    // interruption, and sixteen Makers would interrupt each other for ever.
    let s = situation("You are in the green room. Maker-04 is here.");
    assert_eq!(s.salience, Salience::IDLE);
    assert!(!s.salience.preempts());
}
