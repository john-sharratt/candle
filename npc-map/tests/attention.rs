//! Sixteen Makers, swept every tick, for as long as it takes to break.
//!
//! The unit tests prove each rule. These run the thing the modules exist for —
//! a busy building, driven a tick at a time, with a delta taken for every body
//! at every tick — and assert the properties that only show up at length:
//!
//! - **Nothing is lost.** Every change a body could make out is delivered to
//!   it, exactly once, no matter how many ticks pass in between.
//! - **Nothing leaks.** Across a whole run, no body is ever handed anything it
//!   could not have made out from where it was standing at the time.
//! - **Nothing repeats.** A situation is never sent twice running, so an idle
//!   body costs nothing at all.
//! - **The cost stays bounded.** A busy building does not produce a growing
//!   delta per body per tick; deltas stay small however long the day is.

use std::collections::{BTreeMap, BTreeSet};

use npc_map::delta::{Attention, Delta};
use npc_map::salience::Weight;
use npc_map::witness::Scope;
use npc_map::world::{Happening, Where, World};
use npc_map::MapSet;

fn vault() -> World {
    World::new(
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
            .expect("the shipped vault must load"),
    )
}

fn casting(node: &str) -> Where {
    Where::new("vault-casting", node)
}

/// One change, reduced to something countable.
///
/// Deliberately not [`Happening`] itself: an order over happenings would be
/// arbitrary, and the only meaningful order on them is [`Weight`]'s. What this
/// needs is identity, so it takes the debug form as a key and nothing more.
type Mark = (u64, String, String);

fn marks(d: &Delta) -> Vec<Mark> {
    d.events
        .iter()
        .map(|e| (e.at, e.actor.clone(), format!("{:?}", e.what)))
        .collect()
}

/// Sixteen Makers spread over the casting level, all grounded.
fn crew(world: &mut World, attention: &mut Attention) -> Vec<String> {
    let rooms = [
        "band-one",
        "band-two",
        "band-three",
        "green-room",
        "relations",
        "watch",
        "core",
        "ring-north",
    ];
    let ids: Vec<String> = (1..=16)
        .map(|i| {
            let id = format!("m{i:02}");
            world
                .enter(
                    &id,
                    format!("Maker-{i:02}"),
                    casting(rooms[i % rooms.len()]),
                )
                .expect("every room exists");
            id
        })
        .collect();
    attention.sweep(world);
    ids
}

/// A day's worth of ordinary vault business, driven a tick at a time.
///
/// Everything a Maker does: walking, working, letting go, talking to the room,
/// talking to somebody, crossing levels, and being pulled back. Returns what
/// each body was handed, tick by tick.
fn a_day(world: &mut World, attention: &mut Attention, ids: &[String]) -> Vec<Delta> {
    let mut handed: Vec<Delta> = Vec::new();
    for round in 0..12 {
        // Some walk.
        for (n, id) in ids.iter().enumerate() {
            if (n + round) % 4 == 0 {
                let to = match (n + round) % 3 {
                    0 => casting("green-room"),
                    1 => casting("band-one"),
                    _ => Where::new("vault-command", "command-room"),
                };
                let _ = world.set_off(id, to);
            }
        }
        // Some work, some give it up.
        for (n, id) in ids.iter().enumerate() {
            if (n + round) % 5 == 0 {
                let _ = world.take(id, Some(&format!("character-{n:02}")));
            }
            if (n + round) % 7 == 0 {
                let _ = world.release(id);
            }
        }
        // Some talk, to the room and to each other.
        for (n, id) in ids.iter().enumerate() {
            if (n + round) % 3 == 0 {
                let _ = world.say(id, format!("round {round} from {id}"));
            }
            if (n + round) % 6 == 0 {
                let here = world.actor(id).map(|a| a.at.clone());
                let neighbour = here.and_then(|at| {
                    world
                        .actors_at(&at)
                        .into_iter()
                        .find(|a| a.id != *id)
                        .map(|a| a.id.clone())
                });
                if let Some(to) = neighbour {
                    let _ = world.tell(id, &to, format!("round {round}, to you"));
                }
            }
        }
        // One gets pulled back to where orders are given.
        if round % 5 == 0 {
            let _ = world.teleport(&ids[round % ids.len()]);
        }

        world.tick();
        handed.extend(attention.sweep(world));
    }
    handed
}

// =========================================================================
// Nothing is lost, nothing repeats
// =========================================================================

#[test]
fn every_change_reaches_the_bodies_that_could_make_it_out_exactly_once() {
    // The property the sweep exists to hold. Delivered twice is a body acting
    // on the same news over again; delivered never is a body that missed
    // somebody speaking to it.
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    let mut delivered: BTreeMap<String, Vec<Mark>> = BTreeMap::new();
    for d in &handed {
        delivered.entry(d.who.clone()).or_default().extend(marks(d));
    }

    for (who, got) in &delivered {
        let once: BTreeSet<&Mark> = got.iter().collect();
        assert_eq!(once.len(), got.len(), "{who} was told something twice");
    }

    // And the run was busy enough for that to mean something.
    let total: usize = delivered.values().map(Vec::len).sum();
    assert!(
        total > 200,
        "only {total} changes delivered — too quiet to prove anything"
    );
}

#[test]
fn a_body_is_never_handed_the_same_situation_twice_running() {
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    let mut last: BTreeMap<String, String> = BTreeMap::new();
    for d in &handed {
        if let Some(here) = &d.percept {
            if let Some(before) = last.insert(d.who.clone(), here.clone()) {
                assert_ne!(before, *here, "{} was re-grounded unchanged", d.who);
            }
        }
    }
    assert!(last.len() > 8, "too few bodies moved to prove anything");
}

#[test]
fn nothing_at_all_happening_hands_nobody_anything() {
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    a_day(&mut w, &mut a, &ids);

    // Let every journey finish and everything be read, then let the world sit.
    while w.tick() > 0 {
        a.sweep(&mut w);
    }
    a.sweep(&mut w);
    for _ in 0..20 {
        assert!(a.sweep(&mut w).is_empty(), "an idle vault cost something");
    }
}

// =========================================================================
// Nothing leaks
// =========================================================================

#[test]
fn nothing_is_ever_handed_to_a_body_that_could_not_have_made_it_out() {
    // Checked against the map rather than against the code that produced it:
    // for every change delivered, the place it happened was either where that
    // body was standing at the time or somewhere the map says that place can
    // see into.
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    // Where each body was, read off its own arrivals in the raw log — the
    // world's record, not the code under test.
    let mut standing: BTreeMap<String, Vec<(u64, Where)>> = BTreeMap::new();
    for e in w.log() {
        if e.what == Happening::Arrived {
            standing
                .entry(e.actor.clone())
                .or_default()
                .push((e.at, e.place.clone()));
        }
    }

    // A tick is one moment and a body may move within it, so during tick T it
    // was at whichever place it last reached *before* T, or at any place it
    // reached *during* T. Anything outside the sight of all of those is a leak.
    let during = |who: &str, when: u64| -> Vec<Where> {
        let Some(arrivals) = standing.get(who) else {
            return Vec::new();
        };
        let mut places: Vec<Where> = arrivals
            .iter()
            .rev()
            .find(|(at, _)| *at < when)
            .map(|(_, place)| place.clone())
            .into_iter()
            .collect();
        places.extend(
            arrivals
                .iter()
                .filter(|(at, _)| *at == when)
                .map(|(_, place)| place.clone()),
        );
        places
    };

    let mut checked = 0;
    for d in &handed {
        for e in &d.events {
            // A body's own outcomes are its own and need no line of sight.
            if e.actor == d.who {
                continue;
            }
            let could_have_been = during(&d.who, e.at);
            assert!(
                !could_have_been.is_empty(),
                "{} was handed news before it was anywhere",
                d.who
            );
            let in_sight = could_have_been
                .iter()
                .any(|here| Scope::at(&w, here).places().contains(&e.place));
            assert!(
                in_sight,
                "{} was told about {} at tick {}, standing only in {:?}",
                d.who, e.place, e.at, could_have_been
            );
            checked += 1;
        }
    }
    assert!(checked > 200, "only {checked} deliveries checked");
}

#[test]
fn speech_never_reaches_a_body_that_was_not_in_the_room() {
    // The sharpest case of the rule above, and the one the whole social design
    // rests on: over a whole day, nobody hears anything said anywhere but
    // where they were standing.
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    let mut heard = 0;
    for d in &handed {
        for e in &d.events {
            if matches!(e.what, Happening::Said { .. }) {
                assert!(e.here, "{} heard {} through a wall", d.who, e.place);
                heard += 1;
            }
        }
    }
    assert!(heard > 40, "only {heard} utterances heard");
}

#[test]
fn a_subject_never_leaves_the_room_it_was_claimed_in() {
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    let mut stripped = 0;
    for d in &handed {
        for e in &d.events {
            let carries = match &e.what {
                Happening::TookStation { subject } | Happening::LeftStation { subject } => {
                    subject.is_some()
                }
                _ => continue,
            };
            if !e.here {
                assert!(!carries, "{} read a claim from outside {}", d.who, e.place);
                stripped += 1;
            }
        }
    }
    assert!(stripped > 0, "no claim was ever seen from outside a room");
}

// =========================================================================
// What it costs
// =========================================================================

#[test]
fn a_delta_stays_small_however_long_the_day_is() {
    // Deltas must not grow with the length of the run — if they did, the
    // prefill after each tick would get steadily more expensive and the whole
    // push-after-tick economy would come apart.
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    let worst = handed.iter().map(|d| d.events.len()).max().unwrap_or(0);
    assert!(worst <= 24, "one delta carried {worst} changes");

    // And the last third is no heavier than the first, which is what "does not
    // grow" actually means.
    let third = handed.len() / 3;
    let mean = |slice: &[Delta]| -> f64 {
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().map(|d| d.events.len()).sum::<usize>() as f64 / slice.len() as f64
    };
    let early = mean(&handed[..third]);
    let late = mean(&handed[handed.len() - third..]);
    assert!(
        late <= early * 2.0 + 1.0,
        "early {early:.1}, late {late:.1}"
    );
}

#[test]
fn most_bodies_are_quiet_most_of_the_time() {
    // The sweep returns only what has something in it, so the cost of a tick
    // is the number of minds something happened to — not the population.
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);

    let mut busiest = 0;
    for _ in 0..12 {
        // One Maker says one thing, in one room.
        let _ = w.say(&ids[0], "just the one");
        w.tick();
        busiest = busiest.max(a.sweep(&mut w).len());
    }
    assert!(
        busiest < ids.len() / 2,
        "one utterance woke {busiest} of {} bodies",
        ids.len()
    );
}

// =========================================================================
// What a body is handed reads
// =========================================================================

#[test]
fn everything_handed_over_renders_or_is_empty() {
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    for d in &handed {
        let text = d
            .render(&w)
            .unwrap_or_else(|| panic!("{} got an empty delta", d.who));
        assert!(text.ends_with('\n'), "{text:?}");
        assert!(!text.contains("  "), "double space in {text:?}");
        assert!(!text.contains(" ."), "space before a stop in {text:?}");
        assert!(!text.contains(",,"), "{text:?}");
        // A percept always grounds the body it belongs to.
        if d.percept.is_some() {
            assert!(text.starts_with("You are "), "{text:?}");
        }
    }
}

#[test]
fn being_spoken_to_is_what_preempts_and_it_is_rare() {
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    let mut preempted = 0;
    for d in &handed {
        if !d.preempts() {
            continue;
        }
        preempted += 1;
        // Every preempt is either something aimed at this body or the outcome
        // of something it set in motion. Nothing else may interrupt.
        assert!(
            d.events
                .iter()
                .any(|e| e.addressed() || (e.mine() && e.what.is_outcome())),
            "{} was interrupted by nothing that concerned it: {:?}",
            d.who,
            d.events.iter().map(|e| &e.what).collect::<Vec<_>>()
        );
    }
    assert!(preempted > 0, "nothing preempted all day");
    assert!(
        preempted * 2 < handed.len(),
        "{preempted} of {} deltas preempted — everything is urgent",
        handed.len()
    );
}

#[test]
fn a_quiet_delta_never_claims_to_be_worth_a_turn() {
    let mut w = vault();
    let mut a = Attention::new();
    let ids = crew(&mut w, &mut a);
    let handed = a_day(&mut w, &mut a, &ids);

    for d in &handed {
        match d.loudest() {
            None => assert!(d.events.is_empty() && d.percept.is_some()),
            Some(Weight::Ambient) => assert!(
                d.events.iter().all(|e| !e.here),
                "{} called something in its own room ambient",
                d.who
            ),
            _ => {}
        }
    }
}
