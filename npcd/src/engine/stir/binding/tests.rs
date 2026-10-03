//! A fixture and its object stay in step, and a building says nothing about
//! objects the room does not hold.

use std::time::Duration;

use crate::engine::stir::binding::{Report, Set};
use crate::engine::stir::{Building, Stirring};

/// The four fixtures whose fault is the mode of an operable object:
/// `(fixture id, part, resting mode, fault mode)`.
const BOUND: &[(&str, &str, &str, &str)] = &[
    ("power", "breaker-panel", "closed", "tripped"),
    ("lights", "light-ring", "steady", "failing"),
    ("coolant", "coolant-valve", "tight", "weeping"),
    ("structure", "pressure-door", "seated", "hissing"),
];

/// Words that name a system. A room without the object must never say them.
const SYSTEM_WORDS: &[&str] = &[
    "coolant",
    "breaker",
    "status board",
    "supply panel",
    "pressure door",
    "door seal",
];

fn step(t: u64) -> Duration {
    Duration::from_secs(t)
}

fn lines(parts: &[&str], seeds: u64) -> Vec<String> {
    (0..seeds)
        .flat_map(|seed| {
            Building::new(seed * 104_729 + 7, Vec::new(), parts)
                .run(Duration::ZERO, step(25), 1200)
                .into_iter()
                .map(|(_, s)| s.text)
        })
        .collect()
}

#[test]
fn a_room_with_no_objects_is_fitted_with_only_what_needs_none() {
    let b = Building::new(1, Vec::new(), &[]);
    let ids = b.fitted_ids();
    assert!(ids.contains(&"ambient"), "the quiet always runs: {ids:?}");
    for gated in [
        "power",
        "lights",
        "coolant",
        "structure",
        "air",
        "compute",
        "boards",
        "stores",
    ] {
        assert!(
            !ids.contains(&gated),
            "{gated} runs in a room with nothing for it to be: {ids:?}"
        );
    }
}

#[test]
fn each_object_brings_its_fixture() {
    for (fixture, part, _, _) in BOUND {
        let ids = Building::new(1, Vec::new(), &[part]).fitted_ids();
        assert!(
            ids.contains(fixture),
            "{part} did not bring {fixture}: {ids:?}"
        );
    }
    for (fixture, part) in [
        ("air", "air-handler"),
        ("compute", "compute-rack"),
        ("boards", "status-board"),
        ("stores", "stores"),
    ] {
        let ids = Building::new(1, Vec::new(), &[part]).fitted_ids();
        assert!(
            ids.contains(&fixture),
            "{part} did not bring {fixture}: {ids:?}"
        );
    }
}

#[test]
fn a_room_with_no_objects_still_has_its_quiet() {
    let said = lines(&[], 4);
    assert!(
        said.len() > 100,
        "a room without objects fell silent: {} lines",
        said.len()
    );
}

#[test]
fn a_room_without_the_object_never_names_the_system() {
    for l in lines(&[], 24) {
        let low = l.to_lowercase();
        for word in SYSTEM_WORDS {
            assert!(
                !low.contains(word),
                "a room holding no objects said {word:?}: {l:?}"
            );
        }
    }
}

#[test]
fn a_fixture_that_fails_puts_its_object_in_the_fault_mode_and_recovers_it() {
    for (_, part, ok, fault) in BOUND {
        let mut saw_fault = false;
        // The coolant loop only goes wrong under strain, and the strain is the
        // room's own supply, so its valve is fitted beside a breaker panel.
        let fitted: Vec<&str> = match *part {
            "coolant-valve" => vec!["coolant-valve", "breaker-panel"],
            other => vec![other],
        };
        for seed in 0..40u64 {
            let mut b = Building::new(seed * 31 + 5, Vec::new(), &fitted);
            let mut log: Vec<Set> = Vec::new();
            for t in 0..3000u64 {
                b.next_event(step(t * 25));
                log.extend(b.take_sets().into_iter().filter(|s| s.part == *part));
            }
            let mut want_fault = true;
            for set in &log {
                assert_eq!(set.part, *part);
                let expected = if want_fault { fault } else { ok };
                assert_eq!(
                    set.mode, *expected,
                    "{part}: the object was written out of order: {log:?}"
                );
                want_fault = !want_fault;
            }
            saw_fault |= log.iter().any(|s| s.mode == *fault);
        }
        assert!(saw_fault, "{part}: the fixture never faulted in 40 shifts");
    }
}

#[test]
fn an_object_set_to_its_fault_mode_starts_the_fault() {
    for (fixture, part, _, fault) in BOUND {
        let mut b = Building::new(3, Vec::new(), &[part]);
        assert!(b.standing("here").is_empty());
        let said = b.tend(step(10), &[(part, fault)]);
        assert_eq!(said.len(), 1, "{part}: nothing was heard of it");
        assert_eq!(said[0].from, *fixture);
        let standing = b.standing("here");
        assert_eq!(standing.len(), 1, "{part}: no fault stands");
        assert_eq!(standing[0].room, "here");
        assert!(
            b.take_sets().is_empty(),
            "{part}: the object was written back what it already said"
        );
    }
}

#[test]
fn an_object_reset_to_its_resting_mode_clears_the_fault() {
    for (fixture, part, ok, fault) in BOUND {
        let mut b = Building::new(3, Vec::new(), &[part]);
        b.tend(step(10), &[(part, fault)]);
        let said = b.tend(step(20), &[(part, ok)]);
        assert_eq!(said.len(), 1, "{part}: the reset was not heard");
        assert_eq!(said[0].from, *fixture);
        assert!(b.standing("here").is_empty(), "{part}: the fault stood");
        assert!(b.take_sets().is_empty(), "{part}: the object was echoed");
    }
}

#[test]
fn an_object_that_agrees_with_its_fixture_says_nothing() {
    for (_, part, ok, _) in BOUND {
        let mut b = Building::new(3, Vec::new(), &[part]);
        assert!(b.tend(step(10), &[(part, ok)]).is_empty());
        assert!(b.tend(step(20), &[(part, ok)]).is_empty());
    }
}

#[test]
fn a_part_the_room_does_not_report_is_left_alone() {
    let mut b = Building::new(3, Vec::new(), &["breaker-panel"]);
    assert!(b.tend(step(10), &[]).is_empty());
    assert!(b.standing("here").is_empty());
}

#[test]
fn what_the_room_hears_of_an_operated_object_is_a_whole_sentence() {
    for (_, part, ok, fault) in BOUND {
        let mut b = Building::new(9, Vec::new(), &[part]);
        let mut said = b.tend(step(10), &[(part, fault)]);
        said.extend(b.tend(step(20), &[(part, ok)]));
        for s in said {
            assert!(s.text.chars().next().is_some_and(char::is_uppercase));
            assert!(s.text.ends_with('.'), "{:?}", s.text);
            assert!(s.text.split_whitespace().count() >= 4, "{:?}", s.text);
        }
    }
}

fn breaker() -> Report {
    Report {
        room: "the Plant Room".to_string(),
        object: "the breaker panel",
        system: "the main supply bus",
        trouble: "a breaker has tripped",
    }
}

/// Everything a board room says over `steps`, with `standing` reported to it.
fn board_lines(standing: Vec<Report>, steps: u64) -> Vec<Stirring> {
    let mut b = Building::new(31, Vec::new(), &["status-board"]);
    b.hear_of(standing);
    b.run(Duration::ZERO, step(30), steps as usize)
        .into_iter()
        .map(|(_, s)| s)
        .filter(|s| s.from == "boards")
        .collect()
}

#[test]
fn the_board_files_a_fault_that_really_stands() {
    let said = board_lines(vec![breaker()], 600);
    let entry = said
        .iter()
        .find(|s| s.text.contains("against the main supply bus"))
        .unwrap_or_else(|| panic!("the fault never reached the board: {said:?}"));
    let (part, text) = entry
        .posted
        .as_ref()
        .expect("the entry is written on the board");
    assert_eq!(*part, "status-board");
    assert!(text.contains("the Plant Room"), "{text:?}");
    assert!(text.contains("the breaker panel"), "{text:?}");
    assert!(text.contains("a breaker has tripped"), "{text:?}");
}

#[test]
fn the_board_files_a_standing_fault_once() {
    let said = board_lines(vec![breaker()], 1200);
    let filed = said
        .iter()
        .filter(|s| s.text.contains("fault entry has come up"))
        .count();
    assert_eq!(filed, 1, "{said:?}");
}

#[test]
fn a_board_with_nothing_standing_files_nothing() {
    let said = board_lines(Vec::new(), 1200);
    assert!(!said.is_empty(), "the board never did anything idle");
    assert!(
        said.iter().all(|s| s.posted.is_none()),
        "the board wrote on itself with nothing wrong: {said:?}"
    );
    assert!(
        said.iter().all(|s| !s.text.contains("fault entry")),
        "the board invented a fault: {said:?}"
    );
}

#[test]
fn the_board_takes_down_a_fault_that_is_gone() {
    let mut b = Building::new(31, Vec::new(), &["status-board"]);
    b.hear_of(vec![breaker()]);
    let filed = b.run(Duration::ZERO, step(30), 600);
    assert!(filed.iter().any(|(_, s)| s.text.contains("has come up")));

    b.hear_of(Vec::new());
    let after = b.run(step(600 * 30), step(30), 600);
    let cleared = after
        .iter()
        .map(|(_, s)| s)
        .find(|s| s.text.contains("clears off the status board"))
        .unwrap_or_else(|| panic!("the entry never came down: {after:?}"));
    assert!(
        cleared.text.contains("the main supply bus"),
        "{:?}",
        cleared.text
    );
    let (_, written) = cleared
        .posted
        .as_ref()
        .expect("the clearing is written too");
    assert!(written.contains("the Plant Room"), "{written:?}");
}

#[test]
fn a_standing_fault_in_the_room_is_reported_with_its_object_and_trouble() {
    let mut b = Building::new(3, Vec::new(), &["breaker-panel"]);
    b.tend(step(10), &[("breaker-panel", "tripped")]);
    let standing = b.standing("the Plant Room");
    assert_eq!(standing.len(), 1);
    assert_eq!(standing[0].object, "the breaker panel");
    assert_eq!(standing[0].system, "the main supply bus");
    assert!(!standing[0].trouble.is_empty());
}
