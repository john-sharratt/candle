use serde_json::{json, Value};

use super::*;
use crate::engine::act::Act;
use crate::engine::body::Outcome;
use crate::engine::mission::{Outcome as Verdict, StepOutcome};
use crate::sim::agenda::{Priority, KEEP};
use crate::world::Hosted;

/// The vault with Zen, who takes no orders, and Wick, who does.
fn vault() -> Hosted {
    let h = Hosted::load(
        "creators-vault",
        concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
    )
    .expect("the shipped vault must load");
    let room = npc_map::world::Where::new("vault-command", "command-room");
    h.with(|w| {
        w.enter("zen", "Zen", room.clone()).unwrap();
        w.enter("wick", "Wick", room).unwrap();
    });
    h.with_sim(|s| s.missions.set_unordered("zen", true));
    h
}

fn act(h: &Hosted, body: &str, tool: &'static str, args: Value) -> Outcome {
    perform(
        h,
        body,
        &Act {
            tool,
            args: args.as_object().unwrap().clone(),
        },
    )
}

fn add(h: &Hosted, title: &str, priority: &str) -> Outcome {
    act(
        h,
        "zen",
        "mission_add",
        json!({"title": title, "brief": format!("Do {title}."), "steps": ["go to the watch room", "post work on the order table"], "priority": priority}),
    )
}

/// The agenda's titles, the one worked first.
fn titles(h: &Hosted) -> Vec<String> {
    h.sim(|s| s.missions.agenda_titles("zen"))
}

/// **A mission at `now` is worked at once, and the one it displaces waits
/// where it was left**: its signed-off step still signed off, taken up again
/// when the urgent one is done.
#[test]
fn a_mission_at_now_goes_first_and_the_one_it_displaces_keeps_its_place() {
    let h = vault();
    assert_eq!(
        add(&h, "watch earth", "later"),
        Outcome::Did("On your agenda, and worked now: watch earth.".into())
    );
    h.with_sim(|s| s.missions.posted_work("zen"));
    assert_eq!(
        add(&h, "yen's parcels", "now"),
        Outcome::Did("On your agenda, and worked now: yen's parcels.".into())
    );
    assert_eq!(titles(&h), ["yen's parcels", "watch earth"]);
    assert_eq!(
        add(&h, "the story circle", "soon"),
        Outcome::Did(
            "On your agenda: the story circle. You are working on yen's parcels first.".into()
        )
    );
    assert_eq!(
        titles(&h),
        ["yen's parcels", "the story circle", "watch earth"]
    );

    let prompt = h.sim(|s| s.missions.agenda_prompt("zen")).unwrap().text;
    assert!(prompt.starts_with("Do yen's parcels."), "{prompt}");
    assert!(
        prompt.ends_with(
            "Also on your agenda, set aside until this is done:\n- the story circle (soon) — \
             next: go to the watch room\n- watch earth (later) — next: go to the watch room"
        ),
        "{prompt}"
    );

    // The journey is seen made; the rest of the item is Zen's to judge.
    h.with_sim(|s| {
        s.missions
            .check_off("zen", "go to the watch room", StepOutcome::Achieved)
    });
    assert_eq!(
        act(
            &h,
            "zen",
            "mission_done",
            json!({"mission": "Yen's parcels", "found": "John has been reminded"})
        ),
        Outcome::Did("Done: Yen's parcels. Next on your agenda: the story circle.".into())
    );
    assert_eq!(
        act(
            &h,
            "zen",
            "mission_cancel",
            json!({"mission": "the story circle", "why": "nobody is in the warren"})
        ),
        Outcome::Did("Dropped: the story circle. Next on your agenda: watch earth.".into())
    );
    let worked = h.sim(|s| s.missions.active("zen").cloned()).unwrap();
    assert_eq!(
        worked.todo.iter().map(|t| t.done).collect::<Vec<_>>(),
        [false, true],
        "the step it had done is still done"
    );
    assert_eq!(h.sim(|s| s.missions.chapter("zen")), 0, "no chapter turned");
}

/// **A step the world sees done is not done by saying so**: an agenda item
/// whose journey or reading has not been seen is refused `mission_done`, with
/// how it is done — by attention, or by a Drone — and is still carried.
#[test]
fn an_agenda_item_with_a_step_unseen_is_not_closed_by_saying_so() {
    let h = vault();
    act(
        &h,
        "zen",
        "mission_add",
        json!({"title": "the sensor panel", "brief": "A Drone is to read the sensor panel.", "steps": ["go to the watch room"], "priority": "now"}),
    );
    let Outcome::Refused(why) = act(
        &h,
        "zen",
        "mission_done",
        json!({"mission": "the sensor panel", "found": "active and reading stable"}),
    ) else {
        panic!("a step never seen was closed by saying so");
    };
    assert!(
        why.starts_with(
            "\"the sensor panel\" is not done: \"go to the watch room\" is a step the world sees \
             done, and it has not seen it."
        ),
        "{why}"
    );
    assert!(why.contains("`order_post`"), "{why}");
    assert!(h.sim(|s| s.missions.is_on_mission("zen")), "still carried");
}

/// **Editing keeps what was done**: steps it keeps stay signed off, and a
/// raised priority puts it first.
#[test]
fn an_edit_keeps_what_was_done_and_a_raised_priority_goes_first() {
    let h = vault();
    add(&h, "watch earth", "soon");
    add(&h, "the hatchery", "later");
    h.with_sim(|s| s.missions.posted_work("zen"));
    assert_eq!(
        act(
            &h,
            "zen",
            "mission_edit",
            json!({"mission": "watch earth", "steps": ["post work on the order table", "scan the sensor panel and read what state it is in"]})
        ),
        Outcome::Did("On your agenda, and worked now: watch earth.".into())
    );
    let worked = h.sim(|s| s.missions.active("zen").cloned()).unwrap();
    assert_eq!(
        worked
            .todo
            .iter()
            .map(|t| (t.text.as_str(), t.done))
            .collect::<Vec<_>>(),
        [
            ("post work on the order table", true),
            ("scan the sensor panel and read what state it is in", false)
        ]
    );
    act(
        &h,
        "zen",
        "mission_edit",
        json!({"mission": "the hatchery", "priority": "now"}),
    );
    assert_eq!(titles(&h), ["the hatchery", "watch earth"]);
    assert!(!act(
        &h,
        "zen",
        "mission_edit",
        json!({"mission": "the hatchery"})
    )
    .happened());
    assert!(!act(
        &h,
        "zen",
        "mission_edit",
        json!({"mission": "nothing", "priority": "now"})
    )
    .happened());
}

/// **The agenda is Zen's alone, and it is bounded**: a character that takes
/// orders keeps none; a ninth mission is refused naming the lowest; a title
/// used twice, a mission with no step, or an unknown priority is refused.
#[test]
fn the_agenda_is_bounded_and_kept_only_by_one_who_takes_no_orders() {
    let h = vault();
    assert_eq!(
        act(
            &h,
            "wick",
            "mission_add",
            json!({"title": "x", "brief": "x", "steps": ["go to the watch room"], "priority": "now"})
        ),
        Outcome::Refused(
            "You keep no agenda of your own: your work is what the table gives you.".into()
        )
    );
    for i in 0..KEEP {
        let p = if i == 3 { "later" } else { "soon" };
        assert!(add(&h, &format!("job {i}"), p).happened());
    }
    assert_eq!(
        add(&h, "one too many", "now"),
        Outcome::Refused(format!(
            "Your agenda is full — {KEEP} missions. Finish or drop one first: the lowest is \
             \"job 3\"."
        ))
    );
    assert!(!add(&h, "Job 1", "now").happened(), "a title twice");
    let h = vault();
    assert!(!act(
        &h,
        "zen",
        "mission_add",
        json!({"title": "x", "brief": "x", "steps": [" ", ""], "priority": "now"})
    )
    .happened());
    assert!(!add(&h, "x", "urgent").happened());
}

/// **A mission worked whose steps are all seen done is closed by the
/// engine**, with no report asked, and the next on the agenda taken up.
#[test]
fn what_is_seen_done_on_the_agenda_closes_and_the_next_is_taken_up() {
    let h = vault();
    add(&h, "watch earth", "later");
    add(&h, "yen's parcels", "soon");
    assert_eq!(titles(&h), ["yen's parcels", "watch earth"]);
    h.with_sim(|s| s.missions.posted_work("zen"));
    assert!(
        h.with_sim(|s| s.missions.settle_agendas()).is_empty(),
        "one step still to go"
    );
    h.with_sim(|s| {
        s.missions
            .arrived_in("zen", "the watch room", "the command level")
    });
    let closed = h.with_sim(|s| s.missions.settle_agendas());
    assert_eq!(
        closed,
        [(
            "zen".to_string(),
            "yen's parcels".to_string(),
            Verdict::Pass
        )]
    );
    assert_eq!(titles(&h), ["watch earth"]);
    assert_eq!(
        h.sim(|s| s.missions.active("zen").and_then(|m| m.agendum.clone()))
            .map(|a| a.priority),
        Some(Priority::Later)
    );
}
