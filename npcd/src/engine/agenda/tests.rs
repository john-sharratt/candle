use serde_json::{json, Value};

use super::*;
use crate::engine::act::Act;
use crate::engine::body::Outcome;
use crate::engine::mission::{Mission, Origin, Outcome as Verdict, Todo};
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
        json!({"title": title, "brief": format!("Do {title}."), "steps": "go to the watch room\npost work on the order table", "priority": priority}),
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

/// **An item of the round is not closed by saying so**: `mission_done` is
/// refused for it, naming its next step and the act that does it, and it is
/// still carried.
#[test]
fn an_item_of_the_round_is_not_closed_by_saying_so() {
    use crate::engine::mission_gen::round;
    let h = vault();
    let items = round::parse(
        "rounds/creators-vault.yaml",
        "round:\n  - id: set-the-work\n    brief: Set the work.\n    steps:\n      - post work on \
         the order table\n",
    )
    .unwrap();
    h.with_sim(|s| {
        s.missions.set_round(items);
        round::collect(s, "zen", 0).expect("the round's item is collected");
    });
    assert_eq!(
        act(
            &h,
            "zen",
            "mission_done",
            json!({"mission": "set the work", "found": "posted it on the sensor panel"})
        ),
        Outcome::Refused(
            "\"set the work\" is an item of your round: it is done when its steps are seen done, \
             not when you say so. Its next step is to post work on the order table — \
             `order_post` at the order table, which answers you from wherever your attention is."
                .into()
        )
    );
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
            json!({"mission": "watch earth", "steps": "post work on the order table\n- scan the sensor panel and read what state it is in"})
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
            json!({"title": "x", "brief": "x", "steps": "go to the watch room", "priority": "now"})
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
        json!({"title": "x", "brief": "x", "steps": "\n - \n", "priority": "now"})
    )
    .happened());
    assert!(!add(&h, "x", "urgent").happened());
}

/// **A round item collected joins the agenda at `later`**, so a mission of
/// Zen's own goes ahead of it; and **a mission worked whose steps are all seen
/// done is closed by the engine**, the next taken up.
#[test]
fn a_round_item_waits_behind_zens_own_and_what_is_seen_done_closes() {
    let h = vault();
    h.with_sim(|s| {
        let m = Mission::new(
            "Watch Earth.",
            vec![Todo::new("go to the watch room")],
            Origin::Round {
                id: "watch-earth".into(),
            },
        );
        s.missions.take_round("zen", m, "watch earth", 0);
    });
    add(&h, "yen's parcels", "soon");
    assert_eq!(titles(&h), ["yen's parcels", "watch earth"]);
    h.with_sim(|s| s.missions.posted_work("zen"));
    assert!(
        h.with_sim(|s| s.missions.settle_agendas(0)).is_empty(),
        "one step still to go"
    );
    h.with_sim(|s| {
        s.missions
            .arrived_in("zen", "the watch room", "the command level")
    });
    let closed = h.with_sim(|s| s.missions.settle_agendas(0));
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
