//! Operations through the table: a target in the ledger from the proposal that
//! opens its operation to the step that settles it, on the workflows npcd runs
//! (`docs/npcd_workflows_current.yaml`).

use std::collections::BTreeMap;

use super::{Missions, Settled};
use crate::engine::mission::bank::Facts;
use crate::engine::mission::{Mission, Origin, Outcome, Todo, Work};
use crate::sim::operations::tests::workflows;

/// A table running npcd's workflows.
fn table() -> Missions {
    let mut m = Missions::default();
    m.set_workflows(workflows());
    m
}

/// The workflow and document a target's operation runs on.
fn work_of(target: &str) -> (&'static str, String) {
    match target.split_once(':') {
        Some(("life", who)) => ("life-event", format!("layers/life/{who}/2488 X.md")),
        Some(("pair", _)) => ("correction", "layers/eras/a.md".to_string()),
        _ => (
            "story",
            format!("layers/stories/{}.md", target.replace(':', "-")),
        ),
    }
}

/// Open an operation for `target`, holding `fingerprint`.
fn launch(m: &mut Missions, target: &str, fingerprint: u64) -> u64 {
    let (workflow, document) = work_of(target);
    m.launch(
        workflow,
        "untold",
        target,
        fingerprint,
        &format!("Work on {target}"),
        &document,
        "What happens: something.",
        BTreeMap::new(),
        Vec::new(),
        None,
    )
    .unwrap()
}

/// Put every step a Maker owes on the table, as the engine's loop does: a
/// mission naming the step, its document, and the acts the step adds.
fn put_up(m: &mut Missions) {
    for id in m.operations().awaiting_offer() {
        let o = m.operations().get(id).unwrap();
        let step = o.step().unwrap().to_string();
        let tools = m.operations().offer(id).unwrap().step.tools.clone();
        let mission = Mission::new(
            format!("{step} {}", o.document),
            vec![Todo::new("do it")],
            Origin::Generated {
                generator: o.generator.clone(),
                target: o.target.clone(),
                operation: 0,
                step: String::new(),
            },
        )
        .with_work(Work {
            writes: o.document.clone(),
            reads: Vec::new(),
            min_words: 0,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools,
        });
        m.offer_step(id, mission);
    }
}

/// The step `body` collected, if it collected one of an operation's.
fn collect_step(m: &mut Missions, body: &str) -> Option<(u64, String)> {
    m.collect(body, &Facts::default())
        .operation()
        .map(|(id, step)| (id, step.to_string()))
}

fn a_mission(prompt: &str) -> Mission {
    Mission::new(
        prompt,
        vec![Todo::new("step one")],
        Origin::Lodged {
            by: "u_op".to_string(),
        },
    )
}

/// **An operation runs write, read, review, read again and canon, each Maker
/// step by somebody new to the round, and only then settles its target.**
/// Lodged work still comes first; the pool before the bank; and a done target
/// is free again only once what it is about has changed.
#[test]
fn an_operation_moves_its_target_through_the_ledger() {
    let mut m = table();
    let id = launch(&mut m, "life:keeper", 7);
    assert!(m.blocks("life:keeper", 7), "in hand");
    assert!(m.blocks("life:keeper", 8), "whatever it holds");
    put_up(&mut m);
    m.lodge("bram", a_mission("lodged first"));
    assert_eq!(m.collect("bram", &Facts::default()).prompt, "lodged first");
    m.cancel("bram");

    assert_eq!(collect_step(&mut m, "bram"), Some((id, "write".into())));
    assert_eq!(m.targets()["life:keeper"].state, Settled::Carried);
    m.report("bram", Outcome::Pass, "written", None);
    assert_eq!(
        m.targets()["life:keeper"].state,
        Settled::Carried,
        "written is not done"
    );
    assert_eq!(
        m.operations().awaiting_table(),
        vec![(id, "reading".into())]
    );

    m.table_took(id, "mend", "\"x\" — wrong").unwrap();
    put_up(&mut m);
    assert!(
        matches!(
            m.collect("bram", &Facts::default()).origin,
            Origin::Random { .. }
        ),
        "the writer draws past its own review"
    );
    m.cancel("bram");
    assert_eq!(collect_step(&mut m, "yen"), Some((id, "review".into())));
    m.report("yen", Outcome::Pass, "mended one line", None);
    m.table_took(id, "sound", "").unwrap();
    put_up(&mut m);
    assert_eq!(collect_step(&mut m, "cara"), Some((id, "canon".into())));
    m.report("cara", Outcome::Pass, "agrees with the Concord", None);

    assert!(m.operations().get(id).unwrap().succeeded());
    assert_eq!(m.targets()["life:keeper"].state, Settled::Done);
    assert!(m.blocks("life:keeper", 7), "done, and nothing has changed");
    assert!(!m.blocks("life:keeper", 8), "the life has a new event");
}

/// **A rejection sends the work to be fixed** — by anybody but the rejecter,
/// the writer too; rejected past the workflow's send-backs, the operation
/// fails and its target counts a stuck, so a fresh operation may try it. A
/// step that judges nothing cannot reject.
#[test]
fn a_rejection_sends_the_work_to_be_fixed_until_it_fails() {
    let mut m = table();
    let id = launch(&mut m, "era:x", 1);
    put_up(&mut m);
    collect_step(&mut m, "wren");
    assert!(
        m.reject("wren", "not reviewing").is_none(),
        "writing judges nothing"
    );
    m.report("wren", Outcome::Pass, "written", None);
    m.table_took(id, "sound", "").unwrap();
    put_up(&mut m);
    assert_eq!(collect_step(&mut m, "pax"), Some((id, "review".into())));
    let closed = m.reject("pax", "the scene is summary throughout").unwrap();
    assert_eq!(closed.report.unwrap().outcome, Outcome::Fail);
    let o = m.operations().get(id).unwrap();
    assert_eq!(o.step(), Some("fix"));
    assert_eq!(o.findings(), Some("the scene is summary throughout"));
    assert_eq!(m.targets()["era:x"].state, Settled::Carried);

    put_up(&mut m);
    assert!(!m.has_pooled_for("pax"), "not for its rejecter");
    assert_eq!(
        collect_step(&mut m, "wren"),
        Some((id, "fix".into())),
        "the writer may fix it"
    );

    for round in 0..3 {
        let reviewer = ["bram", "yen"][round % 2];
        m.report("wren", Outcome::Pass, "fixed", None);
        m.table_took(id, "sound", "").unwrap();
        put_up(&mut m);
        assert_eq!(
            collect_step(&mut m, reviewer),
            Some((id, "review".into())),
            "round {round}"
        );
        m.reject(reviewer, "still summary");
        if !m.operations().get(id).unwrap().settled() {
            put_up(&mut m);
            collect_step(&mut m, "wren");
        }
    }
    assert!(m.operations().get(id).unwrap().failed());
    assert_eq!(m.targets()["era:x"].state, Settled::Stuck);
    assert!(!m.blocks("era:x", 1), "stuck once is tried again");
}

/// **A step nobody may take is called off, not left at the table.** With two
/// Makers, the one who wrote cannot review and the one who reviewed cannot
/// check — so the check waits for a Maker who does not exist.
#[test]
fn a_step_no_maker_may_take_is_called_off() {
    let mut m = table();
    let id = launch(&mut m, "life:keeper", 1);
    put_up(&mut m);
    collect_step(&mut m, "wren");
    m.report("wren", Outcome::Pass, "written", None);
    m.table_took(id, "sound", "").unwrap();
    put_up(&mut m);
    collect_step(&mut m, "pax");
    m.report("pax", Outcome::Pass, "checked", None);
    m.table_took(id, "sound", "").unwrap();
    put_up(&mut m);

    assert!(
        m.call_off_untakeable(&[]).is_empty(),
        "nobody bound yet is not nobody left"
    );
    let cast = ["wren".to_string(), "pax".to_string()];
    assert_eq!(m.call_off_untakeable(&cast), vec![id]);
    assert!(m.pooled().is_empty());
    assert!(m.operations().get(id).unwrap().settled());
    assert!(!m.blocks("life:keeper", 1), "its target is free");
}

/// **Stuck where the step says**: a correction stuck at its writing fails, and
/// its target counts it; stuck twice on the same text, it is left until it
/// changes. A review stuck goes to somebody else first.
#[test]
fn stuck_goes_where_the_step_says() {
    let mut m = table();
    for round in 1..=2 {
        launch(&mut m, "pair:a|b", 3);
        put_up(&mut m);
        collect_step(&mut m, "wren");
        m.report("wren", Outcome::Fail, "no way to the desk", None);
        assert_eq!(m.targets()["pair:a|b"].stuck, round);
    }
    assert!(m.blocks("pair:a|b", 3), "stuck twice on the same text");
    assert!(!m.blocks("pair:a|b", 4));

    let id = launch(&mut m, "era:s", 1);
    put_up(&mut m);
    collect_step(&mut m, "wren");
    m.report("wren", Outcome::Pass, "written", None);
    m.table_took(id, "mend", "faults").unwrap();
    put_up(&mut m);
    collect_step(&mut m, "pax");
    m.report("pax", Outcome::Fail, "no desk", None);
    put_up(&mut m);
    assert!(!m.has_pooled_for("pax"));
    assert_eq!(collect_step(&mut m, "bram"), Some((id, "review".into())));
}

/// **An operation can be edited while it runs, called off, moved and
/// reopened**; the brief edits only while its mission waits at the table.
#[test]
fn an_operation_is_edited_cancelled_moved_and_reopened() {
    let mut m = table();
    let id = launch(&mut m, "era:y", 1);
    put_up(&mut m);
    let op = m
        .edit_operation(
            id,
            Some("Operation Long Watch"),
            Some("Tell the siege"),
            Some("New brief."),
        )
        .unwrap();
    assert_eq!(op.name, "Operation Long Watch");
    assert_eq!(m.waiting_brief(id), Some("New brief."));
    collect_step(&mut m, "wren");
    assert_eq!(m.carrying(id), Some("wren"));
    assert!(m.edit_operation(id, None, None, Some("too late")).is_err());
    assert!(m.move_to(id, "read").is_err(), "carried");
    assert!(m.cancel_operation(id, "not wanted"));
    assert!(!m.is_on_mission("wren"), "stood down");
    assert_eq!(m.take_stood_down(), vec!["wren".to_string()]);
    assert!(!m.blocks("era:y", 1));
    assert!(!m.cancel_operation(id, "again"));

    let r = m
        .review_document("story", "read", "layers/stories/old.md")
        .unwrap();
    assert_eq!(m.operations().awaiting_table(), vec![(r, "reading".into())]);
    m.table_took(r, "mend", "faults").unwrap();
    put_up(&mut m);
    m.move_to(r, "read").unwrap();
    assert!(m.pooled().is_empty(), "the waiting review left the table");
    assert_eq!(m.operations().awaiting_table(), vec![(r, "reading".into())]);

    m.reopen(id, "canon").unwrap();
    assert_eq!(m.operations().get(id).unwrap().step(), Some("canon"));
    m.send_to_step(id, "read").unwrap();
    assert_eq!(
        m.operations().get(id).unwrap().step(),
        Some("read"),
        "moved while running"
    );
}

/// **A report the workflow will not take still closes the Maker's mission**,
/// and the step it held goes back on offer — no Maker is left carrying work
/// it can never hand in.
#[test]
fn a_report_the_workflow_refuses_still_closes_the_mission() {
    let mut m = table();
    let id = launch(&mut m, "era:z", 1);
    put_up(&mut m);
    collect_step(&mut m, "wren");
    // The step it carries is moved out from under it by hand.
    m.operations.cancel(id, "moved");
    m.operations.reopen(id, "read").unwrap();
    assert!(m
        .report("wren", Outcome::Pass, "written it", None)
        .is_some());
    assert!(!m.is_on_mission("wren"));

    // One whose operation is no longer in the ledger at all.
    m.assign(
        "pax",
        Mission::new(
            "write it",
            vec![Todo::new("do it")],
            Origin::Generated {
                generator: "untold".into(),
                target: "era:gone".into(),
                operation: 99,
                step: "write".into(),
            },
        ),
    );
    assert!(m.report("pax", Outcome::Fail, "cannot", None).is_some());
    assert!(!m.is_on_mission("pax"));
}

/// **The table is stocked when every Maker has something it may take**, and
/// never past `keep` more than there are Makers.
#[test]
fn the_table_is_stocked_for_every_maker_not_only_by_count() {
    let mut m = table();
    let makers = vec!["ione".to_string(), "paxon".to_string()];
    assert!(!m.stocked_for(&makers, 1), "an empty table");
    let id = launch(&mut m, "era:t", 1);
    put_up(&mut m);
    collect_step(&mut m, "paxon");
    m.report("paxon", Outcome::Pass, "written", None);
    m.table_took(id, "sound", "").unwrap();
    put_up(&mut m);
    assert!(!m.stocked_for(&makers, 1), "nothing here Paxon may take");
    launch(&mut m, "era:u", 1);
    put_up(&mut m);
    assert!(m.stocked_for(&makers, 1), "something for each");
    assert!(!m.stocked_for(&makers, 3), "under keep");
}

/// **Discarding the pool releases every target in it**, and forgetting the
/// settled leaves what is in hand; a reserved target is held against a second
/// generation and let go when nothing comes of it.
#[test]
fn discard_forget_and_reserve_settle_targets_their_own_way() {
    let mut m = table();
    m.decline("pair:c|d", "boundaries", 5, "They agree.");
    assert!(m.blocks("pair:c|d", 5));
    assert!(!m.blocks("pair:c|d", 6));
    launch(&mut m, "era:y", 1);
    launch(&mut m, "era:z", 1);
    put_up(&mut m);
    assert_eq!(m.discard_pool(), 2);
    assert!(!m.blocks("era:y", 1));

    launch(&mut m, "era:w", 1);
    put_up(&mut m);
    assert_eq!(m.forget_settled(), 1, "the declined one");
    assert!(m.blocks("era:w", 1));

    assert!(m.reserve("life:keeper", "lives", 1));
    assert!(!m.reserve("life:keeper", "lives", 1), "held");
    m.release("life:keeper");
    assert!(m.reserve("life:keeper", "lives", 1), "released");
}
