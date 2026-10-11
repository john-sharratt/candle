//! A small `story` and a `repair` workflow, parsed from a whole
//! `missions.yaml` and walked end to end.

use std::collections::BTreeMap;

use super::config::{By, Next, Workflow};
use super::parse::parse_workflows;
use super::prompt::fill;
use super::run::{advance, may_take, offered, start, Taker, Where};

const MISSIONS: &str = r#"system: read by another reader
workflows:
  story:
    send-backs: 2
    steps:
      write:
        by: maker
        edits: new
        context:
          - brief
          - era
        next: read
        prompt:
          start: |
            Write {objective}.
          default: |
            Write {objective} again. What was found: {findings}
      read:
        by: table
        call: reading
        checks:
          - dates
          - names
        next:
          sound: review
          mend: review
          fail: write
        prompt: |
          Above is a draft of {objective}. Read it against the record.
      review:
        by: another
        next:
          pass: canon
          reject: write
        prompt: |
          Review the draft of {objective}. The reading found: {findings}
      canon:
        by: another
        next:
          pass: done
          fail: write
        prompt: |
          Check {objective} against the main storyline.
  repair:
    send-backs: 1
    steps:
      diagnose:
        by: maker
        next: fix
        prompt: |
          Diagnose {objective}.
      fix:
        by: maker
        next: inspect
        prompt: |
          Fix {objective}. Findings: {findings}
      inspect:
        by: another
        next:
          pass: done
          fail: fix
        prompt: |
          Inspect the fix of {objective}.
"#;

fn workflows() -> Vec<Workflow> {
    parse_workflows(MISSIONS).unwrap()
}

fn actor(a: &str) -> Taker {
    Taker::Actor(a.to_string())
}

fn next(step: &str) -> Result<Where, String> {
    Ok(Where::NextStep(step.to_string()))
}

#[test]
fn both_workflows_parse_in_order_with_their_shapes() {
    let all = workflows();
    let names: Vec<&str> = all.iter().map(|w| w.name.as_str()).collect();
    assert_eq!(names, ["story", "repair"]);

    let story = &all[0];
    assert_eq!(story.send_backs, 2);
    let steps: Vec<(&str, By)> = story
        .steps
        .iter()
        .map(|s| (s.name.as_str(), s.by))
        .collect();
    assert_eq!(
        steps,
        [
            ("write", By::Maker),
            ("read", By::Table),
            ("review", By::Another),
            ("canon", By::Another),
        ]
    );
    assert_eq!(story.start().next, Next::Single("read".to_string()));
    assert_eq!(story.start().context, ["brief", "era"]);
    assert_eq!(story.step("read").unwrap().checks, ["dates", "names"]);
    assert_eq!(
        story.step("read").unwrap().targets(),
        ["review", "review", "write"]
    );
    assert_eq!(
        story.step("canon").unwrap().targets(),
        ["done", "write", "failed"]
    );
    assert_eq!(all[1].start().name, "diagnose");
}

/// Write, a reading that sends it back, write again, a reading to mend, a
/// review by a second actor, and a canon check by a third.
#[test]
fn story_routes_back_through_write_and_settles_done() {
    let all = workflows();
    let wf = &all[0];
    let mut run = start(wf);

    assert_eq!(
        advance(wf, &mut run, &actor("ann"), None, "draft one"),
        next("read")
    );
    assert_eq!(
        advance(
            wf,
            &mut run,
            &Taker::Table,
            Some("fail"),
            "the dates disagree"
        ),
        next("write")
    );

    let values = BTreeMap::from([
        ("objective", "the flood of Arn"),
        ("findings", run.last_findings().unwrap_or("")),
    ]);
    assert_eq!(
        fill(offered(wf, &run).unwrap().prompt, &values),
        Ok("Write the flood of Arn again. What was found: the dates disagree\n".to_string())
    );

    assert_eq!(
        advance(wf, &mut run, &actor("ann"), None, "draft two"),
        next("read")
    );
    assert_eq!(
        advance(
            wf,
            &mut run,
            &Taker::Table,
            Some("mend"),
            "one name to mend"
        ),
        next("review")
    );
    assert!(!may_take(wf, &run, &actor("ann")));
    assert_eq!(
        advance(wf, &mut run, &actor("bob"), Some("pass"), "mended"),
        next("canon")
    );
    assert!(!may_take(wf, &run, &actor("bob")));
    assert_eq!(
        advance(wf, &mut run, &actor("cara"), Some("pass"), ""),
        Ok(Where::Done)
    );
    let steps: Vec<&str> = run.history.iter().map(|t| t.step.as_str()).collect();
    assert_eq!(steps, ["write", "read", "write", "read", "review", "canon"]);
    assert_eq!(run.send_backs, 1);
}

#[test]
fn story_rejected_twice_more_runs_out_of_send_backs() {
    let all = workflows();
    let wf = &all[0];
    let mut run = start(wf);
    for (reviewer, n) in [("bob", 1), ("cara", 2)] {
        advance(wf, &mut run, &actor("ann"), None, "").unwrap();
        advance(wf, &mut run, &Taker::Table, Some("sound"), "").unwrap();
        assert_eq!(
            advance(
                wf,
                &mut run,
                &actor(reviewer),
                Some("reject"),
                "it contradicts era two"
            ),
            next("write")
        );
        assert_eq!(run.send_backs, n);
    }
    advance(wf, &mut run, &actor("ann"), None, "").unwrap();
    assert_eq!(
        advance(wf, &mut run, &Taker::Table, Some("fail"), "still wrong"),
        Ok(Where::Failed(
            "sent back 3 times, more than the 2 allowed; step `read` last found: still wrong"
                .to_string()
        ))
    );
}

#[test]
fn repair_sends_a_failed_inspection_back_to_a_fix_the_inspector_does_not_inspect() {
    let all = workflows();
    let wf = &all[1];
    let mut run = start(wf);
    assert_eq!(
        advance(wf, &mut run, &actor("ann"), None, "the hinge"),
        next("fix")
    );
    assert_eq!(
        advance(wf, &mut run, &actor("ann"), None, ""),
        next("inspect")
    );
    assert!(!may_take(wf, &run, &actor("ann")));
    assert_eq!(
        advance(wf, &mut run, &actor("bob"), Some("fail"), "still squeaks"),
        next("fix")
    );
    assert_eq!(run.last_findings(), Some("still squeaks"));
    assert_eq!(
        advance(wf, &mut run, &actor("ann"), None, "oiled"),
        next("inspect")
    );
    assert!(!may_take(wf, &run, &actor("bob")));
    assert!(!may_take(wf, &run, &actor("ann")));
    assert_eq!(
        advance(wf, &mut run, &actor("cara"), Some("fail"), "worse"),
        Ok(Where::Failed(
            "sent back 2 times, more than the 1 allowed; step `inspect` last found: worse"
                .to_string()
        ))
    );
}
