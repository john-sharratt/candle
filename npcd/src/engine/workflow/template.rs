//! `docs/npcd_workflows_current.yaml` — today's pipeline written in the
//! workflow format — parsed whole, with its shapes and routes asserted.

use super::config::{By, Edits, Missions, Next, OnFailed, Variants};
use super::parse::parse_missions;
use super::run::{advance, may_take, start, Taker, Where};

const TEMPLATE: &str = include_str!("../../../../docs/npcd_workflows_current.yaml");

fn missions() -> Missions {
    parse_missions(TEMPLATE).unwrap()
}

fn outcomes(pairs: &[(&str, &str)]) -> Next {
    Next::Outcomes(
        pairs
            .iter()
            .map(|(o, t)| (o.to_string(), t.to_string()))
            .collect(),
    )
}

#[test]
fn the_template_declares_its_workflows_and_generators() {
    let m = missions();
    let names: Vec<&str> = m.workflows.iter().map(|w| w.name.as_str()).collect();
    assert_eq!(names, ["story", "life-event", "correction"]);
    assert_eq!(m.keep, Some(4));
    let generators: Vec<(&str, &str, &str, u32)> = m
        .generators
        .iter()
        .map(|g| {
            (
                g.id.as_str(),
                g.call.as_str(),
                g.workflow.as_str(),
                g.weight,
            )
        })
        .collect();
    assert_eq!(
        generators,
        [
            ("life-event", "life_event", "life-event", 3),
            ("contradiction", "correction", "correction", 2),
            ("untold", "story", "story", 1),
        ]
    );
    assert_eq!(
        m.generators[2].context,
        ["era", "timeline", "stories-told", "names-taken"]
    );
}

#[test]
fn story_steps_and_settings() {
    let m = missions();
    let story = m.workflow("story").unwrap();
    assert_eq!(story.send_backs, 3);
    assert_eq!(story.desk.as_deref(), Some("story-desk"));
    assert_eq!(story.year.as_deref(), Some("era-opens"));
    assert_eq!(story.on_failed, OnFailed::SetAside);
    let steps: Vec<(&str, By)> = story
        .steps
        .iter()
        .map(|s| (s.name.as_str(), s.by))
        .collect();
    assert_eq!(
        steps,
        [
            ("write", By::Maker),
            ("fix", By::Another),
            ("read", By::Table),
            ("review", By::Another),
            ("reread", By::Table),
            ("canon", By::Another),
        ]
    );
    assert_eq!(
        m.workflow("correction").unwrap().on_failed,
        OnFailed::Restore
    );
    assert_eq!(m.workflow("correction").unwrap().year, None);
}

#[test]
fn story_routes() {
    let m = missions();
    let story = m.workflow("story").unwrap();
    let review = story.step("review").unwrap();
    assert_eq!(
        review.next,
        outcomes(&[("pass", "reread"), ("reject", "fix")])
    );
    assert_eq!((review.stuck.as_str(), review.stuck_limit), ("fix", 2));
    assert_eq!(review.tools, ["report_rejected"]);
    assert_eq!(
        review.edits,
        Variants::ByOutcome(vec![
            ("sound".to_string(), Edits::Optional),
            ("mend".to_string(), Edits::Change),
            ("fail".to_string(), Edits::New),
            ("unread".to_string(), Edits::Change),
        ])
    );
    let reread = story.step("reread").unwrap();
    assert_eq!(reread.call.as_deref(), Some("reading"));
    assert_eq!(
        reread.next,
        outcomes(&[
            ("sound", "canon"),
            ("mend", "review"),
            ("fail", "fix"),
            ("unread", "canon")
        ])
    );
    assert_eq!(
        story.step("fix").unwrap().next,
        Next::Single("read".to_string())
    );
    assert_eq!(
        story.step("write").unwrap().edits,
        Variants::One(Edits::New)
    );
}

/// **Every route back — a reading that finds faults after a pass, a rejection
/// — is a send-back**, counted against the workflow's allowance: `fix` is
/// listed before the reading, so reaching it is going back.
#[test]
fn reread_mend_and_a_rejection_are_send_backs() {
    let m = missions();
    let story = m.workflow("story").unwrap();
    let (ann, bob, cara) = (
        Taker::Actor("ann".to_string()),
        Taker::Actor("bob".to_string()),
        Taker::Actor("cara".to_string()),
    );
    let mut run = start(story);
    advance(story, &mut run, &ann, None, "").unwrap();
    advance(story, &mut run, &Taker::Table, Some("sound"), "").unwrap();
    assert_eq!(
        advance(story, &mut run, &bob, Some("pass"), ""),
        Ok(Where::NextStep("reread".to_string()))
    );
    assert_eq!(run.send_backs, 0);
    assert_eq!(
        advance(story, &mut run, &Taker::Table, Some("mend"), "a date"),
        Ok(Where::NextStep("review".to_string()))
    );
    assert_eq!(run.send_backs, 1);
    assert_eq!(run.incoming.as_deref(), Some("mend"));
    assert_eq!(
        advance(story, &mut run, &cara, Some("reject"), "wrong era"),
        Ok(Where::NextStep("fix".to_string()))
    );
    assert_eq!(run.send_backs, 2);
    assert!(
        may_take(story, &run, &ann),
        "a new round: the writer may fix it"
    );
    assert!(!may_take(story, &run, &cara), "not whoever rejected it");
}

#[test]
fn review_variants_have_the_shared_review_text_spliced_in() {
    let m = missions();
    let review = &m.prompts["review"];
    let verdict = &m.prompts["verdict"];
    assert!(review.starts_with("{name} — {objective}.\n\nAnother Maker drafted `{document}`."));
    let story_review = &m.workflow("story").unwrap().step("review").unwrap().prompt;
    let Variants::ByOutcome(variants) = story_review else {
        panic!("story review has prompt variants");
    };
    let keys: Vec<&str> = variants.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["sound", "mend", "fail", "unread"]);
    for (key, text) in variants {
        assert!(text.starts_with(review.as_str()), "{key}");
        assert!(text.ends_with(&format!("{verdict}\n")), "{key}");
        assert!(
            !text.contains("{review}") && !text.contains("{verdict}"),
            "{key}"
        );
    }
    assert!(variants[0].1[review.len()..].starts_with("\nThe table found nothing to mend."));
    let life_review = &m
        .workflow("life-event")
        .unwrap()
        .step("review")
        .unwrap()
        .prompt;
    for key in ["sound", "mend", "fail", "unread"] {
        assert_eq!(
            life_review.variant(key).map(|t| t.trim_end()),
            story_review.variant(key).map(|t| t.trim_end()),
            "{key}"
        );
    }
}

#[test]
fn shared_prompts_resolve_into_table_and_maker_steps() {
    let m = missions();
    let story = m.workflow("story").unwrap();
    let read = &story.step("read").unwrap().prompt;
    assert_eq!(read, &Variants::One(format!("{}\n", m.prompts["reading"])));
    let Variants::One(write) = &story.step("write").unwrap().prompt else {
        panic!("story write has one prompt");
    };
    assert!(write.starts_with("Tell a story the record passes over: \"{title}\"."));
    assert!(write.ends_with(&format!("{}\n", m.prompts["elsewhere"])));
}
