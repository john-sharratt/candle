use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use npc_map::world::Where;
use serde_json::{json, Value};
use tempfile::TempDir;
use zend_vfs::{GitError, Push, PushResult, PushSpec, RemoteName, Repo};

use super::*;
use crate::engine::act::Act;
use crate::engine::mission::{Mission, Origin, Todo, Work};
use crate::engine::work::{self, report_block};
use crate::engine::workflow::Taken;
use crate::opstore::shelf::StoreShelf;
use crate::sim::operations::tests::workflows;
use crate::sim::operations::Operations;
use crate::sim::shelf::Shelf;

const STORY: &str = "layers/stories/the-gate.md";

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn configure(dir: &Path) {
    for (k, v) in [
        ("core.autocrlf", "false"),
        ("user.name", "Mind"),
        ("user.email", "mind@example.com"),
        ("commit.gpgSign", "false"),
    ] {
        git(dir, &["config", k, v]);
    }
}

/// The vault over a mind folder cloned from a bare `origin`, with `m1` on the
/// publish step of an operation whose store holds a story.
fn at_the_merge_table() -> (TempDir, Hosted, u64) {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let seed = root.join("seed");
    std::fs::create_dir_all(seed.join("layers/eras")).unwrap();
    git(&seed, &["init", "-q", "--template="]);
    git(&seed, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    configure(&seed);
    std::fs::write(seed.join("layers/eras/third.md"), "# The third era\n").unwrap();
    git(&seed, &["add", "-A"]);
    git(&seed, &["commit", "-q", "-m", "seed"]);
    git(root, &["clone", "-q", "--bare", "seed", "origin"]);
    git(
        root,
        &["clone", "-q", "-c", "core.autocrlf=false", "origin", "mind"],
    );
    let mind = root.join("mind");
    configure(&mind);

    let h = Hosted::load(
        "creators-vault",
        concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
    )
    .unwrap();
    h.with(|w| {
        w.enter(
            "m1",
            "Perrin Vastwood",
            Where::new("vault-archive", "merge-room"),
        )
        .unwrap()
    });
    h.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
        &mind,
    ))));
    let stores = MindStores::new(Some(&mind));
    stores.set_quiet(Box::new(|_| true));
    h.set_op_stores(Arc::clone(&stores));
    let id = h.with_sim(|s| {
        s.missions.set_workflows(workflows());
        let id = s
            .missions
            .review_document("story", "publish", STORY)
            .unwrap();
        s.missions.assign(
            "m1",
            Mission::new(
                "Land the story in the record.",
                vec![Todo::report("report it at the table")],
                Origin::Generated {
                    generator: "operator".into(),
                    target: format!("doc:{STORY}"),
                    operation: id,
                    step: "publish".into(),
                },
            )
            .with_work(Work {
                writes: STORY.into(),
                reads: Vec::new(),
                min_words: 0,
                edit_optional: true,
                anew: false,
                checks: vec!["published".into()],
                tools: vec![
                    "merge_status".into(),
                    "merge_sync".into(),
                    "merge_publish".into(),
                ],
            }),
        );
        id
    });
    stores.pin(&Owner::operation(h.id(), id));
    StoreShelf {
        stores,
        owner: Owner::operation(h.id(), id),
    }
    .commit(&[(
        STORY.into(),
        Some("# The Gate\n\nA night at the gate.\n".into()),
    )])
    .unwrap();
    (dir, h, id)
}

fn did(o: Outcome) -> String {
    match o {
        Outcome::Did(line) => line,
        other => panic!("not done: {other:?}"),
    }
}

fn refused(o: Outcome) -> String {
    match o {
        Outcome::Refused(why) => why,
        other => panic!("not refused: {other:?}"),
    }
}

/// **A stuck report on work that has not landed is sent to the landing**, not
/// to `report_done`, which would refuse it and send the Maker back: the two
/// desks pointed at each other and a Maker went between them for an hour.
#[test]
fn a_stuck_report_before_the_landing_names_the_landing() {
    let (_dir, h, _id) = at_the_merge_table();
    let why = h
        .sim(|s| work::finished(s, "m1"))
        .expect("a finished mission refuses a stuck report");
    assert!(why.contains("done but the landing"), "{why}");
    assert!(why.contains("has not landed in the record yet"), "{why}");
    assert!(why.contains("`merge_publish`"), "{why}");
    assert!(!why.contains("What is left is the report"), "{why}");
}

/// **The publish step's report waits on the record; `merge_publish` lands
/// the work, records it on the operation, and brings the mind folder up.**
#[test]
fn a_publish_lands_and_the_step_may_be_reported() {
    let (dir, h, id) = at_the_merge_table();
    assert!(report_block(&h, "m1")
        .unwrap()
        .contains("has not landed in the record yet"));
    assert!(did(perform(&h, "m1", "merge_status")).contains("`merge_publish` lands it"));

    let landed = did(perform(&h, "m1", "merge_publish"));
    assert!(
        landed.starts_with("It is in the record now, as "),
        "{landed}"
    );
    assert!(landed.ends_with(&format!("{STORY}. Report it done at the table.")));
    let on_origin = git(&dir.path().join("origin"), &["rev-parse", "main"]);
    assert_eq!(
        h.sim(|s| s.missions.operations().get(id).unwrap().published.clone()),
        Some(on_origin.trim().to_string())
    );
    assert_eq!(
        git(
            &dir.path().join("origin"),
            &["show", &format!("main:{STORY}")]
        ),
        "# The Gate\n\nA night at the gate.\n",
        "origin holds what was published"
    );
    assert_eq!(
        *h.op_stores().unwrap().moves().borrow(),
        0,
        "the mind moves when the log's branch does, not on the publish"
    );
    assert_eq!(report_block(&h, "m1"), None, "landed: the step stands");
    let reported = work::perform(
        &h,
        "m1",
        &act("report_done", json!({"found": "It landed in the record."})),
    );
    assert!(reported.happened(), "{reported:?}");
    assert!(h.sim(|s| s.missions.operations().get(id).unwrap().succeeded()));
}

fn act(tool: &'static str, args: Value) -> Act {
    Act {
        tool,
        args: args.as_object().unwrap().clone(),
    }
}

/// **A conflict is solved at the bench**: the publish is refused naming the
/// document, `merge_sync` marks the overlap in the store, the Maker settles
/// it with the bench's own acts, and the publish lands — without going back
/// through verification.
#[test]
fn a_conflict_is_solved_at_the_bench_and_lands() {
    let (dir, h, id) = at_the_merge_table();
    // Another member publishes the same story first, told differently.
    let other = dir.path().join("other");
    git(
        dir.path(),
        &[
            "clone",
            "-q",
            "-c",
            "core.autocrlf=false",
            "origin",
            "other",
        ],
    );
    configure(&other);
    std::fs::create_dir_all(other.join("layers/stories")).unwrap();
    std::fs::write(other.join(STORY), "# The Gate\n\nA morning at the gate.\n").unwrap();
    git(&other, &["add", "-A"]);
    git(&other, &["commit", "-q", "-m", "elsewhere"]);
    git(&other, &["push", "-q", "origin", "main"]);

    let refused_publish = refused(perform(&h, "m1", "merge_publish"));
    assert!(
        refused_publish.contains(&format!("at the lines it changes, in {STORY}")),
        "{refused_publish}"
    );
    let synced = did(perform(&h, "m1", "merge_sync"));
    assert!(
        synced.contains(&format!("{STORY} hold both sides")),
        "{synced}"
    );
    let marked = h.op_shelf(id).unwrap().read(STORY).unwrap();
    assert!(marked.contains("<<<<<<< yours") && marked.contains(">>>>>>> the record"));
    assert!(refused(perform(&h, "m1", "merge_publish")).contains("still hold the merge's marks"));

    let settled = "# The Gate\n\nA night at the gate, and the morning after.\n";
    assert!(work::perform(
        &h,
        "m1",
        &act("file_write", json!({"path": STORY, "content": settled}))
    )
    .happened());
    assert!(work::perform(
        &h,
        "m1",
        &act("bench_commit", json!({"why": "settled the overlap"}))
    )
    .happened());
    assert!(did(perform(&h, "m1", "merge_publish")).starts_with("It is in the record now"));
    assert_eq!(
        git(
            &dir.path().join("origin"),
            &["show", &format!("main:{STORY}")]
        ),
        settled
    );
    assert!(work::perform(
        &h,
        "m1",
        &act(
            "report_done",
            json!({"found": "Settled the overlap and published it."})
        )
    )
    .happened());
    let op = h.sim(|s| s.missions.operations().get(id).unwrap().clone());
    assert!(op.succeeded(), "{:?}", op.run.at);
    let steps: Vec<&str> = op.run.history.iter().map(|t| t.step.as_str()).collect();
    assert_eq!(steps, ["publish"], "nothing went back through verification");
}

/// A push that cannot reach its agent.
struct NoAgent;

impl Push for NoAgent {
    fn push(&self, _: &Repo, _: &RemoteName, _: &[PushSpec]) -> Result<Vec<PushResult>, GitError> {
        Err(GitError::Io(std::io::Error::other("no agent can push")))
    }
}

/// **A publish that cannot reach git puts the mission down** and has the
/// step offered again later, nothing else changed.
#[test]
fn an_unreachable_publish_puts_the_mission_down() {
    let (_dir, h, id) = at_the_merge_table();
    h.op_stores().unwrap().set_pusher(Arc::new(NoAgent));
    let said = did(perform(&h, "m1", "merge_publish"));
    assert!(said.starts_with("The record cannot be reached: "), "{said}");
    assert!(said.contains("no agent can push"), "{said}");
    assert!(!h.sim(|s| s.missions.is_on_mission("m1")));
    let op = h.sim(|s| s.missions.operations().get(id).unwrap().clone());
    assert_eq!(op.step(), Some("publish"));
    assert_eq!(op.published, None);
    assert!(op.retry.unwrap().why.contains("no agent can push"));
}

/// Nothing here for a body on no publish step.
#[test]
fn a_body_on_no_publish_step_has_nothing_here() {
    let (_dir, h, _) = at_the_merge_table();
    h.with_sim(|s| s.missions.cancel("m1"));
    for tool in ["merge_status", "merge_publish", "merge_sync"] {
        assert_eq!(refused(perform(&h, "m1", tool)), NO_PUBLISH);
    }
}

/// **The commit names the operation, its document, and every step's taker.**
#[test]
fn the_commit_message_names_the_operation_and_its_takers() {
    let mut op = workflows_op();
    op.run.history = vec![
        taken("write", Taker::Actor("m1".into())),
        taken("read", Taker::Table),
        taken("review", Taker::Actor("m2".into())),
    ];
    assert_eq!(
        commit_message("creators-vault", &op),
        "Operation Iron Lantern: Tell the gate\n\nlayers/stories/the-gate.md\n\nSteps:\n- write: \
         m1\n- read: the table\n- review: m2\n\nOperation 1 of creators-vault.\n"
    );
}

fn workflows_op() -> Operation {
    let mut ops = Operations::default();
    ops.set_workflows(workflows());
    let id = ops
        .open("story", None, "untold", "era:x", "Tell the gate", STORY)
        .unwrap();
    ops.get(id).unwrap().clone()
}

fn taken(step: &str, by: Taker) -> Taken {
    Taken {
        step: step.into(),
        by,
        outcome: None,
        to: String::new(),
        notes: String::new(),
    }
}
