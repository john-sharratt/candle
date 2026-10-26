use std::collections::BTreeSet;
use std::path::PathBuf;

use doc_units::Op;

use super::*;
use crate::sim::shelf::{Published, NO_OPERATION};

/// A shelf over a folder that takes commits: reads and lists as the
/// published mind does, and writes what is committed into the folder — a
/// stand-in for an operation's store the tests can look at on disk.
struct Folder(PathBuf);

impl Shelf for Folder {
    fn read(&self, key: &str) -> Option<String> {
        Published::folder(self.0.clone()).read(key)
    }

    fn list(&self, dir: &str) -> BTreeSet<String> {
        Published::folder(self.0.clone()).list(dir)
    }

    fn commit(&self, changes: &[(String, Option<String>)]) -> Result<(), String> {
        for (key, now) in changes {
            let full = self.0.join(key);
            match now {
                Some(text) => {
                    std::fs::create_dir_all(full.parent().unwrap()).unwrap();
                    std::fs::write(&full, text).unwrap();
                }
                None => std::fs::remove_file(&full).unwrap(),
            }
        }
        Ok(())
    }
}

/// A root with two documents in it, at a path unique to this test, and a
/// shelf over it.
fn rooted(name: &str) -> (Benches, PathBuf, Folder) {
    let root = std::env::temp_dir().join(format!("npcd-bench-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("layers/eras")).unwrap();
    std::fs::write(
        root.join("layers/eras/third.md"),
        "the third era\nburned in the spring\n",
    )
    .unwrap();
    std::fs::write(root.join("layers/eras/fourth.md"), "the fourth era\n").unwrap();
    let mut b = Benches::new();
    b.set_mind(crate::mind::Mind::new(Some(root.clone())));
    let s = Folder(root.clone());
    (b, root, s)
}

/// What a working set would write to a path is read in any case, and only
/// while it is a change: an unchanged or removed document has no text.
#[test]
fn a_working_set_names_what_it_would_write() {
    let mut w = Working::default();
    let change = |base: Option<&str>, now: Option<&str>| Change {
        base: base.map(str::to_string),
        now: now.map(str::to_string),
    };
    w.files.insert(
        "layers/stories/a.md".into(),
        change(None, Some("a new story\n")),
    );
    w.files.insert(
        "layers/eras/third.md".into(),
        change(Some("same\n"), Some("same\n")),
    );
    w.files
        .insert("layers/eras/fourth.md".into(), change(Some("gone\n"), None));
    assert_eq!(w.changed_text("Layers/Stories/A.md"), Some("a new story\n"));
    assert_eq!(w.changed_text("layers/eras/third.md"), None, "unchanged");
    assert_eq!(w.changed_text("layers/eras/fourth.md"), None, "taken out");
    assert_eq!(w.changed_text("layers/eras/fifth.md"), None);
}

/// **A working copy comes back through a save and a restore**, under the
/// root the restoring daemon was given rather than the saved one.
#[test]
fn uncommitted_work_survives_a_save_and_restore() {
    let (mut b, root, s) = rooted("restore");
    b.write(
        "m1",
        "the third era",
        "layers/eras/third.md",
        "rewritten\n",
        &s,
    )
    .unwrap();
    let saved: Benches = serde_json::from_value(serde_json::to_value(&b).unwrap()).unwrap();

    let (mut fresh, fresh_root, fs) = rooted("restore-fresh");
    fresh.restore(saved);
    assert_eq!(
        fresh
            .files()
            .and_then(|f| f.folder_root().map(|p| p.to_path_buf())),
        Some(fresh_root.clone())
    );
    assert_eq!(
        fresh.read("m1", "layers/eras/third.md", &fs).unwrap(),
        "rewritten\n"
    );
    assert_eq!(
        fresh.opened("m1").unwrap().changed(),
        [&"layers/eras/third.md".to_string()]
    );
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(fresh_root);
}

#[test]
fn a_world_with_no_root_has_no_documents_and_says_so() {
    let mut b = Benches::new();
    let s = Folder(std::env::temp_dir());
    assert!(!b.has_root());
    let err = b.read("m1", "layers/eras/third.md", &s).unwrap_err();
    assert!(err.contains("no documents"), "{err}");
    assert!(b.write("m1", "x", "layers/eras/third.md", "…", &s).is_err());
    assert!(b.list("m1", "layers/eras", &s).is_err());
}

/// **With no operation there is nowhere to commit** (§10.3): the published
/// mind refuses the commit, the working set stays, and nothing moves.
#[test]
fn the_published_mind_takes_no_commit() {
    let (mut b, root, _) = rooted("published");
    let published = Published::folder(root.clone());
    b.write("m1", "x", "layers/eras/third.md", "an edit\n", &published)
        .unwrap();
    assert_eq!(b.commit("m1", &published), Err(NO_OPERATION.to_string()));
    assert_eq!(b.diff("m1"), ["layers/eras/third.md (changed)"]);
    assert_eq!(
        std::fs::read_to_string(root.join("layers/eras/third.md")).unwrap(),
        "the third era\nburned in the spring\n"
    );
}

/// **Nobody collides with themselves, nor with the engine's tidying.** A
/// body's own earlier commit, or a rebased clerical rewrite, is not a
/// change made under it; somebody else's commit still is.
#[test]
fn only_somebody_elses_commit_is_a_collision() {
    let (mut b, root, s) = rooted("own-commit");
    let path = "layers/eras/third.md";
    b.write("m1", "x", path, "first pass\n", &s).unwrap();
    b.commit("m1", &s).unwrap();
    // It opens again, and its own earlier hand changes the file under it.
    b.write("m1", "x", path, "second pass\n", &s).unwrap();
    std::fs::write(root.join(path), "first pass, tidied\n").unwrap();
    assert!(b.commit("m1", &s).is_ok(), "its own hand is no collision");

    // The engine tidies under somebody else's open copy: rebased, it commits.
    b.write("m2", "x", path, "a mend\n", &s).unwrap();
    std::fs::write(root.join(path), "second pass, tidied\n").unwrap();
    b.rebase(path, &s);
    assert!(b.commit("m2", &s).is_ok());

    // Somebody else's commit under an open copy is still a collision.
    b.write("m3", "x", path, "late\n", &s).unwrap();
    b.write("m1", "x", path, "first in\n", &s).unwrap();
    b.commit("m1", &s).unwrap();
    let e = b.commit("m3", &s).unwrap_err();
    assert!(e.contains("m1 committed over the same ground"), "{e}");
}

#[test]
fn a_read_falls_through_to_the_shelf() {
    let (b, _, s) = rooted("read");
    let got = b.read("m1", "layers/eras/third.md", &s).unwrap();
    assert!(got.starts_with("the third era"), "{got}");
}

/// The whole point of the module: a write is invisible on the shelf and
/// visible to its own author.
#[test]
fn a_write_stays_in_memory_until_it_is_committed() {
    let (mut b, root, s) = rooted("memory");
    b.write(
        "m1",
        "the third era",
        "layers/eras/third.md",
        "rewritten\n",
        &s,
    )
    .unwrap();

    assert_eq!(
        b.read("m1", "layers/eras/third.md", &s).unwrap(),
        "rewritten\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("layers/eras/third.md")).unwrap(),
        "the third era\nburned in the spring\n",
        "the shelf moved before the commit"
    );
    // And invisible to anybody else, which is what makes it *yours*.
    assert!(b
        .read("m2", "layers/eras/third.md", &s)
        .unwrap()
        .starts_with("the third era"));

    b.commit("m1", &s).unwrap();
    assert_eq!(
        std::fs::read_to_string(root.join("layers/eras/third.md")).unwrap(),
        "rewritten\n"
    );
}

#[test]
fn committing_clears_the_working_set_and_names_what_it_wrote() {
    let (mut b, _, s) = rooted("clears");
    b.write("m1", "layers/eras", "layers/eras/third.md", "a\n", &s)
        .unwrap();
    b.write("m1", "layers/eras", "layers/eras/fourth.md", "b\n", &s)
        .unwrap();
    let written = b.commit("m1", &s).unwrap();
    assert_eq!(
        written,
        vec!["layers/eras/fourth.md", "layers/eras/third.md"]
    );
    assert!(b.opened("m1").is_none());
    assert!(b.diff("m1").is_empty());
}

#[test]
fn a_new_document_is_committed() {
    let (mut b, root, s) = rooted("create");
    b.write(
        "m1",
        "a silence",
        "layers/stories/deep/new.md",
        "a night at the gate\n",
        &s,
    )
    .unwrap();
    b.commit("m1", &s).unwrap();
    assert_eq!(
        std::fs::read_to_string(root.join("layers/stories/deep/new.md")).unwrap(),
        "a night at the gate\n"
    );
}

#[test]
fn removing_a_document_takes_it_off_the_shelf_at_the_commit() {
    let (mut b, root, s) = rooted("remove");
    b.remove("m1", "layers/eras", "layers/eras/fourth.md", &s)
        .unwrap();
    assert!(
        root.join("layers/eras/fourth.md").exists(),
        "gone before the commit"
    );
    assert!(b
        .read("m1", "layers/eras/fourth.md", &s)
        .unwrap_err()
        .contains("taken"));
    b.commit("m1", &s).unwrap();
    assert!(!root.join("layers/eras/fourth.md").exists());
}

/// Something not there is refused, naming what its folder holds.
#[test]
fn removing_something_that_is_not_there_is_refused() {
    let (mut b, _, s) = rooted("remove-absent");
    let err = b.remove("m1", "x", "layers/eras/never.md", &s).unwrap_err();
    assert_eq!(
        err,
        "There is no document at layers/eras/never.md. layers/eras holds: \
         layers/eras/fourth.md, layers/eras/third.md."
    );
}

/// An edit of a document not written yet says how one is written: an edit
/// changes what is there.
#[test]
fn editing_a_document_not_written_yet_says_how_one_is_written() {
    let (mut b, _, s) = rooted("edit-absent");
    let err = b
        .edit(
            "m1",
            "x",
            "layers/eras/never.md",
            &change("0tiT".into(), "x", "y"),
            &s,
        )
        .unwrap_err();
    assert!(
        err.starts_with("There is no document at layers/eras/never.md."),
        "{err}"
    );
    assert!(err.contains("written whole with `compose`"), "{err}");
}

// ── anchored edits ──────────────────────────────────────────────────────

/// The anchor of paragraph `i` of a document as `body` has it.
fn anchor_in(b: &Benches, body: &str, path: &str, i: usize, s: &dyn Shelf) -> String {
    let text = b.read(body, path, s).unwrap();
    Doc::new(path, &text).units()[i].anchor.clone()
}

/// `old` inside the anchored paragraph, replaced by `text`.
fn change(at: String, old: &str, text: &str) -> Edit {
    Edit {
        at,
        op: Op::Replace {
            text: text.into(),
            old: Some(old.into()),
            to: None,
        },
    }
}

#[test]
fn an_edit_changes_the_paragraph_its_anchor_names() {
    let (mut b, _, s) = rooted("edit-one");
    let at = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    let reply = b
        .edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            &change(at.clone(), "spring", "autumn"),
            &s,
        )
        .unwrap();
    assert_eq!(
        b.read("m1", "layers/eras/third.md", &s).unwrap(),
        "the third era\nburned in the autumn\n"
    );
    let now = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    assert_eq!(
        reply,
        format!(
            "layers/eras/third.md: Changed {at}.\n{now}| the third era\nburned in the autumn\n"
        )
    );
}

/// An ambiguous edit is the one that silently does the wrong thing, so it
/// is refused and says how many it found — and nothing lands.
#[test]
fn words_that_appear_twice_in_the_paragraph_are_refused_and_counted() {
    let (mut b, root, s) = rooted("edit-two");
    std::fs::write(root.join("layers/eras/third.md"), "a fire\nand a fire\n").unwrap();
    let at = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    let err = b
        .edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            &change(at, "a fire", "a flood"),
            &s,
        )
        .unwrap_err();
    assert!(err.contains("2 times"), "{err}");
    assert!(
        b.opened("m1").is_none_or(|w| w.is_empty()),
        "a refused edit still landed"
    );
}

/// **Words not in the paragraph are refused, and the paragraph is shown** —
/// the text the next edit can take its words from, as it stands in this
/// working set.
#[test]
fn words_not_in_the_paragraph_are_refused_and_it_is_shown() {
    let (mut b, _, s) = rooted("edit-none");
    let at = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    let err = b
        .edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            &change(at.clone(), "a flood", "a fire"),
            &s,
        )
        .unwrap_err();
    assert_eq!(
        err,
        format!(
            "In layers/eras/third.md, `old` does not occur in {at}, which reads:\nthe third \
             era\nburned in the spring."
        )
    );
}

/// An anchor the document does not have is refused with the way on: read it
/// again.
#[test]
fn an_anchor_the_document_does_not_have_is_refused() {
    let (mut b, _, s) = rooted("edit-stale");
    let err = b
        .edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            &change("zzzz".into(), "spring", "autumn"),
            &s,
        )
        .unwrap_err();
    assert!(
        err.starts_with("In layers/eras/third.md, zzzz is not a part of it"),
        "{err}"
    );
    assert!(err.contains("read it again"), "{err}");
}

/// Edits compound in memory: the second reads the first's result, not the
/// shelf's — and the second has to name the paragraph as the first left it.
#[test]
fn a_second_edit_sees_the_first() {
    let (mut b, _, s) = rooted("edit-compound");
    let at = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    b.edit(
        "m1",
        "layers/eras",
        "layers/eras/third.md",
        &change(at.clone(), "third", "thirteenth"),
        &s,
    )
    .unwrap();
    assert!(b
        .edit(
            "m1",
            "layers/eras",
            "layers/eras/third.md",
            &change(at, "spring", "autumn"),
            &s,
        )
        .unwrap_err()
        .contains("read it again"));
    let now = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    b.edit(
        "m1",
        "layers/eras",
        "layers/eras/third.md",
        &change(now, "spring", "autumn"),
        &s,
    )
    .unwrap();
    assert_eq!(
        b.read("m1", "layers/eras/third.md", &s).unwrap(),
        "the thirteenth era\nburned in the autumn\n"
    );
}

// ── collisions ──────────────────────────────────────────────────────────

/// The act somebody else's work can refuse, and the refusal names them.
#[test]
fn a_commit_over_somebody_elses_commit_is_refused_and_names_them() {
    let (mut b, _, s) = rooted("collide");
    b.write("m1", "layers/eras", "layers/eras/third.md", "mine\n", &s)
        .unwrap();
    b.write(
        "m2",
        "layers/eras",
        "layers/eras/third.md",
        "mine too\n",
        &s,
    )
    .unwrap();

    assert_eq!(b.commit("m1", &s).unwrap(), vec!["layers/eras/third.md"]);
    let err = b.commit("m2", &s).unwrap_err();
    assert!(err.contains("m1"), "the other party was not named: {err}");
    assert!(
        b.opened("m2").is_some(),
        "a refused commit threw the work away"
    );
}

/// A refusal on one document leaves the others alone — the character still
/// has every change it made.
#[test]
fn a_collision_writes_none_of_the_batch() {
    let (mut b, root, s) = rooted("collide-batch");
    b.write("m1", "layers/eras", "layers/eras/fourth.md", "safe\n", &s)
        .unwrap();
    b.write(
        "m1",
        "layers/eras",
        "layers/eras/third.md",
        "contested\n",
        &s,
    )
    .unwrap();
    // Somebody else moves one of the two under it.
    std::fs::write(root.join("layers/eras/third.md"), "theirs\n").unwrap();

    assert!(b.commit("m1", &s).is_err());
    assert_eq!(
        std::fs::read_to_string(root.join("layers/eras/fourth.md")).unwrap(),
        "the fourth era\n",
        "the uncontested half was written anyway"
    );
    assert_eq!(b.diff("m1").len(), 2, "the working set did not survive");
}

/// Two bodies on different documents are not each other's business.
#[test]
fn two_bodies_on_different_documents_both_commit() {
    let (mut b, _, s) = rooted("parallel");
    b.write("m1", "layers/eras", "layers/eras/third.md", "a\n", &s)
        .unwrap();
    b.write("m2", "layers/eras", "layers/eras/fourth.md", "b\n", &s)
        .unwrap();
    assert!(b.commit("m1", &s).is_ok());
    assert!(b.commit("m2", &s).is_ok(), "an unrelated document collided");
}

/// A write that says exactly what the file already said is not a change,
/// and must not manufacture a collision for somebody else.
#[test]
fn writing_what_is_already_there_changes_nothing() {
    let (mut b, _, s) = rooted("noop");
    b.write(
        "m1",
        "layers/eras",
        "layers/eras/fourth.md",
        "the fourth era\n",
        &s,
    )
    .unwrap();
    assert!(b.diff("m1").is_empty());
    assert!(b.commit("m1", &s).unwrap().is_empty());
}

// ── the working loop ────────────────────────────────────────────────────

#[test]
fn setting_aside_keeps_the_work_and_picking_it_up_restores_it() {
    let (mut b, _, s) = rooted("stash");
    b.write(
        "m1",
        "the third era",
        "layers/eras/third.md",
        "half done\n",
        &s,
    )
    .unwrap();
    assert!(b.stash("m1"));
    assert!(b.diff("m1").is_empty(), "set-aside work is still open");
    assert!(b
        .read("m1", "layers/eras/third.md", &s)
        .unwrap()
        .starts_with("the third era"));

    assert_eq!(b.pop("m1").unwrap(), "the third era");
    assert_eq!(
        b.read("m1", "layers/eras/third.md", &s).unwrap(),
        "half done\n"
    );
}

#[test]
fn picking_up_set_aside_work_over_live_changes_is_refused() {
    let (mut b, _, s) = rooted("stash-bury");
    b.write("m1", "the third era", "layers/eras/third.md", "first\n", &s)
        .unwrap();
    b.stash("m1");
    b.write(
        "m1",
        "the fourth era",
        "layers/eras/fourth.md",
        "second\n",
        &s,
    )
    .unwrap();
    let err = b.pop("m1").unwrap_err();
    assert!(err.contains("bury"), "{err}");
    assert_eq!(
        b.read("m1", "layers/eras/fourth.md", &s).unwrap(),
        "second\n"
    );
}

#[test]
fn throwing_the_work_away_puts_the_document_back() {
    let (mut b, _, s) = rooted("discard");
    b.write(
        "m1",
        "layers/eras",
        "layers/eras/third.md",
        "wrong from the start\n",
        &s,
    )
    .unwrap();
    assert!(b.discard("m1"));
    assert!(!b.discard("m1"), "threw away twice");
    assert!(b
        .read("m1", "layers/eras/third.md", &s)
        .unwrap()
        .starts_with("the third era"));
}

#[test]
fn opening_a_second_thing_while_one_is_open_is_refused() {
    let (mut b, _, _) = rooted("open-twice");
    assert!(b.open_on("m1", "the third era").unwrap());
    assert!(
        !b.open_on("m1", "the third era").unwrap(),
        "opened it twice"
    );
    let err = b.open_on("m1", "the fourth era").unwrap_err();
    assert!(err.contains("the third era"), "{err}");
}

#[test]
fn offering_and_withdrawing_only_work_on_something_open() {
    let (mut b, _, _) = rooted("offer");
    assert!(!b.set_offered("m1", true));
    b.open_on("m1", "the third era").unwrap();
    assert!(b.set_offered("m1", true));
    assert!(b.opened("m1").unwrap().offered);
    assert!(b.set_offered("m1", false));
    assert!(!b.opened("m1").unwrap().offered);
}

#[test]
fn a_diff_names_what_kind_of_change_each_one_is() {
    let (mut b, _, s) = rooted("diff");
    b.write("m1", "layers/eras", "layers/eras/third.md", "changed\n", &s)
        .unwrap();
    b.write("m1", "layers/eras", "layers/eras/new.md", "made\n", &s)
        .unwrap();
    b.remove("m1", "layers/eras", "layers/eras/fourth.md", &s)
        .unwrap();
    assert_eq!(
        b.diff("m1"),
        vec![
            "layers/eras/fourth.md (gone)",
            "layers/eras/new.md (new)",
            "layers/eras/third.md (changed)",
        ]
    );
}

// ── listing ─────────────────────────────────────────────────────────────

#[test]
fn a_listing_shows_the_shelf_plus_your_own_uncommitted_work() {
    let (mut b, _, s) = rooted("list");
    assert_eq!(
        b.list("m1", "layers/eras", &s).unwrap(),
        vec!["fourth.md", "third.md"]
    );
    // The root lists the areas that can be worked in, not the mind's own
    // configuration sitting beside them.
    assert_eq!(
        b.list("m1", "", &s).unwrap(),
        EDITABLE_AREAS
            .iter()
            .map(|a| format!("{a}/"))
            .collect::<Vec<_>>()
    );

    b.write("m1", "layers/eras", "layers/eras/fifth.md", "new\n", &s)
        .unwrap();
    b.remove("m1", "layers/eras", "layers/eras/fourth.md", &s)
        .unwrap();
    assert_eq!(
        b.list("m1", "layers/eras", &s).unwrap(),
        vec!["fifth.md", "third.md"]
    );
    // …and nobody else's view moved.
    assert_eq!(
        b.list("m2", "layers/eras", &s).unwrap(),
        vec!["fourth.md", "third.md"]
    );
}

/// A listing of one directory does not leak the ones below it.
#[test]
fn a_listing_stops_at_its_own_directory() {
    let (mut b, _, s) = rooted("list-depth");
    b.write("m1", "x", "layers/eras/deep/buried.md", "…", &s)
        .unwrap();
    assert!(!b
        .list("m1", "layers/eras", &s)
        .unwrap()
        .contains(&"buried.md".to_string()));
    assert_eq!(
        b.list("m1", "layers/eras/deep", &s).unwrap(),
        vec!["buried.md"]
    );
}

// ── the path guard ──────────────────────────────────────────────────────

/// The paths here are written by a language model, so every spelling of
/// "leave the directory" has to be refused on the way in.
#[test]
fn a_path_that_leaves_the_root_is_refused_by_every_act() {
    let (mut b, root, s) = rooted("escape");
    for bad in [
        "../secrets.md",
        "layers/eras/../../secrets.md",
        "c:/windows/system.md",
        r"..\..\secrets.md",
        "layers/eras/a\0b.md",
    ] {
        assert!(b.read("m1", bad, &s).is_err(), "read {bad}");
        assert!(b.write("m1", "x", bad, "owned", &s).is_err(), "write {bad}");
        assert!(b.remove("m1", "x", bad, &s).is_err(), "remove {bad}");
    }
    assert!(!root.parent().unwrap().join("secrets.md").exists());
}

/// The mind holds `.md` and `.yaml`. Anything else is not a document a
/// bench writes, and letting one through would put an executable into a
/// directory the daemon reads.
#[test]
fn only_text_documents_can_be_written() {
    let (mut b, _, s) = rooted("ext");
    for bad in ["layers/eras/x.exe", "layers/eras/x.png", "layers/eras/x"] {
        assert!(b.write("m1", "x", bad, "…", &s).is_err(), "{bad}");
    }
    assert!(b
        .write("m1", "x", "layers/eras/x.yaml", "a: 1\n", &s)
        .is_ok());
}

// ── links ───────────────────────────────────────────────────────────────

/// Make a symlink, or `false` where the OS will not allow one.
///
/// Windows needs Developer Mode or an elevated process, so the tests below
/// say out loud when the link half did not run. A silent skip that reads as
/// a pass is worse than no test at all.
fn try_symlink(target: &Path, link: &Path) -> bool {
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link).is_ok()
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).is_ok()
    }
}

/// Make a directory link, by the one route Windows allows unprivileged.
///
/// `mklink /J` makes a junction without Developer Mode, where every symlink
/// API needs it. Shelling out in a test is worth it here because this is
/// the *only* redirection reachable on the platform the daemon runs on, and
/// a guard whose reachable case is untested is a guard nobody has checked.
fn try_dir_link(target: &Path, link: &Path) -> bool {
    #[cfg(windows)]
    {
        if std::os::windows::fs::symlink_dir(target, link).is_ok() {
            return true;
        }
        std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).is_ok()
    }
}

/// **The case that actually runs on Windows.** A junction redirects a whole
/// directory, so a listing walked through one would report documents from
/// somewhere the character never named: it lists as nothing.
#[test]
fn a_directory_link_is_not_walked() {
    let (b, root, s) = rooted("junction");
    if !try_dir_link(&root.join("layers/eras"), &root.join("layers/alias")) {
        eprintln!("SKIPPED: this OS would not create a directory link");
        return;
    }
    assert!(
        b.list("m1", "layers/alias", &s).unwrap().is_empty(),
        "a listing walked through a junction"
    );
    // The real directory is still readable — the guard is about the link,
    // not about the documents behind it.
    assert_eq!(
        b.list("m1", "layers/eras", &s).unwrap(),
        vec!["fourth.md", "third.md"]
    );
}

/// A link pointing **out** of the root is caught by containment — resolving
/// it lands outside and the path is refused.
#[test]
fn a_link_out_of_the_root_cannot_be_read_through() {
    let (b, root, s) = rooted("link-out");
    let outside = root
        .parent()
        .unwrap()
        .join(format!("npcd-bench-outside-{}.md", std::process::id()));
    std::fs::write(&outside, "not yours\n").unwrap();
    if !try_symlink(&outside, &root.join("layers/eras/out.md")) {
        eprintln!("SKIPPED the link half: this OS would not create a symlink");
        return;
    }
    let got = b.read("m1", "layers/eras/out.md", &s);
    assert!(got.is_err(), "read straight through a link out of the root");
    assert!(
        !got.unwrap_or_default().contains("not yours"),
        "content from outside the root came back"
    );
}

/// A link pointing **inside** the root resolves cleanly, so containment
/// says nothing about it — and it would still redirect a read to a document
/// the character did not name. This is the case the leaf check is for.
#[test]
fn a_link_inside_the_root_is_not_followed() {
    let (mut b, root, s) = rooted("link-in");
    if !try_symlink(
        &root.join("layers/eras/third.md"),
        &root.join("layers/eras/alias.md"),
    ) {
        eprintln!("SKIPPED the link half: this OS would not create a symlink");
        return;
    }
    assert!(b.read("m1", "layers/eras/alias.md", &s).is_err());
    assert!(b.remove("m1", "x", "layers/eras/alias.md", &s).is_err());
    // …and the document it pointed at is untouched.
    assert!(b
        .read("m1", "layers/eras/third.md", &s)
        .unwrap()
        .starts_with("the third era"));
}

// ── the read cap ────────────────────────────────────────────────────────

/// A document of `n` numbered lines, for exercising the cap.
fn long_document(root: &Path, lines: usize) {
    let body: String = (1..=lines).map(|i| format!("line {i}\n")).collect();
    std::fs::write(root.join("layers/eras/long.md"), body).unwrap();
}

/// **An uncapped read is a context hazard.** A tool result lands in the
/// conversation verbatim, so one long document would fill a character's
/// whole window and push out the turns that gave it a reason to read. A
/// 900-line paragraph is cut into parts of 80 lines, two to a 200-line page.
#[test]
fn a_read_returns_at_most_one_page() {
    let (b, root, s) = rooted("cap");
    long_document(&root, 900);
    let out = b.excerpt("m1", "layers/eras/long.md", 1, &s).unwrap();
    assert!(
        out.starts_with("\nlayers/eras/long.md (page 1 of 6: paragraphs 1–2 of 12):"),
        "{out}"
    );
    assert!(out.contains("| line 1\n"), "{out}");
    assert!(out.contains("line 160\n"), "{out}");
    assert!(!out.contains("line 161"), "the cap did not hold");
    assert!(
        out.ends_with("There is more: the next read asks for page 2.\n"),
        "{out}"
    );
}

/// The header names the page and the paragraphs on it, so the next read asks
/// for the next page directly rather than having to work it out.
#[test]
fn a_read_continues_from_the_page_it_is_given() {
    let (b, root, s) = rooted("cap-continue");
    long_document(&root, 900);
    let out = b.excerpt("m1", "layers/eras/long.md", 2, &s).unwrap();
    assert!(out.contains("(page 2 of 6: paragraphs 3–4 of 12)"), "{out}");
    assert!(out.contains("| line 161\n"), "{out}");
    assert!(!out.contains("line 160\n"), "{out}");
}

/// The last page says nothing of more, which is how a reader knows it has
/// the end and not another page waiting.
#[test]
fn the_last_page_says_it_is_the_last() {
    let (b, root, s) = rooted("cap-last");
    long_document(&root, 250);
    let out = b.excerpt("m1", "layers/eras/long.md", 2, &s).unwrap();
    assert!(out.contains("(page 2 of 2: paragraphs 3–4 of 4)"), "{out}");
    assert!(!out.contains("There is more"), "{out}");
}

/// A page past the end reads the last page rather than coming back empty,
/// which a model reads as "the document is empty" and then acts on.
#[test]
fn a_page_past_the_end_reads_the_end() {
    let (b, root, s) = rooted("cap-past");
    long_document(&root, 10);
    let out = b.excerpt("m1", "layers/eras/long.md", 9_999, &s).unwrap();
    assert!(out.contains("line 10"), "{out}");
}

#[test]
fn a_short_document_comes_back_whole_and_anchored() {
    let (b, _, s) = rooted("excerpt");
    let out = b.excerpt("m1", "layers/eras/third.md", 1, &s).unwrap();
    let a = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    assert_eq!(
        out,
        format!(
            "\nlayers/eras/third.md (1 paragraph):\n\n```markdown\n{a}| the third era\nburned \
             in the spring\n```\n"
        )
    );
}

#[test]
fn an_empty_document_says_so_rather_than_naming_a_range() {
    let (b, root, s) = rooted("excerpt-empty");
    std::fs::write(root.join("layers/eras/blank.md"), "").unwrap();
    let out = b.excerpt("m1", "layers/eras/blank.md", 1, &s).unwrap();
    assert!(out.contains("(empty)"), "{out}");
    assert!(
        !out.contains("lines 1-0"),
        "a range that reads as a bug: {out}"
    );
}

/// The excerpt shows the reader's own uncommitted work, like a plain read.
#[test]
fn an_excerpt_shows_your_own_changes() {
    let (mut b, _, s) = rooted("excerpt-mine");
    b.write("m1", "x", "layers/eras/third.md", "one\ntwo\nthree\n", &s)
        .unwrap();
    let mine = b.excerpt("m1", "layers/eras/third.md", 1, &s).unwrap();
    assert!(mine.contains("| one\ntwo\nthree\n"), "{mine}");
    let theirs = b.excerpt("m2", "layers/eras/third.md", 1, &s).unwrap();
    assert!(theirs.contains("burned in the spring"), "{theirs}");
}

/// A paragraph's lines stand as they are under its anchor — indentation and
/// all — so what a read shows is the text an edit changes.
#[test]
fn a_paragraph_shows_its_lines_as_they_are() {
    let (b, root, s) = rooted("lines");
    std::fs::write(
        root.join("layers/eras/long.md"),
        "first\n  indented\n\nsecond\n",
    )
    .unwrap();
    let out = b.excerpt("m1", "layers/eras/long.md", 1, &s).unwrap();
    assert!(out.contains("| first\n  indented\n"), "{out}");
    assert!(out.contains("| second\n"), "{out}");
}

/// An edit is applied to the document, not its rendering — the anchors a
/// read shows never land in the text.
#[test]
fn an_edit_is_applied_to_the_document_and_not_its_rendering() {
    let (mut b, _, s) = rooted("raw");
    let at = anchor_in(&b, "m1", "layers/eras/third.md", 0, &s);
    b.edit(
        "m1",
        "x",
        "layers/eras/third.md",
        &change(at, "the third era", "the fourth era"),
        &s,
    )
    .unwrap();
    assert!(b
        .read("m1", "layers/eras/third.md", &s)
        .unwrap()
        .starts_with("the fourth era"));
}

// ── isolation ───────────────────────────────────────────────────────────

/// **No uncommitted change is visible to anybody but the body that made
/// it.** Every mutating act, run by two bodies over the same documents at
/// once, with each one's view asserted after every step.
///
/// The workspace is keyed by body and every entry point takes one, so this
/// holds by construction — but "by construction" is exactly the claim that
/// stops being true the first time somebody adds a method and forgets the
/// argument, and the failure would be one character reading another's
/// half-finished work as though it were canon.
#[test]
fn nothing_uncommitted_leaks_between_bodies() {
    let (mut b, root, s) = rooted("isolation");
    std::fs::write(root.join("layers/eras/shared.md"), "as it stands\n").unwrap();

    // Both open work on the same thing, and both change the same document.
    assert!(b.open_on("m1", "the third era").unwrap());
    assert!(b.open_on("m2", "the third era").unwrap());
    b.write(
        "m1",
        "the third era",
        "layers/eras/shared.md",
        "m1's version\n",
        &s,
    )
    .unwrap();
    b.write(
        "m2",
        "the third era",
        "layers/eras/shared.md",
        "m2's version\n",
        &s,
    )
    .unwrap();

    assert_eq!(
        b.read("m1", "layers/eras/shared.md", &s).unwrap(),
        "m1's version\n"
    );
    assert_eq!(
        b.read("m2", "layers/eras/shared.md", &s).unwrap(),
        "m2's version\n"
    );
    assert_eq!(
        b.read("m3", "layers/eras/shared.md", &s).unwrap(),
        "as it stands\n",
        "an uninvolved body saw somebody's draft"
    );

    // An edit compounds only on its own author's version.
    let at = anchor_in(&b, "m1", "layers/eras/shared.md", 0, &s);
    b.edit(
        "m1",
        "x",
        "layers/eras/shared.md",
        &change(at, "m1's", "m1 has edited its"),
        &s,
    )
    .unwrap();
    assert_eq!(
        b.read("m2", "layers/eras/shared.md", &s).unwrap(),
        "m2's version\n"
    );

    // A creation is invisible, and so is a removal.
    b.write("m1", "x", "layers/eras/mine.md", "only m1 has this\n", &s)
        .unwrap();
    b.remove("m2", "x", "layers/eras/fourth.md", &s).unwrap();
    assert!(
        b.read("m2", "layers/eras/mine.md", &s).is_err(),
        "a new document leaked"
    );
    assert!(
        b.read("m1", "layers/eras/fourth.md", &s).is_ok(),
        "a removal leaked"
    );
    assert!(b
        .list("m2", "layers/eras", &s)
        .unwrap()
        .iter()
        .all(|n| n != "mine.md"));
    assert!(b
        .list("m1", "layers/eras", &s)
        .unwrap()
        .iter()
        .any(|n| n == "fourth.md"));

    // Diffs, excerpts and offers are each their own body's.
    assert!(b.diff("m1").iter().any(|d| d.contains("mine.md")));
    assert!(b.diff("m2").iter().all(|d| !d.contains("mine.md")));
    assert!(b
        .excerpt("m1", "layers/eras/shared.md", 1, &s)
        .unwrap()
        .contains("m1 has edited"));
    assert!(b
        .excerpt("m2", "layers/eras/shared.md", 1, &s)
        .unwrap()
        .contains("m2's version"));
    b.set_offered("m1", true);
    assert!(b.opened("m1").unwrap().offered);
    assert!(!b.opened("m2").unwrap().offered, "an offer leaked");

    // Setting aside empties only one bench, and picking it up fills only it.
    b.stash("m1");
    assert!(b.diff("m1").is_empty());
    assert!(!b.diff("m2").is_empty(), "the other bench was cleared too");
    b.pop("m1").unwrap();
    assert!(b.diff("m1").iter().any(|d| d.contains("mine.md")));

    // Throwing one away leaves the other standing.
    b.discard("m2");
    assert!(b.diff("m2").is_empty());
    assert!(
        !b.diff("m1").is_empty(),
        "one body's discard took another's work"
    );

    // And a commit records exactly one body's changes.
    b.commit("m1", &s).unwrap();
    assert_eq!(
        std::fs::read_to_string(root.join("layers/eras/shared.md")).unwrap(),
        "m1 has edited its version\n"
    );
    assert!(root.join("layers/eras/mine.md").exists());
    assert!(
        root.join("layers/eras/fourth.md").exists(),
        "a discarded removal reached the shelf"
    );
}

/// The same, for the YAML field acts — they go through the working set too,
/// so a mood somebody is part-way through changing is not what anybody else
/// reads.
#[test]
fn a_half_changed_field_is_not_what_anybody_else_reads() {
    let (mut b, root, s) = rooted("isolation-field");
    std::fs::create_dir_all(root.join("moods")).unwrap();
    std::fs::write(
        root.join("moods/undone.yaml"),
        "id: undone\n# why it reads this way\ndescription: As it stands.\n",
    )
    .unwrap();

    b.write_field(
        "m1",
        "x",
        "moods/undone.yaml",
        &["description"],
        "m1's wording.",
        &s,
    )
    .unwrap();
    assert_eq!(
        b.read_field("m1", "moods/undone.yaml", &["description"], &s)
            .unwrap(),
        "m1's wording."
    );
    assert_eq!(
        b.read_field("m2", "moods/undone.yaml", &["description"], &s)
            .unwrap(),
        "As it stands.",
        "an uncommitted field change leaked"
    );
    assert!(std::fs::read_to_string(root.join("moods/undone.yaml"))
        .unwrap()
        .contains("As it stands."));
}

// ── determinism ─────────────────────────────────────────────────────────

/// The property the tool tests rest on: the same acts leave the same state.
#[test]
fn the_same_acts_leave_byte_identical_state() {
    let build = |name: &str| {
        let (mut b, _, s) = rooted(name);
        b.open_on("m1", "the third era").unwrap();
        b.write("m1", "the third era", "layers/eras/third.md", "a\n", &s)
            .unwrap();
        b.write("m1", "the third era", "layers/eras/new.md", "b\n", &s)
            .unwrap();
        b.remove("m1", "the third era", "layers/eras/fourth.md", &s)
            .unwrap();
        b.set_offered("m1", true);
        b
    };
    let (one, two) = (build("det-a"), build("det-b"));
    // The roots differ by design; everything the acts produced does not.
    assert_eq!(one.open, two.open);
    assert_eq!(one.diff("m1"), two.diff("m1"));
    assert_eq!(
        serde_json::to_string(&one.open).unwrap(),
        serde_json::to_string(&two.open).unwrap()
    );
}
