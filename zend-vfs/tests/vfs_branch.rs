//! A conversation's store over a git repository: read through its branch,
//! never its folder, at a base the conversation moves itself.

mod support;

use std::path::Path;
use std::sync::Arc;

use regex::Regex;
use support::{branch, git, put};
use tempfile::TempDir;
use zend_vfs::vfs::{Carried, Resolved};
use zend_vfs::work::{keep_ours, ThreeWay};
use zend_vfs::{
    Base, FileState, GitSource, MergeLabels, Oid, Repo, Rev, Snapshot, VfsError, VfsStore,
};

const TEN: &str = "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n";

/// A repository on `main`: `ten.txt`, `src/lib.rs`, a hidden `.github/ci.yml`
/// and a `secrets/key` no store may read — and a branch `other` with
/// `ten.txt` changed.
fn repo() -> (TempDir, Arc<GitSource>) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    git(root, &["init", "-q", "--template="]);
    git(root, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    git(root, &["config", "core.autocrlf", "false"]);
    put(root, "ten.txt", TEN.as_bytes());
    put(root, "src/lib.rs", b"pub fn lib() {}\n");
    put(root, ".github/ci.yml", b"ci\n");
    put(root, "secrets/key", b"hunter2\n");
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", "base"]);
    git(root, &["checkout", "-q", "-b", "other"]);
    put(root, "ten.txt", TEN.replace("l10\n", "ten\n").as_bytes());
    git(root, &["commit", "-q", "-am", "other"]);
    git(root, &["checkout", "-q", "main"]);
    let source = GitSource::open(root).unwrap();
    (dir, source)
}

fn on_main(source: &Arc<GitSource>) -> VfsStore {
    VfsStore::on_branch(Arc::clone(source), Rev::Branch(branch("main")))
}

/// Commit `path` as `content` on `main`, as a git tool would: objects and a
/// ref, the folder untouched.
fn commit_on_main(root: &Path, path: &str, content: &str) {
    let blob = {
        put(root, ".blob", content.as_bytes());
        let oid = git(root, &["hash-object", "-w", ".blob"]);
        std::fs::remove_file(root.join(".blob")).unwrap();
        oid.trim().to_string()
    };
    let index = root.join(".git").join("zend-test-index");
    let with_index = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .env("GIT_INDEX_FILE", &index)
            .env("GIT_AUTHOR_NAME", "T")
            .env("GIT_AUTHOR_EMAIL", "t@example.com")
            .env("GIT_COMMITTER_NAME", "T")
            .env("GIT_COMMITTER_EMAIL", "t@example.com")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    };
    with_index(&["read-tree", "main"]);
    with_index(&[
        "update-index",
        "--add",
        "--cacheinfo",
        &format!("100644,{blob},{path}"),
    ]);
    let tree = with_index(&["write-tree"]);
    let commit = with_index(&["commit-tree", &tree, "-p", "main", "-m", "tool"]);
    git(root, &["update-ref", "refs/heads/main", &commit]);
    let _ = std::fs::remove_file(index);
}

/// **What the branch committed is what the store reads — never the folder**:
/// a file changed on disk, a file only on disk, reads, listings, searches.
#[test]
fn the_branch_is_read_never_the_folder() {
    let (dir, source) = repo();
    put(dir.path(), "ten.txt", b"a job left this\n");
    put(dir.path(), "made-by-a-job.txt", b"x\n");
    let store = on_main(&source);
    assert_eq!(store.read("ten.txt").unwrap().as_deref(), Some(TEN));
    assert_eq!(store.read("made-by-a-job.txt").unwrap(), None);
    assert_eq!(store.paths(""), ["src/lib.rs", "ten.txt"]);
    let listed: Vec<String> = store
        .list_dir("")
        .unwrap()
        .unwrap()
        .into_iter()
        .map(|e| e.path)
        .collect();
    assert_eq!(listed, ["src", "ten.txt"], "hidden and protected left out");
    assert_eq!(
        store.read(".github/ci.yml").unwrap().as_deref(),
        Some("ci\n")
    );
    assert!(store.read("secrets/key").is_err());
    let hits = store.grep(&Regex::new("l1").unwrap(), "", 10, 10);
    assert_eq!(hits.hits.len(), 2, "l1 and l10");
    let page = store.read_page("ten.txt", 0).unwrap().unwrap();
    assert_eq!((page.total_lines, page.end_line), (10, 10));
    assert_eq!(
        store.read_bytes("src/lib.rs").unwrap().unwrap(),
        b"pub fn lib() {}\n"
    );
}

/// Where `rev` stands in the repository at `root`, as a store's base.
fn base_of(root: &Path, rev: &str) -> Base {
    let id = |spec: String| Oid::parse(git(root, &["rev-parse", &spec]).trim()).unwrap();
    Base::at(
        id(format!("{rev}^{{commit}}")),
        id(format!("{rev}^{{tree}}")),
    )
}

fn keep(c: &Carried<'_>) -> Result<Resolved, VfsError> {
    keep_ours(c)
}

/// **A store reads one commit however the branch moves under it**: another
/// commit to the branch changes nothing it reads, and a new store — a new
/// conversation — takes the branch as it now stands.
#[test]
fn the_base_is_kept_while_the_branch_moves() {
    let (dir, source) = repo();
    let store = on_main(&source);
    store.edit("ten.txt", TEN.replace("l2\n", "two\n")).unwrap();
    let before = store.base().unwrap().unwrap();
    assert_eq!(before, base_of(dir.path(), "main"));

    commit_on_main(dir.path(), "ten.txt", "rewritten elsewhere\n");
    commit_on_main(dir.path(), "added.txt", "added elsewhere\n");
    assert_eq!(store.base().unwrap().unwrap(), before);
    assert_eq!(
        store.read("ten.txt").unwrap().unwrap(),
        TEN.replace("l2\n", "two\n")
    );
    assert_eq!(store.read("added.txt").unwrap(), None);
    assert!(store.conflicts().is_empty());

    let fresh = on_main(&source);
    assert_eq!(
        fresh.read("ten.txt").unwrap().as_deref(),
        Some("rewritten elsewhere\n")
    );
}

/// **Moved onto the conversation's own commit, the store lets go of what it
/// committed** and keeps the rest of its work.
#[test]
fn moving_onto_its_own_commit_lets_go_of_what_landed() {
    let (dir, source) = repo();
    let store = on_main(&source);
    let edited = TEN.replace("l3\n", "three\n");
    store.edit("ten.txt", edited.clone()).unwrap();
    store.write("notes.txt", "not committed\n".into()).unwrap();
    commit_on_main(dir.path(), "ten.txt", &edited);
    let conflicts = store
        .move_base(None, base_of(dir.path(), "main"), &[], &mut keep)
        .unwrap();
    assert!(conflicts.is_empty());
    assert!(!store.is_modified("ten.txt"));
    assert_eq!(store.read("ten.txt").unwrap(), Some(edited));
    assert_eq!(
        store.status(),
        vec![("notes.txt".to_string(), FileState::Added)]
    );
}

/// **A three-way move merges another writer's commit into the conversation's
/// copy**: edits to other lines merge cleanly; edits to the same lines are
/// both kept between markers, flagged until the conversation settles them —
/// and settling means taking the markers out, not just touching the file.
#[test]
fn a_three_way_move_merges_and_marks_what_overlaps() {
    let (dir, source) = repo();
    let repo = Repo::open(dir.path()).unwrap();
    let three = ThreeWay::new(
        &repo,
        MergeLabels {
            ours: "yours",
            base: "base",
            theirs: "main",
        },
    );
    let store = on_main(&source);
    store.edit("ten.txt", TEN.replace("l2\n", "two\n")).unwrap();
    commit_on_main(dir.path(), "ten.txt", &TEN.replace("l9\n", "nine\n"));
    let conflicts = store
        .move_base(
            None,
            base_of(dir.path(), "main"),
            &[],
            &mut |c: &Carried<'_>| three.carried(c),
        )
        .unwrap();
    assert!(conflicts.is_empty());
    let merged = TEN.replace("l2\n", "two\n").replace("l9\n", "nine\n");
    assert_eq!(store.read("ten.txt").unwrap().unwrap(), merged);

    store
        .edit("ten.txt", merged.replace("l5\n", "mine\n"))
        .unwrap();
    let theirs = TEN.replace("l9\n", "nine\n").replace("l5\n", "theirs\n");
    commit_on_main(dir.path(), "ten.txt", &theirs);
    let conflicts = store
        .move_base(
            None,
            base_of(dir.path(), "main"),
            &[],
            &mut |c: &Carried<'_>| three.carried(c),
        )
        .unwrap();
    assert_eq!(conflicts, ["ten.txt"]);
    let marked = store.read("ten.txt").unwrap().unwrap();
    assert_eq!(
        marked,
        TEN.replace("l2\n", "two\n")
            .replace("l9\n", "nine\n")
            .replace(
                "l5\n",
                "<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> main\n"
            )
    );
    assert_eq!(
        store.status(),
        vec![("ten.txt".to_string(), FileState::Modified)]
    );

    // Still marked: still in conflict.
    store
        .edit("ten.txt", marked.replace("l1\n", "one\n"))
        .unwrap();
    assert_eq!(store.conflicts(), ["ten.txt"]);
    // Settled.
    store
        .write("ten.txt", merged.replace("l5\n", "mine and theirs\n"))
        .unwrap();
    assert!(store.conflicts().is_empty());
}

/// **A snapshot keeps the base and the conflicts**, so a conversation
/// restored after the branch moved reads the commit it read before, with
/// the same work over it and the same paths in conflict.
#[test]
fn a_restored_store_reads_its_own_base_and_conflicts() {
    let (dir, source) = repo();
    let repo = Repo::open(dir.path()).unwrap();
    let three = ThreeWay::new(
        &repo,
        MergeLabels {
            ours: "yours",
            base: "base",
            theirs: "main",
        },
    );
    let store = on_main(&source);
    store
        .edit("ten.txt", TEN.replace("l2\n", "mine\n"))
        .unwrap();
    store.write("new.txt", "new\n".into()).unwrap();
    commit_on_main(dir.path(), "ten.txt", &TEN.replace("l2\n", "theirs\n"));
    store
        .move_base(
            None,
            base_of(dir.path(), "main"),
            &[],
            &mut |c: &Carried<'_>| three.carried(c),
        )
        .unwrap();
    let wire = serde_json::to_string(&store.snapshot()).unwrap();

    commit_on_main(dir.path(), "src/lib.rs", "moved on again\n");
    let restored = on_main(&source);
    let snapshot: Snapshot = serde_json::from_str(&wire).unwrap();
    restored.restore(snapshot).unwrap();
    assert_eq!(restored.base().unwrap(), store.base().unwrap());
    assert_eq!(restored.conflicts(), ["ten.txt"]);
    for path in ["ten.txt", "new.txt", "src/lib.rs"] {
        assert_eq!(
            restored.read(path).unwrap(),
            store.read(path).unwrap(),
            "{path}"
        );
    }
    assert_eq!(
        restored.read("src/lib.rs").unwrap().as_deref(),
        Some("pub fn lib() {}\n"),
        "not the commit made after it was saved"
    );
}

/// **Switching to another branch carries the conversation's changes onto
/// it**, merged with what that branch holds, and every file it did not change
/// reads as that branch holds it.
#[test]
fn switching_branch_carries_the_changes() {
    let (dir, source) = repo();
    let repo = Repo::open(dir.path()).unwrap();
    let three = ThreeWay::new(
        &repo,
        MergeLabels {
            ours: "yours",
            base: "main",
            theirs: "other",
        },
    );
    let store = on_main(&source);
    store.write("new.txt", "new\n".into()).unwrap();
    store.edit("ten.txt", TEN.replace("l1\n", "one\n")).unwrap();
    let conflicts = store
        .move_base(
            Some(branch("other")),
            base_of(dir.path(), "other"),
            &[],
            &mut |c: &Carried<'_>| three.carried(c),
        )
        .unwrap();
    assert!(conflicts.is_empty());
    assert_eq!(store.rev(), Some(Rev::Branch(branch("other"))));
    assert_eq!(store.read("new.txt").unwrap().as_deref(), Some("new\n"));
    assert_eq!(
        store.read("ten.txt").unwrap().unwrap(),
        TEN.replace("l1\n", "one\n").replace("l10\n", "ten\n")
    );
}

/// **Naming another branch on a store with no changes lets go of its base**,
/// so it reads that branch as it stands; one holding changes keeps the base
/// they are made on.
#[test]
fn naming_a_branch_repins_only_a_store_with_nothing_to_carry() {
    let (dir, source) = repo();
    let clean = on_main(&source);
    clean.read("ten.txt").unwrap();
    assert!(clean.set_branch(branch("other")));
    assert_eq!(clean.base().unwrap().unwrap(), base_of(dir.path(), "other"));

    let busy = on_main(&source);
    busy.write("x.txt", "x\n".into()).unwrap();
    assert!(busy.set_branch(branch("other")));
    assert_eq!(busy.base().unwrap().unwrap(), base_of(dir.path(), "main"));
}

/// **A reset drops every change and conflict and needs nothing of the old
/// base** — even one the repository no longer holds.
#[test]
fn a_reset_drops_the_work_even_from_a_base_that_is_gone() {
    let (dir, source) = repo();
    let store = on_main(&source);
    let gone = "ce013625030ba8dba906f756967f9e9ca394464a";
    let snapshot: Snapshot = serde_json::from_value(serde_json::json!({
        "base": { "tree": gone, "parents": [gone] },
        "chains": { "new.txt": { "deltas": [
            { "at_ns": 1, "kind": "replace", "content": "mine\n" }
        ], "size": 5 } }
    }))
    .unwrap();
    store.restore(snapshot).unwrap();
    assert!(store.read("ten.txt").is_err(), "its base is gone");
    assert_eq!(
        store.reset_to(None, base_of(dir.path(), "main")).unwrap(),
        1
    );
    assert_eq!(store.read("ten.txt").unwrap().as_deref(), Some(TEN));
    assert!(store.status().is_empty());
}

/// **Naming a branch never drops a merge being finished**, even with
/// nothing uncommitted: its tree and parents are the work.
#[test]
fn naming_a_branch_keeps_a_merge_being_finished() {
    let (dir, source) = repo();
    let store = on_main(&source);
    let main = base_of(dir.path(), "main");
    let other = base_of(dir.path(), "other");
    let merging = Base {
        tree: other.tree.clone(),
        parents: vec![
            main.commit().unwrap().clone(),
            other.commit().unwrap().clone(),
        ],
    };
    store
        .move_base(None, merging.clone(), &[], &mut keep)
        .unwrap();
    assert!(store.set_branch(branch("main")));
    assert_eq!(store.base().unwrap().unwrap(), merging);
}

/// **Keeping a file in conflict exactly as it stands settles it** — the
/// edit that met a delete — while one still marked stays in conflict.
#[test]
fn keeping_a_conflicted_file_as_it_stands_settles_it() {
    let (dir, source) = repo();
    let repo = Repo::open(dir.path()).unwrap();
    let three = ThreeWay::new(
        &repo,
        MergeLabels {
            ours: "yours",
            base: "base",
            theirs: "main",
        },
    );
    let store = on_main(&source);
    store
        .edit("src/lib.rs", "pub fn mine() {}\n".into())
        .unwrap();
    git(dir.path(), &["rm", "-q", "src/lib.rs"]);
    git(dir.path(), &["commit", "-q", "-m", "gone"]);
    store
        .move_base(
            None,
            base_of(dir.path(), "main"),
            &[],
            &mut |c: &Carried<'_>| three.carried(c),
        )
        .unwrap();
    assert_eq!(store.conflicts(), ["src/lib.rs"]);
    assert!(!store
        .edit("src/lib.rs", "pub fn mine() {}\n".into())
        .unwrap());
    assert!(store.conflicts().is_empty(), "kept as it stands: settled");
}

/// **A move that cannot finish changes nothing**: a resolver's refusal
/// leaves the base, the branch and the work as they were.
#[test]
fn a_move_that_fails_changes_nothing() {
    let (dir, source) = repo();
    let store = on_main(&source);
    store.edit("ten.txt", TEN.replace("l1\n", "one\n")).unwrap();
    let before = store.snapshot();
    let refused = store.move_base(
        Some(branch("other")),
        base_of(dir.path(), "other"),
        &[],
        &mut |_: &Carried<'_>| Err(VfsError::Full),
    );
    assert!(refused.is_err());
    assert_eq!(store.snapshot(), before);
    assert_eq!(store.rev(), Some(Rev::Branch(branch("main"))));
}

/// **The conversation's uncommitted work, path by path** — added, modified,
/// deleted against the branch — and **discarding it** leaves the branch.
#[test]
fn status_names_each_change_and_discard_drops_them_all() {
    let (_dir, source) = repo();
    let store = on_main(&source);
    store.write("new.txt", "new\n".into()).unwrap();
    store.edit("ten.txt", TEN.replace("l1\n", "one\n")).unwrap();
    assert!(store.delete("src/lib.rs"));
    assert_eq!(
        store.status(),
        vec![
            ("new.txt".to_string(), FileState::Added),
            ("src/lib.rs".to_string(), FileState::Deleted),
            ("ten.txt".to_string(), FileState::Modified),
        ]
    );
    store.discard();
    assert!(store.status().is_empty());
    assert_eq!(store.read("ten.txt").unwrap().as_deref(), Some(TEN));
    assert!(store.read("src/lib.rs").unwrap().is_some());
    assert_eq!(store.read("new.txt").unwrap(), None);
}

/// A branch that does not exist yet reads as empty, and takes writes.
#[test]
fn a_branch_that_does_not_exist_reads_empty() {
    let (_dir, source) = repo();
    let store = VfsStore::on_branch(source, Rev::Branch(branch("not-yet")));
    assert_eq!(store.read("ten.txt").unwrap(), None);
    assert!(store.paths("").is_empty());
    store.write("first.txt", "first\n".into()).unwrap();
    assert_eq!(store.paths(""), ["first.txt"]);
}
