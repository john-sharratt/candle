//! Materialising conversations onto one checkout and capturing what tools do
//! there — against real repositories and real files.
//!
//! The targeted tests pin each rule: the exact state materialising leaves, the
//! files it must not touch, what capture reads back and what it leaves alone,
//! and every refusal. The generated test then runs several conversations over
//! one checkout, alternating, each materialised, changed by a simulated tool
//! and captured, and checks after every step that the checkout and each
//! conversation's changes agree exactly.

mod support;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use support::{cached_repo, disk, git, put};
use tempfile::TempDir;
use zend_vfs::checkout::{capture, materialize, CheckoutError, Ledger};
use zend_vfs::file_delta::{self, FileDelta};
use zend_vfs::{BranchName, DiskWriteGrant, FileChanges, Head, Oid, Repo, RepoPath, Rev};

// ── Fixture ──────────────────────────────────────────────────────────────────

struct Fixture {
    _dir: TempDir,
    root: PathBuf,
    repo: Repo,
    /// The branch the conversations work on, `main`.
    branch: BranchName,
    /// Its commit.
    base: Oid,
    /// What [`at_base`] has read, by path.
    base_copies: RefCell<BTreeMap<String, Option<Vec<u8>>>>,
}

const LIB: &str = "pub fn one() -> u8 {\n    1\n}\n\npub fn two() -> u8 {\n    2\n}\n\npub fn three() -> u8 {\n    3\n}\n";
const BINARY: &[u8] = &[0, 159, 146, 150, 255, b'\n', 7];

/// A repository with text, a binary file, nested folders and ignore rules,
/// committed on `main`, which is checked out.
fn fixture() -> Fixture {
    let (dir, root, base) = cached_repo("checkout", build_repo);
    let repo = Repo::open(&root).unwrap();
    Fixture {
        _dir: dir,
        root,
        repo,
        branch: branch("main"),
        base,
        base_copies: RefCell::new(BTreeMap::new()),
    }
}

fn build_repo(root: &Path) {
    // No template: git's sample hooks are files every copy would carry.
    git(root, &["init", "-q", "--template="]);
    git(root, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    git(root, &["config", "core.autocrlf", "false"]);
    put(root, ".gitignore", b"target/\n*.log\n");
    put(root, "README.md", b"# app\n\nthe app.\n");
    put(root, "src/lib.rs", LIB.as_bytes());
    put(root, "src/main.rs", b"fn main() {\n    app::one();\n}\n");
    put(root, "assets/logo.bin", BINARY);
    put(root, "docs/guide/intro.md", b"intro\n");
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", "base"]);
}

fn branch(name: &str) -> BranchName {
    BranchName::parse(name).unwrap()
}

fn grant() -> DiskWriteGrant {
    DiskWriteGrant::issue()
}

fn text(s: &str) -> Option<Vec<u8>> {
    Some(s.as_bytes().to_vec())
}

/// What `path` holds at the base commit, as a checkout writes it.
///
/// Read once per path and kept: the base never changes under a test, and
/// the assertions ask for the same paths every round. A test that changes how
/// the base is checked out (its line endings) does so before its first read.
fn at_base(f: &Fixture, path: &str) -> Option<Vec<u8>> {
    if let Some(known) = f.base_copies.borrow().get(path) {
        return known.clone();
    }
    // One read for this path and every path the generated test uses.
    let mut paths: Vec<RepoPath> = PATHS.iter().map(|p| RepoPath::parse(p).unwrap()).collect();
    paths.push(RepoPath::parse(path).unwrap());
    let refs: Vec<&RepoPath> = paths.iter().collect();
    let read = f
        .repo
        .read_checked_out_all(&Rev::Oid(f.base.clone()), &refs)
        .unwrap();
    let mut copies = f.base_copies.borrow_mut();
    for p in &paths {
        copies.insert(p.as_str().to_string(), read.get(p.as_str()).cloned());
    }
    copies[path].clone()
}

/// What the conversation holds at `path`: its chain over the base.
fn view(f: &Fixture, changes: &FileChanges, path: &str) -> Option<Vec<u8>> {
    changes.replay(path, at_base(f, path)).unwrap()
}

/// Change `path` in `changes` to `new`, as the conversation would.
fn change(f: &Fixture, changes: &mut FileChanges, path: &str, new: Option<&[u8]>) {
    let old = view(f, changes, path);
    if let Some(delta) = file_delta::between(old.as_deref(), new) {
        changes.push(path, delta);
    }
}

/// Every path git reports as differing from the base, in order.
fn deviations(f: &Fixture) -> Vec<String> {
    let mut paths: Vec<String> = f
        .repo
        .status()
        .unwrap()
        .iter()
        .map(|e| e.path().as_str().to_string())
        .collect();
    paths.sort();
    paths
}

/// **The checkout holds exactly base + changes**: every changed path its
/// replayed content, and nothing else differing from the base — ignored files
/// aside.
fn assert_holds(f: &Fixture, changes: &FileChanges, context: &str) {
    for (path, _) in changes.iter() {
        assert_eq!(
            disk(&f.root, path),
            view(f, changes, path),
            "{context}: {path} on disk is not the conversation's"
        );
    }
    for path in deviations(f) {
        let changed =
            changes.chain(&path).is_some() && view(f, changes, &path) != at_base(f, &path);
        assert!(
            changed,
            "{context}: {path} differs from the base but the conversation did not change it"
        );
    }
}

fn set_old_mtime(root: &Path, rel: &str) -> SystemTime {
    let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
    std::fs::File::options()
        .write(true)
        .open(root.join(rel))
        .unwrap()
        .set_modified(old)
        .unwrap();
    old
}

fn mtime(root: &Path, rel: &str) -> SystemTime {
    std::fs::metadata(root.join(rel))
        .unwrap()
        .modified()
        .unwrap()
}

/// A conversation that touches every kind of change: an edit, a whole
/// replacement, a deletion, a new file in a new folder, a new binary file, an
/// edit to a binary file, and an ignored file.
fn every_kind(f: &Fixture) -> FileChanges {
    let mut c = FileChanges::new();
    change(
        f,
        &mut c,
        "src/lib.rs",
        Some(LIB.replace("    2\n", "    22\n").as_bytes()),
    );
    change(f, &mut c, "README.md", Some(b"# rewritten\n"));
    change(f, &mut c, "src/main.rs", None);
    change(f, &mut c, "src/new/module.rs", Some(b"pub mod module;\n"));
    change(f, &mut c, "assets/new.bin", Some(&[1, 2, 0xff]));
    change(f, &mut c, "assets/logo.bin", Some(&[9, 9, 0xfe]));
    change(f, &mut c, "notes.log", Some(b"ignored by git\n"));
    c
}

// ── Materialise ──────────────────────────────────────────────────────────────

/// **Base plus every kind of change lands exactly**, the ignored file
/// included, and the ledger names exactly the changed paths.
#[test]
fn every_kind_of_change_materialises_exactly() {
    let f = fixture();
    let changes = every_kind(&f);
    let (ledger, done) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();

    assert_holds(&f, &changes, "first materialise");
    assert_eq!(disk(&f.root, "src/main.rs"), None);
    assert_eq!(disk(&f.root, "notes.log"), text("ignored by git\n"));
    assert_eq!(disk(&f.root, "assets/new.bin"), Some(vec![1, 2, 0xff]));
    assert_eq!(
        ledger.paths().collect::<Vec<_>>(),
        changes.paths().collect::<Vec<_>>()
    );
    assert_eq!(done.written.len(), changes.len());
    assert!(done.kept.is_empty() && !done.reset);
    assert_eq!(
        disk(&f.root, "docs/guide/intro.md"),
        text("intro\n"),
        "an unchanged file is the base's"
    );
}

/// **Materialising the same state again touches nothing** — no file written,
/// every modification time where it was, the conversation's files and the
/// base's alike, and an ignored build output left alone.
#[test]
fn materialising_the_same_state_again_touches_nothing() {
    let f = fixture();
    let changes = every_kind(&f);
    put(&f.root, "target/debug/app.bin", b"build output\n");
    let (ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();

    let watched = [
        "src/lib.rs",
        "README.md",
        "src/new/module.rs",
        "assets/new.bin",
        "notes.log",
        "docs/guide/intro.md",
        "target/debug/app.bin",
    ];
    let old: Vec<SystemTime> = watched.iter().map(|p| set_old_mtime(&f.root, p)).collect();

    let (_, done) = materialize(&f.repo, &grant(), &f.branch, &changes, Some(ledger)).unwrap();
    assert!(done.written.is_empty(), "rewrote {:?}", done.written);
    assert!(done.removed.is_empty() && done.restored.is_empty() && !done.reset);
    assert_eq!(done.kept.len(), changes.len());
    for (path, old) in watched.iter().zip(old) {
        assert_eq!(mtime(&f.root, path), old, "{path} was touched");
    }
    assert_holds(&f, &changes, "second materialise");
}

/// **One conversation after another**: what only the first changed goes back
/// to the base — its new files removed, its deletion undone, its edits
/// reverted — what the second changes lands, and a file both hold with the
/// same bytes is not rewritten. Then back again.
#[test]
fn switching_conversations_leaves_only_the_second() {
    let f = fixture();
    let mut a = FileChanges::new();
    change(
        &f,
        &mut a,
        "src/lib.rs",
        Some(LIB.replace("    1\n", "    11\n").as_bytes()),
    );
    change(&f, &mut a, "src/main.rs", None);
    change(&f, &mut a, "only_a.txt", Some(b"a\n"));
    change(&f, &mut a, "shared.txt", Some(b"same in both\n"));
    change(&f, &mut a, "a.log", Some(b"a's ignored file\n"));
    let mut b = FileChanges::new();
    change(&f, &mut b, "README.md", Some(b"# b\n"));
    change(&f, &mut b, "only_b.txt", Some(b"b\n"));
    change(&f, &mut b, "shared.txt", Some(b"same in both\n"));

    let (ledger, _) = materialize(&f.repo, &grant(), &f.branch, &a, None).unwrap();
    assert_holds(&f, &a, "a");
    let shared_mtime = set_old_mtime(&f.root, "shared.txt");

    let (ledger, done) = materialize(&f.repo, &grant(), &f.branch, &b, Some(ledger)).unwrap();
    assert_holds(&f, &b, "b after a");
    assert_eq!(disk(&f.root, "only_a.txt"), None);
    assert_eq!(disk(&f.root, "a.log"), None, "a's ignored file is gone too");
    assert_eq!(disk(&f.root, "src/main.rs"), at_base(&f, "src/main.rs"));
    assert_eq!(disk(&f.root, "src/lib.rs"), text(LIB));
    assert_eq!(
        mtime(&f.root, "shared.txt"),
        shared_mtime,
        "identical bytes were rewritten"
    );
    assert_eq!(done.kept, ["shared.txt"]);
    let mut restored = done.restored.clone();
    restored.sort();
    assert_eq!(restored, ["src/lib.rs", "src/main.rs"]);

    let (_, _) = materialize(&f.repo, &grant(), &f.branch, &a, Some(ledger)).unwrap();
    assert_holds(&f, &a, "a again");
    assert_eq!(disk(&f.root, "only_b.txt"), None);
    assert_eq!(disk(&f.root, "README.md"), at_base(&f, "README.md"));
}

/// **Drift the ledger never saw is still put right**: a file changed on the
/// checkout by someone else, and a stray file, are found by status and reset.
#[test]
fn drift_on_the_checkout_is_put_right() {
    let f = fixture();
    let changes = every_kind(&f);
    let (ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    put(&f.root, "docs/guide/intro.md", b"someone else's edit\n");
    put(&f.root, "stray.txt", b"left behind\n");
    put(&f.root, "src/lib.rs", b"clobbered\n");

    let (_, done) = materialize(&f.repo, &grant(), &f.branch, &changes, Some(ledger)).unwrap();
    assert_holds(&f, &changes, "after drift");
    assert_eq!(disk(&f.root, "stray.txt"), None);
    assert_eq!(done.written, ["src/lib.rs"]);
}

/// **A different branch, or a changed index, means a reset first** — `HEAD`
/// moved onto the branch — and the conversation's changes land on the
/// branch's commit.
#[test]
fn a_new_branch_or_a_staged_change_resets_the_checkout() {
    let f = fixture();
    let changes = every_kind(&f);
    let (ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();

    // A tool that staged something.
    git(&f.root, &["add", "src/new/module.rs"]);
    let (ledger, done) = materialize(&f.repo, &grant(), &f.branch, &changes, Some(ledger)).unwrap();
    assert!(done.reset && !done.switched);
    assert_holds(&f, &changes, "after a staged change");

    // A second commit, made on branch `next` in a worktree of its own, as the
    // new base — the checkout under test untouched.
    let other = tempfile::tempdir().unwrap();
    let worktree = other.path().join("next");
    let at = worktree.to_str().unwrap();
    git(
        &f.root,
        &["worktree", "add", "-q", "-b", "next", at, "main"],
    );
    put(&worktree, "docs/guide/intro.md", b"intro, revised\n");
    git(&worktree, &["commit", "-q", "-am", "revise"]);
    git(&f.root, &["worktree", "remove", "--force", at]);
    let next = Oid::parse(git(&f.root, &["rev-parse", "refs/heads/next"]).trim()).unwrap();

    let (_, done) =
        materialize(&f.repo, &grant(), &branch("next"), &changes, Some(ledger)).unwrap();
    assert!(done.reset && done.switched);
    assert_eq!(
        f.repo.head().unwrap(),
        Head::Branch {
            branch: branch("next"),
            oid: next
        }
    );
    assert_eq!(
        disk(&f.root, "docs/guide/intro.md"),
        text("intro, revised\n")
    );
    assert_eq!(disk(&f.root, "README.md"), text("# rewritten\n"));
}

/// **A detached `HEAD` is moved onto the branch**, and a branch that does not
/// exist is refused with the checkout as it was.
#[test]
fn a_detached_checkout_is_put_on_the_branch() {
    let f = fixture();
    git(&f.root, &["checkout", "-q", "--detach", f.base.as_str()]);
    let changes = every_kind(&f);
    let (_, done) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    assert!(done.reset && done.switched);
    assert_eq!(f.repo.head().unwrap().branch(), Some(&f.branch));
    assert_holds(&f, &changes, "after the switch");

    let before = disk(&f.root, "README.md");
    assert!(matches!(
        materialize(&f.repo, &grant(), &branch("absent"), &changes, None),
        Err(CheckoutError::Git(_))
    ));
    assert_eq!(disk(&f.root, "README.md"), before);
}

/// **A file written is dated by the write**, however long ago its
/// conversation changed it — so it is never older than a build output
/// another conversation made before it, which a build cache would take as
/// covering it — and a file already holding its content keeps the time it had.
#[test]
fn a_written_file_is_dated_by_the_write() {
    let f = fixture();
    let mut changes = FileChanges::new();
    changes.push_timed(
        "README.md",
        file_delta::TimedDelta {
            at_ns: 1_600_000_000_000_000_000,
            delta: FileDelta::Replace {
                content: "# old change\n".into(),
            },
        },
    );
    changes.push_timed(
        "src/new.rs",
        file_delta::TimedDelta {
            at_ns: 1_600_000_000_000_000_000,
            delta: FileDelta::Replace {
                content: "new\n".into(),
            },
        },
    );
    // Another conversation's build, just before this one is put back.
    put(&f.root, "target/debug/app.bin", b"built\n");
    let built = mtime(&f.root, "target/debug/app.bin");

    let (ledger, done) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    assert_eq!(done.written, ["README.md", "src/new.rs"]);
    for path in ["README.md", "src/new.rs"] {
        assert!(
            mtime(&f.root, path) >= built,
            "{path} is older than the build that came before it"
        );
    }

    // Held already: not written, and its time is whatever it was.
    let kept = set_old_mtime(&f.root, "src/new.rs");
    let (_, done) = materialize(&f.repo, &grant(), &f.branch, &changes, Some(ledger)).unwrap();
    assert!(done.written.is_empty(), "{done:?}");
    assert_eq!(mtime(&f.root, "src/new.rs"), kept);
}

/// **Changes that do not fit the base are refused before anything changes**
/// — the checkout is exactly as it was.
#[test]
fn changes_that_do_not_fit_change_nothing() {
    let f = fixture();
    let good = every_kind(&f);
    let (ledger, _) = materialize(&f.repo, &grant(), &f.branch, &good, None).unwrap();
    let before: Vec<Option<Vec<u8>>> = ["src/lib.rs", "README.md", "notes.log"]
        .iter()
        .map(|p| disk(&f.root, p))
        .collect();

    let mut bad = good.clone();
    bad.push(
        "docs/guide/intro.md",
        FileDelta::Edit {
            splices: vec![file_delta::Splice {
                at: 0,
                removed: "not what the base holds\n".into(),
                inserted: "x\n".into(),
            }],
        },
    );
    match materialize(&f.repo, &grant(), &f.branch, &bad, Some(ledger)) {
        Err(CheckoutError::Diverged { path }) => assert_eq!(path, "docs/guide/intro.md"),
        other => panic!("{other:?}"),
    }
    let after: Vec<Option<Vec<u8>>> = ["src/lib.rs", "README.md", "notes.log"]
        .iter()
        .map(|p| disk(&f.root, p))
        .collect();
    assert_eq!(after, before);
}

/// **A path that could leave the checkout, or reach its git database, is
/// refused before anything changes.**
#[test]
fn an_unsafe_path_is_refused_before_anything_changes() {
    let f = fixture();
    for path in [".git/config", "../outside.txt", "C:/x.txt", "a.txt:stream"] {
        let mut c = FileChanges::new();
        c.push("README.md", FileDelta::Delete);
        c.push(
            path,
            FileDelta::Replace {
                content: "x".into(),
            },
        );
        let result = materialize(&f.repo, &grant(), &f.branch, &c, None);
        assert!(
            matches!(result, Err(CheckoutError::UnsafePath { .. })),
            "{path}: {result:?}"
        );
        assert_eq!(
            disk(&f.root, "README.md"),
            at_base(&f, "README.md"),
            "{path}"
        );
    }
    assert!(!f.root.parent().unwrap().join("outside.txt").exists());
}

/// **A folder a conversation's files made goes with them** — emptied, it is
/// removed up to the first folder that still holds anything — while a folder
/// that also holds a build output stays.
#[test]
fn folders_a_conversation_made_go_with_its_files() {
    let f = fixture();
    let mut changes = FileChanges::new();
    change(&f, &mut changes, "deep/er/new.txt", Some(b"new\n"));
    change(
        &f,
        &mut changes,
        "docs/guide/extra/more.md",
        Some(b"more\n"),
    );
    change(&f, &mut changes, "cache/kept.txt", Some(b"mine\n"));
    let (ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    put(&f.root, "cache/build.log", b"a build output\n");

    materialize(
        &f.repo,
        &grant(),
        &f.branch,
        &FileChanges::new(),
        Some(ledger),
    )
    .unwrap();
    assert!(!f.root.join("deep").exists(), "an emptied folder was left");
    assert!(!f.root.join("docs/guide/extra").exists());
    assert!(
        f.root.join("docs/guide/intro.md").is_file(),
        "a tracked folder stays"
    );
    assert_eq!(disk(&f.root, "cache/build.log"), text("a build output\n"));
    assert_eq!(disk(&f.root, "cache/kept.txt"), None);
}

/// An untracked repository nested in the checkout is a folder, and is never
/// removed.
#[test]
fn a_nested_repository_is_left_alone() {
    let f = fixture();
    let nested = f.root.join("vendor/dep");
    std::fs::create_dir_all(&nested).unwrap();
    git(&nested, &["init", "-q"]);
    put(&nested, "x.txt", b"x\n");
    materialize(&f.repo, &grant(), &f.branch, &FileChanges::new(), None).unwrap();
    assert_eq!(disk(&nested, "x.txt"), text("x\n"));
}

// ── Capture ──────────────────────────────────────────────────────────────────

/// **Everything a tool does comes back as deltas against the conversation's
/// state** — an edit to one of its files, an edit to a base file, a new file,
/// a new binary file, a deletion, the tool putting back a file the
/// conversation had deleted — and appended to its changes they replay to
/// exactly what is on disk. A build output under an ignored folder is not
/// captured.
#[test]
fn a_tools_changes_are_captured_and_replay_to_the_disk() {
    let f = fixture();
    let mut changes = every_kind(&f);
    let (mut ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();

    // The tool.
    let lib = String::from_utf8(disk(&f.root, "src/lib.rs").unwrap()).unwrap();
    put(
        &f.root,
        "src/lib.rs",
        lib.replace("    3\n", "    33\n").as_bytes(),
    );
    put(&f.root, "docs/guide/intro.md", b"intro\nmore\n");
    put(&f.root, "generated/out.rs", b"// generated\n");
    put(&f.root, "generated/blob.bin", &[0xff, 0, 1]);
    std::fs::remove_file(f.root.join("README.md")).unwrap();
    put(&f.root, "src/main.rs", &at_base(&f, "src/main.rs").unwrap());
    put(&f.root, "target/debug/app.bin", b"build output\n");
    put(
        &f.root,
        "build.log",
        b"an ignored log the conversation never wrote\n",
    );

    let captured = capture(&f.repo, &f.branch, &mut ledger).unwrap();
    let paths: Vec<&str> = captured.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(
        paths,
        [
            "README.md",
            "docs/guide/intro.md",
            "generated/blob.bin",
            "generated/out.rs",
            "src/lib.rs",
            "src/main.rs",
        ]
    );
    let kind = |p: &str| &captured.iter().find(|(q, _)| q == p).unwrap().1.delta;
    assert!(matches!(kind("src/lib.rs"), FileDelta::Edit { .. }));
    assert!(matches!(kind("README.md"), FileDelta::Delete));
    assert!(matches!(
        kind("generated/blob.bin"),
        FileDelta::ReplaceBinary { .. }
    ));
    assert!(matches!(kind("src/main.rs"), FileDelta::Replace { .. }));

    changes.extend(captured);
    assert_holds(&f, &changes, "after capture");
    for path in ["generated/out.rs", "generated/blob.bin", "README.md"] {
        assert_eq!(view(&f, &changes, path), disk(&f.root, path), "{path}");
    }

    // The ledger now says what is on disk: materialising the result touches
    // nothing.
    let (_, done) = materialize(&f.repo, &grant(), &f.branch, &changes, Some(ledger)).unwrap();
    assert!(done.written.is_empty(), "rewrote {:?}", done.written);
}

/// **A file taken out of the index but left on disk is not a change** — git
/// reports it both as a staged delete and as untracked, and the base's copy is
/// what it is compared with.
#[test]
fn a_file_removed_from_the_index_only_is_not_a_change() {
    let f = fixture();
    let (mut ledger, _) =
        materialize(&f.repo, &grant(), &f.branch, &FileChanges::new(), None).unwrap();
    git(&f.root, &["rm", "-q", "--cached", "README.md"]);
    assert!(capture(&f.repo, &f.branch, &mut ledger).unwrap().is_empty());
}

/// **A tool that changes nothing yields nothing**, and one that rewrites a
/// file with the bytes it already had yields nothing either.
#[test]
fn a_tool_that_changes_nothing_yields_nothing() {
    let f = fixture();
    let changes = every_kind(&f);
    let (mut ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    assert!(capture(&f.repo, &f.branch, &mut ledger).unwrap().is_empty());

    let same = disk(&f.root, "src/lib.rs").unwrap();
    put(&f.root, "src/lib.rs", &same);
    put(&f.root, "docs/guide/intro.md", b"intro\n");
    assert!(capture(&f.repo, &f.branch, &mut ledger).unwrap().is_empty());
}

/// **A same-size write whose modification time was put back is caught while
/// the stamp is fresh** — within the racy window the ledger compares content
/// instead of trusting the stamp.
#[test]
fn a_restored_mtime_is_caught_inside_the_racy_window() {
    let f = fixture();
    let mut changes = FileChanges::new();
    change(&f, &mut changes, "README.md", Some(b"aaaa\n"));
    let (mut ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    let written = mtime(&f.root, "README.md");
    put(&f.root, "README.md", b"bbbb\n");
    std::fs::File::options()
        .write(true)
        .open(f.root.join("README.md"))
        .unwrap()
        .set_modified(written)
        .unwrap();

    let captured = capture(&f.repo, &f.branch, &mut ledger).unwrap();
    assert_eq!(captured.len(), 1, "{captured:?}");
    changes.extend(captured);
    assert_eq!(view(&f, &changes, "README.md"), text("bbbb\n"));
}

/// **Line endings follow the checkout.** With `core.autocrlf=true` the base
/// is checked out with CRLF; the conversation's changes are made against
/// that, materialise exactly, and a tool's CRLF edit comes back as an edit of
/// CRLF lines.
#[test]
fn a_crlf_checkout_round_trips() {
    let f = fixture();
    git(&f.root, &["config", "core.autocrlf", "true"]);
    for p in [
        "README.md",
        "src/lib.rs",
        "src/main.rs",
        "docs/guide/intro.md",
    ] {
        std::fs::remove_file(f.root.join(p)).unwrap();
    }
    git(&f.root, &["checkout", "-q", "--", "."]);
    assert_eq!(at_base(&f, "README.md"), text("# app\r\n\r\nthe app.\r\n"));

    let mut changes = FileChanges::new();
    let crlf_lib = LIB.replace('\n', "\r\n");
    change(
        &f,
        &mut changes,
        "src/lib.rs",
        Some(crlf_lib.replace("    2\r\n", "    22\r\n").as_bytes()),
    );
    let (mut ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    assert_holds(&f, &changes, "crlf");

    let now = String::from_utf8(disk(&f.root, "src/lib.rs").unwrap()).unwrap();
    put(
        &f.root,
        "src/lib.rs",
        now.replace("    3\r\n", "    33\r\n").as_bytes(),
    );
    let captured = capture(&f.repo, &f.branch, &mut ledger).unwrap();
    let FileDelta::Edit { splices } = &captured[0].1.delta else {
        panic!("{captured:?}");
    };
    assert_eq!(splices[0].inserted, "    33\r\n");
    changes.extend(captured);
    assert_holds(&f, &changes, "crlf after capture");
}

/// A ledger from a previous process works as well as one kept in memory.
#[test]
fn a_ledger_survives_its_wire_form() {
    let f = fixture();
    let changes = every_kind(&f);
    let (ledger, _) = materialize(&f.repo, &grant(), &f.branch, &changes, None).unwrap();
    let wire = serde_json::to_string(&ledger).unwrap();
    let back: Ledger = serde_json::from_str(&wire).unwrap();
    assert_eq!(back, ledger);
    let (_, done) = materialize(&f.repo, &grant(), &f.branch, &changes, Some(back)).unwrap();
    assert!(done.written.is_empty());
}

// ── Generated: many conversations, one checkout ──────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Paths a conversation or a tool may touch: existing files, new ones in new
/// folders, an ignored one.
const PATHS: &[&str] = &[
    "README.md",
    "src/lib.rs",
    "src/main.rs",
    "docs/guide/intro.md",
    "assets/logo.bin",
    "src/extra.rs",
    "notes/a/b.md",
    "local.log",
];

/// A random new version of `old`: a line changed, added or removed, the whole
/// file replaced, binary bytes, or the file deleted.
fn mutate(rng: &mut Lcg, old: Option<&[u8]>, n: &mut usize) -> Option<Vec<u8>> {
    *n += 1;
    match rng.below(6) {
        0 => None,
        1 => Some(vec![0xff, (*n % 251) as u8, 0]),
        2 => Some(format!("whole file {n}\n").into_bytes()),
        _ => {
            let base = old
                .and_then(|b| std::str::from_utf8(b).ok())
                .unwrap_or("")
                .to_string();
            let mut lines: Vec<String> = base.lines().map(str::to_string).collect();
            let at = rng.below(lines.len() + 1);
            match rng.below(3) {
                0 if at < lines.len() => lines[at] = format!("changed {n}"),
                1 if at < lines.len() => {
                    lines.remove(at);
                }
                _ => lines.insert(at, format!("added {n}")),
            }
            Some(
                lines
                    .iter()
                    .map(|l| format!("{l}\n"))
                    .collect::<String>()
                    .into_bytes(),
            )
        }
    }
}

/// **Three conversations take turns on one checkout.** Each round a
/// conversation is materialised, changes some of its own files, a simulated
/// tool changes others on disk, and the tool's changes are captured. After
/// every materialise and every capture the checkout holds exactly that
/// conversation's state; a file no conversation ever touches is never
/// rewritten; and materialising the conversation that just ran again writes
/// nothing.
///
/// A round is three real passes over the checkout, so the rounds are spread
/// over several seeded tests that run side by side, each from a fresh
/// checkout, rather than one long sequence.
fn conversations_take_turns(seed: u64) {
    const ROUNDS: usize = 3;
    let f = fixture();
    let never_touched = set_old_mtime(&f.root, ".gitignore");
    let mut rng = Lcg(seed);
    let mut n = 0;
    let mut conversations: Vec<FileChanges> = vec![FileChanges::new(); 3];
    let mut ledger: Option<Ledger> = None;

    for round in 0..ROUNDS {
        let who = rng.below(conversations.len());
        let c = &mut conversations[who];
        // The conversation changes a file or two of its own.
        for _ in 0..=rng.below(2) {
            let path = PATHS[rng.below(PATHS.len())];
            let old = view(&f, c, path);
            let new = mutate(&mut rng, old.as_deref(), &mut n);
            change(&f, c, path, new.as_deref());
        }

        let (mut l, _) = materialize(&f.repo, &grant(), &f.branch, c, ledger.take())
            .unwrap_or_else(|e| panic!("round {round}: {e}"));
        assert_holds(
            &f,
            c,
            &format!("round {round}, conversation {who} materialised"),
        );

        // The tool changes a file or two on disk.
        for _ in 0..=rng.below(2) {
            let path = PATHS[rng.below(PATHS.len())];
            let old = disk(&f.root, path);
            match mutate(&mut rng, old.as_deref(), &mut n) {
                Some(bytes) => put(&f.root, path, &bytes),
                None => {
                    let _ = std::fs::remove_file(f.root.join(path));
                }
            }
        }
        let captured = capture(&f.repo, &f.branch, &mut l).unwrap();
        c.extend(captured);
        assert_holds(
            &f,
            c,
            &format!("round {round}, conversation {who} captured"),
        );
        for path in PATHS {
            // An ignored file the tool made and the conversation never wrote
            // is a by-product — a build output, a log — and is not captured.
            if path.ends_with(".log") && c.chain(path).is_none() {
                continue;
            }
            assert_eq!(
                disk(&f.root, path),
                view(&f, c, path),
                "round {round}: {path} after capture"
            );
        }

        let (l, done) = materialize(&f.repo, &grant(), &f.branch, c, Some(l)).unwrap();
        assert!(
            done.written.is_empty() && done.removed.is_empty(),
            "round {round}: {done:?}"
        );
        ledger = Some(l);
    }
    assert_eq!(mtime(&f.root, ".gitignore"), never_touched);
}

#[test]
fn conversations_take_turns_seed_1() {
    conversations_take_turns(0xc0ffee);
}

#[test]
fn conversations_take_turns_seed_2() {
    conversations_take_turns(0xbeef);
}

#[test]
fn conversations_take_turns_seed_3() {
    conversations_take_turns(0x5eed);
}
