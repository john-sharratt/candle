//! A conversation's work joining a branch whose record is origin: every
//! commit published whole or not at all, every merge landing in the
//! conversation's own copy — and nothing anyone wrote ever lost.
//!
//! Origin is a bare repository on disk; "someone else" is a second clone of
//! it that commits and pushes the way any other developer would. The daemon's
//! clone is `local`, and each conversation is a store over it.

mod support;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use support::{branch, git, put};
use tempfile::TempDir;
use zend_vfs::{
    merge_into, Base, ChangeSet, Committing, FileState, GitSource, Landed, Merged, NotCommitted,
    Oid, Published, PushOutcome, Rejection, Repo, RepoPath, Rev, Snapshot, VfsStore,
};

const TEN: &str = "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n";
const BINARY: &[u8] = &[0, 1, 2, 3, 0, 255];

/// Origin, the daemon's clone of it, and someone else's.
struct World {
    _dir: TempDir,
    origin: PathBuf,
    local: PathBuf,
    other: PathBuf,
    repo: Repo,
    source: Arc<GitSource>,
}

fn configure(root: &Path) {
    git(root, &["config", "core.autocrlf", "false"]);
    git(root, &["config", "user.name", "T"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "commit.gpgSign", "false"]);
}

impl World {
    /// Origin holding `main` with `ten.txt`, `keep.txt` and a binary file;
    /// both clones on it.
    fn new() -> Self {
        Self::with_origin(true)
    }

    /// The same, with or without an origin: without one, `local` is the
    /// record and `other` stands for nothing.
    fn with_origin(origin: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let (origin_dir, local, other) =
            (root.join("origin"), root.join("local"), root.join("other"));
        let seed = root.join("seed");
        std::fs::create_dir_all(&seed).unwrap();
        git(&seed, &["init", "-q", "--template="]);
        git(&seed, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        configure(&seed);
        put(&seed, "ten.txt", TEN.as_bytes());
        put(&seed, "keep.txt", b"keep\n");
        put(&seed, "logo.bin", BINARY);
        git(&seed, &["add", "-A"]);
        git(&seed, &["commit", "-q", "-m", "base"]);
        if origin {
            git(root, &["clone", "-q", "--bare", "seed", "origin"]);
            for clone in ["local", "other"] {
                git(root, &["clone", "-q", "origin", clone]);
                configure(&root.join(clone));
            }
        } else {
            git(root, &["clone", "-q", "seed", "local"]);
            git(&local, &["remote", "remove", "origin"]);
            configure(&local);
            std::fs::create_dir_all(&other).unwrap();
        }
        let repo = Repo::open(&local).unwrap();
        let source = GitSource::open(&local).unwrap();
        World {
            _dir: dir,
            origin: origin_dir,
            local,
            other,
            repo,
            source,
        }
    }

    /// A new conversation on `main`.
    fn conversation(&self) -> VfsStore {
        VfsStore::on_branch(Arc::clone(&self.source), Rev::Branch(branch("main")))
    }

    /// Someone else commits `files` — `None` deleting — and pushes.
    fn push_other(&self, files: &[(&str, Option<&[u8]>)]) -> Oid {
        git(&self.other, &["pull", "-q", "--ff-only", "origin", "main"]);
        for (path, content) in files {
            match content {
                Some(bytes) => put(&self.other, path, bytes),
                None => std::fs::remove_file(self.other.join(path)).unwrap(),
            }
        }
        git(&self.other, &["add", "-A"]);
        git(&self.other, &["commit", "-q", "-m", "someone else"]);
        git(&self.other, &["push", "-q", "origin", "main"]);
        self.oid(&self.other, "HEAD")
    }

    /// A commit made on this machine and never pushed: `local`'s `main`
    /// moved by hand.
    fn commit_locally(&self, path: &str, content: &str) -> Oid {
        git(&self.local, &["checkout", "-q", "main"]);
        put(&self.local, path, content.as_bytes());
        git(&self.local, &["add", "-A"]);
        git(&self.local, &["commit", "-q", "-m", "local only"]);
        self.oid(&self.local, "HEAD")
    }

    fn oid(&self, root: &Path, rev: &str) -> Oid {
        Oid::parse(git(root, &["rev-parse", rev]).trim()).unwrap()
    }

    fn on_origin(&self) -> Oid {
        self.oid(&self.origin, "refs/heads/main")
    }

    fn local_main(&self) -> Oid {
        self.oid(&self.local, "refs/heads/main")
    }

    /// `path` as origin's `main` holds it.
    fn origin_file(&self, path: &str) -> Option<String> {
        let listed = git(&self.origin, &["ls-tree", "--name-only", "main", path]);
        (!listed.trim().is_empty()).then(|| git(&self.origin, &["show", &format!("main:{path}")]))
    }

    /// The parents of `commit`, in order.
    fn parents(&self, commit: &Oid) -> Vec<Oid> {
        let line = git(
            &self.origin,
            &["rev-list", "--parents", "-n", "1", commit.as_str()],
        );
        line.split_whitespace()
            .skip(1)
            .map(|p| Oid::parse(p).unwrap())
            .collect()
    }

    /// Commit every change `store` holds, as `git_commit from: all_changes` does.
    fn commit(&self, store: &VfsStore) -> Result<Landed, NotCommitted> {
        let main = branch("main");
        let committing = Committing::begin(&self.repo, store, &main).unwrap()?;
        let mut set = ChangeSet::new();
        for (path, state) in store.status() {
            let at = RepoPath::parse(&path).unwrap();
            match state {
                FileState::Deleted => set.delete(at).unwrap(),
                _ => set
                    .write(at, store.read_bytes(&path).unwrap().unwrap(), None)
                    .unwrap(),
            }
        }
        let me = self.repo.identity().unwrap();
        let built = self
            .repo
            .commit_changes(committing.onto(), &set, "work", &me, &me)
            .unwrap();
        committing
            .land(&self.repo, store, &built, "work", &me, &me)
            .unwrap()
    }

    /// Merge the branch as origin now holds it into `store`, as `git_merge`
    /// does.
    fn merge(&self, store: &VfsStore) -> Merged {
        let pulled = self.repo.pull_branch(&branch("main")).unwrap();
        let theirs = pulled.record().unwrap().clone();
        merge_into(&self.repo, store, &theirs, "origin/main").unwrap()
    }
}

fn text(store: &VfsStore, path: &str) -> Option<String> {
    store.read(path).unwrap()
}

// ── committing ──────────────────────────────────────────────────────────────

/// **A commit lands on origin, the local branch follows, and the
/// conversation's copy reads it from the branch**: nothing left uncommitted,
/// its base the new commit.
#[test]
fn a_commit_lands_on_origin_and_the_conversation_follows() {
    let w = World::new();
    let conv = w.conversation();
    conv.edit("ten.txt", TEN.replace("l1\n", "one\n")).unwrap();
    conv.write("new.txt", "new\n".into()).unwrap();
    assert!(conv.delete("keep.txt"));
    let before = w.on_origin();

    let landed = w.commit(&conv).unwrap();
    assert_eq!(
        landed.published,
        Published::Pushed(PushOutcome::FastForward)
    );
    assert_eq!(landed.parents, vec![before]);
    assert_eq!(w.on_origin(), landed.commit);
    assert_eq!(w.local_main(), landed.commit);
    assert_eq!(
        w.origin_file("ten.txt").unwrap(),
        TEN.replace("l1\n", "one\n")
    );
    assert_eq!(w.origin_file("new.txt").as_deref(), Some("new\n"));
    assert_eq!(w.origin_file("keep.txt"), None);
    assert!(conv.status().is_empty());
    assert_eq!(conv.base().unwrap().unwrap().commit(), Some(&landed.commit));
    assert_eq!(text(&conv, "new.txt").as_deref(), Some("new\n"));
}

/// **Origin gained a commit the conversation does not have: the commit is
/// refused, and nothing is written anywhere** — origin, the conversation's
/// base and its work all exactly as they were.
#[test]
fn a_commit_behind_origin_is_refused_with_nothing_written() {
    let w = World::new();
    let conv = w.conversation();
    conv.edit("ten.txt", TEN.replace("l1\n", "one\n")).unwrap();
    let theirs = w.push_other(&[("keep.txt", Some(b"theirs\n"))]);
    let before = conv.snapshot();

    assert_eq!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Behind {
            record: theirs.clone()
        }
    );
    assert_eq!(w.on_origin(), theirs);
    assert_eq!(conv.snapshot(), before);
}

/// **Origin moving between the check and the push refuses the commit** —
/// the lease — and again nothing is written: origin keeps the other
/// writer's commit, the conversation its work.
#[test]
fn origin_moving_during_a_commit_refuses_it_whole() {
    let w = World::new();
    let conv = w.conversation();
    conv.edit("ten.txt", TEN.replace("l1\n", "one\n")).unwrap();
    let before = conv.snapshot();
    let committing = Committing::begin(&w.repo, &conv, &branch("main"))
        .unwrap()
        .unwrap();
    let theirs = w.push_other(&[("keep.txt", Some(b"raced\n"))]);
    let mut set = ChangeSet::new();
    set.write(
        RepoPath::parse("ten.txt").unwrap(),
        text(&conv, "ten.txt").unwrap().into_bytes(),
        None,
    )
    .unwrap();
    let me = w.repo.identity().unwrap();
    let built = w
        .repo
        .commit_changes(committing.onto(), &set, "work", &me, &me)
        .unwrap();
    let outcome = committing
        .land(&w.repo, &conv, &built, "work", &me, &me)
        .unwrap();
    assert_eq!(outcome, Err(NotCommitted::Refused(Rejection::Stale)));
    assert_eq!(w.on_origin(), theirs);
    assert_eq!(conv.snapshot(), before);
}

/// A commit on this machine that builds on `main` — another branch's work —
/// with no ref pointing at it.
fn ahead_of_main(w: &World) -> Oid {
    let tree = git(&w.local, &["rev-parse", "main^{tree}"]);
    let made = git(
        &w.local,
        &["commit-tree", tree.trim(), "-p", "main", "-m", "elsewhere"],
    );
    Oid::parse(made.trim()).unwrap()
}

/// **A fast-forward merge of other work is published as it is**: the base is
/// ahead of the branch, and publishing it moves origin and the local branch
/// onto it with no commit of its own.
#[test]
fn a_fast_forwarded_base_is_published_as_it_is() {
    let w = World::new();
    let conv = w.conversation();
    let theirs = ahead_of_main(&w);
    assert_eq!(
        merge_into(&w.repo, &conv, &theirs, "elsewhere").unwrap(),
        Merged::FastForward { conflicts: vec![] }
    );
    let committing = Committing::begin(&w.repo, &conv, &branch("main"))
        .unwrap()
        .unwrap();
    assert!(committing.ahead());
    assert_eq!(committing.record(), Some(&w.on_origin()));
    let landed = committing.publish_base(&w.repo).unwrap().unwrap();
    assert_eq!(landed.commit, theirs);
    assert!(landed.parents.is_empty(), "no commit was made");
    assert_eq!(w.on_origin(), theirs);
    assert_eq!(w.local_main(), theirs);
    // Published, the base is the branch again: nothing ahead.
    let again = Committing::begin(&w.repo, &conv, &branch("main"))
        .unwrap()
        .unwrap();
    assert!(!again.ahead());
}

/// **Origin moving before the fast-forward is published refuses it**, and
/// nothing moves — the lease, as for any commit.
#[test]
fn origin_moving_refuses_publishing_a_fast_forward() {
    let w = World::new();
    let conv = w.conversation();
    let theirs = ahead_of_main(&w);
    merge_into(&w.repo, &conv, &theirs, "elsewhere").unwrap();
    let committing = Committing::begin(&w.repo, &conv, &branch("main"))
        .unwrap()
        .unwrap();
    let raced = w.push_other(&[("keep.txt", Some(b"raced\n"))]);
    assert_eq!(
        committing.publish_base(&w.repo).unwrap(),
        Err(NotCommitted::Refused(Rejection::Stale))
    );
    assert_eq!(w.on_origin(), raced);
    assert_eq!(conv.base().unwrap().unwrap().commit(), Some(&theirs));
}

/// **Two conversations on one branch**: the first's commit changes nothing
/// the second reads; the second's commit is refused until it merges; then
/// it lands on top, and origin holds both.
#[test]
fn two_conversations_meet_through_a_merge() {
    let w = World::new();
    let (a, b) = (w.conversation(), w.conversation());
    a.edit("ten.txt", TEN.replace("l1\n", "one\n")).unwrap();
    b.edit("ten.txt", TEN.replace("l9\n", "nine\n")).unwrap();
    let first = w.commit(&a).unwrap().commit;
    assert_eq!(
        text(&b, "ten.txt").unwrap(),
        TEN.replace("l9\n", "nine\n"),
        "b still reads its own base"
    );
    assert_eq!(
        w.commit(&b).unwrap_err(),
        NotCommitted::Behind {
            record: first.clone()
        }
    );

    assert_eq!(w.merge(&b), Merged::FastForward { conflicts: vec![] });
    assert_eq!(
        text(&b, "ten.txt").unwrap(),
        TEN.replace("l1\n", "one\n").replace("l9\n", "nine\n")
    );
    let second = w.commit(&b).unwrap();
    assert_eq!(second.parents, vec![first]);
    assert_eq!(
        w.origin_file("ten.txt").unwrap(),
        TEN.replace("l1\n", "one\n").replace("l9\n", "nine\n")
    );
}

/// **A conversation with nothing uncommitted merges by moving its base**,
/// and a merge with nothing new is up to date.
#[test]
fn a_clean_conversation_fast_forwards_and_then_is_up_to_date() {
    let w = World::new();
    let conv = w.conversation();
    assert_eq!(text(&conv, "keep.txt").as_deref(), Some("keep\n"));
    let theirs = w.push_other(&[("keep.txt", Some(b"theirs\n"))]);
    assert_eq!(w.merge(&conv), Merged::FastForward { conflicts: vec![] });
    assert_eq!(conv.base().unwrap().unwrap().commit(), Some(&theirs));
    assert_eq!(text(&conv, "keep.txt").as_deref(), Some("theirs\n"));
    assert!(conv.status().is_empty());
    assert_eq!(w.merge(&conv), Merged::UpToDate);
}

// ── conflicts ───────────────────────────────────────────────────────────────

/// **The same lines changed on both sides are both kept, between markers,
/// in the conversation's copy** — origin untouched — and every commit is
/// refused while they stand; settled, the commit lands with the
/// conversation's resolution on top of the other writer's commit.
#[test]
fn overlapping_changes_are_marked_and_block_commits_until_settled() {
    let w = World::new();
    let conv = w.conversation();
    conv.edit("ten.txt", TEN.replace("l5\n", "mine\n")).unwrap();
    let theirs = w.push_other(&[("ten.txt", Some(TEN.replace("l5\n", "theirs\n").as_bytes()))]);
    let merged = w.merge(&conv);
    assert_eq!(
        merged,
        Merged::FastForward {
            conflicts: vec!["ten.txt".to_string()]
        }
    );
    assert_eq!(
        text(&conv, "ten.txt").unwrap(),
        TEN.replace(
            "l5\n",
            "<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> origin/main\n"
        )
    );
    assert_eq!(w.on_origin(), theirs, "a merge writes nothing to origin");
    assert_eq!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Conflicts(vec!["ten.txt".to_string()])
    );

    conv.write("ten.txt", TEN.replace("l5\n", "mine and theirs\n"))
        .unwrap();
    let landed = w.commit(&conv).unwrap();
    assert_eq!(landed.parents, vec![theirs]);
    assert_eq!(
        w.origin_file("ten.txt").unwrap(),
        TEN.replace("l5\n", "mine and theirs\n")
    );
}

/// **A conflict stays in conflict through later merges** — however cleanly
/// they merge the marked file — so its markers can never be committed.
#[test]
fn a_conflict_outlives_later_merges() {
    let w = World::new();
    let conv = w.conversation();
    conv.edit("ten.txt", TEN.replace("l5\n", "mine\n")).unwrap();
    w.push_other(&[("ten.txt", Some(TEN.replace("l5\n", "theirs\n").as_bytes()))]);
    w.merge(&conv);
    assert_eq!(conv.conflicts(), ["ten.txt"]);
    w.push_other(&[(
        "ten.txt",
        Some(
            TEN.replace("l5\n", "theirs\n")
                .replace("l10\n", "ten\n")
                .as_bytes(),
        ),
    )]);
    assert_eq!(
        w.merge(&conv),
        Merged::FastForward {
            conflicts: vec!["ten.txt".to_string()]
        }
    );
    assert!(text(&conv, "ten.txt").unwrap().contains("<<<<<<< yours"));
    assert!(matches!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Conflicts(_)
    ));
}

/// **Origin moving on the conversation's own side while it finishes a merge
/// of another branch takes that side's place in the merge**, and the merge
/// commit lands on top of it.
#[test]
fn origin_moving_on_the_conversations_side_joins_the_merge() {
    let w = World::new();
    git(&w.other, &["checkout", "-q", "-b", "feature"]);
    put(&w.other, "feature.txt", b"feature\n");
    git(&w.other, &["add", "-A"]);
    git(&w.other, &["commit", "-q", "-m", "feature"]);
    git(&w.other, &["push", "-q", "origin", "feature"]);
    let feature = w.oid(&w.other, "HEAD");
    git(&w.other, &["checkout", "-q", "main"]);

    let conv = w.conversation();
    conv.write("mine.txt", "mine\n".into()).unwrap();
    // A commit on main the feature branch does not have: the merge is a real
    // one.
    let first = w.commit(&conv).unwrap().commit;
    git(&w.local, &["fetch", "-q", "origin"]);
    let merged = merge_into(&w.repo, &conv, &feature, "feature").unwrap();
    assert_eq!(merged, Merged::Merging { conflicts: vec![] });
    assert_eq!(
        conv.base().unwrap().unwrap().parents,
        vec![first.clone(), feature.clone()]
    );

    let later = w.push_other(&[("keep.txt", Some(b"later on main\n"))]);
    assert!(matches!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Behind { .. }
    ));
    assert_eq!(w.merge(&conv), Merged::Merging { conflicts: vec![] });
    assert_eq!(
        conv.base().unwrap().unwrap().parents,
        vec![later.clone(), feature.clone()],
        "main's new tip takes the place of the commit it builds on"
    );
    assert_eq!(w.merge(&conv), Merged::UpToDate);
    let landed = w.commit(&conv).unwrap();
    assert_eq!(w.parents(&landed.commit), vec![later, feature]);
    assert_eq!(w.origin_file("feature.txt").as_deref(), Some("feature\n"));
    assert_eq!(
        w.origin_file("keep.txt").as_deref(),
        Some("later on main\n")
    );
    assert_eq!(w.origin_file("mine.txt").as_deref(), Some("mine\n"));
    assert_eq!(
        git(&w.local, &["for-each-ref", "refs/zend/"]),
        "",
        "the merge's tree is let go once committed"
    );
}

/// **A commit that changes a file the conversation also changed is never
/// undone by the next one**: the conversation's copy keeps both changes.
#[test]
fn a_commit_is_never_undone_by_the_next() {
    let w = World::new();
    let conv = w.conversation();
    conv.edit("ten.txt", TEN.replace("l1\n", "mine\n")).unwrap();
    let committing = Committing::begin(&w.repo, &conv, &branch("main"))
        .unwrap()
        .unwrap();
    let mut set = ChangeSet::new();
    set.write(
        RepoPath::parse("ten.txt").unwrap(),
        TEN.replace("l9\n", "committed\n").into_bytes(),
        None,
    )
    .unwrap();
    let me = w.repo.identity().unwrap();
    let built = w
        .repo
        .commit_changes(committing.onto(), &set, "l9", &me, &me)
        .unwrap();
    let landed = committing
        .land(&w.repo, &conv, &built, "l9", &me, &me)
        .unwrap()
        .unwrap();
    assert_eq!(landed.behind, None);
    assert_eq!(
        text(&conv, "ten.txt").unwrap(),
        TEN.replace("l1\n", "mine\n").replace("l9\n", "committed\n")
    );
    w.commit(&conv).unwrap();
    assert_eq!(
        w.origin_file("ten.txt").unwrap(),
        TEN.replace("l1\n", "mine\n").replace("l9\n", "committed\n")
    );
}

/// **A file the other side deleted and the conversation edited is kept, in
/// conflict** — deleting it or keeping it is the conversation's call, and
/// either settles it.
#[test]
fn an_edit_meeting_a_delete_is_kept_in_conflict() {
    let w = World::new();
    let conv = w.conversation();
    conv.edit("keep.txt", "edited\n".into()).unwrap();
    w.push_other(&[("keep.txt", None)]);
    assert_eq!(
        w.merge(&conv),
        Merged::FastForward {
            conflicts: vec!["keep.txt".to_string()]
        }
    );
    assert_eq!(text(&conv, "keep.txt").as_deref(), Some("edited\n"));
    assert_eq!(
        conv.status(),
        vec![("keep.txt".to_string(), FileState::Added)]
    );
    assert!(matches!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Conflicts(_)
    ));
    assert!(conv.delete("keep.txt"));
    assert!(conv.conflicts().is_empty());
    assert!(conv.status().is_empty(), "deleted on both sides now");
}

/// **A delete meeting the other side's edit keeps the edit, in conflict**;
/// writing the file as it should be settles it.
#[test]
fn a_delete_meeting_an_edit_keeps_the_edit_in_conflict() {
    let w = World::new();
    let conv = w.conversation();
    assert!(conv.delete("keep.txt"));
    w.push_other(&[("keep.txt", Some(b"their edit\n"))]);
    assert_eq!(
        w.merge(&conv),
        Merged::FastForward {
            conflicts: vec!["keep.txt".to_string()]
        }
    );
    assert_eq!(text(&conv, "keep.txt").as_deref(), Some("their edit\n"));
    assert!(matches!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Conflicts(_)
    ));
    conv.write("keep.txt", "their edit, kept\n".into()).unwrap();
    assert!(conv.conflicts().is_empty());
    w.commit(&conv).unwrap();
    assert_eq!(
        w.origin_file("keep.txt").as_deref(),
        Some("their edit, kept\n")
    );
}

/// **A file both sides added, differently, is marked whole**; the same file
/// added identically on both is simply there.
#[test]
fn a_file_added_on_both_sides_is_marked_whole() {
    let w = World::new();
    let conv = w.conversation();
    conv.write("added.txt", "mine\n".into()).unwrap();
    conv.write("same.txt", "same\n".into()).unwrap();
    w.push_other(&[
        ("added.txt", Some(b"theirs\n")),
        ("same.txt", Some(b"same\n")),
    ]);
    assert_eq!(
        w.merge(&conv),
        Merged::FastForward {
            conflicts: vec!["added.txt".to_string()]
        }
    );
    assert_eq!(
        text(&conv, "added.txt").unwrap(),
        "<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> origin/main\n"
    );
    assert!(!conv.is_modified("same.txt"), "the branch holds it now");
}

/// **What the conversation never touched comes in whole — a binary file
/// included** — and needs no settling.
#[test]
fn untouched_files_come_in_whole_binary_included() {
    let w = World::new();
    let conv = w.conversation();
    conv.write("mine.txt", "mine\n".into()).unwrap();
    w.push_other(&[("logo.bin", Some(&[9, 0, 9, 0]))]);
    assert_eq!(w.merge(&conv), Merged::FastForward { conflicts: vec![] });
    assert_eq!(
        conv.read_bytes("logo.bin").unwrap().unwrap(),
        vec![9, 0, 9, 0]
    );
    assert_eq!(
        conv.status(),
        vec![("mine.txt".to_string(), FileState::Added)]
    );
}

/// **A branch origin has let go of is published again by the next commit**,
/// the conversation's work and all.
#[test]
fn a_branch_origin_let_go_of_is_published_again() {
    let w = World::new();
    let conv = w.conversation();
    conv.write("mine.txt", "mine\n".into()).unwrap();
    git(&w.origin, &["update-ref", "-d", "refs/heads/main"]);
    let landed = w.commit(&conv).unwrap();
    assert_eq!(landed.published, Published::Pushed(PushOutcome::Created));
    assert_eq!(w.on_origin(), landed.commit);
    assert_eq!(w.origin_file("mine.txt").as_deref(), Some("mine\n"));
}

/// **A commit while nothing is uncommitted and no merge is under way still
/// lands, as a commit changing nothing** — whether to make one is the
/// caller's decision, and this layer loses nothing either way.
#[test]
fn a_commit_of_no_change_lands_empty() {
    let w = World::new();
    let conv = w.conversation();
    let before = w.on_origin();
    let landed = w.commit(&conv).unwrap();
    assert_eq!(landed.parents, vec![before.clone()]);
    assert_eq!(
        git(
            &w.origin,
            &["rev-parse", &format!("{}^{{tree}}", landed.commit)]
        ),
        git(&w.origin, &["rev-parse", &format!("{before}^{{tree}}")])
    );
}

/// **A file both sides changed that is not text refuses the merge whole**:
/// nothing in the conversation's copy changes.
#[test]
fn a_binary_changed_on_both_sides_refuses_the_merge() {
    let w = World::new();
    let local_only = w.commit_locally("logo.bin", "\u{0}mine\u{0}");
    let conv = w.conversation();
    assert_eq!(conv.base().unwrap().unwrap().commit(), Some(&local_only));
    w.push_other(&[("logo.bin", Some(&[0, 9, 9, 9, 0]))]);
    let before = conv.snapshot();
    let pulled = w.repo.pull_branch(&branch("main")).unwrap();
    assert!(pulled.diverged);
    let refused = merge_into(&w.repo, &conv, pulled.record().unwrap(), "origin/main");
    let message = refused.unwrap_err().to_string();
    assert!(message.contains("logo.bin"), "{message}");
    assert_eq!(conv.snapshot(), before);
}

// ── merges with history of their own ────────────────────────────────────────

/// **A conversation on a commit origin never had merges origin's commits
/// into a merge being finished**: both parents recorded, the conflict marked
/// from their merge base, and the merge commit — once settled — landing on
/// origin and on the local branch, the local commit kept.
#[test]
fn diverged_history_merges_and_lands_a_merge_commit() {
    let w = World::new();
    let local_only = w.commit_locally("ten.txt", &TEN.replace("l2\n", "local\n"));
    let conv = w.conversation();
    conv.write("new.txt", "new\n".into()).unwrap();
    let theirs = w.push_other(&[
        ("ten.txt", Some(TEN.replace("l2\n", "theirs\n").as_bytes())),
        ("keep.txt", Some(b"theirs\n")),
    ]);

    assert_eq!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Behind {
            record: theirs.clone()
        }
    );
    let merged = w.merge(&conv);
    assert_eq!(
        merged,
        Merged::Merging {
            conflicts: vec!["ten.txt".to_string()]
        }
    );
    let base = conv.base().unwrap().unwrap();
    assert_eq!(base.parents, vec![local_only.clone(), theirs.clone()]);
    assert_eq!(
        text(&conv, "ten.txt").unwrap(),
        TEN.replace(
            "l2\n",
            "<<<<<<< yours\nlocal\n=======\ntheirs\n>>>>>>> origin/main\n"
        )
    );
    assert_eq!(
        text(&conv, "keep.txt").as_deref(),
        Some("theirs\n"),
        "settled by the merge"
    );
    assert_eq!(text(&conv, "new.txt").as_deref(), Some("new\n"));

    conv.write("ten.txt", TEN.replace("l2\n", "both\n"))
        .unwrap();
    let landed = w.commit(&conv).unwrap();
    assert_eq!(landed.parents, vec![local_only.clone(), theirs.clone()]);
    assert_eq!(w.parents(&landed.commit), vec![local_only, theirs]);
    assert_eq!(w.on_origin(), landed.commit);
    assert_eq!(w.local_main(), landed.commit);
    assert_eq!(
        w.origin_file("ten.txt").unwrap(),
        TEN.replace("l2\n", "both\n")
    );
    assert_eq!(w.origin_file("keep.txt").as_deref(), Some("theirs\n"));
    assert_eq!(w.origin_file("new.txt").as_deref(), Some("new\n"));
    assert!(conv.status().is_empty() && conv.conflicts().is_empty());
    assert_eq!(conv.base().unwrap().unwrap().merging(), None);
}

/// **A merge with nothing in conflict still needs its commit**, which lands
/// with both parents even though the conversation changed nothing itself.
#[test]
fn a_clean_merge_is_finished_by_a_commit_of_its_own() {
    let w = World::new();
    let local_only = w.commit_locally("new.txt", "local\n");
    let conv = w.conversation();
    let theirs = w.push_other(&[("keep.txt", Some(b"theirs\n"))]);
    assert_eq!(w.merge(&conv), Merged::Merging { conflicts: vec![] });
    assert!(conv.status().is_empty());
    let landed = w.commit(&conv).unwrap();
    assert_eq!(landed.parents, vec![local_only, theirs]);
    assert_eq!(w.origin_file("new.txt").as_deref(), Some("local\n"));
    assert_eq!(w.origin_file("keep.txt").as_deref(), Some("theirs\n"));
}

/// **Origin moving on while a merge is being finished is merged into it**:
/// the conversation's resolution is kept, the newer commit brought in, and
/// the merge commit names it as the second parent.
#[test]
fn origin_moving_during_a_merge_is_merged_into_it() {
    let w = World::new();
    let local_only = w.commit_locally("ten.txt", &TEN.replace("l2\n", "local\n"));
    let conv = w.conversation();
    w.push_other(&[("ten.txt", Some(TEN.replace("l2\n", "theirs\n").as_bytes()))]);
    w.merge(&conv);
    conv.write("ten.txt", TEN.replace("l2\n", "both\n"))
        .unwrap();
    let later = w.push_other(&[("keep.txt", Some(b"later\n"))]);
    assert!(matches!(
        w.commit(&conv).unwrap_err(),
        NotCommitted::Behind { .. }
    ));

    assert_eq!(w.merge(&conv), Merged::Merging { conflicts: vec![] });
    assert_eq!(
        conv.base().unwrap().unwrap().parents,
        vec![local_only.clone(), later.clone()]
    );
    assert_eq!(
        text(&conv, "ten.txt").unwrap(),
        TEN.replace("l2\n", "both\n"),
        "the resolution kept"
    );
    assert_eq!(text(&conv, "keep.txt").as_deref(), Some("later\n"));
    let landed = w.commit(&conv).unwrap();
    assert_eq!(w.parents(&landed.commit), vec![local_only, later]);
    assert_eq!(
        w.origin_file("ten.txt").unwrap(),
        TEN.replace("l2\n", "both\n")
    );
    assert_eq!(w.origin_file("keep.txt").as_deref(), Some("later\n"));
}

/// **A merge being finished survives a restart**: saved and restored, the
/// conversation reads the same, still in conflict, and its commit still
/// records both parents.
#[test]
fn a_merge_being_finished_survives_a_restart() {
    let w = World::new();
    let local_only = w.commit_locally("ten.txt", &TEN.replace("l2\n", "local\n"));
    let conv = w.conversation();
    let theirs = w.push_other(&[("ten.txt", Some(TEN.replace("l2\n", "theirs\n").as_bytes()))]);
    w.merge(&conv);
    let saved = serde_json::to_string(&conv.snapshot()).unwrap();
    drop(conv);

    let restored = w.conversation();
    restored
        .restore(serde_json::from_str::<Snapshot>(&saved).unwrap())
        .unwrap();
    assert_eq!(restored.conflicts(), ["ten.txt"]);
    assert!(matches!(
        w.commit(&restored).unwrap_err(),
        NotCommitted::Conflicts(_)
    ));
    restored
        .write("ten.txt", TEN.replace("l2\n", "both\n"))
        .unwrap();
    let landed = w.commit(&restored).unwrap();
    assert_eq!(w.parents(&landed.commit), vec![local_only, theirs]);
}

/// **A commit made on this machine and never pushed stays on the local
/// branch** when the conversation — which never had it — commits: origin
/// takes the conversation's commit, the local branch keeps its own.
#[test]
fn a_local_commit_the_conversation_never_had_is_kept() {
    let w = World::new();
    let conv = w.conversation();
    conv.write("mine.txt", "mine\n".into()).unwrap();
    let local_only = w.commit_locally("local.txt", "local\n");
    let landed = w.commit(&conv).unwrap();
    assert_eq!(w.on_origin(), landed.commit);
    assert_eq!(w.local_main(), local_only, "never taken off the branch");
}

// ── no origin ───────────────────────────────────────────────────────────────

/// **Without an origin the local branch is the record**: a commit moves it,
/// another conversation's commit makes the next one wait for a merge, and a
/// merge brings it in.
#[test]
fn without_origin_the_local_branch_is_the_record() {
    let w = World::with_origin(false);
    let (a, b) = (w.conversation(), w.conversation());
    a.write("a.txt", "a\n".into()).unwrap();
    b.write("b.txt", "b\n".into()).unwrap();
    let first = w.commit(&a).unwrap();
    assert_eq!(first.published, Published::Local);
    assert_eq!(w.local_main(), first.commit);
    assert_eq!(
        w.commit(&b).unwrap_err(),
        NotCommitted::Behind {
            record: first.commit.clone()
        }
    );
    assert_eq!(w.merge(&b), Merged::FastForward { conflicts: vec![] });
    let second = w.commit(&b).unwrap();
    assert_eq!(second.parents, vec![first.commit]);
    assert_eq!(w.local_main(), second.commit);
    assert_eq!(git(&w.local, &["show", "main:a.txt"]), "a\n");
    assert_eq!(git(&w.local, &["show", "main:b.txt"]), "b\n");
}

// ── the base a conversation reads ───────────────────────────────────────────

/// **A conversation's base is what the branch held when it first read it**,
/// and a merge is the only way the other writer's commit reaches it.
#[test]
fn a_conversation_reads_its_base_until_it_merges() {
    let w = World::new();
    let conv = w.conversation();
    let first: Base = conv.base().unwrap().unwrap();
    w.push_other(&[("keep.txt", Some(b"theirs\n"))]);
    w.repo.pull_branch(&branch("main")).unwrap();
    assert_eq!(conv.base().unwrap().unwrap(), first);
    assert_eq!(text(&conv, "keep.txt").as_deref(), Some("keep\n"));
    w.merge(&conv);
    assert_eq!(text(&conv, "keep.txt").as_deref(), Some("theirs\n"));
}

/// **A merge into a conversation with nothing uncommitted keeps its own
/// committed side**: where its commit and origin's changed the same lines,
/// both are kept between markers — the conversation's labelled `yours`.
#[test]
fn a_merge_with_nothing_uncommitted_keeps_the_conversations_side() {
    let w = World::new();
    w.commit_locally("ten.txt", &TEN.replace("l2\n", "mine\n"));
    let conv = w.conversation();
    assert!(conv.status().is_empty(), "nothing uncommitted");
    w.push_other(&[("ten.txt", Some(TEN.replace("l2\n", "theirs\n").as_bytes()))]);

    let merged = w.merge(&conv);
    assert_eq!(
        merged,
        Merged::Merging {
            conflicts: vec!["ten.txt".to_string()]
        }
    );
    let ten = text(&conv, "ten.txt").unwrap();
    assert!(ten.contains("<<<<<<< yours\nmine\n"), "{ten}");
    assert!(ten.contains("theirs\n>>>>>>> origin/main"), "{ten}");
}

/// **A merge that has been made already is fast-forwarded to**: origin
/// taking a merge of both sides of the one being finished moves the base to
/// it, carries the conversation's work across, and lets the settled tree go.
#[test]
fn a_merge_made_already_is_fast_forwarded_to() {
    let w = World::new();
    let local_only = w.commit_locally("ten.txt", &TEN.replace("l2\n", "mine\n"));
    let conv = w.conversation();
    w.push_other(&[("ten.txt", Some(TEN.replace("l9\n", "theirs\n").as_bytes()))]);
    assert_eq!(w.merge(&conv), Merged::Merging { conflicts: vec![] });
    conv.write("new.txt", "new\n".into()).unwrap();

    // Someone merges the conversation's commit — published on a side branch
    // — into main themselves.
    git(&w.local, &["push", "-q", "origin", "main:side"]);
    git(&w.other, &["pull", "-q", "--ff-only", "origin", "main"]);
    git(&w.other, &["fetch", "-q", "origin", "side"]);
    git(&w.other, &["merge", "-q", "--no-edit", "FETCH_HEAD"]);
    git(&w.other, &["push", "-q", "origin", "main"]);
    let made = w.oid(&w.other, "HEAD");

    assert!(matches!(w.merge(&conv), Merged::FastForward { .. }));
    assert_eq!(conv.base().unwrap().unwrap().parents, vec![made.clone()]);
    assert_eq!(text(&conv, "new.txt").as_deref(), Some("new\n"));
    assert_eq!(
        git(&w.local, &["for-each-ref", "refs/zend/"]),
        "",
        "the settled tree is let go"
    );
    let landed = w.commit(&conv).unwrap();
    assert_eq!(w.parents(&landed.commit), vec![made]);
    assert_eq!(w.origin_file("new.txt").as_deref(), Some("new\n"));
    let _ = local_only;
}

/// **A branch with no commit yet takes its first**: a root commit, with the
/// conversation's files and no parent.
#[test]
fn an_empty_repository_takes_its_first_commit() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    git(root, &["init", "-q", "--template="]);
    git(root, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    configure(root);
    let repo = Repo::open(root).unwrap();
    let conv = VfsStore::on_branch(GitSource::open(root).unwrap(), Rev::Branch(branch("main")));
    conv.write("first.txt", "first\n".into()).unwrap();

    let main = branch("main");
    let committing = Committing::begin(&repo, &conv, &main).unwrap().unwrap();
    let mut set = ChangeSet::new();
    set.write(
        RepoPath::parse("first.txt").unwrap(),
        b"first\n".to_vec(),
        None,
    )
    .unwrap();
    let me = repo.identity().unwrap();
    let built = repo
        .commit_changes(committing.onto(), &set, "first", &me, &me)
        .unwrap();
    let landed = committing
        .land(&repo, &conv, &built, "first", &me, &me)
        .unwrap()
        .unwrap();
    assert_eq!(landed.parents, Vec::<Oid>::new());
    assert_eq!(landed.published, Published::Local);
    let parents = git(root, &["rev-list", "--parents", "-n", "1", "main"]);
    assert_eq!(parents.split_whitespace().count(), 1, "a root commit");
    assert_eq!(git(root, &["show", "main:first.txt"]), "first\n");
    assert!(conv.status().is_empty(), "committed");
}
