//! The merge table against a real record: a bare `origin`, the mind folder
//! cloned from it, and a second clone standing for another member publishing.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use tempfile::TempDir;
use zend_vfs::{fetch_here, GitError, Oid, Push, PushResult, PushSpec, RemoteName, Repo};

use super::follow::{follow, Followed};
use super::publish::Publish;
use super::status::Status;
use super::sync::Synced;
use super::NOT_UNDER_GIT;
use crate::opstore::owner::Owner;
use crate::opstore::shelf::StoreShelf;
use crate::opstore::MindStores;
use crate::sim::shelf::Shelf;

const WORLD: &str = "vault";

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

/// The record: `origin` holding `main` with one era, the mind folder and
/// another member's clone of it, and the stores over the mind.
struct Record {
    _dir: TempDir,
    origin: PathBuf,
    mind: PathBuf,
    other: PathBuf,
    stores: Arc<MindStores>,
}

const ERA: &str = "layers/eras/third.md";
const ERA_TEXT: &str = "# The third era\n\nIt burned in the spring.\n\nNobody rebuilt it.\n";

fn record() -> Record {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let seed = root.join("seed");
    std::fs::create_dir_all(seed.join("layers/eras")).unwrap();
    git(&seed, &["init", "-q", "--template="]);
    git(&seed, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    configure(&seed);
    std::fs::write(seed.join(ERA), ERA_TEXT).unwrap();
    git(&seed, &["add", "-A"]);
    git(&seed, &["commit", "-q", "-m", "seed"]);
    git(root, &["clone", "-q", "--bare", "seed", "origin"]);
    for clone in ["mind", "other"] {
        // Checked out as committed: a clone made under a global
        // `core.autocrlf` would read as changed from the first status.
        git(
            root,
            &["clone", "-q", "-c", "core.autocrlf=false", "origin", clone],
        );
        configure(&root.join(clone));
    }
    let stores = MindStores::new(Some(&root.join("mind")));
    // Held in memory, as a disconnected world's are: no substrate needed.
    stores.set_quiet(Box::new(|_| true));
    Record {
        origin: root.join("origin"),
        mind: root.join("mind"),
        other: root.join("other"),
        stores,
        _dir: dir,
    }
}

impl Record {
    /// Operation `op`, opened on the record as it stands.
    fn open(&self, op: u64) -> StoreShelf {
        self.stores.pin(&Owner::operation(WORLD, op));
        StoreShelf {
            stores: Arc::clone(&self.stores),
            owner: Owner::operation(WORLD, op),
        }
    }

    /// Another member publishes `path` as `text`.
    fn publish_other(&self, path: &str, text: &str) {
        git(&self.other, &["pull", "-q", "--ff-only", "origin", "main"]);
        let full = self.other.join(path);
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, text).unwrap();
        git(&self.other, &["add", "-A"]);
        git(&self.other, &["commit", "-q", "-m", "elsewhere"]);
        git(&self.other, &["push", "-q", "origin", "main"]);
    }

    /// `path` as the record holds it.
    fn on_record(&self, path: &str) -> Option<String> {
        let listed = git(&self.origin, &["ls-tree", "--name-only", "main", path]);
        (!listed.trim().is_empty()).then(|| git(&self.origin, &["show", &format!("main:{path}")]))
    }

    fn status(&self, op: u64) -> Status {
        self.stores
            .merge(&Owner::operation(WORLD, op), |m| m.status())
            .unwrap()
            .unwrap()
            .unwrap()
    }

    fn publish(&self, op: u64) -> Publish {
        self.stores
            .merge(&Owner::operation(WORLD, op), |m| {
                m.publish("Operation Test: publish it\n")
            })
            .unwrap()
            .unwrap()
    }

    fn sync(&self, op: u64) -> Synced {
        self.stores
            .merge(&Owner::operation(WORLD, op), |m| m.sync())
            .unwrap()
            .unwrap()
    }
}

/// **An operation's store lands on the record as one commit**, and the mind
/// folder — what the watcher ingests — is brought up to it.
#[test]
fn a_store_lands_on_the_record_and_the_mind_follows() {
    let r = record();
    let shelf = r.open(1);
    let story = "layers/stories/the-gate.md";
    shelf
        .commit(&[(
            story.into(),
            Some("# The Gate\n\nA night at the gate.\n".into()),
        )])
        .unwrap();
    assert!(matches!(
        r.status(1),
        Status::Changes { record_moved: false, ref conflicts, .. } if conflicts.is_empty()
    ));
    assert!(r.status(1).render().contains("`merge_publish` lands it"));

    let Publish::Landed { commit, paths } = r.publish(1) else {
        panic!("not landed");
    };
    assert_eq!(paths, [story]);
    assert_eq!(
        r.on_record(story).as_deref(),
        Some("# The Gate\n\nA night at the gate.\n")
    );
    assert_eq!(git(&r.origin, &["rev-parse", "main"]).trim(), commit);
    assert_eq!(
        git(&r.origin, &["log", "-1", "--format=%B", "main"]).trim(),
        "Operation Test: publish it"
    );
    assert_eq!(
        r.status(1),
        Status::Nothing,
        "the store holds nothing unpublished"
    );
    assert_eq!(r.publish(1), Publish::Nothing);
    assert!(
        !r.mind.join(story).exists(),
        "the checkout has not moved yet"
    );

    assert_eq!(
        follow(&r.mind, fetch_here()).unwrap(),
        Some(Followed::Moved {
            to: Oid::parse(&commit).unwrap()
        })
    );
    assert_eq!(
        std::fs::read_to_string(r.mind.join(story)).unwrap(),
        "# The Gate\n\nA night at the gate.\n"
    );
    assert_eq!(
        follow(&r.mind, fetch_here()).unwrap(),
        Some(Followed::Level)
    );
}

/// **A record that moved only at other lines is taken in as the work lands**:
/// the work, uncommitted on its base, is carried onto the record's commit and
/// lands on top of it, and the record keeps both changes.
#[test]
fn a_record_moved_elsewhere_is_taken_in_as_the_work_lands() {
    let r = record();
    let shelf = r.open(1);
    shelf
        .commit(&[(
            ERA.into(),
            Some(ERA_TEXT.replace("in the spring", "in the autumn")),
        )])
        .unwrap();
    r.publish_other("layers/stories/elsewhere.md", "Somebody else's story.\n");
    let elsewhere = git(&r.origin, &["rev-parse", "main"]).trim().to_string();
    let Status::Changes {
        record_moved,
        conflicts,
        ..
    } = r.status(1)
    else {
        panic!("no changes");
    };
    assert!(record_moved);
    assert!(conflicts.is_empty());
    assert!(r.status(1).render().contains("at lines it does not touch"));

    let Publish::Landed { commit, .. } = r.publish(1) else {
        panic!("not landed");
    };
    let parents = git(&r.origin, &["rev-list", "--parents", "-n", "1", &commit]);
    assert_eq!(parents.trim(), format!("{commit} {elsewhere}"));
    assert!(r.on_record(ERA).unwrap().contains("in the autumn"));
    assert_eq!(
        r.on_record("layers/stories/elsewhere.md").as_deref(),
        Some("Somebody else's story.\n")
    );
}

/// **An overlap is refused with nothing changed; a sync marks it in the
/// document; once settled at the bench, it lands.**
#[test]
fn an_overlap_is_refused_marked_by_a_sync_and_lands_once_settled() {
    let r = record();
    let shelf = r.open(1);
    let ours = ERA_TEXT.replace("in the spring", "in the autumn");
    shelf.commit(&[(ERA.into(), Some(ours.clone()))]).unwrap();
    r.publish_other(ERA, &ERA_TEXT.replace("in the spring", "in the winter"));

    let Status::Changes { conflicts, .. } = r.status(1) else {
        panic!("no changes");
    };
    assert_eq!(conflicts, [ERA]);
    assert_eq!(r.publish(1), Publish::Conflicts(vec![ERA.to_string()]));
    assert_eq!(
        shelf.read(ERA).as_deref(),
        Some(ours.as_str()),
        "nothing changed"
    );
    assert!(r.on_record(ERA).unwrap().contains("in the winter"));

    assert_eq!(
        r.sync(1),
        Synced::Merged {
            conflicts: vec![ERA.to_string()]
        }
    );
    let marked = shelf.read(ERA).unwrap();
    assert!(marked.contains("<<<<<<< yours"), "{marked}");
    assert!(marked.contains("in the autumn") && marked.contains("in the winter"));
    assert!(marked.contains(">>>>>>> the record"), "{marked}");
    assert_eq!(r.publish(1), Publish::Unsettled(vec![ERA.to_string()]));

    let settled = ERA_TEXT.replace("in the spring", "in the autumn, and again in the winter");
    shelf
        .commit(&[(ERA.into(), Some(settled.clone()))])
        .unwrap();
    assert!(matches!(r.publish(1), Publish::Landed { .. }));
    assert_eq!(r.on_record(ERA).as_deref(), Some(settled.as_str()));
}

/// **The same text published twice merges as a no-op** — an operation carried
/// to its publish on both sides of a partition (§9.4).
#[test]
fn the_same_text_published_twice_is_a_no_op() {
    let r = record();
    let story = "layers/stories/the-gate.md";
    let text = "# The Gate\n\nA night at the gate.\n";
    let here = r.open(1);
    let there = r.open(2);
    here.commit(&[(story.into(), Some(text.into()))]).unwrap();
    there.commit(&[(story.into(), Some(text.into()))]).unwrap();
    let Publish::Landed { commit, .. } = r.publish(1) else {
        panic!("not landed");
    };
    assert_eq!(r.publish(2), Publish::Nothing);
    assert_eq!(git(&r.origin, &["rev-parse", "main"]).trim(), commit);
    assert_eq!(r.on_record(story).as_deref(), Some(text));
}

/// A push that cannot reach its agent.
struct NoAgent;

impl Push for NoAgent {
    fn push(
        &self,
        _repo: &Repo,
        _remote: &RemoteName,
        _specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError> {
        Err(GitError::Io(std::io::Error::other("no agent can push")))
    }
}

/// **A publish that cannot reach git changes nothing**, and says so as
/// unreachable — the step waits rather than failing.
#[test]
fn a_publish_with_no_agent_waits_and_changes_nothing() {
    let r = record();
    r.stores.set_pusher(Arc::new(NoAgent));
    let shelf = r.open(1);
    let story = "layers/stories/the-gate.md";
    shelf
        .commit(&[(story.into(), Some("A night at the gate.\n".into()))])
        .unwrap();
    let Publish::Unreachable(why) = r.publish(1) else {
        panic!("not unreachable");
    };
    assert!(why.contains("no agent can push"), "{why}");
    assert_eq!(r.on_record(story), None);
    assert_eq!(shelf.read(story).as_deref(), Some("A night at the gate.\n"));
    assert!(matches!(r.status(1), Status::Changes { .. }));
}

/// **The mind folder follows another member's publish**, and a folder with
/// changes of its own on disk is left as it is.
#[test]
fn the_mind_follows_the_record_unless_it_has_changes_of_its_own() {
    let r = record();
    r.publish_other("layers/stories/elsewhere.md", "Somebody else's story.\n");
    assert!(matches!(
        follow(&r.mind, fetch_here()).unwrap(),
        Some(Followed::Moved { .. })
    ));
    assert_eq!(
        std::fs::read_to_string(r.mind.join("layers/stories/elsewhere.md")).unwrap(),
        "Somebody else's story.\n"
    );

    std::fs::write(r.mind.join(ERA), "edited by hand\n").unwrap();
    r.publish_other("layers/stories/more.md", "More.\n");
    assert_eq!(
        follow(&r.mind, fetch_here()).unwrap(),
        Some(Followed::Dirty(vec![ERA.to_string()]))
    );
    assert_eq!(
        std::fs::read_to_string(r.mind.join(ERA)).unwrap(),
        "edited by hand\n"
    );
    assert!(!r.mind.join("layers/stories/more.md").exists());
}

/// A mind folder that is not a git repository has no record to land on.
#[test]
fn a_mind_not_under_git_has_no_record() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("mind/layers")).unwrap();
    let stores = MindStores::new(Some(&dir.path().join("mind")));
    stores.set_quiet(Box::new(|_| true));
    assert_eq!(
        stores
            .merge(&Owner::operation(WORLD, 1), |m| m.status())
            .unwrap()
            .err(),
        Some(NOT_UNDER_GIT.to_string())
    );
}
