//! Two members sharing one log, with real repositories on disk
//! (`docs/substrate_git.md` §12).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use substrate_git::{adopt, publish, Alone, Importer, Lease, Log, PackEntry, Records, Store};
use zend_vfs::{
    fetch_here, BranchName, GitError, Lease as PushLease, Oid, Push, PushResult, PushSpec, RefName,
    RemoteName, RemoteUrl, Repo,
};

/// A log both members read: blobs, and keyed objects. `hide_blobs` holds back
/// every blob, as replication delivering the entries first would.
#[derive(Default)]
struct Mem {
    blobs: Mutex<BTreeMap<[u8; 32], Vec<u8>>>,
    objects: Mutex<BTreeMap<String, serde_json::Value>>,
    hide_blobs: AtomicBool,
}

impl Records for Mem {
    fn put_blob(&self, bytes: Vec<u8>) -> anyhow::Result<[u8; 32]> {
        // Any stable name will do here: the real log names a blob by its
        // SHA-256; this one by a hash of the bytes and their length.
        let mut sha = [0u8; 32];
        let h = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ *b as u64).wrapping_mul(0x100000001b3)
        });
        sha[..8].copy_from_slice(&h.to_be_bytes());
        sha[8..16].copy_from_slice(&(bytes.len() as u64).to_be_bytes());
        self.blobs.lock().unwrap().insert(sha, bytes);
        Ok(sha)
    }

    fn blob(&self, sha256: &[u8; 32]) -> anyhow::Result<Option<Vec<u8>>> {
        if self.hide_blobs.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(self.blobs.lock().unwrap().get(sha256).cloned())
    }

    fn put_object(&self, key: &str, value: serde_json::Value) -> anyhow::Result<()> {
        self.objects.lock().unwrap().insert(key.to_string(), value);
        Ok(())
    }

    fn objects_under(&self, prefix: &str) -> Vec<(String, serde_json::Value)> {
        self.objects
            .lock()
            .unwrap()
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .expect("git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

/// A working tree with `commits` commits on `main`.
fn checkout(dir: &Path, commits: usize) -> Oid {
    git(dir, &["init", "-q", "-b", "main"]);
    for i in 0..commits {
        std::fs::write(dir.join("f"), format!("{i}\n")).unwrap();
        git(dir, &["add", "-A"]);
        git(dir, &["commit", "-q", "-m", &format!("c{i}")]);
    }
    Oid::parse(&git(dir, &["rev-parse", "HEAD"])).unwrap()
}

fn main_ref() -> RefName {
    RefName::parse("refs/heads/main").unwrap()
}

fn branch(store: &Store) -> Option<Oid> {
    store.record().unwrap().get(&main_ref()).cloned()
}

/// **A checkout one member adopts reaches another member's store**, objects
/// and branch alike, and a second adoption is a no-op.
#[test]
fn an_adopted_repository_reaches_another_member() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let head = checkout(&src, 3);
    let log = Mem::default();

    let source = Repo::open(&src).unwrap();
    let first = adopt(&log, "candle", &source).unwrap();
    assert_eq!(first.packs, 1);
    assert_eq!(first.refs, vec![(main_ref(), Some(head.clone()))]);
    assert_eq!(adopt(&log, "candle", &source).unwrap().packs, 0);

    let other = Store::open(&tmp.path().join("member-b"), "candle").unwrap();
    let synced = other.sync(&log).unwrap();
    assert_eq!(synced.taken, 1);
    assert_eq!(branch(&other), Some(head.clone()));
    assert!(other.repo().has_object(&head).unwrap());
}

/// **A ref waits for its packs.** Entries that replicate ahead of their blobs
/// leave the branch unset until the blob arrives.
#[test]
fn a_ref_waits_for_its_packs() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let head = checkout(&src, 2);
    let log = Mem::default();
    adopt(&log, "r", &Repo::open(&src).unwrap()).unwrap();

    let store = Store::open(&tmp.path().join("m"), "r").unwrap();
    log.hide_blobs.store(true, Ordering::Relaxed);
    let synced = store.sync(&log).unwrap();
    assert_eq!((synced.taken, synced.waiting), (0, 1));
    assert_eq!(branch(&store), None);

    log.hide_blobs.store(false, Ordering::Relaxed);
    store.sync(&log).unwrap();
    assert_eq!(branch(&store), Some(head));
}

/// **A pack that cannot be taken is reported, not fatal.** Every other pack
/// is still taken and the sync succeeds, naming the bad one; no branch moves
/// while it stands, since the store could not read the history whole.
#[test]
fn a_pack_that_cannot_be_taken_is_reported_and_the_rest_taken() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    checkout(&src, 2);
    let log = Mem::default();
    adopt(&log, "r", &Repo::open(&src).unwrap()).unwrap();
    let bad = Log::new(&log, "r")
        .unwrap()
        .put_pack(
            b"not a pack".to_vec(),
            &PackEntry {
                checksum: "0".repeat(40),
                objects: 1,
            },
        )
        .unwrap();

    let store = Store::open(&tmp.path().join("m"), "r").unwrap();
    let synced = store.sync(&log).unwrap();
    assert_eq!(synced.taken, 1);
    assert_eq!(synced.failed.len(), 1);
    assert_eq!(synced.failed[0].0, bad);
    assert_eq!((synced.held, synced.listed), (1, 2));
    assert_eq!(
        branch(&store),
        None,
        "a branch moved over a pack the store lacks"
    );
}

/// **Only a descendant replaces a ref**, unless forced.
#[test]
fn only_a_descendant_replaces_a_ref_unless_forced() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let head = checkout(&src, 2);
    let log = Mem::default();
    let source = Repo::open(&src).unwrap();
    adopt(&log, "r", &source).unwrap();

    let parent = Oid::parse(&git(&src, &["rev-parse", "HEAD~1"])).unwrap();
    let back = publish(
        &log,
        "r",
        &source,
        &[(main_ref(), Some(parent.clone()))],
        false,
    )
    .unwrap();
    assert_eq!(back.refused, vec![(main_ref(), head.clone())]);
    let held = Log::new(&log, "r").unwrap().refs().unwrap();
    assert_eq!(held["refs/heads/main"].oid.as_deref(), Some(head.as_str()));

    let forced = publish(
        &log,
        "r",
        &source,
        &[(main_ref(), Some(parent.clone()))],
        true,
    )
    .unwrap();
    assert_eq!(forced.refs, vec![(main_ref(), Some(parent))]);
}

/// **A store that missed a forced move follows the commits written on top of
/// it.** `forced` belongs to the one entry that rewrote the branch; the next
/// entry, a plain descendant of the rewrite, is on another line from what the
/// store holds and must still be taken. A move back to an ancestor is not.
#[test]
fn a_store_that_missed_a_forced_move_follows_what_came_after() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let head = checkout(&src, 2);
    let log = Mem::default();
    let source = Repo::open(&src).unwrap();
    adopt(&log, "r", &source).unwrap();
    let store = Store::open(&tmp.path().join("m"), "r").unwrap();
    store.sync(&log).unwrap();
    assert_eq!(branch(&store), Some(head.clone()));

    // Origin rewrites main onto the parent, and a commit lands on top; the
    // store is away for both.
    let parent = Oid::parse(&git(&src, &["rev-parse", "HEAD~1"])).unwrap();
    publish(
        &log,
        "r",
        &source,
        &[(main_ref(), Some(parent.clone()))],
        true,
    )
    .unwrap();
    git(&src, &["reset", "-q", "--hard", parent.as_str()]);
    std::fs::write(src.join("f"), "rewritten\n").unwrap();
    git(&src, &["commit", "-qam", "on the rewrite"]);
    let after = Oid::parse(&git(&src, &["rev-parse", "HEAD"])).unwrap();
    let moved = publish(
        &log,
        "r",
        &source,
        &[(main_ref(), Some(after.clone()))],
        false,
    )
    .unwrap();
    assert_eq!(moved.refs, vec![(main_ref(), Some(after.clone()))]);

    store.sync(&log).unwrap();
    assert_eq!(
        branch(&store),
        Some(after.clone()),
        "stuck on the rewritten-away tip"
    );

    // A stale write back to an ancestor is not taken.
    log.put_object(
        "git/ref/r/refs/heads/main",
        serde_json::json!({ "oid": parent.as_str(), "needs": [], "forced": false }),
    )
    .unwrap();
    store.sync(&log).unwrap();
    assert_eq!(branch(&store), Some(after));
}

/// **A ref needs only the packs written with it**, and a member still reads its
/// whole history: sync moves no ref while any pack the log lists is missing.
#[test]
fn a_ref_needs_its_own_packs_and_syncs_whole() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    checkout(&src, 2);
    let log = Mem::default();
    let source = Repo::open(&src).unwrap();
    adopt(&log, "r", &source).unwrap();

    std::fs::write(src.join("f"), "more\n").unwrap();
    git(&src, &["commit", "-qam", "more"]);
    let head = Oid::parse(&git(&src, &["rev-parse", "HEAD"])).unwrap();
    let moved = publish(
        &log,
        "r",
        &source,
        &[(main_ref(), Some(head.clone()))],
        false,
    )
    .unwrap();
    assert_eq!(moved.packs, 1);

    let packs: BTreeSet<String> = Log::new(&log, "r")
        .unwrap()
        .packs()
        .unwrap()
        .into_keys()
        .collect();
    let held = Log::new(&log, "r").unwrap().refs().unwrap();
    let needs: BTreeSet<String> = held["refs/heads/main"].needs.iter().cloned().collect();
    assert_eq!(packs.len(), 2);
    assert_eq!(needs.len(), 1);
    assert!(needs.is_subset(&packs));

    let store = Store::open(&tmp.path().join("m"), "r").unwrap();
    store.sync(&log).unwrap();
    assert_eq!(branch(&store), Some(head));
}

/// **Deleting the store loses nothing**: reopened and synced, it holds the
/// same branch.
#[test]
fn a_deleted_store_is_rebuilt_from_the_log() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let head = checkout(&src, 2);
    let log = Mem::default();
    adopt(&log, "r", &Repo::open(&src).unwrap()).unwrap();

    let root = tmp.path().join("m");
    Store::open(&root, "r").unwrap().sync(&log).unwrap();
    std::fs::remove_dir_all(root.join("r")).unwrap();
    let store = Store::open(&root, "r").unwrap();
    store.sync(&log).unwrap();
    assert_eq!(branch(&store), Some(head));
}

/// A lease a test switches.
struct Switch(AtomicBool);

impl Lease for Switch {
    fn hold(&self, _key: &str) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// **The importer follows origin**: a fast-forward is published at once, a
/// rewrite only when two ticks agree, and a member without the lease imports
/// nothing.
#[test]
fn the_importer_follows_origin() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir(&work).unwrap();
    checkout(&work, 1);
    let origin = tmp.path().join("origin.git");
    git(
        tmp.path(),
        &[
            "init",
            "-q",
            "--bare",
            "-b",
            "main",
            origin.to_str().unwrap(),
        ],
    );
    git(&work, &["push", "-q", origin.to_str().unwrap(), "main"]);

    let log = Mem::default();
    let url = RemoteUrl::parse(origin.to_str().unwrap()).unwrap();
    let store = Store::open(&tmp.path().join("m"), "r")
        .unwrap()
        .with_origin(&url, fetch_here())
        .unwrap();

    let lease = Arc::new(Switch(AtomicBool::new(false)));
    let importer = Importer::new(lease.clone());
    let idle = importer.tick(&store, &log).unwrap();
    assert!(!idle.fetched);
    assert_eq!(branch(&store), None);
    lease.0.store(true, Ordering::Relaxed);

    let tip = |dir: &Path| Oid::parse(&git(dir, &["rev-parse", "HEAD"])).unwrap();
    importer.tick(&store, &log).unwrap();
    assert_eq!(branch(&store), Some(tip(&work)));

    std::fs::write(work.join("f"), "more\n").unwrap();
    git(&work, &["commit", "-qam", "more"]);
    git(&work, &["push", "-q", origin.to_str().unwrap(), "main"]);
    importer.tick(&store, &log).unwrap();
    assert_eq!(branch(&store), Some(tip(&work)));

    git(&work, &["commit", "-q", "--amend", "-m", "rewritten"]);
    git(&work, &["push", "-qf", origin.to_str().unwrap(), "main"]);
    let before = branch(&store);
    importer.tick(&store, &log).unwrap();
    assert_eq!(branch(&store), before, "one look is not enough");
    importer.tick(&store, &log).unwrap();
    assert_eq!(branch(&store), Some(tip(&work)));
}

/// Pushes from the agent's own checkout: origin moves, and nothing in the
/// store the push was asked of does.
struct FromAgent(PathBuf);

impl Push for FromAgent {
    fn push(
        &self,
        _repo: &Repo,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError> {
        Repo::open(&self.0)?.push_here(remote, specs)
    }
}

/// **A member's own push reaches the others without the lease.** The push
/// moves its record as it lands; its next tick publishes that, and another
/// member's store follows.
#[test]
fn a_members_own_push_is_published_without_the_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir(&work).unwrap();
    checkout(&work, 1);
    let origin = tmp.path().join("origin.git");
    git(
        tmp.path(),
        &[
            "init",
            "-q",
            "--bare",
            "-b",
            "main",
            origin.to_str().unwrap(),
        ],
    );
    git(&work, &["push", "-q", origin.to_str().unwrap(), "main"]);
    let url = RemoteUrl::parse(origin.to_str().unwrap()).unwrap();
    let log = Mem::default();

    let a = Store::open(&tmp.path().join("a"), "r")
        .unwrap()
        .with_origin(&url, fetch_here())
        .unwrap();
    let b = Store::open(&tmp.path().join("b"), "r").unwrap();
    Importer::new(Arc::new(Alone)).tick(&a, &log).unwrap();
    let not_holder = Importer::new(Arc::new(Switch(AtomicBool::new(false))));
    not_holder.tick(&b, &log).unwrap();
    assert_eq!(branch(&b), branch(&a));

    // b holds a commit on top of the record — built in its store, as zend's
    // write tools build one — and pushes it through an agent, whose push
    // touches none of b's refs. Landing moves b's record.
    let base = branch(&b).unwrap();
    std::fs::write(work.join("f"), "from b\n").unwrap();
    git(&work, &["commit", "-qam", "from b"]);
    let new = Oid::parse(&git(&work, &["rev-parse", "HEAD"])).unwrap();
    git(
        &work,
        &[
            "push",
            "-q",
            b.dir().to_str().unwrap(),
            "HEAD:refs/heads/scratch",
        ],
    );
    git(
        &work,
        &["remote", "add", "origin", origin.to_str().unwrap()],
    );
    let pushed = Repo::open(b.dir())
        .unwrap()
        .pushing_through(Arc::new(FromAgent(work.clone())))
        .push(
            &RemoteName::parse("origin").unwrap(),
            &[PushSpec::branch(
                new.clone(),
                BranchName::parse("main").unwrap(),
                PushLease::Expect(base),
            )],
        )
        .unwrap();
    assert!(pushed[0].accepted(), "{pushed:?}");
    assert_eq!(
        branch(&b),
        Some(new.clone()),
        "the landed push moved b's record"
    );

    let ticked = not_holder.tick(&b, &log).unwrap();
    assert_eq!(ticked.pushed.refs, vec![(main_ref(), Some(new.clone()))]);
    Importer::new(Arc::new(Switch(AtomicBool::new(false))))
        .tick(&a, &log)
        .unwrap();
    assert_eq!(branch(&a), Some(new));
}

/// **A member alone imports every repository**: `Alone` always holds.
#[test]
fn a_member_alone_holds_every_lease() {
    assert!(Alone.hold("git/anything"));
}

/// **zend can read a store**: it opens as a `GitSource`, its HEAD is no
/// branch, and its record branches are the log's.
#[test]
fn a_store_reads_as_a_repository_for_zend() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let head = checkout(&src, 2);
    let log = Mem::default();
    adopt(&log, "r", &Repo::open(&src).unwrap()).unwrap();
    let store = Store::open(&tmp.path().join("m"), "r").unwrap();
    store.sync(&log).unwrap();

    let repo = Repo::open(store.dir()).unwrap();
    assert_eq!(
        repo.head()
            .unwrap()
            .branch()
            .map(|b| b.as_str().to_string())
            .as_deref(),
        Some("substrate-store")
    );
    let record = repo.record_branches().unwrap();
    assert_eq!(record.len(), 1);
    assert_eq!(record[0].name.as_str(), "main");
    assert_eq!(record[0].tip, head);
    zend_vfs::GitSource::open(store.dir()).unwrap();
}
