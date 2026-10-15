//! Members sharing one log, each reading a repository through its object
//! database (`docs/substrate_git.md` §12). Origin is a bare repository on disk,
//! and an agent is a checkout of it that answers fetches with bundles and
//! takes pushes as bundles — what the agents do across a cluster.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use substrate_git::{adopt, publish, Alone, Importer, Lease, Log, Records, Store};
use zend_vfs::{
    BranchName, ChangeSet, Fetch, FetchSpec, GitError, GitSource, GitTime, Lease as PushLease, Oid,
    Push, PushResult, PushSpec, RefName, RefUpdate, RemoteName, RemoteRefs, RemoteUrl, Repo,
    RepoPath, Rev, Signature, ORIGIN,
};

/// A log every member reads: blobs, and keyed objects. `hide_blobs` holds
/// back every blob, as replication delivering the entries first would.
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

    fn has_blob(&self, sha256: &[u8; 32]) -> bool {
        !self.hide_blobs.load(Ordering::Relaxed) && self.blobs.lock().unwrap().contains_key(sha256)
    }

    fn put_object(&self, key: &str, value: serde_json::Value) -> anyhow::Result<()> {
        self.objects.lock().unwrap().insert(key.to_string(), value);
        Ok(())
    }

    fn delete_object(&self, key: &str) -> anyhow::Result<()> {
        self.objects.lock().unwrap().remove(key);
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
    std::fs::create_dir_all(dir).unwrap();
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

fn store(log: &Arc<Mem>, name: &str) -> Store {
    Store::open(Arc::clone(log) as Arc<dyn Records>, name).unwrap()
}

/// An agent: a checkout of origin that fetches origin and answers with what
/// the member lacks as a bundle, and that takes a member's push as a bundle
/// and pushes it from here.
struct Agent(PathBuf);

impl Agent {
    fn repo(&self) -> Repo {
        Repo::open(&self.0).unwrap()
    }
}

impl Fetch for Agent {
    fn fetch(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        spec: &FetchSpec,
    ) -> Result<Vec<RefUpdate>, GitError> {
        let agent = self.repo();
        let origin = RemoteName::parse(ORIGIN)?;
        agent.fetch_here(&origin, &FetchSpec::AllBranches)?;
        let mut have = repo.tracking_tips(remote)?;
        have.extend(repo.tracking_tips(&origin)?);
        let fetched = agent.bundle_for_fetch(&origin, spec, &have)?;
        repo.take_fetch(remote, spec, &fetched)
    }

    fn list(&self, _repo: &Repo, _remote: &RemoteName) -> Result<RemoteRefs, GitError> {
        self.repo().ls_remote_here(&RemoteName::parse(ORIGIN)?)
    }
}

impl Push for Agent {
    fn push(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError> {
        let agent = self.repo();
        agent.fetch_here(&RemoteName::parse(ORIGIN)?, &FetchSpec::AllBranches)?;
        if let Some(bundle) = repo.bundle_for_push(remote, specs)? {
            agent.take_bundle(&bundle)?;
        }
        agent.push_here(&RemoteName::parse(ORIGIN)?, specs)
    }
}

/// Origin, a bare repository holding `work`'s `main`, and an agent's
/// checkout of it.
fn origin_and_agent(root: &Path, work: &Path) -> (PathBuf, Agent) {
    let origin = root.join("origin.git");
    git(
        root,
        &[
            "init",
            "-q",
            "--bare",
            "-b",
            "main",
            origin.to_str().unwrap(),
        ],
    );
    git(work, &["push", "-q", origin.to_str().unwrap(), "main"]);
    let agent = root.join("agent");
    git(
        root,
        &[
            "clone",
            "-q",
            origin.to_str().unwrap(),
            agent.to_str().unwrap(),
        ],
    );
    (origin, Agent(agent))
}

/// **A checkout one member adopts reaches another member's database**,
/// objects and branch alike, and a second adoption is a no-op.
#[test]
fn an_adopted_repository_reaches_another_member() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let head = checkout(&src, 3);
    let log = Arc::new(Mem::default());

    let source = Repo::open(&src).unwrap();
    let first = adopt(log.as_ref(), "candle", &source).unwrap();
    assert_eq!(first.packs, 1);
    assert_eq!(first.refs, vec![(main_ref(), Some(head.clone()))]);
    assert_eq!(adopt(log.as_ref(), "candle", &source).unwrap().packs, 0);

    let other = store(&log, "candle");
    let synced = other.sync(log.as_ref()).unwrap();
    assert_eq!((synced.held, synced.listed), (1, 1));
    assert_eq!(branch(&other), Some(head.clone()));
    assert!(other.repo().has_object(&head).unwrap());
}

/// **A ref waits for its packs.** Entries that replicate ahead of their blobs
/// leave the branch unset until the blobs arrive.
#[test]
fn a_ref_waits_for_its_packs() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let head = checkout(&src, 2);
    let log = Arc::new(Mem::default());
    adopt(log.as_ref(), "r", &Repo::open(&src).unwrap()).unwrap();

    log.hide_blobs.store(true, Ordering::Relaxed);
    let s = store(&log, "r");
    let synced = s.sync(log.as_ref()).unwrap();
    assert_eq!((synced.taken, synced.waiting), (0, 1));
    assert_eq!(branch(&s), None);

    log.hide_blobs.store(false, Ordering::Relaxed);
    assert_eq!(s.sync(log.as_ref()).unwrap().taken, 1);
    assert_eq!(branch(&s), Some(head));
}

/// **A pack whose index cannot be read is reported, not fatal.** Every other
/// pack is taken and the sync succeeds, naming the bad one; no branch moves
/// while it stands, since the database could not read the history whole.
#[test]
fn a_pack_that_cannot_be_read_is_reported_and_the_rest_taken() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    checkout(&src, 2);
    let log = Arc::new(Mem::default());
    adopt(log.as_ref(), "r", &Repo::open(&src).unwrap()).unwrap();
    let bad = Log::new(log.as_ref(), "r")
        .unwrap()
        .put_pack(
            b"not a pack".to_vec(),
            b"not an index".to_vec(),
            &"0".repeat(40),
            1,
        )
        .unwrap();

    let s = store(&log, "r");
    let synced = s.sync(log.as_ref()).unwrap();
    assert_eq!(synced.failed.len(), 1);
    assert_eq!(synced.failed[0].0, bad);
    assert_eq!((synced.held, synced.listed), (1, 2));
    assert_eq!(branch(&s), None, "a branch moved over a pack it lacks");
}

/// **Only a descendant replaces a ref**, unless forced.
#[test]
fn only_a_descendant_replaces_a_ref_unless_forced() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let head = checkout(&src, 2);
    let log = Arc::new(Mem::default());
    let source = Repo::open(&src).unwrap();
    adopt(log.as_ref(), "r", &source).unwrap();

    let parent = Oid::parse(&git(&src, &["rev-parse", "HEAD~1"])).unwrap();
    let back = publish(
        log.as_ref(),
        "r",
        &source,
        &[(main_ref(), Some(parent.clone()))],
        false,
    )
    .unwrap();
    assert_eq!(back.refused, vec![(main_ref(), head.clone())]);
    let held = Log::new(log.as_ref(), "r").unwrap().refs().unwrap();
    assert_eq!(held["refs/heads/main"].oid.as_deref(), Some(head.as_str()));

    let forced = publish(
        log.as_ref(),
        "r",
        &source,
        &[(main_ref(), Some(parent.clone()))],
        true,
    )
    .unwrap();
    assert_eq!(forced.refs, vec![(main_ref(), Some(parent))]);
}

/// **A member that missed a forced move follows the commits written on top of
/// it.** `forced` belongs to the one entry that rewrote the branch; the next
/// entry, a plain descendant of the rewrite, is on another line from what the
/// member holds and must still be taken. A move back to an ancestor is not.
#[test]
fn a_member_that_missed_a_forced_move_follows_what_came_after() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let head = checkout(&src, 2);
    let log = Arc::new(Mem::default());
    let source = Repo::open(&src).unwrap();
    adopt(log.as_ref(), "r", &source).unwrap();
    let s = store(&log, "r");
    s.sync(log.as_ref()).unwrap();
    assert_eq!(branch(&s), Some(head.clone()));

    // Origin rewrites main onto the parent, and a commit lands on top; the
    // member is away for both.
    let parent = Oid::parse(&git(&src, &["rev-parse", "HEAD~1"])).unwrap();
    publish(
        log.as_ref(),
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
        log.as_ref(),
        "r",
        &source,
        &[(main_ref(), Some(after.clone()))],
        false,
    )
    .unwrap();
    assert_eq!(moved.refs, vec![(main_ref(), Some(after.clone()))]);

    s.sync(log.as_ref()).unwrap();
    assert_eq!(
        branch(&s),
        Some(after.clone()),
        "stuck on the rewritten-away tip"
    );

    // A stale write back to an ancestor is not taken.
    log.put_object(
        "git/ref/r/refs/heads/main",
        serde_json::json!({ "oid": parent.as_str(), "needs": [], "forced": false }),
    )
    .unwrap();
    s.sync(log.as_ref()).unwrap();
    assert_eq!(branch(&s), Some(after));
}

/// **A ref needs only the packs written with it**, and a member still reads
/// its whole history: sync moves no ref while any pack the log lists is
/// missing.
#[test]
fn a_ref_needs_its_own_packs_and_syncs_whole() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    checkout(&src, 2);
    let log = Arc::new(Mem::default());
    let source = Repo::open(&src).unwrap();
    adopt(log.as_ref(), "r", &source).unwrap();

    std::fs::write(src.join("f"), "more\n").unwrap();
    git(&src, &["commit", "-qam", "more"]);
    let head = Oid::parse(&git(&src, &["rev-parse", "HEAD"])).unwrap();
    let moved = publish(
        log.as_ref(),
        "r",
        &source,
        &[(main_ref(), Some(head.clone()))],
        false,
    )
    .unwrap();
    assert_eq!(moved.packs, 1);

    let packs: BTreeSet<String> = Log::new(log.as_ref(), "r")
        .unwrap()
        .packs()
        .unwrap()
        .into_keys()
        .collect();
    let held = Log::new(log.as_ref(), "r").unwrap().refs().unwrap();
    let needs: BTreeSet<String> = held["refs/heads/main"].needs.iter().cloned().collect();
    assert_eq!(packs.len(), 2);
    assert_eq!(needs.len(), 1);
    assert!(needs.is_subset(&packs));

    let s = store(&log, "r");
    s.sync(log.as_ref()).unwrap();
    assert_eq!(branch(&s), Some(head));
}

/// **A restart loses nothing**: a fresh database over the same log, synced,
/// holds the same branch — there is no folder to keep.
#[test]
fn a_restarted_member_rebuilds_from_the_log() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let head = checkout(&src, 2);
    let log = Arc::new(Mem::default());
    adopt(log.as_ref(), "r", &Repo::open(&src).unwrap()).unwrap();

    store(&log, "r").sync(log.as_ref()).unwrap();
    let again = store(&log, "r");
    again.sync(log.as_ref()).unwrap();
    assert_eq!(branch(&again), Some(head));
}

/// A lease a test switches.
struct Switch(AtomicBool);

impl Lease for Switch {
    fn hold(&self, _key: &str) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// **The importer follows origin through an agent**: a fast-forward is
/// published at once, a rewrite only when two ticks agree, and a member
/// without the lease imports nothing. What the agent answers with goes to the
/// log as a pack.
#[test]
fn the_importer_follows_origin() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    checkout(&work, 1);
    let (origin, agent) = origin_and_agent(tmp.path(), &work);
    let log = Arc::new(Mem::default());
    let url = RemoteUrl::parse(origin.to_str().unwrap()).unwrap();
    let s = store(&log, "r").with_origin(&url, Arc::new(agent)).unwrap();

    let lease = Arc::new(Switch(AtomicBool::new(false)));
    let importer = Importer::new(lease.clone());
    let idle = importer.tick(&s, log.as_ref()).unwrap();
    assert!(!idle.fetched);
    assert_eq!(branch(&s), None);
    lease.0.store(true, Ordering::Relaxed);

    let tip = |dir: &Path| Oid::parse(&git(dir, &["rev-parse", "HEAD"])).unwrap();
    importer.tick(&s, log.as_ref()).unwrap();
    assert_eq!(branch(&s), Some(tip(&work)));
    assert_eq!(
        Log::new(log.as_ref(), "r").unwrap().packs().unwrap().len(),
        1
    );

    std::fs::write(work.join("f"), "more\n").unwrap();
    git(&work, &["commit", "-qam", "more"]);
    git(&work, &["push", "-q", origin.to_str().unwrap(), "main"]);
    importer.tick(&s, log.as_ref()).unwrap();
    assert_eq!(branch(&s), Some(tip(&work)));

    git(&work, &["commit", "-q", "--amend", "-m", "rewritten"]);
    git(&work, &["push", "-qf", origin.to_str().unwrap(), "main"]);
    let before = branch(&s);
    importer.tick(&s, log.as_ref()).unwrap();
    assert_eq!(branch(&s), before, "one look is not enough");
    importer.tick(&s, log.as_ref()).unwrap();
    assert_eq!(branch(&s), Some(tip(&work)));

    // Another member, which never fetched, reads it all from the log.
    let other = store(&log, "r");
    other.sync(log.as_ref()).unwrap();
    assert_eq!(branch(&other), Some(tip(&work)));
}

fn sig() -> Signature {
    Signature::new(
        "Ada",
        "ada@example.com",
        GitTime::parse_raw("1700000000 +0000").unwrap(),
    )
    .unwrap()
}

/// **A member's own commit, pushed through an agent, reaches the others
/// without the lease.** The commit is built in the member's database, the
/// push carries it to the agent as a bundle, landing moves the member's
/// record, and its next tick packs what it made and publishes the branch;
/// another member's database follows.
#[test]
fn a_members_own_push_is_published_without_the_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    checkout(&work, 1);
    let (origin, agent_a) = origin_and_agent(tmp.path(), &work);
    let url = RemoteUrl::parse(origin.to_str().unwrap()).unwrap();
    let log = Arc::new(Mem::default());

    let a = store(&log, "r")
        .with_origin(&url, Arc::new(agent_a))
        .unwrap();
    let b = store(&log, "r");
    Importer::new(Arc::new(Alone))
        .tick(&a, log.as_ref())
        .unwrap();
    let not_holder = Importer::new(Arc::new(Switch(AtomicBool::new(false))));
    not_holder.tick(&b, log.as_ref()).unwrap();
    assert_eq!(branch(&b), branch(&a));

    let base = branch(&b).unwrap();
    let mut change = ChangeSet::new();
    change
        .write(RepoPath::parse("f").unwrap(), b"from b\n".to_vec(), None)
        .unwrap();
    let pushing = Repo::in_substrate(b.objects())
        .unwrap()
        .pushing_through(Arc::new(Agent(tmp.path().join("agent"))));
    let new = pushing
        .commit_changes(&base, &change, "from b", &sig(), &sig())
        .unwrap();
    let pushed = pushing
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
    assert_eq!(
        git(Path::new(origin.to_str().unwrap()), &["rev-parse", "main"]),
        new.as_str(),
        "origin holds the commit b built"
    );

    let ticked = not_holder.tick(&b, log.as_ref()).unwrap();
    assert_eq!(ticked.pushed.refs, vec![(main_ref(), Some(new.clone()))]);
    assert_eq!(b.objects().pending_objects(), 0, "what b made is packed");
    Importer::new(Arc::new(Switch(AtomicBool::new(false))))
        .tick(&a, log.as_ref())
        .unwrap();
    assert_eq!(branch(&a), Some(new));
}

/// **A member alone imports every repository**: `Alone` always holds.
#[test]
fn a_member_alone_holds_every_lease() {
    assert!(Alone.hold("git/anything"));
}

/// **zend can read a repository in the substrate**: it opens as a
/// `GitSource` with no folder, its HEAD is no branch, and its record branches
/// are the log's.
#[test]
fn a_store_reads_as_a_repository_for_zend() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let head = checkout(&src, 2);
    let log = Arc::new(Mem::default());
    adopt(log.as_ref(), "r", &Repo::open(&src).unwrap()).unwrap();
    let s = store(&log, "r");
    s.sync(log.as_ref()).unwrap();

    let repo = Repo::in_substrate(s.objects()).unwrap();
    assert_eq!(repo.folder(), None);
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
    let source = GitSource::of(repo).unwrap();
    assert_eq!(
        source.default_rev(),
        &Rev::Ref(RefName::parse("refs/remotes/origin/main").unwrap())
    );
}
