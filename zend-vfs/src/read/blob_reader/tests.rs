use std::sync::Arc;
use std::time::Instant;

use git2::Oid as LibOid;

use super::*;
use crate::odb::memory_store::MemoryStore;
use crate::odb::pack::fixture::full_pack;
use crate::odb::{PackStore, DEFAULT_PACK_BUDGET};
use crate::testing::TestRepo;
use crate::types::BranchName;

/// A checkout's reader with its child's arguments, deadline and size limit
/// set.
fn batch_reader(
    t: &TestRepo,
    args: &'static [&'static str],
    deadline: Duration,
    max_bytes: u64,
) -> BlobReader {
    BlobReader {
        source: Source::Batch {
            dir: t.path.clone(),
            batch: Mutex::new(None),
            args,
            deadline,
        },
        max_bytes,
    }
}

/// `t`'s every object in a database, as a repository read from the
/// substrate, its refs `t`'s branches.
fn in_database(t: &TestRepo) -> Repo {
    let (pack, _) = full_pack(t);
    let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
    let db = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
    db.take_pack(pack).unwrap();
    let repo = Repo::in_substrate(&db).unwrap();
    for line in t
        .git(&["for-each-ref", "--format=%(objectname) %(refname)"])
        .lines()
    {
        let (oid, name) = line.split_once(' ').unwrap();
        let lib = repo.library().unwrap();
        lib.reference(name, LibOid::from_str(oid).unwrap(), true, "")
            .unwrap();
    }
    repo
}

#[test]
fn headers_parse() {
    assert_eq!(
        parse_header(b"ce013625030ba8dba906f756967f9e9ca394464a blob 6\n", "x").unwrap(),
        Header::Found {
            oid: Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap(),
            kind: "blob".into(),
            size: 6
        }
    );
    assert_eq!(
        parse_header(b"HEAD:x missing\n", "HEAD:x").unwrap(),
        Header::Missing
    );
    assert!(parse_header(b"garbage\n", "x").is_err());
}

/// The fixture both readers are held to: two commits on `main`, an `old`
/// branch at the first, a binary file and a path with spaces and accents.
fn two_commits() -> (TestRepo, Oid) {
    let t = TestRepo::init();
    let binary: Vec<u8> = (0..=255u8).chain(b"\r\n\0\n".iter().copied()).collect();
    t.write("bin.dat", &binary);
    t.write("dir with space/ünïcode.txt", b"first\n");
    let first = t.commit_all("first");
    t.git(&["branch", "old"]);
    t.write("dir with space/ünïcode.txt", b"second\n");
    t.commit_all("second");
    (t, first)
}

fn reads_byte_for_byte(repo: &Repo, first: &Oid) {
    let blobs = repo.blobs();
    let path = RepoPath::parse("dir with space/ünïcode.txt").unwrap();
    let main = Rev::Branch(BranchName::parse("main").unwrap());
    assert_eq!(blobs.read_at(&main, &path).unwrap().unwrap(), b"second\n");
    let old = Rev::Branch(BranchName::parse("old").unwrap());
    assert_eq!(blobs.read_at(&old, &path).unwrap().unwrap(), b"first\n");
    assert_eq!(
        blobs
            .read_at(&Rev::Oid(first.clone()), &path)
            .unwrap()
            .unwrap(),
        b"first\n"
    );
    let binary: Vec<u8> = (0..=255u8).chain(b"\r\n\0\n".iter().copied()).collect();
    assert_eq!(
        blobs
            .read_at(&main, &RepoPath::parse("bin.dat").unwrap())
            .unwrap()
            .unwrap(),
        binary
    );
    let hello = Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap();
    assert_eq!(blobs.read_blob(&hello).unwrap(), None);
    assert_eq!(
        blobs
            .read_at(&main, &RepoPath::parse("nope.rs").unwrap())
            .unwrap(),
        None
    );
    assert!(matches!(
        blobs.read_at(&main, &RepoPath::parse("dir with space").unwrap()),
        Err(GitError::NotABlob { .. })
    ));
    let (commit, _) = blobs.commit_of(&old).unwrap().unwrap();
    assert_eq!(&commit, first);
    let none = Rev::Branch(BranchName::parse("nope").unwrap());
    assert_eq!(blobs.commit_of(&none).unwrap(), None);
}

#[test]
fn a_checkout_reads_at_any_revision_byte_for_byte() {
    let (t, first) = two_commits();
    reads_byte_for_byte(&t.repo(), &first);
}

/// **A repository read from the substrate reads exactly as its checkout**:
/// every question above, answered by libgit2 over the database.
#[test]
fn a_repository_in_the_substrate_reads_as_its_checkout_does() {
    let (t, first) = two_commits();
    reads_byte_for_byte(&in_database(&t), &first);
}

#[test]
fn a_revision_resolves_to_its_commit_and_tree() {
    let t = TestRepo::init();
    let repo = t.repo();
    let blobs = repo.blobs();
    assert_eq!(blobs.commit_of(&Rev::Head).unwrap(), None, "no commit yet");
    t.write("a.txt", b"a\n");
    let first = t.commit_all("first");
    let main = Rev::Branch(BranchName::parse("main").unwrap());
    let (commit, tree) = blobs.commit_of(&main).unwrap().unwrap();
    assert_eq!(commit, first);
    assert_eq!(tree.as_str(), t.git(&["rev-parse", "main^{tree}"]).trim());

    t.write("a.txt", b"b\n");
    let second = t.commit_all("second");
    let (moved, _) = blobs.commit_of(&main).unwrap().unwrap();
    assert_eq!(moved, second, "the same reader sees the branch move");
}

#[test]
fn a_dead_child_is_restarted() {
    let t = TestRepo::init();
    t.write("a.txt", b"a\n");
    t.commit_all("base");
    let blobs = t.repo().blobs();
    let a = RepoPath::parse("a.txt").unwrap();
    assert!(blobs.read_at(&Rev::Head, &a).unwrap().is_some());
    {
        let Source::Batch { batch, .. } = &blobs.source else {
            panic!("a checkout reads through a child");
        };
        let mut guard = batch.lock().unwrap();
        let child = guard.as_mut().unwrap();
        child.tree.kill();
        let _ = child.child.wait();
    }
    assert_eq!(blobs.read_at(&Rev::Head, &a).unwrap().unwrap(), b"a\n");
}

/// **An oversized object is refused from its header**, and the next read
/// still works — of a checkout, on a fresh child; of the database, at once.
#[test]
fn an_object_over_the_limit_is_refused_and_the_reader_recovers() {
    let t = TestRepo::init();
    t.write("big.txt", b"0123456789\n");
    t.write("small.txt", b"ok\n");
    t.commit_all("base");
    let in_db = in_database(&t);
    let readers = [
        batch_reader(&t, &["--batch"], NETWORK_TIMEOUT, 4),
        BlobReader {
            max_bytes: 4,
            ..in_db.blobs()
        },
    ];
    for blobs in readers {
        let main = Rev::Branch(BranchName::parse("main").unwrap());
        match blobs.read_at(&main, &RepoPath::parse("big.txt").unwrap()) {
            Err(GitError::BlobTooLarge { size, limit, .. }) => assert_eq!((size, limit), (11, 4)),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            blobs
                .read_at(&main, &RepoPath::parse("small.txt").unwrap())
                .unwrap()
                .unwrap(),
            b"ok\n"
        );
    }
}

/// **A read that never answers times out**, killing the child. With
/// `--buffer`, `cat-file` holds its answers until its input ends, which a
/// reader never does — a stand-in for a lazy fetch that stalls.
#[test]
fn a_read_that_never_answers_times_out() {
    let t = TestRepo::init();
    t.write("a.txt", b"a\n");
    t.commit_all("base");
    let blobs = batch_reader(
        &t,
        &["--batch", "--buffer"],
        Duration::from_millis(500),
        MAX_BLOB_BYTES,
    );
    let started = Instant::now();
    let got = blobs.read_at(&Rev::Head, &RepoPath::parse("a.txt").unwrap());
    assert!(matches!(got, Err(GitError::Timeout { .. })), "{got:?}");
    assert!(started.elapsed() < Duration::from_secs(10));
    let Source::Batch { batch, .. } = &blobs.source else {
        panic!("a checkout reads through a child");
    };
    assert!(
        batch.lock().unwrap().is_none(),
        "the stuck child was dropped"
    );
}
