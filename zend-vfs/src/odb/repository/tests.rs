use std::path::Path;
use std::sync::Arc;

use git2::{BranchType, ObjectType as LibType, Oid as LibOid, Signature, Sort};

use super::*;
use crate::odb::db::DEFAULT_PACK_BUDGET;
use crate::odb::memory_store::MemoryStore;
use crate::odb::pack::fixture::{full_pack, git_bytes, history};
use crate::odb::refs::STORE_HEAD;
use crate::odb::store::PackStore;
use crate::testing::TestRepo;
use crate::types::Oid;

/// A database holding a git repository's every object, and that repository
/// for oracle answers.
fn loaded(commits: usize) -> (ObjectDb, TestRepo, Vec<Oid>) {
    let (t, made) = history(commits);
    let (pack, _) = full_pack(&t);
    let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
    let db = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
    db.take_pack(pack).unwrap();
    (db, t, made)
}

fn lib(oid: &Oid) -> LibOid {
    LibOid::from_str(oid.as_str()).unwrap()
}

/// **libgit2 reads every object through the database**: a commit, its tree
/// walked to every blob, each blob's bytes what git holds.
#[test]
fn libgit2_reads_objects_through_the_database() {
    let (db, t, made) = loaded(6);
    let repo = repository(&db).unwrap();
    let tip = repo.find_commit(lib(&made[5])).unwrap();
    assert_eq!(tip.message(), Some("commit 5\n"));
    assert_eq!(tip.parent_id(0).unwrap(), lib(&made[4]));
    let tree = tip.tree().unwrap();
    let long = tree.get_path(Path::new("long.txt")).unwrap();
    let blob = repo.find_blob(long.id()).unwrap();
    assert_eq!(
        blob.content(),
        git_bytes_of(&t, &format!("{}:long.txt", made[5])).as_slice()
    );
    let mut files = 0;
    tree.walk(git2::TreeWalkMode::PreOrder, |_, entry| {
        if entry.kind() == Some(LibType::Blob) {
            assert!(repo.find_blob(entry.id()).is_ok());
            files += 1;
        }
        git2::TreeWalkResult::Ok
    })
    .unwrap();
    assert_eq!(files, 2 + 6, "long.txt, bin.dat and a file per commit");
}

fn git_bytes_of(t: &TestRepo, spec: &str) -> Vec<u8> {
    git_bytes(&t.path, &["cat-file", "-p", spec], b"")
}

/// **Refs are the database's**: made, read, listed — `HEAD` symbolic to a
/// branch never made — revision syntax and a history walk resolved over them.
#[test]
fn refs_resolve_and_walk_through_the_refdb() {
    let (db, _t, made) = loaded(5);
    let repo = repository(&db).unwrap();
    let head = repo.find_reference("HEAD").unwrap();
    assert_eq!(head.symbolic_target(), Some(STORE_HEAD));
    assert!(repo.head().is_err(), "HEAD names a branch never made");

    repo.reference("refs/remotes/origin/main", lib(&made[4]), false, "")
        .unwrap();
    assert_eq!(
        repo.revparse_single("origin/main~2").unwrap().id(),
        lib(&made[2])
    );
    let mut walk = repo.revwalk().unwrap();
    walk.set_sorting(Sort::TOPOLOGICAL).unwrap();
    walk.push(lib(&made[4])).unwrap();
    assert_eq!(walk.count(), 5);
    let remote: Vec<String> = repo
        .branches(Some(BranchType::Remote))
        .unwrap()
        .map(|b| b.unwrap().0.name().unwrap().unwrap().to_string())
        .collect();
    assert_eq!(remote, vec!["origin/main"]);
    assert!(
        repo.reference("refs/remotes/origin/main", lib(&made[3]), false, "")
            .is_err(),
        "a ref is not replaced unless forced"
    );
}

/// **A transaction is compare-and-swap**: a locked ref cannot be locked
/// again until the transaction ends, and its end writes the ref.
#[test]
fn a_transaction_locks_and_writes() {
    let (db, _t, made) = loaded(3);
    let repo = repository(&db).unwrap();
    let mut txn = repo.transaction().unwrap();
    txn.lock_ref("refs/heads/work").unwrap();
    let other = repository(&db).unwrap();
    let mut second = other.transaction().unwrap();
    assert!(second.lock_ref("refs/heads/work").is_err(), "locked");
    txn.set_target("refs/heads/work", lib(&made[1]), None, "")
        .unwrap();
    txn.commit().unwrap();
    assert_eq!(
        repo.refname_to_id("refs/heads/work").unwrap(),
        lib(&made[1])
    );
    let mut third = other.transaction().unwrap();
    third.lock_ref("refs/heads/work").unwrap();
    third.remove("refs/heads/work").unwrap();
    third.commit().unwrap();
    assert!(repo.refname_to_id("refs/heads/work").is_err());
}

/// **A write lands among the pending objects** and reads back, through a
/// second repository over the same database.
#[test]
fn a_written_commit_is_read_back_from_another_handle() {
    let (db, _t, made) = loaded(2);
    let repo = repository(&db).unwrap();
    let blob = repo.blob(b"written in process\n").unwrap();
    let parent = repo.find_commit(lib(&made[1])).unwrap();
    let mut builder = repo.treebuilder(Some(&parent.tree().unwrap())).unwrap();
    builder.insert("new.txt", blob, 0o100644).unwrap();
    let tree = repo.find_tree(builder.write().unwrap()).unwrap();
    let sig = Signature::new("Ada", "ada@example.com", &git2::Time::new(1_700_000_000, 0)).unwrap();
    let commit = repo
        .commit(None, &sig, &sig, "in process\n", &tree, &[&parent])
        .unwrap();
    assert_eq!(db.pending_objects(), 3, "a blob, a tree and a commit");

    let other = repository(&db).unwrap();
    let read = other.find_commit(commit).unwrap();
    assert_eq!(read.message(), Some("in process\n"));
    assert_eq!(
        other.find_blob(blob).unwrap().content(),
        b"written in process\n"
    );
}

/// **libgit2's merge base, ancestry and diff run over the database** and
/// agree with git's answers for the same history.
#[test]
fn history_algorithms_agree_with_git() {
    let (db, t, made) = loaded(6);
    let repo = repository(&db).unwrap();
    assert_eq!(
        repo.merge_base(lib(&made[5]), lib(&made[2])).unwrap(),
        lib(&made[2])
    );
    assert!(repo
        .graph_descendant_of(lib(&made[5]), lib(&made[1]))
        .unwrap());
    let old = repo.find_commit(lib(&made[1])).unwrap().tree().unwrap();
    let new = repo.find_commit(lib(&made[4])).unwrap().tree().unwrap();
    let diff = repo
        .diff_tree_to_tree(Some(&old), Some(&new), None)
        .unwrap();
    let ours: Vec<String> = diff
        .deltas()
        .map(|d| d.new_file().path().unwrap().to_string_lossy().into_owned())
        .collect();
    let git = t.git(&[
        "diff-tree",
        "-r",
        "--name-only",
        made[1].as_str(),
        made[4].as_str(),
    ]);
    let git: Vec<&str> = git.lines().collect();
    assert_eq!(ours, git);
}
