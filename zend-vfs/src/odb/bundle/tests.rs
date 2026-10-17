use std::sync::Arc;

use super::*;
use crate::odb::db::{ObjectDb, DEFAULT_PACK_BUDGET};
use crate::odb::memory_store::MemoryStore;
use crate::odb::pack::fixture::{full_pack, git_bytes, history};
use crate::odb::repository::repository;
use crate::odb::store::PackStore;
use crate::testing::TestRepo;
use crate::types::Oid;

fn raw(oid: &Oid) -> RawId {
    RawId::from_oid(oid).unwrap()
}

/// **A bundle cut here is one git takes**: `git bundle verify` passes on a
/// repository holding only its prerequisite, `unbundle` brings the tip in,
/// and the prerequisite named is exactly the commit the bundle's first commit
/// is a child of.
#[test]
fn a_bundle_cut_here_is_one_git_takes() {
    let (t, made) = history(6);
    let (pack, _) = full_pack(&t);
    let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
    let db = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
    db.take_pack(pack).unwrap();
    let repo = repository(&db).unwrap();

    let bundle = cut(
        &repo,
        &[(raw(&made[5]), "refs/heads/main".to_string())],
        &[raw(&made[2])],
    )
    .unwrap()
    .expect("three commits beyond the hidden one");
    let parsed = parse(&bundle).unwrap();
    assert_eq!(parsed.prerequisites, vec![raw(&made[2])]);
    assert_eq!(
        parsed.refs,
        vec![(raw(&made[5]), "refs/heads/main".to_string())]
    );

    t.git(&["branch", "base-only", made[2].as_str()]);
    let other = TestRepo::init();
    other.git(&["fetch", "-q", t.path.to_str().unwrap(), "base-only"]);
    let file = other.path.join("taken.bundle");
    std::fs::write(&file, &bundle).unwrap();
    other.git(&["bundle", "verify", "-q", file.to_str().unwrap()]);
    let listed = other.git(&["bundle", "unbundle", file.to_str().unwrap()]);
    assert!(listed.contains(made[5].as_str()), "{listed}");
    other.git(&["cat-file", "-e", made[5].as_str()]);
}

/// **A bundle git cuts reads here**: its prerequisite, its ref, its pack.
#[test]
fn a_bundle_git_cut_parses() {
    let (t, made) = history(5);
    let bundle = git_bytes(
        &t.path,
        &[
            "bundle",
            "create",
            "-q",
            "-",
            "main",
            &format!("^{}", made[1]),
        ],
        b"",
    );
    let parsed = parse(&bundle).unwrap();
    assert_eq!(parsed.prerequisites, vec![raw(&made[1])]);
    assert_eq!(
        parsed.refs,
        vec![(raw(&made[4]), "refs/heads/main".to_string())]
    );
    assert_eq!(&parsed.pack[..4], b"PACK");
}

/// **Nothing beyond what is hidden is no bundle.**
#[test]
fn a_tip_already_hidden_cuts_no_bundle() {
    let (t, made) = history(3);
    let (pack, _) = full_pack(&t);
    let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
    let db = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
    db.take_pack(pack).unwrap();
    let repo = repository(&db).unwrap();
    let tip = raw(&made[2]);
    assert_eq!(
        cut(&repo, &[(tip, "refs/heads/main".into())], &[tip]).unwrap(),
        None
    );
}

#[test]
fn a_header_that_is_not_a_bundles_is_refused() {
    assert!(parse(b"# v9 git bundle\n\nPACK").is_err());
    assert!(parse(b"# v2 git bundle\nzz refs/heads/x\n\n").is_err());
    assert!(parse(b"# v2 git bundle\n").is_err(), "no blank line");
}
