//! Diaries in the store, against real repositories (`docs/private_mode.md`
//! §5.2, §10): written as one neutral commit the tree never reads, pushed
//! with the rest, carried across a replay, and — sealed — leaving no topic
//! and no entry's text anywhere in the repository's history.

use super::diff::Changes;
use super::tests::{git, note, Repos};
use super::Store;
use crate::diaries::book::{Author, Diary};
use crate::diaries::crypto::{seal, Dek, KdfParams, Kek};
use crate::diaries::layout::{new_salt, DiaryId, SALT};
use crate::ops::fixture::{now, yen};

const FAST: KdfParams = KdfParams {
    m_kib: 64,
    t: 1,
    p: 1,
};

/// Synthetic markers: none of them may ever be found in the clear.
const TOPIC: &str = "SYNTHETIC-TOPIC-QUOKKA-7731";
const TEXT: &str = "SYNTHETIC-ENTRY-WOMBAT-4410";

/// A diary on [`TOPIC`] holding [`TEXT`], sealed under `pass` with `salt`:
/// its id and its file.
fn sealed(pass: &str, salt: &[u8]) -> (DiaryId, Vec<u8>) {
    let mut d = Diary::new(TOPIC, now());
    d.write(now(), Author::User, TEXT);
    let id = DiaryId::random();
    let kek = Kek::derive(pass, salt, FAST).unwrap();
    (id, seal(&kek, &Dek::random(), &id, &d.to_bytes()))
}

fn changes(salt: &[u8], id: &DiaryId, file: &[u8]) -> Changes {
    Changes::from([
        (SALT.to_string(), Some(salt.to_vec())),
        (id.path(), Some(file.to_vec())),
    ])
}

/// **A diary is one commit saying nothing**, the tree untouched and with no
/// problem from the files it never reads.
#[test]
fn a_diary_is_one_commit_the_tree_never_reads() {
    let r = Repos::new();
    let dir = r.clone("a");
    let mut s = Store::open(&dir).unwrap();
    let before = s.tree().clone();
    let salt = new_salt();
    let (id, file) = sealed("synthetic pass", &salt);
    s.write_diaries(&changes(&salt, &id, &file)).unwrap();
    assert_eq!(s.tree(), &before, "the tree does not read diaries/");
    assert!(s.tree().problems.is_empty(), "{:?}", s.tree().problems);
    assert_eq!(s.diary_files(), vec![(id, file.as_slice())]);
    assert_eq!(s.diary_salt(), Some(salt.as_slice()));
    let head = s.head().to_string();
    assert_eq!(
        git(&dir, &["log", "-1", "--format=%an <%ae>|%B", &head]),
        "journal service <journals@tokera>|diaries: update\n\nJournal-Diaries: update"
    );
    assert!(
        s.write_diaries(&changes(&salt, &id, &file)).is_err(),
        "nothing changed"
    );
    let stray = Changes::from([("groups/zen/x.bin".to_string(), Some(vec![1]))]);
    assert!(s.write_diaries(&stray).is_err(), "only diaries' files");
    s.sync().unwrap();
    let reopened = Store::open(&r.clone("b")).unwrap();
    assert_eq!(reopened.diary_file(&id), Some(file.as_slice()));
    assert!(reopened.tree().problems.is_empty());
}

/// **A diary written while origin moved is replayed onto it**, its file
/// whole, beside the newer origin's own writes.
#[test]
fn a_diary_commit_is_replayed_whole_onto_a_moved_origin() {
    let r = Repos::new();
    let mut a = Store::open(&r.clone("a")).unwrap();
    let mut b = Store::open(&r.clone("b")).unwrap();
    b.write(&yen(), &note("Origin moved first."), now())
        .unwrap();
    b.sync().unwrap();
    let salt = new_salt();
    let (id, file) = sealed("synthetic pass", &salt);
    a.write_diaries(&changes(&salt, &id, &file)).unwrap();
    let synced = a.sync().unwrap();
    assert_eq!(synced.replayed, 1);
    assert!(synced.dropped.is_empty(), "{:?}", synced.dropped);
    let check = Store::open(&r.clone("check")).unwrap();
    assert_eq!(check.diary_file(&id), Some(file.as_slice()));
    let deleted = Changes::from([(id.path(), None)]);
    a.write_diaries(&deleted).unwrap();
    a.sync().unwrap();
    assert_eq!(
        Store::open(&r.clone("after")).unwrap().diary_file(&id),
        None
    );
}

/// **The salt is written once**: a write replacing it is refused, and a
/// replay meeting another salt on origin is dropped and said so — never
/// copied over the one origin's diaries are stretched with.
#[test]
fn the_salt_is_never_replaced() {
    let r = Repos::new();
    let mut a = Store::open(&r.clone("a")).unwrap();
    let mut b = Store::open(&r.clone("b")).unwrap();
    let (salt_a, salt_b) = (new_salt(), new_salt());
    let (id_b, file_b) = sealed("synthetic pass", &salt_b);
    b.write_diaries(&changes(&salt_b, &id_b, &file_b)).unwrap();
    b.sync().unwrap();
    let (id_a, file_a) = sealed("synthetic pass", &salt_a);
    a.write_diaries(&changes(&salt_a, &id_a, &file_a)).unwrap();
    let synced = a.sync().unwrap();
    assert_eq!(synced.dropped.len(), 1, "{:?}", synced.dropped);
    let check = Store::open(&r.clone("check")).unwrap();
    assert_eq!(
        check.diary_salt(),
        Some(salt_b.as_slice()),
        "origin's salt stands"
    );
    assert_eq!(check.diary_file(&id_b), Some(file_b.as_slice()));
    let replaced = Changes::from([(SALT.to_string(), Some(salt_a.to_vec()))]);
    assert!(b.write_diaries(&replaced).is_err());
    let removed = Changes::from([(SALT.to_string(), None)]);
    assert!(b.write_diaries(&removed).is_err());
}

/// **No topic and no entry's text is anywhere in the repository**: every
/// object of the agent's checkout and of origin, every commit of history,
/// read whole — after a diary was written, rewritten, a journal note made
/// beside it, and all of it pushed.
#[test]
fn no_topic_or_text_is_anywhere_in_history() {
    let r = Repos::new();
    let dir = r.clone("a");
    let mut s = Store::open(&dir).unwrap();
    let salt = new_salt();
    let (id, file) = sealed("synthetic pass", &salt);
    s.write_diaries(&changes(&salt, &id, &file)).unwrap();
    s.write(&yen(), &note("An ordinary note."), now()).unwrap();
    let (_, again) = sealed("synthetic pass", &salt);
    s.write_diaries(&Changes::from([(id.path(), Some(again))]))
        .unwrap();
    s.sync().unwrap();
    for repo in [&dir, &r.origin] {
        let every = git(repo, &["cat-file", "--batch-all-objects", "--batch"]);
        assert!(every.contains("diaries: update"), "the scan reads commits");
        for marker in [TOPIC, TEXT] {
            assert!(!every.contains(marker), "{marker} in {}", repo.display());
        }
        let log = git(repo, &["log", "--all", "--format=%B"]);
        assert!(!log.contains(&id.to_string()), "no diary's id in a message");
    }
}
