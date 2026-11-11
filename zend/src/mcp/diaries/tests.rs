//! Diaries against a real memory on a real journal repository
//! (`docs/private_mode.md` §10), their passphrases stretched at a cheap cost:
//! the production cost has its own test in `journals::diaries::crypto`.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use journals::diaries::book::Author;
use journals::diaries::crypto::{self, unwrap, KdfParams, Kek};
use journals::diaries::layout::SALT;
use journals::time::Stamp;
use mcp_server::refs::REF_IDLE_MS;
use tempfile::TempDir;

use super::{Diaries, MOVED, NOT_OPEN};
use crate::mcp::memory::tests::{opened, repo};
use crate::mcp::memory::Memory;

const FAST: KdfParams = KdfParams {
    m_kib: 64,
    t: 1,
    p: 1,
};

const NOW: u64 = 1_794_000_000_000;

/// Synthetic markers: none may ever be found in the clear.
const TOPIC: &str = "SYNTHETIC-TOPIC-PLATYPUS-2207";
const TEXT: &str = "SYNTHETIC-ENTRY-ECHIDNA-9134";
const PASS: &str = "synthetic passphrase one";
const OTHER: &str = "synthetic passphrase two";

fn at(s: &str) -> Stamp {
    Stamp::parse(s).unwrap()
}

fn memory() -> (TempDir, Arc<Memory>) {
    let (root, dir) = repo();
    (root, opened(dir))
}

fn diaries() -> Diaries {
    Diaries::with(FAST)
}

/// Every object of the repository at `dir`, read whole.
fn every_object(dir: &Path) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args(["cat-file", "--batch-all-objects", "--batch"])
        .output()
        .expect("git runs");
    assert!(out.status.success());
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// **A passphrase nothing is under opens nothing and makes nothing**: no
/// salt, no file, no commit.
#[test]
fn an_unknown_passphrase_opens_nothing_and_creates_nothing() {
    let (_root, m) = memory();
    let d = diaries();
    assert_eq!(d.open(&m, "o", "c1", PASS, NOW), Ok(0));
    let (files, salt) = m.diaries().unwrap();
    assert!(files.is_empty() && salt.is_none(), "nothing was written");
    assert!(d.is_open("c1"), "the passphrase is open to create with");
    assert!(d.open(&m, "o", "c1", "short", NOW).is_err(), "too short");
}

/// **A create must repeat the open passphrase**, so a typo makes no
/// orphan; a diary on a topic already open is refused.
#[test]
fn a_create_repeats_the_open_passphrase() {
    let (_root, m) = memory();
    let d = diaries();
    assert_eq!(d.create(&m, "c1", TOPIC, PASS, NOW), Err(NOT_OPEN.into()));
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    assert!(d
        .create(&m, "c1", TOPIC, "synthetic passphrase oen", NOW)
        .is_err());
    assert!(m.diaries().unwrap().0.is_empty(), "the typo made nothing");
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    let (files, salt) = m.diaries().unwrap();
    assert_eq!(files.len(), 1);
    assert!(salt.is_some(), "the first diary wrote the salt");
    assert!(
        d.create(&m, "c1", TOPIC, PASS, NOW).is_err(),
        "one diary a topic"
    );
}

/// **Two passphrases' diaries are invisible to each other**, and a
/// conversation sees only what its own passphrase opened.
#[test]
fn two_passphrases_diaries_are_invisible_to_each_other() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.create(&m, "c1", "synthetic one", PASS, NOW).unwrap();
    d.open(&m, "o", "c2", OTHER, NOW).unwrap();
    d.create(&m, "c2", "synthetic two", OTHER, NOW).unwrap();
    let topics = |c: &str| -> Vec<String> {
        d.list(&m, c, NOW)
            .unwrap()
            .into_iter()
            .map(|l| l.topic)
            .collect()
    };
    assert_eq!(topics("c1"), ["synthetic one"]);
    assert_eq!(topics("c2"), ["synthetic two"]);
    assert_eq!(d.open(&m, "o", "c3", PASS, NOW), Ok(1));
    assert_eq!(topics("c3"), ["synthetic one"]);
    assert!(d.read(&m, "c3", "synthetic two", NOW).is_err());
}

/// **An entry written about a past moment reads in its place**, deletes
/// stand after a reopen, and compaction — many deleted, one written at the
/// first's moment — reads back as meant.
#[test]
fn entries_go_in_time_order_and_deletes_and_compaction_stand() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    let w = |ts: &str, text: &str| {
        d.write(&m, "c1", TOPIC, at(ts), Author::User, text, NOW)
            .unwrap()
    };
    let a = w("2026-11-10T09:00:00Z", "monday");
    let b = w("2026-11-12T09:00:00Z", "wednesday");
    let c = w("2026-11-11T09:00:00Z", "tuesday, written last");
    let texts = |c: &str| -> Vec<String> {
        d.read(&m, c, TOPIC, NOW)
            .unwrap()
            .entries
            .into_iter()
            .map(|e| e.text)
            .collect()
    };
    assert_eq!(
        texts("c1"),
        ["monday", "tuesday, written last", "wednesday"]
    );
    assert_eq!(d.delete(&m, "c1", TOPIC, &[b.clone()], NOW), Ok(1));
    assert!(d.delete(&m, "c1", TOPIC, &[b], NOW).is_err(), "gone");
    assert!(d.close("c1"));
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    assert_eq!(
        texts("c1"),
        ["monday", "tuesday, written last"],
        "after a reopen"
    );
    // Compaction: both folded into one, at the first's moment.
    assert_eq!(d.delete(&m, "c1", TOPIC, &[a, c], NOW), Ok(2));
    w("2026-11-10T09:00:00Z", "monday and tuesday, in one");
    w("2026-11-13T09:00:00Z", "thursday");
    assert_eq!(texts("c1"), ["monday and tuesday, in one", "thursday"]);
}

/// **After rotation the new passphrase opens the diary and the old does
/// not.**
#[test]
fn a_rotated_diary_opens_under_the_new_passphrase_only() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    d.write(
        &m,
        "c1",
        TOPIC,
        at("2026-11-10T09:00:00Z"),
        Author::User,
        TEXT,
        NOW,
    )
    .unwrap();
    assert!(d
        .rotate(&m, "c1", TOPIC, OTHER, "synthetic passphrase three", NOW)
        .is_err());
    d.rotate(&m, "c1", TOPIC, PASS, "synthetic passphrase three", NOW)
        .unwrap();
    assert_eq!(d.open(&m, "o", "c2", PASS, NOW), Ok(0), "not under the old");
    assert_eq!(
        d.open(&m, "o", "c3", "synthetic passphrase three", NOW),
        Ok(1)
    );
    assert_eq!(d.read(&m, "c3", TOPIC, NOW).unwrap().entries[0].text, TEXT);
}

/// **A diary moved to a new passphrase stays moved**: another conversation
/// that had it open under the old one may not write it back, and is told.
#[test]
fn a_rotation_is_not_undone_by_another_conversation() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    d.open(&m, "o", "c2", PASS, NOW).unwrap();
    d.rotate(&m, "c1", TOPIC, PASS, OTHER, NOW).unwrap();
    let ts = at("2026-11-10T09:00:00Z");
    assert_eq!(
        d.write(&m, "c2", TOPIC, ts, Author::User, TEXT, NOW),
        Err(MOVED.into())
    );
    assert!(d.read(&m, "c2", TOPIC, NOW).is_err(), "closed there");
    assert_eq!(
        d.open(&m, "o", "c3", OTHER, NOW),
        Ok(1),
        "still the new one's"
    );
    assert_eq!(d.open(&m, "o", "c4", PASS, NOW), Ok(0));
}

/// **A passphrase's topics are one each, across conversations**, and a
/// diary another conversation made is listed without reopening.
#[test]
fn a_topic_is_kept_once_under_a_passphrase() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.open(&m, "o", "c2", PASS, NOW).unwrap();
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    assert!(d.create(&m, "c2", TOPIC, PASS, NOW).is_err(), "c1 made it");
    let listed: Vec<String> = d
        .list(&m, "c2", NOW)
        .unwrap()
        .into_iter()
        .map(|l| l.topic)
        .collect();
    assert_eq!(listed, [TOPIC]);
    assert_eq!(m.diaries().unwrap().0.len(), 1);
}

/// **What is unlocked goes on close, and when the ref has gone idle**.
#[test]
fn unlock_state_is_cleared_on_close_and_on_expiry() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.open(&m, "o", "c2", PASS, NOW).unwrap();
    assert!(d.close("c1"));
    assert!(!d.is_open("c1"));
    assert_eq!(d.list(&m, "c1", NOW), Err(NOT_OPEN.into()));
    let later = NOW + REF_IDLE_MS;
    assert_eq!(d.sweep(later - 1, |_| None), 0, "still within its hours");
    assert_eq!(
        d.sweep(later + 5, |_| Some(later)),
        0,
        "its ref was used since"
    );
    assert_eq!(d.sweep(later + REF_IDLE_MS, |_| Some(later)), 1);
    assert!(!d.is_open("c2"));
}

/// **A large diary reads back whole**: five megabytes, every byte.
#[test]
fn a_five_megabyte_diary_reads_back_whole() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    let big: String = (0..5 * 1024 * 1024 / 64)
        .map(|i| format!("{i:063}\n"))
        .collect();
    d.write(
        &m,
        "c1",
        TOPIC,
        at("2026-11-10T09:00:00Z"),
        Author::User,
        &big,
        NOW,
    )
    .unwrap();
    let read = d.read(&m, "c1", TOPIC, NOW).unwrap();
    assert_eq!(read.entries[0].text.len(), big.len());
    assert_eq!(read.entries[0].text, big);
}

/// **Opens that find nothing wait longer each time**, per account.
#[test]
fn opens_that_find_nothing_back_off() {
    let (_root, m) = memory();
    let d = diaries();
    for _ in 0..3 {
        assert_eq!(d.open(&m, "o", "c1", PASS, NOW), Ok(0));
    }
    let refused = d.open(&m, "o", "c1", PASS, NOW).unwrap_err();
    assert!(
        refused.starts_with("too many opens found nothing"),
        "{refused}"
    );
    assert_eq!(d.open(&m, "someone-else", "c9", PASS, NOW), Ok(0));
}

/// **A rotated diary's later versions open under no old key**: the old
/// passphrase's file in history gives up nothing written since.
#[test]
fn a_rotation_seals_under_a_new_key() {
    let (_root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    let (files, _) = m.diaries().unwrap();
    let (id, before) = files[0].clone();
    d.rotate(&m, "c1", TOPIC, PASS, OTHER, NOW).unwrap();
    let after = m.diary_file(&id).unwrap().unwrap();
    let (_, salt) = m.diaries().unwrap();
    let old_kek = Kek::derive(PASS, &salt.unwrap(), FAST).unwrap();
    let old_dek = unwrap(&old_kek, &id, &before).expect("the old file opens under the old");
    assert!(
        crypto::open(&old_dek, &id, &after).is_err(),
        "the old key opens nothing new"
    );
}

/// **No topic and no entry's text is anywhere in the repository**, every
/// object read whole, after diaries were made, written and rewritten.
#[test]
fn no_topic_or_text_is_anywhere_in_the_repository() {
    let (root, m) = memory();
    let d = diaries();
    d.open(&m, "o", "c1", PASS, NOW).unwrap();
    d.create(&m, "c1", TOPIC, PASS, NOW).unwrap();
    for i in 0..3 {
        d.write(
            &m,
            "c1",
            TOPIC,
            at("2026-11-10T09:00:00Z"),
            Author::User,
            &format!("{TEXT} {i}"),
            NOW,
        )
        .unwrap();
    }
    for dir in [root.path().join("agent"), root.path().join("origin.git")] {
        let every = every_object(&dir);
        for marker in [TOPIC, TEXT] {
            assert!(!every.contains(marker), "{marker} in {}", dir.display());
        }
    }
    assert!(m.diaries().unwrap().1.is_some(), "the salt is {SALT}");
}
