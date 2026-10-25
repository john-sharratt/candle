use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::{file_store_key, Substrate};
use tempfile::TempDir;

use super::owner::Owner;
use super::shelf::StoreShelf;
use super::*;
use crate::sim::shelf::Shelf;

/// A mind folder with one era, and a substrate beside it.
fn setup() -> (TempDir, SharedSubstrate) {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("mind/layers/eras")).unwrap();
    std::fs::write(dir.path().join("mind/layers/eras/a.md"), "the era\n").unwrap();
    let mut substrate = Substrate::new();
    let p =
        SubstratePersistence::open_in_with_substrate(&dir.path().join("substrate"), &mut substrate)
            .unwrap();
    (dir, SharedSubstrate::new(substrate, p))
}

fn stores(dir: &TempDir, shared: &SharedSubstrate) -> Arc<MindStores> {
    let s = MindStores::new(Some(&dir.path().join("mind")));
    s.set_substrate(shared.clone());
    s
}

fn shelf(stores: &Arc<MindStores>, op: u64) -> StoreShelf {
    StoreShelf {
        stores: Arc::clone(stores),
        owner: Owner::operation("vault", op),
    }
}

fn author(stores: &Arc<MindStores>, account: &str) -> StoreShelf {
    StoreShelf {
        stores: Arc::clone(stores),
        owner: Owner::author(account),
    }
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@t"])
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {out:?}");
}

/// The mind folder of [`setup`], made a git repository with its era committed.
fn under_git(dir: &TempDir) {
    let mind = dir.path().join("mind");
    git(&mind, &["init", "-q", "-b", "main"]);
    git(&mind, &["add", "-A"]);
    git(&mind, &["commit", "-q", "-m", "first"]);
}

/// An operation's timeline is one name everywhere, and another operation's,
/// or another world's, is another.
#[test]
fn an_operations_timeline_is_its_world_and_id() {
    let t = |w, op| Owner::operation(w, op).timeline();
    assert_eq!(t("vault", 3), t("vault", 3));
    assert_ne!(t("vault", 3), t("vault", 4));
    assert_ne!(t("vault", 3), t("keep", 3));
}

/// **A draft committed in one step is read in the next — by a store opened
/// afresh, as another member would** — and nothing an operation writes
/// reaches the mind folder.
#[test]
fn a_draft_is_kept_as_the_operations_events_and_never_reaches_the_mind() {
    let (dir, shared) = setup();
    let writer = stores(&dir, &shared);
    shelf(&writer, 1)
        .commit(&[(
            "layers/stories/s.md".into(),
            Some("a night at the gate\n".into()),
        )])
        .unwrap();
    assert!(!dir.path().join("mind/layers/stories/s.md").exists());

    let reader = stores(&dir, &shared);
    let next = shelf(&reader, 1);
    assert_eq!(
        next.read("layers/stories/s.md").as_deref(),
        Some("a night at the gate\n")
    );
    assert_eq!(next.read("layers/eras/a.md").as_deref(), Some("the era\n"));
    assert_eq!(
        next.list("layers/stories").into_iter().collect::<Vec<_>>(),
        ["s.md"]
    );
}

/// **A draft outlives a restart.** Committed, then the substrate closed and
/// opened again from its folder, as a daemon's restart does: the operation's
/// store still holds it. On WW a committed draft read back as no events at all
/// after a restart, and the table called its operation off as "document gone".
#[test]
fn a_draft_is_read_back_after_the_substrate_reopens() {
    let (dir, shared) = setup();
    shelf(&stores(&dir, &shared), 1)
        .commit(&[(
            "layers/stories/s.md".into(),
            Some("a night at the gate\n".into()),
        )])
        .unwrap();
    drop(shared);

    let mut substrate = Substrate::new();
    let p =
        SubstratePersistence::open_in_with_substrate(&dir.path().join("substrate"), &mut substrate)
            .unwrap();
    let reopened = SharedSubstrate::new(substrate, p);
    assert_eq!(
        shelf(&stores(&dir, &reopened), 1)
            .read("layers/stories/s.md")
            .as_deref(),
        Some("a night at the gate\n")
    );
}

/// **Two operations over one base see different files.**
#[test]
fn two_operations_see_their_own_files() {
    let (dir, shared) = setup();
    let s = stores(&dir, &shared);
    shelf(&s, 1)
        .commit(&[("layers/eras/a.md".into(), Some("one's\n".into()))])
        .unwrap();
    shelf(&s, 2)
        .commit(&[("layers/eras/a.md".into(), Some("two's\n".into()))])
        .unwrap();
    assert_eq!(
        shelf(&s, 1).read("layers/eras/a.md").as_deref(),
        Some("one's\n")
    );
    assert_eq!(
        shelf(&s, 2).read("layers/eras/a.md").as_deref(),
        Some("two's\n")
    );
    assert_eq!(
        shelf(&s, 3).read("layers/eras/a.md").as_deref(),
        Some("the era\n")
    );
}

/// **An author's drafts are theirs alone**, and an operation's store over the
/// same mind sees none of them.
#[test]
fn an_authors_drafts_are_their_own() {
    let (dir, shared) = setup();
    let s = stores(&dir, &shared);
    author(&s, "ab")
        .commit(&[("worlds/w.yaml".into(), Some("name: W\n".into()))])
        .unwrap();
    assert!(!dir.path().join("mind/worlds/w.yaml").exists());
    assert_eq!(
        author(&stores(&dir, &shared), "ab")
            .read("worlds/w.yaml")
            .as_deref(),
        Some("name: W\n")
    );
    assert_eq!(author(&s, "cd").read("worlds/w.yaml"), None);
    assert_eq!(shelf(&s, 1).read("worlds/w.yaml"), None);
}

/// **An author with no drafts reads the record as it stands now**; one with
/// drafts reads it at the commit the drafts were made on.
#[test]
fn an_empty_authors_store_follows_the_record() {
    let (dir, shared) = setup();
    under_git(&dir);
    let s = stores(&dir, &shared);
    assert!(s.under_git());
    let mind = dir.path().join("mind");
    assert_eq!(
        author(&s, "ab").read("layers/eras/a.md").as_deref(),
        Some("the era\n")
    );
    author(&s, "cd")
        .commit(&[("worlds/w.yaml".into(), Some("name: W\n".into()))])
        .unwrap();

    std::fs::write(mind.join("layers/eras/a.md"), "the second era\n").unwrap();
    git(&mind, &["commit", "-q", "-am", "second"]);

    assert_eq!(
        author(&s, "ab").read("layers/eras/a.md").as_deref(),
        Some("the second era\n")
    );
    assert_eq!(
        author(&s, "cd").read("layers/eras/a.md").as_deref(),
        Some("the era\n")
    );
    s.discard(&Owner::author("cd"));
    assert_eq!(
        author(&s, "cd").read("layers/eras/a.md").as_deref(),
        Some("the second era\n")
    );
}

/// A mind folder that is not a git repository has no record to land on.
#[test]
fn a_plain_folder_is_not_under_git() {
    let (dir, shared) = setup();
    assert!(!stores(&dir, &shared).under_git());
    assert!(!MindStores::new(None).under_git());
}

/// **A discarded store is gone**, events and all: the next read is the
/// mind as it stands.
#[test]
fn a_discarded_store_is_gone() {
    let (dir, shared) = setup();
    let s = stores(&dir, &shared);
    shelf(&s, 1)
        .commit(&[("layers/eras/a.md".into(), Some("failed\n".into()))])
        .unwrap();
    s.discard(&Owner::operation("vault", 1));
    assert_eq!(
        shelf(&stores(&dir, &shared), 1)
            .read("layers/eras/a.md")
            .as_deref(),
        Some("the era\n")
    );
}

/// **A store is declared at its first save and stays declared**, so no rewrite
/// takes its events for an orphan's; its discard does not release it, since a
/// release landing on another member after its next save there would leave that
/// save undeclared. A start declares a store written before declarations were,
/// but not one discarded since: an empty store has nothing to carry.
#[test]
fn a_store_is_declared_and_its_discard_releases_nothing() {
    let (dir, shared) = setup();
    let s = stores(&dir, &shared);
    let (one, two) = (Owner::operation("vault", 1), Owner::operation("vault", 2));
    let key = |o: &Owner| file_store_key(o.timeline());
    shelf(&s, 1)
        .commit(&[("layers/eras/a.md".into(), Some("kept\n".into()))])
        .unwrap();
    assert!(
        shared.custom_object(&key(&one)).is_some(),
        "declared at save"
    );
    s.discard(&one);
    assert!(
        shared.custom_object(&key(&one)).is_some(),
        "a discard releases nothing"
    );

    // A store saved before declarations were: events, and no declaration.
    shelf(&s, 2)
        .commit(&[("layers/eras/a.md".into(), Some("old\n".into()))])
        .unwrap();
    shared
        .put_custom_object(CustomObjectPayload::ceased(key(&two)))
        .unwrap();
    stores(&dir, &shared);
    assert!(
        shared.custom_object(&key(&two)).is_some(),
        "a start declares a store with live events"
    );

    // A discarded store with no declaration is not brought back as one.
    s.discard(&two);
    shared
        .put_custom_object(CustomObjectPayload::ceased(key(&two)))
        .unwrap();
    stores(&dir, &shared);
    assert!(
        shared.custom_object(&key(&two)).is_none(),
        "an empty store is not declared"
    );
}

/// **In disconnected mode a step's edits write no event**: the store is
/// held on this member, and scrapped with the local world. An author's store
/// belongs to no world, and is written as events whatever runs disconnected.
#[test]
fn a_disconnected_world_writes_no_event() {
    let (dir, shared) = setup();
    let s = stores(&dir, &shared);
    s.set_quiet(Box::new(|world| world == "vault"));
    shelf(&s, 1)
        .commit(&[("layers/eras/a.md".into(), Some("offline\n".into()))])
        .unwrap();
    assert!(shared
        .vfs_events(Owner::operation("vault", 1).timeline().raw())
        .unwrap()
        .is_empty());
    assert_eq!(
        shelf(&s, 1).read("layers/eras/a.md").as_deref(),
        Some("offline\n")
    );
    s.scrap("vault");
    assert_eq!(
        shelf(&s, 1).read("layers/eras/a.md").as_deref(),
        Some("the era\n")
    );

    author(&s, "ab")
        .commit(&[("layers/eras/a.md".into(), Some("drafted\n".into()))])
        .unwrap();
    assert!(!shared
        .vfs_events(Owner::author("ab").timeline().raw())
        .unwrap()
        .is_empty());
}

/// A commit refused partway takes the whole batch back out.
#[test]
fn a_refused_commit_keeps_nothing() {
    let (dir, shared) = setup();
    let s = stores(&dir, &shared);
    let huge = "x".repeat(11 << 20);
    let refused = shelf(&s, 1).commit(&[
        ("layers/eras/a.md".into(), Some("fine\n".into())),
        ("layers/eras/b.md".into(), Some(huge)),
    ]);
    assert!(refused.is_err());
    assert_eq!(
        shelf(&s, 1).read("layers/eras/a.md").as_deref(),
        Some("the era\n")
    );
}
