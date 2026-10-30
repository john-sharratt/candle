//! The store against real repositories: a bare origin seeded with the
//! fixture family, and clones of it standing for the agents that hold it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

use super::*;
use crate::files::{render_lines, Role};
use crate::line::Writer;
use crate::ops::fixture::{john, now, p, sharratt, yen};
use crate::ops::{OnTaken, Which};
use crate::path::Address;
use crate::scope::Client;
use crate::testing::{note as testing_note, slug};
use crate::tree::render::render_all;

/// A `git` command for the fixture, failing the test on failure.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "init.defaultBranch=main",
            "-c",
            "core.autocrlf=false",
        ])
        .args(args)
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

struct Repos {
    _root: TempDir,
    root: PathBuf,
    origin: PathBuf,
}

impl Repos {
    fn new() -> Repos {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().to_path_buf();
        let origin = path.join("origin.git");
        std::fs::create_dir(&origin).unwrap();
        git(&origin, &["init", "-q", "--bare", "-b", "main"]);
        let seed = path.join("seed");
        git(&path, &["clone", "-q", origin.to_str().unwrap(), "seed"]);
        for (file, bytes) in render_all(&sharratt()) {
            let at = seed.join(&file);
            std::fs::create_dir_all(at.parent().unwrap()).unwrap();
            std::fs::write(at, bytes).unwrap();
        }
        std::fs::write(
            seed.join(".gitattributes"),
            "lines.yaml merge=union\n**/archive/*.yaml merge=union\n",
        )
        .unwrap();
        git(&seed, &["add", "-A"]);
        git(&seed, &["commit", "-q", "-m", "seed"]);
        git(&seed, &["push", "-q", "origin", "main"]);
        Repos {
            _root: root,
            root: path,
            origin,
        }
    }

    /// A clone for one agent.
    fn clone(&self, name: &str) -> PathBuf {
        git(
            &self.root,
            &["clone", "-q", self.origin.to_str().unwrap(), name],
        );
        self.root.join(name)
    }

    /// What origin's main holds, read through a fresh clone.
    fn origin_tree(&self, name: &str) -> Tree {
        let dir = self.clone(name);
        Store::open(&dir).unwrap().tree().clone()
    }

    /// A commit straight to origin, as a person or a job would make one.
    fn hand_edit(&self, name: &str, file: &str, text: &str) {
        let dir = self.clone(name);
        let at = dir.join(file);
        std::fs::create_dir_all(at.parent().unwrap()).unwrap();
        std::fs::write(&at, text).unwrap();
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-q", "-m", "By hand"]);
        git(&dir, &["push", "-q", "origin", "main"]);
    }
}

fn note(text: &str) -> Op {
    Op::Note {
        path: p("topics"),
        text: text.into(),
    }
}

fn texts(tree: &Tree, path: &str) -> Vec<String> {
    tree.get(&p(path))
        .unwrap()
        .window
        .iter()
        .map(|l| l.text.clone())
        .collect()
}

#[test]
fn a_write_is_committed_at_once_and_pushed_at_the_sync() {
    let r = Repos::new();
    let dir = r.clone("a");
    let mut s = Store::open(&dir).unwrap();
    assert!(!s.unpushed());
    let a = s
        .write(&yen(), &note("The builder starts Monday."), now())
        .unwrap();
    assert!(a.line.is_some());
    assert!(s.unpushed());
    assert_eq!(texts(s.tree(), "topics"), ["The builder starts Monday."]);
    // The checkout's own branch and working tree are the jobs'; untouched.
    assert!(!dir.join("families/sharratt/topics/lines.yaml").exists());
    let synced = s.sync().unwrap();
    assert_eq!(
        (synced.pushed, synced.replayed, synced.fetched),
        (1, 0, false)
    );
    assert!(!s.unpushed());
    assert_eq!(
        texts(&r.origin_tree("check"), "topics"),
        ["The builder starts Monday."]
    );
    // The commit says who, for whom, through what — and carries the op.
    let log = git(
        &dir,
        &["log", "-1", "--format=%an%n%B", "refs/journal/head"],
    );
    assert!(log.starts_with("grok\nNote on @sharratt/topics\n\nJournal-Writer: grok\nJournal-Person: yen\nJournal-Client: grok\n"), "{log}");
    assert!(log.contains("Journal-Op: {\"op\":\"note\""), "{log}");
}

#[test]
fn two_agents_writing_at_once_both_land() {
    let r = Repos::new();
    let mut a = Store::open(&r.clone("a")).unwrap();
    let mut b = Store::open(&r.clone("b")).unwrap();
    a.write(&yen(), &note("from a"), now()).unwrap();
    b.write(&john(), &note("from b"), now().plus(1)).unwrap();
    a.sync().unwrap();
    let synced = b.sync().unwrap();
    assert_eq!(
        (synced.replayed, synced.dropped.len(), synced.fetched),
        (1, 0, true)
    );
    assert_eq!(texts(b.tree(), "topics"), ["from a", "from b"]);
    assert_eq!(
        texts(&r.origin_tree("check"), "topics"),
        ["from a", "from b"]
    );
    // A takes B's note at its next sync.
    let synced = a.sync().unwrap();
    assert!(synced.fetched);
    assert_eq!(texts(a.tree(), "topics"), ["from a", "from b"]);
}

#[test]
fn a_replayed_line_whose_id_was_taken_takes_a_new_one_and_its_edits_follow() {
    let r = Repos::new();
    let mut a = Store::open(&r.clone("a")).unwrap();
    let mut b = Store::open(&r.clone("b")).unwrap();
    // Two different notes in one minute whose ids collide: a birthday
    // search over the id's sixteen bits finds a pair in a few hundred texts.
    let by = Writer::Client(Client::ClaudeCode);
    let mut seen: BTreeMap<LineId, String> = BTreeMap::new();
    let (xa, xb) = (0..)
        .map(|i| format!("note {i}"))
        .find_map(|t| {
            let id = LineId::mint(now(), by, &t, 0);
            seen.insert(id, t.clone()).map(|first| (first, t))
        })
        .unwrap();
    let ida = a.write(&john(), &note(&xa), now()).unwrap().line.unwrap();
    let idb = b.write(&john(), &note(&xb), now()).unwrap().line.unwrap();
    assert_eq!(ida, idb);
    b.write(
        &john(),
        &Op::Edit {
            address: Address {
                path: p("topics"),
                id: idb.clone(),
            },
            text: "b's own".into(),
        },
        now().plus(1),
    )
    .unwrap();
    assert!(a.sync().unwrap().settled);
    let synced = b.sync().unwrap();
    assert!(synced.settled);
    assert_eq!(synced.dropped, vec![]);
    assert_eq!(synced.replayed, 2);
    assert_eq!(synced.renamed.len(), 1, "{synced:?}");
    assert_eq!(synced.renamed[0].0, idb);
    assert_ne!(synced.renamed[0].1, idb);
    // A's line is untouched; B's edit reached B's own line.
    let topics = b.tree().get(&p("topics")).unwrap();
    assert_eq!(topics.line(&ida).unwrap().text, xa);
    assert_eq!(topics.line(&synced.renamed[0].1).unwrap().text, "b's own");
}

#[test]
fn a_replayed_line_keeps_its_id_when_it_is_free() {
    let r = Repos::new();
    let mut a = Store::open(&r.clone("a")).unwrap();
    let mut b = Store::open(&r.clone("b")).unwrap();
    a.write(&yen(), &note("from a"), now()).unwrap();
    let id = b
        .write(&john(), &note("from b"), now())
        .unwrap()
        .line
        .unwrap();
    a.sync().unwrap();
    let synced = b.sync().unwrap();
    assert_eq!(synced.renamed, vec![]);
    assert!(b.tree().get(&p("topics")).unwrap().has_id(&id));
}

#[test]
fn a_write_the_newer_origin_refuses_is_dropped_and_reported() {
    let r = Repos::new();
    let mut a = Store::open(&r.clone("a")).unwrap();
    let id = a
        .write(&john(), &note("Prefers tea."), now())
        .unwrap()
        .line
        .unwrap();
    a.sync().unwrap();
    let mut b = Store::open(&r.clone("b")).unwrap();
    // A forgets the line while B edits it.
    a.write(
        &john(),
        &Op::Forget {
            path: p("topics"),
            which: Which::Id(id.clone()),
        },
        now(),
    )
    .unwrap();
    b.write(
        &john(),
        &Op::Edit {
            address: Address {
                path: p("topics"),
                id,
            },
            text: "Prefers coffee.".into(),
        },
        now(),
    )
    .unwrap();
    a.sync().unwrap();
    let synced = b.sync().unwrap();
    assert_eq!(synced.replayed, 0);
    assert_eq!(synced.dropped.len(), 1);
    assert!(
        synced.dropped[0].0.starts_with("Edit of @sharratt/topics#"),
        "{:?}",
        synced.dropped
    );
    assert!(
        texts(b.tree(), "topics").is_empty(),
        "a forgotten line is not resurrected"
    );
    assert!(texts(&r.origin_tree("check"), "topics").is_empty());
}

#[test]
fn unpushed_commits_survive_a_restart() {
    let r = Repos::new();
    let dir = r.clone("a");
    {
        let mut s = Store::open(&dir).unwrap();
        s.write(&yen(), &note("kept across a crash"), now())
            .unwrap();
    }
    let mut s = Store::open(&dir).unwrap();
    assert!(s.unpushed());
    assert_eq!(texts(s.tree(), "topics"), ["kept across a crash"]);
    assert_eq!(s.sync().unwrap().pushed, 1);
    assert_eq!(
        texts(&r.origin_tree("check"), "topics"),
        ["kept across a crash"]
    );
}

#[test]
fn a_restart_behind_a_moved_origin_replays_what_it_held() {
    let r = Repos::new();
    let dir = r.clone("a");
    {
        let mut s = Store::open(&dir).unwrap();
        s.write(&yen(), &note("written offline"), now()).unwrap();
    }
    let mut b = Store::open(&r.clone("b")).unwrap();
    b.write(&john(), &note("pushed meanwhile"), now().plus(1))
        .unwrap();
    b.sync().unwrap();
    let mut s = Store::open(&dir).unwrap();
    assert_eq!(
        texts(s.tree(), "topics"),
        ["pushed meanwhile", "written offline"]
    );
    s.sync().unwrap();
    assert_eq!(
        texts(&r.origin_tree("check"), "topics"),
        ["pushed meanwhile", "written offline"]
    );
}

#[test]
fn a_hand_edit_on_origin_is_read_at_the_next_sync() {
    let r = Repos::new();
    let mut s = Store::open(&r.clone("a")).unwrap();
    let line = testing_note("2026-10-30T09:00:00Z", "written by hand");
    r.hand_edit(
        "hand",
        "families/sharratt/topics/lines.yaml",
        &render_lines(&[line]),
    );
    let synced = s.sync().unwrap();
    assert!(synced.fetched);
    assert_eq!(texts(s.tree(), "topics"), ["written by hand"]);
}

#[test]
fn a_broken_hand_edit_is_answered_as_an_error_until_fixed() {
    let r = Repos::new();
    let mut s = Store::open(&r.clone("a")).unwrap();
    r.hand_edit(
        "hand",
        "families/sharratt/topics/lines.yaml",
        "- {id: nope}\n",
    );
    s.sync().unwrap();
    let topics = s.tree().get(&p("topics")).unwrap();
    assert_eq!(topics.problems[0].to_string(), "families/sharratt/topics/lines.yaml:1: `id`: `nope` is not a line id, written 20261030-0912-7e3a");
    assert!(matches!(
        s.write(&yen(), &note("x"), now()),
        Err(StoreError::Journal(Error::Refused(_)))
    ));
    // Other journals are untouched by it.
    s.write(
        &yen(),
        &Op::Note {
            path: p("projects"),
            text: "fine".into(),
        },
        now(),
    )
    .unwrap();
    r.hand_edit("fix", "families/sharratt/topics/lines.yaml", "");
    s.sync().unwrap();
    assert!(s.tree().get(&p("topics")).unwrap().problems.is_empty());
    assert_eq!(texts(s.tree(), "projects"), ["fine"]);
}

#[test]
fn every_kind_of_write_lands_on_origin_as_its_files() {
    let r = Repos::new();
    let mut s = Store::open(&r.clone("a")).unwrap();
    let ops = [
        Op::Create {
            under: p("topics"),
            name: slug("trip"),
            cue: "open for the trip".into(),
            on_taken: OnTaken::Refuse,
        },
        Op::Note {
            path: p("topics/trip"),
            text: "Flights.".into(),
        },
        Op::Create {
            under: p("topics/trip"),
            name: slug("hotels"),
            cue: "open for hotels".into(),
            on_taken: OnTaken::Refuse,
        },
        Op::Note {
            path: p("topics/trip/hotels"),
            text: "Hoi An.".into(),
        },
        Op::Merge {
            from: p("topics/trip/hotels"),
            into: p("topics/trip"),
        },
        Op::Fold {
            path: p("topics/trip"),
        },
        Op::AddMember {
            family: slug("sharratt"),
            person: slug("yen-nhi"),
            name: Some("Yen Nhi".into()),
            emails: vec![],
            role: Role::Member,
        },
    ];
    for op in &ops {
        s.write(&john(), op, now()).unwrap();
    }
    s.sync().unwrap();
    let mine = s.tree().clone();
    assert_eq!(r.origin_tree("check"), mine);
    assert_eq!(texts(&mine, "topics/trip"), ["Flights.", "Hoi An."]);
    assert!(mine.person_file(&slug("yen-nhi")).is_some());
}

#[test]
fn a_write_the_rules_refuse_commits_nothing() {
    let r = Repos::new();
    let mut s = Store::open(&r.clone("a")).unwrap();
    let head = s.head().clone();
    assert!(s
        .write(
            &john(),
            &Op::Note {
                path: p("bots"),
                text: "x".into()
            },
            now()
        )
        .is_err());
    assert_eq!(s.head(), &head);
    assert!(!s.unpushed());
}

#[test]
fn a_repository_with_nothing_on_main_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let origin = root.path().join("o.git");
    std::fs::create_dir(&origin).unwrap();
    git(&origin, &["init", "-q", "--bare", "-b", "main"]);
    git(root.path(), &["clone", "-q", origin.to_str().unwrap(), "c"]);
    let err = Store::open(&root.path().join("c"))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("seed the repository first"), "{err}");
}
