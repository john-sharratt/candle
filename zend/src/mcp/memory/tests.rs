//! The memory on a real journal repository: opened on its thread, written
//! through, kept in step with origin.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use journals::error::Problem;
use journals::files::{Cite, Sentence};
use journals::scope::Channel;
use journals::slug::Slug;
use journals::tree::fingerprint::fingerprint;
use tempfile::TempDir;

use super::who::Who;
use super::*;

#[test]
fn memory_paths_are_told_from_thread_paths() {
    for memory in [
        "@zen",
        "@zen/topics",
        "topics/house",
        "people/yen",
        "here",
        "here#2026-10-30-0912-7e3a",
        "conversations",
        "~",
    ] {
        assert!(is_memory_path(memory), "{memory}");
    }
    for thread in ["self", "T:review", "T:review/verify", "P:candle", "U:", ""] {
        assert!(!is_memory_path(thread), "{thread}");
    }
}

#[test]
fn a_memory_with_no_checkout_says_so() {
    let m = Memory::default();
    assert!(!m.serves(), "no checkout, no writer");
    assert_eq!(m.tree().unwrap_err(), NO_MEMORY);
    let op = Op::MarkStale {
        path: JournalPath::root(Slug::parse("x").unwrap()),
        why: "over its cap".into(),
    };
    let e = m.write(&Actor::Zend, &op).unwrap_err();
    assert_eq!(e.to_string(), NO_MEMORY);
    assert!(
        !e.is_conflict() && !e.is_refusal() && !e.is_broken(),
        "absent is passing"
    );
    let broken = WriteError::Store(StoreError::Journal(Error::Broken(Problem::new(
        "groups/zen/journal.yaml",
        Some(3),
        "`text` is missing",
    ))));
    assert!(broken.is_broken() && !broken.is_refusal() && !broken.is_conflict());
}

fn zen() -> Slug {
    Slug::parse("zen").unwrap()
}

fn john() -> Slug {
    Slug::parse("john").unwrap()
}

fn johns_journal() -> JournalPath {
    moved::own_journal(&zen(), &john()).unwrap()
}

/// **A write's error says what it is by its type**: a summary built over a
/// journal that has moved on is a conflict; one the rules refuse is a
/// refusal; the memory not being there is neither.
#[test]
fn a_write_error_is_told_by_its_kind() {
    let (_root, dir) = repo();
    let m = opened(dir);
    let own = johns_journal();
    let actor = Actor::Person {
        person: john(),
        channel: Channel::Grok,
        admin: false,
    };
    let note = |text: &str| Op::Note {
        path: own.clone(),
        text: text.into(),
        speaker: None,
    };
    let id = m
        .write(&actor, &note("Prefers tea."))
        .unwrap()
        .line
        .unwrap();
    let over = fingerprint(m.tree().unwrap().get(&own).unwrap());
    let summary = |text: &str| Op::Summarize {
        path: own.clone(),
        over: over.clone(),
        sentences: vec![Sentence {
            text: text.into(),
            cites: vec![Cite::Line(id.clone())],
        }],
        digest: None,
        cue: None,
    };
    let refused = m
        .write(&Actor::Zend, &summary(&"tea ".repeat(500)))
        .unwrap_err();
    assert!(refused.is_refusal() && !refused.is_conflict(), "{refused}");
    m.write(&actor, &note("And green tea.")).unwrap();
    let conflict = m.write(&Actor::Zend, &summary("Tea.")).unwrap_err();
    assert!(
        conflict.is_conflict() && !conflict.is_refusal(),
        "{conflict}"
    );
}

fn git(dir: &Path, args: &[&str]) {
    let out = git_run(dir, args);
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn git_run(dir: &Path, args: &[&str]) -> Output {
    Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .expect("git runs")
}

/// A bare origin seeded with the group Zen, its owner John and his own
/// journal, and a clone of it.
pub(crate) fn repo() -> (TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let origin = root.path().join("origin.git");
    std::fs::create_dir(&origin).unwrap();
    git(&origin, &["init", "-q", "--bare", "-b", "main"]);
    git(
        root.path(),
        &["clone", "-q", origin.to_str().unwrap(), "seed"],
    );
    let seed = root.path().join("seed");
    for (file, text) in [
        ("persons/john/person.yaml", "person: john\nname: \"John\"\naccounts:\n  - {email: \"john@example.com\"}\ngroups: [zen]\ndefault_group: zen\n"),
        ("groups/zen/group.yaml", "group: zen\nname: \"Zen\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n"),
        ("groups/zen/people/john/journal.yaml", "name: john\ncue: \"open for John\"\ncreated: 2026-10-30T00:00:00Z\nfolded: false\n"),
    ] {
        let at = seed.join(file);
        std::fs::create_dir_all(at.parent().unwrap()).unwrap();
        std::fs::write(at, text).unwrap();
    }
    git(&seed, &["add", "-A"]);
    git(&seed, &["commit", "-q", "-m", "seed"]);
    git(&seed, &["push", "-q", "origin", "main"]);
    git(
        root.path(),
        &["clone", "-q", origin.to_str().unwrap(), "agent"],
    );
    let agent = root.path().join("agent");
    (root, agent)
}

fn opened(dir: PathBuf) -> Arc<Memory> {
    let m = Arc::new(Memory::default());
    m.start(dir);
    let until = Instant::now() + Duration::from_secs(30);
    while m.tree().is_err() {
        assert!(
            Instant::now() < until,
            "the memory never opened: {:?}",
            m.tree().err()
        );
        thread::sleep(Duration::from_millis(20));
    }
    m
}

#[test]
fn the_memory_opens_on_its_thread_binds_a_caller_and_writes_through() {
    let (_root, dir) = repo();
    let m = opened(dir.clone());
    let tree = m.tree().unwrap();
    let (who, bind) = who::known(&tree, "154797f34c5a6ca6", "John@Example.com").unwrap();
    assert_eq!(who, Who::Person(john()));
    m.write(&Actor::System, &bind.unwrap()).unwrap();
    let (again, bind) = who::known(&m.tree().unwrap(), "154797f34c5a6ca6", "").unwrap();
    assert_eq!(
        (again, bind),
        (Who::Person(john()), None),
        "the account is bound now"
    );
    let conv = conversations::of(&m, &john(), &zen(), Channel::Grok, "session-1").unwrap();
    assert!(conv.show().starts_with("@zen/conversations/"), "{conv}");
    assert!(conv.show().ends_with("-grok"), "{conv}");
    assert_eq!(
        conversations::of(&m, &john(), &zen(), Channel::Grok, "session-1").unwrap(),
        conv,
        "one journal a session"
    );
    let own = johns_journal();
    let actor = Actor::Person {
        person: john(),
        channel: Channel::Grok,
        admin: false,
    };
    m.write(
        &actor,
        &Op::Note {
            path: own.clone(),
            text: "Prefers short answers.".into(),
            speaker: Some(john()),
        },
    )
    .unwrap();
    let text = context::render(
        &m.tree().unwrap(),
        &zen(),
        Some(&conv),
        Channel::ClaudeCode,
        false,
        Stamp::now(),
    );
    assert!(text.contains("This conversation"), "{text}");
    // The sync thread pushes it: a fresh clone of origin reads it. A clone
    // made while a push is landing can find origin's objects locked, which
    // is the push not being there yet.
    let until = Instant::now() + Duration::from_secs(30);
    for attempt in 0.. {
        let check = dir.parent().unwrap().join(format!("check-{attempt}"));
        let cloned = git_run(
            dir.parent().unwrap(),
            &[
                "clone",
                "-q",
                dir.parent().unwrap().join("origin.git").to_str().unwrap(),
                check.file_name().unwrap().to_str().unwrap(),
            ],
        )
        .status
        .success();
        let pushed = cloned
            && Store::open(&check).is_ok_and(|s| {
                s.shared()
                    .get(&own)
                    .is_some_and(|j| j.window.iter().any(|l| l.text == "Prefers short answers."))
            });
        if pushed {
            break;
        }
        assert!(Instant::now() < until, "the note never reached origin");
        thread::sleep(Duration::from_millis(500));
    }
}

/// **A person's old `U:` notes move onto their own journal**: kept facts
/// and episodes alike, each ended in the substrate once moved; a second move
/// finds nothing left.
#[test]
fn old_notes_move_onto_their_journal_and_leave_the_substrate() {
    use crate::mcp::records::MemRecords;
    use crate::mcp::threads::{self, Note};

    let (_root, dir) = repo();
    let m = opened(dir);
    let records = MemRecords::default();
    let at = 1_793_351_402_000;
    for (label, text, channel) in [
        ("Noted:", "the cat is Miso", "grok"),
        ("Done:", "image_generate — a cat", "grok"),
    ] {
        threads::add_note(
            &records,
            "owner",
            "U:",
            &Note {
                at_ms: at,
                label: label.into(),
                text: text.into(),
                worker: None,
                client: Some(channel.into()),
            },
        )
        .unwrap();
    }
    moved::move_all(&m, &records, "owner", &zen(), &john());
    let tree = m.tree().unwrap();
    let own = tree.get(&johns_journal()).unwrap();
    let texts: Vec<&str> = own.window.iter().map(|l| l.text.as_str()).collect();
    assert_eq!(texts.len(), 2);
    assert!(texts.contains(&"the cat is Miso") && texts.contains(&"image_generate — a cat"));
    assert_eq!(own.window[0].at.to_string(), "2026-10-30T09:10:02Z");
    assert!(
        threads::notes(&records, "owner", "U:").is_empty(),
        "moved notes leave the substrate"
    );
    moved::move_all(&m, &records, "owner", &zen(), &john());
    assert_eq!(
        m.tree()
            .unwrap()
            .get(&johns_journal())
            .unwrap()
            .window
            .len(),
        2
    );
}

/// **Reads never wait for the store**: a tree is read while a write holds it.
#[test]
fn a_read_does_not_wait_for_the_store() {
    let (_root, dir) = repo();
    let m = opened(dir);
    let guard = held(&m.store);
    let started = Instant::now();
    assert!(m.tree().is_ok());
    assert!(started.elapsed() < Duration::from_secs(1));
    drop(guard);
}

#[test]
fn a_memory_that_cannot_open_says_why() {
    let root = tempfile::tempdir().unwrap();
    let m = Arc::new(Memory::default());
    m.start(root.path().to_path_buf());
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        match m.tree() {
            Err(e) if e.starts_with("the journal repository could not be read") => break,
            _ => {}
        }
        assert!(Instant::now() < until, "{:?}", m.tree().err());
        thread::sleep(Duration::from_millis(20));
    }
    assert!(!m.serves(), "a memory that failed has no writer");
}
