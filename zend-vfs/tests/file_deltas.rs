//! Building, storing and applying file deltas — end to end.
//!
//! Three layers, each driven by generated edits rather than a handful of
//! hand-picked ones:
//!
//! 1. **The delta itself** (`state::file_delta`): for thousands of generated
//!    edits over LF, CRLF, unterminated and non-ASCII text, the splices built
//!    from an old and a new text replay to the new text exactly, are ordered,
//!    disjoint and minimal at their edges, survive their wire form, and refuse
//!    a text they were not made against.
//! 2. **Through the patch engine into a store**: unified diffs are generated
//!    for each edit, applied by `file_edit`'s engine, and recorded — in an
//!    overlay, where they are held as deltas and the disk is untouched, and in
//!    a direct store, where they land on the file on disk. Both must hold the
//!    same text after every step.
//! 3. **Stored deltas applied elsewhere**: a conversation's recorded chain,
//!    applied to a direct store over a pristine copy of the workspace, puts
//!    exactly the conversation's text on disk; applied to a fresh overlay, it
//!    reproduces the conversation's view.
//!
//! The generator is a fixed-seed LCG, so a failure names its case and
//! reproduces.

use std::path::Path;

use similar::TextDiff;
use zend_vfs::file_delta::{self, Diverged, FileDelta, FileTimes, ReplayError, Splice, TimedDelta};
use zend_vfs::{patch, DiskWriteGrant, VfsStore};

// ── The generator ────────────────────────────────────────────────────────────

/// A fixed-seed linear congruential generator: reproducible, dependency-free.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// How a generated file ends its lines.
#[derive(Clone, Copy, Debug)]
enum Eol {
    Lf,
    Crlf,
}

impl Eol {
    fn as_str(self) -> &'static str {
        match self {
            Eol::Lf => "\n",
            Eol::Crlf => "\r\n",
        }
    }
}

/// A file as lines without terminators, plus how they end.
#[derive(Clone, Debug)]
struct Doc {
    lines: Vec<String>,
    eol: Eol,
    trailing_newline: bool,
}

impl Doc {
    fn text(&self) -> String {
        let mut out = String::new();
        for (i, line) in self.lines.iter().enumerate() {
            out.push_str(line);
            if i + 1 < self.lines.len() || self.trailing_newline {
                out.push_str(self.eol.as_str());
            }
        }
        out
    }
}

/// Text with repeated lines, non-ASCII and blank lines in it — the cases a
/// line diff is most likely to get wrong.
const VOCABULARY: &[&str] = &[
    "fn main() {",
    "}",
    "",
    "    let x = 1;",
    "    return x;",
    "// café — naïve",
    "こんにちは 🌍",
    "    }",
    "use std::io;",
    "#[test]",
];

fn random_line(rng: &mut Lcg, unique: &mut usize) -> String {
    // Mostly distinct lines, sometimes a repeated one.
    if rng.chance(30) {
        VOCABULARY[rng.below(VOCABULARY.len())].to_string()
    } else {
        *unique += 1;
        format!("line {unique} {}", VOCABULARY[rng.below(VOCABULARY.len())])
    }
}

fn random_doc(rng: &mut Lcg, unique: &mut usize) -> Doc {
    let n = rng.below(40);
    Doc {
        lines: (0..n).map(|_| random_line(rng, unique)).collect(),
        eol: if rng.chance(30) { Eol::Crlf } else { Eol::Lf },
        trailing_newline: rng.chance(80),
    }
}

/// One to four random line edits: insertions, deletions and replacements at
/// the start, the middle and the end.
fn mutate(rng: &mut Lcg, doc: &Doc, unique: &mut usize) -> Doc {
    let mut out = doc.clone();
    for _ in 0..=rng.below(4) {
        let len = out.lines.len();
        match rng.below(3) {
            0 => {
                let at = rng.below(len + 1);
                for k in 0..=rng.below(3) {
                    out.lines.insert(at + k, random_line(rng, unique));
                }
            }
            1 if len > 0 => {
                let at = rng.below(len);
                let n = (1 + rng.below(3)).min(len - at);
                out.lines.drain(at..at + n);
            }
            _ if len > 0 => {
                let at = rng.below(len);
                out.lines[at] = random_line(rng, unique);
            }
            _ => out.lines.push(random_line(rng, unique)),
        }
    }
    if rng.chance(5) {
        out.trailing_newline = !out.trailing_newline;
    }
    out
}

// ── 1. The delta itself ─────────────────────────────────────────────────────

/// **Every generated edit's splices replay to its result exactly**, and are
/// what a delta should be: ascending, disjoint, each a real change, no
/// unchanged line carried at either edge but an insertion's anchor, every
/// insertion anchored unless the file was empty, and accounting for every
/// byte.
#[test]
fn generated_edits_build_splices_that_replay_exactly() {
    let mut rng = Lcg(0x5eed);
    let mut unique = 0;
    for case in 0..3_000 {
        let old = random_doc(&mut rng, &mut unique);
        let new = mutate(&mut rng, &old, &mut unique);
        let (old, new) = (old.text(), new.text());
        let splices = file_delta::splices(&old, &new);

        assert_eq!(
            file_delta::apply(&old, &splices).as_deref(),
            Ok(new.as_str()),
            "case {case}: {old:?} -> {new:?} via {splices:?}"
        );
        let mut end = 0;
        for s in &splices {
            assert!(
                s.at >= end,
                "case {case}: splices out of order or overlapping"
            );
            assert_ne!(
                s.removed, s.inserted,
                "case {case}: a splice that changes nothing"
            );
            assert_eq!(&old[s.at..s.at + s.removed.len()], s.removed, "case {case}");
            end = s.at + s.removed.len();
            // Minimal at its edges: the first and last lines of each side
            // differ, or the diff carried an unchanged line it did not need —
            // save the one line an insertion is anchored to, which it removes
            // and puts back beside the new lines.
            let one_line = s.removed.split_inclusive('\n').count() == 1;
            let anchored_before = one_line && s.inserted.starts_with(&s.removed);
            let anchored_after = one_line && s.inserted.ends_with(&s.removed);
            if !s.removed.is_empty() && !s.inserted.is_empty() {
                if !anchored_before {
                    assert_ne!(
                        s.removed.split_inclusive('\n').next(),
                        s.inserted.split_inclusive('\n').next(),
                        "case {case}: {s:?} starts with an unchanged line"
                    );
                }
                if !anchored_after {
                    assert_ne!(
                        s.removed.split_inclusive('\n').next_back(),
                        s.inserted.split_inclusive('\n').next_back(),
                        "case {case}: {s:?} ends with an unchanged line"
                    );
                }
            }
            // Only an empty file takes an insertion with nothing to check.
            if s.removed.is_empty() {
                assert!(old.is_empty(), "case {case}: an unanchored insertion {s:?}");
            }
        }
        let removed: usize = splices.iter().map(|s| s.removed.len()).sum();
        let inserted: usize = splices.iter().map(|s| s.inserted.len()).sum();
        assert_eq!(old.len() - removed + inserted, new.len(), "case {case}");
        if old == new {
            assert!(splices.is_empty(), "case {case}: no change, no splice");
        }
    }
}

/// **The recorded delta — an edit or, past the threshold, a replace —
/// survives its wire form and replays to the result**, from the file it was
/// made against.
#[test]
fn generated_deltas_survive_the_wire_and_replay() {
    let mut rng = Lcg(0xde17a);
    let mut unique = 0;
    let (mut edits, mut replaces) = (0, 0);
    for case in 0..2_000 {
        let old = random_doc(&mut rng, &mut unique);
        let new = mutate(&mut rng, &old, &mut unique);
        let (old, new) = (old.text(), new.text());
        let delta = file_delta::delta(&old, &new);
        match &delta {
            FileDelta::Edit { .. } => edits += 1,
            FileDelta::Replace { .. } => replaces += 1,
            FileDelta::Delete | FileDelta::ReplaceBinary { .. } => {
                panic!("case {case}: text edited into text is {delta:?}")
            }
        }
        let wire = serde_json::to_string(&delta).unwrap();
        let back: FileDelta = serde_json::from_str(&wire).unwrap();
        assert_eq!(back, delta, "case {case}");
        assert_eq!(
            file_delta::replay(Some(old.clone()), &[back]),
            Ok(Some(new.clone())),
            "case {case}: {old:?} -> {new:?} via {delta:?}"
        );
    }
    assert!(
        edits > 100 && replaces > 100,
        "the sweep must exercise both forms: {edits} edits, {replaces} replaces"
    );
}

/// **A chain of generated edits replays to the last text**, each edit made
/// against the one before it — and replaying from anything but the chain's
/// base diverges rather than producing a plausible wrong file.
#[test]
fn generated_chains_replay_from_their_base_only() {
    let mut rng = Lcg(0xc4a1);
    let mut unique = 0;
    for case in 0..300 {
        let base = random_doc(&mut rng, &mut unique);
        let mut doc = base.clone();
        let mut chain = Vec::new();
        for _ in 0..8 {
            let next = mutate(&mut rng, &doc, &mut unique);
            chain.push(FileDelta::Edit {
                splices: file_delta::splices(&doc.text(), &next.text()),
            });
            doc = next;
        }
        assert_eq!(
            file_delta::replay(Some(base.text()), &chain),
            Ok(Some(doc.text())),
            "case {case}"
        );
        // Anything that changes the base under the first real edit is caught.
        let first_edit = chain.iter().find_map(|d| match d {
            FileDelta::Edit { splices } if !splices.is_empty() => Some(&splices[0]),
            _ => None,
        });
        if let Some(s) = first_edit {
            let mut wrong = base.text();
            if s.removed.is_empty() {
                // An insertion checks nothing it can be caught by; skip it.
                continue;
            }
            wrong.replace_range(s.at..s.at + s.removed.len(), "«changed underneath»\n");
            assert!(
                file_delta::replay(Some(wrong), &chain).is_err(),
                "case {case}: replay onto a changed base went through"
            );
        }
    }
}

fn splice(at: usize, removed: &str, inserted: &str) -> Splice {
    Splice {
        at,
        removed: removed.to_string(),
        inserted: inserted.to_string(),
    }
}

/// Splices that cannot have come from one edit are refused, never applied:
/// out of order, overlapping, past the end, inside a character, or naming
/// text that is not there.
#[test]
fn malformed_splices_are_refused() {
    let text = "one\ntwo\nthree\n";
    for (splices, at) in [
        (vec![splice(8, "three\n", ""), splice(0, "one\n", "")], 0),
        (vec![splice(0, "one\ntwo\n", ""), splice(4, "two\n", "")], 4),
        (vec![splice(14, "x", "")], 14),
        (vec![splice(4, "TWO\n", "2\n")], 4),
        (vec![splice(usize::MAX, "x", "")], usize::MAX),
    ] {
        assert_eq!(
            file_delta::apply(text, &splices),
            Err(Diverged { at }),
            "{splices:?}"
        );
    }
    assert_eq!(
        file_delta::apply("é", &[splice(1, "", "x")]),
        Err(Diverged { at: 1 })
    );
}

/// A chain's replace and delete stand on their own; an edit needs a file.
#[test]
fn replay_follows_replace_and_delete() {
    let replace = FileDelta::Replace {
        content: "a\nb\n".into(),
    };
    let edit = FileDelta::Edit {
        splices: vec![splice(2, "b\n", "B\n")],
    };
    assert_eq!(
        file_delta::replay(None, &[replace.clone(), edit.clone()]),
        Ok(Some("a\nB\n".to_string()))
    );
    assert_eq!(
        file_delta::replay(Some("ignored\n".into()), &[replace, FileDelta::Delete]),
        Ok(None)
    );
    assert_eq!(
        file_delta::replay(None, std::slice::from_ref(&edit)),
        Err(ReplayError::Diverged(Diverged { at: 2 }))
    );
    assert_eq!(
        file_delta::replay(Some("kept\n".into()), &[]),
        Ok(Some("kept\n".to_string()))
    );
}

// ── 2 and 3. Through the patch engine, into a store, and onto disk ──────────

fn put(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, text).unwrap();
}

fn on_disk(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel)).unwrap()
}

fn direct(root: &Path) -> VfsStore {
    VfsStore::direct(root, &DiskWriteGrant::issue())
}

/// A file the patch engine can take hunks for without ambiguity: distinct,
/// numbered lines, always newline-terminated, LF or CRLF.
fn patchable_doc(rng: &mut Lcg, unique: &mut usize) -> Doc {
    let n = 6 + rng.below(40);
    Doc {
        lines: (0..n)
            .map(|_| {
                *unique += 1;
                format!("line {unique} {}", VOCABULARY[rng.below(VOCABULARY.len())])
            })
            .collect(),
        eol: if rng.chance(30) { Eol::Crlf } else { Eol::Lf },
        trailing_newline: true,
    }
}

/// One edit a patch can express: insertions, deletions and replacements of
/// distinct lines, the file keeping at least one line of context.
fn patchable_edit(rng: &mut Lcg, doc: &Doc, unique: &mut usize) -> Doc {
    let mut out = doc.clone();
    for _ in 0..=rng.below(3) {
        let len = out.lines.len();
        let mut fresh = || {
            *unique += 1;
            format!("new {unique}")
        };
        match rng.below(3) {
            0 => {
                let at = rng.below(len + 1);
                out.lines.insert(at, fresh());
            }
            1 if len > 3 => {
                let at = rng.below(len);
                out.lines.remove(at);
            }
            _ => {
                let at = rng.below(len);
                out.lines[at] = fresh();
            }
        }
    }
    out
}

/// A unified diff from `old` to `new`, as a model would send `file_edit`.
fn unified(old: &str, new: &str) -> String {
    TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(2)
        .to_string()
}

/// **Patches built, stored and applied — into the overlay and onto disk.**
///
/// For each case a workspace file is edited eight times. Each edit is a
/// unified diff generated against the file as it stands, applied by
/// `file_edit`'s patch engine, and recorded twice: in an overlay store, which
/// holds deltas and never touches its disk, and in a direct store, which
/// writes the existing file on disk. After every step both hold the patch
/// engine's result.
///
/// Then the overlay's recorded chain is applied elsewhere: to a direct store
/// over a pristine copy of the workspace, putting exactly the conversation's
/// text on disk, and to a fresh overlay over another pristine copy,
/// reproducing the conversation's view there.
#[test]
fn patches_are_built_stored_and_applied_to_the_overlay_and_to_disk() {
    let mut rng = Lcg(0x9a7c4);
    let mut unique = 0;
    for case in 0..60 {
        let mut doc = patchable_doc(&mut rng, &mut unique);
        let original = doc.text();
        let [overlay_root, direct_root, replay_root, fresh_root] =
            [(); 4].map(|_| tempfile::tempdir().unwrap());
        for root in [&overlay_root, &direct_root, &replay_root, &fresh_root] {
            put(root.path(), "src/file.rs", &original);
        }
        let overlay = VfsStore::with_root(overlay_root.path());
        let disk = direct(direct_root.path());
        let mut doc_text = original.clone();

        for step in 0..8 {
            let next = patchable_edit(&mut rng, &doc, &mut unique);
            // Changes that cancel out leave nothing to patch.
            if next.text() == doc_text {
                continue;
            }
            let diff = unified(&doc_text, &next.text());
            let patched = patch::apply(&doc_text, &diff)
                .unwrap_or_else(|e| panic!("case {case} step {step}: {e}\n{diff}"));
            assert_eq!(
                patched.content,
                next.text(),
                "case {case} step {step}: the engine's result"
            );

            overlay
                .edit("src/file.rs", patched.content.clone())
                .unwrap();
            disk.edit("src/file.rs", patched.content.clone()).unwrap();
            assert_eq!(
                overlay.read("src/file.rs").unwrap().as_deref(),
                Some(patched.content.as_str()),
                "case {case} step {step}: overlay"
            );
            assert_eq!(
                on_disk(direct_root.path(), "src/file.rs"),
                patched.content,
                "case {case} step {step}: disk"
            );
            doc_text = patched.content;
            doc = next;
        }

        // The overlay never touched its disk, and holds deltas, not copies.
        assert_eq!(on_disk(overlay_root.path(), "src/file.rs"), original);
        let chain = overlay
            .deltas("src/file.rs")
            .expect("the overlay recorded the edits");
        let held: usize = chain.iter().map(|t| t.delta.bytes()).sum();
        assert_eq!(overlay.total_bytes(), held, "case {case}");
        assert!(
            disk.deltas("src/file.rs").is_none(),
            "a direct store holds none"
        );

        // The stored chain applied onto disk puts the conversation's text there.
        let onto_disk = direct(replay_root.path());
        onto_disk.apply("src/file.rs", &chain).unwrap();
        assert_eq!(
            on_disk(replay_root.path(), "src/file.rs"),
            doc_text,
            "case {case}: stored deltas applied to disk"
        );

        // And applied to a fresh overlay it reproduces the view.
        let fresh = VfsStore::with_root(fresh_root.path());
        fresh.apply("src/file.rs", &chain).unwrap();
        assert_eq!(
            fresh.read("src/file.rs").unwrap().as_deref(),
            Some(doc_text.as_str()),
            "case {case}: stored deltas applied to a fresh overlay"
        );
        assert_eq!(on_disk(fresh_root.path(), "src/file.rs"), original);
    }
}

/// **Deltas that do not fit are refused, in both kinds of store, and change
/// nothing** — the file on disk and the overlay's view both stay as they were.
#[test]
fn deltas_that_do_not_fit_are_refused_everywhere() {
    let root = tempfile::tempdir().unwrap();
    put(root.path(), "a.txt", "one\ntwo\nthree\n");
    let stale = [timed(
        1,
        FileDelta::Edit {
            splices: vec![splice(4, "TWO\n", "2\n")],
        },
    )];

    let disk = direct(root.path());
    assert!(disk.apply("a.txt", &stale).is_err());
    assert_eq!(on_disk(root.path(), "a.txt"), "one\ntwo\nthree\n");

    let overlay = VfsStore::with_root(root.path());
    overlay
        .edit("a.txt", "one\ntwo\nthree\nfour\n".into())
        .unwrap();
    let before = overlay.deltas("a.txt");
    assert!(overlay.apply("a.txt", &stale).is_err());
    assert_eq!(overlay.deltas("a.txt"), before);
    assert_eq!(
        overlay.read("a.txt").unwrap().as_deref(),
        Some("one\ntwo\nthree\nfour\n")
    );
}

fn timed(at_ns: i64, delta: FileDelta) -> TimedDelta {
    TimedDelta { at_ns, delta }
}

/// Applied deltas behave as the calls that made them would: a replace
/// supersedes the chain, a delete on a direct store removes the file, and a
/// delete of a file only the overlay made leaves nothing recorded. Each keeps
/// the moment it was made, and the file's times come from them.
#[test]
fn applied_replace_and_delete_behave_as_write_and_delete() {
    let root = tempfile::tempdir().unwrap();
    put(root.path(), "a.txt", "one\n");

    let overlay = VfsStore::with_root(root.path());
    overlay.edit("a.txt", "one\ntwo\n".into()).unwrap();
    let fresh = timed(
        100,
        FileDelta::Replace {
            content: "fresh\n".into(),
        },
    );
    overlay
        .apply("a.txt", std::slice::from_ref(&fresh))
        .unwrap();
    assert_eq!(overlay.deltas("a.txt"), Some(vec![fresh]));
    assert_eq!(
        overlay.times("a.txt"),
        Some(FileTimes {
            ctime_ns: 100,
            mtime_ns: 100
        })
    );
    overlay
        .apply("a.txt", &[timed(200, FileDelta::Delete)])
        .unwrap();
    assert_eq!(overlay.read("a.txt").unwrap(), None);
    assert_eq!(on_disk(root.path(), "a.txt"), "one\n");

    overlay
        .apply(
            "made.txt",
            &[
                timed(
                    1,
                    FileDelta::Replace {
                        content: "x\n".into(),
                    },
                ),
                timed(2, FileDelta::Delete),
            ],
        )
        .unwrap();
    assert_eq!(overlay.deltas("made.txt"), None);
    assert_eq!(overlay.times("made.txt"), None);

    let disk = direct(root.path());
    disk.apply("a.txt", &[timed(3, FileDelta::Delete)]).unwrap();
    assert!(!root.path().join("a.txt").exists());
}

/// **Every VFS operation is stamped with the moment it executed**: a write
/// starts the file's chain, edits after it move only its modification time,
/// and a later write starts the chain again.
#[test]
fn vfs_operations_record_when_they_executed() {
    let root = tempfile::tempdir().unwrap();
    put(root.path(), "a.txt", "one\ntwo\nthree\nfour\n");
    let s = VfsStore::with_root(root.path());
    assert_eq!(s.times("a.txt"), None, "untouched");

    let t0 = file_delta::now_ns();
    s.edit("a.txt", "one\n2\nthree\nfour\n".into()).unwrap();
    let first = s.times("a.txt").unwrap();
    assert!(first.ctime_ns >= t0 && first.mtime_ns == first.ctime_ns);

    std::thread::sleep(std::time::Duration::from_millis(5));
    s.edit("a.txt", "one\n2\nthree\n4\n".into()).unwrap();
    let second = s.times("a.txt").unwrap();
    assert_eq!(
        second.ctime_ns, first.ctime_ns,
        "the chain began at the first edit"
    );
    assert!(second.mtime_ns > first.mtime_ns);

    std::thread::sleep(std::time::Duration::from_millis(5));
    s.write("a.txt", "rewritten\n".into()).unwrap();
    let third = s.times("a.txt").unwrap();
    assert!(
        third.ctime_ns > second.mtime_ns,
        "a write starts the chain again"
    );
    assert_eq!(third.ctime_ns, third.mtime_ns);
    let chain = s.deltas("a.txt").unwrap();
    assert!(chain.windows(2).all(|w| w[0].at_ns <= w[1].at_ns));

    s.delete("a.txt");
    assert!(s.times("a.txt").unwrap().mtime_ns >= third.mtime_ns);
}
