//! A tree back into files. A journal's own files are what its folder holds,
//! its children aside: `journal.yaml` when it has metadata, `lines.yaml` when
//! its window holds lines, an archive per folded period.
//!
//! A journal with problems is never rendered for a write — writes to it are
//! refused (§3.3) — so what its broken files said is never overwritten.

use std::collections::BTreeMap;

use crate::files::render_lines;
use crate::path::{JournalPath, Owner};
use crate::tree::load::{ARCHIVE_DIR, FAMILY_FILE, JOURNAL_FILE, LINES_FILE, PERSON_FILE};
use crate::tree::{Journal, Tree};

/// `path → bytes` for the files of `j` itself, at `path`.
pub fn own_files(path: &JournalPath, j: &Journal) -> BTreeMap<String, Vec<u8>> {
    let folder = path.folder();
    let mut out = BTreeMap::new();
    if let Some(m) = &j.meta {
        out.insert(format!("{folder}/{JOURNAL_FILE}"), m.render().into_bytes());
    }
    if !j.window.is_empty() {
        out.insert(
            format!("{folder}/{LINES_FILE}"),
            render_lines(&j.window).into_bytes(),
        );
    }
    for (period, lines) in &j.archives {
        if !lines.is_empty() {
            out.insert(
                format!("{folder}/{ARCHIVE_DIR}/{period}.yaml"),
                render_lines(lines).into_bytes(),
            );
        }
    }
    out
}

/// The owner's own file, `person.yaml` or `family.yaml`, when it reads.
pub fn owner_file(tree: &Tree, owner: &Owner) -> Option<(String, Vec<u8>)> {
    let folder = owner.folder();
    match owner {
        Owner::Person(p) => tree
            .person_file(p)
            .map(|f| (format!("{folder}/{PERSON_FILE}"), f.render().into_bytes())),
        Owner::Family(f) => tree
            .family_file(f)
            .map(|x| (format!("{folder}/{FAMILY_FILE}"), x.render().into_bytes())),
    }
}

/// Every file of every owner.
pub fn render_all(tree: &Tree) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let owners = tree
        .persons
        .keys()
        .map(|p| Owner::Person(p.clone()))
        .chain(tree.families.keys().map(|f| Owner::Family(f.clone())));
    for owner in owners {
        if let Some((p, b)) = owner_file(tree, &owner) {
            out.insert(p, b);
        }
    }
    for (path, j) in tree.journals() {
        out.extend(own_files(&path, j));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::summary::{Cite, Fingerprint, Sentence};
    use crate::files::{Meta, Summary};
    use crate::path::Place;
    use crate::period::Period;
    use crate::testing::{at, note, slug};
    use crate::tree::load::load;

    const JOHN: &str = "person: john\nname: \"John\"\naccounts:\n  - {email: \"john@example.com\", key: 154797f34c5a6ca6}\nfamilies: [sharratt]\ndefault_family: sharratt\n";
    const SHARRATT: &str = "family: sharratt\nname: \"Sharratt\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n";

    /// A rich tree survives files → tree → files → tree unchanged, byte for byte.
    #[test]
    fn a_tree_round_trips_through_its_files() {
        let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        files.insert("persons/john/person.yaml".into(), JOHN.into());
        files.insert("families/sharratt/family.yaml".into(), SHARRATT.into());
        let mut tree = load(&files);
        let candle_path = JournalPath::new(
            Owner::Family(slug("sharratt")),
            vec![slug("projects"), slug("candle")],
        )
        .unwrap();
        let projects = tree.get_mut(&candle_path.parent().unwrap()).unwrap();
        let mut candle = Journal::new(
            candle_path.place().unwrap(),
            Some(Meta::new(
                "candle",
                Some("open for the engine".into()),
                at("2026-10-30T09:10:02Z"),
            )),
        );
        let old = note("2026-10-28T10:00:00Z", "archived");
        candle
            .archives
            .insert(Period::parse("2026-10-28").unwrap(), vec![old.clone()]);
        candle.window.push(note("2026-10-30T10:00:00Z", "fresh"));
        candle.meta.as_mut().unwrap().summary = Some(Summary {
            over: Fingerprint::of_bytes(b"x"),
            built: at("2026-10-30T11:00:00Z"),
            lines: 1,
            children: 0,
            stale: false,
            digest: None,
            sentences: vec![Sentence {
                text: "It was archived.".into(),
                cites: vec![Cite::Line(old.id.clone())],
            }],
        });
        projects.children.insert(slug("candle"), candle);
        let rendered = render_all(&tree);
        assert_eq!(
            rendered.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "families/sharratt/family.yaml",
                "families/sharratt/projects/candle/archive/2026-10-28.yaml",
                "families/sharratt/projects/candle/journal.yaml",
                "families/sharratt/projects/candle/lines.yaml",
                "persons/john/person.yaml",
            ]
        );
        assert_eq!(rendered["persons/john/person.yaml"], JOHN.as_bytes());
        let reloaded = load(&rendered);
        assert_eq!(reloaded, tree);
        assert_eq!(render_all(&reloaded), rendered);
    }

    #[test]
    fn an_empty_window_writes_no_lines_file() {
        let path = JournalPath::root(Owner::Family(slug("sharratt")));
        let j = Journal::new(Place::Root, None);
        assert!(own_files(&path, &j).is_empty());
    }
}
