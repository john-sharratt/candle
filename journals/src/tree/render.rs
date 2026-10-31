//! A tree back into files. A journal's own files are what its folder holds,
//! its children aside: `journal.yaml` when it has metadata, a file per line
//! of its window, and a folder per folded period holding a file per line.
//! Each group's folder opens with `group.yaml` and a README; each person's
//! holds `person.yaml` alone.
//!
//! A journal with problems is never rendered for a write — writes to it are
//! refused (§3.3) — so what its broken files said is never overwritten.

use std::collections::BTreeMap;

use crate::files::{entry_file, group_readme, README};
use crate::path::JournalPath;
use crate::slug::Slug;
use crate::tree::load::{ARCHIVE_DIR, GROUP_FILE, JOURNAL_FILE, PERSON_FILE};
use crate::tree::{Journal, Tree};

/// `path → bytes` for the files of `j` itself, at `path`.
pub fn own_files(path: &JournalPath, j: &Journal) -> BTreeMap<String, Vec<u8>> {
    let folder = path.folder();
    let mut out = BTreeMap::new();
    if let Some(m) = &j.meta {
        out.insert(format!("{folder}/{JOURNAL_FILE}"), m.render().into_bytes());
    }
    for line in &j.window {
        out.insert(
            format!("{folder}/{}", entry_file(&line.id)),
            line.render().into_bytes(),
        );
    }
    for (period, lines) in &j.archives {
        for line in lines.iter() {
            out.insert(
                format!("{folder}/{ARCHIVE_DIR}/{period}/{}", entry_file(&line.id)),
                line.render().into_bytes(),
            );
        }
    }
    out
}

/// A group's own files, `group.yaml` and the README written from it, when it
/// reads.
pub fn group_files(tree: &Tree, group: &Slug) -> Vec<(String, Vec<u8>)> {
    let Some(f) = tree.group_file(group) else {
        return Vec::new();
    };
    vec![
        (
            format!("groups/{group}/{GROUP_FILE}"),
            f.render().into_bytes(),
        ),
        (
            format!("groups/{group}/{README}"),
            group_readme(f).into_bytes(),
        ),
    ]
}

/// A person's `person.yaml`, when it reads.
pub fn person_files(tree: &Tree, person: &Slug) -> Vec<(String, Vec<u8>)> {
    tree.person_file(person)
        .map(|f| {
            (
                format!("persons/{person}/{PERSON_FILE}"),
                f.render().into_bytes(),
            )
        })
        .into_iter()
        .collect()
}

/// Every file of the repository.
pub fn render_all(tree: &Tree) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for person in tree.persons.keys() {
        out.extend(person_files(tree, person));
    }
    for group in tree.groups.keys() {
        out.extend(group_files(tree, group));
    }
    for (path, j) in tree.journals() {
        out.extend(own_files(&path, j));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::files::summary::{Cite, Fingerprint, Sentence};
    use crate::files::{Focus, Meta, Ref, Summary};
    use crate::path::{Below, Place};
    use crate::period::Period;
    use crate::testing::{at, note, slug};
    use crate::tree::load::load;

    const JOHN: &str = "person: john\nname: John\naccounts:\n  - email: john@example.com\n    key: 154797f34c5a6ca6\ngroups: [zen]\ndefault_group: zen\n";
    const ZEN: &str = "group: zen\nname: Zen\nmembers:\n  - person: john\n    role: owner\n    joined: 2026-10-30T00:00:00Z\n";

    /// A rich tree survives files → tree → files → tree unchanged, byte for byte.
    #[test]
    fn a_tree_round_trips_through_its_files() {
        let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        files.insert("persons/john/person.yaml".into(), JOHN.into());
        files.insert("groups/zen/group.yaml".into(), ZEN.into());
        let mut tree = load(&files);
        let candle_path =
            JournalPath::new(slug("zen"), vec![slug("topics"), slug("candle")]).unwrap();
        let mut candle = Journal::new(
            candle_path.place().unwrap(),
            Some(Meta::new(
                "candle",
                Some("open for the engine".into()),
                at("2026-10-30T09:10:02Z"),
            )),
        );
        candle.meta.as_mut().unwrap().repo = Some(slug("candle"));
        candle.meta.as_mut().unwrap().refs.push(Ref {
            name: slug("john"),
            to: Below::parse("people/john").unwrap(),
            cue: "open for who leads it".into(),
        });
        let old = note("2026-10-28T10:00:00Z", "archived");
        candle.archives.insert(
            Period::parse("2026-10-28").unwrap(),
            Arc::new(vec![old.clone()]),
        );
        let fresh = note("2026-10-30T10:00:00Z", "fresh");
        candle.window.push(fresh.clone());
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
        let john_path = JournalPath::new(slug("zen"), vec![slug("people"), slug("john")]).unwrap();
        let john = Journal::new(
            john_path.place().unwrap(),
            Some(Meta::new(
                "john",
                Some("open for John".into()),
                at("2026-10-30T09:10:02Z"),
            )),
        );
        let conv_path = JournalPath::new(
            slug("zen"),
            vec![slug("conversations"), slug("2026-10-31-0712-grok")],
        )
        .unwrap();
        let mut conv = Journal::new(
            conv_path.place().unwrap(),
            Some(Meta::new(
                "2026-10-31-0712-grok",
                Some("open for a drive".into()),
                at("2026-10-31T07:12:04Z"),
            )),
        );
        conv.meta.as_mut().unwrap().focus = Some(Focus {
            speaker: Some(slug("john")),
            about: vec![Below::parse("topics/candle").unwrap()],
        });
        tree.get_mut(&candle_path.parent().unwrap())
            .unwrap()
            .children
            .insert(slug("candle"), candle);
        tree.get_mut(&john_path.parent().unwrap())
            .unwrap()
            .children
            .insert(slug("john"), john);
        tree.get_mut(&conv_path.parent().unwrap())
            .unwrap()
            .children
            .insert(slug("2026-10-31-0712-grok"), conv);
        let rendered = render_all(&tree);
        assert_eq!(
            rendered.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "groups/zen/README.md".to_string(),
                "groups/zen/conversations/2026-10-31-0712-grok/journal.yaml".into(),
                "groups/zen/group.yaml".into(),
                "groups/zen/people/john/journal.yaml".into(),
                format!("groups/zen/topics/candle/{}", entry_file(&fresh.id)),
                format!(
                    "groups/zen/topics/candle/archive/2026-10-28/{}",
                    entry_file(&old.id)
                ),
                "groups/zen/topics/candle/journal.yaml".into(),
                "persons/john/person.yaml".into(),
            ]
        );
        assert_eq!(rendered["persons/john/person.yaml"], JOHN.as_bytes());
        assert_eq!(rendered["groups/zen/group.yaml"], ZEN.as_bytes());
        assert!(String::from_utf8_lossy(&rendered["groups/zen/README.md"])
            .starts_with("# @zen — Zen\n\nThis folder holds the group's journals"));
        let reloaded = load(&rendered);
        assert_eq!(reloaded, tree);
        assert_eq!(render_all(&reloaded), rendered);
    }

    #[test]
    fn an_empty_window_writes_no_files() {
        let path = JournalPath::root(slug("zen"));
        let j = Journal::new(Place::Root, None);
        assert!(own_files(&path, &j).is_empty());
    }
}
