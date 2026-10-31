//! The repository's files into a [`Tree`]. Nothing here fails: every fault —
//! a stray file, a folder at no place in the tree, a file that does not parse
//! — is kept as a problem on the journal or owner it belongs to (§3.3), and
//! the rest of the tree is read.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::Problem;
use crate::files::{entry_id, in_order, read_entry, GroupFile, Meta, PersonFile, README};
use crate::line::Line;
use crate::path::JournalPath;
use crate::period::Period;
use crate::slug::Slug;
use crate::tree::check::check;
use crate::tree::{skeleton, Group, Journal, Person, Tree};

/// Files at the top of the repository that are not journals.
pub const TOP_FILES: &[&str] = &[".gitattributes", "README.md"];

const PERSONS: &str = "persons";
const GROUPS: &str = "groups";
pub const PERSON_FILE: &str = "person.yaml";
pub const GROUP_FILE: &str = "group.yaml";
pub const JOURNAL_FILE: &str = "journal.yaml";
pub const ARCHIVE_DIR: &str = "archive";

/// One group's files, by path below the group's folder.
type GroupFiles<'a> = BTreeMap<Vec<&'a str>, (&'a str, &'a [u8])>;

/// Read every file of the repository — `path → bytes`, `/`-separated — into
/// a tree, and check it.
pub fn load(files: &BTreeMap<String, Vec<u8>>) -> Tree {
    let mut tree = Tree::default();
    let mut groups: BTreeMap<Slug, GroupFiles<'_>> = BTreeMap::new();
    for (path, bytes) in files {
        let segs: Vec<&str> = path.split('/').collect();
        if segs.len() == 1 && TOP_FILES.contains(&segs[0]) {
            continue;
        }
        if segs.len() < 3 || ![PERSONS, GROUPS].contains(&segs[0]) {
            tree.problems.push(Problem::new(
                path.as_str(),
                None,
                "is no part of the layout: the top holds persons/, groups/, .gitattributes and README.md",
            ));
            continue;
        }
        let slug = match Slug::parse(segs[1]) {
            Ok(s) => s,
            Err(why) => {
                tree.problems.push(Problem::new(path.as_str(), None, why));
                continue;
            }
        };
        if segs[0] == GROUPS {
            groups
                .entry(slug)
                .or_default()
                .insert(segs[2..].to_vec(), (path.as_str(), bytes.as_slice()));
        } else if segs[2..] == [PERSON_FILE] {
            let file = text(path, bytes).and_then(|t| PersonFile::read(path, t));
            tree.persons.insert(
                slug,
                Person {
                    file,
                    problems: Vec::new(),
                },
            );
        } else {
            tree.problems.push(Problem::new(
                path.as_str(),
                None,
                format!("persons/{slug}/ holds person.yaml alone: a person's journals are their groups'"),
            ));
        }
    }
    for (group, files) in groups {
        load_group(&mut tree, group, files);
    }
    check(&mut tree);
    tree
}

fn load_group(tree: &mut Tree, group: Slug, files: GroupFiles<'_>) {
    let mut root = skeleton();
    let folder = format!("{GROUPS}/{group}");
    let mut own: Option<(&str, &[u8])> = None;
    for (rel, (path, bytes)) in &files {
        if rel.len() == 1 && rel[0] == GROUP_FILE {
            own = Some((path, bytes));
            continue;
        }
        if rel.len() == 1 && rel[0] == PERSON_FILE {
            root.problems.push(Problem::new(
                *path,
                None,
                format!("{PERSON_FILE} belongs under {PERSONS}/"),
            ));
            continue;
        }
        // The service writes the group's README from its own file; what a
        // hand edit left there is written over, never read.
        if rel.len() == 1 && rel[0] == README {
            continue;
        }
        if let Err(p) = place_file(&mut root, &group, rel, path, bytes) {
            tree.problems.push(p);
        }
    }
    order(&mut root);
    let file = match own {
        Some((path, bytes)) => text(path, bytes).and_then(|t| GroupFile::read(path, t)),
        None => Err(Problem::new(
            format!("{folder}/{GROUP_FILE}"),
            None,
            format!("{folder} has no {GROUP_FILE}"),
        )),
    };
    tree.groups.insert(
        group,
        Group {
            file,
            root: Arc::new(root),
            problems: Vec::new(),
        },
    );
}

fn text<'a>(path: &str, bytes: &'a [u8]) -> Result<&'a str, Problem> {
    std::str::from_utf8(bytes).map_err(|e| {
        Problem::new(
            path,
            None,
            format!("is not UTF-8 text (byte {})", e.valid_up_to()),
        )
    })
}

/// Every journal's window and archives in the order their lines happened:
/// files are read in name order, and two lines' minutes say nothing of their
/// seconds.
fn order(j: &mut Journal) {
    in_order(&mut j.window);
    for lines in j.archives.values_mut() {
        let lines: &mut Vec<Line> = Arc::make_mut(lines);
        in_order(lines);
    }
    for child in j.children.values_mut() {
        order(child);
    }
}

/// What a journal's folder holds, as its problems say it.
const HOLDS: &str = "a journal's folder holds journal.yaml, an entry per \
`yyyy-mm-dd-hhmm-xxxx.yaml`, archive/<period>/ and its sub-journals' folders";

/// A journal file into its place. A folder at no place in the tree is the
/// returned problem; a fault in the file is kept on its journal.
fn place_file(
    root: &mut Journal,
    group: &Slug,
    rel: &[&str],
    path: &str,
    bytes: &[u8],
) -> Result<(), Problem> {
    let (file, dirs) = rel.split_last().expect("a file has a name");
    // `archive/<period>/<id>.yaml`, or a file straight under `archive/`.
    let (dirs, archive) = match dirs {
        [rest @ .., a, period] if *a == ARCHIVE_DIR => (rest, Some(Some(*period))),
        [rest @ .., a] if *a == ARCHIVE_DIR => (rest, Some(None)),
        _ => (dirs, None),
    };
    let names = dirs
        .iter()
        .map(|d| Slug::parse(d))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|why| Problem::new(path, None, format!("its folder is no journal: {why}")))?;
    let jpath = JournalPath::new(group.clone(), names)
        .map_err(|e| Problem::new(path, None, format!("its folder is no journal: {e}")))?;
    let journal = node(root, &jpath);
    let read = text(path, bytes);
    let entry = |name: &str| {
        entry_id(name)
            .ok_or_else(|| Problem::new(path, None, format!("`{name}` is no entry: {HOLDS}")))
    };
    match (archive, *file) {
        (Some(None), name) => journal.problems.push(Problem::new(
            path,
            None,
            format!("`{name}` is straight under archive/: an archive holds a folder per period, archive/<period>/"),
        )),
        (Some(Some(period)), name) => {
            let period = Period::parse(period).map_err(|why| Problem::new(path, None, why));
            match (period, entry(name), read) {
                (Err(p), _, _) | (_, Err(p), _) | (_, _, Err(p)) => journal.problems.push(p),
                (Ok(period), Ok(id), Ok(t)) => match read_entry(path, id, t) {
                    Ok(line) => {
                        Arc::make_mut(journal.archives.entry(period).or_default()).push(line)
                    }
                    Err(p) => journal.problems.push(p),
                },
            }
        }
        (None, JOURNAL_FILE) => match read.and_then(|t| Meta::read(path, t)) {
            Ok(meta) => journal.meta = Some(meta),
            Err(p) => journal.problems.push(p),
        },
        (None, name) => match (entry(name), read) {
            (Err(p), _) | (_, Err(p)) => journal.problems.push(p),
            (Ok(id), Ok(t)) => match read_entry(path, id, t) {
                Ok(line) => journal.window.push(line),
                Err(p) => journal.problems.push(p),
            },
        },
    }
    Ok(())
}

/// The journal at `path` below `root`, made — with every named journal on
/// the way — when the files reach it first.
fn node<'a>(root: &'a mut Journal, path: &JournalPath) -> &'a mut Journal {
    let mut j = root;
    for depth in 0..path.names().len() {
        let name = &path.names()[depth];
        let place = JournalPath::new(path.group().clone(), path.names()[..=depth].to_vec())
            .and_then(|p| p.place())
            .expect("a prefix of a place is a place");
        j = j
            .children
            .get_or_insert_with(name.clone(), || Journal::new(place, None));
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::entry_file;
    use crate::line::Line;
    use crate::testing::{note, slug};

    fn files(pairs: &[(&str, &str)]) -> BTreeMap<String, Vec<u8>> {
        pairs
            .iter()
            .map(|(p, t)| (p.to_string(), t.as_bytes().to_vec()))
            .collect()
    }

    const JOHN: &str = "person: john\nname: \"John\"\naccounts:\n  - {email: \"john@example.com\"}\ngroups: [zen]\ndefault_group: zen\n";
    const ZEN: &str = "group: zen\nname: \"Zen\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n";

    #[test]
    fn a_seeded_repository_loads_clean() {
        let tree = load(&files(&[
            (".gitattributes", "*.yaml text eol=lf\n"),
            ("README.md", "# journals\n"),
            ("persons/john/person.yaml", JOHN),
            ("groups/zen/group.yaml", ZEN),
        ]));
        assert_eq!(tree.problems, vec![]);
        assert!(tree.person_file(&slug("john")).is_some());
        assert!(tree.group_file(&slug("zen")).is_some());
        let all = tree.journals();
        assert!(all.iter().all(|(_, j)| j.problems.is_empty()));
        // The master and its five sections; a person has no journals.
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn journals_and_their_files_take_their_places() {
        // Read in file-name order — the same minute, the later second first
        // by id — and held in the order they happened.
        let early = note("2026-10-30T09:10:02Z", "Workers push branches.");
        let late = note("2026-10-30T09:10:59Z", "Merging is the user's decision.");
        let old = note("2026-10-29T08:00:00Z", "archived");
        let at = |l: &Line| format!("groups/zen/topics/candle/{}", entry_file(&l.id));
        let tree = load(&files(&[
            ("persons/john/person.yaml", JOHN),
            ("groups/zen/README.md", "# anything a hand left\n"),
            ("groups/zen/group.yaml", ZEN),
            (
                "groups/zen/topics/candle/journal.yaml",
                "name: candle\ncue: \"open for the engine\"\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n",
            ),
            (&at(&late), &late.render()),
            (&at(&early), &early.render()),
            (
                &format!(
                    "groups/zen/topics/candle/archive/2026-10-29/{}",
                    entry_file(&old.id)
                ),
                &old.render(),
            ),
        ]));
        let candle = tree
            .get(&JournalPath::new(slug("zen"), vec![slug("topics"), slug("candle")]).unwrap())
            .unwrap();
        assert_eq!(candle.window, vec![early, late]);
        assert_eq!(
            *candle.archives[&Period::parse("2026-10-29").unwrap()],
            vec![old]
        );
        assert_eq!(candle.problems, vec![]);
        assert_eq!(tree.problems, vec![]);
        assert_eq!(candle.cue().as_deref(), Some("open for the engine"));
    }

    #[test]
    fn every_stray_is_a_problem_and_the_rest_still_loads() {
        let tree = load(&files(&[
            ("persons/john/person.yaml", JOHN),
            ("groups/zen/group.yaml", ZEN),
            ("notes.txt", "x"),
            ("families/sharratt/family.yaml", "x"),
            ("persons/README.md", "x"),
            ("persons/John/person.yaml", JOHN),
            ("persons/john/journal.yaml", "x"),
            ("persons/john/channels/grok/journal.yaml", "x"),
            ("groups/zen/channels/grok/lines.yaml", ""),
            ("groups/zen/projects/candle/lines.yaml", ""),
            ("groups/zen/people/Yen/lines.yaml", ""),
            ("groups/zen/people/archive/lines.yaml", ""),
        ]));
        let mut stray: Vec<&str> = tree.problems.iter().map(|p| p.file.as_str()).collect();
        stray.sort_unstable();
        assert_eq!(
            stray,
            [
                "families/sharratt/family.yaml",
                "groups/zen/channels/grok/lines.yaml",
                "groups/zen/people/Yen/lines.yaml",
                "groups/zen/projects/candle/lines.yaml",
                "notes.txt",
                "persons/John/person.yaml",
                "persons/README.md",
                "persons/john/channels/grok/journal.yaml",
                "persons/john/journal.yaml",
            ]
        );
        assert!(tree.problems.iter().any(|p| p.what
            == "persons/john/ holds person.yaml alone: a person's journals are their groups'"));
        assert!(tree.person_file(&slug("john")).is_some());
        // `people/archive/lines.yaml` sits straight under the archive folder
        // of `people`, where a folder per period belongs.
        let people = &tree.groups[&slug("zen")].root.children[&slug("people")];
        assert!(
            people.problems[0].what.contains("straight under archive/"),
            "{:?}",
            people.problems
        );
    }

    #[test]
    fn a_broken_file_is_kept_on_its_journal() {
        let tree = load(&files(&[
            ("groups/zen/group.yaml", ZEN),
            (
                "groups/zen/conversations/2026-10-30-0910-0001.yaml",
                "at: yesterday\nkind: note\nby: grok\ntext: x\n",
            ),
            ("groups/zen/topics/journal.yaml", "name: [\n"),
            ("groups/zen/threads/2026-10-30-0910-0002.yaml", "\u{0}"),
            ("groups/zen/people/lines.yaml", "- {id: x}\n"),
            ("groups/zen/people/archive/2026-10/notes.md", "x"),
            (
                "groups/zen/conversations/archive/someday/2026-10-30-0910-0003.yaml",
                "x",
            ),
        ]));
        let root = &tree.groups[&slug("zen")].root;
        let conversations = &root.children[&slug("conversations")];
        assert_eq!(
            conversations.problems[0].to_string(),
            "groups/zen/conversations/2026-10-30-0910-0001.yaml:1: `at`: `yesterday` is not a time written 2026-10-30T09:10:02Z"
        );
        assert!(
            conversations.problems[1].what.contains("not a period"),
            "{:?}",
            conversations.problems
        );
        assert!(root.children[&slug("topics")].problems[0]
            .what
            .starts_with("not valid YAML"));
        assert!(!root.children[&slug("threads")].problems.is_empty());
        let people = &root.children[&slug("people")];
        assert!(people.problems[0]
            .what
            .starts_with("`notes.md` is no entry"));
        assert_eq!(
            people.problems[1].what,
            format!("`lines.yaml` is no entry: {HOLDS}")
        );
    }

    #[test]
    fn non_utf8_and_missing_owner_files_are_problems() {
        let mut f = files(&[("groups/zen/topics/journal.yaml", "x")]);
        f.insert("persons/john/person.yaml".into(), vec![0xff, 0xfe]);
        let tree = load(&f);
        let err = tree.persons[&slug("john")].file.as_ref().unwrap_err();
        assert!(err.what.contains("not UTF-8"), "{err}");
        let err = tree.groups[&slug("zen")].file.as_ref().unwrap_err();
        assert_eq!(
            err.to_string(),
            "groups/zen/group.yaml: groups/zen has no group.yaml"
        );
    }

    #[test]
    fn a_person_file_in_a_group_folder_is_a_problem() {
        let tree = load(&files(&[
            ("groups/zen/group.yaml", ZEN),
            ("groups/zen/person.yaml", JOHN),
        ]));
        let root = &tree.groups[&slug("zen")].root;
        assert_eq!(root.problems[0].what, "person.yaml belongs under persons/");
    }

    #[test]
    fn intermediate_named_folders_without_metadata_are_broken() {
        let tree = load(&files(&[
            ("groups/zen/group.yaml", ZEN),
            (
                "groups/zen/people/family/yen/journal.yaml",
                "name: yen\ncue: \"open for Yen\"\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n",
            ),
        ]));
        let family =
            &tree.groups[&slug("zen")].root.children[&slug("people")].children[&slug("family")];
        assert!(
            family
                .problems
                .iter()
                .any(|p| p.what.contains("has no journal.yaml")),
            "{:?}",
            family.problems
        );
        assert!(family.children[&slug("yen")].problems.is_empty());
    }
}
