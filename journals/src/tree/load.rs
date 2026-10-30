//! The repository's files into a [`Tree`]. Nothing here fails: every fault —
//! a stray file, a folder at no place in the tree, a file that does not parse
//! — is kept as a problem on the journal or owner it belongs to (§3.3), and
//! the rest of the tree is read.

use std::collections::BTreeMap;

use crate::error::Problem;
use crate::files::{read_lines, GroupFile, Meta, PersonFile};
use crate::path::{JournalPath, Owner};
use crate::period::Period;
use crate::scope::Scope;
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
pub const LINES_FILE: &str = "lines.yaml";
pub const ARCHIVE_DIR: &str = "archive";

/// One owner's files, by path below the owner's folder.
type OwnerFiles<'a> = BTreeMap<Vec<&'a str>, (&'a str, &'a [u8])>;

/// Read every file of the repository — `path → bytes`, `/`-separated — into
/// a tree, and check it.
pub fn load(files: &BTreeMap<String, Vec<u8>>) -> Tree {
    let mut tree = Tree::default();
    let mut owners: BTreeMap<Owner, OwnerFiles<'_>> = BTreeMap::new();
    for (path, bytes) in files {
        let segs: Vec<&str> = path.split('/').collect();
        if segs.len() == 1 && TOP_FILES.contains(&segs[0]) {
            continue;
        }
        let kind = match segs[0] {
            PERSONS => Some(Owner::Person as fn(Slug) -> Owner),
            GROUPS => Some(Owner::Group as fn(Slug) -> Owner),
            _ => None,
        };
        let (Some(kind), true) = (kind, segs.len() >= 3) else {
            tree.problems.push(Problem::new(
                path.as_str(),
                None,
                "is no part of the layout: the top holds persons/, groups/, .gitattributes and README.md",
            ));
            continue;
        };
        match Slug::parse(segs[1]) {
            Ok(slug) => {
                owners
                    .entry(kind(slug))
                    .or_default()
                    .insert(segs[2..].to_vec(), (path.as_str(), bytes.as_slice()));
            }
            Err(why) => tree.problems.push(Problem::new(path.as_str(), None, why)),
        }
    }
    for (owner, files) in owners {
        load_owner(&mut tree, owner, files);
    }
    check(&mut tree);
    tree
}

fn load_owner(tree: &mut Tree, owner: Owner, files: OwnerFiles<'_>) {
    let mut root = skeleton(owner.scope());
    let (own_name, other_name) = match owner.scope() {
        Scope::Personal => (PERSON_FILE, GROUP_FILE),
        Scope::Group => (GROUP_FILE, PERSON_FILE),
    };
    let folder = owner.folder();
    let mut own: Option<(&str, &[u8])> = None;
    for (rel, (path, bytes)) in &files {
        if rel.len() == 1 && rel[0] == own_name {
            own = Some((path, bytes));
            continue;
        }
        if rel.len() == 1 && rel[0] == other_name {
            root.problems.push(Problem::new(
                *path,
                None,
                format!(
                    "{other_name} belongs under {}/",
                    if other_name == PERSON_FILE {
                        PERSONS
                    } else {
                        GROUPS
                    }
                ),
            ));
            continue;
        }
        if let Err(p) = place_file(&mut root, &owner, rel, path, bytes) {
            tree.problems.push(p);
        }
    }
    let missing = || {
        Problem::new(
            format!("{folder}/{own_name}"),
            None,
            format!("{folder} has no {own_name}"),
        )
    };
    match owner {
        Owner::Person(p) => {
            let file = match own {
                Some((path, bytes)) => text(path, bytes).and_then(|t| PersonFile::read(path, t)),
                None => Err(missing()),
            };
            tree.persons.insert(
                p,
                Person {
                    file,
                    root,
                    problems: Vec::new(),
                },
            );
        }
        Owner::Group(g) => {
            let file = match own {
                Some((path, bytes)) => text(path, bytes).and_then(|t| GroupFile::read(path, t)),
                None => Err(missing()),
            };
            tree.groups.insert(
                g,
                Group {
                    file,
                    root,
                    problems: Vec::new(),
                },
            );
        }
    }
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

/// A journal file into its place. A folder at no place in the tree is the
/// returned problem; a fault in the file is kept on its journal.
fn place_file(
    root: &mut Journal,
    owner: &Owner,
    rel: &[&str],
    path: &str,
    bytes: &[u8],
) -> Result<(), Problem> {
    let (file, dirs) = rel.split_last().expect("a file has a name");
    let (dirs, archive) = match dirs.split_last() {
        Some((last, rest)) if *last == ARCHIVE_DIR => (rest, true),
        _ => (dirs, false),
    };
    let names = dirs
        .iter()
        .map(|d| Slug::parse(d))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|why| Problem::new(path, None, format!("its folder is no journal: {why}")))?;
    let jpath = JournalPath::new(owner.clone(), names)
        .map_err(|e| Problem::new(path, None, format!("its folder is no journal: {e}")))?;
    let journal = node(root, &jpath);
    let read = text(path, bytes);
    match (archive, *file) {
        (true, name) => {
            let period = name
                .strip_suffix(".yaml")
                .ok_or_else(|| format!("`{name}` is not an archive: archive/<period>.yaml"))
                .and_then(Period::parse);
            match (period, read) {
                (Err(why), _) => journal.problems.push(Problem::new(path, None, why)),
                (Ok(_), Err(p)) => journal.problems.push(p),
                (Ok(period), Ok(t)) => match read_lines(path, t) {
                    Ok(lines) => {
                        journal.archives.insert(period, lines);
                    }
                    Err(p) => journal.problems.push(p),
                },
            }
        }
        (false, JOURNAL_FILE) => match read.and_then(|t| Meta::read(path, t)) {
            Ok(meta) => journal.meta = Some(meta),
            Err(p) => journal.problems.push(p),
        },
        (false, LINES_FILE) => match read.and_then(|t| read_lines(path, t)) {
            Ok(lines) => journal.window = lines,
            Err(p) => journal.problems.push(p),
        },
        (false, other) => journal.problems.push(Problem::new(
            path,
            None,
            format!("`{other}` is not a journal file: journal.yaml, lines.yaml or archive/<period>.yaml"),
        )),
    }
    Ok(())
}

/// The journal at `path` below `root`, made — with every named journal on
/// the way — when the files reach it first.
fn node<'a>(root: &'a mut Journal, path: &JournalPath) -> &'a mut Journal {
    let mut j = root;
    for depth in 0..path.names().len() {
        let name = &path.names()[depth];
        let place = JournalPath::new(path.owner().clone(), path.names()[..=depth].to_vec())
            .and_then(|p| p.place())
            .expect("a prefix of a place is a place");
        j = j
            .children
            .entry(name.clone())
            .or_insert_with(|| Journal::new(place, None));
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::render_lines;
    use crate::testing::{note, slug};

    fn files(pairs: &[(&str, &str)]) -> BTreeMap<String, Vec<u8>> {
        pairs
            .iter()
            .map(|(p, t)| (p.to_string(), t.as_bytes().to_vec()))
            .collect()
    }

    const JOHN: &str = "person: john\nname: \"John\"\naccounts:\n  - {email: \"john@example.com\"}\ngroups: [sharratt]\ndefault_group: sharratt\n";
    const SHARRATT: &str = "group: sharratt\nname: \"Sharratt\"\nkind: family\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n";

    #[test]
    fn a_seeded_repository_loads_clean() {
        let tree = load(&files(&[
            (".gitattributes", "lines.yaml merge=union\n"),
            ("README.md", "# journals\n"),
            ("persons/john/person.yaml", JOHN),
            ("groups/sharratt/group.yaml", SHARRATT),
        ]));
        assert_eq!(tree.problems, vec![]);
        assert!(tree.person_file(&slug("john")).is_some());
        assert!(tree.group_file(&slug("sharratt")).is_some());
        let all = tree.journals();
        assert!(all.iter().all(|(_, j)| j.problems.is_empty()));
        // Root, clients and its three clients, people; root and four sections.
        assert_eq!(all.len(), 6 + 5);
    }

    #[test]
    fn journals_and_their_files_take_their_places() {
        let line = note("2026-10-30T09:10:02Z", "Workers push branches.");
        let tree = load(&files(&[
            ("persons/john/person.yaml", JOHN),
            ("groups/sharratt/group.yaml", SHARRATT),
            (
                "groups/sharratt/projects/candle/journal.yaml",
                "name: candle\ncue: \"open for the engine\"\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n",
            ),
            ("groups/sharratt/projects/candle/lines.yaml", &render_lines(std::slice::from_ref(&line))),
            ("groups/sharratt/projects/candle/archive/2026-10-29.yaml", ""),
        ]));
        let candle = tree
            .get(
                &JournalPath::new(
                    Owner::Group(slug("sharratt")),
                    vec![slug("projects"), slug("candle")],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(candle.window, vec![line]);
        assert_eq!(candle.archives.len(), 1);
        assert_eq!(candle.problems, vec![]);
        assert_eq!(candle.cue().as_deref(), Some("open for the engine"));
    }

    #[test]
    fn every_stray_is_a_problem_and_the_rest_still_loads() {
        let tree = load(&files(&[
            ("persons/john/person.yaml", JOHN),
            ("groups/sharratt/group.yaml", SHARRATT),
            ("notes.txt", "x"),
            ("families/sharratt/family.yaml", "x"),
            ("persons/README.md", "x"),
            ("persons/John/person.yaml", JOHN),
            ("groups/sharratt/clients/grok/lines.yaml", ""),
            ("persons/john/projects/candle/lines.yaml", ""),
            ("persons/john/notes/lines.yaml", ""),
            ("persons/john/clients/tesla/lines.yaml", ""),
            ("persons/john/people/Yen/lines.yaml", ""),
            ("persons/john/people/archive/lines.yaml", ""),
        ]));
        let mut stray: Vec<&str> = tree.problems.iter().map(|p| p.file.as_str()).collect();
        stray.sort_unstable();
        assert_eq!(
            stray,
            [
                "families/sharratt/family.yaml",
                "groups/sharratt/clients/grok/lines.yaml",
                "notes.txt",
                "persons/John/person.yaml",
                "persons/README.md",
                "persons/john/clients/tesla/lines.yaml",
                "persons/john/notes/lines.yaml",
                "persons/john/people/Yen/lines.yaml",
                "persons/john/projects/candle/lines.yaml",
            ]
        );
        assert!(tree.person_file(&slug("john")).is_some());
        // `people/archive/lines.yaml` is the archive folder of `people` holding a
        // file that is no period.
        let people = &tree.persons[&slug("john")].root.children[&slug("people")];
        assert!(
            people.problems[0].what.contains("not a period"),
            "{:?}",
            people.problems
        );
    }

    #[test]
    fn a_broken_file_is_kept_on_its_journal() {
        let tree = load(&files(&[
            ("persons/john/person.yaml", JOHN),
            ("persons/john/clients/grok/lines.yaml", "- {id: nope}\n"),
            ("persons/john/clients/other/journal.yaml", "name: [\n"),
            ("persons/john/clients/claude-code/lines.yaml", "\u{0}"),
            ("persons/john/people/notes.md", "x"),
        ]));
        let clients = &tree.persons[&slug("john")].root.children[&slug("clients")];
        assert!(clients.children[&slug("grok")].problems[0]
            .what
            .contains("not a line id"));
        assert!(clients.children[&slug("other")].problems[0]
            .what
            .starts_with("not valid YAML"));
        assert!(!clients.children[&slug("claude-code")].problems.is_empty());
        let people = &tree.persons[&slug("john")].root.children[&slug("people")];
        assert!(people.problems[0].what.contains("not a journal file"));
    }

    #[test]
    fn non_utf8_and_missing_owner_files_are_problems() {
        let mut f = files(&[("groups/sharratt/projects/journal.yaml", "x")]);
        f.insert("persons/john/person.yaml".into(), vec![0xff, 0xfe]);
        let tree = load(&f);
        let err = tree.persons[&slug("john")].file.as_ref().unwrap_err();
        assert!(err.what.contains("not UTF-8"), "{err}");
        let err = tree.groups[&slug("sharratt")].file.as_ref().unwrap_err();
        assert_eq!(
            err.to_string(),
            "groups/sharratt/group.yaml: groups/sharratt has no group.yaml"
        );
    }

    #[test]
    fn an_owner_file_in_the_wrong_kind_of_folder_is_a_problem() {
        let tree = load(&files(&[
            ("persons/john/person.yaml", JOHN),
            ("persons/john/group.yaml", SHARRATT),
        ]));
        let root = &tree.persons[&slug("john")].root;
        assert_eq!(root.problems[0].what, "group.yaml belongs under groups/");
    }

    #[test]
    fn intermediate_named_folders_without_metadata_are_broken() {
        let tree = load(&files(&[
            ("persons/john/person.yaml", JOHN),
            (
                "persons/john/people/family/yen/journal.yaml",
                "name: yen\ncue: \"open for Yen\"\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n",
            ),
        ]));
        let family =
            &tree.persons[&slug("john")].root.children[&slug("people")].children[&slug("family")];
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
