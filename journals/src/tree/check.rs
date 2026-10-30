//! The rules a loaded tree is held to (§2, §3.3, §5.1), each broken one a
//! problem on the journal or owner that breaks it.

use std::collections::BTreeMap;
use std::iter;

use crate::error::Problem;
use crate::files::{AccountKey, Email, Who};
use crate::line::{Folds, LineId};
use crate::path::{JournalPath, Owner, Place};
use crate::scope::Scope;
use crate::slug::Slug;
use crate::tree::load::{JOURNAL_FILE, LINES_FILE};
use crate::tree::{Journal, Tree, MAX_CHILDREN};

/// The longest a journal's summary may be, in characters (§5.1).
pub fn summary_cap(place: Place) -> usize {
    match place {
        Place::Root => 1_500,
        Place::Section(_) => 1_000,
        Place::Client(_) | Place::Named { .. } => 600,
    }
}

/// The longest its voice digest may be; `None` where it has none: only a root
/// has one.
pub fn digest_cap(place: Place, scope: Scope) -> Option<usize> {
    match (place, scope) {
        (Place::Root, Scope::Personal) => Some(600),
        (Place::Root, Scope::Group) => Some(400),
        _ => None,
    }
}

pub fn check(tree: &mut Tree) {
    check_owners(tree);
    check_accounts(tree);
    let owners: Vec<Owner> = tree
        .persons
        .keys()
        .map(|p| Owner::Person(p.clone()))
        .chain(tree.groups.keys().map(|g| Owner::Group(g.clone())))
        .collect();
    for owner in owners {
        let root = tree.root_mut(&owner).expect("listed above");
        check_journal(root, &JournalPath::root(owner));
    }
}

/// Each `person.yaml` and `group.yaml` against the others: a person lists a
/// group exactly when the group lists them. A bot is listed by its group
/// alone.
fn check_owners(tree: &mut Tree) {
    let lists_group = |tree: &Tree, person: &Slug, group: &Slug| {
        tree.person_file(person)
            .is_some_and(|f| f.groups.contains(group))
    };
    let lists_member = |tree: &Tree, group: &Slug, person: &Slug| {
        tree.group_file(group)
            .is_some_and(|f| f.member(person).is_some())
    };
    let mut person_problems: BTreeMap<Slug, Vec<Problem>> = BTreeMap::new();
    for (slug, person) in &tree.persons {
        let Ok(file) = &person.file else { continue };
        let at = format!("persons/{slug}/person.yaml");
        if &file.person != slug {
            person_problems
                .entry(slug.clone())
                .or_default()
                .push(Problem::new(
                    &at,
                    Some(1),
                    format!("says it is {}, but it is in persons/{slug}/", file.person),
                ));
        }
        for group in &file.groups {
            let why = if !tree.groups.contains_key(group) {
                format!("lists the group {group}, which there is no folder for")
            } else if !lists_member(tree, group, slug) {
                format!("lists the group {group}, which does not list {slug}")
            } else {
                continue;
            };
            person_problems
                .entry(slug.clone())
                .or_default()
                .push(Problem::new(&at, None, why));
        }
    }
    let mut group_problems: BTreeMap<Slug, Vec<Problem>> = BTreeMap::new();
    for (slug, group) in &tree.groups {
        let Ok(file) = &group.file else { continue };
        let at = format!("groups/{slug}/group.yaml");
        if &file.group != slug {
            group_problems
                .entry(slug.clone())
                .or_default()
                .push(Problem::new(
                    &at,
                    Some(1),
                    format!("says it is {}, but it is in groups/{slug}/", file.group),
                ));
        }
        for m in &file.members {
            let Who::Person(person) = &m.who else {
                continue;
            };
            let why = if !tree.persons.contains_key(person) {
                format!("lists {person}, who has no folder under persons/")
            } else if !lists_group(tree, person, slug) {
                format!("lists {person}, whose person.yaml does not list {slug}")
            } else {
                continue;
            };
            group_problems
                .entry(slug.clone())
                .or_default()
                .push(Problem::new(&at, None, why));
        }
    }
    for (slug, ps) in person_problems {
        tree.persons
            .get_mut(&slug)
            .expect("from the map")
            .problems
            .extend(ps);
    }
    for (slug, ps) in group_problems {
        tree.groups
            .get_mut(&slug)
            .expect("from the map")
            .problems
            .extend(ps);
    }
}

fn check_accounts(tree: &mut Tree) {
    let mut emails: BTreeMap<&Email, &Slug> = BTreeMap::new();
    let mut keys: BTreeMap<&AccountKey, &Slug> = BTreeMap::new();
    let mut found = Vec::new();
    for (slug, person) in &tree.persons {
        let Ok(file) = &person.file else { continue };
        for a in &file.accounts {
            if let Some(other) = emails.insert(&a.email, slug) {
                found.push(Problem::new(
                    format!("persons/{slug}/person.yaml"),
                    None,
                    format!(
                        "{} is listed by {other} too: an account is one person's",
                        a.email
                    ),
                ));
            }
            if let Some(k) = &a.key {
                if let Some(other) = keys.insert(k, slug) {
                    found.push(Problem::new(
                        format!("persons/{slug}/person.yaml"),
                        None,
                        format!(
                            "the account {k} is listed by {other} too: an account is one person's"
                        ),
                    ));
                }
            }
        }
    }
    tree.problems.extend(found);
}

fn check_journal(j: &mut Journal, path: &JournalPath) {
    let folder = path.folder();
    let meta_file = format!("{folder}/{JOURNAL_FILE}");
    let lines_file = format!("{folder}/{LINES_FILE}");
    let scope = path.owner().scope();
    let mut found = Vec::new();
    let fixed_name = match j.place {
        Place::Root => Some(path.owner().slug().to_string()),
        Place::Section(s) => Some(s.name().to_string()),
        Place::Client(c) => Some(c.name().to_string()),
        Place::Named { .. } => None,
    };
    match (&j.meta, &fixed_name) {
        (None, None) if j.problems.is_empty() => found.push(Problem::new(
            &meta_file,
            None,
            format!("{path} has no journal.yaml"),
        )),
        (None, _) => {}
        (Some(m), Some(fixed)) => {
            if &m.name != fixed {
                found.push(Problem::new(
                    &meta_file,
                    Some(1),
                    format!("`name` is {}, but this is {fixed}", m.name),
                ));
            }
            if m.cue.is_some() {
                found.push(Problem::new(
                    &meta_file,
                    None,
                    "a fixed journal's cue is fixed in code: drop `cue`",
                ));
            }
            if m.folded {
                found.push(Problem::new(
                    &meta_file,
                    None,
                    "a fixed journal is never folded",
                ));
            }
        }
        (Some(m), None) => {
            let own = path.last().map(Slug::as_str).unwrap_or_default();
            if m.name != own {
                found.push(Problem::new(
                    &meta_file,
                    Some(1),
                    format!("`name` is {}, but its folder is {own}", m.name),
                ));
            }
            if m.cue.is_none() {
                found.push(Problem::new(
                    &meta_file,
                    None,
                    "`cue` is missing: a named journal says when to open it",
                ));
            }
        }
    }
    if let Some(m) = &j.meta {
        if m.shared.is_some() && (scope != Scope::Personal || j.is_fixed()) {
            found.push(Problem::new(
                &meta_file,
                None,
                "only a personal named journal is shared",
            ));
        }
        if let Some(s) = &m.summary {
            let cap = summary_cap(j.place);
            if s.len() > cap {
                found.push(Problem::new(
                    &meta_file,
                    None,
                    format!(
                        "the summary is {} characters, over its cap of {cap}",
                        s.len()
                    ),
                ));
            }
            match (&s.digest, digest_cap(j.place, scope)) {
                (Some(_), None) => found.push(Problem::new(
                    &meta_file,
                    None,
                    "only a root has a voice digest",
                )),
                (Some(d), Some(cap)) if d.chars().count() > cap => found.push(Problem::new(
                    &meta_file,
                    None,
                    format!(
                        "the digest is {} characters, over its cap of {cap}",
                        d.chars().count()
                    ),
                )),
                _ => {}
            }
        }
    }
    let live = j.live_count();
    if live > MAX_CHILDREN {
        found.push(Problem::new(
            &meta_file,
            None,
            format!("{live} live sub-journals: at most {MAX_CHILDREN}; fold or merge some"),
        ));
    }
    // A repeat within one file is caught as that file is read, at its line;
    // here, one id in two of the journal's files — a window and an archive,
    // or two archives.
    let mut held: BTreeMap<&LineId, String> = BTreeMap::new();
    let files = iter::once((LINES_FILE.to_string(), &j.window)).chain(
        j.archives
            .iter()
            .map(|(p, lines)| (format!("archive/{p}.yaml"), lines)),
    );
    for (file, lines) in files {
        for line in lines {
            if let Some(first) = held.insert(&line.id, file.clone()) {
                found.push(Problem::new(
                    format!("{folder}/{file}"),
                    None,
                    format!("the id {} is in {first} too; keep it in one", line.id),
                ));
            }
        }
    }
    for (period, lines) in &j.archives {
        for line in lines {
            if !period.holds(line.unit()) || line.level() >= period.level() {
                found.push(Problem::new(
                    format!("{folder}/archive/{period}.yaml"),
                    None,
                    format!("{} is not inside {period}", line.id),
                ));
            }
        }
    }
    for line in &j.window {
        if let Some(Folds::Journal(_)) = &line.folds {
            if j.is_fixed() && matches!(j.place, Place::Root) {
                found.push(Problem::new(
                    &lines_file,
                    None,
                    format!("{}: a root's children are fixed and never fold", line.id),
                ));
            }
        }
    }
    j.problems.extend(found);
    let names: Vec<Slug> = j.children.keys().cloned().collect();
    for name in names {
        let child_path = path
            .child(name.clone())
            .expect("a loaded child is at a place");
        check_journal(
            j.children.get_mut(&name).expect("listed above"),
            &child_path,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::files::render_lines;
    use crate::testing::{note, slug};
    use crate::tree::load::load;
    use crate::tree::Tree;

    const JOHN: &str = "person: john\nname: \"John\"\naccounts:\n  - {email: \"john@example.com\"}\ngroups: [sharratt]\ndefault_group: sharratt\n";
    const YEN: &str = "person: yen\nname: \"Yen\"\naccounts:\n  - {email: \"yen@example.com\"}\ngroups: [sharratt]\ndefault_group: sharratt\n";
    const SHARRATT: &str = "group: sharratt\nname: \"Sharratt\"\nkind: family\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n";

    fn tree(extra: &[(&str, String)]) -> Tree {
        let mut files: BTreeMap<String, Vec<u8>> = [
            ("persons/john/person.yaml", JOHN),
            ("persons/yen/person.yaml", YEN),
            ("groups/sharratt/group.yaml", SHARRATT),
        ]
        .into_iter()
        .map(|(p, t)| (p.to_string(), t.as_bytes().to_vec()))
        .collect();
        for (p, t) in extra {
            files.insert(p.to_string(), t.as_bytes().to_vec());
        }
        load(&files)
    }

    fn named(name: &str, cue: &str) -> String {
        format!("name: {name}\ncue: \"{cue}\"\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n")
    }

    fn all_problems(t: &Tree) -> Vec<String> {
        let mut out: Vec<String> = t.problems.iter().map(ToString::to_string).collect();
        for p in t.persons.values() {
            out.extend(p.problems.iter().map(ToString::to_string));
        }
        for g in t.groups.values() {
            out.extend(g.problems.iter().map(ToString::to_string));
        }
        for (_, j) in t.journals() {
            out.extend(j.problems.iter().map(ToString::to_string));
        }
        out
    }

    fn one_problem(t: &Tree) -> String {
        let all = all_problems(t);
        assert_eq!(all.len(), 1, "{all:#?}");
        all.into_iter().next().unwrap()
    }

    #[test]
    fn the_group_and_its_people_agree() {
        assert_eq!(all_problems(&tree(&[])), Vec::<String>::new());
    }

    #[test]
    fn a_person_and_a_group_that_disagree_are_both_named() {
        let t = tree(&[(
            "persons/yen/person.yaml",
            YEN.replace(
                "groups: [sharratt]\ndefault_group: sharratt\n",
                "groups: []\n",
            ),
        )]);
        assert_eq!(
            one_problem(&t),
            "groups/sharratt/group.yaml: lists yen, whose person.yaml does not list sharratt"
        );
        let t = tree(&[(
            "persons/thanhlong/person.yaml",
            "person: thanhlong\nname: \"Thanh Long\"\naccounts: []\ngroups: [sharratt]\ndefault_group: sharratt\n".into(),
        )]);
        assert_eq!(
            one_problem(&t),
            "persons/thanhlong/person.yaml: lists the group sharratt, which does not list thanhlong"
        );
        let t = tree(&[("persons/yen/person.yaml", YEN.replace("sharratt", "work"))]);
        let all = all_problems(&t);
        assert!(
            all.contains(
                &"persons/yen/person.yaml: lists the group work, which there is no folder for"
                    .to_string()
            ),
            "{all:#?}"
        );
        assert!(
            all.contains(
                &"groups/sharratt/group.yaml: lists yen, whose person.yaml does not list sharratt"
                    .to_string()
            ),
            "{all:#?}"
        );
    }

    /// **A bot is listed by its group alone**: no person.yaml answers for
    /// it, and it needs no folder under persons/.
    #[test]
    fn a_bot_member_needs_no_person() {
        let t = tree(&[(
            "groups/sharratt/group.yaml",
            format!(
                "{SHARRATT}  - {{bot: c3f1a07e, role: reader, joined: 2026-10-30T00:00:00Z}}\n"
            ),
        )]);
        assert_eq!(all_problems(&t), Vec::<String>::new());
        let g = t.group_file(&slug("sharratt")).unwrap();
        assert!(g.bot(&slug("c3f1a07e")).is_some());
    }

    #[test]
    fn a_file_naming_another_owner_than_its_folder_is_a_problem() {
        let t = tree(&[(
            "persons/yen/person.yaml",
            YEN.replace("person: yen", "person: john"),
        )]);
        assert!(all_problems(&t)
            .iter()
            .any(|p| p.contains("says it is john, but it is in persons/yen/")));
        let t = tree(&[(
            "groups/sharratt/group.yaml",
            SHARRATT.replace("group: sharratt", "group: work"),
        )]);
        assert!(all_problems(&t)
            .iter()
            .any(|p| p.contains("says it is work, but it is in groups/sharratt/")));
    }

    #[test]
    fn an_account_belongs_to_one_person() {
        let t = tree(&[(
            "persons/yen/person.yaml",
            YEN.replace("yen@example.com", "john@example.com"),
        )]);
        assert_eq!(
            one_problem(&t),
            "persons/yen/person.yaml: john@example.com is listed by john too: an account is one person's"
        );
    }

    #[test]
    fn named_journals_have_a_matching_name_and_a_cue() {
        let t = tree(&[(
            "groups/sharratt/topics/permits/journal.yaml",
            named("budget", "x"),
        )]);
        assert_eq!(
            one_problem(&t),
            "groups/sharratt/topics/permits/journal.yaml:1: `name` is budget, but its folder is permits"
        );
        let t = tree(&[(
            "groups/sharratt/topics/permits/journal.yaml",
            "name: permits\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n".into(),
        )]);
        assert!(one_problem(&t).contains("`cue` is missing"));
    }

    #[test]
    fn fixed_journals_keep_their_names_and_cues() {
        let t = tree(&[(
            "groups/sharratt/topics/journal.yaml",
            "name: themes\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n".into(),
        )]);
        assert!(one_problem(&t).contains("`name` is themes, but this is topics"));
        let t = tree(&[(
            "groups/sharratt/topics/journal.yaml",
            "name: topics\ncue: \"x\"\ncreated: 2026-10-30T09:10:02Z\nfolded: true\n".into(),
        )]);
        let all = all_problems(&t);
        assert_eq!(all.len(), 2, "{all:#?}");
        let t = tree(&[(
            "groups/sharratt/journal.yaml",
            "name: sharratt\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n".into(),
        )]);
        assert_eq!(all_problems(&t), Vec::<String>::new());
    }

    #[test]
    fn caps_hold_on_summaries_and_digests() {
        let summary = |len: usize, digest: Option<usize>| {
            let mut s = String::from(
                "summary:\n  over: e3b0c44298fc1c14\n  built: 2026-10-30T09:10:02Z\n  lines: 0\n  children: 0\n",
            );
            if let Some(d) = digest {
                s.push_str(&format!("  digest: \"{}\"\n", "d".repeat(d)));
            }
            s.push_str(&format!(
                "  sentences:\n    - {{text: \"{}\", cites: []}}\n",
                "s".repeat(len)
            ));
            s
        };
        let root = |body: String| {
            format!("name: sharratt\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n{body}")
        };
        assert_eq!(
            all_problems(&tree(&[(
                "groups/sharratt/journal.yaml",
                root(summary(1500, Some(400)))
            )])),
            Vec::<String>::new()
        );
        assert!(one_problem(&tree(&[(
            "groups/sharratt/journal.yaml",
            root(summary(1501, None))
        )]))
        .contains("over its cap of 1500"));
        assert!(one_problem(&tree(&[(
            "groups/sharratt/journal.yaml",
            root(summary(10, Some(401)))
        )]))
        .contains("digest is 401"));
        let john_root = format!(
            "name: john\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n{}",
            summary(10, Some(600))
        );
        assert_eq!(
            all_problems(&tree(&[("persons/john/journal.yaml", john_root)])),
            Vec::<String>::new()
        );
        let section = format!(
            "name: topics\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n{}",
            summary(1001, None)
        );
        assert!(
            one_problem(&tree(&[("groups/sharratt/topics/journal.yaml", section)]))
                .contains("cap of 1000")
        );
        let digest_on_section = format!(
            "name: topics\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n{}",
            summary(5, Some(5))
        );
        assert!(one_problem(&tree(&[(
            "groups/sharratt/topics/journal.yaml",
            digest_on_section
        )]))
        .contains("only a root"));
        let leaf = format!("{}{}", named("permits", "x"), summary(601, None));
        assert!(one_problem(&tree(&[(
            "groups/sharratt/topics/permits/journal.yaml",
            leaf
        )]))
        .contains("cap of 600"));
    }

    #[test]
    fn an_eleventh_live_child_is_a_problem_but_folded_ones_are_not() {
        let mut extra = Vec::new();
        for i in 0..11 {
            extra.push((
                format!("groups/sharratt/topics/t{i}/journal.yaml"),
                named(&format!("t{i}"), "x"),
            ));
        }
        let pairs: Vec<(&str, String)> =
            extra.iter().map(|(p, t)| (p.as_str(), t.clone())).collect();
        assert!(one_problem(&tree(&pairs)).contains("11 live sub-journals: at most 10"));
        let mut pairs = pairs.clone();
        pairs[0].1 = pairs[0].1.replace("folded: false", "folded: true");
        assert_eq!(all_problems(&tree(&pairs)), Vec::<String>::new());
    }

    #[test]
    fn a_line_in_two_files_and_an_archive_outside_its_period_are_problems() {
        let line = note("2026-10-29T10:00:00Z", "once");
        let t = tree(&[
            (
                "persons/john/people/journal.yaml",
                "name: people\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n".into(),
            ),
            (
                "persons/john/people/lines.yaml",
                render_lines(std::slice::from_ref(&line)),
            ),
            (
                "persons/john/people/archive/2026-10-29.yaml",
                render_lines(std::slice::from_ref(&line)),
            ),
        ]);
        assert_eq!(
            one_problem(&t),
            format!("persons/john/people/archive/2026-10-29.yaml: the id {} is in lines.yaml too; keep it in one", line.id)
        );
        let t = tree(&[(
            "persons/john/people/archive/2026-10-28.yaml",
            render_lines(&[line]),
        )]);
        assert!(one_problem(&t).contains("is not inside 2026-10-28"));
    }
}
