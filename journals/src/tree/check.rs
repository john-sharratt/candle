//! The rules a loaded tree is held to (§2, §3.3, §5.1), each broken one a
//! problem on the journal or owner that breaks it.

use std::collections::BTreeMap;
use std::iter;

use crate::error::Problem;
use crate::files::{entry_file, AccountKey, Email, Who};
use crate::line::{Folds, LineId};
use crate::path::{JournalPath, Place};
use crate::scope::{conversation_channel, Naming, Section};
use crate::slug::Slug;
use crate::tree::load::{ARCHIVE_DIR, JOURNAL_FILE};
use crate::tree::{Journal, Tree, MAX_CHILDREN};

/// The longest a journal's summary may be, in characters (§5.1).
pub fn summary_cap(place: Place) -> usize {
    match place {
        Place::Root => 1_500,
        Place::Section(_) => 1_000,
        // A conversation's summary is read aloud as "last time" in a voice
        // frame (§6.4), so it stays short; a topic or a person holds many
        // subjects, and its summary names each.
        Place::Named {
            section: Section::Conversations,
            ..
        } => 600,
        Place::Named { .. } => 1_200,
    }
}

/// The longest its voice digest may be; `None` where it has none: only the
/// master and a person's own journal, directly under `people`, have one.
pub fn digest_cap(place: Place) -> Option<usize> {
    match place {
        Place::Root
        | Place::Named {
            section: Section::People,
            depth: 1,
        } => Some(400),
        _ => None,
    }
}

pub fn check(tree: &mut Tree) {
    check_owners(tree);
    check_accounts(tree);
    let groups: Vec<Slug> = tree.groups.keys().cloned().collect();
    for group in &groups {
        let root = tree.root_mut(group).expect("listed above");
        check_journal(root, &JournalPath::root(group.clone()));
    }
    check_links(tree);
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
            let why = match &m.who {
                Who::Person(person) if !tree.persons.contains_key(person) => {
                    format!("lists {person}, who has no folder under persons/")
                }
                Who::Person(person) if !lists_group(tree, person, slug) => {
                    format!("lists {person}, whose person.yaml does not list {slug}")
                }
                Who::Bot(bot) => match m.reads.iter().find(|r| r.at(slug).is_err()) {
                    Some(r) => format!("grants bot {bot} `{r}`, which is no place in a group"),
                    None => continue,
                },
                Who::Person(_) => continue,
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
    let mut found = Vec::new();
    if let (Place::Named { section, depth }, Some(name)) = (j.place, path.last()) {
        if section.naming(depth) == Naming::Dated && conversation_channel(name.as_str()).is_none() {
            found.push(Problem::new(
                &meta_file,
                None,
                format!(
                    "`{name}` is a conversation's folder, named for the minute it began and the \
                     channel it began on: yyyy-mm-dd-hhmm-<channel>, and what it was about if it says"
                ),
            ));
        }
    }
    let fixed_name = match j.place {
        Place::Root => Some(path.group().to_string()),
        Place::Section(s) => Some(s.name().to_string()),
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
        if m.focus.is_some() && !j.place.is_conversation() {
            found.push(Problem::new(
                &meta_file,
                None,
                "only a conversation has a focus",
            ));
        }
        let topic = matches!(
            j.place,
            Place::Named {
                section: Section::Topics,
                ..
            }
        );
        if m.repo.is_some() && !topic {
            found.push(Problem::new(
                &meta_file,
                None,
                "only a topic names a repository",
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
            match (&s.digest, digest_cap(j.place)) {
                (Some(_), None) => found.push(Problem::new(
                    &meta_file,
                    None,
                    "only the master and a person's own journal have a voice digest",
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
            format!(
                "{live} live sub-journals: at most {MAX_CHILDREN}; restructuring it is a coding \
                 change to the journals repository"
            ),
        ));
    }
    // One id in two of the journal's places — its folder and an archive, or
    // two archives — each a file of the same name.
    let mut held: BTreeMap<&LineId, String> = BTreeMap::new();
    let places = iter::once((String::new(), &j.window)).chain(
        j.archives
            .iter()
            .map(|(p, lines)| (format!("{ARCHIVE_DIR}/{p}/"), lines)),
    );
    for (dir, lines) in places {
        for line in lines {
            if let Some(first) = held.insert(&line.id, dir.clone()) {
                let first = if first.is_empty() {
                    "its folder"
                } else {
                    &first
                };
                found.push(Problem::new(
                    format!("{folder}/{dir}{}", entry_file(&line.id)),
                    None,
                    format!("the entry {} is in {first} too; keep it in one", line.id),
                ));
            }
        }
    }
    for (period, lines) in &j.archives {
        for line in lines {
            if !period.holds(line.unit()) || line.level() >= period.level() {
                found.push(Problem::new(
                    format!("{folder}/{ARCHIVE_DIR}/{period}/{}", entry_file(&line.id)),
                    None,
                    format!("{} is not inside {period}", line.id),
                ));
            }
        }
    }
    for line in &j.window {
        if let Some(Folds::Journal(_)) = &line.folds {
            if matches!(j.place, Place::Root) {
                found.push(Problem::new(
                    format!("{folder}/{}", entry_file(&line.id)),
                    None,
                    format!("{}: a master's children are fixed and never fold", line.id),
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

/// What a journal's references and focus name, against the tree: a
/// reference leads to a journal that is there, in its group, and not
/// through another reference (§2.5); a focus names a person's journal and
/// journals that are there (§2.4). A reference never shadows a child.
fn check_links(tree: &mut Tree) {
    let mut found: Vec<(JournalPath, Problem)> = Vec::new();
    for (path, j) in tree.journals() {
        let Some(m) = &j.meta else { continue };
        let meta_file = format!("{}/{JOURNAL_FILE}", path.folder());
        let group = path.group();
        for r in &m.refs {
            let why = if j.children.contains_key(&r.name) {
                Some(format!(
                    "the reference {} has the name of a sub-journal of {path}",
                    r.name
                ))
            } else {
                match r.to.at(group) {
                    Err(e) => Some(format!("the reference {} leads nowhere: {e}", r.name)),
                    Ok(to) if tree.get(&to).is_none() => Some(format!(
                        "the reference {} leads to {to}, which is not there: a reference \
                         leads to a journal, never through another reference",
                        r.name
                    )),
                    Ok(to) if to == path => {
                        Some(format!("the reference {} leads to {path} itself", r.name))
                    }
                    Ok(_) => None,
                }
            };
            if let Some(why) = why {
                found.push((path.clone(), Problem::new(&meta_file, None, why)));
            }
        }
        if let Some(f) = &m.focus {
            if let Some(s) = &f.speaker {
                if !JournalPath::person(group, s).is_ok_and(|p| tree.get(&p).is_some()) {
                    found.push((
                        path.clone(),
                        Problem::new(
                            &meta_file,
                            None,
                            format!("the focus's speaker {s} has no journal under people/"),
                        ),
                    ));
                }
            }
            for a in &f.about {
                if !a.at(group).is_ok_and(|p| tree.get(&p).is_some()) {
                    found.push((
                        path.clone(),
                        Problem::new(
                            &meta_file,
                            None,
                            format!("the focus is about {a}, which is not there"),
                        ),
                    ));
                }
            }
        }
    }
    for (path, p) in found {
        tree.get_mut(&path).expect("listed above").problems.push(p);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::files::entry_file;
    use crate::testing::{note, slug};
    use crate::tree::load::load;
    use crate::tree::Tree;

    const JOHN: &str = "person: john\nname: \"John\"\naccounts:\n  - {email: \"john@example.com\"}\ngroups: [zen]\ndefault_group: zen\n";
    const YEN: &str = "person: yen\nname: \"Yen\"\naccounts:\n  - {email: \"yen@example.com\"}\ngroups: [zen]\ndefault_group: zen\n";
    const ZEN: &str = "group: zen\nname: \"Zen\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n";

    fn tree(extra: &[(&str, String)]) -> Tree {
        let mut files: BTreeMap<String, Vec<u8>> = [
            ("persons/john/person.yaml", JOHN),
            ("persons/yen/person.yaml", YEN),
            ("groups/zen/group.yaml", ZEN),
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
            YEN.replace("groups: [zen]\ndefault_group: zen\n", "groups: []\n"),
        )]);
        assert_eq!(
            one_problem(&t),
            "groups/zen/group.yaml: lists yen, whose person.yaml does not list zen"
        );
        let t = tree(&[(
            "persons/thanhlong/person.yaml",
            "person: thanhlong\nname: \"Thanh Long\"\naccounts: []\ngroups: [zen]\ndefault_group: zen\n".into(),
        )]);
        assert_eq!(
            one_problem(&t),
            "persons/thanhlong/person.yaml: lists the group zen, which does not list thanhlong"
        );
        let t = tree(&[("persons/yen/person.yaml", YEN.replace("zen", "work"))]);
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
                &"groups/zen/group.yaml: lists yen, whose person.yaml does not list zen"
                    .to_string()
            ),
            "{all:#?}"
        );
    }

    /// **A bot is listed by its group alone**, with the subtrees it reads;
    /// a grant that is no place in a group is a problem.
    #[test]
    fn a_bot_member_needs_no_person_and_its_grants_are_places() {
        let with_bot = |reads: &str| {
            tree(&[(
                "groups/zen/group.yaml",
                format!(
                    "{ZEN}  - {{bot: c3f1a07e, role: reader, joined: 2026-10-30T00:00:00Z, reads: [{reads}]}}\n"
                ),
            )])
        };
        let t = with_bot("topics/canon");
        assert_eq!(all_problems(&t), Vec::<String>::new());
        assert!(t
            .group_file(&slug("zen"))
            .unwrap()
            .bot(&slug("c3f1a07e"))
            .is_some());
        assert_eq!(
            one_problem(&with_bot("channels/grok")),
            "groups/zen/group.yaml: grants bot c3f1a07e `channels/grok`, which is no place in a group"
        );
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
            "groups/zen/group.yaml",
            ZEN.replace("group: zen", "group: work"),
        )]);
        assert!(all_problems(&t)
            .iter()
            .any(|p| p.contains("says it is work, but it is in groups/zen/")));
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
            "groups/zen/topics/permits/journal.yaml",
            named("budget", "x"),
        )]);
        assert_eq!(
            one_problem(&t),
            "groups/zen/topics/permits/journal.yaml:1: `name` is budget, but its folder is permits"
        );
        let t = tree(&[(
            "groups/zen/topics/permits/journal.yaml",
            "name: permits\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n".into(),
        )]);
        assert!(one_problem(&t).contains("`cue` is missing"));
    }

    #[test]
    fn fixed_journals_keep_their_names_and_cues() {
        let t = tree(&[(
            "groups/zen/topics/journal.yaml",
            "name: themes\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n".into(),
        )]);
        assert!(one_problem(&t).contains("`name` is themes, but this is topics"));
        let t = tree(&[(
            "groups/zen/topics/journal.yaml",
            "name: topics\ncue: \"x\"\ncreated: 2026-10-30T09:10:02Z\nfolded: true\n".into(),
        )]);
        let all = all_problems(&t);
        assert_eq!(all.len(), 2, "{all:#?}");
        let t = tree(&[(
            "groups/zen/journal.yaml",
            "name: zen\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n".into(),
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
        let master = |body: String| {
            format!("name: zen\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n{body}")
        };
        assert_eq!(
            all_problems(&tree(&[(
                "groups/zen/journal.yaml",
                master(summary(1500, Some(400)))
            )])),
            Vec::<String>::new()
        );
        assert!(one_problem(&tree(&[(
            "groups/zen/journal.yaml",
            master(summary(1501, None))
        )]))
        .contains("over its cap of 1500"));
        assert!(one_problem(&tree(&[(
            "groups/zen/journal.yaml",
            master(summary(10, Some(401)))
        )]))
        .contains("digest is 401"));
        let section = format!(
            "name: topics\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n{}",
            summary(1001, None)
        );
        assert!(
            one_problem(&tree(&[("groups/zen/topics/journal.yaml", section)]))
                .contains("cap of 1000")
        );
        let digest_on_section = format!(
            "name: topics\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n{}",
            summary(5, Some(5))
        );
        assert!(one_problem(&tree(&[(
            "groups/zen/topics/journal.yaml",
            digest_on_section
        )]))
        .contains("only the master and a person's own journal"));
        // A person's own journal has a digest; a topic has none.
        let yen = format!(
            "{}{}",
            named("yen", "open for Yen"),
            summary(1200, Some(400))
        );
        assert_eq!(
            all_problems(&tree(&[("groups/zen/people/yen/journal.yaml", yen)])),
            Vec::<String>::new()
        );
        let leaf = |len, digest| format!("{}{}", named("permits", "x"), summary(len, digest));
        assert_eq!(
            all_problems(&tree(&[(
                "groups/zen/topics/permits/journal.yaml",
                leaf(1200, None)
            )])),
            Vec::<String>::new()
        );
        assert!(one_problem(&tree(&[(
            "groups/zen/topics/permits/journal.yaml",
            leaf(1201, None)
        )]))
        .contains("cap of 1200"));
        assert!(one_problem(&tree(&[(
            "groups/zen/topics/permits/journal.yaml",
            leaf(10, Some(10))
        )]))
        .contains("only the master and a person's own journal"));
        // A conversation is read aloud, so it keeps the short cap.
        let conversation = format!(
            "{}{}",
            named("2026-10-30-0910-claude-code", "x"),
            summary(601, None)
        );
        assert!(one_problem(&tree(&[(
            "groups/zen/conversations/2026-10-30-0910-claude-code/journal.yaml",
            conversation
        )]))
        .contains("cap of 600"));
    }

    #[test]
    fn an_eleventh_live_child_is_a_problem_but_folded_ones_are_not() {
        let mut extra = Vec::new();
        for i in 0..11 {
            extra.push((
                format!("groups/zen/topics/t{i}/journal.yaml"),
                named(&format!("t{i}"), "x"),
            ));
        }
        let pairs: Vec<(&str, String)> =
            extra.iter().map(|(p, t)| (p.as_str(), t.clone())).collect();
        assert!(one_problem(&tree(&pairs)).contains(
            "11 live sub-journals: at most 10; restructuring it is a coding change to the journals repository"
        ));
        let mut pairs = pairs.clone();
        pairs[0].1 = pairs[0].1.replace("folded: false", "folded: true");
        assert_eq!(all_problems(&tree(&pairs)), Vec::<String>::new());
    }

    #[test]
    fn a_line_in_two_places_and_an_archive_outside_its_period_are_problems() {
        let line = note("2026-10-29T10:00:00Z", "once");
        let file = entry_file(&line.id);
        let t = tree(&[
            (format!("groups/zen/people/{file}").as_str(), line.render()),
            (
                format!("groups/zen/people/archive/2026-10-29/{file}").as_str(),
                line.render(),
            ),
        ]);
        assert_eq!(
            one_problem(&t),
            format!(
                "groups/zen/people/archive/2026-10-29/{file}: the entry {} is in its folder too; keep it in one",
                line.id
            )
        );
        let t = tree(&[(
            format!("groups/zen/people/archive/2026-10-28/{file}").as_str(),
            line.render(),
        )]);
        assert!(one_problem(&t).contains("is not inside 2026-10-28"));
    }

    /// **A conversation is named for when it began and its channel**; a
    /// person or a topic for what it is.
    #[test]
    fn a_conversation_not_named_for_its_start_and_channel_is_a_problem() {
        let conv = |name: &str| {
            (
                format!("groups/zen/conversations/{name}/journal.yaml"),
                named(name, "open for a chat"),
            )
        };
        for good in [
            "2026-10-30-2105-grok",
            "2026-10-30-2105-claude-code-permits",
            "2026-10-30-2105-other.2",
        ] {
            let (p, t) = conv(good);
            assert_eq!(
                all_problems(&tree(&[(p.as_str(), t)])),
                Vec::<String>::new(),
                "{good}"
            );
        }
        for bad in ["permits", "2026-10-30-2105"] {
            let (p, t) = conv(bad);
            assert_eq!(
                one_problem(&tree(&[(p.as_str(), t)])),
                format!(
                    "groups/zen/conversations/{bad}/journal.yaml: `{bad}` is a conversation's folder, \
                     named for the minute it began and the channel it began on: \
                     yyyy-mm-dd-hhmm-<channel>, and what it was about if it says"
                )
            );
        }
        let t = tree(&[(
            "groups/zen/people/yen/journal.yaml",
            named("yen", "open for Yen"),
        )]);
        assert_eq!(all_problems(&t), Vec::<String>::new());
    }

    /// **A reference leads to a journal that is there**, never through
    /// another, never by a sub-journal's name; **a focus names what is
    /// there**, and only a conversation has one.
    #[test]
    fn references_and_focus_name_what_is_there() {
        let candle = |refs: &str| {
            (
                "groups/zen/topics/candle/journal.yaml",
                format!(
                    "{}repo: candle\nrefs:\n{refs}",
                    named("candle", "open for the engine")
                ),
            )
        };
        let wasmer = (
            "groups/zen/topics/wasmer/journal.yaml",
            named("wasmer", "open for Wasmer"),
        );
        let link = |name: &str, to: &str| format!("  - {{name: {name}, to: {to}, cue: c}}\n");
        let (p, t) = candle(&link("wasmer", "topics/wasmer"));
        assert_eq!(
            all_problems(&tree(&[(p, t), (wasmer.0, wasmer.1.clone())])),
            Vec::<String>::new()
        );
        let (p, t) = candle(&link("wasmer", "topics/wasmer"));
        assert_eq!(
            one_problem(&tree(&[(p, t)])),
            "groups/zen/topics/candle/journal.yaml: the reference wasmer leads to @zen/topics/wasmer, \
             which is not there: a reference leads to a journal, never through another reference"
        );
        let (p, t) = candle(&link("self", "topics/candle"));
        assert!(one_problem(&tree(&[(p, t)])).contains("leads to @zen/topics/candle itself"));
        let (p, t) = candle(&link("kv", "channels/grok"));
        assert!(one_problem(&tree(&[(p, t)])).contains("the reference kv leads nowhere"));
        let (p, t) = candle(&link("cluster", "topics/wasmer"));
        assert!(one_problem(&tree(&[
            (p, t),
            (wasmer.0, wasmer.1.clone()),
            (
                "groups/zen/topics/candle/cluster/journal.yaml",
                named("cluster", "x")
            ),
        ]))
        .contains("has the name of a sub-journal"));
        let conv = "groups/zen/conversations/2026-10-31-0712-grok/journal.yaml";
        let focused = |focus: &str| {
            format!(
                "{}focus:\n{focus}",
                named("2026-10-31-0712-grok", "a drive")
            )
        };
        assert_eq!(
            all_problems(&tree(&[
                (conv, focused("  speaker: yen\n  about: [topics/wasmer]\n")),
                ("groups/zen/people/yen/journal.yaml", named("yen", "Yen")),
                (wasmer.0, wasmer.1.clone()),
            ])),
            Vec::<String>::new()
        );
        let all = all_problems(&tree(&[(
            conv,
            focused("  speaker: yen\n  about: [topics/x]\n"),
        )]));
        assert_eq!(
            all,
            [
                format!("{conv}: the focus's speaker yen has no journal under people/"),
                format!("{conv}: the focus is about topics/x, which is not there"),
            ]
        );
        assert!(all_problems(&tree(&[
            (
                "groups/zen/topics/wasmer/journal.yaml",
                format!("{}focus:\n  speaker: yen\n", named("wasmer", "x"))
            ),
            ("groups/zen/people/yen/journal.yaml", named("yen", "Yen")),
        ]))
        .contains(
            &"groups/zen/topics/wasmer/journal.yaml: only a conversation has a focus".to_string()
        ));
        assert!(one_problem(&tree(&[(
            "groups/zen/people/yen/journal.yaml",
            format!("{}repo: candle\n", named("yen", "x"))
        )]))
        .contains("only a topic names a repository"));
    }
}
