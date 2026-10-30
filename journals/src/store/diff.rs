//! The files an applied operation changes: for each journal it touched, its
//! own files as they now render against what its folder held; for each it
//! removed, everything under its folder; for each owner, its owner file.

use std::collections::BTreeMap;

use crate::ops::Applied;
use crate::tree::load::{ARCHIVE_DIR, JOURNAL_FILE, LINES_FILE};
use crate::tree::render::{own_files, owner_file};
use crate::tree::Tree;

/// `path → Some(bytes)` to write, `path → None` to delete. Files that would
/// be written unchanged are left out.
pub type Changes = BTreeMap<String, Option<Vec<u8>>>;

/// A journal's own file inside `folder`: not a child's.
fn is_own(folder: &str, path: &str) -> bool {
    let Some(rest) = path.strip_prefix(folder).and_then(|r| r.strip_prefix('/')) else {
        return false;
    };
    match rest.split_once('/') {
        None => rest == JOURNAL_FILE || rest == LINES_FILE,
        Some((dir, file)) => dir == ARCHIVE_DIR && !file.contains('/') && file.ends_with(".yaml"),
    }
}

pub fn changes(before: &BTreeMap<String, Vec<u8>>, after: &Tree, applied: &Applied) -> Changes {
    let mut out = Changes::new();
    for path in &applied.removed {
        let prefix = format!("{}/", path.folder());
        for file in before.keys().filter(|f| f.starts_with(&prefix)) {
            out.insert(file.clone(), None);
        }
    }
    for path in &applied.touched {
        let folder = path.folder();
        for file in before.keys().filter(|f| is_own(&folder, f)) {
            out.insert(file.clone(), None);
        }
        if let Some(j) = after.get(path) {
            for (file, bytes) in own_files(path, j) {
                out.insert(file, Some(bytes));
            }
        }
    }
    for owner in &applied.owners {
        if let Some((file, bytes)) = owner_file(after, owner) {
            out.insert(file, Some(bytes));
        }
    }
    out.retain(|file, change| match (change, before.get(file)) {
        (Some(new), Some(old)) => new != old,
        (None, old) => old.is_some(),
        (Some(_), None) => true,
    });
    out
}

/// `files` with `changes` applied.
pub fn apply_changes(files: &mut BTreeMap<String, Vec<u8>>, changes: &Changes) {
    for (path, change) in changes {
        match change {
            Some(bytes) => {
                files.insert(path.clone(), bytes.clone());
            }
            None => {
                files.remove(path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::files::Role;
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, OnTaken, Op, Which};
    use crate::scope::Client;
    use crate::testing::slug;
    use crate::tree::load::load;
    use crate::tree::render::render_all;

    /// Applying the changes to the files gives exactly the files of the new
    /// tree, whatever the operation.
    fn holds(before_tree: &Tree, op: &Op) -> Changes {
        let before = render_all(before_tree);
        let mut after = before_tree.clone();
        let applied = apply(&mut after, &john(), op, now()).unwrap();
        let ch = changes(&before, &after, &applied);
        let mut files = before.clone();
        apply_changes(&mut files, &ch);
        assert_eq!(files, render_all(&after), "{op:?}");
        assert_eq!(load(&files), after);
        ch
    }

    #[test]
    fn own_files_are_the_journals_not_its_childrens() {
        assert!(is_own("persons/john", "persons/john/journal.yaml"));
        assert!(is_own("persons/john", "persons/john/archive/2026-10.yaml"));
        assert!(!is_own("persons/john", "persons/john/clients/journal.yaml"));
        assert!(!is_own("persons/john", "persons/johnny/journal.yaml"));
        assert!(!is_own("persons/john", "persons/john/person.yaml"));
        assert!(!is_own("persons/john", "persons/john/archive/x/y.yaml"));
    }

    #[test]
    fn a_note_writes_one_file() {
        let t = sharratt();
        let ch = holds(
            &t,
            &Op::Note {
                path: p("topics"),
                text: "x".into(),
            },
        );
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            ["groups/sharratt/topics/lines.yaml"]
        );
    }

    #[test]
    fn forgetting_the_last_line_deletes_lines_yaml() {
        let mut t = sharratt();
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("topics"),
                text: "x".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let ch = holds(
            &t,
            &Op::Forget {
                path: p("topics"),
                which: Which::Id(id),
            },
        );
        assert_eq!(
            ch,
            Changes::from([("groups/sharratt/topics/lines.yaml".to_string(), None)])
        );
    }

    #[test]
    fn a_merge_deletes_the_old_folder_and_writes_the_moved_one() {
        let mut t = sharratt();
        for (under, name) in [("topics", "a"), ("topics", "b"), ("topics/b", "c")] {
            apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: p(under),
                    name: slug(name),
                    cue: "x".into(),
                    on_taken: OnTaken::Refuse,
                },
                now(),
            )
            .unwrap();
        }
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("topics/b/c"),
                text: "deep".into(),
            },
            now(),
        )
        .unwrap();
        let ch = holds(
            &t,
            &Op::Merge {
                from: p("topics/b"),
                into: p("topics/a"),
            },
        );
        assert_eq!(ch.get("groups/sharratt/topics/b/journal.yaml"), Some(&None));
        assert_eq!(ch.get("groups/sharratt/topics/b/c/lines.yaml"), Some(&None));
        assert!(matches!(
            ch.get("groups/sharratt/topics/a/c/lines.yaml"),
            Some(Some(_))
        ));
    }

    #[test]
    fn adding_a_member_writes_both_owner_files() {
        let t = sharratt();
        let op = Op::AddMember {
            group: slug("sharratt"),
            person: slug("yen-nhi"),
            name: Some("Yen Nhi".into()),
            emails: vec![],
            role: Role::Member,
        };
        let ch = holds(&t, &op);
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            ["groups/sharratt/group.yaml", "persons/yen-nhi/person.yaml"]
        );
    }

    #[test]
    fn creating_a_conversation_past_ten_folds_and_writes_three_journals() {
        let mut t = sharratt();
        let server = Actor::Server {
            person: slug("john"),
            client: Client::Grok,
        };
        for i in 0..10 {
            apply(
                &mut t,
                &server,
                &Op::Create {
                    under: p("~/clients/grok"),
                    name: slug(&format!("c{i}")),
                    cue: "x".into(),
                    on_taken: OnTaken::Number,
                },
                now().plus(i),
            )
            .unwrap();
        }
        let before = render_all(&t);
        let mut after = t.clone();
        let applied = apply(
            &mut after,
            &server,
            &Op::Create {
                under: p("~/clients/grok"),
                name: slug("c10"),
                cue: "x".into(),
                on_taken: OnTaken::Number,
            },
            now().plus(100),
        )
        .unwrap();
        let ch = changes(&before, &after, &applied);
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            [
                "persons/john/clients/grok/c0/journal.yaml",
                "persons/john/clients/grok/c10/journal.yaml",
                "persons/john/clients/grok/lines.yaml",
            ]
        );
    }
}
