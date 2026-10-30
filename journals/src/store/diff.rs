//! The files an applied operation changes: for each journal it touched, its
//! own files as they now render against what its folder held; for each it
//! removed, everything under its folder; for each owner, its owner file.

use std::collections::BTreeMap;

use crate::files::entry_id;
use crate::ops::Applied;
use crate::tree::load::{ARCHIVE_DIR, JOURNAL_FILE};
use crate::tree::render::{own_files, owner_files};
use crate::tree::Tree;

/// `path → Some(bytes)` to write, `path → None` to delete. Files that would
/// be written unchanged are left out.
pub type Changes = BTreeMap<String, Option<Vec<u8>>>;

/// A journal's own file inside `folder`: its `journal.yaml`, an entry, or an
/// entry of one of its archives — not a child's.
fn is_own(folder: &str, path: &str) -> bool {
    let Some(rest) = path.strip_prefix(folder).and_then(|r| r.strip_prefix('/')) else {
        return false;
    };
    match rest.split('/').collect::<Vec<_>>().as_slice() {
        [file] => *file == JOURNAL_FILE || entry_id(file).is_some(),
        [dir, _period, file] => *dir == ARCHIVE_DIR && entry_id(file).is_some(),
        _ => false,
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
        for (file, bytes) in owner_files(after, owner) {
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
    use crate::path::Address;
    use crate::period::Period;
    use crate::scope::Channel;
    use crate::testing::{note, slug};
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
        let e = "2026-10-30-0910-4b1c.yaml";
        assert!(is_own("persons/john", "persons/john/journal.yaml"));
        assert!(is_own("persons/john", &format!("persons/john/{e}")));
        assert!(is_own(
            "persons/john",
            &format!("persons/john/archive/2026-10/{e}")
        ));
        assert!(!is_own(
            "persons/john",
            "persons/john/channels/journal.yaml"
        ));
        assert!(!is_own(
            "persons/john",
            &format!("persons/john/channels/{e}")
        ));
        assert!(!is_own("persons/john", "persons/johnny/journal.yaml"));
        assert!(!is_own("persons/john", "persons/john/person.yaml"));
        assert!(!is_own("persons/john", "persons/john/README.md"));
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
        let files: Vec<&String> = ch.keys().collect();
        assert_eq!(files.len(), 1);
        assert!(
            files[0].starts_with("groups/sharratt/topics/2026-10-30-1200-"),
            "{files:?}"
        );
    }

    #[test]
    fn forgetting_the_last_line_deletes_its_file() {
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
        let file = format!("groups/sharratt/topics/{id}.yaml");
        let ch = holds(
            &t,
            &Op::Forget {
                path: p("topics"),
                which: Which::Id(id),
            },
        );
        assert_eq!(ch, Changes::from([(file, None)]));
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
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("topics/b/c"),
                text: "deep".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let ch = holds(
            &t,
            &Op::Merge {
                from: p("topics/b"),
                into: p("topics/a"),
            },
        );
        assert_eq!(ch.get("groups/sharratt/topics/b/journal.yaml"), Some(&None));
        assert_eq!(
            ch.get(&format!("groups/sharratt/topics/b/c/{id}.yaml")),
            Some(&None)
        );
        assert!(matches!(
            ch.get(&format!("groups/sharratt/topics/a/c/{id}.yaml")),
            Some(Some(_))
        ));
    }

    /// **Lines of one second hold one order** in memory and read back from
    /// their files, so a summary over them is never stale for reading them.
    #[test]
    fn lines_of_one_second_read_back_in_the_order_they_are_held() {
        let mut t = sharratt();
        for text in ["zebra", "apple", "mango"] {
            holds(
                &t,
                &Op::Note {
                    path: p("topics"),
                    text: text.into(),
                },
            );
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path: p("topics"),
                    text: text.into(),
                },
                now(),
            )
            .unwrap();
        }
        assert_eq!(load(&render_all(&t)), t);
    }

    /// **Folding, unfolding, and editing an archived line** each write
    /// exactly the files the new tree is.
    #[test]
    fn folds_and_archived_edits_write_their_files() {
        let mut t = sharratt();
        apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("permits"),
                cue: "open for permits".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        let permits = p("topics/permits");
        holds(
            &t,
            &Op::Fold {
                path: permits.clone(),
            },
        );
        let fold = Op::Fold {
            path: permits.clone(),
        };
        apply(&mut t, &john(), &fold, now()).unwrap();
        holds(&t, &Op::Unfold { path: permits });
        let line = note("2026-09-01T10:00:00Z", "an old one");
        t.get_mut(&p("topics"))
            .unwrap()
            .archives
            .insert(Period::parse("2026-09-01").unwrap(), vec![line.clone()]);
        let ch = holds(
            &t,
            &Op::Edit {
                address: Address {
                    path: p("topics"),
                    id: line.id.clone(),
                },
                text: "an old one, mended".into(),
            },
        );
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            [&format!(
                "groups/sharratt/topics/archive/2026-09-01/{}.yaml",
                line.id
            )]
        );
    }

    /// **Sharing a journal writes its copy's entry files** under the group.
    #[test]
    fn a_share_writes_the_copy() {
        let mut t = sharratt();
        apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("~/people"),
                name: slug("builder"),
                cue: "open for the builder".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("~/people/builder"),
                text: "Starts Monday.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let ch = holds(
            &t,
            &Op::Share {
                from: p("~/people/builder"),
                into: p("topics"),
                name: None,
            },
        );
        assert!(
            ch.contains_key(&format!("groups/sharratt/topics/builder/{id}.yaml")),
            "{:?}",
            ch.keys().collect::<Vec<_>>()
        );
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
        // The group's README says nothing of who is in it, so is unchanged;
        // the new person's is new.
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            [
                "groups/sharratt/group.yaml",
                "persons/yen-nhi/README.md",
                "persons/yen-nhi/person.yaml"
            ]
        );
    }

    #[test]
    fn creating_a_conversation_past_ten_folds_and_writes_three_journals() {
        let mut t = sharratt();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        for i in 0..10 {
            apply(
                &mut t,
                &server,
                &Op::Create {
                    under: p("~/channels/grok"),
                    name: slug(&format!("2026-10-30-12{i:02}")),
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
                under: p("~/channels/grok"),
                name: slug("2026-10-30-1210"),
                cue: "x".into(),
                on_taken: OnTaken::Number,
            },
            now().plus(100),
        )
        .unwrap();
        let ch = changes(&before, &after, &applied);
        let files: Vec<&str> = ch.keys().map(String::as_str).collect();
        assert_eq!(files.len(), 3, "{files:?}");
        // The oldest folds, the newest is made, and the fold is an entry.
        assert_eq!(
            files[0],
            "persons/john/channels/grok/2026-10-30-1200/journal.yaml"
        );
        assert!(
            files[1].starts_with("persons/john/channels/grok/2026-10-30-1201-"),
            "{files:?}"
        );
        assert_eq!(
            files[2],
            "persons/john/channels/grok/2026-10-30-1210/journal.yaml"
        );
    }
}
