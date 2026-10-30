//! The files an applied operation changes: for each journal it touched, its
//! own files as they now render against what its folder held; for each
//! owner file it changed, that file.

use std::collections::BTreeMap;

use crate::files::entry_id;
use crate::ops::{Applied, OwnerFile};
use crate::tree::load::{ARCHIVE_DIR, JOURNAL_FILE};
use crate::tree::render::{group_files, own_files, person_files};
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
        let files = match owner {
            OwnerFile::Person(p) => person_files(after, p),
            OwnerFile::Group(g) => group_files(after, g),
        };
        for (file, bytes) in files {
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
    use crate::ops::fixture::{john, now, p, zen};
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

    fn note_op(path: &str, text: &str) -> Op {
        Op::Note {
            path: p(path),
            text: text.into(),
            speaker: None,
        }
    }

    #[test]
    fn own_files_are_the_journals_not_its_childrens() {
        let e = "2026-10-30-0910-4b1c.yaml";
        let topics = "groups/zen/topics";
        assert!(is_own(topics, "groups/zen/topics/journal.yaml"));
        assert!(is_own(topics, &format!("groups/zen/topics/{e}")));
        assert!(is_own(
            topics,
            &format!("groups/zen/topics/archive/2026-10/{e}")
        ));
        assert!(!is_own(topics, "groups/zen/topics/candle/journal.yaml"));
        assert!(!is_own(topics, &format!("groups/zen/topics/candle/{e}")));
        assert!(!is_own(topics, "groups/zen/topicsx/journal.yaml"));
        assert!(!is_own("groups/zen", "groups/zen/group.yaml"));
        assert!(!is_own("groups/zen", "groups/zen/README.md"));
        assert!(!is_own(topics, "groups/zen/topics/archive/x/y.yaml"));
    }

    #[test]
    fn a_note_writes_one_file() {
        let ch = holds(&zen(), &note_op("topics", "x"));
        let files: Vec<&String> = ch.keys().collect();
        assert_eq!(files.len(), 1);
        assert!(
            files[0].starts_with("groups/zen/topics/2026-10-30-1200-"),
            "{files:?}"
        );
    }

    #[test]
    fn forgetting_the_last_line_deletes_its_file() {
        let mut t = zen();
        let id = apply(&mut t, &john(), &note_op("topics", "x"), now())
            .unwrap()
            .line
            .unwrap();
        let file = format!("groups/zen/topics/{id}.yaml");
        let ch = holds(
            &t,
            &Op::Forget {
                path: p("topics"),
                which: Which::Id(id),
            },
        );
        assert_eq!(ch, Changes::from([(file, None)]));
    }

    /// **A reference and a focus are written into their journal's
    /// `journal.yaml`**, and nothing else.
    #[test]
    fn a_link_and_a_focus_write_one_journal_yaml() {
        let mut t = zen();
        apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("candle"),
                cue: "open for the engine".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        let ch = holds(
            &t,
            &Op::Link {
                under: p("topics/candle"),
                name: slug("people"),
                to: p("people"),
                cue: "open for who works on it".into(),
            },
        );
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            ["groups/zen/topics/candle/journal.yaml"]
        );
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let conv = apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("conversations"),
                name: slug("2026-10-30-1200-grok"),
                cue: "open for a drive".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let ch = holds(
            &t,
            &Op::Focus {
                path: conv,
                speaker: None,
                about: Some(vec![p("topics/candle")]),
            },
        );
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            ["groups/zen/conversations/2026-10-30-1200-grok/journal.yaml"]
        );
    }

    /// **Lines of one second hold one order** in memory and read back from
    /// their files, so a summary over them is never stale for reading them.
    #[test]
    fn lines_of_one_second_read_back_in_the_order_they_are_held() {
        let mut t = zen();
        for text in ["zebra", "apple", "mango"] {
            holds(&t, &note_op("topics", text));
            apply(&mut t, &john(), &note_op("topics", text), now()).unwrap();
        }
        assert_eq!(load(&render_all(&t)), t);
    }

    /// **Editing an archived line** writes exactly its own file.
    #[test]
    fn an_archived_edit_writes_its_file() {
        let mut t = zen();
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
                "groups/zen/topics/archive/2026-09-01/{}.yaml",
                line.id
            )]
        );
    }

    #[test]
    fn adding_a_member_writes_both_owner_files_and_their_journal() {
        let t = zen();
        let op = Op::AddMember {
            group: slug("zen"),
            person: slug("yen-nhi"),
            name: Some("Yen Nhi".into()),
            emails: vec![],
            role: Role::Member,
        };
        let ch = holds(&t, &op);
        // The group's README says nothing of who is in it, so is unchanged.
        assert_eq!(
            ch.keys().collect::<Vec<_>>(),
            [
                "groups/zen/group.yaml",
                "groups/zen/people/yen-nhi/journal.yaml",
                "persons/yen-nhi/person.yaml"
            ]
        );
    }

    #[test]
    fn creating_a_conversation_past_ten_folds_and_writes_three_journals() {
        let mut t = zen();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let create = |name: String| Op::Create {
            under: p("conversations"),
            name: slug(&name),
            cue: "x".into(),
            on_taken: OnTaken::Number,
        };
        for i in 0..10 {
            apply(
                &mut t,
                &server,
                &create(format!("2026-10-30-12{i:02}-grok")),
                now().plus(i),
            )
            .unwrap();
        }
        let before = render_all(&t);
        let mut after = t.clone();
        let applied = apply(
            &mut after,
            &server,
            &create("2026-10-30-1210-grok".into()),
            now().plus(100),
        )
        .unwrap();
        let ch = changes(&before, &after, &applied);
        let files: Vec<&str> = ch.keys().map(String::as_str).collect();
        assert_eq!(files.len(), 3, "{files:?}");
        // The oldest folds, the fold is an entry, and the newest is made.
        assert_eq!(
            files[0],
            "groups/zen/conversations/2026-10-30-1200-grok/journal.yaml"
        );
        assert!(
            files[1].starts_with("groups/zen/conversations/2026-10-30-1201-"),
            "{files:?}"
        );
        assert_eq!(
            files[2],
            "groups/zen/conversations/2026-10-30-1210-grok/journal.yaml"
        );
    }
}
