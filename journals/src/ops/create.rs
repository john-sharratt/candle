//! A new named journal, and what an eleventh live child does (§2.3).

use std::collections::BTreeSet;

use crate::error::{Error, Result};
use crate::files::Meta;
use crate::ops::fold::fold_child;
use crate::ops::{Applied, OnTaken};
use crate::path::{JournalPath, Place};
use crate::scope::{conversation_channel, Eleventh, Naming};
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{Journal, Tree, MAX_CHILDREN};

/// The longest a cue may be, in characters.
pub const MAX_CUE: usize = 200;

pub(crate) fn clean_cue(cue: &str) -> Result<String> {
    let cue = cue.trim();
    if cue.is_empty() {
        return Err(Error::invalid("a journal needs a cue: when to open it"));
    }
    if cue.contains('\n') {
        return Err(Error::invalid("a cue is one line"));
    }
    if cue.chars().count() > MAX_CUE {
        return Err(Error::invalid(format!(
            "a cue is at most {MAX_CUE} characters"
        )));
    }
    Ok(cue.to_string())
}

/// The child of `parent` that an eleventh folds, by the section's rule; an
/// error when the rule refuses.
fn to_fold(parent: &Journal, under: &JournalPath, rule: Eleventh) -> Result<Slug> {
    let live = || parent.live().map(|(n, j)| (n.clone(), j));
    let pick = match rule {
        Eleventh::FoldOldest => live()
            .min_by_key(|(n, j)| (j.meta.as_ref().map(|m| m.created), n.clone()))
            .map(|(n, _)| n),
        Eleventh::FoldIdlestClosed => live()
            .filter(|(_, j)| j.is_closed())
            .min_by_key(|(n, j)| (j.last_active(), n.clone()))
            .map(|(n, _)| n),
        Eleventh::FoldIdlest => live()
            .min_by_key(|(n, j)| (j.last_active(), n.clone()))
            .map(|(n, _)| n),
        Eleventh::Refuse => None,
    };
    pick.ok_or_else(|| {
        Error::refused(match rule {
            Eleventh::FoldIdlestClosed => {
                format!("{under} has {MAX_CHILDREN} sub-journals, all still open; close one first")
            }
            _ => format!(
                "{under} has {MAX_CHILDREN} sub-journals; restructuring it is a coding task on \
                 the journals repository"
            ),
        })
    })
}

pub fn create(
    tree: &mut Tree,
    under: &JournalPath,
    name: &Slug,
    cue: &str,
    on_taken: OnTaken,
    now: Stamp,
) -> Result<Applied> {
    let cue = clean_cue(cue)?;
    if under.place()? == Place::Root {
        return Err(Error::refused(format!(
            "{under}'s sections are fixed: create inside one"
        )));
    }
    let rule = under
        .section()
        .expect("below a master is a section")
        .eleventh(under.names().len() - 1);
    let parent = tree.need_mut(under)?;
    parent.writable(&under.show())?;
    // A reference's name is taken too: a path through it must stay one
    // journal's (§2.5).
    let referenced = |n: &Slug| {
        parent
            .meta
            .iter()
            .flat_map(|m| &m.refs)
            .find(|r| &r.name == n)
            .map(|r| r.to.clone())
    };
    let mut taken: BTreeSet<Slug> = parent.children.keys().cloned().collect();
    taken.extend(
        parent
            .meta
            .iter()
            .flat_map(|m| &m.refs)
            .map(|r| r.name.clone()),
    );
    let name = match on_taken {
        OnTaken::Number => name.numbered(&taken),
        OnTaken::Refuse if !parent.children.contains_key(name) && taken.contains(name) => {
            let to = referenced(name).expect("a taken name that is no child is a reference's");
            return Err(Error::refused(format!(
                "{under}/{name} is a reference already, to {to}: name the journal otherwise"
            )));
        }
        OnTaken::Refuse if taken.contains(name) => {
            return Err(Error::refused(format!(
                "{under}/{name} exists already{}",
                if parent.children[name].is_folded() {
                    " (folded: unfolding it is a restructuring, a coding task)"
                } else {
                    ""
                }
            )))
        }
        OnTaken::Refuse => name.clone(),
    };
    let path = under.child(name.clone())?;
    if let Place::Named { section, depth } = path.place()? {
        if section.naming(depth) == Naming::Dated && conversation_channel(name.as_str()).is_none() {
            let minute = now.dated_minute();
            return Err(Error::invalid(format!(
                "{under} holds conversations, each named for the minute it began and its channel: \
                 `{minute}-<channel>`, or `{minute}-<channel>-<what it was about>`"
            )));
        }
    }
    let mut applied = Applied {
        created: Some(path.clone()),
        ..Applied::touching(format!("Create {path}"), vec![])
    };
    if parent.live_count() >= MAX_CHILDREN {
        let victim = to_fold(parent, under, rule)?;
        let folded = fold_child(parent, under, &victim, now)?;
        applied.touched.push(under.clone());
        applied.touched.push(folded.clone());
        applied.folded.push(folded);
    }
    let journal = Journal::new(
        path.place()?,
        Some(Meta::new(name.as_str(), Some(cue), now)),
    );
    parent.children.insert(name, journal);
    applied.touched.push(path);
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, Op};
    use crate::scope::Channel;
    use crate::testing::slug;

    fn create_op(under: &str, name: &str, on_taken: OnTaken) -> Op {
        Op::Create {
            under: p(under),
            name: slug(name),
            cue: format!("open for {name}"),
            on_taken,
        }
    }

    fn server() -> Actor {
        Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        }
    }

    #[test]
    fn a_journal_is_created_with_its_cue() {
        let mut t = zen();
        let a = apply(
            &mut t,
            &john(),
            &create_op("topics", "house-extension", OnTaken::Refuse),
            now(),
        )
        .unwrap();
        let path = a.created.unwrap();
        assert_eq!(path.show(), "@zen/topics/house-extension");
        assert_eq!(a.subject, "Create @zen/topics/house-extension");
        assert_eq!(a.touched, vec![path.clone()]);
        let j = t.get(&path).unwrap();
        assert_eq!(j.cue().as_deref(), Some("open for house-extension"));
        assert_eq!(j.meta.as_ref().unwrap().created, now());
        let nested = apply(
            &mut t,
            &john(),
            &create_op("topics/house-extension", "permits", OnTaken::Refuse),
            now(),
        )
        .unwrap();
        assert_eq!(
            nested.created.unwrap().show(),
            "@zen/topics/house-extension/permits"
        );
    }

    #[test]
    fn a_name_taken_is_refused_or_numbered() {
        let mut t = zen();
        apply(
            &mut t,
            &john(),
            &create_op("topics", "trip", OnTaken::Refuse),
            now(),
        )
        .unwrap();
        assert!(apply(
            &mut t,
            &john(),
            &create_op("topics", "trip", OnTaken::Refuse),
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("exists already"));
        let drive = "2026-10-30-1200-grok-drive";
        let first = apply(
            &mut t,
            &server(),
            &create_op("conversations", drive, OnTaken::Number),
            now(),
        )
        .unwrap();
        let second = apply(
            &mut t,
            &server(),
            &create_op("conversations", drive, OnTaken::Number),
            now(),
        )
        .unwrap();
        assert_eq!(
            first.created.unwrap().show(),
            format!("@zen/conversations/{drive}")
        );
        assert_eq!(
            second.created.unwrap().show(),
            format!("@zen/conversations/{drive}.2")
        );
        // A conversation is named for when it began and its channel.
        for bad in ["drive", "2026-10-30-1200"] {
            assert_eq!(
                apply(
                    &mut t,
                    &server(),
                    &create_op("conversations", bad, OnTaken::Number),
                    now()
                )
                .unwrap_err()
                .to_string(),
                "@zen/conversations holds conversations, each named for the minute it began and \
                 its channel: `2026-10-30-1200-<channel>`, or \
                 `2026-10-30-1200-<channel>-<what it was about>`",
                "{bad}"
            );
        }
    }

    /// **A reference's name is taken**: a sub-journal never shadows it, so
    /// the tree still loads.
    #[test]
    fn a_references_name_is_never_a_new_journals() {
        let mut t = zen();
        for name in ["candle", "wasmer"] {
            apply(
                &mut t,
                &john(),
                &create_op("topics", name, OnTaken::Refuse),
                now(),
            )
            .unwrap();
        }
        apply(
            &mut t,
            &john(),
            &Op::Link {
                under: p("topics/candle"),
                name: slug("wasmer"),
                to: p("topics/wasmer"),
                cue: "open for the WebAssembly work".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(
            apply(
                &mut t,
                &john(),
                &create_op("topics/candle", "wasmer", OnTaken::Refuse),
                now()
            )
            .unwrap_err()
            .to_string(),
            "@zen/topics/candle/wasmer is a reference already, to topics/wasmer: name the \
             journal otherwise"
        );
    }

    #[test]
    fn a_master_is_fixed() {
        let mut t = zen();
        assert!(
            create(&mut t, &p("@zen"), &slug("x"), "x", OnTaken::Refuse, now())
                .unwrap_err()
                .to_string()
                .contains("sections are fixed")
        );
    }

    #[test]
    fn cues_are_one_short_line() {
        let mut t = zen();
        for (cue, says) in [
            ("", "needs a cue"),
            ("a\nb", "one line"),
            (&"c".repeat(201) as &str, "at most 200"),
        ] {
            let op = Op::Create {
                under: p("topics"),
                name: slug("x"),
                cue: cue.into(),
                on_taken: OnTaken::Refuse,
            };
            assert!(
                apply(&mut t, &john(), &op, now())
                    .unwrap_err()
                    .to_string()
                    .contains(says),
                "{cue:?}"
            );
        }
    }

    /// **An eleventh topic is a restructuring**: the tool refuses and names
    /// the coding task.
    #[test]
    fn an_eleventh_topic_is_refused_and_names_the_coding_task() {
        let mut t = zen();
        for i in 0..10 {
            apply(
                &mut t,
                &john(),
                &create_op("topics", &format!("t{i}"), OnTaken::Refuse),
                now(),
            )
            .unwrap();
        }
        assert_eq!(
            apply(&mut t, &john(), &create_op("topics", "t10", OnTaken::Refuse), now())
                .unwrap_err()
                .to_string(),
            "@zen/topics has 10 sub-journals; restructuring it is a coding task on the journals repository"
        );
    }

    #[test]
    fn an_eleventh_conversation_folds_the_oldest() {
        let mut t = zen();
        for i in 0..10 {
            apply(
                &mut t,
                &server(),
                &create_op(
                    "conversations",
                    &format!("2026-10-30-12{i:02}-grok"),
                    OnTaken::Number,
                ),
                now().plus(i),
            )
            .unwrap();
        }
        let a = apply(
            &mut t,
            &server(),
            &create_op("conversations", "2026-10-30-1210-grok", OnTaken::Number),
            now().plus(100),
        )
        .unwrap();
        assert_eq!(
            a.folded.iter().map(|p| p.show()).collect::<Vec<_>>(),
            ["@zen/conversations/2026-10-30-1200-grok"]
        );
        let conversations = t.get(&p("conversations")).unwrap();
        assert_eq!(conversations.live_count(), 10);
        assert!(conversations.children[&slug("2026-10-30-1200-grok")].is_folded());
        assert_eq!(conversations.window.len(), 1, "the fold line");
    }

    #[test]
    fn an_eleventh_thread_folds_the_idlest_closed_one_and_never_an_open_one() {
        let mut t = zen();
        let mk = |t: &mut Tree, name: &str, at: Stamp| {
            apply(
                t,
                &Actor::Zend,
                &Op::Create {
                    under: p("threads"),
                    name: slug(name),
                    cue: "x".into(),
                    on_taken: OnTaken::Number,
                },
                at,
            )
        };
        for i in 0..10 {
            mk(&mut t, &format!("th{i}"), now().plus(i)).unwrap();
        }
        assert!(mk(&mut t, "over", now())
            .unwrap_err()
            .to_string()
            .contains("all still open"));
        // Close two: the one closed earlier is the idler.
        for (th, at) in [("threads/th7", 500), ("threads/th3", 400)] {
            apply(
                &mut t,
                &Actor::Zend,
                &Op::Close { path: p(th) },
                now().plus(at),
            )
            .unwrap();
        }
        let a = mk(&mut t, "th10", now().plus(1000)).unwrap();
        assert_eq!(a.folded[0].show(), "@zen/threads/th3");
    }

    #[test]
    fn an_eleventh_bot_folds_the_idlest() {
        let mut t = zen();
        let npcd = Actor::Npcd {
            person: slug("john"),
        };
        let mk = |t: &mut Tree, name: &str, at: Stamp| {
            apply(
                t,
                &npcd,
                &Op::Create {
                    under: p("bots"),
                    name: slug(name),
                    cue: "x".into(),
                    on_taken: OnTaken::Number,
                },
                at,
            )
            .unwrap()
        };
        for i in 0..10 {
            mk(&mut t, &format!("b{i}"), now().plus(i));
        }
        // b0 has news; b1 is now the idlest.
        apply(
            &mut t,
            &npcd,
            &Op::Episode {
                path: p("bots/b0"),
                text: "went to market".into(),
                speaker: None,
            },
            now().plus(50),
        )
        .unwrap();
        let a = mk(&mut t, "b10", now().plus(60));
        assert_eq!(a.folded[0].show(), "@zen/bots/b1");
    }

    #[test]
    fn archive_is_no_name_for_a_journal() {
        let mut t = zen();
        assert!(apply(
            &mut t,
            &john(),
            &create_op("topics", "archive", OnTaken::Refuse),
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("reserved"));
    }
}
