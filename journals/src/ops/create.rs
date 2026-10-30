//! A new named journal, and what an eleventh live child does (§2.3).

use crate::error::{Error, Result};
use crate::files::Meta;
use crate::ops::fold::fold_child;
use crate::ops::{Applied, OnTaken};
use crate::path::{JournalPath, Place};
use crate::scope::{is_dated, Eleventh, Naming};
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
        Eleventh::Fixed => None,
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
            Eleventh::FoldIdlestClosed => format!(
                "{under} has {MAX_CHILDREN} sub-journals, all still open; close or fold one first"
            ),
            _ => format!("{under} has {MAX_CHILDREN} sub-journals; fold or merge one first"),
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
        .expect("below a root is a section")
        .eleventh(under.names().len() - 1);
    if rule == Eleventh::Fixed {
        return Err(Error::refused(format!(
            "{under}'s journals are fixed: one per channel"
        )));
    }
    let parent = tree.need_mut(under)?;
    parent.writable(&under.show())?;
    let taken = parent.children.keys().cloned().collect();
    let name = match on_taken {
        OnTaken::Number => name.numbered(&taken),
        OnTaken::Refuse if taken.contains(name) => {
            return Err(Error::refused(format!(
                "{under}/{name} exists already{}",
                if parent.children[name].is_folded() {
                    " (folded: unfold it)"
                } else {
                    ""
                }
            )))
        }
        OnTaken::Refuse => name.clone(),
    };
    let path = under.child(name.clone())?;
    if let Place::Named { section, depth } = path.place()? {
        if section.naming(depth) == Naming::Dated && !is_dated(name.as_str()) {
            return Err(Error::invalid(format!(
                "{under} holds conversations, each named for when it began: \
                 `{}`, or `{}-<what it was about>`",
                now.dated_minute(),
                now.dated_minute()
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
    use crate::ops::fixture::{john, now, p, sharratt};
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

    #[test]
    fn a_journal_is_created_with_its_cue() {
        let mut t = sharratt();
        let a = apply(
            &mut t,
            &john(),
            &create_op("topics", "house-build", OnTaken::Refuse),
            now(),
        )
        .unwrap();
        let path = a.created.unwrap();
        assert_eq!(path.show(), "@sharratt/topics/house-build");
        assert_eq!(a.subject, "Create @sharratt/topics/house-build");
        assert_eq!(a.touched, vec![path.clone()]);
        let j = t.get(&path).unwrap();
        assert_eq!(j.cue().as_deref(), Some("open for house-build"));
        assert_eq!(j.meta.as_ref().unwrap().created, now());
        let nested = apply(
            &mut t,
            &john(),
            &create_op("topics/house-build", "permits", OnTaken::Refuse),
            now(),
        )
        .unwrap();
        assert_eq!(
            nested.created.unwrap().show(),
            "@sharratt/topics/house-build/permits"
        );
    }

    #[test]
    fn a_name_taken_is_refused_or_numbered() {
        let mut t = sharratt();
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
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let drive = "2026-10-30-1200-drive";
        let first = apply(
            &mut t,
            &server,
            &create_op("~/channels/grok", drive, OnTaken::Number),
            now(),
        )
        .unwrap();
        let second = apply(
            &mut t,
            &server,
            &create_op("~/channels/grok", drive, OnTaken::Number),
            now(),
        )
        .unwrap();
        assert_eq!(
            first.created.unwrap().show(),
            format!("~/channels/grok/{drive}")
        );
        assert_eq!(
            second.created.unwrap().show(),
            format!("~/channels/grok/{drive}.2")
        );
        // A conversation is named for when it began.
        assert_eq!(
            apply(
                &mut t,
                &server,
                &create_op("~/channels/grok", "drive", OnTaken::Number),
                now()
            )
            .unwrap_err()
            .to_string(),
            "~/channels/grok holds conversations, each named for when it began: \
             `2026-10-30-1200`, or `2026-10-30-1200-<what it was about>`"
        );
    }

    #[test]
    fn roots_and_the_channels_section_are_fixed() {
        let mut t = sharratt();
        assert!(create(
            &mut t,
            &p("@sharratt"),
            &slug("x"),
            "x",
            OnTaken::Refuse,
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("sections are fixed"));
        assert!(create(
            &mut t,
            &p("~/channels"),
            &slug("tesla"),
            "x",
            OnTaken::Refuse,
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("one per channel"));
    }

    #[test]
    fn cues_are_one_short_line() {
        let mut t = sharratt();
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

    #[test]
    fn an_eleventh_topic_is_refused() {
        let mut t = sharratt();
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
            apply(
                &mut t,
                &john(),
                &create_op("topics", "t10", OnTaken::Refuse),
                now()
            )
            .unwrap_err()
            .to_string(),
            "@sharratt/topics has 10 sub-journals; fold or merge one first"
        );
    }

    #[test]
    fn an_eleventh_conversation_folds_the_oldest() {
        let mut t = sharratt();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        for i in 0..10 {
            apply(
                &mut t,
                &server,
                &create_op(
                    "~/channels/grok",
                    &format!("2026-10-30-12{i:02}"),
                    OnTaken::Number,
                ),
                now().plus(i),
            )
            .unwrap();
        }
        let a = apply(
            &mut t,
            &server,
            &create_op("~/channels/grok", "2026-10-30-1210", OnTaken::Number),
            now().plus(100),
        )
        .unwrap();
        assert_eq!(
            a.folded.iter().map(|p| p.show()).collect::<Vec<_>>(),
            ["~/channels/grok/2026-10-30-1200"]
        );
        let grok = t.get(&p("~/channels/grok")).unwrap();
        assert_eq!(grok.live_count(), 10);
        assert!(grok.children[&slug("2026-10-30-1200")].is_folded());
        assert_eq!(grok.window.len(), 1, "the fold line");
    }

    #[test]
    fn an_eleventh_thread_folds_the_idlest_closed_one_and_never_an_open_one() {
        let mut t = sharratt();
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
            .unwrap()
        };
        for i in 0..10 {
            mk(&mut t, &format!("th{i}"), now().plus(i));
        }
        assert!(mk_err(&mut t).contains("all still open"));
        // Close two: the one closed earlier is the idler.
        apply(
            &mut t,
            &Actor::Zend,
            &Op::Close {
                path: p("threads/th7"),
            },
            now().plus(500),
        )
        .unwrap();
        apply(
            &mut t,
            &Actor::Zend,
            &Op::Close {
                path: p("threads/th3"),
            },
            now().plus(400),
        )
        .unwrap();
        let a = mk(&mut t, "th10", now().plus(1000));
        assert_eq!(a.folded[0].show(), "@sharratt/threads/th3");
    }

    fn mk_err(t: &mut Tree) -> String {
        apply(
            t,
            &Actor::Zend,
            &Op::Create {
                under: p("threads"),
                name: slug("over"),
                cue: "x".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap_err()
        .to_string()
    }

    #[test]
    fn an_eleventh_bot_folds_the_idlest() {
        let mut t = sharratt();
        let npcd = Actor::Npcd {
            person: slug("john"),
        };
        for i in 0..10 {
            apply(
                &mut t,
                &npcd,
                &Op::Create {
                    under: p("bots"),
                    name: slug(&format!("b{i}")),
                    cue: "x".into(),
                    on_taken: OnTaken::Number,
                },
                now().plus(i),
            )
            .unwrap();
        }
        // b0 has news; b1 is now the idlest.
        apply(
            &mut t,
            &npcd,
            &Op::Episode {
                path: p("bots/b0"),
                text: "went to market".into(),
            },
            now().plus(50),
        )
        .unwrap();
        let a = apply(
            &mut t,
            &npcd,
            &Op::Create {
                under: p("bots"),
                name: slug("b10"),
                cue: "x".into(),
                on_taken: OnTaken::Number,
            },
            now().plus(60),
        )
        .unwrap();
        assert_eq!(a.folded[0].show(), "@sharratt/bots/b1");
    }

    #[test]
    fn archive_is_no_name_for_a_journal() {
        let mut t = sharratt();
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
