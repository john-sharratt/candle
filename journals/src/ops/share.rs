//! Sharing a personal journal into a family (§3.6): a copy under a shared
//! section, with its lines and everything below it; the personal one is
//! marked as shared and folded. Nothing moves the other way.

use crate::error::{Error, Result};
use crate::ops::create::create;
use crate::ops::fold::fold_child;
use crate::ops::place::{re_place, subtree_paths};
use crate::ops::{Applied, OnTaken};
use crate::path::{JournalPath, Place};
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{Journal, Tree};

/// A `shared:` mark says where a personal journal went; copied into a family,
/// a journal is no personal one and carries none (§3.3).
fn forget_shares(j: &mut Journal) {
    if let Some(m) = j.meta.as_mut() {
        m.shared = None;
    }
    for child in j.children.values_mut() {
        forget_shares(child);
    }
}

pub fn share(
    tree: &mut Tree,
    from: &JournalPath,
    into: &JournalPath,
    name: Option<&Slug>,
    now: Stamp,
) -> Result<Applied> {
    if !matches!(from.place()?, Place::Named { .. }) {
        return Err(Error::refused(format!(
            "{from} is fixed: share one of the journals in it"
        )));
    }
    let source = tree.need(from)?.clone();
    for (names, j) in source.walk() {
        if !j.problems.is_empty() {
            let mut all = from.names().to_vec();
            all.extend(names);
            j.writable(&JournalPath::new(from.owner().clone(), all)?.show())?;
        }
    }
    if let Some(to) = source.meta.as_ref().and_then(|m| m.shared.as_ref()) {
        return Err(Error::refused(format!("{from} is shared already, as {to}")));
    }
    let name = name
        .cloned()
        .unwrap_or_else(|| from.last().expect("named").clone());
    let cue = source.cue().expect("a named journal has a cue");
    let mut applied = create(tree, into, &name, &cue, OnTaken::Refuse, now)?;
    let to = applied.created.clone().expect("create makes a journal");
    // The copy keeps the lines and everything below; its own journal.yaml is
    // the new one `create` wrote, so it starts open, unfolded and dirty.
    let mut copy = source;
    re_place(&mut copy, &to)?;
    for child in copy.children.values_mut() {
        forget_shares(child);
    }
    let target = tree.need_mut(&to)?;
    target.window = copy.window;
    target.archives = copy.archives;
    target.children = copy.children;
    applied.touched.retain(|p| p != &to);
    applied.touched.extend(subtree_paths(&to, tree.need(&to)?));
    // The personal journal: marked, and folded out of its parent's index.
    let parent_path = from.parent().expect("named");
    let parent = tree.need_mut(&parent_path)?;
    let mine = parent
        .children
        .get_mut(from.last().expect("named"))
        .expect("read above");
    mine.meta.as_mut().expect("named").shared = Some(to.show());
    if !mine.is_folded() {
        fold_child(parent, &parent_path, from.last().expect("named"), now)?;
        applied.folded.push(from.clone());
        applied.touched.push(parent_path);
    }
    applied.touched.push(from.clone());
    applied.subject = format!("Share {from} as {to}");
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, Op};
    use crate::scope::Client;
    use crate::testing::slug;
    use crate::tree::load::load;
    use crate::tree::render::render_all;

    #[test]
    fn a_shared_journal_is_copied_whole_and_the_original_folded() {
        let mut t = sharratt();
        let server = Actor::Server {
            person: slug("john"),
            client: Client::Grok,
        };
        let conv = apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("~/clients/grok"),
                name: slug("2026-10-30-2105"),
                cue: "a drive to the coast".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let id = apply(
            &mut t,
            &server,
            &Op::Episode {
                path: conv.clone(),
                text: "Planned the Da Nang trip.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let a = apply(
            &mut t,
            &john(),
            &Op::Share {
                from: conv.clone(),
                into: p("topics"),
                name: Some(slug("trip-plans")),
            },
            now(),
        )
        .unwrap();
        assert_eq!(
            a.subject,
            "Share ~/clients/grok/2026-10-30-2105 as @sharratt/topics/trip-plans"
        );
        let shared = t.get(&p("topics/trip-plans")).unwrap();
        assert_eq!(shared.window[0].id, id, "lines keep their ids");
        assert_eq!(shared.cue().as_deref(), Some("a drive to the coast"));
        let mine = t.get(&conv).unwrap();
        assert!(mine.is_folded());
        assert_eq!(
            mine.meta.as_ref().unwrap().shared.as_deref(),
            Some("@sharratt/topics/trip-plans")
        );
        assert_eq!(mine.window[0].id, id, "the original keeps its lines");
        assert!(a.touched.iter().any(|p| p == &conv));
        assert!(apply(
            &mut t,
            &john(),
            &Op::Share {
                from: conv,
                into: p("topics"),
                name: None
            },
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("shared already"));
    }

    #[test]
    fn a_child_shared_before_is_copied_without_its_mark() {
        let mut t = sharratt();
        let create = |under: &str, name: &str| Op::Create {
            under: p(under),
            name: slug(name),
            cue: format!("open for {name}"),
            on_taken: OnTaken::Refuse,
        };
        apply(&mut t, &john(), &create("~/people", "builder"), now()).unwrap();
        apply(
            &mut t,
            &john(),
            &create("~/people/builder", "permits"),
            now(),
        )
        .unwrap();
        apply(
            &mut t,
            &john(),
            &Op::Share {
                from: p("~/people/builder/permits"),
                into: p("topics"),
                name: None,
            },
            now(),
        )
        .unwrap();
        apply(
            &mut t,
            &john(),
            &Op::Share {
                from: p("~/people/builder"),
                into: p("projects"),
                name: None,
            },
            now(),
        )
        .unwrap();
        let copied = t.get(&p("projects/builder/permits")).unwrap();
        assert_eq!(copied.meta.as_ref().unwrap().shared, None);
        let back = load(&render_all(&t));
        assert!(back
            .get(&p("projects/builder/permits"))
            .unwrap()
            .problems
            .is_empty());
    }

    #[test]
    fn a_share_into_a_full_section_is_refused() {
        let mut t = sharratt();
        for i in 0..10 {
            apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: p("topics"),
                    name: slug(&format!("t{i}")),
                    cue: "x".into(),
                    on_taken: OnTaken::Refuse,
                },
                now(),
            )
            .unwrap();
        }
        let person = apply(
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
        .unwrap()
        .created
        .unwrap();
        let before = t.clone();
        assert!(apply(
            &mut t,
            &john(),
            &Op::Share {
                from: person,
                into: p("topics"),
                name: None
            },
            now()
        )
        .is_err());
        assert_eq!(t, before);
    }
}
