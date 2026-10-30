//! Merging one journal into another (§2.3): its lines join the other's
//! window and archives by time, its children move under the other, and it
//! is gone. Ids stay, so every citation of a line still resolves — at its
//! new journal.

use crate::error::{Error, Result};
use crate::files::in_order;
use crate::ops::place::{re_place, subtree_paths};
use crate::ops::Applied;
use crate::path::{JournalPath, Place};
use crate::tree::{Journal, Tree, MAX_CHILDREN};

fn named(path: &JournalPath) -> Result<()> {
    match path.place()? {
        Place::Named { .. } => Ok(()),
        _ => Err(Error::refused(format!("{path} is fixed: it never merges"))),
    }
}

fn writable_below(j: &Journal, path: &JournalPath) -> Result<()> {
    for (names, k) in j.walk() {
        let mut all = path.names().to_vec();
        all.extend(names);
        let p = JournalPath::new(path.owner().clone(), all)?;
        k.writable(&p.show())?;
    }
    Ok(())
}

pub fn merge(tree: &mut Tree, from: &JournalPath, into: &JournalPath) -> Result<Applied> {
    named(from)?;
    if from == into {
        return Err(Error::refused(format!("{from} cannot merge into itself")));
    }
    if from.owner() != into.owner() {
        return Err(Error::refused(format!(
            "{from} and {into} are in different folders; share it first"
        )));
    }
    if from.holds(into) {
        return Err(Error::refused(format!("{into} is inside {from}")));
    }
    writable_below(tree.need(from)?, from)?;
    tree.need(into)?.writable(&into.show())?;
    let parent_path = from.parent().expect("a named journal has a parent");
    let name = from.last().expect("a named journal has a name").clone();
    let gone = tree
        .need_mut(&parent_path)?
        .children
        .remove(&name)
        .expect("checked above");
    let target = tree.need_mut(into)?;
    if let Some(clash) = gone
        .children
        .keys()
        .find(|n| target.children.contains_key(*n))
    {
        return Err(Error::refused(format!(
            "{into} has a {clash} already; rename or merge one of them first"
        )));
    }
    let live = target.live_count() + gone.live_count();
    if live > MAX_CHILDREN {
        return Err(Error::refused(format!(
            "{into} would have {live} live sub-journals; at most {MAX_CHILDREN}"
        )));
    }
    if let Some(dup) = gone.all_lines().find(|l| target.has_id(&l.id)) {
        return Err(Error::conflict(format!(
            "{from}#{} is in {into} already; forget one first",
            dup.id
        )));
    }
    let Journal {
        window,
        archives,
        children,
        ..
    } = gone;
    target.window.extend(window);
    in_order(&mut target.window);
    for (period, lines) in archives {
        let held = target.archives.entry(period).or_default();
        held.extend(lines);
        in_order(held);
    }
    let mut touched = vec![into.clone()];
    for (child_name, mut child) in children {
        let path = into.child(child_name.clone())?;
        re_place(&mut child, &path)?;
        touched.extend(subtree_paths(&path, &child));
        target.children.insert(child_name, child);
    }
    Ok(Applied {
        removed: vec![from.clone()],
        ..Applied::touching(format!("Merge {from} into {into}"), touched)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Section;
    use crate::testing::slug;

    fn mk(t: &mut Tree, under: &str, name: &str) -> JournalPath {
        apply(
            t,
            &john(),
            &Op::Create {
                under: p(under),
                name: slug(name),
                cue: format!("open for {name}"),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap()
    }

    fn note(t: &mut Tree, path: &JournalPath, text: &str, secs: i64) {
        apply(
            t,
            &john(),
            &Op::Note {
                path: path.clone(),
                text: text.into(),
            },
            now().plus(secs),
        )
        .unwrap();
    }

    #[test]
    fn lines_interleave_by_time_and_children_move_with_their_files() {
        let mut t = sharratt();
        let a = mk(&mut t, "topics", "permits");
        let b = mk(&mut t, "topics", "council");
        let inner = mk(&mut t, "topics/council", "fees");
        note(&mut t, &a, "a1", 10);
        note(&mut t, &b, "b1", 5);
        note(&mut t, &a, "a2", 20);
        note(&mut t, &inner, "deep", 30);
        let got = apply(
            &mut t,
            &john(),
            &Op::Merge {
                from: b.clone(),
                into: a.clone(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(
            got.subject,
            "Merge @sharratt/topics/council into @sharratt/topics/permits"
        );
        assert_eq!(got.removed, vec![b.clone()]);
        assert_eq!(
            got.touched.iter().map(|p| p.show()).collect::<Vec<_>>(),
            ["@sharratt/topics/permits", "@sharratt/topics/permits/fees"]
        );
        assert!(t.get(&b).is_none());
        let merged = t.get(&a).unwrap();
        assert_eq!(
            merged
                .window
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>(),
            ["b1", "a1", "a2"]
        );
        let moved = t.get(&p("topics/permits/fees")).unwrap();
        assert_eq!(moved.window[0].text, "deep");
        assert_eq!(
            moved.place,
            Place::Named {
                section: Section::Topics,
                depth: 2
            }
        );
    }

    #[test]
    fn a_child_merges_into_its_parent() {
        let mut t = sharratt();
        let parent = mk(&mut t, "topics", "trip");
        let child = mk(&mut t, "topics/trip", "flights");
        note(&mut t, &child, "booked", 1);
        apply(
            &mut t,
            &john(),
            &Op::Merge {
                from: child.clone(),
                into: parent.clone(),
            },
            now(),
        )
        .unwrap();
        let j = t.get(&parent).unwrap();
        assert!(j.children.is_empty());
        assert_eq!(j.window[0].text, "booked");
    }

    #[test]
    fn what_cannot_merge_is_refused_and_nothing_changes() {
        let mut t = sharratt();
        let a = mk(&mut t, "topics", "a");
        let b = mk(&mut t, "topics", "b");
        let a_inner = mk(&mut t, "topics/a", "x");
        let b_inner = mk(&mut t, "topics/b", "x");
        let before = t.clone();
        for (from, into, says) in [
            (a.clone(), a.clone(), "into itself"),
            (a.clone(), a_inner.clone(), "is inside"),
            (p("topics"), a.clone(), "is fixed"),
            (b.clone(), a.clone(), "has a x already"),
            (a.clone(), p("~/people"), "different folders"),
        ] {
            let _ = &b_inner;
            let err = merge(&mut t.clone(), &from, &into).unwrap_err().to_string();
            assert!(err.contains(says), "{from} → {into}: {err}");
        }
        assert!(apply(&mut t, &john(), &Op::Merge { from: b, into: a }, now()).is_err());
        assert_eq!(t, before);
    }

    #[test]
    fn a_merge_past_ten_live_children_is_refused() {
        let mut t = sharratt();
        let a = mk(&mut t, "topics", "a");
        let b = mk(&mut t, "topics", "b");
        for i in 0..6 {
            mk(&mut t, "topics/a", &format!("a{i}"));
            mk(&mut t, "topics/b", &format!("b{i}"));
        }
        assert!(
            apply(&mut t, &john(), &Op::Merge { from: b, into: a }, now())
                .unwrap_err()
                .to_string()
                .contains("12 live sub-journals")
        );
    }
}
