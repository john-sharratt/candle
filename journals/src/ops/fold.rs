//! Folding a journal out of its parent's live index, and back (§2.3).
//!
//! A folded journal keeps everything: its folder stays, it is read by path,
//! grepped, and unfolded. Folding writes its summary — or its cue, when it
//! has none yet — into its parent as a `fold` line that names it.

use crate::error::{Error, Result};
use crate::files::in_order;
use crate::line::{Folds, Kind, Line, Writer, MAX_TEXT};
use crate::ops::{mint, Applied};
use crate::path::JournalPath;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{Journal, Tree, MAX_CHILDREN};

/// Fold `parent`'s child `name`: its fold line into the parent, and the
/// child marked folded. Both must be writable.
pub(crate) fn fold_child(
    parent: &mut Journal,
    parent_path: &JournalPath,
    name: &Slug,
    now: Stamp,
) -> Result<JournalPath> {
    let child_path = parent_path.child(name.clone())?;
    parent.writable(&parent_path.show())?;
    let child = parent
        .children
        .get(name)
        .ok_or_else(|| Error::not_found(format!("there is no journal {child_path}")))?;
    child.writable(&child_path.show())?;
    if child.is_fixed() {
        return Err(Error::refused(format!(
            "{child_path} is fixed: it never folds"
        )));
    }
    if child.is_folded() {
        return Err(Error::refused(format!("{child_path} is folded already")));
    }
    let said = child
        .summary()
        .map(|s| s.text())
        .filter(|t| !t.is_empty())
        .or_else(|| child.cue())
        .unwrap_or_else(|| format!("{name}, with nothing summarised yet"));
    let text: String = format!("{name}: {said}").chars().take(MAX_TEXT).collect();
    let line = Line {
        id: mint(parent, now, Writer::Zend, &text),
        at: now,
        kind: Kind::Fold,
        by: Writer::Zend,
        person: None,
        channel: None,
        folds: Some(Folds::Journal(name.clone())),
        proposed: false,
        text,
    };
    parent.window.push(line);
    in_order(&mut parent.window);
    parent
        .children
        .get_mut(name)
        .and_then(|c| c.meta.as_mut())
        .expect("a named child has metadata")
        .folded = true;
    Ok(child_path)
}

pub fn fold(tree: &mut Tree, path: &JournalPath, now: Stamp) -> Result<Applied> {
    let (parent_path, name) = split(path)?;
    let parent = tree.need_mut(&parent_path)?;
    let folded = fold_child(parent, &parent_path, &name, now)?;
    Ok(Applied {
        folded: vec![folded],
        ..Applied::touching(format!("Fold {path}"), vec![parent_path, path.clone()])
    })
}

pub fn unfold(tree: &mut Tree, path: &JournalPath) -> Result<Applied> {
    let (parent_path, name) = split(path)?;
    let parent = tree.need_mut(&parent_path)?;
    parent.writable(&parent_path.show())?;
    let child = parent
        .children
        .get(&name)
        .ok_or_else(|| Error::not_found(format!("there is no journal {path}")))?;
    child.writable(&path.show())?;
    if !child.is_folded() {
        return Err(Error::refused(format!("{path} is not folded")));
    }
    if parent.live_count() >= MAX_CHILDREN {
        return Err(Error::refused(format!(
            "{parent_path} has {MAX_CHILDREN} live sub-journals; fold or merge one first"
        )));
    }
    parent
        .window
        .retain(|l| l.folds.as_ref() != Some(&Folds::Journal(name.clone())));
    parent
        .children
        .get_mut(&name)
        .and_then(|c| c.meta.as_mut())
        .expect("a folded child has metadata")
        .folded = false;
    Ok(Applied::touching(
        format!("Unfold {path}"),
        vec![parent_path, path.clone()],
    ))
}

fn split(path: &JournalPath) -> Result<(JournalPath, Slug)> {
    match (path.parent(), path.last()) {
        (Some(parent), Some(name)) => Ok((parent, name.clone())),
        _ => Err(Error::refused(format!("{path} is a root: it never folds"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, OnTaken, Op};
    use crate::testing::slug;

    fn with_topic(name: &str) -> (Tree, JournalPath) {
        let mut t = sharratt();
        let a = apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug(name),
                cue: format!("open for the {name}"),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        (t, a.created.unwrap())
    }

    #[test]
    fn folding_writes_a_fold_line_and_unfolding_takes_it_back() {
        let (mut t, path) = with_topic("permits");
        let topics = path.parent().unwrap();
        let a = apply(&mut t, &john(), &Op::Fold { path: path.clone() }, now()).unwrap();
        assert_eq!(a.touched, vec![topics.clone(), path.clone()]);
        let parent = t.get(&topics).unwrap();
        assert_eq!(parent.live_count(), 0);
        let line = &parent.window[0];
        assert_eq!(line.kind, Kind::Fold);
        assert_eq!(line.folds, Some(Folds::Journal(slug("permits"))));
        assert_eq!(line.text, "permits: open for the permits");
        assert!(
            apply(&mut t, &john(), &Op::Fold { path: path.clone() }, now())
                .unwrap_err()
                .to_string()
                .contains("folded already")
        );
        apply(&mut t, &john(), &Op::Unfold { path: path.clone() }, now()).unwrap();
        let parent = t.get(&topics).unwrap();
        assert!(parent.window.is_empty());
        assert_eq!(parent.live_count(), 1);
        assert!(apply(&mut t, &john(), &Op::Unfold { path }, now())
            .unwrap_err()
            .to_string()
            .contains("is not folded"));
    }

    #[test]
    fn fixed_journals_and_roots_never_fold() {
        let mut t = sharratt();
        assert!(
            apply(&mut t, &john(), &Op::Fold { path: p("topics") }, now())
                .unwrap_err()
                .to_string()
                .contains("is fixed")
        );
        assert!(fold(&mut t, &p("@sharratt"), now())
            .unwrap_err()
            .to_string()
            .contains("is a root"));
    }

    #[test]
    fn unfolding_into_a_full_parent_is_refused() {
        let (mut t, first) = with_topic("t0");
        apply(
            &mut t,
            &john(),
            &Op::Fold {
                path: first.clone(),
            },
            now(),
        )
        .unwrap();
        for i in 1..=10 {
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
        assert!(apply(&mut t, &john(), &Op::Unfold { path: first }, now())
            .unwrap_err()
            .to_string()
            .contains("10 live sub-journals"));
    }
}
