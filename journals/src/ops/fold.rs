//! Folding a journal out of its parent's live index (§2.3): the service's
//! upkeep when a conversation, thread or bot is the eleventh. No tool folds
//! or unfolds a journal; a restructuring does (§2.6).
//!
//! A folded journal keeps everything: its folder stays, and it is read by
//! path and grepped. Folding writes its summary — or its cue, when it has
//! none yet — into its parent as a `fold` line that names it.

use crate::error::{Error, Result};
use crate::files::in_order;
use crate::line::{Folds, Kind, Line, Writer, MAX_TEXT};
use crate::ops::mint;
use crate::path::JournalPath;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::Journal;

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
        account: None,
        speaker: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::testing::slug;

    #[test]
    fn folding_writes_a_fold_line_naming_the_child() {
        let mut t = zen();
        apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("permits"),
                cue: "open for the permits".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        let topics = p("topics");
        let parent = t.get_mut(&topics).unwrap();
        let folded = fold_child(parent, &topics, &slug("permits"), now()).unwrap();
        assert_eq!(folded.show(), "@zen/topics/permits");
        assert_eq!(parent.live_count(), 0);
        let line = &parent.window[0];
        assert_eq!(line.kind, Kind::Fold);
        assert_eq!(line.folds, Some(Folds::Journal(slug("permits"))));
        assert_eq!(line.text, "permits: open for the permits");
        assert!(fold_child(parent, &topics, &slug("permits"), now())
            .unwrap_err()
            .to_string()
            .contains("folded already"));
        let root = p("@zen");
        let master = t.get_mut(&root).unwrap();
        assert!(fold_child(master, &root, &slug("topics"), now())
            .unwrap_err()
            .to_string()
            .contains("is fixed"));
    }
}
