//! Every journal's files written again as the store now writes them: one
//! commit that brings each entry read in an older form to the current one.
//! An entry once named the app a person spoke through (`by: claude-code`,
//! `channel: grok`); it is read as the person's, and written without it.
//! A journal whose files are broken is left as it is, as every write leaves
//! one (§3.3).

use crate::error::Result;
use crate::ops::Applied;
use crate::tree::Tree;

/// What a restamp touches: every journal of `tree` that can be written. It
/// changes nothing in the tree — the files it commits are the tree rendered.
pub fn restamp(tree: &Tree) -> Result<Applied> {
    let touched = tree
        .journals()
        .into_iter()
        .filter(|(_, j)| j.problems.is_empty())
        .map(|(path, _)| path)
        .collect();
    Ok(Applied::touching(
        "Restamp: every entry written as the store writes it",
        touched,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::actor::Actor;
    use crate::ops::fixture::{john, now, zen};
    use crate::ops::{apply, Op};
    use crate::store::diff::changes;
    use crate::tree::load::load;
    use crate::tree::render::render_all;

    const OLD_NOTE: &str = "groups/zen/topics/2026-10-30-0912-7e3a.yaml";
    const OLD_EPISODE: &str = "groups/zen/people/2026-10-30-0913-1b2c.yaml";

    /// The fixture's files, with a note and an episode written in the form
    /// that named the app.
    fn files_with_old_entries() -> BTreeMap<String, Vec<u8>> {
        let mut files = render_all(&zen());
        files.insert(
            OLD_NOTE.into(),
            b"at: 2026-10-30T09:12:40Z\nkind: note\nby: claude-code\naccount: john\ntext: x\n"
                .to_vec(),
        );
        files.insert(
            OLD_EPISODE.into(),
            b"at: 2026-10-30T09:13:00Z\nkind: episode\nby: server\naccount: john\n\
              channel: grok\ntext: y\n"
                .to_vec(),
        );
        files
    }

    /// **A restamp rewrites exactly the entries that named their app**, each
    /// to its current form byte for byte, and no other file.
    #[test]
    fn a_restamp_rewrites_only_the_entries_that_named_their_app() {
        let before = files_with_old_entries();
        let mut tree = load(&before);
        let applied = apply(&mut tree, &Actor::Zend, &Op::Restamp, now()).unwrap();
        let ch = changes(&before, &tree, &applied);
        assert_eq!(
            ch,
            BTreeMap::from([
                (
                    OLD_EPISODE.to_string(),
                    Some(
                        b"at: 2026-10-30T09:13:00Z\nkind: episode\nby: server\naccount: john\n\
                          text: y\n"
                            .to_vec()
                    )
                ),
                (
                    OLD_NOTE.to_string(),
                    Some(
                        b"at: 2026-10-30T09:12:40Z\nkind: note\naccount: john\ntext: x\n".to_vec()
                    )
                ),
            ])
        );
    }

    /// A tree already in the current form gives a restamp nothing to write.
    #[test]
    fn a_current_tree_gives_a_restamp_nothing_to_write() {
        let before = render_all(&zen());
        let mut tree = load(&before);
        let applied = apply(&mut tree, &Actor::Zend, &Op::Restamp, now()).unwrap();
        assert!(changes(&before, &tree, &applied).is_empty());
    }

    /// **A restamp is zend's**: no member's assistant rewrites every journal.
    #[test]
    fn only_zend_restamps() {
        let mut tree = zen();
        assert!(apply(&mut tree, &john(), &Op::Restamp, now()).is_err());
    }
}
