//! How a path reads after its base moves: kept as the conversation holds
//! it, or merged three ways.

use crate::vfs::{Carried, Resolved, Side};
use crate::write::merge_text::MergeLabels;
use crate::{Repo, VfsError};

/// The conversation's content, whatever the move brought — what its own
/// commit, or a reset that keeps its work, carries.
pub fn keep_ours(carried: &Carried<'_>) -> Result<Resolved, VfsError> {
    Ok(Resolved {
        content: carried.ours.map(str::to_string),
        conflict: false,
    })
}

/// A three-way merge of each path: the conversation's content (`ours`) and
/// the incoming one (`theirs`), each changed from a common `base`.
pub struct ThreeWay<'r> {
    repo: &'r Repo,
    labels: MergeLabels<'r>,
}

impl<'r> ThreeWay<'r> {
    /// Merges in `repo`, with conflict markers naming the sides `labels`.
    pub fn new(repo: &'r Repo, labels: MergeLabels<'r>) -> Self {
        Self { repo, labels }
    }

    /// The path a move carries, merged from its old copy to its new one.
    pub fn carried(&self, carried: &Carried<'_>) -> Result<Resolved, VfsError> {
        self.merge(carried.path, carried.was, carried.ours, carried.now)
    }

    /// `path` merged:
    ///
    /// - one side unchanged from `base` — the other side's content;
    /// - both sides the same — that;
    /// - both changed, both present — merged line by line, overlaps between
    ///   markers and the path in conflict ([`Repo::merge_text`]); a file both
    ///   sides added merges over an empty base;
    /// - one side deleted it and the other changed it — the changed content,
    ///   in conflict: keeping it or deleting it is the conversation's call.
    ///
    /// A side that is not text cannot be merged here, and is refused.
    pub fn merge(
        &self,
        path: &str,
        base: &Side,
        ours: Option<&str>,
        theirs: &Side,
    ) -> Result<Resolved, VfsError> {
        let binary = || {
            VfsError::Unreadable(format!(
                "{path} is not text and changed on both sides; it cannot be merged here"
            ))
        };
        let text = |side: &'_ Side| -> Result<Option<String>, VfsError> {
            match side {
                Side::Absent => Ok(None),
                Side::Text(text) => Ok(Some(text.clone())),
                Side::Binary => Err(binary()),
            }
        };
        let (base, theirs) = (text(base)?, text(theirs)?);
        let ours = ours.map(str::to_string);
        let clean = |content: Option<String>| Resolved {
            content,
            conflict: false,
        };
        if ours == theirs || theirs == base {
            return Ok(clean(ours));
        }
        if ours == base {
            return Ok(clean(theirs));
        }
        match (ours, theirs) {
            (Some(ours), Some(theirs)) => {
                let merged = self
                    .repo
                    .merge_text(base.as_deref().unwrap_or(""), &ours, &theirs, self.labels)
                    .map_err(|e| {
                        VfsError::Unreadable(format!("{path} could not be merged: {e}"))
                    })?;
                Ok(Resolved {
                    content: Some(merged.text),
                    conflict: merged.conflicts > 0,
                })
            }
            (changed, deleted) => Ok(Resolved {
                content: changed.or(deleted),
                conflict: true,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    const LABELS: MergeLabels<'static> = MergeLabels {
        ours: "yours",
        base: "base",
        theirs: "origin/main",
    };

    fn text(s: &str) -> Side {
        Side::Text(s.to_string())
    }

    fn resolved(content: Option<&str>, conflict: bool) -> Resolved {
        Resolved {
            content: content.map(str::to_string),
            conflict,
        }
    }

    /// **Every shape of three versions settles as git would** — one side
    /// changed takes that side, the same change on both is one change,
    /// disjoint edits merge, overlapping ones are marked, and a delete meeting
    /// an edit keeps the edit, in conflict.
    #[test]
    fn every_shape_settles_as_git_would() {
        let t = TestRepo::init();
        let repo = t.repo();
        let merge = ThreeWay::new(&repo, LABELS);
        let m = |base: Side, ours: Option<&str>, theirs: Side| {
            merge.merge("f.txt", &base, ours, &theirs).unwrap()
        };
        assert_eq!(
            m(text("a\n"), Some("a\n"), text("b\n")),
            resolved(Some("b\n"), false),
            "only theirs changed"
        );
        assert_eq!(
            m(text("a\n"), Some("b\n"), text("a\n")),
            resolved(Some("b\n"), false),
            "only ours changed"
        );
        assert_eq!(
            m(text("a\n"), Some("b\n"), text("b\n")),
            resolved(Some("b\n"), false),
            "the same change"
        );
        assert_eq!(
            m(text("a\n"), None, Side::Absent),
            resolved(None, false),
            "both deleted"
        );
        assert_eq!(
            m(text("a\n"), Some("a\n"), Side::Absent),
            resolved(None, false),
            "theirs deleted what ours left alone"
        );
        assert_eq!(
            m(
                text("1\n2\n3\n4\n5\n"),
                Some("one\n2\n3\n4\n5\n"),
                text("1\n2\n3\n4\nfive\n")
            ),
            resolved(Some("one\n2\n3\n4\nfive\n"), false),
            "disjoint edits"
        );
        assert_eq!(
            m(text("a\n"), Some("mine\n"), text("theirs\n")),
            resolved(
                Some("<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> origin/main\n"),
                true
            ),
            "overlapping edits"
        );
        assert_eq!(
            m(Side::Absent, Some("mine\n"), text("theirs\n")),
            resolved(
                Some("<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> origin/main\n"),
                true
            ),
            "both added"
        );
        assert_eq!(
            m(text("a\n"), None, text("edited\n")),
            resolved(Some("edited\n"), true),
            "ours deleted, theirs edited"
        );
        assert_eq!(
            m(text("a\n"), Some("edited\n"), Side::Absent),
            resolved(Some("edited\n"), true),
            "ours edited, theirs deleted"
        );
    }

    /// **A binary side is refused**, never guessed at.
    #[test]
    fn a_binary_side_is_refused() {
        let t = TestRepo::init();
        let repo = t.repo();
        let merge = ThreeWay::new(&repo, LABELS);
        assert!(merge
            .merge("b.dat", &Side::Binary, Some("x\n"), &Side::Binary)
            .is_err());
        assert!(merge
            .merge("b.dat", &text("a\n"), Some("x\n"), &Side::Binary)
            .is_err());
    }
}
