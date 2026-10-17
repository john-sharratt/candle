//! Three versions of one text merged line by line, overlaps kept between
//! conflict markers — what `git merge` leaves in a working tree's file. The
//! merge is git's own xdiff merge, run in this process
//! ([`merge_file`](crate::library::merge_file)).

use crate::error::GitError;
use crate::library::merge_file::{conflicts_in, merge_file, Labels};
use crate::runner::utf8;
use crate::Repo;

/// A merged text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedText {
    /// The merge, with each overlap between conflict markers.
    pub text: String,
    /// How many overlaps it holds; `0` for a clean merge.
    pub conflicts: usize,
}

/// The three sides' names as the conflict markers show them.
#[derive(Debug, Clone, Copy)]
pub struct MergeLabels<'a> {
    pub ours: &'a str,
    pub base: &'a str,
    pub theirs: &'a str,
}

impl Repo {
    /// Merge `ours` and `theirs`, each changed from `base`, line by line. A
    /// change only one side made is taken; changes that overlap are both kept,
    /// between `<<<<<<< ours` / `=======` / `>>>>>>> theirs` markers with the
    /// names `labels` gives.
    pub fn merge_text(
        &self,
        base: &str,
        ours: &str,
        theirs: &str,
        labels: MergeLabels<'_>,
    ) -> Result<MergedText, GitError> {
        let merged = merge_file(
            base.as_bytes(),
            ours.as_bytes(),
            theirs.as_bytes(),
            Some(Labels {
                ours: labels.ours,
                base: labels.base,
                theirs: labels.theirs,
            }),
        )?;
        let conflicts = if merged.clean {
            0
        } else {
            conflicts_in(&merged.bytes).max(1)
        };
        Ok(MergedText {
            text: utf8("merge-file", merged.bytes)?,
            conflicts,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::odb::memory_store::MemoryStore;
    use crate::odb::{ObjectDb, PackStore, DEFAULT_PACK_BUDGET};
    use crate::testing::TestRepo;

    const LABELS: MergeLabels<'static> = MergeLabels {
        ours: "yours",
        base: "base",
        theirs: "origin/main",
    };

    fn lines(n: usize) -> Vec<String> {
        (1..=n).map(|i| format!("line {i}\n")).collect()
    }

    /// **Changes to different lines merge cleanly**, both kept.
    #[test]
    fn changes_to_different_lines_merge_cleanly() {
        let t = TestRepo::init();
        let base = lines(10);
        let mut ours = base.clone();
        ours[1] = "OURS\n".into();
        let mut theirs = base.clone();
        theirs[8] = "THEIRS\n".into();
        let merged = t
            .repo()
            .merge_text(&base.concat(), &ours.concat(), &theirs.concat(), LABELS)
            .unwrap();
        let mut both = base.clone();
        both[1] = "OURS\n".into();
        both[8] = "THEIRS\n".into();
        assert_eq!(
            merged,
            MergedText {
                text: both.concat(),
                conflicts: 0
            }
        );
    }

    /// **Overlapping changes are both kept between labelled markers**, byte
    /// for byte what `git merge-file` writes.
    #[test]
    fn overlapping_changes_are_kept_between_markers() {
        let t = TestRepo::init();
        let merged = t
            .repo()
            .merge_text("a\nb\nc\n", "a\nmine\nc\n", "a\ntheirs\nc\n", LABELS)
            .unwrap();
        assert_eq!(
            merged,
            MergedText {
                text: "a\n<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> origin/main\nc\n"
                    .to_string(),
                conflicts: 1
            }
        );
    }

    /// **A repository read from the substrate merges text the same**, with no
    /// folder and no `git` program.
    #[test]
    fn a_repository_in_the_substrate_merges_the_same() {
        let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
        let db = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
        let merged = Repo::in_substrate(&db)
            .unwrap()
            .merge_text("a\nb\nc\n", "a\nmine\nc\n", "a\ntheirs\nc\n", LABELS)
            .unwrap();
        assert_eq!(merged.conflicts, 1);
        assert!(merged.text.contains("<<<<<<< yours\nmine\n"));
    }
}
