//! Three versions of one text merged line by line, overlaps kept between
//! conflict markers — what `git merge` leaves in a working tree's file.

use crate::error::GitError;
use crate::runner::utf8;
use crate::write::scratch::ScratchFile;
use crate::Repo;

/// The largest conflict count `merge-file` reports; any exit status above it
/// is an error.
const MAX_REPORTED_CONFLICTS: i32 = 127;

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
        let base_file = ScratchFile::new(self.git_dir(), "merge-base", base.as_bytes())?;
        let ours_file = ScratchFile::new(self.git_dir(), "merge-ours", ours.as_bytes())?;
        let theirs_file = ScratchFile::new(self.git_dir(), "merge-theirs", theirs.as_bytes())?;
        let out = self
            .git("merge-file")
            .args([
                "-p",
                "-L",
                labels.ours,
                "-L",
                labels.base,
                "-L",
                labels.theirs,
            ])
            .arg("--end-of-options")
            .arg(&ours_file.0)
            .arg(&base_file.0)
            .arg(&theirs_file.0)
            .run()?;
        // Exit 0: clean. 1–127: that many conflicts, capped. Anything else:
        // an error.
        let conflicts = match out.status {
            Some(n) if (0..=MAX_REPORTED_CONFLICTS).contains(&n) => n as usize,
            status => {
                return Err(GitError::Unclassified {
                    args: vec!["merge-file".to_string()],
                    status,
                    stderr: out.stderr,
                })
            }
        };
        Ok(MergedText {
            text: utf8("merge-file", out.stdout)?,
            conflicts,
        })
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

    /// No scratch file outlives the merge.
    #[test]
    fn scratch_files_are_removed() {
        let t = TestRepo::init();
        t.repo().merge_text("a\n", "b\n", "c\n", LABELS).unwrap();
        let left = std::fs::read_dir(t.path.join(".git"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("zen-"))
            .count();
        assert_eq!(left, 0);
    }
}
