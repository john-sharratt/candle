//! A fresh repository.

use std::path::Path;

use crate::error::GitError;
use crate::runner::Invocation;
use crate::setup::require_empty_target;
use crate::types::BranchName;
use crate::version;
use crate::Repo;

impl Repo {
    /// Create an empty repository at `dir` — which must not exist or must
    /// be an empty folder — with `branch` as its unborn first branch.
    pub fn init(dir: &Path, branch: &BranchName) -> Result<Repo, GitError> {
        version::installed()?;
        require_empty_target(dir)?;
        std::fs::create_dir_all(dir)?;
        // `init -b` needs 2.28; pointing the unborn HEAD at the branch is
        // the same thing on every version.
        Invocation::new(dir, "init").arg("-q").run_ok()?;
        Invocation::new(dir, "symbolic-ref")
            .arg("HEAD")
            .arg(branch.to_ref().as_str())
            .run_ok()?;
        Repo::open(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::head::Head;
    use crate::testing::scratch;

    #[test]
    fn a_fresh_repository_is_unborn_on_its_branch() {
        let dir = tempfile::Builder::new()
            .prefix("init-")
            .tempdir_in(scratch())
            .unwrap();
        let target = dir.path().join("fresh");
        let trunk = BranchName::parse("trunk").unwrap();
        let repo = Repo::init(&target, &trunk).unwrap();
        assert_eq!(repo.head().unwrap(), Head::Unborn(trunk));
        assert!(target.join(".git").is_dir());
    }

    #[test]
    fn a_non_empty_or_relative_target_is_refused() {
        let dir = tempfile::Builder::new()
            .prefix("init-")
            .tempdir_in(scratch())
            .unwrap();
        std::fs::write(dir.path().join("file"), b"x").unwrap();
        let main = BranchName::parse("main").unwrap();
        assert!(matches!(
            Repo::init(dir.path(), &main),
            Err(GitError::InvalidInput(_))
        ));
        assert!(matches!(
            Repo::init(Path::new("relative/path"), &main),
            Err(GitError::InvalidInput(_))
        ));
    }
}
