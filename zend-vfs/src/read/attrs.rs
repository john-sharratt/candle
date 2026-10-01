//! Which paths `.gitignore` excludes.

use crate::error::GitError;
use crate::types::RepoPath;
use crate::Repo;

impl Repo {
    /// The subset of `paths` the repository's ignore rules exclude. A tracked
    /// file is never reported: ignore rules do not apply to what git already
    /// tracks.
    ///
    /// `check-ignore` refuses the literal-pathspec setting every other call
    /// runs under, so it is lifted here; a path beginning with `:` is then
    /// read as pathspec magic, which `check-ignore` refuses as an error
    /// rather than misreading.
    pub fn ignored(&self, paths: &[&RepoPath]) -> Result<Vec<RepoPath>, GitError> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let mut input = Vec::new();
        for p in paths {
            input.extend_from_slice(p.as_str().as_bytes());
            input.push(0);
        }
        let out = self
            .git("check-ignore")
            .args(["--stdin", "-z"])
            .env("GIT_LITERAL_PATHSPECS", "0")
            .stdin(input)
            .read_only()
            .run_accepting(&[0, 1])?;
        let text = std::str::from_utf8(&out.stdout)
            .map_err(|e| GitError::malformed("check-ignore", e.to_string()))?;
        text.split('\0')
            .filter(|p| !p.is_empty())
            .map(RepoPath::parse)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::TestRepo;
    use crate::types::RepoPath;

    #[test]
    fn ignored_paths_are_reported_and_tracked_ones_are_not() {
        let t = TestRepo::init();
        t.write(".gitignore", b"*.log\nbuild/\n");
        t.write("tracked.log", b"forced in\n");
        t.git(&["add", "-f", "tracked.log"]);
        t.commit_all("base");
        let p = |s| RepoPath::parse(s).unwrap();
        let (a, b, c, d) = (
            p("new.log"),
            p("build/out.bin"),
            p("src/lib.rs"),
            p("tracked.log"),
        );
        let ignored = t.repo().ignored(&[&a, &b, &c, &d]).unwrap();
        assert_eq!(ignored, vec![p("new.log"), p("build/out.bin")]);
        assert!(t.repo().ignored(&[&c]).unwrap().is_empty());
    }
}
