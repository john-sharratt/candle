//! Repository-relative paths, as git stores them.

use std::fmt;

use crate::error::GitError;

/// The path segment a repository keeps its secrets under. The file tools
/// refuse it (`zend_tools::state::vfs::PROTECTED_SEGMENT`), and a
/// [`ChangeSet`](crate::ChangeSet) refuses it too, so nothing the model cannot
/// read can be committed on its behalf.
pub const PROTECTED_SEGMENT: &str = "secrets";

/// A path inside a repository: `/`-separated, relative, with no empty, `.` or
/// `..` component, no `.git` component (nor its Windows short name
/// `git~1`), no backslash and no control character.
///
/// Structural only: it is what every path git prints satisfies, so reading
/// repository state never fails on a path. The protected segment is a rule
/// about writes, enforced by [`ChangeSet`](crate::ChangeSet).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RepoPath(String);

/// A component as Windows would open it: case folded, trailing dots and
/// spaces dropped — so `.GIT.` and `Secrets ` are the names they alias.
fn windows_name(component: &str) -> String {
    component.trim_end_matches(['.', ' ']).to_ascii_lowercase()
}

impl RepoPath {
    pub fn parse(path: &str) -> Result<Self, GitError> {
        let refuse = |why: &str| Err(GitError::invalid(format!("path {path:?} {why}")));
        if path.is_empty() {
            return refuse("is empty");
        }
        if path.starts_with('/') {
            return refuse("is absolute");
        }
        if path.contains('\\') {
            return refuse("contains a backslash");
        }
        if path.chars().any(|c| c.is_control()) {
            return refuse("contains a control character");
        }
        for component in path.split('/') {
            match component {
                "" => return refuse("has an empty component"),
                "." | ".." => return refuse("has a `.` or `..` component"),
                _ => {}
            }
            let name = windows_name(component);
            if name == ".git" || name == "git~1" {
                return refuse("reaches into `.git`");
            }
        }
        Ok(Self(path.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether any component is [`PROTECTED_SEGMENT`], as Windows would
    /// resolve it.
    pub fn is_protected(&self) -> bool {
        self.0
            .split('/')
            .any(|c| windows_name(c) == PROTECTED_SEGMENT)
    }

    /// `name` joined under `dir`, validated as a whole.
    pub fn join(dir: &RepoPath, name: &str) -> Result<Self, GitError> {
        Self::parse(&format!("{}/{name}", dir.0))
    }
}

impl fmt::Display for RepoPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for RepoPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RepoPath({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_paths_parse() {
        for ok in [
            "README.md",
            "src/lib.rs",
            "a dir/with spaces.txt",
            "ünïcode/файл.rs",
            ".gitignore",
            ".github/workflows/ci.yml",
            "git~2",
        ] {
            assert_eq!(RepoPath::parse(ok).unwrap().as_str(), ok);
        }
    }

    #[test]
    fn escaping_and_git_internal_paths_are_refused() {
        for bad in [
            "",
            "/etc/passwd",
            "../sibling/x",
            "a/../../x",
            "a/./b",
            "a//b",
            "a/",
            ".git/config",
            "sub/.git/hooks/pre-push",
            ".GIT/config",
            ".git./config",
            "GIT~1/config",
            "a\\b",
            "new\nline",
            "nul\0byte",
        ] {
            assert!(RepoPath::parse(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn the_protected_segment_is_detected_as_windows_resolves_it() {
        for protected in [
            "secrets/key",
            "web/secrets/auth.yaml",
            "Secrets/x",
            "SECRETS. /x",
        ] {
            assert!(
                RepoPath::parse(protected).unwrap().is_protected(),
                "{protected:?}"
            );
        }
        for open in ["secretsfile", "my-secrets/x", "src/secret.rs"] {
            assert!(!RepoPath::parse(open).unwrap().is_protected(), "{open:?}");
        }
    }

    #[test]
    fn join_validates_the_result() {
        let dir = RepoPath::parse("src").unwrap();
        assert_eq!(
            RepoPath::join(&dir, "lib.rs").unwrap().as_str(),
            "src/lib.rs"
        );
        assert!(RepoPath::join(&dir, "..").is_err());
    }
}
