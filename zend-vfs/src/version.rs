//! The installed git's version, and the oldest the layer runs against.
//!
//! The floor is set by what the layer's commands need, not by security
//! releases: a deployment on an older distribution's git must work. The
//! protections those releases added are built into the layer instead, so
//! they hold on every supported version (`docs/zend_git.md` §4.7).

use std::fmt;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use crate::error::GitError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GitVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// The oldest git the layer runs against: 2.24, the release that added
/// `--end-of-options`, which every invocation that takes a value relies on.
/// Every other command and flag the layer uses is older. The test suite is
/// run against this release as well as current ones.
pub const MINIMUM: GitVersion = GitVersion {
    major: 2,
    minor: 24,
    patch: 0,
};

impl GitVersion {
    /// Parse `git --version` output, e.g. `git version 2.45.1.windows.1`.
    pub fn parse(output: &str) -> Result<Self, GitError> {
        let bad = || GitError::malformed("--version", output.trim().to_string());
        let rest = output.trim().strip_prefix("git version ").ok_or_else(bad)?;
        let mut parts = rest.split(['.', ' ']);
        let mut next = || -> Result<u32, GitError> {
            parts.next().ok_or_else(bad)?.parse().map_err(|_| bad())
        };
        let major = next()?;
        let minor = next()?;
        // `2.45.windows.1`-style strings have no patch number.
        let patch = next().unwrap_or(0);
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}

impl fmt::Display for GitVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The installed git's version, read once per process and checked against
/// [`MINIMUM`].
pub fn installed() -> Result<GitVersion, GitError> {
    static FOUND: OnceLock<Result<GitVersion, String>> = OnceLock::new();
    let found = FOUND.get_or_init(|| {
        let output = Command::new("git")
            .arg("--version")
            .stdin(Stdio::null())
            .output()
            .map_err(|e| e.to_string())?;
        GitVersion::parse(&String::from_utf8_lossy(&output.stdout)).map_err(|e| e.to_string())
    });
    let version = match found {
        Ok(v) => *v,
        Err(e) => return Err(GitError::GitMissing(std::io::Error::other(e.clone()))),
    };
    check(version)
}

fn check(version: GitVersion) -> Result<GitVersion, GitError> {
    if version < MINIMUM {
        return Err(GitError::GitTooOld {
            found: version,
            need: MINIMUM,
        });
    }
    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_and_vendor_version_strings_parse() {
        for (text, expected) in [
            ("git version 2.45.1.windows.1\n", (2, 45, 1)),
            ("git version 2.24.1.windows.2\n", (2, 24, 1)),
            ("git version 2.39.3 (Apple Git-146)\n", (2, 39, 3)),
        ] {
            let v = GitVersion::parse(text).unwrap();
            assert_eq!((v.major, v.minor, v.patch), expected, "{text:?}");
        }
    }

    #[test]
    fn versions_order_numerically() {
        let v = |s| GitVersion::parse(s).unwrap();
        assert!(v("git version 2.9.0") < v("git version 2.24.0"));
        assert!(v("git version 2.24.0") < v("git version 2.24.1"));
        assert!(v("git version 2.23.9") < MINIMUM);
    }

    #[test]
    fn something_that_is_not_a_version_is_malformed() {
        assert!(GitVersion::parse("command not found").is_err());
    }

    #[test]
    fn only_releases_before_the_minimum_are_refused() {
        let found = GitVersion::parse("git version 2.23.4").unwrap();
        assert!(matches!(
            check(found),
            Err(GitError::GitTooOld { need, .. }) if need == MINIMUM
        ));
        for ok in [
            "git version 2.24.0",
            "git version 2.34.1",
            "git version 2.45.1.windows.1",
            "git version 2.55.0.windows.5",
            "git version 3.0.0",
        ] {
            assert!(check(GitVersion::parse(ok).unwrap()).is_ok(), "{ok}");
        }
    }

    #[test]
    fn the_installed_git_meets_the_minimum() {
        let v = installed().expect("the test machine has git");
        assert!(v >= MINIMUM, "{v}");
    }
}
