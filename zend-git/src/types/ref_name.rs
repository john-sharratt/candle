//! Reference, branch and remote names, validated by `git check-ref-format`'s
//! rules in Rust so an invalid name never reaches a command line.

use std::fmt;

use crate::error::GitError;

/// Why `name` is not a valid full reference name, or `None` when it is.
///
/// The rules of `git check-ref-format` for a multi-level name:
/// no empty component, no component starting with `.` or ending with
/// `.lock`, no `..`, no ASCII control character, space, `~ ^ : ? * [ \`,
/// no `@{`, not `@`, and no trailing `.`.
fn ref_format_violation(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        return Some("is empty");
    }
    if name == "@" {
        return Some("is `@`");
    }
    if name.contains("..") {
        return Some("contains `..`");
    }
    if name.contains("@{") {
        return Some("contains `@{`");
    }
    if name.ends_with('.') {
        return Some("ends with `.`");
    }
    for c in name.chars() {
        if c.is_ascii_control() {
            return Some("contains a control character");
        }
        if matches!(c, ' ' | '~' | '^' | ':' | '?' | '*' | '[' | '\\') {
            return Some("contains one of ` ~^:?*[\\`");
        }
    }
    for component in name.split('/') {
        if component.is_empty() {
            return Some("has an empty component");
        }
        if component.starts_with('.') {
            return Some("has a component starting with `.`");
        }
        if component.ends_with(".lock") {
            return Some("has a component ending with `.lock`");
        }
    }
    None
}

/// A full reference name under `refs/`, e.g. `refs/heads/main`.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RefName(String);

impl RefName {
    pub fn parse(name: &str) -> Result<Self, GitError> {
        if !name.starts_with("refs/") {
            return Err(GitError::invalid(format!(
                "ref {name:?} is not under refs/"
            )));
        }
        if let Some(why) = ref_format_violation(name) {
            return Err(GitError::invalid(format!("ref {name:?} {why}")));
        }
        Ok(Self(name.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The branch this ref names, when it is under `refs/heads/`.
    pub fn branch(&self) -> Option<BranchName> {
        self.0
            .strip_prefix(BranchName::PREFIX)
            .map(|short| BranchName(short.to_string()))
    }
}

impl fmt::Display for RefName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for RefName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RefName({})", self.0)
    }
}

/// A branch, by its short name (`main`, `zen/fix-tick`). Its ref is
/// `refs/heads/<name>`.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BranchName(String);

impl BranchName {
    const PREFIX: &'static str = "refs/heads/";

    pub fn parse(name: &str) -> Result<Self, GitError> {
        if name.starts_with('-') {
            return Err(GitError::invalid(format!(
                "branch {name:?} starts with `-`"
            )));
        }
        if name == "HEAD" {
            return Err(GitError::invalid("a branch may not be named HEAD"));
        }
        if let Some(why) = ref_format_violation(name) {
            return Err(GitError::invalid(format!("branch {name:?} {why}")));
        }
        Ok(Self(name.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The branch's full reference, `refs/heads/<name>`.
    pub fn to_ref(&self) -> RefName {
        RefName(format!("{}{}", Self::PREFIX, self.0))
    }
}

impl fmt::Display for BranchName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for BranchName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BranchName({})", self.0)
    }
}

/// A configured remote's name, e.g. `origin`: one component.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RemoteName(String);

impl RemoteName {
    pub fn parse(name: &str) -> Result<Self, GitError> {
        if name.starts_with('-') || name.contains('/') {
            return Err(GitError::invalid(format!(
                "remote {name:?} must be one component not starting with `-`"
            )));
        }
        if let Some(why) = ref_format_violation(name) {
            return Err(GitError::invalid(format!("remote {name:?} {why}")));
        }
        Ok(Self(name.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The remote-tracking ref for `branch`: `refs/remotes/<remote>/<branch>`.
    pub fn tracking(&self, branch: &BranchName) -> RefName {
        RefName(format!("refs/remotes/{}/{}", self.0, branch.0))
    }
}

impl fmt::Display for RemoteName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for RemoteName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RemoteName({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::runner::Invocation;
    use crate::testing::TestRepo;

    /// Names the Rust rules and `git check-ref-format` must agree on, as
    /// `(name, valid)`.
    const REF_TABLE: &[(&str, bool)] = &[
        ("refs/heads/main", true),
        ("refs/heads/zen/fix-tick", true),
        ("refs/heads/feature.x", true),
        ("refs/heads/ünïcode", true),
        ("refs/heads/a..b", false),
        ("refs/heads/.hidden", false),
        ("refs/heads/x.lock", false),
        ("refs/heads/x/", false),
        ("refs/heads//x", false),
        ("refs/heads/x.", false),
        ("refs/heads/a b", false),
        ("refs/heads/a~1", false),
        ("refs/heads/a^", false),
        ("refs/heads/a:b", false),
        ("refs/heads/a?", false),
        ("refs/heads/a*", false),
        ("refs/heads/a[b", false),
        ("refs/heads/a\\b", false),
        ("refs/heads/a@{1}", false),
        ("refs/heads/tab\there", false),
        ("refs/heads/@", true),
    ];

    #[test]
    fn the_ref_table_holds_for_the_rust_rules() {
        for (name, valid) in REF_TABLE {
            assert_eq!(RefName::parse(name).is_ok(), *valid, "{name:?}");
        }
    }

    /// **The Rust rules agree with git's own**, name for name, so the two
    /// cannot drift apart.
    #[test]
    fn the_ref_table_holds_for_the_installed_git() {
        let t = TestRepo::init();
        for (name, valid) in REF_TABLE {
            let out = Invocation::new(&t.path, "check-ref-format")
                .arg(*name)
                .run()
                .unwrap();
            assert_eq!(out.status == Some(0), *valid, "git disagrees on {name:?}");
        }
    }

    #[test]
    fn a_ref_outside_refs_is_refused() {
        assert!(RefName::parse("heads/main").is_err());
        assert!(RefName::parse("HEAD").is_err());
    }

    #[test]
    fn branches_refuse_flags_and_head() {
        for bad in ["-x", "--upload-pack=evil", "HEAD", "a..b", "", "@"] {
            assert!(BranchName::parse(bad).is_err(), "{bad:?} should be refused");
        }
        let b = BranchName::parse("zen/fix-tick").unwrap();
        assert_eq!(b.to_ref().as_str(), "refs/heads/zen/fix-tick");
        assert_eq!(b.to_ref().branch(), Some(b));
    }

    #[test]
    fn remotes_are_one_component() {
        assert!(RemoteName::parse("origin").is_ok());
        for bad in ["a/b", "-x", "", "a..b"] {
            assert!(RemoteName::parse(bad).is_err(), "{bad:?} should be refused");
        }
        let origin = RemoteName::parse("origin").unwrap();
        let main = BranchName::parse("main").unwrap();
        assert_eq!(origin.tracking(&main).as_str(), "refs/remotes/origin/main");
    }
}
