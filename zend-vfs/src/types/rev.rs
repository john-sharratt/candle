//! Revisions, as the typed values a caller may name — never a free string.

use crate::types::{BranchName, Oid, RefName, TagName};

/// Something that resolves to a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rev {
    /// The working tree's `HEAD`.
    Head,
    Oid(Oid),
    Branch(BranchName),
    Tag(TagName),
    Ref(RefName),
}

impl Rev {
    /// The revision as git's command line takes it. A branch is spelled as
    /// its full ref, so a tag or file of the same name can never shadow it.
    /// No form begins with `-`.
    pub(crate) fn spec(&self) -> String {
        match self {
            Self::Head => "HEAD".to_string(),
            Self::Oid(oid) => oid.to_string(),
            Self::Branch(branch) => branch.to_ref().to_string(),
            Self::Tag(tag) => tag.to_ref().to_string(),
            Self::Ref(name) => name.to_string(),
        }
    }
}

impl From<Oid> for Rev {
    fn from(oid: Oid) -> Self {
        Self::Oid(oid)
    }
}

impl From<BranchName> for Rev {
    fn from(branch: BranchName) -> Self {
        Self::Branch(branch)
    }
}

impl From<RefName> for Rev {
    fn from(name: RefName) -> Self {
        Self::Ref(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_revision_is_spelled_unambiguously() {
        let oid = Oid::parse("4b825dc642cb6eb9a060e54bf8d69288fbee4904").unwrap();
        assert_eq!(Rev::Head.spec(), "HEAD");
        assert_eq!(Rev::from(oid.clone()).spec(), oid.as_str());
        assert_eq!(
            Rev::from(BranchName::parse("main").unwrap()).spec(),
            "refs/heads/main"
        );
        assert_eq!(
            Rev::from(RefName::parse("refs/remotes/origin/main").unwrap()).spec(),
            "refs/remotes/origin/main"
        );
    }
}
