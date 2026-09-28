//! Tag names.

use std::fmt;

use crate::error::GitError;
use crate::types::RefName;

/// A tag, by its short name (`v1.2.0`). Its ref is `refs/tags/<name>`.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TagName(String);

impl TagName {
    const PREFIX: &'static str = "refs/tags/";

    pub fn parse(name: &str) -> Result<Self, GitError> {
        if name.starts_with('-') {
            return Err(GitError::invalid(format!("tag {name:?} starts with `-`")));
        }
        // The full ref carries every check-ref-format rule.
        RefName::parse(&format!("{}{name}", Self::PREFIX))
            .map_err(|_| GitError::invalid(format!("tag {name:?} is not a valid ref name")))?;
        Ok(Self(name.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn to_ref(&self) -> RefName {
        RefName::parse(&format!("{}{}", Self::PREFIX, self.0)).expect("validated at parse")
    }

    /// The tag a ref under `refs/tags/` names.
    pub(crate) fn from_ref(name: &RefName) -> Option<Self> {
        name.as_str()
            .strip_prefix(Self::PREFIX)
            .map(|short| Self(short.to_string()))
    }
}

impl fmt::Display for TagName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for TagName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TagName({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_validate_like_refs_and_refuse_flags() {
        let v = TagName::parse("v1.2.0").unwrap();
        assert_eq!(v.to_ref().as_str(), "refs/tags/v1.2.0");
        assert_eq!(TagName::from_ref(&v.to_ref()), Some(v));
        for bad in ["-x", "", "a..b", "v1 0", "v1^", "x.lock"] {
            assert!(TagName::parse(bad).is_err(), "{bad:?} should be refused");
        }
    }
}
