//! What a store's changes are made on — the conversation's own `HEAD`.

use serde::{Deserialize, Serialize};

use crate::{Oid, VfsError};

/// The tree a store's changes are made against, and the commits a commit of
/// them has for parents.
///
/// A store over a branch takes its base from the branch the first time it
/// reads it, and keeps it: after that, the base moves only when the
/// conversation moves it — a commit of its own, a merge, a switch, a reset.
/// So a conversation reads one consistent commit however the branch moves
/// under it, and another conversation's commit to the same branch reaches it
/// only through a merge, which shows where the two meet.
///
/// One parent, normally, whose tree is [`Self::tree`]. Two while a merge is
/// being finished: the conversation's commit and the one merged into it, with
/// the tree the merge settled — and every conflict it left for the
/// conversation to settle in its own copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Base {
    /// `None` for a branch with no commit yet: no files.
    pub tree: Option<Oid>,
    pub parents: Vec<Oid>,
}

impl Base {
    /// `commit`, whose tree is `tree`.
    pub fn at(commit: Oid, tree: Oid) -> Self {
        Self {
            tree: Some(tree),
            parents: vec![commit],
        }
    }

    /// Nothing yet: a branch with no commit.
    pub fn empty() -> Self {
        Self {
            tree: None,
            parents: Vec::new(),
        }
    }

    /// The conversation's own commit — the first parent.
    pub fn commit(&self) -> Option<&Oid> {
        self.parents.first()
    }

    /// The commit being merged in, while a merge is being finished.
    pub fn merging(&self) -> Option<&Oid> {
        self.parents.get(1)
    }

    pub(super) fn saved(&self) -> SavedBase {
        SavedBase {
            tree: self.tree.as_ref().map(|t| t.as_str().to_string()),
            parents: self
                .parents
                .iter()
                .map(|p| p.as_str().to_string())
                .collect(),
        }
    }
}

/// A [`Base`] as a conversation saves it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedBase {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tree: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    parents: Vec<String>,
}

impl SavedBase {
    pub(super) fn parse(&self) -> Result<Base, VfsError> {
        let id = |s: &String| {
            Oid::parse(s)
                .map_err(|e| VfsError::Unreadable(format!("the saved base names no object: {e}")))
        };
        Ok(Base {
            tree: self.tree.as_ref().map(id).transpose()?,
            parents: self.parents.iter().map(id).collect::<Result<_, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    /// **A base saves and parses back exactly**, a merge's two parents in
    /// order; one naming something that is not an object id is refused.
    #[test]
    fn a_base_saves_and_parses_back() {
        let merging = Base {
            tree: Some(Oid::parse(A).unwrap()),
            parents: vec![Oid::parse(A).unwrap(), Oid::parse(B).unwrap()],
        };
        assert_eq!(merging.saved().parse().unwrap(), merging);
        assert_eq!(merging.commit(), Some(&Oid::parse(A).unwrap()));
        assert_eq!(merging.merging(), Some(&Oid::parse(B).unwrap()));
        assert_eq!(
            serde_json::to_value(merging.saved()).unwrap(),
            serde_json::json!({"tree": A, "parents": [A, B]})
        );
        assert_eq!(Base::empty().saved().parse().unwrap(), Base::empty());
        assert_eq!(
            serde_json::to_value(Base::empty().saved()).unwrap(),
            serde_json::json!({})
        );
        let bad: SavedBase = serde_json::from_value(serde_json::json!({"tree": "nope"})).unwrap();
        assert!(bad.parse().is_err());
    }
}
