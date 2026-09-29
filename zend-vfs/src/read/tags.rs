//! Listing tags.

use crate::error::GitError;
use crate::runner::utf8;
use crate::types::{Oid, RefName, TagName};
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    pub name: TagName,
    /// What the tag ref holds: the tag object for an annotated tag, the
    /// tagged object itself for a lightweight one.
    pub oid: Oid,
    /// The object the tag finally points at.
    pub target: Oid,
    pub annotated: bool,
}

pub(crate) fn parse_tags(out: &str) -> Result<Vec<Tag>, GitError> {
    out.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let bad = || GitError::malformed("for-each-ref", line.to_string());
            let fields: Vec<&str> = line.split('\0').collect();
            let [refname, oid, kind, peeled] = fields[..] else {
                return Err(bad());
            };
            let name = TagName::from_ref(&RefName::parse(refname)?).ok_or_else(bad)?;
            let oid = Oid::parse(oid)?;
            let annotated = kind == "tag";
            let target = if annotated {
                Oid::parse(peeled)?
            } else {
                oid.clone()
            };
            Ok(Tag {
                name,
                oid,
                target,
                annotated,
            })
        })
        .collect()
}

impl Repo {
    /// Every tag, by name.
    pub fn tags(&self) -> Result<Vec<Tag>, GitError> {
        let out = self
            .git("for-each-ref")
            .arg("--format=%(refname)%00%(objectname)%00%(objecttype)%00%(*objectname)")
            .arg("refs/tags/")
            .read_only()
            .run_ok()?;
        parse_tags(&utf8("for-each-ref", out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lightweight_and_annotated_tags_parse() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let t = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
        let out = format!("refs/tags/light\0{a}\0commit\0\nrefs/tags/v1\0{t}\0tag\0{a}\n");
        let tags = parse_tags(&out).unwrap();
        assert_eq!(tags[0].name.as_str(), "light");
        assert!(!tags[0].annotated);
        assert_eq!(tags[0].target.as_str(), a);
        assert!(tags[1].annotated);
        assert_eq!(tags[1].oid.as_str(), t);
        assert_eq!(tags[1].target.as_str(), a);
    }
}
