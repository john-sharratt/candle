//! Creating and deleting tags.

use crate::error::GitError;
use crate::runner::utf8;
use crate::types::{Oid, Signature, TagName};
use crate::write::fast_import::normalize_message;
use crate::write::ref_txn::{RefOp, RefTransaction};
use crate::Repo;

/// What makes a tag annotated: a message and who tagged it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagAnnotation {
    pub message: String,
    pub tagger: Signature,
}

/// The tag object `git mktag` validates and stores.
pub(crate) fn tag_object(
    object: &Oid,
    kind: &str,
    name: &TagName,
    annotation: &TagAnnotation,
) -> String {
    format!(
        "object {object}\ntype {kind}\ntag {name}\ntagger {}\n\n{}",
        annotation.tagger.to_header(),
        normalize_message(&annotation.message)
    )
}

impl Repo {
    /// Tag `target`, which must not already be tagged `name`. An annotation
    /// makes an annotated tag object; without one the tag is lightweight.
    /// Returns what the tag ref now holds.
    pub fn create_tag(
        &self,
        name: &TagName,
        target: &Oid,
        annotation: Option<&TagAnnotation>,
    ) -> Result<Oid, GitError> {
        let _write = self.write_lock();
        let value = match annotation {
            None => target.clone(),
            Some(annotation) => {
                let kind = self
                    .git("cat-file")
                    .args(["-t", "--end-of-options"])
                    .arg(target.as_str())
                    .about_rev(target.to_string())
                    .run_ok()?;
                let kind = utf8("cat-file", kind)?;
                let object = tag_object(target, kind.trim(), name, annotation);
                let out = self.git("mktag").stdin(object.into_bytes()).run_ok()?;
                Oid::parse(utf8("mktag", out)?.trim())?
            }
        };
        self.update_refs_locked(&RefTransaction::new().push(RefOp::Create {
            name: name.to_ref(),
            new: value.clone(),
        }))?;
        Ok(value)
    }

    /// Delete tag `name`, which must hold `old`.
    pub fn delete_tag(&self, name: &TagName, old: &Oid) -> Result<(), GitError> {
        self.update_refs(&RefTransaction::new().push(RefOp::Delete {
            name: name.to_ref(),
            old: old.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestRepo, SETUP_DATE};
    use crate::types::{GitTime, Rev};

    fn setup_sig() -> Signature {
        Signature::new(
            "Setup",
            "setup@example.com",
            GitTime::parse_raw(SETUP_DATE).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn the_tag_object_is_exactly_these_bytes() {
        let oid = Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap();
        let obj = tag_object(
            &oid,
            "commit",
            &TagName::parse("v1.0").unwrap(),
            &TagAnnotation {
                message: "Release 1.0".into(),
                tagger: setup_sig(),
            },
        );
        assert_eq!(
            obj,
            "object ce013625030ba8dba906f756967f9e9ca394464a\ntype commit\ntag v1.0\n\
tagger Setup <setup@example.com> 1700000000 +0000\n\nRelease 1.0\n"
        );
    }

    /// **An annotated tag is the object `git tag -a` makes** — same id —
    /// for the same target, name, tagger and message.
    #[test]
    fn an_annotated_tag_matches_git_tag() {
        let t = TestRepo::init();
        t.write("a", b"a\n");
        let c = t.commit_all("base");
        t.git(&["tag", "-a", "-m", "Release 1.0", "oracle", c.as_str()]);
        let oracle_obj = t.oid("refs/tags/oracle");
        let repo = t.repo();
        let v = TagName::parse("oracle-twin").unwrap();

        // Same fields but a different name: rebuild the oracle's object with
        // our name to compare byte for byte.
        let ours = repo
            .create_tag(
                &v,
                &c,
                Some(&TagAnnotation {
                    message: "Release 1.0".into(),
                    tagger: setup_sig(),
                }),
            )
            .unwrap();
        let theirs = t.git(&["cat-file", "tag", oracle_obj.as_str()]);
        let mine = t.git(&["cat-file", "tag", ours.as_str()]);
        assert_eq!(mine, theirs.replace("tag oracle\n", "tag oracle-twin\n"));

        let tags = repo.tags().unwrap();
        let twin = tags.iter().find(|t| t.name == v).unwrap();
        assert!(twin.annotated);
        assert_eq!(twin.oid, ours);
        assert_eq!(twin.target, c);
        assert_eq!(
            repo.resolve(&Rev::Tag(v)).unwrap(),
            c,
            "peels to the commit"
        );
    }

    #[test]
    fn lightweight_tags_create_once_and_delete_under_a_lease() {
        let t = TestRepo::init();
        t.write("a", b"1\n");
        let first = t.commit_all("first");
        t.write("a", b"2\n");
        let second = t.commit_all("second");
        let repo = t.repo();
        let v = TagName::parse("v1").unwrap();

        assert_eq!(repo.create_tag(&v, &first, None).unwrap(), first);
        assert!(matches!(
            repo.create_tag(&v, &second, None),
            Err(GitError::StaleRef { .. })
        ));
        let tags = repo.tags().unwrap();
        assert_eq!(tags.len(), 1);
        assert!(!tags[0].annotated);
        assert_eq!(tags[0].target, first);

        assert!(matches!(
            repo.delete_tag(&v, &second),
            Err(GitError::StaleRef { .. })
        ));
        repo.delete_tag(&v, &first).unwrap();
        assert!(repo.tags().unwrap().is_empty());
    }
}
