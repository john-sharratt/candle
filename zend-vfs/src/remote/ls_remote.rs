//! What a remote's refs point at, without fetching anything.

use crate::error::GitError;
use crate::types::{BranchName, Oid, RefName, RemoteName};
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteRefs {
    /// The branch the remote's `HEAD` points at, when it says.
    pub head_branch: Option<BranchName>,
    /// Every ref under `refs/`, peeled tag entries excluded.
    pub refs: Vec<(RefName, Oid)>,
}

impl RemoteRefs {
    /// Where `branch` points on the remote, if it exists there.
    pub fn branch(&self, branch: &BranchName) -> Option<&Oid> {
        self.get(&branch.to_ref())
    }

    /// What `name` holds on the remote — a tag's own object, unpeeled — if
    /// it exists there.
    pub fn get(&self, name: &RefName) -> Option<&Oid> {
        self.refs.iter().find(|(r, _)| r == name).map(|(_, o)| o)
    }
}

/// Parse `ls-remote --symref` output.
pub(crate) fn parse_ls_remote(out: &str) -> Result<RemoteRefs, GitError> {
    let mut head_branch = None;
    let mut refs = Vec::new();
    for line in out.lines().filter(|l| !l.is_empty()) {
        let bad = || GitError::malformed("ls-remote", line.to_string());
        let (left, name) = line.split_once('\t').ok_or_else(bad)?;
        if let Some(target) = left.strip_prefix("ref: ") {
            if name == "HEAD" {
                head_branch = RefName::parse(target)?.branch();
            }
            continue;
        }
        if name == "HEAD" || name.ends_with("^{}") {
            continue;
        }
        refs.push((RefName::parse(name)?, Oid::parse(left)?));
    }
    Ok(RemoteRefs { head_branch, refs })
}

impl Repo {
    pub fn ls_remote(&self, remote: &RemoteName) -> Result<RemoteRefs, GitError> {
        self.require_remote(remote)?;
        let out = self
            .git("ls-remote")
            .args(["--symref", "--end-of-options"])
            .arg(remote.as_str())
            .network(remote)
            .run_ok()?;
        let text =
            String::from_utf8(out).map_err(|e| GitError::malformed("ls-remote", e.to_string()))?;
        parse_ls_remote(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    #[test]
    fn output_parses_skipping_head_and_peeled_tags() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let b = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
        let out = format!(
            "ref: refs/heads/main\tHEAD\n{a}\tHEAD\n{a}\trefs/heads/main\n{b}\trefs/tags/v1\n{a}\trefs/tags/v1^{{}}\n"
        );
        let r = parse_ls_remote(&out).unwrap();
        assert_eq!(r.head_branch, Some(BranchName::parse("main").unwrap()));
        assert_eq!(r.refs.len(), 2);
        assert_eq!(
            r.branch(&BranchName::parse("main").unwrap())
                .unwrap()
                .as_str(),
            a
        );
    }

    #[test]
    fn a_remote_lists_its_branches() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"a\n");
        let c = t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "origin", "main", "main:side"]);
        let r = t
            .repo()
            .ls_remote(&RemoteName::parse("origin").unwrap())
            .unwrap();
        assert_eq!(r.head_branch, Some(BranchName::parse("main").unwrap()));
        assert_eq!(r.branch(&BranchName::parse("side").unwrap()), Some(&c));
    }

    #[test]
    fn an_unreachable_remote_is_typed() {
        let t = TestRepo::init();
        let missing = t.path.join("no-such-origin");
        t.git(&["remote", "add", "origin", missing.to_str().unwrap()]);
        let e = t
            .repo()
            .ls_remote(&RemoteName::parse("origin").unwrap())
            .unwrap_err();
        assert!(matches!(e, GitError::RemoteUnreachable { .. }), "{e}");
    }
}
