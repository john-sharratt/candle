//! The remote-tracking branches the last fetch left locally.

use crate::error::GitError;
use crate::runner::utf8;
use crate::types::{BranchName, Oid, RefName, RemoteName};
use crate::Repo;

/// A branch as last fetched from a remote: `refs/remotes/<remote>/<branch>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteBranch {
    pub remote: RemoteName,
    pub branch: BranchName,
    pub oid: Oid,
}

impl RemoteBranch {
    pub fn to_ref(&self) -> RefName {
        self.remote.tracking(&self.branch)
    }
}

/// `%(refname)\0%(objectname)\0%(symref)` lines. Symbolic entries — a
/// remote's `HEAD` — are skipped: they name a branch, they are not one.
pub(crate) fn parse_remote_branches(out: &str) -> Result<Vec<RemoteBranch>, GitError> {
    let mut branches = Vec::new();
    for line in out.lines().filter(|l| !l.is_empty()) {
        let bad = || GitError::malformed("for-each-ref", line.to_string());
        let fields: Vec<&str> = line.split('\0').collect();
        let [refname, oid, symref] = fields[..] else {
            return Err(bad());
        };
        if !symref.is_empty() {
            continue;
        }
        let rest = refname.strip_prefix("refs/remotes/").ok_or_else(bad)?;
        let (remote, branch) = rest.split_once('/').ok_or_else(bad)?;
        branches.push(RemoteBranch {
            remote: RemoteName::parse(remote)?,
            branch: BranchName::parse(branch)?,
            oid: Oid::parse(oid)?,
        });
    }
    Ok(branches)
}

impl Repo {
    /// Every remote-tracking branch, as of the last fetch or push. Reads
    /// nothing from the network — [`Repo::ls_remote`] does.
    pub fn remote_branches(&self) -> Result<Vec<RemoteBranch>, GitError> {
        let out = self
            .git("for-each-ref")
            .arg("--format=%(refname)%00%(objectname)%00%(symref)")
            .arg("refs/remotes/")
            .read_only()
            .run_ok()?;
        parse_remote_branches(&utf8("for-each-ref", out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::fetch::FetchSpec;
    use crate::testing::TestRepo;

    #[test]
    fn lines_parse_skipping_symbolic_refs() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let out = format!(
            "refs/remotes/origin/HEAD\0{a}\0refs/remotes/origin/main\n\
             refs/remotes/origin/main\0{a}\0\n\
             refs/remotes/upstream/zen/work\0{a}\0\n"
        );
        let b = parse_remote_branches(&out).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].remote.as_str(), "origin");
        assert_eq!(b[0].branch.as_str(), "main");
        assert_eq!(b[1].remote.as_str(), "upstream");
        assert_eq!(b[1].branch.as_str(), "zen/work");
        assert_eq!(b[1].to_ref().as_str(), "refs/remotes/upstream/zen/work");
    }

    #[test]
    fn every_fetched_branch_of_every_remote_is_listed() {
        let origin = TestRepo::bare();
        let shared = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"a\n");
        let c = t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["remote", "add", "upstream", &shared.url()]);
        t.git(&["push", "-q", "origin", "main", "main:zen/work"]);
        t.git(&["push", "-q", "upstream", "main:trunk"]);
        let repo = t.repo();
        for r in ["origin", "upstream"] {
            repo.fetch(&RemoteName::parse(r).unwrap(), &FetchSpec::AllBranches)
                .unwrap();
        }
        let mut got: Vec<String> = repo
            .remote_branches()
            .unwrap()
            .into_iter()
            .map(|b| {
                assert_eq!(b.oid, c);
                b.to_ref().to_string()
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                "refs/remotes/origin/main",
                "refs/remotes/origin/zen/work",
                "refs/remotes/upstream/trunk"
            ]
        );
    }
}
