//! Fetching into remote-tracking refs, and nothing else.

use std::collections::BTreeMap;

use crate::error::GitError;
use crate::library::refs;
use crate::runner::utf8;
use crate::types::{BranchName, Oid, RefName, RemoteName, Rev};
use crate::Repo;

/// What to fetch. Every form writes only under `refs/remotes/<remote>/`, so
/// a fetch can never move a local branch, and no tags are fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchSpec {
    /// One branch, into its remote-tracking ref.
    Branch(BranchName),
    /// Every branch, into remote-tracking refs, pruning ones deleted
    /// on the remote.
    AllBranches,
}

impl FetchSpec {
    fn refspec(&self, remote: &RemoteName) -> String {
        match self {
            Self::Branch(branch) => {
                format!("+{}:{}", branch.to_ref(), remote.tracking(branch))
            }
            Self::AllBranches => format!("+refs/heads/*:refs/remotes/{remote}/*"),
        }
    }

    /// The refs this fetch may change: one tracking ref, or the remote's
    /// whole tracking namespace.
    fn scope(&self, remote: &RemoteName) -> String {
        match self {
            Self::Branch(branch) => remote.tracking(branch).to_string(),
            Self::AllBranches => format!("refs/remotes/{remote}/"),
        }
    }
}

/// How a fetch changed one remote-tracking ref.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchFlag {
    New,
    FastForward,
    /// Moved to a commit that does not descend from the old one — the
    /// remote branch was rewritten.
    Forced,
    /// Deleted on the remote, and so removed here.
    Pruned,
}

/// One remote-tracking ref a fetch changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefUpdate {
    pub flag: FetchFlag,
    pub old: Option<Oid>,
    pub new: Option<Oid>,
    pub local: RefName,
}

/// `%(refname)\0%(objectname)\0%(symref)` lines, symbolic refs skipped.
fn parse_snapshot(out: &str) -> Result<BTreeMap<RefName, Oid>, GitError> {
    let mut refs = BTreeMap::new();
    for line in out.lines().filter(|l| !l.is_empty()) {
        let fields: Vec<&str> = line.split('\0').collect();
        let [name, oid, symref] = fields[..] else {
            return Err(GitError::malformed("for-each-ref", line.to_string()));
        };
        if symref.is_empty() {
            refs.insert(RefName::parse(name)?, Oid::parse(oid)?);
        }
    }
    Ok(refs)
}

impl Repo {
    /// The refs under `scope` — a ref name or a namespace ending in `/`.
    fn snapshot(&self, scope: &str) -> Result<BTreeMap<RefName, Oid>, GitError> {
        if let Some(lib) = self.library() {
            return refs::snapshot(&lib, scope);
        }
        let out = self
            .git("for-each-ref")
            .arg("--format=%(refname)%00%(objectname)%00%(symref)")
            .arg(scope)
            .read_only()
            .run_ok()?;
        parse_snapshot(&utf8("for-each-ref", out)?)
    }

    /// Fetch `spec` from `remote` and report what changed.
    ///
    /// The changes are read from the tracking refs themselves — snapshotted
    /// before and after — rather than from git's report of them, which only
    /// newer releases print in a machine-readable form.
    pub fn fetch(&self, remote: &RemoteName, spec: &FetchSpec) -> Result<Vec<RefUpdate>, GitError> {
        self.require_remote(remote)?;
        let _write = self.write_lock();
        let scope = spec.scope(remote);
        let before = self.snapshot(&scope)?;
        let mut inv = self
            .git("fetch")
            .args(["--quiet", "--no-tags", "--no-recurse-submodules"]);
        match spec {
            FetchSpec::AllBranches => inv = inv.arg("--prune"),
            // A branch the remote does not have is an unknown revision.
            FetchSpec::Branch(branch) => inv = inv.about_rev(branch.to_ref().to_string()),
        }
        inv.arg("--end-of-options")
            .arg(remote.as_str())
            .arg(spec.refspec(remote))
            .transfer(remote)
            .run_ok()?;
        let after = self.snapshot(&scope)?;

        let mut updates = Vec::new();
        for (name, new) in &after {
            let flag = match before.get(name) {
                None => FetchFlag::New,
                Some(old) if old == new => continue,
                Some(old) => {
                    if self.is_ancestor(&Rev::Oid(old.clone()), &Rev::Oid(new.clone()))? {
                        FetchFlag::FastForward
                    } else {
                        FetchFlag::Forced
                    }
                }
            };
            updates.push(RefUpdate {
                flag,
                old: before.get(name).cloned(),
                new: Some(new.clone()),
                local: name.clone(),
            });
        }
        for (name, old) in &before {
            if !after.contains_key(name) {
                updates.push(RefUpdate {
                    flag: FetchFlag::Pruned,
                    old: Some(old.clone()),
                    new: None,
                    local: name.clone(),
                });
            }
        }
        Ok(updates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    #[test]
    fn snapshot_lines_parse_skipping_symbolic_refs() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let out = format!(
            "refs/remotes/origin/HEAD\0{a}\0refs/remotes/origin/main\nrefs/remotes/origin/main\0{a}\0\n"
        );
        let refs = parse_snapshot(&out).unwrap();
        assert_eq!(refs.len(), 1);
        assert!(refs.contains_key(&RefName::parse("refs/remotes/origin/main").unwrap()));
    }

    #[test]
    fn refspecs_only_target_remote_tracking_refs() {
        let origin = RemoteName::parse("origin").unwrap();
        assert_eq!(
            FetchSpec::Branch(BranchName::parse("main").unwrap()).refspec(&origin),
            "+refs/heads/main:refs/remotes/origin/main"
        );
        assert_eq!(
            FetchSpec::AllBranches.refspec(&origin),
            "+refs/heads/*:refs/remotes/origin/*"
        );
    }

    #[test]
    fn fetch_reports_new_fast_forward_forced_and_pruned_and_never_moves_a_local_branch() {
        let origin = TestRepo::bare();
        let upstream = TestRepo::init();
        upstream.write("a", b"1\n");
        let first = upstream.commit_all("first");
        upstream.git(&["remote", "add", "origin", &origin.url()]);
        upstream.git(&[
            "push",
            "-q",
            "origin",
            "main",
            "main:gone",
            "main:rewritten",
        ]);

        let t = TestRepo::init();
        t.git(&["remote", "add", "origin", &origin.url()]);
        let repo = t.repo();
        let o = RemoteName::parse("origin").unwrap();
        let updates = repo.fetch(&o, &FetchSpec::AllBranches).unwrap();
        let mut names: Vec<&str> = updates.iter().map(|u| u.local.as_str()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "refs/remotes/origin/gone",
                "refs/remotes/origin/main",
                "refs/remotes/origin/rewritten"
            ]
        );
        assert!(updates.iter().all(|u| u.flag == FetchFlag::New));

        upstream.write("a", b"2\n");
        let second = upstream.commit_all("second");
        upstream.git(&["checkout", "-q", "--orphan", "fresh"]);
        upstream.write("a", b"unrelated\n");
        let unrelated = upstream.commit_all("unrelated");
        upstream.git(&[
            "push",
            "-q",
            "-f",
            "origin",
            "main",
            ":gone",
            "fresh:rewritten",
        ]);
        let updates = repo.fetch(&o, &FetchSpec::AllBranches).unwrap();
        let find = |name: &str| {
            updates
                .iter()
                .find(|u| u.local.as_str() == name)
                .cloned()
                .unwrap_or_else(|| panic!("{name} missing from {updates:?}"))
        };
        let main = find("refs/remotes/origin/main");
        assert_eq!(
            (main.flag, main.old, main.new),
            (
                FetchFlag::FastForward,
                Some(first.clone()),
                Some(second.clone())
            )
        );
        let rewritten = find("refs/remotes/origin/rewritten");
        assert_eq!(
            (rewritten.flag, rewritten.new),
            (FetchFlag::Forced, Some(unrelated))
        );
        let gone = find("refs/remotes/origin/gone");
        assert_eq!(
            (gone.flag, gone.old, gone.new),
            (FetchFlag::Pruned, Some(first), None)
        );

        // No local branch appeared or moved.
        assert_eq!(repo.branches().unwrap(), vec![]);

        // A branch the remote does not have is an unknown revision.
        let absent = BranchName::parse("absent").unwrap();
        assert!(matches!(
            repo.fetch(&o, &FetchSpec::Branch(absent)),
            Err(GitError::UnknownRevision { .. })
        ));

        // Nothing changed: nothing reported. One branch: only that ref.
        let main_branch = BranchName::parse("main").unwrap();
        assert!(repo
            .fetch(&o, &FetchSpec::Branch(main_branch.clone()))
            .unwrap()
            .is_empty());
        assert_eq!(
            repo.ref_target(&o.tracking(&main_branch)).unwrap(),
            Some(second)
        );
    }
}
