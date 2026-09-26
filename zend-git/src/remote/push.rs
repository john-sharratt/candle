//! Pushing to remote branches and tags, each change guarded by a lease.

use crate::classify::classify;
use crate::error::GitError;
use crate::runner::Context;
use crate::types::{BranchName, Oid, RefName, RemoteName, TagName};
use crate::Repo;

/// What the remote ref must hold for the push to go ahead. There is no
/// unconditional force: every push is a compare-and-swap on the remote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lease {
    /// The ref must hold this object.
    Expect(Oid),
    /// The ref must not exist.
    Absent,
}

/// The remote ref a push changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushTarget {
    Branch(BranchName),
    Tag(TagName),
}

impl PushTarget {
    pub fn to_ref(&self) -> RefName {
        match self {
            Self::Branch(b) => b.to_ref(),
            Self::Tag(t) => t.to_ref(),
        }
    }

    fn from_ref(name: &RefName) -> Option<Self> {
        name.branch()
            .map(Self::Branch)
            .or_else(|| TagName::from_ref(name).map(Self::Tag))
    }
}

/// What to do to the target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushAction {
    /// Set the remote ref to this object (a commit, or a tag object).
    Update(Oid),
    /// Delete the remote ref.
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushSpec {
    pub action: PushAction,
    pub target: PushTarget,
    pub lease: Lease,
}

impl PushSpec {
    /// Set `branch` on the remote to `source`.
    pub fn branch(source: Oid, branch: BranchName, lease: Lease) -> Self {
        Self {
            action: PushAction::Update(source),
            target: PushTarget::Branch(branch),
            lease,
        }
    }

    /// Set `tag` on the remote to `object`; the tag must not exist there yet.
    pub fn tag(object: Oid, tag: TagName) -> Self {
        Self {
            action: PushAction::Update(object),
            target: PushTarget::Tag(tag),
            lease: Lease::Absent,
        }
    }

    /// Delete `branch` on the remote, which must hold `expected`.
    pub fn delete_branch(branch: BranchName, expected: Oid) -> Self {
        Self {
            action: PushAction::Delete,
            target: PushTarget::Branch(branch),
            lease: Lease::Expect(expected),
        }
    }

    /// Delete `tag` on the remote, which must hold `expected`.
    pub fn delete_tag(tag: TagName, expected: Oid) -> Self {
        Self {
            action: PushAction::Delete,
            target: PushTarget::Tag(tag),
            lease: Lease::Expect(expected),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// The remote branch did not hold what the lease expected.
    Stale,
    /// Another branch in the same atomic push was rejected.
    AtomicAborted,
    /// The remote refused it (a protected branch, a server hook), with its
    /// reason.
    Remote(String),
    /// Any other rejection, with git's summary.
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    Created,
    FastForward,
    /// The branch moved to a commit that does not descend from what it held
    /// — allowed because the lease matched.
    Forced,
    Deleted,
    UpToDate,
    Rejected(Rejection),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushResult {
    pub target: PushTarget,
    pub outcome: PushOutcome,
}

impl PushResult {
    pub fn accepted(&self) -> bool {
        !matches!(self.outcome, PushOutcome::Rejected(_))
    }
}

/// `<flag>\t<from>:<to>\t<summary>` lines from `push --porcelain`.
pub(crate) fn parse_push(out: &str) -> Result<Vec<PushResult>, GitError> {
    let mut results = Vec::new();
    for line in out.lines() {
        let mut parts = line.splitn(3, '\t');
        let (Some(flag), Some(refs), Some(summary)) = (parts.next(), parts.next(), parts.next())
        else {
            continue; // `To <url>`, `Done`
        };
        let bad = || GitError::malformed("push", line.to_string());
        let to = refs.rsplit_once(':').ok_or_else(bad)?.1;
        let target = PushTarget::from_ref(&RefName::parse(to)?).ok_or_else(bad)?;
        let reason = || {
            summary
                .split_once('(')
                .map(|(_, r)| r.trim_end_matches(')').to_string())
                .unwrap_or_default()
        };
        let outcome = match flag {
            " " => PushOutcome::FastForward,
            "+" => PushOutcome::Forced,
            "*" => PushOutcome::Created,
            "-" => PushOutcome::Deleted,
            "=" => PushOutcome::UpToDate,
            "!" if summary.starts_with("[remote rejected]") => {
                PushOutcome::Rejected(Rejection::Remote(reason()))
            }
            "!" if summary.contains("(stale info)") => PushOutcome::Rejected(Rejection::Stale),
            "!" if summary.contains("(atomic push failed)") => {
                PushOutcome::Rejected(Rejection::AtomicAborted)
            }
            "!" => PushOutcome::Rejected(Rejection::Other(summary.to_string())),
            _ => return Err(bad()),
        };
        results.push(PushResult { target, outcome });
    }
    Ok(results)
}

impl Repo {
    /// Push every spec to `remote` in one atomic push: all refs change or
    /// none do. A rejection is a result, not an error — the error cases are
    /// the transport's (unreachable, authentication).
    ///
    /// Deleting with [`Lease::Absent`] is refused: there is nothing to delete.
    pub fn push(
        &self,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(spec) = specs
            .iter()
            .find(|s| s.action == PushAction::Delete && s.lease == Lease::Absent)
        {
            return Err(GitError::invalid(format!(
                "deleting {} with an absent lease deletes nothing",
                spec.target.to_ref()
            )));
        }
        self.require_remote(remote)?;
        let _write = self.write_lock();
        let mut inv = self.git("push").args([
            "--porcelain",
            "--atomic",
            "--no-verify",
            "--no-recurse-submodules",
        ]);
        for spec in specs {
            let expect = match &spec.lease {
                Lease::Expect(oid) => oid.to_string(),
                Lease::Absent => String::new(),
            };
            inv = inv.arg(format!(
                "--force-with-lease={}:{expect}",
                spec.target.to_ref()
            ));
        }
        inv = inv.arg("--end-of-options").arg(remote.as_str());
        for spec in specs {
            let source = match &spec.action {
                PushAction::Update(oid) => oid.to_string(),
                PushAction::Delete => String::new(),
            };
            inv = inv.arg(format!("{source}:{}", spec.target.to_ref()));
        }
        // Exit 1 carries per-branch rejections on stdout; anything else
        // non-zero is the transport's failure, classified from stderr.
        let out = inv.transfer(remote).run_accepting(&[0, 1])?;
        let text = String::from_utf8(out.stdout)
            .map_err(|e| GitError::malformed("push", e.to_string()))?;
        let results = parse_push(&text)?;
        if results.is_empty() && out.status != Some(0) {
            return Err(self.push_failed(remote, out.status, out.stderr));
        }
        Ok(results)
    }

    /// A push that exited 1 without a result per ref: git's own reason,
    /// classified — and so with any credential in the remote's URL redacted
    /// — like every other failure the runner sees.
    fn push_failed(&self, remote: &RemoteName, status: Option<i32>, stderr: String) -> GitError {
        let context = Context {
            remote: Some(remote.clone()),
            rev: None,
        };
        classify(self.dir(), &context, vec!["push".into()], status, stderr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::fetch::FetchSpec;
    use crate::testing::{git_in, TestRepo};

    #[test]
    fn porcelain_lines_parse_to_outcomes() {
        // Built with `concat!` rather than `\` continuations, which would
        // strip the leading space that is the fast-forward flag.
        let out = concat!(
            "To file:///x\n",
            "*\trefs/heads/a:refs/heads/new\t[new branch]\n",
            " \trefs/heads/a:refs/heads/ff\t1111111..2222222\n",
            "+\trefs/heads/a:refs/heads/forced\t1111111...2222222 (forced update)\n",
            "=\trefs/heads/a:refs/heads/same\t[up to date]\n",
            "!\trefs/heads/a:refs/heads/stale\t[rejected] (stale info)\n",
            "!\trefs/heads/a:refs/heads/atomic\t[rejected] (atomic push failed)\n",
            "!\trefs/heads/a:refs/heads/protected\t[remote rejected] (protected branch hook declined)\n",
            "-\t:refs/heads/gone\t[deleted]\n",
            "*\t4b825dc642cb6eb9a060e54bf8d69288fbee4904:refs/tags/v1\t[new tag]\n",
            "Done\n",
        );
        let r = parse_push(out).unwrap();
        let outcomes: Vec<(String, &PushOutcome)> = r
            .iter()
            .map(|p| (p.target.to_ref().to_string(), &p.outcome))
            .collect();
        let h = |s: &str| format!("refs/heads/{s}");
        assert_eq!(
            outcomes,
            vec![
                (h("new"), &PushOutcome::Created),
                (h("ff"), &PushOutcome::FastForward),
                (h("forced"), &PushOutcome::Forced),
                (h("same"), &PushOutcome::UpToDate),
                (h("stale"), &PushOutcome::Rejected(Rejection::Stale)),
                (
                    h("atomic"),
                    &PushOutcome::Rejected(Rejection::AtomicAborted)
                ),
                (
                    h("protected"),
                    &PushOutcome::Rejected(Rejection::Remote(
                        "protected branch hook declined".into()
                    ))
                ),
                (h("gone"), &PushOutcome::Deleted),
                ("refs/tags/v1".to_string(), &PushOutcome::Created),
            ]
        );
    }

    /// **A push that fails before reporting any ref carries no credential.**
    /// Git names the remote's URL in that error, token and all.
    #[test]
    fn a_push_failure_without_results_is_redacted() {
        let t = TestRepo::init();
        let e = t.repo().push_failed(
            &RemoteName::parse("origin").unwrap(),
            Some(1),
            "error: failed to push some refs to 'https://me:ghp_secret@github.com/x/y.git'\n"
                .into(),
        );
        let shown = format!("{e:?} {e}");
        assert!(!shown.contains("ghp_secret"), "{shown}");
        assert!(shown.contains("github.com/x/y.git"), "{shown}");
    }

    struct Setup {
        origin: TestRepo,
        t: TestRepo,
        first: Oid,
        second: Oid,
        remote: RemoteName,
    }

    fn setup() -> Setup {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"1\n");
        let first = t.commit_all("first");
        t.write("a", b"2\n");
        let second = t.commit_all("second");
        t.git(&["remote", "add", "origin", &origin.url()]);
        Setup {
            origin,
            t,
            first,
            second,
            remote: RemoteName::parse("origin").unwrap(),
        }
    }

    fn on_origin_ref(s: &Setup, name: &str) -> Option<Oid> {
        let out = s
            .origin
            .git(&["for-each-ref", "--format=%(objectname)", name]);
        let out = out.trim();
        (!out.is_empty()).then(|| Oid::parse(out).unwrap())
    }

    fn on_origin(s: &Setup, branch: &str) -> Option<Oid> {
        on_origin_ref(s, &format!("refs/heads/{branch}"))
    }

    fn spec(source: &Oid, branch: &str, lease: Lease) -> PushSpec {
        PushSpec::branch(source.clone(), BranchName::parse(branch).unwrap(), lease)
    }

    /// **A remote branch is deleted only when it holds what the caller
    /// expects** — a branch someone else moved survives the delete.
    #[test]
    fn a_remote_branch_is_deleted_under_a_lease() {
        let s = setup();
        let repo = s.t.repo();
        let work = BranchName::parse("zen/done").unwrap();
        repo.push(&s.remote, &[spec(&s.second, "zen/done", Lease::Absent)])
            .unwrap();

        let r = repo
            .push(
                &s.remote,
                &[PushSpec::delete_branch(work.clone(), s.first.clone())],
            )
            .unwrap();
        assert_eq!(r[0].outcome, PushOutcome::Rejected(Rejection::Stale));
        assert_eq!(on_origin(&s, "zen/done"), Some(s.second.clone()));

        let r = repo
            .push(
                &s.remote,
                &[PushSpec::delete_branch(work.clone(), s.second.clone())],
            )
            .unwrap();
        assert_eq!(
            r,
            vec![PushResult {
                target: PushTarget::Branch(work),
                outcome: PushOutcome::Deleted
            }]
        );
        assert_eq!(on_origin(&s, "zen/done"), None);
    }

    #[test]
    fn deleting_with_an_absent_lease_is_refused_before_any_push() {
        let s = setup();
        let spec = PushSpec {
            action: PushAction::Delete,
            target: PushTarget::Branch(BranchName::parse("x").unwrap()),
            lease: Lease::Absent,
        };
        assert!(matches!(
            s.t.repo().push(&s.remote, &[spec]),
            Err(GitError::InvalidInput(_))
        ));
    }

    /// Tags push once — a second push of a different object to the same tag
    /// is refused — and delete under a lease.
    #[test]
    fn tags_push_once_and_delete_under_a_lease() {
        let s = setup();
        let repo = s.t.repo();
        let v1 = TagName::parse("v1").unwrap();
        let r = repo
            .push(&s.remote, &[PushSpec::tag(s.first.clone(), v1.clone())])
            .unwrap();
        assert_eq!(r[0].target, PushTarget::Tag(v1.clone()));
        assert_eq!(r[0].outcome, PushOutcome::Created);
        assert_eq!(on_origin_ref(&s, "refs/tags/v1"), Some(s.first.clone()));

        let r = repo
            .push(&s.remote, &[PushSpec::tag(s.second.clone(), v1.clone())])
            .unwrap();
        assert!(!r[0].accepted());
        assert_eq!(on_origin_ref(&s, "refs/tags/v1"), Some(s.first.clone()));

        let r = repo
            .push(&s.remote, &[PushSpec::delete_tag(v1, s.first.clone())])
            .unwrap();
        assert_eq!(r[0].outcome, PushOutcome::Deleted);
        assert_eq!(on_origin_ref(&s, "refs/tags/v1"), None);
    }

    #[test]
    fn a_new_branch_pushes_with_an_absent_lease_and_moves_with_an_expected_one() {
        let s = setup();
        let repo = s.t.repo();
        let r = repo
            .push(&s.remote, &[spec(&s.first, "zen/work", Lease::Absent)])
            .unwrap();
        assert_eq!(r[0].outcome, PushOutcome::Created);
        assert_eq!(on_origin(&s, "zen/work"), Some(s.first.clone()));

        let r = repo
            .push(
                &s.remote,
                &[spec(&s.second, "zen/work", Lease::Expect(s.first.clone()))],
            )
            .unwrap();
        assert_eq!(r[0].outcome, PushOutcome::FastForward);
        assert_eq!(on_origin(&s, "zen/work"), Some(s.second.clone()));

        // A matching lease also allows a non-fast-forward.
        let r = repo
            .push(
                &s.remote,
                &[spec(&s.first, "zen/work", Lease::Expect(s.second.clone()))],
            )
            .unwrap();
        assert_eq!(r[0].outcome, PushOutcome::Forced);
    }

    /// **A lease that does not match refuses the push**: someone else moved
    /// the branch, and their commit is kept.
    #[test]
    fn a_stale_lease_is_rejected_and_the_remote_is_unchanged() {
        let s = setup();
        let repo = s.t.repo();
        repo.push(&s.remote, &[spec(&s.second, "zen/work", Lease::Absent)])
            .unwrap();

        let r = repo
            .push(&s.remote, &[spec(&s.first, "zen/work", Lease::Absent)])
            .unwrap();
        assert_eq!(r[0].outcome, PushOutcome::Rejected(Rejection::Stale));
        let r = repo
            .push(
                &s.remote,
                &[spec(&s.first, "zen/work", Lease::Expect(s.first.clone()))],
            )
            .unwrap();
        assert_eq!(r[0].outcome, PushOutcome::Rejected(Rejection::Stale));
        assert_eq!(on_origin(&s, "zen/work"), Some(s.second.clone()));
    }

    /// **Atomic**: one rejected branch leaves every branch in the push
    /// unchanged on the remote.
    #[test]
    fn one_rejection_aborts_the_whole_push() {
        let s = setup();
        let repo = s.t.repo();
        repo.push(&s.remote, &[spec(&s.second, "taken", Lease::Absent)])
            .unwrap();
        let r = repo
            .push(
                &s.remote,
                &[
                    spec(&s.first, "fresh", Lease::Absent),
                    spec(&s.first, "taken", Lease::Absent),
                ],
            )
            .unwrap();
        assert!(r.iter().all(|p| !p.accepted()), "{r:?}");
        assert_eq!(on_origin(&s, "fresh"), None);
        assert_eq!(on_origin(&s, "taken"), Some(s.second.clone()));
    }

    /// **The user's hooks never run.** A `pre-push` and a
    /// `reference-transaction` hook that would leave marker files do not.
    #[test]
    fn hooks_in_the_repository_do_not_run() {
        let s = setup();
        let hooks = s.t.path.join(".git/hooks");
        for hook in ["pre-push", "reference-transaction"] {
            let marker = s.t.path.join(format!("{hook}.ran"));
            let script = format!(
                "#!/bin/sh\ntouch '{}'\n",
                marker.to_string_lossy().replace('\\', "/")
            );
            std::fs::write(hooks.join(hook), script).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(hooks.join(hook), std::fs::Permissions::from_mode(0o755))
                    .unwrap();
            }
        }
        // The hooks are live for ordinary git in this repository: a plain
        // push runs `pre-push`, which every supported git has.
        // (`reference-transaction` only exists from 2.28.)
        s.t.git(&[
            "-c",
            "core.hooksPath=.git/hooks",
            "push",
            "-q",
            "origin",
            &format!("{}:refs/heads/probe", s.first),
        ]);
        assert!(s.t.path.join("pre-push.ran").exists(), "the probe hook ran");
        std::fs::remove_file(s.t.path.join("pre-push.ran")).unwrap();
        let _ = std::fs::remove_file(s.t.path.join("reference-transaction.ran"));

        let repo = s.t.repo();
        repo.create_branch(&BranchName::parse("local").unwrap(), &s.first)
            .unwrap();
        repo.push(&s.remote, &[spec(&s.first, "zen/work", Lease::Absent)])
            .unwrap();
        assert!(!s.t.path.join("pre-push.ran").exists());
        assert!(!s.t.path.join("reference-transaction.ran").exists());
    }

    /// **An unconfigured remote name is refused, never taken as a path.**
    /// A repository sits at `<repo>/elsewhere`; git would push there if
    /// asked to push to "elsewhere".
    #[test]
    fn an_unconfigured_remote_name_is_refused_everywhere() {
        let s = setup();
        let decoy = s.t.path.join("elsewhere");
        std::fs::create_dir_all(&decoy).unwrap();
        git_in(&decoy, &["init", "-q", "--bare"]);
        let repo = s.t.repo();
        let elsewhere = RemoteName::parse("elsewhere").unwrap();
        assert!(matches!(
            repo.push(&elsewhere, &[spec(&s.first, "x", Lease::Absent)]),
            Err(GitError::UnknownRemote { .. })
        ));
        assert!(matches!(
            repo.fetch(&elsewhere, &FetchSpec::AllBranches),
            Err(GitError::UnknownRemote { .. })
        ));
        assert!(matches!(
            repo.ls_remote(&elsewhere),
            Err(GitError::UnknownRemote { .. })
        ));
        let pushed = git_in(&decoy, &["for-each-ref"]);
        assert!(pushed.is_empty(), "nothing reached the decoy: {pushed}");
    }

    #[test]
    fn an_unreachable_remote_is_an_error_not_a_rejection() {
        let s = setup();
        s.t.git(&[
            "remote",
            "set-url",
            "origin",
            s.t.path.join("missing").to_str().unwrap(),
        ]);
        let e =
            s.t.repo()
                .push(&s.remote, &[spec(&s.first, "x", Lease::Absent)])
                .unwrap_err();
        assert!(matches!(e, GitError::RemoteUnreachable { .. }), "{e}");
    }
}
