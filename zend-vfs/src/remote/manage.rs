//! Adding, removing and re-pointing remotes: a checkout's in its config, a
//! repository read from the substrate's in memory (`docs/substrate_git.md`
//! §4.3).

use crate::error::GitError;
use crate::types::{RemoteName, RemoteUrl};
use crate::write::ref_txn::{RefOp, RefTransaction};
use crate::Repo;

/// Which of a remote's URLs to set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrlKind {
    /// The URL fetches use — and pushes too, unless a push URL is set.
    Fetch,
    /// A separate URL for pushes only.
    Push,
}

impl Repo {
    /// Add remote `name` at `url`, fetching every branch into
    /// `refs/remotes/<name>/`. Nothing is fetched now.
    pub fn add_remote(&self, name: &RemoteName, url: &RemoteUrl) -> Result<(), GitError> {
        let _write = self.write_lock();
        if self.require_remote(name).is_ok() {
            return Err(GitError::RemoteExists {
                remote: name.clone(),
            });
        }
        if let Some(db) = self.objects() {
            db.settings().set_remote(name.as_str(), url.as_str());
            return Ok(());
        }
        // `remote` subcommands take no `--end-of-options` before 2.30. Neither
        // value can be read as a flag: a `RemoteName` and a `RemoteUrl` both
        // refuse a leading `-`.
        self.git("remote")
            .arg("add")
            .args([name.as_str(), url.as_str()])
            .run_ok()?;
        Ok(())
    }

    /// Remove remote `name`, its config and its remote-tracking branches.
    /// Branches that tracked it keep their commits and lose their upstream.
    pub fn remove_remote(&self, name: &RemoteName) -> Result<(), GitError> {
        let _write = self.write_lock();
        self.require_remote(name)?;
        if let Some(db) = self.objects() {
            let tracking = self.refs_under(&format!("refs/remotes/{name}/"))?;
            let txn = tracking
                .into_iter()
                .fold(RefTransaction::new(), |txn, (name, old)| {
                    txn.push(RefOp::Delete { name, old })
                });
            if !txn.ops().is_empty() {
                self.update_refs_locked(&txn)?;
            }
            db.settings().remove_remote(name.as_str());
            return Ok(());
        }
        self.git("remote")
            .arg("remove")
            .arg(name.as_str())
            .run_ok()?;
        Ok(())
    }

    /// Point remote `name`'s fetch or push URL at `url`.
    pub fn set_remote_url(
        &self,
        name: &RemoteName,
        url: &RemoteUrl,
        kind: UrlKind,
    ) -> Result<(), GitError> {
        let _write = self.write_lock();
        self.require_remote(name)?;
        if let Some(db) = self.objects() {
            return match kind {
                UrlKind::Fetch => {
                    db.settings().set_remote(name.as_str(), url.as_str());
                    Ok(())
                }
                // Its pushes go where its pusher sends them, an agent.
                UrlKind::Push => Err(GitError::invalid(
                    "a repository read from the substrate has one URL per remote; its pushes \
                     are made through an agent",
                )),
            };
        }
        // `set-url` refuses a push URL that is not configured yet, so a
        // first push URL is written as config.
        let key = match kind {
            UrlKind::Fetch => format!("remote.{name}.url"),
            UrlKind::Push => format!("remote.{name}.pushurl"),
        };
        self.git("config")
            .arg("--end-of-options")
            .args([key.as_str(), url.as_str()])
            .run_ok()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::refs::Remote;
    use crate::remote::fetch::FetchSpec;
    use crate::testing::TestRepo;
    use crate::types::BranchName;

    fn url(s: &str) -> RemoteUrl {
        RemoteUrl::parse(s).unwrap()
    }

    #[test]
    fn remotes_are_added_repointed_and_removed() {
        let t = TestRepo::init();
        let repo = t.repo();
        let up = RemoteName::parse("upstream").unwrap();
        repo.add_remote(&up, &url("git@example.com:a/b.git"))
            .unwrap();
        assert!(matches!(
            repo.add_remote(&up, &url("git@example.com:c/d.git")),
            Err(GitError::RemoteExists { .. })
        ));

        repo.set_remote_url(&up, &url("git@example.com:x/y.git"), UrlKind::Fetch)
            .unwrap();
        repo.set_remote_url(&up, &url("ssh://push.example.com/x/y.git"), UrlKind::Push)
            .unwrap();
        assert_eq!(
            repo.remotes().unwrap(),
            vec![Remote {
                name: up.clone(),
                fetch_url: "git@example.com:x/y.git".into(),
                push_url: Some("ssh://push.example.com/x/y.git".into()),
            }]
        );
        assert_eq!(
            t.git(&["config", "remote.upstream.fetch"]).trim(),
            "+refs/heads/*:refs/remotes/upstream/*"
        );

        repo.remove_remote(&up).unwrap();
        assert!(repo.remotes().unwrap().is_empty());
        let missing = RemoteName::parse("nope").unwrap();
        assert!(matches!(
            repo.remove_remote(&missing),
            Err(GitError::UnknownRemote { .. })
        ));
        assert!(matches!(
            repo.set_remote_url(&missing, &url("x"), UrlKind::Fetch),
            Err(GitError::UnknownRemote { .. })
        ));
    }

    #[test]
    fn removing_a_remote_removes_its_tracking_branches() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"a\n");
        t.commit_all("base");
        t.git(&["push", "-q", &origin.url(), "main"]);
        let repo = t.repo();
        let o = RemoteName::parse("origin").unwrap();
        repo.add_remote(&o, &url(&origin.url())).unwrap();
        repo.fetch(&o, &FetchSpec::AllBranches).unwrap();
        assert_eq!(repo.remote_branches().unwrap().len(), 1);
        repo.remove_remote(&o).unwrap();
        assert!(repo.remote_branches().unwrap().is_empty());
        assert!(repo
            .ref_target(&o.tracking(&BranchName::parse("main").unwrap()))
            .unwrap()
            .is_none());
    }

    /// The `ext::` transport is refused at the type, and disabled in config
    /// for a URL that reached git some other way.
    #[test]
    fn the_command_transport_never_runs() {
        let t = TestRepo::init();
        let marker = t.path.join("pwned");
        let evil = format!(
            "ext::sh -c touch% {}",
            marker.to_string_lossy().replace('\\', "/")
        );
        assert!(RemoteUrl::parse(&evil).is_err());
        t.git(&["remote", "add", "evil", &evil]);
        let e = t
            .repo()
            .ls_remote(&RemoteName::parse("evil").unwrap())
            .unwrap_err();
        assert!(!marker.exists(), "the ext transport ran: {e}");
    }
}
