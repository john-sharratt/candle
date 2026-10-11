//! The repositories a zend member advertises in its `HELLO`
//! (`docs/replicated_substrate.md` §10.1): every git repository of the
//! workspace manifest, with its origin's URL. `Sticky` and `Best` consider
//! only members that hold every repository a conversation names. A member
//! pushes through agents (§11.5), so it claims no push rights of its own.

use cluster::frame::RepoAd;
use zend_vfs::{Repo, Workspace, ORIGIN};

pub fn repo_ads(workspace: &Workspace) -> Vec<RepoAd> {
    workspace
        .repos()
        .iter()
        .filter(|r| r.dir.join(".git").exists())
        .map(|r| RepoAd {
            name: r.name.clone(),
            origin: Repo::open(&r.dir)
                .and_then(|repo| repo.remotes())
                .ok()
                .and_then(|remotes| remotes.into_iter().find(|x| x.name.as_str() == ORIGIN))
                .map(|x| x.fetch_url)
                .unwrap_or_default(),
            push: false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use zend_vfs::RepoSpec;

    use super::*;

    fn git(dir: &std::path::Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
    }

    /// A git repository is advertised with its origin; a plain folder is not.
    #[test]
    fn each_git_repository_is_advertised_with_its_origin() {
        let dir = tempfile::tempdir().unwrap();
        let (app, plain) = (dir.path().join("app"), dir.path().join("plain"));
        std::fs::create_dir_all(&app).unwrap();
        std::fs::create_dir_all(&plain).unwrap();
        git(&app, &["init", "-q"]);
        git(
            &app,
            &["remote", "add", "origin", "https://example.com/app.git"],
        );
        let ws = Workspace::new(
            dir.path(),
            vec![RepoSpec::named("app"), RepoSpec::named("plain")],
        )
        .unwrap();
        assert_eq!(
            repo_ads(&ws),
            vec![RepoAd {
                name: "app".into(),
                origin: "https://example.com/app.git".into(),
                push: false,
            }]
        );
    }
}
