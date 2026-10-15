//! What a checkout keeps in `.git/config` that a repository read from the
//! substrate keeps in memory (`docs/substrate_git.md` §4.3): its remotes and
//! its branches' upstreams. Each remote fetches `refs/heads/*` into
//! `refs/remotes/<name>/*`, as `git remote add` sets it up.

use std::collections::BTreeMap;

/// A repository's remotes and upstreams.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Settings {
    /// Each remote's URL, by name.
    remotes: BTreeMap<String, String>,
    /// Each branch's upstream: the remote — `.` for a local branch — and the
    /// full ref it tracks there.
    upstreams: BTreeMap<String, (String, String)>,
}

impl Settings {
    /// Every remote, by name, with its URL.
    pub(crate) fn remotes(&self) -> &BTreeMap<String, String> {
        &self.remotes
    }

    /// Add or repoint remote `name`.
    pub(crate) fn set_remote(&mut self, name: &str, url: &str) {
        self.remotes.insert(name.to_string(), url.to_string());
    }

    /// Remove remote `name`; whether it was there.
    pub(crate) fn remove_remote(&mut self, name: &str) -> bool {
        self.remotes.remove(name).is_some()
    }

    /// What `branch` tracks: the remote and the full ref on it.
    pub(crate) fn upstream(&self, branch: &str) -> Option<&(String, String)> {
        self.upstreams.get(branch)
    }

    /// Make `branch` track `merge` on `remote`.
    pub(crate) fn set_upstream(&mut self, branch: &str, remote: &str, merge: &str) {
        self.upstreams
            .insert(branch.to_string(), (remote.to_string(), merge.to_string()));
    }

    /// Stop `branch` tracking anything.
    pub(crate) fn unset_upstream(&mut self, branch: &str) {
        self.upstreams.remove(branch);
    }

    /// Where `remote`'s fetch puts `merge`, a full branch ref:
    /// `refs/remotes/<remote>/<branch>`; `merge` itself for `.`.
    pub(crate) fn tracking_of(remote: &str, merge: &str) -> Option<String> {
        if remote == "." {
            return Some(merge.to_string());
        }
        let branch = merge.strip_prefix("refs/heads/")?;
        Some(format!("refs/remotes/{remote}/{branch}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstreams_and_remotes_set_and_clear() {
        let mut s = Settings::default();
        s.set_remote("origin", "agent:candle");
        s.set_upstream("main", "origin", "refs/heads/main");
        assert_eq!(s.remotes()["origin"], "agent:candle");
        assert_eq!(
            s.upstream("main"),
            Some(&("origin".to_string(), "refs/heads/main".to_string()))
        );
        s.unset_upstream("main");
        assert_eq!(s.upstream("main"), None);
        assert!(s.remove_remote("origin"));
        assert!(!s.remove_remote("origin"));
    }

    #[test]
    fn a_remotes_fetch_maps_branches_to_its_tracking_refs() {
        assert_eq!(
            Settings::tracking_of("origin", "refs/heads/zen/work").as_deref(),
            Some("refs/remotes/origin/zen/work")
        );
        assert_eq!(
            Settings::tracking_of(".", "refs/heads/main").as_deref(),
            Some("refs/heads/main")
        );
        assert_eq!(Settings::tracking_of("origin", "refs/tags/v1"), None);
    }
}
