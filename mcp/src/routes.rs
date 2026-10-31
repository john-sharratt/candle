//! Which backend runs each tool, by name: learned from every listing and
//! kept across restarts, so a call is routed without listing first.
//!
//! A tool's backend does not depend on who lists it. A listing is filtered by
//! its caller's role, so what one caller's listing leaves out is not gone:
//! each listing adds to the routes, and the latest says where a name runs. A
//! name no backend lists any more stays routed to the one that did, which
//! answers that it has no such tool.
//!
//! The routes are kept in a file beside the node's, written whole and moved
//! into place on each change, so a restart routes at once what was routed
//! before it.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// The routes, and where they are kept.
pub struct Routes {
    by_name: RwLock<HashMap<String, usize>>,
    /// The file they are kept in, naming each backend by its name.
    file: Option<PathBuf>,
    /// The backends' names, in order: what a backend's index means.
    backends: Vec<String>,
}

impl Routes {
    /// Routes kept nowhere.
    pub fn in_memory(backends: Vec<String>) -> Self {
        Self {
            by_name: RwLock::new(HashMap::new()),
            file: None,
            backends,
        }
    }

    /// The routes kept in `file`, as last written, for `backends`: a backend
    /// the file names that is not served any more routes nothing, and a file
    /// that does not read is no routes at all.
    pub fn kept_in(file: &Path, backends: Vec<String>) -> Self {
        let by_name = fs::read(file)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<HashMap<String, String>>(&bytes).ok())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(tool, backend)| {
                let at = backends.iter().position(|b| *b == backend)?;
                Some((tool, at))
            })
            .collect();
        Self {
            by_name: RwLock::new(by_name),
            file: Some(file.to_path_buf()),
            backends,
        }
    }

    /// The backend that runs `tool`.
    pub fn get(&self, tool: &str) -> Option<usize> {
        self.by_name
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(tool)
            .copied()
    }

    /// Add what a listing found: each name to the backend that listed it.
    /// Kept in the file when anything changed — written under the lock, so
    /// two listings at once leave the later routes whole, never one half of
    /// each. Routes change only when a build's tools do, so it is rare.
    pub fn learn(&self, learned: HashMap<String, usize>) {
        let mut by_name = self.by_name.write().unwrap_or_else(|e| e.into_inner());
        let mut changed = false;
        for (tool, backend) in learned {
            if by_name.insert(tool, backend) != Some(backend) {
                changed = true;
            }
        }
        let Some(file) = self.file.as_ref().filter(|_| changed) else {
            return;
        };
        if let Err(e) = write_whole(file, &self.named(&by_name)) {
            tracing::warn!("mcp: the tools' routes could not be kept: {e}");
        }
    }

    fn named(&self, by_name: &HashMap<String, usize>) -> HashMap<String, String> {
        by_name
            .iter()
            .filter_map(|(tool, at)| Some((tool.clone(), self.backends.get(*at)?.clone())))
            .collect()
    }
}

/// `routes` written to `file` whole: to a file beside it, then moved over it,
/// so a write cut off leaves the last one.
fn write_whole(file: &Path, routes: &HashMap<String, String>) -> anyhow::Result<()> {
    let fresh = file.with_extension("json.new");
    fs::write(&fresh, serde_json::to_vec(routes)?)?;
    fs::rename(&fresh, file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backends() -> Vec<String> {
        vec!["coding".into(), "bot".into()]
    }

    /// **A listing adds to the routes**: what another caller's listing left
    /// out stays routed, and the latest listing says where a name runs.
    #[test]
    fn a_listing_adds_to_the_routes() {
        let r = Routes::in_memory(backends());
        r.learn(HashMap::from([
            ("repo_task_start".to_string(), 0),
            ("cluster_command".to_string(), 0),
            ("bot_list".to_string(), 1),
        ]));
        // A user's listing, which has no admin's tools.
        r.learn(HashMap::from([("repo_task_start".to_string(), 0)]));
        assert_eq!(
            r.get("cluster_command"),
            Some(0),
            "an admin's tool stays routed"
        );
        assert_eq!(r.get("bot_list"), Some(1));
        r.learn(HashMap::from([("bot_list".to_string(), 0)]));
        assert_eq!(r.get("bot_list"), Some(0), "the latest listing says");
        assert_eq!(r.get("nothing"), None);
    }

    /// **The routes outlive a restart**, by the backends' names: one the file
    /// names that is no longer served routes nothing, and a file that does
    /// not read is no routes.
    #[test]
    fn the_routes_outlive_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("routes.json");
        let r = Routes::kept_in(&file, backends());
        assert_eq!(r.get("bot_list"), None, "no file yet");
        r.learn(HashMap::from([
            ("repo_task_start".to_string(), 0),
            ("bot_list".to_string(), 1),
        ]));
        let again = Routes::kept_in(&file, backends());
        assert_eq!(again.get("repo_task_start"), Some(0));
        assert_eq!(again.get("bot_list"), Some(1));
        let reordered = Routes::kept_in(&file, vec!["bot".into(), "coding".into()]);
        assert_eq!(reordered.get("bot_list"), Some(0), "by name, not by place");
        let fewer = Routes::kept_in(&file, vec!["coding".into()]);
        assert_eq!(fewer.get("bot_list"), None);
        fs::write(&file, b"not json").unwrap();
        assert_eq!(
            Routes::kept_in(&file, backends()).get("repo_task_start"),
            None
        );
    }
}
