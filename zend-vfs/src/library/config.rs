//! What the repository's configuration holds, from libgit2: its remotes and
//! the identity commits are made under.
//!
//! libgit2 reads the same files, in the same order, that `git config` does —
//! system, global, the repository's own — and the value that wins is the one
//! `--get` would have printed.

use std::collections::BTreeMap;

use git2::Repository;

use super::{failed, is_absent};
use crate::error::GitError;
use crate::read::refs::{remotes_from, Remote};

/// Every configured remote, by name.
pub(crate) fn remotes(lib: &Repository) -> Result<Vec<Remote>, GitError> {
    let config = lib.config().map_err(|e| failed("config", e))?;
    let mut fetch: BTreeMap<String, String> = BTreeMap::new();
    let mut push: BTreeMap<String, String> = BTreeMap::new();
    let mut entries = config
        .entries(Some(r"^remote\..*\.(url|pushurl)$"))
        .map_err(|e| failed("config", e))?;
    while let Some(entry) = entries.next() {
        let entry = entry.map_err(|e| failed("config", e))?;
        let (Some(key), Some(value)) = (entry.name(), entry.value()) else {
            continue;
        };
        let Some(rest) = key.strip_prefix("remote.") else {
            continue;
        };
        if let Some(name) = rest.strip_suffix(".pushurl") {
            push.insert(name.to_string(), value.to_string());
        } else if let Some(name) = rest.strip_suffix(".url") {
            fetch.insert(name.to_string(), value.to_string());
        }
    }
    remotes_from(fetch, push)
}

/// The configured `user.name` and `user.email`, either `None` when unset.
pub(crate) fn identity(lib: &Repository) -> Result<(Option<String>, Option<String>), GitError> {
    let config = lib.config().map_err(|e| failed("config", e))?;
    let get = |key: &str| match config.get_string(key) {
        Ok(value) => Ok(Some(value.trim().to_string())),
        Err(e) if is_absent(&e) => Ok(None),
        Err(e) => Err(failed("config", e)),
    };
    Ok((get("user.name")?, get("user.email")?))
}
