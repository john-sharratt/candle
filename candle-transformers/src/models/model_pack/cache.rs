//! Where model packs live.

use std::path::PathBuf;

/// The model cache: `~/.cache/zend/models`, with each repo's packs (and any
/// prepared artifact built from it) under `<repo-with-dashes>/`.
///
/// One definition for every reader and writer — the daemon, the gates, the
/// prepare step — so a pack built by one is where the others look.
/// `USERPROFILE` first, then `HOME`.
pub fn cache_root() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache")
        .join("zend")
        .join("models")
}
