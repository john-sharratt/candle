//! A task a conversation of Zen's starts begins on that conversation's work
//! (`docs/zen_coding_challenge_fixes.md` Cause 5): on its branches, with its
//! uncommitted changes. A reviewer or a test-writer reads the change as it
//! stands, without a commit first.
//!
//! The starter is named on the task's conversation before its first turn
//! ([`mark`]). When that conversation is first built, its fresh stores are put
//! on the starter's branches and given the starter's changes as its last
//! finished tool round saved them, saved as the task's own file events, and
//! the mark is cleared ([`take`]). From then on the task's changes are its
//! own: nothing it does reaches the starter's stores, and nothing the starter
//! does later reaches its.

use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use candle_conversation::Result as EngineResult;
use zend_vfs::RepoFiles;

use crate::conv_overlay::{self, Mirror};

/// The metadata key naming the conversation a task's work begins from; empty
/// once taken.
pub const FROM_KEY: &str = "files_from";

/// Name `starter` as the conversation `timeline` begins on.
pub fn mark(
    engine: &ConversationEngine,
    timeline: TimelineId,
    starter: &str,
) -> EngineResult<()> {
    engine.set_conversation_metadata(timeline, FROM_KEY, starter)
}

/// The conversation `timeline` is still to take its work from; `None` once
/// taken, or for one no conversation started.
pub fn starter(engine: &ConversationEngine, timeline: TimelineId) -> Option<String> {
    engine
        .conversation_metadata(timeline)?
        .get(FROM_KEY)
        .filter(|from| !from.is_empty())
        .cloned()
}

/// Put `files`, a fresh set, on `theirs`' branches with `theirs`' changes.
/// A repository whose branch or changes are refused is logged and left as it
/// was. Returns how many changed files it laid down.
pub fn lay(files: &RepoFiles, theirs: &RepoFiles) -> usize {
    for (repo, why) in files.set_branches(&theirs.branches()) {
        tracing::warn!(%repo, "a task could not take its starter's branch: {why}");
    }
    let changes = theirs.snapshots();
    let counts: Vec<(String, usize)> = changes
        .iter()
        .map(|(repo, s)| (repo.clone(), s.chains().len()))
        .collect();
    let refused = files.restore(changes);
    for (repo, why) in &refused {
        tracing::warn!(%repo, "a task could not take its starter's changes: {why}");
    }
    counts
        .iter()
        .filter(|(repo, _)| !refused.iter().any(|(r, _)| r == repo))
        .map(|(_, n)| n)
        .sum()
}

/// [`lay`] `theirs` into `files`, `timeline`'s stores as its first build
/// restored them, save them as `timeline`'s file events and branches, and
/// clear its mark. Returns how many changed files it took.
pub fn take(
    engine: &ConversationEngine,
    timeline: TimelineId,
    files: &RepoFiles,
    mirror: &Mutex<Mirror>,
    theirs: &RepoFiles,
) -> usize {
    let took = lay(files, theirs);
    conv_overlay::save(engine, timeline, files, mirror);
    if let Err(e) = engine.set_conversation_metadata(timeline, FROM_KEY, "") {
        tracing::warn!("a task's starter mark could not be cleared: {e}");
    }
    took
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use zend_vfs::{RepoSpec, Workspace};

    use super::*;

    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "core.hooksPath=",
                "-c",
                "user.name=T",
                "-c",
                "user.email=t@x",
            ])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// **A task's stores take its starter's branch and changes**: a fresh set
    /// is put on the branch the starter works on and reads the file the
    /// starter changed as the starter left it, and a file the starter never
    /// touched as the branch holds it.
    #[test]
    fn a_task_begins_on_its_starters_branch_and_changes() {
        let root = tempfile::tempdir().unwrap();
        let code = root.path().join("code");
        std::fs::create_dir_all(&code).unwrap();
        git(&code, &["init", "-q", "-b", "main"]);
        std::fs::write(code.join("a.txt"), "one\n").unwrap();
        std::fs::write(code.join("b.txt"), "two\n").unwrap();
        git(&code, &["add", "a.txt", "b.txt"]);
        git(&code, &["commit", "-q", "-m", "first"]);
        git(&code, &["branch", "zen/work"]);
        let ws = Workspace::new(root.path(), vec![RepoSpec::named("code")]).unwrap();
        let theirs = RepoFiles::overlay(ws).unwrap();
        let on = [("code".to_string(), "zen/work".to_string())].into();
        assert!(theirs.set_branches(&on).is_empty());
        theirs
            .repo("code")
            .unwrap()
            .write("a.txt", "changed\n".into())
            .unwrap();

        let files = theirs.fresh();
        assert_eq!(lay(&files, &theirs), 1);
        assert_eq!(files.branches(), on);
        let store = files.repo("code").unwrap();
        assert_eq!(store.read("a.txt").unwrap().as_deref(), Some("changed\n"));
        assert_eq!(store.read("b.txt").unwrap().as_deref(), Some("two\n"));
    }

    /// **A starter with no changes gives its branch alone**, and lays down
    /// nothing.
    #[test]
    fn a_starter_with_no_changes_gives_its_branch() {
        let root = tempfile::tempdir().unwrap();
        let code = root.path().join("code");
        std::fs::create_dir_all(&code).unwrap();
        git(&code, &["init", "-q", "-b", "main"]);
        git(&code, &["commit", "-q", "--allow-empty", "-m", "first"]);
        git(&code, &["branch", "zen/work"]);
        let ws = Workspace::new(root.path(), vec![RepoSpec::named("code")]).unwrap();
        let theirs = RepoFiles::overlay(ws).unwrap();
        let on = [("code".to_string(), "zen/work".to_string())].into();
        assert!(theirs.set_branches(&on).is_empty());
        let files = theirs.fresh();
        assert_eq!(lay(&files, &theirs), 0);
        assert_eq!(files.branches(), on);
    }
}
