//! The seed: the knowledge base's mandatory files
//! (`docs/placed_section_layout.md` §3.5b) — for each repository, in the
//! workspace's order, its root listing and its anchor documents (README,
//! ARCHITECTURE, AGENTS, CLAUDE) as its default branch holds them.
//!
//! They are ingested as every file is, each cut into sections and sealed
//! alone, and named here at startup as the files every free knowledge base the
//! seed dial leaves on places whole, inside the `files` range and outside the
//! competition. The range must hold them with room for the rest; a schema
//! whose range does not is refused.

use candle_conversation::files::{leaves_room, mandatory_records};
use candle_conversation::projection::{Builder, SystemPromptItem};
use candle_conversation::ConversationEngine;
use zend_vfs::vfs::Tree;
use zend_vfs::{GitError, Rev};

use super::{records, FILES_COLLECTION};
use crate::branch_ingest::walk::RepoBranches;
use crate::journal::opening::SeedRead;

/// Anchor documents, in seed order after a repository's root listing.
/// Matched case-insensitively against the repository's own root files —
/// never a subfolder's.
pub const ANCHOR_FILES: [&str; 4] = ["README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"];

/// The workspace paths of one repository's seed: its root listing, then each
/// anchor document its default branch holds at the root, matched
/// case-insensitively, in [`ANCHOR_FILES`] order. A repository with no branch
/// contributes nothing.
pub fn repo_seed(repo: &RepoBranches) -> Result<Vec<String>, GitError> {
    let Some(tip) = repo.tips.first() else {
        return Ok(Vec::new());
    };
    let Some((_, tree_id)) = repo.repo.blobs().commit_of(&Rev::Oid(tip.tip.clone()))? else {
        return Ok(Vec::new());
    };
    let tree = Tree::read(&repo.repo, &tree_id)?;
    let children = tree.children("");
    let mut out = vec![format!("{}/", repo.name)];
    for anchor in ANCHOR_FILES {
        let found = children
            .iter()
            .find(|(path, _, is_dir)| !is_dir && path.eq_ignore_ascii_case(anchor));
        if let Some((path, _, _)) = found {
            out.push(format!("{}/{path}", repo.name));
        }
    }
    Ok(out)
}

/// What the seed's `paths` read: each root listing by its repository, each
/// anchor by its path — what a dialogue's journal opens by saying it read.
pub fn seed_read(paths: &[String]) -> SeedRead {
    let mut read = SeedRead::default();
    for path in paths {
        match path.strip_suffix('/') {
            Some(repo) => read.listings.push(repo.to_string()),
            None => read.documents.push(path.clone()),
        }
    }
    read
}

/// The positions `builder`'s `files` collection reserves on an engine of
/// `native` context; `None` when it reserves none.
fn files_range(builder: &Builder, native: Option<usize>) -> Option<usize> {
    builder
        .schema()
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            SystemPromptItem::Collection(c) if c.name == FILES_COLLECTION => c.range,
            _ => None,
        })
        .map(|range| range.positions(native))
}

/// Name the seed of `repos` as `engine`'s mandatory files, returning what it
/// reads. A repository whose tree cannot be read is left out of the seed, and
/// said so. Refused when the mandatory files the substrate already holds take
/// more than half of `builder`'s `files` range.
pub fn install(
    engine: &ConversationEngine,
    builder: &Builder,
    repos: &[RepoBranches],
) -> anyhow::Result<SeedRead> {
    let mut paths = Vec::new();
    for repo in repos {
        match repo_seed(repo) {
            Ok(of_repo) => paths.extend(of_repo),
            Err(e) => tracing::warn!(
                repo = %repo.name,
                "the seed leaves this repository out: its tree could not be read: {e}"
            ),
        }
    }
    let conversation = engine.conversation();
    let native = conversation.read().trained_window();
    if let Some(positions) = files_range(builder, native) {
        let held = records::all(&conversation);
        let mandatory = mandatory_records(&held, &paths);
        anyhow::ensure!(
            leaves_room(&mandatory, positions),
            "the seed's {} files take {} tokens, more than half the knowledge base's \
             {positions}-position range",
            mandatory.len(),
            mandatory.iter().map(|r| r.tokens()).sum::<u32>(),
        );
    }
    engine.set_mandatory_files(paths.clone());
    tracing::info!(
        files = paths.len(),
        "seed installed: the repositories' roots, the knowledge base's mandatory files",
    );
    Ok(seed_read(&paths))
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use zend_vfs::Repo;

    use super::*;

    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
    }

    fn write(dir: &Path, rel: &str, text: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn repo() -> (tempfile::TempDir, RepoBranches) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        git(dir, &["init", "-q", "-b", "main"]);
        git(dir, &["config", "user.email", "t@t"]);
        git(dir, &["config", "user.name", "t"]);
        write(dir, "README.md", "# R\n");
        write(dir, "agents.md", "rules");
        write(dir, "src/lib.rs", "fn x() {}\n");
        write(dir, "docs/README.md", "not the root's\n");
        git(dir, &["add", "-A"]);
        git(dir, &["commit", "-q", "-m", "c"]);
        let branches = RepoBranches::read("r", Repo::open(dir).unwrap()).unwrap();
        (tmp, branches)
    }

    /// **The seed names a repository's root listing and its anchor
    /// documents**: the listing first, then README and the lower-case
    /// `agents.md` matched as AGENTS, a subfolder's README left out.
    #[test]
    fn the_seed_names_the_root_listing_and_its_anchor_documents() {
        let (_tmp, repo) = repo();
        let paths = repo_seed(&repo).unwrap();
        assert_eq!(paths, ["r/", "r/README.md", "r/agents.md"]);
        let read = seed_read(&paths);
        assert_eq!(read.listings, ["r"]);
        assert_eq!(read.documents, ["r/README.md", "r/agents.md"]);
    }

    /// **The files range is read off the schema** for the engine's context.
    #[test]
    fn the_files_range_is_read_off_the_schema() {
        let yaml = r#"
system_prompt:
  items:
    - kind: section
      id: open
      content: "open"
    - kind: collection
      name: files
      selection: { kind: always_visible }
      range: { tokens: 1000, small_context: 100 }
      sections: []
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget: { priority: 100 }
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: c, user_prompt: c }
        assistant: { system_prompt: c, user_prompt: c }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let b = Builder::from_yaml(yaml).unwrap();
        assert_eq!(files_range(&b, Some(262_144)), Some(1_000));
        assert_eq!(files_range(&b, Some(32_768)), Some(100));
    }
}
