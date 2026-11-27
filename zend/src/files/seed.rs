//! The seed every conversation opens on (`docs/provenance_system_collection.md`
//! §6.2b): for each repository, in the workspace's order, its root listing
//! and its anchor documents — README, ARCHITECTURE, AGENTS, CLAUDE — as its
//! default branch holds them, in the knowledge base's notation.
//!
//! It is one section of the prompt, the `seed` optional's present side,
//! rendered here at startup and installed into the schema before any
//! conversation is built, as the tool catalog is. Sealed behind `files_open`
//! and read by the branch checkpoint's recurrence, it is the part of the
//! knowledge base every conversation's recurrent state has read.

use candle_conversation::files::{assemble_pieces, FileUnit, RenderedFile};
use candle_conversation::projection::Builder;
use zend_vfs::vfs::Tree;
use zend_vfs::Rev;

use super::listing::listing_text;
use super::sections::sections;
use crate::branch_ingest::walk::RepoBranches;
use crate::journal::opening::SeedRead;

/// The tree node whose present side the seed is installed into.
pub const SEED_NODE: &str = "seed";

/// Anchor documents, in seed order after a repository's root listing.
/// Matched case-insensitively against the repository's own root files —
/// never a subfolder's.
pub const ANCHOR_FILES: [&str; 4] = ["README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"];

/// The seed's files of one repository: its root listing, then each anchor
/// document its default branch holds at the root, matched case-insensitively,
/// in [`ANCHOR_FILES`] order, each cut into the knowledge base's sections. A
/// repository with no branch contributes nothing.
pub fn repo_seed(repo: &RepoBranches) -> anyhow::Result<Vec<RenderedFile>> {
    let Some(tip) = repo.tips.first() else {
        return Ok(Vec::new());
    };
    let Some((_, tree_id)) = repo.repo.blobs().commit_of(&Rev::Oid(tip.tip.clone()))? else {
        return Ok(Vec::new());
    };
    let tree = Tree::read(&repo.repo, &tree_id)?;
    let children = tree.children("");
    let root = format!("{}/", repo.name);
    let mut out = vec![RenderedFile {
        chunks: sections(&root, FileUnit::Folder, &listing_text(&children))?,
        path: root,
    }];
    for anchor in ANCHOR_FILES {
        let found = children
            .iter()
            .find(|(path, _, is_dir)| !is_dir && path.eq_ignore_ascii_case(anchor));
        let Some((path, _, _)) = found else {
            continue;
        };
        let Some((blob, _)) = tree.file(path) else {
            continue;
        };
        let Some(bytes) = repo.repo.blobs().read_blob(blob)? else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        let path = format!("{}/{path}", repo.name);
        out.push(RenderedFile {
            chunks: sections(&path, FileUnit::File, &text)?,
            path,
        });
    }
    Ok(out)
}

/// The seed's text: `files`' sections as the knowledge base places them after
/// `files_open`.
pub fn seed_text(files: &[RenderedFile]) -> String {
    assemble_pieces(files).into_iter().skip(1).collect()
}

/// What `files`, the seed's, read: each root listing by its repository, each
/// anchor by its path — what a dialogue's journal opens by saying it read.
pub fn seed_read(files: &[RenderedFile]) -> SeedRead {
    let mut read = SeedRead::default();
    for f in files {
        match f.path.strip_suffix('/') {
            Some(repo) => read.listings.push(repo.to_string()),
            None => read.documents.push(f.path.clone()),
        }
    }
    read
}

/// Render the seed from `repos` and install it into `builder`'s `seed`
/// optional, returning what it read. A repository whose tree cannot be read
/// is left out of the seed, and said so.
pub fn install(builder: &mut Builder, repos: &[RepoBranches]) -> anyhow::Result<SeedRead> {
    let mut files = Vec::new();
    for repo in repos {
        match repo_seed(repo) {
            Ok(of_repo) => files.extend(of_repo),
            Err(e) => tracing::warn!(
                repo = %repo.name,
                "the seed leaves this repository out: its tree could not be read: {e}"
            ),
        }
    }
    let text = seed_text(&files);
    builder
        .set_tree_option_content(SEED_NODE, "present", text.clone())
        .map_err(|e| anyhow::anyhow!("installing the seed: {e}"))?;
    tracing::info!(
        files = files.len(),
        bytes = text.len(),
        "seed installed: the repositories' roots, in the knowledge base",
    );
    Ok(seed_read(&files))
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

    /// **The seed renders a repository's root listing and its anchor
    /// documents, in the knowledge base's notation**: the listing first, then
    /// README and the lower-case `agents.md` matched as AGENTS, a subfolder's
    /// README left out; each a section, its text what follows `files_open`.
    #[test]
    fn the_seed_renders_its_files_in_the_knowledge_bases_notation() {
        let (_tmp, repo) = repo();
        let files = repo_seed(&repo).unwrap();
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["r/", "r/README.md", "r/agents.md"]);
        let text = seed_text(&files);
        assert!(
            text.starts_with(
                "## The content from r/:\n\n```\nREADME.md 4\nagents.md 5\ndocs/\nsrc/\n```\n\n\
                 ## The content from r/README.md:\n\n```markdown file=r/README.md part="
            ),
            "{text}"
        );
        assert!(
            text.ends_with("end of r/agents.md lines 1 of 1 — the end of the file\n\n"),
            "{text}"
        );
        let read = seed_read(&files);
        assert_eq!(read.listings, ["r"]);
        assert_eq!(read.documents, ["r/README.md", "r/agents.md"]);
    }
}
