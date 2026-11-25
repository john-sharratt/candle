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

use candle_conversation::files::{assemble_pieces, Chunk, LineSpan, RenderedFile};
use candle_conversation::projection::Builder;
use zend_vfs::vfs::Tree;
use zend_vfs::{GitError, Rev};

use super::listing::listing_text;
use crate::branch_ingest::prime::ANCHOR_FILES;
use crate::branch_ingest::walk::RepoBranches;

/// The tree node whose present side the seed is installed into.
pub const SEED_NODE: &str = "seed";

/// `text` as one chunk of a file: every line ending its `\n`.
fn whole(text: &str) -> Chunk {
    let text = if text.is_empty() || text.ends_with('\n') {
        text.to_string()
    } else {
        format!("{text}\n")
    };
    let lines = text.matches('\n').count() as u32;
    Chunk {
        lines: LineSpan {
            first: 1,
            last: lines.max(1),
        },
        text,
    }
}

/// The seed's files of one repository: its root listing, then each anchor
/// document its default branch holds at the root, matched case-insensitively,
/// in [`ANCHOR_FILES`] order. A repository with no branch contributes nothing.
pub fn repo_seed(repo: &RepoBranches) -> Result<Vec<RenderedFile>, GitError> {
    let Some(tip) = repo.tips.first() else {
        return Ok(Vec::new());
    };
    let Some((_, tree_id)) = repo.repo.blobs().commit_of(&Rev::Oid(tip.tip.clone()))? else {
        return Ok(Vec::new());
    };
    let tree = Tree::read(&repo.repo, &tree_id)?;
    let children = tree.children("");
    let mut out = vec![RenderedFile {
        path: format!("{}/", repo.name),
        chunks: vec![whole(&listing_text(&children))],
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
        out.push(RenderedFile {
            path: format!("{}/{path}", repo.name),
            chunks: vec![whole(&text)],
        });
    }
    Ok(out)
}

/// The seed's text: `files`' snippets as the knowledge base places them after
/// `files_open`, the last ending on its closing fence, which the close follows.
pub fn seed_text(files: &[RenderedFile]) -> String {
    assemble_pieces(files).into_iter().skip(1).collect()
}

/// Render the seed from `repos` and install it into `builder`'s `seed`
/// optional, returning its text's length in bytes. A repository whose tree
/// cannot be read is left out of the seed, and said so.
pub fn install(builder: &mut Builder, repos: &[RepoBranches]) -> anyhow::Result<usize> {
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
    Ok(text.len())
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
    /// documents, in Q1's notation**: the listing first, then README and the
    /// lower-case `agents.md` matched as AGENTS, a subfolder's README left
    /// out; the text is what follows `files_open`, ending on the last fence.
    #[test]
    fn the_seed_renders_its_files_in_the_knowledge_bases_notation() {
        let (_tmp, repo) = repo();
        let files = repo_seed(&repo).unwrap();
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["r/", "r/README.md", "r/agents.md"]);
        assert_eq!(
            seed_text(&files),
            "## The content from r/:\n\n```\nREADME.md 4\nagents.md 5\ndocs/\nsrc/\n```\n\n\
             ## The content from r/README.md:\n\n```\n# R\n```\n\n\
             ## The content from r/agents.md:\n\n```\nrules\n```\n"
        );
    }
}
