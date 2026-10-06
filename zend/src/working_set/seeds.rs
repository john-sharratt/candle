//! The folders and files a dialogue starts out attending to
//! (`docs/zend_working_set.md` §4.7).
//!
//! A seed is a workspace path: `repo/dir/` names a folder's `repo_map` unit,
//! `repo/path/file` a file's `code_reading` conversation, each resolved as the
//! dialogue's own base holds it. Resolved once, when a dialogue opens, into
//! provenance at the seeding momentum; from there a seed is ranked, picked and
//! faded like any other file, and the retrieval scope drops one the
//! conversation has since changed.

use std::collections::HashSet;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::working_set::WorkingSetConfig;
use candle_conversation::ConversationEngine;
use zend_vfs::RepoFiles;

use crate::code_read::chain_finished;
use crate::fast_path::coverage::Coverage;
use crate::fast_path::file_conversation;

/// Finds the committed `repo_map` unit for a folder — repository and path
/// inside it, `""` for its root — as the conversation's base lists it
/// (`RetrievalScope::folder_unit`).
pub type FolderOf<'a> = &'a dyn Fn(&str, &str) -> Option<TimelineId>;

/// One seed path, read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seed<'a> {
    /// A folder unit: the repository and the folder inside it, `""` for its root.
    Folder { repo: &'a str, inner: &'a str },
    /// A file: the repository and the path inside it.
    File { repo: &'a str, rel: &'a str },
}

/// `path` as a seed — `None` when it names no repository.
pub fn parse(path: &str) -> Option<Seed<'_>> {
    let (repo, rest) = path.split_once('/')?;
    if repo.is_empty() {
        return None;
    }
    if rest.is_empty() || rest.ends_with('/') {
        return Some(Seed::Folder {
            repo,
            inner: rest.trim_end_matches('/'),
        });
    }
    Some(Seed::File { repo, rel: rest })
}

/// What resolving a dialogue's seeds consults.
pub struct Seeding<'a> {
    pub engine: &'a Mutex<ConversationEngine>,
    pub files: &'a RepoFiles,
    pub folder_of: FolderOf<'a>,
    pub coverage: &'a Coverage,
}

impl Seeding<'_> {
    /// Resolve `config`'s seeds for `target` and seed its provenance with them,
    /// in list order while the budget has room. A seed that resolves to
    /// nothing — changed on this branch, never ingested, never read whole — is
    /// skipped, and so is one the priming chain already carries. One that does
    /// not fit, or has no recorded size, is left out with a warning naming it.
    /// Returns how many were seeded.
    pub fn apply(&self, target: TimelineId, config: &WorkingSetConfig) -> usize {
        let lineage: HashSet<TimelineId> = self
            .engine
            .lock()
            .unwrap()
            .inherited_chain(target)
            .into_iter()
            .collect();
        let mut resolved: Vec<(TimelineId, &str)> = Vec::new();
        for path in &config.seeds {
            match self.resolve(path) {
                Some(timeline) if lineage.contains(&timeline) => {}
                Some(timeline) => resolved.push((timeline, path)),
                None => tracing::debug!(
                    target: "zend::working_set",
                    seed = %path,
                    "seed resolves to no complete conversation on this branch — skipped",
                ),
            }
        }
        let candidates: Vec<TimelineId> = resolved.iter().map(|(tl, _)| *tl).collect();
        let refused = self.engine.lock().unwrap().working_set_seed(
            target,
            &candidates,
            config.limits(),
            config.seed_momentum(),
        );
        if !refused.is_empty() {
            let names: Vec<&str> = resolved
                .iter()
                .filter(|(tl, _)| refused.contains(tl))
                .map(|(_, path)| *path)
                .collect();
            tracing::warn!(
                target: "zend::working_set",
                timeline = target.raw(),
                left_out = ?names,
                "seeds that did not fit the working set's budget, or have no recorded size, \
                 were left out",
            );
        }
        candidates.len() - refused.len()
    }

    /// The conversation `path` names on this branch, if a finished one does.
    fn resolve(&self, path: &str) -> Option<TimelineId> {
        match parse(path)? {
            Seed::Folder { repo, inner } => {
                // The lookup takes the engine lock itself.
                let unit = (self.folder_of)(repo, inner)?;
                chain_finished(&self.engine.lock().unwrap(), unit).then_some(unit)
            }
            Seed::File { repo, rel } => {
                let e = self.engine.lock().unwrap();
                file_conversation(&e, self.files, self.coverage, repo, rel).map(|(tl, _)| tl)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trailing_slash_names_a_folder_and_none_a_file() {
        assert_eq!(
            parse("candle/zend/src/"),
            Some(Seed::Folder {
                repo: "candle",
                inner: "zend/src"
            })
        );
        assert_eq!(
            parse("candle/"),
            Some(Seed::Folder {
                repo: "candle",
                inner: ""
            })
        );
        assert_eq!(
            parse("candle/Cargo.toml"),
            Some(Seed::File {
                repo: "candle",
                rel: "Cargo.toml"
            })
        );
        assert_eq!(
            parse("candle/docs/zend_working_set.md"),
            Some(Seed::File {
                repo: "candle",
                rel: "docs/zend_working_set.md"
            })
        );
    }

    /// A path with no repository in front of it names nothing.
    #[test]
    fn a_path_without_a_repository_is_no_seed() {
        assert_eq!(parse("Cargo.toml"), None);
        assert_eq!(parse("/zend/"), None);
    }
}
