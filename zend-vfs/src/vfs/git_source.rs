//! One repository's committed files, shared by every store that reads it.
//!
//! A conversation's store reads a repository through its branch, never its
//! folder: the folder is the sandbox's, and holds whatever the last command
//! run there left. [`GitSource`] is what reads the branch. There is one per
//! repository, and every conversation's store over that repository shares
//! it, so a daemon with a hundred conversations still runs one git process
//! per repository:
//!
//! - **One running `cat-file --batch`** answers both questions a store asks:
//!   which commit and tree a branch holds (`<branch>^{commit}`, asked when a
//!   store first reads the branch and takes it as its base), and a file's
//!   bytes.
//! - **The trees read most recently are kept**, parsed, by id. A tree is
//!   listed with one `ls-tree -r` the first time any store asks for it; after
//!   that, a store's listings, searches and existence checks run in memory.

use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::base::Base;
use super::tree::Tree;
use crate::{BlobReader, GitError, Oid, Repo, Rev};

/// How many parsed trees a source keeps: a branch's tree and the one before
/// it, for a handful of branches read at once.
const KEPT_TREES: usize = 8;

/// A repository's branches, read through git.
pub struct GitSource {
    repo: Repo,
    blobs: BlobReader,
    /// The trees read most recently, oldest first.
    trees: Mutex<Vec<Arc<Tree>>>,
    /// What a store reads when it is given no branch: the branch checked out
    /// when the source was opened, or that commit when none was.
    default: Rev,
}

impl fmt::Debug for GitSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GitSource")
            .field("dir", &self.repo.dir())
            .field("default", &self.default)
            .finish()
    }
}

impl GitSource {
    /// The repository whose top level is `dir`.
    pub fn open(dir: &Path) -> Result<Arc<Self>, GitError> {
        let repo = Repo::open(dir)?;
        let default = match repo.head()?.branch() {
            Some(branch) => Rev::Branch(branch.clone()),
            None => Rev::Head,
        };
        let blobs = repo.blobs();
        Ok(Arc::new(Self {
            repo,
            blobs,
            trees: Mutex::new(Vec::new()),
            default,
        }))
    }

    /// The repository's folder.
    pub fn dir(&self) -> &Path {
        self.repo.dir()
    }

    /// What a store over this repository reads when it is given no branch.
    pub fn default_rev(&self) -> &Rev {
        &self.default
    }

    /// Where `rev` stands now, as a store's base: its commit and that
    /// commit's tree, or [`Base::empty`] when `rev` names nothing.
    pub(crate) fn base_at(&self, rev: &Rev) -> Result<Base, GitError> {
        Ok(match self.blobs.commit_of(rev)? {
            Some((commit, tree)) => Base::at(commit, tree),
            None => Base::empty(),
        })
    }

    /// The tree a base's changes are made against.
    pub(crate) fn tree_of(&self, base: &Base) -> Result<Arc<Tree>, GitError> {
        match &base.tree {
            Some(id) => self.tree(id),
            None => Ok(Arc::new(Tree::empty())),
        }
    }

    /// The tree `id`, from the kept ones or listed now.
    pub(crate) fn tree(&self, id: &Oid) -> Result<Arc<Tree>, GitError> {
        if let Some(kept) = self.kept(id) {
            return Ok(kept);
        }
        let tree = Arc::new(Tree::from_listing(id.clone(), self.repo.ls_tree_all(id)?));
        let mut trees = self.trees.lock().unwrap_or_else(|e| e.into_inner());
        trees.retain(|t| t.id() != Some(id));
        trees.push(Arc::clone(&tree));
        if trees.len() > KEPT_TREES {
            trees.remove(0);
        }
        Ok(tree)
    }

    /// The bytes of the blob `id`.
    pub(crate) fn blob(&self, id: &Oid) -> Result<Vec<u8>, GitError> {
        self.blobs
            .read_blob(id)?
            .ok_or_else(|| GitError::invalid(format!("the repository holds no blob {id}")))
    }

    fn kept(&self, id: &Oid) -> Option<Arc<Tree>> {
        let mut trees = self.trees.lock().unwrap_or_else(|e| e.into_inner());
        let at = trees.iter().position(|t| t.id() == Some(id))?;
        // Most recently used last, so the oldest is the one let go.
        let tree = trees.remove(at);
        trees.push(Arc::clone(&tree));
        Some(tree)
    }

    #[cfg(test)]
    fn kept_ids(&self) -> Vec<Oid> {
        let trees = self.trees.lock().unwrap();
        trees.iter().filter_map(|t| t.id().cloned()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;
    use crate::BranchName;

    fn branch(name: &str) -> Rev {
        Rev::Branch(BranchName::parse(name).unwrap())
    }

    fn tree_at(source: &GitSource, rev: &Rev) -> Arc<Tree> {
        source.tree_of(&source.base_at(rev).unwrap()).unwrap()
    }

    /// **A branch's base is where it stands now** — the same source sees it
    /// move — and a file's bytes come from the blob its tree names.
    #[test]
    fn a_branch_reads_as_it_stands() {
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        let first = t.commit_all("first");
        let source = GitSource::open(&t.path).unwrap();
        assert_eq!(source.default_rev(), &branch("main"));

        let base = source.base_at(&branch("main")).unwrap();
        assert_eq!(base.parents, vec![first]);
        let tree = source.tree_of(&base).unwrap();
        let (blob, size) = tree.file("a.txt").unwrap();
        assert_eq!((source.blob(blob).unwrap(), size), (b"one\n".to_vec(), 4));

        t.write("a.txt", b"two!\n");
        t.commit_all("second");
        let moved = tree_at(&source, &branch("main"));
        assert_ne!(moved.id(), tree.id());
        let (blob, _) = moved.file("a.txt").unwrap();
        assert_eq!(source.blob(blob).unwrap(), b"two!\n");
    }

    /// **The working tree is never read** — only what the branch committed.
    #[test]
    fn what_is_on_disk_but_not_committed_is_not_there() {
        let t = TestRepo::init();
        t.write("a.txt", b"committed\n");
        t.commit_all("first");
        t.write("a.txt", b"on disk only\n");
        t.write("new.txt", b"untracked\n");
        let source = GitSource::open(&t.path).unwrap();
        let tree = tree_at(&source, &branch("main"));
        let (blob, _) = tree.file("a.txt").unwrap();
        assert_eq!(source.blob(blob).unwrap(), b"committed\n");
        assert_eq!(tree.file("new.txt"), None);
    }

    /// A branch that does not exist, or a repository with no commit, reads
    /// as empty rather than failing.
    #[test]
    fn a_missing_branch_reads_as_empty() {
        let t = TestRepo::init();
        let source = GitSource::open(&t.path).unwrap();
        assert!(tree_at(&source, &branch("main")).files_under("").is_empty());
        assert_eq!(source.base_at(&branch("nope")).unwrap(), Base::empty());
        assert!(tree_at(&source, &branch("nope")).id().is_none());
    }

    /// **A tree is listed once and kept**; asking again answers from memory,
    /// and the least recently used goes first once the source holds enough.
    #[test]
    fn trees_are_kept_most_recent_last() {
        let t = TestRepo::init();
        let source = GitSource::open(&t.path).unwrap();
        let mut ids = Vec::new();
        for n in 0..=KEPT_TREES {
            t.write("n.txt", n.to_string().as_bytes());
            t.commit_all("next");
            ids.push(tree_at(&source, &branch("main")).id().cloned().unwrap());
        }
        assert_eq!(
            source.kept_ids(),
            ids[1..].to_vec(),
            "the oldest was let go"
        );
        let again = source.tree(&ids[1]).unwrap();
        assert_eq!(again.id(), Some(&ids[1]));
        assert_eq!(
            source.kept_ids().last(),
            Some(&ids[1]),
            "used again, kept longest"
        );
    }
}
