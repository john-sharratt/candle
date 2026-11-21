//! The store, its two lower-layer kinds, and the base a branch-backed store reads.
//!
//! A store is an in-memory session layer over a read-only lower layer: nothing, a
//! folder on disk, or a git branch. Over a branch the store reads one commit, its
//! base, taken the first time it reads the branch and kept after, so the branch
//! moving under it changes nothing it reads. The base moves only when the
//! conversation moves it. A base naming objects this repository does not hold is
//! repinned to where the branch stands, keeping the session's changes; any other
//! failure says nothing about the base, and moving off it would drop the
//! conversation's own unpushed commit.

use super::*;

/// What a store reads beneath its session's changes.
#[derive(Default)]
pub(super) enum Lower {
    /// Nothing: the store is its upper layer alone.
    #[default]
    None,
    /// A folder on disk.
    Folder(PathBuf),
    /// A git repository's branch — or, with no branch known, whatever its
    /// `HEAD` names.
    Branch {
        source: Arc<GitSource>,
        rev: RwLock<Rev>,
    },
}

/// Union-mount of a session-private in-memory layer over a read-only
/// repository.
#[derive(Default)]
pub struct VfsStore {
    pub(super) upper: RwLock<Upper>,
    pub(super) lower: Lower,
}

impl VfsStore {
    /// Upper layer only — nothing beneath it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overlay the upper layer on `root`, a folder that is not a git
    /// repository.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            upper: RwLock::new(Upper::default()),
            lower: Lower::Folder(root.into()),
        }
    }

    /// Overlay the upper layer on `rev` of the repository `source` reads —
    /// a branch, as it moves.
    pub fn on_branch(source: Arc<GitSource>, rev: Rev) -> Self {
        Self {
            upper: RwLock::new(Upper::default()),
            lower: Lower::Branch {
                source,
                rev: RwLock::new(rev),
            },
        }
    }

    /// The folder of the repository beneath this store — the one it reads,
    /// or, over a branch, the one the branch belongs to. `None` for a store
    /// with nothing beneath it, or over a repository read from the substrate,
    /// which has no folder.
    pub fn root(&self) -> Option<&Path> {
        match &self.lower {
            Lower::None => None,
            Lower::Folder(root) => Some(root),
            Lower::Branch { source, .. } => source.folder(),
        }
    }

    /// What the store reads through, over a branch; `None` for a folder or
    /// for nothing.
    pub fn rev(&self) -> Option<Rev> {
        match &self.lower {
            Lower::Branch { rev, .. } => Some(rev.read().unwrap().clone()),
            _ => None,
        }
    }

    /// Name `branch` as the one this store is on. A store holding no changes
    /// and no merge being finished lets go of its base too, and takes the
    /// branch's on its next read; any other keeps the base its work is made
    /// on — a conversation being restored, whose saved base [`Self::restore`]
    /// puts back. Moving a store with work onto another branch is
    /// [`Self::move_base`]. `false`, and nothing changes, for a store that is
    /// not over a branch.
    pub fn set_branch(&self, branch: BranchName) -> bool {
        match &self.lower {
            Lower::Branch { rev, .. } => {
                let mut upper = self.upper.write().unwrap();
                *rev.write().unwrap() = Rev::Branch(branch);
                let merging = upper.base.as_ref().is_some_and(|b| b.merging().is_some());
                if upper.chains.is_empty() && !merging {
                    upper.base = None;
                }
                true
            }
            _ => false,
        }
    }

    /// Put the store at `to`, on `branch`, with none of its work: every
    /// change and every conflict dropped, and any merge being finished with
    /// them. Unlike [`Self::move_base`] nothing is carried, so nothing about
    /// the work — nor the base it was made on, which may no longer be in the
    /// repository — can stand in the way. Returns how many changed paths
    /// were dropped.
    pub fn reset_to(&self, branch: Option<BranchName>, to: Base) -> Result<usize, VfsError> {
        let Lower::Branch { rev, .. } = &self.lower else {
            return Err(VfsError::Unwritable(
                "only a store over a branch has a base to move".into(),
            ));
        };
        let mut upper = self.upper.write().unwrap();
        let dropped = upper.chains.len();
        *upper = Upper {
            chains: HashMap::new(),
            base: Some(to),
        };
        if let Some(branch) = branch {
            *rev.write().unwrap() = Rev::Branch(branch);
        }
        Ok(dropped)
    }

    /// What the session's changes are made on, over a branch — taken from
    /// the branch now if the store has not read it yet. `None` for a store
    /// over a folder or nothing.
    pub fn base(&self) -> Result<Option<Base>, VfsError> {
        match &self.lower {
            Lower::Branch { source, rev } => Ok(Some(self.pinned(source, rev)?)),
            _ => Ok(None),
        }
    }

    /// What the session's changes are made on, without taking it: the base
    /// the store holds, or — for a store that has not read its branch yet —
    /// where the branch stands now, which its first read would take. The
    /// store is left as it was either way. `None` for a store over a folder
    /// or nothing.
    pub fn peek_base(&self) -> Result<Option<Base>, VfsError> {
        let Lower::Branch { source, rev } = &self.lower else {
            return Ok(None);
        };
        if let Some(base) = &self.upper.read().unwrap().base {
            return Ok(Some(base.clone()));
        }
        Ok(Some(Self::branch_base(source, rev)?))
    }

    /// Every file of `base`, as its tree lists them — over a branch; `None`
    /// for a store over a folder or nothing. The session's changes are not in
    /// it. A tree read recently is answered from memory.
    pub fn tree_at(&self, base: &Base) -> Result<Option<Arc<Tree>>, VfsError> {
        let Lower::Branch { source, .. } = &self.lower else {
            return Ok(None);
        };
        let tree = source.tree_of(base).map_err(|e| {
            VfsError::Unreadable(format!("the base could not be read from git: {e}"))
        })?;
        Ok(Some(tree))
    }

    /// The bytes of the blob `id` from the repository behind a store over a
    /// branch; `None` for a store over a folder or nothing. The session's
    /// changes are never consulted: a blob id names bytes, which no change
    /// the session makes can alter.
    pub fn blob_at(&self, id: &Oid) -> Result<Option<Vec<u8>>, VfsError> {
        let Lower::Branch { source, .. } = &self.lower else {
            return Ok(None);
        };
        let bytes = source.blob(id).map_err(|e| {
            VfsError::Unreadable(format!("blob {id} could not be read from git: {e}"))
        })?;
        Ok(Some(bytes))
    }

    /// What names `path`'s content as the lower layer holds it: its blob id
    /// and size. Over a branch the base's tree says, with nothing read; over
    /// a folder the file is read and its id computed as git would
    /// ([`ObjectFormat::blob_id`], SHA-1). `None` when the session has
    /// changed the path — its own copy is not the lower layer's — or when
    /// there is no such file.
    pub fn content_id(&self, path: &str) -> Result<Option<(Oid, u64)>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        if self.holds(&norm) {
            return Ok(None);
        }
        match self.view()? {
            View::Empty => Ok(None),
            View::Branch { tree, .. } => {
                Ok(tree.file(&norm).map(|(blob, size)| (blob.clone(), size)))
            }
            View::Folder(root) => Ok(folder::read_bytes(root, &norm)?
                .map(|bytes| (ObjectFormat::Sha1.blob_id(&bytes), bytes.len() as u64))),
        }
    }

    /// The store's base, taken from the branch it names the first time.
    pub(super) fn pinned(&self, source: &GitSource, rev: &RwLock<Rev>) -> Result<Base, VfsError> {
        if let Some(base) = &self.upper.read().unwrap().base {
            return Ok(base.clone());
        }
        let found = Self::branch_base(source, rev)?;
        let mut upper = self.upper.write().unwrap();
        // Another caller may have taken it between the two locks.
        Ok(upper.base.get_or_insert(found).clone())
    }

    /// Where the branch the store names stands now.
    pub(super) fn branch_base(source: &GitSource, rev: &RwLock<Rev>) -> Result<Base, VfsError> {
        let rev = rev.read().unwrap().clone();
        source.base_at(&rev).map_err(|e| {
            VfsError::Unreadable(format!("the branch could not be read from git: {e}"))
        })
    }

    /// Pin the store to `base`, whatever it names.
    #[cfg(test)]
    pub(crate) fn set_base_for_test(&self, base: Base) -> Result<(), VfsError> {
        self.upper.write().unwrap().base = Some(base);
        Ok(())
    }

    /// Move a store whose base can no longer be read onto where its branch
    /// stands, keeping the session's changes on top, and its tree.
    ///
    /// A base names a commit made on one member, held there until it is packed;
    /// a worker carried to another member, or a restart, can leave it naming
    /// objects this database never had. Refused, every read of the repository
    /// failed for the rest of the conversation and two workers gave up on a
    /// job they had half written (2026-10-31). Unchanged branch, or none readable,
    /// is the refusal it always was.
    pub(super) fn repin(
        &self,
        source: &GitSource,
        rev: &RwLock<Rev>,
        lost: &Base,
        why: GitError,
    ) -> Result<Arc<Tree>, VfsError> {
        let unreadable = |e: &dyn fmt::Display| {
            VfsError::Unreadable(format!("the base could not be read from git: {e}"))
        };
        // Only an object git does not hold. Any other failure — a timeout, a
        // process that would not start — says nothing about the base, and
        // moving off it would drop the conversation's own unpushed commit.
        let missing = matches!(why, GitError::UnknownRevision { .. })
            || why.to_string().contains("not found");
        let current = Self::branch_base(source, rev)?;
        if !missing || &current == lost {
            return Err(unreadable(&why));
        }
        let tree = source.tree_of(&current).map_err(|e| unreadable(&e))?;
        tracing::warn!(
            lost = ?lost.commit(),
            current = ?current.commit(),
            "a store's base could not be read from git ({why}): it moves onto its branch, \
             its changes kept"
        );
        self.upper.write().unwrap().base = Some(current);
        Ok(tree)
    }

    /// The lower layer: over a branch, the tree of the store's base.
    pub(super) fn view(&self) -> Result<View<'_>, VfsError> {
        match &self.lower {
            Lower::None => Ok(View::Empty),
            Lower::Folder(root) => Ok(View::Folder(root)),
            Lower::Branch { source, rev } => {
                let base = self.pinned(source, rev)?;
                let tree = match source.tree_of(&base) {
                    Ok(tree) => tree,
                    Err(e) => self.repin(source, rev, &base, e)?,
                };
                Ok(View::Branch { source, tree })
            }
        }
    }
}
