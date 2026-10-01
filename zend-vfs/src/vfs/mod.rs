//! Overlay filesystem backing the `file_*` tools — one store per repository.
//!
//! A workspace holds several repositories ([`super::workspace`]), and each has
//! its own store: [`super::files::RepoFiles`] routes a call's `repo` argument to
//! it. Inside a store every path is relative to that repository's folder, which
//! this module calls the *root*.
//!
//! Two layers, in the union-mount sense:
//!
//! * **Upper** — in memory, the session's changes and nothing else: for each
//!   path it changed, the chain of [`FileDelta`]s that took the file from where
//!   it started to where it stands now. A whole-file write is one
//!   [`FileDelta::Replace`]; an edit is one [`FileDelta::Edit`] carrying only
//!   the lines it changed — or, when it changes more than half the file's
//!   lines, a replace of the whole file; a deletion is a [`FileDelta::Delete`].
//!   A replace or a delete supersedes every earlier delta for its path.
//! * **Lower** — the repository as committed, read-only, one of:
//!   - **a branch** ([`VfsStore::on_branch`]) — a git repository is read
//!     through the branch the conversation works on, as that branch's tree
//!     holds it, never through its folder: the folder is the sandbox's, and
//!     holds whatever the last command run there left in it
//!     ([`git_source`]);
//!   - **a folder** ([`VfsStore::with_root`]) — a folder that is not a git
//!     repository, read as it stands on disk ([`folder`]);
//!   - **nothing** ([`VfsStore::new`]) — the store is its upper layer alone.
//!
//! One store belongs to one conversation (see [`super::files`]): a session's
//! changes are its own, never visible to another conversation.
//!
//! A read of a path the session has not changed falls through to the lower
//! layer, so a tool call sees the real project without the session having to
//! load it. A read of a path it has changed replays that path's chain — onto
//! the lower layer's copy when the chain opens with an edit. A write always
//! lands in the upper layer — the repository is **never** modified.
//!
//! A chain that opens with an edit depends on the copy it was made against.
//! Each splice names the text it removes, so when a folder's copy changes on
//! disk underneath the edit, replay refuses it as [`VfsError::Diverged`] rather
//! than splicing the change into the wrong place; writing the whole file
//! supersedes the edit and settles the file again.
//!
//! A store over a branch reads it at one commit, its [`Base`]: taken from the
//! branch the first time the store reads it, and kept. The branch moving under
//! it — another conversation's commit, a push from elsewhere — changes nothing
//! it reads. The base moves only when the conversation moves it
//! ([`VfsStore::move_base`]): its own commit, a merge, a switch, a reset. Each
//! move carries the chains onto the new base's tree ([`carry`]), asking the
//! move how every path the two trees hold differently should now read — so a
//! merge is where another writer's changes meet this conversation's, and any
//! overlap is left in the conversation's copy between conflict markers, the
//! path flagged until the conversation settles it ([`VfsStore::conflicts`]).
//!
//! Deleting a file the lower layer holds records a [`FileDelta::Delete`]
//! instead of touching it: the path then reads as absent and stops appearing in
//! listings. Writing the path supersedes the deletion.
//!
//! # Path normalisation
//!
//! Paths are normalised to one canonical key before use: a leading `/` is
//! stripped, `.` and empty segments collapse, and `..` pops the stack (it can
//! never escape the root — popping an empty stack is a no-op). So
//! `/src/main.rs`, `./src/main.rs`, `src/util/../main.rs`, and `src/main.rs` are
//! all the same key, `src/main.rs`, in both layers.
//!
//! # Lower-layer rules
//!
//! A branch holds only what was committed, so `target/` and friends never
//! appear; a folder's walk is `ignore`-driven (the same crate ripgrep uses), so
//! `.gitignore`, `.ignore`, the global git ignore apply to it. In both, hidden
//! files are excluded from listings the way `ls` excludes them, but they still
//! *read* fine by exact path: `.gitignore` does not show up in `list` and does
//! resolve in `read`.
//!
//! Files above [`MAX_LOWER_FILE_BYTES`] are listed but refuse to read, as do files
//! whose bytes are not valid UTF-8; both surface as [`VfsError::Unreadable`].
//!
//! # Protected paths
//!
//! Any path with a component in [`PROTECTED_SEGMENTS`] is refused outright, in
//! both layers and by every operation: [`VfsError::Forbidden`]. Two segments
//! are protected.
//!
//! [`PROTECTED_SEGMENT`] — `secrets/` — covers the gateway's
//! `web/secrets/auth.yaml` and anything else a repository keeps there. The
//! daemon's own keys live outside every repository
//! (the tool layer's `Secrets`); this guard is what protects the ones
//! that do not.
//!
//! [`PROTECTED_GIT_DIR`] — `.git/` — covers a repository's git database, for
//! reasons set out on the constant: it holds remote URLs with their
//! credentials intact, which the git layer redacts and this would not, and
//! hand-parsing it produces confidently wrong answers where the `git_*` tools
//! give right ones.
//!
//! **This is not the same protection as `.gitignore`, and the difference is the
//! whole point.** The ignore rules are consulted by the listing walk and by
//! nothing else — a read resolves a normalised key straight to a path under the
//! root and opens it. So before this guard existed, a gitignored secret was
//! invisible to `file_list` and served in full by `file_read`, which is the
//! worst of both worlds: hidden from the operator auditing what the model can
//! see, and one call away from the transcript.
//!
//! The refusal is enforced where each lower layer is built, rather than at
//! each call site — a guard that has to be remembered at N call sites is a
//! guard that is missing at one of them: a folder reaches every file through
//! one funnel ([`folder::path`]), and a branch's tree never holds a protected
//! path at all ([`tree`]). Normalisation runs first, so alternate spellings
//! (`/secrets/x`, `a/../secrets/x`, backslashes) all collapse onto the same
//! key before the check sees it.
//!
//! # Size cap
//!
//! The upper layer is capped at 10 MiB per store, counted as the bytes its
//! deltas hold (enforced on each write and edit). Reading through to the
//! workspace costs nothing against the cap because nothing is retained, and
//! neither does the unchanged part of an edited file; a change that would not
//! fit returns [`VfsError::Full`] and records nothing.

mod base;
mod carry;
mod conflict;
mod folder;
pub mod git_source;
pub mod tree;
mod view;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

pub use self::base::{Base, SavedBase};
pub use self::carry::{Carried, Resolved, Resolver, Side};
pub use self::conflict::has_markers;
use self::git_source::GitSource;
pub use self::tree::Tree;
use self::view::View;
use super::file_delta::{self, FileDelta, FileTimes, ReplayError, TimedDelta};
use crate::{BranchName, FileChanges, ObjectFormat, Oid, Rev};

const MAX_BYTES: usize = 10 * 1024 * 1024; // 10 MiB

/// Largest workspace file the lower layer will read into a tool response.
/// Listing is unaffected — an oversize file still shows up with its true size.
pub const MAX_LOWER_FILE_BYTES: u64 = 4 * 1024 * 1024; // 4 MiB

/// Path segment marking a directory the tools may not touch.
///
/// A repository's secrets live in a `secrets/` directory — `web/secrets/auth.yaml`
/// for the gateway's sign-in config, `npcd/secrets/` for npcd's. One name, matched at any depth, so a new secrets directory is
/// protected the day it is created rather than the day someone remembers to add
/// it to a list.
pub const PROTECTED_SEGMENT: &str = "secrets";

/// A repository's git database, protected for two separate reasons.
///
/// **It leaks what the git layer redacts.** `.git/config` holds a remote's URL
/// verbatim, credentials and all — `url = https://user:token@host` — and
/// the git layer redacts exactly that before any remote reaches a tool response.
/// A `file_read` of the same file hands the token over whole, so leaving
/// `.git` readable makes the redaction decorative.
///
/// **And reading it gives wrong answers.** Measured on a live turn: asked how
/// many branches and tags a repository had, the model bypassed `git_refs`,
/// read `.git/packed-refs` and `.git/refs/heads` itself, and answered 53 and
/// 23 where the truth was 56 and 24 — loose refs, packed refs and symbolic
/// refs are a database, not a list, and hand-parsing them is wrong in ways
/// that look plausible. The git tools exist to answer those questions; this
/// closes the shortcut around them.
pub const PROTECTED_GIT_DIR: &str = ".git";

/// Every path segment the tools may not touch, matched at any depth.
pub const PROTECTED_SEGMENTS: [&str; 2] = [PROTECTED_SEGMENT, PROTECTED_GIT_DIR];

#[derive(Debug)]
pub enum VfsError {
    Full,
    /// A workspace file exists but cannot be served as text — too large, or not
    /// valid UTF-8.
    Unreadable(String),
    /// The path is under a [`PROTECTED_SEGMENT`] directory. Refused whether or
    /// not it exists: saying "not found" for a real file and "forbidden" for a
    /// missing one would turn the error into an oracle for what is there.
    Forbidden(String),
    /// A change this store cannot make — a base moved on a store that reads
    /// no branch.
    Unwritable(String),
    /// The session edited the workspace's copy of a file, and that copy has
    /// since changed on disk so the edit no longer fits it.
    Diverged(String),
}

impl fmt::Display for VfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VfsError::Full => write!(f, "VFS storage limit exceeded (10 MiB)"),
            VfsError::Unreadable(why) => write!(f, "{why}"),
            VfsError::Forbidden(path) => write!(
                f,
                "{path} is under a protected directory ({PROTECTED_SEGMENT}/ or \
                 {PROTECTED_GIT_DIR}/) and cannot be read, written or listed by \
                 tools; for a repository's branches, tags, history or file \
                 contents at a revision, use the git_* tools rather than its \
                 {PROTECTED_GIT_DIR}/ folder"
            ),
            VfsError::Unwritable(why) | VfsError::Diverged(why) => write!(f, "{why}"),
        }
    }
}

/// One matching line from [`VfsStore::grep`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepHit {
    pub path: String,
    /// 1-based line number within the file.
    pub line_no: u32,
    /// The matching line, with trailing `\r` and whitespace trimmed.
    pub line: String,
    /// `true` when the hit came from this session's own copy of the file.
    pub modified: bool,
}

/// What a [`VfsStore::grep`] pass found.
#[derive(Debug, Default)]
pub struct GrepOutcome {
    pub hits: Vec<GrepHit>,
    /// Files whose contents were actually scanned — the denominator that tells a
    /// caller whether "no matches" means "searched a lot and found nothing" or
    /// "the prefix matched nothing to search".
    pub files_searched: usize,
    /// `true` when the scan stopped at its hit ceiling, so the result is a
    /// prefix of what is there rather than all of it.
    pub truncated: bool,
}

/// One entry in a directory listing: normalised path, byte size. No line
/// count — that would cost opening every file in the walk to compute, and a
/// listing that never opens a file is the whole point of `file_list`. A file's length reaches the model
/// through `file_read`'s own opening fence instead (`page=0/13 lines=2499`),
/// which is exact, costs nothing extra, and arrives at the moment the
/// number is actually needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListEntry {
    pub path: String,
    /// `None` for a directory entry — a directory has no size of its own.
    pub bytes: Option<usize>,
    /// `true` when this entry is a subdirectory rather than a file. A listing
    /// is one level deep, so a subdirectory appears as itself — never
    /// expanded into the files it holds.
    pub dir: bool,
    /// `true` when the entry is the session's own copy (upper layer) rather than
    /// a file read straight off the workspace. Always `false` for a directory
    /// entry: a directory is not itself written, only the files inside it.
    pub modified: bool,
}

/// How the session has changed a path, against the lower layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FileState {
    /// The lower layer has no such file.
    Added,
    Modified,
    Deleted,
}

/// Lines per [`VfsStore::read_page`] page.
///
/// The same size the rest of the system already treats as one excerpt, so a
/// live read and the prefilled excerpts the model was conditioned on are the
/// same kind of object — a scope, not a module. `zend`'s `repo_scan::anchor`
/// bounds its anchor excerpts at 200 by the identical `start + LIMIT - 1`
/// clamp, and the `code_reading` ingest carves scopes at 150
/// (`MAX_SCOPE_LINES`), so every `file_read` exchange in the corpus already
/// fits inside this cap and none had to be re-cut for it.
///
/// Chosen this small on measurement, not guesswork: one `file_read` of a
/// 2,499-line module at a wider size put 144 KB into a live conversation, and
/// three such reads made the next turn a 53,288-token prefill — three minutes
/// and fifty seconds of wall clock for one turn with the KV pool ratcheted 6
/// GB against a card already at 99%.
pub const PAGE_LINES: u32 = 200;

/// One page of a file's lines, 0-based.
#[derive(Debug)]
pub struct PageResult {
    /// The page actually returned. Clamped into range the same way the file
    /// tools clamp a listing page: a request past the end yields the last page
    /// rather than an empty one.
    pub page: u32,
    /// First line of the page, 1-based.
    pub start_line: u32,
    /// Last line of the page, 1-based and inclusive. `0` for an empty file.
    pub end_line: u32,
    pub total_lines: u32,
    pub total_pages: u32,
    /// The page's lines, joined by `\n` — never the whole file.
    pub body: String,
}

/// The session layer: for each path the session has changed, the deltas that
/// took it from where it started to where it stands now. What is held is the
/// changes, never a copy of the result — see [`file_delta`](super::file_delta).
#[derive(Default)]
struct Upper {
    chains: HashMap<String, Chain>,
    /// Over a branch, what the chains are made on — `None` until the store
    /// first reads the branch and takes it.
    base: Option<Base>,
}

impl Upper {
    fn bytes(&self) -> usize {
        self.chains.values().map(Chain::bytes).sum()
    }

    /// Whether the session has `norm` deleted: it starts with a
    /// [`FileDelta::Delete`] — a whiteout over the workspace's copy.
    fn deleted(&self, norm: &str) -> bool {
        self.chains.get(norm).is_some_and(|c| c.size.is_none())
    }
}

/// One path's deltas, in order.
///
/// A [`FileDelta::Replace`] or [`FileDelta::Delete`] supersedes everything
/// before it, so a chain is at most one of those followed by edits. A chain
/// that starts with an edit changes the workspace's copy of the file, and is
/// replayed onto it.
#[derive(Clone, PartialEq)]
struct Chain {
    /// Each with the moment its operation executed.
    deltas: Vec<TimedDelta>,
    /// The file's size after the last delta, `None` once deleted — kept so a
    /// listing never replays a chain to report a size.
    size: Option<usize>,
    /// Left in conflict by a move of the base, until a change of the
    /// session's own leaves it without a conflict's markers.
    conflict: bool,
}

impl Chain {
    fn bytes(&self) -> usize {
        self.deltas.iter().map(|t| t.delta.bytes()).sum()
    }
}

/// What a change to one path was computed against: the base and the path's
/// chain as they stood when the file was read. A change is recorded only if
/// both still stand when the write lock is taken — otherwise a move of the
/// base, or another change to the path, landed in between, and the change is
/// computed again against what is there now.
#[derive(PartialEq)]
struct Seen {
    base: Option<Base>,
    chain: Option<Chain>,
}

impl Seen {
    fn of(upper: &Upper, norm: &str) -> Self {
        Self {
            base: upper.base.clone(),
            chain: upper.chains.get(norm).cloned(),
        }
    }

    fn stands(&self, upper: &Upper, norm: &str) -> bool {
        self.base == upper.base && self.chain.as_ref() == upper.chains.get(norm)
    }
}

/// A store's session state, saved: each changed path's chain — every delta
/// with the moment it was made, the size the chain leaves the file at (`None`
/// once deleted), and whether it is in conflict — and, over a branch, the
/// [`Base`] they are made on, so a restored conversation reads the commit it
/// read before however the branch has moved since.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base: Option<SavedBase>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    chains: BTreeMap<String, SavedChain>,
}

impl Snapshot {
    /// A snapshot of `base` and `chains` — what a saved form other than this
    /// one's own serialisation is read back into.
    pub fn new(base: Option<SavedBase>, chains: BTreeMap<String, SavedChain>) -> Self {
        Self { base, chains }
    }

    /// Whether there is nothing to save: no base taken, no path changed.
    pub fn is_empty(&self) -> bool {
        self.base.is_none() && self.chains.is_empty()
    }

    /// What the chains are made on; `None` before the store first read its
    /// branch.
    pub fn base(&self) -> Option<&SavedBase> {
        self.base.as_ref()
    }

    /// Every changed path's chain, by path.
    pub fn chains(&self) -> &BTreeMap<String, SavedChain> {
        &self.chains
    }
}

/// One path's saved chain — see [`Snapshot`]: every delta with the moment it
/// was made, the size it leaves the file at (`None` once deleted), and
/// whether it is in conflict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedChain {
    pub deltas: Vec<TimedDelta>,
    pub size: Option<usize>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub conflict: bool,
}

/// What a store reads beneath its session's changes.
#[derive(Default)]
enum Lower {
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
    upper: RwLock<Upper>,
    lower: Lower,
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
    /// with nothing beneath it.
    pub fn root(&self) -> Option<&Path> {
        match &self.lower {
            Lower::None => None,
            Lower::Folder(root) => Some(root),
            Lower::Branch { source, .. } => Some(source.dir()),
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
    fn pinned(&self, source: &GitSource, rev: &RwLock<Rev>) -> Result<Base, VfsError> {
        if let Some(base) = &self.upper.read().unwrap().base {
            return Ok(base.clone());
        }
        let found = Self::branch_base(source, rev)?;
        let mut upper = self.upper.write().unwrap();
        // Another caller may have taken it between the two locks.
        Ok(upper.base.get_or_insert(found).clone())
    }

    /// Where the branch the store names stands now.
    fn branch_base(source: &GitSource, rev: &RwLock<Rev>) -> Result<Base, VfsError> {
        let rev = rev.read().unwrap().clone();
        source.base_at(&rev).map_err(|e| {
            VfsError::Unreadable(format!("the branch could not be read from git: {e}"))
        })
    }

    /// The lower layer: over a branch, the tree of the store's base.
    fn view(&self) -> Result<View<'_>, VfsError> {
        match &self.lower {
            Lower::None => Ok(View::Empty),
            Lower::Folder(root) => Ok(View::Folder(root)),
            Lower::Branch { source, rev } => {
                let base = self.pinned(source, rev)?;
                let tree = source.tree_of(&base).map_err(|e| {
                    VfsError::Unreadable(format!("the base could not be read from git: {e}"))
                })?;
                Ok(View::Branch { source, tree })
            }
        }
    }

    /// Move the store onto `to` — and, given `branch`, onto that branch —
    /// carrying the session's changes across ([`carry`]): every changed path
    /// the old and new trees hold differently, and each of `extra`, is put
    /// to `resolve`, and its answer recorded against the new tree. Returns
    /// every path left in conflict. Everything is resolved first, so a
    /// failure — a refusal from `resolve`, or changes that would no longer
    /// fit the size cap — leaves the store as it was. Only a store over a
    /// branch has a base to move.
    pub fn move_base(
        &self,
        branch: Option<BranchName>,
        to: Base,
        extra: &[String],
        resolve: &mut Resolver<'_>,
    ) -> Result<Vec<String>, VfsError> {
        let Lower::Branch { source, rev } = &self.lower else {
            return Err(VfsError::Unwritable(
                "only a store over a branch has a base to move".into(),
            ));
        };
        let unreadable =
            |e| VfsError::Unreadable(format!("the base could not be read from git: {e}"));
        // Held for the whole move: no change of the conversation's can land
        // between reading the base the chains are made on and replacing it.
        let mut upper = self.upper.write().unwrap();
        // With nothing to carry — no change of the conversation's, and no
        // path the caller asks to reconsider — the old tree is never read, so
        // a base the repository no longer holds is still left behind.
        let nothing_to_carry = upper.chains.is_empty() && extra.is_empty();
        let old_tree = match (&upper.base, nothing_to_carry) {
            (_, true) => Arc::new(Tree::empty()),
            (Some(from), false) => source.tree_of(from).map_err(unreadable)?,
            (None, false) => {
                let rev = rev.read().unwrap().clone();
                let from = source.base_at(&rev).map_err(unreadable)?;
                source.tree_of(&from).map_err(unreadable)?
            }
        };
        let old = View::Branch {
            source,
            tree: old_tree,
        };
        let new = View::Branch {
            source,
            tree: source.tree_of(&to).map_err(unreadable)?,
        };
        let mut chains = upper.chains.clone();
        carry::carry(&mut chains, &old, &new, extra, resolve)?;
        let moved = Upper {
            chains,
            base: Some(to),
        };
        if moved.bytes() > MAX_BYTES {
            return Err(VfsError::Full);
        }
        *upper = moved;
        if let Some(branch) = branch {
            *rev.write().unwrap() = Rev::Branch(branch);
        }
        Ok(Self::conflicted(&upper))
    }

    /// Every path left in conflict by a move of the base, sorted — the
    /// conversation's to settle before it commits.
    pub fn conflicts(&self) -> Vec<String> {
        Self::conflicted(&self.upper.read().unwrap())
    }

    fn conflicted(upper: &Upper) -> Vec<String> {
        let mut out: Vec<String> = upper
            .chains
            .iter()
            .filter(|(_, c)| c.conflict)
            .map(|(path, _)| path.clone())
            .collect();
        out.sort();
        out
    }

    /// Whether `norm`, about to hold `content`, stays in conflict: it was,
    /// and a conflict's markers are still in it.
    fn still_conflicted(upper: &Upper, norm: &str, content: Option<&str>) -> bool {
        upper.chains.get(norm).is_some_and(|c| c.conflict) && content.is_some_and(has_markers)
    }

    /// Replace the whole file with `content` — recorded as one
    /// [`FileDelta::Replace`], superseding every earlier delta for the path,
    /// a deletion included. Returns whether this created a path that did not
    /// previously resolve: replacing a workspace file for the first time is an
    /// overwrite, and writing over a deletion is a creation.
    pub fn write(&self, path: &str, content: String) -> Result<bool, VfsError> {
        let norm = Self::normalize(path);
        // Nothing reaches disk, but a session planting a decoy at a protected
        // path would have later reads find it: refused like a read.
        Self::guard(&norm)?;
        let view = self.view()?;
        let created = !self.resolves_as_file(&view, &norm);
        let size = content.len();
        let mut guard = self.upper.write().unwrap();
        let conflict = Self::still_conflicted(&guard, &norm, Some(&content));
        let chain = Chain {
            deltas: vec![TimedDelta::now(FileDelta::Replace { content })],
            size: Some(size),
            conflict,
        };
        Self::set_capped(&mut guard, norm, chain)?;
        Ok(created)
    }

    /// Change the file to `content` by an edit — recorded as one
    /// [`FileDelta::Edit`] carrying only the lines that differ from what the
    /// file holds now, or, when those are more than
    /// [`REPLACE_ABOVE_PERCENT`](file_delta::REPLACE_ABOVE_PERCENT) of its
    /// lines, as the whole file, exactly as a [`Self::write`] records it. The
    /// file must exist. Returns whether anything changed: an edit that leaves
    /// the file as it was records nothing.
    pub fn edit(&self, path: &str, content: String) -> Result<bool, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        loop {
            let seen = Seen::of(&self.upper.read().unwrap(), &norm);
            let view = self.view()?;
            let current = self
                .current(&view, &norm)?
                .ok_or_else(|| VfsError::Unreadable(format!("{norm} does not exist to edit")))?;
            if current == content {
                // Keeping a file in conflict exactly as it stands — the side
                // of an edit that met a delete, say — settles it, markers
                // aside.
                let mut guard = self.upper.write().unwrap();
                if !seen.stands(&guard, &norm) {
                    continue;
                }
                if let Some(chain) = guard.chains.get_mut(&norm) {
                    if chain.conflict && !has_markers(&content) {
                        chain.conflict = false;
                    }
                }
                return Ok(false);
            }
            let size = Some(content.len());
            let delta = TimedDelta::now(file_delta::delta(&current, &content));
            let mut guard = self.upper.write().unwrap();
            if !seen.stands(&guard, &norm) {
                continue;
            }
            let conflict = Self::still_conflicted(&guard, &norm, Some(&content));
            let chain = match delta.delta {
                // A replacement supersedes the chain, as a write does.
                FileDelta::Replace { .. } => Chain {
                    deltas: vec![delta],
                    size,
                    conflict,
                },
                _ => {
                    let mut chain = seen.chain.unwrap_or(Chain {
                        deltas: Vec::new(),
                        size: None,
                        conflict: false,
                    });
                    chain.deltas.push(delta);
                    chain.size = size;
                    chain.conflict = conflict;
                    chain
                }
            };
            Self::set_capped(&mut guard, norm, chain)?;
            return Ok(true);
        }
    }

    /// Resolve a path through the overlay: the session's deltas replayed onto
    /// the workspace's copy, or the workspace's copy alone when the session has
    /// not changed it. `Ok(None)` means the path does not exist in either
    /// layer, or the session deleted it.
    pub fn read(&self, path: &str) -> Result<Option<String>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        let view = self.view()?;
        self.current(&view, &norm)
    }

    /// Whether the session has changed `path` — written, edited or deleted it —
    /// so that what a read returns is no longer the lower layer's copy.
    pub fn is_modified(&self, path: &str) -> bool {
        self.holds(&Self::normalize(path))
    }

    /// Every path the session has changed, sorted.
    pub fn changed_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self.upper.read().unwrap().chains.keys().cloned().collect();
        paths.sort();
        paths
    }

    /// Whether the session layer holds a chain for `norm`.
    fn holds(&self, norm: &str) -> bool {
        self.upper.read().unwrap().chains.contains_key(norm)
    }

    /// `norm`'s content as this store resolves it: the session's chain
    /// replayed, or the lower layer's copy.
    ///
    /// A chain opening with an edit is replayed onto the lower layer's copy as
    /// it is now. When that copy has changed underneath the edit, the edit no
    /// longer fits it and the read is refused as [`VfsError::Diverged`] — the
    /// alternative is splicing the session's change into the wrong place.
    fn current(&self, view: &View<'_>, norm: &str) -> Result<Option<String>, VfsError> {
        let chain = self.upper.read().unwrap().chains.get(norm).cloned();
        let Some(chain) = chain else {
            return view.read_text(norm);
        };
        let base = match chain.deltas.first().map(|t| &t.delta) {
            Some(FileDelta::Edit { .. }) => view.read_text(norm)?,
            _ => None,
        };
        file_delta::replay(base, chain.deltas.iter().map(|t| &t.delta)).map_err(|e| match e {
            ReplayError::Diverged(_) => Self::diverged(norm),
            ReplayError::NotText => Self::not_text(norm),
        })
    }

    fn not_text(norm: &str) -> VfsError {
        VfsError::Unreadable(format!("{norm} is not valid UTF-8 text"))
    }

    /// The deltas this store holds for `path` — the session's chain, oldest
    /// first, each with the moment its operation executed — or `None` when the
    /// session has not changed it.
    pub fn deltas(&self, path: &str) -> Option<Vec<TimedDelta>> {
        let norm = Self::normalize(path);
        self.upper
            .read()
            .unwrap()
            .chains
            .get(&norm)
            .map(|c| c.deltas.clone())
    }

    /// When the session's chain for `path` began and when its latest change
    /// was made, or `None` when the session has not changed it.
    pub fn times(&self, path: &str) -> Option<FileTimes> {
        let norm = Self::normalize(path);
        FileTimes::of(&self.upper.read().unwrap().chains.get(&norm)?.deltas)
    }

    /// Apply `deltas` — made against the file as this store holds it now — to
    /// `path`, in order, recorded as extending the path's chain exactly as the
    /// edits and writes that made them would have.
    ///
    /// Each delta keeps the moment it was made. Checked before anything
    /// changes: deltas that do not fit the file as it stands are refused as
    /// [`VfsError::Diverged`] and the store is left as it was.
    pub fn apply(&self, path: &str, deltas: &[TimedDelta]) -> Result<(), VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        if deltas.is_empty() {
            return Ok(());
        }
        loop {
            let seen = Seen::of(&self.upper.read().unwrap(), &norm);
            let view = self.view()?;
            let replayed =
                file_delta::replay(self.current(&view, &norm)?, deltas.iter().map(|t| &t.delta));
            let result = replayed.map_err(|e| match e {
                ReplayError::Diverged(_) => VfsError::Diverged(format!(
                    "the changes do not fit {norm} as it stands — read it and build them again"
                )),
                ReplayError::NotText => Self::not_text(&norm),
            })?;
            let in_lower = view.is_file(&norm);
            let mut guard = self.upper.write().unwrap();
            if !seen.stands(&guard, &norm) {
                continue;
            }
            let conflict = Self::still_conflicted(&guard, &norm, result.as_deref());
            let mut chain = seen.chain.unwrap_or(Chain {
                deltas: Vec::new(),
                size: None,
                conflict: false,
            });
            chain.deltas.extend(deltas.iter().cloned());
            // A replace or a delete supersedes everything before it.
            if let Some(last) = chain.deltas.iter().rposition(|t| t.delta.supersedes()) {
                chain.deltas.drain(..last);
            }
            chain.size = result.as_ref().map(String::len);
            chain.conflict = conflict;
            // Deleting a file only the session made leaves nothing to record.
            if chain.size.is_none() && !in_lower {
                guard.chains.remove(&norm);
                return Ok(());
            }
            return Self::set_capped(&mut guard, norm, chain);
        }
    }

    /// The session's state as it is saved: every changed path's chain, each
    /// delta with its moment, the size it leaves the file at and whether it
    /// is in conflict, with the base they are made on — what
    /// [`Self::restore`] puts back exactly, with nothing replayed.
    pub fn snapshot(&self) -> Snapshot {
        Self::saved(&self.upper.read().unwrap())
    }

    fn saved(upper: &Upper) -> Snapshot {
        Snapshot {
            base: upper.base.as_ref().map(Base::saved),
            chains: upper
                .chains
                .iter()
                .map(|(path, chain)| {
                    let saved = SavedChain {
                        deltas: chain.deltas.clone(),
                        size: chain.size,
                        conflict: chain.conflict,
                    };
                    (path.clone(), saved)
                })
                .collect(),
        }
    }

    /// Put back what a [`Self::snapshot`] saved, replacing whatever the
    /// session layer holds — how a conversation's own copy of a repository
    /// outlives the store that held it, reading the base it read before. A
    /// protected path is refused, and so is a set past the size cap or a base
    /// that does not name objects; any of these leaves the store as it was.
    pub fn restore(&self, snapshot: Snapshot) -> Result<(), VfsError> {
        if snapshot.is_empty() {
            return Ok(());
        }
        let restored = Self::upper_of(snapshot)?;
        *self.upper.write().unwrap() = restored;
        Ok(())
    }

    /// Put back `before`, a [`Self::snapshot`] taken before a change, only if
    /// the store still holds exactly `after` — what that change left. Returns
    /// whether it did: a change of the conversation's that landed since is
    /// never undone by rolling back another.
    pub fn roll_back(&self, after: &Snapshot, before: Snapshot) -> Result<bool, VfsError> {
        let restored = Self::upper_of(before)?;
        let mut upper = self.upper.write().unwrap();
        if Self::saved(&upper) != *after {
            return Ok(false);
        }
        *upper = restored;
        Ok(true)
    }

    /// The session layer `snapshot` saved, checked: no protected path, within
    /// the size cap, a base naming objects.
    fn upper_of(snapshot: Snapshot) -> Result<Upper, VfsError> {
        let base = snapshot.base.as_ref().map(SavedBase::parse).transpose()?;
        let mut chains = HashMap::with_capacity(snapshot.chains.len());
        for (path, saved) in snapshot.chains {
            let norm = Self::normalize(&path);
            Self::guard(&norm)?;
            chains.insert(
                norm,
                Chain {
                    deltas: saved.deltas,
                    size: saved.size,
                    conflict: saved.conflict,
                },
            );
        }
        let restored = Upper { chains, base };
        if restored.bytes() > MAX_BYTES {
            return Err(VfsError::Full);
        }
        Ok(restored)
    }

    /// Every change the session holds, path by path, made against its base's
    /// tree — what a checkout replays onto a repository's files to put this
    /// conversation on disk (see [`crate::checkout`]).
    pub fn changes(&self) -> Result<FileChanges, VfsError> {
        self.view()?;
        let upper = self.upper.read().unwrap();
        let mut changes = FileChanges::new();
        for (path, chain) in &upper.chains {
            for delta in &chain.deltas {
                changes.push_timed(path.clone(), delta.clone());
            }
        }
        Ok(changes)
    }

    fn diverged(norm: &str) -> VfsError {
        VfsError::Diverged(format!(
            "{norm} has changed on disk since this conversation edited it, so the edit no \
             longer fits it — write the whole file to set its content"
        ))
    }

    /// Refusal of a lower-layer file above [`MAX_LOWER_FILE_BYTES`].
    fn too_large(norm: &str, bytes: u64) -> VfsError {
        VfsError::Unreadable(format!(
            "{norm} is {bytes} bytes, above the {MAX_LOWER_FILE_BYTES}-byte workspace read limit",
        ))
    }

    /// A file's bytes, resolved through the overlay the same way
    /// [`Self::read`] is, for a caller that records content rather than
    /// showing it — `git_commit`'s `take`. Neither the UTF-8 requirement nor
    /// [`MAX_LOWER_FILE_BYTES`] applies, since nothing here reaches a tool
    /// response; every rule on which file a path may open does, because it
    /// reads through the same lower layer every other read does.
    pub fn read_bytes(&self, path: &str) -> Result<Option<Vec<u8>>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        let view = self.view()?;
        if self.holds(&norm) {
            return Ok(self.current(&view, &norm)?.map(String::into_bytes));
        }
        view.read_bytes(&norm)
    }

    /// One [`PAGE_LINES`]-line page of a file, resolved through the overlay the
    /// same way [`Self::read`] is. `Ok(None)` means the path does not exist in
    /// either layer.
    ///
    /// A folder's file is streamed line by line rather than materialised
    /// (`file_read` used to read the whole file into a `String` and a
    /// `Vec<&str>` slice of it just to return 300 lines); a branch's blob,
    /// already bounded by [`MAX_LOWER_FILE_BYTES`], is paged from its text. A
    /// file the session changed is replayed from its deltas once and paged
    /// from that text.
    pub fn read_page(&self, path: &str, page: u32) -> Result<Option<PageResult>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        let view = self.view()?;
        if self.holds(&norm) {
            let Some(text) = self.current(&view, &norm)? else {
                return Ok(None);
            };
            return Self::paginate(std::io::Cursor::new(text.as_bytes()), page)
                .map(Some)
                .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")));
        }
        view.read_page(&norm, page)
    }

    /// One level of `dir`'s contents, upper layer shadowing the workspace:
    /// the files and subdirectories directly inside it, never a deeper file
    /// flattened up into its listing. Whiteouted paths are omitted. Sorted by
    /// path. `Ok(None)` means `dir` does not resolve to a directory in either
    /// layer — a missing path, or one that names a file.
    pub fn list_dir(&self, dir: &str) -> Result<Option<Vec<ListEntry>>, VfsError> {
        let norm = Self::normalize(dir);
        // A protected folder named directly lists as empty — never an error
        // that would confirm or deny it, and never its contents.
        if Self::is_protected(&norm) {
            return Ok(Some(Vec::new()));
        }
        let view = self.view()?;
        // `norm` resolving as a file rules out a directory listing outright —
        // checked first, and independently of the walk below, because a
        // conflicting write elsewhere in the upper layer (e.g. a file at
        // `"a/b.rs/c.rs"` alongside a real file at `"a/b.rs"`) would otherwise
        // make `upper_has_children` true for `"a/b.rs"` and this would
        // silently resolve a file path as a directory.
        if self.resolves_as_file(&view, &norm) {
            return Ok(None);
        }
        let mut out: Vec<ListEntry> = Vec::new();
        // Keyed on `(path, is_dir)`, not just `path`: a plain file at `"a"`
        // and another upper file at `"a/b.rs"` (which collapses to a
        // directory entry also named `"a"`) are two genuinely different
        // things sharing one name — deduping on the name alone would drop
        // whichever the map happened to iterate second, silently hiding it
        // from the listing rather than just looking odd.
        let mut seen: HashSet<(String, bool)> = HashSet::new();
        let mut upper_has_children = false;

        {
            let guard = self.upper.read().unwrap();
            for (k, chain) in guard.chains.iter() {
                // A deleted path shadows the workspace's entry and lists as
                // nothing.
                let Some(size) = chain.size else {
                    seen.insert((k.clone(), false));
                    continue;
                };
                let Some((child, is_dir)) = Self::immediate_child(&norm, k) else {
                    continue;
                };
                upper_has_children = true;
                if !seen.insert((child.clone(), is_dir)) {
                    continue;
                }
                out.push(if is_dir {
                    ListEntry {
                        path: child,
                        bytes: None,
                        dir: true,
                        modified: false,
                    }
                } else {
                    ListEntry {
                        path: child,
                        bytes: Some(size),
                        dir: false,
                        modified: true,
                    }
                });
            }
        }

        let lower_is_dir = view.is_dir(&norm);
        if !lower_is_dir && !upper_has_children && !norm.is_empty() {
            return Ok(None);
        }

        if lower_is_dir || norm.is_empty() {
            for (path, bytes, is_dir) in view.children(&norm) {
                if seen.contains(&(path.clone(), is_dir)) {
                    continue;
                }
                seen.insert((path.clone(), is_dir));
                out.push(ListEntry {
                    path,
                    bytes,
                    dir: is_dir,
                    modified: false,
                });
            }
        }

        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Some(out))
    }

    /// Whether `norm` resolves to a file in either layer — key/metadata
    /// existence only, never a content read (matching `list_dir`'s "never
    /// opens a file" contract). Upper-first, respecting whiteouts, the same
    /// precedence `read` uses.
    fn resolves_as_file(&self, view: &View<'_>, norm: &str) -> bool {
        if norm.is_empty() {
            return false;
        }
        if let Some(chain) = self.upper.read().unwrap().chains.get(norm) {
            return chain.size.is_some();
        }
        view.is_file(norm)
    }

    /// Remove a path from the overlay. A file the lower layer holds is
    /// recorded as one [`FileDelta::Delete`], superseding the path's earlier
    /// deltas, so it stops resolving; a file only the session made just loses
    /// its deltas. Returns whether the path resolved before the call. The
    /// lower layer is never touched.
    pub fn delete(&self, path: &str) -> bool {
        let norm = Self::normalize(path);
        if Self::is_protected(&norm) {
            return false;
        }
        let Ok(view) = self.view() else {
            return false;
        };
        let in_lower = view.is_file(&norm);
        let mut guard = self.upper.write().unwrap();
        if guard.deleted(&norm) {
            return false;
        }
        let had_upper = guard.chains.remove(&norm).is_some();
        if in_lower {
            guard.chains.insert(
                norm,
                Chain {
                    deltas: vec![TimedDelta::now(FileDelta::Delete)],
                    size: None,
                    conflict: false,
                },
            );
        }
        had_upper || in_lower
    }

    /// Every path the session has changed, sorted, and how — against the
    /// lower layer as it stands now: what a status of the conversation's
    /// uncommitted work reports.
    pub fn status(&self) -> Vec<(String, FileState)> {
        let Ok(view) = self.view() else {
            return Vec::new();
        };
        let upper = self.upper.read().unwrap();
        let mut out: Vec<(String, FileState)> = upper
            .chains
            .iter()
            .map(|(path, chain)| {
                let state = match (chain.size, view.is_file(path)) {
                    (None, _) => FileState::Deleted,
                    (Some(_), true) => FileState::Modified,
                    (Some(_), false) => FileState::Added,
                };
                (path.clone(), state)
            })
            .collect();
        out.sort();
        out
    }

    /// Drop every change the session holds: every path reads as the lower
    /// layer holds it again. What a hard reset does to the conversation's
    /// uncommitted work.
    pub fn discard(&self) {
        let mut upper = self.upper.write().unwrap();
        upper.chains.clear();
    }

    /// Bytes the session's deltas hold. Workspace files cost nothing — they
    /// are read on demand and never retained — and neither does the unchanged
    /// part of a file the session edited.
    pub fn total_bytes(&self) -> usize {
        self.upper.read().unwrap().bytes()
    }

    // ── Upper-layer helpers ──────────────────────────────────────────────────

    /// Install `chain` as `norm`'s, unless the session layer would then hold
    /// more than [`MAX_BYTES`] — refused as [`VfsError::Full`] with the layer
    /// left exactly as it was, a deletion included.
    fn set_capped(upper: &mut Upper, norm: String, chain: Chain) -> Result<(), VfsError> {
        let old = upper.chains.get(&norm).map_or(0, Chain::bytes);
        if upper.bytes() - old + chain.bytes() > MAX_BYTES {
            return Err(VfsError::Full);
        }
        upper.chains.insert(norm, chain);
        Ok(())
    }

    // ── Path rules ───────────────────────────────────────────────────────────

    /// Whether every segment of `norm` is a name a folder opens as itself — no
    /// `:`, no trailing `.` or space, no 8.3 short-name tail.
    ///
    /// The workspace walks apply it too, so a listing never shows a file the
    /// store cannot open. A Linux workspace can hold `logs/12:00.txt` or
    /// `notes~2.md`; listed but refused by every read, the model was shown a
    /// file it could name and never read.
    pub(crate) fn addressable(norm: &str) -> bool {
        !norm
            .split('/')
            .any(|s| s.contains(':') || s.ends_with('.') || s.ends_with(' ') || is_short_name(s))
    }

    /// Whether a normalised key names something under a protected directory.
    ///
    /// Compared the way Windows resolves a name — case-insensitively, with
    /// trailing dots and spaces dropped — so `Secrets/`, `SECRETS/` and
    /// `secrets./` are the directory they open, not three unprotected ones.
    pub fn is_protected(norm: &str) -> bool {
        norm.split('/').any(|s| {
            let s = s.trim_end_matches(['.', ' ']);
            PROTECTED_SEGMENTS.iter().any(|p| s.eq_ignore_ascii_case(p))
        })
    }

    /// `Err(Forbidden)` for a protected key, `Ok(())` otherwise.
    fn guard(norm: &str) -> Result<(), VfsError> {
        if Self::is_protected(norm) {
            return Err(VfsError::Forbidden(norm.to_string()));
        }
        Ok(())
    }

    /// Stream `reader` line by line and return page `requested_page`
    /// ([`PAGE_LINES`]-line strides, 0-based), without ever holding more than
    /// two pages' worth of lines at once.
    ///
    /// A request past the end clamps to the last page — the same "over-shoot
    /// reads as the tail, not an error" rule the file tools' paging applies to
    /// `file_list` — which is why the last [`PAGE_LINES`] lines are
    /// held in `tail` the whole way through: by the time EOF says the request
    /// was out of range, the candidate page's lines are long gone, and a
    /// second pass would cost exactly the whole-file read this exists to avoid.
    fn paginate(
        mut reader: impl std::io::BufRead,
        requested_page: u32,
    ) -> std::io::Result<PageResult> {
        let want_start = u64::from(requested_page) * u64::from(PAGE_LINES);
        let want_end = want_start + u64::from(PAGE_LINES);

        let mut candidate: Vec<String> = Vec::new();
        let mut tail: std::collections::VecDeque<String> =
            std::collections::VecDeque::with_capacity(PAGE_LINES as usize + 1);
        let mut total_lines: u64 = 0;
        let mut buf = String::new();
        loop {
            buf.clear();
            // `read_line` requires valid UTF-8 (an `io::Error` otherwise), which
            // is the same "the whole file must be text" contract a whole-file
            // read enforces up front — just discovered while streaming instead of
            // before returning anything.
            let n = reader.read_line(&mut buf)?;
            if n == 0 {
                break;
            }
            // Split on '\n' only, matching `file_read`'s old `content.split('\n')`
            // exactly: a trailing '\r' on a CRLF file stays part of the line.
            let line = buf.strip_suffix('\n').unwrap_or(&buf).to_string();
            if total_lines >= want_start && total_lines < want_end {
                candidate.push(line.clone());
            }
            tail.push_back(line);
            if tail.len() as u32 > PAGE_LINES {
                tail.pop_front();
            }
            total_lines += 1;
        }

        if total_lines == 0 {
            return Ok(PageResult {
                page: 0,
                start_line: 1,
                end_line: 0,
                total_lines: 0,
                total_pages: 0,
                body: String::new(),
            });
        }

        let total_pages = ((total_lines - 1) / u64::from(PAGE_LINES) + 1) as u32;
        let (page, start_line_0, lines) = if want_start < total_lines {
            (requested_page, want_start, candidate)
        } else {
            // `tail` holds the last (up to) `PAGE_LINES` lines seen, which is
            // wider than the last page whenever the file's length is not an
            // exact multiple of the stride — take only the suffix the last
            // page actually covers, not the whole buffer.
            let last_page = total_pages - 1;
            let last_start = u64::from(last_page) * u64::from(PAGE_LINES);
            let need = (total_lines - last_start) as usize;
            let skip = tail.len().saturating_sub(need);
            (
                last_page,
                last_start,
                tail.into_iter().skip(skip).collect::<Vec<_>>(),
            )
        };

        let start_line = (start_line_0 + 1) as u32;
        let end_line = start_line + (lines.len() as u32).saturating_sub(1);
        Ok(PageResult {
            page,
            start_line,
            end_line,
            total_lines: total_lines as u32,
            total_pages,
            body: lines.join("\n"),
        })
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Every path visible under `prefix`, upper layer shadowing the lower.
    ///
    /// Unlike [`VfsStore::list_dir`] this reads no file contents, so it stays
    /// cheap over a whole repository — line counts are what make a full
    /// listing expensive, and a path search does not need them.
    pub fn paths(&self, prefix: &str) -> Vec<String> {
        match self.view() {
            Ok(view) => self.paths_in(&view, prefix),
            Err(_) => Vec::new(),
        }
    }

    /// [`Self::paths`], against `view`.
    fn paths_in(&self, view: &View<'_>, prefix: &str) -> Vec<String> {
        let norm_prefix = Self::normalize(prefix);
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();
        {
            let guard = self.upper.read().unwrap();
            for (k, chain) in guard.chains.iter() {
                seen.insert(k.clone());
                if chain.size.is_some()
                    && Self::matches_prefix(k, &norm_prefix)
                    && !Self::is_protected(k)
                {
                    out.push(k.clone());
                }
            }
        }
        for path in view.files_under(&norm_prefix) {
            if !seen.contains(&path) {
                out.push(path);
            }
        }
        out.sort();
        out
    }

    /// Scan file contents under `prefix` for `re`.
    ///
    /// Files that cannot be scanned — oversize, not UTF-8, vanished between the
    /// walk and the read — are skipped rather than failing the pass: a single
    /// binary blob in a tree must not turn a whole search into an error.
    pub fn grep(
        &self,
        re: &regex::Regex,
        prefix: &str,
        max_per_file: usize,
        max_total: usize,
    ) -> GrepOutcome {
        let mut out = GrepOutcome::default();
        let Ok(view) = self.view() else {
            return out;
        };
        for path in self.paths_in(&view, prefix) {
            let modified = self.holds(&path);
            let Ok(Some(content)) = self.current(&view, &path) else {
                continue;
            };
            out.files_searched += 1;

            let mut in_file = 0usize;
            for (idx, line) in content.lines().enumerate() {
                if !re.is_match(line) {
                    continue;
                }
                if out.hits.len() >= max_total {
                    out.truncated = true;
                    return out;
                }
                out.hits.push(GrepHit {
                    path: path.clone(),
                    line_no: idx as u32 + 1,
                    line: line.trim_end().to_string(),
                    modified,
                });
                in_file += 1;
                if in_file >= max_per_file {
                    // One file monopolising the budget would hide every other
                    // file that matches, which is the answer the caller wants.
                    out.truncated = true;
                    break;
                }
            }
        }
        out
    }

    // ── Path handling ────────────────────────────────────────────────────────

    /// What a one-level listing of `dir` shows for `key`, given `key` lives
    /// somewhere under `dir` (or `dir` is the empty root prefix): `key` itself
    /// when it is a direct child, or the immediate subdirectory's own path when
    /// `key` is deeper — collapsed rather than flattened all the way down, the
    /// same way a real directory listing never expands a subdirectory's
    /// contents into itself. `None` when `key` is not under `dir` at all.
    fn immediate_child(dir: &str, key: &str) -> Option<(String, bool)> {
        let remainder = if dir.is_empty() {
            key
        } else {
            key.strip_prefix(dir)?.strip_prefix('/')?
        };
        if remainder.is_empty() {
            return None;
        }
        Some(match remainder.find('/') {
            None => (key.to_string(), false),
            Some(slash) => {
                let first = &remainder[..slash];
                let path = if dir.is_empty() {
                    first.to_string()
                } else {
                    format!("{dir}/{first}")
                };
                (path, true)
            }
        })
    }

    /// `true` when `key` is under `prefix`. Plain string-prefix semantics, as
    /// the search tools' `prefix` parameter documents — so `src/` and `src`
    /// and even the partial `src/ma` all select `src/main.rs`. An empty
    /// prefix matches everything.
    fn matches_prefix(key: &str, prefix: &str) -> bool {
        prefix.is_empty() || key.starts_with(prefix)
    }

    /// Canonical overlay key for a caller-supplied path. See the module docs.
    pub fn normalize(path: &str) -> String {
        let path = path.trim_start_matches('/');
        let mut parts: Vec<&str> = Vec::new();
        for segment in path.split(['/', '\\']) {
            match segment {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                s => parts.push(s),
            }
        }
        parts.join("/")
    }
}

/// Whether `segment` has the shape of a Windows 8.3 short name's tilde tail —
/// a `~` followed by a digit (`SECRET~1`, `PROGRA~2.TXT`).
fn is_short_name(segment: &str) -> bool {
    segment
        .as_bytes()
        .windows(2)
        .any(|w| w[0] == b'~' && w[1].is_ascii_digit())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    use tempfile::TempDir;

    fn store_with_tree() -> (TempDir, VfsStore) {
        let dir = tempfile::tempdir().unwrap();
        put(dir.path(), "README.md", "# project\n");
        put(dir.path(), "src/main.rs", "fn main() {}\n");
        put(dir.path(), "src/util/helper.rs", "pub fn h() {}\n");
        let store = VfsStore::with_root(dir.path());
        (dir, store)
    }

    fn put(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    fn listed(store: &VfsStore, dir: &str) -> Vec<String> {
        store
            .list_dir(dir)
            .unwrap()
            .unwrap_or_default()
            .into_iter()
            .map(|e| e.path)
            .collect()
    }

    // ── the changes a checkout replays ───────────────────────────────────────

    /// **`changes` is every chain the session holds, delta for delta** — and
    /// replayed onto the workspace's copies it gives exactly what the store
    /// reads.
    #[test]
    fn changes_are_every_chain_and_replay_to_what_the_store_reads() {
        let (dir, store) = store_with_tree();
        store.edit("README.md", "# project\nmore\n".into()).unwrap();
        store.write("src/new.rs", "new\n".into()).unwrap();
        store.delete("src/main.rs");

        let changes = store.changes().unwrap();
        assert_eq!(
            changes.paths().collect::<Vec<_>>(),
            ["README.md", "src/main.rs", "src/new.rs"]
        );
        for path in ["README.md", "src/main.rs", "src/new.rs"] {
            assert_eq!(changes.chain(path).unwrap(), store.deltas(path).unwrap());
            let lower = std::fs::read(dir.path().join(path)).ok();
            assert_eq!(
                changes.replay(path, lower).unwrap(),
                store.read(path).unwrap().map(String::into_bytes),
                "{path}"
            );
        }
        assert!(VfsStore::new().changes().unwrap().is_empty());
    }

    // ── saving and restoring ─────────────────────────────────────────────────

    /// **A snapshot restores exactly, through its wire form**: a new store
    /// over the same folder reads what the old one did — an edit's chain, a
    /// write, a deletion — with every delta's moment kept and nothing
    /// replayed to get there.
    #[test]
    fn a_snapshot_restores_exactly_through_its_wire_form() {
        let (dir, store) = store_with_tree();
        store.edit("README.md", "# project\nmore\n".into()).unwrap();
        store
            .edit("README.md", "# project\nmore\nand more\n".into())
            .unwrap();
        store.write("src/new.rs", "new\n".into()).unwrap();
        store.delete("src/main.rs");

        let wire = serde_json::to_string(&store.snapshot()).unwrap();
        let back: Snapshot = serde_json::from_str(&wire).unwrap();
        assert_eq!(back, store.snapshot());

        let restored = VfsStore::with_root(dir.path());
        restored.restore(back).unwrap();
        for path in [
            "README.md",
            "src/new.rs",
            "src/main.rs",
            "src/util/helper.rs",
        ] {
            assert_eq!(
                restored.read(path).unwrap(),
                store.read(path).unwrap(),
                "{path}"
            );
            assert_eq!(restored.deltas(path), store.deltas(path), "{path}");
        }
        assert_eq!(listed(&restored, "src"), listed(&store, "src"));
        assert_eq!(restored.total_bytes(), store.total_bytes());
    }

    /// **Restoring replaces what the store held**, and an empty snapshot
    /// leaves an empty store empty.
    #[test]
    fn restoring_replaces_the_session_layer() {
        let (dir, store) = store_with_tree();
        store.write("a.txt", "a\n".into()).unwrap();
        let saved = store.snapshot();
        let other = VfsStore::with_root(dir.path());
        other.write("b.txt", "b\n".into()).unwrap();
        other.restore(saved).unwrap();
        assert!(other.is_modified("a.txt"));
        assert!(!other.is_modified("b.txt"));
        assert!(VfsStore::new().snapshot().is_empty());
        let empty = VfsStore::new();
        empty.restore(Snapshot::default()).unwrap();
        assert!(empty.snapshot().is_empty());
    }

    /// **A snapshot naming a protected path, or past the cap, is refused**
    /// and the store is left as it was.
    #[test]
    fn a_bad_snapshot_is_refused_whole() {
        let (_dir, store) = store_with_tree();
        store.write("kept.txt", "kept\n".into()).unwrap();
        let protected: Snapshot =
            serde_json::from_str(r#"{"chains":{"secrets/key.txt":{"deltas":[],"size":1}}}"#)
                .unwrap();
        assert!(matches!(
            store.restore(protected),
            Err(VfsError::Forbidden(_))
        ));
        let huge = "x".repeat(MAX_BYTES + 1);
        let too_big: Snapshot = serde_json::from_value(serde_json::json!({
            "chains": {
                "big.txt": {
                    "deltas": [{ "at_ns": 1, "kind": "replace", "content": huge }],
                    "size": MAX_BYTES + 1
                }
            }
        }))
        .unwrap();
        let no_tree: Snapshot = serde_json::from_value(serde_json::json!({
            "base": { "tree": "not an object id" },
            "chains": { "a.txt": { "deltas": [], "size": 1 } }
        }))
        .unwrap();
        assert!(matches!(
            store.restore(no_tree),
            Err(VfsError::Unreadable(_))
        ));
        assert!(
            serde_json::from_str::<Snapshot>(r#"{"a.txt":{"deltas":[],"size":1}}"#).is_err(),
            "a map of paths with no `chains` is not a snapshot"
        );
        assert!(matches!(store.restore(too_big), Err(VfsError::Full)));
        assert!(
            store.is_modified("kept.txt"),
            "the store was left as it was"
        );
    }

    // ── links ────────────────────────────────────────────────────────────────

    /// Link folder `link` to `target`: a symlink on Unix, a junction on
    /// Windows, which needs no privilege to create.
    fn link_dir(target: &Path, link: &Path) {
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, link).unwrap();
        #[cfg(windows)]
        {
            let out = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "mklink /J failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    /// **A linked folder leading outside the root is invisible and
    /// unwritable.** The daemon's secrets live outside every repository, so a
    /// link from a repository to them is the one way the file tools could
    /// reach them.
    #[test]
    fn a_folder_link_leading_outside_the_root_is_not_read_listed_searched_or_written() {
        let outside = tempfile::tempdir().unwrap();
        put(
            outside.path(),
            "secrets.yaml",
            "tavily_api_key: tvly-live\n",
        );
        let root = tempfile::tempdir().unwrap();
        put(root.path(), "src/lib.rs", "fn f() {}\n");
        link_dir(outside.path(), &root.path().join("escape"));

        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("escape/secrets.yaml").unwrap(), None);
        assert_eq!(s.read_bytes("escape/secrets.yaml").unwrap(), None);
        assert!(s.read_page("escape/secrets.yaml", 0).unwrap().is_none());
        assert_eq!(s.list_dir("escape").unwrap(), None);
        assert!(!listed(&s, "").contains(&"escape".to_string()));
        assert!(s.paths("escape").is_empty());
        let re = regex::Regex::new("tvly").unwrap();
        assert!(s.grep(&re, "", 10, 10).hits.is_empty());

        let _ = s.write("escape/planted.txt", "x".into());
        assert!(!outside.path().join("planted.txt").exists());
        assert!(!s.delete("escape/secrets.yaml"));
        assert!(outside.path().join("secrets.yaml").exists());
    }

    /// A link into the repository's own protected folder is refused too —
    /// the protection is on where a path leads, not how it is spelled.
    #[test]
    fn a_folder_link_into_a_protected_folder_is_refused() {
        let root = tempfile::tempdir().unwrap();
        put(
            root.path(),
            "secrets/tools.yaml",
            "tavily_api_key: tvly-live\n",
        );
        link_dir(&root.path().join("secrets"), &root.path().join("innocent"));
        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("innocent/tools.yaml").unwrap(), None);
        assert_eq!(s.read_bytes("innocent/tools.yaml").unwrap(), None);
        assert_eq!(s.list_dir("innocent").unwrap(), None);
    }

    /// **Bytes read the way text does, minus the text rules.** A binary file
    /// comes back whole; a drive-qualified path, a protected path and a
    /// folder do not come back at all.
    #[test]
    fn read_bytes_returns_any_file_and_only_a_file_under_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("logo.bin"), [0u8, 159, 146, 150, 255]).unwrap();
        put(root.path(), "src/lib.rs", "fn f() {}\n");
        put(root.path(), "secrets/key.yaml", "k: v\n");
        let outside = tempfile::tempdir().unwrap();
        put(outside.path(), "id_rsa", "private\n");
        let s = VfsStore::with_root(root.path());

        assert_eq!(
            s.read_bytes("logo.bin").unwrap(),
            Some(vec![0u8, 159, 146, 150, 255])
        );
        assert!(s.read("logo.bin").is_err(), "text reads still refuse it");
        assert_eq!(s.read_bytes("src").unwrap(), None);
        assert!(matches!(
            s.read_bytes("secrets/key.yaml"),
            Err(VfsError::Forbidden(_))
        ));
        let abs = outside.path().join("id_rsa");
        let spelled = abs.to_string_lossy().replace('\\', "/");
        assert_eq!(s.read_bytes(&spelled).unwrap(), None, "{spelled}");
    }

    /// A link that stays inside the root, and out of protected folders,
    /// still works.
    #[test]
    fn a_folder_link_inside_the_root_still_reads() {
        let root = tempfile::tempdir().unwrap();
        put(root.path(), "real/a.txt", "hello\n");
        link_dir(&root.path().join("real"), &root.path().join("alias"));
        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("alias/a.txt").unwrap().as_deref(), Some("hello\n"));
    }

    /// File symlinks, which Windows only creates with privilege.
    #[cfg(unix)]
    #[test]
    fn a_file_symlink_to_outside_or_to_a_protected_file_is_refused() {
        use std::os::unix::fs::symlink;
        let outside = tempfile::tempdir().unwrap();
        put(
            outside.path(),
            "secrets.yaml",
            "tavily_api_key: tvly-live\n",
        );
        let root = tempfile::tempdir().unwrap();
        put(root.path(), "secrets/tools.yaml", "key: live\n");
        put(root.path(), "ok.txt", "fine\n");
        symlink(
            outside.path().join("secrets.yaml"),
            root.path().join("notes.md"),
        )
        .unwrap();
        symlink(
            root.path().join("secrets/tools.yaml"),
            root.path().join("cfg.yaml"),
        )
        .unwrap();
        symlink(root.path().join("ok.txt"), root.path().join("ok-link.txt")).unwrap();
        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("notes.md").unwrap(), None);
        assert_eq!(s.read("cfg.yaml").unwrap(), None);
        assert_eq!(s.read("ok-link.txt").unwrap().as_deref(), Some("fine\n"));
    }

    /// **The guards hold.** A protected path is refused and left untouched,
    /// and `..` cannot climb out of the workspace — normalisation pins it to
    /// the root, so the write lands on the root's own key.
    #[test]
    fn a_store_cannot_touch_secrets_or_leave_the_root() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        put(&root, "secrets/tools.yaml", "key: real\n");
        let s = VfsStore::with_root(&root);

        assert!(matches!(
            s.write("secrets/tools.yaml", "key: planted\n".into()),
            Err(VfsError::Forbidden(_))
        ));
        assert!(!s.delete("secrets/tools.yaml"));
        assert_eq!(
            std::fs::read_to_string(root.join("secrets/tools.yaml")).unwrap(),
            "key: real\n"
        );

        s.write("../../escaped.txt", "x".into()).unwrap();
        assert!(!outer.path().join("escaped.txt").exists());
        assert_eq!(s.read("escaped.txt").unwrap().as_deref(), Some("x"));
    }

    /// **A host path is not a workspace path.** An absolute path to a file
    /// outside the workspace — on Windows a drive-prefixed one, which a join
    /// would have taken in place of the root — reads nothing and writes nothing
    /// there, and neither does a drive-relative path or an alternate stream.
    #[test]
    fn a_host_path_cannot_reach_outside_the_workspace() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        std::fs::create_dir_all(&root).unwrap();
        put(outer.path(), "outside.txt", "host secret\n");
        let outside = outer.path().join("outside.txt");
        let planted = outer.path().join("planted.txt");

        let overlay = VfsStore::with_root(&root);
        assert_eq!(
            overlay.read(&outside.to_string_lossy()).unwrap(),
            None,
            "an absolute host path reads nothing"
        );
        assert!(listed(&overlay, &outer.path().to_string_lossy()).is_empty());
        let _ = overlay.write(&planted.to_string_lossy(), "x".into());
        assert!(!planted.exists(), "a write left the workspace");
        for path in [
            "C:x.txt",
            "C:/x.txt",
            r"\\?\C:\x.txt",
            "notes.txt:hidden",
            "SECRET~1/tools.yaml",
            "docs~2/x.md",
        ] {
            assert!(
                !matches!(overlay.read(path), Ok(Some(_))),
                "{path:?} read through to the host"
            );
        }
    }

    /// **Another spelling of `secrets/` is still `secrets/`.** Windows opens a
    /// name case-insensitively and drops trailing dots and spaces, so each of
    /// these reaches the protected directory there and is refused everywhere.
    #[test]
    fn every_spelling_of_the_protected_directory_is_protected() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        put(&root, "secrets/tools.yaml", "key: real\n");
        let s = VfsStore::with_root(&root);
        for path in [
            "Secrets/tools.yaml",
            "SECRETS/tools.yaml",
            "secrets./tools.yaml",
            "secrets /tools.yaml",
        ] {
            assert!(
                VfsStore::is_protected(&VfsStore::normalize(path)),
                "{path:?}"
            );
            assert!(s.write(path, "key: planted\n".into()).is_err(), "{path:?}");
            assert!(
                !matches!(s.read(path), Ok(Some(_))),
                "{path:?} read the protected file"
            );
        }
        assert_eq!(
            std::fs::read_to_string(root.join("secrets/tools.yaml")).unwrap(),
            "key: real\n"
        );
        assert!(!VfsStore::is_protected("docs/secretsauce.md"));
    }

    /// **A repository's `.git` is protected, at any depth and by any spelling.**
    ///
    /// Two independent reasons, both measured. `.git/config` holds a remote's
    /// URL with its credentials intact — the git layer redacts exactly that, so
    /// leaving this readable would make the redaction decorative. And on a live
    /// turn the model bypassed `git_refs`, hand-parsed `.git/packed-refs`, and
    /// answered 53 branches and 23 tags where the truth was 56 and 24.
    #[test]
    fn a_repositorys_git_database_is_protected() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        put(
            &root,
            ".git/config",
            "[remote \"origin\"]\n\turl = https://u:ghp_secrettoken@example.com/a.git\n",
        );
        put(&root, ".git/packed-refs", "abc refs/heads/main\n");
        put(&root, "src/lib.rs", "pub fn ok() {}\n");
        let s = VfsStore::with_root(&root);

        for path in [
            ".git/config",
            ".git/packed-refs",
            ".git/refs/heads/main",
            ".GIT/config",
            ".git./config",
            "nested/repo/.git/config",
        ] {
            assert!(
                VfsStore::is_protected(&VfsStore::normalize(path)),
                "{path:?} is not protected"
            );
            assert!(
                !matches!(s.read(path), Ok(Some(_))),
                "{path:?} was readable"
            );
            assert!(
                s.write(path, "planted".into()).is_err(),
                "{path:?} was written"
            );
        }

        // The token never reaches a caller by any route.
        assert!(s.read(".git/config").is_err());
        let listed = s.list_dir("").unwrap().unwrap_or_default();
        assert!(
            !format!("{listed:?}").contains(".git"),
            "a listing named the git database: {listed:?}"
        );
        let re = regex::Regex::new("ghp_secrettoken").unwrap();
        let hits = s.grep(&re, "", 50, 20);
        assert!(
            !format!("{hits:?}").contains("ghp_secrettoken"),
            "grep reached into .git: {hits:?}"
        );

        // A file merely *called* something gitish is untouched.
        assert!(!VfsStore::is_protected("docs/.gitignore"));
        assert!(!VfsStore::is_protected("src/gitmodules.rs"));
        assert!(matches!(s.read("src/lib.rs"), Ok(Some(_))));
    }

    /// **A listing shows only files a read can open.** `notes~2.md` has the
    /// shape of a Windows short name, so reads refuse it — and the listing,
    /// the path search and grep leave it out rather than show a file that
    /// cannot be read.
    #[test]
    fn a_file_the_store_cannot_open_is_not_listed() {
        let (dir, store) = store_with_tree();
        put(dir.path(), "notes~2.md", "tilde\n");
        assert_eq!(store.read("notes~2.md").unwrap(), None);
        assert!(!listed(&store, "").contains(&"notes~2.md".to_string()));
        assert!(!store.paths("").contains(&"notes~2.md".to_string()));
        assert!(store.paths("").contains(&"src/main.rs".to_string()));
    }

    #[test]
    fn a_short_name_is_a_tilde_and_a_digit() {
        assert!(is_short_name("SECRET~1"));
        assert!(is_short_name("PROGRA~2.TXT"));
        assert!(!is_short_name("notes~"));
        assert!(!is_short_name("~draft.md"));
        assert!(!is_short_name("plain.rs"));
    }

    /// **A store never touches the disk** — a write, an overwrite and a
    /// delete all land in the session's own layer.
    #[test]
    fn a_write_never_reaches_disk() {
        let (dir, s) = store_with_tree();
        s.write("new.txt", "x".into()).unwrap();
        s.write("README.md", "changed".into()).unwrap();
        assert!(s.delete("src/main.rs"));
        assert!(!dir.path().join("new.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("README.md")).unwrap(),
            "# project\n"
        );
        assert!(dir.path().join("src/main.rs").exists());
    }

    // ── normalize ────────────────────────────────────────────────────────────

    #[test]
    fn normalize_collapses_to_one_canonical_key() {
        for spelling in [
            "src/main.rs",
            "/src/main.rs",
            "./src/main.rs",
            "src/./main.rs",
            "src//main.rs",
            "src/util/../main.rs",
            "/./src/../src/main.rs",
        ] {
            assert_eq!(
                VfsStore::normalize(spelling),
                "src/main.rs",
                "spelling {spelling:?}",
            );
        }
    }

    /// Windows-style separators are accepted, so a model echoing a path back from
    /// a Windows-hosted daemon still addresses the same entry.
    #[test]
    fn normalize_accepts_backslash_separators() {
        assert_eq!(VfsStore::normalize(r"src\main.rs"), "src/main.rs");
        assert_eq!(VfsStore::normalize(r"\src\main.rs"), "src/main.rs");
    }

    /// `..` can never climb above the root: popping an empty stack is a no-op, so
    /// a traversal attempt lands back inside the workspace.
    #[test]
    fn normalize_cannot_escape_the_root() {
        assert_eq!(VfsStore::normalize("../../../etc/passwd"), "etc/passwd");
        assert_eq!(VfsStore::normalize("/../../.."), "");
        assert_eq!(VfsStore::normalize(".."), "");
    }

    /// **No segment is special.** A folder named `workspace` is an ordinary
    /// folder wherever it sits, and the root spells as the empty key.
    #[test]
    fn normalize_treats_every_segment_as_a_plain_name() {
        assert_eq!(
            VfsStore::normalize("workspace/src/a.rs"),
            "workspace/src/a.rs"
        );
        assert_eq!(VfsStore::normalize("/workspace"), "workspace");
        assert_eq!(VfsStore::normalize("/"), "");
        assert_eq!(VfsStore::normalize(""), "");
    }

    #[test]
    fn immediate_child_collapses_deeper_paths_to_their_subdirectory() {
        assert_eq!(
            VfsStore::immediate_child("", "README.md"),
            Some(("README.md".to_string(), false)),
        );
        assert_eq!(
            VfsStore::immediate_child("", "src/main.rs"),
            Some(("src".to_string(), true)),
        );
        assert_eq!(
            VfsStore::immediate_child("src", "src/main.rs"),
            Some(("src/main.rs".to_string(), false)),
        );
        assert_eq!(
            VfsStore::immediate_child("src", "src/util/helper.rs"),
            Some(("src/util".to_string(), true)),
        );
        assert_eq!(
            VfsStore::immediate_child("src", "src/util/deep/leaf.rs"),
            Some(("src/util".to_string(), true)),
            "three levels down still collapses to the immediate subdirectory, not the leaf",
        );
    }

    /// A directory boundary, not a string prefix: `src` must not falsely match
    /// `srcx/…` the way a plain `starts_with` would.
    #[test]
    fn immediate_child_respects_the_directory_boundary() {
        assert_eq!(VfsStore::immediate_child("src", "srcx/file.rs"), None);
        assert_eq!(VfsStore::immediate_child("src", "docs/readme.md"), None);
        assert_eq!(
            VfsStore::immediate_child("src", "src"),
            None,
            "src is dir itself, not a child of it"
        );
    }

    // ── Upper layer alone ────────────────────────────────────────────────────

    #[test]
    fn upper_only_store_round_trips_and_accounts_bytes() {
        let s = VfsStore::new();
        assert_eq!(s.total_bytes(), 0);
        assert!(s.root().is_none());

        assert!(
            s.write("a.txt", "hello".into()).unwrap(),
            "first write creates"
        );
        assert_eq!(s.read("a.txt").unwrap().as_deref(), Some("hello"));
        assert_eq!(s.total_bytes(), 5);

        assert!(
            !s.write("a.txt", "hi".into()).unwrap(),
            "second write overwrites",
        );
        assert_eq!(
            s.total_bytes(),
            2,
            "overwriting with less must release budget",
        );
        assert_eq!(s.read("missing.txt").unwrap(), None);
    }

    #[test]
    fn upper_only_delete_removes_without_leaving_a_whiteout() {
        let s = VfsStore::new();
        s.write("a.txt", "x".into()).unwrap();
        assert!(s.delete("a.txt"));
        assert_eq!(s.read("a.txt").unwrap(), None);
        assert_eq!(s.total_bytes(), 0);
        assert!(!s.delete("a.txt"), "second delete finds nothing");
        // With nothing below to hide, the path is simply gone — a later write is
        // an ordinary creation.
        assert!(s.write("a.txt", "y".into()).unwrap());
    }

    #[test]
    fn different_spellings_address_one_entry() {
        let s = VfsStore::new();
        s.write("/src/main.rs", "one".into()).unwrap();
        s.write("src/main.rs", "two".into()).unwrap();
        assert_eq!(
            s.read("./src/../src/main.rs").unwrap().as_deref(),
            Some("two")
        );
        assert_eq!(listed(&s, ""), vec!["src"], "one collapsed entry, not two");
        assert_eq!(listed(&s, "src"), vec!["src/main.rs"]);
        assert_eq!(s.total_bytes(), 3, "one entry, not two");
    }

    // ── Capacity ─────────────────────────────────────────────────────────────

    #[test]
    fn write_beyond_the_cap_is_rejected_and_changes_nothing() {
        let s = VfsStore::new();
        s.write("big.bin", "x".repeat(MAX_BYTES - 10)).unwrap();
        let before = s.total_bytes();

        let err = s.write("more.bin", "y".repeat(64)).unwrap_err();
        assert!(matches!(err, VfsError::Full));
        assert_eq!(s.total_bytes(), before, "a rejected write must not consume");
        assert_eq!(s.read("more.bin").unwrap(), None);

        // What does fit still succeeds.
        s.write("more.bin", "y".repeat(10)).unwrap();
        assert_eq!(s.total_bytes(), MAX_BYTES);
    }

    /// A write that the cap rejects must leave the overlay exactly as it was —
    /// including a whiteout it was about to clear. Otherwise a failed write
    /// resurrects a file the session had deleted.
    #[test]
    fn a_rejected_write_does_not_clear_a_whiteout() {
        let (_dir, s) = store_with_tree();
        s.write("filler.bin", "x".repeat(MAX_BYTES - 10)).unwrap();
        assert!(s.delete("README.md"), "whiteout the lower file");
        assert_eq!(s.read("README.md").unwrap(), None);

        let err = s.write("README.md", "y".repeat(4096)).unwrap_err();
        assert!(matches!(err, VfsError::Full), "{err:?}");

        assert_eq!(
            s.read("README.md").unwrap(),
            None,
            "the failed write must not have resurrected the workspace file",
        );
        assert!(!listed(&s, "").contains(&"README.md".to_string()));
    }

    /// The cap counts the *replacement*, not the sum: overwriting a large file
    /// with a large file is fine even though their total exceeds the budget.
    #[test]
    fn overwrite_is_measured_against_the_slot_it_replaces() {
        let s = VfsStore::new();
        s.write("big.bin", "x".repeat(MAX_BYTES - 100)).unwrap();
        s.write("big.bin", "y".repeat(MAX_BYTES - 100)).unwrap();
        assert_eq!(s.total_bytes(), MAX_BYTES - 100);
    }

    // ── Lower layer: read-through ────────────────────────────────────────────

    #[test]
    fn read_falls_through_and_costs_no_budget() {
        let (_dir, s) = store_with_tree();
        assert_eq!(
            s.read("src/main.rs").unwrap().as_deref(),
            Some("fn main() {}\n")
        );
        assert_eq!(
            s.read("/src/util/helper.rs").unwrap().as_deref(),
            Some("pub fn h() {}\n"),
        );
        assert_eq!(s.total_bytes(), 0, "reading through retains nothing");
        assert_eq!(s.read("src/nope.rs").unwrap(), None);
    }

    /// A directory resolves as absent rather than erroring — `read` answers about
    /// files.
    #[test]
    fn reading_a_directory_path_is_absent() {
        let (_dir, s) = store_with_tree();
        assert_eq!(s.read("src").unwrap(), None);
        assert_eq!(s.read("src/util").unwrap(), None);
        assert_eq!(s.read("").unwrap(), None, "the root itself is not a file");
    }

    // ── read_page ────────────────────────────────────────────────────────────

    /// `n` lines, each its own line number so a page's content is checkable
    /// without re-deriving it — line `k` (1-based) is the string `k`.
    fn numbered_lines(n: u32) -> String {
        (1..=n)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    #[test]
    fn a_short_file_is_entirely_page_0() {
        let (_dir, s) = store_with_tree();
        let p = s.read_page("src/main.rs", 0).unwrap().unwrap();
        assert_eq!(p.page, 0);
        assert_eq!(p.total_pages, 1);
        assert_eq!(p.start_line, 1);
        assert_eq!(p.end_line, 1);
        assert_eq!(p.total_lines, 1);
        assert_eq!(p.body, "fn main() {}");
    }

    #[test]
    fn a_file_of_exactly_650_lines_pages_at_200_line_strides() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "big.txt", &numbered_lines(650));

        let p0 = s.read_page("big.txt", 0).unwrap().unwrap();
        assert_eq!(p0.page, 0);
        assert_eq!(p0.total_pages, 4);
        assert_eq!(p0.start_line, 1);
        assert_eq!(p0.end_line, 200);
        assert_eq!(p0.total_lines, 650);
        assert_eq!(p0.body.lines().next(), Some("1"));
        assert_eq!(p0.body.lines().last(), Some("200"));

        let p1 = s.read_page("big.txt", 1).unwrap().unwrap();
        assert_eq!(p1.page, 1);
        assert_eq!(p1.start_line, 201);
        assert_eq!(p1.end_line, 400);
        assert_eq!(p1.body.lines().next(), Some("201"));
        assert_eq!(p1.body.lines().last(), Some("400"));

        let p3 = s.read_page("big.txt", 3).unwrap().unwrap();
        assert_eq!(p3.page, 3);
        assert_eq!(p3.start_line, 601);
        assert_eq!(p3.end_line, 650);
        assert_eq!(p3.body.lines().next(), Some("601"));
        assert_eq!(p3.body.lines().last(), Some("650"));
    }

    /// A page past the end clamps to the last one — the same "over-shoot reads
    /// as the tail, not an error" rule `file_list`'s own paging applies.
    #[test]
    fn a_page_past_the_end_clamps_to_the_last_page() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "big.txt", &numbered_lines(650));
        let clamped = s.read_page("big.txt", 99).unwrap().unwrap();
        let last = s.read_page("big.txt", 3).unwrap().unwrap();
        assert_eq!(clamped.page, last.page);
        assert_eq!(clamped.start_line, last.start_line);
        assert_eq!(clamped.end_line, last.end_line);
        assert_eq!(clamped.body, last.body);
    }

    #[test]
    fn an_empty_file_reports_page_0_of_0() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "empty.txt", "");
        let p = s.read_page("empty.txt", 0).unwrap().unwrap();
        assert_eq!(p.page, 0);
        assert_eq!(p.start_line, 1);
        assert_eq!(p.end_line, 0);
        assert_eq!(p.total_lines, 0);
        assert_eq!(p.total_pages, 0);
        assert_eq!(p.body, "");
    }

    #[test]
    fn read_page_of_a_missing_path_is_none() {
        let (_dir, s) = store_with_tree();
        assert!(s.read_page("nope.rs", 0).unwrap().is_none());
    }

    /// The upper layer pages from the same in-memory string via a cursor —
    /// no second copy, but the same page math as the workspace layer.
    #[test]
    fn upper_layer_files_page_too() {
        let s = VfsStore::new();
        s.write("big.txt", numbered_lines(650)).unwrap();
        let p1 = s.read_page("big.txt", 1).unwrap().unwrap();
        assert_eq!(p1.start_line, 201);
        assert_eq!(p1.end_line, 400);
        assert_eq!(p1.total_pages, 4);
        assert_eq!(p1.body.lines().next(), Some("201"));
    }

    #[test]
    fn oversize_lower_file_refuses_to_page() {
        let (dir, s) = store_with_tree();
        let big = (MAX_LOWER_FILE_BYTES + 1) as usize;
        std::fs::write(dir.path().join("huge.txt"), vec![b'a'; big]).unwrap();
        let err = s.read_page("huge.txt", 0).unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");
    }

    #[test]
    fn non_utf8_lower_file_is_unreadable() {
        let (dir, s) = store_with_tree();
        std::fs::write(dir.path().join("blob.bin"), [0xff, 0xfe, 0x00]).unwrap();
        let err = s.read("blob.bin").unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");
    }

    /// **`list_dir` never opens a file.** A listing that had to decode every
    /// file's bytes to report on it would refuse exactly where `read` does; it
    /// does not, because it never reads past `entry.metadata()` — the whole
    /// point of dropping `ListEntry::lines`, which used to force exactly that
    /// open.
    #[test]
    fn non_utf8_lower_file_still_lists() {
        let (dir, s) = store_with_tree();
        std::fs::write(dir.path().join("blob.bin"), [0xff, 0xfe, 0x00]).unwrap();
        let entries = s.list_dir("").unwrap().unwrap();
        let entry = entries
            .iter()
            .find(|e| e.path == "blob.bin")
            .expect("a non-UTF-8 file still lists");
        assert_eq!(entry.bytes, Some(3));
    }

    #[test]
    fn oversize_lower_file_lists_but_refuses_to_read() {
        let (dir, s) = store_with_tree();
        let big = (MAX_LOWER_FILE_BYTES + 1) as usize;
        std::fs::write(dir.path().join("huge.txt"), vec![b'a'; big]).unwrap();

        let err = s.read("huge.txt").unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");

        let entries = s.list_dir("").unwrap().unwrap();
        let entry = entries
            .iter()
            .find(|e| e.path == "huge.txt")
            .expect("oversize files still list");
        assert_eq!(entry.bytes, Some(big), "with their true size");
    }

    // ── Lower layer: listing ─────────────────────────────────────────────────

    #[test]
    fn list_is_sorted_and_one_level_deep() {
        let (_dir, s) = store_with_tree();
        assert_eq!(listed(&s, ""), vec!["README.md", "src"]);
        for dir in ["src", "src/", "/src"] {
            assert_eq!(
                listed(&s, dir),
                vec!["src/main.rs", "src/util"],
                "dir {dir:?}",
            );
        }
        assert_eq!(listed(&s, "src/util"), vec!["src/util/helper.rs"]);
    }

    #[test]
    fn list_dir_of_a_missing_directory_is_not_found() {
        let (_dir, s) = store_with_tree();
        assert_eq!(s.list_dir("nothing/here").unwrap(), None);
    }

    /// A real directory path only — the argument is a directory to list, not a
    /// string prefix, so a partial segment or an exact file name no longer
    /// resolves to anything.
    #[test]
    fn list_dir_of_a_file_path_is_not_found() {
        let (_dir, s) = store_with_tree();
        assert_eq!(s.list_dir("README.md").unwrap(), None);
        assert_eq!(s.list_dir("src/ma").unwrap(), None);
    }

    /// A conflicting write elsewhere in the upper layer must not make a real
    /// file resolve as a directory too — `"README.md"` stays `None` even
    /// though `"README.md/notes.txt"` nonsensically implies it is one.
    #[test]
    fn list_dir_of_a_file_stays_not_found_despite_a_conflicting_deeper_write() {
        let (_dir, s) = store_with_tree();
        s.write("README.md/notes.txt", "oops".into()).unwrap();
        assert_eq!(s.list_dir("README.md").unwrap(), None);
    }

    /// A plain upper file and another upper file that collapses to a
    /// directory of the same name are two different things sharing one
    /// name — both must survive the listing, not silently collide into one.
    #[test]
    fn a_file_and_a_same_named_collapsed_directory_both_list() {
        let s = VfsStore::new();
        s.write("src", "plain file".into()).unwrap();
        s.write("src/foo.rs", "nested".into()).unwrap();
        let entries = s.list_dir("").unwrap().unwrap();
        let file = entries
            .iter()
            .find(|e| e.path == "src" && !e.dir)
            .expect("the plain file at \"src\" must still be listed");
        assert_eq!(file.bytes, Some("plain file".len()));
        let dir = entries
            .iter()
            .find(|e| e.path == "src" && e.dir)
            .expect("the collapsed directory named \"src\" must still be listed");
        assert_eq!(dir.bytes, None);
    }

    #[test]
    fn subdirectory_entries_carry_no_size() {
        let (_dir, s) = store_with_tree();
        let entries = s.list_dir("").unwrap().unwrap();
        let src = entries.iter().find(|e| e.path == "src").unwrap();
        assert!(src.dir);
        assert_eq!(src.bytes, None);
        assert!(!src.modified);
    }

    #[test]
    fn upper_shadows_lower_exactly_once() {
        let (_dir, s) = store_with_tree();
        s.write("src/main.rs", "fn main() { /* mine */ }\n".into())
            .unwrap();
        let entries = s.list_dir("src").unwrap().unwrap();
        assert_eq!(
            entries.iter().filter(|e| e.path == "src/main.rs").count(),
            1,
            "a shadowed path must appear once, not twice",
        );
        let main = entries.iter().find(|e| e.path == "src/main.rs").unwrap();
        assert!(main.modified);
        assert_eq!(main.bytes, Some("fn main() { /* mine */ }\n".len()));
        let util = entries.iter().find(|e| e.path == "src/util").unwrap();
        assert!(util.dir);
        assert!(!util.modified, "a directory entry is never itself modified");
    }

    #[test]
    fn a_session_only_file_lists_alongside_workspace_files() {
        let (_dir, s) = store_with_tree();
        s.write("src/scratch.rs", "// draft\n".into()).unwrap();
        assert_eq!(
            listed(&s, "src"),
            vec!["src/main.rs", "src/scratch.rs", "src/util"],
        );
    }

    /// A directory that exists only because the session wrote into it — never
    /// present on disk — still lists, synthesised purely from the upper layer.
    #[test]
    fn a_session_only_directory_lists_even_without_a_workspace_counterpart() {
        let (_dir, s) = store_with_tree();
        s.write("gen/a.txt", "hi".into()).unwrap();
        assert_eq!(listed(&s, "gen"), vec!["gen/a.txt"]);
        assert!(listed(&s, "").contains(&"gen".to_string()));
    }

    // ── Whiteouts ────────────────────────────────────────────────────────────

    #[test]
    fn deleting_a_lower_file_whiteouts_it_without_touching_disk() {
        let (dir, s) = store_with_tree();
        assert!(s.delete("README.md"));

        assert_eq!(s.read("README.md").unwrap(), None);
        assert!(!listed(&s, "").contains(&"README.md".to_string()));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("README.md")).unwrap(),
            "# project\n",
            "the file on disk must be untouched",
        );
        assert!(!s.delete("README.md"), "already whiteouted");
    }

    /// The whiteout must survive the alias: deleting by one spelling hides the
    /// path under every other.
    #[test]
    fn a_whiteout_applies_to_every_spelling_of_the_path() {
        let (_dir, s) = store_with_tree();
        s.delete("/src/main.rs");
        assert_eq!(s.read("src/main.rs").unwrap(), None);
        assert_eq!(s.read("./src/main.rs").unwrap(), None);
    }

    #[test]
    fn writing_over_a_whiteout_clears_it_and_counts_as_creation() {
        let (_dir, s) = store_with_tree();
        s.delete("README.md");
        assert!(
            s.write("README.md", "# mine\n".into()).unwrap(),
            "the path did not resolve while the whiteout stood",
        );
        assert_eq!(s.read("README.md").unwrap().as_deref(), Some("# mine\n"));
        assert!(listed(&s, "").contains(&"README.md".to_string()));
        // And deleting again re-hides it, since the lower file is still there.
        assert!(s.delete("README.md"));
        assert_eq!(s.read("README.md").unwrap(), None);
    }

    #[test]
    fn deleting_a_shadowed_path_hides_both_layers() {
        let (_dir, s) = store_with_tree();
        s.write("src/main.rs", "mine".into()).unwrap();
        assert!(s.delete("src/main.rs"));
        assert_eq!(
            s.read("src/main.rs").unwrap(),
            None,
            "the workspace file must not resurface once the shadow is removed",
        );
        assert_eq!(s.total_bytes(), 0);
        assert!(!listed(&s, "src").contains(&"src/main.rs".to_string()));
    }

    /// Shadowing a lower file is an overwrite, not a creation — the path already
    /// resolved before the call.
    #[test]
    fn first_write_over_a_lower_file_reports_overwrite() {
        let (_dir, s) = store_with_tree();
        assert!(!s.write("README.md", "changed".into()).unwrap());
        assert!(s.write("brand-new.md", "fresh".into()).unwrap());
    }

    // ── Ignore rules ─────────────────────────────────────────────────────────

    #[test]
    fn ignored_and_hidden_paths_are_omitted_from_listings_but_still_read() {
        let (dir, s) = store_with_tree();
        put(dir.path(), ".gitignore", "ignored/\n");
        put(dir.path(), "ignored/secret.txt", "shh\n");
        put(dir.path(), ".env", "TOKEN=1\n");

        let all = listed(&s, "");
        assert!(!all.contains(&"ignored".to_string()), "{all:?}");
        assert!(!all.contains(&".env".to_string()), "{all:?}");
        assert!(!all.contains(&".gitignore".to_string()), "{all:?}");

        // Hidden files are `ls`-invisible, not unreachable.
        assert_eq!(s.read(".env").unwrap().as_deref(), Some("TOKEN=1\n"));
        // An ignored file is genuinely out of scope for listing, but reading it by
        // exact path still works — `read` never consults the ignore rules.
        assert_eq!(
            s.read("ignored/secret.txt").unwrap().as_deref(),
            Some("shh\n")
        );
    }

    // ── Sharing ──────────────────────────────────────────────────────────────

    /// One store is shared across the whole daemon behind an `Arc`, so concurrent
    /// tool calls hit it in parallel. Distinct paths must all survive, byte
    /// accounting must stay exact, and reads that race writes must never observe
    /// a torn value.
    #[test]
    fn concurrent_writes_and_reads_stay_consistent() {
        use std::sync::Arc;

        let (_dir, store) = store_with_tree();
        let store = Arc::new(store);
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let s = Arc::clone(&store);
                std::thread::spawn(move || {
                    for i in 0..50 {
                        s.write(&format!("gen/{t}-{i}.txt"), format!("{t}:{i}"))
                            .unwrap();
                        // Racing the workspace layer at the same time.
                        assert_eq!(
                            s.read("src/util/helper.rs").unwrap().as_deref(),
                            Some("pub fn h() {}\n"),
                        );
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(listed(&store, "gen/").len(), 8 * 50);
        let expected: usize = (0..8)
            .flat_map(|t| (0..50).map(move |i| format!("{t}:{i}").len()))
            .sum();
        assert_eq!(store.total_bytes(), expected, "byte accounting drifted");
    }

    /// Concurrent writers to one path resolve to some single winner — never a
    /// blend of the two, and never double-counted bytes.
    #[test]
    fn concurrent_writes_to_one_path_leave_exactly_one_winner() {
        use std::sync::Arc;

        let store = Arc::new(VfsStore::new());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let s = Arc::clone(&store);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        s.write("contended.txt", format!("writer-{t}")).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }

        let final_value = store.read("contended.txt").unwrap().unwrap();
        assert!(
            (0..8).any(|t| final_value == format!("writer-{t}")),
            "torn value {final_value:?}",
        );
        assert_eq!(store.list_dir("").unwrap().unwrap().len(), 1);
        assert_eq!(store.total_bytes(), final_value.len());
    }

    #[test]
    fn unicode_content_survives_both_layers() {
        let (dir, s) = store_with_tree();
        let text = "こんにちは 🌍 — overlay\n";
        put(dir.path(), "uni.txt", text);
        assert_eq!(s.read("uni.txt").unwrap().as_deref(), Some(text));

        let edited = format!("{text}さようなら\n");
        s.write("uni.txt", edited.clone()).unwrap();
        assert_eq!(s.read("uni.txt").unwrap().as_deref(), Some(edited.as_str()));
        assert_eq!(s.total_bytes(), edited.len(), "bytes, not chars");
    }

    // ── Deltas ───────────────────────────────────────────────────────────────

    /// A workspace file of `n` numbered lines.
    fn numbered(n: usize) -> String {
        (1..=n).map(|i| format!("line {i}\n")).collect()
    }

    /// **An edit holds only the lines it changed**, not the file it landed
    /// in: one changed line of a 500-line workspace file costs that line, and
    /// every read replays it onto the workspace's copy.
    #[test]
    fn an_edit_holds_only_its_changed_lines() {
        let (dir, s) = store_with_tree();
        let original = numbered(500);
        put(dir.path(), "big.txt", &original);
        let edited = original.replace("line 250\n", "line two-fifty\n");

        assert!(s.edit("big.txt", edited.clone()).unwrap());
        assert_eq!(
            s.total_bytes(),
            "line 250\n".len() + "line two-fifty\n".len()
        );
        assert_eq!(s.read("big.txt").unwrap().as_deref(), Some(edited.as_str()));
        assert!(s.is_modified("big.txt"));
        let listed = s.list_dir("").unwrap().unwrap();
        let big = listed.iter().find(|e| e.path == "big.txt").unwrap();
        assert_eq!((big.bytes, big.modified), (Some(edited.len()), true));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("big.txt")).unwrap(),
            original,
            "the workspace is untouched"
        );
    }

    /// Edits stack: each is positioned in the text the one before it left,
    /// and all of them replay in order.
    #[test]
    fn successive_edits_replay_in_order() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "cfg.toml", "a = 1\nb = 2\nc = 3\n");
        s.edit("cfg.toml", "a = 1\nb = 20\nc = 3\n".into()).unwrap();
        s.edit("cfg.toml", "a = 1\nb = 20\nc = 3\nd = 4\n".into())
            .unwrap();
        s.edit("cfg.toml", "a = 10\nb = 20\nc = 3\nd = 4\n".into())
            .unwrap();
        assert_eq!(
            s.read("cfg.toml").unwrap().as_deref(),
            Some("a = 10\nb = 20\nc = 3\nd = 4\n")
        );
    }

    /// **A write replaces the whole file and supersedes its edits**: what is
    /// held afterwards is the one replacement, and it no longer depends on the
    /// workspace's copy.
    #[test]
    fn a_write_supersedes_every_earlier_edit() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "notes.md", "one\ntwo\n");
        s.edit("notes.md", "one\n2\n".into()).unwrap();
        assert!(!s.write("notes.md", "fresh\n".into()).unwrap());
        assert_eq!(s.total_bytes(), "fresh\n".len());
        put(dir.path(), "notes.md", "changed underneath\n");
        assert_eq!(s.read("notes.md").unwrap().as_deref(), Some("fresh\n"));
    }

    /// An edit over a session-written file stacks on the write, and never
    /// touches the workspace.
    #[test]
    fn an_edit_stacks_on_a_write() {
        let s = VfsStore::new();
        s.write("new.rs", "fn a() {}\nfn b() {}\n".into()).unwrap();
        s.edit("new.rs", "fn a() {}\nfn c() {}\n".into()).unwrap();
        assert_eq!(
            s.read("new.rs").unwrap().as_deref(),
            Some("fn a() {}\nfn c() {}\n")
        );
        assert_eq!(
            s.total_bytes(),
            "fn a() {}\nfn b() {}\n".len() + "fn b() {}\n".len() + "fn c() {}\n".len()
        );
    }

    /// **An edit whose workspace copy changed underneath it is refused**, not
    /// spliced into the wrong place — and a whole-file write settles it.
    #[test]
    fn an_edit_over_a_file_changed_on_disk_diverges() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "lib.rs", "fn one() {}\nfn two() {}\n");
        s.edit("lib.rs", "fn one() {}\nfn TWO() {}\n".into())
            .unwrap();
        put(dir.path(), "lib.rs", "fn zero() {}\n");

        assert!(matches!(s.read("lib.rs"), Err(VfsError::Diverged(_))));
        assert!(matches!(
            s.read_page("lib.rs", 0),
            Err(VfsError::Diverged(_))
        ));
        s.write("lib.rs", "fn settled() {}\n".into()).unwrap();
        assert_eq!(
            s.read("lib.rs").unwrap().as_deref(),
            Some("fn settled() {}\n")
        );
    }

    /// **An edit that rewrites most of a file is held as the whole file**:
    /// the same one replacement a write makes, superseding the edits before
    /// it and no longer depending on the workspace's copy.
    #[test]
    fn an_edit_that_rewrites_most_of_a_file_is_held_whole() {
        let (dir, s) = store_with_tree();
        put(
            dir.path(),
            "small.rs",
            "fn a() {}\nfn b() {}\nfn c() {}\nfn d() {}\n",
        );
        s.edit(
            "small.rs",
            "fn a() {}\nfn B() {}\nfn c() {}\nfn d() {}\n".into(),
        )
        .unwrap();
        let rewritten = "fn w() {}\nfn x() {}\nfn y() {}\nfn d() {}\n";
        s.edit("small.rs", rewritten.into()).unwrap();
        assert_eq!(
            s.total_bytes(),
            rewritten.len(),
            "one replacement, nothing else"
        );

        put(dir.path(), "small.rs", "changed underneath\n");
        assert_eq!(s.read("small.rs").unwrap().as_deref(), Some(rewritten));
    }

    /// An edit needs a file to edit, and one that changes nothing records
    /// nothing.
    #[test]
    fn an_edit_needs_a_file_and_a_change() {
        let (_dir, s) = store_with_tree();
        assert!(matches!(
            s.edit("absent.rs", "x".into()),
            Err(VfsError::Unreadable(_))
        ));
        assert!(!s.edit("README.md", "# project\n".into()).unwrap());
        assert!(!s.is_modified("README.md"));
        assert_eq!(s.total_bytes(), 0);

        s.delete("README.md");
        assert!(matches!(
            s.edit("README.md", "back".into()),
            Err(VfsError::Unreadable(_))
        ));
    }
}
