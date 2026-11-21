//! Overlay filesystem behind the `file_*` tools: a session's changes over a
//! read-only repository.
//!
//! A store keeps the session's changes in memory, each path as the chain of deltas
//! that took it from its starting point to where it stands, and reads everything
//! else from the repository beneath it: nothing, a folder, or a git branch read at
//! one commit. A read replays a chain onto the lower copy, and no write reaches the
//! repository. Moving a branch-backed store's base carries its changes onto the
//! new tree, and any overlap is left between conflict markers for the conversation
//! to settle. Protected directories, `secrets/` and `.git`, are refused where each
//! lower layer is built, so every operation is held to them.
//!
//! - `store`: the store, its two layers, and the base a branch is read at.
//! - `session`: writes, edits and reads of a session's own changes.
//! - `chains`: the per-path delta chains and the stale-base check on a change.
//! - `base_move`: moving the base and carrying changes onto the new tree.
//! - `carry`: carrying the chains onto a moved base's tree.
//! - `conflict`: the paths a base move leaves in conflict.
//! - `keys`: path normalisation and the name rules.
//! - `listing`: directory listings, path search and grep.
//! - `page`: paging a file into fixed-size pages of lines.
//! - `protection`: the protected directories.
//! - `limits`: the size caps.
//! - `snapshot`: saving and restoring session state.
//! - `errors`: the reasons a read or change is refused.
//! - `folder`: a folder on disk as the lower layer.
//! - `git_source`: a git branch's tree and blobs as the lower layer.
//! - `tree`: the file list of a branch's tree.
//! - `view`: the lower layer as the tree of the store's base.
//! - `tests_edit`, `tests_listing`, `tests_normalise`, `tests_paging`,
//!   `tests_security`, `tests_snapshot`, `tests_upper`: tests.

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
use crate::{BranchName, FileChanges, GitError, ObjectFormat, Oid, Rev};

// ── Tests ────────────────────────────────────────────────────────────────────

mod base_move;
mod chains;
mod errors;
mod keys;
mod limits;
mod listing;
mod page;
mod protection;
mod session;
mod snapshot;
mod store;
#[cfg(test)]
mod tests_edit;
#[cfg(test)]
mod tests_listing;
#[cfg(test)]
mod tests_normalise;
#[cfg(test)]
mod tests_paging;
#[cfg(test)]
mod tests_security;
#[cfg(test)]
mod tests_snapshot;
#[cfg(test)]
mod tests_upper;

use chains::*;
pub use errors::*;
#[cfg(test)]
use keys::*;
pub use limits::*;
pub use listing::*;
pub use page::*;
pub use protection::*;
pub use snapshot::*;
pub use store::*;
