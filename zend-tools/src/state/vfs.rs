//! Overlay filesystem backing the `file_*` tools.
//!
//! Two layers, in the union-mount sense:
//!
//! * **Upper** — an in-memory `HashMap<String, String>` (normalised path → UTF-8
//!   content) holding everything the session has written, plus a set of
//!   *whiteouts* marking lower-layer paths the session has deleted.
//! * **Lower** — the daemon's working directory, read-only. Present only when a
//!   workspace root is configured ([`VfsStore::with_workspace`]); without one the
//!   store degenerates to the upper layer alone.
//!
//! A read resolves upper-first and falls through to the workspace, so a tool call
//! sees the real project without the session having to load it. A write always
//! lands in the upper layer — the workspace is **never** modified. Editing a file
//! that exists only in the workspace therefore reads it from below and writes the
//! result above: the write *is* the copy-up, so it happens only when the edit
//! succeeds, and every later read of that path sees the session's copy.
//!
//! Deleting a workspace-backed file records a whiteout instead of touching disk:
//! the path then reads as absent and stops appearing in listings, but the file on
//! disk is untouched. Writing to a whiteouted path clears the whiteout.
//!
//! # Path normalisation
//!
//! Paths are normalised to one canonical key before use: a leading `/` is
//! stripped, `.` and empty segments collapse, `..` pops the stack (it can never
//! escape the root — popping an empty stack is a no-op), and a leading
//! `workspace/` segment is dropped because `/workspace` is the mount point the
//! tool definitions document for the working directory. So `/workspace/src/main.rs`,
//! `workspace/src/main.rs`, `./src/main.rs`, and `src/main.rs` are all the same
//! key, `src/main.rs`, in both layers. A project containing a genuine top-level
//! `workspace/` directory cannot address it through these tools.
//!
//! # Lower-layer rules
//!
//! The workspace walk is `ignore`-driven (the same crate ripgrep uses), so
//! `.gitignore`, `.ignore`, the global git ignore, and hidden-file rules all
//! apply — `target/` and friends never appear. Hidden files are excluded from
//! listings the way `ls` excludes them, but they still *read* fine by exact path:
//! `.gitignore` does not show up in `list` and does resolve in `read`.
//!
//! Files above [`MAX_LOWER_FILE_BYTES`] are listed but refuse to read, as do files
//! whose bytes are not valid UTF-8; both surface as [`VfsError::Unreadable`].
//!
//! # Protected paths
//!
//! Any path with a [`PROTECTED_SEGMENT`] component is refused outright, in both
//! layers and by every operation: [`VfsError::Forbidden`]. That covers
//! `secrets/tools.yaml`, `web/secrets/auth.yaml`, and anything else a deployment
//! keeps in a `secrets/` directory.
//!
//! **This is not the same protection as `.gitignore`, and the difference is the
//! whole point.** The ignore rules are consulted by the listing walk and by
//! nothing else — a read resolves a normalised key straight to a path under the
//! root and opens it. So before this guard existed, a gitignored secret was
//! invisible to `file_list` and served in full by `file_read`, which is the
//! worst of both worlds: hidden from the operator auditing what the model can
//! see, and one call away from the transcript.
//!
//! The refusal is enforced in [`VfsStore::lower_path`], the single funnel every
//! lower-layer read goes through, rather than at each call site — a guard that
//! has to be remembered at N call sites is a guard that is missing at one of
//! them. Normalisation runs first, so alternate spellings (`/secrets/x`,
//! `a/../secrets/x`, `workspace/secrets/x`, backslashes) all collapse onto the
//! same key before the check sees it.
//!
//! # Size cap
//!
//! The upper layer is capped at 10 MiB per store (enforced on each `write`).
//! Reading through to the workspace costs nothing against the cap because nothing
//! is retained; a copy-up does, and returns [`VfsError::Full`] if it would not fit.
//!
//! # Direct mode
//!
//! [`VfsStore::direct`] is the same store with no upper layer: a write lands in
//! the workspace on disk, a delete removes the file, and every read therefore
//! sees what is on disk. It backs the daemon's Mutable tools mode, for a caller
//! entitled to change the project itself rather than a session copy of it.
//! Everything else holds unchanged — path normalisation (so `..` still cannot
//! leave the root) and the protected-path refusal in particular, which in this
//! mode is what stops a tool overwriting a deployment's secrets.
//!
//! A direct write goes to a sibling temporary file first and is renamed over
//! the target, so a write interrupted partway leaves the old file whole rather
//! than truncated.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use ignore::WalkBuilder;

use crate::grants::DiskWriteGrant;

const MAX_BYTES: usize = 10 * 1024 * 1024; // 10 MiB

/// Largest workspace file the lower layer will read into a tool response.
/// Listing is unaffected — an oversize file still shows up with its true size.
pub const MAX_LOWER_FILE_BYTES: u64 = 4 * 1024 * 1024; // 4 MiB

/// The mount-point segment the tool definitions use for the working directory.
/// Stripped during normalisation so `/workspace/src` and `src` are one key.
const MOUNT_SEGMENT: &str = "workspace";

/// Path segment marking a directory the tools may not touch.
///
/// A deployment's secrets live in a `secrets/` directory — `secrets/tools.yaml`
/// for the daemon's own API keys, `web/secrets/auth.yaml` for the gateway's
/// sign-in config. One name, matched at any depth, so a new secrets directory is
/// protected the day it is created rather than the day someone remembers to add
/// it to a list.
pub const PROTECTED_SEGMENT: &str = "secrets";

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
    /// A [`VfsStore::direct`] write could not reach the disk — a permission, a
    /// full volume, a path that names a directory.
    Unwritable(String),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfsError::Full => write!(f, "VFS storage limit exceeded (10 MiB)"),
            VfsError::Unreadable(why) => write!(f, "{why}"),
            VfsError::Forbidden(path) => write!(
                f,
                "{path} is under a {PROTECTED_SEGMENT}/ directory and cannot be \
                 read, written or listed by tools"
            ),
            VfsError::Unwritable(why) => write!(f, "{why}"),
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

/// One entry in a listing: normalised path and byte size.
///
/// **No line count.** Producing one means opening and UTF-8-decoding every file
/// the walk touches — the whole tree, before paging, to fill fifty rows. `bytes` comes free from the directory metadata the walk
/// already has; lines do not, and a listing is not worth reading a codebase for.
/// A file's length reaches the model through `file_read`'s own header instead
/// (`(lines 1-200 of 2499)`), which is exact, costs nothing extra, and arrives
/// at the moment the number is actually needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListEntry {
    pub path: String,
    pub bytes: usize,
    /// `true` when the entry is the session's own copy (upper layer) rather than
    /// a file read straight off the workspace.
    pub modified: bool,
}

#[derive(Default)]
struct Upper {
    files: HashMap<String, String>,
    /// Lower-layer paths the session deleted. Never contains a path that is also
    /// in `files` — writing clears the whiteout, deleting an upper file that has
    /// no lower counterpart just removes it.
    whiteouts: HashSet<String>,
}

/// Where a store's writes land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Layering {
    /// In the in-memory upper layer, over a read-only workspace.
    #[default]
    Overlay,
    /// On disk, in the workspace itself — see the module's "Direct mode".
    Direct,
}

/// Union-mount of a session-private in-memory layer over the read-only workspace
/// — or, built with [`VfsStore::direct`], the workspace itself.
#[derive(Default)]
pub struct VfsStore {
    upper: RwLock<Upper>,
    /// Lower layer root. `None` leaves the store upper-only.
    workspace: Option<PathBuf>,
    layering: Layering,
}

impl VfsStore {
    /// Upper layer only — no workspace fall-through.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overlay the upper layer on `root`, the daemon's working directory.
    pub fn with_workspace(root: impl Into<PathBuf>) -> Self {
        Self {
            upper: RwLock::new(Upper::default()),
            workspace: Some(root.into()),
            layering: Layering::Overlay,
        }
    }

    /// The workspace at `root` with no overlay: writes and deletes change the
    /// files on disk. See the module's "Direct mode".
    ///
    /// Takes the [`DiskWriteGrant`] only [`Grants::disk_write`](crate::Grants::disk_write)
    /// makes, so a store that writes the disk exists only where that capability
    /// was granted.
    pub fn direct(root: impl Into<PathBuf>, _grant: DiskWriteGrant) -> Self {
        Self {
            upper: RwLock::new(Upper::default()),
            workspace: Some(root.into()),
            layering: Layering::Direct,
        }
    }

    /// Whether writes and deletes change the workspace on disk.
    pub fn is_direct(&self) -> bool {
        self.layering == Layering::Direct
    }

    /// The configured lower-layer root, if any.
    pub fn workspace(&self) -> Option<&Path> {
        self.workspace.as_deref()
    }

    /// Write into the upper layer, clearing any whiteout on the path. Returns
    /// whether this created a path that did not previously resolve — shadowing a
    /// workspace file for the first time counts as an overwrite, not a creation,
    /// because the path already resolved before the call. Writing over a whiteout
    /// *is* a creation: the path did not resolve while the whiteout stood.
    ///
    /// On a [`direct`](Self::direct) store the file is written on disk instead,
    /// and `true` means it did not exist there before.
    pub fn write(&self, path: &str, content: String) -> Result<bool, VfsError> {
        let norm = Self::normalize(path);
        // Refused in both modes. On an overlay nothing reaches disk, so this
        // stops a session planting a decoy at a protected path that later reads
        // would then find; on a direct store it is what keeps a tool from
        // overwriting the deployment's secrets.
        Self::guard(&norm)?;
        if self.is_direct() {
            return self.write_disk(&norm, &content);
        }
        let in_lower = self.lower_exists(&norm);
        let mut guard = self.upper.write().unwrap();
        let whiteouted = guard.whiteouts.contains(&norm);
        let created = if whiteouted {
            true
        } else {
            !guard.files.contains_key(&norm) && !in_lower
        };
        // Insert before clearing the whiteout: a cap rejection has to leave the
        // overlay exactly as it was, or a failed write resurrects a file the
        // session deleted.
        Self::insert_capped(&mut guard, norm.clone(), content)?;
        guard.whiteouts.remove(&norm);
        Ok(created)
    }

    /// Resolve a path through the overlay: upper layer first, then the workspace.
    /// `Ok(None)` means the path does not exist in either layer (or is whiteouted).
    pub fn read(&self, path: &str) -> Result<Option<String>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        {
            let guard = self.upper.read().unwrap();
            if let Some(v) = guard.files.get(&norm) {
                return Ok(Some(v.clone()));
            }
            if guard.whiteouts.contains(&norm) {
                return Ok(None);
            }
        }
        self.read_lower(&norm)
    }

    /// Union listing under `prefix`, upper layer shadowing the workspace.
    /// Whiteouted paths are omitted. Sorted by path.
    pub fn list(&self, prefix: &str) -> Vec<ListEntry> {
        let norm_prefix = Self::normalize(prefix);
        let mut out: Vec<ListEntry> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        {
            let guard = self.upper.read().unwrap();
            for (k, v) in guard.files.iter() {
                if !Self::matches_prefix(k, &norm_prefix) {
                    continue;
                }
                seen.insert(k.clone());
                out.push(ListEntry {
                    path: k.clone(),
                    bytes: v.len(),
                    modified: true,
                });
            }
            for w in guard.whiteouts.iter() {
                seen.insert(w.clone());
            }
        }

        for (path, bytes) in self.list_lower(&norm_prefix) {
            if seen.contains(&path) {
                continue;
            }
            out.push(ListEntry {
                path,
                bytes,
                modified: false,
            });
        }

        out.sort_by(|a, b| a.path.cmp(&b.path));
        out
    }

    /// Remove a path from the overlay. An upper-layer file is dropped; a
    /// workspace-backed file gets a whiteout so it stops resolving. Returns
    /// whether the path resolved before the call. The workspace is never touched.
    ///
    /// On a [`direct`](Self::direct) store the file is removed from disk, and the
    /// result is whether it existed and was removed.
    pub fn delete(&self, path: &str) -> bool {
        let norm = Self::normalize(path);
        if Self::is_protected(&norm) {
            return false;
        }
        if self.is_direct() {
            return self
                .lower_path(&norm)
                .is_some_and(|abs| abs.is_file() && std::fs::remove_file(abs).is_ok());
        }
        let in_lower = self.lower_exists(&norm);
        let mut guard = self.upper.write().unwrap();
        if guard.whiteouts.contains(&norm) {
            return false;
        }
        let had_upper = guard.files.remove(&norm).is_some();
        if in_lower {
            guard.whiteouts.insert(norm);
        }
        had_upper || in_lower
    }

    /// Bytes held in the upper layer. Workspace files cost nothing — they are
    /// read on demand and never retained.
    pub fn total_bytes(&self) -> usize {
        self.upper
            .read()
            .unwrap()
            .files
            .values()
            .map(|v| v.len())
            .sum()
    }

    // ── Upper-layer helpers ──────────────────────────────────────────────────

    fn insert_capped(upper: &mut Upper, norm: String, content: String) -> Result<(), VfsError> {
        let existing: usize = upper.files.values().map(|v| v.len()).sum();
        let old_len = upper.files.get(&norm).map(|v| v.len()).unwrap_or(0);
        if existing - old_len + content.len() > MAX_BYTES {
            return Err(VfsError::Full);
        }
        upper.files.insert(norm, content);
        Ok(())
    }

    // ── Direct-mode helpers ──────────────────────────────────────────────────

    /// Write `content` to `norm` on disk, creating its parent directories.
    /// Returns whether the file did not exist before.
    ///
    /// Through a sibling temporary file renamed over the target: a rename
    /// replaces the file in one step, so a write cut short leaves the old
    /// content whole rather than a truncated file.
    fn write_disk(&self, norm: &str, content: &str) -> Result<bool, VfsError> {
        let abs = self.lower_path(norm).ok_or_else(|| {
            VfsError::Unwritable(format!("{norm:?} does not name a file in the workspace"))
        })?;
        let fail = |what: &str, e: std::io::Error| {
            VfsError::Unwritable(format!("{norm} could not be written ({what}: {e})"))
        };
        let existed = abs.is_file();
        if abs.is_dir() {
            return Err(VfsError::Unwritable(format!("{norm} is a directory")));
        }
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent).map_err(|e| fail("creating its directory", e))?;
        }
        let file_name = abs
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let temp = abs.with_file_name(format!(".{file_name}.zend-write"));
        std::fs::write(&temp, content).map_err(|e| fail("writing", e))?;
        if let Err(e) = std::fs::rename(&temp, &abs) {
            let _ = std::fs::remove_file(&temp);
            return Err(fail("replacing the file", e));
        }
        Ok(!existed)
    }

    // ── Lower-layer helpers ──────────────────────────────────────────────────

    /// Absolute path of `norm` under the workspace, or `None` when there is no
    /// lower layer or the key is empty (the root itself is not a file).
    fn lower_path(&self, norm: &str) -> Option<PathBuf> {
        let root = self.workspace.as_ref()?;
        if norm.is_empty() {
            return None;
        }
        // The single funnel for lower-layer access. Guarding here rather than at
        // each caller is what makes the protection total: `read_lower`,
        // `lower_exists`, and anything added later inherit it without knowing it
        // exists.
        if Self::is_protected(norm) {
            return None;
        }
        Some(root.join(norm))
    }

    /// Whether a normalised key names something under a protected directory.
    pub fn is_protected(norm: &str) -> bool {
        norm.split('/').any(|s| s == PROTECTED_SEGMENT)
    }

    /// `Err(Forbidden)` for a protected key, `Ok(())` otherwise.
    fn guard(norm: &str) -> Result<(), VfsError> {
        if Self::is_protected(norm) {
            return Err(VfsError::Forbidden(norm.to_string()));
        }
        Ok(())
    }

    fn lower_exists(&self, norm: &str) -> bool {
        self.lower_path(norm).is_some_and(|p| p.is_file())
    }

    fn read_lower(&self, norm: &str) -> Result<Option<String>, VfsError> {
        let Some(abs) = self.lower_path(norm) else {
            return Ok(None);
        };
        let Ok(meta) = std::fs::metadata(&abs) else {
            return Ok(None);
        };
        if !meta.is_file() {
            return Ok(None);
        }
        if meta.len() > MAX_LOWER_FILE_BYTES {
            return Err(VfsError::Unreadable(format!(
                "{norm} is {} bytes, above the {MAX_LOWER_FILE_BYTES}-byte workspace read limit",
                meta.len(),
            )));
        }
        let bytes = std::fs::read(&abs)
            .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")))?;
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| VfsError::Unreadable(format!("{norm} is not valid UTF-8 text")))
    }

    /// The ignore-driven walker both lower-layer passes use.
    ///
    /// One builder, so a listing and a search can never disagree about what is
    /// visible — a file hidden from `file_list` but reachable by `file_grep`
    /// would be the same class of hole as the read path that ignored these
    /// rules entirely.
    fn lower_walker(root: &Path) -> ignore::Walk {
        WalkBuilder::new(root)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .ignore(true)
            .require_git(false)
            .parents(true)
            .build()
    }

    /// Normalised keys of the workspace files under `norm_prefix`.
    ///
    /// Deliberately stats nothing and reads nothing: this backs path search and
    /// the candidate list for a content search, which need only the names.
    fn walk_lower_paths(&self, norm_prefix: &str) -> Vec<String> {
        let Some(root) = self.workspace.as_ref() else {
            return Vec::new();
        };
        let prefix_dir = root.join(norm_prefix);
        let (walk_root, filter) = if !norm_prefix.is_empty() && prefix_dir.is_dir() {
            (prefix_dir, None)
        } else {
            (root.clone(), Some(norm_prefix))
        };

        let mut out = Vec::new();
        for entry in Self::lower_walker(&walk_root).flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let Ok(rel) = entry.path().strip_prefix(root) else {
                continue;
            };
            let key = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            if Self::is_protected(&key) {
                continue;
            }
            if let Some(p) = filter {
                if !Self::matches_prefix(&key, p) {
                    continue;
                }
            }
            out.push(key);
        }
        out
    }

    /// Walk the workspace under `norm_prefix`, honouring every ignore file the
    /// `ignore` crate knows. Returns `(normalised path, bytes)`.
    ///
    /// **Metadata only — nothing here opens a file.** `bytes` is the directory
    /// entry's own length, so the cost of a listing is the walk. A line count
    /// would mean a `read` plus a UTF-8 decode of every file the walk touches —
    /// on this workspace ~2,900 files, for a listing that pages down to fifty
    /// rows.
    fn list_lower(&self, norm_prefix: &str) -> Vec<(String, usize)> {
        let Some(root) = self.workspace.as_ref() else {
            return Vec::new();
        };
        // Walking from the prefix directory (when it is one) keeps a narrow
        // listing cheap on a large repository; otherwise walk the root and filter,
        // which is what a partial-segment prefix like `src/ma` needs.
        let prefix_dir = root.join(norm_prefix);
        let (walk_root, filter) = if !norm_prefix.is_empty() && prefix_dir.is_dir() {
            (prefix_dir, None)
        } else {
            (root.clone(), Some(norm_prefix))
        };

        let mut out = Vec::new();
        for entry in Self::lower_walker(&walk_root).flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let Ok(rel) = entry.path().strip_prefix(root) else {
                continue;
            };
            let key = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            if Self::is_protected(&key) {
                continue;
            }
            if let Some(p) = filter {
                if !Self::matches_prefix(&key, p) {
                    continue;
                }
            }
            let Ok(meta) = entry.metadata() else { continue };
            out.push((key, meta.len() as usize));
        }
        out
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Every path visible under `prefix`, upper layer shadowing the workspace.
    ///
    /// Unlike [`VfsStore::list`] this reads no file contents, so it stays cheap
    /// over a whole repository — line counts are what make a full listing
    /// expensive, and a path search does not need them.
    pub fn paths(&self, prefix: &str) -> Vec<String> {
        let norm_prefix = Self::normalize(prefix);
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();
        {
            let guard = self.upper.read().unwrap();
            for k in guard.files.keys() {
                if Self::matches_prefix(k, &norm_prefix) && !Self::is_protected(k) {
                    seen.insert(k.clone());
                    out.push(k.clone());
                }
            }
            for w in guard.whiteouts.iter() {
                seen.insert(w.clone());
            }
        }
        for path in self.walk_lower_paths(&norm_prefix) {
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
        for path in self.paths(prefix) {
            let (content, modified) = {
                let guard = self.upper.read().unwrap();
                match guard.files.get(&path) {
                    Some(v) => (Some(v.clone()), true),
                    None => (None, false),
                }
            };
            let content = match content {
                Some(c) => c,
                None => match self.read_lower(&path) {
                    Ok(Some(c)) => c,
                    _ => continue,
                },
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

    /// `true` when `key` is under `prefix`. Plain string-prefix semantics, as the
    /// tool's `prefix` parameter documents — so `src/` and `src` and even the
    /// partial `src/ma` all select `src/main.rs`. An empty prefix matches
    /// everything.
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
        // `/workspace` is the documented mount point of the working directory.
        if parts.first() == Some(&MOUNT_SEGMENT) {
            parts.remove(0);
        }
        parts.join("/")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    use tempfile::TempDir;

    use crate::grants::Grants;

    fn store_with_tree() -> (TempDir, VfsStore) {
        let dir = tempfile::tempdir().unwrap();
        put(dir.path(), "README.md", "# project\n");
        put(dir.path(), "src/main.rs", "fn main() {}\n");
        put(dir.path(), "src/util/helper.rs", "pub fn h() {}\n");
        let store = VfsStore::with_workspace(dir.path());
        (dir, store)
    }

    fn put(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    fn listed(store: &VfsStore, prefix: &str) -> Vec<String> {
        store.list(prefix).into_iter().map(|e| e.path).collect()
    }

    // ── direct mode ──────────────────────────────────────────────────────────

    fn granted() -> DiskWriteGrant {
        Grants::ALL.disk_write().unwrap()
    }

    /// **A direct write is a file on disk**, created with its directories, and
    /// every read — this store's and the filesystem's — sees it.
    #[test]
    fn a_direct_write_lands_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let s = VfsStore::direct(dir.path(), granted());
        assert!(s.is_direct());

        assert!(
            s.write("docs/new/note.md", "hello\n".into()).unwrap(),
            "created"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("docs/new/note.md")).unwrap(),
            "hello\n"
        );
        assert_eq!(
            s.read("docs/new/note.md").unwrap().as_deref(),
            Some("hello\n")
        );
        assert_eq!(s.total_bytes(), 0, "nothing is held in memory");

        // Overwriting is not a creation, and replaces the content.
        assert!(!s
            .write("/workspace/docs/new/note.md", "bye\n".into())
            .unwrap());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("docs/new/note.md")).unwrap(),
            "bye\n"
        );
        // No temporary file is left beside it.
        let names: Vec<String> = std::fs::read_dir(dir.path().join("docs/new"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["note.md"]);
    }

    /// A direct delete removes the file from disk; a missing one reports false.
    #[test]
    fn a_direct_delete_removes_the_file() {
        let dir = tempfile::tempdir().unwrap();
        put(dir.path(), "gone.txt", "x");
        let s = VfsStore::direct(dir.path(), granted());
        assert!(s.delete("gone.txt"));
        assert!(!dir.path().join("gone.txt").exists());
        assert!(!s.delete("gone.txt"), "nothing left to delete");
    }

    /// **The guards hold on disk.** A protected path is refused and left
    /// untouched, and `..` cannot climb out of the workspace — normalisation
    /// pins it to the root, so the write lands inside it.
    #[test]
    fn a_direct_store_cannot_touch_secrets_or_leave_the_root() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        put(&root, "secrets/tools.yaml", "key: real\n");
        let s = VfsStore::direct(&root, granted());

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
        assert!(root.join("escaped.txt").exists());
    }

    /// Writing where a directory stands is an error the model can read, not a
    /// silent success.
    #[test]
    fn a_direct_write_over_a_directory_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        put(dir.path(), "src/main.rs", "fn main() {}\n");
        let s = VfsStore::direct(dir.path(), granted());
        assert!(matches!(
            s.write("src", "x".into()),
            Err(VfsError::Unwritable(_))
        ));
    }

    /// The overlay, by contrast, never touches the disk — the property Mutable
    /// is the explicit exception to.
    #[test]
    fn an_overlay_write_never_reaches_disk() {
        let (dir, s) = store_with_tree();
        assert!(!s.is_direct());
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
            "/workspace/src/main.rs",
            "workspace/src/main.rs",
            "/workspace/./src/../src/main.rs",
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
        assert_eq!(
            VfsStore::normalize(r"\workspace\src\main.rs"),
            "src/main.rs"
        );
    }

    /// `..` can never climb above the root: popping an empty stack is a no-op, so
    /// a traversal attempt lands back inside the workspace.
    #[test]
    fn normalize_cannot_escape_the_root() {
        assert_eq!(VfsStore::normalize("../../../etc/passwd"), "etc/passwd");
        assert_eq!(VfsStore::normalize("/../../.."), "");
        assert_eq!(VfsStore::normalize(".."), "");
    }

    /// Only a *leading* `workspace` segment is the mount point; one nested deeper
    /// is an ordinary directory name.
    #[test]
    fn normalize_strips_only_the_leading_mount_segment() {
        assert_eq!(
            VfsStore::normalize("src/workspace/a.rs"),
            "src/workspace/a.rs"
        );
        assert_eq!(
            VfsStore::normalize("workspace/workspace/a.rs"),
            "workspace/a.rs"
        );
        // The mount point on its own is the root, not a file.
        assert_eq!(VfsStore::normalize("/workspace"), "");
        assert_eq!(VfsStore::normalize("/"), "");
        assert_eq!(VfsStore::normalize(""), "");
    }

    #[test]
    fn matches_prefix_is_plain_string_prefix() {
        assert!(VfsStore::matches_prefix("src/main.rs", ""));
        assert!(VfsStore::matches_prefix("src/main.rs", "src"));
        assert!(VfsStore::matches_prefix("src/main.rs", "src/"));
        assert!(VfsStore::matches_prefix("src/main.rs", "src/ma"));
        assert!(VfsStore::matches_prefix("src/main.rs", "src/main.rs"));
        assert!(!VfsStore::matches_prefix("src/main.rs", "srcx"));
        assert!(!VfsStore::matches_prefix("src/main.rs", "docs/"));
    }

    // ── Upper layer alone ────────────────────────────────────────────────────

    #[test]
    fn upper_only_store_round_trips_and_accounts_bytes() {
        let s = VfsStore::new();
        assert_eq!(s.total_bytes(), 0);
        assert!(s.workspace().is_none());

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
        s.write("/workspace/src/main.rs", "one".into()).unwrap();
        s.write("src/main.rs", "two".into()).unwrap();
        assert_eq!(
            s.read("./src/../src/main.rs").unwrap().as_deref(),
            Some("two")
        );
        assert_eq!(listed(&s, ""), vec!["src/main.rs"]);
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
            s.read("/workspace/src/util/helper.rs").unwrap().as_deref(),
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

    #[test]
    fn non_utf8_lower_file_is_unreadable() {
        let (dir, s) = store_with_tree();
        std::fs::write(dir.path().join("blob.bin"), [0xff, 0xfe, 0x00]).unwrap();
        let err = s.read("blob.bin").unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");
    }

    #[test]
    fn oversize_lower_file_lists_but_refuses_to_read() {
        let (dir, s) = store_with_tree();
        let big = (MAX_LOWER_FILE_BYTES + 1) as usize;
        std::fs::write(dir.path().join("huge.txt"), vec![b'a'; big]).unwrap();

        let err = s.read("huge.txt").unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");

        let entry = s
            .list("huge.txt")
            .into_iter()
            .next()
            .expect("oversize files still list");
        assert_eq!(entry.bytes, big, "with their true size");
    }

    // ── Lower layer: listing ─────────────────────────────────────────────────

    #[test]
    fn list_is_sorted_and_prefix_scoped() {
        let (_dir, s) = store_with_tree();
        assert_eq!(
            listed(&s, ""),
            vec!["README.md", "src/main.rs", "src/util/helper.rs"],
        );
        for prefix in ["src", "src/", "/workspace/src"] {
            assert_eq!(
                listed(&s, prefix),
                vec!["src/main.rs", "src/util/helper.rs"],
                "prefix {prefix:?}",
            );
        }
        assert_eq!(listed(&s, "src/util"), vec!["src/util/helper.rs"]);
        assert!(listed(&s, "nothing/here").is_empty());
    }

    /// A prefix naming a file rather than a directory still resolves — the walk
    /// falls back to filtering from the root.
    #[test]
    fn list_accepts_a_file_or_partial_segment_as_prefix() {
        let (_dir, s) = store_with_tree();
        assert_eq!(listed(&s, "README.md"), vec!["README.md"]);
        assert_eq!(listed(&s, "src/ma"), vec!["src/main.rs"]);
    }

    #[test]
    fn list_reports_sizes_from_disk() {
        let (_dir, s) = store_with_tree();
        let e = &s.list("src/main.rs")[0];
        assert_eq!(e.bytes, "fn main() {}\n".len());
        assert!(!e.modified);
    }

    /// **A listing never opens a file.** `bytes` comes from the walk's metadata;
    /// anything that needs a file's contents — a line count, once — turns a
    /// listing into a read of the whole tree. A file that cannot be opened at
    /// all therefore still lists, with its true size, which is the cheapest
    /// available proof that nothing on this path reads.
    #[test]
    fn listing_does_not_read_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        // Not valid UTF-8: a listing that decoded contents would have to either
        // fail or special-case this, and it does neither because it never looks.
        std::fs::write(dir.path().join("blob.bin"), [0xffu8, 0xfe, 0x00, 0x01]).unwrap();
        let s = VfsStore::with_workspace(dir.path());

        let e = &s.list("blob.bin")[0];
        assert_eq!(e.path, "blob.bin");
        assert_eq!(e.bytes, 4, "size comes from metadata, not from decoding");

        // And reading it is still the error it always was — the listing's
        // silence about contents is not the read path going soft.
        assert!(matches!(s.read("blob.bin"), Err(VfsError::Unreadable(_))));
    }

    #[test]
    fn upper_shadows_lower_exactly_once() {
        let (_dir, s) = store_with_tree();
        s.write("src/main.rs", "fn main() { /* mine */ }\n".into())
            .unwrap();
        let entries = s.list("src/");
        assert_eq!(
            entries.iter().filter(|e| e.path == "src/main.rs").count(),
            1,
            "a shadowed path must appear once, not twice",
        );
        let main = entries.iter().find(|e| e.path == "src/main.rs").unwrap();
        assert!(main.modified);
        assert_eq!(main.bytes, "fn main() { /* mine */ }\n".len());
        assert!(
            !entries
                .iter()
                .find(|e| e.path == "src/util/helper.rs")
                .unwrap()
                .modified
        );
    }

    #[test]
    fn a_session_only_file_lists_alongside_workspace_files() {
        let (_dir, s) = store_with_tree();
        s.write("src/scratch.rs", "// draft\n".into()).unwrap();
        assert_eq!(
            listed(&s, "src/"),
            vec!["src/main.rs", "src/scratch.rs", "src/util/helper.rs"],
        );
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
        s.delete("/workspace/src/main.rs");
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
        assert!(!listed(&s, "").contains(&"src/main.rs".to_string()));
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
        assert!(!all.iter().any(|p| p.starts_with("ignored/")), "{all:?}");
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
        assert_eq!(store.list("").len(), 1);
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
}
