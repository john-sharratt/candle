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
//! # Size cap
//!
//! The upper layer is capped at 10 MiB per store (enforced on each `write`).
//! Reading through to the workspace costs nothing against the cap because nothing
//! is retained; a copy-up does, and returns [`VfsError::Full`] if it would not fit.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use ignore::WalkBuilder;

const MAX_BYTES: usize = 10 * 1024 * 1024; // 10 MiB

/// Largest workspace file the lower layer will read into a tool response.
/// Listing is unaffected — an oversize file still shows up with its true size.
pub const MAX_LOWER_FILE_BYTES: u64 = 4 * 1024 * 1024; // 4 MiB

/// The mount-point segment the tool definitions use for the working directory.
/// Stripped during normalisation so `/workspace/src` and `src` are one key.
const MOUNT_SEGMENT: &str = "workspace";

#[derive(Debug)]
pub enum VfsError {
    Full,
    /// A workspace file exists but cannot be served as text — too large, or not
    /// valid UTF-8.
    Unreadable(String),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfsError::Full => write!(f, "VFS storage limit exceeded (10 MiB)"),
            VfsError::Unreadable(why) => write!(f, "{why}"),
        }
    }
}

/// One entry in a directory listing: normalised path, byte size. No line
/// count — that would cost opening every file in the walk to compute (see the
/// history on [`VfsStore::list_lower_dir`]), and a listing that never opens a
/// file is the whole point of `file_list`.
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

/// Lines per [`VfsStore::read_page`] page.
pub const PAGE_LINES: u32 = 300;

/// One page of a file's lines, 0-based.
#[derive(Debug)]
pub struct PageResult {
    /// The page actually returned. Clamped into range the same way
    /// [`crate::tools::file::Paging`] clamps a listing page: a request past
    /// the end yields the last page rather than an empty one.
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

#[derive(Default)]
struct Upper {
    files: HashMap<String, String>,
    /// Lower-layer paths the session deleted. Never contains a path that is also
    /// in `files` — writing clears the whiteout, deleting an upper file that has
    /// no lower counterpart just removes it.
    whiteouts: HashSet<String>,
}

/// Union-mount of a session-private in-memory layer over the read-only workspace.
#[derive(Default)]
pub struct VfsStore {
    upper: RwLock<Upper>,
    /// Lower layer root. `None` leaves the store upper-only.
    workspace: Option<PathBuf>,
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
        }
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
    pub fn write(&self, path: &str, content: String) -> Result<bool, VfsError> {
        let norm = Self::normalize(path);
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

    /// One [`PAGE_LINES`]-line page of a file, resolved through the overlay the
    /// same way [`Self::read`] is. `Ok(None)` means the path does not exist in
    /// either layer.
    ///
    /// Bounded memory regardless of file size: the workspace layer streams the
    /// file line by line rather than materialising it (`file_read` used to read
    /// the whole file into a `String` and a `Vec<&str>` slice of it just to
    /// return 300 lines). The upper layer is already resident (session writes
    /// are capped at 10 MiB total), so it pages from a cursor over the same
    /// string rather than a second copy.
    pub fn read_page(&self, path: &str, page: u32) -> Result<Option<PageResult>, VfsError> {
        let norm = Self::normalize(path);
        {
            let guard = self.upper.read().unwrap();
            if let Some(v) = guard.files.get(&norm) {
                return Self::paginate(std::io::Cursor::new(v.as_bytes()), page)
                    .map(Some)
                    .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")));
            }
            if guard.whiteouts.contains(&norm) {
                return Ok(None);
            }
        }
        self.read_lower_page(&norm, page)
    }

    /// One level of `dir`'s contents, upper layer shadowing the workspace:
    /// the files and subdirectories directly inside it, never a deeper file
    /// flattened up into its listing. Whiteouted paths are omitted. Sorted by
    /// path. `Ok(None)` means `dir` does not resolve to a directory in either
    /// layer — a missing path, or one that names a file.
    pub fn list_dir(&self, dir: &str) -> Result<Option<Vec<ListEntry>>, VfsError> {
        let norm = Self::normalize(dir);
        // `norm` resolving as a file rules out a directory listing outright —
        // checked first, and independently of the walk below, because a
        // conflicting write elsewhere in the upper layer (e.g. a file at
        // `"a/b.rs/c.rs"` alongside a real file at `"a/b.rs"`) would otherwise
        // make `upper_has_children` true for `"a/b.rs"` and this would
        // silently resolve a file path as a directory.
        if self.resolves_as_file(&norm) {
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
            for (k, v) in guard.files.iter() {
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
                        bytes: Some(v.len()),
                        dir: false,
                        modified: true,
                    }
                });
            }
            for w in guard.whiteouts.iter() {
                seen.insert((w.clone(), false));
            }
        }

        let lower_is_dir = self.lower_dir_exists(&norm);
        if !lower_is_dir && !upper_has_children && !norm.is_empty() {
            return Ok(None);
        }

        if lower_is_dir || norm.is_empty() {
            for (path, bytes, is_dir) in self.list_lower_dir(&norm) {
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
    fn resolves_as_file(&self, norm: &str) -> bool {
        if norm.is_empty() {
            return false;
        }
        {
            let guard = self.upper.read().unwrap();
            if guard.whiteouts.contains(norm) {
                return false;
            }
            if guard.files.contains_key(norm) {
                return true;
            }
        }
        self.lower_exists(norm)
    }

    /// Remove a path from the overlay. An upper-layer file is dropped; a
    /// workspace-backed file gets a whiteout so it stops resolving. Returns
    /// whether the path resolved before the call. The workspace is never touched.
    pub fn delete(&self, path: &str) -> bool {
        let norm = Self::normalize(path);
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

    // ── Lower-layer helpers ──────────────────────────────────────────────────

    /// Absolute path of `norm` under the workspace, or `None` when there is no
    /// lower layer or the key is empty (the root itself is not a file).
    fn lower_path(&self, norm: &str) -> Option<PathBuf> {
        let root = self.workspace.as_ref()?;
        if norm.is_empty() {
            return None;
        }
        Some(root.join(norm))
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

    /// [`Self::read_page`]'s workspace half: open the file and stream it rather
    /// than reading it whole, so a 4 MiB file costs [`PAGE_LINES`] lines of
    /// memory, not 4 MiB.
    fn read_lower_page(&self, norm: &str, page: u32) -> Result<Option<PageResult>, VfsError> {
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
        let file = std::fs::File::open(&abs)
            .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")))?;
        Self::paginate(std::io::BufReader::new(file), page)
            .map(Some)
            .map_err(|e| VfsError::Unreadable(format!("{norm} is not valid UTF-8 text: {e}")))
    }

    /// Stream `reader` line by line and return page `requested_page`
    /// ([`PAGE_LINES`]-line strides, 0-based), without ever holding more than
    /// two pages' worth of lines at once.
    ///
    /// A request past the end clamps to the last page — the same "over-shoot
    /// reads as the tail, not an error" rule [`crate::tools::file::Paging::of`]
    /// applies to `file_list` — which is why the last [`PAGE_LINES`] lines are
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
            // is the same "the whole file must be text" contract `read_lower`
            // enforces up front — just discovered while streaming instead of
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

    /// `true` when `norm` names a real directory in the workspace — the empty
    /// path (the root) always does, when there is a workspace at all.
    fn lower_dir_exists(&self, norm: &str) -> bool {
        let Some(root) = self.workspace.as_ref() else {
            return false;
        };
        if norm.is_empty() {
            root.is_dir()
        } else {
            root.join(norm).is_dir()
        }
    }

    /// One level of the workspace directory `norm`: its immediate file and
    /// subdirectory children, honouring every ignore file the `ignore` crate
    /// knows. Bounded to depth 1, so a listing costs one directory's worth of
    /// `readdir`, never a walk of a whole subtree — what makes `file_list`
    /// cheap on a large repository no matter how deep `norm` is. Returns
    /// `(normalised path, bytes, is_dir)`; `bytes` is `None` for a
    /// subdirectory and metadata-only for a file — never a file open, which is
    /// what makes a listing cheap even where it does open a directory (see the
    /// history in [`Self::read_lower_page`]'s sibling doc).
    fn list_lower_dir(&self, norm: &str) -> Vec<(String, Option<usize>, bool)> {
        let Some(root) = self.workspace.as_ref() else {
            return Vec::new();
        };
        let walk_root = if norm.is_empty() {
            root.clone()
        } else {
            root.join(norm)
        };

        let mut out = Vec::new();
        let walker = WalkBuilder::new(&walk_root)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .ignore(true)
            .require_git(false)
            .parents(true)
            .max_depth(Some(1))
            .build();
        for entry in walker.flatten() {
            // Depth 0 is `walk_root` itself, not a child of it.
            if entry.depth() == 0 {
                continue;
            }
            let file_type = entry.file_type();
            let is_dir = file_type.is_some_and(|t| t.is_dir());
            let is_file = file_type.is_some_and(|t| t.is_file());
            if !is_dir && !is_file {
                continue;
            }
            let name = entry.file_name().to_string_lossy();
            let path = if norm.is_empty() {
                name.into_owned()
            } else {
                format!("{norm}/{name}")
            };
            if is_dir {
                out.push((path, None, true));
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            out.push((path, Some(meta.len() as usize), false));
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

    fn listed(store: &VfsStore, dir: &str) -> Vec<String> {
        store
            .list_dir(dir)
            .unwrap()
            .unwrap_or_default()
            .into_iter()
            .map(|e| e.path)
            .collect()
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
    fn a_file_of_exactly_650_lines_pages_at_300_line_strides() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "big.txt", &numbered_lines(650));

        let p0 = s.read_page("big.txt", 0).unwrap().unwrap();
        assert_eq!(p0.page, 0);
        assert_eq!(p0.total_pages, 3);
        assert_eq!(p0.start_line, 1);
        assert_eq!(p0.end_line, 300);
        assert_eq!(p0.total_lines, 650);
        assert_eq!(p0.body.lines().next(), Some("1"));
        assert_eq!(p0.body.lines().last(), Some("300"));

        let p1 = s.read_page("big.txt", 1).unwrap().unwrap();
        assert_eq!(p1.page, 1);
        assert_eq!(p1.start_line, 301);
        assert_eq!(p1.end_line, 600);
        assert_eq!(p1.body.lines().next(), Some("301"));
        assert_eq!(p1.body.lines().last(), Some("600"));

        let p2 = s.read_page("big.txt", 2).unwrap().unwrap();
        assert_eq!(p2.page, 2);
        assert_eq!(p2.start_line, 601);
        assert_eq!(p2.end_line, 650);
        assert_eq!(p2.body.lines().next(), Some("601"));
        assert_eq!(p2.body.lines().last(), Some("650"));
    }

    /// A page past the end clamps to the last one — the same "over-shoot reads
    /// as the tail, not an error" rule `file_list`'s own paging applies.
    #[test]
    fn a_page_past_the_end_clamps_to_the_last_page() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "big.txt", &numbered_lines(650));
        let clamped = s.read_page("big.txt", 99).unwrap().unwrap();
        let last = s.read_page("big.txt", 2).unwrap().unwrap();
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
        assert_eq!(p1.start_line, 301);
        assert_eq!(p1.end_line, 600);
        assert_eq!(p1.total_pages, 3);
        assert_eq!(p1.body.lines().next(), Some("301"));
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
        for dir in ["src", "src/", "/workspace/src"] {
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
}
