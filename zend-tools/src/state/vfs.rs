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

/// One entry in a listing: normalised path, byte size, line count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListEntry {
    pub path: String,
    pub bytes: usize,
    pub lines: usize,
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
                    lines: v.lines().count(),
                    modified: true,
                });
            }
            for w in guard.whiteouts.iter() {
                seen.insert(w.clone());
            }
        }

        for (path, bytes, lines) in self.list_lower(&norm_prefix) {
            if seen.contains(&path) {
                continue;
            }
            out.push(ListEntry {
                path,
                bytes,
                lines,
                modified: false,
            });
        }

        out.sort_by(|a, b| a.path.cmp(&b.path));
        out
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

    /// Walk the workspace under `norm_prefix`, honouring every ignore file the
    /// `ignore` crate knows. Returns `(normalised path, bytes, lines)`.
    ///
    /// Line counts require reading each file, so they are only computed for files
    /// within [`MAX_LOWER_FILE_BYTES`] that parse as UTF-8; anything else reports
    /// `0` lines alongside its true byte size.
    fn list_lower(&self, norm_prefix: &str) -> Vec<(String, usize, usize)> {
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
        let walker = WalkBuilder::new(&walk_root)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .ignore(true)
            .require_git(false)
            .parents(true)
            .build();
        for entry in walker.flatten() {
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
            if let Some(p) = filter {
                if !Self::matches_prefix(&key, p) {
                    continue;
                }
            }
            let Ok(meta) = entry.metadata() else { continue };
            let bytes = meta.len();
            let lines = if bytes <= MAX_LOWER_FILE_BYTES {
                std::fs::read(entry.path())
                    .ok()
                    .and_then(|b| String::from_utf8(b).ok())
                    .map(|s| s.lines().count())
                    .unwrap_or(0)
            } else {
                0
            };
            out.push((key, bytes as usize, lines));
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
        assert_eq!(entry.lines, 0, "but no line count, since it is not read");
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
    fn list_reports_line_counts_and_sizes_from_disk() {
        let (_dir, s) = store_with_tree();
        let e = &s.list("src/main.rs")[0];
        assert_eq!(e.bytes, "fn main() {}\n".len());
        assert_eq!(e.lines, 1);
        assert!(!e.modified);
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
