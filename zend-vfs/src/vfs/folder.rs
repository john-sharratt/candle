//! A folder on disk as a store's lower layer.
//!
//! What a store reads when its repository is not under git — the daemon's
//! uploads folder, a scratch workspace. The store only reads it: every change
//! is held in the overlay. A repository under git is read through its branch instead
//! ([`super::git_source`]).
//!
//! Every function takes the folder, `root`, and a normalised key, and every
//! one reaches a file through [`path`] — the single funnel that refuses what
//! the store refuses: a protected path, a spelling that would leave `root`,
//! and a link that leads out of it.

use std::fs::File;
use std::io::BufReader;
use std::path::{Component, Path, PathBuf};

use ignore::WalkBuilder;

use super::{PageResult, VfsError, VfsStore, MAX_LOWER_FILE_BYTES};

/// Absolute path of `norm` under `root`, or `None` when the key is empty
/// (the root itself is not a file) or refused.
pub(super) fn path(root: &Path, norm: &str) -> Option<PathBuf> {
    if norm.is_empty() {
        return None;
    }
    // The single funnel for lower-layer access. Guarding here rather than at
    // each caller is what makes the protection total: every read, listing and
    // existence check inherits it without knowing it exists.
    if VfsStore::is_protected(norm) {
        return None;
    }
    contained(root, norm)
}

/// [`under`], and then the same question asked of where the path really
/// leads: with every symlink and junction on the way resolved, it must still
/// be under `root` and outside every protected folder.
///
/// The spelling checks alone are not enough. A symlink committed to a
/// repository — `notes.md -> ~/.zend/secrets.yaml`, or a folder junction to
/// the user's home — spells a plain workspace path and opens a file outside
/// every repository, which is exactly where the daemon's secrets live; one
/// pointing at the repository's own `secrets/` folder opens it under an
/// unprotected name. A path that does not exist (yet, or any longer) is
/// judged by its deepest existing ancestor, so a missing file under a linked
/// folder is refused too rather than reported absent.
pub(super) fn contained(root: &Path, norm: &str) -> Option<PathBuf> {
    let joined = under(root, norm)?;
    let real_root = root.canonicalize().ok()?;
    let mut probe = joined.clone();
    let real = loop {
        match probe.canonicalize() {
            Ok(real) => break real,
            Err(_) => {
                if !probe.pop() {
                    return None;
                }
            }
        }
    };
    let rel = real.strip_prefix(&real_root).ok()?;
    let key = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    if VfsStore::is_protected(&key) {
        return None;
    }
    Some(joined)
}

/// `root.join(norm)`, only when the result stays under `root`.
///
/// Normalisation removes `..` and leading separators, but not a Windows
/// drive or device prefix: `C:/Users/x/.ssh/id_rsa` normalises to itself,
/// and joining a path that carries a prefix *replaces* the root — so a
/// `file_read` or a script's `vfs.read` reached any file on the host. A segment with a `:` in it is a drive (`C:`), a device path
/// (`\\?\C:\`), or an NTFS alternate stream (`notes.txt:hidden`), and none
/// of those names a workspace file; the component check refuses anything
/// else that is not a plain name.
///
/// So is any other spelling Windows would resolve to a different name: a
/// segment ending in a dot or a space (both dropped when the name is
/// opened), and an 8.3 short name (`SECRET~1`), which opens the long name
/// it abbreviates — `secrets/` included — past every check made on the
/// spelling.
pub(super) fn under(root: &Path, norm: &str) -> Option<PathBuf> {
    if !VfsStore::addressable(norm) {
        return None;
    }
    if !Path::new(norm)
        .components()
        .all(|c| matches!(c, Component::Normal(_)))
    {
        return None;
    }
    Some(root.join(norm))
}

/// Whether `norm` is a file under `root`.
pub(super) fn is_file(root: &Path, norm: &str) -> bool {
    path(root, norm).is_some_and(|p| p.is_file())
}

/// `true` when `norm` names a real directory under `root` — the empty path
/// (the root itself) always does, when the root exists.
pub(super) fn is_dir(root: &Path, norm: &str) -> bool {
    if norm.is_empty() {
        root.is_dir()
    } else {
        contained(root, norm).is_some_and(|p| p.is_dir())
    }
}

/// The file at `norm` as text, `None` when there is no file. Refused above
/// [`MAX_LOWER_FILE_BYTES`] and when its bytes are not UTF-8.
pub(super) fn read_text(root: &Path, norm: &str) -> Result<Option<String>, VfsError> {
    let Some(abs) = path(root, norm) else {
        return Ok(None);
    };
    let Ok(meta) = std::fs::metadata(&abs) else {
        return Ok(None);
    };
    if !meta.is_file() {
        return Ok(None);
    }
    if meta.len() > MAX_LOWER_FILE_BYTES {
        return Err(VfsStore::too_large(norm, meta.len()));
    }
    let bytes = std::fs::read(&abs)
        .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")))?;
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| VfsError::Unreadable(format!("{norm} is not valid UTF-8 text")))
}

/// The file at `norm`, byte for byte and at any size, `None` when there is
/// no file.
pub(super) fn read_bytes(root: &Path, norm: &str) -> Result<Option<Vec<u8>>, VfsError> {
    let Some(abs) = path(root, norm) else {
        return Ok(None);
    };
    if !std::fs::metadata(&abs).is_ok_and(|m| m.is_file()) {
        return Ok(None);
    }
    std::fs::read(&abs)
        .map(Some)
        .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")))
}

/// One page of the file at `norm`: opened and streamed rather than read
/// whole, so a 4 MiB file costs [`super::PAGE_LINES`] lines of memory, not
/// 4 MiB.
pub(super) fn read_page(
    root: &Path,
    norm: &str,
    page: u32,
) -> Result<Option<PageResult>, VfsError> {
    let Some(abs) = path(root, norm) else {
        return Ok(None);
    };
    let Ok(meta) = std::fs::metadata(&abs) else {
        return Ok(None);
    };
    if !meta.is_file() {
        return Ok(None);
    }
    if meta.len() > MAX_LOWER_FILE_BYTES {
        return Err(VfsStore::too_large(norm, meta.len()));
    }
    let file = File::open(&abs)
        .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")))?;
    VfsStore::paginate(BufReader::new(file), page)
        .map(Some)
        .map_err(|e| VfsError::Unreadable(format!("{norm} is not valid UTF-8 text: {e}")))
}

/// One level of the folder `norm`: its immediate file and subdirectory
/// children, honouring every ignore file the `ignore` crate knows. Bounded to
/// depth 1, so a listing costs one directory's worth of `readdir`, never a
/// walk of a whole subtree — what makes `file_list` cheap on a large folder
/// no matter how deep `norm` is. Returns `(normalised path, bytes, is_dir)`;
/// `bytes` is `None` for a subdirectory and metadata-only for a file — never
/// a file open. A protected entry is dropped rather than listed — the same
/// silence `secrets/` gets everywhere else — so listing the protected
/// directory itself, or a parent that contains one, comes back empty of it
/// rather than erroring, and never names it. An entry
/// [`VfsStore::addressable`] refuses (a `:` in the name, an 8.3 short-name
/// tail, …) is dropped too — a listing never shows a file the store cannot
/// open.
pub(super) fn children(root: &Path, norm: &str) -> Vec<(String, Option<usize>, bool)> {
    let walk_root = if norm.is_empty() {
        root.to_path_buf()
    } else {
        let Some(p) = contained(root, norm) else {
            return Vec::new();
        };
        p
    };

    let mut out = Vec::new();
    for entry in walker(&walk_root, Some(1)).flatten() {
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
        if VfsStore::is_protected(&path) || !VfsStore::addressable(&path) {
            continue;
        }
        if is_dir {
            out.push((path, None, true));
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        out.push((path, Some(meta.len() as usize), false));
    }
    out
}

/// Normalised keys of the files under `norm_prefix`.
///
/// Deliberately stats nothing and reads nothing: this backs path search and
/// the candidate list for a content search, which need only the names.
pub(super) fn files_under(root: &Path, norm_prefix: &str) -> Vec<String> {
    let (walk_root, filter) = walk_start(root, norm_prefix);

    let mut out = Vec::new();
    for entry in walker(&walk_root, None).flatten() {
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
        if VfsStore::is_protected(&key) || !VfsStore::addressable(&key) {
            continue;
        }
        if let Some(p) = filter {
            if !VfsStore::matches_prefix(&key, p) {
                continue;
            }
        }
        out.push(key);
    }
    out
}

/// The ignore-driven walker both folder passes use.
///
/// One builder, so a listing and a search can never disagree about what is
/// visible — a file hidden from `file_list` but reachable by `file_grep`
/// would be the same class of hole as the read path that ignored these
/// rules entirely. `max_depth` is `Some(1)` for a one-level directory
/// listing and `None` for a search, which walks the whole subtree.
fn walker(root: &Path, max_depth: Option<usize>) -> ignore::Walk {
    WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .require_git(false)
        .parents(true)
        .max_depth(max_depth)
        .build()
}

/// Where a walk for `norm_prefix` starts, and the prefix filter its keys
/// still need.
///
/// Walking from the prefix directory (when it is one under the root) keeps
/// a narrow listing cheap on a large folder; otherwise the walk covers the
/// root and filters, which is what a partial-segment prefix like `src/ma`
/// needs — and what a prefix naming somewhere outside the root gets, so it
/// lists nothing rather than walking another drive.
fn walk_start<'p>(root: &Path, norm_prefix: &'p str) -> (PathBuf, Option<&'p str>) {
    match contained(root, norm_prefix) {
        Some(dir) if !norm_prefix.is_empty() && dir.is_dir() => (dir, None),
        _ => (root.to_path_buf(), Some(norm_prefix)),
    }
}
