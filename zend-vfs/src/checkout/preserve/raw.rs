//! Clearing away what a run left, whatever it named its files.
//!
//! A run's command can create a file under any name the file system allows —
//! one that is not UTF-8, one with a `:` or a trailing dot, which the paths
//! a conversation writes may never have. Putting the checkout back must
//! clear such a file all the same: a restore that refuses one name refuses
//! forever, and leaves the owner's work set aside. So the listings here are
//! git's bytes, never parsed into repository paths, and removal walks them
//! on disk with the one rule that matters — a link is removed, never
//! followed.

use std::path::{Path, PathBuf};

use super::ignored::Ignored;
use crate::checkout::CheckoutError;
use crate::{GitError, Repo};

/// Every untracked file the checkout's current rules do not ignore, as git
/// names it. A folder holding a repository of its own is listed as the
/// folder, ending in `/`.
pub(super) fn untracked(repo: &Repo) -> Result<Vec<Vec<u8>>, CheckoutError> {
    listed(repo, "ls-files", &["-z", "--others", "--exclude-standard"])
}

/// Every tracked path whose file differs from the index — changed, deleted,
/// or replaced by something else.
pub(super) fn changed_tracked(repo: &Repo) -> Result<Vec<Vec<u8>>, CheckoutError> {
    listed(repo, "diff-files", &["-z", "--name-only"])
}

fn listed(repo: &Repo, subcommand: &str, args: &[&str]) -> Result<Vec<Vec<u8>>, CheckoutError> {
    let out = repo.git(subcommand).args(args).read_only().run_ok()?;
    Ok(out
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .map(<[u8]>::to_vec)
        .collect())
}

/// What stands at a listed path, seen without following any link.
enum Standing {
    Nothing,
    File,
    Folder,
    /// The path itself, or the folder at `at` on the way to it, is a link.
    Link {
        at: PathBuf,
    },
}

/// `rel` under `root`, walked component by component. Git never names a
/// path with an empty, `.` or `..` component; one that does is refused
/// rather than walked.
fn standing(root: &Path, rel: &[u8]) -> Result<(PathBuf, Standing), CheckoutError> {
    let parts: Vec<&[u8]> = rel
        .strip_suffix(b"/")
        .unwrap_or(rel)
        .split(|b| *b == b'/')
        .collect();
    if parts
        .iter()
        .any(|part| part.is_empty() || *part == b"." || *part == b"..")
    {
        return Err(unlisted(rel));
    }
    let mut abs = root.to_path_buf();
    for (i, part) in parts.iter().enumerate() {
        abs.push(component(part).ok_or_else(|| unlisted(rel))?);
        let meta = match std::fs::symlink_metadata(&abs) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((abs, Standing::Nothing))
            }
            Err(e) => return Err(CheckoutError::io(&shown(rel), e)),
        };
        let last = i + 1 == parts.len();
        if meta.file_type().is_symlink() {
            return Ok((abs.clone(), Standing::Link { at: abs }));
        }
        if last {
            let found = if meta.is_dir() {
                Standing::Folder
            } else {
                Standing::File
            };
            return Ok((abs, found));
        }
        if !meta.is_dir() {
            return Ok((abs, Standing::Nothing));
        }
    }
    Ok((abs, Standing::Nothing))
}

#[cfg(unix)]
fn component(part: &[u8]) -> Option<PathBuf> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    Some(PathBuf::from(OsStr::from_bytes(part)))
}

/// Git for Windows names every path in UTF-8.
#[cfg(not(unix))]
fn component(part: &[u8]) -> Option<PathBuf> {
    std::str::from_utf8(part).ok().map(PathBuf::from)
}

fn shown(rel: &[u8]) -> String {
    String::from_utf8_lossy(rel).into_owned()
}

fn unlisted(rel: &[u8]) -> CheckoutError {
    GitError::malformed(
        "ls-files",
        format!("a path git would not name: {}", shown(rel)),
    )
    .into()
}

/// Remove a link — the link itself, never what it points at.
fn unlink(at: &Path, rel: &[u8]) -> Result<(), CheckoutError> {
    match std::fs::remove_file(at).or_else(|_| std::fs::remove_dir(at)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(CheckoutError::io(&shown(rel), e)),
    }
}

/// The link at `rel`, or on the way to it, removed; anything else left.
pub(super) fn remove_link(root: &Path, rel: &[u8]) -> Result<(), CheckoutError> {
    match standing(root, rel)? {
        (_, Standing::Link { at }) => unlink(&at, rel),
        _ => Ok(()),
    }
}

/// An untracked file the run left at `rel`, removed — a link as a link —
/// with every folder above it the removal left empty, unless what was
/// ignored when the checkout was set aside covers it. A folder is a
/// repository of its own, and is left.
pub(super) fn remove_untracked(
    root: &Path,
    rel: &[u8],
    kept: &Ignored,
) -> Result<(), CheckoutError> {
    if kept.covers(rel.strip_suffix(b"/").unwrap_or(rel)) {
        return Ok(());
    }
    let (abs, found) = standing(root, rel)?;
    let gone = match found {
        Standing::Nothing | Standing::Folder => return Ok(()),
        Standing::Link { at } => {
            unlink(&at, rel)?;
            at
        }
        Standing::File => {
            writable(&abs).map_err(|e| CheckoutError::io(&shown(rel), e))?;
            match std::fs::remove_file(&abs) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(CheckoutError::io(&shown(rel), e)),
            }
            abs
        }
    };
    let mut folder = gone.parent().map(Path::to_path_buf);
    while let Some(dir) = folder {
        if dir == root || std::fs::remove_dir(&dir).is_err() {
            break;
        }
        folder = dir.parent().map(Path::to_path_buf);
    }
    Ok(())
}

/// Before a tracked file goes back at `rel`: a link there or on the way is
/// removed, and an empty folder standing in the file's place.
pub(super) fn clear_the_way(root: &Path, rel: &[u8]) -> Result<(), CheckoutError> {
    match standing(root, rel)? {
        (_, Standing::Link { at }) => unlink(&at, rel),
        (abs, Standing::Folder) => match std::fs::remove_dir(&abs) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::DirectoryNotEmpty => Ok(()),
            Err(e) => Err(CheckoutError::io(&shown(rel), e)),
        },
        _ => Ok(()),
    }
}

/// Clear a read-only flag, so the file can be removed.
fn writable(abs: &Path) -> std::io::Result<()> {
    let meta = std::fs::symlink_metadata(abs)?;
    if !meta.permissions().readonly() {
        return Ok(());
    }
    let mut permissions = meta.permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(permissions.mode() | 0o200);
    }
    #[cfg(windows)]
    #[allow(clippy::permissions_set_readonly_false)]
    permissions.set_readonly(false);
    std::fs::set_permissions(abs, permissions)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A file is removed with the folders it leaves empty; a folder, a
    /// covered file and a path that is not there are left.**
    #[test]
    fn untracked_files_go_and_covered_ones_stay() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("deep/er")).unwrap();
        std::fs::write(root.join("deep/er/made.txt"), b"x").unwrap();
        std::fs::write(root.join(".env"), b"secret").unwrap();
        std::fs::create_dir_all(root.join("nested")).unwrap();
        let kept = Ignored::from_bytes(b".env\0");

        remove_untracked(root, b"deep/er/made.txt", &kept).unwrap();
        assert!(!root.join("deep").exists(), "the emptied folders go too");
        remove_untracked(root, b".env", &kept).unwrap();
        assert!(root.join(".env").exists(), "covered, so kept");
        remove_untracked(root, b"nested/", &kept).unwrap();
        assert!(root.join("nested").exists(), "a folder is left");
        remove_untracked(root, b"never/was.txt", &kept).unwrap();
    }

    /// **A path with an empty or parent component is refused**, never
    /// walked.
    #[test]
    fn a_path_git_would_not_name_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        for bad in [&b"../outside"[..], b"a//b", b"a/./b"] {
            assert!(remove_untracked(dir.path(), bad, &Ignored::default()).is_err());
        }
    }

    /// **A name no conversation could write is cleared all the same.**
    #[cfg(unix)]
    #[test]
    fn a_name_that_is_no_repository_path_is_cleared() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for name in [&b"a:b"[..], b"trailing.", b"\xff\xfe"] {
            let path = root.join(component(name).unwrap());
            std::fs::write(&path, b"x").unwrap();
            remove_untracked(root, name, &Ignored::default()).unwrap();
            assert!(!path.exists());
        }
    }
}
