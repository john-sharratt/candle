//! Turning a repository-relative path into the file in the checkout it may
//! write — and refusing every path that could lead anywhere else.
//!
//! A path arrives from a conversation's changes or from git's own listing, and
//! materialising writes it to disk. [`RepoPath`] already refuses what git
//! refuses — empty, absolute, `.` and `..` components, a backslash, a control
//! character, any `.git` component — and this adds what a Windows filesystem
//! would read differently from how it is spelled: a `:` (a drive, `C:`, or an
//! alternate data stream, `a.txt:hidden`) and a component ending in a dot or a
//! space (which Windows drops, so `.git.` opens `.git`). Then the path is
//! walked on disk: a symbolic link or junction anywhere on the way would lead
//! a write outside the checkout, so [`resolve`] refuses one.
//!
//! A tool run on the checkout can leave links of its own, and reading back or
//! clearing away what it did has to see them without following them.
//! [`inspect`] walks the same way but reports what it found — a link as the
//! path itself, or one on the way — instead of refusing it.

use std::path::{Path, PathBuf};

use super::error::CheckoutError;
use crate::RepoPath;

/// The parsed path and the absolute file it names in the checkout at `root`.
pub(crate) struct Target {
    pub repo_path: RepoPath,
    pub abs: PathBuf,
}

/// What stands at a path in the checkout, seen without following any link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Found {
    Absent,
    File,
    Folder,
    /// The path itself is a symbolic link or junction.
    Link,
    /// A folder on the way is a link: `rel` (repository-relative) at `abs`.
    /// What lies past it is outside the checkout's control.
    BehindLink {
        rel: String,
        abs: PathBuf,
    },
}

/// `path` inside the checkout at `root`, refused unless every component is a
/// plain name and nothing on the way to it — the file included — is a link.
/// The file itself need not exist.
pub(crate) fn resolve(root: &Path, path: &str) -> Result<Target, CheckoutError> {
    let (target, found) = inspect(root, path)?;
    match found {
        Found::Link | Found::BehindLink { .. } => Err(CheckoutError::unsafe_path(
            path,
            "a symbolic link or junction on the way could lead outside the checkout",
        )),
        _ => Ok(target),
    }
}

/// `path` inside the checkout at `root` and what stands there. Refuses a
/// path that is not a plain repository path; a link is reported, not
/// refused, and never followed.
pub(crate) fn inspect(root: &Path, path: &str) -> Result<(Target, Found), CheckoutError> {
    let repo_path =
        RepoPath::parse(path).map_err(|e| CheckoutError::unsafe_path(path, &e.to_string()))?;
    let components: Vec<String> = repo_path.as_str().split('/').map(str::to_string).collect();
    for component in &components {
        if component.contains(':') {
            return Err(CheckoutError::unsafe_path(
                path,
                "a `:` names a drive or an alternate data stream, not a file",
            ));
        }
        if component.ends_with(['.', ' ']) {
            return Err(CheckoutError::unsafe_path(
                path,
                "a name ending in a dot or a space opens a different name on Windows",
            ));
        }
    }
    let target = Target {
        abs: root.join(repo_path.as_str()),
        repo_path,
    };
    let mut abs = root.to_path_buf();
    for (i, component) in components.iter().enumerate() {
        abs.push(component);
        let meta = match std::fs::symlink_metadata(&abs) {
            Ok(meta) => meta,
            // Nothing there: nothing further down exists either.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((target, Found::Absent));
            }
            Err(e) => return Err(CheckoutError::io(path, e)),
        };
        let last = i + 1 == components.len();
        if meta.file_type().is_symlink() {
            let found = if last {
                Found::Link
            } else {
                Found::BehindLink {
                    rel: components[..=i].join("/"),
                    abs: abs.clone(),
                }
            };
            return Ok((target, found));
        }
        if last {
            let found = if meta.is_dir() {
                Found::Folder
            } else {
                Found::File
            };
            return Ok((target, found));
        }
        if !meta.is_dir() {
            // A file where a folder is needed: nothing can be past it.
            return Ok((target, Found::Absent));
        }
    }
    unreachable!("a repository path has at least one component")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(root: &Path, path: &str) -> String {
        match resolve(root, path) {
            Err(CheckoutError::UnsafePath { why, .. }) => why,
            Err(other) => panic!("{path}: {other}"),
            Ok(_) => panic!("{path} was allowed"),
        }
    }

    #[test]
    fn a_plain_path_resolves_under_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let t = resolve(dir.path(), "src/new dir/café.rs").unwrap();
        assert_eq!(t.abs, dir.path().join("src/new dir/café.rs"));
        assert_eq!(t.repo_path.as_str(), "src/new dir/café.rs");
    }

    /// **Every spelling that leaves the checkout, or enters its git database,
    /// is refused** — including the ones only Windows reads differently.
    #[test]
    fn escaping_and_aliasing_paths_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        for path in [
            "",
            "/etc/passwd",
            "../outside",
            "a/../../outside",
            "a\\b",
            ".git/config",
            "sub/.GIT/hooks/pre-commit",
            "C:/Windows/win.ini",
            "C:x",
            "notes.txt:hidden",
            ".git./config",
            "trailing /x",
            "dot./x",
            "nul\0byte",
        ] {
            refused(dir.path(), path);
        }
    }

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
            assert!(out.status.success(), "mklink /J failed");
        }
    }

    /// **A link on the way — to a folder or as the file itself — is
    /// refused**, since writing through it could land outside the checkout.
    #[test]
    fn a_link_on_the_way_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let root = dir.path();
        link_dir(outside.path(), &root.join("linked"));
        let why = refused(root, "linked/file.txt");
        assert!(why.contains("link"), "{why}");
        refused(root, "linked");
    }

    /// **Inspection names what stands at a path without following a link**:
    /// a file, a folder, nothing, a link as the path, a link on the way (named
    /// by its own path), and a file standing where a folder would be.
    #[test]
    fn inspection_reports_links_without_following_them() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("there.txt"), b"x").unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("sub/deeper")).unwrap();
        std::fs::write(root.join("sub/file.txt"), b"x").unwrap();
        link_dir(outside.path(), &root.join("sub").join("linked"));
        let found = |path: &str| inspect(root, path).unwrap().1;

        assert_eq!(found("sub/file.txt"), Found::File);
        assert_eq!(found("sub/deeper"), Found::Folder);
        assert_eq!(found("sub/absent.txt"), Found::Absent);
        assert_eq!(found("absent/deeper/x.txt"), Found::Absent);
        assert_eq!(found("sub/file.txt/under"), Found::Absent);
        assert_eq!(found("sub/linked"), Found::Link);
        assert_eq!(
            found("sub/linked/there.txt"),
            Found::BehindLink {
                rel: "sub/linked".into(),
                abs: root.join("sub").join("linked"),
            }
        );
        // Still refused, as a resolve refuses it.
        assert!(inspect(root, "../outside").is_err());
        assert!(inspect(root, "notes.txt:hidden").is_err());
    }
}
