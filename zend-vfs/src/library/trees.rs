//! Tree listings, from libgit2: what a revision's folders hold, in the order
//! `ls-tree` prints it — git's tree order — with the paths and modes it prints.

use std::path::Path;

use git2::{ObjectType, Repository, Tree};

use super::{failed, is_absent, oid_of, revision};
use crate::error::GitError;
use crate::read::tree::{ObjectKind, SizedEntry, TreeEntry};
use crate::types::{FileMode, Oid, RepoPath, Rev};

/// The tree `spec` names, peeled from a commit or tag if need be.
fn tree_of<'r>(lib: &'r Repository, spec: &str) -> Result<Tree<'r>, GitError> {
    lib.revparse_single(spec)
        .and_then(|object| object.peel_to_tree())
        .map_err(|e| revision(e, spec))
}

/// One entry as `ls-tree` prints it, its path made from the folder it is in.
fn entry_of(folder: &str, entry: &git2::TreeEntry<'_>) -> Result<TreeEntry, GitError> {
    let name = entry
        .name()
        .ok_or_else(|| GitError::malformed("ls-tree", "a path that is not UTF-8"))?;
    let path = if folder.is_empty() {
        name.to_string()
    } else {
        format!("{folder}/{name}")
    };
    let kind = match entry.kind() {
        Some(ObjectType::Blob) => ObjectKind::Blob,
        Some(ObjectType::Tree) => ObjectKind::Tree,
        Some(ObjectType::Commit) => ObjectKind::Commit,
        _ => return Err(GitError::malformed("ls-tree", path)),
    };
    Ok(TreeEntry {
        mode: FileMode::parse(&format!("{:06o}", entry.filemode()))?,
        kind,
        oid: oid_of(entry.id())?,
        path: RepoPath::parse(&path)?,
    })
}

/// The entries directly inside `dir` at `rev` — the root when `dir` is `None`.
pub(crate) fn ls_tree(
    lib: &Repository,
    rev: &Rev,
    dir: Option<&RepoPath>,
) -> Result<Vec<TreeEntry>, GitError> {
    let root = tree_of(lib, &rev.spec())?;
    let (folder, tree) = match dir {
        None => (String::new(), root),
        Some(dir) => {
            let inside = match root.get_path(Path::new(dir.as_str())) {
                Ok(entry) if entry.kind() == Some(ObjectType::Tree) => entry,
                // A file, a submodule, or nothing: no folder to list.
                Ok(_) => return Ok(Vec::new()),
                Err(e) if is_absent(&e) => return Ok(Vec::new()),
                Err(e) => return Err(failed("get_path", e)),
            };
            let tree = lib
                .find_tree(inside.id())
                .map_err(|e| failed("find_tree", e))?;
            (dir.as_str().to_string(), tree)
        }
    };
    tree.iter().map(|entry| entry_of(&folder, &entry)).collect()
}

/// Every entry of the tree `tree`, at any depth — folders included, each before
/// what it holds — with every blob's size. `None` when a blob's size cannot be
/// read from what the repository holds (a partial clone has not fetched it),
/// which `git` fetches on demand and this does not.
pub(crate) fn ls_tree_all(
    lib: &Repository,
    tree: &Oid,
) -> Result<Option<Vec<SizedEntry>>, GitError> {
    let id = git2::Oid::from_str(tree.as_str()).map_err(|e| failed("from_str", e))?;
    let root = lib
        .find_object(id, None)
        .and_then(|object| object.peel_to_tree())
        .map_err(|e| revision(e, tree.as_str()))?;
    let odb = lib.odb().map_err(|e| failed("odb", e))?;
    let mut out = Vec::new();
    if !walk_all(lib, &odb, &root, "", &mut out)? {
        return Ok(None);
    }
    Ok(Some(out))
}

/// [`ls_tree_all`]'s walk of one folder; `false` once a blob's size is not
/// there to be read.
fn walk_all(
    lib: &Repository,
    odb: &git2::Odb<'_>,
    tree: &Tree<'_>,
    folder: &str,
    out: &mut Vec<SizedEntry>,
) -> Result<bool, GitError> {
    for entry in tree.iter() {
        let listed = entry_of(folder, &entry)?;
        let size = match listed.kind {
            ObjectKind::Blob => match odb.read_header(entry.id()) {
                Ok((size, _)) => Some(size as u64),
                Err(e) if is_absent(&e) => return Ok(false),
                Err(e) => return Err(failed("read_header", e)),
            },
            ObjectKind::Tree | ObjectKind::Commit => None,
        };
        let descend = listed.kind == ObjectKind::Tree;
        let path = listed.path.as_str().to_string();
        out.push(SizedEntry {
            entry: listed,
            size,
        });
        if descend {
            let inner = lib
                .find_tree(entry.id())
                .map_err(|e| failed("find_tree", e))?;
            if !walk_all(lib, odb, &inner, &path, out)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// The entries at exactly `paths` in `rev` — a file's blob or a folder's tree,
/// in tree order. Paths `rev` does not hold are absent from the result.
pub(crate) fn tree_entries(
    lib: &Repository,
    rev: &Rev,
    paths: &[&RepoPath],
) -> Result<Vec<TreeEntry>, GitError> {
    let root = tree_of(lib, &rev.spec())?;
    let wanted: Vec<&str> = paths.iter().map(|p| p.as_str()).collect();
    let mut out = Vec::new();
    pick(lib, &root, "", &wanted, &mut out)?;
    Ok(out)
}

/// Walk `tree` in order, keeping the entries whose paths are wanted, and going
/// into a folder only when a wanted path lies beneath it.
fn pick(
    lib: &Repository,
    tree: &Tree<'_>,
    folder: &str,
    wanted: &[&str],
    out: &mut Vec<TreeEntry>,
) -> Result<(), GitError> {
    for entry in tree.iter() {
        let listed = entry_of(folder, &entry)?;
        let path = listed.path.as_str().to_string();
        let below = format!("{path}/");
        let descend =
            listed.kind == ObjectKind::Tree && wanted.iter().any(|w| w.starts_with(&below));
        if wanted.contains(&path.as_str()) {
            out.push(listed);
        }
        if descend {
            let inner = lib
                .find_tree(entry.id())
                .map_err(|e| failed("find_tree", e))?;
            pick(lib, &inner, &path, wanted, out)?;
        }
    }
    Ok(())
}
