//! A checkout's index and working-tree changes, captured as commits and put
//! back byte for byte.
//!
//! The index is written as a tree as it stands, and the entries added with
//! `git add -N` — which a tree cannot hold — are listed. The working tree is
//! captured as a second tree holding what differs from the index — every
//! file `status` names, untracked ones included, and every file git was told
//! to leave alone (`skip-worktree`, `assume-unchanged`), whose edits
//! `status` never shows — plus any other paths the caller names: each hashed
//! exactly as it is on disk, with no line-ending conversion or filter
//! applied, and a link as its target. A file's permission bits, and whether
//! a link points at a folder, are recorded beside it. Paths `status` names
//! that hold no file — deleted, or a folder standing where the index has a
//! file — are listed. Both trees are held by commits, and the commits by
//! refs, so nothing git prunes can take them.
//!
//! Putting back first removes each path that held no file, then writes each
//! captured file's bytes exactly with its permissions, reads the index tree
//! back into the index, adds the `-N` entries again, and marks the flagged
//! files as they were. A file whose content matches its commit but whose
//! line endings do not — and which neither the caller named nor `status`
//! reports — comes back as a checkout writes it; and two paths hard-linked
//! to one file come back as two files with the same bytes.
//!
//! **Onto a branch that moved on.** A checkout is put back on its branch as
//! it stands, and the branch may have moved on from the commit the snapshot
//! was taken over — a commit published to it, or the branch following its
//! record, while the checkout was set aside. The index tree then holds the
//! old commit's files, and reading it back whole would stage the reverse of
//! everything the branch gained. So the checkout's own staged changes — the
//! index tree against the commit it was taken over — are laid onto the
//! branch as it stands instead, and a file captured only to keep its bytes,
//! which the checkout had not changed, is not written over a path the branch
//! changed since.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
#[cfg(windows)]
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use super::journal::Flag;
use crate::checkout::materialize;
use crate::checkout::target::{self, Found};
use crate::checkout::CheckoutError;
use crate::runner::utf8;
use crate::write::scratch::PrivateIndex;
use crate::{
    DiffEntry, DiffStatus, FileMode, GitError, GitTime, Oid, Repo, Rev, Signature, StatusCode,
    StatusEntry,
};

/// Who the captured commits are recorded as, whatever identity the
/// repository has — or lacks.
const RECORDER: (&str, &str) = ("zend", "zend@localhost");

/// Paths per `git add -N`, well inside a command line's length.
const BATCH: usize = 100;

/// What [`capture`] took.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Snapshot {
    /// The commit holding the index.
    pub index: Oid,
    /// The commit holding every captured working-tree file; `None` when
    /// nothing differed.
    pub files: Option<Oid>,
    pub deleted: Vec<String>,
    pub flags: Vec<Flag>,
    pub intent_to_add: Vec<String>,
    pub perms: BTreeMap<String, u32>,
    pub link_dirs: Vec<String>,
    /// Captured paths the checkout had not changed — taken only so that
    /// their bytes, line endings and all, come back exactly.
    pub unchanged: Vec<String>,
}

impl Snapshot {
    /// Every path the checkout had changed itself over `taken_over`, the
    /// commit the snapshot was taken over: staged, in the working tree,
    /// deleted, added with `-N`, or flagged to be left alone.
    pub fn own_changes(
        &self,
        repo: &Repo,
        taken_over: &Oid,
    ) -> Result<BTreeSet<String>, CheckoutError> {
        let mut own: BTreeSet<String> = staged(repo, taken_over, &self.index)?
            .iter()
            .flat_map(|e| e.old.iter().chain(&e.new))
            .map(|side| side.path.as_str().to_string())
            .collect();
        if let Some(files) = &self.files {
            own.extend(
                repo.ls_tree_all(files)?
                    .into_iter()
                    .map(|sized| sized.entry.path.as_str().to_string())
                    .filter(|path| !self.unchanged.contains(path)),
            );
        }
        own.extend(self.deleted.iter().cloned());
        own.extend(self.intent_to_add.iter().cloned());
        own.extend(self.flags.iter().map(|f| f.path.clone()));
        Ok(own)
    }
}

/// Capture `repo`'s index and working-tree changes over `head`, and the
/// file at each of `also` whatever `status` says of it. Changes nothing in
/// the checkout. An index holding unresolved conflicts cannot be written as
/// a tree, and is refused.
pub(super) fn capture(
    repo: &Repo,
    head: &Oid,
    why: &str,
    also: &[String],
) -> Result<Snapshot, CheckoutError> {
    let me = recorder()?;
    let index_tree = {
        let _write = repo.write_lock();
        let out = repo.git("write-tree").run().map_err(CheckoutError::Git)?;
        if out.status != Some(0) {
            return Err(GitError::invalid(format!(
                "the checkout's index cannot be set aside — it holds unresolved conflicts, \
                 or entries git cannot write as a tree: {}",
                out.stderr.trim()
            ))
            .into());
        }
        Oid::parse(utf8("write-tree", out.stdout)?.trim())?
    };
    let index = repo.commit_tree(
        &index_tree,
        &[head],
        &format!("zend: the index while {why}"),
        &me,
        &me,
    )?;

    let flags = flags(repo)?;
    let modes = index_modes(repo)?;
    // Paths `status` names, which hold no file when there is none there;
    // and paths that are captured only where a file stands.
    let mut named: BTreeSet<String> = BTreeSet::new();
    let mut intent_to_add = Vec::new();
    for entry in repo.status()? {
        match &entry {
            StatusEntry::Renamed { from, .. } => {
                named.insert(from.as_str().to_string());
            }
            StatusEntry::Changed { xy, path, .. }
                if xy.index == StatusCode::Unmodified && xy.worktree == StatusCode::Added =>
            {
                intent_to_add.push(path.as_str().to_string());
            }
            _ => {}
        }
        named.insert(entry.path().as_str().to_string());
    }
    let mut present_only: BTreeSet<String> = flags.iter().map(|f| f.path.clone()).collect();
    present_only.extend(also.iter().cloned());

    let unchanged: Vec<String> = also
        .iter()
        .filter(|p| !named.contains(*p) && !flags.iter().any(|f| &f.path == *p))
        .cloned()
        .collect();

    let root = repo.dir().to_path_buf();
    let mut regular: Vec<(String, FileMode)> = Vec::new();
    let mut entries: Vec<(FileMode, Oid, String)> = Vec::new();
    let mut deleted = Vec::new();
    let mut perms = BTreeMap::new();
    let mut link_dirs = Vec::new();
    let mut links: BTreeSet<String> = BTreeSet::new();
    let mut take_link = |path: &str, abs: &Path| -> Result<(), CheckoutError> {
        if !links.insert(path.to_string()) {
            return Ok(());
        }
        let points_to = std::fs::read_link(abs)
            .map_err(|e| CheckoutError::io(path, e))?
            .to_string_lossy()
            .replace('\\', "/");
        if std::fs::metadata(abs).is_ok_and(|m| m.is_dir()) {
            link_dirs.push(path.to_string());
        }
        let oid = hash_bytes(repo, points_to.into_bytes())?;
        entries.push((FileMode::Symlink, oid, path.to_string()));
        Ok(())
    };
    let paths = named.iter().map(|p| (p, true)).chain(
        present_only
            .iter()
            .filter(|p| !named.contains(*p))
            .map(|p| (p, false)),
    );
    for (path, is_named) in paths {
        let (target, found) = target::inspect(&root, path)?;
        match found {
            Found::File => {
                let mode = file_mode(&target.abs, modes.get(path).copied());
                if let Some(bits) = permission_bits(&target.abs) {
                    perms.insert(path.clone(), bits);
                }
                regular.push((path.clone(), mode));
            }
            Found::Link => take_link(path, &target.abs)?,
            // What lies behind a link is outside the checkout; the link on
            // the way is the checkout's own, and is what is kept — git lists
            // a junction's files rather than the junction.
            Found::BehindLink { rel, abs } => take_link(&rel, &abs)?,
            // Nothing there — deleted, or `git rm`'d, or the source of a
            // rename — or a folder where the index has a file: the path
            // holds no file. A folder that is a repository of its own is
            // never removed, so listing it changes nothing.
            Found::Absent | Found::Folder if is_named => deleted.push(path.clone()),
            // A path only named to be captured where it stands, standing
            // empty.
            Found::Absent | Found::Folder => {}
        }
    }
    let oids = hash_paths(repo, regular.iter().map(|(p, _)| p.as_str()))?;
    for ((path, mode), oid) in regular.into_iter().zip(oids) {
        entries.push((mode, oid, path));
    }

    let files = if entries.is_empty() {
        None
    } else {
        let tree = tree_of(repo, &entries)?;
        Some(repo.commit_tree(
            &tree,
            &[&index],
            &format!("zend: the working tree while {why}"),
            &me,
            &me,
        )?)
    };
    Ok(Snapshot {
        index,
        files,
        deleted,
        flags,
        intent_to_add,
        perms,
        link_dirs,
        unchanged,
    })
}

/// Put `snapshot`, taken over the commit `taken_over`, back into `repo`'s
/// checkout, which a checkout of its branch has just put at `now` — that
/// commit, or one the branch moved on to. See the module. Every step is safe
/// to repeat.
pub(super) fn restore(
    repo: &Repo,
    snapshot: &Snapshot,
    taken_over: &Oid,
    now: &Oid,
) -> Result<(), CheckoutError> {
    let moved_on = now != taken_over;
    // What the branch changed since: a file the checkout had not changed is
    // the branch's there now.
    let gained: BTreeSet<String> = if moved_on {
        repo.diff(&Rev::Oid(taken_over.clone()), &Rev::Oid(now.clone()), &[])?
            .iter()
            .flat_map(|e| e.old.iter().chain(&e.new))
            .map(|side| side.path.as_str().to_string())
            .collect()
    } else {
        BTreeSet::new()
    };
    let root = repo.dir().to_path_buf();
    let mut cleared = Vec::new();
    // What held no file first: a file standing where a captured folder's
    // files go, or a folder where a captured file goes, is out of the way.
    for path in &snapshot.deleted {
        materialize::remove(&root, path, &mut cleared)?;
    }
    if let Some(files) = &snapshot.files {
        for sized in repo.ls_tree_all(files)? {
            let entry = sized.entry;
            if !entry.mode.is_blob() {
                continue;
            }
            let path = entry.path.as_str();
            if gained.contains(path) && snapshot.unchanged.iter().any(|u| u == path) {
                continue;
            }
            materialize::clear_the_way(&root, path, &mut cleared)?;
            let bytes = repo
                .git("cat-file")
                .args(["blob", "--end-of-options"])
                .arg(entry.oid.as_str())
                .read_only()
                .run_ok()?;
            let abs = root.join(path);
            if entry.mode == FileMode::Symlink {
                materialize::remove(&root, path, &mut cleared)?;
                if let Some(parent) = abs.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| CheckoutError::io(path, e))?;
                }
                let to_folder = snapshot.link_dirs.iter().any(|d| d == path);
                link(&String::from_utf8_lossy(&bytes), &abs, to_folder)
                    .map_err(|e| CheckoutError::io(path, e))?;
            } else {
                materialize::write(&abs, path, &bytes)?;
                set_mode(
                    &abs,
                    entry.mode == FileMode::Executable,
                    snapshot.perms.get(path).copied(),
                )
                .map_err(|e| CheckoutError::io(path, e))?;
            }
        }
    }
    // Read before the write lock: the diff runs git of its own.
    let own_staged = if moved_on {
        Some(staged(repo, taken_over, &snapshot.index)?)
    } else {
        None
    };
    {
        let _write = repo.write_lock();
        match &own_staged {
            // The checkout's index, as it was over the commit it stands on.
            None => {
                repo.git("read-tree")
                    .arg("--end-of-options")
                    .arg(snapshot.index.as_str())
                    .about_rev(snapshot.index.as_str())
                    .run_ok()?;
            }
            // Its staged changes, laid onto the branch as it stands: the
            // checkout of `now` left the index at `now`'s tree.
            Some(entries) if !entries.is_empty() => {
                repo.git("update-index")
                    .args(["-z", "--index-info"])
                    .stdin(index_info(entries))
                    .run_ok()?;
            }
            Some(_) => {}
        }
        for batch in snapshot.intent_to_add.chunks(BATCH) {
            repo.git("add")
                .args(["--intent-to-add", "--"])
                .args(batch.iter().map(String::as_str))
                .run_ok()?;
        }
        let marked = |flag: &str, pick: fn(&Flag) -> bool| -> Result<(), GitError> {
            let list: Vec<u8> = snapshot
                .flags
                .iter()
                .filter(|f| pick(f))
                .flat_map(|f| f.path.bytes().chain(std::iter::once(0)))
                .collect();
            if list.is_empty() {
                return Ok(());
            }
            repo.git("update-index")
                .args([flag, "-z", "--stdin"])
                .stdin(list)
                .run_ok()
                .map(drop)
        };
        marked("--skip-worktree", |f| f.skip_worktree)?;
        marked("--assume-unchanged", |f| f.assume_unchanged)?;
        // The index read back has no stat information: refreshing it saves
        // the next status a hash of every file. Exit 1 only means it found
        // changes, which are the checkout's own.
        repo.git("update-index")
            .args(["-q", "--refresh"])
            .run_accepting(&[0, 1])?;
    }
    Ok(())
}

/// The checkout's staged changes: its index, held by the commit `index`,
/// against the commit it was taken over.
fn staged(repo: &Repo, taken_over: &Oid, index: &Oid) -> Result<Vec<DiffEntry>, CheckoutError> {
    Ok(repo.diff(&Rev::Oid(taken_over.clone()), &Rev::Oid(index.clone()), &[])?)
}

/// `update-index --index-info` input setting each of `entries`' paths as
/// the change leaves it: a path it removes — deleted, or the source of a
/// rename — is dropped from the index, one it leaves in place is set.
fn index_info(entries: &[DiffEntry]) -> Vec<u8> {
    let mut info = Vec::new();
    for entry in entries {
        let kept = matches!(entry.status, DiffStatus::Copied(_));
        if let Some(old) = entry.old.as_ref().filter(|_| !kept) {
            let moved = entry.new.as_ref().is_none_or(|new| new.path != old.path);
            if moved {
                let none = "0".repeat(old.oid.as_ref().map_or(40, |o| o.as_str().len()));
                info.extend(format!("0 {none}\t{}\0", old.path.as_str()).into_bytes());
            }
        }
        if let Some(new) = &entry.new {
            if let Some(oid) = &new.oid {
                let line = format!("{} {oid}\t{}\0", new.mode.as_str(), new.path.as_str());
                info.extend(line.into_bytes());
            }
        }
    }
    info
}

fn recorder() -> Result<Signature, CheckoutError> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Ok(Signature::new(
        RECORDER.0,
        RECORDER.1,
        GitTime {
            seconds,
            offset_minutes: 0,
        },
    )?)
}

/// Every index entry git was told to leave alone, from `ls-files -v`: `S`
/// (or `s`) marks skip-worktree, a lower-case tag assume-unchanged.
fn flags(repo: &Repo) -> Result<Vec<Flag>, CheckoutError> {
    let out = repo
        .git("ls-files")
        .args(["-v", "-z"])
        .read_only()
        .run_ok()?;
    let text = utf8("ls-files", out)?;
    Ok(text
        .split('\0')
        .filter_map(|record| {
            let (tag, path) = record.split_once(' ')?;
            let tag = tag.chars().next()?;
            let skip_worktree = tag.eq_ignore_ascii_case(&'S');
            let assume_unchanged = tag.is_ascii_lowercase();
            (skip_worktree || assume_unchanged).then(|| Flag {
                path: path.to_string(),
                skip_worktree,
                assume_unchanged,
            })
        })
        .collect())
}

/// Every path the index holds, with its mode.
fn index_modes(repo: &Repo) -> Result<BTreeMap<String, FileMode>, CheckoutError> {
    let out = repo
        .git("ls-files")
        .args(["-s", "-z"])
        .read_only()
        .run_ok()?;
    let text = utf8("ls-files", out)?;
    let mut modes = BTreeMap::new();
    for record in text.split('\0').filter(|r| !r.is_empty()) {
        let bad = || GitError::malformed("ls-files", record.to_string());
        let (meta, path) = record.split_once('\t').ok_or_else(bad)?;
        let mode = meta.split(' ').next().ok_or_else(bad)?;
        modes.insert(path.to_string(), FileMode::parse(mode)?);
    }
    Ok(modes)
}

/// The mode to record a file at `abs` with: executable as the file system
/// says where it can say, as the index says elsewhere.
fn file_mode(abs: &Path, indexed: Option<FileMode>) -> FileMode {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = indexed;
        match std::fs::metadata(abs) {
            Ok(meta) if meta.permissions().mode() & 0o111 != 0 => FileMode::Executable,
            _ => FileMode::Regular,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = abs;
        match indexed {
            Some(FileMode::Executable) => FileMode::Executable,
            _ => FileMode::Regular,
        }
    }
}

/// A file's permission bits, where the file system keeps them.
fn permission_bits(abs: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(abs)
            .ok()
            .map(|m| m.permissions().mode() & 0o7777)
    }
    #[cfg(not(unix))]
    {
        let _ = abs;
        None
    }
}

/// A restored file's permissions: exactly `bits` where they were recorded,
/// otherwise the executable bits as its mode says.
fn set_mode(abs: &Path, executable: bool, bits: Option<u32>) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(abs)?.permissions();
        let mode = permissions.mode();
        permissions.set_mode(match bits {
            Some(bits) => (mode & !0o7777) | bits,
            None if executable => mode | 0o111,
            None => mode & !0o111,
        });
        std::fs::set_permissions(abs, permissions)
    }
    #[cfg(not(unix))]
    {
        let _ = (abs, executable, bits);
        Ok(())
    }
}

/// A link at `abs` to `points_to`. On Windows a link to a folder is made as
/// a folder link, or — where making links needs a privilege the process
/// lacks — as a junction, which needs none.
fn link(points_to: &str, abs: &Path, to_folder: bool) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let _ = to_folder;
        std::os::unix::fs::symlink(points_to, abs)
    }
    #[cfg(windows)]
    {
        let native = points_to.replace('/', "\\");
        if !to_folder {
            return std::os::windows::fs::symlink_file(native, abs);
        }
        match std::os::windows::fs::symlink_dir(&native, abs) {
            Ok(()) => Ok(()),
            Err(_) => {
                // A junction names its folder absolutely.
                let target = abs
                    .parent()
                    .map(|p| p.join(&native))
                    .unwrap_or_else(|| native.clone().into());
                let out = Command::new("cmd")
                    .args(["/C", "mklink", "/J"])
                    .arg(abs)
                    .arg(target)
                    .output()?;
                if out.status.success() {
                    Ok(())
                } else {
                    Err(std::io::Error::other(
                        String::from_utf8_lossy(&out.stderr).trim().to_string(),
                    ))
                }
            }
        }
    }
}

/// `bytes` stored as a blob, unconverted.
fn hash_bytes(repo: &Repo, bytes: Vec<u8>) -> Result<Oid, CheckoutError> {
    let out = repo
        .git("hash-object")
        .args(["-w", "--no-filters", "--stdin"])
        .stdin(bytes)
        .run_ok()?;
    Ok(Oid::parse(utf8("hash-object", out)?.trim())?)
}

/// Each file at `paths` stored as a blob, unconverted, in order.
fn hash_paths<'a>(
    repo: &Repo,
    paths: impl Iterator<Item = &'a str>,
) -> Result<Vec<Oid>, CheckoutError> {
    let list: Vec<u8> = paths
        .flat_map(|p| p.bytes().chain(std::iter::once(b'\n')))
        .collect();
    if list.is_empty() {
        return Ok(Vec::new());
    }
    let out = repo
        .git("hash-object")
        .args(["-w", "--no-filters", "--stdin-paths"])
        .stdin(list)
        .run_ok()?;
    utf8("hash-object", out)?
        .lines()
        .map(|l| Oid::parse(l.trim()).map_err(CheckoutError::from))
        .collect()
}

/// A tree holding exactly `entries`.
fn tree_of(repo: &Repo, entries: &[(FileMode, Oid, String)]) -> Result<Oid, CheckoutError> {
    let index = PrivateIndex::new(repo.git_dir());
    let info: Vec<u8> = entries
        .iter()
        .flat_map(|(mode, oid, path)| format!("{} {oid}\t{path}\0", mode.as_str()).into_bytes())
        .collect();
    repo.git("update-index")
        .args(["-z", "--index-info"])
        .env("GIT_INDEX_FILE", &index.0)
        .stdin(info)
        .run_ok()?;
    let tree = repo
        .git("write-tree")
        .env("GIT_INDEX_FILE", &index.0)
        .run_ok()?;
    Ok(Oid::parse(utf8("write-tree", tree)?.trim())?)
}
