//! The ignored files a run could clobber, moved out of its way and back.
//!
//! A captured snapshot holds every file git sees; an ignored file it does
//! not. A run leaves ignored files where they are — a build cache is the
//! point of reusing the folder — except where it lands something of its own:
//! at a path the conversation writes, and at a path the branch it is put on
//! tracks where the checkout's own branch does not (a forced checkout
//! overwrites an ignored file standing there, and the checkout back deletes
//! it). An ignored file, link or folder standing at such a path, or where
//! one of its folders would go, is moved aside into the preservation's
//! folder.
//!
//! Each move is journalled before it is made, and undone only from the copy
//! kept aside: a path the journal lists whose copy is not there was never
//! moved, or is back already, and whatever stands at it is left alone. So a
//! crash between journalling and moving, in either direction, loses nothing.
//!
//! A landing path with nothing standing at it is recorded as vacant:
//! whatever the run leaves there is the run's, and goes. Nothing else is
//! ever removed: a path whose file could not be moved is neither.

use std::collections::BTreeSet;
use std::path::Path;

use super::journal::{Journal, Place};
use crate::checkout::materialize;
use crate::checkout::target::{self, Found};
use crate::checkout::CheckoutError;
use crate::runner::utf8;
use crate::{Oid, Repo, RepoPath, Rev};

/// Where a run will land things, worked out before it starts.
pub(super) struct Landing {
    /// The paths it may land something at that `HEAD` holds nothing at —
    /// those it writes, those the branch at the target tracks, and each
    /// folder on the way to them.
    pub untracked: Vec<String>,
    /// The files `HEAD` tracks that the run will rewrite: those it writes,
    /// and those the target holds differently or not at all.
    pub tracked: Vec<String>,
}

/// Where a run that writes `writes` and puts the checkout on `target` lands
/// things.
pub(super) fn landing(
    repo: &Repo,
    writes: &[String],
    target: Option<&Oid>,
) -> Result<Landing, CheckoutError> {
    let mut paths: BTreeSet<String> = writes.iter().cloned().collect();
    let mut rewritten: BTreeSet<String> = writes.iter().cloned().collect();
    if let Some(target) = target {
        for entry in repo.diff(&Rev::Head, &Rev::Oid(target.clone()), &[])? {
            if let Some(new) = &entry.new {
                paths.insert(new.path.as_str().to_string());
            }
            if let Some(old) = &entry.old {
                rewritten.insert(old.path.as_str().to_string());
            }
        }
    }
    let with_folders: BTreeSet<String> = paths
        .iter()
        .flat_map(|p| {
            let parts: Vec<&str> = p.split('/').collect();
            (1..=parts.len())
                .map(move |n| parts[..n].join("/"))
                .collect::<Vec<_>>()
        })
        .collect();
    let held = held_at_head(repo, with_folders.iter().chain(rewritten.iter()))?;
    Ok(Landing {
        untracked: with_folders
            .into_iter()
            .filter(|p| !held.contains(p))
            .collect(),
        tracked: rewritten.into_iter().filter(|p| held.contains(p)).collect(),
    })
}

/// Which of `paths` `HEAD` holds anything at — a file or a folder.
fn held_at_head<'a>(
    repo: &Repo,
    paths: impl Iterator<Item = &'a String>,
) -> Result<BTreeSet<String>, CheckoutError> {
    /// Paths per `ls-tree`, well inside a command line's length.
    const BATCH: usize = 100;
    let parsed: Vec<RepoPath> = paths.filter_map(|p| RepoPath::parse(p).ok()).collect();
    let mut held = BTreeSet::new();
    for batch in parsed.chunks(BATCH) {
        let refs: Vec<&RepoPath> = batch.iter().collect();
        for entry in repo.tree_entries(&Rev::Head, &refs)? {
            held.insert(entry.path.as_str().to_string());
        }
    }
    Ok(held)
}

/// Which of `paths` the checkout's ignore rules ignore — a folder by the
/// rules that match folders, since each is looked at on disk.
pub(super) fn ignored(repo: &Repo, paths: &[String]) -> Result<BTreeSet<String>, CheckoutError> {
    if paths.is_empty() {
        return Ok(BTreeSet::new());
    }
    let list: Vec<u8> = paths
        .iter()
        .flat_map(|p| p.bytes().chain(std::iter::once(0)))
        .collect();
    // Exit 1: none of them is ignored.
    let out = repo
        .git("check-ignore")
        .args(["-z", "--stdin", "--no-index"])
        .env("GIT_LITERAL_PATHSPECS", "0")
        .stdin(list)
        .read_only()
        .run_accepting(&[0, 1])?;
    Ok(utf8("check-ignore", out.stdout)?
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect())
}

/// Move aside every ignored file, link or folder standing at one of
/// `paths`, and record the paths nothing stands at — each in `journal`
/// before it is done.
pub(super) fn set_aside(
    repo: &Repo,
    place: &Place,
    journal: &mut Journal,
    paths: &[String],
) -> Result<(), CheckoutError> {
    let root = repo.dir();
    let ignored = ignored(repo, paths)?;
    // Shortest first: a folder on the way is dealt with before what is in it.
    let mut ordered: Vec<&String> = paths.iter().collect();
    ordered.sort_by_key(|p| (p.matches('/').count(), p.as_str()));
    for path in ordered {
        if journal.moved.iter().any(|m| is_under(path, m)) {
            continue;
        }
        let (target, found) = target::inspect(root, path)?;
        match found {
            Found::Absent => {
                if !journal.vacant.iter().any(|v| is_under(path, v)) {
                    journal.vacant.push(path.clone());
                    place.save(journal)?;
                }
            }
            Found::File | Found::Link | Found::Folder if ignored.contains(path) => {
                let kept = place.kept(path);
                if let Some(parent) = kept.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| CheckoutError::io(path, e))?;
                }
                journal.moved.push(path.clone());
                place.save(journal)?;
                std::fs::rename(&target.abs, &kept).map_err(|e| CheckoutError::io(path, e))?;
            }
            // A folder not ignored is walked into by the paths under it; a
            // link on the way is moved aside when its own path comes up; and
            // an untracked file that is not ignored is in the snapshot.
            Found::File | Found::Link | Found::Folder | Found::BehindLink { .. } => {}
        }
    }
    Ok(())
}

/// Move every file aside back where it was — taking away what the run left
/// in its place, never through a link — and, given `clear_vacant`, take
/// away what the run left at the vacant paths; each recorded in `journal`
/// as it is done. A moved path whose copy is not kept aside is left as it
/// stands: it was never moved, or it is back.
pub(super) fn put_back(
    repo: &Repo,
    place: &Place,
    journal: &mut Journal,
    clear_vacant: bool,
) -> Result<(), CheckoutError> {
    let root = repo.dir();
    let mut gone = Vec::new();
    if clear_vacant {
        let held = held_at_head(repo, journal.vacant.iter())?;
        for path in journal.vacant.clone() {
            if !held.contains(&path) {
                clear(root, &path, &mut gone)?;
            }
        }
    }
    while let Some(path) = journal.moved.last().cloned() {
        let kept = place.kept(&path);
        if std::fs::symlink_metadata(&kept).is_ok() {
            clear(root, &path, &mut gone)?;
            let home = root.join(&path);
            if let Some(parent) = home.parent() {
                std::fs::create_dir_all(parent).map_err(|e| CheckoutError::io(&path, e))?;
            }
            std::fs::rename(&kept, &home).map_err(|e| CheckoutError::io(&path, e))?;
        }
        journal.moved.pop();
        place.save(journal)?;
    }
    Ok(())
}

/// Whether anything is kept aside for a moved path whose home is taken now
/// — which only someone other than the run can have done, so that putting it
/// back would take theirs.
pub(super) fn taken_homes(
    repo: &Repo,
    place: &Place,
    journal: &Journal,
) -> Result<Vec<String>, CheckoutError> {
    let mut taken = Vec::new();
    for path in &journal.moved {
        let kept = std::fs::symlink_metadata(place.kept(path)).is_ok();
        let (_, found) = target::inspect(repo.dir(), path)?;
        if kept && found != Found::Absent {
            taken.push(path.clone());
        }
    }
    Ok(taken)
}

/// Whatever stands at `path`, gone, without following a link: a file or a
/// link, a link on the way, or a folder the run made there, with all it
/// holds.
fn clear(root: &Path, path: &str, gone: &mut Vec<String>) -> Result<(), CheckoutError> {
    materialize::clear_the_way(root, path, gone)?;
    let (target, found) = target::inspect(root, path)?;
    match found {
        Found::File | Found::Link => materialize::remove(root, path, gone),
        Found::Folder => {
            std::fs::remove_dir_all(&target.abs).map_err(|e| CheckoutError::io(path, e))
        }
        Found::Absent | Found::BehindLink { .. } => Ok(()),
    }
}

/// Whether `path` is `folder` or lies inside it.
fn is_under(path: &str, folder: &str) -> bool {
    path == folder
        || path
            .strip_prefix(folder)
            .is_some_and(|rest| rest.starts_with('/'))
}
