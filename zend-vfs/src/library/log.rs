//! Commit history from libgit2, as `git log` gives it.
//!
//! - **The walk** is git's own: newest first by commit date, a tie going to
//!   the commit queued first, from the range's tip and hiding what its
//!   excludes reach.
//! - **A path limit** keeps the commits that change those paths, with git's
//!   default history simplification: a merge the same at those paths as one
//!   of its parents is not shown, and only that parent's line is followed.
//! - **A file's history** follows it back through renames, as `--follow`
//!   does: where a commit adds the path, a rename that commit made names the
//!   path to follow before it.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

use git2::{Commit, Delta, DiffFindOptions, DiffOptions, Oid as LibOid, Repository, Time};

use super::{failed, oid_of, revision};
use crate::error::GitError;
use crate::read::log::{CommitInfo, LogRange};
use crate::types::{GitTime, RepoPath, Rev, Signature};

fn commit_id(lib: &Repository, rev: &Rev) -> Result<LibOid, GitError> {
    let spec = rev.spec();
    let object = lib.revparse_single(&spec).map_err(|e| revision(e, &spec))?;
    let commit = object.peel_to_commit().map_err(|e| revision(e, &spec))?;
    Ok(commit.id())
}

fn time(t: Time) -> GitTime {
    GitTime {
        seconds: t.seconds(),
        offset_minutes: t.offset_minutes(),
    }
}

/// `commit` as the layer describes one.
pub(crate) fn info(commit: &Commit<'_>) -> Result<CommitInfo, GitError> {
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
    let (author, committer) = (commit.author(), commit.committer());
    Ok(CommitInfo {
        oid: oid_of(commit.id())?,
        parents: commit.parent_ids().map(oid_of).collect::<Result<_, _>>()?,
        author: Signature::recorded(
            &text(author.name_bytes()),
            &text(author.email_bytes()),
            time(author.when()),
        ),
        committer: Signature::recorded(
            &text(committer.name_bytes()),
            &text(committer.email_bytes()),
            time(committer.when()),
        ),
        message: text(commit.message_raw_bytes()),
    })
}

/// Every commit the range's excludes reach.
fn hidden(lib: &Repository, range: &LogRange) -> Result<HashSet<LibOid>, GitError> {
    let mut hidden = HashSet::new();
    if range.exclude.is_empty() {
        return Ok(hidden);
    }
    let mut walk = lib.revwalk().map_err(|e| failed("revwalk", e))?;
    for exclude in &range.exclude {
        walk.push(commit_id(lib, exclude)?)
            .map_err(|e| failed("revwalk_push", e))?;
    }
    for id in walk {
        hidden.insert(id.map_err(|e| failed("revwalk", e))?);
    }
    Ok(hidden)
}

/// Git's own walk: the newest commit by commit date taken first, a tie going
/// to the one queued first, and the parents `follow` names for each commit
/// queued in that order as it is taken. `visit` is handed each commit taken,
/// and says which of its parents to follow; the walk ends once it says stop.
fn git_walk(
    lib: &Repository,
    range: &LogRange,
    mut visit: impl FnMut(&Commit<'_>) -> Result<Option<Vec<LibOid>>, GitError>,
) -> Result<(), GitError> {
    let hidden = hidden(lib, range)?;
    let mut queued: BinaryHeap<(i64, Reverse<u64>, LibOid)> = BinaryHeap::new();
    let mut seen: HashSet<LibOid> = HashSet::new();
    let mut order = 0u64;
    let mut queue = |queued: &mut BinaryHeap<_>,
                     seen: &mut HashSet<LibOid>,
                     id: LibOid|
     -> Result<(), GitError> {
        if hidden.contains(&id) || !seen.insert(id) {
            return Ok(());
        }
        let when = lib
            .find_commit(id)
            .map_err(|e| failed("find_commit", e))?
            .committer()
            .when()
            .seconds();
        order += 1;
        queued.push((when, Reverse(order), id));
        Ok(())
    };
    queue(&mut queued, &mut seen, commit_id(lib, &range.to)?)?;
    while let Some((_, _, id)) = queued.pop() {
        let commit = lib.find_commit(id).map_err(|e| failed("find_commit", e))?;
        let Some(follow) = visit(&commit)? else {
            return Ok(());
        };
        for parent in follow {
            queue(&mut queued, &mut seen, parent)?;
        }
    }
    Ok(())
}

/// What each of `paths` is in `commit`'s tree: its id there, or `None`.
fn at_paths(commit: &Commit<'_>, paths: &[RepoPath]) -> Result<Vec<Option<LibOid>>, GitError> {
    let tree = commit.tree().map_err(|e| failed("tree", e))?;
    Ok(paths
        .iter()
        .map(|p| tree.get_path(Path::new(p.as_str())).ok().map(|e| e.id()))
        .collect())
}

/// Up to `limit` commits in `range`, newest first.
pub(crate) fn log(
    lib: &Repository,
    range: &LogRange,
    limit: usize,
) -> Result<Vec<CommitInfo>, GitError> {
    let mut out = Vec::new();
    if limit == 0 {
        return Ok(out);
    }
    git_walk(lib, range, |commit| {
        let parents: Vec<Commit<'_>> = commit.parents().collect();
        if range.paths.is_empty() {
            out.push(info(commit)?);
            return Ok((out.len() < limit).then(|| parents.iter().map(Commit::id).collect()));
        }
        // Simplified as it is taken: a merge the same as one parent at the
        // paths follows only that parent, and is not shown.
        let here = at_paths(commit, &range.paths)?;
        let mut same_as = None;
        for (i, parent) in parents.iter().enumerate() {
            if at_paths(parent, &range.paths)? == here {
                same_as = Some(i);
                break;
            }
        }
        let shown = match (parents.len(), same_as) {
            (0, _) => here.iter().any(Option::is_some),
            (_, Some(_)) => false,
            (_, None) => true,
        };
        if shown {
            out.push(info(commit)?);
        }
        let follow = match same_as {
            Some(i) if parents.len() > 1 => vec![parents[i].id()],
            _ => parents.iter().map(Commit::id).collect(),
        };
        Ok((out.len() < limit).then_some(follow))
    })?;
    Ok(out)
}

/// Up to `limit` commits that changed `path`, newest first, following it back
/// through renames.
pub(crate) fn file_history(
    lib: &Repository,
    rev: &Rev,
    path: &RepoPath,
    limit: usize,
) -> Result<Vec<CommitInfo>, GitError> {
    let mut following = path.as_str().to_string();
    let mut out = Vec::new();
    if limit == 0 {
        return Ok(out);
    }
    git_walk(lib, &LogRange::of(rev.clone()), |commit| {
        let parents: Vec<LibOid> = commit.parent_ids().collect();
        let tree = commit.tree().map_err(|e| failed("tree", e))?;
        let here = tree.get_path(Path::new(&following)).ok().map(|e| e.id());
        let parent = commit.parents().next();
        let before = match &parent {
            Some(p) => p
                .tree()
                .map_err(|e| failed("tree", e))?
                .get_path(Path::new(&following))
                .ok()
                .map(|e| e.id()),
            None => None,
        };
        if here == before {
            return Ok(Some(parents));
        }
        out.push(info(commit)?);
        // Added here: renamed from another path, the one to follow from now.
        if let (Some(_), None, Some(parent)) = (here, before, &parent) {
            let old_tree = parent.tree().map_err(|e| failed("tree", e))?;
            let mut options = DiffOptions::new();
            let mut diff = lib
                .diff_tree_to_tree(Some(&old_tree), Some(&tree), Some(&mut options))
                .map_err(|e| failed("diff", e))?;
            diff.find_similar(Some(DiffFindOptions::new().renames(true)))
                .map_err(|e| failed("find_similar", e))?;
            let renamed = diff.deltas().find_map(|d| {
                let to = d.new_file().path()?.to_str()?;
                (d.status() == Delta::Renamed && to == following)
                    .then(|| d.old_file().path()?.to_str().map(str::to_string))
                    .flatten()
            });
            if let Some(from) = renamed {
                following = from;
            }
        }
        Ok((out.len() < limit).then_some(parents))
    })?;
    Ok(out)
}
