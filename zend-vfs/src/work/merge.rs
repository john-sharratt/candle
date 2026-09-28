//! Merging a commit into a conversation's work.

use std::collections::HashMap;

use super::three_way::ThreeWay;
use super::{files, held};
use crate::vfs::{Base, Carried, Side};
use crate::write::merge_text::MergeLabels;
use crate::{BlobReader, GitError, Oid, Repo, RepoPath, Rev, VfsStore};

/// What a merge did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Merged {
    /// The conversation already has every commit it would bring.
    UpToDate,
    /// The conversation's base had nothing the incoming commit lacks: the
    /// base is that commit now, and the conversation's changes are carried
    /// onto it.
    FastForward { conflicts: Vec<String> },
    /// Both had commits of their own: the merge is being finished — the
    /// next commit of the conversation's has both as parents.
    Merging { conflicts: Vec<String> },
}

impl Merged {
    /// Every path left in conflict for the conversation to settle.
    pub fn conflicts(&self) -> &[String] {
        match self {
            Merged::UpToDate => &[],
            Merged::FastForward { conflicts } | Merged::Merging { conflicts } => conflicts,
        }
    }
}

/// Merge `theirs` into the conversation `store` holds over `repo`, the way
/// `git merge` merges into a working tree — but into the conversation's own
/// copy, never a folder:
///
/// - `theirs` already in the conversation's history: nothing to do;
/// - the conversation's commit in `theirs`' history: the base moves to
///   `theirs`, and each path the conversation changed that `theirs` changed
///   too is merged three ways ([`ThreeWay`]);
/// - otherwise the two commits' trees are merged over their merge base as far
///   as they settle, the base becomes that tree with both commits as parents,
///   and each path the commits both changed — and each the conversation
///   changed that the merge moved — is merged three ways into the
///   conversation's copy.
///
/// While a merge is being finished, whatever comes in is merged into the
/// tree that merge settled, the same way: a commit building on some of the
/// merge's parents — origin moved on meanwhile, on either side — takes their
/// place, and one building on none of them joins the merge as a parent of
/// its own. One already in any parent's history is up to date; one holding
/// every parent is the merge made already, and the base fast-forwards to it.
///
/// Overlaps are left between conflict markers labelled `yours` and
/// `theirs_label`, and flagged until the conversation settles them. A merge
/// meeting a file that is not text and changed on both sides is refused
/// whole: nothing changes.
pub fn merge_into(
    repo: &Repo,
    store: &VfsStore,
    theirs: &Oid,
    theirs_label: &str,
) -> Result<Merged, GitError> {
    let base = store
        .base()
        .map_err(files)?
        .ok_or_else(|| GitError::invalid("these files are not read from a branch"))?;
    let blobs = repo.blobs();
    let (theirs, their_tree) = blobs
        .commit_of(&Rev::Oid(theirs.clone()))?
        .ok_or_else(|| GitError::invalid(format!("no commit {theirs} to merge")))?;
    let labels = MergeLabels {
        ours: "yours",
        base: "base",
        theirs: theirs_label,
    };
    let three = ThreeWay::new(repo, labels);

    let Some(ours) = base.commit().cloned() else {
        // Nothing committed yet: what comes in is the whole of it.
        let conflicts = store
            .move_base(
                None,
                Base::at(theirs, their_tree),
                &[],
                &mut |c: &Carried<'_>| three.carried(c),
            )
            .map_err(files)?;
        return Ok(Merged::FastForward { conflicts });
    };
    let t = Rev::Oid(theirs.clone());
    // What the trees merge from, what they merge, and the parents the merge
    // will have: the conversation's commit and its merge base with `theirs`
    // — or, while a merge is being finished, the tree that merge settled,
    // merged on from the parent `theirs` builds on, which `theirs` then
    // takes the place of; one it builds on none of joins the merge as a
    // parent of its own.
    let (merge_base, ours_tree, parents) = match base.merging() {
        Some(_) => {
            for parent in &base.parents {
                if parent == &theirs || repo.is_ancestor(&t, &Rev::Oid(parent.clone()))? {
                    return Ok(Merged::UpToDate);
                }
            }
            // The merge's parents `theirs` already has: it takes their place,
            // so no parent of the commit is in another's history.
            let mut built_on = Vec::new();
            for (at, parent) in base.parents.iter().enumerate() {
                if repo.is_ancestor(&Rev::Oid(parent.clone()), &t)? {
                    built_on.push(at);
                }
            }
            let tree = base
                .tree
                .clone()
                .ok_or_else(|| GitError::invalid("the merge being finished here holds no tree"))?;
            if built_on.len() == base.parents.len() {
                // `theirs` holds every side of the merge: it is the merge
                // made already. The base moves to it, and what the
                // conversation settled is carried onto it three ways from
                // the settled tree.
                let conflicts = store
                    .move_base(
                        None,
                        Base::at(theirs, their_tree),
                        &[],
                        &mut |c: &Carried<'_>| three.carried(c),
                    )
                    .map_err(files)?;
                held::release(repo, &base);
                return Ok(Merged::FastForward { conflicts });
            }
            let mut parents = Vec::with_capacity(base.parents.len() + 1);
            for (at, parent) in base.parents.iter().enumerate() {
                if built_on.first() == Some(&at) {
                    parents.push(theirs.clone());
                } else if !built_on.contains(&at) {
                    parents.push(parent.clone());
                }
            }
            let merge_base = match built_on.first() {
                Some(&at) => base.parents[at].clone(),
                None => {
                    parents.push(theirs.clone());
                    repo.merge_base(&Rev::Oid(ours.clone()), &t)?
                        .ok_or_else(|| {
                            GitError::invalid(format!("{ours} and {theirs} share no history"))
                        })?
                }
            };
            (merge_base, tree, parents)
        }
        None => {
            let o = Rev::Oid(ours.clone());
            if ours == theirs || repo.is_ancestor(&t, &o)? {
                return Ok(Merged::UpToDate);
            }
            if repo.is_ancestor(&o, &t)? {
                let conflicts = store
                    .move_base(
                        None,
                        Base::at(theirs, their_tree),
                        &[],
                        &mut |c: &Carried<'_>| three.carried(c),
                    )
                    .map_err(files)?;
                return Ok(Merged::FastForward { conflicts });
            }
            let merge_base = repo.merge_base(&o, &t)?.ok_or_else(|| {
                GitError::invalid(format!("{ours} and {theirs} share no history"))
            })?;
            (merge_base, ours.clone(), vec![ours.clone(), theirs.clone()])
        }
    };
    let partial = repo.merge_trees_keeping_ours(&merge_base, &ours_tree, &theirs)?;
    // The paths both commits changed: merged from their merge base, the
    // conversation's copy standing for ours. Each is read before anything
    // changes; one that is not text on any side refuses the merge.
    let mut sides: HashMap<String, (Side, Side)> = HashMap::new();
    let mut not_text = Vec::new();
    for path in &partial.conflicts {
        let read = |rev: &Oid| side(&blobs, rev, path);
        let (was, mine, now) = (read(&merge_base)?, read(&ours_tree)?, read(&theirs)?);
        if [&was, &mine, &now].contains(&&Side::Binary) {
            not_text.push(path.as_str().to_string());
            continue;
        }
        sides.insert(path.as_str().to_string(), (was, now));
    }
    if !not_text.is_empty() {
        return Err(GitError::invalid(format!(
            "these files are not text and changed on both sides, so they cannot be merged \
             here: {}",
            not_text.join(", ")
        )));
    }
    let extra: Vec<String> = sides.keys().cloned().collect();
    let to = Base {
        tree: Some(partial.tree),
        parents,
    };
    // The settled tree is held by a ref of its own until the merge is
    // committed or dropped: nothing else in the repository reaches it, and
    // git would prune it in time. Holds are counted, so this one is let go
    // exactly once — here if the move fails, or when the base next moves
    // off it — and the one on the tree this replaces is let go once the
    // move stands.
    held::hold(repo, &to)?;
    let settled = to.clone();
    let conflicts = store
        .move_base(
            None,
            to,
            &extra,
            &mut |c: &Carried<'_>| match sides.get(c.path) {
                Some((was, now)) => three.merge(c.path, was, c.ours, now),
                None => three.carried(c),
            },
        )
        .map_err(|e| {
            held::release(repo, &settled);
            files(e)
        })?;
    held::release(repo, &base);
    Ok(Merged::Merging { conflicts })
}

/// `path` at the commit `rev` as a [`Side`].
fn side(blobs: &BlobReader, rev: &Oid, path: &RepoPath) -> Result<Side, GitError> {
    Ok(match blobs.read_at(&Rev::Oid(rev.clone()), path) {
        Ok(Some(bytes)) => match String::from_utf8(bytes) {
            Ok(text) => Side::Text(text),
            Err(_) => Side::Binary,
        },
        Ok(None) => Side::Absent,
        Err(GitError::BlobTooLarge { .. }) | Err(GitError::NotABlob { .. }) => Side::Binary,
        Err(e) => return Err(e),
    })
}
