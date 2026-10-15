//! Differences between commits from libgit2: what changed, the changes as
//! hunks, and the patch as bytes `git apply` takes.
//!
//! Renames are detected as `-M` detects them — 50% similar — and a file that
//! changed type is one entry, as git's raw format reports it.

use std::mem;
use std::path::Path;

use git2::{
    Delta, Diff, DiffDelta, DiffFile, DiffFindOptions, DiffFormat, DiffLine, DiffOptions,
    Oid as LibOid, Patch, Repository,
};
use libgit2_sys as raw;

use super::{failed, oid_of, revision};
use crate::error::GitError;
use crate::read::diff::{DiffEntry, DiffSide, DiffStatus};
use crate::read::patch::{FilePatch, Hunk, LineKind, PatchLine};
use crate::types::{FileMode, RepoPath, Rev};

/// The tree diff from `from` to `to`, limited to `paths`, renames found.
fn tree_diff<'r>(
    lib: &'r Repository,
    from: &Rev,
    to: &Rev,
    paths: &[&RepoPath],
    options: &mut DiffOptions,
) -> Result<Diff<'r>, GitError> {
    let tree = |rev: &Rev| {
        let spec = rev.spec();
        lib.revparse_single(&spec)
            .and_then(|object| object.peel_to_tree())
            .map_err(|e| revision(e, &spec))
    };
    let (old, new) = (tree(from)?, tree(to)?);
    options
        .include_typechange(true)
        .disable_pathspec_match(true);
    for path in paths {
        options.pathspec(path.as_str());
    }
    let mut diff = lib
        .diff_tree_to_tree(Some(&old), Some(&new), Some(options))
        .map_err(|e| failed("diff_tree_to_tree", e))?;
    diff.find_similar(Some(DiffFindOptions::new().renames(true)))
        .map_err(|e| failed("find_similar", e))?;
    Ok(diff)
}

fn side_path(path: Option<&Path>) -> Result<RepoPath, GitError> {
    let path = path
        .and_then(Path::to_str)
        .ok_or_else(|| GitError::malformed("libgit2", "a diff path that is not UTF-8"))?;
    RepoPath::parse(&path.replace('\\', "/"))
}

const _: () = assert!(
    mem::size_of::<DiffDelta<'static>>() == mem::size_of::<*mut raw::git_diff_delta>(),
    "git2::DiffDelta is no longer its raw pointer alone"
);

/// The similarity libgit2 scored a rename or copy at, 0–100, which `git2`
/// does not expose: read from the delta it wraps, whose pointer is all a
/// `DiffDelta` holds (the assertion above breaks the build if that changes).
fn similarity(delta: &DiffDelta<'_>) -> u16 {
    // SAFETY: a `DiffDelta` is its `*mut git_diff_delta`, alive as long as
    // the diff it came from, which `delta` borrows.
    unsafe {
        let ptr = *(delta as *const DiffDelta<'_> as *const *const raw::git_diff_delta);
        (*ptr).similarity
    }
}

/// One delta as the layer's entry.
fn entry_of(delta: &DiffDelta<'_>) -> Result<DiffEntry, GitError> {
    let side = |file: DiffFile<'_>| -> Result<Option<DiffSide>, GitError> {
        let mode = u32::from(file.mode());
        if mode == 0 {
            return Ok(None);
        }
        Ok(Some(DiffSide {
            mode: FileMode::parse(&format!("{mode:06o}"))?,
            oid: if file.id().is_zero() {
                None
            } else {
                Some(oid_of(file.id())?)
            },
            path: side_path(file.path())?,
        }))
    };
    let score = || similarity(delta).min(100) as u8;
    let status = match delta.status() {
        Delta::Added => DiffStatus::Added,
        Delta::Deleted => DiffStatus::Deleted,
        Delta::Modified => DiffStatus::Modified,
        Delta::Typechange => DiffStatus::TypeChanged,
        Delta::Conflicted => DiffStatus::Unmerged,
        Delta::Renamed => DiffStatus::Renamed(score()),
        Delta::Copied => DiffStatus::Copied(score()),
        other => {
            return Err(GitError::malformed(
                "libgit2",
                format!("a delta of status {other:?}"),
            ))
        }
    };
    Ok(DiffEntry {
        status,
        old: side(delta.old_file())?,
        new: side(delta.new_file())?,
    })
}

/// What changed from `from` to `to`, with renames, limited to `paths`.
pub(crate) fn diff(
    lib: &Repository,
    from: &Rev,
    to: &Rev,
    paths: &[&RepoPath],
) -> Result<Vec<DiffEntry>, GitError> {
    let diff = tree_diff(lib, from, to, paths, &mut DiffOptions::new())?;
    diff.deltas().map(|d| entry_of(&d)).collect()
}

/// The patches from `from` to `to`, `context` lines around each change.
pub(crate) fn patches(
    lib: &Repository,
    from: &Rev,
    to: &Rev,
    context: u32,
    paths: &[&RepoPath],
) -> Result<Vec<FilePatch>, GitError> {
    let mut options = DiffOptions::new();
    options.context_lines(context);
    let diff = tree_diff(lib, from, to, paths, &mut options)?;
    let mut out = Vec::new();
    for i in 0..diff.deltas().len() {
        let delta = diff.get_delta(i).expect("in range");
        let entry = entry_of(&delta)?;
        // A type change is a deletion and a creation, as git writes it: the
        // old side's lines all removed, then the new side's all added.
        let typechange = delta.status() == Delta::Typechange;
        let content = |id: LibOid| -> Result<Vec<u8>, GitError> {
            Ok(lib
                .find_blob(id)
                .map_err(|e| failed("find_blob", e))?
                .content()
                .to_vec())
        };
        let (old_bytes, new_bytes) = match typechange {
            true => (
                content(delta.old_file().id())?,
                content(delta.new_file().id())?,
            ),
            false => (Vec::new(), Vec::new()),
        };
        let parts: Vec<Option<Patch<'_>>> = if typechange {
            let mut each = DiffOptions::new();
            each.context_lines(context);
            let removed = Patch::from_buffers(&old_bytes, None, &[], None, Some(&mut each))
                .map_err(|e| failed("patch_from_buffers", e))?;
            let added = Patch::from_buffers(&[], None, &new_bytes, None, Some(&mut each))
                .map_err(|e| failed("patch_from_buffers", e))?;
            vec![Some(removed), Some(added)]
        } else {
            vec![Patch::from_diff(&diff, i).map_err(|e| failed("patch_from_diff", e))?]
        };
        let binary = delta.flags().is_binary() || parts.iter().any(Option::is_none);
        let mut hunks = Vec::new();
        for patch in parts.into_iter().flatten().filter(|_| !binary) {
            for h in 0..patch.num_hunks() {
                let (hunk, count) = patch.hunk(h).map_err(|e| failed("patch_hunk", e))?;
                let header = String::from_utf8_lossy(hunk.header()).into_owned();
                let context = header
                    .trim_end_matches('\n')
                    .splitn(3, "@@")
                    .nth(2)
                    .map(|c| c.strip_prefix(' ').unwrap_or(c).to_string())
                    .unwrap_or_default();
                let mut lines: Vec<PatchLine> = Vec::new();
                for l in 0..count {
                    let line = patch
                        .line_in_hunk(h, l)
                        .map_err(|e| failed("patch_line", e))?;
                    push_line(&mut lines, &line);
                }
                hunks.push(Hunk {
                    old_start: hunk.old_start(),
                    old_lines: hunk.old_lines(),
                    new_start: hunk.new_start(),
                    new_lines: hunk.new_lines(),
                    context,
                    lines,
                });
            }
        }
        out.push(FilePatch {
            entry,
            binary,
            hunks,
        });
    }
    Ok(out)
}

/// `line` added to `lines`: a content line, or the mark that the line before
/// ends the file with no newline.
fn push_line(lines: &mut Vec<PatchLine>, line: &DiffLine<'_>) {
    let kind = match line.origin() {
        ' ' => LineKind::Context,
        '+' => LineKind::Added,
        '-' => LineKind::Removed,
        '=' | '>' | '<' => {
            if let Some(last) = lines.last_mut() {
                last.no_newline_at_end = true;
            }
            return;
        }
        _ => return,
    };
    let text = line.content();
    let text = text.strip_suffix(b"\n").unwrap_or(text);
    lines.push(PatchLine {
        kind,
        text: text.to_vec(),
        no_newline_at_end: false,
    });
}

/// The patch from `from` to `to` as bytes `git apply` takes: full ids, and
/// binary content as git's binary patches.
pub(crate) fn patch_bytes(lib: &Repository, from: &Rev, to: &Rev) -> Result<Vec<u8>, GitError> {
    let mut options = DiffOptions::new();
    options.show_binary(true).id_abbrev(40);
    let diff = tree_diff(lib, from, to, &[], &mut options)?;
    let mut out = Vec::new();
    diff.print(DiffFormat::Patch, |_, _, line| {
        if matches!(line.origin(), '+' | '-' | ' ') {
            out.push(line.origin() as u8);
        }
        out.extend_from_slice(line.content());
        true
    })
    .map_err(|e| failed("diff_print", e))?;
    Ok(out)
}
