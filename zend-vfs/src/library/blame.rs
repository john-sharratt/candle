//! Who last changed each line of a file, from libgit2's blame — a port of
//! git's own, following the file back through renames — each line given its
//! commit's author, committer and summary as `--line-porcelain` gives them.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::Path;

use git2::{BlameOptions, Oid as LibOid, Repository};

use super::{failed, oid_of, revision};
use crate::error::GitError;
use crate::library::log::info;
use crate::read::blame::{BlameLine, LineRange};
use crate::read::log::CommitInfo;
use crate::types::{RepoPath, Rev};

/// Blame `path` at `rev`, within `lines` when given. The caller has checked
/// `rev` holds a file there.
pub(crate) fn blame(
    lib: &Repository,
    rev: &Rev,
    path: &RepoPath,
    lines: Option<LineRange>,
) -> Result<Vec<BlameLine>, GitError> {
    let spec = rev.spec();
    let commit = lib
        .revparse_single(&spec)
        .and_then(|object| object.peel_to_commit())
        .map_err(|e| revision(e, &spec))?;
    let mut options = BlameOptions::new();
    options.newest_commit(commit.id());
    if let Some(range) = lines {
        options
            .min_line(range.start as usize)
            .max_line(range.end as usize);
    }
    let blamed = lib
        .blame_file(Path::new(path.as_str()), Some(&mut options))
        .map_err(|e| failed("blame_file", e))?;
    let content = commit
        .tree()
        .and_then(|tree| tree.get_path(Path::new(path.as_str())))
        .and_then(|entry| lib.find_blob(entry.id()))
        .map_err(|e| failed("find_blob", e))?;
    let file: Vec<&[u8]> = content.content().split(|&b| b == b'\n').collect();

    let mut commits: HashMap<LibOid, CommitInfo> = HashMap::new();
    let mut out = Vec::new();
    for hunk in blamed.iter() {
        let id = hunk.orig_commit_id();
        let info = match commits.entry(id) {
            Entry::Occupied(held) => held.into_mut(),
            Entry::Vacant(slot) => {
                let found = lib.find_commit(id).map_err(|e| failed("find_commit", e))?;
                slot.insert(info(&found)?)
            }
        };
        let orig_path = match hunk.path() {
            Some(p) => RepoPath::parse(&p.to_string_lossy().replace('\\', "/"))?,
            None => path.clone(),
        };
        for k in 0..hunk.lines_in_hunk() {
            let final_line = hunk.final_start_line() + k;
            out.push(BlameLine {
                commit: oid_of(id)?,
                orig_line: (hunk.orig_start_line() + k) as u32,
                final_line: final_line as u32,
                orig_path: orig_path.clone(),
                author: info.author.clone(),
                committer: info.committer.clone(),
                summary: info.subject().to_string(),
                content: file
                    .get(final_line - 1)
                    .map(|l| l.to_vec())
                    .unwrap_or_default(),
            });
        }
    }
    Ok(out)
}
