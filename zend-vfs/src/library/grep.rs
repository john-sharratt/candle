//! Searching a revision's files in this process, as `git grep -I` does: its
//! tree walked in order, binary files skipped by git's own test, and each line
//! matched as an extended regular expression or a fixed string.

use git2::{ObjectType, Oid as LibOid, Repository, TreeWalkMode, TreeWalkResult};
use regex::bytes::{Regex, RegexBuilder};

use super::{failed, revision};
use crate::error::GitError;
use crate::read::grep::{GrepHit, GrepQuery};
use crate::types::{RepoPath, Rev};

/// How much of a file git reads to judge it binary: a NUL in it is binary.
const BINARY_PROBE: usize = 8000;

fn pattern(query: &GrepQuery) -> Result<Regex, GitError> {
    let source = if query.fixed {
        regex::escape(&query.pattern)
    } else {
        query.pattern.clone()
    };
    RegexBuilder::new(&source)
        .case_insensitive(query.ignore_case)
        .unicode(false)
        .build()
        .map_err(|e| GitError::invalid(format!("the pattern {:?}: {e}", query.pattern)))
}

/// Whether `path` lies under one of `paths` — itself, or a folder holding it —
/// or `paths` is empty.
fn wanted(path: &str, paths: &[RepoPath]) -> bool {
    paths.is_empty()
        || paths.iter().any(|p| {
            let p = p.as_str();
            path == p
                || path
                    .strip_prefix(p)
                    .is_some_and(|rest| rest.starts_with('/'))
        })
}

/// Every line `query` matches in the files of `rev`, in tree order, at most
/// `query.max_per_file` from each.
pub(crate) fn grep(
    lib: &Repository,
    rev: &Rev,
    query: &GrepQuery,
) -> Result<Vec<GrepHit>, GitError> {
    let regex = pattern(query)?;
    let spec = rev.spec();
    let tree = lib
        .revparse_single(&spec)
        .and_then(|object| object.peel_to_tree())
        .map_err(|e| revision(e, &spec))?;
    let mut files: Vec<(String, LibOid)> = Vec::new();
    tree.walk(TreeWalkMode::PreOrder, |folder, entry| {
        if entry.kind() == Some(ObjectType::Blob) && entry.filemode() != 0o120000 {
            if let Some(name) = entry.name() {
                files.push((format!("{folder}{name}"), entry.id()));
            }
        }
        TreeWalkResult::Ok
    })
    .map_err(|e| failed("tree_walk", e))?;

    let mut hits = Vec::new();
    for (path, id) in files {
        if !wanted(&path, &query.paths) {
            continue;
        }
        let blob = lib.find_blob(id).map_err(|e| failed("find_blob", e))?;
        let content = blob.content();
        if content[..content.len().min(BINARY_PROBE)].contains(&0) {
            continue;
        }
        let mut lines: Vec<&[u8]> = content.split(|&b| b == b'\n').collect();
        // The empty piece after a final newline is no line.
        if content.ends_with(b"\n") {
            lines.pop();
        }
        let mut found = 0u32;
        for (n, line) in lines.into_iter().enumerate() {
            if !regex.is_match(line) {
                continue;
            }
            hits.push(GrepHit {
                path: RepoPath::parse(&path)?,
                line: n as u32 + 1,
                text: line.to_vec(),
            });
            found += 1;
            if query.max_per_file.is_some_and(|max| found >= max) {
                break;
            }
        }
    }
    Ok(hits)
}
