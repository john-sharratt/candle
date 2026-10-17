//! Git bundles, cut and read in process (`docs/substrate_git.md` §4.4).
//!
//! A bundle is a header — `# v2 git bundle`, or `# v3 git bundle` and its
//! capability lines; a `-<id>` line per prerequisite commit the receiver must
//! hold; a `<id> <ref>` line per tip — a blank line, and a pack. One cut here
//! carries every object the tips reach that the hidden commits do not, and
//! names as prerequisites exactly the commits outside it that its commits are
//! children of, as `git bundle create` does.

use std::collections::HashSet;
use std::str;

use git2::{Buf, Oid as LibOid, Repository, Revwalk};

use super::db::ObjectDb;
use super::pack::RawId;
use crate::error::GitError;
use crate::remote::packs::object_count;

const V2: &str = "# v2 git bundle";
const V3: &str = "# v3 git bundle";

fn lib_error(what: &str, e: git2::Error) -> GitError {
    GitError::Packs(format!("libgit2 {what}: {}", e.message()))
}

fn lib_oid(id: &RawId) -> Result<LibOid, GitError> {
    LibOid::from_bytes(&id.0).map_err(|e| lib_error("oid", e))
}

/// A bundle of what `tips` — each a commit and the ref it is named by —
/// reach beyond `hidden`; `None` when that is nothing.
pub(crate) fn cut(
    repo: &Repository,
    tips: &[(RawId, String)],
    hidden: &[RawId],
) -> Result<Option<Vec<u8>>, GitError> {
    let walk = || -> Result<Revwalk<'_>, GitError> {
        let mut walk = repo.revwalk().map_err(|e| lib_error("revwalk", e))?;
        for (tip, _) in tips {
            walk.push(lib_oid(tip)?)
                .map_err(|e| lib_error("revwalk_push", e))?;
        }
        for id in hidden {
            // A hidden commit this repository lacks hides nothing it holds.
            if repo.find_commit(lib_oid(id)?).is_ok() {
                walk.hide(lib_oid(id)?)
                    .map_err(|e| lib_error("revwalk_hide", e))?;
            }
        }
        Ok(walk)
    };
    let included: HashSet<LibOid> = walk()?
        .collect::<Result<_, _>>()
        .map_err(|e| lib_error("revwalk", e))?;
    if included.is_empty() {
        return Ok(None);
    }
    let mut prerequisites: Vec<LibOid> = Vec::new();
    for oid in &included {
        let commit = repo
            .find_commit(*oid)
            .map_err(|e| lib_error("find_commit", e))?;
        for parent in commit.parent_ids() {
            if !included.contains(&parent) && !prerequisites.contains(&parent) {
                prerequisites.push(parent);
            }
        }
    }
    prerequisites.sort();

    let mut builder = repo
        .packbuilder()
        .map_err(|e| lib_error("packbuilder", e))?;
    builder
        .insert_walk(&mut walk()?)
        .map_err(|e| lib_error("packbuilder_insert_walk", e))?;
    let mut pack = Buf::new();
    builder
        .write_buf(&mut pack)
        .map_err(|e| lib_error("packbuilder_write_buf", e))?;

    let mut out = format!("{V2}\n").into_bytes();
    for prerequisite in prerequisites {
        out.extend_from_slice(format!("-{prerequisite}\n").as_bytes());
    }
    for (tip, name) in tips {
        out.extend_from_slice(format!("{tip} {name}\n").as_bytes());
    }
    out.push(b'\n');
    out.extend_from_slice(&pack);
    Ok(Some(out))
}

/// A bundle, read.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Parsed<'a> {
    pub(crate) prerequisites: Vec<RawId>,
    pub(crate) refs: Vec<(RawId, String)>,
    pub(crate) pack: &'a [u8],
}

fn bad(detail: &str) -> GitError {
    GitError::malformed("bundle", detail.to_string())
}

fn id_at(text: &str) -> Result<RawId, GitError> {
    let hex = text.get(..40).ok_or_else(|| bad("an id cut short"))?;
    let mut raw = [0u8; 20];
    for (i, byte) in raw.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| bad("an id that is not hex"))?;
    }
    Ok(RawId(raw))
}

/// Read `bundle`'s header and find its pack.
pub(crate) fn parse(bundle: &[u8]) -> Result<Parsed<'_>, GitError> {
    let mut at = 0;
    let mut line = |bundle: &[u8]| -> Result<String, GitError> {
        let end = bundle[at..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| bad("a header with no end"))?;
        let text = str::from_utf8(&bundle[at..at + end])
            .map_err(|_| bad("a header that is not UTF-8"))?
            .to_string();
        at += end + 1;
        Ok(text)
    };
    let signature = line(bundle)?;
    if signature != V2 && signature != V3 {
        return Err(bad("not a version 2 or 3 bundle"));
    }
    let mut parsed = Parsed {
        prerequisites: Vec::new(),
        refs: Vec::new(),
        pack: &[],
    };
    loop {
        let text = line(bundle)?;
        if text.is_empty() {
            break;
        }
        if let Some(capability) = text.strip_prefix('@') {
            if capability.starts_with("object-format=") && capability != "object-format=sha1" {
                return Err(bad("a bundle of another object format than SHA-1"));
            }
            continue;
        }
        match text.strip_prefix('-') {
            Some(rest) => parsed.prerequisites.push(id_at(rest)?),
            None => {
                let id = id_at(&text)?;
                let name = text
                    .get(41..)
                    .ok_or_else(|| bad("a ref line with no name"))?;
                parsed.refs.push((id, name.to_string()));
            }
        }
    }
    parsed.pack = &bundle[at..];
    Ok(parsed)
}

impl ObjectDb {
    /// Take `bundle`'s objects: its prerequisites must be held, and its pack
    /// goes to the store as any pack taken does (§4.2).
    pub(crate) fn take_bundle(&self, bundle: &[u8]) -> Result<(), GitError> {
        let parsed = parse(bundle)?;
        for prerequisite in &parsed.prerequisites {
            if !self.contains(prerequisite) {
                return Err(GitError::Packs(format!(
                    "the bundle needs commit {prerequisite}, which is not held"
                )));
            }
        }
        if object_count(parsed.pack) > 0 {
            self.take_pack(parsed.pack.to_vec())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
