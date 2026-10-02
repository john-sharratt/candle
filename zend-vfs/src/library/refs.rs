//! Refs, `HEAD`, branches, tags and remote-tracking branches, from libgit2.
//!
//! Every list is ordered by full ref name, bytewise, as `for-each-ref` orders
//! it, so a caller cannot tell which answered.

use std::collections::BTreeMap;

use git2::{Reference, ReferenceType, Repository};

use super::{failed, is_absent, oid_of, revision};
use crate::error::GitError;
use crate::read::head::Head;
use crate::read::refs::{Branch, Upstream};
use crate::read::remote_branches::RemoteBranch;
use crate::read::tags::Tag;
use crate::types::{BranchName, Oid, RefName, RemoteName, Rev, TagName};

/// The commit `rev` names.
pub(crate) fn resolve(lib: &Repository, rev: &Rev) -> Result<Oid, GitError> {
    let spec = rev.spec();
    let object = lib
        .revparse_single(&format!("{spec}^{{commit}}"))
        .map_err(|e| revision(e, &spec))?;
    oid_of(object.id())
}

/// The object `rev` names, unpeeled: an annotated tag's own id.
pub(crate) fn resolve_object(lib: &Repository, rev: &Rev) -> Result<Oid, GitError> {
    let spec = rev.spec();
    let object = lib.revparse_single(&spec).map_err(|e| revision(e, &spec))?;
    oid_of(object.id())
}

/// What the full ref `name` points at, a symbolic ref followed; `None` when
/// there is no such ref.
pub(crate) fn ref_target(lib: &Repository, name: &RefName) -> Result<Option<Oid>, GitError> {
    match lib.refname_to_id(name.as_str()) {
        Ok(id) => Ok(Some(oid_of(id)?)),
        Err(e) if is_absent(&e) => Ok(None),
        Err(e) => Err(failed("refname_to_id", e)),
    }
}

/// Every ref whose full name begins with `prefix`, by name. A symbolic ref
/// whose target is gone — a remote's `HEAD` after its branch was pruned — is
/// left out, as `for-each-ref` ignores a broken ref.
fn refs_with_prefix<'r>(lib: &'r Repository, prefix: &str) -> Result<Vec<Reference<'r>>, GitError> {
    let mut found = Vec::new();
    for reference in lib.references().map_err(|e| failed("references", e))? {
        let reference = reference.map_err(|e| failed("references", e))?;
        if reference.name_bytes().starts_with(prefix.as_bytes()) && reference.resolve().is_ok() {
            found.push(reference);
        }
    }
    found.sort_by(|a, b| a.name_bytes().cmp(b.name_bytes()));
    Ok(found)
}

/// The id a ref finally points at, a symbolic ref followed.
fn target_of(reference: &Reference<'_>) -> Result<Oid, GitError> {
    let direct = reference.resolve().map_err(|e| failed("resolve", e))?;
    let id = direct
        .target()
        .ok_or_else(|| GitError::malformed("libgit2", "a ref with no target"))?;
    oid_of(id)
}

fn name_of(reference: &Reference<'_>) -> Result<String, GitError> {
    reference
        .name()
        .map(str::to_string)
        .ok_or_else(|| GitError::malformed("libgit2", "a ref name that is not UTF-8"))
}

/// Every ref under `folder` — `refs/...`, ending in `/` — with what it points
/// at.
pub(crate) fn refs_under(lib: &Repository, folder: &str) -> Result<Vec<(RefName, Oid)>, GitError> {
    refs_with_prefix(lib, folder)?
        .iter()
        .map(|r| Ok((RefName::parse(&name_of(r)?)?, target_of(r)?)))
        .collect()
}

/// The direct refs under `scope` — a ref name or a namespace ending in `/` —
/// as `for-each-ref <scope>` matches them: a name matches itself and what lies
/// beneath it. Symbolic refs are skipped.
pub(crate) fn snapshot(lib: &Repository, scope: &str) -> Result<BTreeMap<RefName, Oid>, GitError> {
    let beneath = if scope.ends_with('/') {
        scope.to_string()
    } else {
        format!("{scope}/")
    };
    let mut refs = BTreeMap::new();
    for reference in lib.references().map_err(|e| failed("references", e))? {
        let reference = reference.map_err(|e| failed("references", e))?;
        if reference.kind() == Some(ReferenceType::Symbolic) {
            continue;
        }
        let name = name_of(&reference)?;
        if name == scope || name.starts_with(&beneath) {
            refs.insert(RefName::parse(&name)?, target_of(&reference)?);
        }
    }
    Ok(refs)
}

/// What `HEAD` is.
pub(crate) fn head(lib: &Repository) -> Result<Head, GitError> {
    let head = lib
        .find_reference("HEAD")
        .map_err(|e| failed("find_reference", e))?;
    let Some(target) = head.symbolic_target() else {
        let id = head
            .target()
            .ok_or_else(|| GitError::malformed("libgit2", "HEAD with no target"))?;
        return Ok(Head::Detached(oid_of(id)?));
    };
    let name = RefName::parse(target)?;
    let branch = name
        .branch()
        .ok_or_else(|| GitError::malformed("rev-parse", format!("HEAD points at {name}")))?;
    match ref_target(lib, &name)? {
        Some(oid) => Ok(Head::Branch { branch, oid }),
        None => Ok(Head::Unborn(branch)),
    }
}

/// Every local branch, with its upstream when one is configured.
pub(crate) fn branches(lib: &Repository) -> Result<Vec<Branch>, GitError> {
    let config = lib.config().map_err(|e| failed("config", e))?;
    let mut out = Vec::new();
    for reference in refs_with_prefix(lib, "refs/heads/")? {
        let full = name_of(&reference)?;
        let name = RefName::parse(&full)?
            .branch()
            .ok_or_else(|| GitError::malformed("libgit2", full.clone()))?;
        let oid = target_of(&reference)?;
        let setting = |key: &str| {
            config
                .get_string(&format!("branch.{}.{key}", name.as_str()))
                .ok()
                .filter(|v| !v.is_empty())
        };
        let upstream = match (setting("remote"), setting("merge")) {
            (Some(remote), Some(merge)) => Some(upstream_of(lib, &oid, &remote, &merge)?),
            _ => None,
        };
        out.push(Branch {
            name,
            oid,
            upstream,
        });
    }
    Ok(out)
}

/// The branch a local branch tracks, and how far apart they stand.
///
/// `remote` is `.` for another local branch, else a remote's name; `merge` is
/// the full ref the branch tracks on it. The local ref compared against is the
/// merge ref itself for `.`, else what the remote's fetch refspecs map it to —
/// none mapping it, nothing is compared, which `for-each-ref` reports as no
/// tracking line at all.
fn upstream_of(
    lib: &Repository,
    local: &Oid,
    remote: &str,
    merge: &str,
) -> Result<Upstream, GitError> {
    let branch = RefName::parse(merge)?
        .branch()
        .ok_or_else(|| GitError::malformed("for-each-ref", merge.to_string()))?;
    let tracking = if remote == "." {
        Some(merge.to_string())
    } else {
        lib.find_remote(remote).ok().and_then(|r| {
            r.refspecs().find_map(|spec| {
                (spec.direction() == git2::Direction::Fetch && spec.src_matches(merge))
                    .then(|| spec.transform(merge).ok())
                    .flatten()
                    .and_then(|mapped| mapped.as_str().map(str::to_string))
            })
        })
    };
    let (mut ahead, mut behind, mut gone) = (0, 0, false);
    if let Some(tracking) = tracking {
        match lib.refname_to_id(&tracking) {
            Ok(upstream) => {
                let local =
                    git2::Oid::from_str(local.as_str()).map_err(|e| failed("from_str", e))?;
                let (a, b) = lib
                    .graph_ahead_behind(local, upstream)
                    .map_err(|e| failed("graph_ahead_behind", e))?;
                (ahead, behind) = (a as u32, b as u32);
            }
            Err(e) if is_absent(&e) => gone = true,
            Err(e) => return Err(failed("refname_to_id", e)),
        }
    }
    Ok(Upstream {
        remote: match remote {
            "." => None,
            name => Some(RemoteName::parse(name)?),
        },
        branch,
        ahead,
        behind,
        gone,
    })
}

/// Every tag, by name.
pub(crate) fn tags(lib: &Repository) -> Result<Vec<Tag>, GitError> {
    let mut out = Vec::new();
    for reference in refs_with_prefix(lib, "refs/tags/")? {
        let full = name_of(&reference)?;
        let bad = || GitError::malformed("for-each-ref", full.clone());
        let name = TagName::from_ref(&RefName::parse(&full)?).ok_or_else(bad)?;
        let id = reference
            .resolve()
            .map_err(|e| failed("resolve", e))?
            .target()
            .ok_or_else(bad)?;
        let object = lib
            .find_object(id, None)
            .map_err(|e| failed("find_object", e))?;
        let (annotated, target) = match object.as_tag() {
            Some(tag) => (true, oid_of(tag.target_id())?),
            None => (false, oid_of(id)?),
        };
        out.push(Tag {
            name,
            oid: oid_of(id)?,
            target,
            annotated,
        });
    }
    Ok(out)
}

/// Every remote-tracking branch, as of the last fetch or push. A remote's own
/// `HEAD` — a symbolic ref — names a branch and is not one, and is left out.
pub(crate) fn remote_branches(lib: &Repository) -> Result<Vec<RemoteBranch>, GitError> {
    let mut out = Vec::new();
    for reference in refs_with_prefix(lib, "refs/remotes/")? {
        if reference.kind() == Some(ReferenceType::Symbolic) {
            continue;
        }
        let full = name_of(&reference)?;
        let bad = || GitError::malformed("for-each-ref", full.clone());
        let rest = full.strip_prefix("refs/remotes/").ok_or_else(bad)?;
        let (remote, branch) = rest.split_once('/').ok_or_else(bad)?;
        out.push(RemoteBranch {
            remote: RemoteName::parse(remote)?,
            branch: BranchName::parse(branch)?,
            oid: target_of(&reference)?,
        });
    }
    Ok(out)
}
