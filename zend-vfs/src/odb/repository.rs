//! A libgit2 repository over an [`ObjectDb`]: an object database holding only
//! our backend, wrapped as a repository with no folder, its refs our refdb
//! (`docs/substrate_git.md` §4).
//!
//! It is built through libgit2's C interface and handed to the `git2` crate as
//! a `Repository`, which owns and frees it. `git2` keeps a repository as its
//! raw pointer and nothing else; the assertion below breaks the build if a
//! release ever changes that, rather than letting a mismatched value through.

use std::mem;
use std::os::raw::c_int;
use std::ptr;

use libgit2_sys as raw;

use super::db::ObjectDb;
use super::{backend, refdb};
use crate::error::GitError;

const _: () = assert!(
    mem::size_of::<git2::Repository>() == mem::size_of::<*mut raw::git_repository>(),
    "git2::Repository is no longer its raw pointer alone"
);

/// libgit2's failure `rc` in `what`, with its message.
fn failure(what: &str, rc: c_int) -> GitError {
    let detail = git2::Error::last_error(rc).message().to_string();
    GitError::Packs(format!("libgit2 {what}: {detail}"))
}

fn check(what: &str, rc: c_int) -> Result<(), GitError> {
    match rc {
        0.. => Ok(()),
        _ => Err(failure(what, rc)),
    }
}

/// A repository over `db`, its objects and refs read and written through it.
pub(crate) fn repository(db: &ObjectDb) -> Result<git2::Repository, GitError> {
    raw::init();
    // SAFETY: each pointer is made by libgit2 here and handed on to the next
    // call, which takes its own reference; ours are released as each step
    // ends, and on failure what was made is freed.
    unsafe {
        let mut odb = ptr::null_mut();
        check("odb_new", raw::git_odb_new(&mut odb))?;
        let objects = backend::new(db.clone())?;
        let added = raw::git_odb_add_backend(odb, objects, 1);
        if added < 0 {
            if let Some(free) = (*objects).free {
                free(objects);
            }
            raw::git_odb_free(odb);
            return Err(failure("odb_add_backend", added));
        }

        let mut repo = ptr::null_mut();
        let wrapped = raw::git_repository_wrap_odb(&mut repo, odb);
        raw::git_odb_free(odb);
        check("repository_wrap_odb", wrapped)?;

        let mut refs = ptr::null_mut();
        let made = raw::git_refdb_new(&mut refs, repo);
        if made < 0 {
            raw::git_repository_free(repo);
            return Err(failure("refdb_new", made));
        }
        let backend = match refdb::new(db.clone()) {
            Ok(backend) => backend,
            Err(e) => {
                raw::git_refdb_free(refs);
                raw::git_repository_free(repo);
                return Err(e);
            }
        };
        let set = raw::git_refdb_set_backend(refs, backend);
        if set < 0 {
            if let Some(free) = (*backend).free {
                free(backend);
            }
            raw::git_refdb_free(refs);
            raw::git_repository_free(repo);
            return Err(failure("refdb_set_backend", set));
        }
        let installed = raw::git_repository_set_refdb(repo, refs);
        raw::git_refdb_free(refs);
        if installed < 0 {
            raw::git_repository_free(repo);
            return Err(failure("repository_set_refdb", installed));
        }
        // SAFETY: the assertion above: `git2::Repository` is the pointer.
        Ok(mem::transmute::<*mut raw::git_repository, git2::Repository>(repo))
    }
}

#[cfg(test)]
mod tests;
