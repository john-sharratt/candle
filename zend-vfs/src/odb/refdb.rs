//! The `git_refdb_backend` libgit2 reads and writes a repository's refs
//! through (`docs/substrate_git.md` §4.3), over the database's in-memory refs.
//!
//! A write or delete that names the value it expects is a compare-and-swap; a
//! lock for a transaction holds the ref until the transaction ends, and its
//! end writes or deletes it. There are no reflogs: nothing in a repository read
//! from the substrate keeps one.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{mem, ptr};

use libgit2_sys as raw;

use super::db::ObjectDb;
use super::pack::{RawId, HASH};
use super::refs::RefValue;
use crate::error::GitError;

extern "C" {
    // libgit2's own allocators for a reference a backend hands it, exported
    // (`git2/sys/refs.h`) and not bound by libgit2-sys.
    fn git_reference__alloc(
        name: *const c_char,
        oid: *const raw::git_oid,
        peel: *const raw::git_oid,
    ) -> *mut raw::git_reference;
    fn git_reference__alloc_symbolic(
        name: *const c_char,
        target: *const c_char,
    ) -> *mut raw::git_reference;
}

#[repr(C)]
struct Backend {
    parent: raw::git_refdb_backend,
    db: ObjectDb,
}

/// A backend over `db`'s refs, owned by the `git_refdb` it is set on.
pub(crate) fn new(db: ObjectDb) -> Result<*mut raw::git_refdb_backend, GitError> {
    // SAFETY: an all-zero backend is every callback `None`; the init sets its
    // version.
    let mut backend = Box::new(Backend {
        parent: unsafe { mem::zeroed() },
        db,
    });
    let rc =
        unsafe { raw::git_refdb_init_backend(&mut backend.parent, raw::GIT_REFDB_BACKEND_VERSION) };
    if rc != 0 {
        return Err(GitError::Packs(
            "libgit2 refused the ref backend's version".into(),
        ));
    }
    let b = &mut backend.parent;
    b.exists = Some(exists);
    b.lookup = Some(lookup);
    b.iterator = Some(iterator);
    b.write = Some(write);
    b.rename = Some(rename);
    b.del = Some(del);
    b.compress = Some(compress);
    b.has_log = Some(has_log);
    b.ensure_log = Some(ensure_log);
    b.free = Some(free);
    b.reflog_read = Some(reflog_read);
    b.reflog_write = Some(reflog_write);
    b.reflog_rename = Some(reflog_rename);
    b.reflog_delete = Some(reflog_delete);
    b.lock = Some(lock);
    b.unlock = Some(unlock);
    Ok(Box::into_raw(backend) as *mut raw::git_refdb_backend)
}

/// SAFETY: `backend` is one [`new`] made, alive until [`free`].
unsafe fn db<'a>(backend: *mut raw::git_refdb_backend) -> &'a ObjectDb {
    &(*(backend as *mut Backend)).db
}

fn fail(e: GitError) -> c_int {
    let message = CString::new(e.to_string().replace('\0', " ")).unwrap_or_default();
    // SAFETY: libgit2 copies the message.
    unsafe {
        raw::git_error_set_str(raw::GIT_ERROR_REFERENCE as c_int, message.as_ptr());
    }
    raw::GIT_ERROR
}

fn guarded(f: impl FnOnce() -> Result<c_int, GitError>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(code)) => code,
        Ok(Err(e)) => fail(e),
        Err(_) => fail(GitError::Packs("the ref backend panicked".into())),
    }
}

/// A C string libgit2 handed over, as text.
fn text(s: *const c_char) -> Result<String, GitError> {
    // SAFETY: libgit2 hands a valid, NUL-terminated string.
    unsafe { CStr::from_ptr(s) }
        .to_str()
        .map(str::to_string)
        .map_err(|_| GitError::invalid("a ref name that is not UTF-8"))
}

fn oid_of(id: &RawId) -> raw::git_oid {
    let mut oid = raw::git_oid {
        id: [0; raw::GIT_OID_MAX_SIZE],
    };
    oid.id[..HASH].copy_from_slice(&id.0);
    oid
}

fn id_of(oid: *const raw::git_oid) -> RawId {
    // SAFETY: libgit2 hands a valid id.
    let bytes = unsafe { &(*oid).id };
    RawId(bytes[..HASH].try_into().expect("twenty bytes"))
}

/// `value` under `name` as a `git_reference` libgit2 owns and frees.
fn reference(name: &str, value: &RefValue) -> Result<*mut raw::git_reference, GitError> {
    let name = CString::new(name).map_err(|_| GitError::invalid("a ref name holding NUL"))?;
    // SAFETY: libgit2 copies both strings and the id.
    let made = unsafe {
        match value {
            RefValue::Direct(id) => git_reference__alloc(name.as_ptr(), &oid_of(id), ptr::null()),
            RefValue::Symbolic(target) => {
                let target = CString::new(target.as_str())
                    .map_err(|_| GitError::invalid("a ref target holding NUL"))?;
                git_reference__alloc_symbolic(name.as_ptr(), target.as_ptr())
            }
        }
    };
    if made.is_null() {
        return Err(GitError::Packs(
            "libgit2 could not allocate a reference".into(),
        ));
    }
    Ok(made)
}

/// What `reference` holds, as the database keeps it.
fn value_of(reference: *const raw::git_reference) -> Result<RefValue, GitError> {
    // SAFETY: libgit2 hands a valid reference.
    unsafe {
        match raw::git_reference_type(reference) {
            raw::GIT_REFERENCE_SYMBOLIC => Ok(RefValue::Symbolic(text(
                raw::git_reference_symbolic_target(reference),
            )?)),
            _ => {
                let target = raw::git_reference_target(reference);
                if target.is_null() {
                    return Err(GitError::invalid("a direct reference with no target"));
                }
                Ok(RefValue::Direct(id_of(target)))
            }
        }
    }
}

/// Whether `held` is what a caller expecting `old_id` or `old_target` — each
/// unchecked when null — expects.
fn as_expected(
    held: Option<&RefValue>,
    old_id: *const raw::git_oid,
    old_target: *const c_char,
) -> Result<bool, GitError> {
    if !old_id.is_null() {
        let expected = id_of(old_id);
        if held != Some(&RefValue::Direct(expected)) {
            return Ok(false);
        }
    }
    if !old_target.is_null() {
        let expected = text(old_target)?;
        if held != Some(&RefValue::Symbolic(expected)) {
            return Ok(false);
        }
    }
    Ok(true)
}

extern "C" fn exists(
    out: *mut c_int,
    backend: *mut raw::git_refdb_backend,
    name: *const c_char,
) -> c_int {
    guarded(|| {
        let name = text(name)?;
        // SAFETY: `backend` is ours; `out` libgit2's.
        let found = unsafe { db(backend) }.refs().get(&name).is_some();
        unsafe { *out = c_int::from(found) };
        Ok(0)
    })
}

extern "C" fn lookup(
    out: *mut *mut raw::git_reference,
    backend: *mut raw::git_refdb_backend,
    name: *const c_char,
) -> c_int {
    guarded(|| {
        let name = text(name)?;
        // SAFETY: `backend` is ours.
        let value = unsafe { db(backend) }.refs().get(&name).cloned();
        match value {
            Some(value) => {
                let made = reference(&name, &value)?;
                unsafe { *out = made };
                Ok(0)
            }
            None => Ok(raw::GIT_ENOTFOUND),
        }
    })
}

/// A listing of refs, libgit2's iterator first.
#[repr(C)]
struct Iter {
    parent: raw::git_reference_iterator,
    refs: Vec<(CString, RefValue)>,
    at: usize,
}

extern "C" fn iterator(
    out: *mut *mut raw::git_reference_iterator,
    backend: *mut raw::git_refdb_backend,
    glob: *const c_char,
) -> c_int {
    guarded(|| {
        let glob = if glob.is_null() {
            None
        } else {
            Some(text(glob)?)
        };
        // SAFETY: `backend` is ours.
        let listed = unsafe { db(backend) }.refs().matching(glob.as_deref());
        // As git's own listing: the refs under `refs/`, never `HEAD`.
        let refs = listed
            .into_iter()
            .filter(|(name, _)| name.starts_with("refs/"))
            .map(|(name, value)| {
                Ok((
                    CString::new(name).map_err(|_| GitError::invalid("a ref name holding NUL"))?,
                    value,
                ))
            })
            .collect::<Result<Vec<_>, GitError>>()?;
        let iter = Box::new(Iter {
            parent: raw::git_reference_iterator {
                db: ptr::null_mut(),
                next: Some(iter_next),
                next_name: Some(iter_next_name),
                free: Some(iter_free),
            },
            refs,
            at: 0,
        });
        unsafe { *out = Box::into_raw(iter) as *mut raw::git_reference_iterator };
        Ok(0)
    })
}

extern "C" fn iter_next(
    out: *mut *mut raw::git_reference,
    iter: *mut raw::git_reference_iterator,
) -> c_int {
    guarded(|| {
        // SAFETY: `iter` is one `iterator` boxed.
        let iter = unsafe { &mut *(iter as *mut Iter) };
        let Some((name, value)) = iter.refs.get(iter.at) else {
            return Ok(raw::GIT_ITEROVER);
        };
        let name = name
            .to_str()
            .map_err(|_| GitError::invalid("a ref name that is not UTF-8"))?;
        let made = reference(name, value)?;
        iter.at += 1;
        unsafe { *out = made };
        Ok(0)
    })
}

extern "C" fn iter_next_name(
    out: *mut *const c_char,
    iter: *mut raw::git_reference_iterator,
) -> c_int {
    guarded(|| {
        // SAFETY: `iter` is one `iterator` boxed; the name lives as long as it.
        let iter = unsafe { &mut *(iter as *mut Iter) };
        let Some((name, _)) = iter.refs.get(iter.at) else {
            return Ok(raw::GIT_ITEROVER);
        };
        unsafe { *out = name.as_ptr() };
        iter.at += 1;
        Ok(0)
    })
}

extern "C" fn iter_free(iter: *mut raw::git_reference_iterator) {
    // SAFETY: freed once, by libgit2.
    let _ = catch_unwind(|| unsafe { drop(Box::from_raw(iter as *mut Iter)) });
}

extern "C" fn write(
    backend: *mut raw::git_refdb_backend,
    reference: *const raw::git_reference,
    force: c_int,
    _who: *const raw::git_signature,
    _message: *const c_char,
    old_id: *const raw::git_oid,
    old_target: *const c_char,
) -> c_int {
    guarded(|| {
        // SAFETY: libgit2 hands a valid reference; `backend` is ours.
        let name = text(unsafe { raw::git_reference_name(reference) })?;
        let value = value_of(reference)?;
        let db = unsafe { db(backend) };
        let mut refs = db.refs();
        let held = refs.get(&name).cloned();
        if force == 0 && held.is_some() && old_id.is_null() && old_target.is_null() {
            return Ok(raw::GIT_EEXISTS);
        }
        if !as_expected(held.as_ref(), old_id, old_target)? {
            return Ok(raw::GIT_EMODIFIED);
        }
        refs.set(&name, value)?;
        Ok(0)
    })
}

extern "C" fn rename(
    out: *mut *mut raw::git_reference,
    backend: *mut raw::git_refdb_backend,
    old_name: *const c_char,
    new_name: *const c_char,
    force: c_int,
    _who: *const raw::git_signature,
    _message: *const c_char,
) -> c_int {
    guarded(|| {
        let (old_name, new_name) = (text(old_name)?, text(new_name)?);
        // SAFETY: `backend` is ours.
        let db = unsafe { db(backend) };
        let mut refs = db.refs();
        let Some(value) = refs.get(&old_name).cloned() else {
            return Ok(raw::GIT_ENOTFOUND);
        };
        if force == 0 && refs.get(&new_name).is_some() {
            return Ok(raw::GIT_EEXISTS);
        }
        refs.set(&new_name, value.clone())?;
        refs.delete(&old_name)?;
        let made = reference(&new_name, &value)?;
        unsafe { *out = made };
        Ok(0)
    })
}

extern "C" fn del(
    backend: *mut raw::git_refdb_backend,
    name: *const c_char,
    old_id: *const raw::git_oid,
    old_target: *const c_char,
) -> c_int {
    guarded(|| {
        let name = text(name)?;
        // SAFETY: `backend` is ours.
        let db = unsafe { db(backend) };
        let mut refs = db.refs();
        let held = refs.get(&name).cloned();
        if held.is_none() {
            return Ok(raw::GIT_ENOTFOUND);
        }
        if !as_expected(held.as_ref(), old_id, old_target)? {
            return Ok(raw::GIT_EMODIFIED);
        }
        refs.delete(&name)?;
        Ok(0)
    })
}

extern "C" fn compress(_backend: *mut raw::git_refdb_backend) -> c_int {
    0
}

extern "C" fn has_log(_backend: *mut raw::git_refdb_backend, _name: *const c_char) -> c_int {
    0
}

extern "C" fn ensure_log(_backend: *mut raw::git_refdb_backend, _name: *const c_char) -> c_int {
    0
}

extern "C" fn reflog_read(
    _out: *mut *mut raw::git_reflog,
    _backend: *mut raw::git_refdb_backend,
    _name: *const c_char,
) -> c_int {
    fail(GitError::invalid(
        "a repository read from the substrate keeps no reflog",
    ))
}

extern "C" fn reflog_write(
    _backend: *mut raw::git_refdb_backend,
    _reflog: *mut raw::git_reflog,
) -> c_int {
    0
}

extern "C" fn reflog_rename(
    _backend: *mut raw::git_refdb_backend,
    _old: *const c_char,
    _new: *const c_char,
) -> c_int {
    0
}

extern "C" fn reflog_delete(_backend: *mut raw::git_refdb_backend, _name: *const c_char) -> c_int {
    0
}

extern "C" fn lock(
    payload: *mut *mut c_void,
    backend: *mut raw::git_refdb_backend,
    name: *const c_char,
) -> c_int {
    guarded(|| {
        let name = text(name)?;
        // SAFETY: `backend` is ours.
        if !unsafe { db(backend) }.refs().lock(&name) {
            return Ok(raw::GIT_ELOCKED);
        }
        let held = Box::new(name);
        unsafe { *payload = Box::into_raw(held) as *mut c_void };
        Ok(0)
    })
}

/// How a transaction ends a lock: `1` writes the ref, `2` deletes it, `0`
/// leaves it as it was.
const UNLOCK_WRITE: c_int = 1;
const UNLOCK_DELETE: c_int = 2;

extern "C" fn unlock(
    backend: *mut raw::git_refdb_backend,
    payload: *mut c_void,
    success: c_int,
    _update_reflog: c_int,
    reference: *const raw::git_reference,
    _who: *const raw::git_signature,
    _message: *const c_char,
) -> c_int {
    guarded(|| {
        // SAFETY: `payload` is the name `lock` boxed, released once here.
        let name = *unsafe { Box::from_raw(payload as *mut String) };
        let db = unsafe { db(backend) };
        let mut refs = db.refs();
        refs.unlock(&name);
        match success {
            UNLOCK_WRITE => {
                let value = value_of(reference)?;
                refs.set(&name, value)?;
            }
            UNLOCK_DELETE => {
                refs.delete(&name)?;
            }
            _ => {}
        }
        Ok(0)
    })
}

extern "C" fn free(backend: *mut raw::git_refdb_backend) {
    // SAFETY: freed once, by its refdb.
    let _ = catch_unwind(|| unsafe { drop(Box::from_raw(backend as *mut Backend)) });
}
