//! The `git_odb_backend` libgit2 reads and writes a repository's objects
//! through (`docs/substrate_git.md` §4.1, §4.4).
//!
//! Reads answer from the database: its pending objects, then its packs. Writes
//! land among the pending objects. Every callback catches a panic before it
//! can unwind into C, and reports a failure as libgit2's own error code with
//! its message set.

use std::ffi::CString;
use std::os::raw::{c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{mem, ptr, slice};

use libgit2_sys as raw;

use super::db::ObjectDb;
use super::pack::{Object, ObjectType, RawId, HASH};
use crate::error::GitError;

/// The backend: libgit2's struct first, so a pointer to one is a pointer to
/// the other.
#[repr(C)]
struct Backend {
    parent: raw::git_odb_backend,
    db: ObjectDb,
}

/// A backend over `db`, owned by whatever `git_odb` it is added to, which
/// frees it.
pub(crate) fn new(db: ObjectDb) -> Result<*mut raw::git_odb_backend, GitError> {
    // SAFETY: an all-zero `git_odb_backend` is every callback `None` and every
    // pointer null; `git_odb_init_backend` then sets its version.
    let mut backend = Box::new(Backend {
        parent: unsafe { mem::zeroed() },
        db,
    });
    let rc =
        unsafe { raw::git_odb_init_backend(&mut backend.parent, raw::GIT_ODB_BACKEND_VERSION) };
    if rc != 0 {
        return Err(GitError::Packs(
            "libgit2 refused the object backend's version".into(),
        ));
    }
    backend.parent.read = Some(read);
    backend.parent.read_prefix = Some(read_prefix);
    backend.parent.read_header = Some(read_header);
    backend.parent.write = Some(write);
    backend.parent.exists = Some(exists);
    backend.parent.exists_prefix = Some(exists_prefix);
    backend.parent.refresh = Some(refresh);
    backend.parent.foreach = Some(foreach);
    backend.parent.free = Some(free);
    Ok(Box::into_raw(backend) as *mut raw::git_odb_backend)
}

/// The database a callback's backend pointer carries.
///
/// SAFETY: `backend` is one [`new`] made, alive until [`free`].
unsafe fn db<'a>(backend: *mut raw::git_odb_backend) -> &'a ObjectDb {
    &(*(backend as *mut Backend)).db
}

fn id_of(oid: *const raw::git_oid) -> RawId {
    // SAFETY: libgit2 hands a valid id; a SHA-1 one is its first 20 bytes.
    let bytes = unsafe { &(*oid).id };
    RawId(bytes[..HASH].try_into().expect("twenty bytes"))
}

fn set_oid(out: *mut raw::git_oid, id: &RawId) {
    // SAFETY: libgit2 hands a writable id.
    let bytes: &mut [u8; raw::GIT_OID_MAX_SIZE] = unsafe { &mut (*out).id };
    *bytes = [0; raw::GIT_OID_MAX_SIZE];
    bytes[..HASH].copy_from_slice(&id.0);
}

fn kind_code(kind: ObjectType) -> raw::git_object_t {
    match kind {
        ObjectType::Commit => raw::GIT_OBJECT_COMMIT,
        ObjectType::Tree => raw::GIT_OBJECT_TREE,
        ObjectType::Blob => raw::GIT_OBJECT_BLOB,
        ObjectType::Tag => raw::GIT_OBJECT_TAG,
    }
}

fn kind_of(code: raw::git_object_t) -> Option<ObjectType> {
    match code {
        raw::GIT_OBJECT_COMMIT => Some(ObjectType::Commit),
        raw::GIT_OBJECT_TREE => Some(ObjectType::Tree),
        raw::GIT_OBJECT_BLOB => Some(ObjectType::Blob),
        raw::GIT_OBJECT_TAG => Some(ObjectType::Tag),
        _ => None,
    }
}

/// `e` as libgit2's last error, and its generic failure code.
fn fail(e: GitError) -> c_int {
    let message = CString::new(e.to_string().replace('\0', " ")).unwrap_or_default();
    // SAFETY: libgit2 copies the message.
    unsafe {
        raw::git_error_set_str(raw::GIT_ERROR_ODB as c_int, message.as_ptr());
    }
    raw::GIT_ERROR
}

/// Run `f`, a panic reported as a failure rather than unwound into C.
fn guarded(f: impl FnOnce() -> Result<c_int, GitError>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(code)) => code,
        Ok(Err(e)) => fail(e),
        Err(_) => fail(GitError::Packs("the object backend panicked".into())),
    }
}

/// Hand `object` to libgit2 in a buffer it owns.
///
/// SAFETY: the out pointers are libgit2's, and `backend` is ours.
unsafe fn hand_over(
    object: &Object,
    data_out: *mut *mut c_void,
    len_out: *mut usize,
    type_out: *mut raw::git_object_t,
    backend: *mut raw::git_odb_backend,
) -> Result<c_int, GitError> {
    let len = object.data.len();
    let buffer = raw::git_odb_backend_data_alloc(backend, len.max(1)) as *mut u8;
    if buffer.is_null() {
        return Err(GitError::Packs(
            "libgit2 could not allocate an object's buffer".into(),
        ));
    }
    ptr::copy_nonoverlapping(object.data.as_ptr(), buffer, len);
    *data_out = buffer as *mut c_void;
    *len_out = len;
    *type_out = kind_code(object.kind);
    Ok(0)
}

extern "C" fn read(
    data_out: *mut *mut c_void,
    len_out: *mut usize,
    type_out: *mut raw::git_object_t,
    backend: *mut raw::git_odb_backend,
    oid: *const raw::git_oid,
) -> c_int {
    guarded(|| {
        let id = id_of(oid);
        // SAFETY: `backend` is ours, alive while libgit2 calls it.
        match unsafe { db(backend) }.read(&id)? {
            Some(object) => unsafe { hand_over(&object, data_out, len_out, type_out, backend) },
            None => Ok(raw::GIT_ENOTFOUND),
        }
    })
}

/// The one id a prefix names: `Err(code)` for none, or for more than one.
fn unique(db: &ObjectDb, short: *const raw::git_oid, len: usize) -> Result<RawId, c_int> {
    let prefix = id_of(short);
    let found = db.prefixed(&prefix, len);
    match found.as_slice() {
        [one] => Ok(*one),
        [] => Err(raw::GIT_ENOTFOUND),
        _ => Err(raw::GIT_EAMBIGUOUS),
    }
}

extern "C" fn read_prefix(
    out: *mut raw::git_oid,
    data_out: *mut *mut c_void,
    len_out: *mut usize,
    type_out: *mut raw::git_object_t,
    backend: *mut raw::git_odb_backend,
    short: *const raw::git_oid,
    len: usize,
) -> c_int {
    guarded(|| {
        // SAFETY: `backend` is ours.
        let db = unsafe { db(backend) };
        let id = match unique(db, short, len) {
            Ok(id) => id,
            Err(code) => return Ok(code),
        };
        match db.read(&id)? {
            Some(object) => {
                set_oid(out, &id);
                unsafe { hand_over(&object, data_out, len_out, type_out, backend) }
            }
            None => Ok(raw::GIT_ENOTFOUND),
        }
    })
}

extern "C" fn read_header(
    len_out: *mut usize,
    type_out: *mut raw::git_object_t,
    backend: *mut raw::git_odb_backend,
    oid: *const raw::git_oid,
) -> c_int {
    guarded(|| {
        let id = id_of(oid);
        // SAFETY: `backend` is ours; the out pointers are libgit2's.
        match unsafe { db(backend) }.header(&id)? {
            Some((kind, size)) => unsafe {
                *len_out = size as usize;
                *type_out = kind_code(kind);
                Ok(0)
            },
            None => Ok(raw::GIT_ENOTFOUND),
        }
    })
}

extern "C" fn write(
    backend: *mut raw::git_odb_backend,
    _oid: *const raw::git_oid,
    data: *const c_void,
    len: usize,
    kind: raw::git_object_t,
) -> c_int {
    guarded(|| {
        let kind = kind_of(kind)
            .ok_or_else(|| GitError::Packs(format!("an object of type {kind} to write")))?;
        // SAFETY: libgit2 hands `len` readable bytes.
        let bytes = if len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(data as *const u8, len) }.to_vec()
        };
        unsafe { db(backend) }.write(Object { kind, data: bytes });
        Ok(0)
    })
}

extern "C" fn exists(backend: *mut raw::git_odb_backend, oid: *const raw::git_oid) -> c_int {
    guarded(|| {
        let id = id_of(oid);
        // SAFETY: `backend` is ours.
        Ok(c_int::from(unsafe { db(backend) }.contains(&id)))
    })
}

extern "C" fn exists_prefix(
    out: *mut raw::git_oid,
    backend: *mut raw::git_odb_backend,
    short: *const raw::git_oid,
    len: usize,
) -> c_int {
    guarded(|| {
        // SAFETY: `backend` is ours.
        match unique(unsafe { db(backend) }, short, len) {
            Ok(id) => {
                set_oid(out, &id);
                Ok(0)
            }
            Err(code) => Ok(code),
        }
    })
}

extern "C" fn refresh(backend: *mut raw::git_odb_backend) -> c_int {
    // SAFETY: `backend` is ours.
    guarded(|| unsafe { db(backend) }.refresh().map(|_| 0))
}

extern "C" fn foreach(
    backend: *mut raw::git_odb_backend,
    callback: raw::git_odb_foreach_cb,
    payload: *mut c_void,
) -> c_int {
    guarded(|| {
        // SAFETY: `backend` is ours.
        let db = unsafe { db(backend) };
        let mut ids: Vec<RawId> = db.packs().ids().collect();
        ids.extend(db.pending().ids());
        ids.sort();
        ids.dedup();
        let Some(callback) = callback else {
            return Ok(0);
        };
        for id in ids {
            let mut oid = raw::git_oid {
                id: [0; raw::GIT_OID_MAX_SIZE],
            };
            set_oid(&mut oid, &id);
            // libgit2's callback, with its own payload.
            let rc = callback(&oid, payload);
            if rc != 0 {
                return Ok(rc);
            }
        }
        Ok(0)
    })
}

extern "C" fn free(backend: *mut raw::git_odb_backend) {
    // SAFETY: `backend` is one `new` boxed, freed once, by its odb.
    let _ = catch_unwind(|| unsafe { drop(Box::from_raw(backend as *mut Backend)) });
}
