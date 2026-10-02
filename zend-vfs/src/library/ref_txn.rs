//! Compare-and-swap reference updates, from libgit2.
//!
//! Every ref the transaction names is locked first, each expected value is then
//! checked against what the ref holds under that lock, and only when every
//! check passes is anything written — so a stale ref fails the whole
//! transaction with nothing changed, as `update-ref --stdin` does.

use git2::{ErrorCode, Oid as LibOid, Repository};

use super::{failed, is_absent};
use crate::error::GitError;
use crate::types::{Oid, RefName};
use crate::write::ref_txn::RefOp;

/// The id the ref named `name` holds itself, or `None` when there is no such ref.
fn held(lib: &Repository, name: &RefName) -> Result<Option<Oid>, GitError> {
    match lib.find_reference(name.as_str()) {
        Ok(reference) => match reference.resolve() {
            Ok(direct) => match direct.target() {
                Some(id) => Ok(Some(Oid::parse(&id.to_string())?)),
                None => Ok(None),
            },
            Err(e) if is_absent(&e) => Ok(None),
            Err(e) => Err(failed("resolve", e)),
        },
        Err(e) if is_absent(&e) => Ok(None),
        Err(e) => Err(failed("find_reference", e)),
    }
}

fn lib_id(id: &Oid) -> Result<LibOid, GitError> {
    LibOid::from_str(id.as_str()).map_err(|e| failed("from_str", e))
}

fn stale(name: &RefName, expected: &str, found: Option<&Oid>) -> GitError {
    let found = found.map_or_else(|| "nothing".to_string(), |id| id.to_string());
    GitError::StaleRef {
        name: name.clone(),
        detail: format!("expected {expected}, found {found}"),
    }
}

/// Apply `ops` all together or not at all.
pub(crate) fn apply(lib: &Repository, ops: &[RefOp]) -> Result<(), GitError> {
    let mut txn = lib.transaction().map_err(|e| failed("transaction", e))?;
    let mut locked: Vec<&RefName> = Vec::new();
    for op in ops {
        let name = op.name();
        // `update-ref` refuses a second update of a ref in one transaction.
        if locked.contains(&name) {
            return Err(GitError::invalid(format!(
                "multiple updates for {name} in one transaction"
            )));
        }
        locked.push(name);
        txn.lock_ref(name.as_str()).map_err(|e| {
            if e.code() == ErrorCode::Locked {
                GitError::RefLocked { name: name.clone() }
            } else {
                failed("lock_ref", e)
            }
        })?;
    }
    for op in ops {
        let name = op.name();
        let now = held(lib, name)?;
        match op {
            RefOp::Create { .. } => {
                if now.is_some() {
                    return Err(stale(name, "no such ref", now.as_ref()));
                }
            }
            RefOp::Update { old, .. } | RefOp::Delete { old, .. } => {
                if now.as_ref() != Some(old) {
                    return Err(stale(name, old.as_str(), now.as_ref()));
                }
            }
        }
    }
    for op in ops {
        match op {
            RefOp::Create { name, new } | RefOp::Update { name, new, .. } => {
                txn.set_target(name.as_str(), lib_id(new)?, None, "")
                    .map_err(|e| failed("set_target", e))?;
            }
            RefOp::Delete { name, .. } => {
                txn.remove(name.as_str()).map_err(|e| failed("remove", e))?;
            }
        }
    }
    txn.commit().map_err(|e| failed("commit", e))
}
