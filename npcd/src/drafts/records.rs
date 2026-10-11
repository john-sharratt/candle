//! An authored collection — the worlds, the personalities — as one author sees
//! it (`docs/replicated_substrate.md` §10.3): the registry's records, which
//! are what the mind folder holds as published, with the author's own drafts
//! laid over them.

use std::collections::BTreeMap;

use anyhow::anyhow;
use serde_json::Value;

use super::Author;
use crate::registry::authored::{path_of, record_of, render};
use crate::registry::id::check;
use crate::registry::{PutError, Record, Registry};
use crate::sim::shelf::Shelf;

/// Every record of `registry`, kept in `folder`, as `author` sees it.
pub fn records(author: &Author, registry: &Registry, folder: &str) -> Vec<Record> {
    let mut by_id: BTreeMap<String, Record> =
        registry.iter().map(|r| (r.id.clone(), r.clone())).collect();
    for (file, text) in author.drafted_in(folder) {
        let Some(stem) = file.strip_suffix(".yaml") else {
            continue;
        };
        match text.and_then(|t| record_of(folder, &file, &t)) {
            Some(record) => {
                by_id.insert(record.id.clone(), record);
            }
            None => {
                by_id.remove(stem);
            }
        }
    }
    by_id.into_values().collect()
}

/// Record `id` of `registry`, kept in `folder`, as `author` sees it.
pub fn record(author: &Author, registry: &Registry, folder: &str, id: &str) -> Option<Record> {
    let path = path_of(folder, id);
    match author.drafted().get(&path) {
        Some(text) => text
            .as_deref()
            .and_then(|t| record_of(folder, &format!("{id}.yaml"), t)),
        None => registry.get(id).cloned(),
    }
}

/// Keep `body` as record `id` of the collection in `folder`: rendered against
/// the text `author` sees, so its comments survive, and committed as theirs.
pub fn save(
    author: &Author,
    kind: &str,
    folder: &str,
    id: &str,
    body: &Value,
) -> Result<(), PutError> {
    let path = path_of(folder, id);
    let yaml = render(kind, id, author.read(&path).as_deref(), body)?;
    // Refused rather than written through: a link would put the document
    // wherever it points, and a folder is not one to replace.
    if author.not_a_document(&path) {
        return Err(PutError::NotAPlainFile);
    }
    author
        .commit(&[(path, Some(yaml))])
        .map_err(|e| PutError::Io(anyhow!(e)))
}

/// Remove record `id` of the collection in `folder` as `author`; whether it
/// was there to remove.
pub fn remove(author: &Author, folder: &str, id: &str) -> Result<bool, PutError> {
    check(id).map_err(PutError::BadId)?;
    let path = path_of(folder, id);
    if author.read(&path).is_none() {
        return Ok(false);
    }
    author
        .commit(&[(path, None)])
        .map_err(|e| PutError::Io(anyhow!(e)))?;
    Ok(true)
}
