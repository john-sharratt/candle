//! A served page's recurrent record: the rows of its `in_context` reply as a
//! reader's recurrence took them (`docs/recurrent_reply_calibration.md` §3–§5).
//!
//! One record per page of a `code_reading` unit. The object, under
//! [`slot_key`], names the recipe the rows were calibrated under and the blob
//! that holds them; the rows themselves are the blob, so the substrate holds a
//! hash in RAM per page rather than the rows. A record whose recipe is not the
//! current one still serves until the sweep replaces it.

use std::sync::Arc;

use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::recurrent_span::ReplyRows;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::mcp::records::Records;

/// Where every record's key starts.
pub const PREFIX: &str = "recurrent/";

/// The record of page `page` of the unit committed under `content_key`.
pub fn slot_key(content_key: &str, page: u32) -> String {
    format!("{PREFIX}{content_key}/{page}")
}

/// What a record was calibrated over: the texts prefilled in the fork after the
/// unit — the request, the call, the reply — and the frame's system prompt. A
/// change to any of them is a new recipe.
pub fn recipe(request: &str, call: &str, reply: &str, system_prompt: &str) -> String {
    let mut h = Sha256::new();
    for part in [request, call, reply, system_prompt] {
        h.update(part.as_bytes());
        h.update([0u8]);
    }
    hex(&h.finalize()[..8])
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    if s.len() != 64 {
        return None;
    }
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// A slot's object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplyRecord {
    pub recipe: String,
    /// The SHA-256 of the blob holding the rows, hex.
    pub rows: String,
}

/// The record under `key`, if any.
pub fn record(store: &SharedSubstrate, key: &str) -> Option<ReplyRecord> {
    store.get(key).and_then(|v| serde_json::from_value(v).ok())
}

/// The rows the record under `key` holds, read from its blob.
pub fn rows(store: &SharedSubstrate, key: &str) -> anyhow::Result<Option<Arc<ReplyRows>>> {
    let Some(rec) = record(store, key) else {
        return Ok(None);
    };
    let sha =
        unhex(&rec.rows).ok_or_else(|| anyhow::anyhow!("record `{key}`: a malformed blob hash"))?;
    let Some(bytes) = store
        .blob(&sha)
        .map_err(|e| anyhow::anyhow!("record `{key}`: its blob would not read: {e}"))?
    else {
        return Ok(None);
    };
    Ok(Some(Arc::new(ReplyRows::decode(&bytes)?)))
}

/// Keep `rows` under `key`, calibrated under `recipe`, and retire the blob the
/// record held before — only once the new one is committed.
pub fn keep(
    store: &SharedSubstrate,
    key: &str,
    recipe: &str,
    rows: &ReplyRows,
) -> anyhow::Result<()> {
    let before = record(store, key);
    let sha = store
        .put_blob(rows.encode())
        .map_err(|e| anyhow::anyhow!("record `{key}`: the rows were not kept: {e}"))?;
    store.put(
        key,
        json!(ReplyRecord {
            recipe: recipe.to_string(),
            rows: hex(&sha),
        }),
    )?;
    retire_rows(store, before.as_ref(), &sha)
}

/// End the record under `key`, and its blob with it.
pub fn forget(store: &SharedSubstrate, key: &str) -> anyhow::Result<()> {
    let before = record(store, key);
    store.cease(key)?;
    retire_rows(store, before.as_ref(), &[0u8; 32])
}

fn retire_rows(
    store: &SharedSubstrate,
    before: Option<&ReplyRecord>,
    kept: &[u8; 32],
) -> anyhow::Result<()> {
    match before.and_then(|r| unhex(&r.rows)) {
        Some(old) if &old != kept => store
            .retire_blob(&old)
            .map_err(|e| anyhow::anyhow!("an old record's rows were not retired: {e}")),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_is_keyed_by_its_unit_and_page() {
        assert_eq!(
            slot_key("file:candle/a.rs@ab12", 2),
            "recurrent/file:candle/a.rs@ab12/2"
        );
    }

    /// Against a fixed digest: the recipe is a contract a stored record is read
    /// back under, so it may not drift.
    #[test]
    fn a_recipe_is_its_four_texts() {
        let r = recipe("req", "call", "reply", "sys");
        let mut h = Sha256::new();
        h.update(b"req\0call\0reply\0sys\0");
        assert_eq!(r, hex(&h.finalize()[..8]));
        assert_eq!(r.len(), 16);
        assert_ne!(recipe("req", "call", "reply", "sys!"), r);
        assert_ne!(
            recipe("re", "qcall", "reply", "sys"),
            r,
            "the parts are delimited"
        );
    }

    #[test]
    fn a_blob_hash_reads_back() {
        let sha = [0xabu8; 32];
        assert_eq!(unhex(&hex(&sha)), Some(sha));
        assert_eq!(unhex("ab"), None);
    }
}
