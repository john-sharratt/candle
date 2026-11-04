//! Ends the recurrent reply records an earlier release kept: each object under
//! `recurrent/` (`{recipe, rows}`) is ceased and the blob of rows it names is
//! retired. Both tombstones replicate, so whichever member runs a pass first
//! clears every member, and the blobs' bytes go at the next compaction.

use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::ConversationEngine;

use crate::mcp::records::Records;

/// Where every record's key starts.
const PREFIX: &str = "recurrent/";

/// End every reply record and its rows. Returns how many it ended.
pub fn retire_reply_records(engine: &ConversationEngine) -> usize {
    let store = engine.conversation().shared();
    store
        .under(PREFIX)
        .into_iter()
        .filter(|(key, value)| retire(&store, key, value).is_ok())
        .count()
}

fn retire(store: &SharedSubstrate, key: &str, value: &serde_json::Value) -> anyhow::Result<()> {
    store.cease(key)?;
    if let Some(sha) = value
        .get("rows")
        .and_then(|v| v.as_str())
        .and_then(blob_hash)
    {
        store
            .retire_blob(&sha)
            .map_err(|e| anyhow::anyhow!("record `{key}`: its rows were not retired: {e}"))?;
    }
    Ok(())
}

/// A record's blob hash, 64 hex digits.
fn blob_hash(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_s_blob_hash_reads_back() {
        let hex = "ab".repeat(32);
        assert_eq!(blob_hash(&hex), Some([0xab; 32]));
        assert_eq!(blob_hash("ab"), None);
        assert_eq!(blob_hash(&"zz".repeat(32)), None);
    }
}
