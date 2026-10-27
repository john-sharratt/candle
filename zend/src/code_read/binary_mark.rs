//! The files a reading found to be binary behind a text extension, recorded
//! in the substrate.
//!
//! A reading sniffs a file's bytes before minting its conversation and skips
//! one that is binary — a compiled dump or an instruction listing committed as
//! `*.txt`. The verdict was held in memory alone, so a file never read stayed
//! unread to the planner: every pass and every restart queued it again, and a
//! member that did not hold its ingest lease never sniffed it at all
//! (BlackWorld's `perf-investigation/*.txt` and flash-attn instruction
//! summaries, 2026-10-27). Recorded as a keyed object, the verdict survives a
//! restart and replicates to every member, and the planner leaves the file out
//! ([`recorded`]).

use std::collections::HashSet;

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::projection::Conversation;
use serde_json::Value;

/// The keyed objects' prefix: one per binary file's content key.
const PREFIX: &str = "code_read/binary/";

/// Record that the file `content_key` names is binary.
pub fn record(conversation: &Conversation, content_key: &str) -> anyhow::Result<()> {
    conversation
        .shared()
        .put_custom_object(CustomObjectPayload::new(
            format!("{PREFIX}{content_key}"),
            Value::Null,
        ))?;
    Ok(())
}

/// Every content key recorded binary.
pub fn recorded(conversation: &Conversation) -> HashSet<String> {
    conversation
        .shared()
        .custom_objects_under(PREFIX)
        .into_iter()
        .filter_map(|(obj, _)| obj.key.strip_prefix(PREFIX).map(str::to_owned))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A file recorded binary is binary to every later reader of the
    /// substrate**, by its content key alone, and nothing else is.
    #[test]
    fn a_recorded_file_reads_back_by_its_content_key() {
        let conversation = Conversation::new();
        assert!(recorded(&conversation).is_empty());
        record(&conversation, "blob:ab12").unwrap();
        record(&conversation, "blob:cd34").unwrap();
        record(&conversation, "blob:ab12").unwrap();
        assert_eq!(
            recorded(&conversation),
            HashSet::from(["blob:ab12".to_string(), "blob:cd34".to_string()])
        );
    }
}
