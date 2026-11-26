//! How often a unit's calibration has failed, recorded in the substrate.
//!
//! A file of the knowledge base is a record, not a conversation, so the count
//! lives in a keyed object of its own — `calibration/failed/{unit}`, holding
//! `"{marker} {n}"` — which replicates as the record does: a member that
//! takes the unit over knows how often it failed elsewhere.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::projection::Conversation;
use serde_json::Value;

const PREFIX: &str = "calibration/failed/";

/// What `unit`'s failures have recorded, if any: `"{marker} {n}"`.
pub fn recorded(conversation: &Conversation, unit: &str) -> Option<String> {
    let obj = conversation
        .shared()
        .custom_object(&format!("{PREFIX}{unit}"))?;
    if obj.ceased {
        return None;
    }
    obj.blob.as_str().map(str::to_owned)
}

/// Record that `unit` has failed `n` times under `marker`.
pub fn record(conversation: &Conversation, unit: &str, marker: &str, n: u32) -> anyhow::Result<()> {
    conversation
        .shared()
        .put_custom_object(CustomObjectPayload::new(
            format!("{PREFIX}{unit}"),
            Value::String(format!("{marker} {n}")),
        ))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A unit's failures read back as written, and another unit has none.**
    #[test]
    fn a_units_failures_read_back_as_written() {
        let conversation = Conversation::new();
        assert_eq!(recorded(&conversation, "empty:v7:a@1"), None);
        record(
            &conversation,
            "empty:v7:a@1",
            "code_reading|ab|empty:v7:a@1",
            2,
        )
        .unwrap();
        assert_eq!(
            recorded(&conversation, "empty:v7:a@1").as_deref(),
            Some("code_reading|ab|empty:v7:a@1 2")
        );
        assert_eq!(recorded(&conversation, "target:v7:a@1"), None);
    }
}
