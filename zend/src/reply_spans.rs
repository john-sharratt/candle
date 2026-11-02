//! The replay spans of a round's results turn: each block a served read
//! answered, with the rows of the page's recurrent record
//! (`docs/recurrent_reply_calibration.md` §7).

use std::ops::Range;

use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::recurrent_span::{ArmKind, TextSpan};

use crate::fast_path::Served;
use crate::reply_record::{self, slot_key};
use crate::tools::ToolResult;

/// A span per block of `results` (whose bytes are `blocks`) that one of
/// `served` answered and whose page holds a record. A served read with no
/// record replays nothing, and its reply prefills as any other.
///
/// Blocking: each record's rows are a blob read.
pub fn replay_spans(
    store: &SharedSubstrate,
    served: &[Served],
    results: &[ToolResult],
    blocks: &[Range<usize>],
) -> Vec<TextSpan> {
    let mut spans = Vec::new();
    for (result, block) in results.iter().zip(blocks) {
        let Some(hit) = served.iter().find(|s| s.response == result.response) else {
            continue;
        };
        let key = slot_key(&hit.content_key, hit.page);
        match reply_record::rows(store, &key) {
            Ok(Some(rows)) => spans.push(TextSpan {
                bytes: block.clone(),
                kind: ArmKind::Replay(rows),
            }),
            Ok(None) => {}
            Err(e) => tracing::warn!(
                target: "zend::fast_path",
                record = %key,
                "a served read's record would not read — its reply prefills alone: {e:#}",
            ),
        }
    }
    spans
}
