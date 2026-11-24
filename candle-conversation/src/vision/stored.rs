//! The image spans of stored turns, for a path that runs them through the
//! model again.
//!
//! A sealed turn records `<|image_pad|>` for each image row and a marker per
//! image in its user text ([`super::marker`]). A path that prefills the turn a
//! second time — the recurrent catch-up over spliced turns — finds each
//! image's rows by its marker and hands them to [`super::engine_ids`] as spans.

use super::spans::{turn_spans, ImageSpan, VisionTokens};
use super::store::load_rows;
use crate::projection::{Conversation, TimelineId, TurnIndex};

/// The image spans of `turns` laid end to end in order, each turn's ids
/// whole: positions count from the first id of the first turn. Only a turn's
/// user half holds images. Empty for a model with no vision tokens.
pub fn stored_spans(
    substrate: &Conversation,
    tokens: Option<VisionTokens>,
    turns: &[(TimelineId, TurnIndex)],
) -> Result<Vec<ImageSpan>, String> {
    let Some(tokens) = tokens else {
        return Ok(Vec::new());
    };
    // The halves under the read lock, the rows after it: a blob is read
    // through the persistence path, which takes its own locks.
    let halves: Vec<(usize, Vec<u32>, String)> = {
        let view = substrate.read();
        let mut at = 0usize;
        let mut halves = Vec::with_capacity(turns.len());
        for &(timeline, index) in turns {
            if let Some(half) = view.user_half(timeline, index) {
                halves.push((at + half.offset, half.ids.to_vec(), half.text.to_string()));
            }
            // The turn's whole ids, as `token_ids_of` lays them end to end.
            at += view.assistant_token_ids_of(timeline, index).len();
        }
        halves
    };
    let shared = substrate.shared();
    let mut spans = Vec::new();
    for (at, ids, text) in halves {
        let found = turn_spans(&ids, &text, tokens, |id| load_rows(&shared, id))?;
        spans.extend(found.iter().map(|s| s.shifted(at)));
    }
    Ok(spans)
}
