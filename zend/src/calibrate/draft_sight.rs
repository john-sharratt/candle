//! What of its unit's reading a draft has in front of it.
//!
//! A draft writes a unit's questions from the reading its projection places
//! ahead of the request (`run::draft`). A draft that holds none of it is asked
//! for a file's questions knowing only the file's name, and answers with an
//! API the name suggests: `projection_assembler.rs` was asked about a
//! `ProjectionError::CycleDetected` and a `ProjectionConfig { include,
//! exclude, allow_cycles }`, none of which it holds, and never about the
//! `glue_islands` it is built around. Sealed, such questions vote for the file
//! on questions it does not answer and for none it does.

use candle_conversation::projection::TimelineId;
use candle_conversation::PromptDump;

/// Tokens of `dump` that are turns of `reading`.
pub(super) fn reading_held(dump: &PromptDump, reading: TimelineId) -> usize {
    dump.pieces
        .iter()
        .filter(|p| p.timeline == Some(reading.raw()))
        .map(|p| p.token_count)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::{PieceKind, PromptPiece};

    fn piece(kind: PieceKind, timeline: Option<u64>, tokens: usize) -> PromptPiece {
        PromptPiece {
            kind,
            name: String::new(),
            section_id: None,
            timeline,
            index: None,
            role: None,
            score: 0.0,
            qualified: false,
            token_count: tokens,
            token_ids: Vec::new(),
            text: String::new(),
        }
    }

    /// **Only the reading's own turns count**: the system prompt, the draft's
    /// own request and another timeline's turns hold none of it.
    #[test]
    fn only_the_readings_turns_are_held() {
        let reading = TimelineId::from_raw(7).unwrap();
        let dump = PromptDump {
            pieces: vec![
                piece(PieceKind::Section, None, 2_011),
                piece(PieceKind::Turn, Some(7), 3_614),
                piece(PieceKind::Turn, Some(7), 2_900),
                piece(PieceKind::Turn, Some(9), 400),
                piece(PieceKind::Turn, Some(11), 312),
            ],
            token_count: 9_237,
            token_ids: Vec::new(),
            text: String::new(),
        };
        assert_eq!(reading_held(&dump, reading), 6_514);
        assert_eq!(reading_held(&dump, TimelineId::from_raw(8).unwrap()), 0);
    }
}
