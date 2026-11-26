//! What of its file a draft has in front of it.
//!
//! A draft writes a file's questions from the file its knowledge base holds
//! (`run::draft`, pinned to the file alone). A draft that holds none of it is
//! asked for a file's questions knowing only the file's name, and answers with
//! an API the name suggests: `projection_assembler.rs` was asked about a
//! `ProjectionError::CycleDetected` and a `ProjectionConfig { include,
//! exclude, allow_cycles }`, none of which it holds, and never about the
//! `glue_islands` it is built around. Sealed, such questions vote for the file
//! on questions it does not answer and for none it does.

use std::collections::HashSet;

use candle_conversation::PromptDump;

/// Tokens of `dump` that are the file's members — the sections named in
/// `members`, by raw id.
pub(super) fn members_held(dump: &PromptDump, members: &HashSet<u32>) -> usize {
    dump.pieces
        .iter()
        .filter(|p| p.section_id.is_some_and(|id| members.contains(&id)))
        .map(|p| p.token_count)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::{PieceKind, PromptPiece};

    fn piece(kind: PieceKind, section_id: Option<u32>, tokens: usize) -> PromptPiece {
        PromptPiece {
            kind,
            name: String::new(),
            section_id,
            timeline: None,
            index: None,
            role: None,
            score: 0.0,
            qualified: false,
            token_count: tokens,
            token_ids: Vec::new(),
            text: String::new(),
        }
    }

    /// **Only the file's own members count**: the system prompt, another
    /// section and the draft's own request hold none of it.
    #[test]
    fn only_the_files_members_are_held() {
        let dump = PromptDump {
            pieces: vec![
                piece(PieceKind::Section, Some(1), 2_011),
                piece(PieceKind::Section, Some(501), 12),
                piece(PieceKind::Section, Some(502), 2_900),
                piece(PieceKind::Section, Some(600), 400),
                piece(PieceKind::Turn, None, 312),
            ],
            token_count: 5_635,
            token_ids: Vec::new(),
            text: String::new(),
        };
        assert_eq!(members_held(&dump, &HashSet::from([501, 502])), 2_912);
        assert_eq!(members_held(&dump, &HashSet::from([700])), 0);
    }
}
