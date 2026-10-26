//! A question's thinking: what Flash-Next reasons when it is asked the
//! question alone, kept as the opening of the question's calibration answer.
//!
//! A live projection probes the trailing window of what the model is
//! decoding, and for most of a turn that is its own thinking. A question
//! exemplar alone resembles only the turn's user half. Asked under the system
//! prompt every conversation opens with, nothing projected yet, the model's
//! thinking about the question is what a probe at the start of a turn holds
//! while it decides what to pull in.

use candle_conversation::TurnText;

/// The reasoning block's tags, as the model writes them.
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// The thinking `written` opened with: the text of its first reasoning block,
/// to its close or, when the budget ran out first, to where it stopped.
/// `None` when it did not think. Only the model's own tags bound it — a
/// literal piece is content, never a tag.
pub fn thinking_of(written: &TurnText) -> Option<String> {
    let text: String = written
        .pieces()
        .iter()
        .filter(|p| !p.literal)
        .map(|p| p.text.as_str())
        .collect();
    let opened = &text[text.find(THINK_OPEN)? + THINK_OPEN.len()..];
    let body = opened
        .find(THINK_CLOSE)
        .map_or(opened, |c| &opened[..c])
        .trim();
    (!body.is_empty()).then(|| body.to_string())
}

/// The calibration answer `thinking` stands in: a closed reasoning block, the
/// tokens a live turn lays down when it thinks the same thing.
pub fn exemplar(thinking: &str) -> String {
    format!("{THINK_OPEN}\n{thinking}\n{THINK_CLOSE}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed block is read to its close; the answer after it is not
    /// thinking.
    #[test]
    fn a_closed_block_is_read_to_its_close() {
        let t = TurnText::markup("<think>\nIt needs the scan.\n</think>\n\nThe scan.");
        assert_eq!(thinking_of(&t).as_deref(), Some("It needs the scan."));
    }

    /// A block the budget cut off is read to where it stopped.
    #[test]
    fn a_block_cut_off_is_read_to_where_it_stopped() {
        let t = TurnText::markup("<think>\nFirst I look at the arena, then");
        assert_eq!(
            thinking_of(&t).as_deref(),
            Some("First I look at the arena, then")
        );
    }

    /// No block, an empty one, or tags inside literal text: no thinking.
    #[test]
    fn no_block_an_empty_one_or_a_quoted_tag_is_no_thinking() {
        assert_eq!(thinking_of(&TurnText::markup("The answer.")), None);
        assert_eq!(
            thinking_of(&TurnText::markup("<think>\n\n</think>Hi")),
            None
        );
        assert_eq!(
            thinking_of(&TurnText::literal("<think>quoted</think>")),
            None
        );
    }

    #[test]
    fn an_exemplar_is_a_closed_block() {
        assert_eq!(exemplar("It reads."), "<think>\nIt reads.\n</think>");
    }
}
