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
/// to its close or, when the budget ran out first, to where it stopped. A
/// block the prompt opened carries no `<think>` of its own and starts the
/// text. `None` when it did not think.
///
/// **Only the model's own tags bound it.** A markup piece may hold a tag; a
/// literal piece is content wherever it falls — the decoded words of the
/// block arrive as literal pieces between the markup tags, and a `<think>`
/// inside one is a quotation, never a tag.
pub fn thinking_of(written: &TurnText) -> Option<String> {
    // `before`: content ahead of any tag, the block itself when the prompt
    // opened it. `body`: content inside a block the model opened.
    let (mut before, mut body) = (String::new(), String::new());
    let (mut open, mut closed) = (false, false);
    for piece in written.pieces() {
        if closed {
            break;
        }
        if piece.literal {
            if open { &mut body } else { &mut before }.push_str(&piece.text);
            continue;
        }
        let mut rest = piece.text.as_str();
        loop {
            if open {
                match rest.find(THINK_CLOSE) {
                    Some(c) => {
                        body.push_str(&rest[..c]);
                        closed = true;
                    }
                    None => body.push_str(rest),
                }
                break;
            }
            let (o, c) = (rest.find(THINK_OPEN), rest.find(THINK_CLOSE));
            if let Some(o) = o.filter(|&o| c.is_none_or(|c| o < c)) {
                open = true;
                rest = &rest[o + THINK_OPEN.len()..];
            } else if let Some(c) = c {
                before.push_str(&rest[..c]);
                body = std::mem::take(&mut before);
                closed = true;
                break;
            } else {
                before.push_str(rest);
                break;
            }
        }
    }
    if !open && !closed {
        return None;
    }
    let body = body.trim();
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

    /// **The block's words arrive as literal pieces between the model's
    /// tags** — as a decoded turn hands them over — and are its content.
    #[test]
    fn literal_words_between_the_tags_are_the_thinking() {
        let t = TurnText::markup("<think>\n")
            .then_literal("The user is asking about the scan.")
            .then_markup("\n</think>\n\n")
            .then_literal("The scan.");
        assert_eq!(
            thinking_of(&t).as_deref(),
            Some("The user is asking about the scan.")
        );
        let cut = TurnText::markup("<think>\n").then_literal("First the arena, then");
        assert_eq!(thinking_of(&cut).as_deref(), Some("First the arena, then"));
    }

    /// A block the prompt opened starts the text and is read to its close.
    #[test]
    fn a_block_the_prompt_opened_is_read_from_the_start() {
        let t = TurnText::markup("It needs the scan.\n</think>\n\nThe scan.");
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
