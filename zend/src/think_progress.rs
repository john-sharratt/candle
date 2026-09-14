//! How far the reasoning of the turn being decoded has got, for the GUI.
//!
//! The thinking block shows its token count as it runs and its total once it
//! closes. Both are counted the way the turn records its reasoning at seal: the
//! generated tokens up to and including `</think>` — preamble and markers too —
//! so the total the stream reports is the length a reloaded history shows for
//! the same turn (`TurnLayout::thinking_length`).
//!
//! The turn's block is the first `<think>` it decodes; a later one is text. A
//! block that closes empty is dropped from the stream (the answer does not
//! open on a collapsed `<think></think>`), so it reports no total either — the
//! GUI would have no block to put it on.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// What the decode so far says about the turn's reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkUpdate {
    /// The block is open; this many tokens generated so far.
    Running(usize),
    /// The block closed after this many generated tokens.
    Done(usize),
}

#[derive(Debug, Default)]
pub struct ThinkProgress {
    /// Byte offset of the turn's `<think>` in the decoded text, once seen.
    open_at: Option<usize>,
    /// The block has closed; nothing more to report.
    closed: bool,
}

impl ThinkProgress {
    /// Read the decode so far — `text`, `generated` tokens long — and say what
    /// changed, if anything is worth reporting.
    pub fn observe(&mut self, text: &str, generated: usize) -> Option<ThinkUpdate> {
        if self.closed {
            return None;
        }
        let open = match self.open_at {
            Some(open) => open,
            None => {
                let open = text.find(OPEN)?;
                self.open_at = Some(open);
                open
            }
        };
        // The text is re-decoded from the whole run each token; a byte-fallback
        // sequence completing can reshape its tail, so read it defensively.
        let inner = text.get(open + OPEN.len()..)?;
        match inner.find(CLOSE) {
            None => Some(ThinkUpdate::Running(generated)),
            Some(end) => {
                self.closed = true;
                (!inner[..end].trim().is_empty()).then_some(ThinkUpdate::Done(generated))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_reported_before_the_block_opens() {
        let mut p = ThinkProgress::default();
        assert_eq!(p.observe("", 0), None);
        assert_eq!(p.observe("Sure", 1), None);
    }

    #[test]
    fn an_open_block_reports_its_count_and_a_closed_one_its_total() {
        let mut p = ThinkProgress::default();
        assert_eq!(p.observe("<think>", 1), Some(ThinkUpdate::Running(1)));
        assert_eq!(p.observe("<think>\nWhy", 3), Some(ThinkUpdate::Running(3)));
        assert_eq!(
            p.observe("<think>\nWhy not</think>", 5),
            Some(ThinkUpdate::Done(5))
        );
        // Closed: the answer that follows is not reasoning.
        assert_eq!(p.observe("<think>\nWhy not</think>\n\nBecause", 7), None);
    }

    #[test]
    fn a_preamble_before_the_block_counts_as_reasoning() {
        let mut p = ThinkProgress::default();
        assert_eq!(p.observe("Hmm", 1), None);
        assert_eq!(
            p.observe("Hmm<think>so</think>", 4),
            Some(ThinkUpdate::Done(4))
        );
    }

    #[test]
    fn a_block_that_closes_empty_reports_no_total() {
        let mut p = ThinkProgress::default();
        assert_eq!(p.observe("<think>\n", 2), Some(ThinkUpdate::Running(2)));
        assert_eq!(p.observe("<think>\n\n</think>", 3), None);
        assert_eq!(p.observe("<think>\n\n</think>Hi", 4), None);
    }

    #[test]
    fn a_second_block_is_text() {
        let mut p = ThinkProgress::default();
        assert_eq!(p.observe("<think>a</think>", 3), Some(ThinkUpdate::Done(3)));
        assert_eq!(p.observe("<think>a</think>b<think>c", 6), None);
    }
}
