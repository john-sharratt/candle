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
    /// How much of the text has been searched: a marker can only end in what
    /// came since, so each look searches from just before it.
    scanned: usize,
}

/// The longest marker, less one: how far back a marker straddling the
/// searched and the new text can start.
const OVERLAP: usize = CLOSE.len() - 1;

/// The first byte offset at or after `from` where `marker` starts in `text`.
fn find_from(text: &str, from: usize, marker: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let from = from.min(bytes.len());
    bytes[from..]
        .windows(marker.len())
        .position(|w| w == marker.as_bytes())
        .map(|i| from + i)
}

/// Where the answer starts in `text` — past the last `</think>` — given that
/// it started at `was` before the bytes from `from` on were added. Only the
/// new bytes, and the tail of the old a close may straddle, are searched, so
/// the last occurrence wins as a whole-text `rfind` would have it.
pub fn answer_start(text: &str, from: usize, was: usize) -> usize {
    let bytes = text.as_bytes();
    let start = from.saturating_sub(OVERLAP).min(bytes.len());
    bytes[start..]
        .windows(CLOSE.len())
        .rposition(|w| w == CLOSE.as_bytes())
        .map_or(was, |i| start + i + CLOSE.len())
}

impl ThinkProgress {
    /// Read the decode so far — `text`, `generated` tokens long, grown from
    /// what the last look read — and say what changed, if anything is worth
    /// reporting.
    pub fn observe(&mut self, text: &str, generated: usize) -> Option<ThinkUpdate> {
        if self.closed {
            return None;
        }
        let from = self.scanned.saturating_sub(OVERLAP);
        self.scanned = text.len();
        let open = match self.open_at {
            Some(open) => open,
            None => {
                let open = find_from(text, from, OPEN)?;
                self.open_at = Some(open);
                open
            }
        };
        let inner_at = open + OPEN.len();
        match find_from(text, from.max(inner_at), CLOSE) {
            None => Some(ThinkUpdate::Running(generated)),
            Some(end) => {
                self.closed = true;
                (!text[inner_at..end].trim().is_empty()).then_some(ThinkUpdate::Done(generated))
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

    /// **A marker split across two looks is found**: each look searches only
    /// what came since the last, and the tail a marker can straddle.
    #[test]
    fn a_marker_split_across_looks_is_found() {
        let mut p = ThinkProgress::default();
        assert_eq!(p.observe("<think>a</thi", 4), Some(ThinkUpdate::Running(4)));
        assert_eq!(p.observe("<think>a</think>", 5), Some(ThinkUpdate::Done(5)));
        let mut q = ThinkProgress::default();
        assert_eq!(q.observe("<thi", 1), None);
        assert_eq!(q.observe("<think>a", 2), Some(ThinkUpdate::Running(2)));
    }

    /// **The answer starts past the last close**, found in what was added —
    /// a close straddling the old text and the new included — and stays where
    /// it was when the new bytes hold none.
    #[test]
    fn the_answer_starts_past_the_last_close() {
        let text = "<think>a</think>b";
        assert_eq!(answer_start(text, 12, 0), 16, "a close cut by the step");
        assert_eq!(answer_start(text, 16, 16), 16, "nothing new closes");
        assert_eq!(answer_start("<think>a</think>b</think>c", 17, 16), 25);
        assert_eq!(answer_start("plain", 3, 0), 0);
    }

    #[test]
    fn a_second_block_is_text() {
        let mut p = ThinkProgress::default();
        assert_eq!(p.observe("<think>a</think>", 3), Some(ThinkUpdate::Done(3)));
        assert_eq!(p.observe("<think>a</think>b<think>c", 6), None);
    }
}
