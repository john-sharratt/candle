//! Whether a committed reading holds its file.
//!
//! A reading's chain can stop part-way ([`chain_break`]), and it can finish
//! without ever having read the file: its opening reads named some other path,
//! came back `not_found`, and what it answered is about the search. On
//! 2026-11-11 about 115 readings went that way — `candle-conversation/src/
//! turn_layout.rs` read as `src/turn_layout.rs` nine times over — and each,
//! committed as finished, had superseded the reading that did read its file.
//! Both are broken: the pass reads the file again, and the fast path never
//! hands one to a dialogue as a file it has read.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, PoisonError};

use candle_conversation::chain_health::{chain_break, ChainBreak};
use candle_conversation::projection::{TimelineId, TurnIndex};
use candle_conversation::substrate::Substrate;

/// Why a committed reading is no reading of its file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Broken {
    /// Its chain stopped part-way ([`chain_break`]).
    Chain(ChainBreak),
    /// Its opening reads came back `not_found` ([`missed_read`]).
    MissedRead,
}

/// The turn a reading's opening reads come back in: the opening turn makes
/// the calls, and the next one carries their results.
const OPENING_RESULTS: TurnIndex = TurnIndex(1);

/// Each committed reading's [`missed_read`] verdict, once its opening results
/// have landed. A committed reading's turns never change, and its results are
/// the file's whole content, so the verdict is read from them once rather
/// than on every pass, link and hand-back that asks.
static MISSED: OnceLock<Mutex<HashMap<TimelineId, bool>>> = OnceLock::new();

/// Why `timeline`, a committed reading, does not hold its file — `None` when
/// it does: its chain finished and its opening reads found the file.
pub(super) fn reading_break(substrate: &Substrate, timeline: TimelineId) -> Option<Broken> {
    if let Some(why) = chain_break(substrate, timeline) {
        return Some(Broken::Chain(why));
    }
    missed(substrate, timeline).then_some(Broken::MissedRead)
}

/// [`missed_read`] of `timeline`'s opening results, from [`MISSED`] once known.
/// A reading whose results have not landed yet missed nothing so far, and is
/// asked again.
fn missed(substrate: &Substrate, timeline: TimelineId) -> bool {
    let cache = MISSED.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&known) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&timeline)
    {
        return known;
    }
    let Some(verdict) = substrate.with_user_text(timeline, OPENING_RESULTS, missed_read) else {
        return false;
    };
    cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(timeline, verdict);
    verdict
}

/// Whether a tool round's results, as the reading was shown them, hold a
/// `file_read` that found no file: a `<tool_response>` opening a line whose
/// body — after its `[i/n file_read]` label when the round made several
/// calls — is the `not_found` error. A file's content inside a result is in
/// parts whose every line is numbered (`+1 …`), so a file that merely quotes
/// such a response cannot look like one.
pub(super) fn missed_read(results: &str) -> bool {
    const NOT_FOUND: &str = r#"{"error":"not_found""#;
    let mut lines = results.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.strip_prefix("<tool_response>") else {
            continue;
        };
        let body = match rest.starts_with('[') && rest.ends_with(" file_read]") {
            true => lines.next().unwrap_or(""),
            false => rest,
        };
        if body.starts_with(NOT_FOUND) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A committed reading whose reads found no file is broken**, whether
    /// the round made one call or several — the shape 2026-11-11's readings
    /// stored — and one that read its file, or whose file merely quotes such
    /// a response, is not.
    #[test]
    fn a_reading_whose_reads_found_nothing_is_broken() {
        let several = "<tool_response>[1/9 file_read]\n\
            {\"error\":\"not_found\",\"detail\":\"file not found: src/turn_layout.rs\"}</tool_response>\n\
            <tool_response>[2/9 file_read]\n\
            {\"error\":\"not_found\",\"detail\":\"file not found: src/turn_layout.rs\"}</tool_response>";
        assert!(missed_read(several));
        let one = "<tool_response>{\"error\":\"not_found\",\"detail\":\"file not found: a.rs\"}</tool_response>";
        assert!(missed_read(one));
        let read = "<tool_response>```rust file=candle/a.rs part=p1 lines=1-2\n +1 fn a() {}\n +2 \n```\n</tool_response>";
        assert!(!missed_read(read));
        let quoted = "<tool_response>```rust file=candle/t.rs part=p1 lines=1\n \
            +1 let s = \"<tool_response>{\\\"error\\\":\\\"not_found\\\"\";\n```\n</tool_response>";
        assert!(
            !missed_read(quoted),
            "a file quoting a failed read is no failed read"
        );
        let other_error =
            "<tool_response>{\"error\":\"too_large\",\"detail\":\"x\"}</tool_response>";
        assert!(!missed_read(other_error));
    }
}
