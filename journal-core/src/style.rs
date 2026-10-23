//! How a journal reads to the one who wrote it: the headings its claims are
//! grouped under, the notes a line carries, and how a span of time is written.
//!
//! An entry is rendered once, when it is kept, because the section it becomes
//! is sealed and cannot be rendered again when it is gathered. So everything the
//! reader is told — when the stretch was, how each line came to be known —
//! is fixed in the text by the [`Style`] the journal was kept under. A character
//! dates its stretch on its world's calendar and reads "Heard:" for what a peer
//! said; a coding assistant dates it in UTC and reads "User said:".

/// The wording an entry renders with.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    /// How the span `from_ms ..= to_ms` is written at the head of an entry.
    pub span: fn(from_ms: u64, to_ms: u64) -> String,
    /// Heading of what was observed: a tool result, an event received.
    pub saw: &'static str,
    /// Heading of what somebody else said.
    pub heard: &'static str,
    /// Heading of the writer's own acts.
    pub did: &'static str,
    /// Heading of what was worked out and not seen or heard.
    pub inferred: &'static str,
    /// Heading of what the writer means to do next.
    pub meaning_to: &'static str,
    /// Heading of the items an entry opened.
    pub left_open: &'static str,
    /// Heading of the items an entry settled.
    pub settled: &'static str,
    /// Appended to a claim the check replaced with what the world read.
    pub corrected: &'static str,
    /// Appended to a claim that may have stopped being true.
    pub recheck: &'static str,
    /// First line of the section the open items are carried in.
    pub open_heading: &'static str,
    /// How one turn's time is written beside it in the stretch a draft is shown.
    pub clock: fn(at_ms: u64) -> String,
    /// The line the stretch a draft is shown opens with.
    pub stretch: fn(from_ms: u64, to_ms: u64) -> String,
    /// What the writer is asked once the stretch is on the table: the call, and
    /// what each part of it is.
    pub write_question: &'static str,
}

#[cfg(test)]
pub(crate) mod testing {
    use super::Style;

    fn span(from_ms: u64, to_ms: u64) -> String {
        format!("[{from_ms}–{to_ms}]")
    }

    fn clock(at_ms: u64) -> String {
        format!("t{at_ms}")
    }

    fn stretch(from_ms: u64, to_ms: u64) -> String {
        format!("From {from_ms} to {to_ms}:")
    }

    /// A style whose every string is short and distinct, so a test can tell
    /// which one a line was rendered under.
    pub const PLAIN: Style = Style {
        span,
        saw: "Saw:",
        heard: "Heard:",
        did: "Did:",
        inferred: "Concluded:",
        meaning_to: "Meaning to:",
        left_open: "Left open:",
        settled: "Settled:",
        corrected: " (corrected)",
        recheck: " (re-check)",
        open_heading: "Open:",
        clock,
        stretch,
        write_question: "Write it up with journal_write.",
    };
}
