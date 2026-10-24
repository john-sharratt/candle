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
    /// The word an entry's number follows at its head: "Entry 3 [span]".
    pub entry: &'static str,
    /// What names the latest entry in a draft's question, before the step test:
    /// "Your latest entry:".
    pub latest: &'static str,
    /// How the span `from_ms ..= to_ms` is written where an entry is shown on
    /// its own — to an inspector.
    pub span: fn(from_ms: u64, to_ms: u64) -> String,
    /// The day `at_ms` falls on, as the journal marks a new day.
    pub day: fn(at_ms: u64) -> String,
    /// The hour `at_ms` falls in, as the journal marks a new hour.
    pub hour: fn(at_ms: u64) -> String,
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
    /// Opens what the world read, after a claim whose fact it read differently:
    /// `" (the workspace reads"` makes `… (the workspace reads file:a exists as
    /// true)`.
    pub corrected: &'static str,
    /// Appended to a claim that may have stopped being true.
    pub recheck: &'static str,
    /// First line of the section the open items are carried in.
    pub open_heading: &'static str,
    /// That section when nothing is open — said, not left out, so a reader
    /// asked what is open does not fill the silence in.
    pub nothing_open: &'static str,
    /// The journal's last line: where it ends, so what the reader remembers
    /// from elsewhere is not taken for something it wrote down.
    pub end: &'static str,
    /// How one turn's time is written beside it in the stretch a draft is shown.
    pub clock: fn(at_ms: u64) -> String,
    /// The line the stretch a draft is shown opens with.
    pub stretch: fn(from_ms: u64, to_ms: u64) -> String,
    /// What the writer is asked once the stretch is on the table: the call, and
    /// what each part of it is.
    pub write_question: &'static str,
    /// Follows `write_question` when the journal already holds an entry: that
    /// `journal_skip`, with a reason, is the answer when the stretch adds
    /// nothing to it.
    pub skip_question: &'static str,
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

    /// A "day" of 1000 ms.
    fn day(at_ms: u64) -> String {
        format!("d{}", at_ms / 1000)
    }

    /// An "hour" of 100 ms.
    fn hour(at_ms: u64) -> String {
        format!("h{}", at_ms / 100)
    }

    /// A style whose every string is short and distinct, so a test can tell
    /// which one a line was rendered under.
    pub const PLAIN: Style = Style {
        entry: "Entry",
        latest: "Latest:",
        span,
        day,
        hour,
        saw: "Saw:",
        heard: "Heard:",
        did: "Did:",
        inferred: "Concluded:",
        meaning_to: "Meaning to:",
        left_open: "Left open:",
        settled: "Settled:",
        corrected: " (reads",
        recheck: " (re-check)",
        open_heading: "Open:",
        nothing_open: "Open: nothing.",
        end: "End.",
        clock,
        stretch,
        write_question: "Write it up with journal_write.",
        skip_question: "Or skip it with journal_skip.",
    };
}
