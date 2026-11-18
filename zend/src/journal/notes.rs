//! A turn's notes: the conclusions its thinking reached, written down as the
//! work goes (`docs/zend_journal.md` §5.2).
//!
//! A note is a `journal_note` call in the reply, and its words are that call's
//! `text`. Nothing else holds them: the turn is their one source, so the
//! journal reads a note where the reply wrote it. In the parts a draft is shown
//! and the text an entry keeps, each note call reads as `Note: <text>` rather
//! than as the markup of a call, so the drafter writes its entry from the
//! conclusions and `journal_read` and `journal_grep` find them as notes.

use std::sync::OnceLock;

use regex::Regex;

use crate::tools::extract_tool_calls;

/// The tool a note is written with.
pub const JOURNAL_NOTE_TOOL: &str = "journal_note";

/// What a note reads as in the journal's parts.
const NOTE: &str = "Note: ";

/// Every tool-call block, whole.
fn call_blocks() -> &'static Regex {
    static BLOCKS: OnceLock<Regex> = OnceLock::new();
    BLOCKS.get_or_init(|| Regex::new(r"(?s)<tool_call>.*?</tool_call>").expect("static regex"))
}

/// `text` with each note call it makes read as `Note: <text>`, and every other
/// call left as it was written.
pub fn marked(text: &str) -> String {
    call_blocks()
        .replace_all(text, |block: &regex::Captures| {
            let whole = &block[0];
            match extract_tool_calls(whole).as_slice() {
                [call] if call.name == JOURNAL_NOTE_TOOL => {
                    match call.arguments.get("text").and_then(|t| t.as_str()) {
                        Some(note) => format!("{NOTE}{}", note.trim()),
                        None => whole.to_string(),
                    }
                }
                _ => whole.to_string(),
            }
        })
        .into_owned()
}

/// The notes `text` writes to the conversation's own journal, in order: each
/// `journal_note` call's `text` whose `journal` is left out or `"self"`. A note
/// to a memory journal is a memory write, not a step of the request
/// (`docs/request_steps.md` §2.1).
pub fn own_notes(text: &str) -> Vec<String> {
    call_blocks()
        .find_iter(text)
        .flat_map(|block| extract_tool_calls(block.as_str()))
        .filter(|call| call.name == JOURNAL_NOTE_TOOL)
        .filter(|call| {
            call.arguments
                .get("journal")
                .and_then(|j| j.as_str())
                .is_none_or(|j| j == "self")
        })
        .filter_map(|call| {
            call.arguments
                .get("text")
                .and_then(|t| t.as_str())
                .map(|t| t.trim().to_string())
        })
        .filter(|t| !t.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A reply's own notes are its steps, in order**: in both call
    /// syntaxes, `journal` left out or `"self"`. A note to a memory journal,
    /// another call and an empty note are none of them.
    #[test]
    fn a_replys_own_notes_are_read_in_order() {
        let text = "<tool_call>\n{\"name\": \"journal_note\", \"arguments\": {\"text\": \" First \
                    finding. \"}}\n</tool_call>\n<tool_call>\n{\"name\": \"journal_note\", \
                    \"arguments\": {\"text\": \"To memory.\", \"journal\": \"~/notes\"}}\n\
                    </tool_call>\n<tool_call>\n<function=journal_note>\n<parameter=text>\nSecond \
                    finding.\n</parameter>\n<parameter=journal>\nself\n</parameter>\n</function>\n\
                    </tool_call>\n<tool_call>\n{\"name\": \"file_read\", \"arguments\": {\"path\": \
                    \"a.rs\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"journal_note\", \
                    \"arguments\": {\"text\": \"  \"}}\n</tool_call>";
        assert_eq!(own_notes(text), ["First finding.", "Second finding."]);
        assert!(own_notes("no calls at all").is_empty());
    }

    /// **A note call reads as the note**, in both call syntaxes the models
    /// write, and every other call stays as written.
    #[test]
    fn a_note_call_reads_as_its_note_and_other_calls_stay() {
        let json = "<tool_call>\n{\"name\": \"journal_note\", \"arguments\": {\"text\": \"The \
                    stall is the scan's lock, not the GPU.\"}}\n</tool_call>\n<tool_call>\n\
                    {\"name\": \"file_read\", \"arguments\": {\"path\": \"scan.rs\"}}\n</tool_call>";
        assert_eq!(
            marked(json),
            "Note: The stall is the scan's lock, not the GPU.\n<tool_call>\n{\"name\": \
             \"file_read\", \"arguments\": {\"path\": \"scan.rs\"}}\n</tool_call>"
        );
        let block = "<tool_call>\n<function=journal_note>\n<parameter=text>\nscan.rs holds the \
                     lock across the await\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            marked(block),
            "Note: scan.rs holds the lock across the await"
        );
        let other = "Done. <tool_call>\n<function=file_list>\n<parameter=path>\nsrc\n</parameter>\
                     \n</function>\n</tool_call>";
        assert_eq!(marked(other), other);
    }
}
