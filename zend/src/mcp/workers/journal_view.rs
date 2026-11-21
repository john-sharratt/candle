//! Reads a journal view and a final reply: the newest entry, the entries after a
//! point, the reply of a turn and its question.
//!
//! A final reply is what follows the first closing tag of its reasoning, so a
//! reply that quotes that tag keeps everything after its own thinking. A question
//! is the block a reply ends with, from its marker line on, and there is none when
//! the reply asks nothing. An entry's summary is its `Entry N` line, skipping the
//! date the journal puts above an entry that starts a new hour.

use super::*;

/// The newest entry id `view` — a journal as `GET …/journal` answers it —
/// holds.
pub fn newest_entry(view: Option<&Value>) -> Option<u64> {
    view?
        .get("entries")?
        .as_array()?
        .iter()
        .filter_map(|e| e.get("id").and_then(Value::as_u64))
        .max()
}

/// The entries `view` holds after `after`, in order.
pub fn entries_after(view: Option<&Value>, after: Option<u64>) -> Vec<Entry> {
    let Some(entries) = view
        .and_then(|v| v.get("entries"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|e| {
            let id = e.get("id")?.as_u64()?;
            let text = e.get("text")?.as_str()?.to_string();
            after.is_none_or(|a| id > a).then_some(Entry { id, text })
        })
        .collect()
}

/// Whether `view` says a draft is running.
pub fn drafting(view: Option<&Value>) -> bool {
    view.and_then(|v| v.get("drafting"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The reply of a final round, whole: what follows the `</think>` that closes
/// its reasoning — the first. A reply may quote `</think>` itself, as a review
/// of the code that closes a turn does, and the reply runs on past the quote.
pub fn final_reply(text: &str) -> String {
    text.split_once("</think>")
        .map_or(text, |(_, after)| after)
        .trim()
        .to_string()
}

/// The question block a reply ends with, from its [`QUESTION_MARK`] line on;
/// `None` when it asks nothing.
pub fn question_of(reply: &str) -> Option<String> {
    let at = reply
        .rmatch_indices(QUESTION_MARK)
        .map(|(i, _)| i)
        .find(|&i| i == 0 || reply[..i].ends_with('\n'))?;
    let block = reply[at..].trim();
    (block.len() > QUESTION_MARK.len()).then(|| block.to_string())
}

/// The first line of `text`, at most `max` characters.
pub fn first_line(text: &str, max: usize) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut out: String = line.chars().take(max).collect();
    if line.chars().count() > max {
        out.push('…');
    }
    out
}

/// An entry's summary: its `Entry N — …` line, which follows the date the
/// journal puts above an entry that starts a new hour; the first line when it
/// has none.
pub fn summary_line(entry: &str, max: usize) -> String {
    match entry.lines().find(|l| l.trim_start().starts_with("Entry ")) {
        Some(line) => first_line(line, max),
        None => first_line(entry, max),
    }
}
