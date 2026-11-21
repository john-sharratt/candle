//! Reads a model's story decode back into prose, a cast and an outline, removing
//! reasoning and markup on the way.
//!
//! A thinking model's reasoning block and any code fence are stripped, because
//! the prose is written into the character's memory and read back to them as
//! their own. A heading is recognised however the model dressed it, as `## Cast`,
//! as the same text in inline code, or as a bare label, and the prose ends at the
//! first heading of any kind. Entity ids are normalised to lowercase hyphenated
//! slugs, since an id is the join key between a relationship and a character.

/// Drop a leading reasoning block.
///
/// **A thinking model's reasoning is not the document.** Qwen3.5 opens with `<think>…</think>`
/// and the analysis inside it is about the *task* — "Total = 2792 - 2461 + 1 = 332. Correct." —
/// which is exactly the sort of thing that must never reach a character's memory. Stored
/// verbatim it would be ingested, gathered, and eventually read back to the character as
/// something they remember thinking.
///
/// Tolerant of the block being absent, unterminated, or preceded by whitespace, because all
/// three occur and none of them is a reason to lose the prose.
pub(super) fn strip_reasoning(raw: &str) -> &str {
    let t = raw.trim_start();
    let Some(rest) = t.strip_prefix("<think>") else {
        return raw;
    };
    match rest.find("</think>") {
        Some(end) => rest[end + "</think>".len()..].trim_start(),
        // Opened and never closed: the whole decode was reasoning and there is no document in
        // it. Better an empty answer the caller reports than a page of the model's notes.
        None => "",
    }
}

/// Drop the code-fence lines a model wraps a long answer in.
///
/// **A fence is markup, not diary prose.** Asked for a page of prose followed by
/// two `##` sections, a model reasonably decides the whole thing is a document
/// and wraps it in ```` ``` ````. Nothing downstream minds — the headings still
/// parse, the outline still lands — except that the fence markers themselves sit
/// in the prose, and the prose is what gets written into
/// `layers/memory/<who>/life-story.md` and read back to that character as their
/// own memory. A life that opens on a row of backticks is a life with markup in
/// it.
///
/// Every line that is *only* a fence goes, wherever it sits, rather than just a
/// matched opening and closing pair: an answer that closes its fence and then
/// opens an empty one has occurred, and a line of backticks is never a sentence
/// somebody wrote about their life, so there is nothing to be careful of.
pub(super) fn unfence(raw: &str) -> String {
    raw.lines()
        .filter(|l| !l.trim_start().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every heading the ladder asks any rung for.
///
/// Named as a set so a *bare* label — a line that is just `Cast`, with no `##`
/// in front of it — can be recognised as the heading it plainly is. A line of
/// diary prose is never one of these words and nothing else.
pub(super) const SECTIONS: [&str; 3] = ["Cast", "Years", "Days"];

/// The heading a line names, if it names one.
///
/// # Three spellings, one heading
///
/// Asked for `## Cast`, models have returned all of these, and each one used to
/// lose the entire structured tail — read as prose, pasted onto the end of the
/// character's own life story, with no cast and no outline for the rungs below:
///
/// - `## Cast` — what was asked for.
/// - `` `## Cast` `` — the shape was shown to it as inline code, so it answered
///   in inline code. Fixed at the source too, but the decoration is not
///   information and there is no reason to be strict about it.
/// - `Cast` — no marker at all, the heading as a plain label. One character's
///   decode carried a complete cast and a six-year outline this way and parsed
///   to nothing.
pub(super) fn heading_name(line: &str) -> Option<String> {
    let t = line.trim().trim_matches('`').trim();
    if let Some(rest) = t.strip_prefix("##") {
        return Some(rest.trim().to_string());
    }
    SECTIONS
        .iter()
        .find(|s| t.eq_ignore_ascii_case(s))
        .map(|s| s.to_string())
}

/// Split a decode into the text before a heading and the lines under it.
///
/// Case-insensitive on the heading and tolerant of leading whitespace, because
/// a model that writes `##Cast` or indents the block has still answered the
/// question. Everything after the *next* heading belongs to that one.
pub(super) fn section<'a>(raw: &'a str, heading: &str) -> (String, Vec<&'a str>) {
    let mut before = String::new();
    let mut lines = Vec::new();
    let mut inside = false;
    // **The prose ends at the first heading, whichever heading that is.** The
    // test used to be "have we collected any of our own lines yet", which is only
    // the same question when the wanted heading comes first. A decode that writes
    // `## Outline` before `## Cast` — which the story phase asks for in that order
    // — had the whole outline block read as prose, so the arc a year expands
    // arrived with a table of year titles pasted onto the end of it.
    let mut heading_seen = false;
    for line in raw.lines() {
        // A heading is a heading however the model dressed it — see
        // [`heading_name`], which is where the three spellings are recorded.
        let t = line.trim().trim_matches('`').trim();
        if let Some(name) = heading_name(line) {
            inside = name.eq_ignore_ascii_case(heading);
            heading_seen = true;
            continue;
        }
        if inside {
            if !t.is_empty() {
                lines.push(t);
            }
        } else if !heading_seen {
            before.push_str(line);
            before.push('\n');
        }
        // Otherwise: under some other heading. Neither prose nor ours.
    }
    (before.trim().to_string(), lines)
}

/// One `- a | b | c` row, split on pipes with the leading bullet removed.
pub(super) fn row(line: &str) -> Vec<String> {
    line.trim_start_matches(['-', '*', '•'])
        .trim()
        .split('|')
        .map(|s| s.trim().to_string())
        .collect()
}

/// An entity id as the rest of the life will spell it.
///
/// Normalised here rather than trusted, because it is the join key between a
/// relationship and an NPC, and a stray capital or space in one document is a
/// second person who never appears again.
pub fn slug(raw: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in raw.trim().chars() {
        if c.is_ascii_alphanumeric() {
            if dash && !out.is_empty() {
                out.push('-');
            }
            dash = false;
            out.extend(c.to_lowercase());
        } else {
            dash = true;
        }
    }
    out
}
