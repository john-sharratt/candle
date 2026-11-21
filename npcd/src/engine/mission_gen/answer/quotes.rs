//! Checks that a quoted span appears in its source document, ignoring formatting differences.
//!
//! A quote is accepted when the document holds it under a comparison that ignores case, spacing, markdown emphasis, link markup and curly quotes, since a faithful copy can still differ in those. Markdown links are reduced to their words before comparison, so a quote of a linked phrase matches the plain text it was copied from. Spans in single quotes are gathered as candidates, and the caller keeps only the ones the document holds, because an apostrophe can open a span that is not a quote.

use super::*;

/// The shortest quote that counts as a quote.
pub(in super::super) const MIN_QUOTE_CHARS: usize = 20;

/// A quote from a document, checked to be in it.
pub(super) fn quoted(q: &str, text: &str, which: &str) -> Result<String, String> {
    let q = q.trim().trim_matches(['"', '“', '”', '\'']).trim();
    if q.chars().count() < MIN_QUOTE_CHARS {
        return Err(format!(
            "The quote from {which} is too short to be checked; quote the whole clashing \
             sentence."
        ));
    }
    match contains(text, q) {
        true => Ok(q.to_string()),
        false => Err(format!(
            "\"{q}\" is not in {which}. Quote its words exactly, or answer `{NOTHING}` if the two \
             do not actually contradict."
        )),
    }
}

/// Whether `text` holds `q`, ignoring case, spacing, markdown emphasis and
/// links, and curly quotes — the differences a faithful copy can still have.
///
/// **Links especially.** An era reads "mobile fortress-[towers](/tower)" on the
/// page and "mobile fortress-towers" to anybody quoting it; a story's quote was
/// refused three times for being faithful.
pub(in super::super) fn contains(text: &str, q: &str) -> bool {
    let norm = |s: &str| -> String {
        unlink(s)
            .chars()
            .filter(|c| !matches!(c, '*' | '_' | '`'))
            .map(|c| match c {
                '“' | '”' => '"',
                '‘' | '’' => '\'',
                '—' | '–' => '-',
                c => c.to_ascii_lowercase(),
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let q = norm(q);
    !q.is_empty() && norm(text).contains(&q)
}

/// Markdown links reduced to their words: `[towers](/tower)` → `towers`.
pub(in super::super) fn unlink(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let link = after.find("](").and_then(|mid| {
            let close = after[mid + 2..].find(')')?;
            Some((mid, mid + 2 + close + 1))
        });
        match link {
            Some((mid, end)) if !after[..mid].contains(['[', ']']) => {
                out.push_str(&rest[..open]);
                out.push_str(&after[..mid]);
                rest = &after[end..];
            }
            _ => {
                out.push_str(&rest[..=open]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Each sentence of a text, with its closing stop — a quote given without
/// quote marks is still a quote, and may be several sentences of which some
/// are the model's own.
pub(super) fn sentences_of(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in s.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            out.push(current.trim().trim_matches('"').trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().trim_matches('"').trim().to_string());
    }
    out
}

/// Every span between single quotes of at least `min` characters — how a
/// reading quoted, as often as not. An apostrophe opens a span too ("Verdi's
/// role as a '…'"), so some spans are noise; the caller keeps only those the
/// document holds.
pub(in super::super) fn single_quoted(s: &str, min: usize) -> Vec<String> {
    let s = s.replace(['‘', '’'], "'");
    s.split('\'')
        .map(str::trim)
        .filter(|q| q.chars().count() >= min)
        .map(str::to_string)
        .collect()
}

/// Every span between double quotes of at least `min` characters.
pub(in super::super) fn quotes_in(s: &str, min: usize) -> Vec<String> {
    let s = s.replace(['“', '”'], "\"");
    s.split('"')
        .skip(1)
        .step_by(2)
        .map(str::trim)
        .filter(|q| q.chars().count() >= min)
        .map(str::to_string)
        .collect()
}
