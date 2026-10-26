//! A file carved into units and anchored.

use crate::anchor;
use crate::carve::{self, types::ChunkKind};
use crate::Language;

/// One unit of a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// Its anchor — unique in the document.
    pub anchor: String,
    /// The scope path a code unit is shown under (`impl KvCache > fn
    /// seal_chunk`); `None` for a unit named by its own first line — a
    /// paragraph, a YAML line, a run of lines no item claims.
    pub label: Option<String>,
    pub kind: ChunkKind,
    /// Its first line, from 0.
    pub start: usize,
    /// One past its last line.
    pub end: usize,
}

/// A document's text, carved and anchored.
#[derive(Debug, Clone)]
pub struct Doc {
    language: Language,
    /// Each line with its own terminator; the last may have none.
    lines: Vec<String>,
    units: Vec<Unit>,
}

impl Doc {
    /// `text`, carved as the language `path` names.
    pub fn new(path: &str, text: &str) -> Doc {
        Doc::of(Language::of_path(path), text)
    }

    /// `text`, carved as `language`.
    pub fn of(language: Language, text: &str) -> Doc {
        let lines: Vec<String> = text.split_inclusive('\n').map(str::to_string).collect();
        let scopes = carve::carve(text, language);
        let texts: Vec<String> = scopes
            .iter()
            .map(|s| join(&lines[s.start_line as usize - 1..s.end_line as usize]))
            .collect();
        let anchors = anchor::assign(texts.iter().map(String::as_str));
        let units = scopes
            .into_iter()
            .zip(anchors)
            .map(|(s, anchor)| Unit {
                anchor,
                label: (!s.path.is_empty()).then(|| s.qualified_path()),
                kind: s.kind,
                start: s.start_line as usize - 1,
                end: s.end_line as usize,
            })
            .collect();
        Doc {
            language,
            lines,
            units,
        }
    }

    pub fn language(&self) -> Language {
        self.language
    }

    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    /// The document's lines, each with its own terminator.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// The whole text.
    pub fn text(&self) -> String {
        self.lines.concat()
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Unit `i`'s text: its lines, without the last one's terminator.
    pub fn unit_text(&self, i: usize) -> String {
        let u = &self.units[i];
        join(&self.lines[u.start..u.end])
    }

    /// The unit holding `anchor`.
    pub fn find(&self, anchor: &str) -> Option<usize> {
        self.units.iter().position(|u| u.anchor == anchor)
    }

    /// The unit line `line` (from 0) is in, if it is in one.
    pub fn unit_at(&self, line: usize) -> Option<usize> {
        self.units
            .iter()
            .position(|u| u.start <= line && line < u.end)
    }

    /// Every anchor, in document order.
    pub fn anchors(&self) -> Vec<String> {
        self.units.iter().map(|u| u.anchor.clone()).collect()
    }

    /// Whether the text parses as its language without a syntax error —
    /// `None` for a language with no grammar.
    pub fn parses(&self) -> Option<bool> {
        carve::parsers::parses(self.text().as_bytes(), self.language)
    }
}

/// Lines joined, without the last one's terminator.
fn join(lines: &[String]) -> String {
    let mut s = lines.concat();
    if s.ends_with('\n') {
        s.pop();
        if s.ends_with('\r') {
            s.pop();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_story_is_anchored_by_paragraph() {
        let text = "The bay was warm.\nThe tank cycled.\n\nYou said fourteen.\n";
        let doc = Doc::new("layers/life/bram/The Body.md", text);
        assert_eq!(doc.units().len(), 2);
        assert_eq!(doc.unit_text(0), "The bay was warm.\nThe tank cycled.");
        assert_eq!(doc.unit_text(1), "You said fourteen.");
        assert_eq!(
            doc.units()[1].anchor,
            anchor::anchor(0, "You said fourteen.")
        );
        assert_eq!(doc.units()[1].label, None);
        assert_eq!(doc.find(&doc.units()[1].anchor), Some(1));
        assert_eq!(doc.text(), text);
    }

    #[test]
    fn code_units_carry_their_scope_path() {
        let doc = Doc::new("a.rs", "struct Foo;\nimpl Foo {\n    fn bar(&self) {}\n}\n");
        let labels: Vec<Option<&str>> = doc.units().iter().map(|u| u.label.as_deref()).collect();
        assert_eq!(
            labels,
            vec![Some("struct Foo"), None, Some("impl Foo > fn bar"), None]
        );
        assert_eq!(doc.unit_text(2), "    fn bar(&self) {}");
    }

    /// The anchor is the unit's text alone: the same function under another
    /// path keeps its anchor.
    #[test]
    fn the_label_is_not_hashed() {
        let a = Doc::new("a.rs", "impl A {\n    fn go() {}\n}\n");
        let b = Doc::new("a.rs", "impl B {\n    fn go() {}\n}\n");
        let go = |d: &Doc| {
            d.units()
                .iter()
                .find(|u| u.label.as_deref().is_some_and(|l| l.ends_with("fn go")))
                .unwrap()
                .anchor
                .clone()
        };
        assert_eq!(go(&a), go(&b));
    }

    #[test]
    fn crlf_text_round_trips_and_anchors_as_lf() {
        let crlf = Doc::new("a.md", "One.\r\n\r\nTwo.\r\n");
        let lf = Doc::new("a.md", "One.\n\nTwo.\n");
        assert_eq!(crlf.text(), "One.\r\n\r\nTwo.\r\n");
        assert_eq!(crlf.anchors(), lf.anchors());
        assert_eq!(crlf.unit_text(0), "One.");
    }

    #[test]
    fn unit_at_names_the_unit_a_line_is_in() {
        let doc = Doc::new("a.md", "One.\n\nTwo.\nStill two.\n");
        assert_eq!(doc.unit_at(0), Some(0));
        assert_eq!(doc.unit_at(1), None);
        assert_eq!(doc.unit_at(3), Some(1));
    }
}
