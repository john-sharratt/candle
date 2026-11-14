//! What a walk has written of the call it is in: the tool, as its name branch
//! spelled it, and each field's value — written in a span, chosen by an enum's
//! branch, or fixed by the tool's schema — what a later field's held values
//! depend on (`docs/stencil_tree.md` §9.2).

use std::collections::HashMap;
use std::sync::Arc;

use super::element_close::PARAM_CLOSE;
use super::vocab::TokenId;

#[derive(Debug, Clone, Default)]
pub struct CallSoFar {
    tool: Option<Arc<str>>,
    /// The tokens the arm the walk is taking has taken, in a branch that names
    /// a call's tool or chooses an argument's value.
    arm: Vec<TokenId>,
    written: HashMap<Arc<str>, String>,
    /// The keyed span being written, how its value is written, and its bytes
    /// so far.
    open: Option<(Arc<str>, ValueText, Vec<u8>)>,
}

/// How a span writes its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueText {
    /// A JSON string, quoted and escaped.
    Json,
    /// A function block's raw value, ended by `</parameter>`.
    Raw,
}

impl CallSoFar {
    /// The call's tool, as its name branch chose it; empty before one has.
    pub fn tool(&self) -> &str {
        self.tool.as_deref().unwrap_or("")
    }

    /// Each field the call has written, by field.
    pub fn written(&self) -> &HashMap<Arc<str>, String> {
        &self.written
    }

    /// A branch naming a tool or choosing a value begins.
    pub fn begin_arm(&mut self) {
        self.arm.clear();
    }

    /// `token` of the arm the walk is taking.
    pub fn take(&mut self, token: TokenId) {
        self.arm.push(token);
    }

    /// The tokens the arm has taken.
    pub fn arm(&self) -> &[TokenId] {
        &self.arm
    }

    /// The name branch completed on `tool`: a new call, nothing of it written
    /// yet but the values its schema fixes, `fixed`.
    pub fn named(&mut self, tool: Option<Arc<str>>, fixed: &[(Arc<str>, Arc<str>)]) {
        self.tool = tool;
        self.arm.clear();
        self.written.clear();
        self.open = None;
        for (field, value) in fixed {
            self.written.insert(Arc::clone(field), value.to_string());
        }
    }

    /// A branch chose `value` for `field`.
    pub fn chose(&mut self, field: &Arc<str>, value: &str) {
        self.arm.clear();
        self.written.insert(Arc::clone(field), value.to_string());
    }

    /// A span writing `field`, as `text`, begins.
    pub fn open(&mut self, field: Arc<str>, text: ValueText) {
        self.open = Some((field, text, Vec::new()));
    }

    /// `bytes` of the span being written.
    pub fn feed(&mut self, bytes: &[u8]) {
        if let Some((_, _, value)) = &mut self.open {
            value.extend_from_slice(bytes);
        }
    }

    /// The span being written closed: its value is the call's.
    pub fn close(&mut self) {
        if let Some((field, text, bytes)) = self.open.take() {
            let value = match text {
                ValueText::Json => value_of(&bytes),
                ValueText::Raw => raw_value_of(&bytes),
            };
            self.written.insert(field, value);
        }
    }
}

/// A string value's content from the bytes its span took: past the space and
/// quote it may open with, up to its closing quote.
fn value_of(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let text = text.trim_start();
    let text = text.strip_prefix('"').unwrap_or(text);
    let mut out = String::new();
    let mut escaped = false;
    for c in text.chars() {
        match (escaped, c) {
            (false, '\\') => escaped = true,
            (false, '"') => break,
            _ => {
                escaped = false;
                out.push(c);
            }
        }
    }
    out
}

/// A raw value's content from the bytes its span took: up to its
/// `</parameter>`, the one framing line break before it taken off, as every
/// reader of a block takes it (`element_close`).
fn raw_value_of(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let value = text.split(PARAM_CLOSE).next().unwrap_or_default();
    let value = value.strip_suffix('\n').unwrap_or(value);
    value.strip_suffix('\r').unwrap_or(value).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Naming a call starts it afresh**: its fields are recorded as written,
    /// and the next call's name clears them.
    #[test]
    fn naming_a_call_starts_it_afresh() {
        let mut call = CallSoFar::default();
        call.begin_arm();
        call.take(7);
        call.take(9);
        assert_eq!(call.arm(), [7, 9]);
        call.named(Some(Arc::from("file_edit")), &[]);
        assert_eq!(call.tool(), "file_edit");
        call.open(Arc::from("part"), ValueText::Json);
        call.feed(b" \"Keh");
        call.feed(b"T\",");
        call.close();
        assert_eq!(call.written()[&Arc::from("part")], "KehT");
        call.begin_arm();
        call.named(Some(Arc::from("file_read")), &[]);
        assert_eq!(call.tool(), "file_read");
        assert!(call.written().is_empty());
    }

    /// **A value a branch chose, or the schema fixed, is written as a span's
    /// is**: a fixed value from the call's naming, a chosen one as its arm
    /// completes.
    #[test]
    fn chosen_and_fixed_values_are_written() {
        let mut call = CallSoFar::default();
        let repo: Arc<str> = Arc::from("repo");
        let page: Arc<str> = Arc::from("kind");
        call.named(
            Some(Arc::from("file_list")),
            &[(Arc::clone(&page), Arc::from("dir"))],
        );
        assert_eq!(call.written()[&page], "dir");
        call.begin_arm();
        call.take(4);
        call.chose(&repo, "candle");
        assert!(call.arm().is_empty());
        assert_eq!(call.written()[&repo], "candle");
        assert_eq!(call.written()[&page], "dir");
    }

    /// **A value is its content**: past an opening space and quote, up to the
    /// closing quote, an escaped quote kept — exactly these strings.
    #[test]
    fn a_value_is_read_between_its_quotes() {
        assert_eq!(value_of(b" \"+37\""), "+37");
        assert_eq!(value_of(b"src/a.rs\"}"), "src/a.rs");
        assert_eq!(value_of(br#" "a\"b""#), "a\"b");
        assert_eq!(value_of(b"+3"), "+3");
    }

    /// **A raw value is its content up to its `</parameter>`**, the closing
    /// tags after it and one framing line break before it left out.
    #[test]
    fn a_raw_value_is_read_up_to_its_close() {
        assert_eq!(
            raw_value_of(b"candle-conversation/src/cancel.rs</parameter>\n<parameter="),
            "candle-conversation/src/cancel.rs"
        );
        assert_eq!(raw_value_of(b"1-120\n</parameter>\n</function>"), "1-120");
        assert_eq!(raw_value_of(b"a \"b\"\r\n</parameter>"), "a \"b\"");
        assert_eq!(raw_value_of(b"src/a"), "src/a");
        let mut call = CallSoFar::default();
        call.named(Some(Arc::from("file_read")), &[]);
        call.open(Arc::from("path"), ValueText::Raw);
        call.feed(b"src/a.rs</para");
        call.feed(b"meter>\n<parameter=");
        call.close();
        assert_eq!(call.written()[&Arc::from("path")], "src/a.rs");
    }
}
