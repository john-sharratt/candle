//! Where a function block's raw value ends — the one rule the grammar that
//! writes the block and every reader of it share.
//!
//! Qwen3.5/3.8 write a call as XML-shaped elements whose values are raw:
//!
//! ```text
//! <function=write>
//! <parameter=path>
//! a.js</parameter>
//! <parameter=content>
//! …the file, unescaped…</parameter>
//! </function>
//! ```
//!
//! A value has no escaping, so the text `</parameter>` alone cannot end it: a
//! file about the call format holds that text, and so do its tests. Cut there,
//! every write of such a file failed (`docs/zen_coding_session_fixes.md`,
//! Cause 1a). **A value ends at a `</parameter>` that structure follows** — a
//! separator from [`SEPARATORS`], then the next element's tag, `<parameter=` or
//! `</function>`. Any other `</parameter>` is content.
//!
//! The grammar closes a value's span on exactly these strings
//! ([`Terminator::Until`]'s tails, [`tails`]), and the readers find a value's
//! end with [`value_end`], so a block decodes and reads back the same.
//!
//! [`Terminator::Until`]: super::Terminator::Until

/// What ends one element's value, when structure follows it.
pub const PARAM_CLOSE: &str = "</parameter>";

/// What opens the next element.
pub const PARAM_TAG: &str = "<parameter=";

/// What closes the function, after its last element.
pub const FUNCTION_TAG: &str = "</function>";

/// The JSON escape a reader writes for `<` in a block translated to JSON, so
/// no tag inside a value survives into the translation.
pub const LT_ESCAPE: &str = "\\u003c";

/// What may stand between a value's `</parameter>` and the next tag: the
/// template's newline, and the few variants a model writes instead.
pub const SEPARATORS: [&str; 5] = ["\n", "\r\n", "\n\n", " \n", ""];

/// Every tail after `</parameter>` that ends a value: a separator, then a
/// tag. The first five open another element, the last five close the
/// function; tail `k` is separator `k % 5` before tag `k / 5`.
static BOTH: [&str; 10] = [
    "\n<parameter=",
    "\r\n<parameter=",
    "\n\n<parameter=",
    " \n<parameter=",
    "<parameter=",
    "\n</function>",
    "\r\n</function>",
    "\n\n</function>",
    " \n</function>",
    "</function>",
];

/// Which tags may follow a value: what the grammar offers next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Follows {
    /// Another element, or the function's close.
    Either,
    /// Another element only: a required one is still owed.
    Element,
    /// The function's close only: every element is written.
    Close,
}

impl Follows {
    /// The tails that end a value followed by these tags.
    pub fn tails(self) -> &'static [&'static str] {
        tails(self)
    }

    /// The tag tail `k` of [`Self::tails`] writes.
    pub fn tag(self, k: usize) -> &'static str {
        self.tails()[k].trim_start_matches([' ', '\r', '\n'])
    }
}

/// The tails that end a value followed by `follows`' tags.
pub fn tails(follows: Follows) -> &'static [&'static str] {
    match follows {
        Follows::Either => &BOTH,
        Follows::Element => &BOTH[..5],
        Follows::Close => &BOTH[5..],
    }
}

/// Where the value starting at `from` ends: the offset of its closing
/// `</parameter>`, the first one that structure follows. `None` when no
/// `</parameter>` in `text` is followed by a tag — a value cut short.
pub fn value_end(text: &str, from: usize) -> Option<usize> {
    let mut at = from;
    while let Some(rel) = text.get(at..)?.find(PARAM_CLOSE) {
        let close = at + rel;
        let after = &text[close + PARAM_CLOSE.len()..];
        if BOTH.iter().any(|tail| after.starts_with(tail)) {
            return Some(close);
        }
        at = close + PARAM_CLOSE.len();
    }
    None
}

/// One function block, read whole: its tool, its arguments in order, and the
/// offset just past its real `</function>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block<'a> {
    pub name: &'a str,
    /// Each argument's name and raw value, its framing newlines — one after
    /// the opening tag, one before the close — taken off.
    pub args: Vec<(&'a str, &'a str)>,
    pub end: usize,
}

/// The function block opening at `text`'s first `<function=`, read value by
/// value with [`value_end`], so a value holding `</parameter>`, `</function>`
/// or `</tool_call>` keeps it. `None` when the block is not complete — a value
/// with no close that structure follows, or anything but an element or the
/// close between elements.
pub fn read_block(text: &str) -> Option<Block<'_>> {
    let open = text.find("<function=")?;
    let name_end = open + text[open..].find('>')?;
    let name = text[open + "<function=".len()..name_end].trim();
    let mut args = Vec::new();
    let mut at = name_end + 1;
    loop {
        let rest = &text[at..];
        at += rest.len() - rest.trim_start().len();
        let rest = &text[at..];
        if rest.starts_with(FUNCTION_TAG) {
            return Some(Block {
                name,
                args,
                end: at + FUNCTION_TAG.len(),
            });
        }
        let named = rest.strip_prefix(PARAM_TAG)?;
        let key_len = named.find('>')?;
        let key = named[..key_len].trim();
        let value = at + PARAM_TAG.len() + key_len + 1;
        let close = value_end(text, value)?;
        let raw = &text[value..close];
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        args.push((key, raw));
        at = close + PARAM_CLOSE.len();
    }
}

/// Where the function block opening at `text`'s first `<function=` ends: the
/// offset just past its real `</function>` ([`read_block`]).
pub fn function_end(text: &str) -> Option<usize> {
    read_block(text).map(|b| b.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The first `</parameter>` that structure follows ends a value**, and
    /// every other one is content — whatever separator the model wrote.
    #[test]
    fn a_value_ends_where_structure_follows() {
        let text = "a</parameter> b</parameter>x</parameter>\n<parameter=next>";
        assert_eq!(value_end(text, 0), Some(text.rfind(PARAM_CLOSE).unwrap()));
        for sep in SEPARATORS {
            for tag in [PARAM_TAG, FUNCTION_TAG] {
                let text = format!("v</parameter>{sep}{tag}");
                assert_eq!(value_end(&text, 0), Some(1), "{text:?}");
            }
        }
        assert_eq!(value_end("cut </parameter> short", 0), None);
        assert_eq!(value_end("no close at all", 0), None);
    }

    /// **A function block's end is its real `</function>`**: a value holding
    /// `</function>`, `</tool_call>` or `</parameter>` keeps them.
    #[test]
    fn a_blocks_end_is_after_its_last_value() {
        let content = "x</parameter> y</function> z</tool_call>";
        let block = format!(
            "<function=write>\n<parameter=path>\na.js</parameter>\n<parameter=content>\n{content}</parameter>\n</function>\n</tool_call>"
        );
        let read = read_block(&block).unwrap();
        assert_eq!(read.name, "write");
        assert_eq!(read.args, [("path", "a.js"), ("content", content)]);
        assert_eq!(&block[read.end..], "\n</tool_call>");
        assert_eq!(function_end(&block), Some(read.end));
        assert_eq!(
            function_end("<function=datetime>\n</function>"),
            Some("<function=datetime>\n</function>".len())
        );
        assert_eq!(
            function_end("<function=write>\n<parameter=path>\na.js"),
            None
        );
    }

    /// The tails split by tag, separators in the same order on both sides.
    #[test]
    fn the_tails_are_each_separator_before_each_tag() {
        assert_eq!(tails(Follows::Element).len(), SEPARATORS.len());
        assert_eq!(tails(Follows::Close).len(), SEPARATORS.len());
        for (k, sep) in SEPARATORS.iter().enumerate() {
            assert_eq!(tails(Follows::Element)[k], format!("{sep}{PARAM_TAG}"));
            assert_eq!(tails(Follows::Close)[k], format!("{sep}{FUNCTION_TAG}"));
        }
        assert_eq!(Follows::Either.tag(7), FUNCTION_TAG);
        assert_eq!(Follows::Either.tag(1), PARAM_TAG);
    }
}
