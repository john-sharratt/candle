//! A JSON tool call read as it decodes (`{"name": …, "arguments": {…}}`), for
//! [`crate::parts`]: its tool's name once it is written, and each argument's
//! value as it arrives — a string's text a chunk at a time, unescaped; any
//! other value whole, as its JSON, once it ends. Reading is string-aware, so a
//! value holding `</tool_call>` or a brace stays inside its string.
//!
//! The reader keeps only what it has not released: a key being read, the
//! tool's name, a non-string value's JSON. A file's content streams through.

/// What a chunk of the call releases, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonOut {
    /// The tool the call names.
    Name(String),
    /// The next chunk of argument `name`'s value.
    Arg { name: String, text: String },
}

/// The keys an argument object is found under.
const ARGUMENT_KEYS: [&str; 2] = ["arguments", "parameters"];

#[derive(Debug, Clone, PartialEq, Eq)]
enum Frame {
    Object { key: Option<String> },
    Array,
}

/// What a string being read is.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Role {
    Key,
    Name,
    Arg(String),
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Escape {
    None,
    Slash,
    /// `\u` and the hex digits read so far.
    Hex(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Lex {
    /// A value may start.
    Value,
    /// An object's key, or its close.
    Key,
    /// The `:` after a key.
    Colon,
    Str {
        role: Role,
        text: String,
        escape: Escape,
        /// A high surrogate waiting for its low half.
        high: Option<u32>,
    },
    /// A number, `true`, `false` or `null`.
    Bare,
    /// A value ended: a `,` or a close follows.
    After,
    /// The call's object is closed.
    Done,
    /// It is no JSON call.
    Bad,
}

/// One JSON call's reader. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonCallStream {
    stack: Vec<Frame>,
    lex: Lex,
    /// An argument whose value is not a string: its name, its JSON so far,
    /// and the depth its value started at.
    capture: Option<(String, String, usize)>,
}

impl Default for JsonCallStream {
    fn default() -> Self {
        Self {
            stack: Vec::new(),
            lex: Lex::Value,
            capture: None,
        }
    }
}

impl JsonCallStream {
    /// Whether the call's object has closed.
    pub fn is_done(&self) -> bool {
        self.lex == Lex::Done
    }

    /// Whether what was read is no JSON call.
    pub fn is_bad(&self) -> bool {
        self.lex == Lex::Bad
    }

    /// Read `text`, up to the object's close; how many bytes of it were read,
    /// and what they release. A string's text read so far is released at the
    /// end of the chunk.
    pub fn push(&mut self, text: &str) -> (usize, Vec<JsonOut>) {
        let mut out = Vec::new();
        for (at, c) in text.char_indices() {
            if matches!(self.lex, Lex::Done | Lex::Bad) {
                return (at, out);
            }
            self.step(c, &mut out);
        }
        if let Lex::Str {
            role: Role::Arg(name),
            text,
            ..
        } = &mut self.lex
        {
            if !text.is_empty() {
                out.push(JsonOut::Arg {
                    name: name.clone(),
                    text: std::mem::take(text),
                });
            }
        }
        (text.len(), out)
    }

    /// The argument a value starting here is the value of, when it is one.
    fn argument(&self) -> Option<String> {
        match self.stack.as_slice() {
            [Frame::Object { key: Some(top) }, Frame::Object { key: Some(arg) }]
                if ARGUMENT_KEYS.contains(&top.as_str()) =>
            {
                Some(arg.clone())
            }
            _ => None,
        }
    }

    /// Whether a value starting here is the call's tool name.
    fn is_name(&self) -> bool {
        matches!(self.stack.as_slice(), [Frame::Object { key: Some(k) }] if k == "name")
    }

    fn step(&mut self, c: char, out: &mut Vec<JsonOut>) {
        if let Some((_, raw, _)) = &mut self.capture {
            raw.push(c);
        }
        match std::mem::replace(&mut self.lex, Lex::Bad) {
            Lex::Value => self.value(c, out),
            Lex::Key => match c {
                c if c.is_whitespace() => self.lex = Lex::Key,
                '"' => self.string(Role::Key),
                '}' => self.close(out),
                _ => {}
            },
            Lex::Colon => match c {
                c if c.is_whitespace() => self.lex = Lex::Colon,
                ':' => self.lex = Lex::Value,
                _ => {}
            },
            Lex::Str {
                role,
                text,
                escape,
                high,
            } => self.in_string(c, role, text, escape, high, out),
            Lex::Bare => match c {
                ',' | '}' | ']' => {
                    self.ended(out, true);
                    self.after(c, out);
                }
                c if c.is_whitespace() => self.ended(out, true),
                _ => self.lex = Lex::Bare,
            },
            Lex::After => self.after(c, out),
            done @ (Lex::Done | Lex::Bad) => self.lex = done,
        }
    }

    fn value(&mut self, c: char, out: &mut Vec<JsonOut>) {
        if c.is_whitespace() {
            self.lex = Lex::Value;
            return;
        }
        let argument = self.argument();
        if self.stack.is_empty() && c != '{' {
            return;
        }
        match c {
            '"' => {
                let role = match argument {
                    Some(name) => Role::Arg(name),
                    None if self.is_name() => Role::Name,
                    None => Role::Other,
                };
                self.string(role);
            }
            '{' | '[' => {
                if let Some(name) = argument {
                    self.capture = Some((name, c.to_string(), self.stack.len()));
                }
                self.stack.push(match c {
                    '{' => Frame::Object { key: None },
                    _ => Frame::Array,
                });
                self.lex = match c {
                    '{' => Lex::Key,
                    _ => Lex::Value,
                };
            }
            // An empty array closes where its first value would start.
            ']' if self.stack.last() == Some(&Frame::Array) => self.close(out),
            c if c == '-' || c.is_ascii_alphanumeric() => {
                if let Some(name) = argument {
                    self.capture = Some((name, c.to_string(), self.stack.len()));
                }
                self.lex = Lex::Bare;
            }
            _ => {}
        }
    }

    fn string(&mut self, role: Role) {
        self.lex = Lex::Str {
            role,
            text: String::new(),
            escape: Escape::None,
            high: None,
        };
    }

    fn in_string(
        &mut self,
        c: char,
        role: Role,
        mut text: String,
        escape: Escape,
        mut high: Option<u32>,
        out: &mut Vec<JsonOut>,
    ) {
        let mut escape = escape;
        match escape {
            Escape::None => match c {
                '"' => {
                    self.string_ended(role, text, out);
                    return;
                }
                '\\' => escape = Escape::Slash,
                c => text.push(c),
            },
            Escape::Slash => {
                escape = Escape::None;
                match c {
                    'n' => text.push('\n'),
                    't' => text.push('\t'),
                    'r' => text.push('\r'),
                    'b' => text.push('\u{8}'),
                    'f' => text.push('\u{c}'),
                    'u' => escape = Escape::Hex(String::new()),
                    c => text.push(c),
                }
            }
            Escape::Hex(mut hex) => {
                hex.push(c);
                escape = Escape::Hex(hex.clone());
                if hex.len() == 4 {
                    escape = Escape::None;
                    let Ok(code) = u32::from_str_radix(&hex, 16) else {
                        return;
                    };
                    match (high.take(), code) {
                        (None, 0xD800..=0xDBFF) => high = Some(code),
                        (Some(h), 0xDC00..=0xDFFF) => text.extend(char::from_u32(
                            0x10000 + ((h - 0xD800) << 10) + (code - 0xDC00),
                        )),
                        (_, code) => text.push(char::from_u32(code).unwrap_or('\u{fffd}')),
                    }
                }
            }
        }
        self.lex = Lex::Str {
            role,
            text,
            escape,
            high,
        };
    }

    fn string_ended(&mut self, role: Role, text: String, out: &mut Vec<JsonOut>) {
        match role {
            Role::Key => {
                if let Some(Frame::Object { key }) = self.stack.last_mut() {
                    *key = Some(text);
                }
                self.lex = Lex::Colon;
                return;
            }
            Role::Name => out.push(JsonOut::Name(text)),
            Role::Arg(name) if !text.is_empty() => out.push(JsonOut::Arg { name, text }),
            Role::Arg(_) | Role::Other => {}
        }
        self.ended(out, false);
    }

    /// A value ended. A non-string argument's JSON is released once its value
    /// is back at the depth it started at; `trailing` drops the character
    /// that ended a bare value.
    fn ended(&mut self, out: &mut Vec<JsonOut>, trailing: bool) {
        self.lex = Lex::After;
        let at_start = self
            .capture
            .as_ref()
            .is_some_and(|(_, _, depth)| *depth == self.stack.len());
        if at_start {
            if let Some((name, mut raw, _)) = self.capture.take() {
                if trailing {
                    raw.pop();
                }
                out.push(JsonOut::Arg { name, text: raw });
            }
        }
    }

    fn after(&mut self, c: char, out: &mut Vec<JsonOut>) {
        match (c, self.stack.last()) {
            (c, _) if c.is_whitespace() => self.lex = Lex::After,
            (',', Some(Frame::Object { .. })) => {
                if let Some(Frame::Object { key }) = self.stack.last_mut() {
                    *key = None;
                }
                self.lex = Lex::Key;
            }
            (',', Some(Frame::Array)) => self.lex = Lex::Value,
            ('}', Some(Frame::Object { .. })) | (']', Some(Frame::Array)) => self.close(out),
            _ => {}
        }
    }

    /// The innermost object or array closed.
    fn close(&mut self, out: &mut Vec<JsonOut>) {
        self.stack.pop();
        if self.stack.is_empty() {
            self.lex = Lex::Done;
            return;
        }
        self.ended(out, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg(name: &str, text: &str) -> JsonOut {
        JsonOut::Arg {
            name: name.into(),
            text: text.into(),
        }
    }

    /// Read `text` a character at a time, merging an argument's chunks.
    fn read(text: &str) -> (JsonCallStream, Vec<JsonOut>, usize) {
        let mut s = JsonCallStream::default();
        let mut all: Vec<JsonOut> = Vec::new();
        let mut read = 0;
        for (at, c) in text.char_indices() {
            let (n, out) = s.push(&text[at..at + c.len_utf8()]);
            read += n;
            for o in out {
                match (all.last_mut(), o) {
                    (
                        Some(JsonOut::Arg { name: a, text: t }),
                        JsonOut::Arg {
                            name: b,
                            text: more,
                        },
                    ) if *a == b => t.push_str(&more),
                    (_, o) => all.push(o),
                }
            }
            if s.is_done() || s.is_bad() {
                break;
            }
        }
        (s, all, read)
    }

    /// **A call's name and each argument arrive as they decode**, a string's
    /// text unescaped and any other value as its JSON.
    #[test]
    fn a_call_streams_its_name_and_arguments() {
        let call = r#"{"name": "file_write", "arguments": {"path": "a.md", "content": "line \"one\"\nlineé 😀", "n": 12, "flags": {"x": [1, "}"]}, "ok": true}}"#;
        let (s, got, read) = read(call);
        assert!(s.is_done());
        assert_eq!(read, call.len());
        assert_eq!(
            got,
            vec![
                JsonOut::Name("file_write".into()),
                arg("path", "a.md"),
                arg("content", "line \"one\"\nlineé 😀"),
                arg("n", "12"),
                arg("flags", r#"{"x": [1, "}"]}"#),
                arg("ok", "true"),
            ]
        );
    }

    /// **A string's text is released before the string ends**, so a long
    /// value streams.
    #[test]
    fn a_strings_text_is_released_as_it_arrives() {
        let mut s = JsonCallStream::default();
        let (_, first) = s.push(r#"{"name": "file_write", "arguments": {"content": "half a"#);
        assert_eq!(
            first,
            vec![JsonOut::Name("file_write".into()), arg("content", "half a")]
        );
        let (_, next) = s.push(r#" file</tool_call>"#);
        assert_eq!(next, vec![arg("content", " file</tool_call>")]);
        assert!(!s.is_done());
    }

    /// **Reading stops at the object's close**, leaving what follows it.
    #[test]
    fn reading_stops_at_the_close() {
        let mut s = JsonCallStream::default();
        let text = r#"{"name": "a", "arguments": {}}</tool_call>after"#;
        let (n, out) = s.push(text);
        assert!(s.is_done());
        assert_eq!(&text[n..], "</tool_call>after");
        assert_eq!(out, vec![JsonOut::Name("a".into())]);
    }

    /// **What is no object is no call.**
    #[test]
    fn text_that_is_no_object_is_bad() {
        let mut s = JsonCallStream::default();
        s.push("  hello");
        assert!(s.is_bad());
    }
}
