//! An incremental JSON value lexer that knows, byte by byte, whether what has
//! been written can still become valid JSON — and what would finish it.
//!
//! # Why a free span needs one
//!
//! A free-text span is decoded **unmasked**: the stencil cannot stop the model
//! writing a token, only decide what to commit once it has. A terminator that
//! merely counts brackets commits whatever arrives, so a model that closes an
//! array with `}`, stops after `"end_line":`, writes a raw newline into a
//! command, or writes `True` produces a call that does not parse — and a call
//! that does not parse is no call at all, taking every argument the model *did*
//! get right with it.
//!
//! This lexer tracks the full JSON state — a container stack, string escapes,
//! number and literal sub-states — so the span can do three things with a
//! token:
//!
//! - **Commit it** when every byte extends a valid value.
//! - **Rewrite it** when a byte is wrong only in how it is spelled, keeping
//!   what the model meant (see below).
//! - **End the value** at the first byte that cannot continue it, writing the
//!   smallest text that completes what is open ([`JsonLexer::completion`]):
//!   close the string, supply the missing digit or value, close the containers
//!   in the order they opened.
//!
//! The same completion finishes a value the decode cuts short — an intercepted
//! EOS, or the span's hard limit.
//!
//! # A repair keeps the meaning
//!
//! Models write JSON the way the languages they learned it beside write their
//! literals — Python's `True` and `None`, a single-quoted string, an unquoted
//! key. Each of those has exactly one JSON meaning, and ending the value there
//! would throw it away: `True` would become `null`, the opposite of what was
//! said. So the spelling is corrected and the value goes on:
//!
//! | written | committed |
//! |---|---|
//! | `True` `TRUE` `False` `NULL` (any case) | `true` `true` `false` `null` |
//! | `None` `nil` `undefined` | `null` |
//! | `NaN` `Infinity` `inf` | `null` — how `JSON.stringify` writes a non-finite number |
//! | `'text'`, `{'key': …}` | `"text"`, `{"key": …}` — an inner `"` escaped |
//! | `{key: …}`, `{1: …}` | `{"key": …}`, `{"1": …}` |
//! | `\'` `\x41` `\a` `\v` in a string | `'` `A` `` `` |
//! | a raw control character in a string | its escape (`\n`, ``) |
//! | `{"a" 1}` | `{"a": 1}` — the missing colon |
//! | `[1 2]`, `{"a": 1 "b": 2}` | `[1, 2]`, `{"a": 1, "b": 2}` — the missing comma |
//! | `.5`, `-.5`, `5.` | `0.5`, `-0.5`, `5.0` |
//! | `[1, 2,]` with the `,` and `]` in one token | `[1, 2]` |
//!
//! Only the current token can be rewritten — what is committed is in the K/V.
//! A trailing comma committed by an earlier token cannot be taken back, so that
//! array is completed with the smallest value instead (`[1, 2, null]`); the
//! caller sees [`Effect::TrailingComma`] and decides which applies.
//!
//! A value that is complete at the top level ends where JSON says it does: a
//! string or container on its own closing byte, a number or literal only when
//! a delimiter of the enclosing object arrives (`,`, `}`, `]`), which belongs
//! to the next node.

/// One open container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    Array,
    Object,
}

/// Where a string is in an escape sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Escape {
    None,
    /// After `\`.
    Backslash,
    /// After `\u` and this many hex digits.
    Unicode(u8),
}

/// The quote a string was opened with. A single-quoted string is written as a
/// double-quoted one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quote {
    Double,
    Single,
}

/// Where a number is: each state names the last thing written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Number {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigit,
}

/// A word a value may start with, and the JSON literal it means.
///
/// Matched without regard to case. No word maps to two literals, and every
/// word sharing a first letter maps to the same one, so what to write is known
/// from the first byte even while the word is still ambiguous.
const WORDS: &[(&[u8], &[u8])] = &[
    (b"true", b"true"),
    (b"false", b"false"),
    (b"null", b"null"),
    // Python.
    (b"none", b"null"),
    // Ruby, Lua, Go.
    (b"nil", b"null"),
    // JavaScript.
    (b"undefined", b"null"),
    // A number JSON cannot hold, written the way `JSON.stringify` writes it.
    (b"nan", b"null"),
    (b"infinity", b"null"),
    (b"inf", b"null"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// A value is expected. `closes_array`: directly after `[`, where `]` is
    /// also legal.
    Value {
        closes_array: bool,
    },
    /// Inside a string; `key` when it is an object key.
    String {
        key: bool,
        quote: Quote,
        escape: Escape,
    },
    /// Inside an unquoted object key, whose opening quote has been written.
    BareKey,
    Number(Number),
    /// Partway through a word from [`WORDS`]: `candidates` is a bitmask of the
    /// words still matching, `at` how many bytes have been read.
    Literal {
        candidates: u16,
        at: usize,
    },
    /// A value is complete; a separator or closer may follow.
    AfterValue,
    /// Directly after `{`: a key or `}`.
    KeyOrClose,
    /// After `,` in an object: a key.
    Key,
    /// After a key: `:`.
    Colon,
}

/// What one byte does to the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// The value continues.
    Continue,
    /// The value is complete through this byte — a top-level string or
    /// container closed.
    Complete,
    /// The value was already complete; this byte belongs to what encloses it.
    Delimiter,
    /// The byte cannot be part of the value. Anything in [`Step::write`] is
    /// committed first (a word finished before the byte), and then
    /// [`JsonLexer::completion`] finishes the value as written.
    Invalid,
    /// A closer directly after a separator — `[1, 2,]`, `{"a": 1,}`. Nothing
    /// changed. If the separator can still be taken back, the caller does so
    /// and calls [`JsonLexer::retract_separator`], then feeds the byte again;
    /// otherwise this is [`Effect::Invalid`].
    TrailingComma,
}

/// The outcome of one byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// `Some` when the byte must be written as these bytes instead of itself
    /// (possibly none of them).
    pub write: Option<Vec<u8>>,
    pub effect: Effect,
}

impl Step {
    fn as_is(effect: Effect) -> Self {
        Step {
            write: None,
            effect,
        }
    }

    fn written(write: Vec<u8>, effect: Effect) -> Self {
        Step {
            write: Some(write),
            effect,
        }
    }
}

/// The lexer state for one value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonLexer {
    stack: Vec<Frame>,
    state: State,
    /// The last byte written into the value, `None` before the first. It
    /// decides whether inserted text needs a space: after the key's `:` or a
    /// `,` a supplied `null` does; after whitespace it does not.
    last: Option<u8>,
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// A byte that can appear in an unquoted key.
fn is_key_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// The escape for a control character inside a string.
fn escape_control(b: u8) -> Vec<u8> {
    match b {
        b'\n' => b"\\n".to_vec(),
        b'\t' => b"\\t".to_vec(),
        b'\r' => b"\\r".to_vec(),
        0x08 => b"\\b".to_vec(),
        0x0c => b"\\f".to_vec(),
        _ => format!("\\u{b:04x}").into_bytes(),
    }
}

/// The literal every word in `candidates` means.
fn target_of(candidates: u16) -> &'static [u8] {
    let first = candidates.trailing_zeros() as usize;
    WORDS[first].1
}

/// `prefix`, then what `inner` writes for byte `b` — itself, unless `inner`
/// did not take it.
fn prefixed(mut prefix: Vec<u8>, b: u8, inner: Step) -> Step {
    match (inner.write, inner.effect) {
        (Some(w), effect) => {
            prefix.extend(w);
            Step::written(prefix, effect)
        }
        (None, effect @ (Effect::Invalid | Effect::Delimiter | Effect::TrailingComma)) => {
            Step::written(prefix, effect)
        }
        (None, effect) => {
            prefix.push(b);
            Step::written(prefix, effect)
        }
    }
}

impl JsonLexer {
    /// Any JSON value, from its first byte.
    pub fn value() -> Self {
        JsonLexer {
            stack: Vec::new(),
            state: State::Value {
                closes_array: false,
            },
            last: None,
        }
    }

    /// The content of a string whose opening quote is already written.
    pub fn string() -> Self {
        JsonLexer {
            stack: Vec::new(),
            state: State::String {
                key: false,
                quote: Quote::Double,
                escape: Escape::None,
            },
            last: Some(b'"'),
        }
    }

    /// Feed one byte.
    pub fn step(&mut self, b: u8) -> Step {
        let step = self.transition(b);
        match (&step.write, step.effect) {
            (Some(w), _) => {
                if let Some(&l) = w.last() {
                    self.last = Some(l);
                }
            }
            (None, Effect::Continue | Effect::Complete) => self.last = Some(b),
            (None, _) => {}
        }
        step
    }

    /// Take back the separator before a closer ([`Effect::TrailingComma`]):
    /// the value before it was complete, so the closer is read after that
    /// value. `last` is the last byte still written.
    pub fn retract_separator(&mut self, last: Option<u8>) {
        self.state = State::AfterValue;
        self.last = last;
    }

    /// The smallest text that turns what has been written into a complete
    /// value. Empty when it already is one.
    pub fn completion(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let after_separator = matches!(self.last, None | Some(b':') | Some(b','));
        match self.state {
            State::Value { closes_array: true } | State::AfterValue | State::KeyOrClose => {}
            State::Value {
                closes_array: false,
            } => out.extend_from_slice(if after_separator { b" null" } else { b"null" }),
            State::String { key, escape, .. } => {
                match escape {
                    // The written `\` and this one make a literal backslash.
                    Escape::Backslash => out.push(b'\\'),
                    Escape::Unicode(n) => out.extend(vec![b'0'; 4 - n as usize]),
                    Escape::None => {}
                }
                out.push(b'"');
                if key {
                    out.extend_from_slice(b": null");
                }
            }
            State::BareKey => out.extend_from_slice(b"\": null"),
            State::Number(Number::Minus | Number::Dot | Number::Exp | Number::ExpSign) => {
                out.push(b'0')
            }
            State::Number(_) => {}
            State::Literal { candidates, at } => {
                let target = target_of(candidates);
                out.extend_from_slice(&target[at.min(target.len())..]);
            }
            State::Key => out.extend_from_slice(if after_separator {
                b" \"\": null"
            } else {
                b"\"\": null"
            }),
            State::Colon => out.extend_from_slice(b": null"),
        }
        out.extend(self.stack.iter().rev().map(|f| match f {
            Frame::Array => b']',
            Frame::Object => b'}',
        }));
        out
    }

    fn transition(&mut self, b: u8) -> Step {
        match self.state {
            State::Value { closes_array } => self.value_byte(b, closes_array),
            State::String { key, quote, escape } => self.string_byte(b, key, quote, escape),
            State::BareKey => self.bare_key_byte(b),
            State::Number(n) => self.number(b, n),
            State::Literal { candidates, at } => self.literal(b, candidates, at),
            State::AfterValue => self.after_value(b),
            State::KeyOrClose => match b {
                _ if is_ws(b) => Step::as_is(Effect::Continue),
                b'}' => self.close_frame(),
                _ => self.key_start(b),
            },
            State::Key => match b {
                _ if is_ws(b) => Step::as_is(Effect::Continue),
                b'}' => Step::as_is(Effect::TrailingComma),
                _ => self.key_start(b),
            },
            State::Colon => match b {
                _ if is_ws(b) => Step::as_is(Effect::Continue),
                b':' => {
                    self.state = State::Value {
                        closes_array: false,
                    };
                    Step::as_is(Effect::Continue)
                }
                // The colon is missing: a value began where it belonged.
                _ => {
                    self.state = State::Value {
                        closes_array: false,
                    };
                    let inner = self.value_byte(b, false);
                    if inner.effect == Effect::Invalid {
                        self.state = State::Colon;
                        return Step::as_is(Effect::Invalid);
                    }
                    let colon = match self.last.is_some_and(is_ws) {
                        true => b":".to_vec(),
                        false => b": ".to_vec(),
                    };
                    prefixed(colon, b, inner)
                }
            },
        }
    }

    fn value_byte(&mut self, b: u8, closes_array: bool) -> Step {
        self.state = match b {
            _ if is_ws(b) => return Step::as_is(Effect::Continue),
            b'"' => State::String {
                key: false,
                quote: Quote::Double,
                escape: Escape::None,
            },
            b'\'' => {
                self.state = State::String {
                    key: false,
                    quote: Quote::Single,
                    escape: Escape::None,
                };
                return Step::written(b"\"".to_vec(), Effect::Continue);
            }
            b'[' => {
                self.stack.push(Frame::Array);
                State::Value { closes_array: true }
            }
            b'{' => {
                self.stack.push(Frame::Object);
                State::KeyOrClose
            }
            b'-' => State::Number(Number::Minus),
            b'0' => State::Number(Number::Zero),
            b'1'..=b'9' => State::Number(Number::Int),
            // `.5` means `0.5`.
            b'.' => {
                self.state = State::Number(Number::Dot);
                return Step::written(b"0.".to_vec(), Effect::Continue);
            }
            b']' if closes_array => return self.close_frame(),
            // A closer straight after a separator in an array.
            b']' if self.stack.last() == Some(&Frame::Array) => {
                return Step::as_is(Effect::TrailingComma)
            }
            _ if b.is_ascii_alphabetic() => {
                let candidates = (0..WORDS.len())
                    .filter(|&i| WORDS[i].0[0] == b.to_ascii_lowercase())
                    .fold(0u16, |m, i| m | 1 << i);
                if candidates == 0 {
                    return Step::as_is(Effect::Invalid);
                }
                return self.literal(b, candidates, 0);
            }
            _ => return Step::as_is(Effect::Invalid),
        };
        Step::as_is(Effect::Continue)
    }

    /// The start of an object key: quoted either way, or bare.
    fn key_start(&mut self, b: u8) -> Step {
        match b {
            b'"' => {
                self.state = State::String {
                    key: true,
                    quote: Quote::Double,
                    escape: Escape::None,
                };
                Step::as_is(Effect::Continue)
            }
            b'\'' => {
                self.state = State::String {
                    key: true,
                    quote: Quote::Single,
                    escape: Escape::None,
                };
                Step::written(b"\"".to_vec(), Effect::Continue)
            }
            _ if is_key_byte(b) => {
                self.state = State::BareKey;
                Step::written(vec![b'"', b], Effect::Continue)
            }
            _ => Step::as_is(Effect::Invalid),
        }
    }

    fn bare_key_byte(&mut self, b: u8) -> Step {
        match b {
            _ if is_key_byte(b) => Step::as_is(Effect::Continue),
            b':' => {
                self.state = State::Value {
                    closes_array: false,
                };
                Step::written(b"\":".to_vec(), Effect::Continue)
            }
            _ if is_ws(b) => {
                self.state = State::Colon;
                Step::written(vec![b'"', b], Effect::Continue)
            }
            _ => Step::as_is(Effect::Invalid),
        }
    }

    fn string_byte(&mut self, b: u8, key: bool, quote: Quote, escape: Escape) -> Step {
        let set = |lexer: &mut Self, escape| lexer.state = State::String { key, quote, escape };
        match escape {
            Escape::None => match (b, quote) {
                (b'"', Quote::Double) | (b'\'', Quote::Single) => {
                    let close = match quote {
                        Quote::Double => None,
                        Quote::Single => Some(b"\"".to_vec()),
                    };
                    let effect = if key {
                        self.state = State::Colon;
                        Effect::Continue
                    } else {
                        self.state = State::AfterValue;
                        match self.stack.is_empty() {
                            true => Effect::Complete,
                            false => Effect::Continue,
                        }
                    };
                    Step {
                        write: close,
                        effect,
                    }
                }
                // Content in a single-quoted string, and a quote in JSON.
                (b'"', Quote::Single) => Step::written(b"\\\"".to_vec(), Effect::Continue),
                (b'\\', _) => {
                    set(self, Escape::Backslash);
                    Step::as_is(Effect::Continue)
                }
                (0x00..=0x1f, _) => Step::written(escape_control(b), Effect::Continue),
                _ => Step::as_is(Effect::Continue),
            },
            Escape::Backslash => match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    set(self, Escape::None);
                    Step::as_is(Effect::Continue)
                }
                b'u' => {
                    set(self, Escape::Unicode(0));
                    Step::as_is(Effect::Continue)
                }
                // The escapes other languages have, as the `\u` JSON spells
                // them with — the backslash is already written.
                b'\'' => {
                    set(self, Escape::None);
                    Step::written(b"u0027".to_vec(), Effect::Continue)
                }
                b'a' => {
                    set(self, Escape::None);
                    Step::written(b"u0007".to_vec(), Effect::Continue)
                }
                b'v' => {
                    set(self, Escape::None);
                    Step::written(b"u000b".to_vec(), Effect::Continue)
                }
                // `\xHH`: the two hex digits that follow finish a `\u00HH`.
                b'x' => {
                    set(self, Escape::Unicode(2));
                    Step::written(b"u00".to_vec(), Effect::Continue)
                }
                // No meaning in any of them. A second backslash turns the
                // written one into a literal, and the byte follows as content —
                // which is what Python makes of `'\q'`.
                _ => {
                    set(self, Escape::None);
                    let mut write = vec![b'\\'];
                    match b {
                        0x00..=0x1f => write.extend(escape_control(b)),
                        _ => write.push(b),
                    }
                    Step::written(write, Effect::Continue)
                }
            },
            Escape::Unicode(n) if b.is_ascii_hexdigit() => {
                set(
                    self,
                    match n {
                        3 => Escape::None,
                        _ => Escape::Unicode(n + 1),
                    },
                );
                Step::as_is(Effect::Continue)
            }
            // A `\u` with too few digits: pad it with zeros, then read the byte
            // as ordinary content — which may be the closing quote.
            Escape::Unicode(n) => {
                set(self, Escape::None);
                let inner = self.transition(b);
                prefixed(vec![b'0'; 4 - n as usize], b, inner)
            }
        }
    }

    fn number(&mut self, b: u8, n: Number) -> Step {
        let next = match (n, b) {
            (Number::Minus, b'0') => Number::Zero,
            (Number::Minus, b'1'..=b'9') => Number::Int,
            // `-.5` means `-0.5`.
            (Number::Minus, b'.') => {
                self.state = State::Number(Number::Dot);
                return Step::written(b"0.".to_vec(), Effect::Continue);
            }
            (Number::Int, b'0'..=b'9') => Number::Int,
            (Number::Zero | Number::Int, b'.') => Number::Dot,
            (Number::Dot | Number::Frac, b'0'..=b'9') => Number::Frac,
            (Number::Zero | Number::Int | Number::Frac, b'e' | b'E') => Number::Exp,
            (Number::Exp, b'+' | b'-') => Number::ExpSign,
            (Number::Exp | Number::ExpSign | Number::ExpDigit, b'0'..=b'9') => Number::ExpDigit,
            // `5.` means `5.0`: the digit is supplied, and the byte is read
            // after it.
            (Number::Dot, _) => {
                self.state = State::Number(Number::Frac);
                let inner = self.number(b, Number::Frac);
                return prefixed(b"0".to_vec(), b, inner);
            }
            // A number that is complete here ends, and the byte is read after
            // it — a separator, a closer, whitespace, or nothing JSON allows.
            (Number::Zero | Number::Int | Number::Frac | Number::ExpDigit, _) => {
                self.state = State::AfterValue;
                let step = self.after_value(b);
                if step.effect == Effect::Invalid {
                    self.state = State::Number(n);
                }
                return step;
            }
            _ => return Step::as_is(Effect::Invalid),
        };
        self.state = State::Number(next);
        Step::as_is(Effect::Continue)
    }

    fn literal(&mut self, b: u8, candidates: u16, at: usize) -> Step {
        let target = target_of(candidates);
        let lower = b.to_ascii_lowercase();
        let matching = (0..WORDS.len())
            .filter(|&i| candidates & (1 << i) != 0 && WORDS[i].0.get(at) == Some(&lower))
            .fold(0u16, |m, i| m | 1 << i);
        if matching == 0 {
            // A word that is already whole here ends, and the byte is read
            // after it: `inf` followed by `,` is not a broken `infinity`.
            let whole =
                (0..WORDS.len()).any(|i| candidates & (1 << i) != 0 && WORDS[i].0.len() == at);
            if !whole {
                return Step::as_is(Effect::Invalid);
            }
            self.state = State::AfterValue;
            let inner = self.after_value(b);
            return prefixed(target[at.min(target.len())..].to_vec(), b, inner);
        }
        let mut write = Vec::new();
        if let Some(&t) = target.get(at) {
            write.push(t);
        }
        let at = at + 1;
        let finished = (0..WORDS.len())
            .filter(|&i| matching & (1 << i) != 0)
            .all(|i| WORDS[i].0.len() == at);
        if finished {
            write.extend_from_slice(&target[at.min(target.len())..]);
            self.state = State::AfterValue;
        } else {
            self.state = State::Literal {
                candidates: matching,
                at,
            };
        }
        match write.as_slice() == [b] {
            true => Step::as_is(Effect::Continue),
            false => Step::written(write, Effect::Continue),
        }
    }

    fn after_value(&mut self, b: u8) -> Step {
        if is_ws(b) {
            return Step::as_is(Effect::Continue);
        }
        match (self.stack.last(), b) {
            (None, b',' | b'}' | b']') => Step::as_is(Effect::Delimiter),
            (Some(Frame::Array), b',') => {
                self.state = State::Value {
                    closes_array: false,
                };
                Step::as_is(Effect::Continue)
            }
            (Some(Frame::Object), b',') => {
                self.state = State::Key;
                Step::as_is(Effect::Continue)
            }
            (Some(Frame::Array), b']') | (Some(Frame::Object), b'}') => self.close_frame(),
            // The comma is missing: the next element or key began where it
            // belonged.
            (Some(frame), _) => {
                let inner = match frame {
                    Frame::Array => {
                        self.state = State::Value {
                            closes_array: false,
                        };
                        self.value_byte(b, false)
                    }
                    Frame::Object => self.key_start(b),
                };
                if matches!(inner.effect, Effect::Invalid | Effect::TrailingComma) {
                    self.state = State::AfterValue;
                    return Step::as_is(Effect::Invalid);
                }
                let comma = match self.last.is_some_and(is_ws) {
                    true => b",".to_vec(),
                    false => b", ".to_vec(),
                };
                prefixed(comma, b, inner)
            }
            (None, _) => Step::as_is(Effect::Invalid),
        }
    }

    fn close_frame(&mut self) -> Step {
        self.stack.pop();
        self.state = State::AfterValue;
        Step::as_is(match self.stack.is_empty() {
            true => Effect::Complete,
            false => Effect::Continue,
        })
    }
}

#[cfg(test)]
mod tests;
