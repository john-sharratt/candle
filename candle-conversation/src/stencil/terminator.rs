//! Free-text terminators — the escape- and nesting-aware byte lexer that ends a
//! free-text span.
//!
//! It runs over the decoded **bytes** of each token (not token identity), which
//! is what makes it robust to however the tokenizer happened to chunk a value.
//! State carries across `feed` calls so a span that spans many tokens is lexed
//! correctly.
//!
//! The two JSON terminators do more than find the end: they validate as they go
//! (see [`JsonLexer`]), and a token that would make the value invalid comes back
//! as [`Feed::Rewrite`] — the bytes to commit in its place.

use super::json_lexer::{Effect, JsonLexer};

/// What ends a free-text span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminator {
    /// A JSON string value whose opening quote is already written: ends at the
    /// first UNESCAPED `"`, which is consumed.  A raw control character is
    /// rewritten as its escape, an escape JSON does not have as a literal
    /// backslash, a short `\u` padded — so the string always parses.
    JsonString,
    /// A JSON string value **including its opening quote** — the span starts
    /// at the field's colon, the model writes ` "` (or ` ""`) itself, and the
    /// span ends at the first unescaped `"` after that opening one, consumed.
    ///
    /// Exists because the opening quote is not the grammar's to write. Qwen's
    /// tokenizer spells an empty string as ONE token, ` ""`, and a non-empty
    /// one as ` "` followed by content, so a grammar that prefills ` "` has
    /// already chosen "non-empty" on the model's behalf. Asked for an empty
    /// value from there, the model writes the token that follows ` ""` in its
    /// training — `}}` — and a [`Terminator::JsonString`] span swallows it,
    /// the call's own close with it, as string content. A live turn produced
    /// `{"prefix": "}}\n</tool_call>"}}` exactly that way and ended silently.
    ///
    /// Before the opening quote only whitespace belongs to the value. Any other
    /// byte there — a `}}` where the value belonged, or bare text with no
    /// quote in front of it — ends the span with the token rewritten as a whole
    /// empty value, so the call's structure stays the grammar's. Were bare text
    /// let through, its first later `"` would open the string instead and the
    /// call's own close would be swallowed as content.
    JsonStringValue,
    /// A JSON number value: lookahead-terminated at the first byte that cannot
    /// extend a number.  The terminator byte is NOT consumed — it belongs to the
    /// following node.  `integer_only` drops `.`/`e`/`E`.
    JsonNumber { integer_only: bool },
    /// A balanced `open`/`close` structure (a raw object/array value): ends when
    /// nesting depth returns to 0.  String-aware — brackets inside a `"…"` are
    /// ignored, with `\` escaping inside that string — so a `}` in a string
    /// never affects depth.  The span includes the opening and closing bracket;
    /// the closing bracket is consumed.
    Balanced { open: u8, close: u8 },
    /// Any JSON value (number, `true`/`false`/`null`, string, array, object) as
    /// the value of an object field, respecting nested `[]`/`{}` and strings.
    /// A string, array or object that IS the value ends at its own closing
    /// quote or bracket, which is consumed — the value is complete there, and
    /// ending it anywhere later lets the model write past it (`["a"]]`). A bare
    /// scalar is lookahead-terminated at the first `,`, `}` or `]` at the
    /// ENCLOSING object's depth (depth 0); that delimiter is NOT consumed — it
    /// belongs to the following node.
    ///
    /// Validated byte by byte: the first byte that cannot continue the value
    /// ends it, with the value completed as written — a mismatched closer
    /// closes what was actually opened, an empty value becomes `null`. Guarantees
    /// a valid JSON value without enforcing its scalar type.
    JsonValue,
    /// No byte delimiter at all: `feed` always returns `Continue`.  The span
    /// ends only via a close *token* (`FreeTextSpan::close_token`), an EOS
    /// sample (`eos_ends`), or the hard `forced_after` limit.  Used by the
    /// thinking-block steering tree, whose `</think>` close is a token, not a
    /// byte pattern.
    Never,
    /// Raw text ending at a literal marker — `"\n</parameter>"`.
    ///
    /// **The value is not JSON**, which is the whole reason this exists.
    /// Qwen3.5's tool-call syntax puts each argument in its own element and
    /// takes the value as unescaped text, so a quote, a backslash or a newline
    /// inside it is ordinary content rather than something to escape. None of
    /// the JSON terminators can express that: `JsonString` would close on the
    /// first bare `"` in a character's prose.
    ///
    /// The marker is **consumed**, exactly as `JsonString` consumes its closing
    /// quote. Two consequences, both deliberate:
    ///
    /// * The tree does not emit the marker as a static after the span, because
    ///   the span has already produced it.
    /// * The model has to write it, and a decode that never does runs to
    ///   `forced_after` — the same contract `JsonString` has always had for its
    ///   closing quote, rather than a new failure mode.
    ///
    /// Matching runs across token boundaries: a marker is several BPE pieces
    /// (`</parameter>` is not a special token in any vocabulary this targets),
    /// so a partial match at the end of one token has to survive into the next.
    Until { marker: &'static str },
}

impl Terminator {
    /// Whether this terminator is lookahead — its closing delimiter is decoded
    /// but belongs to the *next* node (the session pushes it back), rather than
    /// being consumed by the span (`JsonString`/`Balanced`).
    pub fn is_lookahead(self) -> bool {
        matches!(self, Terminator::JsonNumber { .. } | Terminator::JsonValue)
    }

    pub fn start(self) -> TerminatorState {
        TerminatorState {
            kind: self,
            depth: 0,
            in_string: false,
            escaped: false,
            started: false,
            spaced: false,
            matched: 0,
            json: match self {
                Terminator::JsonString => Some(JsonLexer::string()),
                Terminator::JsonValue => Some(JsonLexer::value()),
                _ => None,
            },
        }
    }
}

/// The running lexer state for one active span.
#[derive(Debug, Clone)]
pub struct TerminatorState {
    kind: Terminator,
    /// Bracket nesting depth (`Balanced`).
    depth: u32,
    /// Inside a nested `"…"` (string-aware bracket matching).
    in_string: bool,
    /// Previous byte was an unconsumed backslash.
    escaped: bool,
    /// `Balanced` has seen its first `open` (so a later return to depth 0 is a
    /// real close, not the pre-open state).
    started: bool,
    /// [`Terminator::JsonStringValue`]: whitespace has been written ahead of the
    /// opening quote, so an empty value written in its place needs no space of
    /// its own.
    spaced: bool,
    /// [`Terminator::Until`]: how many bytes of the marker match so far.
    ///
    /// Carried on the state rather than recomputed per token because a marker
    /// spans tokens — `</parameter>` is several BPE pieces — so a partial match
    /// at the end of one token must survive into the next.
    matched: usize,
    /// [`Terminator::JsonString`] / [`Terminator::JsonValue`]: the value's
    /// full JSON state. [`Terminator::JsonStringValue`] gains it at its opening
    /// quote, so `None` there means the value has not opened.
    json: Option<JsonLexer>,
}

/// The outcome of feeding one token's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Feed {
    /// The span continues.
    Continue,
    /// The terminator fired.  `consumed` bytes of this token belong to the span
    /// (for `JsonString`/`Balanced` this includes the closing delimiter; for
    /// `JsonNumber` it excludes the lookahead byte).  If `consumed <
    /// bytes.len()`, the close fell mid-token and the leftover bytes belong to
    /// the following node — the integration heals this (§7.3); standalone
    /// callers observe it via `consumed`.
    Close { consumed: usize },
    /// The token cannot be committed as written: commit `bytes` in its place.
    ///
    /// Either a byte had to be escaped (and the span goes on, `closed: false`),
    /// or a byte could not continue the value at all, in which case `bytes` is
    /// what was valid before it plus the text that completes the value, and the
    /// span has ended (`closed: true`). Empty `bytes` means the token
    /// contributes nothing — it is dropped.
    Rewrite { bytes: Vec<u8>, closed: bool },
}

impl TerminatorState {
    pub fn terminator(&self) -> Terminator {
        self.kind
    }

    /// Whether the value this span lexes has opened — its opening quote seen,
    /// for [`Terminator::JsonStringValue`]. Every other terminator's value is
    /// open from its first byte, so this is `true` for them unconditionally.
    pub fn opened(&self) -> bool {
        self.kind != Terminator::JsonStringValue || self.json.is_some()
    }

    /// Feed one token's decoded bytes.  Returns `Close` the moment the
    /// terminator fires, with the count of bytes that belong to the span.
    pub fn feed(&mut self, bytes: &[u8]) -> Feed {
        match self.kind {
            Terminator::JsonString | Terminator::JsonValue => self.feed_json(bytes),
            Terminator::JsonStringValue => self.feed_string_value(bytes),
            Terminator::JsonNumber { integer_only } => self.feed_number(bytes, integer_only),
            Terminator::Balanced { open, close } => self.feed_balanced(bytes, open, close),
            // No byte pattern ever closes this span — only a close token, EOS,
            // or the hard limit (all handled by the session, not the lexer).
            Terminator::Never => Feed::Continue,
            Terminator::Until { marker } => self.feed_until(bytes, marker.as_bytes()),
        }
    }

    /// The text that finishes the span as written so far — what the tree writes
    /// when the span ends **without its terminator firing**: an intercepted
    /// EOS, or the hard limit.
    ///
    /// # Why the grammar has to own its own closing text
    ///
    /// A consuming terminator (`JsonString`'s `"`, `Until`'s `</parameter>`)
    /// leaves its closing text in the output only because the *model* wrote
    /// it. That is fine on the path where the model reaches it and wrong on
    /// every other path: a span cut short by EOS closed with nothing, so a JSON
    /// string ran on unquoted and an element ran straight into whatever the
    /// tree emitted next.
    ///
    /// Measured live: a `reflect` whose last argument was cut short arrived as
    /// `<parameter=my_reflections>\ntext</function>`, and the argument was
    /// dropped — taking the two the character *had* written down with it, as
    /// "needed `my_reflections` and did not have it".
    ///
    /// For a JSON value the text depends on where it was cut — `"`, `]}`,
    /// ` null`, a missing digit — so it is computed from the lexer state rather
    /// than fixed per span. Empty when the span is already complete, and for a
    /// lookahead scalar or [`Terminator::Never`], which have nothing of their
    /// own to close.
    pub fn completion(&self) -> Vec<u8> {
        match (self.kind, &self.json) {
            (_, Some(json)) => json.completion(),
            // Cut short before its opening quote, a string value holds no
            // string at all: a lone `"` would open one that never closes, so it
            // is finished as a whole empty value.
            (Terminator::JsonStringValue, None) => match self.spaced {
                true => b"\"\"".to_vec(),
                false => b" \"\"".to_vec(),
            },
            (Terminator::Until { marker }, _) => marker.as_bytes().to_vec(),
            (Terminator::Balanced { close, .. }, _) => vec![close; self.depth.max(1) as usize],
            _ => Vec::new(),
        }
    }

    /// [`Terminator::JsonString`] / [`Terminator::JsonValue`]: validate each
    /// byte, and rewrite the token when it cannot be committed as written.
    fn feed_json(&mut self, bytes: &[u8]) -> Feed {
        let json = self
            .json
            .as_mut()
            .expect("a JSON terminator starts with its lexer");
        // `Some` from the first rewritten byte on: the bytes to commit.
        let mut written: Option<Vec<u8>> = None;
        for (i, &b) in bytes.iter().enumerate() {
            let mut step = json.step(b);
            if step.effect == Effect::TrailingComma {
                let mut out = written.take().unwrap_or_else(|| bytes[..i].to_vec());
                match out
                    .iter()
                    .rposition(|&c| !matches!(c, b' ' | b'\t' | b'\n' | b'\r'))
                {
                    // The separator is in this token, so it is not committed
                    // yet: take it back, with the whitespace after it, and the
                    // closer closes what came before.
                    Some(p) if out[p] == b',' => {
                        out.truncate(p);
                        json.retract_separator(out.last().copied());
                        written = Some(out);
                        step = json.step(b);
                    }
                    // Committed by an earlier token: it stays, and the value is
                    // completed after it.
                    _ => {
                        out.extend(json.completion());
                        return Feed::Rewrite {
                            bytes: out,
                            closed: true,
                        };
                    }
                }
            }
            match step.effect {
                Effect::Invalid => {
                    let mut out = written.unwrap_or_else(|| bytes[..i].to_vec());
                    out.extend(step.write.unwrap_or_default());
                    out.extend(json.completion());
                    return Feed::Rewrite {
                        bytes: out,
                        closed: true,
                    };
                }
                Effect::Delimiter => {
                    return match (written, step.write) {
                        (None, None) => Feed::Close { consumed: i },
                        // The delimiter is not committed with a rewrite: the
                        // successor writes it, or the model chooses it again.
                        (out, write) => {
                            let mut out = out.unwrap_or_else(|| bytes[..i].to_vec());
                            out.extend(write.unwrap_or_default());
                            Feed::Rewrite {
                                bytes: out,
                                closed: true,
                            }
                        }
                    };
                }
                Effect::Continue | Effect::Complete | Effect::TrailingComma => {}
            }
            match (step.write, written.as_mut()) {
                (Some(w), None) => {
                    let mut out = bytes[..i].to_vec();
                    out.extend(w);
                    written = Some(out);
                }
                (Some(w), Some(out)) => out.extend(w),
                (None, Some(out)) => out.push(b),
                (None, None) => {}
            }
            if step.effect == Effect::Complete {
                return match written {
                    None => Feed::Close { consumed: i + 1 },
                    Some(out) => Feed::Rewrite {
                        bytes: out,
                        closed: true,
                    },
                };
            }
        }
        match written {
            None => Feed::Continue,
            Some(out) => Feed::Rewrite {
                bytes: out,
                closed: false,
            },
        }
    }

    /// Raw bytes until `marker`, which is consumed. See [`Terminator::Until`].
    ///
    /// The rescan on a mismatch is the part worth reading. A naive
    /// implementation resets the match to zero, which loses a marker that
    /// overlaps its own failed prefix — for `</p</parameter>` the `</p` matches
    /// three bytes, the `<` that follows is not `a`, and dropping to zero skips
    /// past the real marker's opening `<`. So a failure retries the current
    /// byte against progressively shorter prefixes, which is the naive-but-
    /// correct search; the markers here are a dozen bytes and appear once, so
    /// the cost of a proper failure function would buy nothing measurable.
    fn feed_until(&mut self, bytes: &[u8], marker: &[u8]) -> Feed {
        if marker.is_empty() {
            return Feed::Continue;
        }
        for (i, &b) in bytes.iter().enumerate() {
            loop {
                if b == marker[self.matched] {
                    self.matched += 1;
                    if self.matched == marker.len() {
                        self.matched = 0;
                        // Consumed through this byte, as `JsonString` consumes
                        // its closing quote. Anything after it in this token
                        // belongs to the next node; the session heals that.
                        return Feed::Close { consumed: i + 1 };
                    }
                    break;
                }
                if self.matched == 0 {
                    break;
                }
                // Retry this byte against the next-shortest prefix that could
                // still be live.
                self.matched -= 1;
                let keep = self.matched;
                self.matched = (1..=keep)
                    .rev()
                    .find(|&n| marker[..n] == marker[keep + 1 - n..=keep])
                    .unwrap_or(0);
            }
        }
        Feed::Continue
    }

    /// See [`Terminator::JsonStringValue`]. Before the opening quote only
    /// whitespace is the value's; from the quote on, the value is an ordinary
    /// JSON string and [`JsonLexer::string`] lexes it exactly as it lexes a
    /// [`Terminator::JsonString`] — escapes rewritten, completion computed.
    fn feed_string_value(&mut self, bytes: &[u8]) -> Feed {
        if self.json.is_some() {
            return self.feed_json(bytes);
        }
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.spaced = true,
                b'"' => {
                    self.json = Some(JsonLexer::string());
                    // The rest of this token is string content, with its
                    // offsets and any rewrite shifted past the opening quote.
                    let head = i + 1;
                    return match self.feed_json(&bytes[head..]) {
                        Feed::Continue => Feed::Continue,
                        Feed::Close { consumed } => Feed::Close {
                            consumed: head + consumed,
                        },
                        Feed::Rewrite {
                            bytes: tail,
                            closed,
                        } => {
                            let mut out = bytes[..head].to_vec();
                            out.extend(tail);
                            Feed::Rewrite { bytes: out, closed }
                        }
                    };
                }
                // **The model skipped the value.** The token cannot stay —
                // nothing can be written in front of it once it is in the
                // sequence, and after it the call reads `"prefix":}}`, not
                // JSON — so it is replaced by the whitespace it opened with and
                // a whole empty string, and the model decides again at the
                // successor.
                _ => {
                    let mut out = bytes[..i].to_vec();
                    if !self.spaced {
                        out.push(b' ');
                    }
                    out.extend_from_slice(b"\"\"");
                    return Feed::Rewrite {
                        bytes: out,
                        closed: true,
                    };
                }
            }
        }
        Feed::Continue
    }

    fn feed_number(&mut self, bytes: &[u8], integer_only: bool) -> Feed {
        for (i, &b) in bytes.iter().enumerate() {
            if !is_number_byte(b, integer_only) {
                return Feed::Close { consumed: i };
            }
        }
        Feed::Continue
    }

    fn feed_balanced(&mut self, bytes: &[u8], open: u8, close: u8) -> Feed {
        for (i, &b) in bytes.iter().enumerate() {
            if self.in_string {
                // Inside a nested string: only escaping and the closing quote matter.
                if self.escaped {
                    self.escaped = false;
                } else if b == b'\\' {
                    self.escaped = true;
                } else if b == b'"' {
                    self.in_string = false;
                }
                continue;
            }
            match b {
                b'"' => self.in_string = true,
                _ if b == open => {
                    self.depth += 1;
                    self.started = true;
                }
                _ if b == close => {
                    // Guard against a stray close before any open.
                    self.depth = self.depth.saturating_sub(1);
                    if self.started && self.depth == 0 {
                        return Feed::Close { consumed: i + 1 };
                    }
                }
                _ => {}
            }
        }
        Feed::Continue
    }
}

fn is_number_byte(b: u8, integer_only: bool) -> bool {
    match b {
        b'0'..=b'9' | b'-' | b'+' => true,
        b'.' | b'e' | b'E' => !integer_only,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(term: Terminator, chunks: &[&[u8]]) -> (usize, Option<usize>) {
        // Returns (chunk index that closed, consumed) or (n, None) if never.
        let mut st = term.start();
        for (i, c) in chunks.iter().enumerate() {
            if let Feed::Close { consumed } = st.feed(c) {
                return (i, Some(consumed));
            }
        }
        (chunks.len(), None)
    }

    // ── JsonStringValue ─────────────────────────────────────────────────────

    /// The empty string as Qwen spells it — ` ""`, one token — closes on its
    /// second quote. This is the case the prefilled opening quote made
    /// unreachable.
    #[test]
    fn string_value_empty_as_one_token() {
        assert_eq!(run(Terminator::JsonStringValue, &[b" \"\""]), (0, Some(3)));
    }

    /// A non-empty string: ` "` then content, closing on a quote merged with
    /// the call's own closers — the tail belongs to the next node.
    #[test]
    fn string_value_nonempty_with_merged_close() {
        assert_eq!(
            run(Terminator::JsonStringValue, &[b" \"", b"src", b"\"}}"]),
            (2, Some(1))
        );
    }

    /// The opening quote is not the close: a span that ended on it would read
    /// every string as empty.
    #[test]
    fn string_value_opening_quote_does_not_close() {
        assert_eq!(
            run(Terminator::JsonStringValue, &[b" \"", b"a\\\"b", b"\""]),
            (2, Some(1))
        );
    }

    /// **Anything but whitespace before the opening quote is a skipped value.**
    /// The failure this terminator exists for — `}}` where the value belonged
    /// — is replaced by a whole empty string instead of being swallowed, with
    /// the call's own close, as the contents of one. Bare text is as unopened
    /// as a delimiter: let through, its first later quote would open the string
    /// and swallow the call's close.
    #[test]
    fn string_value_skipped_before_open_is_written_empty() {
        let cases: [(&[u8], &[u8]); 5] = [
            (b"}}", b" \"\""),
            (b" ,", b" \"\""),
            (b"]", b" \"\""),
            (b"src", b" \"\""),
            (b" src/main.rs", b" \"\""),
        ];
        for (chunk, written) in cases {
            let mut st = Terminator::JsonStringValue.start();
            assert_eq!(
                st.feed(chunk),
                Feed::Rewrite {
                    bytes: written.to_vec(),
                    closed: true
                },
                "{chunk:?}"
            );
        }
        // Whitespace committed by an earlier token is not doubled.
        let mut st = Terminator::JsonStringValue.start();
        assert_eq!(st.feed(b" "), Feed::Continue);
        assert_eq!(
            st.feed(b"}}"),
            Feed::Rewrite {
                bytes: b"\"\"".to_vec(),
                closed: true
            }
        );
        // Once the quote is out the value is open, and a delimiter is content.
        let mut st = Terminator::JsonStringValue.start();
        assert_eq!(st.feed(b" \"}}"), Feed::Continue);
        assert!(st.opened());
    }

    /// Cut short before its quote, the value completes as a whole empty
    /// string; after it, as the string lexer completes any open string.
    #[test]
    fn string_value_completion_depends_on_whether_it_opened() {
        let st = Terminator::JsonStringValue.start();
        assert!(!st.opened());
        assert_eq!(st.completion(), b" \"\"");
        let mut st = Terminator::JsonStringValue.start();
        assert_eq!(st.feed(b" \"ab"), Feed::Continue);
        assert_eq!(st.completion(), b"\"");
    }

    /// Delimiters INSIDE the string are content, as in any JSON string.
    #[test]
    fn string_value_delimiters_inside_are_content() {
        assert_eq!(
            run(Terminator::JsonStringValue, &[b" \"a,}b]", b"\""]),
            (1, Some(1))
        );
    }

    // ── JsonString ──────────────────────────────────────────────────────────

    #[test]
    fn string_plain_close() {
        assert_eq!(run(Terminator::JsonString, &[b"abc", b"\""]), (1, Some(1)));
    }

    #[test]
    fn string_close_mid_token() {
        // The closing quote is inside a token that also carries trailing bytes.
        assert_eq!(run(Terminator::JsonString, &[b"ab\"}"]), (0, Some(3)));
    }

    #[test]
    fn string_escaped_quote_does_not_close() {
        // \"  then a real close.
        assert_eq!(
            run(Terminator::JsonString, &[b"a\\\"b", b"\""]),
            (1, Some(1))
        );
    }

    #[test]
    fn string_escaped_backslash_then_quote_closes() {
        // \\  consumes the backslash pair; the following " IS a real close.
        assert_eq!(run(Terminator::JsonString, &[b"a\\\\\""]), (0, Some(4)));
    }

    #[test]
    fn string_escape_split_across_tokens() {
        // backslash ends token 0; quote starts token 1 — escaped, no close;
        // then a real close.
        assert_eq!(
            run(Terminator::JsonString, &[b"a\\", b"\"b", b"\""]),
            (2, Some(1))
        );
    }

    #[test]
    fn string_utf8_value() {
        assert_eq!(
            run(Terminator::JsonString, &["日本語".as_bytes(), b"\""]),
            (1, Some(1))
        );
    }

    #[test]
    fn string_never_closes() {
        assert_eq!(run(Terminator::JsonString, &[b"abc", b"def"]), (2, None));
    }

    // ── JsonNumber ──────────────────────────────────────────────────────────

    #[test]
    fn number_lookahead_terminator_not_consumed() {
        // "123" then ',' — closes at the comma, consuming 0 of that token.
        assert_eq!(
            run(
                Terminator::JsonNumber {
                    integer_only: false
                },
                &[b"123", b","]
            ),
            (1, Some(0))
        );
    }

    #[test]
    fn number_float_parts() {
        let t = Terminator::JsonNumber {
            integer_only: false,
        };
        assert_eq!(run(t, &[b"-1.5e-3", b"}"]), (1, Some(0)));
    }

    #[test]
    fn number_integer_only_stops_at_dot() {
        let t = Terminator::JsonNumber { integer_only: true };
        // "12" then ".5" — '.' is not an integer byte → close at index 0 of ".5".
        assert_eq!(run(t, &[b"12", b".5"]), (1, Some(0)));
    }

    #[test]
    fn number_terminator_mid_token() {
        let t = Terminator::JsonNumber {
            integer_only: false,
        };
        // "12}" — close at the '}', consuming the "12".
        assert_eq!(run(t, &[b"12}"]), (0, Some(2)));
    }

    // ── Balanced ────────────────────────────────────────────────────────────

    fn braces() -> Terminator {
        Terminator::Balanced {
            open: b'{',
            close: b'}',
        }
    }

    #[test]
    fn balanced_simple() {
        assert_eq!(run(braces(), &[b"{", b"a", b"}"]), (2, Some(1)));
    }

    #[test]
    fn balanced_nested() {
        assert_eq!(run(braces(), &[b"{a{b}c}"]), (0, Some(7)));
    }

    #[test]
    fn balanced_brace_inside_string_ignored() {
        // The '}' inside the string must not close; the final real '}' does.
        assert_eq!(run(braces(), &[b"{\"x}y\"}"]), (0, Some(7)));
    }

    #[test]
    fn balanced_escaped_quote_inside_string() {
        // 8 bytes: { " a \ " } " }  — the \" stays in the string, the } inside
        // it is ignored, the string closes at the second ", then the outer } (the
        // 8th byte) closes.
        assert_eq!(run(braces(), &[b"{\"a\\\"}\"}"]), (0, Some(8)));
    }

    #[test]
    fn balanced_brackets() {
        let t = Terminator::Balanced {
            open: b'[',
            close: b']',
        };
        assert_eq!(run(t, &[b"[1,[2],3]"]), (0, Some(9)));
    }

    #[test]
    fn balanced_split_across_tokens() {
        assert_eq!(
            run(braces(), &[b"{", b"\"k\":", b"[", b"]", b"}"]),
            (4, Some(1))
        );
    }

    // ── JsonValue (any value, lookahead at top-level , or }) ─────────────────

    #[test]
    fn value_scalar_then_comma() {
        // "42" then ',' (separator) — lookahead, consumes 0 of the ',' token.
        assert_eq!(run(Terminator::JsonValue, &[b"42", b","]), (1, Some(0)));
    }

    #[test]
    fn value_scalar_then_close() {
        assert_eq!(run(Terminator::JsonValue, &[b"true", b"}"]), (1, Some(0)));
    }

    #[test]
    fn value_array_with_inner_commas() {
        // [1,2,3] — inner commas are at depth 1; the array's own ] completes it.
        assert_eq!(
            run(Terminator::JsonValue, &[b"[1,2,3]", b"}"]),
            (0, Some(7))
        );
    }

    #[test]
    fn value_object_then_close() {
        // {"k":1} — the value's own } (depth 1->0) completes it and is consumed.
        assert_eq!(
            run(Terminator::JsonValue, &[b"{\"k\":1}", b"}"]),
            (0, Some(7))
        );
    }

    #[test]
    fn value_nested_object_and_array() {
        assert_eq!(
            run(
                Terminator::JsonValue,
                &[b"{\"a\":[1,{\"b\":2}],\"c\":3}", b","]
            ),
            (0, Some(23))
        );
    }

    #[test]
    fn value_string_with_comma_and_brace_inside() {
        // A top-level string value: the , and } inside it are ignored, and its
        // closing quote completes it.
        assert_eq!(
            run(Terminator::JsonValue, &[b"\"a,b}c\"", b","]),
            (0, Some(7))
        );
    }

    /// The live failure: the model closed its array and wrote another `]`.
    /// The span ends on the array's own bracket, so the stray one never gets
    /// into the value — the grammar writes what follows.
    #[test]
    fn value_array_is_complete_at_its_own_close() {
        assert_eq!(
            run(Terminator::JsonValue, &[b" [\"dir\"]", b"]", b"}"]),
            (0, Some(8))
        );
        // A stray bracket in the same token is left over for the next node.
        assert_eq!(run(Terminator::JsonValue, &[b"[1]]}"]), (0, Some(3)));
    }

    /// A `]` at depth 0 closes nothing the value opened: it ends a bare scalar
    /// as lookahead instead of being absorbed into it.
    #[test]
    fn value_stray_close_bracket_ends_a_scalar() {
        assert_eq!(run(Terminator::JsonValue, &[b"42", b"]"]), (1, Some(0)));
    }

    #[test]
    fn value_delimiter_mid_token() {
        // "42}" as one token: closes at the '}', consuming the "42".
        assert_eq!(run(Terminator::JsonValue, &[b"42}"]), (0, Some(2)));
    }

    /// A token that cannot be committed as written comes back as the bytes to
    /// commit instead — escaped in place while the span goes on, or completed
    /// and closed at the first byte that cannot continue it.
    #[test]
    fn a_token_that_breaks_the_value_is_rewritten() {
        let mut st = Terminator::JsonString.start();
        assert_eq!(st.feed(b"ab"), Feed::Continue);
        assert_eq!(
            st.feed(b"c\nd"),
            Feed::Rewrite {
                bytes: b"c\\nd".to_vec(),
                closed: false
            }
        );
        // Escaped earlier in the token, closed later in it: one rewrite.
        assert_eq!(
            st.feed(b"\te\"]"),
            Feed::Rewrite {
                bytes: b"\\te\"".to_vec(),
                closed: true
            }
        );

        let mut st = Terminator::JsonValue.start();
        assert_eq!(st.feed(b" [1, [2"), Feed::Continue);
        assert_eq!(
            st.feed(b"3}"),
            Feed::Rewrite {
                bytes: b"3]]".to_vec(),
                closed: true
            }
        );

        // Nothing written, nothing to complete with: the token is dropped.
        let mut st = Terminator::JsonValue.start();
        assert_eq!(st.feed(b" 5 "), Feed::Continue);
        assert_eq!(
            st.feed(b"6"),
            Feed::Rewrite {
                bytes: Vec::new(),
                closed: true
            }
        );
    }

    /// What a span cut short writes: computed from where the JSON was cut, the
    /// marker for a raw value, nothing for a span with no closing text.
    #[test]
    fn a_cut_span_completes_from_where_it_was_cut() {
        let mut st = Terminator::JsonValue.start();
        st.feed(b" {\"a\": [\"x");
        assert_eq!(st.completion(), b"\"]}");

        let mut st = Terminator::JsonString.start();
        st.feed(b"abc\\");
        assert_eq!(st.completion(), b"\\\"");

        let st = Terminator::Until { marker: "</p>" }.start();
        assert_eq!(st.completion(), b"</p>");
        assert!(Terminator::Never.start().completion().is_empty());

        let mut st = Terminator::JsonValue.start();
        st.feed(b" 42");
        assert!(
            st.completion().is_empty(),
            "a complete scalar needs nothing"
        );
    }

    // ── Never (token-closed span; bytes never close it) ──────────────────────

    #[test]
    fn never_does_not_close_on_any_bytes() {
        // Quotes, braces, commas, EOS-looking bytes — none close a Never span.
        assert_eq!(
            run(Terminator::Never, &[b"\"}],", b"abc", b"</think>"]),
            (3, None)
        );
    }

    #[test]
    fn never_is_not_lookahead() {
        assert!(!Terminator::Never.is_lookahead());
    }

    // ── Until: raw text to a literal marker ─────────────────────────────────

    const PARAM: Terminator = Terminator::Until {
        marker: "\n</parameter>",
    };

    /// Feed `chunks` in order; give back where it closed and how much of that
    /// chunk the span took. Chunks stand in for tokens, which is the whole
    /// point — a marker is several BPE pieces and the match has to survive the
    /// boundaries between them.
    fn feed_all(t: Terminator, chunks: &[&str]) -> Option<(usize, usize)> {
        let mut st = t.start();
        for (n, c) in chunks.iter().enumerate() {
            if let Feed::Close { consumed } = st.feed(c.as_bytes()) {
                return Some((n, consumed));
            }
        }
        None
    }

    #[test]
    fn until_closes_on_the_marker_and_consumes_it() {
        assert_eq!(feed_all(PARAM, &["hello\n</parameter>"]), Some((0, 18)));
        assert!(
            !PARAM.is_lookahead(),
            "the marker is consumed, not pushed back"
        );
    }

    /// **The property the whole thing turns on.** `</parameter>` is not a
    /// special token in any vocabulary this targets, so it arrives in pieces —
    /// a matcher that reset between tokens would never fire.
    #[test]
    fn until_matches_a_marker_split_across_tokens() {
        assert_eq!(
            feed_all(PARAM, &["hello", "\n</", "param", "eter>"]),
            Some((3, 5))
        );
        // Byte at a time is the same answer — the state is what carries it.
        let split: Vec<String> = "x\n</parameter>".chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = split.iter().map(|s| s.as_str()).collect();
        assert!(feed_all(PARAM, &refs).is_some());
    }

    /// The value is raw text, so what would end a JSON string is ordinary
    /// content here. This is the reason `JsonString` could not be reused.
    #[test]
    fn until_does_not_close_on_quotes_backslashes_or_newlines() {
        let prose = "she said \"no\" \\ and left\nthen came back\n";
        assert_eq!(
            feed_all(PARAM, &[prose]),
            None,
            "closed early on JSON syntax"
        );
        assert!(feed_all(PARAM, &[prose, "\n</parameter>"]).is_some());
    }

    /// A near-miss that overlaps the real marker's opening byte. Resetting the
    /// match to zero on a mismatch loses this — the `<` that begins the true
    /// marker is skipped, and the span never closes.
    #[test]
    fn until_recovers_from_a_false_start_that_overlaps_the_marker() {
        assert!(
            feed_all(PARAM, &["a\n</p\n</parameter>"]).is_some(),
            "a failed partial match swallowed the marker that followed it"
        );
        assert!(feed_all(PARAM, &["\n</paramX\n</parameter>"]).is_some());
    }

    /// Bytes after the marker in the same token belong to the next node, and
    /// `consumed` is what says so.
    #[test]
    fn until_reports_only_the_bytes_that_belong_to_the_span() {
        let (chunk, consumed) =
            feed_all(PARAM, &["v\n</parameter>\n<parameter=next>"]).expect("closed");
        assert_eq!(chunk, 0);
        assert_eq!(consumed, "v\n</parameter>".len());
    }
}
