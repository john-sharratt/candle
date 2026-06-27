//! Free-text terminators — the escape- and nesting-aware byte lexer that ends a
//! free-text span.
//!
//! It runs over the decoded **bytes** of each token (not token identity), which
//! is what makes it robust to however the tokenizer happened to chunk a value.
//! State (`escaped` / `in_string` / `depth`) carries across `feed` calls so a
//! span that spans many tokens is lexed correctly.

/// What ends a free-text span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminator {
    /// A JSON string value: ends at the first UNESCAPED `"`.  A `\` escapes the
    /// next byte, so `\"` and `\\` are handled.  The closing quote is consumed.
    JsonString,
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
    /// the value of an object field: lookahead-terminated at the first `,` or
    /// `}` seen at the ENCLOSING object's depth (depth 0), respecting nested
    /// `[]`/`{}` and strings.  The delimiter is NOT consumed — it belongs to the
    /// following node.  Guarantees a structurally-valid JSON value without
    /// enforcing its scalar type.
    JsonValue,
    /// No byte delimiter at all: `feed` always returns `Continue`.  The span
    /// ends only via a close *token* (`FreeTextSpan::close_token`), an EOS
    /// sample (`eos_ends`), or the hard `forced_after` limit.  Used by the
    /// thinking-block steering tree, whose `</think>` close is a token, not a
    /// byte pattern.
    Never,
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
}

/// The outcome of feeding one token's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl TerminatorState {
    pub fn terminator(&self) -> Terminator {
        self.kind
    }

    /// Feed one token's decoded bytes.  Returns `Close` the moment the
    /// terminator fires, with the count of bytes that belong to the span.
    pub fn feed(&mut self, bytes: &[u8]) -> Feed {
        match self.kind {
            Terminator::JsonString => self.feed_json_string(bytes),
            Terminator::JsonNumber { integer_only } => self.feed_number(bytes, integer_only),
            Terminator::Balanced { open, close } => self.feed_balanced(bytes, open, close),
            Terminator::JsonValue => self.feed_value(bytes),
            // No byte pattern ever closes this span — only a close token, EOS,
            // or the hard limit (all handled by the session, not the lexer).
            Terminator::Never => Feed::Continue,
        }
    }

    fn feed_value(&mut self, bytes: &[u8]) -> Feed {
        for (i, &b) in bytes.iter().enumerate() {
            if self.in_string {
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
                b'[' | b'{' => self.depth += 1,
                b']' => self.depth = self.depth.saturating_sub(1),
                b'}' => {
                    // At depth 0 this `}` is the ENCLOSING object's close (the
                    // value is complete) — lookahead, not consumed.
                    if self.depth == 0 {
                        return Feed::Close { consumed: i };
                    }
                    self.depth -= 1;
                }
                b',' => {
                    // A `,` at depth 0 separates this field from the next —
                    // lookahead, not consumed.
                    if self.depth == 0 {
                        return Feed::Close { consumed: i };
                    }
                }
                _ => {}
            }
        }
        Feed::Continue
    }

    fn feed_json_string(&mut self, bytes: &[u8]) -> Feed {
        for (i, &b) in bytes.iter().enumerate() {
            if self.escaped {
                self.escaped = false;
            } else if b == b'\\' {
                self.escaped = true;
            } else if b == b'"' {
                return Feed::Close { consumed: i + 1 };
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
        // [1,2,3] — inner commas are at depth 1; closes only on the outer }.
        assert_eq!(
            run(Terminator::JsonValue, &[b"[1,2,3]", b"}"]),
            (1, Some(0))
        );
    }

    #[test]
    fn value_object_then_close() {
        // {"k":1} then the enclosing } — the value's own } is depth 1->0.
        assert_eq!(
            run(Terminator::JsonValue, &[b"{\"k\":1}", b"}"]),
            (1, Some(0))
        );
    }

    #[test]
    fn value_nested_object_and_array() {
        assert_eq!(
            run(
                Terminator::JsonValue,
                &[b"{\"a\":[1,{\"b\":2}],\"c\":3}", b","]
            ),
            (1, Some(0))
        );
    }

    #[test]
    fn value_string_with_comma_and_brace_inside() {
        // A top-level string value: the , and } inside it must be ignored.
        assert_eq!(
            run(Terminator::JsonValue, &[b"\"a,b}c\"", b","]),
            (1, Some(0))
        );
    }

    #[test]
    fn value_delimiter_mid_token() {
        // "42}" as one token: closes at the '}', consuming the "42".
        assert_eq!(run(Terminator::JsonValue, &[b"42}"]), (0, Some(2)));
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
}
