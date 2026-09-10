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

    /// The text this terminator **consumes** when it fires — and therefore the
    /// text the grammar has to write itself if the span ends any other way.
    ///
    /// A consuming terminator leaves its delimiter in the output only because
    /// the model wrote it. That holds on the path where the model reaches the
    /// delimiter and nowhere else: a span cut short by an intercepted EOS
    /// closed with nothing, so a JSON string ran on unquoted and an element ran
    /// into the next tag. Both make the whole call unreadable, which costs the
    /// arguments the model *did* finish as well as the one it did not.
    ///
    /// `None` for the lookahead terminators, whose delimiter belongs to the
    /// successor and is emitted by it regardless, and for [`Terminator::Never`],
    /// which has no delimiter of its own.
    pub fn consumed_close(self) -> Option<String> {
        match self {
            Terminator::JsonString => Some("\"".to_string()),
            Terminator::Balanced { close, .. } => Some((close as char).to_string()),
            Terminator::Until { marker } => Some(marker.to_string()),
            Terminator::JsonNumber { .. } | Terminator::JsonValue | Terminator::Never => None,
        }
    }

    pub fn start(self) -> TerminatorState {
        TerminatorState {
            kind: self,
            depth: 0,
            in_string: false,
            escaped: false,
            started: false,
            matched: 0,
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
    /// [`Terminator::Until`]: how many bytes of the marker match so far.
    ///
    /// Carried on the state rather than recomputed per token because a marker
    /// spans tokens — `</parameter>` is several BPE pieces — so a partial match
    /// at the end of one token must survive into the next.
    matched: usize,
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
            Terminator::Until { marker } => self.feed_until(bytes, marker.as_bytes()),
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
                // A `,` at depth 0 separates this field from the next —
                // lookahead, not consumed.
                b',' if self.depth == 0 => return Feed::Close { consumed: i },
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
