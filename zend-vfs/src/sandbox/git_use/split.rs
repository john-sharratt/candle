//! Cutting a script into the commands it runs.
//!
//! A script is cut at every operator outside quotes — `;` `&` `|`, newlines,
//! brackets — and at each command substitution (`$(…)`, and sh's backticks),
//! which opens a command even inside double quotes. What follows a
//! substitution's close is the rest of the command it sits in
//! (`echo $(date) git`), not a command, and is dropped. Every construct that
//! looks like an operator but holds words is kept whole as text:
//!
//! | Dialect | Text, not commands |
//! |---|---|
//! | all | quotes; the dialect's escape (`\` `^` `` ` ``) and what it escapes; `&` in a redirection (`2>&1`, `&>log`) |
//! | sh | `${…}`, `$((…))` and `((…))` arithmetic, `name=(…)` arrays, `name()`, brace expansion `{a,b}`, `case` patterns, `#` comments, here-document bodies |
//! | cmd | `;` and braces (not operators in cmd), the brackets `echo`/`set`/`title` print, a `for` set — save `for /f`'s quoted command, which runs — and `rem`/`::` comments |
//! | PowerShell | `${…}`, `@{…}` tables, what stands before a `{ }` block (a keyword, a `switch` pattern, a function's name), `#` and `<# #>` comments |

use super::Dialect;

/// `script`, read as `dialect`, cut into the pieces that start a command.
pub(super) fn commands(script: &str, dialect: Dialect) -> Vec<String> {
    let mut splitter = Splitter {
        chars: script.chars().collect(),
        i: 0,
        dialect,
        out: Vec::new(),
        current: String::new(),
        starts: true,
        single: false,
        double: false,
        open: Vec::new(),
        backtick: None,
        heredocs: Vec::new(),
        cases: 0,
        pattern_next: false,
    };
    splitter.run();
    splitter.out
}

struct Splitter {
    chars: Vec<char>,
    i: usize,
    dialect: Dialect,
    out: Vec<String>,
    current: String,
    /// Whether `current` starts a command.
    starts: bool,
    single: bool,
    double: bool,
    /// For each open `$(`, whether it was opened inside double quotes — what
    /// its `)` goes back to.
    open: Vec<bool>,
    /// An open backtick, and the same.
    backtick: Option<bool>,
    /// Here-documents whose bodies begin at the next line: delimiter, and
    /// whether leading tabs are stripped (`<<-`).
    heredocs: Vec<(String, bool)>,
    /// sh `case` statements open.
    cases: usize,
    /// Whether the next piece is a `case` pattern (after `in`, or `;;`).
    pattern_next: bool,
}

impl Splitter {
    fn peek(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.i + ahead).copied()
    }

    fn push(&mut self, c: char) {
        self.current.push(c);
    }

    /// End the current piece; the next starts a command when `next_starts`.
    fn cut(&mut self, next_starts: bool) {
        let piece = std::mem::take(&mut self.current);
        if self.starts {
            if self.dialect == Dialect::Posix {
                if is_case_header(&piece) {
                    self.cases += 1;
                    self.pattern_next = true;
                } else if first_word(&piece) == "esac" {
                    self.cases = self.cases.saturating_sub(1);
                    self.pattern_next = false;
                }
            }
            self.out.push(piece);
        }
        self.starts = next_starts;
    }

    /// Drop the current piece unread — it is not a command — and start the
    /// next.
    fn drop_piece(&mut self) {
        self.current.clear();
        self.starts = true;
    }

    /// Take the text from `self.i`, which opens with `open`, through its
    /// matching `close`, as text.
    fn balanced(&mut self, open: char, close: char) {
        let mut depth = 0usize;
        while let Some(c) = self.peek(0) {
            self.push(c);
            self.i += 1;
            if c == open {
                depth += 1;
            } else if c == close {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return;
                }
            }
        }
    }

    /// Whether the text so far is an sh `case` pattern — not once the word
    /// read is `esac`, which ends the statement instead.
    fn in_pattern(&self) -> bool {
        self.dialect == Dialect::Posix
            && ((self.pattern_next && first_word(&self.current) != "esac")
                || is_case_header(&self.current))
    }

    fn at_word_start(&self) -> bool {
        self.current.chars().last().is_none_or(char::is_whitespace)
    }

    fn run(&mut self) {
        let dialect = self.dialect;
        while let Some(c) = self.peek(0) {
            let next = self.peek(1);
            if self.single {
                self.single = c != '\'';
                self.push(c);
                self.i += 1;
                continue;
            }
            if c == dialect.escape() && !(dialect == Dialect::Cmd && self.double) {
                self.push(c);
                if let Some(n) = next {
                    self.push(n);
                }
                self.i += 2;
                continue;
            }
            if dialect != Dialect::Cmd && c == '$' && next == Some('{') {
                self.push(c);
                self.i += 1;
                self.balanced('{', '}');
                continue;
            }
            if dialect != Dialect::Cmd && c == '$' && next == Some('(') {
                if dialect == Dialect::Posix && self.peek(2) == Some('(') {
                    // `$((…))` arithmetic.
                    self.push(c);
                    self.i += 1;
                    self.balanced('(', ')');
                    continue;
                }
                self.open.push(self.double);
                self.double = false;
                self.cut(true);
                self.i += 2;
                continue;
            }
            if dialect == Dialect::Posix && c == '`' {
                match self.backtick.take() {
                    Some(was) => {
                        self.double = was;
                        self.cut(false);
                    }
                    None => {
                        self.backtick = Some(self.double);
                        self.double = false;
                        self.cut(true);
                    }
                }
                self.i += 1;
                continue;
            }
            if self.double {
                // Everything else in double quotes is text.
                self.double = c != '"';
                self.push(c);
                self.i += 1;
                continue;
            }
            if self.comment(c) {
                while self.peek(0).is_some_and(|c| c != '\n') {
                    self.i += 1;
                }
                continue;
            }
            if dialect == Dialect::PowerShell && c == '<' && next == Some('#') {
                self.i += 2;
                while self.peek(0).is_some()
                    && !(self.peek(0) == Some('#') && self.peek(1) == Some('>'))
                {
                    self.i += 1;
                }
                self.i += 2;
                continue;
            }
            if dialect == Dialect::Posix
                && c == '<'
                && next == Some('<')
                && self.peek(2) != Some('<')
            {
                self.heredoc();
                continue;
            }
            self.i += 1;
            self.operator(c, next);
        }
        self.cut(true);
    }

    /// Whether `c` opens a comment, which runs to the end of the line.
    fn comment(&self, c: char) -> bool {
        match self.dialect {
            Dialect::Posix | Dialect::PowerShell => c == '#' && self.at_word_start(),
            Dialect::Cmd => {
                self.current.trim().trim_start_matches('@').is_empty()
                    && is_cmd_comment(&self.chars[self.i..])
            }
        }
    }

    /// `<<DELIM` (or `<<-DELIM`): its body, from the next line, is skipped at
    /// that line's end.
    fn heredoc(&mut self) {
        self.i += 2;
        let strip_tabs = self.peek(0) == Some('-');
        if strip_tabs {
            self.i += 1;
        }
        while self.peek(0).is_some_and(|c| c == ' ' || c == '\t') {
            self.i += 1;
        }
        let mut delimiter = String::new();
        while let Some(d) = self.peek(0) {
            if d.is_whitespace() || ";&|<>()".contains(d) {
                break;
            }
            if d != '\'' && d != '"' && d != '\\' {
                delimiter.push(d);
            }
            self.i += 1;
        }
        self.heredocs.push((delimiter, strip_tabs));
        self.current.push_str("<<");
    }

    /// Skip the bodies of the here-documents opened on the line just ended.
    fn heredoc_bodies(&mut self) {
        for (delimiter, strip_tabs) in std::mem::take(&mut self.heredocs) {
            while self.i < self.chars.len() {
                let end = self.chars[self.i..]
                    .iter()
                    .position(|c| *c == '\n')
                    .map_or(self.chars.len(), |n| self.i + n);
                let line: String = self.chars[self.i..end].iter().collect();
                self.i = end + 1;
                let line = line.trim_end_matches('\r');
                let line = if strip_tabs {
                    line.trim_start_matches('\t')
                } else {
                    line
                };
                if line == delimiter {
                    break;
                }
            }
        }
    }

    /// `c`, just consumed, outside any quote.
    fn operator(&mut self, c: char, next: Option<char>) {
        let dialect = self.dialect;
        match c {
            '\'' if dialect != Dialect::Cmd => {
                self.single = true;
                self.push(c);
            }
            '"' => {
                self.double = true;
                self.push(c);
            }
            '(' => self.open_bracket(next),
            ')' => {
                if let Some(was) = self.open.pop() {
                    self.double = was;
                    self.cut(false);
                } else if self.in_pattern() {
                    if is_case_header(&self.current) {
                        self.cases += 1;
                    }
                    self.pattern_next = false;
                    self.drop_piece();
                } else {
                    // A group's close: an operator, or `else (…)`, comes next.
                    self.cut(true);
                }
            }
            // A redirection's `&` — `2>&1`, `&>log` — is not an operator.
            '&' if matches!(self.current.chars().last(), Some('>' | '<')) || next == Some('>') => {
                self.push(c);
            }
            // PowerShell's call operator, where a command starts.
            '&' if dialect == Dialect::PowerShell
                && self.current.trim().is_empty()
                && next != Some('&')
                && self.chars.get(self.i.wrapping_sub(2)) != Some(&'&') =>
            {
                self.current.push_str("& ");
            }
            '|' if self.in_pattern() => self.push(c),
            ';' if dialect == Dialect::Cmd => self.push(c),
            ';' => {
                let ends_case_item = next == Some(';') || next == Some('&');
                if ends_case_item {
                    self.i += 1;
                    if self.peek(0) == Some('&') {
                        self.i += 1;
                    }
                }
                self.cut(true);
                if ends_case_item {
                    self.pattern_next = self.cases > 0;
                }
            }
            '{' => match dialect {
                Dialect::Cmd => self.push(c),
                // A group is `{` as a word; `{a,b}` is brace expansion.
                Dialect::Posix if next.is_none_or(char::is_whitespace) => self.cut(true),
                Dialect::Posix => self.push(c),
                Dialect::PowerShell if self.current.ends_with('@') => {
                    self.i -= 1;
                    self.balanced('{', '}');
                }
                // What stands before a block is a keyword, a condition's
                // remains, a switch pattern or a function's name — the block
                // holds the commands.
                Dialect::PowerShell => self.drop_piece(),
            },
            '}' => match dialect {
                Dialect::Cmd => self.push(c),
                Dialect::Posix if self.current.trim().is_empty() || self.at_word_start() => {
                    self.cut(true)
                }
                Dialect::Posix => self.push(c),
                Dialect::PowerShell => self.cut(true),
            },
            '\n' => {
                self.cut(true);
                self.heredoc_bodies();
            }
            '&' | '|' | '\r' => self.cut(true),
            _ => self.push(c),
        }
    }

    /// `(`, just consumed.
    fn open_bracket(&mut self, next: Option<char>) {
        match self.dialect {
            // cmd prints the brackets `echo`, `set` and `title` are given.
            Dialect::Cmd if prints_brackets(&self.current) => self.push('('),
            // `for … in (set) do …`: the set is a list, not commands — save
            // `for /f`'s quoted command, which runs.
            Dialect::Cmd if cmd_first_word(&self.current) == "for" => {
                let close = self.chars[self.i..]
                    .iter()
                    .position(|c| *c == ')')
                    .map_or(self.chars.len(), |n| self.i + n);
                let set: String = self.chars[self.i..close].iter().collect();
                self.cut(true);
                let set = set.trim();
                let quoted = set
                    .strip_prefix('\'')
                    .and_then(|s| s.strip_suffix('\''))
                    .or_else(|| set.strip_prefix('`').and_then(|s| s.strip_suffix('`')));
                if let Some(command) = quoted {
                    self.current.push_str(command);
                    self.cut(true);
                }
                self.i = close + 1;
            }
            // A `case` pattern's optional opening bracket.
            Dialect::Posix if self.in_pattern() => {}
            // `name()`: a function's definition.
            Dialect::Posix if next == Some(')') => {
                self.push('(');
                self.push(')');
                self.i += 1;
            }
            // `name=(…)` arrays and `((…))` arithmetic.
            Dialect::Posix if self.current.ends_with('=') || next == Some('(') => {
                self.i -= 1;
                self.balanced('(', ')');
            }
            _ => self.cut(true),
        }
    }
}

/// `piece`'s first word.
fn first_word(piece: &str) -> &str {
    piece.split_whitespace().next().unwrap_or("")
}

/// Whether `piece` opens an sh `case`: `case WORD in`, patterns perhaps
/// following.
fn is_case_header(piece: &str) -> bool {
    let mut words = piece.split_whitespace();
    words.next() == Some("case") && words.skip(1).any(|w| w == "in")
}

/// Whether the cmd command so far is one whose brackets are text.
fn prints_brackets(current: &str) -> bool {
    matches!(cmd_first_word(current).as_str(), "echo" | "set" | "title")
}

/// The first word of the cmd command so far, lower-cased, past a leading `@`.
fn cmd_first_word(current: &str) -> String {
    current
        .trim_start()
        .trim_start_matches('@')
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Whether `rest` opens a cmd comment: `rem` as a word, or `::`.
fn is_cmd_comment(rest: &[char]) -> bool {
    let head: String = rest.iter().take(4).collect::<String>().to_ascii_lowercase();
    head.starts_with("::")
        || (head.starts_with("rem") && head.chars().nth(3).is_none_or(char::is_whitespace))
}
