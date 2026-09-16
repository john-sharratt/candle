//! Unified-diff patch engine behind [`file_edit`](super::edit).
//!
//! # The format
//!
//! A patch is a unified-diff *body*: one or more hunks, each opened by an
//! `@@ -old_start[,old_count] +new_start[,new_count] @@` header and followed by
//! its lines — `' '` context, `'-'` removed, `'+'` added. Everything ahead of
//! the first hunk is a file header and is skipped, whether that is `---`/`+++`
//! or the `diff --git` and `index` lines a `git diff` opens with: the call's
//! `path` argument decides which file is edited, so nothing up there changes the
//! outcome and refusing it would only cost a round trip to re-send the same
//! patch trimmed. `\ No newline at end of file` annotates the line above it
//! rather than being a line of its own, and is skipped too. An empty line inside
//! a hunk is an empty context line, since a blank context line loses its single
//! leading space to every tool that strips trailing whitespace. A body line
//! opening with any other character is [`PatchError::Malformed`].
//!
//! The `@@` counts are read but not enforced. A hunk is located by its content,
//! so a miscounted header is not a reason to refuse an otherwise applicable
//! patch; the counts that matter are the lines actually present in the body.
//!
//! # Locating a hunk
//!
//! A hunk's *pre-image* is its context and removed lines in order, and the
//! engine looks for exactly that run of whole lines in the file. There is no
//! fuzz — every line must match in full, and a line matches only as a whole
//! line, never as a substring of one. The header's line numbers are a hint: when
//! the pre-image occurs in several places the occurrence nearest the hinted
//! position wins, and a tie is [`PatchError::Ambiguous`] rather than a guess.
//!
//! # Already-applied hunks
//!
//! Sending the same patch twice is a no-op rather than a second edit. Which of
//! the two images establishes that depends on what the hunk does, because each
//! is evidence only of one thing:
//!
//! - **A hunk that removes** leaves its pre-image behind only while the change
//!   is outstanding — the removed lines are gone once it applies. So the
//!   pre-image is looked for first, and the post-image answers for the rest:
//!   `-retries = 3` / `+retries = 30` meets `retries = 30` on the second pass,
//!   finds no `retries = 3` to change, and leaves `retries = 300` unwritten.
//! - **A hunk that only adds** leaves its pre-image standing either way — the
//!   pre-image is just the context it sits among, which the insertion does not
//!   disturb. Its *post-image* is the discriminator and is looked for FIRST. Ask
//!   the pre-image first here and it always answers "not yet applied", so a
//!   re-sent patch inserts its lines again, and again.
//! - **A hunk that only removes** cannot be recognised as already applied at
//!   all. What it leaves behind is its context alone, which is still there
//!   whatever else the file now holds — a bare context match would report a
//!   removal that never happened as a success that wrote nothing. Its pre-image
//!   is the only evidence, and its absence is [`PatchError::Unmatched`]. (The
//!   degenerate case of a deletion with no context at all, whose post-image is
//!   empty and therefore "present" everywhere, falls out of the same rule.)
//!
//! # All or nothing
//!
//! Hunks apply in order against the file as it stands, each searched from the
//! end of the one before, so two hunks never overlap. A hunk that neither
//! applies nor is already applied is [`PatchError::Unmatched`] and the whole call
//! fails: the working copy is built to one side and handed back only once every
//! hunk has landed, so a partially patched file is never produced.
//!
//! # Line endings
//!
//! Matching ignores the line terminator, so a CRLF file takes an LF patch and the
//! reverse. Rebuilding preserves it: each surviving line keeps the terminator it
//! had, an added line takes the file's own, and the file's trailing newline — or
//! its absence — is preserved.

use std::fmt;

/// Which side of the diff a hunk line belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    /// In both images: a `' '` line.
    Context,
    /// Only in the pre-image: a `'-'` line.
    Removed,
    /// Only in the post-image: a `'+'` line.
    Added,
}

/// One line of a hunk body, with its prefix decoded and stripped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HunkLine {
    pub kind: LineKind,
    pub text: String,
}

/// One `@@` hunk: the header it was announced by, the positions that header
/// claims, and the body lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    /// The `@@` line verbatim. Quoted back in an error so the caller can see
    /// which hunk of its own patch failed.
    pub header: String,
    /// 1-based line the hunk claims to start at in the file being patched. A
    /// hint for choosing between equal matches, never a requirement.
    pub old_start: usize,
    /// 1-based line the hunk claims to start at in the patched result.
    pub new_start: usize,
    pub lines: Vec<HunkLine>,
}

impl Hunk {
    /// The lines this hunk expects to find: context and removed, in order.
    pub fn pre_image(&self) -> Vec<&str> {
        self.image(|kind| kind != LineKind::Added)
    }

    /// The lines this hunk leaves behind: context and added, in order.
    pub fn post_image(&self) -> Vec<&str> {
        self.image(|kind| kind != LineKind::Removed)
    }

    /// Whether the hunk introduces a line.
    ///
    /// Only then does its post-image carry evidence that the change was made:
    /// what a hunk adds is absent until it applies, whereas the context it
    /// quotes is there either way.
    pub fn adds(&self) -> bool {
        self.lines.iter().any(|line| line.kind == LineKind::Added)
    }

    /// Whether the hunk takes a line away.
    ///
    /// Only then does its pre-image carry evidence that the change has *not*
    /// been made: what a hunk removes is gone once it applies, whereas a hunk
    /// that merely adds leaves its whole pre-image standing.
    pub fn removes(&self) -> bool {
        self.lines.iter().any(|line| line.kind == LineKind::Removed)
    }

    fn image(&self, keep: fn(LineKind) -> bool) -> Vec<&str> {
        self.lines
            .iter()
            .filter(|line| keep(line.kind))
            .map(|line| line.text.as_str())
            .collect()
    }
}

/// Why a patch did not apply. Each variant carries the message handed to the
/// model, naming the hunk that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchError {
    /// The patch text is not a unified-diff body this engine can read. The
    /// caller sent something other than `@@` hunks of `' '` / `'-'` / `'+'`
    /// lines, or a hunk with nothing to locate it by.
    Malformed(String),
    /// A hunk's pre-image matches in more than one place and the header's line
    /// numbers do not pick one of them.
    Ambiguous(String),
    /// The hunk neither applies nor is already there: its pre-image is not in
    /// the file, and no post-image settled it either — because the post-image
    /// was absent too, or because the hunk only removes lines and so has no
    /// post-image that could be evidence of anything.
    Unmatched(String),
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatchError::Malformed(why)
            | PatchError::Ambiguous(why)
            | PatchError::Unmatched(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for PatchError {}

/// A patched file: the content to store, and what the hunks did to get it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patched {
    pub content: String,
    /// Hunks that changed the file.
    pub applied: usize,
    /// Hunks whose change was already in the file.
    pub already_applied: usize,
}

/// A line's terminator, kept separate from its text so matching can ignore it
/// and rebuilding can restore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Eol {
    /// The last line of a file that does not end with a newline.
    None,
    Lf,
    Crlf,
}

impl Eol {
    fn as_str(self) -> &'static str {
        match self {
            Eol::None => "",
            Eol::Lf => "\n",
            Eol::Crlf => "\r\n",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileLine {
    text: String,
    eol: Eol,
}

/// Where an image was found in the file.
enum Located {
    At(usize),
    Absent,
    Ambiguous,
}

/// Read a patch into its hunks.
///
/// Every hunk is checked to carry at least one context or removed line, because
/// a hunk with an empty pre-image has nothing for the engine to find it by and
/// would have to be placed on its line number alone.
pub fn parse(patch: &str) -> Result<Vec<Hunk>, PatchError> {
    let mut hunks: Vec<Hunk> = Vec::new();
    for raw in body_lines(patch) {
        if raw.starts_with("@@") {
            hunks.push(parse_header(raw)?);
            continue;
        }
        let Some(open) = hunks.last() else {
            // Ahead of the first hunk stands the file header, in whatever form
            // the sender's diff tool wrote it — `---`/`+++`, `diff --git`,
            // `index`, a blank line. None of it selects the file, so none of it
            // is read. A patch carrying no hunk at all is still refused below.
            continue;
        };
        // `\ No newline at end of file` is a note about the line above it, not a
        // line of the hunk. The engine reads the file's own terminator.
        if raw.starts_with("\\ ") {
            continue;
        }
        let line = parse_body_line(raw, &open.header)?;
        hunks
            .last_mut()
            .expect("the hunk just borrowed is still there")
            .lines
            .push(line);
    }

    if hunks.is_empty() {
        return Err(PatchError::Malformed(
            "the patch carries no `@@` hunk — a patch is one or more unified-diff hunks"
                .to_string(),
        ));
    }
    for (index, hunk) in hunks.iter().enumerate() {
        if hunk.pre_image().is_empty() {
            return Err(PatchError::Malformed(format!(
                "{} carries no context or removed line, so there is nothing to locate it by — \
                 include the lines it sits among",
                label(index, hunk)
            )));
        }
    }
    Ok(hunks)
}

/// Apply `patch` to `content`.
///
/// Either every hunk lands — by changing the file or by being found already
/// applied — or an error comes back and the caller has nothing to store.
pub fn apply(content: &str, patch: &str) -> Result<Patched, PatchError> {
    let hunks = parse(patch)?;

    let mut lines = split_lines(content);
    let default = default_eol(&lines);
    // The file's trailing newline, or its absence, is a property of the file
    // rather than of whichever line happens to be last after patching.
    let trailing = lines.last().map_or(Eol::None, |line| line.eol);

    let mut applied = 0;
    let mut already_applied = 0;
    // One past the previous hunk, so hunks cannot overlap and a repeated block
    // is matched once per hunk rather than by all of them at the same place.
    let mut cursor = 0;

    for (index, hunk) in hunks.iter().enumerate() {
        let pre = hunk.pre_image();
        let post = hunk.post_image();
        // **Which image settles the hunk's state depends on what the hunk does.**
        //
        // A hunk that only ADDS leaves its pre-image standing — the pre-image is
        // the context it sits among, and that context is there whether or not
        // the hunk has run. So the post-image is consulted FIRST: reading a
        // surviving pre-image as "not yet applied" inserts the same lines a
        // second time on a re-send, and `file_edit` is declared replay-safe on
        // precisely the promise that it does not.
        //
        // A hunk that REMOVES is the other way about: what it removes is gone
        // once it applies, so a pre-image match is proof the change is still
        // outstanding, and it is consulted first.
        let insertion_only = hunk.adds() && !hunk.removes();

        if insertion_only {
            match locate(&lines, &post, cursor, hunk.new_start.saturating_sub(1)) {
                Located::At(start) => {
                    cursor = start + post.len();
                    already_applied += 1;
                    continue;
                }
                Located::Ambiguous => return Err(PatchError::Ambiguous(ambiguous(index, hunk))),
                Located::Absent => {}
            }
        }

        match locate(&lines, &pre, cursor, hunk.old_start.saturating_sub(1)) {
            Located::At(start) => {
                let replacement = rewrite(&lines[start..start + pre.len()], hunk, default);
                cursor = start + replacement.len();
                lines.splice(start..start + pre.len(), replacement);
                applied += 1;
            }
            Located::Ambiguous => return Err(PatchError::Ambiguous(ambiguous(index, hunk))),
            Located::Absent => {
                // An insertion-only hunk has already had its post-image tried
                // above. A hunk that removes without adding has no post-image
                // worth trying at all: what it leaves behind is bare context,
                // which survives whatever else the file now holds, so matching
                // it would report a removal that never happened as a success
                // that wrote nothing. Both are a genuine failure to locate.
                if insertion_only || !hunk.adds() {
                    return Err(PatchError::Unmatched(unmatched(index, hunk)));
                }
                match locate(&lines, &post, cursor, hunk.new_start.saturating_sub(1)) {
                    Located::At(start) => {
                        cursor = start + post.len();
                        already_applied += 1;
                    }
                    Located::Ambiguous => {
                        return Err(PatchError::Ambiguous(ambiguous(index, hunk)))
                    }
                    Located::Absent => return Err(PatchError::Unmatched(unmatched(index, hunk))),
                }
            }
        }
    }

    Ok(Patched {
        content: join_lines(&lines, default, trailing),
        applied,
        already_applied,
    })
}

// ── Parsing ──────────────────────────────────────────────────────────────────

/// The patch's lines, with the terminator artefacts removed: the `\r` of a patch
/// that travels with CRLF of its own, and the empty element a final newline
/// leaves behind.
fn body_lines(patch: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = patch
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

fn parse_header(raw: &str) -> Result<Hunk, PatchError> {
    let malformed = || {
        PatchError::Malformed(format!(
            "hunk header {raw:?} is not of the form `@@ -old_start,old_count \
             +new_start,new_count @@`"
        ))
    };
    let rest = raw.strip_prefix("@@").ok_or_else(malformed)?;
    let (ranges, _section) = rest.split_once("@@").ok_or_else(malformed)?;
    let mut fields = ranges.split_whitespace();
    let old = fields
        .next()
        .and_then(|field| field.strip_prefix('-'))
        .ok_or_else(malformed)?;
    let new = fields
        .next()
        .and_then(|field| field.strip_prefix('+'))
        .ok_or_else(malformed)?;
    if fields.next().is_some() {
        return Err(malformed());
    }
    Ok(Hunk {
        header: raw.to_string(),
        old_start: parse_start(old).ok_or_else(malformed)?,
        new_start: parse_start(new).ok_or_else(malformed)?,
        lines: Vec::new(),
    })
}

/// The start line of an `a,b` range field, or of a bare `a` (a one-line range).
/// The count is parsed only to reject a malformed header; the engine locates by
/// content and never consults it.
fn parse_start(field: &str) -> Option<usize> {
    match field.split_once(',') {
        Some((start, count)) => {
            count.parse::<usize>().ok()?;
            start.parse().ok()
        }
        None => field.parse().ok(),
    }
}

fn parse_body_line(raw: &str, header: &str) -> Result<HunkLine, PatchError> {
    let (kind, text) = match raw.chars().next() {
        // A blank context line arrives stripped of its single leading space
        // often enough that reading it as anything else would reject ordinary
        // patches.
        None => (LineKind::Context, ""),
        Some(' ') => (LineKind::Context, &raw[1..]),
        Some('-') => (LineKind::Removed, &raw[1..]),
        Some('+') => (LineKind::Added, &raw[1..]),
        Some(_) => {
            return Err(PatchError::Malformed(format!(
                "line {raw:?} in hunk ({header}) opens with none of ' ' (context), '-' \
                 (removed) or '+' (added)"
            )))
        }
    };
    Ok(HunkLine {
        kind,
        text: text.to_string(),
    })
}

// ── Lines and terminators ────────────────────────────────────────────────────

fn split_lines(content: &str) -> Vec<FileLine> {
    let mut out = Vec::new();
    let mut rest = content;
    while let Some(at) = rest.find('\n') {
        let (line, tail) = rest.split_at(at + 1);
        let body = &line[..line.len() - 1];
        let (text, eol) = match body.strip_suffix('\r') {
            Some(text) => (text, Eol::Crlf),
            None => (body, Eol::Lf),
        };
        out.push(FileLine {
            text: text.to_string(),
            eol,
        });
        rest = tail;
    }
    if !rest.is_empty() {
        out.push(FileLine {
            text: rest.to_string(),
            eol: Eol::None,
        });
    }
    out
}

/// The terminator an added line takes: the file's first real one, falling back
/// to LF for a file that has none (a single line with no trailing newline).
fn default_eol(lines: &[FileLine]) -> Eol {
    lines
        .iter()
        .map(|line| line.eol)
        .find(|eol| *eol != Eol::None)
        .unwrap_or(Eol::Lf)
}

fn join_lines(lines: &[FileLine], default: Eol, trailing: Eol) -> String {
    let mut out = String::new();
    for (index, line) in lines.iter().enumerate() {
        out.push_str(&line.text);
        let eol = if index + 1 == lines.len() {
            trailing
        } else if line.eol == Eol::None {
            default
        } else {
            line.eol
        };
        out.push_str(eol.as_str());
    }
    out
}

// ── Matching ─────────────────────────────────────────────────────────────────

/// Find `image` in `lines` at or after `from`, preferring the occurrence nearest
/// `hint`. An empty image is never located: it would be "present" everywhere.
fn locate(lines: &[FileLine], image: &[&str], from: usize, hint: usize) -> Located {
    if image.is_empty() || lines.len() < image.len() {
        return Located::Absent;
    }
    let mut best: Option<(usize, usize)> = None;
    let mut tied = false;
    for start in from..=(lines.len() - image.len()) {
        if !matches_at(lines, image, start) {
            continue;
        }
        let distance = start.abs_diff(hint);
        match best {
            Some((best_distance, _)) if distance > best_distance => {}
            Some((best_distance, _)) if distance == best_distance => tied = true,
            _ => {
                best = Some((distance, start));
                tied = false;
            }
        }
    }
    match best {
        Some((_, start)) if !tied => Located::At(start),
        Some(_) => Located::Ambiguous,
        None => Located::Absent,
    }
}

fn matches_at(lines: &[FileLine], image: &[&str], start: usize) -> bool {
    image
        .iter()
        .enumerate()
        .all(|(offset, want)| lines[start + offset].text == *want)
}

/// The lines a hunk leaves in place of the run it matched: each context line
/// keeps the terminator it had, each added line takes the file's.
fn rewrite(matched: &[FileLine], hunk: &Hunk, default: Eol) -> Vec<FileLine> {
    let mut out = Vec::with_capacity(hunk.lines.len());
    let mut source = 0;
    for line in &hunk.lines {
        match line.kind {
            LineKind::Context => {
                out.push(matched[source].clone());
                source += 1;
            }
            LineKind::Removed => source += 1,
            LineKind::Added => out.push(FileLine {
                text: line.text.clone(),
                eol: default,
            }),
        }
    }
    out
}

// ── Messages ─────────────────────────────────────────────────────────────────

fn label(index: usize, hunk: &Hunk) -> String {
    format!("hunk {} ({})", index + 1, hunk.header)
}

fn ambiguous(index: usize, hunk: &Hunk) -> String {
    format!(
        "{} matches in more than one place and its line numbers do not pick one — include more \
         surrounding context",
        label(index, hunk)
    )
}

fn unmatched(index: usize, hunk: &Hunk) -> String {
    format!(
        "{} does not apply: its context is not in the file, and its change is not already there \
         — re-read the file and build the hunk from what it says now",
        label(index, hunk)
    )
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn patched(content: &str, patch: &str) -> Patched {
        apply(content, patch).expect("the patch applies")
    }

    fn line(kind: LineKind, text: &str) -> HunkLine {
        HunkLine {
            kind,
            text: text.to_string(),
        }
    }

    // ── Parsing ──────────────────────────────────────────────────────────────

    /// The three line kinds, the header's two positions, and the header text
    /// kept verbatim for error messages.
    #[test]
    fn a_hunk_parses_into_its_positions_and_lines() {
        let hunks = parse("@@ -4,3 +4,4 @@ fn main()\n ctx\n-gone\n+new\n+also\n").unwrap();
        assert_eq!(
            hunks,
            vec![Hunk {
                header: "@@ -4,3 +4,4 @@ fn main()".to_string(),
                old_start: 4,
                new_start: 4,
                lines: vec![
                    line(LineKind::Context, "ctx"),
                    line(LineKind::Removed, "gone"),
                    line(LineKind::Added, "new"),
                    line(LineKind::Added, "also"),
                ],
            }]
        );
        assert_eq!(hunks[0].pre_image(), vec!["ctx", "gone"]);
        assert_eq!(hunks[0].post_image(), vec!["ctx", "new", "also"]);
    }

    /// A one-line range is written without a count.
    #[test]
    fn a_range_without_a_count_parses() {
        let hunks = parse("@@ -7 +7 @@\n-a\n+b\n").unwrap();
        assert_eq!(hunks[0].old_start, 7);
        assert_eq!(hunks[0].new_start, 7);
    }

    /// The `---`/`+++` header names a file, but the call's `path` argument is
    /// what decides which file is edited, so it is read past.
    #[test]
    fn a_file_header_before_the_first_hunk_is_ignored() {
        let hunks = parse("--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-a\n+b\n").unwrap();
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].header, "@@ -1 +1 @@");
    }

    #[test]
    fn an_empty_patch_is_malformed() {
        assert_eq!(
            apply("a\n", ""),
            Err(PatchError::Malformed(
                "the patch carries no `@@` hunk — a patch is one or more unified-diff hunks"
                    .to_string()
            ))
        );
    }

    /// Body lines with no hunk header above them are header text as far as the
    /// parser is concerned, so the patch reduces to no hunks at all — which is
    /// what the refusal names.
    #[test]
    fn a_patch_with_no_hunk_header_is_malformed() {
        assert_eq!(
            parse("-a\n+b\n"),
            Err(PatchError::Malformed(
                "the patch carries no `@@` hunk — a patch is one or more unified-diff hunks"
                    .to_string()
            ))
        );
    }

    /// A patch pasted straight out of `git diff` carries its own preamble. The
    /// `path` argument is what selects the file, so the preamble decides
    /// nothing and is skipped — refusing it would cost a round trip to re-send
    /// the same patch with the top trimmed off.
    #[test]
    fn a_git_preamble_is_skipped() {
        let hunks = parse("diff --git a/x b/x\nindex 1..2 100644\n@@ -1 +1 @@\n-a\n+b\n").unwrap();
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].header, "@@ -1 +1 @@");
        assert_eq!(
            patched("a\n", "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n").content,
            "b\n"
        );
    }

    /// `\ No newline at end of file` describes the line above it. Reading it as
    /// a hunk line would refuse an ordinary diff of a file that ends without a
    /// newline; the engine takes the terminator from the file itself.
    #[test]
    fn a_no_newline_marker_is_skipped() {
        let out = patched("a\nb", "@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n\\ No newline at end of file\n");
        assert_eq!(out.content, "a\nc");
        assert_eq!((out.applied, out.already_applied), (1, 0));
    }

    #[test]
    fn a_body_line_with_a_foreign_prefix_is_malformed() {
        assert_eq!(
            parse("@@ -1 +1 @@\n-a\n*b\n"),
            Err(PatchError::Malformed(
                "line \"*b\" in hunk (@@ -1 +1 @@) opens with none of ' ' (context), '-' \
                 (removed) or '+' (added)"
                    .to_string()
            ))
        );
    }

    #[test]
    fn a_malformed_header_is_named_in_full() {
        assert_eq!(
            parse("@@ 1,2 3,4 @@\n-a\n+b\n"),
            Err(PatchError::Malformed(
                "hunk header \"@@ 1,2 3,4 @@\" is not of the form `@@ -old_start,old_count \
                 +new_start,new_count @@`"
                    .to_string()
            ))
        );
    }

    /// A hunk of nothing but additions could only be placed by its line number,
    /// which is the fragility this engine exists to remove.
    #[test]
    fn a_hunk_with_no_pre_image_is_malformed() {
        assert_eq!(
            parse("@@ -1,0 +1,1 @@\n+added\n"),
            Err(PatchError::Malformed(
                "hunk 1 (@@ -1,0 +1,1 @@) carries no context or removed line, so there is \
                 nothing to locate it by — include the lines it sits among"
                    .to_string()
            ))
        );
    }

    /// A blank line inside a hunk is an empty context line: the single leading
    /// space does not survive trailing-whitespace stripping.
    #[test]
    fn a_blank_line_inside_a_hunk_is_empty_context() {
        let out = patched("a\n\nb\n", "@@ -1,3 +1,3 @@\n a\n\n-b\n+B\n");
        assert_eq!(out.content, "a\n\nB\n");
    }

    // ── Applying ─────────────────────────────────────────────────────────────

    #[test]
    fn a_single_hunk_applies() {
        let out = patched(
            "[server]\nhost = \"localhost\"\nport = 8080\n",
            "@@ -1,3 +1,3 @@\n [server]\n-host = \"localhost\"\n+host = \"0.0.0.0\"\n port = 8080\n",
        );
        assert_eq!(
            out,
            Patched {
                content: "[server]\nhost = \"0.0.0.0\"\nport = 8080\n".to_string(),
                applied: 1,
                already_applied: 0,
            }
        );
    }

    /// Two hunks land in one call, the second searched from the end of the
    /// first.
    #[test]
    fn two_hunks_apply_in_order() {
        let out = patched(
            "one\ntwo\nthree\nfour\nfive\nsix\n",
            "@@ -1,2 +1,2 @@\n one\n-two\n+2\n@@ -5,2 +5,2 @@\n five\n-six\n+6\n",
        );
        assert_eq!(out.content, "one\n2\nthree\nfour\nfive\n6\n");
        assert_eq!((out.applied, out.already_applied), (2, 0));
    }

    /// The case the engine is for: the header's line numbers are nowhere near
    /// the truth, and the hunk still lands because its context is found.
    #[test]
    fn a_hunk_is_found_by_content_when_its_line_numbers_are_wrong() {
        let content = "a\nb\nc\nd\ntarget\ne\n";
        let out = patched(content, "@@ -900,2 +900,2 @@\n-target\n+hit\n e\n");
        assert_eq!(out.content, "a\nb\nc\nd\nhit\ne\n");
        assert_eq!((out.applied, out.already_applied), (1, 0));
    }

    /// Two equally-distant matches are a refusal, not a coin flip.
    #[test]
    fn a_hunk_matching_twice_is_refused_as_ambiguous() {
        // `x` sits at index 1 and index 3; the hint (line 3 → index 2) is one
        // away from each.
        let err = apply("a\nx\nb\nx\nc\n", "@@ -3 +3 @@\n-x\n+y\n").unwrap_err();
        assert_eq!(
            err,
            PatchError::Ambiguous(
                "hunk 1 (@@ -3 +3 @@) matches in more than one place and its line numbers do \
                 not pick one — include more surrounding context"
                    .to_string()
            )
        );
    }

    /// When the numbers do single one out, they are used: the nearest match to
    /// the hinted position wins and the call succeeds.
    #[test]
    fn the_line_hint_picks_between_two_matches() {
        let content = "fn a() {\n    log();\n}\nfn b() {\n    log();\n}\n";
        let out = patched(content, "@@ -5 +5 @@\n-    log();\n+    trace();\n");
        assert_eq!(
            out.content, "fn a() {\n    log();\n}\nfn b() {\n    trace();\n}\n",
            "the hint names the second site, so the first must be left alone",
        );
    }

    #[test]
    fn a_hunk_that_matches_nothing_names_itself() {
        let err = apply("a\nb\n", "@@ -1,2 +1,2 @@\n a\n-absent\n+new\n").unwrap_err();
        assert_eq!(
            err,
            PatchError::Unmatched(
                "hunk 1 (@@ -1,2 +1,2 @@) does not apply: its context is not in the file, and \
                 its change is not already there — re-read the file and build the hunk from \
                 what it says now"
                    .to_string()
            )
        );
    }

    // ── Already applied ──────────────────────────────────────────────────────

    #[test]
    fn an_already_applied_hunk_is_detected_and_skipped() {
        let patch = "@@ -1,2 +1,2 @@\n [server]\n-port = 8080\n+port = 9090\n";
        let out = patched("[server]\nport = 9090\n", patch);
        assert_eq!(
            out,
            Patched {
                content: "[server]\nport = 9090\n".to_string(),
                applied: 0,
                already_applied: 1,
            }
        );
    }

    /// Applying the same patch twice succeeds both times, and the second pass
    /// writes nothing.
    #[test]
    fn a_patch_applied_twice_is_a_no_op_the_second_time() {
        let patch = "@@ -1,2 +1,2 @@\n one\n-two\n+2\n";
        let first = patched("one\ntwo\n", patch);
        assert_eq!((first.applied, first.already_applied), (1, 0));
        let second = patched(&first.content, patch);
        assert_eq!(second.content, first.content);
        assert_eq!((second.applied, second.already_applied), (0, 1));
    }

    /// **The case this engine was built for.** The added text contains the
    /// removed text, so a substring-replacing edit would turn `30` into `300`
    /// on a second pass. Whole-line matching sees `retries = 30` is not
    /// `retries = 3` and reads the post-image instead.
    #[test]
    fn re_applying_a_hunk_whose_addition_contains_its_removal_changes_nothing() {
        let patch = "@@ -1,2 +1,2 @@\n [retry]\n-retries = 3\n+retries = 30\n";
        let once = patched("[retry]\nretries = 3\n", patch);
        assert_eq!(once.content, "[retry]\nretries = 30\n");
        assert_eq!((once.applied, once.already_applied), (1, 0));

        let twice = patched(&once.content, patch);
        assert_eq!(
            twice.content, "[retry]\nretries = 30\n",
            "`retries = 300` would mean the hunk applied a second time",
        );
        assert_eq!((twice.applied, twice.already_applied), (0, 1));
    }

    /// A patch half of which has landed already: each hunk is judged on its own.
    #[test]
    fn a_half_applied_patch_reports_both_counts() {
        let out = patched(
            "a\nB\nc\nd\n",
            "@@ -1,2 +1,2 @@\n a\n-b\n+B\n@@ -3,2 +3,2 @@\n c\n-d\n+D\n",
        );
        assert_eq!(out.content, "a\nB\nc\nD\n");
        assert_eq!((out.applied, out.already_applied), (1, 1));
    }

    /// A hunk that only removes has no already-applied test: what it leaves
    /// behind is its context, which proves nothing. With no context at all the
    /// post-image is empty and "present" everywhere, which is the same rule at
    /// its limit.
    #[test]
    fn a_contextless_deletion_is_not_reported_as_already_applied() {
        let patch = "@@ -2 +2,0 @@\n-gone\n";
        let out = patched("keep\ngone\n", patch);
        assert_eq!(out.content, "keep\n");
        assert_eq!((out.applied, out.already_applied), (1, 0));

        let err = apply(&out.content, patch).unwrap_err();
        assert!(matches!(err, PatchError::Unmatched(_)), "{err:?}");
    }

    /// **A hunk that only adds is idempotent — the property `file_edit` is
    /// declared replay-safe on.**
    ///
    /// Its pre-image is the context it sits among, which the insertion does not
    /// disturb, so the pre-image still matches on a second pass. Consulting it
    /// first therefore reports "not yet applied" forever and the lines go in
    /// again, and again — the shape a resumed tool round re-issues verbatim.
    ///
    /// The added line sits at the hunk's EDGE deliberately. Insert between two
    /// context lines and the pre-image stops being contiguous once the lines
    /// land, so it no longer matches and the bug hides; the real cases (a
    /// trailing `.env` in a `.gitignore`, an import at the top) all add at an
    /// edge.
    #[test]
    fn an_insertion_only_hunk_re_sent_inserts_nothing_further() {
        let patch = "@@ -1 +1,2 @@\n alpha\n+inserted\n";
        let once = patched("alpha\nbeta\n", patch);
        assert_eq!(once.content, "alpha\ninserted\nbeta\n");
        assert_eq!((once.applied, once.already_applied), (1, 0));

        let twice = patched(&once.content, patch);
        assert_eq!(
            twice.content, "alpha\ninserted\nbeta\n",
            "a second `inserted` would mean the hunk applied twice",
        );
        assert_eq!((twice.applied, twice.already_applied), (0, 1));
    }

    /// **A removal that has not happened is a failure, not a success that wrote
    /// nothing.**
    ///
    /// The hunk's post-image here is the single context line `keep`, which is
    /// still in the file — but the file no longer holds `gone` either, so the
    /// removal never ran and the file says something else entirely. Reading
    /// that context match as "already applied" returns success, writes nothing,
    /// and tells the model its edit landed.
    #[test]
    fn a_deletion_whose_context_survives_is_unmatched_not_already_applied() {
        let patch = "@@ -1,2 +1 @@\n keep\n-gone\n";
        let err = apply("keep\nsomething else\n", patch).unwrap_err();
        assert!(
            matches!(err, PatchError::Unmatched(_)),
            "the lone surviving context line must not be read as evidence: {err:?}",
        );
    }

    // ── All or nothing ───────────────────────────────────────────────────────

    /// A patch whose second hunk fails produces no content at all — the first
    /// hunk's change is discarded with the working copy.
    #[test]
    fn one_failing_hunk_abandons_the_whole_patch() {
        let content = "alpha\nbeta\ngamma\ndelta\n";
        let result = apply(
            content,
            "@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n@@ -3,2 +3,2 @@\n gamma\n-absent\n+new\n",
        );
        assert_eq!(
            result,
            Err(PatchError::Unmatched(
                "hunk 2 (@@ -3,2 +3,2 @@) does not apply: its context is not in the file, and \
                 its change is not already there — re-read the file and build the hunk from \
                 what it says now"
                    .to_string()
            )),
            "the error names the FIRST failing hunk, and no content comes back",
        );
    }

    // ── Line endings ─────────────────────────────────────────────────────────

    /// A CRLF file stays CRLF, including the line the patch adds, and an LF
    /// patch matches it.
    #[test]
    fn a_crlf_file_keeps_its_terminators() {
        let out = patched(
            "one\r\ntwo\r\nthree\r\n",
            "@@ -1,3 +1,4 @@\n one\n-two\n+2\n+two-and-a-half\n three\n",
        );
        assert_eq!(out.content, "one\r\n2\r\ntwo-and-a-half\r\nthree\r\n");
    }

    /// And the reverse: a patch that travels with CRLF applies to an LF file
    /// without dragging `\r` into it.
    #[test]
    fn a_crlf_patch_applies_to_an_lf_file() {
        let out = patched("one\ntwo\n", "@@ -1,2 +1,2 @@\r\n one\r\n-two\r\n+2\r\n");
        assert_eq!(out.content, "one\n2\n");
    }

    /// A file that does not end with a newline does not acquire one, whether
    /// the last line is rewritten or a line is added after it.
    #[test]
    fn a_file_without_a_trailing_newline_does_not_gain_one() {
        let rewritten = patched("alpha\nbeta", "@@ -2 +2 @@\n-beta\n+gamma\n");
        assert_eq!(rewritten.content, "alpha\ngamma");

        let extended = patched("alpha", "@@ -1 +1,2 @@\n alpha\n+beta\n");
        assert_eq!(extended.content, "alpha\nbeta");
    }

    /// A file that does end with one keeps it when the last line changes.
    #[test]
    fn a_trailing_newline_survives_an_edit_to_the_last_line() {
        let out = patched("alpha\nbeta\n", "@@ -2 +2 @@\n-beta\n+gamma\n");
        assert_eq!(out.content, "alpha\ngamma\n");
    }

    /// Adding and deleting at once, with the terminator of the file carried
    /// onto the added lines.
    #[test]
    fn a_hunk_may_add_and_remove_together() {
        let out = patched(
            "head\nold one\nold two\ntail\n",
            "@@ -1,4 +1,4 @@\n head\n-old one\n-old two\n+new one\n+new two\n tail\n",
        );
        assert_eq!(out.content, "head\nnew one\nnew two\ntail\n");
    }
}
