//! file_grep tool — regular-expression search over file contents.

use regex::RegexBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{FileError, Paging};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Matching lines per page.
pub const GREP_PAGE_HITS: usize = 40;

/// Hits taken from any one file before moving on. A generated file with a
/// thousand matches would otherwise fill the whole result and hide every other
/// file that matched — and "which files contain this" is usually the question.
pub const MAX_HITS_PER_FILE: usize = 20;

/// Hard ceiling on a single scan, across all files.
pub const MAX_TOTAL_HITS: usize = 600;

/// Longest line returned intact. A minified bundle is one 200 KB line, and a
/// single hit on it would otherwise be larger than the whole rest of the result.
const MAX_LINE_CHARS: usize = 400;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct GrepRequest {
    /// Regular expression to search for (Rust `regex` syntax — `\b`, `[A-Z]`,
    /// `foo|bar`, `^fn `). A plain string works as itself. Required.
    #[validate(length(min = 1))]
    pub pattern: String,
    /// Restrict the search to paths beginning with this prefix (e.g.
    /// `candle-nn/src/`). Omit to search the whole project.
    pub prefix: Option<String>,
    /// Match without regard to case. Defaults to false (case-sensitive), which
    /// is what searching for a Rust identifier wants.
    pub ignore_case: Option<bool>,
    /// Zero-based page of results. Defaults to 0. When the response's
    /// `paging.next_page` is set, pass it here for the following page.
    pub page: Option<u32>,
}

#[derive(Serialize)]
pub struct GrepMatch {
    pub path: String,
    /// 1-based line number — `(line - 1) / PAGE_LINES` is the page to pass
    /// `file_read` to see the surrounding code.
    pub line: u32,
    /// The matching line, trimmed of trailing whitespace and truncated if very
    /// long.
    pub text: String,
    /// Present only when this session has edited the file, so the hit is in the
    /// session's copy rather than in what is on disk.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub modified: bool,
}

#[derive(Serialize)]
pub struct GrepResponse {
    pub matches: Vec<GrepMatch>,
    pub paging: Paging,
    /// How many files were actually scanned. Distinguishes "searched 4,000
    /// files and this pattern is genuinely absent" from "the prefix matched
    /// nothing, so nothing was searched".
    pub files_searched: usize,
    /// `true` when the scan hit its ceiling and the project may hold more
    /// matches than are reported. Narrow with `prefix` or a tighter pattern.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

pub struct FileGrep;

impl Tool for FileGrep {
    const NAME: &'static str = "file_grep";
    const DESCRIPTION: &'static str =
        "Search the CONTENTS of every file in the project for a string or regular \
         expression, and return the matching lines with their file paths and line \
         numbers. Use for: finding where a function, type, constant or error \
         message is defined or used; checking whether something exists in the \
         codebase at all; tracing callers of an API; locating a config key, a magic \
         string, or a TODO. Triggered by \"where is X defined\", \"who calls\", \
         \"find all uses of\", \"search the code for\", \"does the codebase \
         contain\", \"grep for\", \"which file has\", \"find the string\". Takes a \
         regex (`^pub fn `, `Error::\\w+`, `foo|bar`), an optional path prefix to \
         narrow the search, and optional ignore_case. Returns path, line number and \
         the matching line, paged, with files_searched so an empty result is \
         unambiguous. THIS IS THE TOOL FOR FINDING CODE BY CONTENT — reach for it \
         before guessing at directory names with file_list. Use file_search to find \
         a file by its NAME; use file_read with the line number this returns to see \
         the surrounding code.";

    type Request = GrepRequest;
    type Response = GrepResponse;
    type Error = FileError;

    /// Reads files; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: GrepRequest) -> Result<GrepResponse, FileError> {
        // A bad pattern is the caller's to fix, and the regex crate's message
        // names the offending position — far more useful than "invalid regex".
        let re = RegexBuilder::new(&req.pattern)
            .case_insensitive(req.ignore_case.unwrap_or(false))
            .size_limit(1 << 20)
            .build()
            .map_err(|e| FileError::InvalidArguments(format!("invalid regex: {e}")))?;

        let prefix = req.prefix.as_deref().unwrap_or("");
        let outcome = ctx.vfs.grep(&re, prefix, MAX_HITS_PER_FILE, MAX_TOTAL_HITS);

        let paging = Paging::of(outcome.hits.len(), req.page.unwrap_or(0), GREP_PAGE_HITS);
        let matches = outcome
            .hits
            .into_iter()
            .skip(paging.skipped())
            .take(GREP_PAGE_HITS)
            .map(|h| GrepMatch {
                path: h.path,
                line: h.line_no,
                text: truncate(&h.line),
                modified: h.modified,
            })
            .collect();

        Ok(GrepResponse {
            matches,
            paging,
            files_searched: outcome.files_searched,
            truncated: outcome.truncated,
        })
    }
}

/// Clip a very long line, marking that it was clipped.
///
/// Counts characters rather than bytes and cuts on a character boundary: a
/// byte-wise cut through a multi-byte character would panic on the slice.
fn truncate(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_CHARS {
        return line.to_string();
    }
    let cut: String = line.chars().take(MAX_LINE_CHARS).collect();
    format!("{cut} … [line truncated]")
}

pub const FILE_GREP: RegisteredTool = RegisteredTool::new::<FileGrep>();

#[cfg(test)]
mod tests {
    use super::{truncate, MAX_LINE_CHARS};

    #[test]
    fn a_short_line_is_returned_intact() {
        assert_eq!(truncate("fn main() {}"), "fn main() {}");
    }

    #[test]
    fn a_long_line_is_clipped_and_says_so() {
        let long = "x".repeat(MAX_LINE_CHARS + 50);
        let out = truncate(&long);
        assert!(out.ends_with(" … [line truncated]"), "{out}");
        assert_eq!(
            out.chars().count(),
            MAX_LINE_CHARS + " … [line truncated]".chars().count()
        );
    }

    /// A multi-byte character at the cut point must not panic, which a
    /// byte-indexed slice would.
    #[test]
    fn a_multibyte_line_cuts_on_a_character_boundary() {
        let long = "→".repeat(MAX_LINE_CHARS + 10);
        let out = truncate(&long);
        assert!(out.starts_with('→'));
    }
}
