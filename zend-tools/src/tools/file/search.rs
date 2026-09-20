//! file_search tool — find files by name or path.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{FileError, Paging};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Paths per page. A path is far cheaper than a `file_list` entry (no size or
/// line count), so a page holds more of them while staying near the same token
/// budget.
pub const SEARCH_PAGE_ENTRIES: usize = 60;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SearchRequest {
    /// What to look for in the file's path — a bare name (`main.rs`), a stem
    /// (`compress`), an extension (`.toml`), or a fragment of a directory
    /// (`kv_cache/chunked`). Matched case-insensitively against the whole
    /// path. Supports `*` as a wildcard (`*.rs`, `src/*/mod.rs`). Required.
    #[validate(length(min = 1))]
    pub query: String,
    /// Restrict the search to paths beginning with this prefix (e.g.
    /// `candle-nn/src/`). Omit to search the whole project.
    pub prefix: Option<String>,
    /// Zero-based page of results. Defaults to 0. When the response's
    /// `paging.next_page` is set, pass it here for the following page.
    pub page: Option<u32>,
}

#[derive(Serialize)]
pub struct SearchResponse {
    /// Matching paths, shortest first then alphabetical — the shortest path
    /// matching a name is usually the definition rather than a vendored or
    /// generated copy of it.
    pub files: Vec<String>,
    pub paging: Paging,
}

pub struct FileSearch;

impl Tool for FileSearch {
    const NAME: &'static str = "file_search";
    const DESCRIPTION: &'static str =
        "Find files by NAME or PATH anywhere in the project, without knowing which \
         directory they are in. Give a filename (`config.rs`), a stem (`compress`), \
         an extension (`.toml`), a path fragment (`kv_cache/chunked`), or a glob \
         (`*_test.rs`, `src/*/mod.rs`); matching is case-insensitive over the whole \
         path. Use for: locating a file whose name you know but whose directory you \
         do not, checking whether a module exists, finding every file of a kind, \
         discovering where a subsystem lives before reading it. Triggered by \
         \"where is\", \"find the file\", \"which file is\", \"locate\", \"is there a \
         file called\", \"what files are named\", \"show me all the .rs files\". \
         Returns paths only, shortest first, paged. THIS IS THE TOOL FOR FINDING A \
         FILE — do not guess directory names and call file_list repeatedly; one \
         file_search over the whole project replaces that entirely. Use file_grep to \
         search file CONTENTS for a string or symbol; use file_list to enumerate a \
         directory you already know; use file_read once you have the path.";

    type Request = SearchRequest;
    type Response = SearchResponse;
    type Error = FileError;

    /// Searches names; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: SearchRequest) -> Result<SearchResponse, FileError> {
        let prefix = req.prefix.as_deref().unwrap_or("");
        let query = req.query.to_ascii_lowercase();

        let mut all: Vec<String> = ctx
            .vfs
            .paths(prefix)
            .into_iter()
            .filter(|p| matches(&p.to_ascii_lowercase(), &query))
            .collect();
        // Shortest first: `config.rs` at a crate root beats a deeply nested
        // vendored copy of the same name, and that is nearly always the one the
        // caller meant.
        all.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

        let paging = Paging::of(all.len(), req.page.unwrap_or(0), SEARCH_PAGE_ENTRIES);
        let files = all
            .into_iter()
            .skip(paging.skipped())
            .take(SEARCH_PAGE_ENTRIES)
            .collect();
        Ok(SearchResponse { files, paging })
    }
}

/// Whether `path` (already lowercased) matches `query` (already lowercased).
///
/// A query with no `*` is a plain substring test, which is what a caller typing
/// a filename means. With `*` it is an anchored wildcard match, where `*` spans
/// any run of characters including `/` — so `src/*/mod.rs` finds a `mod.rs` any
/// depth below `src/`, which is the reading that makes a glob useful on paths
/// rather than on a single directory.
fn matches(path: &str, query: &str) -> bool {
    if !query.contains('*') {
        return path.contains(query);
    }
    let parts: Vec<&str> = query.split('*').collect();
    let mut cursor = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        let first = i == 0;
        let last = i == parts.len() - 1;
        match path[cursor..].find(part) {
            Some(at) => {
                // A leading literal must sit at the start: `*.rs` may match
                // anywhere, `src*` may not.
                if first && at != 0 {
                    return false;
                }
                cursor += at + part.len();
            }
            None => return false,
        }
        // A trailing literal must end the path, so `*.rs` does not match
        // `a.rs.bak`.
        if last && cursor != path.len() {
            return false;
        }
    }
    true
}

pub const FILE_SEARCH: RegisteredTool = RegisteredTool::new::<FileSearch>();

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn a_plain_query_is_a_substring_of_the_path() {
        assert!(matches("candle-nn/src/kv_cache/mod.rs", "kv_cache"));
        assert!(matches("candle-nn/src/kv_cache/mod.rs", "mod.rs"));
        assert!(!matches("candle-nn/src/kv_cache/mod.rs", "missing"));
    }

    #[test]
    fn a_leading_wildcard_matches_any_prefix() {
        assert!(matches("a/b/c.rs", "*.rs"));
        assert!(!matches("a/b/c.rs", "*.toml"));
    }

    /// A trailing literal must end the path — the property that stops `*.rs`
    /// matching an editor backup.
    #[test]
    fn a_trailing_literal_anchors_at_the_end() {
        assert!(!matches("a/b/c.rs.bk", "*.rs"));
    }

    /// A leading literal anchors at the start, so a glob is a path pattern and
    /// not another substring test.
    #[test]
    fn a_leading_literal_anchors_at_the_start() {
        assert!(matches("src/a/mod.rs", "src/*/mod.rs"));
        assert!(!matches("crates/src/a/mod.rs", "src/*mod.rs"));
    }

    /// `*` spans `/`, so one pattern reaches any depth.
    #[test]
    fn a_wildcard_spans_directory_separators() {
        assert!(matches("src/a/b/c/mod.rs", "src/*/mod.rs"));
    }
}
