//! notes_search tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::NotesError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SearchRequest {
    /// Content substring to search for. Defaults to "" (no text filter); at least one of query or tags must be supplied.
    pub query: Option<String>,
    /// Tags that returned notes must include. Defaults to none; at least one of query or tags must be supplied.
    pub tags: Option<Vec<String>>,
    /// Maximum results to return (1–50). Defaults to 10.
    #[validate(range(min = 1, max = 50))]
    pub max_results: Option<u32>,
}

#[derive(Serialize)]
pub struct SearchEntry {
    pub key: String,
    pub snippet: String,
    pub tags: Vec<String>,
    pub updated_at: String,
    pub rank: f64,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchEntry>,
    pub total_matches: usize,
}

pub struct NotesSearch;

impl Tool for NotesSearch {
    const NAME: &'static str = "notes_search";
    const DESCRIPTION: &'static str =
        "Find stored notes by a content substring or tag match, returning \
         ranked hits with text snippets. Use when the note's exact key is \
         unknown; supply a query, tags, or both.";

    type Request = SearchRequest;
    type Response = SearchResponse;
    type Error = NotesError;

    fn run(ctx: &ToolContext, req: SearchRequest) -> Result<SearchResponse, NotesError> {
        let query = req.query.as_deref().unwrap_or("");
        let tags = req.tags.as_deref().unwrap_or(&[]);
        if query.is_empty() && tags.is_empty() {
            return Err(NotesError::NoSearchCriteria);
        }
        let max = req.max_results.unwrap_or(10) as usize;
        let (results, total) = ctx.notes.search(query, tags, max);
        let entries = results
            .into_iter()
            .map(|r| SearchEntry {
                key: r.key,
                snippet: r.snippet,
                tags: r.tags,
                updated_at: r.updated_at,
                rank: r.rank,
            })
            .collect();
        Ok(SearchResponse {
            results: entries,
            total_matches: total,
        })
    }
}

pub const NOTES_SEARCH: RegisteredTool = RegisteredTool::new::<NotesSearch>();
