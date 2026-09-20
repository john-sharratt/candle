//! `web_search` tool — search the web via Tavily API.
//!
//! The API key comes from the daemon's secrets document
//! ([`ToolSecrets`](crate::state::ToolSecrets)), read once at startup from a
//! per-user file outside the workspace. It is deliberately not read from the
//! process environment: the key would then have to be exported by whatever
//! launches the daemon, and every child process would inherit it.

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::state::ToolSecrets;
use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// Search query string.
    #[validate(length(min = 1))]
    pub query: String,
    /// Maximum number of ranked results to return (1-10). Default: 5.
    #[validate(range(min = 1, max = 10))]
    pub max_results: Option<u8>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: f64,
}

#[derive(Serialize)]
pub struct Response {
    /// Today's date (UTC, `YYYY-MM-DD`) — the day the search ran.
    ///
    /// The model has no clock of its own and dates the world by its training:
    /// asked for the latest Rust release it searched "latest stable Rust
    /// release 2025" a year late, and reported a release the results
    /// themselves showed had been superseded. Carried on every search so
    /// "latest" is judged against today, not against what the model remembers.
    pub searched_on: String,
    pub results: Vec<SearchResult>,
}

/// `now` as the `YYYY-MM-DD` date [`Response::searched_on`] carries.
fn search_date(now: DateTime<Utc>) -> String {
    now.format("%Y-%m-%d").to_string()
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("search unavailable: {0}")]
    SearchUnavailable(String),
}

impl ToolError for SearchError {
    fn code(&self) -> &'static str {
        match self {
            SearchError::SearchUnavailable(_) => "search_unavailable",
        }
    }
}

pub struct WebSearchTool;

impl Tool for WebSearchTool {
    const NAME: &'static str = "web_search";
    const DESCRIPTION: &'static str =
        "Search the web for information using a query string and return ranked results with \
         title, URL, snippet, and relevance score. Use for: looking up current information, \
         finding articles or documentation, researching a topic, locating a URL when only the \
         topic is known, getting recent news, identifying who or what something is. Triggered \
         by \"search for\", \"look up\", \"find information about\", \"what is X\", \"who is\", \
         \"recent news on\", \"google\", \"search the web\". Returns up to 10 ranked results \
         and searched_on, today's date: judge \"latest\" against it, prefer the newest \
         result, and put no year in a query the user did not give one. \
         For DNS records use dns_lookup; for fetching a specific URL already known, use \
         web_fetch; for authenticated API calls use http_session_*.";

    type Request = Request;
    type Response = Response;
    type Error = SearchError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, SearchError> {
        // The message names the file the operator has to edit. A model that
        // reads "search unavailable" can say something useful to the user with
        // this, and an operator reading the log is told the fix.
        let Some(api_key) = ctx.secrets.tavily_api_key() else {
            return Err(SearchError::SearchUnavailable(format!(
                "no tavily_api_key configured (set it in {} under the daemon's \
                 working directory)",
                ToolSecrets::RELATIVE_PATH
            )));
        };

        let max_results = req.max_results.unwrap_or(5);
        let body = serde_json::json!({
            "api_key": api_key,
            "query": req.query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": false,
        });

        let resp = ctx
            .http()
            .map_err(|e| SearchError::SearchUnavailable(e.to_string()))?
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .map_err(|e| SearchError::SearchUnavailable(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(SearchError::SearchUnavailable(format!(
                "HTTP {}",
                resp.status()
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .map_err(|e| SearchError::SearchUnavailable(e.to_string()))?;

        let results = json["results"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|r| SearchResult {
                        title: r["title"].as_str().unwrap_or("").to_string(),
                        url: r["url"].as_str().unwrap_or("").to_string(),
                        snippet: r["content"].as_str().unwrap_or("").to_string(),
                        score: r["score"].as_f64().unwrap_or(0.0),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Response {
            searched_on: search_date(Utc::now()),
            results,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebSearchTool>();

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    /// The date leads the response, so it is read before the results it dates.
    #[test]
    fn a_response_says_what_day_it_is_before_its_results() {
        let now = Utc.with_ymd_and_hms(2026, 9, 3, 23, 59, 0).unwrap();
        let resp = Response {
            searched_on: search_date(now),
            results: vec![SearchResult {
                title: "Announcing Rust 1.98.1".into(),
                url: "https://blog.rust-lang.org/".into(),
                snippet: "…".into(),
                score: 0.9,
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(
            json.starts_with(r#"{"searched_on":"2026-09-03","results":["#),
            "{json}"
        );
    }
}
