//! `web_search` tool — search the web via Tavily API.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    #[validate(length(min = 1))]
    pub query: String,
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
    pub results: Vec<SearchResult>,
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
         \"recent news on\", \"google\", \"search the web\". Returns up to 10 ranked results. \
         For DNS records use dns_lookup; for fetching a specific URL already known, use \
         web_fetch; for authenticated API calls use http_session_*.";

    type Request = Request;
    type Response = Response;
    type Error = SearchError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, SearchError> {
        let api_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            return Err(SearchError::SearchUnavailable(
                "TAVILY_API_KEY not set".to_string(),
            ));
        }

        let max_results = req.max_results.unwrap_or(5);
        let body = serde_json::json!({
            "api_key": api_key,
            "query": req.query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": false,
        });

        let resp = ctx
            .http_client
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

        Ok(Response { results })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebSearchTool>();
