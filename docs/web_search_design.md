# Gemini-Powered Web Tooling: the `web_*` family

A build guide for a family of web tools that run on the Gemini API. Gemini
handles search, curation, fetching, conversion, extraction, comparison, and
synthesis; each tool shapes one request and parses one structured JSON response
into the `zend-tools` three-type pattern.

The family, by the operation the model chooses between:

| Tool | Operation |
|---|---|
| `web_search` | Grounded Google search → curated, ranked results (+ optional answer) |
| `web_fetch` | Read one or more URLs → clean Markdown content for each (HTML/PDF/image) |
| `web_extract` | Pull caller-defined typed fields from one or more URLs, conforming to a JSON Schema |
| `web_compare` | Read 2–8 URLs → answer a comparative question across them |
| `web_deep_research` | Open-ended objective → multi-search, synthesised, cited report |

The capability this is built on: **Gemini 3.x combines hosted tools (Grounding
with Google Search, URL Context) with Structured Outputs in a single call** — so
search → curate → format into your schema happens in one round trip, no second
parsing pass.

---

## Setup

- **Model:** `gemini-3.5-flash`. Grounding + URL context + structured output are
  supported, and Flash pricing suits high-volume fetched content. Pin a **3.x**
  model — structured output combined with `google_search` requires it.
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent`
- **Auth:** `x-goog-api-key: $GEMINI_API_KEY` header.

Every tool is one POST to this endpoint, differing only in which hosted tool it
enables and which schema it constrains to. The shared helper below centralises
that plumbing so each tool declares only its prompt and schema.

---

## Shared helper: `gemini`

All `web_*` tools issue the same `generateContent` call shape and parse the same
response envelope. Factor it out.

```rust
//! `gemini` — shared helpers for the Gemini-backed web tool family.

use thiserror::Error;

use crate::ToolContext;

pub const MODEL: &str = "gemini-3.5-flash"; // 3.x required for tool + structured output
const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/models";

#[derive(Debug, Error)]
pub enum GeminiError {
    #[error("GEMINI_API_KEY not set")]
    NotConfigured,
    #[error("rate limited")]
    RateLimited,
    #[error("gemini HTTP {0}")]
    Http(u16),
    #[error("gemini transport: {0}")]
    Transport(String),
    #[error("gemini response: {0}")]
    BadResponse(String),
}

/// A real source Gemini retrieved, from groundingMetadata.groundingChunks.
#[derive(Debug, Clone)]
pub struct GroundingSource {
    pub url: String,
    pub title: String,
}

/// Per-URL retrieval outcome from urlContextMetadata — lets a tool tell
/// "Gemini fetched the page" from "Gemini could not retrieve it" (paywall,
/// bot-challenge, 404) without inspecting page bytes itself.
#[derive(Debug, Clone)]
pub struct UrlStatus {
    pub url: String,
    pub retrieved: bool,
}

pub struct GeminiResult {
    /// The structured payload, already parsed from the model's text part.
    pub data: serde_json::Value,
    /// Authoritative sources actually retrieved (use these, not URLs the model
    /// may have written into the payload).
    pub grounding: Vec<GroundingSource>,
    /// Per-URL fetch outcomes for url_context calls (empty for search-only).
    pub url_statuses: Vec<UrlStatus>,
}

/// POST a generateContent body and parse the structured + grounding response.
pub fn generate(ctx: &ToolContext, body: serde_json::Value) -> Result<GeminiResult, GeminiError> {
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        return Err(GeminiError::NotConfigured);
    }

    let url = format!("{ENDPOINT}/{MODEL}:generateContent");
    let resp = ctx
        .http_client
        .post(&url)
        .header("x-goog-api-key", &api_key)
        .json(&body)
        .send()
        .map_err(|e| GeminiError::Transport(e.to_string()))?;

    match resp.status().as_u16() {
        200 => {}
        429 => return Err(GeminiError::RateLimited),
        other => return Err(GeminiError::Http(other)),
    }

    let json: serde_json::Value =
        resp.json().map_err(|e| GeminiError::Transport(e.to_string()))?;
    let cand = &json["candidates"][0];

    // With responseMimeType=application/json the payload is a JSON string in the
    // text part; parse it back into a Value.
    let text = cand["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| GeminiError::BadResponse("no text part".into()))?;
    let data = serde_json::from_str(text)
        .map_err(|e| GeminiError::BadResponse(format!("payload not valid JSON: {e}")))?;

    let grounding = cand["groundingMetadata"]["groundingChunks"]
        .as_array()
        .map(|chunks| {
            chunks
                .iter()
                .filter_map(|c| {
                    let w = &c["web"];
                    Some(GroundingSource {
                        url: w["uri"].as_str()?.to_string(),
                        title: w["title"].as_str().unwrap_or("").to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let url_statuses = cand["urlContextMetadata"]["urlMetadata"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|m| {
                    Some(UrlStatus {
                        url: m["retrievedUrl"].as_str()?.to_string(),
                        retrieved: m["urlRetrievalStatus"].as_str()
                            == Some("URL_RETRIEVAL_STATUS_SUCCESS"),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(GeminiResult { data, grounding, url_statuses })
}
```

---

## `web_search` — grounded, curated, schema-formatted

Query in → Gemini rewrites it into Google searches → searches the live index →
curates, dedupes, ranks → returns results shaped to your schema, plus an optional
synthesised answer.

### Request to Gemini

```jsonc
{
  "contents": [{
    "parts": [{
      "text": "Find the most relevant, recent, and authoritative web results for: <QUERY>. \
               Prefer primary sources. Deduplicate near-identical pages. Return at most N results, \
               each with a concise relevance-focused snippet describing why it answers the query."
    }]
  }],
  "tools": [{ "google_search": {} }],
  "generationConfig": {
    "responseMimeType": "application/json",
    "responseSchema": {
      "type": "object",
      "properties": {
        "answer": { "type": "string" },
        "results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "title":     { "type": "string" },
              "url":       { "type": "string" },
              "snippet":   { "type": "string" },
              "published": { "type": "string" }
            },
            "required": ["title", "url", "snippet"]
          }
        }
      },
      "required": ["results"]
    }
  }
}
```

### Build notes

- **Map URLs from grounding metadata.** Read the real sources from
  `candidates[0].groundingMetadata.groundingChunks` and use those as the
  authoritative URL set.
- **`answer` is optional.** Include it when this tool acts as a one-shot
  answerer; drop it from the schema when you only want curated URLs to feed
  `web_fetch`.

### Tool (three-type pattern)

```rust
#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    #[validate(length(min = 1))]
    pub query: String,
    #[validate(range(min = 1, max = 10))]
    /// Maximum results to return. Defaults to 5.
    pub max_results: Option<u8>,
    /// When true, include a synthesised answer alongside curated results.
    pub want_answer: Option<bool>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub published: Option<String>,
}

#[derive(Serialize)]
pub struct Response {
    pub answer: Option<String>,
    pub results: Vec<SearchResult>,
}
```

Error codes: `search_not_configured`, `rate_limited`, `search_unavailable`,
`grounding_empty`.

---

## `web_fetch` — read one or more URLs via Gemini, no local fetch

URLs in → Gemini fetches each page server-side (via `url_context`) → converts
each to clean Markdown with headings, lists, and tables preserved → returns one
content entry per URL. HTML, PDF, and images are all handled by the same call;
Gemini detects the type. Pass a single URL for one page, or several to read them
in one round trip (e.g. the top results from `web_search`). Nothing is fetched
locally, so there is no SSRF guard, no redirect handling, and no Cloudflare
exposure on your side — any URL Gemini could not retrieve is reported in
`unretrieved` rather than silently dropped.

### Conversion prompt (transcribe, don't summarise)

```
For EACH URL below, extract that document's full textual content as clean
Markdown. Preserve headings, lists, and tables. Report each document's type
(html, pdf, image, or other). Return one entry per URL. Do NOT summarise, add
commentary, or omit sections. Output only the structured result.
```

### Request to Gemini (`url_context`, structured)

```jsonc
{
  "contents": [{
    "parts": [{ "text": "<transcription prompt above>\n\nURLs:\n<URL 1>\n<URL 2>" }]
  }],
  "tools": [{ "url_context": {} }],
  "generationConfig": {
    "responseMimeType": "application/json",
    "responseSchema": {
      "type": "object",
      "properties": {
        "documents": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "url":      { "type": "string" },
              "title":    { "type": "string" },
              "content":  { "type": "string" },
              "doc_type": { "type": "string" }
            },
            "required": ["url", "content", "doc_type"]
          }
        }
      },
      "required": ["documents"]
    }
  }
}
```

### Build notes

- **`unretrieved` comes from retrieval status, not page content.** Build it from
  `url_statuses` (parsed from `urlContextMetadata`); any URL that came back
  not-retrieved goes there. No interstitial-HTML sniffing — Gemini fetched it, so
  you read a status field instead of pattern-matching block pages. The model sees
  exactly what was read and what was missed.
- **Type detection is reported, not sniffed.** Gemini fills each `doc_type`; no
  local `Content-Type` inspection.
- **Truncation stays UTF-8-safe** (`char_indices().nth()`) and applies per
  document.

### Tool (three-type pattern)

```rust
//! `web_fetch` tool — read one or more URLs' content via Gemini url_context.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::gemini::{self, GeminiError};
use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// One or more URLs to read. Pass one for a single page, or several to read
    /// them in a single call.
    #[validate(length(min = 1, max = 8))]
    pub urls: Vec<String>,
    /// Per-document content cap in tokens. Defaults to 4000.
    #[validate(range(min = 500, max = 16000))]
    pub max_tokens: Option<u32>,
}

#[derive(Serialize)]
pub struct FetchedDoc {
    pub url: String,
    pub title: String,
    pub content: String,
    pub truncated: bool,
    /// Document type as detected by Gemini: "html" | "pdf" | "image" | "other".
    pub doc_type: String,
}

#[derive(Serialize)]
pub struct Response {
    /// One entry per successfully read URL.
    pub documents: Vec<FetchedDoc>,
    /// URLs Gemini could not retrieve (paywalled, challenged, or unavailable).
    pub unretrieved: Vec<String>,
}

#[derive(serde::Deserialize)]
struct RawDoc {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    content: String,
    #[serde(default)]
    doc_type: String,
}

#[derive(serde::Deserialize)]
struct Payload {
    #[serde(default)]
    documents: Vec<RawDoc>,
}

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("fetch unavailable: {0}")]
    Unavailable(String),
    #[error("rate limited")]
    RateLimited,
}

impl ToolError for FetchError {
    fn code(&self) -> &'static str {
        match self {
            FetchError::UrlBlocked(_) => "url_blocked",
            FetchError::Unavailable(_) => "fetch_unavailable",
            FetchError::RateLimited => "rate_limited",
        }
    }
}

impl From<GeminiError> for FetchError {
    fn from(e: GeminiError) -> Self {
        match e {
            GeminiError::RateLimited => FetchError::RateLimited,
            other => FetchError::Unavailable(other.to_string()),
        }
    }
}

/// UTF-8-safe truncation on a char boundary.
fn truncate_tokens(s: String, max_tokens: u32) -> (String, bool) {
    let max_chars = (max_tokens as usize) * 4;
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => (s[..idx].to_string(), true),
        None => (s, false),
    }
}

pub struct WebFetchTool;

impl Tool for WebFetchTool {
    const NAME: &'static str = "web_fetch";
    const DESCRIPTION: &'static str =
        "Read one or more public web pages or documents by URL and return each one's main content \
         as clean Markdown. Handles HTML, PDFs, and images (tables and layout preserved). Pass a \
         single URL to read one page, or several URLs (e.g. top results from web_search) to read \
         them in one call. Use for: reading a specific article the user linked, retrieving \
         documentation, pulling a PDF report, reading several sources before answering. Triggered \
         by \"read this page\", \"fetch the article at\", \"what does this URL say\", \"open this \
         link\", \"get me the content of\", \"download the PDF at\", \"read these\", or the user \
         pasting URLs. Returns a document per URL (title, content, detected type) plus a list of \
         any URLs that could not be retrieved. Use web_search when the URL is not yet known; use \
         web_extract to pull specific typed fields rather than full prose; use web_compare to \
         answer a single question across several URLs.";

    type Request = Request;
    type Response = Response;
    type Error = FetchError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, FetchError> {
        for u in &req.urls {
            if !u.starts_with("http://") && !u.starts_with("https://") {
                return Err(FetchError::UrlBlocked(format!("invalid URL: {u}")));
            }
        }
        let url_list = req.urls.join("\n");

        let prompt = format!(
            "For EACH URL below, extract that document's full textual content as clean Markdown. \
             Preserve headings, lists, and tables. Report each document's type (html, pdf, image, \
             or other). Return one entry per URL. Do NOT summarise, add commentary, or omit \
             sections. Output only the structured result.\n\nURLs:\n{url_list}"
        );

        let body = serde_json::json!({
            "contents": [{ "parts": [{ "text": prompt }]}],
            "tools": [{ "url_context": {} }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url":      { "type": "string" },
                                    "title":    { "type": "string" },
                                    "content":  { "type": "string" },
                                    "doc_type": { "type": "string" }
                                },
                                "required": ["url", "content", "doc_type"]
                            }
                        }
                    },
                    "required": ["documents"]
                }
            }
        });

        let result = gemini::generate(ctx, body)?;
        let payload: Payload = serde_json::from_value(result.data)
            .map_err(|e| FetchError::Unavailable(format!("bad payload: {e}")))?;

        let max_tokens = req.max_tokens.unwrap_or(4000);
        let documents = payload
            .documents
            .into_iter()
            .map(|d| {
                let (content, truncated) = truncate_tokens(d.content, max_tokens);
                FetchedDoc { url: d.url, title: d.title, content, truncated, doc_type: d.doc_type }
            })
            .collect();

        let unretrieved = result
            .url_statuses
            .iter()
            .filter(|s| !s.retrieved)
            .map(|s| s.url.clone())
            .collect();

        Ok(Response { documents, unretrieved })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebFetchTool>();
```

> **No local fetch, anywhere in this family.** Every tool reaches the web through
> Gemini's hosted tools (`google_search` or `url_context`), which run
> server-side on Google's network. Your process never makes an outbound web
> request to an arbitrary URL, so there is no SSRF surface and no private-URL
> guard to maintain in any tool. Walled pages (Cloudflare challenges, logins)
> fail with a retrieval status you can read — they are not accessible, but they
> also cost you no bot-war maintenance.

---

## `web_extract` — typed fields from one or more URLs via caller schema

The model supplies a JSON Schema describing exactly the fields it wants; that
schema is applied to each URL. One tool, unlimited extraction shapes, one or many
pages per call.

### Tool (three-type pattern)

```rust
//! `web_extract` tool — pull caller-defined typed fields from a URL.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::gemini::{self, GeminiError};
use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// One or more page/document URLs to extract data from. The same schema is
    /// applied to each.
    #[validate(length(min = 1, max = 8))]
    pub urls: Vec<String>,
    /// A JSON Schema OBJECT describing exactly the fields to extract from each
    /// URL. Each result's `data` will conform to it. Example:
    /// {"type":"object","properties":{"price":{"type":"number"},
    /// "name":{"type":"string"}},"required":["name"]}
    pub schema: serde_json::Value,
}

#[derive(Serialize)]
pub struct ExtractResult {
    pub url: String,
    /// Extracted data, conforming to the requested schema.
    pub data: serde_json::Value,
}

#[derive(Serialize)]
pub struct Response {
    /// One entry per URL Gemini could read.
    pub results: Vec<ExtractResult>,
    /// URLs Gemini could not retrieve.
    pub unretrieved: Vec<String>,
}

#[derive(serde::Deserialize)]
struct RawResult {
    #[serde(default)]
    url: String,
    data: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct Payload {
    #[serde(default)]
    results: Vec<RawResult>,
}

#[derive(Debug, Error)]
pub enum ExtractError {
    #[error("invalid schema: {0}")]
    InvalidSchema(String),
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("extraction unavailable: {0}")]
    Unavailable(String),
    #[error("rate limited")]
    RateLimited,
}

impl ToolError for ExtractError {
    fn code(&self) -> &'static str {
        match self {
            ExtractError::InvalidSchema(_) => "invalid_schema",
            ExtractError::UrlBlocked(_) => "url_blocked",
            ExtractError::Unavailable(_) => "extract_unavailable",
            ExtractError::RateLimited => "rate_limited",
        }
    }
}

impl From<GeminiError> for ExtractError {
    fn from(e: GeminiError) -> Self {
        match e {
            GeminiError::RateLimited => ExtractError::RateLimited,
            other => ExtractError::Unavailable(other.to_string()),
        }
    }
}

pub struct WebExtractTool;

impl Tool for WebExtractTool {
    const NAME: &'static str = "web_extract";
    const DESCRIPTION: &'static str =
        "Extract specific, typed data fields from one or more web pages or documents by URL, \
         conforming to a JSON Schema you provide. Use when you know exactly what data you want: \
         prices and specs from product pages, rows from a table, dates/parties from a filing, \
         contact details. Supply `urls` and a `schema` (a JSON Schema object); each result's \
         `data` matches it. Pass several URLs to extract the same fields from each in one call. \
         Triggered by \"extract the X from\", \"get the price/specs/table from\", \"pull the \
         structured data\", \"parse fields from these pages\". If you only need to read a page as \
         prose, prefer web_fetch; use web_search when the URLs are not yet known.";

    type Request = Request;
    type Response = Response;
    type Error = ExtractError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, ExtractError> {
        for u in &req.urls {
            if !u.starts_with("http://") && !u.starts_with("https://") {
                return Err(ExtractError::UrlBlocked(format!("invalid URL: {u}")));
            }
        }
        if !req.schema.is_object() {
            return Err(ExtractError::InvalidSchema("schema must be a JSON object".into()));
        }
        let url_list = req.urls.join("\n");

        // Wrap the caller's per-URL schema in a results array keyed by URL.
        let response_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url":  { "type": "string" },
                            "data": req.schema
                        },
                        "required": ["url", "data"]
                    }
                }
            },
            "required": ["results"]
        });

        let body = serde_json::json!({
            "contents": [{ "parts": [{ "text": format!(
                "For EACH URL below, extract data strictly matching the `data` schema. Use only \
                 information present on that page; if a field is absent, omit it or use null per \
                 the schema. Do not invent values. Return one result object per URL.\n\nURLs:\n{}",
                url_list
            )}]}],
            "tools": [{ "url_context": {} }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            }
        });

        let result = gemini::generate(ctx, body)?;
        let payload: Payload = serde_json::from_value(result.data)
            .map_err(|e| ExtractError::Unavailable(format!("bad payload: {e}")))?;

        let results = payload
            .results
            .into_iter()
            .map(|r| ExtractResult { url: r.url, data: r.data })
            .collect();

        let unretrieved = result
            .url_statuses
            .iter()
            .filter(|s| !s.retrieved)
            .map(|s| s.url.clone())
            .collect();

        Ok(Response { results, unretrieved })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebExtractTool>();
```

> **Schema caveat.** Gemini supports only a subset of JSON Schema. A caller schema
> using exotic keywords may return a Gemini 400, surfacing as
> `extract_unavailable`. If common in practice, sanitise the schema to the
> supported subset before forwarding.

---

## `web_compare` — read multiple URLs, answer a comparative question

`url_context` accepts multiple URLs in one call. Returns a fixed comparison
shape: summary, concrete differences, and each source's position.

### Tool (three-type pattern)

```rust
//! `web_compare` tool — read multiple URLs and answer a comparative question.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::gemini::{self, GeminiError};
use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// 2–8 URLs to compare.
    #[validate(length(min = 2, max = 8))]
    pub urls: Vec<String>,
    /// The comparative question, e.g. "how do these pricing pages differ?"
    #[validate(length(min = 1))]
    pub question: String,
}

#[derive(Serialize, serde::Deserialize)]
pub struct SourceNote {
    pub url: String,
    pub position: String,
}

#[derive(Serialize)]
pub struct Response {
    pub summary: String,
    pub differences: Vec<String>,
    pub per_source: Vec<SourceNote>,
    /// URLs actually read for the comparison.
    pub source_urls: Vec<String>,
    /// URLs that could not be retrieved (the comparison excludes these).
    pub unretrieved: Vec<String>,
}

#[derive(serde::Deserialize)]
struct Payload {
    summary: String,
    differences: Vec<String>,
    per_source: Vec<SourceNote>,
}

#[derive(Debug, Error)]
pub enum CompareError {
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("comparison unavailable: {0}")]
    Unavailable(String),
    #[error("rate limited")]
    RateLimited,
}

impl ToolError for CompareError {
    fn code(&self) -> &'static str {
        match self {
            CompareError::UrlBlocked(_) => "url_blocked",
            CompareError::Unavailable(_) => "compare_unavailable",
            CompareError::RateLimited => "rate_limited",
        }
    }
}

impl From<GeminiError> for CompareError {
    fn from(e: GeminiError) -> Self {
        match e {
            GeminiError::RateLimited => CompareError::RateLimited,
            other => CompareError::Unavailable(other.to_string()),
        }
    }
}

pub struct WebCompareTool;

impl Tool for WebCompareTool {
    const NAME: &'static str = "web_compare";
    const DESCRIPTION: &'static str =
        "Read 2–8 web pages or documents by URL and answer a comparative question across them — \
         reconciling sources, identifying differences, tracking changes between versions. Supply \
         `urls` and a `question`. Returns a summary, a list of concrete differences, and each \
         source's position. Triggered by \"compare these\", \"how do X and Y differ\", \
         \"reconcile these sources\", \"what changed between\", \"which of these says\". Use \
         web_fetch for a single URL's raw content; use web_search to find the URLs first.";

    type Request = Request;
    type Response = Response;
    type Error = CompareError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, CompareError> {
        for u in &req.urls {
            if !u.starts_with("http://") && !u.starts_with("https://") {
                return Err(CompareError::UrlBlocked(format!("invalid URL: {u}")));
            }
        }
        let url_list = req.urls.join("\n");

        let body = serde_json::json!({
            "contents": [{ "parts": [{ "text": format!(
                "Read ALL of the following URLs and answer the comparative question. Base every \
                 statement only on the page contents; do not use outside knowledge.\n\n\
                 Question: {}\n\nURLs:\n{}", req.question, url_list
            )}]}],
            "tools": [{ "url_context": {} }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string" },
                        "differences": { "type": "array", "items": { "type": "string" } },
                        "per_source": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": { "type": "string" },
                                    "position": { "type": "string" }
                                },
                                "required": ["url", "position"]
                            }
                        }
                    },
                    "required": ["summary", "differences", "per_source"]
                }
            }
        });

        let result = gemini::generate(ctx, body)?;
        let payload: Payload = serde_json::from_value(result.data)
            .map_err(|e| CompareError::Unavailable(format!("bad payload: {e}")))?;

        // url_context populates url_statuses (not grounding, which is search-only),
        // so source attribution comes from there.
        let source_urls = result
            .url_statuses
            .iter()
            .filter(|s| s.retrieved)
            .map(|s| s.url.clone())
            .collect();
        let unretrieved = result
            .url_statuses
            .iter()
            .filter(|s| !s.retrieved)
            .map(|s| s.url.clone())
            .collect();

        Ok(Response {
            summary: payload.summary,
            differences: payload.differences,
            per_source: payload.per_source,
            source_urls,
            unretrieved,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebCompareTool>();
```

---

## `web_deep_research` — grounded, multi-search, cited report

One grounded call; the report comes from the structured payload, citations from
grounding metadata (authoritative sources, never URLs the model wrote into prose).
Named distinctly from `web_search` so the model isn't choosing between two
near-identical names — this is the heavy, multi-source one.

### Tool (three-type pattern)

```rust
//! `web_deep_research` tool — grounded, multi-query research returning a cited report.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::gemini::{self, GeminiError};
use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// The research objective, e.g. "current state of RISC-V server adoption".
    #[validate(length(min = 1))]
    pub objective: String,
    /// How broadly to research. Defaults to balanced.
    pub depth: Option<Depth>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Depth {
    Quick,
    Balanced,
    Thorough,
}

impl Depth {
    fn hint(&self) -> &'static str {
        match self {
            Depth::Quick => "Run 1–2 focused searches; concise report.",
            Depth::Balanced => "Decompose into a few sub-questions; search each; balanced report.",
            Depth::Thorough => "Decompose broadly; search many angles; comprehensive report.",
        }
    }
}

#[derive(Serialize)]
pub struct Citation {
    pub url: String,
    pub title: String,
}

#[derive(Serialize)]
pub struct Response {
    /// The synthesised report, in Markdown.
    pub report: String,
    /// Sources actually retrieved during research.
    pub citations: Vec<Citation>,
}

#[derive(serde::Deserialize)]
struct Payload {
    report: String,
}

#[derive(Debug, Error)]
pub enum ResearchError {
    #[error("research unavailable: {0}")]
    Unavailable(String),
    #[error("no grounded results")]
    GroundingEmpty,
    #[error("rate limited")]
    RateLimited,
}

impl ToolError for ResearchError {
    fn code(&self) -> &'static str {
        match self {
            ResearchError::Unavailable(_) => "research_unavailable",
            ResearchError::GroundingEmpty => "grounding_empty",
            ResearchError::RateLimited => "rate_limited",
        }
    }
}

impl From<GeminiError> for ResearchError {
    fn from(e: GeminiError) -> Self {
        match e {
            GeminiError::RateLimited => ResearchError::RateLimited,
            other => ResearchError::Unavailable(other.to_string()),
        }
    }
}

pub struct WebDeepResearchTool;

impl Tool for WebDeepResearchTool {
    const NAME: &'static str = "web_deep_research";
    const DESCRIPTION: &'static str =
        "Research an open-ended objective across the live web and return a synthesised, cited \
         report. Runs multiple grounded searches internally and reconciles them. Use for broad \
         questions needing several sources: \"research the current state of X\", \"give me a \
         briefing on Y\", \"investigate Z and summarise findings\". Returns a Markdown report \
         plus citations. This is the heavy, multi-source tool — use web_search for a quick lookup \
         or a single fact, and web_deep_research when you want a finished, sourced answer.";

    type Request = Request;
    type Response = Response;
    type Error = ResearchError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, ResearchError> {
        let depth = req.depth.unwrap_or(Depth::Balanced);

        let body = serde_json::json!({
            "contents": [{ "parts": [{ "text": format!(
                "Research the following objective thoroughly using web search. {} Ground every \
                 claim in retrieved sources. Write a clear, well-structured Markdown report. Do \
                 not fabricate sources.\n\nObjective: {}", depth.hint(), req.objective
            )}]}],
            "tools": [{ "google_search": {} }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": { "report": { "type": "string" } },
                    "required": ["report"]
                }
            }
        });

        let result = gemini::generate(ctx, body)?;
        if result.grounding.is_empty() {
            return Err(ResearchError::GroundingEmpty);
        }
        let payload: Payload = serde_json::from_value(result.data)
            .map_err(|e| ResearchError::Unavailable(format!("bad payload: {e}")))?;

        Ok(Response {
            report: payload.report,
            citations: result
                .grounding
                .into_iter()
                .map(|g| Citation { url: g.url, title: g.title })
                .collect(),
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebDeepResearchTool>();
```

---

## Wiring

Declare the shared helper and the tool modules, then register all five:

```rust
mod gemini;
mod web_search;
mod web_fetch;
mod web_extract;
mod web_compare;
mod web_deep_research;

// wherever the registry collects RegisteredTool consts:
web_search::REGISTRATION,
web_fetch::REGISTRATION,
web_extract::REGISTRATION,
web_compare::REGISTRATION,
web_deep_research::REGISTRATION,
```

---

## Build checklist

- [ ] Pin a **Gemini 3.x** model; confirm grounding + structured output returns
      200.
- [ ] Set `GEMINI_API_KEY`.
- [ ] Add the shared `gemini` helper module.
- [ ] `web_search`: grounded call with `responseSchema`; take URLs from
      `groundingMetadata.groundingChunks`.
- [ ] `web_fetch`: `url_context` call returning a `documents` array (one per URL);
      build `unretrieved` from `url_statuses`; UTF-8-safe per-document truncation.
- [ ] `web_extract`: validate `schema` is an object; wrap it per-URL in a
      `results` array; build `unretrieved` from `url_statuses`.
- [ ] `web_compare`: 2–8 URLs in prompt; fixed comparison schema; source_urls and
      `unretrieved` from `url_statuses` (not grounding).
- [ ] `web_deep_research`: grounded report; citations from grounding metadata;
      `grounding_empty` distinct from failure.
- [ ] Register all five `REGISTRATION` consts.
- [ ] Confirm reqwest `gzip` feature state matches your header choices.