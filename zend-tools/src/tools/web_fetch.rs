//! `web_fetch` tool — fetch and clean a web page.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// URL to fetch; must start with http:// or https://. Private/localhost URLs are blocked.
    #[validate(length(min = 1))]
    pub url: String,
    /// Approximate max tokens of content to return (500-16000); content is truncated
    /// at roughly 4 chars per token. Default: 4000.
    #[validate(range(min = 500, max = 16000))]
    pub max_tokens: Option<u32>,
}

#[derive(Serialize)]
pub struct Response {
    pub url: String,
    pub final_url: String,
    pub title: String,
    pub content: String,
    pub truncated: bool,
}

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("fetch failed: {0}")]
    FetchFailed(String),
    #[error("HTTP error {status}: {detail}")]
    HttpError { status: u16, detail: String },
}

impl ToolError for FetchError {
    fn code(&self) -> &'static str {
        match self {
            FetchError::UrlBlocked(_) => "url_blocked",
            FetchError::FetchFailed(_) => "fetch_failed",
            FetchError::HttpError { .. } => "http_error",
        }
    }

    fn detail(&self) -> String {
        match self {
            FetchError::HttpError { status, detail } => {
                // Include status in detail as JSON-embedded
                format!("HTTP {status}: {detail}")
            }
            _ => self.to_string(),
        }
    }
}

pub fn is_private_url(url_str: &str) -> bool {
    let Ok(parsed) = url::Url::parse(url_str) else {
        return true;
    };
    let host = parsed.host_str().unwrap_or("");
    if host == "localhost" || host == "127.0.0.1" || host == "::1" {
        return true;
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return is_private_ip(ip);
    }
    false
}

pub fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                // 169.254.x.x
                || (v4.octets()[0] == 169 && v4.octets()[1] == 254)
        }
        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
    }
}

fn html_to_text(html: &str) -> (String, String) {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    // Extract title
    let title_sel = Selector::parse("title").unwrap();
    let title = document
        .select(&title_sel)
        .next()
        .map(|e| e.text().collect::<String>().trim().to_string())
        .unwrap_or_default();

    // Build text from body
    let body_sel = Selector::parse("body").unwrap();
    let body = document.select(&body_sel).next();

    let mut text = String::new();
    if let Some(body) = body {
        for node in body.descendants() {
            if let Some(elem) = node.value().as_element() {
                match elem.name() {
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                        let level = elem
                            .name()
                            .chars()
                            .nth(1)
                            .unwrap_or('1')
                            .to_digit(10)
                            .unwrap_or(1) as usize;
                        let hashes = "#".repeat(level);
                        if let Some(t) = scraper::ElementRef::wrap(node) {
                            let content: String = t.text().collect();
                            text.push_str(&format!("\n{} {}\n", hashes, content.trim()));
                        }
                    }
                    "p" => {
                        if let Some(t) = scraper::ElementRef::wrap(node) {
                            let content: String = t.text().collect();
                            let trimmed = content.trim();
                            if !trimmed.is_empty() {
                                text.push('\n');
                                text.push_str(trimmed);
                                text.push('\n');
                            }
                        }
                    }
                    "script" | "style" | "nav" | "header" | "footer" => {}
                    _ => {}
                }
            } else if let Some(txt) = node.value().as_text() {
                let s = txt.trim();
                if !s.is_empty() && s.len() > 2 {
                    // avoid noise
                }
            }
        }
        // If text is sparse, fall back to simple text extraction
        if text.trim().len() < 100 {
            text = body.text().collect::<Vec<_>>().join(" ");
            // Collapse whitespace
            text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        }
    }

    (title, text)
}

pub struct WebFetchTool;

impl Tool for WebFetchTool {
    const NAME: &'static str = "web_fetch";
    const DESCRIPTION: &'static str =
        "Fetch a single public web page or document by URL and return its main content as \
         cleaned text. Use for: reading a specific article the user linked, retrieving \
         documentation pages, pulling content from a known URL, getting context about a page \
         the user mentioned. Triggered by \"read this page\", \"fetch the article at\", \"what \
         does this URL say\", \"open this link\", \"get me the content of\", \"summarise this \
         page\", or the user pasting a URL. Returns title, cleaned body text, final URL after \
         redirects, and a truncated flag. Use web_search when the URL is not yet known. Use \
         http_session_* for authenticated API calls or operations needing cookies/auth state.";

    type Request = Request;
    type Response = Response;
    type Error = FetchError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, FetchError> {
        if !req.url.starts_with("http://") && !req.url.starts_with("https://") {
            return Err(FetchError::UrlBlocked(
                "URL must start with http:// or https://".to_string(),
            ));
        }
        if is_private_url(&req.url) {
            return Err(FetchError::UrlBlocked(
                "private/localhost URLs are not allowed".to_string(),
            ));
        }

        let resp = ctx
            .http_client
            .get(&req.url)
            .header("User-Agent", "Mozilla/5.0 (compatible; zend-tools/0.1)")
            .send()
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?;

        let status = resp.status();
        let final_url = resp.url().to_string();

        if !status.is_success() {
            return Err(FetchError::HttpError {
                status: status.as_u16(),
                detail: status.canonical_reason().unwrap_or("").to_string(),
            });
        }

        let html = resp
            .text()
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?;
        let (title, content) = html_to_text(&html);

        let max_chars = (req.max_tokens.unwrap_or(4000) * 4) as usize;
        let (content, truncated) = if content.len() > max_chars {
            (content[..max_chars].to_string(), true)
        } else {
            (content, false)
        };

        Ok(Response {
            url: req.url,
            final_url,
            title,
            content,
            truncated,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebFetchTool>();
