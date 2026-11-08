//! `web_fetch` tool — fetch a web page and read it as text
//! ([`html_text`]), from the public internet only ([`crate::public_fetch`]).

mod html_text;

use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use self::html_text::html_to_text;
use crate::public_fetch::{public_client, FetchError};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

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
    /// How many pictures the page showed, each marked `[image…]` in
    /// `content`: a table or chart drawn as one is not in the text.
    pub images: usize,
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
         redirects, a truncated flag, and `images`: how many pictures the page showed, each \
         marked `[image: …]` in the text where it stood. A table or chart published as a \
         picture is not in the text — say what you could not read, never that it does not \
         exist. Use web_search when the URL is not yet known. Use \
         http_session_* for authenticated API calls or operations needing cookies/auth state.";

    type Request = Request;
    type Response = Response;
    type Error = FetchError;

    /// Retrieves a page by GET; nothing at the far end changes.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, FetchError> {
        let pinned = public_client(ctx.grants(), &req.url, Duration::from_secs(30))?;
        let resp = pinned
            .client
            .get(&pinned.url)
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
        let page = html_to_text(&html);
        let (title, content, images) = (page.title, page.text, page.images);

        let max_chars = (req.max_tokens.unwrap_or(4000) * 4) as usize;
        let (content, truncated) = if content.len() > max_chars {
            // Cut on a character boundary: a byte index inside a multi-byte
            // character would panic the call.
            let mut end = max_chars;
            while !content.is_char_boundary(end) {
                end -= 1;
            }
            (content[..end].to_string(), true)
        } else {
            (content, false)
        };

        Ok(Response {
            url: req.url,
            final_url,
            title,
            content,
            truncated,
            images,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebFetchTool>();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grants::{Capability, Grants};
    use crate::ToolError;

    /// The fetch itself refuses a local target before any request is sent,
    /// whether it is named or literal.
    #[test]
    fn a_fetch_of_a_local_target_is_blocked() {
        let ctx = ToolContext::new().granting(Grants::NONE.with(Capability::Network));
        for url in [
            "http://127.0.0.1:8081/v1/status",
            "http://localhost:8081/",
            "http://[::ffff:127.0.0.1]/",
            "http://192.168.0.5:8081/",
        ] {
            let err = WebFetchTool::run(
                &ctx,
                Request {
                    url: url.to_string(),
                    max_tokens: None,
                },
            )
            .err()
            .unwrap_or_else(|| panic!("{url} was fetched"));
            assert_eq!(err.code(), "url_blocked", "{url}: {err}");
        }
    }
}
