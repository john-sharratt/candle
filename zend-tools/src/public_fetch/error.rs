//! Why a fetch did not happen.

use thiserror::Error;

use crate::ToolError;

#[derive(Debug, Error)]
pub enum FetchError {
    /// The URL, or an address it or a redirect resolves to, is not one a fetch
    /// may reach.
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("fetch failed: {0}")]
    FetchFailed(String),
    #[error("HTTP error {status}: {detail}")]
    HttpError { status: u16, detail: String },
    /// The body ran past its cap, and was abandoned there.
    #[error("the answer is larger than {cap} bytes")]
    TooLarge { cap: usize },
}

impl ToolError for FetchError {
    fn code(&self) -> &'static str {
        match self {
            FetchError::UrlBlocked(_) => "url_blocked",
            FetchError::FetchFailed(_) => "fetch_failed",
            FetchError::HttpError { .. } => "http_error",
            FetchError::TooLarge { .. } => "too_large",
        }
    }

    fn detail(&self) -> String {
        match self {
            FetchError::HttpError { status, detail } => format!("HTTP {status}: {detail}"),
            _ => self.to_string(),
        }
    }
}
