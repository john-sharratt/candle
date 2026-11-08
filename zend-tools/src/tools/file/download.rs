//! `file_download`: what a public URL serves, written to a file of the
//! session's checkout without passing through the conversation
//! (`docs/mcp_file_tools.md` §3.6). The answer is the file's size, digest and
//! type, never its bytes; it goes into the repository with the worker's other
//! changes, through the repository's gate.

use std::time::Duration;

use reqwest::header::CONTENT_TYPE;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use validator::Validate;

use super::transfer_error::TransferError;
use super::FileError;
use crate::public_fetch::{public_client, read_within, FetchError};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// The most bytes one download writes.
pub const MAX_DOWNLOAD_BYTES: usize = 8 * 1024 * 1024;

/// How long a download may take, end to end.
const WITHIN: Duration = Duration::from_secs(30);

#[derive(Deserialize, JsonSchema, Validate)]
pub struct DownloadRequest {
    /// The repository the file goes in. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The file to write, relative to the repository (e.g. `library/art/moon.png`); created or replaced in this session. Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// What to fetch: an `http://` or `https://` URL on the public internet. Private and local addresses are refused, and so is every redirect to one. Required.
    #[validate(length(min = 1))]
    pub url: String,
}

#[derive(Serialize)]
pub struct DownloadResponse {
    pub repo: String,
    pub path: String,
    pub bytes: usize,
    /// The file's SHA-256, in hex.
    pub sha256: String,
    /// What the server said it is, when it said.
    pub content_type: Option<String>,
    pub created: bool,
}

pub struct FileDownload;

impl Tool for FileDownload {
    const NAME: &'static str = "file_download";
    const DESCRIPTION: &'static str =
        "Download what a public URL serves into a file of the repository — a picture, a font, \
         a library, a raw file from another repository — without reading it into the \
         conversation. Use it instead of web_fetch or http_request plus write whenever the \
         file itself is wanted, binary or text. Answers the file's size, SHA-256 and type, \
         never its bytes; the file is committed with the rest of your changes. At most 8 MiB, \
         30 seconds, five redirects; private and local addresses are refused. Triggered by \
         \"download this into\", \"save the picture at\", \"put that file in the repo\", \
         \"vendor this library\".";

    type Request = DownloadRequest;
    type Response = DownloadResponse;
    type Error = TransferError;

    /// The file ends up holding what the URL serves, so downloading it again
    /// leaves what the first download left.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: DownloadRequest) -> Result<DownloadResponse, TransferError> {
        let pinned = public_client(ctx.grants(), &req.url, WITHIN)?;
        let store = ctx.files.repo(&req.repo).map_err(FileError::from)?;
        let resp = pinned
            .client
            .get(&pinned.url)
            .send()
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(FetchError::HttpError {
                status: status.as_u16(),
                detail: status.canonical_reason().unwrap_or("").to_string(),
            }
            .into());
        }
        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let bytes = read_within(resp, MAX_DOWNLOAD_BYTES)?;
        let len = bytes.len();
        let sha256 = hex::encode(Sha256::digest(&bytes));
        let created = store
            .write_bytes(&req.path, bytes)
            .map_err(FileError::from)?;
        Ok(DownloadResponse {
            repo: req.repo,
            path: req.path,
            bytes: len,
            sha256,
            content_type,
            created,
        })
    }
}

pub const FILE_DOWNLOAD: RegisteredTool = RegisteredTool::new::<FileDownload>();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grants::{Capability, Grants};
    use crate::ToolError;

    /// **A local target is refused before anything is fetched or written.**
    #[test]
    fn a_local_target_is_refused() {
        let ctx = ToolContext::new().granting(Grants::NONE.with(Capability::Network));
        for url in [
            "http://127.0.0.1:8092/",
            "http://localhost/",
            "http://10.0.0.1/a.png",
        ] {
            let err = FileDownload::run(
                &ctx,
                DownloadRequest {
                    repo: "candle".into(),
                    path: "a.png".into(),
                    url: url.into(),
                },
            )
            .err()
            .unwrap_or_else(|| panic!("{url} was fetched"));
            assert_eq!(err.code(), "url_blocked", "{url}");
        }
    }
}
