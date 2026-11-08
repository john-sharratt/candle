//! `file_upload`: a file of the session's checkout sent to a public URL by
//! `PUT` or `POST`, without passing through the conversation
//! (`docs/mcp_file_tools.md` §3.6). The answer is the server's status and the
//! first 4 KiB of what it said back.

use std::time::Duration;

use reqwest::header::CONTENT_TYPE;
use reqwest::Method;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::transfer_error::TransferError;
use super::FileError;
use crate::public_fetch::{public_client, read_head, FetchError};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// The most of the server's answer shown.
const HEAD: usize = 4 * 1024;

/// How long an upload may take, end to end.
const WITHIN: Duration = Duration::from_secs(60);

#[derive(Deserialize, JsonSchema, Validate)]
pub struct UploadRequest {
    /// The repository the file is in. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The file to send, relative to the repository, as this session holds it. Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Where to send it: an `http://` or `https://` URL on the public internet. Private and local addresses are refused. Required.
    #[validate(length(min = 1))]
    pub url: String,
    /// `PUT` (the default) or `POST`.
    pub method: Option<String>,
    /// The `Content-Type` to send; none when left out.
    pub content_type: Option<String>,
}

#[derive(Serialize)]
pub struct UploadResponse {
    pub status: u16,
    /// The first 4 KiB of the server's answer.
    pub answer: String,
    /// Whether the server said more than that.
    pub truncated: bool,
    pub bytes_sent: usize,
}

pub struct FileUpload;

impl Tool for FileUpload {
    const NAME: &'static str = "file_upload";
    const DESCRIPTION: &'static str =
        "Send a file of the repository to a public URL by PUT (the default) or POST, without \
         reading it into the conversation — a built artefact to a release host, a picture to \
         a service that takes uploads. Answers the server's status and the first 4 KiB of its \
         answer. Private and local addresses are refused; 60 seconds at most. For calls that \
         need a sign-in or cookies use http_session_*.";

    type Request = UploadRequest;
    type Response = UploadResponse;
    type Error = TransferError;

    /// Sending it again sends it again: the far end may keep both.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Unsafe
    }

    fn run(ctx: &ToolContext, req: UploadRequest) -> Result<UploadResponse, TransferError> {
        let method = match req
            .method
            .as_deref()
            .map(str::to_ascii_uppercase)
            .as_deref()
        {
            None | Some("PUT") => Method::PUT,
            Some("POST") => Method::POST,
            Some(other) => {
                return Err(FileError::InvalidArguments(format!(
                    "`method` is PUT or POST, not {other}"
                ))
                .into())
            }
        };
        let pinned = public_client(ctx.grants(), &req.url, WITHIN)?;
        let store = ctx.files.repo(&req.repo).map_err(FileError::from)?;
        let bytes = store
            .read_bytes(&req.path)
            .map_err(FileError::from)?
            .ok_or_else(|| FileError::NotFound(req.path.clone()))?;
        let sent = bytes.len();
        let mut request = pinned.client.request(method, &pinned.url).body(bytes);
        if let Some(t) = &req.content_type {
            request = request.header(CONTENT_TYPE, t);
        }
        let resp = request
            .send()
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?;
        let status = resp.status().as_u16();
        let (answer, truncated) = read_head(resp, HEAD);
        Ok(UploadResponse {
            status,
            answer,
            truncated,
            bytes_sent: sent,
        })
    }
}

pub const FILE_UPLOAD: RegisteredTool = RegisteredTool::new::<FileUpload>();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grants::{Capability, Grants};
    use crate::ToolError;

    fn request(url: &str, method: Option<&str>) -> UploadRequest {
        UploadRequest {
            repo: "candle".into(),
            path: "a.bin".into(),
            url: url.into(),
            method: method.map(Into::into),
            content_type: None,
        }
    }

    /// **A local target is refused before anything is read or sent**, and a
    /// method other than PUT or POST is refused as an argument.
    #[test]
    fn a_local_target_and_another_method_are_refused() {
        let ctx = ToolContext::new().granting(Grants::NONE.with(Capability::Network));
        let err = FileUpload::run(&ctx, request("http://127.0.0.1:8092/stage", None))
            .err()
            .expect("refused");
        assert_eq!(err.code(), "url_blocked");
        let err = FileUpload::run(&ctx, request("https://example.com/", Some("DELETE")))
            .err()
            .expect("refused");
        assert_eq!(err.code(), "invalid_arguments");
    }
}
