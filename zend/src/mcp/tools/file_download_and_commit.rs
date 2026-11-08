//! `file_download_and_commit {repo, message, files: [{url, path}], branch?}`:
//! what each public URL serves — a picture an image tool drew, a font, a
//! library, a raw file from a public repository — landed as one commit
//! (`docs/mcp_file_tools.md` §3.6), its bytes never passing through the
//! caller. Each URL is fetched through the public-only guard (§3.7); one that
//! fails, and nothing lands.

use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};
use zend_tools::grants::{Capability, Grants};
use zend_tools::public_fetch::{public_client, read_within, FetchError};
use zend_tools::tools::file::download::MAX_DOWNLOAD_BYTES;

use super::landing::{land, refused, target};
use super::{Answer, Ctx};
use crate::file_commit::change::Change;

pub const NAME: &str = "file_download_and_commit";

/// The most bytes one call lands.
const MAX_CALL_BYTES: usize = 16 * 1024 * 1024;

/// How long each fetch may take.
const WITHIN: Duration = Duration::from_secs(30);

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Commit what public links serve — a picture `image_generate` or `image_edit` \
            answered with, a font or library from a CDN, a raw file from a public repository — to \
            one of the user's repositories, without the bytes passing through you, all at once or \
            not at all. Each link is fetched by Zen from the public internet only: private and \
            local addresses, and redirects to them, are refused; a link behind a sign-in (a \
            claude.ai artifact) cannot be fetched — send such bytes with `file_upload_and_commit`. \
            At most 8 MiB a file, 16 MiB a call, 30 seconds a link. The answer says `live` with the \
            commit, or `refused` with why.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "The repository, by name: `yenbot`, `candle`, …"},
                "message": {"type": "string", "description": "The commit message: what changed and why, in a line."},
                "branch": {"type": "string", "description": "The branch to commit on; the repository's main branch by default."},
                "files": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "An `http://` or `https://` link on the public internet."},
                            "path": {"type": "string", "description": "Where it goes, relative to the repository's root."}
                        },
                        "required": ["url", "path"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["repo", "message", "files"],
            "additionalProperties": false
        }
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    url: String,
    path: String,
}

/// What `url` serves, at most `cap` bytes.
fn fetch(url: &str, cap: usize) -> Result<Vec<u8>, FetchError> {
    let pinned = public_client(Grants::NONE.with(Capability::Network), url, WITHIN)?;
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
        });
    }
    read_within(resp, cap)
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (repo, _) = match target(ctx, args) {
        Ok(t) => t,
        Err(a) => return a,
    };
    let files: Vec<File> = match args.get("files").cloned().map(serde_json::from_value) {
        Some(Ok(files)) => files,
        Some(Err(e)) => return Answer::error(format!("`files` is not a list of files: {e}")),
        None => return Answer::error("`files` is missing"),
    };
    let mut changes = Vec::with_capacity(files.len());
    let mut total = 0usize;
    for f in &files {
        let left = MAX_CALL_BYTES.saturating_sub(total);
        match fetch(&f.url, MAX_DOWNLOAD_BYTES.min(left)) {
            Ok(bytes) => {
                total += bytes.len();
                changes.push(Change::Bytes {
                    path: f.path.clone(),
                    bytes,
                });
            }
            Err(e) => return refused(&repo, &format!("{}: {e}", f.url)),
        }
    }
    land(ctx, args, &changes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A local link is refused before anything is fetched**, so nothing is
    /// committed.
    #[test]
    fn a_local_link_is_refused() {
        for url in [
            "http://127.0.0.1:8092/restart/motherd",
            "http://localhost/x.png",
        ] {
            assert!(
                matches!(fetch(url, 1024), Err(FetchError::UrlBlocked(_))),
                "{url} was fetched"
            );
        }
    }
}
