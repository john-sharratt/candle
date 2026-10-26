//! `image_upload {}`: a link that takes one of the user's pictures into Zen,
//! for `image_edit`.
//!
//! A conversation cannot carry a picture to a tool: the only way is the model
//! writing the file out as base64 in the call, and a phone photo is millions of
//! characters. Its sandbox can, though — it holds the user's upload as a file
//! — so this answers a link signed for ten minutes ([`crate::image_links`])
//! that the sandbox sends the file to, `PUT /v1/image/upload`. Zen keeps the
//! picture for an hour, upright and as a JPEG, and the upload's answer names
//! it; `image_edit` takes that name as often as wanted.

use std::sync::Arc;

use axum::http::HeaderMap;
use mcp_server::origin;
use serde_json::{json, Value};

use super::{Answer, Tool};
use crate::api::{now_ms, Authored};

pub struct Upload;

impl Tool for Upload {
    const NAME: &'static str = "image_upload";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Get a link to send one of the user's own pictures to Zen, so \
                image_edit can change it — a photo or image the user gave you. Answers a URL and \
                the command that sends the file from your sandbox; the answer to that names the \
                picture, which you pass to image_edit as `image`. Takes JPEG, PNG, WebP, GIF, BMP \
                or TIFF up to 32 MB; Zen turns it upright and keeps it for an hour. Read the \
                guide `image` first.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, _args: &Value) -> Answer {
        let url = format!(
            "{}{}",
            origin::of(headers),
            s.image_links.upload_path(now_ms() / 1000)
        );
        Answer::ok(instructions(&url))
    }
}

/// What the tool answers: the link, how to send the file to it, and what
/// comes back.
fn instructions(url: &str) -> String {
    format!(
        "Send the picture, as a file from your sandbox, within 10 minutes:\n\n\
         curl -sS -X PUT --data-binary @<file> '{url}'\n\n\
         It answers {{\"image\": \"<id>\", \"width\": …, \"height\": …, \"link\": …}}: pass the \
         id (or the link) to image_edit as `image`. Zen keeps the picture for an hour. Keep the URL whole — its \
         `exp` and `sig` are the permission. If your sandbox cannot reach the URL, tell the user \
         the picture cannot be sent from here."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_says_how_to_send_the_file_and_what_comes_back() {
        let a = instructions("https://bot.tokera.com/v1/image/upload?exp=1&sig=ab");
        assert!(
            a.contains("curl -sS -X PUT --data-binary @<file> 'https://bot.tokera.com/v1/image/upload?exp=1&sig=ab'"),
            "{a}"
        );
        assert!(a.contains("{\"image\": \"<id>\""), "{a}");
        assert!(a.contains("image_edit as `image`"), "{a}");
    }
}
