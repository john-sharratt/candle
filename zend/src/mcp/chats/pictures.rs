//! The pictures a chat message shows Zen: each one the user's own, sent in
//! with `image_upload` and named here by the id (or link) it answered with.
//!
//! A conversation cannot carry a picture to a tool, so the picture is fetched
//! up the cluster as the caller (`GET /v1/image/{id}`), as their browser would
//! fetch it. A chat's turn runs here and reads it kept
//! ([`ZendSession::keep_image`]); a conversation's turn runs where the gateway
//! places it and takes its bytes along.

use std::sync::Arc;
use std::time::Duration;

use axum::http::Method;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use candle_conversation::vision::prepare::mime_of;

use crate::mcp::caller::Caller;
use crate::mcp::ops;
use crate::mcp::workers::loading;
use crate::session::ZendSession;

/// The most pictures one message shows.
pub const MAX_PICTURES: usize = 4;

/// How long a picture's fetch may take.
const FETCH_WITHIN: Duration = Duration::from_secs(30);

/// The picture id `named` gives: an id as `image_upload` answered it, or the
/// id at the end of its link.
pub fn picture_id(named: &str) -> Result<String, String> {
    let named = named.trim();
    let id = match named.rsplit_once("/v1/image/") {
        Some((_, rest)) => {
            let path = rest.split('?').next().unwrap_or(rest);
            path.rsplit('/').next().unwrap_or(path)
        }
        None => named,
    };
    let fits = !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if fits {
        Ok(id.to_string())
    } else {
        Err(format!(
            "`{}` is not a picture `image_upload` answered: pass its id or its link",
            named.chars().take(80).collect::<String>()
        ))
    }
}

/// `named`'s pictures, fetched as `caller`: each one's bytes, in order.
pub async fn fetch(
    session: &Arc<ZendSession>,
    caller: &Caller,
    named: &[String],
) -> Result<Vec<Vec<u8>>, String> {
    if named.len() > MAX_PICTURES {
        return Err(format!(
            "{} pictures: a message shows at most {MAX_PICTURES}",
            named.len()
        ));
    }
    let ids = named
        .iter()
        .map(|n| picture_id(n))
        .collect::<Result<Vec<_>, _>>()?;
    let mut fetched = Vec::with_capacity(ids.len());
    for id in ids {
        let path = format!("/v1/image/{id}");
        let reply = ops::up(
            session,
            caller,
            Method::GET,
            &path,
            None,
            None,
            FETCH_WITHIN,
        )
        .await?;
        match reply.status {
            200 if reply.complete => fetched.push(reply.body.to_vec()),
            200 => return Err(format!("picture `{id}` did not arrive whole")),
            404 => {
                return Err(format!(
                    "no picture `{id}`: Zen keeps a picture an hour after `image_upload` — \
                     send it again"
                ))
            }
            status => return Err(format!("picture `{id}` could not be fetched ({status})")),
        }
    }
    Ok(fetched)
}

/// `pictures` kept as images on this member, each read by the tower here:
/// their kept ids, in order. For a turn that runs on this member — one the
/// gateway places may land where the kept image has not yet replicated, and
/// takes the bytes instead ([`data_url`]).
pub async fn keep(
    session: &Arc<ZendSession>,
    pictures: Vec<Vec<u8>>,
) -> Result<Vec<String>, String> {
    let mut kept = Vec::with_capacity(pictures.len());
    for (i, bytes) in pictures.into_iter().enumerate() {
        let keeper = Arc::clone(session);
        let image = tokio::task::spawn_blocking(move || keeper.keep_image(bytes))
            .await
            .map_err(|e| format!("reading picture {} panicked: {e}", i + 1))?
            .ok_or_else(loading)?
            .map_err(|e| format!("picture {}: {e}", i + 1))?;
        kept.push(image.id);
    }
    Ok(kept)
}

/// `bytes` as a `data:` URL, typed by their format.
pub fn data_url(bytes: &[u8]) -> String {
    format!("data:{};base64,{}", mime_of(bytes), STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_is_named_by_its_id_or_its_link() {
        assert_eq!(picture_id("a1b2c3"), Ok("a1b2c3".into()));
        assert_eq!(picture_id(" a1b2c3 "), Ok("a1b2c3".into()));
        assert_eq!(
            picture_id("https://bot.tokera.com/v1/image/link/a1b2c3?exp=1&sig=ab"),
            Ok("a1b2c3".into())
        );
        assert_eq!(
            picture_id("https://bot.tokera.com/v1/image/a1b2c3"),
            Ok("a1b2c3".into())
        );
    }

    #[test]
    fn a_picture_travels_as_a_typed_data_url() {
        let png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR";
        assert_eq!(
            data_url(png),
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        );
    }

    #[test]
    fn anything_else_is_refused() {
        assert!(picture_id("").is_err());
        assert!(picture_id("../etc/passwd").is_err());
        assert!(picture_id("https://example.com/cat.png").is_err());
    }
}
