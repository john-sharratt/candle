//! A message's images: how a request names one, and the bytes it names.
//!
//! An image part's `url` is a `data:` URL carrying the file, or the id
//! `POST /v1/images` answered with for one already kept. Nothing on the
//! request path fetches from the network; a client sends the bytes.
//!
//! Design: `docs/flash_next_vision.md` §6.1.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::vision::prepare::ID_HEX;
use candle_conversation::vision::{load_original, turn_images, TurnImage};
use candle_conversation::ConversationEngine;

/// An image a request names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageRef {
    /// The file itself, from a `data:` URL.
    Bytes(Vec<u8>),
    /// An image the substrate keeps, by its id.
    Stored(String),
}

/// What `url` names, or why it names nothing this daemon reads.
pub fn parse(url: &str) -> Result<ImageRef, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (head, data) = rest
            .split_once(',')
            .ok_or_else(|| "a data: URL with no `,` before its data".to_string())?;
        if !head.ends_with(";base64") {
            return Err(format!("a data: URL that is not base64 (`{head}`)"));
        }
        return STANDARD
            .decode(data.trim())
            .map(ImageRef::Bytes)
            .map_err(|e| format!("a data: URL whose base64 does not decode: {e}"));
    }
    let is_id = url.len() == ID_HEX && url.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if is_id {
        return Ok(ImageRef::Stored(url.to_string()));
    }
    Err(format!(
        "an image url is a data: URL or an image id from POST /v1/images, not `{}`",
        url.chars().take(40).collect::<String>()
    ))
}

/// Every url of `urls`, parsed; the first that names nothing refuses them all.
pub fn parse_all(urls: &[String]) -> Result<Vec<ImageRef>, String> {
    urls.iter().map(|u| parse(u)).collect()
}

/// `refs` as a turn's images: a stored one's original read back, each read by
/// the tower unless an earlier turn already did. Blocking: the tower runs as a
/// guest between waves, and this waits for it.
pub fn resolve(
    engine: &ConversationEngine,
    substrate: &SharedSubstrate,
    refs: Vec<ImageRef>,
) -> Result<Vec<TurnImage>, String> {
    let uploads = refs
        .into_iter()
        .map(|r| match r {
            ImageRef::Bytes(b) => Ok(b),
            ImageRef::Stored(id) => {
                load_original(substrate, &id)?.ok_or_else(|| format!("no image {id} is kept"))
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    turn_images(engine, substrate, uploads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_data_url_carries_its_bytes() {
        assert_eq!(
            parse("data:image/png;base64,AAEC/w=="),
            Ok(ImageRef::Bytes(vec![0, 1, 2, 255]))
        );
    }

    #[test]
    fn an_id_names_a_kept_image() {
        assert_eq!(
            parse("0123456789abcdef"),
            Ok(ImageRef::Stored("0123456789abcdef".into()))
        );
    }

    /// A url that names nothing readable is refused, never fetched.
    #[test]
    fn anything_else_is_refused() {
        assert!(parse("https://example.com/cat.png").is_err());
        assert!(parse("data:image/png,rawbytes").is_err());
        assert!(parse("data:image/png;base64,***").is_err());
        assert!(parse("0123456789ABCDEF").is_err());
        assert!(parse("0123456789abcde").is_err());
        let both = ["0123456789abcdef".to_string(), "nope".to_string()];
        assert!(parse_all(&both).is_err());
    }
}
