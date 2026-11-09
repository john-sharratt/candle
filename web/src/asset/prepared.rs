//! A file made ready for the wire once: its type, its tags, and its bytes
//! plain and — when compressing it is worth doing — gzipped.
//!
//! Hashing and gzipping a file per request cost the 327 KB console about 6 ms
//! a load. Prepared once per version of the file and kept
//! ([`crate::content::Roots::prepared`]), a request picks a representation
//! and answers.

use axum::body::Bytes;
use sha2::{Digest, Sha256};

use super::{compressible, gz, MIN_COMPRESS};

/// One file, ready to answer with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    pub mime: String,
    /// The plain body's entity tag, `"<sha16>"`.
    pub tag: String,
    /// The gzipped body's, `"<sha16>-gz"`, when there is one.
    pub tag_gz: Option<String>,
    pub plain: Bytes,
    pub gz: Option<Bytes>,
}

impl Prepared {
    /// `bytes` as a file of type `mime`.
    pub fn typed(mime: &str, bytes: Bytes) -> Prepared {
        let hex = hex_of(&bytes);
        Prepared::tagged(mime, bytes, &hex)
    }

    /// `bytes` as the file `name`, its type guessed from the name.
    pub fn of(name: &str, bytes: Bytes) -> Prepared {
        Prepared::typed(&mime_of(name), bytes)
    }

    /// `bytes` of type `mime`, already hashed to `hex` — a file whose tag was
    /// kept and whose bytes were read again to answer.
    pub fn tagged(mime: &str, bytes: Bytes, hex: &str) -> Prepared {
        let gzip = compressible(mime) && bytes.len() >= MIN_COMPRESS;
        let (gz, tag_gz) = if gzip {
            (Some(Bytes::from(gz(&bytes))), Some(tag_of(hex, true)))
        } else {
            (None, None)
        };
        Prepared {
            mime: mime.to_string(),
            tag: tag_of(hex, false),
            tag_gz,
            plain: bytes,
            gz,
        }
    }
}

/// The type of the file `name`.
pub fn mime_of(name: &str) -> String {
    mime_guess::from_path(name)
        .first_or_octet_stream()
        .as_ref()
        .to_string()
}

/// Sixteen hex characters of SHA-256 — sixty-four bits, which is not a
/// security boundary and does not need to be. The consequence of a collision
/// is one stale file for one client until its next change, and the
/// alternative is a longer header on every response for ever.
pub fn hex_of(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A strong entity tag for the content hashed to `hex`. The tag identifies the
/// *representation*, so the compressed body carries a different one: a cache
/// holding both must be able to tell them apart, and `Vary` alone does not do
/// that for a client that revalidates.
pub fn tag_of(hex: &str, gzip: bool) -> String {
    if gzip {
        format!("\"{hex}-gz\"")
    } else {
        format!("\"{hex}\"")
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::GzDecoder;

    use super::*;

    fn gunzip(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        GzDecoder::new(bytes).read_to_end(&mut out).unwrap();
        out
    }

    /// **A file worth compressing is prepared both ways**: the plain body as
    /// given, a gzip stream (its fixed header bytes, and inflating to the
    /// plain body), and both tags as literals.
    #[test]
    fn a_stylesheet_is_prepared_plain_and_gzipped() {
        let css = Bytes::from(".a { color: red; }\n".repeat(100));
        let p = Prepared::of("app.css", css.clone());
        assert_eq!(p.mime, "text/css");
        assert_eq!(p.plain, css);
        let hex = hex_of(&css);
        assert_eq!(hex.len(), 16);
        assert_eq!(p.tag, format!("\"{hex}\""));
        assert_eq!(p.tag_gz, Some(format!("\"{hex}-gz\"")));
        let gz = p.gz.expect("compressed");
        assert_eq!(&gz[..4], [0x1f, 0x8b, 0x08, 0x00], "the gzip header");
        assert_eq!(gunzip(&gz), css);
    }

    /// The tag is the content's: `a` hashes to its published SHA-256 prefix.
    #[test]
    fn the_tag_is_the_sha256_prefix() {
        let p = Prepared::typed("text/plain", Bytes::from_static(b"a"));
        assert_eq!(p.tag, "\"ca978112ca1bbdca\"");
        assert_eq!(p.tag_gz, None, "too small to gain from gzip");
        assert_eq!(p.gz, None);
    }

    /// A picture is never gzipped, however large.
    #[test]
    fn a_picture_is_prepared_plain_only() {
        let p = Prepared::of("mark.png", Bytes::from(vec![7u8; 4000]));
        assert_eq!(p.mime, "image/png");
        assert_eq!((p.gz, p.tag_gz), (None, None));
    }
}
