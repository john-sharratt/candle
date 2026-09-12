//! Images a person uploaded, on disk.
//!
//! A portrait is a file. It needs no engine, no model and no GPU — which is why
//! it is here rather than behind [`crate::engine`], and why the console's
//! upload silently dropping the file was a gap rather than a limitation.
//!
//! # What it stored, and what it did with it
//!
//! The create page read the chosen file into an object URL, set
//! `draft.portrait`, called it uploaded, and then never sent it: `create()`
//! posts a name, a world, a personality and a description. The image existed
//! only as a blob in the browser and went away with the tab. The record has had
//! `portrait_image_id` and `portrait_origin` fields the whole time, and nothing
//! ever filled them.
//!
//! # Content-addressed
//!
//! The id is a hash of the bytes, so uploading the same portrait twice stores
//! it once and produces the same id — which also means an id cannot be guessed
//! from a counter, and cannot collide by accident.
//!
//! # What is accepted
//!
//! PNG, JPEG, WebP and GIF, recognised by their **magic bytes** rather than by
//! a `Content-Type` header or a file extension. Both of those are the client's
//! claim about the file; the first bytes are the file. Anything else is refused
//! before it reaches the disk, so this directory holds images and nothing else —
//! it is served back to browsers, and a stored `.html` served from this origin
//! would run in it.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Four megabytes. A generous portrait and nowhere near a way to fill a disk
/// one upload at a time.
pub const MAX_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub enum ImageError {
    /// Not one of the formats this accepts, by its own first bytes.
    NotAnImage,
    TooLarge(usize),
    NotFound,
    Io(std::io::Error),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAnImage => write!(
                f,
                "that is not a PNG, JPEG, WebP or GIF — judged by the file itself, not its name"
            ),
            Self::TooLarge(n) => write!(f, "{n} bytes exceeds the {MAX_BYTES} byte limit"),
            Self::NotFound => write!(f, "no such image"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

/// The extension these bytes should be stored under, from the bytes.
///
/// A `Content-Type` header and a file name are both things the client says. The
/// magic number is the file, and this directory is served back over HTTP, so
/// what goes into it has to be decided here.
fn sniff(bytes: &[u8]) -> Option<&'static str> {
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";
    const GIF87: &[u8] = b"GIF87a";
    const GIF89: &[u8] = b"GIF89a";

    if bytes.starts_with(PNG) {
        return Some("png");
    }
    // JPEG: SOI marker. The third byte varies by encoder, so only two are fixed.
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("jpg");
    }
    if bytes.starts_with(GIF87) || bytes.starts_with(GIF89) {
        return Some("gif");
    }
    // WebP is a RIFF container whose form type is `WEBP` at offset 8 — the
    // `RIFF` alone would also match a WAV.
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("webp");
    }
    None
}

/// What an extension is served as. **The only** extension-to-type mapping here:
/// `sniff` decides what a file is and this decides how to send it back, and if
/// the two were separate lists they would be free to disagree about a format
/// one of them had learned.
fn mime_for(ext: &str) -> Option<&'static str> {
    match ext {
        "png" => Some("image/png"),
        "jpg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

/// Where images live, under the daemon's data directory.
#[derive(Debug, Clone)]
pub struct Images {
    dir: PathBuf,
}

impl Images {
    pub fn new(data: &Path) -> Self {
        Self {
            dir: data.join("images"),
        }
    }

    /// Store one, and give back the id it is addressed by.
    ///
    /// The id is `img_<16 hex of the content hash>.<ext>` — content-addressed,
    /// so the same portrait uploaded twice is stored once, and idempotent, so a
    /// retried upload is not a second file.
    pub fn put(&self, bytes: &[u8]) -> Result<String, ImageError> {
        if bytes.len() > MAX_BYTES {
            return Err(ImageError::TooLarge(bytes.len()));
        }
        let ext = sniff(bytes).ok_or(ImageError::NotAnImage)?;

        let digest = Sha256::digest(bytes);
        let hex: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
        let id = format!("img_{hex}.{ext}");

        std::fs::create_dir_all(&self.dir).map_err(ImageError::Io)?;
        let path = self.dir.join(&id);
        // Already there: the same bytes hash the same, so there is nothing to
        // write and nothing to check.
        if path.exists() {
            return Ok(id);
        }
        // Beside the target, then renamed — the same atomicity the mind's
        // documents get, so an interrupted upload leaves no half-written image
        // for the next request to serve.
        let tmp = self.dir.join(format!(".{id}.part"));
        std::fs::write(&tmp, bytes).map_err(ImageError::Io)?;
        std::fs::rename(&tmp, &path).map_err(ImageError::Io)?;
        Ok(id)
    }

    /// Read one back, with the type to serve it as.
    ///
    /// The id is checked against the shape [`Self::put`] produces rather than
    /// joined onto the directory as given: it arrives in a URL, and a path is
    /// the one thing a URL must never be able to become.
    pub fn get(&self, id: &str) -> Result<(Vec<u8>, &'static str), ImageError> {
        let format = valid_id(id).ok_or(ImageError::NotFound)?;
        let bytes = std::fs::read(self.dir.join(id)).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => ImageError::NotFound,
            _ => ImageError::Io(e),
        })?;
        Ok((bytes, format))
    }
}

/// Whether this is an id this module minted, and what it serves as.
///
/// `img_` then exactly sixteen lowercase hex characters then a known extension.
/// Nothing else — no separators, no dots, no traversal, and nothing that could
/// be a name outside the directory even before the join.
fn valid_id(id: &str) -> Option<&'static str> {
    let rest = id.strip_prefix("img_")?;
    let (hex, ext) = rest.split_once('.')?;
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return None;
    }
    mime_for(ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("npcd-img-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn png() -> Vec<u8> {
        let mut v = b"\x89PNG\r\n\x1a\n".to_vec();
        v.extend_from_slice(&[0u8; 64]);
        v
    }

    #[test]
    fn an_image_round_trips_and_keeps_its_type() {
        let images = Images::new(&tmp("round"));
        let id = images.put(&png()).expect("stored");
        assert!(id.starts_with("img_") && id.ends_with(".png"), "{id}");
        let (back, mime) = images.get(&id).expect("read back");
        assert_eq!(back, png());
        assert_eq!(mime, "image/png");
    }

    /// Content-addressed: the same portrait twice is one file and one id.
    #[test]
    fn the_same_bytes_store_once() {
        let images = Images::new(&tmp("dedup"));
        let a = images.put(&png()).unwrap();
        let b = images.put(&png()).unwrap();
        assert_eq!(a, b);
    }

    /// **The format is judged by the file, not by what the client called it.**
    ///
    /// These are served back over HTTP from this origin, so a stored document
    /// that a browser would execute is the thing to keep out — and a
    /// `Content-Type` header is the uploader's claim, not a fact.
    #[test]
    fn anything_that_is_not_an_image_is_refused() {
        let images = Images::new(&tmp("sniff"));
        for bad in [
            &b"<html><script>alert(1)</script></html>"[..],
            &b"GIF87"[..],        // truncated magic
            &b"RIFF____WAVE"[..], // a RIFF that is not a WebP
            &b""[..],
        ] {
            assert!(
                matches!(images.put(bad), Err(ImageError::NotAnImage)),
                "accepted {bad:?}"
            );
        }
        // And the ones that are.
        assert!(images.put(&png()).is_ok());
        assert!(images.put(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0]).is_ok());
        let mut webp = b"RIFF\0\0\0\0WEBP".to_vec();
        webp.extend_from_slice(&[0u8; 8]);
        assert!(images.put(&webp).is_ok());
    }

    #[test]
    fn an_oversized_upload_is_refused_before_it_is_written() {
        let images = Images::new(&tmp("big"));
        let mut huge = png();
        huge.resize(MAX_BYTES + 1, 0);
        assert!(matches!(images.put(&huge), Err(ImageError::TooLarge(_))));
    }

    /// **An id arrives in a URL, so it must not be able to become a path.**
    #[test]
    fn a_crafted_id_cannot_reach_outside_the_directory() {
        let dir = tmp("escape");
        std::fs::write(dir.join("secret.txt"), b"not yours").unwrap();
        let images = Images::new(&dir);
        for bad in [
            "../secret.txt",
            "img_../../secret.txt",
            "img_0011223344556677.png/../../secret.txt",
            "img_00112233445566.png",    // too short
            "img_00112233445566778.png", // too long
            "img_00112233445566GG.png",  // not hex
            "img_0011223344556677.html",
            "img_0011223344556677.PNG", // the case this mints is lower
            "secret.txt",
            "",
        ] {
            assert!(
                matches!(images.get(bad), Err(ImageError::NotFound)),
                "`{bad}` was not refused"
            );
        }
    }

    /// **Everything `sniff` accepts, `mime_for` can serve.**
    ///
    /// The two are the only place formats are named, and a format learned by
    /// one and not the other is an image this daemon will store and then refuse
    /// to hand back — a `404` on a file that is sitting right there.
    #[test]
    fn every_format_accepted_can_also_be_served() {
        let mut samples: Vec<Vec<u8>> = vec![png(), vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 0]];
        samples.push(b"GIF89a".to_vec());
        let mut webp = b"RIFF\0\0\0\0WEBP".to_vec();
        webp.extend_from_slice(&[0u8; 8]);
        samples.push(webp);

        for bytes in &samples {
            let ext = sniff(bytes).expect("a sample this test says is an image");
            assert!(
                mime_for(ext).is_some(),
                "`{ext}` is stored but cannot be served"
            );
        }
        // And the round trip proves it end to end, not just in the tables.
        let images = Images::new(&tmp("formats"));
        for bytes in &samples {
            let id = images.put(bytes).expect("stored");
            assert!(images.get(&id).is_ok(), "{id} could not be read back");
        }
    }

    #[test]
    fn an_image_that_is_not_there_says_so() {
        let images = Images::new(&tmp("missing"));
        assert!(matches!(
            images.get("img_0011223344556677.png"),
            Err(ImageError::NotFound)
        ));
    }
}
