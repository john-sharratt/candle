//! Images a person uploaded, as blobs in the substrate.
//!
//! A portrait needs no engine, no model and no GPU — which is why it is here
//! rather than behind [`crate::engine`], and why the console's upload silently
//! dropping the file was a gap rather than a limitation. Each is a `Blob`
//! record named by its SHA-256 (`docs/replicated_substrate.md` §10.3), so
//! every member of the cluster serves the same portrait for a character
//! whichever member it was uploaded to.
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
//! The id is the bytes' SHA-256 and their format, `<64 hex>.<ext>`, so
//! uploading the same portrait twice stores it once and produces the same id —
//! which also means an id cannot be guessed from a counter, and cannot collide
//! by accident.
//!
//! # What is accepted
//!
//! PNG, JPEG, WebP and GIF, recognised by their **magic bytes** rather than by
//! a `Content-Type` header or a file extension. Both of those are the client's
//! claim about the file; the first bytes are the file. Anything else is refused
//! before it is stored, so what this serves back to browsers is images and
//! nothing else — a stored `.html` served from this origin would run in it.

mod drawn;

pub use drawn::KEPT as DRAWN_KEPT;

use std::collections::HashMap;
use std::io;
use std::path::Path;

use candle_conversation::persistence::blob::{hex, parse_hex};
use candle_conversation::persistence::{PersistenceError, SharedSubstrate};

/// Four megabytes. A generous portrait and nowhere near a way to fill a disk
/// one upload at a time.
pub const MAX_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub enum ImageError {
    /// Not one of the formats this accepts, by its own first bytes.
    NotAnImage,
    TooLarge(usize),
    NotFound,
    Io(io::Error),
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

/// The substrate failed to store or read one.
impl From<PersistenceError> for ImageError {
    fn from(e: PersistenceError) -> Self {
        Self::Io(io::Error::other(format!("{e:?}")))
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

/// The images the substrate holds.
#[derive(Clone)]
pub struct Images {
    shared: SharedSubstrate,
}

impl Images {
    /// The images `shared` holds — taking in, once, the files a daemon kept
    /// in `legacy` before images were blobs, and removing the folder after.
    /// Returns each taken-in file's old id with its new one, for the records
    /// that name it.
    pub fn open(
        shared: SharedSubstrate,
        legacy: &Path,
    ) -> Result<(Images, HashMap<String, String>), ImageError> {
        let images = Images { shared };
        let mut renames = HashMap::new();
        if legacy.is_dir() {
            for entry in std::fs::read_dir(legacy).map_err(ImageError::Io)?.flatten() {
                let old = entry.file_name().to_string_lossy().into_owned();
                if !old.starts_with("img_") {
                    continue;
                }
                let bytes = std::fs::read(entry.path()).map_err(ImageError::Io)?;
                match images.put(&bytes) {
                    Ok(new) => {
                        renames.insert(old, new);
                    }
                    Err(e) => tracing::warn!("image {old} was not taken in: {e}"),
                }
            }
            std::fs::remove_dir_all(legacy).map_err(ImageError::Io)?;
            tracing::info!(
                "images: {} taken in from {}",
                renames.len(),
                legacy.display()
            );
        }
        Ok((images, renames))
    }

    /// Store one, and give back the id it is addressed by.
    ///
    /// The id is `<64 hex of the content hash>.<ext>` — content-addressed, so
    /// the same portrait uploaded twice is stored once, and idempotent, so a
    /// retried upload is not a second blob. A picture drawn for a conversation
    /// that is stored this way — a portrait made of it — is kept from then on
    /// rather than retired after its day (`drawn`).
    pub fn put(&self, bytes: &[u8]) -> Result<String, ImageError> {
        let id = self.store(bytes)?;
        self.keep_drawn(&id)?;
        Ok(id)
    }

    /// Store `bytes` as an image and give back its id.
    fn store(&self, bytes: &[u8]) -> Result<String, ImageError> {
        if bytes.len() > MAX_BYTES {
            return Err(ImageError::TooLarge(bytes.len()));
        }
        let ext = sniff(bytes).ok_or(ImageError::NotAnImage)?;
        let sha256 = self.shared.put_blob(bytes.to_vec())?;
        Ok(format!("{}.{ext}", hex(&sha256)))
    }

    /// Read one back, with the type to serve it as.
    ///
    /// The id is checked against the shape [`Self::put`] produces: it arrives
    /// in a URL, and only a hash and a known format are ever looked up.
    pub fn get(&self, id: &str) -> Result<(Vec<u8>, &'static str), ImageError> {
        let (sha256, format) = valid_id(id).ok_or(ImageError::NotFound)?;
        let bytes = self.shared.blob(&sha256)?.ok_or(ImageError::NotFound)?;
        Ok((bytes, format))
    }
}

/// Whether this is an id this module minted — the hash it names, and what it
/// serves as.
///
/// Exactly sixty-four lowercase hex characters, a dot, and a known extension.
/// Nothing else.
fn valid_id(id: &str) -> Option<([u8; 32], &'static str)> {
    let (hash, ext) = id.split_once('.')?;
    if hash.bytes().any(|b| b.is_ascii_uppercase()) {
        return None;
    }
    Some((parse_hex(hash)?, mime_for(ext)?))
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    /// The images of a fresh substrate in a directory of the test's own.
    fn store() -> (TempDir, Images) {
        let dir = TempDir::new().unwrap();
        let (images, _) = open(dir.path());
        (dir, images)
    }

    fn open(dir: &Path) -> (Images, HashMap<String, String>) {
        let shared = Npcs::load(dir).unwrap().substrate();
        Images::open(shared, &dir.join("images")).unwrap()
    }

    fn png() -> Vec<u8> {
        let mut v = b"\x89PNG\r\n\x1a\n".to_vec();
        v.extend_from_slice(&[0u8; 64]);
        v
    }

    #[test]
    fn an_image_round_trips_and_keeps_its_type() {
        let (_dir, images) = store();
        let id = images.put(&png()).expect("stored");
        let (hash, ext) = id.split_once('.').unwrap();
        assert_eq!((hash.len(), ext), (64, "png"), "{id}");
        let (back, mime) = images.get(&id).expect("read back");
        assert_eq!(back, png());
        assert_eq!(mime, "image/png");
    }

    /// Content-addressed: the same portrait twice is one blob and one id, and
    /// the id is the bytes' SHA-256.
    #[test]
    fn the_same_bytes_store_once_under_their_hash() {
        let (_dir, images) = store();
        let a = images.put(&png()).unwrap();
        let b = images.put(&png()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, format!("{}.png", hex(&Sha256::digest(png()).into())));
    }

    /// **An image outlives a restart**, and one kept as a file before images
    /// were blobs is taken in once — its old id renamed for the records that
    /// name it, and the folder gone.
    #[test]
    fn an_old_file_is_taken_in_and_every_image_outlives_a_restart() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("images")).unwrap();
        std::fs::write(dir.path().join("images/img_0011223344556677.png"), png()).unwrap();
        let (images, renames) = open(dir.path());
        let new = renames["img_0011223344556677.png"].clone();
        assert_eq!(images.get(&new).unwrap().0, png());
        assert!(!dir.path().join("images").exists());
        drop(images);

        let (again, renames) = open(dir.path());
        assert!(renames.is_empty(), "taken in once");
        assert_eq!(again.get(&new).unwrap().0, png());
    }

    /// **The format is judged by the file, not by what the client called it.**
    ///
    /// These are served back over HTTP from this origin, so a stored document
    /// that a browser would execute is the thing to keep out — and a
    /// `Content-Type` header is the uploader's claim, not a fact.
    #[test]
    fn anything_that_is_not_an_image_is_refused() {
        let (_dir, images) = store();
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
        let (_dir, images) = store();
        let mut huge = png();
        huge.resize(MAX_BYTES + 1, 0);
        assert!(matches!(images.put(&huge), Err(ImageError::TooLarge(_))));
    }

    /// **An id arrives in a URL, so only a hash and a known format are ever
    /// looked up.**
    #[test]
    fn a_crafted_id_is_not_found() {
        let (_dir, images) = store();
        let hash = "00".repeat(32);
        for bad in [
            "../secret.txt".to_string(),
            format!("{hash}.png/../../secret.txt"),
            format!("{}.png", &hash[..62]), // too short
            format!("{hash}0.png"),         // too long
            format!("{}GG.png", &hash[..62]),
            format!("{hash}.html"),
            format!("{hash}.PNG"),
            format!("{}.png", hash.to_uppercase()),
            "img_0011223344556677.png".to_string(),
            "secret.txt".to_string(),
            String::new(),
        ] {
            let bad = bad.as_str();
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
        let (_dir, images) = store();
        for bytes in &samples {
            let id = images.put(bytes).expect("stored");
            assert!(images.get(&id).is_ok(), "{id} could not be read back");
        }
    }

    #[test]
    fn an_image_that_is_not_there_says_so() {
        let (_dir, images) = store();
        assert!(matches!(
            images.get(&format!("{}.png", "ab".repeat(32))),
            Err(ImageError::NotFound)
        ));
    }
}
