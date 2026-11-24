//! An uploaded image, made ready for the tower: decoded, sized to its patch
//! grid by `smart_resize`, and named by the digest of the bytes it arrived as.
//!
//! Host work, once per image, on the caller's thread — never inside a drain.

use candle_transformers::models::qwen_vision::smart_resize;
use image::imageops::FilterType;
use sha2::{Digest, Sha256};

use crate::guest::VisionImage;

/// Hex digits of an image's digest that name it in a turn's text and in the
/// substrate: 64 bits, so two images of one conversation never share one.
pub const ID_HEX: usize = 16;

/// An image ready for the tower.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedImage {
    /// The first [`ID_HEX`] hex digits of the original bytes' SHA-256.
    pub id: String,
    /// The original bytes' full SHA-256, hex.
    pub sha256: String,
    /// The size it arrived at.
    pub width: u32,
    pub height: u32,
    /// Resized to its grid.
    pub image: VisionImage,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The id the image whose file is `bytes` is kept under — what [`prepare`]
/// names it — without decoding it.
pub fn image_id(bytes: &[u8]) -> String {
    sha256_hex(bytes)[..ID_HEX].to_string()
}

/// Decode `bytes` (JPEG, PNG, WebP, GIF's first frame, BMP, TIFF) and size it
/// for the tower.
pub fn prepare(bytes: &[u8]) -> Result<PreparedImage, String> {
    let decoded = image::load_from_memory(bytes).map_err(|e| format!("not an image: {e}"))?;
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();
    let grid = smart_resize(height as usize, width as usize)?;
    let (gw, gh) = (grid.width as u32, grid.height as u32);
    let sized = if (gw, gh) == (width, height) {
        rgb
    } else {
        image::imageops::resize(&rgb, gw, gh, FilterType::CatmullRom)
    };
    let sha256 = sha256_hex(bytes);
    Ok(PreparedImage {
        id: sha256[..ID_HEX].to_string(),
        sha256,
        width,
        height,
        image: VisionImage {
            rgb: sized.into_raw(),
            height: gh,
            width: gw,
        },
    })
}

/// The media type of an image's bytes, by their format's signature.
pub fn mime_of(bytes: &[u8]) -> &'static str {
    image::guess_format(bytes)
        .map(|f| f.to_mime_type())
        .unwrap_or("application/octet-stream")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, ImageFormat, Rgb};

    #[test]
    fn an_images_type_is_read_from_its_bytes() {
        assert_eq!(mime_of(&png(2, 2).0), "image/png");
        assert_eq!(mime_of(b"\xff\xd8\xff\xe0rest"), "image/jpeg");
        assert_eq!(mime_of(b"hello"), "application/octet-stream");
    }

    fn png(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let raw = img.clone().into_raw();
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, ImageFormat::Png).unwrap();
        (out.into_inner(), raw)
    }

    /// An image already at its grid keeps its exact pixels.
    #[test]
    fn an_image_at_its_grid_keeps_its_pixels() {
        let (bytes, raw) = png(640, 480);
        let p = prepare(&bytes).unwrap();
        assert_eq!((p.width, p.height), (640, 480));
        assert_eq!((p.image.width, p.image.height), (640, 480));
        assert_eq!(p.image.rgb, raw);
        assert_eq!(p.id.len(), ID_HEX);
        assert!(p.sha256.starts_with(&p.id));
    }

    /// Another size is resized to `smart_resize`'s grid.
    #[test]
    fn an_image_is_resized_to_its_grid() {
        let (bytes, _) = png(100, 50);
        let p = prepare(&bytes).unwrap();
        assert_eq!((p.image.width, p.image.height), (384, 192));
        assert_eq!(p.image.rgb.len(), 384 * 192 * 3);
    }

    /// An image's id is the start of its bytes' SHA-256 — `abc`'s is
    /// `ba7816bf8f01cfea` — whether it is decoded or not.
    #[test]
    fn an_image_is_named_by_its_bytes_digest() {
        assert_eq!(image_id(b"abc"), "ba7816bf8f01cfea");
        let (bytes, _) = png(64, 64);
        assert_eq!(image_id(&bytes), prepare(&bytes).unwrap().id);
    }

    #[test]
    fn bytes_that_are_not_an_image_are_refused() {
        assert!(prepare(b"hello").is_err());
    }
}
