//! An image's thumbnail: what a conversation draws in place of the original,
//! so a history of photographs loads in kilobytes, with the original fetched
//! only when one is opened.
//!
//! Made once, when the image is prepared, and kept beside it in the substrate
//! (`vision::store`).

use image::codecs::jpeg::JpegEncoder;
use image::imageops::{resize, FilterType};
use image::RgbImage;

/// The longest side of a thumbnail, in pixels: twice the 220 the web UI draws
/// a message's images at, so it is sharp on a high-density screen.
pub const THUMB_EDGE: u32 = 480;

/// JPEG quality of a thumbnail.
const THUMB_QUALITY: u8 = 82;

/// The size `width × height` takes as a thumbnail: within [`THUMB_EDGE`] on its
/// longest side, its aspect kept, at least one pixel each way, and never
/// enlarged.
pub fn thumb_size(width: u32, height: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= THUMB_EDGE {
        return (width, height);
    }
    let scale = |side: u32| {
        ((side as u64 * THUMB_EDGE as u64 + longest as u64 / 2) / longest as u64).max(1) as u32
    };
    (scale(width), scale(height))
}

/// `rgb`'s thumbnail, as a JPEG.
pub fn thumbnail(rgb: &RgbImage) -> Result<Vec<u8>, String> {
    let (width, height) = thumb_size(rgb.width(), rgb.height());
    let mut out = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut out, THUMB_QUALITY);
    if (width, height) == rgb.dimensions() {
        rgb.write_with_encoder(encoder)
    } else {
        resize(rgb, width, height, FilterType::Triangle).write_with_encoder(encoder)
    }
    .map_err(|e| format!("a thumbnail: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{load_from_memory, Rgb};

    /// The longest side comes to the edge, the other in proportion, rounded;
    /// an image already within it keeps its size.
    #[test]
    fn a_thumbnail_fits_the_edge_and_keeps_its_aspect() {
        assert_eq!(thumb_size(1920, 1080), (480, 270));
        assert_eq!(thumb_size(768, 512), (480, 320));
        assert_eq!(thumb_size(1000, 3000), (160, 480));
        assert_eq!(thumb_size(300, 200), (300, 200));
        assert_eq!(thumb_size(5000, 1), (480, 1));
    }

    /// The thumbnail is a JPEG of that size.
    #[test]
    fn a_thumbnail_is_a_jpeg_of_its_size() {
        let rgb = RgbImage::from_fn(768, 512, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 7]));
        let jpeg = thumbnail(&rgb).unwrap();
        assert_eq!(&jpeg[..3], &[0xff, 0xd8, 0xff]);
        let back = load_from_memory(&jpeg).unwrap();
        assert_eq!((back.width(), back.height()), (480, 320));
    }
}
