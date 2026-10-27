//! Turning an uploaded picture into a reference the image guest can start from.
//!
//! # Why this is the boundary's job and not the guest's
//!
//! What arrives is a file somebody dragged onto a web page: some format, some
//! size, possibly not a picture at all, possibly a decompression bomb. What the
//! guest takes is RGB8 at exactly the draw's own width and height — an
//! invariant [`candle_conversation::guest::GuestRequest::check`] enforces at
//! submission.
//!
//! Everything between those two is here, on purpose. A drain evicts the
//! engine's whole working set before the guest loads, so a malformed upload
//! discovered *inside* one costs every character in every world its resident KV
//! to find out that a JPEG was truncated. Decoding at the boundary makes that a
//! 400 before anything is queued.
//!
//! # Cover, and why there is no dial for it
//!
//! The reference is **centre-cropped to the draw's aspect and scaled to fill
//! it**. The alternatives are worse in ways that are not a matter of taste:
//! letterboxing hands the model bars of flat colour inside the picture it is
//! completing, and it renders them — they are part of the composition it was
//! given. Stretching distorts every face in the frame, and the model faithfully
//! reproduces the distortion.
//!
//! Cropping loses the edges of a picture whose shape does not match the output,
//! which is a real cost and a visible, predictable one: the caller can see what
//! survived and pick an output shape that suits their reference. That is why
//! the sizes on offer include both 3:2 buckets.

use std::io::Cursor;

use candle_conversation::guest::ImageReference;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, Rgb, RgbImage};

/// The largest upload accepted, before decoding.
///
/// A reference is a working file rather than a stored one, so it is routinely a
/// camera original or a screenshot that nobody has thought about compressing —
/// a 50-megapixel phone JPEG runs to 15 MB, a large PNG screenshot past 20 —
/// and it is thrown away the moment the draw is queued.
pub const MAX_BYTES: usize = 32 * 1024 * 1024;

/// The largest decoded picture accepted, per side.
pub const MAX_SIDE: u32 = 16384;

/// The most pixels a decoded picture may hold: a 64-megapixel camera's.
///
/// A compression bomb is a small file that decodes to an enormous surface, so
/// the byte limit above does not bound the memory this costs — the pixel count
/// does (four bytes each while it is decoded), and it is checked from the
/// header before any pixels are produced.
pub const MAX_PIXELS: u64 = 64_000_000;

/// The filter used to scale a reference.
///
/// Lanczos3 rather than the cheaper triangle: the picture is about to be read
/// by a convolutional encoder, and resampling artefacts are structure as far as
/// that encoder is concerned. It is a few milliseconds once per draw.
const FILTER: FilterType = FilterType::Lanczos3;

/// Decode an upload and fit it to a draw of `width` × `height`.
///
/// The error is a sentence for the caller, because every one of these is
/// something they can act on — a file that is not a picture, one that is too
/// large, one whose format this build cannot read.
pub fn conform(bytes: &[u8], width: u32, height: u32, hold: f32) -> Result<ImageReference, String> {
    if bytes.is_empty() {
        return Err("the reference is empty".into());
    }
    if bytes.len() > MAX_BYTES {
        return Err(format!(
            "the reference is {} bytes and the limit is {MAX_BYTES} — it is a working file, not a \
             stored one, so it is worth shrinking before sending",
            bytes.len()
        ));
    }
    if width == 0 || height == 0 {
        return Err("a reference needs a draw with a non-zero size to fit to".into());
    }
    let fitted = cover(&decode(bytes)?, width, height);
    Ok(ImageReference {
        pixels: fitted.into_raw(),
        hold,
    })
}

/// The picture `bytes` hold the way a person sees it: whichever format it
/// came in, turned as its camera recorded, and any transparency laid over
/// white.
///
/// **Turned**, because a phone stores a portrait photo as a landscape sensor
/// image with an EXIF note saying which way is up; read without the note, it
/// arrives on its side and the draw redraws it on its side. **Over white**,
/// because dropping the alpha channel leaves whatever colour the transparent
/// pixels happened to hold — usually black — which the model reads as part of
/// the picture.
pub fn decode(bytes: &[u8]) -> Result<RgbImage, String> {
    Ok(over_white(decode_upright(bytes)?))
}

/// The picture `bytes` hold, turned as its camera recorded, its transparency
/// kept — what an edit that is not a draw works on.
pub fn decode_upright(bytes: &[u8]) -> Result<DynamicImage, String> {
    let mut decoder = decoder(bytes)?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("the picture could not be decoded: {e}"))?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// The longest side an uploaded picture is kept at: twice the largest draw,
/// so an edit at any size starts from more than it needs and nothing larger
/// is carried around the cluster.
pub const UPLOAD_SIDE: u32 = 2048;

/// The JPEG quality an uploaded picture is kept at: past where re-encoding
/// shows, since it is the start of every edit made from it.
const UPLOAD_QUALITY: u8 = 92;

/// An uploaded picture as Zen keeps it: whatever it came as, decoded
/// ([`decode`] — turned, transparency over white), brought within
/// [`UPLOAD_SIDE`], and encoded as a JPEG. One format and a bounded size for
/// everything downstream, and a 50-megapixel camera original kept as a
/// fraction of a megabyte. Answers the JPEG and its width and height.
pub fn normalised(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    if bytes.is_empty() {
        return Err("the picture is empty".into());
    }
    if bytes.len() > MAX_BYTES {
        return Err(format!(
            "the picture is {} bytes and the limit is {MAX_BYTES}",
            bytes.len()
        ));
    }
    let mut img = decode(bytes)?;
    let (w, h) = img.dimensions();
    if w.max(h) > UPLOAD_SIDE {
        let scale = UPLOAD_SIDE as f64 / w.max(h) as f64;
        let (nw, nh) = (
            ((w as f64 * scale).round() as u32).max(1),
            ((h as f64 * scale).round() as u32).max(1),
        );
        img = image::imageops::resize(&img, nw, nh, FILTER);
    }
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, UPLOAD_QUALITY)
        .encode_image(&img)
        .map_err(|e| format!("the picture could not be encoded: {e}"))?;
    Ok((jpeg, img.width(), img.height()))
}

/// The picture's width and height as [`decode`] will turn it, read from its
/// header alone.
pub fn dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    let mut decoder = decoder(bytes)?;
    let (w, h) = decoder.dimensions();
    let turned = matches!(
        decoder.orientation().unwrap_or(Orientation::NoTransforms),
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    );
    Ok(if turned { (h, w) } else { (w, h) })
}

/// A decoder for `bytes`, its format recognised from the bytes themselves and
/// its size checked **from the header, before any pixels exist** — a
/// decompression bomb is a small file that decodes to an enormous surface, so
/// this is the check the byte limit cannot make.
fn decoder(bytes: &[u8]) -> Result<impl ImageDecoder + '_, String> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("the picture could not be read: {e}"))?;
    if reader.format().is_none() {
        return Err(
            "the picture is not in a format this can read: JPEG, PNG, WebP, GIF, BMP or TIFF"
                .into(),
        );
    }
    let decoder = reader
        .into_decoder()
        .map_err(|e| format!("the picture could not be read: {e}"))?;
    let (w, h) = decoder.dimensions();
    if w > MAX_SIDE || h > MAX_SIDE || w as u64 * h as u64 > MAX_PIXELS {
        return Err(format!(
            "the picture is {w}×{h}; the limit is {MAX_SIDE} per side and {} megapixels",
            MAX_PIXELS / 1_000_000
        ));
    }
    Ok(decoder)
}

/// `img` as RGB, any transparency laid over white.
fn over_white(img: DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let [r, g, b, a] = rgba.get_pixel(x, y).0;
        let blend = |c: u8| ((c as u32 * a as u32 + 255 * (255 - a as u32) + 127) / 255) as u8;
        Rgb([blend(r), blend(g), blend(b)])
    })
}

/// Scale to fill `width` × `height` and centre-crop the overflow.
///
/// Split out from [`conform`] so the geometry — which is the part with an
/// off-by-one in it — is testable without encoding a picture first.
fn cover(src: &image::RgbImage, width: u32, height: u32) -> image::RgbImage {
    let (sw, sh) = (src.width().max(1), src.height().max(1));
    // Scale by whichever axis is *short* relative to the target, so the result
    // covers it on both. `ceil` rather than round: a scale that lands a
    // fraction under would leave a one-pixel strip with nothing in it.
    let scale = (width as f64 / sw as f64).max(height as f64 / sh as f64);
    let rw = ((sw as f64 * scale).ceil() as u32).max(width);
    let rh = ((sh as f64 * scale).ceil() as u32).max(height);
    let resized = image::imageops::resize(src, rw, rh, FILTER);
    let x = (rw - width) / 2;
    let y = (rh - height) / 2;
    image::imageops::crop_imm(&resized, x, y, width, height).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageFormat;

    /// A PNG of `w`×`h`, with a distinguishable left half so a crop is visible.
    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut img = RgbImage::new(w, h);
        for (x, _, p) in img.enumerate_pixels_mut() {
            *p = if x < w / 2 {
                Rgb([200, 30, 30])
            } else {
                Rgb([30, 30, 200])
            };
        }
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    /// **The invariant the guest depends on**, from every input shape: exactly
    /// `w · h · 3` bytes at the draw's own size. The guest reshapes these
    /// without re-deriving the size, so anything else reaches the card.
    ///
    /// **The sizes are small on purpose, and the shapes are the point.** This
    /// ran on 1920×1080 sources against 1248×832 draws and took **17 seconds** —
    /// on its own, more than the rest of the crate's suite put together — because
    /// [`FILTER`] is Lanczos3 and a debug build resamples every output pixel
    /// through a six-tap kernel per channel. What is under test is the size
    /// arithmetic in [`cover`], which is scale-free: what has to vary is the
    /// *relationships* — an aspect wider than the target and narrower than it,
    /// an exact match, an upscale, a degenerate 1×1 and an extreme 17×3 — and
    /// every one of those survives at a tenth of the dimensions. The aspect
    /// ratios are the originals: 192×108 is 16:9 and 60×90 is 2:3.
    ///
    /// The limits that genuinely need large numbers — the byte cap and the
    /// pixel-bomb — are asserted in the test below from a *header*, which costs
    /// nothing precisely because no pixels are ever produced.
    #[test]
    fn every_upload_conforms_to_the_draws_own_size() {
        for (sw, sh) in [(64, 64), (192, 108), (60, 90), (17, 3), (1, 1)] {
            // Encoded once per source rather than once per pair: the inner loop
            // does not vary it, and PNG-encoding the same picture three times
            // was a third of this test's remaining cost.
            let src = png(sw, sh);
            for (dw, dh) in [(128u32, 128u32), (156, 104), (96, 160)] {
                let r = conform(&src, dw, dh, 0.45).unwrap();
                assert_eq!(
                    r.pixels.len(),
                    dw as usize * dh as usize * 3,
                    "{sw}×{sh} → {dw}×{dh}"
                );
                assert_eq!(r.hold, 0.45);
            }
        }
    }

    /// A picture already the right shape is not distorted, and one that is
    /// wider than the target keeps its middle rather than its left edge.
    #[test]
    fn a_cover_crop_keeps_the_centre() {
        // 200 wide, left half red, right half blue, cropped to a 100-wide
        // square: the surviving strip straddles the join, so both are present.
        let mut src = RgbImage::new(200, 100);
        for (x, _, p) in src.enumerate_pixels_mut() {
            *p = if x < 100 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            };
        }
        let out = cover(&src, 100, 100);
        assert_eq!(out.dimensions(), (100, 100));
        assert_eq!(out.get_pixel(0, 50), &Rgb([255, 0, 0]), "the left edge");
        assert_eq!(out.get_pixel(99, 50), &Rgb([0, 0, 255]), "the right edge");
    }

    /// The scale is taken from the axis that needs it most, so the result
    /// covers the target on both — a scale from the wrong axis leaves a strip
    /// of nothing, which would be a band of black in the reference.
    ///
    /// **Both the sources and the target are small, and they had to shrink
    /// together.** This test cost eleven seconds on 4000×100 and 100×4000
    /// sources against a 512×320 target, and the reason is worth stating because
    /// it is not the source size: [`cover`] scales the whole picture to fill the
    /// target and crops afterwards, so a 40:1 source covering a 1.6:1 target is
    /// upscaled to 12800×320 — and the taller one to 512×20480 — before 96% of
    /// it is thrown away. Fifteen million pixels through a Lanczos3 kernel in a
    /// debug build.
    ///
    /// The *aspect* is what drives that, not the pixel count, so shrinking the
    /// sources alone would have changed nothing. Target and sources both come
    /// down by roughly ten, the 40:1 extremes are kept in both directions, and
    /// the assertions are untouched.
    #[test]
    fn a_cover_fills_both_axes_whichever_way_the_aspect_runs() {
        for (sw, sh) in [(400u32, 10u32), (10, 400), (150, 150)] {
            let src = RgbImage::from_pixel(sw, sh, Rgb([9, 9, 9]));
            let out = cover(&src, 64, 40);
            assert_eq!(out.dimensions(), (64, 40), "{sw}×{sh}");
            // Every pixel came from the source, so none is the zero a
            // short scale would have left behind.
            assert!(out.pixels().all(|p| *p == Rgb([9, 9, 9])), "{sw}×{sh}");
        }
    }

    /// `img` encoded as `format`.
    fn encoded(img: DynamicImage, format: ImageFormat) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, format).unwrap();
        out.into_inner()
    }

    /// **Every format a person is likely to hand over is read**, from its own
    /// bytes, to the same picture.
    #[test]
    fn every_reasonable_format_is_read() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(6, 4, Rgb([40, 120, 200])));
        for format in [
            ImageFormat::Png,
            ImageFormat::Jpeg,
            ImageFormat::WebP,
            ImageFormat::Gif,
            ImageFormat::Bmp,
            ImageFormat::Tiff,
        ] {
            let bytes = encoded(img.clone(), format);
            assert_eq!(dimensions(&bytes), Ok((6, 4)), "{format:?}");
            let back = decode(&bytes).unwrap_or_else(|e| panic!("{format:?}: {e}"));
            assert_eq!(back.dimensions(), (6, 4), "{format:?}");
            assert!(conform(&bytes, 32, 32, 0.5).is_ok(), "{format:?}");
        }
    }

    /// **Transparency is laid over white**, exactly: opaque pixels keep their
    /// colour, clear ones become white, half-clear ones meet halfway.
    #[test]
    fn transparency_is_laid_over_white() {
        let mut rgba = image::RgbaImage::new(3, 1);
        rgba.put_pixel(0, 0, image::Rgba([10, 20, 30, 255]));
        rgba.put_pixel(1, 0, image::Rgba([10, 20, 30, 0]));
        rgba.put_pixel(2, 0, image::Rgba([0, 0, 0, 128]));
        let out = over_white(DynamicImage::ImageRgba8(rgba));
        assert_eq!(out.get_pixel(0, 0), &Rgb([10, 20, 30]));
        assert_eq!(out.get_pixel(1, 0), &Rgb([255, 255, 255]));
        assert_eq!(out.get_pixel(2, 0), &Rgb([127, 127, 127]));
    }

    /// A JPEG of `w`×`h` carrying the EXIF orientation `orientation`, built by
    /// hand: an APP1 segment spliced in after the start-of-image marker, holding
    /// a little-endian TIFF header and one IFD entry, tag 0x0112.
    fn jpeg_with_orientation(w: u32, h: u32, orientation: u8) -> Vec<u8> {
        let jpeg = encoded(
            DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([90, 90, 90]))),
            ImageFormat::Jpeg,
        );
        let mut app1 = vec![0xFF, 0xE1, 0x00, 0x22];
        app1.extend_from_slice(b"Exif\0\0");
        app1.extend_from_slice(&[0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00]);
        app1.extend_from_slice(&[0x01, 0x00]);
        app1.extend_from_slice(&[0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00]);
        app1.extend_from_slice(&[orientation, 0x00, 0x00, 0x00]);
        app1.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        assert_eq!(app1.len(), 2 + 0x22, "the segment's length field");
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&app1);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    /// **A phone's portrait photo comes out upright**: stored 8×4 with the
    /// EXIF note "turn 90°", it is read as 4×8, header and pixels alike.
    #[test]
    fn a_photo_is_turned_the_way_its_camera_recorded() {
        let turned = jpeg_with_orientation(8, 4, 6);
        assert_eq!(dimensions(&turned), Ok((4, 8)));
        assert_eq!(decode(&turned).unwrap().dimensions(), (4, 8));
        let upright = jpeg_with_orientation(8, 4, 1);
        assert_eq!(dimensions(&upright), Ok((8, 4)));
        assert_eq!(decode(&upright).unwrap().dimensions(), (8, 4));
    }

    /// **An upload is kept as an upright JPEG within the side limit**: a
    /// sideways phone photo comes out turned, a large one scaled to fit, a
    /// small one left its size, and a PNG with transparency becomes a JPEG.
    #[test]
    fn an_upload_is_kept_as_an_upright_jpeg_within_the_limit() {
        let (jpeg, w, h) = normalised(&jpeg_with_orientation(8, 4, 6)).unwrap();
        assert_eq!((w, h), (4, 8));
        assert_eq!(&jpeg[..3], &[0xFF, 0xD8, 0xFF]);
        assert_eq!(dimensions(&jpeg), Ok((4, 8)));

        let big = encoded(
            DynamicImage::ImageRgb8(RgbImage::new(UPLOAD_SIDE * 2, 100)),
            ImageFormat::Png,
        );
        assert_eq!(normalised(&big).unwrap().1, UPLOAD_SIDE);
        assert_eq!(normalised(&big).unwrap().2, 50);

        let clear = encoded(
            DynamicImage::ImageRgba8(image::RgbaImage::new(5, 3)),
            ImageFormat::Png,
        );
        let (jpeg, w, h) = normalised(&clear).unwrap();
        assert_eq!((w, h, &jpeg[..2]), (5, 3, &[0xFF, 0xD8][..]));
        assert!(normalised(&[]).is_err());
        assert!(normalised(b"not a picture").is_err());
    }

    /// **The three refusals a caller can act on, each named.** A byte limit, a
    /// pixel limit read from the header before the bomb is decoded, and a file
    /// that is not a picture at all.
    #[test]
    fn an_upload_that_cannot_be_used_is_refused_with_a_reason() {
        let e = conform(&[], 512, 512, 0.4).unwrap_err();
        assert!(e.contains("empty"), "{e}");

        let e = conform(&vec![0u8; MAX_BYTES + 1], 512, 512, 0.4).unwrap_err();
        assert!(e.contains("limit"), "{e}");

        let e = conform(b"this is not a picture at all", 512, 512, 0.4).unwrap_err();
        assert!(e.contains("read"), "{e}");

        // A real PNG whose *header* claims more than the side limit. Written by
        // hand rather than encoded, which is the whole point — a bomb is
        // exactly a file whose declared size costs nothing to state.
        let mut bomb = png(8, 8);
        // The IHDR width field: 8 bytes of signature, 4 of length, 4 of "IHDR".
        bomb[16..20].copy_from_slice(&(MAX_SIDE + 1).to_be_bytes());
        let e = conform(&bomb, 512, 512, 0.4).unwrap_err();
        assert!(
            e.contains("limit") || e.contains("read"),
            "an oversized header was not refused: {e}"
        );
    }
}
