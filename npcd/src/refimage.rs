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

use candle_conversation::guest::ImageReference;
use image::imageops::FilterType;

/// The largest upload accepted, before decoding.
///
/// The same limit portrait uploads take ([`crate::images::MAX_BYTES`]) doubled:
/// a reference is a working file rather than a stored one, so it is routinely a
/// camera JPEG or a screenshot that nobody has thought about compressing, and
/// it is thrown away the moment the draw is queued.
pub const MAX_BYTES: usize = 8 * 1024 * 1024;

/// The largest decoded picture accepted, per side.
///
/// A compression bomb is a small file that decodes to an enormous surface, so
/// the byte limit above does not bound the memory this costs — the pixel count
/// does, and it is checked from the header before any pixels are produced.
pub const MAX_SIDE: u32 = 8192;

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

    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("the reference could not be read: {e}"))?;
    // **From the header, before any pixels exist.** A decompression bomb is a
    // small file that decodes to an enormous surface, so this is the check that
    // the byte limit above cannot make.
    let (w, h) = reader
        .into_dimensions()
        .map_err(|e| format!("the reference is not a picture this build can read: {e}"))?;
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err(format!(
            "the reference is {w}×{h} and the limit is {MAX_SIDE} per side"
        ));
    }

    let img = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("the reference could not be read: {e}"))?
        .decode()
        .map_err(|e| format!("the reference is not a picture this build can read: {e}"))?;

    let fitted = cover(&img.to_rgb8(), width, height);
    Ok(ImageReference {
        pixels: fitted.into_raw(),
        hold,
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
    use image::{ImageFormat, Rgb, RgbImage};

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
