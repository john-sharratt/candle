//! What `image_edit` does to a picture without a model: resize, crop, adjust
//! its brightness, contrast and saturation, rotate or flip it — and encode the
//! result as the file kept.
//!
//! Each is a pure function of the picture and its dials, so each is tested
//! against exact pixel values. Transparency, where a picture has it (one whose
//! background was removed), is carried through every operation and kept in a
//! PNG; a JPEG has none, so it is laid over white first.

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb, RgbImage, Rgba};
use serde_json::Value;

/// The longest side an operation makes: an upload's limit, so nothing an edit
/// makes is larger than what could have been uploaded.
pub const MAX_SIDE: u32 = 2048;

/// The resampling filter for every resize.
const FILTER: FilterType = FilterType::Lanczos3;

/// The JPEG quality a result is kept at — an upload's, past where a step of
/// re-encoding shows.
const JPEG_QUALITY: u8 = 92;

/// The size `img` is resized to: `width` and `height` as given, one of them
/// with the other following the picture's aspect, or `scale` times its own.
pub fn resized_size(
    (w, h): (u32, u32),
    width: Option<u32>,
    height: Option<u32>,
    scale: Option<f64>,
) -> Result<(u32, u32), String> {
    let side = |v: f64| v.round().max(1.0) as u32;
    let (nw, nh) = match (width, height, scale) {
        (Some(nw), Some(nh), None) => (nw, nh),
        (Some(nw), None, None) => (nw, side(nw as f64 * h as f64 / w as f64)),
        (None, Some(nh), None) => (side(nh as f64 * w as f64 / h as f64), nh),
        (None, None, Some(s)) if s > 0.0 => (side(w as f64 * s), side(h as f64 * s)),
        (None, None, Some(_)) => return Err("`scale` is more than 0".into()),
        (None, None, None) => {
            return Err("resize takes `width`, `height` (one keeps the shape), or `scale`".into())
        }
        _ => return Err("resize takes `width`/`height`, or `scale`, not both".into()),
    };
    if nw == 0 || nh == 0 || nw > MAX_SIDE || nh > MAX_SIDE {
        return Err(format!(
            "{nw}×{nh} is outside 1 to {MAX_SIDE} pixels a side"
        ));
    }
    Ok((nw, nh))
}

/// `img` resized to `(w, h)`.
pub fn resize(img: &DynamicImage, (w, h): (u32, u32)) -> DynamicImage {
    img.resize_exact(w, h, FILTER)
}

/// What a crop keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Crop {
    /// The largest centred rectangle of this width-to-height ratio.
    Aspect(u32, u32),
    /// A rectangle as fractions of the picture: its left, top, width, height.
    Box { x: f64, y: f64, w: f64, h: f64 },
}

impl Crop {
    /// The `crop` argument: an aspect (`square`, `16:9`, `9:16`, `4:3`,
    /// `3:4`, `3:2`, `2:3`) or a box `{x, y, w, h}` of fractions.
    pub fn parse(v: &Value) -> Result<Crop, String> {
        if let Some(s) = v.as_str() {
            return match s.trim() {
                "square" | "1:1" => Ok(Crop::Aspect(1, 1)),
                other => {
                    let (a, b) = other.split_once(':').ok_or(ASPECTS)?;
                    match (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                        (Ok(a), Ok(b)) if a > 0 && b > 0 => Ok(Crop::Aspect(a, b)),
                        _ => Err(ASPECTS.into()),
                    }
                }
            };
        }
        let f = |k: &str| {
            v.get(k).and_then(Value::as_f64).ok_or(format!(
                "a crop box needs `{k}`, a fraction of the picture from 0 to 1"
            ))
        };
        let (x, y, w, h) = (f("x")?, f("y")?, f("w")?, f("h")?);
        let inside = |a: f64| (0.0..=1.0).contains(&a);
        if !(inside(x)
            && inside(y)
            && w > 0.0
            && h > 0.0
            && x + w <= 1.0 + 1e-9
            && y + h <= 1.0 + 1e-9)
        {
            return Err(
                "a crop box is fractions of the picture: x and y from 0, w and h more than 0, \
                 x + w and y + h at most 1"
                    .into(),
            );
        }
        Ok(Crop::Box { x, y, w, h })
    }

    /// The pixel rectangle this keeps of a `(w, h)` picture: left, top,
    /// width, height, at least one pixel each way.
    pub fn rect(self, (w, h): (u32, u32)) -> (u32, u32, u32, u32) {
        match self {
            Crop::Aspect(a, b) => {
                // As wide as the picture allows at this ratio, else as tall.
                let (cw, ch) = if w as u64 * b as u64 >= h as u64 * a as u64 {
                    ((h as u64 * a as u64 / b as u64) as u32, h)
                } else {
                    (w, (w as u64 * b as u64 / a as u64) as u32)
                };
                let (cw, ch) = (cw.max(1), ch.max(1));
                ((w - cw) / 2, (h - ch) / 2, cw, ch)
            }
            Crop::Box { x, y, w: bw, h: bh } => {
                let px = |f: f64, of: u32| ((f * of as f64).round() as u32).min(of);
                let (left, top) = (px(x, w).min(w - 1), px(y, h).min(h - 1));
                let cw = px(bw, w).clamp(1, w - left);
                let ch = px(bh, h).clamp(1, h - top);
                (left, top, cw, ch)
            }
        }
    }
}

const ASPECTS: &str = "`crop` is an aspect — square, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3 — or a box \
     {x, y, w, h} of fractions";

/// `img` cropped as `crop` says.
pub fn crop(img: &DynamicImage, crop: Crop) -> DynamicImage {
    let (x, y, w, h) = crop.rect(img.dimensions());
    img.crop_imm(x, y, w, h)
}

/// The dials of an adjustment, each from −1 to +1, 0 leaving it as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Adjust {
    /// Added to every channel, as a share of half the range: +1 adds 128.
    pub brightness: f64,
    /// The spread about mid-grey: −1 flattens to grey, +1 doubles it.
    pub contrast: f64,
    /// The spread about each pixel's own luma: −1 is greyscale, +1 doubles it.
    pub saturation: f64,
}

impl Adjust {
    /// One pixel's colour adjusted: saturation, then contrast, then brightness,
    /// each rounded to the nearest value and held in range.
    pub fn pixel(&self, [r, g, b]: [u8; 3]) -> [u8; 3] {
        let luma = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
        let one = |c: u8| {
            let mut v = c as f64;
            v = luma + (v - luma) * (1.0 + self.saturation);
            v = (v - 128.0) * (1.0 + self.contrast) + 128.0;
            v += 128.0 * self.brightness;
            v.round().clamp(0.0, 255.0) as u8
        };
        [one(r), one(g), one(b)]
    }
}

/// `img` adjusted by `dials`; its transparency, if any, kept.
pub fn adjust(img: &DynamicImage, dials: Adjust) -> DynamicImage {
    if img.color().has_alpha() {
        let mut rgba = img.to_rgba8();
        for p in rgba.pixels_mut() {
            let [r, g, b, a] = p.0;
            let [r, g, b] = dials.pixel([r, g, b]);
            *p = Rgba([r, g, b, a]);
        }
        DynamicImage::ImageRgba8(rgba)
    } else {
        let mut rgb = img.to_rgb8();
        for p in rgb.pixels_mut() {
            *p = Rgb(dials.pixel(p.0));
        }
        DynamicImage::ImageRgb8(rgb)
    }
}

/// A mirroring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flip {
    Horizontal,
    Vertical,
}

/// `img` turned clockwise by `degrees` (0, 90, 180 or 270), then mirrored.
pub fn rotate(img: &DynamicImage, degrees: u32, flip: Option<Flip>) -> DynamicImage {
    let turned = match degrees {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img.clone(),
    };
    match flip {
        Some(Flip::Horizontal) => turned.fliph(),
        Some(Flip::Vertical) => turned.flipv(),
        None => turned,
    }
}

/// The side of a [`on_checkerboard`] square, in pixels.
const CHECKER: u32 = 16;

/// `img` laid over a light-grey checkerboard, as transparency is shown: what a
/// model looking at a cutout sees of it. A picture's transparency is only in
/// its alpha, which a look at its colours does not show — over the board, a
/// removed background reads as removed.
pub fn on_checkerboard(img: &DynamicImage) -> RgbImage {
    let rgba = img.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let square = if (x / CHECKER + y / CHECKER).is_multiple_of(2) {
            255u32
        } else {
            204
        };
        let [r, g, b, a] = rgba.get_pixel(x, y).0;
        let blend = |c: u8| ((c as u32 * a as u32 + square * (255 - a as u32) + 127) / 255) as u8;
        Rgb([blend(r), blend(g), blend(b)])
    })
}

/// `img` as the file kept: a PNG (transparency kept) or a JPEG (laid over
/// white first), with its type.
pub fn encode(img: &DynamicImage, png: bool) -> Result<(Vec<u8>, &'static str), String> {
    if png {
        let mut out = Cursor::new(Vec::new());
        img.write_to(&mut out, ImageFormat::Png)
            .map_err(|e| format!("the picture could not be encoded: {e}"))?;
        return Ok((out.into_inner(), "image/png"));
    }
    let rgb: RgbImage = if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
            let [r, g, b, a] = rgba.get_pixel(x, y).0;
            let blend = |c: u8| ((c as u32 * a as u32 + 255 * (255 - a as u32) + 127) / 255) as u8;
            Rgb([blend(r), blend(g), blend(b)])
        })
    } else {
        img.to_rgb8()
    };
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY)
        .encode_image(&rgb)
        .map_err(|e| format!("the picture could not be encoded: {e}"))?;
    Ok((jpeg, "image/jpeg"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A `w`×`h` picture whose pixel at (x, y) is (x, y, 7) — every pixel
    /// says where it came from.
    fn coords(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| Rgb([x as u8, y as u8, 7])))
    }

    #[test]
    fn a_resize_keeps_the_shape_unless_both_sides_are_given() {
        assert_eq!(
            resized_size((1920, 1080), Some(512), None, None),
            Ok((512, 288))
        );
        assert_eq!(
            resized_size((1920, 1080), None, Some(540), None),
            Ok((960, 540))
        );
        assert_eq!(
            resized_size((1920, 1080), Some(100), Some(100), None),
            Ok((100, 100))
        );
        assert_eq!(
            resized_size((768, 512), None, None, Some(2.0)),
            Ok((1536, 1024))
        );
        assert_eq!(
            resized_size((768, 512), None, None, Some(0.25)),
            Ok((192, 128))
        );
        assert!(
            resized_size((768, 512), None, None, Some(4.0)).is_err(),
            "past 2048"
        );
        assert!(resized_size((768, 512), Some(10), None, Some(2.0)).is_err());
        assert!(resized_size((768, 512), None, None, None).is_err());
        assert!(resized_size((768, 512), None, None, Some(0.0)).is_err());
        assert_eq!(resize(&coords(8, 4), (4, 2)).dimensions(), (4, 2));
    }

    #[test]
    fn a_crop_by_aspect_keeps_the_largest_centred_rectangle() {
        assert_eq!(Crop::Aspect(1, 1).rect((1920, 1080)), (420, 0, 1080, 1080));
        assert_eq!(Crop::Aspect(16, 9).rect((1000, 1000)), (0, 219, 1000, 562));
        assert_eq!(Crop::Aspect(9, 16).rect((1920, 1080)), (656, 0, 607, 1080));
        assert_eq!(Crop::Aspect(4, 3).rect((800, 600)), (0, 0, 800, 600));
    }

    #[test]
    fn a_crop_by_box_keeps_those_fractions_and_those_pixels() {
        let b = Crop::parse(&json!({"x": 0.5, "y": 0.25, "w": 0.5, "h": 0.5})).unwrap();
        assert_eq!(b.rect((8, 4)), (4, 1, 4, 2));
        let out = crop(&coords(8, 4), b).to_rgb8();
        assert_eq!(out.dimensions(), (4, 2));
        assert_eq!(out.get_pixel(0, 0), &Rgb([4, 1, 7]), "the box's top left");
        assert_eq!(out.get_pixel(3, 1), &Rgb([7, 2, 7]), "its bottom right");
    }

    #[test]
    fn a_crop_is_named_by_an_aspect_or_a_box_and_nothing_else() {
        assert_eq!(Crop::parse(&json!("square")), Ok(Crop::Aspect(1, 1)));
        assert_eq!(Crop::parse(&json!(" 16:9 ")), Ok(Crop::Aspect(16, 9)));
        for bad in [
            json!("wide"),
            json!("0:9"),
            json!({"x": 0.5, "y": 0, "w": 0.6, "h": 1}),
            json!({"x": 0, "y": 0, "w": 0, "h": 1}),
            json!({"x": 0, "y": 0, "w": 1}),
        ] {
            assert!(Crop::parse(&bad).is_err(), "{bad}");
        }
    }

    /// **The dials, by their exact values.** Each alone, then all three in
    /// their order: saturation, contrast, brightness.
    #[test]
    fn an_adjustment_moves_each_channel_by_its_dial() {
        let px = [200, 100, 50];
        assert_eq!(Adjust::default().pixel(px), px, "zero leaves it");
        let b = |v| Adjust {
            brightness: v,
            ..Default::default()
        };
        assert_eq!(b(0.25).pixel(px), [232, 132, 82]);
        assert_eq!(b(-1.0).pixel(px), [72, 0, 0]);
        let c = |v| Adjust {
            contrast: v,
            ..Default::default()
        };
        assert_eq!(c(-1.0).pixel(px), [128, 128, 128], "flat grey");
        assert_eq!(c(1.0).pixel(px), [255, 72, 0]);
        let s = |v| Adjust {
            saturation: v,
            ..Default::default()
        };
        // Luma 0.299·200 + 0.587·100 + 0.114·50 = 124.2.
        assert_eq!(s(-1.0).pixel(px), [124, 124, 124], "greyscale");
        assert_eq!(s(0.5).pixel(px), [238, 88, 13]);
        let all = Adjust {
            brightness: 0.1,
            contrast: 0.2,
            saturation: -0.5,
        };
        // Saturation: 162.1, 112.1, 87.1; contrast: 168.92, 108.92, 78.92;
        // brightness +12.8: 181.72, 121.72, 91.72.
        assert_eq!(all.pixel(px), [182, 122, 92]);
    }

    #[test]
    fn transparency_survives_an_adjustment_and_is_kept_in_a_png() {
        let mut rgba = image::RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, Rgba([200, 100, 50, 0]));
        rgba.put_pixel(1, 0, Rgba([200, 100, 50, 255]));
        let out = adjust(
            &DynamicImage::ImageRgba8(rgba),
            Adjust {
                brightness: 0.25,
                ..Default::default()
            },
        )
        .to_rgba8();
        assert_eq!(out.get_pixel(0, 0), &Rgba([232, 132, 82, 0]));

        let (png, mime) = encode(&DynamicImage::ImageRgba8(out.clone()), true).unwrap();
        assert_eq!(mime, "image/png");
        let back = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!(back.get_pixel(0, 0).0[3], 0, "transparency kept");
        let (jpeg, mime) = encode(&DynamicImage::ImageRgba8(out), false).unwrap();
        assert_eq!((mime, &jpeg[..2]), ("image/jpeg", &[0xFF, 0xD8][..]));
    }

    /// **Transparency shows as the board**: a clear pixel is the square under
    /// it, an opaque one itself, a half-clear one between.
    #[test]
    fn a_cutout_on_the_checkerboard_shows_where_it_is_clear() {
        let mut rgba = image::RgbaImage::new(32, 1);
        rgba.put_pixel(0, 0, Rgba([10, 20, 30, 0]));
        rgba.put_pixel(16, 0, Rgba([10, 20, 30, 0]));
        rgba.put_pixel(1, 0, Rgba([10, 20, 30, 255]));
        rgba.put_pixel(17, 0, Rgba([0, 0, 0, 128]));
        let out = on_checkerboard(&DynamicImage::ImageRgba8(rgba));
        assert_eq!(out.get_pixel(0, 0), &Rgb([255, 255, 255]), "a light square");
        assert_eq!(out.get_pixel(16, 0), &Rgb([204, 204, 204]), "a dark square");
        assert_eq!(out.get_pixel(1, 0), &Rgb([10, 20, 30]), "opaque");
        // (0·128 + 204·127 + 127) / 255 = 102.
        assert_eq!(out.get_pixel(17, 0), &Rgb([102, 102, 102]));
    }

    #[test]
    fn a_rotation_turns_clockwise_then_mirrors() {
        let img = coords(3, 2);
        let r = rotate(&img, 90, None).to_rgb8();
        assert_eq!(r.dimensions(), (2, 3));
        // Clockwise: the bottom-left corner comes to the top left.
        assert_eq!(r.get_pixel(0, 0), &Rgb([0, 1, 7]));
        let r = rotate(&img, 180, Some(Flip::Horizontal)).to_rgb8();
        assert_eq!(r.get_pixel(0, 0), &Rgb([0, 1, 7]), "turned, then mirrored");
        let r = rotate(&img, 0, Some(Flip::Vertical)).to_rgb8();
        assert_eq!(r.get_pixel(0, 0), &Rgb([0, 1, 7]));
        assert_eq!(rotate(&img, 270, None).dimensions(), (2, 3));
    }
}
