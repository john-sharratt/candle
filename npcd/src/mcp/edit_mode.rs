//! What `image_edit` does: its `mode`, the arguments each mode takes, and the
//! modes that need no model — resize, crop, adjust, rotate — as a function of
//! the arguments and the picture.
//!
//! **One flat set of arguments; each mode reads its own.** A model calls a
//! tool most reliably with plain optional arguments, not nested shapes, so
//! every mode's dials sit side by side and an argument given to a mode that
//! does not read it is refused, naming the ones that mode does — the call is
//! corrected in one step rather than guessed at.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use image::{DynamicImage, GenericImageView};
use serde_json::Value;

use super::image::{Drawn, Format};
use crate::image_ops::{self, Adjust, Crop, Flip};
use crate::refimage;

/// An edit's mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Redraw by a prompt, starting from the picture (the image guest).
    Prompt,
    Resize,
    Crop,
    /// Brightness, contrast and saturation.
    Adjust,
    /// Turn by a quarter, a half or three quarters, and mirror.
    Rotate,
    /// Separate the subject from its background (the matte guest).
    RemoveBackground,
}

/// The arguments every mode takes.
const COMMON: [&str; 3] = ["image", "mode", "format"];

impl Mode {
    /// The `mode` argument; `prompt` when absent.
    pub fn of(args: &Value) -> Result<Mode, String> {
        let named = match args.get("mode") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_str().map(str::trim).ok_or(
                "`mode` is a word: prompt, resize, crop, adjust, rotate or remove_background",
            )?),
        };
        match named {
            None | Some("") | Some("prompt") => Ok(Mode::Prompt),
            Some("resize") => Ok(Mode::Resize),
            Some("crop") => Ok(Mode::Crop),
            Some("adjust") => Ok(Mode::Adjust),
            Some("rotate") => Ok(Mode::Rotate),
            Some("remove_background") => Ok(Mode::RemoveBackground),
            Some(other) => Err(format!(
                "`mode` is prompt, resize, crop, adjust, rotate or remove_background, not `{other}`"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Prompt => "prompt",
            Mode::Resize => "resize",
            Mode::Crop => "crop",
            Mode::Adjust => "adjust",
            Mode::Rotate => "rotate",
            Mode::RemoveBackground => "remove_background",
        }
    }

    /// The arguments this mode reads, beside [`COMMON`].
    pub fn params(self) -> &'static [&'static str] {
        match self {
            Mode::Prompt => &[
                "prompt",
                "hold",
                "width",
                "height",
                "steps",
                "seed",
                "shift",
                "authorization",
            ],
            Mode::Resize => &["width", "height", "scale"],
            Mode::Crop => &["crop"],
            Mode::Adjust => &["brightness", "contrast", "saturation"],
            Mode::Rotate => &["rotate", "flip"],
            Mode::RemoveBackground => &[],
        }
    }

    /// Refuse an argument this mode does not read, naming those it does.
    pub fn check(self, args: &Value) -> Result<(), String> {
        let Some(given) = args.as_object() else {
            return Ok(());
        };
        if self == Mode::RemoveBackground
            && given.get("format").and_then(Value::as_str) == Some("jpeg")
        {
            return Err(
                "remove_background answers a PNG, to keep the transparency a JPEG cannot hold"
                    .into(),
            );
        }
        let stray: Vec<&str> = given
            .iter()
            .filter(|(k, v)| !v.is_null() && !COMMON.contains(&k.as_str()))
            .map(|(k, _)| k.as_str())
            .filter(|k| !self.params().contains(k))
            .collect();
        if stray.is_empty() {
            return Ok(());
        }
        let takes = if self.params().is_empty() {
            "nothing beyond image and format".to_string()
        } else {
            self.params().join(", ")
        };
        Err(format!(
            "mode `{}` does not take {}; it takes {takes}",
            self.name(),
            stray
                .iter()
                .map(|k| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// A pixel mode — resize, crop, adjust or rotate — applied to `source`, kept
/// as `format`.
pub(super) fn pixel(
    mode: Mode,
    args: &Value,
    source: &[u8],
    format: Format,
) -> Result<Drawn, String> {
    // Upright, transparency kept, within the limits every picture is held to —
    // and within what an edit makes, a portrait larger than that brought in
    // first.
    let mut img = refimage::decode_upright(source)?;
    let (w, h) = img.dimensions();
    if w.max(h) > image_ops::MAX_SIDE {
        let size = image_ops::resized_size(
            (w, h),
            (w >= h).then_some(image_ops::MAX_SIDE),
            (h > w).then_some(image_ops::MAX_SIDE),
            None,
        )?;
        img = image_ops::resize(&img, size);
    }
    let (img, done) = match mode {
        Mode::Resize => {
            let side = |k: &str| side_arg(args, k);
            let scale = number(args, "scale")?;
            let size =
                image_ops::resized_size(img.dimensions(), side("width")?, side("height")?, scale)?;
            (image_ops::resize(&img, size), "Resized".to_string())
        }
        Mode::Crop => {
            let crop = Crop::parse(args.get("crop").ok_or(
                "crop takes `crop`: an aspect (square, 16:9, 9:16, 4:3, 3:2…) or a box {x, y, w, h}",
            )?)?;
            (image_ops::crop(&img, crop), "Cropped".to_string())
        }
        Mode::Adjust => {
            let dial = |k: &str| -> Result<f64, String> {
                match number(args, k)? {
                    None => Ok(0.0),
                    Some(v) if (-1.0..=1.0).contains(&v) => Ok(v),
                    Some(_) => Err(format!("`{k}` is from -1 to 1, 0 leaving it as it is")),
                }
            };
            let dials = Adjust {
                brightness: dial("brightness")?,
                contrast: dial("contrast")?,
                saturation: dial("saturation")?,
            };
            if dials == Adjust::default() {
                return Err(
                    "adjust takes brightness, contrast or saturation, each from -1 to 1".into(),
                );
            }
            let said = format!(
                "Adjusted brightness {:+.2}, contrast {:+.2}, saturation {:+.2}",
                dials.brightness, dials.contrast, dials.saturation
            );
            (image_ops::adjust(&img, dials), said)
        }
        Mode::Rotate => {
            let degrees = match args.get("rotate").filter(|v| !v.is_null()) {
                None => 0,
                Some(v) => match v.as_u64() {
                    Some(d @ (90 | 180 | 270)) => d as u32,
                    _ => return Err("`rotate` is 90, 180 or 270 — degrees clockwise".into()),
                },
            };
            let flip = match args.get("flip").and_then(Value::as_str).map(str::trim) {
                None => None,
                Some("horizontal") => Some(Flip::Horizontal),
                Some("vertical") => Some(Flip::Vertical),
                Some(other) => {
                    return Err(format!("`flip` is horizontal or vertical, not `{other}`"))
                }
            };
            if degrees == 0 && flip.is_none() {
                return Err("rotate takes `rotate` (90, 180, 270) or `flip`, or both".into());
            }
            let mut said = match degrees {
                0 => "Mirrored".to_string(),
                d => format!("Rotated {d}° clockwise"),
            };
            if let (Some(f), true) = (flip, degrees != 0) {
                said.push_str(match f {
                    Flip::Horizontal => " and mirrored left to right",
                    Flip::Vertical => " and mirrored top to bottom",
                });
            }
            (image_ops::rotate(&img, degrees, flip), said)
        }
        Mode::Prompt | Mode::RemoveBackground => {
            return Err(format!("mode `{}` is not a pixel mode", mode.name()))
        }
    };
    let (w, h) = img.dimensions();
    // A picture with transparency stays a PNG unless a JPEG is asked for, so a
    // cutout edited on keeps its cutting.
    let png = match args.get("format").filter(|v| !v.is_null()) {
        None => img.color().has_alpha(),
        Some(_) => format == Format::Png,
    };
    let (file, mime) = image_ops::encode(&img, png)?;
    let (shown, shown_mime) = preview(&img, &file, mime)?;
    Ok(Drawn {
        said: format!("{done}: now {w}x{h}."),
        file,
        shown,
        shown_mime,
    })
}

/// The longest side of the picture an answer carries for the model to look
/// at: what a client takes inline, as a draw's own limit. The file kept is
/// the whole picture.
const PREVIEW_SIDE: u32 = 1024;

/// What an answer shows the model, in base64 with its type: **always a JPEG
/// within [`PREVIEW_SIDE`]** — the file itself when it already is one, else
/// the picture brought within it, a picture with transparency laid over a
/// checkerboard ([`image_ops::on_checkerboard`]) so its cutting shows.
///
/// The answer is one message, and on a slow link it is the whole of the wait:
/// a cutout's PNG runs to a megabyte and took minutes to reach the client
/// where its JPEG takes seconds — and a PNG's transparency is not seen at all
/// by a model reading its colours. The file kept is the PNG, whole.
pub(super) fn preview(
    img: &DynamicImage,
    file: &[u8],
    mime: &'static str,
) -> Result<(String, &'static str), String> {
    let (w, h) = img.dimensions();
    let small = w.max(h) <= PREVIEW_SIDE;
    if small && mime == "image/jpeg" {
        return Ok((BASE64.encode(file), mime));
    }
    let mut shown = if img.color().has_alpha() {
        DynamicImage::ImageRgb8(image_ops::on_checkerboard(img))
    } else {
        img.clone()
    };
    if !small {
        let size = image_ops::resized_size(
            (w, h),
            (w >= h).then_some(PREVIEW_SIDE),
            (h > w).then_some(PREVIEW_SIDE),
            None,
        )?;
        shown = image_ops::resize(&shown, size);
    }
    let (jpeg, jpeg_mime) = image_ops::encode(&shown, false)?;
    Ok((BASE64.encode(jpeg), jpeg_mime))
}

/// A number argument, absent as `None`.
fn number(args: &Value, key: &str) -> Result<Option<f64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or(format!("`{key}` is a number")),
    }
}

/// A side in pixels, absent as `None`.
fn side_arg(args: &Value, key: &str) -> Result<Option<u32>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n > 0)
            .map(Some)
            .ok_or(format!("`{key}` is a whole number of pixels")),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    use serde_json::json;

    use super::*;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = RgbImage::from_fn(w, h, |x, y| Rgb([x as u8, y as u8, 7]));
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut out, ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    fn decoded(d: &Drawn) -> DynamicImage {
        image::load_from_memory(&d.file).unwrap()
    }

    #[test]
    fn the_mode_is_named_or_prompt_by_default() {
        assert_eq!(Mode::of(&json!({})), Ok(Mode::Prompt));
        assert_eq!(Mode::of(&json!({"mode": "crop"})), Ok(Mode::Crop));
        assert_eq!(
            Mode::of(&json!({"mode": "remove_background"})),
            Ok(Mode::RemoveBackground)
        );
        assert!(Mode::of(&json!({"mode": "blur"})).is_err());
        assert!(Mode::of(&json!({"mode": 1})).is_err(), "not a word");
    }

    /// **A picture with transparency stays a PNG unless a JPEG is asked
    /// for**, and is shown as a JPEG over the checkerboard; a large one is
    /// shown within 1024 while the file kept is whole.
    #[test]
    fn a_cutout_stays_a_png_and_a_large_picture_is_previewed() {
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image::RgbaImage::new(8, 4))
            .write_to(&mut out, ImageFormat::Png)
            .unwrap();
        let clear = out.into_inner();
        let d = pixel(Mode::Rotate, &json!({"rotate": 180}), &clear, Format::Jpeg).unwrap();
        assert_eq!(
            &d.file[..4],
            b"\x89PNG",
            "format not given: the cutout's own"
        );
        assert_eq!(d.shown_mime, "image/jpeg");
        let shown = image::load_from_memory(&BASE64.decode(&d.shown).unwrap())
            .unwrap()
            .to_rgb8();
        assert_eq!(shown.dimensions(), (8, 4));
        let lum = |p: &Rgb<u8>| p.0.iter().map(|&c| c as u32).sum::<u32>() / 3;
        assert!(
            lum(shown.get_pixel(0, 0)) > 240,
            "clear shows the light square"
        );
        let d = pixel(
            Mode::Rotate,
            &json!({"rotate": 180, "format": "jpeg"}),
            &clear,
            Format::Jpeg,
        )
        .unwrap();
        assert_eq!(&d.file[..2], &[0xFF, 0xD8], "asked for");

        let big = png(1536, 768);
        let d = pixel(Mode::Rotate, &json!({"rotate": 180}), &big, Format::Jpeg).unwrap();
        assert_eq!(decoded(&d).dimensions(), (1536, 768), "the file is whole");
        let shown = image::load_from_memory(&BASE64.decode(&d.shown).unwrap()).unwrap();
        assert_eq!(
            (shown.dimensions(), d.shown_mime),
            ((1024, 512), "image/jpeg"),
            "what is shown is a JPEG within 1024"
        );
    }

    #[test]
    fn remove_background_answers_a_png_and_says_so() {
        assert!(Mode::RemoveBackground
            .check(&json!({"image": "x", "format": "jpeg"}))
            .is_err());
        assert_eq!(
            Mode::RemoveBackground.check(&json!({"image": "x", "format": "png"})),
            Ok(())
        );
    }

    /// **An argument of another mode is refused, naming this mode's own.**
    #[test]
    fn a_stray_argument_names_what_the_mode_takes() {
        let crop = json!({"image": "x", "mode": "crop", "crop": "square", "format": "png"});
        assert_eq!(Mode::Crop.check(&crop), Ok(()));
        assert_eq!(
            Mode::Crop.check(&json!({"image": "x", "crop": "square", "brightness": 0.2})),
            Err("mode `crop` does not take `brightness`; it takes crop".into())
        );
        assert_eq!(
            Mode::RemoveBackground.check(&json!({"image": "x", "prompt": "p"})),
            Err(
                "mode `remove_background` does not take `prompt`; it takes nothing beyond image \
                 and format"
                    .into()
            )
        );
        assert_eq!(
            Mode::Prompt.check(&json!({"image": "x", "prompt": "p", "hold": 0.5, "seed": null})),
            Ok(())
        );
    }

    #[test]
    fn each_pixel_mode_makes_its_picture() {
        let src = png(8, 4);
        let d = pixel(Mode::Resize, &json!({"width": 4}), &src, Format::Png).unwrap();
        assert_eq!(
            (decoded(&d).dimensions(), d.said.as_str()),
            ((4, 2), "Resized: now 4x2.")
        );

        let d = pixel(
            Mode::Crop,
            &json!({"crop": "square", "format": "png"}),
            &src,
            Format::Png,
        )
        .unwrap();
        let out = decoded(&d).to_rgb8();
        assert_eq!(out.dimensions(), (4, 4));
        assert_eq!(out.get_pixel(0, 0), &Rgb([2, 0, 7]), "the centre square");

        let d = pixel(
            Mode::Adjust,
            &json!({"brightness": 0.25, "format": "png"}),
            &src,
            Format::Png,
        )
        .unwrap();
        assert_eq!(decoded(&d).to_rgb8().get_pixel(0, 0), &Rgb([32, 32, 39]));
        assert_eq!(
            d.said,
            "Adjusted brightness +0.25, contrast +0.00, saturation +0.00: now 8x4."
        );

        let d = pixel(Mode::Rotate, &json!({"rotate": 90}), &src, Format::Png).unwrap();
        assert_eq!(decoded(&d).dimensions(), (4, 8));
        assert_eq!(d.said, "Rotated 90° clockwise: now 4x8.");
        let d = pixel(
            Mode::Rotate,
            &json!({"flip": "horizontal", "format": "png"}),
            &src,
            Format::Png,
        )
        .unwrap();
        assert_eq!(decoded(&d).to_rgb8().get_pixel(0, 0), &Rgb([7, 0, 7]));
        assert_eq!(d.said, "Mirrored: now 8x4.");

        let d = pixel(Mode::Crop, &json!({"crop": "square"}), &src, Format::Jpeg).unwrap();
        assert_eq!(&d.file[..2], &[0xFF, 0xD8], "a JPEG");
        assert_eq!(
            (d.shown.as_str(), d.shown_mime),
            (BASE64.encode(&d.file).as_str(), "image/jpeg"),
            "a small JPEG is shown as itself"
        );
    }

    #[test]
    fn a_pixel_mode_with_nothing_to_do_says_what_it_takes() {
        let src = png(8, 4);
        for (mode, args) in [
            (Mode::Resize, json!({})),
            (Mode::Crop, json!({})),
            (Mode::Adjust, json!({})),
            (Mode::Adjust, json!({"brightness": 2})),
            (Mode::Rotate, json!({})),
            (Mode::Rotate, json!({"rotate": 45})),
            (Mode::Rotate, json!({"flip": "diagonal"})),
        ] {
            assert!(
                pixel(mode, &args, &src, Format::Png).is_err(),
                "{mode:?} {args}"
            );
        }
    }
}
