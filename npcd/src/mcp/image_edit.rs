//! `image_edit {image, mode?, …}`: change a picture — redraw it by a prompt,
//! resize, crop, adjust, rotate, or remove its background
//! ([`super::edit_mode`]).
//!
//! **`prompt`** (the default) is the same draw as `image_generate`
//! (`POST /v1/image/generate`), started from the picture: the picture is the
//! draw's **reference**: the walk starts from it rather than
//! from noise, and `hold` is how much of it survives. So the prompt describes
//! the whole picture as it should end up, not only the change, and the answer
//! is a new picture with its own id and link — which is itself something to
//! edit, so a picture is refined a step at a time. The chain keeps two: an
//! edit kept an hour, and the picture two edits back retired as the new one
//! is made (`crate::images::Kept`), so the step before is always there to
//! undo to and repeated edits do not pile up.
//!
//! **The other modes need no draw.** Resize, crop, adjust and rotate are
//! pixel work done here in a moment ([`crate::image_ops`]), never queued
//! behind a draw and never short of room on the card; remove_background is
//! the matte guest (`POST /v1/image/cutout`), one forward, answered as a PNG
//! with the background transparent. Every mode answers alike — a new image in
//! the chain, its link, its lifetime.
//!
//! **A picture is named, never sent.** `image` takes the link or id a draw, an
//! edit or an upload (`image_upload`) answered with, and the bytes are read
//! from the image store — so a refinement costs the conversation one line, and
//! a picture uploaded once is edited as often as wanted.
//!
//! **An edit is placed among the members that hold its picture.** The picture
//! is its key (`image`, a subject the gateway routes by), and the member that
//! stored it let the key go at once, at its own records then
//! (`Authored::hand_over_image`): the gateway places the edit, as it places a
//! conversation's next turn, on the best member that has applied them — never
//! one that would wait — with the storing member the fallback
//! (`docs/cooperative_admission.md` §2.2). A draw names no picture and goes to
//! the best member anywhere. An image is immutable and named by its hash, so it
//! is never stale — only present or not; not present means it expired, or the
//! member that stored it failed before its records left, which no wait mends,
//! so the edit says so at once and the caller stores it again.

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use candle_conversation::guest::MAX_REFERENCE_HOLD;
use cluster::dispatch::image_id;
use serde_json::{json, Value};
use tokio::task::spawn_blocking;

use super::args::string;
use super::edit_mode::{pixel, preview, Mode};
use super::image::{answer, body, draw, format, Drawn, Made, MAX_SIDE};
use super::route::refused;
use super::{Answer, Tool};
use crate::api::Authored;
use crate::guest_routes::{post_cutout, CutoutBody, ImageBody};
use crate::image_ops;
use crate::images::{ImageError, Images};
use crate::refimage;

pub struct Edit;

/// How much of the picture survives when the caller does not say.
///
/// Above the draw's own reference default (0.25), which is set for a new
/// subject in a held composition: an edit keeps the subject too, and from about
/// 0.45 the prompt can no longer change what the picture shows of it. Half
/// keeps the picture recognisably itself while a detail added or removed still
/// lands; the guide says which way to move it.
pub const HOLD: f32 = 0.5;

impl Tool for Edit {
    const NAME: &'static str = "image_edit";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Edit a picture: redraw it by a prompt (add or remove a detail, change \
                the light, the colours or the style), resize it, crop it, adjust its brightness, \
                contrast or saturation, rotate or mirror it, or remove its background — on the \
                user's own image model. `image` is the link or id image_generate, an earlier \
                image_edit, or image_upload (for a picture of the user's own) answered with; \
                `mode` picks the edit (prompt by default), and each mode takes only its own \
                arguments. For prompt, the prompt describes the whole picture as it should end \
                up, and `hold` is how much of the original survives. Every edit makes a new \
                image with its own link, kept for 1 hour or until two more edits follow it; the \
                image before stays until the new one is edited, so editing from it undoes a \
                step. Read the guide `image` first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "The picture to edit: the link or image id an image_generate, image_edit or image_upload answer gave."},
                    "mode": {"type": "string", "enum": ["prompt", "resize", "crop", "adjust", "rotate", "remove_background"], "default": "prompt", "description": "The edit: prompt (redraw by a prompt), resize, crop, adjust (brightness, contrast, saturation), rotate (and mirror), remove_background."},
                    "prompt": {"type": "string", "description": "prompt: the whole picture as it should end up, with the change in it."},
                    "hold": {"type": "number", "minimum": 0.0, "maximum": MAX_REFERENCE_HOLD, "description": "prompt: how much of the original survives, 0.5 when absent — about 0.3 for a large change, 0.5 to add or remove a detail, 0.7 for a light touch."},
                    "width": {"type": "integer", "minimum": 1, "maximum": image_ops::MAX_SIDE, "description": "prompt or resize: width in pixels (prompt: a multiple of 16, at most 1024); the other side follows the picture's shape when only one is given."},
                    "height": {"type": "integer", "minimum": 1, "maximum": image_ops::MAX_SIDE, "description": "prompt or resize: height in pixels, as width."},
                    "scale": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 4.0, "description": "resize: a factor of the picture's own size, in place of width and height (at most 2048 a side)."},
                    "crop": {"description": "crop: an aspect cut from the centre — square, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3 — or a box {x, y, w, h} as fractions of the picture (0 to 1, from the top left).", "oneOf": [{"type": "string"}, {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"}}, "required": ["x", "y", "w", "h"]}]},
                    "brightness": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "adjust: -1 to 1, 0 unchanged (0.2 is a little brighter)."},
                    "contrast": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "adjust: -1 (flat grey) to 1, 0 unchanged."},
                    "saturation": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "adjust: -1 (greyscale) to 1, 0 unchanged."},
                    "rotate": {"type": "integer", "enum": [90, 180, 270], "description": "rotate: degrees clockwise."},
                    "flip": {"type": "string", "enum": ["horizontal", "vertical"], "description": "rotate: mirror left to right, or top to bottom (after any turn)."},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 100, "description": "prompt: denoise steps, as image_generate's (12 when absent)."},
                    "seed": {"type": "integer", "minimum": 0, "description": "prompt: seed; random when absent."},
                    "shift": {"type": "number", "minimum": 1.0, "maximum": 8.0, "description": "prompt: the schedule's shift; the deployment's own when absent."},
                    "format": {"type": "string", "enum": ["jpeg", "png"], "default": "jpeg", "description": "How the picture comes back; remove_background is always a PNG, to keep its transparency."}
                },
                "required": ["image"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let checked = Mode::of(args).and_then(|mode| {
            mode.check(args)?;
            Ok((mode, format(args)?))
        });
        let (mode, format) = match checked {
            Ok(m) => m,
            Err(e) => return Answer::error(e),
        };
        let source = match source(s, args) {
            Ok(b) => b,
            Err(e) => return Answer::error(e),
        };
        let from = image_id(args["image"].as_str().unwrap_or_default());
        let made = Made::Edited { from };
        match mode {
            Mode::Prompt => match edit_body(args, &source) {
                Ok(body) => draw(s, headers, body, format, made).await,
                Err(e) => Answer::error(e),
            },
            Mode::RemoveBackground => match remove_background(s, &source).await {
                Ok(drawn) => answer(s, headers, drawn, made).await,
                Err(e) => Answer::error(e),
            },
            _ => {
                // A moment's pixel work, but a 2048-pixel resample all the
                // same: off the async workers.
                let args = args.clone();
                match spawn_blocking(move || pixel(mode, &args, &source, format)).await {
                    Ok(Ok(drawn)) => answer(s, headers, drawn, made).await,
                    Ok(Err(e)) => Answer::error(e),
                    Err(e) => Answer::error(format!("the edit failed: {e}")),
                }
            }
        }
    }
}

/// The picture with its background removed — the matte guest, through the
/// route that serves it (`POST /v1/image/cutout`) — as a PNG, transparent where
/// the background was.
async fn remove_background(s: &Arc<Authored>, source: &[u8]) -> Result<Drawn, String> {
    let body = CutoutBody {
        png_base64: BASE64.encode(source),
    };
    let res = post_cutout(State(Arc::clone(s)), Json(body)).await;
    let status = res.status();
    let bytes = to_bytes(res.into_body(), CUTOUT_LIMIT)
        .await
        .map_err(|e| format!("the cutout could not be read: {e}"))?;
    if !status.is_success() {
        return Err(refused(status.as_u16(), &bytes));
    }
    let v: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("the cutout's answer: {e}"))?;
    let png64 = v["png_base64"]
        .as_str()
        .ok_or("the cutout answered without a picture")?;
    let file = BASE64
        .decode(png64.as_bytes())
        .map_err(|e| format!("the cutout's picture is not base64: {e}"))?;
    // Decoding a 2048-pixel PNG and encoding its preview: off the async workers.
    let (file, (shown, shown_mime)) = spawn_blocking(move || {
        let img = image::load_from_memory(&file)
            .map_err(|e| format!("the cutout's picture could not be read: {e}"))?;
        let shown = preview(&img, &file, "image/png")?;
        Ok::<_, String>((file, shown))
    })
    .await
    .map_err(|e| format!("the cutout's preview failed: {e}"))??;
    Ok(Drawn {
        said: format!(
            "Removed the background: now a {}x{} PNG, transparent where the background was \
             (shown here over a checkerboard; the file at the link is the transparent PNG).",
            v["width"], v["height"]
        ),
        file,
        shown,
        shown_mime,
    })
}

/// The most of the cutout's answer read back: a 2048-pixel PNG with
/// transparency, in base64, with room.
const CUTOUT_LIMIT: usize = 48 << 20;

/// The bytes of the picture `image` names.
fn source(s: &Authored, args: &Value) -> Result<Vec<u8>, String> {
    let id = image_id(string(args, "image")?);
    if !Images::is_id(id) {
        return Err(format!(
            "`{id}` is not an image id — pass the link or id a draw, edit or upload answered with"
        ));
    }
    match s.images.get(id) {
        Ok((bytes, _)) => Ok(bytes),
        Err(ImageError::NotFound) => Err(unavailable(id)),
        Err(e) => Err(format!("`{id}` could not be read: {e}")),
    }
}

/// What an edit says when its picture is not here: it is not anywhere it can
/// be reached, and waiting will not bring it, so the picture is stored again.
fn unavailable(id: &str) -> String {
    format!(
        "image `{id}` is not available: it expired (a draw or edit is kept 24 hours, an upload \
         one hour) or the server holding it failed before it was shared. Store it again — \
         image_upload for the user's picture, or draw it again — then edit the new one"
    )
}

/// The draw's body: the tool's dials as `image_generate` reads them, the
/// picture as its reference, held by `hold`, and the picture's own shape when
/// no size is given.
fn edit_body(args: &Value, source: &[u8]) -> Result<ImageBody, String> {
    if source.len() > refimage::MAX_BYTES {
        return Err(format!(
            "the picture is {} bytes and the limit is {}",
            source.len(),
            refimage::MAX_BYTES
        ));
    }
    let mut body = body(args)?;
    let given = |key: &str| args.get(key).filter(|v| !v.is_null()).is_some();
    let (given_w, given_h) = (given("width"), given("height"));
    // The schema's sides are a resize's too; a redraw's are the draw's own.
    for (side, given, key) in [
        (body.width, given_w, "width"),
        (body.height, given_h, "height"),
    ] {
        if given && (side < 16 || side % 16 != 0) {
            return Err(format!(
                "a prompt edit's `{key}` is a multiple of 16 from 16 to {MAX_SIDE}"
            ));
        }
    }
    if !(given_w && given_h) {
        let (w, h) = refimage::dimensions(source)?;
        (body.width, body.height) = shaped(
            (w, h),
            given_w.then_some(body.width),
            given_h.then_some(body.height),
        );
    }
    body.reference_hold = match args.get("hold") {
        None | Some(Value::Null) => HOLD,
        Some(v) => match v.as_f64() {
            Some(h) if (0.0..=MAX_REFERENCE_HOLD as f64).contains(&h) => h as f32,
            _ => return Err(format!("`hold` is a number from 0 to {MAX_REFERENCE_HOLD}")),
        },
    };
    body.reference = Some(BASE64.encode(source));
    Ok(body)
}

/// A `w` × `h` picture's shape as a draw can take it: the long side at most
/// [`MAX_SIDE`], both sides multiples of 16, the aspect kept as near as that
/// allows.
fn fitted_size(w: u32, h: u32) -> (u32, u32) {
    let (w, h) = (w.max(1), h.max(1));
    let long = w.max(h).min(MAX_SIDE);
    let scale = long as f64 / w.max(h) as f64;
    (snap(w as f64 * scale), snap(h as f64 * scale))
}

/// The draw's size for a `picture` of `(w, h)` when the caller gave `width`,
/// `height`, or neither: a side given is kept and the other follows the
/// picture's aspect; with neither, [`fitted_size`].
fn shaped(picture: (u32, u32), width: Option<u32>, height: Option<u32>) -> (u32, u32) {
    let (w, h) = (picture.0.max(1) as f64, picture.1.max(1) as f64);
    match (width, height) {
        (Some(width), Some(height)) => (width, height),
        (Some(width), None) => (width, snap(width as f64 * h / w)),
        (None, Some(height)) => (snap(height as f64 * w / h), height),
        (None, None) => fitted_size(picture.0, picture.1),
    }
}

/// A side as a draw takes it: the nearest multiple of 16, within the limits.
fn snap(side: f64) -> u32 {
    ((side / 16.0).round() as u32 * 16).clamp(16, MAX_SIDE)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn a_link_or_an_id_names_the_image() {
        let id = "ab".repeat(32) + ".jpg";
        for given in [
            format!("https://bot.tokera.com/v1/image/link/{id}?exp=1&sig=cd"),
            format!("https://bot.tokera.com/v1/image/{id}"),
            id.clone(),
        ] {
            assert_eq!(image_id(&given), id, "{given}");
        }
    }

    /// **The original's shape is kept**, its long side brought to the draw's
    /// limit and both sides to multiples of 16.
    #[test]
    fn a_picture_keeps_its_shape_within_the_draws_limits() {
        assert_eq!(fitted_size(576, 1024), (576, 1024));
        assert_eq!(fitted_size(768, 768), (768, 768));
        assert_eq!(fitted_size(4032, 3024), (1024, 768));
        assert_eq!(fitted_size(1080, 1920), (576, 1024));
        assert_eq!(fitted_size(500, 333), (496, 336));
        assert_eq!(fitted_size(3000, 10), (1024, 16));
    }

    /// **One side given keeps it and the other follows the picture.**
    #[test]
    fn one_side_given_and_the_other_follows_the_picture() {
        assert_eq!(shaped((1920, 1080), Some(512), None), (512, 288));
        assert_eq!(shaped((1920, 1080), None, Some(576)), (1024, 576));
        assert_eq!(shaped((1080, 1920), None, Some(1024)), (576, 1024));
        assert_eq!(shaped((1920, 1080), Some(512), Some(512)), (512, 512));
        assert_eq!(shaped((1920, 1080), None, None), (1024, 576));
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::new(w, h))
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    /// **The picture is the reference, held by `hold`**, and sizes the draw
    /// unless a size is given.
    #[test]
    fn the_picture_is_the_reference_and_sizes_the_draw() {
        let pic = png(1920, 1080);
        let b = edit_body(&json!({"prompt": "add a lantern"}), &pic).unwrap();
        assert_eq!((b.width, b.height), (1024, 576));
        assert_eq!(b.reference_hold, HOLD);
        assert_eq!(b.reference.as_deref(), Some(BASE64.encode(&pic).as_str()));

        let b = edit_body(
            &json!({"prompt": "x", "hold": 0.7, "width": 512, "height": 512}),
            &pic,
        )
        .unwrap();
        assert_eq!((b.width, b.height, b.reference_hold), (512, 512, 0.7));
        let b = edit_body(&json!({"prompt": "x", "width": 512}), &pic).unwrap();
        assert_eq!((b.width, b.height), (512, 288));

        assert!(edit_body(&json!({"prompt": "x", "hold": 1.0}), &pic).is_err());
        assert!(
            edit_body(&json!({"prompt": "x", "width": 500}), &pic).is_err(),
            "a redraw's side is a multiple of 16"
        );
        assert!(edit_body(&json!({"prompt": "x"}), b"not a picture").is_err());
        assert!(edit_body(&json!({"prompt": "  "}), &pic).is_err());
    }
}
