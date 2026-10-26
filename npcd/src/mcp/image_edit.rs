//! `image_edit {image, prompt, hold?, width?, height?, steps?, seed?, shift?,
//! format?}`: change a picture by a prompt — the same draw as
//! `image_generate` (`POST /v1/image/generate`), started from the picture.
//!
//! The picture is the draw's **reference**: the walk starts from it rather than
//! from noise, and `hold` is how much of it survives. So the prompt describes
//! the whole picture as it should end up, not only the change, and the answer
//! is a new picture with its own id and link — which is itself something to
//! edit, so a picture is refined a step at a time. The chain keeps two: an
//! edit kept an hour, and the picture two edits back retired as the new one
//! is made (`crate::images::Kept`), so the step before is always there to
//! undo to and repeated edits do not pile up.
//!
//! **A picture is named, never sent.** `image` takes the link or id a draw, an
//! edit or an upload (`image_upload`) answered with, and the bytes are read
//! from the image store — so a refinement costs the conversation one line, and
//! a picture uploaded once is edited as often as wanted.
//!
//! **Every member holds every image, so whichever serves the edit has it.** An
//! image is a blob in the substrate, which replicates, and the gateway orders
//! a write's records ahead of its response: a call it routes afterwards, to
//! any member, is served only once that member has applied them
//! (`docs/replicated_substrate.md` §6.4). An image is immutable and named by
//! its hash, so it is never stale — only present or not. Not present means it
//! expired, or the member that stored it failed before its records left; it
//! will not arrive by waiting, so the edit says so at once and the caller
//! stores it again.

use std::sync::Arc;

use axum::http::HeaderMap;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use candle_conversation::guest::MAX_REFERENCE_HOLD;
use serde_json::{json, Value};

use super::args::string;
use super::image::{body, draw, format, Made, MAX_SIDE, STEPS};
use super::{Answer, Tool};
use crate::api::Authored;
use crate::guest_routes::ImageBody;
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
            "description": "Edit a picture by a prompt — add or remove a detail, change the light, \
                the colours or the style, refine a draw — on the user's own image model. `image` \
                is the link or id image_generate, an earlier image_edit, or image_upload (for a \
                picture of the user's own) answered with. The prompt describes the whole picture \
                as it should end up, not only the change; `hold` is how much of the original \
                survives. Every edit makes a new image with its own link, kept for 1 hour or \
                until two more edits follow it; the image before stays until the new one is \
                edited, so editing from it undoes a step. Read the guide `image` first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "The picture to edit: the link or image id an image_generate, image_edit or image_upload answer gave."},
                    "prompt": {"type": "string", "description": "The whole picture as it should end up, with the change in it."},
                    "hold": {"type": "number", "minimum": 0.0, "maximum": MAX_REFERENCE_HOLD, "default": HOLD, "description": "How much of the original survives: about 0.3 for a large change, 0.5 to add or remove a detail, 0.7 for a light touch."},
                    "width": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "description": "Width in pixels, a multiple of 16; the original's shape when absent."},
                    "height": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "description": "Height in pixels, a multiple of 16; the original's shape when absent."},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": STEPS, "description": "Denoise steps, as image_generate's."},
                    "seed": {"type": "integer", "minimum": 0, "description": "Seed; random when absent."},
                    "shift": {"type": "number", "minimum": 1.0, "maximum": 8.0, "description": "The schedule's shift; the deployment's own when absent."},
                    "format": {"type": "string", "enum": ["jpeg", "png"], "default": "jpeg", "description": "How the picture comes back."}
                },
                "required": ["image", "prompt"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let source = match source(s, args) {
            Ok(b) => b,
            Err(e) => return Answer::error(e),
        };
        let from = image_id(args["image"].as_str().unwrap_or_default());
        match edit_body(args, &source).and_then(|b| Ok((b, format(args)?))) {
            Ok((body, format)) => draw(s, headers, body, format, Made::Edited { from }).await,
            Err(e) => Answer::error(e),
        }
    }
}

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

/// The image id in a link this server gave — `…/v1/image/link/<id>?exp=…&sig=…`
/// or `…/v1/image/<id>` — or the id itself.
fn image_id(link: &str) -> &str {
    let path = link.split(['?', '#']).next().unwrap_or(link);
    path.rsplit('/').next().unwrap_or(path)
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
        assert!(edit_body(&json!({"prompt": "x"}), b"not a picture").is_err());
        assert!(edit_body(&json!({"prompt": "  "}), &pic).is_err());
    }
}
