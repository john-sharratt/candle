//! `image_generate {prompt, width?, height?, steps?, seed?, shift?, format?}`:
//! draw a picture — `POST /v1/image/generate`, the console's own draw.
//!
//! The route streams NDJSON (`loading`, a `step` per unit, then `done` with
//! the PNG); the tool reads the stream to its end and answers with the last
//! line. The picture comes back as an MCP image block beside a line of text,
//! so the model sees what it drew rather than a base64 string it cannot read.
//!
//! **JPEG unless PNG is asked for.** The answer is one message, and on a slow
//! link to the client it is the whole of the wait: a 1024×768 PNG is 1.2 MB of
//! base64, and through a gateway on a degraded Wi-Fi link it reached the client
//! two minutes after the draw had finished. A JPEG of it is about a sixth of
//! that, and reads the same to a model.
//!
//! The draw takes the route's compliance gate and its standing checkpoint; the
//! admin-only `restricted` one is not offered here.

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use candle_conversation::guest::{ImageLora, DEFAULT_REFERENCE_HOLD};
use image::codecs::jpeg::JpegEncoder;
use image::ImageFormat;
use serde_json::{json, Value};

use super::args::string;
use super::route::refused;
use super::{Answer, Picture, Tool};
use crate::api::Authored;
use crate::guest_routes::{post_image, ImageBody};

pub struct Generate;

/// The largest side the tool draws. The guest takes 2048, but a picture handed
/// back into a conversation is read at about a megapixel and a 2048² PNG is
/// past what a client accepts inline.
pub const MAX_SIDE: u32 = 1024;

/// The steps a draw takes when the caller names none.
///
/// Twelve rather than the route's twenty-four: a conversation asking for a
/// picture wants it back, and twelve is half the denoise for a finish the eye
/// does not tell apart at this size. Eight, the model's trained count, is the
/// fast end.
pub const STEPS: u32 = 12;

/// The most of the route's stream read back: a 1024² PNG in base64 with room.
const STREAM_LIMIT: usize = 16 << 20;

/// The quality a picture is sent back at as JPEG — where the artefacts stop
/// showing at a glance, which is how a conversation looks at it.
const JPEG_QUALITY: u8 = 90;

/// What the picture is sent back as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Jpeg,
    Png,
}

impl Tool for Generate {
    const NAME: &'static str = "image_generate";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Generate an image from a text prompt (Z-Image-Turbo) and return it — a \
                JPEG, or a PNG when `format` asks for one. Describe the subject, style, lighting and composition in plain sentences; \
                variety comes from the prompt, not the seed. A draw takes about half a minute \
                to two, and progress arrives while it works. Prompts are checked first and a \
                declined one is refused.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What to draw."},
                    "width": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "default": 512, "description": "Width in pixels, a multiple of 16."},
                    "height": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "default": 512, "description": "Height in pixels, a multiple of 16."},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": STEPS, "description": "Denoise steps: 12 is the default and a good finish; 8 (the model's trained count) for speed. More buys finish, not variety."},
                    "seed": {"type": "integer", "minimum": 0, "description": "Seed, for repeating a draw; random when absent."},
                    "shift": {"type": "number", "minimum": 1.0, "maximum": 8.0, "description": "The schedule's shift; the deployment's own when absent."},
                    "format": {"type": "string", "enum": ["jpeg", "png"], "default": "jpeg", "description": "How the picture comes back: jpeg (about a sixth of the bytes) or png (lossless)."}
                },
                "required": ["prompt"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let (body, format) = match body(args).and_then(|b| Ok((b, format(args)?))) {
            Ok(b) => b,
            Err(e) => return Answer::error(e),
        };
        let res = post_image(State(Arc::clone(s)), headers.clone(), Json(body)).await;
        let status = res.status();
        let bytes = match to_bytes(res.into_body(), STREAM_LIMIT).await {
            Ok(b) => b,
            Err(e) => return Answer::error(format!("the draw could not be read: {e}")),
        };
        if !status.is_success() {
            return Answer::error(refused(status.as_u16(), &bytes));
        }
        outcome(&bytes, format)
    }
}

/// The `format` argument.
fn format(args: &Value) -> Result<Format, String> {
    match args.get("format").and_then(Value::as_str).map(str::trim) {
        None | Some("") | Some("jpeg") | Some("jpg") => Ok(Format::Jpeg),
        Some("png") => Ok(Format::Png),
        Some(other) => Err(format!("`format` is jpeg or png, not `{other}`")),
    }
}

/// The route's PNG, base64, as the picture `format` asks for.
fn picture(png_base64: &str, format: Format) -> Result<Picture, String> {
    if format == Format::Png {
        return Ok(Picture {
            data: png_base64.to_string(),
            mime_type: "image/png".into(),
        });
    }
    let png = BASE64
        .decode(png_base64.as_bytes())
        .map_err(|e| format!("the picture is not base64: {e}"))?;
    let rgb = image::load_from_memory_with_format(&png, ImageFormat::Png)
        .map_err(|e| format!("the picture would not decode: {e}"))?
        .to_rgb8();
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY)
        .encode_image(&rgb)
        .map_err(|e| format!("the picture would not encode: {e}"))?;
    Ok(Picture {
        data: BASE64.encode(&jpeg),
        mime_type: "image/jpeg".into(),
    })
}

/// The route's body from the tool's arguments.
fn body(args: &Value) -> Result<ImageBody, String> {
    let prompt = string(args, "prompt")?.to_string();
    let side = |key: &str| -> Result<u32, String> {
        match args.get(key) {
            None | Some(Value::Null) => Ok(512),
            Some(v) => match v.as_u64() {
                Some(n) if n <= MAX_SIDE as u64 => Ok(n as u32),
                _ => Err(format!("`{key}` is a whole number up to {MAX_SIDE}")),
            },
        }
    };
    let steps = match args.get("steps") {
        None | Some(Value::Null) => STEPS,
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("`steps` is a whole number")?,
    };
    let seed = match args.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or("`seed` is a whole number")?),
    };
    let shift = match args.get("shift") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_f64().ok_or("`shift` is a number")?),
    };
    Ok(ImageBody {
        prompt,
        width: side("width")?,
        height: side("height")?,
        steps,
        seed,
        lora: ImageLora::Diversity,
        reference: None,
        reference_hold: DEFAULT_REFERENCE_HOLD,
        shift,
    })
}

/// The answer the stream's last line carries, its picture as `format`.
fn outcome(stream: &[u8], format: Format) -> Answer {
    let last = String::from_utf8_lossy(stream)
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l).ok());
    let Some(last) = last else {
        return Answer::error("the draw ended without an answer");
    };
    match last["event"].as_str() {
        Some("done") => match last["png_base64"].as_str().map(|p| picture(p, format)) {
            Some(Ok(picture)) => Answer::picture(
                format!(
                    "Drew a {}x{} image, seed {}.",
                    last["width"], last["height"], last["seed"]
                ),
                picture,
            ),
            Some(Err(e)) => Answer::error(e),
            None => Answer::error("the draw finished without a picture"),
        },
        Some("error") => {
            let retry = if last["retry"] == true {
                " (worth retrying)"
            } else {
                ""
            };
            Answer::error(format!(
                "{}: {}{retry}",
                last["error"].as_str().unwrap_or("failed"),
                last["detail"].as_str().unwrap_or("")
            ))
        }
        _ => Answer::error(format!("the draw ended on `{last}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_take_the_routes_defaults_and_refuse_a_side_past_the_limit() {
        let b = body(&json!({"prompt": " a red fox "})).unwrap();
        assert_eq!(
            (b.prompt.as_str(), b.width, b.height, b.steps),
            ("a red fox", 512, 512, STEPS)
        );
        assert_eq!((b.seed, b.shift), (None, None));
        let b = body(&json!({"prompt": "x", "width": 768, "seed": 7, "shift": 3.0})).unwrap();
        assert_eq!((b.width, b.seed, b.shift), (768, Some(7), Some(3.0)));
        assert!(body(&json!({"prompt": "x", "width": 2048})).is_err());
        assert!(body(&json!({"prompt": "  "})).is_err());
    }

    #[test]
    fn a_finished_draw_is_a_picture_and_an_error_line_is_an_error() {
        let done = b"{\"event\":\"loading\"}\n{\"event\":\"step\",\"done\":1,\"total\":9}\n\
            {\"event\":\"done\",\"width\":512,\"height\":512,\"png_base64\":\"iVBO\",\"seed\":7}\n";
        let a = outcome(done, Format::Png);
        assert!(!a.is_error);
        assert_eq!(a.text, "Drew a 512x512 image, seed 7.");
        assert_eq!(a.picture.unwrap().data, "iVBO");

        let failed =
            b"{\"event\":\"error\",\"error\":\"no_room\",\"detail\":\"full\",\"retry\":true}\n";
        let a = outcome(failed, Format::Jpeg);
        assert!(a.is_error);
        assert_eq!(a.text, "no_room: full (worth retrying)");
        assert!(outcome(b"", Format::Jpeg).is_error);
    }

    /// **A JPEG unless PNG is asked for**, and it is a real one: the route's
    /// PNG decoded and encoded again, at the picture's own size.
    #[test]
    fn the_picture_comes_back_as_jpeg_unless_png_is_asked_for() {
        assert_eq!(format(&json!({})).unwrap(), Format::Jpeg);
        assert_eq!(format(&json!({"format": "png"})).unwrap(), Format::Png);
        assert!(format(&json!({"format": "gif"})).is_err());

        let img =
            image::RgbImage::from_fn(32, 16, |x, y| image::Rgb([x as u8 * 8, y as u8 * 16, 90]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();
        let p = picture(&BASE64.encode(&png), Format::Jpeg).unwrap();
        assert_eq!(p.mime_type, "image/jpeg");
        let jpeg = BASE64.decode(p.data.as_bytes()).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "not a JPEG");
        let back = image::load_from_memory_with_format(&jpeg, ImageFormat::Jpeg).unwrap();
        assert_eq!((back.width(), back.height()), (32, 16));
    }
}
