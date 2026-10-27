//! `image_generate {prompt, width?, height?, steps?, seed?, shift?, format?}`:
//! draw a picture — `POST /v1/image/generate`, the console's own draw.
//!
//! The route streams NDJSON (`loading`, a `step` per unit, then `done` with
//! the PNG); the tool reads the stream to its end and answers with the last
//! line. The picture comes back as an MCP image block beside a line of text,
//! so the model sees what it drew rather than a base64 string it cannot read.
//!
//! **And a link to the file.** The image block reaches the model, not the
//! person: a client may show no picture from a tool at all. So the draw is
//! stored for a day ([`crate::images`]) and the answer carries a link signed
//! for as long ([`crate::image_links`]) that opens it without signing in —
//! what the model fetches to hand the person the file.
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
use std::time::Instant;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use candle_conversation::guest::{ImageLora, DEFAULT_REFERENCE_HOLD, EXACT_IN_JSON};
use cluster::dispatch::DRAW_DEFAULT_SIDE;
use image::codecs::jpeg::JpegEncoder;
use image::ImageFormat;
use mcp_server::origin;
use serde_json::{json, Value};

use super::args::string;
use super::route::refused;
use super::{Answer, Picture, Tool};
use crate::api::{now_ms, Authored};
use crate::guest_routes::{post_image, ImageBody};
use crate::image_seed;
use crate::images::Kept;

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
pub(super) enum Format {
    Jpeg,
    Png,
}

impl Tool for Generate {
    const NAME: &'static str = "image_generate";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Draw, paint or generate a picture, image, photo or illustration — the \
                user's own image model (Z-Image-Turbo, on their GPU). Use it for any request to \
                draw or create an image, rather than drawing one yourself in SVG, code or a \
                widget. To change a picture already drawn, edited or uploaded — \"make it a \
                black cat\", \"add a hat\", \"brighter\" — use image_edit with its link instead: \
                a new draw is a different picture. The answer is the picture for you to look at — a JPEG, or a PNG when `format` asks for one — and a \
                link to the file that works for 24 hours without signing in. The user may not see \
                the picture in the tool result: to show it to them, download the link and send \
                them the file, or give them the link. Describe the subject, style, lighting and \
                composition in plain sentences; variety comes from the prompt, not the seed. A \
                draw takes about half a minute to two, and progress arrives while it works. \
                Prompts are checked first and a declined one is refused. Read the guide `image` \
                first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What to draw."},
                    "width": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "default": DRAW_DEFAULT_SIDE, "description": "Width in pixels, a multiple of 16."},
                    "height": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "default": DRAW_DEFAULT_SIDE, "description": "Height in pixels, a multiple of 16."},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": STEPS, "description": "Denoise steps: 12 is the default and a good finish; 8 (the model's trained count) for speed. More buys finish, not variety."},
                    "seed": {"type": "integer", "minimum": 0, "maximum": EXACT_IN_JSON, "description": "Seed, for repeating a draw; random when absent. Below 2^53, so it comes back exact."},
                    "shift": {"type": "number", "minimum": 1.0, "maximum": 8.0, "description": "The schedule's shift; the deployment's own when absent."},
                    "format": {"type": "string", "enum": ["jpeg", "png"], "default": "jpeg", "description": "How the picture comes back: jpeg (about a sixth of the bytes) or png (lossless)."}
                },
                "required": ["prompt"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        match body(args).and_then(|b| Ok((b, format(args)?))) {
            Ok((body, format)) => draw(s, headers, body, format, Made::Drawn).await,
            Err(e) => Answer::error(e),
        }
    }
}

/// What a draw makes: a new picture, or an edit of one.
#[derive(Clone, Copy, Debug)]
pub(super) enum Made<'a> {
    /// `image_generate`: kept a day.
    Drawn,
    /// `image_edit` of `from`: kept an hour or until two edits on.
    Edited { from: &'a str },
}

impl Made<'_> {
    /// What the answer says was done: a picture drawn, or one drawn again.
    fn verb(self) -> &'static str {
        match self {
            Made::Drawn => "Drew",
            Made::Edited { .. } => "Redrew",
        }
    }
}

/// Run the draw `body` asks for and answer with its picture as `format`, and
/// a link to it for as long as it is kept — `image_generate`'s answer, and
/// `image_edit`'s.
pub(super) async fn draw(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    body: ImageBody,
    format: Format,
    made: Made<'_>,
) -> Answer {
    let res = post_image(State(Arc::clone(s)), headers.clone(), Json(body)).await;
    let status = res.status();
    let bytes = match to_bytes(res.into_body(), STREAM_LIMIT).await {
        Ok(b) => b,
        Err(e) => return Answer::error(format!("the draw could not be read: {e}")),
    };
    if !status.is_success() {
        return Answer::error(refused(status.as_u16(), &bytes));
    }
    match outcome(&bytes, format, made.verb()) {
        Ok(drawn) => answer(s, headers, drawn, made).await,
        Err(e) => Answer::error(e),
    }
}

/// Keep `drawn` — what a draw or an edit made — and answer with it: stored
/// for as long as `made` says, handed over, the picture two edits back
/// retired, and a link for its lifetime.
pub(super) async fn answer(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    drawn: Drawn,
    made: Made<'_>,
) -> Answer {
    let now = now_ms();
    let started = Instant::now();
    let mut put_ms = None;
    // Handed over the moment it is stored, ahead of anything that awaits, so
    // an edit of it is placed among its holders however this call ends.
    let stored = match made {
        Made::Drawn => s
            .images
            .put_kept(&drawn.file, Kept::Drawn, now)
            .inspect(|id| s.hand_over_image(id))
            .map(|id| (id, Kept::Drawn)),
        Made::Edited { from } => match s.images.put_edited(&drawn.file, from, now) {
            Ok((id, gone)) => {
                s.hand_over_image(&id);
                put_ms = Some(started.elapsed().as_millis() as u64);
                if let Some(gone) = gone {
                    retire_unless_portrait(s, &gone).await;
                }
                Ok((id, Kept::Edited))
            }
            Err(e) => Err(e),
        },
    };
    let total_ms = started.elapsed().as_millis() as u64;
    let put_ms = put_ms.unwrap_or(total_ms);
    // The store's two stages: writing the picture, and retiring the one two
    // edits back (a scan of the portraits that might still use it).
    tracing::info!(
        target: "npcd::mcp::image",
        bytes = drawn.file.len(),
        put_ms,
        retire_ms = total_ms - put_ms,
        "image kept"
    );
    let link = stored.map(|(id, why)| {
        let path = s.image_links.path(&id, now / 1000, why);
        (format!("{}{path}", origin::of(headers)), why)
    });
    Answer::picture(
        said(&drawn.said, link.map_err(|e| e.to_string())),
        Picture {
            data: drawn.shown,
            mime_type: drawn.shown_mime.into(),
        },
    )
}

/// Retire image `id`, two edits back, unless a portrait uses it. A failure is
/// logged and left to the sweep, which retires it after its hour: the edit
/// that asked has already succeeded.
async fn retire_unless_portrait(s: &Authored, id: &str) {
    if s.portrait_image_ids().await.contains(id) {
        return;
    }
    if let Err(e) = s.images.retire(id) {
        tracing::warn!("images: {id}, two edits back, could not be retired: {e}");
    }
}

/// A finished picture: what it says of itself, its file — the bytes stored —
/// and what the answer shows the model, in base64 with its type: the file
/// itself, or a smaller JPEG of it ([`super::edit_mode::preview`]).
#[derive(Debug)]
pub(super) struct Drawn {
    pub said: String,
    pub file: Vec<u8>,
    pub shown: String,
    pub shown_mime: &'static str,
}

/// The answer's text: the draw's own line, then its link and how long the
/// picture lives — or, when the file could not be stored, that there is none
/// and why.
///
/// **It says that a change is an edit of this picture**, at the moment the
/// next request is read: a model holding only the tool descriptions answers
/// "make it a black cat" with a new draw — another cat, another room.
fn said(drawn: &str, link: Result<(String, Kept), String>) -> String {
    match link {
        Ok((url, Kept::Edited)) => format!(
            "{drawn} A new image, kept for 1 hour, or until the picture edited from it is \
             edited in turn. The image it was edited from stays until this one is edited — edit \
             from that to undo this step. {EDIT_IT} The file, for 1 hour and without signing \
             in: {url}"
        ),
        Ok((url, _)) => {
            format!("{drawn} {EDIT_IT} The file, for 24 hours and without signing in: {url}")
        }
        Err(e) => format!("{drawn} There is no link to the file: it could not be stored ({e})."),
    }
}

/// What a stored picture's answer says to do with a change asked of it.
const EDIT_IT: &str = "When the user asks to change this picture, edit it: pass this link \
                       to image_edit as `image`, rather than drawing a new one.";

/// The `format` argument.
pub(super) fn format(args: &Value) -> Result<Format, String> {
    match args.get("format").and_then(Value::as_str).map(str::trim) {
        None | Some("") | Some("jpeg") | Some("jpg") => Ok(Format::Jpeg),
        Some("png") => Ok(Format::Png),
        Some(other) => Err(format!("`format` is jpeg or png, not `{other}`")),
    }
}

/// The route's PNG, base64, as the file `format` asks for, with its type.
fn file(png_base64: &str, format: Format) -> Result<(Vec<u8>, &'static str), String> {
    let png = BASE64
        .decode(png_base64.as_bytes())
        .map_err(|e| format!("the picture is not base64: {e}"))?;
    if format == Format::Png {
        return Ok((png, "image/png"));
    }
    let rgb = image::load_from_memory_with_format(&png, ImageFormat::Png)
        .map_err(|e| format!("the picture would not decode: {e}"))?
        .to_rgb8();
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY)
        .encode_image(&rgb)
        .map_err(|e| format!("the picture would not encode: {e}"))?;
    Ok((jpeg, "image/jpeg"))
}

/// The route's body from the tool's arguments.
pub(super) fn body(args: &Value) -> Result<ImageBody, String> {
    let prompt = string(args, "prompt")?.to_string();
    let side = |key: &str| -> Result<u32, String> {
        match args.get(key) {
            None | Some(Value::Null) => Ok(DRAW_DEFAULT_SIDE),
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

/// The draw the stream's last line carries, its file as `format`, said with
/// `verb` ("Drew", "Redrew"), or what went wrong.
fn outcome(stream: &[u8], format: Format, verb: &str) -> Result<Drawn, String> {
    let last = String::from_utf8_lossy(stream)
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l).ok());
    let Some(last) = last else {
        return Err("the draw ended without an answer".into());
    };
    match last["event"].as_str() {
        Some("done") => {
            let png = last["png_base64"]
                .as_str()
                .ok_or("the draw finished without a picture")?;
            let (file, mime) = file(png, format)?;
            // The seed it drew with, kept in the file, so an edit of it starts
            // from the same noise (`crate::image_seed`).
            let file = match last["seed"].as_u64() {
                Some(seed) => image_seed::stamp(&file, seed),
                None => file,
            };
            // A draw is within a megapixel and shown as itself; a PNG is the
            // route's own, already in base64.
            let shown = match format {
                Format::Png => png.to_string(),
                Format::Jpeg => BASE64.encode(&file),
            };
            Ok(Drawn {
                said: format!(
                    "{verb} a {}x{} image, seed {}.",
                    last["width"], last["height"], last["seed"]
                ),
                file,
                shown,
                shown_mime: mime,
            })
        }
        Some("error") => {
            let retry = if last["retry"] == true {
                " (worth retrying)"
            } else {
                ""
            };
            Err(format!(
                "{}: {}{retry}",
                last["error"].as_str().unwrap_or("failed"),
                last["detail"].as_str().unwrap_or("")
            ))
        }
        _ => Err(format!("the draw ended on `{last}`")),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{DynamicImage, RgbImage};

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
    fn a_finished_draw_is_a_file_and_an_error_line_is_an_error() {
        let done = b"{\"event\":\"loading\"}\n{\"event\":\"step\",\"done\":1,\"total\":9}\n\
            {\"event\":\"done\",\"width\":512,\"height\":512,\"png_base64\":\"iVBO\",\"seed\":7}\n";
        let d = outcome(done, Format::Png, Made::Drawn.verb()).unwrap();
        assert_eq!(d.said, "Drew a 512x512 image, seed 7.");
        assert_eq!(
            (d.file.as_slice(), d.shown.as_str(), d.shown_mime),
            (&[0x89, 0x50, 0x4E][..], "iVBO", "image/png")
        );
        let edited = Made::Edited { from: "x" }.verb();
        assert_eq!(
            outcome(done, Format::Png, edited).unwrap().said,
            "Redrew a 512x512 image, seed 7.",
            "an edit's answer says it drew again"
        );

        // A real picture keeps the seed it was drawn with, in either format.
        let mut png = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::new(8, 8))
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();
        let real = format!(
            "{{\"event\":\"done\",\"width\":8,\"height\":8,\"png_base64\":\"{}\",\"seed\":4321}}\n",
            BASE64.encode(&png)
        );
        for format in [Format::Png, Format::Jpeg] {
            let d = outcome(real.as_bytes(), format, "Drew").unwrap();
            assert_eq!(image_seed::read(&d.file), Some(4321), "{format:?}");
        }

        let failed =
            b"{\"event\":\"error\",\"error\":\"no_room\",\"detail\":\"full\",\"retry\":true}\n";
        assert_eq!(
            outcome(failed, Format::Jpeg, "Drew").unwrap_err(),
            "no_room: full (worth retrying)"
        );
        assert!(outcome(b"", Format::Jpeg, "Drew").is_err());
    }

    /// **The answer names the link and how long it works**, or says there is
    /// none and why.
    #[test]
    fn the_answer_carries_the_files_link_or_why_there_is_none() {
        assert_eq!(
            said(
                "Drew a 512x512 image, seed 7.",
                Ok((
                    "https://bot.tokera.com/v1/image/link/ab.jpg?exp=9&sig=cd".into(),
                    Kept::Drawn
                ))
            ),
            "Drew a 512x512 image, seed 7. When the user asks to change this picture, edit it: \
             pass this link to image_edit as `image`, rather than drawing a new one. The file, \
             for 24 hours and without signing in: \
             https://bot.tokera.com/v1/image/link/ab.jpg?exp=9&sig=cd"
        );
        assert_eq!(
            said("Drew it.", Ok(("U".into(), Kept::Edited))),
            "Drew it. A new image, kept for 1 hour, or until the picture edited from it is \
             edited in turn. The image it was edited from stays until this one is edited — edit \
             from that to undo this step. When the user asks to change this picture, edit it: \
             pass this link to image_edit as `image`, rather than drawing a new one. The file, \
             for 1 hour and without signing in: U"
        );
        assert_eq!(
            said("Drew it.", Err("no such image".into())),
            "Drew it. There is no link to the file: it could not be stored (no such image)."
        );
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
        let (same, mime) = file(&BASE64.encode(&png), Format::Png).unwrap();
        assert_eq!((same, mime), (png.clone(), "image/png"));
        let (jpeg, mime) = file(&BASE64.encode(&png), Format::Jpeg).unwrap();
        assert_eq!(mime, "image/jpeg");
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "not a JPEG");
        let back = image::load_from_memory_with_format(&jpeg, ImageFormat::Jpeg).unwrap();
        assert_eq!((back.width(), back.height()), (32, 16));
    }
}
