//! `image_generate {prompt, width?, height?, steps?, seed?, shift?}`: draw a
//! picture — `POST /v1/image/generate`, the console's own draw.
//!
//! The route streams NDJSON (`loading`, a `step` per unit, then `done` with
//! the PNG); the tool reads the stream to its end and answers with the last
//! line. The picture comes back as an MCP image block beside a line of text,
//! so the model sees what it drew rather than a base64 string it cannot read.
//!
//! The draw takes the route's compliance gate and its standing checkpoint; the
//! admin-only `restricted` one is not offered here.

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use candle_conversation::guest::{ImageLora, DEFAULT_REFERENCE_HOLD};
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

/// The most of the route's stream read back: a 1024² PNG in base64 with room.
const STREAM_LIMIT: usize = 16 << 20;

impl Tool for Generate {
    const NAME: &'static str = "image_generate";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Generate an image from a text prompt (Z-Image-Turbo) and return it as a \
                PNG. Describe the subject, style, lighting and composition in plain sentences; \
                variety comes from the prompt, not the seed. A draw takes about 10-60 seconds. \
                Prompts are checked first and a declined one is refused.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What to draw."},
                    "width": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "default": 512, "description": "Width in pixels, a multiple of 16."},
                    "height": {"type": "integer", "minimum": 16, "maximum": MAX_SIDE, "multipleOf": 16, "default": 512, "description": "Height in pixels, a multiple of 16."},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 24, "description": "Denoise steps: more buys finish, not variety; 8 is the model's trained count."},
                    "seed": {"type": "integer", "minimum": 0, "description": "Seed, for repeating a draw; random when absent."},
                    "shift": {"type": "number", "minimum": 1.0, "maximum": 8.0, "description": "The schedule's shift; the deployment's own when absent."}
                },
                "required": ["prompt"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let body = match body(args) {
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
        outcome(&bytes)
    }
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
        None | Some(Value::Null) => 24,
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

/// The answer the stream's last line carries.
fn outcome(stream: &[u8]) -> Answer {
    let last = String::from_utf8_lossy(stream)
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l).ok());
    let Some(last) = last else {
        return Answer::error("the draw ended without an answer");
    };
    match last["event"].as_str() {
        Some("done") => match last["png_base64"].as_str() {
            Some(png) => Answer::picture(
                format!(
                    "Drew a {}x{} image, seed {}.",
                    last["width"], last["height"], last["seed"]
                ),
                Picture {
                    data: png.to_string(),
                    mime_type: "image/png".into(),
                },
            ),
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
            ("a red fox", 512, 512, 24)
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
        let a = outcome(done);
        assert!(!a.is_error);
        assert_eq!(a.text, "Drew a 512x512 image, seed 7.");
        assert_eq!(a.picture.unwrap().data, "iVBO");

        let failed =
            b"{\"event\":\"error\",\"error\":\"no_room\",\"detail\":\"full\",\"retry\":true}\n";
        let a = outcome(failed);
        assert!(a.is_error);
        assert_eq!(a.text, "no_room: full (worth retrying)");
        assert!(outcome(b"").is_error);
    }
}
