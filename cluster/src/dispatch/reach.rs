//! What a route asks of the member that serves it beyond its dispatch (§5.2,
//! `Reach`), and the size of a draw read from its body.
//!
//! **A draw goes to a member whose card reaches it.** Each npcd member draws up
//! to the pixel area its card's row allows (`candle_conversation::guest::reach`)
//! and says so in `CAPACITY` (`guest_reach`, the side of that square). A route
//! declared [`Reach::Draw`] has its body read for the draw's `width × height`,
//! and the gateway places it only among the members whose square holds that
//! area — the strongest not drawing, and in turn the rest. When none does it
//! places it among those that reach furthest, and the member's own refusal
//! names the largest draw the cluster serves.
//!
//! The size is read where the member reads it: a JSON-RPC `tools/call` of a
//! draw tool carries it among its `arguments`, a draw route's own body at its
//! top level; any other JSON-RPC message asks for no draw. A side not given is the draw's default, [`DRAW_DEFAULT_SIDE`].
//! An edit's size follows its picture when a side is missing, which only the
//! member holding the picture knows, so an edit is sized here only when it
//! names both sides; otherwise it is sized 0 — every card reaches it, and it
//! still counts as a draw on the card it goes to — and its member fits it to
//! its own reach.

use serde_json::Value;

/// What a route needs of its member besides its dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reach {
    /// Any member its dispatch picks.
    #[default]
    Any,
    /// A member whose card draws the body's `width × height`.
    Draw,
}

/// A side a draw leaves out: what npcd draws when a caller names none.
pub const DRAW_DEFAULT_SIDE: u32 = 512;

/// The MCP tool that draws from nothing.
pub const MCP_DRAW: &str = "image_generate";

/// The MCP tool that draws from a picture, in its `prompt` mode.
pub const MCP_EDIT: &str = "image_edit";

/// The pixel area the draw in `body` asks for; `None` when it draws nothing —
/// an MCP call of any other tool, an edit that only resizes, crops, adjusts or
/// turns, or another message. An edit its member sizes — one that does not name
/// both sides, or one that takes the background away — is 0: it reaches every
/// card, and is still a draw on the card it goes to.
pub fn draw_area(body: &Value) -> Option<u64> {
    if body.get("jsonrpc").is_none() {
        return area(body, DRAW_DEFAULT_SIDE);
    }
    if body.get("method").and_then(Value::as_str) != Some("tools/call") {
        return None;
    }
    let params = body.get("params")?;
    let args = params.get("arguments")?;
    match params.get("name").and_then(Value::as_str)? {
        MCP_DRAW => area(args, DRAW_DEFAULT_SIDE),
        MCP_EDIT => match args.get("mode").and_then(Value::as_str) {
            None | Some("prompt") => {
                let both = side(args, "width").is_some() && side(args, "height").is_some();
                if both {
                    area(args, 0)
                } else {
                    Some(0)
                }
            }
            Some("remove_background") => Some(0),
            Some(_) => None,
        },
        _ => None,
    }
}

/// `width × height` as `fields` names them, a side not named being `default`.
fn area(fields: &Value, default: u32) -> Option<u64> {
    let w = side(fields, "width").unwrap_or(u64::from(default));
    let h = side(fields, "height").unwrap_or(u64::from(default));
    Some(w * h)
}

fn side(fields: &Value, name: &str) -> Option<u64> {
    fields.get(name).and_then(Value::as_u64)
}

/// Whether a member whose square side is `side` draws `area` pixels.
pub fn reaches(side: u32, area: u64) -> bool {
    u64::from(side) * u64::from(side) >= area
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn call(name: &str, args: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": args}})
    }

    /// **A draw route's body is sized at its top level**, a side it leaves out
    /// being the default.
    #[test]
    fn a_draw_route_is_sized_from_its_body() {
        assert_eq!(
            draw_area(&json!({"prompt": "p", "width": 2048, "height": 1024})),
            Some(2_097_152)
        );
        assert_eq!(
            draw_area(&json!({"prompt": "p", "width": 2048})),
            Some(1_048_576)
        );
        assert_eq!(draw_area(&json!({"prompt": "p"})), Some(262_144));
    }

    /// **An MCP draw is sized from its arguments; an edit that draws by both
    /// sides it names, or 0 when its member sizes it; an edit that only
    /// reshapes, and any other tool, not at all.**
    #[test]
    fn an_mcp_call_is_sized_only_when_it_draws() {
        assert_eq!(
            draw_area(&call(MCP_DRAW, json!({"prompt": "p", "width": 1024}))),
            Some(524_288)
        );
        assert_eq!(
            draw_area(&call(MCP_DRAW, json!({"prompt": "p"}))),
            Some(262_144)
        );
        let edit = |args| draw_area(&call(MCP_EDIT, args));
        assert_eq!(
            edit(json!({"image": "x", "width": 1024, "height": 768})),
            Some(786_432)
        );
        assert_eq!(
            edit(json!({"image": "x", "mode": "prompt", "width": 64, "height": 32})),
            Some(2_048)
        );
        assert_eq!(edit(json!({"image": "x", "width": 1024})), Some(0));
        assert_eq!(edit(json!({"image": "x", "prompt": "p"})), Some(0));
        assert_eq!(
            edit(json!({"image": "x", "mode": "remove_background"})),
            Some(0)
        );
        assert_eq!(
            edit(json!({"image": "x", "mode": "resize", "width": 4096, "height": 4096})),
            None
        );
        assert_eq!(
            draw_area(&call("bot_mission_read", json!({"bot": "b"}))),
            None
        );
        assert_eq!(
            draw_area(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})),
            None
        );
    }

    /// The reach is an area: a square of `side` holds any shape of as many
    /// pixels, and no more.
    #[test]
    fn a_side_reaches_its_square() {
        assert!(reaches(1024, 1024 * 1024));
        assert!(reaches(1024, 768 * 1344));
        assert!(!reaches(1024, 1024 * 1040));
        assert!(reaches(2048, 2048 * 2048));
        assert!(!reaches(0, 1));
    }
}
