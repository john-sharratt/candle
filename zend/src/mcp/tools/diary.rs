//! What the `diary_*` tools share (`docs/private_mode.md` §5.4): the line
//! every description opens with, the conversation a call keeps its diaries
//! in, and how a passphrase is read — exactly as given, never trimmed.

use mcp_server::access::DIARIES_PRIVATE;
use serde_json::Value;

use super::Ctx;

/// What every diary tool's description says of diaries, and nothing more.
pub const DIARIES: &str =
    "Topic-based diaries, encrypted at rest, opened by passphrase, available in private mode.";

/// The private conversation a diary call is in, by its ref's key: the tool
/// table lets no other reach here (`super::permitted`).
pub fn conversation(ctx: &Ctx) -> Result<String, String> {
    match ctx.conversation {
        Some(r) if r.private => Ok(r.conversation()),
        _ => Err(DIARIES_PRIVATE.to_string()),
    }
}

/// The passphrase `args` give under `key`, exactly as given.
pub fn passphrase<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("`{key}` is missing: the diary passphrase"))
}
