//! The cast written out.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::persistence::blob::hex;
use crate::persistence::record::NpcPayload;

use super::format::{Header, Line};
use super::portrait_blob;

/// The export of `npcs` — every character's current record, tombstoned ones
/// included, so their ids stay taken — taken at `taken_at_ms`: the header,
/// each character in id order, then each portrait a character names that
/// `blob` finds, in the order of its name. Each line ends in a newline.
pub fn export(
    mut npcs: Vec<NpcPayload>,
    blob: impl Fn(&[u8; 32]) -> anyhow::Result<Option<Vec<u8>>>,
    taken_at_ms: u64,
) -> anyhow::Result<String> {
    npcs.sort_by_key(|n| n.npc_id);
    let mut names: Vec<[u8; 32]> = npcs
        .iter()
        .filter_map(|n| n.portrait_image_id.as_deref().and_then(portrait_blob))
        .collect();
    names.sort();
    names.dedup();
    let mut out = serde_json::to_string(&Header::new(taken_at_ms))?;
    out.push('\n');
    for npc in npcs {
        out.push_str(&serde_json::to_string(&Line::Npc { npc })?);
        out.push('\n');
    }
    for name in names {
        let Some(bytes) = blob(&name)? else {
            continue;
        };
        out.push_str(&serde_json::to_string(&Line::Blob {
            blob: hex(&name),
            bytes_b64: STANDARD.encode(bytes),
        })?);
        out.push('\n');
    }
    Ok(out)
}
