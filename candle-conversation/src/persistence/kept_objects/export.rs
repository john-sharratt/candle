//! A substrate's kept objects as one export.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::persistence::blob::hex;
use crate::persistence::identity::RecordIdentity;
use crate::persistence::record::CustomObjectPayload;

use super::classify::is_kept;
use super::format::{Header, Line, Stamp};
use super::names::named_blobs;

/// The export of `held` — every custom object a substrate holds, tombstones
/// included, with the record behind each — taken by `daemon` at
/// `taken_at_ms`: the header, then each kept object or tombstone in the order
/// of its stamp, then each blob a kept live object names that `blob` finds, in
/// the order of its name. Each line ends in a newline.
pub fn export(
    held: Vec<(CustomObjectPayload, RecordIdentity)>,
    blob: impl Fn(&[u8; 32]) -> anyhow::Result<Option<Vec<u8>>>,
    daemon: &str,
    taken_at_ms: u64,
) -> anyhow::Result<String> {
    let mut kept: Vec<(CustomObjectPayload, Stamp)> = held
        .into_iter()
        .filter(|(obj, _)| is_kept(&obj.key))
        .map(|(obj, id)| (obj, Stamp::from(id.stamp())))
        .collect();
    kept.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.key.cmp(&b.0.key)));
    let mut names: Vec<[u8; 32]> = kept
        .iter()
        .filter(|(obj, _)| !obj.ceased)
        .flat_map(|(obj, _)| named_blobs(obj))
        .collect();
    names.sort();
    names.dedup();

    let mut out = serde_json::to_string(&Header::new(daemon, taken_at_ms))?;
    out.push('\n');
    for (object, stamp) in kept {
        out.push_str(&serde_json::to_string(&Line::Object { object, stamp })?);
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
