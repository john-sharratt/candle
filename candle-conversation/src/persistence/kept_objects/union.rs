//! Several members' exports as one set of winners.
//!
//! Each member's export holds what that member had; an object one member held
//! that had not reached the others is in its export alone. The union takes
//! every key's line with the greatest original stamp, a tombstone competing
//! like any write, so a key deleted on one member before its tombstone reached
//! another stays deleted. A key whose winner is a tombstone is not written.

use std::collections::HashMap;

use anyhow::{bail, Context};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::persistence::blob::parse_hex;
use crate::persistence::record::CustomObjectPayload;

use super::format::{Header, Line, Stamp, FORMAT, VERSION};
use super::names::named_blobs;

/// What an import writes: the live winners in the order of their original
/// stamps, and the blobs they name that some export carried.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Union {
    pub objects: Vec<CustomObjectPayload>,
    pub blobs: Vec<([u8; 32], Vec<u8>)>,
    /// Keys whose winner was a tombstone, and so are not written.
    pub ceased: usize,
}

/// The exports in `text`, one after another: each begins at its header line.
fn exports_in(text: &str) -> anyhow::Result<Vec<(Header, Vec<&str>)>> {
    let mut out: Vec<(Header, Vec<&str>)> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(header) = serde_json::from_str::<Header>(line) {
            out.push((header, Vec::new()));
            continue;
        }
        match out.last_mut() {
            Some((_, lines)) => lines.push(line),
            None => bail!("line {}: an export begins with its header", n + 1),
        }
    }
    Ok(out)
}

/// The union of the exports `text` holds, one after another — one from each
/// member — each refused unless it is a `daemon` export of this format.
pub fn union(text: &str, daemon: &str) -> anyhow::Result<Union> {
    let mut winners: HashMap<String, (Stamp, CustomObjectPayload)> = HashMap::new();
    let mut blobs: HashMap<[u8; 32], Vec<u8>> = HashMap::new();
    let exports = exports_in(text)?;
    if exports.is_empty() {
        bail!("no export to import");
    }
    for (e, (header, lines)) in exports.into_iter().enumerate() {
        if header.format != FORMAT || header.version != VERSION {
            bail!(
                "export {e} is {} version {}, not {FORMAT} version {VERSION}",
                header.format,
                header.version
            );
        }
        if header.daemon != daemon {
            bail!("export {e} is {}'s, not {daemon}'s", header.daemon);
        }
        for (n, line) in lines.into_iter().enumerate() {
            let parsed: Line = serde_json::from_str(line)
                .with_context(|| format!("export {e}, line {}: not an object or a blob", n + 2))?;
            match parsed {
                Line::Object { object, stamp } => {
                    let wins = winners.get(&object.key).is_none_or(|(s, _)| stamp > *s);
                    if wins {
                        winners.insert(object.key.clone(), (stamp, object));
                    }
                }
                Line::Blob { blob, bytes_b64 } => {
                    let name = parse_hex(&blob)
                        .with_context(|| format!("export {e}: `{blob}` names no blob"))?;
                    let bytes = STANDARD
                        .decode(bytes_b64)
                        .with_context(|| format!("export {e}: blob {blob} is not base64"))?;
                    blobs.insert(name, bytes);
                }
            }
        }
    }
    let mut live: Vec<(Stamp, CustomObjectPayload)> = Vec::new();
    let mut ceased = 0;
    for (_, (stamp, obj)) in winners {
        if obj.ceased {
            ceased += 1;
        } else {
            live.push((stamp, obj));
        }
    }
    live.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.key.cmp(&b.1.key)));
    let mut names: Vec<[u8; 32]> = live.iter().flat_map(|(_, o)| named_blobs(o)).collect();
    names.sort();
    names.dedup();
    Ok(Union {
        blobs: names
            .into_iter()
            .filter_map(|n| blobs.remove(&n).map(|b| (n, b)))
            .collect(),
        objects: live.into_iter().map(|(_, o)| o).collect(),
        ceased,
    })
}
