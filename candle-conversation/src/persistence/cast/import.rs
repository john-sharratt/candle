//! A cast export read back and written into a substrate.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::Serialize;

use crate::persistence::blob::parse_hex;
use crate::persistence::record::NpcPayload;

use super::format::{Header, Line, FORMAT, VERSION};

/// What an import wrote.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct CastImport {
    pub npcs_written: usize,
    pub blobs_written: usize,
}

/// Write the cast `text` holds: each portrait through `put_blob` first, so no
/// character names a portrait not yet held, then every character through
/// `put_npcs` in one batch, each record verbatim — its id, owner, revision and
/// last place kept.
pub fn import(
    text: &str,
    mut put_blob: impl FnMut(Vec<u8>) -> anyhow::Result<[u8; 32]>,
    put_npcs: impl FnOnce(Vec<NpcPayload>) -> anyhow::Result<()>,
) -> anyhow::Result<CastImport> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Header = match lines.next() {
        Some(l) => serde_json::from_str(l)?,
        None => anyhow::bail!("a cast export has no header"),
    };
    if header.format != FORMAT || header.version != VERSION {
        anyhow::bail!(
            "not a cast export: format {} version {}",
            header.format,
            header.version
        );
    }
    let mut npcs = Vec::new();
    let mut report = CastImport::default();
    for line in lines {
        match serde_json::from_str(line)? {
            Line::Npc { npc } => npcs.push(npc),
            Line::Blob { blob, bytes_b64 } => {
                let name = parse_hex(&blob)
                    .ok_or_else(|| anyhow::anyhow!("a portrait named {blob:?} is no SHA-256"))?;
                let written = put_blob(STANDARD.decode(bytes_b64)?)?;
                if written != name {
                    anyhow::bail!("an imported portrait's bytes do not hash to its name");
                }
                report.blobs_written += 1;
            }
        }
    }
    report.npcs_written = npcs.len();
    put_npcs(npcs)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use sha2::{Digest, Sha256};

    use super::super::{export, portrait_blob};
    use super::*;
    use crate::persistence::blob::hex;

    fn npc(id: u64, name: &str, portrait: Option<String>) -> NpcPayload {
        let mut n: NpcPayload = serde_json::from_value(serde_json::json!({
            "npc_id": id,
            "owner_id": "u_1",
            "revision": 3,
            "created_ms": 1,
            "updated_ms": 2,
            "state": "idle",
            "name": name,
            "world_id": "moon",
            "personality_id": "commander",
            "hidden": false,
            "heartbeat_ms": 1000,
            "salience_gate": 0.5,
            "tags": [],
            "persona_description": "p",
            "persona_origin": "authored",
            "portrait_image_id": null,
            "portrait_origin": null,
        }))
        .unwrap_or_else(|e| panic!("a test character: {e}"));
        n.portrait_image_id = portrait;
        n
    }

    /// **A cast exported and imported is the same cast**: every record
    /// verbatim, in id order, and each portrait a character names carried with
    /// it once, whatever how many name it.
    #[test]
    fn a_cast_round_trips_with_its_portraits() {
        let bytes = b"a portrait".to_vec();
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let id = format!("{}.png", hex(&sha));
        let cast = vec![
            npc(2, "Wick", Some(id.clone())),
            npc(1, "Zen", Some(id.clone())),
            npc(3, "Umbra", None),
        ];
        let held: BTreeMap<[u8; 32], Vec<u8>> = [(sha, bytes.clone())].into();
        let text = export(cast.clone(), |n| Ok(held.get(n).cloned()), 5).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 5, "{text}");
        assert_eq!(
            lines[0],
            r#"{"format":"npc-cast","version":1,"taken_at_ms":5}"#
        );
        assert!(lines[1].contains(r#""name":"Zen""#), "id order");
        assert_eq!(
            lines[4],
            format!(
                r#"{{"blob":"{}","bytes_b64":"YSBwb3J0cmFpdA=="}}"#,
                hex(&sha)
            )
        );

        let blobs = RefCell::new(BTreeMap::new());
        let written = RefCell::new(Vec::new());
        let report = import(
            &text,
            |b| {
                let name: [u8; 32] = Sha256::digest(&b).into();
                blobs.borrow_mut().insert(name, b);
                Ok(name)
            },
            |n| {
                *written.borrow_mut() = n;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            report,
            CastImport {
                npcs_written: 3,
                blobs_written: 1
            }
        );
        let mut want = cast;
        want.sort_by_key(|n| n.npc_id);
        assert_eq!(*written.borrow(), want);
        assert_eq!(blobs.borrow()[&sha], bytes);
    }

    /// A portrait whose bytes do not hash to its name, and a file that is not
    /// a cast export, are refused.
    #[test]
    fn a_wrong_portrait_or_another_format_is_refused() {
        let other = r#"{"format":"substrate-objects","version":1,"taken_at_ms":1}"#;
        assert!(import(other, |_| Ok([0; 32]), |_| Ok(())).is_err());
        let lying = format!(
            "{}\n{{\"blob\":\"{}\",\"bytes_b64\":\"AQI=\"}}\n",
            r#"{"format":"npc-cast","version":1,"taken_at_ms":1}"#,
            hex(&[7; 32])
        );
        assert!(import(&lying, |_| Ok([1; 32]), |_| Ok(())).is_err());
    }

    #[test]
    fn a_portrait_id_names_its_blob() {
        let sha = [0xab; 32];
        assert_eq!(portrait_blob(&format!("{}.webp", hex(&sha))), Some(sha));
        assert_eq!(portrait_blob("not-an-image"), None);
        assert_eq!(portrait_blob("abc.png"), None);
    }
}
