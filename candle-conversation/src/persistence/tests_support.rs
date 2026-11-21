//! Shared helpers for the persistence tests: a character record, a directory fingerprint and model keys.
//!
//! The fingerprint lists every entry under a directory with its length and digest, so the read-only tests can show that a store is untouched, including any directory a write would have created. A model key is described by its file name and its weights, so two keys with the same weights are one model.

use super::*;

/// A plain character record, for the tests in this module and in
/// [`maintenance`] that need one to survive something.
#[cfg(test)]
pub(crate) fn test_npc(id: u64, name: &str) -> NpcPayload {
    NpcPayload {
        npc_id: id,
        owner_id: "u1".to_string(),
        revision: 1,
        created_ms: 1_000,
        updated_ms: 1_000,
        state: "active".to_string(),
        name: name.to_string(),
        world_id: "battle-cities".to_string(),
        personality_id: "maker".to_string(),
        hidden: false,
        heartbeat_ms: 60_000,
        salience_gate: 0.5,
        tags: Vec::new(),
        persona_description: String::new(),
        persona_origin: "authored".to_string(),
        portrait_image_id: None,
        portrait_origin: None,
        at: None,
        mood: None,
        suspended_from: None,
        beliefs: Vec::new(),
        relationships: Vec::new(),
        agency: Vec::new(),
        modulation: record::Modulation::default(),
    }
}

/// Every entry under `dir`, recursively, as `(relative path, length,
/// SHA-256)` sorted by path — the before/after the read-only tests hold a
/// store to. A directory appears as its path with a trailing `/`, length 0 and
/// a zero digest, so a created directory shows up as well as a created file.
#[cfg(test)]
pub(crate) fn dir_fingerprint(dir: &Path) -> Vec<(String, u64, [u8; 32])> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, u64, [u8; 32])>) {
        for entry in std::fs::read_dir(dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            let rel = path
                .strip_prefix(root)
                .expect("entry under the root")
                .to_string_lossy()
                .replace('\\', "/");
            if path.is_dir() {
                out.push((format!("{rel}/"), 0, [0u8; 32]));
                walk(root, &path, out);
            } else {
                let bytes = std::fs::read(&path).expect("read file");
                out.push((rel, bytes.len() as u64, sha256(&bytes)));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::identity::{ModelKey, RecordedModel};

    use streams::SectionDecl;

    pub(in super::super) fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_sp_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// A model key described as `file`, whose weights are `weights` repeated —
    /// so two keys with the same `weights` are one model, whatever `file` says.
    pub(in super::super) fn model(file: &str, weights: u8) -> ModelKey {
        ModelKey {
            arch: "Qwen3".into(),
            repo: "test/repo".into(),
            file: file.into(),
            weights: [weights; 32],
            tokenizer: [0; 32],
            rope: "file-stated".into(),
        }
    }

    pub(in super::super) fn recorded(sp: &SubstratePersistence) -> Option<RecordedModel> {
        sp.recorded_model().map(|r| r.unwrap())
    }

    pub(in super::super) fn section(name: &str, prefix: u64) -> StreamDecl {
        StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: content_hash::ContentHash { lo: prefix, hi: 0 },
                section_hash: content_hash::hash_bytes(name.as_bytes()),
            },
            debug_name: name.to_string(),
        })
    }

    /// Every character record in the store, by id, as a fresh open sees them.
    pub(in super::super) fn cast_on_disk(dir: &Path) -> Vec<u64> {
        let mut seen = Vec::new();
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate_and_sink(
            dir,
            &mut substrate,
            Admit::Unbound,
            |entry| {
                if entry.record.header.record_type == RecordType::Npc {
                    seen.push(entry.record.header.stream_id);
                }
            },
            &|_, _, _| {},
        )
        .unwrap();
        seen.sort_unstable();
        seen.dedup();
        seen
    }

    pub(in super::super) fn chunk_payload(seed: u32) -> ChunkPayload {
        ChunkPayload {
            offset: seed as u16,
            k_formats: vec![4, 4, 4, 4],
            v_formats: vec![5, 5, 5, 5],
            k_pal: vec![seed as u8; 4],
            v_pal: vec![(seed + 1) as u8; 2],
            k_scale: vec![seed as f32, seed as f32 * 0.5],
            v_scale: vec![seed as f32 + 0.25],
            kv_bytes: (0..512u32).map(|i| ((i + seed * 13) % 256) as u8).collect(),
        }
    }
}
