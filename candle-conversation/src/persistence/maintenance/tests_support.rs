//! Fixtures shared by the maintenance tests.
//!
//! They build sealed logs with chosen payloads, list record identities on disk
//! in append order, and create segments holding section tombstones or a turn's
//! declaration. Each fixture is arranged so one segment is the one a pass will
//! choose, which lets a test control what that pass must carry.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::record::ChunkPayload;

    use crate::persistence::{Admit, SubstratePersistence, SUBSTRATE_DIR};

    use crate::substrate::Substrate;

    use std::path::PathBuf;

    pub(in super::super) fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("maint_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
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

    pub(in super::super) fn sealed_log(dir: &std::path::Path, id: u64) -> PathBuf {
        dir.join(SUBSTRATE_DIR).join(format!("seg-{id:010}.log"))
    }

    /// Encoded snapshot payload for `timeline` at `turn` with a recognisable
    /// `fill` byte — the fixture shared by the snapshot-relocation tests.
    pub(in super::super) fn snapshot_payload(timeline: u64, turn: u32, fill: u8) -> Vec<u8> {
        use crate::persistence::record::{SnapshotLayer, SnapshotPayload};
        SnapshotPayload {
            timeline_id: timeline,
            turn_index: turn,
            schedule_hash: 0xA5A5,
            layers: vec![SnapshotLayer {
                layer_index: 0,
                n_v_heads: 1,
                d_v: 4,
                d_k: 4,
                state: vec![fill; 64],
                conv_channels: 2,
                conv_tail_cols: 2,
                conv_tail: vec![fill; 16],
            }],
            // Filled, not empty: a relocation that carried the layers and
            // dropped the model's own state would still pass every assertion
            // that only inspects the layers.
            aux: vec![fill; 24],
        }
        .encode()
    }

    /// A chunk payload with a caller-sized `kv_bytes` (constant fill — cheap to
    /// build) for the perf benchmark.
    pub(in super::super) fn filled_payload(seed: u32, kv_len: usize) -> ChunkPayload {
        let mut p = chunk_payload(seed);
        p.kv_bytes = vec![(seed & 0xff) as u8; kv_len];
        p
    }

    pub(in super::super) fn mb_per_s(bytes: u64, dur: std::time::Duration) -> f64 {
        let secs = dur.as_secs_f64().max(1e-9);
        (bytes as f64 / (1024.0 * 1024.0)) / secs
    }

    /// Every record on disk in append order: its type, stream, chunk index and
    /// identity.
    pub(in super::super) fn identities_on_disk(
        dir: &std::path::Path,
    ) -> Vec<(RecordType, u64, u64, RecordIdentity)> {
        let mut seen = Vec::new();
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate_and_sink(
            dir,
            &mut substrate,
            Admit::Unbound,
            |entry| {
                let h = &entry.record.header;
                seen.push((h.record_type, h.stream_id, h.chunk_index, h.identity()));
            },
            &|_, _, _| {},
        )
        .unwrap();
        seen
    }

    /// A store whose sealed segment 1 holds `n` section tombstones and nothing
    /// else, and whose sealed segment 2 holds a live stream with a superseded
    /// chunk — dead weight an op has reason to compact. The tombstones end
    /// sections `0x5000..0x5000 + n`, written before generations.
    pub(in super::super) fn store_with_section_tombstones(tag: &str, n: u64) -> PathBuf {
        let dir = tmp_dir(tag);
        let decl = turn_decl(707, 0);
        let sid = decl.stream_id();
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        for i in 0..n {
            sp.write_section_tombstone(0x5000 + i, None, None).unwrap();
        }
        sp.commit().unwrap();
        sp.seal_active().unwrap();
        sp.declare_stream(&decl).unwrap();
        sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
            .unwrap();
        sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(2))
            .unwrap();
        sp.commit().unwrap();
        sp.seal_active().unwrap();
        dir
    }

    /// One live turn stream, for tests that need a segment holding something
    /// worth keeping.
    pub(in super::super) fn turn_decl(
        timeline_id: u64,
        turn_index: u32,
    ) -> crate::persistence::streams::StreamDecl {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        StreamDecl::Turn(TurnDecl {
            timeline_id,
            turn_index,
            turn_id_day: 0,
            turn_id_seq: turn_index + 1,
            role: 2,
            block_start: 0,
            block_end: 16,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        })
    }
}
