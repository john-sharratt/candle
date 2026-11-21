//! Tests and times the incremental compaction of a single segment.
//!
//! An ignored benchmark times each phase of relocating the live half of a
//! segment. The other tests check that a fully superseded segment is dropped,
//! and that a partly live one is compacted with its live record still readable
//! after a reload.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::streams::StreamId;

    use crate::persistence::SubstratePersistence;

    use crate::substrate::Substrate;

    use super::super::tests_support::tests::{
        chunk_payload, filled_payload, mb_per_s, sealed_log, tmp_dir,
    };

    /// Performance benchmark for the incremental **compact** — the expensive
    /// maintenance op (relocation I/O). Builds a realistically-sized sealed
    /// segment (half live, half superseded), then times each phase of relocating
    /// its live half into the active. Warm-cache numbers, so they isolate the
    /// *code* cost (per-record syscalls + decode/re-encode + append), which is
    /// what "as fast as possible" targets — cold-disk throughput is disk-bound.
    ///
    /// Run: `cargo test -p candle-conversation --release --lib
    /// maintenance_perf_compact -- --ignored --nocapture`
    #[test]
    #[ignore = "perf benchmark — run explicitly with --ignored --nocapture (prefer --release)"]
    fn maintenance_perf_compact() {
        use std::time::Instant;

        // Scale knobs — ~16 KB payloads (representative KV chunk), half dead.
        const LIVE: u64 = 8_000;
        const DEAD: u64 = 8_000;
        const KV_BYTES: usize = 16 * 1024;

        let dir = tmp_dir("perf_compact");
        let sid = StreamId(1);

        let t_build = Instant::now();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // Seg 1: LIVE + DEAD chunks.
            for i in 0..(LIVE + DEAD) {
                sp.write_chunk(sid, i, 32, 4, None, &filled_payload(i as u32, KV_BYTES))
                    .unwrap();
            }
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 sealed
                                       // Active (seg 2) supersedes the DEAD half.
            for i in LIVE..(LIVE + DEAD) {
                sp.write_chunk(sid, i, 32, 4, None, &filled_payload(i as u32 + 1, KV_BYTES))
                    .unwrap();
            }
            sp.commit().unwrap();
        }
        let seg1_total = std::fs::metadata(sealed_log(&dir, 1)).unwrap().len();
        eprintln!(
            "\n── compact benchmark ─────────────────────────────────\nbuild: {:?}  |  seg-1 on disk: {} MB ({} chunks, {}/{} live)",
            t_build.elapsed(),
            seg1_total >> 20,
            LIVE + DEAD,
            LIVE,
            LIVE + DEAD,
        );

        // Reopen to rebuild the segmented index.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let relocated = seg1_total / 2; // ≈ the live half

        // Phase 1 — plan (snapshot).
        let t = Instant::now();
        let (chunk_relocs, _, _, _, _) = sp
            .live_indexes()
            .gather_relocations(&substrate, &[SegmentId(1)]);
        let plan_el = t.elapsed();

        // Reference: the naive per-record read path (`read_chunk` decodes +
        // CRC-verifies each record). The compact below uses coalesced reads +
        // verbatim staging instead, so it should beat even this read-only loop.
        let t = Instant::now();
        let mut read_bytes = 0u64;
        for &(_, idx, _) in &chunk_relocs {
            let p = sp.read_chunk(&substrate, sid, idx).unwrap();
            read_bytes += p.kv_bytes.len() as u64;
        }
        let read_el = t.elapsed();

        // Full compact (execute I/O + index apply + drop).
        let t = Instant::now();
        sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
            .unwrap();
        let compact_el = t.elapsed();

        eprintln!(
            "plan (gather {} relocs): {plan_el:?}\n\
             per-record read reference ({} live chunks, {} MB): {read_el:?}  →  {:.0} MB/s\n\
             full compact (coalesced read + verbatim stage + fsync + drop, ~{} MB): {compact_el:?}  →  {:.0} MB/s",
            chunk_relocs.len(),
            chunk_relocs.len(),
            read_bytes >> 20,
            mb_per_s(read_bytes, read_el),
            relocated >> 20,
            mb_per_s(relocated, compact_el),
        );
        assert!(!sealed_log(&dir, 1).exists(), "seg 1 compacted away");
        // Sanity: the relocated live chunks read back correctly.
        assert_eq!(
            sp.read_chunk(&substrate, sid, 0).unwrap().kv_bytes.len(),
            KV_BYTES
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A sealed segment whose only read-back record is superseded by a later
    /// segment is fully dead — maintenance drops it (O(1) unlink), and the live
    /// version still reads correctly and survives a reload.
    #[test]
    fn maintenance_drops_a_fully_superseded_segment() {
        let dir = tmp_dir("drop");
        let sid = StreamId(101);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 sealed, holds chunk-0 v1
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(2))
                .unwrap(); // seg 2 supersedes
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&0].segment,
                SegmentId(2),
                "the higher-id (newer) copy wins"
            );
            assert!(sealed_log(&dir, 1).exists());
            assert!(
                sp.run_maintenance(&mut substrate).unwrap().is_some(),
                "seg 1 is fully dead"
            );
            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was dropped");
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), chunk_payload(2));
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), chunk_payload(2));
            assert_eq!(substrate.live_chunk_count(), 1);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A partially-dead sealed segment is compacted: its live record is
    /// relocated into the active (the index repointed), then it is dropped.
    /// Both chunks read correctly afterwards and survive a reload.
    #[test]
    fn maintenance_compacts_and_relocates_a_live_record() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("compact");
        // The stream needs a decl to be live — a decl-less stream is an orphan
        // whose chunks are (correctly) reclaimed rather than relocated.
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 202,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 2,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(10))
                .unwrap(); // seg 1, live
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(11))
                .unwrap(); // seg 1, will be superseded
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(12))
                .unwrap(); // seg 2 supersedes chunk-1
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&0].segment,
                SegmentId(1)
            );
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&1].segment,
                SegmentId(2)
            );

            sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
                .unwrap();

            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was compacted away");
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&0].segment,
                SegmentId(2),
                "chunk-0 was relocated into the active segment"
            );
            assert_eq!(
                sp.read_chunk(&substrate, sid, 0).unwrap(),
                chunk_payload(10)
            );
            assert_eq!(
                sp.read_chunk(&substrate, sid, 1).unwrap(),
                chunk_payload(12)
            );
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                sp.read_chunk(&substrate, sid, 0).unwrap(),
                chunk_payload(10)
            );
            assert_eq!(
                sp.read_chunk(&substrate, sid, 1).unwrap(),
                chunk_payload(12)
            );
            assert_eq!(substrate.live_chunk_count(), 2);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
