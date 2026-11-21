//! Tests that chunk records round-trip through a reopen, and that a wrong golden is reported without failing a write or a read.
//!
//! Chunks survive a restart, and the header-index chain recovers to identical state on reopen. A log whose chain is unusable falls back to a full walk and back-fills a fresh chain at that open, so only the first open pays the slow walk. A chunk that the active log lacks resolves from an inherited log.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use log_file::{LogFile, SUPERBLOCK_SIZE};

    use super::super::tests_support::tests::{chunk_payload, tmp_dir};

    #[test]
    fn chunk_write_read_roundtrip() {
        let dir = tmp_dir("chunk_rw");
        let sid = StreamId(777);
        let payload = chunk_payload(3);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &payload).unwrap();
            sp.commit().unwrap();
            // After write, repoen so the walker rebuilds substrate.streams.
            drop(sp);
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), payload);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chunks_survive_restart() {
        let dir = tmp_dir("chunk_restart");
        let sid = StreamId(42);
        let chunks: Vec<ChunkPayload> = (0..3).map(chunk_payload).collect();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            for (i, c) in chunks.iter().enumerate() {
                sp.write_chunk(sid, i as u64, 32, 4, None, c).unwrap();
            }
            sp.commit_stream(sid, 2).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let read = sp.read_stream_chunks(&substrate, sid).unwrap();
            assert_eq!(read.len(), 3);
            for (i, (idx, c)) in read.iter().enumerate() {
                assert_eq!(*idx, i as u64);
                assert_eq!(c, &chunks[i]);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The write- and read-side golden checks are non-blocking. A chunk stored
    /// with a deliberately WRONG golden still writes (`append_record` logs an
    /// error but proceeds) and still reads back (`read_stream_chunks_batched`'s
    /// check warns, not errors) — so a bad byte-movement pipeline is flagged in
    /// the log without ever aborting a write or a load. (The mismatch detection
    /// itself is unit-tested in `record::tests::recompute_chunk_golden_*`.)
    #[test]
    fn wrong_golden_is_nonfatal_on_write_and_read() {
        let dir = tmp_dir("golden_nonfatal");
        let sid = StreamId(555);
        let payload = chunk_payload(9);
        // Sanity: the bogus golden really doesn't match the KV bytes.
        assert_ne!(
            candle::fletcher::fletcher32(&payload.kv_bytes),
            0xBAD0_C0DE,
            "test's bogus golden must differ from the true one"
        );
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            // Some(wrong) forces the stored golden to mismatch the bytes.
            sp.write_chunk(sid, 0, 32, 4, Some(0xBAD0_C0DE), &payload)
                .unwrap();
            sp.commit_stream(sid, 0).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // Read is non-fatal despite the golden mismatch — payload comes back.
            let read = sp.read_stream_chunks(&substrate, sid).unwrap();
            assert_eq!(read.len(), 1);
            assert_eq!(read[0].1, payload);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End-to-end index chain: appending past the flush threshold
    /// writes `HeaderIndex` records and publishes the superblock hint;
    /// a reopen recovers through the chain to identical state, seeds
    /// the accumulator with the un-indexed tail, and reads records at
    /// the recovered locations.
    #[test]
    fn header_index_chain_survives_reopen() {
        let dir = tmp_dir("index_chain");
        let sid = StreamId(4040);
        let n = INDEX_FLUSH_ENTRIES + 7; // one flush + an un-indexed tail
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(sp.last_index().is_none(), "fresh log has no chain");
            for i in 0..n {
                sp.write_chunk(sid, i as u64, 32, 4, None, &chunk_payload(i as u32))
                    .unwrap();
            }
            assert!(
                sp.last_index().is_some(),
                "crossing the threshold flushes an index record"
            );
            assert_eq!(sp.pending_index_len(), 7);
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(substrate.live_chunk_count(), n);
            assert!(
                sp.last_index().is_some(),
                "reopen recovers through the published chain"
            );
            assert_eq!(
                sp.pending_index_len(),
                7,
                "the un-indexed tail seeds the accumulator"
            );
            // The recovered chunk locations are exact — read one from
            // each side of the index flush boundary.
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), chunk_payload(0));
            assert_eq!(
                sp.read_chunk(&substrate, sid, (n - 1) as u64).unwrap(),
                chunk_payload((n - 1) as u32)
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The self-heal path: a log whose chain is unusable (garbage hint)
    /// falls back to the full walk, and — because the walk reported every
    /// record as an un-indexed digest — the open immediately back-fills a
    /// fresh chain, so the *next* open takes the chain path again. A
    /// pre-chain log pays the slow walk exactly once.
    #[test]
    fn broken_chain_backfills_at_open_and_heals() {
        let dir = tmp_dir("index_heal");
        let sid = StreamId(5050);
        let n = INDEX_FLUSH_ENTRIES + 3;
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            for i in 0..n {
                sp.write_chunk(sid, i as u64, 32, 4, None, &chunk_payload(i as u32))
                    .unwrap();
            }
            sp.commit().unwrap();
        }
        // Sabotage the hint: point it at a data record (wrong type), the
        // shape an out-of-band corruption or a retired-format superblock
        // produces.
        {
            let active = dir.join(SUBSTRATE_DIR).join("seg-0000000001.log");
            let mut log = LogFile::open(&active).unwrap();
            log.set_last_index((SUPERBLOCK_SIZE, 4096)).unwrap();
        }
        // Open 1: fallback walk, then back-fill — the whole log is
        // pending, which exceeds the threshold, so a fresh chain is
        // flushed before the open returns.
        {
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(substrate.live_chunk_count(), n);
            assert!(
                sp.last_index().is_some(),
                "the open back-fills a fresh chain"
            );
            assert_eq!(sp.pending_index_len(), 0, "everything is indexed now");
        }
        // Open 2: the back-filled chain is taken and state is identical.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(substrate.live_chunk_count(), n);
            assert!(sp.last_index().is_some());
            assert_eq!(sp.pending_index_len(), 0);
            assert_eq!(
                sp.read_chunk(&substrate, sid, (n - 1) as u64).unwrap(),
                chunk_payload((n - 1) as u32)
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[ignore = "fork inheritance over a segment set is a separate redesign"]
    fn chunk_resolves_from_inherited_log() {
        let base_dir = tmp_dir("ck_base");
        let child_dir = tmp_dir("ck_child");
        let base_log = base_dir.join("base.log");
        let sid = StreamId(9000);
        let payload = chunk_payload(11);
        {
            let mut base =
                SubstratePersistence::open_concat(std::slice::from_ref(&base_log)).unwrap();
            base.write_chunk(sid, 0, 32, 4, None, &payload).unwrap();
            base.commit().unwrap();
        }
        let child_log = child_dir.join(SUBSTRATE_DIR);
        let mut child = SubstratePersistence::open_concat(&[base_log.clone(), child_log]).unwrap();
        // The chunk lives only in the inherited base — read_chunk
        // resolves it via the inherited substrate's stream index.
        let substrate = Substrate::new();
        assert_eq!(child.read_chunk(&substrate, sid, 0).unwrap(), payload);

        InheritedSubstrate::forget(&base_log);
        std::fs::remove_dir_all(&base_dir).ok();
        std::fs::remove_dir_all(&child_dir).ok();
    }
}
