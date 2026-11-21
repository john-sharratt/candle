//! Tests what counts as live weight in a segment, which decides when maintenance
//! picks it.
//!
//! Retired turns, branch checkpoints, current metadata, distilled corpora and the
//! other carried records must each count live or dead exactly as their carry rule
//! says. Counting carried bytes as dead makes a segment look reclaimable, and it
//! is then carried forward again on the next pass.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::identity::Stamped;

    use crate::persistence::streams::{StreamDecl, StreamId, TurnDecl};

    use crate::persistence::SubstratePersistence;

    use crate::projection::TimelineId;

    use crate::substrate::Substrate;

    use super::super::tests_support::tests::tmp_dir;

    /// **A turn retired from a LIVE timeline stops counting as live weight.**
    ///
    /// This is what `retention` writes when a conversation's tail falls past
    /// `keep_turns`, and it was invisible here: `classify` only knew whole
    /// *timeline* tombstones, so a retired turn's chunks and tokens still
    /// counted live, the dead ratio stayed at zero, and the compaction trigger
    /// never fired. Measured on a live daemon: ~950 of every 1000 turns
    /// retired exactly as configured, dead-byte ratio **0.0%**, and the log
    /// reached 132 GB before the disk stopped it.
    ///
    /// The `StreamDecl` stays live on purpose — the compactor emits it as a
    /// placeholder so the timeline's later turns keep their stream ids.
    #[test]
    fn a_retired_turn_stops_counting_as_live_weight() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("retired_turn_liveness");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 4242,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id();
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        // The decl is written as a real record: its liveness comes from the
        // on-disk metadata map, not from the in-RAM index, and it is what must
        // survive the turn being retired.
        let declared = sp.declare_stream(&decl).unwrap();
        sp.commit().unwrap();
        substrate.apply_stream_decl(sid, decl.clone(), declared.id);
        // The index entries a written turn leaves behind — what `segment_liveness`
        // actually reads. Placed directly so the test is about the liveness rule
        // rather than about the append path.
        substrate.apply_chunk_loc(
            sid,
            0,
            crate::persistence::manifest::ChunkLoc {
                segment: crate::persistence::segment::FIRST_SEGMENT,
                offset: 4096,
                payload_len: 100,
                record_size: 8192,
                token_count: 32,
                format: 4,
            },
            sp.mint(),
        );

        let before: u64 = sp.segment_liveness(&substrate).values().sum();
        assert!(before > 0, "a live turn's chunk is live weight");

        // Exactly what `retention::retire_expired` does to a tail turn.
        substrate.tombstone_turn(TimelineId::from_raw(4242).unwrap(), 0);

        let after: u64 = sp.segment_liveness(&substrate).values().sum();
        assert!(
            after < before,
            "a retired turn still counted as live weight: {before} -> {after}"
        );
        assert!(
            after > 0,
            "the StreamDecl placeholder must stay live — the compactor emits it"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A live branch checkpoint is live weight.** Maintenance carries it
    /// forward, so counting it dead made the segment it landed in look
    /// reclaimable and carried it again on the next pass.
    #[test]
    fn a_live_branch_checkpoint_counts_as_live_weight() {
        let dir = tmp_dir("branch_live");
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let sid = StreamId(0xB4A9C4);
        let Stamped { value: loc, id } = sp
            .write_branch_checkpoint(sid, &[&[7u8; 10_000][..]])
            .unwrap();
        sp.commit().unwrap();
        let before: u64 = sp.segment_liveness(&substrate).values().sum();
        substrate.apply_branch_checkpoint_loc(sid, loc, id);
        let after: u64 = sp.segment_liveness(&substrate).values().sum();
        assert_eq!(after - before, loc.record_size);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A stream's CURRENT per-stream metadata (here a `WideQSig`) counts as LIVE
    /// weight in `segment_liveness`, so a segment holding only live metadata is
    /// not seen as reclaimable dead — the fix that stops such a segment from being
    /// re-emitted-forward on every maintenance pass. Also survives a reload (the
    /// location map is rebuilt from the walk).
    #[test]
    fn current_metadata_counts_as_live_weight() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("meta_live");
        // A LIVE stream: it needs a StreamDecl to be reconstructible, otherwise
        // it is an orphan whose residual metadata is (correctly) dead weight.
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 777,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
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
            let declared = sp.declare_stream(&decl).unwrap();
            substrate.apply_stream_decl(sid, decl.clone(), declared.id);
            sp.append_wide_q_sigs(sid, b"wide-q-signature-bytes", sp.mint())
                .unwrap();
            sp.commit().unwrap();
            let live: u64 = sp.segment_liveness(&substrate).values().sum();
            assert!(
                live > 0,
                "a stream's current WideQSig metadata must count as live weight"
            );
            // Superseding it leaves exactly one live copy (LWW) — liveness stays
            // bounded, it does not accumulate per re-write.
            sp.append_wide_q_sigs(sid, b"newer-wide-q-signature", sp.mint())
                .unwrap();
            sp.commit().unwrap();
            assert_eq!(
                sp.metadata_locs
                    .iter()
                    .filter(|((rt, s, _), _)| *rt == RecordType::WideQSig && *s == sid.0)
                    .count(),
                1,
                "the greatest stamp: one current metadata location per (type, stream)"
            );
        }
        // Rebuilt from the walk on reload.
        {
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let live: u64 = sp.segment_liveness(&substrate).values().sum();
            assert!(live > 0, "metadata liveness must survive a reload");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **What maintenance carries counts as live, and nothing else.** A
    /// tombstoned-but-distilled timeline — the calibration corpus — keeps its
    /// decl and, under `ProvenanceOnly`, its signatures, and every maintenance
    /// pass re-emits them; so they must count live, or the segment each pass
    /// writes them into reads dead and is compacted by the next pass, forever.
    /// A timeline tombstoned without distillation carries nothing and counts
    /// nothing.
    #[test]
    fn a_distilled_corpus_counts_as_live_weight() {
        let decl = |timeline_id: u64| {
            StreamDecl::Turn(TurnDecl {
                timeline_id,
                turn_index: 0,
                turn_id_day: 0,
                turn_id_seq: 1,
                role: 2,
                block_start: 0,
                block_end: 1,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };
        let dir = tmp_dir("distilled_live");
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let live_of = |sp: &SubstratePersistence, substrate: &Substrate| -> u64 {
            sp.segment_liveness(substrate).values().sum()
        };
        let corpus = decl(9001);
        let sid = corpus.stream_id();
        let declared = sp.declare_stream(&corpus).unwrap();
        substrate.apply_stream_decl(sid, corpus.clone(), declared.id);
        sp.append_wide_q_sigs(sid, &[7u8; 4096], sp.mint()).unwrap();
        sp.commit().unwrap();
        let with_sig = live_of(&sp, &substrate);

        let tl = TimelineId::from_raw(9001).unwrap();
        substrate.distill_timeline(tl, DistillMode::ProvenanceOnly);
        substrate.tombstone_timeline(tl);
        assert_eq!(
            live_of(&sp, &substrate),
            with_sig,
            "the corpus's decl and signature are carried, so they count live"
        );

        // TextOnly drops the signature, keeps the decl.
        substrate.distill_timeline(tl, DistillMode::TextOnly);
        let text_only = live_of(&sp, &substrate);
        assert!(
            text_only > 0 && text_only < with_sig,
            "a TextOnly corpus keeps its decl but not its signature: {text_only} of {with_sig}"
        );

        // An undistilled tombstone carries nothing.
        let retired = decl(9002);
        let retired_sid = retired.stream_id();
        let declared = sp.declare_stream(&retired).unwrap();
        substrate.apply_stream_decl(retired_sid, retired, declared.id);
        sp.append_wide_q_sigs(retired_sid, &[9u8; 4096], sp.mint())
            .unwrap();
        sp.commit().unwrap();
        substrate.tombstone_timeline(TimelineId::from_raw(9002).unwrap());
        assert_eq!(
            live_of(&sp, &substrate),
            text_only,
            "a retired conversation's records count nothing"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
