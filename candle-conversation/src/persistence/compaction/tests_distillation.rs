//! Tests which records a compaction keeps for distilled and tombstoned
//! timelines.
//!
//! A tombstoned timeline is dropped with its records unless it is also
//! distilled. That pair is the provenance corpus, and its marker, label and
//! signatures must survive, or the next reload condemns the exemplar. A
//! distilled turn keeps its signature and drops its content. A text-only
//! distillation keeps the tokens and drops the signature, the KV and the
//! projection events.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::log_file::{read_record_at, MemLog, SUPERBLOCK_SIZE};

    use super::super::tests_support::tests::{has_synth, has_type, record, NO_EXPIRY};

    /// Compactor drops every record bound to a tombstoned timeline.
    /// The simulation: write a `Label` for timeline X plus a stream
    /// decl/chunk for X, then a `Tombstone` for X.  The
    /// compactor's `collect_live_records` output must contain
    /// neither the Label nor the chunk.
    /// A tombstoned timeline that is ALSO distilled is the provenance corpus —
    /// calibration exemplars, archived then distilled `ProvenanceOnly` then
    /// tombstoned so they leave the live gather while still answering the belief
    /// scan by signature. Compaction must retain them by their MODE, and must
    /// keep re-emitting the `Distilled` marker.
    ///
    /// Losing the marker is what made this self-sustaining: the next reload saw
    /// a content-declaring turn with no KV and no distill exemption
    /// (`TurnIntegrity::MissingKv`), integrity repair tombstoned it "for
    /// corruption", and the calibration resume filter regenerated the case —
    /// ~196 exemplars rebuilt every boot.
    #[test]
    fn distilled_tombstoned_timeline_keeps_its_marker_and_sigs() {
        use crate::persistence::record::{DistillPayload, TombstonePayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let corpus_tl: u64 = 4242;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: corpus_tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, 300, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 300, 0, b"corpus-kv"));
        blob.extend_from_slice(&record(RecordType::WideQSig, 300, 0, b"corpus-sig"));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: corpus_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: corpus_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        // The marker survives, at its mode — without it the exemption is lost.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Distilled
                    && it
                        .synth_payload()
                        .and_then(|p| DistillPayload::decode(p).ok())
                        .map(|d| d.mode)
                        == Some(DistillMode::ProvenanceOnly)),
            "distilled+tombstoned timeline must keep its Distilled marker",
        );
        // ProvenanceOnly retention still applies: the signature is what the
        // belief scan reads, so it stays; the KV is shed as usual.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::WideQSig
                    && it.header().stream_id == 300),
            "the provenance corpus keeps its WideQSig",
        );
        assert!(
            !live
                .iter()
                .any(|it| it.header().record_type == RecordType::Chunk
                    && it.header().stream_id == 300),
            "ProvenanceOnly still sheds the KV",
        );
    }

    /// The corpus's Label must survive its tombstone too.
    ///
    /// The Label carries `custom` — which holds the calibration marker the
    /// resume filter matches on — and the archived flag it then tests. Dropping
    /// it leaves an exemplar that cannot be recognised as done even once the
    /// distill marker is found, so the case regenerates anyway.
    #[test]
    fn distilled_tombstoned_timeline_keeps_its_label_and_marker() {
        use crate::persistence::record::{DistillPayload, TombstonePayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let corpus_tl: u64 = 777;
        let retired_tl: u64 = 888;
        let mut custom = std::collections::BTreeMap::new();
        custom.insert("calib_marker".to_string(), "file_read|abc".to_string());

        // Both need a turn stream, or the timeline is never registered and
        // `live_conv_meta` (which the Label re-emit walks) never sees it.
        let turn_decl = |tl: u64| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
                turn_index: 0,
                turn_id_day: 0,
                turn_id_seq: 1,
                role: 1,
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

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            400,
            0,
            &turn_decl(corpus_tl).encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            500,
            0,
            &turn_decl(retired_tl).encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::super::manifest::encode_label_payload(
                corpus_tl,
                "corpus-conv",
                "Corpus",
                &custom,
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::super::manifest::encode_label_payload(
                retired_tl,
                "retired-conv",
                "Retired",
                &custom,
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: corpus_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        for tl in [corpus_tl, retired_tl] {
            blob.extend_from_slice(&record(
                RecordType::Tombstone,
                0,
                0,
                &TombstonePayload {
                    timeline_id: tl,
                    turn_index: None,
                    turn_writer: 0,
                    reason: None,
                }
                .encode(),
            ));
        }

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        let has_label = |needle: &str| {
            live.iter().any(|it| {
                it.header().record_type == RecordType::Label
                    && it
                        .synth_payload()
                        .and_then(|p| std::str::from_utf8(p).ok())
                        .map(|s| s.contains(needle))
                        .unwrap_or(false)
            })
        };
        assert!(
            has_label("corpus-conv"),
            "distilled+tombstoned keeps its Label (marker + archived flag)"
        );
        assert!(
            !has_label("retired-conv"),
            "an ordinary tombstone still drops its Label"
        );
    }

    #[test]
    fn tombstoned_timeline_records_drop_during_compaction() {
        use crate::persistence::record::TombstonePayload;
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let dead_tl: u64 = 12345;
        let alive_tl: u64 = 67890;

        // Build a redo log that contains:
        //   - a Label record for the dead timeline
        //   - a Label record for the alive timeline
        //   - a TurnDecl stream for the dead timeline + one Chunk
        //   - a TurnDecl stream for the alive timeline + one Chunk
        //   - the Tombstone naming the dead timeline
        let turn_decl = |tl: u64| TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        };
        let dead_decl = StreamDecl::Turn(turn_decl(dead_tl));
        let alive_decl = StreamDecl::Turn(turn_decl(alive_tl));
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::super::manifest::encode_label_payload(
                dead_tl,
                "dead-conv",
                "Dead",
                &std::collections::BTreeMap::new(),
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::super::manifest::encode_label_payload(
                alive_tl,
                "alive-conv",
                "Alive",
                &std::collections::BTreeMap::new(),
            ),
        ));
        blob.extend_from_slice(&record(RecordType::StreamDecl, 100, 0, &dead_decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 100, 0, b"dead-chunk-payload"));
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            200,
            0,
            &alive_decl.encode(),
        ));
        blob.extend_from_slice(&record(RecordType::Chunk, 200, 0, b"alive-chunk-payload"));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: dead_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        // The dead timeline's records are physically gone from the live set.
        // Chunks are read-back (`Raw`) items keyed by stream id; the dead
        // stream (100) must contribute none.
        assert!(
            !live
                .iter()
                .any(|it| it.header().record_type == RecordType::Chunk
                    && it.header().stream_id == 100),
            "tombstoned timeline's chunk must be dropped during compaction",
        );
        assert!(
            !live.iter().any(|it| it
                .synth_payload()
                .and_then(|p| std::str::from_utf8(p).ok())
                .map(|s| s.contains("dead-conv"))
                .unwrap_or(false)),
            "tombstoned timeline's Label must be dropped during compaction",
        );
        // The alive timeline's records survive: its chunk (stream 200) present,
        // its Label carrying "alive-conv" present.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Chunk
                    && it.header().stream_id == 200),
            "alive timeline's chunk must survive compaction",
        );
        assert!(
            live.iter().any(|it| it
                .synth_payload()
                .and_then(|p| std::str::from_utf8(p).ok())
                .map(|s| s.contains("alive-conv"))
                .unwrap_or(false)),
            "alive timeline's Label must survive compaction",
        );
    }

    #[test]
    fn projection_events_record_survives_compaction() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 42,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = 4242u64; // header stream id ties the two records to one stream
        let proj_payload = br#"[{"start_token":0,"seconds":3.0,"buckets":[]}]"#.to_vec();

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::ProjectionEvents, sid, 0, &proj_payload));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            has_synth(&live, RecordType::ProjectionEvents, &proj_payload),
            "ProjectionEvents record must survive compaction with its payload intact",
        );
    }

    #[test]
    fn wide_q_sigs_record_survives_compaction() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 43,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = 4343u64; // header stream id ties the two records to one stream
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            has_synth(&live, RecordType::WideQSig, &wide_payload),
            "WideQSig record must survive compaction with its payload intact",
        );
    }

    #[test]
    fn distilled_turn_keeps_sig_drops_content() {
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let tl = 77u64;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            // A calibration turn: gather-scope tags (unchanged by distillation).
            tags: vec!["tool".to_string(), "calculator".to_string()],
        });
        let sid = 7777u64;
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, sid, 0, b"kv-chunk-content"));
        blob.extend_from_slice(&record(RecordType::Tokens, sid, 0, b"token-content"));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));
        // The distillation marker for the timeline (provenance-only).
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        // The declaration and the sig survive — the belief gallery still finds it.
        assert!(
            has_type(&live, RecordType::StreamDecl),
            "provenance-only turn keeps its StreamDecl (tags + gallery scope)",
        );
        assert!(
            has_synth(&live, RecordType::WideQSig, &wide_payload),
            "provenance-only turn keeps its WideQSig",
        );
        // The content is reclaimed.
        assert!(
            !has_type(&live, RecordType::Chunk),
            "provenance-only turn drops its KV chunks",
        );
        assert!(
            !has_type(&live, RecordType::Tokens),
            "provenance-only turn drops its tokens",
        );
        // The marker survives the compaction it just shed through, at its mode.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Distilled
                    && it
                        .synth_payload()
                        .and_then(|p| DistillPayload::decode(p).ok())
                        .map(|d| d.mode)
                        == Some(DistillMode::ProvenanceOnly)),
            "distilled marker re-emitted with its mode",
        );
    }

    /// The text-only degree (archived conversations): keep the StreamDecl +
    /// tokens as a plain read-only record; drop the KV chunks, the WideQSig, and
    /// the projection events.
    #[test]
    fn text_only_distill_keeps_tokens_drops_sig_and_kv() {
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let tl = 91u64;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = 9191u64;
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, sid, 0, b"kv-chunk-content"));
        blob.extend_from_slice(&record(RecordType::Tokens, sid, 0, b"token-content"));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));
        blob.extend_from_slice(&record(
            RecordType::ProjectionEvents,
            sid,
            0,
            b"proj-events",
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::TextOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            has_type(&live, RecordType::StreamDecl),
            "text-only keeps its StreamDecl"
        );
        // Tokens are kept as a read-back item; read its source bytes to confirm
        // the readable record survives.
        let tokens = live
            .iter()
            .find(|it| it.header().record_type == RecordType::Tokens)
            .expect("text-only keeps its tokens (the readable record)");
        let (off, size) = raw_loc(tokens);
        assert_eq!(
            read_record_at(&mut mem, off, size).unwrap().payload,
            b"token-content",
            "text-only keeps its tokens (the readable record)",
        );
        assert!(
            !has_type(&live, RecordType::Chunk),
            "text-only drops its KV chunks"
        );
        assert!(
            !has_type(&live, RecordType::WideQSig),
            "text-only drops its signature"
        );
        assert!(
            !has_type(&live, RecordType::ProjectionEvents),
            "text-only drops its projection events",
        );
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Distilled
                    && it
                        .synth_payload()
                        .and_then(|p| DistillPayload::decode(p).ok())
                        .map(|d| d.mode)
                        == Some(DistillMode::TextOnly)),
            "text-only marker re-emitted with its mode",
        );
    }
}
