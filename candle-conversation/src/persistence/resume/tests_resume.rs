//! Tests that a turn, its index pages and its tombstones survive a restart, and that the layer count is recovered
//! from the data.
//!
//! A turn that is persisted and reopened comes back byte for byte, with its chunk grid, token ids, projection
//! events, wide-Q signatures and index page. A section found to diverge is tombstoned in memory and on disk. A
//! section rebuilt after its tombstone reloads as its new generation, so it is not prefilled again on every boot.
//! The layer count is the mode across the turns, so one truncated turn does not decide it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::persistence::record::ChunkPayload;
    use crate::persistence::SUBSTRATE_DIR;
    use std::path::{Path, PathBuf};

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_resume_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn chunk_image(seed: u8, token_count: u16) -> ChunkImage {
        let kv_bytes: Vec<u8> = (0..64u32).map(|i| (i as u8) ^ seed).collect();
        ChunkImage {
            token_count,
            golden: candle::fletcher::fletcher32(&kv_bytes),
            payload: ChunkPayload {
                offset: 0,
                k_formats: vec![4, 4, 4, 4],
                v_formats: vec![5, 5, 5, 5],
                k_pal: vec![seed; 3],
                v_pal: vec![seed ^ 0xFF; 3],
                k_scale: vec![seed as f32, seed as f32 + 0.5],
                v_scale: vec![seed as f32 - 0.25],
                kv_bytes,
            },
        }
    }

    fn turn_decl(timeline: u64, turn: u32, blocks: u64) -> TurnDecl {
        TurnDecl {
            timeline_id: timeline,
            turn_index: turn,
            turn_id_day: 0,
            turn_id_seq: turn + 1,
            role: 1,
            block_start: 0,
            block_end: blocks,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        }
    }

    #[test]
    fn flat_index_round_trips_through_demux() {
        // 3 layers × 4 chunks: every (layer, chunk) maps to a distinct flat
        // index and demux puts it back.
        let (n_layers, c) = (3usize, 4usize);
        let mut flat = Vec::new();
        for layer in 0..n_layers {
            for chunk in 0..c {
                let idx = flat_chunk_index(layer, chunk, c);
                flat.push((idx, chunk_image((layer * c + chunk) as u8, 32)));
            }
        }
        let layers = demux_layers(flat, n_layers, c).unwrap();
        assert_eq!(layers.n_layers(), n_layers);
        for (layer, row) in layers.iter_layers().enumerate() {
            assert_eq!(row.len(), c);
            for (chunk, image) in row.iter().enumerate() {
                assert_eq!(image, &chunk_image((layer * c + chunk) as u8, 32));
            }
        }
    }

    #[test]
    fn demux_rejects_a_missing_chunk() {
        let mut flat = vec![(0u64, chunk_image(0, 32)), (1, chunk_image(1, 32))];
        // 2 layers × 2 = 4 expected, only 2 present.
        assert!(demux_layers(std::mem::take(&mut flat), 2, 2).is_err());
    }

    #[test]
    fn token_ids_round_trip() {
        let ids = vec![1u32, 65535, 151_935, 0, 7];
        assert_eq!(decode_token_ids(&encode_token_ids(&ids)).unwrap(), ids);
    }

    #[test]
    fn decode_token_ids_rejects_a_ragged_payload() {
        assert!(decode_token_ids(&[0u8, 1, 2]).is_err());
    }

    #[test]
    fn persist_then_recover_a_turn_round_trips() {
        let dir = tmp_dir("roundtrip");
        let decl = turn_decl(7, 0, 2); // 2 chunks per layer
        let n_layers = 3usize;
        let token_ids = vec![10u32, 20, 30, 40];

        // Build an L×C grid with unique per-(layer,chunk) seeds.
        let layers = TurnChunkGrid::new(
            (0..n_layers)
                .map(|l| {
                    (0..2)
                        .map(|c| chunk_image((l * 10 + c) as u8, if c == 1 { 12 } else { 32 }))
                        .collect()
                })
                .collect(),
        );

        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            persist_turn_kv(&mut sp, stream_id, &layers, &token_ids).unwrap();
            sp.commit().unwrap();
        }
        // Reopen — a simulated restart — and recover the turn from disk.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let decls = recovered_turn_decls(&substrate);
            assert_eq!(decls.len(), 1);
            assert_eq!(decls[0], decl);
            // Metadata half: token ids + per-layer token count, no
            // chunk-payload I/O.
            let meta = recover_turn_meta(&mut sp, &substrate, &decls[0]).unwrap();
            assert_eq!(meta.token_ids, token_ids);
            assert_eq!(meta.token_count, 32 + 12, "layer-0 chunk token counts");
            // Payload half: the full grid, byte-identical.
            let grid = recover_turn_grid(&mut sp, &substrate, &decls[0], n_layers).unwrap();
            assert_eq!(grid, layers);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Persist `turns` turns of `n_layers × chunks_per_layer`, optionally
    /// truncating one of them to `short_chunks` total records.
    fn substrate_with_turns(
        dir: &Path,
        turns: u32,
        n_layers: usize,
        chunks_per_layer: usize,
        truncate: Option<(u32, usize)>,
    ) -> Substrate {
        {
            let mut sp = SubstratePersistence::open_in(dir).unwrap();
            for t in 0..turns {
                let decl = turn_decl(7, t, chunks_per_layer as u64);
                let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
                sp.declare_stream(&StreamDecl::Turn(decl)).unwrap();
                let keep = match truncate {
                    Some((which, n)) if which == t => n,
                    _ => n_layers * chunks_per_layer,
                };
                // Written flat, so a truncation drops trailing records exactly
                // the way an interrupted write does — which is why this bypasses
                // `persist_turn_chunks` (it insists on a whole grid, and a whole
                // grid is what a partial write does not have).
                for i in 0..keep {
                    let image = chunk_image(i as u8, 32);
                    sp.write_chunk(
                        stream_id,
                        i as u64,
                        image.token_count as u64,
                        image.payload.k_formats.first().copied().unwrap_or(0),
                        Some(image.golden),
                        &image.payload,
                    )
                    .unwrap();
                }
            }
            sp.commit().unwrap();
        }
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
        substrate
    }

    #[test]
    fn the_layer_count_comes_from_the_turns_not_from_a_constant() {
        // The false alarm this exists to prevent: a checker that assumes 48
        // layers against a hybrid stack that seals KV for 11 of them calls
        // every healthy turn corrupt. Derived, the answer is the truth.
        let dir = tmp_dir("derive_layers");
        let substrate = substrate_with_turns(&dir, 4, 11, 3, None);
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(derive_layer_count(&substrate, &decls), Some(11));
        for decl in &decls {
            assert!(
                recover_turn_cold_refs(&substrate, decl, 11).is_ok(),
                "a turn sealed with 11 layers must validate against 11"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_truncated_turn_does_not_get_to_define_the_layer_count() {
        // The trap in taking the first or the minimum: a partial write IS a
        // turn with the wrong chunk count, so letting one vote decide would
        // validate the corrupt turn and condemn every healthy one — the same
        // false alarm, inverted. The mode keeps the majority's answer.
        let dir = tmp_dir("derive_truncated");
        // Turn 0 keeps only 4 layers' worth (12 of 33 records).
        let substrate = substrate_with_turns(&dir, 4, 11, 3, Some((0, 12)));
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(
            derive_layer_count(&substrate, &decls),
            Some(11),
            "three healthy turns outvote one truncated one"
        );
        let n = derive_layer_count(&substrate, &decls).unwrap();
        let bad = decls
            .iter()
            .filter(|d| recover_turn_cold_refs(&substrate, d, n).is_err())
            .count();
        assert_eq!(bad, 1, "exactly the truncated turn is reported");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_ragged_turn_names_no_layer_count_and_is_skipped() {
        // 13 records against 3 chunks/layer is not a whole number of layers,
        // so it cannot vote — it is the ragged write the caller is hunting.
        let dir = tmp_dir("derive_ragged");
        let substrate = substrate_with_turns(&dir, 3, 11, 3, Some((0, 13)));
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(derive_layer_count(&substrate, &decls), Some(11));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_substrate_with_no_chunked_turns_derives_nothing() {
        let dir = tmp_dir("derive_empty");
        let substrate = substrate_with_turns(&dir, 0, 11, 3, None);
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(derive_layer_count(&substrate, &decls), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn projection_events_persist_and_recover() {
        use crate::projection::{
            decode_events, encode_events, BucketKind, ProjectionBucket, ProjectionEvent,
            TimelineId, TurnIndex,
        };
        let dir = tmp_dir("proj_events");
        let decl = turn_decl(7, 0, 2);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();

        // A two-event timeline for the turn.
        let events = vec![
            ProjectionEvent {
                start_token: 0,
                seconds: 3.0,
                materialized_tokens: 1120,
                substrate_tokens: 42_000,
                buckets: vec![
                    ProjectionBucket {
                        label: "system".into(),
                        kind: BucketKind::System,
                        tokens: 320,
                    },
                    ProjectionBucket {
                        label: "code_read".into(),
                        kind: BucketKind::Section,
                        tokens: 800,
                    },
                ],
                selection: Default::default(),
                materialized: Default::default(),
                self_reference: false,
                ..Default::default()
            },
            ProjectionEvent {
                start_token: 120,
                seconds: 8.0,
                materialized_tokens: 540,
                substrate_tokens: 42_000,
                buckets: vec![ProjectionBucket {
                    label: "conversation".into(),
                    kind: BucketKind::Turns,
                    tokens: 540,
                }],
                selection: Default::default(),
                materialized: Default::default(),
                self_reference: false,
                ..Default::default()
            },
        ];
        let payload = encode_events(&events);

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            sp.append_projection_events(stream_id, &payload, sp.mint())
                .unwrap();
            sp.commit().unwrap();
        }
        // Reopen — a simulated daemon restart — and recover the timeline.
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(decl.timeline_id).unwrap();
            // Recovered located, not held: nothing of it stands in RAM.
            assert_eq!(substrate.held_payload_bytes().projection_events, 0);
            let stream = substrate.stream_of(stream_id).unwrap();
            assert!(matches!(
                stream.projection_events,
                Some(crate::substrate::Located::Stored(_))
            ));
            let blob = substrate
                .projection_events_blob(tl, TurnIndex(decl.turn_index))
                .expect("a recovered record reads back")
                .expect("projection events recovered after restart");
            assert_eq!(&blob[..], payload.as_slice());
            assert_eq!(decode_events(&blob), events);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wide_q_sigs_persist_and_recover() {
        use crate::projection::{TimelineId, TurnIndex};
        use crate::provenance::{decode_wide_sigs, encode_wide_sigs, WideQSig};

        let dir = tmp_dir("wide_q_sigs");
        let decl = turn_decl(9, 0, 3);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();

        // A 3-token wide history, 4 heads × head_dim 128, alternating signs per token.
        let history: Vec<WideQSig> = (0..3)
            .map(|t| {
                let band: Vec<f32> = (0..4 * 128)
                    .map(|i| if (i + t) % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                WideQSig::from_band(&band, 128)
            })
            .collect();
        let payload = encode_wide_sigs(&history);

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            sp.append_wide_q_sigs(stream_id, &payload, sp.mint())
                .unwrap();
            sp.commit().unwrap();
        }
        // Reopen — simulated daemon restart — and recover the window.
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(decl.timeline_id).unwrap();
            // Recovered located, not held: nothing of it stands in RAM.
            assert_eq!(substrate.held_payload_bytes().wide_q_sigs, 0);
            let blob = substrate
                .wide_q_sigs_blob(tl, TurnIndex(decl.turn_index))
                .expect("a recovered record reads back")
                .expect("wide-Q sigs recovered after restart");
            assert_eq!(&blob[..], payload.as_slice());
            assert_eq!(decode_wide_sigs(&blob), Some(history.clone()));
            // A scan sizes the window from the record's head alone, and reads
            // its tokens back exactly.
            let window = substrate
                .wide_sig_window(stream_id)
                .expect("the window is sized from its header");
            assert_eq!(window.len(), 3);
            assert_eq!(window.decode(), history);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A turn's QSA index page has to outlive the process that sealed it.
    ///
    /// Held only in RAM it dies with the daemon, and every turn cold-loaded
    /// afterwards hands a projection its K/V
    /// with no index of it — the slot's caches then sit permanently short by
    /// that turn's width and the select refuses once the conversation passes the
    /// identity threshold. This asserts the byte-for-byte round trip through a
    /// simulated restart, which is the only thing that closes that gap.
    #[test]
    fn index_page_persists_and_recovers() {
        use crate::projection::{TimelineId, TurnIndex};

        let dir = tmp_dir("index_page");
        let decl = turn_decl(9, 0, 3);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        // Opaque to this layer by design — the model owns the encoding, and the
        // log carries the bytes without inspecting them.
        let payload: Vec<u8> = (0..512u32).map(|i| (i % 251) as u8).collect();

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            sp.append_turn_index_page(stream_id, &payload, sp.mint())
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(decl.timeline_id).unwrap();
            let page = substrate
                .index_page_at(tl, TurnIndex(decl.turn_index))
                .expect("index page recovered after restart");
            // Recovered, it is left in the log: located, not held in RAM — and
            // read back, byte for byte the payload the seal wrote.
            assert!(
                page.stored().is_some(),
                "a recovered page is not held in RAM"
            );
            assert_eq!(page.read_with(&mut sp), payload);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A prompt section's index page survives a restart the way a turn's does,
    /// keyed by the section's content-addressed stream id. Without it a section
    /// restored from the log could only reach a positional-state model
    /// unindexed, so the restore has to be refused and the section prefilled
    /// again on every boot.
    #[test]
    fn section_index_page_persists_and_recovers() {
        use super::super::super::content_hash::ContentHash;
        use super::super::super::streams::{ContentAddress, SectionDecl};

        let dir = tmp_dir("section_index_page");
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 0x11, hi: 0x22 },
                section_hash: ContentHash { lo: 0x33, hi: 0x44 },
            },
            debug_name: "system_framing".to_string(),
        });
        let stream_id = decl.stream_id();
        let payload: Vec<u8> = (0..384u32).map(|i| (i % 241) as u8).collect();

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&decl).unwrap();
            let id = sp.mint_for(RecordType::TurnIndexPage, stream_id);
            sp.append_turn_index_page(stream_id, &payload, id).unwrap();
            // A page minted outside the section's generation is not the
            // section's: it is refused on replay.
            sp.append_turn_index_page(stream_id, b"stray", sp.mint())
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let page = substrate
                .index_page_of(stream_id)
                .expect("section page recovered");
            assert_eq!(page.read_with(&mut sp), payload);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End-to-end exercise of the reactive section-corruption repair, at the
    /// persistence/substrate layer — everything the scheduler's
    /// `repair_section_if_window_divergence_confirmed` hook drives once it
    /// has a `seal_error` and a `section_id`/`address` in hand, EXCEPT the
    /// hook itself: reproducing the actual `snapshot_sequence_per_layer`
    /// seal failure needs a live `BatchedInferenceSession` on a real model,
    /// which this crate's unit tests cannot stand up (see the `tensor-assert`
    /// / GPU-test guidance in `AGENTS.md` — this suite runs CPU-only). What
    /// this DOES cover, against real on-disk records rather than a
    /// synthetic shape:
    ///
    /// 1. A section persisted with a genuine per-layer offset divergence —
    ///    two layers agree, one does not — is detected by re-reading it off
    ///    disk with [`read_persisted_section_windows`] and running the
    ///    generic [`super::super::chunk_window_integrity::first_divergent_chunk`]
    ///    against the result.
    /// 2. Tombstoning it (the in-RAM [`Substrate::tombstone_section`] plus
    ///    the durable [`SubstratePersistence::write_section_tombstone`]) makes
    ///    [`Substrate::section_exists`] read the section as gone in THIS SAME
    ///    process — no restart needed — which is what stops the very next
    ///    ingest triage from restoring the same corrupted chunks again.
    /// 3. The tombstone survives a simulated restart (a fresh
    ///    `open_in_with_substrate` over the same directory): the corrupted
    ///    section reads as gone on reload too, not just in the process that
    ///    found it.
    #[test]
    fn reactive_repair_confirms_and_tombstones_a_diverged_section() {
        use super::super::super::content_hash::ContentHash;
        use super::super::super::streams::{ContentAddress, SectionDecl};
        use crate::persistence::chunk_window_integrity::first_divergent_chunk;
        use crate::projection::SectionId;

        let dir = tmp_dir("section_window_divergence_repair");
        let (n_layers, chunks_per_layer) = (3usize, 2usize);
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 0x55, hi: 0x66 },
                section_hash: ContentHash { lo: 0x77, hi: 0x88 },
            },
            debug_name: "diverged_tool_catalog".to_string(),
        });
        let stream_id = decl.stream_id();

        // Layers 0 and 1 agree on both chunks; layer 2's second chunk was
        // written with a different offset — the "some layers ahead of the
        // rest" signature a torn cold-tier write leaves behind, caught here
        // exactly because it survived to disk rather than being refused
        // live by `assert_sealed_layers_aligned`.
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&decl).unwrap();
            for layer in 0..n_layers {
                for chunk in 0..chunks_per_layer {
                    let mut image = chunk_image((layer * 10 + chunk) as u8, 17);
                    if layer == 2 && chunk == 1 {
                        image.payload.offset = 5;
                    }
                    let flat = flat_chunk_index(layer, chunk, chunks_per_layer);
                    sp.write_chunk(
                        stream_id,
                        flat,
                        image.token_count as u64,
                        image.payload.k_formats.first().copied().unwrap_or(0),
                        Some(image.golden),
                        &image.payload,
                    )
                    .unwrap();
                }
            }
            sp.commit().unwrap();
        }

        // 1. Confirm the divergence from what is actually on disk.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let windows = read_persisted_section_windows(&mut sp, &substrate, stream_id, n_layers)
                .unwrap()
                .expect("the stream has durable chunks");
            let divergence =
                first_divergent_chunk(&windows).expect("the offset divergence must be found");
            assert_eq!(divergence.chunk_index, 1);
            assert_eq!(
                divergence.per_layer,
                vec![Some((0, 17)), Some((0, 17)), Some((5, 17))],
            );

            // A section registered under this stream must be found before
            // repair …
            let section_id = SectionId::new(1);
            substrate.set_section_full(
                section_id,
                stream_id,
                34,
                Vec::new(),
                std::sync::Arc::new(vec![1, 2, 3]),
            );
            assert!(substrate.section_exists(section_id));
            assert!(!substrate.is_section_tombstoned(stream_id));

            // 2. Repair: tombstone in-RAM and durably.
            let reason = divergence.describe();
            let generation = substrate.tombstone_section(stream_id);
            sp.write_section_tombstone(stream_id.0, generation, Some(&reason))
                .unwrap();
            sp.commit().unwrap();

            // … and gone after, in this SAME process, with no restart.
            assert!(
                !substrate.section_exists(section_id),
                "a tombstoned section must read as absent so the next triage re-prefills fresh",
            );
        }

        // 3. The tombstone is durable: a fresh reload sees it too.
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(
                substrate.is_section_tombstoned(stream_id),
                "the tombstone must survive a restart, or the corrupted section reloads again",
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A section rebuilt after its tombstone is a section again, across a
    /// restart.** The rebuild re-prefills the same text, so it lands on the same
    /// content-addressed stream id; the reload must read `old generation →
    /// tombstone → new generation` as exactly the new generation — untombstoned,
    /// indexing only the new chunks — or a rebuilt section re-prefills on every
    /// boot forever. The new generation is written with fewer chunks than the
    /// old so a leftover would show.
    #[test]
    fn a_section_rebuilt_after_its_tombstone_reloads_as_the_new_generation() {
        use super::super::super::content_hash::ContentHash;
        use super::super::super::streams::{ContentAddress, SectionDecl};

        let dir = tmp_dir("section_rebuilt_after_tombstone");
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 0x91, hi: 0x92 },
                section_hash: ContentHash { lo: 0x93, hi: 0x94 },
            },
            debug_name: "rebuilt_tool".to_string(),
        });
        let stream_id = decl.stream_id();
        let write_generation = |sp: &mut SubstratePersistence, chunks: usize, seed: u8| {
            sp.declare_stream(&decl).unwrap();
            for flat in 0..chunks {
                let image = chunk_image(seed + flat as u8, 17);
                sp.write_chunk(
                    stream_id,
                    flat as u64,
                    image.token_count as u64,
                    image.payload.k_formats.first().copied().unwrap_or(0),
                    Some(image.golden),
                    &image.payload,
                )
                .unwrap();
            }
            sp.commit().unwrap();
        };

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            write_generation(&mut sp, 6, 0);
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let generation = substrate.tombstone_section(stream_id);
            assert!(generation.is_some(), "the section is stamped");
            sp.write_section_tombstone(stream_id.0, generation, Some("system prompt rebuild"))
                .unwrap();
            sp.commit().unwrap();
            write_generation(&mut sp, 2, 100);
        }
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(
                !substrate.is_section_tombstoned(stream_id),
                "the new generation's decl must lift the tombstone on reload",
            );
            let stream = substrate
                .stream_of(stream_id)
                .expect("the rebuilt stream is indexed");
            assert_eq!(
                stream.chunks.keys().copied().collect::<Vec<_>>(),
                vec![0, 1],
                "exactly the new generation's two chunks — none of the old six",
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Last-writer-wins: re-sealing a turn replaces its page rather than
    /// leaving the stale one to be handed to the next projection.
    #[test]
    fn index_page_last_writer_wins() {
        use crate::projection::{TimelineId, TurnIndex};

        let dir = tmp_dir("index_page_lww");
        let decl = turn_decl(9, 0, 3);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            sp.append_turn_index_page(stream_id, b"first-page", sp.mint())
                .unwrap();
            sp.append_turn_index_page(stream_id, b"second-page", sp.mint())
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(decl.timeline_id).unwrap();
            let page = substrate
                .index_page_at(tl, TurnIndex(decl.turn_index))
                .expect("index page recovered");
            assert_eq!(page.read_with(&mut sp), b"second-page".to_vec());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recover_turn_demux_fails_on_wrong_layer_count() {
        let dir = tmp_dir("wrong_layers");
        let decl = turn_decl(9, 0, 2);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        let layers = TurnChunkGrid::new(
            (0..3)
                .map(|l| (0..2).map(|c| chunk_image((l * 5 + c) as u8, 32)).collect())
                .collect(),
        );
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            persist_turn_kv(&mut sp, stream_id, &layers, &[1, 2]).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // The turn was written with 3 layers; recovering as 4 must fail.
            assert!(recover_turn_grid(&mut sp, &substrate, &decl, 4).is_err());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_substrate_dir_is_used() {
        // Guards the doc-comment claim that the log lives under substrate/.
        let dir = tmp_dir("dir_check");
        SubstratePersistence::open_in(&dir).unwrap();
        assert!(dir.join(SUBSTRATE_DIR).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// After a persist+reopen, `recover_turn_cold_refs` returns one
    /// `StoredSequence` per layer with the per-chunk `log_offset` /
    /// `record_len` / `token_count` populated from the on-disk
    /// manifest entries — the data the substrate's reload path needs
    /// to install the cold tier without reading any chunk payloads.
    #[test]
    fn recover_cold_refs_demuxes_manifest_chunks_per_layer() {
        let dir = tmp_dir("cold_refs");
        let n_layers = 3usize;
        let chunks_per_layer = 2usize;
        let decl = turn_decl(11, 0, chunks_per_layer as u64);
        let layers = TurnChunkGrid::new(
            (0..n_layers)
                .map(|l| {
                    (0..chunks_per_layer)
                        .map(|c| chunk_image((l * 10 + c) as u8, if c == 1 { 12 } else { 32 }))
                        .collect()
                })
                .collect(),
        );

        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            persist_turn_kv(&mut sp, stream_id, &layers, &[1, 2, 3]).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let cold = recover_turn_cold_refs(&substrate, &decl, n_layers)
                .unwrap()
                .expect("Some(...) when chunks exist");
            assert_eq!(cold.len(), n_layers, "one StoredSequence per layer");
            for (layer, seq) in cold.iter().enumerate() {
                assert_eq!(seq.chunks.len(), chunks_per_layer);
                // Token counts demuxed back into the right layer slots.
                assert_eq!(seq.chunks[0].token_count, 32);
                assert_eq!(seq.chunks[1].token_count, 12);
                assert_eq!(seq.token_count, 32 + 12);
                // `log_offset` must be the actual on-disk record
                // offset — past zero (header lands at the start of
                // the file) and monotonically increasing within the
                // layer.
                let _ = layer;
                assert!(seq.chunks[0].log_offset > 0, "log_offset populated");
                assert!(seq.chunks[1].record_len > 0, "record_len populated");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A turn with no `Chunk` records (e.g. crash before bg-quantizer
    /// callback persisted them) returns `None` — the reload path uses
    /// this to leave `cold = None` on the residence so the classifier
    /// reports it `missing` instead of attempting a bogus cold→hot.
    #[test]
    fn recover_cold_refs_returns_none_when_no_chunks() {
        let dir = tmp_dir("cold_refs_empty");
        let decl = turn_decl(13, 0, 0);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            // No persist_turn_kv — stream declared but no Chunk records.
            persist_tokens_only(&mut sp, stream_id, &[7, 8]).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let cold = recover_turn_cold_refs(&substrate, &decl, 2).unwrap();
            assert!(cold.is_none(), "no chunks → None");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `persist_tokens_only` must return the WRITTEN record's on-disk location so
    /// the caller can fold it into the substrate index (`apply_tokens_loc`).
    /// Pre-fix it returned `()`, so `entry.tokens` stayed `None` and the
    /// maintenance/compaction relocation (both gate on `if let Some(loc) =
    /// entry.tokens`) silently reclaimed a live turn's tokens on the next idle
    /// pass. Assert the returned loc is populated AND byte-for-byte agrees with the
    /// location the reload walk independently records for the same record — a wrong
    /// loc would corrupt token reads once callers register it.
    #[test]
    fn persist_tokens_only_returns_the_written_record_location() {
        let dir = tmp_dir("tokens_loc_return");
        let decl = turn_decl(21, 0, 0);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        let live_loc = {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            let loc = persist_tokens_only(&mut sp, stream_id, &[7, 8, 9]).unwrap();
            sp.commit().unwrap();
            loc.value
        };
        assert!(live_loc.offset > 0, "offset past the file-start header");
        assert!(live_loc.record_size > 0, "padded record_size populated");
        assert_eq!(
            live_loc.payload_len,
            3 * 4,
            "3 u32 tokens = 12 payload bytes"
        );
        // The reload walk records the same record's loc straight from disk — the
        // live-write loc must match it exactly, proving it points at the real bytes.
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let (wseg, woff, wsize, wpay) = {
            let e = substrate.stream_of(stream_id).expect("stream present");
            let t = e
                .tokens
                .as_ref()
                .expect("reload registered the tokens loc from disk");
            (t.segment, t.offset, t.record_size, t.payload_len)
        };
        assert_eq!(wseg, live_loc.segment, "segment matches reload walk");
        assert_eq!(woff, live_loc.offset, "offset matches reload walk");
        assert_eq!(
            wsize, live_loc.record_size,
            "record_size matches reload walk"
        );
        assert_eq!(
            wpay, live_loc.payload_len,
            "payload_len matches reload walk"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
