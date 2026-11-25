# The `files` collection — build plan and status

The staged build of `docs/provenance_system_collection.md` (the design,
authoritative; its §12 records decisions D1–D18). This file is the working
plan: each stage's scope at file level, and where the build stands. It is
retired once the rollout's test step reports.

**Branch:** `provenance-system-collection`. Every commit passes
`cargo check --workspace`, `cargo check --workspace --features cuda`,
`cargo test -p candle-conversation --features hub --lib` and
`cargo test -p zend --lib`, and is formatted and clippy-clean. The branch
merges to `main` once, after Stage 8; the rollout's steps 3–5 follow.

**Mandate (John, 2026-11-26):** build the whole design, roll it out per §9,
test it and report the results, with no further input from him; every act
authorized — commits, deploys, both zend wipes. The one stop: if attention
alone does not read the dynamic members (§8, §9 step 5), stop and report.

Status: `[ ]` todo · `[~]` in progress · `[x]` done (commit).

## Rollout

- [x] §9 step 1 — WhiteWorld's zend out: `apps: [npcd]` in WhiteWorld's
  `C:\prog\motherd-local.yaml`, its zend substrate (`D:\prog\substrate`,
  486.89 GB) wiped. WhiteWorld's motherd restarts after AlienWorld's trial.
- [ ] Stages 1–8 below.
- [ ] §9 step 3 — wipe BlackWorld's zend substrate, `max_depth: 1` in
  BlackWorld's `D:/prog/motherd-local.yaml`, deploy the merged build.
- [ ] §9 step 4 — ingest to settled: every root file prefilled and
  calibrated, normalization run, `tenants.zend.settled` true.
- [ ] §9 step 5 — the test: the four-way ranking split (content /
  empty-frame calibrations / target-frame calibrations / all), in-view
  naming on the production chat path, the forced inventory, sealed against
  fresh, the production placement, the close-row GPU tests, no reasoning
  loops.

## Stage 1 — the knowledge notation and the piece plan (S)

Add `candle-conversation/src/files/{mod,render,key}.rs`.
`render::knowledge_text` reproduces Q1 (`zend/examples/dependency_battery/probe.rs`
`qwen_knowledge`, which then uses it). `FileKey(u64)` is the first 8 bytes
of `content_hash` over `v{N}:{path}@{blob}`; `LineSpan { first, last }`
1-based inclusive; `ChunkKey { file, lines }`. The pieces and the chunk cut
are D2 and D3. Tests: `a_whole_file_renders_as_q1_rendered_it`,
`a_file_with_a_gap_keeps_one_heading_and_marks_the_gap`,
`files_are_joined_by_a_blank_line_and_the_last_ends_on_its_fence`,
`file_key_is_the_digest_of_its_content_key`,
`a_chunk_cut_moves_to_a_seam_that_tokenizes_alike`, and the real-vocabulary
`tests/knowledge_pieces_real_vocab.rs::assembled_pieces_tokenize_as_the_knowledge_text`
(token ids, through `tests/common/real_vocab.rs::for_each_tokenizer`).

## Stage 2 — file records, the member seal, signatures on sections (L)

- Split `conversation/submitted_sections.rs`'s `SealJob` (:294-597) into
  `conversation/section_seal/{job,address,start_state}.rs`.
- New `conversation/member_seal.rs`: `Sequence::seal_members(&[MemberSpec])
  -> Vec<MemberHandle>`. Prefix D4 (`birth_prefix`,
  `sequence_construction.rs:449-451`, then `files_open`). Address
  `shared_section_prefix(chain_of(prefix ++ after))`. Start from the
  predecessor's end, else `state_after` + `continue_state`
  (`prefix_state.rs:194`). `keep_end` → `Sharing::SharedEnd` (a file's last
  chunk, every seed link). Registered under one owner in the substrate's
  files registry.
- `SchedulerRequest::IngestSection` (`requests.rs:306-356`) gains
  `signatures: bool`. `section_finish.rs:19-64` passes it, and
  `perform_seal.rs`'s Section arm (:618-724) enqueues `encode_wide_sigs_with`
  under `section_stream_id(address)` (the signatures are gathered at
  :183-196). Check that `wide_sig_window` (`live_state.rs:540`) indexes
  section streams.
- Add `SchedulerRequest::RestoreSections` (the batch of :454-470),
  `substrate/files.rs` (registry, `register_file`, `files_under`,
  `file_by_path`, `deliverable_file`), `substrate/files_apply.rs` (a record's
  arrival), `engine/files_restore.rs` (boot, `custom_objects_under("files/")`).
- Data, `candle-conversation/src/files/{record,entry,pieces,seed,pin,member_role}.rs`:
  - `FileRecord { content_key, path, blob, unit: File|Folder, lines, lang,
    branches, commit, quant, heading: MemberRef, chunks:
    Vec<ChunkRecord { lines, member }> }`, `MemberRef { address:
    ContentAddress, tokens }`, a JSON custom object `files/{path}@{blob}`.
  - `FileEntry { key, record, kind, heading: MemberSlot, chunks, gallery }`,
    `MemberSlot { stream, tokens, section: Option<SectionId> }`.
  - `FilesPieces { gap, closing_join, closing_last }`.
  - `SeedChain { links: Vec<SeedLink { file, sections }>, end: StreamId,
    tokens }`.
  - `FilesPin { Empty, Only(FileKey) }`.
  - `MemberRole { Heading, Chunk, Gap, Closing, Seed }`.
- Substrate: `files: PMap<FileKey, Arc<FileEntry>>`, `files_by_path`,
  `files_pieces`, `seed_chain`, and the calibration index keyed
  `FileKey → Vec<TimelineId>`.
- Tests: `a_file_record_round_trips_as_its_custom_object` (raw JSON),
  `a_member_is_addressed_by_its_prefix_and_earlier_chunks`,
  `a_member_is_left_out_of_every_other_sections_prefix_hash`,
  `a_registered_file_is_deliverable_only_once_every_member_stream_is_persisted`,
  `a_replicated_record_registers_its_entry`; scheduler, toy model:
  `a_member_seal_writes_its_signature_and_index_page`,
  `restore_sections_installs_cold_markers_in_one_request`.

## Stage 3 — the close row; `system_close` out of the checkpoint (L)

Engine-wide, npcd included.

- `SectionSchema.close_row: bool`; YAML `close_row: true` on `system_close`
  (zend `projection.yaml:704-706`, `npcd/projection.yaml:153`). A close row
  over `TIMELINE_ROWS` (8) tokens refuses to load.
- Never in `fixed_prefix` (`sequence_construction.rs:449-451`) or the
  `prompt_branch` tokens (:733-743); `emit.rs` never emits it.
- `turn_submitter.rs:494-540`: `turn_head_tokens` prepends the close row on
  an opening — no turn in the conversation or its lineage — as a leading
  glue segment before `user_start` (D7), turn groups included. The span is
  left out of the turn's signatures, as reasoning is
  (`perform_seal.rs:193-196`).
- The opening's apply (`apply_build.rs`, `prefill.rs:70-105`) stages the
  opening state (D6) before the opening prefill. The state after
  `files_open` comes from `conversation/files_open_state.rs`, memoized per
  branch and persisted with the branch checkpoint's payload, keyed
  `hash(branch ‖ files_open)`. An opening's first reading installs it
  (`first_reading.rs`, D10).
- Tests: `the_close_row_is_out_of_the_birth_prefix_and_the_checkpoint`,
  `an_opening_turn_bakes_the_close_row_before_user_start`,
  `a_turn_after_the_first_bakes_no_close_row`,
  `a_fork_with_lineage_bakes_none`,
  `a_close_row_longer_than_eight_tokens_refuses_to_load`. Update
  `tests_branch_checkpoint.rs`, `tests_turn_assembly.rs`,
  `every_inter_turn_island_is_assistant_end_then_user_start` and npcd's
  schema tests. GPU (`#[ignore]`, `--test-threads=1`):
  `an_opening_state_is_the_frame_state_advanced_by_the_close_row`,
  `a_running_conversations_state_is_the_same_before_and_after_a_reprojection`.

## Stage 4 — the collection in the projection (XL)

**4a, schema and emit.**
- `parse_groups.rs:271-283` lifts the `working_set` refusal for a collection
  declaring `kinds:`.
- `projection/schema/files.rs`: `FilesSchema { kinds: Vec<FilesKind { name,
  share, paths, except, recency, policy, locality }> }`.
- `emit.rs` returns `files_at`; `collection.rs:66,:354` route the
  collection there.
- `project/files_emit.rs` emits `files_open`, each file's heading, chunks,
  gap and `closing_join`/`closing_last`, then the seed, all as
  `SealedKind::Section` with a `MemberRole`.
- `slot_state.rs:274-283` `prompt_end_of` stops at the first member, and
  `pin_ranges_of` never pins a member.
- Tests: `a_files_collection_parses_with_kinds_and_shares`,
  `the_knowledge_base_is_the_end_of_the_system_prompt`,
  `a_partial_file_keeps_one_heading_and_a_gap_piece`,
  `the_prompt_ends_at_files_open`, `an_empty_pin_emits_the_heading_alone`,
  `an_only_pin_emits_that_files_members_alone`; update
  `tests_schema_validation.rs`.

**4b, the scan.**
- `belief_files.rs:44-61` `FileScan` takes `ScanId::{Timeline, File}`;
  `assemble_member_scans`.
- `resolver/belief_files_kinds.rs` `score_belief_files` mirrors
  `belief_groups.rs:28-529`: `ChildKey::file`, `kind_z`, `vote_top_k`, the
  seed left out as `exclude_lineage` left the chain out,
  `ProjectionScores::set_chunk`.
- `belief_gallery.rs:169-283` preloads the files kinds (`Holder::Shared`,
  keyed by member stream).
- Normalization takes `ChildKey::file`; `warm_ingest` stays for npcd.
- Tests: `a_files_scan_stamps_each_chunk_with_its_files_score`,
  `calibrations_vote_for_their_file_key`,
  `seed_files_compete_and_are_not_offered`, `a_kind_reads_only_its_paths`;
  port `tests_belief`'s cases.

**4c, the working set, the pool and the fill.**
- `working_set/state.rs`: re-keyed to `FileKey`; `seed`/`holds_out`/`seeds`
  go; `placed_order` is added.
- `registry.rs`: candidates and observation by `FileKey`;
  `working_set_seed` goes.
- `working_set_observe.rs`, and `working_set_pick.rs` (in `placed_order`).
- `dialogue_fill.rs`: a `FilePiece` has no floor; plan over chunk measures;
  order out.
- `sink.rs`:
  - 4b reads members;
  - the lineage drop (:639-648) goes;
  - the fill's room is taken before the splice (window − reserve −
    unspliced system − decode_pos − seed);
  - `LINEAGE_BASE` (:1018-1063) goes;
  - `first_request` becomes the root exchange (D8);
  - `files_emit` is spliced at `files_at` in order: at the opening, fill
    order; later, `placed_order` kept and then newcomers;
  - the block reorder (:1265-1294) and faces (:1296-1306) go.

  Split the fill and splice into `pipeline/files_fill.rs`.
- Pool pages and hits per `ChunkKey` (`pooled_rounds.rs`, `pool_fetch.rs`,
  `apply_build.rs:573-579`; `HitKey::{Round, Chunk}`).
- `ContentResolver`:
  - adds `files_members`, `file_chunks`, `chunk_token_count`, `chunk_score`,
    `chunk_reads`, `member_section`, `files_pieces`, `seed_chain`, `seeded`,
    `seed_held_out`, `files_boundary`;
  - drops `seed_timelines`, `face_of`, `unit_of`;
  - keys `working_set_standing` by `FileKey`.
- Tests:
  - new: `a_kept_file_keeps_its_place_and_a_newcomer_follows_the_kept_ones`,
    `the_opening_orders_files_strongest_first`,
    `a_files_chunks_compete_alone_with_no_floor`,
    `the_root_exchange_is_a_floor_for_a_fork`,
    `members_are_counted_out_of_the_room_and_the_fixed_part_is_measured_without_them`,
    `pooled_chunks_are_declared_per_page`;
  - ported to `FileKey`: the working set's tests;
  - deleted: the seed tests and the face tests.

## Stage 5 — ingest as chunk prefill; folders as listings (XL)

- `zend/src/code_read/process.rs::process_one_file` (:20-455) keeps the
  lease, the binary sniff and the deferred tombstone, and becomes:
  1. cut (`PAGE_LINES` and the seam);
  2. `base.seal_members([heading, chunks, keep_end on the last])`;
  3. write the `FileRecord`;
  4. tombstone what it supersedes;
  5. calibrate (Stage 7).
- `zend/src/files/{cut,listing,record}.rs`.
- `repo_scan`'s folder unit is its listing (D14), through `process_one_file`
  as `FileUnit::Folder`. `repo_scan`'s conversation and summary path goes.
- Deleted: `code_read/{reading,reading_break,reading_holds}.rs`,
  `zend/src/faces.rs`, `candle-conversation/src/substrate/faces.rs`,
  `zend/src/ingest_snapshots.rs`, the chain checks,
  `retire_crashed_partials`.
- `ingest_backlog` and `settled` text follow.
- Tests:
  `a_file_is_cut_into_whole_parts_of_at_most_200_lines`,
  `a_folders_listing_names_each_entry_repository_relative_with_its_size`,
  `a_committed_file_writes_one_record_naming_every_chunk`,
  `a_superseded_record_is_retired_only_after_the_new_one_commits`,
  `a_binary_file_writes_no_record`.

## Stage 6 — the seed chain, pinned last; the opening on its snapshot (L)

- `branch_ingest/prime.rs` keeps `links()`. `prime` and `repoint` become one
  `seal_members` chain over the links (`keep_end` on each link's last) into
  a `SeedChain`, set on the substrate. D16's refusal.
- Deleted:
  - `code_read/link.rs`, `RefreshContext::link_parent`;
  - `SEED_PARENT`/`UNIT_PARENT`/`unit_parent`/`seed`/`set_seed`/`seed_chain`/`parented_on_a_base`
    (`registry.rs:201-329`);
  - `INHERITED_CHAIN_TOKEN_CAP` and its cap in `inherited_chain` (the walk
    stays for forks);
  - `SeedRecurrentFromLineage` and `seed_recurrent_from_lineage`;
  - `set_forked_from_seed` (`stream_turn.rs:189-258`), which becomes
    recording `seeded` (D9);
  - `working_set.seeds`, `zend/src/working_set/seeds.rs`,
    `SelectionOrigin::Seed`.
- `journal::opening::SeedRead` reads the `SeedChain`. The opening's state is
  the `SeedChain`'s end.
- Tests: `the_seed_chain_reseals_from_the_first_changed_link_on`,
  `an_unchanged_chain_restores_every_link`,
  `a_seed_too_large_for_the_slot_refuses_with_both_sizes`,
  `a_seeded_conversation_keeps_its_seed_whatever_later_dials_say`,
  `the_seed_dial_off_opens_on_the_files_open_state`; GPU
  `an_opening_state_is_the_seed_end_snapshot_advanced_by_the_close_row`.

## Stage 7 — calibration: the draft and the alternating frames (L)

- `calibrate/run.rs`:
  - units are file records;
  - `calibration_meta` carries the `FileKey` and the frame;
  - the draft is D11's conversation, with `FilesPin::Only` and the target
    state, no fork;
  - `windows.rs` is re-keyed to chunks;
  - in `thoughts`, odd questions take `Empty` and even ones `Only(file)`;
  - `seal_calibration` runs per frame.
- `draft_sight`'s `reading_held` counts the target's members.
  `sections.rs` drops `OWN_FILE` and the link arm.
- `CalibrateSpec` reads the `files` collection (D13); the prompts say the
  file is in the knowledge base above.
- The substrate's `calibrates(tl) -> Option<FileKey>`; the calibration
  record is keyed by `FileKey`. `settled` reports a file without a
  calibration in `waiting_on`.
- Tests: `questions_alternate_empty_then_target_in_draft_order`,
  `a_calibration_names_its_file_key_and_frame`,
  `a_draft_holds_only_its_file_in_the_knowledge_base`,
  `an_empty_frame_is_the_heading_alone`,
  `a_file_without_calibration_keeps_the_memory_unsettled`.
- Exit: the end-to-end path runs — depth-1 ingest, calibration,
  normalization, a dialogue whose system prompt holds the knowledge base.

## Stage 8 — prompt text, removals, tools, documents (M)

- `projection.yaml`:
  - `history_stance`, `grounding` and `grounding_no_tools` are rewritten per
    §7.2 and pinned by schema tests;
  - the `repo_map` and `code_reading` layers (:726-1182) are deleted;
  - `files_open` and the `files` collection are added, with the kinds
    `readings`, `guides`, `journals` and `folders`, carrying the old groups'
    policies and `calibrate:`.
- Examples: `substrate_inspect`'s faces view and `walk_code_read` are
  deleted; `provenance_probe`, `provenance_calibrate`,
  `calibrate_alignment` and `provenance_gates` are re-keyed (the lab's
  `--calibrations none|empty|target|all`, D12).
- Tools: the MCP `timeline_read` probe and the layer tools are re-keyed.
- Check whether `FORCE_TOOL_SELECTOR` still has a user.
- D17: `UnitKind::File` goes.
- Documents: §10's list, and D1–D18 brought up to date.
- `cargo fmt` and `cargo clippy --workspace --tests --examples -- -D warnings`.
