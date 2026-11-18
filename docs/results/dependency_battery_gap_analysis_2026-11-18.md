# Battery gap analysis: the scoring lab against production (2026-11-18)

The scoring lab (`provenance_scoring_lab_2026-11-16.md`) put the §9.12 dependency battery's
answer files at MRR 0.70 / hit@10 0.78. The same formula went live on the code-reading groups
(010405b50). The battery on production then scored 46.2% mean recall
(`dependency_battery_2026-11-18.md`). This note finds where the two diverge. For each of the 24
questions, it reads the prompt production actually built through
`GET /v1/conversations/<id>/prompt-tokens` and sets it beside the answer.

**Method.** BlackWorld, builds 5184 → 5195 (e29848d02). One private Zen chat per question, run
one at a time, each with the battery's standard prefix ("Answer from your memory of the
repository only — do not call any tool …"). The prompt dump was taken after the answer, and
the chat closed before the next question. The answer files are the battery's `files` lists
(54 files over 23 questions; a4 names none).

## Headline

The 54 answer files split three ways:

| Fate of the answer file | Files | Share |
|---|---:|---:|
| Placed in the prompt | 28 | 52% |
| Can never be placed: over the 48K `max_file_tokens` cap | 12 | 22% |
| Retrieval miss: under the cap, not drawn | 14 | 26% |

The 28 placed files split again:

| Placed answer file | Files |
|---|---:|
| Used in the answer | 17 |
| Placed and ignored: the answer says it never saw the file, or does without it | 10 |
| Ambiguous (the answer is equally AGENTS.md's text) | 1 |

So the lab's number measures one stage of three:

1. Retrieval would find the answer.
2. The file fits the placement.
3. The model reads what was placed.

Production loses about a fifth of the answer files at stage 2. It loses about a third of the
files that survive to stage 3.

## The gaps, one by one

### 1. Whole-file placement under a per-file cap (structural; the lab cannot see it)

The working set refuses any file over `max_file_tokens` (48,000) outright
(`candle-conversation/src/working_set/state.rs:261`). Five battery answer files are over it at
about 20 tokens per line:

| File | Lines | Questions |
|---|---:|---|
| `candle-conversation/src/scheduler/mod.rs` | 18,692 | d3, d8, t6, t8 |
| `candle-core/src/quantized/cuda.rs` | 8,177 | d5 |
| `candle-conversation/src/conversation.rs` | 6,371 | d1, t2, t6 |
| `candle-nn/src/kv_cache/chunked/region_pool.rs` | 5,369 | d2, a2 |
| `candle-conversation/src/scheduler/projection_assembler.rs` | 4,409 | t7, a3 |

t7 and a3 have no other answer file, so they cannot pass. The lab ranks these files like any
other and counts a top-10 rank as a hit.

### 2. A file budget, not a file count

The lab's recall is "an answer file in the top `budget_files` (10)". Production places whole
files into the scopes group's share of a 64K working set, about 52–62K tokens. That is 4–14
files per prompt here, median 7. One 37K file (`expert_lre/cache.rs`) leaves room for about
five more.

### 3. The probe's text differs: the battery prefix draws two hubs into every prompt

The lab probes the bare question (`lab_probes.rs:120`, `{"text": question}`). Production probes
the whole user turn, prefix included. The prefix ("answer from your memory … do not read or
search any file. If you do not remember, say so") reads like `yenbot/src/ai/ask.rs`'s header:
"Asking about an app's own data … is search first, then the model".

- `ask.rs` was placed in **all 24** prompts, at rank 1–4.
- `yenbot/apps/lifecycle/stores/assistant.js` was placed in 22 prompts.
- Together they took 11.0% of all scopes tokens (159K of 1.45M).

The lab's ranking never had them.

### 4. Placed but not read (the reading stage, which the lab has no stage for)

Ten placed answer files went unused. The answers fail in three recurring ways:

- **Says it never saw a placed file.**
  - **d4.** `gpu.rs` #1, `window_score.rs` #2, `scan.rs` #3 and `packed.rs` #5 were all
    placed. The answer quotes `gpu.rs` and says it saw no `scan.rs` or `window_score.rs`.
  - **d7.** All three files were placed: `cache.rs` holding `PINNED_LAYERS: usize = 2`,
    46K tokens before the end, plus `layer_loader.rs` and `expert_lre/mod.rs`. The answer
    says it never saw the definition.
  - **a6.** `wide_sig.rs` was placed #1 at momentum 157K. t1 answered from the same file in
    depth: `fold_admits`, and `decode_wide_sigs_checked` refusing with `warn_fold_mismatch`.
    a6 ignores it and reasons from a "fixed hash" story instead.
- **Uses the content and credits the wrong file.**
  - **d7.** It quotes `layer_loader.rs`'s `PINNED_LAYERS.min(cfg.num_layers)` as `cache.rs`.
  - **t5.** It gives the constant, `batch_victims` and `retract_zone`, all from `cache.rs`,
    as "from `pad.rs`, the only file I've read".
  - **t1.** It uses `scan.rs`'s kernel and says it read only `wide_sig.rs`.

  For a "where is X defined" question this is a miss even when the content is right. All
  eight direct questions are that kind, which is a large part of why the direct category
  scores 24%.
- **Names files it read that are not in the prompt.**
  - **a4.** "I read the full `turn_placement.rs`": it is not in the prompt at all.
  - **d7.** "I read only `research.md`": a guidance item 75K tokens back.
  - **a7.** It names `chat_frames.rs`, which is not among its scopes.

Prompt position does not explain which placed files are used. In t4, t8, t6, t3 and t5, the
file placed directly after the leading hub was used heavily. In d4, d5, t2 and a6 the file
in that slot was ignored. Placed file pieces carry the same `file_read` tool-call and response
framing as every other read. This is the 11-12 hypothesis
(`model_not_seeing_context_2026-11-12.md`): what the model believes it has read is decided by
the recurrent state, not by the K/V. Its account of "what I read" is evidence about that
state, not about the prompt.

### 5. Constant overhead the lab does not carry

These are the same in every prompt, whatever the question:

- The priming chain (12 READMEs, ARCHITECTURE.md and AGENTS.md files, ~55K tokens).
- The guidance group (~14K tokens at `score_threshold 0.0`). In all 24 prompts it held
  unrelated guides and journal entries: `ab-trial.md`, `delegate.md`, Battle Cities canon,
  the family story.

The priming chain is not waste. The questions that scored full marks without their files
(a2, a5, a8, much of t3) were answered from AGENTS.md and ARCHITECTURE.md.

## Per question

Rank `#n` is the file's rank by momentum among the prompt's scopes files. `CAP` means over
`max_file_tokens`; `—` means not drawn. Recall is hand-scored by the battery's substring rule
on this run's answer.

| Q | Answer files | Recall | What the answer did |
|---|---|---:|---|
| d1 | recorded_reply.rs #2, conversation.rs CAP | 0/2 | Declined; said it read `store_tests.rs` |
| d2 | span_geometry.rs #2, region_pool.rs CAP | 0/2 | Declined; cited AGENTS.md, `compact_mint.rs` |
| d3 | elevate.rs #1 (17.6K), scheduler/mod.rs CAP | 0/2 | Said it read `transfer.rs` (#4); `elevate_to_hot` "only in prose" |
| d4 | gpu.rs #1, window_score.rs #2, scan.rs #3, packed.rs #5, resolver.rs —, belief_files.rs — | 5/6 (1 real) | Used `gpu.rs`; named the rest as unseen |
| d5 | dispatch.rs #1, dspark_experts.rs #2, moe_bucketize.cu —, cuda.rs CAP | 2/4 | Used `dispatch.rs`; never mentioned `dspark_experts.rs` |
| d6 | cache.rs #1, pipeline.rs — | 1/2 | Definition right with its part and lines; caller unknown |
| d7 | cache.rs #2, layer_loader.rs #3, mod.rs #6 | 1/3 | Quoted `layer_loader.rs` as `cache.rs`; denied seeing the value |
| d8 | wide_sig.rs —, scheduler/mod.rs CAP | 0/2 | Drew `summary_tree` for "fold"; reported it accurately |
| t1 | wide_sig.rs #1, scan.rs #2, window_score.rs — | 3/4 | Deep and correct; credited it all to `wide_sig.rs` |
| t2 | replay.rs #1, recorded_reply.rs #2, turn.rs #7, conversation.rs CAP | 2/4 | Traced `replay.rs`; said the extractor was never read (it was, #2) |
| t3 | arena_table.rs #1, arena_table.cuh #5, types.rs —, blocks.cuh — | 4/4 | From the tables and ARCHITECTURE.md §4.1 |
| t4 | dispatch.rs #1, moe_bucketize.cu #3, stager.rs — | 4/4 | Full trace from `dispatch.rs` |
| t5 | cache.rs #1, startup.rs — | 2/3 | Correct from `cache.rs`, credited to `pad.rs` |
| t6 | config.rs —, conversation.rs CAP, scheduler/mod.rs CAP | 0/4 | Drew `projection/schema.rs`; denied `continuous_fair_waves.md` (#4) |
| t7 | projection_assembler.rs CAP | 0/3 | Nothing relevant drawn; top momentum 27K (`ask.rs`) |
| t8 | elevate.rs #1, scheduler/mod.rs CAP | 3/3 | Correct, credited correctly |
| a1 | lib.rs #2, types.rs #3 | 3/3 | Correct |
| a2 | span_geometry.rs —, region_pool.rs CAP | 2/2 | From AGENTS.md and `vram_span_partition.md` |
| a3 | projection_assembler.rs CAP | 0/3 | Drew `models/builder.rs` |
| a4 | (none) | 0/2 | Named `turn_placement.rs` as read: not in the prompt |
| a5 | dispatch.rs #1, moe_bucketize.cu — | 2/2 | AGENTS.md invariant 3, nearly verbatim |
| a6 | wide_sig.rs #1 (157K) | 1/2 | Ignored the file; reasoned from a "fixed hash" |
| a7 | turn_layout.rs — | 1/3 | Nothing relevant drawn |
| a8 | segmented_log.rs #2 | 2/2 | Correct, with segment detail from the file |

Hand-scored mean recall is ≈50% with 7 full marks, against 46.2% and 6 earlier today. The two
runs are within the run-to-run variance already recorded (t6 scored 0/4 concurrent and 2/4
alone).

## What follows

Each gap is a separate fix, and the lab should measure the first three before production
does:

- **Placement by part, not by whole file**, so a file over the cap is placed as the parts
  that answer. This alone reopens 12 answer files and two questions that cannot pass today.
- **A token budget in the lab**: recall as "an answer file within the placed tokens",
  applying `max_file_tokens`, instead of "in the top 10".
- **The lab probes what production probes**: the user turn as sent. Alternatively, the
  battery's prefix is moved out of the turn the retrieval reads. The hubs it pulls in are an
  artefact of the battery, not of the questions.
- **The reading stage**, 10 of 28 placed answer files ignored, is not a retrieval problem, and
  no change to the scoring formula reaches it. It is the recurrence question the 11-12
  hand-over left open.

The dumps behind this note were taken from production. They are not kept in the repository.
