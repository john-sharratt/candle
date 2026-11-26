# The `files` collection on BlackWorld — rollout step 5, 2026-11-26

The knowledge base of `docs/provenance_system_collection.md`, built whole and
rolled out per its §9 on BlackWorld (build `7bd8a8956`, a fresh zend
substrate, `max_depth: 1`). The rollout stops at step 5: attention-only
reading of a dynamically placed file fails, and the fresh control on the same
prompt succeeds — the condition §9 names for stopping.

## The memory

- Ingest: 4 folders and 20 files sealed as members (no decode); on restart
  every member restored from its stream.
- Seed: 12 files, 142,929 bytes, ~37K tokens (each repository's root listing
  and its anchors).
- Calibration: 40 calibrations (20 files × the empty and target frames)
  sealed, none failed, in ~2.5 min. Every draft held its whole file
  (`held` = the file's tokens).
- Settled: `waiting_on: []`.

## Step 5 — the production chat path, tools off, a fresh conversation each

Placements are read point by point from `/v1/conversations/{id}/placements`
(each point's knowledge-base files, recorded since `7bd8a8956`).

| Question | Target placed | Answer |
|---|---|---|
| battle-cities' toolchain | `rust-toolchain.toml` at every point from token 0 | "I don't have the file's literal contents in context" — answered from the README's mention |
| candle's `clippy.toml` | at every point from token 0 | **read**: the `disallowed-methods` list and its header, verbatim |
| yenbot's dependencies | `yenbot/Cargo.toml` at every point from token 0 | "I don't have the contents of `yenbot/Cargo.toml` in my context" |
| `missing-lore.txt` | at every point from token 0 | "I don't have the content … in my context" |
| AGENTS.md on env-var flags (seed) | in the seed | **read**, quoted |
| a deep file not ingested (control) | absent | says it is absent; invents nothing |
| `candle/TODO.md` | not placed | answered — by calling `file_read`, which ran although `tools` was `none` |

Dynamically placed files read: **1 of 4**. Seed files: read. Absent files: no
invention. No reasoning loop (completions 207–1,280 tokens).

Forced inventory (a question naming no file, so the seed alone placed): the
model listed its knowledge base by path and said nothing else was loaded, but
labelled candle's root listing `battle-cities/` and claimed
`battle-cities/Cargo.toml`, not placed. The prompt carried the 12 headings
correctly, in order.

## Sealed vs fresh — the stop

The yenbot conversation's live prompt (86,542 tokens: system prompt, the
dynamic members — `yenbot/Cargo.toml` at piece 23 — the seed, the close) was
sent unchanged as the system message of a passthrough request, prefilled
fresh, with the same question. Fresh, the model reads `yenbot/Cargo.toml` and
lists every dependency, version and feature exactly. Sealed, it says the file
is not there.

The prompt text is the same; what differs is how the members' K/V was made.
A member is sealed behind the prompt up to `files_open`; a dialogue's
recurrent layers start from the seed branch's checkpoint and never run over a
dynamically placed member (§8, §12 D11) — only the attention layers read it.
Fresh, the recurrence runs over the file as it is prefilled.

## Also found

- `tools: "none"` does not stop a turn's tool calls on this path: the `todo`
  question called `file_read` four times and it ran.
- The dynamic knowledge base places files the seed already holds
  (`candle/AGENTS.md`, `candle/ARCHITECTURE.md`, …): the seed's files are
  ingested as ordinary files (§12 D5) and nothing keeps them out of the fill.
- The four-way ranking split (content / empty / target / all) is not run: the
  scoring lab and `/v1/substrate/zend/scan` still read the per-file reading
  conversations this change removed, and are not yet re-keyed to file records.
