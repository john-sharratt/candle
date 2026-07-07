# Tool-provenance distillation

The calibration corpus is stored as full conversation turns — trajectory tokens,
KV chunks, **and** the per-turn wide-Q signature (`WideQSig`). But only the
signature is ever read at runtime: tool selection scores the `tools` collection's
belief gallery (`Resolver::belief_gallery`), which reads each in-scope turn's
`wide_q_sigs` and never touches its tokens or KV. And the trajectories are
redundant on disk anyway — the exported `zend-tools/examples/*.md` files are the
source of truth the prefill-calibration path reads.

So once a calibration turn's sig is captured, its content is dead weight.
**Distillation** reclaims it: keep the `StreamDecl` (its tags + gallery scope) and
the `WideQSig`, drop the tokens and KV chunks. A distilled calibration turn is a
few hundred bytes instead of a full trajectory's worth of KV.

## Mechanism

A **per-timeline `Distilled` marker**, mirroring the `Tombstone` pattern: a
substrate set + a `RecordType::Distilled` redo-log record, reached by compaction.
`Tombstone` = "drop everything"; `Distilled` = "keep sig + decl, drop content".

1. **The marker.** `Conversation::distill_timeline(timeline)` (and the
   `ConversationEngine::distill_timeline` wrapper) writes a `Distilled` record and
   sets the in-RAM `Substrate::distilled_timelines` bit. `is_timeline_distilled`
   queries it. Idempotent — duplicate markers replay identically.

2. **Marking calibration material.** The calibration loop
   (`zend/src/session.rs`) marks a conversation for distillation in **both**
   paths, each gated on `is_timeline_distilled` so an already-distilled timeline
   is never re-marked:
   - **Seal path** — right after archiving a freshly-completed case.
   - **Resume path** — when a prior run's case is found already archived (it's
     *skipped*, so this is the only place existing corpus gets marked).

   This is the step that was missing: without the resume-path mark, calibration
   material already in the substrate from a prior run would never be distilled.

3. **A compaction branch.** `collect_live_records`
   (`persistence/compaction.rs`) already emits each record type of a turn
   independently. For a turn whose `timeline_id` is in the distilled set, it emits
   the `StreamDecl` + `WideQSig` (and `ProjectionEvents`) but **skips the KV
   chunks and the `Tokens` record**. Tombstoned timelines still drop entirely.

Blanket rule: every calibration conversation is marked distilled, so the whole
corpus becomes sig-only after a compaction pass. The turn tags are untouched —
the gallery still finds each sig via its `["tool", <name>]` tags.

## Why the sig is unchanged (and correct)

The retained `WideQSig` is exactly the sig captured during calibration — nothing
about it changes. It is the same sig the belief gallery scores today; we simply
stop persisting the content behind it. There is no re-capture, no clean-vs-dirty
context question — the turn is identical, minus its (unused, redundantly-stored)
trajectory.

## Load-cycle behaviour (auto-compaction, no loop)

Compaction is where distillation is *realised*, so the loader runs it
automatically — but only when there's something to reclaim, and in a way that
provably terminates:

- **Auto-trigger** — after the substrate is loaded, compaction runs when it's
  enabled (on by default; opt out with `--no-compact-substrate`) **and**
  `engine.substrate_has_reclaimable()` (the loaded substrate holds tombstoned or
  distilled timelines). A clean reload with no markers skips it entirely.
- **Markers are consumed** — `collect_live_records` never re-emits `Tombstone` or
  `Distilled` records, so after a compaction pass those sets are empty on the next
  reload → the auto-trigger doesn't fire again.
- **Marking is KV-gated** — a calibration timeline is distill-marked only if it
  `timeline_has_kv()` (content still present) and isn't already marked. Once
  compaction has reclaimed a timeline's KV, it's never re-marked — so no new
  `Distilled` record is written, and compaction won't re-trigger.

Together these give a self-terminating cycle:

| Startup | Loaded state | Compact? | Calibration marks |
|--------|--------------|----------|-------------------|
| 1 (fresh) | no markers | skip | marks each case (has KV) |
| 2 | has `Distilled` markers | **yes** → reclaims KV, consumes markers | cases now have no KV → no marks |
| 3+ | no markers | skip | no KV → no marks |

Reaches steady sig-only state after one reclaim pass; never loops.

- **Reload after compaction** — compaction rewrites the log, and while
  `p.compact` rebuilds the stream index, the scheduler-side KV residence is not
  walker-built. So the loader calls `engine.reload_substrate()` (a
  `ReconstructSubstrate` scheduler request re-running `reconstruct_substrate` on
  the scheduler thread) and blocks on it, rebuilding all offsets / KV pointers
  from the compacted log before serving.

## What survives a distilled corpus

- **Kept:** `StreamDecl` (tags, gallery scope), `WideQSig`, `ProjectionEvents`,
  the conversation's `ConvState`/`Label` (archive flag, resume marker).
- **Reclaimed:** `Tokens`, KV `Chunk` records.
- **Consequence:** `calib-check` (which decodes turn tokens to audit tool-calls)
  cannot run against a distilled corpus — it must run before distillation or on a
  fresh rebuild. The prefill trajectories remain available in the `.md` files.
