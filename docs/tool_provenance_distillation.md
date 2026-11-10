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

## Load-cycle behaviour (no loop)

Compaction is where distillation is *realised*. Two properties keep that from
looping — and neither is "the marker goes away", which is the opposite of what
happens:

- **Markers SURVIVE the compaction that acts on them.** Both `Tombstone` and
  `Distilled` are re-emitted from the substrate's live state by
  `collect_live_records` and by the incremental `gather_resident_set`, and they
  have to be: the `Distilled` marker is what exempts a distilled turn from the
  `MissingKv` integrity verdict. A distilled turn has legitimately shed its
  content, so a *consumed* marker would leave the next reload condemning the
  very corpus the distillation created. See `persistence/survival.rs` for the
  full record-type contract.
- **The trigger is dead weight, not markers.** `should_compact` fires on
  `total_record_bytes >= COMPACTION_MIN_LOG_BYTES` **and** `dead_ratio >=
  COMPACTION_DEAD_RATIO_THRESHOLD` — a liveness-derived fraction that *falls* as
  the pass reclaims. So a pass that reclaims makes the next one less likely,
  regardless of how many markers are standing.
- **Marking is once per timeline** — a calibration timeline is distill-marked
  unless it is marked already (`is_timeline_distilled`), and the marker is
  resident, so a reclaimed timeline is never re-marked and no second
  `Distilled` record is written.
- **Marking is not gated on K/V reaching disk.** An exemplar is transient
  (`mark_timeline_transient`), so none of its chunks is ever written cold: a
  gate on on-disk chunks was always false and no exemplar was ever marked.
  Each was then kept archived with its tokens and a declaration claiming K/V,
  and every other member held its turns as still arriving
  (`docs/replicated_substrate.md` §6.5) — 2,462 on BlackWorld, ~85K on
  WhiteWorld (2026-11-11). A layer's calibrations (`zend/src/calibrate/seal.rs`)
  carried the same gate: 105,679 of BlackWorld's own turns in the calibration
  layer held no chunk and no marker. The marker is what completes them there:
  applying a remote `Distilled` record makes each of its timeline's turns a
  candidate for adoption. What is already sealed is marked as it is found —
  the tool calibration's resume filter its archived exemplars, and every
  layer pass first every archived calibration in the substrate, of any layer
  (`calibrate::mark_sealed`), since a layer that no longer declares
  `calibrate:` runs no pass of its own — so a store sealed under the gate is
  repaired at its next load.

Together: the marker set is monotonic (it only grows, one marker a timeline),
and the trigger is a ratio that the reclaim itself drives down.
Reaches a steady sig-only state after one reclaim pass; never loops.

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
