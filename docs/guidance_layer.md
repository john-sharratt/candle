# Guidance: guides and journals placed by provenance, calibrated as they are used

A worker's guides (`mcp/src/guides/*.md`) and the journals (the `journals`
repository) are already read: `code_reading` holds a unit for every file of every
workspace repository (`zend::retrieval_scope`), and a dialogue's working set can
place any of them. But they reach a worker only two ways today. Either the worker
calls `guide_read` / `journal_read` itself, which it often does not think to do,
or a guide wins a place among the code files, competing for the same budget as
`.rs` files and scored by calibration questions written for code.

This layer makes guidance a thing of its own. The units are the same
`code_reading` units, faces and all (`docs/zend_working_set.md` §4.9): nothing is
read again. They are selected by **repository and path**, placed under a
**budget of their own**, and calibrated from **the moments workers actually read
them**.

## 1. What it is

A `guidance` group on the `code_reading` layer, declared in the schema with a
filter over the units it draws:

```yaml
- id: guidance
  persisted_id: 13
  draws:
    from: scopes
    paths: [candle/mcp/src/guides/, journals/]
  selection: { kind: working_set, share: guidance }
```

- **It draws, it does not own.** A unit belongs to `scopes` as it always has.
  `paths` are workspace-relative prefixes, and `from` names an earlier
  working-set group of the same layer. The substrate records the pair
  (`Substrate::set_group_draws`). A drawing group holds no conversation of its
  own: its candidates are its source's (`active_timelines_for_group`), as its own
  retrieval scope names them.
- **One group or the other.** `zend::retrieval_scope` splits each target's
  files. A file under one of the `paths` goes into the drawing group's scope and
  out of the files' scope. The scan, selection, the working-set share
  (`Substrate::placing_group`) and the members a group emits
  (`working_set_members`) all follow that split, so a unit is placed through one
  group, never both.
- **Its own pool.** `WorkingSetShare::Guidance` is bounded by the working set's
  `guidance_tokens`, a pool beside `budget_tokens` rather than inside it.
  Guidance dislodges only guidance, and nothing else dislodges it: a guide never
  dislodges a code file, and no burst of code crowds out the guide the worker
  needs. Without `guidance_tokens` no guide enters.
- **The same scope.** A unit is offered only when the conversation's base holds
  its version (`retrieval_scope`), as for every `code_reading` unit. Its own
  scan competition (`cases: file`, per group) keeps a guide from being ranked
  against every `.rs` file.

## 2. Calibration as it runs

A `code_reading` unit is calibrated by drafting questions it answers
(`zend::calibrate`). That suits code, where the probe is a question. A guide is
needed when a worker's own reasoning reaches a point ("this is on main, now I
need it on BlackWorld"), and that rarely reads as a question about the guide.

So a guidance unit is calibrated **from use**:

1. **Every read is a labelled exemplar.** A tool round's read of a unit a
   drawing group draws counts: a `guide_read {name}` (the guide at
   `candle/mcp/src/guides/<name>.md`), or a `file_read` of a file under the
   group's `paths`, such as a journal. The reasoning the turn wrote before the
   call is the state in which that unit was wanted. After the round, on a thread
   of its own (`calibrate_guidance_reads`), zend seals one **use-exemplar** of
   the unit at the version the conversation's base holds (`zend::guidance`).
   - The case's user half is the start of the turn's request (600 characters).
   - Its answer is the end of the reasoning (1024 characters, about a live
     probe's 256-token window), as a closed `<think>` block.
   - It is sealed exactly as a drafted calibration is (`calibrate::seal`): a
     conversation of the hidden calibration layer carrying
     `Substrate::CALIBRATES_KEY` for the unit, archived and distilled
     provenance-only.
   - It is named by `guidance_use` (the unit's content key) rather than the
     drafted calibrations' marker, so a layer's calibration pass neither counts
     it nor retires it.
   - Its signature votes for the unit wherever the unit is scored, exactly as a
     drafted calibration's does.
   - A round that wrote no reasoning ahead of its calls teaches nothing, and
     neither does a read that failed.
2. **The draft seeds a unit no one has read yet.** The layer's `calibrate:`
   block declares a `guidance_prompt`. A unit under any drawing group's `paths`
   is drafted with it instead of the layer's `prompt` (`CalibrateSpec::
   template_for`). It asks for the questions a worker asks itself mid-task at
   the moment it needs the unit, in its own words, never naming the file.
   - The parser keeps questions only, so the draft writes questions.
   - Each question is asked again with thinking on, as every drafted question
     is, so the seed is a worker's reasoning too.
   - The marker digests the template the unit is asked under. A changed
     guidance prompt calibrates the drawn units again, and no code file.
   Use-exemplars join the drafted ones as reads happen.
3. **Bounded.** A unit keeps at most `guidance::record::MAX_EXEMPLARS` (16)
   use-exemplars, the newest by `guidance_used_at`, so a guide read a thousand
   times does not dominate the gallery, and reasoning that has drifted ages out.
   Each seal retires the unit's oldest past that bound, and every use-exemplar
   whose unit is gone. A new version of a file is a new unit, and its use
   calibration starts empty.

Nothing about this is offline. A read in a worker's turn is calibration for the
next turn of every conversation.

## 3. Measuring it

Each placement and each read is already recorded (the working set's marks, the
tool round). Per unit, three counts:

- **hit**: placed, then read or cited in the same turn;
- **missed**: read by the worker while not placed;
- **idle**: placed, and the turn neither read nor cited it.

These are what the self-improvement mission reads to retune `guidance_tokens` and
the group's band, and what decides whether the mined calibration is working: the
missed count should fall as exemplars accumulate.

## 4. Journals

Journals reach the same group by the same filter, with two differences:

- **They change continuously.** A journal file's new version is a new
  `code_reading` unit, read when the branch ingest sees it, and its use
  calibration starts empty. The unit's cue seeding (§2.2) covers it until it is
  read.
- **The newest note wins.** Journal entries are append-only, so an older entry
  and the one that superseded it can both match. That needs a recency term in the
  group's selection, and is not part of this first cut.

## 5. Build order

1. **Built.** The drawing group: `draws:` on a group, its candidates taken from
   its source, the drawn units leaving the owning group's candidates; the
   `guidance` share and `guidance_tokens`. Tests against the substrate, the
   retrieval scope and the working set.
2. **Built.** Use calibration: a read in a tool round becomes an exemplar of its
   unit, kept per unit up to `MAX_EXEMPLARS`. Tests for the read's file, the
   exemplar text and which exemplars are retired.
3. **Built.** Seeding for units no one has read: the `guidance_prompt`.
4. The hit / missed / idle counts.
5. Journals' recency term.
