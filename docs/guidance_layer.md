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
    except: [README.md, group.yaml, journal.yaml, person.yaml]
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
  out of the files' scope, unless its file name is one of `except`. Those are a
  folder's hub files (a journal group's README, `group.yaml`), which match every
  question and answer none. Drawn, with no use-exemplars yet, they took every
  placement and were never read, so they stay with `scopes`. The scan, selection, the working-set share
  (`Substrate::placing_group`) and the members a group emits
  (`working_set_members`) all follow that split, so a unit is placed through one
  group, never both.
- **Its own pool.** `WorkingSetShare::Guidance` is bounded by the working set's
  `guidance_tokens` (40K, eight to ten guides and journal notes at a time), a
  pool beside `budget_tokens` (122K) rather than inside it; the two were sized
  together, so the dialogue's total stays where it was.
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

A turn keeps a `guidance::Tally`. For every round it notes the drawn units the
working set placed (members whose `placing_group` is a drawing group), each
with the fresh score the round's scan gave it (`WorkingSet::fresh_of`), and the
drawn units the round read. At the turn's end it logs, per unit:

- **missed**: read by the turn while not placed — placement failed to offer it;
- **read anyway**: placed, and read in the same turn — placement offered it and
  did not spare the read;
- **relevant**: placed, never read, and the turn's own scans kept matching it —
  its peak fresh score reached the working set's `min_momentum`. This is the
  case placement exists for: the worker's reasoning was on what the unit says,
  and the unit was already in front of it;
- **idle**: placed, never read, and no scan matched it — it took room it did not
  use.

The log line is `zend::guidance` "guidance this turn", with each count and its
paths, and only for a turn that placed or read guidance.

**Why "read anyway" is not a hit.** A placed unit is attended and recalled, yet
the model's own sense of what it has read comes from what its recurrence ran
over (`docs/inline_read_injection.md`): it can judge a placed page unread and
fetch it again. So a read of a placed unit is evidence the placement did not
count as reading, and a high "read anyway" count says placement is spending
room on units the worker will read regardless. "Relevant" is the signal of
placement working. It is a proxy: a matching scan says the reasoning was on the
unit's subject while it was placed, not that the reply used it.

The self-improvement mission reads these lines to retune `guidance_tokens` and
the group's band, and to judge the mined calibration: "missed" should fall as
exemplars accumulate, and "relevant" should rise against "read anyway".

## 4. Journals

Journals reach the same group by the same filter, with two differences:

- **They change continuously.** A journal file's new version is a new
  `code_reading` unit, read when the branch ingest sees it, and its use
  calibration starts empty. The unit's cue seeding (§2.2) covers it until it is
  read.
- **The newest note wins.** A journal is a folder of entries, one file each,
  named `YYYY-MM-DD-HHMM-xxxx.yaml` by the minute it was written (UTC, the
  entry's own `at`). Entries are append-only and name nothing they replace, so
  an older entry and the one that superseded it can both match. The group's
  selection therefore carries a recency term:

  ```yaml
  selection:
    kind: working_set
    share: guidance
    recency: { half_life_days: 30, dated_by: path }
  ```

  `dated_by` names the unit metadata that holds its path (zend's `code_reading`
  units carry it as `path`), so the engine reads a stamp without knowing whose
  units these are. A drawn unit whose path's file name opens with that date stamp has its fresh
  score scaled by `0.5^(age_days / half_life_days)` before it feeds momentum
  (`working_set_fresh`), age measured from the stamp to the scan. A unit whose
  name carries no stamp — every guide — is not scaled: a guide is as current as
  its version. The stamp is read from that metadata, so no unit is re-read and
  the 3,800 journal units already ingested are dated as they stand.

## 5. Build order

1. **Built.** The drawing group: `draws:` on a group, its candidates taken from
   its source, the drawn units leaving the owning group's candidates; the
   `guidance` share and `guidance_tokens`. Tests against the substrate, the
   retrieval scope and the working set.
2. **Built.** Use calibration: a read in a tool round becomes an exemplar of its
   unit, kept per unit up to `MAX_EXEMPLARS`. Tests for the read's file, the
   exemplar text and which exemplars are retired.
3. **Built.** Seeding for units no one has read: the `guidance_prompt`.
4. **Built.** The per-turn counts, logged per turn.
5. The counts split as §3 now gives them — "read anyway" and "relevant" in
   place of hit and idle — from each placed unit's peak fresh score
   (`WorkingSet::fresh_of`).
6. Journals' recency term (§4), and the pool at 40K beside a 122K files budget
   (§1).
