# Tool calibration

How a daemon teaches provenance which tool a question is for. One phase, in
`host::calibration`, run at load by every tenant that offers tools: `zend` over
its tool files, `npcd` over its act catalog.

## 1. Why

A daemon's tools are one collection of its system prompt. A turn shows a few of
them, chosen by the belief scan: each member is scored by the turns that carry
its tag, against what the turn opens on. A tool with no such turns has nothing
to be scored by, the gallery is empty, and selection falls back to catalog
order — the log line `belief gallery EMPTY … tool selection falls back to
catalog order`. That is what npcd ran on: 208 acts ranked by nothing, the same
ten shown whatever was asked.

Calibration writes those turns ahead of time, from what the tool's author wrote
down.

## 2. What a tool gives

`host::calibration::Calibrated`:

| | |
|---|---|
| `name` | the tool's member name in the live `tools` collection — the tag its turns carry |
| `line` | the section the model reads when choosing it |
| `examples` | trajectories: a prompt, the assistant header, think → call, with `<\|projection\|>` at each capture point; or a bare prompt |
| `questions` | plain ways the tool is asked for — no reasoning, no answer |

Routing happens on the question. A trajectory is mostly reasoning and call, so
its question is a few percent of it; a question exemplar is only that part,
which is why a tool carries many of them.

| Daemon | Definition | Where |
|---|---|---|
| zend | `ToolDef`, one YAML file a tool | `zend-tools/definitions/*.yaml` (`zend_tools::definitions`), `zend/src/tool_def/mod.rs` |
| npcd | `ActDef`, built from each `Tool` | `npcd/src/engine/calibrate/` |

An npcd act's `Example { situation, call, because }` is rendered as a
trajectory — the situation, `because` as the reasoning, the call — and its
situation is a question besides. Further questions are in
`calibrate/questions.rs`, keyed by act.

## 3. The phase

`host::calibration::run`:

1. **Cases.** One per trajectory; one *group* per tool holding all of its
   questions (and its bare prompts), prefilled together and sealed a turn each.
2. **Marker.** Each case has a marker: the tool's name and a digest of its
   `line` and the case's text. Rewording a description, a parameter, an example
   or a question changes it.
3. **Retire.** A calibration conversation whose marker is no current case's is
   tombstoned, so it stops answering the scan.
4. **Resume.** A case whose conversation is archived is done. One that is
   tagged but not archived was cut off: tombstoned and run again.
5. **Run.** Every remaining case is prefilled — not decoded — on the hidden
   `Reserved::Calibration` layer, under a projection that holds the whole
   catalog and shows the one tool the case pins. Its turns are tagged
   `tool` and the tool's name, kept lossless until the seal captures the wide-Q
   signature, never written cold, then archived and distilled to provenance
   only. Sixteen cases in flight, refilled eight at a time; the first runs alone.

The daemon supplies the frame the tool is shown inside (`Frame`), its progress
callback, and its stop.

## 4. The live collection

The live `tools` collection names the tag in its policy, as zend's does:

```yaml
policy:
  preset: committed_tool_scope
  tags: [tool]
  question_pin: true
```

npcd offers a turn exactly the acts its grammar allows and shows the ten of
those that rank highest (`SelectionRule::Offered`), with the budget's floor at
ten: an act is ranked, never gated out. An act offered in one place only is
shown wherever it is offered, whatever it ranks (`tools::shown_wherever_offered`).

## 5. Cost

The first start after a definition changes calibrates the changed cases; every
later start finds them archived. A from-empty npcd catalog is a few hundred
short prefills. Run that first start outside keep's ten-minute trial if the
whole catalog is new.
