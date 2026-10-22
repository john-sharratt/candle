# zend's compounding layers

How the layers above `code_reading` are built: `static_analysis`,
`dependency_analysis`, `architectural_analysis`, `critical_analysis` and
`bug_analysis` per file, and `daily_history` and `dream_log` per day. The code
is `zend/src/layers/`, the declarations are the `ingest:` blocks in
`zend/src/prompts/projection.yaml`, and the design they come from is
`docs/coding_assistant.md` §"Trunk Session — Phased Parallel Analysis".

## Declaring a layer

A layer is built from the layers below it when it declares an `ingest:` block:

```yaml
- name: dependency_analysis
  dials: { thinking_effort: balanced }
  ingest:
    unit: file              # file | day
    after: static_analysis  # the layer each unit builds on
    edges: true             # the answer names the files it depends on
    prompt: |
      What do the structs, traits and functions in `<<path>>` …
```

- **`unit: file`** gives one unit per unit of the `after` layer, all the way
  down to one per `code_reading` file. The prompt can use the placeholders `<<path>>`, `<<repo>>`
  and `<<file>>`.
- **`unit: day`** gives one unit per closed UTC day of dialogue. With no
  `after`, the day's conversations are rendered into `<<transcript>>`. With an
  `after`, the unit builds on that layer's unit for the same day.
- **Thinking** is the layer's own `thinking_effort` dial, the level a dialogue
  turn on the layer would use. Nothing caps a unit's thinking or answer.
- **Tools.** A layer may give its units read-only tools (`tools:` in the
  block); every analysis layer gives `file_read`, `file_list`, `file_search` and
  `file_grep` (`READ_ONLY_TOOLS`). They read the workspace at the commit of the
  unit's file, the catalog is pinned to exactly them, a call of any other is
  refused rather than run, and a layer naming anything else is refused when the
  schema loads. A unit reads for as many rounds as it needs; a closing round
  (after `MAX_TOOL_ROUNDS`, a backstop) bans the tool-call opener so it answers
  in prose. A layer with no tools gets that closing round alone.
- **The spec is part of what a unit stands on.** Each unit records a digest of
  its layer's prompt, tools, thinking and edges; one built under another is
  stale, so changing a layer's question rebuilds it and the layers above.
- **Placeholders** are `<<name>>`, not `{name}`: the whole schema is a template
  whose `{ident}` must be a template variable.

`candle-conversation` carries the block through unread (`LayerSchema::ingest`).
Only zend interprets it.

## A unit is a conversation

Each unit is a hidden conversation, forked from its layer's base, and its
lineage is the unit it builds on. A file's `static_analysis` therefore continues
that file's own reading, `dependency_analysis` continues the static analysis,
and so on up. Each question is asked of a conversation that has already read
the file and every analysis of it below. The lineage is the target group, and
it is selected whole, so the file is always in context.

**The layers compound.** A unit also retrieves the layers of units below it,
for every file on its branches, by provenance. Its own layer stays self-local,
so a unit never retrieves its siblings. This is
`Substrate::set_ingest_sees` / `is_self_local_group`: an append-only target
reads the groups it was given across conversations, and masks every other group
to its own timeline as before. `repo_map` and `code_reading` are given none, so
their units are unchanged. A unit is never offered its own file's units below:
they are its lineage already.

**Each layer below is drawn at its own window.** A unit's own group, its
lineage, is kept whole and asks the budget for nothing. Its turn budget is the
sum of the windows of the layers it draws on, each capped at its own
(`project.rs`, Steps 9–11). Priced from the unit's own 8k window instead, the
lineage took nearly all of it, and the layers below shared what was left: a
fourth layer up drew nothing at all. Which units each layer draws is then the
ordinary belief selection (`default_policy: high_recall_scope`), so a clear
leader can hold back weaker neighbours.

Every unit keeps the projection its own signature selects, as a dialogue turn
does, and logs it (`zend::layers::provenance`): which units of other files it
drew, the best-scored candidates, the scope each layer offered, and the tokens
per group.

## What stands, and what goes

Each unit records:
- the key of the unit it was built on (`after`);
- for `edges` layers, the version of every file its file depends on (`deps`);
- the `code_reading` key of its file (`file_key`).

The planner (`layers/plan.rs`) works per layer, per pass:

- **A unit stands** while the unit below it stands and every dependency is at
  the version it recorded, on a branch the unit's file is on.
- **A stale unit is tombstoned at once.** It is not kept until its replacement
  commits, so the layer briefly has no unit for the file rather than an
  analysis of another version. (`repo_map` and `code_reading` keep a stale
  version until the new one commits, because a version a branch held is still
  true of that branch. An analysis of a version is not.)
- **A unit is read only once what it builds on stands.** A file whose reading
  is being redone has nothing to build on in any higher layer until that
  reading commits.

A dependency analysis is keyed by its dependencies too. When it is redone
because one of them moved on, it becomes a new key, and every unit above it
goes stale with it. That is how a change follows the edges: **the same file and
its dependents** are rebuilt, in every layer from the dependency analysis up.

Edges resolve against every file a branch holds, the ones no layer reads
included (`Makefile`, images): an edit to one still reaches the analyses that
name it.

**A unit stands for the branches its dependencies are current on.** One version
of a file is on many branches, and the files it depends on need not be. The
planner judges each unit branch by branch: the one there first takes every
branch it is current on, the next the branches left, and a branch no unit is
current on gets a unit of its own. A unit records its dependencies as the
first of its branches sees them, so it is current on that branch at least and
the splitting ends. A unit that depends on nothing stands for every branch.

Every pass first sweeps every layer, planning and tombstoning without reading,
before the layers below read anything (`sweep_layer`): a unit made stale by
work below goes the moment that work is found.

## Lower layers first

`BranchIngest::pass` walks the layers bottom up. It carries what stands in each
layer (`Ground`) to the next, so a layer is planned against exactly what is
current below it, after that layer has read what it queued.

A pass request that arrives while a pass runs raises `ingest_worker::PREEMPT`.
A layer of units checks it between units and stops, and the next pass starts
from the bottom, so a moved branch reaches `repo_map` and `code_reading` before
any layer above them reads another unit.

The day layers have no branch move to wake them, so the session wakes the worker
every 30 minutes and a day that has closed gets read. That wake does not
preempt: a day closing is no work for the layers below.

## What it costs

- A document's edges are broad: `ARCHITECTURE.md` names about 90 files, so a
  change to any of them rebuilds the document's analyses from the dependency
  layer up.
- Many long-lived branches multiply units: a file gets one analysis per
  distinct set of dependency versions across the branches that hold it.
- The upper layers think at `deep` and `exhaustive`, and read with their
  tools. On a 24 GB card at depth 1, a layer of 24 units takes about an hour
  (dependency) to two and a half hours (bug).
- Every unit inherits its lower units' whole conversations, their tool reads
  included: an upper unit's lineage reaches the inherited-chain cap. On a 64 GB
  host the first tool-using upper layer once failed every unit on a
  driver-level out-of-memory (pinned host memory) and did not on a fresh
  process; host-memory admission is the engine's, and unproven here.

## Tied to a branch

A unit of a file layer carries its file's branches, and its group is scoped the
way `code_reading`'s is. A dialogue retrieves an analysis only when its base
tree holds the file version the analysis is of (`RetrievalScope::with_derived`).
A unit retrieves from the layers below only what stands on one of its own
branches. Day layers are of no branch and are not scoped.

## Testing it small

`--max-depth 1` bounds what is read to each repository's root files, so every
layer can be watched end to end on a workstation in minutes. The `zend::layers`
log target names each layer's plan per pass (standing, stale, queued) and each
unit read.
