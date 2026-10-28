# zend's compounding layers

How the layers above `code_reading` are built. zend's schema declares three,
per file and side by side — `dependency_analysis`, `code_review` and
`security_review`, each `after: code_reading` — so an edit to a file makes all
three of its units stale at once and each is read again. The engine also builds
layers per day (`unit: day`), which zend's schema declares none of. The code is
`zend/src/layers/`, the declarations are the `ingest:` blocks in
`zend/src/prompts/projection.yaml`, and the design they come from is
`docs/coding_assistant.md` §"Trunk Session — Phased Parallel Analysis".

## Declaring a layer

A layer is built from the layers below it when it declares an `ingest:` block:

```yaml
- name: security_review
  dials: { thinking_effort: balanced }
  ingest:
    unit: file              # file | day
    after: code_reading     # the layer each unit builds on
    edges: false            # true: the answer names the files it depends on
    expires_days: 30        # read again within this many days
    tools: [file_read, file_list, file_search, file_grep, web_search]
    prompt: |
      Review `<<path>>` in the `<<repo>>` repository for security. …
```

- **`unit: file`** gives one unit per unit of the `after` layer, all the way
  down to one per `code_reading` file. The prompt can use the placeholders `<<path>>`, `<<repo>>`
  and `<<file>>`.
- **`unit: day`** gives one unit per closed UTC day of dialogue. With no
  `after`, the day's conversations are rendered into `<<transcript>>`. With an
  `after`, the unit builds on that layer's unit for the same day.
- **Thinking** is the layer's own `thinking_effort` dial, the level a dialogue
  turn on the layer would use. Nothing caps a unit's thinking or answer.
- **Tools.** A layer may give its units tools (`tools:` in the block) that
  change nothing. The read-only tools — `file_read`, `file_list`,
  `file_search` and `file_grep` (`READ_ONLY_TOOLS`) — read the workspace at the
  commit of the unit's file. The research tool — `web_search`
  (`RESEARCH_TOOLS`) — searches the web, for a layer whose answer depends on
  what is known outside the branches. The catalog is pinned to exactly the layer's
  tools, a call of any other is refused rather than run, and a layer naming
  anything else is refused when the schema loads. A unit reads for as many
  rounds as it needs; a closing round (after `MAX_TOOL_ROUNDS`, a backstop)
  bans the tool-call opener so it answers in prose. A layer with no tools gets
  that closing round alone.
- **A researching unit gets the network and nothing else.** Ingest runs under
  Restricted's grants, which hold no capability; a unit of a layer naming a
  research tool runs with `Capability::Network` added and nothing more
  (`run::unit_grants`), so it can reach the web and still cannot write, run a
  program or read a credential. Restricted's call grammar does not compile the
  network tools, so such a unit decodes under Comprehensive's
  (`LayerSpec::tool_mode`) — its prompt still lists only its own tools.
  `web_search` also needs the daemon's `tavily_api_key`; without one it answers
  with an error the unit reports.
- **Search, never fetch.** A unit runs unattended over whatever the branches
  hold, so anyone who can push a branch writes text it reads. Given the
  read-only tools and `web_fetch` together, an instruction planted in a file
  could read the workspace and send it out in a URL to any public host.
  `web_search` sends only a query, and only to the configured search provider,
  so a unit has no channel to a host of the text's choosing; a layer naming
  `web_fetch` is refused when the schema loads.
- **Expiry.** `expires_days: N` makes each unit stand at most `N` days though
  nothing it was built on moved — see [Units that expire](#units-that-expire).
- **The spec is part of what a unit stands on.** Each unit records a digest of
  its layer's prompt, tools, thinking and edges; one built under another is
  stale, so changing a layer's question rebuilds it and the layers above. The
  expiry is not in the digest: changing it moves when units expire, and
  rebuilds none.
- **Placeholders** are `<<name>>`, not `{name}`: the whole schema is a template
  whose `{ident}` must be a template variable.

`candle-conversation` carries the block through unread (`LayerSchema::ingest`).
Only zend interprets it.

## A unit is a conversation

Each unit is a hidden conversation, forked from its layer's base, and its
lineage is the unit it builds on. A file's `dependency_analysis`, `code_review`
and `security_review` each continue that file's own reading; a layer declared
`after` one of them would continue that, and so on up. Each question is asked
of a conversation that has already read the file and every analysis of it
below. The lineage is the target group, and it is selected whole, so the file
is always in context.

**The layers compound.** A unit also retrieves the layers of units declared
before it, for every file on its branches, by provenance: a `security_review`
unit can draw on other files' dependency analyses and code reviews. Its own
layer stays self-local, so a unit never retrieves its siblings. This is
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

**A unit stands as one exchange.** A unit is read in many turns — its
question, each round of tool calls and what they answered, the reasoning
between them — and what it stands for is its last answer. Once it has answered
it is written again (`layers::compact`) as a conversation of one exchange,
minted exactly as the reading was (same base, lineage, retrieval scopes and
early tags):

- the user turn is a short question naming the layer and the subject —
  `Code review: candle/zend/src/rate.rs` — not the layer's prompt;
- the assistant turn is the answer, its reasoning set aside, prefilled as a
  no-reasoning content turn (`Sequence::insert_turn_staged`), not decoded.

That conversation is committed — content key, dependencies, calibration — and
the reading is tombstoned before it is minted, so a worker holds one
conversation at a time. A unit that cannot be written again is not committed
as its reading: both go, and it is read again next pass. What the substrate
keeps of a unit is the answer's own K/V, and what a unit above inherits as
its lineage is each unit below's answer, not its reading.

A unit's subject names it in the substrate view, as a file's path names its
reading.

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

**What a pass cannot see, it does not judge.**
- **A held-back repository:** its tree could not be listed this pass, so its
  units stand untouched, and a dependency on one of its files counts as
  current. No layer that names edges reads that pass, since an edge into the
  held-back repository could not be keyed.
- **A layer below that was not planned:** it is disabled, or the pass stopped
  short of it. The layer above is left as it is rather than read as empty.

**An answer is only an answer when it finished.** A unit whose thinking never
closed, whose turn spent its budget, or whose closing round still called a
tool is a failure, read again next pass. Each unit is tried once a pass.

**Dependencies are the unit's branch's.** A name resolves first inside the
unit's own repository. It counts as an edge only if the branch the unit is
recorded for holds that file. Numbered items, Markdown links and `path:12`
forms are read as well as bullets.

## Lower layers first

`BranchIngest::pass` walks the layers bottom up. It carries what stands in each
layer (`Ground`) to the next, so a layer is planned against exactly what is
current below it, after that layer has read what it queued. A layer reads until
nothing is queued in it before the layer above reads: a unit that splits leaves
branches it is not current on, and their units are read before anything above.
It stops early when a pass is asked for, and when a read leaves its plan
unchanged (a unit that would not commit is tried again next pass).

**The default branch goes first, all the way up.** The pass climbs the layers
twice. The first climb reads only units on a repository's default branch
(`main`, else `master`) and units of no branch (the days). The second climb
reads every other branch's units, bottom up again. Lower layers still go
first on every branch, but `main` is analysed to the top before any other
branch's units are read in any layer (`layers::Reach`).

A pass request that arrives while a pass runs raises `ingest_worker::PREEMPT`.
A layer of units checks it between units and stops, and the next pass starts
from the bottom, so a moved branch reaches `repo_map` and `code_reading` before
any layer above them reads another unit.

A layer that changes with time rather than with its branches — a day layer,
whose day closes at midnight, or a layer whose units expire — has no branch
move to wake it. While a layer passed this boot is either, the session wakes
the worker every 30 minutes (`LAYER_CLOCK_EVERY`). That wake does not preempt:
a day closing or a unit expiring is no work for the layers below.

## Units that expire

`security_review`'s answer depends on what is known against what a file uses,
which changes while the file does not. With `expires_days: 30` each of its units
is read again about every thirty days (`layers/expiry.rs`).

**Each file has its own phase, so the layer never churns at once.** Read
together, a layer's units would all expire together. Instead the period is cut
at boundaries offset by a digest of the file's path, and a unit stands until a
boundary of its file after it was built:

- the first units of a layer, built together, expire spread evenly across the
  period;
- read again at its boundary, a unit stands exactly one period, so once the
  layer is under way each file is read again every thirty days and the layer
  reads a thirtieth of its files a day;
- a unit stands at least a thirtieth of the period (a day of thirty): one
  rebuilt just before its boundary — because its file was edited — skips that
  boundary and stands to the next, rather than being read again within the hour;
- the phase is a digest, not a random draw, so every member of a cluster finds
  the same unit expired at the same time and builds it once.

A unit's age is its first turn's wall time, as the substrate records it for
every turn; nothing extra is written.

**An expired unit stands until its replacement commits.** Unlike a unit whose
file moved on, it is not wrong, only old: tombstoned first, a pass that stopped
or a read that failed would leave the file with no unit at all. The planner
keeps it behind every unit still in its time and queues the file to be read
again; once the replacement commits it takes the old unit's branches, and the
old unit, standing for none, goes. Each reading of an expiring unit is keyed by
the term it is read for — the boundary it will expire at — so the replacement
is a unit of its own key, never one the expired unit already holds, and every
member reading within one term leases the same job.

An edit to the file still rebuilds its unit immediately, as for any layer.

## What it costs

- A document's edges are broad: `ARCHITECTURE.md` names about 90 files, so a
  change to any of them rebuilds the document's analyses from the dependency
  layer up.
- Many long-lived branches multiply units: a file gets one analysis per
  distinct set of dependency versions across the branches that hold it.
- An edit to a file reads three units of it, one per layer, each at
  `balanced`. Before the five-layer stack was removed, on a 24 GB card at
  depth 1, the dependency layer's 24 units took about an hour.
- `security_review` searches the web: up to four searches per unit, as its
  prompt asks, on every read. Its first build searches for every file; after
  that, a thirtieth of its files a day plus each edited file.
- Every unit inherits its lower units as their one exchange each — question
  and answer, no tool reads or reasoning — on top of the file's reading. Before
  units were written again that way, an upper unit inherited whole readings
  and its lineage reached the inherited-chain cap; on a 64 GB host the first
  tool-using upper layer once failed every unit on a driver-level out-of-memory
  (pinned host memory).
- Writing a unit again costs one prefill of its question and answer after its
  reading, on the same worker.

## Tied to a branch

A unit of a file layer carries its file's branches, and its group is scoped the
way `code_reading`'s is.

**What a dialogue retrieves** (`RetrievalScope::with_derived`): an analysis is
in scope when the base tree holds both the file version it is of and the
versions of the files in the same repository that it depends on. It leaves
scope when the conversation changes its file or one of those dependencies.

**What a unit retrieves from the layers below:**
- In its own repository: what stands on one of its own branches.
- In another repository: what stands on that repository's default branch. A
  branch name means nothing across repositories.
- For a day unit, which is of no branch: what stands on each repository's
  default branch.

A unit's own lineage is never offered back to it as a candidate.

**Engine side.** A unit's projection record is one read-only scan, which
teaches no score level that dialogue retrieval reads. Only a layer that
compounds on a layer below draws that layer's window into its budget.

## Testing it small

`--max-depth 1` bounds what is read to each repository's root files, so every
layer can be watched end to end on a workstation in minutes. The `zend::layers`
log target names each layer's plan per pass (standing, stale, queued) and each
unit read.
