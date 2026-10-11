# npcd operation workflows

An operation is a piece of work the command table holds: a story to write, a
correction to make, a repair. A **workflow** sets out how that work goes: its
steps, who takes each one, what each is told, what each does to the document,
and where each result leads. Workflows are written in a mind's
`missions.yaml`, next to the generators that propose work and the shared
prompts. The engine does not know what any step is for, so a story pipeline
and a repair job run on the same machinery with different YAML.

Code: `npcd/src/engine/workflow/`. Today's pipeline, written in this format,
is in [`npcd_workflows_current.yaml`](npcd_workflows_current.yaml). It is the
parse target, and a test loads it.

## Format

```yaml
keep: 4                    # waiting missions the table is kept stocked with

prompts:                   # named texts any prompt may include
  verdict: |
    Then give your verdict at the table…

generators:                # what proposes work
  - id: untold
    call: story            # the table call that proposes
    workflow: story        # the workflow an accepted proposal opens
    weight: 1              # how often it is drawn (default 1)
    context: [era, timeline]
    prompt: |
      Choose one moment the era passes over…

workflows:
  repair:
    send-backs: 3          # default 3
    desk: workshop         # optional: where the work is done
    year: now              # optional: when the work is done
    on-failed: discard     # discard | keep (default discard)
    steps:                 # in order; the first is the start
      fix:
        by: maker
        edits:
          start: new
          default: change
        next: inspect
        stuck: failed      # default failed
        tools: [file_edit]
        prompt:
          start: |
            Fix {objective}.
          default: |
            Fix {objective} again. The inspection found: {findings}
      inspect:
        by: table
        call: inspection
        next:
          sound: publish
          faulty: fix
        prompt: |
          Inspect the fix of {objective}. {verdict}
      publish:
        by: maker
        desk: merge-table  # this step's own desk, over the workflow's
        edits: optional    # only to resolve a conflict
        tools: [merge_status, merge_sync, merge_publish]
        checks: [published]
        next: done
        prompt: |
          Land {objective} in the record.
```

**Each step works in its operation's store.** An operation owns one store over
the mind repository (`docs/replicated_substrate.md` §10.3). Every step's
mission reads through it and commits into it, so a writer's draft is in front
of the reviewer and the checker of the same operation and of no one else, and
nothing reaches the record until the publish step lands the store there.
`on-failed` says what becomes of a failed operation's store: `discard` drops
it, and `keep` leaves it, unpublished, for an operator to reopen.

Other top-level keys belong to other readers and are ignored. Inside these
sections, an unknown key is a load error.

### Step keys

A step has exactly these keys:

| Key | Meaning |
|---|---|
| `by` | `maker` (any actor), `another` (an actor who has not acted in the current round), `table` (the engine, through `call`) |
| `call` | table steps only, and required on them: the Rust-registered decode that answers. Its verdicts are the step's outcomes. |
| `prompt` | the text, or a map from the incoming outcome to the text |
| `edits` | `new`, `change` or `optional` (the default), or a map from the incoming outcome to one of these |
| `next` | a target for a step with one result, or a non-empty map of outcome → target |
| `stuck` | actor steps only: where a `stuck` report leads once `stuck-limit` have been made in the round (default `failed`) |
| `stuck-limit` | actor steps only: at least 1 (default 1) |
| `desk` | where this step is taken, over the workflow's `desk` |
| `tools`, `checks`, `context` | YAML lists of names, carried as names. Rust gives them meaning. |

A target is a step, `done`, or `failed`. `cancelled` is an end state that only
the engine can set, so no step leads there. No step may be named for an end
state, and `stuck` is never a key in `next`.

**Only a publish step leads to `done`.** A publish step is one whose `checks`
include `published`, which holds the step open until the operation's store
has landed on the record. A step that leads to `done` without it is a load
error, so verification before publishing is a rule of the format.

A publish that cannot reach git — `origin` down, or no agent to push with —
neither settles nor counts as stuck. The operation records when it may be
tried again, 30 minutes on, and the table does not offer its step before
then.

### Variants

A `prompt` or `edits` map is keyed by the outcome that led into the step: the
previous step's outcome, or `start` when the run enters the step at its start.
The value for a step is picked in this order:

1. the incoming outcome's key;
2. `default`;
3. otherwise it is an error at offer time.

A plain completion (from a step with a single `next`) has no outcome, so only
`default` answers it.

### Prompts

A prompt is resolved in two stages.

**Includes, at load.** These references are spliced in verbatim:

- `{name}`: a shared prompt from `prompts:`.
- `{step.prompt}`: another step's prompt in the same workflow.
- `{workflow.step.prompt}`: a step's prompt in any workflow.
- `{[workflow.]step.prompt.variant}`: one variant of a step's prompt.

Includes resolve inside shared prompts, generator prompts and every variant,
and they nest. Each of these is a load error:

- a reference to a step that does not exist;
- a reference to a variant that does not exist;
- including a prompt that has variants without naming one;
- a same-workflow reference inside a shared or generator prompt;
- a cycle of includes.

A `{name}` that names no shared prompt is left for the next stage, so a shared
prompt shadows an operation placeholder with the same name.

**Operation placeholders, at offer time.** The engine fills `{objective}`,
`{findings}`, the proposal's fields and the rest. A placeholder is `{`, then a
letter or `_`, then letters, digits or `_`, then `}`. Any other brace is text.
A placeholder with no value is an error that names every such placeholder; it
is never left blank. An empty string counts as a value.

## Running an operation

An operation persists one `Run`, which holds:

- where it stands: `next_step`, `done`, `failed` with a reason, or `cancelled`
  with a reason;
- the incoming outcome;
- its history: each step taken, by `table` or an actor, with the outcome, where
  it led, and the notes;
- where the current round starts;
- its send-back count;
- each step's stuck reports in this round.

### Rounds and send-backs

A route to an earlier step, or to the same step, is a **send-back**. It opens
a new round, and the round's first entry is the step that sent the work back.
`another` is judged within the round. So whoever sent the work back does not
take the fix, but the writer from an earlier round may review it. The run
fails on the send-back that takes it past the workflow's `send-backs`, and the
reason quotes what that step found.

### Stuck

Every actor step also offers the outcome `stuck`. A stuck report stays on the
step, and the step is offered to someone who has not reported it stuck this
round. Once `stuck-limit` reports have been made in the round, the step routes
to its `stuck` target. Reaching an earlier step this way is a send-back like
any other.

### Functions

| Function | What it does |
|---|---|
| `start(workflow)` | opens a run at the first step |
| `start_at(workflow, step)` | opens a run at any step; incoming outcome `start` |
| `offered(workflow, run)` | returns the waiting step, its selected prompt, and its edits |
| `may_take(workflow, run, taker)` | says whether `taker` may take the waiting step |
| `advance(workflow, run, taker, outcome, notes)` | takes the step: `None` for a plain completion, otherwise an outcome the step offers, or `stuck`. A refused report leaves the run unchanged. |
| `cancel(run, why)` | settles an unsettled run as cancelled |
| `reopen(workflow, run, step)` | puts a settled run back on a step in a new round, with its send-backs counted afresh |
| `run.last_findings()` | the latest step's notes, for `{findings}` |

## Today's pipeline in this form

The hardcoded pipeline (`sim/operations.rs`, `engine/mission_gen/run.rs`) maps
onto `story`, `life-event` and `correction`:

| Today | Workflow |
|---|---|
| the generator's proposal, `no_mission` | a `generators:` entry. A declined proposal opens nothing. |
| `Phase::Drafting` | `write`, `by: maker`, `edits: new`, `next: read` |
| `Phase::Reading` | `read`, `by: table`, `call: reading`, which always leads to `review` with its verdict as the incoming outcome |
| `Phase::Reviewing` (a second Maker) | `review`, `by: another`, with prompt and edits chosen by the reading's verdict; `pass: reread`, `reject: fix` |
| the table's second reading | `reread`. `mend` sends the work back to `review`. |
| `Phase::Checking` (a third Maker) | `canon`, `by: another`; `pass: publish`, `reject: fix` |
| the commit into the mind folder | `publish`, at the merge table, the only step leading to `done` |
| `Phase::Fixing`, carrying `fault` | `fix`, `by: another`, `next: read` (a send-back), with the fault as `{findings}` |
| `FIX_LIMIT` | `send-backs` |
| `REVIEW_STUCK_LIMIT` | `stuck-limit` |
| `needs_canon(document)` | whether the workflow has a `canon` step |
| `leaves_on_failure`, `before` | `on-failed: discard` or `keep`: a failed draft never reached the record |
| `writer`/`reviewer`/`checker`, `round` | the run's history and rounds |
| `Phase::Succeeded` / `Failed` / `Cancelled` | `done` / `failed` / `cancelled` |
