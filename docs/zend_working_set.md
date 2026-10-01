# The working set — what a dialogue already knows about the code

**Status:** design complete, not built — §10 is the build order. Replaces the
fast-path budget described in `zend_branch_ingest.md` §6 (`fast_path_window`);
it does not sit beside it.

## 1. The idea

Every dialogue carries a **working set**: a token budget (350K) of
already-ingested `code_reading` files and `repo_map` folders, projected ahead of
its own turns. The unit is always a **whole ingest conversation** — one file or
one folder — so an item moves between tiers without changing shape.

1. **Seeds** — documents, folders and files named in `projection.yaml` that every
   conversation should simply know. Always present.
2. **Locks** — files and folders the model asked for with `file_read` /
   `file_list` that the corpus already holds. The call is answered "already in
   context" and the content is pinned for the rest of the task.
3. **Provenance** — the files the conversation keeps attending to, ranked by a
   per-file momentum score, filling what seeds and locks leave.

Locks push provenance out; nothing pushes a lock or a seed out. When a lock would
not fit, the tool call runs for real — the fast path is an optimisation, and
running out of it costs a read, never correctness. A real user turn, or a round
that changes code, releases the locks into provenance, so a continuous agentic
flow keeps getting the fast path.

**Ingestion is not changed.** The belief scan already scores every exchange of
every ingested file; the working set only reads those scores.

## 2. What exists today

| Piece | Where | What it does |
|---|---|---|
| Tool-call screening | `zend/src/fast_path.rs` (`screen`, `Screen::hit`) | Replaces a `file_read`/`file_list` whose target a finished ingest conversation holds with a served answer. Keyed `path@blob` from the conversation's own branch. |
| Call site | `zend/src/session.rs:3868-3929` | Runs the screen on each round, before dispatch. |
| Admission + LRU | `candle-conversation/src/substrate.rs:3039` (`fast_path_admit`) | Per-target list, evicted from the tail at `code_reading`'s `fast_path_window` (300K). |
| Splice | `projection/resolver.rs:4532-4552` | Injected conversations join the **dialogue group** after the lineage. |
| Selection | `projection/selection.rs:227-290` | `select_conversation` treats every turn not on the dialogue's own timeline as inherited: never trimmed. |
| Belief | `provenance/belief.rs:120-126` | `acc = max(0, acc − β·m) + fresh`, `m` = the **group's** highest belief — winner-takes-most, built for tool selection. |
| Collection `mandatory` | `projection/schema.rs:539-544` | Members emitted outside the budget while the rest are ranked. Collections only. |
| Tool-round gate | `project.rs:1279`, `projection.yaml:579,735` | `repo_map` and `code_reading` are left out of every tool round. |

## 3. Faults in today's fast path

Verified in the code. Items 1–6 are live bugs; the design removes each, and
§4.5 lists the fix.

1. **A promised file can be dropped.** A selected turn that cannot be made hot
   is dropped with a warning (`scheduler/projection_assembler.rs:1530`) after the
   model was told "already in context". A warm lift is all-or-nothing per layer
   (`scheduler/mod.rs:9613`), so one shortfall drops a batch.
2. **Edit then read in one round is served stale.** The screen runs on the whole
   round before any call executes (`session.rs:3896` before `:3944`), so
   `[file_edit X, file_read X]` serves X's pre-edit blob.
3. **A partial chain is served as the whole file.** `chain_finished` checks the
   chain closed, not that every page was read (`code_read/chain_health.rs:78-94`);
   a chain cut at `MAX_FILE_READ_ROUNDS` still answers "its full contents".
4. **A tombstoned lock keeps emitting.** `tombstone_timeline`
   (`substrate.rs:4143`) never touches the fast-path set, and
   `fast_path_injections` does not filter it.
5. **LRU evicts a hit already answered.** Two hits in one round can evict the
   first after its `already_read` went out (`substrate.rs:3071-3089`).
6. **The size gate is an estimate.** bytes/4 against a chain that carries
   numbered lines, per-page framing and thinking turns — roughly 1.4× the
   estimate; a timeline with no recorded total costs 0 (`substrate.rs:3067`).
7. **Pinned content starves the dialogue.** Injections sit in the dialogue group,
   and `select_conversation` gives historical turns only what recent + inherited
   leave (`selection.rs:271-278`) — with a large working set, nothing.
8. **Placement is accidental.** Emission re-sorts every group by `TurnKey`
   (`project.rs:1944-1946`), i.e. by timeline id: a file lands before or after the
   whole dialogue depending on when it was ingested.

## 4. The design

### 4.1 Shape

The working set is **not** part of the dialogue group. Each pinned item is a
forced member of its **own layer's group** — a file of `code_reading/scopes`, a
folder of `repo_map/structure` — the turn-group equivalent of a collection's
`mandatory`. Provenance fills the same groups. Layers already emit in rank order
ahead of the dialogue, so:

```
system prompt
repo_map/structure      pinned folders + provenance folders   ┐ working set,
code_reading/scopes     pinned files   + provenance files     ┘ ≤ budget_tokens
lineage + dialogue      untouched: recent 16 + historical 8, its own 116K window
```

This one move fixes §3.7 (the dialogue's budget no longer carries the working
set) and §3.8 (the working set is always ahead of the dialogue). Within a group,
emission stays in `TurnKey` order; no new ordering code.

**Budget.** The two groups are sized by the working set, not by the flexbox:
`project.rs` takes their selections off the top and distributes the dialogue's
`window` over the rest as today. Their `budget:` shares in `projection.yaml` are
deleted.

- `repo_map` — pinned folders, then provenance folders, up to `folder_tokens`.
- `code_reading` — pinned files, then provenance files, up to what remains of
  `budget_tokens`.

Two budgets because the two groups' scores are normalised separately and cannot
be ranked against each other; folders are small, so a fixed slice is enough.

### 4.2 Configuration

On the dialogue layer, replacing `fast_path_window` (deleted):

```yaml
  - name: dialogue
    working_set:
      budget_tokens: 350000      # everything below, together
      folder_tokens: 30000       # repo_map's share: pinned + provenance folders
      beta: 0.2                  # per-file momentum leak (§4.3)
      min_momentum: 100          # below this a file is not a provenance candidate
      max_file_tokens: 100000    # one file above this is never locked or pinned
      seeds:
        - candle/zend/
        - candle/zend/src/
        - candle/zend-tools/src/
        - candle/candle-conversation/src/
        - candle/candle-transformers/src/models/
        - candle/candle-nn/src/kv_cache/
        - candle/candle-core/src/
        - candle/candle-kernels/src/
        - candle/docs/
        - candle/Cargo.toml
        - candle/docs/deployment.md
        - candle/docs/zend_branch_ingest.md
        - candle/docs/zend_working_set.md
        - candle/zend/src/lib.rs
        - candle/zend/src/main.rs
        - candle/zend-tools/src/lib.rs
        - candle/candle-conversation/src/lib.rs
      release_on: [write, file_edit, file_delete,
                   git_commit, git_merge, git_ref, git_switch, git_reset,
                   run_command, code_run, code_session_exec]
```

- `release_on` names are **checked against the tool registry at load**; an
  unknown name is a load error. (`file_write` is an alias of `write`, and
  `code_run`/`code_session_exec` write the conversation's files too.)
- Seeds leave out what the priming chain already carries — the repository roots,
  `README`, `CLAUDE.md` — because lineage emits them anyway. Resolution skips any
  seed whose timeline is in the dialogue's `inherited_chain`, so an overlap in
  the list costs nothing.
- Every seed sits within `--max-depth 3`; a deeper file has no conversation.

### 4.3 State and momentum

`Substrate::fast_path` becomes `HashMap<TimelineId, WorkingSet>`, in
`candle-conversation/src/working_set.rs`:

```rust
pub struct WorkingSet {
    seeds: Vec<TimelineId>,
    locks: Vec<(TimelineId, u64)>,      // with the lock's sequence number
    momentum: HashMap<TimelineId, f32>, // provenance candidates
    next_seq: u64,
}
```

**Momentum is per file and leaks by its own value**:
`m ← (1 − β)·m + fresh`, where `fresh` is the file's best exchange score this
reprojection (the scan already produces per-exchange normalised scores; a max
per timeline is one pass). A file hit steadily settles at `fresh/β`; one left
alone decays geometrically. Below `min_momentum` (config, default 100 — the
noise floor `code_reading/scopes` gates at today, `projection.yaml:821`) a file
is dropped from the map, and only files at or above it are provenance
candidates, so an idle budget stays empty rather than filling with noise.

This is deliberately **not** the RelLeak belief (`belief.rs`). RelLeak leaks by
the group's leader, which is right for picking three tools and wrong here: a
file next to a strong leader is wiped in one step, and every non-leader sits at
its fresh score, so nothing accumulates. Keeping momentum in `WorkingSet` also
sidesteps two properties of the belief carry that would break it — the carry
records only emitted turns, and it is halved at every submit, which in an
agentic flow is every round.

`observe` runs once per reprojection, on the line after the belief scan
(`scheduler/mod.rs:10101`): `score_beliefs` already returns `group_candidates`,
every scanned group's fresh `(TurnKey, score)` list. For the two working-set
groups the max per timeline is taken and handed to the dialogue's `WorkingSet`
under the substrate write lock. The turn's opening projection is not scanned
(it reads zero scores, `resolver.rs:686`), so it projects the momentum as it
stood — which is what an opening should see. Ranking is
`(momentum desc, lock seq desc)` — the most recently locked file wins a tie.

**Locks survive a restart; momentum need not.** A lock is a promise already
written into the conversation's history — the `already_read` reply — so it must
come back with the conversation. It does not need storing to do so: the history
*is* the record (§4.8). Momentum is a ranking, not a promise; it stays in-memory
and rebuilds from the scan within a few reprojections.

### 4.4 Projection

A `working_set` selection rule, declared on `repo_map/structure` and
`code_reading/scopes`, replaces their `top_k`:

1. The group's pinned timelines (seeds, then locks), whole.
2. Provenance: the group's timelines by momentum, whole, skipping pinned ones,
   until the group's budget (§4.1).

The rule reads the dialogue's `WorkingSet` through the resolver
(`ContentResolver::working_set`, following `fast_path_injections`); a target
with no `WorkingSet` — every ingest conversation — selects nothing, so ingest
projections are unaffected.

**Off the flexbox, by an existing precedent.** A `score_density` group already
has its picks decided upstream and emitted verbatim, skipping the bounded pass
(`project.rs:1851-1864`). A `working_set` group takes the same path, and one more
step: its natural consumption is left out of `layer_items`/`group_items`
(`project.rs:1768-1848`), so the flexbox distributes the dialogue's `window`
over the other groups as if the working set were not there. Its selection is
sized by its own budget and never trimmed.

**Budget between the two groups.** `repo_map/structure` selects first, up to
`folder_tokens`; `code_reading/scopes` gets `budget_tokens` minus what the
folder group actually selected. Pinned items count first in each; provenance
fills what is left.

`in_tool_rounds` becomes `true` on both layers — the working set must be present
in the rounds that lean on it. `locality`, `anchor`, `budget_adaptive` and the
`policy:` blocks on these two groups are deleted; they tune scope-level belief,
which no longer selects anything here. The scan's own settings
(`fusion`, `layer_weights`, `question_pin`, `level_prior`) stay — they decide the
fresh scores momentum is built from.

The dialogue group loses its injection splice (`resolver.rs:4546-4551`): it holds
lineage and its own turns only, and its 116K window is its own again.

### 4.5 Locks — the screen

The screen stays in `fast_path.rs`; its admission changes and four checks are
added.

- **No eviction.** `WorkingSet::lock` admits while pinned + the new file fit the
  budget, otherwise refuses and the call runs for real (§3.5).
- **Real cost.** The per-file cap and the budget both use
  `timeline_token_totals`; a timeline with no total is a miss (§3.6).
- **A round is screened only up to its first `release_on` call.** Calls after it
  run for real, because the write may change what they read (§3.2).
- **Only complete reads are served.** `file_read` takes one argument that
  matters here, `page` (`zend-tools/src/tools/file/read.rs:40`), and returns one
  `PAGE_LINES` page per call. A chain is complete when its call turns'
  `file_read` pages — `tool_round::plan` on each call turn's assistant half,
  paired in order with the next turn's `<tool_response>` blocks, counting a page
  only when its response carries no `error` — cover `0 ..
  ceil(LINES_KEY / PAGE_LINES)`. A chain cut at `MAX_FILE_READ_ROUNDS`, or one
  that skipped a page, runs for real (§3.3). A finished chain never changes, so
  the answer is computed once per timeline and cached beside the screen.
  Ingestion is not changed. A folder unit holds page 0 of its listing, which is
  the only page the screen serves, so folders need no coverage check.
- **Tombstone removes.** `tombstone_timeline` drops the timeline from every
  working set — seeds, locks and momentum (§3.4).

**Pinned content must be hot, loudly.** Two changes:

- **Refuse, don't drop.** `apply_projection`'s "selected turn has no hot sealed
  K/V; dropping it" (`projection_assembler.rs:1530`) becomes an error when the
  turn's timeline is pinned in the target's `WorkingSet` — the model was told it
  has it (§3.1). Provenance turns carry no promise and keep today's drop.
- **Protect every dialogue's pins.** `working_set_pins` is one set, replaced
  wholesale at each elevate (`substrate.rs:1780`), so it only covers the
  projection being elevated. `evict_hot_to_free` (`substrate.rs:~2645`, called
  from `evict_to_fit_incoming`, `scheduler/mod.rs:9603`) additionally skips any
  residence whose turn is pinned in *any* `WorkingSet`, so one dialogue's
  elevation cannot evict another's promise.

A working set too large for the card therefore fails loudly on its first
projection rather than serving silently less. `budget_tokens` is sized for the
card it runs on (§5).

### 4.6 Release

A lock is released by:

- **a real user turn** — `iteration == 0` in `run_inference_stream`, before the
  scope is applied and the turn submitted (`session.rs:~3472`). A resumed turn
  starts its loop at 1 (`session.rs:3388`), so it is never mistaken for one;
- **a dispatched round containing a `release_on` call** — checked on the round's
  results, beside the existing branch-move scan (`session.rs:3967`), **and** on
  the resumed-round dispatch (`session.rs:3333-3389`, `Dispatch::Resumed`),
  which runs a round interrupted by a restart without passing the screen.

Release clears the locks and gives each released file momentum `1000/β` — the
settled level of a file hit at full strength every reprojection — so a file the
model keeps working in stays, and one read once decays out over the following
reprojections. Seeds are re-resolved at the same moment, so a seed the
conversation has edited or moved off stops projecting its old content.

**The promise is scoped to the task, in words.** A released file can decay out
while its `already_read` reply stays in the history, and a model told it has a
file does not read it again. So the served reply says what the lock actually
guarantees:

> `` `{path}` `` in {repo} is unchanged since it was read, and its full contents
> ({lines} lines) are in your context for the current task — including any lines
> this call asked for. After you change code, or when a new request starts, read
> it again if you need it.

A re-read after release is cheap either way: the file is still in the corpus, so
the screen serves it again and re-locks it — an elevation, not a read.

### 4.7 Seeds

Resolved per dialogue at open and at every release: a file through
`CONTENT_KEY` at the blob the dialogue's branch holds, a folder through
`folder_unit`. A miss (changed, not ingested, in the lineage) is skipped. Seeds
past the budget are admitted in list order until the next would not fit, with a
warning naming the rest.

### 4.8 Restart

Locks are rebuilt from **turn tags** the dialogue already persists; no new
record, no text parsing.

**Tagging.** Every turn carries `TurnOptions::tags` onto its `TurnDecl`
(`turn.rs:99-102`, `substrate.rs:973-976`) — empty for live dialogue turns
today. The submit that carries a round's results tags that turn with what the
round did, in the order it happened:

- `lock:<timeline>` — one per call the screen served, naming the exact
  conversation it pinned;
- `release` — on a real user turn, and on the turn carrying the results of a
  round that made a `release_on` call. A round's reads served *before* its first
  write come first in the list, then `release`, so replaying the list in order
  releases them too, exactly as the live round did.

The tags are inert to the projection: a group filters by tag only when its
policy declares tags, and never the target group (`project.rs:1313`), so no
dialogue selection changes.

**Rebuild.** Walk the dialogue's turns back from the newest to the last turn
tagged `release`, then apply every tag from there forward in order: `lock:<tl>`
admits `tl` with the next lock sequence, `release` clears. A timeline tombstoned
since is dropped (§4.5). Seeds re-resolve; momentum starts empty.

**The tags decide, not a re-run of the screen.** Re-screening could answer
differently — seeds resolved to other sizes, the budget came out tighter — and a
model holding an `already_read` with no lock behind it is the failure this whole
section exists to prevent. The tags are what the model was told, recorded as
the conversation it was told about.

This replaces `fast_path::rebuild`, which parsed every `<tool_call>` the
conversation ever made — before any release — re-resolved each by path, and
re-decided it.

The same tags give the GUI a served-call marker for free, and give any later
audit of "what did the fast path promise this conversation" a direct answer.

## 5. Constraints and cost

- **Context length.** Flash-Next is trained to 262,144 positions; progressive
  YaRN (`qwen4exp/rope.rs`) carries it ×2 to 524,288. 350K + the dialogue's 116K
  (which includes the 100K lineage cap) is ~466K. A slot's rung follows its
  deepest position, so **a dialogue past 262K runs factor 2 at every position,
  its own turns included** — the working set changes the whole conversation's
  rotation, not just its reach. The only length checks are against the RoPE
  reach (`prefill.rs:2052`); `max_seq_len` is ignored for this architecture.
- **Sparse attention.** Flash-Next has 12 attention layers (every 4th of 48) and
  36 recurrent. Spliced content reaches the attention layers only, through QSA's
  indexer, which keeps `top_k = 2048` index rows per query per layer. "The model
  knows this file" means the indexer *can select* it.
- **Hot residency (estimate).** 2 KV heads × 256 × K+V × 12 layers; at C5
  roughly 2.3–2.5 GB of hot K/V per 350K tokens, plus ~0.5 GB of QSA index —
  more for many small turns, since index pages round up to 256 rows per layer.
  Content shared between dialogues is one copy.
- **Decode.** The indexer scores every page each step: ~720 MB read per sequence
  per step at 466K.
- **Churn.** Dialogues reproject every 64 tokens. Whole-file provenance turns
  over in large units; momentum (rather than the fresh score) is what keeps that
  turnover slow.
- **Speculative decoding.** Flash-Next's draft step builds plain `rope_cos_sin`
  beside the rung set (`qwen4exp/draft.rs:627-634`). Speculation is off on this
  model today (`draft_budget 0` at load), so it does not arise; before it is
  enabled, the draft head must be checked to rotate by the slot's rung past
  262K, or its drafts will be rejected at depth.

Budget ships at 350K (D10); the §7 measurement watches these numbers.

## 6. Code map

| Change | File |
|---|---|
| Change | Where |
|---|---|
| `WorkingSet`: lock / release / observe / select, per-file momentum | `candle-conversation/src/working_set.rs` (new) |
| `fast_path` map → `WorkingSet` map; `tombstone_timeline` drops from every set | `candle-conversation/src/substrate.rs:3023-3104`, `:4143` |
| `observe` on the line after the scan | `candle-conversation/src/scheduler/mod.rs:10101` |
| `working_set` selection rule | `projection/schema.rs` (`SelectionRule`), `projection/yaml.rs`, `projection/project.rs` (phase-1 arm beside `score_density`, `:1343`) |
| Working-set groups off the flexbox, emitted verbatim | `projection/project.rs:1768-1864` |
| Dialogue group loses the injection splice | `projection/resolver.rs:4546-4551` |
| Pinned turn not hot is an error | `scheduler/projection_assembler.rs:1530` |
| Eviction skips every dialogue's pins | `substrate.rs` (`evict_hot_to_free`) |
| `working_set:` block; registry check of `release_on`; delete `fast_path_window` | `projection/yaml.rs`, `projection/schema.rs:825-833`, `zend/src/session.rs` (load) |
| Screen: `lock`, first-write split, coverage check (cached per timeline), real token cost, reworded reply | `zend/src/fast_path.rs` |
| Tag the results turn `lock:<tl>` / `release`; rebuild from tags (replaces `rebuild`) | `zend/src/session.rs:3489` (submit options), `zend/src/fast_path.rs` |
| Release on a user turn and on `release_on` rounds, live and resumed | `zend/src/session.rs:~3472`, `:3967`, `:3333-3389` |
| Seed resolution at open and at release | `zend/src/working_set_seeds.rs` (new) |
| Layer config | `zend/src/prompts/projection.yaml` |

## 7. Tests (CPU unless marked)

- `WorkingSet`: lock admits until full, then refuses without evicting; release
  clears locks and sets momentum `1000/β`; `observe` settles a steady file at
  `fresh/β` and decays an idle one; a file below `min_momentum` leaves the map
  and is never a candidate; ties go to the later lock; a tombstoned timeline
  leaves every tier.
- Selection: pinned timelines emit whole and are never trimmed; provenance fills
  to the group budget by momentum and never repeats a pinned timeline;
  `folder_tokens` bounds `repo_map` and `code_reading` gets the rest; the
  flexbox distributes the dialogue's `window` exactly as it would with no
  working set (the dialogue group's selection is identical with and without a
  full working set); an ingest target selects nothing.
- Screen: a hit while full runs for real; calls after a `release_on` call in the
  same round run for real; a timeline with no token total runs for real.
- Coverage: a chain whose served pages cover `0..ceil(lines/PAGE_LINES)` is
  complete; one missing a page, or whose page response was an error, is not; the
  answer is cached per timeline.
- Hot: a pinned turn with no hot K/V fails `apply_projection`; a provenance turn
  with none is dropped as today; `evict_hot_to_free` never frees a residence
  pinned by another dialogue's working set.
- Release: user turn releases, a tool-response continuation does not, a
  resumed turn does not, a `release_on` round does (live and resumed); seeds
  re-resolve.
- The served reply carries no `error`/`detail` and scopes its promise to the
  current task (exact string).
- Config: an unknown `release_on` name fails the load.
- Tags: a round with served calls tags its results turn `lock:<tl>` per served
  call in order; a user turn and a `release_on` round's results turn carry
  `release`; reads served before a round's first write precede its `release`.
- Rebuild: locks come back from the tags after the last `release`, and none from
  before it; a served lock is restored even when a fresh screen would now refuse
  it; a tombstoned timeline is dropped; lock order matches tag order.
- The tags change no dialogue selection (a projection with and without them is
  identical).
- **Behavioural, daemon:** with the working set present in a tool round, the
  round answers the user's question, not an ingest request turn.
- **GPU, daemon stopped:** hot residency, QSA index bytes and decode t/s at
  0 / 116K / 350K of working set.

## 8. Decisions

| # | Question | Decided |
|---|---|---|
| D1 | Past the 262K trained length | Progressive YaRN ×2 reaches 524,288. Keep 350K. |
| D2 | What release does to locks | Demote to provenance with momentum `1000/β` (the settled level of a full-strength hit). |
| D3 | Tie-break | Most recently locked wins. |
| D4 | Which calls release | File writes, git writes, `run_command`, `code_run`, `code_session_exec` — registry-checked names. |
| D5 | Where the working set sits | Its own layers' groups, which already emit ahead of the dialogue. |
| D6 | Tool rounds | `in_tool_rounds: true`; ingestion unchanged. |
| D7 | Provenance unit | Whole files and folders. |
| D8 | Seed list | §4.2 draft, to be edited. |
| D9 | Fade rate | Per-file momentum at β 0.2, leaking by its own value. |
| D10 | Default budget | 350K; measured alongside. |
| D11 | Momentum store | In `WorkingSet`, not the belief carry; in-memory. |
| D12 | Restart | Served calls are tagged on their turn (`lock:<tl>`, `release`) through the existing persisted turn tags; locks rebuild from the tags since the last `release`. Momentum rebuilds from the scan. |
| D13 | Folder vs file budget | `folder_tokens` slice for `repo_map`, the rest for `code_reading`. |
| D14 | A released file decaying out under an `already_read` reply | The served reply scopes its promise to the current task and says to re-read after a change or a new request; a re-read re-locks it cheaply. |
| D15 | "Complete read" without touching ingestion | Derived from the chain's own `file_read` pages and responses, cached per timeline. |
| D16 | A pinned file that cannot be made hot | An error, never a silent drop; eviction protects every dialogue's pins. |
| D17 | Provenance floor | `min_momentum: 100`, the noise floor `code_reading` already gates at. |

## 9. Deliberately not done

- **No persistence of momentum.** It is a ranking, and rebuilds from the scan
  within a few reprojections.
- **No stable-prefix ordering within the working set.** Reprojection is a
  zero-copy rebuild; ordering inside the block buys nothing measurable.
- **No cross-group ranking.** Two budgets instead of a common score scale.
- **No rewriting of old replies.** A released file's `already_read` stays as it
  was sealed; the reply's own wording (D14) carries the scope instead.

## 10. Build order

Each step lands with its tests and leaves the daemon working.

1. **Fix today's fast path** (§3.1–3.6): refuse-don't-drop for injected turns,
   first-write split, coverage check, tombstone removal, no LRU, real token cost,
   reworded reply. Independent of everything below.
2. **`WorkingSet` + tags + rebuild**, still spliced into the dialogue group as
   today: locks, release, `lock:`/`release` tags, rebuild from tags.
3. **The `working_set` rule**: move pinned content into the two layers' groups,
   take them off the flexbox, delete the dialogue splice, add seeds.
4. **Momentum**: `observe` after the scan, provenance tier, `in_tool_rounds`.
5. **GPU measurement** (§7) with the daemon stopped; the behavioural tool-round
   test.
