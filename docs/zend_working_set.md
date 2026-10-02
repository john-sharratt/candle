# The working set — what a dialogue already knows about the code

**Status:** built (§10 steps 1–4); the GPU measurement of §7's last row is
§10 step 5. Replaces the fast path's least-recently-used budget
(`fast_path_window`, deleted); it does not sit beside it.

## 1. The idea

Every dialogue carries a **working set**: a token budget (250K) of
already-ingested `code_reading` files and `repo_map` folders, projected ahead of
its own turns as **one sequence in the order its members entered**. The unit is
always a **whole ingest conversation** — one file or one folder — so an item
moves between tiers without changing shape or place.

1. **Locks** — files and folders the model asked for with `file_read` /
   `file_list` that the corpus already holds. The call is answered "already in
   context" and the content is pinned for the rest of the task.
2. **Provenance** — the files the conversation keeps attending to, by a
   per-file momentum score. A lock released at the end of a task becomes
   provenance where it stands.
3. **Seeds** — documents, folders and files named in `projection.yaml` that a
   conversation starts out attending to: its first provenance, at the momentum
   of one full-strength hit (1000), fading like any other file it stops
   attending to.

```
[system + priming chain][ provenance, in insertion order | locks, newest first ][ dialogue … question ]
                                                         ^ every member enters here
```

What every conversation must always carry is the **priming chain**'s — the
repository roots, `README`, `ARCHITECTURE`, `CLAUDE.md` — inherited through
lineage, not held by the working set.

Everything enters at one **insertion point**: a lock just before it, so the
oldest lock stays nearest the dialogue; provenance just after the provenance
already there, so the newest provenance sits beside the locks. A member keeps
its place until it leaves, and the gap closes at once. A newcomer that needs
room dislodges the weakest provenance — a lock always, new provenance only when
it is clearly stronger; nothing dislodges a lock. When a lock cannot fit beside
the locks already held, the tool call runs for real — the fast path is an
optimisation, and running out of it costs a read, never correctness. A real
user turn, or a round that changes code, releases the locks into provenance, so
a continuous agentic flow keeps getting the fast path.

**Ingestion is not changed.** The belief scan already scores every exchange of
every ingested file; the working set only reads those scores.

## 2. What the fast path was

§2 and §3 record the fast path as it stood before the working set replaced it;
the line references are to that code.

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
   first after its `in_context` went out (`substrate.rs:3071-3089`).
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
set) and §3.8 (the working set is always ahead of the dialogue).

**Order within a group: the working set's own — insertion order.**
`working_set_pick::pick` emits the group's members exactly as the `WorkingSet`
sequence holds them (§4.3): provenance in the order it entered, then this
task's locks with the newest first and the oldest nearest the dialogue. Each
conversation's turns come in order, and `project.rs` does not re-sort a
working-set group.

The order is built so that **every change happens in the middle.** RoPE
attention depends on the distance between a key and the query. A change at one
position leaves everything to its right at the same distance from the question
(it shifts with the question) and everything to its left at the same distance
from the system prompt. Every change to the sequence is an insertion at the
insertion point or a removal, so:

- **Locks are relative-stable.** A new lock enters left of the older ones, locks
  are never dislodged, and release moves nothing — a lock keeps its exact
  distance to the question for the whole task. These are the files the model
  was just told it has, and retrieval falls off with distance: measured
  2026-09-30 on Flash-Next, a pinned file at the start of a ~390K-token prompt
  could not be read back (and was confused with another file's pages), the same
  file near the end was read correctly. So they sit nearest the question and
  hold still.
- **Long-lived provenance is absolute-stable.** Nothing enters to its left, so it
  keeps its place beside the system prompt.
- **The churn lands in the middle.** Newcomers enter at the insertion point and
  the weakest — usually the youngest — are dislodged beside it, where attention
  is weakest anyway (lost in the middle).
- **A rebuild keeps the slot's prefix up to the first piece that differs**
  (`scheduler/piece_identity.rs`), so a change re-places only what lies to its
  right; nothing to its left is rebuilt.

(2026-10-01: this replaced two earlier orders — provenance by momentum band,
which moved a member each time its momentum doubled or halved, and provenance
newest-first, which put every newcomer at the far left and rebuilt the whole
working set behind it.)

The distance limit is not solved by ordering: as the dialogue grows, everything
ahead of it moves further back. That remains open.

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
      budget_tokens: 250000      # everything below, together
      folder_tokens: 10000       # repo_map's share: pinned + provenance folders
      beta: 0.2                  # per-file momentum leak (§4.3)
      min_momentum: 100          # below this a file is not a provenance candidate
      max_file_tokens: 100000    # one file above this is never seeded or locked
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
  unknown name is a load error (`zend::working_set::dialogue_config`). A call
  releases under any name the registry answers to, so the model writing the
  alias `file_write` releases as `write` does. (`code_run`/`code_session_exec`
  write the conversation's files too.)
- Seeds leave out what the priming chain already carries — the repository roots,
  `README`, `CLAUDE.md` — because lineage emits them anyway. Resolution skips any
  seed whose timeline is in the dialogue's `inherited_chain`, so an overlap in
  the list costs nothing.
- Every seed sits within `--max-depth 3`; a deeper file has no conversation.

### 4.3 State and momentum

The substrate keeps one `WorkingSet` per dialogue,
`HashMap<TimelineId, WorkingSet>`, in `candle-conversation/src/working_set/`:

```rust
pub struct WorkingSet {
    members: Vec<Member>,                // emission order: { timeline, tokens, share, locked }
    insert_at: usize,                    // the insertion point
    momentum: HashMap<TimelineId, f32>,  // members and candidates; a lock's is frozen
}
```

The **budget is enforced on entry**, so membership is state, not a fresh pick
each projection: an unchanged set projects identically. Each member draws on its
group's share — folders up to `folder_tokens`, files on what `budget_tokens`
leaves — which the substrate knows per group (`Substrate::set_working_set_share`,
set at setup from each `working_set` rule's `share`). Every operation:

| Operation | What happens |
|---|---|
| `lock` | A member is locked **where it stands**. A newcomer dislodges the weakest provenance (lowest momentum; on a tie the one nearest the insertion point) until it fits, then enters just before the insertion point. A lock never dislodges a lock: one that cannot fit is refused (`Refusal::Full`). |
| `observe` | Momentum decays for everything not locked; a conversation under `min_momentum` leaves the map and, if a member, the sequence — the gap closes. An unlocked member its scope no longer offers leaves too. Then the strongest non-members try to enter at the insertion point (which moves past them), each dislodging only provenance it beats by `DISLODGE_MARGIN` (×1.5) — without the margin two files near the floor trade places every reprojection and each trade rebuilds everything to its right. |
| `release` | Every lock becomes provenance **in place**, at momentum `max(m, 1000/β)`, and the insertion point moves to the end — the next task's members enter beside the dialogue. Nothing moves. |
| `seed` | At open, in list order: each seed gains `seed_momentum` (1000) and enters at the insertion point while the budget has room — seeds dislodge nothing. One that misses for room stays a candidate. |
| `restore_lock` | A lock the history records, put back before the insertion point whatever it costs (§4.8). |
| `remove` | A tombstoned conversation leaves; the gap closes. |

An unattended seed falls under the floor after
`ln(1000 / min_momentum) / ln(1 / (1 − β))` reprojections — eleven at the
defaults.

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

`observe` runs once per reprojection, right after the belief scan
(`reproject_view_prepare`): `score_beliefs` already returns `group_candidates`,
every scanned group's fresh `(TurnKey, score)` list — a working-set group is
not belief-driven but is still scanned (`GroupSchema::is_scanned`). For the two
working-set groups the max per timeline is taken and handed to the dialogue's
`WorkingSet` under the substrate write lock. The turn's opening projection is
not scanned (it reads zero scores), so it projects the momentum as it stood —
which is what an opening should see. Momentum ranks only who enters and who is
dislodged; it never places anyone.

**Locks survive a restart; momentum need not.** A lock is a promise already
written into the conversation's history — the `in_context` reply — so it must
come back with the conversation. It does not need storing to do so: the history
*is* the record (§4.8). Momentum is a ranking, not a promise; it stays in-memory
and rebuilds from the scan within a few reprojections.

### 4.4 Projection

A `working_set` selection rule, declared on `repo_map/structure` and
`code_reading/scopes`, replaces their `top_k` —
`selection: { kind: working_set, share: folders }` on the folder group,
`{ kind: working_set, share: remainder }` on the file group
(`SelectionRule::WorkingSet { share }`; a collection may not declare it):

the group's working-set members, whole, in the working set's order (§4.3) —
nothing is chosen at projection time.

The rule reads the dialogue's `WorkingSet` through the resolver
(`ContentResolver::working_set_members`, filtered to the group); a target whose
layer declares no `working_set` — every ingest conversation — selects nothing,
so ingest projections are unaffected. An ingest conversation generating into
one of these groups reads its own turns there as it always did. Provenance is
filtered by the conversation's retrieval scope like any other selection from a
scoped group, and leaves the set at the next observation; locks are not,
being promises already made.

**Off the flexbox, by an existing precedent.** A `score_density` group already
has its picks decided upstream and emitted verbatim, skipping the bounded pass
(`project.rs:1851-1864`). A `working_set` group takes the same path, and one more
step: its natural consumption is left out of `layer_items`/`group_items`
(`project.rs:1768-1848`), so the flexbox distributes the dialogue's `window`
over the other groups as if the working set were not there. Its selection is
sized by its own budget and never trimmed.

**Budget between the two groups.** Enforced when a member enters (§4.3): the
folder members hold at most `folder_tokens` between them, and the files take
what `budget_tokens` leaves.

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

- **A lock dislodges provenance, never a lock.** `WorkingSet::lock` makes room by
  dislodging the weakest provenance; when only locks would have to go, it
  refuses and the call runs for real (§3.5). A file already in the set is
  locked in place.
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
  working set — locks and momentum, seeds included (§3.4).

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

Release turns every lock into provenance **where it stands** and gives it
momentum `1000/β` — the settled level of a file hit at full strength every
reprojection — so a file the model keeps working in stays, and one read once
decays out over the following reprojections. The insertion point moves to the
end, so the next task's members enter beside the dialogue and nothing already
in the set moves. Seeds are not touched: re-seeding at every release — and a real
user turn is one — would lift them back to 1000 each turn and they would never
fade. A seed the conversation has edited or moved off leaves the way any stale
provenance does: the retrieval scope stops offering its old conversation.

**The reply is a status and an anchor; the rule lives in the system prompt.**
A served call answers

```json
{"status":"in_context","anchor":"file=candle/Cargo.toml"}
{"status":"in_context","anchor":"the `zend/src/` folder in the `candle` repository"}
```

and nothing else. The reply stands in for a read, so every token it spends is
paid on every served call; what the status means is said once, in the system
prompt's `in_context` section, whose id is the status word — the reply recalls
the rule by the same token rather than restating it. The rule is general: placed
content begins with an anchor naming it; reason with what is here first; items
stay while used and fade after; before relying on one, check its anchor is
present, and if it is not or the model is unsure, call the tool, which answers
`in_context` when the content is already here and keeps it for the task. It
names no tool and no format, so a new kind of placed content needs no prompt
change — the literal anchor arrives in the reply.

**The anchor is byte-identical to what the content begins with.** A file's is the
`file=<repo>/<path>` attribute every page's opening fence carries
(`zend_tools::tools::file::render::file_anchor`, used by both the fence and the
reply). A folder's is the phrase its unit's opening request names it by
(`repo_scan::render::folder_anchor`): the listing's JSON names only its
repository, and the call is rendered in the checkpoint's own syntax, so the
request's phrase is the one dialect-independent string every folder unit
carries. Measured 2026-09-30: told only "in your context", the model searched its
own reads, found none, called the reply false and refused to use the file —
although both of the file's ingest turns were pinned in every projection of that
turn. The status is not "already read" for the same reason: a served file is
usually a background read, not one the dialogue made.

**Each page names its file at both ends.** A page opens on a fence carrying
`file=<repo>/<path> page=N/M lines=T` and closes on `end of <repo>/<path> page
N/M`, with no header line before it. Measured 2026-09-30: with a single header
line above 200 numbered lines, the model bound content near the bottom of a page to
the wrong file among look-alike pages — the name sat 200 lines away from the text
it labelled. Bracketing the page puts its identity next to both halves of it.

A re-read after release is cheap either way: the file is still in the corpus, so
the screen serves it again and re-locks it — an elevation, not a read.

### 4.7 Seeds

Resolved per dialogue once, when it opens: a file through `CONTENT_KEY` at the
blob the dialogue's branch holds, a folder through `folder_unit`. A miss
(changed, not ingested, in the lineage) is skipped. Each resolved seed gains the
seeding momentum and enters provenance in list order while the budget has room,
dislodging nothing (§4.3) — so the seeds are the leftmost members, beside the
system prompt and the priming chain. One that does not fit, has no recorded
size, or is past `max_file_tokens` is left out with a warning naming it; one
that missed only for room stays a candidate and can enter later on its
momentum.

### 4.8 Restart

Locks are rebuilt from **turn tags** the dialogue already persists; no new
record, no text parsing.

**Tagging.** Every turn carries `TurnOptions::tags` onto its `TurnDecl` —
otherwise empty for live dialogue turns. The submit that carries a round's
results tags that turn with what the round did, in the order it happened
(`candle_conversation::working_set::marks`, `zend::working_set::round_marks`):

- `working_set:lock:<timeline>` — one per call the screen served, naming the
  exact conversation it pinned;
- `working_set:release` — on a real user turn, and on the turn carrying the
  results of a round that made a `release_on` call. A round's reads served
  *before* its first write come first in the list, then the release, so
  replaying the list in order releases them too, exactly as the live round did.

**The marks are not gather scope.** A turn's tags also name the tag-scoped
galleries it belongs to, and a turn with no tags is what ordinary dialogue is:
the untagged belief gallery, the normalization warm-up's dialogue replay and
the seal-time observation all select on it. Every such reader goes through
`marks::gather_tags` / `marks::is_dialogue`, so a dialogue turn carrying only
marks stays dialogue. The projection itself filters by tag only when a group's
policy declares tags, and never the target group, so no dialogue selection
changes.

**Rebuild.** Apply every mark from the dialogue's first turn forward in order:
a lock admits its conversation with the next lock sequence, a release clears —
so what stands is the locks after the last release (`marks::standing_locks`,
`zend::working_set::restore`). A timeline tombstoned since is dropped (§4.5).
Momentum starts empty; the seeds re-enter it at the seeding momentum.

**The tags decide, not a re-run of the screen.** Re-screening could answer
differently — seeds resolved to other sizes, the budget came out tighter — and a
model holding an `in_context` with no lock behind it is the failure this whole
section exists to prevent. The tags are what the model was told, recorded as
the conversation it was told about.

This replaces `fast_path::rebuild`, which parsed every `<tool_call>` the
conversation ever made — before any release — re-resolved each by path, and
re-decided it.

The same tags give the GUI a served-call marker for free, and give any later
audit of "what did the fast path promise this conversation" a direct answer.

## 5. Constraints and cost

- **Context length.** Flash-Next is trained to 262,144 positions; progressive
  YaRN (`qwen4exp/rope.rs`) carries it ×2 to 524,288. 250K + the dialogue's 116K
  (which includes the 100K lineage cap) is ~366K, leaving ~158K before the slot
  climbs to ×4. A slot's rung follows its deepest position, so **a dialogue past
  262K runs factor 2 at every position, its own turns included** — the working
  set changes the whole conversation's rotation, not just its reach. The
  shipped schema's `rope: min_yarn_factor: 2` puts every dialogue on ×2 from
  its first token, so the rotation does not change when it crosses 262K. The only length checks are against the RoPE
  reach (`prefill.rs:2052`); `max_seq_len` is ignored for this architecture.
- **Sparse attention.** Flash-Next has 12 attention layers (every 4th of 48) and
  36 recurrent. Spliced content reaches the attention layers only, through QSA's
  indexer, which keeps `top_k = 2048` index rows per query per layer. "The model
  knows this file" means the indexer *can select* it.
- **Hot residency (estimate).** 2 KV heads × 256 × K+V × 12 layers; at C5
  roughly 1.6–1.8 GB of hot K/V per 250K tokens, plus ~0.5 GB of QSA index —
  more for many small turns, since index pages round up to 256 rows per layer.
  Content shared between dialogues is one copy.
- **Decode.** The indexer scores every page each step: ~720 MB read per sequence
  per step at 366K.
- **Churn.** Dialogues reproject every 64 tokens. Whole-file provenance turns
  over in large units; momentum (rather than the fresh score) is what keeps that
  turnover slow.
- **Speculative decoding.** Flash-Next's draft step builds plain `rope_cos_sin`
  beside the rung set (`qwen4exp/draft.rs:627-634`). Speculation is off on this
  model today (`draft_budget 0` at load), so it does not arise; before it is
  enabled, the draft head must be checked to rotate by the slot's rung past
  262K, or its drafts will be rejected at depth.

Budget ships at 250K (D10); the §7 measurement watches these numbers.

## 6. Code map

| Piece | Where |
|---|---|
| `WorkingSet`: seed / lock / restore / release / observe / provenance, per-file momentum | `candle-conversation/src/working_set/state.rs` |
| `WorkingSetConfig` (the `working_set:` block) | `candle-conversation/src/working_set/config.rs` |
| The marks, `is_dialogue` / `gather_tags`, `standing_locks` | `candle-conversation/src/working_set/marks.rs` |
| Per-dialogue sets on the substrate; `tombstone_timeline` drops from every set; eviction skips every dialogue's pins | `candle-conversation/src/substrate.rs` (`working_set_*`, `evict_hot_to_free`) |
| `observe` after the scan | `scheduler/mod.rs` (`reproject_view_prepare`), `Conversation::observe_working_set`, `projection/working_set_observe.rs` |
| `working_set` selection rule, `WorkingSetShare` | `projection/schema.rs`, `projection/yaml.rs` |
| Filling the groups; off the flexbox, emitted verbatim | `projection/project.rs` (Step 4b), `projection/working_set_pick.rs` |
| The dialogue group holds lineage + its own turns; members per group | `projection/resolver.rs` (`TargetedRead::group_turns`, `working_set_members`) |
| Pinned turn not hot is an error | `scheduler/projection_assembler.rs` (`is_promised`) |
| Screen: lock, first-write split, reworded reply | `zend/src/fast_path.rs` |
| Coverage check, cached per chain | `zend/src/fast_path/coverage.rs` |
| Registry check of `release_on`, alias-aware release, `round_marks` | `zend/src/working_set.rs` |
| Seed resolution | `zend/src/working_set/seeds.rs` |
| Locks restored from the marks | `zend/src/working_set/restore.rs` |
| Open, release on a user turn and on `release_on` rounds (live and resumed), marks on the results turn | `zend/src/session.rs` (`run_inference_stream`, `release_working_set`) |
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
  resumed turn does not, a `release_on` round does (live and resumed); the
  seeds are left to fade.
- The served reply carries no `error`/`detail` and scopes its promise to the
  current task (exact string).
- Config: an unknown `release_on` name fails the load.
- Marks: a round with served calls marks its results turn with a lock per
  served call in order; a user turn and a `release_on` round's results turn
  carry the release; reads served before a round's first write precede its
  release; a dialogue turn carrying only marks is still dialogue.
- Rebuild: locks come back from the marks after the last release, and none from
  before it; a served lock is restored even when a fresh screen would now refuse
  it; a tombstoned timeline is dropped; lock order matches mark order.
- The tags change no dialogue selection (a projection with and without them is
  identical).
- **Behavioural, daemon:** with the working set present in a tool round, the
  round answers the user's question, not an ingest request turn.
- **GPU, daemon stopped:** hot residency, QSA index bytes and decode t/s at
  0 / 116K / 250K of working set.

## 8. Decisions

| # | Question | Decided |
|---|---|---|
| D1 | Past the 262K trained length | Progressive YaRN ×2 reaches 524,288; the schema floors every sequence at ×2 (`rope: min_yarn_factor: 2`). |
| D2 | What release does to locks | Demote to provenance with momentum `1000/β` (the settled level of a full-strength hit). |
| D3 | Tie-break | Most recently locked wins. |
| D4 | Which calls release | File writes, git writes, `run_command`, `code_run`, `code_session_exec` — registry-checked names. |
| D5 | Where the working set sits | Its own layers' groups, which already emit ahead of the dialogue. |
| D6 | Tool rounds | `in_tool_rounds: true`; ingestion unchanged. |
| D7 | Provenance unit | Whole files and folders. |
| D8 | Seed list | §4.2 draft, to be edited. |
| D9 | Fade rate | Per-file momentum at β 0.2, leaking by its own value. |
| D10 | Default budget | 250K — with the 116K window, ~158K clear of the ×2 ceiling before YaRN climbs to ×4; measured alongside. |
| D11 | Momentum store | In `WorkingSet`, not the belief carry; in-memory. |
| D12 | Restart | Served calls are marked on their turn (`working_set:lock:<tl>`, `working_set:release`) through the existing persisted turn tags, which every gather-scope reader ignores; locks rebuild from the marks since the last release. Momentum rebuilds from the scan. |
| D13 | Folder vs file budget | `folder_tokens` slice for `repo_map`, the rest for `code_reading`. |
| D14 | A released file decaying out under an `in_context` reply | The system prompt's `in_context` rule says placed content fades when unused and to check its anchor before relying on it; a re-read re-locks it cheaply. |
| D15 | "Complete read" without touching ingestion | Derived from the chain's own `file_read` pages and responses, cached per timeline. |
| D16 | A pinned file that cannot be made hot | An error, never a silent drop; eviction protects every dialogue's pins. |
| D17 | Provenance floor | `min_momentum: 100`, the noise floor `code_reading` already gates at. |

## 9. Deliberately not done

- **No persistence of momentum.** It is a ranking, and rebuilds from the scan
  within a few reprojections.
- **No cross-group ranking.** Two budgets instead of a common score scale.
- **No rewriting of old replies.** A released file's `in_context` stays as it
  was sealed; the system prompt's rule (D14) carries the scope instead.

## 10. Build order

Each step lands with its tests and leaves the daemon working.

1. **Fix the fast path** (§3.1–3.6): refuse-don't-drop for pinned turns,
   first-write split, coverage check, tombstone removal, no LRU, real token cost,
   reworded reply. — built.
2. **`WorkingSet` + marks + rebuild**: locks, release, the marks, rebuild from
   the marks. — built.
3. **The `working_set` rule**: pinned content in the two layers' groups, off the
   flexbox, the dialogue splice deleted, seeds. — built.
4. **Momentum**: `observe` after the scan, provenance tier, `in_tool_rounds`. —
   built.
5. **GPU measurement** (§7) with the daemon stopped; the behavioural tool-round
   test.

   **Measured 2026-09-30, RTX PRO 5000 72 GB, Flash-Next, one live dialogue.**
   13 of the 17 seeds resolved (the other four had no complete conversation on
   the branch); a `file_read` of a committed file was served and locked; a
   restart restored that lock from the marks (`restored=1`) and the follow-up
   was answered correctly. The working set filled to the whole budget on the
   first scanned reprojection — the prefix went from ~1,255 blocks to 12,229
   (≈391K tokens, 135 turns) — because `min_momentum: 100` admits every file
   whose first fresh score clears 100, and most do. At that size:

   | | before | with a full working set |
   |---|---:|---:|
   | first reprojection of the turn | ≈0.28 s (full rebuild) | 6.2 s (3.9 s elevate, 1.4 s scan) |
   | each later reprojection | 0.085 s kept – 0.28 s rebuilt | 1.0–1.15 s (≈0.6 s apply, 0.24 s view) |
   | decode, single stream | 22–26 t/s | 14 t/s (19 t/s on a short follow-up) |
   | hot K/V | — | 1.48 GB, 5,814 resident turns |

   The prompt reported 381K–385K tokens. The model also read the pinned ingest
   conversations' `file_read` rounds as its own earlier reads ("Earlier I read
   README.md, ARCHITECTURE.md, zend_working_set.md …").
