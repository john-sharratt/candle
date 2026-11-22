# The working set — what a dialogue already knows about the code

**Status:** built (§10 steps 1–4); the GPU measurement of §7's last row is
§10 step 5. Replaces the fast path's least-recently-used budget
(`fast_path_window`, deleted); it does not sit beside it.

**Revised by `docs/hit_driven_projection.md` §7.2 and §12 items 18–26
(2026-11-18, built 2026-11-19), which supersede this document where they
differ. Of §4.2's block, `budget_tokens`, `folder_tokens`, `guidance_tokens`,
`max_file_tokens` and `max_admits` are gone: the pool's size is the card's
(`working_set::PoolSize`), and the slot's caps are `shares:`.**
- The working set's momentum, seeds and withheld set rank the **pool**, the
  fill's candidate set of whole conversations holding index pages only.
  Whole-file membership charged against `budget_tokens`, `max_admits`, the
  dislodge margin and the `max_file_tokens` refusal (§4.3, §4.4) go.
- The slot fill places whole turns under caps that are shares of the room:
  readings 22.5%, folders 2%, guides 5%, journals 10.5%, the dialogue 60%.
  `budget_tokens`, `folder_tokens`, `guidance_tokens` and `max_file_tokens`
  (§4.2) become those shares.
- Type shares (§4.10) bind on the pool.
- **Locks are removed** (built). zend runs every `file_read` for real (§4.5),
  so the lock tier went with the screen that served it: `WorkingSet::lock`,
  release, the lock marks and their rebuild, `lock_tokens`, `release_on`, and
  the refusal of a pinned turn that is not hot. This document no longer
  describes them; §4.6 and §4.8 record why they existed.

The other items are rewritten here as each lands.

## 1. The idea

Every dialogue carries a **working set**: a voluntary token budget (64K) of
already-ingested `code_reading` files and `repo_map` folders, projected ahead of
its own turns as **one sequence in the order its members entered**. The unit is
always a **whole ingest conversation** — one file or one folder.

1. **Provenance** — the files the conversation keeps attending to, by a
   per-file momentum score.
2. **Seeds** — documents, folders and files named in `projection.yaml` that a
   conversation starts out attending to: its first provenance, at the momentum
   of one full-strength hit (1000), fading like any other file it stops
   attending to.

**Every `file_read` runs for real** (§4.5). The working set once answered a read
of a file the corpus held whole with an `in_context` pointer and locked that
reading in; the content was placed, but in the short term the model did not
tie the pointer to it, and a re-fetch got the same pointer
(`docs/archived/zen_coding_session_fixes.md`, Cause 2). The page now arrives as a tool
result in the dialogue, and a file the turn's loop has read is kept out of the
working set until the next question, so it is never carried twice.

```
[system + priming chain][ provenance, in insertion order ][ dialogue … question ]
                                                         ^ every member enters here
```

What every conversation must always carry is the **priming chain**'s — the
repository roots, `README`, `ARCHITECTURE`, `AGENTS.md` — inherited through
lineage, not held by the working set. A new dialogue records the **seed alias**
(`forked_from = "seed"`, `Substrate::SEED_PARENT`) as its parent, never a
timeline id: each member resolves it to the end of the chain it holds now
(`Substrate::set_seed`, named by every prime: at boot, and whenever a link
changes). The chain's links are
ingest units every member holds, so the seed is there whichever member serves
the conversation, and a chain rebuilt on a later boot is the one every existing
conversation opens on — nothing on the conversation is rewritten. Prime also
re-points any conversation still parented on a boot's base conversation at the
alias (`ConversationEngine::reseed_base_children`).

Every member enters at the **end** of the sequence, beside the dialogue, so
the newest provenance sits nearest the question. A member keeps its place until
it leaves, and the gap closes at once. A newcomer that needs room dislodges the
weakest provenance only when it is clearly stronger.

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

The working set is **not** part of the dialogue group. Each member is a member
of its **own layer's group** — a file of `code_reading/scopes`, a folder of
`repo_map/structure`. For a target that declares a working set the projection
emits (`projection/project/pipeline/sink.rs` Step 12):

```
system prompt
inherited chain         the dialogue group's lineage turns
repo_map/structure      provenance folders       ┐ working set: provenance ≤ budget_tokens
code_reading/scopes     provenance files         ┘
dialogue                its own turns: recent 16 + historical 8, its own 180K window
```

On an engine whose attention selects, this is the seating order of one slot
fill (`docs/hit_driven_projection.md` §7.2): the lineage, the working set and
the dialogue's exchanges are placed together by score after their floors, and
carved back into this order.

The working set sits between the conversation's opening and its own turns. In
plain layer order it sat behind every other layer's groups and the whole
inherited chain: measured 2026-10-24, a paper the model had been served sat
~130K tokens before the question, where the model could not find it.

This one move fixes §3.7 (the dialogue's budget no longer carries the working
set) and §3.8 (the working set is always ahead of the dialogue).

**Order within a group: the working set's own — insertion order.**
`working_set_pick::pick` emits the group's members exactly as the `WorkingSet`
sequence holds them (§4.3): provenance in the order it entered, each at its
momentum. Each conversation's turns come in order, and `projection/project/pipeline/sink.rs` does not
re-sort a working-set group.

The order is built so that **changes happen on the right.** RoPE attention
depends on the distance between a key and the query. A change at one position
leaves everything to its right at the same distance from the question (it
shifts with the question) and everything to its left at the same distance from
the system prompt. Every change to the sequence is an append or a removal, so:

- **Long-lived provenance is absolute-stable.** Nothing enters to its left, so it
  keeps its place beside the system prompt.
- **The churn lands beside the dialogue.** Newcomers enter at the end and the
  weakest — usually the youngest — are dislodged there. Retrieval falls off with
  distance (measured 2026-09-30 on Flash-Next: a file at the start of a
  ~390K-token prompt could not be read back, the same file near the end was
  read correctly), so the newest provenance sits nearest the question.
- **A rebuild keeps the slot's prefix up to the first piece that differs**
  (`scheduler/piece_identity.rs`), so a change re-places only what lies to its
  right; nothing to its left is rebuilt.

(2026-10-01: this replaced two earlier orders — provenance by momentum band,
which moved a member each time its momentum doubled or halved, and provenance
newest-first, which put every newcomer at the far left and rebuilt the whole
working set behind it.)

The distance limit is not solved by ordering: as the dialogue grows, everything
ahead of it moves further back. That remains open.

**Budget.** The two groups are sized by the working set, not by the flexbox.
On a selecting engine the shares below are caps on the slot fill
(`projection/fill.rs`), which places the working set and the dialogue in one
pass; on any other engine `projection/project/pipeline/sink.rs` takes the working set's selections off
the top and distributes the dialogue's `window` over the rest. Their
`budget:` shares in `projection.yaml` are deleted.

- `repo_map` — provenance folders up to `folder_tokens`.
- `code_reading` — provenance files up to what remains of `budget_tokens`.

**What fills the voluntary budget today (open, §4.3).** Measured on a
conversation about the paper, 2026-10-24/25:

| Voluntary budget | What filled it | Effect |
|---|---|---|
| 250K | ~215K–248K of provenance on momentum; 321K–350K materialized per decode | the paper the model asked for sat ~125K tokens from the question |
| 80K | ~70K of test files (`kernel_layout_tests.rs` 32K, `decode_layout_exact.rs` 15K, …), the same on every question | the model misattributed facts between those files and the paper |
| 40K | one 32K test file, ranked first on every question | — |
| 20K | `decode_layout_exact.rs`, a flash-attn header, a wasm worker, game asset folders | — |

The scan cannot rank files against each other, so what the budget holds is
largely arbitrary until the scores momentum is fed are made comparable across
files (§4.3). Shrinking the budget hides that rather than fixing it, and with
it provenance.

**No link of the dialogue's own chain is ever a member.** The chain is projected
as the conversation's opening already; admitted on momentum it emitted twice —
measured: `AGENTS.md`'s 46K characters in both the working set and the chain
(`Substrate::working_set_candidate`).

Two budgets because the two groups' scores are normalised separately and cannot
be ranked against each other; folders are small, so a fixed slice is enough.

### 4.2 Configuration

On the dialogue layer, replacing `fast_path_window` (deleted):

```yaml
  - name: dialogue
    working_set:
      reserve_tokens: 16384      # what arrives between two projections; the slot is the engine's trained window less this
      budget_tokens: 64000       # seeds + provenance, folders and files together
      folder_tokens: 10000       # repo_map's share of budget_tokens
      guidance_tokens: 16000     # the guidance pool, beside budget_tokens
      beta: 0.2                  # per-file momentum leak (§4.3)
      min_momentum: 100          # below this a file is not a provenance candidate
      max_file_tokens: 48000     # the most of one reading the slot places (§4.4)
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
```

- Seeds leave out what the priming chain already carries — the repository roots,
  `README`, `AGENTS.md` — because lineage emits them anyway. Resolution skips any
  seed whose timeline is in the dialogue's `inherited_chain`, so an overlap in
  the list costs nothing.
- Every seed sits within `--max-depth 3`; a deeper file has no conversation.

### 4.3 State and momentum

One reprojection admits at most `working_set.max_admits` provenance newcomers
(default 2). The rest keep their momentum and enter on later reprojections. See
`docs/projection_swap_cap.md`.

The substrate keeps one `WorkingSet` per dialogue,
`HashMap<TimelineId, WorkingSet>`, in `candle-conversation/src/working_set/`:

```rust
pub struct WorkingSet {
    members: Vec<Member>,                // emission order: { timeline, tokens, share }
    momentum: HashMap<TimelineId, f32>,  // members and candidates
}
```

The **budget is enforced on entry**, so membership is state, not a fresh pick
each projection: an unchanged set projects identically. Each member draws on its
group's share — folders up to `folder_tokens`, files on what `budget_tokens`
leaves — which the substrate knows per group (`Substrate::set_working_set_share`,
set at setup from each `working_set` rule's `share`). Every operation:

| Operation | What happens |
|---|---|
| `observe` | Momentum decays for everything; a conversation under `min_momentum` leaves the map and, if a member, the sequence — the gap closes. A member its scope no longer offers leaves too. Then the strongest non-members try to enter at the end, each dislodging only provenance it beats by `DISLODGE_MARGIN` (×1.5) — without the margin two files near the floor trade places every reprojection and each trade rebuilds everything to its right. |
| `seed` | At open, in list order: each seed gains `seed_momentum` (1000) and enters at the end while the budget has room — seeds dislodge nothing. One that misses for room stays a candidate. |
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

**Open: the scores momentum is fed cannot rank files against each other.** A
raw vote is `z × margin` inside the file's own competition
(`belief_files::competitions`): it says how far one exchange stands out from
the file's other exchanges, not how well the file answers the probe. The
normalised score divides by each exchange's learned level, and its band runs
~1000× over on short probes — momentum took a one-off match at 643,768
(measured 2026-10-24). On a conversation about the paper the provenance budget
filled with test files, a plotting script and game asset folders on every
question (§4.1).

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

**Momentum need not survive a restart.** It is a ranking, not a promise; it
stays in-memory and rebuilds from the scan within a few reprojections.

### 4.4 Projection

A `working_set` selection rule, declared on `repo_map/structure` and
`code_reading/scopes`, replaces their `top_k` —
`selection: { kind: working_set, share: folders }` on the folder group,
`{ kind: working_set, share: remainder }` on the file group
(`SelectionRule::WorkingSet { share }`; a collection may not declare it):

the group's working-set members, whole, in the working set's order (§4.3) —
nothing is chosen at projection time.

**A reading longer than `max_file_tokens` is carved, not refused.** On an
engine whose fill carves the slot (`docs/hit_driven_projection.md` §7.2),
the fill places at most `max_file_tokens` of one reading: its first and last
turns and the page turns that score highest, each whole (§8.1 there). So a
reading longer than the cap is admitted as a seed or as provenance, and
charged the cap against the budget, not its whole size
(`Candidate::carved_to`, applied by `Substrate::provenance_charge`). On an
engine that emits members whole, it is never a member.

Measured 2026-11-18, the §9.12 battery's answers sit in five files past the
cap: `scheduler/mod.rs` (306K tokens), `quantized/cuda.rs`,
`conversation.rs`, `region_pool.rs` and `projection_assembler.rs`. Refused
whole, they could never be placed, and two questions had no other answer
file (`docs/results/dependency_battery_gap_analysis_2026-11-18.md`). The same day
showed the fill had never run on the daemon: `SubstrateRead`'s
`ContentResolver` took the trait's default `trained_window` of `None`, so
every dialogue projected as if its engine did not select. Across 30 prompts,
all 2,187 placed turns were whole, and the working-set groups held about 75K
against the 64K budget each time.

The rule reads the dialogue's `WorkingSet` through the resolver
(`ContentResolver::working_set_members`, filtered to the group); a target whose
layer declares no `working_set` — every ingest conversation — selects nothing,
so ingest projections are unaffected. An ingest conversation generating into
one of these groups reads its own turns there as it always did. Provenance is
filtered by the conversation's retrieval scope like any other selection from a
scoped group, and leaves the set at the next observation.

**Off the flexbox, by an existing precedent.** A `score_density` group already
has its picks decided upstream and emitted verbatim, skipping the bounded pass
(`projection/project/pipeline/sink.rs:1851-1864`). A `working_set` group takes the same path, and one more
step: its natural consumption is left out of `layer_items`/`group_items`
(`projection/project/pipeline/sink.rs:1768-1848`), so the flexbox distributes the dialogue's `window`
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
lineage and its own turns only, and its 96K window is its own again.

### 4.5 Reads — every one runs, and the loop withholds what it read

**There is no screen.** A `file_read` runs and its page is prefilled as the
tool's response, like every other call. `zend/src/fast_path.rs`, the lock
promises and their restore are gone, and so is the `in_context` reply. No
system-prompt rule tells the model that placed content fades or to check an
anchor before stating a fact: that rule sent it to re-read the seed it held
(2026-11-11). `grounding` covers fetching what is not here and never inventing
a file's contents.

**The loop withholds what it read** (`zend::session::withhold_read_files`,
`zend::working_set::corpus::readings_of`, `WorkingSet::withhold`). After a
round, every successful `file_read`'s file, at the version the read returned,
names its ingested readings, and those are kept from entering the working set
for the rest of the turn's loop:

1. **Entry is blocked; nothing is evicted.** A withheld conversation cannot be
   admitted, but keeps its momentum. One already placed when the model reads
   it stays where it is: evicting it would close the gap and re-prefill every
   later member and the whole dialogue, for one file's tokens.
2. **The whole file.** A read of any page withholds the file's readings.
3. **Until the next question.** A new user turn clears the set
   (`working_set_clear_withheld`): the reads may be summarised away by then,
   and provenance may bring the file back. A resumed round keeps its loop's.
4. **A file the conversation changed withholds nothing**: its read returned
   its own copy, which no reading holds.

**Tombstone removes.** `tombstone_timeline` drops the timeline from every
working set, momentum and seeds included (§3.4).

**A selected turn with no hot K/V is dropped with a warning**
(`apply_projection`). No member is a promise the model was told it holds, so
none is an error; `working_set_pins` protects the projection being elevated
from eviction.

### 4.6 Locks, and why they went

The screen (`zend/src/fast_path.rs`) once answered a `file_read` of a file the
corpus held whole with an `in_context` reply naming the page's anchor, and
locked that reading into the working set: on top of the voluntary budget,
never dislodged, and released only when the promising reply left the
projection, a write named its file, or a command ran. Locks were marked on
turn tags (`working_set:lock:<tl>`, `working_set:release…`) and rebuilt from
them on restart, and a locked turn that could not be made hot failed the
projection.

Two findings shaped it and still hold for anything placed. **A new user message
must not end what the model is still reading**: measured 2026-10-24
(conversation `6e8ad6ea`), a paper released at a new message faded while its
reply stayed in view, and the model called its own earlier quotations
fabricated. **A placed part must name its file next to its lines**: every
part's block opens on a fence carrying `file=<repo>/<path> part=… lines=…`
(`zend_tools::tools::file::render`), because with one header 200 lines away the
model bound content to the wrong file (2026-09-30).

The screen went first (§4.5): in the short term the model did not tie the
pointer to the placed content, and a re-fetch got the same pointer. With every
read running for real, nothing drove a lock, and the tier, its marks, its
restart and `lock_tokens` / `release_on` were deleted
(`docs/hit_driven_projection.md` §12 item 26).

### 4.7 Seeds

Resolved per dialogue once, when it opens: a file through `CONTENT_KEY` at the
blob the dialogue's branch holds, a folder through `folder_unit`. A miss
(changed, not ingested, in the lineage) is skipped. Each resolved seed gains the
seeding momentum and enters provenance in list order while the budget has room,
dislodging nothing (§4.3) — so the seeds are the leftmost members, beside the
system prompt and the priming chain. One that does not fit, has no recorded
size, or, where members are emitted whole, is past `max_file_tokens` (§4.4) is
left out with a warning naming it; one
that missed only for room stays a candidate and can enter later on its
momentum.

**A request's first reading holds the seeds out.** While a new question waits
for its first boundary, the seeds the set was given are left out of the
projection (`WorkingSet::holds_out`): the question is read and scored once
without them, and then again with them and what it ranked
(`docs/hit_driven_projection.md` §7.4). They stay members throughout, with
their momentum; only that first reading does not see them.

### 4.8 Restart

A working set is rebuilt at open: momentum starts empty and the seeds re-enter
it at the seeding momentum (§4.7); the scan rebuilds the rest within a few
reprojections.

**Marks are not gather scope.** Dialogue turns written while locks existed
carry `working_set:` tags. A turn's tags also name the tag-scoped galleries it
belongs to, and a turn with no tags is what ordinary dialogue is: the untagged
belief gallery, the normalization warm-up's dialogue replay and the seal-time
observation all select on it. Every such reader goes through
`marks::gather_tags` / `marks::is_dialogue`, so a turn carrying only those
marks stays dialogue.

### 4.9 Faces — a reading reaches a dialogue without its request

An ingest unit opens on the request that commissioned its reading: a file's on
"Read the entire contents of `X` … The file is N lines long … summarize …", a
folder's on "Summarize the `X/` folder …". The reader has to be asked. A
dialogue that places the reading must not hold that request, because a user turn
reads as the user's. Measured 2026-11-07: with the working set present, 6 of 12
workers answered a placed request in place of their own task, describing a
file nobody had asked about in the layout the request asks for, and quoting
the request's line count as the size of the file they were given. A section of
the system prompt saying "those requests are not yours" did not hold against a
dozen of them framed exactly like the user's own.

**Each unit has a face.** A face is a one-turn conversation on the reserved
`Face` layer (`Reserved::Face`). It is framed by the prelude every dialogue
primes on. Every face of one fill forks a single base conversation, so the
prelude is primed once per fill.

**A face keeps no recurrent snapshot**, like every ingest unit
(`zend::ingest_snapshots`). It is noted unresumed before its turn seals. A
snapshot is the whole recurrent state, 65–112 MB on Flash-Next whatever the
turn's length. The first backfill sealed faces without the mark: about 4 GB of
redo log a minute across a corpus of ~3,400 units, which filled WhiteWorld's
substrate drive (2026-11-07). The load-time sweep `retire_finished` retires any
snapshot a face kept. Its user half is only the reading's anchor, `file=<repo>/<path>` or
`folder=<repo>/<dir>` (`zend::faces::anchor`). Its assistant half is the unit's
own opening, the calls it made, prefilled. The unit names its face under
`face`, and the face names its unit under `face_of`. Both are plain `custom`
entries, like `forked_from`.

**A dialogue holds the opening by the face.** Every selected turn becomes a
`ResolvedTurn` at one place (`projection/project/resolved.rs`). There, a projection
whose target is not an append-only ingest layer emits `Substrate::face_of(turn)`
in place of a unit's opening turn, when the unit has a live, sealed face. The
face is a turn like any other, with its own K/V, layout and index pages, so
nothing is windowed and nothing goes in unindexed. A unit being read projects
its own opening, which is the request it is answering.

**What was selected is still the unit's opening.** Selection, belief, coupling
and the window all run on the unit's turns. The projection event records the
opening (`Substrate::unit_of`) with its score, origin and belief carry, and only
its token count is the face's. A face's own layer is reserved, so no retrieval
offers it as content of its own. The materialized spine names the face, so the
projection panel, which draws its cards from the spine, is sent the bodies of
the spine's turns as well as the selection's (`zend/src/api/projections.rs`,
`panel_turns`). Sent the selection's alone, every face's card had no body and
drew blank under the face's token count (2026-11-11).

**Rejected alternatives, recorded because each was proposed.**

- **Cutting the request out of the sealed turn at projection.** K/V can be
  windowed only where a turn's index pages break (`kept_outside`). Pages break
  where a unit begins, at the first decoded token and after `</think>`, never at
  the question's edges. So a file's question shares its first page with its
  role markers, and a folder's whole request turn (question and prefilled call)
  is one page. The stored rows cannot be kept for cut K/V either. A row is the
  indexer's `k_proj` of hidden states, pooled over `ratio` tokens and placed by
  position, so rows for a missing span point QSA at whatever now sits there.
- **Rewording the ingest request and moving the instruction into the layer's
  system prompt.** Every file would have to be read again, which is hours at
  ingest's rate. Until that finished, the old units would carry their requests,
  giving two paths or an outage.

**Lifecycle.** `zend::faces::fill` runs at the start and at the end of every
branch-ingest pass. A pass can queue thousands of readings, and a unit already
read must not wait out the whole pass for its face. It seals a face for every
committed file and folder unit without a live one,
each under a `face/<content key>` lease so two members do not both seal it. A
unit read this pass and one read before faces existed meet the same rule, and
filling is one short prefill per unit, never a re-read. A face names its unit
before its turn is prefilled, and the unit names the face only once that turn
holds, so a face that failed part-way is never pointed at. Tombstoning a unit
tombstones its face (`Conversation::tombstone_timeline`). A face that is gone
or not yet sealed reads as absent, and the unit's own opening stands.

### 4.10 Type shares — no kind of document takes the whole budget

**Why.** The leak-free dependency battery (24 questions of the form "where is
`tier_fits` defined and who calls it?", BlackWorld, 2026-11-16) found the
answers drawn from the same few documents nearly every time — `README.md`,
`AGENTS.md`, `ARCHITECTURE.md`, `docs/*.md`, the MCP guides — and almost never
from the code file that answers the question. A hub document matches nearly any
question about the code it describes, so on momentum documents filled the
voluntary budget and the code that ranked below them never entered.

**What.** The dialogue's `working_set:` names types of document and the most of
`budget_tokens` each may hold:

```yaml
      type_shares:
        - name: docs
          files: ['\.(md|markdown|mdx|txt|rst|adoc)$']
          max_share: 0.35
        - name: config
          files:
            - '\.(ya?ml|toml|json|jsonc|ini|cfg|conf|env|properties)$'
            - '(^|/)(Cargo\.toml|package\.json|pyproject\.toml|go\.mod)$'
          max_share: 0.15
```

A unit's type is the first type a pattern of which matches its subject — the
`path` a `code_reading` unit records, `<repo>/<path>`. A unit matching none —
code, tests, kernels, scripts, and every folder — is unbounded. Shares are in
`(0, 1]`, names unique, every pattern must compile; a declaration that breaks
any of those is refused at load (`working_set::TypeShares::new`).

**How it applies.** A share is a cap, not a reservation: it is checked where
`folder_tokens` is, at admission (`WorkingSet::room_for`). A typed newcomer
whose type would pass its share dislodges only weaker voluntary members of its
own type — by the same `DISLODGE_MARGIN` — so a burst of strong documents can
take each other's places but never the room the code would have. When it beats
none of them it waits on its momentum, and the next-best candidate by score is
considered. Seeds keep to the shares too. When the budget shrinks under a
standing set, `observe` trims each type's weakest members back under its share
(`WorkingSet::trim_types`). Guidance is its own pool and holds no type's
share. Provenance orders everything inside the caps exactly as before.

The substrate types each candidate (`Substrate::working_set_candidate`) from the
classifier zend installs at setup beside the groups' shares
(`ConversationEngine::set_working_set_types`), and every member carries its type
and cap.

## 5. Constraints and cost

- **Context length.** Flash-Next is trained to 262,144 positions, and the
  dialogue's slot stays inside them (`docs/hit_driven_projection.md` §4): the
  slot is the engine's trained window less the 16K reserve, and one fill
  places everything in it after the system prompt — floors, then rounds by
  score (`hit_driven_projection.md` §7.2). The 64K voluntary, 16K of guidance
  and the dialogue's 180K window are caps on that fill, not reservations, so a
  thin working set leaves its room to the dialogue and a short dialogue leaves
  it to the working set. A reserve that passes the trained window fills YaRN
  ×2 with one warning. A slot's rung follows its deepest
  position, so a dialogue inside the trained window runs the trained RoPE at
  every position. Progressive YaRN (`qwen4exp/rope.rs`) still carries a
  sequence a caller drives deeper ×2 to 524,288. The only length checks are
  against the RoPE reach (`scheduler/prefill/first_samples.rs:2052`); `max_seq_len` is ignored for this
  architecture.
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

Budget ships at 64K voluntary + 16K of guidance (D10), inside the
trained window's slot with the reserve, the conversation fitted to what is left; the §7
measurement watches these numbers.

## 6. Code map

| Piece | Where |
|---|---|
| `WorkingSet`: seed / observe / remove / withhold, per-file momentum | `candle-conversation/src/working_set/state.rs` |
| `WorkingSetConfig` (the `working_set:` block) | `candle-conversation/src/working_set/config.rs` |
| `is_mark`, `is_dialogue` / `gather_tags` | `candle-conversation/src/working_set/marks.rs` |
| Per-dialogue sets on the substrate; `tombstone_timeline` drops from every set | `candle-conversation/src/substrate/registry.rs` (`working_set_*`) |
| `observe` after the scan | `scheduler/mod.rs` (`reproject_view_prepare`), `Conversation::observe_working_set`, `projection/working_set_observe.rs` |
| `working_set` selection rule, `WorkingSetShare` | `projection/schema/mod.rs`, `projection/yaml/mod.rs` |
| Filling the groups; off the flexbox, emitted verbatim | `projection/project/pipeline/sink.rs` (Step 4b), `projection/working_set_pick.rs` |
| The dialogue group holds lineage + its own turns; members per group | `projection/resolver/targeted_read.rs` (`TargetedRead::group_turns`, `working_set_members`) |
| Withheld files: the set of files a turn's loop read for real, kept out of provenance's newcomers | `candle-conversation/src/working_set/state.rs` (`withhold`, `clear_withheld`), `substrate/registry.rs` / `engine/retrieval.rs` (`working_set_withhold`, `working_set_clear_withheld`) |
| Which ingested readings stand for a file, and whether a reading covers it whole | `zend/src/working_set/corpus.rs`, `zend/src/working_set/coverage.rs` |
| The dialogue layer's `working_set:` block | `zend/src/working_set.rs` (`dialogue_config`) |
| Seed resolution | `zend/src/working_set/seeds.rs` |
| Open, the read files withheld after each round, cleared at a new question | `zend/src/session/inference_stream/stream_turn.rs` (`run_inference_stream`), `zend/src/session/working_reads.rs` (`withhold_read_files`) |
| Layer config | `zend/src/prompts/projection.yaml` |
| Faces: `face_of` / `unit_of`, the `face` / `face_of` keys | `candle-conversation/src/substrate/faces.rs` |
| A dialogue holds an opening by its face; the event records the unit | `projection/project/pipeline/sink.rs`, `projection/event/classify.rs` (`build_selection`) |
| A unit's tombstone takes its face | `projection/resolver/lifecycle.rs` (`Conversation::tombstone_timeline`) |
| Sealing faces after each pass, the `face/` lease | `zend/src/faces.rs`, `zend/src/branch_ingest/mod.rs` (`readings`), `zend/src/ingest_lease.rs` |

## 7. Tests (CPU unless marked)

- `WorkingSet`: a newcomer enters at the end and dislodges only provenance it
  beats by the margin; a link of the dialogue's own chain is never seeded or
  admitted; `observe` settles a steady file at `fresh/β` and decays an idle
  one; a file below `min_momentum` leaves the map and is never a candidate; a
  withheld file keeps its momentum but does not enter; a tombstoned timeline
  leaves the set.
- Selection: members emit whole, in the set's order, at their momentum;
  `folder_tokens` bounds `repo_map` and `code_reading` gets the rest; the
  flexbox distributes the dialogue's `window` exactly as it would with no
  working set (the dialogue group's selection is identical with and without a
  full working set); a focused projection carries no working set; an ingest
  target selects nothing.
- Coverage: a reading whose pages cover `0..ceil(lines/PAGE_LINES)` covers its
  file; one missing a page, or whose page response was an error, does not.
- Hot: a selected turn with no hot K/V is dropped with a warning.
- Marks: a dialogue turn carrying only `working_set:` marks is still dialogue.
- **Behavioural, daemon:** with the working set present in a tool round, the
  round answers the user's question, not an ingest request turn.
- **GPU, daemon stopped:** hot residency, QSA index bytes and decode t/s at
  0 / 116K / 250K of working set.

## 8. Decisions

| # | Question | Decided |
|---|---|---|
| D1 | Past the 262K trained length | The dialogue goes there only when its reserve passes the trained window, and then fills YaRN ×2 with one warning; otherwise the slot fill stays inside the trained window, and every sequence takes the rung its reach calls for (`docs/hit_driven_projection.md` §4, §7.2). |
| D2–D4 | Locks: release, tie-break, which calls release | Withdrawn with the lock tier (§4.6). |
| D5 | Where the working set sits | Its own layers' groups, which already emit ahead of the dialogue. |
| D6 | Tool rounds | `in_tool_rounds: true`; ingestion unchanged. |
| D7 | Provenance unit | Whole files and folders are what enters and leaves. Where the fill carves the slot, a reading past `max_file_tokens` is a member charged the cap, and the fill keeps its floor and best-scored page turns (§4.4). |
| D8 | Seed list | §4.2 draft, to be edited. |
| D9 | Fade rate | Per-file momentum at β 0.2, leaking by its own value. |
| D10 | Default budget | 64K voluntary, guidance 16K, caps on one fill of the trained window less the 16K reserve, beside the dialogue's 180K window (`docs/hit_driven_projection.md` §4, §7.2). Strength is momentum lifted by the model's own hits (§4.3, `hit_driven_projection.md` §7.1). |
| D11 | Momentum store | In `WorkingSet`, not the belief carry; in-memory. |
| D12 | Restart | Nothing is restored: seeds re-enter at open and momentum rebuilds from the scan. |
| D13 | Folder vs file budget | `folder_tokens` slice for `repo_map`, the rest for `code_reading`. |
| D14 | A file decaying out under an `in_context` reply | No reply says `in_context`: every read runs (§4.5). |
| D15 | "Complete read" without touching ingestion | Derived from the chain's own `file_read` pages and responses. |
| D16 | A selected file that cannot be made hot | Dropped with a warning; nothing placed is a promise. |
| D17 | Provenance floor | `min_momentum: 100`, the noise floor `code_reading` already gates at. |

## 9. Deliberately not done

- **No persistence of momentum.** It is a ranking, and rebuilds from the scan
  within a few reprojections.
- **No cross-group ranking.** Two budgets instead of a common score scale.
- **No rewriting of old replies.** An `in_context` sealed while the screen
  existed stays as it was.

## 10. Build order

Each step lands with its tests and leaves the daemon working.

1. **Fix the fast path** (§3.1–3.6): refuse-don't-drop for pinned turns,
   first-write split, coverage check, tombstone removal, no LRU, real token cost,
   reworded reply. — built.
2. **`WorkingSet` + marks + rebuild**: locks, release, the marks, rebuild from
   the marks. — built, and removed with the lock tier (§4.6).
3. **The `working_set` rule**: the set's members in the two layers' groups, off the
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
