# Behavioural test catalogue — conversational memory on a recurrent stack

Companion to `deltanet_state_persistence.md`. That document describes an
implementation; this one describes what a conversation must **do**, in terms a
test can assert without knowing how any of it works.

## Why this is written implementation-independently

The defect class this whole area exists to remove is *fluent amnesia*: a model
whose memory has silently gone reads perfectly, matches every shape, and errors
nowhere. Tests written against the implementation inherit its blind spots —
they assert that `fork_recurrent` copied a buffer, which is true and says
nothing about whether the conversation remembers anything.

So every test below is stated as an **invariant over observable behaviour**:
given a scenario a user can produce, the conversation must behave thus. Where an
invariant has no observable today, that is recorded as an enablement item rather
than worked around, because a scenario nobody can observe is a scenario nobody
is testing.

Two consequences worth stating up front:

- **A test that cannot be expressed is a finding.** Three of the entries below
  (B7, A7, C4) could not be written without first answering a design question,
  and one of those — B7 — turned out to be a live gap on the `code_read` path.
- **The assertions are on memory, never on text.** Text is corroboration. A
  conversation that has forgotten everything still produces fluent, on-topic,
  grammatical replies, so an output-shaped assertion cannot distinguish the
  failure from success. The one exception is Family G, where the *quality* of
  the text is the measurement — and there the control matters more than the
  metric.

## The control

Family G needs a conversation that has full K/V and no memory — the exact
condition §1 describes. That is reachable through a production path with no test
hook and no flag: **write a snapshot whose schedule hash does not match the
loaded model, then resume from it.** The restore refuses (correctly, loudly),
the K/V splices in full, and the conversation runs with its recurrent layers at
zero.

This is the amnesia control. Every quality claim below is a paired comparison
against it, because "the model answered well" is not evidence and "the model
answered well *and better than the same model without its memory*" is.

## Tiers

| tier | cost | where |
|------|------|-------|
| **2** | one model load, minutes | `candle-transformers` (model hooks) |
| **3** | full daemon, minutes–tens of minutes | `zend/tests/` |
| **3-cruise** | tens of turns × repeats, ~1 h | `zend/tests/`, run deliberately |

Everything is `#[ignore]`d with its cost and run line, matching the existing
convention.

---

## Family A — Continuity

*A conversation's memory of its own history is complete, and does not depend on
how the engine arrived at it.*

Every entry is the same oracle in a different costume: **do the thing, and do
the thing with an interruption in the middle, and compare.** It needs no
reference implementation, which is what makes it decisive when the reference is
itself suspect (§6.5).

| # | Invariant | Tier | Needs |
|---|-----------|------|-------|
| **A1** | N turns run straight through, and the same N turns with a **daemon restart** midway: the restored memory, every recovered turn's K/V content, the layout, and the projected context are **byte-identical** to the uninterrupted twin, and the conversation continues. *(Token-identical continuation is deliberately not asserted — see the note below.)* | 3 | — |
| **A2** | The same, with restarts at **several** points — the invariant must not depend on where the seam falls. | 3-cruise | — |
| **A3** | The same, with a **reprojection** midway (the projection changes; the conversation does not). Memory follows the token stream the conversation actually saw, not the reprojected order. | 3 | — |
| **A4** | The same, with the conversation's K/V **migrated to warm and cold and elevated back**. Memory is not tiered; a migration must be invisible to it. | 3 | E7 |
| **A5** | The same, across a **compaction**. | 3 | E2 |
| **A6** | The same, across an in-place **substrate reload** (no process restart). | 3 | — |
| **A7** | The same, with a **compression/summary turn** inserted midway. *See the note below — this needs a decision first.* | 3 | decision, E1 |
| **A8** | A restarted conversation **carries its behaviour**: resumes of identically-built sealed conversations both recall their topic. *(Token-identity between them is printed as a diagnostic, not asserted — same grounds as A1.)* | 3 | — |
| **A9** | Memory at turn N is reachable from turn 0 for every N — no depth at which continuity quietly stops. | 3-cruise | E1 |

> **A7 — decided, by measurement.** Compression replaces a span of turns with a
> summary in the projection, but the conversation's memory accumulated over the
> *originals* — a superset of what the K/V now shows. The decision: this is the
> **designed behaviour**, not a defect — it is the recurrent layers doing
> exactly their job, carrying what the attention window no longer shows. The
> executable form (`a7_recall_survives_a_compressed_span`) states a fact,
> drives the conversation until the summariser has produced a summary leaf over
> the fact's span, and gates the deterministic halves: compression engages, and
> the conversation still operates with its memory intact. The recall itself is
> REPORTED there and **measured in the cruise's deep arm** (`recall_quality`,
> depth past the raw tail, memory-vs-amnesia margin) — because single-shot
> recall of an ~11-turn-old fact across a compressed span sits near threshold
> on this model (observed 1-in-3 across runs), and this document's own rule for
> text assertions is "a single sample is an anecdote — fail on the margin."
> That near-threshold cliff is itself a finding worth the paper's attention:
> it is the O(1) state's fact-competition curve showing at measurable depth.
>
> **A7 could not even run at first, and that was the campaign's biggest catch:
> the summariser was entirely dead on this lineage.** Its instruction framing
> was pushed as `Generated` segments, which the assembler reserves as a
> gap-fill island — and `reserve_glue_island` (correctly) refuses islands on a
> model that cannot gap-fill. Every summary probe errored and soft-retried
> forever; no conversation on the recurrent lineage had ever been summarised,
> and the failure was invisible until the harness routed WARNs to stderr. The
> fix makes the framing a `NewUserMessage` — the deferred tail append the
> dialogue path already uses, computable by a recurrence because its state has
> accumulated through the whole prefix. The refusal did its job perfectly the
> whole time; nothing could hear it.

> **A1 failed the first time it ran, and it took two fixes — both on production
> paths, both invisible to every other oracle.**
>
> **Defect 1 — resume clobbered the restored memory with the base
> conversation's.** The daemon resumes a client's conversation by forking its
> *base* conversation onto the client's timeline, and the fork decided "may the
> child take the parent's live memory?" by *which constructor was called*
> rather than by *whose history the target timeline holds* — so every resumed
> conversation ran on the base conversation's memory, copied over the top of
> its own correctly-restored snapshot. The fix derives the inheritance from the
> one comparison that settles it (fork target == parent's own timeline),
> asserted at unit scope in `fork_inherits_history_tests`.
>
> **Defect 2 — resume seeded selection belief from nothing.** With the memory
> restore byte-perfect (instrumented: restored == sealed == the uninterrupted
> twin at the same boundary), the first resumed decode still diverged. The
> carried selection belief — what a submit-time projection seeds its
> section/tool selection from — is keyed by *slot*, and its comment declared it
> "empty only for a genuinely fresh conversation". A resumed conversation is
> not fresh, but its slot is: it selected from an empty prior while its
> uninterrupted twin carried a prior evolved over every turn, and near a
> selection gate (a tools-catalog entry, a grounding variant) that flips a
> section in or out of the projected context. Deterministic, marginal, and past
> every recall probe — the replies even reconverged within two turns as belief
> re-accumulated, while the memory digests never did. The fix
> (`restore_carried_belief`) folds each recovered turn's last persisted
> projection event in turn order — the same fold the live seal harvest performs
> — and is pinned at unit scope in
> `a_resumed_slot_seeds_carried_belief_from_the_recovered_selections`.
>
> The shared moral: a resume restores a conversation's durable state, and that
> state has **three** parts — K/V, recurrent memory, and selection belief.
> Each was restored correctly *individually* before A1 first ran; the two
> defects were both in what the resume *didn't know it had to carry*.
>
> **What A1 asserts, and what it deliberately does not.** After both fixes a
> residual reply divergence remained at the first resumed decode, bit-stable
> across runs. It was chased to ground with one instrument per theory, each
> asserted rather than assumed: the restored recurrent state (byte-equal to
> the sealed record *and* to the uninterrupted twin), every recovered turn's
> K/V content (formats, palettes, scales, raw arena bytes — byte-equal across
> the restart, `ReadTurnKvDigest`), the projected layout (block-equal), the
> opening selection, materialized glue, and exact prefill strings (equal), and
> wave-shape reproducibility (B8: a fresh slot recomputing all its glue in one
> wave continues bit-identically in-process). With **every input byte proven
> equal**, the residual divergence is argmax flipping on near-ties under
> engine-instance numerics — allocator and library state, the class the decode
> reproducibility campaigns documented — which no persistence machinery can
> remove, and which also produces the intermittent in-process divergence
> between identically-prompted content-rich siblings (C6 vs the topic pair:
> template turns sit far from ties and always reproduce; content-rich turns
> sit near them and sometimes flip). A1 therefore asserts the byte identities
> above plus continued operation; **token-identical continuation across engine
> instances is not a property this stack promises**, and the semantic half of
> the claim is carried by the recall gates (A8's topic recall, B1a, B6).

---

## Family B — Fork semantics

*A fork is a conversation that shares a past and owns its future.*

| # | Invariant | Tier | Needs |
|---|-----------|------|-------|
| **B1** | A fork continues **token-identically** to its parent from the fork point, under identical sampling. | 3 | — |
| **B2** | Turns taken on the fork never change the parent's memory. | 3 | — |
| **B3** | Turns taken on the parent never change the fork's memory. | 3 | — |
| **B4** | A fork taken at turn N has exactly the memory the parent had at turn N — not the parent's memory *now*. | 3 | — |
| **B5** | Forking mid-turn is **refused**, with an error that says why. A fork whose memory ran ahead of its K/V is unrepairable, so the refusal is the feature. | 3 | — |
| **B6** | Forking a conversation the engine no longer holds (post-restart) gives the same result as forking a live one. | 3 | — |
| **B7** | **Parallel scope ingest**: N scopes fork from one conversation, each ingests, and all their turns are spliced back. The parent's memory afterwards is *«decided semantics»*. | 3 | **decision**, E1 |

> **B7 was a live gap, decided as (c) and closed.** `splice_scope_turns`
> adopts each scope's sealed turns onto the file timeline **by reference** —
> K/V the parent's forward never saw. §4.4 calls this shape a *divergent
> join* and puts it out of scope — *"there is no operation that means 'and
> also these other tokens'"* — but the `code_read` path performs one today,
> and the adopted turns land at the **tail**, which a recurrence *can*
> absorb. The implemented answer is **(c)**: `catch_memory_up_to` re-prefills
> the adopted tokens through the parent's recurrence (one prefill per
> splice), and the test asserts all three consequences — the memory **moves**
> off the fork point, the parent can **recall the injected content** (the
> semantic proof, and on this architecture the load-bearing one: K/V alone
> reads as forgotten, so a catch-up that produced a wrong state would pass
> the movement check while leaving the conversation blind to every file it
> just read), and the whole operation is a **deterministic function of the
> adopted stream**. Rejected alternatives, for the record: (a) K/V-only
> contribution — measured equivalent to the conversation never having read
> the files; (b) adopting one scope's memory — arbitrary, and arbitrary is
> wrong here.

---

## Family C — Isolation

*Memory belongs to exactly one conversation.*

| # | Invariant | Tier | Needs |
|---|-----------|------|-------|
| **C1** | Several conversations interleaved through the engine each recall their own facts, and no two share memory. | 3 | — |
| **C2** | A new conversation started after many others have closed inherits **nothing** — the recycled-slot case, under real churn rather than a unit-test slot id. | 3 | E1 |
| **C3** | Compacting, distilling or tombstoning one conversation leaves every other conversation's memory untouched. | 3 | E2 |
| **C4** | Two workspaces running the same prompt do not share conversation memory. *Whether they should share prompt-branch memory is open — see E-family.* | 3 | — |
| **C5** | A conversation freed while another is mid-turn does not disturb it. | 3 | E1 |

---

## Family D — Durability and refusal

*What was sealed comes back. What was not, does not. Everything else is refused
out loud.*

| # | Invariant | Tier | Needs |
|---|-----------|------|-------|
| **D1** | Memory after a reload equals memory at the seal, exactly, for every sealed turn. | 3 | — |
| **D2** | A turn that never sealed does not appear after a reload — in the K/V or in memory. | 3 | E3 |
| **D3** | **Torn seal**: memory recorded for a turn whose records never landed is refused, not installed. Installing it puts memory a turn ahead of the K/V, which is the mirror of the defect this area exists to remove. | 3 | E3 |
| **D4** | Memory written under a different model geometry is refused. | 3 | — |
| **D5** | A **corrupt** memory record fails that one resume and does not take down the reload. | 3 | E3 |
| **D6** | A distilled conversation is unresumable, and says so with a reason distinct from "no memory yet". | 3 | E2 |
| **D7** | Shedding the turn a memory record was taken at makes that record unusable, and it is refused rather than silently applied. | 3 | — |
| **D8** | **Every** refusal above is distinguishable in the log. A conversation that resumed empty and one that resumed correctly are indistinguishable from the outside; the log is the only place the difference exists. | 3 | — |

> **Family D notes, as built.** D2/D3 run against a real torn write: the E3
> fault hook (`persistence::writer::fault::drop_next_tokens`, test-helpers
> only) drops exactly one turn's `Tokens` record while its snapshot lands —
> the one tear the §4.1 ordering permits — and the restart suite asserts the
> turn is absent (D2) and the too-new snapshot refused (D3). D5 corrupts a
> record with undecodable bytes (no fault hook needed — the snapshot enqueue
> is a public path) and asserts the reload survives, the refusal carries its
> own distinguishable WARN ("failed to decode", distinct from D4's "hash
> mismatch"), and the turn's other record classes stay intact and
> materialisable. D5 deliberately does NOT assert recall through the
> surviving K/V: on this architecture a conversation with K/V and no
> recurrent memory **has forgotten** — the design doc's §1 sentence, measured
> coming true — and that forgetting is precisely the amnesia control the
> G-family leans on. The first D5 draft asserted the opposite and failed; the
> failure was the architecture stating its own contract. **D7 is restated**: no production operation sheds a
> single sealed turn today (per-turn tombstones exist only on the splice and
> replay paths; user-facing shedding is whole-timeline), so the executable D7
> tombstones the timeline and asserts its record is unusable rather than
> applied; the per-turn boundary arithmetic is pinned at tier 1
> (`a_snapshot_newer_than_the_recovered_history_is_rejected`). **D8** is
> carried by construction rather than by one test: each refusal path's
> distinguishable message is asserted where that path is exercised (B5's
> "turn in flight", the spec-decode refusal's "recurrent state", the
> gap-fill refusal naming the island, the restore path's per-reason WARNs
> pinned by the tier-1 refusal tests).

---

## Family E — The system prompt

*A conversation begins knowing its instructions, in every layer.*

| # | Invariant | Tier | Needs |
|---|-----------|------|-------|
| **E1** | A brand-new conversation's memory already reflects the system prompt — not just its K/V. Its first turn behaves as though it had read the prompt. | 3 | E1 |
| **E2** | Two conversations opened on **different selector branches** start with different memory, and each matches its own branch. | 3 | E1 |
| **E3** | A conversation that restores stored prompt memory behaves identically to one that computed it from the tokens. | 3 | E4 |
| **E4** | Editing the prompt does **not** reuse the old branch's memory. A prompt edit that silently reused stale memory would give a model confidently following instructions it no longer has. | 3 | — |
| **E5** | Repeated prompt edits leave storage bounded. | 3 | E2 |
| **E6** | A branch no conversation selects is never computed. | 3 | — |
| **E7** | Opening many conversations on the same branch computes it **once**. | 3 | — |

> **Family E notes, as built.** E1 asserts a brand-new conversation carries
> non-empty memory before its first turn (the installed prompt checkpoint).
> E4/E6/E7 are asserted together on the `branch_checkpoint_counts` deltas: an
> edited prompt — a schema edit landing **across a restart**, the production
> shape — computes a new branch (E4), exactly once however many conversations
> open on it (E7), and nothing else computes (E6). (An in-process second
> builder is NOT a prompt edit: independent builders alias their numeric
> section ids, which is `set_projection`'s documented shared-ids contract.)
> E3's claim — restored prompt memory ≡ computed — is carried by the A1
> instrument plus the tier-1 checkpoint round-trip tests; E5's storage bound
> (`MAX_BRANCH_CHECKPOINTS`) is pinned at tier 1 in the compaction tests.
>
> **E2 found the fifth production defect of this campaign.** The composer
> dials arrive with the FIRST turn's `TurnOptions.selection`, but the branch
> checkpoint installs at create — under the default selection, because create
> cannot know the dials — and nothing re-keyed it: every conversation whose
> first turn carried non-default dials ran on the default branch's prompt
> memory while projecting the dialed prompt. The checkpoint counters sitting
> still on a dialed first turn is what convicted it. The fix rebuilds and
> installs the selected branch's checkpoint at the first dialed submit, inside
> the one window where a swap is sound — the state is still exactly the
> installed checkpoint (`state_is_prompt_only`, true from create-install until
> the first turn advances the state, and never on forks, whose state is real).

---

## Family F — Capability honesty

*The engine refuses what it cannot do, before it does it wrongly.*

| # | Invariant | Tier | Needs |
|---|-----------|------|-------|
| **F1** | Speculative decode is refused on a model whose memory cannot be rewound — at the entry point, not after the work. | 2 | — |
| **F2** | A projection that would require filling a hole mid-sequence is refused up front, naming what asked for it. | 3 | — |
| **F3** | Every refusal names the operation and the reason, not a generic failure. | 2/3 | — |
| **F4** | No reachable path rewinds a conversation's memory. | 2 | — |

> **Family F notes, as built.** F1:
> `speculative_decode_is_refused_for_a_model_carrying_recurrent_state`
> (tier 2, at the entry point, message asserted). F2: the refusal lives in
> `reserve_glue_island` and names the island size and reason; its gate input —
> a recurrent model reports `can_gap_fill = false` — is pinned by
> `a_recurrent_model_reports_it_cannot_gap_fill`, which is the property the
> test double once got wrong. F3 is asserted per-refusal where each is
> exercised (see the D8 note). F4: the spec-decode accept path is the only
> caller of sequence truncation and is refused (F1); the `<think>` partial
> truncate was removed by P3.

---

## Family G — Quality

*The point of all of it. Every entry is a paired comparison against the amnesia
control.*

These are the only tests whose assertion is on text, and they need repetition:
a single sample is an anecdote. Report the margin and the sample size, and fail
on the margin rather than on any individual answer.

| # | Invariant | Tier | Needs |
|---|-----------|------|-------|
| **G1** | Recall of a fact stated at turn 1, asked at turn N, is **better with memory than without**, at a stated margin over a stated number of trials. | 3-cruise | — |
| **G2** | The margin does not collapse as N grows — the memory keeps contributing at depth. | 3-cruise | — |
| **G3** | Thinking turns do not degrade later recall. The discarded reasoning is proven not to reach memory (tier 2); this is the behavioural face of the same claim. | 3-cruise | — |
| **G4** | A projection that reserves no glue matches the glue-based baseline in quality — the two baked approximations (§11a.2b) measured rather than argued. | 3-cruise | control |
| **G5** | Provenance retrieval on this stack retains quality against the outgoing model. | 3-cruise | fixture |
| **G6** | A conversation that restored prompt memory follows its instructions better than the amnesia control on the **first** turn — where the prompt is all there is. | 3 | — |

> **Family G notes, as built.** The cruise (`recall_quality.rs`) runs paired
> trials against the real amnesia control in two arms — shallow (raw-tail
> depth) and deep (past the summariser's raw tail, the fact's span
> compressed) — with the control proven amnesiac before any margin is read.
> Hard-gated: every memory-carrying shallow resume recalls its fact (G1's
> absolute half, across thinking turns — G3's behavioural face), and the
> shallow amnesia arm never out-recalls the memory arm. Both margins are
> REPORTED each run; the deep margin is never inequality-gated at gate-run
> trial counts.
>
> **What the first measured runs taught:** K/V-only recall is weak and
> threshold-dependent (a single flat mention fails — D5; a doubly-stated
> salient fact can cross on attention alone), and once the summariser
> engages, its nodes keep specific facts — so a compressed span's summary
> TEXT can carry a fact for both arms. Recall on this stack is served by
> **layered carriers** — raw K/V, then summary text, then recurrent memory —
> and the amnesia control isolates memory's contribution only where the
> other layers don't already serve the fact. That layering is itself a
> result the paper should state; the cruise's per-run margins are its
> standing measurement. G4 needs the no-glue projection control and G5 the
> outgoing-model fixture; neither exists in this repo yet, and both entries
> stay open until their fixture does. G6 needs an instruction-following
> scorer (a judge or a rubric); the closest standing evidence is E1 plus the
> restored-vs-computed byte equality at the seam.

---

## Family H — Cost and scaling

*Numbers, taken rather than assumed. Each fails only on a threshold that would
change a decision.*

| # | Measurement | Tier | Needs |
|---|-------------|------|-------|
| **H1** | Seal latency as a fraction of turn wall time. | 3 | — |
| **H2** | Memory write rate per turn, and **steady-state log size at depth** — the practical face of the storage question. | 3-cruise | E2 |
| **H3** | Per-turn device copy traffic. | 3 | — |
| **H4** | Cost of a conversation's first turn, with and without stored prompt memory. | 3 | — |
| **H5** | Prefill and decode throughput against the outgoing model. | 3 | — |

> **Family H notes, as built.** `h_cost_measurements` prints the whole
> suite's accumulated numbers each run — seal exports (count, total wall,
> average bytes: H1/H3's per-turn write and copy traffic), per-turn state
> forks, and branch checkpoint computed/installed counts (H4's practical
> face: an installed checkpoint IS the first turn skipping the compute) —
> and gates only the decision-changing threshold: average seal export an
> order of magnitude over the measurement that closed P8. H2 rides the same
> counters at cruise depth. H5 needs the outgoing model loaded side by side
> — the same fixture G5 waits on.

---

## Enablement

Each item is justified by the tests it unblocks, and nothing here is a
behaviour switch — they are observability and "do the background work now"
levers, both `#[cfg(feature = "test-helpers")]`.

| # | What | Status | Unblocks |
|---|------|--------|----------|
| **E1** | **Memory observability**: a digest of a live conversation's memory without needing a seal, plus live-conversation and stored-branch counts. | **Built** — `memory`/`memory_digest`/`memory_is_empty`, `live_memory_count`, `turn_kv_digest`, `sealed_block_count`, `say_opening`, `branch_checkpoint_counts`, `recurrent_state_cost`. | A7, A9, B7, C2, C5, E1, E2 |
| **E2** | **Force the background work**: compaction, persistence drain. | **Built** — `compact_substrate` + `reload_substrate` are public engine calls; the demote's `flush: true` drains the persistence queue. | A5, C3, D6, E5, H2 |
| **E3** | **Fault injection** on the durable write path — drop a specific record. | **Built** — `persistence::writer::fault::drop_next_tokens` (test-helpers), one-shot per stream. | D2, D3, D5 |
| **E4** | **A reference path**: prefill a token stream and decode it, without the projection machinery. | **Exists as `BranchCheckpointPass`** — a genuine ordered prefill outside the projection; "restored ≡ computed" is additionally evidenced non-circularly by the A1 seam comparison. | E3 |
| **E5** | **A canonical deterministic run** — one place that fixes sampling, seed and budget. | **Built** — `build_engine` (argmax, seed 0, fixed budget). | all of A, B |
| **E6** | **A shared tier-3 harness.** | **Built** — `zend/tests/common/mod.rs`. | all of tier 3 |
| **E7** | **Force tier migration / eviction.** | **Built** — `demote_timelines_hot(&[tl], true)`. | A4 |

## Order

1. **The two decisions** — B7 (divergent join on `code_read`) and A7
   (compression semantics). Both are live paths; both block their own tests.
2. **E5 + E6**, then the tests that need nothing: A1, A3, A6, A8, B1–B6, C1, C4,
   D1, D4, D7, D8, E4, E6, E7, F1–F4, H1, H3.
3. **E1**, then A9, C2, C5, E1, E2, G6.
4. **E2 + E7**, then A4, A5, C3, D6, E5, H2 — H2 closes the storage question.
5. **E3 + E4**, then D2, D3, D5, E3.
6. **Family G** last, as a deliberate cruise run.

Steps 2 and 3 are where design gaps are most likely to surface, because they are
the first time the engine is asked to be equivalent to itself across an
interruption.
