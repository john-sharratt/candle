# The Asynchronous Mind — Hierarchy, Invented Tools, and Learned Memory

*How a mind decomposes into levels; why tools are fixed only where the mind touches
reality; and why the invention of a tool by one level, for the level below it, is the
durable form of reasoning this architecture has been missing.*

Companion to [`theory_of_the_mind.md`](theory_of_the_mind.md), which argues *which*
parallel conversations a mind needs. This document specifies how they are stacked, what
passes between them, and how what is learned survives a wiped substrate.

---

## 1. The inversion

In a conventional agent stack the tool catalogue is fixed for the whole system. The model
reasons in natural language about which of those fixed tools to call, that reasoning
evaporates when the turn ends, and the tool call is the only durable artifact. Planning is
therefore something the model must redo, from prose, on every turn — and prose
re-interpreted every turn drifts.

This architecture inverts it. **Tools are fixed only at the boundary with reality.**
Everywhere above that boundary, a level *invents* the tools the level below it will be
given.

The consequence is the central claim of this document:

> The invention of a tool **is** the reasoning. It is not a description of a plan; it is
> the plan, reified into the exact representation the level below is trained to consume,
> and enforced there as a grammar.

A plan expressed as prose must be re-read and re-interpreted, and can be ignored. A plan
expressed as the tool vocabulary of the level below cannot be ignored, because the level
below is incapable of emitting anything outside it. Reasoning stops being ephemeral and
becomes a constraint.

### 1.1 The compiler this describes already exists

Nothing in §1 requires new machinery. It reassigns the *authorship* of machinery that is
already built and already proven:

| Piece | Where | What it does today |
|---|---|---|
| `ToolSpec` list for a turn | `npcd/src/engine/tools.rs` — `specs_within` | Produces the tool set a body can actually use from where it is standing |
| Argument binding to live sets | `npcd/src/engine/tools.rs` — `Choices`, `Within` | Binds an argument to the people actually present, the rooms actually reachable |
| Grammar the turn begins inside | `candle-conversation/src/conversation.rs` — `TurnOptions::turn_grammar` | Forces the turn to open inside a supplied tree rather than merely detecting a call |
| Prefill-then-constrain | `candle-conversation/src/stencil/driver.rs` — `Driver::opening` | Emits the scaffold as token ids and hands back the first sampled decision point |

Today `specs_within` reads an authored constant. In this design it reads a list produced
by the level above. The compiler is the same; only the front end changes. The constrained
decoder those specs compile into is specified in [`stencil_tree.md`](stencil_tree.md); the
building the bottom level walks around in is [`vault_world.md`](vault_world.md).

This is also why the design is not speculative about enforcement. A stencil built this way
took projected tool-call correctness from 8/16 to 16/16, and the argument-binding
discipline is what finally killed a `move_to` loop that refusal alone could not stop.
Invented tools inherit that enforcement for free.

### 1.2 The rule that makes invention legitimate

The classic failure of hierarchical planning under an LLM is a level that "decomposes" a
goal by restating it — `solve_the_problem()` — producing a stack of synonyms and no
progress. The floor that prevents it is the discipline the bottom level already keeps:

> **An invented tool must bind every required argument to a set the level below can
> enumerate. A tool whose arguments cannot be grounded is not a tool, it is a wish.**

At the bottom this rule is already mechanical: an empty live set means *drop the
parameter* if it is optional and *drop the tool* if it is required, because a zero-arm
branch is unrepresentable and silently stops the whole grammar compiling. Generalised
upward, the same rule is what forces each level to actually decompose. A level that cannot
say what the arms are has not thought about the problem yet.

---

## 2. The constraint gradient

Levels are ordered by how close they are to irreversible action, and constraint is applied
in proportion:

```
  strategy      free-form reasoning, no schema           search is free
     ↓          invents ↓
  mission       invented tools, grounded arguments       search is cheap
     ↓          invents ↓
  task          invented tools, narrowed further         search is bounded
     ↓          restricted to ↓
  act           fixed catalogue, grammar-locked          search is forbidden
     ↓
  the world
```

The principle stated generally: **constraint proportional to consequence.** Inventing a
tool costs nothing and is reversible. `move_to` changes the world. So the search space is
left unconstrained where search is safe, and closed to a single legal grammar where it is
not.

"A high level can solve anything" is the honest version of that: the space is unbounded
where nothing irreversible happens, and reality's fixed API is honoured at the one place
that touches reality.

Only the bottom level's catalogue is authored. Everything above it is invented, per
mission, at run time.

---

## 3. The vertical protocol

Levels are stackable rather than special-cased because every adjacent pair speaks the same
two-channel protocol. This is the narrow waist, and it is what allows a fourth or fifth
level to be added without inventing plumbing.

### 3.1 Downward — four knobs

A level drives the level below it by exactly four means, and no others:

1. **Restricting the tool set** — the invented vocabulary for this task, and nothing else.
2. **Injecting a description into the system prompt** — the mission and task statements are
   embedded in the frame rather than passed as content.
3. **Supplying an idle driver** — what to do when nothing has happened.
4. **Setting a timeout** — after which the task fails without the model's participation.

Together these are the complete specification of a task. There is no fifth channel, and
adding one would break the interchangeability of levels.

### 3.2 Upward — two verbs and a payload

The level below reports through exactly two signals:

- `task_complete(description)`
- `task_failed(description)`

Structured routing, unstructured payload. The verb tells the parent how to route; the free
text is what one language model passes to another well. A parent conversation is otherwise
asleep: it wakes on a signal, reasons, invents the next tool call, and sleeps again.

### 3.3 Failure must be typed, or credit lands on the wrong level

Two verbs are enough for routing but not enough for learning. "Failed" is three distinct
events with three different culprits, and collapsing them teaches the wrong lesson:

| Failure | Meaning | Blame |
|---|---|---|
| **Wrong outcome** | Executed the task correctly; the goal did not move | The *mission* — the parent chose badly |
| **Unfit tool** | Could not express what the situation needed | The *inventor* — the tool did not fit reality |
| **Never engaged** | Timed out, looped, or no-op'd | The *engine* — driver, prompt, or grammar |

These are not hypothetical categories. They are the three defects found while building the
current act loop: a standing task containing motion imperatives that produced nineteen
moves in nineteen turns and no conversation; a `move_to` that no-op'd in a loop; and a
stencil bound to a prefilled token that therefore never armed. Each needed repair at a
different level, and an archive that recorded all three as "failed" would have driven
repair at the wrong one every time.

A timeout and a considered failure are opposite events and must not archive as the same
token.

---

## 4. Why the hierarchy is affordable

A stack of always-resident reasoning conversations is not affordable on any engine. This
one is, and the reason is a property of the KV subsystem rather than of the agent design.

**A level costs a wake, not a residency.** A mission conversation is idle almost all of the
time — it does nothing between one `task_complete` and the next. Its KV therefore lives in
the warm or cold tier and is elevated on wake (`elevate_to_hot`; cold is the redo log at
`.substrate/substrate.log` — see [`archived/kv_tier_migration.md`](archived/kv_tier_migration.md)).
The marginal cost of a level is one elevation and one prefill per event, not continuous
decode.

**A long-lived level does not degrade.** Provenance-selected attention gives O(1)
numerical error at any context depth ([`unbounded_agents.md`](unbounded_agents.md)), so a
mission conversation that has been running for a week reasons as well as one opened this
morning. A slow brain that gets worse the longer it thinks is useless; this is the property
that makes a slow brain worth having.

**Levels do not contend.** They are separate sessions in the wave batcher, stepped
coherently through layers alongside every other session. Fast reaction and slow
deliberation run in parallel by construction rather than by scheduling effort.

This is the sense in which the design is specific to this engine. On a conventional stack
it is correct and unaffordable.

---

## 5. The work surface is a git repository

The lowest level operates on files and tool invocations, because that is what modern models
are trained on and where their competence is highest. The thing being operated on is the
mind folder as a git repository: **working at a bench is a branch; finishing is a merge and
a commit.**

Choosing git buys three things beyond the training-distribution match.

### 5.1 Merge conflict is the convergence engine

The open problem in the current cast is that nothing gives sixteen Makers a reason to be in
the same room, so conversations are sparse and have to be manufactured. A repository solves
it mechanically rather than narratively. Two Makers editing overlapping spans produce a
conflict that names both parties, cannot be ignored, and has exactly one resolution
procedure — which is the trigger condition for the *Settling* cluster in
[`maker_repertoire.md`](maker_repertoire.md).

Parallel work manufactures social work as a side effect. No idle pump can do that, because
an idle pump has no reason to prefer one interlocutor over another.

### 5.2 Blame is the custody chain

`git blame` is a chain of custody with a timestamp on every link. The *Custody* cluster —
"trace a thing back through everyone who has held it, and find the point where the chain
goes quiet" — has a literal implementation, and provenance stops being a field somebody has
to remember to write.

### 5.3 The repository is the oracle

This is the most important of the three, and §7 depends on it.

### 5.4 Authors are the bots

The Makers are not simulating authorship of the world; they *are* the authoring pipeline
for it. The NPC layer and the content-generation layer are the same layer, which means the
world grows because characters live in it, and the work they do is real work with a real
artifact rather than make-work with a plausible description.

---

## 6. Where the authored repertoire sits

[`maker_repertoire.md`](maker_repertoire.md) — 480 tasks in 71 clusters — is not the bottom
of the stack and not the top. It is the **seed vocabulary of the mission level**: the
authored prior that lets an NPC with an empty archive invent from something rather than
from nothing.

Two properties of that document turn out to be load-bearing here rather than decorative:

**The currencies are a type system.** Every cluster declares `needs → leaves` — `finding →
claim`, `draft → verdict`, `accord, verdict → filed`. An invented tool's signature is what
it consumes and what it produces, so composing a mission out of tools is type-checking, and
a mission that does not type-check is one whose steps do not join. This was written as a
documentation device. In this design it is the composition interface.

**The invariants are the validity rules for invention.** In particular invariant 2 (*a task
is never a tool with a new name*) and invariant 6 (*you can tell when it is done*) are the
two tests an invented tool must pass. A tool the level below satisfies with a single act is
not a task; a tool whose completion cannot be observed cannot produce a trustworthy
`task_complete`.

---

## 7. The archive, and why it is learning

When a mission ends, its conversations are stripped and archived together with their
provenance signatures. Retention cascades: *n* successful missions per goal, *n* percent of
failed ones, at most *n* goals, and so on upward. Everything is written to disk and seeded
into the substrate on start, so **wiping the substrate is a cache flush, not amnesia.**

### 7.1 Retrieval is by attentional similarity, not by keyword

The archive is useful because of *how* it is recalled. Q vectors captured live during decode
are scanned against the resident gallery arena
(`candle-conversation/src/provenance/gallery_arena/scan.rs`), so what surfaces is the
archived mission whose signature resembles the situation the NPC is in *now* — not the most
recent, and not the one sharing the most nouns. That is case-based reasoning with a
similarity metric the model itself produced, which is the correct shape for this: no
gradient step is required for an NPC to get better, only good retrieval over honest labels.

Provenance signatures are therefore not metadata about the memory. They are the index by
which the memory becomes advice.

### 7.2 Keeping failures, and keeping fewer of them

Failures vastly outnumber successes. Retaining them at parity produces a timid NPC that has
mostly read about things going wrong; retaining none produces one that repeats them
indefinitely. The percentage is the exploration/exploitation dial, and it is the single
knob most likely to need measurement rather than argument.

### 7.3 The archive is only as good as its labels

**`task_complete` is self-reported, and it is the label on the training data.** This is the
one failure in the design that does not degrade gracefully. A model that declares success
when it did not succeed does not merely lose an entry — that mission is archived as an
*exemplar*, retrieved by similarity the next time a comparable situation arises, and taught
forward. A poisoned archive is worse than an empty one.

So an outcome must be corroborated by something that is not the model's opinion of itself.
The repository supplies exactly that, for free:

- Did the branch merge, or is it still open?
- Did the validators pass?
- Did the file actually change?

None of those are negotiable by the agent that produced them. This is the reason §5.3 is
the most important property of choosing git: **the work surface is also the oracle that
keeps the archive honest.**

The argument for taking this seriously is close at hand. During the map extension that
added the Service and Appraisal rooms, the work was believed finished; `npc-map`'s own
tests then rejected it twice — once for a part clause written as a sentence fragment, once
for a capacity invariant the change had silently invalidated. Had that been an NPC filing
`task_complete`, both defects would have entered the archive as success and been retrieved
as a model to copy.

### 7.4 Missions are generated from what was learned

On completion or failure, the NPC generates its next missions and tasks out of the
provenance of what it has already tried. This closes the loop: the archive is not a log that
is written and never read, it is the input to the next act of invention. An NPC's
competence is the accumulated shape of its own retained history, and two NPCs with the same
authored repertoire and different histories are genuinely different characters.

---

## 8. What this replaces

The current loop nudges an idle character after `IDLE_AFTER_MS` (`npcd/src/engine/runtime.rs`)
with a standing instruction. This does not drive behaviour: it produces an act, not a
pursuit, and the act it produces is uncorrelated with anything the character wants. Every
symptom traced during its development — a cast that scatters, conversation that has to be
manufactured, motion in place of substance — is a symptom of having no level above the act
loop with an opinion about what the act loop should be doing.

The idle driver survives in this design, but demoted: it becomes one of the four downward
knobs a mission sets for its current task (§3.1), rather than the only thing standing
between a character and doing nothing.

The existing goal tree is the seed of the persistent side. `AuthoredStrategy`
(`candle-conversation/src/persistence/record.rs`) already carries `strategy_id`,
`statement`, `parent_id` and `state`, and `parent_id` already models mission-to-task
decomposition. Nothing currently writes that tree; in this design the mission level is what
writes it.

---

## 9. What exists, and what this design adds

Stated plainly, so the document is not read as a description of working code.

**Exists today:** the act loop and its perception of immediate surroundings; the fixed
nine-tool catalogue with availability filtering and argument binding; grammar-constrained
turns via `turn_grammar` and the stencil driver; waiting and patience
(`npcd/src/engine/waiting.rs`); the three-tier KV cache and provenance retrieval; the
persisted goal tree, unwritten; the authored repertoire and the vault that houses it.

**This design adds:** levels above the act loop; tool invention as the downward channel;
the two-verb upward protocol with typed failure; the git working surface; the retention
cascade and its disk form; and mission generation from retrieved provenance.

**Undecided, and worth deciding before building:**

- **How many levels.** Each costs a wake, and a level that wakes too rarely accumulates no
  signal. Three — act, mission, strategy — is the likely natural depth. The retention
  cascade cannot be specified until this is fixed.
- **The retention ratios.** See §7.2.
- **Pressure toward the untried.** Preferential retrieval of what worked converges an NPC
  on a small repertoire. The 480-task vocabulary makes novelty *available*; something must
  still make it *attractive*, or sixteen Makers will run five missions forever.
- **Whether a level may invent tools for a level more than one step below it.** Permitting
  it collapses the gradient; forbidding it costs a round trip. The default should be
  forbidding it.

---

## 10. Why the shape is right

Three properties, in the order they matter.

**It plays to what the models are actually good at.** Agentic flows over files and tool
calls, on a branch, with a commit at the end, are the densest part of the training
distribution for a modern model. The design does not ask a model to be a character in a
simulation; it asks it to do work of a kind it has seen an enormous amount of, and lets
character fall out of the choices it makes while doing it.

**It works around the context window without giving up what was learned.** No conversation
has to hold everything, because every level holds only its own concern and wakes only on
events at its own grain. What crosses between them is a verb and a description. What
survives them is on disk.

**Deliberation and reaction do not trade off.** They are separate sessions with separate
KV, so the slow brain thinking hard about a mission does not slow the fast brain answering
somebody who just walked in. Most agent architectures have to choose; this one does not,
for reasons that belong to the inference engine rather than to the architecture.

The individual ingredients are not unprecedented — hierarchical task networks, blackboard
architectures, case-based reasoning. The move that is not standard is treating **tool
invention as the durable form of reasoning**: a plan that is compiled into a grammar
instead of restated in prose cannot drift, cannot be ignored, and is executable by the
level it was invented for without any interpretation step in between.
