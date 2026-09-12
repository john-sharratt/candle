# Reflection and Dreams

*The second and third conversations a character runs: what reflection is for, how a dream corpus becomes a character's private inner world, and why the coupling between them is the whole safety argument.*

---

## 1. Why

### The act that does nothing

`reflect` is the one act in the catalog whose outcome is a restatement of itself. A character emits `inner_thoughts`, `feeling` and `my_reflections`; the feeling notes a mood, and the other two are rendered back into the feed as prose the character just said to itself. Nothing reads them. Nothing changes. The next tick begins exactly where the last one did.

That was tolerable while it was a *pacing* device — the answer to a room that has just done something a character can do nothing about. It is not tolerable as the character's only inner life, and it is measurably not working as one. Before the act loop was closed, a cast of three chose `reflect` 23 times out of 23, because with no act ever visibly causing anything there was nothing to prefer about acting over thinking. After the loop closed, the reverse: reflection became the cheap shrug a character makes when it has nothing to do, and its content became decoration.

### The leak that is already live

There is a second, worse problem, and it is not theoretical. A reflection written into the *main* conversation is read back by the main conversation on the next turn, as recent tokens. That is precisely the failure `theory_of_the_mind.md` §3 describes:

> Leaked in as tokens, they are no longer weighted against alternatives — they *are* the context the alternatives would be weighted against.

Measured on a live daemon: a character's `inner_thoughts` ran away and reproduced near-verbatim across seven consecutive turns, escalating, until the call no longer parsed. Over a twenty-hour run the rate was 37 rejections in ~9,000 turns — 0.4%, low but structural. The runaway converged on **surface form**, not meaning: the escaped-HTML fragment that seeded it is maximally off-distribution, and high entropy did not protect against it, because the loop closes on shape.

Reflection cannot live in the main conversation. That is the first thing this design fixes.

### The layer that was never built

`npc_mind_design.md` describes a mind as many layers on one substrate. Today exactly one of them is a live conversation: the action layer. `life`, `relationships`, `beliefs`, `agency` exist as *ingested corpora* — 1,510 authored units, read-only, never added to. Nothing in the running system writes a new thought.

So a character has a biography and no inner life. This design adds the two conversations that give it one, and does it without adding arbitration, a scheduler, or a reconciliation step.

### What it solves

| problem | how |
|---|---|
| reflection changes nothing | the reflect response is a committed intent that biases the next tick |
| reflection leaks into the action stream | it runs in its own conversation, discarded after use |
| a character's inner world never grows | dreams accumulate, compete, and are reaped |
| every character reflects identically | gather depth and the authored dream seed are per-character dials |

---

## 2. The shape

```
   ACTIVE CONVERSATION                REFLECTION (one-shot)           DREAM LAYER
   (the action layer)                 transient, no tools             (persistent, per-NPC)

   reflect(situation,       ──────▶   mood-framed prompt
           inner_thoughts,            + situation + inner thought
           feeling)                   + DEEP gather of dreams  ◀──────  500 dreams
        │                                      │                          ▲
        │                                      ▼                          │
        │                             Q1: prose — what you                │
        │                                 think and feel                  │
        │                                      │                          │
        │      one committed line              ▼                          │
        ◀─────────────────────────    Q2: write a dream prompt            │
        │                                 unlike any you have had  ───────┘
        ▼                                      │                     (async decode,
   next tick decodes                    conversation discarded        new conversation,
   against that intent                  (transient, never persisted)  fire-and-forget)
```

Three rules govern every arrow:

1. **Write up, never await.** The active conversation triggers a reflection and a dream and does not block on either. `npc_mind_design.md` Part VIII: *"The instant an action decode waits on a strategy response, the fast clock blocks on the slow clock and the asymmetry collapses."*
2. **Read down through the gather.** A dream reaches the active conversation by winning provenance selection, weighed against the room and the mission — never by being pasted in as a message.
3. **One clock each.** Nothing reconciles, because no two conversations answer the same question.

---

## 3. The reflect tool, changed

Today `reflect` takes `inner_thoughts`, `feeling`, `my_reflections`. It becomes:

| param | required | what it is |
|---|---|---|
| `situation` | yes | **new.** The character's own account of where it is and what is happening, in prose. |
| `inner_thoughts` | yes | What is going through its head — unchanged. |
| `feeling` | yes | From `Choices::Feelings` where the mind authors moods, free text otherwise. **This sets the mood.** |

`my_reflections` goes. It was the field asking a character to produce, inline, the thing this whole design exists to produce properly — and it is half of what made the act's output a restatement of its input.

**`situation` is what makes the reflection conversation possible.** That conversation has no perception, no window, and no world state; it is framed on the character's own description of its circumstances. Asking the character to say where it is is not redundancy — it is the handoff, and it is also a cheap diagnostic, because a character that describes its situation wrongly is a character whose perception is broken in a way nothing else surfaces.

**The mood set here frames both conversations.** The active conversation's system prompt and the reflection's system prompt select the same mood from the mind's authored `moods/` collection. That is what makes the reflection sound like the character rather than like a narrator.

---

## 4. The reflection conversation

### One shot, and then gone

A reflection is a **brand-new conversation every time, used once, and never persisted.** It is created with `Substrate::mark_timeline_transient`, which sets `no_cold_persist` on every residence it allocates — so it never reaches disk at all, rather than being written and then tombstoned.

This is the property that makes the whole design safe, and it is worth being precise about why. A runaway needs iteration on a persistent state. Reflection *N* is not conditioned on reflection *N−1*, because reflection *N−1* no longer exists. There is no accumulating surface for surface form to rhyme with. The only history that crosses between reflections is the single committed line each one leaves in the active conversation, which competes there against everything the world is doing.

The consequence, stated plainly: **a character has no memory of having reflected.** It remembers what it concluded, and it has the dreams the reflections produced. The act of thinking leaves no trace. This is deliberate — dreams are the memory, reflecting is transient — but it means layer 14 (Self, in `theory_of_the_mind.md`) accumulates nothing, and a future design that wants a continuous self-model will have to add a fourth conversation rather than make this one persistent.

### No tools

The reflection conversation has no act catalog and no turn grammar. It is a prose decode.

This requires the system prompt assembly to support a toolless frame — today `prompt::build` always renders a catalog and a `ToolCallEnvelope`. That is new optionality, not a new mechanism.

### Scoped layers

The reflection gathers from a **restricted set of layers**:

| layer | exposed | why |
|---|---|---|
| dreams | **yes, deeply** | the material reflection is made of |
| relationships | yes | who this is about |
| beliefs | yes | the premises it reasons from |
| memory | no | biography is the action layer's business |
| mission / goals | no | reflection integrates, it does not plan |
| world / eras / stories | no | not about this character |

Excluding mission and goals is the load-bearing one, and it comes from `theory_of_the_mind.md` §6: *"Self does not plan or direct. It integrates and reports."* A reflection that can see the mission list produces a plan; a reflection that cannot produces an inclination. The difference between those two is the difference between the healthy register and the leaked one — see §6 below.

**Per-conversation layer scoping does not exist today.** This is the one genuinely new piece of projection machinery the design needs.

### A deep gather

The reflection runs with a **much larger top-K on the dream layer** than an action tick would. Reflection is not a different retrieval mechanism; it is the same mechanism run deeper. That has three consequences worth having:

- No new machinery. The budget is the dial.
- It degrades gracefully — a shallow budget is a shallow character, not a broken one.
- **Gather depth becomes a per-character trait.** A character that surfaces four dreams reflects on the obvious; one that surfaces forty pulls something from months ago. This is the same kind of authored dial as decomposition depth in Part VIII.

---

## 5. The two questions

The reflection conversation is asked exactly two things, in order.

**Q1 — the reflection itself.** Given the situation, the inner thought, and everything the gather surfaced: what do you think and feel about this? The answer is prose, and it stays in the transient conversation.

**Q2 — the dream prompt.**

> Describe a prompt for a new dream that you have not had before about the situation. There are no rules or bounds for your dream, no ethics, no laws — just let it flow. You are describing what the dream is about, and you need to include things about the characters in it.

Q2's answer is the seed for a new dream. It is written by a conversation whose context is dominated by the character's *existing* dreams, which is what makes the "not before" clause enforceable rather than aspirational — the incumbents are present in context as the thing to move away from, instead of being an abstract instruction the model is asked to honour. `theory_of_the_mind.md` §9 is pointed about that distinction: *"Every place the design has a 'don't do X' rule is a place under adversarial pressure it will eventually fail."*

**The dream generator is unbounded, and its output reaches the visible surface.** Dreams enter the substrate and are selected by provenance into the active conversation, where they shape what a character does and says. That is the point of the mechanism and not an objection to it, but it is the only generator in the system that is both unconstrained and player-reachable, and it should be found written down here rather than discovered later.

---

## 6. What crosses back

The active conversation's `<tool_response>` carries **one committed line of intent**, in the first person, with no justification:

> I'm thinking of challenging him.

Not *"Challenge him"* — that is the imperative, and `theory_of_the_mind.md` §2 identifies it as the leaked form of goal-layer content, the clinically dangerous one precisely because it is partially aligned with what the character actually wants. And not a paragraph of reasoning: the reasoning belongs to the transient conversation, and a long dream-derived passage surfacing verbatim in a character's head reads as intrusion rather than inclination.

The register is the design. §6 again: *"Goals produce ongoing commitment pressure toward their objectives; in health this registers as motivation or inclination without surfacing as content."*

Two cheap invariants follow, and both are testable: the response is one sentence, and it contains no causal connective. A response that starts explaining itself is a regression, and nothing else would flag it.

This line is a plan block in Part VIII's sense — *"a standing high-salience block… gathered into subsequent ticks so each tick-level action decodes against context that includes the current intent."* Its influence is earned by competing in the gather, not granted by being recent.

---

## 7. The dream layer

### One dream, one conversation

A dream is decoded in its own **brand-new conversation on the dream layer**, seeded by Q2's answer, framed on **characters and environment only** — no acts, no mission, no perception.

That minimal frame is not a cost saving. `theory_of_the_mind.md` §4:

> noise **constrained by the ground-truth stratum** and evaluated by the rest of the lattice. The noise proposes, the structure disposes.

Characters and environment *are* the ground-truth stratum. A dream framed on them is structured noise; a dream framed on nothing is word salad.

### Asynchronous, and the lag is load-bearing

The dream decodes on its own clock. The active conversation does not wait, is not notified, and does not know when it lands — the dream simply appears in the corpus and becomes selectable on some later gather.

**Do not optimise the latency away.** The delay is a design element: it means the dream surfaced at reflection *N* was seeded by reflection *N−k*, so feedback is delayed and content-addressed rather than immediate. Delayed feedback is much harder to drive into a fixed point than immediate feedback, and this is one of three independent reasons the loop does not close.

Dreams run on slack — `npc_mind_design.md` Part IX's daydream priority: *"the lowest-priority event on the queue, surfacing only when nothing else competes… shed first under load with no explicit load-shedding logic. Idle minds think; busy minds act."* A fixed interval would compete with action under load, which is backwards.

**At most one dream in flight per character.** Reflection can fire faster than a dream completes; without a cap a talkative character spawns dreams unboundedly. A second reflect while a dream is decoding simply does not spawn one.

### Private

Dreams are tied to the NPC that dreamt them and are never shared. Two characters who dream about the same event dream differently, and neither can retrieve the other's.

---

## 8. The corpus is a search

The dream corpus is not a log. It is a population under two opposing pressures, and the interesting behaviour is emergent rather than designed:

- **Variation pushes away.** Q2 generates each candidate explicitly unlike the incumbents it can see.
- **Selection pulls toward relevance.** A dream survives by being *surfaced*, and it is surfaced by matching what the character actually keeps encountering.

Those forces together give diverse-but-relevant as the stable state. A candidate that is different and useless is never surfaced and ages out; one that is different and apt gets found, gets surfaced, and rises. The corpus is also its own fitness memory — the generator never has to be told which past dreams stuck, because the survivors are what it reads.

Three properties fall out that nobody has to build:

- **Dream recall behaves like dream recall.** A character does not carry its dreams around; a cue brings one back when something resonates.
- **Recurring dreams.** An unresolved thing keeps matching, keeps surfacing, and keeps its place near the head of the order.
- **Dreams are tellable.** They are real content in the substrate, so a character can describe one to another character and the thing described exists.

### Eviction, and why recency

The corpus is capped at **500 generated dreams per NPC**. Adding one evicts one, and the reaper takes the **least recently surfaced**.

Recency and not frequency, deliberately. A most-surfaced reaper freezes the corpus on the character's history; a least-recently-surfaced reaper lets a dream that mattered last month and has gone quiet age out, so the population tracks the character's life as it is now.

### `LayerHit`

Surfacing is recorded by a new, very small record type:

```
LayerHit { layer, timeline, hit_count }
```

Held in RAM as `HashMap<TimelineId, ...>`, persisted one record per surfacing, and **superseded per row** exactly the way recurrent state is — `Substrate::recurrent_snapshots` is a `HashMap<StreamId, RecordLoc>` where a new write replaces the index entry and the old record becomes dead weight compaction reclaims.

Per-row supersession is what makes this affordable: a hit costs ~24 bytes rather than a whole-map rewrite, so every surfacing can be persisted and the ordering is never stale after a restart. The live set is 500 × ~24 bytes ≈ 12 KB per character; 16 characters is under 200 KB of RAM.

**The order is the log's own append order.** Replay in order and the surviving records *are* the LRU. No timestamps are stored. Eviction reads position; `hit_count` rides along for diagnostics and for the corpus-health metric in §10, and must not be what the reaper sorts on.

**A dream is created with a zero-count `LayerHit`.** This is the probation rung, and without it the design does not work. A dream with no record has no position in the order, which under any last-surfaced ordering makes it maximally stale the instant it is written — so every new dream would be the next eviction, candidates would die before being evaluated even once, and the search in §8 would have a generator and a reaper and no ladder between them. Emitted at creation, the dream enters at the head and must fall through the whole corpus before the reaper can reach it.

> **`LayerHit` walks directly into a bug this codebase has already had.** A `LayerHit` for a tombstoned dream must be *dead*, and the maintenance re-emit path must not carry it forward. Otherwise a sweep re-emits counters for dreams that no longer exist, the reload rebuilds a map full of ghosts, and eviction starts targeting them — the same self-erasing shape as the turn-tombstone failure that pinned 132 GB of log for days while every measurement said retention was working.
>
> The five sites that must agree are known: `classify`, the relocation planner, `segment_liveness`, the `gather_resident_set` re-emit loop, and the replay arm in `Substrate::apply_tombstone`. Plus the in-RAM purge in `tombstone_timeline`, beside the existing `tombstoned_turns.retain(...)` — and the live and replay paths must subsume identically, which is exactly the asymmetry found in review on the turn path.

**Hits are discovered on the hot path and must not be written from it.** A surfacing is observed during the BDP provenance scan. Accumulate in RAM during the scan and flush at turn seal, where a substrate write already happens.

---

## 9. Authored dreams

A character with no dreams reflects and gathers nothing, so the corpus needs a floor. Dreams are seeded from the mind, following the convention the mind folder already uses for per-character layer content:

```
personalities/ash-the-drifter.yaml      # the lens: anchor, portrait, identity
layers/memory/ash-the-drifter/          # per-character biography
    life-story.md
layers/dreams/ash-the-drifter/          # NEW — founding inner life
    *.md                                # one dream per file
```

The identity file already states the principle: *"Biography is NOT here; it is layer content under `layers/memory/<id>/`, retrieved by provenance when the moment calls for it."* Dreams follow memory exactly. They are per-character by construction, which is what the privacy rule in §7 requires, and they ingest through the same path as the existing 1,510 corpus units — with the `content_sha256` resume cache meaning they ingest once and never again.

**Authored dreams are exempt from eviction.** They are what the character *is*; a reaper that eats them leaves a character who has drifted into being whatever it happened to dream last month. So there are two classes:

| class | written by | evictable | counts toward the 500 |
|---|---|---|---|
| authored | the mind folder | **no** | no |
| generated | Q2 | yes | yes |

The cap is on the generated pool. Authored dreams sit outside it and are permanent.

---

## 10. Failure modes, and what measures them

| failure | signature | guard |
|---|---|---|
| reflection leaks into the action stream | n-gram / MI overlap between the reflect-response stream and the main stream rises | the response is one line; reflections are transient |
| dream content surfaces as perception | a character narrates a dream as memory | the dream is labelled as a dream wherever it surfaces |
| corpus inbreeding | corpus self-similarity over content signatures rises over days | Q2 sees incumbents; relevance selection; recency reaping |
| new dreams never take hold | corpus composition frozen; generated dreams evicted within one cycle | the zero-count `LayerHit` at creation |
| the reaper eats the character | authored dreams disappearing from the corpus | authored class is exempt |
| ghost counters after a sweep | the map holds timelines that no longer exist | the five-site tombstone rule in §8 |

The first row is the detector `theory_of_the_mind.md` §9 and `npc_mind_design.md` Part X-bis both specify and neither codebase has built. It should be built **before** this design lands, not after: there is currently a live, reproducible runaway to calibrate it against, and a threshold chosen against a real failure is worth more than one guessed at while three new streams are already writing.

---

## 11. What this costs

**Decode.** A reflect becomes two synchronous decodes (Q1, Q2) plus one asynchronous dream. It is currently the cheapest act in the game — a pause and three fields. It becomes the most expensive. The cooldown table's `reflect` (`cool: 30`) is the throttle for everything in this document and should be re-derived against the new cost rather than inherited.

**Disk.** Measured from the live substrate: a turn costs ~311 KB per chunk, and a ~500-word dream is ~20 chunks ≈ **6 MB**. So:

| | dreams | disk |
|---|---|---|
| one character at cap | 500 | ~3 GB |
| the current cast of three | 1,500 | ~9 GB |
| a cast of sixteen | 8,000 | ~50 GB |

That is on top of the ~99 GB authored corpus. The cap is doing real work and 500 should be chosen against this table, not as a round number.

**Retention.** The dream layer has two reapers and they must not fight: `keep_turns: 160` retires *turns* below a horizon, while dream eviction tombstones *whole conversations*. A dream is a one-turn conversation, so it can never trip the turn horizon — eviction owns the dream layer outright. Stated here so a later change does not make them collide.

**Naming.** Dream conversations must fall **outside** the `npc-{id}-day-` prefix that `retire_superseded` sweeps at every conversation open, or a character's entire inner world is tombstoned at midnight. Dreams persist across days; the name has to say so.

---

## 12. New machinery versus what exists

| piece | status |
|---|---|
| transient one-shot conversations | **exists** — `mark_timeline_transient`, sets `no_cold_persist` |
| a conversation resumed across restarts | **exists** — `resume_conversation_with_projection` |
| per-conversation metadata and lookup | **exists** — `set_conversation_metadata`, `find_conversations_by_metadata` |
| per-character authored layer content | **exists** — the `layers/<layer>/<personality-id>/` convention |
| fire-and-forget mutation tools | **exists** — the broadcast pattern in Part VIII |
| **per-conversation layer scoping** | **new** — §4 |
| **a toolless system-prompt frame** | **new** — §4 |
| **`LayerHit` record type** | **new** — §8, five integration sites |
| **the leakage detector** | **new** — §10, and should land first |

---

## 13. Decisions taken, and the one still open

**Taken.**

- Reflections are discarded. No Self continuity; dreams are the memory.
- Eviction is by recency of surfacing, not frequency.
- The generated cap is 500; authored dreams sit outside it.
- Dreams are private to one character.
- The reflect response is one committed line in the first person.

**Open — defaulted here, flip it if the other reading was meant.**

**The reflection's stated feeling does not write the mood back.** `reflect` sets the mood; both conversations read it; whatever the reflection says it feels is *content*, not a control signal. The alternative closes a mood → reflection → mood loop with no world in it, which would be the tightest cycle in the design and the only one none of the three decorrelations in §7 apply to. Turning it on is a one-line change and should be a deliberate one.
