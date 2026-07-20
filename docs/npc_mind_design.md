# The Asynchronous Mind — NPC Cognitive Architecture

## A complete design for persistent, lived NPCs on a single-card inference engine

This document defines the cognitive architecture for persistent NPCs in Battle Cities. It assumes the inference engine of the companion paper, *One Card, One Stack* — provenance-indexed attention, Speculative Context Decode, two-phase KV quantization, an unbounded three-tier paged context, and the Asymptotic Numerical Stability guarantee that lets a single NPC's memory grow without bound while per-step numerical error stays O(1). Every mechanism here is built on those primitives. Nothing here requires a second model, a rules engine, or a behaviour-tree bolted alongside the transformer. There is one mechanism, applied everywhere.

---

## The one idea

Every behaviour in this architecture is the same operation seen from a different angle: **gather the relevant substrate under a salience budget, attend over it through a fixed lens, act, and write the result back.** There is no separate filtering system, no routing table, no conditional-trigger machinery, no mood state-machine. There is one asynchronous event loop per NPC, a layered substrate it reads, and provenance attention deciding what is salient at each step.

The design discipline is a single sentence: **the model nudges its mutable substrate but never controls it, and it cannot touch its immutable core at all.** No mutable layer is ever written by fiat — the model emits mutation events that land on the substrate as one more entry, and what a layer *becomes* is the gather over all such entries under salience. The immutable core is never written by anyone; it is read-only by construction. From these two facts — influence over the mutable, no access to the fixed — every visible behaviour follows.

Two consequences are used throughout and stated once here:

**Filtering is selection, not a gate.** The question "should this belief affect behaviour?" is never evaluated by a predicate. The belief block either wins the gather against the current cognitive state, or it does not. A filter that drops a signal before it lands is control; letting it land and lose the salience competition is influence. We do the second, everywhere — and the same is true of selecting *which parts of the fixed core to surface*: that is filtering the immutable substrate for cognitive load and consistency, never mutating it.

**Convergence is attention, not reconciliation.** When several things land on the substrate, they are not merged by a reconciliation step. They co-occur in the working set, and the softmax over the gathered set *is* the integration. Gather-and-attend is the merge.

Everything below elaborates these two sentences across an immutable core, a mutable lived substrate, and one asynchronous loop.

---

## The design at a glance

Before the mechanism, the shape. An NPC is four things stacked around a single act of cognition, and almost everything in this document is a detail of one of them.

At the centre is **the gather**: each time the NPC thinks, provenance selection pulls the currently-relevant blocks from its layers into a bounded working set, the model attends over them, and the result fans out as acts and as new entries written back to the layers. This is the only place cognition happens. It runs as one **asynchronous loop** — events arrive in an inbox, the loop drains them, gathers, decodes one step, fans out, and waits for more. Conversation pace and world pace are decoupled because the loop ticks at the world's tempo while talking happens only when the act layer chooses to.

Around that centre sit the other three things. **Below** is the immutable core: world knowledge and the archetype, a read-only shared prefix that is never changed and never gathered, the fixed lens at the base of the attention stack — it is the system prompt, the lowest tokens everything else is read against. **Above it** are the mutable conversation-layers, the substrate of the mind — perception, action, agency, relationships, beliefs, memory — each an append-only stream that drifts under selection. **Above those** is the Cortex: the heart of the mind, the thing that gathers across the layers, attends, decides, and acts. **On top** is the interaction layer, the surface between the mind and an operator. It does not fabricate the NPC's output — the words are always narrated from real acts — but it is not inert either: downward it *injects events* (an operator speaking becomes an event in the inbox), upward it *observes acts and narrates them*. Inject down, narrate up, never fabricate.

The naming is deliberate. The Cortex sits over the deeper structures, reads across all of them at once, and integrates them into a single response — which is exactly what a cortex does over subcortical material. The layers below are the standing dispositions, drives, and memory; the Cortex is the sheet that gathers across them and acts. Two kinds of thing touch the Cortex without being layers in it: **modulation parameters** (affect, threat, curiosity) bias its selection without contributing content, and a **monitor** (metacognition) watches its health from outside without participating.

The attention stack runs bottom to top — immutable core at the base, substrate above it, Cortex above that, interaction at the surface — which is the order the KV cache is actually assembled in: everything is read *upward* through the fixed lens at the bottom.

```
                   ┌─────────────────────────────────────────────────┐
   operator ◀────▶│  INTERACTION — the surface                      │
   / player        │  injects events downward (player speaks → inbox)│
                   │  narrates acts upward · never fabricates output │
                   └───────────────────────┬─────────────────────────┘
                          inject ▼  ▲ narrate
  ────────────────────────────────────────────────────────────────────────
                  ┌────────────────────────────────────────────────┐
                  │  CORTEX — the heart of the mind                │
   ┌───────────┐  │  gathers across the layers under budget B,     │    ┌───────────┐
   │MODULATION │  │  attends, decides, acts · one async loop       │    │ MONITOR   │
   │affect ·   │─▶│  ticking at world tempo · acts fan out to the  │◀▶│ metacog · │
   │threat ·   │  │  world (speak · move · world-mutation)         │    │ leak /    │
   │curiosity  │  │                                                │    │ runaway   │
   └───────────┘  └───────────────────────┬────────────────────────┘    └───────────┘
      bias ─────────────┘      gather ▲  ▼ write back
                  ┌────────────────────────────────────────────────┐
                  │  SUBSTRATE — the mutable conversation-layers   │
                  │  perception · action · agency ·                │
                  │  relationships · beliefs · memory              │
                  │  append-only streams that drift under selection│
                  └───────────────────────┬────────────────────────┘
                       read through ▲  ▼ the lens
                  ┌────────────────────────────────────────────────┐
                  │  IMMUTABLE CORE — the lens (system prompt)     │
                  │  world + archetype · read-only · CoW           │
                  │  never gathered · the base of the attention    │
                  │  stack — everything is read upward through it  │
                  └────────────────────────────────────────────────┘
```

> This is the orientation; the detailed topology — how the layers sit *within* the Cortex's gather, how the system-prompt view is assembled from the core, how modulation feeds in — is the figure in Part III. Read this one for the shape, that one for the wiring.

---

## How the layers were chosen

The layer decomposition was not picked by intuition. It was derived from two directions at once and kept only where they agreed.

The first direction is engineering need: what must run in parallel for a mind to produce a good verdict? Most of a mind's work never surfaces. You do not experience your visual system deciding where an edge is, or the process that decides whether a stranger is trustworthy — you get the verdict, not the deliberation. A complete substrate is the set of conversations that must run sub-perceptually for the surfaced verdict to be right, and you can reason about that set from function: there must be something tracking the world (perception), something pursuing aims (agency), something calibrating per-entity (relationships), something holding premises (beliefs), something accumulating history (memory).

The second direction is a check against a known catalogue of what happens when pieces of a real mind come apart. Clinical dissociation is informative here precisely because it is *selective*: the conditions in which one function fails while others hold are evidence that the functions are genuinely separate. The relevant move is narrow and worth stating carefully, because it is easy to overclaim. The fact that distinct functions fail distinctly is good evidence that **the functions are separable** — it is *not* evidence that any particular mind is wired in layers, nor that the mechanism resembles this architecture's gather. We use the clinical picture only for the defensible claim: these are separable functions, and a system that conflates their outputs degrades in describable ways. Whether a brain implements them as layers, and whether the resemblance to attention-weighted gather is real or merely a satisfying redescription, is not something this document claims to know. The engineering stands on the separability alone; the convergence is corroboration, not foundation.

What the two directions agree on, after the sorting in Part III that strips out everything which is really a modulation parameter or a boundary or a read-pattern rather than a conversation, is a small set: perception, action, agency, relationships, beliefs, memory — read through an immutable lens, biased by a few modulation parameters, watched by one monitor. That is the whole architecture, and the rest of this document is its mechanism.

---

## Part I — Identity: the immutable core

### Fixed by construction, not by enforcement

The deepest structural decision is the split between an **immutable core** and a **mutable lived substrate** — and the recognition that this split is the *same fact* as the copy-on-write sharing model, seen from the identity side rather than the memory side.

The world knowledge and the archetype — core identity (values), voice (expression rules), and behavioral model (processing rules) — are a copy-on-write KV prefix shared by the world (all NPCs) and the archetype (~300 NPCs per type). Because that prefix is physically shared and read-only, it **cannot drift** — not because a rule forbids mutation, but because mutation would break the sharing. The identity guarantee the long-horizon vision needs, that an NPC is still recognisably *itself* after a year of play, is therefore free from the VRAM economy already adopted for cost. One mechanism, two payoffs: the shared archetype is cheap, and it is a hard identity floor that no accumulation of experience can erode.

This resolves the tension between "everything is mutable substrate that selection governs" and "the core must hold." Both are right, about different layers, and the boundary between them is exactly the CoW boundary: **layers that are shared read-only are the layers that cannot be nudged and constitute identity; layers that are per-NPC and writable are the layers selection governs and constitute lived depth.**

### The core is a lens, not a store of conclusions

The immutable core is not inert, and it stores no conclusions. It stores the value hierarchy and the behavioral processing model — the function that *weights what the mutable layers surface*. Experience is the input; the archetype is the processing.

A betrayal lands in the relationship layer, which is mutable and drifts and accumulates. Whether it reads as "the unforgivable act" or "a cost of doing business" is the archetype doing the weighting, and that weighting never changes. A Loyal Soldier can lose faith in *this specific chain of command* without ceasing to be someone for whom betrayal is axiomatic: the belief about the commander drifted; the function that evaluates betrayal did not. The mutable layers change what the NPC knows and feels; the fixed core changes what those facts and feelings *mean to this person*. Depth comes from the layers; identity comes from the lens; and they cannot contaminate each other, because one is writable substrate and the other is read-only prefix.

### The sorting razor

This gives a precise test for what belongs where, and it is the razor that separates genuine fixed processing from learned conclusions:

> Anything that is a **value** or a **way of processing** belongs in the immutable core. Anything that is a **fact, feeling, relationship, or conclusion** belongs in the mutable gather-layers.

| Content | Home | Why |
|---|---|---|
| "Betrayal is unforgivable" | Immutable core | A value |
| "I weight direct observation over second-hand intel" | Immutable core | A processing rule |
| "Alice betrayed me" | Memory (mutable) | A fact |
| "I distrust Alice now" | Relationship (mutable) | A conclusion — emergent from reading the fact through the value |
| "Forest recon needs 2× observation time" | Strategic learning (mutable) | A learned conclusion |

### The lens is always-on, and never competes for budget

A fixed core that failed to be gathered on a given tick would weight nothing — immutability is not presence. The hazard is that a strong signal (a rage) could win the whole working-set budget and crowd the archetype out, leaving the NPC acting on raw feeling with no lens.

The resolution requires no authored salience floor: because the archetype is the CoW prefix, **it does not compete for the gather budget at all.** It is structurally always resident — the fixed lens at the base of every assembly, not something selection chooses to include. Identity is not *in* the working set; it is the frame the working set is read *inside*. So what stops a rage from crowding out the soldier is that the rage competes for the *memory* budget, while the soldier was never in that budget — he is the prefix the rage is read through. An NPC in a rage is a Loyal Soldier raging, not a generic rage.

---

## Part II — The system prompt is the immutable core, viewed

The system prompt is not a layer and it is not a scratchpad. **It is the immutable core, surfaced.** Everything in the system prompt is a section of the read-only substrate; what varies turn to turn is *which sections are surfaced into the frame and how*, never the sections themselves. This is the single most important clarification over earlier designs: the per-turn variation in the system prompt is **selection over a fixed substrate, not mutation of it.** Selecting which immutable sections to show — to reduce cognitive load and produce consistent responses — is filtering, and filtering is not writing.

The whole immutable lens is always resident. The system prompt is a **token-budgeted, attention-selected view onto it.** Sections compete for that budget the same way memory blocks compete for the working-set budget, and provenance decides which sections of the fixed substrate are most relevant to surface for the current exchange.

### Three selection disciplines over one immutable substrate

Within the system prompt, different sections are surfaced under different disciplines — but all three are *selection over the fixed core*, never mutation:

**Structurally always-present (the lens).** Core identity, voice, behavioral model, world grounding. These never drop. They are the frame, surfaced every turn regardless of query.

**Spiking (mood).** Mood lives in the system prompt as a section selected from the immutable register set — confident, tense, grieving, analytical, and so on. It is **event-driven and threshold-gated, not drifting.** It holds its current register until provenance scores a *different* register above a spike threshold at an SCD barrier, and then it snaps. No gradual blend, no continuous interpolation — a mood is stable until attention spikes hard enough to cross the threshold, then it changes for the next decode window. This produces seamless but discrete emotional transitions: a briefing that shifts from confident to tense at a reasoning-step boundary still reads as the same briefing, the structure intact, the emotion swapped. The mood is not a mutable state the model edits; it is a section of the fixed register library, selected by a spike.

**Locked top-k (response template).** The response template — military briefing, casual conversation, battlefield urgency, merchant negotiation, whispered conspiracy, storytelling — is selected once at turn start by top-k provenance match against the incoming query, and exactly one is allowed. It is then **frozen for the entire decode.** This is deliberately *not* spiky: structural mode cannot change mid-response without breaking coherence, so unlike mood it does not re-evaluate at barriers. One template, selected once, fixed for the turn. Mood colours *what emotion* sits inside the structural mode; the template sets the structural mode itself and holds it.

The contrast is the point: **mood is spiking selection (re-evaluated at every barrier, snaps on threshold); template is locked selection (chosen once, immovable for the decode).** Both are views onto the immutable substrate. Neither mutates anything.

### Situation and concern sections

The remaining system-prompt sections — a compressed identity anchor, the current situation (mission/strategy/task/perception state), and the top active concerns — are likewise surfaced views, selected by provenance relevance to the incoming exchange. A concern that has already been voiced and resolved scores lower and stops being surfaced; this is the same fade-by-non-selection that governs the mutable substrate, applied to which sections of the frame are worth the budget. When the NPC has a genuine low-confidence gap and its conversational partner can resolve it, a curiosity-shaped section is surfaced that biases generation toward asking rather than asserting — again, not a triggered slot, but a gap-section winning the budget because it is salient to the exchange.

---

## Part III — The mutable lived substrate

The lived substrate is per-NPC, writable only through mutation events, gather-selected, and free to drift. It is where depth accumulates while the core holds.

```
  ═══ top of the attention stack ════════════════════════════════════════
                 ┌──────────────────────────────────────────────────────┐
INTERACTION      │  injects events down · narrates acts up ·            │
the surface      │  never fabricates output · rate-decoupled            │
                 └───────────────────────────┬──────────────────────────┘
                                     narrate ▲  │  ▼ inject
                 ┌──────────────────────────────────────────────────────┐
CORTEX           │  ░░ gathers·attends·decides·acts · budget B ░░       │
the heart of     │                                                      │
the mind —       │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
gathers          │  │PERCEPT  │ │ACTION   │ │AGENCY   │ │RELATN   │     │
across the       │  │sub-fed  │ │acts+spk │ │mission  │ │per-enty │     │
layers,          │  │fast clk │ │tool     │ │strategy │ │trust    │     │
composes         │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
by               │  ┌───────────────────────┐ ┌───────────────────────┐ │
selection        │  │BELIEFS                │ │MEMORY                 │ │
                 │  │read · NEVER written   │ │unbounded · O(1) error │ │
                 │  │by action; evidence    │ │conv · learning ·      │ │
                 │  │threshold only         │ │dreams                 │ │
                 │  └───────────────────────┘ └───────────────────────┘ │
                 └───────────────────────────┬──────────────────────────┘
                                     gather ▲  │  ▼ write back
                 ┌──────────────────────────────────────────────────────┐
IMMUTABLE        │  WORLD       geo · rules    CoW·1·all NPCs           │
                 ├──────────────────────────────────────────────────────┤
CORE             │  ARCHETYPE   identity ·     CoW·1/type·~300          │
the lens,        │              voice · model       READ-ONLY           │
                 └──────────────────────────────────────────────────────┘
                   surfaced per turn as the SYSTEM-PROMPT view (§II):
                   ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
base of            │templat│ │mood   │ │identty│ │situatn│ │concrns│
the stack          │lck top│ │spiking│ │anchor │ │—      │ │+curios│
                   └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
  ═══ base of the attention stack ═══════════════════════════════════════
```

> **How to read this figure.** Bottom to top is the attention stack, the order the KV cache is assembled: the **immutable core** at the base (the lens, never gathered, surfaced as the system-prompt view), the **Cortex** above it gathering across the six **conversation-layers** and composing them by selection, and the **interaction surface** on top. Beliefs are drawn set apart because, alone among the layers, the Cortex may read them but never write them — only the evidence-threshold process can. Modulation and the monitor are omitted here to keep the stack clean; they appear in the glance figure above.

Each mutable conversation-layer is the same object: an append-only event stream, a mutation tool that feeds it, and provenance gather that reads it. The only things distinguishing them are **feed rate** (clock) and **content** — and even the clock is emergent from how fast events arrive and fade. Relationships drift over many interactions; mission over a campaign; memory accumulates without bound. This is not N subsystems; it is one mechanism run N times at different rates.

Not everything that shapes behaviour is a conversation, though, and conflating the kinds is how a layer count balloons. Three distinct kinds of object act on the gather, and only the first is a layer in the substrate:

- **Conversation-layers** are streams the NPC gathers over — perception, action, agency, relationships, beliefs, memory. These are the layers proper.
- **Modulation parameters** bias the gather's weighting but produce no content to gather — affect/mood, threat, curiosity. They are *priors on selection*, not streams. Mood biases everything toward its register, threat weights danger-relevant blocks up, curiosity weights novelty up; none of them is a conversation, and modelling them as one is a category error that inflates the architecture. They are weights on the gather.
- **The monitor** (metacognition) sits *outside* the gather and reads variance and cross-stream overlap to detect leak and degradation; it gathers nothing and is gathered by nothing (Part X-bis).

Adding a genuinely new dimension of inner life later — reputation, debts owed — is a new conversation-layer plus a mutation tool. Adding a new *bias* — vengefulness, caution — is a new modulation parameter, far cheaper, because it is a weight rather than a stream.

Note that **mood spans two of these kinds, which is why it caused confusion.** As a *modulation parameter* it is a standing weight on selection. But it also has a *source* that drifts in the substrate — the accumulating emotional evidence (events, slights, victories) — and a *register* that spikes into the system prompt, the discrete emotional mode currently surfaced from the immutable register library. The source drifts continuously and is gathered like substrate; the register snaps on a threshold and is selected like a system-prompt section; the bias it exerts on every other gather is the modulation. One phenomenon, seen at three points in the pipeline.

### How the lens and the substrate combine at assembly

KV assembly order *is* the weighting mechanism. The immutable core sits beneath the gathered mutable blocks, so attention reads every memory, belief, relationship, and perception *in the light of* the fixed values and processing model surfaced above it. Provenance selects which mutable blocks enter the working set; assembly ordering ensures whatever is selected is attended through the lens. There is no separate weighting system — ordering plus attention is the weighting.

---

## Part III-bis — Beliefs: the protected middle layer

Beliefs are the one conversation-layer that does not behave like the others, and the reason they need their own treatment is that the architecture as described so far has only two settings — the immutable core that can never change, and the mutable layers that drift freely under selection — and a mind needs a third thing between them.

### Why fade is the wrong mechanism for a premise

Everything mutable fades by non-selection: a relationship nothing reinforces stops being gathered, a mood nothing feeds thins out. That is correct for feelings and conclusions, but it is exactly wrong for a *premise*. "Hess is a man of his word" is not a value like "betrayal is unforgivable" (that is immutable archetype) and it is not a drifting feeling (that is mood). It is something the NPC holds *true*, reasons *from*, and — crucially — should not be able to **talk itself out of under pressure.** Fade gives an NPC the worst failure mode of current agents: the test fails, so it concludes the test is wrong; the world contradicts the plan, so it concludes the world must be mistaken. A premise that erodes the moment attention drifts elsewhere is a premise the NPC can rationalise away whenever it is inconvenient — which is not conviction, it is convenience wearing conviction's clothes.

### The mechanic: readable by action, writable only by evidence

A belief is a block the action layer can **read and reason from** but **cannot emit a mutation to.** The NPC cannot decide to revise a belief, the same way a person cannot decide to stop believing something — you do not choose it, evidence makes it untenable. So beliefs change through exactly one path: a dedicated evidence-accumulation process, running on the slow daydream/sleep clock, that counts disconfirming events against a frame-defined threshold and rewrites the belief only when the threshold is crossed. The action layer can *notice* a belief is under pressure and even `speak` about the tension — "everything I'm seeing contradicts what I know about Hess" — but it cannot resolve that tension by fiat. Resolution is evidence-driven and slow.

This is the same shape as the immutable core's protection, applied one level softer: the core is protected by being a read-only CoW prefix (cannot change at all); beliefs are protected by being read-only *to the action layer* while remaining writable by the evidence mechanism (can change, but only the hard way). The protection is **topological, not prompted** — it is enforced by the substrate's read/write permissions, not by an instruction telling the NPC to be principled. Every "don't talk yourself out of it" rule expressed as a prompt is a rule that fails under pressure; making the self-write structurally unrepresentable is a rule that cannot fail.

```
                            ┌──────────────────────────────┐
       read / reason from   │ BELIEFS                      │
   ┌──────────────┐  ┌─────▶│ "Hess keeps his word"        │
   │ ACTION layer │──┘      │ premises · write-protected   │
   │ (the tick)   │         └───────────────┬──────────────┘
   └──────┬───────┘             ▲           │
          │                     │           ╳ no write path from
          │ may `speak` the     │ rewrites  │ action — structurally
          │ tension, cannot     │ ONLY on   │ absent
          │ resolve it          │           ▼
          │                ┌────┴─────────────────────────┐
          └───────────────▶│ EVIDENCE-THRESHOLD process   │
       disconfirming       │ (daydream / sleep clock)     │
       events accumulate ─▶│ counts disconfirmation;      │
                           │ rewrites belief iff threshold│
                           │ crossed                      │
                           └──────────────────────────────┘
```

> The action layer reads beliefs and may even voice that one is under strain — "everything I see contradicts what I know about Hess" — but the arrow from action to belief is *structurally absent*. The only writer is the slow evidence process, and only once disconfirmation crosses the frame's threshold.

### Why this is the difference between a character and a rationaliser

For a game NPC this is directly the line between someone with convictions and someone who bends whatever way the moment pushes. An NPC whose beliefs are write-protected holds its ground when the world is merely inconvenient and changes its mind only when the evidence genuinely accumulates — and when it does change, the change is legible and earned, because it took a threshold of disconfirmation to get there. Varek trusting Hess until enough betrayals stack up, then *durably* not trusting him, reads as a person updating. Varek revising his read of Hess every time it is tactically convenient reads as nothing at all. The write-protection is what makes belief-change mean something when it finally happens.

### The belief/relationship boundary

Beliefs and relationships look adjacent but differ on exactly this axis. A relationship is a *calibration trajectory* that drifts continuously with each interaction — it is supposed to move easily, because trust is meant to track recent behaviour. A belief is a *premise* that resists casual movement and yields only to threshold. "I trust Alice today a little less than yesterday" is relationship drift, correct and fluid. "Alice is fundamentally not who she claims to be" is a belief, and it should take real evidence to write, not one bad afternoon. The same event can touch both — a betrayal nudges the relationship immediately and contributes one increment toward the belief threshold — but they update on different mechanisms and different clocks, and keeping them separate is what stops a single slight from rewriting a conviction.

---

## Part IV — Filtering, routing, and cross-layer influence are all selection

This replaces every conditional filter, activation predicate, subscription table, and source-to-output lookup that a more procedural design would require. None of them exist here. Each becomes a facet of the one gather.

**No filters — provenance attention is the filter.** A mutation always lands. Whether it influences anything is whether the provenance match pulls it into the working set when its layer is next attended. This is strictly safer than a conditional filter against the only failure mode that matters: a filter that drops contradicting evidence before it lands makes a delusion permanent, whereas evidence that always lands — even when it loses — accumulates, reinforces its own provenance cluster, and eventually out-competes a false belief. Soft fade by non-selection (reversible) and hard forget by consolidation (permanent) are the only two forgetting mechanisms, operating at two clocks. Nothing is ever suppressed at emission.

**No routing table — the layer is the topic, and "active" is the filter.** Mutation tools are typed by which layer they target. A broadcast lands on the *active set* of its target layer, where "active" is not a stored flag but the salience state already computed — a faded conversation is below the active cut, does not catch the broadcast, makes no progress, and dies by non-selection. There is no subscription table to maintain, because membership-in-a-layer plus liveness *is* the subscription, and both are computed, not stored. The mutation tools are themselves the broadcast vocabulary; which tool fires is provenance tool-selection, the mechanism the companion paper validates at 86 tools.

**No source-to-output table — visible behaviour is a projection of gathered state.** Where a procedural design would map mood-to-posture by lookup, this architecture projects a distilled state vector off the gathered cognitive state and publishes it to the animation and reflex consumers. That projection is *distilled*, not a full KV gather, because those consumers are latency-critical and must never sit on the decode path. Rich form (the reasoning, the tokens) goes where it can be afforded — the log, consolidated lazily; distilled form (the state vector) goes where latency is critical. Same generative act, two representations matched to consumer cost — the engine's prefill/decode discipline applied one level up.

**Cross-layer influence is co-presence in the working set.** Layers influence one another because a single gather can span several of them, and the softmax weights them against each other. A generous-disposition belief that is provenance-adjacent to a relationship decision is pulled into the same working set as a resentment mutation; the denser cluster wins attention; the weaker is present but outweighed. Cross-layer coherence is not enforced by a filter — it *is* the attention weighting over a multi-layer gather. Attention is the reconciler, now operating across layers.

---

## Part V — Stability, drift, and the center of gravity

With filters gone, the working-set budget is the only lever on cross-layer influence, and that is where the single genuine calibration risk lives.

The risk is runaway. Global salience across the mutable layers is mostly correct — a battlefield reversal *should* wash over mission, mood-evidence, and belief at once. But it admits a feedback loop: a mood spike wins the budget, the action it drives writes more mood-adjacent events, those reinforce the cluster, the cluster keeps winning, and the NPC becomes whatever it last felt most strongly with no built-in damper, because a cluster that keeps winning selection never fades by non-selection.

The restoring force is not an authored reserved slice of budget. It is **provenance density.** A deeply held value or a core relationship survives the storm because it is provenance-adjacent to enough of what the NPC is and does that it keeps winning a share of the gather even while a mood dominates. The center of gravity that makes a character rather than a mood is simply which regions of the substrate are richly enough connected to survive competition. Stability is not installed; stability is what a dense, coherent provenance structure *is*. An NPC with shallow, scattered beliefs would genuinely be blown around by every strong feeling — which is correct, because that is what a shallow person is like.

And the deepest layer of stability is exempt from the competition entirely: the immutable core is the prefix, never a gathered block, so identity is structurally guaranteed while the emergent stability above only has to carry beliefs, relationships, and mission, where drift is desirable anyway. The hard floor (identity) is free from CoW; the soft stability (character) is free from provenance density. No authored floor at any level.

Because filtering, routing, cross-layer influence, and stability are now all the gather, the entire control surface concentrates into one place: the salience function and the budget. That is the strength — one thing to get right — and the exposure — one thing to get wrong. It is the trade the companion paper names: the O(N) error problem is exchanged for a retrieval-quality problem the system bounds but cannot eliminate, and this architecture extends that same trade to the whole personality model. It is all retrieval quality.

---

## Part VI — The asynchronous event engine

A user turn is not a conversation turn. It is one event among many. A "conversation" is an emergent pattern — what it looks like when a player and an NPC happen to exchange events quickly — not a structure the engine hardcodes. In an MMO, "whose turn is it?" has no answer when three players and two NPCs share a square; turn-based conversation is a degenerate special case of an event stream, and going asynchronous stops pretending that case is general.

### The core asymmetry: perception is prefill, action is decode

The asymmetry that makes async pay off is the one the inference engine is built around. Ingesting events — *I got hit*, *ally down*, *advance* — is reading tokens into context: the cheap, batchable, parallel path. Generating an action is autoregressive decode: the expensive, serial path. The engine refuses to pay decode for perception. Under high event rate — a battle — fifty events land and are absorbed in batched prefill, one pass across the events and across every NPC in the fight at once, while decode is spent only at the moments an NPC actually acts. Event rate and action rate decouple, which is what stops decode latency from scaling with chaos.

### The unit of parallelism is the NPC, not the conversation

An NPC is one mind: one inbox, one serialized cognitive process folding over it. Parallelism is across minds — many NPCs in one GPU batch — and out in the world — events genuinely arriving at once — never inside a single head running several conversation threads that later reconcile.

A popular NPC fielding three players does not spin up three mind-copies. The three players' events land in her one salience-weighted inbox. When she is scheduled to think, the assembler builds one prompt and she emits one cognitive step that fans out into several actions — reply to A, glance at the fight, ignore C for now. Serial cognition, multi-action output, parallel I/O. The three conversations are reconstructed by observers from her stream of addressed utterances; they are a reading of the stream, not threads in her head.

This is why the popular NPC is the *cheapest*, not the most expensive: her world, core, and surfaced-frame blocks are one shared prefix, cloned by reference in microseconds, and the three replies diverge only at their suffixes attending over that resident prefix. Maximal prefix reuse is the strongest batch the engine can form. The barmaid with three customers is the best-case batch.

### The tick loop

Each NPC is a loop, and the only question each iteration is whether there is anything worth thinking about.

```
      ┌────────────────────────┐
      │ INBOX                  │◀──── timer event (heartbeat;
      │ world · inject ·       │      salience sets the rate)
      │ broadcasts · timer     │
      └───────────┬────────────┘
                  │
           empty? ├──── yes ───▶ ┌────────────────────────┐
                  │               │ BLOCK                  │
              no  │               │ 0 decode, not batched  │
                  │               └───────────┬────────────┘
                  ▼◀──── event arrives ──────┘
      ┌────────────────────────┐
      │ DRAIN inbox            │
      │ batched prefill —      │
      │ CHEAP perception       │
      └───────────┬────────────┘
                  ▼
      ┌────────────────────────┐
      │ GATHER                 │      ┌────────────────────────┐
      │ working set, budget B  │      │ PREEMPT                │
      └───────────┬────────────┘      │ high-salience event    │
                  ▼                   │ (damage · ally down ·  │
      ┌────────────────────────┐      │ direct order) jumps    │
      │ DECODE                 │      │ the queue, forces tick │
      │ one step, EXPENSIVE,   │      └───────────┬────────────┘
      │ through the lens       │◀─────────────────┘
      └───────────┬────────────┘
                  ▼
      ┌────────────────────────┐
      │ FAN-OUT                │──▶ acts → arbiter / log
      │ + mutation events      │──▶ mutations → substrate layers
      └───────────┬────────────┘
                  │
                  └──▶ loop: back to INBOX
```

In pseudocode:

```
  loop:
    block on inbox until an event arrives        # truly event-driven, never polled
    drain all pending events                     # batched prefill — cheap perception
    gather working set under salience budget      # provenance selection across layers
    decode one cognitive step                    # expensive, serial, through the lens
    fan out actions + mutation events            # write back to world and substrate
    # action finishes → loop; inbox empty → block again
```

The loop is continuous under load: as soon as the last action finishes, if events are waiting, it ticks again, and a busy NPC takes back-to-back ticks each draining a *fat batch* of events — bigger, better-informed thinking steps rather than one-event-at-a-time thrashing. Idle means blocked, and blocked costs nothing: an NPC waiting on an empty inbox burns no decode and is not in the batch, so only NPCs with pending events occupy the GPU. That is what lets the population scale — the GPU works on exactly the minds that have something to process.

A scheduled timer event sits on the same queue so that "nothing arrives" cannot mean "dead forever." It is an enqueued future event, not a spin-check, and its interval is the NPC's idle metabolism: salience sets the heartbeat, so a guard at his post ticks slowly while the same guard who just heard a noise ticks tight until things settle, then relaxes. Alertness for free, with no separate system.

### Salience-gated ticks and preemption

Under sustained load the inbox never empties, so the tick cannot fire on "ingestion done." It is salience-gated. Ambient load lets ticks fire lazily, draining the accumulated batch. A high-salience event — took damage, ally died, a direct order — is a preempt: it does not wait for the current action to finish, it interrupts and forces a tick now. This produces a continuous gradient with no modes and no combat branch anywhere in the code: an idle NPC has a slow heartbeat and the occasional empty wake; a busy NPC takes continuous ticks over fat batches; an NPC under fire runs the same loop while preempts keep yanking it to act now. The loop's tempo tracks the world's tempo, and that is what reads as alive.

### Perception never blocks on action

Ingestion and action overlap — new events prefill into context while the previous action is still decoding, the same dual-session overlap the engine uses for its probe and decode loop. This bounds the one real hazard of acting in a moving world: an action decoded against state that has already moved. The events that arrived mid-decode are already in the next tick's context, the reflex layer covers the gap, and the stale window is just action-decode latency — the thing already being optimised. Throughput and staleness are the same lever, not a tradeoff.

### Three cost-stratified layers of response

The gradient of tool availability — limited tools while ingesting, full tools at the tick — is itself the cost stratification, and it must hold or the asymmetry leaks.

| Layer | Clock | Cost | Acts how |
|---|---|---|---|
| Reflex | Per event | Microseconds — rule, distilled head, value lookup, off the GPU | Provisional action immediately: flinch, raise guard, begin the ordered advance |
| Deliberative | Per tick | One decode, batched across minds | Considered action in light of plan and lens |
| Planning | Per episode (daydream/sleep) | The expensive pass | Sets and revises intent blocks |

The reflex emits a provisional action immediately; the deliberative layer preempts and revises it when it arrives — *no, the death-prediction overrides the order, abort the advance.* There is no single instant of choice; there is a running intent that lower layers set by default and higher layers overwrite when they catch up. The instant ingestion does decode-to-maybe-act, decode is back on the perception path and the win is gone — so reflexes must never be a forward pass.

### Speech is a tool call; world acts commit through the arbiter

The NPC does not emit dialogue as a separate channel. **Speaking is a tool call** — the action layer calls `speak` exactly as it calls `advance`, `hold`, or `flag_anomaly`, on the same fan-out. There is only one place the NPC produces anything: the action layer's tool calls. This collapses what would otherwise be two kinds of output into one and yields a guarantee a generated chat channel cannot offer — because the words an operator reads are a *rendering of a speak-act the NPC actually took*, the conversational surface is welded to ground truth. An NPC cannot tell you the eastern flank is quiet while its action layer is reacting to being flanked there, because its words are a view of its acts, not an independent stream that could diverge from them.

Among those calls, two commit disciplines still differ. A `speak` call is append-only and conflict-free — three replies coexist on the log and merge for free, and the interaction layer narrates them. A physical mutation cannot both-happen: the log can hold both "advance" and "hold" as utterances, but the world cannot hold both positions, so world-mutating acts that race go through an optimistic-concurrency arbiter — the action carries the world-version, the log offset, it reasoned over, and the arbiter checks at commit whether that premise still holds, applying it if valid and reconciling or discarding it if stale. Same fan-out, two commit paths: speech and other append-only acts converge via the log, world-mutations via the arbiter.

---

## Part VII — The interaction layer

At the very top of the stack, above every other layer, sits the **interaction layer**: the surface through which an operator talks to the *mind*, and the thing that replaces the traditional chat channel entirely. It is a read-only observer over the action stream, and its defining property is that **it generates nothing.** It does not decode replies. It watches the action layer's tool calls — `speak` above all, but also movement and gesture — and narrates the selected ones into a coherent conversational surface for the operator. The dialogue an operator reads is a *summary of acts the NPC actually took*, never an independently generated channel that could say something the NPC did not do.

This inverts how a chat channel normally works. A traditional channel *is* the generator: you address it, it decodes a reply, and that reply can drift from whatever the character is actually doing in the world. The interaction layer cannot drift, because it is downstream of action rather than parallel to it. When an operator speaks, that input is an **event injected into the NPC's inbox** — scoped to the interaction, not broadcast to the world square — and the action layer responds to it like any other event by deciding (or not) to call `speak`. The interaction layer then renders that speak-act back to the operator. The operator's turn and the NPC's reply are both mediated through the same event substrate; the reply is a narration of the act the operator's event provoked.

The layer's second job is **rate decoupling.** World events tick at the world's tempo; the interaction runs at the pace of two people talking. Because speech is a tool call on the action stream, the operator's conversational tempo is simply how often the action layer chooses to call `speak` relative to how fast world-events churn beneath it. Varek can be ingesting fifty battle-events and acting on them while calling `speak` to the operator every few seconds; the interaction layer narrates only the speak-calls and whatever world-acts are worth surfacing as colour — "he glances east mid-sentence" — so the operator experiences a person at conversational pace while the mind runs at world pace underneath. The interaction layer is the filter that selects which acts become visible dialogue and at what tempo, hiding the event firehose below.

### Many interactions over one stream

Because the interaction layer is an observer rather than a session, **multiple interactions can run at once over the same action stream.** Three operators talking to Varek are not three conversations in his head — they are three observers reading the one stream, each scoped to a particular interlocutor. The NPC stays one mind, one inbox, one serialized action stream; the multiplicity lives entirely in the read-only observer layer, which is the cheap place for it to be. This is the popular-NPC result one level up: the expensive thing — the mind — is shared, and divergence exists only at the rendered surface. "User" is precisely this scope — the context of who is speaking or acting on the NPC — and it scopes both the injected event (this came from operator A) and the narration (render back to A what is observable in A's vantage).

### Inject, then narrate: the response is a summary of elapsed action

An interaction turn is not request-and-response. It is **inject an event, then narrate the slice of action-stream that elapses.** The reply to A is not "the NPC's answer to A's question" — it is a summary of *everything that happened on the action stream between A's turn and the response point*, narrated through A's scope. If A asks "what do you see?" and before the next tick Varek is flanked, calls `speak`, moves east, and flags an anomaly, A's narration is the sum of that: "Varek starts to answer, then breaks off as the eastern line buckles — he's moving before he finishes the sentence." This is what makes the interaction feel like talking to someone living in real time rather than a chatbot taking turns.

This forces a real pipeline stage. The interaction cannot narrate at the instant the event is injected, because the acts it must summarise have not happened yet — they happen on the next tick. So the flow is: inject A's event into the inbox → it rides to the next tick → the action layer ticks and fans out acts → a **gathering phase** collects the acts that elapsed in A's scope → narration begins. The wait for the tick boundary is the structural reason the response is a summary and not an echo: it is summarising a bounded window of elapsed action.

The window closes **on the tick.** Narration inherits the tick's definition of "done" rather than inventing its own — the tick is already where acts fan out, where prefill-refresh seals, where mood re-evaluates, where strategy broadcasts are consumed, so closing narration on the same edge means the summary always describes a clean, sealed unit of cognition, never a half-finished thought caught mid-decode. One clock, and the interaction layer reads off its boundaries. The accepted cost is that a deliberation spanning several ticks is narrated in installments — one beat per tick rather than one paragraph for the whole arc — which reads as someone thinking out loud in real time and matches the live act-stream the operator is already watching. Continuity across installments comes for free as long as each tick's narration attends over the prior beats in its gathered context: the summary of tick N reads tick N−1's summary, so the installments unfold continuously rather than restarting cold.

### Two streams at two latencies

The operator does not wait for the gathering phase to see anything. **Acts stream live as they occur** — Varek moving, drawing, the raw `speak` landing — surfaced to each scoped observer in real time, *before* the narrative summary begins. Then at tick close the gathering phase fires and the **narrative summary streams** — the considered read that ties the window into prose. Two streams on one interaction at two latencies: the immediate act-stream (low latency, "he's moving, he's drawing his blade") and the tick-bounded narration ("what just happened, and what he says about it"). This is how watching a person act and then explain themselves actually works.

```
  TIME ──────────────────────────────────────────────────────────────▶
                                            TICK boundary ─┐
                                                           │ window closes
  WORLD      ▓▓ ▓ ▓▓▓ ▓ ▓▓ ▓▓▓ ▓ ▓ ▓▓▓ ▓ ▓▓ ▓▓ ▓ ▓▓▓ ▓ ▓▓ ▓│▓ ▓▓ ▓ ▓▓▓
  events     fast churn underneath — ingested as batched prefill, cheap
                  │                                        │
  OPERATOR   ─────●────────────────────────────────────────│──────────▶
  A injects       "what do you see?"  (private to A)        │
                  │                                         │
                  └──▶ rides to next tick ──┐               │
                                            ▼               │
  ACTION                       ┌────────────────────────┐   │
  (the tick)                   │ DRAIN · GATHER · DECODE │   │
                               │ fan-out: speak, move E, │   │
                               │ flag-anomaly, strat-bc. │   │
                               └───────────┬─────────────┘   │
                                           │                 │
  LIVE       ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  ████ ██ ████│  acts stream   │
  act-stream  (nothing yet)     he moves,    │  to A live,    │
  to A                          draws        │  before        │
                                             │  narration     │
                                             ▼                ▼
  NARRATION  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ███████████████ ──▶
  to A        (waits for the window to close)     "You ask what he sees;
                                                   before he can answer
                                                   he's moving as the
                                                   line buckles."
             scoped to A's vantage · woven, not concatenated ·
             strat-bcast left out (no observable trace)
```

> **What the figure shows.** The world ticks fast (top lane); the operator's inject is one private event that rides to the next tick; the action layer drains the whole window at the tick and fans out acts. Two return lanes at two latencies: live acts stream to A *during* the window, the narration streams *at tick close*, scoped to what A can observe and woven into one beat. The internal strategy broadcast happened in the same window but never reaches A — no observable trace.

### Scoping is by observability, not relevance

A tick that consumes A's event drained the *whole* inbox — A's turn, the battle, B's question, a timer-fired daydream — and all of it produced acts in that window. The unrelated acts are not noise to filter out; they are the texture that makes the NPC read as a person with a life rather than a presence that exists only when addressed. So the gathering phase **scopes, it does not filter for relevance.** What A sees is the acts in A's scope plus the unrelated acts *observable from A's vantage*: if Varek breaks off mid-sentence to react to the eastern line buckling, A sees it — not because it concerns A, but because A is standing there watching him do it. If Varek simultaneously fires an internal strategy broadcast about the northern trade route, A does not see it — it left no observable trace. Relevance-scoping would strip the life out; observability-scoping keeps exactly the ambient texture that sells the NPC as real. The narration's job at tick close is therefore to **weave, not concatenate** — "you asked what he sees; before he could finish he was already moving as the eastern line gave way" is one continuous beat, because it happened in one tick to one mind, and the lens renders it as a single person split between two demands rather than two stapled reports.

This yields a parallax property in the multi-observer case: the *same* unrelated act is narrated differently to different observers because their vantages differ. The flanking A sees as "he broke off to look east," B — positioned east — sees Varek turn *toward*. One act on the stream, scoped through two observabilities, narrated through one lens, producing two true-but-different reads. The action stream is the single ground truth; the scoped narrations are the parallax — exactly how two people watching the same person from different sides of a room would each describe it.

The line that keeps this honest: **inject-events are private to their interlocutor's scope; acts — including `speak` — are visible by observability.** A hears what Varek *says* to B if A is in earshot, because that is an act in shared space — correct and lifelike. But B's injected event (B's question) is not an act and never surfaces to A as content; A sees Varek's *response* to B, never B's private injection, unless B spoke aloud in shared space. Hold that line and the ambient bleed-through stays truthful instead of leaking one operator's private channel into another's narration.

The same mechanism serves two privilege levels. A player talking to an NPC in-world gets an interaction scoped to diegetic reality — they address the character, and the narration stays inside the fiction. An operator talking to the mind from outside world-events gets an interaction that can see and address the mind directly, outside the fiction. One layer, one narration-of-action mechanism, two scopes: the difference is what the interaction is permitted to see and inject, not how it works.

Because the interaction layer is a view of acts rather than a generator, several earlier "bridges" need no special surface. The NPC asking a genuine question is the action layer choosing to call `speak` with an inquiry because a low-confidence block won the gather; the NPC speaking first is a `speak` call provoked by a player-entered-square event outranking silence; an emotional shift mid-conversation is a mood spike changing the register the next `speak` is decoded through. In every case the interaction layer simply narrates the resulting speak-act. There is one tool vocabulary, one fan-out, and one observer on top of it.

---

## Part VIII — Strategy and planning as slow-clock layers

Planning lives on a different clock from the tick. Bolt it onto every tick and the NPC re-derives its life story before each action — slow, and thrashing as cheap events jitter the inputs. Leave it out and the NPC is a brilliant reactive agent that goes nowhere, responding perfectly to each event but never deciding to desert. So planning is present but not per-tick: another layer on the same substrate at a slower clock.

A plan is a standing high-salience block — *reach the city by nightfall*, *work up to betraying the captain* — written by the deliberative layer, persisting, and gathered into subsequent ticks with high salience so that each tick-level action decodes against context that includes the current intent. The plan biases moment-to-moment choices without dictating them; the NPC is not executing a script, it is acting each tick in the presence of a goal it can still see. Abandoning a plan is not a special operation — it is the intent block losing the salience competition to reality. The captain you meant to betray gets transferred, that strategy stops being surfaced, and you never had to detect invalidation.

Strategies are a conversation in the layer beneath action, several active or finished at once. Within-strategy progress is free: the action layer, acting with a strategy pulled forward, writes consequences back into the strategy's stream, so next time the strategy is gathered, what comes forward is its current folded reading including its progress. Progression and selection are the same event — a strategy advances by winning action-attention often enough to keep generating actions toward itself, because being pulled forward and acted on *is* a unit of progress. Discrete transitions, active to finished or to death, are the output of a strategy tick that folds the progress stream plus recent events and either advances the strategy, marks it finished, or lets it die. Dense-activity strategies self-trigger and progress fast; dormant ones get their periodic re-read on the daydream and sleep clock; finished strategies stay on the substrate as material new strategies are drawn from.

The action-to-strategy coupling is a **fire-and-forget broadcast** — the action layer, the only thing in contact with the world, calls a mutation tool that writes to the strategy layer and returns immediately, with no strategy-computed result flowing back into the action context. The instant an action decode waits on a strategy response, the fast clock blocks on the slow clock and the asymmetry collapses; so the rule is write up, do not await, and let the strategy layer consume on its own tick. The significance judgment costs nothing extra, because deciding to call the broadcast tool is part of the action tick's existing decode.

Decomposition is the one place genuine structure is authored. A goal like *control the northern trade route* does not emit a tick-level action — a single decode bridging a multi-day goal to a micro-action tends to come out incoherent — so a strategy tick decomposes into sub-goal blocks, and progression becomes recursive: a sub-goal completing broadcasts goal-achieved upward, waking the parent. Because a child was decomposed from its parent, the two are provenance-adjacent by construction, so the completion routes to the parent by content-adjacency on the active set with no explicit hierarchy edge. This recursion — the decompose operation and the notion of a sub-goal being done enough to advance its parent — is the dial that sets what kind of NPC you get. Flat single-intent, the goal simply present in context biasing ticks and revised at daydream and sleep, is emergent and cheap and enough for an ambient inhabitant whose strategy is really persistent mood-with-direction. A nested goal-stack, where decomposition writes a sub-goal lineage that holds a multi-step scheme across episodes, is what a player can read and counter — a warlord running a recognisable weeks-long campaign. Decomposition depth is a per-NPC dial: a wandering civilian gets flat, an enemy commander gets deep.

A genuinely novel consequence that no active strategy anticipated has no content-neighbour among existing strategies, so layer-scoped routing delivers it to nobody. That is not a bug if caught at the right clock: orphan consequences are the seed an idle daydream tick picks up to *form* a new strategy, content-addressed against the persona and drives layer rather than the strategy layer — a surprise births a strategy precisely when it resonates with what the NPC fundamentally wants. Progression gathers consequence against existing strategy; formation gathers surprise against drives. One mechanism, two targets.

---

## Part IX — Memory, consolidation, and the only garbage collector

The substrate accumulates; salience gates what influences behaviour; consolidation is the only true garbage collector. There are two forgetting mechanisms at two clocks. Attention-fade is soft, reversible, and continuous: a block nothing reinforces stops being surfaced by the gather, dormant rather than deleted, so the right cue can resurface it years later — which is how grudges and trauma actually behave, an emergent feature rather than something faked. Consolidation-forget is hard, permanent, and bounded to the sleep fold, which decides what *not* to carry into consolidated memory; a daydream or memory is re-awakenable right up until the sleep pass chooses not to keep it.

Daydream and sleep are one consolidation at two timescales. Daydream is the lowest-priority event on the queue, surfacing only when nothing else competes — an idle NPC spending its own slack — and shed first under load with no explicit load-shedding logic, because the lowest-value work is naturally crowded out when the GPU is busy with NPCs that have real events. Idle minds think; busy minds act. Sleep is the scheduled full pass: convergence over what happened, how relationships changed, what contradictions appeared, and what to prioritise, plus dream cycles that re-run unresolved scenarios against the current evidence base.

A daydream writes to two places at once, for the two timescales. Into the volatile substrate it writes a compressed state-delta — a mood-evidence shift, biased thresholds — that colours the lower layers now, so a brooding NPC is already shorter-tempered at the next event rather than only after it next sleeps; the reflex layer reads this distilled projection, never a full daydream-block gather. Into the conversation stream it writes the full reasoning trace that sleep distills into semantic memory, the lasting record. Wake clears the volatile substrate and re-seeds it from what consolidation kept, which is why mood fades overnight unless the daydream was significant enough to survive the fold.

Rumination and decay fall out for free, because volatile mood-evidence lives as blocks the deliberative tick gathers. Decay is non-selection: a grievance nothing feeds stops being gathered and fades. Reinforcement is provenance-cluster density: each return to a theme writes another block in the same neighbourhood, the cluster densifies, it keeps winning selection, and the NPC becomes progressively more fixed on it — rumination — while an NPC whose idle thoughts wander stays balanced, and a single bad afternoon against a day of other events loses the competition and thins on its own. Decay, reinforcement, and rumination are one rule written nowhere.

The one question with teeth over a long-lived character is whether the sleep fold is lossy enough to keep the substrate bounded across a year of play. That is a consolidation-policy problem — lossy summarisation, salience-weighted retention, emotional tagging — and it is contained: it runs at a boundary, on one NPC, off the hot path, and does not ripple through the system. It is the right shape for the one hard problem to be the only hard problem.

---

## Part X — Behavioural expression as emergent output

A mind without expression is invisible, and the difference between "impressively intelligent when you talk to it" and "I believe this NPC is alive" is the set of output channels that bridge internal state to visible behaviour. None of them is a triggered slot with an activation predicate. Each is an emergent consequence of the gather.

Initiation — the NPC speaking first — is the NPC ticking on the event of a player entering its square, and whether it speaks first is whether a concern block wins the gather and outranks silence, with the resulting `speak` call narrated by the interaction layer; salience decay, having already voiced a concern, is non-selection rather than a state transition. Autonomous action is the action layer's decode fanning out world-acts through the arbiter, with the archetype lens biasing execution because the lens is the prefix every action is read through — a Paranoid Survivor scouts escape routes first not because a rule says so but because that disposition is in the lens the action is decoded through. Involuntary state is the distilled projection off gathered mood and conflict, published to the animation channel — a projection, not a lookup table. Curiosity is a low-confidence belief block, provenance-adjacent to the current exchange, out-competing assertion so the action layer calls `speak` with a question rather than an assertion — the NPC asks because the gap won the gather. Mind-changing is accumulated contradicting evidence — which always lands, never filtered at emission — reinforcing its cluster until it out-competes a prior resolution at the next dream tick, so the NPC's position shifts because selection shifted, not because a re-evaluation trigger fired.

The common thread is that the behaviours are readings of one substrate, not modules bolted to it — which is the mark of the architecture being right. A procedural design describes these behaviours and then hard-codes a trigger for each; this architecture derives all of them from the single mechanism.

---

## Part X-bis — Metacognition: the monitor outside the gather

There is one component that is not a conversation-layer, not a modulation parameter, and not part of the gather at all: a monitor that reads the *health* of the whole system from the outside. It earns separate status because the thing it watches for is a failure the gather cannot prevent on its own.

### The runaway-loop exposure

The architecture's strength is also its characteristic failure mode. Gather promotes substrate into the working set; the action layer attends over it and acts; the act writes back into the substrate; the next gather pulls that back in. Under normal conditions this is just the NPC thinking. But attention is recency-weighted, so recently-written content pulls disproportionately on the next step — and that opens a loop: the NPC's own reflected reasoning gets gathered as if it were fresh signal, shapes the next output, reseeds the substrate, and within a few cycles the NPC is ruminating in a tightening spiral disconnected from world-events, each layer reading the last as evidence. This is not prevented by any boundary already described — it lives *inside* attention-weighted gather, which is exactly where the architecture does its work. The same loop that produces healthy entrenchment (a grievance reinforced into a grudge) produces pathological rumination when it tightens faster than world-events can perturb it.

```
   HEALTHY                                  RUNAWAY
   world-events perturb the loop            loop tightens faster than world perturbs

      world ──┐                                world ··┐  drowned out
              ▼                                        ▼
        ┌─────────┐                               ┌─────────┐
        │ GATHER  │◀────┐                         │ GATHER  │◀────┐
        └────┬────┘     │                         └────┬────┘      │
             ▼          │ writes                       ▼           │ writes back
        ┌─────────┐     │ back —                   ┌─────────┐     │ louder each
        │ DECODE  │     │ one voice                │ DECODE  │     │ cycle (own
        └────┬────┘     │ among many               └────┬────┘     │ reasoning read
             ▼          │                               ▼          │ as fresh signal)
        ┌─────────┐     │                          ┌─────────┐     │
        │  ACT    │─────┘                          │  ACT    │─────┘
        └─────────┘                                └─────────┘
      external signal stays                    own output dominates the
      louder than self-echo                    next gather → spiral inward
```

> The diagram on the left is just an NPC thinking — its own writes are one input among many, and world-events keep the loop honest. On the right the *same wiring* tips: when self-echo outweighs external signal on each successive gather, the loop closes on itself. Nothing in the topology changed; the balance of salience did. That is why this cannot be prevented structurally and must be *measured*.

### The detector is the narration-honesty guard

The monitor that catches this is cheap and — usefully — is the **same instrument** as the guard that protects the interaction layer's can't-lie property. In a healthy run, the interaction-layer narration is correlated with substrate *content* but not with substrate *surface form* — the NPC's words track what it is thinking, not the raw token-shapes of its internal streams. In a leaking or ruminating run, the surface forms begin to rhyme: the narration starts echoing phrasings that only make sense if the surface had read the substrate's raw form rather than rendered committed acts. Rising n-gram / mutual-information overlap between the narration stream and the substrate streams is the operational signature of both failures at once — the runaway loop tightening, and the narration drifting off committed acts toward raw substrate echo. One metric, two guarantees.

### Why it must be separate from the interaction layer

The monitor cannot be folded into the interaction layer (or any layer that produces output), for a structural reason: detecting your own degradation requires reading yourself as an object, which a layer producing first-person narration structurally cannot do while it is producing that narration. A flattening interaction layer cannot be the thing that reports "the interaction layer is flattening" — it would have to flatten further to make the observation. So the monitor sits outside the gather, reading variance and cross-stream overlap, and can raise a health signal — *this NPC is approaching the runaway band* — without itself being caught in the loop it watches. It does not act and it does not narrate; it measures.

### Graceful degradation as the production property

The payoff of having the monitor is that degradation becomes *navigable* rather than a cliff. Because the same noise that lets an NPC explore and surprise is the noise that, amplified, produces rumination, a well-built NPC of this kind will *have* near-failure states — and that is desirable, since a character that cannot fail in mind-like ways reads as scripted (Part X-ter). The monitor's job is to keep those states in the expressive band — "slightly more fixated than usual," which reads as a brooding character — and to flag when an NPC is sliding past it toward incoherence, so consolidation can intervene at the next sleep fold. The design goal is that an NPC fails toward *more characterful* before it fails toward *broken*, and that the gradient between them is measured rather than discovered too late.

---

## Part X-ter — Failure modes as a character palette

A consequence worth stating positively, because it inverts how failure is usually treated. A perfectly coherent NPC — optimal every move, on-theme every response, never surprising — is the uncanny valley of strategic play. The texture of personhood is partly the *near*-failures, and this architecture produces them for free as the extremes of dials it already has.

An NPC whose threat modulation over-weights under sustained siege starts reading hostile intent into neutral acts — that is *paranoia*, and it makes a more compelling enemy commander than one who assesses threats correctly every time. An NPC whose affect modulation over-dampens after a defeat goes quiet and initiates nothing — that is *grief*, and it reads as more real than one who recovers on schedule. An NPC whose belief threshold is set high holds convictions stubbornly against mounting evidence — that is *zealotry* or *loyalty* depending on the belief. None of these are new systems; they are the modulation parameters and the belief threshold pushed toward their edges. The document's catalogue of failure modes is, read this way, a catalogue of *character states* reachable by moving existing dials.

The discipline is that these must stay in the expressive band and remain recoverable — paranoia as flavour, not as a permanent break with the world. That bound is exactly what the metacognition monitor (Part X-bis) provides: it is the instrument that lets you push an NPC toward a characterful near-edge *deliberately*, with a measured signal for when it is about to tip past character into incoherence. Failure-as-flavour is a feature you can dial; the monitor is what keeps the dial from going to eleven on its own.

---

## Part XI — Propagating back: individual, collective, individual

The learning loop is bidirectional, and the immutable-core decision is what keeps it simple. The hard constraint is that **the archetype core — values, voice, processing model — never propagates back, because it is immutable.** What propagates is the one part of the shared layer always designed to evolve: strategic doctrine. The individual's lived substrate stays local and never aggregates.

```
   PLAYER ── assigns novel mission ──▶ NPC (individual)
                                         │ executes through:
                                         │   immutable archetype lens   (fixed)
                                         │ + personal lived substrate    (mutable, local)
                                         ▼
                                  outcome → MEMORY (strategic-learning block, local)
                                         │  sync learning upward
                                         ▼
                                  CORE SERVER  (aggregates, runs no AI)
                                  counts outcomes across all NPCs of this archetype
                                         │  statistical significance reached
                                         ▼
                                  ARCHETYPE DOCTRINE updated worldwide
                                  (the one evolving part of the shared layer)
                                         │  published to every node
                                         ▼
                                  all NPCs of this archetype see new doctrine
                                  at next spawn / fork refresh
```

Three things share that diagram and behave differently. Identity — what a Loyal Soldier *is* — is shared and fixed forever. Doctrine — how a Loyal Soldier *fights*, flanking ratios and crossing tactics — is shared and evolving, the part of the archetype explicitly marked mutable. Lived experience — this NPC's relationships, beliefs, grudges, and memories — is individual and evolving and never aggregates, and is what makes one NPC differ from another despite identical archetype and identical doctrine. The individual feeds the collective only through doctrine; the collective feeds new individuals through both fixed identity and current doctrine; the individual's lived depth stays its own.

Because the propagating quantity is doctrine only, the core server aggregates rather than computes cognition: it collects strategic-learning entries, computes statistical significance, and publishes doctrine updates over one authoritative channel. Intelligence is distributed — every player's idle GPU processes NPC lives from the global queue — while coordination is centralised. More players means more idle compute means smarter NPCs at zero marginal cost. A personal learning is promoted when it clears a significance threshold across enough NPCs of the same archetype; promotion copies the generalisable tactic into the mutable doctrine sub-layer, version-bumped, while the lived experience of discovering it stays in the originating NPC's local substrate. The immutable-core decision is precisely what lets propagation be a counting problem rather than a model merge.

---

## Part XII — The frontier: what only running it will tell you

The architecture concentrates every behaviour into the salience function and the budget. That is deliberate, and it is the whole bet. The questions that remain are therefore not architectural but calibration questions, answerable only under real event load.

Whether one depth-band weighting serves mission, mood, and belief gather, or whether they diverge the way the companion paper's eight content types did, with per-layer calibrated weights the expected outcome and the pragmatic-dominant universal default only the cold-start. Whether idle NPCs drift somewhere interesting or rut on the highest-salience cluster — the early tell for whether daydream salience is exploring or exploiting, a dial set only by watching. Whether the sleep fold is lossy enough to keep a year-old NPC's substrate bounded without erasing the lived depth that makes it a character. Where the belief-revision threshold sits — high enough that convictions resist convenient erosion, low enough that a genuinely wrong premise eventually yields, with the same trade-off between zealotry and gullibility at the two extremes. And whether global salience self-corrects via the changing event stream, or whether some NPCs fall into attractor states that a denser provenance prior on core drives — not an authored budget floor, but a richer connection structure on what the NPC fundamentally wants — would prevent.

The metacognition monitor turns the last of these from a hope into a measurement: the runaway band is not a thing to guess at but a thing to watch the overlap metric approach, so the open question is less "will NPCs ruminate" than "where do we set the threshold at which the monitor flags an NPC sliding from characterful fixation toward incoherence" — which, like the rest, is read off real runs rather than reasoned out in advance.

None of these is a design flaw to fix before running. They are the questions the design *exposes* as the real ones, which is what a correct architecture does: it makes the hard problems small, contained, and empirical. Build it, tick it under load, and watch which way each dial wants to go.