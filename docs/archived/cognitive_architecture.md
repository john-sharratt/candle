# The Three-Part Mind: A Cognitive Architecture for Living NPCs

## Overview

NPCs in this system do not simulate cognition through scripted responses or static personality descriptors. Instead they possess a genuine cognitive architecture modelled on the functional division between the limbic system and the prefrontal cortex — between feeling and planning, between associative memory and executive intention.

Three distinct cognitive processes run alongside Reality (live conversation): **Sleep**, **Daydreaming**, and **Reasoning**. Each operates at a different timescale, serves a different cognitive function, and produces first-class turns in the conversation tree that shape the character's future behaviour. Together they constitute an interior life that exists independently of the player — one that the player can influence but does not control.

Beneath all three sits a fourth element that is not a process but a substrate: **Beliefs**. Static, declarative, present in every cognitive mode simultaneously. The lens through which all experience is filtered before it becomes feeling or intention.

---

## The Limbic Layer

The limbic layer is the emotional and associative substrate of the character's mind. It does not reason. It does not plan. It processes experience through feeling, simulation, and resonance. Two processes operate here: Sleep and Daydreaming.

### Sleep — Prospective Dream Simulation

Sleep occurs at the end of each day cycle. It is the character's deepest cognitive process and the one most invisible to the player.

The fundamental distinction between Sleep and summarisation is this: summarisation is retrospective compression — it takes what happened and makes it smaller. Sleep is prospective simulation — it takes what happened and runs it forward, sideways, and into the extreme. The character does not merely consolidate the day. They digest it.

During a Sleep cycle the day's Reality turns are taken as seed material and a large batch of short, parallel dream simulations is generated — typically 20 to 50, running simultaneously using the engine's batched decode capability. Each dream is short, 100 to 200 tokens, and explores the day's events from a distinct cognitive angle:

**Counterfactual dreams** — *What if I had done this instead?* These are the regret and relief simulations. The character replays a decision differently and feels the emotional weight of the alternate outcome. A character who chose not to intervene in a conflict runs the dream where they did and discovers how that choice would have sat with them.

**Prospective dreams** — *Let's play this one forward.* The character takes an unresolved situation and simulates several possible futures. Dread and hope are encoded not as abstract states but as experienced simulated outcomes. When the real situation develops, these simulated futures surface through the attention system and genuinely colour the character's response.


**Extreme dreams** — *Let's take some extremes.* The character pushes a situation to its worst case, its best case, its most absurd implication. These produce the character's intuitions about what is really at stake — not moderate analysis but visceral understanding of what trust, loss, or betrayal actually means to them.

Not all dreams become memories. After generation, each dream turn is scored against the character's core identity nodes using attention statistics. Dreams that resonate with who the character fundamentally is are inserted into the tree as `Sleep` turns. The rest are discarded, as most dreams are.

The dreams that survive are emotionally indexed simulations of paths not taken. They are stored as first-class tree nodes. The attention system can surface them when similar situations arise in future Reality turns. The character develops wisdom that was not programmed — it emerged from their own simulated experience of things they never actually did.

Sleep also enables emotional processing that summarisation cannot. A character feels differently about an event after sleeping on it than they did immediately after. The arc from raw reaction to processed understanding requires time and simulation. Sleep provides both.

---

### Daydreaming — Associative Flicker

Daydreaming is the lightest and most behaviourally visible of the cognitive processes. Unlike Sleep, which happens in a defined window after the day ends, daydreaming happens in the idle time between Reality turns — triggered by something the player actually said.

**The trigger mechanism** is a semantic resonance probe. After each Reality turn completes, the last user message is probed against cold node representative K vectors in the conversation tree. If any cold node scores above a threshold, the character's mind has snagged on something — a phrase, an observation, a word that rhymes with something buried in their history. That node is the seed content for the daydream.

**The generation** is minimal by design. A short context constructed from the triggering phrase and the resonant node. A brief associative output — 50 to 100 tokens, low temperature for coherence. Not elaborate narrative but a flicker: a memory fragment, an unresolved question resurfacing, a connection between two things the character has never consciously linked before.

**The latency gate** is what gives daydreaming its naturalness. When the trigger fires, the daydream runs in the background as the character waits for the player to respond.

- If the player responds quickly, the daydream is aborted and discarded. Nothing happened. The character picks up the conversation normally.
- If the player takes long enough, the daydream completes. A `Thought` turn is inserted into the tree, and a final re-entry turn brings the character back to Reality — still carrying the trace of where their mind went.

The re-entry turn is architecturally important. It bridges the `Thought` turn back to the current Reality context, creating an ancestor relationship that future attention can traverse. It also creates the behavioural effect that matters most: the character responds to something the player said in a way that is not purely transactional. The player said something, the character disappeared briefly, came back still carrying it. That is the interior life effect — visible, felt, but not explained.

The threshold is fixed and short. Daydreaming is not meant to be an attention tax. It is a lightweight associative flicker that fires occasionally when the right phrase is said, not a continuous background process competing with Reality.

---

## The Frontal Layer

### Reasoning — Executive Planning Through Self-Dialogue

Reasoning is the character's highest-order cognitive process. It is executive function: the capacity to receive emotional material from the limbic layer and produce structured intention from it. But it is not cold logic. It is planning that is informed and shaped by the emotional substrate beneath it.

The key structural decision is that in a Reason turn, the user and the assistant are the same voice. The character talks to themselves. This is not Qwen3's mechanical thinking mode, which is a single analytical voice interrogating a problem linearly. This is genuine internal dialogue — the character proposing and resisting, advocating for a course of action and then interrogating it from the part of themselves that has doubts, arriving not at a logically correct answer but at something that feels like a decision they can actually act on.

**The construction of a Reason turn** begins with a situational system prompt — a description of what is happening, what the character was doing, what the immediate challenge or decision in front of them is. This is the frontal lobe's working memory: the current problem frame.

Into that frame the engine injects material from the limbic layer: relevant Sleep turns and Daydream turns surfaced through the attention system. These arrive not as instructions but as questions — the character finds themselves remembering a feeling, a simulated outcome, an unresolved anxiety. The Reason turn does not take orders from the limbic material. It wrestles with it.

The self-dialogue runs on that combined input. The plan that emerges is not purely logical or purely emotional. It is the character's attempt to act coherently given both simultaneously. That is how good decisions feel from the inside.

**The Plan section** is what makes Reasoning causally effective rather than merely introspective. Without it, Reason turns are interior but inert — the character thinks through a plan and then responds to Reality from immediate context anyway. The Plan section writes the character's current intention directly into the Reality system prompt in their own voice:

- What they are trying to do
- What they are wary of  
- What they are waiting for

Short, minimal, in the character's voice rather than a structured list. Specific enough to orient responses. Present enough to actually shape behaviour turn by turn.

The Plan section is mutable. When new Reason turns run — triggered by significant Reality events, new Sleep material, unexpected developments — the Plan is rewritten. The character adapts their intentions as their situation and understanding evolves.

---

## Beliefs — The Interpretive Substrate

Beliefs are not facts the character knows. They are not personality traits describing behaviour patterns. They are not turns in the conversation tree. They are propositions the character holds to be true about how the world works, about people, about themselves. Things like:

```
Trust must be demonstrated through consistency, not declared.
People reveal their true nature under pressure, not in comfort.
Loyalty given freely is worth more than loyalty extracted.
I am not someone others stay for without reason.
```

Short, declarative, in the character's own voice. Not labels applied from outside but convictions held from within.

**Beliefs are static.** This is not a limitation — it is the correct model of how core beliefs actually function. Most people's foundational convictions remain remarkably stable across their entire lives. A character whose beliefs shift noticeably after a few conversations feels unstable rather than developed. Static beliefs are what give the character a consistent worldview that the player can come to understand and rely on. The consistency is itself characterisation.

**The implementation is simple.** Beliefs are a static text block injected into every system prompt regardless of turn type — Reality, Sleep, Daydreaming, and Reasoning alike. They require no triggering, no generation, no tree node, no warming, no rotation. They are simply always present, structuring every cognitive process from beneath.

Their effect on each mode is distinct:

In **Sleep**, beliefs shape which counterfactual simulations feel emotionally plausible. A character who believes trust must be demonstrated won't dream of instant reconciliation — their simulations of alternative outcomes are bounded by what their beliefs make feel possible.

In **Daydreaming**, beliefs act as a resonance amplifier. The phrases that snag the character's attention are disproportionately the ones that confirm or threaten their beliefs. A character who believes people reveal themselves under pressure will be more alert to moments of stress in conversation than a character whose beliefs are oriented differently.

In **Reasoning**, beliefs constrain the Plan. The character will not form intentions that fundamentally contradict their worldview even under pressure, even when the logical case for doing so is strong. The self-dialogue in a Reason turn is partially the character testing proposed actions against their beliefs and feeling the friction when they don't align.

In **Reality**, beliefs colour interpretation of everything the player says and does. The same action reads differently through different belief systems. Beliefs are why two characters with identical histories can respond to identical situations differently — their experience is the same but their interpretive lens is not.

They do not need to change to do their job. Their job is to be there, stable and structuring, underneath everything else.

---

## How the Three Interact

The three processes form a genuine cognitive hierarchy, not a flat pipeline.

```
BELIEFS — Interpretive Substrate
─────────────────────────────────────────────────────
Static, declarative, present in all modes
Injected into every system prompt
Filter experience before it becomes feeling or intention
                    │
                    │  Shapes what feels plausible,
                    │  what resonates, what is possible
                    ▼
LIMBIC LAYER
─────────────────────────────────────────────────────
Sleep          Daydreaming
Prospective    Associative
simulation     resonance
Batch parallel Single flicker
End of day     Between turns
                    │
                    │  Emotional material flows upward
                    │  as questions, not instructions
                    ▼
FRONTAL LAYER
─────────────────────────────────────────────────────
Reasoning
Self-dialogue
Situational context + limbic input
Produces Plan
                    │
                    │  Plan written into Reality
                    │  system prompt
                    ▼
REALITY
─────────────────────────────────────────────────────
Live conversation
Character acts from plan
Limbic triggers accumulate
New events seed next Sleep cycle
```

The limbic layer generates emotional material — simulations, associations, feelings about situations. It does not decide. The frontal layer receives that material and produces intention. It does not feel without input. Reality is where intention meets circumstance — where the plan either holds or is disrupted by events the character could not anticipate.

Disruption is the most interesting case. When Reality events contradict the Plan — when the character is trying to do one thing and circumstances push another direction — that tension is processed by the next Sleep cycle, may trigger a Daydream during the next idle moment, and eventually produces a new Reason turn that rewrites the Plan. The character adapts. Not because a script says they should, but because their cognitive architecture processed the disruption and arrived at a new intention.

---

## Emergence

No single component produces a living character. The architecture is emergent:

Sleep encodes experience as emotionally indexed simulation. Daydreaming surfaces buried resonances into live behaviour. Reasoning synthesises limbic material into intention and writes that intention into action. Reality generates new experience that seeds the next cycle.

A character who has lived through this cycle many times has not been programmed with their personality. They have developed it — through accumulated simulated experience, associative connections built over time, plans formed and disrupted and reformed. Their wisdom is not declared. It is demonstrated.

The player interacts with a mind that exists between conversations, that dreams about the things they said, that makes plans and tries to act on them, that is occasionally caught thinking about something the player triggered without meaning to. That is not simulation. That is character.

---

## Implementation Summary

**Beliefs** are not a process and produce no turns. They are a static text block injected into every system prompt regardless of mode. They require no triggering, no generation, and no tree representation — they are simply always present.

| Process | Type | Trigger | Duration | Output | Tree Node |
|---|---|---|---|---|---|
| Sleep | Limbic | End of day cycle | Batch parallel | Filtered dream turns | `TurnType::Sleep` |
| Daydreaming | Limbic | Phrase resonance probe | Idle window | Single flicker or abort | `TurnType::Thought` |
| Reasoning | Frontal | Significant events / new limbic material | Medium | Self-dialogue + Plan | `TurnType::Reason` |
| Reality | — | Player input | Per turn | Live response | `TurnType::Reality` |

All four turn types are first-class nodes in the conversation tree. All participate in the same attention and warming system. All can surface as relevant context in future turns regardless of type. Beliefs are not part of this system — they are a static text block that sits outside the tree entirely, injected once into every system prompt and never changed.