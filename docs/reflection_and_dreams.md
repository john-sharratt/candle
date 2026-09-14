# Reflection and Dreams

*The second and third conversations a character runs: what reflection is for, how a dream corpus becomes a character's private inner world, and why the coupling between them is the whole safety argument.*

**Where this stands.** The reflection conversation is built and produces dream briefs: `npcd/src/engine/reflect.rs`, `POST /v1/npc/:nid/reflect`, with both its questions authored in the mind's `projection.yaml` under `reflection:`. The character's own `reflect` act runs it (§3), asynchronously, and is answered with the reflection's first line. The brief is dreamt in a conversation of its own and the dream is written to a per-character `dreams` layer a line to a turn, each line signed (§7, `npcd/src/engine/dreams.rs`). Dreams are gathered back by provenance: three lines deep into a turn in the room, eight into a reflection (§4). What is not built is the rest of the gather — `relationships`, `beliefs`, the mood — eviction and `LayerHit` (§8), and the authored seed dreams (§9). Each section says so where it applies.

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

**Built.** `reflect` takes `situation`, `inner_thoughts` and `feeling`, and a reflect the world takes starts a reflection (`Runtime::begin_reflection`):

| param | required | what it is |
|---|---|---|
| `situation` | yes | The character's own account of where it is and what is happening, in prose. It opens the reflection conversation. |
| `inner_thoughts` | yes | What is going through its head — unchanged. |
| `feeling` | yes | From `Choices::Feelings` where the mind authors moods, free text otherwise. **This sets the mood.** |

`my_reflections` is gone. It was the field asking a character to produce, inline, the thing this whole design exists to produce properly — and it is half of what made the act's output a restatement of its input.

**The act's answer is the reflection's first line.** The reflection runs on a thread of its own; the tick driver goes straight on to everybody else. Only the reflecting character is held (`Scheduler::hold`), so the room — which is delivered every moment — cannot give it a turn before its reflect has been answered; what arrives meanwhile is queued, not dropped. The moment question one is answered, the line is handed back as the reflect call's `<tool_response>` (every answer the turn owes, in call order) and the character is released. The loosed turn, question two, the retry, the repair passes and the dream all carry on behind it, on the same thread, with nobody waiting. A reflect is never refused for being soon after another: a character is held until its answer lands, so it cannot ask twice at once, and only the dream work after the answer is one at a time (§7). A reflect that cannot be answered by a reflection — a daemon whose mind authors no questions, or a reflection that failed before its first answer — is answered, in the character's own voice, that nothing came: *"I don't seem to be able to reflect on anything right now."* (`body::NO_REFLECTION`). Never an echo of the character's own arguments; its thought and feeling are already in its call.

**The row says what came back.** In the feed and the character's window a reflect reads `reflect — <inner thought> → <what came back>`: its situation and feeling are the reflection's input and are not listed, and the right of the arrow is exactly what the character read in its `<tool_response>`. The row is recorded as the act alone when the reflect lands and completed when the answer does (`Scheduler::amend_act`).

**`situation` is what makes the reflection conversation possible.** That conversation has no perception, no window, and no world state; it is framed on the character's own description of its circumstances. Asking the character to say where it is is not redundancy — it is the handoff, and it is also a cheap diagnostic, because a character that describes its situation wrongly is a character whose perception is broken in a way nothing else surfaces. The route takes it too, optionally; a caller that gives none gets the world's percept of the room the body stands in (`Runtime::situation_of`), which is the only situation there is to be had without the character's own.

**The mood set here frames both conversations.** The active conversation's system prompt and the reflection's system prompt select the same mood from the mind's authored `moods/` collection. That is what makes the reflection sound like the character rather than like a narrator.

**Unbuilt, and for the same reason as everything else in §4.** A mood is selected out of a collection by the gather, and the reflection does not gather — `Persona` carries no mood field at all, so the reflection's frame has none. The `feeling` the route accepts is recorded and never reaches the model, which is why varying it across eight test inputs changed nothing about the output. It is one line once the reflection gathers, and impossible before.

### What the frame is missing, and why it shows

The same root cause costs the reflection three other things, and together they explain most of what is wrong with its output.

**The personality anchor did not arrive, and now does.** `Persona.personality` is a slug — *"never rendered; it is how a turn pins its personality's anchor in the projection"* — so the anchor is a projection collection member, selected per turn by `Installed::selection_for`. A reflection used to build its frame from `prompt::build_for` and select nothing, so the anchor was authored, installed, and invisible. It now opens on the mind's schema and pins the same identity members an acting turn does, and a daemon with no mind renders the anchor into the frame (`Persona::anchor`). What its absence cost: For a Maker that is the whole of who it is: *"You write a world, and you are not the only one… you have no world of your own."* Without it the frame opens `You are Tace.` and goes straight to `The world you live in:` — and the character, told it lives in Battle Cities and given nothing to contradict that, placed itself in a room on Level 3, handed itself wrenches and soldering irons, and invented a colleague called Kael.

That last part is §7's invented-colleague failure arriving one layer earlier than the design expected it — not in the dream, but in the reflection that seeds it.

**`relationships` and `beliefs` do not arrive either**, for the same reason, though they are listed as gathered in §4's table.

So what a reflection still does not read is everything gathered: `relationships`, `beliefs`, the mood. The briefs are well-formed without them, which is what makes the gap easy to miss.

**The Makers are a special case the design had not considered.** A Maker does not live in the world it is attached to — it writes it, from a vault outside that world's time, and what it files is what happens out there. `prompt.rs` renders the world under a hardcoded heading, `The world you live in:`, which is true for Kaelor and false for a Maker. The correction is authored in the personality anchor (`personalities/maker.yaml`), which is the right home for it because the relationship is a property of the personality rather than of the world — every character in `battle-cities` shares one `setting` and they do not share a relationship to it. That fix only lands once the anchor reaches the frame, which is the paragraph above. The heading itself stays wrong for a Maker until either the anchor overrides it in practice or the personality declares its relation and the prompt forks on it.

---

## 4. The reflection conversation

### One shot, and then gone

A reflection is a **brand-new conversation every time, used once, and never persisted.** It is marked with `mark_timeline_transient` before its first turn seals, which sets `no_cold_persist` on every residence it allocates, so it never reaches disk at all rather than being written and then deleted. Marking it *after* the first turn retracts nothing — by then the turn is already on the cold path.

It is **also tombstoned** when the questions are answered. The two do different jobs and neither substitutes for the other: transience keeps it off disk, and the tombstone is what keeps it out of every later gather and out of `find_conversations_by_metadata`, so nothing can resume or surface it while the process lives.

This is the property that makes the whole design safe, and it is worth being precise about why. A runaway needs iteration on a persistent state. Reflection *N* is not conditioned on reflection *N−1*, because reflection *N−1* no longer exists. There is no accumulating surface for surface form to rhyme with. The only history that crosses between reflections is the single committed line each one leaves in the active conversation, which competes there against everything the world is doing.

The consequence, stated plainly: **a character has no memory of having reflected.** It remembers what it concluded, and it has the dreams the reflections produced. The act of thinking leaves no trace. This is deliberate — dreams are the memory, reflecting is transient — but it means layer 14 (Self, in `theory_of_the_mind.md`) accumulates nothing, and a future design that wants a continuous self-model will have to add a fourth conversation rather than make this one persistent.

### No tools

The reflection conversation has none of the world's acts and no acting grammar.

**Built.** Under the mind's schema it opens on the same prompt a live conversation does, and the `tools` collection there is selected per turn by name (`tools::show`): the reflection's question shows `reflection`, every dream turn shows `dream`, and the loosed turn shows nothing — so `tools_overview`, which is gated on the list, goes with it. No world act is ever named on any of its turns. Every turn also selects the schema's `stance: reflecting` branch (`prompt::STANCE_SELECTOR`); left unselected it fell to `acting`, whose grounding forbids inventing a person in the conversation that writes a dream. The rendered frame below is what a daemon with no mind gets.

**Built**, as a fork in the frame rather than as optionality bolted onto it: `prompt::Stance` is `Acting` / `Reflecting` / `Dreaming`, and `prompt::build_for` renders one. `Stance::Reflecting` emits a frame that says the character has stopped — nothing it says is heard, nothing it says is an act — and lists only the two answers it is asked for, `reflection` and `dream`, with the call shape to write them in. Never a world act, because a reflection offered one will eventually call it, and an act emitted from a conversation the world is not reading is an act the character believes it performed and did not.

The one thing that *is* still a grammar is the think stencil — see §5.

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

**Dreams are scoped per conversation; the other layers are not yet.** A turn group can now be scoped after construction to the turns carrying one of a set of tags, and given its own depth (`Builder::set_group_tags`, `Builder::set_group_selection`); projection and the belief scan both apply the tags (`ContentResolver::turn_carries`), except on a projection's own target group, which is its own timeline already.

**Seen because it is ranked, not because of where it is declared.** A projection sees the layers beneath its target, and "beneath" was declaration order — so a layer appended to a live schema sat on top, invisible to every conversation, and appending is the only way to add one: declaration order also fixes every layer's and group's id, and those are persisted with every timeline. Layers now carry a `rank` (default: declaration index), and a projection sees every layer of lower rank. `dreams` is declared last and ranked `-1`, beneath `interaction`. Found live: dreams were kept and no turn was ever reminded of one. The dream group is closed when the schema is built and opened per conversation to one character's own tag — three lines deep for a turn in the room, eight for a reflection. `relationships` and `beliefs` are still not gathered into a reflection, and nothing hides mission and goals from it, because nothing gathers those either.

`sampled_axes` now comes from the corpus when the act reflects: a random eight of the character's own dream assumptions, read from each dream's conversation metadata (`dreams::sample_axes`). The route still takes them from its caller.

One thing follows that is worth stating. `depends_on*` in the schema takes a `CollectionId` in all four of its variants, so a section cannot be gated on the stance selector. `tools_overview` is gated on the `tools` collection instead — and a reflection turn shows its one answer through that collection, so it reads the acting overview beside the `reflecting` stance it selects. Gating a section on a selector's option is the missing piece, and until it exists that contradiction stands.

### A deep gather

The reflection runs with a **much larger top-K on the dream layer** than an action tick would. Reflection is not a different retrieval mechanism; it is the same mechanism run deeper. That has three consequences worth having:

- No new machinery. The budget is the dial.
- It degrades gracefully — a shallow budget is a shallow character, not a broken one.
- **Gather depth becomes a per-character trait.** A character that surfaces four dreams reflects on the obvious; one that surfaces forty pulls something from months ago. This is the same kind of authored dial as decomposition depth in Part VIII.

---

## 5. The exchange

**Built**: `npcd/src/engine/reflect.rs`, behind `POST /v1/npc/:nid/reflect`.

The reflection is not a pair of questions answered in prose. It is an **exchange of calls**, in the one direction the checkpoint is trained for:

```text
user   — prose. The caller states what the character has and asks for one thing.
assistant — a tool call, held to a stencil. The answer, in named fields.
user   — <tool_response>. Accepted, or refused with the reason, in words.
```

That direction matters and was got wrong twice before it was got right. Models are trained to **emit** `<tool_call>` and to **read** `<tool_response>`; `mind::compose` says so for the acting loop, where a result arrives in the *user* half of the next turn and the model reads that wrapper as *this is what came back*. A caller that emits tool calls is asking the character to answer its own question, and an assistant that emits tool responses is producing a shape it has only ever seen as input.

Everything the format used to be asked for in prose is now structure. Both fields of a dream are named parameters of a call the grammar forces, so a decode cannot fail to produce them — a missing brief, once the commonest fault and once enough to cost a whole run its dream, is unreachable. And a parameter span ends on its own terminator — a JSON string's closing quote, or an element's `</parameter>`, whichever call shape the dialect uses — so the model closes a field by writing it rather than by sampling EOS, and a span cut short by an intercepted EOS or its length limit has the close written by the tree. That is what finally stopped briefs dying mid-word at 29 tokens: every truncation measured was EOS winning inside a span whose only other exit was a token budget.

### What is asked for

Two entries, in `reflect::ASKED`, rendered into the frame with their parameters and descriptions:

| call | fields | what it is |
|---|---|---|
| `reflection` | `said` | one sentence, first person — the line that crosses back |
| `dream` | `assumption`, `brief` | the axis and the dream |

**Deliberately not in `tools::CATALOG`.** That catalog is what a character may *do*; every entry there reaches the world and is offered in the acting grammar. These reach nothing. Putting them there would offer a live character the chance to `dream` at somebody, and `act::parse` would begin accepting them on an acting turn.

`reflection` and not `think`: past tense, because the character is not being told to perform thinking, it is being asked what it *has* come to think. That is the register §6 requires.

### The order, and why the reflection is first

The reflection is asked **before** any dream exists. That is what keeps dream content out of the action stream — the only line the character ever sees is written when there is nothing yet to leak. The safety property comes from the sequence rather than from asking the model nicely not to repeat itself, which is the difference between a property and a hope.

Then the character is let go: one unstencilled turn where it thinks in its own voice with nothing holding the shape. That is not what crosses back — it is what the dream is written out of, and a dream seeded from a sentence is thinner than one seeded from a thought.

### The questions are still content

The prose asks are authored in the mind's `projection.yaml` under `reflection:` — `question_one`, `question_two`, `question_two_retry`, and the `repair` list — read once at load by `npcd::engine::schema::reflection` and sent verbatim. `reflect.rs` owns the machinery and no wording. `mind/layers/dreams/_dream-prompt-system.md` carries the reasoning behind each clause.

What has *left* those turns is the format. They no longer describe labels, because the grammar emits them.

The engine fills two slots in `question_two`: `[[domain]]` and `[[sampled_axes]]`. Double square brackets, because a single brace pair around a bare identifier is the schema builder's own template syntax — it resolves the whole file before parsing it, so a slot spelled with braces does not fail the reflection, it fails `projection.yaml` entirely and every character falls back to a schema that cannot gather, with one ERROR line at startup as the only symptom. The reader refuses a `question_two` missing either slot.

### Validation is in-band, and in words

A length outside its window is answered with a `<tool_response>` that says so and asks again — the wrapper the model already reads verdicts in, and the mechanism the acting frame already trains: *"An act can be refused, and a refusal says why. Do not call the same act again as though you had not been told."*

**In words, never in tokens.** A model asked to count its own tokens cannot; told *"that was 47 tokens, produce 200"* it pads rather than writes. The refusal says *"that is about thirty-five words and needs to be about two hundred"*.

Capped at two rounds. Uncapped refusal inside a blocking act is the one way this becomes genuinely expensive, and a third round is a decode that is not going to converge.

**Only genuine failures go through this path** — an empty field, or a length outside the window. Editorial guidance is a prose turn, because a `<tool_response>` that says *failed* about a call that succeeded teaches the model that success is arbitrary and blunts the signal for the case where it is real.

Q2 is written by a conversation whose context carries a *sample* of the character's existing dream axes, which is what makes the "not before" clause enforceable rather than aspirational — the incumbents are present as the thing to move away from, instead of being an abstract instruction the model is asked to honour. `theory_of_the_mind.md` §9 is pointed about that distinction: *"Every place the design has a 'don't do X' rule is a place under adversarial pressure it will eventually fail."*

A sample and never the whole corpus. Shown every axis it had used, the generator stopped inventing and recombined three of them; shown a random eight of the same set it found an axis nowhere on the list. The failure is anchoring rather than exhaustion, so it gets worse as the corpus grows.

### The behaviour space is rotated by the engine, not chosen by the model

`DOMAINS` in `reflect.rs` is eight cells — work, body, the building, time, the people you know, the things that are not you, being addressed at all, language — and the route advances through them. Asked to pick freely, the generator returned to what it already knew: every axis it produced unprompted was about the character's relationship to its own work. Rotating the domain is what a quality-diversity search does with a behaviour space — define the dimensions, then fill them.

### Structural checks and one retry

The engine does not judge a brief. It checks its shape: both fields present, an assumption that is one short line — one sentence, at most thirty words — and is not one of the sampled axes, a last sentence that is neither a question, a realisation, nor an explanation, a brief addressed to the dreamer as `you`, at least forty words, and a decode that reached an end rather than stopping mid-sentence. All but the fields and the sampled-axes check came from live runs — a whole dream written into the assumption field and then pinned as unchangeable by every repair pass after it, first-person drift carried forward from Q1's own voice in about one run in three, a decode that answers the *format* with one line under the label, a brief cut mid-word that passed every other check because it had no last sentence, and a conclusion reached by explaining rather than by any flagged word.

A failure gets exactly one retry, which restates the whole form rather than naming which check fired: told specifically what it got wrong, the checkpoint fixes that one thing and breaks another. Then the better of the two is kept, which is not the same as the second — a clean retry always wins, a faulty retry wins only when it produced a brief and the first attempt did not, and otherwise the first stands with the rejected retry and its fault both reported.

### The repair pass

**This section reverses a decision an earlier draft took deliberately.** That draft said *"one retry and no more: a third decode inside a blocking call costs the character more than a mediocre dream does."* The reasoning was sound and it was aimed at the wrong failure.

A retry answers a brief that came out **malformed**. It does nothing for a brief that is well-formed and is not a dream, and across 27 live samples that was the majority case: every structural check passing while the three rules in `mind/layers/dreams/_dream-prompt-system.md` did not. The dominant fault by a distance was the strangeness spreading past the one hole — Makers looping in time with the dreamer, hands that do not move, a colleague unimpeded by gravity — which is the ordinary day going strange all over, and reads as fantasy rather than as a dream. No cheap check can see it. So it is repaired rather than detected.

After the brief parses, the conversation runs a fixed review, authored as `reflection.repair` in `projection.yaml` and sent **unconditionally**. Each pass rewrites the whole brief and re-emits both fields, so every answer is parsed and checked exactly like the first, and a pass that does not parse — or that faults where its predecessor did not — is discarded with the previous brief standing. A repair that makes a brief worse is a real outcome, and taking it because it came last is how a review becomes a downgrade. Every pass is recorded either way, taken or not, because a response showing only the final brief cannot distinguish a review that improved it three times from one that was thrown away three times.

The order is load-bearing:

| pass | asks for | why here |
|---|---|---|
| 1 | edit it down to about 200 words, same events, nothing new | first, while the brief is longest and there is most to cut. A word count rather than "shorter", because told to shorten the generator trims adjectives and keeps every event — the opposite of what a dream wants |
| 2 | more imaginative — the same dream, one or two peculiar touches on the one wrong thing | a generator asked to be vivid and careful at once plays safe and writes neither. Aimed at the detail, because aimed at the strangeness it wrote a different dream each pass |
| 3 | every strange thing kept, everything else completely ordinary | the ordinary put back around the invention, after it — asked for both at once, the generator delivers neither |
| 4 | a last read-through: count the strange things, keep only the named assumption | **last**, so the final answer is the one holding the rule that breaks most often. Pushing after collapsing would undo the collapse |

The wording is the mind's — `reflection.repair` in `projection.yaml` — and this table follows it rather than the other way round.

Pass 4 also puts a drifted brief back on its own axis, which addresses the second-commonest measured failure: an `assumption` naming one thing and a `brief` dreaming another, 3 of 16 runs. That matters more than it looks — §8 makes the corpus a search over those axis strings, so a mismatched axis indexes a dream that was never had.

Measured on the call shape, one run: the dream and all three passes at 339, 328, 328, 328 tokens, three of three taken, no faults and no refusals needed. Two iterations earlier the same passes were collapsing to 29.

### `<think>` is closed by a stencil, not by a sampling budget

The reflection carries the think half of the acting grammar and nothing else — `compile_think_tree(ThinkMode::Off, …)`, which injects `</think>` as the next token after `<think>` so a reasoning block is unrepresentable rather than merely discouraged. Nothing else is masked, which is what a reflection needs and what an action loop would not give it.

Measured without it: Q2 closed its block anyway, because its prompt ends in a rigid output format that anchors it, while Q1 — an open question — ran the block to five thousand words of the model deliberating about which act to call, in the voice of a character it had invented. The prompt was correct throughout and nothing was masking the token. 52s down to 19s, and zero think tokens.

**The dream generator is unbounded, and its output reaches the visible surface.** Dreams enter the substrate and are selected by provenance into the active conversation, where they shape what a character does and says. That is the point of the mechanism and not an objection to it, but it is the only generator in the system that is both unconstrained and player-reachable, and it should be found written down here rather than discovered later.

---

## 6. What crosses back

The active conversation's `<tool_response>` carries **one committed line of intent**, in the first person, with no justification:

> I'm thinking of challenging him.

Not *"Challenge him"* — that is the imperative, and `theory_of_the_mind.md` §2 identifies it as the leaked form of goal-layer content, the clinically dangerous one precisely because it is partially aligned with what the character actually wants. And not a paragraph of reasoning: the reasoning belongs to the transient conversation, and a long dream-derived passage surfacing verbatim in a character's head reads as intrusion rather than inclination.

The register is the design. §6 again: *"Goals produce ongoing commitment pressure toward their objectives; in health this registers as motivation or inclination without surfacing as content."*

Two cheap invariants follow, and both are testable: the response is one sentence, and it contains no causal connective. A response that starts explaining itself is a regression, and nothing else would flag it.

**Both are built** — `reflect::reflection_fault`, reported on every reflection as `reflection_fault`.

**And a third, found live: the line is said as experience, never as fact.** Asked only for an inclination, reflections came back as assertions — *"I am the room itself"*, *"I am the vibration in the pipe"* — and a line that lands in the character's own context is read the next turn as something it holds true about itself. *"I feel like I am the room itself"* is an image the character had; *"I am the room itself"* is a claim it will act on. The line must carry an experiential frame — a feeling, a seeming, a wanting, a comparison held as one (`reflect::states_it_as_fact`) — and question one asks for exactly that. Like the other two it is refused in words and asked again, through the same refusal round (`MAX_REFUSALS`), and reported as `reflection_fault` if the last attempt still misses. Reported rather than retried: the line is already the character's, and asking a second time for the same sentence is how a reflection becomes a negotiation.

They needed building. Asked as an open question with no bound, Q1 returned four paragraphs and 268 tokens — not a line of intent, not what any other act returns, and at `context_window_turns: 32` enough to crowd the world out of the tail on its own. Q1's authored wording now asks for one sentence and names the connectives, and the exchange holds it there underneath the wording, because the wording alone did not: an answer over forty words, or one that runs to a second sentence or explains itself, is refused in words and asked again.

This line is a plan block in Part VIII's sense — *"a standing high-salience block… gathered into subsequent ticks so each tick-level action decodes against context that includes the current intent."* Its influence is earned by competing in the gather, not granted by being recent.

---

## 7. The dream layer

### One dream, one conversation

**Built** (`npcd/src/engine/dreams.rs`), as two conversations rather than one. The dream is decoded in a brand-new transient conversation seeded by the brief, under the schema's `dreaming` stance with the character's identity members pinned — who it is, its world, the building it stands in — and no acts shown; it is tombstoned after, like a reflection. The dream is then **kept** in a second conversation, on the `dreams` layer: one prefilled turn per line of the story (a line is a sentence), each with `Something you dreamt:` as its user half and its projection persisted as its signature, in order, so each line is signed with the ones before it in view. Two conversations because the one that decodes carries the brief as its prompt, and a brief in the corpus would be recalled as though it were the dream. `relationships` is not yet in the frame, so the invented-colleague risk below is still open.

The design as first written:

A dream is decoded in its own **brand-new conversation on the dream layer**, seeded by Q2's answer, framed on **characters, environment, and `relationships`** — no acts, no mission, no perception.

`relationships` is in that list for a reason found by testing rather than by design. Framed on a *description* of the cast — what a Maker is, what the machines are — but no roster of who actually exists, a decode invented a colleague: it named another Maker, gave her a errand and a way of carrying boxes, and wrote her into the dream. That is correct dream behaviour and the wrong thing for this engine, because a dream is real substrate content that provenance gathers back, so an invented person can surface later and be treated as somebody real. A character whose first principle is that an invented date is worse than an admitted gap should not acquire a colleague by dreaming one.

The `relationships` layer is already exactly this: per-entity calibration, `top_k 6`, maintained during play. Handing it to the dream conversation costs one layer in the scope list and gives the dream real people to reach for. The dream-label at surfacing stays as the second line of defence — but the first line is not needing it.

That minimal frame is not a cost saving. `theory_of_the_mind.md` §4:

> noise **constrained by the ground-truth stratum** and evaluated by the rest of the lattice. The noise proposes, the structure disposes.

Characters and environment *are* the ground-truth stratum. A dream framed on them is structured noise; a dream framed on nothing is word salad.

### Asynchronous, and the lag is load-bearing

The dream decodes on its own clock. The active conversation does not wait, is not notified, and does not know when it lands — the dream simply appears in the corpus and becomes selectable on some later gather.

**Do not optimise the latency away.** The delay is a design element: it means the dream surfaced at reflection *N* was seeded by reflection *N−k*, so feedback is delayed and content-addressed rather than immediate. Delayed feedback is much harder to drive into a fixed point than immediate feedback, and this is one of three independent reasons the loop does not close.

Dreams run on slack — `npc_mind_design.md` Part IX's daydream priority: *"the lowest-priority event on the queue, surfacing only when nothing else competes… shed first under load with no explicit load-shedding logic. Idle minds think; busy minds act."* A fixed interval would compete with action under load, which is backwards.

**At most one dream in flight per character.** Reflection can fire faster than a dream completes; without a cap a talkative character spawns dreams unboundedly. A second reflect while a dream is decoding simply does not spawn one. **Built** (`Runtime::dreaming`): the slot is taken the moment question one is answered and held through question two, the retry, the repair passes, the dream and its keeping. A reflect that lands while it is taken still gets its own reflection and its own answer; its conversation ends there, with no brief asked for and no dream. An earlier build held the slot for the whole reflection instead, and refused every reflect behind it — measured, six of ten reflects answered that nothing came, because a slot held for three to seven minutes sat behind a thirty-second cooldown.

**Not on slack yet.** The dream runs on its own thread the moment its brief exists, not as the lowest-priority work on the queue; under load it competes with action like any other decode.

### Private

Dreams are tied to the NPC that dreamt them and are never shared. Two characters who dream about the same event dream differently, and neither can retrieve the other's. **Built**: every dream turn is tagged `dreams:<npc id>`, and a conversation reads the dream group only through a scope naming its own character.

**Naming, built as §11 asks.** A dream's conversation is `npc-<id>-dream-<timeline>`, outside the `npc-<id>-day-` prefix every conversation open sweeps, so neither the midnight rollover nor `--forget-conversations` touches a character's dreams.

### Recalled on the normalized band

**Built.** A turn in the room can be reminded of up to three lines of its own dreams (`dreams::IN_ACTING`; eight in a reflection), and a line comes back only by winning the gather. Its score is normalized the way `repo_map`'s is (`docs/provenance_score_normalization.md`): each line's raw vote is divided by that line's own learned hit level — how strongly it matches itself — so a line recalled at ~1000 is one the moment matches as well as the line matches itself.

The levels are learned from the traffic that reads them. The group is tag-scoped to its owner and a turn teaches only the scopes its tags name, so every turn a character takes carries its own dreams tag: its seal teaches its own dreams' levels. A line that matches every turn — one line was recalled at over half a million, every round — ends up with a high level and is discounted; one that rarely resonates stands out when it does.

Self-match is the cold start: every dream at load (`Minds::warm_dreams`) and each new one as it is kept (`dreams::keep`). It starts low — the vote is a sum over a probe's strongest tokens, and a line probing itself is one sentence where a turn probes with its whole tail — so live traffic lifts it within a few turns. Before either, every line was divided by the cold-start prior, which multiplies the raw vote rather than normalizing it: recalled lines scored from about four thousand to over a million, the group's gate sat below all of them, and every turn took its top three whatever they were.

The gate is `repo_map`'s as it stands — `score_threshold: 1.0`, a belief band of 250, β 0.65 — and is **not yet tuned for dreams**. The recall log reports each recalled line's score on the normalized band, which is what a dream-specific value would be set from.

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

**Decode.** A reflect becomes **seven synchronous decodes** — Q1, the loosed turn after it, Q2, and the four-pass repair review in §5 — plus a retry on the runs where the brief came out malformed and a refusal round wherever an answer came back the wrong length, and then one asynchronous dream. It is currently the cheapest act in the game: a pause and three fields. It becomes by a distance the most expensive.

That figure was two when this section was written, before the loosed turn and the repair pass. The cooldown table's `reflect` (`cool: 30`) is the throttle for everything in this document and must be re-derived against seven rather than inherited from a design that costed two — it is the one number that decides whether a cast spends its day thinking.

Measured on Qwen3.5-9B with the two-decode shape, a reflection took 11–27s wall clock with a live cast ticking beside it. The loosed turn and the repair pass are five more decodes of the same order, so the honest estimate is on the order of half a minute per reflect, and the act blocks for all of it. §4's *"blocking costs nothing it had not already spent"* is an argument about a pause act, and it holds less comfortably at thirty seconds than at eleven.

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
| **per-conversation layer scoping** | **built for dreams** — tag-scoped turn groups and a per-conversation depth, §4 |
| **a toolless system-prompt frame** | **built** — the `stance` selector's `reflecting` and `dreaming` branches, §4 |
| **the reflect act answered asynchronously** | **built** — `Runtime::begin_reflection`, `Scheduler::hold`, §3 |
| **the dream conversation and the dream layer** | **built** — `engine::dreams`, a line to a turn, each signed, §7 |
| **`LayerHit` record type** | **new** — §8, five integration sites |
| **eviction at 500** | **new** — §8; nothing reaps the corpus yet |
| **authored seed dreams** | **new** — §9; `layers/dreams/maker/seed-dreams.md` is one file of 105 briefs and the per-entry ingest it asks for does not exist, so the layer declares no `ingest_unit` and is not read from disk |
| **the leakage detector** | **new** — §10, and should land first |

---

## 13. Decisions taken, and the one still open

**Taken.**

- Reflections are discarded. No Self continuity; dreams are the memory.
- Eviction is by recency of surfacing, not frequency.
- The generated cap is 500; authored dreams sit outside it.
- Dreams are private to one character.
- The reflect response is one committed line in the first person. Checked, not hoped for — §6.

**Reversed, with the reason recorded.**

- *"One retry and no more."* Now one retry **plus** an unconditional three-pass repair review — see §5. The retry was aimed at a malformed brief; the measured failure was a well-formed brief that was not a dream. The cost went from two synchronous decodes to five and §11 is rewritten against it.

**Open — defaulted here, flip it if the other reading was meant.**

**The reflection's stated feeling does not write the mood back.** `reflect` sets the mood; both conversations read it; whatever the reflection says it feels is *content*, not a control signal. The alternative closes a mood → reflection → mood loop with no world in it, which would be the tightest cycle in the design and the only one none of the three decorrelations in §7 apply to. Turning it on is a one-line change and should be a deliberate one.
