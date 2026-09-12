# Tools Against the Interaction System

*The vault's tool list assumes every interlocutor is another Maker. A user is not
a Maker. This reconciles the 129 against the four interaction modes, the narrator,
and the image paths — and marks what is missing, what is added, and what should
not survive review.*

Third of three: [`tool_surface_audit.md`](tool_surface_audit.md) settled what the
tools are, [`tool_world_state.md`](tool_world_state.md) settled what they touch,
and this settles whether they are the right tools once a human is in the room.

Everything here is a specification question. Implementation is deliberately out of
scope — the point is to get the vocabulary right *before* the tests are written
against it, because a test written against the wrong tool is worse than no test.

---

## 1. What the interaction system actually is

`POST /v1/npc/{id}/interaction` opens a forked substrate against the `interaction`
layer. **Mode is immutable for its life** — a mode change is `409` and the client
opens a new one, because "hanging up and walking over really is a different
encounter."

| Mode | Observable | Extra tools | Idle |
|---|---|---|---|
| `physical` | speech, movement, gesture, expression, ambient acts | — | 5 min |
| `video_call` | speech, expression, framed gesture | `send_image` | 10 min |
| `voice_call` | speech, audible action | **none** | 10 min |
| `instant_message` | speech only, text-shaped | `send_image` | 24 h |

Three properties matter for the tool list:

**Mode gates the catalogue, not just the view.** This is the same discipline the
body catalogue already keeps — absent rather than present-and-refused.

**The NPC emits intent; the narrator writes the words.** `NarratorInput` is
`Say | Act | Scene | Cue | Beat`, and those are also the turn kinds the console
renders. One vocabulary, both directions.

**A slash command is a tool schema.** `GET /v1/commands` returns the same
schemars-generated shape `ToolInfo.parameters` carries, and commands emit a
`perception`, an `interaction_event`, or an `environment_event`. So the user's act
vocabulary and the NPC's act vocabulary are the same kind of object, and can be
checked against each other.

**There are three image paths, not one:**

| Path | Driven by | Produces | Exists |
|---|---|---|---|
| Scene imagery | the narrator, from the interlocutor's vantage | an illustration of the moment | specified (§34) |
| `send_image` | the NPC, in messaging modes | a picture sent to a person | specified (§17) |
| `portrait_draw` | a Maker at an easel | a durable artifact | **no store** |

---

## 2. Missing — the interlocutor can be a person, and nothing accounts for it

### 2.1 An NPC cannot end a conversation

Every interaction ends by client `DELETE` or by idle timeout. **No tool
disengages.**

In physical mode `move_to` walks away, which is honest. In the three remote modes
there is nothing: an NPC being talked at over instant message has no act available
that stops it, for up to 24 hours.

Two clusters need this and neither is decorative. *Boundaries* — "say no to a
request from somebody you like." *What an encounter leaves* — "**Leave a
conversation that is draining you before you say something that makes it worse,
and say why you are going.**" That task is unexecutable as written.

**Added: `sign_off`** — messaging modes only, symmetric with `send_image`. Takes
an intent, because leaving is a thing you mean, and the narrator renders the
parting. The interaction archives exactly as a `DELETE` would.

### 2.2 An NPC cannot start a conversation

Interactions are opened by a client, always. An NPC has no way to reach somebody
who is not standing next to it.

Between Makers this is invisible — they are co-present, so `move_to` and `say`
suffice. With a player it is a hole in the product: an NPC that can only ever
answer is *"a presence that exists only when addressed"*, which §17 names as the
exact failure the observability rules exist to prevent.

Four clusters depend on it. *Long estrangement* — "speak first to somebody you
have not properly spoken to in a long time." *Repair* — "admit you were wrong, to
the person you were wrong at." *Promises* — `remind` presupposes reaching
somebody. *Absent third parties*.

**Added: `reach_out`** — messaging modes only, because you cannot physically
appear beside somebody. Opens an interaction the way a client would. This is the
NPC texting you first, which is the single most characterful thing a messaging NPC
can do and is currently impossible.

### 2.3 There is no gathering, because an interaction has one interlocutor

`interlocutor` is a single object. A reading at the long table, a settlement
between two holders at the concordance table, and the whole *Gatherings* cluster
all need more than two parties.

This is not a missing tool — it is a missing shape in the interaction model, and
it is what `gather_*` and the four settling tables were quietly assuming. Flagged
here rather than in §4 because the tools are right and the substrate under them is
not.

---

## 3. Two defects in what exists

### 3.1 `Mode` has two values; the contract has four

`npcd::engine::tools::Mode` is `Physical | Messaging`, with `Messaging` documented
as *"text, voice, letters — anything where the parties are not co-present."*

The contract gives `send_image` to `video_call` and `instant_message` and
**withholds it from `voice_call`**. The two-value collapse hands it to voice
calls, so an NPC on a phone call is offered a way to text a photo down it.

It is the exact failure `Availability` exists to prevent — a model handed a field
fills it in. **`Mode` must carry the contract's four values**, whatever the
availability rule then reduces them to.

### 3.2 `send_image` binds its target to the wrong set

`LIVE` binds `("send_image", "to", Choices::Company)`, and `Within::company` is
*who is standing in the room*.

In a messaging interaction the NPC is somewhere in the vault with other Makers
around it, texting a player who is nowhere near. The grammar would offer it the
names of the Makers beside it and refuse the one name that is valid — while §17
says the target "is required and validated against the interaction's
interlocutor."

The binding is a different live set: **the parties to this interaction**, not the
bodies in this room. The same set `sign_off` and `reach_out` need. There is no
`Choices` variant for it.

**Added: `Choices::Interlocutor`.** Not a tool, but it is the argument binding
three tools depend on, and its absence is why this went unnoticed.

---

## 4. Under review — poor fit or no real impact

### 4.1 `character_write_beliefs` — normatively forbidden

Dispute **E** in the state document is resolved, and against it. §16 is categorical:

> The engine enforces this at the registry: a tool declaring `beliefs` in
> `writes_layers` is **rejected at registration** with `tool_writes_protected_layer`,
> and the generic catalog contains none. An attempt to reach these endpoints from a
> tool context returns **422 `action_plane_belief_write`**.

The rule is about *tools*, not about whose beliefs. A Maker at a character
terminal acts through the action plane, so the tool cannot exist.

**This is better design than the tool it removes.** A Maker writes what *happened*
— `character_write_memories` — and the evidence process on the sleep clock earns
the belief from it. A tool that grants a belief directly bypasses the one process
that makes a belief mean anything, and the vault would be manufacturing conviction
without evidence.

**Removed.** The character terminal keeps `read`, `write_identity`, `write_wants`,
`write_memories`. Relationships carry no such restriction on either plane, so
`character_settle_relation` stands.

### 4.2 Twenty-five `*_read_*` tools against one `observe`

`observe` already absorbed `listen` and `inspect`, on a stated principle:

> Acts merge when the world cannot represent the distinction. What matters is that
> a step spent looking comes back with something the character did not have.

There are now **25 reading tools** across the namespaces — `chronicle_read_era`,
`story_read_filed`, `record_read_description`, `plan_read`, `standard_read`,
`portrait_read_hung`, and twenty more. Each is "look at the thing this station is
for."

The case for keeping them: they return *structured* state, not a narrated percept.
`plan_read` returns a strategy tree; `observe` returns prose.

The case against: at a terminal, reading is what a terminal does, and the terminal
already knows what it holds. **One `read` per part** — parameterised by what is
being read — would collapse 25 tools to roughly 15 and remove the judgement call
of *which* read to call, which is a decision the model should not have to make
when it is standing at exactly one thing.

Worth deciding before the tests are written, because this is 25 test files or 15.

### 4.3 `follow`'s `distance` has no world to be true in

`distance: close | at a distance | out of sight` — but `Where` is a node, and the
world models no distance within one. Two bodies in a room are in the room. The
parameter is a free string that changes nothing, and its own example leans on it
hard: *"Following close would satisfy the verb and fail the order."*

Either the world grows sub-node proximity, or `distance` is narration and should
be folded into an intent rather than sitting as a parameter that reads as
mechanical and is not.

### 4.4 `room_sit` claims nothing

`seat` is `kind: seat`, not `station`, and carries no `binds`. `World::take`
claims a station's subject. A seat has no subject, so sitting produces a
`TookStation { subject: None }` event — rendered as *"sat down to work"* — and
changes nothing else.

That is not nothing: being seated is observable and *Idle company* wants it. But
it should be checked that a seat is takeable at all, and that sixteen Makers and
ten seats interact the way the capacity invariants assume.

### 4.5 Two patience mechanisms that have never been compared

`wait_for` carries its own patience; interactions carry a per-mode idle timeout of
5 min / 10 min / 24 h. Nothing reconciles them. An NPC that `wait_for`s inside an
instant-message interaction is waiting under two clocks with different opinions.

### 4.6 Two naming schemes for one person

`tell` and `ask` bind `to` against `Within::company` — the names the world writes
down. An interaction's interlocutor carries a `display` name, and §12 makes the
*unique name* the address. A player called "Ilse" in an interaction and a body in
a room are addressed differently, and the tool list does not say which wins.

### 4.7 `story_read_aloud` / `hear_draft` / `give_opinion`

Dispute **G** stands and §2.3 sharpens it. A reading is a multi-party interaction,
and multi-party interactions do not exist. Until they do, these three are `say`,
perception, and `tell` with a draft argument — the `room.talk` defect one level
subtler.

They earn their place only if a reading carries real state: listeners, and verdicts
recorded against the draft. That is what *Testing it* and *Review & sign-off* need
in order to produce `verdict` at all, so the answer is probably to build the shape
rather than drop the tools — but not to write tests against them first.

---

## 5. Two disputes the interaction system resolves

Both were filed under **D — no store, and the store is a design decision**. The
store exists; it is the interaction system.

**`enquiry_*` (5 tools) — the outside is the player.** *Service* wanted questions
arriving from beyond the vault, and engine note 22 said nothing produces them. An
interaction opened by `{ kind: "player" }` **is** a question from outside, phrased
in somebody else's words, which is precisely what `enquiry_take_question` describes.
The enquiry desk binds an open interaction. No invented queue.

**`creator_present` — the counterpart is the interlocutor.** Engine note 21 said
nothing defines what happens when there is nobody to present to. The answer is
that presenting requires an open interaction, and without one the tool is absent
rather than refused — the same rule as `tell` when alone.

This makes *Service* the one cluster that touches the product's real users, and it
is the strongest available answer to the pressure-toward-the-untried problem the
hierarchy design left open: **a real person asking a real question is a source of
novelty no retrieval policy can manufacture.**

---

## 6. The reconciled count

| | Before | After |
|---|---|---|
| Declared | 129 | **130** |
| Added | — | `sign_off`, `reach_out` |
| Removed | — | `character_write_beliefs` |
| Disputes resolved | — | **E** (against), **D**/enquiry (5), **D**/creator (1) |
| Defects to fix before tests | — | `Mode` 2→4 values; `send_image` target binding |
| Under review | — | 25 reading tools; `follow.distance`; `room_sit`; two clocks; two naming schemes; 3 reading-aloud tools |

New argument binding: **`Choices::Interlocutor`** — the parties to this
interaction, as against the bodies in this room.

**What to settle before writing tests**, in order, because each changes what the
tests assert rather than merely how many there are:

1. `Mode`'s value set — it changes every availability assertion.
2. One `read` per part, or 25 named reads — it changes the file count by ten.
3. Whether a reading is multi-party state — it decides three tools and the whole
   `verdict` currency.
4. The address of a person — world name or unique name.
