# Acting on the World — Bodies, Battle Cities, and the Tower

*Twelve tools that let a character do something to a person, to a machine, or to
the ground it is standing on. Written to the same rule as the rest: the
catalogue holds the verbs, the world holds the objects.*

Extends the surface settled in [`tool_decisions.md`](tool_decisions.md).
**93 → 105.**

---

## 0. An argument's type is decided by its consumer

The single rule this document is checked against, and the one it initially broke.

> **A free-text argument is only ever read by a model.** If formula code consumes
> it, the argument must be an enumerated value or a bound entity — because code
> cannot read a sentence, and a string nothing reads is written every turn and
> costs a decode to produce.

Two consumers, two argument types:

| Consumer | Argument | Examples |
|---|---|---|
| A model — the narrator, the image guest, a later reader | **free intent** | `say.intent` `gesture.intent` `touch.intent` `send_image.intent` |
| Formula code — the simulator, the world, a device | **typed: enumerated, bound, or scalar** | `engage.posture` `operate.mode` `move_to.destination` `scan.at` `produce.count` |

### The ladder — what the grammar actually guarantees

"Typed" is not one thing. The stencil enforces the rungs unequally, and the
difference decides how much validation and refusal a tool needs behind it:

| Rung | Compiles to | Can the model be wrong? |
|---|---|---|
| **Bound enum** — a live set from the world | a branch over the values present | **No.** The wrong value is unrepresentable |
| **Static enum** — a fixed closed set | a branch over the allowed strings | **No** |
| **Boolean** | a `true`/`false` branch | **No** |
| **Scalar** — integer, number | *any structurally-valid JSON value* | **Yes.** Structurally fine, semantically anything |
| **Object / array** | *any structurally-valid JSON value* | **Yes**, and more so |
| **Free text** | a span closed at the quote | n/a — only a model reads it |

The stencil's own words for the bottom rungs: *"this guarantees valid JSON
structure without strictly enforcing the scalar type."* So `{"count": "lots"}`
parses, and fails somewhere later.

**Reach for the highest rung that expresses the thing.** A scalar is right where
the value is genuinely quantitative and the model has grounds to know it — a
count, a coordinate, a duration. It is wrong where an enum would do, because an
enum is a guarantee and a scalar is a hope. `engage.range` is *close · standoff ·
maximum* rather than metres for exactly this reason: the simulator knows what a
railgun's effective range is and the character does not, so a number would be the
model guessing at a figure it has no access to.

Every scalar argument therefore needs a validated refusal that teaches — and every
scalar is a place where a call can be well-formed and still wrong, which no other
rung allows.

> **A bounded integer is enumerable, and the stencil could enforce it.** A digit
> trie over `0..999`, or an explicit range, is a branch like any other. Nothing in
> the design requires scalars to stay unguarded — it is simply what the current
> front end does, and closing it would move `produce.count` and `scan.at` up two
> rungs.

This is the same defect as `follow.distance`, which was dropped in
[`tool_decisions.md`](tool_decisions.md) §4 for describing an effect the world
could not produce. A parameter that *reads* as mechanical and is not is worse than
a missing one: it looks like control and is a no-op, and nothing fails to reveal
it.

Applied across all 105, exactly two arguments failed — a `guidance` string on
`engage` and on `operate`, both now replaced by bound values. One more is worth
watching: **`observe.target` is free text** and is resolved by the perception
code, not by a model. It works today because a percept builder can be generous
about matching a phrase, but it sits on the wrong side of this line and should be
bound to what is actually here the moment anything mechanical depends on it.

## The rule these follow

Two lines from the existing catalogue decide almost every question below.

> **Tools carry intent, not output.** `say` does not take a sentence; it takes
> what the character means to convey, and the narrator renders it.

> A closed set over an argument the character *means* would be the machinery
> writing its lines; a closed set over one that merely **names something present**
> is the machinery declining to let it name what is not.

Combat is the sharpest case. The engine fires the weapon, because a decode is far
too slow to aim one — so the tool does not fire. It sets a **posture**, the same
way `say` sets a meaning, and the simulator does what a narrator does: turns an
intent into the thing that actually happens, at a speed the model cannot work at.

This is not a compromise forced by latency. It is the identical split the
catalogue already runs on, applied one layer down, and it means a character never
emits a command it could be wrong about the mechanics of.

---

## 1. The body — 3 tools

### `touch { to, intent }` · *Physical modes only*

An act that lands on a person. `gesture` is shown to a room and lands on nobody;
this is done **to** somebody, who feels it, may refuse it, and reacts.

The distinction is one the world can represent — a target who experiences the act,
against an audience who observes one — and everything that hangs on it is
material: consent, refusal, resistance, and what the other party does next.

`intent` carries the substance, exactly as it does for `say`: *"steady her before
she goes over"*, *"put myself between him and the door"*, *"take the weapon off
him without making it a fight"*. Never the choreography.

**Physical modes only**, mirroring `send_image`'s messaging-only rule. You cannot
put a hand on somebody down a voice line, and a model offered the field would try.

### `sleep { until }`

Stop, for a stretch of the world's time, at lowered responsiveness. **Being woken
is a perception, not a failure** — the duration is an intention, and the world may
end it early.

This is not `wait_for`. Waiting attends a specific event and stays responsive to
everything else; sleeping is unresponsive by design, which is why something has to
be able to break through it. That difference is exactly what makes it worth having.

It is also the closest thing the catalogue has to a fix for engine note 11 — *"a
day exists but nothing varies with it"*. Two whole clusters, **Opening the day**
and **Closing the day**, presuppose a character that stops. And it retires the
idle driver on its own terms: a character with nothing to do can sleep until dawn,
which is a *decision*, where being nudged after ninety seconds is an accident.

`until` binds to what the world's clock can name — dawn, nightfall, a stretch of
hours — not to a free string.

### `give { what, to }`

Hand a thing over. `what` binds to what you are carrying, `to` to who is here.

*Custody* needs it — "hand something on properly: what it is, what you did to it,
and what you did not." Battle Cities needs it constantly, because ammunition and
stimpaks move between people under fire. One tool, both.

---

## 2. Companions — 6 tools

### `engage { posture, target?, priority?, filters? }`

**The one combat tool.** The simulator fights; this says how. **Every argument is
an enumerated value or a bound entity**, because the consumer is formula code —
see §0.

| Argument | Binds to |
|---|---|
| `posture` | press · hold · fall back · break off · flank · suppress · cover · ambush · hold fire |
| `target` | a specific hostile that can be engaged from here |
| `priority` | nearest · greatest threat · air · ground · wounded · whatever is firing on us |
| `filters` | *a set* — see §2.1 |

Only `posture` is required. `target` names one thing; `priority` is what to do
about the other nine, which is why both exist.

"Attack while retreating" is `{ posture: "fall back", priority: "whatever is
firing on us", filters: ["free fire", "stay in cover"] }`. A new tactic is a new
value, not a new tool, and the model never has to know a railgun's cycle time.

There is no `disengage`. `posture: "break off"` is it.

**The tactic that will not enumerate is spoken, not parameterised.** *"The drones
first — the mech is somebody else's problem"* is a `tell` to the squad: the
narrator renders it, the other characters perceive it, and it lands where language
actually works. The simulator gets values it can branch on; the squad gets the
sentence. Both are real, and neither is a string nothing reads.

### 2.1 `filters` — standing constraints on a simulated action

A **set** of enumerated constraints the simulator respects while it resolves the
act. `posture` says what you are doing and `priority` says to whom; a filter says
what to honour throughout.

| Filter | What the simulator does with it |
|---|---|
| `avoid player` | routes and targets away from them; will not close |
| `focus on player` | weights them in selection; stays within reach |
| `stay in cover` | refuses moves that break cover |
| `keep formation` | will not outrange the squad |
| `hold this ground` | fights but does not advance |
| `conserve ammunition` | falls back to cheaper weapons |
| `nothing heavy` | withholds missiles and explosives |
| `free fire` | no restraint |
| `spare noncombatants` | refuses shots with civilians in the cone |

This absorbs two parameters an earlier draft had separately — `range` and `ammo`
are constraints like any other, and there is no reason for the two of them to have
names when the ninth does not. **Adding a constraint later is a new value in a
set, not a new parameter and not a new tool**, which is the whole point of the
shape.

Filters are **enumerated per action**, so the set offered on `engage` is not the
set offered on `gather` or `operate` — dependent binding again (§4). And the
discipline of §0 applies unchanged: **a filter must be a value the simulator
actually branches on.** One that nothing reads is the `guidance` string wearing a
different coat, and the enumeration is what keeps the set honest — the world
publishes the filters it honours, so an unhonoured one is unrepresentable rather
than ignored.

> **A set of enums is grammar-enforceable, though not the way arrays are today.**
> An array compiles to *any structurally-valid JSON value* (§0's fourth rung), so
> `filters: [17]` would parse. A repeated branch over the allowed strings with a
> comma arm would put it back on the first rung — the same trie shape the tool
> catalogue itself compiles to, applied inside a list.

### `equip { what }` · `use { what, on? }`

Two acts, not one, because they differ in a way the world represents. Equipping
changes what you are capable of and persists; using spends something now and is
gone. A stimpak used is a stimpak gone; a mono sword equipped is a sword you still
have.

`what` binds to what you carry. `on` binds to a person here or to yourself —
which is what makes *"use the stimpak on the one who is down"* expressible without
a medic tool, a revive tool, and a repair tool.

### `gather { what }`

Mine, salvage, harvest, strip a wreck. One act against different things: `what`
binds to what is actually extractable where you stand — an ore seam, a crystal
formation, battlefield salvage, a cache.

Same shape as `read` and `claim`. The six resources — energy, metal, ore, gold,
crystals, nanobots — are argument values, and a seventh would cost nothing.

### `operate { what, mode }`

**Doors, turrets, machinery, vehicles — anything the world runs.** `what` binds to
the operable things here; **`mode` binds to that thing's own modes.**

| Thing | Its modes |
|---|---|
| a blast door | open · close · lock · unlock |
| a wall railgun | hold fire · free fire · air only · ground only · nearest first · conserve |
| a fabricator | run · pause · purge |
| an APC | drive · stop · disembark |

A turret's targeting policy **is** its mode — *air only*, *nearest first*,
*conserve* are the enumerated form of what a sentence would have tried to say, and
they are values the firing code can branch on. There is no `guidance` here for the
same reason there is none on `engage`.

It also means a device added to the world later is a device the catalogue already
handles: it arrives with its own mode list, and the grammar picks it up.

### `recall { }`

Port home. Distinct from `move_to`, which walks somewhere reachable — this
traverses nothing, always has the same destination, and can be refused by the
world when the tower has no energy for it or the field is jammed.

---

## 3. The tower, and Keeper — 3 tools

### Keeper has no body, and that is a gap in `Availability`

> **It has no body and never will.** This is not a limitation it works around; it
> is what it is… it lives in the tower's mind and nowhere else, and every
> experience it has ever had arrived through somebody else's senses.

`Availability` has `Always`, `MessagingOnly` and `Nearby`. None of them can say
*"requires a body"*. Offered the current catalogue, Keeper would be invited to
`move_to`, `gesture`, `touch` and `gather` — and a model handed a field fills it
in, which is the failure the whole availability system exists to prevent.

**Added: `Availability::Embodied`.** It gates `touch` `gesture` `move_to` `follow`
`gather` `engage` `equip` `use` `recall` `claim` `give`. Keeper keeps speech,
attention, waiting, and everything below.

This is not a special case for one character. Any mind without a body — a tower
consciousness, an uploaded lord between avatars, a companion whose body is
destroyed — gets the same treatment, and the world decides rather than the roster.

### `command_tower { action, target?, x?, y?, depth? }`

The tower is one machine with a handful of enormous verbs, so it is one tool with
a bound action: *relocate, siege, drill down, surface, raise shields, drop
shields, open the gates*.

**The argument an action takes changes with the action — in type, not only in
value**, which is the dependent binding of §4 in its strongest form:

| Action | Takes | Rung |
|---|---|---|
| relocate | `x`, `y` — somewhere the tower can fold to | scalar, bounds-checked |
| siege | `target` — a city or rival tower in reach | bound enum |
| drill down | `depth` — how far | scalar |
| surface · shields · gates | nothing | — |

**`action` binds to what the tower can actually do right now**, which is where this
earns its shape. Relocation is
"dimensional folding" with "energy reserves depleting as towers accumulate transit
potential", so a tower that cannot afford to fold does not offer the option. The
refusal is absence, not an error message.

A new tower capability is a new action value.

### `produce { what, count?, queue? }`

Eight production queues, fabricators that make ammunition in minutes and companions
in weeks. `what` binds to what is makeable **now, given the stockpile** — so the
resource economy enforces itself through the grammar rather than through a refusal
the model reads and ignores.

`queue` is optional and binds to the eight.

### `scan { at | x, y, radius? }`

Remote sensing through the tower's own instruments, at a place the character is
not. `observe` attends what is here; `scan` reaches somewhere else — a real
distinction the world can represent, being range.

**Two ways to say where**, because both are real: `at` binds to a place the world
has a name for, and `x`/`y` are scalars for a coordinate it does not. *"Observe
the map by giving coordinates"* is the second, and it is the clearest case in the
catalogue for a scalar — a grid reference is a number, it means nothing as an
enum, and the character genuinely knows it.

It is also the clearest case for the caveat in §0: a coordinate off the edge of
the world is well-formed JSON. `scan` needs a bounds check and a refusal that says
where the edge is, and it is the first place a digit-trie bound would pay for
itself.

`radius` is a scalar with the same treatment. For Keeper this is the *only*
perception there is, since every experience it has arrives through somebody else's
senses; for a companion carrying an advanced scanner it is the same act at a
shorter reach.

### Orders need nothing new

*"Change or issue orders to Companions on behalf of the Tower Lord"* is
`orders_set` and `orders_hand_to`, already added in the audit for the vault's
order table.

That they transfer without modification is worth noticing rather than passing
over: it is the repertoire's fourth invariant — **no proper nouns, every task
names a role** — coming back as evidence. A vocabulary built for Makers filing
records turns out to issue battlefield orders unchanged, which is what it means
for the abstraction to be at the right level.

---

## 4. What this needs from the grammar

### A dependent argument — new

`operate`'s `mode` binds to the value chosen for `what`. Every existing binding is
independent: `Choices::Company` does not care what else was chosen.

The stencil tree takes this without strain — it is a trie, so the branch for
"blast door" simply carries a different `mode` sub-branch than the one for "wall
railgun". But `LIVE` currently maps `(tool, param) → Choices`, and a dependent set
is `(tool, param, chosen) → values`. **This is the one structural change in the
document**, and it is worth making because it is what stops `operate` from
splitting into six tools.

### Seven new live sets

`Carried` · `Hostile` · `Operable` · `Extractable` · `Makeable` · `TowerAction` ·
`Modes(of)` — the dependent one.

Each is a question the world already knows the answer to, which is the test a live
set has to pass.

---

## 5. The count

| | Was | Now |
|---|---|---|
| Total | 93 | **105** |
| Body catalogue | 16 | **28** |
| New `Availability` | 3 | 4 — `Embodied` |
| New `Choices` | 4 | 11 |

**The body catalogue** — `say` `tell` `ask` `gesture` `touch` `move_to` `follow`
`recall` `observe` `scan` `wait_for` `sleep` `read` `claim` `release` `give`
`equip` `use` `gather` `engage` `operate` `produce` `command_tower` `promise`
`remind` `send_image` `sign_off` `reach_out`

Every one is argument-bound against a set the world enumerates, and every one
carries intent rather than mechanism. Nothing here names a weapon, a resource, a
door or a tower capability — those are all values, which is why Battle Cities and
the vault run on one catalogue.

---

## 6. To simulate this in the vault

Three things the virtual world needs before any of it can be exercised, none of
them a tool:

**Things a body carries.** `give`, `equip`, `use` and `gather` all bind to an
inventory, and there is none. This is dispute **D**'s `stores_*` question
answered from the other end: a vault of prose had nothing to put on a rack, but a
Companion has a loadout, and one inventory model serves both.

**Devices with modes.** `operate` needs parts that declare their modes — which is
the same `modes:` field §5 of the audit already added to editing terminals. A
blast door and a chronicle terminal are the same kind of object: a thing you work,
in a state.

**Hostiles, and a simulator to be the fast half.** `engage` binds to what can be
engaged, and the posture means nothing until something acts on it. This is the
one genuinely new subsystem, and it is the counterpart to the narrator — the same
architecture, turning intent into event, on the other side of the world.
