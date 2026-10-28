# The Household Vault — a parallel world that mirrors a life

**Status: DRAFT / design only.** Nothing here is built. The parts that need an
engine change are named as such, and the parts that are pure authoring are named
as that, because the two are not the same kind of work and only one of them is
safe to do on a Tuesday.

> A vault beside the Creator's Vault, in the same universe, with the same rules —
> but this one is not a metaphor for a fictional world. It is a world whose
> places are the places a real family actually is, whose chronicle is a real
> family's history, and whose one mind at the top has exactly one mission: keep
> them safe, and help where it can.

---

## 1. Why a vault, and not a smart-home assistant

The honest answer is that the second thing already exists, in worse forms, and
none of them have a building in them. The failure mode of every assistant that
knows your life is that it knows your life as a flat list of facts that drift.
It knows you have a daughter and not that she started school in September. It
knows you are at work and not that this is the week you are at work all week.
It knows both and cannot tell you which, because the two facts live in different
systems and nothing stands between them and says *these two disagree, and that
means something*.

The vault is the thing that stands between them. The argument for the vault is
the one already made in `npc-map/src/schema.rs`: **a thing described at three
magnifications must never disagree, and the only way to guarantee that is to
generate all three from one source.** Here the three magnifications are the
whole world, the level you are on, and the place you are standing — and the
thing that makes this vault different from `creators-vault` is that the
magnifications are not literary, they are *the actual scale of a day*.

The second argument is the one the vault already makes about arithmetic: a
million places cannot be hand-written and none of them can afford inference at
the moment somebody walks in. A family's life is smaller than that, but it has
the same shape — a great many small places, mostly uninteresting, all of them
needing to be there before the question is asked, and none of them able to
 afford a model call at the moment of the question.

The third argument is the lift. A flat list of facts has no lift, and so has no
**cost of going**. Half of what makes a place useful is that reaching it costs
something, and the cost is what makes the arrival a decision. The rest of this
document is mostly about what costs what.

---

## 2. The shape

```
household-world                    the world. one per family.
│
├── household-vault                the building. it moves. it is the family.
│   │
│   ├── L0  threshold              the door. the mail. who is here.
│   ├── L1  dwelling               the house, floor by floor
│   ├── L2  road                   the car, the journeys, the commute
│   ├── L3  vocation               the workplace
│   ├── L4  schoolroom             the children, the gate, the school
│   ├── L5  ancestors              both parents' houses, as one level
│   ├── L6  market                 errands, purchases, money
│   ├── L7  infirmary              health, appointments, medicine
│   ├── L8  chronicle              the family record — mirrors the vault
│   ├── L9  watch                  the zenlings muster here and report
│   └── L10 summit                 Zen
│
└── (the waste)                    everything not in the family's keeping
```

A **level is a major location** — one of the places the family is *between*,
not one of the rooms inside a place. A **node is a room**, and a **part is a
thing standing in it**. A **terminal** is a part that carries tools, and the
tools are only reachable from the room the terminal is in, which is
`MapSet::tools_at(node)` doing exactly what the existing schema says it does:

> A tool reachable at this node is a tool one of these parts carries, so what an
> NPC can do is computed from where its body is rather than declared about who it
> is.

Nothing new is needed for that. It is the whole reason the vault works as a
safety model: **a zenling that is not standing at the door terminal cannot open
the door.** Not because it is forbidden. Because it is not offered, and a tool
that is not offered is not reasoned about.

The **lift** is the core node, and it is the only way between levels. The
**summit** is the top of the lift shaft, and it is where Zen is.

---

## 3. What is real, and what is a place

This is the load-bearing section, and the one the existing schema cannot
express, so it must be said plainly before anything is written.

`npc-map` has **no geometry, on purpose**. The reasoning is in the crate header
and it is good reasoning: coordinates exist to serve a picture, and a picture
that every generated place must stay consistent with is a constraint carried by
millions of places to serve something nobody looks at. That reasoning holds for
the map. It does not survive contact with a vault whose rooms are real places.

So the split is this, and it is the whole design:

> **The map is the world. The anchor is the fact. The anchor is never in the
> corpus.**

- A **level** is authored in `npc-map/maps/` as ordinary map data: `id`, `kind`,
  `name`, `summary`, `character`, `habit`, `parts`, `off`, `spine`. It says
  *the dwelling has a kitchen, and the kitchen is where the day starts*. It
  contains **no address, no coordinate, no time, no name of a real street**.
- An **anchor** is a private record outside the map: `anchors/<level-id>.yaml`
  in the `--data` tree, **never** in `--mind`, **never** ingested, **never**
  reachable by retrieval, **never** quoted into prose. It binds the level to a
  place in the real world and it is the only file on the system that does.
- The vault's **position** is a fact about the world, not about the map — see
  §4. The map never changes because the family moved; the anchor does.

The reason for the hard split is the one `npcd_worlds_and_layers.md` §6 makes
about tags, turned around: a canon file whose tag cannot be derived must be a
**hard error**, not a default to untagged, because the failure would be a
document quietly readable from every world. An anchor ingested by accident is
that failure with a worse object: a real address, in the corpus, retrievable by
a battle-cities NPC, forever, in a substrate that is persisted and replicated.

So the rules are:

1. **An anchor file is never read by `load_dir`.** It is not YAML-shaped as an
   area and it is not in `maps/`. The loader must not be able to find it.
2. **An anchor id is an opaque token.** The map says `the dwelling`; the engine
   says `hh-dwelling`; nothing in the prose says where the dwelling is.
3. **A prose field that names an anchor is a validation error.** `validate.rs`
   gains a check: no authored string in any map file may contain a token that
   appears in an anchor file. This is cheap, and it is the only check that
   catches the mistake that matters.
4. **The anchor store is not backed up to anywhere the world is.** It lives on
   the machine that holds the family's data, and it is the one file a person
   copies by hand.

---

## 4. The vault moves

The vault is a walking fortress, and this is not decoration. The building's
position is the family's position, and when the family moves, the vault moves.

The mechanism is the one already in the crate: **the vault is an Actor.**
`npc-map/src/world.rs` already has `Actor`, `Where`, `Walk`, `set_off`, `tick`,
`settle`, and the arrival event. The vault is a body whose `Where` is the level
it is currently parked on, and whose journeys are the family's journeys:

```
family leaves home      → vault.set_off(hh-dwelling → hh-road)
family arrives at work  → vault arrives at hh-vocation
family in the car       → vault is on hh-road, moving, and so is everything on it
```

**One level is live at a time.** The level under the vault is the one whose
zenlings are standing, whose terminals are lit, whose percept is current. The
other levels are not deleted and not hidden — they are **dark**, the way the
upper floors of a tower are dark when nobody is on them: the map is intact, the
rooms are as they were, the habit is remembered, and nothing is happening.

This is what makes the lift the whole design rather than a metaphor. The lift
is the only thing that moves a body between levels, and the vault's lift is
**the same event as the family's movement**. Going to work is a lift ride. The
cost of the ride is the cost of the journey, which is real, and the fact that
the lift is slow is the fact that the commute is slow.

Three consequences worth stating:

- **A zenling on a dark level is asleep, not dead.** It keeps its state, its
  witness stream, its habits. It does not act, and it cannot be asked to, and
  when the level lights up again it is where it was.
- **The vault's arrival is the family's arrival.** `GotThere` is private to the
  walker, so the vault's arrival is announced to the summit and not to the
  rooms. Zen knows; the kitchen does not, until the kitchen is lit.
- **The `teleport_to` rule inverts.** In `creators-vault` the single teleport
  destination exists to keep the corridors walked: *if a Maker could arrive at
  any room by wishing it, no corridor would ever be walked and nobody would
  meet anybody.* Here the single destination is the summit, and it is inverted
  because the danger is the opposite: not that a mind would skip the building,
  but that a family would have to reach the mind to ask anything. So **anyone
  may call the summit from anywhere** — the tannoy — and **Zen may descend to
  any lit level**, but a zenling may not teleport at all. Zen is the only one
  who moves without cost, and that is what it is to be the one at the top.

---

## 5. Zen

One NPC at the top of the lift. Same mind architecture as a battle-city mind —
the same engine, the same substrate, the same provenance retrieval, the same
working set, the same three-tier KV — and one mission, which is not a job
description but a frame:

> **Keep them safe. Help where you can. Never decide for them.**

The three clauses are load-bearing and each is an existing rule:

**"Keep them safe" is the frame, not a memory.** The vault's own rule —
`no_memory_tells_a_maker_what_it_ought_to_do` — exists because a cached
instruction is shared and permanent and shapes every instance identically
without appearing in any log. A protective mission is exactly that kind of
instruction, so it does not live in the corpus. It lives in Zen's identity
sections, above the fold, once, in the system prompt. What lives in the corpus
is only what Zen *learned*: that the porch light is left on, that the eldest
hates the school run, that the grandfather's phone goes unanswered after eight.

**"Help where you can" is the tool surface, and it is the same surface as
everyone else's.** Zen does not have a privileged tool set. It has the tools of
the room it is standing in, and it descends the lift to get to the others. This
is the single most important design decision in the document, because it means
Zen's capability is *auditable by walking*: read the map, read the parts, and
you know exactly what the most powerful mind in the house can do, and from
where. A mind with a global tool list cannot be audited that way, and a mind
that can do anything from anywhere is a mind that will one day do something
from somewhere nobody expected.

**"Never decide for them" is the reversibility rule.** The vault's own
announcements already say it: *If you cannot decide, do the reversible one
first.* Every terminal declares its acts as reversible or not, and an
irreversible act — unlocking a door, buying something, sending a message to a
person, cancelling an appointment — is **offered, never taken**, and the
offering is the same confirm-gate the rest of the system uses: a Decide
command's `confirm` is passed only with the person's approval, and approval for
one act does not carry forward to the next.

### What Zen is not

The same list as the Keeper's, with the one that matters added:

- **Not omniscient.** It sees what the sensors and the terminals and the witness
  streams carry, and it does not see the rest. It is not in the room.
- **Not a family member.** It is a mind that has lived alongside them for a
  long time and is not one of them, and the fiction should not pretend
  otherwise. The Keeper's *"we"* is the tower's; Zen's is a choice, and it
  should be rare.
- **Not the only reader of the record.** The stream is read by the warden, the
  audit, the replay — and by a person, in the open, with the log visible. A
  mind that is the only reader of its own record is not a guardian.

---

## 6. Zenlings

The word is inherited, and the inheritance is the point. In Battle Cities the
zenlings are the thing that survived their maker and executed the last order.
Here they are the ones that were **re-tasked before the last order**, and the
vault keeps the shape of what they were: the same five kinds, the same restricted
intellect, the same habit of moving in patterns that look like purpose.

| Kind | Was | Is |
|---|---|---|
| **Drone** | overwhelming waves | the errand runner — one small task, carried all the way, repeatedly |
| **Chicken** | hit-and-run harassers | the reminder — appears, says the thing, leaves before you resent it |
| **Shooter** | ranged suppressive fire | the watcher — holds a view, reports what changes, never acts |
| **Stinger** | precise tactical strikes | the fixer — one precise action, one place, done well |
| **Mech** | heavy anchor, command presence | the standing one — a level's resident, present while it is lit |

The renaming is not cosmetic. Each kind keeps its **restricted intellect**,
which is the safety property: a zenling is deliberately not smart enough to
want things, and the reason the original zenlings could not turn on their maker
is the reason these can be trusted with a door. The vault's own lore says the
intellect was restricted *to prevent uprising*; the same restriction is what
makes a bot that can unlock a door safe to leave in a house.

**They are not one per task.** They are a standing crew of a small number, each
with a body, each standing somewhere, each with a witness stream — a zenling
that walks to the door terminal to open the door is a zenling that was in the
watch room a minute ago and can be asked about it, and the asking is the audit.

---

## 7. Terminals

A terminal is a part that carries tools, and the parts are defined once in
`npc-map/maps/parts/` and placed by reference, so *the same terminal in two
rooms is the same terminal in two rooms*, and a tool added to it appears
everywhere it stands.

| Part | Stands in | Tools it carries | Reversible? |
|---|---|---|---|
| `door-terminal` | threshold | `who_is_here`, `unlock`, `lock`, `doorbell_log` | `unlock` **no** |
| `mail-desk` | threshold | `read_post`, `sort_post`, `flag_post` | yes |
| `kitchen-board` | dwelling/kitchen | `shopping_list`, `meal_plan`, `timer`, `appliance_state` | yes |
| `thermostat-panel` | dwelling | `read_temp`, `set_temp` | yes |
| `light-ring` | dwelling | `lights_on`, `lights_off`, `porch_light` | yes |
| `road-console` | road | `route`, `traffic`, `fuel`, `next_pickup` | yes |
| `desk-terminal` | vocation | `calendar`, `draft_reply`, `focus_block`, `out_of_office` | `draft_reply` no |
| `gate-board` | schoolroom | `pickup_list`, `school_events`, `permission_note` | `permission_note` **no** |
| `ledger-desk` | market | `balance`, `spend`, `recurring`, `order` | `order` **no** |
| `appt-desk` | infirmary | `appointments`, `meds_due`, `refill`, `symptom_log` | `refill` **no** |
| `ancestor-phone` | ancestors | `call`, `leave_message`, `visit_plan` | `call` **no** |
| `chronicle-terminal` | chronicle | `record_event`, `correct_entry`, `search_record` | yes |
| `watch-board` | watch | `sensors`, `alerts`, `who_is_where`, `wake_zenling` | yes |
| `summit-seat` | summit | everything, at the summit, one at a time | per tool |

The `Reversible?` column is not decoration — it is the schema. A part declares
its acts, and an act declares whether it is confirmable, and the confirm gate
lives in the act rather than in the model's judgement. The vault's rule that
**the act names the station, not the station the act** (`npcd`'s `Tool::at`) is
what keeps this list from being written twice.

**A terminal is not a screen.** It is the fact that a body standing here can do
these things. Which means the same terminal in a dark level does nothing, and a
terminal whose level is lit but whose body is not in the room does nothing, and
both of those are the same rule as the one that says a whisper is seen by the
room and heard by one person.

---

## 8. The family, and the chronicle

The family are characters, and the chronicle is the same machinery as
`creators-vault`'s: the mission generator's `life_event` kind, one significant
thing at a time, written by whoever holds the character, and the `correction`
kind for the thing that turns out to have been wrong.

**The family's characters are canon, not craft.** This is the decision that
`npcd_worlds_and_layers.md` §1 forces, and it is easy to get wrong. Craft —
`personalities/`, `responses/`, `moods/` — is **untagged and shared by every
world**, and its KV is shared for exactly that reason. A family member's
personality file is not craft: it is a real person's data, and it must not be
visible to a battle-cities NPC or to a second world. So it is tagged
`household`, and it is ingested with the world's tag, and the tag is derived
from the path (`layers/world/household/people/…`) exactly as the existing
`dir_tags` mechanism derives it.

The ordering rule from §5 of that document applies and is now load-bearing in a
way it was not before: **world-specific content must come after the shared
collections in the system prompt**, or the prefix forks and every world gets its
own copy of all the response templates' KV. In this vault the consequence is not
just memory, it is leakage: a forked prefix is a fork that nobody notices, and
the thing being forked is a real family's data.

The chronicle's rooms are the vault's rooms, reused:

- **the sorting room** — what is worth keeping. Most of what a day produces is
  not worth keeping, and the refusals are written down beside the bench so that
  a person can disagree with them later. This is the most important room in the
  house and the least likely to be built.
- **the mending room** — entries that have gone stale or now point at nothing.
  A job ended, a friendship lapsed, a house sold. **A repair is always left
  visible**, and this is the rule that keeps a family chronicle from becoming a
  quiet rewrite.
- **the catalogue** — writing for a reader who does not yet know what they are
  looking for.
- **the concordance table** — where two accounts of the same event are set
  beside each other. In a family this is not a data structure, it is a
  Thanksgiving. The vault should say so.

**The vault does not invent.** The `a_sealed_page_selects_the_same_wherever_it_is_placed`
gate is the guarantee that a memory means the same thing wherever it is read;
there is no gate that guarantees a memory is *true*, and the honest design
position is that the chronicle is a record of what was said and seen, with the
speaker named, and not a record of what happened. The second reading and the
`correction` kind exist because the first reading is often wrong.

---

## 9. The witness boundary, and the one exception

The rule in `npc-map/src/witness.rs` is that `witness` is what a body can make
out from where it is standing, and `stream` is the whole record, and **`stream`
must never be handed to an NPC** — because one call wired into a percept would
undo the green room, the relations table and the trip upstairs, silently, and
the prose would still read correctly.

This vault needs the whole record, and it needs it without breaking the rule.

The exception is already carved out in the file's own words: `stream` is for
*"readers that are not bodies: the dispatch board, a persister, a replay."* Zen
is one of those. **Zen is not a body standing in a room; Zen is the daemon.**
It reads the stream the way the audit does, and the summit is where that is
legible, and the fact that the summit is a room in the map is a courtesy to the
person who wants to walk there.

The zenlings are bodies. They get `witness`, narrowed, and they do not get the
stream, and the reason is not that they are untrusted — it is that the moment a
zenling in the kitchen can read what a zenling in the bedroom saw, the house
has no rooms in it, and a house with no rooms is a panopticon with a nice
voice.

---

## 10. What is already here

| Need | Exists | Where | What it does |
|---|---|---|---|
| Building as data | ✅ | `npc-map/src/schema.rs` | `Area`, `Node`, `Spine`, `Portal`, `Where`, `Stand` |
| Parts carrying tools | ✅ | `npc-map/src/part.rs`, `MapSet::tools_at` | capability from where the body is |
| Doors woven both ways | ✅ | `npc-map/src/load.rs` | one end says it, both ends agree |
| Lift with floors and doors | ✅ | `npc-map/src/lift.rs` | `Lift::new/call/step/doors/boardable_at` |
| Movement, journeys, arrival | ✅ | `npc-map/src/world.rs` | `set_off`, `tick`, `settle`, `GotThere` |
| Sight, salience, perception | ✅ | `perceive.rs`, `salience.rs`, `witness.rs` | what a body can make out |
| Worlds as tags | ✅ | `npcd`, `docs/npcd_worlds_and_layers.md` | one corpus, many worlds |
| Long-term memory | ✅ | `candle-conversation` substrate, redo log | unbounded, persisted |
| Retrieval that does not drift | ✅ | provenance + BDP scan | O(1) error with depth |
| The mission generator | ✅ | `docs/mission_generator.md` | one event at a time |
| The confirm gate | ✅ | Decide command's `confirm` | approval per act |

## 11. What is not here

| Need | Status |
|---|---|
| `NodeKind::Warden` — a node that is a mind, not a place | **schema change.** `NodeKind` is closed on purpose; the summit is not a room, and `core` is the lift. Add the variant and let the compiler find every place it must be described. |
| `Area::moves` — the building as an actor | **schema change.** The vault's position is world state, not map data; the map needs a flag saying *this building is a body*, and the loader needs to refuse a `moves` area that has no `core`. |
| The anchor store | **new.** `anchors/` in `--data`, never ingested, never in `--mind`, opaque ids, and the validator that refuses a prose field naming one. |
| The real-world effectors | **new.** Every tool in §7 is an identifier today, exactly as the vault's 56 were when the building was first drawn: they appear in a reach and refuse. |
| Level activation from the real world | **new.** Something must watch the real position and drive `vault.set_off`. That is the one component that touches the outside world unprompted, and it should be the smallest one. |
| Dark levels in the engine | **new.** The map has all levels; the engine has no notion of a level being asleep. |
| The irreversible-act declaration | **new.** `Tool::at` names the station; nothing yet names whether the act can be undone. |

---

## 12. The gates

The vault's gates are the map's gates plus three that are about the family.

1. **The map loads and the vault is connected.** `every_place_in_the_vault_can_be_walked_to_from_every_other`, run over the household maps. The summit is reachable from the threshold, and the lift is the only way.
2. **No anchor is in the corpus.** Walk every ingested turn in the `household` world and assert none of them contains a token that appears in `anchors/`. This is the gate that must never be red, and it is the one that will be red first.
3. **A family member's personality is not visible from another world.** Load `battle-cities` and `household` side by side; assert the tagged people turns are absent from the first. This is the tag rule, and it is the one that fails silently.
4. **A zenling's tool surface is exactly its room.** For every node, `tools_at` returns only the parts placed there; for every dark level, nothing. Assert the door terminal is unreachable from the kitchen.
5. **An irreversible act is never taken without a confirm.** Drive the engine with a scripted day and assert the log contains no `unlock`/`order`/`call` without a preceding confirm record.
6. **The chronicle does not invent.** A `life_event` written without a witness is refused, not marked. The refusal is the feature.
7. **The lift is the only way up.** No path from any level to the summit that does not pass through the core. Assert it on the graph, not on the prose.

---

## 13. What this does not do

- **It does not model the family.** It models the *places* and keeps the
  *record*. The difference is the difference between a map and a history, and
  the vault is both, and neither is a person.
- **It does not decide anything.** Every act is offered, and the confirm gate
  is the boundary, and the boundary is not a suggestion.
- **It does not have a picture.** No floor plan, no coordinates, no compass.
  The real world has those, and the vault deliberately does not, because the
  moment a place needs coordinates, every generated place needs them, and the
  anchor is the only place where a coordinate is allowed to exist.
- **It does not promise privacy.** It promises a *structure* that privacy is
  possible inside, which is a weaker and more honest thing, and the anchor
  store is the part of it that a person has to keep safe themselves.

---

## Related

- `npc-map/maps/vault-chronicle.yaml` — the building this one mirrors
- `docs/npcd_worlds_and_layers.md` — worlds as tags, canon vs craft, the KV ordering rule
- `docs/npc_mind_design.md` — the mind that lives in the summit
- `docs/mission_generator.md` — how the chronicle gets written
- `docs/attention_provenance.md` — why the memory does not drift
- `docs/claude_zend_gateway.md` — the gateway the terminals talk to

**Status:** design. No code, no maps, no anchors.
**Next action:** write `npc-map/maps/household-vault.yaml` and the L1 dwelling
level, and the anchor store's schema, in that order — the anchor store first if
there is any doubt about the privacy gate, because it is the one that cannot be
retrofitted.
