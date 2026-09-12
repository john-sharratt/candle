# The Vault

The Makers work in a building. Six levels, a lift and a stair. A Maker comes up
to the command level for orders, descends to the level that holds the job, walks
to a console, and works. Sixteen of them, at once, and every working level can
hold all sixteen.

**The building is data, not prose.** It lives as YAML in `npc-map/maps/`,
deployed to `<mind>/map/`. The description an NPC carries is generated from it,
and so is everything else that has to describe it. This page is the reasoning;
the description of record is what the generator emits:

```
cargo run -p npc-map --example vault_memory
cargo run -p npc-map --example vault_memory -- vault-casting
```

## Why it is data

The same building has to be described at three magnifications — the world an
NPC carries, the level it is standing on, and the place it is standing — and
those three must never disagree. Written by hand they drift, quietly, because
nothing checks one paragraph against another.

The other reason is arithmetic. A game world has millions of places in it. None
can be hand-written, and none can afford inference at the moment somebody walks
in. So the structure is authored or generated, the character of each place is
stored beside it, and the description is a pure function of the two — same bytes
every time, for nothing.

**There is no floor plan and no geometry.** An earlier draft of this page was
six ASCII maps. They were useful for thinking and are the wrong artifact to
keep: a map's meaning lives in column alignment, which does not survive
tokenisation, so it never reaches a model at all — and coordinates exist to
serve a picture, which is a constraint every one of a million generated places
would have to satisfy for something nobody looks at. What an NPC needs to
navigate is which places adjoin which, and that is a graph.

## The rules the plan is drawn from

**A level is a job, not a folder.** The mind stores a character across four
sections — `personalities/`, `layers/agency/`, `layers/beliefs/`,
`layers/memory/`. Those are four files and one job. A building laid out by
storage would make a Maker ride the lift to change a belief; a building laid out
by job gives it one station that reaches all four.

**Reading is a tool; writing is a place.** Moving between levels is for changing
the task. Everything inside a task is a verb at the station, including reading
anything in the building — reading takes no lock and changes nothing, so there
is no reason to make it cost a journey. Writing is the part that needs a place,
because writing is what conflicts.

**The plan affords sixteen; policy decides how many are live.** Eras overlap and
portraits are judged against the ones beside them, so those levels will want a
cap. But a cap is a rule and can change on a Tuesday; a floor plan cannot. The
building affords the maximum and the contention rules decide what runs, which
keeps a policy question out of the architecture.

**Some work belongs to nobody alone.** Each level has a table for the thing that
two holders own between them and neither can write by themselves — the
concordance table for a boundary between two eras, the road table for a route
between two places, the relations table for what stands between two characters.
The result lands in both files.

**And the walls are already real.** `MindPath` refuses `..`, refuses `.exe`, and
refuses anything that is not a named section (`npcd/src/mind/path.rs`). A Maker
cannot wander into `.substrate` because it is not addressable. The building's
walls and the path guard are the same wall.

There is one vault per world. This one is Battle Cities.

## Memory and perception

Two separate things, and the line between them is sharp: **a percept is what
arrives without asking; everything else is an action.**

**Memory** is what a Maker knows because it works here — the whole building,
every level, every room, what each affords and what adjoins what. It never
changes, so it sits at the front of the context and is shared by all sixteen. It
is about 2,700 words, call it 3,000 tokens, paid once.

**Perception** is what is true here and now: who is in this room, which stations
are lit, what has changed. It is small and it goes at the back. It reports only
what differs from the permanent world — the green room always has twelve seats,
so a percept never mentions them.

Three consequences the design turns on:

- **Perception is sight only, and deliberately poor.** You can see a lit station
  and a person at it. You cannot see what they have written. Every social
  mechanism in the building depends on that asymmetry: if a Maker at one station
  can perceive what another has written, nobody ever needs to talk, the green
  room is furniture, and the relations table is a formality.
- **A corridor sees through its doorways.** Without that a Maker shut in a room
  has no reason ever to look up, and the building goes socially dead.
- **The roster, the gap ledger and the timeline wall are places, not context.**
  They are read by walking to them. That is both the fiction and the right
  memory hierarchy, because the constraint is the same in both: do not carry
  what you are not using.

**Affordances, never recommendations.** A memory says what a place *is*, never
what to do about it. *The quiet room is where reading happens undisturbed* is a
fact about a room; *go to the quiet room when you need to concentrate* is a
standing order smuggled into the world. The second kind is worse in memory than
anywhere else, because it is cached and shared — one line of it would shape all
sixteen Makers identically, for ever, without appearing in any log.
`no_memory_tells_a_maker_what_it_ought_to_do` in `npc-map/tests/vault.rs` is
aimed squarely at it.

## Three ways of knowing, and they are different modules

Memory, perception and the stream answer three different questions, and an
earlier version of this had all three tangled in one function.

| Module | Question | Shape |
|---|---|---|
| `describe` | what is this building? | static, shared, cached, ~2,700 words |
| `perceive` | what is true here, now? | pure snapshot, no cursor, two paragraphs |
| `witness` | what happened while I was not looking? | ordered, one cursor per body, place-scoped |
| `stream` | what happened, everywhere? | ordered, named cursors, **no place filter** |

The split between the last two is the load-bearing one. `witness` is what a
*body* can make out from where it is standing: its own room and what it can see
into, narrowed on the way out — you see a console light up in the next room, not
what is on it. `stream` is the whole record, for readers that are not bodies:
the dispatch board, a persister, a replay.

**`stream` must never be handed to an NPC.** One call wired into a percept would
undo the green room, the relations table and the trip upstairs, and it would do
it silently — the prose would still read correctly and the building would simply
stop mattering. `npc-map/tests/reach.rs` states the boundary as a property over
every pair of rooms in the vault, because a leak there announces itself no other
way.

## Getting about

**There is no physics here and no metres.** A journey is measured in *stops* —
the places somebody would actually pause and reconsider — and a stop is a tick,
which is a turn. `set_off` returns how many; the body covers one leg per
`World::tick`.

A leg is a maximal run of the route on one side of a level boundary, which
makes the vault's numbers these:

| Journey | Stops | Doorways |
|---|---|---|
| Anywhere on your own level | 1 | 1–5 |
| The lift ride, however many levels | 1 | 1–5 |
| The far corner of the top level to the command room | 3 | 9 |

So crossing a level is free enough not to think about, and changing level costs
— which is the shape the job wants: micro-tasks stay on your floor, changing
job is a journey. Nine doorways compressing to three stops is a compression of
the building, not a flattening of it.

The three stops on the long trip are the three a person would name: *you go to
the lift, you ride it, you walk to the room.* Cutting the journey there rather
than at every doorway is what makes each pause worth a turn — nobody
reconsiders halfway down a corridor, and everybody reconsiders at a lift.

**The outcome is an event, not a return value.** By the time a journey has
succeeded or failed the body has had turns in between, so the answer is news and
is delivered like news. This is the one place a body is told about its own
doings: it knows it set off, it does not yet know whether it arrived.

Every journey that begins is answered exactly once — `GotThere`, or
`LostTheWay` with a reason (diverted, teleported, sat down). A journey dropped
in silence leaves a body waiting for ever. And a journey answered by its own
destination *succeeded*, however it got there: a Maker who teleports to the room
it was walking to arrived.

**One primitive moves a body.** `World::place` emits the events and settles the
journey; `World::tick` is a caller of it, and so is a game that owns its own
movement. Battle Cities walks a tile grid at its own pace and says where the
body ended up; the vault advances a leg per tick. **Nothing about how long a
journey takes lives in this crate** — landing on your route continues it,
landing on your destination finishes it, landing anywhere else abandons it and
says so.

**Teleporting is one destination.** A Maker can jump to the command room from
anywhere. A shortcut that went *anywhere* would dissolve the building; one that
goes to the room everybody has to reach most often keeps every other distance
intact, because every trip out is still walked. It is a map fact (`teleport_to:`
on the building), not an engine fact, so a world that never named one has no
shortcut.

**What the stops buy** is being caught at the lift. Everybody changing level
passes through one, so the lift lobby is where a Maker finds out who else is
about — and the narration has to condense to keep that readable: a body crossing
your field of view is one thing that happened, not one thing per room, or the
prose degrades exactly as the building gets busy.

## Talking

**Delivery is by place; direction is by address.** Speech lands in the room and
everybody standing there gets it. Who it was aimed at is a fact about the
utterance, not a filter on who receives it — so one event reads three ways:

```
addressed to you   Maker-01 told you, "get out of here."
overheard          Maker-01 told Maker-02, "get out of here."
to the room        Maker-01 said, "the redoubt burned twice."
```

Being told something and watching somebody else be told it are different facts,
and both are true of one event. That is the parallax property, and it costs
nothing extra — the world records the whole utterance and each reader narrows it
on the way out.

Two consequences. **Sixteen Makers in one building need no interaction forks** —
a conversation is a reading of the act stream, not a thread in anybody's head;
the room is the channel. And **who you may address is a product of location**,
like the tools within reach: the percept already names who is here, so the
vocabulary is in context and addressing somebody who is not in the room is
refused. Talking *about* an absent person is undirected speech with their name
in the words.

Nothing said in a room reaches the corridor outside it. You can see who is in
the green room from the south run; you cannot hear them, which is why walking in
to ask is still necessary.

## How a level memory is shaped

Four beats, and no more: what the level is for, what it is like, one sentence
of the shape of the route, and its rooms under three headings — *the work
here*, *to consult*, *for company*. Those three are the choice an NPC arriving
is actually making.

**Corridors are not described at all.** An earlier version named every passage
and indexed the rooms by which one they opened off. It was correct and it was
exhausting: a reader had to hold six abstract names in mind to work out where
anything was, and the corridors were the one thing nobody ever wanted to go to.
Which door leads where is a question for the moment somebody is walking, and
perception answers it then. `no_level_memory_names_a_corridor` keeps it that
way.

**Rooms that do the same job are described once between them.** Three bands are
one fact, not three, which is how anybody who works there thinks of them.

## Parts, and where the tools come from

A node is somewhere to stand. Everything *in* it is a **part** — a terminal, a
board, a chair — defined once in its own file under `npc-map/maps/parts/` and
placed by reference. So *world history terminal* means the same thing on every
level of every building that has one, and gains a tool everywhere at once when
it gains one there.

**The tools hang on the part, not on the NPC.** Standing at a chronicle
terminal is what makes rewriting an era possible; walking away is what makes it
impossible. `MapSet::tools_at(node)` reads the tool surface off the map, so
what a Maker can do is computed from where its body is — a tool it is not
standing next to is never offered and cannot be reasoned about wrongly. A
corridor comes back empty, which is the point.

Each part carries **two descriptions for two readers**:

- `short` — one clause for a level's prose, what somebody glancing round the
  room would say. It must name its own subject, because several parts stand in
  one room and their clauses run together.
- `long` — **never** prose. The provenance carried alongside the tools when
  they are offered: what this thing is, what it does, what taking one commits
  you to. Read at the moment the tools are, not while describing a level.

They are separated because they are wanted at different times and at very
different lengths. A level that inlined every long description would be
unreadable; a tool offered without one would be unusable.

`binds` lives on the part too — it is the terminal that holds an era, not the
room, and the same terminal claims the same thing wherever it stands.

## How a level is authored

A node says what it opens `off` — usually one corridor — and the loader weaves
the doors at both ends. A spine that loops has its own links woven from the
order alone. So a level is fifteen `off:` lines rather than sixty exits that
have to agree, and the commonest authoring fault in the whole schema, a door
wired one way, is no longer expressible.

Sight is stored the way it is described: through your own door is the rule and
is woven, so the `sees` field carries only what breaks it — a gallery rail over
a room it cannot reach.

Bearings were removed. They described a level nobody can see, cost a line on
every door, and answered no question an NPC asks: *which way round the ring*
comes from the spine, and *how do I reach the green room* is a route, not a
heading.

## Deriving this from the game world

Battle Cities is a terrain grid with buildings placed on it, and the buildings
are made of parts — walls, floors, doors, turrets. That model hands most of
this over for free: **flood-fill the floor parts bounded by walls, and each
connected region is a node; each door part joining two regions is an exit.**
Rooms and doors are the standard output of walking a tile map.

What flood-fill cannot supply is the half worth reading — a name, a kind, what
a place is for, what it is like. That comes from metadata on the parts, in
exactly the fields the schema already has. **This schema is the metadata
schema; there is no translation layer.**

**The reference runs one way.** A building instance names the area it is; a
marker part names the node it stands in. Nothing in a map file names a
coordinate — mirror the geometry in here and it goes stale the first time a
level designer nudges a wall, silently, because there is no picture left to
check it against.

Three consequences already built:

- **`ground`** on a node — the surfaces underfoot, as a list. The whole of what
  a terrain grid contributes to a memory: not per pixel, a summary of the
  place, and empty indoors where a floor is a part like any other.
- **A `ground` node kind** for open ground outdoors. Work / social / store /
  passage / core describes a building and does not describe a crossroads.
- **An area may hold both its own nodes and child areas**, which is what the
  outdoor map needs: its own places, and the buildings standing on them as
  children. A door from outside is a portal to a node on a *level* of the
  building, never to the building itself — an instance out here, all its parts
  once inside.

Two things stay different indoors and out. Sight indoors follows a rule — you
see into a room from the corridor it opens off — and only what breaks it is
worth stating; outdoors nothing bounds anything, so there is no rule and every
sightline is its own fact. And flood-fill regions are not always rooms: a hall
floods as one region that may want to be three, a room split by a half-wall
floods as two. Markers have to win, or the level geometry silently redraws the
NPC's mental map.

## The levels

| | Level | What is taken up there |
|---|---|---|
| 1 | the command level | orders, intake, and enquiries — nothing the world is made of is written here |
| 2 | the chronicle | one era, at sixteen stations |
| 3 | the story level | one gap in the record, at sixteen |
| 4 | the cartography level | one place, at sixteen — and the whole geography, at one |
| 5 | the casting level | one character, at sixteen |
| 6 | the portrait level | one character, at sixteen |

Every level is the same shape: a ring corridor braced across by two cross runs,
so circulation is a loop with rungs rather than a spine with dead ends and no
doorway the whole crew funnels through. The lift and the stair land at the same
place on every level.

## The rooms the repertoire needed

The building was drawn before the task list was. Washing that list against
archival practice, the craft of scene construction and life aboard a habitat that
runs itself produced four functions with nowhere to happen, so seven rooms and
eight parts were added — see [`maker_repertoire.md`](maker_repertoire.md) for the
clusters they stand under.

| Room | Level, off | Parts | Namespace |
|---|---|---|---|
| the sorting room | chronicle, lower cross run | appraisal bench, 2 seats | `record.` |
| the catalogue | chronicle, lower cross run | catalogue, 2 chronicle terminals | `record.` |
| the mending room | chronicle, west run | mending bench | `record.` |
| the board room | story, lower cross run | structure board, 6 seats | `structure.` |
| the receiving room | command, cross run | accession desk *(binds the intake)*, stores | `record.`, `stores.` |
| the enquiry room | command, south run | enquiry desk *(binds an open enquiry)*, 4 seats | `enquiry.` |
| the plant room | command, west run | plant panel | `plant.` |

Three of them change what the building *is* rather than merely adding to it.

**The command level is no longer work-free.** It was authored as the one level
nothing happens on, and its `lacks` line said so. Intake and enquiry are work, so
the line now says *no making* rather than *no work*. The capacity invariant in
`npc-map/tests/vault.rs` that guarded it asserted zero stations there; that was a
proxy for "nothing is made here" and is now stated as what it actually protects —
single posts, never banks of desks, so no standing order can queue the crew
behind them.

**The enquiry room is the only room that faces outward.** Every other room
consumes work from the standing list. This one takes a question from somebody
downstream of the whole building, phrased in their words rather than the vault's,
and the ones that cannot be answered are the valuable ones: an unanswerable
enquiry names a hole nobody inside had noticed. It is a second source of work
beside the list.

**The sorting room refuses.** Nothing in the building previously judged whether a
thing was worth keeping at all, which is the judgement an archive makes first.
Most of what a world throws off is not worth keeping, and the refusals are
written down beside the bench so that whoever comes after can disagree with them.

**None of the tools these parts name are implemented.** They are identifiers on
the parts, so they appear in a Maker's reach and refuse — the same state the
original 56 were in when the building was first drawn.

## What is not in the building

**`responses/` and `moods/` have no level.** They are the craft libraries, and
neither is one of the jobs a Maker was given. Still editable through the console
by a person. If refining them turns out to be Maker work it is a seventh level,
not a desk on an existing one, because it is a different job.

**`settings/` has no level,** deliberately: it is how the mind is configured,
which is not Maker work and not something a Maker should reach.

**`worlds/` has no level.** A world is a filter over the corpus and there is one
vault per world, so the filter *is* the building.

## Open

- **The portrait level has no storage.** Every mind section is markdown or YAML
  and `npcd/src/mind/path.rs` refuses `.png`. The studios are described but the
  plate room has nothing real to file until this is decided: a section with a
  binary format, or an art store beside the corpus that addresses may name.
- **What role a Maker holds.** Mind writes are admin today, and a Maker doing
  its job is a write. It must not be admin, because admin reaches `settings/` —
  which has no level precisely so it stays out of reach.
- **Where a mini story is stored** — an entry under the era it fills, or a
  section of its own.
- **The per-level caps.** The plan affords sixteen everywhere; the chronicle and
  the portrait level both want fewer live at once, for the reasons above. Those
  numbers belong to the contention rules and are not set.
- **Nothing is wired to the substrate.** The tools a room puts within reach are
  identifiers; no Maker has yet been handed a percept and asked what to do.
