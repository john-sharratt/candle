# The Command Table's Mission Generator

The Makers exist to finish a world's storyline. They correct where the record contradicts
itself, they tell what it passes over, and they give every character a life — one significant
event at a time. The command table is where that work is handed out. This document is how the
table finds the work and writes it up as missions a Maker can carry out.

Code: `npcd/src/engine/mission_gen/`. Configuration: `<mind>/missions.yaml`. It sits on the
mission system in `npcd/src/engine/mission.rs` and `npcd/src/sim/missions.rs`, inside the
building `vault_world.md` describes, and it is the "missions are generated" half of
`asynchronous_mind_hierarchy.md` §7.4.

## Why the routine bank was not enough

The bank (`engine::mission::bank`) builds missions from what is *around* a character — the
rooms, the people, the machines. It cannot know what the record *lacks*, so the work it sets is
make-work with a plausible description: read the coolant valve, look in on somebody, walk the
halls. The generator reads the record itself.

## The shape

```
missions.yaml ──► a generator (kind, prompt, weight)
                      │
corpus (eras, lives,  ▼
 stories) ─────► target::next ── the ledger blocks what is in hand or settled
                      │
                      ▼
                material::render ── the record, shown first
                      │
                      ▼
       the model, held to one call per kind  (or no_mission)
                      │
                      ▼
       answer::check ── every field held to the corpus; the engine writes the brief and path
                      │
                      ▼
       (correction only) a second reading: do they contradict?
                      │
                      ▼
       Missions::offer ── the pool at the table; collected after lodged, before the bank
```

A loop keeps each world's pool at `keep` missions while its table is open
(`mission_gen::run::spawn`). An operator can ask for one now
(`POST /v1/pulse/missions/generate`), read the pool and the ledger
(`GET /v1/pulse/missions/pool`), and clear them (`DELETE …/pool`, `?ledger=true` to forget the
settled targets too).

## Who chooses what

**The engine chooses what a mission is about; the model writes it.** Coverage is structural:
asked to pick freely, a model returns to what it already knows — the reflection's domains
measured exactly that (`reflection_and_dreams.md`). So the target is a count over the corpus:

| Kind | Target | Ranked by |
|---|---|---|
| `life_event` | a character's life, `layers/life/<who>/` | fewest events written |
| `contradiction` | two documents | neighbouring eras first, then each life event against its era |
| `gap` | an era | fewest stories that name it |
| `review` | a document a generated mission wrote | oldest first |

The cast is every personality with a life or memory folder. A Maker, which writes the world and
is not in it, has neither.

## The ledger

Every target the generator finds is recorded on `Missions` with what became of it — pooled,
carried, done, stuck, or nothing to do — and the fingerprint of what it held then. A target is
reserved the moment it is chosen (`Missions::reserve`), so the loop and an operator's request
generating together cannot both choose it, and released if no mission comes of it. A generated
mission an operator replaces with one of their own releases its target. In hand is
always blocked. Done and nothing-to-do are blocked until the fingerprint moves: a life gains an
event, a document is edited. Stuck is retried until it has been stuck twice on the same text. A
mission called off releases its target.

## The answer is a call, and the call is the reasoning

Each kind has its own call, whose fields are the decisions the work turns on. Free-text briefs
were measured first and failed in every way that matters: a life-event brief recited the
character's own description back at it with no event in it, another addressed the character
instead of the Maker, and a correction called "Zen's project began in 2474" and "Zen woke in
2487" a contradiction. A field the model must fill is a decision it must make
(`asynchronous_mind_hierarchy.md` §1).

| Call | Fields, in the order they are written | Checked |
|---|---|---|
| `life_event` | `date`, `title`, `leaves`, `agrees_with`, `happens` | the date parses as a life document's; no event covers it; it is inside the eras and not after the present; it falls in the life's longest unwritten stretch (or away from its only event); word bounds; no loop |
| `correction` | `quote_a`, `quote_b`, `wrong` (A/B), `why`, `change` | each quote is in its document; then a second reading |
| `story` | `title`, `when`, `where`, `who`, `agrees_with`, `happens` | the title makes a new file; `agrees_with` quotes the era's own words |
| `review` | `checked`, `faults`, `verdict` (keep/revise), `change` | `checked` is a real reading (20+ words) whatever the verdict; a revision quotes the document's own wrong sentences, found in any field |
| `no_mission` | `why` | a sentence |

**A review is the craft's second reading** — the repertoire's *Testing it* (`draft → verdict`).
Every document a generated mission writes joins the review line when its mission is reported
done (`Missions::written`); an operator can put any document in line
(`POST /v1/pulse/missions/review`). A review either keeps the document or sets a revision
mission that quotes what is wrong and says what to change; a revision done takes the document
out of the line, so a revised document is not reviewed again for having been revised. With only
a `problems` field the model wrote it empty and kept everything; the separate `checked` field —
the reading itself, never empty — is what made reviews catch a Zen standing in a command room
four years after the era has it leave the galaxy, and a Mech "watching for forty years" in the
year the watching began.

**The long field goes last.** The grammar writes the fields in order, and with `happens` before
them the model spent itself on the scene and left `leaves` and `agrees_with` empty — the same
lesson the dream call taught (`reflection_and_dreams.md`).

The engine writes the brief and the path from the checked fields, so neither can drift from
the other. A life event's brief also says who its subject is (from their anchor), quotes how
their written events sound, and names the world's own setting — a Keeper with no body wrote a
Zenling's year as "I have no body at all", and a story in a world of towers and machines came
back with quills and parchment. Where `agrees_with` names no era or event by title, the engine's
own grounding (the era the date falls in and the nearest written event) stands alone; facts that
say nowhere where they are written are not passed on.

A refused answer is shown back to the model with the reason, up to three times; a target that
never yields an acceptable answer is set aside until it changes.

**Quotes are checked, ignoring what a faithful copy loses** — case, spacing, emphasis, curly
quotes, and markdown links (`[towers](/tower)` is "towers" to anybody quoting it).

**A correction is put to a second reading.** The model is asked, under a one-token `yes`/`no`
grammar, whether the two quoted statements contradict each other. Anything but a yes declines
the mission. It is asked that way round so that doubt declines: asked "can both be true?", a
no-leaning reading kept "the last Keeper instance was lost in 2534" against "the Portal Retreat
began in 2537" as a contradiction to fix.

**Where a life's next event goes is the engine's rule.** Left to choose, the model wrote the day
after the latest written event every time. The material states the longest unwritten stretch of
the life and the answer is held to it.

## A mission carries its work

A generated mission carries a `Work`: the document it writes and those to read first. Its steps
are ones the engine can see done:

1. `go to band one on the casting level` — a journey that names its level, so a room name
   another level also uses cannot be signed off on the wrong floor. Arrival is matched
   against the room *and* the level.
2. `read <path>` — signed off when the body reads that document at a bench.
3. `write <path> and commit it` (`change …` for a correction) — signed off when a commit
   writes that path.
4. Report at the table.

**The report waits for the record.** `report_done` is refused until the mission's document has
been committed by the reporter *while it carried the mission* — its write step signed off as
achieved, which only a commit of that document does (`Mission::written_up`). The character's
word that the work is finished is the label `asynchronous_mind_hierarchy.md` §7.3 says must
never stand uncorroborated; the commit is the corroboration. A story was once reported done
with its write step struck "thwarted" by the guardian on the character's word and no document
anywhere, so the guardian's step tracker now leaves every step the engine sees
(`mission::engine_sees`) to the engine. The record is the mission's own rather than the bench's
list of who last committed what: that list is not kept across a restart, and it would pass a
review whose Maker had written the document before the review began.

**A mission writes its document and nothing else.** While a body carries a mission with work,
`file_write`, `file_edit` and `file_delete` on any other path are refused. A Keeper sent to read
an era rewrote its closing line with a flourish of its own, and another left a second copy of
its event under a name it made up.

**A document is committed whole.** A mission's document below its floor (`Work::min_words` —
250 words for a life event, 380 for a story) is refused at `bench_commit`, while it can still be
finished.

**The compass says the act.** For a step about a document the mission compass names the exact
`invoke` — the desk's `file_read` with the path, or `file_write` then `bench_commit` — when the
body stands at a desk, and the way to the nearest desk when it does not. Told only "next: read
layers/eras/…", a Keeper in the writing room reflected and dreamt for ten minutes and never
touched the desk.

The bench decides where a document is written: lives and personalities at a character terminal,
stories at a story desk, places at the map table, the rest of the world's history at a
chronicle terminal. `file_write` and `file_list` are at every writing bench for this reason —
held to the story desk, a life could not be written on the level that writes lives.

## Getting there: the lift

Every mission crosses levels, and the first live runs lost nearly all of them to the lift, in two
ways, both fixed in the world rather than in the prompt:

- **A ride could be abandoned by accident.** A rider waits on its origin landing until the car
  opens at its floor, and it was offered turns — and `move_to` — while it waited. One taken
  walked it off the landing and cancelled the ride. A body that has asked to ride is now offered
  nowhere to walk until it is set down (`engine::body::destinations`), and the compass is silent
  while it rides.
- **Calling the car and boarding it were two acts with a wait between.** Makers called it and
  left before it came, again and again, until the cast agreed among themselves that the lift was
  broken. `lift_use` is now offered on any landing: with the car away it calls it and the body
  waits, boarding when it opens there (`npc_map::world::World::ride_lift`).

## The prompts

`missions.yaml` holds the system voice, `keep`, and one entry per generator: `id`, `kind`,
`weight`, and `prompt`. The prompt explains the call's fields; it never asks for a brief or a
path. The material comes first in what the model reads and the prompt after it, so the last
thing read is what to do — instruction first, then a character's anchor in the second person,
and the model went on in the anchor's voice.

## Measured

Live runs of 2026-10-09 on the RTX 3090, Qwen3-30B-A3B serving both the generator and the
Makers, with the table open and the pool kept at 4:

| run | minutes | missions done | stuck | offered | declined | targets set aside |
|---|---:|---:|---:|---:|---:|---:|
| first soak with typed calls | 8 | 6 | 0 | — | — | — |
| final soak | 30 | 6 | 0 | 12 | 9 | 4 |

Declines are the checks working: corrections the yes/no judge did not find contradictory,
life events dated outside the open stretch, and quotes not found in their documents. A
target is set aside after three refused answers, so a decline costs one decode, never a
mission.

What was written: stories (the `gap` kind) read as scenes — place, people, dialogue, in the
world's voice — and clear their 380-word floor comfortably. Life events are sound in date
and grounding but some carry the writing Maker's own manner into the character's voice.
Reviews reason well about faults; carrying out a `revise` verdict is the weakest step, since
a Maker revising a document can leave part of the fault in place.
