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
       Missions::launch ── an operation opens; its draft goes on the table
                      │
          one Maker drafts it ── the quality gate holds the report (gates.rs)
                      │
                      ▼
       the table reads the draft (reading.rs) ── verdict sound / mend / fail, faults quoted
                      │
                      ▼
       a different Maker reviews it ── mends in place and passes, or rejects
                      │
                      ▼
       succeeded: the document stands     failed: it moves to rejected/, the target is retried
```

A loop keeps each world's pool at `keep` missions while its table is open
(`mission_gen::run::spawn`), reading waiting drafts before it opens new operations. An operator
can ask for one now (`POST /v1/pulse/missions/generate`), read the pool and the ledger
(`GET /v1/pulse/missions/pool`), and clear them (`DELETE …/pool`, `?ledger=true` to forget the
settled targets too).

## Operations

**The table holds the objective; a mission is one stage of it.** Every mission the generator
writes opens an operation (`sim::operations`) — named like "Operation Iron Lantern", from two
word lists in the world's register — and is its draft. The stages:

1. **Draft.** One Maker writes the document. Its report is refused until the document passes
   the engine's **quality gate** (`mission_gen::gates`), with each fault worded as what to
   change: the word bounds of its form (life event 250–800, story 380–1300), a story's heading
   and a life event's lack of one, no paragraph past 160 words, no sentence said twice, no
   four-word phrase coming back more than three times, a life event in the voice the rest of
   that life is written in (the canon's lives are all "you"), and nothing of the scaffolding
   ("Title:", tool calls) in a story. **Nor the writers' room** (`mission_gen::leakage`): the
   writers' vocabulary — every level, room and part named on the map the Makers stand in, the
   names of the people in it, and the bench's words for writing ("commit", "the draft") — is
   compared with the world's own, its eras and world documents; a term of the first the second
   never uses is the vault in the work, and is refused at every stage. Nine of eleven pieces
   judged against the lore had it, and review and canon passed every one.
2. **The table's reading.** The table reads the draft against what it answers to — its era,
   the rest of its life, the world — under the `reading` prompt of `missions.yaml`, and answers
   `notes` (its working, never kept), `event` (what happens in it, in a sentence, or `none`),
   `checked`, `faults` (quoting the draft) and a verdict: `sound`, `mend` or `fail`. A reading
   whose `event` is `none` cannot be `sound`: the table passed a life event in which a door was
   opened on an empty room and shut again. On the last attempt a reading that still says both is
   taken at its word — `mend`, with "no event" as its fault — since one that insisted three times
   otherwise left the reviewer no reading at all. **A draft stands on two sound readings**: a
   `sound` reading is followed by a second, with a fresh seed, and a second that finds faults is
   the one the review gets — a single reading talked itself into passing that same door ("while
   quiet, it has a clear beginning and end"). Before it reads, the table tidies the draft
   (`gates::tidy`), which also drops a line that is only the document's own path. A draft whose
   operation has a brief is read beside it — `# What it was to tell`, the brief's "What happens"
   (`run::reading_prompt`) — and one that does not tell that event is not `sound`. The review carries the draft's own brief's "What
   happens" (`Operation::brief`), so a reviewer writing it anew knows the event it was to tell.
3. **Review.** A review mission goes on the table carrying that reading. **No Maker takes two
   stages of one operation** (`Operations::may_take`): the writer never reviews its own draft,
   and a review reported stuck goes to somebody new. The reviewer reads the draft and what it
   answers to, mends what can be mended with `file_edit`, and `report_done` passes the
   operation — held to the same gate — or `report_rejected` fails it.
4. **Canon check** (`mission_gen::canon`), for a life event or a story. A third Maker, who has
   carried no other stage, checks the reviewed document against the main storyline: its steps
   read the draft and the eras around it — the one it is set in and those either side — and
   neither verdict is taken until they are read. It looks for major contradictions only (a
   date, an outcome, who was where, what existed), and decides for itself: accept it, put a
   contradiction right with an edit and accept it, or reject it. A correction skips this
   stage — agreeing with the record is what it was for.

**Every stage starts in the past.** Each stage of an operation on a life event or a story
(draft, review, canon check) opens with two steps: go to the time room on the time level, and
`set your time to Y at a time machine` — Y the life event's own year, or the year its story's
era opens (`canon::set_in`; a correction, answering to two eras, has none). `time_travel` there
sets the mission's year and signs the step off (`Mission::travelled`); until the work is
reported nothing after Y reaches the Maker's recall (`docs/npcd_worlds_and_layers.md`), and no
report is taken while the step is still to do. Writing 2950, Makers had been handed the next
century's eras by their own recall.

**Repair before verdict.** When the table's reading finds faults (`mend` or `fail`), the review
carries a repair step — the draft changed and committed — and neither verdict is taken until it
is done: the reviewer puts right what the table found, writing the draft again whole when its
voice is wrong, and rejects only what its mending could not save. Left to judge without
mending, reviewers rejected nine drafts in nine. A `sound` reading leaves the edit optional.

**The table has the last word on what a review mended** (`Operations::table_read`). A review
that passes a draft the table did not find sound sends the mended text back to the table, which
reads it again: sound, it goes on to its canon check; still wanting, it goes to another review,
by somebody new; read three times (`READINGS_LIMIT`) and still wanting, the operation fails in
the table's words and the draft leaves the record. The table reads with the subject's anchor,
its other events and the era in front of it; a reviewer mends in a few turns. Before this, a Zen
the table failed for having hands and a chair was "mended" by its reviewer and stood.
An operator can send a review still waiting at the table back to be read again
(`POST /v1/pulse/operations/:wid/:oid/read-again`).

**A rejection shows its evidence** (`mission_gen::rejection`). `report_rejected` is refused
unless its reason quotes the draft as it stands (the same 12-character quote a reading is held
to), and on the canon check unless it also names an era it contradicts, by title or year. A
Maker that had just rejected one story as "structurally broken — it loops endlessly" rejected
the next document it was given, a life event that had passed review, with the same words: the
reason was its own previous report, still in its history, and nothing in the draft. A draft
already gone from the record needs no quote. Likewise a failing reading whose `faults` say only
"Nothing." is taken as listing none, so the reviewer is pointed at what the table checked
rather than told the table found nothing wrong.

**Clerical form is the engine's** (`gates::tidy`): before the table reads a draft and before a
report is gated, a life event loses a heading it opened on, a story gains its title, and a
run-on paragraph is broken at sentence ends. Reviewers refused at every report for a heading
line went round until they despaired; nobody is asked to judge what needs no judgement.

**The table hands its work out.** A Maker free at the open table is given the next mission it
may take (`Runtime::at_table_summons`) rather than told to collect one — told every tick,
Makers stood at the table scanning and reflecting with thirty-five reviews waiting. A reviewer
whose reading is done is asked for its verdict, not for "what you found": a review has nothing
to find.

**Memory holds only what passed** (`engine::mind_record`). Every conversation a mind document
becomes — a layer document, a life episode, each belief and relationship the episode forms — is
marked with the document's path. At boot, a document gone from disk or changed is retired from
the substrate before it is ingested again, and a document an operation is still drafting or
reviewing is held out of the ingest altogether; a rejected draft is retired the moment its
operation fails. Conversations written before the mark are found by a life event's date and
title, or a layer document's address.

A rejected draft is moved to `rejected/<operation>/` in the mind (`Benches::retire`), where no
world reads it and an operator still can; its target counts a stuck and is tried again by a
fresh operation, up to the ledger's limit. Only a draft leaves — a life event or a story
(`Operation::leaves_on_failure`). A correction works on a document the record already held, so
when it fails its document is put back to what it said when the operation opened
(`Operation::before`, `Sim::set_aside_failed`): moving the era aside would lose canon, and leaving
the rejected edit would let it stand — unless a later operation on the same document has since
succeeded, whose accepted text that would overwrite. Every failed operation is settled this way by
the generator's loop, however it failed — rejected, read past the limit, or stuck — and a
rejection is settled at once as well. A target is settled when its operation is, not when
its draft is reported. An operator can put any document already on the record through the
reading and a review (`POST /v1/pulse/operations {path}`).

The operations tab of the npcd console (`/operations`) shows each running operation's place
in the chain and who carries it, the table's reading and the log of every stage, and folds the
finished away below. An operation can be renamed, its objective restated, the brief of its
waiting mission rewritten (`PATCH /v1/pulse/operations/:wid/:oid`), or called off
(`POST …/:oid/cancel`), which stands down whoever is carrying it.

## Who chooses what

**The engine chooses what a mission is about; the model writes it.** Coverage is structural:
asked to pick freely, a model returns to what it already knows — the reflection's domains
measured exactly that (`reflection_and_dreams.md`). So the target is a count over the corpus:

| Kind | Target | Ranked by |
|---|---|---|
| `life_event` | a character's life, `layers/life/<who>/` | fewest events written |
| `contradiction` | two documents | neighbouring eras first, then each life event against its era |
| `gap` | an era | fewest stories that name it |

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
| `life_event` | `date`, `title`, `leaves`, `agrees_with`, `turns`, `happens` | the date parses as a life document's; no event covers it; it is inside the eras and not after the present; it falls in the life's longest unwritten stretch (or away from its only event); `turns` is an act (6–60 words, not "nothing"/waiting/remembering — a brief whose subject sat in the dark waiting was written as a mood and failed); word bounds; no loop |
| `correction` | `quote_a`, `quote_b`, `wrong` (A/B), `why`, `change` | each quote is in its document; then a second reading |
| `story` | `title`, `when`, `where`, `who`, `agrees_with`, `turns`, `happens` | the title makes a new file; `agrees_with` quotes the era's own words; `turns` is an act, as for a life event |
| `reading` (the table's, of a draft) | `notes`, `event`, `checked`, `faults`, `verdict` (sound/mend/fail) | `notes` is the reader's working, unchecked and never shown; `event` is a sentence (≤ 80 words), and `none` is never `sound`; `checked` is a real reading (20–700 words) whatever the verdict; a fault quotes the draft's own wrong sentences, found in `checked` or `faults` |
| `no_mission` | `why` | a sentence |

**The table's reading is the craft's second reading** — the repertoire's *Testing it*
(`draft → verdict`). With only a `problems` field the model wrote it empty and kept everything;
the separate `checked` field — the reading itself, never empty — is what made it catch a Zen
standing in a command room four years after the era has it leave the galaxy, and a Mech
"watching for forty years" in the year the watching began. `notes` comes first because the call
opens straight into its first field: with no place to work, a reading that had dates to set
against the eras did its working in `checked`, ran to twelve hundred words, and was refused for
length on every attempt. Standing alone, as a review
generator that set revision missions, its findings were carried out only in part — Verdi's
revision kept the lines it was told to cut — which is why a reading is now handed to a second
Maker who must pass or reject, rather than a revision set for the first.

**The long field goes last.** The grammar writes the fields in order, and with `happens` before
them the model spent itself on the scene and left `leaves` and `agrees_with` empty — the same
lesson the dream call taught (`reflection_and_dreams.md`).

The engine writes the brief and the path from the checked fields, so neither can drift from
the other. A life event's brief also says who its subject is (from their anchor), quotes how
their written events sound, and names the world's own setting — a Keeper with no body wrote a
Zenling's year as "I have no body at all", and a story in a world of towers and machines came
back with quills and parchment. Where `agrees_with` names no era or event by title, the engine's
own grounding (the era the date falls in and the entries of the life just before it) stands
alone; facts that say nowhere where they are written are not passed on.

**The Maker researches before it writes, up to the year and nothing after**
(`mission_gen::research`). Every draft's steps read, between reaching the desk and writing, what
the record already holds around the work — the same rule for any document an operation writes,
in any world, from four things: the year the work is set in (the one its time step sets), the
folder it is written into, the dates of the documents already written, and the names its brief
uses. In order, up to five documents:

1. the era the year falls in, and the era before it when the year is within ten years of the
   era's opening;
2. the two latest documents in its own folder dated before it — the entries a life holds before
   the day being written, the stories told before this one;
3. up to two documents elsewhere, dated in the thirty years before it, that name somebody or
   somewhere the brief names.

They are read oldest first, and the brief lists each with what it is read for. Nothing later than
the year is read: a draft used to read the single written event nearest its date, which for a day
in 2950 was one in 3086, and nothing that led up to it — and with so little of the record in front
of it, the Maker wrote from the room it stood in. A later entry the work must not contradict is the
canon check's to hold it to. The voice example the brief quotes is taken from before the year too,
where the life has one.

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

**Writing is done sitting down** (`engine::compose`). A Maker whose mission writes a document is
offered `compose` at a desk. The sitting is a clean conversation opened for it: the mind's writing
voice (`writing` in `missions.yaml`) is put the brief, every document the research steps sent the
Maker to — in full, each cut at 900 words — and the piece as the record has it when there is one
(the text the table read, never the Maker's hand-typed working copy, and nothing when a review
writes a failed draft anew), and answers with the whole piece through one
`draft(text)` call. A piece under its floor is carried on in the same sitting — the writer is shown
what it has written and asked for what comes next, which is put after it — up to three answers in
all: asked instead to write it whole again at full length, it wrote the same 203 words back. That goes into the Maker's working set through
`file_write`, held to the same checks, and the Maker reads it back, mends it and commits it. The
sitting is not a fork of the Maker's own conversation: with thirty thousand tokens of lifts, desks
and colleagues above the brief, composed pieces still carried the vault, and a reviewer told to
write a draft again was not shown the draft and wrote another piece from nothing. A
draft put together as one `file_write` argument in the middle of the Maker's running conversation
carried that conversation — the vault, colleagues by name, a plot about somebody writing — because
the room was the most concrete thing in front of it and the brief was far above. So for that
Maker `compose` is how the piece is written whole: `file_write` is not offered to it
(`mission_acts::offered`), and `file_edit` stays for the single passages the gates name — an edit
of the mission's prose piece that takes more than half its words is refused and pointed at
`compose` (`engine::passage`), because a Maker composed its story and then replaced the whole of
it through one `file_edit` with the vault in it again; a
composed draft rewritten by hand afterwards lost what the composing gave it ("I am writing this from
the chronicle level, where the light ring hums").

A Maker's uncommitted working copy outlives a reboot: each world's benches are written to the
substrate as `benches/<world>` beside `missions/<world>` whenever they change, and restored under
the mind the restarted daemon was given (`Benches::restore`). Saved without them, every restart
sent each Maker back to an empty working set, and one that had composed its piece composed it again.
While the working copy holds the piece at its floor, the compass says to read it back and commit
it, not to write it. A commit collides only with somebody else's: a body's own earlier commit is
no collision, and the table's clerical tidy before a reading rebases every open working copy of
the document (`Benches::rebase`) — a reviewer restored from a restart was refused its commit for
a change "under" it whose hand was its own.

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
mission whose Maker had written the document before it took the mission up. A review is the one
mission that may leave its document as it found it (`Work::edit_optional`): it has no write
step, and is held instead to having read the draft and to the quality gate.

**A mission writes its document and nothing else.** While a body carries a mission with work,
`file_write`, `file_edit` and `file_delete` on any other path are refused. A Keeper sent to read
an era rewrote its closing line with a flourish of its own, and another left a second copy of
its event under a name it made up.

**A document is committed whole.** A mission's document below its floor (`Work::min_words` —
250 words for a life event, 380 for a story) is refused at `bench_commit`, while it can still be
finished.

**A refused document is work still to do** (`Mission::reopen_write`). When the gate refuses an
operation's document at a report, its writing step is reopened — or, for a stage that had none,
"change … and commit it" is added before the report — so the compass sends the Maker back to its
desk. With every step still ticked, Makers were told in one breath to write the draft again and
that the work was done and to report it now, at a table two levels from the only desk they could
write at; they reported, were refused, and ended on a bench telling each other there was nothing
left to do.

**An operation's Maker reads, writes and commits — nothing else at the bench.** While a body
carries any stage of an operation, the bench's working-set verbs (`bench_stash`,
`bench_stash_pop`, `bench_stage`, `bench_unstage`, `bench_restore`, `bench_branch`,
`bench_blame`, `bench_log`, `bench_diff`) are not offered (`mission_acts::NOT_FOR_OPERATIONS`).
Reviewers given them stashed their own mend, committed nothing ("you have nothing open to
merge"), popped, staged and restored round and round, and one cut a story to 155 words in the
churn.

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
