# npcd guardian: proposal

The guardian watches each NPC's conversation from outside, decides whether the
NPC is stuck or off its mission, and steers it with the least force that works.
It is a set of Rust modules assembled by a builder from the `guardian:` section
of the YAML configuration. This proposal is grounded in live testing of the
`ask`, `mind_control`, `mission`, `mission/step` and `mission/cancel` routes
against three NPCs (Vael Fane, Pax Veridian, Ulysses Thorne) after a restart and
a day roll-over.

## What testing showed

1. **The mission is in the prompt and still loses to the scene.** After restart
   the `mission/standing` member was present in every NPC's system prompt and
   restored from the persisted `missions/<world>` object. The `mission` ask
   check nonetheless answered from recent scene history (bridge console codes,
   sensor panel, siege ammunition), not from the lodged mission. The prompt
   alone is not enough steering for this model with thinking off.
2. **Prompt order was a bug, now fixed.** `mission_intro` ("What has been asked
   of you:") rendered after the mission text because the collection was declared
   before the section in `projection.yaml`. The section now precedes the
   collection. The ask answers did not change by themselves, so ordering is
   necessary but not sufficient.
3. **One `mind_control` is enough to redirect, not to hold.** Vael, nudged with
   the ledger errand, answered the `mission` check about the ledger on the next
   check and within a few acts asked Pax who last touched it. Pax, nudged toward
   the foundry, queried a fabricator and then drifted back to the coolant
   conversation within about five acts. The effect lasts a handful of acts.
4. **Other NPCs hijack missions.** Vael's question to Pax ("who last touched the
   eastern ledger") pulled Pax off his own freshly lodged mission; Pax answered
   and reflected on the question instead. Social pull from a peer outweighs the
   mission section.
5. **Nobody ticks steps.** After many acts, including Ulysses setting turrets
   to conserve ammunition (exactly his mission), no todo item was ever marked
   done. Progress has to be detected and recorded from outside.
6. **A fresh mission takes effect within a turn when `start: true`.** Ulysses
   began on the rampart turrets immediately; Pax did not, because he was
   mid-conversation. Mission changes land fastest on an NPC with no open
   question waiting on it.
7. **Ask is cheap and read-only.** The `looping`, `lost`, `mission` and
   `context` checks are answered in free text without disturbing the main
   conversation, so they can be run before and after a nudge to measure it.
   Check names are case-sensitive and lower-case in the API.
8. **Non-ASCII in curl bodies breaks JSON on this shell** (an em dash gave
   "invalid unicode code point"). That is a property of the test shell, not of
   the guardian: its text is built in Rust and delivered in-process.

## Design (as implemented)

Code lives in `npcd/src/engine/guardian/`, one concern per file.

### Builder and configuration

`<data>/guardian.yaml`; no file, or `enabled: false`, means no guardian. Unknown
fields and unknown module kinds are refused.

```yaml
enabled: true
scan_every_secs: 45     # how often each NPC is read
cooldown_secs: 420      # least time between two things done to one NPC
settle_secs: 150        # how long a nudge is given before it is judged
modules:
  - kind: drift
  - kind: looping
  - kind: stall
    no_progress_secs: 600
  - kind: step_tracker
    confirmations: 2
  - kind: journal
escalation: [nudge, restate, flag]
```

`refresh` (the standing text as a fresh instruction) is still a rung that can be
listed; the shipped configuration leaves it out because it reads as a command
and is the most disruptive of the three.

`GuardianBuilder` (`builder.rs`) takes modules, escalation rungs and the three
durations, either fluently or `from_config`, and `build()` refuses an empty
module or rung list, a zero scan interval, or `settle > cooldown`.

A `Module` (`module.rs`) is pure: `question(&NpcView) -> Option<Question>` says
what to ask, `judge(&NpcView, Option<&str>) -> Verdict` reads the answer. A
verdict is `Healthy`, `Concern(OffMission | Looping | Stalled)`,
`TickStep(step, outcome)` or `Journal { worth }`. The runner owns every side effect. A module is shown an
`NpcView`: the mission, how long its progress mark has stood still, the stations
in the room, and the character's latest acts (`recent_acts`, rendered as the
pulse feed shows them, oldest first, read from the scheduler's tick ring).

### Modules

- **drift**: asks a closed question built from the mission ("part of what I was
  asked" / "something else"); `something else` is `OffMission`. A peer pulling
  the NPC off its errand (finding 4) reads as drift, so there is no separate
  hijack module.
- **looping**: puts no question. A character asked whether it is repeating
  itself says yes about one quiet turn, so it is judged from its acts: the same
  act (results after the arrow ignored, case ignored) four times in the last
  eight is `Looping`, unless the mission's progress mark moved since the last
  scan, or the act is `act` (a fight is repetition by nature).
- **stall**: no question; `Stalled` when an open step exists and the mission's
  progress mark (prompt, steps done, steps total) has not moved for
  `no_progress_secs`.
- **step_tracker**: asks whether the first open step is done, or was tried and
  could not be done ("yes, it is done" / "I tried and could not do it" / "not
  yet"), and signs it off only after `confirmations` consecutive identical
  answers for the same step (finding 5), and only when a recent act shares a word
  of four letters or more with the step: a claim nothing it did bears out is not
  recorded. The sign-off states how the step turned out, a `StepOutcome`:
  `achieved` or `thwarted`. It is stored on the step (`Todo::outcome`), returned
  in the mission API's `todo[].outcome` (`null` while open; the operator's
  `POST .../mission/step` takes `outcome`, default `achieved`), logged as
  `tick` with the outcome, and shown in the GUI's Mission tab as a green tick or
  a red cross. The character's own step list reads `[done]` or `[could not be
  done]`. It never asks about or ticks
  the step that is the report back (`Todo::reports`): going back to the table is
  not reporting, so that step stays open until the character files `report_done`
  or `report_stuck`, and the nudges name it as the next thing.
- **journal**: asks, when a stretch of turns is not in the journal yet
  (`NpcView.journal`, from `JournalState::waiting`), whether, given what happened
  and what the journal in its system prompt already says, a new entry is needed,
  and for a sentence of why. `Journal { worth: true }` has the runner draft the
  stretch (`Runtime::decide_journal`, `docs/journal.md` §2); `worth: false` closes
  the stretch, so the next question is only about what has happened since. An
  unreadable answer is `Healthy` and the stretch stays waiting. It is a
  background question like the others and changes nothing the character reads
  until an entry is kept. Without `kind: journal` in `guardian.yaml`, nothing is
  journaled.

### Escalation ladder

`ladder.rs` is a pure state machine over a caller-supplied clock. A concern takes
the current rung; the NPC is then left alone for `settle_secs` (no asks, no
pressure). After settling, a well NPC records `held` and resets to the first
rung; an unwell one records `failed` and moves up a rung once `cooldown_secs`
has passed since the last act. `flag` is raised once and nothing more is done
until the NPC is seen well.

Rung wording is built in `nudge.rs`:

1. **nudge**: a `mind_control` thought restating the ask and the next open step
   ("The errand you were given is still yours: ..."); for looping, "You keep
   covering the same ground..." with the ask, closing with the way to put a
   blocker on the record: if what it needs is not to be had and nobody can give
   it, `invoke` the order table's `report_stuck` with what stopped it (the
   address when the table is in the room, otherwise "go back to the table").
2. **restate**: names the pull of others: "Whatever else is asking for you, what
   you were sent to do still stands: ...".
3. **refresh**: the mission's standing text, delivered as a `Nudge` event.
4. **flag**: nothing is said to the NPC; the record is the signal.

`nudge` and `restate` also name the station to act on. Each scan reads the
stations in the NPC's room (`Probe::stations`: name, address, verbs). When the
open step (or, with none, the ask) shares a word of four letters or more with a
station's name or verbs, the text ends "You can do it from the bridge console:
`invoke` http://local/tower/bridge~0/command_tower." A station the step is not
about is not named.

Without a mission only looping NPCs get a generic "do something different" line.

### Before / after checks

The asks that raise a concern are the "before"; the same asks run on the first
scan after the settle window are the "after". `held` and `failed` records are
that comparison.

### Operator log

`GET /v1/pulse/guardian` (admin) returns `{enabled, records}`, a ring of the last
500 records `{at_ms, npc_id, kind, detail}` with kinds `check`, `tick`, `nudge`,
`restate`, `refresh`, `flag`, `held`, `failed`. Ladder state is in memory and
starts over on restart.

## Open questions for the next test round

- How long does a nudge hold when repeated at the cadence the guardian would use?
- Does restating the mission in the nudge hold longer than a question about the
  next step?
- Does raising the mission section's priority, or moving it after the scene
  sections in the prompt, reduce the drift without any nudge?
- Does a smaller scene window reduce drift (listed earlier as an experiment)?
