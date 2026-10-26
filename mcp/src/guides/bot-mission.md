---
name: bot-mission
description: How to give a bot a mission and see it through — a brief that says what done looks like, steps it can act on, following it as it works, and adjusting or cancelling it.
tools: [bot_mission_give, bot_mission_read, bot_mission_step, bot_mission_cancel, bot_list]
---
# A bot mission

## Before

`bot_mission_read {bot}`: giving a mission with `start` (the default)
replaces whatever it was carrying. If it is mid-mission, either queue yours
(`start: false`) or wait.

## Write the brief

A bot reads its brief and its steps every turn; they are its whole
instruction. Write them as you would a task prompt:

- **brief**: what is asked, why, and what done looks like — what it must find
  out, make or change, and what its report must say.
- **steps**: the actions in order, each addressed to the bot and something it
  can do in its world: "go to the plant room", "scan the coolant valve and
  read its state", "find Paxon Vael and ask who was there", "go back to the
  table and report". With no steps, they are drawn from the rooms, machines
  and people the brief names.
- Keep one mission to one goal; several goals are several missions (queued)
  or several bots (`bot-campaign`).

`bot_mission_give {bot, brief, steps, start}` — `start: true` carries it now
and rouses the bot to read it; `start: false` queues it for the bot to collect
at the command table, the place in its world where bots take up missions.

## Follow it

`bot_mission_read {bot}` again as it works: each step and whether it was
achieved or thwarted. It works in its own time — check every few minutes, not
every few seconds.

## Adjust

- `bot_mission_step {bot, add}` — add a step it turns out to need.
- `bot_mission_step {bot, done, outcome}` — sign off a step yourself
  (achieved, or thwarted when it cannot be done).
- `bot_mission_cancel {bot}` — call it off; the bot is told. Queued missions
  are untouched.

## When it ends

Its report (pass or fail, and its account) and its answer are in
`bot_mission_read`. Read `bot-debrief` for what to do with them.
