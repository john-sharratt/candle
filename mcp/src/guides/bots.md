---
name: bots
description: What bots are and how to work with them — list them, read what each is doing, and pick the guide for what you want of them. Read before any other bot guide.
tools: [bot_list, bot_mission_read, bot_message_read, bot_channel_read]
---
# Bots

A **bot** is an autonomous agent living in a world. It acts on its own: it
moves through rooms, uses the machines there, talks to other bots, carries
missions and reports on them. You manage it — you do not drive it step by
step. It works in its own time, so what you ask of it shows up minutes later,
not in the answer to your call.

## When the bot tools are missing

The `bot_*` tools come from the bots' own service. While it is restarting or
down, a new conversation lists no `bot_*` tool at all — and a call already
listed answers that it is unavailable. `cluster_status` shows it: the `npcd`
line with no member ready. Tell the user, and try again in a few minutes; the
guides stay readable meanwhile.

## Find them

`bot_list` — your bots, each with its id (what every other `bot_*` tool takes
as `bot`), name, world, state (active, idle, asleep, suspended) and mood.
Filter by `state`, `world`, or text in the name (`q`).

## See what one is doing

- `bot_mission_read {bot}` — the mission it carries (its brief, each step and
  how it went), the last one it finished (its answer and report), and how
  many more are queued.
- `bot_message_read {bot}` — your message thread with it.
- `bot_channel_read {world}` — what its world's group chat carried.

An idle bot runs **routines** of its own — reading one station after another,
a few minutes each — and a routine shows in `bot_mission_read` as the mission
it carries, just like one you gave. Giving a mission with `start: true`
(the default) replaces whatever it carries, routine or not; there is no need
to wait for a routine to end.

Missions themselves — giving, adjusting, cancelling — are `bot-mission`'s:
`bot_mission_give`, `bot_mission_step`, `bot_mission_cancel`.

## What next

| You want | Guide |
|---|---|
| A bot to do something and report | `bot-mission` |
| A bot to write or edit a document | `bot-documents` |
| Several bots working one goal | `bot-campaign` |
| To talk to a bot | `bot-message` |
| To follow bots over time | `bot-watch` |
| What came of a finished mission | `bot-debrief` |

## Notes

- You reach only your own bots; another person's are not shown to you.
- A bot not in a world has no body: it cannot carry a mission or a message.
- Posting on a world's channel needs the game's Creator role.
