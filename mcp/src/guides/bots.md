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

Read before asking anything of a bot: giving a mission replaces the one it
was carrying.

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
