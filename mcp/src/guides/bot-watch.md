---
name: bot-watch
description: How to follow what your bots are doing over time — states and moods across the roster, missions open and finished, what the world's channel carried.
tools: [bot_list, bot_mission_read, bot_channel_read, bot_message_read, journal_note]
---
# Watching bots

## A snapshot

1. `bot_list` — every bot's mood, and its state: asleep or suspended bots
   cannot work. The list says nothing of missions, and "idle" does not mean
   not working (`bots`).
2. `bot_mission_read` for each bot — which ones carry a mission, how far
   along, what each last finished. Each answer holds the whole brief, so
   keep your notes to the mission's goal, its step count and its report.
3. `bot_channel_read {world}` — what the world's channel carried since you
   joined it.
4. Summarise: who is working on what, what finished, what looks stuck,
   anything unusual.

No answer carries a time. "Changed" and "finished since you last looked" mean
against your previous snapshot, so the first snapshot is a baseline: say so,
rather than calling a mission stuck from one reading.

## Over time

Keep a thread for the world or the campaign and note each snapshot's
findings (`journal_note`), so the next one says what changed. When the
coding tools are missing, keep the snapshot in your answer to the user
instead. The bot tools
are yours alone — Zen's workers cannot call them — so a watch is your own
snapshots, taken again: every few minutes while a mission is moving, an hour
or more apart for a quiet roster. Report only exceptions — a mission
thwarted, a bot gone idle mid-mission, a mood that turned.
