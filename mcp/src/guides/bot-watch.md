---
name: bot-watch
description: How to follow what your bots are doing over time — states and moods across the roster, missions open and finished, what the world's channel carried.
tools: [bot_list, bot_mission_read, bot_channel_read, bot_message_read, journal_note]
---
# Watching bots

## A snapshot

1. `bot_list` — every bot's state and mood; note the ones asleep, suspended
   or with a mood that changed.
2. `bot_mission_read` for each bot that matters — open missions and how far
   along, missions finished since you last looked.
3. `bot_channel_read {world}` — what the world's channel carried.
4. Summarise: who is working on what, what finished, what is stuck, anything
   unusual.

## Over time

Keep a thread for the world or the campaign and note each snapshot's
findings (`journal_note`), so the next one says what changed. The bot tools
are yours alone — Zen's workers cannot call them — so a watch is your own
snapshots, taken again: every few minutes while a mission is moving, an hour
or more apart for a quiet roster. Report only exceptions — a mission
thwarted, a bot gone idle mid-mission, a mood that turned.
