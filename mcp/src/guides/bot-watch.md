---
name: bot-watch
description: How to follow what your bots are doing over time — states and moods across the roster, missions open and finished, what the world's channel carried.
tools: [bot_list, bot_mission_read, bot_channel_read, bot_message_read]
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
findings, so the next one says what changed. For a long watch, hand the
polling to a worker (`watch`) with the bot tools named in its prompt, and
have it report only exceptions — a mission thwarted, a bot gone idle
mid-mission.
