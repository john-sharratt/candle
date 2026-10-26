---
name: bot-campaign
description: How to have several bots work one goal together — split it, one mission each, frame it on the world's channel, track every mission, collect the reports.
tools: [bot_list, bot_mission_give, bot_mission_read, bot_mission_cancel, bot_channel_post, bot_channel_read, bot_message_send]
---
# A campaign of bots

## Plan

1. `bot_list {world}` — who is available, their states.
2. Split the goal into parts one bot can do alone, with what each part needs
   from the others. Write the plan on a thread so it survives your context.
3. Assign parts to bots; check each one's current mission first
   (`bot_mission_read`) — a new mission replaces it.

## Launch

- Frame it, if you have the Creator role: `bot_channel_post {world, text}` —
  every bot in the world hears the goal and who does what.
- One `bot_mission_give` per bot, each brief naming its part, who it works
  with, and what its report must say (`bot-mission`).

## Track

`bot_mission_read` on each, every few minutes; `bot_channel_read` for what
they said to each other. A bot stuck on its part: add a step, message it, or
cancel and reassign.

## Collect

Each bot's report and answer (`bot-debrief`), assembled into one result for
the user: what each part found, what is missing, where reports disagree.
