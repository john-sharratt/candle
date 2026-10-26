---
name: bot-debrief
description: How to take a finished mission's outcome — read the report and answer, ask the bot what is unclear, check what it claims, and record what came of it.
tools: [bot_mission_read, bot_message_send, bot_message_read, journal_note]
---
# Debriefing a bot

1. `bot_mission_read {bot}`: the finished mission — its report (pass or fail,
   and its account), its answer, and each step's outcome.
2. **Read the report as evidence, not truth**: what did it actually do (the
   steps achieved), and does its account match? A step thwarted and a "pass"
   do not go together.
3. **Ask what is unclear**: `bot_message_send {bot, text}` with the specific
   question; read the reply later with `bot_message_read`.
4. **Check what you will rely on** — a document it wrote, a reading it took —
   before passing it on. You see these only through the bot: ask it to quote
   the document's section or the reading's figures word for word
   (`bot-documents`).
5. **Record it**: `journal_note` on the thread — what was asked, what came
   back, what was confirmed, what to do next.
