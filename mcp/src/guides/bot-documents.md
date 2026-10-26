---
name: bot-documents
description: How to have a bot write or edit a document — a brief that names the document, its purpose, sources and shape; the bot's report; and a review of what it wrote before anyone relies on it.
tools: [bot_mission_give, bot_mission_read, bot_message_send, bot_message_read, task_start, task_wait]
---
# Bots writing documents

Bots can read and edit the documents of their world — reports, records, world
material — working a document in parts. A document is a mission like any
other (`bot-mission`); what changes is the brief.

## The brief

```
<What to write or change>: <the document, by name or path>.
Purpose: <who reads it and what for>.
Sources: <what it is to be drawn from — documents in the world, people to
ask, machines to read>; nothing it did not read goes in.
Shape: <sections, length, tone>.
Done when: <the condition>; then report what you wrote, section by section,
and anything you could not find.
```

Steps, when the work has an order: "read the plant-room log", "ask Ione about
the valve fault", "write the incident section", "go back to the table and
report".

## While it works

`bot_mission_read {bot}` to follow the steps; `bot_message_send` to answer it
or add something it needs; `bot_message_read` for what it said back.

## Review before relying on it

What a bot writes is a draft. Read the document, or delegate a review to a
worker (`task_start` with `effort: "exhaustive"`): does each statement come
from the sources named, is the shape what was asked, is anything invented?
Fix it with a follow-up mission, or note what is wrong for the user.

## From research

To turn web research into world material, verify the research first
(`research`, `verify`), then give the bot the verified findings as its source
(`research-to-bots`).
