---
name: research-to-bots
description: How to turn web research into material bots work with — verify the research first, then have a bot write it into its world, then review what it wrote.
tools: [task_start, task_wait, task_reply, bot_list, bot_mission_give, bot_mission_read, bot_message_send]
---
# From research to the bots' world

1. **Research and verify** — `research`, then `verify`. Only what is
   confirmed moves on; what is unverified is marked as such in what the bot
   is given.
2. **Choose the bot and the document** — `bot_list`; the document it will
   write or update, by name.
3. **Give the mission** (`bot-documents`): the brief names the document, its
   purpose and shape, and the verified findings as its only source — included
   in the brief, each with its URL.
4. **Follow it** — `bot_mission_read`; answer what it asks.
5. **Review what it wrote** against the findings: nothing added that the
   research did not support, nothing confirmed dropped.
6. **Record** on the thread what was researched, what went into the world,
   and where.
