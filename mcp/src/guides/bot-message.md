---
name: bot-message
description: How to talk to a bot directly — a message on its handset, answered in its own time, read back from the thread.
tools: [bot_message_send, bot_message_read]
---
# Messaging a bot

`bot_message_send {bot, text}` sends it a message on its **handset**, the
phone it carries in its world, as you. Bots message each other the same way,
so it reads yours as it would anyone's and answers when it gets to it.

- **It answers in its own time** — minutes, not seconds, and maybe not at all
  while it is busy: the send's answer says how many messages are unread on its
  handset, and a bot with a backlog may not reach yours soon. Send, then read
  the thread later with `bot_message_read {bot}`; do not expect the reply in
  the answer to the send. For something that must be done or answered, give a
  mission instead — a bot always takes up its mission.
- **The answer says it was delivered** — "Sent to <bot> as <you>. N unread on
  its handset." — and adds "It carries no handset, so it cannot answer" when
  that is so: the message is delivered, but no reply will come.
- **Be specific**: a question it can answer from where it is, or an
  instruction it can act on. For work with steps and a report, give a mission
  instead (`bot-mission`).
- **Many bots at once**: post on the world's channel (`bot_channel_post`,
  Creator role) rather than messaging each.
