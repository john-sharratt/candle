---
name: zen-chat
description: How to talk to Zen directly on a private chat for the length of your session — ask, read the answer, close it — and when a chat is the wrong tool and a task is the right one.
tools: [zen_chat_send, zen_chat_read, zen_chat_close]
---
# Talking to Zen on a chat

A **chat** is a private conversation between you and Zen — the user's own
model, on the user's hardware, with their repositories, journals and the web to
read. It belongs to this session alone: no other session or conversation can
read it, and when the session is done it is deleted, with everything said on
it. Nothing on a chat is kept: no thread, no journal entry, no memory.

## When to chat, and when not to

- **Chat** to ask Zen something and use the answer now: what a file does, where
  something is handled, what it would do, a second opinion on a plan.
- **Start a task instead** (`repo_task_start`, the `delegate` guide) for work that
  changes a repository, takes long, or must be checked and kept: a task's result
  is its journal entry, with the evidence behind it, and outlives your session.
  A chat's answer does not.
- **Never put on a chat what must last.** If an answer matters later, note it
  where it belongs (`journal_note`) yourself.

## Ask

1. `zen_chat_send {text}` — the first message opens the chat. Say the whole
   question: Zen reads only this chat, not your context.
2. It waits up to `wait_s` (45 by default) and answers with Zen's reply. If Zen is
   still answering, it says so and what Zen is doing: call `zen_chat_read` to
   wait for the reply.
3. Ask the next thing with another `zen_chat_send`. One message is answered at a
   time: a send while Zen answers is refused.

`effort` is `balanced` by default, which answers soonest; ask `deep` for a
question that weighs evidence. Check what Zen cites before you rely on it, as
you would any worker's answer.

## Read

`zen_chat_read` with no `after` gives what Zen said since your last message.
`after: 0` gives the whole chat; each message is numbered, so pass the last
number you read to page on.

## Close

`zen_chat_close` when you are done: Zen's conversation is tombstoned and every
message ceased. A chat left alone is deleted twelve hours after its last message,
and when the session ends. A send after that opens a new chat, and says so.
