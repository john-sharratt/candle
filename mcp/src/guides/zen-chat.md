---
name: zen-chat
description: How to talk to Zen directly on a private chat — open one, ask, read the answer, close it — and when a chat is the wrong tool and a task is the right one.
tools: [zen_chat_send, zen_chat_read, zen_chat_close]
---
# Talking to Zen on a chat

A **chat** is a private conversation between you and Zen — the user's own
model, on the user's hardware, with their repositories, journals and the web to
read. It is named by its id, which the send that opened it answers with, and it
is deleted, with everything said on it, when you close it or after twelve hours
with nothing said. Nothing on a chat is kept: no thread, no journal entry, no
memory.

## The chat's id is yours alone

Your connection to Zen is shared by every conversation the user has through
this connector, so nothing but the id tells your chat from another's.

- **Open your own chat.** Send with no `chat` for each conversation or subject
  you start.
- **Pass its id on every later call** — `zen_chat_send`, `zen_chat_read`,
  `zen_chat_close`. Every answer names it: ``Chat `<id>` with Zen``.
- **Never use an id you did not open in this conversation**, even one you find
  in memory or another session's notes: that chat is someone else's
  conversation.

## When to chat, and when not to

- **Chat** to ask Zen something and use the answer now: what a file does, where
  something is handled, what it would do, a second opinion on a plan.
- **Start a task instead** (`repo_task_start`, the `delegate` guide) for work that
  changes a repository, takes long, or must be checked and kept: a task's result
  is its journal entry, with the evidence behind it, and outlives the chat.
  A chat's answer does not.
- **Never put on a chat what must last.** If an answer matters later, note it
  where it belongs (`journal_note`) yourself.

## Ask

1. `zen_chat_send {text}` — no `chat` opens a new one; the answer names its id.
   Say the whole question: Zen reads only this chat, not your context.
2. It waits up to `wait_s` (45 by default) and answers with Zen's reply. If Zen is
   still answering, it says so and what Zen is doing: call `zen_chat_read
   {chat}` to wait for the reply.
3. Ask the next thing with `zen_chat_send {text, chat}`. One message is answered
   at a time: a send while Zen answers is refused.

`effort` is `balanced` by default, which answers soonest; ask `deep` for a
question that weighs evidence. Check what Zen cites before you rely on it, as
you would any worker's answer.

## Read

`zen_chat_read {chat}` with no `after` gives what Zen said since your last
message. `after: 0` gives the whole chat; each message is numbered, so pass the
last number you read to page on.

## Close

`zen_chat_close {chat}` when you are done: Zen's conversation is tombstoned and
every message ceased. A chat left alone is deleted twelve hours after its last
message. A call naming a deleted chat says it is not open: send with no `chat`
to start again.

## Speaking in one of the user's conversations

When the user asks you to say something in one of their conversations in the
web UI, pass that conversation's id — what its `#conv=` link names — as `chat`.
`zen_chat_send {chat, text}` is then the conversation's next turn, exactly as if
the user had typed it, at the conversation's own dials, and kept with it;
`zen_chat_read {chat}` reads it from the user's last message on. Only a
conversation the user can open is reached, and `zen_chat_close` never takes
one.

- **Only when the user asks.** It is their conversation: say in it what they
  asked you to say, and nothing on your own account.
- **Say the whole instruction.** The conversation reads its own history, not
  your context.
- **Read the reply before you report it**, and check what it says it did against
  what it cites, as you would any worker's answer.
