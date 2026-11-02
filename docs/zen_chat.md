# Zen chat: a private conversation for the life of a session

Claude, in any session that reaches zend through `/mcp`, can talk to Zen
directly: send it a message, read what it answers, and close the chat. The chat
belongs to the MCP session that opened it, nothing else can address it, and when
the session is done it is **disposed of**: its conversation tombstoned, and every
record it made ceased. It is the one thing `/mcp` makes that is not kept.

This is the opposite of a thread (`docs/claude_zend_gateway.md` §4). A thread is
the record of a feature's work, and archiving it keeps the journal and every
worker's evidence. A chat is a conversation, and disposing of it keeps nothing.

## 1. Tools

| Tool | Does | Returns |
|---|---|---|
| `zen_chat_send {text, effort?, wait_s?}` | Sends `text` to Zen on this session's chat, opening the chat on first use, and waits up to `wait_s` (45 by default) for the answer. Zen thinks at `effort` — `balanced` by default, so a chat answers soonest | Zen's answer, or that Zen is still answering and what it is doing |
| `zen_chat_read {after?, wait_s?}` | Reads the chat's messages after message `after`; with no `after`, Zen's messages since Claude's last one. While Zen is answering and nothing new is there, waits up to `wait_s` (45 by default) | The messages, each numbered, and whether Zen is still answering |
| `zen_chat_close {}` | Disposes of the chat now. A Zen answer still running is stopped at its next tool round and the chat is disposed of as it ends | `disposed`, or `closing` until the answer stops |

`zen_chat_read` only reads (`readOnlyHint`): it writes nothing, not even a read
cursor, so the caller passes `after` to page on (MCP request-path invariant 2,
`CLAUDE.md`). It runs no model and is in the latency gate
(`zend/src/mcp/latency_tests.rs`).

A send while Zen is still answering is refused, naming `zen_chat_read`: one
message is answered at a time, so a reply is never interleaved with a second
question it has not read.

## 2. What a chat is

- **The chat's record** is `mcp/chat/<owner>/<id>`: its id (16 hex digits,
  random), the MCP session it belongs to, when it opened, when the last message
  was said, and whether it is closing (`zend/src/mcp/chats/record.rs`).
- **Its messages** are records under `mcp/chat-said/<owner>/<id>/`, one per
  message, keyed by time and a sequence like a thread's notes, so key order is
  the order they were said. A message is who said it — `claude` or `zen` — its
  text, and whether it is a failure Zen's turn ended in. A message's number is
  its place in that order, from 1.
- **Zen's side** is a worker (`docs/claude_zend_gateway.md` §4) of kind `chat`
  under the thread `~<id>`, named `zen`. A thread name starting `~` is no name a
  person can give (`names::check` refuses it), so no `T:` path addresses a
  chat's thread or its worker, and no thread a person attaches can collide with
  one. Its conversation is `mcp:<owner>:~<id>/zen`.
- **The session holds the chat's id** under `chat` (`zend/src/mcp/state.rs`), as it holds
  the thread `thread_attach` bound it to. A 2026-07-28 session through the
  aggregator is keyed by the aggregator's own session (`docs/mcp_app.md` §3),
  so each client session — each Claude Code session — has a chat of its own.

A chat worker runs every turn exactly as a task's does — the same submit, the
same move to a faster member (§6.3 there), the same resume after a restart — with
four differences, all keyed on its kind (`WorkerKind::Chat`):

1. **Its message is the text as sent.** No question block is appended: Zen
   answers Claude in the chat itself, so a chat worker never settles as a
   question.
2. **No journal draft and no citation audit.** A task's result is its journal
   entry; a chat's is its answer, which is ready when the turn ends. Neither the
   draft's wait nor an audit send-back stands between Claude and the reply.
3. **No line on a thread's journal.** The notes `workers` enters for a task are
   not entered for a chat (`workers::note`), so the chat's thread holds no copy
   of what was said.
4. **Its answer is a message.** As its turn settles, the reply — or the failure,
   said as one — is added to the chat's messages before its record says it has
   ended, so a reader woken by the record finds the message there.

The `chat` option of the `role` selector (`zend/src/prompts/projection.yaml`)
tells Zen what the conversation is: Claude writing to it from a session of its
own, answered plainly, the chat deleted afterwards. A chat worker runs with the
tools of the caller's role, like a task.

A failed turn's worker is not continued, and — unlike a failed task — its
conversation is not distilled: it is left as it is until disposal tombstones it.
The next send starts a fresh worker on the same chat (`zen.2`), shown the chat's
last messages, and the messages already said stay readable.

The session's chat is read from the session's record, and a new chat held on it,
under the same lock a send begins its turn under, so two sends at once on a
session that holds none open one chat between them, and the second is refused
while Zen answers the first. The chat's record is written before anything else a
send writes, so the sweep finds whatever a failed send left behind.

## 3. When it is disposed of

A chat is disposed of when any of these holds:

- **Claude closes it** with `zen_chat_close`.
- **Its session is gone**: the session's record is ceased — by a handshake
  client's `DELETE`, or a month unused — or the session holds another chat.
- **Nothing has been said on it for `CHAT_IDLE_MS`**, twelve hours. A client
  is not obliged to end its session, and Claude Code through claude.ai's
  connectors never says when it has, so idleness is what ends most chats. A send
  after that opens a new chat, and the answer says the earlier one was disposed
  of.

The sweep (`zend/src/mcp/chats/sweep.rs`) runs on every member every ten
minutes. A chat whose session record is not found is given ten minutes from its
opening before it counts as gone, so a record still on its way to this member is
not taken for a session that ended.

**Disposing** (`zend/src/mcp/chats/dispose.rs`) reads, judges and marks the chat
under the lock a turn begins under (`workers::holding`), and does the rest
outside it:

1. Under the lock, a chat the sweep found due is judged again — one spoken on
   since the sweep read it is kept — and the chat is marked closing. If Zen is
   answering, the answer is stopped at its next tool round (`stops::request`)
   and nothing else happens yet: the chat is disposed of as the answer settles,
   or by the next sweep. A closing chat with nothing answering can have no turn
   begun on it again — a send opens a new chat, and a lost answer is not
   resumed (`resuming::chat_open`, read under the same lock) — so what follows
   holds no lock, and runs off the async runtime's threads.
2. Every worker's conversation is tombstoned (`ZendSession::tombstone_conversation`):
   the timeline is hidden at once and its turns, KV and provenance are reclaimed
   at the next compaction; its journal text is ceased with it.
3. Every record is ceased: each worker's, its nudges, its stop, its reported
   mark, the thread's notes and record, the messages, and the chat's own.

A tombstone that fails — the model still loading — leaves every record in place,
the chat closing, for the next sweep to finish. Two members disposing of one chat
at once tombstone it twice, which is harmless.

## 4. What is never kept

- **No episode.** The aggregator records an episode on the user's memory for each
  call that does more than read (`docs/mcp_user_context.md` §2.3), summarised by
  its first string argument — which for `zen_chat_send` is the message. The chat's
  tools are never episodes (`mcp/src/context.rs`, `CHAT`).
- **No thread.** A chat's thread is left out of `thread_list`, the idle archive
  sweep (`archiving::sweep`) never distils it — disposal is its end — and no tool
  that takes a path can name it.
- **No borrowing.** The dialogue layer populates only a conversation's own group
  (`dialogue/primary_conversation`, `projection.yaml`), so a chat's turns are
  never placed into another conversation's prompt while it lives.
- **Logs carry no text.** `mcp: handled` and `mcp: called` name the tool and its
  time, never its arguments.

What the chat is private *from* is every other session and conversation; it is
not hidden from its owner's own view of zend, where `GET /v1/conversations` lists
a person's worker conversations to them and to Admins (`docs/claude_zend_gateway.md`
§3.2), until it is tombstoned.

## 5. Tests

- `chats/record.rs`: a chat and its messages read back as written, in order,
  numbered from 1; the keys an owner's chat is kept under.
- `chats/talk.rs`: the first send opens a chat and a `zen` worker of kind chat;
  a later send continues the same worker; a send while Zen answers is refused; a
  send after a failed turn starts `zen.2`; a send on a disposed chat opens a new
  one and says so; a read without `after` gives Zen's messages since Claude's
  last; a read writes no record.
- `chats/dispose.rs`: disposal ceases every record the chat made and tombstones
  every worker's conversation, leaving another chat's and a thread's records
  untouched; a chat with Zen answering is marked closing and not disposed of; a
  failed tombstone ceases nothing.
- `chats/sweep.rs`: a chat is due when closing, idle twelve hours, or its session
  gone past the grace — and not otherwise.
- `workers.rs`: a chat turn settles done with its reply, never as a question; a
  chat worker's notes are not entered.
- `mcp/src/context.rs`: no chat tool is ever an episode.
