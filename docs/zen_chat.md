# Zen chat: a private conversation, deleted when it is done

Claude, in any session that reaches zend through `/mcp`, can talk to Zen
directly: open a chat, send it messages, read what Zen answers, and close it. A
chat is named by an id the send that opened it answers with, belongs to the
person who opened it, and when it is done it is **disposed of**: its
conversation tombstoned, and every record it made ceased. It is the one thing
`/mcp` makes that is not kept.

**A chat is never tied to the MCP session it was opened on.** claude.ai keeps
one MCP session per person across every conversation that person has through
its connector — two Claude Code sessions, started apart, call on the same one
(measured 2026-11-02: a second headless session's first send landed as message
#3 of the first session's chat). Nothing on the wire tells one Claude Code
session from another, so the id is the only thing that can: each conversation
opens its own chat and passes its id on every later call.

This is the opposite of a thread (`docs/claude_zend_gateway.md` §4). A thread is
the record of a feature's work, and archiving it keeps the journal and every
worker's evidence. A chat is a conversation, and disposing of it keeps nothing.

## 1. Tools

| Tool | Does | Returns |
|---|---|---|
| `zen_chat_send {text, chat?, effort?, wait_s?}` | Sends `text` to Zen on the chat `chat` names, or opens a new chat when it names none, and waits up to `wait_s` (45 by default) for the answer. Zen thinks at `effort` — `balanced` by default, so a chat answers soonest. `chat` may instead name a web UI conversation (§6) | The chat's id, Zen's answer, or that Zen is still answering and what it is doing |
| `zen_chat_read {chat, after?, wait_s?}` | Reads the chat's messages after message `after`; with no `after`, Zen's messages since Claude's last one. While Zen is answering and nothing new is there, waits up to `wait_s` (45 by default). `chat` may instead name a web UI conversation (§6) | The messages, each numbered, and whether Zen is still answering |
| `zen_chat_close {chat}` | Disposes of the chat now. A Zen answer still running is stopped at its next tool round and the chat is disposed of as it ends | `disposed`, or `closing` until the answer stops |

Every answer names the chat (``Chat `<id>` with Zen: …``), so the id is in
front of Claude whenever it needs it. `chat` is checked for a chat id's shape —
16 lowercase hex digits — before anything is read. A chat that is closing, was
disposed of, or is another person's is answered alike: no chat by that id is
open, everything said on it is gone, and a send naming no chat opens a new one.

**One person keeps at most `MAX_CHATS` chats, sixteen** (`chats/talk.rs`). Each
holds a conversation and a worker until it is closed or idle twelve hours, so a
caller that loses its ids and opens a chat on every send is refused once it has
sixteen, told to go on with one by its id or close those it is done with. A
closing chat counts until it is disposed of.

`zen_chat_read` only reads (`readOnlyHint`): it writes nothing, not even a read
cursor, so the caller passes `after` to page on (MCP request-path invariant 2,
`AGENTS.md`). It runs no model and is in the latency gate
(`zend/src/mcp/latency_tests.rs`).

A send while Zen is still answering is refused, naming `zen_chat_read`: one
message is answered at a time, so a reply is never interleaved with a second
question it has not read.

## 2. What a chat is

- **The chat's record** is `mcp/chat/<owner>/<id>`: its id (16 hex digits,
  from the OS's randomness, so it is known only to the call it was answered
  to), when it opened, when the last message was said, and whether it is
  closing (`zend/src/mcp/chats/record.rs`). Every key starts with the owner, so
  one person's id never reaches another's chat.
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
tools of the caller's role, like a task, and acts as a coder does: it deploys,
restarts and runs Care commands, and it starts and steers tasks of its own,
which a task never does (`docs/zen_tool_reach.md` §6.1, §6.3).

A failed turn's worker is not continued, and — unlike a failed task — its
conversation is not distilled: it is left as it is until disposal tombstones it.
The next send starts a fresh worker on the same chat (`zen.2`), shown the chat's
last messages, and the messages already said stay readable.

A send reads the chat it names under the same lock a send begins its turn under
and a disposal marks a chat closing under, so two sends at once on one chat begin
one turn — the second is refused while Zen answers the first — and no send lands
on a chat being disposed of. Two sends naming no chat open two chats. The chat's
record is written before anything else a send writes, so the sweep finds
whatever a failed send left behind.

## 3. When it is disposed of

A chat is disposed of when any of these holds:

- **Claude closes it** with `zen_chat_close`.
- **Nothing has been said on it for `CHAT_IDLE_MS`**, twelve hours. A client
  is not obliged to say a conversation has ended, and Claude Code through
  claude.ai's connectors never does, so idleness is what ends a chat Claude does
  not close. A send naming it after that is answered that it is not open.

A session's end is no trigger: a claude.ai session outlives every conversation
on it, so it would end a chat later than idleness does or never.

The sweep (`zend/src/mcp/chats/sweep.rs`) runs on every member every ten
minutes.

**Disposing** (`zend/src/mcp/chats/dispose.rs`) reads, judges and marks the chat
under the lock a turn begins under (`workers::holding`), and does the rest
outside it:

1. Under the lock, a chat the sweep found due is judged again — one spoken on
   since the sweep read it is kept — and the chat is marked closing. If Zen is
   answering, the answer is stopped at its next tool round (`stops::request`)
   and nothing else happens yet: the chat is disposed of as the answer settles,
   or by the next sweep. A closing chat with nothing answering can have no turn
   begun on it again — a send naming it is refused, and a lost answer is not
   resumed (`resuming::chat_open`, read under the same lock) — so what follows
   holds no lock, and runs off the async runtime's threads. A send naming it is
   refused as not open from the moment it is marked.
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

What the chat is private *from* is every other person, and every conversation of
its owner's that was not handed its id. The id is the boundary within one
person's sessions, so Claude never passes an id it did not open in this
conversation; the guide (`mcp/src/guides/zen-chat.md`) says so. It is not hidden
from its owner's own view of zend, where `GET /v1/conversations` lists a
person's worker conversations to them and to Admins
(`docs/claude_zend_gateway.md` §3.2), until it is tombstoned.

## 5. Tests

- `chats/record.rs`: a chat and its messages read back as written, in order,
  numbered from 1; the keys an owner's chat is kept under.
- `chats/talk.rs`: a send naming no chat opens one and a `zen` worker of kind
  chat; a later send continues the same worker; a send while Zen answers is
  refused; a send after a failed turn starts `zen.2`; a second send naming no
  chat opens a second chat, apart from the first; a send naming another
  person's chat, or one closing, is refused as not open; a read without `after`
  gives Zen's messages since Claude's last; a read writes no record; every
  answer names its chat; a send opening a seventeenth chat is refused, and the
  count is each person's own.
- `tools/zen_chat_send.rs`: a `chat` that is not a chat id — a number, `null`,
  empty — is refused, never taken for none.
- `latency_tests.rs`: `zen_chat_read` on a chat opened through `zen_chat_send`,
  by the id its answer gives, within the budget.
- `chats/naming.rs`: only a chat id's shape is taken for one.
- `chats/dispose.rs`: disposal ceases every record the chat made and tombstones
  every worker's conversation, leaving another chat's and a thread's records
  untouched; a chat with Zen answering is marked closing and not disposed of; a
  failed tombstone ceases nothing.
- `chats/sweep.rs`: a chat is due when closing or idle twelve hours — and not
  otherwise.
- `workers.rs`: a chat turn settles done with its reply, never as a question; a
  chat worker's notes are not entered.
- `mcp/src/context.rs`: no chat tool is ever an episode.
- `chats/dialogue/target.rs`: a chat's id names the chat, a web UI conversation's
  id the conversation; a task's conversation, escaped or not, is pointed at
  `repo_task_reply`; nothing else of any other shape is taken.
- `chats/dialogue/mod.rs`: a send carries the conversation's own dials, and a
  named effort overrides its effort; a read with no `after` starts after the last
  message a person wrote; a read renders each message under its number.
- `chats/dialogue/reply.rs`: a finished stream is its text and done; a failed one
  says why after what it wrote; a body cut at the deadline is read to its last
  whole frame.
- `chats/dialogue/prose.rs`: a reply loses its reasoning and says its calls; an
  unclosed block runs to the end; a round's results are counted.
- `ownership.rs`: a `/mcp` caller's viewer opens what the web routes would open
  for the same person.

## 6. Speaking in a conversation of the web UI's

`chat` may name a conversation of the web UI's by its id, as the UI's `#conv=`
link names it (`chats/dialogue/`), when the user asks Claude to speak in it. The
message is that conversation's next turn — kept, as the conversation is; nothing
of §2–§4 applies to it, and `zen_chat_close` takes only a chat's id, so it can
never dispose of one.

**A send is the web UI's own request, sent for the caller.** It goes up the
member's socket to a gateway as `POST /v1/chat/completions` for that
conversation, stating the caller's identity (`ops::up`), so it meets everything a
person's browser meets: the gateway places the turn on the member holding the
conversation's claim — or, the claim let go, on one holding the conversation
complete (`docs/cooperative_admission.md` §2.2) — and the route checks the
caller's role and that they may open the conversation. Nothing in the tool
decides where a turn runs or who may run one. The request carries the
conversation's own dials (`ZendSession::conversation_dials`), since the web UI
sends its composer's on every turn and a turn without them would reset them;
an `effort` named overrides the effort.

**Who may name one is the web routes' rule** (`ownership`): a conversation the
web UI made names no owner and opens for an Admin alone. One the caller may not
open, and one that is not there, are answered alike, so a refusal never says a
conversation exists. A task's conversation (`mcp:…`) is refused and pointed at
`repo_task_reply`, since a turn sent past its worker would leave the worker's
state behind; an archived conversation takes no turn.

A send is refused while the conversation answers — a turn live on the member
asked, or the claim held by another — naming `zen_chat_read`. A send waits up to
`wait_s` reading the turn's stream; a turn still running when the wait ends
runs on, and `zen_chat_read` reads it.

A read gives the conversation's messages as a reader wants them: reasoning
dropped, each call as `[called <tool>]`, a round's results as a count, each
message at most 6,000 characters. With no `after` it starts after the last
message a person wrote. It waits, as a chat's read does, while Zen answers and
nothing new is there; the turn's rounds reach the member asked replicated, so it
looks every two seconds.
