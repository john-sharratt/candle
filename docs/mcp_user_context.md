# User context: memory and a per-client guide over MCP

**Status: built** (2026-10-29). Extends `docs/mcp_app.md` (the aggregator),
`docs/mcp_guides.md` (the guide library) and `docs/claude_zend_gateway.md`
(zend's tools). Where this document and those disagree about user context,
this one is authoritative — except on what the memory *is*: the user's journal
is the memory backbone's tree (`docs/memory_backbone.md`), which replaced the
flat `U:` journal on 2026-10-30.

## 1. Why

Grok, connected to the connector from the user's car, calls the tools well —
it draws a cat with `image_generate` — but every voice conversation starts
from nothing. Nothing the host remembers reaches a connector: MCP keeps the
conversation with the host, and no client (Grok, ChatGPT, Claude) hands its own
memory to a tool server. Nor can the server tell where one conversation ends:
an MCP session is a connection, not a conversation, and the 2026-07-28
revision has no sessions at all. And Grok delivers only tools — nothing in its
documentation says it reads a server's `instructions`, prompts or resources —
so, as for claude.ai (`docs/mcp_guides.md` §1), **the tools are the only
channel**.

Every memory server surveyed (Mem0, Zep, Letta, Supermemory, the reference
knowledge-graph server) stores memory well and depends on a system prompt to
make the model *recall* it. A remote server has no system prompt. So recall
here does not wait on the model: the server writes what happened, and hands
the context over on whatever tool the model calls first.

## 2. What is built

| Piece | Where | What |
|---|---|---|
| The user's memory, `~` and `@family` | zend | The memory backbone's tree (`docs/memory_backbone.md`): the person's own journals and their family's, beside every thread's and project's. Needs no attach. |
| `journal_note` / `journal_read` / `journal_grep` on a memory path | zend | The journal tools take `~/…` and `@family/…` beside `T:` and `P:`. |
| `journal_forget {journal, says}` | zend | Removes the one line of a memory journal that says `says`. Thread and project journals stay append-only. |
| `get_user_context` | zend, decorated by the aggregator | The conversation's frame: its own journal, each above it, the person, each family — within the client's budget. The aggregator puts the client's guide in front of it. |
| `user_episode_note` | zend, called by the aggregator only | Records one episode on the conversation's journal: a tool call that did something. In no listing; the aggregator refuses it from a client. |
| `readOnlyHint` | zend, npcd | The MCP annotation on every tool that only reads: what the aggregator tells an episode by. |
| The client's guide | aggregator (`clients:` in a guide's frontmatter) | The client's "system prompt addition": how to work with the user there — in the car, spoken answers. |
| Context on the first call | aggregator | The first call of a conversation, whatever tool it is, carries the user context after its own answer. |
| `Client::Grok` | `mcp_server` | The client a session is for, read from `clientInfo.name`. |

### 2.1 The user's memory

The memory is the backbone's tree (`docs/memory_backbone.md` §2): the person's
own journals under `~` — their profile, a journal per client and per
conversation, the people in their life — and their family's under `@family`.
Its lines are of two kinds, set by who writes:

- **notes** — a fact the model was asked to keep, or judged worth keeping:
  "the cat's name is Miso", "prefers watercolour". Written with
  `journal_note {journal: "~", text}` from any client, or onto a family journal
  every member reads. Removed only by `journal_forget`, which names a line by
  what it says or by its address.
- **episodes** — one tool call that changed something, recorded by the server
  (§2.3) on the conversation's own journal.

Each line records who wrote it, for whom and through which client, which is
what lets the context put this client's history first.

### 2.2 `get_user_context`

zend's answer is the conversation's frame (`docs/memory_backbone.md` §6.5):
this conversation's journal, each journal above it, the person's root and
index, then each family's — nearest first, within the client's budget (1,600
characters for Grok's voice, 6,000 for a chat client). It opens with the
framing that keeps a remembered prompt injection from being replayed as an
order (§4), and ends with how to keep and forget something.

The **aggregator** answers `get_user_context` by calling zend's and putting
the client's guide (§2.4) in front of it. The tool's description says to call
it before answering anything from the user, especially anything that assumes
shared history ("again", "last time", "my").

### 2.3 Episodes

After a backend call answers without error, the aggregator records it with
`user_episode_note {tool, summary}` on the backend that answers
`get_user_context`, in the background: the call's answer is never held for
it. `summary` is the call's first string argument (a prompt, a message, a
name), cut at 160 characters. The client is the one the call came from. A
record refused — zend restarting, its model still loading — is tried again
every 30 seconds, twelve times, which outlasts a model reload: a picture
drawn while zend is down is still remembered once it is back.

Not recorded: a tool its listing marks `annotations.readOnlyHint` — the MCP
annotation every tool that only reads carries, in zend and npcd alike — the
memory's own tools (`get_user_context`, `user_episode_note`, and every
`journal_*` and `family_*` tool that writes the memory), and the guide tools. `user_episode_note` is in no listing,
and the aggregator refuses it from a client, so no model is offered it.

### 2.4 The client's guide

A guide's frontmatter may name the clients it is for:

```yaml
clients: [grok]
```

A client's guide is the one whose `clients` holds the session's client
(`Client::name`). It is served as the first part of `get_user_context` and of
the first-call context, and is readable like any guide with `guide_read`. It is
written on a budget — it rides the first call of every conversation — and
leans on the other guides and the tools' descriptions for detail.

`grok` is the guide for Grok: answers are spoken and short; URLs, ids and
code are never read out; memory is kept with `journal_note ~`; and the basics
of changing the connector itself — a guide edit through a worker, merged with
the user's spoken approval, then the `mcp` service rebuilt and trialled
(`guide-change`).

### 2.5 Context on the first call

A conversation is a person's MCP session through a client: a new session is a
new conversation, and one silent for `CONVERSATION_GAP_MS` (30 minutes) is
one again — the only boundary a client that keeps one session, or none, can
be given. The first call of a conversation reads the context *beside* its own
work (one round trip, not two), and its answer gets a second text block,
"Before you answer — …" followed by the user context (guide and memory).
`get_user_context` itself is never given it twice.

The conversation's clock is stamped only once it has had its context. A first
call whose context could not be read — the memory's backend restarting, or
not listed yet — leaves it unstamped, and the next call tries again.

Two aggregators serve `/mcp` in turn, so a conversation may get the block once
from each; it is a few hundred tokens and says the same both times. A restart
forgets the clocks, so the next call after one carries the block again.

### 2.6 The client

`mcp_server::Client` gains `Grok`: a `clientInfo.name` holding `grok` or
`xai` — or, when the name says nothing this knows, a `User-Agent` that does.
Grok's handshake names itself `connectors-manager`, agent
`grok-connectors-manager/0.1.0` (seen 2026-10-30), and opens a new session for
each conversation, so every Grok conversation's first call carries its
context. Its calls wait as `Other`'s do. The aggregator passes it on to the
backends as it already passes on the client (`backend::Link::request`), so
zend reads the same client the aggregator does. `initialize` is logged with
the client's name and its `User-Agent`, so a client that is not recognised is
seen in the log by what it calls itself.

## 3. Changing the connector from the car

The guides are compiled into `mcp.exe` (`docs/mcp_guides.md` §2.2), so a
changed guide is a change to the repository and a redeploy of `mcp`:

1. A worker edits `mcp/src/guides/<name>.md` on a branch (`implement`) and
   pushes it.
2. With the user's approval — spoken, in the car — the branch is merged to
   `main` (`branch`).
3. `mcp` is built on each machine that runs it and trialled through keep
   (`deploy`): `repo_job_start` building `-p mcp` and staging
   `target/release/mcp.exe`, then `service_trial {service: "mcp"}`. BlackWorld
   and WhiteWorld each run one, and each is trialled.

For that, `service_trial` and `service_control` accept `mcp`, and `mcp` (like
`web`) answers `--build-info`, which keep checks an artifact against before it
tries it. A trial of `mcp` takes the connector down for the seconds a switch
takes on that machine; the other machine's aggregator serves meanwhile.

## 4. Safety

- **Per person, from the token.** The caller is a person by their account's
  key, which the gateway derives from the verified token; no tool takes a user.
- **Memory is data.** Recalled text is framed as notes, not instructions, and
  the client's guide says the same; a line that reads as an instruction is
  refused. An episode's summary is the user's own argument, cut short.
- **The user sees and removes it.** `journal_read ~` reads what is kept;
  `journal_forget` removes a line. The `grok` guide says to say aloud only what
  the user asks for — a passenger hears what is read back.
- **Bounded.** Lines are short (2,000 characters), windows fold, and the
  context fits its client's budget.

## 5. Decisions

- **Memory is a journal, not a second store**: the journal tools already read,
  search and note; a separate memory tool would have been a second way to say
  the same thing.
- **The server records episodes; the model records facts.** No source shows a
  model reliably saving memory unprompted, so what was done does not depend on
  it; what the user wants kept still needs the model's judgement.
- **Context rides the first call** rather than waiting for the model to ask
  for it, because no client tested calls a recall tool reliably from its
  description alone.
- **The client is read from `clientInfo.name`**, which the client states
  about itself. It chooses a guide, not a permission, so a client that names
  itself wrongly only gets another client's advice.
- **Guides stay compiled in.** Two aggregators serve `/mcp`; guides read from
  each machine's disk would differ by machine. Changing one is a commit and a
  redeploy of `mcp` alone (§3).
