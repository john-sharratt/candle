# Private mode and diaries

**Status: building** (branch `aw/private-mode`, from 2026-11-11). Owner: John.
Components: the `mcp` aggregator, `mcp-server`, zend's MCP tools and memory,
the `journals` crate and repository, the guides. Extends `docs/mcp_app.md`,
`docs/mcp_user_context.md` and `docs/memory_backbone.md`; where they disagree
about conversations, sessions or what a call may write, this document is
authoritative.

## 1. Summary

Every Zen MCP call names its conversation by a **ref**, a handle the server
issues at `get_user_context`. A ref carries its conversation's **private
flag**: a private conversation is read-only against shared memory and every
other persistent surface, for as long as it lasts. **Diaries** are
topic-based notes, each encrypted at rest in the journals repository, opened
by passphrase, usable only in a private conversation. Journals are the shared
half of Zen; private mode and diaries are the personal half. The server
enforces the boundary, not the guides.

## 2. Why

Zen's memory is shared across members and clients, so some conversations must
not write to it:

- **Confidential work**: material that must not land in group-readable memory.
- **Shared accounts and devices**: a session must not record what one person
  said where another will read it.
- **Testing**: headless `claude -p` runs fill conversation journals with test
  episodes. Tests run without polluting memory.
- **Client parity**: when the client is incognito, Zen is too.
- **Secrets hygiene**: credentials sit in plaintext in journals; encrypted
  storage is overdue.

Before this, none of it was enforceable: guides told models to write what they
learn, and a guide is a remembered rule. This replaces it with a mechanism.

**Non-goals.** Protection against an attacker with root on a host while a
diary is open (plaintext is in process memory then). Hiding that diaries
exist. Erasure: deleting entries changes the current diary file only; older
encrypted versions stay in git history and open with the same passphrase.
Passphrase recovery: there is none.

## 3. Refs

A ref **is** the conversation: the server's handle for it, and nothing else
names a conversation. How a client uses MCP sessions plays no part —
claude.ai shares one MCP session across every chat and Claude Code session of
a person, and Grok opens one for every call — so nothing about a conversation
is keyed by, bound to, or spread across an MCP session.

- **`get_user_context {ref?, private_mode?}`** answers the conversation's ref
  on its first line. Without `ref` it begins a conversation: a new ref,
  private when `private_mode` is true. With a live `ref` it answers that same
  ref, and `private_mode` changes nothing. With an expired one it answers a
  new ref that keeps the old one's private flag — a private conversation never
  falls back to normal — private as well when `private_mode` asks.
- **Every other Zen tool requires `ref`**, the guide tools included. A missing,
  forged or expired ref is refused, naming what to do (fail closed). There is
  no default ref.
- **A ref is bound to the account that opened it.** Another account's call
  with it is refused as a forgery, so a ref read from a transcript is no use to
  anyone else.
- **Idle expiry: 4 hours** ([`REF_IDLE_MS`]). A ref unused that long is
  expired.
- **Episodes, the conversation's journal, `journal_focus` and attribution key
  on the ref**, so two conversations through one connector session are two
  conversations, and a Grok drive is one conversation however many sessions it
  opens.

### 3.1 What a ref is

`mcp_server::refs`: 15 bytes, written base64url — the alphabet of
`file_read`'s part anchors — in 20 characters, short enough for a model to
carry on every call —

```
id (6, random) || stamp (4, BE: bit 31 private, bits 0–30 seconds since 2026-01-01) || mac (5)
mac = HMAC-SHA256(key, owner || 0x00 || id || stamp)[..5]
```

- **Collisions:** two conversations share an id about once in 2^24 after
  2^12 of them, once in 2^8 after 2^20.
- **Guessing:** a ref opens only with its MAC, so forging one for one's own
  account takes about 2^40 calls; naming another conversation's needs its
  48-bit id as well, and another account's opens for nobody.
- **The stamp** runs to 2094.

The key is derived from the cluster's group key for `mcp conversation refs`
(`GroupKey::derive`), so both aggregators and every zend member open a ref the
same way and none needs a lookup to know whether a ref is real, whose it is,
or whether it is private. A process with no cluster mints its own key.

**The conversation's key** — what records, the conversation's journal and its
noted calls are kept under — is the first 8 bytes of SHA-256 of the id, in
hex: the ref itself is never written anywhere.

### 3.2 Liveness

Two aggregators serve `/mcp` in turn and keep nothing between them, so a ref's
last use is kept where both can read it: a replicated record in zend's
substrate, `mcp/ref/<conversation key>`, holding only `used_ms` — no account,
no flag.

- A ref **issued** within the last 4 h is live, read from the ref alone.
- Each aggregator notes each ref's last use in memory. One used here within
  4 h is live.
- Otherwise the aggregator asks zend (`conversation_ref_seen`, in no listing):
  live when the record says it was used within 4 h.
- Each aggregator tells zend a ref's use, in the background and at most once
  every 10 minutes per ref ([`REF_TOUCH_EVERY_MS`]), so the record is never
  more than that behind; the call that used it never waits for it.
- zend's tend ceases every ref record idle past 4 h.

A ref's record is ceased like any session's: a tombstone, gone from every
member's log at its next compaction.

### 3.3 What carries the ref

The aggregator takes `ref` out of a call's arguments and sends it to the
backend in `_meta["com.tokera/ref"]` (`mcp_server::protocol::REF_META`),
beside the client and its own session; `get_user_context` keeps it in its
arguments too, being the tool that reads it. A backend opens it with the same
key; a call reaching a backend directly — npcd's `service:zen` nudges to
zend — carries none, and a tool that needs a conversation says so.

The aggregator adds `ref` (required) to every listed tool's input schema
except `get_user_context`, whose own schema names `ref` and `private_mode`.

### 3.4 Conversations' journals

A conversation's journal is opened at its first write and never on a read —
`get_user_context` included (`docs/memory_backbone.md` §6.6) — and is kept
under its ref's conversation key. "Context on the first call"
(`docs/mcp_user_context.md` §2.5) is gone: every conversation begins with
`get_user_context`, which is its context.

## 4. Private mode

### 4.1 The flag

`private_mode: true` at the start of a conversation makes its ref private. The
flag is in the ref: nothing changes it, and a ref renewed after expiry keeps
it. Private applies to the conversation it was asked for, never to another.

### 4.2 What a private conversation may do

- Every tool the table (§4.3) classifies **read**.
- `diary_*` (§5).

Everything else is refused with `not available in private mode`.

### 4.3 The table

`mcp_server::access`: **one table classifies every tool every backend serves
as read or write**, the guide tools and diaries included. A tool not in it is
a write. Each server's tests enumerate its own registered tools and fail on
any the table does not classify — zend's, npcd's, zenj's and the aggregator's
own — so an unclassified tool fails the build.

Write, refused while private:

- the memory: `journal_note`, `journal_focus`, `journal_create`,
  `journal_edit`, `journal_link`, `journal_accept`, `journal_forget`, the
  `group_*` writes;
- messaging: `bot_message_send`, `bot_channel_post`, `bot_mission_give`,
  `bot_mission_step`, `bot_mission_cancel`, `zen_chat_send`, `zen_chat_close`;
- code, files and operations: `file_*_and_commit`, `repo_task_*`,
  `repo_job_start`, `service_deploy`, `service_control`, `cluster_command`,
  `gateway_command`, `job_cancel`;
- images: `image_generate`, `image_edit`, `image_upload` (prompts persist);
- anything else that changes persistent state: `timeline_delete`,
  `thread_archive`, `thread_attach`, `approval_grant`, `layer_toggle`, and
  zenj's queue, request, hands, moment, sfx, said, midnight, energy and
  lights tools.

### 4.4 Footprint

- A private conversation never opens a conversation journal, records no
  episode, and never appears in "where the last conversation left off" (it has
  no journal for that to read).
- **Logs**: no per-call line for a private call — neither the aggregator's
  `mcp: called` nor any server's `mcp: handled` — and no error text. A
  `diary_*` call's arguments and results are never logged, private or not.
  `mcp_server::Server::logs` asks the server whether a call is logged.
- No reported marks: what `job_status` and the like hand over is not marked
  reported for a private conversation.
- Its context is served with the `private-mode` guide, and zend's frame for
  it ends with how it reads, never with how to write.
- What remains of a private conversation is its ref record (§3.2): a digest
  and a time, ceased four hours after its last use.

## 5. Diaries

### 5.1 Model

- A diary is a topic. It has no owner, no group and no user identity: it
  belongs to whoever holds its passphrase. Group owners' rights do not reach
  it.
- One passphrase opens many topics. `diary_open` unlocks every diary encrypted
  under that passphrase; diaries under another passphrase stay invisible, and
  nothing says whether they exist.

### 5.2 Storage

In the journals repository, one encrypted file per diary, in a folder hidden by
its leading dot so that no listing names it — a folder's children, a file list,
an ingest and a folder unit's key all pass over what starts with a dot. A
listing that named the folder changed the key of the repository root's
priming-chain link the day the first diary was written:

- `.diaries/<diary_id>.bin`, where `diary_id` is 16 random bytes in hex — it
  says nothing of the topic. A file is bound to its id, never to its folder
  (§5.3), so moving the folder leaves every diary readable.
- `.diaries/salt`: one repository-wide random salt (16 bytes), created with the
  first diary.
- Each write rewrites that diary's whole file, re-encrypted under a fresh
  nonce. A file shows only its padded size and its commit times; how many
  entries it holds does not show.
- Each file is padded to a power of two from 4 KiB.
- Each commit's message is `diaries: update`, with one trailer,
  `Journal-Diaries: update`, by which a replay recognises it — no ids, no
  topic.
- **The journal service treats `.diaries/` as opaque**: the tree load skips it
  (`journals::tree::load`), so folds, summaries, voice digests, cues,
  `journal_grep`, `get_user_context` and every index — all of which read the
  tree — never see it; `journals check` passes over it; zend's branch ingest
  admits nothing under it; and retrieval never reads it. Tests hold each.
- A diary commit not yet on origin is replayed by copying its `.diaries/`
  files onto the new base: each write is the whole file, so the last one
  stands. One dropped because origin keeps another salt (§5.3) drops every
  diary commit after it too: they were sealed under the same lost salt.

### 5.3 Crypto (`journals::diaries::crypto`)

- passphrase + salt → **Argon2id** (m = 256 MiB, t = 12, p = 1: ~1 s on
  BlackWorld in a release build, from 63 ms measured there at 64 MiB and
  t = 3) → 32-byte master key, derived once per `diary_open`.
- HKDF-SHA256(master, info `zen diary kek`) → **kek**.
- File: `version (1) || wrapped_dek (16 + 32 + 24 nonce: XChaCha20-Poly1305
  under kek, AAD = diary_id) || nonce (24) || ciphertext (XChaCha20-Poly1305
  under the DEK, AAD = diary_id)`, padded.
- `diary_open` trial-unwraps every diary file's `wrapped_dek` with the kek;
  those that open are that passphrase's.
- Plaintext: `{topic, created, entries: [{id, ts, author: user|assistant,
  text}]}` as JSON. The topic is inside the ciphertext. Entry ids are 8 random
  bytes in hex, stable.
- Rotation seals the diary again under a **new DEK**, wrapped under the new
  passphrase's kek: git history keeps the old file, which the old passphrase
  still opens, but neither the old passphrase nor the old DEK opens anything
  written from then on. A conversation that had the diary open under the old
  passphrase finds it moved at its next change, and it is closed there —
  never sealed back under the old.
- Master keys, keks and DEKs are zeroized when dropped, Argon2's working
  memory with them (its `zeroize` feature), and so is every plaintext buffer
  a diary is sealed from or opened into. What cannot be wiped is the request
  itself: a passphrase and an entry's text arrive in the call's JSON, and a
  diary read leaves in its answer. No recovery path.
- **The salt is written once.** A write that would replace or remove it is
  refused, and a replay meeting another salt on origin is dropped and said
  so, since every diary is stretched with the one there.
- **Limits**: a diary holds at most 32 MiB of plaintext, so its file — 64 MiB
  padded — stays within what the repository's host takes in one file. Each
  write is the whole file again; compaction is how a diary is kept small.

### 5.4 Tools

Every one requires a private ref and is refused otherwise. Each is served by
a member that holds the journals checkout, at zend's diaries path
(`/v1/mcp/diaries`), which only those members declare.

**Every diary call of a conversation reaches one member.** What a
conversation unlocked is held in the memory of the member that opened it, and
more than one member holds the checkout, so a call dealt to another would
find nothing open. The aggregator names the conversation's diary key
(`diary-` followed by its conversation key) in the `x-tokera-diary` header of
every diary call; the diaries path is sticky on that header. The first call
is placed among the members declaring the path; the member that serves a
`diary_open` holds the key's claim — whatever the open found, zero included
— until the conversation's diaries are locked by `diary_close` or by the
sweep, and every later call goes to it. The claim is held past the request
that took it, unlike a turn's. A member that restarts loses both what was
unlocked and the claim, and the conversation opens again wherever its next
open is placed.

| Tool | Does |
|---|---|
| `diary_open {ref, passphrase}` | Unlock every diary under this passphrase for this conversation. Answers how many opened, and nothing else. Zero is no error: the opened passphrase can then create diaries. Every open is counted against the account before the passphrase is tried, one at a time; after three that found nothing each waits twice the last, up to an hour, and an open that finds a diary takes back only its own count. At most two passphrases are stretched at once on the member; an open refused before its passphrase is tried — the member busy stretching others, the memory unavailable — is counted against no one. |
| `diary_create {ref, topic, passphrase}` | Create a diary under the open passphrase, which must be repeated and match — a typo makes no orphan. A passphrase's topics are one each, across every conversation it is open in. |
| `diary_list {ref}` | The open topics, each with its entry count and first and last entry `ts`, those another conversation created under the passphrase among them. |
| `diary_read {ref, topic}` | The whole diary: every entry in `ts` order, with ids. No summary, no cap: a diary's answers are never cut to fit, as other tools' are. |
| `diary_write {ref, topic, text, ts?, author?}` | Insert an entry, at `ts` (default now; a past `ts` places it there). Answers its id. |
| `diary_delete {ref, topic, ids}` | Delete one or more entries. |
| `diary_rotate {ref, topic, passphrase, new_passphrase}` | Rewrap a diary under a new passphrase. Refused when the new passphrase already keeps a diary on that topic: its topics stay one each. |
| `diary_close {ref}` | Lock. |

- Unlock state lives in memory only, keyed by the conversation, and is never
  persisted. It is cleared by `diary_close`, and when the ref expires (the
  memory member's tend reads the ref's record).
- Manual compaction is `diary_delete` of many entries and `diary_write` of one
  replacement at the right past `ts`.
- The descriptions say only: topic-based diaries, encrypted at rest, opened by
  passphrase, available in private mode.

## 6. Guides

- A new **`private-mode`** guide: what private mode is and why (§2), what is
  blocked, that diaries exist only in it, and the diary routine —
  `diary_open` when asked → `diary_read` the whole topic → work → write new
  entries as the conversation goes → `diary_close`. Compaction only when the
  user asks.
- **Both ways**: in private mode nothing learned goes anywhere but a diary —
  the server enforces it, and the guide says so, so models do not fight the
  refusals. In a normal conversation diaries are personal: never open, list or
  mention them unless the user asks in this conversation, and never infer
  anything about a person from one existing.
- **`claude` and `grok`**: call `get_user_context` first, with
  `private_mode: true` when any of §7's cases apply; pass `ref` on every call.
- **`memory`**: one line — private conversations are read-only; see
  `private-mode`.
- **Server instructions**: pass `ref` on every call; diaries are personal and
  touched only when the user asks.

## 7. `get_user_context`'s description

It documents `private_mode`, with when to pass `true`: the client says the
conversation is incognito, private or temporary (claude.ai's incognito chats);
the user asks to keep it off the record, private or out of memory; confidential
work material; a shared device or account; personal matters — health,
finances, anything the user says is theirs alone; automated tests or
experiments; the user asks to open or work in a diary. And it says: private
mode is read-only for shared memory, cannot be turned off for the rest of the
conversation, and is the only mode diaries work in.

## 8. Observability

Private conversations and diary calls are counted, as aggregate counters only
— no ids, no arguments. `diary_open`'s backoff is held in memory.

## 9. Clients

Private mode and diaries work the same through every client the gateway
serves — Claude, Claude Code, Grok. No client is special-cased: Grok's
per-call sessions are nothing to a ref.

## 10. Tests

All must pass before release.

**Refs**: a call without a ref is refused; a forged one is refused; another
account's is refused; `get_user_context` with the ref answers the same ref and
cannot change `private_mode`; an idle ref expires at 4 h, and a renewed private
one stays private; a ref live on one aggregator is live on the other.

**Private mode**: every write-classified tool is refused under a private ref
(generated from the table); an unclassified tool fails each server's tests; an
expired private ref still refuses writes; a private ref's flag reaches no other
conversation.

**Footprint**: a private `get_user_context` opens no conversation journal; a
normal one opens none until the first write; no episode is recorded for a
private ref; no log line carries a `diary_*` argument or result, and no
private call is logged; `.diaries/` is read by no fold, summary, grep, cue,
index or `get_user_context`, by no ingest and by no retrieval.

**Diaries**: refused under a normal ref; `diary_open` with an unknown
passphrase opens nothing and creates nothing; `diary_create` with a mismatched
passphrase is refused; two passphrases' diaries are invisible to each other; a
scan of every object in the repository's history finds no topic and no entry
text (synthetic markers); a past `ts` reads back in place; `diary_delete`
removes entries, and a reopen confirms; the compaction round trip reads back
right; after rotation the new passphrase opens it and the old does not; unlock
state is cleared on close and on expiry; a 5 MB diary reads back whole.

**Clients**: the same private-mode and diary tests pass with a Grok client
opening a new MCP session for every call.

## 11. Rollout

1. Refs, and conversations keyed by them; guides and descriptions pass refs;
   refs mandatory. (There is no flag: refs are the path from the change that
   lands them.)
2. The private flag, the table, `get_user_context`'s description, the
   footprint.
3. Diaries: storage, crypto, tools, the `private-mode` guide.
4. Delete conversation journals that only ever held reads — with John's
   approval of the list.

Nothing already in the journals is moved into a diary (John, 2026-11-11):
diaries hold what is written to them from private conversations on.

## 12. Decisions

- **The ref is the conversation, and holds its own private flag** (John,
  2026-11-11): how a client uses MCP sessions is irrelevant to it. The draft's
  binding of a ref to its MCP session, and its spreading of the private latch
  across a session, are gone: claude.ai's one session per person would have
  made one incognito chat lock every conversation for weeks, and Grok's
  session per call would have refused every Grok call after the first.
- **A ref is sealed, not looked up**: both aggregators and zend open it with a
  key derived from the group key, so a private flag is enforced with no hop on
  any call; only liveness is shared, through a record.
- **Refs are mandatory from the change that adds them**, without a flag
  (CLAUDE.md: no feature flags). The clients are models reading the tools'
  schemas and descriptions, which change in the same release.
- **The guide tools need a ref too**: every conversation then starts at
  `get_user_context`, which makes "context on the first call" redundant, and
  it is removed.
- **What a conversation holds follows its ref, never a session**: the thread
  `thread_attach` binds is a record per account and conversation
  (`mcp/attached/<owner>/<conversation>`), ceased four hours after the ref's
  last use, rather than a value held on the MCP session — on which Grok's
  next call would not find it, and claude.ai's one session would gather one
  per chat for ever.
- **Liveness is the aggregator's to say, `get_user_context` included**: a
  ref it finds live goes to zend in `_meta` and is answered as it is; one it
  does not is renewed. Its use is told only once it is found live, so an
  expired ref is never revived by naming it.
- **A private conversation's context is served with the `private-mode`
  guide** in place of the client's, and zend's frame ends with how a private
  conversation reads rather than how to write: every client's guide says to
  write.
- **A private call keeps no reported marks**: `job_status` and the other
  readers that hand over journal entries mark them reported once delivered —
  a record, keyed by account — and a private conversation leaves none.
- **The ref is the conversation's key everywhere, but it is never written**:
  records, journals and noted calls are kept under a digest of its id.
