---
name: private-mode
description: What a private conversation is — read-only for Zen's shared memory and everything else that lasts, to its end — when to begin one, what it refuses, and how to keep a diary in one.
tools: [get_user_context, diary_open, diary_create, diary_list, diary_read, diary_write, diary_delete, diary_rotate, diary_close]
---
# Private mode

Zen's memory is shared: every member of a group, and every AI tied to it,
reads what any of them writes. Some conversations must not write there —
confidential work, a shared device or account, something personal, a test —
so a conversation can begin **private**.

## Beginning one

`get_user_context {private_mode: true}`, as the conversation's first call.
Begin private when:

- the client says the chat is incognito, private or temporary (a claude.ai
  incognito chat);
- the user asks to keep it off the record, private, or out of memory;
- it is confidential work material that must not land in shared memory;
- the device or account is shared, and what one person says must not be kept
  for the others;
- it is personal — health, finances, anything the user says is theirs alone;
- it is an automated test or an experiment;
- the user asks to open or work in a diary.

The answer's first line is the conversation's `ref`, and the next says it is
private. Pass the ref on every call, as always.

## What it does

- **Every read works**: the memory, threads, files, jobs, logs, bots' state.
- **Everything that lasts or tells someone is refused** with `not available in
  private mode`: notes and every other journal write, focus, groups;
  messages, posts and missions to bots; chats with Zen; commits, tasks, jobs,
  deploys and every operation; drawing, editing or uploading a picture; and
  anything else that changes what is kept.
- **It leaves no footprint**: no conversation journal, no record of what it
  did, no log of its calls.
- **It lasts to the conversation's end.** Nothing turns it off, and a ref
  renewed after four idle hours stays private.

The server enforces all of this — the refusals are the mechanism, not a
mistake to work around. Do not retry a refused write another way, or begin a
second, normal conversation to make it: tell the user it is not available in
private mode. Nothing learned here goes anywhere but a diary.

## Diaries

A diary is a topic — health, a project, anything — kept encrypted in Zen's
repository and opened by a passphrase the user holds. It belongs to whoever
holds the passphrase: no group, no owner, and no one else can tell it is
there. There is no recovering a lost passphrase. Diaries work only in a
private conversation.

The routine:

1. **When the user asks**, `diary_open {passphrase}`: every diary under that
   passphrase opens; the answer says how many. None is no error — a new
   passphrase — and `diary_create {topic, passphrase}` (the same passphrase
   again) makes one.
2. **`diary_read` the whole topic** before working on it: every entry, in
   time order.
3. **Work**, and **`diary_write` new entries as the conversation goes** —
   `author: user` for the user's own words, `assistant` for yours; `ts` for a
   moment in the past, where the entry then reads.
4. **`diary_close`** when done; diaries also lock four hours after the
   conversation's last call.

Compact a diary only when the user asks: `diary_delete` the entries being
folded, then one `diary_write` of the replacement at the first one's `ts`.
`diary_rotate` moves a diary to a new passphrase.

Passphrases are the user's: never note one, repeat one back, or suggest one
they did not ask for.

## Normal conversations

Private mode belongs to the conversation that asked for it. A conversation
that did not begin private writes as always; it never becomes private later.

**In a normal conversation, diaries are personal.** Never open, list or
mention them unless the user asks in that conversation, and never infer
anything about a person from a diary existing — or from asking about one.
