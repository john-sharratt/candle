---
name: private-mode
description: What a private conversation is — read-only for Zen's shared memory and everything else that lasts, to its end — when to begin one, and what it refuses.
tools: [get_user_context]
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
private mode, and keep what was learned in this conversation alone.

## Normal conversations

Private mode belongs to the conversation that asked for it. A conversation
that did not begin private writes as always; it never becomes private later.
