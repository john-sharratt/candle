---
name: groups
description: How the memory is split into groups — making one, saying what it is for, adding and removing people and bots, and choosing where a person's conversations open.
tools: [group_create, group_describe, group_add, group_remove, group_default, get_user_context, journal_read]
---
# Groups

A **group** is a memory of its own: its own master (`@<group>`), people,
topics, conversations, threads and bots. Nothing in one group's journals mixes
with another's. A conversation sees **only its person's groups' journals**,
on every route into its context — the frame, the reads, and what provenance
places for it. A household, a team and a game's world are each a group.

| To | Call | Who |
|---|---|---|
| Make one | `group_create {group, name, about}` — the user is its first owner | any member |
| Say what it is for | `group_describe {about, group?}` — every frame and its README open with it | an owner |
| Add a person | `group_add {person, name?, emails?, role?, group?}` — `member` by default, `owner`, or `reader` | an owner; listing emails is an estate admin's |
| Add a bot | `group_add {bot, reads, group?}` — a reader of the paths it is granted | an owner |
| Take someone out | `group_remove {person \| bot, group?}` — what they wrote stays theirs | an owner; anyone leaving |
| Choose where conversations open | `group_default {group}` | the person themselves, only on their own choice |

`group` is the conversation's own when left out, and is written `@<slug>`.

## Rules

- **A person's own choice is theirs.** Never set someone's default group, or
  leave a group for them, on your own initiative.
- **An email binds an account.** Listing an email lets whoever signs in with
  it become that person: only an estate admin lists one, and only for the
  person who owns the address.
- **A bot reads, never writes,** and only the paths it is granted.
- **The last owner stays.** A group always has one.
- Reading another group's journals is by naming it, `@<group>/…`, and only
  for a group the person is in.
