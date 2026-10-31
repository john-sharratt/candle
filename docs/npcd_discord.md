# npcd on Discord

Every character a world runs gets a Discord account of its own, and the world's
messaging is mirrored with Discord in both directions: each group a character is in
is a Discord channel, each direct message between a person and a character is a
Discord DM with that character's bot, and a person who writes on Discord is heard in
the world exactly as one who writes in the console. Background jobs keep the two in
step through restarts, gaps and authority moves.

This document is the design. Code: `npcd/src/discord/` (new). It sits on npcd's
messaging (`npcd/src/sim/phone.rs`, `npcd/src/engine/runtime.rs` `message_npc` /
`say_on_channel`, `npcd/src/engine/enact.rs` the `message` family of acts) and on the
world authority (`npcd/src/authority/`, `docs/replicated_substrate.md` §9).

---

## 1. What exists, and what it lacks

npcd already has three kinds of conversation, all held in one store, the world's
`Sim.threads` (`sim/phone.rs:100-170`):

| npcd | What it is | Who writes it |
|---|---|---|
| **The channel** (`CHANNEL = "the channel"`, `phone.rs:83`) | one per world, permanent; every enrolled person and character is a member | a person with the Creator role (`POST /v1/world/:wid/channel`), a character's `message` act to `"the channel"` |
| **A group** (`Kind::Group`: `named`, or more than two members) | opened by a character's `open_group`, or a DM that gained a third member by `invite` | characters' `message` / `invite` / `open_group` acts |
| **A DM** (`Kind::Direct`) | a person and a character, or two characters | `POST /v1/npc/:nid/message` (`bot_message_send`), a character's `message` act |

A message is delivered to a character by `deliver_messages` (`engine/environment.rs:389-449`)
as an `EventKind::Message`, urgent for a DM; the character answers with the same
`message` act; every write is a `WorldOp` applied on the world's **authority**
(`authority/perform.rs:43-64`), the one member holding the world's lease, and reaches
every other member as substrate records.

What a mirror needs and the store does not have:

- **No message identity.** A message is `{from, intent, seen_by}`
  (`phone.rs:193-203`): no id, no time, no author account — `from` is a display name
  (`engine/mod.rs:1134-1166`), and positions shift as the window is trimmed
  (`phone.rs:251-268`). A mirror cannot say "this one I have posted" or "this one came
  from Discord".
- **No hook when a message is created.** Every message passes through
  `Threads::send` (`phone.rs:541`); nothing observes it, and the codebase chose
  polling over a broadcast deliberately (`engine/mod.rs:1440-1448`).
- **No external identity** on an account or a character.
- **No outbound HTTP** in npcd (`host` carries `reqwest` and `tokio-tungstenite`).

§4 adds the first three; the fourth is a dependency.

---

## 2. Discord's constraints

From Discord's documentation (October 2026; `docs.discord.com/developers`):

- **A bot account is made by hand.** There is no API to create an application or its
  bot user; each is created in the Developer Portal. A **team holds at most 75
  applications**. A bot's own name and avatar can be set by API (`PATCH /users/@me`).
- **One gateway connection per bot token** for receiving messages; ordinary messages
  arrive only over the gateway (the interactions endpoint carries commands and
  components, never messages). **1,000 IDENTIFYs per token per day.**
- **Intents.** `GUILDS`, `GUILD_MESSAGES`, `DIRECT_MESSAGES`, and the privileged
  `MESSAGE_CONTENT` — toggled on in the portal for an application in fewer than 100
  servers, which every NPC bot here is. Without it a bot reads only its own messages,
  its DMs, and messages that mention it.
- **DMs need a shared server.** A bot can open a DM (`POST /users/@me/channels`), but
  only to a user it shares a server with (error 50278), and Discord asks that DMs start
  from the user.
- **Webhooks** post with a per-message `username` and `avatar_url`, into a channel or a
  thread; 15 per channel. They cannot DM and cannot read.
- **Rate limits.** 50 requests/s per bot; per-route buckets read from headers, never
  hard-coded; **10,000 invalid requests (401/403/429) per 10 minutes per IP**, shared by
  every token on that IP, past which Cloudflare bans the IP.
- **Messages** up to 2,000 characters; attachments up to 20 MiB; only the author edits.
- **Policy.** Self-bots (user tokens) are forbidden; every account here is a bot
  application carrying Discord's BOT badge. Impersonating people or organisations is
  forbidden; fictional characters labelled as bots are not. No unsolicited bulk DMs.

---

## 3. The shape

```
                 Discord                                   npcd (the world's authority)
  ┌──────────────────────────────────┐            ┌──────────────────────────────────────┐
  │ Guild "Life"                      │            │ discord::Bridge (one per world)        │
  │  #the-channel   ◄──────────────── │◄── post ───│  outbound: per-thread cursor, beat    │
  │  #<group>  (one per named group)  │            │  inbound:  gateway events → WorldOp   │
  │  DM: person ↔ Zen's bot           │──events───►│  reconcile: channels, members, gaps   │
  │  DM: person ↔ Drone-Kettle's bot  │            │                                        │
  └──────────────────────────────────┘            │  one gateway shard per character bot  │
                                                   └──────────────────────────────────────┘
```

- **A world is a Discord server (guild).** The world document names it
  (`worlds/<id>.yaml`, `discord: { guild: <id> }`), authored with the rest of the world.
- **The channel** is `#the-channel` in that guild.
- **A named group** is a text channel of its own, created by the bridge, whose
  permission overwrites admit exactly the thread's members: each member character's bot
  and each member person's Discord account.
- **A person ↔ character DM** is a Discord DM between that person and that character's
  bot.
- **A character ↔ character DM** is private to the two characters in the world, and is
  mirrored only for the characters' **owners**: a private thread under `#backstage`,
  read-only, visible to the owners of both characters. A world can leave it out
  (`discord: { backstage: false }`).

### 3.1 Who speaks on Discord

- **A character with its own bot** posts everything — channel, groups, DMs — as that bot,
  under its own name and portrait. This is the whole point: the character is a member
  of the server, appears in the member list, and is DMed like a person.
- **A character without a bot** (past the team's 75, or not yet provisioned) speaks in
  channels and groups through the guild's **persona webhook**, one per channel, with
  its name and portrait per message. It has no DMs on Discord; its DMs stay in the
  console.
- **A person writing in the console** is mirrored into a group by the persona webhook
  under the person's own name, and into a DM by the character's bot as a quoted line
  (`> John (console): …`): a bot cannot post as a person, and a webhook cannot DM.
- **Reading** a guild's channels is done by **one bot per world** — the world's
  *listener*, the character the world names (`discord: { listener: zen-moon }`), with
  `MESSAGE_CONTENT` on — so a message is received once, not once per bot in the channel.
  Every other bot connects with `DIRECT_MESSAGES` only, and hears its own DMs.

### 3.2 People

A person on Discord is a person in npcd only once **linked**: the account record
(`accounts.rs:151-200`) gains `discord: Option<DiscordUser { id, username }>`. Linking:
the console shows a one-time code; the person sends it to any of the world's bots in a
DM; the bot that receives it writes the link. An unlinked author's messages are not
heard in the world, and the bot answers once, telling them how to link.

A linked person is then the same author in both places: their Discord messages arrive
as `WorldOp::Message { from: <their speaking name> }` exactly as `post_message`
(`engine/mod.rs:1179-1234`) builds it, with their ownership and roles — so a channel
post from Discord still needs the Creator role, and a DM to a character still needs
the character to be one they may message.

---

## 4. Changes to messaging

### 4.1 A message has an identity

`Message` gains:

```rust
pub struct Message {
    pub from: String,
    pub intent: String,
    pub seen_by: Vec<String>,
    /// Per-thread, strictly increasing, never reused — survives trimming.
    pub seq: u64,
    /// When it was written, in real milliseconds.
    pub at_ms: u64,
    /// Where it came from, when not from the world: the Discord message it
    /// mirrors, so the bridge never posts it back.
    pub origin: Option<Origin>,
}
```

`Thread` gains `next_seq: u64`. `Threads::send` assigns `seq` and `at_ms`; the trimming
in `phone.rs:251-268` no longer changes what a message is called. `Origin` is
`Discord { message: u64, channel: u64 }`.

### 4.2 The bridge's own store

What maps npcd to Discord is the bridge's, not the sim's, in the world's records
(`world/<wid>/discord/...`, written by the authority, replicated like every world
record):

| Record | Holds |
|---|---|
| `discord/thread/<thid>` | `{ channel_id, kind, mirrored_through: seq, last_seen_discord: message_id }` |
| `discord/dm/<npc_id>/<person>` | `{ channel_id, mirrored_through, last_seen_discord }` |
| `discord/bot/<npc_id>` | `{ application_id, user_id, token: <sealed> }` |

### 4.3 Bot tokens

A character's token is sealed with a key the cluster derives (`join.derive_key("npcd
discord tokens")`, the pattern of `image_links.rs`) and stored in `discord/bot/<npc_id>`,
so every member can take over a world and its bots; it is never in the character
record (which `/v1/npcs/export` hands out whole), never in git, and never logged. The
console's character page takes a token once (`PUT /v1/npc/:nid/discord`, owner only),
checks it (`GET /users/@me`), sets the bot's name and avatar from the character
(`PATCH /users/@me`), and seals it.

---

## 5. Keeping it in sync

Everything runs on the **world's authority**: every message is written there first, so
the outbound side sees it in the same beat, and an inbound message becomes a `WorldOp`
without a hop. The bridge is started and stopped with authority: `authority::driver`
already decides when a member takes or loses a world, and the bridge follows it.

### 5.1 Outbound — npcd to Discord

Each world beat (`authority/beat.rs:69-83`, after `write_world`), the bridge walks the
world's threads and, for each mapped thread, posts every message with
`seq > mirrored_through` and no `origin`:

1. Rendered as the console renders it — the character's line as voiced, not the raw
   `intent` (`phone.rs:196-198`) — through the same path `get_messages` uses
   (`engine/mod.rs:1237-1276`).
2. Posted by the right sender (§3.1), split at 2,000 characters on a sentence boundary.
3. `mirrored_through` advances only on Discord's `200` with the message id it returned.

A post that fails with 429 waits `retry_after`; one that fails otherwise leaves the
cursor where it was, and the next beat tries again — so a gap is retried, never
skipped. Posting is per channel and in order.

### 5.2 Inbound — Discord to npcd

The listener (for channels) and each character's bot (for its DMs) receive
`MESSAGE_CREATE`. The bridge drops:

- a message any of the world's own bots or webhooks wrote (by author id and
  `webhook_id`) — the echo of an outbound post;
- one from an unlinked person (§3.2);
- one already applied (its id is a recorded `origin` in the thread).

and applies the rest as the console's routes would: a DM as `WorldOp::Message`, the
channel as `WorldOp::Channel`, a group as a send into that thread, each carrying
`origin` so it is never posted back. `last_seen_discord` advances with each.

Attachments become a link line (`[image: <url>]`) until npcd threads carry images.
Edits and deletions on Discord are not replayed into the world: a world's past is not
rewritten by an edit.

### 5.3 Reconcile — the background job

Every five minutes, and once when a member takes a world:

- **Channels.** Every named group has its channel, every channel's permission
  overwrites match the thread's members (a member who left is removed, a new one
  added), and a group a character signed off from that nobody remains in is archived
  (made read-only), not deleted.
- **Gaps.** For each mapped channel and DM, `GET /channels/{id}/messages?after=
  <last_seen_discord>` fetches what arrived while no member held the world — a restart,
  a move — and applies it through §5.2, oldest first.
- **Bots.** Every provisioned character's bot is connected, in the guild, and named and
  pictured as the character currently is.

### 5.4 Authority moves

When the authority moves, the old holder closes its gateway connections and the new
holder opens them; IDENTIFY's 1,000 a day per token is far above any move rate. The
records the old holder wrote — `mirrored_through`, `last_seen_discord` — arrive with
replication before the new holder's first beat as its authority (§9.3), so it resumes
where the old one stopped. A message the old holder posted but did not record is posted
again at most once; the reconcile's gap fetch recognises its own echo by author and
drops it.

---

## 6. Running many bots

One process holds every bot's gateway connection: **twilight** (`twilight-gateway`,
`twilight-http` 0.17), one `Shard` per token in one tokio runtime, one HTTP client per
token for its own buckets, and a shared identify queue. `serenity` builds a whole
client per token and is the heavier fit.

The **per-IP invalid-request limit is shared by every token**, so the bridge keeps one
counter for the host: past 5,000 401/403/429 responses in ten minutes it stops sending
for the rest of the window and says so. A 401 (a token revoked in the portal) marks
that bot `revoked` on the console and stops its shard rather than retrying.

---

## 7. What it costs a person to set up

For each world:

1. Create a Discord server for it, and a Discord **team** that will own the bots.
2. For each character that should have its own account: create an application in the
   team, add a bot, enable `MESSAGE_CONTENT` (listener) or leave it off (others), copy
   the token into the character's page in the console. npcd names it, pictures it and
   invites it (the bot's invite URL is shown with the token).
3. Put the guild id in the world document.

Linking people is theirs: a code from the console, sent to a bot.

---

## 8. Build order

1. `Message.seq` / `at_ms` / `origin` and `Thread.next_seq` (§4.1), with tests that a
   trimmed thread keeps every surviving message's `seq`.
2. The bridge's records and sealed tokens (§4.2–4.3); `PUT /v1/npc/:nid/discord`.
3. Outbound for the channel (§5.1), with a fake Discord HTTP server in tests asserting
   the exact requests and that a 429 and a 500 are retried in order.
4. Inbound for the channel and DMs (§5.2), echo suppression and `origin` tests.
5. Groups as channels and the reconcile (§5.3).
6. Account linking (§3.2).
7. Authority moves (§5.4), with a two-member test that the cursor carries over.
8. Backstage threads for character-to-character DMs.

---

## 9. Open questions

- **Rendering.** Is the console's voiced line computed when it is read, and at what
  cost? If it is a model call per read, the bridge renders once per message and keeps
  the text with the cursor, rather than rendering on every reconcile.
- **Which world first.** `life` (Zen and its Drones, the family) is the natural first
  guild; Battle Cities' cast is larger than one team's 75.
- **The family's Discord accounts.** Linking assumes each person has one; a person
  without one keeps the console.
- **Voice.** Discord voice channels are out of scope here.
