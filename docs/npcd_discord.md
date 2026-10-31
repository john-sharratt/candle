# npcd on Discord

**Discord is where people talk to the characters.** A person talks with the cast in the
world's server, and DMs a character there as they would a friend: a character its owner
has given a Discord bot of its own is DMed as that bot, and every other character is
"DMed" in a private thread in the world's server, where it speaks through the persona
webhook (§3.1). The console and an assistant's `bot_message_send` remain ways in, but
the secondary ones.

**The substrate stays the record.** What is said is kept in the world, where the
characters hear it, answer it and remember it, and Discord mirrors it in both
directions: each group a character is in is a Discord channel, each direct message
between a person and a character is a Discord DM with that character's bot or, for a
character without one, a private thread, and a person who writes on Discord is heard in
the world exactly as one who writes in the console. Background jobs keep the two in step through restarts, gaps and authority
moves. §1.1 says why Discord is not the record.

This document is the design. Code: `npcd/src/discord/` — so far the bot provisioning
of §4.3 (`seal`, `store`, `client`, `profile`, `invite`, `routes`); §8 says what comes
next, in the order it is built. It sits on npcd's
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

### 1.1 Why Discord is not the record

Making Discord the store of the world's messages — reading them back from it rather
than mirroring them — was considered and set aside:

- **Characters could not message each other privately.** A Discord bot cannot open a
  DM with another bot, so every character-to-character DM would have to become a
  hidden channel.
- **The world would stop with Discord.** Characters hear messages as world events each
  500 ms beat (`deliver_messages`); an outage, a revoked token or the per-IP ban (§2)
  would leave every character deaf and mute, where today a world runs on even cut off
  from its cluster.
- **Character traffic is the heavy traffic.** The cast talks among itself far more
  than people talk to it; Discord's rate limits would throttle the world, not a mirror.
- **Not every character has an account.** A team holds 75 applications; Battle
  Cities' cast is larger.
- **It is the characters' memory.** What a character said is its substrate — what
  replication, replay and provenance over memory read. Kept on Discord it would sit
  behind another company's API and retention, and so would the family's conversations.

So people's way in is Discord, and the record is the world's.

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
  │ any server the person shares      │            │ discord::Bots (every bot of a cast     │
  │  DM: person ↔ Zen's bot           │──events───►│   in a world this member runs)         │
  │                                   │◄── post ───│  one gateway shard per bot, DMs        │
  ├──────────────────────────────────┤            ├──────────────────────────────────────┤
  │ Guild "Life" (the world's server) │            │ discord::Bridge (a world with a guild) │
  │  #the-channel   ◄──────────────── │◄── post ───│  outbound: per-thread cursor, beat    │
  │  #<group>  (one per named group)  │            │  inbound:  listener events → WorldOp  │
  │  #dms ▸ private thread: person ↔  │──events───►│  reconcile: channels, members, gaps   │
  │         Drone-Kettle (webhook)    │            │                                        │
  └──────────────────────────────────┘             └──────────────────────────────────────┘
```

There are two parts, and the first stands without the second:

- **Bots** (§3.3). A character with its own bot is online on Discord whenever its world
  is running, and is DMed there, in any server the person and the bot share. This needs
  no server set up for the world.
- **The world's server** (§4.4). A world may also have a Discord server, set in the
  console. Then the world's conversations are mirrored into it: the channel, every
  group, and the "DMs" of characters without a bot of their own.

In a world with a server:

- **A world is a Discord server (guild)**, named by the world's `discord/guild` record
  (§4.4).
- **The channel** is `#the-channel` in that guild.
- **A named group** is a text channel of its own, created by the bridge, whose
  permission overwrites admit exactly the thread's members: each member character's bot
  and each member person's Discord account.
- **A person ↔ character DM** is a Discord DM between that person and that character's
  bot, when its owner has given it one — with or without a world server; otherwise a
  private thread in the world's server, holding the person and the world's listener, in
  which the character speaks through the persona webhook (§3.1).
- **A character ↔ character DM** is private to the two characters in the world, and is
  mirrored only for the characters' **owners**: a private thread under `#backstage`,
  read-only, visible to the owners of both characters. A world can leave it out
  (`backstage: false` on its `discord/guild` record, §4.4).

### 3.1 Who speaks on Discord

A real bot is **opt-in, per character**. Discord makes a bot only by hand (§2), so a
character has one when its owner made one in the Developer Portal and connected it in
the console (§4.3); nothing assumes that every character has one, and most need not.

- **A character with its own bot** posts everything — channel, groups, DMs — as that bot,
  under its own name and portrait. The character is a member of the server, appears in
  the member list, and is DMed like a person.
- **A character without a bot** is on Discord only in a world with a server (§4.4).
  There — not provisioned, or past the team's 75 — it speaks in
  channels, groups and its "DMs" through the guild's **persona webhook**, one per
  channel, with its name and portrait per message. Its DMs are **private threads** in
  the world's server: a person who wants to talk to it alone is added to a private
  thread under a `#dms` channel, which the bridge opens (with Create Private Threads)
  and in which the character's lines are webhook posts and the person's are their own.
  A webhook cannot DM (§2); a private thread is the nearest thing a server has.
- **A person writing on Discord** — the usual case — is already there; the bridge only
  carries the message into the world (§5.2).
- **A person writing through a secondary way in** — the console, or an assistant's
  `bot_message_send` — is mirrored into a group by the persona webhook under the
  person's own name, into a bot's DM by the character's bot as a quoted line
  (`> John (console): …`) — a bot cannot post as a person, and a webhook cannot DM — and
  into a private thread by the webhook under the person's name, as in a group.
- **Reading** a guild's channels is done by **one bot per world** — the world's
  *listener*, the character its `discord/guild` record names (§4.4), with
  `MESSAGE_CONTENT` on — so a message is received once, not once per bot in the channel.
  The listener hears the private threads too, so a world with a server needs one
  provisioned bot at least. Every other bot connects with `DIRECT_MESSAGES` only, and
  hears its own DMs; a DM's content reaches its bot without `MESSAGE_CONTENT`.

### 3.2 People

Because Discord is the way in, linking is the first thing a person does, and the
console's own message pages show a person's threads with the character but point them
to the character's bot to talk.

A person on Discord is a person in npcd only once **linked**: the account record
(`accounts.rs:151-200`) gains `discord: Option<DiscordUser { id, username, linked_ms }>`,
and a replicated index `discord/user/<discord user id>` → the account key, so an
arriving message finds its account without a scan.

**Linking**, on the person's **Profile** page (`web/content/npcd/pages/profile.js`), in a
Discord section:

1. "Link Discord" asks `POST /v1/me/discord/link`, which answers a **code**: six
   digits, valid for **ten minutes**, used **once**, one standing code per account (asking
   again replaces it). Codes are the custom object `discord/link/<code>` →
   `{ account, expires_ms }`, so any member's bot can redeem one. The page shows the
   code and the names of the bots it can be sent to — every connected bot of a character
   the person may message.
2. The person DMs the code to any of those bots. The bot that receives it checks the
   code is live, writes the account's `discord` field and the index, tombstones the
   code, and answers "Linked — you are <name> here."
3. The Profile page polls `GET /v1/me/discord` and shows "Linked as @<username>", with
   an Unlink (`DELETE /v1/me/discord`, clearing the field and the index).

A Discord account links to one tokera account; linking it again moves it, and the
previous account's field is cleared. A code is not a secret worth more than its ten
minutes: it names no account and grants only that link.

**An unlinked author** is not heard in the world. Their first message to a bot is
answered once — "I don't know you yet. Open <console>/#/profile, choose Link Discord,
and send me the code." — and further messages from them are dropped silently for an
hour, so a stranger cannot make a bot talk on a loop. The bridge keeps that
"told" set in memory; a restart tells them once more.

A linked person is then the same author in both places: their Discord messages arrive
as `WorldOp::Message { from: <their speaking name> }`, the name `speaking_as`
(`engine/mod.rs:1153-1166`) gives their account, exactly as `post_message`
(`engine/mod.rs:1179-1234`) builds it — so a conversation started in the console goes
on in Discord, one thread. **Who may DM a character** is who may see it: the
`visible_to` check `owned_by` makes (`engine/mod.rs:672-692`). A linked person DMing a
character they may not see is answered once, "I can't talk with you", and nothing
enters the world. A channel post from Discord still needs the Creator role.

### 3.3 Where a bot runs, and how it shows

**A bot is online where its character's world is run.** A character's body is in one
world; that world's authority (`authority::driver`) is the one member that may write
it. So the member holding a world's authority opens a gateway shard for every
provisioned bot of a character with a body in that world, and closes it when it loses
the authority (§5.4). Discord allows one gateway connection per token (§2), and this
is what keeps it at one: two members never hold one world.

- **No world, no bot.** A character with no body in a running world has no shard; its
  bot shows offline on Discord, and a DM to it waits on Discord until the world runs
  again, when the gap fetch (§5.3) picks it up.
- **A record change moves the shard.** The authority watches `discord/bot/<npc_id>`
  for the characters it runs: a new or changed token opens (or reopens) the shard, a
  tombstone closes it.
- **Presence.** An open shard sets the bot online, with an activity line naming the
  world ("in Life"). Discord shows it offline once the shard closes.

**The bot's state** is the replicated custom object `discord/state/<npc_id>`, written by
the member running the shard on each change (not on every beat):

| `state` | Meaning |
|---|---|
| `online` | a shard is open and Discord answered `READY`; `member` names who runs it |
| `connecting` | a shard is opening or resuming |
| `revoked` | Discord answered 401 to the token, so the shard was stopped (§6); the owner must connect a fresh token |
| `no_world` | the character has no body in a running world |

`{ state, since_ms, member, world }`. `GET /v1/npc/:nid/discord` (§4.3) answers it as
`status`, and the Messages-tab panel shows it beside the bot's name: "online in Life",
"offline — Zen is in no running world", "Discord refused the token — connect a new one".

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
| `discord/guild` | `{ guild_id, listener: <npc_id>, backstage: bool }` — the world's server (§4.4) |
| `discord/thread/<thid>` | `{ channel_id, kind, mirrored_through: seq, last_seen_discord: message_id }` |
| `discord/dm/<npc_id>/<account>` | `{ channel_id, user_id, mirrored_through, last_seen_discord }` — a bot's DM with a linked person, keyed by the person's account key, not their display name |

A character's bot is not a world record: it belongs to the character, whichever world
it is in, and is kept beside it (§4.3). Nor are a person's link (§3.2) and a bot's
state (§3.3): those are custom objects outside any world, `discord/user/…`,
`discord/link/…` and `discord/state/…`.

### 4.3 Bot tokens

A character's bot is the replicated custom object **`discord/bot/<npc_id>`**
(`npcd/src/discord/store.rs`, the pattern of `accounts.rs`), not a world record:

```json
{ "application_id": "<id>", "user_id": "<id>", "username": "<name>",
  "token": "<sealed>", "connected_ms": 1793000000000 }
```

The token is sealed with **ChaCha20-Poly1305** under a key the cluster derives
(`join.derive_key("npcd discord tokens")`, the pattern of `image_links.rs`; a daemon
with no cluster mints its own at start, so a token it sealed must be connected again
after a restart). A sealed token is `base64(nonce(12) ‖ ciphertext ‖ tag(16))`, a fresh
nonce per seal, with the record's key as associated data, so a sealed token copied onto
another character's record does not open (`npcd/src/discord/seal.rs`). Every member can
therefore take over a world and its bots. The token is never in the character record
(which `/v1/npcs/export` hands out whole), never in git, never logged, and never in any
route's answer, sealed or not.

Three routes, the owner's like every route on a character (`User`, then ownership; a
character that is somebody else's is the same 404 as one that does not exist):

| Route | Does |
|---|---|
| `GET /v1/npc/:nid/discord` | `{"connected": false}`, or `{"connected": true, application_id, user_id, username, connected_ms, invite_url, status}` — `status` the bot's state (§3.3) |
| `PUT /v1/npc/:nid/discord` `{"token"}` | checks the token (`GET /users/@me`, refusing a user's token; `GET /applications/@me`), names the bot as the character (2–32 characters, cut to fit) and pictures it with the portrait if there is one (`PATCH /users/@me`, sending only what differs — Discord allows few renames an hour), seals the token, writes the record, and answers as `GET` |
| `DELETE /v1/npc/:nid/discord` | writes the record's tombstone; `{"connected": false}`. The bot stays the owner's in the portal |

A token Discord refuses is `400 {"error": "discord_token_rejected"}` (a user's token,
`discord_not_a_bot`); a rate limit is `429 discord_rate_limited` with `retry_after` and a
`Retry-After` header; any other failure is `502`. None is retried, for the per-IP limit
(§2), and nothing is written unless every call succeeded. `invite_url` is
`https://discord.com/oauth2/authorize?client_id=<application_id>&scope=bot&permissions=344134372416`:
View Channels, Send Messages, Read Message History, Embed Links, Attach Files, Add
Reactions, Send Messages in Threads, Create Private Threads and Manage Webhooks
(`npcd/src/discord/invite.rs`).

In the console it is the **Discord** panel at the top of the character's Messages tab.
Unconnected, it offers "Set up Discord", which opens the steps: open the Developer
Portal and make a New Application named as the character; open Bot, Reset Token and
copy it; paste it into the panel and Connect. A refusal is shown under the field.
Connected, it says which bot is the character and its status (§3.3), links the invite,
and has a Disconnect that asks first. Beside the invite it says what the invite is for:
"Discord opens a DM with a bot only when you share a server with it. Invite <name> to
any server of yours — the world's, or your own — then DM it there." The invite page lists
only servers where the person has Manage Server, so the panel adds "No server listed?
Create one in Discord with + → Create My Own."

### 4.4 The world's server

A world's Discord server is set in the console, by a person with the Creator role, on
the world's page (`web/content/npcd/pages/worlds.js`), in a Discord section:

| Route | Does |
|---|---|
| `GET /v1/world/:wid/discord` | `{"guild": null}`, or `{"guild": {guild_id, name, listener, backstage}}` |
| `PUT /v1/world/:wid/discord` `{guild_id, listener, backstage}` | checks that `listener` is a character with a body in the world and a connected bot, then asks Discord as that bot (`GET /guilds/{guild_id}`) — the bot must already be in the server, and Discord's answer gives the server's name; writes the world's `discord/guild` record through the authority (a `WorldOp`), and answers as `GET` |
| `DELETE /v1/world/:wid/discord` | tombstones the record; the mirror stops, and the bots' DMs (§3.3) go on |

The section lists the world's characters that have bots, to pick the listener from,
and says what the listener needs: `MESSAGE_CONTENT` turned on under Bot in the portal,
and the invite link used once to add it to the server. A refused `PUT` says which of the
three checks failed. The record is the world's, replicated with it, so a world moved to
another member keeps its server.

---

## 5. Keeping it in sync

Everything runs on the **world's authority**: every message is written there first, so
the outbound side sees it in the same beat, and an inbound message becomes a `WorldOp`
without a hop. The bots and the bridge are started and stopped with authority:
`authority::driver` already decides when a member takes or loses a world, and both
follow it.

### 5.1 Outbound — npcd to Discord

Each world beat (`authority/beat.rs:69-83`, after `write_world`), the world's bots and
bridge walk the threads and, for each mapped thread or DM, post every message with
`seq > mirrored_through` and no `origin`:

1. **The text is the message's own.** A message holds what was said, `intent`
   (`phone.rs:193-203`), and the console shows exactly that: `get_messages` reads it
   off the thread (`runtime.rs` `messages_with`) with no model call. Discord is sent the
   same text, so the two never disagree and a post costs nothing to render.
2. Posted by the right sender (§3.1), split at 2,000 characters on a sentence boundary.
3. `mirrored_through` advances only on Discord's `200` with the message id it returned.

**A DM is posted to the person's DM channel with the bot.** Its channel id is the
`discord/dm` record's; a person who has linked but never DMed this bot gets one opened
(`POST /users/@me/channels {recipient_id}`) the first time the character writes to
them, which Discord allows only if they share a server (50278: the message stays
unposted, and the panel's status says "can't reach <person> — no shared server").

**Typing.** From when a DM from Discord is delivered to the character
(`deliver_messages`) until the character answers on that thread or its turn ends —
two minutes at most — its bot shows typing in that DM (`POST /channels/{id}/typing`,
renewed every 8 s, as Discord's indicator lasts 10), so a person who wrote sees the
character is answering.

A post that fails with 429 waits `retry_after`; one that fails otherwise leaves the
cursor where it was, and the next beat tries again — so a gap is retried, never
skipped. Posting is per channel and in order.

### 5.2 Inbound — Discord to npcd

The listener (for channels) and each character's bot (for its DMs) receive
`MESSAGE_CREATE`. The bridge drops:

- a message any of the world's own bots or webhooks wrote (by author id and
  `webhook_id`) — the echo of an outbound post;
- one from an unlinked person (§3.2), after answering them once;
- one already applied (its id is a recorded `origin` in the thread).

**A DM** arrives on one bot's shard, so it names its character: the bot's
`discord/bot/<npc_id>` record. Its author's `discord/user/<id>` index names their
account. Then, in order:

1. A message that is exactly a live link code (§3.2) links the account, and goes no
   further.
2. A person who may not see the character (`visible_to`) is answered once and dropped.
3. Otherwise it is `WorldOp::Message { npc, from: speaking_as(account), text }`, the op
   `post_message` writes, with `origin`, and the `discord/dm/<npc_id>/<account>`
   record is written or advanced.

**The channel and groups** arrive on the listener's shard and apply as the console's
routes would: the channel as `WorldOp::Channel`, a group as a send into that thread,
each carrying `origin` so it is never posted back. `last_seen_discord` advances with
each.

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
- **Bots.** Every provisioned character's bot is connected, in the guild when the world
  has one, and named and pictured as the character currently is; its state (§3.3) is
  what it says.
- **DMs.** The gap fetch covers every `discord/dm` record too, so a DM sent while the
  world was not running is answered once it is.

### 5.4 Authority moves

When the authority moves, the old holder closes its gateway connections — every bot of
the world's cast, and the listener — and writes each bot's state `connecting`; the new
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
that bot `revoked` (§3.3) and stops its shard rather than retrying; a new token
connected in the panel reopens it.

**A shard is a task of the authority**, one per bot, in `npcd/src/discord/gateway/`:
it IDENTIFYs with `DIRECT_MESSAGES` (and, for the listener, `GUILDS`,
`GUILD_MESSAGES`, `MESSAGE_CONTENT`), resumes on a dropped connection, and hands each
`MESSAGE_CREATE` to the world's inbound queue (§5.2), which the beat drains. It never
writes the world itself.

---

## 7. What it costs a person to set up

**To DM one character** — the first thing, and enough on its own:

1. Its owner follows **Set up Discord** on the character's Messages tab (§4.3): a New
   Application named as the character, Bot → Reset Token, the token pasted in and
   connected. npcd names the bot and pictures it.
2. The panel's "Invite to a server" adds the bot to a server the people who will DM it
   are in — the owner's own is fine.
3. Each person links once, on their Profile page (§3.2): a code from the console, sent
   to the bot in a DM. From then on they DM the character as they would anyone.

**To mirror a whole world** — the channel, the groups, the other characters:

1. Create a Discord server for it, and a Discord **team** that will own the world's
   bots (so they are not one person's).
2. Choose the listener — a character with its own bot — and turn `MESSAGE_CONTENT` on
   under its Bot page; invite it to the server.
3. On the world's page, set the server and the listener (§4.4). Every character without
   a bot needs nothing: it speaks through the persona webhook.

---

## 8. Build order

Ordered so that a person can talk to a character on Discord as early as possible,
since that is the way in.

1. **Built:** sealed bot tokens and `GET`/`PUT`/`DELETE /v1/npc/:nid/discord`, with the
   Messages-tab panel (§4.3).
2. `Message.seq` / `at_ms` / `origin` and `Thread.next_seq` (§4.1), with tests that a
   trimmed thread keeps every surviving message's `seq`.
3. The bot runtime (§3.3, §6): a shard per bot on the authority of its character's
   world, presence, the `discord/state` record and the panel's status. A fake Discord
   gateway in tests drives `HELLO`/`READY`/`RESUMED`, a 4004 close (revoked), and a move
   of authority closing one shard and opening another.
4. Account linking (§3.2): the Profile section, the code routes, redeeming a code over a
   DM, the unlinked answer and its hour of silence.
5. DMs both ways (§5.1–5.2): a linked person DMs a character's bot and the character
   answers there — the first thing anyone uses — with typing, the visibility refusal
   and the DM gap fetch. A fake Discord HTTP server in tests asserts the exact requests,
   echo suppression, `origin`, and that a 429 and a 500 are retried in order.
6. The world's server (§4.4) and the channel both ways.
7. Groups as channels, persona webhooks and private-thread "DMs", and the reconcile
   (§5.3).
8. Authority moves (§5.4), with a two-member test that the cursor carries over.
9. Backstage threads for character-to-character DMs.

---

## 9. Decided

What this design settled, and where:

- **Rendering.** A message is stored as what was said, and the console shows that
  text without a model call; Discord gets the same text (§5.1).
- **Where a bot runs.** On the authority of its character's world, independent of any
  world server (§3.3).
- **A DM needs a shared server**, any server; the panel says so (§4.3).
- **Who may DM a character** is who may see it (§3.2).
- **The first world** is `life`: Zen and its Drones, and the family. Battle Cities'
  cast is larger than one team's 75 and comes after, with most of its characters on the
  persona webhook.
- **People without Discord** keep the console, which stays a full way in: it reads and
  writes the same threads.
- **Out of scope:** Discord voice channels, slash commands, and replaying Discord edits
  and deletions into the world (§5.2).
