# Renaming a World

A world's **id** is the key everything it keeps is filed under: its records in
the substrate, its characters' bindings, its folders in the mind. A world's
**name** is what it is shown as, and changing that changes nothing else
(`worlds/<id>.yaml` `name:`, and the map's world area `name:`). This document is
how a world's id changes **without losing anything it has** — the first case
being `life` → `moon`. This document is the design; the code answers to it.

## 1. What carries the id

| What | Where | On a rename |
|---|---|---|
| The world's records — sim, bodies, missions, the Discord mirror's guild, threads, webhooks and channel bindings, servers, DMs, NPC leases | substrate custom objects under `world/<id>/` (`world_state::keys`) | **taken in** (§3) |
| Each character's world | the NPC record's `world_id` (`npcs.rs`, `Casting`) | **rewritten** (§3) |
| The world's mind files | `worlds/<id>.yaml`, `map/<id>/`, `missions/<id>.yaml` and `missions/<id>/`, `rounds/<id>.yaml`, `standing/<id>/` | **moved** in the mind, in one commit (§4) |
| An operation's store | a timeline named by a hash over the world id (`opstore::owner`) | **drained first** (§4): none may be open |
| An owner's unpublished drafts of the world's mind files | the owner's draft store, by path | **published or dropped first** (§4) |
| The authority lease | Paxos key `world/<id>` | none: renewed every 10 s, the old one lapses |
| A bot's state | `discord/state/<npc>` `world` | none: rewritten by the shard that runs it |
| Bodies' bindings, authority roles | in memory | none: rebuilt from the records at load |

Words that merely *say* "life" — `layers/life/…` (a character's life story),
`life:<who>` tags, `lifegen` — are not the world's id and are not touched.

## 2. Declared in the mind: `formerly`

The world file under its new id names the id it had:

```yaml
# worlds/moon.yaml
id: moon
name: Moon
formerly: [life]
```

`formerly` is a list, because a world renamed twice keeps both. A world id named
in any world's `formerly` is **never a world of its own again**: a
`worlds/<old>.yaml` left beside a world that names it as former is refused at
hosting, with an error naming the file to remove (`tenant::host_worlds`).

## 3. Taken in once, by the authority

When the world's authority loads a world that has **no records of its own** and
names a `formerly`, it takes the former world's in, the way a world kept before
records was taken in (`world_state::legacy`):

1. **Records.** Every live object under `world/<old>/` is written under
   `world/<new>/` with the same entity and payload, in one substrate batch.
2. **Characters.** Every NPC record with `world_id: <old>` is written with
   `world_id: <new>`.
3. **The old records are tombstoned** once the new ones are written — never
   before, so a crash between the two leaves both whole, never neither (§5).
4. The world then loads from its records as any world does: the same sim, the
   same characters in the same rooms, the same Discord server and channels
   (their records moved with it; the channels in Discord are untouched).

Read only for a world with no records of its own, so once taken in nothing here
is consulted again. A world whose former has no records is left as seeded.

**Ceasing is told apart from renaming.** A world gone from the mind has its
records ceased (`cease_world`) — except one named in another world's
`formerly`, which is the new world's to take in, and is never ceased.

## 4. The rename, in order

1. **Code first.** A build that knows `formerly` is deployed everywhere; with no
   world naming one, it changes nothing.
2. **Settle what the old id holds outside its records.** No operation open in
   the world (`life` runs none), and no unpublished draft of its mind files.
3. **One mind commit** moves every path in §1's mind row from `<old>` to
   `<new>`, sets `id:` and the map's `within:` lines, and adds `formerly`.
4. The next load takes the world in (§3) on the authority; the other members
   take the batch by replication.
5. **Confirm**: the characters are where they were, Zen's agenda and threads
   are as they were, the Discord mirror posts into the same channels, and no
   object under `world/<old>/` is live.

## 5. Where it lives, and its tests

- `world_state::renamed` — the formers read off the worlds' files, the take-in
  and the retire. Tests: a world names the ids it had; a renamed world takes in
  its former record for record, the former is retired, and a late write under
  an earlier epoch stays dead.
- `authority::beat` — `restore` takes the former in when the world has no
  records of its own; `write_world`'s first success retires it. Test: a world
  renamed comes up with its bodies where they stood and its table as it was,
  writes its own records, and the former's are gone — once.
- `authority::worlds::Standing` — the formers, and which world is owed a
  retire. Test: it is handed back once.
- `Npcs::rehome` — the characters moved, by `Authored::renamings` at startup
  and at every reload. Test: each character of the old id moves, another
  world's stays, a second move finds none.
- `Authored::reload_authored` — a gone world that is a former id is unhosted,
  never ceased.

The take-in is two writes, not one batch: the new world's records at its
authority's first beat, then the former's tombstones. A failure between them
leaves both, and the next boot finds the new world's records and takes nothing
in; the former is retired by hand (`store::cease_world`) or on a later rename.
