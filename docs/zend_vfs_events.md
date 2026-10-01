# Zend: A Conversation's Files as Substrate Events

Status: built.

## 1. Summary

A conversation's changes to a repository's files live in its file store (`zend_vfs::VfsStore`): per path, a chain of deltas over the conversation's pinned base (`docs/zend_git.md` §7.9). The substrate keeps the same thing as **events**: one record per delta, owned by the conversation's timeline, appended as the conversation works. The two mirror each other without either calling the other: after every tool round zend compares each store with what the substrate holds and writes the difference; when a conversation is built again, its events are read back and replayed into fresh stores.

The lifecycle follows the conversation's:

- A path whose chain the store let go of — committed (`git_commit`), discarded (`git_reset --hard`), or rewritten (a merge, a move of the base) — has its events **explicitly tombstoned**: one tombstone record per path, killing every event of that path so far.
- A conversation that is **tombstoned** takes its events with it **implicitly**: no record is written per event. Its events are orphaned — their parent is dead — and every rewrite of the log (compaction, maintenance) stops carrying them, exactly as it stops carrying the rest of the conversation's records.

## 2. Records

Two record types, both keyed in the header by the conversation's timeline (`stream_id`) and a **sequence number** (`chunk_index`) from one counter per timeline. The payloads are JSON.

| Type | Tag | Payload | Read at open |
|---|---|---|---|
| `VfsEvent` | 25 | `{ timeline_id, seq, repo, key, body }` — `body` is opaque to the substrate | no — only when the conversation resumes |
| `VfsTombstone` | 26 | `{ timeline_id, seq, repo, key, kills: [seq…] }` | yes |

`repo` names the repository; `key` names what the event belongs to inside it — a path, or the empty key for the store's base. The substrate never interprets either, nor `body`: it keys by `(timeline, seq)`, which the header carries, so opening the log never reads an event's body. A tombstone lists the sequence numbers it kills rather than naming a key and a range, for the same reason: the index can drop them without having read the events.

Sequence numbers are assigned by the persistence layer when a batch is written, one counter per timeline, continuing from the highest it has seen — so a tombstone's `kills` always names events written before it.

A batch writes its **events first, then its tombstones**. It is staged, not committed: the persistence thread's group commit makes it durable with everything else staged beside it, so a tool round never waits on an `fsync`. Reading a timeline's events back commits whatever is staged first, so a conversation built again moments after a save reads that save.

The log has no batch marker: after a crash it keeps every record written before the tear and none after, so any prefix of a batch can survive. Two orderings make every prefix safe. Tombstones come after events, so a batch cut short leaves the old events beside the new ones, never a tombstone with nothing written in its place. And each path's events end with its `state`, which is what commits the deltas before it (§4), so deltas cut off from their state are recognised as torn and dropped by replay rather than half-applied (§5).

## 3. The index and the log

`SubstratePersistence` holds a `VfsIndex` — per timeline, the location of every **live** event and every tombstone — the way it holds `npc_locs`: built by the open walk, kept current by every append, rebuilt after compaction. Event bodies are never in RAM.

| Happens | Index |
|---|---|
| a `VfsEvent` is appended or walked | its location recorded, unless a tombstone already killed its seq or its timeline is tombstoned |
| a `VfsTombstone` is appended or walked | the events it kills dropped — their bytes counted dead — and its own location recorded |
| a timeline `Tombstone` is appended or walked | the timeline's whole entry dropped, its events' bytes counted dead; later records for it are ignored |

**Survival** (`persistence/survival.rs`): `VfsEvent` is `Relocated`, carried verbatim by location; `VfsTombstone` is `Maintained` — relocated by maintenance, left behind by compaction.

- **Compaction** carries every live event of a timeline that is registered, not tombstoned and not distilled — nothing else. Tombstones are not carried: the events they killed do not survive the rewrite either, so nothing is left for them to kill. An event whose timeline never registered is an orphan and is dropped the same way.
- **Maintenance** relocates, off its target segments, the live events of live timelines and the tombstones of live timelines — each only if the index still names it where the plan found it, so a tombstone written between plan and execute is never out-run by a copy of an event it killed. A tombstone must outlive every copy of the events it killed, and those may sit in older segments the op does not touch; relocating it forward keeps it after them in log order. A tombstoned timeline's events are relocated by nothing: the timeline's own `Tombstone`, which maintenance re-emits, keeps them dead until the segments holding them are dropped.
- **Liveness** counts the bytes of exactly what those two carry, so a segment holding only dead events reads as reclaimable.

## 4. Mirroring (zend)

`zend/src/conv_overlay/`. Each conversation keeps a **mirror** beside its file stores: exactly what its live events say — per repository the base and, per path, the chain the events make — with the sequence number of every live event behind each. After every tool round, for each repository:

- **A path whose chain only grew** — the mirrored deltas are a prefix of the store's — has its new deltas appended as `delta` events, then a `state` event (size, conflict flag). The state is written after any new delta whether or not it changed, because it is what commits them; with no new delta it is written only when it changed.
- **A path whose chain was rewritten** — not a prefix — is written whole again, its first event marked `start`, and its old events tombstoned. So is a path a torn save left (§5): an append would be read as following the deltas replay dropped.
- **A path the store no longer holds** — committed, discarded — has its events tombstoned.
- **The base**, when it moved, is written as a `base` event and the previous one tombstoned.

The mirror records a write by the same two rules replay reads events with — absorb each event, then apply each kill — so the mirror after a write is the one a replay of the log would build.

The mirror starts **unread**, and an unread mirror writes nothing: until the conversation's events have been read back, anything written would sit beside events the mirror never knew of, and a later replay would read both. A write that fails part way leaves in the log whatever landed, so the mirror is read back from the log rather than guessed; if that fails too, it is left unread and nothing more is saved for the conversation until it is built again.

Event bodies (`conv_overlay/event.rs`):

| `kind` | Key | Carries |
|---|---|---|
| `base` | empty | the store's base: tree and parents; the latest wins |
| `delta` | the path | one timed delta, appended to the path's chain — or, with `start`, beginning the chain again — once a `state` commits it |
| `state` | the path | commits the deltas since the last state; carries the chain's size after its last delta (`null` once deleted) and whether it is in conflict; with `start`, a chain with no delta at all — a conflict kept exactly as the base holds it |

## 5. Resume

When a conversation's state is built, its live events are read in `(repo, key, seq)` order and replayed into one `Snapshot` per repository — the latest `base`, and each path's chain as its last `state` committed it — and restored into the fresh stores. Deltas read since a path's last state wait for the next one: a `state` commits them, extending the chain or, when they began with `start`, replacing it; a later `start` supersedes whatever waited.

Deltas still waiting at the end are what a torn save left. They are dropped: the path keeps its last committed chain — intact, since the tombstones that would have killed it come after the events and never landed — and a path no state ever committed is not restored at all. The rest of the repository replays as normal: one torn path never takes the others with it. The mirror is exactly what was read, including every event a `start` superseded or a tear left uncommitted, and marks a torn path so the next save writes it whole and kills them. An event under a key it cannot be under, or a body that is no event, makes the repository's events unreplayable, never replayed into a wrong state.

A repository whose events do not replay, or whose snapshot the store refuses (a base the repository no longer holds, a chain past the size cap), is not lost: its events are copied under the repository name `<repo>#unrestored-<n>` — copies first, originals tombstoned after, so a move cut short leaves both — and its store starts fresh. No repository has that name, so it is never restored into anything and never tombstoned by a mirror — kept for good. When the events cannot be read at all, the mirror stays unread and nothing of the conversation is overwritten.

## 6. What replaced what

`ConvState` keeps the conversation's `archived` flag and the branch it is on in each repository. The file changes it used to carry whole, as one value per repository rewritten after every round, are these events. There is no migration: a store written by the earlier form loses its conversations' uncommitted changes.

## 7. Testing

- Payloads encode to exact bytes and round-trip (`persistence/vfs/payload.rs`).
- The index: an event after its tombstone is dropped, one before it too; a timeline tombstone drops everything and ignores later events; sequence numbers continue past every seq seen; killed bytes are counted dead.
- Survival: a store holding events and tombstones of a live timeline, a tombstoned one and an unregistered one — compaction carries exactly the live events; maintenance relocates live events and tombstones and nothing of the dead; reopening reads the same events.
- Staging: a write is staged, not committed, and reads back at once.
- zend: the mirror's difference for every case in §4 — growth (always ending with its state), a changed state, a rewrite, a dropped path, a moved base, a repository gone, a chain with no delta, a torn path — as exact batches; an unread mirror writes nothing; replay builds the snapshot a store saved, the right chain from a batch cut short between its events and tombstones, and the last committed chain from one torn inside its events — a rewrite, an append ending in a delete, a new file — without losing the repository's other paths; a `start` after a tear supersedes it; events under the wrong key or that are no event are refused; changes written over several rounds replay into stores holding exactly what the originals did, and the replayed mirror writes nothing more.
