# Submitted sections — per-conversation collection members

A schema's sections are sealed when a conversation is created and belong to every
conversation built from it. A **submitted section** is a member of a top-level
collection that exists for one conversation only, added after that conversation
opened. `Sequence::submit_section` returns a `SectionRef` at once; the section is
tokenized and sealed on a background thread and joins the collection's projection
candidates from the next turn after it seals.

## API

| Call | Effect |
|---|---|
| `submit_section(collection, name, content, priority) -> SectionRef` | Allocate an id, claim it for this timeline, seal in the background. |
| `SectionRef::wait()` / `state()` | Block until sealed (`Ready{blocks}`), or observe `Pending` / `Failed` / `Removed`. |
| `remove_section(&SectionRef)` / `remove_section_named(name)` | Drop the member, its K/V and its stream. |
| `submitted_sections()` | Handles for the sections this conversation holds. |

The name is unique among the conversation's submitted sections and may not equal a
schema section's name. Submitting a name again replaces the earlier section. The
collection must be top-level: a collection inside a section tree seals each member
once per branch, and submission refuses it with an explicit error.

## Ids

Owned ids come from the substrate allocator (`OwnedSections`), in a partition below
`TRANSIENT_FLOOR`. A builder-allocated id is max+1 over the schema's ids, so nothing
is builder-allocated on the merged clone; `Builder::add_section_to_collection_with_id`
takes the id the substrate already chose.

## Sealing and the address

The prefix a member seals against is the walk `prefix_before_collection` makes over
the system prompt: bare sections, non-template members of earlier collections, and a
section tree's default-present ids, up to the target collection. The address is
`ContentAddress { prefix_hash: owned_section_prefix(timeline, name, prefix_hash), section_hash }`.
The salt keeps two conversations that submit identical content from sharing a
stream, which matters because removal tombstones the stream.

The sealed K/V goes through the normal persisted path: declare stream, cold persist,
`RestoreSection` after a restart when the same content is submitted under the same
name. A failed restore falls back to ingest.

## Projection

`Sequence.projection` is the merged builder (base plus Ready sections) and all
internal readers use it. `Sequence.base_projection` is what the caller supplied.
`projection()` and `set_projection()` operate on the **base**, because a caller that
rewrites the schema each turn (npcd's journal layer does) would otherwise find the
overlay's names already present and duplicate them. The overlay is re-merged when a
turn is submitted. Forks start with the base and an empty submitted list.

## Removal and tombstoning

Removal marks the handle `Removed`, drops the claim, writes a `SectionTombstone`
record, drops the in-memory entry and sends `RetireSections` so the scheduler tables
are cleared. Tombstoning the timeline cascades the same release to every owned
section. While a turn is in flight the release is deferred, because the turn may be
attending the K/V; it is flushed when the next turn is submitted.

A seal that finishes after its claim was released (removed or tombstoned
mid-seal) retakes the claim under a distinct key, releases it and retires the id,
so a newer same-name section is never clobbered.

## Known limits

- A fork's tombstone does not clear the scheduler tables for sections the fork
  itself submitted; the substrate entry is released.
- Recurrent / hybrid models: whether a member sealed mid-sequence is gap-filled
  correctly in recurrent state has not been verified. Test it on a hybrid model
  before relying on submission there.
