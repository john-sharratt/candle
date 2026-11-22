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
| `submit_section(collection, name, content, priority, after, sharing) -> Result<SectionRef>` | Allocate an id, claim it for this timeline, seal in the background. `after` names the submitted sections this one continues (empty for a section that stands alone); `sharing` says whether the text is this conversation's own or shared (`docs/shared_journal_sections.md`). |
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
section tree's default-present ids, up to the target collection, followed by the
`after` sections it continues. The address is
`ContentAddress { prefix_hash: owned_section_prefix(timeline, name, prefix_hash), section_hash }`.
The salt keeps two conversations that submit identical content from sharing a
stream, which matters because removal tombstones the stream. A section submitted
with `Sharing::Shared` is addressed by content alone (`shared_section_prefix`)
instead, so every conversation that would seal exactly that text restores one
stream.

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

## Recurrent models

A section's recurrence starts where the prompt before the collection, then the
sections it continues, leave it (`SealJob::start_state`). The section it
continues directly hands over the recurrent state its seal ended on when it was
sealed here; one restored from the log hands over the end state its stream
keeps when it ended a shared run (`docs/shared_journal_sections.md` §4.1);
otherwise the start is worked out from the state after the schema's prefix,
run on through the sections it continues (`prefix_state::state_after`,
`continue_state`). A host copy of an end state is the largest thing a section
holds, so it is let go once a later section and that section's successor have
sealed (`forget_passed_end_states`).

The projection's gap-fill is not what carries a submitted section's recurrence.
A model whose `model_core_properties().can_gap_fill` is false
(`carries_recurrent_state()`) has no glue island reserved for it: the assembler
refuses one (`projection_assembler/walk.rs`), because a token inserted
mid-sequence has no computable output. The seal's carried state is what keeps a
section read after others computed as it is read.

## Tombstoning a timeline

`Engine::tombstone_timeline` releases every section the timeline owns, forks
included (each section is owned by the timeline that submitted it), and clears
the scheduler's per-section tables for them with `RetireSections`.
