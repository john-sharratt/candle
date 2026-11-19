# Shared journal sections

A new conversation's first token waits for its journal. This document makes the
part of the journal that is not the conversation's own a **pre-sealed, shared,
replicated cache**: read in milliseconds, computed exactly as the conversation
would compute it, and held nowhere but the substrate until a conversation reads
it.

## 1. The problem, measured

The first round of a new conversation waits up to `OPENING_SEAL_WAIT` (2 s) for
its journal to seal (`zend/src/journal/opening.rs`). On an idle BlackWorld
(2026-11-19) that wait was 0.98–1.6 s of a 1.4–2.0 s first token. A Zen chat's
journal is six sections in one chain (`Linking::Chain`), each sealed after the
one before it, ~160 ms a link:

| # | Section | Tokens | Holds | Whose |
|---|---|---|---|---|
| 1 | `lineage/N` | ~230 | an ancestor's journal (project rules, thread notes) | the thread's |
| 2 | `journal/history` | 3 | the entries' heading | everyone's |
| 3 | `journal/1` | ~120 | opening entry: what the priming chain read | the member's, per hour |
| 4 | `journal/2` | ~32 | opening entry: "then read other files … opening question" | everyone's, per hour |
| 5 | `journal/close` | ~32 | the open items (none) and the journal's end | everyone's |
| 6 | `lineage/reader` | ~56 | "This conversation is T:<thread>/<name> …" | the conversation's |

A conversation not owned through `/mcp` carries 2–5 only. Memory topics
(`memory/N`, `zend/src/mcp/memory_sections.rs`) sit between 1 and 2 when the
thread is about any, and are the thread's.

Only 6 is the conversation's own. 1–5 are the same text, after the same system
prompt, for every new conversation of a thread within one hour — an entry's
date line names its day and hour (`Entry::when`). They are prefilled anew each
time only because an **owned** section's address is salted with its owner
(`owned_section_prefix`), so no two conversations ever share one.

## 2. What is shared

A section of a chained collection is **shared** when its text is no
conversation's own, and so is every section before it in the chain. The run of
shared sections at the head of a chain is its **shared run**; everything after
it is owned, as today. For a journal:

- shared: lineage journals, memory topics, the history heading, the opening
  entries (`from_turn == to_turn == 0`), and the closing section while the
  journal holds only the opening;
- owned: the reader's line, every entry drafted from the conversation's turns,
  and everything after the first owned section.

zend decides which sections are shared; the crate takes the decision with the
section (`Sharing`).

## 3. Addressing

A shared section is addressed by content alone:
`prefix_hash = shared_section_prefix(chain)`, a domain-separated hash of the
content chain of the prompt prefix and the sections before it, and
`section_hash = hash_tokens(tokens)`. No owner, no name. Two conversations
that would compute the same K/V address the same stream.

Its stream is declared under `debug_name = "shared/<name>"`, which is what
tells a shared stream apart (`StreamIndex::sections_named_under("shared/")`).

A cache entry is therefore **only ever hit by a conversation that would have
computed exactly it**: any difference in the prompt, in an earlier section, in
the text, in the date line, or in the model's key moves the address. A stale
entry cannot be read; it is merely not hit.

## 4. One code path: a hit is a seal someone else ran earlier

There is no cache read path. A section whose stream is persisted is restored
instead of prefilled (`SealJob::seal`, `RestoreSection`) — that is how an owned
section returns after a restart, and it is how a shared one is hit. A miss is
today's prefill, unchanged. Whoever seals a shared section first — the
pre-warmer (§7) or a conversation that got there first — produces the entry
every later conversation restores.

### 4.1 The state a chain runs on

A chained section starts its recurrence from the state the section before it
ended on. Sealed in one run that state is handed over in memory
(`SectionRef::end_state`); a restored section hands over nothing. So **the last
section of a shared run stores the state it ends on, in its own stream**
(`SectionEndState`, §5), and the first owned section after a restored run
starts from it.

- Full miss: the run is prefilled; the owned section starts from the state the
  last shared section handed over, which that seal also stored.
- Full hit: the run is restored; the owned section starts from the stored state
  — the bytes the producer's seal handed over.

On one member these are the same bytes: the producer ran the same seal, alone,
on the same model. That is the equivalence this design promises, and the test
in §8 asserts it on raw bytes.

A **partial** hit — the head of a run restored, the rest prefilled (an hour
turning over mid-chain, a lineage note landing) — starts the first prefilled
section as a mid-chain reseal does today: `continue_state` over the restored
sections. The pre-warmer exists so that this is rare.

Across members a hit reads the producer member's bytes, as every shared schema
section already does (`StreamIndex`: two writers kept, the newest stamp read).

### 4.2 The form a later section attends over

A submitted section is sealed native and quantized at the next turn's seal
(`quantize_pending_sections`), because a build still in flight reads the native
bytes. Left so, an owned section prefilled straight after its shared run in the
conversation that sealed the run would attend over the run's native K/V, while
one in a conversation that restored the run attends over the quantized K/V the
substrate keeps — a hit and a miss reading different bytes. And a run sealed
ahead of any conversation would never reach disk, having no turn of its own.

So **a shared section is quantized at its own seal** (the `SealAction::Section`
arm). Nothing reads it yet — the section continuing it waits for this seal to
settle — so every reader after it, shared or owned, here or in a conversation
that restores it, attends over the one form the substrate keeps, and the
section is persisted at once.

## 5. Storage

- **Sections** are ordinary section streams: declared, chunked, tokens, index
  page, written to the redo log, replicated, tiered hot → warm → cold.
- **A section's end state** is a third kind of `Snapshot` record, beside a
  conversation's snapshot and a prompt branch's checkpoint, told apart by its
  stream id alone: `section_end_state_stream_id(section stream)`. Its payload
  is a `SnapshotPayload` with no timeline and no turn
  (`conversation/section_end_state.rs`), written by
  `WriteJob::SectionEndState` — the snapshot append and single tail, reported
  as no slot's state. Written only for the last section of a shared run
  (`Sharing::SharedEnd`), after the seal settles, so the section continuing it
  starts from the handed state without waiting on the copy. It carries an
  identity, so it replicates with the stream; ending the section's generation
  drops it (`Substrate::end_section`), and a later generation of the same
  content keeps one again. It is read when a section follows a restored one
  (`SealJob::start_state`), and never held after the install.

Nothing of a shared section is resident until a conversation reads it.
`RestoreSection` installs a cold marker; the projection that selects the section
lifts it (`elevate_to_hot`). The pre-warmer lets go of what it sealed once its
chunks are durable (§7).

## 6. Lifetime

- A conversation **never tombstones a shared stream.** Releasing a shared
  section drops the conversation's claim and its in-memory entry only
  (`release_owned_section` on a `shared/` stream).
- Shared streams are retired by the **sweep** (§7.3): one no current chain names
  and no live section entry holds, older than the grace period, is tombstoned.
  A conversation that still holds it keeps reading its resident copy; one that
  comes back after the sweep misses and seals it again.

## 7. Pre-emption

### 7.1 The warmer

A zend task seals, at `Low` band, the shared run a new conversation of each
**warm scope** would carry:

- every owner whose conversations carried a journal in the last day — a new
  thread's lineage and the journal opening (§7.4);
- the journal opening alone, for conversations owned by no thread.

It seals through a transient conversation of its own (`mark_timeline_transient`)
with `Sharing::Shared` sections — the same `submit_section` a conversation runs
— waits until the run's chunks are durable, then releases it (§6), so the run
is held by nothing but the substrate.

### 7.2 When

A run is resealed when what it is made of moves:

- the hour turns over — the next hour's run is sealed in the hour's last five
  minutes, so the first conversation after it hits;
- a scope's lineage changes (a journal note on a project);
- the member primes again (the opening's first entry names what it read);
- the warmer starts (the substrate may hold runs from before, which are hit).

The warmer looks every minute and seals a run only when its text moved: it
keeps each scope's and hour's run by a digest of its sections' names, texts and
sharing.

### 7.3 One producer

Only the member holding the `prewarm/<scope>` lease seals a scope's run, and
only the one holding `prewarm/sweep` sweeps. The others receive both by
replication. A member that cannot propose seals itself, as it reads a unit: a
run sealed twice keeps one stream.

### 7.4 Scopes

A scope is what a new conversation's shared run depends on: an **owner** — a
new thread of theirs carries the owner's project journals and no thread journal
or memory yet — or **threadless**, the opening alone. The warmer builds a
scope's run with the very `journal_sections` a conversation builds, for a
conversation id of a thread no conversation is in, with a fresh journal holding
the opening; the run is its sections up to the first that is the conversation's
own. Owners come from the conversations the substrate holds at start and from
every journal carried since, for a day after the last.

The sweep runs only after a pass that sealed or found every due run, so the
current set it spares is whole. It retires by ending the generation
(`retire_section_generation`), which leaves a resident copy serving.

## 8. Tests

- **Addressing** — raw bytes of `shared_section_prefix`; equal for two owners,
  different for a different chain.
- **Lifetime** — releasing a shared section leaves its stream untombstoned; an
  owned one's is tombstoned; the sweep retires only unheld, unnamed, old shared
  streams.
- **End state** — the payload round-trips byte for byte; the writer lands it on
  the snapshot tail, now and on reload, reporting no slot's state; it goes with
  its section's generation and comes back with a later one.
- **Equivalence** (`zend/tests/shared_journal_sections.rs`, the hybrid model) —
  one conversation seals a shared run and a section of its own after it; the
  state kept on the run's end equals, raw bytes, the state its seal handed over;
  a second conversation restores the run (two restores, one prefill) onto the
  same streams, so its own section starts from exactly that state over exactly
  that K/V.
- **zend** — the chain rule (`journal::sharing`), the opening told apart from a
  drafted entry (`journal::opening`), and the warmer's scopes, hours, memory and
  sweep choice (`session::prewarm`).
