# Worlds, layers and the shared corpus

How `npcd` serves one written corpus to many characters: canon ingested once
into shared layers, craft ingested once untagged, and what a character reads of
the canon scoped per character at each turn.

Companion to `docs/archived/npc_api_gui_design.md` (§20 worlds, §38 world editor) and
`theory_of_the_mind.md`. Where this document and the code disagree, the code
wins and this document is wrong — every mechanical claim below was checked
against the implementation, and the checks are named so they can be re-run.

> **Replaced design.** The first design made a world a *filter by tag over one
> corpus*: canon under `layers/world/<world>/` ingested with the world's slug as
> its tag, a hard error for an underivable tag, a `tag -> Vec<TurnKey>` index
> built at ingest, and a `WorldScoped` resolver wrapper selecting by that tag.
> The code chose otherwise (§4, §6, §7): a layer's documents are tagged with the
> **layer's name**, and what a character reads is scoped per character, per
> bearing, by `bearings.yaml` and `set_retrieval_scope`. The sections below
> describe what is built and keep the original as a note where it differed.

---

## 1. The shape

One substrate. One corpus per layer. A **layer's** documents are shared ground
for every character whose bearing draws on them.

```
mind repository (read from the substrate, `docs/substrate_git.md` §8)
  projection.yaml        the schema
  bearings.yaml          which shared layers a character draws on, by where it stands (§4)
  worlds/                world documents — one YAML per world
  layers/                LAYER content. One folder per layer, named for the layer.
    world/…              ingested with tag "world"
    memory/<character>/… per-character: tags "memory" and "memory:<character>"
  responses/             COLLECTION content. Untagged: shared by every world.
  moods/
```

Two kinds of content, and the distinction runs through everything below:

| | **Canon** | **Craft** |
|---|---|---|
| what it is | facts — cities, factions, history | how a character speaks and feels |
| example | `layers/world/cities.md` | `responses/acknowledge_then_move_forward.yaml` |
| projected as | layer turns | system-prompt collection sections |
| tagged | with its layer's name (`LayerSource::tags`) | **not at all** |
| visible to | characters whose bearing draws on that layer's documents | every world |
| KV | one copy per document turn | **one copy, shared** (§5) |

A world document's `excludes` list (`collections::excludes_of`) names the
section categories it does not admit; absent, a world admits every category.
Craft is shared unless a world says otherwise.

---

## 2. Why tags

Tags are what the substrate's gather scopes on:

- **One ingest, one substrate.** Craft is read once, not once per world.
- **Sharing is the default, not an exception.** See §3 — untagged content is
  automatically visible everywhere.

A layer's folder name is the layer's name in the schema, so the tag is derived
from where a document stands and nothing needs to be kept in step.

---

## 3. The tag rule, and the doc that had it backwards

`projection::resolver` gathers with:

```rust
let in_scope = if tags.is_empty() {
    d.tags.is_empty()                       // empty filter -> UNTAGGED turns only
} else {
    d.tags.iter().any(|t| tags.contains(t)) // named filter -> those tags only
};
```

So:

- **empty filter** admits only turns that are themselves untagged
- **named filter** admits only turns carrying one of those tags, and excludes
  untagged ones

That is the whole mechanism. Untagged content is the shared corpus; tagged
content is scoped. Craft is ingested untagged, canon is ingested with its
world's tag, and no exclusion list is needed anywhere.

`SelectionPolicy`'s doc comment used to say the opposite — *"an empty `tags`
list means all projections in scope"*. It was wrong, and not harmlessly: under
that reading every tagged turn would be visible from every unfiltered node,
silently, because nothing about the output would look wrong. The comment is
fixed and the behaviour is pinned by
`substrate::tests_listing::an_empty_tag_filter_admits_only_untagged_turns`.

---

## 4. Where the filter attaches — collections vs layers

These two paths do **not** behave the same, and the difference decides the
design.

**Collections already filter.** `belief_gallery` applies `policy.tags` per
collection. So the craft side works today with no change: declare `tags: []`
and a collection sees the shared, untagged library.

**Layers do not filter by the schema's tags, and are scoped two other ways.**
`score_belief_groups` has no reference to the schema's tags, and a layer group's
candidates come from `ContentResolver::group_turns(group)`, which masks by
**timeline**, never by tag.

1. **Tag scope per conversation.** A turn group can be scoped to turns carrying
   one of a set of tags (`Builder::set_group_tags`), applied in both the belief
   scan and projection via `ContentResolver::turn_carries`. It is set on a
   conversation's own copy of the builder rather than in the YAML, because the
   scope is whose data a reader is, and an empty list still means the whole
   group — unlike a collection's empty filter. The dream layer uses it
   (`npcd/src/engine/dreams.rs`, `docs/reflection_and_dreams.md` §4).
2. **Retrieval scope per character, per bearing.** The shared canon is scoped by
   the set of documents (timelines) a character's bearing draws on — §10 and
   `npcd/src/engine/bearings.rs`. Each shared group is marked scoped
   (`mark_group_scoped`), and before each turn `Scopes::apply` calls
   `ConversationEngine::set_retrieval_scope(timeline, group, allowed)`; the
   belief scan and the selection read only those documents.

> **Replaced design.** The canon gap was first to be closed with a
> `WorldScoped<'_, S>` resolver wrapper filtering `group_turns` by the world's
> tag, chosen over an engine change to `project_with_mode(...)` because that
> signature is also called by `zend`. The built scope is the engine's own
> `set_retrieval_scope`, keyed by document and fed by `bearings.yaml`, so what a
> character may recall depends on where it stands, which a per-world tag could
> not express.

---

## 5. Shared KV — real, but conditional

Two stream-id derivations, and only one is content-addressed:

```rust
section_stream_id(addr: ContentAddress)     // hash(prefix_hash, section_hash)
turn_stream_id(timeline_id, turn_index)     // "identity-addressed"
```

**Craft gets shared KV automatically.** Sections are content-addressed, so the
same response template is the same stream is the same KV, for every world that
projects it. This is the "shared KV in the substrate" property, and it costs
nothing to obtain — it is a consequence of ingesting craft once, untagged.

**Canon does not, and should not.** Turns are identity-addressed by
`(timeline, index)`. Two identical canon turns in different timelines are
different streams. Since canon is world-exclusive there is nothing to share.

### The condition, which is easy to lose

`ContentAddress` is `{ prefix_hash, section_hash }`. A section's KV is shared
only when the **section content and everything before it in the prompt** both
match.

> **Ordering rule.** World-specific content must come **after** the shared
> collections in the system prompt. A world's `setting:` injected before them
> forks the prefix, and every world then gets its own copy of every response
> template's KV — the exact opposite of the intent, with no error and no
> visible symptom beyond memory.

The section order is declared in the mind repository's `projection.yaml`, which
is not in this tree; its current order cannot be confirmed here. The rule is
what binds it: world material comes after the shared collections.

---

## 6. Layer becomes tag at ingest

`npcd/src/engine/ingest.rs` walks `layers/<layer>/` for every layer that
declares an `ingest_unit:` and has a folder. Every document becomes one
prefilled turn (address as the user half, content as the assistant half) tagged
by `LayerSource::tags`:

- a shared layer: `[<layer>]` — `layers/world/cities.md` carries `["world"]`;
- a per-character layer (every subdirectory names a personality, e.g.
  `layers/memory/zen/`): `[<layer>, "<layer>:<character>"]`, so a character's own
  memory reaches it and nobody else's does.

The document's address inside its layer (`world/factions/hess`) is what
`bearings.yaml` reads a topic from (`bearings::topic`: the folder beneath the
layer, or the document's own name at the layer's root). A layer's folder *is* the
layer's name, so a tag cannot fail to derive and the hard-error case does not
arise. `beliefs`, `relationships` and `agency` are not ingested from disk
(`LIFE_DERIVED`): a life story produces them.

> **Replaced design.** Canon was to live under `layers/world/<world-slug>/` with
> the slug derived from the path as the tag (as `repo_scan::dir_tags` does for a
> repository's directories), and an underivable tag was to be a hard error so a
> document could never fall through to untagged and become readable from every
> world. With no per-world tag there is no such failure; per-world separation of
> canon is a matter of what a character's bearing draws on.

---

## 7. Selecting a scope is a built set, not `turn_with_tag`

`Substrate::turn_with_tag` resolved **one** member — a group's declared default
— and was unsuitable for bulk selection on two counts (the function and the test
that pinned them are no longer in the tree):

1. It was a linear scan over every stream decl: a scan per turn, quadratic in
   the corpus.
2. **With more than one match the winner was arbitrary** — `all_streams().find_map(..)`
   over a map, so iteration was neither insertion nor index order.

The built selection reads each shared group's documents once (`Shelf::read`: each
document's conversation, its address, the year it is dated by) and builds, per
bearing and per working year, the `HashSet<TimelineId>` of documents the bearing
admits (`Scopes::at`, cached in a map keyed by `(bearing, year)`). There is no
`tag -> Vec<TurnKey>` index.

> **Replaced design.** A `tag -> Vec<TurnKey>` index built at ingest and
> updated on re-ingest, to serve `WorldScoped`. The scope set is built from the
> shelf instead, because the scope is by address topic and year, not by tag.

---

## 8. Worlds are files, and there is no "New World"

A world is `worlds/<slug>.yaml` in the mind repository: authored content, on
the same side of the line as the corpus. It is a record the API serves
(`npcd/src/api/worlds.rs`: listing, `GET`/`PUT`/`DELETE` of a world, its clock,
`excludes`-filtered collections). It carries no tag selection: canon is scoped
per character by bearing (§4), not by a tag the world names.

> **What a human wrote lives in the mind repository; what the engine wrote
> lives in the data directory.** The mind (`projection.yaml`, `bearings.yaml`,
> `layers/`, the collections, `worlds/`) is read from the substrate with
> `--mind-repo` (`docs/substrate_git.md` §8); the substrate and the ingest ledger
> (`substrate/ingest-ledger.json`) are the engine's. The ledger sits inside the
> substrate on purpose, so that wiping the substrate wipes the claim that its
> documents are in it.

**There is no "New World" button.** An empty world is non-functional — no canon
means the `world` layer projects nothing — so a button that creates a container
hands back something broken and calls it success. A world is a YAML file in
the mind: making one is a file operation and a commit, which is what authored
content should be. Characters are what users create; worlds are written.

---

## 9. The GUI never says "filter"

Filtering is what the engine does. What an author does is write a world. This
section is the intended console behaviour; the console is not in this
repository, so the first two points are not verifiable here. The API beneath
them is: `npcd/src/drafts/` holds edits to mind files (worlds included) and
publishes them as a commit.

- **World selection** is a chip in the same shape as the roster's existing
  `tag` / `state` / `world_id` controls.
- **The world editor** is a file editor over the mind's files — browse, edit,
  create.
- **A changed document is re-ingested as a new turn.** The ledger
  (`engine/watcher.rs`) hashes each document: unchanged writes nothing; changed
  tombstones the old turn and ingests the new content as a fresh one; removed
  is retired at the next start.

The tag is never surfaced. It is storage, not a concept the author meets.

---

## 10. Risks

**Retrieval funnel.** The reference schema in this tree (`npcd/projection.yaml`)
gives the `world` layer a single `canon` group of `top_k` 6, an 8,000-token
window and `score_threshold` 0.30. A small `k` over a large corpus makes that
layer's threshold load-bearing in a way no other layer's is, and bad retrieval
presents as *"the character doesn't know its own world"* — which reads as a
model fault rather than a scoring one. Measure early. The corpus size and the
mind's own layer settings are in the mind repository and are not stated here.

**Ingest cost.** A document is a real prefill (the ledger's own comment puts it
at about four seconds on a 3090 serially), so ingest runs a window of 16
one-conversation-per-document sequences co-batched into one prefill forward
(`ingest::WINDOW`) and a persisted content-hash ledger makes a restart write
only what changed. The loading screen carries the progress (`ingest_unit:`).
Every layer ingests at boot, in dependency order (`ingest::ORDER`: `world`
first, `memory` last); there is no lazy per-world ingest.

**Re-ingest.** A changed document is reconciled by the ledger at start:
the old turn is tombstoned and the new content ingested as a fresh turn
(`engine/watcher.rs`). I found no path that re-ingests a document into a
running engine on save; that remains unbuilt.

**Where a document lands, and who can read it.** Two rules decide whether a
shared document ever reaches a character, and both were broken at once, so the
`world`, `eras` and `stories` layers reached nobody:

- *A shared document is written into its own layer's group*
  (`schema::document_target`): the layer's one group, when it selects by score.
  A timeline's group is fixed when it is created, and a projection reads its
  own target group only along its own timeline, so the ~1,300 documents first
  written into the live conversation group were scored against no character's
  turn. At boot each such document still standing in the live group is retired
  (found by its layer tag) and written again into its layer's group, which is
  marked append-only (`loader.rs`, `mark_layer_append_only`) — a document's own
  projection reads only itself, as zend writes a file onto its ingest layer.
  Every document is one exchange, so a group's documents compete as one pool and
  their hit levels are learned from the characters' turns
  (`docs/provenance_score_normalization.md` §3.3a).
- *A projection sees only the layers ranked beneath its target.* A layer's rank
  is its declaration index unless it declares `rank:`, so a shared layer declared
  after the character's own layers is never in its gather unless it declares a
  negative rank to sit beneath them. The reference schema in this tree does this
  for `world` (`rank: -1`). The ranks of the mind's other shared layers (`eras`,
  `stories`, `dreams`) are configured in the mind repository's `projection.yaml`,
  which is not in this tree, so their values are not stated here.

**What a character draws on follows where it stands and when it works**
(`engine::bearings`, the mind's `bearings.yaml`). Before each turn every shared
group is scoped (`set_retrieval_scope`) to the documents the character's bearing
draws on: per layer, `all` or a list of topics (a topic is the folder beneath
the layer, or a document's own name at its root). The first rule whose parts
stand in the character's room is its bearing; a room with none is `elsewhere`; a
personality with draws of its own has them wherever it is; a mind with no
`bearings.yaml` draws on everything everywhere; a layer a bearing does not name
offers nothing there. Which parts and which layers each bearing names is in the
mind repository's `bearings.yaml`, so no particular room-to-layer mapping is
stated here.

A character working in a year of the past — set at a time machine
(`Mission::year`, which ends with the mission) — is held to it by the `timed:`
table of `bearings.yaml`: a layer marked `era` offers the era the year falls in
and the one before (the years the layer's own documents are dated by), and a
layer marked `dated` offers the documents that say they are set on or before the
year, an undated one held back since nothing shows it is not later. Which
layers are marked which is the mind's. Relevance gates (`min_score`,
`evict_score` on a group) are likewise schema values in the mind repository.

Every character turn logs what it recalled from beneath, layer by layer, with
its bearing, its year and each layer's best (`recalled from beneath at
story-desk in 2950 — eras 1 (best 2349 eras/the-tower-age), …`).

---

## 11. What was verified, and how

| Claim | Check |
|---|---|
| empty filter = untagged only | `substrate::tests_listing::an_empty_tag_filter_admits_only_untagged_turns` |
| collections filter by tag | `projection/resolver/belief_gallery.rs` `belief_gallery`, `belief_collections.rs` `score_belief_collections` |
| layer groups filter by tag only where `Builder::set_group_tags` set them | `projection/resolver/belief_groups.rs` `score_belief_groups`; candidates from `group_turns` |
| shared canon is scoped per bearing by document set | `npcd/src/engine/bearings.rs` `Scopes::apply` → `set_retrieval_scope` |
| documents are tagged by layer name (and `layer:character`) | `npcd/src/engine/ingest.rs` `LayerSource::tags` |
| `group_turns` masks by timeline only | `projection/resolver/targeted_read.rs` impl |
| sections are content-addressed | `persistence/content_hash.rs` `section_stream_id(ContentAddress)` |
| turns are identity-addressed | `persistence/content_hash.rs` `turn_stream_id(timeline, index)` |
| ingest carries per-turn tags | `TurnSink::insert_prefill_turn(.., tags)` (`zend/src/raw_read/mod.rs`) |
| a path already becomes a tag | `dir_tags` (`zend/src/repo_scan/ingest_plan.rs`) |
| layer content folder = layer name | `npcd/src/engine/ingest.rs` `sources`: `layers/<name>` |

---

## 12. Status

| Piece | Status |
|---|---|
| `layers/<layer>/` ingest, tagged by layer name (and character) | Built (`engine/ingest.rs`) |
| World documents `worlds/*.yaml` and their API | Built (`api/worlds.rs`) |
| Per-character canon scope by bearing | Built (`engine/bearings.rs`) |
| Per-world path→tag derivation, hard error on an underivable tag | Replaced design (§6): not built, and not needed |
| `tag -> Vec<TurnKey>` index and `WorldScoped` wrapper | Replaced design (§4, §7): not built |
| World chip and file-tree editor in the console | Not verifiable here: the console is not in this repository |
| Re-ingest of a document into a running engine on save | Not built: the ledger reconciles at start |
