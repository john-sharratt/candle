# Worlds, layers and the shared corpus

How `npcd` turns one written corpus into many worlds, and why it does it with
tags rather than directories.

Companion to `npc_api_gui_design.md` (§20 worlds, §38 world editor) and
`theory_of_the_mind.md`. Where this document and the code disagree, the code
wins and this document is wrong — every mechanical claim below was checked
against the implementation, and the checks are named so they can be re-run.

---

## 1. The shape

One substrate. One corpus. A world is a **filter over it**, not a copy of it.

```
mind/
  projection.yaml        the schema
  worlds/                world metadata — one YAML per world
    battle-cities.yaml
    ardh.yaml
  layers/                LAYER content. Tagged by world.
    world/
      battle-cities/…    ingested with tag "battle-cities"
      ardh/…             ingested with tag "ardh"
  responses/             COLLECTION content. Untagged: shared by every world.
  moods/
  identities/
```

Two kinds of content, and the distinction runs through everything below:

| | **Canon** | **Craft** |
|---|---|---|
| what it is | a world's facts — cities, factions, history | how a character speaks and feels |
| example | `layers/world/battle-cities/cities/abora.md` | `responses/acknowledge_then_move_forward.yaml` |
| projected as | layer turns | system-prompt collection sections |
| tagged | with its world | **not at all** |
| visible to | that world only | every world |
| KV | one copy per world | **one copy, shared** (§5) |

`ardh.yaml`'s existing `templates: { responses: override, moods: default }` is
this table already: craft is shared unless a world says otherwise.

---

## 2. Why tags and not directories

An earlier draft gave each world its own directory tree and ingested them
separately. Tags are better on three counts:

- **One ingest, one substrate.** Craft is read once, not once per world.
- **Sharing is the default, not an exception.** See §3 — untagged content is
  automatically visible everywhere.
- **A document can belong to two worlds.** Available if ever wanted; never
  required.

And it removes the mapping table entirely. A world does not declare which
directories it draws from; the ingest path *derives* the tag, and the world
names the tag. There is nothing to keep in step.

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
`substrate::tests::an_empty_tag_filter_admits_only_untagged_turns`.

---

## 4. Where the filter attaches — collections vs layers

These two paths do **not** behave the same, and the difference decides the
design.

**Collections already filter.** `belief_gallery` applies `policy.tags` per
collection. So the craft side works today with no change: declare `tags: []`
and a collection sees the shared, untagged library.

**Layers do not.** `score_belief_groups` contains no reference to tags at all.
A layer group's candidates come from `ContentResolver::group_turns(group)`
(`project.rs:1144`, whose comment notes *"group_turns has already masked the
candidates"*), and the implementation masks by **timeline**, never by tag.

That is the gap, and it is on the side that matters most: the `world` layer is
the canon. Two ways to close it:

1. **Engine change** — add a runtime tag scope to `project_with_mode(...)`.
   Correct, invasive, and it changes a signature `zend` also calls.
2. **Resolver wrapper** — `ContentResolver` is caller-supplied. A
   `WorldScoped<'_, S>` wraps the substrate and filters `group_turns` by the
   world's tag.

**Take the wrapper.** No engine change, no `zend` risk, and it lands exactly
where "what is visible" is already decided. `policy.tags` stays schema-static
for craft, which is correct — craft scoping *is* an authoring decision. World
scoping is data, and data belongs in the resolver.

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
> forks the prefix, and every world then gets its own copy of all 596 response
> templates' KV — the exact opposite of the intent, with no error and no
> visible symptom beyond memory.

`mind/projection.yaml` currently orders: framing sections, then
`identity_anchor` / `identity` / `response` / `mood`, then `history_stance`,
`grounding`, `tools`. World material belongs at or after `history_stance`,
never above `identity_anchor`.

---

## 6. Path becomes tag at ingest

The mechanism exists. `TurnSink::insert_prefill_turn(user, assistant, tags)`
already carries per-turn tags, *"persisted on the TurnDecl so tag-scoped
provenance galleries admit the turn"*, and `repo_scan` already derives a tag
from a path:

```rust
fn dir_tags(unit: &DirUnit) -> Vec<String> {
    vec!["repo_map".to_string(), unit.dir.clone()]   // pass name + directory
}
```

npcd's ingest does the same with the world slug: walking
`layers/world/battle-cities/cities/abora.md` yields `["battle-cities"]`, and
optionally the deeper segments (`cities`) for sub-world filters later.

**Untagged is shared, so a missing tag leaks.** A canon file under `layers/`
whose tag cannot be derived must be a **hard error**, not a default to
untagged — the failure would otherwise be a document quietly readable from
every world.

---

## 7. Selecting a scope needs an index, not `turn_with_tag`

`Substrate::turn_with_tag` resolves **one** member — a group's declared default
— and is unsuitable for bulk selection on two counts, both pinned by
`substrate::tests::turn_with_tag_finds_one_arbitrary_member_by_scanning`:

1. It is a linear scan over every stream decl. Selecting a scope's whole corpus
   through it is a scan per turn: quadratic in the corpus.
2. **With more than one match the winner is arbitrary.** It is
   `all_streams().find_map(..)` over a map, so iteration is neither insertion
   nor index order. The test originally asserted `TurnIndex(0)` and got
   `TurnIndex(1)`. Its doc's *"expected to identify a unique turn"* is a
   precondition, not a hint.

`WorldScoped` therefore needs a `tag -> Vec<TurnKey>` index built **once at
ingest** and updated on re-ingest.

---

## 8. Worlds are files, and there is no "New World"

A world is `mind/worlds/<slug>.yaml`: name, setting, public, and the tag(s) it
selects. It lives in the mind because it is authored content — the same side of
the line as the corpus it indexes.

> **`--data` is what the engine wrote. `--mind` is what a human wrote.**

`.substrate/` and `accounts/` are `--data`. `projection.yaml`, `layers/`,
`responses/`, `moods/`, `personalities/`, `worlds/` are `--mind`. `npcd` resolves
all of them beside the schema when `--mind` names one, and falls back to `--data`
only for a daemon run without a mind at all.

**There is no "New World" button.** An empty world is non-functional — no canon
means the `world` layer projects nothing — so a button that creates a container
hands back something broken and calls it success. A world is a YAML file and a
tagged directory: making one is a file operation and a commit, which is what
authored content should be. Characters are what users create; worlds are
written.

---

## 9. The GUI never says "filter"

Filtering is what the engine does. What an author does is write a world.

- **World selection** is a chip in the same shape as the roster's existing
  `tag` / `state` / `world_id` controls.
- **The world editor** is a file tree over that world's tagged content —
  browse, edit, create. Editing a world edits the real files, because the
  editor *is* a file editor.
- **Saving re-ingests** that one document: new tokens, new content hash, new
  stream, the old one superseded. Same last-writer-wins path NPC records use.

The tag is never surfaced. It is storage, not a concept the author meets.

---

## 10. Risks

**Retrieval funnel.** The `world` layer is `top-k 6` against an 8,000-token
window, over a corpus of ~1,267 documents. Six of 1,267 makes that layer's
`score_threshold` load-bearing in a way no other layer's is, and bad retrieval
presents as *"the character doesn't know its own world"* — which reads as a
model fault rather than a scoring one. Measure early.

**Ingest cost.** A per-world corpus is a startup ingest; `zend` has the
loading-screen machinery (`ingest_unit:`, progress counters) precisely because
it is slow. Decide between ingesting every world at boot and lazily on first
character.

**Live re-ingest.** Editing a document while an engine is running is the one
piece with no existing precedent here. Everything else in this design is an
arrangement of parts that already work.

---

## 11. What was verified, and how

| Claim | Check |
|---|---|
| empty filter = untagged only | `substrate::tests::an_empty_tag_filter_admits_only_untagged_turns` |
| `turn_with_tag` scans, and is arbitrary on multiple matches | `substrate::tests::turn_with_tag_finds_one_arbitrary_member_by_scanning` |
| collections filter by tag | `resolver.rs` `belief_gallery`, `score_belief_collections` |
| layer groups do **not** | `score_belief_groups` — no `tags` reference; candidates from `group_turns` |
| `group_turns` masks by timeline only | `resolver.rs` impl |
| sections are content-addressed | `content_hash.rs` `section_stream_id(ContentAddress)` |
| turns are identity-addressed | `content_hash.rs` `turn_stream_id(timeline, index)` |
| ingest carries per-turn tags | `TurnSink::insert_prefill_turn(.., tags)` |
| a path already becomes a tag | `repo_scan::dir_tags` |
| layer content folder = layer name | `ingest.rs` — `default_folder = other` |

---

## 12. Order of work

1. `layers/` container and the path→tag derivation, with a hard error for an
   underivable tag.
2. `mind/worlds/*.yaml` and its loader; move `worlds/` and `personalities/` into
   the mind.
3. The `tag -> Vec<TurnKey>` index at ingest.
4. `WorldScoped` resolver wrapper.
5. `mind/lore/` → `mind/layers/world/battle-cities/`.
6. GUI: world chip, then the file-tree editor.
7. Live re-ingest on save.

Steps 1–4 are engine-side and testable without a GUI; 5 is a `mv` that should
happen while nothing depends on the path.
