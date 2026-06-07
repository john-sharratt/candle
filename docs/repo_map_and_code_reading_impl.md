# Zen Code — Repo Map + Code Reading Implementation

## Goal

Implement Phases 1 and 2 of the trunk-analysis pipeline (per
[`coding_assistant.md`](coding_assistant.md)) inside the `zend` daemon.
Together these two phases populate the foundational layers of the projection
schema — `repo_map` and `code_reading` — with the entire workspace as
**prefilled turns** before the daemon starts serving developer chat.

This is the heart of unbounded context: every later phase (static / dependency
/ architectural / critical / bug analysis, daily history, dreams, and the live
`dialogue` layer) attends back over the repo-map and code-reading turns via
attentional provenance retrieval. Getting these two layers right is the proof
that the substrate, projection engine, and provenance scan compose into a
useful coding assistant at all — every layer after this is variations on the
same theme.

## Non-goals

- **Phases 3–7.** Static, dependency, architectural, critical, and bug
  analysis are decode-heavy. They land after this work as separate documents.
- **API enhancement.** The provider-agnostic background enhancement loop is
  Phase 4 in the master plan; not in scope here.
- **Shadow files / `.zend/` storage.** The shadow-file schema in
  `coding_assistant.md` is the persistence story for analysis outputs that
  must survive across daemon restarts and live alongside source files in
  git. v1 of this work runs the full repo scan + carve on every daemon start
  and persists the resulting turns only via the substrate redo log (the
  in-workspace `.substrate/` directory). Shadow-file extraction is a follow-up.
- **Reconcile waves.** File-watch-driven re-prefill of changed files is
  follow-up work. v1 is a one-shot bulk pass at startup.

---

## Architecture

### Two new prefill-only conversations

The projection schema at [`zend/src/prompts/projection.yaml`](../zend/src/prompts/projection.yaml)
declares the `repo_map` and `code_reading` layers with their own
`system_prompt:` blocks and per-group selection rules. Today only the
`dialogue` layer is populated (`base_conv`). v1 of this work adds two more
project-lifetime conversations on the same engine:

```
ENGINE (workspace substrate, shared by all sequences)
│
├── base_conv      — dialogue layer / primary_conversation group
│                    Live developer chat. Forks per conv_id.
│
├── titler         — Reserved::Titler layer / group
│                    Sidebar title generation. Exists already.
│
├── repo_map_conv  — repo_map layer / structure group           (NEW)
│                    Holds ONE prefilled turn: a directory tree.
│                    Always-visible (selection: always_visible).
│
└── code_read_conv — code_reading layer / scopes group          (NEW)
                     Holds ONE prefilled turn per (file × scope) chunk.
                     TopK selection — projection picks 8 per query.
```

Both new conversations live on their own [`TimelineId`](../candle-conversation/src/projection/ids.rs)
inside the same workspace substrate as `base_conv`. They never serve user
queries directly — they exist solely so their sealed K/V is reachable by the
`dialogue` layer's BDP-scored retrieval at decode time. Because `repo_map`
and `code_reading` sit below `dialogue` in the layer ordering, the dialogue
projection sees them; the foundational layers cannot see each other or any
higher layer (the projection masking rule from `infinite_conversations.md`).

### Why prefill-only

Both phases are **ingestion** — no decode, no reasoning, no model output.
The user turn is the schema prompt (e.g. *"List the complete file structure"*
or *"File: src/foo.rs / Scope: impl Foo > fn bar / Lines: 10-25"*); the
assistant turn is the raw content (the rendered directory tree, or the raw
source code for that scope). We use [`Sequence::insert_turn(user_message,
assistant_text)`](../candle-conversation/src/conversation.rs) — the existing
no-decode path that prefills a complete user-and-assistant exchange in one
forward pass and seals it as a normal turn. No new substrate plumbing is
required; insertion goes through the same projection-then-prefill path that
`submit_turn` uses, so the resulting turn carries real BDP fingerprints
captured under the layer's own system prompt.

Cost: one prefill pass per turn. The whole repo at a few hundred files ×
typical 10–30 scopes per file lands in **prefill-dominant** time — the
4090 mobile baseline ships ~2,500 t/s of parallel prefill, so a 200-file
codebase at ~50K LoC is roughly a 1–3 minute pass at daemon startup. That
cost is paid once per daemon start (until shadow-file caching lands), and
the resulting K/V is available to every developer query for the lifetime
of the daemon.

### Where it slots in

The daemon already declares the right load steps —
[`LoadStep::RepoScan`](../zend/src/loading.rs) and `LoadStep::CodeRead` are
placeholders in the state machine, walked through instantly today
([`zend/src/session.rs`](../zend/src/session.rs) lines 825–828). This work
fills them in.

```
LoadStep::Model         — model weights
LoadStep::Substrate     — redo-log replay
LoadStep::Sections      — pinned system-prompt sections
LoadStep::RepoScan      ← Phase 1 lands here   (single prefill turn)
LoadStep::CodeRead      ← Phase 2 lands here   (N prefill turns)
                          → mark_ready, chat unlocks
```

Per-step progress is already wired — model and section prefill already
report `(done, total)` into [`LoadProgress::set_step_progress`](../zend/src/loading.rs);
the new steps do the same so the loading overlay reads usefully.

---

## Phase 1 — Repo Map

### Scope

Single conversation, single turn, no decode. The assistant prefill is a
deterministic rendered directory tree of every file in scope.

### File enumeration

Workspace traversal uses the [`ignore`](https://docs.rs/ignore) crate —
the file-walking engine that backs ripgrep. It respects:

- `.gitignore` (every level of the tree)
- `.git/info/exclude`
- The global git ignore (if configured)
- `.ignore` files (ripgrep convention; useful for project-local rules)
- Hidden files (skipped by default)
- Symlinks (followed when explicitly enabled; default off)

`ignore` is added as a `zend` workspace dep alongside the existing axum /
reqwest stack. We do **not** thread it through `candle-conversation` —
file-system concerns belong in the daemon.

```toml
# zend/Cargo.toml
ignore = "0.4"
```

### Default watch patterns

The MVP scans every file `ignore` surfaces, then filters by extension
against a built-in allowlist that mirrors the
[`coding_assistant.md`](coding_assistant.md) example config. The list is
hard-coded in v1; configuration via `.zend/config.yaml` is a follow-up:

```
Rust:           .rs
Python:         .py, .pyi
TypeScript/JS:  .ts, .tsx, .js, .jsx, .mjs, .cjs
Go:             .go
Markdown:       .md
YAML / TOML:    .yaml, .yml, .toml
JSON:           .json   (config files only — large generated JSON skipped via size cap)
Plain text:     .txt
```

A size ceiling — **256 KB per file** — rejects oversize artefacts (generated
JSON, vendored bundles, minified output). Above the cap, files are skipped
silently and counted in a `n_skipped_oversize` log line at the end of the
walk.

`.zend/` itself is excluded so the daemon's own outputs never appear in
the repo map.

### Tree rendering

Files are grouped by parent directory, sorted lexicographically within
each directory, and rendered with line-art prefixes. The output is
deterministic — same workspace, same tree — so the fingerprints stay
stable across reruns.

```
src/ (42 files, 8 directories)
├── auth/
│   ├── handler.rs          (247 lines, Rust)
│   ├── validator.rs        (183 lines, Rust)
│   └── mod.rs              (12 lines, Rust)
├── db/
│   ├── schema.rs           (156 lines, Rust)
│   └── queries.rs          (312 lines, Rust)
└── ...

Cargo.toml (workspace: 3 members)
docs/ (6 files)
└── ...
```

The renderer is a small standalone module
([`zend/src/repo_scan/render.rs`](../zend/src/repo_scan/render.rs)) so it's
unit-testable independent of any file-system or model setup.

### Per-file metadata

Each leaf carries three pieces:

- **Line count** — counted on disk via a one-pass byte scan
  (newline count + 1). No tokenisation. Cheap.
- **Language** — derived from the extension via a small lookup table
  alongside the watch-pattern list above.
- **Workspace manifests** — `Cargo.toml`, `package.json`, `pyproject.toml`,
  `go.mod` are tagged with their workspace/module shape (member count
  for Cargo, name for the others). This is the "module structure" hint
  from the design doc — useful for the repo map to convey project
  topology beyond raw tree shape.

### User turn (the schema prompt)

The text is hard-coded to match the design doc, sent verbatim as the
user message:

```
List the complete file structure of this repository.
For each file, show: path, size in lines, and file type.
Organise by directory. Note any module structure
(Cargo.toml workspaces, Python packages, Go modules, etc).
```

### Assistant turn (the prefilled tree)

The deterministic rendered tree from the previous step.
`Sequence::insert_turn(user, assistant)` prefills both sides through the
projection-aware path: the user turn enters the substrate, the assistant
turn enters with its raw content as a no-decode prefill, both seal as a
single turn pair under the `repo_map` layer's `system_prompt` framing
(*"You are absorbing the structural shape of a software repository…"*).
BDP fingerprints are captured during prefill and become retrievable.

### Module layout

```
zend/src/repo_scan/
├── mod.rs           — public API: scan_workspace(root) -> RepoMap
├── walk.rs          — ignore-driven enumeration + filtering + metadata
├── render.rs        — tree formatter (deterministic output)
└── types.rs         — FileEntry, DirEntry, RepoMap structs
```

The crate-public entry point is one function:

```rust
pub fn ingest_repo_map(
    engine: &ConversationEngine,
    proj_builder: &projection::Builder,
    workspace: &Path,
    progress: &LoadProgress,
) -> anyhow::Result<Sequence>;
```

Returns the `repo_map_conv` Sequence so `InferenceState` can hold it
alongside `base_conv` and `titler`. Progress is reported as
`(files_walked, total_estimated)` — the estimate is the file count from
the initial walk before filtering, refined to exact once the walk
completes.

---

## Phase 2 — Code Reading

### Scope

For every file in the repo map, parse into scope-aware chunks (one chunk
per function / impl / struct / etc.), then insert one turn pair per
chunk under the `code_reading` layer.

### Scope-aware carving

`coding_assistant.md` mandates tree-sitter for production-grade scope
detection. For v1 we ship a layered approach so we can land Phase 2
end-to-end without a tree-sitter dependency landing first:

| Tier | Mechanism | Languages covered v1 |
|------|-----------|----------------------|
| 1 | Tree-sitter (preferred) | Rust, Python, TypeScript, JavaScript, Go |
| 2 | Regex / brace-aware fallback | Anything tree-sitter doesn't load |
| 3 | Header-based fallback | Markdown (`##` headings), YAML/TOML (top-level keys) |
| 4 | Fixed-window fallback | Anything else (100-line windows) |

Tier 1 is the v1 target. The fallback tiers exist as defence in depth so
Phase 2 doesn't silently skip a whole language family if its tree-sitter
grammar fails to load.

**Tree-sitter integration.** We pull `tree-sitter` plus the per-language
grammar crates (`tree-sitter-rust`, `tree-sitter-python`,
`tree-sitter-typescript`, `tree-sitter-javascript`, `tree-sitter-go`) as
direct deps. A small `scope_query.scm` per language extracts the node
kinds we care about — functions, methods, structs, classes, traits,
impls, top-level imports, top-level docs. The query files live alongside
the parser glue in `zend/src/code_read/parsers/`. Each language module
is ~50 lines: load grammar, run query, map captures to `Scope` structs.

**Carving rules** are exactly the ones from `coding_assistant.md` §
"Carving rules" — function/method body, struct/class/enum/trait, impl
header, module-level constant group, import group, module docs, top-level
expression. Size limits: **150-line maximum** per chunk (split at
sub-block boundaries when exceeded), **3-line minimum** (tiny items
group with adjacent same-kind neighbours).

### Scope header format

The header is the schema prompt for each per-scope turn — `>` separates
nesting levels. Format matches the design doc verbatim so the layer's
`system_prompt` (which trains the model to expect these headers) stays
calibrated:

```
File: src/lib.rs > mod cache > impl KvCache > fn seal_chunk
Lines: 142-187
```

Determinism matters: same source produces the same header. Re-running
the carve on an unchanged file emits byte-identical user-turn text, so
fingerprints reproduce.

### User / assistant pair per scope

For each scope:

```rust
let user = format!(
    "File: {path} > {scope_path}\nLines: {start}-{end}",
    path = file.path_relative_to_root(),
    scope_path = scope.qualified_path(),
    start = scope.start_line,
    end = scope.end_line,
);
let assistant = source_bytes_for(scope.start_line..=scope.end_line);
code_read_conv.insert_turn(&user, &assistant)?;
```

That's the entire ingestion loop. The layer's `system_prompt` carries
the framing once; every per-scope turn rides the same framing on the
sealed K/V side.

### Module layout

```
zend/src/code_read/
├── mod.rs                     — public API: ingest_code_reading(...)
├── carve.rs                   — top-level dispatch: pick parser, run, collect scopes
├── parsers/
│   ├── mod.rs
│   ├── rust.rs                — tree-sitter glue + scope_query.scm
│   ├── python.rs              — same shape
│   ├── typescript.rs
│   ├── javascript.rs
│   ├── go.rs
│   ├── markdown.rs            — header-based (tier 3)
│   ├── structured_config.rs   — YAML/TOML/JSON (tier 3)
│   └── fallback.rs            — fixed-window (tier 4)
├── header.rs                  — deterministic scope-header rendering
└── types.rs                   — Scope, ChunkKind, CarvedFile
```

`ingest_code_reading` walks the repo map's file list, carves each file,
and pumps each scope into the `code_read_conv` Sequence as an
`insert_turn` pair. Progress is reported as
`(scopes_done, scopes_total)` — total is computed after the carve pass
completes and before the prefill loop begins, so the progress bar is
accurate from the first turn.

---

## Wiring into the daemon

### `LoadStep` transitions

[`InferenceState::load`](../zend/src/session.rs) ends today with the
section-prefill pass for `base_conv` and the construction of `titler`.
After both succeed, we extend the constructor to mint the two new
conversations on the same `engine` and run the ingestion passes before
returning.

```rust
// session.rs — InferenceState::load (extended)

let base_conv = engine.new_conversation_with_projection_progress(...)?;
let titler   = engine.new_reserved_conversation(...)?;

// Phase 1 — repo map
progress.set_step(LoadStep::RepoScan);
let repo_map_conv = crate::repo_scan::ingest_repo_map(
    &engine, &proj_builder, &workspace, &progress,
)?;

// Phase 2 — code reading
progress.set_step(LoadStep::CodeRead);
let code_read_conv = crate::code_read::ingest_code_reading(
    &engine, &proj_builder, &workspace, &repo_map, &progress,
)?;

Ok(Arc::new(Self {
    decoder,
    engine: Mutex::new(engine),
    conversations: Mutex::new(HashMap::new()),
    base_conv: Mutex::new(base_conv),
    repo_map_conv: Mutex::new(repo_map_conv),   // NEW
    code_read_conv: Mutex::new(code_read_conv), // NEW
    titler: Mutex::new(titler),
    titler_timeline,
    tokenizer,
    tool_host: ToolHost::new(),
}))
```

The two new fields live on `InferenceState` so the substrate keeps them
alive for the daemon's lifetime — they own slots, they own KV residency,
they own redo-log streams. Dropping them mid-session would tear down
exactly the institutional knowledge we just spent minutes prefilling.

### Layer / group resolution

The two new conversations resolve their layer + group from the projection
builder by name, mirroring how `dialogue_layer` / `primary_group` are
resolved today:

```rust
let repo_map_layer = proj_builder.id_for_layer("repo_map")
    .ok_or_else(|| anyhow!("projection schema missing 'repo_map' layer"))?;
let structure_group = proj_builder.id_for_group("structure")
    .ok_or_else(|| anyhow!("projection schema missing 'structure' group"))?;

let code_reading_layer = proj_builder.id_for_layer("code_reading")
    .ok_or_else(|| anyhow!("projection schema missing 'code_reading' layer"))?;
let scopes_group = proj_builder.id_for_group("scopes")
    .ok_or_else(|| anyhow!("projection schema missing 'scopes' group"))?;
```

The system-prompt for each layer comes from the layer's own
`system_prompt:` block in the YAML — already authored
([`projection.yaml`](../zend/src/prompts/projection.yaml) lines 95–103 for
`repo_map`, lines 126–136 for `code_reading`). The conversation is
constructed with the formatted version of its layer's prompt; the
projection engine handles section emission at apply time per layer.

### Conversation construction helper

`new_conversation_with_projection_progress` already does the right thing
for `base_conv`; the two new conversations follow the same shape. Since
the projection builder is consumed by the existing `base_conv`
construction (the schema is pinned per-engine), we either:

1. Clone the builder before each construction (cheap — schemas are
   `Arc`-backed), or
2. Construct all three conversations from the same builder by passing it
   through a small helper that takes `(builder_clone, layer, group)` and
   returns a Sequence.

Option 1 is the chosen direction — explicit, matches the existing
pattern, no new helper required. `projection::Builder` already implements
`Clone`.

### Persistence semantics

Each new conversation has its own `TimelineId` and writes turns through
the same substrate redo log every other conversation uses. On daemon
restart, the redo-log replay path rehydrates these timelines just like
the dialogue timelines — except their turns are pure prefills with no
decode tail. The walker-driven hydration path already handles this
correctly (no special casing needed): substrate restores the turn
records and their KV residencies; the next projection that targets
`dialogue` retrieves from the repo-map and code-reading timelines via
BDP scan as if they had always been there.

If a developer wipes `.substrate/`, the next startup re-runs both phases
from scratch — same as how the section-prefill pass currently rebuilds
on a fresh workspace. v1 always re-runs both phases unconditionally;
detecting "no change since last run" is the shadow-file follow-up.

---

## Test strategy

Three tiers, mirroring the existing test layout
([`zend/tests/infinite_conversation_smoke.rs`](../zend/tests/infinite_conversation_smoke.rs)
+ `_deep.rs`).

### Tier 1 — Pure-function unit tests (no model)

Unit-testable independent of the engine, run on every `cargo test`.

```
zend/src/repo_scan/walk.rs::tests
    ├── walk_respects_gitignore
    ├── walk_filters_by_extension_allowlist
    ├── walk_skips_oversize_files
    ├── walk_excludes_zend_dir
    └── walk_metadata_counts_lines_correctly

zend/src/repo_scan/render.rs::tests
    ├── render_is_deterministic_on_same_input
    ├── render_groups_files_by_directory
    ├── render_includes_module_structure_for_cargo_workspace
    └── render_round_trip_byte_identical

zend/src/code_read/parsers/rust.rs::tests
    ├── extracts_top_level_fn
    ├── extracts_impl_blocks_with_method_scopes
    ├── extracts_struct_and_trait_definitions
    ├── groups_imports_into_one_chunk
    └── splits_oversize_function_at_sub_block

zend/src/code_read/parsers/python.rs::tests        (same shape)
zend/src/code_read/parsers/typescript.rs::tests    (same shape)
zend/src/code_read/parsers/go.rs::tests            (same shape)

zend/src/code_read/header.rs::tests
    ├── header_format_matches_design_doc
    ├── header_hierarchy_uses_gt_separator
    └── header_is_deterministic
```

Tests use the `tempfile` workspace dev-dep (already present) for
on-disk fixture trees. Each fixture is a handful of small files in a
`TempDir`, scanned through `walk_workspace` and asserted against an
expected `RepoMap` value or rendered tree string.

### Tier 2 — Daemon integration (engine, no model load)

Run the full `repo_scan` + `code_read` pipeline against a synthetic
workspace using a **mock engine** that records the `(user, assistant)`
pairs passed to `insert_turn` but doesn't actually run a model. Verifies
the wiring without the 30B-load cost.

Implementation: a `MockSequence` trait in `zend/tests/common/` that
captures `insert_turn` calls into a `Vec<(String, String)>`. The
ingestion modules accept `&dyn TurnInserter` instead of `&mut Sequence`
behind a small newtype — production wires real `Sequence`, tests wire
the mock.

```
zend/tests/repo_scan_integration.rs
    ├── repo_scan_emits_one_turn_with_expected_user_prompt
    ├── repo_scan_assistant_text_matches_render_output
    └── repo_scan_progress_reports_file_walk_then_completion

zend/tests/code_read_integration.rs
    ├── code_read_one_turn_per_scope_per_file
    ├── code_read_user_prompt_is_scope_header
    ├── code_read_assistant_text_is_raw_source_slice
    ├── code_read_skips_files_outside_watch_patterns
    └── code_read_falls_back_to_fixed_window_on_parse_failure
```

### Tier 3 — End-to-end with the real model (`#[ignore]` by default)

Same harness as `infinite_conversation_smoke` — loads Qwen3-30B-A3B,
runs the actual prefill, then issues a developer query that should
retrieve content from the repo-map and code-reading timelines.

```
zend/tests/zen_code_phase12_smoke.rs

    #[test]
    #[ignore = "Tier 3: loads Qwen3-30B-A3B + scans workspace + recall (~3 min)"]
    fn phase12_recalls_a_known_function_in_the_test_fixture() {
        // 1. Spin up zend against a controlled fixture workspace
        //    containing a known file with a known function name
        //    that's unlikely to appear anywhere else.
        // 2. Wait for LoadStep::CodeRead -> mark_ready.
        // 3. Issue a dialogue query: "Which file defines
        //    fn xyzzy_unique_identifier_42 ?"
        // 4. Assert the response mentions the expected file path.
    }
```

This is the proof that the foundational layers actually contribute to
the dialogue projection — if BDP retrieval is broken or the layer
masking is wrong, this query fails. The test fixture is a small,
deterministic mini-workspace checked into `zend/tests/fixtures/`.

A second `#[ignore]` test exercises the substrate-restart path: run
the pipeline once, kill the daemon, restart against the same workspace,
verify both timelines are recovered with the same turn count and the
same recall query still works.

---

## Open questions (track separately, do not block this work)

1. **Shadow-file format on disk.** The design doc specifies per-file
   `.{filename}.zend.yaml` storage with phase-keyed turns. v1 stores
   nothing extra — only the substrate redo log. The shadow-file layer
   is a clean follow-up that reads from the substrate's recovered
   timelines and serialises per file.

2. **Reconcile waves.** File-watch on the workspace tree → re-carve →
   re-insert. The mechanism is well-defined; ordering relative to the
   live dialogue layer (do we pause new chat during reconcile? mid-turn
   re-projection?) wants its own document.

3. **Carve quality measurement.** A tier-3 calibration suite that
   measures BDP MRR / Top-1 over the code-reading layer against
   handwritten ground-truth queries. The depth weights in
   `projection.yaml` for `code_reading` are already calibrated
   (`sem:3, prag:4`); a fresh measurement after this work lands will
   confirm the carve faithfully reproduces those numbers.

4. **Configuration.** Watch patterns + size caps are hard-coded in v1.
   `.zend/config.yaml` parsing already exists for other knobs; extending
   it to cover the scan inputs is a one-screen change but wants its own
   review.

---

## Implementation order

The work breaks down cleanly along module boundaries — each piece can
land, be reviewed, and ship its own tests before the next one starts.

1. **`zend/src/repo_scan/walk.rs`** + tests — file enumeration + metadata,
   no model in the loop.
2. **`zend/src/repo_scan/render.rs`** + tests — tree formatter, byte-identical
   determinism.
3. **`zend/src/repo_scan/mod.rs`** — public `ingest_repo_map` entry point,
   threads `LoadProgress::set_step_progress`.
4. **Tier 2 mock-engine harness** in `zend/tests/common/` — so 5 and 7
   can be integration-tested without the model.
5. **Wire Phase 1 into `InferenceState::load`** — `repo_map_conv` field,
   `LoadStep::RepoScan` transition, mock-engine integration test.
6. **`zend/src/code_read/parsers/{rust,python,typescript,javascript,go,markdown,structured_config,fallback}.rs`**
   + per-parser unit tests — each parser lands independently.
7. **`zend/src/code_read/carve.rs`** + tests — language dispatch +
   tier-fallback policy.
8. **`zend/src/code_read/header.rs`** + tests — deterministic scope-header
   rendering.
9. **`zend/src/code_read/mod.rs`** — public `ingest_code_reading` entry
   point, threads progress.
10. **Wire Phase 2 into `InferenceState::load`** — `code_read_conv` field,
    `LoadStep::CodeRead` transition, mock-engine integration test.
11. **Tier 3 end-to-end test** — Qwen3-30B-A3B load, recall query, pass.

Each step is reviewable on its own; nothing later than step 5 changes the
behaviour of the running daemon for users who haven't opted into the new
phases. Step 11 is the gate that says we shipped: the dialogue layer
actually reaches back into the foundational layers and retrieves what
the developer asked about.
