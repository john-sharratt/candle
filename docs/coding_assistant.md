# Zen Code — Design Document

## Overview

Zen Code is a persistent AI coding assistant that accumulates institutional knowledge about a codebase over time. It consists of two components: a local inference daemon (`zend`) that runs the LLM and manages the persistent context, and a VS Code extension (`zen-vscode`) forked from Continue that provides the developer interface. The daemon also serves a read-only web chat interface for browser-based codebase queries.

The system is architecturally distinct from every existing coding assistant in one respect: it never starts over. A single unbounded trunk session persists across the lifetime of the project. The trunk is built through phased parallel analysis (repo map, code reading, static analysis, dependency analysis, architectural analysis, critical analysis), extended daily with converged developer conversations and overnight dream explorations, and stored as version-controlled YAML shadow files alongside the source code. Institutional knowledge compounds over time — the longer the system runs, the deeper its understanding becomes.

The inference engine is the custom Candle fork described in the companion paper, providing unbounded three-tier paged context with O(1) error at any depth, attentional provenance indexing for 3–10ms retrieval over the full knowledge base, adaptive per-block KV quantisation, and 64 concurrent session capacity on a single consumer GPU.

---

## Component 1: `zend` — The Inference Daemon

### Purpose

A long-running local binary that owns the LLM, the persistent KV cache, the provenance index, and the cognitive lifecycle (active reasoning, daydreaming, sleep).

### Responsibilities

- Run Qwen3-30B-A3B (or Qwen3.5-35B-A3B) on local GPU via the custom Candle inference engine
- Maintain a single persistent trunk session per repository with fork-based developer sessions
- Serve an OpenAI-compatible `/v1/chat/completions` endpoint on localhost for the VS Code extension
- Serve a read-only web chat interface on port 80 for browser-based codebase queries
- Manage the three-tier KV cache (VRAM hot, CPU warm, disk cold)
- Run the attentional provenance index over shadow files and `.zend/` knowledge base
- Execute the phased trunk analysis pipeline (Phases 1–6) on startup and reconcile waves
- Execute cognitive lifecycle: active query serving, daydreaming during idle, overnight sleep (converge, reconcile, dream, rebuild)
- Watch the working tree for file changes and queue reconcile waves
- Load and save shadow files (`.{filename}.zend.yaml`) alongside source files
- Load and save daily histories and dream logs in `.zend/`

### API Surface

```
POST /v1/chat/completions           — OpenAI-compatible, used by VS Code extension
                                       auto-creates or resumes fork based on message prefix hash
POST /v1/chat/completions?mode=web  — ephemeral fork, no tools, read-only system prompt
POST /v1/zen/status                 — daemon health, GPU utilisation, index stats, active forks
POST /v1/zen/rebuild                — trigger manual full or incremental rebuild
POST /v1/zen/dream                  — force a daydream cycle on current fork
POST /v1/zen/converge               — trigger manual sleep/convergence of all forks
POST /v1/zen/sync                   — force immediate fork refresh onto latest trunk head
GET  /v1/zen/retrieval-log          — last N retrieval events with fingerprint scores
GET  /v1/zen/session                — current fork metadata (depth, token count, tier distribution)
GET  /v1/zen/trunk                  — trunk metadata (total depth, last convergence, fork count, last update)
GET  /v1/zen/forks                  — list active forks with developer, start time, depth, last refresh
GET  /                              — web interface SPA (read-only codebase chat)
```

---

### Session Management — Fork Model with Live Trunk Sync

The daemon maintains a single persistent **trunk** session per repository. The trunk is the accumulated institutional knowledge: phased code analysis, daily conversation histories, dream logs. Unlike a traditional immutable trunk, it grows **continuously** — incremental analysis, daydream outputs, and file change reasoning append to the trunk throughout the day, not only during sleep.

When a developer connects, the daemon **forks** from the trunk. The session ID is the hash of the trunk's first N tokens — any connection sharing that prefix is identified as belonging to this repo. The fork shares the trunk's KV cache via page-table reference into the same paged blocks (copy-on-write semantics). No re-prefill of the trunk prefix. The fork only prefills whatever new content the extension adds on top (system prompt, tool context, developer greeting).

```
TRUNK (persistent, grows continuously during the day)
│
│  ← incremental analysis appended
│  ← daydream insights appended
│  ← file change reasoning appended
│
├── Fork A (Alice, working on auth refactor)
│   ├── query → response → tool call → response → ...
│   └── daydream cycles interleaved during idle
│
├── Fork B (Bob, working on schema migration)
│   └── query → response → ...
│
├── Fork C (Carol, adding new API endpoint)
│   └── query → response → ...
│
└── Fork W (web interface, ephemeral, read-only)
    └── query → response → ... (discarded on tab close)
```

**Live trunk updates.** The trunk is not frozen when forks are active. As the daemon processes file changes, completes incremental analyses, or generates daydream insights, these append to the trunk. Because the system uses provenance-selected attention (not full sequential attention), the fork does not need sequential consistency with the trunk — it needs the provenance index to reflect the latest trunk state so that retrieval surfaces current knowledge. The provenance index is shared and updates atomically as new trunk content is fingerprinted.

**Fork refresh.** When the trunk advances meaningfully — new analysis completed, significant daydream insight indexed, another developer's file changes analysed — active forks are refreshed:

1. The fork's conversation history (queries, responses, tool calls) is captured as a compact token sequence
2. A new fork is spawned from the updated trunk head
3. The captured conversation is replayed via prefill onto the new fork
4. The old fork's KV cache is released
5. The developer's next query is served from the refreshed fork — they see no interruption

The cost is one prefill of the fork's conversation length. A fork with 50 turns of developer interaction is maybe 20K–50K tokens. At parallel prefill throughput (2,500 t/s) that's 8–20 seconds in the background. The developer continues working on the old fork until the refresh completes, then switches atomically.

```
TRUNK v1 ─── Fork A (working) ──── query → response → ...
  │
  │  ← new analysis appended
  │
TRUNK v2 ─── Fork A' (refreshed) ── [replayed conversation] → ready
  │
  │          Fork A (old) ── discarded after switchover
```

**Implications for multi-developer collaboration.** Fork refresh means all developers see each other's work in near-real-time without waiting for sleep convergence:

1. Alice saves a file change to `auth_handler.rs`
2. The daemon detects the change and runs incremental analysis
3. The analysis appends to the trunk
4. The provenance index updates with new fingerprints
5. Bob's fork is scheduled for refresh — his next idle period triggers replay onto the new trunk head
6. Bob asks "does the schema migration affect auth?" — the provenance system retrieves Alice's fresh analysis from the updated trunk
7. The answer incorporates Alice's work that was committed minutes ago

**Refresh scheduling.** Not every trunk update triggers a fork refresh. The daemon uses a significance threshold:

```
IMMEDIATE    — another developer changed a file in the same module
               the fork developer is actively working on
SOON (1min)  — new architectural analysis completed for a related module
LAZY (5min)  — daydream insight indexed, unrelated file change analysed
SKIP         — trunk metadata update only (timestamps, stats)
```

The developer can also pull manually via a command palette action: "Zen: Sync with trunk".

**Collaborative agents.** Fork refresh enables multiple AI agents working on the same codebase simultaneously with shared awareness. Two agents can be forked from the same trunk, assigned different tasks, and as each makes progress their file changes flow through the trunk and into each other's refreshed forks. No coordination protocol needed — the trunk mediates everything.

**Fork lifecycle:**
```
CONNECT  → fork from trunk head (instant, shared KV prefix)
ACTIVE   → developer queries served from fork
IDLE     → daydream cycles on fork (low priority)
REFRESH  → trunk has advanced, fork replays onto new trunk head
CONVERGE → sleep processes fork into trunk (nightly)
DISCARD  → fork KV cache released
```

**Single developer simplification:** with one developer, there is one fork. The trunk still updates during the day as incremental analysis completes. Fork refresh still occurs. Convergence is simpler — no cross-fork reasoning needed.

---

### Trunk Session — Phased Parallel Analysis

The trunk session is not a single linear conversation. It is structured as a sequence of **phases**, where each phase forks many parallel sessions (one per file in scope) and converges them back into the trunk before the next phase begins. Within a phase, parallel forks **cannot attend to each other's tokens** — they are independent and unaware of sibling sessions. However, they **can attend to all tokens from all sessions in all previous phases**. This creates a layered understanding that builds progressively from raw code to architectural reasoning.

```
PHASE 1 — Repo Map (single session, not per-file)
│
│   Enumerates full directory tree of all files in scope
│   with file sizes, types, and module structure
│
▼ CONVERGE — Phase 1 tokens visible to all subsequent phases
│
PHASE 2 — Code Reading (per-file, parallel)
│
├── Fork: src/auth/handler.rs        (scope-aware carving, dumps content)
├── Fork: src/auth/validator.rs
├── Fork: src/db/schema.rs
├── Fork: src/rate_limit/limiter.rs
└── ... (all files in scope)
│
▼ CONVERGE — all Phase 1 + 2 tokens visible to Phase 3
│
PHASE 3 — Static Analysis (per-file, parallel)
│
├── Fork: src/auth/handler.rs        (identifies structs, traits, functions)
├── Fork: src/auth/validator.rs      (can attend to Phase 1+2 of ALL files)
└── ...
│
▼ CONVERGE — all Phase 1 + 2 + 3 tokens visible to Phase 4
│
PHASE 4 — Dependency Analysis (per-file, parallel)
│
├── Fork: src/auth/handler.rs        (maps what this file depends on)
├── Fork: src/auth/validator.rs      (can attend to Phase 1+2+3 of ALL files)
└── ...
│
▼ CONVERGE — all Phase 1 + 2 + 3 + 4 tokens visible to Phase 5
│
PHASE 5 — Architectural Analysis (per-file, parallel)
│
├── Fork: src/auth/handler.rs        (reasons about purpose and design)
└── ...
│
▼ CONVERGE — all Phase 1 + 2 + 3 + 4 + 5 tokens visible to Phase 6
│
PHASE 6 — Critical Analysis (per-file, parallel)
│
├── Fork: src/auth/handler.rs        (evaluates quality, identifies risks)
└── ...
│
▼ CONVERGE — full code analysis complete

DAILY HISTORY — converged developer conversations (appended during sleep)

DREAM LOG — overnight speculative explorations (appended during sleep)
```

#### Phase Definitions

**Phase 1 — Repo Map.** A single session (not per-file) that enumerates the full directory tree of all files matching the watch patterns. The user turn asks:

```
List the complete file structure of this repository.
For each file, show: path, size in lines, and file type.
Organise by directory. Note any module structure
(Cargo.toml workspaces, Python packages, Go modules, etc).
```

The assistant turn is a **prefill** — the directory listing is generated by the daemon from the filesystem and injected directly. No decode needed. The output is a structured map of the entire repository:

```
src/ (42 files)
├── auth/
│   ├── handler.rs          (247 lines, Rust)
│   ├── validator.rs        (183 lines, Rust)
│   ├── token_format.rs     (91 lines, Rust)
│   └── mod.rs              (12 lines, Rust)
├── db/
│   ├── schema.rs           (156 lines, Rust)
│   └── queries.rs          (312 lines, Rust)
├── rate_limit/
│   └── limiter.rs          (198 lines, Rust)
├── middleware/
│   └── auth_middleware.rs  (87 lines, Rust)
└── ...

Cargo.toml (workspace: 3 members)
docs/ (6 files)
├── architecture.md         (1,204 lines, Markdown)
└── ...
```

**Why the repo map matters.** Every subsequent phase can attend to Phase 1. This means when Phase 4 (dependency analysis) processes `handler.rs`, the model already knows that `rate_limit/limiter.rs` exists, how large it is, and where it sits in the module hierarchy — even before reading its code. Without the repo map, the model can only reference files it has already seen in Phase 2 code reading. With the repo map, it knows the complete shape of the codebase from the start. This is particularly important for dependency analysis — the model can identify a dependency on a file it knows exists (from the map) even if that file's code hasn't been read yet in the current phase.

The repo map is also the cheapest phase — one prefill of a directory listing, typically under 1,000 tokens for a 200-file project. The provenance fingerprints captured here encode the structural shape of the repository, enabling retrieval queries like "which modules exist in the auth subsystem" to match directly.

**Phase 1 storage.** The repo map is stored in a single shadow file at the repo root: `.repo.zend.yaml`. It is regenerated on startup if any files have been added, removed, or moved. It does not track content changes — only structural changes to the file tree.

**Phase 2 — Code Reading (Scope-Aware Carving).** Before ingestion, the daemon parses each file into **scope-aware chunks** — complete logical units rather than arbitrary line ranges. Each chunk is a semantically coherent block of code: a function, a method, a struct/class definition, an impl block, a module-level constant group, an import section, or a top-level documentation block.

The parser is language-aware but lightweight — it uses tree-sitter or similar scope detection to identify function/method/class boundaries without full semantic analysis. For non-code files (markdown, YAML, config), the parser falls back to section-based chunking (headers, document boundaries) or fixed-size chunking as a last resort.

Each chunk gets a **scope header** in the user prompt:

```
File: /src/auth/handler.rs
Scope: impl AuthHandler > fn validate_token(&self, token: &str) -> Result<Claims>
Lines: 47-93
```

The assistant turn is a **prefill** — the raw code for that scope is injected directly as the assistant response. No decode needed. Pure ingestion.

**Why scope-aware carving matters for provenance.** The K/Q fingerprints captured during Phase 2 encode the semantic content of whatever the model processes. With naive line-based chunking, a fingerprint might encode half of one function and half of another — the provenance signal is blurred. With scope-aware chunking, the fingerprint for `fn validate_token` encodes exactly and only that function. The scope header amplifies this further — fingerprints encode both hierarchical location and content.

**Carving rules:**

```
SCOPE PRIORITY (highest to lowest):
  1. Function / method body       — one chunk per fn/method
  2. Struct / class / enum / trait — definition + fields as one chunk
  3. Impl block header            — the impl line itself
  4. Module-level constants       — grouped into one chunk
  5. Import / use statements      — grouped into one chunk
  6. Module-level documentation   — grouped into one chunk
  7. Top-level expressions        — one chunk per expression

SIZE LIMITS:
  - Maximum chunk: 150 lines (split at logical sub-blocks if exceeded)
  - Minimum chunk: 3 lines (tiny items grouped with adjacent similar items)

NON-CODE FILES:
  - Markdown: split by ## headers
  - YAML/TOML/JSON: split by top-level keys
  - Plain text: split by paragraph
  - Fallback: 100-line fixed chunks
```

**Scope header format** — hierarchy uses `>` as the nesting separator:

```
Rust:   File: src/lib.rs > mod cache > impl KvCache > fn seal_chunk
Python: File: src/auth/handler.py > class AuthHandler > def validate_token
TS/JS:  File: src/components/Auth.tsx > function AuthProvider > useEffect[1]
Go:     File: pkg/auth/handler.go > func (h *Handler) ValidateToken
```

The scope header is deterministic — same code produces same header. Fingerprints for unchanged code are reproducible across re-ingestion.

**Phase 3 — Static Analysis.** For each file, a user turn asks:

```
What structs, classes, traits, functions, enums, type aliases,
and other code artifacts are defined in this file?
```

The model decodes a structured enumeration of every code artifact. It can attend to Phase 1 (repo map) and Phase 2 (all code) of every file in scope.

**Phase 4 — Dependency Analysis.** For each file, a user turn asks:

```
What do the structs, traits, and functions in this file depend on?
For each dependency, identify: the source file, the specific artifact,
and whether the dependency is direct (import/call), transitive
(via an intermediate), or implicit (shared resource, convention, assumption).
```

The model can attend to Phases 1–3 (repo map, all code, all artifacts) of every file. It maps the dependency graph from genuine understanding, not just import statements.

**Phase 5 — Architectural Analysis.** For each file, a user turn asks:

```
Reason about the purpose and architectural significance of everything
in this file. Why does it exist? What role does it play in the system?
What design decisions does it embody? What assumptions does it encode?
```

The model can attend to Phases 1–4 of every file. The output is deep architectural understanding.

**Phase 6 — Critical Analysis.** For each file, a user turn asks:

```
Fully reason about what is good and what is bad about everything
in this file. Identify: code quality issues, architectural risks,
performance concerns, security vulnerabilities, maintainability
problems, missing error handling, implicit assumptions that could
break, and opportunities for improvement.
```

The model can attend to Phases 1–5 of every file. Daydreaming draws heavily on Phase 6 outputs.

#### Attention Visibility Matrix

| Analysing Phase | Can attend to |
|---|---|
| Phase 1 (repo map) | Nothing — filesystem enumeration, no prior context |
| Phase 2 (code reading) | Phase 1 (repo map) |
| Phase 3 (static analysis) | Phase 1 + 2 of ALL files |
| Phase 4 (dependency analysis) | Phase 1 + 2 + 3 of ALL files |
| Phase 5 (architectural analysis) | Phase 1 + 2 + 3 + 4 of ALL files |
| Phase 6 (critical analysis) | Phase 1 + 2 + 3 + 4 + 5 of ALL files |
| Daily history / dreams | Full code analysis + prior histories/dreams |
| Developer fork queries | Full trunk (all phases + all histories + all dreams) |

Parallel forks within the same phase **never** attend to each other. This is a design choice: the phase boundary is the consistency barrier.

#### Parallelism

Within each phase, all file forks run as concurrent sessions. The wave-batched expert kernel coalesces work across all forks. With 64 concurrent session capacity, a phase across 200 files completes in 4 batches of ~50. Prefill-dominant phases (Phase 1 repo map, Phase 2 code reading) are extremely fast. Decode-dominant phases (Phase 5, 6) are slower but still parallelised.

#### Reconcile Wave

When a file changes, a **reconcile wave** walks through Phases 2–6 for the changed file with **full attention over the existing trunk**.

```
File change detected: src/auth/handler.rs
(Phase 1 repo map is updated if files were added/removed/moved)

RECONCILE PHASE 2 — re-read the changed file
    attends to: existing trunk (all phases, all files)
    output: updated code content turns (full file re-carve and prefill)

RECONCILE PHASE 3 — re-analyse artifacts
    attends to: existing trunk + reconcile Phase 2

RECONCILE PHASE 4 — re-analyse dependencies
    attends to: existing trunk + reconcile Phase 2 + 3

RECONCILE PHASE 5 — re-reason architecture
    attends to: existing trunk + reconcile Phase 2 + 3 + 4

RECONCILE PHASE 6 — re-evaluate critically
    attends to: existing trunk + reconcile Phase 2 + 3 + 4 + 5
```

**Recursive deepening.** If a reconcile wave identifies that a change in file A affects files B and C, they are queued for their own waves. Each wave sees results of prior waves.

**Reconcile throttling:**

```
IMMEDIATE    — the changed file itself (always)
NEXT_BATCH   — direct dependents
DEFERRED     — transitive dependents (queued, processed during idle/sleep)
```

---

### Daily Conversation History

After the five code analysis phases, the trunk extends with **daily conversation history**. All developer forks from a given day are gathered and during sleep the daemon processes them into a consolidated daily entry.

When sleep triggers, the daemon converges daily forks:

1. **Gather** — collect all active forks from today
2. **Converge** — extract the full conversation history from each fork
3. **Summarise** — the model summarises each fork independently
4. **Reason** — structured questions over the day's combined work:

```
Q1: "What were the main objectives today? What was accomplished
     and what remains unfinished?"
Q2: "What architectural decisions were made? What was the reasoning
     and what alternatives were considered?"
Q3: "What new dependencies or implicit couplings were introduced?
     Do any create risks?"
Q4: "What did you learn about the codebase that you didn't know
     yesterday?"
Q5: "What should the developer focus on tomorrow?"
```

Each question is a user turn. The model decodes its answer with full trunk attention. The answers compound — Q5 can reference Q4's insights.

---

### Sleep — Dream Phase

After daily conversations are converged, the daemon enters the **dream phase**. Dreaming is speculative exploration — architectural what-if scenarios that the day's work suggests but nobody explicitly asked about.

**Scenario generation.** The daemon asks the model to generate 8–12 concrete scenarios:

```
Based on today's work and the current state of the codebase,
generate 8-12 speculative scenarios worth exploring. Each should
be a concrete architectural what-if starting with "If I was to..."

Prioritise scenarios that:
1. Connect to today's work
2. Address risks identified in Phase 5
3. Explore alternatives to recent decisions
4. Challenge unexamined architectural assumptions
```

If no significant work was done, scenarios are drawn from Phase 5 risks and prior dreams.

**Execution.** Each scenario runs as a parallel fork with full trunk attention. Dream forks cannot attend to each other. The model reasons freely — open-ended exploration within a time budget (4 hours total).

**Convergence.** After the dream window:

```
Q1: "Which dreams produced the most actionable insights?"
Q2: "Did any dreams discover the same problem from different angles?"
Q3: "Which dreams challenge existing architectural decisions?"
Q4: "What concrete suggestions should be surfaced tomorrow morning?"
```

---

### Full Sleep Sequence

```
SLEEP TRIGGER (cron or manual)
│
├── 1. CONVERGE daily forks
│   ├── gather all active forks
│   ├── summarise each fork
│   └── structured reasoning (Q1–Q5)
│       → write .zend/{date}.yaml
│
├── 2. RECONCILE changed files
│   ├── identify files changed today
│   ├── run reconcile waves (Phases 2–6)
│   ├── recursive cascade for affected dependents
│   └── update shadow files
│
├── 3. DREAM (4 hours)
│   ├── generate 8–12 speculative scenarios
│   ├── execute parallel dream forks
│   ├── converge dreams (Q1–Q4)
│   └── write .zend/{date}.dreams.yaml
│
├── 4. REBUILD provenance index
│   └── full index rebuild over updated trunk
│
└── 5. READY
    └── daemon awaits morning fork connections
```

Converge before reconcile (so reconciliation has today's context). Reconcile before dream (so dreams reason over the freshest analysis). Dream before rebuild (so the index includes dream insights). Full cycle: approximately 5–6 hours.

---

### Cognitive Lifecycle

```
ACTIVE       — developer is querying, full GPU priority to decode
               operates on the fork, not the trunk

DAYDREAM     — no query for >30 seconds, background free-association
               runs on the fork, retrieves from shared provenance index
               model wanders, explores connections, indexes insights

SLEEP        — triggered manually or by cron, five stages:
               CONVERGE → RECONCILE → DREAM → REBUILD → READY
```

Daydream runs as a low-priority concurrent session. Developer queries always pre-empt daydream generation.

---

### File Watching

The daemon watches the repo working tree via `inotify`/`fsevents`. On file save:

1. Raw file content is injected into the active fork's context (immediate, local to fork)
2. File is queued for reconcile wave (Phases 2–6)
3. When analysis completes, the result appends to the trunk
4. Stale provenance fingerprints are invalidated
5. New fingerprints are inserted
6. Active forks are scheduled for refresh based on significance threshold

A file save triggers two parallel paths: immediate raw injection into the developer's fork, and background analysis that flows through the trunk to all forks.

---

### GPU Resource Allocation

```
Active query session:    priority 0 (always served first)
Fork refresh (prefill):  priority 1 (replay onto new trunk head)
Daydream session:        priority 2 (uses idle cycles only)
Incremental rebuild:     priority 3 (background, interruptible)
```

All share the same expert cache, KV hot tier, and wave-batched kernel. Fork refresh runs at priority 1 — never pre-empts an active query.

---

### API-Enhanced Reasoning

When an external API key is provided at daemon launch, the daemon runs an asynchronous background process that progressively upgrades the trunk's analysis phases with higher-quality reasoning from an external API model. This does not replace the local Qwen analysis — it enhances it. The local analysis runs immediately and provides instant results. The API enhancement runs in the background and replaces local analysis turns with higher-quality versions as they complete.

**How it works:**

1. The daemon maintains a **last-touched priority queue** — files are ranked by how recently they were modified, with the most recently changed files at the top
2. A background thread picks the highest-priority file that has not yet been API-enhanced (or whose enhanced analysis is older than the current local analysis)
3. For the selected file and phase (Phases 3–6 — the decode phases where reasoning quality matters), the daemon uses the provenance system to construct the optimal context payload for Claude
4. The request is sent to the Claude API with the constructed context
5. The API response replaces the local analysis for that phase
6. The response text is prefilled locally through the local model to capture fresh Q/K provenance fingerprints
7. The shadow file is updated with the new analysis, fingerprints, and the external model's analyst tag
8. The trunk and provenance index update

**Provenance-driven context construction.** This is the key architectural advantage. The daemon doesn't send the API model the entire codebase — it uses the provenance system to construct a focused, relevant context window for each request. For Phase 4 (dependency analysis) of `auth_handler.rs`, the provenance system identifies:

- The Phase 2 code of `auth_handler.rs` itself (always included)
- The Phase 2 code of files that `auth_handler.rs` imports
- The Phase 3 artifacts of files that share implicit dependencies
- The Phase 5 architectural analysis of the containing module
- Any prior API-enhanced analysis of related files

The provenance scores determine what fits within Claude's context window, ranked by relevance. The API model reasons over a dense, curated context rather than a raw dump — producing better analysis than either model alone, because the provenance system has already done the retrieval work.

**Request format.** Each API call is a single chat completion with the constructed context as a system message and the phase prompt as the user message:

```
System: You are analysing a specific file within a larger codebase.
        The following context has been selected as most relevant
        to this file by an attentional provenance system.

        [provenance-selected context: code, artifacts, dependencies,
         prior analysis — ranked by relevance score, truncated to
         fit the API model's context window]

User:   [Phase prompt, e.g. "What do the structs, traits, and
         functions in this file depend on? For each dependency..."]
```

**Phase coverage.** Phases 1 and 2 (repo map and code reading) are prefill-only — no reasoning, no decode, no benefit from API enhancement. Phases 3–6 (static analysis, dependency analysis, architectural analysis, critical analysis) are decode phases where reasoning quality directly determines output quality. API enhancement targets these four phases.

**Processing order.** Within each file, phases are enhanced in order (3 → 4 → 5 → 6) because each phase's analysis is richer when prior phases are already API-enhanced. Across files, the last-touched priority queue ensures that the files the developer is actively working on get API-enhanced first.

```
PRIORITY QUEUE (last-touched ordering):

1. src/auth/handler.rs      (modified 10 minutes ago)
   └── Phase 3: pending → Phase 4: pending → Phase 5: pending → Phase 6: pending

2. src/db/schema.rs          (modified 2 hours ago)
   └── Phase 3: api-done → Phase 4: pending → ...

3. src/middleware/auth.rs    (modified yesterday)
   └── Phase 3: api-done → Phase 4: api-done → Phase 5: api-done → Phase 6: pending

4. src/rate_limit/limiter.rs (modified last week)
   └── all phases: api-done ✓
```

**Cost management.** The daemon tracks API spend and respects configurable limits:

```yaml
# .zend/config.yaml
enhanced_reasoning:
  enabled: true
  provider: anthropic             # anthropic | openai | openai_compatible
  api_key: ${API_KEY}
  model: claude-opus-4-6          # any model the provider supports
  api_base: null                  # override for openai_compatible
  daily_budget: 10.00             # USD, stop enhancement when reached
  priority: last_touched          # last_touched | round_robin | phase6_first
  phases: [3, 4, 5, 6]           # which phases to enhance
  max_context_tokens: 100000      # max tokens sent per request
```

When the daily budget is exhausted, the daemon continues with local model analysis only. The existing local analysis is always available — API enhancement is additive, never blocking.

**Shadow file tracking.** Each phase in the shadow file tracks which model produced the analysis and whether API enhancement is complete for the current file version:

```yaml
source_hash: sha256:a1b2c3d4...    # hash of source at analysis time

phases:
  static_analysis:
    analyst: qwen3-30b-a3b          # local analysis (immediate)
    enhanced_analyst: claude-opus-4-6  # or gpt-5, gemini-2.5-pro, etc
    enhanced_at: 2026-04-05T15:30:00Z
    enhanced_source_hash: sha256:a1b2c3d4...  # source hash when API model ran
    turns:
      - role: assistant
        content: |
          [API-generated analysis, prefilled locally for fingerprints]
    fingerprints: { ... }            # from local prefill of API response text

  dependency_analysis:
    analyst: qwen3-30b-a3b
    enhanced_analyst: null             # not yet enhanced
    enhanced_at: null
    enhanced_source_hash: null
    # ...

  architectural_analysis:
    analyst: qwen3-30b-a3b
    enhanced_analyst: claude-opus-4-6
    enhanced_at: 2026-04-05T15:45:00Z
    enhanced_source_hash: sha256:a1b2c3d4...
    # ...
```

**Completion logic.** The `enhanced_source_hash` field is the key. The daemon skips enhancement for a phase when `enhanced_source_hash` matches the current `source_hash` — meaning the API model has already reasoned over this exact version of the file for this phase. When the file changes, `source_hash` updates but `enhanced_source_hash` retains the old value, so the mismatch re-queues the file for enhancement.

A file is **fully enhanced** when all four decode phases (3–6) have `enhanced_source_hash == source_hash`. Fully enhanced files drop off the priority queue entirely. Over time, if left running, every file in the codebase reaches full API enhancement and the background thread idles — no wasted API calls.

```
PRIORITY QUEUE STATE (steady state after extended operation):

src/auth/handler.rs      — fully enhanced ✓ (all phases, current hash)
src/db/schema.rs          — fully enhanced ✓
src/middleware/auth.rs    — fully enhanced ✓
src/rate_limit/limiter.rs — fully enhanced ✓
... (entire codebase)

Queue: empty — background thread idles until a file changes
```

When a file changes:
1. Reconcile wave runs Phases 2–6 with local model (immediate)
2. `source_hash` updates to new value
3. `enhanced_source_hash` on all phases now mismatches → file re-enters the queue
4. Background thread picks it up and re-enhances phases 3 → 4 → 5 → 6
5. Each phase's `enhanced_source_hash` updates as it completes
6. All four match → file drops off queue again

The system converges to full API enhancement and stays there, re-enhancing only what changes. Steady-state API cost approaches zero for stable codebases. The system converges to full enhancement regardless of which provider is used.

**The flywheel.** API-enhanced analysis produces richer dependency graphs, deeper architectural reasoning, and more precise critical evaluation. These richer turns become part of the trunk. When the next API request is constructed, the provenance system draws on these richer turns as context — producing even better analysis for the next file. The API model is enhancing its own future context. The more files that get API-enhanced, the better every subsequent enhancement becomes.

**Offline operation.** If no API key is provided, or if the network is unavailable, the daemon runs entirely on local model analysis. The architecture doesn't depend on external APIs — it benefits from them. A developer on a plane has full functionality with local model analysis. When they land and the daemon reconnects, the background enhancement resumes where it left off.

---

### Web Interface — Read-Only Codebase Chat

The daemon serves a web chat application on port 80 (configurable). A browser-accessible interface for anyone to ask questions about the codebase without installing the VS Code extension, without tools, and without the ability to modify anything.

**Architecture.** Spawns an **ephemeral fork** from the trunk on each browser connection. Shares the trunk's KV cache prefix. Retrieves from the shared provenance index. Sees the full institutional knowledge.

**System prompt:**

```
You are a senior engineer with deep, comprehensive knowledge of
this codebase. You have read every file, analysed every dependency,
reasoned about every architectural decision, and critically
evaluated every module.

Answer any question the user asks about the code, its architecture,
its design decisions, its dependencies, its risks, and its history.

You cannot modify any files. You cannot run any commands. You are
a knowledgeable advisor, not an agent.

If documentation, papers, or other non-code files are in the
repository, you have the same depth of understanding of those.
```

**Non-persistent sessions.** Browser tab closes, fork is discarded. No history saved. Web forks are never included in daily convergence.

**Use cases:** paper reviewers, new hire onboarding, code review alongside PRs, demo/verification (the web app from the paper).

**Implementation.** SPA served as static files. Backend is `/v1/chat/completions?mode=web`. Web queries run at priority 0.

---

## Component 2: `zen-vscode` — The VS Code Extension

### Purpose

A fork of Continue that replaces Continue's retrieval and context gathering with the daemon's provenance-based retrieval, while retaining Continue's UI, tool execution, and editor integration.

### What We Keep From Continue

- Chat sidebar UI
- Inline diff application
- File editing tool execution
- Terminal command execution
- Agent mode UI (permission prompts, tool call display)
- Tab completion integration
- Keyboard shortcuts and command palette

### What We Remove From Continue

- Codebase embedding and indexing (`@codebase` provider)
- File content retrieval for context
- All retrieval-related context providers
- Retrieval tool definitions from the system prompt
- Continue's own session management

### What We Add

- **Retrieval log panel** — sidebar panel showing provenance retrieval at each hop. Displays: which entries were retrieved, fingerprint match scores, cognitive state that triggered retrieval, how the working set evolved.

- **Daemon status bar** — connection state, cognitive mode (active/daydream/sleep), GPU utilisation, session depth, index size, trunk sync status (click to force sync).

- **Dream viewer** — tree view of daydream and sleep-dream insights organised by module. Links to `.zend/` YAML files.

- **Context explorer** — tree view of shadow files and `.zend/` showing institutional knowledge across all five phases, decision history, dream log, session summaries. Inline diff support.

- **Rebuild and sync commands** — "Zen: Sync with trunk", "Zen: Trigger rebuild", "Zen: Force daydream", "Zen: Converge all forks".

- **Pre-selected tool injection** — provenance identifies query intent before generation, extension sends only the relevant tool definition.

### System Prompt

```
You are a senior engineer with deep knowledge of this codebase.
The context provided contains architectural analysis, dependency
information, decision history, and code relevant to the query.
Reason over what is provided. Do not search for additional context.

When you need to make changes, use the provided tool.
```

### Communication With Daemon

```
Extension → Daemon:   standard OpenAI /v1/chat/completions
Daemon → Extension:   SSE streaming response
Extension → Daemon:   /v1/zen/* endpoints for status, control, retrieval logs
```

---

## Component 3: Storage — Shadow Files and `.zend/`

### Overview

Institutional knowledge is stored in two locations, both version-controlled alongside the source code:

- **Shadow files** (`.{filename}.zend.yaml`) — per-file, live alongside the source files they describe. Contain Phase 2–6 analysis turns, provenance fingerprints, and reconcile history.
- **`.zend/` directory** — repo root. Contains daily conversation histories, dream logs, and daemon configuration.

```
project-root/
├── src/
│   ├── auth/
│   │   ├── handler.rs
│   │   ├── .handler.rs.zend.yaml
│   │   ├── validator.rs
│   │   └── .validator.rs.zend.yaml
│   └── db/
│       ├── schema.rs
│       └── .schema.rs.zend.yaml
├── .zend/
│   ├── config.yaml
│   ├── 2026-04-01.yaml
│   ├── 2026-04-01.dreams.yaml
│   ├── 2026-04-02.yaml
│   ├── 2026-04-02.dreams.yaml
│   └── ...
└── .gitattributes
```

### Shadow File Schema

```yaml
source_file: src/auth/handler.rs
source_hash: sha256:a1b2c3d4...
last_analysed: 2026-04-05T14:22:00Z
model: qwen3-30b-a3b
provenance_model: qwen3-30b-a3b

phases:
  code_reading:
    turns:
      - scope: "module imports"
        lines: [1, 11]
        role: user
        content: "File: /src/auth/handler.rs\nScope: module imports\nLines: 1-11"
      - scope: "module imports"
        lines: [1, 11]
        role: assistant
        content: |
          use crate::session::SessionStore;
          use crate::token::Claims;
        prefill: true
      - scope: "impl AuthHandler > fn validate_token"
        lines: [47, 93]
        role: user
        content: "File: /src/auth/handler.rs\nScope: impl AuthHandler > fn validate_token\nLines: 47-93"
      - scope: "impl AuthHandler > fn validate_token"
        lines: [47, 93]
        role: assistant
        content: |
          [raw function code]
        prefill: true
    fingerprints_per_scope:
      "module imports":
        k_lower: "base64:..."
        k_mid: "base64:..."
        k_upper: "base64:..."
        q_lower: "base64:..."
        q_mid: "base64:..."
        q_upper: "base64:..."
        scales: "base64:..."
      "impl AuthHandler > fn validate_token":
        k_lower: "base64:..."
        # ...

  static_analysis:
    turns:
      - role: user
        content: "What structs, classes, traits, functions..."
      - role: assistant
        content: |
          [decoded artifact enumeration]
    fingerprints: { k_lower: "base64:...", ... }

  dependency_analysis:
    turns: [...]
    fingerprints: { ... }

  architectural_analysis:
    turns: [...]
    fingerprints: { ... }

  critical_analysis:
    turns: [...]
    fingerprints: { ... }

reconcile_history:
  - date: 2026-04-05T14:22:00Z
    trigger: file_change
    source_hash_before: sha256:9f8e7d6c...
    source_hash_after: sha256:a1b2c3d4...
    cascade_depth: 0
    affected_dependents:
      - src/middleware/auth_middleware.rs
```

### Daily History Schema (`.zend/{date}.yaml`)

```yaml
date: 2026-04-05
type: daily_history

conversations:
  - fork_id: alice-auth-refactor
    developer: alice
    started: 2026-04-05T09:15:00Z
    ended: 2026-04-05T17:42:00Z
    turns:
      - role: user
        content: "Should we get rid of JWT?"
      - role: assistant
        content: |
          [full response]
    fingerprints: { k_lower: "base64:...", ... }

  - fork_id: bob-schema-migration
    developer: bob
    turns: [...]
    fingerprints: { ... }

summary:
  turns:
    - role: user
      content: "What were the main objectives today?..."
    - role: assistant
      content: |
        [decoded summary]
    # ... Q2–Q5
  fingerprints: { k_lower: "base64:...", ... }

cross_fork_reasoning: |
  [if multiple developers: cross-fork implications identified during convergence]

model: qwen3-30b-a3b
provenance_model: qwen3-30b-a3b
```

### Dream Log Schema (`.zend/{date}.dreams.yaml`)

```yaml
date: 2026-04-05
type: dream_log
dream_duration_hours: 4.0
scenarios_completed: 8

scenarios:
  - id: dream-001
    assignment: |
      If I was to refactor the authentication system to use
      event-driven token validation, what would the work look like?
    source: daily_work
    related_files:
      - src/auth/handler.rs
      - src/auth/validator.rs
    exploration:
      turns:
        - role: user
          content: |
            [dream assignment]
        - role: assistant
          content: |
            [free-form dream reasoning]
      fingerprints: { k_lower: "base64:...", ... }

  - id: dream-002
    assignment: |
      If I was to split the rate limiter into a separate service...
    source: phase5_risk
    exploration:
      turns: [...]
      fingerprints: { ... }

dream_summary:
  turns:
    - role: user
      content: "Which dreams produced the most actionable insights?..."
    - role: assistant
      content: |
        [decoded ranking]
    # ... Q2–Q4
  fingerprints: { k_lower: "base64:...", ... }

model: qwen3-30b-a3b
provenance_model: qwen3-30b-a3b
```

### Shadow File Lifecycle

**Loading (startup).** For each file matching watch patterns:

1. Check for shadow file — if missing, run Phases 2–6, write shadow file
2. If present — check `source_hash` against current file hash
3. Hash matches — load turns and fingerprints into trunk (prefill, no decode)
4. Hash differs — trigger reconcile wave
5. `provenance_model` differs — prefill existing content through new model to regenerate fingerprints (no decode needed)

**Saving.** Written as each phase completes. Atomic writes (temp + rename).

**Model change.** Turn content is text (model-independent). Fingerprints are model-specific. On model change, prefill existing content to regenerate fingerprints. Fast — no generation needed.

**Change detection.** Event-driven via `inotify`/`fsevents`. No polling.

### Trunk Assembly

On startup:

1. Load all shadow files
2. Load all `.zend/{date}.yaml` and `.zend/{date}.dreams.yaml`
3. Build dependency graph from Phase 4 data
4. Topologically sort files by dependency order
5. Assemble trunk: Phase 1 (repo map) → Phase 2 (sorted) → Phase 3 → ... → Phase 6 → daily histories (chronological) → dream logs (chronological)
6. Prefill to build KV cache
7. Build provenance index from loaded fingerprints

Older daily histories and dream logs are **summary-dominant** in the trunk — only summary and reasoning turns consume trunk tokens. Raw conversations are in `.zend/` YAML and retrievable via provenance but don't consume trunk token budget. This keeps the trunk bounded while retrieval depth is unbounded — the Asymptotic Numerical Stability theorem applied to the product design.

### .gitattributes

```
**/.*.zend.yaml linguist-language=YAML
.zend/**/*.yaml linguist-language=YAML
```

---

## Configuration — `.zend/config.yaml`

```yaml
daemon:
  port: 8420                      # API port
  web_port: 80                    # web interface port
  model: qwen3-30b-a3b
  gpu_layers: all
  expert_cache_percent: 50
  kv_compression: C3              # balanced tier default

cognitive:
  daydream_idle_threshold: 30s
  daydream_max_tokens: 500
  sleep_schedule: "0 2 * * *"     # cron: 2am daily
  dream_hours: 4
  dream_scenarios: 10

enhanced_reasoning:
  enabled: true                   # false for fully offline / local-only operation
  provider: anthropic             # anthropic | openai | openai_compatible
  api_key: ${API_KEY}
  model: claude-opus-4-6          # any model the provider supports
  api_base: null                  # override for openai_compatible providers
  daily_budget: 10.00             # USD, stop enhancement when reached
  priority: last_touched          # last_touched | round_robin | phase6_first
  phases: [3, 4, 5, 6]           # which phases to enhance
  max_context_tokens: 100000      # max tokens sent per request

context:
  repo_root: auto
  watch_patterns:
    - "**/*.rs"
    - "**/*.py"
    - "**/*.ts"
    - "**/*.go"
    - "**/*.md"
  ignore_patterns:
    - "target/**"
    - "node_modules/**"
    - ".zend/**"
    - "**/.*.zend.yaml"
```

---

## Development Phases

### Phase 1 — Foundation (weeks 1–3)

- `zend` binary serving OpenAI-compatible endpoint with persistent KV cache
- Single trunk session with fork-based developer sessions
- Shadow file loading and saving (Phase 1 repo map and Phase 2 code reading only)
- Fork Continue, strip retrieval, point at localhost
- Basic system prompt, action tools only (file edit, terminal)

### Phase 2 — Phased Trunk Analysis (weeks 4–6)

- Full Phase 1–6 pipeline with repo map and scope-aware carving (tree-sitter)
- Parallel phase execution across files
- Shadow file schema with per-scope fingerprints
- Trunk assembly from shadow files on startup
- Provenance retrieval active during decode

### Phase 3 — Live Sync and Cognitive Lifecycle (weeks 7–10)

- File watching with reconcile waves
- Live trunk updates and fork refresh
- Daydream mode during idle cycles
- Retrieval log panel in VS Code extension
- Daemon status bar with trunk sync indicator
- Pre-selected tool injection

### Phase 4 — Sleep Cycle and API Enhancement (weeks 11–14)

- Daily conversation convergence with structured reasoning (Q1–Q5)
- Reconcile wave integration into sleep sequence
- Dream phase with scenario generation and parallel execution
- Dream convergence with structured reasoning (Q1–Q4)
- `.zend/` daily history and dream log storage
- API-enhanced reasoning: background integration with provenance-driven context
- Last-touched priority queue, daily budget tracking, provider-agnostic shadow file tagging
- Dream viewer and context explorer panels

### Phase 5 — Web Interface and Multi-Developer (weeks 15–18)

- Read-only web chat interface on port 80
- Ephemeral fork spawning for web sessions
- Multi-developer fork convergence with cross-fork reasoning
- Prompt engineering optimisation
- Documentation, packaging, installer
- Demo video production

---

## Name

**Zen Code** — from Battle Cities: The Awakening of Zen. The system that never stops thinking about your code.