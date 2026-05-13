# Cognitive Coding Architecture
**Design Document — v1.0 | March 2026**

---

## Abstract

This document specifies a novel coding agent architecture that extends an existing inference stack — Parallel KV Injection, Attentional Provenance Indexing, and KV tiering — to produce an agent with genuine persistent understanding of a codebase. Unlike existing coding assistants which perform one-shot RAG retrieval before generation, this architecture enables the model to continuously pull in exactly what it needs at each decode step, driven by its own attention signal. Combined with a structured learning phase and edit-triggered re-reasoning, the system maintains temporal coherence: its understanding of the codebase is always consistent with the actual state of the code.

---

## 1. The Problem With Current Coding Assistants

Every existing coding assistant — Cursor, Copilot, Aider — shares the same fundamental architecture: embed the codebase into vectors, retrieve the top-k chunks by cosine similarity before generation begins, and stuff them into a fixed context window. This design has three compounding failure modes.

**One-shot bet.** The retrieval decision is made before the model has started reasoning. By the time the model discovers it needs something, the context window is already fixed.

**Wrong signal.** Embedding similarity is a weak proxy for what the model will find cognitively relevant during reasoning. A type definition may be lexically distant from the bug site but architecturally critical.

**Stateless.** Each session starts cold. The model re-reads files it has read before, re-traces dependencies it has traced before, and loses any understanding accumulated in prior sessions.

> **Key insight:** The retrieval problem is not a ranking problem. It is a cognition problem. The model itself is the best oracle for what it needs — the signal just has to be captured and acted on.

---

## 2. Architecture Overview

The system is composed of four interlocking layers, each building on the existing inference stack.

| Component | Role |
|---|---|
| **Learning Phase** | Structured ingestion of the codebase with explicit reasoning and sleep boundaries, producing a consolidated KV representation of codebase understanding |
| **KV Tiering** | Unbounded context via GPU / RAM / NVMe hierarchy. Hot files in GPU KV, recent in RAM, archive on NVMe. Sequential access pattern maps naturally to mmap |
| **Q-Driven Online Retrieval** | Live Q vectors generated at each decode step drive continuous retrieval from the KV index. Context assembles itself token-by-token, driven by the model's own attention signal |
| **Edit-Triggered Re-Reasoning** | Any file write immediately triggers re-ingestion of that file, re-reasoning, and re-sleep. The agent resumes with understanding that includes the new code as if it had always existed |

---

## 3. The Learning Phase

### 3.1 Structured Ingestion

Before the agent begins any task, it ingests the entire codebase in a deliberate, structured pass. Files are fed with explicit metadata and boundary markers that direct the model when to reason and when to sleep.

Each file feed follows this structure:

- **File metadata header** — path, language, module membership, known dependents, last modified
- **Raw file content** with line-numbered boundaries
- **Explicit reasoning prompt** — what does this module do, what are its interfaces, what are its dependencies, what are its known issues
- **Sleep boundary marker** — integrate this understanding before proceeding

The structure of the feed shapes the structure of the resulting understanding. The model does not organise the knowledge spontaneously — the ingestion protocol directs it deliberately.

### 3.2 Sleep Consolidation

At each sleep boundary, the model consolidates the reasoning into the KV conversation history. This is not summarisation — it is genuine comprehension encoded as KV states. The conversation history after the learning phase contains:

- Module-level understanding of every file in the codebase
- Cross-module dependency relationships and data flow
- Architectural patterns and design decisions
- Known weaknesses, edge cases, and technical debt
- Interface contracts between modules

> **Critical property:** When a coding task begins, the agent does not retrieve understanding of the codebase. It already has that understanding, encoded in its conversation KV, the same way a senior engineer carries years of system familiarity before opening a file.

---

## 4. Q-Driven Online Retrieval

### 4.1 The Core Insight

In the existing Attentional Provenance Indexing design, Q vector fingerprints are captured during dedicated probe passes before generation. These probe passes simulate what the model would attend to if it were generating.

During actual decode, however, the model generates Q vectors at every single step as a free byproduct of normal inference. These Q vectors are the strongest possible signal of cognitive focus — not simulated attention, but real attention during actual reasoning. They are currently discarded after the attention calculation.

### 4.2 The Decode Loop

With online retrieval, the decode loop becomes:

1. Token N generates a Q vector as part of normal attention computation
2. The Q vector triggers an async flat INT8 scan over the KV index — running while token N+1 computes
3. If retrieval fires, the matched KV states are injected before token N+2
4. RoPE applies position rotations at attention compute time based on injected position indices — no positional encoding mismatch
5. Generation continues without pause

> **Positional encoding note:** RoPE is applied at attention compute time, not stored in the KV cache. Injected KV states are assigned position indices at injection time and RoPE handles the rotation correctly on the next attention step. Mid-decode injection is architecturally clean.

### 4.3 Near-Perfect Active Context

Because the active window assembles itself token-by-token driven by the model's own attention signal, at every decode step the model has exactly what it needs to generate the next token well. Not an approximation. Not a pre-assembled best guess. The actual optimal context for that exact moment in the reasoning chain.

This eliminates the lost-in-the-middle problem entirely. The model is not attending over a flat sequence where middle content decays — it is retrieving and activating only what is cognitively relevant at each step. Irrelevant content never activates and incurs no attention cost.

### 4.4 Large File Navigation

A file that would never fit in any context window — 50,000 lines, for example — has its KV state pre-computed and paged across the tier hierarchy. The model only ever holds the locally relevant slice in active attention at any moment, but it can reach anywhere in the file because retrieval is continuous and model-driven.

The model behaves as if it has read and comprehended the entire file, without the quadratic attention cost of attending over all of it simultaneously.

---

## 5. Tool Invocation Architecture

### 5.1 The Problem With Current Tool Selection

In current agentic coding systems, the model sees a list of available tools and must reason about which one to invoke. This reasoning step — selection from a large menu — is where most tool call failures occur: wrong tool selected, schema confusion from overlapping options, or hallucinated parameters.

### 5.2 Q-Driven Tool Selection

Tool selection follows the same mechanism as mood selection in the NPC architecture. As the model decodes toward a tool invocation, its Q vectors signal what it is about to do — write a file, run a command, execute tests. The attentional provenance system reads this signal and injects exactly one tool schema.

By the time the model is generating the invocation, the selection is already made. The model's task is purely parameter generation against a single known schema — a dramatically easier problem than selection from a large menu.

> **Reliability:** Tool call failures require selecting the wrong tool or confusing schemas. If only one schema is ever presented, the entire class of selection failure is architecturally eliminated.

### 5.3 Reduced Tool Pressure

Current agentic coding loops are tool-call heavy because the model must explicitly request everything it needs: read this file, search for this symbol, get this definition. Each call is a round trip that dominates the task timeline.

With Q-driven retrieval absorbing context assembly into the inference loop, tool calls are reserved for what they are actually for — side effects. Writing files, running tests, executing commands. The ratio of real work to overhead collapses in favour of real work.

---

## 6. Edit-Triggered Re-Reasoning

### 6.1 The Temporal Coherence Problem

Every current coding agent has a stale understanding problem. The agent makes an edit, but its internal model of the codebase still reflects the pre-edit state. Subsequent reasoning may contradict the edit, fail to propagate its implications, or lose track of what changed. Understanding drifts from ground truth as edits accumulate.

### 6.2 The Re-Reasoning Protocol

The instant the agent writes to a file, before resuming any coding activity, the following sequence executes:

1. Re-feed the modified file with its metadata and explicit reasoning prompt
2. Re-reason the file in the context of the existing consolidated understanding
3. Re-dream the implications — what other modules are affected, what interfaces have changed, what assumptions are now invalid
4. Re-sleep — integrate the updated understanding back into the conversation KV
5. Resume the coding task

The agent resumes from a state where it has already integrated the consequences of the edit — as if the new code had always existed. There is no lag between action and comprehension.

> **Temporal coherence:** The agent's self-model of the codebase is always consistent with the actual codebase. Not eventually consistent. Not approximately consistent. Always. This is a property no current coding system possesses.

### 6.3 Compounding Understanding

Each session adds new understanding. Refactors, new modules, discovered edge cases. The next learning phase consolidates that delta into the existing understanding. The agent becomes progressively more knowledgeable about the specific codebase over time — the same way a long-tenured engineer does. The closest human analogy is the difference between a contractor who has never seen the codebase and a staff engineer who has been living in it for two years.

---

## 7. Single Agent Design

### 7.1 Why Sub-Agent Architectures Are the Wrong Answer

Sub-agent orchestration is the standard response to large context and cognitive focus problems in current agentic systems. A planner decomposes tasks and spawns narrow sub-agents to execute discrete pieces. This architecture exists to solve two problems: context budget constraints and cognitive overload from competing demands on limited active parameters.

Both problems dissolve in this system.

### 7.2 Context Budget: Dissolved

The KV tiering architecture is unbounded. Context size is not a constraint. Irrelevant history sits cold in the tier hierarchy and never activates. There is no context budget to manage and therefore no reason to fork conversations for budget reasons.

### 7.3 Cognitive Focus: Dissolved

The near-perfect active context property means the model is never overwhelmed by irrelevant content — it never activates. The MoE expert routing handles specialisation at the parameter level. Q-driven retrieval handles specialisation at the context level. Two complementary focusing mechanisms operate simultaneously without requiring explicit architectural separation.

### 7.4 The Result

One persistent agent conversation. Full history. Unbounded KV tier. Q-driven active window assembling itself continuously. The agent plans, reasons, codes, runs tools, and integrates results — all in one conversation — and the context is always near-optimal for whatever it is currently doing.

Simpler architecture. Better performance. The system argues against its own complexity.

---

## 8. Contrast With Current Market

| Property | Current Assistants | This Architecture |
|---|---|---|
| Context assembly | One-shot RAG before generation | Continuous, token-by-token, model-driven |
| Retrieval signal | Embedding cosine similarity | Live Q vectors from actual reasoning |
| Session memory | None — starts cold every session | Persistent KV encoding full codebase understanding |
| Large file handling | Must fit in context window | Unbounded — active slice only, full file retrievable |
| Post-edit coherence | Stale until next retrieval | Immediate re-reasoning, always consistent |
| Tool selection | Model reasons over full tool menu | Q-driven single schema injection |
| Agent structure | Sub-agent orchestration | Single persistent conversation |
| Context focus | Pre-selected, fixed | Self-focusing per decode step |

---

## 9. Implementation Priorities

### Phase 1 — Learning Phase Protocol
Define and implement the structured file ingestion format: metadata headers, reasoning prompts, and sleep boundary markers. Implement the learning phase runner that feeds files in dependency order and triggers consolidation at boundaries. Validate that the resulting conversation KV encodes retrievable architectural understanding.

### Phase 2 — Online Q Retrieval Benchmarking
Benchmark the flat INT8 SIMD scan at decode-step granularity. Establish whether scan latency fits within one decode step's compute time on the 3090 — if so, retrieval is genuinely zero-latency and continuous. This is the critical performance gate for the entire architecture.

### Phase 3 — Edit-Triggered Re-Reasoning
Instrument the file write path to trigger re-ingestion automatically. Implement the re-reasoning and re-sleep sequence. Validate temporal coherence by testing that the agent's post-edit behaviour reflects the new code state immediately.

### Phase 4 — Q-Driven Tool Selection
Implement Q-fingerprint classification for tool intent, following the mood selection mechanism from the NPC architecture. Build the single-schema injection pipeline. Validate against tool selection error rates in current agentic loops.

---

## 10. Conclusion

The architecture described here is not an incremental improvement on existing coding assistants. It is a different class of system.

Existing systems retrieve context before reasoning. This system retrieves context during reasoning, driven by the model's own attention signal, at every decode step.

Existing systems start cold each session. This system carries genuine persistent understanding of the codebase, compounding over time.

Existing systems drift from ground truth as edits accumulate. This system maintains temporal coherence — its model of the codebase is always consistent with the actual code.

The result is not a coding assistant. It is an agent that has genuine expertise in a specific codebase — the computational equivalent of a staff engineer who has been living in the system for years.

---

> **Built on:** Rust/Candle inference engine, paged KV cache, adaptive MoE expert caching, quantised KV in batched paged attention kernels, Parallel KV Injection, and Attentional Provenance Indexing.