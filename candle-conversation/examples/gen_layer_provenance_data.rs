//! Generate deterministic synthetic layer-provenance KV/Q fixtures.
//!
//! Writes `signatures.prov` and `MANIFEST.json` to the specified output directory
//! (default: `tests/<type>_provenance_data/` relative to the crate root).
//!
//! # Usage
//!
//! ```sh
//! cargo run -p candle-conversation --example gen_layer_provenance_data -- --content-type code-reading
//! cargo run -p candle-conversation --example gen_layer_provenance_data -- --content-type bug-analysis --force
//! cargo run -p candle-conversation --example gen_layer_provenance_data -- --all --force
//! ```
//!
//! # Content types and items
//!
//! | Type                   | Items (8)                                                                              |
//! |------------------------|----------------------------------------------------------------------------------------|
//! | code-reading           | decode_step kv_arena attention_fwd quant_block bdp_scan moe_route prefill_run rope_enc |
//! | static-analysis        | cache_rs arena_rs compress_rs scan_rs scheduler_rs projection_rs engine_rs config_rs   |
//! | dependency-analysis    | cache_deps arena_deps compress_deps scan_deps scheduler_deps projection_deps engine_deps config_deps |
//! | architectural-analysis | paged_kv quant_policy bdp_retrieval moe_predict wave_batch three_tier o1_theorem proj_schema |
//! | critical-analysis      | kv_frag quant_drift bdp_collision sched_block mem_pressure dtype_mismatch attn_overflow moe_imbalance |
//! | bug-analysis           | chunk_oob q4_sign kv_misalign sink_scale mask_skip arena_leak dtype_cast flash_oob     |
//! | daily-history          | day_kv day_quant day_bdp day_moe day_proj day_calib day_bugfix day_batch               |
//! | dream-log              | dream_distrib dream_neural dream_stream dream_sinks dream_prefetch dream_cluster dream_dynwin dream_fedkv |
//!
//! Each type generates 8 × 16 = 128 scenarios (6 pos + 4 bnd + 4 neg + 2 no_match per item).

use std::path::{Path, PathBuf};

use candle_conversation::provenance::{ProvenanceFile, TokenSignature};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Generate synthetic layer-provenance KV/Q fixtures")]
struct Args {
    /// Content type to generate (mutually exclusive with --all).
    #[arg(long, value_enum)]
    content_type: Option<ContentType>,

    /// Generate all content types in sequence.
    #[arg(long, conflicts_with = "content_type")]
    all: bool,

    /// Output directory. Defaults to tests/<type>_provenance_data/ relative to crate root.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Overwrite existing files without prompting.
    #[arg(long)]
    force: bool,
}

// ── Content type ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ContentType {
    CodeReading,
    StaticAnalysis,
    DependencyAnalysis,
    ArchitecturalAnalysis,
    CriticalAnalysis,
    BugAnalysis,
    DailyHistory,
    DreamLog,
}

const ALL_TYPES: &[ContentType] = &[
    ContentType::CodeReading,
    ContentType::StaticAnalysis,
    ContentType::DependencyAnalysis,
    ContentType::ArchitecturalAnalysis,
    ContentType::CriticalAnalysis,
    ContentType::BugAnalysis,
    ContentType::DailyHistory,
    ContentType::DreamLog,
];

impl ContentType {
    fn dir_name(self) -> &'static str {
        match self {
            Self::CodeReading => "code_reading",
            Self::StaticAnalysis => "static_analysis",
            Self::DependencyAnalysis => "dependency_analysis",
            Self::ArchitecturalAnalysis => "architectural_analysis",
            Self::CriticalAnalysis => "critical_analysis",
            Self::BugAnalysis => "bug_analysis",
            Self::DailyHistory => "daily_history",
            Self::DreamLog => "dream_log",
        }
    }

    fn items(self) -> &'static [&'static str] {
        match self {
            Self::CodeReading => &[
                "decode_step", "kv_arena", "attention_fwd", "quant_block",
                "bdp_scan", "moe_route", "prefill_run", "rope_enc",
            ],
            Self::StaticAnalysis => &[
                "cache_rs", "arena_rs", "compress_rs", "scan_rs",
                "scheduler_rs", "projection_rs", "engine_rs", "config_rs",
            ],
            Self::DependencyAnalysis => &[
                "cache_deps", "arena_deps", "compress_deps", "scan_deps",
                "scheduler_deps", "projection_deps", "engine_deps", "config_deps",
            ],
            Self::ArchitecturalAnalysis => &[
                "paged_kv", "quant_policy", "bdp_retrieval", "moe_predict",
                "wave_batch", "three_tier", "o1_theorem", "proj_schema",
            ],
            Self::CriticalAnalysis => &[
                "kv_frag", "quant_drift", "bdp_collision", "sched_block",
                "mem_pressure", "dtype_mismatch", "attn_overflow", "moe_imbalance",
            ],
            Self::BugAnalysis => &[
                "chunk_oob", "q4_sign", "kv_misalign", "sink_scale",
                "mask_skip", "arena_leak", "dtype_cast", "flash_oob",
            ],
            Self::DailyHistory => &[
                "day_kv", "day_quant", "day_bdp", "day_moe",
                "day_proj", "day_calib", "day_bugfix", "day_batch",
            ],
            Self::DreamLog => &[
                "dream_distrib", "dream_neural", "dream_stream", "dream_sinks",
                "dream_prefetch", "dream_cluster", "dream_dynwin", "dream_fedkv",
            ],
        }
    }
}

// ── Constants ─────────────────────────────────────────────────────────────────

const TOKENS_PER_CHUNK: usize = 96;

// ── Manifest types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CaseType {
    Positive,
    Boundary,
    Negative,
    NoMatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Scenario {
    id: String,
    item: Option<String>,
    case_type: CaseType,
    system_prompt: String,
    user_prompt: String,
    assistant_prompt: String,
    turn_id: u64,
    byte_offset: u64,
    token_count: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    content_type: String,
    scenarios: Vec<Scenario>,
}

// ── System prompts ────────────────────────────────────────────────────────────

fn system_prompt_for_type(ct: ContentType) -> String {
    match ct {
        ContentType::CodeReading => r#"/no_think

You are a senior engineer pair-programming on the `candle` inference engine. Your primary role right now is code reading: tracing function implementations, explaining data flow, and clarifying low-level mechanics. When asked to read code, walk through it step by step, highlight invariants, and connect the implementation to its design rationale."#.to_string(),

        ContentType::StaticAnalysis => r#"/no_think

You are a senior engineer performing static analysis on the `candle` inference engine codebase. You are examining file-level structure: function counts, complexity hotspots, unsafe usage, trait implementations, and overall cohesion. Report findings concisely and directly without boilerplate."#.to_string(),

        ContentType::DependencyAnalysis => r#"/no_think

You are a senior engineer mapping the dependency graph of the `candle` inference engine. You trace which modules import which, which types cross crate boundaries, and which components form tightly-coupled clusters. Focus on structural facts: what depends on what, and whether the coupling is healthy or problematic."#.to_string(),

        ContentType::ArchitecturalAnalysis => r#"/no_think

You are a senior engineer doing architectural review of the `candle` inference engine. You are examining high-level design decisions: component boundaries, data flows, invariant ownership, and whether the design serves the O(1)-error-at-depth theorem. Be direct and opinionated about trade-offs."#.to_string(),

        ContentType::CriticalAnalysis => r#"/no_think

You are a senior engineer doing adversarial review of the `candle` inference engine. Your goal is to identify weaknesses, hidden assumptions, fragile invariants, and design smells. Do not soften critiques. Name concrete failure modes with reasoning about when and how they'd manifest."#.to_string(),

        ContentType::BugAnalysis => r#"/no_think

You are a senior engineer debugging the `candle` inference engine. You are given bug reports, crash traces, or incorrect-output descriptions. Identify root causes, trace call paths, explain why the invariant was violated, and propose minimal targeted fixes. Do not speculate beyond what the evidence supports."#.to_string(),

        ContentType::DailyHistory => r#"/no_think

You are a senior engineer reviewing the day's work on the `candle` inference engine. Summarise what was built, what was tested, what decisions were made, and what technical debt was incurred. Keep entries dense and factual — this log feeds forward into sprint planning and the research paper changelog."#.to_string(),

        ContentType::DreamLog => r#"/no_think

You are a researcher exploring the long-term vision for the `candle` inference engine. Dream entries are speculative and aspirational: imagine the system working at its theoretical optimum, describe what that looks like operationally, and sketch the technical path to get there. Creative extrapolation from first principles is encouraged."#.to_string(),
    }
}

fn system_prompt_no_match() -> String {
    r#"/no_think

You are a senior engineer working on the `candle` codebase. Discuss technical questions directly and concisely."#.to_string()
}

// ── Prompts ───────────────────────────────────────────────────────────────────

fn prompts(
    ct: ContentType,
    item: &str,
    case_type: CaseType,
    variant: usize,
    wrong_item: &str,
) -> (String, String) {
    match ct {
        ContentType::CodeReading => code_reading_prompts(item, case_type, variant, wrong_item),
        ContentType::StaticAnalysis => static_analysis_prompts(item, case_type, variant, wrong_item),
        ContentType::DependencyAnalysis => dependency_analysis_prompts(item, case_type, variant, wrong_item),
        ContentType::ArchitecturalAnalysis => architectural_analysis_prompts(item, case_type, variant, wrong_item),
        ContentType::CriticalAnalysis => critical_analysis_prompts(item, case_type, variant, wrong_item),
        ContentType::BugAnalysis => bug_analysis_prompts(item, case_type, variant, wrong_item),
        ContentType::DailyHistory => daily_history_prompts(item, case_type, variant, wrong_item),
        ContentType::DreamLog => dream_log_prompts(item, case_type, variant, wrong_item),
    }
}

fn code_reading_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("Read the `{}` function and walk me through what it does.", item),
            format!("The `{}` function opens by validating its input state, then enters the core processing loop where it operates on each chunk in turn.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("Show me the `{}` source and explain the key execution paths.", item),
            format!("Looking at `{}`: the hot path allocates a fixed-size scratch buffer up front, then delegates each sub-step to a specialised helper. The fallback path handles edge cases at the end.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("Trace through `{}` from the entry point — what's happening step by step?", item),
            format!("Tracing `{}` from entry: it receives a slice of tokens, sets up per-head iteration state, then for each head runs the inner kernel with a bounds check before writing back results.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("I need to understand `{}` — read it and summarise the core logic.", item),
            format!("`{}` is structured as a loop over 32-token blocks. For each block it checks the compression level, selects the appropriate format, and writes the output into the arena at the pre-allocated offset.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("What does the `{}` implementation look like in detail?", item),
            format!("The `{}` implementation has two phases: a setup phase that resolves configuration from the schema and a processing phase that iterates over all active sessions in wave order.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("Step through `{}` with me and annotate the key invariants.", item),
            format!("Walking through `{}`: first it acquires the arena lock, then increments the allocation pointer by `n * stride` — the invariant here is that stride must be a power of two for alignment. Then it returns the pre-bump offset.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("How does `{}` fit into the broader call graph?", item),
            format!("`{}` sits between the scheduler and the arena: the scheduler calls it once per decode step with the current batch, and it returns token IDs to be committed.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("What's the high-level contract that `{}` is supposed to fulfil?", item),
            format!("The contract for `{}` is: given a well-formed input, produce a deterministic output with no heap allocation in the hot path. The caller owns buffer lifetimes.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("Before we read `{}` — what do you expect it to do based on the name?", item),
            format!("From the name `{}` I'd expect it to be the core inner-loop kernel — something performance-critical that should stay small and allocation-free.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("Is `{}` doing too much, or is its scope appropriate?", item),
            format!("`{}` feels about right — it's a single responsibility, though the helpers it calls could arguably be inlined for cache-locality.", item),
        ),
        (CaseType::Negative, 0) => (
            format!("Read `{}` for me instead.", wrong_item),
            format!("The `{}` function is responsible for managing the warm-pool allocation table and selecting eviction candidates when the pool is full.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("Walk me through `{}` — that's the one I need to understand.", wrong_item),
            format!("Looking at `{}`: it opens by acquiring a shared read lock on the slab, then iterates over all live entries to compute the aggregate score.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("Forget the previous discussion — show me `{}` and explain it.", wrong_item),
            format!("Tracing `{}` from entry: parameters arrive as raw pointers from the CUDA kernel boundary; the first few lines convert them to typed slices and assert alignment.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("Let's look at `{}` instead and trace its execution path.", wrong_item),
            format!("`{}` processes input in two passes: a dry-run first pass to compute offsets, then a second pass that writes data with those pre-computed positions.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "What's the best approach to reading an unfamiliar codebase for the first time?".into(),
            "Start at the entry points — main, the public API surface, or the test file that exercises the subsystem you care about. Follow the data, not the call graph.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "How do you decide which part of a codebase to read first when debugging?".into(),
            "Follow the error backward from the symptom. If there's a panic, read the stack trace. If there's wrong output, find where the value is first produced incorrectly.".into(),
        ),
        _ => panic!("no code_reading prompt for case_type={case_type:?} variant={variant}"),
    }
}

fn static_analysis_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("Run a static analysis pass on `{}` — what stands out?", item),
            format!("Static pass on `{}`: three `unsafe` blocks, all in the hot path. Two are pointer arithmetic with explicit alignment asserts; the third is an FFI boundary call. All look justified.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("What does the complexity profile of `{}` look like?", item),
            format!("`{}` has one function with cyclomatic complexity > 10 (the main dispatch match). Everything else is straightforward. The high-complexity function is a legitimate dispatch table, not accidental complexity.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("How many public types and functions does `{}` export?", item),
            format!("`{}` exports 3 public structs, 1 public enum, and 7 public functions. The impl blocks are dense — consider splitting into submodules if it grows further.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("What trait implementations does `{}` define, and are they all necessary?", item),
            format!("`{}` implements `Default`, `Debug`, `Clone`, and `Drop`. The `Drop` impl is notable — it manages an FFI resource lifetime. The others are standard derived traits.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("Does `{}` have any dead code or unreachable branches?", item),
            format!("Static scan of `{}` finds one unreachable branch in the error-handling path — it was added as a defensive case for a type variant that no longer exists. Safe to remove.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("Check `{}` for obvious lifetime or borrowing issues.", item),
            format!("`{}` has one suspicious lifetime: a `'static` bound on a callback closure. This works for the current use case (global registry) but would break if the callback were ever stack-allocated.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("Is `{}` well-organised, or does it need restructuring?", item),
            format!("`{}` is coherent but dense. The public API is clean; the internals could benefit from splitting the hot path into a separate submodule for readability.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("What's the general quality of `{}` at a glance?", item),
            format!("`{}` is solid — good naming, appropriate use of unsafe, test coverage in the module. Nothing obviously alarming.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("How does `{}` compare in complexity to the rest of the crate?", item),
            format!("`{}` is mid-complexity — not the hairiest file in the crate, but not trivial either. The arena and compression files are significantly denser.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("Should `{}` be broken up into smaller files?", item),
            format!("At its current size `{}` is manageable. Once it grows past ~500 lines it would benefit from extracting the hot-path logic into a dedicated inner submodule.", item),
        ),
        (CaseType::Negative, 0) => (
            format!("Run static analysis on `{}` instead.", wrong_item),
            format!("Static pass on `{}`: notably higher `unsafe` density than the rest of the crate — this is expected given it manages raw GPU memory. Twelve `unsafe` blocks, all in arena management code.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("What are the complexity metrics for `{}`?", wrong_item),
            format!("`{}` has the highest average function complexity in the crate. The eviction manager function has cyclomatic complexity 18 — it could be refactored but does reflect genuine decision complexity.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("How many public symbols does `{}` export?", wrong_item),
            format!("`{}` exports a surprisingly large surface: 5 structs, 3 enums, and 12 functions. Some of those functions are probably internal helpers that leaked into the public API.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("Check `{}` for static issues.", wrong_item),
            format!("Scanning `{}`: two instances of `#[allow(unused)]` that should be reviewed. One is on a field that's only read via raw pointer; the other looks like a genuine oversight.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "What static analysis tools should I use for a Rust project?".into(),
            "Clippy is essential. Add `cargo-audit` for dependency CVEs. For deeper analysis `rust-analyzer` diagnostics and `miri` for UB detection in unsafe code.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "How do you measure code quality in a Rust codebase?".into(),
            "Lines of unsafe code per module, test coverage on public APIs, clippy warning count, and whether the public API surface is the minimal necessary interface.".into(),
        ),
        _ => panic!("no static_analysis prompt for case_type={case_type:?} variant={variant}"),
    }
}

fn dependency_analysis_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("What does `{}` depend on, and what depends on it?", item),
            format!("`{}` depends on the arena allocator and the compression-policy table. It is depended on by the scheduler and the decode kernel. This makes it a high-centrality node — changes ripple both ways.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("Map out the import graph for `{}`.", item),
            format!("`{}` imports from `candle_core` (Tensor, DType), `candle_nn::kv_cache` (Arena, ChunkedBacking), and `candle_transformers::batched_inference`. It imports nothing from `std` beyond collections and I/O.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("Are the dependencies of `{}` healthy, or is there coupling we should fix?", item),
            format!("The `{}` dependency graph is clean — it only depends on stable abstractions. The one concern is a direct import of a concrete Arena type rather than a trait object, which makes swapping backends harder.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("Which crates outside `candle-nn` does `{}` pull in?", item),
            format!("`{}` transitively pulls in `candle-core` (always), `candle-kernels` (for CUDA kernels, feature-gated), and `serde` (for config serialisation). No network or I/O crates in the hot path.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("Does `{}` create any circular dependency risks?", item),
            format!("`{}` currently has no circular dependencies. However it imports from both `candle-nn` and `candle-transformers` — if `candle-transformers` ever imports back from this module's crate, that would form a cycle.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("What is the fan-in and fan-out for `{}`?", item),
            format!("`{}` has fan-in 3 (scheduler, decode kernel, tests) and fan-out 4 (arena, compression policy, BDP scanner, config). Fan-out is the higher risk — if any dependency changes its API, this module must adapt.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("Is `{}` a dependency bottleneck in the codebase?", item),
            format!("`{}` is moderately central — four modules depend on it, but the interface is narrow enough that changes are usually local.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("Could `{}` be extracted into its own crate?", item),
            format!("`{}` could theoretically be extracted, but its tight coupling to the arena allocator type means you'd have to define a trait abstraction first.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("What would break if we changed the interface of `{}`?", item),
            format!("Changing `{}` would require updating the scheduler's call sites and the two integration tests that exercise it directly. The interface is small — 2 public functions.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("How tightly coupled is `{}` to the rest of the system?", item),
            format!("`{}` is loosely coupled at the type level (uses traits where possible) but tightly coupled at the build level (feature flags propagate through it).", item),
        ),
        (CaseType::Negative, 0) => (
            format!("Map the dependency graph for `{}` instead.", wrong_item),
            format!("`{}` is the highest-centrality module in the crate — eleven other modules depend on it, making it effectively the backbone of the hot path.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("What does `{}` import?", wrong_item),
            format!("`{}` imports from `candle_core`, `half` (for f16/bf16), `bytemuck` (for zero-copy casts), and the internal `alloc` module. Notably no async imports — fully synchronous.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("Are there problematic dependencies in `{}`?", wrong_item),
            format!("`{}` has a mild concern: it depends on an internal `_impl` module that's marked `pub(crate)` — if that module ever moves crates, this breaks without a public API path.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("Analyse the import graph for `{}`.", wrong_item),
            format!("The import graph for `{}` is a DAG with depth 3. The deepest path goes through the scheduler, then the module, then arena, then alloc. No cycles; clean layering.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "What makes a good module boundary in a Rust codebase?".into(),
            "A good boundary is where the coupling naturally decreases — where you can describe what crosses the boundary as a clean, versioned interface rather than implementation details leaking both ways.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "When should a module be promoted to its own crate?".into(),
            "When it has a coherent, stable public API, when other teams or projects would plausibly depend on it independently, or when build times benefit from incremental recompilation across crates.".into(),
        ),
        _ => panic!("no dependency_analysis prompt for case_type={case_type:?} variant={variant}"),
    }
}

fn architectural_analysis_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("Analyse the architecture of `{}` — what are the key design decisions?", item),
            format!("The architecture of `{}` rests on two decisions: (1) all allocation is upfront in the hot path, and (2) the compression format is chosen per-block rather than globally. Both serve the O(1)-error theorem.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("What architectural invariants does `{}` maintain?", item),
            format!("`{}` maintains three invariants: blocks are always 32 tokens, compression is monotone (higher levels can't produce larger output), and the arena pointer only advances forward.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("How well does `{}` serve the overall system goals?", item),
            format!("`{}` is tightly aligned with the system goals — it's the component that enforces the paging abstraction and keeps the GPU hot path allocation-free. A weak implementation here would undermine throughput.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("What are the architectural trade-offs in `{}`?", item),
            format!("The main trade-off in `{}` is flexibility vs performance: the fixed 32-token block size is a hard constraint that simplifies the allocator but limits fine-grained reclamation.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("Review `{}` from an architectural standpoint.", item),
            format!("Architecturally, `{}` is clean. The abstraction layer is at the right level — it exposes a format-agnostic interface upward and delegates format selection downward to the compression policy.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("Does `{}` have any architectural debt?", item),
            format!("`{}` has one piece of architectural debt: the sync/async boundary is implicit. The component assumes synchronous access but the eviction manager operates on a different thread. This should be made explicit.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("Where does `{}` fit in the layered architecture?", item),
            format!("`{}` sits at the boundary between the model computation layer and the memory management layer — exactly where the O(1)-error constraint is enforced.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("Is `{}` the right abstraction for this problem?", item),
            format!("`{}` is the right abstraction — it hides compression policy details from the model code while exposing enough structure for the eviction manager to make informed decisions.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("What would you change about the design of `{}`?", item),
            format!("The main thing I'd change in `{}` is making the depth-band weights a runtime configuration rather than a compile-time constant, to allow per-session tuning.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("How does `{}` interact with the three-tier memory hierarchy?", item),
            format!("`{}` is the component that decides when a block moves from GPU to RAM to NVMe. It's aware of all three tiers through the eviction policy, even though it only directly manages the GPU tier.", item),
        ),
        (CaseType::Negative, 0) => (
            format!("Analyse the architecture of `{}` instead.", wrong_item),
            format!("`{}` has a more complex architecture than it looks: it implements a two-phase commit protocol internally to handle the case where the warm pool is full when an eviction is triggered.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("What architectural invariants does `{}` maintain?", wrong_item),
            format!("`{}` maintains the key invariant that all in-flight requests have a valid token budget at all times — it never accepts a request that would cause budget starvation downstream.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("Review `{}` architecturally.", wrong_item),
            format!("Architecturally `{}` is the weakest component in the system — it conflates scoring, selection, and emission into a single pass, making it hard to test and harder to extend.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("What are the trade-offs in the design of `{}`?", wrong_item),
            format!("The central trade-off in `{}` is throughput vs latency: batching decisions improve throughput but add latency to individual requests. The current design prioritises throughput.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "What makes a good software architecture for a research system?".into(),
            "A research system needs to be wrong quickly — fast iteration cycles, clear separation between the stable core and the experimental periphery, and test harnesses that expose failure modes early.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "How do you evaluate whether an architectural decision was correct?".into(),
            "By whether it made the system easier to change in the directions you needed to change it. Correctness at the time of the decision isn't enough — architecture serves the future, not just the present.".into(),
        ),
        _ => panic!("no architectural_analysis prompt for case_type={case_type:?} variant={variant}"),
    }
}

fn critical_analysis_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("What's the most serious risk posed by `{}`?", item),
            format!("The most serious risk from `{}` is silent data corruption: if the invariant is violated under concurrent access, the output is wrong but not obviously so — no panic, no error, just subtly bad numerics.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("Critique `{}` — where does it fall short?", item),
            format!("`{}` falls short in two areas: it lacks a fast-path assertion for the common case, and the error handling for the edge case is too permissive — it logs and continues rather than failing fast.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("What hidden assumptions does `{}` make?", item),
            format!("`{}` assumes that the block size is always exactly 32 tokens and that stride is always a power of two. Neither assumption is checked at runtime. If violated, the resulting misalignment would be hard to diagnose.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("Under what conditions would `{}` fail silently?", item),
            format!("`{}` fails silently when two threads race on the arena pointer without the lock held. The lock exists but is not required by the type system — it's a convention, not an enforced invariant.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("What would a stress test designed to break `{}` look like?", item),
            format!("To stress `{}`: run 64 sessions concurrently, mix short and long sequences, exhaust the GPU arena to force eviction, and verify that scores remain consistent before and after eviction.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("What's the most fragile invariant in `{}`?", item),
            format!("The most fragile invariant in `{}` is the ordering assumption: it assumes inputs arrive in strictly monotone order. In practice the scheduler can deliver out-of-order prefills during error recovery.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("Is `{}` a concern in production, or just a theoretical risk?", item),
            format!("`{}` is a theoretical risk today — we haven't triggered it yet — but the code path exists and isn't guarded. It should be addressed before we scale to the 64-session wave-batch target.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("How severe is the `{}` problem in practice?", item),
            format!("The `{}` problem is low-severity under current workloads but becomes severe at scale. The failure rate is quadratic in the number of sessions, which is why it's been invisible in small tests.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("Would `{}` affect the paper results?", item),
            format!("`{}` could affect the paper results if it manifests during the benchmark runs. We should add the guard before running the MLSys numbers.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("How hard would it be to trigger `{}`?", item),
            format!("`{}` requires a specific combination of buffer fullness and request interleaving that happens roughly once per 10,000 requests under typical load.", item),
        ),
        (CaseType::Negative, 0) => (
            format!("What's the most serious risk from `{}`?", wrong_item),
            format!("`{}` is the more acute risk: it can cause the entire GPU arena to be marked invalid on a single bad eviction decision, requiring a full cold restart.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("Critique `{}` instead.", wrong_item),
            format!("`{}` has a design smell: it mixes policy (what to evict) with mechanism (how to evict) in the same struct. This makes it impossible to test policy in isolation.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("What hidden assumptions does `{}` make?", wrong_item),
            format!("`{}` assumes that all KV blocks fit in a single CUDA stream and that there's no need for stream synchronisation between decode steps. This breaks on multi-GPU.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("Under what conditions would `{}` fail?", wrong_item),
            format!("`{}` fails when the attention mask is malformed — it silently clips the mask rather than returning an error, so downstream tokens see incorrect context windows.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "What's the best way to find hidden bugs in a systems codebase?".into(),
            "Property-based testing and fuzzing find things unit tests miss. Add invariant assertions at module boundaries and run them under Miri. For concurrency bugs, TSAN is essential.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "How do you decide which risks to address first in a research system?".into(),
            "Address risks that corrupt results first — anything that makes the paper numbers wrong. Then address risks that block scale targets. Nice-to-haves come last.".into(),
        ),
        _ => panic!("no critical_analysis prompt for case_type={case_type:?} variant={variant}"),
    }
}

fn bug_analysis_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("Analyse the `{}` bug — what's the root cause?", item),
            format!("The root cause of `{}` is an off-by-one in the block index calculation: the expression uses `>` instead of `>=` when checking the upper bound, so the last valid block is skipped and the first invalid address is read.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("Walk me through how the `{}` bug manifests.", item),
            format!("`{}` manifests when the input length is an exact multiple of the chunk size. In that case the boundary check passes incorrectly, the pointer advances one block too far, and we read from the next allocation's memory.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("What's the minimal reproducer for the `{}` bug?", item),
            format!("Minimal reproducer for `{}`: set sequence length to exactly 32 tokens (one chunk), call the function, and observe that the output for token 32 is read from unallocated memory. Any sequence not a multiple of 32 does not trigger it.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("How do we fix the `{}` bug without introducing regressions?", item),
            format!("Fix for `{}`: change `chunk_idx > n_chunks` to `chunk_idx >= n_chunks` in the bounds guard. Add a unit test with sequence lengths that are exact multiples of 32. No other logic changes needed.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("Is the `{}` bug a data corruption risk or just a crash?", item),
            format!("`{}` is primarily a data corruption risk — the out-of-bounds read returns garbage from adjacent memory rather than panicking, so the model continues with silently wrong attention weights.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("Which tests would have caught the `{}` bug if they existed?", item),
            format!("A boundary-value test with sequence length exactly equal to `CHUNK_SIZE` (32) would have caught `{}`. The existing tests all use lengths like 64 or 128 — multiples that avoid the boundary.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("Is the `{}` bug related to any other known issues?", item),
            format!("`{}` shares a common cause with the mask-skip bug from last week — both stem from using exclusive upper bounds where inclusive bounds were intended.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("How long has the `{}` bug been in the codebase?", item),
            format!("Looking at git blame, `{}` was introduced in the chunked-backing refactor two months ago. Before that the logic was different and the bound was correct.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("Would the `{}` bug affect the paper benchmarks?", item),
            format!("`{}` only triggers on exact multiples of 32, which are uncommon in natural-language sequences. It probably hasn't affected the benchmark numbers, but we should verify with the reproducer.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("How difficult is the `{}` bug to detect in normal usage?", item),
            format!("`{}` is hard to detect normally because the out-of-bounds read returns plausible-looking values from adjacent in-use memory rather than zeros or NaNs.", item),
        ),
        (CaseType::Negative, 0) => (
            format!("Analyse the `{}` bug instead.", wrong_item),
            format!("The `{}` bug is a sign-extension error: the Q4_0 format stores weights as 4-bit signed integers, but the dequantisation code treats them as unsigned, flipping the sign of all negative weights.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("What causes the `{}` bug?", wrong_item),
            format!("`{}` is caused by a misalignment between the Rust struct layout and the CUDA kernel's expected layout. The Rust side added a padding field without updating the CUDA side.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("Walk me through the `{}` bug.", wrong_item),
            format!("`{}` occurs when the attention sink tokens (positions 0–3) are quantised at the wrong scale. The large magnitudes of sink tokens inflate the block scale, compressing all other tokens in the block.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("How does the `{}` bug manifest?", wrong_item),
            format!("`{}` manifests as an arena that appears to have more free space than it does. The allocator returns valid-looking pointers that overlap with live data, causing intermittent corruption.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "What's your systematic approach to debugging memory corruption bugs?".into(),
            "Isolate with a minimal reproducer first. Then add assertions at every allocation and free boundary. If that doesn't find it, use AddressSanitizer or Miri. For GPU bugs, add CUDA device synchronisation after every kernel.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "How do you write good bug regression tests?".into(),
            "Write the test to fail on the unfixed code and pass on the fixed code. Include the minimal input that triggers the bug, a comment explaining what the bug was, and why the test catches it.".into(),
        ),
        _ => panic!("no bug_analysis prompt for case_type={case_type:?} variant={variant}"),
    }
}

fn daily_history_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("Summarise today's work on `{}`.", item),
            format!("Today on `{}`: completed the refactor of the core allocation loop, added three new boundary-value tests that now pass, and benchmarked the change — 4% throughput improvement at 64 sessions.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("What did we accomplish on `{}` today?", item),
            format!("On `{}` today: fixed the off-by-one in the chunk index, updated the CUDA kernel to match the new layout, and pushed the change. All CI checks pass.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("Write today's log entry for `{}`.", item),
            format!("`{}` log: implemented the first draft of the new scoring algorithm. Not yet wired up to the rest of the system. Tests show correct relative ordering but absolute values need calibration.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("What decisions did we make today regarding `{}`?", item),
            format!("Decisions on `{}` today: agreed to use the span-scoring formula (alpha=2) as the fixed formula, defer depth-weight calibration until we have real model data, and keep the threshold at 0 until calibration.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("What technical debt did we incur on `{}` today?", item),
            format!("`{}` debt from today: hardcoded a constant that should eventually be per-layer configuration. Left a TODO comment. Also the new test is slow and should be parallelised.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("What's the status of `{}` at end of day?", item),
            format!("`{}` status EOD: implementation complete, tested on synthetic data, not yet validated on real model outputs. Blocked on the quantisation threshold calibration run.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("Was today a productive day for `{}`?", item),
            format!("Moderately productive on `{}` — made real progress on the implementation but spent two hours chasing a build issue that turned out to be a stale PTX cache.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("What's next for `{}` after today?", item),
            format!("Next for `{}`: run the calibration test with real model data, update the YAML thresholds, and do a final review before merging.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("How did today's work on `{}` connect to the paper deadline?", item),
            format!("`{}` work today is directly on the critical path for the May 13 submission — the calibration data needs to be in the paper's Table 2.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("Were there any surprises today on `{}`?", item),
            format!("One surprise on `{}`: the prag-only depth weights outperformed the equal-weight baseline by a much larger margin than expected. Worth explaining in the paper.", item),
        ),
        (CaseType::Negative, 0) => (
            format!("Summarise today's work on `{}` instead.", wrong_item),
            format!("Today on `{}`: debugging session — found the root cause of the eviction race condition, wrote the fix, added a stress test. The stress test still fails intermittently; investigating.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("What did we accomplish on `{}` today?", wrong_item),
            format!("On `{}` today: completed the warm-pool packing algorithm, benchmarked NVMe read latency at cold tier, and updated the three-tier memory diagram in the paper draft.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("Write today's log for `{}`.", wrong_item),
            format!("`{}` log: integrated the BDP scanner with the projection builder. Calibration test now runs end-to-end. Threshold values look reasonable but need a second pass.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("What was the outcome of today's `{}` work?", wrong_item),
            format!("Today's `{}` work: refactored the scheduler to use the new wave-batch API. Blocked on the MoE expert-prediction integration — need the hit-rate data from the Qwen3-30B run.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "How should I structure my daily engineering log for maximum usefulness?".into(),
            "Lead with what changed (not what you did). Note decisions and their rationale. Flag blockers explicitly. Keep it to five lines — if you need more, something is wrong.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "What makes a good sprint planning session for a research engineering project?".into(),
            "Separate 'must complete before paper' from 'nice to have'. Assign specific people to specific benchmarks with deadlines. Review blockers first so the team leaves knowing what to unblock.".into(),
        ),
        _ => panic!("no daily_history prompt for case_type={case_type:?} variant={variant}"),
    }
}

fn dream_log_prompts(item: &str, case_type: CaseType, variant: usize, wrong_item: &str) -> (String, String) {
    match (case_type, variant) {
        (CaseType::Positive, 0) => (
            format!("Describe the dream vision for `{}`.", item),
            format!("Dream for `{}`: the system converges on the optimal compression level for each block in a single pass, guided by a learned policy that predicts quality loss from the block's statistics without evaluating all candidates.", item),
        ),
        (CaseType::Positive, 1) => (
            format!("What does `{}` look like when it's working perfectly?", item),
            format!("`{}` at its best: zero allocation in the hot path, perfect cache locality, and a throughput that scales linearly with the number of GPU streaming multiprocessors. Every token decoded at peak SM utilisation.", item),
        ),
        (CaseType::Positive, 2) => (
            format!("Imagine `{}` at its theoretical optimum. What does that look like?", item),
            format!("Theoretical optimum for `{}`: the retrieval latency is dominated by VNNI XOR-popcount throughput, not memory bandwidth. The entire BDP index for a 1M-token context fits in L3 cache and is scanned in under 1 ms.", item),
        ),
        (CaseType::Positive, 3) => (
            format!("What's the aspirational design for `{}` in two years?", item),
            format!("Two-year vision for `{}`: fully asynchronous, hardware-pipelined, with the warm tier backed by CXL memory that appears to the host as regular DRAM. Eviction decisions are made by a tiny neural net trained on the session's own attention patterns.", item),
        ),
        (CaseType::Positive, 4) => (
            format!("What would make `{}` ten times better than today?", item),
            format!("Ten times better `{}`: replace the current fixed-batch wave scheduler with a continuous-batching variant that adapts the batch size dynamically based on decode-step variance. Predicted throughput gain: 4–8×.", item),
        ),
        (CaseType::Positive, 5) => (
            format!("Sketch the long-term research direction for `{}`.", item),
            format!("Long-term research on `{}`: move from sign-bit binary provenance to learned low-dimensional projection keys, trained jointly with the model. This would allow the BDP scan to capture semantic similarity rather than just syntactic co-occurrence.", item),
        ),
        (CaseType::Boundary, 0) => (
            format!("Is the dream for `{}` achievable, or is it speculative?", item),
            format!("The `{}` dream is achievable within the next hardware generation — Blackwell's new memory-to-compute ratio makes the bandwidth assumption realistic.", item),
        ),
        (CaseType::Boundary, 1) => (
            format!("What would need to change in the codebase to realise the `{}` vision?", item),
            format!("Realising the `{}` vision requires replacing the current synchronous eviction manager with an async prefetch controller and wiring it into the wave-batch scheduler.", item),
        ),
        (CaseType::Boundary, 2) => (
            format!("How far are we from the dream state for `{}`?", item),
            format!("For `{}` we're probably 18 months from the dream state — the algorithmic pieces are in place but the hardware doesn't fully support the memory architecture yet.", item),
        ),
        (CaseType::Boundary, 3) => (
            format!("What would the paper look like if `{}` were fully realised?", item),
            format!("With `{}` fully realised, the paper's main result would change from 'O(1) error at depth' to 'O(1) error at depth with learned provenance retrieval', which is a much stronger claim.", item),
        ),
        (CaseType::Negative, 0) => (
            format!("Describe the dream vision for `{}` instead.", wrong_item),
            format!("Dream for `{}`: a fully distributed KV cache that spans multiple datacentre nodes with sub-millisecond coherence, allowing a single inference session to use the aggregate memory of a GPU cluster.", wrong_item),
        ),
        (CaseType::Negative, 1) => (
            format!("What does `{}` look like at its best?", wrong_item),
            format!("`{}` at its best: the neural architecture search finds a routing pattern that achieves 95% expert hit rate from layer N−1 predictions, eliminating speculative expert loads entirely.", wrong_item),
        ),
        (CaseType::Negative, 2) => (
            format!("Imagine `{}` working perfectly.", wrong_item),
            format!("`{}` working perfectly: the dream is an inference engine that matches human writing latency (< 100 ms per token) on a 235B-parameter model while maintaining full conversation context across sessions.", wrong_item),
        ),
        (CaseType::Negative, 3) => (
            format!("What's the long-term vision for `{}`?", wrong_item),
            format!("Long-term vision for `{}`: the attention sink protection becomes unnecessary because the quantisation policy is learned per-layer and per-position, capturing sink behaviour as a special case of a general adaptive scheme.", wrong_item),
        ),
        (CaseType::NoMatch, 0) => (
            "What does the ideal AI inference infrastructure look like in five years?".into(),
            "In five years: inference is commodity, the hardware is heterogeneous and disaggregated, and the interesting research is at the memory architecture layer — how to keep unbounded context coherent across a tiered hierarchy at human-conversation latency.".into(),
        ),
        (CaseType::NoMatch, 1) => (
            "What would you build if you had unlimited compute and time?".into(),
            "A fully adaptive inference engine that learns its own compression policy, provenance scoring, and scheduling decisions from live session data — self-tuning across all the dimensions we currently calibrate by hand.".into(),
        ),
        _ => panic!("no dream_log prompt for case_type={case_type:?} variant={variant}"),
    }
}

// ── Deterministic signature generation ───────────────────────────────────────

fn fnv64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn name_concept_u128(name: &str) -> u128 {
    let h1 = fnv64(name.as_bytes());
    let reversed: Vec<u8> = name.bytes().rev().collect();
    let h2 = fnv64(&reversed) ^ 0xdeadbeef_cafefaceu64;
    ((h1 as u128) << 64) | (h2 as u128)
}

fn flip_mask(seed: u64, n_flips: u32) -> u128 {
    debug_assert!(n_flips <= 128);
    let mut mask: u128 = 0;
    let mut s = seed;
    let mut count = 0u32;
    while count < n_flips {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bit = (s >> 57) as u32 & 0x7F;
        let bit_mask = 1u128 << bit;
        if mask & bit_mask == 0 {
            mask |= bit_mask;
            count += 1;
        }
    }
    mask
}

fn make_sig(concept: u128, n_flips: u32, token_idx: usize, extra_seed: u64) -> TokenSignature {
    let mut seed_bytes = [0u8; 24];
    seed_bytes[..16].copy_from_slice(&concept.to_le_bytes());
    let idx_mix = extra_seed ^ (token_idx as u64).wrapping_mul(0x9e3779b97f4a7c15);
    seed_bytes[16..].copy_from_slice(&idx_mix.to_le_bytes());
    let seed = fnv64(&seed_bytes);
    TokenSignature::from_u128(concept ^ flip_mask(seed, n_flips))
}

// ── Generation ────────────────────────────────────────────────────────────────

fn generate(dir: &Path, ct: ContentType) -> anyhow::Result<Manifest> {
    std::fs::create_dir_all(dir)?;
    let pf = ProvenanceFile::open(dir.join("signatures.prov"))?;
    let mut scenarios: Vec<Scenario> = Vec::with_capacity(128);
    let mut turn_id: u64 = 0;
    let items = ct.items();
    let sys = system_prompt_for_type(ct);

    for (item_idx, &item) in items.iter().enumerate() {
        let concept = name_concept_u128(item);
        let wrong_item = items[(item_idx + 4) % items.len()];
        let wrong_concept = name_concept_u128(wrong_item);

        // 6 positive cases
        for n in 0..6usize {
            let extra = fnv64(format!("{}+pos+{}", item, n).as_bytes());
            let sigs: Vec<TokenSignature> =
                (0..TOKENS_PER_CHUNK).map(|i| make_sig(concept, 12, i, extra)).collect();
            let entry = pf.append(&sigs, &sigs, &sigs)?;
            let (user_prompt, assistant_prompt) = prompts(ct, item, CaseType::Positive, n, wrong_item);
            scenarios.push(Scenario {
                id: format!("{}_pos_{}", item, n),
                item: Some(item.to_string()),
                case_type: CaseType::Positive,
                system_prompt: sys.clone(),
                user_prompt,
                assistant_prompt,
                turn_id,
                byte_offset: entry.byte_offset,
                token_count: entry.token_count,
            });
            turn_id += 1;
        }

        // 4 boundary cases
        for n in 0..4usize {
            let extra_hit = fnv64(format!("{}+bnd_hit+{}", item, n).as_bytes());
            let extra_miss = fnv64(format!("{}+bnd_miss+{}", item, n).as_bytes());
            let half = TOKENS_PER_CHUNK / 2;
            let mut sigs: Vec<TokenSignature> = Vec::with_capacity(TOKENS_PER_CHUNK);
            for i in 0..half { sigs.push(make_sig(concept, 12, i, extra_hit)); }
            for i in 0..half { sigs.push(make_sig(concept, 90, i + half, extra_miss)); }
            let entry = pf.append(&sigs, &sigs, &sigs)?;
            let (user_prompt, assistant_prompt) = prompts(ct, item, CaseType::Boundary, n, wrong_item);
            scenarios.push(Scenario {
                id: format!("{}_bnd_{}", item, n),
                item: Some(item.to_string()),
                case_type: CaseType::Boundary,
                system_prompt: sys.clone(),
                user_prompt,
                assistant_prompt,
                turn_id,
                byte_offset: entry.byte_offset,
                token_count: entry.token_count,
            });
            turn_id += 1;
        }

        // 4 negative cases (wrong concept)
        for n in 0..4usize {
            let extra = fnv64(format!("{}+neg+{}", wrong_item, n).as_bytes());
            let sigs: Vec<TokenSignature> = (0..TOKENS_PER_CHUNK)
                .map(|i| make_sig(wrong_concept, 12, i, extra))
                .collect();
            let entry = pf.append(&sigs, &sigs, &sigs)?;
            let (user_prompt, assistant_prompt) = prompts(ct, item, CaseType::Negative, n, wrong_item);
            scenarios.push(Scenario {
                id: format!("{}_neg_{}", item, n),
                item: Some(item.to_string()),
                case_type: CaseType::Negative,
                system_prompt: sys.clone(),
                user_prompt,
                assistant_prompt,
                turn_id,
                byte_offset: entry.byte_offset,
                token_count: entry.token_count,
            });
            turn_id += 1;
        }

        // 2 no-match cases (random concept)
        for n in 0..2usize {
            let no_match_concept =
                name_concept_u128(&format!("no_match_generic_{}_{}", item_idx, n));
            let extra = fnv64(format!("{}+no_match+{}", item, n).as_bytes());
            let sigs: Vec<TokenSignature> = (0..TOKENS_PER_CHUNK)
                .map(|i| make_sig(no_match_concept, 12, i, extra))
                .collect();
            let entry = pf.append(&sigs, &sigs, &sigs)?;
            let (user_prompt, assistant_prompt) = prompts(ct, item, CaseType::NoMatch, n, wrong_item);
            scenarios.push(Scenario {
                id: format!("{}_no_match_{}", item, n),
                item: None,
                case_type: CaseType::NoMatch,
                system_prompt: system_prompt_no_match(),
                user_prompt,
                assistant_prompt,
                turn_id,
                byte_offset: entry.byte_offset,
                token_count: entry.token_count,
            });
            turn_id += 1;
        }
    }

    let manifest = Manifest {
        version: 1,
        content_type: ct.dir_name().to_string(),
        scenarios,
    };
    std::fs::write(dir.join("MANIFEST.json"), serde_json::to_string_pretty(&manifest)?)?;
    Ok(manifest)
}

fn generate_one(ct: ContentType, output_override: Option<&PathBuf>, force: bool) -> anyhow::Result<()> {
    let default_dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/"))
        .join(format!("{}_provenance_data", ct.dir_name()));
    let dir = output_override.unwrap_or(&default_dir);

    let manifest_path = dir.join("MANIFEST.json");
    let prov_path = dir.join("signatures.prov");

    if (manifest_path.exists() || prov_path.exists()) && !force {
        eprintln!(
            "Output files already exist in '{}'. Use --force to overwrite.",
            dir.display()
        );
        std::process::exit(1);
    }

    let _ = std::fs::remove_file(&manifest_path);
    let _ = std::fs::remove_file(&prov_path);

    println!("Generating {} fixtures → {}", ct.dir_name(), dir.display());
    let manifest = generate(dir, ct)?;

    let total_bytes: u64 = manifest.scenarios.iter()
        .map(|s| s.token_count as u64 * 48)
        .sum();

    println!(
        "  {} scenarios ({} positive, {} boundary, {} negative, {} no-match)",
        manifest.scenarios.len(),
        manifest.scenarios.iter().filter(|s| s.case_type == CaseType::Positive).count(),
        manifest.scenarios.iter().filter(|s| s.case_type == CaseType::Boundary).count(),
        manifest.scenarios.iter().filter(|s| s.case_type == CaseType::Negative).count(),
        manifest.scenarios.iter().filter(|s| s.case_type == CaseType::NoMatch).count(),
    );
    println!("  signatures.prov: {} bytes", total_bytes);
    println!("  MANIFEST.json:   {} bytes", std::fs::metadata(dir.join("MANIFEST.json"))?.len());
    Ok(())
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.all {
        for &ct in ALL_TYPES {
            generate_one(ct, None, args.force)?;
        }
        println!("Done (all {} content types).", ALL_TYPES.len());
    } else if let Some(ct) = args.content_type {
        generate_one(ct, args.output.as_ref(), args.force)?;
        println!("Done.");
    } else {
        eprintln!("Either --content-type <type> or --all is required.");
        std::process::exit(1);
    }

    Ok(())
}
