//! The limits and fixed names of a code-reading ingest pass.
//!
//! Ingest runs eight workers, a width set by what the card can hold rather than
//! by throughput: each worker carries a long context, and past that width the
//! resident K/V pushes the expert weights out of the span. Each reply makes one
//! `file_read` call, so each tool round of a reading is one page that the
//! projection can place or drop at that page's cost. The resubmission rounds
//! decide how long a file refused for want of ground keeps coming back before a
//! refusal counts as a failure.

use super::*;

/// Metadata key holding a file conversation's workspace-relative path.
pub(crate) const PATH_KEY: &str = Substrate::PATH_KEY;

/// Maximum tolerated per-file summary decode failures in a single
/// ingestion pass before the whole refresh aborts.  A single
/// failure can happen for legitimate reasons (scheduler hiccup,
/// transient resource pressure); a cascade signals something
/// systemic.
pub const MAX_DECODE_FAILURES: usize = 16;

/// Rounds a pass gives the files whose attempt found no ground — a refused tier,
/// a full span, the KV budget — after the round that refused them. Each such
/// attempt is dropped whole (its partial tombstoned, its slot freed) and the file
/// goes back through admission once the rest of the round has finished and given
/// its ground back. Only in the last of them does a refusal count as a failure.
pub(super) const GROUND_RESUBMITS: usize = 2;

/// Key this pass reports completeness under (see [`crate::ingest_report`]).
pub const PASS_NAME: &str = "code_read";

/// The one tool a file's hidden conversation may call — forced into the
/// catalog via [`FORCE_TOOL_SELECTOR`] so the model reads coherently and
/// never wanders into an unrelated tool.
pub(crate) const FILE_READ_TOOL: &str = "file_read";

/// Decode budget per turn in a file's hidden conversation: room for the
/// `<tool_call>` or the file's description. Thinking is off, so none of it goes
/// to a reasoning block.
pub(super) const FILE_TURN_MAX_TOKENS: usize = 1536;

/// `file_read` calls a reading's reply may make: one. Each reply reads one page
/// — one range of whole parts, at most [`PAGE_LINES`] lines — so each tool round
/// of the reading is one page, and the projection places or drops it at that
/// page's cost (`docs/hit_driven_projection.md` §8.2). A reply of several reads
/// would be a round of several pages, placed or dropped together.
pub const READ_CALLS_PER_REPLY: usize = 1;

/// Number of files ingested concurrently by the worker pool. Each worker owns
/// one file's hidden conversation end to end — mint it, run it to a final
/// answer, tag it, free it — so this is the concurrency knob: the number of
/// real conversations feeding the engine's shared multi-session batching at
/// once (their prefills and decodes coalesce into the same forwards every
/// other concurrent conversation's do).
///
/// **Invariant:** each worker holds exactly ONE conversation slot, so this
/// must stay under the model's sequence-slot capacity with headroom for the
/// non-ingest slots the engine also needs concurrently: the live dialogue
/// session, the async summariser's compression passes, etc.
///
/// **8, and the ceiling here is VRAM rather than throughput.** Each worker is
/// serialised on its own file's whole turn — a prefill that attends over the
/// inherited priming chain (~32k KV, ~9.4 s) and then a decode — so the pool
/// width, not the wave, is what bounds the file phase.
///
/// Measured on the 72 GB card over the same corpus:
///
///    8   0.61 files/min   48 files in 79 min
///   12   0.58 files/min   stable for three hours
///   16   0.41 files/min   16 files in 39 min
///   32   1.66 files/min   K/V ran 13.9 GB against a 2.5 GB budget, the weight
///                         zone slid to 30.6 GB against a 28.8 GB floor, and the
///                         daemon died after ~30 minutes — no panic, no poison,
///                         the log simply stops
///
/// A file conversation is not a directory conversation: it carries 876k KV per
/// decode forward against a directory's 25k, so a width that is comfortable for
/// `repo_map` exhausts the card here.
///
/// **The curve is not monotonic, and 16 sits past its turn.** Each worker holds a
/// ~77k-token context, so at 16 the resident K/V pushes the expert weights out of
/// the span: the hit rate falls to ~50% mid-sweep, an 8k prefill forward costs
/// 5–7 s instead of ~1 s, and the wave loop spends ~10 s per pass in KV relief
/// that returns `reclaimed_mib=0` because nothing in the span is actually free.
/// Narrowing the pool buys back the weight zone, which is worth more than the
/// extra concurrency — 8 is 1.5× the rate of 16 on the same corpus. Below 8 the
/// card is no longer the constraint and the rate falls off with the width.
///
/// Re-measured at 32 with the K/V now a small tenant (≈1.2 GB at 8): the decode
/// width did grow (5–23 sequences against 1–2), but the decode rate barely moved
/// — 124 tok/s against 107 — because each step got dearer: the routed experts
/// went from 0.3 to 1.2 ms a layer at a 90–94% hit rate, as the extra contexts
/// and every 8k prefill's transient tier pushed the zone off its working set.
/// Prefill fell from 1,725 to 1,154 tok/s, with the hit rate at 56% inside a
/// 17-sequence prefill. Width is not the lever here; the expert zone is.
///
/// **16 since admission prices each joining turn against the zone.** Those
/// widths were run while a turn that joined a spent forward was either judged
/// on the floor alone or refused on any byte, so the pool's width reached the
/// card unpriced or not at all. A joining turn is now admitted only while the
/// wave steps more decodes a second with it than without it at the residency
/// it leaves (`WaveRate::try_join_trading`), so the pool offers width and the
/// zone decides how much of it runs. At 8 on Qwen3.8-Flash-Next (BlackWorld,
/// 2026-10-27, build 3897) a file's chain — its reading's summary, its
/// calibration draft — ran one row at a time: 3.7 rows per decode forward,
/// 90 tok/s decode, 161 files an hour.
pub const CODE_READ_PARALLELISM: usize = 16;

/// What a reading that answered nothing fails with.
pub(super) const READ_NOTHING: &str = "the reading ended without an answer";

/// What a reading whose `file_read` found no file fails with.
pub(super) const READ_ELSEWHERE: &str = "the reading's file_read found no file";
