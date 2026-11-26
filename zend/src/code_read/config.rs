//! The limits and fixed names of a code-reading ingest pass.
//!
//! The resubmission rounds decide how long a file refused for want of ground
//! keeps coming back before a refusal counts as a failure.

use super::*;

/// Metadata key holding a conversation's workspace-relative path.
pub(crate) const PATH_KEY: &str = Substrate::PATH_KEY;

/// Maximum tolerated per-file seal failures in a single ingestion pass
/// before the whole refresh aborts. A single failure can happen for
/// legitimate reasons (scheduler hiccup, transient resource pressure); a
/// cascade signals something systemic.
pub const MAX_DECODE_FAILURES: usize = 16;

/// Rounds a pass gives the files whose seal found no ground — a refused tier,
/// a full span, the KV budget — after the round that refused them. The file
/// goes back through admission once the rest of the round has finished and
/// given its ground back. Only in the last of them does a refusal count as a
/// failure.
pub(super) const GROUND_RESUBMITS: usize = 2;

/// Key this pass reports completeness under (see [`crate::ingest_report`]).
pub const PASS_NAME: &str = "code_read";

/// Number of files sealed concurrently by the worker pool. A file's seal is
/// its members' prefills, one after another, each on a scratch slot freed
/// when it seals; this many files' prefills coalesce into the same forwards
/// every other concurrent conversation's do. Admission prices each joining
/// prefill against the expert zone (`WaveRate::try_join_trading`), so the
/// pool offers width and the zone decides how much of it runs.
pub const CODE_READ_PARALLELISM: usize = 16;
