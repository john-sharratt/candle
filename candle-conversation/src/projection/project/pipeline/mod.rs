//! Turns a schema and a resolver into the context a projection target emits.
//!
//! The pipeline masks the layers a target can see, scores and gates each group's
//! turns, and makes an unbounded selection. The budget is then distributed across
//! layers and groups in one flexbox pass, and a bounded selection trims each group
//! to its share. A dialogue's window is filled by score rather than flexed, and a
//! working-set group is emitted verbatim in its own order, so the prefix stays
//! stable across rebuilds. `run` is pure given a valid schema and a working
//! resolver.
//!
//! - `sink`: the pipeline with a diagnostic sink for score-density selection.

use super::*;

// ── Main entry-point ──────────────────────────────────────────────────────────

/// Run the full projection pipeline.
///
/// Called from [`super::Builder::project`]. See module-level docs for the
/// 12-step pipeline. Pure given a valid schema and a working resolver —
/// never errors.
pub fn run<R: ContentResolver>(
    schema: &Schema,
    target: ProjectionTarget,
    resolver: &R,
    mode: ProjectionMode,
    selection: &SelectionState,
) -> Projection {
    run_with_sink(
        schema,
        target,
        resolver,
        mode,
        selection,
        &PriorBelief::default(),
        None,
        &mut |_| {},
    )
}

mod files_anchor;
mod files_splice;
mod sink;

pub(crate) use files_anchor::{files_position, place_files};
pub(crate) use files_splice::files_splice;
pub use sink::*;
