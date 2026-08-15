//! `GET /v1/repo_map` — is the repo map complete, and if not, what is missing.
//!
//! A failed directory ingest keeps its prior generation live, so a partial map
//! is indistinguishable from a whole one at query time. This endpoint makes the
//! difference explicit: `incomplete` is the single flag to branch on, and each
//! failure carries the directory plus the full error chain that produced it.

use axum::Json;
use serde::Serialize;

use std::collections::BTreeMap;

use crate::ingest_report::{self, IngestReport};

#[derive(Serialize)]
pub struct Completeness {
    /// `false` until any pass has run — distinct from "ran and was clean".
    pub has_run: bool,
    /// The one predicate callers should branch on: any pass left a stale map.
    pub incomplete: bool,
    /// Per-pass detail, keyed by pass name (`repo_map`, `code_read`, …).
    pub passes: BTreeMap<String, IngestReport>,
}

pub async fn completeness() -> Json<Completeness> {
    let passes = ingest_report::latest();
    Json(Completeness {
        has_run: !passes.is_empty(),
        incomplete: passes.values().any(|r| r.is_incomplete()),
        passes,
    })
}
