//! `GET /v1/kv/census?sample=N&seed=S` — the K/V census of this member's
//! substrate (`docs/unbounded_agents.md` §9.6.9): every live chunk's stored
//! bytes and tokens by its representative format, exact from memory, and the
//! per-sub-band format mix and BF16 equivalent from a sample of `N` chunk
//! heads read from disk (`substrate::kv_census`).

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use candle_conversation::substrate::kv_census::KvCensus;
use serde::Deserialize;

use crate::session::ZendSession;

/// The sample a census reads when the request names none.
pub const DEFAULT_SAMPLE: usize = 4_096;
/// The most heads one census reads.
pub const MAX_SAMPLE: usize = 65_536;
/// The seed a census samples under when the request names none.
pub const DEFAULT_SEED: u64 = 0x2026_1123;

#[derive(Debug, Default, Deserialize)]
pub struct CensusQuery {
    #[serde(default)]
    pub sample: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl CensusQuery {
    /// The sample and seed asked for, the sample capped at [`MAX_SAMPLE`].
    pub fn resolved(&self) -> (usize, u64) {
        (
            self.sample.unwrap_or(DEFAULT_SAMPLE).min(MAX_SAMPLE),
            self.seed.unwrap_or(DEFAULT_SEED),
        )
    }
}

pub async fn get(
    State(session): State<Arc<ZendSession>>,
    Query(q): Query<CensusQuery>,
) -> Result<Json<KvCensus>, StatusCode> {
    let (sample, seed) = q.resolved();
    tokio::task::spawn_blocking(move || session.kv_census(sample, seed))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map(Json)
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_names_its_sample_or_takes_the_default() {
        assert_eq!(CensusQuery::default().resolved(), (4_096, 0x2026_1123));
        let q = CensusQuery {
            sample: Some(1_000_000),
            seed: Some(3),
        };
        assert_eq!(q.resolved(), (65_536, 3), "the sample is capped");
    }
}
