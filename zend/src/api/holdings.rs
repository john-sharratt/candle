//! `GET /v1/holdings` — who is holding the span right now, and **why**.
//!
//! The companion to [`super::memory`], which answers *how much* of each tenant
//! is resident. That question has never been the hard one. The hard one is why
//! a byte is still there, and it is not answerable from any total: a weight
//! zone pinned 600 MiB above its hold for four hours, with eviction running on
//! every pass and freeing nothing, reads identically whether the K/V is held by
//! conversations mid-reply, by turns merely queued behind ninety others, or by
//! views nobody finalised — and those want completely different fixes.
//!
//! So this serves the per-slot census: every live slot, what it holds in block
//! tables and in substrate residences, how much of that eviction could actually
//! take at this instant, and the list of reasons it is being held. The
//! by-reason rollup answers "what would I have to change to get this ground
//! back" directly.
//!
//! Read from the scheduler's published snapshot with no engine lock. The census
//! is taken when the engine is short of ground — which is exactly when it is
//! worth having — so `census` is `null` on a run that has never been tight.

use axum::Json;
use candle_conversation::holdings::Holdings;
use serde::Serialize;

/// Response body for `GET /v1/holdings`.
#[derive(Serialize)]
pub struct HoldingsDump {
    /// The latest per-slot census. `null` until the engine has first been short
    /// of ground — a healthy run takes none, and that absence is itself the
    /// answer to "is anything being held that shouldn't be".
    pub census: Option<Holdings>,
    /// The census as one line, for a human reading the response directly.
    pub summary: Option<String>,
}

pub async fn dump() -> Json<HoldingsDump> {
    let census = candle_conversation::holdings::latest();
    let summary = census.as_ref().map(|c| c.summary());
    Json(HoldingsDump { census, summary })
}
