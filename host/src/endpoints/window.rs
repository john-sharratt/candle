//! `?span_s=N` — the window of slot rows a page draws.
//!
//! The engine's rings hold hours of rows. A page that draws the last twenty
//! minutes, polling every four seconds, refetched all of them every time:
//! `/v1/telemetry` was 2.9 MB a member and `/v1/phases` 0.8 MB on BlackWorld
//! (2026-10-29), and the gateway carried each twice over — every member's body
//! and the fused cluster copy. Asked for its window, an answer carries only it.

use serde::Deserialize;

use super::slots::now_ms;

/// The query: the last `span_s` seconds of rows, or every row without it.
#[derive(Debug, Default, Deserialize)]
pub struct Window {
    span_s: Option<u64>,
}

impl Window {
    /// The first slot start the window keeps, as of `now_ms`; `0` keeps all.
    pub fn since(&self, now_ms: u64) -> u64 {
        self.span_s
            .map_or(0, |span| now_ms.saturating_sub(span.saturating_mul(1000)))
    }

    /// [`Self::since`] as of now.
    pub fn since_now(&self) -> u64 {
        self.since(now_ms())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span counts back from now; none keeps every row; a span longer than
    /// the clock has run keeps every row too.
    #[test]
    fn a_span_counts_back_from_now() {
        let w = |span_s| Window { span_s };
        assert_eq!(w(Some(1_200)).since(5_000_000), 3_800_000);
        assert_eq!(w(None).since(5_000_000), 0);
        assert_eq!(w(Some(u64::MAX)).since(5_000_000), 0);
    }
}
