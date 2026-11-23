//! The load the battery puts on the card, sampled while it runs
//! (`docs/unbounded_agents.md` §9.6.9, §10.2): every [`EVERY`], the daemon's
//! `/v1/telemetry` (the KV format mix, migration, pressure) and `/v1/memory`
//! (VRAM by tenant), beside the answers in flight, one JSON line per sample on
//! one time axis.

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::client::Zend;

/// How often the load is sampled.
pub const EVERY: Duration = Duration::from_secs(5);

/// One sample's line.
pub fn line(t_ms: u128, in_flight: usize, telemetry: Value, memory: Value) -> Value {
    json!({"t_ms": t_ms, "in_flight": in_flight, "telemetry": telemetry, "memory": memory})
}

/// Sample until `done` is set, writing each line to `out`. A failed read is
/// recorded as its error, never skipped: a gap in the series is a gap the
/// figure must show.
pub async fn sample(
    zend: Arc<Zend>,
    in_flight: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
    mut out: File,
) {
    let mut tick = tokio::time::interval(EVERY);
    while !done.load(Ordering::Acquire) {
        tick.tick().await;
        let (telemetry, memory) =
            tokio::join!(zend.sample("/v1/telemetry"), zend.sample("/v1/memory"));
        let or_error =
            |r: anyhow::Result<Value>| r.unwrap_or_else(|e| json!({"error": format!("{e:#}")}));
        let t_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let l = line(
            t_ms,
            in_flight.load(Ordering::Acquire),
            or_error(telemetry),
            or_error(memory),
        );
        if writeln!(out, "{l}").and_then(|_| out.flush()).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_puts_both_reads_on_one_time_axis() {
        assert_eq!(
            line(1_700, 16, json!({"mix": {}}), json!({"report": {}})),
            json!({"t_ms": 1_700, "in_flight": 16, "telemetry": {"mix": {}}, "memory": {"report": {}}})
        );
    }
}
