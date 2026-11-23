//! The load the battery puts on the card, sampled while it runs
//! (`docs/unbounded_agents.md` §9.6.9, §10.2): every [`EVERY`], the daemon's
//! `/v1/telemetry` (the KV format mix, migration, pressure) and `/v1/memory`
//! (VRAM by tenant), beside the answers in flight, one JSON line per sample on
//! one time axis.
//!
//! `/v1/telemetry` answers with its whole rolling window — every wave, mix and
//! migration point of the last hours, 2.5 MB a read on BlackWorld — so a
//! sample keeps only the newest point of each series ([`current`]): the
//! reading as of that sample. Kept whole, 5-second samples of the window grew
//! one launch's file to 12 GB of the same points repeated.

use std::fs::File;
use std::io::{BufRead, Write};
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

/// The reading `telemetry` holds now: every series in it — an array of points,
/// oldest first, at any depth — replaced by its newest point. An empty series
/// is `null`.
pub fn current(telemetry: Value) -> Value {
    match telemetry {
        Value::Array(points) => points.into_iter().last().map_or(Value::Null, current),
        Value::Object(fields) => {
            Value::Object(fields.into_iter().map(|(k, v)| (k, current(v))).collect())
        }
        v => v,
    }
}

/// Every sample `lines` holds, each one's telemetry reduced to its
/// [`current`] reading, and the numbers (from 1) of the lines that are not a
/// whole sample. A launch stopped mid-write leaves its last line cut short,
/// and the next launch appends to it; that one sample is lost, and named.
pub fn read(lines: impl BufRead) -> anyhow::Result<(Vec<Value>, Vec<usize>)> {
    let mut samples = Vec::new();
    let mut cut = Vec::new();
    for (i, line) in lines.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(mut sample) => {
                if let Some(t) = sample.get_mut("telemetry") {
                    *t = current(t.take());
                }
                samples.push(sample);
            }
            Err(_) => cut.push(i + 1),
        }
    }
    Ok((samples, cut))
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
            current(or_error(telemetry)),
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

    #[test]
    fn reading_reduces_each_sample_and_names_a_cut_line() {
        let file = concat!(
            r#"{"t_ms":1,"in_flight":2,"telemetry":{"waves":[{"at":1},{"at":2}]},"memory":{}}"#,
            "\n",
            r#"{"t_ms":6,"in_flight":2,"telemetry":{"wav"#,
            r#"{"t_ms":9,"in_flight":0,"telemetry":{"waves":[]},"memory":{}}"#,
            "\n\n",
            r#"{"t_ms":14,"in_flight":1,"telemetry":{"waves":[{"at":13}]},"memory":{}}"#,
            "\n",
        );
        let (samples, cut) = read(file.as_bytes()).unwrap();
        assert_eq!(
            samples,
            vec![
                json!({"t_ms": 1, "in_flight": 2, "telemetry": {"waves": {"at": 2}}, "memory": {}}),
                json!({"t_ms": 14, "in_flight": 1, "telemetry": {"waves": {"at": 13}}, "memory": {}}),
            ]
        );
        assert_eq!(cut, vec![2]);
    }

    #[test]
    fn a_sample_keeps_the_newest_point_of_every_series() {
        let window = json!({
            "sample_period_s": 3,
            "waves": [{"at": 1, "fwd": 2.0}, {"at": 4, "fwd": 5.0, "vram": {"free": 9.0}}],
            "mix": [],
            "series": {"t": [1, 4], "kv_mib": [10, 11], "max_batch": 8},
        });
        assert_eq!(
            current(window),
            json!({
                "sample_period_s": 3,
                "waves": {"at": 4, "fwd": 5.0, "vram": {"free": 9.0}},
                "mix": null,
                "series": {"t": 4, "kv_mib": 11, "max_batch": 8},
            })
        );
    }
}
