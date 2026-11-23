//! The card under the battery's load (`docs/unbounded_agents.md` §9.6.9): the
//! answers in flight, the KV the span holds for them, the tokens that KV
//! stands for and their size at BF16, and the expert zone beside it — a mean
//! and a peak each, over the samples taken while answers were in flight.
//!
//! Read from each telemetry sample's newest wave (`telemetry/waves/vram`, MiB
//! by tenant) and newest format census (`telemetry/mix/tok`, the tokens
//! resident in each format, which add up to the tokens resident). The tokens'
//! BF16 size is their count times the BF16 bytes a token takes, read from the
//! substrate's K/V census (`census_start.json`).

use serde_json::Value;

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * MIB;

/// One sample's reading of the card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    pub in_flight: f64,
    pub kv_bytes: f64,
    pub tokens: f64,
    pub expert_bytes: f64,
}

/// The reading `sample` carries, or `None` when nothing was in flight or a
/// read failed.
pub fn reading(sample: &Value) -> Option<Reading> {
    let in_flight = sample.get("in_flight")?.as_f64()?;
    if in_flight <= 0.0 {
        return None;
    }
    let vram = sample.pointer("/telemetry/waves/vram")?;
    let tokens = sample
        .pointer("/telemetry/mix/tok")?
        .as_object()?
        .values()
        .filter_map(Value::as_f64)
        .sum();
    Some(Reading {
        in_flight,
        kv_bytes: vram.get("kv")?.as_f64()? * MIB,
        tokens,
        expert_bytes: vram.get("experts")?.as_f64()? * MIB,
    })
}

/// One row of the load table: its label, mean and peak, already in the
/// table's unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub label: &'static str,
    pub mean: f64,
    pub peak: f64,
}

/// The load table over `samples`, with `bf16_per_token` the BF16 bytes one
/// token's K/V takes. Empty when no sample had answers in flight.
pub fn table(samples: &[Value], bf16_per_token: f64) -> Vec<Row> {
    let readings: Vec<Reading> = samples.iter().filter_map(reading).collect();
    if readings.is_empty() {
        return Vec::new();
    }
    let row = |label, f: &dyn Fn(&Reading) -> f64| {
        let xs: Vec<f64> = readings.iter().map(f).collect();
        Row {
            label,
            mean: xs.iter().sum::<f64>() / xs.len() as f64,
            peak: xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    };
    vec![
        row("Sessions in flight", &|r| r.in_flight),
        row("KV bytes resident (GiB)", &|r| r.kv_bytes / GIB),
        row("Tokens resident", &|r| r.tokens),
        row("The same tokens at BF16 (GiB)", &|r| {
            r.tokens * bf16_per_token / GIB
        }),
        row("Effective compression", &|r| {
            if r.kv_bytes > 0.0 {
                r.tokens * bf16_per_token / r.kv_bytes
            } else {
                0.0
            }
        }),
        row("Expert zone (GiB)", &|r| r.expert_bytes / GIB),
    ]
}

/// The BF16 bytes per token a K/V census (`GET /v1/kv/census`) records.
pub fn bf16_per_token(census: &Value) -> Option<f64> {
    let bf16 = census.get("bf16_bytes_estimate")?.as_f64()?;
    let tokens = census.get("tokens")?.as_f64()?;
    (tokens > 0.0).then(|| bf16 / tokens)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn sample(in_flight: u64, kv_mib: f64, tok: Value, experts_mib: f64) -> Value {
        json!({
            "t_ms": 0,
            "in_flight": in_flight,
            "telemetry": {
                "waves": {"vram": {"kv": kv_mib, "experts": experts_mib}},
                "mix": {"at": 0, "tok": tok},
            },
            "memory": {},
        })
    }

    #[test]
    fn a_reading_sums_the_formats_and_skips_idle_and_failed_samples() {
        assert_eq!(
            reading(&sample(
                8,
                1024.0,
                json!({"Q4_0": 3000.0, "other": 1000.0}),
                2048.0
            )),
            Some(Reading {
                in_flight: 8.0,
                kv_bytes: GIB,
                tokens: 4000.0,
                expert_bytes: 2.0 * GIB,
            })
        );
        assert_eq!(reading(&sample(0, 1024.0, json!({"Q4_0": 1.0}), 1.0)), None);
        assert_eq!(
            reading(&json!({"in_flight": 4, "telemetry": {"error": "timed out"}})),
            None
        );
    }

    #[test]
    fn the_table_takes_mean_and_peak_over_the_loaded_samples() {
        let samples = vec![
            sample(4, 512.0, json!({"Q4_0": 1_048_576.0}), 40_960.0),
            sample(0, 9_999.0, json!({"Q4_0": 9.0}), 1.0),
            sample(8, 1536.0, json!({"Q4_0": 2_097_152.0}), 51_200.0),
        ];
        let t = table(&samples, 2048.0);
        let got: Vec<(&str, f64, f64)> = t.iter().map(|r| (r.label, r.mean, r.peak)).collect();
        assert_eq!(
            got,
            vec![
                ("Sessions in flight", 6.0, 8.0),
                ("KV bytes resident (GiB)", 1.0, 1.5),
                ("Tokens resident", 1_572_864.0, 2_097_152.0),
                ("The same tokens at BF16 (GiB)", 3.0, 4.0),
                ("Effective compression", (4.0 + 8.0 / 3.0) / 2.0, 4.0),
                ("Expert zone (GiB)", 45.0, 50.0),
            ]
        );
        assert!(table(&samples[1..2], 2048.0).is_empty());
    }

    #[test]
    fn a_census_gives_its_bf16_bytes_per_token() {
        assert_eq!(
            bf16_per_token(&json!({"tokens": 4, "bf16_bytes_estimate": 8192})),
            Some(2048.0)
        );
        assert_eq!(
            bf16_per_token(&json!({"tokens": 0, "bf16_bytes_estimate": 1})),
            None
        );
    }
}
