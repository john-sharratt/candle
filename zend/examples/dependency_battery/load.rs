//! The card under the battery's load (`docs/unbounded_agents.md` §9.6.9): the
//! answers in flight, the KV the span holds for them, the tokens that KV
//! stands for and their size at BF16, and the expert zone beside it — a mean
//! and a peak each, over the samples taken while answers were in flight; and
//! the same samples banded by how many answers were in flight, with the share
//! of the resident tokens each format holds.
//!
//! Read from each telemetry sample's newest wave (`telemetry/waves/vram`, MiB
//! by tenant) and newest format census (`telemetry/mix/tok`, the tokens
//! resident in each format, which add up to the tokens resident). The tokens'
//! BF16 size is their count times the BF16 bytes a token takes across the
//! model's K/V layers, from the substrate's K/V census (`census_start.json`),
//! which counts every layer's tokens where the format census counts each
//! token once.
//!
//! The KV figure is the KV arenas the span holds, which is at least the
//! payload in them, so the compression read against it is a floor on the
//! payload's.

use std::collections::BTreeMap;

use serde_json::Value;

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * MIB;

/// Answers in flight per band of the load breakdown.
pub const BAND_WIDTH: f64 = 4.0;

/// One sample's reading of the card.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub in_flight: f64,
    pub kv_bytes: f64,
    pub tokens: f64,
    pub expert_bytes: f64,
    /// The tokens resident in each format.
    pub formats: BTreeMap<String, f64>,
}

/// The reading `sample` carries, or `None` when nothing was in flight or a
/// read failed.
pub fn reading(sample: &Value) -> Option<Reading> {
    let in_flight = sample.get("in_flight")?.as_f64()?;
    if in_flight <= 0.0 {
        return None;
    }
    let vram = sample.pointer("/telemetry/waves/vram")?;
    let formats: BTreeMap<String, f64> = sample
        .pointer("/telemetry/mix/tok")?
        .as_object()?
        .iter()
        .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
        .collect();
    Some(Reading {
        in_flight,
        kv_bytes: vram.get("kv")?.as_f64()? * MIB,
        tokens: formats.values().sum(),
        expert_bytes: vram.get("experts")?.as_f64()? * MIB,
        formats,
    })
}

/// The BF16 size of `r`'s tokens over the KV the span holds for them.
fn compression(r: &Reading, bf16_per_token: f64) -> f64 {
    if r.kv_bytes > 0.0 {
        r.tokens * bf16_per_token / r.kv_bytes
    } else {
        0.0
    }
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
/// token's K/V takes over every K/V layer. Empty when no sample had answers
/// in flight.
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
        row("KV arenas held (GiB)", &|r| r.kv_bytes / GIB),
        row("Tokens resident", &|r| r.tokens),
        row("The same tokens at BF16 (GiB)", &|r| {
            r.tokens * bf16_per_token / GIB
        }),
        row("Effective compression (BF16 / arenas held)", &|r| {
            compression(r, bf16_per_token)
        }),
        row("Expert zone (GiB)", &|r| r.expert_bytes / GIB),
    ]
}

/// The samples with between `lo` and `hi` answers in flight, both inclusive.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    pub lo: u32,
    pub hi: u32,
    pub samples: usize,
    pub tokens: f64,
    pub compression: f64,
    /// Each format's share of the band's resident tokens, summed over its
    /// samples before dividing.
    pub shares: BTreeMap<String, f64>,
}

/// The loaded samples banded [`BAND_WIDTH`] answers in flight at a time,
/// lowest first, each band's tokens and compression its samples' means.
pub fn bands(samples: &[Value], bf16_per_token: f64) -> Vec<Band> {
    let mut grouped: BTreeMap<u32, Vec<Reading>> = BTreeMap::new();
    for r in samples.iter().filter_map(reading) {
        let band = ((r.in_flight - 1.0) / BAND_WIDTH).floor() as u32;
        grouped.entry(band).or_default().push(r);
    }
    grouped
        .into_iter()
        .map(|(band, rs)| {
            let n = rs.len() as f64;
            let total: f64 = rs.iter().map(|r| r.tokens).sum();
            let mut shares: BTreeMap<String, f64> = BTreeMap::new();
            for r in &rs {
                for (f, t) in &r.formats {
                    *shares.entry(f.clone()).or_default() += t;
                }
            }
            if total > 0.0 {
                shares.values_mut().for_each(|s| *s /= total);
            }
            let width = BAND_WIDTH as u32;
            Band {
                lo: band * width + 1,
                hi: (band + 1) * width,
                samples: rs.len(),
                tokens: total / n,
                compression: rs
                    .iter()
                    .map(|r| compression(r, bf16_per_token))
                    .sum::<f64>()
                    / n,
                shares,
            }
        })
        .collect()
}

/// The BF16 bytes one token takes over the model's `kv_layers` K/V layers,
/// from a K/V census (`GET /v1/kv/census`), whose `tokens` count every
/// layer's.
pub fn bf16_per_token(census: &Value, kv_layers: u32) -> Option<f64> {
    let bf16 = census.get("bf16_bytes_estimate")?.as_f64()?;
    let tokens = census.get("tokens")?.as_f64()?;
    (tokens > 0.0).then(|| bf16 / tokens * kv_layers as f64)
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
                formats: BTreeMap::from([
                    ("Q4_0".to_string(), 3000.0),
                    ("other".to_string(), 1000.0)
                ]),
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
                ("KV arenas held (GiB)", 1.0, 1.5),
                ("Tokens resident", 1_572_864.0, 2_097_152.0),
                ("The same tokens at BF16 (GiB)", 3.0, 4.0),
                (
                    "Effective compression (BF16 / arenas held)",
                    (4.0 + 8.0 / 3.0) / 2.0,
                    4.0
                ),
                ("Expert zone (GiB)", 45.0, 50.0),
            ]
        );
        assert!(table(&samples[1..2], 2048.0).is_empty());
    }

    /// Samples fall in bands of four answers in flight; a band's shares are
    /// of its tokens summed, and its compression the mean of its samples'.
    #[test]
    fn bands_group_by_answers_in_flight() {
        let samples = vec![
            sample(1, 1024.0, json!({"Q8_0": 1_048_576.0}), 0.0),
            sample(
                4,
                1024.0,
                json!({"Q4_0": 1_048_576.0, "Q8_0": 1_048_576.0}),
                0.0,
            ),
            sample(0, 1024.0, json!({"Q4_0": 1.0}), 0.0),
            sample(13, 1024.0, json!({"Q4_0": 4_194_304.0}), 0.0),
        ];
        let got = bands(&samples, 1024.0);
        assert_eq!(
            got,
            vec![
                Band {
                    lo: 1,
                    hi: 4,
                    samples: 2,
                    tokens: 1_572_864.0,
                    compression: 1.5,
                    shares: BTreeMap::from([
                        ("Q4_0".to_string(), 1.0 / 3.0),
                        ("Q8_0".to_string(), 2.0 / 3.0)
                    ]),
                },
                Band {
                    lo: 13,
                    hi: 16,
                    samples: 1,
                    tokens: 4_194_304.0,
                    compression: 4.0,
                    shares: BTreeMap::from([("Q4_0".to_string(), 1.0)]),
                },
            ]
        );
    }

    /// The census counts a token once per K/V layer, so its per-token BF16
    /// size is multiplied back up by the layers.
    #[test]
    fn a_census_gives_its_bf16_bytes_per_token_over_every_layer() {
        assert_eq!(
            bf16_per_token(&json!({"tokens": 4, "bf16_bytes_estimate": 8192}), 12),
            Some(24_576.0)
        );
        assert_eq!(
            bf16_per_token(&json!({"tokens": 0, "bf16_bytes_estimate": 1}), 12),
            None
        );
    }
}
