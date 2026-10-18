//! The scheduler's `wave` line (`candle-conversation` `scheduler::WaveStats::
//! flush`), read back into [`Sample`]s: one per channel the window ran.
//!
//! ```text
//! wave 2.2s: prefill fwds=9 seqs avg=7.0 max=7 tok/fwd avg=170 tok total=1530
//!   kv/fwd avg=441726 kv total=3975534 fwd avg=66ms | decode fwds=20 seqs avg=1.0
//!   max=1 kv/fwd avg=49188 fwd avg=98ms | kv-vram … | experts hit=95.23% miss=1787 …
//! ```
//!
//! A segment is `|`-separated; within one, a bare word names the `avg=` or
//! `total=` after it (`seqs avg`, `tok/fwd avg`, `kv/fwd avg`, `fwd avg`).
//! The section channel rides decode's sweep and is not a forward of its own,
//! so it is not read.

use std::collections::HashMap;

use crate::sample::{Channel, Sample};

/// The samples one log line carries: none unless it is a `wave` line.
pub fn parse(line: &str) -> Vec<Sample> {
    let Some(at) = line.find("wave ") else {
        return Vec::new();
    };
    let Some(colon) = line[at..].find("s: ") else {
        return Vec::new();
    };
    let body = &line[at + colon + 3..];
    let segments: Vec<HashMap<String, String>> = body.split(" | ").map(fields).collect();
    let hit = segments.iter().find_map(|s| {
        s.get("experts.hit")
            .and_then(|v| v.trim_end_matches('%').parse::<f64>().ok())
            .map(|p| p / 100.0)
    });
    let mut out = Vec::new();
    for s in &segments {
        let channel = match s.get("").map(String::as_str) {
            Some("prefill") => Channel::Prefill,
            Some("decode") => Channel::Decode,
            _ => continue,
        };
        let num = |k: &str| {
            s.get(k)
                .and_then(|v| v.trim_end_matches("ms").parse::<f64>().ok())
        };
        let (Some(fwds), Some(rows), Some(kv), Some(ms)) = (
            num("fwds"),
            num("seqs.avg"),
            num("kv/fwd.avg"),
            num("fwd.avg"),
        ) else {
            continue;
        };
        let tokens = match channel {
            Channel::Prefill => num("tok/fwd.avg").unwrap_or(rows),
            Channel::Decode => rows,
        };
        out.push(Sample {
            channel,
            fwds: fwds as u32,
            rows,
            tokens,
            kv,
            ms,
            hit,
            alone: true,
        });
    }
    if out.len() > 1 {
        for s in &mut out {
            s.alone = false;
        }
    }
    out
}

/// One segment's fields: its first word under `""`, `k=v` as `k`, and a
/// `k=v` after a bare word `w` as `w.k`.
fn fields(segment: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut words = segment.split_whitespace();
    if let Some(first) = words.next() {
        out.insert(String::new(), first.to_string());
    }
    let mut label: Option<&str> = None;
    for w in words {
        match w.split_once('=') {
            Some((k, v)) => {
                let key = match label.take() {
                    Some(l) => format!("{l}.{k}"),
                    None => k.to_string(),
                };
                out.insert(key, v.to_string());
            }
            None => label = Some(w),
        }
    }
    // `experts hit=…`: the segment's first word labels the first pair.
    if out.get("").map(String::as_str) == Some("experts") {
        if let Some(v) = out.remove("hit") {
            out.insert("experts.hit".into(), v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIXED: &str = "2026-10-18T12:11:11.483553Z  INFO candle_conversation::scheduler: wave 2.2s: prefill fwds=9 seqs avg=7.0 max=7 tok/fwd avg=170 tok total=1530 kv/fwd avg=441726 kv total=3975534 fwd avg=66ms | decode fwds=20 seqs avg=1.0 max=1 kv/fwd avg=49188 fwd avg=98ms | kv-vram budget=8256MiB used=304MiB | backlog=56161tok | experts hit=95.23% miss=1787 cold=0 promoted=554 | graphs waves=78 segs=2 fold=0/0ms reshape=0/0ms inst=0/1ms rec=18ms nodes/seg=186 held=99x/18852n/135MiB phase=lock banned=10 proven=14 reopened=3 unkept=3";

    /// Both channels of a mixed window, each with the window's hit rate.
    #[test]
    fn a_mixed_window_is_two_samples() {
        assert_eq!(
            parse(MIXED),
            vec![
                Sample {
                    channel: Channel::Prefill,
                    fwds: 9,
                    rows: 7.0,
                    tokens: 170.0,
                    kv: 441726.0,
                    ms: 66.0,
                    hit: Some(0.9523),
                    alone: false,
                },
                Sample {
                    channel: Channel::Decode,
                    fwds: 20,
                    rows: 1.0,
                    tokens: 1.0,
                    kv: 49188.0,
                    ms: 98.0,
                    hit: Some(0.9523),
                    alone: false,
                },
            ]
        );
    }

    /// A decode-only window with no expert segment has no hit rate.
    #[test]
    fn a_window_without_experts_has_no_hit_rate() {
        let line = "wave 2.0s: decode fwds=54 seqs avg=24.5 max=25 kv/fwd avg=57547 fwd avg=35ms | kv-vram budget=14896MiB used=320MiB";
        assert_eq!(
            parse(line),
            vec![Sample {
                channel: Channel::Decode,
                fwds: 54,
                rows: 24.5,
                tokens: 24.5,
                kv: 57547.0,
                ms: 35.0,
                hit: None,
                alone: true,
            }]
        );
    }

    /// Anything but a wave line, and a window with no forwards, carry none.
    #[test]
    fn other_lines_carry_nothing() {
        assert!(parse("INFO zend::session: tool catalog installed n_tools=109").is_empty());
        assert!(parse("wave 2.0s: (no forwards) | kv-vram budget=1MiB used=1MiB").is_empty());
    }
}
