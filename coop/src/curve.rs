//! A member's measured step time by batch size: its samples bucketed by rows
//! per forward, each bucket the forward-weighted mean of what its windows
//! reported. Where the parametric [`ForwardCost`](crate::cost::ForwardCost)
//! explains a channel poorly, this is what was actually seen.

use crate::sample::{Channel, Sample};

/// The rows-per-forward buckets: `1`, `2..=3`, `4..=7`, … `256..`.
pub const BUCKETS: [f64; 9] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

/// One bucket's forward-weighted means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// The bucket's lower bound.
    pub from: f64,
    pub windows: usize,
    pub fwds: u64,
    pub rows: f64,
    pub tokens: f64,
    pub kv: f64,
    pub ms: f64,
    pub hit: f64,
}

impl Point {
    /// Tokens per second a forward of this bucket delivers.
    pub fn tokens_per_s(&self) -> f64 {
        if self.ms > 0.0 {
            self.tokens * 1000.0 / self.ms
        } else {
            0.0
        }
    }
}

/// `channel`'s samples kept by `keep`, bucketed by rows. Buckets nothing fell
/// in are left out.
pub fn curve(samples: &[Sample], channel: Channel, keep: impl Fn(&Sample) -> bool) -> Vec<Point> {
    let mut out = Vec::new();
    for (i, from) in BUCKETS.iter().enumerate() {
        let to = BUCKETS.get(i + 1).copied().unwrap_or(f64::INFINITY);
        let in_bucket: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.channel == channel && s.fwds > 0 && keep(s))
            .filter(|s| s.rows >= *from && s.rows < to)
            .collect();
        if in_bucket.is_empty() {
            continue;
        }
        let w: f64 = in_bucket.iter().map(|s| f64::from(s.fwds)).sum();
        let mean = |f: &dyn Fn(&Sample) -> f64| {
            in_bucket
                .iter()
                .map(|s| f64::from(s.fwds) * f(s))
                .sum::<f64>()
                / w
        };
        // Over the windows that reported one.
        let hit_w: f64 = in_bucket
            .iter()
            .filter(|s| s.hit.is_some())
            .map(|s| f64::from(s.fwds))
            .sum();
        let hit = in_bucket
            .iter()
            .filter_map(|s| s.hit.map(|h| f64::from(s.fwds) * h))
            .sum::<f64>()
            / hit_w;
        out.push(Point {
            from: *from,
            windows: in_bucket.len(),
            fwds: w as u64,
            rows: mean(&|s| s.rows),
            tokens: mean(&|s| s.tokens),
            kv: mean(&|s| s.kv),
            ms: mean(&|s| s.ms),
            hit,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(rows: f64, ms: f64, fwds: u32) -> Sample {
        Sample {
            channel: Channel::Decode,
            fwds,
            rows,
            tokens: rows,
            kv: 1000.0 * rows,
            ms,
            hit: Some(0.9),
            alone: true,
        }
    }

    /// Windows fall into their rows bucket and average by forwards.
    #[test]
    fn windows_are_bucketed_and_weighted_by_forwards() {
        let s = [
            decode(1.0, 20.0, 10),
            decode(1.0, 50.0, 30),
            decode(3.0, 30.0, 5),
            decode(20.0, 40.0, 2),
        ];
        let c = curve(&s, Channel::Decode, |_| true);
        assert_eq!(c.len(), 3);
        assert_eq!((c[0].from, c[0].fwds, c[0].ms), (1.0, 40, 42.5));
        assert_eq!((c[1].from, c[1].rows, c[1].ms), (2.0, 3.0, 30.0));
        assert_eq!((c[2].from, c[2].windows), (16.0, 1));
        assert_eq!(c[2].tokens_per_s(), 500.0);
    }

    /// The hit rate averages over the windows that reported one; a bucket
    /// with none reads `NaN`.
    #[test]
    fn the_hit_rate_skips_windows_without_one() {
        let mut quiet = decode(1.0, 20.0, 10);
        quiet.hit = None;
        let s = [quiet.clone(), decode(1.0, 20.0, 30)];
        assert_eq!(curve(&s, Channel::Decode, |_| true)[0].hit, 0.9);
        assert!(curve(&[quiet], Channel::Decode, |_| true)[0].hit.is_nan());
    }

    /// The filter and the channel both narrow it.
    #[test]
    fn the_filter_and_channel_narrow_it() {
        let s = [decode(1.0, 20.0, 1), decode(1.0, 90.0, 1)];
        let c = curve(&s, Channel::Decode, |s| s.ms < 50.0);
        assert_eq!(c[0].ms, 20.0);
        assert!(curve(&s, Channel::Prefill, |_| true).is_empty());
    }
}
