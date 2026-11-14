//! The `kv_gap` line (`docs/kv_quantize_in_the_gap.md` §3.7): one line for each
//! second in which a gap ran, summed over that second's gaps — blocks installed
//! and deferred, rows discarded and blocks found moved, gaps that quantized
//! nothing because one was outstanding, jobs launched and how many led as
//! retries, the classes whose lease came up short and the first reason given,
//! and the time the two calls took on the scheduler thread.

use std::time::{Duration, Instant};

use candle_nn::kv_cache::{GapCatchUp, GapQuantize};

/// How often the line is written while gaps run.
const WINDOW: Duration = Duration::from_secs(1);

/// The sums since the line was last written.
#[derive(Debug, Default)]
pub(super) struct KvGapLine {
    since: Option<Instant>,
    gaps: u64,
    installed: u64,
    deferred: u64,
    stale: u64,
    moved: u64,
    outstanding: u64,
    jobs: u64,
    retried: u64,
    short_slots: u64,
    refused: Option<String>,
    errored: bool,
    host_us: u64,
}

impl KvGapLine {
    /// Fold one catch-up in.
    pub(super) fn caught_up(&mut self, c: &GapCatchUp) {
        if !c.caught_up {
            return;
        }
        self.installed += c.installed as u64;
        self.deferred += c.deferred as u64;
        self.stale += c.stale as u64;
        self.moved += c.moved as u64;
        self.errored |= c.errored;
    }

    /// Fold one quantize in.
    pub(super) fn quantized(&mut self, q: &GapQuantize) {
        if q.outstanding {
            self.outstanding += 1;
            return;
        }
        if q.jobs == 0 {
            return;
        }
        self.gaps += 1;
        self.jobs += q.jobs as u64;
        self.retried += q.retried as u64;
        self.short_slots += q.short.iter().sum::<usize>() as u64;
        if self.refused.is_none() {
            self.refused = q
                .refusals
                .first()
                .map(|r| format!("class {} short {}: {}", r.class, r.short, r.why));
        }
    }

    /// Time the gap's two calls took on the scheduler thread.
    pub(super) fn spent(&mut self, d: Duration) {
        self.host_us += d.as_micros() as u64;
    }

    /// Write the line if a window has passed with anything in it, and start the
    /// next window.
    pub(super) fn flush_due(&mut self, now: Instant) {
        let since = *self.since.get_or_insert(now);
        if now.duration_since(since) < WINDOW {
            return;
        }
        let quiet = self.gaps == 0 && self.installed == 0 && self.outstanding == 0;
        if !quiet {
            tracing::info!(
                target: "candle_conversation::scheduler::kv_gap",
                gaps = self.gaps,
                installed = self.installed,
                deferred = self.deferred,
                stale = self.stale,
                moved = self.moved,
                outstanding = self.outstanding,
                jobs = self.jobs,
                retried = self.retried,
                short_slots = self.short_slots,
                refused = self.refused.as_deref().unwrap_or(""),
                errored = self.errored,
                host_us = self.host_us,
                "kv_gap"
            );
        }
        *self = Self {
            since: Some(now),
            ..Self::default()
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize(jobs: usize, retried: usize) -> GapQuantize {
        GapQuantize {
            jobs,
            retried,
            ..GapQuantize::default()
        }
    }

    #[test]
    fn a_catch_up_that_did_not_happen_adds_nothing() {
        let mut l = KvGapLine::default();
        l.caught_up(&GapCatchUp::default());
        assert_eq!((l.installed, l.deferred), (0, 0));
    }

    #[test]
    fn gaps_sum_over_the_window() {
        let mut l = KvGapLine::default();
        l.quantized(&quantize(9, 0));
        l.quantized(&quantize(3, 2));
        l.quantized(&GapQuantize {
            outstanding: true,
            ..GapQuantize::default()
        });
        l.quantized(&quantize(0, 0));
        l.caught_up(&GapCatchUp {
            caught_up: true,
            installed: 7,
            deferred: 2,
            stale: 1,
            moved: 1,
            errored: false,
        });
        assert_eq!((l.gaps, l.jobs, l.retried, l.outstanding), (2, 12, 2, 1));
        assert_eq!((l.installed, l.deferred, l.stale, l.moved), (7, 2, 1, 1));
    }

    #[test]
    fn the_window_starts_over_once_written() {
        let mut l = KvGapLine::default();
        let t0 = Instant::now();
        l.flush_due(t0);
        l.quantized(&quantize(5, 0));
        l.flush_due(t0 + Duration::from_millis(500));
        assert_eq!(l.gaps, 1, "inside the window: kept");
        l.flush_due(t0 + Duration::from_millis(1500));
        assert_eq!(l.gaps, 0, "written and cleared");
    }
}
