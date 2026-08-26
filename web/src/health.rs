//! Per-upstream health, with exponential backoff and automatic recovery.
//!
//! The problem this solves: when a backend is down, every request pays a full
//! connect timeout before failing. Fifty visitors then hold fifty sockets open
//! against a machine that cannot answer, and the page takes five seconds to
//! render an error it already knew about.
//!
//! So a failure opens a window. Requests inside it fail immediately with a
//! `Retry-After` and never touch the network. The window doubles per
//! consecutive failure up to the configured ceiling (10s by default), and when
//! it expires exactly one request is let through as a probe — success resets
//! everything. Recovery needs no operator and no restart.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::config::Backoff;

/// How long a probe may be outstanding before another is allowed through.
///
/// A probe is released by [`Health::gate`] and retired by [`Health::on_success`]
/// or [`Health::on_failure`] — both of which run *after* the request future
/// resolves. If that future is instead dropped, neither runs. The likeliest
/// moment for exactly that is while the probe sits in a connect timeout against
/// the machine presumed down, and the error page auto-refreshes: the client goes
/// away, the future is cancelled, and without this the upstream stays flagged
/// as probing with an expired window, so every later `gate` reports `Blocked`
/// for the life of the process. Recovery would need the restart this module
/// exists to avoid.
///
/// Pre-empting a probe that is merely slow is not a failure of this bound. Past
/// this long the probe has either connected — in which case the upstream is
/// answering and a second request is correct — or it is gone.
const ABANDONED_PROBE_AFTER: Duration = Duration::from_secs(60);

#[derive(Debug)]
struct Upstream {
    failures: u32,
    /// When the backoff window ends. `None` means healthy.
    open_until: Option<Instant>,
    /// When the in-flight probe was released, so only ONE request tests a
    /// recovering upstream — otherwise every request queued behind the window is
    /// released at once and stampedes the machine that is still coming back up.
    ///
    /// An instant rather than a flag because a probe can be cancelled without
    /// ever reporting back, and "since when" is what distinguishes a probe in
    /// flight from one that will never return. See [`ABANDONED_PROBE_AFTER`].
    probe_since: Option<Instant>,
    last_error: Option<String>,
}

impl Upstream {
    fn new() -> Self {
        Self {
            failures: 0,
            open_until: None,
            probe_since: None,
            last_error: None,
        }
    }
}

#[derive(Clone)]
pub struct Health {
    inner: Arc<Mutex<HashMap<String, Upstream>>>,
    backoff: Backoff,
}

/// What the caller should do with this request.
#[derive(Debug, PartialEq, Eq)]
pub enum Gate {
    /// Send it.
    Go,
    /// Backoff window is open and a probe is already out; fail fast.
    Blocked {
        retry_after: Duration,
        last_error: Option<String>,
    },
}

impl Health {
    pub fn new(backoff: Backoff) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            backoff,
        }
    }

    /// Decide whether to attempt `key` now.
    pub fn gate(&self, key: &str) -> Gate {
        let mut map = self.inner.lock().unwrap();
        let up = map.entry(key.to_string()).or_insert_with(Upstream::new);
        let Some(until) = up.open_until else {
            return Gate::Go;
        };

        let now = Instant::now();
        if now >= until {
            // Window expired — release exactly one probe, and treat one that has
            // been out too long as gone rather than in flight, so a cancelled
            // probe cannot hold the upstream shut for good.
            let in_flight = up
                .probe_since
                .is_some_and(|since| now.duration_since(since) < ABANDONED_PROBE_AFTER);
            if !in_flight {
                up.probe_since = Some(now);
                return Gate::Go;
            }
            // A probe is already out; hold the rest for one more short beat
            // rather than letting them pile onto it.
            return Gate::Blocked {
                retry_after: self.backoff.delay(1),
                last_error: up.last_error.clone(),
            };
        }

        Gate::Blocked {
            retry_after: until.saturating_duration_since(now),
            last_error: up.last_error.clone(),
        }
    }

    pub fn on_success(&self, key: &str) {
        let mut map = self.inner.lock().unwrap();
        if let Some(up) = map.get_mut(key) {
            let was_down = up.open_until.is_some();
            up.failures = 0;
            up.open_until = None;
            up.last_error = None;
            up.probe_since = None;
            if was_down {
                tracing::info!(upstream = key, "upstream recovered");
            }
        }
    }

    pub fn on_failure(&self, key: &str, err: &str) -> Duration {
        let mut map = self.inner.lock().unwrap();
        let up = map.entry(key.to_string()).or_insert_with(Upstream::new);
        up.failures = up.failures.saturating_add(1);
        let delay = self.backoff.delay(up.failures);
        up.open_until = Some(Instant::now() + delay);
        up.last_error = Some(err.to_string());
        up.probe_since = None;
        if up.failures == 1 {
            tracing::warn!(upstream = key, error = err, "upstream down — backing off");
        }
        delay
    }

    /// Pretend the in-flight probe was released `by` ago.
    ///
    /// [`ABANDONED_PROBE_AFTER`] is a minute, which a unit test cannot wait out
    /// and should not have to. Rewinding the timestamp reaches the same state a
    /// cancelled probe leaves behind, which is the state under test.
    #[cfg(test)]
    fn backdate_probe(&self, key: &str, by: Duration) {
        let mut map = self.inner.lock().unwrap();
        if let Some(up) = map.get_mut(key) {
            up.probe_since = up.probe_since.and_then(|t| t.checked_sub(by));
        }
    }

    /// Snapshot for the status endpoint / error page.
    pub fn report(&self) -> Vec<(String, u32, Option<Duration>, Option<String>)> {
        let map = self.inner.lock().unwrap();
        let now = Instant::now();
        let mut out: Vec<_> = map
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.failures,
                    v.open_until.map(|u| u.saturating_duration_since(now)),
                    v.last_error.clone(),
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn health() -> Health {
        Health::new(Backoff {
            initial_ms: 100,
            max_ms: 10_000,
        })
    }

    #[test]
    fn unknown_upstream_is_allowed() {
        assert_eq!(health().gate("a"), Gate::Go);
    }

    #[test]
    fn failure_opens_a_window_that_blocks() {
        let h = health();
        h.on_failure("a", "refused");
        assert!(matches!(h.gate("a"), Gate::Blocked { .. }));
    }

    #[test]
    fn window_doubles_per_consecutive_failure() {
        let h = health();
        assert_eq!(h.on_failure("a", "x").as_millis(), 100);
        assert_eq!(h.on_failure("a", "x").as_millis(), 200);
        assert_eq!(h.on_failure("a", "x").as_millis(), 400);
    }

    #[test]
    fn window_caps_at_the_configured_ceiling() {
        let h = Health::new(Backoff {
            initial_ms: 1000,
            max_ms: 10_000,
        });
        for _ in 0..20 {
            h.on_failure("a", "x");
        }
        assert_eq!(h.on_failure("a", "x").as_millis(), 10_000);
    }

    #[test]
    fn success_resets_the_window() {
        let h = health();
        h.on_failure("a", "x");
        h.on_failure("a", "x");
        h.on_success("a");
        assert_eq!(h.gate("a"), Gate::Go);
        // …and the next failure starts from the base delay again.
        assert_eq!(h.on_failure("a", "x").as_millis(), 100);
    }

    #[test]
    fn expired_window_releases_exactly_one_probe() {
        let h = Health::new(Backoff {
            initial_ms: 1,
            max_ms: 10,
        });
        h.on_failure("a", "x");
        std::thread::sleep(Duration::from_millis(6));
        // First caller through the expired window probes…
        assert_eq!(h.gate("a"), Gate::Go);
        // …and the rest are held rather than stampeding the recovering machine.
        assert!(matches!(h.gate("a"), Gate::Blocked { .. }));
    }

    /// A probe that never reports back must not hold the upstream shut.
    ///
    /// `gate` releases the probe; only `on_success`/`on_failure` retire it, and
    /// both run after the request future resolves. Drop that future — the client
    /// disconnects, or the error page auto-refreshes during the connect timeout
    /// against the machine presumed down — and neither runs. Before the deadline
    /// this left the upstream permanently `Blocked` with an expired window: 503
    /// for the life of the process, recoverable only by the restart this module
    /// exists to make unnecessary.
    #[test]
    fn an_abandoned_probe_does_not_wedge_the_upstream_shut() {
        let h = Health::new(Backoff {
            initial_ms: 1,
            max_ms: 10,
        });
        h.on_failure("a", "x");
        std::thread::sleep(Duration::from_millis(6));

        // A probe goes out, and is then cancelled — no on_success, no on_failure.
        assert_eq!(h.gate("a"), Gate::Go);
        assert!(matches!(h.gate("a"), Gate::Blocked { .. }));

        // Long enough later, it is treated as gone and another is let through.
        h.backdate_probe("a", ABANDONED_PROBE_AFTER + Duration::from_secs(1));
        assert_eq!(
            h.gate("a"),
            Gate::Go,
            "a cancelled probe held the upstream shut for good"
        );

        // And that one is still the only one: recovery, not a stampede.
        assert!(matches!(h.gate("a"), Gate::Blocked { .. }));
    }

    #[test]
    fn upstreams_are_tracked_independently() {
        let h = health();
        h.on_failure("a", "x");
        assert_eq!(h.gate("b"), Gate::Go);
    }
}
