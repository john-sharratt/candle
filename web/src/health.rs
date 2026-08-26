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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::config::Backoff;

#[derive(Debug)]
struct Upstream {
    failures: u32,
    /// When the backoff window ends. `None` means healthy.
    open_until: Option<Instant>,
    /// Set while a probe is in flight so only ONE request tests a recovering
    /// upstream — otherwise every request queued behind the window is released
    /// at once and stampedes the machine that is still coming back up.
    probing: AtomicBool,
    last_error: Option<String>,
}

impl Upstream {
    fn new() -> Self {
        Self {
            failures: 0,
            open_until: None,
            probing: AtomicBool::new(false),
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
            // Window expired — release exactly one probe.
            if !up.probing.swap(true, Ordering::SeqCst) {
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
            up.probing.store(false, Ordering::SeqCst);
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
        up.probing.store(false, Ordering::SeqCst);
        if up.failures == 1 {
            tracing::warn!(upstream = key, error = err, "upstream down — backing off");
        }
        delay
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

    #[test]
    fn upstreams_are_tracked_independently() {
        let h = health();
        h.on_failure("a", "x");
        assert_eq!(h.gate("b"), Gate::Go);
    }
}
