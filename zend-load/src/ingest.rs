//! An ingest watched from outside: a member reads a checkout it has not read
//! (its repo_scan directories and code_read files, one merged backlog), and
//! zend-load follows that backlog on `/v1/status` until it drains — its wall
//! time and how many units it read a minute — so the same ingest can be run
//! alone, as a build's baseline, and with decode clients beside it.
//!
//! Starting the ingest is the member's operator's step: the checkout goes into
//! its workspace, which the member reads at start. zend-load waits for the
//! backlog to appear.

use std::time::Duration;

use serde_json::Value;
use tokio::time::{sleep, Instant};

use crate::client::Target;

/// How often the backlog is read.
const TICK: Duration = Duration::from_secs(5);

/// The backlog a `/v1/status` answer reports, as `(processed, total)`: a
/// member's own, or every member's summed when the gateway aggregates. `None`
/// when nothing is pending.
pub fn backlog(status: &Value) -> Option<(u64, u64)> {
    let one = |v: &Value| -> Option<(u64, u64)> {
        let b = &v["tenants"]["zend"]["ingest_backlog"];
        Some((b["processed"].as_u64()?, b["total"].as_u64()?))
    };
    match status["members"].as_array() {
        Some(ms) => ms
            .iter()
            .filter_map(|m| one(&m["body"]))
            .reduce(|a, b| (a.0 + b.0, a.1 + b.1)),
        None => one(status),
    }
}

/// An ingest as it was seen.
#[derive(Debug, Clone, PartialEq)]
pub struct Watched {
    /// From the backlog appearing to its draining; `None` if it never
    /// appeared within the wait, or had not drained by the limit.
    pub wall: Option<Duration>,
    /// The most units the backlog held at once: what the ingest read.
    pub units: u64,
    /// When it appeared, counted from the start of the watch.
    pub appeared_after: Option<Duration>,
}

impl Watched {
    pub fn units_per_minute(&self) -> Option<f64> {
        let wall = self.wall?.as_secs_f64();
        (wall > 0.0).then(|| self.units as f64 * 60.0 / wall)
    }
}

/// Read `base`'s status as `target` addresses it.
async fn status(target: &Target, base: &str) -> Option<Value> {
    let res = target
        .http
        .get(format!("{base}/v1/status"))
        .header("host", &target.host)
        .send()
        .await
        .ok()?;
    res.json().await.ok()
}

/// Wait at most `appear` for a backlog, then follow it until it drains, for
/// at most `limit`. `on_appear` runs once, when it appears.
pub async fn watch(
    target: &Target,
    base: &str,
    appear: Duration,
    limit: Duration,
    mut on_appear: impl FnMut(),
) -> Watched {
    let started = Instant::now();
    let mut seen: Option<Instant> = None;
    let mut units = 0;
    loop {
        let now = Instant::now();
        let pending = status(target, base).await.and_then(|s| backlog(&s));
        match (seen, pending) {
            (None, Some((_, total))) => {
                seen = Some(now);
                units = total;
                on_appear();
            }
            (None, None) if now - started >= appear => {
                return Watched {
                    wall: None,
                    units: 0,
                    appeared_after: None,
                };
            }
            (Some(_), Some((_, total))) => units = units.max(total),
            (Some(at), None) => {
                return Watched {
                    wall: Some(now - at),
                    units,
                    appeared_after: Some(at - started),
                };
            }
            (None, None) => {}
        }
        if let Some(at) = seen {
            if now - at >= limit {
                return Watched {
                    wall: None,
                    units,
                    appeared_after: Some(at - started),
                };
            }
        }
        sleep(TICK).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    /// A member's backlog is read from its tenant line; an aggregate sums
    /// every member's; none pending is `None`.
    #[test]
    fn the_backlog_is_read_from_a_member_or_summed_over_members() {
        let member = json!({"tenants": {"zend": {"ingest_backlog":
            {"processed": 3, "total": 859, "last_item": "a.rs"}}}});
        assert_eq!(backlog(&member), Some((3, 859)));
        let idle = json!({"tenants": {"zend": {"ingest_backlog": null}}});
        assert_eq!(backlog(&idle), None);
        let both = json!({"members": [
            {"body": member},
            {"body": {"tenants": {"zend": {"ingest_backlog": {"processed": 1, "total": 10}}}}},
            {"body": idle},
        ]});
        assert_eq!(backlog(&both), Some((4, 869)));
    }

    /// Units a minute are the units over the wall time; an ingest that never
    /// finished has no rate.
    #[test]
    fn the_rate_is_units_over_wall_time() {
        let w = Watched {
            wall: Some(Duration::from_secs(120)),
            units: 600,
            appeared_after: Some(Duration::from_secs(5)),
        };
        assert_eq!(w.units_per_minute(), Some(300.0));
        let unfinished = Watched { wall: None, ..w };
        assert_eq!(unfinished.units_per_minute(), None);
    }
}
