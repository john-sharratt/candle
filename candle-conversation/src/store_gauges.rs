//! Every open store's memory gauge, published for the memory census
//! (`docs/memory_census.md` §5): the persistence pass takes one once a
//! [`GAUGE_EVERY`] and files it here, and `/v1/memory` and the census log read
//! them without walking a map.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::persistence::gauge::PersistenceGauge;
use crate::projection::Conversation;
use crate::substrate::gauge::SubstrateGauge;

/// How often a store is gauged.
pub const GAUGE_EVERY: Duration = Duration::from_secs(60);

/// One store's gauge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoreGauge {
    /// The store's directory: which tenant's substrate this is.
    pub dir: String,
    pub gauged_unix_ms: u64,
    /// How long taking it held the persistence pass, in microseconds.
    pub took_us: u64,
    pub substrate: SubstrateGauge,
    pub persistence: PersistenceGauge,
}

struct Registry {
    gauges: BTreeMap<String, StoreGauge>,
    taken: BTreeMap<String, Instant>,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(Registry {
            gauges: BTreeMap::new(),
            taken: BTreeMap::new(),
        })
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Whether the store at `dir` is due a gauge, marking it taken if so.
fn due(dir: &str) -> bool {
    let mut r = registry().lock().unwrap_or_else(|e| e.into_inner());
    let due = r
        .taken
        .get(dir)
        .is_none_or(|at| at.elapsed() >= GAUGE_EVERY);
    if due {
        r.taken.insert(dir.to_string(), Instant::now());
    }
    due
}

/// File `gauge`, replacing the store's last.
pub fn publish(gauge: StoreGauge) {
    let mut r = registry().lock().unwrap_or_else(|e| e.into_inner());
    r.gauges.insert(gauge.dir.clone(), gauge);
}

/// Gauge `conversation`'s store if it is due, and file it.
pub fn gauge_if_due(conversation: &Conversation, dir: &Path) {
    let dir = dir.display().to_string();
    if !due(&dir) {
        return;
    }
    let t = Instant::now();
    let (substrate, persistence) = conversation.memory_gauge();
    publish(StoreGauge {
        dir,
        gauged_unix_ms: now_ms(),
        took_us: t.elapsed().as_micros() as u64,
        substrate,
        persistence,
    });
}

/// Every store's last gauge, by directory.
pub fn all() -> Vec<StoreGauge> {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .gauges
        .values()
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A store is gauged once a [`GAUGE_EVERY`]**: the second ask inside
    /// it is not due.
    #[test]
    fn a_store_is_due_once_a_minute() {
        let dir = "test://a-store-is-due-once-a-minute";
        assert!(due(dir));
        assert!(!due(dir));
    }

    #[test]
    fn a_published_gauge_replaces_the_last() {
        let g = |ms| StoreGauge {
            dir: "test://replaces".into(),
            gauged_unix_ms: ms,
            took_us: 0,
            substrate: SubstrateGauge::default(),
            persistence: PersistenceGauge::default(),
        };
        publish(g(1));
        publish(g(2));
        let mine: Vec<StoreGauge> = all()
            .into_iter()
            .filter(|s| s.dir == "test://replaces")
            .collect();
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].gauged_unix_ms, 2);
    }
}
