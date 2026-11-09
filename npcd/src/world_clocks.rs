//! Every world's clock, as its document's `time` block holds it, read without
//! a lock.
//!
//! What time it is for a character is asked per character per beat by the tick
//! driver, and by eleven routes. Each used to take the worlds registry's lock —
//! a blocking read from the driver's thread, a hop to the blocking pool from a
//! handler — so a reload held every beat. The raw `time` blocks are kept here
//! instead, published as one snapshot: [`WorldClocks::store`] replaces it
//! whenever the registry is read again, and a reader loads it. The block is
//! parsed on every read ([`Clock::of_time`]), as the registry's was, so a field
//! the author left out still means "now" on every call.

use std::collections::BTreeMap;

use arc_swap::ArcSwap;
use serde_json::{Map, Value};

use crate::clock::Clock;
use crate::npcs::Cast;
use crate::registry::Registry;

/// Each world's `time` block, `None` for a world whose document has none.
type Times = BTreeMap<String, Option<Map<String, Value>>>;

/// The worlds' clocks, as the registry last held them.
#[derive(Default)]
pub struct WorldClocks {
    times: ArcSwap<Times>,
}

impl WorldClocks {
    /// The clocks `worlds` holds.
    pub fn of(worlds: &Registry) -> Self {
        let clocks = Self::default();
        clocks.store(worlds);
        clocks
    }

    /// Take the clocks `worlds` holds now, in place of the ones before.
    pub fn store(&self, worlds: &Registry) {
        let times: Times = worlds
            .iter()
            .map(|r| {
                let time = r.body.get("time").and_then(Value::as_object).cloned();
                (r.id.clone(), time)
            })
            .collect();
        self.times.store(times.into());
    }

    /// What time it is in world `world` at `now_ms`, in world milliseconds:
    /// `None` for a world the registry does not hold.
    pub fn world_now(&self, world: &str, now_ms: i64) -> Option<u64> {
        let times = self.times.load();
        let time = times.get(world)?;
        Some(Clock::of_time(time.as_ref(), now_ms).now(now_ms).max(0) as u64)
    }

    /// What time it is for character `npc_id` at `now_ms`: its world's, from
    /// `npcs`. Zero for a character with no world, or whose world is not held —
    /// a character whose world cannot be read has no time of its own, and
    /// inventing one would put it in a day the world has never been in.
    pub fn world_ms(&self, npcs: &Cast, npc_id: u64, now_ms: i64) -> u64 {
        npcs.world_of(npc_id)
            .and_then(|world| self.world_now(world, now_ms))
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const T0: i64 = 1_700_000_000_000;

    fn times(pairs: &[(&str, Option<Value>)]) -> WorldClocks {
        let clocks = WorldClocks::default();
        let times: Times = pairs
            .iter()
            .map(|(id, t)| {
                (
                    id.to_string(),
                    t.as_ref().and_then(Value::as_object).cloned(),
                )
            })
            .collect();
        clocks.times.store(times.into());
        clocks
    }

    /// **Each world reads its own clock**: a paced one runs at its pace, one
    /// with no `time` block is started now, one with only a `world_ms` stands
    /// still at it — no `at_ms` reads as now, every call — and a world not held
    /// has no time.
    #[test]
    fn each_world_reads_its_own_clock_at_now() {
        let c = times(&[
            (
                "paced",
                Some(json!({"world_ms": 0, "at_ms": T0, "scale": 60})),
            ),
            ("bare", None),
            ("fixed", Some(json!({"world_ms": 500}))),
        ]);
        let now = T0 + 1000;
        assert_eq!(c.world_now("paced", now), Some(60_000));
        assert_eq!(c.world_now("bare", now), Some(now as u64));
        assert_eq!(c.world_now("fixed", now), Some(500));
        assert_eq!(c.world_now("fixed", now + 99_000), Some(500));
        assert_eq!(c.world_now("absent", now), None);
    }

    /// **A read needs no runtime and takes no lock**: on a multi-thread
    /// runtime's worker, where a blocking read of the registry panicked, it
    /// answers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_read_on_a_runtime_worker_answers() {
        let c = times(&[("fixed", Some(json!({"world_ms": 500})))]);
        let answered = tokio::spawn(async move { c.world_now("fixed", T0) })
            .await
            .unwrap();
        assert_eq!(answered, Some(500));
    }
}
