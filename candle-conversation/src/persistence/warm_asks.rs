//! The turns a persistence thread has been asked to bring from cold into warm
//! (`PersistenceTrigger::warm_turns`, `docs/kv_quantize_in_the_gap.md` §5).
//!
//! Shared by the thread and every trigger it hands out. An ask only queues the
//! keys; the thread takes the whole queue at its next pass, each key once, in
//! the order first asked, and warms what fits its share of the warm budget.
//! What does not fit is dropped rather than carried: an ask is best-effort,
//! and the asker sends its current list again when it wants it.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::projection::TurnKey;

/// The queue of turns to warm.
#[derive(Clone, Default)]
pub struct WarmAsks {
    queue: Arc<Mutex<Vec<TurnKey>>>,
}

impl WarmAsks {
    /// Queue `turns` for the next pass.
    pub fn ask(&self, turns: Vec<TurnKey>) {
        if turns.is_empty() {
            return;
        }
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(turns);
    }

    /// Take every queued turn, each once, in the order first asked.
    pub fn take(&self) -> Vec<TurnKey> {
        let queued = std::mem::take(&mut *self.queue.lock().unwrap_or_else(|e| e.into_inner()));
        let mut seen = HashSet::with_capacity(queued.len());
        queued.into_iter().filter(|k| seen.insert(*k)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{TimelineId, TurnIndex};

    fn key(t: u64, i: u32) -> TurnKey {
        TurnKey::new(
            TimelineId::from_raw(t).expect("a nonzero timeline"),
            TurnIndex(i),
        )
    }

    #[test]
    fn a_take_hands_every_ask_once_in_the_order_first_asked() {
        let asks = WarmAsks::default();
        asks.ask(vec![key(1, 0), key(1, 1)]);
        asks.ask(vec![key(1, 1), key(2, 0), key(1, 0)]);
        assert_eq!(asks.take(), vec![key(1, 0), key(1, 1), key(2, 0)]);
        assert!(asks.take().is_empty(), "a take empties the queue");
    }

    #[test]
    fn a_clone_shares_the_queue() {
        let asks = WarmAsks::default();
        asks.clone().ask(vec![key(3, 4)]);
        assert_eq!(asks.take(), vec![key(3, 4)]);
    }
}
